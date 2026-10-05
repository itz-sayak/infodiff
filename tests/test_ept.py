# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch

from infodiff.models.neural_pt import EPTConfig, EPTTPP


def test_exact_compensator_matches_quadrature():
    torch.manual_seed(0)
    cfg = EPTConfig(n_marks=3, hidden=16, n_rates=3, n_channels=2, phases=3, tau_min=0.05, tau_max=20.0)
    m = EPTTPP(cfg).double()
    with torch.no_grad():  # make every head non-trivial
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    B, L = 3, 6
    dts = torch.rand(B, L, dtype=torch.float64) * 3.0
    dts[:, 0] = 0
    marks = torch.randint(0, 3, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    with torch.no_grad():
        log_lam, comp, states, mus = m.forward(dts, marks, mask)
        H = m._last_h
        M = cfg.n_marks
        for n in range(1, L):
            grid = torch.linspace(0, 1, 20001, dtype=torch.float64)[None, :] * dts[:, n:n + 1]
            lam, _ = m._curve(states[:, n - 1], mus[:, n - 1], grid, H[:, n - 1])
            num = torch.trapezoid(lam.sum(-1), grid, dim=1)
            assert torch.allclose(num, comp[:, n - 1], rtol=1e-5, atol=1e-8), (num, comp[:, n - 1])
            # log-intensity of the observed mark at the event matches the curve end-point
            ref = torch.log(lam[torch.arange(B), -1, marks[:, n]])
            assert torch.allclose(ref, log_lam[:, n - 1], atol=1e-6)


def _eptx(renewal=True, rn_shift=True, **kw):
    torch.manual_seed(1)
    cfg = EPTConfig(n_marks=3, hidden=16, n_rates=3, n_channels=2, phases=2, tau_min=0.05, tau_max=20.0,
                    renewal=renewal, rn_shift=rn_shift and renewal, rn_scales=6, rn_orders=(1, 4), rn_lo=1e-3, rn_hi=10.0,
                    input_v2=True, gap_eps=1e-6, gap_mu=-1.0, gap_sd=2.0, tie_thr=1e-3, **kw)
    m = EPTTPP(cfg).double()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return cfg, m


def _toy(L=6, B=3, M=3):
    torch.manual_seed(2)
    dts = torch.rand(B, L, dtype=torch.float64) * 3.0
    dts[:, 1] = 2e-3  # a near-tie
    dts[:, 0] = 0
    return dts, torch.randint(0, M, (B, L)), torch.ones(B, L, dtype=torch.bool)


def test_eptx_compensator_and_intensity_match_quadrature():
    cfg, m = _eptx()
    dts, marks, mask = _toy()
    B, L = dts.shape
    with torch.no_grad():
        log_lam, comp, states, mus = m.forward(dts, marks, mask)
        H = m._last_h
        for n in range(1, L):
            u = torch.linspace(0, 1, 200001, dtype=torch.float64)
            grid = (u ** 3)[None, :] * dts[:, n:n + 1]  # dense near 0 for the fast atoms
            lam, Lam = m._curve(states[:, n - 1], mus[:, n - 1], grid, H[:, n - 1])
            num = torch.trapezoid(lam.sum(-1), grid, dim=1)
            assert torch.allclose(num, comp[:, n - 1], rtol=2e-4), (num, comp[:, n - 1])
            assert torch.allclose(Lam[:, -1], comp[:, n - 1], rtol=1e-10)
            ref = torch.log(lam[torch.arange(B), -1, marks[:, n]])
            assert torch.allclose(ref, log_lam[:, n - 1], atol=1e-8)


def test_eptx_renewal_channel_is_a_defective_density():
    """Renewal channel alone: sum_k int p(t,k) dt = 1 - w_defect."""
    cfg, m = _eptx()
    h = torch.randn(4, cfg.hidden, dtype=torch.float64)
    with torch.no_grad():
        logw, logq, _ = m._rn_heads(h)
        u = torch.linspace(-12, 4, 400001, dtype=torch.float64)
        grid = torch.pow(10.0, u)[None, :].expand(4, -1)
        lr, cr = m._rn_log_lam(logw[:, None, :], logq[:, None], grid)
        dens = torch.exp(lr).sum(-1) * torch.exp(-cr)  # lambda * S
        mass = torch.trapezoid(dens, grid, dim=1)
        assert torch.allclose(mass, 1 - torch.exp(logw[:, -1]), atol=2e-4), mass


def test_eptx_nests_ept_when_defect_weight_is_one():
    cfg, m = _eptx()
    _, m0 = _eptx(renewal=False)
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("rn_")}
    m0.load_state_dict(sd)
    with torch.no_grad():
        m.rn_w.weight.zero_()
        m.rn_w.bias.fill_(-200.0)
        m.rn_w.bias[-1] = 0.0
        dts, marks, mask = _toy()
        a = m.loglik(dts, marks, mask)[0]
        b = m0.loglik(dts, marks, mask)[0]
    assert torch.allclose(a, b, atol=1e-8), (a, b)


def test_eptx_log_survival_is_stable_for_huge_lags():
    cfg, m = _eptx()
    with torch.no_grad():
        logf, logQ = m._rn_logf_logQ(torch.tensor([1e-12, 1.0, 1e6], dtype=torch.float64))
    assert torch.isfinite(logf).all() and torch.isfinite(logQ).all()
    assert (logQ <= 1e-12).all()


def test_eptx_sharp_atoms_survival_matches_quadrature():
    """Large Erlang orders use the incomplete-gamma path; check S = 1 - int f on a grid."""
    cfg = EPTConfig(n_marks=2, hidden=8, n_rates=2, n_channels=1, renewal=True, rn_scales=3,
                    rn_orders=(4, 64, 256), rn_lo=0.1, rn_hi=10.0)
    m = EPTTPP(cfg).double()
    t = torch.linspace(1e-9, 60.0, 600001, dtype=torch.float64)
    with torch.no_grad():
        logf, logQ = m._rn_logf_logQ(t)
    F = torch.cumulative_trapezoid(torch.exp(logf), t, dim=0)
    assert torch.allclose(torch.exp(logQ[1:]), 1 - F, atol=2e-4)


def test_time_plus_mark_equals_total_loglik():
    cfg, m = _eptx()
    dts, marks, mask = _toy()
    with torch.no_grad():
        log_lam, comp, _, _ = m.forward(dts, marks, mask)
        tot = (log_lam - comp).sum()
        time_ll = (m._last_log_tot - comp).sum()
        mark_ll = (log_lam - m._last_log_tot).sum()
    assert torch.allclose(time_ll + mark_ll, tot)
    assert (log_lam <= m._last_log_tot + 1e-9).all()  # mark log-probabilities are <= 0


def test_deep_encoder_one_layer_matches_default():
    torch.manual_seed(3)
    a = EPTTPP(EPTConfig(n_marks=3, hidden=16, n_rates=3, n_channels=2)).double()
    torch.manual_seed(3)
    b = EPTTPP(EPTConfig(n_marks=3, hidden=16, n_rates=3, n_channels=2, n_layers=1)).double()
    dts, marks, mask = _toy()
    with torch.no_grad():
        assert torch.allclose(a.loglik(dts, marks, mask)[0], b.loglik(dts, marks, mask)[0])
        deep = EPTTPP(EPTConfig(n_marks=3, hidden=16, n_rates=3, n_channels=2, n_layers=3,
                                layer_norm=True, dropout=0.1)).double().eval()
        ll, n = deep.loglik(dts, marks, mask)
    assert torch.isfinite(ll)


def test_eptx_sharp_atoms_finite_gradients_with_zero_gaps():
    torch.manual_seed(0)
    cfg = EPTConfig(n_marks=3, hidden=16, n_rates=3, n_channels=2, renewal=True, rn_shift=True, rn_scales=6,
                    rn_orders=(1, 4, 16, 64, 256, 1024), rn_lo=0.01, rn_hi=4.0, input_v2=True)
    m = EPTTPP(cfg)
    dts = torch.rand(4, 6) * 0.8
    dts[:, 0] = 0
    dts[0, 3] = 0.0  # a zero gap
    marks = torch.randint(0, 3, (4, 6))
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[1, 4:] = False
    ll, _ = m.loglik(dts, marks, mask)
    ll.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def _tf_model():
    torch.manual_seed(4)
    cfg = EPTConfig(n_marks=3, hidden=16, n_rates=3, n_channels=2, phases=2, tau_min=0.05, tau_max=20.0,
                    renewal=True, rn_scales=6, rn_orders=(1, 4), rn_lo=1e-3, rn_hi=10.0, input_v2=True,
                    gap_eps=1e-6, gap_mu=-1.0, gap_sd=2.0, tie_thr=1e-3, encoder="transformer",
                    tf_layers=2, tf_heads=2)
    return cfg, EPTTPP(cfg).double().eval()


def test_transformer_encoder_is_causal():
    cfg, m = _tf_model()
    dts, marks, mask = _toy(L=7)
    with torch.no_grad():
        h1 = m._encode_all(dts, marks, mask)
        dts2, marks2 = dts.clone(), marks.clone()
        dts2[:, 5] += 1.0
        marks2[:, 5] = (marks2[:, 5] + 1) % 3
        h2 = m._encode_all(dts2, marks2, mask)
    assert torch.allclose(h1[:, :5], h2[:, :5], atol=1e-10)
    assert not torch.allclose(h1[:, 5:], h2[:, 5:])


def test_transformer_encoder_exact_compensator():
    cfg, m = _tf_model()
    dts, marks, mask = _toy()
    B, L = dts.shape
    with torch.no_grad():
        log_lam, comp, states, mus = m.forward(dts, marks, mask)
        H = m._last_h
        for n in range(1, L):
            u = torch.linspace(0, 1, 200001, dtype=torch.float64)
            grid = (u ** 3)[None, :] * dts[:, n:n + 1]
            lam, Lam = m._curve(states[:, n - 1], mus[:, n - 1], grid, H[:, n - 1])
            assert torch.allclose(torch.trapezoid(lam.sum(-1), grid, dim=1), comp[:, n - 1], rtol=2e-4)
            assert torch.allclose(torch.log(lam[torch.arange(B), -1, marks[:, n]]), log_lam[:, n - 1], atol=1e-8)


def _eptx_quantile(**kw):
    from infodiff.experiments.track_b import quantile_atoms
    rng = np.random.default_rng(0)
    gaps = np.concatenate([rng.uniform(0.010, 0.014, 300), rng.uniform(0.65, 0.80, 600)])
    qm, qr, qd = quantile_atoms(gaps, 12, 1.0, 512)
    return _eptx(rn_q_means=qm, rn_q_orders=qr, rn_q_delta=qd, **kw)


def test_quantile_atoms_are_sharp_where_data_are_dense():
    from infodiff.experiments.track_b import quantile_atoms
    gaps = np.concatenate([np.full(10, 0.1), np.linspace(1.0, 1.1, 200), np.linspace(5, 50, 20)])
    m, R, d = quantile_atoms(gaps, 20, 1.0, 4096)
    assert len(m) == len(R) == len(d) and np.all(np.diff(m) > 0)
    R = np.asarray(R)
    inside = (np.asarray(m) > 1.0) & (np.asarray(m) < 1.1)
    assert R[inside].min() > R[~inside].max()  # densest region gets the sharpest atoms


def test_quantile_atoms_exact_compensator():
    cfg, m = _eptx_quantile()
    dts, marks, mask = _toy()
    dts[:, 2] = 0.7  # inside the dense band
    B, L = dts.shape
    with torch.no_grad():
        log_lam, comp, states, mus = m.forward(dts, marks, mask)
        H = m._last_h
        for n in range(1, L):
            u = torch.linspace(0, 1, 400001, dtype=torch.float64)
            grid = (u ** 3)[None, :] * dts[:, n:n + 1]
            lam, Lam = m._curve(states[:, n - 1], mus[:, n - 1], grid, H[:, n - 1])
            num = torch.trapezoid(lam.sum(-1), grid, dim=1)
            assert torch.allclose(num, comp[:, n - 1], rtol=5e-4), (n, num, comp[:, n - 1])
            assert torch.allclose(torch.log(lam[torch.arange(B), -1, marks[:, n]]), log_lam[:, n - 1], atol=1e-8)


def test_residual_mark_head_starts_at_intensity_marks():
    _, a = _eptx()
    _, b = _eptx(mark_head="residual")
    b.load_state_dict(a.state_dict(), strict=False)  # mk_head keeps its zero output layer
    with torch.no_grad():
        b.mk_head[2].weight.zero_()
        b.mk_head[2].bias.zero_()
        dts, marks, mask = _toy()
        la, ca, _, _ = a.forward(dts, marks, mask)
        lb, cb, _, _ = b.forward(dts, marks, mask)
    assert torch.allclose(ca, cb) and torch.allclose(la, lb, atol=1e-9)


def test_residual_mark_head_is_exact_and_normalised():
    cfg, m = _eptx(mark_head="residual")
    dts, marks, mask = _toy()
    B, L = dts.shape
    with torch.no_grad():
        log_lam, comp, states, mus = m.forward(dts, marks, mask)
        H = m._last_h
        assert (log_lam <= m._last_log_tot + 1e-9).all()
        for n in range(1, L):
            u = torch.linspace(0, 1, 200001, dtype=torch.float64)
            grid = (u ** 3)[None, :] * dts[:, n:n + 1]
            lam, Lam = m._curve(states[:, n - 1], mus[:, n - 1], grid, H[:, n - 1])
            num = torch.trapezoid(lam.sum(-1), grid, dim=1)
            assert torch.allclose(num, comp[:, n - 1], rtol=2e-4)
            assert torch.allclose(torch.log(lam[torch.arange(B), -1, marks[:, n]]), log_lam[:, n - 1], atol=1e-8)


def _ragged(L=9, B=4, M=3):
    torch.manual_seed(5)
    dts = torch.rand(B, L, dtype=torch.float64) * 3.0
    dts[:, 2] = 1e-3
    dts[:, 0] = 0
    marks = torch.randint(0, M, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    for b, n in enumerate([L, L - 3, 4, 2]):
        mask[b, n:] = False
        marks[b, n:] = M
        dts[b, n:] = 0
    return dts, marks, mask


import pytest


@pytest.mark.parametrize("kw", [
    dict(renewal=False),
    dict(),
    dict(mark_head="residual"),
    dict(n_layers=3, layer_norm=True),
    dict(encoder="transformer"),
    dict(mark_mixing=False, gompertz=False),
    dict(quantile=True),
    dict(quantile=True, mark_head="residual"),
])
def test_parallel_forward_equals_event_loop(kw):
    q = kw.pop("quantile", False)
    cfg, m = _eptx_quantile(**kw) if q else _eptx(**kw)
    m.eval()
    dts, marks, mask = _ragged()
    sel = mask[:, 1:]
    m.cfg.parallel = False
    a = m.forward(dts, marks, mask)
    ta = m._last_log_tot
    lla, _ = m.loglik(dts, marks, mask)
    ga = torch.autograd.grad(lla, [p for p in m.parameters()], allow_unused=True)
    m.cfg.parallel = True
    b = m.forward(dts, marks, mask)
    tb = m._last_log_tot
    llb, _ = m.loglik(dts, marks, mask)
    gb = torch.autograd.grad(llb, [p for p in m.parameters()], allow_unused=True)
    assert torch.allclose(a[0][sel], b[0][sel], atol=1e-9) and torch.allclose(a[1][sel], b[1][sel], atol=1e-9)
    assert torch.allclose(ta[sel], tb[sel], atol=1e-9)
    valid = mask
    assert torch.allclose(a[2][valid], b[2][valid], atol=1e-9) and torch.allclose(a[3][valid], b[3][valid], atol=1e-9)
    assert torch.allclose(lla, llb, atol=1e-8)
    for x, y in zip(ga, gb):
        if x is not None or y is not None:
            assert torch.allclose(x, y, atol=1e-7, rtol=1e-6)


def test_init_from_hawkes_reproduces_linear_hawkes_loglik():
    """Warm start at a linear phase-type Hawkes process: the model's exact log-likelihood equals a
    brute-force Hawkes log-likelihood (intensity sums over past events, Erlang CDF compensator)."""
    from scipy.special import gammainc, gammaln
    M, K, R = 3, 4, 2
    betas = np.array([0.3, 1.0, 3.0, 10.0])
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 0.1, (M, M, K, R))
    mu = rng.uniform(0.05, 0.3, M)
    cfg = EPTConfig(n_marks=M, hidden=8, n_rates=2, n_channels=1, hb_rates=K, hb_phases=R,
                    hb_betas=tuple(betas), renewal=True, rn_scales=4, rn_orders=(1, 4), rn_lo=0.01, rn_hi=10.0)
    m = EPTTPP(cfg).double()
    m.init_from_hawkes(A, mu, eps_logit=-40.0)
    dts, marks, mask = _toy(L=8, B=2, M=M)
    with torch.no_grad():
        ll, _ = m.loglik(dts, marks, mask)
    ref = 0.0
    for b in range(dts.shape[0]):
        t = np.cumsum(dts[b].numpy())
        y = marks[b].numpy()
        for n in range(1, len(t)):
            lag, lag0 = t[n] - t[:n], t[n - 1] - t[:n]
            g = np.exp(np.log(betas)[None, :, None] + np.arange(R) * np.log(betas[None, :, None] * lag[:, None, None])
                       - betas[None, :, None] * lag[:, None, None] - gammaln(np.arange(R) + 1))
            lam = mu[y[n]] + (A[y[n], y[:n]] * g).sum()
            G = gammainc(np.arange(R)[None, None, :] + 1, betas[None, :, None] * lag[:, None, None]) \
                - gammainc(np.arange(R)[None, None, :] + 1, betas[None, :, None] * lag0[:, None, None])
            comp = mu.sum() * (t[n] - t[n - 1]) + (A[:, y[:n]].transpose(1, 0, 2, 3) * G[:, None]).sum()
            ref += np.log(lam) - comp
    assert abs(float(ll) - ref) < 1e-8 * max(1.0, abs(ref)), (float(ll), ref)


@pytest.mark.parametrize("kw", [dict(), dict(mark_head="residual"), dict(renewal=False)])
def test_channel_parts_sum_to_compensator(kw):
    cfg, m = _eptx(**kw)
    dts, marks, mask = _ragged()
    with torch.no_grad():
        _, comp, _, _ = m.forward(dts, marks, mask)
    tot = sum(m._last_comp_parts.values())
    assert torch.allclose(tot, comp, atol=1e-12) and all((v >= -1e-12).all() for v in m._last_comp_parts.values())
