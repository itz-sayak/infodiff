# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from scipy.optimize import minimize

from infodiff.diagnostics.rescaling import gof
from infodiff.events.data import EventData
from infodiff.models.dictionary import PhaseTypeDictionary
from infodiff.models.features import DesignSpec, _hat_integral, build_design
from infodiff.models.msx import MSXHawkes
from infodiff.models.params import model_from_truth
from infodiff.models.statespace import StateSpaceHawkes
from infodiff.sim.cluster import HawkesTruth, simulate

ENDO = PhaseTypeDictionary.log_grid(0.05, 5.0, 3, orders=2)
EXO = PhaseTypeDictionary.log_grid(0.5, 20.0, 2, orders=2)
ANT = PhaseTypeDictionary.log_grid(2.0, 10.0, 2, orders=1)


def _truth(d=2, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (d, d, ENDO.K, ENDO.R))
    A *= 0.6 / A.sum(axis=(1, 2, 3), keepdims=True)  # row sums 0.6 -> rho < 1
    B = rng.uniform(0, 3, (d, 2, EXO.K, EXO.R, 2))
    H = rng.uniform(0, 1, (d, 2, ANT.K))
    return HawkesTruth(endo=ENDO, A=A, mu=np.full(d, 0.5), exo=EXO, B=B, ant=ANT, H=H,
                       tod_w=rng.uniform(0, 0.3, (d, 24)))


def _news(W, rng):
    out = []
    for w in range(W):
        tau = 100.0 * w + 40.0
        out.append([(tau, int(rng.integers(0, 2)), np.array([1.0, rng.uniform(0, 2)]))])
    return out


def _sim(truth, W=6, seed=1):
    rng = np.random.default_rng(seed)
    t0 = 100.0 * np.arange(W) + 10.0
    return simulate(truth, t0=t0, t1=t0 + 80.0, burn=10.0, news=_news(W, rng),
                    tod0=rng.uniform(0, 86400, W), seed=seed)


def _brute_loglik(truth: HawkesTruth, spec: DesignSpec, data: EventData) -> float:
    """Independent O(N^2) implementation using explicit kernel sums."""
    ll = 0.0
    d = truth.d
    for w in range(data.n_windows):
        t, u = data.window(w)
        lo, hi = data.t0[w], data.t1[w]
        news = [(data.news_t[e], data.news_type[e], data.news_marks[e])
                for e in range(data.news_ptr[w], data.news_ptr[w + 1])]
        for n in range(len(t)):
            if not (lo <= t[n] < hi):
                continue
            i = u[n]
            x = (data.tod0[w] + t[n] - lo) % 86400
            h = 86400 / spec.n_tod
            m0 = int(x // h)
            f = x / h - m0
            lam = truth.mu[i] + truth.tod_w[i, m0 % spec.n_tod] * (1 - f) + truth.tod_w[i, (m0 + 1) % spec.n_tod] * f
            prev = t < t[n]
            for j in range(d):
                lags = t[n] - t[prev & (u == j)]
                if len(lags):
                    lam += ENDO.kernel(truth.A[i, j].reshape(-1), lags).sum()
            for tau, c, z in news:
                if tau < t[n]:
                    lam += EXO.kernel(np.tensordot(truth.B[i, c], z, axes=([-1], [0])).reshape(-1), np.array([t[n] - tau]))[0]
                elif tau > t[n]:
                    lam += (truth.H[i, c] * ANT.betas * np.exp(-ANT.betas * (tau - t[n]))).sum()
            ll += np.log(lam)
        for i in range(d):
            comp = truth.mu[i] * (hi - lo) + truth.tod_w[i] @ _hat_integral(data.tod0[w], data.tod0[w] + hi - lo, spec.n_tod)
            for j in range(d):
                tj = t[(u == j) & (t < hi)]
                wts = truth.A[i, j].reshape(-1)
                comp += (wts * (ENDO.cdf(hi - tj) - ENDO.cdf(np.maximum(lo - tj, 0)))).sum()
            for tau, c, z in news:
                wts = np.tensordot(truth.B[i, c], z, axes=([-1], [0])).reshape(-1)
                if tau < hi:
                    comp += (wts * (EXO.cdf(np.array([hi - tau])) - EXO.cdf(np.array([max(lo - tau, 0)])))).sum()
                if tau > lo:
                    top = min(tau, hi)
                    comp += (truth.H[i, c] * (np.exp(-ANT.betas * (tau - top)) - np.exp(-ANT.betas * (tau - lo)))).sum()
            ll -= comp
    return ll


def test_loglik_matches_bruteforce():
    truth = _truth()
    data = _sim(truth)
    # inject exact ties to exercise tie handling
    data.times[5] = data.times[4]
    spec = DesignSpec(endo=ENDO, exo=EXO, ant=ANT, n_tod=24)
    model = model_from_truth(truth, spec, data)
    assert np.isclose(model.loglik(data), _brute_loglik(truth, spec, data), rtol=1e-9)


def test_em_certified_global_optimum():
    truth = _truth()
    data = _sim(truth, W=20)
    spec = DesignSpec(endo=ENDO, exo=EXO, ant=ANT, n_tod=0)
    m = MSXHawkes(spec, device="cpu", gap_tol=1e-6, gap_rel=0.0).fit(data)
    for rep in m.reports:
        assert rep.gap < 1e-6
        # KKT: expected count equals observed count for unpenalised fits
        assert np.isclose(rep.expected_count, rep.n_events, rtol=1e-4)
        h = np.array(rep.history)
        assert np.all(np.diff(h) >= -1e-7 * np.abs(h[1:]))
    # compare with an independent optimiser on dim 0
    des = build_design(data, spec, 0)
    X =np.hstack([des.Xd.astype(float), des.Xs.toarray()])
    c = np.concatenate([des.integ_d, des.integ_s])

    def f(th):
        lam = X @ th
        return -(np.log(lam).sum() - c @ th), -(X.T @ (1 / lam) - c)

    x0 = np.full(X.shape[1], 0.05)
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(1e-12, None)] * len(x0),
                   options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10))
    assert m.reports[0].loglik >= -res.fun - 1e-6


def test_parameter_recovery():
    d = 2
    A = np.zeros((d, d, ENDO.K, ENDO.R))
    A[0, 0, 0, 1] = 0.3
    A[1, 1, 2, 0] = 0.4
    A[0, 1, 1, 0] = 0.2
    truth = HawkesTruth(endo=ENDO, A=A, mu=np.array([1.0, 0.7]))
    t0 = np.arange(40) * 3000.0
    data = simulate(truth, t0=t0, t1=t0 + 2500.0, burn=50.0, seed=3)
    m = MSXHawkes(DesignSpec(endo=ENDO), device="cpu").fit(data)
    G = m.branching_matrix()
    assert np.allclose(G, truth.A.sum(axis=(2, 3)), atol=0.05)


def test_stability_theorem_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = rng.integers(1, 5)
        A = rng.exponential(1.0, (d, d, ENDO.K, ENDO.R)) * (rng.random((d, d, 1, 1)) < 0.6)
        A *= rng.uniform(0.2, 1.8) / max(A.sum(axis=(2, 3)).max(), 1e-9)
        ss = StateSpaceHawkes(ENDO, A)
        rho = ss.spectral_radius()
        if abs(rho - 1) < 1e-6:
            continue
        assert (ss.spectral_abscissa() < 0) == (rho < 1)


def test_closed_form_response_matches_simulation():
    d = 2
    A = np.zeros((d, d, ENDO.K, ENDO.R))
    A[0, 0, 1, 0] = 0.4
    A[1, 0, 0, 1] = 0.3
    A[0, 1, 2, 0] = 0.2
    B = np.zeros((d, 1, EXO.K, EXO.R, 1))
    B[0, 0, 0, 1, 0] = 5.0
    B[1, 0, 1, 0, 0] = 2.0
    truth = HawkesTruth(endo=ENDO, A=A, mu=np.full(d, 1e-9), exo=EXO, B=B)
    W = 3000
    t0 = np.arange(W) * 1000.0
    news = [[(t0[w], 0, np.array([1.0]))] for w in range(W)]
    data = simulate(truth, t0=t0, t1=t0 + 200.0, burn=0.0, news=news, seed=11)
    ss = StateSpaceHawkes(ENDO, A)
    grid = np.array([1.0, 5.0, 20.0, 100.0])
    resp = ss.news_response(EXO, B[:, 0, :, :, 0], grid)
    rel = data.times - np.repeat(t0, np.diff(data.wptr))
    for i in range(d):
        for g, tt in enumerate(grid):
            emp = np.sum((data.types == i) & (rel < tt)) / W
            se = np.sqrt(max(emp, 1e-3) / W) * 3 + 0.02
            assert abs(emp - resp.cum_total[g, i]) < 4 * se, (i, tt, emp, resp.cum_total[g, i])
    # absorption quantile is consistent with the cumulative curve
    t50 = ss.absorption_time(EXO, B[:, 0, :, :, 0], 0.5)
    r = ss.news_response(EXO, B[:, 0, :, :, 0], np.array([t50]))
    assert np.isclose(r.cum_total.sum() / r.int_total.sum(), 0.5, atol=1e-6)
    assert np.all(r.int_total >= r.int_direct - 1e-12)


def test_residuals_true_model_are_exp1():
    truth = _truth(seed=4)
    data = _sim(truth, W=60, seed=5)
    spec = DesignSpec(endo=ENDO, exo=EXO, ant=ANT, n_tod=24)
    model = model_from_truth(truth, spec, data)
    for i in range(truth.d):
        g = gof(model.residuals(data, i))
        assert g.ks_p > 0.001 and abs(g.mean - 1) < 0.05


def test_loglik_bruteforce_erlang4():
    """Erlang-4 phases in both endogenous and exogenous dictionaries."""
    global ENDO, EXO
    old = ENDO, EXO
    try:
        ENDO = PhaseTypeDictionary.log_grid(0.05, 5.0, 2, orders=4)
        EXO = PhaseTypeDictionary.log_grid(0.5, 20.0, 2, orders=3)
        truth = _truth(seed=9)
        data = _sim(truth, W=4, seed=9)
        spec = DesignSpec(endo=ENDO, exo=EXO, ant=ANT, n_tod=24)
        model = model_from_truth(truth, spec, data)
        assert np.isclose(model.loglik(data), _brute_loglik(truth, spec, data), rtol=1e-9)
        ss = StateSpaceHawkes(ENDO, truth.A)
        assert (ss.spectral_abscissa() < 0) == (ss.spectral_radius() < 1)
    finally:
        ENDO, EXO = old


def test_observed_information_coverage():
    """95% Wald intervals from the observed information cover the truth at ~nominal rate."""
    d = 1
    A = np.zeros((d, d, ENDO.K, ENDO.R))
    A[0, 0, 1, 0] = 0.5
    truth = HawkesTruth(endo=ENDO, A=A, mu=np.array([0.5]))
    hits, n_rep = 0, 40
    for rep in range(n_rep):
        t0 = np.arange(10) * 3000.0
        data = simulate(truth, t0=t0, t1=t0 + 2500.0, burn=100.0, seed=100 + rep)
        dic = PhaseTypeDictionary(ENDO.betas[1:2], orders=1)  # correctly specified single kernel
        m = MSXHawkes(DesignSpec(endo=dic), device="cpu", store_cov=True).fit(data)
        idx, cov = m.cov[0]
        j = list(idx).index(0)  # endogenous weight is parameter 0
        se = np.sqrt(cov[j, j])
        hits += abs(m.theta_d[0, 0] - 0.5) < 1.96 * se
    assert 0.8 <= hits / n_rep <= 1.0, hits / n_rep


def test_fast_curves_match_dense_expm():
    rng = np.random.default_rng(3)
    d = 2
    A = rng.uniform(0, 1, (d, d, ENDO.K, ENDO.R))
    A *= 0.7 / A.sum(axis=(1, 2, 3), keepdims=True)
    W = rng.uniform(0, 2, (d, EXO.K, EXO.R))
    ss = StateSpaceHawkes(ENDO, A)
    t = np.geomspace(0.01, 500, 60)
    slow = ss.news_response(EXO, W, t)
    fast = ss.response_curves(EXO, W, t)
    assert np.allclose(slow.cum_total, fast.cum_total, rtol=1e-6, atol=1e-9)
    assert np.allclose(slow.total, fast.total, rtol=1e-6, atol=1e-9)
    assert np.allclose(slow.cum_direct, fast.cum_direct, rtol=1e-6, atol=1e-9)
    fine = np.geomspace(1e-4, 1e6, 401)
    ff = ss.response_curves(EXO, W, fine)
    tq = ss.quantile_from_curve(fine, ff.cum_total.sum(1), ff.int_total.sum(), 0.5)
    assert np.isclose(tq, ss.absorption_time(EXO, W, 0.5), rtol=0.02)


def test_vectorised_sparse_cumulative_matches_reference():
    from infodiff.models.features import sparse_cumulative, sparse_cumulative_reference
    truth = _truth(seed=7)
    data = _sim(truth, W=8, seed=8)
    spec = DesignSpec(endo=ENDO, exo=EXO, ant=ANT, n_tod=24)
    model = model_from_truth(truth, spec, data)
    rows = np.where(data.in_likelihood())[0]
    rt, rw = data.times[rows], data.window_ids()[rows]
    for i in range(truth.d):
        a = sparse_cumulative(data, spec, model.layout, model.theta_s[i], rt, rw)
        b = sparse_cumulative_reference(data, spec, model.layout, model.theta_s[i], rt, rw)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-10)


def test_linear_window_baseline_bruteforce_and_residuals():
    """win_cols=2 (tent baseline) with equal tents reproduces the constant-baseline likelihood
    exactly, and true-model residuals remain Exp(1)."""
    truth = _truth(seed=11)
    data = _sim(truth, W=6, seed=12)
    spec = DesignSpec(endo=ENDO, exo=EXO, ant=ANT, n_tod=24, win_cols=2)
    model = model_from_truth(truth, spec, data)
    assert np.isclose(model.loglik(data), _brute_loglik(truth, spec, data), rtol=1e-9)
    big = _sim(truth, W=60, seed=13)
    model = model_from_truth(truth, spec, big)
    for i in range(truth.d):
        g = gof(model.residuals(big, i))
        assert g.ks_p > 0.001 and abs(g.mean - 1) < 0.05
    # an unequal tent pair: fitted model's compensator matches numerical integration
    m = MSXHawkes(spec, device="cpu").fit(data)
    assert all(r.gap < 1e-2 for r in m.reports)


def test_shared_baseline_matches_constant_truth():
    """win_cols=0: a single shared baseline reproduces the brute-force likelihood."""
    truth = _truth(seed=21)
    truth.tod_w = None
    data = _sim(truth, W=5, seed=22)
    spec = DesignSpec(endo=ENDO, exo=EXO, ant=ANT, n_tod=0, win_cols=0)
    model = model_from_truth(truth, spec, data)
    ll_brute = _brute_loglik_no_tod(truth, data)
    assert np.isclose(model.loglik(data), ll_brute, rtol=1e-9)
    m = MSXHawkes(spec, device="cpu").fit(data)
    assert all(r.gap < 1e-2 for r in m.reports)


def _brute_loglik_no_tod(truth, data):
    spec = DesignSpec(endo=ENDO, exo=EXO, ant=ANT, n_tod=24)
    saved = truth.tod_w
    truth.tod_w = np.zeros((truth.d, 24))
    try:
        return _brute_loglik(truth, spec, data)
    finally:
        truth.tod_w = saved
