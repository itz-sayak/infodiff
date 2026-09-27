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
            x0, pk, h = states[:, n - 1], mus[:, n - 1], H[:, n - 1]
            mu, g, w = pk[:, :M], pk[:, M:3 * M], pk[:, 3 * M:3 * M + 2]
            grid = torch.linspace(0, 1, 20001, dtype=torch.float64)[None, :] * dts[:, n:n + 1]
            xs, _ = m.evolve(x0[:, None, :].expand(-1, grid.shape[1], -1), grid)
            lam = mu[:, None, :] + m._readout(m.C(h), xs)
            gv, _ = m._gomp_terms(g[:, None, :], w[:, None, :].expand(-1, grid.shape[1], -1), grid)
            lam = lam + gv
            num = torch.trapezoid(lam.sum(-1), grid, dim=1)
            assert torch.allclose(num, comp[:, n - 1], rtol=1e-5, atol=1e-8), (num, comp[:, n - 1])
            # log-intensity of the observed mark at the event matches the curve end-point
            ref = torch.log(lam[torch.arange(B), -1, marks[:, n]])
            assert torch.allclose(ref, log_lam[:, n - 1], atol=1e-6)
