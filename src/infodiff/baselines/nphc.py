"""NPHC — Non-Parametric Hawkes Cumulant matching (Achab et al., JMLR 2017).

tick 0.8 ships the PyTorch solver as an unimplemented stub, so the objective is
implemented here (PyTorch, Adam) on top of tick's C++ cumulant estimators.

With R = (I - G)^{-1} and L = diag(Lambda):
    C(R)      = R L R^T
    K^c_ij(R) = sum_m R_im^2 C_jm + 2 R_im C_im R_jm - 2 R_im^2 R_jm Lambda_m
objective (1-k) ||K^c(R) - K_hat||^2 + k ||C(R) - C_hat||^2,  k = cs-ratio as in the paper.
"""
from __future__ import annotations

import numpy as np
import torch


def nphc_fit(events: list, end_times: np.ndarray, H: float, max_iter: int = 5000, lr: float = 1e-2,
             seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    from tick.hawkes import HawkesCumulantMatching
    m = HawkesCumulantMatching(integration_support=H)
    from tick.hawkes.inference.base import LearnerHawkesNoParam
    LearnerHawkesNoParam.fit(m, events, end_times=end_times)
    m.compute_cumulants()
    L = np.asarray(m.mean_intensity, float)
    Ch = np.asarray(m.covariance, float)
    Kh = np.asarray(m.skewness, float)
    kappa = np.linalg.norm(Kh) ** 2 / (np.linalg.norm(Kh) ** 2 + np.linalg.norm(Ch) ** 2)

    # starting point from the paper: R0 = C^{1/2} L^{-1/2}
    w, V = np.linalg.eigh(0.5 * (Ch + Ch.T))
    sqrtC = V @ np.diag(np.sqrt(np.maximum(w, 1e-12))) @ V.T
    R0 = sqrtC @ np.diag(1.0 / np.sqrt(L))

    torch.manual_seed(seed)
    Lt = torch.tensor(L, dtype=torch.float64)
    Ct = torch.tensor(Ch, dtype=torch.float64)
    Kt = torch.tensor(Kh, dtype=torch.float64)
    R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([R], lr=lr)
    prev = np.inf
    for it in range(max_iter):
        opt.zero_grad()
        C = R @ torch.diag(Lt) @ R.T
        R2 = R * R
        K = R2 @ C.T + 2 * (R * C) @ R.T - 2 * R2 @ (R.T * Lt[:, None])
        loss = (1 - kappa) * torch.sum((K - Kt) ** 2) + kappa * torch.sum((C - Ct) ** 2)
        loss.backward()
        opt.step()
        cur = float(loss)
        if abs(prev - cur) < 1e-12 * max(1.0, abs(cur)):
            break
        prev = cur
    Rn = R.detach().numpy()
    G = np.eye(len(L)) - np.linalg.inv(Rn)
    mu = np.linalg.solve(Rn, L)
    return G, mu
