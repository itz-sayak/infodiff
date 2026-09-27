"""Track D — post-release forecasting on the news panel (chronological split).

Targets for each test release (time tau) and asset a, horizons h in {60, 300, 900} s:
  N_a(h)     number of delta-crossings in (tau, tau+h]      -> activity
  RV_a(h)    = delta_a^2 * N_a(h)  (quadratic variation of the delta-sampled price)
  D_a(h)     = delta_a * (N_up - N_down)                  -> signed drift (bps)

MSX closed-form forecast (no simulation).  With z = [x; y] the endogenous and
exogenous phase-type states, the mean intensity obeys
    z' = A z + u,   u = [Gin mu; 0],   E[lambda(t)] = mu + C z(t),
so  E[N(tau, tau+h]] = mu h + C [A^{-1}(e^{Ah} - I) z0 + A^{-1}(A^{-1}(e^{Ah} - I) - h I) u],
where z0 collects the pre-release history (exact phase-type states at tau) and the
release impulse with its observed surprise marks, and mu is the window baseline
re-estimated from the pre-release part of the test window only.

Baselines
  climatology   mean of the target over training releases of the same type
  event-study   OLS of log(1+N) on log(1+pre-release counts over 5/30 min), |z| and
                type dummies (Andersen et al. 2003 style), fitted on training releases
  HAR           OLS of log(1+N) on log(1+N) over the previous 1, 5 and 30 minutes
Metrics: RMSE of log(1+N), QLIKE of RV, sign hit-rate and correlation of drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve

from ..events.data import EventData
from ..models.msx import MSXHawkes
from ..models.statespace import _blocks

HORIZONS = (60.0, 300.0, 900.0)


def _pre_states(data: EventData, w: int, tau: float, dic) -> np.ndarray:
    """Endogenous phase-type states just before tau: s_{jkr} = sum_{t_m < tau, u_m = j} g_kr(tau - t_m)."""
    t, u = data.window(w)
    m = t < tau
    out = np.zeros((data.n_dims, dic.size))
    for j in range(data.n_dims):
        lags = tau - t[m & (u == j)]
        lags = lags[lags < 50.0 / dic.betas.min()]
        if len(lags):
            out[j] = dic.pdf(lags).sum(0)
    return out.reshape(-1)


def _baseline_pre(model: MSXHawkes, data: EventData, w: int, tau: float, i: int, x_rows, n_iter: int = 50) -> float:
    """ML window intercept for dim i from the pre-release part of window w, other terms fixed."""
    t, u = data.window(w)
    lo = data.t0[w]
    sel = (t >= lo) & (t < tau) & (u == i)
    T = tau - lo
    n = sel.sum()
    if n == 0:
        return 1e-6
    r = x_rows  # other-term intensities at those events (>= 0)
    c = max(n / T - r.mean(), 1e-6)
    for _ in range(n_iter):  # Newton on sum 1/(c + r) = T
        f = np.sum(1.0 / (c + r)) - T
        fp = -np.sum(1.0 / (c + r) ** 2)
        c_new = max(c - f / fp, 0.5 * c)
        if abs(c_new - c) < 1e-10 * c:
            break
        c = c_new
    return c


def msx_forecast(model: MSXHawkes, data: EventData, w: int, tau: float, news_idx: list[int],
                 horizons=HORIZONS) -> np.ndarray:
    """Expected counts (len(horizons), d) of each dimension in (tau, tau+h]."""
    spec = model.spec
    d = data.n_dims
    endo, exo = spec.endo, spec.exo
    F, Gin = _blocks(endo, d)
    Theta = model.theta_d  # (d, d*K*R)
    x0 = _pre_states(data, w, tau, endo)
    B = model.exo_weights(data.n_news_types, data.n_marks)  # (d, C, Ke, Re, M)
    Fe1, _ = _blocks(exo, 1)
    ny = Fe1.shape[0]
    blocks_W, y0s = [], []
    for e in news_idx:
        c = data.news_type[e]
        z = data.news_marks[e]
        W = np.tensordot(B[:, c], z, axes=([-1], [0])).reshape(d, -1)
        lag = max(tau - data.news_t[e], 0.0)
        # exo states at tau (+ the impulse if the release is at tau): y = g_kr(lag) with the
        # impulse limit y0[::R] = beta at lag 0
        if lag == 0.0:
            y = np.zeros(ny)
            y[::exo.R] = exo.betas
        else:
            y = exo.pdf(np.array([lag]))[0]
        blocks_W.append(W)
        y0s.append(y)
    nx = F.shape[0]
    E = len(blocks_W)
    A = np.zeros((nx + E * ny, nx + E * ny))
    A[:nx, :nx] = F + Gin @ Theta
    C = np.zeros((d, nx + E * ny))
    C[:, :nx] = Theta
    for k, W in enumerate(blocks_W):
        sl = slice(nx + k * ny, nx + (k + 1) * ny)
        A[:nx, sl] = Gin @ W
        A[sl, sl] = Fe1
        C[:, sl] = W
    z0 = np.concatenate([x0] + y0s)
    # baseline: window intercept re-estimated on the pre-release data + time-of-day hats at tau
    lay = model.layout
    mu = np.zeros(d)
    tod = (data.tod0[w] + (tau - data.t0[w])) % 86400.0
    if spec.n_tod:
        h = 86400.0 / spec.n_tod
        m0 = int(tod // h)
        f = tod / h - m0
        hats = model.theta_s[:, lay["off_tod"]:lay["off_tod"] + spec.n_tod]
        mu += hats[:, m0 % spec.n_tod] * (1 - f) + hats[:, (m0 + 1) % spec.n_tod] * f
    mu += model.window_levels(data.n_windows)[:, w]
    u_in = np.zeros(nx + E * ny)
    u_in[:nx] = Gin @ mu
    Ainv = np.linalg.inv(A)
    out = []
    for hz in horizons:
        eA = expm(A * hz)
        I = np.eye(A.shape[0])
        int_z = Ainv @ (eA - I) @ z0 + Ainv @ (Ainv @ (eA - I) - hz * I) @ u_in
        out.append(mu * hz + C @ int_z)
    return np.array(out)


def realised(data: EventData, w: int, tau: float, horizons=HORIZONS) -> np.ndarray:
    t, u = data.window(w)
    out = []
    for hz in horizons:
        m = (t > tau) & (t <= tau + hz)
        out.append(np.bincount(u[m], minlength=data.n_dims))
    return np.array(out, float)


def pre_counts(data: EventData, w: int, tau: float, spans=(60.0, 300.0, 1800.0)) -> np.ndarray:
    t, u = data.window(w)
    out = []
    for s in spans:
        m = (t >= tau - s) & (t < tau)
        out.append(np.bincount(u[m], minlength=data.n_dims))
    return np.array(out, float)


def metrics(pred_n: np.ndarray, true_n: np.ndarray, deltas: np.ndarray) -> dict:
    """pred_n, true_n: (R, H, d) counts; deltas (A,) bps for assets (dims 2a, 2a+1)."""
    A = len(deltas)
    act_p = pred_n.reshape(*pred_n.shape[:2], A, 2).sum(-1)
    act_t = true_n.reshape(*true_n.shape[:2], A, 2).sum(-1)
    rmse_log = np.sqrt(np.mean((np.log1p(act_p) - np.log1p(act_t)) ** 2, axis=0))  # (H, A)
    rv_p = np.maximum(act_p, 1e-3) * deltas ** 2
    rv_t = np.maximum(act_t, 0.5) * deltas ** 2
    qlike = np.mean(rv_t / rv_p - np.log(rv_t / rv_p) - 1, axis=0)
    dp = (pred_n[..., 0::2] - pred_n[..., 1::2]) * deltas
    dt = (true_n[..., 0::2] - true_n[..., 1::2]) * deltas
    hit = np.mean(np.sign(dp) == np.sign(dt), axis=0)
    corr = np.array([[np.corrcoef(dp[:, h, a], dt[:, h, a])[0, 1] if dp[:, h, a].std() > 0 else np.nan
                      for a in range(A)] for h in range(dp.shape[1])])
    return dict(rmse_log=rmse_log.tolist(), qlike=qlike.tolist(), drift_hit=hit.tolist(), drift_corr=corr.tolist())


def save(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=float))
