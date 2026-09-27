"""Classical Hawkes estimators behind one interface, scored with one likelihood engine.

Every fitted model exposes: branching matrix G, baseline mu, kernel(i, j, t) and an
exact held-out log-likelihood.  Exponential-sum models (tick ExpKern / SumExpKern /
ADM4, MSX) are scored with the exact phase-type recursion; tabulated nonparametric
kernels (tick EM, ConditionalLaw) with an exact direct summation over the kernel
support (piecewise-constant or piecewise-linear interpolation of the estimate).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numba import njit

from ..events.data import EventData
from ..models.dictionary import PhaseTypeDictionary
from ..models.features import DesignSpec, build_design
from ..models.msx import MSXHawkes

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class Fitted:
    name: str
    G: np.ndarray
    mu: np.ndarray
    seconds: float
    kernel: Callable[[int, int, np.ndarray], np.ndarray] | None = None
    loglik: Callable[[EventData], float] | None = None
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ likelihood engines
def phase_type_loglik(dic: PhaseTypeDictionary, A: np.ndarray, mu: np.ndarray, data: EventData) -> float:
    """Exact LL with constant baseline mu (d,) and weights A (d, d, K, R)."""
    ll = 0.0
    for i in range(data.n_dims):
        des = build_design(data, DesignSpec(endo=dic), i)
        lam = des.Xd.astype(np.float64) @ A[i].reshape(-1) + mu[i]
        ll += np.log(lam).sum() - des.integ_d @ A[i].reshape(-1) - mu[i] * (data.t1 - data.t0).sum()
    return float(ll)


@njit(cache=True)
def _tab_loglik(times, types, wptr, t0, t1, mu, grid, vals, cum, linear):
    d = mu.shape[0]
    L = grid.shape[0]
    S = grid[L - 1]
    ll = 0.0
    for w in range(t0.shape[0]):
        a, b = wptr[w], wptr[w + 1]
        for n in range(a, b):
            t = times[n]
            if t < t0[w] or t >= t1[w]:
                continue
            i = types[n]
            lam = mu[i]
            m = n - 1
            while m >= a and t - times[m] < S:
                lag = t - times[m]
                if lag > 0.0:
                    # locate bin
                    lo, hi = 0, L - 1
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if grid[mid] <= lag:
                            lo = mid
                        else:
                            hi = mid
                    if linear:
                        f = (lag - grid[lo]) / (grid[hi] - grid[lo])
                        lam += vals[i, types[m], lo] * (1 - f) + vals[i, types[m], hi] * f
                    else:
                        lam += vals[i, types[m], lo]
                m -= 1
            ll += np.log(max(lam, 1e-300))
        for i in range(d):
            ll -= mu[i] * (t1[w] - t0[w])
        for m in range(a, b):
            if times[m] >= t1[w]:
                continue
            lo_lag = max(t0[w] - times[m], 0.0)
            hi_lag = min(t1[w] - times[m], S)
            for i in range(d):
                ll -= _cum_at(grid, cum[i, types[m]], hi_lag) - _cum_at(grid, cum[i, types[m]], lo_lag)
    return ll


@njit(cache=True)
def _cum_at(grid, cum, x):
    L = grid.shape[0]
    if x <= grid[0]:
        return 0.0
    if x >= grid[L - 1]:
        return cum[L - 1]
    lo, hi = 0, L - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - grid[lo]) / (grid[hi] - grid[lo])
    return cum[lo] * (1 - f) + cum[hi] * f


def tabulated_loglik(grid: np.ndarray, vals: np.ndarray, mu: np.ndarray, data: EventData, linear: bool) -> float:
    """grid (L,) lags starting at 0; vals (d, d, L) kernel values (bin values if not linear)."""
    vals = np.maximum(vals, 0.0)
    if linear:
        seg = 0.5 * (vals[..., 1:] + vals[..., :-1]) * np.diff(grid)
    else:
        seg = vals[..., :-1] * np.diff(grid)
    cum = np.concatenate([np.zeros(vals.shape[:2] + (1,)), np.cumsum(seg, axis=-1)], axis=-1)
    return float(_tab_loglik(data.times, data.types, data.wptr, data.t0, data.t1, np.asarray(mu, float),
                             grid, vals, cum, linear))


# ------------------------------------------------------------------ helpers
def to_tick(data: EventData) -> tuple[list, list]:
    events, ends = [], []
    for w in range(data.n_windows):
        t, u = data.window(w)
        m = (t >= data.t0[w]) & (t < data.t1[w])
        t, u = t[m] - data.t0[w], u[m]
        events.append([np.ascontiguousarray(t[u == i]) for i in range(data.n_dims)])
        ends.append(float(data.t1[w] - data.t0[w]))
    return events, ends


def _exp_sum_kernel(decays, adj):
    decays = np.atleast_1d(decays)
    adj = adj.reshape(adj.shape[0], adj.shape[1], -1)

    def k(i, j, t):
        t = np.asarray(t, float)
        return (adj[i, j][:, None] * decays[:, None] * np.exp(-decays[:, None] * t[None, :])).sum(0)
    return k


def _exp_sum_loglik(decays, adj, mu):
    dic = PhaseTypeDictionary(np.atleast_1d(np.asarray(decays, float)), orders=1)
    A = adj.reshape(adj.shape[0], adj.shape[1], -1, 1)
    return lambda data: phase_type_loglik(dic, A, mu, data)


# ------------------------------------------------------------------ estimators
def fit_msx(data: EventData, dic: PhaseTypeDictionary, name: str = "MSX", l1: float = 0.0, **kw) -> Fitted:
    t = time.time()
    m = MSXHawkes(DesignSpec(endo=dic), l1_endo=l1, **kw).fit(data)
    A = m.endo_weights()
    mu = m.theta_s[:, : data.n_windows].mean(axis=1)
    sec = time.time() - t

    def k(i, j, tt):
        return dic.kernel(A[i, j].reshape(-1), np.asarray(tt, float))

    return Fitted(name, A.sum(axis=(2, 3)), mu, sec, k, lambda dd: phase_type_loglik(dic, A, mu, dd),
                  extra=dict(gap=max(r.gap for r in m.reports), model=m))


def fit_msx_auto(data: EventData, dic: PhaseTypeDictionary, orders=(1, 2, 3, 4), l1_grid=(0.0, 0.01, 0.05),
                 val_frac: float = 0.2, name: str = "MSX-auto (ours)", seed: int = 0) -> Fitted:
    """MSX with (Erlang order R, exposure-weighted L1) chosen on held-out windows, then refit on all."""
    t = time.time()
    W = data.n_windows
    if W < 3:
        raise ValueError("fit_msx_auto needs >= 3 windows")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(W)
    n_val = max(1, int(round(val_frac * W)))
    d_tr, d_val = data.subset(np.sort(perm[n_val:])), data.subset(np.sort(perm[:n_val]))
    scores = {}
    for R in orders:
        dR = PhaseTypeDictionary(dic.betas, orders=R)
        for l1 in l1_grid:
            scores[(R, l1)] = fit_msx(d_tr, dR, l1=l1).loglik(d_val)
    R, l1 = max(scores, key=scores.get)
    f = fit_msx(data, PhaseTypeDictionary(dic.betas, orders=R), name=name, l1=l1)
    f.seconds = time.time() - t
    f.extra.update(R=R, l1=l1)
    return f


def _tick_mle(model, d: int, n_ker: int, max_iter: int = 3000, tol: float = 1e-10):
    """Positive-constrained MLE with tick's accelerated gradient solver (as its learners do)."""
    from tick.prox import ProxPositive
    from tick.solver import AGD
    last = None
    for scale in (1.0, 0.1, 0.01):  # retry from smaller starts if tick's line search leaves the domain
        x0 = np.concatenate([np.full(d, 0.1), np.full(d * d * n_ker, 0.01 * scale)])
        solver = AGD(max_iter=max_iter, tol=tol, linesearch=True, verbose=False)
        solver.set_model(model).set_prox(ProxPositive())
        try:
            x = solver.solve(x0)
            return x[:d], x[d:].reshape(d, d, n_ker), -model.loss(x)
        except RuntimeError as e:
            last = e
    raise last


def fit_tick_expkern(data: EventData, decays_grid: np.ndarray) -> Fitted:
    from tick.hawkes.model import ModelHawkesExpKernLogLik
    ev, ends = to_tick(data)
    d = data.n_dims
    t = time.time()
    best = None
    for beta in decays_grid:
        try:
            model = ModelHawkesExpKernLogLik(float(beta), n_threads=4).fit(ev, end_times=np.asarray(ends))
            mu, adj, sc = _tick_mle(model, d, 1)
        except Exception:
            continue
        if np.isfinite(sc) and (best is None or sc > best[0]):
            best = (sc, beta, adj[..., 0], mu)
    sec = time.time() - t
    _, beta, adj, mu = best
    return Fitted("tick-ExpKern(MLE, best decay)", adj, mu, sec, _exp_sum_kernel(beta, adj),
                  _exp_sum_loglik(beta, adj, mu), extra=dict(decay=beta))


def fit_tick_sumexp(data: EventData, decays: np.ndarray) -> Fitted:
    """tick's SumExp *likelihood* model on the same decay grid as MSX (exponential phases only)."""
    from tick.hawkes.model import ModelHawkesSumExpKernLogLik
    ev, ends = to_tick(data)
    d = data.n_dims
    t = time.time()
    model = ModelHawkesSumExpKernLogLik(np.asarray(decays, float), n_threads=4).fit(ev, end_times=np.asarray(ends))
    mu, adj, _ = _tick_mle(model, d, len(decays))
    sec = time.time() - t
    mu = np.maximum(mu, 1e-12)
    return Fitted("tick-SumExpKern(MLE, AGD)", adj.sum(-1), mu, sec, _exp_sum_kernel(decays, adj),
                  _exp_sum_loglik(decays, adj, mu))


def fit_tick_em(data: EventData, edges: np.ndarray) -> Fitted:
    from tick.hawkes import HawkesEM
    ev, ends = to_tick(data)
    t = time.time()
    em = HawkesEM(kernel_discretization=np.asarray(edges, float), max_iter=300, tol=1e-7, n_threads=4)
    em.fit(ev, end_times=ends)
    sec = time.time() - t
    vals = np.concatenate([em.kernel, em.kernel[..., -1:] * 0], axis=-1)  # bin values, last edge -> 0

    def k(i, j, tt):
        idx = np.clip(np.searchsorted(edges, tt, side="right") - 1, 0, len(edges) - 1)
        out = vals[i, j][idx]
        out[np.asarray(tt) >= edges[-1]] = 0.0
        return out

    return Fitted("tick-EM(nonparam)", em.get_kernel_norms(), em.baseline.copy(), sec, k,
                  lambda dd: tabulated_loglik(np.asarray(edges, float), vals, em.baseline, dd, linear=False))


def fit_tick_adm4(data: EventData, decays_grid: np.ndarray) -> Fitted:
    """ADM4 (Zhou et al. 2013; low-rank + sparse exp-kernel Hawkes); decay chosen by training LL."""
    from tick.hawkes import HawkesADM4
    ev, ends = to_tick(data)
    t = time.time()
    best = None
    for decay in decays_grid:
        try:
            m = HawkesADM4(decay=float(decay), C=1e3, lasso_nuclear_ratio=0.5, max_iter=200, tol=1e-7, n_threads=4)
            m.fit(ev, end_times=np.asarray(ends))
            adj, mu = m.adjacency.copy(), np.maximum(m.baseline.copy(), 1e-12)
            ll = _exp_sum_loglik(decay, adj, mu)(data)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[0]):
            best = (ll, decay, adj, mu)
    sec = time.time() - t
    _, decay, adj, mu = best
    return Fitted("tick-ADM4", adj, mu, sec, _exp_sum_kernel(decay, adj), _exp_sum_loglik(decay, adj, mu),
                  extra=dict(decay=decay))


def fit_tick_nphc(data: EventData, H: float) -> Fitted:
    """NPHC with tick's cumulant estimators and our PyTorch solver of the paper's objective."""
    from .nphc import nphc_fit
    ev, ends = to_tick(data)
    t = time.time()
    G, mu = nphc_fit(ev, np.asarray(ends), H)
    return Fitted("NPHC(cumulants)", G, mu, time.time() - t)


def fit_tick_claw(data: EventData, configs: list[tuple] | None = None) -> Fitted:
    """Bacry-Muzy conditional-law (Wiener-Hopf) estimator; the quadrature configuration
    is chosen by training likelihood among a small grid (it is known to be sensitive)."""
    from tick.hawkes import HawkesConditionalLaw
    ev, ends = to_tick(data)
    d = data.n_dims
    configs = configs or [(0.1, 1e-4, 40.0, 40.0, "lin"), (0.01, 1e-3, 100.0, 300.0, "log"),
                          (0.05, 1e-3, 20.0, 100.0, "log"), (0.001, 1e-3, 10.0, 50.0, "log")]
    t = time.time()
    best = None
    for (delta, mn, mx, sup, qm) in configs:
        try:
            m = HawkesConditionalLaw(delta_lag=delta, min_lag=mn, max_lag=mx, n_quad=60, max_support=sup,
                                     min_support=mn, quad_method=qm, n_threads=4)
            for real, T in zip(ev, ends):
                m.incremental_fit(real, compute=False, T=T)
            m.compute()
            grid = np.concatenate([[0.0], np.geomspace(mn, sup, 200)])
            vals = np.stack([np.stack([np.maximum(np.nan_to_num(m.get_kernel_values(i, j, grid)), 0)
                                       for j in range(d)]) for i in range(d)])
            vals[..., 0] = vals[..., 1]
            mu = np.maximum(np.asarray(m.baseline, float), 1e-9)
            ll = tabulated_loglik(grid, vals, mu, data, linear=True)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[0]):
            best = (ll, grid, vals, mu, np.asarray(m.get_kernel_norms()))
    sec = time.time() - t
    _, grid, vals, mu, G = best

    def k(i, j, tt):
        return np.interp(tt, grid, vals[i, j], right=0.0)

    return Fitted("tick-ConditionalLaw", G, mu, sec, k,
                  lambda dd: tabulated_loglik(grid, vals, mu, dd, linear=True))
