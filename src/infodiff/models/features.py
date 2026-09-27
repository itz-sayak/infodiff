"""Design-matrix construction for linear-in-parameters Hawkes intensities.

For a target dimension i the intensity at time t is

    lambda_i(t) = x(t) . theta_i,   theta_i >= 0,

where x(t) >= 0 stacks
  * a dense *endogenous* block: phase-type excitation states s_{j,k,r}(t) driven by
    past events of every source j (numba recursion, O(N d K R));
  * a sparse block: per-window intercepts, periodic hat functions of local
    time-of-day (seasonality), mark-modulated exogenous news kernels, and
    anticipatory (pre-announcement) kernels.

Every block is also integrated exactly over the likelihood intervals, so the
log-likelihood  sum_n log(x_n . theta) - theta . Phi  is available in closed form.

Simultaneous events (identical timestamps) never excite each other: states are
recorded for the whole tie group before the group's jumps are applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numba import njit

from ..events.data import EventData
from .dictionary import PhaseTypeDictionary

DAY = 86400.0
_CUT = 50.0  # basis values with beta*lag > _CUT are treated as exactly zero


# ----------------------------------------------------------------------------------
# Endogenous block (numba)
# ----------------------------------------------------------------------------------
@njit(cache=True)
def _poisson_weights(bdt, R, w, Pg):
    """w_j = e^{-y} y^j / j!,  Pg_j = P(j+1, y) = 1 - sum_{i<=j} w_i  for j < R."""
    term = np.exp(-bdt)
    acc = 0.0
    for j in range(R):
        if j > 0:
            term = term * bdt / j
        w[j] = term
        acc += term
        Pg[j] = max(1.0 - acc, 0.0)


@njit(cache=True)
def _advance(S, C, betas, R, dt, acc, w, Pg, tmp):
    """Exact Erlang-chain evolution of states S[d, K, R] over dt; if acc, add integrals to C.

    Per rate block s' = beta (N - I) s:  s_r(dt) = sum_{i<=r} w_i s_{r-i},
    int_0^dt s_r = sum_{i<=r} P(i+1, beta dt) s_{r-i} / beta."""
    d, K = S.shape[0], S.shape[1]
    for k in range(K):
        b = betas[k]
        _poisson_weights(b * dt, R, w, Pg)
        for j in range(d):
            nz = False
            for r in range(R):
                if S[j, k, r] != 0.0:
                    nz = True
                    break
            if not nz:
                continue
            for r in range(R):
                v = 0.0
                ig = 0.0
                for i in range(r + 1):
                    v += w[i] * S[j, k, r - i]
                    ig += Pg[i] * S[j, k, r - i]
                tmp[r] = v
                if acc:
                    C[j, k, r] += ig / b
            for r in range(R):
                S[j, k, r] = tmp[r]


@njit(cache=True)
def _advance_split(S, C, betas, R, tcur, tnext, t0, w, Pg, tmp):
    if tnext <= tcur:
        return
    if tnext <= t0:
        _advance(S, C, betas, R, tnext - tcur, False, w, Pg, tmp)
    elif tcur >= t0:
        _advance(S, C, betas, R, tnext - tcur, True, w, Pg, tmp)
    else:
        _advance(S, C, betas, R, t0 - tcur, False, w, Pg, tmp)
        _advance(S, C, betas, R, tnext - t0, True, w, Pg, tmp)


@njit(cache=True)
def _endo_features(times, types, wptr, t0, t1, d, betas, R, target, want_cum):
    K = betas.shape[0]
    P = d * K * R
    W = t0.shape[0]
    n = 0
    for w_ in range(W):
        for i in range(wptr[w_], wptr[w_ + 1]):
            if times[i] >= t0[w_] and times[i] < t1[w_] and (target < 0 or types[i] == target):
                n += 1
    X = np.zeros((n, P), dtype=np.float32)
    Cm = np.zeros((n if want_cum else 0, P))
    rows = np.empty(n, dtype=np.int64)
    integ = np.zeros(P)
    S = np.zeros((d, K, R))
    C = np.zeros((d, K, R))
    w = np.zeros(R)
    Pg = np.zeros(R)
    tmp = np.zeros(R)
    r_ = 0
    for w_ in range(W):
        S[:] = 0.0
        C[:] = 0.0
        a = wptr[w_]
        b = wptr[w_ + 1]
        tcur = t0[w_]
        if b > a and times[a] < tcur:
            tcur = times[a]
        i = a
        while i < b:
            t = times[i]
            if t >= t1[w_]:
                break
            _advance_split(S, C, betas, R, tcur, t, t0[w_], w, Pg, tmp)
            tcur = t
            j = i
            while j < b and times[j] == t:
                j += 1
            if t >= t0[w_]:
                for m in range(i, j):
                    if target < 0 or types[m] == target:
                        rows[r_] = m
                        for src in range(d):
                            for k in range(K):
                                for rr in range(R):
                                    c = (src * K + k) * R + rr
                                    X[r_, c] = S[src, k, rr]
                                    if want_cum:
                                        Cm[r_, c] = C[src, k, rr]
                        r_ += 1
            for m in range(i, j):
                u = types[m]
                for k in range(K):
                    S[u, k, 0] += betas[k]
            i = j
        _advance_split(S, C, betas, R, tcur, t1[w_], t0[w_], w, Pg, tmp)
        for src in range(d):
            for k in range(K):
                for rr in range(R):
                    integ[(src * K + k) * R + rr] += C[src, k, rr]
    return rows, X, Cm, integ


# ----------------------------------------------------------------------------------
# Sparse block (numba): window intercepts, time-of-day hats, news, anticipation
# ----------------------------------------------------------------------------------
@njit(cache=True)
def _sparse_rows(row_t, row_w, t0, tod0, news_ptr, news_t, news_type, news_marks,
                 n_tod, exo_b, exo_R, ant_b, n_types, off_tod, off_exo, off_ant, count_only,
                 indptr, indices, data):
    n = row_t.shape[0]
    M = news_marks.shape[1]
    Ke = exo_b.shape[0]
    Ka = ant_b.shape[0]
    h = 86400.0 / max(n_tod, 1)
    nnz = 0
    for r in range(n):
        if not count_only:
            indptr[r] = nnz
        t = row_t[r]
        w = row_w[r]
        # window intercept
        if not count_only:
            indices[nnz] = w
            data[nnz] = 1.0
        nnz += 1
        # time-of-day hats (periodic piecewise-linear partition of unity)
        if n_tod > 0:
            x = (tod0[w] + (t - t0[w])) % 86400.0
            m0 = int(x // h)
            frac = x / h - m0
            m1 = (m0 + 1) % n_tod
            if not count_only:
                indices[nnz] = off_tod + (m0 % n_tod)
                data[nnz] = 1.0 - frac
                indices[nnz + 1] = off_tod + m1
                data[nnz + 1] = frac
            nnz += 2
        for e in range(news_ptr[w], news_ptr[w + 1]):
            tau = news_t[e]
            c = news_type[e]
            if tau < t and Ke > 0:
                lag = t - tau
                for k in range(Ke):
                    b = exo_b[k]
                    if b * lag > 50.0:
                        continue
                    val = b * np.exp(-b * lag)
                    for rr in range(exo_R):
                        if rr > 0:
                            val = val * b * lag / rr
                        base = off_exo + ((c * Ke + k) * exo_R + rr) * M
                        for m in range(M):
                            z = news_marks[e, m]
                            if z == 0.0:
                                continue
                            if not count_only:
                                indices[nnz] = base + m
                                data[nnz] = val * z
                            nnz += 1
            elif tau > t and Ka > 0:
                lag = tau - t
                for k in range(Ka):
                    b = ant_b[k]
                    if b * lag > 50.0:
                        continue
                    if not count_only:
                        indices[nnz] = off_ant + c * Ka + k
                        data[nnz] = b * np.exp(-b * lag)
                    nnz += 1
    if not count_only:
        indptr[n] = nnz
    return nnz


def _hat_integral(a: float, b: float, n_tod: int) -> np.ndarray:
    """Exact integral of each periodic hat function over local-clock interval [a, b]."""
    out = np.zeros(n_tod)
    if n_tod == 0 or b <= a:
        return out
    h = DAY / n_tod
    k0, k1 = int(np.floor(a / h)), int(np.floor(b / h))
    pts = np.concatenate([[a], h * np.arange(k0 + 1, k1 + 1), [b]])
    pts = np.unique(pts)
    for lo, hi in zip(pts[:-1], pts[1:]):
        if hi <= lo:
            continue
        m = int(np.floor(lo / h + 1e-12))
        f_lo, f_hi = lo / h - m, hi / h - m
        width = hi - lo
        # on [m h, (m+1) h]: hat_m = 1 - f, hat_{m+1} = f (linear -> trapezoid exact)
        out[m % n_tod] += width * (1.0 - 0.5 * (f_lo + f_hi))
        out[(m + 1) % n_tod] += width * 0.5 * (f_lo + f_hi)
    return out


# ----------------------------------------------------------------------------------
# Public design object
# ----------------------------------------------------------------------------------
@dataclass
class DesignSpec:
    endo: PhaseTypeDictionary
    exo: PhaseTypeDictionary | None = None
    ant: PhaseTypeDictionary | None = None  # anticipation uses order-1 elements only
    n_tod: int = 0  # number of periodic time-of-day hat functions (0 disables)

    def layout(self, n_dims: int, n_windows: int, n_types: int, n_marks: int) -> dict:
        Pd = n_dims * self.endo.size
        off_tod = n_windows
        off_exo = off_tod + self.n_tod
        n_exo = n_types * self.exo.size * n_marks if self.exo is not None else 0
        off_ant = off_exo + n_exo
        n_ant = n_types * self.ant.K if self.ant is not None else 0
        return dict(Pd=Pd, off_tod=off_tod, off_exo=off_exo, off_ant=off_ant,
                    Ps=off_ant + n_ant, n_exo=n_exo, n_ant=n_ant)


@dataclass
class Design:
    """Design for one target dimension: rows = its in-likelihood events."""
    rows: np.ndarray  # indices into data.times
    Xd: np.ndarray  # (n, Pd) float32 endogenous block
    Xs: sp.csr_matrix  # (n, Ps) sparse block
    integ_d: np.ndarray  # (Pd,)
    integ_s: np.ndarray  # (Ps,)
    layout: dict
    cum_d: np.ndarray | None = None  # (n, Pd) cumulative endogenous integrals (for residuals)

    @property
    def n(self) -> int:
        return len(self.rows)


def _empty_dict() -> tuple[np.ndarray, int]:
    return np.zeros(0), 1


def build_design(data: EventData, spec: DesignSpec, target: int, want_cum: bool = False) -> Design:
    lay = spec.layout(data.n_dims, data.n_windows, max(data.n_news_types, 1), data.n_marks)
    rows, Xd, Cm, integ_d = _endo_features(
        data.times, data.types, data.wptr, data.t0, data.t1, data.n_dims,
        spec.endo.betas, spec.endo.R, target, want_cum)
    row_t = data.times[rows]
    row_w = data.window_ids()[rows]
    exo_b, exo_R = (spec.exo.betas, spec.exo.R) if spec.exo is not None else _empty_dict()
    ant_b = spec.ant.betas if spec.ant is not None else np.zeros(0)
    args = (row_t, row_w, data.t0, data.tod0, data.news_ptr, data.news_t, data.news_type,
            data.news_marks, spec.n_tod, exo_b, exo_R, ant_b, max(data.n_news_types, 1),
            lay["off_tod"], lay["off_exo"], lay["off_ant"])
    dummy_i, dummy_f = np.zeros(1, np.int64), np.zeros(1)
    nnz = _sparse_rows(*args, True, dummy_i, dummy_i, dummy_f)
    indptr = np.zeros(len(rows) + 1, np.int64)
    indices = np.zeros(nnz, np.int64)
    vals = np.zeros(nnz)
    _sparse_rows(*args, False, indptr, indices, vals)
    Xs = sp.csr_matrix((vals, indices, indptr), shape=(len(rows), lay["Ps"]))
    integ_s = sparse_integrals(data, spec, lay)
    return Design(rows=rows, Xd=Xd, Xs=Xs, integ_d=integ_d, integ_s=integ_s, layout=lay,
                  cum_d=Cm if want_cum else None)


def _news_integral_rows(spec: DesignSpec, lay: dict, M: int, tau: float, c: int,
                        z: np.ndarray, lo: float, hi: float, out: np.ndarray) -> None:
    """Add integrals over [lo, hi) of news (tau, c, z) basis functions into out."""
    if spec.exo is not None and tau < hi:
        seg = spec.exo.cdf(np.array([hi - tau]))[0] - spec.exo.cdf(np.array([max(lo - tau, 0.0)]))[0]
        base = lay["off_exo"] + c * spec.exo.size * M
        block = (seg[:, None] * z[None, :]).reshape(-1)
        out[base:base + block.size] += block
    if spec.ant is not None and tau > lo:
        top = min(tau, hi)
        seg = np.exp(-spec.ant.betas * (tau - top)) - np.exp(-spec.ant.betas * (tau - lo))
        base = lay["off_ant"] + c * spec.ant.K
        out[base:base + spec.ant.K] += seg


def sparse_integrals(data: EventData, spec: DesignSpec, lay: dict) -> np.ndarray:
    out = np.zeros(lay["Ps"])
    M = data.n_marks
    for w in range(data.n_windows):
        lo, hi = data.t0[w], data.t1[w]
        out[w] += hi - lo
        if spec.n_tod:
            out[lay["off_tod"]:lay["off_tod"] + spec.n_tod] += _hat_integral(
                data.tod0[w], data.tod0[w] + (hi - lo), spec.n_tod)
        for e in range(data.news_ptr[w], data.news_ptr[w + 1]):
            _news_integral_rows(spec, lay, M, data.news_t[e], int(data.news_type[e]),
                                data.news_marks[e], lo, hi, out)
    return out


def sparse_cumulative_reference(data: EventData, spec: DesignSpec, lay: dict, theta_s: np.ndarray,
                      row_t: np.ndarray, row_w: np.ndarray) -> np.ndarray:
    """theta_s . int_{t0}^{t} x_sparse(u) du for each (row time t, window) — used for residuals."""
    out = np.zeros(len(row_t))
    M = data.n_marks
    buf = np.zeros(lay["Ps"])
    for r, (t, w) in enumerate(zip(row_t, row_w)):
        lo = data.t0[w]
        val = theta_s[w] * (t - lo)
        if spec.n_tod:
            val += theta_s[lay["off_tod"]:lay["off_tod"] + spec.n_tod] @ _hat_integral(
                data.tod0[w], data.tod0[w] + (t - lo), spec.n_tod)
        if data.news_ptr[w + 1] > data.news_ptr[w]:
            buf[:] = 0.0
            for e in range(data.news_ptr[w], data.news_ptr[w + 1]):
                _news_integral_rows(spec, lay, M, data.news_t[e], int(data.news_type[e]),
                                    data.news_marks[e], lo, t, buf)
            val += theta_s[lay["off_exo"]:] @ buf[lay["off_exo"]:]
        out[r] = val
    return out


def _tod_cumulative(tod0: float, t0: float, t: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact int_{t0}^{t} sum_m w_m hat_m(tod(u)) du for sorted-or-not t (vectorised).

    The seasonal rate is piecewise linear in u with kinks where the local clock crosses a
    knot; integrate exactly with the trapezoid rule on the knot grid and interpolate the
    quadratic antiderivative within each piece."""
    n_tod = len(weights)
    h = DAY / n_tod
    t = np.asarray(t, float)
    if len(t) == 0:
        return np.zeros(0)
    x_hi = tod0 + (t.max() - t0)
    k0, k1 = int(np.floor(tod0 / h)), int(np.floor(x_hi / h))
    xs = np.unique(np.concatenate([[tod0], h * np.arange(k0 + 1, k1 + 1), [x_hi]]))

    def rate(x):
        m0 = np.floor(x / h).astype(np.int64)
        f = x / h - m0
        return weights[m0 % n_tod] * (1 - f) + weights[(m0 + 1) % n_tod] * f

    rx = rate(xs)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rx[1:] + rx[:-1]) * np.diff(xs))])
    x = tod0 + (t - t0)
    k = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 1)
    return cum[k] + 0.5 * (x - xs[k]) * (rx[k] + rate(x))


def sparse_cumulative(data: EventData, spec: DesignSpec, lay: dict, theta_s: np.ndarray,
                      row_t: np.ndarray, row_w: np.ndarray) -> np.ndarray:
    """theta_s . int_{t0}^{t} x_sparse(u) du for each (row time t, window), vectorised per window."""
    out = np.zeros(len(row_t))
    M = data.n_marks
    order = np.argsort(row_w, kind="stable")
    bounds = np.searchsorted(row_w[order], np.arange(data.n_windows + 1))
    th_tod = theta_s[lay["off_tod"]:lay["off_tod"] + spec.n_tod] if spec.n_tod else None
    for w in range(data.n_windows):
        idx = order[bounds[w]:bounds[w + 1]]
        if len(idx) == 0:
            continue
        t = row_t[idx]
        lo = data.t0[w]
        val = theta_s[w] * (t - lo)
        if spec.n_tod:
            val = val + _tod_cumulative(data.tod0[w], lo, t, th_tod)
        for e in range(data.news_ptr[w], data.news_ptr[w + 1]):
            tau, c, z = data.news_t[e], int(data.news_type[e]), data.news_marks[e]
            if spec.exo is not None:
                base = lay["off_exo"] + c * spec.exo.size * M
                W = theta_s[base:base + spec.exo.size * M].reshape(spec.exo.size, M) @ z  # (K*R,)
                if np.any(W > 0):
                    after = t > tau
                    if after.any():
                        start = spec.exo.cdf(np.array([max(lo - tau, 0.0)]))[0] @ W
                        val[after] += spec.exo.cdf(t[after] - tau) @ W - start
            if spec.ant is not None and tau > lo:
                base = lay["off_ant"] + c * spec.ant.K
                a = theta_s[base:base + spec.ant.K]
                if np.any(a > 0):
                    top = np.minimum(tau, t)
                    val += (np.exp(-spec.ant.betas[None, :] * (tau - top)[:, None])
                            - np.exp(-spec.ant.betas * (tau - lo))[None, :]) @ a
        out[idx] = val
    return out
