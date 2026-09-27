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
def _advance(S1, S2, C1, C2, betas, R, dt, acc):
    d, K = S1.shape
    for k in range(K):
        b = betas[k]
        e = np.exp(-b * dt)
        one_m = (1.0 - e) / b
        g2 = (1.0 - e * (1.0 + b * dt)) / b
        for j in range(d):
            s1 = S1[j, k]
            if acc:
                C1[j, k] += s1 * one_m
                if R == 2:
                    C2[j, k] += S2[j, k] * one_m + s1 * g2
            if R == 2:
                S2[j, k] = e * (S2[j, k] + b * dt * s1)
            S1[j, k] = e * s1


@njit(cache=True)
def _advance_split(S1, S2, C1, C2, betas, R, tcur, tnext, t0):
    if tnext <= tcur:
        return
    if tnext <= t0:
        _advance(S1, S2, C1, C2, betas, R, tnext - tcur, False)
    elif tcur >= t0:
        _advance(S1, S2, C1, C2, betas, R, tnext - tcur, True)
    else:
        _advance(S1, S2, C1, C2, betas, R, t0 - tcur, False)
        _advance(S1, S2, C1, C2, betas, R, tnext - t0, True)


@njit(cache=True)
def _endo_features(times, types, wptr, t0, t1, d, betas, R, target, want_cum):
    K = betas.shape[0]
    P = d * K * R
    W = t0.shape[0]
    n = 0
    for w in range(W):
        for i in range(wptr[w], wptr[w + 1]):
            if times[i] >= t0[w] and times[i] < t1[w] and (target < 0 or types[i] == target):
                n += 1
    X = np.zeros((n, P), dtype=np.float32)
    Cm = np.zeros((n if want_cum else 0, P))
    rows = np.empty(n, dtype=np.int64)
    integ = np.zeros(P)
    S1 = np.zeros((d, K))
    S2 = np.zeros((d, K))
    C1 = np.zeros((d, K))
    C2 = np.zeros((d, K))
    r = 0
    for w in range(W):
        S1[:] = 0.0
        S2[:] = 0.0
        C1[:] = 0.0
        C2[:] = 0.0
        a = wptr[w]
        b = wptr[w + 1]
        tcur = t0[w]
        if b > a and times[a] < tcur:
            tcur = times[a]
        i = a
        while i < b:
            t = times[i]
            if t >= t1[w]:
                break
            _advance_split(S1, S2, C1, C2, betas, R, tcur, t, t0[w])
            tcur = t
            j = i
            while j < b and times[j] == t:
                j += 1
            if t >= t0[w]:
                for m in range(i, j):
                    if target < 0 or types[m] == target:
                        rows[r] = m
                        for src in range(d):
                            for k in range(K):
                                c = (src * K + k) * R
                                X[r, c] = S1[src, k]
                                if R == 2:
                                    X[r, c + 1] = S2[src, k]
                                if want_cum:
                                    Cm[r, c] = C1[src, k]
                                    if R == 2:
                                        Cm[r, c + 1] = C2[src, k]
                        r += 1
            for m in range(i, j):
                u = types[m]
                for k in range(K):
                    S1[u, k] += betas[k]
            i = j
        _advance_split(S1, S2, C1, C2, betas, R, tcur, t1[w], t0[w])
        for src in range(d):
            for k in range(K):
                c = (src * K + k) * R
                integ[c] += C1[src, k]
                if R == 2:
                    integ[c + 1] += C2[src, k]
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
                    ex = np.exp(-b * lag)
                    for rr in range(exo_R):
                        val = b * ex if rr == 0 else b * b * lag * ex
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


def sparse_cumulative(data: EventData, spec: DesignSpec, lay: dict, theta_s: np.ndarray,
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
