"""Exact cluster (branching) simulation of phase-type Hawkes processes with news.

Immigrants come from (i) the baseline mu_i(t) (thinning), (ii) news-triggered
exogenous kernels and (iii) anticipatory pre-news kernels.  Every event of type j
then spawns Poisson(G_ij) children of type i with phase-type lags.  Offspring counts
and lags are sampled generation by generation (vectorised), which is exact and far
faster than Ogata thinning for near-critical processes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..events.data import EventData
from ..models.dictionary import PhaseTypeDictionary


@dataclass
class HawkesTruth:
    endo: PhaseTypeDictionary
    A: np.ndarray  # (d, d, K, R) weights; kernel_{ij}(t) = sum A_ijkr g_kr(t)
    mu: np.ndarray  # (d,) constant baseline, or (d, W) per-window baseline
    exo: PhaseTypeDictionary | None = None
    B: np.ndarray | None = None  # (d, C, Ke, Re, M)
    ant: PhaseTypeDictionary | None = None
    H: np.ndarray | None = None  # (d, C, Ka)
    tod_w: np.ndarray | None = None  # (d, n_tod) seasonal hat weights (additive)

    @property
    def d(self) -> int:
        return self.A.shape[0]


def _sample_lags(rng, dic: PhaseTypeDictionary, w: np.ndarray, n: int) -> np.ndarray:
    """n lags from the mixture with (unnormalised) weights w over (k, r)."""
    p = w.reshape(-1) / w.sum()
    comp = rng.choice(len(p), size=n, p=p)
    k, r = np.divmod(comp, dic.R)
    return rng.gamma(shape=r + 1.0, scale=1.0 / dic.betas[k])


def _baseline_events(rng, truth: HawkesTruth, w: int, lo: float, hi: float, tod0: float, t0: float):
    d = truth.d
    mu = truth.mu[:, w] if truth.mu.ndim == 2 else truth.mu
    ts, us = [], []
    for i in range(d):
        if truth.tod_w is None:
            n = rng.poisson(mu[i] * (hi - lo))
            ts.append(rng.uniform(lo, hi, n))
            us.append(np.full(n, i))
            continue
        n_tod = truth.tod_w.shape[1]
        h = 86400.0 / n_tod

        def rate(t):
            x = (tod0 + (t - t0)) % 86400.0
            m0 = np.floor(x / h).astype(int)
            f = x / h - m0
            return mu[i] + truth.tod_w[i, m0 % n_tod] * (1 - f) + truth.tod_w[i, (m0 + 1) % n_tod] * f

        lmax = mu[i] + truth.tod_w[i].max()
        n = rng.poisson(lmax * (hi - lo))
        cand = rng.uniform(lo, hi, n)
        keep = rng.uniform(0, lmax, n) < rate(cand)
        ts.append(cand[keep])
        us.append(np.full(keep.sum(), i))
    return np.concatenate(ts), np.concatenate(us)


def simulate(truth: HawkesTruth, t0: np.ndarray, t1: np.ndarray, burn: float = 0.0,
             news: list[list[tuple[float, int, np.ndarray]]] | None = None,
             tod0: np.ndarray | None = None, seed: int = 0, max_events: int = 50_000_000) -> EventData:
    """Simulate independent windows [t0 - burn, t1); likelihood interval is [t0, t1)."""
    rng = np.random.default_rng(seed)
    d = truth.d
    W = len(t0)
    tod0 = np.zeros(W) if tod0 is None else tod0
    G = truth.A.sum(axis=(2, 3))
    all_t, all_u, ptr = [], [], [0]
    nptr, nt, nty, nm = [0], [], [], []
    total = 0
    for w in range(W):
        lo, hi = t0[w] - burn, t1[w]
        ts, us = _baseline_events(rng, truth, w, lo, hi, tod0[w] - burn, lo)
        gen_t, gen_u = [ts], [us]
        wnews = news[w] if news is not None else []
        for (tau, c, z) in wnews:
            for i in range(d):
                if truth.B is not None:
                    wts = np.tensordot(truth.B[i, c], z, axes=([-1], [0]))  # (Ke, Re)
                    tot = wts.sum()
                    if tot > 0:
                        n = rng.poisson(tot)
                        lag = _sample_lags(rng, truth.exo, wts, n)
                        gen_t.append(tau + lag)
                        gen_u.append(np.full(n, i))
                if truth.H is not None:
                    wts = truth.H[i, c]
                    tot = wts.sum()
                    if tot > 0:
                        n = rng.poisson(tot)
                        k = rng.choice(len(wts), size=n, p=wts / tot)
                        gen_t.append(tau - rng.exponential(1.0 / truth.ant.betas[k]))
                        gen_u.append(np.full(n, i))
            nt.append(tau)
            nty.append(c)
            nm.append(np.asarray(z, dtype=float))
        cur_t = np.concatenate(gen_t)
        cur_u = np.concatenate(gen_u).astype(np.int64)
        m = (cur_t >= lo) & (cur_t < hi)
        cur_t, cur_u = cur_t[m], cur_u[m]
        out_t, out_u = [cur_t], [cur_u]
        while len(cur_t):
            nxt_t, nxt_u = [], []
            for j in range(d):
                pj = cur_t[cur_u == j]
                if not len(pj):
                    continue
                for i in range(d):
                    if G[i, j] <= 0:
                        continue
                    nk = rng.poisson(G[i, j], size=len(pj))
                    n = nk.sum()
                    if n == 0:
                        continue
                    lag = _sample_lags(rng, truth.endo, truth.A[i, j], n)
                    ct = np.repeat(pj, nk) + lag
                    keep = ct < hi
                    nxt_t.append(ct[keep])
                    nxt_u.append(np.full(keep.sum(), i))
            if not nxt_t:
                break
            cur_t = np.concatenate(nxt_t)
            cur_u = np.concatenate(nxt_u).astype(np.int64)
            out_t.append(cur_t)
            out_u.append(cur_u)
            total += len(cur_t)
            if total > max_events:
                raise RuntimeError("simulation exploded; is the process stable?")
        tt = np.concatenate(out_t)
        uu = np.concatenate(out_u)
        o = np.argsort(tt, kind="stable")
        all_t.append(tt[o])
        all_u.append(uu[o])
        ptr.append(ptr[-1] + len(tt))
        nptr.append(nptr[-1] + len(wnews))
    M = len(nm[0]) if nm else 1
    C = truth.B.shape[1] if truth.B is not None else (truth.H.shape[1] if truth.H is not None else 0)
    return EventData(times=np.concatenate(all_t), types=np.concatenate(all_u), wptr=np.asarray(ptr),
                     t0=np.asarray(t0, float), t1=np.asarray(t1, float), n_dims=d, tod0=tod0,
                     news_ptr=np.asarray(nptr), news_t=np.asarray(nt, float),
                     news_type=np.asarray(nty, np.int32),
                     news_marks=np.asarray(nm, float).reshape(-1, M), n_news_types=C)
