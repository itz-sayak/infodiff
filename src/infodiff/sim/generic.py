"""Cluster simulation for Hawkes processes with arbitrary (non phase-type) kernels.

Used by the synthetic benchmark so that estimators are also tested *outside* their
model class: true power-law (Lomax) kernels as reported for FX/futures, gamma humps,
and multiscale mixtures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..events.data import EventData


@dataclass
class Shape:
    """Normalised lag density (integrates to 1)."""
    kind: str  # "exp" | "lomax" | "gamma"
    a: float  # exp: rate; lomax: tau0; gamma: shape
    b: float = 0.0  # lomax: p (tail exponent, density ~ t^-p); gamma: rate

    def pdf(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, float)
        if self.kind == "exp":
            return self.a * np.exp(-self.a * t)
        if self.kind == "lomax":
            return (self.b - 1) / self.a * (1 + t / self.a) ** (-self.b)
        return stats.gamma.pdf(t, self.a, scale=1 / self.b)

    def sample(self, rng, n: int) -> np.ndarray:
        if self.kind == "exp":
            return rng.exponential(1 / self.a, n)
        if self.kind == "lomax":
            u = rng.random(n)
            return self.a * ((1 - u) ** (-1 / (self.b - 1)) - 1)
        return rng.gamma(self.a, 1 / self.b, n)


@dataclass
class GenericKernel:
    weights: list[float]
    shapes: list[Shape]

    @property
    def norm(self) -> float:
        return float(sum(self.weights))

    def pdf(self, t) -> np.ndarray:
        return sum(w * s.pdf(t) for w, s in zip(self.weights, self.shapes))

    def sample(self, rng, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros(0)
        p = np.asarray(self.weights) / self.norm
        comp = rng.choice(len(p), size=n, p=p)
        out = np.empty(n)
        for c in range(len(p)):
            m = comp == c
            out[m] = self.shapes[c].sample(rng, m.sum())
        return out


@dataclass
class GenericHawkes:
    mu: np.ndarray  # (d,)
    kernels: dict  # (i, j) -> GenericKernel  (effect of source j on target i)

    @property
    def d(self) -> int:
        return len(self.mu)

    def branching(self) -> np.ndarray:
        G = np.zeros((self.d, self.d))
        for (i, j), k in self.kernels.items():
            G[i, j] = k.norm
        return G

    def kernel(self, i: int, j: int, t: np.ndarray) -> np.ndarray:
        k = self.kernels.get((i, j))
        return np.zeros_like(np.asarray(t, float)) if k is None else k.pdf(t)


def simulate_generic(model: GenericHawkes, n_windows: int, length: float, seed: int = 0) -> EventData:
    rng = np.random.default_rng(seed)
    d = model.d
    G = model.branching()
    if np.max(np.abs(np.linalg.eigvals(G))) >= 1:
        raise ValueError("unstable process")
    all_t, all_u, ptr = [], [], [0]
    for w in range(n_windows):
        n0 = rng.poisson(model.mu * length)
        cur_t = np.concatenate([rng.uniform(0, length, n) for n in n0])
        cur_u = np.concatenate([np.full(n, i) for i, n in enumerate(n0)])
        out_t, out_u = [cur_t], [cur_u]
        while len(cur_t):
            nt, nu = [], []
            for (i, j), k in model.kernels.items():
                par = cur_t[cur_u == j]
                if not len(par):
                    continue
                nk = rng.poisson(k.norm, len(par))
                ct = np.repeat(par, nk) + k.sample(rng, nk.sum())
                ct = ct[ct < length]
                nt.append(ct)
                nu.append(np.full(len(ct), i))
            if not nt:
                break
            cur_t, cur_u = np.concatenate(nt), np.concatenate(nu)
            out_t.append(cur_t)
            out_u.append(cur_u)
        t = np.concatenate(out_t)
        u = np.concatenate(out_u)
        o = np.argsort(t, kind="stable")
        all_t.append(t[o] + w * length * 2)
        all_u.append(u[o])
        ptr.append(ptr[-1] + len(t))
    starts = np.arange(n_windows) * length * 2
    return EventData(times=np.concatenate(all_t), types=np.concatenate(all_u), wptr=np.asarray(ptr),
                     t0=starts, t1=starts + length, n_dims=d)
