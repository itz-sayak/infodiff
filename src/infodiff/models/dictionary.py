"""Phase-type kernel dictionaries.

Each dictionary element is a normalised phase-type density (integrates to 1):

* order 1 (exponential):  g(t) = beta * exp(-beta t)
* order 2 (Erlang-2):     g(t) = beta^2 * t * exp(-beta t)

A kernel is a nonnegative combination  phi(t) = sum_{k,r} a_{kr} g_{kr}(t),  so the
branching ratio is simply sum_{k,r} a_{kr}.  Exponential mixtures approximate any
completely monotone kernel (Bernstein); adding Erlang-2 phases lets the kernel vanish
at the origin (hump shapes, as found empirically for FX quote dynamics), and
phase-type densities are dense in the densities on R_+.  Every element admits a
finite-dimensional Markov (state-space) realisation, which the likelihood recursion
and the echo-corrected impulse responses exploit.

Column order everywhere is (k, r) with r varying fastest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseTypeDictionary:
    betas: np.ndarray  # (K,) decay rates in 1/s
    orders: int = 2  # R: 1 -> exponential only, 2 -> exponential + Erlang-2

    def __post_init__(self):
        b = np.asarray(self.betas, dtype=np.float64)
        if b.ndim != 1 or np.any(b <= 0):
            raise ValueError("betas must be a 1-D array of positive rates")
        if self.orders not in (1, 2):
            raise ValueError("orders must be 1 or 2")
        object.__setattr__(self, "betas", b)

    @classmethod
    def log_grid(cls, tau_min: float, tau_max: float, n: int, orders: int = 2) -> "PhaseTypeDictionary":
        """Time scales 1/beta log-spaced between tau_min and tau_max seconds."""
        return cls(betas=1.0 / np.geomspace(tau_min, tau_max, n), orders=orders)

    @property
    def K(self) -> int:
        return len(self.betas)

    @property
    def R(self) -> int:
        return self.orders

    @property
    def size(self) -> int:
        return self.K * self.R

    def pdf(self, t: np.ndarray) -> np.ndarray:
        """Basis densities at lags t >= 0 -> (len(t), K*R)."""
        t = np.asarray(t, dtype=np.float64)[:, None]
        b = self.betas[None, :]
        e = np.exp(-b * t)
        out = [b * e]
        if self.R == 2:
            out.append(b * b * t * e)
        return np.stack(out, axis=-1).reshape(len(t), -1)

    def cdf(self, t: np.ndarray) -> np.ndarray:
        """Integral of each basis density on [0, t] -> (len(t), K*R)."""
        t = np.maximum(np.asarray(t, dtype=np.float64), 0.0)[:, None]
        bt = self.betas[None, :] * t
        e = np.exp(-bt)
        out = [1.0 - e]
        if self.R == 2:
            out.append(1.0 - e * (1.0 + bt))
        return np.stack(out, axis=-1).reshape(len(t), -1)

    def mean_lags(self) -> np.ndarray:
        """Mean lag of each basis density (1/beta and 2/beta) -> (K*R,)."""
        m = [1.0 / self.betas]
        if self.R == 2:
            m.append(2.0 / self.betas)
        return np.stack(m, axis=-1).reshape(-1)

    def kernel(self, weights: np.ndarray, t: np.ndarray) -> np.ndarray:
        """phi(t) for weights (..., K*R) -> (..., len(t))."""
        return np.asarray(weights) @ self.pdf(t).T
