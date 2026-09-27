"""Phase-type kernel dictionaries.

Each dictionary element is a normalised phase-type density (integrates to 1):

* order r+1 (Erlang-(r+1)):  g_r(t) = beta (beta t)^r exp(-beta t) / r!,   r = 0..R-1
  (order 1 is the exponential; order R has coefficient of variation 1/sqrt(R))

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
from scipy.special import gammainc, gammaln


@dataclass(frozen=True)
class PhaseTypeDictionary:
    betas: np.ndarray  # (K,) decay rates in 1/s
    orders: int = 2  # R: Erlang orders 1..R per rate (1 -> exponential only)

    def __post_init__(self):
        b = np.asarray(self.betas, dtype=np.float64)
        if b.ndim != 1 or np.any(b <= 0):
            raise ValueError("betas must be a 1-D array of positive rates")
        if not 1 <= int(self.orders) <= 12:
            raise ValueError("orders must be in 1..12")
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
        t = np.asarray(t, dtype=np.float64)[:, None, None]
        b = self.betas[None, :, None]
        r = np.arange(self.R)[None, None, :]
        bt = b * t
        with np.errstate(divide="ignore"):
            logv = np.log(b) + r * np.log(np.where(bt > 0, bt, 1.0)) - bt - gammaln(r + 1)
        v = np.exp(logv)
        v = np.where((bt == 0) & (r > 0), 0.0, v)
        return v.reshape(t.shape[0], -1)

    def cdf(self, t: np.ndarray) -> np.ndarray:
        """Integral of each basis density on [0, t] (regularised lower gamma) -> (len(t), K*R)."""
        t = np.maximum(np.asarray(t, dtype=np.float64), 0.0)[:, None, None]
        bt = self.betas[None, :, None] * t
        r = np.arange(self.R)[None, None, :]
        return gammainc(r + 1, bt).reshape(t.shape[0], -1)

    def mean_lags(self) -> np.ndarray:
        """Mean lag of each basis density ((r+1)/beta) -> (K*R,)."""
        return ((np.arange(self.R)[None, :] + 1) / self.betas[:, None]).reshape(-1)

    def kernel(self, weights: np.ndarray, t: np.ndarray) -> np.ndarray:
        """phi(t) for weights (..., K*R) -> (..., len(t))."""
        return np.asarray(weights) @ self.pdf(t).T
