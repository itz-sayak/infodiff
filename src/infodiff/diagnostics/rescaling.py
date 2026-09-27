"""Goodness-of-fit via the time-rescaling theorem.

If lambda is the true conditional intensity, the compensator increments
tau_n = int_{t_{n-1}}^{t_n} lambda(s) ds are iid Exp(1).  We provide

* KS and Anderson–Darling statistics against Exp(1),
* the Excess-Dispersion (ED) test used by Rambaldi et al. (2015) — so rejection
  rates are directly comparable with the literature,
* a Ljung–Box test for serial dependence of the residuals,
* parametric-bootstrap calibration (naive KS p-values are invalid once parameters
  are estimated from the same data).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GofResult:
    n: int
    ks: float
    ks_p: float
    ad: float
    ed_z: float
    ed_p: float
    lb_p: float
    mean: float
    var: float


def anderson_darling_exp1(x: np.ndarray) -> float:
    """AD statistic against the fully specified Exp(1) (no parameter estimation)."""
    x = np.sort(np.asarray(x))
    n = len(x)
    F = np.clip(1.0 - np.exp(-x), 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(F) + np.log1p(-F[::-1]))))


def excess_dispersion(x: np.ndarray) -> tuple[float, float]:
    """ED test (Engle & Russell 1998): sqrt(n) (var - 1)/sqrt(8) ~ N(0,1) under Exp(1)."""
    n = len(x)
    z = np.sqrt(n) * (np.var(x) - 1.0) / np.sqrt(8.0)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def ljung_box_p(x: np.ndarray, lags: int = 20) -> float:
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    if n <= lags + 1:
        return np.nan
    den = np.dot(x, x)
    q = 0.0
    for k in range(1, lags + 1):
        rk = np.dot(x[:-k], x[k:]) / den
        q += rk * rk / (n - k)
    q *= n * (n + 2)
    return float(stats.chi2.sf(q, lags))


def gof(res: np.ndarray, lags: int = 20) -> GofResult:
    res = np.asarray(res, dtype=float)
    ks = stats.kstest(res, "expon")
    ed_z, ed_p = excess_dispersion(res)
    return GofResult(n=len(res), ks=float(ks.statistic), ks_p=float(ks.pvalue),
                     ad=anderson_darling_exp1(res), ed_z=ed_z, ed_p=ed_p,
                     lb_p=ljung_box_p(res, lags), mean=float(res.mean()), var=float(res.var()))


def qq_exp1(res: np.ndarray, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Theoretical vs empirical Exp(1) quantiles (log-spaced probabilities for tails)."""
    res = np.sort(np.asarray(res))
    p = np.unique(np.concatenate([np.geomspace(1e-5, 0.5, n_points // 2),
                                  1 - np.geomspace(1e-5, 0.5, n_points // 2)]))
    p = p[(p > 0) & (p < 1)]
    return -np.log1p(-p), np.quantile(res, p)


def bootstrap_pvalue(stat_obs: float, stat_boot: np.ndarray) -> float:
    """Upper-tail parametric-bootstrap p-value with the +1 correction."""
    stat_boot = np.asarray(stat_boot)
    return float((1 + np.sum(stat_boot >= stat_obs)) / (1 + len(stat_boot)))
