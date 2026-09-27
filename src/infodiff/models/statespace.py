"""Markov embedding of phase-type Hawkes processes and echo-corrected news responses.

States: for each source j and dictionary element (k, r)
    order 1:  x' = -beta x + beta * dN_j            (x = sum beta e^{-beta(t-t_n)})
    order 2:  x2' = -beta x2 + beta x1              (x2 = sum beta^2 (t-t_n) e^{-beta(t-t_n)})
Taking expectations, the *excess* mean intensity r(t) = E[lambda(t)] - baseline caused
by a news shock obeys the linear system

    z' = A z,   z = [x; y],   A = [[F + Gin Theta, Gin Theta_e], [0, F_e]],
    r(t) = C z(t),  C = [Theta, Theta_e],  z(0) = [0; y0],

where y are the exogenous (news) states.  Hence the full impulse response including
every generation of endogenous echoes is r(t) = C e^{At} z0, its integral is
-C A^{-1} z0, and absorption quantiles follow in closed form.

Theorem (stability).  F + Gin Theta is Metzler; with F Hurwitz-Metzler and Gin Theta >= 0
(a regular splitting), F + Gin Theta is Hurwitz  <=>  rho(-F^{-1} Gin Theta) < 1, and
rho(-F^{-1} Gin Theta) = rho(Theta (-F)^{-1} Gin) = rho(G) with G_ij = sum_{k,r} A_ijkr
the branching matrix (each basis density integrates to one).  See tests for a numerical
check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve

from .dictionary import PhaseTypeDictionary


def _blocks(dic: PhaseTypeDictionary, n_src: int):
    K, R = dic.K, dic.R
    n = n_src * K * R
    F = np.zeros((n, n))
    Gin = np.zeros((n, n_src))
    for j in range(n_src):
        for k in range(K):
            b = dic.betas[k]
            i1 = (j * K + k) * R
            Gin[i1, j] = b
            for r in range(R):
                F[i1 + r, i1 + r] = -b
                if r > 0:
                    F[i1 + r, i1 + r - 1] = b
    return F, Gin


@dataclass
class NewsResponse:
    t: np.ndarray  # (T,) time grid (s after news)
    total: np.ndarray  # (T, d) excess intensity incl. echoes
    direct: np.ndarray  # (T, d) direct news kernel only
    cum_total: np.ndarray  # (T, d)
    cum_direct: np.ndarray  # (T, d)
    int_total: np.ndarray  # (d,) expected extra events per news
    int_direct: np.ndarray  # (d,)


class StateSpaceHawkes:
    def __init__(self, endo: PhaseTypeDictionary, A: np.ndarray):
        """A: (d, d, K, R) endogenous weights (target, source, k, r)."""
        self.endo = endo
        self.d = A.shape[0]
        self.A = A
        F, Gin = _blocks(endo, self.d)
        self.F, self.Gin = F, Gin
        self.Theta = A.reshape(self.d, -1)
        self.M = F + Gin @ self.Theta

    # -------------------------------------------------------------- stability
    def branching(self) -> np.ndarray:
        return self.A.sum(axis=(2, 3))

    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.branching()))))

    def spectral_abscissa(self) -> float:
        return float(np.max(np.linalg.eigvals(self.M).real))

    def relaxation_time(self) -> float:
        """Slowest mode of the echo dynamics: 1 / -max Re eig(F + Gin Theta)."""
        a = self.spectral_abscissa()
        return np.inf if a >= 0 else -1.0 / a

    def endogeneity(self, baseline: np.ndarray) -> np.ndarray:
        """Stationary fraction of events that are endogenous per dimension, given baseline rates."""
        G = self.branching()
        Lam = solve(np.eye(self.d) - G, baseline)
        return 1.0 - baseline / Lam

    # -------------------------------------------------------------- news response
    def _system(self, exo: PhaseTypeDictionary, W: np.ndarray):
        """W: (d, Ke, Re) effective news-kernel weights for one news (marks applied)."""
        Fe, _ = _blocks(exo, 1)
        Th_e = W.reshape(self.d, -1)
        nx, ny = self.M.shape[0], Fe.shape[0]
        A = np.zeros((nx + ny, nx + ny))
        A[:nx, :nx] = self.M
        A[:nx, nx:] = self.Gin @ Th_e
        A[nx:, nx:] = Fe
        C = np.hstack([self.Theta, Th_e])
        y0 = np.zeros(ny)
        y0[::exo.R] = exo.betas
        z0 = np.concatenate([np.zeros(nx), y0])
        return A, C, z0, Fe, Th_e, y0

    def news_response(self, exo: PhaseTypeDictionary, W: np.ndarray, t: np.ndarray) -> NewsResponse:
        A, C, z0, Fe, Th_e, y0 = self._system(exo, W)
        Ainv_z0 = solve(A, z0)
        Feinv_y0 = solve(Fe, y0)
        tot, dirc, ctot, cdir = [], [], [], []
        for s in t:
            ez = expm(A * s) @ z0
            ey = expm(Fe * s) @ y0
            tot.append(C @ ez)
            dirc.append(Th_e @ ey)
            ctot.append(C @ (solve(A, ez) - Ainv_z0))
            cdir.append(Th_e @ (solve(Fe, ey) - Feinv_y0))
        return NewsResponse(t=np.asarray(t), total=np.array(tot), direct=np.array(dirc),
                            cum_total=np.array(ctot), cum_direct=np.array(cdir),
                            int_total=-C @ Ainv_z0, int_direct=-Th_e @ Feinv_y0)

    def absorption_time(self, exo: PhaseTypeDictionary, W: np.ndarray, q: float = 0.5,
                        dims: np.ndarray | None = None, signs: np.ndarray | None = None,
                        direct: bool = False, t_lo: float = 1e-5, t_hi: float = 1e6) -> float:
        """Time by which a fraction q of the (echo-corrected unless direct=True) response
        has arrived, aggregated over `dims` with optional `signs` (e.g. +up/-down drift)."""
        A, C, z0, Fe, Th_e, y0 = self._system(exo, W)
        if direct:
            A, C, z0 = Fe, Th_e, y0
        dims = np.arange(self.d) if dims is None else np.asarray(dims)
        s = np.ones(len(dims)) if signs is None else np.asarray(signs, float)
        c = s @ C[dims]
        Ainv_z0 = solve(A, z0)
        total = -c @ Ainv_z0
        if abs(total) < 1e-300:
            return np.nan

        def frac(tt):
            return (c @ (solve(A, expm(A * tt) @ z0) - Ainv_z0)) / total

        lo, hi = np.log(t_lo), np.log(t_hi)
        if frac(np.exp(hi)) < q:
            return np.inf
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if frac(np.exp(mid)) < q:
                lo = mid
            else:
                hi = mid
        return float(np.exp(0.5 * (lo + hi)))

    def response_curves(self, exo: PhaseTypeDictionary, W: np.ndarray, t_grid: np.ndarray) -> NewsResponse:
        """Echo-corrected and direct responses on an increasing grid.

        Marches the augmented generator M = [[A, z0], [0, 0]] along the grid with exact
        dense matrix exponentials of the step lengths; the last column of the running
        state is Z(t) = int_0^t e^{As} z0 ds, so r(t) = C (A Z(t) + z0) and cum(t) = C Z(t)."""
        A, C, z0, Fe, Th_e, y0 = self._system(exo, W)

        def curves(Am, Cm, v0):
            n = Am.shape[0]
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = Am
            M[:n, n] = v0
            state = np.zeros(n + 1)
            state[n] = 1.0
            Z = np.zeros((len(t_grid), n))
            t_prev = 0.0
            for k, t in enumerate(t_grid):
                state = expm(M * (t - t_prev)) @ state
                Z[k] = state[:n]
                t_prev = t
            return (Z @ Am.T + v0) @ Cm.T, Z @ Cm.T, -Cm @ solve(Am, v0)

        tot, ctot, itot = curves(A, C, z0)
        dirc, cdir, idir = curves(Fe, Th_e, y0)
        return NewsResponse(t=np.asarray(t_grid), total=tot, direct=dirc, cum_total=ctot, cum_direct=cdir,
                            int_total=itot, int_direct=idir)

    def refine_quantile(self, exo: PhaseTypeDictionary, W: np.ndarray, dims: np.ndarray, q: float,
                        t_guess: float, direct: bool = False, iters: int = 3, width: float = 1.25) -> float:
        """Polish a grid quantile with a few exact bisection steps in log t."""
        if not np.isfinite(t_guess):
            return t_guess
        A, C, z0, Fe, Th_e, y0 = self._system(exo, W)
        if direct:
            A, C, z0 = Fe, Th_e, y0
        c = C[np.asarray(dims)].sum(0)
        Ainv_z0 = solve(A, z0)
        total = -c @ Ainv_z0

        def frac(tt):
            return (c @ (solve(A, expm(A * tt) @ z0) - Ainv_z0)) / total

        lo, hi = np.log(t_guess / width), np.log(t_guess * width)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if frac(np.exp(mid)) < q:
                lo = mid
            else:
                hi = mid
        return float(np.exp(0.5 * (lo + hi)))

    @staticmethod
    def quantile_from_curve(t_grid: np.ndarray, cum: np.ndarray, total: float, q: float) -> float:
        """t with cum(t)/total = q by log-linear interpolation on the grid (inf if not reached)."""
        if not np.isfinite(total) or total <= 0:
            return np.nan
        f = cum / total
        k = np.searchsorted(f, q)
        if k >= len(f):
            return np.inf
        if k == 0:
            return float(t_grid[0])
        f0, f1 = f[k - 1], f[k]
        w = (q - f0) / max(f1 - f0, 1e-300)
        return float(np.exp(np.log(t_grid[k - 1]) + w * (np.log(t_grid[k]) - np.log(t_grid[k - 1]))))

    def amplification(self, exo: PhaseTypeDictionary, W: np.ndarray) -> np.ndarray:
        """Total / direct expected extra events per dimension (echo multiplier)."""
        A, C, z0, Fe, Th_e, y0 = self._system(exo, W)
        tot = -C @ solve(A, z0)
        dirc = -Th_e @ solve(Fe, y0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return tot / dirc
