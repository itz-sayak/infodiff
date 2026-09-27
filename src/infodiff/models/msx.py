"""MSX-Hawkes: multiscale, state-space, exogenous-marked Hawkes estimator.

Per target dimension i the problem is

    maximise  L(theta) = sum_n log(x_n . theta) - (Phi + gamma) . theta,   theta >= 0,

a concave program (Poisson linear-inverse form).  We solve it with the
multiplicative EM / Richardson–Lucy iteration

    theta_p  <-  theta_p * (X^T (1/lambda))_p / (Phi_p + gamma_p),

accelerated by SQUAREM (Varadhan & Roland 2008) with a monotonicity safeguard, and
certify global optimality with the Fenchel dual bound

    L(theta)  <=  D(v) = -N - sum_n log v_n    for any v > 0 with X^T v <= Phi + gamma,

taking v = (1/lambda) / max_p[(X^T(1/lambda))_p / (Phi_p+gamma_p)].  The reported
`gap = D(v) - L(theta) >= 0` bounds the sub-optimality of the returned estimate.
Nonnegative L1 penalties are linear and are absorbed into Phi (gamma).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from ..events.data import EventData
from .features import Design, DesignSpec, build_design, sparse_cumulative


def _to_torch_csr(X: sp.csr_matrix, device, dtype):
    X = X.tocsr()
    return torch.sparse_csr_tensor(
        torch.as_tensor(X.indptr, dtype=torch.int64),
        torch.as_tensor(X.indices, dtype=torch.int64),
        torch.as_tensor(X.data, dtype=dtype),
        size=X.shape, device=device)


class _Problem:
    """Concave Poisson-linear problem on a device."""

    def __init__(self, design: Design, gamma_d: np.ndarray, gamma_s: np.ndarray, device: str, dtype):
        self.device, self.dtype = device, dtype
        self.Xd = torch.as_tensor(design.Xd, device=device, dtype=dtype)
        self.Xs = _to_torch_csr(design.Xs, device, dtype)
        self.XsT = _to_torch_csr(design.Xs.T.tocsr(), device, dtype)
        self.c = torch.as_tensor(np.concatenate([design.integ_d + gamma_d, design.integ_s + gamma_s]),
                                 device=device, dtype=torch.float64)
        self.Pd = design.Xd.shape[1]
        self.n = design.n

    def lam(self, th):
        thd = th[: self.Pd].to(self.dtype)
        ths = th[self.Pd:].to(self.dtype)
        out = torch.mv(self.Xs, ths)
        if self.Pd:
            out = out + torch.mv(self.Xd, thd)
        return out

    def xt(self, v):
        v = v.to(self.dtype)
        parts = []
        if self.Pd:
            parts.append(torch.mv(self.Xd.T, v).double())
        parts.append(torch.mv(self.XsT, v).double())
        return torch.cat(parts)

    def loglik(self, th, lam=None):
        lam = self.lam(th) if lam is None else lam
        return torch.log(lam.double()).sum() - torch.dot(self.c, th)

    def em_step(self, th):
        lam = self.lam(th)
        num = self.xt(1.0 / lam)
        return th * num / self.c, lam, num

    def dual_gap(self, th):
        lam = self.lam(th).double()
        num = self.xt(1.0 / lam)
        s = torch.max(num / self.c)
        # v = (1/lam)/s  ->  D(v) = -n - sum log v = -n + sum log lam + n log s
        primal = torch.log(lam).sum() - torch.dot(self.c, th)
        dual = -self.n + torch.log(lam).sum() + self.n * torch.log(s)
        return float(dual - primal), float(primal)


def _interior_point(prob: _Problem, Xs_sp: sp.csr_matrix, th: torch.Tensor, gap_tol: float,
                    max_newton: int = 200, verbose: bool = False) -> torch.Tensor:
    """Log-barrier Newton polish: max L(theta) + mu * sum log theta, mu -> 0.

    Hessian blocks: Xd^T D Xd (dense, device), Xd^T D Xs (sparse@dense, device),
    Xs^T D Xs (scipy sparse, CPU);  D = diag(1/lambda^2).  Quadratic convergence
    turns the slowly-converging EM iterate into a certified optimum."""
    P = th.numel()
    Pd = prob.Pd
    th = torch.clamp(th, min=1e-12 * float(th.max()) + 1e-300)
    lam = prob.lam(th).double()
    g0 = prob.xt(1.0 / lam) - prob.c
    mu = max(float(torch.mean(torch.abs(g0 * th))), 1e-10)

    def barrier_obj(t):
        l = prob.lam(t).double()
        if torch.any(l <= 0) or torch.any(t <= 0):
            return -np.inf
        return float(torch.log(l).sum() - torch.dot(prob.c, t) + mu * torch.log(t).sum())

    for it in range(max_newton):
        lam = prob.lam(th).double()
        inv = 1.0 / lam
        g = prob.xt(inv) - prob.c + mu / th
        Dv = (inv * inv)
        H = torch.zeros((P, P), dtype=torch.float64, device=th.device)
        if Pd:
            XdD = prob.Xd.double() * Dv[:, None]
            H[:Pd, :Pd] = prob.Xd.double().T @ XdD
            H[Pd:, :Pd] = torch.sparse.mm(prob.XsT.double() if prob.dtype != torch.float64 else prob.XsT, XdD)
            H[:Pd, Pd:] = H[Pd:, :Pd].T
        Dn = Dv.cpu().numpy()
        Hss = (Xs_sp.T @ Xs_sp.multiply(Dn[:, None])).toarray()
        H[Pd:, Pd:] = torch.as_tensor(Hss, device=th.device)
        H.diagonal().add_(mu / th ** 2)
        try:
            L = torch.linalg.cholesky(H)
            step = torch.cholesky_solve(g[:, None], L)[:, 0]
        except RuntimeError:
            H.diagonal().add_(1e-10 * float(H.diagonal().max()))
            step = torch.linalg.solve(H, g)
        dec2 = float(torch.dot(g, step))  # Newton decrement^2
        # fraction-to-boundary step length
        neg = step < 0
        a = 1.0
        if torch.any(neg):
            a = min(1.0, 0.99 * float(torch.min(-th[neg] / step[neg])))
        f0 = barrier_obj(th)
        while a > 1e-12:
            cand = th + a * step
            f1 = barrier_obj(cand)
            if f1 >= f0 + 1e-4 * a * dec2:
                break
            a *= 0.5
        th = th + a * step
        if dec2 < 1e-9:
            if mu * P < 0.1 * gap_tol:
                break
            mu *= 0.1
        if verbose and it % 10 == 0:
            print(f"  newton {it:3d} mu={mu:.1e} dec2={dec2:.2e} a={a:.2e}")
    return th


@dataclass
class FitReport:
    dim: int
    n_events: int
    loglik: float
    gap: float
    iters: int
    seconds: float
    expected_count: float  # sum_p theta_p Phi_p; equals n_events at the (unpenalised) optimum
    history: list = field(default_factory=list)


def _squarem(prob: _Problem, th0: torch.Tensor, max_iter: int, tol: float, verbose: bool):
    """SQUAREM-3 acceleration of the multiplicative EM map with monotone safeguard."""
    th = th0.clone()
    ll = float(prob.loglik(th))
    hist = [ll]
    it = 0
    for it in range(1, max_iter + 1):
        th1, _, _ = prob.em_step(th)
        th2, _, _ = prob.em_step(th1)
        r = th1 - th
        v = th2 - th1 - r
        vn = torch.linalg.vector_norm(v)
        if vn <= 1e-300:
            th_new = th2
        else:
            alpha = -torch.linalg.vector_norm(r) / vn
            alpha = torch.clamp(alpha, max=-1.0)
            th_new = th - 2 * alpha * r + alpha * alpha * v
            th_new = torch.clamp(th_new, min=0.0)
            # stabilising EM step from the extrapolated point
            ok = bool(torch.all(prob.lam(th_new) > 0))
            if ok:
                th_new, _, _ = prob.em_step(th_new)
                ok = float(prob.loglik(th_new)) >= float(prob.loglik(th2)) - 1e-9 * abs(ll)
            if not ok:
                th_new = th2
        ll_new = float(prob.loglik(th_new))
        hist.append(ll_new)
        rel = abs(ll_new - ll) / (abs(ll) + 1.0)
        th, ll = th_new, ll_new
        if verbose and it % 50 == 0:
            print(f"  iter {it:5d}  ll={ll:.6f}  rel={rel:.2e}")
        if rel < tol:
            break
    return th, hist, it


class MSXHawkes:
    """Multiscale phase-type Hawkes with exogenous marked news and seasonal baselines."""

    def __init__(self, spec: DesignSpec, l1_endo: float = 0.0, l1_exo: float = 0.0, l1_tod: float = 1e-3,
                 device: str | None = None, dtype=torch.float64, max_iter: int = 400, tol: float = 1e-9,
                 gap_tol: float = 1e-4, newton: bool = True, verbose: bool = False):
        self.newton = newton
        self.spec = spec
        self.l1_endo, self.l1_exo, self.l1_tod = l1_endo, l1_exo, l1_tod
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_iter, self.tol, self.gap_tol = max_iter, tol, gap_tol
        self.verbose = verbose
        self.theta_d: np.ndarray | None = None  # (d, Pd)
        self.theta_s: np.ndarray | None = None  # (d, Ps)
        self.reports: list[FitReport] = []
        self.layout: dict | None = None
        self.n_dims = 0

    # ------------------------------------------------------------------ fitting
    def _gammas(self, design: Design):
        """Exposure-weighted L1: gamma_p = l1 * Phi_p (unit-free; theta_p = 0 unless its
        score exceeds (1 + l1) times its exposure at the optimum)."""
        lay = design.layout
        gd = self.l1_endo * design.integ_d
        gs = np.zeros(lay["Ps"])
        gs[lay["off_tod"]:lay["off_exo"]] = self.l1_tod * design.integ_s[lay["off_tod"]:lay["off_exo"]]
        gs[lay["off_exo"]:] = self.l1_exo * design.integ_s[lay["off_exo"]:]
        return gd, gs

    def fit(self, data: EventData, dims: list[int] | None = None) -> "MSXHawkes":
        d = data.n_dims
        self.n_dims = d
        dims = list(range(d)) if dims is None else dims
        lay = self.spec.layout(d, data.n_windows, max(data.n_news_types, 1), data.n_marks)
        self.layout = lay
        self.theta_d = np.zeros((d, lay["Pd"]))
        self.theta_s = np.zeros((d, lay["Ps"]))
        self.reports = []
        for i in dims:
            t_start = time.time()
            des = build_design(data, self.spec, i)
            if des.n == 0:
                continue
            gd, gs = self._gammas(des)
            prob = _Problem(des, gd, gs, self.device, self.dtype)
            th0 = self._init_theta(des, data, i)
            th0 = torch.as_tensor(th0, device=self.device, dtype=torch.float64)
            th, hist, it = _squarem(prob, th0, self.max_iter, self.tol, self.verbose)
            gap, ll = prob.dual_gap(th)
            extra = 0
            if gap > self.gap_tol and self.newton:
                th_n = _interior_point(prob, des.Xs, th, self.gap_tol, verbose=self.verbose)
                gap_n, ll_n = prob.dual_gap(th_n)
                if ll_n >= ll - 1e-9:
                    th, gap, ll = th_n, gap_n, ll_n
                hist.append(ll)
            while gap > self.gap_tol and extra < 5000:  # EM fallback
                for _ in range(100):
                    th, _, _ = prob.em_step(th)
                extra += 100
                gap, ll = prob.dual_gap(th)
            thn = th.cpu().numpy()
            self.theta_d[i] = thn[: lay["Pd"]]
            self.theta_s[i] = thn[lay["Pd"]:]
            integ = np.concatenate([des.integ_d, des.integ_s])
            self.reports.append(FitReport(dim=i, n_events=des.n, loglik=ll, gap=gap, iters=it + extra,
                                          seconds=time.time() - t_start,
                                          expected_count=float(thn @ integ), history=hist))
            if self.verbose:
                print(f"dim {i}: n={des.n} ll={ll:.4f} gap={gap:.2e} iters={it + extra}")
            del prob
        return self

    def _init_theta(self, des: Design, data: EventData, i: int) -> np.ndarray:
        """Strictly positive start: baseline explains ~half the events, kernels share the rest."""
        lay = des.layout
        n = des.n
        th_d = np.full(lay["Pd"], 0.5 / max(lay["Pd"], 1)) if lay["Pd"] else np.zeros(0)
        th_s = np.zeros(lay["Ps"])
        cnt = np.bincount(data.window_ids()[des.rows], minlength=data.n_windows)
        T = np.maximum(data.t1 - data.t0, 1e-9)
        th_s[: data.n_windows] = np.maximum(0.5 * cnt / T, 1e-3 * n / T.sum())
        rest = slice(lay["off_tod"], lay["Ps"])
        m = lay["Ps"] - lay["off_tod"]
        if m:
            th_s[rest] = 1e-2 * n / max(des.integ_s[rest].sum(), 1e-9) / m + 1e-12
        return np.concatenate([th_d, th_s])

    # ------------------------------------------------------------------ accessors
    def endo_weights(self) -> np.ndarray:
        """(d_target, d_source, K, R) excitation weights."""
        e = self.spec.endo
        return self.theta_d.reshape(self.n_dims, self.n_dims, e.K, e.R)

    def branching_matrix(self) -> np.ndarray:
        return self.endo_weights().sum(axis=(2, 3))

    def exo_weights(self, n_types: int, n_marks: int) -> np.ndarray:
        """(d, C, Ke, Re, M) news-kernel weights."""
        lay, ex = self.layout, self.spec.exo
        blk = self.theta_s[:, lay["off_exo"]:lay["off_exo"] + lay["n_exo"]]
        return blk.reshape(self.n_dims, n_types, ex.K, ex.R, n_marks)

    def ant_weights(self, n_types: int) -> np.ndarray:
        lay = self.layout
        blk = self.theta_s[:, lay["off_ant"]:lay["off_ant"] + lay["n_ant"]]
        return blk.reshape(self.n_dims, n_types, self.spec.ant.K)

    # ------------------------------------------------------------------ evaluation
    def loglik(self, data: EventData, per_dim: bool = False):
        """Exact log-likelihood of (possibly new) data under the fitted parameters.

        Window intercepts are data-specific; for held-out windows call
        `transfer_baseline` first (or fit intercepts only via `fit_baseline_only`)."""
        out = np.zeros(data.n_dims)
        for i in range(data.n_dims):
            des = build_design(data, self.spec, i)
            lam = des.Xd.astype(np.float64) @ self.theta_d[i] + des.Xs @ self.theta_s[i]
            comp = des.integ_d @ self.theta_d[i] + des.integ_s @ self.theta_s[i]
            out[i] = np.log(lam).sum() - comp
        return out if per_dim else out.sum()

    def residuals(self, data: EventData, dim: int) -> np.ndarray:
        """Time-rescaled inter-event compensator increments for one dimension (Exp(1) under H0)."""
        des = build_design(data, self.spec, dim, want_cum=True)
        row_t = data.times[des.rows]
        row_w = data.window_ids()[des.rows]
        comp = des.cum_d @ self.theta_d[dim] + sparse_cumulative(
            data, self.spec, des.layout, self.theta_s[dim], row_t, row_w)
        res = np.empty_like(comp)
        first = np.r_[True, row_w[1:] != row_w[:-1]]
        res[first] = comp[first]
        res[~first] = np.diff(comp)[~first[1:]]
        return res

    def intensity_at(self, data: EventData, dim: int) -> tuple[np.ndarray, np.ndarray]:
        des = build_design(data, self.spec, dim)
        lam = des.Xd.astype(np.float64) @ self.theta_d[dim] + des.Xs @ self.theta_s[dim]
        return data.times[des.rows], lam
