"""EPT-TPP: an Exact-likelihood neural Phase-Type Temporal Point Process.

State.  A nonnegative latent vector x(t) in R^P_{>=0} made of phase-type channels:
for channel c and learnable rate beta_k
        x1' = -beta_k x1,           x2' = -beta_k x2 + beta_k x1     (Erlang-2 phase)
between events.  At event n (mark m_n, gap dt_n) a GRU summary h_n drives a gated,
nonnegative jump
        x(t_n+) = sigma(W_g h_n) * x(t_n-) + softplus(W_j h_n),
and the marked intensities are a nonnegative read-out plus two bounded hazard terms
        lambda_m(t) = mu_m(h_n) + sum_p C_mp(h_n) x_p(t)
                      + g1_m(h_n) e^{-a(h_n) s} + g2_m(h_n) (1 - e^{-kappa(h_n) s}),  s = t - t_n,
with C, mu, g1, g2, a, kappa >= 0: history-dependent decaying hazards (RMTPP with
w <= 0) and rising-but-saturating hazards (an unbounded rising Gompertz hazard overflows
and is ill-posed for heavy-tailed gaps).
Consequences
  * lambda >= mu > 0 always; the compensator between events is available in closed
    form (no Monte-Carlo integral, unlike NHP/THP/AttNHP/S2P2), so training and
    evaluation use the *exact* log-likelihood;
  * with an identity GRU/gate, linear jumps and g = 0 the model is exactly the MSX
    multivariate phase-type Hawkes process; with C = 0 it is RMTPP (Du et al. 2016),
    with C = 0 and g2 = 0 it is RMTPP restricted to decaying hazards, while the compensator
    stays exact (int g1 e^{-a s} = g1 (1 - e^{-a dt}) / a,
    int g2 (1 - e^{-k s}) = g2 (dt - (1 - e^{-k dt}) / k));
  * next-event time and mark predictions are Bayes-optimal functionals of the exact
    density, evaluated by 1-D quadrature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EPTConfig:
    n_marks: int
    hidden: int = 64
    n_rates: int = 8
    n_channels: int = 4
    phases: int = 2  # Erlang chain length R (CV = 1/sqrt(R)); 1 = exponential
    tau_min: float = 1e-2  # initial time-scale range (overridden from data)
    tau_max: float = 1e2
    dropout: float = 0.0
    mark_emb: int = 32
    gompertz: bool = True
    mark_mixing: bool = True  # C_mp(h) = c_p softmax_m(W_p h): history-dependent marks


class EPTTPP(nn.Module):
    def __init__(self, cfg: EPTConfig):
        super().__init__()
        self.cfg = cfg
        K, Cn = cfg.n_rates, cfg.n_channels
        self.R = cfg.phases
        self.P = Cn * K * self.R
        taus = torch.logspace(math.log10(cfg.tau_min), math.log10(cfg.tau_max), K)
        # one learnable rate per (channel, k); Erlang-R mean lag = R / beta
        self.log_beta = nn.Parameter(torch.log(self.R / taus).repeat(Cn) + 0.05 * torch.randn(Cn * K))
        self.emb = nn.Embedding(cfg.n_marks + 1, cfg.mark_emb)  # last index = BOS
        self.gru = nn.GRUCell(cfg.mark_emb + 2, cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)
        self.jump = nn.Linear(cfg.hidden, self.P)
        self.gate = nn.Linear(cfg.hidden, self.P)
        self.mu = nn.Linear(cfg.hidden, cfg.n_marks)
        if cfg.gompertz:
            self.g_head = nn.Linear(cfg.hidden, 2 * cfg.n_marks)
            self.w_head = nn.Linear(cfg.hidden, 2)
            nn.init.constant_(self.g_head.bias, -4.0)
            nn.init.zeros_(self.w_head.weight)
            nn.init.zeros_(self.w_head.bias)
        self.C_raw = nn.Parameter(torch.randn(cfg.n_marks, self.P) * 0.1 - 2.0)
        if cfg.mark_mixing:
            self.c_tot = nn.Parameter(torch.full((self.P,), -2.0))
            self.mix = nn.Linear(cfg.hidden, self.P * cfg.n_marks)
            nn.init.zeros_(self.mix.weight)
        nn.init.constant_(self.gate.bias, 2.0)  # start close to additive (Hawkes-like) updates
        nn.init.constant_(self.jump.bias, -1.0)
        nn.init.constant_(self.mu.bias, -2.0)
        self.t_scale = float(math.sqrt(cfg.tau_min * cfg.tau_max))  # input normalisation

    # ------------------------------------------------------------ phase-type algebra
    def _betas(self):
        return torch.exp(self.log_beta)  # (Cn*K,)

    def _split(self, x):
        return x.view(*x.shape[:-1], -1, self.R)

    def evolve(self, x, dt):
        """Exact Erlang-chain evolution and integral.

        Per rate block s' = beta (N - I) s (N = down-shift), hence
          s_r(dt)       = sum_{j<r} w_j s_{r-j}(0),     w_j = e^{-y} y^j / j!,  y = beta dt
          int_0^dt s_r  = sum_{j<r} s_{r-j}(0) P(j+1, y) / beta
        with P the regularised lower incomplete gamma, P(j+1, y) = 1 - sum_{i<=j} w_i."""
        b = self._betas()
        xs = self._split(x)  # (..., B, R)
        y = b * dt.unsqueeze(-1)  # (..., B)
        R = self.R
        logy = torch.log(y.clamp_min(1e-30))
        j = torch.arange(R, device=x.device, dtype=x.dtype)
        logw = -y.unsqueeze(-1) + j * logy.unsqueeze(-1) - torch.lgamma(j + 1)
        w = torch.exp(logw)  # (..., B, R)
        if R > 1:
            w = torch.where(y.unsqueeze(-1) > 0, w, (j == 0).to(x.dtype).expand_as(w))
        Pg = (1.0 - torch.cumsum(w, -1)).clamp_min(0.0)  # P(j+1, y)
        new = torch.zeros_like(xs)
        integ = torch.zeros_like(xs)
        for r in range(R):
            src = xs[..., : r + 1].flip(-1)  # s_r, s_{r-1}, ..., s_0  (lags j = 0..r)
            new[..., r] = (w[..., : r + 1] * src).sum(-1)
            integ[..., r] = (Pg[..., : r + 1] * src).sum(-1)
        integ = integ / b.unsqueeze(-1)
        return new.flatten(-2), integ.flatten(-2)

    def C(self, h=None):
        """Mark read-out (M, P), or (B, M, P) when marks mix with the history h."""
        if not self.cfg.mark_mixing:
            return F.softplus(self.C_raw)
        M, P = self.cfg.n_marks, self.P
        logits = self.mix(h).view(-1, P, M) + self.C_raw.T[None]
        pi = torch.softmax(logits, dim=-1)  # (B, P, M): mark distribution of each component
        return (F.softplus(self.c_tot)[None, :, None] * pi).transpose(1, 2)  # (B, M, P)

    def _readout(self, Cm, x):
        """lambda contribution C x for Cm (M,P) or (B,M,P) and x (B,P) or (B,G,P)."""
        if Cm.dim() == 2:
            return x @ Cm.T
        if x.dim() == 2:
            return torch.einsum("bmp,bp->bm", Cm, x)
        return torch.einsum("bmp,bgp->bgm", Cm, x)

    def _gomp(self, h):
        """Hazard-head parameters packed as (g1, g2) (B, M) each and rates (a, kappa) (B,)."""
        if not self.cfg.gompertz:
            return None, None
        M = self.cfg.n_marks
        gg = F.softplus(self.g_head(h))
        rates = F.softplus(self.w_head(h)) / self.t_scale  # (B, 2) >= 0
        return torch.cat([gg[..., :M], gg[..., M:]], -1), rates

    def _gomp_terms(self, g, w, dt):
        """Values and integrals over [0, dt] of g1 e^{-a s} + g2 (1 - e^{-k s})."""
        M = self.cfg.n_marks
        g1, g2 = g[..., :M], g[..., M:]
        a, k = w[..., 0], w[..., 1]
        ea = torch.exp(-a * dt)
        ek = torch.exp(-k * dt)
        val = g1 * ea.unsqueeze(-1) + g2 * (1 - ek).unsqueeze(-1)
        ia = torch.where(a * dt < 1e-6, dt, -torch.expm1(-a * dt) / a.clamp_min(1e-30))
        ik = torch.where(k * dt < 1e-6, 0.5 * k * dt * dt, dt + torch.expm1(-k * dt) / k.clamp_min(1e-30))
        integ = g1 * ia.unsqueeze(-1) + g2 * ik.unsqueeze(-1)
        return val, integ

    # ------------------------------------------------------------ forward pass
    def forward(self, dts, marks, mask):
        """dts, marks, mask: (B, L); dts[:,0] ignored (first event is conditioning).

        Returns per-position quantities for positions 1..L-1:
          log_lam_mark (B, L-1), comp (B, L-1) compensator over (t_{n-1}, t_n],
          and the post-event states needed for prediction."""
        B, L = marks.shape
        dev = dts.device
        h = torch.zeros(B, self.cfg.hidden, device=dev, dtype=dts.dtype)
        x = torch.zeros(B, self.P, device=dev, dtype=dts.dtype)
        mixing = self.cfg.mark_mixing
        Cm = None if mixing else self.C()
        Csum = None if mixing else Cm.sum(0)
        log_lam, comp, states, mus, hs = [], [], [], [], []
        for n in range(L):
            if n > 0:
                dt = dts[:, n]
                x_left, integ = self.evolve(x, dt)
                lam_all = mu + self._readout(Cm, x_left)  # (B, M)
                c_n = mu.sum(-1) * dt + (integ @ Csum if not mixing else integ @ F.softplus(self.c_tot))
                if g is not None:
                    gv, gi = self._gomp_terms(g, w, dt)
                    lam_all = lam_all + gv
                    c_n = c_n + gi.sum(-1)
                m = marks[:, n].clamp(max=self.cfg.n_marks - 1)
                log_lam.append(torch.log(lam_all.gather(1, m[:, None])[:, 0] + 1e-12))
                comp.append(c_n)
            else:
                x_left = x
            gap = dts[:, n] if n > 0 else torch.zeros(B, device=dev, dtype=dts.dtype)
            inp = torch.cat([self.emb(marks[:, n].clamp(max=self.cfg.n_marks)),
                             torch.log1p(gap / self.t_scale)[:, None], torch.log(gap / self.t_scale + 1e-3)[:, None]], -1)
            h_new = self.gru(self.drop(inp), h)
            valid = mask[:, n].unsqueeze(-1)
            h = torch.where(valid, h_new, h)
            x_new = torch.sigmoid(self.gate(h)) * x_left + F.softplus(self.jump(h))
            x = torch.where(valid, x_new, x)
            mu = F.softplus(self.mu(h)) + 1e-6
            g, w = self._gomp(h)
            if mixing:
                Cm = self.C(h)
            hs.append(h)
            states.append(x)
            mus.append(mu if g is None else torch.cat([mu, g, w], -1))
        self._last_h = torch.stack(hs, 1)  # post-event GRU states, used by prediction
        return torch.stack(log_lam, 1), torch.stack(comp, 1), torch.stack(states, 1), torch.stack(mus, 1)

    def loglik(self, dts, marks, mask):
        """Exact LL summed over scored events (positions 1..L-1 with mask) and their count."""
        log_lam, comp, _, _ = self.forward(dts, marks, mask)
        m = mask[:, 1:].float()
        ll = ((log_lam - comp) * m).sum()
        return ll, m.sum()

    # ------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict_next(self, x, mu, s_max: torch.Tensor, n_grid: int = 400, h=None):
        """Bayes-optimal next gap E[dt] and marginal mark argmax from post-event state.

        x (N, P), mu (N, M), s_max (N,) horizon where survival is negligible."""
        Cm = self.C(h) if self.cfg.mark_mixing else self.C()
        Csum = F.softplus(self.c_tot) if self.cfg.mark_mixing else Cm.sum(0)
        N = x.shape[0]
        M = self.cfg.n_marks
        g = w = None
        if self.cfg.gompertz:
            mu, g, w = mu[:, :M], mu[:, M:3 * M], mu[:, 3 * M:3 * M + 2]
        u = torch.linspace(0, 1, n_grid, device=x.device)
        grid = s_max[:, None] * (torch.expm1(6 * u) / math.expm1(6))[None, :]  # dense near 0
        xs = x[:, None, :].expand(N, n_grid, -1)
        x_t, integ = self.evolve(xs, grid)
        Lam = mu.sum(-1, keepdim=True) * grid + integ @ Csum
        lam = mu[:, None, :] + self._readout(Cm, x_t)  # (N, G, M)
        if g is not None:
            gv, gi = self._gomp_terms(g[:, None, :], w[:, None, :].expand(-1, grid.shape[1], -1), grid)
            lam = lam + gv
            Lam = Lam + gi.sum(-1)
        S = torch.exp(-Lam)
        dens_m = lam * S[..., None]
        w = torch.diff(grid, dim=1)
        trap = lambda f: (0.5 * (f[:, 1:] + f[:, :-1]) * (w if f.dim() == 2 else w[..., None])).sum(1)
        e_dt = trap(S)  # E[dt] = int S
        p_mark = trap(dens_m)
        return e_dt, p_mark.argmax(-1), S[:, -1]
