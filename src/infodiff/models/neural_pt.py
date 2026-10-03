# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
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

Hawkes backbone (residual structure).  Alongside the neural state, an explicit
multivariate phase-type Hawkes state x_h (fixed log-grid rates, one Erlang chain per
source mark and rate) receives the *linear* jump beta_k for the mark of each event and
is read out by a nonnegative matrix C_h.  With the neural read-outs at zero the model is
exactly MSX-Hawkes, so the network only has to learn departures from linear Hawkes; the
compensator remains exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    hawkes_backbone: bool = True  # explicit linear multivariate phase-type Hawkes state
    hb_rates: int = 10
    hb_phases: int = 2
    # EPT-X: history-conditioned defective hyper-Erlang renewal channel (competing risks)
    renewal: bool = False
    ept_channel: bool = True  # False: renewal channel only (ablation)
    rn_shift: bool = False  # history-dependent log-rate shift of each scale within its grid cell
    rn_scales: int = 24  # log-grid of mean gaps [rn_lo, rn_hi]
    rn_orders: tuple = field(default_factory=lambda: (1, 4, 16))  # Erlang orders per scale
    rn_lo: float = 1e-10
    rn_hi: float = 1e3
    # input encoding v2: standardised log-gap + tie flag (v1 saturates below ~1e-7 t_scale)
    input_v2: bool = False
    gap_eps: float = 1e-12
    gap_mu: float = 0.0
    gap_sd: float = 1.0
    tie_thr: float = 0.0
    # deep event encoder: stacked residual GRU layers (dropout, LayerNorm); only h_n changes,
    # so the between-event intensity and compensator stay closed form
    n_layers: int = 1
    layer_norm: bool = False
    # history encoder: "gru" (sequential) or "transformer" (causal self-attention over past
    # events, computed in parallel; the intensity and compensator between events are unchanged)
    encoder: str = "gru"
    tf_layers: int = 2
    tf_heads: int = 4
    tf_max_len: int = 4096


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
        self.gru = nn.GRUCell(cfg.mark_emb + (3 if cfg.input_v2 else 2), cfg.hidden)
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
        if cfg.hawkes_backbone:
            taus_h = torch.logspace(math.log10(cfg.tau_min), math.log10(cfg.tau_max), cfg.hb_rates)
            self.register_buffer("hb_betas", 1.0 / taus_h)
            self.Ph = cfg.n_marks * cfg.hb_rates * cfg.hb_phases
            self.Ch_raw = nn.Parameter(torch.full((cfg.n_marks, self.Ph), -5.0))
        else:
            self.Ph = 0
        if cfg.renewal:
            S, O = cfg.rn_scales, len(cfg.rn_orders)
            taus = torch.logspace(math.log10(cfg.rn_lo), math.log10(cfg.rn_hi), S)
            R = torch.tensor(cfg.rn_orders, dtype=torch.float32)
            # atom j = s * O + o: Erlang(R_o, beta = R_o / tau_s), mean tau_s
            self.register_buffer("rn_R", R.repeat(S))
            self.register_buffer("rn_beta", (R[None, :] / taus[:, None]).flatten())
            self.J = S * O
            self.rn_w = nn.Linear(cfg.hidden, self.J + 1)  # last logit: defect ("no renewal event")
            self.rn_q = nn.Linear(cfg.hidden, S * cfg.n_marks)  # mark law per time scale
            self.rn_qb = nn.Parameter(torch.zeros(S, cfg.n_marks))
            nn.init.zeros_(self.rn_q.weight)
            nn.init.zeros_(self.rn_q.bias)
            if cfg.rn_shift:  # c_s(h) in (-delta, delta): atoms slide continuously within their cell
                self.rn_c = nn.Linear(cfg.hidden, S)
                nn.init.zeros_(self.rn_c.weight)
                nn.init.zeros_(self.rn_c.bias)
                self.rn_delta = 0.5 * math.log(cfg.rn_hi / cfg.rn_lo) / max(S - 1, 1)
        nn.init.constant_(self.gate.bias, 2.0)  # start close to additive (Hawkes-like) updates
        nn.init.constant_(self.jump.bias, -1.0)
        nn.init.constant_(self.mu.bias, -2.0)
        self.t_scale = float(math.sqrt(cfg.tau_min * cfg.tau_max))  # input normalisation
        if cfg.n_layers > 1:  # created last so that the default model's initialisation is unchanged
            self.gru_up = nn.ModuleList([nn.GRUCell(cfg.hidden, cfg.hidden) for _ in range(cfg.n_layers - 1)])
        if cfg.layer_norm:
            self.lns = nn.ModuleList([nn.LayerNorm(cfg.hidden) for _ in range(cfg.n_layers)])
        if cfg.encoder == "transformer":  # created last: the default model's initialisation is unchanged
            d_in = cfg.mark_emb + (3 if cfg.input_v2 else 2)
            self.tf_in = nn.Linear(d_in, cfg.hidden)
            self.tf_pos = nn.Embedding(cfg.tf_max_len, cfg.hidden)
            layer = nn.TransformerEncoderLayer(cfg.hidden, cfg.tf_heads, 2 * cfg.hidden, cfg.dropout,
                                               batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, cfg.tf_layers, enable_nested_tensor=False)
            self.tf_out = nn.LayerNorm(cfg.hidden)

    def _encode_all(self, dts, marks, mask):
        """Causal Transformer summaries h_n of events 0..n for every position (B, L, hidden)."""
        B, L = marks.shape
        gap = dts.clone()
        gap[:, 0] = 0.0
        feats = torch.cat([self.emb(marks.clamp(max=self.cfg.n_marks))]
                          + [f.view(B, L, 1) for f in self._gap_features(gap.reshape(-1))], -1)
        pos = torch.arange(L, device=dts.device).clamp(max=self.cfg.tf_max_len - 1)
        x = self.tf_in(self.drop(feats)) + self.tf_pos(pos)[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dts.device), 1)
        return self.tf_out(self.tf(x, mask=causal, src_key_padding_mask=~mask))

    def _encode(self, inp, hs):
        """One event step of the (possibly deep) encoder. hs: list of per-layer states.
        Returns the new per-layer states and the top output h_n used by every head."""
        new = [self.gru(self.drop(inp), hs[0])]
        x = self.lns[0](new[0]) if self.cfg.layer_norm else new[0]
        for l in range(1, self.cfg.n_layers):
            hl = self.gru_up[l - 1](x, hs[l])
            new.append(hl)
            x = x + self.drop(hl)
            if self.cfg.layer_norm:
                x = self.lns[l](x)
        return new, x

    # ------------------------------------------------------------ phase-type algebra
    def _betas(self):
        return torch.exp(self.log_beta)  # (Cn*K,)

    def _split(self, x):
        return x.view(*x.shape[:-1], -1, self.R)

    def evolve(self, x, dt):
        return self._evolve_with(x, dt, self._betas(), self.R)

    def _evolve_hb(self, x, dt):
        return self._evolve_with(x, dt, self.hb_betas.repeat(self.cfg.n_marks), self.cfg.hb_phases)

    @staticmethod
    def _evolve_with(x, dt, b, R):
        """Exact Erlang-chain evolution and integral.

        Per rate block s' = beta (N - I) s (N = down-shift), hence
          s_r(dt)       = sum_{j<r} w_j s_{r-j}(0),     w_j = e^{-y} y^j / j!,  y = beta dt
          int_0^dt s_r  = sum_{j<r} s_{r-j}(0) P(j+1, y) / beta
        with P the regularised lower incomplete gamma, P(j+1, y) = 1 - sum_{i<=j} w_i."""
        xs = x.view(*x.shape[:-1], -1, R)  # (..., B, R)
        y = b * dt.unsqueeze(-1)  # (..., B)
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

    # ------------------------------------------------------------ EPT-X renewal channel
    def _rn_heads(self, h):
        """log atom weights (B, J+1; last = defect), log mark law per scale (B, S, M) and the
        log-rate shift of every atom (B, J) or None."""
        S, M = self.cfg.rn_scales, self.cfg.n_marks
        logw = torch.log_softmax(self.rn_w(h), -1)
        logq = torch.log_softmax(self.rn_q(h).view(-1, S, M) + self.rn_qb[None], -1)
        logb = None
        if self.cfg.rn_shift:
            logb = (self.rn_delta * torch.tanh(self.rn_c(h))).repeat_interleave(len(self.cfg.rn_orders), -1)
        return logw, logq, logb

    def _rn_logf_logQ(self, dt, logb=None):
        """Erlang log-density and log-survival of every atom at lag dt (...,) -> (..., J).
        Integer order R:  Q(R, y) = e^{-y} sum_{i<R} y^i / i!  (exact, stable in log space)."""
        log_beta = torch.log(self.rn_beta) if logb is None else torch.log(self.rn_beta) + logb
        y = torch.exp(log_beta) * dt.unsqueeze(-1)
        logy = torch.log(y.clamp_min(1e-30))
        R = self.rn_R
        logf = log_beta + (R - 1) * logy - y - torch.lgamma(R)
        n_exact = int(min(R.max().item(), 16))
        i = torch.arange(n_exact, device=dt.device, dtype=dt.dtype)
        terms = i * logy.unsqueeze(-1) - torch.lgamma(i + 1)
        terms = torch.where(i < R.unsqueeze(-1), terms, torch.full_like(terms, -math.inf))
        logQ = -y + torch.logsumexp(terms, -1)
        if R.max() > 16:  # sharp atoms (CV = R^-1/2 <= 1/4): regularised upper incomplete gamma
            big = R > 16
            # evaluate only where used, at y > 0: the gradient of gammaincc at y = 0 is 0 * log 0 = NaN
            a = torch.where(big, R, torch.full_like(R, 17.0)).double().expand_as(y)
            yb = torch.where(big, y, torch.ones_like(y)).double().clamp_min(1e-30)
            qb = torch.special.gammaincc(a, yb).clamp_min(1e-300)
            logQ = torch.where(big, torch.log(qb).to(dt.dtype), logQ)
        return logf, logQ

    def _rn_logS(self, logw, logQ):
        """log survival of the defective mixture: defect atom has Q = 1."""
        body = logw[..., :-1] + logQ
        return torch.logsumexp(torch.cat([body, logw[..., -1:].expand(*body.shape[:-1], 1)], -1), -1)

    def _rn_log_lam(self, logw, logq, dt, marks=None, logb=None, want_total=False):
        """log renewal intensity of each mark (..., M), or of the given marks (...,), and the
        compensator -log S(dt); with want_total also the log total renewal intensity."""
        O = len(self.cfg.rn_orders)
        logf, logQ = self._rn_logf_logQ(dt, logb)
        logS = self._rn_logS(logw, logQ)
        base = logw[..., :-1] + logf  # (..., J)
        if marks is not None:
            lq = logq.gather(-1, marks[:, None, None].expand(-1, logq.shape[1], 1))[..., 0]  # (B, S)
            lp = torch.logsumexp(base + lq.repeat_interleave(O, -1), -1)
            if want_total:
                return lp - logS, -logS, torch.logsumexp(base, -1) - logS
            return lp - logS, -logS
        lq = logq.repeat_interleave(O, -2)  # (..., J, M)
        lp = torch.logsumexp(base.unsqueeze(-1) + lq, -2)
        return lp - logS.unsqueeze(-1), -logS

    def _gap_features(self, gap):
        if not self.cfg.input_v2:
            return [torch.log1p(gap / self.t_scale)[:, None], torch.log(gap / self.t_scale + 1e-3)[:, None]]
        z = (torch.log(gap + self.cfg.gap_eps) - self.cfg.gap_mu) / self.cfg.gap_sd
        return [z[:, None], torch.log1p(gap / self.t_scale)[:, None], (gap < self.cfg.tie_thr).to(gap.dtype)[:, None]]

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
        hb = self.cfg.hawkes_backbone
        xh = torch.zeros(B, self.Ph, device=dev, dtype=dts.dtype) if hb else None
        Chb = F.softplus(self.Ch_raw) if hb else None
        mixing = self.cfg.mark_mixing
        rn = self.cfg.renewal
        Cm = None if mixing else self.C()
        Csum = None if mixing else Cm.sum(0)
        log_lam, comp, states, mus, hs, log_tot = [], [], [], [], [], []
        H = [h] * self.cfg.n_layers  # per-layer encoder states
        tf = self.cfg.encoder == "transformer"
        Htf = self._encode_all(dts, marks, mask) if tf else None
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
                if hb:
                    xh_left, integ_h = self._evolve_hb(xh, dt)
                    lam_all = lam_all + xh_left @ Chb.T
                    c_n = c_n + integ_h @ Chb.sum(0)
                m = marks[:, n].clamp(max=self.cfg.n_marks - 1)
                ll_n = torch.log(lam_all.gather(1, m[:, None])[:, 0] + 1e-12)
                lt_n = torch.log(lam_all.sum(-1) + 1e-12 * self.cfg.n_marks)  # total (time term)
                if rn:
                    lr, cr, lrt = self._rn_log_lam(rn_w, rn_q, dt, m, rn_b, want_total=True)
                    if self.cfg.ept_channel:
                        ll_n, c_n, lt_n = torch.logaddexp(ll_n, lr), c_n + cr, torch.logaddexp(lt_n, lrt)
                    else:
                        ll_n, c_n, lt_n = lr, cr, lrt
                log_lam.append(ll_n)
                log_tot.append(lt_n)
                comp.append(c_n)
            else:
                x_left = x
                xh_left = xh
            valid = mask[:, n].unsqueeze(-1)
            if tf:
                h_new = Htf[:, n]
            else:
                gap = dts[:, n] if n > 0 else torch.zeros(B, device=dev, dtype=dts.dtype)
                inp = torch.cat([self.emb(marks[:, n].clamp(max=self.cfg.n_marks))] + self._gap_features(gap), -1)
                H_new, h_new = self._encode(inp, H)
                H = [torch.where(valid, a, b) for a, b in zip(H_new, H)]
            h = torch.where(valid, h_new, h)
            x_new = torch.sigmoid(self.gate(h)) * x_left + F.softplus(self.jump(h))
            x = torch.where(valid, x_new, x)
            if hb:  # linear Hawkes jump: phase 0 of every rate of the event's mark gets beta_k
                Kh, Rh = self.cfg.hb_rates, self.cfg.hb_phases
                onehot = F.one_hot(marks[:, n].clamp(max=self.cfg.n_marks - 1), self.cfg.n_marks).to(dts.dtype)
                jmp = torch.zeros(B, self.cfg.n_marks, Kh, Rh, device=dev, dtype=dts.dtype)
                jmp[..., 0] = onehot[:, :, None] * self.hb_betas[None, None, :]
                xh = torch.where(valid, xh_left + jmp.flatten(1), xh)
            mu = F.softplus(self.mu(h)) + 1e-6
            g, w = self._gomp(h)
            if mixing:
                Cm = self.C(h)
            if rn:
                rn_w, rn_q, rn_b = self._rn_heads(h)
            hs.append(h)
            states.append(torch.cat([x, xh], -1) if hb else x)
            mus.append(mu if g is None else torch.cat([mu, g, w], -1))
        self._last_h = torch.stack(hs, 1)
        self._last_log_tot = torch.stack(log_tot, 1) if log_tot else None  # log total intensity at events  # post-event GRU states, used by prediction
        return torch.stack(log_lam, 1), torch.stack(comp, 1), torch.stack(states, 1), torch.stack(mus, 1)

    def loglik(self, dts, marks, mask):
        """Exact LL summed over scored events (positions 1..L-1 with mask) and their count."""
        log_lam, comp, _, _ = self.forward(dts, marks, mask)
        m = mask[:, 1:].float()
        ll = ((log_lam - comp) * m).sum()
        return ll, m.sum()

    # ------------------------------------------------------------ prediction
    def _curve(self, xpack, pk, grid, h=None):
        """Marked intensities lambda (N, G, M) and compensators Lam (N, G) on a lag grid (N, G)
        from post-event packed states xpack (N, P + Ph) and head parameters pk."""
        M = self.cfg.n_marks
        x, xh = xpack[:, : self.P], xpack[:, self.P:]
        Cm = self.C(h) if self.cfg.mark_mixing else self.C()
        Csum = F.softplus(self.c_tot) if self.cfg.mark_mixing else Cm.sum(0)
        mu, g, w = pk[:, :M], None, None
        if self.cfg.gompertz:
            g, w = pk[:, M:3 * M], pk[:, 3 * M:3 * M + 2]
        N, G = grid.shape
        x_t, integ = self.evolve(x[:, None, :].expand(N, G, -1), grid)
        Lam = mu.sum(-1, keepdim=True) * grid + integ @ Csum
        lam = mu[:, None, :] + self._readout(Cm, x_t)
        if g is not None:
            gv, gi = self._gomp_terms(g[:, None, :], w[:, None, :].expand(-1, G, -1), grid)
            lam = lam + gv
            Lam = Lam + gi.sum(-1)
        if self.cfg.hawkes_backbone:
            Chb = F.softplus(self.Ch_raw)
            xh_t, ih = self._evolve_hb(xh[:, None, :].expand(N, G, -1), grid)
            lam = lam + xh_t @ Chb.T
            Lam = Lam + ih @ Chb.sum(0)
        if self.cfg.renewal:
            logw, logq, logb = self._rn_heads(h)
            lr, cr = self._rn_log_lam(logw[:, None, :], logq[:, None, :, :], grid,
                                      logb=None if logb is None else logb[:, None, :])
            if self.cfg.ept_channel:
                lam, Lam = lam + torch.exp(lr), Lam + cr
            else:
                lam, Lam = torch.exp(lr), cr
        return lam, Lam

    @torch.no_grad()
    def predict_next(self, x, mu, s_max: torch.Tensor, n_grid: int = 400, h=None):
        """Bayes-optimal next gap E[dt] and marginal mark argmax from post-event state."""
        u = torch.linspace(0, 1, n_grid, device=x.device, dtype=x.dtype)
        if self.cfg.renewal:  # log grid from below the fastest atom: the burst mode is resolved
            lo = torch.log(torch.as_tensor(self.cfg.rn_lo * 1e-2, dtype=x.dtype, device=x.device))
            g = torch.exp(lo + (torch.log(s_max)[:, None] - lo) * u[None, 1:])
            grid = torch.cat([torch.zeros_like(g[:, :1]), g], 1)
        else:
            grid = s_max[:, None] * (torch.expm1(6 * u) / math.expm1(6))[None, :]  # dense near 0
        lam, Lam = self._curve(x, mu, grid, h)
        S = torch.exp(-Lam)
        dens_m = lam * S[..., None]
        wg = torch.diff(grid, dim=1)
        trap = lambda f: (0.5 * (f[:, 1:] + f[:, :-1]) * (wg if f.dim() == 2 else wg[..., None])).sum(1)
        return trap(S), trap(dens_m).argmax(-1), S[:, -1]
