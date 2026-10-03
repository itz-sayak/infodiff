# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Main study: echo-corrected absorption of scheduled macro news across markets.

Model (per year and pooled): 12-dimensional MSX-Hawkes on delta-crossing events of
EURUSD, USDJPY, XAUUSD, SPX, BTC, ETH (up/down each) with
  * endogenous phase-type kernels (5 ms .. 10 min, Erlang orders 1..2),
  * news kernels per release type with Erlang orders 1..3 (reaction humps) and marks
    [1, z+, z-] (standardised surprise),
  * anticipatory pre-release kernels (10 s .. 10 min),
  * per-window intercepts + 15-min time-of-day hats identified by placebo windows,
  * a PLACEBO news type on matched control windows (falsification: its kernel ~ 0).

Outputs (results/json/main_*.json): branching matrix and spectral radius, per
(type, asset) direct vs echo-corrected t50/t90, amplification, price-discovery curves
for +-1 sd surprises, placebo mass, goodness-of-fit (KS/AD/ED per dimension and the
Rambaldi-style per-window ED rejection rate).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..diagnostics.rescaling import excess_dispersion, gof
from ..events.data import EventData
from ..models.dictionary import PhaseTypeDictionary
from ..models.features import DesignSpec
from ..models.msx import MSXHawkes
from ..models.statespace import StateSpaceHawkes


@dataclass
class MainSpec:
    endo: PhaseTypeDictionary = PhaseTypeDictionary.log_grid(0.005, 600.0, 10, orders=2)
    # news kernels: time scales 0.1 s .. 300 s (max mean lag 15 min, well inside the 60-min
    # post window) so that they cannot act as within-window trend absorbers
    exo: PhaseTypeDictionary = PhaseTypeDictionary.log_grid(0.1, 300.0, 8, orders=3)
    ant: PhaseTypeDictionary = PhaseTypeDictionary.log_grid(10.0, 300.0, 4, orders=1)
    n_tod: int = 96
    win_cols: int = 2  # per-window linear baseline (identified with the placebo windows)
    l1_endo: float = 0.0
    l1_exo: float = 0.8  # exposure-weighted L1 on news/anticipation kernels, selected on held-out 2024 windows

    def design(self) -> DesignSpec:
        return DesignSpec(endo=self.endo, exo=self.exo, ant=self.ant, n_tod=self.n_tod, win_cols=self.win_cols)


def fit(data: EventData, ms: MainSpec, device: str | None = None, verbose: bool = True,
        store_cov: bool = True) -> MSXHawkes:
    m = MSXHawkes(ms.design(), l1_endo=ms.l1_endo, l1_exo=ms.l1_exo, device=device, verbose=False,
                  gap_tol=1e-2, gap_rel=1e-7, store_cov=store_cov, dtype=__import__("torch").float32)
    t = time.time()
    m.fit(data)
    if verbose:
        for r in m.reports:
            print(f"  dim {r.dim:2d} n={r.n_events:8d} ll={r.loglik:14.2f} gap={r.gap:.1e} it={r.iters} {r.seconds:.0f}s",
                  flush=True)
        print(f"  total fit {time.time() - t:.0f}s", flush=True)
    return m


def news_weights(m: MSXHawkes, data: EventData, c: int, z: float) -> np.ndarray:
    """Effective (d, Ke, Re) news-kernel weights for type c at standardised surprise z."""
    B = m.exo_weights(data.n_news_types, data.n_marks)  # (d, C, Ke, Re, M)
    marks = np.array([1.0, max(z, 0.0), max(-z, 0.0)])[: data.n_marks]
    return np.tensordot(B[:, c], marks, axes=([-1], [0]))


def type_mass_se(m: MSXHawkes, data: EventData) -> dict:
    """Total direct-kernel mass per release type (constant mark) and its standard error."""
    if not m.cov:
        return {}
    lay, E, M = m.layout, m.spec.exo.size, data.n_marks
    Pd = m.theta_d.shape[1]
    out = {}
    for name, c in data.meta["type_index"].items():
        mass, var = 0.0, 0.0
        for i in range(data.n_dims):
            cols = Pd + lay["off_exo"] + c * E * M + np.arange(E) * M
            idx, cov = m.cov[i]
            th = np.concatenate([m.theta_d[i], m.theta_s[i]])
            mass += float(th[cols].sum())
            where = {v: k for k, v in enumerate(idx)}
            pos = [where[cc] for cc in cols if cc in where]
            if pos:
                var += float(cov[np.ix_(pos, pos)].sum())
        out[name] = dict(mass=mass, se=float(np.sqrt(max(var, 0.0))))
    return out


def analyse(m: MSXHawkes, data: EventData, ms: MainSpec, out: Path, tag: str,
            t_grid: np.ndarray = np.geomspace(0.05, 3600, 60)) -> dict:
    assets = data.meta["assets"]
    deltas = data.meta["deltas_bps"]
    types = data.meta["type_index"]
    ss = StateSpaceHawkes(ms.endo, m.endo_weights())
    G = ss.branching()
    res = dict(tag=tag, assets=assets, deltas_bps=deltas, n_windows=int(data.n_windows),
               n_events=int(data.counts().sum()), G=G.tolist(), rho=ss.spectral_radius(),
               relaxation_time_s=ss.relaxation_time(), per_type={})
    lay = m.layout
    base = m.window_levels(data.n_windows).mean(axis=1)
    res["endogeneity"] = ss.endogeneity(np.maximum(base, 1e-12)).tolist() if res["rho"] < 1 else None
    t_start = time.time()
    fine = np.geomspace(1e-3, 1e6, 181)  # 20 points per decade; quantiles refined exactly
    keep = np.unique(np.searchsorted(fine, t_grid))  # coarse subset stored for plots
    keep = keep[keep < len(fine)]
    for name, c in types.items():
        entry = {}
        for zlab, z in (("z0", 0.0), ("zpos", 1.0), ("zneg", -1.0)):
            W = news_weights(m, data, c, z)
            if W.sum() <= 0:
                continue
            r = ss.response_curves(ms.exo, W, fine)
            per_asset = {}
            for a, asset in enumerate(assets):
                up, dn = 2 * a, 2 * a + 1
                act_tot = r.int_total[up] + r.int_total[dn]
                act_dir = r.int_direct[up] + r.int_direct[dn]
                if act_dir <= 1e-9:
                    continue
                ct = r.cum_total[:, up] + r.cum_total[:, dn]
                cd = r.cum_direct[:, up] + r.cum_direct[:, dn]
                q = ss.quantile_from_curve
                dm = np.array([up, dn])
                ref = lambda qq, cur, tot, direct: ss.refine_quantile(ms.exo, W, dm, qq, q(fine, cur, tot, qq), direct)
                row = dict(extra_events_direct=float(act_dir), extra_events_total=float(act_tot),
                           # the echo multiplier is only meaningful for a non-negligible direct response
                           amplification=float(act_tot / act_dir) if act_dir >= 0.5 else float("nan"),
                           t50_direct=ref(0.5, cd, act_dir, True), t50_total=ref(0.5, ct, act_tot, False),
                           t90_direct=ref(0.9, cd, act_dir, True), t90_total=ref(0.9, ct, act_tot, False))
                # each event moves the price by (mean size) x delta: one event per timestamp may
                # cross several delta levels (size mark)
                step = deltas[asset] * data.meta.get("mean_event_size", {}).get(asset, 1.0)
                drift = step * (r.cum_total[:, up] - r.cum_total[:, dn])
                row["drift_bps_final"] = float(step * (r.int_total[up] - r.int_total[dn]))
                row["drift_bps_curve"] = drift[keep].tolist()
                row["activity_curve_total"] = (r.total[keep, up] + r.total[keep, dn]).tolist()
                row["activity_curve_direct"] = (r.direct[keep, up] + r.direct[keep, dn]).tolist()
                per_asset[asset] = row
            entry[zlab] = per_asset
        res["per_type"][name] = entry
    t_grid = fine[keep]
    res["t_grid"] = t_grid.tolist()
    print(f"  responses done in {time.time() - t_start:.0f}s", flush=True)
    # placebo falsification: total news-kernel mass of the PLACEBO type vs real types
    pl_types = {k: c for k, c in types.items() if k.startswith("PLACEBO")}
    if pl_types:
        res["placebo_mass_per_dim"] = {k: news_weights(m, data, c, 0.0).sum(axis=(1, 2)).tolist()
                                       for k, c in pl_types.items()}
        res["real_mass_per_dim"] = {k: news_weights(m, data, c, 0.0).sum(axis=(1, 2)).tolist()
                                    for k, c in types.items() if not k.startswith("PLACEBO")}
        # standard errors of the per-type kernel masses from the observed information
        res["mass_se"] = type_mass_se(m, data)
    # goodness of fit
    gofs, ed_rej = [], []
    wid = data.window_ids()
    for i in range(data.n_dims):
        rsd = m.residuals(data, i)
        g = gof(rsd)
        gofs.append(dict(dim=i, n=g.n, ks=g.ks, ks_p=g.ks_p, ad=g.ad, ed_z=g.ed_z, mean=g.mean, var=g.var))
        # Rambaldi-style: ED test per window, rejection rate at 5%
        rows_w = wid[(data.types == i) & data.in_likelihood()]
        rej = [excess_dispersion(rsd[rows_w == w])[1] < 0.05 for w in np.unique(rows_w) if (rows_w == w).sum() >= 30]
        ed_rej.append(float(np.mean(rej)) if rej else None)
    res["gof"] = gofs
    print(f"  goodness of fit done in {time.time() - t_start:.0f}s", flush=True)
    res["ed_reject_rate_per_dim"] = ed_rej
    res["fit_reports"] = [dict(dim=r.dim, n=r.n_events, ll=r.loglik, gap=r.gap, seconds=r.seconds) for r in m.reports]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"main_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    return res
