# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Track A — synthetic recovery benchmark.

Truths are simulated *outside* the MSX model class where possible (true power-law and
gamma kernels) so the comparison is not rigged.  Every method is scored on identical
train/test simulations:

  G_err     ||G_hat - G||_F / ||G||_F
  rho_err   |rho(G_hat) - rho(G)|
  kern_L1   sum_ij int |phi_hat - phi| dt / sum_ij int phi dt      (lags 1e-4..1e4 s)
  AUC / AP  edge recovery from |G_hat| scores (scenarios with absent edges)
  dLL       (LL_oracle - LL_method) / N_test   nats/event on held-out data (>= 0 ideal)
  seconds   wall time
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..baselines import classical as C
from ..models.dictionary import PhaseTypeDictionary
from ..sim.generic import GenericHawkes, GenericKernel, Shape, simulate_generic

LAGS = np.geomspace(1e-4, 1e4, 4000)


def scenarios() -> dict:
    out = {}
    # S1 exponential (home turf of ExpKern/ADM4)
    G = np.array([[0.5, 0.2], [0.1, 0.4]])
    out["S1-exp"] = (GenericHawkes(np.array([0.5, 0.5]), {(i, j): GenericKernel([G[i, j]], [Shape("exp", 2.0)])
                                                           for i in range(2) for j in range(2)}), 10, 3000.0)
    # S2 power law (Hardiman / Rambaldi: tail exponent ~1.5, short-lag cutoff tau0)
    G = np.array([[0.6, 0.15], [0.15, 0.6]])
    out["S2-powerlaw"] = (GenericHawkes(np.array([0.3, 0.3]), {(i, j): GenericKernel([G[i, j]], [Shape("lomax", 0.05, 1.5)])
                                                               for i in range(2) for j in range(2)}), 10, 3000.0)
    # S3 hump (kernel vanishes at 0, peaks at ~0.7s) + fast exponential
    ks = {}
    for i in range(2):
        for j in range(2):
            ks[(i, j)] = GenericKernel([0.35 if i == j else 0.1, 0.1], [Shape("gamma", 3.0, 3.0), Shape("exp", 50.0)])
    out["S3-hump"] = (GenericHawkes(np.array([0.4, 0.4]), ks), 10, 3000.0)
    # S4 multiscale sparse 4-d network: 10 ms + 30 s components
    rng = np.random.default_rng(7)
    ks = {}
    edges = [(0, 0), (1, 1), (2, 2), (3, 3), (1, 0), (2, 1), (3, 2), (0, 3)]
    for (i, j) in edges:
        n = 0.8 * (0.55 if i == j else 0.25)
        ks[(i, j)] = GenericKernel([0.6 * n, 0.4 * n], [Shape("exp", 100.0), Shape("exp", 1 / 30.0)])
    out["S4-multiscale"] = (GenericHawkes(np.full(4, 0.2), ks), 10, 4000.0)
    # S5 10-d sparse network with power-law kernels (graph recovery)
    d = 10
    ks = {}
    for i in range(d):
        ks[(i, i)] = GenericKernel([0.3], [Shape("lomax", 0.1, 2.0)])
    cand = [(i, j) for i in range(d) for j in range(d) if i != j]
    for idx in rng.choice(len(cand), 18, replace=False):
        i, j = cand[idx]
        ks[(i, j)] = GenericKernel([rng.uniform(0.1, 0.25)], [Shape("lomax", 0.1, 2.0)])
    GH = GenericHawkes(np.full(d, 0.1), ks)
    Gm = GH.branching()
    s = 0.7 / np.max(np.abs(np.linalg.eigvals(Gm)))
    for k in ks.values():
        k.weights = [w * s for w in k.weights]
    out["S5-network10"] = (GH, 10, 5000.0)
    return out


def oracle_loglik(truth: GenericHawkes, data) -> float:
    grid = np.concatenate([[0.0], np.geomspace(1e-6, 1e5, 6000)])
    d = truth.d
    vals = np.stack([np.stack([truth.kernel(i, j, grid) for j in range(d)]) for i in range(d)])
    vals[..., 0] = vals[..., 1]
    return C.tabulated_loglik(grid, vals, truth.mu, data, linear=True)


def kernel_l1(truth: GenericHawkes, fit: C.Fitted) -> float:
    if fit.kernel is None:
        return float("nan")
    num = den = 0.0
    for i in range(truth.d):
        for j in range(truth.d):
            tr = truth.kernel(i, j, LAGS)
            es = np.nan_to_num(fit.kernel(i, j, LAGS))
            num += np.trapezoid(np.abs(es - tr), LAGS)
            den += np.trapezoid(tr, LAGS)
    return float(num / den)


def run_method(name: str, train, dic: PhaseTypeDictionary) -> C.Fitted:
    if name == "MSX":
        return C.fit_msx(train, dic, "MSX (ours)")
    if name == "MSX-auto":
        return C.fit_msx_auto(train, dic)
    if name == "ADM4":
        return C.fit_tick_adm4(train, 1 / np.array([0.01, 0.1, 0.5, 1.0, 10.0, 100.0]))
    if name == "MSX-R4":
        return C.fit_msx(train, PhaseTypeDictionary(dic.betas, orders=4), "MSX-R4 (ours, fixed Erlang-4)")
    if name == "MSX-exp":
        return C.fit_msx(train, PhaseTypeDictionary(dic.betas, orders=1), "MSX-exp (ablation: no Erlang)")
    if name == "ExpKern":
        return C.fit_tick_expkern(train, 1 / np.array([0.01, 0.1, 0.5, 1.0, 10.0, 100.0]))
    if name == "SumExp":
        return C.fit_tick_sumexp(train, dic.betas)
    if name == "EM":
        return C.fit_tick_em(train, np.concatenate([[0.0], np.geomspace(1e-3, 300.0, 30)]))
    if name == "NPHC":
        return C.fit_tick_nphc(train, 100.0)
    if name == "CondLaw":
        return C.fit_tick_claw(train)
    raise KeyError(name)


METHODS = ["MSX-auto", "MSX", "MSX-R4", "MSX-exp", "ExpKern", "SumExp", "ADM4", "EM", "NPHC", "CondLaw"]
CACHEABLE = {"ExpKern", "SumExp", "ADM4", "EM", "NPHC", "CondLaw", "MSX-exp"}  # unchanged by MSX upgrades


def run(out_dir: Path, seeds=(0, 1, 2, 3, 4), only: list[str] | None = None, methods=METHODS) -> list[dict]:
    dic = PhaseTypeDictionary.log_grid(1e-3, 1e3, 13, orders=2)
    rows = []
    cache_file = out_dir / "track_a.json"
    cache = {}
    if cache_file.exists():
        for r in json.loads(cache_file.read_text()):
            if "error" not in r:
                cache[(r["scenario"], r["seed"], r.get("key", ""))] = r
    for sname, (truth, W, L) in scenarios().items():
        if only and sname not in only:
            continue
        Gt = truth.branching()
        rho_t = float(np.max(np.abs(np.linalg.eigvals(Gt))))
        for seed in seeds:
            train = simulate_generic(truth, W, L, seed=1000 + seed)
            test = simulate_generic(truth, W, L, seed=2000 + seed)
            n_test = int((test.types >= 0).sum())
            ll_or = oracle_loglik(truth, test)
            for mname in methods:
                hit = cache.get((sname, seed, mname))
                if hit is not None and mname in CACHEABLE:
                    rows.append(hit)
                    continue
                t = time.time()
                try:
                    fit = run_method(mname, train, dic)
                except Exception as e:  # record failures honestly
                    rows.append(dict(scenario=sname, seed=seed, method=mname, error=repr(e)[:200]))
                    print(f"{sname} seed={seed} {mname}: FAILED {e!r}"[:200], flush=True)
                    continue
                G = np.nan_to_num(np.asarray(fit.G, float))
                row = dict(scenario=sname, seed=seed, method=fit.name, key=mname, n_train=int(len(train.times)),
                           G_err=float(np.linalg.norm(G - Gt) / np.linalg.norm(Gt)),
                           rho_err=float(abs(np.max(np.abs(np.linalg.eigvals(G))) - rho_t)),
                           kern_L1=kernel_l1(truth, fit), seconds=fit.seconds)
                if (Gt == 0).any():
                    off = ~np.eye(truth.d, dtype=bool)
                    y = (Gt[off] > 0).astype(int)
                    row["AUC"] = float(roc_auc_score(y, np.abs(G[off])))
                    row["AP"] = float(average_precision_score(y, np.abs(G[off])))
                if fit.loglik is not None:
                    try:
                        row["dLL"] = float((ll_or - fit.loglik(test)) / n_test)
                    except Exception as e:
                        row["dLL_error"] = repr(e)[:200]
                if "gap" in fit.extra:
                    row["cert_gap"] = fit.extra["gap"]
                for k in ("R", "l1", "decay"):
                    if k in fit.extra:
                        row[k] = fit.extra[k]
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            done = {(r["scenario"], r["seed"], r.get("key")) for r in rows}
            keep = [r for r in cache.values() if (r["scenario"], r["seed"], r.get("key")) not in done]
            (out_dir / "track_a.json").write_text(json.dumps(rows + keep, indent=1))
    return rows
