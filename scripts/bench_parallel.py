# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Seconds per training epoch (including the validation pass) of EPT-X with the event loop vs the parallel scan (same model).
    python scripts/bench_parallel.py <dataset> '<json cfg>' [device]
Writes results/json/bench_parallel_<dataset>.json."""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
from infodiff.experiments.track_b import batches, evaluate, load_split, quantile_atoms  # noqa: E402
from infodiff.models.neural_pt import EPTConfig, EPTTPP  # noqa: E402

ds, cfg_kw = sys.argv[1], json.loads(sys.argv[2])
dev = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
tr, M = load_split(ds, "train")
va, _ = load_split(ds, "validation")
gaps = np.concatenate([d[1:] for d, _ in tr])
gaps = gaps[gaps > 0]
q001, q01, q999 = (float(np.quantile(gaps, q)) for q in (0.001, 0.01, 0.999))
lg = np.log(gaps + 0.1 * q01)
qm, qr, qd = quantile_atoms(gaps, cfg_kw.pop("rn_quantile", 0)) if cfg_kw.get("renewal") else ((), (), ())
bs = cfg_kw.pop("bs", 64)
cfg = EPTConfig(n_marks=M, tau_min=float(np.quantile(gaps, 0.02)), tau_max=float(np.quantile(gaps, 0.995)) * 5,
                rn_lo=0.5 * q001, rn_hi=5 * q999, input_v2=bool(cfg_kw.get("renewal")), gap_eps=0.1 * q01,
                gap_mu=float(lg.mean()), gap_sd=float(lg.std() + 1e-6), tie_thr=10 * q01,
                rn_q_means=qm, rn_q_orders=qr, rn_q_delta=qd, **cfg_kw)
out = {}
for par in (False, True):
    torch.manual_seed(0)
    model = EPTTPP(cfg).to(dev)
    model.cfg.parallel = par
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    tot = 0.0
    for dts, mk, msk in batches(tr, M, bs, True, dev, rng):
        ll, n = model.loglik(dts, mk, msk)
        loss = -ll / n
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(ll)
    evaluate(model, va, M, dev)  # validation pass, as in every training epoch
    if dev == "cuda":
        torch.cuda.synchronize()
    out["parallel" if par else "event_loop"] = time.time() - t0
    print(ds, "parallel" if par else "event loop", f"{time.time() - t0:.1f}s", flush=True)
out.update(dataset=ds, device=dev, config=cfg_kw, speedup=out["event_loop"] / out["parallel"])
Path(f"results/json/bench_parallel_{ds}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
