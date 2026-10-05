# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Exact channel attribution of fitted EPT-X models: the share of the expected number of test
events contributed by each channel (baseline, neural Erlang state, hazard terms, Hawkes backbone,
renewal), computed from the closed-form compensator. Re-evaluates the selected final run (seed 0)
of each dataset from its checkpoint, without training.
    python scripts/channel_attribution.py ds:tag [ds:tag ...]  -> results/json/channel_attribution.json"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from infodiff.experiments.track_b import train_one  # noqa: E402

out = Path("results/json/channel_attribution.json")
rows = json.loads(out.read_text()) if out.exists() else {}
for spec in sys.argv[1:]:
    ds, tag = spec.split(":")
    run = json.loads(Path(f"results/json/ept_runs/{ds}__{tag}_s0.json").read_text())
    cfg = {k: v for k, v in run["config"].items() if k not in ("verbose",)}
    r = train_one(ds, 0, ckpt=f"results/ckpt/{ds}__{tag}_s0.pt", eval_only=True, **cfg)
    assert abs(r["ll_per_event"] - run["ll_per_event"]) < 1e-3, (r["ll_per_event"], run["ll_per_event"])
    rows[ds] = dict(tag=tag, ll_per_event=r["ll_per_event"], channel_share=r["channel_share"],
                    backbone_rho=r.get("backbone_rho"))
    print(ds, tag, json.dumps(rows[ds]), flush=True)
    out.write_text(json.dumps(rows, indent=1))
