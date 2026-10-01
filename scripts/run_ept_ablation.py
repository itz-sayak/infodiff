# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""One EPT-X ablation run: python scripts/run_ept_ablation.py <dataset> <seed> <tag> '<json cfg>'
Writes results/json/eptx_ablation_<dataset>_<tag>.json (one file per run: safe to run in parallel)."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from infodiff.experiments.track_b import train_one_safe as train_one
ds, seed, tag, cfg = sys.argv[1], int(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
r = train_one(ds, seed, verbose=True, **cfg)
r["config"], r["tag"] = cfg, tag
Path(f"results/json/eptx_ablation_{ds}_{tag}.json").write_text(json.dumps(r, indent=1))
print("RESULT " + json.dumps(r), flush=True)
