"""Train one EPT/EPT-X seed and write results/json/ept_runs/<dataset>__<tag>_s<seed>.json.
One file per run, so seeds of the same dataset can run in parallel (GPU + CPU workers).
    python scripts/run_ept_seed.py <dataset> <seed> <tag> '<json config | auto[:overrides]>'"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from infodiff.experiments.track_b import train_one_safe as train_one
ds, seed, tag, spec = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
if spec.startswith("auto"):  # validation-selected config (+ optional overrides after ':')
    cfg = json.loads(Path("configs/ept_best.json").read_text())
    if ":" in spec:
        cfg.update(json.loads(spec.split(":", 1)[1]))
else:
    cfg = json.loads(spec)
ck = Path("results/ckpt") / f"{ds}__{tag}_s{seed}.pt"  # resumable: rerun the same command after a crash
r = train_one(ds, seed, ckpt=str(ck), **cfg)
r["config"], r["tag"] = cfg, tag
out = Path("results/json/ept_runs") / f"{ds}__{tag}_s{seed}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(r, indent=1))
print("RESULT " + json.dumps(r), flush=True)
