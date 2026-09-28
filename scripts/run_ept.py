"""Train EPT-TPP on one dataset for several seeds; append results to results/json/ept_<dataset>.json.
    python scripts/run_ept.py <dataset> <seeds comma> '<json config>'
"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from infodiff.experiments.track_b import train_one_safe as train_one

ds, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
if sys.argv[3].startswith("auto"):  # validation-selected config (+ optional overrides after ':')
    cfg = json.loads(Path("configs/ept_best.json").read_text())
    if ":" in sys.argv[3]:
        cfg.update(json.loads(sys.argv[3].split(":", 1)[1]))
else:
    cfg = json.loads(sys.argv[3])
out = Path("results/json") / f"ept_{ds}.json"
rows = json.loads(out.read_text()) if out.exists() else []
for s in seeds:
    r = train_one(ds, s, **cfg)
    r["config"] = cfg
    rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print("RESULT " + json.dumps(r), flush=True)
