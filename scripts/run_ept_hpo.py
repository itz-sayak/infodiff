"""EPT hyperparameter selection on validation log-likelihood (seed 0, 100 epochs).
Appends to results/json/ept_hpo.json and writes configs/ept_best.json = config with the best
mean validation-LL rank across the tuning datasets (used by run_ept.py "auto")."""
import itertools, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from infodiff.experiments.track_b import train_one

datasets = sys.argv[1].split(",")
grid = [dict(hidden=h, n_rates=8, phases=r, epochs=100) for h, r in itertools.product((64, 128), (2, 4))]
out = Path("results/json/ept_hpo.json")
rows = json.loads(out.read_text()) if out.exists() else []
done = {(r["dataset"], json.dumps(r["config"], sort_keys=True)) for r in rows}
for ds in datasets:
    for g in grid:
        key = (ds, json.dumps(g, sort_keys=True))
        if key in done:
            continue
        r = train_one(ds, 0, **g)
        r["config"] = g
        rows.append(r)
        out.write_text(json.dumps(rows, indent=1))
        print(ds, g, "val", round(r["val_ll"], 4), "test", round(r["ll_per_event"], 4), flush=True)
# rank configs by validation LL within each dataset, choose best mean rank
cfgs = [json.dumps(g, sort_keys=True) for g in grid]
ranks = []
for ds in datasets:
    v = {json.dumps(r["config"], sort_keys=True): r["val_ll"] for r in rows if r["dataset"] == ds}
    order = sorted(cfgs, key=lambda c: -v.get(c, -np.inf))
    ranks.append({c: order.index(c) for c in cfgs})
best = min(cfgs, key=lambda c: np.mean([rk[c] for rk in ranks]))
best = json.loads(best); best.pop("epochs", None)
Path("configs/ept_best.json").write_text(json.dumps(best, indent=1))
print("selected", best)
