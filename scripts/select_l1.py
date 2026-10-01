# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Select the exposure-weighted L1 level on news/anticipation kernels by held-out windows.

    python scripts/select_l1.py 2024 [l1 values comma-separated]

Windows are split 80/20 (stratified: news vs placebo, seed 0).  For each l1 the model is
fitted on the 80% and scored by the profile log-likelihood of the 20% (window baselines
re-estimated, all shared parameters fixed).  Also reports release vs placebo kernel masses.
Results -> results/json/l1_select_<year>.json
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from infodiff.events.data import concat_eventdata
from infodiff.events.windows import load_eventdata
from infodiff.experiments.main_study import MainSpec
from infodiff.models.msx import MSXHawkes

years = sys.argv[1].split(",")
grid = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
data = concat_eventdata([load_eventdata(Path(f"data/processed/panel_{y}.pkl")) for y in years])
pl = np.array([m["placebo"] for m in data.meta["windows"]])
rng = np.random.default_rng(0)
val = np.zeros(data.n_windows, bool)
for grp in (np.where(pl)[0], np.where(~pl)[0]):
    val[rng.choice(grp, max(1, int(round(0.2 * len(grp)))), replace=False)] = True
tr, va = data.subset(np.where(~val)[0]), data.subset(np.where(val)[0])
for d_, idx in ((tr, np.where(~val)[0]), (va, np.where(val)[0])):
    d_.meta["windows"] = [data.meta["windows"][i] for i in idx]
out = Path(f"results/json/l1_select_{'_'.join(years)}.json")
rows = json.loads(out.read_text()) if out.exists() else []
done = {r["l1"] for r in rows}
types = data.meta["type_index"]
for l1 in grid:
    if l1 in done:
        continue
    t = time.time()
    ms = MainSpec(l1_exo=l1)
    m = MSXHawkes(ms.design(), l1_exo=l1, gap_tol=1e-2, gap_rel=1e-7, dtype=__import__("torch").float32).fit(tr)
    ll = m.heldout_loglik(va)
    B = m.exo_weights(data.n_news_types, data.n_marks)
    mass = {k: float(B[:, c, :, :, 0].sum()) for k, c in types.items()}
    row = dict(l1=l1, val_ll=ll, n_val_events=int(va.counts().sum()), mass=mass,
               max_gap=max(r.gap for r in m.reports), seconds=time.time() - t)
    rows.append(row)
    out.write_text(json.dumps(rows, indent=1))
    real = np.mean([v for k, v in mass.items() if not k.startswith("PLACEBO")])
    plc = np.mean([v for k, v in mass.items() if k.startswith("PLACEBO")])
    print(f"l1={l1:5.2f} val_ll={ll:14.2f} mean release mass {real:7.1f} mean placebo mass {plc:7.1f} "
          f"gap {row['max_gap']:.1e} {row['seconds']:.0f}s", flush=True)
best = max(rows, key=lambda r: r["val_ll"])
print("selected l1", best["l1"])
