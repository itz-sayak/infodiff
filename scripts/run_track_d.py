"""Track D evaluation (CPU): load the 2022-2024 MSX fit, forecast 2025-2026 releases.
    python scripts/run_track_d.py [model_tag]
"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from infodiff.events.data import concat_eventdata
from infodiff.events.windows import load_eventdata
from infodiff.experiments import track_d_run as D
from infodiff.models.msx import MSXHawkes

tag = sys.argv[1] if len(sys.argv) > 1 else "train_2022_2024"
train = concat_eventdata([load_eventdata(Path(f"data/processed/panel_{y}.pkl")) for y in (2022, 2023, 2024)])
test = concat_eventdata([load_eventdata(Path(f"data/processed/panel_{y}.pkl")) for y in (2025, 2026)])
blob = pickle.load(open(f"results/models/msx_{tag}.pkl", "rb"))
m = MSXHawkes(blob["spec"].design())
m.theta_d, m.theta_s, m.layout, m.n_dims = blob["theta_d"], blob["theta_s"], blob["layout"], train.n_dims
tr = D.collect(train, None, None)
te = D.collect(test, m, train.n_windows)
assets = train.meta["assets"]
deltas = np.array([train.meta["deltas_bps"][a] for a in assets])
res = D.evaluate_all(tr, te, deltas, train.n_news_types, assets)
D.save(res, Path("results/json/track_d.json"))
for name, r in res["activity"].items():
    print(f"{name:28s} RMSE(log1p N) by horizon (mean over assets):", np.round(np.mean(r["rmse_log"], axis=1), 4),
          " QLIKE:", np.round(np.mean(r["qlike"], axis=1), 4))
for name, r in res["drift"].items():
    print(f"{name:28s} drift hit-rate:", np.round(np.nanmean(r["drift_hit"], axis=1), 3), " corr:", np.round(np.nanmean(r["drift_corr"], axis=1), 3))
