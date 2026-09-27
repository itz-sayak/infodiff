"""Fit + analyse the main-study MSX model on one or more yearly panels.
    python scripts/run_main.py 2022            -> results/json/main_2022.json
    python scripts/run_main.py 2022,2023,2024,2025,2026 pooled
"""
import pickle, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from infodiff.events.data import concat_eventdata
from infodiff.events.windows import load_eventdata
from infodiff.experiments.main_study import MainSpec, analyse, fit

years = sys.argv[1].split(",")
tag = sys.argv[2] if len(sys.argv) > 2 else "_".join(years)
data = concat_eventdata([load_eventdata(Path(f"data/processed/panel_{y}.pkl")) for y in years])
print(f"{tag}: windows={data.n_windows} events={len(data.times)}", flush=True)
ms = MainSpec()
t = time.time()
mpath = Path(f"results/models/msx_{tag}.pkl")
if mpath.exists():  # reuse a certified fit; analysis only
    from infodiff.models.msx import MSXHawkes
    blob = pickle.load(open(mpath, "rb"))
    m = MSXHawkes(blob["spec"].design())
    m.theta_d, m.theta_s, m.layout, m.cov, m.reports = blob["theta_d"], blob["theta_s"], blob["layout"], blob["cov"], blob["reports"]
    m.n_dims = data.n_dims
    print("loaded", mpath, "max gap", max(r.gap for r in m.reports), flush=True)
else:
    m = fit(data, ms)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "wb") as f:
        pickle.dump(dict(theta_d=m.theta_d, theta_s=m.theta_s, layout=m.layout, cov=m.cov, reports=m.reports,
                         spec=ms), f, protocol=5)
res = analyse(m, data, ms, Path("results/json"), tag)
print("rho", res["rho"], "relaxation_s", res["relaxation_time_s"], f"total {time.time() - t:.0f}s", flush=True)
