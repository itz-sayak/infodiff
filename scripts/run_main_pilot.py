import time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from infodiff.events.windows import load_eventdata
from infodiff.experiments.main_study import MainSpec, fit, analyse
data = load_eventdata(Path("data/processed/pilot_2022_2023.pkl"))
ms = MainSpec()
t = time.time()
m = fit(data, ms)
res = analyse(m, data, ms, Path("results/json"), "pilot_2022_2023")
print("rho", res["rho"], "relax", res["relaxation_time_s"], "time", time.time() - t)
for k in ["CPI", "NFP", "FOMC", "PLACEBO"]:
    e = res["per_type"].get(k, {}).get("z0", {})
    for a, r in e.items():
        print(k, a, {q: round(r[q], 3) for q in ["extra_events_direct", "extra_events_total", "amplification", "t50_direct", "t50_total", "t90_total"]})
print("placebo mass", [round(x, 3) for x in res.get("placebo_mass_per_dim", [])])
print("ED reject", res["ed_reject_rate_per_dim"])
