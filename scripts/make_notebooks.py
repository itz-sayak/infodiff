"""Create the walkthrough notebooks (executed later with nbconvert)."""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
NB.mkdir(exist_ok=True)


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, NB / name)


SETUP = """import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # notebooks run on CPU
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
import numpy as np, json, warnings
warnings.filterwarnings("ignore")"""

nb("01_certified_estimator.ipynb", [
    ("md", "# 01 · MSX-Hawkes: a certified global MLE\n\nSimulate a 2-d phase-type Hawkes process with a news shock, fit MSX, and read the Fenchel duality-gap certificate (Proposition 2)."),
    ("code", SETUP),
    ("code", """from infodiff.models.dictionary import PhaseTypeDictionary
from infodiff.models.features import DesignSpec
from infodiff.models.msx import MSXHawkes
from infodiff.sim.cluster import HawkesTruth, simulate
endo = PhaseTypeDictionary.log_grid(0.05, 5.0, 3, orders=2)
A = np.zeros((2, 2, endo.K, endo.R)); A[0, 0, 0, 1] = 0.3; A[1, 1, 2, 0] = 0.4; A[0, 1, 1, 0] = 0.2
truth = HawkesTruth(endo=endo, A=A, mu=np.array([1.0, 0.7]))
t0 = np.arange(20) * 3000.0
data = simulate(truth, t0=t0, t1=t0 + 2500.0, burn=50.0, seed=3)
print("events per dim:", data.counts())"""),
    ("code", """m = MSXHawkes(DesignSpec(endo=endo), device="cpu", gap_tol=1e-6, gap_rel=0.0).fit(data)
for r in m.reports:
    print(f"dim {r.dim}: loglik {r.loglik:.3f}  certified gap {r.gap:.2e} nats  E[N]/N = {r.expected_count / r.n_events:.6f}")
print("true branching matrix\\n", A.sum(axis=(2, 3)))
print("estimated\\n", m.branching_matrix().round(3))"""),
])

nb("02_echo_corrected_absorption.ipynb", [
    ("md", "# 02 · Markov embedding and echo-corrected absorption\n\nThe mean response to a release is $C e^{\\mathcal A t} z_0$ (Corollary 4). We compare the direct news kernel with the full response and check the stability theorem."),
    ("code", SETUP),
    ("code", """from infodiff.models.dictionary import PhaseTypeDictionary
from infodiff.models.statespace import StateSpaceHawkes
endo = PhaseTypeDictionary.log_grid(0.05, 5.0, 3, orders=2)
exo = PhaseTypeDictionary.log_grid(0.5, 20.0, 2, orders=3)
A = np.zeros((2, 2, endo.K, endo.R)); A[0, 0, 1, 0] = 0.5; A[1, 0, 0, 1] = 0.3; A[0, 1, 2, 0] = 0.25
W = np.zeros((2, exo.K, exo.R)); W[0, 0, 1] = 3.0
ss = StateSpaceHawkes(endo, A)
print("rho(G) =", round(ss.spectral_radius(), 3), " spectral abscissa =", round(ss.spectral_abscissa(), 4))
print("t50 direct =", round(ss.absorption_time(exo, W, 0.5, direct=True), 2), "s;  t50 echo-corrected =",
      round(ss.absorption_time(exo, W, 0.5), 2), "s;  amplification =", ss.amplification(exo, W).round(2))"""),
    ("code", """import matplotlib.pyplot as plt
t = np.geomspace(0.01, 500, 200)
r = ss.response_curves(exo, W, t)
plt.loglog(t, r.total.sum(1), label="echo-corrected", color="#2a78d6")
plt.loglog(t, np.maximum(r.direct.sum(1), 1e-8), label="direct kernel", color="#eb6834")
plt.xlabel("seconds after release"); plt.ylabel("excess intensity"); plt.legend(frameon=False); plt.show()"""),
])

nb("03_data_pipeline.ipynb", [
    ("md", "# 03 · Data: calendar, surprises and δ-crossing events\n\nRequires the downloaded data (`scripts/download_ticks.py`, calendar build)."),
    ("code", SETUP),
    ("code", """import pandas as pd
cal = pd.read_parquet("data/processed/calendar_surprises.parquet")
print(cal.groupby("kind").agg(n=("event_id", "size"), with_expectation=("expected", "count")))
cal[cal.kind == "CPI"][["event_id", "actual", "expected", "z", "exp_source"]].tail(5)"""),
    ("code", """card = json.load(open("results/json/data_card.json"))
pd.DataFrame(card["years"]).T[["windows", "news", "placebo", "dropped", "events"]]"""),
])

nb("04_main_study.ipynb", [
    ("md", "# 04 · Main study results (pooled 2022–2026)\n\nLoads the analysis JSON written by `scripts/run_main.py`."),
    ("code", SETUP),
    ("code", """r = json.load(open("results/json/main_pooled.json"))
print("rho(G) =", round(r["rho"], 3), "; slowest echo mode:", round(r["relaxation_time_s"] / 60, 1), "min")
se = r["mass_se"]
import pandas as pd
kinds = [k for k in se if not k.startswith("PLACEBO")]
pd.DataFrame({"release": [se[k]["mass"] for k in kinds], "placebo": [se["PLACEBO_" + k]["mass"] for k in kinds]}, index=kinds).round(2)"""),
    ("code", """rows = []
for typ in ["CPI", "NFP", "FOMC", "PPI"]:
    for a, x in r["per_type"][typ]["z0"].items():
        rows.append(dict(release=typ, asset=a, t50_direct=x["t50_direct"], t50_echo=x["t50_total"], t90_echo=x["t90_total"]))
pd.DataFrame(rows).round(1)"""),
])

nb("05_benchmarks.ipynb", [
    ("md", "# 05 · Benchmarks\n\nTrack A (synthetic), Track C (LOBSTER) and Track D (forecasting) summaries from `results/json`."),
    ("code", SETUP),
    ("code", """import pandas as pd
a = pd.DataFrame([x for x in json.load(open("results/json/track_a.json")) if "error" not in x])
a.groupby(["scenario", "method"])[["G_err", "kern_L1", "dLL"]].mean().round(4)"""),
    ("code", """p = Path("results/json/track_d.json")
if p.exists():
    d = json.load(open(p))
    print({k: np.round(np.mean(v["rmse_log"], axis=1), 3).tolist() for k, v in d["activity"].items()})"""),
])
print("notebooks written:", sorted(x.name for x in NB.glob("*.ipynb")))
