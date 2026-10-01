# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Build dashboard/index.html (self-contained) from results/json. Re-run after new results."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "results" / "json"


def load(n):
    f = J / n
    return json.loads(f.read_text()) if f.exists() else None


pooled = load("main_pooled.json")
data = {"t": pooled["t_grid"], "rho": pooled["rho"], "relax": pooled["relaxation_time_s"], "types": {},
        "mass": pooled["mass_se"], "years": {}, "trackA": {}, "trackC": {}}
for typ, e in pooled["per_type"].items():
    if typ.startswith("PLACEBO"):
        continue
    z0, zp, zn = e.get("z0", {}), e.get("zpos", {}), e.get("zneg", {})
    data["types"][typ] = {a: dict(tot=[round(v, 6) for v in r["activity_curve_total"]],
                                  dir=[round(v, 6) for v in r["activity_curve_direct"]],
                                  t50d=r["t50_direct"], t50=r["t50_total"], t90=r["t90_total"],
                                  amp=r["amplification"] if np.isfinite(r["amplification"]) else None,
                                  xd=r["extra_events_direct"], xt=r["extra_events_total"],
                                  up=zp.get(a, {}).get("drift_bps_final"), dn=zn.get(a, {}).get("drift_bps_final"))
                          for a, r in z0.items()}
for y in ("2022", "2023", "2024", "2025", "2026"):
    r = load(f"main_{y}.json")
    if r:
        data["years"][y] = dict(rho=r["rho"], windows=r["n_windows"], events=r["n_events"])
ta = load("track_a.json") or []
for r in ta:
    if "error" in r or "dLL" not in r:
        continue
    d = data["trackA"].setdefault(r["scenario"], {}).setdefault(r["method"], [])
    d.append(r["dLL"])
data["trackA"] = {s: {m: float(np.mean(v)) for m, v in ms.items()} for s, ms in data["trackA"].items()}
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import TK, track_c_results  # same validation-best budget selection as the paper

tc = track_c_results()
for tk in TK:
    data["trackC"][tk.upper()] = {m: float(np.mean(v[tk])) for m, v in tc.items() if tk in v}

html = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
out = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
(ROOT / "dashboard" / "index.html").write_text(out, encoding="utf-8")
print("dashboard written", round(len(out) / 1024), "KB")
