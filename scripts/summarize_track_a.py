# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Aggregate Track A results (mean ± std over seeds) into a markdown table."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "json" / "track_a.json"
df = pd.DataFrame(json.loads(src.read_text()))
if "error" in df:
    fails = df[df.error.notna()]
    if len(fails):
        print("failures:\n", fails[["scenario", "seed", "method", "error"]].to_string())
    df = df[df.error.isna()]
metrics = [m for m in ["G_err", "rho_err", "kern_L1", "dLL", "AUC", "AP", "seconds"] if m in df]
lines = []
for sc, g in df.groupby("scenario", sort=False):
    lines.append(f"\n### {sc}  (seeds={g.seed.nunique()}, N_train≈{int(g.n_train.mean())})\n")
    lines.append("| method | " + " | ".join(metrics) + " |")
    lines.append("|---|" + "---|" * len(metrics))
    agg = g.groupby("method", sort=False)[metrics].agg(["mean", "std"])
    best = {m: (agg[(m, "mean")].max() if m in ("AUC", "AP") else agg[(m, "mean")].min()) for m in metrics}
    for meth, r in agg.iterrows():
        cells = []
        for m in metrics:
            mu, sd = r[(m, "mean")], r[(m, "std")]
            if np.isnan(mu):
                cells.append("--")
                continue
            fmt = f"{mu:.1f}" if m == "seconds" else f"{mu:.4f}" if m == "dLL" else f"{mu:.3f}"
            sdf = "" if np.isnan(sd) else (f" ± {sd:.1f}" if m == "seconds" else f" ± {sd:.4f}" if m == "dLL" else f" ± {sd:.3f}")
            cell = fmt + sdf
            if m != "seconds" and np.isclose(mu, best[m]):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {meth} | " + " | ".join(cells) + " |")
out = "\n".join(lines)
print(out)
(ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "tables" / "track_a.md").write_text(out, encoding="utf-8")
