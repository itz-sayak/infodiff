"""Paper figures from results/json (static PDFs, light theme).

Design rules (dataviz skill): small multiples instead of >3 categorical colours; validated
slots blue #2a78d6 / orange #eb6834 / aqua #1baf7a (aqua always direct-labelled); thin
2px lines; recessive grid; text in ink colours, never series colours; one y-axis per panel.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "results" / "json"
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c8c7c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(name):
    f = J / name
    return json.loads(f.read_text()) if f.exists() else None


def main_result():
    for tag in ("pooled", "2025", "2024", "2026", "2023", "2022"):
        r = load(f"main_{tag}.json")
        if r:
            return tag, r
    return None, None


def fig_absorption(tag, r, typ="CPI"):
    t = np.asarray(r["t_grid"])
    e = r["per_type"].get(typ, {}).get("z0", {})
    if not e:
        return
    assets = list(e)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
    for ax, asset in zip(axes.ravel(), assets):
        row = e[asset]
        tot, dirc = np.asarray(row["activity_curve_total"]), np.asarray(row["activity_curve_direct"])
        ax.loglog(t, np.maximum(tot, 1e-6), color=BLUE)
        ax.loglog(t, np.maximum(dirc, 1e-6), color=ORANGE)
        ax.set_title(asset, color=INK, fontsize=8, loc="left")
        ax.axvline(row["t50_total"], color=BLUE, lw=0.8, ls=":")
        ax.axvline(row["t50_direct"], color=ORANGE, lw=0.8, ls=":")
        ax.set_ylim(bottom=max(1e-4, np.nanmax(tot) * 1e-4))
        sec = lambda v: f"{v:.1f}s" if v < 10 else f"{v:.0f}s"
        amp = row["amplification"]
        amp_s = f"amplif. {amp:.1f}x" if np.isfinite(amp) else "amplif. n/a (direct < 0.5 events)"
        ax.text(0.98, 0.95, f"$t_{{50}}$ echo {sec(row['t50_total'])}\n$t_{{50}}$ direct {sec(row['t50_direct'])}\n{amp_s}",
                transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=6.5)
    for ax in axes.ravel()[len(assets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("seconds after release")
    for ax in axes[:, 0]:
        ax.set_ylabel("excess intensity (events/s)")
    h = [plt.Line2D([], [], color=BLUE), plt.Line2D([], [], color=ORANGE)]
    fig.legend(h, ["echo-corrected response", "direct news kernel"], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"absorption_{typ}_{tag}.pdf")
    plt.close(fig)


def fig_t50(tag, r):
    rows = []
    for typ, e in r["per_type"].items():
        if typ == "PLACEBO":
            continue
        for asset, row in e.get("z0", {}).items():
            if np.isfinite(row["t50_direct"]) and np.isfinite(row["t50_total"]):
                rows.append((f"{typ} · {asset}", row["t50_direct"], row["t50_total"]))
    if not rows:
        return
    rows.sort(key=lambda x: x[2])
    fig, ax = plt.subplots(figsize=(4.8, 0.16 * len(rows) + 0.8))
    y = np.arange(len(rows))
    for k, (_, a, b) in enumerate(rows):
        ax.plot([a, b], [k, k], color="#d8d7d2", lw=1.2, zorder=1)
    ax.scatter([r_[1] for r_ in rows], y, color=ORANGE, s=16, zorder=2, label="direct kernel")
    ax.scatter([r_[2] for r_ in rows], y, color=BLUE, s=16, zorder=3, label="echo-corrected")
    ax.set_yticks(y, [r_[0] for r_ in rows], fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("time to absorb 50% of the release response (s)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / f"t50_{tag}.pdf")
    plt.close(fig)


def fig_drift(tag, r, typ="CPI"):
    t = np.asarray(r["t_grid"])
    e = r["per_type"].get(typ, {})
    zp, zn = e.get("zpos", {}), e.get("zneg", {})
    if not zp:
        return
    assets = list(zp)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8), sharex=True)
    for ax, asset in zip(axes.ravel(), assets):
        ax.semilogx(t, zp[asset]["drift_bps_curve"], color=BLUE)
        if asset in zn:
            ax.semilogx(t, zn[asset]["drift_bps_curve"], color=ORANGE)
        ax.axhline(0, color="#c8c7c2", lw=0.8)
        ax.set_title(asset, color=INK, fontsize=8, loc="left")
    for ax in axes.ravel()[len(assets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("seconds after release")
    for ax in axes[:, 0]:
        ax.set_ylabel("expected drift (bp)")
    h = [plt.Line2D([], [], color=BLUE), plt.Line2D([], [], color=ORANGE)]
    fig.legend(h, ["+1 sd surprise", "-1 sd surprise"], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"drift_{typ}_{tag}.pdf")
    plt.close(fig)


def fig_rho_by_year():
    yrs, rho = [], []
    for y in ("2022", "2023", "2024", "2025", "2026"):
        r = load(f"main_{y}.json")
        if r:
            yrs.append(int(y))
            rho.append(r["rho"])
    if len(yrs) < 2:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(yrs, rho, color=BLUE, marker="o", ms=5)
    for x, v in zip(yrs, rho):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", color=INK2, fontsize=7)
    ax.set_xticks(yrs)
    ax.set_ylabel(r"spectral radius $\rho(G)$")
    fig.savefig(OUT / "rho_by_year.pdf")
    plt.close(fig)


def fig_placebo(tag, r):
    se = r.get("mass_se", {})
    if not se:
        return
    kinds = [k for k in se if not k.startswith("PLACEBO")]
    x = np.arange(len(kinds))
    real = [se[k]["mass"] for k in kinds]
    plc = [se.get(f"PLACEBO_{k}", {}).get("mass", np.nan) for k in kinds]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(x - 0.2, real, 0.38, color=BLUE, edgecolor="white", linewidth=1, label="release")
    ax.bar(x + 0.2, plc, 0.38, color=GRAY, edgecolor="white", linewidth=1, label="matched placebo (same clock time)")
    ax.set_xticks(x, kinds, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("direct news-kernel mass\n(extra events per release)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / f"placebo_{tag}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    tag, r = main_result()
    if r:
        for typ in ("CPI", "NFP", "FOMC"):
            fig_absorption(tag, r, typ)
            fig_drift(tag, r, typ)
        fig_t50(tag, r)
        fig_placebo(tag, r)
    fig_rho_by_year()
    print(sorted(p.name for p in OUT.glob("*.pdf")))
