"""Generate LaTeX result tables for the manuscript from results/json/*.json.

Every number in the paper's result tables comes from this script; missing inputs produce
a table row marked "pending" rather than a made-up value.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "results" / "json"
OUT = ROOT / "manuscript" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    f = J / name
    return json.loads(f.read_text()) if f.exists() else None


def fmt(mu, sd=None, d=3, bold=False):
    if mu is None or (isinstance(mu, float) and not np.isfinite(mu)):
        return "--"
    s = f"{mu:.{d}f}" + (f" $\\pm$ {sd:.{d}f}" if sd is not None and np.isfinite(sd) else "")
    return f"\\textbf{{{s}}}" if bold else s


# ------------------------------------------------------------------ Track A
def table_a():
    rows = load("track_a.json")
    if not rows:
        return
    df = pd.DataFrame([r for r in rows if "error" not in r])
    lines = ["\\begin{tabular}{llccccc}", "\\toprule",
             "Scenario & Method & $G$ err. & $\\rho$ err. & kernel $L^1$ & dLL ($10^{-3}$) & AUC \\\\", "\\midrule"]
    for sc, g in df.groupby("scenario", sort=False):
        num = g[["method"] + [c for c in ("G_err", "rho_err", "kern_L1", "dLL", "AUC") if c in g]]
        agg = num.groupby("method", sort=False).agg(["mean", "std"])
        best_dll = agg[("dLL", "mean")].min() if ("dLL", "mean") in agg else np.nan
        first = True
        for meth, r in agg.iterrows():
            dll = r.get(("dLL", "mean"), np.nan)
            cells = [fmt(r[("G_err", "mean")], r[("G_err", "std")]),
                     fmt(r[("rho_err", "mean")], r[("rho_err", "std")]),
                     fmt(r.get(("kern_L1", "mean"), np.nan)),
                     fmt(1e3 * dll if np.isfinite(dll) else np.nan, 1e3 * r.get(("dLL", "std"), np.nan), 2,
                         bold=np.isfinite(dll) and np.isclose(dll, best_dll)),
                     fmt(r.get(("AUC", "mean"), np.nan))]
            lines.append(f"{sc if first else ''} & {meth} & " + " & ".join(cells) + " \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (OUT / "track_a.tex").write_text("\n".join(lines))


# ------------------------------------------------------------------ Track B
PUBLISHED_S2P2 = {  # Chang et al. (2025), Table 2(a): test LL per event (5 seeds)
    "RMTPP": dict(amazon=-2.136, retweet=-7.098, taxi=0.346, taobao=1.003, stackoverflow=-2.480),
    "SAHP": dict(amazon=-2.074, retweet=-6.708, taxi=0.298, taobao=1.168, stackoverflow=-2.341),
    "THP": dict(amazon=-2.096, retweet=-6.659, taxi=0.372, taobao=0.790, stackoverflow=-2.338),
    "IFTPP": dict(amazon=0.496, retweet=-10.344, taxi=0.453, taobao=1.318, stackoverflow=-2.233),
    "MHP": dict(amazon=-2.091, retweet=-6.564, taxi=0.370, taobao=0.636, stackoverflow=-2.346),
    "NHP": dict(amazon=0.129, retweet=-6.348, taxi=0.514, taobao=1.157, stackoverflow=-2.241),
    "AttNHP": dict(amazon=0.484, retweet=-6.499, taxi=0.493, taobao=1.259, stackoverflow=-2.194),
    "S2P2": dict(amazon=0.781, retweet=-6.365, taxi=0.522, taobao=1.304, stackoverflow=-2.163),
}
DS_B = ["amazon", "retweet", "taxi", "taobao", "stackoverflow"]


def table_b():
    ours = {}
    for ds in DS_B:
        rows = load(f"ept_{ds}.json")
        if rows:
            v = np.array([r["ll_per_event"] for r in rows if r.get("epochs", 0) >= 100])
            if len(v):
                ours[ds] = (v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, len(v))
    best = {ds: max([PUBLISHED_S2P2[m][ds] for m in PUBLISHED_S2P2] + ([ours[ds][0]] if ds in ours else []))
            for ds in DS_B}
    lines = ["\\begin{tabular}{l" + "c" * len(DS_B) + "}", "\\toprule",
             "Model & " + " & ".join(d.capitalize() for d in DS_B) + " \\\\", "\\midrule"]
    for m, v in PUBLISHED_S2P2.items():
        lines.append(m + "$^\\dagger$ & " + " & ".join(fmt(v[d], bold=np.isclose(v[d], best[d])) for d in DS_B) + " \\\\")
    cells = []
    for d in DS_B:
        if d in ours:
            mu, sd, n = ours[d]
            cells.append(fmt(mu, sd, bold=np.isclose(mu, best[d])) + f"$_{{({n})}}$")
        else:
            cells.append("pending")
    lines += ["\\midrule", "EPT-TPP (ours, exact LL) & " + " & ".join(cells) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    (OUT / "track_b.tex").write_text("\n".join(lines))


# ------------------------------------------------------------------ Track C
TK = ["aapl", "amzn", "goog", "intc", "msft"]


def table_c():
    res = {}  # method -> tk -> list
    for tk in TK:
        for r in load(f"track_c_classical_lob_{tk}.json") or []:
            if "error" not in r:
                res.setdefault(r["method"], {}).setdefault(tk, []).append(r["ll_per_event"])
        for r in load(f"ept_lob_{tk}.json") or []:
            res.setdefault("EPT-TPP (ours)", {}).setdefault(tk, []).append(r["ll_per_event"])
    for r in load("easytpp_results.json") or []:
        if r["dataset"].startswith("lob_") and r.get("ll_per_event") is not None:
            res.setdefault(r["model"], {}).setdefault(r["dataset"][4:], []).append(r["ll_per_event"])
    if not res:
        return
    best = {tk: max((np.mean(v[tk]) for v in res.values() if tk in v), default=np.nan) for tk in TK}
    lines = ["\\begin{tabular}{l" + "c" * len(TK) + "}", "\\toprule",
             "Model & " + " & ".join(t.upper() for t in TK) + " \\\\", "\\midrule"]
    for m, v in res.items():
        cells = []
        for tk in TK:
            if tk in v:
                a = np.array(v[tk])
                cells.append(fmt(a.mean(), a.std(ddof=1) if len(a) > 1 else None, bold=np.isclose(a.mean(), best[tk])))
            else:
                cells.append("--")
        lines.append(m + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "track_c.tex").write_text("\n".join(lines))


# ------------------------------------------------------------------ Track D
def table_d():
    r = load("track_d.json")
    if not r:
        return
    H = r["horizons_s"]
    lines = ["\\begin{tabular}{l" + "c" * (2 * len(H)) + "}", "\\toprule",
             "& \\multicolumn{%d}{c}{RMSE $\\log(1+N)$} & \\multicolumn{%d}{c}{QLIKE (RV)} \\\\" % (len(H), len(H)),
             "Method & " + " & ".join(f"{int(h / 60)}m" for h in H) * 1 + " & " + " & ".join(f"{int(h / 60)}m" for h in H) + " \\\\",
             "\\midrule"]
    act = r["activity"]
    rm = {k: np.mean(v["rmse_log"], axis=1) for k, v in act.items()}
    ql = {k: np.mean(v["qlike"], axis=1) for k, v in act.items()}
    for k in act:
        c1 = [fmt(rm[k][h], bold=np.isclose(rm[k][h], min(x[h] for x in rm.values()))) for h in range(len(H))]
        c2 = [fmt(ql[k][h], bold=np.isclose(ql[k][h], min(x[h] for x in ql.values()))) for h in range(len(H))]
        lines.append(k + " & " + " & ".join(c1 + c2) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "track_d.tex").write_text("\n".join(lines))


# ------------------------------------------------------------------ main study
def table_main():
    years = ["2022", "2023", "2024", "2025", "2026", "pooled"]
    res = {y: load(f"main_{y}.json") for y in years}
    res = {y: v for y, v in res.items() if v}
    if not res:
        return
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Panel & windows & events & $\\rho(G)$ & slowest echo mode (s) \\\\", "\\midrule"]
    for y, v in res.items():
        lines.append(f"{y} & {v['n_windows']} & {v['n_events']:,} & {v['rho']:.3f} & {v['relaxation_time_s']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "main_overview.tex").write_text("\n".join(lines))
    # absorption table from pooled (or latest available) fit
    key = "pooled" if "pooled" in res else list(res)[-1]
    v = res[key]
    lines = ["\\begin{tabular}{llccccc}", "\\toprule",
             "Release & Asset & $t_{50}$ direct & $t_{50}$ echo-corr. & $t_{90}$ echo-corr. & amplification & drift$_{+1\\sigma}$ (bp) \\\\",
             "\\midrule"]
    for typ in ["CPI", "NFP", "FOMC", "FOMCPC", "PPI", "RETAIL", "GDP", "CLAIMS", "ECB", "ECBPC", "BOJ"]:
        e = v["per_type"].get(typ, {})
        z0, zp = e.get("z0", {}), e.get("zpos", {})
        for asset, r in z0.items():
            dp = zp.get(asset, {}).get("drift_bps_final", np.nan)
            lines.append(f"{typ} & {asset} & {fmt(r['t50_direct'], d=1)} & {fmt(r['t50_total'], d=1)} & "
                         f"{fmt(r['t90_total'], d=1)} & {fmt(r['amplification'], d=2)} & {fmt(dp, d=2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"main_absorption.tex").write_text("\n".join(lines))


def numbers():
    """LaTeX macros for every number quoted in the prose (computed, never typed)."""
    out = []
    r = load("main_pooled.json")
    if r:
        out.append(f"\\newcommand{{\\rhoPooled}}{{{r['rho']:.2f}}}")
        out.append(f"\\newcommand{{\\relaxPooled}}{{{r['relaxation_time_s'] / 60:.0f}}}")
        se = r["mass_se"]
        kinds = [k for k in se if not k.startswith("PLACEBO")]
        ratio = {k: se[k]["mass"] / max(se[f"PLACEBO_{k}"]["mass"], 1e-9) for k in kinds}
        for k in ("CPI", "NFP", "PPI", "FOMC", "RETAIL"):
            out.append(f"\\newcommand{{\\placebo{k.capitalize()}}}{{{ratio[k]:.1f}}}")
        out.append(f"\\newcommand{{\\nPassPlacebo}}{{{sum(v >= 1.5 for v in ratio.values())}}}")
        out.append(f"\\newcommand{{\\nKinds}}{{{len(kinds)}}}")
        sig = ("CPI", "NFP", "PPI", "FOMC")
        t50d = [x["t50_direct"] for k in sig for x in r["per_type"][k]["z0"].values()]
        t50e = [x["t50_total"] for k in sig for x in r["per_type"][k]["z0"].values()]
        t90e = [x["t90_total"] for k in sig for x in r["per_type"][k]["z0"].values()]
        out.append(f"\\newcommand{{\\tDirMed}}{{{np.median(t50d):.1f}}}")
        out.append(f"\\newcommand{{\\tEchoLo}}{{{min(t50e):.0f}}}\\newcommand{{\\tEchoHi}}{{{max(t50e):.0f}}}")
        out.append(f"\\newcommand{{\\tNinetyLo}}{{{min(t90e) / 60:.0f}}}\\newcommand{{\\tNinetyHi}}{{{max(t90e) / 60:.0f}}}")
        out.append(f"\\newcommand{{\\ksMax}}{{{max(g['ks'] for g in r['gof']):.3f}}}")
        ed = [x for x in r["ed_reject_rate_per_dim"] if x is not None]
        out.append(f"\\newcommand{{\\edLo}}{{{100 * min(ed):.0f}}}\\newcommand{{\\edHi}}{{{100 * max(ed):.0f}}}")
        out.append(f"\\newcommand{{\\gapMaxPooled}}{{{max(x['gap'] for x in r['fit_reports']):.4f}}}")
    rhos = []
    for y in ("2022", "2023", "2024", "2025", "2026"):
        v = load(f"main_{y}.json")
        if v:
            rhos.append(f"{v['rho']:.2f}")
    if rhos:
        out.append(f"\\newcommand{{\\rhoByYear}}{{{', '.join(rhos)}}}")
    (ROOT / "manuscript" / "numbers.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    for f in (table_a, table_b, table_c, table_d, table_main, numbers):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # keep going; report
            print("FAILED", f.__name__, repr(e))
