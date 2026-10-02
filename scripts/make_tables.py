# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
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


def load_ept(ds):
    """All EPT/EPT-X runs of a dataset: the per-dataset file plus one-file-per-run results."""
    rows = list(load(f"ept_{ds}.json") or [])
    for f in sorted((J / "ept_runs").glob(f"{ds}__*.json")):
        rows.append(json.loads(f.read_text()))
    return rows


def fmt(mu, sd=None, d=3, bold=False):
    if mu is None or (isinstance(mu, float) and not np.isfinite(mu)):
        return "--"
    s = f"{mu:.{d}f}" + (f" $\\pm$ {sd:.{d}f}" if sd is not None and np.isfinite(sd) else "")
    return f"\\textbf{{{s}}}" if bold else s


# ------------------------------------------------------------------ table style
# booktabs tables: italic group headers, our models on a light band (\rowcolor{oursbg},
# defined in main.tex), best per column bold, second best underlined, s.d. as a small "±".
NICE = {
    "tick-ExpKern(MLE, best decay)": "Exp.\\ kernel MLE", "tick-SumExpKern(MLE, AGD)": "Sum-of-exp.\\ MLE",
    "tick-EM(nonparam)": "Hawkes EM (nonparam.)", "tick-ADM4": "ADM4", "tick-ConditionalLaw": "Conditional law",
    "NPHC(cumulants)": "NPHC (cumulants)", "MSX (ours)": "MSX ($R=2$)", "MSX (ours, R=2)": "MSX ($R=2$)",
    "MSX-auto (ours)": "MSX-auto", "MSX-R4 (ours, fixed Erlang-4)": "MSX ($R=4$)",
    "MSX-exp (ablation: no Erlang)": "MSX, exponential only", "EPT-TPP (ours)": "EPT", "EPT-X (ours)": "EPT-X",
    "MSX closed form (ours)": "MSX closed form", "event-study OLS": "Event-study OLS", "climatology": "Climatology",
}
OURS_ORDER = ["MSX ($R=2$)", "MSX ($R=4$)", "MSX, exponential only", "MSX-auto", "MSX closed form", "EPT", "EPT-X"]


def nice(m):
    return NICE.get(m, m)


def is_ours(m):
    return "ours" in m or "ablation" in m


def ranks(vals, higher=True):
    """Indices of the best and second-best finite values."""
    ok = [(v, i) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]
    ok.sort(reverse=higher)
    return (ok[0][1] if ok else None), (ok[1][1] if len(ok) > 1 else None)


def cell(mu, sd=None, d=3, rank=0):
    """rank: 1 best (bold), 2 second (underline)."""
    if mu is None or not np.isfinite(mu):
        return "--"
    s = f"{mu:.{d}f}".replace("-", "$-$")
    s = "\\textbf{" + s + "}" if rank == 1 else ("\\underline{" + s + "}" if rank == 2 else s)
    if sd is not None and np.isfinite(sd):
        s += "{\\scriptsize\\,$\\pm$" + f"{sd:.{d}f}" + "}"
    return s


def group_row(title, ncol):
    return f"\\addlinespace[3pt]\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{title}}}}} \\\\[1pt]"


def grouped_table(cols, groups, values, higher=True, d=3, exclude_rank=()):
    """cols: column headers; groups: [(title, [row names], shaded)]; values: row -> col -> (mu, sd)."""
    allrows = [r for _, rows, _ in groups for r in rows if r not in exclude_rank]
    rk = {}
    for c in cols:
        v = [values.get(r, {}).get(c, (np.nan, None))[0] for r in allrows]
        b, s = ranks(v, higher)
        rk[c] = (allrows[b] if b is not None else None, allrows[s] if s is not None else None)
    n = len(cols) + 1
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule", "& " + " & ".join(cols) + " \\\\", "\\midrule"]
    first = True
    for title, rows, shaded in groups:
        if not rows:
            continue
        lines.append(group_row(title, n) if not first else group_row(title, n).replace("\\addlinespace[3pt]", ""))
        first = False
        for r in rows:
            cells = []
            for c in cols:
                mu, sd = values.get(r, {}).get(c, (np.nan, None))
                cells.append(cell(mu, sd, d, 1 if rk[c][0] == r else 2 if rk[c][1] == r else 0))
            if r in exclude_rank:  # shown for reference only: greyed out
                cells = ["\\textcolor{gray}{" + c + "}" for c in cells]
                name = "\\textcolor{gray}{" + r + "}"
            else:
                name = r
            lines.append(("\\rowcolor{oursbg} " if shaded else "") + "\\quad " + name + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def mean_sd(xs):
    a = np.asarray(xs, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else None)


# ------------------------------------------------------------------ Track A
def table_a():
    """Compact Track A table: dLL (held-out deficit to the truth, 1e-3 nats/event) per scenario;
    the full metric table goes to the appendix (track_a_full.tex)."""
    rows = load("track_a.json")
    if rows:
        df = pd.DataFrame([r for r in rows if "error" not in r and "dLL" in r])
        scen = list(dict.fromkeys(df["scenario"]))
        heads = {s: s.split("-", 1)[0] + " " + s.split("-", 1)[1].replace("network10", "network").replace("powerlaw", "power law")
                 for s in scen}
        values = {}
        for (s, m), g in df.groupby(["scenario", "method"]):
            values.setdefault(nice(m), {})[heads[s]] = mean_sd(1e3 * g["dLL"].values)
        meths = list(dict.fromkeys(nice(m) for m in df["method"]))
        ours = [m for m in OURS_ORDER if m in meths]
        others = [m for m in meths if m not in ours]
        tab = grouped_table([heads[s] for s in scen], [("Classical estimators", others, False), ("Ours", ours, True)],
                            values, higher=False, d=2)
        (OUT / "track_a.tex").write_text(tab)
    table_a_full()


def table_a_full():
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
    (OUT / "track_a_full.tex").write_text("\n".join(lines).replace("tick-", "").replace("(ours", "(ours"))


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
    # our models: EPT-X / EPT (current architecture: rows carry a "model" key; older rows are
    # superseded provenance) and MSX + classical Hawkes from track_b_classical_<ds>.json
    ours = {}  # name -> ds -> list of test LL/event
    for ds in DS_B:
        groups = {}  # (name, config tag) -> [(val, test)]; one config per model is kept, chosen on validation
        for r in load_ept(ds):
            if "model" in r:
                nm = "EPT-X (ours)" if r["model"] == "EPT-X" else "EPT-TPP (ours)"
                groups.setdefault((nm, r.get("tag", "default")), []).append((r["val_ll"], r["ll_per_event"]))
        for nm in {k[0] for k in groups}:
            tag = max((k for k in groups if k[0] == nm), key=lambda k: np.mean([v for v, _ in groups[k]]))
            ours.setdefault(nm, {})[ds] = [t for _, t in groups[tag]]
        for r in load(f"track_b_classical_{ds}.json") or []:
            if "ll_per_event" in r:
                ours.setdefault(r["method"], {}).setdefault(ds, []).append(r["ll_per_event"])
    head = {d: ("StackOverflow" if d == "stackoverflow" else d.capitalize()) for d in DS_B}
    values = {}
    for m, v in PUBLISHED_S2P2.items():
        values[m.replace("IFTPP", "IntensityFree")] = {head[d]: (v[d], None) for d in DS_B}
    for m, dv in ours.items():
        values[nice(m)] = {head[d]: mean_sd(x) for d, x in dv.items()}
    pub = [m.replace("IFTPP", "IntensityFree") for m in PUBLISHED_S2P2]
    mine = [m for m in OURS_ORDER if m in values]
    classical = sorted(nice(m) for m in ours if not is_ours(m) and nice(m) not in mine)
    tab = grouped_table([head[d] for d in DS_B],
                        [("Neural TPPs, published (Chang et al., 2025)", pub, False),
                         ("Classical Hawkes, our runs", classical, False), ("Ours", mine, True)], values)
    (OUT / "track_b.tex").write_text(tab)
    seeds = {nice(m): {head[d]: len(x) for d, x in dv.items()} for m, dv in ours.items()}
    (OUT / "track_b_seeds.json").write_text(json.dumps(seeds, indent=1))


def table_ablation():
    """EPT-X component ablation: validation LL per event (selection protocol; one seed unless
    several final seeds exist). Rows are only filled where the run exists."""
    dsets = [("taxi", "Taxi"), ("taobao", "Taobao"), ("stackoverflow", "StackOverflow"), ("amazon", "Amazon"),
             ("lob_intc", "INTC")]
    variants = [("EPT (no renewal channel)", ("runs", "ept")), ("EPT-X", ("runs", "eptx")),
                ("\\quad + sharp atoms (orders to 1024)", ("abl", "sel_sharp")),
                ("\\quad + sharp atoms, 48 scales", ("abl", "sel_sharp48")),
                ("\\quad + deep encoder (3 layers)", ("abl", "sel_deep")),
                ("\\quad + sharp atoms + deep encoder", ("abl", "sel_sharpdeep")),
                ("\\quad + wide encoder (128 units)", ("abl", "sel_wide")),
                ("\\quad batch 256 (S2P2 setting)", ("abl", "sel_bs256"))]
    values = {}
    for name, (kind, tag) in variants:
        for ds, head in dsets:
            if kind == "runs":
                rows = [json.loads(f.read_text()) for f in sorted((J / "ept_runs").glob(f"{ds}__{tag}_s*.json"))]
                rows = [r for r in rows if r.get("tag") == tag]  # exact config tag (eptx != eptx_sharp)
                if ds.startswith("lob_"):  # LOBSTER EPT/EPT-X live in ept_lob_<tk>.json (+ per-run files)
                    want = "EPT-X" if tag == "eptx" else "EPT"
                    rows = [r for r in load_ept(ds) if r.get("model", "EPT") == want
                            and r.get("config", {}).get("epochs", 100) <= 300]
                vals = [r["val_ll"] for r in rows]
            else:
                f = J / f"eptx_ablation_{ds}_{tag}.json"
                vals = [json.loads(f.read_text())["val_ll"]] if f.exists() else []
            if vals:
                values.setdefault(name, {})[head] = mean_sd(vals)
    rows = [n for n, _ in variants if n in values]
    tab = grouped_table([h for _, h in dsets], [("Validation log-likelihood per event", rows, False)], values)
    (OUT / "ablation.tex").write_text(tab)


# ------------------------------------------------------------------ Track C
TK = ["aapl", "amzn", "goog", "intc", "msft"]


def _budget_select(runs):
    """runs: list of (budget, val_ll, test_ll). Keep the budget with the best mean validation LL."""
    by = {}
    for b, v, t in runs:
        by.setdefault(b, []).append((v, t))
    b = max(by, key=lambda k: np.mean([v for v, _ in by[k]]))
    return [t for _, t in by[b]], b


CLAMPED_IF = r"IntensityFree$^\ast$ (as shipped)"


def _lob_name(model):
    """EasyTPP's IntensityFree clamps gaps at 1e-5 s before scoring (not a likelihood of the
    observed gaps); the unclamped re-run is the IntensityFree entry, the shipped one is marked."""
    if model == "IntensityFree":
        return CLAMPED_IF
    if model.startswith("IntensityFree (min dt"):
        return "IntensityFree"
    return model


def track_c_results():
    """method -> tk -> list of test LL/event.  Neural models (EPT and EasyTPP) are reported at the
    epoch budget with the best validation likelihood, never averaged across budgets."""
    res, raw = {}, {}
    for tk in TK:
        for r in load(f"track_c_classical_lob_{tk}.json") or []:
            if "error" not in r:
                res.setdefault(r["method"], {}).setdefault(tk, []).append(r["ll_per_event"])
        for r in load_ept(f"lob_{tk}"):
            nm = "EPT-X (ours)" if r.get("model") == "EPT-X" else "EPT-TPP (ours)"
            budget = f'{r.get("config", {}).get("epochs", r["epochs"])}|{r.get("tag", "default")}'  # budget and config
            raw.setdefault((nm, tk), []).append((budget, r["val_ll"], r["ll_per_event"]))
    for r in load("easytpp_results.json") or []:
        if r["dataset"].startswith("lob_") and r.get("ll_per_event") is not None:
            raw.setdefault((_lob_name(r["model"]), r["dataset"][4:]), []).append(
                (r.get("max_epoch") or 100, r.get("val_ll", -np.inf), r["ll_per_event"]))
    for (m, tk), runs in raw.items():
        res.setdefault(m, {})[tk] = _budget_select(runs)[0]
    return res


def table_c():
    res = track_c_results()
    if not res:
        return
    values = {nice(m): {tk.upper(): mean_sd(v[tk]) for tk in v} for m, v in res.items()}
    neural = ["RMTPP", "THP", "SAHP", "NHP", "AttNHP", "S2P2", "IntensityFree", CLAMPED_IF]
    neural = [m for m in neural if m in values]
    mine = [m for m in OURS_ORDER if m in values]
    classical = [nice(m) for m in res if not is_ours(m) and nice(m) not in neural and nice(m) not in mine]
    tab = grouped_table([t.upper() for t in TK],
                        [("Classical Hawkes", classical, False), ("Neural TPPs (EasyTPP, our runs)", neural, False),
                         ("Ours", mine, True)], values, exclude_rank=(CLAMPED_IF,))
    (OUT / "track_c.tex").write_text(tab)


# ------------------------------------------------------------------ Track D
def table_d():
    r = load("track_d.json")
    if not r:
        return
    H = r["horizons_s"]
    n = len(H)
    lines = ["\\begin{tabular}{l" + "r" * (2 * n) + "}", "\\toprule",
             "& \\multicolumn{%d}{c}{RMSE of $\\log(1+N)$} & \\multicolumn{%d}{c}{QLIKE of realised variance} \\\\" % (n, n),
             "\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}" % (n + 1, n + 2, 2 * n + 1),
             "Method & " + " & ".join(f"{int(h / 60)} min" for h in H) + " & " + " & ".join(f"{int(h / 60)} min" for h in H)
             + " \\\\", "\\midrule"]
    act = r["activity"]
    keys = sorted(act, key=lambda k: "ours" in k)  # baselines first, ours last (shaded)
    rm = {k: np.mean(act[k]["rmse_log"], axis=1) for k in keys}
    ql = {k: np.mean(act[k]["qlike"], axis=1) for k in keys}
    rk = {}
    for name, tab in (("rm", rm), ("ql", ql)):
        for h in range(n):
            b, s = ranks([tab[k][h] for k in keys], higher=False)
            rk[name, h] = (keys[b], keys[s])
    for k in keys:
        c = [cell(rm[k][h], rank=1 if rk["rm", h][0] == k else 2 if rk["rm", h][1] == k else 0) for h in range(n)]
        c += [cell(ql[k][h], rank=1 if rk["ql", h][0] == k else 2 if rk["ql", h][1] == k else 0) for h in range(n)]
        lines.append(("\\rowcolor{oursbg} " if "ours" in k else "") + nice(k) + " & " + " & ".join(c) + " \\\\")
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
    # compact release x asset grids (a 66-row long table does not fit a page)
    kinds = ["CPI", "NFP", "PPI", "FOMC", "RETAIL", "GDP", "FOMCPC", "CLAIMS", "ECB", "ECBPC", "BOJ"]
    names = dict(RETAIL="Retail", FOMCPC="FOMC press conf.", CLAIMS="Claims", ECBPC="ECB press conf.", BOJ="BoJ")
    assets = list(v["per_type"]["CPI"]["z0"])
    se = v.get("mass_se", {})
    passes = {k: se.get(k, {}).get("mass", 0) >= 1.5 * se.get(f"PLACEBO_{k}", {}).get("mass", np.inf) for k in kinds}
    head = " & ".join(a.replace("XAUUSD", "Gold").replace("SPX", "S\\&P 500") for a in assets)

    def grid(cell, fname):
        lines = ["\\begin{tabular}{l" + "c" * len(assets) + "}", "\\toprule", f"Release & {head} \\\\", "\\midrule"]
        for typ in kinds:
            e = v["per_type"].get(typ, {})
            if typ == "GDP":
                lines.append("\\midrule")
            lab = names.get(typ, typ) + ("" if passes[typ] else "$^\\ast$")
            lines.append(lab + " & " + " & ".join(cell(e, a) for a in assets) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / fname).write_text("\n".join(lines))

    def t50_t90(r):  # t50 in seconds, t90 in minutes
        a, b = r["t50_total"], r["t90_total"]
        return "--" if not (np.isfinite(a) and np.isfinite(b)) else f"{a:.0f} / {b / 60:.1f}"

    grid(lambda e, a: t50_t90(e["z0"][a]) if a in e.get("z0", {}) else "--", "main_absorption.tex")
    def bp(x):
        x = round(float(x), 1)
        return "0.0" if x == 0 else f"{x:+.1f}"

    grid(lambda e, a: bp(e["zpos"][a]["drift_bps_final"]) if a in e.get("zpos", {}) else "--",
         "main_drift.tex")


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
        out.append(f"\\newcommand{{\\nWindows}}{{{r['n_windows']}}}")
        out.append(f"\\newcommand{{\\nEventsScored}}{{{r['n_events'] / 1e6:.1f}}}")
        dc = load("data_card.json")
        if dc:
            yrs = dc["years"].values()
            out.append(f"\\newcommand{{\\nEventsAll}}{{{sum(v['events'] for v in yrs) / 1e6:.1f}}}")
            out.append(f"\\newcommand{{\\nNewsWindows}}{{{sum(v['news'] for v in yrs)}}}"
                       f"\\newcommand{{\\nPlaceboWindows}}{{{sum(v['placebo'] for v in yrs)}}}")
    # specification history (superseded runs are kept, never deleted)
    sup = J / "superseded"
    v1 = sup / "main_2024_v1_placebo_failed.json"
    v2 = sup / "main_2024_v2_single_placebo.json"
    if v1.exists():
        a = json.loads(v1.read_text())
        out.append(f"\\newcommand{{\\vOnePlacebo}}{{{np.sum(a['placebo_mass_per_dim']):.0f}}}")
        out.append(f"\\newcommand{{\\vOneKSmax}}{{{max(g['ks'] for g in a['gof']):.2f}}}")
    if v2.exists():
        a = json.loads(v2.read_text())
        out.append(f"\\newcommand{{\\vTwoPlacebo}}{{{np.sum(a['placebo_mass_per_dim']):.1f}}}")
    nb = sup / "ept_lob_aapl_no_backbone.json"
    if nb.exists():
        rows = [r for r in json.loads(nb.read_text()) if r.get("epochs", 0) >= 100]
        out.append(f"\\newcommand{{\\eptNoBackbone}}{{{np.mean([r['ll_per_event'] for r in rows]):.2f}}}")
    l1 = load("l1_select_2024.json")
    if l1:
        rows = sorted(l1, key=lambda x: x["l1"])
        best = max(rows, key=lambda x: x["val_ll"])
        out.append(f"\\newcommand{{\\lOneSelected}}{{{best['l1']:.1f}}}")
        lines = ["\\begin{tabular}{cccc}", "\\toprule",
                 "$\\ell_1$ level & held-out LL (rel.) & mean release mass & mean placebo mass \\\\", "\\midrule"]
        for x in rows:
            rel = np.mean([v for k, v in x["mass"].items() if not k.startswith("PLACEBO")])
            plc = np.mean([v for k, v in x["mass"].items() if k.startswith("PLACEBO")])
            mark = "\\textbf" if x is best else ""
            lines.append(f"{x['l1']:.1f} & {mark}{{{x['val_ll'] - best['val_ll']:.1f}}} & {rel:.2f} & {plc:.2f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / "l1_selection.tex").write_text("\n".join(lines))
    # Track C: how many tickers our models win, margin over the best neural baseline
    wins, margins = 0, []
    neural = {"NHP", "S2P2", "THP", "RMTPP", "SAHP", "AttNHP", "IntensityFree"}
    tc = track_c_results()
    n_done = 0
    for tk in TK:
        mean = {m: float(np.mean(v[tk])) for m, v in tc.items() if tk in v}
        ours = [x for m, x in mean.items() if "ours" in m]
        nb = [x for m, x in mean.items() if m in neural]
        others = [x for m, x in mean.items() if "ours" not in m and m != CLAMPED_IF]
        if ours and len(nb) == len(neural):
            n_done += 1
            wins += max(ours) > max(others)
            margins.append(max(ours) - max(nb))
    if n_done:
        out.append(f"\\newcommand{{\\trackCwins}}{{{wins}}}\\newcommand{{\\trackCdone}}{{{n_done}}}")
        out.append(f"\\newcommand{{\\trackCmarginLo}}{{{min(margins):.2f}}}\\newcommand{{\\trackCmarginHi}}{{{max(margins):.2f}}}")
    d = load("track_d.json")
    if d:
        corr = {k: np.nanmean(v["drift_corr"], axis=1) for k, v in d["drift"].items()}
        ours = corr.get("MSX closed form (ours)")
        cands = [v for k, v in corr.items() if "ours" not in k and np.isfinite(v).any()]
        base = max(cands, key=lambda v: np.nanmean(v))  # martingale has undefined correlation
        if ours is not None:
            out.append(f"\\newcommand{{\\driftCorrOursOne}}{{{ours[0]:.2f}}}\\newcommand{{\\driftCorrBaseOne}}{{{base[0]:.2f}}}")
        rm = {k: np.mean(v["rmse_log"], axis=1) for k, v in d["activity"].items()}
        best = min(rm, key=lambda k: rm[k][-1])
        out.append(f"\\newcommand{{\\bestActivityModel}}{{{best}}}")
        out.append(f"\\newcommand{{\\nTestReleases}}{{{d['n_test']}}}")
    sf = OUT / "track_b_seeds.json"
    if sf.exists():  # seed counts quoted in the Track B caption
        s = json.loads(sf.read_text()).get("EPT-X", {})
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
        txt = "; ".join(f"{d} {words.get(n, n)}" for d, n in s.items())
        out.append(f"\\newcommand{{\\eptxSeeds}}{{{txt}}}")
    rhos = []
    for y in ("2022", "2023", "2024", "2025", "2026"):
        v = load(f"main_{y}.json")
        if v:
            rhos.append(f"{v['rho']:.2f}")
    if rhos:
        out.append(f"\\newcommand{{\\rhoByYear}}{{{', '.join(rhos)}}}")
    (ROOT / "manuscript" / "numbers.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    for f in (table_a, table_b, table_ablation, table_c, table_d, table_main, numbers):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # keep going; report
            print("FAILED", f.__name__, repr(e))
