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
    if abs(mu) >= 1000:  # large magnitudes (e.g. Retweet RMSE in seconds): no decimals
        d = 0
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
    metrics = [("G_err", "$G$ error", 1, False, 3), ("rho_err", "$\\rho$ error", 1, False, 3),
               ("kern_L1", "kernel $L^1$", 1, False, 3), ("dLL", "dLL ($10^{-3}$)", 1e3, False, 2),
               ("AUC", "edge AUC", 1, True, 3)]
    names = {"S1-exp": "S1: exponential", "S2-powerlaw": "S2: power law", "S3-hump": "S3: hump",
             "S4-multiscale": "S4: multiscale network", "S5-network10": "S5: 10-dim.\\ power-law network"}
    ncol = len(metrics) + 1
    lines = ["\\begin{tabular}{l" + "r" * len(metrics) + "}", "\\toprule",
             "& " + " & ".join(h for _, h, _, _, _ in metrics) + " \\\\", "\\midrule"]
    first = True
    for sc, g in df.groupby("scenario", sort=False):
        agg = g.groupby("method", sort=False)
        meths = list(agg.groups)
        meths = [m for m in meths if not is_ours(m)] + [m for m in meths if is_ours(m)]
        stats = {m: {k: (agg.get_group(m)[k].dropna() * s if k in g else pd.Series(dtype=float))
                     for k, _, s, _, _ in metrics} for m in meths}
        rk = {}
        for k, _, _, hi, _ in metrics:
            b, s2 = ranks([stats[m][k].mean() if len(stats[m][k]) else np.nan for m in meths], hi)
            rk[k] = (meths[b] if b is not None else None, meths[s2] if s2 is not None else None)
        hdr = group_row(names.get(sc, sc), ncol)
        lines.append(hdr.replace("\\addlinespace[3pt]", "") if first else hdr)
        first = False
        for m in meths:
            cells = []
            for k, _, _, hi, d in metrics:
                x = stats[m][k]
                if k == "AUC" and sc in ("S1-exp", "S2-powerlaw", "S3-hump"):
                    cells.append("n/a")  # complete true graph: AUC undefined
                elif not len(x):
                    cells.append("--")
                else:
                    rank = 1 if rk[k][0] == m else 2 if rk[k][1] == m else 0
                    cells.append(cell(x.mean(), x.std(ddof=1) if len(x) > 1 else None, d, rank))
            shade = "\\rowcolor{oursbg} " if is_ours(m) else ""
            lines.append(f"{shade}\\quad {nice(m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "track_a_full.tex").write_text("\n".join(lines))


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
# Chang et al. (2025), Table 2(b) next-event time RMSE and 2(c) next-mark accuracy (%), means over
# five seeds, transcribed from paper/chang2024_s2p2_state_space_pp.pdf (page 9)
PUBLISHED_RMSE = {
    "RMTPP": dict(amazon=0.338, retweet=16488, taxi=0.283, taobao=0.126, stackoverflow=1.049),
    "SAHP": dict(amazon=0.335, retweet=16102, taxi=0.290, taobao=0.126, stackoverflow=1.031),
    "THP": dict(amazon=0.332, retweet=16268, taxi=0.285, taobao=0.125, stackoverflow=1.033),
    "IFTPP": dict(amazon=0.327, retweet=16625, taxi=0.362, taobao=0.125, stackoverflow=1.340),
    "MHP": dict(amazon=0.329, retweet=16109, taxi=0.284, taobao=0.126, stackoverflow=1.046),
    "NHP": dict(amazon=0.339, retweet=15911, taxi=0.282, taobao=0.126, stackoverflow=1.019),
    "AttNHP": dict(amazon=2.656, retweet=16171, taxi=1.739, taobao=0.130, stackoverflow=1.256),
    "S2P2": dict(amazon=0.327, retweet=15987, taxi=0.281, taobao=0.126, stackoverflow=1.014),
}
PUBLISHED_ACC = {
    "RMTPP": dict(amazon=30.8, retweet=53.4, taxi=91.4, taobao=60.9, stackoverflow=45.6),
    "SAHP": dict(amazon=32.4, retweet=57.5, taxi=91.4, taobao=60.5, stackoverflow=44.7),
    "THP": dict(amazon=34.6, retweet=60.2, taxi=91.4, taobao=60.0, stackoverflow=46.6),
    "IFTPP": dict(amazon=35.9, retweet=50.4, taxi=91.8, taobao=61.0, stackoverflow=45.6),
    "MHP": dict(amazon=35.1, retweet=60.0, taxi=91.4, taobao=60.7, stackoverflow=46.5),
    "NHP": dict(amazon=39.4, retweet=61.4, taxi=92.9, taobao=61.5, stackoverflow=47.1),
    "AttNHP": dict(amazon=38.9, retweet=60.7, taxi=92.6, taobao=61.3, stackoverflow=48.2),
    "S2P2": dict(amazon=40.7, retweet=61.3, taxi=93.1, taobao=61.1, stackoverflow=47.5),
}
_PRED = {}  # filled by table_b: model -> ds -> (rmse list, acc list) of the validation-selected config


def table_b():
    # our models: EPT-X / EPT (current architecture: rows carry a "model" key; older rows are
    # superseded provenance) and MSX + classical Hawkes from track_b_classical_<ds>.json
    ours = {}  # name -> ds -> list of test LL/event
    for ds in DS_B:
        groups = {}  # (name, config tag) -> [(val, test)]; one config per model is kept, chosen on validation
        for r in load_ept(ds):
            if "model" in r:
                nm = "EPT-X (ours)" if r["model"] == "EPT-X" else "EPT-TPP (ours)"
                groups.setdefault((nm, r.get("tag", "default")), []).append(
                    (r["val_ll"], r["ll_per_event"], r.get("rmse"), r.get("acc")))
        for nm in {k[0] for k in groups}:
            tag = max((k for k in groups if k[0] == nm), key=lambda k: np.mean([g[0] for g in groups[k]]))
            ours.setdefault(nm, {})[ds] = [g[1] for g in groups[tag]]
            _PRED.setdefault(nm, {})[ds] = ([g[2] for g in groups[tag] if g[2] is not None],
                                            [100 * g[3] for g in groups[tag] if g[3] is not None])
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


def table_pred():
    """Next-event prediction on the public benchmarks: time RMSE (lower better) and mark accuracy
    (%, higher better); published numbers vs EPT-X (validation-selected configuration)."""
    head = {d: ("StackOverflow" if d == "stackoverflow" else d.capitalize()) for d in DS_B}
    pub = [m.replace("IFTPP", "IntensityFree") for m in PUBLISHED_RMSE]
    for name, published, idx, higher, d in (("rmse", PUBLISHED_RMSE, 0, False, 3), ("acc", PUBLISHED_ACC, 1, True, 1)):
        values = {m.replace("IFTPP", "IntensityFree"): {head[k]: (v[k], None) for k in DS_B} for m, v in published.items()}
        x = _PRED.get("EPT-X (ours)", {})
        values["EPT-X"] = {head[k]: mean_sd(x[k][idx]) for k in x if len(x[k][idx])}
        tab = grouped_table([head[k] for k in DS_B], [("Published (Chang et al., 2025)", pub, False),
                                                      ("Ours", ["EPT-X"], True)], values, higher=higher, d=d)
        (OUT / f"track_b_{name}.tex").write_text(tab)


def table_cost():
    """Training cost per epoch and parameter count (EPT-X: exact likelihood; EasyTPP baselines:
    Monte-Carlo compensator, trained for their full epoch budget). Device read from the queue log."""
    logs = ROOT / "results" / "logs"

    def device(job):
        hits = list(logs.glob(f"*/{job}.log"))
        return ("GPU" if hits[0].parent.name == "gpu" else "CPU (4 thr.)") if hits else "?"

    rows = []
    for ds, label, job in (("taxi", "Taxi", "eptx_taxi_s0"), ("lob_aapl", "LOBSTER AAPL", "eptx_lob_aapl_sharp_s0")):
        tag = "eptx" if ds == "taxi" else "eptx_sharp"
        f = J / "ept_runs" / f"{ds}__{tag}_s0.json"
        if f.exists():
            r = json.loads(f.read_text())
            rows.append((label, "EPT-X (ours)", device(job), r["seconds"] / r["epochs"], r["n_params"], "exact"))
        for m in ("S2P2", "NHP", "AttNHP", "THP", "IntensityFree"):
            cand = [r for r in load("easytpp_results.json") or [] if r["dataset"] == ds and r["model"] == m
                    and r.get("seconds") and (r.get("max_epoch") in (None, 100, 300))]
            if cand:
                r = cand[0]
                rows.append((label, m, "GPU", r["seconds"] / (r.get("max_epoch") or 300), r["n_params"],
                             "exact" if m == "IntensityFree" else "Monte Carlo"))
    lines = ["\\begin{tabular}{llllrr}", "\\toprule",
             "Data & Model & Compensator & Device & s / epoch & Parameters \\\\", "\\midrule"]
    last = None
    for label, m, dev, spe, npar, comp in rows:
        if last is not None and label != last:
            lines.append("\\midrule")
        shade = "\\rowcolor{oursbg} " if "ours" in m else ""
        lines.append(f"{shade}{label if label != last else ''} & {m.replace(' (ours)', '')} & {comp} & {dev} & "
                     f"{spe:.1f} & {npar / 1e3:.0f}k \\\\")
        last = label
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "cost.tex").write_text("\n".join(lines))


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
                ("\\quad + attention encoder (causal Transformer)", ("abl", "sel_attn")),
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
                            and r.get("tag", tag) == tag  # base configuration only (untagged = base)
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
            # budget and config; untagged rows predate tags and are the base configuration
            tag = r.get("tag", "eptx" if r.get("model") == "EPT-X" else "ept")
            budget = f'{r.get("config", {}).get("epochs", r["epochs"])}|{tag}'
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
    gains = [np.mean(tc["EPT-X (ours)"][tk]) - np.mean(tc["EPT-TPP (ours)"][tk]) for tk in TK
             if tk in tc.get("EPT-X (ours)", {}) and tk in tc.get("EPT-TPP (ours)", {})]
    if gains:  # renewal-channel gain of EPT-X over EPT on the order books
        out.append(f"\\newcommand{{\\eptxGainLo}}{{{min(gains):.2f}}}\\newcommand{{\\eptxGainHi}}{{{max(gains):.2f}}}")
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
    for f in (table_a, table_b, table_pred, table_cost, table_ablation, table_c, table_d, table_main, numbers):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # keep going; report
            print("FAILED", f.__name__, repr(e))
