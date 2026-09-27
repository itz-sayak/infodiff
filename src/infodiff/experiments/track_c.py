"""Track C — LOBSTER order-book event streams: classical Hawkes family under the
EasyTPP / S2P2 protocol (test log-likelihood per scored event; the first event of each
sequence conditions, the compensator runs from t_0 to t_N).

Each sequence becomes one window whose first event is burn-in history, t0 = t_0 + eps,
t1 = t_N + eps.  Baselines are constant (the train mean of window intercepts for MSX),
so every method is scored with the same kind of stationary model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..baselines import classical as C
from ..events.data import EventData
from ..models.dictionary import PhaseTypeDictionary
from .track_b import load_split

EPS = 1e-9


def seqs_to_eventdata(seqs, n_types: int) -> EventData:
    times, types, ptr, t0, t1 = [], [], [0], [], []
    offset = 0.0
    for dts, marks in seqs:
        t = np.cumsum(np.asarray(dts, float))
        t = t - t[0] + offset
        times.append(t)
        types.append(np.asarray(marks, int))
        ptr.append(ptr[-1] + len(t))
        t0.append(t[0] + EPS)
        t1.append(t[-1] + EPS)
        offset = t[-1] + 1e4  # windows are independent; keep times increasing
    return EventData(times=np.concatenate(times), types=np.concatenate(types).astype(np.int32),
                     wptr=np.asarray(ptr), t0=np.asarray(t0), t1=np.asarray(t1), n_dims=n_types)


def n_scored(data: EventData) -> int:
    return int(data.in_likelihood().sum())


def run(dataset: str, out: Path, methods=("MSX-auto", "MSX", "ExpKern", "SumExp", "EM", "ADM4", "CondLaw")) -> list:
    tr, M = load_split(dataset, "train")
    va, _ = load_split(dataset, "validation")
    te, _ = load_split(dataset, "test")
    d_tr, d_va, d_te = (seqs_to_eventdata(s, M) for s in (tr, va, te))
    gaps = np.concatenate([np.asarray(d)[1:] for d, _ in tr])
    gaps = gaps[gaps > 0]
    lo, hi = float(np.quantile(gaps, 0.01)), float(np.quantile(gaps, 0.999)) * 5
    dic = PhaseTypeDictionary.log_grid(lo, hi, 12, orders=2)
    exp_grid = 1.0 / np.geomspace(lo * 3, hi / 3, 6)
    rows = []
    for mname in methods:
        t = time.time()
        try:
            if mname == "MSX-auto":
                # select (R, l1) on the official validation split
                best = None
                for R in (1, 2, 3, 4):
                    for l1 in (0.0, 0.01, 0.05):
                        f = C.fit_msx(d_tr, PhaseTypeDictionary(dic.betas, orders=R), l1=l1)
                        v = f.loglik(d_va) / n_scored(d_va)
                        if best is None or v > best[0]:
                            best = (v, R, l1, f)
                _, R, l1, fit = best
                fit.name, fit.extra["R"], fit.extra["l1"] = "MSX-auto (ours)", R, l1
            elif mname == "MSX":
                fit = C.fit_msx(d_tr, dic, "MSX (ours, R=2)")
            elif mname == "ExpKern":
                fit = C.fit_tick_expkern(d_tr, exp_grid)
            elif mname == "SumExp":
                fit = C.fit_tick_sumexp(d_tr, dic.betas)
            elif mname == "EM":
                fit = C.fit_tick_em(d_tr, np.concatenate([[0.0], np.geomspace(lo, hi, 30)]))
            elif mname == "ADM4":
                fit = C.fit_tick_adm4(d_tr, exp_grid)
            elif mname == "CondLaw":
                fit = C.fit_tick_claw(d_tr, [(lo, lo, hi / 10, hi, "log")])
            else:
                raise KeyError(mname)
            row = dict(dataset=dataset, method=fit.name, key=mname,
                       ll_per_event=fit.loglik(d_te) / n_scored(d_te),
                       val_ll=fit.loglik(d_va) / n_scored(d_va), seconds=time.time() - t,
                       **{k: v for k, v in fit.extra.items() if k in ("R", "l1", "decay", "gap")})
        except Exception as e:
            row = dict(dataset=dataset, key=mname, error=repr(e)[:300])
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1, default=float))
    return rows
