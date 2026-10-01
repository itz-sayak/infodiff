# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""LOBSTER level-1 message files -> 6-type marked event sequences (EasyTPP schema).

Types (following the standard 6-dimensional LOB Hawkes specification, e.g. Bacry,
Jaisson & Muzy 2016; Jain et al. 2024):
    0 limit order bid      (type 1, direction +1)
    1 limit order ask      (type 1, direction -1)
    2 cancel/delete bid    (types 2,3, direction +1)
    3 cancel/delete ask    (types 2,3, direction -1)
    4 market buy           (executions 4,5 against the ask: direction -1)
    5 market sell          (executions 4,5 against the bid: direction +1)
Halts (7) and cross trades (6) are dropped.  The trading day is split chronologically
60/20/20 into train/validation/test and cut into consecutive sequences of `seq_len`
events.  Times are seconds after the start of each sequence.

Ties.  6-10% of messages share a nanosecond timestamp with their predecessor (one order
walking several queue entries).  EasyTPP requires strictly positive gaps, so each tied
message is displaced by an independent U(0, 100 ns) offset (seeded), events are re-sorted,
and any residual tie (float64 resolution) is separated by 1 ns -- the same
randomisation-within-resolution device as Rambaldi et al. (2015).
All models are trained and scored on the identical jittered files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TYPE_NAMES = ["LO_bid", "LO_ask", "CXL_bid", "CXL_ask", "MO_buy", "MO_sell"]


def lobster_events(msg_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(msg_csv, header=None, names=["t", "type", "oid", "size", "price", "dir"])
    typ = np.full(len(df), -1)
    t, d = df["type"].values, df["dir"].values
    typ[(t == 1) & (d == 1)] = 0
    typ[(t == 1) & (d == -1)] = 1
    typ[np.isin(t, (2, 3)) & (d == 1)] = 2
    typ[np.isin(t, (2, 3)) & (d == -1)] = 3
    typ[np.isin(t, (4, 5)) & (d == -1)] = 4
    typ[np.isin(t, (4, 5)) & (d == 1)] = 5
    keep = typ >= 0
    t_out, k_out = df.t.values[keep].astype(np.float64), typ[keep]
    tied = np.r_[False, np.diff(t_out) == 0]
    rng = np.random.default_rng(2012)
    t_out = t_out + tied * rng.uniform(0.0, 1e-7, len(t_out))
    o = np.argsort(t_out, kind="stable")
    t_out, k_out = t_out[o], k_out[o]
    # jitter below float64 resolution at t ~ 5e4 s can leave exact ties: enforce >= 1 ns spacing
    for i in np.where(np.diff(t_out) <= 0)[0] + 1:
        if t_out[i] <= t_out[i - 1]:
            t_out[i] = t_out[i - 1] + 1e-9
    bad = np.where(np.diff(t_out) <= 0)[0]
    while len(bad):  # cascade (rare)
        for i in bad + 1:
            t_out[i] = max(t_out[i], t_out[i - 1] + 1e-9)
        bad = np.where(np.diff(t_out) <= 0)[0]
    return t_out, k_out


def to_sequences(t: np.ndarray, k: np.ndarray, seq_len: int) -> list[dict]:
    out = []
    for a in range(0, len(t) - seq_len + 1, seq_len):
        tt = t[a:a + seq_len] - t[a]
        dt = np.diff(tt, prepend=tt[0])
        out.append(dict(seq_len=seq_len, time_since_start=tt.tolist(), time_since_last_event=dt.tolist(),
                        type_event=k[a:a + seq_len].astype(int).tolist(), dim_process=6))
    return out


def build(raw: Path = Path("data/raw/lobster"), out: Path = Path("data/processed/tpp"), seq_len: int = 128,
          tickers=("AAPL", "AMZN", "GOOG", "INTC", "MSFT")) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for tk in tickers:
        t, k = lobster_events(raw / f"{tk}_message_1.csv")
        lo, hi = t.min(), t.max()
        c1, c2 = lo + 0.6 * (hi - lo), lo + 0.8 * (hi - lo)
        parts = {"train": t < c1, "validation": (t >= c1) & (t < c2), "test": t >= c2}
        stats[tk] = {}
        for split, m in parts.items():
            seqs = to_sequences(t[m], k[m], seq_len)
            with open(out / f"lob_{tk.lower()}_{split}.jsonl", "w") as f:
                for s in seqs:
                    f.write(json.dumps(s) + "\n")
            stats[tk][split] = dict(n_seq=len(seqs), n_events=int(m.sum()),
                                    type_freq=np.bincount(k[m], minlength=6).tolist())
    (out / "lob_stats.json").write_text(json.dumps(stats, indent=1))
    return stats
