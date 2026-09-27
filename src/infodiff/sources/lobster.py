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
    return df.t.values[keep].astype(np.float64), typ[keep]


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
