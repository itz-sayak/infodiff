# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Dukascopy tick downloader (run from a network where datafeed.dukascopy.com is reachable).

Fills the HistData gap (Feb–Jul 2023 and scattered days) for EURUSD, USDJPY, XAUUSD and
the S&P 500 CFD.  Output matches the project tick schema (t_ms UTC, bid, ask):

    data/interim/ticks_dukascopy/<PAIR>/<YYYYMM>.parquet

Usage (only needs Python 3.9+, requests, pandas, pyarrow):
    python scripts/dukascopy_fill.py 2023-02-01 2023-07-31
    python scripts/dukascopy_fill.py 2022-01-01 2026-08-31      # full cross-check copy

Then copy the ticks_dukascopy folder into D:/Quant/data/interim/ on the project machine.
Dukascopy bi5 format: hourly LZMA files of 20-byte big-endian records
(ms offset in hour uint32, ask uint32, bid uint32, ask volume float32, bid volume float32);
prices are integers in instrument points.
"""
from __future__ import annotations

import lzma
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# project name -> (Dukascopy symbol, point size)
INSTRUMENTS = {"EURUSD": ("EURUSD", 1e-5), "USDJPY": ("USDJPY", 1e-3), "XAUUSD": ("XAUUSD", 1e-3),
               "SPXUSD": ("USA500IDXUSD", 1e-3)}
URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m0:02d}/{d:02d}/{h:02d}h_ticks.bi5"
OUT = Path(__file__).resolve().parents[1] / "data" / "interim" / "ticks_dukascopy"


def fetch_hour(sym: str, ts: pd.Timestamp, point: float, session: requests.Session):
    url = URL.format(sym=sym, y=ts.year, m0=ts.month - 1, d=ts.day, h=ts.hour)  # month is 0-based
    for attempt in range(5):
        try:
            r = session.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 404 or not r.content:
                return None
            raw = lzma.decompress(r.content)
            n = len(raw) // 20
            rec = np.frombuffer(raw[: n * 20], dtype=np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
                                                              ("av", ">f4"), ("bv", ">f4")]))
            base = int(ts.value // 10**6)
            return pd.DataFrame(dict(t_ms=base + rec["ms"].astype(np.int64), bid=rec["bid"] * point,
                                     ask=rec["ask"] * point))
        except (requests.RequestException, lzma.LZMAError):
            time.sleep(2 * (attempt + 1))
    return None


def month(pair: str, ym: pd.Period) -> str:
    sym, point = INSTRUMENTS[pair]
    out = OUT / pair / f"{ym.year}{ym.month:02d}.parquet"
    if out.exists():
        return "cached"
    hours = pd.date_range(ym.start_time, ym.end_time, freq="h", tz="UTC")
    hours = hours[(hours.dayofweek < 5) | ((hours.dayofweek == 6) & (hours.hour >= 21))]  # FX week
    s = requests.Session()
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda h: fetch_hour(sym, h, point, s), hours) if p is not None]
    if not parts:
        return "empty"
    df = pd.concat(parts).sort_values("t_ms", kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd", index=False)
    return f"ok {len(df)}"


if __name__ == "__main__":
    start, end = sys.argv[1], sys.argv[2]
    for ym in pd.period_range(start[:7], end[:7], freq="M"):
        for pair in INSTRUMENTS:
            print(pair, ym, month(pair, ym), flush=True)
