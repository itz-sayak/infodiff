# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Tick-data downloaders -> parquet.

HistData.com (FX / metals / index CFDs)
    Monthly ASCII tick files, rows "YYYYMMDD HHMMSSmmm,bid,ask,vol" stamped in fixed
    EST (UTC-5, no daylight saving).  Stored as data/interim/ticks/<PAIR>/<YYYYMM>.parquet
    with columns t_ms (UTC epoch ms), bid, ask.

Binance USD-M futures aggTrades (crypto)
    Daily zips from data.binance.vision, columns agg_id, price, qty, first_id, last_id,
    transact_time, is_buyer_maker (newer files carry a header row).  Stored as
    data/interim/ticks/<SYMBOL>/<YYYYMMDD>.parquet with t_ms, price, qty, buyer_maker.

Both are idempotent (existing outputs are skipped) so interrupted runs resume.
"""
from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

UA = {"User-Agent": "Mozilla/5.0 (research; infodiff)"}
EST_OFFSET_MS = 5 * 3600 * 1000


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    tmp.replace(path)


# ---------------------------------------------------------------------------- HistData
def histdata_month(pair: str, year: int, month: int, root: Path, session: requests.Session | None = None) -> str:
    out = root / "interim" / "ticks" / pair / f"{year}{month:02d}.parquet"
    if out.exists():
        return "cached"
    s = session or requests.Session()
    page = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/{pair.lower()}/{year}/{month}"
    html = s.get(page, headers=UA, timeout=60).text
    import re
    m = re.search(r'id="tk" value="([0-9a-f]+)"', html)
    if not m:
        return "no-token"
    form = dict(tk=m.group(1), date=str(year), datemonth=f"{year}{month:02d}", platform="ASCII",
                timeframe="T", fxpair=pair.upper())
    r = s.post("https://www.histdata.com/get.php", data=form, headers=dict(UA, Referer=page), timeout=600)
    if r.status_code != 200 or not r.content.startswith(b"PK"):
        return f"failed {r.status_code}"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        raw = pd.read_csv(z.open(name), header=None, names=["ts", "bid", "ask", "vol"],
                          dtype={"ts": str, "bid": np.float64, "ask": np.float64})
    ts = pd.to_datetime(raw.ts, format="%Y%m%d %H%M%S%f")
    t_ms = ts.values.astype("datetime64[ms]").astype(np.int64) + EST_OFFSET_MS
    df = pd.DataFrame(dict(t_ms=t_ms, bid=raw.bid.values, ask=raw.ask.values))
    df = df.sort_values("t_ms", kind="stable")
    _write(df, out)
    return f"ok {len(df)}"


def histdata_all(pairs: list[str], start: str, end: str, root: Path, workers: int = 3) -> None:
    months = pd.period_range(start, end, freq="M")
    jobs = [(p, m.year, m.month) for p in pairs for m in months]
    s = requests.Session()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(histdata_month, p, y, mo, root, s): (p, y, mo) for p, y, mo in jobs}
        for f in as_completed(futs):
            p, y, mo = futs[f]
            try:
                print(f"histdata {p} {y}-{mo:02d}: {f.result()}", flush=True)
            except Exception as e:  # keep going; rerun resumes
                print(f"histdata {p} {y}-{mo:02d}: ERROR {e}", flush=True)


# ---------------------------------------------------------------------------- Binance
def binance_day(symbol: str, day: pd.Timestamp, root: Path, session: requests.Session | None = None) -> str:
    out = root / "interim" / "ticks" / symbol / f"{day:%Y%m%d}.parquet"
    if out.exists():
        return "cached"
    s = session or requests.Session()
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
    for attempt in range(5):
        try:
            r = s.get(url, headers=UA, timeout=600)
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    else:
        return "failed network"
    if r.status_code == 404:
        return "missing"
    if r.status_code != 200:
        return f"failed {r.status_code}"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        first = z.open(name).readline()
        header = 0 if first[:1].isalpha() else None
        df = pd.read_csv(z.open(name), header=header,
                         names=["agg_id", "price", "qty", "first_id", "last_id", "t", "maker"],
                         usecols=[1, 2, 5, 6])
    t = df["t"].values.astype(np.int64)
    if t.max() > 10**14:  # microsecond stamps in newer files
        t //= 1000
    maker = df["maker"].astype(str).str.lower().eq("true").values
    out_df = pd.DataFrame(dict(t_ms=t, price=df["price"].values.astype(np.float64),
                               qty=df["qty"].values.astype(np.float32), buyer_maker=maker))
    _write(out_df.sort_values("t_ms", kind="stable"), out)
    return f"ok {len(out_df)}"


def binance_all(symbols: list[str], start: str, end: str, root: Path, workers: int = 6, weekdays_only: bool = True) -> None:
    days = pd.date_range(start, end, freq="D")
    if weekdays_only:
        days = days[days.dayofweek < 5]
    jobs = [(s, d) for d in days for s in symbols]
    sess = requests.Session()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(binance_day, s, d, root, sess): (s, d) for s, d in jobs}
        for f in as_completed(futs):
            s, d = futs[f]
            try:
                print(f"binance {s} {d:%Y-%m-%d}: {f.result()}", flush=True)
            except Exception as e:
                print(f"binance {s} {d:%Y-%m-%d}: ERROR {e}", flush=True)
