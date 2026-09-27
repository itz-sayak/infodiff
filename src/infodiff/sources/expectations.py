"""Pre-release expectations and standardised surprises for scheduled releases.

Sources (all free, reachable, and dated so that no post-release information leaks):
  CPI     Cleveland Fed inflation nowcast for the reference month, last value strictly
          before the release date.
  GDP     Atlanta Fed GDPNow: model forecast right before BEA's advance estimate.
  NFP     real-time statistical expectation: mean of the previous 3 first-print changes.
  CLAIMS  previous week's first print (random walk).
  PPI,
  RETAIL  mean of the previous 6 first-print m/m changes.
Survey consensus would be preferable but is not freely available; the statistical
expectations are documented as such in the paper.

Standardised surprise z = (actual - expected) / s, with s the expanding standard
deviation of *past* surprises of the same type (>= 8 past values; the first 8 events
use the std of those 8).  z is winsorised at +-4.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/expectations")


def cleveland_cpi_nowcasts(path: Path = RAW / "cleveland_nowcast_month.json") -> pd.DataFrame:
    js = json.loads(Path(path).read_text())
    rows = []
    for block in js:
        sub = block["chart"]["subcaption"]  # target month "YYYY-M"
        y, m = map(int, sub.split("-"))
        ref = pd.Timestamp(year=y, month=m, day=1)
        for ds in block["dataset"]:
            if ds["seriesname"] not in ("CPI Inflation", "Core CPI Inflation"):
                continue
            for dpt in ds["data"]:
                v = dpt.get("value", "")
                mm = re.search(r"\{br\}(\d{2})/(\d{2})\{br\}", dpt.get("tooltext", ""))
                if v == "" or not mm:
                    continue
                mo, dd = int(mm.group(1)), int(mm.group(2))
                yr = y + (1 if mo < m else 0)  # nowcast dates run into the following months
                rows.append(dict(ref_period=ref, asof_date=pd.Timestamp(year=yr, month=mo, day=dd),
                                 series="core" if "Core" in ds["seriesname"] else "headline", value=float(v)))
    return pd.DataFrame(rows)


def gdpnow_track_record(path: Path = RAW / "gdpnow.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="TrackRecord")
    df = df.rename(columns={"Quarter being forecasted": "quarter",
                            "Model Forecast Right Before BEA's Advance Estimate": "expected",
                            "BEA's Advance Estimate": "advance", "Release Date": "release"})
    return df[["quarter", "expected", "advance", "release"]].dropna()


def _standardise(err: np.ndarray, min_past: int = 8, clip: float = 4.0) -> np.ndarray:
    z = np.full(len(err), np.nan)
    ok = ~np.isnan(err)
    idx = np.where(ok)[0]
    for n, i in enumerate(idx):
        past = err[idx[:n]]
        s = np.std(past, ddof=1) if n >= min_past else np.std(err[idx[:min_past]], ddof=1)
        z[i] = np.clip(err[i] / s, -clip, clip) if s > 0 else 0.0
    return z


def surprises(cal: pd.DataFrame) -> pd.DataFrame:
    """Add columns expected, surprise, z, exp_source to the calendar rows that have them."""
    cal = cal.copy()
    cal["expected"] = np.nan
    cal["exp_source"] = None
    # CPI: Cleveland nowcast
    nc = cleveland_cpi_nowcasts()
    head = nc[nc.series == "headline"]
    for i, r in cal[cal.kind == "CPI"].iterrows():
        rd = r.t_utc.tz_convert("America/New_York").normalize().tz_localize(None)
        sub = head[(head.ref_period == r.ref_period) & (head.asof_date < rd)]
        if len(sub):
            cal.at[i, "expected"] = sub.sort_values("asof_date").value.iloc[-1]
            cal.at[i, "exp_source"] = "Cleveland Fed nowcast"
    # GDP: GDPNow
    tr = gdpnow_track_record()
    tr["release"] = pd.to_datetime(tr.release).dt.normalize()
    for i, r in cal[cal.kind == "GDP"].iterrows():
        rd = r.t_utc.tz_convert("America/New_York").normalize().tz_localize(None)
        hit = tr[tr.release == rd]
        if len(hit):
            cal.at[i, "expected"] = float(hit.expected.iloc[0])
            cal.at[i, "exp_source"] = "Atlanta Fed GDPNow"
    # statistical real-time expectations from the sequence of first prints
    for kind, window in (("NFP", 3), ("CLAIMS", 1), ("PPI", 6), ("RETAIL", 6)):
        sub = cal[cal.kind == kind].sort_values("t_utc")
        vals = sub.actual.values
        for n, i in enumerate(sub.index):
            if n >= window:
                cal.at[i, "expected"] = float(np.mean(vals[n - window:n]))
                cal.at[i, "exp_source"] = f"real-time mean of previous {window} first prints"
    cal["surprise"] = cal.actual - cal.expected
    cal["z"] = np.nan
    for kind, sub in cal.sort_values("t_utc").groupby("kind"):
        if sub.surprise.notna().any():
            cal.loc[sub.index, "z"] = _standardise(sub.surprise.values.astype(float))
    return cal


def mark_vector(row) -> np.ndarray:
    """Nonnegative mark basis [1, z+, z-]; releases without a numeric surprise get [1, 0, 0]."""
    z = row.get("z", np.nan) if hasattr(row, "get") else np.nan
    if z is None or not np.isfinite(z):
        return np.array([1.0, 0.0, 0.0])
    return np.array([1.0, max(z, 0.0), max(-z, 0.0)])
