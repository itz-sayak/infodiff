"""Scheduled macro-release calendar with exact UTC timestamps and first-print values.

* US data releases (CPI, NFP, PPI, Retail Sales, GDP advance, Jobless Claims): dates and
  *initial-release* values from ALFRED (output_type=4), so seasonal-factor-only
  releases are dropped automatically.  All at 08:30 America/New_York.
* FOMC: meeting calendars from federalreserve.gov; statement 14:00 ET, press
  conference 14:30 ET (every meeting since 2019).
* ECB: monetary-policy decision dates from ecb.europa.eu; 13:45/14:30 CET before
  2022-07-21 and 14:15/14:45 CET from then on.
* BoJ: statement dates from boj.or.jp; the release time is not fixed (latent onset),
  a nominal 12:00 JST is stored and flagged.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research; infodiff)"}
NY, CET, JST, UTC = ZoneInfo("America/New_York"), ZoneInfo("Europe/Berlin"), ZoneInfo("Asia/Tokyo"), ZoneInfo("UTC")
FRED = "https://api.stlouisfed.org/fred"


@dataclass(frozen=True)
class UsRelease:
    kind: str
    series: str
    transform: str  # "pct_mom" | "diff" | "level"
    freq_months: int = 1


US_RELEASES = [
    UsRelease("CPI", "CPIAUCSL", "pct_mom"),
    UsRelease("CORECPI", "CPILFESL", "pct_mom"),
    UsRelease("NFP", "PAYEMS", "diff"),
    UsRelease("UNRATE", "UNRATE", "level"),
    UsRelease("PPI", "PPIFIS", "pct_mom"),
    UsRelease("RETAIL", "RSAFS", "pct_mom"),
    UsRelease("GDP", "A191RL1Q225SBEA", "level", 3),
    UsRelease("CLAIMS", "ICSA", "level", 0),
]
# Companion series are merged into the headline event of the same release.
COMPANION = {"CORECPI": "CPI", "UNRATE": "NFP"}


def _fred(path: str, **params) -> dict:
    params = dict(params, api_key=os.environ["FRED_API_KEY"], file_type="json")
    for attempt in range(6):
        r = requests.get(f"{FRED}/{path}", params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"FRED request failed: {path} {params.get('series_id')}")


def _vintage_values(series: str, when: str, start: str) -> pd.Series:
    js = _fred("series/observations", series_id=series, realtime_start=when, realtime_end=when,
               observation_start=start)
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in js["observations"] if o["value"] != "."})
    return s.sort_index()


def us_releases(start: str, end: str) -> pd.DataFrame:
    rows = []
    for rel in US_RELEASES:
        obs_start = (pd.Timestamp(start) - pd.DateOffset(months=15)).strftime("%Y-%m-%d")
        js = _fred("series/observations", series_id=rel.series, output_type=4, observation_start=obs_start,
                   realtime_start="1776-07-04", realtime_end="9999-12-31")
        for o in js["observations"]:
            rd = o["realtime_start"]
            if not (start <= rd <= end) or o["value"] == ".":
                continue
            ref = pd.Timestamp(o["date"])
            actual = float(o["value"])
            prior = None
            if rel.transform in ("pct_mom", "diff"):
                vint = _vintage_values(rel.series, rd, (ref - pd.DateOffset(months=2)).strftime("%Y-%m-%d"))
                prev = vint[vint.index < ref]
                if len(prev) and ref in vint.index:
                    p = prev.iloc[-1]
                    actual_lvl = vint.loc[ref]
                    actual = 100 * (actual_lvl / p - 1) if rel.transform == "pct_mom" else actual_lvl - p
                    prev2 = prev.iloc[:-1]
                    if len(prev2):
                        prior = 100 * (p / prev2.iloc[-1] - 1) if rel.transform == "pct_mom" else p - prev2.iloc[-1]
            t_local = datetime.combine(date.fromisoformat(rd), datetime.min.time()).replace(hour=8, minute=30, tzinfo=NY)
            rows.append(dict(kind=rel.kind, t_utc=pd.Timestamp(t_local.astimezone(UTC)), ref_period=ref,
                             actual=actual, prior=prior, source="ALFRED", onset_known=True))
    df = pd.DataFrame(rows)
    return df


def fomc_meetings(start: str, end: str) -> pd.DataFrame:
    html = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=60).text
    rows = []
    months = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
    for block in re.split(r"(?=\d{4} FOMC Meetings)", html)[1:]:
        year = int(block[:4])
        for mon, days in re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<', block, re.S):
            if "notation" in days.lower() or "unscheduled" in days.lower():
                continue
            last_mon = mon.split("/")[-1].strip()[:3].lower()
            dd = re.findall(r"\d+", days)
            if not dd or last_mon not in months:
                continue
            day = int(dd[-1])
            d = date(year, months[last_mon], day)
            for kind, hh, mm in (("FOMC", 14, 0), ("FOMCPC", 14, 30)):
                t = datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY).astimezone(UTC)
                rows.append(dict(kind=kind, t_utc=pd.Timestamp(t), ref_period=pd.Timestamp(d), actual=None,
                                 prior=None, source="federalreserve.gov", onset_known=True))
    df = pd.DataFrame(rows)
    return df[(df.t_utc >= pd.Timestamp(start, tz="UTC")) & (df.t_utc <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]


def ecb_meetings(start: str, end: str) -> pd.DataFrame:
    rows = []
    switch = date(2022, 7, 21)
    for y in range(int(start[:4]), int(end[:4]) + 1):
        html = requests.get(f"https://www.ecb.europa.eu/press/govcdec/mopo/{y}/html/index_include.en.html",
                            headers=UA, timeout=60).text
        for iso, title in re.findall(r'<dt isoDate="(\d{4}-\d{2}-\d{2})">.*?<dd>.*?<a[^>]*>([^<]+)</a>', html, re.S):
            if "monetary policy decisions" not in title.lower():
                continue
            d = date.fromisoformat(iso)
            dec, pc = ((14, 15), (14, 45)) if d >= switch else ((13, 45), (14, 30))
            for kind, (hh, mm) in (("ECB", dec), ("ECBPC", pc)):
                t = datetime(d.year, d.month, d.day, hh, mm, tzinfo=CET).astimezone(UTC)
                rows.append(dict(kind=kind, t_utc=pd.Timestamp(t), ref_period=pd.Timestamp(d), actual=None,
                                 prior=None, source="ecb.europa.eu", onset_known=True))
    df = pd.DataFrame(rows).drop_duplicates(["kind", "t_utc"])
    return df[(df.t_utc >= pd.Timestamp(start, tz="UTC")) & (df.t_utc <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]


def boj_meetings(start: str, end: str) -> pd.DataFrame:
    rows = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        html = requests.get(f"https://www.boj.or.jp/en/mopo/mpmdeci/state_{y}/index.htm", headers=UA, timeout=60).text
        codes = set(re.findall(r"/k(\d{6})a\.(?:htm|pdf)", html))
        for c in sorted(codes):
            d = date(2000 + int(c[:2]), int(c[2:4]), int(c[4:6]))
            t = datetime(d.year, d.month, d.day, 12, 0, tzinfo=JST).astimezone(UTC)
            rows.append(dict(kind="BOJ", t_utc=pd.Timestamp(t), ref_period=pd.Timestamp(d), actual=None, prior=None,
                             source="boj.or.jp", onset_known=False))
    df = pd.DataFrame(rows)
    return df[(df.t_utc >= pd.Timestamp(start, tz="UTC")) & (df.t_utc <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]


def build_calendar(start: str = "2022-01-01", end: str = "2026-08-31", out: Path | None = None) -> pd.DataFrame:
    parts = [us_releases(start, end), fomc_meetings(start, end), ecb_meetings(start, end), boj_meetings(start, end)]
    cal = pd.concat(parts, ignore_index=True)
    # merge companion series (core CPI, unemployment rate) into their headline events
    comp = cal[cal.kind.isin(COMPANION)].copy()
    cal = cal[~cal.kind.isin(COMPANION)].copy()
    for ck, hk in COMPANION.items():
        sub = comp[comp.kind == ck].set_index("t_utc")["actual"].rename(f"{ck.lower()}_actual")
        cal = cal.merge(sub, left_on="t_utc", right_index=True, how="left")
        cal.loc[cal.kind != hk, f"{ck.lower()}_actual"] = float("nan")
    cal = cal.sort_values(["t_utc", "kind"]).reset_index(drop=True)
    cal["event_id"] = cal.kind + "_" + cal.t_utc.dt.strftime("%Y%m%dT%H%M")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        cal.to_parquet(out)
    return cal
