"""Build windowed multi-asset event datasets around scheduled releases (+ placebo windows).

Windows
    Releases closer than `cluster_gap` are merged into one cluster (e.g. FOMC statement
    + press conference, CPI + jobless claims at 08:30).  The likelihood interval is
    [first - pre, last + post), with a further `burn` of history before it.

Placebo windows
    For each cluster a matched control window at the same UTC clock times on the same
    weekday 1-3 weeks away with no scheduled release within +-3h.  Its pseudo-releases
    carry the extra news type PLACEBO, whose fitted kernel must be ~0 (falsification
    test); placebo windows also identify the time-of-day seasonality separately from
    the news response.

Dimensions
    asset a -> dims (2a: up-crossing, 2a+1: down-crossing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .crossing import crossing_events
from .data import EventData

MIN = 60.0
ASSETS = {  # name -> (source kind, file key)
    "EURUSD": ("fx", "EURUSD"), "USDJPY": ("fx", "USDJPY"), "XAUUSD": ("fx", "XAUUSD"),
    "SPX": ("fx", "SPXUSD"), "BTC": ("crypto", "BTCUSDT"), "ETH": ("crypto", "ETHUSDT"),
}


@dataclass
class WindowSpec:
    kinds: list[str]
    assets: list[str]
    deltas_bps: dict[str, float]
    pre: float = 30 * MIN
    post: float = 60 * MIN
    burn: float = 30 * MIN
    cluster_gap: float = 60 * MIN
    placebo: bool = True
    tick_root: Path = Path("data/interim/ticks")
    min_ticks_per_10min: int = 5
    extra: dict = field(default_factory=dict)


class TickStore:
    """Lazy per-month (FX) / per-day (crypto) tick loading with a small LRU cache."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @lru_cache(maxsize=24)
    def _fx_month(self, key: str, ym: str):
        f = self.root / key / f"{ym}.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        return df.t_ms.values / 1e3, 0.5 * (df.bid.values + df.ask.values)

    @lru_cache(maxsize=16)
    def _crypto_day(self, key: str, ymd: str):
        f = self.root / key / f"{ymd}.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f, columns=["t_ms", "price"])
        return df.t_ms.values / 1e3, df.price.values

    def span(self, asset: str, lo: float, hi: float):
        kind, key = ASSETS[asset]
        chunks = []
        cur = pd.Timestamp(lo, unit="s", tz="UTC").normalize()
        end = pd.Timestamp(hi, unit="s", tz="UTC")
        seen = set()
        while cur <= end:
            tag = cur.strftime("%Y%m") if kind == "fx" else cur.strftime("%Y%m%d")
            if tag not in seen:
                seen.add(tag)
                got = self._fx_month(key, tag) if kind == "fx" else self._crypto_day(key, tag)
                if got is not None:
                    t, p = got
                    a, b = np.searchsorted(t, lo), np.searchsorted(t, hi)
                    chunks.append((t[a:b], p[a:b]))
            cur += pd.Timedelta(days=1)
        if not chunks:
            return np.zeros(0), np.zeros(0)
        return np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])


def cluster_calendar(cal: pd.DataFrame, kinds: list[str], gap: float) -> list[pd.DataFrame]:
    sub = cal[cal.kind.isin(kinds)].sort_values("t_utc").reset_index(drop=True)
    ts = sub.t_utc.map(lambda x: x.timestamp()).values
    groups, start = [], 0
    for i in range(1, len(sub) + 1):
        if i == len(sub) or ts[i] - ts[i - 1] > gap:
            groups.append(sub.iloc[start:i])
            start = i
    return groups


def _placebo_shift(cl_lo: float, cl_hi: float, busy: np.ndarray) -> float | None:
    for weeks in (-1, 1, -2, 2, -3, 3):
        s = weeks * 7 * 86400.0
        lo, hi = cl_lo + s - 3 * 3600, cl_hi + s + 3 * 3600
        a, b = np.searchsorted(busy, lo), np.searchsorted(busy, hi)
        if a == b:
            return s
    return None


def build_windows(cal: pd.DataFrame, spec: WindowSpec, marks_fn=None, verbose: bool = True) -> EventData:
    """marks_fn(row) -> nonnegative mark vector (M,), default [1]."""
    store = TickStore(spec.tick_root)
    kinds = list(spec.kinds)
    type_index = {k: i for i, k in enumerate(kinds + (["PLACEBO"] if spec.placebo else []))}
    busy = np.sort(cal.t_utc.map(lambda x: x.timestamp()).values)
    clusters = cluster_calendar(cal, kinds, spec.cluster_gap)
    M = len(marks_fn(clusters[0].iloc[0])) if marks_fn else 1

    plan = []  # (lo, hi, news list, meta)
    for cl in clusters:
        tt = cl.t_utc.map(lambda x: x.timestamp()).values
        lo, hi = tt.min() - spec.pre, tt.max() + spec.post
        news = [(t, type_index[k], marks_fn(r) if marks_fn else np.ones(1)) for t, k, (_, r) in zip(tt, cl.kind, cl.iterrows())]
        plan.append((lo, hi, news, dict(placebo=False, events=list(cl.event_id), kinds=list(cl.kind))))
        if spec.placebo:
            s = _placebo_shift(lo, hi, busy)
            if s is not None:
                pn = [(t + s, type_index["PLACEBO"], np.eye(1, M, 0)[0]) for t in tt]
                plan.append((lo + s, hi + s, pn, dict(placebo=True, events=[f"PLACEBO<{e}" for e in cl.event_id],
                                                      kinds=["PLACEBO"] * len(tt))))

    d = 2 * len(spec.assets)
    times, types, ptr, t0s, t1s, tod0, nptr, nt, nty, nm, metas = [], [], [0], [], [], [], [0], [], [], [], []
    dropped = 0
    for k, (lo, hi, news, meta) in enumerate(plan):
        seg_t, seg_u, ok = [], [], True
        cover = {}
        for a, asset in enumerate(spec.assets):
            t, p = store.span(asset, lo - spec.burn, hi)
            blocks = np.histogram(t, bins=np.arange(lo - spec.burn, hi + 1, 600.0))[0] if len(t) else np.zeros(1)
            cover[asset] = float(np.mean(blocks >= spec.min_ticks_per_10min))
            if cover[asset] < 0.9:
                ok = False
                break
            et, es = crossing_events(t, p, spec.deltas_bps[asset])
            seg_t.append(et)
            seg_u.append(np.where(es > 0, 2 * a, 2 * a + 1))
        if not ok:
            dropped += 1
            continue
        tt = np.concatenate(seg_t)
        uu = np.concatenate(seg_u).astype(np.int32)
        o = np.argsort(tt, kind="stable")
        times.append(tt[o])
        types.append(uu[o])
        ptr.append(ptr[-1] + len(tt))
        t0s.append(lo)
        t1s.append(hi)
        ts = pd.Timestamp(lo, unit="s", tz="UTC")
        tod0.append(ts.hour * 3600 + ts.minute * 60 + ts.second)
        for (tn, c, z) in news:
            nt.append(tn)
            nty.append(c)
            nm.append(np.asarray(z, float))
        nptr.append(nptr[-1] + len(news))
        meta.update(coverage=cover, n_events=int(len(tt)), t0=lo, t1=hi)
        metas.append(meta)
        if verbose and len(metas) % 100 == 0:
            print(f"  built {len(metas)} windows ({dropped} dropped)", flush=True)
    data = EventData(times=np.concatenate(times), types=np.concatenate(types), wptr=np.asarray(ptr),
                     t0=np.asarray(t0s), t1=np.asarray(t1s), n_dims=d, tod0=np.asarray(tod0, float),
                     news_ptr=np.asarray(nptr), news_t=np.asarray(nt), news_type=np.asarray(nty, np.int32),
                     news_marks=np.asarray(nm).reshape(-1, M), n_news_types=len(type_index),
                     meta=dict(windows=metas, type_index=type_index, assets=list(spec.assets),
                               deltas_bps=dict(spec.deltas_bps), dropped=dropped))
    return data


def calibrate_delta(store: TickStore, asset: str, days: list[pd.Timestamp], target_rate: float = 0.2,
                    grid=(0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)) -> float:
    """Smallest grid delta whose median daily event rate (events/s over 07-20 UTC) <= target."""
    rates = {g: [] for g in grid}
    for day in days:
        lo = day.timestamp() + 7 * 3600
        hi = day.timestamp() + 20 * 3600
        t, p = store.span(asset, lo, hi)
        if len(t) < 1000:
            continue
        for g in grid:
            et, _ = crossing_events(t, p, g)
            rates[g].append(len(et) / (hi - lo))
    for g in grid:
        if rates[g] and np.median(rates[g]) <= target_rate:
            return g
    return grid[-1]


def save_eventdata(data: EventData, path: Path) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=5)


def load_eventdata(path: Path) -> EventData:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
