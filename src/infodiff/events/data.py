# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Windowed multivariate event data with exogenous (news) events.

Data are a set of independent observation windows.  Window w has a likelihood
interval [t0[w], t1[w]); events of the window with time < t0[w] form a burn-in
history that excites the process but does not enter the likelihood.  News events of
a window may fall before t0 (they still excite) or after t1 (they can only act
through the anticipatory kernel).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EventData:
    times: np.ndarray  # (N,) float64, sorted within each window
    types: np.ndarray  # (N,) int32 in [0, n_dims)
    wptr: np.ndarray  # (W+1,) int64 offsets into times/types
    t0: np.ndarray  # (W,) likelihood start
    t1: np.ndarray  # (W,) likelihood end
    n_dims: int
    tod0: np.ndarray | None = None  # (W,) local clock seconds-of-day at t0 (for seasonal baseline)
    news_ptr: np.ndarray | None = None  # (W+1,)
    news_t: np.ndarray | None = None  # (E,)
    news_type: np.ndarray | None = None  # (E,) int32
    news_marks: np.ndarray | None = None  # (E, M) nonnegative mark features
    n_news_types: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.ascontiguousarray(self.times, dtype=np.float64)
        self.types = np.ascontiguousarray(self.types, dtype=np.int32)
        self.wptr = np.ascontiguousarray(self.wptr, dtype=np.int64)
        self.t0 = np.ascontiguousarray(self.t0, dtype=np.float64)
        self.t1 = np.ascontiguousarray(self.t1, dtype=np.float64)
        W = len(self.t0)
        if len(self.wptr) != W + 1 or len(self.t1) != W:
            raise ValueError("inconsistent window arrays")
        if self.tod0 is None:
            self.tod0 = np.zeros(W)
        if self.news_ptr is None:
            self.news_ptr = np.zeros(W + 1, dtype=np.int64)
            self.news_t = np.zeros(0)
            self.news_type = np.zeros(0, dtype=np.int32)
            self.news_marks = np.zeros((0, 1))
        self.news_ptr = np.ascontiguousarray(self.news_ptr, dtype=np.int64)
        self.news_t = np.ascontiguousarray(self.news_t, dtype=np.float64)
        self.news_type = np.ascontiguousarray(self.news_type, dtype=np.int32)
        self.news_marks = np.ascontiguousarray(np.atleast_2d(self.news_marks), dtype=np.float64)
        if len(self.news_t) == 0:
            self.news_marks = self.news_marks.reshape(0, max(self.news_marks.shape[-1], 1))
        if np.any(self.news_marks < 0):
            raise ValueError("news marks must be nonnegative basis values")

    @property
    def n_windows(self) -> int:
        return len(self.t0)

    @property
    def n_marks(self) -> int:
        return self.news_marks.shape[1]

    def window(self, w: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.wptr[w], self.wptr[w + 1]
        return self.times[a:b], self.types[a:b]

    def in_likelihood(self) -> np.ndarray:
        """Boolean mask of events inside their window's likelihood interval."""
        wid = np.repeat(np.arange(self.n_windows), np.diff(self.wptr))
        return (self.times >= self.t0[wid]) & (self.times < self.t1[wid])

    def window_ids(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_windows), np.diff(self.wptr))

    def subset(self, windows: np.ndarray) -> "EventData":
        """New EventData holding only the given windows (order preserved, repeats allowed)."""
        windows = np.asarray(windows, dtype=np.int64)
        seg_t, seg_u, ptr = [], [], [0]
        nt, nty, nm, nptr = [], [], [], [0]
        for w in windows:
            a, b = self.wptr[w], self.wptr[w + 1]
            seg_t.append(self.times[a:b])
            seg_u.append(self.types[a:b])
            ptr.append(ptr[-1] + (b - a))
            c, e = self.news_ptr[w], self.news_ptr[w + 1]
            nt.append(self.news_t[c:e])
            nty.append(self.news_type[c:e])
            nm.append(self.news_marks[c:e])
            nptr.append(nptr[-1] + (e - c))
        M = self.n_marks
        return EventData(
            times=np.concatenate(seg_t) if seg_t else np.zeros(0),
            types=np.concatenate(seg_u) if seg_u else np.zeros(0, np.int32),
            wptr=np.asarray(ptr),
            t0=self.t0[windows],
            t1=self.t1[windows],
            n_dims=self.n_dims,
            tod0=self.tod0[windows],
            news_ptr=np.asarray(nptr),
            news_t=np.concatenate(nt) if nt else np.zeros(0),
            news_type=np.concatenate(nty) if nty else np.zeros(0, np.int32),
            news_marks=np.concatenate(nm) if nm else np.zeros((0, M)),
            n_news_types=self.n_news_types,
            meta=dict(self.meta),
        )

    def counts(self) -> np.ndarray:
        """Number of in-likelihood events per dimension."""
        return np.bincount(self.types[self.in_likelihood()], minlength=self.n_dims)


def concat_eventdata(parts: list[EventData]) -> EventData:
    """Concatenate window sets that share dimensions, news types and mark layout."""
    t, u, ptr, nt, nty, nm, nptr, metas = [], [], [0], [], [], [], [0], []
    for p in parts:
        t.append(p.times)
        u.append(p.types)
        ptr.extend(list(ptr[-1] + p.wptr[1:]))
        nt.append(p.news_t)
        nty.append(p.news_type)
        nm.append(p.news_marks)
        nptr.extend(list(nptr[-1] + p.news_ptr[1:]))
        metas.extend(p.meta.get("windows", []))
    first = parts[0]
    meta = {k: v for k, v in first.meta.items() if k != "windows"}
    meta["windows"] = metas
    if all("mean_event_size" in p.meta for p in parts):  # event-count weighted across parts
        assets = first.meta.get("assets", [])
        sizes = {}
        for a_idx, a in enumerate(assets):
            w = np.array([np.isin(p.types, [2 * a_idx, 2 * a_idx + 1]).sum() for p in parts], float)
            v = np.array([p.meta["mean_event_size"][a] for p in parts])
            sizes[a] = float((w * v).sum() / max(w.sum(), 1.0))
        meta["mean_event_size"] = sizes
    return EventData(times=np.concatenate(t), types=np.concatenate(u), wptr=np.asarray(ptr),
                     t0=np.concatenate([p.t0 for p in parts]), t1=np.concatenate([p.t1 for p in parts]),
                     n_dims=first.n_dims, tod0=np.concatenate([p.tod0 for p in parts]),
                     news_ptr=np.asarray(nptr), news_t=np.concatenate(nt), news_type=np.concatenate(nty),
                     news_marks=np.concatenate(nm), n_news_types=first.n_news_types, meta=meta)
