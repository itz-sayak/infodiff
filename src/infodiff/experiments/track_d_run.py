# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Track D driver: chronological post-release forecasting benchmark.

Train: news + placebo windows of 2022-2024 (MSX fit with the main-study spec).
Test : news windows of 2025-2026.  For every test release cluster (anchor tau = first
release of the window) and horizon h in {60, 300, 900} s we forecast, per asset,
activity N (delta-crossings), realised variance delta^2 N and signed drift
delta (N_up - N_down), using only information available at tau:
  * MSX: closed-form conditional expectation (track_d.msx_forecast) with the window
    intercept re-estimated by ML on the pre-release part of the test window only;
  * climatology: training mean of log1p(N) for the same release type;
  * event-study: OLS of log1p(N) on log1p(pre-release counts over 5 and 30 min), |z|
    and release-type dummies (fitted on training releases);
  * HAR: OLS of log1p(N) on log1p(counts over the previous 1, 5 and 30 min).
Drift baselines: zero (martingale) and the training mean drift of the same type and
surprise sign.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..events.data import EventData
from ..models.features import build_design
from ..models.msx import MSXHawkes
from .track_d import HORIZONS, metrics, msx_forecast, pre_counts, realised


def _window_data(data: EventData, w: int, t1: float | None = None) -> EventData:
    sub = data.subset(np.array([w]))
    if t1 is not None:
        sub.t1 = np.array([t1])
    return sub


def pre_release_model(model: MSXHawkes, data: EventData, w: int, tau: float, W_train: int,
                      n_newton: int = 60) -> MSXHawkes:
    """Copy of `model` for the single window w, whose intercepts are the ML estimates on
    [t0_w, tau) with every other parameter fixed (no information after tau is used)."""
    d = data.n_dims
    pre = _window_data(data, w, t1=tau)
    lay_test = model.spec.layout(d, 1, max(data.n_news_types, 1), data.n_marks)
    wc = model.spec.win_cols
    th_s = np.concatenate([np.zeros((d, wc)), model.theta_s[:, W_train * wc:]], axis=1)
    T = tau - pre.t0[0]
    for i in range(d):
        des = build_design(pre, model.spec, i)
        if des.n == 0:
            th_s[i, :wc] = 1e-6
            continue
        # other-term intensities at the pre-release events of dim i (intercept column 0 excluded)
        Xs = des.Xs.tocsc()
        other = des.Xd.astype(np.float64) @ model.theta_d[i] + Xs[:, wc:] @ th_s[i, wc:]
        # other-term compensator over [t0, tau) is irrelevant for the intercept's score
        c = max(des.n / T - other.mean(), 1e-6)
        for _ in range(n_newton):  # maximise sum log(c + r_n) - c T
            f = np.sum(1.0 / (c + other)) - T
            fp = -np.sum(1.0 / (c + other) ** 2)
            c_new = max(c - f / fp, 0.5 * c)
            if abs(c_new - c) <= 1e-10 * c:
                c = c_new
                break
            c = c_new
        th_s[i, :wc] = c  # constant level estimated on pre-release data (both tents equal)
    m = copy.copy(model)
    m.theta_s = th_s
    m.layout = lay_test
    return m


def _type_of_window(data: EventData, w: int) -> tuple[int, float]:
    a, b = data.news_ptr[w], data.news_ptr[w + 1]
    first = int(np.argmin(data.news_t[a:b])) + a
    z = data.news_marks[first]
    zz = (z[1] - z[2]) if len(z) >= 3 else 0.0
    return int(data.news_type[first]), float(zz)


def collect(data: EventData, model: MSXHawkes | None, W_train: int | None):
    """Targets, features and (optionally) MSX forecasts for all news windows of `data`."""
    out = dict(types=[], z=[], real=[], pre=[], msx=[])
    placebo = np.array([m["placebo"] for m in data.meta["windows"]])
    for w in np.where(~placebo)[0]:
        a, b = data.news_ptr[w], data.news_ptr[w + 1]
        if b == a:
            continue
        tau = float(data.news_t[a:b].min())
        c, z = _type_of_window(data, w)
        out["types"].append(c)
        out["z"].append(z)
        out["real"].append(realised(data, w, tau))
        out["pre"].append(pre_counts(data, w, tau))
        if model is not None:
            m_w = pre_release_model(model, data, w, tau, W_train)
            sub = _window_data(data, w)
            idx = list(range(sub.news_ptr[0], sub.news_ptr[1]))
            idx = [e for e in idx if sub.news_t[e] <= tau]
            out["msx"].append(msx_forecast(m_w, sub, 0, tau, idx))
    return {k: np.asarray(v) for k, v in out.items()}


def _ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ beta


def baselines(train: dict, test: dict, n_types: int) -> dict:
    """Return dict name -> predicted counts (R, H, d)."""
    R, H, d = test["real"].shape
    preds = {}
    # climatology: per type mean of log1p(N) (fallback: global mean)
    clim = np.zeros((R, H, d))
    for r, c in enumerate(test["types"]):
        m = train["types"] == c
        src = train["real"][m] if m.sum() >= 3 else train["real"]
        clim[r] = np.expm1(np.log1p(src).mean(0))
    preds["climatology"] = clim

    def design(dset, cols):
        X = [np.log1p(dset["pre"][:, s, :]) for s in cols]
        return X

    onehot = lambda t: np.eye(n_types)[t][:, 1:]
    es = np.zeros((R, H, d))
    har = np.zeros((R, H, d))
    for i in range(d):
        Xtr_es = np.column_stack([np.log1p(train["pre"][:, 1, i]), np.log1p(train["pre"][:, 2, i]),
                                  np.abs(train["z"]), onehot(train["types"])])
        Xte_es = np.column_stack([np.log1p(test["pre"][:, 1, i]), np.log1p(test["pre"][:, 2, i]),
                                  np.abs(test["z"]), onehot(test["types"])])
        Xtr_h = np.log1p(train["pre"][:, :, i])
        Xte_h = np.log1p(test["pre"][:, :, i])
        for h in range(H):
            y = np.log1p(train["real"][:, h, i])
            es[:, h, i] = np.expm1(np.maximum(_ols(Xtr_es, y)(Xte_es), 0))
            har[:, h, i] = np.expm1(np.maximum(_ols(Xtr_h, y)(Xte_h), 0))
    preds["event-study OLS"] = es
    preds["HAR"] = har
    return preds


def drift_baselines(train: dict, test: dict) -> dict:
    R, H, d = test["real"].shape
    zero = np.zeros((R, H, d))
    zero[..., 0::2] = zero[..., 1::2] = 1.0  # equal up/down -> zero drift
    sign_mean = np.zeros((R, H, d))
    for r, (c, z) in enumerate(zip(test["types"], test["z"])):
        m = (train["types"] == c) & (np.sign(train["z"]) == np.sign(z))
        src = train["real"][m] if m.sum() >= 3 else train["real"][train["types"] == c]
        sign_mean[r] = src.mean(0) if len(src) else 0.0
    return {"martingale (zero drift)": zero, "type x sign mean": sign_mean}


def evaluate_all(train: dict, test: dict, deltas: np.ndarray, n_types: int, assets: list[str]) -> dict:
    res = {"horizons_s": list(HORIZONS), "assets": assets, "n_test": int(len(test["real"])),
           "n_train": int(len(train["real"])), "activity": {}, "drift": {}}
    act = baselines(train, test, n_types)
    if len(test["msx"]):
        act = {"MSX closed form (ours)": test["msx"], **act}
    for name, p in act.items():
        res["activity"][name] = metrics(p, test["real"], deltas)
    dr = drift_baselines(train, test)
    if len(test["msx"]):
        dr = {"MSX closed form (ours)": test["msx"], **dr}
    for name, p in dr.items():
        res["drift"][name] = metrics(p, test["real"], deltas)
    return res


def save(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=float))
