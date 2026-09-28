"""Map between structured parameters (HawkesTruth-style arrays) and MSX theta vectors."""
from __future__ import annotations

import numpy as np

from ..events.data import EventData
from ..sim.cluster import HawkesTruth
from .features import DesignSpec
from .msx import MSXHawkes


def theta_from_truth(truth: HawkesTruth, spec: DesignSpec, data: EventData) -> tuple[np.ndarray, np.ndarray, dict]:
    d = truth.d
    lay = spec.layout(d, data.n_windows, max(data.n_news_types, 1), data.n_marks)
    th_d = truth.A.reshape(d, -1).copy()
    th_s = np.zeros((d, lay["Ps"]))
    mu = truth.mu if truth.mu.ndim == 2 else np.repeat(truth.mu[:, None], data.n_windows, axis=1)
    if spec.win_cols == 0:
        th_s[:, 0] = mu[:, 0]
    else:
        th_s[:, : data.n_windows * spec.win_cols] = np.repeat(mu, spec.win_cols, axis=1)
    if spec.n_tod and truth.tod_w is not None:
        th_s[:, lay["off_tod"]:lay["off_tod"] + spec.n_tod] = truth.tod_w
    if spec.exo is not None and truth.B is not None:
        th_s[:, lay["off_exo"]:lay["off_exo"] + lay["n_exo"]] = truth.B.reshape(d, -1)
    if spec.ant is not None and truth.H is not None:
        th_s[:, lay["off_ant"]:lay["off_ant"] + lay["n_ant"]] = truth.H.reshape(d, -1)
    return th_d, th_s, lay


def model_from_truth(truth: HawkesTruth, spec: DesignSpec, data: EventData) -> MSXHawkes:
    m = MSXHawkes(spec)
    m.theta_d, m.theta_s, m.layout = theta_from_truth(truth, spec, data)
    m.n_dims = truth.d
    return m
