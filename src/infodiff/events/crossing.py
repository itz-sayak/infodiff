# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""delta-crossing ("intrinsic time") price events.

An up (down) event is emitted each time the log price moves +delta (-delta) away from
the reference level of the previous event (a jump of several levels at one timestamp
emits one event, see below).  Consequences:

* assets become comparable in bps units (delta is in basis points of log price);
* bid-ask bounce is filtered whenever delta exceeds the half-spread;
* the price change over any interval equals delta * sum(sizes of up events) -
  delta * sum(sizes of down events) up to one delta, so an echo-corrected intensity
  response times the mean event size integrates to an expected price-drift curve.

Simple point process.  Prices are first collapsed to the last price per timestamp
(several trades or quote updates can share a millisecond when one order sweeps the
book), and a jump across several delta levels at one timestamp emits a *single* event
whose size (number of levels) is returned as a mark.  Hence no two events of the same
series share a timestamp -- the orderliness assumed by the Hawkes likelihood and by the
time-rescaling goodness-of-fit test.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _crossings(t, logp, delta):
    n = t.shape[0]
    out_t = np.empty(n, dtype=np.float64)
    out_s = np.empty(n, dtype=np.int8)
    out_k = np.empty(n, dtype=np.int32)
    k = 0
    if n == 0:
        return out_t[:0], out_s[:0], out_k[:0]
    ref = logp[0]
    i = 1
    while i < n:
        # collapse to the last price of this timestamp
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        diff = logp[j] - ref
        if diff >= delta or diff <= -delta:
            m = int(abs(diff) // delta)
            sgn = 1 if diff > 0 else -1
            ref += sgn * m * delta
            out_t[k] = t[i]
            out_s[k] = sgn
            out_k[k] = m
            k += 1
        i = j + 1
    return out_t[:k], out_s[:k], out_k[:k]


def crossing_events(t_sec: np.ndarray, price: np.ndarray, delta_bps: float, return_sizes: bool = False):
    """Return (times, signs[, sizes]) of delta-crossings; delta in bps of log price."""
    t_sec = np.ascontiguousarray(t_sec, dtype=np.float64)
    logp = np.log(np.ascontiguousarray(price, dtype=np.float64))
    t, s, k = _crossings(t_sec, logp, delta_bps * 1e-4)
    return (t, s, k) if return_sizes else (t, s)
