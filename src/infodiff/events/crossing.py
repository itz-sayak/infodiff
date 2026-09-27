"""delta-crossing ("intrinsic time") price events.

An up (down) event is emitted each time the log price moves +delta (-delta) away from
the reference level of the previous event; a jump of n*delta at one timestamp emits n
tied events (ties never excite each other in the likelihood).  Consequences:

* assets become comparable in bps units (delta is in basis points of log price);
* bid-ask bounce is filtered whenever delta exceeds the half-spread;
* the price change over any interval equals delta * (N_up - N_down) up to one delta,
  so an echo-corrected intensity response integrates *exactly* to an expected
  price-drift (price-discovery) curve.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _crossings(t, logp, delta, max_per_tick):
    n = t.shape[0]
    out_t = np.empty(n * 2, dtype=np.float64)
    out_s = np.empty(n * 2, dtype=np.int8)
    k = 0
    if n == 0:
        return out_t[:0], out_s[:0]
    ref = logp[0]
    for i in range(1, n):
        diff = logp[i] - ref
        if diff >= delta or diff <= -delta:
            m = int(abs(diff) // delta)
            sgn = 1 if diff > 0 else -1
            ref += sgn * m * delta
            m = min(m, max_per_tick)
            if k + m > out_t.shape[0]:
                grow = np.empty(out_t.shape[0] * 2 + m, dtype=np.float64)
                grow[:k] = out_t[:k]
                out_t = grow
                g2 = np.empty(grow.shape[0], dtype=np.int8)
                g2[:k] = out_s[:k]
                out_s = g2
            for _ in range(m):
                out_t[k] = t[i]
                out_s[k] = sgn
                k += 1
    return out_t[:k], out_s[:k]


def crossing_events(t_sec: np.ndarray, price: np.ndarray, delta_bps: float, max_per_tick: int = 50):
    """Return (times, signs) of delta-crossings; delta in bps of log price."""
    t_sec = np.ascontiguousarray(t_sec, dtype=np.float64)
    logp = np.log(np.ascontiguousarray(price, dtype=np.float64))
    return _crossings(t_sec, logp, delta_bps * 1e-4, max_per_tick)
