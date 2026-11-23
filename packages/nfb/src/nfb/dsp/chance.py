"""The chance spread of one trial, from pairs of readings of the person's own level.

A trial's result is a weighted sum of readings whose weights add to zero, so its variance is
`-sum_k D(k) * sum_i w_i * w_(i+k)`, where `D(k)` is the mean squared change at lag k. Each pair of valid
readings up to one trial apart adds to `D`; lags not yet seen are filled from a random walk fitted to the
seen ones, so a value exists from the first usable pair. No task and no trial are needed.
"""

from __future__ import annotations

import math
import threading
from collections import deque

import numpy as np

MEMORY_S = 120.0
"""Signal seconds of pairs behind the value."""

BUCKET_S = 0.5
"""Signal seconds per bucket, the step in which pairs leave the memory."""

GAP_S = 1.0
"""A time step longer than this starts a new stretch."""

SETTLE_S = 1.0
"""The first part of each stretch left out: the step response of the 1 Hz order-2 lowpass."""


class PairSpread:
    """The chance spread of one trial's result, in the unit of the readings.

    `feed` runs on the feed thread for every reading; `value` and `seconds` can be read from any thread.
    The reading rate comes from the first time step unless `hz` is given; readings sit on that grid, and
    a missing reading is a hole in the grid, not a step.
    """

    def __init__(self, baseline_s: float, active_s: float, hz: float | None = None):
        self.baseline_s = baseline_s
        self.active_s = active_s
        self._lock = threading.Lock()
        self._hz: float | None = None
        self._last_t: float | None = None
        self._stretch_t = 0.0
        self._pending: tuple[float, float, bool] | None = None
        self._buckets: deque = deque()
        self._seconds = 0.0
        self._cache: float | None = None
        self._dirty = False
        if hz is not None:
            self._grid(hz)

    def _grid(self, hz: float) -> None:
        self._hz = hz
        nb = max(1, int(round(self.baseline_s * hz)))
        na = max(1, int(round(self.active_s * hz)))
        w = np.r_[np.full(nb, -1.0 / nb), np.full(na, 1.0 / na)]
        k = len(w) - 1
        self._k = k
        self._weights = np.array([-float(np.dot(w[:-j], w[j:])) for j in range(1, k + 1)])
        self._lags = np.arange(1, k + 1, dtype=float)
        self._x = np.zeros(2 * k)                # each entry twice, so the last k are one slice
        self._ok = np.zeros(2 * k)
        self._at = 0
        self._sums = np.zeros(k)
        self._counts = np.zeros(k)
        self._new_bucket()

    def _new_bucket(self) -> None:
        self._bucket = [0.0, np.zeros(self._k), np.zeros(self._k)]
        self._buckets.append(self._bucket)

    def _push(self, x: float, ok: float) -> None:
        at, k = self._at, self._k
        self._x[at] = self._x[at + k] = x
        self._ok[at] = self._ok[at + k] = ok
        self._at = (at + 1) % k

    def _clear(self) -> None:
        self._x[:] = 0.0
        self._ok[:] = 0.0

    def feed(self, t: float, value: float, valid: bool = True, fresh: bool = False) -> None:
        """One reading at signal time `t`. A non-finite value counts as invalid. `fresh` starts a new
        stretch where the time runs on but the signal before does not precede this reading."""
        usable = bool(valid) and math.isfinite(value)
        with self._lock:
            last = self._last_t
            self._last_t = t
            fresh = fresh or last is None or t < last or t - last > GAP_S
            if fresh:
                self._stretch_t = t
            if self._hz is None:
                if fresh or t == last:
                    self._pending = (t, value, usable)
                    return
                self._grid(1.0 / (t - last))
                pt, pv, pu = self._pending
                self._pending = None
                self._take(pt, pv, pu, 0)
                last = pt
            elif fresh:
                self._clear()
                last = None
            self._take(t, value, usable, 0 if last is None else round((t - last) * self._hz) - 1)

    def _take(self, t: float, value: float, usable: bool, holes: int) -> None:
        for _ in range(min(max(holes, 0), self._k)):
            self._push(0.0, 0.0)
        usable = usable and t - self._stretch_t >= SETTLE_S
        if usable:
            at, k = self._at, self._k
            ok = self._ok[at:at + k][::-1]
            d = value - self._x[at:at + k][::-1]
            d *= d
            d *= ok
            bucket = self._bucket
            bucket[1] += d
            bucket[2] += ok
            self._sums += d
            self._counts += ok
            step = 1.0 / self._hz
            bucket[0] += step
            self._seconds += step
            if bucket[0] >= BUCKET_S - 1e-9:
                self._new_bucket()
                self._expire()
            self._dirty = True
        self._push(value if usable else 0.0, 1.0 if usable else 0.0)

    def _expire(self) -> None:
        while len(self._buckets) > 1 and self._seconds - self._buckets[0][0] >= MEMORY_S - 1e-9:
            seconds, sums, counts = self._buckets.popleft()
            self._seconds -= seconds
            self._sums -= sums
            self._counts -= counts

    @property
    def value(self) -> float | None:
        """The chance spread of one trial's result, or None before one usable pair exists."""
        with self._lock:
            if self._dirty:
                self._cache = self._spread()
                self._dirty = False
            return self._cache

    @property
    def seconds(self) -> float:
        """Signal seconds of readings behind the value, at most `MEMORY_S` plus one bucket."""
        with self._lock:
            return self._seconds

    def _spread(self) -> float | None:
        seen = self._counts > 0.5
        if not seen.any():
            return None
        d = np.zeros(self._k)
        d[seen] = np.maximum(self._sums[seen], 0.0) / self._counts[seen]
        if not seen.all():
            lags = self._lags[seen]
            slope = float(np.dot(lags, d[seen]) / np.dot(lags, lags))
            d[~seen] = slope * self._lags[~seen]
        return math.sqrt(max(0.0, float(np.dot(self._weights, d))))
