"""Rolling building blocks over the processed stream: what the signal looks like right now.

This file answers "what does the signal look
like", and what the trainee should *see* is the metric/strategy layer's job. Every window is
time-based, in seconds, so the numbers mean the same thing at the simulator's rate and the device's.

One departure, learned on the HRV screen the hard way: **motion-gated frames do not enter the
buffers.** The processor holds them flat at the last good value, which is right for continuity — but a
2 s cooldown of repeated constants entering a 5 s stability window reads as sudden calm, and
"variability collapses whenever you move your head" trains head-stillness, not the brain. The hole
this leaves is spanned by the time-based windows exactly as a withheld beat is spanned by the
tachogram's interpolation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .processor import QUALITY_MOTION, Processed


@dataclass(frozen=True)
class HegState:
    """A snapshot after each frame. Everything a screen draws or a metric reads comes from here."""

    t: float = 0.0
    hbo: float = 0.0
    hbr: float = 0.0
    raw_ir: float = 0.0
    quality: float = 0.0
    smoothed_hbo: float = 0.0
    smoothed_hbr: float = 0.0
    hbo_std_short: float = 0.0
    oxy_std_short: float = 0.0
    hbo_slope: float = 0.0
    samples_seen: int = 0
    gate_fraction: float = 0.0
    temp_c: float = 0.0


def _trim(buf: deque, cutoff_t: float) -> None:
    while buf and buf[0][0] < cutoff_t:
        buf.popleft()


def _values(buf: deque) -> np.ndarray:
    return np.fromiter((v for _, v in buf), dtype=np.float64, count=len(buf))


class SignalProcessor:
    """Time-windowed smoothing, stability and trend over the projected HbO/HbR stream."""

    def __init__(self, smoothing_window_s: float = 2.0, trend_window_s: float = 5.0,
                 stability_window_s: float = 5.0):
        self.smoothing_window_s = smoothing_window_s
        self.trend_window_s = trend_window_s
        self.stability_window_s = stability_window_s
        self._hbo: deque = deque()
        self._hbr: deque = deque()
        self._stab: deque = deque()
        self._oxy_stab: deque = deque()
        self._trend: deque = deque()
        self._n = 0
        self._last: HegState | None = None

    def reset(self) -> None:
        for buf in (self._hbo, self._hbr, self._stab, self._oxy_stab, self._trend):
            buf.clear()
        self._n = 0
        self._last = None

    def push(self, sample: Processed, gate_fraction: float = 0.0) -> HegState:
        self._n += 1
        t = sample.t
        if sample.quality == QUALITY_MOTION and self._last is not None:
            # Held-flat frames advance the clock, not the statistics — see the module docstring.
            state = HegState(**{**self._last.__dict__, "t": t, "quality": sample.quality,
                                "samples_seen": self._n, "gate_fraction": gate_fraction})
            self._last = state
            return state

        self._hbo.append((t, sample.hbo))
        self._hbr.append((t, sample.hbr))
        _trim(self._hbo, t - self.smoothing_window_s)
        _trim(self._hbr, t - self.smoothing_window_s)
        smoothed_hbo = float(_values(self._hbo).mean())
        smoothed_hbr = float(_values(self._hbr).mean())

        self._stab.append((t, smoothed_hbo))
        _trim(self._stab, t - self.stability_window_s)
        hbo_std = float(_values(self._stab).std()) if len(self._stab) >= 2 else 0.0

        self._oxy_stab.append((t, smoothed_hbo - smoothed_hbr))
        _trim(self._oxy_stab, t - self.stability_window_s)
        oxy_std = float(_values(self._oxy_stab).std()) if len(self._oxy_stab) >= 2 else 0.0

        self._trend.append((t, smoothed_hbo))
        _trim(self._trend, t - self.trend_window_s)
        slope = 0.0
        if len(self._trend) >= 2:
            ts = np.fromiter((p[0] for p in self._trend), dtype=np.float64, count=len(self._trend))
            ys = _values(self._trend)
            ts = ts - ts[0]
            if ts[-1] > 1e-6:
                num = float(((ts - ts.mean()) * (ys - ys.mean())).sum())
                den = float(((ts - ts.mean()) ** 2).sum())
                slope = num / den if den > 1e-9 else 0.0

        state = HegState(t=t, hbo=sample.hbo, hbr=sample.hbr, raw_ir=sample.raw_ir,
                         quality=sample.quality, smoothed_hbo=smoothed_hbo,
                         smoothed_hbr=smoothed_hbr, hbo_std_short=hbo_std,
                         oxy_std_short=oxy_std, hbo_slope=slope, samples_seen=self._n,
                         gate_fraction=gate_fraction, temp_c=sample.temp_c)
        self._last = state
        return state
