"""A heart that answers the paced breath through a resonance, for tests of a pacer that the heart corrects.

The response of the heart rate to the size of the breath is a second-order band-pass: its gain is largest,
and its lag is zero, at the heart's own rate; above that rate the heart rate peaks late, below it early. A beat
falls where the summed rate reaches a whole beat, so an interval ends at its beat, as the processor reads it,
and carries noise. The rate has a slow drift. Nothing here comes from a recording.
"""

from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np

from nfb.hrv import BeatState
from nfb.hrv.sweep import breath_turn

from conftest import SWEEP_EPOCH

STEP_S = 0.05
"""The step of the filter. At six breaths a minute it is a hundredth of a cycle."""

WINDOW_S = 180.0
"""The beats that the state lists, as the processor's buffer does."""


def volume(cycles: float, share: float) -> float:
    """The size of the breath, 0 to 1, after `cycles` of pacer: the ring's own turn function."""
    inhale, phase = breath_turn(cycles % 1.0, share)
    return phase if inhale else 1.0 - phase


def fundamental(share: float) -> float:
    """The amplitude of the fundamental of `volume`, so that a unit of input swings the heart by its gain."""
    k = 1.0 / (share * (1.0 - share))
    return k * math.sin(math.pi * share) / math.pi ** 2


def response(rate_per_min: float, resonance_per_min: float, q: float) -> tuple[float, float]:
    """The gain and the lag, in a share of the breath (positive is late), of the band-pass at a breathing
    rate: `1 / (1 + jQ (r - 1/r))` with `r` the rate over the resonance."""
    r = rate_per_min / resonance_per_min
    x = q * (r - 1.0 / r)
    return 1.0 / math.hypot(1.0, x), math.atan(x) / (2.0 * math.pi)


class ResonantHeart:
    """Beats whose rate follows the paced breath. `advance` takes the heart forward to a time, with the
    cycles of the pacer as a function of time, and `state` is what the processor would show then."""

    def __init__(self, resonance_per_min: float, *, q: float = 1.5, swing_bpm: float = 3.0,
                 base_bpm: float = 66.0, noise_ms: float = 15.0, drift_bpm: float = 0.0,
                 drift_period_s: float = 420.0, seed: int = 0, start_s: float = 0.0):
        self.resonance_per_min = resonance_per_min
        self.q = q
        self.swing_bpm = swing_bpm
        self.base_bpm = base_bpm
        self.noise_ms = noise_ms
        self.drift_bpm = drift_bpm
        self.drift_period_s = drift_period_s
        self._w0 = 2.0 * math.pi * resonance_per_min / 60.0
        self._rng = np.random.default_rng(seed)
        self._phase0 = float(self._rng.uniform(0.0, 2.0 * math.pi))
        self._x = self._v = 0.0
        self._t = start_s
        self._acc = 0.3                     # the heart's own phase, in beats; a beat falls at each whole one
        self._last: float | None = None
        self.stamps: list[float] = []       # every beat made, whether or not a look saw it
        self.rr_ms: list[float] = []

    def rate_bpm(self, t: float, y: float) -> float:
        drift = self.drift_bpm * math.sin(2.0 * math.pi * t / self.drift_period_s + self._phase0)
        return self.base_bpm + drift + self.swing_bpm * y

    def advance(self, to_s: float, cycles_at, share: float = 0.5, breathing: bool = True) -> None:
        """Run the filter and the beats up to `to_s`. `cycles_at(t)` gives the pacer's cycles at a time;
        `breathing` False is a heart that does not follow it (the input is the mean size)."""
        w, q, amp = self._w0, self.q, fundamental(share)
        while self._t < to_s - 1e-12:
            step = min(STEP_S, to_s - self._t)
            u = (volume(cycles_at(self._t), share) - 0.5) / amp if breathing else 0.0
            self._v += step * (u - (w / q) * self._v - w * w * self._x)
            self._x += step * self._v
            self._t += step
            rate = self.rate_bpm(self._t, (w / q) * self._v) / 60.0
            self._acc += step * rate
            if self._acc >= 1.0:
                # A beat falls where the rate, summed, reaches a whole beat, so an interval is the rate over it.
                stamp = self._t - (self._acc - 1.0) / rate
                self._acc -= 1.0
                stamp += float(self._rng.normal(0.0, self.noise_ms / 1000.0 / math.sqrt(2.0)))
                self.rr_ms.append(60000.0 / self.base_bpm if self._last is None else 1000.0 * (stamp - self._last))
                self.stamps.append(stamp)
                self._last = stamp

    def state(self, lag_s: float = 0.0, until_s: float | None = None, holes=()) -> BeatState:
        """The snapshot after the newest beat that is no later than `until_s` (the heart's own time when
        None), without the beats that fall in a `(from, to)` of `holes`, where the sensor gave none. A delay
        of `lag_s` moves every stamp and the arrival with it, as a slow link does."""
        last = len(self.stamps) if until_s is None else bisect_right(self.stamps, until_s)
        first = bisect_right(self.stamps, (self.stamps[last - 1] if last else 0.0) - WINDOW_S)
        keep = [i for i in range(first, last) if not any(a <= self.stamps[i] < b for a, b in holes)]
        if not keep:
            return BeatState(t=SWEEP_EPOCH + (self._t if until_s is None else until_s) + lag_s)
        newest = self.stamps[keep[-1]]
        return BeatState(t=SWEEP_EPOCH + lag_s + newest, age_s=[newest - self.stamps[i] for i in keep],
                         tachogram=[60000.0 / self.rr_ms[i] for i in keep])


def sweep_heart(resonance_per_min: float, seed: int, *, schedule=None, share: float = 0.5, **heart):
    """A whole sweep over beats of a `ResonantHeart` that the schedule's pacer drives, looked at every half
    second as the screen does. Returns the ended `Sweep`. `heart` is passed to the heart."""
    from conftest import SWEEP_MONO0
    from nfb.hrv.sweep import Sweep, SweepSchedule

    schedule = schedule or SweepSchedule()
    n = schedule.breaths
    starts = np.array([schedule.start_s(k) for k in range(n)])
    periods = np.array([schedule.period_s(k) for k in range(n)])

    def cycles(t):
        k = 0 if t < 0.0 else min(n - 1, int(np.searchsorted(starts, t, side="right")) - 1)
        return k + (t - starts[k]) / periods[k]

    pulse = ResonantHeart(resonance_per_min, seed=seed, **heart)
    sweep = Sweep(schedule)
    t = 0.0
    while sweep.active or t == 0.0:
        pulse.advance(t, cycles, share)
        sweep.feed(SWEEP_MONO0 + t, pulse.state(), None, None, share)
        t += 0.5
        assert t < schedule.total_s + 60.0, "the sweep did not end"
    return sweep


def sweep_beats(resonance_per_min: float, seed: int, schedule, share: float = 0.5, **heart):
    """The beats of a `ResonantHeart` that the schedule's pacer drives from second 0 of the sweep to 20 s
    after its end, as stamps and intervals, for a test that feeds them to a session itself."""
    n = schedule.breaths
    starts = np.array([schedule.start_s(k) for k in range(n)])
    periods = np.array([schedule.period_s(k) for k in range(n)])

    def cycles(t):
        k = 0 if t < 0.0 else min(n - 1, int(np.searchsorted(starts, t, side="right")) - 1)
        return k + (t - starts[k]) / periods[k]

    pulse = ResonantHeart(resonance_per_min, seed=seed, start_s=-20.0, **heart)
    pulse.advance(schedule.total_s + 20.0, cycles, share)
    return np.array(pulse.stamps), np.array(pulse.rr_ms)
