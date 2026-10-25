"""Where pupil diameters come from: a simulator, or the eye tracker in `tracker`.

**The simulator is a person, not a sine.** A screen that trains voluntary control has to be seen doing
its whole job — a cue, a baseline, an effort, a verdict — and a signal that ignores the cue can only ever
show failed trials. So `SimulatedPupil` takes an `effort` from outside, −1 to 1, and moves toward it with
the lag a real pupil has: the voluntary change becomes visible about a second after the effort starts,
and it is small, a few tenths of a millimetre on a pupil of four or five. Around that it draws what makes
the real signal hard: slow drift, the pupil's own unrest at a fraction of a hertz, measurement noise,
and blinks, each with a lid edge on both sides where the pupil is partly covered.

Nothing in acquisition knows what the effort is for. It is a property of the simulated eye, and whoever
holds the source sets it.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Iterator, Protocol

PUPIL_RATE_HZ = 60.0
"""The simulator's sample rate. 60 Hz is what a screen-based eye tracker delivers; every window
downstream is time-based, so this is a default and not a contract."""


@dataclass(frozen=True)
class PupilSample:
    """One reading of one eye. `diameter_mm` is None when no pupil was found (a closed eye)."""

    t: float                    # epoch seconds
    diameter_mm: float | None
    confidence: float           # 0 to 1: how sure the detector is that it saw the whole pupil
    distance_mm: float | None = None    # eyes to sensor, where the source knows it
    left_mm: float | None = None        # each eye, where the source measures both
    right_mm: float | None = None


class PupilSource(Protocol):
    def samples(self) -> Iterator[PupilSample]: ...
    def close(self) -> None: ...
    def describe(self) -> str: ...


def problem(kind: str = "simulator") -> str | None:
    """Why a source would not open, before it is tried. The simulator always opens."""
    if kind == "tracker":
        from .tracker import problem as tracker_problem
        return tracker_problem()
    return None


class SimulatedPupil:
    """Synthetic diameters at 60 Hz from an eye that answers an effort.

    `effort` is read on every sample, so it can be changed while the iterator runs. `response_mm` is how
    far a full effort moves the pupil once it has settled: zero is a person with no control at all, and
    a few tenths of a millimetre is a trained one. Making the pupil smaller is harder than making it
    larger, so a downward effort gets `DOWN_SHARE` of the response.
    """

    DOWN_SHARE = 0.7
    RESPONSE_TAU_S = 2.0
    DRIFT_TAU_S = 90.0
    DRIFT_SD_MM = 0.25
    UNREST_TAU_S = 2.0
    UNREST_SD_MM = 0.04
    NOISE_MM = 0.012
    BLINK_EVERY_S = 4.0
    BLINK_S = (0.10, 0.30)
    LID_EDGE_S = 0.05

    def __init__(self, base_mm: float = 4.5, response_mm: float = 0.25,
                 sample_rate_hz: float = PUPIL_RATE_HZ, seed: int | None = None,
                 realtime: bool = True, blinks: bool = True):
        self.base_mm = base_mm
        self.response_mm = response_mm
        self.effort = 0.0
        self.realtime = realtime
        self.blinks = blinks
        self._dt = 1.0 / max(1.0, sample_rate_hz)
        self._rng = random.Random(seed)
        self._stop = False

    def _next_blink(self, now: float) -> float:
        return now + max(1.5, self._rng.expovariate(1.0 / self.BLINK_EVERY_S))

    def samples(self) -> Iterator[PupilSample]:
        rng, dt = self._rng, self._dt
        t0, elapsed = time.time(), 0.0
        drift = unrest = answer = 0.0
        phases = (rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi))
        blink_at = self._next_blink(0.0)
        blink_len = rng.uniform(*self.BLINK_S)
        while not self._stop:
            drift += (-drift * dt / self.DRIFT_TAU_S
                      + self.DRIFT_SD_MM * math.sqrt(2 * dt / self.DRIFT_TAU_S) * rng.gauss(0, 1))
            unrest += (-unrest * dt / self.UNREST_TAU_S
                       + self.UNREST_SD_MM * math.sqrt(2 * dt / self.UNREST_TAU_S) * rng.gauss(0, 1))
            effort = max(-1.0, min(1.0, float(self.effort)))
            wanted = effort * self.response_mm * (1.0 if effort >= 0 else self.DOWN_SHARE)
            answer += (wanted - answer) * (1.0 - math.exp(-dt / self.RESPONSE_TAU_S))
            hippus = (0.05 * math.sin(2 * math.pi * 0.07 * elapsed + phases[0])
                      + 0.03 * math.sin(2 * math.pi * 0.21 * elapsed + phases[1]))
            diameter = (self.base_mm + drift + unrest + hippus + answer
                        + rng.gauss(0, self.NOISE_MM))
            confidence = 0.95

            if self.blinks:
                if elapsed >= blink_at + blink_len + self.LID_EDGE_S:
                    blink_at = self._next_blink(elapsed)
                    blink_len = rng.uniform(*self.BLINK_S)
                if blink_at <= elapsed < blink_at + blink_len:
                    diameter, confidence = None, 0.0
                elif (blink_at - self.LID_EDGE_S <= elapsed < blink_at
                      or blink_at + blink_len <= elapsed < blink_at + blink_len + self.LID_EDGE_S):
                    # The lid covers the top of the pupil: the detector sees a shorter pupil, and knows it.
                    diameter, confidence = diameter * 0.8, 0.4

            yield PupilSample(t=t0 + elapsed, diameter_mm=diameter, confidence=confidence)
            elapsed += dt
            if self.realtime:
                time.sleep(dt)

    def close(self) -> None:
        self._stop = True

    def describe(self) -> str:
        return f"simulated eye · {1.0 / self._dt:g} readings/s"


def pupil_source(kind: str, **options) -> PupilSource:
    if kind == "simulator":
        return SimulatedPupil(**options)
    if kind == "tracker":
        from .tracker import TrackerPupil
        return TrackerPupil(**options)
    raise ValueError(f"no pupil source called {kind!r}")
