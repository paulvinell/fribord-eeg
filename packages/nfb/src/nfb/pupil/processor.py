"""Which pupil readings to trust, and the diameter that is fed back.

Four rules reject a reading, and a rejected reading never moves the diameter: it holds the last good
value, as the HEG screen holds through motion.

- **No pupil, or a detector that is not sure.** A blink gives no diameter, and a lid edge gives a short one
  with a low confidence.
- **A diameter no pupil has.** Outside 1.5 to 9 mm is a detection fault, not a pupil.
- **A change faster than a pupil moves.** The light reflex constricts at a few millimetres a second; a
  step faster than `MAX_SPEED_MM_S` is a lid or a lost detection. The step is measured from the last
  reading that was kept, and after `RESYNC_S` without one the next reading is kept as it is, so a real
  change during a long rejection cannot lock every later reading out.
- **The first readings after a blink.** The lid is still moving off the pupil for a while after the
  detector finds it again.

What is kept is averaged over the last two readings, which is all the smoothing a 60 Hz diameter needs
before the screen draws it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .source import PupilSample

MIN_MM, MAX_MM = 1.5, 9.0
MIN_CONFIDENCE = 0.6
MAX_SPEED_MM_S = 10.0
RESYNC_S = 0.5
AFTER_BLINK_S = 0.1
SMOOTH_N = 2
WINDOW_S = 30.0
"""The rejected share is over this window: a live figure, so a bad minute shows while it happens."""
BLINK_WINDOW_S = 60.0


@dataclass(frozen=True)
class PupilState:
    t: float
    diameter_mm: float | None    # the last kept diameter, smoothed; None before the first one
    raw_mm: float | None         # what this reading said, kept or not
    valid: bool                  # whether this reading was kept
    rejected_fraction: float     # share of readings rejected over the last `WINDOW_S`
    blinks_per_min: float | None  # None until a full `BLINK_WINDOW_S` has been seen
    samples_seen: int


class PupilProcessor:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._kept: deque[float] = deque(maxlen=SMOOTH_N)
        self._last_kept: tuple[float, float] | None = None
        self._closed_until: float | None = None
        self._was_closed = False
        self._judged: deque[tuple[float, bool]] = deque()
        self._rejected = 0
        self._blinks: deque[float] = deque()
        self._first_t: float | None = None
        self.samples_seen = 0

    def _keep(self, s: PupilSample) -> bool:
        d = s.diameter_mm
        if d is None or s.confidence < MIN_CONFIDENCE:
            return False
        if not MIN_MM <= d <= MAX_MM:
            return False
        if self._closed_until is not None and s.t < self._closed_until:
            return False
        if self._last_kept is not None:
            t0, d0 = self._last_kept
            gap = s.t - t0
            if 0 < gap < RESYNC_S and abs(d - d0) / gap > MAX_SPEED_MM_S:
                return False
        return True

    def push(self, s: PupilSample) -> PupilState:
        self.samples_seen += 1
        if self._first_t is None:
            self._first_t = s.t
        closed = s.diameter_mm is None
        if closed and not self._was_closed:
            self._blinks.append(s.t)
        if closed:
            self._closed_until = s.t + AFTER_BLINK_S
        self._was_closed = closed

        kept = self._keep(s)
        if kept:
            self._kept.append(s.diameter_mm)
            self._last_kept = (s.t, s.diameter_mm)

        self._judged.append((s.t, kept))
        self._rejected += not kept
        while self._judged and s.t - self._judged[0][0] > WINDOW_S:
            self._rejected -= not self._judged.popleft()[1]
        while self._blinks and s.t - self._blinks[0] > BLINK_WINDOW_S:
            self._blinks.popleft()
        rejected = self._rejected / len(self._judged)
        full = s.t - self._first_t >= BLINK_WINDOW_S
        return PupilState(
            t=s.t,
            diameter_mm=sum(self._kept) / len(self._kept) if self._kept else None,
            raw_mm=s.diameter_mm,
            valid=kept,
            rejected_fraction=rejected,
            blinks_per_min=len(self._blinks) * 60.0 / BLINK_WINDOW_S if full else None,
            samples_seen=self.samples_seen,
        )
