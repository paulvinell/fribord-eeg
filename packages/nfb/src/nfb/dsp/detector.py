"""A wavelet watching a signal as it arrives: its amplitude against its own usual size, and the bursts
that come out.

`wavelet.BurstDetector` takes a level in the amplitude's own unit, and a level in microvolts means a
different thing at every site and for every person. So the level here is a multiple of the amplitude's
median over the last minute: a detector at an unknown site starts at a level that means "well above
usual" there.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .wavelet import BurstDetector, Wavelet, WaveletEnvelope

USUAL_SECONDS = 60.0
"""How far back the amplitude's usual size is taken from."""

USUAL_STEP_SECONDS = 0.02
"""One amplitude in every this many seconds goes into the median. The amplitude is already smooth over the
wavelet's length, the shortest of which is 100 ms, so a finer step adds samples and no information."""

VETO_HOLD_SECONDS = 0.3
"""How long a veto lasts after the muscle reading falls. The wavelet's output outlasts the muscle by the
wavelet's own length, so a veto that ends with the clench lets the clench's tail through: on real jaw
clenches, 2 to 5 false bursts a minute without the hold and 0 to 1 with it."""

USUAL_EVERY_SECONDS = 0.25
"""How often the median is taken again. It moves over a minute, so four times a second is plenty."""


class RunningMedian:
    """The median of a quantity over the last `seconds`, streaming, from one sample in every `step`.

    The median rather than the mean, because the bursts themselves are in the window: a mean is pulled up
    by every burst it is meant to be the background for.
    """

    def __init__(self, rate_hz: float, seconds: float = USUAL_SECONDS,
                 step_seconds: float = USUAL_STEP_SECONDS, every_seconds: float = USUAL_EVERY_SECONDS):
        self.rate_hz = rate_hz
        self.stride = max(1, int(round(step_seconds * rate_hz)))
        self._kept: deque[float] = deque(maxlen=max(1, int(round(seconds / step_seconds))))
        self._every = max(1, int(round(every_seconds / step_seconds)))
        self._phase = 0
        self._since = 0
        self.value: float | None = None

    def update(self, x: np.ndarray) -> float | None:
        """Take a block in. Returns the median with it, or None before the first sample."""
        x = np.asarray(x, dtype=np.float64)
        picked = x[(-self._phase) % self.stride::self.stride]
        self._phase = (self._phase + len(x)) % self.stride
        self._kept.extend(picked.tolist())
        self._since += len(picked)
        # From the first sample, and again every `every`: a value is in use as soon as there is one.
        if self._kept and (self.value is None or self._since >= self._every):
            self.value = float(np.median(np.fromiter(self._kept, dtype=np.float64)))
            self._since = 0
        return self.value

    @property
    def seconds(self) -> float:
        """How much signal the median is taken over now, which is less than `seconds` at the start."""
        return len(self._kept) * self.stride / self.rate_hz


@dataclass
class Reading:
    """What one block came to, per sample. `usual` is the amplitude's median as of this block."""

    amplitude: np.ndarray
    usual: float | None
    on: np.ndarray


class LiveDetector:
    """One wavelet on one channel: amplitude, its usual size, and the bursts.

    `level` is a multiple of the usual size. `start` is the absolute index of the first sample this
    detector sees, so its bursts are in the same indices as everything else on the channel.

    A `veto` given to `process` holds a burst back from being confirmed while it is true and for
    `VETO_HOLD_SECONDS` after: a muscle reading above its level. Bursts it held back for good are
    counted by `vetoed`.

    Changing the detection settings keeps everything measured; a new wavelet is a new detector.
    """

    def __init__(self, wavelet: Wavelet, rate_hz: float, level: float = 2.5, min_seconds: float = 0.1,
                 start: int = 0):
        self.wavelet = wavelet
        self.rate_hz = rate_hz
        self.start = start
        self.envelope = WaveletEnvelope(wavelet, rate_hz)
        self.usual = RunningMedian(rate_hz)
        self._detector = BurstDetector(rate_hz, level, min_seconds)
        self._min_seconds = min_seconds
        # The same detector without the veto, started at the first veto: what the veto took away is the
        # bursts this one found and the vetoed one did not.
        self._unvetoed: BurstDetector | None = None
        self._unvetoed_from = 0
        self.samples = 0
        self._since_veto = 10 ** 9

    def _held(self, veto: np.ndarray) -> np.ndarray:
        """The veto, kept on for `VETO_HOLD_SECONDS` after its last true sample."""
        at = np.arange(len(veto))
        last = np.maximum.accumulate(np.where(veto, at, -1))
        since = np.where(last >= 0, at - last, self._since_veto + at + 1)
        if len(veto):
            self._since_veto = int(since[-1])
        return since <= int(VETO_HOLD_SECONDS * self.rate_hz)

    @property
    def level(self) -> float:
        return self._detector.level

    def set_detection(self, level: float, min_seconds: float) -> None:
        self._min_seconds = min_seconds
        for detector in (self._detector, self._unvetoed):
            if detector is not None:
                detector.set(level, min_seconds, self.rate_hz)

    def process(self, x: np.ndarray, veto: np.ndarray | None = None) -> Reading:
        amplitude = self.envelope.process(np.asarray(x, dtype=np.float64))
        usual = self.usual.update(amplitude)
        relative = amplitude / usual if usual else np.zeros_like(amplitude)
        gate = None
        if veto is not None:
            if self._unvetoed is None:
                self._unvetoed = BurstDetector(self.rate_hz, self._detector.level, self._min_seconds)
                self._unvetoed_from = self.samples
            self._unvetoed.process(relative)
            gate = ~self._held(np.asarray(veto, dtype=bool))
        else:
            self._unvetoed = None
        on = self._detector.process(relative, gate)
        self.samples += len(amplitude)
        return Reading(amplitude, usual, on)

    def vetoed(self, since: int) -> int:
        """How many bursts confirmed from `since` on would have counted without the veto, and did not."""
        if self._unvetoed is None:
            return 0
        offset = self.start + self._unvetoed_from
        kept = [b for b in self.found() if b[2] >= since]
        lost = 0
        for a, b, c in self._unvetoed.finish():
            a, b, c = a + offset, b + offset, c + offset
            if b >= since and not any(s < c and a < e for s, _, e in kept):
                lost += 1
        return lost

    def found(self) -> list[tuple[int, int, int]]:
        """Every burst so far as `(start, confirmed, end)` absolute indices, one still on ending now."""
        return [(self.start + a, self.start + b, self.start + c) for a, b, c in self._detector.finish()]
