"""A spindle on one channel: a short burst of a rhythm far above its usual size, and the search for the
frequency a person's spindles are at.

The numbers are from one recording at two central sites, 1000 sps: spindles near 25 Hz, about 0.3 s long,
15 µV at the peak over a background of 3 µV, five a minute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .detector import LiveDetector, RunningMedian
from .fit import TooFew, fit_design
from .wavelet import MuscleEnvelope, Wavelet, WaveletEnvelope

SPINDLE_HZ = 25.0

SPINDLE_CYCLES = 4.0
"""Short on purpose. Single spindles were from 21 to 29 Hz, and four cycles are wide enough to hold them
all. Four found as many as ten and confirmed them 100 ms sooner."""

SPINDLE_LEVEL = 4.0
"""Times the amplitude's median over the last minute. Found 91 to 93% of the spindles 155 to 190 ms after
their start, and 0 to 0.2 a minute in a copy of the signal with its timing scrambled."""

MUSCLE_LEVEL = 2.0
"""Times the muscle reading's median over the last minute. At 3 a clench at a fifth of full strength got
through on the channel farther from the jaw."""

SEARCH_BAND = (13.0, 35.0)
SEARCH_SECONDS = 300.0
SEARCH_RATE_HZ = 250.0
"""The search keeps its signal at this rate: the fit gave the same frequency within 0.1 Hz as at
1000 sps, in a fifth of the time."""


class SpindleDetector:
    """Samples in, spindles out. With `reject_muscle`, a spindle is not confirmed while the channel's own
    muscle reading is high: a hard jaw clench reads as thirty to forty spindles a minute without it."""

    def __init__(self, rate_hz: float, frequency_hz: float = SPINDLE_HZ, level: float = SPINDLE_LEVEL,
                 reject_muscle: bool = False):
        self.rate_hz = rate_hz
        self.frequency_hz = frequency_hz
        self._live = LiveDetector(Wavelet(frequency_hz, SPINDLE_CYCLES), rate_hz, level=level,
                                  min_seconds=0.0)
        self._muscle = MuscleEnvelope(rate_hz) if reject_muscle else None
        self._muscle_usual = RunningMedian(rate_hz)
        self._on = False
        self.count = 0
        self.amplitude = np.zeros(0)
        self.usual: float | None = None

    @property
    def level(self) -> float:
        return self._live.level

    def set_level(self, level: float) -> None:
        """Change how large a spindle must be, from the next sample. Everything measured is kept."""
        self._live.set_detection(level, 0.0)

    @property
    def level_uv(self) -> float | None:
        """The level in microvolts now: the multiple times the usual amplitude."""
        return None if self.usual is None else self.level * self.usual

    @property
    def on(self) -> bool:
        """Whether a confirmed spindle is on at the newest sample."""
        return self._on

    @property
    def rejected(self) -> int:
        """How many spindles the muscle reading held back."""
        return self._live.vetoed(0)

    def found(self) -> list[tuple[int, int, int]]:
        """Every spindle so far as `(start, confirmed, end)` sample indices."""
        return self._live.found()

    def process(self, x: np.ndarray) -> int:
        """Take a block in. Returns how many spindles were confirmed in it."""
        x = np.asarray(x, dtype=np.float64)
        veto = None
        if self._muscle is not None:
            reading = self._muscle.process(x)
            usual = self._muscle_usual.update(reading)
            veto = reading >= MUSCLE_LEVEL * usual if usual else np.zeros(len(x), dtype=bool)
        result = self._live.process(x, veto)
        self.amplitude, self.usual = result.amplitude, result.usual
        if not len(x):
            return 0
        before = np.concatenate(([self._on], result.on[:-1]))
        confirmed = int(np.count_nonzero(result.on & ~before))
        self._on = bool(result.on[-1])
        self.count += confirmed
        return confirmed


@dataclass(frozen=True)
class Found:
    """What a search has so far. There is always a frequency; `spindles` says what it rests on."""

    frequency_hz: float
    spindles: int
    seconds: float


def _turning(x: np.ndarray, rate_hz: float, wavelet: Wavelet) -> float | None:
    """How fast the wavelet's output turns over the whole stretch, each sample counting by its power."""
    n = wavelet.taps(rate_hz)
    if len(x) < 2 * n:
        return None
    y = WaveletEnvelope(wavelet, rate_hz).analytic(x)[n:]
    return float(np.angle(np.sum(y[1:] * np.conj(y[:-1]))) * rate_hz / (2 * np.pi))


def strongest_hz(x: np.ndarray, rate_hz: float, low_hz: float, high_hz: float) -> float | None:
    """Where the band is strongest, from any length of signal a wavelet fits in twice.

    It needs no bursts, so it answers within a quarter of a second. On the recording it was within 3 Hz
    of the fitted frequency after one second and within 1 Hz after ten.
    """
    centre = (low_hz + high_hz) / 2
    first = _turning(x, rate_hz, Wavelet(centre, max(2.0, centre / (high_hz - low_hz))))
    if first is None:
        return None
    first = float(np.clip(first, low_hz, high_hz))
    second = _turning(x, rate_hz, Wavelet(first, SPINDLE_CYCLES))
    return first if second is None else float(np.clip(second, low_hz, high_hz))


class SpindleSearch:
    """Keeps the signal while it runs, and says at any moment where its spindles are.

    `push` takes every block; `estimate` answers from what has come so far, from the first block. The
    answer is `start_hz` until a wavelet fits in the signal, then where the band is strongest, then the
    fit of the spindles themselves once there are enough: on the recording after 30 to 60 s, and within
    0.5 Hz of it from then on.
    """

    def __init__(self, rate_hz: float, start_hz: float = SPINDLE_HZ,
                 band: tuple[float, float] = SEARCH_BAND, seconds: float = SEARCH_SECONDS):
        self.band = band
        self.start_hz = start_hz
        self._step = max(1, int(rate_hz // SEARCH_RATE_HZ))
        self.rate_hz = rate_hz / self._step
        self._sos = signal.butter(8, 0.4 * self.rate_hz, fs=rate_hz, output="sos") if self._step > 1 else None
        self._zi = signal.sosfilt_zi(self._sos) * 0.0 if self._step > 1 else None
        self._phase = 0
        self._kept = np.zeros(int(seconds * self.rate_hz))
        self._filled = 0

    @property
    def seconds(self) -> float:
        return self._filled / self.rate_hz

    def push(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if self._sos is not None and len(x):
            x, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
        picked = x[(-self._phase) % self._step::self._step]
        self._phase = (self._phase + len(x)) % self._step
        n = min(len(picked), len(self._kept))
        if n:
            self._kept = np.roll(self._kept, -n)
            self._kept[-n:] = picked[-n:]
            self._filled = min(len(self._kept), self._filled + n)

    def estimate(self) -> Found:
        x = self._kept[len(self._kept) - self._filled:]
        try:
            if self.seconds < 1.0:          # shorter than the wavelets the fit starts with
                raise TooFew(0, "so far")
            fit = fit_design(x, self.rate_hz, *self.band)
        except TooFew as few:
            strongest = strongest_hz(x, self.rate_hz, *self.band)
            return Found(round(strongest, 1) if strongest else self.start_hz, few.bursts, self.seconds)
        return Found(fit.wavelet.frequency_hz, fit.bursts, self.seconds)
