"""Wavelets for burst detection: a short template of a rhythm, slid along the signal.

A cFIR is selective by being long, one second for every band. A rhythm that comes in bursts is an event
with a length of its own, and the best linear detector of a known shape in noise is that shape run
backwards as a filter. So the length here is a number of *cycles*, and the outline is part of the design.

Every design reads a steady sine at its frequency at its true amplitude (peak convention, as the cFIR)
with the output's angle on the newest sample, and reads nothing for a standing offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

@dataclass(frozen=True)
class Wavelet:
    """What a wavelet looks for.

    `cycles` is the whole length, in cycles of `frequency_hz` — the control that trades speed against
    selectivity. `rise` is where the outline peaks, as a fraction of the length in the template's own
    time: 0.5 is a burst that grows and fades evenly, a smaller value one that starts abruptly and fades.
    """

    frequency_hz: float
    cycles: float
    rise: float = 0.5

    def __post_init__(self):
        if self.frequency_hz <= 0:
            raise ValueError(f"invalid frequency {self.frequency_hz}")
        if self.cycles < 1:
            raise ValueError("a wavelet needs at least one cycle to name a frequency")
        if not 0.0 <= self.rise <= 1.0:
            raise ValueError(f"rise is a fraction of the length, not {self.rise}")

    @property
    def seconds(self) -> float:
        return self.cycles / self.frequency_hz

    def taps(self, rate_hz: float) -> int:
        return max(4, int(round(self.seconds * rate_hz)))


def outline(n: int, rise: float = 0.5) -> np.ndarray:
    """How the wavelet grows and fades: two half-cosines meeting at the peak, zero at both ends."""
    m = np.arange(n, dtype=np.float64)
    peak = rise * (n - 1)
    out = np.empty(n)
    up = m <= peak
    out[up] = 0.5 * (1 - np.cos(np.pi * m[up] / peak)) if peak > 0 else 1.0
    out[~up] = 0.5 * (1 + np.cos(np.pi * (m[~up] - peak) / (n - 1 - peak)))
    return out


def gain_at(taps: np.ndarray, frequency_hz: float, rate_hz: float) -> complex:
    """Complex response of the taps at one frequency."""
    k = np.arange(len(taps))
    return complex(np.sum(taps * np.exp(-2j * np.pi * frequency_hz * k / rate_hz)))


def design_wavelet(wavelet: Wavelet, rate_hz: float) -> np.ndarray:
    """Complex taps whose output is the wavelet's match against the signal ending at each sample.

    Returns taps for `lfilter`. The modulus of the output is amplitude and its angle is phase.
    """
    if wavelet.frequency_hz >= rate_hz / 2:
        raise ValueError(f"{wavelet.frequency_hz:g} Hz is above Nyquist at {rate_hz:g} sps")
    n = wavelet.taps(rate_hz)
    env = outline(n, wavelet.rise)
    # Phase is counted back from the newest sample, so the output's angle is the rhythm's phase now.
    template = env * np.exp(2j * np.pi * wavelet.frequency_hz * (np.arange(n) - (n - 1)) / rate_hz)
    # The offset a short template would pass, taken out along the outline so the ends stay at zero.
    template = template - env * (template.sum() / env.sum())
    taps = np.conj(template[::-1])
    gain = gain_at(taps, wavelet.frequency_hz, rate_hz)
    if gain == 0:
        raise ValueError("the wavelet has no response at its own frequency")
    # 2 as in `design_cfir`; the complex gain, not its size, so the angle is the phase at the newest sample.
    return taps * (2.0 / gain)


def width_hz(taps: np.ndarray, frequency_hz: float, rate_hz: float, n_fft: int = 8192) -> float:
    """How far apart the two frequencies are at which a sine reads 71% of its amplitude."""
    freqs = np.fft.fftfreq(n_fft, 1 / rate_hz)
    response = np.abs(np.fft.fft(taps, n_fft)) / 2.0
    keep = freqs > 0
    freqs, response = freqs[keep], response[keep]
    order = np.argsort(freqs)
    freqs, response = freqs[order], response[order]
    centre = int(np.argmin(np.abs(freqs - frequency_hz)))
    lo = hi = centre
    while lo > 0 and response[lo] >= np.sqrt(0.5):
        lo -= 1
    while hi < len(freqs) - 1 and response[hi] >= np.sqrt(0.5):
        hi += 1
    return float(freqs[hi] - freqs[lo])


class WaveletEnvelope:
    """Streaming wavelet: blocks of samples in, the match out. The counterpart of `CFIREnvelope`."""

    def __init__(self, wavelet: Wavelet, rate_hz: float):
        self.wavelet = wavelet
        self.rate_hz = rate_hz
        self.taps = design_wavelet(wavelet, rate_hz)
        self._zi = np.zeros(len(self.taps) - 1, dtype=complex)

    def reset(self) -> None:
        self._zi = np.zeros(len(self.taps) - 1, dtype=complex)

    def analytic(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not len(x):
            return np.zeros(0, dtype=complex)
        y, self._zi = signal.lfilter(self.taps, [1.0], x, zi=self._zi)
        return y

    def process(self, x: np.ndarray) -> np.ndarray:
        return np.abs(self.analytic(x))

    @property
    def width_hz(self) -> float:
        return width_hz(self.taps, self.wavelet.frequency_hz, self.rate_hz)


class _Steady:
    """One-pole average of a per-sample quantity, streaming."""

    def __init__(self, seconds: float, rate_hz: float):
        self._a = 1.0 - np.exp(-1.0 / (seconds * rate_hz))
        self._zi = np.zeros(1)

    def reset(self) -> None:
        self._zi = np.zeros(1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self._zi = signal.lfilter([self._a], [1.0, self._a - 1.0], x, zi=self._zi)
        return y


def bursts(amplitude: np.ndarray, rate_hz: float, level: float, min_seconds: float = 0.0,
           release: float = 0.7, gate: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Stretches where the amplitude is a burst: `(start, confirmed, end)` sample indices.

    A burst starts where the amplitude rises above `level` and ends where it falls below
    `release × level`, so one burst that dips is not two. It counts only if it lasts `min_seconds`, and
    `confirmed` is the sample at which that was known — the earliest a live detector can act on it.
    With a `gate`, one truth value per sample, it is confirmed at the first sample from then on at
    which the gate holds, and a burst that ends before the gate holds does not count.
    """
    detector = BurstDetector(rate_hz, level, min_seconds, release)
    detector.process(np.asarray(amplitude, dtype=np.float64), gate)
    return detector.finish()


class BurstDetector:
    """`bursts`, streaming: amplitude in, whether a confirmed burst is on at each sample out."""

    def __init__(self, rate_hz: float, level: float, min_seconds: float = 0.0, release: float = 0.7):
        self.level = level
        self.release = release
        self.need = int(round(min_seconds * rate_hz))
        self.found: list[tuple[int, int, int]] = []
        self._i = 0
        self._start: int | None = None
        self._confirmed: int | None = None

    def set(self, level: float, min_seconds: float, rate_hz: float) -> None:
        """Change what counts as a burst from the next sample; a burst that is on stays on."""
        self.level = level
        self.need = int(round(min_seconds * rate_hz))

    def process(self, amplitude: np.ndarray, gate: np.ndarray | None = None) -> np.ndarray:
        on = np.zeros(len(amplitude), dtype=bool)
        for k, a in enumerate(amplitude):
            i = self._i + k
            if self._start is None:
                if a >= self.level:
                    self._start = i
            elif a < self.release * self.level:
                self._close(i)
            if (self._start is not None and self._confirmed is None and i - self._start >= self.need
                    and (gate is None or gate[k])):
                self._confirmed = i
            on[k] = self._confirmed is not None
        self._i += len(amplitude)
        return on

    def _close(self, end: int) -> None:
        if self._confirmed is not None:
            self.found.append((self._start, self._confirmed, end))
        self._start = self._confirmed = None

    def finish(self) -> list[tuple[int, int, int]]:
        """Every burst so far, with one still on closed at the newest sample."""
        open_now = [(self._start, self._confirmed, self._i)] if self._confirmed is not None else []
        return self.found + open_now


MAINS_HZ = 50.0

MUSCLE_SECONDS = 0.05


def muscle_band(rate_hz: float, mains_hz: float = MAINS_HZ) -> tuple[float, float]:
    """Where the muscle reading listens: above the rhythms, and clear of the mains' strong harmonics.

    Recorded at 1000 sps with the mains' own frequency notched out, a scalp channel at rest carried
    11 to 19 µV at twice the mains and 3 to 5 µV at four times it, against 2 µV of everything else above
    55 Hz. A reading that includes them barely moves for a clench. So it listens between the two where
    the rate reaches, 110 to 190 Hz, and between the mains and its double where it does not, 55 to 95 Hz.
    """
    low, high = 2 * mains_hz + 10, 4 * mains_hz - 10
    if high < 0.45 * rate_hz:
        return low, high
    return mains_hz + 5, 2 * mains_hz - 5


class MuscleEnvelope:
    """How strong a channel is where muscle is and the brain's rhythms are not, per sample.

    Muscle is strong far above 55 Hz and the brain is not, so a burst on another channel while this is
    high is the muscle's. On a scalp channel clenched at the jaw it read 3 to 7 times its resting size.
    """

    def __init__(self, rate_hz: float, band: tuple[float, float] | None = None,
                 seconds: float = MUSCLE_SECONDS):
        low, high = band or muscle_band(rate_hz)
        if high >= rate_hz / 2:
            raise ValueError(f"{high:g} Hz is above Nyquist at {rate_hz:g} sps")
        self._sos = signal.butter(6, [low, high], btype="bandpass", fs=rate_hz, output="sos")
        self._zi = signal.sosfilt_zi(self._sos) * 0.0
        self._power = _Steady(seconds, rate_hz)

    def reset(self) -> None:
        self._zi = signal.sosfilt_zi(self._sos) * 0.0
        self._power.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not len(x):
            return np.zeros(0)
        y, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
        return np.sqrt(self._power(y * y))
