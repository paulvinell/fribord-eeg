"""Fit a wavelet to the bursts in a stretch of signal: their frequency and their length.

The numbers are fitted and not the average burst itself, because an average of fifty bursts still
carries their noise, and a template made of it would look for that noise too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wavelet import Wavelet, WaveletEnvelope, bursts

FIT_LEVEL = 2.5
"""A burst, while fitting, is the amplitude above this many times its median."""

MIN_BURSTS = 8
"""Fewer bursts than this describe themselves, not the rhythm."""


class TooFew(ValueError):
    """The stretch holds fewer bursts than a fit can rest on. `bursts` is how many it holds."""

    def __init__(self, bursts: int, where: str):
        super().__init__(f"{bursts} bursts {where} are too few to fit")
        self.bursts = bursts


@dataclass(frozen=True)
class Fit:
    wavelet: Wavelet
    bursts: int


def _found(x: np.ndarray, rate_hz: float, wavelet: Wavelet):
    """The wavelet's output and the bursts in it that outlast the wavelet itself.

    A background with no rhythm also crosses the level, in blobs about as long as the wavelet, because
    that is how long its output stays alike. Those say nothing about the rhythm, so they are left out.
    """
    y = WaveletEnvelope(wavelet, rate_hz).analytic(x)
    a = np.abs(y)
    n = wavelet.taps(rate_hz)
    level = FIT_LEVEL * float(np.median(a[n:]))
    return y, a, [b for b in bursts(a, rate_hz, level, min_seconds=wavelet.seconds) if b[0] >= n]


def _turning(y: np.ndarray, found, rate_hz: float) -> float:
    """The frequency of the bursts: how fast the wavelet's output turns, the large samples counting most."""
    turn = sum(np.sum(y[s + 1:e] * np.conj(y[s:e - 1])) for s, _, e in found)
    return float(np.angle(turn) * rate_hz / (2 * np.pi))


def _half_height(a: np.ndarray, start: int, end: int) -> tuple[int, int]:
    """Where a burst's amplitude is above half its own peak."""
    peak = start + int(np.argmax(a[start:end]))
    left = right = peak
    while left > 0 and a[left - 1] >= a[peak] / 2:
        left -= 1
    while right < len(a) - 1 and a[right + 1] >= a[peak] / 2:
        right += 1
    return left, right


def fit_design(x: np.ndarray, rate_hz: float, low_hz: float, high_hz: float) -> Fit:
    """The wavelet that describes the bursts between `low_hz` and `high_hz` in `x`.

    Two passes. The first finds bursts with a wavelet short enough to pass the whole range and reads
    their frequency. The second finds them again at that frequency and reads the frequency once more
    and their length.

    Where the outline peaks is not fitted: read from the peak of each burst, bursts planted to peak at
    0.25 and at 0.75 of their length came out as 0.37 and 0.45.
    """
    x = np.asarray(x, dtype=np.float64)
    centre = (low_hz + high_hz) / 2
    y, _, found = _found(x, rate_hz, Wavelet(centre, max(2.0, centre / (high_hz - low_hz))))
    if len(found) < MIN_BURSTS:
        raise TooFew(len(found), f"between {low_hz:g} and {high_hz:g} Hz")
    probe = Wavelet(float(np.clip(_turning(y, found, rate_hz), low_hz, high_hz)), 4)
    y1, a1, found = _found(x, rate_hz, probe)
    if len(found) < MIN_BURSTS:
        raise TooFew(len(found), f"at {probe.frequency_hz:.1f} Hz")
    frequency = float(np.clip(_turning(y1, found, rate_hz), low_hz, high_hz))

    halves = [_half_height(a1, s, e) for s, _, e in found]
    # A burst's width in the output is its own and the wavelet's, added as squares.
    own = np.sqrt(np.maximum(np.median([r - l for l, r in halves]) ** 2
                             - (probe.taps(rate_hz) / 2) ** 2, 0.0))
    cycles = float(np.clip(2 * own / rate_hz * frequency, 2.0, 20.0))
    return Fit(Wavelet(round(frequency, 1), round(cycles, 1)), len(found))
