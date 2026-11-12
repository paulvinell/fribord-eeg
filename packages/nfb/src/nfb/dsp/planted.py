"""Bursts planted in a background at known times, and the score of a detector against them.

A detector of bursts cannot be rated on a recording alone, because nothing says where the bursts are.
Here the answer is known: the bursts are put in, so found, false and late are counts and not opinions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wavelet import outline

BACKGROUND_SLOPE = 1.4
"""How fast the background's power falls with frequency, as the exponent of 1/f. Measured 1.4 from 2 to
40 Hz on a parietal recording with the eyes open, at 10 µV rms in all."""


def pink(n: int, rate_hz: float, rng: np.random.Generator, rms: float = 10.0,
         slope: float = BACKGROUND_SLOPE) -> np.ndarray:
    """A background with no rhythm in it: power falling as 1/f^slope above 1 Hz."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / rate_hz)
    spectrum[1:] *= np.maximum(f[1:], 1.0) ** (-slope / 2)
    spectrum[0] = 0
    x = np.fft.irfft(spectrum, n)
    return x * (rms / x.std())


@dataclass(frozen=True)
class Planted:
    """The bursts to plant. Each one draws its own frequency, length and size around these."""

    frequency_hz: float
    seconds: tuple[float, float]
    amplitude: float
    per_minute: float = 6.0
    jitter_hz: float = 0.0
    rise: float = 0.5


def plant(x: np.ndarray, rate_hz: float, rng: np.random.Generator,
          spec: Planted) -> list[tuple[int, int]]:
    """Add bursts to `x` in place. Returns where each one starts and ends, in samples."""
    truth = []
    gap = 60.0 / spec.per_minute
    edge = int(2 * rate_hz)
    t = edge
    while True:
        t += int(rng.uniform(0.5 * gap, 1.5 * gap) * rate_hz)
        n = int(rng.uniform(*spec.seconds) * rate_hz)
        if t + n >= len(x) - edge:
            return truth
        size = spec.amplitude * rng.uniform(0.7, 1.3)
        f = spec.frequency_hz + rng.uniform(-spec.jitter_hz, spec.jitter_hz)
        cycle = 2 * np.pi * f * np.arange(n) / rate_hz + rng.uniform(0, 2 * np.pi)
        x[t:t + n] += size * outline(n, spec.rise) * np.cos(cycle)
        truth.append((t, t + n))
        t += n


@dataclass(frozen=True)
class Score:
    found: float                    # the fraction of the planted bursts detected
    false_per_minute: float         # detections where nothing was planted
    delay_seconds: float            # median, from the start of a burst to its detection


def score_found(found: list[tuple[int, int, int]], truth: list[tuple[int, int]], samples: int,
                rate_hz: float, tail_seconds: float = 0.3) -> Score:
    """Rate detections, as `bursts` or a `BurstDetector` lists them, against the planted bursts.

    A detection counts for a burst when it is confirmed between the burst's start and `tail_seconds`
    after its end: a causal detector answers late, and a late answer is still an answer.
    """
    confirmed = np.array([b[1] for b in found], dtype=int)
    tail = int(tail_seconds * rate_hz)
    delays = []
    false = len(confirmed)
    for start, end in truth:
        inside = confirmed[(confirmed >= start) & (confirmed < end + tail)]
        false -= len(inside)
        if len(inside):
            delays.append((inside[0] - start) / rate_hz)
    minutes = samples / rate_hz / 60.0
    return Score(len(delays) / len(truth) if truth else 0.0, false / minutes if minutes else 0.0,
                 float(np.median(delays)) if delays else float("nan"))
