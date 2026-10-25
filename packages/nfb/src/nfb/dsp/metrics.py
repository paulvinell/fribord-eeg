"""Metrics: blocks in, readings out.

A metric is anything that consumes `Block`s and emits `Reading`s. That is the entire contract, so
adding one does not touch acquisition or feedback. Two families are provided because they trade
off differently and neurofeedback uses both:

- *continuous* metrics (filter and envelope) update every sample with latency set by the
  smoothing time constant. Low latency, no frequency resolution beyond the filter's.
- *windowed* metrics (spectral, nonlinear) need a whole window before they can say anything, so
  latency is at least the window length. Better frequency resolution and access to measures that
  are undefined pointwise.

Every metric reports two different timings, because conflating them is misleading:

- `delay_seconds` is a genuine lag — the signal cannot influence the output any sooner. Filter
  group delay for continuous metrics, the window length for windowed ones. Irreducible without
  changing the filter or the window.
- `smoothing_seconds` is the averaging time constant, which is a deliberate choice about how
  twitchy the feedback should be, not a cost imposed by anything.

`latency_seconds` is their sum, a conservative upper bound. Reach for `delay_seconds` when asking
"how fast could this possibly be", and measure the step response when the answer matters — with a
0.5 s constant, band power reaches half of a step at ~290 ms and 90% at ~930 ms, while the
irreducible part is only ~110 ms.

Metrics mark readings from unusable data `valid=False` rather than dropping them, so a consumer can
distinguish "no signal" from "no data".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy import signal

from ..stream import Block, ChannelSpec
from .buffer import AlignedWindower, Window, Windower
from .filters import ELLIPTIC_RIPPLE_DB, Band, OnePole, StreamFilter, bandpass


@dataclass(frozen=True)
class Reading:
    """One measurement: what, where, when, and whether to trust it."""

    name: str
    value: float
    t: float                             # seconds from session start, from sample index
    channel: int | tuple[int, ...]
    unit: str = ""
    valid: bool = True


class Metric(Protocol):
    def feed(self, block: Block) -> list[Reading]: ...
    @property
    def latency_seconds(self) -> float: ...
    @property
    def delay_seconds(self) -> float: ...
    @property
    def smoothing_seconds(self) -> float: ...


class BandPower:
    """Continuous band amplitude: bandpass, then a smoothed envelope.

    The classic neurofeedback measurement and the low-latency option — output updates every
    sample, delayed by the filter's group delay plus the smoothing time constant. Reported in the
    channel's own units (µV for EEG) as amplitude, not power, because that is what thresholds are
    conventionally set in.

    `envelope` picks the amplitude convention, and it matters for comparability:

    - "rms" (default) — smoothed mean square, then square root. The physically standard measure.
    - "meanabs" — smoothed absolute value (`abs(bandpass(x))` fed
      into a one-pole with a 0.5 s time constant). For a sine, mean-abs is 2/pi of the peak
      against RMS's 1/sqrt(2), so meanabs reads about 0.900x the RMS figure. Use it when a
      threshold came from software that computed it this way, or the numbers will not transfer.

    Emits at `output_hz` by decimating; the underlying computation still runs at full rate.
    """

    def __init__(self, spec: ChannelSpec, band: Band, tau_seconds: float = 0.5,
                 output_hz: float = 10.0, envelope: str = "rms", order: int = 2,
                 kind: str = "butterworth", ripple_db: float = ELLIPTIC_RIPPLE_DB):
        if envelope not in ("rms", "meanabs"):
            raise ValueError(f"envelope must be 'rms' or 'meanabs', got {envelope!r}")
        self.spec = spec
        self.band = band
        self.envelope = envelope
        self.order = order
        self.kind = kind
        self.ripple_db = ripple_db
        self.filter = bandpass(band, spec.rate_hz, order=order, kind=kind, ripple_db=ripple_db)
        self.smoother = OnePole(tau_seconds, spec.rate_hz)
        self.stride = max(1, int(round(spec.rate_hz / output_hz)))
        self._next_index: int | None = None
        self._expected: int | None = None

    @property
    def name(self) -> str:
        return f"{self.band.label}_rms"

    @property
    def delay_seconds(self) -> float:
        """Irreducible lag: the bandpass group delay. Lower the order to reduce it."""
        return self.filter.latency_seconds

    @property
    def smoothing_seconds(self) -> float:
        return self.smoother.tau

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds + self.smoothing_seconds

    def feed(self, block: Block) -> list[Reading]:
        if block.channel != self.spec.index:
            return []
        if self._expected is not None and block.start_index != self._expected:
            # a gap: filter and smoother state describes samples that no longer precede these,
            # and the output cursor points into data that never arrived
            self.filter.reset()
            self.smoother.reset()
            self._next_index = block.start_index
        self._expected = block.end_index

        x = np.asarray(block.samples, dtype=np.float64) * self.spec.scale
        filtered = self.filter.process(x)
        if self.envelope == "rms":
            amplitude = np.sqrt(np.maximum(self.smoother.process(filtered ** 2), 0.0))
        else:
            amplitude = self.smoother.process(np.abs(filtered))

        if self._next_index is None:
            self._next_index = block.start_index
        out = []
        while self._next_index < block.end_index:
            i = self._next_index - block.start_index
            out.append(Reading(
                name=self.name,
                value=float(amplitude[i]),
                t=self._next_index / self.spec.rate_hz,
                channel=self.spec.index,
                unit=self.spec.unit,
                valid=block.valid,
            ))
            self._next_index += self.stride
        return out


class WindowMetric:
    """Applies a window function to one channel, on a fixed grid.

    Use for anything undefined pointwise — spectra, entropy, complexity. Latency is the window
    length, since nothing can be reported until the window is full.
    """

    def __init__(self, spec: ChannelSpec, name: str, fn: Callable[[Window], float],
                 window_seconds: float = 2.0, hop_seconds: float | None = None,
                 unit: str = ""):
        size = max(1, int(round(window_seconds * spec.rate_hz)))
        hop = size if hop_seconds is None else max(1, int(round(hop_seconds * spec.rate_hz)))
        self.windower = Windower(spec, size, hop)
        self.name = name
        self.fn = fn
        self.unit = unit
        self.spec = spec

    @property
    def delay_seconds(self) -> float:
        """The whole window: its last sample is needed before anything can be reported."""
        return self.windower.size / self.spec.rate_hz

    @property
    def smoothing_seconds(self) -> float:
        return 0.0

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds

    def feed(self, block: Block) -> list[Reading]:
        if block.channel != self.spec.index:
            return []
        out = []
        for w in self.windower.push(block):
            try:
                value = float(self.fn(w))
            except (ValueError, FloatingPointError):
                continue
            out.append(Reading(self.name, value, w.centre_seconds, self.spec.index,
                               self.unit, w.valid))
        return out


class PairMetric:
    """Applies a function to time-aligned windows from two or more channels.

    The alignment is the point: cross-channel measures on non-simultaneous samples are wrong in
    a way that looks plausible.
    """

    def __init__(self, specs: dict[int, ChannelSpec], name: str,
                 fn: Callable[[dict[int, Window]], float],
                 window_seconds: float = 2.0, hop_seconds: float | None = None,
                 unit: str = ""):
        rate = next(iter(specs.values())).rate_hz
        size = max(1, int(round(window_seconds * rate)))
        hop = size if hop_seconds is None else max(1, int(round(hop_seconds * rate)))
        self.windower = AlignedWindower(specs, size, hop)
        self.name = name
        self.fn = fn
        self.unit = unit
        self.channels = tuple(sorted(specs))
        self.rate_hz = rate
        self.size = size

    @property
    def delay_seconds(self) -> float:
        return self.size / self.rate_hz

    @property
    def smoothing_seconds(self) -> float:
        return 0.0

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds

    def feed(self, block: Block) -> list[Reading]:
        out = []
        for group in self.windower.push(block):
            try:
                value = float(self.fn(group))
            except (ValueError, FloatingPointError):
                continue
            any_window = next(iter(group.values()))
            out.append(Reading(self.name, value, any_window.centre_seconds, self.channels,
                               self.unit, all(w.valid for w in group.values())))
        return out


# --- window functions ---------------------------------------------------------------------
# Plain functions over arrays, so they can be tested and used offline without any pipeline.

def band_power(x: np.ndarray, rate_hz: float, band: Band) -> float:
    """Mean power spectral density within a band, via Welch. Units are (signal unit)²/Hz."""
    nperseg = min(len(x), int(rate_hz))
    freqs, psd = signal.welch(x, fs=rate_hz, nperseg=nperseg)
    sel = (freqs >= band.low) & (freqs <= band.high)
    if not sel.any():
        raise ValueError(f"no spectral bins inside {band.label} at {rate_hz:g} sps")
    return float(psd[sel].mean())


def band_amplitude(x: np.ndarray, rate_hz: float, band: Band) -> float:
    """RMS amplitude within a band, from the spectrum. Comparable to BandPower's output."""
    nperseg = min(len(x), int(rate_hz))
    freqs, psd = signal.welch(x, fs=rate_hz, nperseg=nperseg)
    sel = (freqs >= band.low) & (freqs <= band.high)
    if not sel.any():
        raise ValueError(f"no spectral bins inside {band.label} at {rate_hz:g} sps")
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sqrt(psd[sel].sum() * df))


def band_ratio(x: np.ndarray, rate_hz: float, numerator: Band, denominator: Band) -> float:
    """Ratio of band amplitudes — the shape most classic protocols are expressed in."""
    den = band_amplitude(x, rate_hz, denominator)
    if den == 0:
        raise ValueError("denominator band has zero amplitude")
    return band_amplitude(x, rate_hz, numerator) / den


def coherence(a: np.ndarray, b: np.ndarray, rate_hz: float, band: Band) -> float:
    """Mean magnitude-squared coherence in a band: 0 unrelated, 1 linearly related."""
    nperseg = min(len(a), int(rate_hz))
    freqs, cxy = signal.coherence(a, b, fs=rate_hz, nperseg=nperseg)
    sel = (freqs >= band.low) & (freqs <= band.high)
    if not sel.any():
        raise ValueError(f"no spectral bins inside {band.label} at {rate_hz:g} sps")
    return float(cxy[sel].mean())


def phase_locking_value(a: np.ndarray, b: np.ndarray, rate_hz: float, band: Band) -> float:
    """Phase locking value in a band: 0 uniform phase difference, 1 constant offset.

    Unlike coherence this ignores amplitude entirely, so it does not rise merely because both
    signals are strong at the same time.
    """
    fa = bandpass(band, rate_hz).process(a)
    fb = bandpass(band, rate_hz).process(b)
    pa = np.angle(signal.hilbert(fa))
    pb = np.angle(signal.hilbert(fb))
    return float(np.abs(np.mean(np.exp(1j * (pa - pb)))))


def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Sample entropy (Richman & Moorman): higher means less self-similar, less predictable.

    `r` defaults to 0.2 standard deviations, the usual choice. Cost is O(n²) in memory and time,
    so keep windows to a few hundred samples — 2 s at 250 sps is 500, which is fine.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < m + 2:
        raise ValueError(f"need at least {m + 2} samples")
    if r is None:
        r = 0.2 * float(x.std())
    if r <= 0:
        return 0.0                       # a constant signal is perfectly predictable

    def matches(length: int) -> int:
        emb = np.lib.stride_tricks.sliding_window_view(x, length)
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(axis=2)
        return int(np.triu(d <= r, k=1).sum())

    b = matches(m)
    a = matches(m + 1)
    if b == 0 or a == 0:
        raise ValueError("no template matches at this tolerance")
    return float(-np.log(a / b))


def lempel_ziv(x: np.ndarray) -> float:
    """Normalised Lempel-Ziv complexity of the median-binarised signal.

    Counts distinct substrings needed to reconstruct the sequence (Kaspar & Schuster),
    normalised by n/log2(n) so it is comparable across window lengths: ~1 for a random
    sequence, near 0 for a regular one. Cheap, unlike sample entropy.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        raise ValueError("need at least 4 samples")
    s = (x > np.median(x)).astype(np.int8)
    i, k, l, c, kmax = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            kmax = max(kmax, k)
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return float(c * np.log2(n) / n)
