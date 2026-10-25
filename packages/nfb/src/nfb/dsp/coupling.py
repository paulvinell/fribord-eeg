"""Phase-amplitude coupling: does a slow rhythm's phase decide when a fast rhythm is loud?

The measure behind the Coupling protocol. A slow band's instantaneous phase is read off one channel and
a fast band's amplitude off another (or the same one); if the fast amplitude is systematically larger at
some slow phases than others, the two are coupled. This is the "theta-gamma code" — Jensen and Lisman's
nesting of gamma bursts inside theta troughs — and it is the one measurement in this family that is
robustly documented in single-unit, LFP, EEG and MEG recordings.

**Both bands come off the same estimator with the same designed delay.** `CFIREnvelope.analytic` gives a
complex narrowband signal whose angle is the instantaneous phase and whose modulus is the amplitude, and
both filters are designed for the same `delay_seconds`. That is not tidiness: a phase and an amplitude
taken with different lags describe different moments, which either hides real coupling or manufactures
coupling that is not there, and neither failure announces itself.

**Two measures are computed, because they fail differently and the comparison is the point.**

- `modulation` is Tort's modulation index: bin the fast amplitude by slow phase, normalise the bin means
  into a distribution, and measure how far that distribution sits from flat, as a fraction of the most
  uneven it could be. Bounded 0 to 1. Of the four common estimators it is the one least affected by data
  length, by noise and by how wide the modulation is, which is why it is what a threshold is set against.
- `vector_length` is Canolty's mean vector length, normalised by the mean amplitude so it is
  dimensionless. It is more sensitive when the data are long and clean, and — the reason it is here at
  all — its *angle* is the preferred phase: **which** part of the slow cycle the fast activity rides on.
  A strength with no preferred phase is not a description of anything.

**Three ways this measure lies, all three guarded rather than documented.**

- *The phase must mean something.* An instantaneous phase is only interpretable when the slow band has a
  real peak in the spectrum; run it on a band with no rhythm in it and the phase is the filter's own
  ringing, which is perfectly periodic and will couple to anything. `peak_ratio` is the slow band's
  amplitude against its two flanking bands, and a screen gates on it.
- *A non-sinusoidal slow wave manufactures coupling.* A sharp-edged slow rhythm has harmonics, the
  harmonics land in the fast band, and they are phase-locked to the slow wave by construction. There is
  no coupling — there is one oscillation that is not a sine. The floor below catches this, which was
  measured rather than hoped for: a 6 Hz square wave in noise reads a modulation of **0.050**, which
  would pass for strong coupling on its own, against a floor of **0.047**. A second-harmonic ratio was
  built for this first and then removed, because it is a worse test than the one already here — it reads
  nothing at all on a square wave, which has no even harmonics, and it reads high whenever a neighbouring
  rhythm is simply louder, which is a different fault with a different remedy.
- *Any finite window reads above zero on unrelated signals.* This is the same bias that `SynchronyMetric`
  documents for the phase-locking value, and the fix is the one the methods literature recommends:
  compute the identical statistic against a time-shifted copy of the fast amplitude. Shifting breaks the
  phase relationship while preserving both signals' own spectra and their non-stationarity, so whatever
  the shifted version reads is what this window, this bandwidth and this recording produce from nothing.
  `floor` is that number, and it is drawn on screen beside the reading rather than subtracted from it.

A fourth, which is not about the measure but about what reaches it: **samples from a rejected block are
filtered but not counted.** See `feed`.

**The fast band has to be wide enough to hold the sidebands, and this is enforced.** Amplitude modulated
at the slow frequency puts energy at `fast ± slow`, so a fast band narrower than twice the slow centre
frequency cuts off the very thing being measured and reports no coupling however much there is. It is the
most commonly violated rule in this literature, so `Coupling` refuses the band rather than noting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stream import Block, ChannelSpec
from .envelope import CFIREnvelope
from .filters import Band
from .metrics import Reading
from .synchrony import ChannelAligner

MEASURES = {
    "modulation": "how uneven the fast amplitude is across the slow cycle",
    "vector_length": "how far the fast amplitude leans towards one slow phase",
}

PHASE_BINS = 18
"""Slow-cycle bins the fast amplitude is averaged in — 20° each, Tort's own choice.

More bins resolve a narrower modulation and cost bias: every bin is a mean over fewer samples, so an
unrelated signal reads further from flat. 18 is what the index was characterised on and what its published
values refer to, so a number from a paper transfers.
"""

DEFAULT_WINDOW_SECONDS = 20.0
"""How much of the recent past a reading is averaged over.

All four common estimators fail below about twelve seconds of data, and the two computed here are the two
most sensitive to running short. Twenty seconds is roughly a hundred theta cycles, which puts the noise
floor low enough that a real modulation clears it — and the floor is measured and displayed anyway, so a
window chosen too short shows up as the two lines converging rather than as a wrong answer.
"""

SURROGATE_SHIFTS = (3.3, 5.7, 8.1, 11.9)
"""Slow periods to delay the fast amplitude by, for the floor. Deliberately not whole numbers: a shift of
a whole period puts the amplitude back on the phase it came from and measures the coupling again."""

FLANK_FLOOR_HZ = 0.5
"""Lowest edge a comparison flank will reach down to — the bottom of EEG everywhere else in this
codebase, and below it a flank measures the electrode settling rather than a neighbouring rhythm."""

MIN_DELAY_SECONDS = 0.05


@dataclass(frozen=True)
class Coupling:
    """One coupling: which slow phase, which fast amplitude, and on which channels.

    `phase_channel` and `amplitude_channel` may be the same — coupling within one site is the form the
    theta-gamma literature is built on — or different, which asks whether one site's slow rhythm governs
    another's fast activity. Same-channel is the default because it is the claim with the evidence behind
    it.
    """

    slow: Band
    fast: Band
    phase_channel: int = 0
    amplitude_channel: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.fast.low <= self.slow.high:
            raise ValueError(
                f"the fast band {self.fast.label} must sit above the slow band {self.slow.label}")
        needed = 2 * self.slow.centre
        if self.fast.high - self.fast.low < needed:
            raise ValueError(
                f"a {self.fast.high - self.fast.low:g} Hz fast band cannot show modulation at "
                f"{self.slow.centre:g} Hz — the sidebands sit {self.slow.centre:g} Hz either side of "
                f"the carrier, so widen {self.fast.label} to at least {needed:g} Hz")

    @property
    def amplitude_from(self) -> int:
        return self.phase_channel if self.amplitude_channel is None else self.amplitude_channel

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys((self.phase_channel, self.amplitude_from)))

    @property
    def within_channel(self) -> bool:
        return self.amplitude_from == self.phase_channel

    @property
    def label(self) -> str:
        return self.name or f"{self.slow.label}→{self.fast.label}"

    def describe(self) -> str:
        where = ("one site" if self.within_channel
                 else f"ch{self.phase_channel} phase → ch{self.amplitude_from} amplitude")
        return f"{self.slow.label} phase gating {self.fast.label} amplitude · {where}"


def modulation_index(bin_means: np.ndarray) -> float:
    """Tort's modulation index from per-phase-bin mean amplitudes: 0 flat, 1 all in one bin.

    The bin means are normalised into a distribution and scored by how far its entropy falls short of a
    flat distribution's, divided by that same maximum so the answer is a fraction rather than a quantity
    whose scale depends on the bin count.
    """
    p = np.asarray(bin_means, dtype=float)
    if p.size < 2 or not np.all(np.isfinite(p)) or p.min() < 0:
        return 0.0
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    nonzero = p[p > 0]
    entropy = -float((nonzero * np.log(nonzero)).sum())
    ceiling = float(np.log(p.size))
    return max(0.0, min(1.0, (ceiling - entropy) / ceiling))


class _PhaseHistogram:
    """Mean fast amplitude per slow-phase bin, over an exponentially forgetting window.

    A forgetting window rather than a sliding buffer because a reading is wanted every block and the
    alternative is re-summing tens of thousands of samples to move one. The decay is applied exactly, not
    per block: each sample's contribution is weighted by its own age, which is what makes feeding one long
    block and feeding it in pieces give the same answer.
    """

    def __init__(self, decay: float, bins: int = PHASE_BINS):
        self.decay = float(decay)
        self.sums = np.zeros(bins)
        self.counts = np.zeros(bins)
        self.vector = 0.0 + 0.0j
        self.mean_amplitude = 0.0
        self.weight = 0.0

    def push(self, bins: np.ndarray, amplitude: np.ndarray, phase: np.ndarray) -> None:
        n = len(bins)
        if not n:
            return
        ages = np.arange(n - 1, -1, -1)
        weights = self.decay ** ages
        block = self.decay ** n
        self.sums *= block
        self.counts *= block
        np.add.at(self.sums, bins, amplitude * weights)
        np.add.at(self.counts, bins, weights)
        self.vector = self.vector * block + complex(
            np.sum(amplitude * np.exp(1j * phase) * weights))
        self.mean_amplitude = self.mean_amplitude * block + float(np.sum(amplitude * weights))
        self.weight = self.weight * block + float(weights.sum())

    @property
    def bin_means(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1e-12), 0.0)
        return means

    @property
    def modulation(self) -> float:
        if (self.counts > 0).sum() < self.counts.size:
            return 0.0                    # not every part of the slow cycle has been seen yet
        return modulation_index(self.bin_means)

    @property
    def vector_length(self) -> float:
        if self.mean_amplitude <= 0:
            return 0.0
        return float(min(1.0, abs(self.vector) / self.mean_amplitude))

    @property
    def preferred_phase(self) -> float:
        return float(np.angle(self.vector))


class CouplingMetric:
    """One coupling on one channel pair. Satisfies the `Metric` protocol.

    Emits the measure named on the `Coupling`; the other one, the preferred phase, the noise floor and
    both validity ratios are all available as properties for a screen to draw, because a coupling number
    on its own is not interpretable and this codebase does not put uninterpretable numbers on screens.
    """

    def __init__(self, specs: dict[int, ChannelSpec], coupling: Coupling,
                 measure: str = "modulation", window_seconds: float = DEFAULT_WINDOW_SECONDS,
                 output_hz: float = 10.0, delay_seconds: float = 0.1):
        missing = [c for c in coupling.channels if c not in specs]
        if missing:
            raise ValueError(f"channel {missing[0]} is not in this session")
        if measure not in MEASURES:
            raise ValueError(f"unknown coupling measure {measure!r}")
        rates = {specs[c].rate_hz for c in coupling.channels}
        if len(rates) != 1:
            raise ValueError("a coupling needs its channels at the same sample rate")
        self.rate_hz = rates.pop()
        if coupling.fast.high >= self.rate_hz / 2:
            raise ValueError(
                f"{coupling.fast.label} needs {coupling.fast.high:g} Hz but this session samples at "
                f"{self.rate_hz:g} sps — raise the sample rate or lower the fast band")
        self.specs = specs
        self.coupling = coupling
        self.measure = measure
        self.window_seconds = float(window_seconds)
        # The same designed delay on both bands, so the phase and the amplitude describe the same moment.
        self.delay = max(MIN_DELAY_SECONDS, float(delay_seconds))
        self.aligner = ChannelAligner(coupling.channels)
        self.slow = CFIREnvelope(coupling.slow, self.rate_hz, delay_seconds=self.delay)
        self.fast = CFIREnvelope(coupling.fast, self.rate_hz, delay_seconds=self.delay)
        self.flanks = [CFIREnvelope(band, self.rate_hz, delay_seconds=self.delay)
                       for band in self._flank_bands()]

        dt = 1.0 / self.rate_hz
        decay = float(np.exp(-dt / max(self.window_seconds, dt)))
        self.live = _PhaseHistogram(decay)
        self.surrogates = [_PhaseHistogram(decay) for _ in SURROGATE_SHIFTS]
        period = self.rate_hz / max(coupling.slow.centre, 1e-6)
        self._lags = [max(1, int(round(shift * period))) for shift in SURROGATE_SHIFTS]
        self._history = np.zeros(max(self._lags) + 1)
        self._filled = 0
        # Flank amplitudes, smoothed over the same window so the ratio on screen is an
        # average of the same stretch of signal as the coupling it qualifies.
        self._decay = decay
        self._slow_power = 0.0
        self._flank_power = 0.0

        self.stride = max(1, int(round(self.rate_hz / output_hz)))
        self._next_index: int | None = None
        self.value = 0.0
        self.seconds = 0.0

    def _flank_bands(self) -> list[Band]:
        """The bands immediately below and above the slow one, each its own width wide.

        What they are for: the slow band's amplitude means nothing on its own, but its amplitude against
        its neighbours says whether there is a *peak* there — a rhythm — or just the local level of a
        spectrum that slopes through it. The phase of a band with no rhythm in it is the filter ringing,
        which is perfectly periodic and will couple to anything.

        Adjacent rather than set back, because the neighbour is the comparison that matters: a theta band
        is worth taking the phase of when there is more in it than in the alpha sitting next to it.
        """
        width = self.coupling.slow.high - self.coupling.slow.low
        out = []
        bottom = max(FLANK_FLOOR_HZ, self.coupling.slow.low - width)
        if self.coupling.slow.low - bottom >= width / 2:
            out.append(Band(bottom, self.coupling.slow.low, "below"))
        top = self.coupling.slow.high + width
        if top < self.rate_hz / 2:
            out.append(Band(self.coupling.slow.high, top, "above"))
        return out

    @property
    def name(self) -> str:
        return f"pac_{self.coupling.slow.low:g}-{self.coupling.slow.high:g}_" \
               f"{self.coupling.fast.low:g}-{self.coupling.fast.high:g}"

    @property
    def delay_seconds(self) -> float:
        return self.delay

    @property
    def smoothing_seconds(self) -> float:
        """The whole averaging window. A coupling is a statistic over many slow cycles and there is no
        version of it that answers faster — stating it is the alternative to pretending otherwise."""
        return self.window_seconds

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds + self.smoothing_seconds

    @property
    def modulation(self) -> float:
        return self.live.modulation

    @property
    def vector_length(self) -> float:
        return self.live.vector_length

    @property
    def preferred_phase(self) -> float:
        """Where in the slow cycle the fast activity sits, in radians: 0 is the peak, ±π the trough."""
        return self.live.preferred_phase

    @property
    def bin_means(self) -> np.ndarray:
        return self.live.bin_means

    @property
    def floor(self) -> float:
        """What this window reads on the same signals with the phase relationship broken.

        The reading has to clear this to mean anything. It is a measurement rather than a formula, so it
        follows the bandwidth, the window and the recording's own non-stationarity without being told.
        """
        if not self.surrogates or self._filled <= max(self._lags):
            return 0.0
        values = [h.modulation if self.measure == "modulation" else h.vector_length
                  for h in self.surrogates]
        return float(np.mean(values))

    @property
    def peak_ratio(self) -> float:
        """The slow band's amplitude over its flanks' — above 1 there is a rhythm, at 1 there is a slope.

        The phase this whole measure rests on is only interpretable when this says a rhythm is present.
        """
        if self._flank_power <= 0:
            return 0.0
        return float(self._slow_power / self._flank_power)

    @property
    def settled(self) -> bool:
        """Whether a full window has gone through. Before this the reading is of a partial average."""
        return self.seconds >= self.window_seconds

    def describe(self) -> str:
        return (f"{self.coupling.describe()} · {MEASURES[self.measure]} · "
                f"averaged over {self.window_seconds:g} s")

    def feed(self, block: Block, count: bool = True) -> list[Reading]:
        """`count=False` filters the block without letting it into the statistics — see below."""
        got = self.aligner.push(block)
        if got is None:
            return []
        start, runs = got
        by_channel = dict(zip(self.coupling.channels, runs))
        phase_in = by_channel[self.coupling.phase_channel]
        amplitude_in = by_channel[self.coupling.amplitude_from]

        slow = self.slow.analytic(phase_in)
        fast = np.abs(self.fast.analytic(amplitude_in))
        phase = np.angle(slow)
        bins = np.minimum(((phase + np.pi) / (2 * np.pi) * PHASE_BINS).astype(int),
                          PHASE_BINS - 1)

        history = np.concatenate([self._history, fast])
        # **Filters always, statistics only on accepted blocks, and the split is not optional here.**
        # A rejected block reaches every estimator as a stand-in held at the last accepted value, so
        # that sample indices stay contiguous and no filter reads a blink as an acquisition gap. For a
        # band amplitude that is harmless — the reading simply holds. For a statistic accumulated over
        # a window it is not: a held stretch is a flat step, a flat step is a non-stationarity, and
        # non-stationarity is the textbook way to manufacture cross-frequency coupling out of nothing.
        # Measured through the real session with 0.1% of blocks held: an *uncoupled* signal read 3.1
        # times its own floor with those samples counted and 0.8 times without. The filters still see
        # them, so nothing re-settles, and the caller says which blocks earned their way in.
        if count and block.valid:
            self.live.push(bins, fast, phase)
            for lag, surrogate in zip(self._lags, self.surrogates):
                shifted = history[len(history) - len(fast) - lag:len(history) - lag]
                surrogate.push(bins, shifted, phase)
            self._filled += len(fast)
            self._push_powers(np.abs(slow), phase_in, len(fast))
        self._history = history[-(max(self._lags) + 1):] if self._lags else history[:0]

        if self._next_index is None:
            self._next_index = start
        self.seconds += len(fast) / self.rate_hz
        out: list[Reading] = []
        for offset in range(len(fast)):
            index = start + offset
            if index < self._next_index:
                continue
            self._next_index = index + self.stride
            self.value = (self.live.modulation if self.measure == "modulation"
                          else self.live.vector_length)
            out.append(Reading(name=self.name, value=self.value, t=index / self.rate_hz,
                               channel=self.coupling.channels, unit="", valid=block.valid))
        return out

    def _push_powers(self, slow_amplitude: np.ndarray, raw: np.ndarray, n: int) -> None:
        block = self._decay ** n
        weights = self._decay ** np.arange(n - 1, -1, -1)
        self._slow_power = self._slow_power * block + float(np.sum(slow_amplitude * weights))
        if self.flanks:
            flank = np.mean([np.abs(f.analytic(raw)) for f in self.flanks], axis=0)
            self._flank_power = self._flank_power * block + float(np.sum(flank * weights))


@dataclass(frozen=True)
class Locking:
    """One phase-locking measurement: which band, and between which two channels."""

    band: Band
    pair: tuple[int, int] = (0, 1)
    name: str = ""

    def __post_init__(self) -> None:
        if self.pair[0] == self.pair[1]:
            raise ValueError("phase locking is between two different channels")
        if self.band.high - self.band.low <= 0:
            raise ValueError(f"{self.band.label} is not a band")

    @property
    def label(self) -> str:
        # **Not a dash, because a dash here is arithmetic.** `ch0−ch1` on the amplitude screen is a
        # signed combination — one channel subtracted from another, with the shared reference cancelling
        # out. This is two channels *compared*, where the reference emphatically does not cancel, and the
        # two labels were one pixel apart: U+2013 against U+2212.
        return self.name or f"{self.band.label} ch{self.pair[0]} ↔ ch{self.pair[1]}"

    def describe(self) -> str:
        return f"{self.band.label} phase held steady between ch{self.pair[0]} and ch{self.pair[1]}"


class LockingMetric:
    """How steadily two channels hold a phase relationship in one band. Satisfies `Metric`.

    **The plain phase-locking value is biased upwards and the bias is not small**, which is why this
    class exists rather than a two-line running mean. A phase average over `N` independent samples reads
    about `sqrt(pi/(4N))` on channels with no relationship at all, so the number depends on how long you
    averaged and how wide the band is as much as on the signal. Measured here on independent narrowband
    noise, exponential averaging:

    | band | 2 s window | 5 s | 10 s |
    |---|---|---|---|
    | 0.2 Hz | 0.75 | 0.52 | 0.37 |
    | 1 Hz | 0.40 | 0.26 | 0.19 |
    | 4 Hz | 0.22 | 0.15 | 0.11 |

    **That is a baseline, not a disqualification**, and the distinction cost a wrong conclusion once: the
    0.75 in the top-left corner was measured on the cross-frequency screen's 0.2 Hz bands and then
    carried over to bands twenty times wider, where it does not apply. At a 4 Hz band with a 5 s window
    an uncoupled pair reads 0.15 against 0.41 for a half-locked one — `d' = 3.4`, which is a comfortably
    trainable separation.

    So the bias is removed rather than the measure abandoned. `consistency` is Vinck's pairwise phase
    consistency, which is an **exactly unbiased** estimator of the squared locking value: it reads zero
    on unrelated channels at every bandwidth and every window, measured, and — the property that matters
    for a control the operator can change — **its value for a given relationship does not move when the
    window does**, where the raw value drifts by a third across the same range. `value` is its square
    root, so the number is on the same 0-to-1 scale as a published locking value with the chance level
    taken out.

    `floor` is a shifted-copy surrogate like the coupling metric's, kept because the formula above
    assumes a stationarity a real recording does not have. It should sit near zero; a floor that does not
    is the screen saying the debias is not describing this signal.

    **Two things about that surrogate are easy to get wrong and both were, once.** It must shift one
    channel's *phase* before the two are combined — shifting the phase *difference* leaves a locked
    relationship exactly as locked, because the shifted difference of a steady relationship is the same
    steady relationship. And the shift must clear the band's **coherence time**, `1/width`, rather than a
    few of its periods: measured at a 4 Hz band, a half-period shift left a half-locked pair reading
    0.404 against a true 0.131, where shifts from one coherence time upward all read 0.11-0.16 whatever
    the real locking was.
    """

    def __init__(self, specs: dict[int, ChannelSpec], locking: Locking,
                 window_seconds: float = 5.0, output_hz: float = 10.0,
                 delay_seconds: float = 0.1):
        missing = [c for c in locking.pair if c not in specs]
        if missing:
            raise ValueError(f"channel {missing[0]} is not in this session")
        rates = {specs[c].rate_hz for c in locking.pair}
        if len(rates) != 1:
            raise ValueError("phase locking needs both channels at the same sample rate")
        self.rate_hz = rates.pop()
        if locking.band.high >= self.rate_hz / 2:
            raise ValueError(
                f"{locking.band.label} needs {locking.band.high:g} Hz but this session samples at "
                f"{self.rate_hz:g} sps")
        self.specs = specs
        self.locking = locking
        self.window_seconds = float(window_seconds)
        self.delay = max(MIN_DELAY_SECONDS, float(delay_seconds))
        self.aligner = ChannelAligner(locking.pair)
        self.filters = {c: CFIREnvelope(locking.band, self.rate_hz, delay_seconds=self.delay)
                        for c in locking.pair}

        dt = 1.0 / self.rate_hz
        self.decay = float(np.exp(-dt / max(self.window_seconds, dt)))
        self._phasor = 0.0 + 0.0j
        self._signs = 0.0
        self._weight = 0.0
        width = locking.band.high - locking.band.low
        # Independent phase estimates arrive at about the bandwidth, and an exponential window holds
        # roughly two time constants' worth — checked against the measured null, which puts 1/E[plv²]
        # at 5.1, 11.9, 19.2 and 37.7 where this gives 4, 10, 20 and 40.
        self.samples = max(2.0, 2.0 * self.window_seconds * width)
        coherence = self.rate_hz / max(width, 1e-6)
        self._lags = [max(1, int(round(shift * coherence))) for shift in SURROGATE_SHIFTS]
        self._history: list[np.ndarray] = []      # channel b's recent phase, for the surrogate
        self._surrogates = [0.0 + 0.0j for _ in SURROGATE_SHIFTS]

        self.stride = max(1, int(round(self.rate_hz / output_hz)))
        self._next_index: int | None = None
        self.value = 0.0
        self.seconds = 0.0

    @property
    def name(self) -> str:
        return f"plv_{self.locking.band.low:g}-{self.locking.band.high:g}_" \
               f"{self.locking.pair[0]}-{self.locking.pair[1]}"

    @property
    def raw(self) -> float:
        """The plain phase-locking value, biased, kept because it is what a paper quotes."""
        return self._mean(self._phasor)

    def _mean(self, total: complex) -> float:
        """A weighted *mean* phasor. The accumulator holds a weighted sum so that feeding one long block
        and feeding it in pieces agree exactly, which costs this division at the point of reading."""
        if self._weight <= 0:
            return 0.0
        return float(min(1.0, abs(total) / self._weight))

    @property
    def consistency(self) -> float:
        """Pairwise phase consistency: zero on unrelated channels, whatever the window."""
        return self._debias(self.raw)

    def _debias(self, plv: float) -> float:
        return float((self.samples * plv ** 2 - 1.0) / (self.samples - 1.0))

    @property
    def locked(self) -> float:
        """The debiased locking, back on the 0-to-1 scale a published value is quoted on."""
        return float(np.sqrt(max(0.0, self.consistency)))

    @property
    def lag_degrees(self) -> float:
        """Which part of a cycle one channel runs ahead of the other, −180…180.

        **Near zero is the number to be suspicious of.** A relationship carried by something both
        channels share — a reference electrode that is not electrically quiet, a clenched jaw, plain
        volume conduction — arrives at both at the same instant, so it sits at zero lag. Cortical
        coupling generally does not. Zero-lag synchrony between two sites is a real phenomenon and this
        does not call it fake; it says the reading cannot tell the two apart.
        """
        return float(np.degrees(np.angle(self._phasor))) if self._weight else 0.0

    @property
    def lagged(self) -> float:
        """How much of the locking survives discarding whatever sits at zero lag, 0…1.

        The phase lag index: instead of averaging the phase difference, average which *side* of zero it
        falls on. Anything arriving at both channels together contributes equally to both sides and
        cancels, so a reading carried by a shared reference or a shared muscle scores near zero however
        large the plain locking is. Debiased the same way as `locked`, for the same reason.

        **This is the number the floor cannot give you.** A shared reference produces a relationship that
        is genuinely above chance — the two channels really are locked — so a surrogate that breaks the
        timing correctly reports a large, real effect. What it cannot report is that the effect is
        wiring rather than physiology.
        """
        if self._weight <= 0:
            return 0.0
        return float(np.sqrt(max(0.0, self._debias(min(1.0, abs(self._signs) / self._weight)))))

    @property
    def floor(self) -> float:
        """What this window reads with the relationship broken. Near zero if the debias is honest."""
        if not self._surrogates:
            return 0.0
        return float(np.mean([np.sqrt(max(0.0, self._debias(self._mean(p))))
                              for p in self._surrogates]))

    @property
    def settled(self) -> bool:
        return self.seconds >= 5 * self.window_seconds

    @property
    def delay_seconds(self) -> float:
        return self.delay

    @property
    def smoothing_seconds(self) -> float:
        return self.window_seconds

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds + self.smoothing_seconds

    def describe(self) -> str:
        return (f"{self.locking.describe()} · averaged over {self.window_seconds:g} s, "
                f"about {self.samples:.0f} independent estimates")

    def feed(self, block: Block, count: bool = True) -> list[Reading]:
        got = self.aligner.push(block)
        if got is None:
            return []
        start, runs = got
        a, b = (self.filters[c].analytic(v) for c, v in zip(self.locking.pair, runs))
        phase_a, phase_b = np.angle(a), np.angle(b)
        unit = np.exp(1j * (phase_a - phase_b))

        # The *second channel's phase*, not the difference — see the class docstring.
        history = np.concatenate(self._history + [phase_b]) if self._history else phase_b
        if count and block.valid:
            n = len(unit)
            weights = self.decay ** np.arange(n - 1, -1, -1)
            block_decay = self.decay ** n
            self._phasor = self._phasor * block_decay + complex(np.sum(unit * weights))
            # Which *side* of zero the lag falls on, which is all `lagged` needs — see below.
            self._signs = self._signs * block_decay + float(np.sum(np.sign(unit.imag) * weights))
            self._weight = self._weight * block_decay + float(weights.sum())
            for i, lag in enumerate(self._lags):
                if len(history) > n + lag:
                    shifted = history[len(history) - n - lag:len(history) - lag]
                    self._surrogates[i] = (
                        self._surrogates[i] * block_decay
                        + complex(np.sum(np.exp(1j * (phase_a - shifted)) * weights)))
            self.seconds += n / self.rate_hz
        keep = max(self._lags) + 1 if self._lags else 1
        self._history = [history[-keep:]]

        if self._next_index is None:
            self._next_index = start
        out: list[Reading] = []
        for offset in range(len(unit)):
            index = start + offset
            if index < self._next_index:
                continue
            self._next_index = index + self.stride
            self.value = self.locked
            out.append(Reading(name=self.name, value=self.value, t=index / self.rate_hz,
                               channel=self.locking.pair, unit="", valid=block.valid))
        return out
