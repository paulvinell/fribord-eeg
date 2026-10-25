"""What to feed back on the heart channel — a catalogue, not one hardcoded quantity.

A screen picks a metric, normalises it, and routes the result into feedback. This file is the first of
those two layers and `nfb_console.normalise` is the second; what
carries the result is the same four mechanisms every protocol in this repo shares, so there is no third.

The reason it is a catalogue is that these eleven numbers are not rankable. `sharpness` is the closest thing
to the target of a single spectral peak at precisely 0.1 Hz, and it is also the harshest, because
it punishes a peak outside its narrow band as hard as no peak at all. `rmssd` and `sdnn` say nothing about *where* the
variability is and everything about how much there is, which is what you want when somebody's spectrum has
not organised yet. `lf_hf_ratio` is the one that speaks to a split spectrum directly. Which of them
should drive the room is a clinical choice, and hardcoding it — which is what this screen did — makes that
choice silently on the operator's behalf.

**Three of these are scored against a target frequency, and that target is the operator's.** `sharpness`,
`peak_proximity` and `spectrum_distance` read it off the spectrum they were handed rather than from a
module constant — see `beats.TARGET_HZ` for why 0.1 Hz is a starting point rather than a
fact. `coherence`, `resonance_ratio` and `lf_hf_ratio` do not: the first locates its own peak and the other
two integrate the standard published bands, which must not move or their numbers stop being comparable with
anyone else's.

**A metric is a plain function of `BeatState`.** No state of its own: everything rolling lives either in the
processor that produced the state or in the strategy that normalises the result. That split is what lets any
metric pair with any strategy.

`invert` is the one piece of metadata that is not documentation: it names **which way this metric is usually
trained**. Some are better when *low* — the distance from an ideal spectrum, the LF/HF ratio when the concern
is sympathetic drive — and feedback that got stronger as things got worse would be actively harmful, so the
sensible aim is recorded here rather than left to whoever wires it up.

It is a **default, not a fact**. The operator picks the direction on the screen, the way a band on the
amplitude screen can be set to increase or decrease: LF/HF is inverted here because sympathetic drive is the
usual concern, and somebody training a flat, over-parasympathetic profile wants the other direction. What must
never happen is a direction nobody chose, which is why every metric ships with one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .beats import HF_BAND, TARGET_HZ  # TARGET_HZ only as a fallback; see `_target_of`

LF_BAND = (0.04, 0.15)
"""Low frequency, the conventional HRV band, and **not** the same as `RESONANCE_BAND`.

0.04–0.15 Hz is the standard definition and it is what an LF/HF ratio means; `RESONANCE_BAND` runs to 0.26 Hz
because that is the coherence window. Using one for the other would give a number nobody could
compare with a published figure."""

RSA_WINDOW_S = 20.0
"""How much of the tachogram the respiratory swing is measured over. Two full breaths at six a minute, so a
peak and a trough are both inside it whatever the phase; shorter and the swing depends on when you looked."""


@dataclass(frozen=True)
class MetricSpec:
    """One trainable quantity, with the metadata a screen needs to offer it honestly."""

    id: str
    label: str
    unit: str
    description: str
    extract: Callable[[object], float]
    invert: bool = False
    """True when *low* is the good end — the direction this metric is **usually** trained in, and the default
    the screen starts from. See the module docstring: it is a default rather than a fact, because a metric like
    LF/HF is trained both ways depending on what is wrong."""

    strategies: tuple[str, ...] = ()
    """Which normalisations suit it, best first. Advice rather than a restriction: the pane offers all of
    them and marks these, because "compatible" is a judgement about distributions and somebody who knows
    their own signal may disagree."""
    detail: str = ""
    ready: Callable[[object], bool] = field(default=lambda s: True)
    """Whether this metric **can be computed at all** — not whether it has settled.

    The distinction is the whole of this repo's position on opening a session: a metric with no input is
    `None`, because a zero would be a reading and would anchor a rolling scale; a metric with two beats
    of input is a number, coarse, and it is shown and fed back from the first moment it exists. Nothing
    here waits for a value to stop moving."""

    @property
    def default_direction(self) -> str:
        """Which way to train it unless told otherwise, in the same two words the EEG screens' targets use."""
        return "decrease" if self.invert else "increase"


def _target_of(spectrum) -> float:
    """The frequency this spectrum was scored against.

    Read off the spectrum rather than from the module constant, because the target is the operator's to
    set — see `beats.TARGET_HZ`. The fallback covers a caller holding a hand-built `Spectrum` from
    before the field existed.
    """
    return float(getattr(spectrum, "target_hz", TARGET_HZ))


def _spectral(state) -> bool:
    """Ready as soon as there is a spectrum. **Nothing here waits for one to settle.**

    This gate used to demand four cycles of the target — forty seconds at 0.1 Hz, sixty-two at 0.065 —
    on the argument that a curve can be labelled coarse where a number cannot. That is the "cannot" and
    "will not be accurate" confusion for the third time in this file's history, and it is the same
    answer as the other two: the transform is defined on the span it has, the value it produces is the
    real value for that span, and a number moving from rough to settled in front of somebody is more
    informative than a dash that turns into a number with no visible relationship to what they were
    doing. Watching it stabilise **is** the reading.

    So the only floor is the spectrum existing at all, which `_spectrum` already sets at three beats and
    four grid samples — arithmetic, not judgement. What a short window costs is on the screen where it
    belongs: the span and the bin width under the curve, every second of the session.
    """
    spectrum = getattr(state, "spectrum", None)
    return bool(spectrum is not None and spectrum.ready)


def _beats_ready(n: int) -> Callable[[object], bool]:
    """A ready predicate for the time-domain metrics: at least `n` beats behind the number.

    **`n` is the arithmetic floor and nothing above it.** These metrics used to default to always-ready
    and return a literal 0.0 until they had beats, which is not harmless — a zero is a reading, it lands
    in the rolling normaliser and defines the bottom of its scale for the whole window, and measured, the
    trainee saw a near-constant 74% carrying a third of the real dynamic range for two minutes. So the
    distinction "no reading yet" versus "a reading of zero" stays, and `metric_value` says the first with
    `None`.

    What went with it, and should not have, was padding: ten beats for RMSSD where two intervals make a
    difference, eight for the swing where four points make a peak and a trough. That is waiting for a
    number to settle, which is not this predicate's business — the caller can watch it settle.
    """
    return lambda s: bool(getattr(s, "beats", 0) >= n)


def _peak_proximity(state) -> float:
    """How close the strongest in-band peak is to the target, as 1.0 at it falling to 0 at the edges.

    Scaled by **each side's own half-width**, because the searchable band is asymmetric around the
    target: at 0.1 Hz there is 0.06 Hz of room below and 0.30 above, and moving the target makes it more
    asymmetric still. One shared radius — which is what this used — made the number fall to zero only at
    the top: a spectrum collapsed onto a 0.04 Hz very-low-frequency peak, the sympathetic half of the
    very failure this protocol watches for, scored **0.80** and read as most of the way to breathing
    right. Per-side scaling puts both band edges at 0, which is what the sentence above promises, and it
    keeps doing so wherever the target sits.

    The search range stays the standard `LF_BAND`–`HF_BAND` span whatever the target is, so this metric
    always answers the same question — "of everything that could be a respiratory peak, how close is the
    strongest to where I am aiming" — rather than quietly narrowing its own question along with the aim.
    """
    spectrum = state.spectrum
    peak = spectrum.band_peak((LF_BAND[0], HF_BAND[1]))
    if peak is None:
        return 0.0
    target = _target_of(spectrum)
    reach = (target - LF_BAND[0]) if peak.hz < target else (HF_BAND[1] - target)
    return float(max(0.0, 1.0 - abs(peak.hz - target) / max(reach, 1e-9)))


def _lf_hf(state) -> float:
    spectrum = state.spectrum
    hf = spectrum.band_power(HF_BAND)
    return float(spectrum.band_power(LF_BAND) / hf) if hf > 0 else 0.0


RSA_MIN_BEATS = 4
"""Beats needed before a heart-rate swing exists at all. **Four, which is the arithmetic floor** — a
maximum and a minimum with something between them, which is what `_rsa_amplitude` reads and what it
returns 0.0 without.

It was eight, then five, and both were caution stacked on top of arithmetic: a judgement that four beats
of swing is not *worth* much, spent as seconds of an empty screen at the opening of every session. It is
not this constant's job to decide that. Four beats at a resting rate is four seconds."""

TIME_DOMAIN_MIN_BEATS = 3
"""And the floor for RMSSD, SDNN and pNN50: three beats make two intervals, which make one successive
difference. It was ten, for the same reason and at the same price."""


def _rsa_amplitude(state) -> float:
    """Peak-to-trough heart-rate swing over `RSA_WINDOW_S`, in bpm.

    The plainest reading of respiratory sinus arrhythmia there is, and the one a trainee can feel: at
    resonance the rate visibly climbs on the inhale and falls on the exhale, and the size of that climb is
    what breathing at the right rate produces.
    """
    ages = np.asarray(state.age_s, dtype=float)
    tach = np.asarray(state.tachogram, dtype=float)
    if ages.size < 4 or ages.size != tach.size:
        return 0.0
    window = tach[ages <= RSA_WINDOW_S]
    return float(window.max() - window.min()) if window.size >= 4 else 0.0


HRV_METRICS: dict[str, MetricSpec] = {
    "coherence": MetricSpec(
        id="coherence", label="coherence", unit="",
        description="how isolated the strongest peak is",
        detail=("The peak is located anywhere in "
                "0.03–0.27 Hz and the power on each side of it is punished, multiplicatively. That "
                "rewards *entrainment* — one dominant narrow peak — rather than in-band energy, which is "
                "the distinction the coherence ratio below cannot make. Reported as ln(1+ratio), "
                "because the raw ratio "
                "runs into the hundreds of thousands.\n\n"
                "The peak being *located* rather than assumed is why this does not penalise somebody "
                "whose resonant frequency sits at 0.08 or 0.12 Hz."),
        extract=lambda s: s.spectrum.coherence_score, ready=_spectral,
        strategies=("rolling_minmax", "zscore", "fixed_range"),
    ),
    "resonance_ratio": MetricSpec(
        id="resonance_ratio", label="coherence ratio", unit="",
        description="fraction of HRV power inside 0.04–0.26 Hz",
        detail=("Already 0…1, so it needs no calibration — pair it with a fixed range over [0, 1] and it "
                "works from the first window.\n\n"
                "What it cannot do is tell a sharp peak from a flat band: a spectrum with all its power "
                "spread evenly across 0.04–0.26 Hz scores a perfect 1.0. Use it when you want a stable, "
                "uncalibrated number, and `coherence` when you want entrainment."),
        extract=lambda s: s.spectrum.resonance_ratio, ready=_spectral,
        strategies=("fixed_range", "zscore", "rolling_minmax"),
    ),
    "sharpness": MetricSpec(
        id="sharpness", label="target sharpness", unit="",
        description="fraction of HRV power in the narrow band around the target frequency",
        detail=("The closest match to the stated target — 'a single spectral peak at precisely 0.1 Hz' — "
                "because it punishes power leaking out of band, a broad or multi-peaked spectrum, and a "
                "peak at the wrong frequency, all in one number.\n\n"
                "It is also the harshest for that reason: with the target at 0.1 and the band at ±0.02, "
                "somebody whose own resonance is at 0.07 is penalised as hard as somebody with no peak "
                "at all. That is what the target setting on the Train tab is for — move it to the rate "
                "you actually resonate at and this becomes the sharpest measure here rather than the "
                "least fair one. If you do not know the rate yet, `coherence` finds the peak instead of "
                "assuming it."),
        extract=lambda s: s.spectrum.sharpness, ready=_spectral,
        strategies=("fixed_range", "zscore", "rolling_minmax"),
    ),
    "peak_proximity": MetricSpec(
        id="peak_proximity", label="peak proximity", unit="",
        description="how close the strongest peak is to the target, ignoring how sharp it is",
        detail=("1.0 with the peak exactly on the target, falling to 0 at the edges of the searchable "
                "range — which stays 0.04–0.4 Hz wherever the target is put, so the question does not "
                "narrow with the aim. Deliberately blind to sharpness: it answers 'am I breathing at "
                "the right rate' and nothing else, which makes it the metric to use while somebody is "
                "hunting for the rate and a poor one once they have found it."),
        extract=_peak_proximity, ready=_spectral,
        strategies=("fixed_range", "rolling_minmax", "zscore"),
    ),
    "spectrum_distance": MetricSpec(
        id="spectrum_distance", label="distance from ideal", unit="Hz",
        description="how far the spectrum is from a single spike at the target",
        detail=("An earth-mover distance: the average distance, in hertz, that the spectrum's power would "
                "have to be carried to pile it all at the target. 0 is the ideal, so **low is good** and the "
                "feedback is inverted.\n\n"
                "Continuous in *how far off* the energy is rather than in whether it crossed a band edge, "
                "which is what a person adjusting their breathing needs the number to do."),
        extract=lambda s: s.spectrum.distance_hz, invert=True, ready=_spectral,
        strategies=("rolling_minmax", "zscore", "fixed_range"),
    ),
    "lf_hf_ratio": MetricSpec(
        id="lf_hf_ratio", label="LF / HF", unit="",
        description="low-frequency power over high-frequency power",
        detail=("The conventional bands, 0.04–0.15 against 0.15–0.4 Hz. It is the number that speaks to "
                "the split failure — sympathetic drive and parasympathetic drive at once — and it is "
                "also the most argued-over quantity in the whole field: reading it as a sympathetic "
                "balance is not supported, and it moves with breathing rate whatever the state.\n\n"
                "Offered because it says something no other metric here does, and inverted because the "
                "concern is LF drive rising. Watch the spectrum beside it rather than trusting it alone."),
        extract=_lf_hf, invert=True, ready=_spectral,
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
    "rsa_amplitude": MetricSpec(
        id="rsa_amplitude", label="RSA swing", unit="bpm",
        description="peak-to-trough heart-rate swing over the last 20 s",
        detail=("The plainest reading of respiratory sinus arrhythmia, and the one a trainee can feel: at "
                "resonance the rate visibly climbs on the inhale and falls on the exhale. Bigger is "
                "better.\n\n"
                "It needs no spectrum, so it is the metric that works first — useful in the opening "
                "minute while the spectral ones are still filling their window."),
        extract=_rsa_amplitude, ready=_beats_ready(RSA_MIN_BEATS),
        strategies=("rolling_minmax", "zscore", "fixed_range"),
    ),
    "rmssd": MetricSpec(
        id="rmssd", label="RMSSD", unit="ms",
        description="root mean square of successive interval differences",
        detail=("Beat-to-beat variability, dominated by the fast parasympathetic component. Says how much "
                "variability there is and nothing about where it sits in frequency, which makes it a "
                "robust thing to reinforce before a spectrum has organised — and a poor target "
                "afterwards, since it rises with any jitter including artifact."),
        extract=lambda s: s.rmssd_ms, ready=_beats_ready(TIME_DOMAIN_MIN_BEATS),
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
    "sdnn": MetricSpec(
        id="sdnn", label="SDNN", unit="ms",
        description="standard deviation of the intervals",
        detail=("Total variability over the whole buffer, so it includes the slow drifts RMSSD misses. "
                "Slower to move than anything else here, which makes it a reasonable session-level "
                "readout and a sluggish feedback signal."),
        extract=lambda s: s.sdnn_ms, ready=_beats_ready(TIME_DOMAIN_MIN_BEATS),
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
    "pnn50": MetricSpec(
        id="pnn50", label="pNN50", unit="%",
        description="share of successive intervals differing by more than 50 ms",
        detail=("A counted rather than averaged measure, so one artifact moves it far less than it moves "
                "RMSSD. The cost is coarseness: at low variability it sits at zero and says nothing, "
                "because no interval pair crosses the 50 ms line at all."),
        extract=lambda s: s.pnn50_pct, ready=_beats_ready(TIME_DOMAIN_MIN_BEATS),
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
    "hr_smoothed": MetricSpec(
        id="hr_smoothed", label="heart rate", unit="bpm",
        description="the smoothed rate itself",
        detail=("Not a variability measure, and offered because down-training it is a legitimate thing to "
                "want: a rate falling over a session is what relaxation looks like whatever the spectrum "
                "does. Inverted for that reason.\n\n"
                "It is also the only metric here that is meaningful from the second beat, which makes it "
                "the one to use for checking that a feedback channel is wired up at all."),
        extract=lambda s: s.hr_smoothed, invert=True,
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
}

DEFAULT_METRIC = "coherence"
"""The default is `coherence`, because it is the one measure here that distinguishes entrainment from in-band
energy. `resonance_ratio` with a fixed range needs no calibration, but `rolling_minmax` does the calibration
automatically."""


def metric_value(metric: MetricSpec, state) -> float | None:
    """The metric's value for this state, or None while it has nothing to say.

    None rather than 0.0, and the distinction matters: zero is a reading and "no reading yet" is not. A
    channel handed 0.0 would drive itself to its floor and look like a bad result.
    """
    if state is None or not metric.ready(state):
        return None
    try:
        return float(metric.extract(state))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None
