"""The amplitude protocol screen.

Organised in tabs, because the things on it do not change at the same rate. Targets and traces are
watched second by second; the spectrum is consulted; feedback settings are decided once and then left
alone. A single scrolling page gives all three equal height, which means the tenant that never moves
takes a third of the window.

- **Train** — the target graph, the raw traces, the target list. Nothing else: this is the tab that is
  watched for twenty minutes, so anything that is not being read during training belongs elsewhere.
- **Spectrum** — the frequency content of one channel, scaled on both axes, with band tiles.
- **Trends** — where each band has been over a chosen span, with averages and peak frequencies.
- **Feedback** — what the person is getting, the signal settings, then one pane per mechanism.
- **Session** — the optional maximum length, and the cues that mark time inside it.

The feedback signal is driven by *every* target, rewards and inhibits, all channels. Nominating one
target as the driver would mean the rest are measured, displayed, and then ignored by the only output
the person can perceive — the screen and the sound would be training different things.

**A card says two things about its target, and they are separate questions**: what it does to the
feedback (reward, inhibit, monitor) and which way it is trained (increase, decrease). They were one
control, which made "reward" a synonym for "increase" and left no way to express an arrangement whose
subject band is trained *downward* — Ros et al.'s alpha-down is the standing case. See `targets.ROLES`.

Other decisions worth stating: band edges are editable numbers, and adding a target appends a card and
scrolls to it rather than opening a menu of presets — the card can already set every field, so a menu
only makes you choose the band before you can see what you are choosing for. Thresholds are fixed or
automatic, and a *calibration* is a phase rather than a third mode. Every graph states its window and
its scale, because silent autoscaling makes a trace decorative. A target reads one channel or the channels
**added together** — a real sum, filtered once, so a rhythm shared between two sites in phase adds and one
that is not does not.

**Custom is that same combining with a sign per channel**, which is how a difference is trained. `ch0−ch1`
under a linked reference is `(C4 − Pz) − (C3 − Pz)`: the reference cancels and the montage that comes out
is C4 − C3, which would otherwise mean rewiring. It is the last item in the source menu and it opens a
dialog rather than adding items to the menu, because the menu's entries are single choices and this one has
operands — see `open_combination`, and `Target.signs` for why a signed target does not follow the live
channels the way the sum does.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import ALPHA, DELTA, SMR, THETA, Band, ChannelAligner
from nfb.dsp.artifact import JUDGE_FLOOR_HZ
from nfb.stream import NO_SIGNAL, OK, Block

from . import chrome, plan_pane, plots, signal_pane, spectrum_pane, trace_rows, trends_pane
from . import theme as t
from . import target_card
from .protocols import Protocol
from .target_card import BAR_W, TargetCard
from .targets import PAIR, ROLES, Target
from .plots import (StickyCeiling, bars, bin_centre_x, filled, fmt_amplitude, grid,
                    gridlines, nice_ceiling, place_tick_labels, place_value_labels, polyline, tick_axis,
                    value_axis)
from .traces import RawRing, TraceBuffer
from .training import CFIR_DELAY, DEFAULT_FILTER, METRIC_HZ, TrainingSession
from .widgets import num_box, panel, pill, tiny

TRACE_H = 92
# The ring's band is this screen's choice — **to 60 Hz so that mains is visible**, where CFC stops at 45.
# Everything about drawing it is `spectrum_pane`, including the scale choices and the tick step.
SPECTRUM_LO, SPECTRUM_HI = 1.0, 60.0
SPECTRUM_BIN_HZ = 1.0
SUM_KEY = "sum"
"""Where a channel number goes in an estimator key, for the one signal that is not a channel.

A summed target is measured once on the channels added together, so its key cannot carry a channel number —
and must not be able to collide with one, which is why it is a string. Alpha-theta keys its whole metric
table this way for the same reason."""
GRAPH_H = 176
SIGNAL_BAR_W = 300
BAR_W = 176                                        # the per-target bar on a card
TARGETS_W = 430                                    # the target column, fixed; the graph flexes
AXIS_W = 40                                        # room for amplitude labels left of a plot
TICK_W = 30                                        # a frequency label, centred on its gridline
MOVING_AVERAGE_SECONDS = 30.0

# µV full scale for the raw trace. **±50 is the default, not ±100, and that was a real bug rather than a
# preference**: measured on this device with a good electrode at C4 against a linked Pz reference, resting EEG
# runs **12.6 µV peak-to-peak** (p99 13.8, max 17.3 over 30 s). On ±100 that fills 5.8 px of a 92 px trace,
# which reads as a flat line with nothing happening — the screen looked broken on a first session while the
# signal was perfectly good. ±10 exists because a quiet channel is the case that needs it most.
GAINS = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0)
GAIN_HEADROOM = 1.6        # how much room above the recent peak `fit` leaves before it clips
TRACE_WINDOWS = (2.0, 5.0, 10.0)                   # seconds across the raw trace
GRAPH_WINDOWS = (("30 s", 30.0), ("2 min", 120.0), ("10 min", 600.0), ("session", 0.0))

# **Direction names the movement; the role says what is done about it** — see `targets.ROLES`. The menu
# spells the consequence out rather than leaving it to be inferred, because "inhibit increase" is the one
# combination whose plain reading and whose threshold point opposite ways: rises are what is suppressed,
# so it is met while the band stays *low*.
MENU_NOTES = {
    ("reward", True): "earned as this band rises",
    ("reward", False): "earned as this band falls",
    ("inhibit", True): "fires as this band rises — expect it to fall",
    ("inhibit", False): "fires as this band falls — expect it to rise",
}

# What a session opens with, per protocol: `(low, high, direction, role, source)`. Keyed by code so a
# second protocol that wants this screen with different starting targets is a table entry rather than a
# second implementation.
#
# **The inhibits open on the sum and the reward on one channel**, which is the asymmetry the two are for.
# A reward is a claim about a site — alpha at this electrode, trained upward — and putting it on the sum
# would train something no montage names. An inhibit is a veto on contamination, and contamination is not
# site-specific: a jaw clench, a drifting electrode or mains pickup arrives on whatever is wired, so a
# veto that watches one channel is blind to exactly the input the operator is not looking at. Watching the
# channels added together catches it wherever it lands, which is what a guard band is for.
#
# A new target added by hand starts as a copy of the first entry rather than opening a menu.
#
# **Every row is `increase` and the roles do the rest**, which is the whole shape of the two-axis model:
# reward rises in alpha, inhibit rises in delta and in high beta. Those last two used to be spelled
# `decrease` — the same behaviour, described by the side of the line it is met on rather than by what is
# being suppressed — which made the ordinary arrangement read as three targets pointing two ways when it
# is one sentence with one word changed. See `targets.ROLES`.
#
# A preset that trains its subject band downward, as Ros et al.'s alpha-down does, is a row here with
# `("decrease", "reward")` and nothing else; it is not a protocol and has no card in the launcher.
FIRST_CHANNEL = -2
"""Pseudo-source for the table only: whichever channel this session opens with. `PAIR` is a real source
and means the sum; there is no constant meaning "the first one", because everywhere else the source is
already known by the time a target is built."""

DEFAULTS: dict[str, tuple[tuple[float, float, str, str, int], ...]] = {
    "AMP": ((8.0, 12.0, "increase", "reward", FIRST_CHANNEL),
            (2.0, 6.0, "increase", "inhibit", PAIR),
            (19.0, 38.0, "increase", "inhibit", PAIR)),
}
HANDLES = tuple(DEFAULTS)          # which protocol codes this screen can run
DEFAULT_TARGETS = DEFAULTS["AMP"]

class AmplitudeSession(TrainingSession):
    """Band amplitude per channel: traces, spectra and targets.

    Everything generic — the source, the clock, the plan, artifact rejection, the one feedback signal and
    its outputs — is `TrainingSession`. What is here is what makes this the *amplitude* screen: a cFIR
    estimator per channel per band, a raw ring for the spectrum, and one more estimator on the channels
    summed, which `PAIR` selects.
    """

    sustains_reward = True
    """**This screen's condition is every target met at once, and holding that is a state worth marking.**

    Unlike alpha-theta's, the distinction is not borrowed from anyone: there is no reference design behind
    this screen to be faithful to, so the question is only whether "you are there" and "you have been there
    for the last few seconds" are different things here — and with three targets that have to be satisfied
    together, they plainly are.

    It is declared because the doubling arithmetic in `_maybe_click` runs on **every** session and only a
    screen that sets this gets the row that configures it. Left unset, an operator on this screen could hear
    a mark change character with nothing on the Feedback tab saying why or letting them turn it off — see
    `TrainingSession.sustains_reward`, where the hidden case is now closed on the screens that do not set it.
    """

    music_sources = ("noise", "file", "tone", "eeg")
    """Adds the pitch-tracking tone to the generic three.

    The tone maps the combined score between two frequencies, so it needs nothing this screen does not
    already have — the score is what every mechanism here follows. It earns its place because sound is
    this screen's only continuous channel that does not compete with looking at the traces: a pitch is
    read without turning your head, where the number and the bars are not. `crossfade` stays out for the
    reason `music_sources` gives — there is no state here for the two ends to stand for.
    """

    tunes_envelope = True
    """`envelope_seconds` is part of this screen's estimator identity — `filter_for` keys on it
    and `retune_filter` rebuilds when it moves, which is what makes the SIGNAL panel's first step a
    live control here and absent on the lock-in screens."""

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0):
        super().__init__(protocol, path=path, device=device, length_seconds=length_seconds,
                         source=source, channels=channels)
        self.trace_seconds = TRACE_WINDOWS[1]
        self.gain_uv = GAINS[2]
        self.graph_window = GRAPH_WINDOWS[1][1]

        self.traces = {i: TraceBuffer(320, self.trace_seconds, s.rate_hz)
                       for i, s in self.specs.items()}
        self.rings = {i: RawRing(2.0, s.rate_hz, SPECTRUM_LO, SPECTRUM_HI, SPECTRUM_BIN_HZ)
                      for i, s in self.specs.items()}
        # Keyed by the design delay as well as the band, because the delay follows the *direction* and
        # one band can be trained in both — two estimators is 0.06% of a core against silently giving one
        # direction the other's filter.
        self.metrics: dict[tuple, object] = {}

        # **The summed signal, which this screen did not have.** What stood in for it measured each channel
        # separately and averaged the answers, so it could not see phase at all — see `targets.PAIR`. Adding
        # the channels first and filtering once is the measure a shared reference makes worth having: a rhythm
        # present at two sites in phase adds, and one that is not does not.
        #
        # Blocks arrive a channel at a time, so adding whatever landed most recently would drift the channels
        # against each other by up to a frame; `ChannelAligner` hands back the runs covering the same samples.
        # `site` is whose spec the summed signal borrows, because an estimator checks a block's channel
        # against its own spec and there is no channel number that means "the sum" — the same arrangement
        # alpha-theta uses, for the same reason.
        #
        # **Two aligners, and they are not interchangeable** — the same division `ilf_view` records. An
        # aligner is stateful and hands each run over exactly once, so one object cannot serve both the
        # measurement and the display. They also see different data on purpose: `consume` runs only on blocks
        # the artifact gate accepted, while `observe_raw` sees everything, and a sum spectrum that hid what
        # was rejected would disagree with the per-channel spectra beside it.
        self.sites = tuple(self.specs)
        self.site = next(iter(self.specs), 0)
        # **The measuring aligner exists with one channel too, and the display one does not.** The sum of a
        # single channel is that channel, which `ChannelAligner` already handles — it hands back each
        # block's own run — so a target on the sum measures something real whatever is wired. That matters
        # because the default inhibits open on the sum: gating it on a second channel would leave a session
        # on one electrode with two inhibits reading nothing and no card saying why.
        #
        # The *spectrum* is the other way round. A sum spectrum beside a single channel's is the same
        # picture twice, offered as though it were a choice, so that one still needs two.
        self.aligner = ChannelAligner(self.sites) if self.sites else None
        self.display_aligner = ChannelAligner(self.sites) if len(self.sites) > 1 else None
        # Offered in the Spectrum tab as well as measured, since it is the same signal a summed target is
        # measured on — so the spectrum and the reward are looking at one thing. **After** the channels, which
        # is the opposite of the coupling screen's order and is the claim: there every reward is measured on
        # the sum, here a target sits on one channel unless somebody moves it. See `sum_spectrum_leads`.
        self.sum_ring = (RawRing(2.0, self.rate, SPECTRUM_LO, SPECTRUM_HI, SPECTRUM_BIN_HZ)
                         if self.display_aligner is not None else None)

        first = next(iter(self.specs), 0)
        for low, high, direction, role, source in DEFAULTS.get(protocol.code, DEFAULT_TARGETS):
            self.add_target(first if source == FIRST_CHANNEL else source,
                            Band(low, high, ""), direction, role)

    # --- targets -------------------------------------------------------------------------

    def _ensure_metrics(self, tgt: Target) -> None:
        # **One estimator for a summed target, several for a mean one**, and that is the whole difference
        # between the two sources expressed as code: a sum is one signal, so it is filtered once; a mean is
        # N signals filtered separately and averaged afterwards.
        if tgt.source == PAIR:
            key = self.key_for(tgt)
            spec = self.specs.get(self.site)
            if spec is not None and key not in self.metrics:
                try:
                    self.metrics[key] = self.build_metric(spec, tgt.band, tgt)
                except ValueError:
                    pass            # band reaches Nyquist at this rate; leave it unmeasured
            return
        for ch in [tgt.source]:
            if ch not in self.specs:
                continue
            key = (ch, self.band_key(tgt.band), self.filter_for(tgt))
            if key not in self.metrics:
                try:
                    self.metrics[key] = self.build_metric(self.specs[ch], tgt.band, tgt)
                except ValueError:
                    pass            # band reaches Nyquist at this rate; leave it unmeasured

    def add_target(self, source: int, band: Band, direction: str = "increase",
                   role: str = "reward") -> Target:
        """A reward on the way up unless told otherwise, which is the default everywhere — see
        `targets.ROLES`. The presets state both; a target added by hand states neither."""
        tgt = Target(source, Band(band.low, band.high, band.name), direction, role=role)
        self._set_source(tgt, source)
        self._ensure_metrics(tgt)
        self.targets.append(tgt)
        return tgt

    def summed_sites(self) -> tuple[int, ...]:
        """Which channels actually feed the sum: the ones delivering signal.

        **A dead electrode used to take the whole summed target down with it, silently.** `pair` names
        every contributing channel and `target_is_live` requires all of them, which is right for a
        measurement *between* channels — a synchrony over a pair is meaningless if one side is gone. A sum
        is not that: it is one signal that several electrodes contribute to, and dropping one contributor
        leaves a smaller sum rather than no measurement. Reported: four channels enabled with three
        electrodes wired, and **both default inhibits sat inert at 0.00 µV for the whole session**, never
        calibrated, never in the level, never in the click — with nothing on the card saying why. The
        default inhibits open on the sum, so that is the ordinary case rather than a corner.

        An unknown channel counts as contributing, the same polarity `target_is_live` documents: exclusion
        needs evidence. With nothing live at all this falls back to every site, so the target reads dead —
        which it is — rather than reading an empty sum as a healthy one.
        """
        live = tuple(c for c in self.sites if self.state.get(c) is not False)
        return live or self.sites

    def _judge_channel(self, block) -> None:
        """Judge the channel, then keep every summed target pointed at the channels that survived.

        Repointing `pair` rather than overriding `target_is_live` is what keeps the card honest: the label
        is built from `pair`, so a target measured on two of four electrodes says `ch0+ch1` instead of
        claiming four — and the inherited "every contributing channel live" rule stays exactly as written,
        because the contributors really are the live ones now.

        The estimator is untouched: it is keyed by band and filter, not by which channels feed it, so a
        channel dropping out changes the amplitude arriving at a running filter and nothing else. That is
        the truthful description of what happened to the signal.
        """
        super()._judge_channel(block)
        sites = self.summed_sites()
        for tgt in self.targets:
            # Only the plain sum follows the live channels. A signed combination names its channels, and
            # dropping one would change the montage rather than shrink the sum — see `Target.signs`.
            if tgt.source == PAIR and tgt.signs is None and tgt.pair != sites:
                tgt.pair = sites

    def set_combination(self, tgt: Target, channels, signs) -> None:
        """Measure this target on named channels with a sign each — what the Custom source sets.

        The estimator is rebuilt because this is a different signal, not the same one scaled: `ch0−ch1`
        and `ch0+ch1` are two montages and two readings, so they cannot share a key or a history."""
        tgt.source = PAIR
        tgt.pair = tuple(channels)
        tgt.signs = tuple(1 if s >= 0 else -1 for s in signs)
        tgt.reset_stats()
        self._ensure_metrics(tgt)
        self._drop_unused_metrics()

    def _set_source(self, tgt: Target, source: int) -> None:
        """Point a target at a channel, the mean, or the sum — and keep `pair` telling the truth.

        `pair` is what makes a summed target say "ch0+ch1" on its card and what makes `target_is_live`
        require *every* contributing channel rather than whichever block last arrived. So it is set when the
        source becomes the sum and cleared when it stops being one: a stale `pair` left behind by switching
        back to one channel would keep both claims running with nothing behind them.
        """
        tgt.source = source
        tgt.pair = self.summed_sites() if source == PAIR else None
        tgt.signs = None                # the plain sum, and no longer whatever combination preceded it

    def retune_filter(self, tgt: Target, kind: str | None = None, order: int | None = None,
                      cfir_delay: float | None = None, ripple: float | None = None) -> None:
        """Change what measures this target. Builds the estimator it now needs; the one it was using stays
        only if another target is still on it, since they are keyed by the filter.

        Switching between cFIR and the IIR kinds changes the amplitude *convention* — peak against RMS
        or mean-abs, a fixed ratio of √2-and-a-bit on a sine — so the threshold and the collected
        statistics stop describing what is now being measured. The threshold is rescaled by the sine
        ratio, which keeps a settled fixed target approximately right instead of exactly 29% wrong (one
        click used to silence a settled reward until someone pressed redefine).

        **The readings are rescaled too, rather than thrown away.** They used to be reset, on the argument
        that a window holding two conventions holds neither — right about the window, wrong about the
        readings, which differ from the new convention by that same one constant. So changing a filter
        blanked the card and started a fresh thirty-second calibration, on a screen where the whole point
        of pressing the pill is to hear what the other filter does. `Target.rescale_history` converts them
        and the card keeps answering; the threshold and the history stay in one unit, so the hit record is
        untouched. What is genuinely lost is nothing: the estimator's own transient is under a second
        against a thirty-second window, and a `redefine` afterwards is available for anyone who wants the
        threshold re-derived from data the new filter actually produced.
        """
        conventions = {"rms": 1 / np.sqrt(2.0), "meanabs": 2.0 / np.pi}  # of peak, for a sine
        before = self.filter_for(tgt)
        was = conventions.get("" if before[0] == "cfir" else before[-1], 1.0)
        if kind is not None:
            tgt.filter_kind = kind
        if order is not None:
            tgt.filter_order = max(1, min(10, int(order)))
        if cfir_delay is not None:
            tgt.cfir_delay = round(max(0.02, min(1.0, cfir_delay)), 3)
        if ripple is not None:
            tgt.filter_ripple = round(max(0.05, min(6.0, ripple)), 2)
        after = self.filter_for(tgt)
        now = conventions.get("" if after[0] == "cfir" else after[-1], 1.0)
        if was != now:
            tgt.threshold *= now / was
            tgt.rescale_history(now / was)
        self._ensure_metrics(tgt)
        self._drop_unused_metrics()

    def _drop_unused_metrics(self) -> None:
        wanted = set()
        for tgt in self.targets:
            key = (self.band_key(tgt.band), self.filter_for(tgt))
            if tgt.source == PAIR:
                wanted.add((self.combination_key(tgt), *key))
                continue
            wanted.add((tgt.source, *key))
        for key in [k for k in self.metrics if k not in wanted]:
            del self.metrics[key]
            self.amplitudes.pop(key, None)

    def retune(self, tgt: Target, low: float | None = None, high: float | None = None,
               source: int | None = None) -> None:
        if source is not None:
            # A *source* change is the genuine restart: the values came from a different electrode, so
            # nothing collected is about the new one. A threshold or band edit is not — see `Target.rescore`.
            self._set_source(tgt, source)
            tgt.reset_stats()
        elif tgt.source == PAIR:
            # An **adopted** target arrives carrying the previous session's `pair`, and the sites are this
            # session's. `adopt_targets` reaches this method with no source, which is the one path where a
            # summed target can end up naming electrodes that belong to a session that has ended.
            tgt.pair = self.summed_sites()
        if low is not None or high is not None:
            tgt.set_band(low, high)
        self._ensure_metrics(tgt)
        self._drop_unused_metrics()

    def combination_key(self, tgt: Target):
        """Which combined signal this target is measured on.

        `SUM_KEY` for the plain sum, because there is one such signal here however many electrodes feed
        it. A signed combination is its own signal and gets its own key, so two targets on `ch0−ch1` share
        one estimator while a third on `ch0+ch1` gets another — the keys are what stop those two being
        silently the same measurement."""
        if tgt.signs is None:
            return SUM_KEY
        return (SUM_KEY, tuple(zip(tgt.pair or (), tgt.signs)))

    def key_for(self, tgt: Target):
        # Keyed by the combined signal rather than by a channel for a summed target, and because `_due`
        # compares this against the keys `consume` reports fresh.
        if tgt.source == PAIR:
            return (self.combination_key(tgt), self.band_key(tgt.band), self.filter_for(tgt))
        return (tgt.source, self.band_key(tgt.band), self.filter_for(tgt))

    def value_for(self, tgt: Target) -> float:
        return self.amplitudes.get(self.key_for(tgt), 0.0)

    def latency_for(self, tgt: Target) -> float | None:
        """How far behind the signal this target's reading is. Static per target."""
        metric = self.metrics.get(self.key_for(tgt))
        return metric.latency_seconds if metric is not None else None

    # --- feed ----------------------------------------------------------------------------

    def observe_raw(self, block, judged) -> None:
        """Traces and spectra see every sample, including the ones the gate rejected.

        Deliberately: an operator who cannot see what was rejected cannot tell a blink from a lost
        electrode, and the whole reason the rejection is visible on this screen is so the decision to
        trust it is an informed one.
        """
        ch = block.channel
        values = np.asarray(block.values)
        # The trace is drawn band-limited; the spectrum is not, because the spectrum's whole job is to
        # show you what is in the signal — including the hum the trace filters out.
        self.traces[ch].push(self.for_display(ch, values), block.valid, judged.accepted)
        self.rings[ch].push(values)
        summed = self._summed(self.display_aligner, block)
        if summed is not None and self.sum_ring is not None:
            self.sum_ring.push(np.asarray(summed.values))
        if self.feedback.music.source == "eeg" and ch == self.focus:
            self.music.feed(values)

    def _combined(self, paired, combo):
        """One combined `Block` from an aligner's runs: the plain sum, or named channels with signs.

        Same arithmetic either way — counts added, one sign per channel — so nothing downstream can tell
        a difference from a sum, which is the point: it is one signal and one estimator measures it.

        **A signed combination is usable only if every channel it names is.** The plain sum can lose a
        contributor and stay a sum; `ch0 − ch1` with ch1 gone is not a quieter difference, it is ch0. So
        this reports no-signal rather than handing back something that would read as a montage nobody
        chose."""
        start, runs = paired
        spec = self.specs[self.site]
        if combo == SUM_KEY:
            wanted = {c: 1 for c in self.summed_sites()}
        else:
            wanted = {c: s for c, s in combo[1]}
        counts = np.zeros(len(runs[0]))
        for ch, values in zip(self.sites, runs):
            if ch in wanted:
                counts += wanted[ch] * values / spec.scale
        worst = OK if all(self.state.get(c) is not False for c in wanted) else NO_SIGNAL
        return Block(self.site, start, tuple(int(round(v)) for v in counts), spec, state=worst)

    def _summed(self, aligner, block):
        """The channels added, as one `Block`, or None until every channel covers the same samples.

        The display path's own aligner, so the picture and the measurement stay lined up without either
        consuming the other's runs. `_combined` does the arithmetic for both.
        """
        if aligner is None:
            return None
        paired = aligner.push(block)
        if paired is None:
            return None
        return self._combined(paired, SUM_KEY)

    # --- what this protocol requires -------------------------------------------------------

    ESTIMATOR_LEAK = 0.06
    """How much of an out-of-band signal reaches a band estimate, measured on the shipped cFIR design.

    `design_cfir` truncates an ideal response to 250 taps without a taper, and a truncated filter leaks:
    the response outside the band is a floor of about 6% of the in-band gain at **every** frequency,
    including DC. Measured on the 8-12 Hz design at 100 ms, relative to in-band 1.0 — 0 Hz 0.062,
    0.5 Hz 0.054, 2 Hz 0.078, 20 Hz 0.062. A Butterworth order 2 over the same band reads 0.00002 at
    0.05 Hz, so this is the estimator's own property and not something the band width forces.

    Six percent is harmless against anything the size of EEG and ruinous against electrode drift, which is
    why `CFIREnvelope` now high-passes two octaves below its own band before the fit ever sees the signal.
    Kept as a measured property of the design rather than as a live hazard: it still governs how much of a
    *neighbouring band* reaches an estimate, which no high-pass touches — see `wiring_problem` for what is
    left of the corner's part in it.
    """

    def wiring_problem(self) -> str | None:
        """Why the numbers would be wrong, if they would be. None when there is nothing to say.

        **One thing only, and it is the device's high-pass corner** — but not the thing it used to be, and
        the change is worth stating because the old reason was much more serious than the one that remains.

        It used to be that the *measurement* was drift. `ESTIMATOR_LEAK` puts 6% of everything outside a
        cFIR's band inside its estimate, the standing drift at the 0.01 Hz corner is hundreds of
        microvolts — 43 to 444 µV RMS below 0.5 Hz on the two captures in `sessions/` — and 6% of that is
        larger than alpha. The same captures read **20 to 44 µV of "alpha" against 3 µV of real alpha**,
        correlating 0.77–0.98 with the drift and 0.00–0.06 with the alpha, so every threshold and every
        click followed the electrodes settling. That is fixed at the estimator now: `CFIREnvelope` carries
        a drift high-pass two octaves below its own band, which costs 0.7 ms at 12 Hz and takes the same
        captures to 2.0–2.1 µV. It should have been fixed there when it was first measured; warning about
        a setting was treating a screen's symptom.

        What remains is real and smaller: **the drift is still in the signal even though it is out of the
        bands**, so it uses up the amplifier's input range. Measured on those captures, the wired channels
        peak at 48–85% of the ±800 µV range on drift alone, and a channel that reaches the rail stops
        carrying EEG at all rather than carrying it noisily. The artifact gate cannot help — it judges
        above `JUDGE_FLOOR_HZ`, so it never sees the excursion that spends the headroom.

        The corner is a device setting rather than a protocol one, and it is left where the last screen
        needed it: the infra-low, waveform and SCP screens all require 0.01 Hz, so arriving here with it
        still set is the ordinary route rather than a mistake. Hence a warning and not a refusal.

        Silent where a target actually trains below the floor, which on this screen would have to be
        somebody's deliberate band edit — there the slow content is the signal and the corner is right.
        """
        if self.corner_hz >= JUDGE_FLOOR_HZ or self.trained_floor_hz < JUDGE_FLOOR_HZ:
            return None
        # Kept under the screens' 110-character limit, which is why it names the setting and the remedy and
        # leaves the measurement to this docstring.
        return (f"device high-pass {self.corner_hz:g} Hz: drift is out of the bands now, but it still "
                f"fills the input range — 0.5 Hz is safer")

    def consume(self, block) -> set:
        ch = block.channel
        fresh: set = set()
        # Pushed once per block however many combinations are in use: the aligner is stateful, so a
        # second push would consume a run nothing had measured.
        paired = self.aligner.push(block) if self.aligner is not None else None
        if paired is not None:
            combined: dict = {}
            for mkey, metric in list(self.metrics.items()):
                combo = mkey[0]
                if combo != SUM_KEY and not (isinstance(combo, tuple) and combo[0] == SUM_KEY):
                    continue
                if combo not in combined:
                    combined[combo] = self._combined(paired, combo)
                for r in metric.feed(combined[combo]):
                    self.amplitudes[mkey] = r.value
                    fresh.add(mkey)
        for mkey, metric in list(self.metrics.items()):
            if mkey[0] == ch:
                for r in metric.feed(block):
                    self.amplitudes[mkey] = r.value
                    fresh.add(mkey)
        return fresh

def session_for(*args, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None,
                **kwargs) -> AmplitudeSession:
    """What `app.pick` calls. A named factory rather than the class, so the dispatch table in `app`
    treats every screen module the same way regardless of what its session class is called — and so a
    screen can ignore what it does not use, which is what the heart and headband arguments are here.

    Named rather than swallowed by `**kwargs`: they would otherwise be forwarded straight into the
    session's constructor, which is a `TypeError` at the moment `app.pick` runs — i.e. on the click that
    opens the screen."""
    _ = heart_choice, heart_runner, mendi_choice, mendi_runner, pupil_choice, pupil_runner
    return AmplitudeSession(*args, **kwargs)


# --- the screen --------------------------------------------------------------------------

def view(page: ft.Page, session: AmplitudeSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    """Build once, mutate on Flet's loop.

    `on_stop` ends the session and leaves you **here**, on a fresh one. It used to go to the launcher, which
    was wrong: stopping is something you do *to the session*, not a way of leaving the screen, and the way
    out is the logo — which is already the way out from everywhere else.
    """

    fb = session.feedback
    # The Feedback and Session tabs are the shared panes: nothing on either depends on what is being
    # trained, so they are built here rather than reimplemented per protocol.
    signal = signal_pane.SignalPane(page, session)
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width(reserve: int = TARGETS_W) -> float:
        """How wide to draw, in canvas pixels.

        Shapes are placed in absolute coordinates while the canvas itself is `expand`, so drawing
        wider than the pane pushes the pane's own contents out of view — which is exactly how the
        graph-window pills disappeared at the default window size. Flet exposes no post-layout width,
        so this derives it from the window and errs narrow: unused space on the right is invisible,
        overflow is not.
        """
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - reserve - 70)

    graph = cv.Canvas(shapes=[], expand=True, height=GRAPH_H)
    graph_axis, graph_axis_labels = value_axis(GRAPH_H, count=4, width=AXIS_W)
    # The caption's "0–8 µV full scale" was the only statement of scale, which leaves the reader doing
    # arithmetic against gridlines that name nothing. Sticky rather than raw window max, because an axis
    # that renumbers itself every frame is the strobe `StickyCeiling` exists to prevent.
    graph_ceiling = StickyCeiling()
    graph_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    graph_legend = ft.Row(spacing=14)
    graph_empty = ft.Text("waiting for data", size=t.SMALL, color=t.FAINT)
    window_pills = ft.Row(spacing=6)

    def set_window(secs: float) -> None:
        session.graph_window = secs
        build_pills()
        draw()

    def build_pills() -> None:
        window_pills.controls = [
            pill(name, session.graph_window == secs, (lambda s=secs: lambda _e: set_window(s))())
            for name, secs in GRAPH_WINDOWS]

    # ---- raw trace ----
    # `notes=True` because this screen has the room: the words go over the trace, and a flat grey line with a
    # small red dot beside it is what a broken app looks like.
    traces = trace_rows.TraceRows(page, session, plot_width, height=TRACE_H, notes=True)
    gain_pills = ft.Row(spacing=6)

    def set_gain(uv: float) -> None:
        session.gain_uv = uv
        build_gain_pills()
        draw()

    def fit_gain(_e=None) -> None:
        """Pick the smallest preset that holds what the traces are actually doing.

        Not autoscaling: it *selects a preset*, so the caption still names one fixed number and a trace can
        still be compared with one from yesterday. It exists because the right scale is unguessable before
        you have seen the signal — resting EEG here measures 12–20 µV peak-to-peak, which on ±100 is 6 px
        of a 92 px trace and reads as a flat line.
        """
        peaks = [buf.scale() for buf in session.traces.values() if buf.valid]
        peak = max(peaks) if peaks else 0.0
        wanted = peak * GAIN_HEADROOM
        session.gain_uv = next((g for g in GAINS if g >= wanted), GAINS[-1])
        build_gain_pills()
        draw()

    def build_gain_pills() -> None:
        gain_pills.controls = [
            *[pill(f"±{g:g}", session.gain_uv == g, (lambda v=g: lambda _e: set_gain(v))())
              for g in GAINS],
            ft.Container(width=8),
            pill("fit", False, fit_gain),
        ]

    # ---- spectrum ----
    # The whole tab is `spectrum_pane.SpectrumPane`, shared with every other screen that has one. It was
    # inline here until alpha-theta and CFC needed it, and it is the second-largest thing to have come out
    # of this file after the Feedback tab.
    spectrum = spectrum_pane.SpectrumPane(page, session, plot_width)
    trends = trends_pane.TrendsPane(page, session, plot_width)

    # ---- targets ----
    target_column = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO)
    cards: dict[int, TargetCard] = {}

    def filter_rows(tgt, redraw) -> list:
        """The per-target filter gear, built by `target_card` so every screen's is the same dialog."""
        return target_card.filter_rows(session, tgt, lambda: (rebuild_targets(), redraw()))

    def slide(tgt, delta: float) -> None:
        """Move a band by `delta` and keep its width — what the arrows on either edge do."""
        tgt.shift_band(delta)
        session.retune(tgt)
        rebuild_targets()

    def open_combination(tgt: Target) -> None:
        """The Custom source: a sign per channel, and the expression it comes to.

        A dialog rather than more items in the menu, because the menu's entries are single choices and
        this one has operands — and because the pairs grow as the square of the channel count while a
        sign per channel grows with it.

        It opens on whatever the target is measured on now, so Custom edits the source rather than being
        a separate thing to switch into: one channel arrives as that channel `+`, the sum as every
        channel `+`.
        """
        show = getattr(page, "show_dialog", None)
        if show is None:
            return
        if tgt.signs is not None and tgt.pair:
            chosen = {c: s for c, s in zip(tgt.pair, tgt.signs)}
        elif tgt.source == PAIR:
            chosen = {c: 1 for c in (tgt.pair or session.sites)}
        else:
            chosen = {tgt.source: 1}

        expression = ft.Text("", size=t.BODY, color=t.TEXT)
        note = ft.Text("", size=t.SMALL, color=t.FAINT)
        rows: list[ft.Control] = []

        def refresh() -> None:
            picked = [c for c in session.sites if chosen.get(c)]
            expression.value = (Target(PAIR, tgt.band, pair=tuple(picked),
                                       signs=tuple(chosen[c] for c in picked)).source_label
                                if picked else "—")
            minus = [c for c in picked if chosen[c] < 0]
            if not picked:
                note.value = "a target has to be measured on something"
            elif not minus:
                note.value = ("one channel" if len(picked) == 1
                              else "added — a shared reference is in both, and adds with them")
            elif len(picked) == 2 and len(minus) == 1:
                note.value = "a difference — a shared reference cancels, leaving the two sites"
            else:
                note.value = "signed — inert if any of these channels dies"
            for row in rows:
                row.controls[1] = segment(row.data)
            try:
                page.update()
            except Exception:
                pass

        def set_sign(ch: int, sign: int | None) -> None:
            if sign is None:
                chosen.pop(ch, None)
            else:
                chosen[ch] = sign
            refresh()

        def segment(ch: int) -> ft.Control:
            """Three options as **one** control, because they are one question about this channel.

            One border around the group with flush cells inside it, rather than three bordered pills: a
            row of separate boxes reads as three independent switches, which is the thing this is not."""
            have = chosen.get(ch)
            cells: list[ft.Control] = []
            for n, (label, value) in enumerate((("+", 1), ("−", -1), ("off", None))):
                if n:
                    cells.append(ft.Container(width=1, height=22, bgcolor=t.BORDER))
                on = have == value
                cells.append(ft.Container(
                    content=ft.Text(label, size=t.SMALL,
                                    color=t.ON_ACCENT if on else t.MUTED,
                                    weight=ft.FontWeight.W_500 if on else ft.FontWeight.W_400),
                    width=34, height=22, alignment=ft.Alignment.CENTER,
                    bgcolor=t.ACCENT if on else None,
                    on_click=(lambda c=ch, v=value: lambda _e: set_sign(c, v))()))
            return ft.Container(content=ft.Row(cells, spacing=0),
                                border=ft.Border.all(1, t.BORDER), border_radius=6)

        for ch in session.sites:
            row = ft.Row([ft.Container(content=ft.Text(f"ch{ch}", size=t.BODY, color=t.TEXT), width=44),
                          segment(ch)],
                         spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER)
            row.data = ch
            rows.append(row)

        def apply(_e=None) -> None:
            picked = [c for c in session.sites if chosen.get(c)]
            if not picked:
                return
            session.set_combination(tgt, picked, [chosen[c] for c in picked])
            page.pop_dialog()
            rebuild_targets()

        refresh()
        body = ft.Column([*rows,
                          ft.Container(height=1, bgcolor=t.BORDER),
                          ft.Row([ft.Text("measures", size=t.SMALL, color=t.FAINT),
                                  ft.Container(content=expression, bgcolor=t.SURFACE_HI,
                                               border_radius=6, padding=ft.Padding(9, 4, 9, 4))],
                                 spacing=9, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                          note],
                         # **Tight, or the column takes every pixel the dialog will give it.** A Flet
                         # column fills the main axis by default, which turned four short rows into a
                         # dialog the height of the window with the content stranded at the top.
                         spacing=8, tight=True, width=250)
        page.show_dialog(ft.AlertDialog(
            title=ft.Text("Custom source", size=t.H2, color=t.TEXT, weight=ft.FontWeight.W_500),
            content=ft.Container(width=250, content=body),
            bgcolor=t.SURFACE,
            actions=[
                ft.Container(content=ft.Text("cancel", size=t.BODY, color=t.MUTED),
                             padding=ft.Padding(12, 7, 12, 7), border_radius=6,
                             on_click=lambda _e: page.pop_dialog()),
                ft.Container(content=ft.Text("measure this", size=t.BODY, color=t.ACCENT),
                             padding=ft.Padding(12, 7, 12, 7), border_radius=6,
                             on_click=apply)]))

    def rebuild_targets() -> None:
        cards.clear()
        rows = []
        for tgt in session.targets:
            # **"SUM" on the button, `ch0+ch1+ch2+ch3` in its tooltip.** The identity row is the widest
            # thing on a fixed 430 px card — a colour chip, this button, two band edges with steppers, the
            # direction and the remove — and the only part of it whose width depends on the session is this
            # label. Spelled out on four channels it is 15 characters, which pushed the remove button past
            # the card's right edge; on the sum it is also the one label that says the same thing every
            # time, since a summed target here always spans every channel (`_set_source` writes
            # `self.sites`). So the arithmetic moves to the tooltip and the card keeps its shape at any
            # channel count. Everything that has room still spells it: the graph legend, the target's
            # `label`, and the filter dialog's "measures".
            # A custom combination spells itself out — `ch0−ch1` and `ch0+ch1` are two montages, where SUM
            # is always every channel. Past two channels it hits the width problem SUM already has.
            summed = tgt.pair is not None
            custom = summed and tgt.signs is not None
            spelled = custom and len(tgt.pair or ()) <= 2
            source_menu = ft.PopupMenuButton(
                content=ft.Container(
                    content=ft.Text(tgt.source_label if spelled or not summed
                                    else ("CUSTOM" if custom else "SUM"),
                                    size=t.BODY, color=t.TEXT),
                    tooltip=(f"{tgt.source_label} — "
                             f"{'combined' if custom else 'added'}, then measured"
                             if summed else None),
                    padding=ft.Padding(8, 4, 8, 4), border_radius=6,
                    bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
                # One channel, or the channels added. The label says *added* because that is the whole content
                # of the option: it is one signal, filtered once, so what it can see and a single channel
                # cannot is phase. See `targets.PAIR`.
                items=[ft.PopupMenuItem(
                    content=ft.Text(f"ch{c}", size=t.BODY),
                    on_click=(lambda g=tgt, c_=c: lambda _e:
                              (session.retune(g, source=c_), rebuild_targets()))())
                    for c in session.specs] +
                ([ft.PopupMenuItem(
                    content=ft.Text("SUM (channels added, then measured)", size=t.BODY),
                    on_click=(lambda g=tgt: lambda _e:
                              (session.retune(g, source=PAIR), rebuild_targets()))()),
                  ft.PopupMenuItem(
                      content=ft.Text("Custom (a sign per channel)…", size=t.BODY),
                      on_click=(lambda g=tgt: lambda _e:
                                open_combination(g))())]
                 if session.aligner is not None else []))

            # **Two questions, one menu.** What this target does to the feedback — reward, inhibit or
            # monitor — and which way it is being trained. They were a single chip cycling increase →
            # decrease → monitor, which made "reward" mean "increase" and left no way to say the thing an
            # alpha-down arrangement needs: the band the session is *about*, trained downward, leading the
            # level rather than sharing the inhibits' 20%. See `targets.ROLES`.
            #
            # A menu rather than two chips, and the reason is the row: the identity row is the widest
            # thing on a fixed 430 px card, and two worded chips overran it and clipped the remove button
            # — the same failure the SUM label caused above. It is also the same idiom as the source menu
            # beside it, every combination is spelled out in words where the button is terse, and
            # **monitor appears once with no direction**, so the rule that a monitor is not trained in a
            # direction falls out of the menu rather than needing a control to be hidden.
            #
            # Colour carries the role and the arrow carries the direction. It was the other way round,
            # which is now the wrong way: what a card list is scanned for is what is driving the output.
            #
            # `reset_stats` on every change, because the comparison has changed for a different reason
            # each time — a new direction re-judges every value against the other side of the threshold,
            # and a new role re-aims the threshold itself.
            def set_kind(g, role, direction=None):
                g.role = role
                if direction is not None:
                    g.direction = direction
                g.reset_stats()
                rebuild_targets()

            role_colour = (t.MUTED if tgt.monitor else
                           t.ACCENT if tgt.role == "reward" else t.WARN)
            direction = ft.PopupMenuButton(
                content=ft.Container(
                    content=ft.Text(
                        tgt.role_label if tgt.monitor else
                        f"{tgt.role_label} {'↑' if tgt.direction == 'increase' else '↓'}",
                        size=t.SMALL, color=role_colour, weight=ft.FontWeight.W_600),
                    tooltip="what this target does to the feedback, and which way it is trained",
                    padding=ft.Padding(8, 4, 8, 4), border_radius=6,
                    bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
                # A monitor keeps whatever direction it had — it still has a threshold and an aim, and
                # `percent` is measured against one — so choosing it here sets the role alone, and
                # choosing a role back restores the direction rather than inventing one.
                items=[ft.PopupMenuItem(
                    content=ft.Text(f"{role.upper()}   {'↑ increase' if up else '↓ decrease'}"
                                    f"   ·   {MENU_NOTES[(role, up)]}", size=t.BODY),
                    on_click=(lambda g=tgt, r=role, d=("increase" if up else "decrease"):
                              lambda _e: set_kind(g, r, d))())
                    for role in ROLES if role != "monitor" for up in (True, False)]
                + [ft.PopupMenuItem(
                    content=ft.Text("MONITOR   ·   measured and drawn, drives nothing", size=t.BODY),
                    on_click=(lambda g=tgt: lambda _e: set_kind(g, "monitor"))())])

            identity = ft.Row(
                [ft.Container(width=3, height=18, bgcolor=tgt.colour, border_radius=2),
                 source_menu,
                 # Arrows slide the whole band and keep its width; typing sets that edge alone. See
                 # `Target.shift_band` — nudging a band is hunting for a frequency, not resizing it.
                 num_box(tgt.band.low,
                         (lambda g=tgt: lambda v:
                          (session.retune(g, low=v), rebuild_targets()))(),
                         step=0.1, low=0.1, high=tgt.band.high - 0.1,
                         on_step=(lambda g=tgt: lambda d: slide(g, d))()),
                 ft.Text("–", size=t.BODY, color=t.FAINT),
                 num_box(tgt.band.high,
                         (lambda g=tgt: lambda v:
                          (session.retune(g, high=v), rebuild_targets()))(),
                         suffix="Hz", step=0.1, low=tgt.band.low + 0.1,
                         on_step=(lambda g=tgt: lambda d: slide(g, d))()),
                 ft.Container(expand=True), direction, ft.Container(width=8),
                 tiny("×", (lambda g=tgt: lambda _e:
                            (session.remove_target(g), rebuild_targets()))())],
                spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER)

            card = TargetCard(tgt, identity, rebuild_targets, page=page,
                              modes=session.threshold_modes,
                              settings=(lambda g=tgt: lambda redraw: filter_rows(g, redraw))())
            cards[id(tgt)] = card
            rows.append(card.control)
        target_column.controls = rows
        signal.rebuild_contributions(session.targets)
        page.update()

    def add_target(_e=None) -> None:
        """Append a card and scroll to it. A menu of presets made you choose the band before you could
        see the card, and every field on the card can already change the band."""
        low, high = DEFAULTS.get(session.protocol.code, DEFAULT_TARGETS)[0][:2]
        # **The band is borrowed from the preset's first entry; nothing else is.** The source is
        # `session.focus`, the channel being looked at, rather than the preset's. A card
        # added by hand is a reward on the way up — the default everywhere — whatever the protocol opened
        # with. Inheriting them from entry zero would hand an alpha-down screen a second down-trained
        # reward, when what is almost always being added there is an inhibit.
        session.add_target(session.focus, Band(low, high, ""))
        rebuild_targets()

        async def reveal() -> None:
            # scroll_to is a coroutine and belongs on Flet's loop, like every other UI call here.
            # A short wait first, so the rebuilt list has been laid out and the end exists to scroll to.
            await asyncio.sleep(chrome.DRAW_SECONDS)
            try:
                await target_column.scroll_to(offset=-1, duration=250)
            except Exception:
                pass            # a scroll we cannot perform must not lose the card

        page.run_task(reveal)

    add_button = ft.Container(content=ft.Text("+ add", size=t.SMALL, color=t.ACCENT),
                              padding=ft.Padding(10, 5, 10, 5), border_radius=6,
                              bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE), on_click=add_target)


    # ---- tabs ----
    # The window pills live in the panel's own header rather than under the graph. Below the graph
    # they shared a row with the caption inside a column whose width the plot could exceed, and at the
    # default window size they were pushed off the pane entirely.
    train_left = ft.Column([
        panel("TARGETS", ft.Container(content=graph_legend, padding=ft.Padding(0, 0, 0, 8)),
              ft.Row([graph_axis,
                      ft.Stack([ft.Container(content=graph, expand=True),
                                ft.Container(content=graph_empty,
                                             alignment=ft.Alignment.CENTER, expand=True)],
                               expand=True, height=GRAPH_H)],
                     spacing=0),
              ft.Container(height=6), graph_caption,
              trailing=window_pills),
        ft.Container(height=12),
        panel("RAW", *traces.rows, ft.Container(height=6), traces.caption,
              trailing=gain_pills),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    # Above the graph rather than beside a card, and in the same place every other screen puts one: what
    # it says invalidates every target at once, so it cannot belong to any of them.
    warning = ft.Container(visible=False, bgcolor=t.SURFACE_HI, border_radius=8,
                           padding=ft.Padding(12, 9, 12, 9),
                           content=ft.Text("", size=t.SMALL, color=t.BAD))

    train_tab = ft.Container(
        content=ft.Column([warning, ft.Container(height=10),
                           ft.Row([train_left,
                                   ft.Column([panel("TRAINING",
                                                    ft.Container(content=target_column, expand=True),
                                                    expand=True, trailing=add_button)],
                                             spacing=0, width=TARGETS_W)],
                                  spacing=14, expand=True)],
                          spacing=0, expand=True),
        padding=ft.Padding(14, 12, 14, 14), expand=True)

    panes = [train_tab, spectrum.control, trends.control, signal.control, plan_pane_.control]

    def select_tab(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs(["Train", "Spectrum", "Trends", "Feedback", "Session"], select_tab)

    # ---- draw ----
    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        unit = "%" if session.relative else "µV"

        plan_pane_.refresh()
        signal.refresh()
        spectrum.refresh()
        # Sampling is on the feed, not here — see `TrainingSession.feed_block` — so the history covers
        # the session whether or not this tab is ever opened, and a replay collects what a live run does.
        trends.refresh()
        if not train_tab.visible:
            page.update()
            return              # a hidden pane must cost nothing; see SpectrumPane.refresh

        # Per frame, not once at build: the corner is read from what is streaming, so it arrives with the
        # first block and changes when the device screen is used mid-session.
        trouble = session.wiring_problem()
        warning.visible = trouble is not None
        if trouble:
            warning.content.value = trouble

        # target graph: the value, its moving average, and the threshold, over the chosen window
        span = session.graph_window
        pts_per_second = 2.0                       # trend is appended twice a second
        shapes = []
        w, h = plot_width(reserve=TARGETS_W + AXIS_W), float(GRAPH_H)
        any_data = False
        peak = 1e-9
        series = []
        for n, tgt in enumerate(session.targets):
            # **The window is asked for, not sliced off afterwards.** Averaging the whole trend and then
            # keeping a tenth of it is work that grows with the session and is thrown away, which is what
            # made this screen heavier the longer it ran — see `Target.moving_average`.
            keep = int(span * pts_per_second) if span else 0
            data = list(tgt.trend)[-keep:] if keep else list(tgt.trend)
            average = tgt.moving_average(int(MOVING_AVERAGE_SECONDS * pts_per_second),
                                         span=keep or None)
            # No more points than there are pixels to draw them on. A "session" window holds up to
            # TREND_POINTS, and every one of them is a point Flet serialises to the client each frame for
            # no visible gain — two points on one pixel is one pixel.
            if len(data) > w:
                take = np.linspace(0, len(data) - 1, int(w)).astype(int)
                data = [data[i] for i in take]
                average = [average[i] for i in take] if len(average) == len(take) else average[-int(w):]
            if len(data) >= 2:
                any_data = True
                peak = max(peak, max(data), tgt.threshold)
            series.append((tgt, data, average, tgt.colour))
        # The axis, and the ceiling it is drawn to: sticky, so the numbers hold still while the window's
        # own maximum wobbles — the same three rules the spectrum's scale needed. The traces render a
        # little under the top instead of always touching it, which is what makes the numbers mean
        # something: a plot normalised to its own maximum shows shape and hides size.
        ceiling = graph_ceiling.update(peak, time.monotonic())
        place_value_labels(graph_axis_labels, h, ceiling, unit=unit)
        if any_data:
            shapes.extend(grid(w, h))
            for tgt, data, average, colour in series:
                if len(data) < 2:
                    continue
                step = w / max(1, len(data) - 1)
                shapes.append(polyline(
                    [ft.Offset(i * step, h - (v / ceiling) * h) for i, v in enumerate(data)],
                    colour, 0.9))
                # the moving average, heavier than the instantaneous trace it smooths
                shapes.append(polyline(
                    [ft.Offset(i * step, h - (v / ceiling) * h) for i, v in enumerate(average)],
                    colour, 2.2))
                ty = h - (tgt.threshold / ceiling) * h
                shapes.append(polyline([ft.Offset(0, ty), ft.Offset(w, ty)], t.FAINT, 0.8))
        graph.shapes = shapes
        graph_empty.visible = not any_data
        window_name = next((n for n, s in GRAPH_WINDOWS if s == span), "session")
        graph_caption.value = (f"{window_name} across · thick line = {MOVING_AVERAGE_SECONDS:g} s "
                               f"moving average · grey = threshold · "
                               f"0–{ceiling:g} {unit} full scale")
        graph_legend.controls = [
            ft.Row([ft.Container(width=10, height=2, bgcolor=colour, border_radius=1),
                    ft.Text(f"{tgt.label} {tgt.role} {tgt.verb.lower()}", size=t.SMALL,
                            color=t.MUTED)],
                   spacing=6)
            for tgt, _data, _avg, colour in series]

        # raw traces at a fixed, stated gain — the shared rows, which also carry the click that overrules
        # the device about a channel. See `trace_rows`.
        traces.refresh(session.trace_seconds,
                       extra=(f"{1000 * session.trace_seconds / 10:.0f} ms/div",))

        # target rows
        for tgt in session.targets:
            card = cards.get(id(tgt))
            if card is not None:
                card.refresh(session.value_for(tgt), unit, session.response_time(tgt))

        page.update()

    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)

    build_pills()
    build_gain_pills()
    rebuild_targets()
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)
