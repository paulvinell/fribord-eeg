"""What every training session does, regardless of what it trains.

Extracted from `AmplitudeSession` when the second protocol arrived, and the split is the interesting
part. A protocol screen turns out to own surprisingly little:

| here | in the protocol |
|---|---|
| owning the source, trimming to the channel count | which metrics to build |
| artifact rejection | what the targets are |
| the transport, the clock, the plan and its cues | the Train tab's layout |
| the one feedback signal, its smoothing, the click's phase | the wiring it requires |
| driving the shadow, the system volume and the music | |
| the rate discipline on `Target.observe` | |

The rate discipline is the one worth pointing at, because it was a real bug and it is invisible: a
target's declared rate sizes *every* window it has — calibration, the automatic tracker, "last 30 s on
target" — so a target observed once per arriving block runs at blocks × channels per second, and a 30 s
calibration finishes in five. `_due` is what keeps the declared rate and the actual rate the same, and it
belongs here rather than in a screen, because getting it wrong is silent in both.

Subclasses implement `consume(block)` — feed your metrics, return the keys that produced a fresh reading
— plus `value_for(target)` and `key_for(target)`. Everything else follows.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

import optima

from nfb import Device, Mode, ReplaySource, WaitingSource
from nfb.dsp import (ArtifactGates, Band, BandPower, CFIRBandPower, DRIFT_CORNER_MAX_HZ,
                     ELLIPTIC_RIPPLE_DB, highpass, lowpass)
from nfb.dsp import artifact as dsp_artifact
from nfb.stream import Block, Gap

from . import cues, music, prefs, quality, sentry, shadow, sysvolume
from .feedback import CONDITION, HOLD_FRACTION, LEVEL, STATE, FeedbackSettings
from .plan import SessionPlan
from .protocols import Protocol
from .spectrum_pane import SpectrumSource
from .targets import MODES, PAIR, Target

METRIC_HZ = 20.0            # what the estimators are asked to emit at
TREND_SECONDS = 0.5         # how often a target's plotted trend gains a point


def true_metric_hz(rate_hz: float) -> float:
    """The rate readings actually arrive at, which is not quite `METRIC_HZ`.

    Every estimator decimates by an integer stride, `round(rate / METRIC_HZ)` — 12 at 250 sps — so asking
    for 20 Hz yields 250/12 = 20.83. Targets were told the asked-for 20.0, which made every window sized
    from it 4% short: the 30 s calibration was 28.8 s, the "last 30 s on target" 28.8. Small, but every
    number on screen claimed a length it did not have; the target is told the rate its readings really
    come at instead.
    """
    return rate_hz / max(1, round(rate_hz / METRIC_HZ)) if rate_hz > 0 else METRIC_HZ

# Set NFB_DEBUG=1 to have a running session write what it is actually receiving to this file.
#
# Here for the same reason `device_panel` has its own log, and learned the same way: a live symptom on the
# screen can be invisible to every test in the repo, because the tests feed a recording through the same
# objects and the objects are fine. A frozen trace was diagnosed wrongly three times from the outside —
# as electrode contact, as an epoch ending the feed, as draw cost — while nothing recorded what the
# session was being handed. Guessing is what this file exists to replace.
SESSION_LOG = "/tmp/nfb-session.log"


def slog(msg: str) -> None:
    if not os.environ.get("NFB_DEBUG"):
        return
    try:
        with open(SESSION_LOG, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} {msg}\n")
    except OSError:
        pass                    # diagnostics must never take a session down

# What measures a target, per role, and the two differ because the jobs do.
#
# A **reward** drives moment-to-moment reinforcement, and the one dose-response study we have is blunt
# about the budget: 244 ms end to end worked, 494 ms was much weaker, 744 ms produced no sustained change
# at all (Belinskaia et al. 2020). cFIR is the best thing we have there — measured on a 12-15 Hz band at
# 250 sps it is ~120 ms against an elliptic at 99 ms and a Butterworth order 2 at 150 — so cFIR is
# not the fastest of the three, and what it buys over the fastest is a flat group delay across the band and
# no envelope smoother at all, since it estimates the amplitude directly.
#
# An **inhibit** is a veto. Noticing an excursion 300 ms late costs almost nothing, since the excursion
# outlasts that and the consequence is an alert rather than a moment-locked reinforcement. What it needs
# instead is separation from its neighbours, or adjacent bins are not really separate bins.
#
# The elliptic design is 2 biquads, 0.5 dB passband ripple, 60 dB stopband.
#
# **Keyed by role, not by direction.** Everything above is an argument about what the estimate is *used
# for* — a reward is moment-locked reinforcement and wants the delay short, an inhibit is a veto and wants
# the separation — and neither sentence mentions which way the band is being trained. A reward trained
# downward is still moment-locked. A monitor drives nothing, and takes the reward's settings because it is
# usually a band somebody is deciding whether to promote.
# **The reward default is the elliptic, and it changed because the delay argument above was answered
# with a bound rather than a measurement.** cFIR's whole case is having no envelope smoother — but that is
# also what its reading feels like. Measured on a real 11.7–14.7 Hz session, both estimators on the same
# signal with thresholds derived at the same 90% aim: cFIR crosses its threshold **172 times a minute** and
# moves 32% of its own median between updates 50 ms apart, where the elliptic at order 2 crosses **31**
# times and moves 7%. Three times a second is not feedback about a rhythm, it is chatter around a line.
#
# What that steadiness costs is 48 ms — **168 ms against 216** to report half of a burst at the band centre,
# measured through both shipping estimators. The 450 ms this decision was previously argued against is
# `latency_seconds`, which adds the smoother's whole time constant to the group delay as a bound; the
# response is not the bound, and picking the default off the bound was picking it off the wrong number.
#
# Reported as a session that "felt different on the elliptic filter", which it was.
DEFAULT_FILTER = {"reward": "elliptic", "inhibit": "cfir", "monitor": "elliptic"}
CFIR_DELAY = {"reward": 0.1, "inhibit": 0.3, "monitor": 0.1}
DEFAULT_ORDER = 2                # the conventional realtime choice; two biquads
DEFAULT_RIPPLE = ELLIPTIC_RIPPLE_DB
"""dB, elliptic only. Taken from `nfb.dsp` rather than restated, because it was restated and the two copies
disagreed the moment one of them was corrected."""

# **A channel is condemned by its samples, never by the amplifier's word alone**, and getting to that took
# three tries. See `_judge_channel`, which is the whole rule, and `_looks_open`, which is the evidence.
#
# There used to be a `SIGNAL_GRACE_SECONDS = 30.0` here: how long a channel could report "no signal" while
# its samples said otherwise before it was treated as dead. It is gone, and what it was really measuring is
# worth keeping.
#
# **A per-block flag was being used as a per-channel verdict, and one frame in a thousand killed the
# session.** `Block.valid` describes the block it came on; `state` describes the channel. The feed assigned
# one to the other, so a single frame with the device's no-signal bit set flipped the channel to dead until
# the next good frame — and on this hardware that happens on working channels. The measurement in
# `Block.state`'s own docstring says so: a known-good electrode at C4 delivered 19960 samples at state 0
# **and one frame at state 2**. What that looked like is exactly what was reported: the raw trace fine on
# both channels, contact fine on both (14/15 and 14/26 kOhm), and no feedback at all — because every target
# on the infra-low screen is measured on the *pair*, `target_is_live` requires both channels, and one
# channel flickering to invalid takes the whole reward down with it.
#
# The grace period fixed the flicker and left the sustained case, which is the one that was then reported:
# good impedance, the 0.5 Hz corner, a signal that looks fine on the trace, and the channel dead anyway. The
# timer only ever ran *down*. Nothing in the code could acquit a channel — `_looks_open` was consulted in
# one branch, where returning True condemned it immediately, and there was no branch where healthy samples
# rescued one. So once the amplifier asserted its bit for half a minute the channel stayed dead until the
# amplifier itself changed its mind, and it has no obligation to: its own wording is "signal far below range
# **for a long time**", 800 µV is the narrowest range a voltage channel offers, and 20 µV of EEG sits at
# 2.5% of full scale. A conservative below-range detector latching on a good, quiet channel is not a fault
# in the device. Treating its verdict as unappealable was ours.
#
# A duration was never the right instrument for that. "How long has the device been complaining" is not
# evidence about the electrode; "what do the samples look like" is, and it is available on every block.

DEFAULT_CORNER_HZ = 0.5
"""The corner assumed when the source cannot say — a text export, which carries no device config. The
the usual EEG choice and the driver's default, so it is the right guess; `corner_hz` reads the real one
from the config blob whenever there is one, which is every live session and every frame recording."""


def corner_of(source, specs=None) -> float:
    """The device high-pass corner behind a source, in Hz.

    A function rather than a method because two unrelated session classes need the same answer — the
    training sessions here and the launcher's readout in `session_view`, which also puts a cFIR on live
    samples and would otherwise print an alpha figure made of drift. A second copy of this lookup is how
    the two would come to disagree.

    Per-channel because the field is, and the worst channel wins: the corner that spoils a measurement is
    the one worth acting on, not the friendliest one present.
    """
    cfg = getattr(source, "config", None)
    if not cfg:
        return DEFAULT_CORNER_HZ
    corners = []
    for channel in (specs or {0: None}):
        try:
            corners.append(optima.lower_limit_hz(cfg, channel))
        except ValueError:
            continue                # a function with no corner in the table; not a voltage channel
    return max(corners) if corners else DEFAULT_CORNER_HZ


def drift_for(corner_hz: float) -> float | None:
    """What a cFIR fed by a source at this corner must high-pass for itself, or None.

    A cFIR leaks ~6% of everything outside its band, DC included, so an electrode's standing offset lands
    inside every estimate: measured, a 2.11 µV band read as **12.30 µV** at the 0.01 Hz corner and 2.13 at
    0.5. One of the two cases needs an answer, which is why this is asked rather than assumed — and why
    the IIR kinds never carried it, a real band-pass rejecting what a least-squares fit only attenuates.
    """
    return None if corner_hz >= dsp_artifact.JUDGE_FLOOR_HZ else DRIFT_CORNER_MAX_HZ


class LevelRange:
    """Where the level sits and how far it swings, measured from this session and then held.

    **The level's own calibration, and it is the same idea as a threshold's.** A threshold is derived from
    the first thirty seconds and then holds still, because a number that keeps re-deriving itself pins the
    quantity it is measuring. The level had no equivalent: it was handed to outputs with fixed ends without
    anyone asking what range it actually occupies, and the answer measured on a real session is 0.58 to 0.81
    — a quarter of the range it was being mapped through, on top of a mechanism that only sweeps 30 to 70 of
    the system volume. Three compressions in a row, and no control addressed any of them.

    **Held against re-centring, not held rigid — the ends ratchet outward.** A rolling stretch would
    re-normalise the trainee every window, so a genuinely better state would fade back to the middle within
    a minute — the trap that `threshold_modes` describes for automatic thresholds, arrived at from the other
    end. Once this settles, doing better than you did during calibration reads as *more* output, and stays
    there. But held *rigid*, the design has a wall in it: `shape` runs out of travel half a span above the
    calibrated high, and a training session is built to push the level past its own first minute. Measured
    on `sessions/20260812` with the elliptic reward default, the shaped level sat hard at 1.0 for 6.5% of a
    flat resting replay — and a live session that strengthened past its first minute pinned volume and
    brightness at their ceilings for good, which the operator experienced as no feedback at all. (The old
    cFIR default never showed this, not because the map was sound but because its chatter only visited high
    scores in ~100 ms flickers that the level smoothing averaged away; the steadier reading holds them for
    seconds, and they sail through.) So when the level *lives* beyond an end, that end moves out to meet it
    over `RATCHET_SECONDS` — outward only, never back in, never re-centred — which keeps both promises: a
    better state still reads high, and the top of the output stays a gradient instead of becoming a wall.

    Provisional from the first values and refined until it settles, because nothing here waits for a number
    to settle before it feeds anything back. Before there is anything to
    measure, `span` is zero and `FeedbackSettings.shape` is the identity.
    """

    WINDOW_SECONDS = 30.0
    """How much history the middle and the spread are read from. The thresholds' own calibration window."""

    SETTLE_SECONDS = 60.0
    """When it stops moving — **deliberately not 30.** The level is built from scores against thresholds,
    and those are provisional for their own first thirty seconds, so a level range derived from that period
    is measuring the thresholds settling as much as the person. At 60 s the window holds 30–60 s: after the
    thresholds have landed, and still inside the first minute."""

    MINIMUM = 20
    """Values before a range can be derived at all — one second at the metric rate."""

    FLOOR = 0.05
    """The narrowest swing that will be believed, as a fraction of the level's 0…1 range.

    Without it, a calibration that happens to catch a still minute divides by nearly nothing and the
    session that follows spends itself pinned against both ends alternately. Errs towards a calmer output,
    which is the right way round: too little movement is the complaint that brought this about, and it is
    visible in a minute, where an output slamming between its ends is unusable immediately.
    """

    RATCHET_SECONDS = 10.0
    """How quickly an end chases a level living beyond it, once settled — the time constant of an
    exponential approach. Slow enough that a 100 ms flicker past the end moves it by a tenth of a percent
    of the excess and the map stays a calibration rather than a follower; fast enough that a level parked
    against the ceiling has its gradient back within half a minute rather than for the rest of the hour."""

    def __init__(self) -> None:
        self._values: deque[tuple[float, float]] = deque()
        self.centre = 0.5
        self.span = 0.0
        self.settled = False
        self._first: float | None = None
        self._derived_at: float | None = None
        self._last_now: float | None = None

    def observe(self, level: float, now: float | None) -> None:
        """One reading of the unshaped level, at `now` in **signal time** — seconds of samples, so a
        replay reproduces a live session exactly, which is the same clock `_advance_level` runs on."""
        if now is None:
            return
        prev, self._last_now = self._last_now, now
        if self.settled:
            self._ratchet(float(level), prev, now)
            return
        if self._first is None:
            self._first = now
        self._values.append((now, float(level)))
        while self._values and self._values[0][0] < now - self.WINDOW_SECONDS:
            self._values.popleft()
        done = now - self._first >= self.SETTLE_SECONDS
        # Rate-limited for the reason `Target.observe` is: this is two percentiles over a 30 s window, and
        # a window that long cannot meaningfully move in 50 ms. The final derivation is not rate-limited,
        # so settling happens exactly when the window is full.
        if not done and self._derived_at is not None and now - self._derived_at < 1.0:
            return
        if len(self._values) < self.MINIMUM:
            return
        self._derived_at = now
        values = np.fromiter((v for _t, v in self._values), dtype=float, count=len(self._values))
        low, high = np.percentile(values, [5, 95])
        self.centre = float((low + high) / 2.0)
        self.span = max(self.FLOOR, float(high - low))
        self.settled = done

    def _ratchet(self, level: float, prev: float | None, now: float) -> None:
        """Move an end of the travel out to meet a level living beyond it. See the class docstring.

        The ends here are `centre ± span` — where `shape` runs out of travel at a sensitivity of 1 — not
        the measured 5th/95th percentiles, so nothing happens while the level is merely in the outer
        quarter of the output: only what the map cannot express moves it. Exponential in signal time, so
        how far an end moves depends on how long the level has been outside, not on how often anyone asked.
        """
        dt = now - prev if prev is not None else 0.0
        if dt <= 0:
            return
        top, bottom = self.centre + self.span, self.centre - self.span
        gain = 1.0 - math.exp(-dt / self.RATCHET_SECONDS)
        if level > top:
            top += gain * (level - top)
        elif level < bottom:
            bottom += gain * (level - bottom)
        else:
            return
        self.centre, self.span = (top + bottom) / 2.0, (top - bottom) / 2.0

    def remaining(self, now: float | None) -> float:
        """Seconds until it holds still, for the panel to count down. Zero once settled."""
        if self.settled or self._first is None or now is None:
            return 0.0 if self.settled else self.SETTLE_SECONDS
        return max(0.0, self.SETTLE_SECONDS - (now - self._first))


class TrainingSession:
    """The device, the clock, the targets and the one signal. Subclass for a protocol."""

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0):
        self.protocol = protocol
        self.device = device
        self.plan = SessionPlan(length_seconds)
        self.cue = cues.play          # swappable, so tests do not make noise
        self.display = shadow.DisplayShadow()
        self.output_volume = sysvolume.SystemVolume()
        self.music = music.MusicOutput()
        if source is not None:
            self.source = source          # already-built file source; see inputs.InputChoice
        elif device is not None and not device.status.connected:
            # **A protocol screen opens without an amplifier**, waiting. The channel layout comes from the
            # settings rather than from the device, so every trace and card draws "waiting for data" — and
            # the red chip beside them says why, which it did anyway. Refusing to open the screen was the
            # first fix and it was the wrong one: it took the screen away in exchange for information
            # already on it. Connecting from the chip and coming back builds the session again, live.
            self.source = WaitingSource(device.status.settings.build())
        elif device is not None:
            # **A session needs the device measuring, and must not assume it is idle.** Impedance testing
            # is an exclusive mode, so leaving the device screen into a protocol with the contact check
            # still running left the device in `TESTING` — and this silently subscribed to the *impedance*
            # stream. Reproduced: the session came up with a single channel at 3.90625 Hz, twenty-two of
            # three hundred and twenty trace columns filled in six seconds, and not one target ever fed.
            # On screen that is a flat trace with an occasional spike and no feedback, which is exactly how
            # it was reported — and `narrow_to_wired` made it look ordinary by leaving one channel enabled.
            if device.status.mode is Mode.TESTING:
                device.stop()
            if device.status.mode is Mode.IDLE:
                device.start_measurement()
            self.source = device.subscribe(protocol.code.lower())
        else:
            # loop by default: a 20 s capture has to stand in for a device indefinitely
            self.source = ReplaySource(path, realtime=True, loop=True)
        # Before trimming, because *which* specs arrived and at what rate is the thing a frozen trace
        # cannot tell you from the outside. A session subscribed while the device was still in impedance
        # mode, for instance, would report 3.9 Hz specs here and nothing on screen would say so.
        slog(f"session {protocol.code} source={type(self.source).__name__} "
             f"device_mode={device.status.mode.value if device is not None else '-'} "
             f"specs={{{', '.join(f'{c}: {s.rate_hz:g}Hz/dec{s.decimation}' for c, s in self.source.channels().items())}}} "
             f"asked_channels={channels}")
        self.specs = self.source.channels()
        if channels:
            # Use only the first N. The device may still be streaming four — a recording certainly is —
            # and showing traces and cards for inputs nobody wired up is worse than useless, because a
            # flat line is indistinguishable from a working channel on a quiet subject.
            self.specs = {i: spec for i, spec in list(self.specs.items())[:channels]}
        self.rate = next(iter(self.specs.values())).rate_hz if self.specs else 250.0

        self.gates = ArtifactGates()
        # Duration-watching quality alarms, on every session; only a screen that arms `audible` beeps.
        self.sentry = sentry.SignalSentry()
        # Where the session went bad, second by second, so "when" is answerable after the fact.
        self.quality = quality.QualityLog()
        self.band_trends = None
        """Where each band has been, kept on the session because that is what it is a history *of*.

        Filled by `trends_pane.TrendsPane`, which owns its shape — the bands and the cadence are the
        screen's choice. It lives here because the view is rebuilt on every return from the device
        screen and a history rebuilt with it is a history lost.
        """
        self._display_filters: dict[int, object] = {}
        self._response_cache: dict[tuple, float] = {}
        self.state: dict[int, bool | None] = dict.fromkeys(self.specs, None)
        # What the blocks amount to, kept apart from `state` so an operator override can win without the
        # next block overwriting it — see `trust_channel`, and `_judge_channel` for how it is decided.
        self._device_says: dict[int, bool] = {}
        self.trusted: dict[int, bool] = {}
        # The standing offset and the closest approach to the rail, per channel — see `_watch_offset`.
        self._offset_uv: dict[int, float] = {}
        self._headroom_uv: dict[int, float] = {}
        """Per channel: True usable, False the device says no signal, **None not heard from yet**.

        Three states rather than two, because it started as `False` for every channel and that made *not
        knowing yet* indistinguishable from *known bad* — a screen said "no signal, check the electrode"
        before a single block had arrived. It also contradicted `target_is_live`, whose whole rule is that
        exclusion needs evidence: with every key present and False, every target was excluded from the
        feedback until its first block, rather than the documented "an unknown channel counts as live".
        """
        self.targets: list[Target] = []
        self.amplitudes: dict[tuple, float] = {}
        self.feedback = FeedbackSettings()
        # A mechanism this protocol does not offer is *off*, not merely unlisted. Relying on the default
        # would leave it running with no control on screen the moment a default changed or a subclass
        # enabled it — which is the swallowed-refusal shape: invisible on screen, real in the output.
        for name, mech in self.feedback.mechanisms.items():
            if name not in self.offers:
                mech.enabled = False
        self.gain_uv = 100.0
        self.focus = next(iter(self.specs), 0)

        self._accumulated = 0.0
        self._resumed_at: float | None = None
        self.paused = True
        self.finished = False
        self.gaps = 0
        # Counted for the feed log, and initialised here rather than in `run` because `feed_block` is
        # callable without it — that is the whole point of the split.
        self._seen: dict[int, int] = {}
        self._states: dict[tuple[int, int], int] = {}
        self._skipped = 0
        self.feeding = False
        """Whether a feed thread is currently consuming the source. False for a session that has been
        suspended for a trip to the device screen, which is what tells the app to re-arm rather than build a
        new one."""
        self.error: str | None = None
        self._last_trend = 0.0
        self.stream_seconds = 0.0
        """Where the signal is, in seconds from its first sample — the clock a replay keeps. Anything
        sampled "every N seconds of session" samples on this, not on `elapsed`, or a replay fed faster
        than real time collects nothing: the band trends read "waiting for data" on a screen that had
        been fed two and a half minutes."""
        self._level = 0.0
        self._level_t: float | None = None    # signal time of the last level advance
        # What the level's own range turns out to be, so `sensitivity` has something to be a gain *of*.
        # Session history, like a threshold's calibration, so it is not carried across a stop by `adopt`.
        self.level_range = LevelRange()
        # The output pump: run() starts it, _drive_outputs feeds it, stop() joins it. None means "apply
        # inline", which is what unit tests that drive the internals directly get.
        self._output_state: tuple | None = None
        self._output_wake = threading.Event()
        self._output_thread: threading.Thread | None = None
        self._last_click: float | None = None
        # (time, met) over the last hold window — a *duty cycle*, not a run length. See `HOLD_FRACTION`:
        # the raw condition flickers at about 3 Hz, so a continuity test can never pass.
        self._condition_log: deque[tuple[float, bool]] = deque()
        self._alerting: set[int] = set()
        self._last_alert: dict[int, float] = {}
        self._stop = False
        # Set by `view()` to its own `draw`. The session owns the hook rather than the view keeping it
        # private, so the draw loop, the transport and a headless test all reach the same function — and
        # a screen with no way to be redrawn from outside cannot be tested at all.
        self.redraw: Callable[[], None] = lambda: None

    # --- what a protocol supplies ----------------------------------------------------------

    def consume(self, block) -> set:
        """Feed this block to the protocol's metrics; return the keys that produced a reading."""
        raise NotImplementedError

    def value_for(self, tgt: Target) -> float:
        raise NotImplementedError

    def key_for(self, tgt: Target):
        """The metric key whose freshness this target follows."""
        raise NotImplementedError

    music_sources: tuple[str, ...] = ("noise", "file", "eeg")
    """Which music sources this protocol can use, and therefore which the Feedback tab offers.

    Not every source is meaningful everywhere, and a control that makes no sense on the screen it is on is
    worse than a missing one — it invites an operator to choose something that cannot work and then leaves
    them to discover it. `crossfade` is the case: it needs two ends that stand for two *states* and a
    protocol that knows which state you are in, so a screen with no `state_balance` would render it as music
    frozen at one end. The traffic goes the other way too — a protocol whose whole continuous channel is a
    fade between two states has no use for a single fixed tilt or a backing track, so alpha-theta narrows
    this to `crossfade` alone rather than adding to the pile.
    """

    normalisations: tuple[str, ...] = ("sd", "threshold", "absolute")
    """Which `scale from` rules this protocol offers. `absolute` is a fixed step **in microvolts**, so a
    screen whose trained quantity is not in microvolts — the coupling screen's readings live between 0 and
    5 percent — drops it: a 5 µV step on that signal is a score that never moves, which is a control that
    cannot work on the screen it is on."""

    pinned_feedback: frozenset[str] = frozenset()
    """Which feedback settings this protocol *owns*, so the Feedback tab shows them locked.

    **The generic Feedback tab was the wrong tenant on a specialty screen.** It offered every choice on
    every protocol, including the ones the protocol had already made and depends on: switching
    cross-frequency coupling's level rule from `scaled` to `weighted` breaks the interlock between its
    bins and its rewards, and nothing said so. That is the same disease `music_sources` was written for,
    treated in one place and left everywhere else.

    The cut is **by ownership, not by screen**, which is what makes it survive the next protocol:

    - **conditioning** — artifact rejection, band averaging, the display corner. Step 0 and step 1 of
      turning microvolts into a number, nobody's protocol, adjustable everywhere.
    - **calibration** — volumes, floors and ceilings, intervals, which screen a shadow is on. About the
      person and the room, wrong differently for everyone, adjustable everywhere including here.
    - **arrangement** — which mechanisms exist, what each reads, how the targets combine. Protocol
      knowledge. Named here, rendered inert with its reason beside it.

    Names understood by the pane: `combine`, and one per mechanism (`sound`, `shadow`, `click`, `alert`,
    `music`), which locks *what that mechanism is* — on or off, its direction, what it is drawn by,
    whether the music is gated — and never its calibration.

    **Locked is not hidden.** Every pinned control still shows what was chosen and that alternatives
    exist, because a screen that feeds something back without saying what is the failure these
    conventions keep catching. Only pin what the protocol actually sets: pinning a default it never
    decided freezes our choice and calls it the protocol's.
    """

    combines: tuple[str, ...] = ("weighted", "weakest", "scaled")
    """Which level rules this protocol offers. **`either` is not in the default, and that is the point.**

    It means the rewards are *alternatives* — two states you can be in one of — which is a fact about the
    protocol and not a preference: on a screen training alpha up and theta down, "whichever reward is doing
    best" is not a different way of combining them, it is a claim about them that is false. Offering it
    everywhere is the same disease as offering Quiet Reward on the amplitude screen.
    It is also what made `condition from` look inapplicable there: with `either` reachable, the condition
    had two forms and neither was pressable; with it gone, the condition has exactly one form per protocol
    and is stated rather than offered.
    """

    offers: tuple[str, ...] = ("sound", "shadow", "click", "music")
    """Which mechanisms this protocol offers at all. **The alert is not in the default.**

    It fires when a target crosses to the wrong side of its threshold, and whether that is an event or a
    nuisance depends entirely on how often the threshold is crossed — which is a property of the
    *protocol's* aims, not of the mechanism. Cross-frequency coupling budgets 5-15% of the session for
    hearing something and derives every threshold from that, so an alert there is the exception it is
    meant to be. On the amplitude screen an inhibit is aimed at 75% met, so it is over the line a quarter
    of the time and the alert fires about as often as the interval gate allows, all session. That is not a
    configuration mistake to be tuned out; the mechanism only means something where the thresholds were
    chosen with it in mind.

    So it is offered where a protocol says so, and a protocol that wants it has to have the hit rates to
    match. Same shape as `music_sources` and for the same reason: a control that cannot work on the screen
    it is on is worse than a missing one.
    """

    arrangements: tuple[str, ...] = ()
    """Which of the named arrangements this protocol offers, if any — see `FEEDBACK_MODELS`.

    Empty means the panel is absent rather than empty, because the three are not universal: they are
    specific to the protocols that define them, and Quiet Reward on an amplitude screen has no meaning. Where they *are* offered, they own the mechanism enables, so
    the reason beside a locked control names the arrangement rather than the protocol.
    """

    state_ends: tuple[str, str] | None = None
    """What the two ends of a crossfade *mean*, low end first, matching `Music.colour` and
    `second_colour`. Protocol knowledge, like `state_balance` — it is what lets the shared pane label the two
    colour rows with the states they stand for instead of "first" and "second"."""

    state_sensitivity_range: tuple[float, float] | None = None
    """The bounds this protocol allows on `state_sensitivity`, or None for no row at all.

    **A hook rather than a universal control**, the same shape as `state_ends` and `music_sources`. A state
    that arrives already bounded 0…1 by construction — the heart screen's coherence, HEG's normalised
    signal, alpha-theta's ratio — has nothing a gain would fix: it already uses its range, and a dial there
    would be a second way to say what the mechanism's own range says. It is a real question exactly where
    the state comes from a *measurement in microvolts* whose size is the trainee's, which is ILF Synchrony
    and so far only ILF Synchrony.

    A range rather than a list of offered values, because this is a continuous quantity and a ladder of
    pills would be somebody's guess at which four points on it matter. 1.4 is a legitimate answer to "a bit
    more than it does now" and no ladder short enough to fit a panel head contains it.
    """

    state_sensitivity: float = 1.0
    """How hard the state drives whatever follows it. 1.0 is the protocol's own calibration."""

    def set_state_sensitivity(self, value: float) -> None:
        low, high = self.state_sensitivity_range or (0.125, 8.0)
        self.state_sensitivity = min(high, max(low, float(value)))

    @property
    def state_full_uv(self) -> float | None:
        """What ±full-scale of the state works out to in the signal's own unit, where that can be said.

        None here: only a protocol whose state is a measurement has such a number, and it is what lets the
        sensitivity row print a consequence rather than a multiplier."""
        return None

    @property
    def state_balance(self) -> float | None:
        """Where a crossfade should sit, 0…1, or None to leave it alone.

        Here rather than in `feedback` because it is not the level: the two ends of a crossfade are two
        *states*, and which state you are in is a different question from how well you are doing. Only a
        protocol knows what the ends are, so only a protocol can answer it.
        """
        return None

    def adopt(self, previous: "TrainingSession") -> None:
        """Take over everything the operator *configured* on the session being replaced.

        **Stopping ends a session; it does not undo an afternoon's setting up.** Stop re-enters the
        protocol on a fresh session, which is right — but a fresh one was built from the protocol's
        defaults, so training SMR and pressing stop put the band back to 8-12 Hz, and with it every
        volume, every mechanism, the arrangement, the session length and the artifact policy. That is not
        a fresh session, it is a factory reset wearing one.

        The line is the same one the suspend/rearm work drew: **the session is the history, the
        configuration is not.** So the targets come across as objects — bands, directions, thresholds,
        modes, filters — with their *statistics* cleared, because hits and trends belong to the session
        that ended. Feedback settings, the plan and the artifact policy come across whole, since none of
        them is a measurement.

        Only from a session of the same protocol, which `Console.pick` enforces: adopting CFC's twelve bins
        into an amplitude screen would be nonsense.
        """
        if previous is self or previous.protocol.code != self.protocol.code:
            return
        self.feedback = previous.feedback
        self.plan = previous.plan
        self.plan.rewind()          # the new session's cues are due again from its own zero
        self.gates.policy = previous.gates.policy
        self.adopt_view(previous)
        self.adopt_targets(previous)

    def adopt_targets(self, previous: "TrainingSession") -> None:
        """Carry the target set across, statistics cleared. Overridden where the targets are not a list an
        operator edits freely — see `CFCSession`, whose bins are the protocol's rather than anyone's.

        The estimators belong to the new session and are keyed by band and filter, so every adopted target
        has to build its own: one carried across with no metric behind it would read zero for ever.
        """
        self.targets = previous.targets
        for tgt in self.targets:
            tgt.reset_stats()
            self.retune(tgt)

    def adopt_view(self, previous: "TrainingSession") -> None:
        """Scales, windows and which channel is in focus: settings that live on the session because a
        screen is rebuilt from it. Overridden where a protocol has more of them."""
        for attr in ("relative", "spectrum_scale", "gain_uv", "trace_seconds", "graph_window",
                     "focus", "unbiased"):
            if hasattr(previous, attr) and hasattr(self, attr):
                setattr(self, attr, getattr(previous, attr))

    conditions_signal: bool = True
    """Whether this session gates a *sample* stream, which is what `gates` is. False on the HRV screen: an
    RR interval is not a sample, so there is no per-block amplitude to reject and the Feedback tab's
    ARTIFACTS panel has nothing to show."""

    @property
    def trained_floor_hz(self) -> float:
        """The lowest frequency any measurement on this screen depends on.

        **Only the artifact panel reads it, and only to admit what the gate cannot see.** The gate judges
        excursions on 0.5–50 Hz and `dsp.artifact.JUDGE_FLOOR_HZ` explains at length why that floor has to
        exist — following the amplifier down to its 0.01 Hz corner made electrode drift set the threshold
        and rejected 78–91% of every block. It is the right floor and it is also a real blind spot: an
        infra-low screen trains *entirely underneath it*, so drift, sweat and a settling electrode — the
        artifacts that actually spoil that work — reach the reward untouched while the panel says rejection
        is on.

        Saying so is not the fix and should not be mistaken for one; the fix is a detector for the slow
        band, which does not exist yet. But an operator who can read "rejecting 2% of the session" beside a
        0.05 Hz reward and cannot read "none of it from the band you are training" has been told something
        that is true and understands the opposite.

        Derived from what the targets are measured over, so it needs no protocol to declare anything. The
        two screens whose targets carry no band at all — the waveform screen and SCP, which are lowpasses
        from DC and are the extreme case — override it.
        """
        lows = [tgt.band.low for tgt in self.targets if getattr(tgt, "band", None) is not None]
        return min(lows) if lows else dsp_artifact.JUDGE_FLOOR_HZ

    sustains_reward: bool = False
    """Whether this protocol's reward has a *sustained* form distinct from a momentary one — the Click panel's
    "doubles after" row.

    The mechanism is generic: `_maybe_click` doubles the mark once the condition has held for
    `HOLD_FRACTION` of `Click.hold_seconds`, and that arithmetic works on any protocol's condition. The
    **design** is not generic. Both the criterion and the three seconds belong to the alpha-theta
    protocol, which has a momentary reward event and a three-second sustained one beside it, and
    `at_view` is the only screen that sets it. Everywhere else the row sat in the Click panel defaulted to 0,
    captioned "never", looking like a core parameter of a mechanism every protocol has.

    Same rule as `Shadow.follows`, which is only offered where a protocol *has* a state: a control that
    nothing on this screen gives a meaning to should not be on this screen. Where a protocol has
    no sustained reward, doubling the mark would invent a second reward event.

    **Off means off, not merely unoffered.** `restore_calibration` zeroes `Click.hold_seconds` on a screen
    that leaves this unset, because hiding the row while the arithmetic still runs is worse than either
    showing it or removing it: the mark would change character with nothing on screen saying why."""

    hold_choices: dict | None = None
    """How often the measured value may change, as `{seconds: label}`, or None where a protocol has no such
    choice. Read by the SIGNAL panel; the protocol owns both the numbers and the words.

    A hook rather than a setting because only the protocol knows whether its reference holds its value at
    all, and what the alternatives *mean*: on the infra-low screen one option is a 2 s
    sample-and-hold and the other is our continuous feed, and those are not two speeds of one idea — they
    change how many values the threshold window contains and therefore what the threshold is.

    A screen offering this also has to honour `hold_seconds`."""

    hold_seconds: float = 0.0
    """The sample-and-hold period in force, 0 for none. Only meaningful where `hold_choices` is set."""

    tunes_envelope: bool = False
    """Whether `FeedbackSettings.envelope_seconds` actually reaches this session's estimators.

    True only where the band estimator is built from `filter_for` — the amplitude and alpha-theta screens,
    which are also the two that implement `retune_filter`. **Everywhere else the row was worse than
    useless: it crashed.** `SignalPane._set_field` calls `retune_filter` when the envelope constant changes,
    because that constant is part of an estimator's identity, and a screen built on `LockinBandPower` has no
    such method — so touching "band averaging" on the infra-low screen raised `AttributeError` out of a Flet
    event handler, which surfaces as a traceback in the terminal and nothing at all on screen.

    It should not have been offered there even if it had not crashed, and the reason is the one
    `scores_bands` gives below: the lock-in screens average their bands over their *own* post-filter, which
    has its own control with its own number on it. Two controls for one quantity, one of them
    inert, is how a panel stops being trustworthy.

    Defaults False so a new protocol has to claim it rather than inherit a crash."""

    scores_bands: bool = True
    """Whether the level is built by scoring targets against thresholds — steps 1 to 4 of the SIGNAL panel.

    False on the HRV screen, where the level comes from a metric and a normalisation instead. The four steps
    are not merely uninteresting there, they are inert: there is no band to average, the scaling already
    mapped the metric onto 0…1 so scoring it against a spread would be a second normalisation, and with one
    target every combining rule is the same rule. Only the output smoothing survives, and it applies to any
    level however it was arrived at."""

    level_source: tuple[str, str] | None = None
    """Where the level comes from and what half way up it means, for screens that do not score targets.

    The two lines the readout otherwise takes from the combining rule — "each target's score, rewards and
    inhibits averaged separately" and "half = the mix is favourable". Both are false on a screen whose level
    is a normalised metric, and they were printed there: a strip describing inhibits that do not exist, above
    mechanisms that were following something else entirely. `None` means the rule's own wording is right."""

    OFFSET_TAU_SECONDS = 5.0
    """How fast the standing-offset readout follows. Slow, because it is a *standing* offset and the number
    is read to decide whether a channel is about to rail, not to watch it wobble."""

    def _watch_offset(self, block) -> None:
        """Track each channel's DC offset and its headroom to the rail, in µV.

        **The number that explains a no-signal verdict, and the one thing nothing on screen could show.**
        At the device's 0.01 Hz corner an electrode pair's standing potential difference passes straight
        through — measured 305 and 311 µV on a real capture, with one channel reaching 652 µV against an
        800 µV rail. Rail and the device reports no signal, `_looks_open` agrees with it, and the channel goes
        dead.

        **Contact impedance does not predict this and cannot diagnose it.** They are different quantities:
        impedance is how easily current flows, offset is the potential difference between two electrodes,
        and the same capture had 14 and 26 kOhm — fine — beside a 311 µV offset. So an operator seeing a
        dead channel checks contact, finds it good, and has nowhere else to look.

        There was nowhere else to look here either, and that is partly this codebase's doing: the raw trace
        is drawn through a high-pass at `DISPLAY_FLOOR_HZ`, which is what makes it readable at this corner and
        also removes the offset from the only place it was visible.
        """
        values = block.values
        if not values:
            return
        mean = sum(values) / len(values)
        step = min(1.0, (len(values) / max(1e-6, block.spec.rate_hz)) / self.OFFSET_TAU_SECONDS)
        prior = self._offset_uv.get(block.channel)
        self._offset_uv[block.channel] = mean if prior is None else prior + step * (mean - prior)
        full = abs(block.spec.scale) * optima.ADC_FULL_SCALE
        room = full - max(abs(v) for v in values)
        seen = self._headroom_uv.get(block.channel)
        self._headroom_uv[block.channel] = room if seen is None else min(seen, room)

    def offset_uv(self, channel: int) -> float | None:
        """The channel's standing offset, or None before anything has arrived."""
        return self._offset_uv.get(channel)

    def headroom_uv(self, channel: int) -> float | None:
        """The closest this channel has come to the measurement rail, in µV. The *worst* of the session
        rather than the current value, because a channel that grazed the rail once will do it again and the
        operator needs to know that after it has recovered."""
        return self._headroom_uv.get(channel)

    OPEN_FRACTION = 0.98
    """How close to full scale counts as an open input. Measured: an unconnected input rails to *exactly*
    ±800.0 µV on the 800 µV range, so anything at 98% of scale is the rail and not a signal."""

    FLAT_SPAN_COUNTS = 2
    """How many raw counts of spread still count as dead flat over a block. Exact equality was the old
    test, and it assumed a shorted pair delivers literally constant counts — but a real short still sees
    the front end's own noise, plausibly a count or two, so the exact test could only ever catch digital
    pathologies. Two counts is 0.05 µV, forty times below the quietest EEG over any block, and the true
    figure awaits a deliberately shorted-pair capture."""

    def _looks_open(self, block) -> bool:
        """Whether the samples say this input is disconnected. **The only thing that condemns a channel.**

        An open input rails to full scale and a shorted pair reads a dead flat line, and either is visible in
        the block itself — so a real disconnection is caught on the very first block, and everything else is
        a channel delivering something. Both halves are load-bearing now that `_judge_channel` reads this in
        both directions: it is the conviction *and*, by returning False, the acquittal.

        Two things make False trustworthy on this hardware. A disconnected input does not float and drift
        here, it rails to **exactly** ±800.0 µV — measured, not assumed — so "not railed" really does rule
        out an open electrode. And quantisation is 0.0244 µV per count at this range, which is what makes
        a span within `FLAT_SPAN_COUNTS` over a whole block mean shorted rather than merely quiet: on a
        live channel eight near-identical counts do not happen. Checked against the 6 August bench
        capture, where the two unconnected channels railed on 421 and 1870 blocks of 1870 and the two
        live ones on none — the flat branch, unlike the rail, has never been exercised by a real short.
        """
        values = block.values
        if not values:
            return True
        full = abs(block.spec.scale) * optima.ADC_FULL_SCALE
        if full and min(abs(v) for v in values) >= self.OPEN_FRACTION * full:
            return True                     # railed: an open input
        # signed samples (abs called a ±20 µV alternation constant); near-flat, see FLAT_SPAN_COUNTS
        return max(values) - min(values) <= self.FLAT_SPAN_COUNTS * abs(block.spec.scale)

    def trust_channel(self, channel: int, trusted: bool | None) -> None:
        """The operator's override of the device's ruling, from the trace where they can see the signal.

        `True` forces live, `False` forces dead, `None` hands the decision back. It exists because the device
        is sometimes simply wrong and the operator can see that it is: 800 µV is the narrowest range a
        voltage channel offers, so 20 µV of EEG sits at 2.5% of full scale and the amplifier's own
        below-range detector fires on a perfectly good channel. Nothing in software can adjudicate that
        better than the person looking at the waveform, so the last word belongs to them.
        """
        if trusted is None:
            self.trusted.pop(channel, None)
        else:
            self.trusted[channel] = trusted
        self.state[channel] = self._resolve(channel)

    def _resolve(self, channel: int) -> bool | None:
        override = self.trusted.get(channel)
        return override if override is not None else self._device_says.get(channel)

    def _judge_channel(self, block) -> None:
        """Update a channel's live/dead state from one block.

        Two inputs, in order of authority: the operator's override, then the block. The device's flag opens
        the question and never settles it — `_looks_open` does, and it now answers in **both** directions.
        `None` survives until the first block, because "not heard from" is neither good nor bad and the
        screens have a state for it.

        **The samples can acquit as well as convict, and that asymmetry was the whole defect.** Railed or
        dead flat is a disconnection, visible in the block, and condemned on the first one. Anything else is
        a channel delivering something — so a live channel recovers on its next block rather than waiting for
        the amplifier to withdraw a complaint it is under no obligation to withdraw. The comment above
        `_judge_channel`'s constants has the history and why a timer was the wrong instrument.

        Two consequences worth being explicit about, because both are choices rather than oversights:

        - **A channel can no longer be dead while its trace looks alive**, which is the state that was
          reported and has no honest reading. If the samples are a signal, the channel is live and the
          measurement is the operator's to judge from the trace and the contact check.
        - **Over-range is not honoured separately.** `Block.valid` is false for it too, and a block where
          only *some* samples clip does not look open, so it reads as live. That is right: the channel is
          delivering, the artifact gate's ceiling rejects the clipped blocks, and `headroom_uv` already
          reports how close to the rail the session has come. A fully clipped block rails, and rails are
          condemned like any other.
        """
        channel = block.channel
        if block.valid or not self._looks_open(block):
            self._device_says[channel] = True
        else:
            self._device_says[channel] = False
        self.state[channel] = self._resolve(channel)

    def dead_channels(self) -> list[str]:
        """Which inputs are delivering nothing, for the Feedback tab to name. A hook because `state` means
        different things on different sessions — here a per-channel dict, on the HRV screen a beat."""
        return [f"ch{c}" for c, ok in self.state.items() if ok is False]

    @property
    def corner_hz(self) -> float:
        """The device's high-pass corner, **read from what is actually streaming**.

        The infra-low and coupling screens both need this — a reward band below the corner measures the
        filter's roll-off rather than the signal, and both warn about it. Both used to read
        `device.status.settings`, which exists only live, so **replaying a capture made at 0.01 Hz reported
        0.5 and warned about wiring that was already right.**

        Asking the source instead of the operator is the point. The corner is in the config blob, the blob
        is in the session header, and the header is what the replay decodes with — so the number is already
        on disk and a flag to restate it could only ever disagree with it. Every frame-based source carries
        `config` for this, and a text export carries none, which is the one case where 0.5 is a guess rather
        than a reading.

        Per-channel because the field is, and the worst channel wins: a warning should describe the corner
        that will spoil the measurement, not the friendliest one present.
        """
        return corner_of(self.source, self.specs)

    def companion(self) -> dict:
        """What a recording's own format cannot carry, for `nfb.notes` to write beside it.

        **The signal states in their own file, which is what makes EDF a real choice rather than a
        lossy one.** A native frame recording already holds all of this; an EDF export holds none of
        it and cannot — decoded samples on a fixed grid is precisely why other tools read it. So the
        states, the per-second quality and the config blob travel alongside, and the pair together
        answers everything the native format answers except re-decoding the raw frames.
        """
        from datetime import datetime, timezone

        cfg = getattr(self.source, "config", b"") or b""
        return {
            "session": {
                "protocol": self.protocol.code,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": cfg.hex(),
                "cell_seconds": quality.CELL_SECONDS,
                "corner_hz": self.corner_hz,
                "gaps": self.gaps,
                "channels": {str(c): {"rate_hz": s.rate_hz, "scale": s.scale, "unit": s.unit,
                                      "decimation": s.decimation}
                             for c, s in self.specs.items()},
                "forced": {str(c): v for c, v in sorted(self.trusted.items())},
            },
            "alarms": [(datetime.fromtimestamp(at, tz=timezone.utc).isoformat(timespec="seconds"),
                        event) for at, event in self.sentry.events],
            "quality": {c: self.quality.series(c) for c in self.specs},
            "marks": [(datetime.fromtimestamp(at, tz=timezone.utc).isoformat(timespec="seconds"),
                       note) for at, note in self.marks()],
        }

    def marks(self) -> list:
        """Moments this protocol judged worth keeping, as `(wall_time, note)`. Empty for most screens.

        Wall time and not the monotonic clock the sessions run on, for the reason the sentry's events are:
        these are read back from a file beside the recording, where only a real timestamp means anything.
        A monotonic value lands in 1970 and raises nothing."""
        return []

    def calibration(self) -> dict:
        """What this session sets by ear and by eye, for `prefs` to keep between launches.

        A hook rather than a field read, because not every screen is built on a `FeedbackSettings` — see
        `prefs.restore`, where assuming one was a crash on the HRV screen.
        """
        return prefs.capture(self.feedback)

    def restore_calibration(self, values: dict) -> int:
        n = prefs.apply(self.feedback, values)
        # **A screen with no "doubles after" row must not carry a hold.** The doubling itself is generic —
        # `_maybe_click` reads `Click.hold_seconds` on every protocol — while the *row* that sets it is
        # offered only where `sustains_reward` is on. A stored non-zero value therefore restored into a
        # screen with no control for it and no mention of it, and the mark changed character with nothing
        # on the Feedback tab able to say so or turn it off. A file written by a build that offered the row
        # everywhere is exactly that case, and it survives an upgrade.
        if not self.sustains_reward:
            self.feedback.click.hold_seconds = 0.0
        return n

    sum_spectrum_leads: bool = False
    """Whether the summed signal's spectrum is offered before the channels or after them.

    **The order is a claim, which is why it is a per-screen declaration and not a constant.** Where every
    reward is measured on the sum — coupling, alpha-theta — the sum leads, because it is the instrument and
    the channels are the diagnostic beside it. Where nothing is measured on a sum, or a target sits on one
    channel unless somebody moves it, the channels lead: putting the sum first there would present a
    diagnostic as the instrument.

    It was five hand-written lists before this, and they had drifted into three different orders with two
    screens offering no sum at all. Declaring it makes disagreeing deliberate."""

    def spectrum_sources(self) -> list:
        """What the Spectrum tab can draw, in order — see `spectrum_pane.SpectrumSource`.

        Built from `rings` (one per channel) and `spectrum_sum_ring()`, so a screen contributes the rings its
        own band limits need and says nothing about assembling them. Empty means the screen has no spectrum.
        """
        channels = [SpectrumSource(f"ch{c}", ring, channel=c)
                    for c, ring in getattr(self, "rings", {}).items()]
        ring = self.spectrum_sum_ring()
        if ring is None:
            return channels
        # No `channel`: a sum is not a channel and must not move the focus that decides which channel a new
        # target lands on.
        summed = [SpectrumSource("sum", ring)]
        return summed + channels if self.sum_spectrum_leads else channels + summed

    def spectrum_sum_ring(self):
        """The ring holding the summed signal's spectrum, or None where there is nothing to add.

        A screen that measures on a sum has a fed `sum_ring` and that wins. Every other screen with two or
        more channels at one rate gets a `SummedRing` derived from its channel rings — see that class for
        why — so "sum" is on the Spectrum tab wherever there is something to add, not only where a
        protocol happened to plumb it. One channel, or channels at different rates, is nothing to add.

        The synchrony and ILF screens override this to offer their fed ring only when they have a pair,
        because their ring is built unconditionally and on one channel it would name an electrode that
        is not there.
        """
        ring = getattr(self, "sum_ring", None)
        if ring is not None:
            return ring
        rings = getattr(self, "rings", {})
        if len(rings) < 2 or len({r.rate_hz for r in rings.values()}) != 1:
            return None
        derived = getattr(self, "_derived_sum_ring", None)
        if derived is None or derived.rings != list(rings.values()):
            from .traces import SummedRing
            derived = self._derived_sum_ring = SummedRing(rings.values())
        return derived

    relative: bool = False
    """Spectrum in percent of total rather than microvolts. On the session because the target cards read
    it too — their unit has to agree with the axis it is being compared against."""

    spectrum_scale: float = 0.0
    """µV full scale for the spectrum; 0 is the sticky auto."""

    threshold_modes: tuple[str, ...] = MODES
    """Which threshold modes this protocol allows on its cards. **A protocol may require that its
    thresholds hold still**, and the reason is the same one in two places:

    - alpha-theta trains theta *upward* over half an hour. A threshold tracking a rolling window follows
      theta up and pins the percentage where it started, so the trainee improves and the number does not
      move — erasing the change the protocol exists to produce. Its own plan ships with autothresholding
      off.
    - cross-frequency coupling trains **time above threshold** directly: "the goal of this type of training
      is to increase the client's production of reward 1 and reward 2". A tracker pins that production by
      construction, so the quantity being trained cannot show a change either.

    Both therefore offer `fixed` alone, derived once from the first 30 s. `redefine` is unaffected — that is
    an operator action, not a mode. Leaving `automatic` pressable there was a control that turned the
    protocol into a different one without saying so.
    """

    tone_bias: tuple[str, str] | None = None
    """How a keyed tone chooses when more than one reward is available: (biased, unbiased) labels, or None
    where the protocol has no such choice.

    Protocol-supplied *labels*, like `state_ends`, because only the protocol knows what the preference means
    — "theta first" against "whichever is better" — and the pane knows only that there are two ways of
    resolving it. It sits beside the tone on the Feedback tab rather than on the Train tab, where it began:
    it is sound configuration, and Train is for what you watch.
    """

    unbiased: bool = False
    """Whether a keyed tone takes the better-scoring reward rather than the protocol's preferred one."""

    def earning_target(self) -> Target | None:
        """Which target a keyed reward click should sound like, or None for the fixed sound.

        A protocol with one condition has no answer to this and does not need one — the click means the
        whole condition was met. A protocol whose rewards are *alternatives* does: the pitch is what says
        which of them you are earning.
        """
        return None

    # --- targets ---------------------------------------------------------------------------

    @staticmethod
    def band_key(band: Band) -> str:
        return f"{band.low:g}-{band.high:g}"

    def remove_target(self, tgt: Target) -> None:
        if tgt in self.targets:
            self.targets.remove(tgt)

    def _due(self, tgt: Target, fresh: set) -> bool:
        """Whether this block produced a new reading for this target.

        One reading per target per arriving reading, whether the target reads one channel or several. A
        multi-channel target has *one* estimator — its channels are added before anything measures them — so
        its key is in `fresh` exactly once however many electrodes feed it. Observing once per contributing
        channel instead would advance it N times as often as a single-channel target and make every window it
        has N times too short, which is the bug this method exists to prevent.
        """
        return self.key_for(tgt) in fresh

    # --- the one signal --------------------------------------------------------------------

    @property
    def level(self) -> float:
        """0…1, from all targets at once, smoothed. Every mechanism follows this one number."""
        return self._level

    @property
    def active_targets(self) -> list[Target]:
        """Targets whose channels are actually delivering signal.

        A channel the device reports inactive or unusable produces a value of zero, its threshold
        calibrates to zero, and its score is therefore exactly 0.5 — forever. Under `weakest` that **caps
        the feedback at half** permanently, and under any rule it drags the average toward the middle.
        Measured on a four-column export whose last two columns are padding: twelve of twenty-four bins sat
        at 0.500 and the level never moved off 0.5.

        A channel with no signal is not a condition anybody can satisfy, so it is not part of the signal.
        It stays visible on the screen — an unsatisfiable condition you cannot see is worse than one you
        can — and it stops driving the output.
        """
        # `monitor` here rather than in each caller: a monitor is measured, calibrated, drawn and scored,
        # and the *only* thing that makes it a monitor is not reaching the output. One place, so a new
        # mechanism cannot accidentally read one.
        return [t for t in self.targets
                if not getattr(t, "monitor", False) and self.target_is_live(t)]

    def channel_targets(self, channel: str) -> list[Target]:
        """Which live targets drive one of the two output channels — the level, or the condition.

        Both channels read every target by default, and for most protocols that is the only coherent
        answer: the targets *are* what is being trained, so measuring one and then leaving it out of an
        output is the failure `level()` exists to prevent.

        It is a hook because a protocol can have **two descriptions of the same state** and want a
        different one on each channel — alpha-theta has theta and alpha as alternatives *and* their
        difference, and which belongs on a continuous fade is not the same question as which belongs on a
        discrete tone. Only the protocol knows that, which is the same reason `state_balance` and
        `earning_target` are hooks.

        Whatever this returns has to be the whole story for that channel, because `FeedbackSettings` sees
        nothing else: a target left out of both channels is measured and displayed and fed back nowhere,
        and the screen has to say so rather than let it look like an output.
        """
        return self.active_targets

    def target_is_live(self, tgt: Target) -> bool:
        """Exclusion needs *evidence* of no signal, so an unknown channel counts as live.

        The polarity matters: guessing that a channel is dead and dropping it from the signal is a silent
        change to what is being trained, while keeping a channel that turns out to be dead is visible on
        the trace within a second.
        """
        # `is not False` throughout, not truthiness: None means "not heard from yet" and has to count as
        # live, which is the polarity this docstring is about. `bool(None)` would silently invert it.
        if tgt.pair is not None:
            return all(self.state.get(ch) is not False for ch in tgt.pair)
        if tgt.source == PAIR:
            return any(v is not False for v in self.state.values()) if self.state else True
        return self.state.get(tgt.source) is not False

    @property
    def raw_level(self) -> float:
        """The combining rule's answer, before the level's own range and the sensitivity reach it."""
        return self.feedback.level(self.channel_targets(LEVEL))

    def shaped_level(self, raw: float, now: float | None = None) -> float:
        """`raw` moved onto the output's range by the one dial — the last step of the signal chain.

        **Only where the level is built by scoring targets**, which is exactly what `scores_bands` names.
        A screen whose level is already a normalised metric bounded 0…1 by construction — the heart screen's
        coherence, HEG's signal — has nothing a gain would fix and no calibrated middle to move it about:
        it already uses its range, and stretching it again would be a second normalisation of a number that
        has had one. Same argument `state_sensitivity_range` makes for the other sensitivity.
        """
        if not self.scores_bands:
            return raw
        self.level_range.observe(raw, now)
        return self.feedback.shape(raw, self.level_range.centre, self.level_range.span)

    def _advance_level(self, now: float | None = None) -> None:
        """Advance the smoothed level toward the combining rule's answer.

        `now` is **signal time** — the arriving block's end, in seconds of samples — and the step is the
        time since the last advance, not a constant. It was `dt = 1/METRIC_HZ`, which assumed one advance
        per metric tick; the loop actually advances once per arriving block group that produced a fresh
        reading, so with targets spread over three channels the constant was applied at 3× the assumed
        rate (measured 60.7 Hz against the assumed 20) and the configured 0.3 s smoothing silently
        behaved as 0.1 s. Signal time rather than the wall clock so a replay reproduces a live session
        exactly, and blocks from different channels carrying the same instant cost ~0 extra smoothing
        steps, which is the correct count.
        """
        tau = max(0.0, self.feedback.smoothing_seconds)
        # Shaped *before* smoothing, which is the order the SIGNAL panel prints: combine, then how much of
        # the output's travel that swing should use, then how twitchy it is. Smoothing a raw level and
        # shaping afterwards would put a gain on the output of the smoother, so the same dial would also
        # change how fast the level appears to move.
        target = self.shaped_level(self.raw_level, now)
        if tau <= 0:
            self._level = target
            self._level_t = now
            return
        if now is None or self._level_t is None:
            dt = 1.0 / METRIC_HZ            # no clock: one nominal tick, as before
        else:
            dt = max(0.0, now - self._level_t)
        self._level_t = now
        if dt > 0.0:
            self._level += (1.0 - math.exp(-dt / tau)) * (target - self._level)

    def _drive_outputs(self) -> None:
        """Hand the level to the continuous outputs, and keep the music running.

        Every one of these owns its own take/follow/release lifecycle, so switching a mechanism off here
        is all it takes to restore what it was holding.

        **The applying happens on the output pump, not here.** This runs on the feed thread — the one
        that decodes and filters against a 32 ms budget — and the output calls are the slowest things in
        the process: a CoreAudio volume write is 1.1 ms at p99, the shadow's display lookup is a
        window-server round trip measured at 13 ms worst, and the music player can spawn a subprocess.
        So this method only *snapshots* what the outputs should follow and wakes the pump; the pump
        applies the freshest snapshot and drops any it missed, which is the correct semantics for
        outputs whose smoothing is in wall-clock seconds. Without a live pump — unit tests drive this
        method directly — it applies inline, exactly as it always did.
        """
        # **Not while stopping**, and this is a shutdown ordering race rather than a nicety. `stop()` sets
        # `_stop`, which ends the pump's loop — but the feed thread is still inside `source.blocks()` and
        # gets at least one more iteration before it notices. With the pump dead, the fallback below fired,
        # so the last apply of every session ran *on the feed thread*: precisely the CoreAudio and
        # window-server calls this whole mechanism exists to keep off it. `stop()` releases the outputs
        # immediately afterwards, so there was nothing to be gained by applying either.
        #
        # It presented as a test that failed about one run in six — `test_continuous_outputs_apply_on_their
        # _own_thread_not_the_feed` — which is what an intermittent assertion usually is: a real race, seen
        # from the outside.
        if self._stop or self.paused:
            return
        met = self.feedback.on_target(self.channel_targets(CONDITION))
        self._output_state = (self._level, met, self.shadow_input(), self.state_balance)
        pump = self._output_thread
        if pump is not None and pump.is_alive():
            self._output_wake.set()
        else:
            self._apply_outputs(*self._output_state)

    def _apply_outputs(self, level: float, met: bool, shadow_in: float,
                       balance: float | None) -> None:
        sh = self.feedback.shadow
        self.display.apply(sh.enabled and sh.supported, sh.screen, *sh.ends(),
                           sh.map(shadow_in), smoothing=sh.smoothing_seconds)
        snd = self.feedback.sound
        self.output_volume.apply(snd.enabled, *snd.ends(), snd.map(level))
        # Quiet Reward pauses the music when the condition *fails*, which is the DVD model without a
        # DVD: "the playback will pause whenever one of the Reward instruments falls below threshold".
        #
        # The Shepard's pitch can follow the protocol's *state* instead of the level — see
        # `Music.follows`. Falls back to the level when there is no state, the same honest reading
        # `shadow_input` documents: the setting is only offered where a state exists, so reaching this
        # with none means the protocol changed under a stored setting.
        music = self.feedback.music
        music_in = float(balance) if music.follows == STATE and balance is not None else level
        self.music.apply(music, rate_hz=self.rate,
                         hold=self.music_hold(met), balance=balance, level=music_in)

    def _output_pump(self) -> None:
        """Apply output snapshots for as long as the session runs, one at a time, freshest wins.

        A dedicated thread so the feed thread never waits on CoreAudio or the window server. It also
        serialises apply against release: while the pump is alive, both go through here, so a release
        requested mid-apply lands after it rather than interleaved with it.
        """
        while not self._stop:
            if not self._output_wake.wait(timeout=0.5):
                continue
            self._output_wake.clear()
            try:
                self._apply_snapshot(self._output_state)
            except Exception as exc:
                slog(f"output pump: {type(exc).__name__}: {exc}")

    def _apply_snapshot(self, state: tuple | None) -> None:
        """One step of the pump: apply a snapshot, or hand the outputs back.

        **Paused releases, and it used to merely skip.** The pause asks for a release by publishing
        `None`, but the feed thread can be a block deep in `_drive_outputs` when that lands, and its
        snapshot then overwrites the `None`. The pump saw a snapshot, saw `paused`, and skipped — so
        nothing was applied and nothing was released, and whatever was sounding kept sounding until the
        session resumed or stopped. Reported as a tone that went on playing through a pause.

        Skipping was never a state worth having: a paused session feeds nothing back, so there is only
        one right answer while `paused` and it is the same one `None` gets.
        """
        if state is None or self.paused:
            self._release_outputs()
            return
        self._apply_outputs(*state)

    music_failing: tuple[str, str] | None = None
    """What "failing" means for the playback gate on this screen, when it is not the default — as
    `(label, why)`, or None for the standard "a target is failing".

    Exists because a protocol may override `music_hold` and change what the gate keys on, and a control
    whose behaviour differs from its label is a defect however good the reason: ILF Synchrony keys the
    failing polarity on its inhibits, because its reward is met 10% of the time by design and gating on
    the whole condition would silence the playback most of the session. The pane prints the label on the
    pill and the why in the note, so the override is visible where the choice is made."""

    def music_hold(self, met: bool) -> bool:
        """Whether the music should be silent right now. The music settings' own answer by default.

        A hook because a protocol can have a *second* reason to hush it that no setting describes —
        alpha-theta's sleep guard drops the music so its knock arrives into silence. Override and `or` rather than replace, or a protocol adding a reason
        would quietly remove `stops_when`.
        """
        return self.feedback.music.holds(met)

    def _drive_shadow(self) -> None:
        sh = self.feedback.shadow
        self.display.apply(sh.enabled and sh.supported, sh.screen, *sh.ends(),
                           sh.map(self.shadow_input()), smoothing=sh.smoothing_seconds)

    def shadow_input(self) -> float:
        """What the shadow follows: the level, or the state where the protocol has one.

        Falls back to the level when `follows` says state and `state_balance` is None, which is not a
        silent substitution but the only honest reading — the setting is offered on the pane *only* where a
        state exists, so reaching this with none means the protocol changed under a stored setting.
        """
        if self.feedback.shadow.follows == STATE:
            balance = self.state_balance
            if balance is not None:
                return float(balance)
        return self._level

    def _maybe_click(self) -> None:
        """Fire the reward click when it is due.

        The first click lands the instant every target is met rather than one interval later, because the
        useful thing about a discrete reward is learning *when* you got there.

        The interval is enforced across the *whole session*, not per stretch of being on target. An
        earlier version cleared the phase whenever the targets stopped being met, so that re-entering
        would click at once — and a signal sitting near its threshold flickers, so it machine-gunned:
        measured gaps of 0.15 s at a 1 s interval. Keeping the phase means a long absence still clicks
        immediately on return, and flicker cannot beat the interval.

        **`hold_seconds` changes the mark's character rather than gating it**, so the stream stays one event
        per interval and what you hear is `blip … blip … blip-blip … blip-blip` — the **transition** is the
        information. No extra sound fires at the moment it changes: the next scheduled mark simply comes out
        doubled, which keeps the rule above that flicker cannot beat the interval. Zero by default.

        The criterion is a *duty cycle* over the window and not an unbroken run — see `HOLD_FRACTION` for the
        measurement that forced that, which is the same flicker this docstring already opens with.
        """
        click = self.feedback.click
        if not click.enabled:
            self._last_click = None
            self._condition_log.clear()
            return
        now = time.monotonic()
        met = self.feedback.on_target(self.channel_targets(CONDITION))
        # Recorded whether met or not, because the hold criterion is a *fraction* of the window and needs the
        # misses as much as the hits. Trimmed to the window, so this stays a handful of samples.
        self._condition_log.append((now, met))
        # Trimmed to the window **plus a margin**, deliberately: `click_held` refuses to answer until it can
        # see a sample older than the window, and trimming to exactly the window means the oldest sample is
        # always younger than it — so the criterion could never once be true. Off by one, and silent.
        horizon = now - (max(1.0, click.hold_seconds) + 1.0)
        while self._condition_log and self._condition_log[0][0] < horizon:
            self._condition_log.popleft()
        if not met:
            return
        if self._last_click is None or now - self._last_click >= click.interval_seconds:
            self._last_click = now
            # **A keyed reward is a tone, not a click**, where the protocol has an earning target. The
            # alert's `frequency_click` is the right shape for its job — twelve bins, and you want to know
            # roughly where the trouble is without a sound that rings — and the wrong shape for this one:
            # its noise bursts are heard as clicks rather than pitches, and across theta and alpha they
            # land under an octave apart. The two alpha-theta reward sounds are *tones* two and a half
            # octaves apart, because their whole job is to be told from each other with the eyes closed.
            # See `cues.reward_tone`. How loud is the system volume's business, not the click's.
            earning = self.earning_target() if click.keyed else None
            sound = (cues.tone_at(cues.target_pitch(earning)) if earning is not None
                     else click.sound)
            self.cue(cues.held(sound) if self.click_held(now) else sound, click.volume)

    def click_held(self, now: float | None = None) -> bool:
        """Whether the condition has been *held* long enough for the mark to be the doubled one.

        A duty cycle over the window rather than an unbroken run — see `HOLD_FRACTION`, where the measurement
        that forced that is recorded. A brief drop does not reset it, which is the point: the state being
        marked is "you have been there for the last few seconds", and on a real signal nobody is ever there
        continuously.

        Needs a *full* window before it can be true, so the doubled mark cannot appear in the first seconds
        of a session on two samples that both happened to be met.

        Public because the pane prints it — the alternative is asking the audio layer what it last played,
        which is how a readout and the sound it describes drift apart.
        """
        click = self.feedback.click
        if not click.hold_seconds:
            return False
        return self.duty_held(self._condition_log, click.hold_seconds, now)

    @staticmethod
    def duty_held(log, seconds: float, now: float | None = None) -> bool:
        """Whether a `(at, flag)` log says the flag has been up for most of the last `seconds`.

        One implementation, because a second screen now asks the same question of a different condition —
        and the two things easy to get wrong are both here: it is a fraction rather than an unbroken run,
        and it refuses to answer until a sample older than the window exists."""
        if not log:
            return False
        now = time.monotonic() if now is None else now
        if now - log[0][0] < seconds:
            return False                    # not enough history to say
        window = [flag for at, flag in log if at >= now - seconds]
        return bool(window) and sum(window) / len(window) >= HOLD_FRACTION

    def _maybe_alert(self) -> None:
        """Sound the alert for a target that is *failing*, which is the opposite of the click.

        CFC's own arrangement: "anytime an inhibit bin goes above threshold there will be an audio
        alert". The sound is **keyed to the failing target's frequency** — a slow bin thocks, a fast one
        ticks — so a run of alerts says where in the spectrum the trouble is. Deriving the sound from the
        band means twelve bins need no configuration and a thirteenth already sounds right.

        One sound at a time, and the *worst* offender rather than the first in the list: several
        simultaneous alerts is noise rather than information, and picking by list order would let a bin
        that is barely over mask one that is far over.
        """
        alert = self.feedback.alert
        if not alert.enabled:
            self._alerting.clear()
            return
        now = time.monotonic()
        # Not monitors: a monitor observes without judging, which is the invariant `active_targets`
        # enforces everywhere else — an alert sounding for one would be the display alarming on itself.
        failing = [(i, tgt) for i, tgt in enumerate(self.targets)
                   if tgt.total and not tgt.monitor and not tgt.wanted and self.target_is_live(tgt)]
        self._alerting = {i for i, _tgt in failing}
        if not failing:
            return
        # worst = furthest the wrong side of its own threshold, in that target's own scale
        i, tgt = max(failing, key=lambda pair: 1.0 - pair[1].score(self.feedback.normalisation))
        last = self._last_alert.get(i)
        if last is not None and now - last < alert.interval_seconds:
            return
        self._last_alert[i] = now
        # A target without a band — a derived quantity like theta-minus-alpha — has no frequency for a
        # keyed sound to encode, so it falls back to the configured one rather than raising in the feed
        # thread, which a broad `except` up there would turn into a dead session.
        band = getattr(tgt, "band", None)
        keyed = alert.keyed and band is not None
        self.cue(cues.frequency_click(band.centre) if keyed else alert.sound, alert.volume)

    @property
    def outputs(self) -> dict[str, float]:
        """What each enabled mechanism would currently be doing."""
        if not self.active_targets:
            return {}
        return self.feedback.outputs_for(
            self._level, self.feedback.on_target(self.channel_targets(CONDITION)))

    # --- transport -----------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        running = (time.monotonic() - self._resumed_at) if self._resumed_at is not None else 0.0
        return self._accumulated + running

    @property
    def running(self) -> bool:
        return not self.paused

    @property
    def length(self) -> float | None:
        """The optional maximum, or None for open-ended — which is the usual case."""
        return self.plan.length_seconds

    def toggle_pause(self) -> None:
        """Pause and resume, **and let go of every output while paused.**

        The feed loop skips the outputs while `paused`, which stops them being *updated* and not being
        *held*: the Shepard tone sang on at whatever pitch it had reached, the display stayed dimmed and the
        system volume stayed where the signal left it. A paused session is not feeding anything back, so
        holding the machine's display and speakers is the same failure `suspend` already refuses — and
        `suspend` releases them for exactly this reason, which is why leaving for the device screen stopped
        the tone and pausing did not.

        Resuming does not need to restart anything by hand: every one of these owns its own
        take/follow/release lifecycle inside `apply`, so the next `_drive_outputs` takes them again.
        """
        if self.paused:
            self.paused = False
            self._resumed_at = time.monotonic()
        else:
            self._accumulated = self.elapsed
            self._resumed_at = None
            self.paused = True
            self._outputs_off()

    def _outputs_off(self) -> None:
        """Hand the outputs back, through the pump when one is running.

        Routing through the pump is what makes release safe against a concurrent apply: the pump does
        one thing at a time, so a release requested while an apply is mid-flight lands after it instead
        of interleaving CoreAudio and window-server calls from two threads. Without a live pump —
        session never started, or already stopping — the release runs inline, which is also what keeps
        the pause/suspend tests synchronous.
        """
        pump = self._output_thread
        if pump is not None and pump.is_alive():
            self._output_state = None
            self._output_wake.set()
        else:
            self._release_outputs()

    def _release_outputs(self) -> None:
        """Hand back the display, the system volume and the audio. Safe to call repeatedly."""
        self.display.release()
        self.output_volume.release()
        self.music.release()

    def tick(self) -> None:
        """Advance the clock and fire anything due. Called from the draw loop."""
        if not self.running:
            return
        elapsed = self.elapsed
        for cue in self.plan.due(elapsed):
            self.cue(cue)
        if self.plan.complete(elapsed):
            self.finished = True
            self._accumulated = self.plan.length_seconds
            self._resumed_at = None
            self.paused = True

    # --- going to the device screen and coming back -------------------------------------

    def suspend(self) -> None:
        """Stop consuming and let go of every output, keeping the session itself intact.

        **Checking impedance mid-session is an ordinary thing to do** — you pause, look at contact, come
        back and carry on — and impedance testing is an exclusive device mode, so the feed has to stop. What
        must *not* happen is the session dying: the elapsed clock, every target's trend and calibration, the
        artifact gate's running level and the on-target history are the session, and starting a fresh one
        with the same settings is a different session that merely looks alike.
        `stop()` remains the destructive path, for the stop button, which means what it says.

        The clock freezes where it is, because time spent looking at electrodes is not time on task. Outputs
        are released rather than left held: a screen dimmed by a session nobody is watching is the failure
        `shadow`'s whole safety story is about.
        """
        if not self.paused:
            self.toggle_pause()         # which releases them
        self._outputs_off()             # and again, for a suspend that arrives already paused

    @property
    def suspended(self) -> bool:
        """Whether the feed has stopped but the session is still alive and re-armable."""
        return not self._stop and not self.feeding

    def rearm(self) -> None:
        """Put the device back in measuring mode so `run()` can be called again.

        The subscription survives a device stop — it is only handed an `EpochEnd`, which ends the iteration
        rather than the object — so re-arming needs the mode back and a fresh feed thread. The decoder is
        rebuilt on the new epoch, so per-channel sample indices restart and a `Gap` is reported; that is the
        truthful description, since there really is a hole in the signal as long as you were away.

        **It must not assume the device is idle, which is the same mistake `__init__` documents and this
        method is the far likelier way to make it.** Leaving the device screen by the crumb does not stop a
        running contact check — only pressing the button again does — so coming back from an impedance
        measurement found the device in `TESTING`, and this did nothing at all. The mode never changed, the
        surviving subscription was handed *impedance* frames, and the screen came back with a running clock,
        flat traces and no target ever fed. That is exactly the failure `__init__` guards against, arrived at
        by the route somebody actually takes: pause, check contact, come back.

        Stopping first is also what hands the channel count back. `narrow_to_wired` borrows it for the
        duration of a check and `Device.stop` restores it, so a session re-armed without stopping would
        resume on however many channels the contact check happened to leave enabled.
        """
        if self.device is None:
            return
        if self.device.status.mode is Mode.TESTING:
            self.device.stop()
        if self.device.status.mode is Mode.IDLE:
            self.device.start_measurement()

    def stop(self) -> None:
        self._stop = True
        self.paused = True
        # **Keep what was set by ear.** Only calibration — see `prefs` for the line and why arrangement and
        # bands are not on the other side of it. On `stop` rather than on every change, because a session
        # ends far less often than a slider moves and a write per drag is a write per frame.
        prefs.remember(self.protocol.code, self)
        # The pump first: wake it so it sees _stop and exits, wait for any in-flight apply to finish,
        # then release inline. The join has a timeout because stop() must never hang on a wedged
        # CoreAudio call, and release is idempotent so releasing after a straggling apply still wins.
        pump, self._output_thread = self._output_thread, None
        if pump is not None and pump.is_alive():
            self._output_wake.set()
            pump.join(timeout=1.0)
        # never leave the operator's machine dimmed, muted, or playing music
        self._release_outputs()
        try:
            self.source.close()
        except Exception:
            pass
        if self.device is not None:
            self.device.stop()

    # --- feed ----------------------------------------------------------------------------

    def feed_block(self, item) -> None:
        """One item from the source: judged, measured, and fed back if it earns it.

        Extracted from `run`'s loop unchanged. A caller that already has blocks — a test, a benchmark —
        drives the session with this and needs no thread, no source and no waiting.
        """
        if isinstance(item, Gap):
            self.gaps += 1
            return
        self._seen[item.channel] = self._seen.get(item.channel, 0) + len(item.values)
        key = (item.channel, item.state)
        self._states[key] = self._states.get(key, 0) + 1
        if item.channel not in self.specs:
            self._skipped += 1
            return            # a channel we are not using this session
        self._watch_offset(item)
        self._judge_channel(item)
        self.sentry.saw_block()
        self.sentry.tick(self)
        # Only judge what the **resolved** channel state calls usable — the verdict the dot and
        # the targets already follow, refreshed by `_judge_channel` just above, in which the
        # operator's override wins. It was the block's own `valid` flag, and this line was the one
        # place the override did not reach: on an infra-low channel the device flags "no signal"
        # continuously (20 µV of EEG is 2.5% of its narrowest range), so forcing the channel on
        # turned the dot green and counted it for every target while the metrics were still fed
        # `pass_through`'s stand-in — a constant at the last accepted sample, which on a channel
        # never once accepted is zero. On screen: a slow trace pinned flat to the axis beside a
        # healthy raw trace and a caption saying "forced on", which is exactly how it was reported.
        #
        # A channel known bad *bypasses* the gate rather than failing it, and that stays true here:
        # a genuinely open input flips the resolved state False via `_looks_open` on its first
        # block, so a rail still idles the filters instead of feeding them, while the traces and
        # spectra keep drawing (`judged.block` is untouched). Measured on the no-electrode bench
        # recording: railed to ±800 µV, every block failed the ceiling, and gating it froze the
        # whole screen. Forcing a channel *off* now also routes it here, which is what the
        # override's tooltip promises: out of every measurement, still on the trace.
        usable = self.state.get(item.channel) is not False
        judged = self.gates.feed(item) if usable else self.gates.pass_through(item)
        self.quality.observe(item.channel, item.end_index / item.spec.rate_hz, judged.total,
                             judged.rejected, not usable)
        self.observe_raw(judged.block, judged)
        # The metrics always advance, on `fed` — for a held block that is the gate's stand-in at
        # the last accepted value, which keeps sample indices contiguous. Skipping the block
        # instead made every estimator read the rejection as an acquisition gap and reset its
        # filters, re-running the whole settling time per blink; see the hold note in
        # `nfb.dsp.artifact`. What a rejection excludes is *training*: below, a block that was
        # not accepted observes no target, advances no level, and fires no reward.
        fresh = self.consume(judged.fed)
        if self.paused or not judged.accepted:
            return
        advanced = False
        for tgt in self.targets:
            # Live targets only. A channel the device calls unusable is already out of the
            # *feedback* via `active_targets`, but its readings still arrived here and poured
            # rail-derived values into the threshold windows — so an automatic threshold
            # tracked the ±800 µV rail, and a `fixed` one that settled during the bad stretch
            # was wrong for the whole session.
            if self._due(tgt, fresh) and self.target_is_live(tgt):
                tgt.observe(self.value_for(tgt), true_metric_hz(item.spec.rate_hz))
                advanced = True
        if advanced:
            self._advance_level(item.end_index / item.spec.rate_hz)
            self._maybe_click()
            self._maybe_alert()
            self._drive_outputs()
        # **Stream time, not the clock.** On the wall clock a replay fed faster than real time — every
        # test, every screenshot — gained a trend point per half second of *machine* time, so thirty
        # seconds of signal drew as one or two points depending on how warm the cache was, and the
        # target plot read "waiting for data" on a screen that had plainly been fed.
        now = item.end_index / item.spec.rate_hz
        self.stream_seconds = now
        if now - self._last_trend >= TREND_SECONDS or now < self._last_trend:
            self._last_trend = now
            for tgt in self.targets:
                tgt.trend.append(tgt.value)
        # The band history is sampled here, on the feed, for the same reason the trend is: ticked from
        # the draw loop it depended on how often screens drew, and a replay draws once.
        if self.band_trends is not None:
            self.band_trends.tick(self.spectrum_sources(), now)

    def run(self) -> None:
        """Consume the source until it ends or the session stops. The thread's job, and only that.

        **The work per item is `feed_block`, and it is a method rather than the inside of this loop** so
        that anything wanting to drive a session — a test above all — can hand it blocks directly instead
        of starting this thread and then waiting on a clock for it to catch up. Waiting on a thread is
        how a test ends up measuring the scheduler; every screen test in this suite used to do it.
        """
        self.feeding = True
        if self._output_thread is None or not self._output_thread.is_alive():
            self._output_thread = threading.Thread(target=self._output_pump, daemon=True,
                                                   name="outputs")
            self._output_thread.start()
        next_report = time.monotonic() + 2.0
        try:
            for item in self.source.blocks():
                if self._stop:
                    return
                now_ = time.monotonic()
                if now_ >= next_report:
                    next_report = now_ + 2.0
                    slog(f"feed: blocks/ch {self._seen} states {self._states} "
                         f"not-in-specs {self._skipped} gaps {self.gaps} paused {self.paused}")
                self.feed_block(item)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            slog(f"feed RAISED {self.error}")
        # Reached when the source's iterator ends — for a subscription that means the epoch ended, and
        # nothing restarts this thread. Logged because it leaves a live screen drawing a frozen buffer with
        # `error` unset, which is indistinguishable from a stalled UI.
        finally:
            self.feeding = False
        slog(f"feed ENDED after blocks/ch {self._seen} states {self._states} gaps {self.gaps} error {self.error}")

    DISPLAY_LIMIT_HZ = 45.0
    DISPLAY_FLOOR_HZ = dsp_artifact.JUDGE_FLOOR_HZ
    DISPLAY_ORDER = 4
    """The raw trace is drawn through a band-pass, and **the caption has to say so**.

    **The lower edge had the same bug the artifact gate had, and it presented as an invisible trace.** The
    filter was a lowpass alone, so the drawn band was "everything the amplifier passes, up to 45 Hz" — fine
    at the 0.5 Hz corner where the hardware supplied the lower edge, and useless at 0.01 Hz, where 305 µV of
    electrode drift against a ±100 µV full scale puts the whole envelope off-screen. What the operator sees
    then is the zero line, drawn unconditionally as a reference, and nothing else: a screen that looks like a
    dead channel while the amplifier is delivering perfectly good EEG 40× smaller than the drift on top of it.
    Taken from `JUDGE_FLOOR_HZ` rather than restated, so the band you see is the band the gate judges. **Order
    1 on that edge**, not `DISPLAY_ORDER` — for the reason the order-6 paragraph below already gives: a
    high-order high-pass rings on a blink, and overshoot the signal does not have is the worst thing a trace
    can show. A single pole decays monotonically, and it is what the hardware corner it stands in for is.

    The alternative was a gain control reaching 800 µV, and it is worse for this protocol: it would show the
    drift filling the pane with the EEG invisible inside it, which is the same failure the `DISPLAY_LIMIT_HZ`
    note below describes for mains hum. The slow waveform has its own pane, drawn from the metric; this one is
    for what the slow pane cannot show.

    One thing it costs, and the state dot is what covers it: an open input rails to a *constant* ±800 µV, and
    a high-pass of a constant is zero, so a channel with no electrode now draws flat at zero rather than
    pinned at full scale. Which channel is dead is still on the screen — the dot beside the trace goes red and
    the envelope is drawn in the muted colour — because `Block.state` says so and the trace never did.

    **The display filter is one large lowpass rather than anything clever.** A raw display channel is a
    stream with its own filter like any other, and 40 Hz is the widest filter the program builds by default.

    **We use 45 rather than 40, because CFC's highest bin is 25-45 Hz.** A corner at 40 would sit
    *inside* a band being measured, and the trace is what an operator reads to sanity-check what the bins are
    doing — the one place a display filter must not be quietly shaping the picture. 45 also stays below mains,
    so the fundamental remains the hardware notch's job rather than being half-attenuated here.

    **Order 4 rather than 2, because a trace has no latency budget.** Measured at 1000 sps: order 4 is
    **-1.4 dB at 40 Hz against order 2's -2.1** — flatter exactly where CFC needs it — while taking 100 Hz
    from -14.5 dB to **-28.7** and 200 Hz from -28.3 to -56.6. The cost is 9.4 ms of group delay at 10 Hz,
    which is nothing to a picture nobody times anything from. Order 6 is better again on both counts (-0.9 dB
    at 40, -43 at 100) and is not used because it rings harder on a blink: overshoot the signal does not have
    is worse on a trace than a few dB of hum.

    Not re-filtering what the hardware filtered — that rule is about *measurement*, and nothing here changes
    what a target reads. It is that the hardware's band limit follows the sample rate, 0.4x it, so the 1000 sps
    default passes everything to 400 Hz where 250 sps would have stopped at 100. Measured on this device with
    a real electrode: 554 uV peak-to-peak of which **60 uV lives above 100 Hz and the strongest single
    frequency is exactly 100.0 Hz** — the second harmonic of mains, which the device's 50 Hz notch does not
    touch. Alpha underneath it was 2.6 uV.

    Undrawable as well as unwanted: at 5 s across, the trace has ~64 columns per second, so anything above
    ~32 Hz fills every column's min/max and the waveform becomes an even comb of full-height strokes. So the
    trace was showing hum with the EEG invisible inside it, at 1/20th the amplitude.

    A display filter is the standard answer and the honest one *provided it is labelled*, which is why every
    screen prints it next to the window and the scale.
    """

    def for_display(self, channel: int, values) -> "np.ndarray":
        """`values` band-limited for drawing. Measurement never sees this.

        A list of stages rather than one filter, for the reason `ArtifactGate._band_limited` gives: the
        upper edge is conditional — `lowpass` returns None on a channel decimated below 100 sps — and the
        lower edge is not.
        """
        arr = np.asarray(values, dtype=float)
        if channel not in self._display_filters:
            spec = self.specs.get(channel)
            self._display_filters[channel] = [] if spec is None else [
                f for f in (highpass(self.DISPLAY_FLOOR_HZ, spec.rate_hz, order=1),
                            lowpass(self.DISPLAY_LIMIT_HZ, spec.rate_hz, order=self.DISPLAY_ORDER))
                if f is not None]
        for filt in self._display_filters[channel]:
            arr = filt.process(arr)
        return arr

    def observe_raw(self, block, judged) -> None:
        """Everything that wants the samples themselves — traces, spectra, sonification — including the
        ones a gate rejected. Override; the default does nothing."""
        _ = block, judged

    # --- helpers a screen wants ----------------------------------------------------------

    REJECTION_VISIBLE = 0.02
    """Above this the trace is drawn amber. Low, because the point is to notice it starting."""

    def filter_for(self, tgt: Target) -> tuple:
        """Everything that decides which estimator this target gets, and so the key it is stored under.

        Each kind carries the parameter it actually has: cFIR a design delay, the IIR kinds an order and a
        smoothing constant. Mixing them into one abstract "how late" number was the mistake this replaced.
        """
        kind = tgt.filter_kind or DEFAULT_FILTER[tgt.role]
        if kind == "cfir":
            return (kind, round(tgt.cfir_delay or CFIR_DELAY[tgt.role], 3))
        # The envelope constant and convention are the session's, not the target's — see
        # `Target.filter_ripple`. Both are part of the key, because two targets that differ in either
        # need different estimators.
        smoothing = max(0.01, self.feedback.envelope_seconds)
        ripple = tgt.filter_ripple or DEFAULT_RIPPLE if kind == "elliptic" else 0.0
        convention = self.feedback.envelope_convention or "rms"
        return (kind, tgt.filter_order or DEFAULT_ORDER, round(smoothing, 3), round(ripple, 2),
                convention)

    @property
    def drift_hz(self) -> float | None:
        """What a cFIR on this session's samples has to remove for itself — see `drift_for`."""
        return drift_for(self.corner_hz)

    def build_metric(self, spec, band: Band, tgt: Target):
        """The estimator a target asks for. Raises `ValueError` if the band will not fit the rate."""
        design = self.filter_for(tgt)
        if design[0] == "cfir":
            return CFIRBandPower(spec, band, delay_seconds=design[1], output_hz=METRIC_HZ,
                                 drift_hz=self.drift_hz)
        # The convention comes through the design key rather than being hardcoded here — "rms" was, for
        # a long time, which made the mean-abs convention unreachable and undercut the stated reason
        # the IIR kinds exist: a threshold means what it meant only under the convention it was set in.
        return BandPower(spec, band, output_hz=METRIC_HZ, order=design[1], kind=design[0],
                         envelope=design[4], tau_seconds=design[2], ripple_db=design[3])

    def response_time(self, tgt: Target) -> float | None:
        """How long the reading takes to show half of a burst in this band, in seconds.

        Concretely, and it is worth being concrete because the abstract phrasing was unreadable: a tone at
        the band centre switches on, and this is the delay before the number reports half its eventual
        size.

        **Not** `latency_seconds`, which is group delay plus smoothing — a conservative *bound*, not a
        delay. The two diverge badly on the IIR path, where the smoother contributes its whole time
        constant to the bound but only part of it to the half-response, and they nearly coincide for cFIR,
        which has no smoother at all. Reporting the bound for both and calling it "behind the signal" made
        the kinds look further apart than they are and made the number look arbitrary.

        Measured by running a burst at the band centre through a throwaway copy of the estimator, so it is
        the answer for this exact configuration rather than a formula that only holds for one of them.
        Cached on the configuration, since that is all it depends on.
        """
        if tgt.band is None:
            # A target with no band has no burst at a centre frequency to answer about — a coupling
            # index, a difference between two other targets. `Target.band` documents None as a state
            # rather than an absence, so this is the answer and not a guard against a bad caller.
            return None
        key = self.key_for(tgt)
        if key in self._response_cache:
            return self._response_cache[key]
        source = next(iter(self.specs), None) if tgt.source == PAIR else tgt.source
        spec = self.specs.get(source)
        if spec is None:
            return None
        try:
            metric = self.build_metric(spec, tgt.band, tgt)
        except ValueError:
            return None
        rate = spec.rate_hz
        n = int(min(24.0, max(8.0, 8 * (metric.latency_seconds + 0.5))) * rate)
        t = np.arange(n) / rate
        onset = n // 3
        x = np.where(np.arange(n) >= onset, np.sin(2 * np.pi * tgt.band.centre * t), 0.0)
        scale = 1.0 / max(spec.scale, 1e-12)
        readings: list[tuple[int, float]] = []
        for i in range(0, n - 7, 8):
            block = Block(spec.index, i, tuple(int(v * scale) for v in x[i:i + 8]), spec)
            for r in metric.feed(block):
                readings.append((i + 7, r.value))
        if len(readings) < 10:
            return None
        idx = np.array([a for a, _ in readings])
        val = np.array([b for _, b in readings])
        settled = val[idx > onset + int(0.7 * (n - onset))]
        final = float(np.median(settled)) if settled.size else 0.0
        if final <= 0:
            return None
        after = np.where(idx >= onset)[0]
        crossed = after[val[after] >= 0.5 * final]
        if not crossed.size:
            return None
        answer = float((idx[crossed[0]] - onset) / rate)
        self._response_cache[key] = answer
        return answer

    def rejection(self, channel: int) -> float:
        """How much of this channel is being thrown away right now, 0…1.

        A **channel** fact, not a target one — several targets can share a channel, so this belongs beside
        the trace where there is one of them per channel, and not on the cards where it would appear three
        times and read like a property of the band.
        """
        gate = self.gates.gates.get(channel)
        return gate.rate if gate is not None else 0.0

    def rejecting(self, channel: int) -> bool:
        return self.rejection(channel) > self.REJECTION_VISIBLE

