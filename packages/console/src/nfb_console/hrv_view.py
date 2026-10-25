"""HRV training toward a single spectral peak.

Not a coherence score and not the resonance-frequency protocol: "the goal is to get the HRV to show a **single
spectral peak at precisely 0.1 Hz**. You can do this simply by breathing in for 5 seconds and out for 5
seconds while remembering to watch your heart rate and only breath in when the heart rate is accelerating
and out when the heart slows down."

Two things follow from that sentence:

**The target is the shape of the spectrum**, so the screen shows the shape. A coherence score cannot say
which of the two ways of being wrong you are in — flat, or split. The instruments live in `heart_panel.py`,
shared with the CFC screen, because the same things matter whether HRV is the session or is running
beside the EEG.

**The pacer is locked to the heart, not to a clock.** "Only breathe in when the heart rate is accelerating"
makes the cue a function of the tachogram. A metronome is the fallback when the beat signal is poor, and the
control says which is running, because they are different instruments and one must not pass for the other.
Its cycle is settable over 6–30 s, because a resonant frequency is individual, commonly
4.5–7 breaths a minute, so a fixed 10 s paced everybody at six whatever suited them.

**And so is the frequency being scored, for the same reason.** The pacer's cycle was settable while the
number it was judged against stayed at 0.1 Hz, which paced somebody at their own rate and then marked them
down for it — the sharpness metric reads zero for a clean peak at 0.065 Hz, because ±0.02 Hz around 0.1 does
not contain it. `target_hz` moves both together: it is carried on the processor, stamped onto every spectrum,
and the metronome's cycle is derived from it. See `nfb.hrv.beats.TARGET_HZ`.

**And the breath is not the only way in.** The screen used to have a pacer and instruments and nothing else,
which made conscious breathing the whole protocol. Two separate paths now run beside it, and they are not the
same thing:

**And the breath is not the only way in.** The screen used to have a pacer and instruments and nothing else,
which made conscious breathing the whole protocol. It now drives the same four mechanisms every other screen
has — the master volume, the screen brightness, a discrete mark, and the music — so the room can carry the
signal with nothing instructed.

**What this screen adds is where the level comes from, not what carries it.** On an EEG
screen the level is targets scored against thresholds derived from your own first thirty seconds. Here it is a
*metric* chosen from eleven (`nfb.hrv.metrics`) and a *scaling* chosen from five (`normalise`), which is the
layer we had nothing for: several of these quantities are already bounded fractions where a threshold throws
away what 1.0 means, and three of them are better when low, which a threshold direction cannot express without
inverting its own derivation too. Those two rows sit above the shared Feedback pane; everything from the level
outward is unchanged.

Two wrong turns are recorded because both were shipped. The first bolted a coherence `Target` onto this screen
so it could use the mechanisms, which was the right *instinct* and the wrong seam — the threshold was doing
nothing the scaling does not do better. The second replaced the mechanisms with a bespoke "passive" layer of
three channels, which reimplemented the shadow, reimplemented the music as an inverted noise mask (that is
`music` on noise plus `sound.invert`, which is the Quiet Reward arrangement and already existed), dropped the
master volume entirely, and added a tint. The mechanisms are the general, calibrated, persisted machinery;
replacing them was the redundancy, not sharing them.

The one genuinely new rendering from that layer survives as music: `music.source = "tone"`, a bounded pitch
between two frequencies. It is kept because nothing else here carries a value with no reference point needed —
you hear which way a pitch moved without remembering where it was, and a volume cannot do that.

It is a **protocol you can choose**, not a gate. One school of practice trains HRV before any neurofeedback;
the method worked for years without it, so the software offers HRV rather than enforcing an order — and
offers it *alongside* the EEG.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import flet as ft
import flet.canvas as cv

from nfb.hrv import (DEFAULT_METRIC, HRV_METRICS, NARROW_HALF_RANGE_HZ, TARGET_RANGE_HZ,
                     metric_value)
from nfb.hrv.metrics import RSA_MIN_BEATS

from . import (chrome, cues, heart, heart_panel, music, normalise, plan_pane, prefs, shadow,
               signal_pane, sysvolume)
from . import theme as t
from .feedback import CONDITION, LEVEL, FeedbackSettings
from .inputs import HeartChoice
from .plan import SessionPlan
from .protocols import Protocol
from .targets import Target
from .widgets import help_button, labelled, num_box, panel, pill

HANDLES = ("HRV",)

SIDE_W = 330
RING_SIDE = 112         # the breathing cue, at the top of an inhale
RING_MIN = 56           # and at the bottom of an exhale
ARC_SIDE = 148          # the level arc drawn around it
ARC_WIDTH = 9

# **The two breath cues.** A higher tone to breathe
# in, a lower one to breathe out, which needs no learning: the pitch goes the way the breath goes. Both are
# `cues.tone_at`, a 160 ms sine, so they cannot be confused with the reward click (a few milliseconds of
# noise) or with the twelve inhibit alerts (noise as well, and pitched by band).
#
# It is a cue and not feedback: it says when, not how well, so it belongs with the reminder beep rather
# than with the mechanisms in the Feedback pane. A paced breath you have to *watch* is a paced breath that
# owns your eyes for the whole session, which defeats a protocol whose instruments you are supposed to be
# reading.
PACER_IN_HZ = 880.0
PACER_OUT_HZ = 660.0

# **The metronome's full cycle, and it was hardcoded at 10 s.** In for five and out for five is six breaths
# a minute, which is 0.1 Hz, which is the frequency the whole protocol aims the spectral peak at — so 10 s is
# the right default and was never the problem. The problem was that it could not be moved: a resonant
# frequency is individual, commonly 4.5–7 breaths a minute, and somebody whose own is 5.5 was being paced at
# 6 with no way to say so.
#
# 6–30 s, a plain user setting with no resonance search and no adaptation of any kind.
#
# **Half-second steps rather than whole seconds.** We generate the cue, so nothing stops us being finer —
# and at this end of the range a second is a coarse step: 10 s to 11 s moves the
# paced rate from 6.0 to 5.5 breaths a minute, which steps straight over 5.7 and 5.8. Half a second
# halves that.
CONDITION_LEVEL = 0.5
"""Where the level counts as met, for the click and the music's stop — the middle of the scaled range.

There is nothing to calibrate because the scaling already did it: whatever strategy is running has mapped the
metric onto 0…1, so the midpoint is "better than the middle of what this scale represents". On the EEG screens
that job belongs to a threshold derived from thirty seconds of the trainee's own signal, and here it would be
a second calibration on top of the first."""

OBSERVE_HZ = 10.0
"""How often the level is recomputed, matched to the draw loop's 0.1 s tick — it is the *display* rate, not
the metric's. The metric itself moves far more slowly: the spectrum is recomputed from a two-minute window and
the tachogram gains a value only once a beat, so most of these ticks push a repeat. That is fine for the two
things this figure is used for — smoothing the level over a wall-clock time constant, and giving the target a
nominal rate — and both would be wrong if they instead assumed a value per beat, whose spacing varies with the
heart rate being trained."""

WARMUP_GATE = 0.0
"""How much of a rolling strategy's window must exist before the signal drives anything. **None of it.**

This was a quarter — thirty seconds of the 120 s window — on the argument that the room should not follow
a range built from two readings. The argument is real and the remedy was wrong, and it is the third time
this screen has made the same trade: the spectrum once refused to exist below eighty seconds, then below
twenty, and the spectral metrics once waited four cycles of the target. Every one of them bought accuracy
with an opening minute in which somebody breathes and nothing answers.

A rolling scale with two readings in it is not lying — it reports the exact middle, which is what "I have
seen one value" honestly maps to, and it widens as the person's real range appears. Watching the level
firm up over the first half-minute is more use than a silent room followed by a level with no visible
relationship to what you were doing. The warm-up is still *reported* — the amber bar beside the readout,
and `Strategy.warmup` behind it — because saying "this is still settling" costs nothing and withholding
the number costs the opening of the session.

Kept as a named constant at zero rather than deleted, because the thing it must not silently become
again is a number."""

# **The stand-in covers absence, and only absence.** It used to cover roughness as well — the spectral
# metrics waited four cycles of the target and the rolling scale wanted a quarter of its window, so the
# opening minute of every session had no level of its own and this carried it. Both of those gates are
# gone (see `WARMUP_GATE`), so what is left for the stand-in is the few seconds in which the chosen metric
# has no value at all: fewer beats than a difference between beats needs.
#
# It stays rather than being deleted with them, because those seconds are real and they are the moment
# somebody presses play. And it is the *time-domain twin* of what the spectrum will measure: the
# peak-to-trough swing of the heart rate, which at resonance is the same behaviour seen without a Fourier
# transform. So it trains the same thing rather than something convenient, which is the only reason a
# stand-in is honest at all — and the screen says it is standing in, because a level that means something
# different from the metric named above it must not pass for that metric.
#
# Fixed range rather than rolling, so its values never enter the window the real metric is scaled by.
# 2 to 20 bpm covers the resting range: a flat, unmodulated tachogram swings a couple of bpm, and a deep
# resonant breath at six a minute swings fifteen to twenty.
STANDIN_METRIC = "rsa_amplitude"
STANDIN_RANGE_BPM = (2.0, 20.0)

PACER_PERIOD_S = 10.0
PACER_RANGE_S = (6.0, 30.0)
PACER_STEP_S = 0.5

TARGET_STEP_HZ = 0.005
"""The step the target moves in. Half a bin of a three-minute window, and fine enough to land on a rate
somebody found by trial: 0.065 Hz is 3.9 breaths a minute, and the next step either way is a tenth of a
breath a minute apart."""

# **Moving the target moves the metronome with it, and only in that direction.** They are one quantity
# said two ways — a frequency to breathe at — and a screen that let them disagree would be pacing somebody
# at six breaths a minute while scoring them for four. The metronome's cycle is therefore recomputed from
# the target whenever the target changes; the reverse is *not* wired up, because the cycle is clamped and
# snapped to half a second and feeding that back would drag the target off wherever it was set.


class HRVSession:
    """The heart, the clock, and the pacer. Not a `TrainingSession`.

    Deliberately not: that class is built around blocks of evenly spaced samples with an artifact gate and
    band metrics, none of which apply to an irregular beat stream. What it *does* share is the plan and its
    cues, and those are composed rather than inherited. The beats themselves come from a `HeartRunner`,
    which is the same object the CFC screen borrows.
    """

    def __init__(self, protocol: Protocol, choice: HeartChoice | None = None,
                 length_seconds: float | None = None, runner: heart.HeartRunner | None = None,
                 offered: HeartChoice | None = None):
        self.protocol = protocol
        self.plan = SessionPlan(length_seconds)
        self.cue = cues.play
        # A simulator by default *here only*: choosing this protocol means you want to train HRV, and
        # opening it to an empty screen because nothing is paired would be useless. On the CFC screen the
        # default is off, because there the heart is an addition and inventing beats beside a real EEG
        # would be a lie about what is on the subject. "Here only" is a promise `stop()` has to keep:
        # the runner is the app's, shared with CFC, and `attach` writes the choice onto it — so an
        # invented simulator has to be handed back switched off, or one visit to this screen leaves the
        # whole app on simulated beats. `_invented` is that memo.
        self._invented = choice is None
        self.choice = choice if choice is not None else HeartChoice(kind="simulator")
        # The app's own choice, kept even when it was off and we invented a simulator instead — because
        # the operator can go to the device screen mid-session and connect something, and that is the
        # normal way this screen gets a real source. Without it the session came back holding the
        # simulator it had invented, and `run` then handed that to the shared runner, which *switched a
        # connected sensor off*. See `adopt_offered`.
        self.offered = offered if offered is not None else choice
        self.heart = runner if runner is not None else heart.HeartRunner(self.choice)
        self.pacer_locked = True
        self.pacer_period = PACER_PERIOD_S
        self.pacer_sound = True
        # The target lives on the processor, which is the app's and shared with the CFC screen's copy of
        # these instruments — so setting it here sets it there too, which is right: it is the person's
        # resonant frequency and not this screen's opinion. Read back through `target_hz` rather than
        # mirrored into a second field, because two copies of one setting is how the plot and the
        # percentage come apart. See `target_hz` and `set_target_hz` below.
        self._breathing_in: bool | None = None
        self.finished = False

        # **The mechanisms are the shared ones**, and that is the point of them: sound is the master volume,
        # shadow is the screen, click is a discrete mark, music is the continuous sound — each with its own
        # floor, ceiling and calibration, each persisted, each configured through the pane every other
        # protocol uses.
        #
        # What this screen replaces is not the mechanisms but *where the level comes from*. On an EEG screen
        # it is targets against thresholds derived from your own first thirty seconds. Here it is a metric
        # chosen from eleven and a scaling chosen from five — which is the thing that was actually missing,
        # and it slots in underneath the mechanisms rather than beside them.
        #
        # A first attempt at this built a parallel "passive" layer with its own channels: a second
        # `DisplayShadow`, a second `NoisePlayer`, no master volume at all, and a bespoke tint. Three of those
        # were the mechanisms reimplemented — the noise mask in particular is `music` on noise with
        # `sound.invert`, which is the Quiet Reward arrangement and already existed.
        self.feedback = FeedbackSettings()
        self.display = shadow.DisplayShadow()
        self.output_volume = sysvolume.SystemVolume()
        self.music = music.MusicOutput()
        for name, mech in self.feedback.mechanisms.items():
            if name not in self.offers:
                mech.enabled = False
        self.feedback.music.enabled = True
        self.feedback.music.source = "tone"

        # **The whole feedback path, in three layers**: which metric, how it is scaled, and what carries it
        # — `nfb.hrv.metrics` and `normalise`.
        #
        # There used to be a second one beside it: a coherence `Target` with a calibrated threshold driving
        # the EEG screens' mechanisms through `SignalPane`. It was deleted, and the deletion is the point.
        # Two tabs configured the feedback, two `DisplayShadow` objects wrote to the one brightness knob, two
        # `NoisePlayer`s could sound at once, and picking a different metric here left that one still training
        # coherence. What the threshold model uniquely offered — a boolean condition, the click and the alert,
        # calibration from your own first thirty seconds — is either unsuitable for a measure windowed over two minutes or is
        # what `zscore` and `rolling_minmax` already do, without a hidden calibration phase.
        self.metric = DEFAULT_METRIC
        # **Which way you are training it, and it is yours to set.** It was the metric's `invert` flag and
        # nothing else — a property of the number, unreachable from the screen — which meant the one thing an
        # operator most obviously wants to say about a quantity could not be said. LF/HF is the case that
        # makes it plain: inverted here because sympathetic drive is the usual concern, and exactly backwards
        # for somebody training a flat, over-parasympathetic profile. The metric supplies the *default*, the
        # same way a protocol's bands arrive as increase or decrease on the amplitude screen and can be
        # cycled there.
        self.direction = HRV_METRICS[DEFAULT_METRIC].default_direction
        self.strategy_id = normalise.DEFAULT_STRATEGY
        self.strategy_params: dict = {}
        self._strategy = normalise.build(self.strategy_id)
        self.raw_metric: float | None = None
        # A stand-in for the opening minute, on its own fixed scale. See STANDIN_METRIC.
        self._standin = normalise.build("fixed_range", low=STANDIN_RANGE_BPM[0],
                                        high=STANDIN_RANGE_BPM[1])
        self.standing_in = False
        self._signal: float | None = None
        self._level = 0.0
        self._level_at: float | None = None
        self._last_click: float | None = None
        # One target, and it exists for three specific jobs rather than to hold a threshold: it is what the
        # shared pane draws a contribution row for, what tells that pane data is arriving, and what the click
        # asks. Its value is the **already-normalised** signal and its threshold is the middle of that range,
        # so there is nothing to calibrate — the scaling did that — and `_settled` says so.
        self.signal_target = Target(0, None, "increase", threshold=CONDITION_LEVEL, mode="fixed",
                                    name="signal", unit="")
        self.signal_target._settled = True
        self.targets = [self.signal_target]
        self._accumulated = 0.0
        self._resumed_at: float | None = None
        self.paused = True
        self._stop = False
        self.redraw: Callable[[], None] = lambda: None

    # --- what the shared Feedback pane reads ------------------------------------------------
    #
    # These are the declarations `TrainingSession` carries as class attributes. They are repeated rather than
    # inherited because this is not one — an RR interval is not a sample and none of that class's block,
    # spec or artifact machinery applies — and repeating nine defaults is a smaller price than making a
    # beat stream pretend to be a sample stream.
    offers = ("sound", "shadow", "click", "music")
    """All four, and the click is deliberately among them: an earlier version refused it on the grounds that
    a coherence measure wanders across its threshold and would fire in clusters. That is what
    `Click.interval_seconds` is for, and it is the same flicker every screen's condition has."""
    music_sources = ("tone", "crossfade", "noise", "shepard", "file")
    """`tone` first, because a bounded pitch is the one source that carries a value with no reference point
    needed — you hear which way it moved without remembering where it was, which no volume can do. `eeg` is
    absent: it sonifies a sample stream and there is none here."""
    combines = ("weighted",)
    """One target, so every rule reduces to the same thing. Stated rather than offered."""
    pinned_feedback = frozenset()
    arrangements: tuple[str, ...] = ()
    state_ends = ("scattered", "coherent")
    tone_bias = None
    unbiased = False
    gates = None
    scores_bands = False
    """The level comes from a metric and a scaling rather than from targets against thresholds, so the four
    steps that build a score have nothing to act on here — see `TrainingSession.scores_bands`."""
    sustains_reward = False
    """No sustained reward event here, so the Click panel's "doubles after" row is
    absent — see `TrainingSession.sustains_reward`."""
    hold_choices = None
    """No sample-and-hold choice here: there is none for this protocol and a beat is already an
    event rather than a stream — see `TrainingSession.hold_choices`."""
    hold_seconds = 0.0
    tunes_envelope = False
    """No band estimator takes `envelope_seconds` here, so the SIGNAL panel's band-averaging row is absent —
    see `TrainingSession.tunes_envelope`. This class does not inherit `TrainingSession`, so every capability
    flag the shared panes read has to be declared, and a missing one is an `AttributeError` in a Flet event
    handler rather than a wrong default."""
    conditions_signal = False
    """No sample-artifact gate: that is amplitude rejection on blocks of evenly spaced samples. Beats are
    filtered too — `nfb.hrv.ArtifactFilter` — and the reject fraction is reported beside the tachogram, where
    it can be read against the signal it applies to."""

    def level_note(self) -> tuple[str, str] | None:
        """Why there is no level, when the beats are arriving perfectly well.

        One honest reason left, and it is not "no channel is delivering signal", which is what the shared
        pane said in red while the tachogram drew beside it: there are not yet enough beats for a
        difference between beats to exist. The other used to be a rolling scale that had not filled a
        quarter of its window — that one is gone, because the level no longer waits for it. See
        `WARMUP_GATE`.
        """
        if self.error is not None or self.heart.stale or not self.heart.ready:
            return None                       # genuinely about the source; the pane names it
        if self.raw_metric is None:
            beats = self.state.beats if self.state is not None else 0
            return (f"counting beats · {beats} of {RSA_MIN_BEATS}",
                    "variability is a difference between beats, so it needs a few of them to exist at "
                    "all — four, which is the arithmetic floor and not a settling time")
        return None

    def dead_channels(self) -> list[str]:
        if self.error is not None or self.heart.stale:
            return ["the beat source"]
        return [] if self.heart.ready else ["the beat source"]

    @property
    def spectrum_ready(self) -> bool:
        state = self.state
        return bool(state is not None and state.spectrum.ready)

    # --- surviving a stop and a relaunch ----------------------------------------------------

    def adopt(self, previous: "HRVSession") -> None:
        """Take over what the operator configured on the session being replaced.

        **This was missing entirely, and `Console.pick` calls it**, so pressing stop on this screen and
        re-entering the protocol raised `AttributeError`. Same line as `TrainingSession.adopt` draws: the
        session is the history and the configuration is not, so the plan, the pacer and the whole feedback
        arrangement come across while the beats and the normaliser's window do not — those were measured on
        a session that has ended.
        """
        if previous is self or previous.protocol.code != self.protocol.code:
            return
        self.plan = previous.plan
        self.plan.rewind()
        self.pacer_locked = previous.pacer_locked
        self.pacer_period = previous.pacer_period
        self.pacer_sound = previous.pacer_sound
        # The target needs no line here — it lives on the shared processor, which outlives both sessions.
        # Listed anyway, because "everything the operator configured" is the promise above and a reader
        # checking it against this list would otherwise conclude the target was dropped.
        self.feedback = previous.feedback
        self.metric = previous.metric
        self.direction = previous.direction
        self.set_strategy(previous.strategy_id, **previous.strategy_params)

    def calibration(self) -> dict:
        """What `prefs` keeps between launches — the mechanisms' own whitelist, plus the pacer.

        The pacer's period is calibration in the same sense a volume is: it is the person's own resonant
        rate, wrong differently for everyone and set once. **So is the target**, and more so — it is the
        same physical fact stated in hertz, found by hunting for it over a session or two, and asking
        somebody to re-find it every launch would be the same failure as losing a volume. The metric and
        the scaling are deliberately not here — they are what is being trained, the same category as a
        band or a threshold, and prefs' own docstring is why those never persist.
        """
        return {**prefs.capture(self.feedback), "pacer_period": self.pacer_period,
                "pacer_locked": self.pacer_locked, "pacer_sound": self.pacer_sound,
                "target_hz": self.target_hz, "target_half_hz": self.target_half_hz}

    def restore_calibration(self, values: dict) -> int:
        used = prefs.apply(self.feedback, values)
        # Before the pacer, because setting the target sets the pacer's period from it — restored in the
        # other order, a stored period would be overwritten by one derived from the target.
        target = values.get("target_hz")
        half = values.get("target_half_hz")
        if isinstance(target, (int, float)) and not isinstance(target, bool):
            self.set_target_hz(float(target),
                               float(half) if isinstance(half, (int, float))
                               and not isinstance(half, bool) else None)
            used += 1
        period = values.get("pacer_period")
        if isinstance(period, (int, float)) and not isinstance(period, bool):
            self.set_pacer_period(float(period))
            used += 1
        locked = values.get("pacer_locked")
        if isinstance(locked, bool):
            self.pacer_locked = locked
            used += 1
        sound = values.get("pacer_sound")
        if isinstance(sound, bool):
            self.pacer_sound = sound
            used += 1
        return used

    # --- where the level comes from ------------------------------------------------------------------

    @property
    def metric_spec(self):
        return HRV_METRICS[self.metric]

    def set_metric(self, metric_id: str) -> None:
        """Point the level at a different quantity, and rebuild the normaliser.

        Rebuilt, not kept: a rolling strategy's window holds values in the *old* metric's units, so carrying
        it across would normalise milliseconds against a history of dimensionless ratios and the signal would
        sit pinned at an end until the window aged out. The default strategy moves with the metric too, since
        each one names which suit it.

        **So does the direction**, and it has to: "decrease" carried from the distance-from-ideal onto RMSSD
        would train somebody toward *less* variability with nothing on the screen having changed but the
        metric's name. Each one ships the aim it is usually trained in, and the row says when you have moved
        off it.
        """
        if metric_id not in HRV_METRICS:
            return
        self.metric = metric_id
        self.direction = HRV_METRICS[metric_id].default_direction
        wanted = HRV_METRICS[metric_id].strategies
        if wanted and self.strategy_id not in wanted:
            self.strategy_id = wanted[0]
        self.set_strategy(self.strategy_id)

    def set_direction(self, direction: str) -> None:
        """Train the metric up or down. Resets the scaling's history, because it does not carry over — the
        rolling window is fine, but the level is about to mean the opposite and the on-target history behind
        the readout was measured under the other aim."""
        if direction not in ("increase", "decrease") or direction == self.direction:
            return
        self.direction = direction
        self._signal = None
        self.signal_target.reset_stats()
        self.signal_target._settled = True

    @property
    def inverts(self) -> bool:
        """Whether the level is `1 - signal`: training the metric down."""
        return self.direction == "decrease"

    def set_strategy(self, strategy_id: str, **params) -> None:
        if strategy_id not in normalise.STRATEGIES:
            return
        self.strategy_id = strategy_id
        self.strategy_params = dict(params)
        self._strategy = normalise.build(strategy_id, **params)
        self.warm_from_history()
        self.signal_target.reset_stats()
        self.signal_target._settled = True

    def warm_from_history(self) -> None:
        """Fill a freshly built scale from the beats already in hand.

        **Changing the metric used to silence the feedback for half a minute.** The normaliser has to be
        rebuilt — a rolling window holds values in the old metric's units, and carrying it across would
        scale milliseconds against a history of dimensionless ratios — but rebuilding it left it empty,
        and an empty scale reports the exact middle until real values widen it. The level no longer waits
        for that (see `WARMUP_GATE`), so this is about the level being *right* rather than about it
        existing: the beats are still here, three minutes of them, and a metric is a pure function of a
        state, so the new scale is handed the same values it would have had if it had been running all
        along.

        History is mapped onto the session clock so the newest replayed point lands at now, which keeps
        the strategy's own windowing monotonic when live values start arriving on top.
        """
        states = self.heart.processor.replay(step_s=1.0)
        if not states:
            self._signal = None
            return
        newest = states[-1].t
        now = self.elapsed
        for state in states:
            value = metric_value(self.metric_spec, state)
            if value is not None:
                self._strategy.push(value, now - (newest - state.t))
        self._signal = None

    @property
    def strategy(self) -> normalise.Strategy:
        return self._strategy

    @property
    def level_source(self) -> tuple[str, str]:
        """What the readout says the level is, in place of the combining rule's wording.

        It read "each target's score — rewards and inhibits averaged separately, then mixed" above "half = the
        mix is favourable", on a screen with one target, no inhibits and a level that comes from neither.
        """
        spec = self.metric_spec
        scale = normalise.STRATEGIES[self.strategy_id].label
        aim = "less" if self.inverts else "more"
        return (f"{aim} {spec.label}, on a {scale} scale",
                f"half = the middle of that scale — {self._strategy.describe()}")

    @property
    def level(self) -> float:
        """The 0…1 the mechanisms follow. From the metric and the scaling, not from a threshold."""
        return self._level

    @property
    def signal(self) -> float | None:
        """The normalised value, or None while the metric has nothing to say. **Not zero** — zero is a
        reading, and a channel handed it would drive itself to its floor and read as a bad result."""
        return self._signal

    @property
    def active_targets(self) -> list[Target]:
        return self.targets if self._signal is not None else []

    def channel_targets(self, channel: str) -> list[Target]:
        return self.active_targets

    @property
    def state_balance(self) -> float | None:
        """Where a crossfade sits — the normalised signal, which is already bounded 0…1."""
        return self._signal

    def observe(self) -> None:
        """Metric, scaling, then — if the session is running — the shared mechanisms.

        The inversion is applied to the *normalised* value rather than to the metric, so a strategy's window
        still sees the quantity in its own units and its own direction — inverting first would turn a rolling
        minimum into a rolling maximum and make the pane's description of it wrong.

        Measuring while paused is deliberate: the scale wants history, the readout is worth watching before
        you start, and neither touches the room. Only `_render` does, and it is skipped.
        """
        # A frozen snapshot is not a reading. `state` holds the last beat for ever, so without this
        # gate a dead strap kept driving the volume, the shadow and the clicks indefinitely — and once
        # the rolling window filled with the one repeated value, the level parked at 0.5 unexplained.
        if self.error is not None or self.heart.stale:
            self.raw_metric = None
            self._signal = None
            self.release_outputs()
            return
        raw = metric_value(self.metric_spec, self.state)
        # **The chosen metric's scale is fed whenever it has a value, whether or not it is driving.** It
        # was fed only once it was already usable, so the stand-in carried the opening and then handed
        # over to a scale with nothing in it — which gated the level and put "calibrating" on the screen
        # in the middle of a session that had been running for a minute. Warming and driving are separate
        # things and only one of them touches the room.
        scaled = self._strategy.push(raw, self.elapsed) if raw is not None else None
        # **The stand-in covers absence, not roughness.** It used to hold the level until the scale was
        # a quarter full, on the reasoning that a scale built from two readings is not worth following.
        # That is a settling time, and this screen has now spent three rounds learning that a settling
        # time is not a reason to withhold a number: the chosen metric drives the room from the first
        # value it has, and the warm-up bar says it is still firming up. See WARMUP_GATE.
        self.standing_in = scaled is None or self._strategy.warmup < WARMUP_GATE

        if not self.standing_in:
            self.raw_metric = raw
            self._signal = (1.0 - scaled) if self.inverts else scaled
        else:
            # The stand-in: the swing of the heart rate, which is what the spectrum will measure without
            # the transform, on its own fixed scale. See STANDIN_METRIC.
            stand = metric_value(HRV_METRICS[STANDIN_METRIC], self.state)
            self.raw_metric = stand if stand is not None else raw
            if stand is None:
                # Neither has anything: the opening seconds, before four beats exist.
                self._signal = None
                self.release_outputs()
                return
            self._signal = self._standin.push(stand, self.elapsed)

        self.signal_target.observe(self._signal, OBSERVE_HZ)
        self._render(self._signal)

    def _render(self, signal: float) -> None:
        """Smooth the level and hand it to the mechanisms. Split out so the stand-in metric above takes
        the identical path from here on: what is being scaled differs, what the room does with the result
        does not.

        The one place that touches the room, so it is the one place that asks whether the session is
        running. `toggle_pause` has already released the outputs by the time this is skipped.
        """
        if not self.running:
            return
        tau = max(0.0, self.feedback.smoothing_seconds)
        now = self.elapsed
        if tau <= 0:
            self._level = signal
        else:
            # Session time, not a nominal tick: the loop is a sleep plus a frame's work, so a fixed
            # `1/OBSERVE_HZ` step always smooths harder than configured by whatever the frame costs —
            # the same defect `TrainingSession._advance_level` and the shadow both carried. `elapsed`
            # rather than the wall clock because it is the clock the strategies already run on: it
            # freezes across a pause, and it follows a replayed or accelerated session instead of the
            # room's.
            dt = (now - self._level_at) if self._level_at is not None else 1.0 / OBSERVE_HZ
            if dt > 0:
                self._level += (1.0 - math.exp(-dt / tau)) * (signal - self._level)
        self._level_at = now
        self._drive_outputs()

    @property
    def splitting(self) -> bool:
        """Whether the spectrum currently shows the gas-and-brake failure."""
        state = self.state
        return bool(state is not None and state.spectrum.ready and state.spectrum.split)

    def _drive_outputs(self) -> None:
        """Hand the level to the mechanisms. The same four every other screen has.

        Written here rather than inherited because this is not a `TrainingSession` — but it is the same eight
        lines, and if a third screen ever needs them they should move rather than be copied again.

        **The split fails the condition.** Measured across all eleven metrics: RMSSD and pNN50 read
        *better* during the deliberate split than at perfect resonance, and both are trained upward — so
        without this, the beginner metrics pay clicks for the exact failure the spectrum panel is
        warning about in red. The level is left alone: it is a continuous display and the operator can
        see it against the warning, where a click is an event that means "earned" and the split is the
        one state in this protocol where nothing is. CFC's inhibit bins are the same interlock on the
        EEG side, which is where the shape of this comes from.
        """
        met = self.feedback.on_target(self.channel_targets(CONDITION)) and not self.splitting
        sh = self.feedback.shadow
        follows = self.state_balance if sh.follows == "state" else self._level
        self.display.apply(sh.enabled and sh.supported, sh.screen, *sh.ends(),
                           sh.map(follows if follows is not None else self._level))
        snd = self.feedback.sound
        self.output_volume.apply(snd.enabled, *snd.ends(), snd.map(self._level))
        self.music.apply(self.feedback.music, hold=self.feedback.music.holds(met),
                         balance=self.state_balance, level=self._level)
        self._maybe_click(met)

    def _maybe_click(self, met: bool) -> None:
        """The discrete mark, on the same interval rule as everywhere else.

        It is offered here — an earlier version of this screen refused it, on the grounds that a coherence
        measure wanders across a threshold and would fire in clusters. That is what `interval_seconds` is
        for, and it is the same flicker every other screen's condition has; the gate does not care where the
        flicker came from.
        """
        click = self.feedback.click
        if not click.enabled or not met:
            if not click.enabled:
                self._last_click = None
            return
        now = time.monotonic()
        if self._last_click is None or now - self._last_click >= click.interval_seconds:
            self._last_click = now
            self.cue(click.sound, click.volume)

    def release_outputs(self) -> None:
        self.display.release()
        self.output_volume.release()
        self.music.release()

    @property
    def error(self) -> str | None:
        return self.heart.error

    @property
    def state(self):
        return self.heart.state

    # --- transport, the contract the toolbar expects -----------------------------------------

    @property
    def elapsed(self) -> float:
        running = (time.monotonic() - self._resumed_at) if self._resumed_at is not None else 0.0
        return self._accumulated + running

    @property
    def running(self) -> bool:
        return not self.paused

    @property
    def length(self) -> float | None:
        return self.plan.length_seconds

    def toggle_pause(self) -> None:
        """Pause and resume, **and let go of every output while paused** — `TrainingSession`'s rule,
        reimplemented here because this session does not inherit it. The draw loop skips `observe()`
        while paused, which stops the outputs being *updated*, not being *held*: the display stayed
        dimmed, the volume stayed where the signal left it, and the tone sang on. Resuming needs no
        restart — every mechanism owns its take/follow/release lifecycle inside `apply`."""
        if self.paused:
            self.paused = False
            self._resumed_at = time.monotonic()
        else:
            self._accumulated = self.elapsed
            self._resumed_at = None
            self.paused = True
            self.release_outputs()

    def tick(self) -> None:
        if not self.running:
            return
        self.cue_breath()
        elapsed = self.elapsed
        for cue in self.plan.due(elapsed):
            self.cue(cue)
        if self.plan.complete(elapsed):
            self.finished = True
            self._accumulated = self.plan.length_seconds
            self._resumed_at = None
            self.paused = True
            # A timed session that ends must not leave the room held any more than a paused one.
            self.release_outputs()

    def adopt_offered(self) -> None:
        """Take the app's source if it has become a real one since this session started.

        Called on the way in and on the way back from the device screen, which are the two moments the
        answer can have changed. A simulator invented here is a stand-in for having nothing chosen, so
        anything actually chosen outranks it — and the memo that hands the runner back switched off is
        dropped with it, since it is no longer ours to hand back.
        """
        if self._invented and self.offered is not None and self.offered.on:
            self.choice = self.offered
            self._invented = False

    def run(self) -> None:
        """Reading beats is the runner's own thread, so this only has to ask for it."""
        self.adopt_offered()
        self.heart.attach(self.choice)

    def stop(self) -> None:
        self._stop = True
        self.paused = True
        # Keep what was set by ear, as every other screen does on the way out.
        prefs.remember(self.protocol.code, self)
        self.release_outputs()
        self.heart.release()
        if self._invented:
            # The simulator was this screen's default, never the operator's choice. Hand the shared
            # runner back switched off with the simulated state cleared — CFC borrows it, and drawing
            # invented beats beside a real EEG is exactly what its own docstring forbids.
            self.heart.choice = HeartChoice(kind="off")
            self.heart.state = None
            self.heart.error = None
            self.heart.seen_at = None
            self.heart.processor.reset()

    # --- going to the device screen and coming back ------------------------------------------
    # The chip on the shared toolbar calls these on whatever session is up, and this one not having
    # them was an AttributeError inside a Flet event handler — a click that silently did nothing.
    # There is no exclusive device mode to give up here (the heart is Bluetooth, not the amplifier),
    # so suspend only has to stop holding the room and rearm only has to carry on.

    def suspend(self) -> None:
        """Stop driving the outputs and release them, keeping the session alive to come back to."""
        if not self.paused:
            self.toggle_pause()             # which releases the outputs
        self.release_outputs()              # and again, for a suspend that arrives already paused

    @property
    def suspended(self) -> bool:
        return not self._stop

    def rearm(self) -> None:
        """Nothing to rebuild: the heart runner never stopped and the clock stayed frozen where
        `suspend` left it. The operator resumes when ready, exactly as after a pause."""

    # --- the breathing cue -----------------------------------------------------------------

    def breathe_in(self) -> bool:
        """Whether the person should be inhaling now.

        Locked to the heart when there are beats to lock to, and to the clock when there are not. The
        locked version is the instruction as written; the clock version is what remains when contact is
        poor, and it must not silently pretend to be the same thing.
        """
        if self.pacer_locked and self.heart.ready:
            return self.state.accelerating
        half = self.pacer_period / 2.0
        return (self.elapsed % self.pacer_period) < half

    def breath_phase(self) -> float | None:
        """How far through the current half-cycle, 0…1, for the ring to breathe with.

        Only the metronome has one. The heart-locked pacer turns on an *event* — the heart starting to
        slow — and how far through the current phase you are is precisely what it does not know, so a
        ring that swelled smoothly there would be animating a guess. It holds still instead, and the note
        under it already says which of the two is running.
        """
        if self.pacer_locked and self.heart.ready:
            return None
        half = self.pacer_period / 2.0
        return (self.elapsed % half) / half

    def cue_breath(self) -> None:
        """Sound the turn, once, whichever pacer is running. Called every draw; only a change makes a
        sound, and the first call after a start sounds the phase you are entering."""
        now = self.breathe_in()
        if now == self._breathing_in:
            return
        self._breathing_in = now
        if self.pacer_sound:
            self.cue(cues.tone_at(PACER_IN_HZ if now else PACER_OUT_HZ))

    def pacer_note(self) -> str:
        if self.pacer_locked and self.heart.ready:
            return "following the heart — in as it accelerates, out as it slows"
        if self.pacer_locked:
            return "waiting for beats before it can follow the heart"
        return (f"a metronome on a {self.pacer_period:g} s cycle — "
                f"{60.0 / self.pacer_period:.1f} breaths a minute, {1.0 / self.pacer_period:.3f} Hz — "
                f"not the instruction, but it survives poor contact")

    def set_pacer_period(self, seconds: float) -> None:
        """Clamped into the range and snapped to `PACER_STEP_S`, not out of it — a control that cannot leave
        its own default is indistinguishable from a disabled one."""
        lo, hi = PACER_RANGE_S
        snapped = round(float(seconds) / PACER_STEP_S) * PACER_STEP_S
        self.pacer_period = float(min(hi, max(lo, snapped)))

    # --- the target frequency ---------------------------------------------------------------

    @property
    def target_hz(self) -> float:
        """The frequency the spectral metrics and the marked band are scored against.

        One copy, held by the processor, because that is what stamps it onto every spectrum the screen
        draws a number off. See `nfb.hrv.beats.TARGET_HZ` for why 0.1 is a default and not a fact."""
        return self.heart.processor.target_hz

    @property
    def target_half_hz(self) -> float:
        """Half the width of the band around it that `target sharpness` integrates."""
        return self.heart.processor.narrow_half_hz

    @property
    def target_breaths_per_min(self) -> float:
        return self.target_hz * 60.0

    def set_target_hz(self, hz: float, half_hz: float | None = None) -> None:
        """Move the target, and move the metronome with it.

        **Nothing is reset and nothing needs to be.** The spectrum is rebuilt from the buffered beats on
        the next one — about a second — so the plot, the percentage and the metrics catch up together
        without a gap. The normaliser is deliberately *left alone*: its window holds the same metric in
        the same units, and the values in it were honestly measured, so rebuilding it would silence the
        feedback for thirty seconds to remove a bias that ages out on its own in two minutes.
        """
        self.heart.processor.set_target(hz, half_hz)
        self.set_pacer_period(1.0 / self.target_hz)

    def target_note(self) -> str:
        """What the target row says under itself: one fact, in the unit a person can act on, where hertz
        is the unit the spectrum is drawn in. Everything else is behind the panel's `?`.

        It also carried "62 s to measure" — how long the spectral metrics used to wait at that target.
        Nothing waits now, so the figure would be describing a rule that no longer exists."""
        return f"{self.target_breaths_per_min:.1f} breaths a minute"


BREATHE_HELP = (
    "**What you see.** A target frequency, shown in Hz and in breaths a minute, and a band around it. "
    "You can set both.\n\n"
    "**What the target means.** 0.1 Hz, six breaths a minute, is the usual goal for this training. Your "
    "own resonant rate can differ: people range from 4.5 to 7 breaths a minute, with taller adults "
    "nearer the slow end. Set the target to your own rate instead of the usual goal.\n\n"
    "**What moves with the target.** On the spectrum, the marked band, the line and the percentage "
    "under it follow the target. On the Feedback tab, three metrics follow it: target sharpness, peak "
    "proximity and distance from ideal. The metronome's cycle follows it too.\n\n"
    "**What stays fixed.** Coherence, the coherence ratio and LF/HF use fixed published bands, and do "
    "not move with the target. This keeps them comparable across sessions. Coherence finds the peak on "
    "its own, so use it to search for your own rate.\n\n"
    "**How to find your rate.** Set the **pacer** to **metronome**. Set the cycle to "
    f"{PACER_PERIOD_S:g} s, which is 6 breaths a minute. Make the cycle longer in steps of "
    f"{PACER_STEP_S:g} s, and breathe with it at each step. "
    "Watch for the point where the tachogram's swing and the spectral peak are both "
    f"largest. Set the **target** to that rate. The target box moves in steps of {TARGET_STEP_HZ:g} Hz. "
    "Every number on the screen updates from the beats already recorded, "
    "and it settles as more beats arrive.\n\n"
    "**The band** sets how exact a match must be: the sharpness metric scores only the power inside it. "
    "±0.02 Hz is about one breath a minute either side.\n\n"
    "**The pacer** tells you when to breathe, not how fast. 'Follow the heart' matches the written "
    "instruction: breathe in as the heart rate rises, out as it falls. It needs beats to follow, so it "
    "switches to the metronome, shown in amber, when there are none. The metronome's cycle follows the "
    "target, and you can move the cycle on its own without moving the target.\n\n"
    f"**The sound** is a tone at each turn: {PACER_IN_HZ:g} Hz going in, {PACER_OUT_HZ:g} Hz going out. "
    "You can follow the pitch instead of watching the ring."
)

SIGNAL_HELP = (
    "**What you see.** Three rows: the quantity to feed back, the direction to train it in, and how it "
    "is scaled. Together they set the level. The volume, the screen, the click and the music all follow "
    "that level.\n\n"
    "**What the quantity means.** There are eleven quantities to choose from. Target sharpness scores a "
    "single spectral peak at the target frequency. It penalises a peak outside its band as hard as no "
    "peak at all. Set the target to your own resonant rate first, to get the sharpest score. Coherence "
    "finds the peak instead of assuming where it is, so use it while you search for your rate. The "
    "variability metrics show how much variability there is, not where it sits in the spectrum. They "
    "give a steady reading before the spectrum has settled, and a weaker one after. The RSA swing needs "
    f"no spectrum, so it gives a reading after {RSA_MIN_BEATS} beats.\n\n"
    "**What to do.** Pick a quantity, then set the direction to increase or decrease. Each metric starts "
    "at the direction it is usually trained in. Three start at decrease — distance from the ideal "
    "spectrum, the LF/HF ratio, and heart rate — because a higher value there is worse. LF/HF also "
    "starts inverted, for high sympathetic drive. Set it to increase instead if you are training up from "
    "a flat, low-arousal profile. Changing the quantity resets the direction to its own default.\n\n"
    "**What the scaling means.** A metric has its own unit, and a mechanism needs a value from 0 to 1. "
    "The scales **recent range** and **standard deviations** follow the last one or two minutes. Any "
    "metric gives a usable reading at once. Steady improvement then reads as average, because the scale "
    "rises with you. The scale **a fixed range** stays the same across sessions instead, so today "
    "compares with last week. You must set the range yourself. The bar beside the reading shows the "
    "warm-up. Amber means the scale still has little history.\n\n"
    "**The condition.** The click and the music's stop trigger when the level passes the middle of the "
    "scaled range."
)


class SignalSource:
    """Two rows: which quantity, and how it is scaled into the 0…1 the mechanisms follow.

    A small panel above the shared Feedback pane rather than a tab of its own. It replaces the part of that
    pane which does not apply here — band averaging and a threshold-derived score — and leaves everything
    from the level outward exactly as every other screen has it.

    It was briefly a whole parallel system: four panels on their own tab, with channel selectors that
    duplicated the shadow and the music, and each selected item's two paragraphs of prose inlined. The prose is
    behind the one `?` now, which is where this repo puts that material everywhere else.
    """

    def __init__(self, page: ft.Page, session: "HRVSession"):
        self.page = page
        self.session = session
        self.rows = ft.Column(spacing=12)
        self.readout = ft.Text("", size=t.BODY, color=t.TEXT)
        self.note = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.warmup = ft.Container(width=0, height=5, border_radius=3, bgcolor=t.ACCENT,
                                   left=0, bottom=0)
        bar = ft.Stack([ft.Container(width=140, height=5, border_radius=3, bgcolor=t.SURFACE_HI,
                                     left=0, bottom=0), self.warmup], width=140, height=5)
        self.control = panel(
            "SIGNAL SOURCE",
            ft.Row([self.readout, ft.Container(width=14), bar],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Container(height=4), self.note,
            ft.Container(height=14), self.rows,
            help=help_button(page, "Signal source", SIGNAL_HELP))
        self.rebuild()

    def rebuild(self) -> None:
        session = self.session
        spec = session.metric_spec
        usual = spec.default_direction
        self.rows.controls = [
            labelled("feed back", self._chooser(spec), note=spec.description, width=76),
            # Its own row, above the scaling and below the metric, because it is a statement about neither:
            # the metric is the quantity, the scaling is the shape, and this is what you are asking for.
            labelled("aim to",
                     *[pill(k, session.direction == k,
                            (lambda k_=k: lambda _e: (session.set_direction(k_), self.rebuild()))())
                       for k in ("increase", "decrease")],
                     note=(f"the usual aim for {spec.label}" if session.direction == usual else
                           f"{spec.label} is usually trained to {usual}"),
                     width=76),
            labelled("scaled by",
                     *[pill(normalise.STRATEGIES[k].label, session.strategy_id == k,
                            (lambda k_=k: lambda _e: (session.set_strategy(k_), self.rebuild()))())
                       for k in normalise.STRATEGIES],
                     *self._params(),
                     note=normalise.STRATEGIES[session.strategy_id].description, width=76),
        ]
        self.page.update()

    def _chooser(self, spec) -> ft.Control:
        """A menu rather than eleven pills: pills would wrap to four rows, and the labels are not
        self-explanatory, so each needs its one-line description beside it as you consider it."""
        return ft.PopupMenuButton(
            content=ft.Container(
                content=ft.Row([ft.Text(spec.label, size=t.BODY, color=t.TEXT),
                                ft.Text("▾", size=9, color=t.FAINT)], spacing=6),
                padding=ft.Padding(10, 6, 10, 6), border_radius=6,
                border=ft.Border.all(1, t.BORDER)),
            items=[ft.PopupMenuItem(
                content=ft.Column([ft.Text(m.label, size=t.BODY),
                                   ft.Text(m.description, size=t.SMALL, color=t.MUTED)], spacing=1),
                on_click=(lambda i=m.id: lambda _e: self._pick(i))())
                for m in HRV_METRICS.values()])

    def _params(self) -> list:
        """Only the knobs the selected scaling has. One that does not apply is absent rather than inert."""
        strategy = self.session.strategy
        out: list[ft.Control] = []
        if isinstance(strategy, (normalise.ZScore, normalise.RollingMinMax)):
            out += [ft.Text("over", size=t.SMALL, color=t.FAINT),
                    num_box(strategy.window_s, lambda v: self._set(window_s=v),
                            suffix="s", step=15, low=10, high=600)]
        if isinstance(strategy, normalise.ZScore):
            out += [num_box(strategy.clamp_k, lambda v: self._set(clamp_k=v),
                            suffix="sd", step=0.5, low=0.5, high=5.0)]
        if isinstance(strategy, normalise.FixedRange):
            out += [num_box(strategy.low, lambda v: self._set(low=v), step=0.1),
                    ft.Text("to", size=t.SMALL, color=t.FAINT),
                    num_box(strategy.high, lambda v: self._set(high=v), step=0.1)]
        if isinstance(strategy, (normalise.Saturating, normalise.Tanh)):
            out += [ft.Text("at", size=t.SMALL, color=t.FAINT),
                    num_box(strategy.scale, lambda v: self._set(scale=v), step=0.1, low=0.01)]
        return out

    def _pick(self, metric_id: str) -> None:
        self.session.set_metric(metric_id)
        self.rebuild()

    def _set(self, **params) -> None:
        session = self.session
        session.set_strategy(session.strategy_id, **{**session.strategy_params, **params})
        self.rebuild()

    def refresh(self) -> None:
        session = self.session
        spec = session.metric_spec
        raw, signal = session.raw_metric, session.signal
        if session.standing_in and raw is not None:
            # **Never let the stand-in pass for the metric named above it.** The number here is a swing in
            # bpm and the row above says "coherence"; showing one under the other's name is the single way
            # this arrangement could mislead, so the label is the stand-in's own and the note says why it
            # is on screen.
            standin = HRV_METRICS[STANDIN_METRIC]
            self.readout.value = (f"{raw:.3g} {standin.unit}"
                                  + (f"  →  {signal:.0%}" if signal is not None else ""))
            self.readout.color = t.TEXT
            self.note.value = (f"{standin.label}, standing in — {spec.label} has no value yet")
        elif raw is None:
            self.readout.value = "—"
            self.readout.color = t.FAINT
            self.note.value = f"waiting for {spec.label}"
        else:
            unit = f" {spec.unit}" if spec.unit else ""
            self.readout.value = f"{raw:.3g}{unit}" + (f"  →  {signal:.0%}" if signal is not None else "")
            self.readout.color = t.TEXT
            self.note.value = f"{spec.label} · {session.strategy.describe()}"
        self.warmup.width = 140 * session.strategy.warmup
        self.warmup.bgcolor = t.ACCENT if session.strategy.warmup >= 1.0 else t.WARN


def _level_arc(side: float, value: float | None) -> list:
    """The level as an arc **around** the breathing ring, from the top, clockwise.

    One widget carrying two things: the ring says breathe in or out, the arc says how the signal is doing.
    A second dial beside it would be two concentric circles doing different jobs, which is worse than one
    doing both — and a bar would invite reading the *gap* to its right, which on a rolling scale is a
    quantity that means nothing.

    An instrument rather than a configurable channel. It costs no space, it needs no setting, and it is on
    the tab you sit with rather than the one you configure.
    """
    import math

    inset = ARC_WIDTH / 2
    left = top = inset
    span = side - 2 * inset
    track = cv.Arc(left, top, span, span, 0.0, 2 * math.pi,
                   paint=ft.Paint(color=t.SURFACE_HI, stroke_width=ARC_WIDTH,
                                  style=ft.PaintingStyle.STROKE))
    if value is None:
        return [track]
    return [track,
            cv.Arc(left, top, span, span, -math.pi / 2, 2 * math.pi * min(1.0, max(0.0, value)),
                   paint=ft.Paint(color=t.ACCENT if value < 0.75 else t.GOOD,
                                  stroke_width=ARC_WIDTH, style=ft.PaintingStyle.STROKE))]


def session_for(protocol: Protocol, path=None, device=None, length_seconds=None,
                source=None, channels: int = 0, heart_choice: HeartChoice | None = None,
                heart_runner: heart.HeartRunner | None = None,
                mendi_choice=None, mendi_runner=None, pupil_choice=None,
                pupil_runner=None) -> HRVSession:
    """Same signature as the other screens' factories, so `app.pick` need not special-case it. The EEG
    and headband arguments are accepted and ignored: this protocol uses neither the amplifier nor the
    fNIRS source."""
    _ = path, device, source, channels, mendi_choice, mendi_runner, pupil_choice, pupil_runner
    choice = heart_choice if (heart_choice is not None and heart_choice.on) else None
    return HRVSession(protocol, choice=choice, length_seconds=length_seconds,
                      runner=heart_runner, offered=heart_choice)


INSTRUMENT_NAMES = {
    "usb": "HeartMath · USB",
    "strap": "heart-rate strap",
    "simulator": "simulated beats",
}


def _instrument(session: HRVSession) -> tuple[str, str, str]:
    """What the device chip should say: the source's short name, its state, and the badge colour.

    Short on purpose. The runner's own `describe()` is a sentence — it has to be, since it carries the
    reason a cable would not open — and the SOURCE panel is where that sentence belongs. The chip has
    room for a name and three or four words.
    """
    runner = session.heart
    name = INSTRUMENT_NAMES.get(session.choice.kind, "heart rate")
    if runner.error:
        return name, "not connected · see the device screen", t.BAD
    if runner.stale:
        return name, "connected, no beats for a while", t.WARN
    if runner.ready:
        state = session.state
        beats = f"{state.hr_smoothed:.0f} bpm" if state is not None else "beats arriving"
        # No "not the amplifier": the chip beside it already says HeartMath, and telling somebody that a
        # thing is not the other thing they can see it is not spends the one line this has on nothing.
        return name, beats, t.GOOD
    if runner.running:
        # **"Waiting for beats" covered three situations and named none of them**: nothing arriving on the
        # cable, a waveform with no pulse in it, and a waveform we are failing to find beats in. The source
        # knows which, and they want different things from the operator.
        say = getattr(runner.source, "progress", None)
        return name, (say() if say else "connected, waiting for beats"), t.ACCENT
    return name, "not started", t.MUTED


def view(page: ft.Page, session: HRVSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width() -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - SIDE_W - 70 - heart_panel.AXIS_W)

    heart_ui = heart_panel.HeartPanel(width_of=plot_width)

    # **The instruction sits under the circle, not inside it.** Inside, it had to fit the circle at its
    # smallest — the bottom of an exhale — so "breathe in" wrapped and clipped for most of the cycle. A
    # disc that changes size cannot also be a label.
    pacer_ring = ft.Container(width=RING_SIDE, height=RING_SIDE, border_radius=RING_SIDE // 2,
                              bgcolor=t.SURFACE_HI)
    pacer_word = ft.Text("—", size=17, color=t.MUTED, text_align=ft.TextAlign.CENTER)
    arc_note = ft.Text("", size=t.SMALL, color=t.FAINT, text_align=ft.TextAlign.CENTER)
    # **The visual channel goes here, not on the tab you configure it from.** Its whole purpose is to be seen
    # while the session runs. The gauge takes the ring's own place rather than adding a panel above it —
    # two concentric circles doing different jobs is worse than one — and the tint is this panel's
    # background, so it costs no space at all and sits directly behind the breathing cue: the ring says
    # breathe, the colour says how it is going.
    level_arc = cv.Canvas(shapes=[], width=ARC_SIDE, height=ARC_SIDE)
    pacer_note = ft.Text("", size=t.SMALL, color=t.FAINT)
    pacer_rows = ft.Column(spacing=12)

    def set_locked(locked: bool) -> None:
        session.pacer_locked = locked
        build_pacer()

    def set_period(seconds: float) -> None:
        session.set_pacer_period(seconds)
        build_pacer()

    def set_sound(on: bool) -> None:
        session.pacer_sound = on
        build_pacer()

    def set_target(hz: float) -> None:
        session.set_target_hz(hz)
        build_pacer()

    def set_target_width(half_hz: float) -> None:
        session.set_target_hz(session.target_hz, half_hz)
        build_pacer()

    def build_pacer() -> None:
        lo, hi = PACER_RANGE_S
        rows = [
            labelled("pacer",
                     pill("follow the heart", session.pacer_locked, lambda _e: set_locked(True)),
                     pill("metronome", not session.pacer_locked, lambda _e: set_locked(False)),
                     note="", width=54),
        ]
        # Only under the metronome. It was shown under both, on the argument that the locked pacer falls
        # back to it while waiting for beats — but a control that does nothing in the mode you are in is
        # a question about a mode you did not pick, and the fallback is over as soon as beats arrive.
        if not session.pacer_locked:
            rows.append(labelled(
                "cycle",
                num_box(session.pacer_period, set_period, suffix="s", step=PACER_STEP_S,
                        low=lo, high=hi),
                note=f"{60.0 / session.pacer_period:.1f} breaths a minute · from the target"))
        rows += [
            labelled("sound",
                     pill("on", session.pacer_sound, lambda _e: set_sound(True)),
                     pill("off", not session.pacer_sound, lambda _e: set_sound(False)),
                     note="a tone at each turn", width=54),
            pacer_note,
            # **Last, and quiet.** The target is the frequency every spectral number is scored against,
            # which makes it the most consequential control here and the least often touched: it is found
            # once, over a session or two, and then persists. Put at the top it pushed the breathing ring
            # — the thing you actually watch — down the panel, and its explanation is behind the `?` with
            # everything else's. Two numbers and one short line of live figures is the whole row.
            ft.Container(height=2),
            labelled("target",
                     num_box(session.target_hz, set_target, suffix="Hz", step=TARGET_STEP_HZ,
                             low=TARGET_RANGE_HZ[0], high=TARGET_RANGE_HZ[1], width=54),
                     note=session.target_note(), width=54),
            labelled("band",
                     num_box(session.target_half_hz, set_target_width, suffix="Hz",
                             step=TARGET_STEP_HZ, low=NARROW_HALF_RANGE_HZ[0],
                             high=NARROW_HALF_RANGE_HZ[1], width=54),
                     note="either side of it", width=54),
        ]
        pacer_rows.controls = rows
        page.update()

    left = ft.Column([
        panel("HRV SPECTRUM", *heart_ui.spectrum_rows()),
        ft.Container(height=12),
        panel("HEART RATE", *heart_ui.tacho_rows()),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    breathe_panel = panel(
        "BREATHE",
        ft.Row([ft.Stack([ft.Container(content=level_arc, width=ARC_SIDE, height=ARC_SIDE),
                          ft.Container(content=pacer_ring, width=ARC_SIDE, height=ARC_SIDE,
                                       alignment=ft.Alignment.CENTER)],
                         width=ARC_SIDE, height=ARC_SIDE)],
               alignment=ft.MainAxisAlignment.CENTER),
        ft.Container(height=8),
        ft.Row([pacer_word], alignment=ft.MainAxisAlignment.CENTER),
        # **The arc had nothing anywhere saying what it was.** It is the level — the same number the
        # SIGNAL panel reports and the mechanisms run on — drawn around the breathing ring so the two
        # things you watch are one object: the ring says breathe, the arc says how it is going.
        ft.Container(height=6),
        ft.Row([arc_note], alignment=ft.MainAxisAlignment.CENTER),
        ft.Container(height=12), pacer_rows,
        help=help_button(page, "The target and the pacer", BREATHE_HELP))

    right = ft.Column([
        breathe_panel,
        ft.Container(height=12),
        panel("NOW", heart_ui.tiles),
        ft.Container(height=12),
        panel("SOURCE", heart_ui.source_note,
              ft.Container(height=8),
              ft.Text("nothing here has to be watched for the feedback to work — see the Feedback tab",
                      size=t.SMALL, color=t.FAINT)),
    ], spacing=0, width=SIDE_W, scroll=ft.ScrollMode.AUTO)

    train_tab = ft.Container(content=ft.Row([left, right], spacing=14, expand=True),
                             padding=ft.Padding(14, 12, 14, 14), expand=True)
    # The shared Feedback pane, with the two rows that say where its level comes from above it. Everything
    # from the level outward — the volume, the screen, the click, the music — is the machinery every other
    # protocol uses, unchanged.
    source_ui = SignalSource(page, session)
    # `inset=False`: it is not a tab here but the lower half of one, so it takes neither a tab's padding, nor
    # a tab's scroll region, nor a tab's hidden-until-selected state — the last of which meant the whole
    # OUTPUT panel, every mechanism on this screen, rendered invisible.
    signal = signal_pane.SignalPane(page, session, inset=False)
    feedback_tab = ft.Container(
        content=ft.Column([source_ui.control, ft.Container(height=12), signal.control],
                          spacing=0, expand=True, scroll=ft.ScrollMode.AUTO),
        padding=ft.Padding(14, 12, 14, 14), expand=True, visible=False)
    panes = [train_tab, feedback_tab, plan_pane_.control]

    def select_tab(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs(["Train", "Feedback", "Session"], select_tab)
    shown: list = [None]

    def draw() -> None:
        session.tick()
        # The chip answers "where are these numbers coming from", and on this screen the answer is never
        # the amplifier. It read "no device" beside a connected pulse sensor, which is true of the
        # amplifier and useless as an answer — and a grey "no device" beside a working instrument reads
        # as a fault.
        # Only when the words change: `set_source` refreshes the whole toolbar, and doing that ten times
        # a second alongside the refresh below is two of everything for a chip that changes twice a
        # session.
        instrument = _instrument(session)
        if instrument != shown[0]:
            shown[0] = instrument
            toolbar.set_source(instrument[0], note=instrument[1], tone=instrument[2])
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        heart_ui.refresh(session.heart)
        # **Always measured, driven only while running.** The outputs are the room and a paused session
        # that kept dimming the screen is the failure `shadow`'s whole safety story is about — but that
        # rule was applied to the *measurement* too, so a rolling scale learnt nothing until you pressed
        # play and then gated the feedback for the first thirty seconds of the session. It is warm now
        # before you start, and `observe` is where the line between measuring and driving is drawn.
        session.observe()
        source_ui.refresh()
        signal.refresh()

        level_arc.shapes = _level_arc(ARC_SIDE, session.signal)
        # What the outer ring is, in the words the rest of the screen uses for it. The number is the
        # level; the name is whichever metric is producing it, including the stand-in, so the caption
        # never claims the chosen metric before that metric can speak.
        measuring = (HRV_METRICS[STANDIN_METRIC] if session.standing_in else session.metric_spec).label
        arc_note.value = (f"the outer ring is the level · {measuring} at {session.signal:.0%}"
                          if session.signal is not None else
                          f"the outer ring is the level · waiting for {measuring}")

        inhale = session.breathe_in()
        pacer_ring.bgcolor = t.ACCENT_DIM if inhale else t.SURFACE_HI
        pacer_word.value = "breathe in" if inhale else "breathe out"
        pacer_word.color = t.TEXT
        # **The ring breathes at the tempo it is asking for.** Words that flip every five seconds tell you
        # what to do but not how far through it you are, so the breath ends up led by the word and not by
        # the pace — you inhale in two seconds and wait. A circle that grows for the whole inhale and
        # shrinks for the whole exhale is the instruction and the pace in one object. Only the metronome
        # has a phase to animate; see `breath_phase`.
        phase = session.breath_phase()
        side = (RING_SIDE if phase is None else
                RING_MIN + (RING_SIDE - RING_MIN) * (phase if inhale else 1.0 - phase))
        pacer_ring.width = pacer_ring.height = side
        pacer_ring.border_radius = side / 2
        pacer_note.value = session.pacer_note()
        # **Amber when "follow the heart" is not following the heart.** With no beats it falls back to the
        # clock, which is right — but the fallback is invisible in the pacer's behaviour, so the mode looked
        # broken rather than waiting. The words said so all along, in the same grey as everything else.
        waiting = session.pacer_locked and not session.heart.ready
        pacer_note.color = t.WARN if waiting else t.FAINT
        page.update()

    # No amplifier channels on this screen, so the count in the toolbar is a control for a device
    # it never touches. `app.pick` turns it back on for the screens that do.
    toolbar.set_channels_visible(False)
    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)
    build_pacer()
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)
