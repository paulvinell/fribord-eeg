"""HEG: training prefrontal blood flow, from a Mendi headband.

The third protocol here whose signal is not EEG, and the second to be built on the layers the HRV screen
introduced — a **metric** chosen from a catalogue (`nfb.heg.metrics`) and a **scaling** chosen from five
(`normalise`), with everything from the level outward being the same four mechanisms every other screen
uses. Nothing about the feedback path is new; what is new is where the number comes from.

**Why there is no threshold and no calibration phase.** The projection that produces HbO and HbR is not
Beer-Lambert — it is bilateral averaging with the middle channel subtracted as common mode — so its
absolute scale is arbitrary and differs between headbands, between placements, and between one session
and the next on the same forehead. A threshold in those units would mean nothing, and thirty seconds of
calibration would only be measuring the offset. The rolling scalings do that job continuously and say so
on the warm-up bar.

**Motion is a hold, not a dropout, and that distinction is the whole reason this screen reads quality.**
The processor holds the optical channels flat for a two-second cooldown after a jolt, because the skin
takes that long to settle mechanically. If the screen treated that as a dead source it would release the
outputs several times a minute on a person who is merely alive; if it treated it as data it would feed
two seconds of constants into the rolling scale, which reads as sudden perfect steadiness — and on a
variability metric that is *rewarded*, so the room would pay for moving your head. So: hold. The level
stays where it was, the outputs keep running on it, and the headline says why.

**What is not here, and deliberately.** No pacer: the HRV screen's breathing cue is locked to the
tachogram and there is no equivalent instruction for blood flow. No second calibration on top of the
scaling. And no attempt to name a "good" absolute value, for the reason in the first paragraph.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import flet as ft
import flet.canvas as cv

from nfb.heg import DEFAULT_HEG_METRIC, HEG_METRICS, QUALITY_MOTION
from nfb.hrv.metrics import metric_value

from . import (chrome, cues, heg_panel, mendi, music, normalise, plan_pane, prefs, shadow,
               signal_pane, sysvolume)
from . import theme as t
from .feedback import CONDITION, FeedbackSettings
from .hrv_view import _level_arc
from .inputs import MendiChoice
from .plan import SessionPlan
from .protocols import Protocol
from .targets import Target
from .widgets import help_button, labelled, num_box, panel, pill

HANDLES = ("HEG",)

SIDE_W = 330
ARC_SIDE = 148

CONDITION_LEVEL = 0.5
"""Where the level counts as met, for the click and the music's stop — the middle of the scaled range.

There is nothing to calibrate because the scaling already did it, and here that is not merely convenient:
the projection's units are arbitrary, so there is no absolute number a threshold could be set to."""

OBSERVE_HZ = 10.0
"""How often the level is recomputed, matched to the draw loop's 0.1 s tick. The *display* rate, not the
device's — frames arrive at about 30 Hz, so most of these ticks read a state that has already moved on by
two or three frames, which is immaterial for a signal low-passed at 0.5 Hz."""

WARMUP_GATE = 0.0
"""How much of a rolling strategy's window must exist before the signal drives anything. **None of it.**

This was a quarter — thirty seconds of the window — on the argument that the room should not follow a
range built from two readings. `hrv_view.WARMUP_GATE` records why that trade is the wrong one and this
screen was simply not brought along: a rolling scale with two readings in it is not lying, it reports
the exact middle, which is what "I have seen one value" honestly maps to, and it widens as the person's
real range appears. Watching the level firm up beats a silent opening minute followed by a level with no
visible relationship to anything you did.

It is worse here than on the HRV screen, in fact, and the reason is the gate above: what it withheld was
the first thirty seconds of *this* metric, on a signal whose absolute scale is arbitrary — so there was
nothing else on the screen to read in the meantime either.

The warm-up is still reported — the amber bar beside the readout — because saying "this is still
settling" costs nothing where withholding the number costs the opening of the session. Kept as a named
constant at zero rather than deleted, because the thing it must not silently become again is a number."""


class HEGSession:
    """The headband, the clock, and the level. Not a `TrainingSession`.

    Deliberately not, for the reason `HRVSession` is not: that class is built around blocks of evenly
    spaced amplifier samples with an artifact gate and band metrics, none of which describe a 30 Hz
    optical stream with its own motion gate. The plan and its cues are composed rather than inherited.
    The frames themselves come from a `MendiRunner`, which is the app's, not this screen's.
    """

    def __init__(self, protocol: Protocol, choice: MendiChoice | None = None,
                 length_seconds: float | None = None, runner: mendi.MendiRunner | None = None,
                 offered: MendiChoice | None = None):
        self.protocol = protocol
        self.plan = SessionPlan(length_seconds)
        self.cue = cues.play
        # A simulator by default *here only*, exactly as on the HRV screen: choosing this protocol means
        # you want to train HEG, and opening it to an empty screen because nothing is paired would be
        # useless. "Here only" is a promise `stop()` has to keep — the runner is the app's and `attach`
        # writes the choice onto it, so an invented simulator has to be handed back switched off or one
        # visit here leaves the whole app on invented frames. `_invented` is that memo.
        self._invented = choice is None
        self.choice = choice if choice is not None else MendiChoice(kind="simulator")
        # The app's own choice, kept even when it was off — the operator can pair a headband on the
        # device screen mid-session, and coming back holding an invented simulator would hand that
        # simulator to the shared runner and switch the real headband off. See `adopt_offered`.
        self.offered = offered if offered is not None else choice
        self.heg = runner if runner is not None else mendi.MendiRunner(self.choice)
        self.finished = False

        # The shared mechanisms, unchanged: sound is the master volume, shadow is the screen, click is a
        # discrete mark, music is the continuous sound — each with its own floor, ceiling and calibration,
        # each persisted, each configured through the pane every other protocol uses.
        self.feedback = FeedbackSettings()
        self.display = shadow.DisplayShadow()
        self.output_volume = sysvolume.SystemVolume()
        self.music = music.MusicOutput()
        for name, mech in self.feedback.mechanisms.items():
            if name not in self.offers:
                mech.enabled = False
        self.feedback.music.enabled = True
        self.feedback.music.source = "tone"

        self.metric = DEFAULT_HEG_METRIC
        # The metric supplies the *default* aim and the operator owns it, the same way a band on the
        # amplitude screen arrives as increase or decrease and can be cycled there. Two of these metrics
        # are trained down — the variability ones — and somebody working on recovery from over-arousal
        # may well want the other direction on them.
        self.direction = HEG_METRICS[DEFAULT_HEG_METRIC].default_direction
        self.strategy_id = HEG_METRICS[DEFAULT_HEG_METRIC].strategies[0]
        self.strategy_params: dict = {}
        self._strategy = normalise.build(self.strategy_id)
        self.raw_metric: float | None = None
        self._signal: float | None = None
        self._level = 0.0
        self._level_at: float | None = None
        self._last_click: float | None = None
        # One target, for three specific jobs rather than to hold a threshold: it is what the shared pane
        # draws a contribution row for, what tells that pane data is arriving, and what the click asks.
        # Its value is the **already-normalised** signal and its threshold is the middle of that range.
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
    # Declared rather than inherited, because this is not a `TrainingSession` — a frame is not a sample
    # block and none of that class's spec or artifact machinery applies. A missing one of these is an
    # AttributeError inside a Flet event handler rather than a wrong default, which is why they are all
    # spelled out even where the value is the boring one.
    offers = ("sound", "shadow", "click", "music")
    music_sources = ("tone", "crossfade", "noise", "shepard", "file")
    """`tone` first, because a bounded pitch carries a value with no reference point needed — you hear
    which way it moved without remembering where it was. That is worth more here than anywhere: the
    signal's absolute level is arbitrary, so *which way it moved* is the whole of what there is to know.
    `eeg` is absent — it sonifies a sample stream and there is none here."""
    combines = ("weighted",)
    """One target, so every rule reduces to the same thing. Stated rather than offered."""
    pinned_feedback = frozenset()
    arrangements: tuple[str, ...] = ()
    state_ends = ("scattered", "coherent")
    tone_bias = None
    unbiased = False
    gates = None
    scores_bands = False
    """The level comes from a metric and a scaling rather than from targets against thresholds, so the
    four steps that build a score have nothing to act on — see `TrainingSession.scores_bands`."""
    sustains_reward = False
    hold_choices = None
    hold_seconds = 0.0
    tunes_envelope = False
    """No band estimator takes `envelope_seconds` here, so the SIGNAL panel's band-averaging row is
    absent — see `TrainingSession.tunes_envelope`."""
    conditions_signal = False
    """No sample-artifact gate: that is amplitude rejection on blocks of evenly spaced samples. Frames
    are gated too — for *motion*, in `nfb.heg.FrameProcessor` — and the share held is reported beside the
    trace, where it can be read against the signal it applies to."""

    def level_note(self) -> tuple[str, str] | None:
        """Why there is no level while frames arrive. `HRVSession.level_note`'s twin — see there."""
        if self.error is not None or self.heg.stale or not self.heg.ready:
            return None
        if self.raw_metric is None:
            return (f"waiting for {self.metric_spec.label}",
                    "frames are arriving · this metric needs more frames before it can show a value")
        # No note for a half-warm scale: the level is running on it, the bar beside the readout says how
        # much of the window it has, and a panel-wide "no level yet" beside a level that exists reads as
        # a fault. See `WARMUP_GATE`.
        return None

    def dead_channels(self) -> list[str]:
        if self.error is not None or self.heg.stale:
            return ["the headband"]
        return [] if self.heg.ready else ["the headband"]

    # --- surviving a stop and a relaunch ----------------------------------------------------

    def adopt(self, previous: "HEGSession") -> None:
        """Take over what the operator configured on the session being replaced.

        The same line `TrainingSession.adopt` draws: the session is the history and the configuration is
        not, so the plan and the whole feedback arrangement come across while the frames and the
        normaliser's window do not — those were measured on a session that has ended.
        """
        if previous is self or previous.protocol.code != self.protocol.code:
            return
        self.plan = previous.plan
        self.plan.rewind()
        self.feedback = previous.feedback
        self.metric = previous.metric
        self.direction = previous.direction
        self.set_strategy(previous.strategy_id, **previous.strategy_params)

    def calibration(self) -> dict:
        """What `prefs` keeps between launches — the mechanisms' own whitelist and nothing else.

        The metric and the scaling are deliberately absent: they are what is being trained, the same
        category as a band or a threshold, and prefs' own docstring is why those never persist.
        """
        return prefs.capture(self.feedback)

    def restore_calibration(self, values: dict) -> int:
        return prefs.apply(self.feedback, values)

    # --- where the level comes from ------------------------------------------------------------------

    @property
    def metric_spec(self):
        return HEG_METRICS[self.metric]

    def set_metric(self, metric_id: str) -> None:
        """Point the level at a different quantity, and rebuild the normaliser.

        Rebuilt, not kept: a rolling strategy's window holds values in the *old* metric's units, and here
        they are not even the same order of magnitude — a slope in units per second against a history of
        raw projection levels would sit pinned at an end until the window aged out. The direction moves
        with the metric too, because "decrease" carried from a variability metric onto HbO would train
        somebody toward *less* blood flow with nothing on the screen having changed but a name.
        """
        if metric_id not in HEG_METRICS:
            return
        self.metric = metric_id
        self.direction = HEG_METRICS[metric_id].default_direction
        wanted = HEG_METRICS[metric_id].strategies
        if wanted and self.strategy_id not in wanted:
            self.strategy_id = wanted[0]
        self.set_strategy(self.strategy_id)

    def set_direction(self, direction: str) -> None:
        """Train the metric up or down. Resets the scaling's history, because the level is about to mean
        the opposite and the on-target history behind the readout was measured under the other aim."""
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
        self._signal = None
        self.signal_target.reset_stats()
        self.signal_target._settled = True

    @property
    def strategy(self) -> normalise.Strategy:
        return self._strategy

    # --- the motion hold -----------------------------------------------------------------------
    #
    # Read and written straight through to the processor rather than mirrored on the session. The
    # processor is the thing doing it, it is on the app's runner rather than this session's, and a copy
    # here would be a second answer to "is the hold on" that a stop, a restart or a swapped headband
    # could leave disagreeing with the first.

    @property
    def motion_hold(self) -> bool:
        return self.heg.processor.motion_gate

    @motion_hold.setter
    def motion_hold(self, on: bool) -> None:
        self.heg.processor.motion_gate = bool(on)

    @property
    def motion_sensitivity(self) -> float:
        return self.heg.processor.motion_sensitivity

    @motion_sensitivity.setter
    def motion_sensitivity(self, value: float) -> None:
        self.heg.processor.motion_sensitivity = value

    @property
    def level_source(self) -> tuple[str, str]:
        """What the readout says the level is, in place of the combining rule's wording."""
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

    @property
    def holding(self) -> bool:
        """Whether the last frame was inside a motion cooldown, and the level is therefore frozen."""
        state = self.state
        return bool(state is not None and state.quality == QUALITY_MOTION)

    def observe(self) -> None:
        """Metric, scaling, then the shared mechanisms. The whole path, called once per frame.

        The inversion is applied to the *normalised* value rather than to the metric, so a strategy's
        window still sees the quantity in its own units and its own direction — inverting first would
        turn a rolling minimum into a rolling maximum and make the pane's description of it wrong.
        """
        # A frozen snapshot is not a reading. `state` holds the last frame for ever, so without this gate
        # a dead headband would keep driving the volume, the shadow and the clicks from it indefinitely —
        # the HRV screen's bug, and at 30 Hz the staleness window can be much tighter.
        if self.error is not None or self.heg.stale:
            self.raw_metric = None
            self._signal = None
            self.release_outputs()
            return
        # **Motion is a hold, not a dropout.** The processor has held the optical channels flat for its
        # two-second cooldown, so there is nothing new to push — and pushing the held constants would
        # feed the rolling scale a stretch of perfect steadiness, which a variability metric rewards. So
        # the strategy is not fed, the level is not moved, and the outputs are *not* released: releasing
        # would strobe the room every time the person shifted in their chair.
        if self.holding:
            return
        raw = metric_value(self.metric_spec, self.state)
        self.raw_metric = raw
        if raw is None:
            self._signal = None
            self.release_outputs()
            return
        signal = self._strategy.push(raw, self.elapsed)
        if self._strategy.warmup < WARMUP_GATE:
            # Zero now, so this never runs — kept because the constant is kept, and for the same reason.
            self._signal = None
            self.release_outputs()
            return
        if self.inverts:
            signal = 1.0 - signal
        self._signal = signal
        self.signal_target.observe(signal, OBSERVE_HZ)

        tau = max(0.0, self.feedback.smoothing_seconds)
        now = self.elapsed
        if tau <= 0:
            self._level = signal
        else:
            # Session time, not a nominal tick: the loop is a sleep plus a frame's work, so a fixed
            # `1/OBSERVE_HZ` step always smooths harder than configured by whatever the frame costs.
            # `elapsed` rather than the wall clock because it is the clock the strategies already run on:
            # it freezes across a pause, and it follows an accelerated session instead of the room's.
            dt = (now - self._level_at) if self._level_at is not None else 1.0 / OBSERVE_HZ
            if dt > 0:
                self._level += (1.0 - math.exp(-dt / tau)) * (signal - self._level)
        self._level_at = now
        self._drive_outputs()

    def _drive_outputs(self) -> None:
        """Hand the level to the mechanisms. The same four every other screen has.

        Written here rather than inherited because this is not a `TrainingSession` — but it is the same
        eight lines as on the HRV screen, and a third copy should become a shared function rather than a
        fourth.
        """
        met = self.feedback.on_target(self.channel_targets(CONDITION))
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
        """The discrete mark, on the same interval rule as everywhere else."""
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
        return self.heg.error

    @property
    def state(self):
        return self.heg.state

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
        reimplemented here because this session does not inherit it. Skipping `observe()` while paused
        stops the outputs being *updated*, not being *held*: without this the display stays dimmed and
        the music sings on. Resuming needs no restart — every mechanism owns its take/follow/release
        lifecycle inside `apply`."""
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
        """Take the app's headband if one has been paired since this session started. `HRVSession`'s
        twin — see there for why an invented simulator must lose to anything actually chosen."""
        if self._invented and self.offered is not None and self.offered.on:
            self.choice = self.offered
            self._invented = False

    def run(self) -> None:
        """Reading frames is the runner's own thread, so this only has to ask for it."""
        self.adopt_offered()
        self.heg.attach(self.choice)

    def stop(self) -> None:
        self._stop = True
        self.paused = True
        # Keep what was set by ear, as every other screen does on the way out.
        prefs.remember(self.protocol.code, self)
        self.release_outputs()
        self.heg.release()
        if self._invented:
            # The simulator was this screen's default, never the operator's choice. Hand the shared
            # runner back switched off with the invented state cleared, so nothing else in the app finds
            # simulated frames it never asked for.
            self.heg.choice = MendiChoice(kind="off")
            self.heg.state = None
            self.heg.error = None
            self.heg.seen_at = None
            self.heg.processor.reset()
            self.heg.signal.reset()

    # --- going to the device screen and coming back ------------------------------------------
    # The chip on the shared toolbar calls these on whatever session is up. There is no exclusive device
    # mode to give up here — the headband is Bluetooth, not the amplifier — so suspend only has to stop
    # holding the room and rearm only has to carry on.

    def suspend(self) -> None:
        """Stop driving the outputs and release them, keeping the session alive to come back to."""
        if not self.paused:
            self.toggle_pause()             # which releases the outputs
        self.release_outputs()              # and again, for a suspend that arrives already paused

    @property
    def suspended(self) -> bool:
        return not self._stop

    def rearm(self) -> None:
        """Nothing to rebuild: the runner never stopped and the clock stayed frozen where `suspend` left
        it. The operator resumes when ready, exactly as after a pause."""


SIGNAL_HELP = (
    "**What you see.** Four rows. **feed back** sets the quantity. **aim to** sets the direction to train it in. "
    "**scaled by** sets how it is scaled. **when you move** sets what happens during movement. The first "
    "three rows set the level. The volume, the screen, the click and the music all follow "
    "that level.\n\n"
    "**What the quantity means.** There are six quantities to choose from. HbO is the main activation "
    "signal, and most people mean HbO when they say HEG. HbO minus HbR swings more than either signal "
    "alone, and it removes much of the shared heart, breathing and scalp noise. It is the default signal "
    "for that reason. The two variability metrics show how steady the activation is, not how high it "
    "is. The trend shows the slope of the signal. It rises when attention returns after a lapse.\n\n"
    "**What to do.** Pick a quantity, then set the direction to increase or decrease. The two "
    "variability metrics start set to decrease, because a low value means steady activation. Every "
    "other metric starts set to increase. You can change the direction at any time. Changing the "
    "quantity resets the direction to that quantity's own default.\n\n"
    "**What the scaling means.** The raw signal has no fixed unit, and it changes with the headband's "
    "position on the head. Instead of a fixed threshold, the scale adjusts to your own range as you "
    "train. The bar beside the reading shows how much of your range the scale has learned so far. "
    "Amber means it is still learning.\n\n"
    "**What motion does.** A jolt moves the sensors out of contact for a moment. When this happens, the "
    "screen shows \"held — motion in the last two seconds\" and the level holds its last value. The "
    "volume, screen, click and music keep running on that held value instead of stopping. The note "
    "beside the **when you move** row shows the share of frames held so far.\n\n"
    "**The motion check.** The accelerometer has no fixed unit, so the check compares recent movement "
    "with your own average stillness, by a set multiple. Raise the number after "
    "**over** if an ordinary session holds too many frames. Lower it if movement still gets through. "
    "Select **keep training** to train on every frame, including frames with movement in them. Select "
    "**hold** to switch the check on again. The note beside the row shows the share held either way."
)


class SignalSource:
    """Two rows: which quantity, and how it is scaled into the 0…1 the mechanisms follow.

    A near-twin of `hrv_view.SignalSource` against a different catalogue. It is a copy rather than a
    shared class because the two screens are the only users and the shapes may yet diverge — but they
    have not diverged yet, so **a third screen wanting this should extract it rather than copy it
    again**, taking the catalogue as an argument.
    """

    def __init__(self, page: ft.Page, session: "HEGSession"):
        self.page = page
        self.session = session
        self.rows = ft.Column(spacing=12)
        self.readout = ft.Text("", size=t.BODY, color=t.TEXT)
        self.note = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.motion_note = ft.Text("", size=t.SMALL, color=t.FAINT, expand=True)
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
            labelled("when you move", *self._motion(), width=76),
        ]
        self._refresh_motion()
        self.page.update()

    def _chooser(self, spec) -> ft.Control:
        """A menu rather than six pills: the labels are not self-explanatory, so each needs its one-line
        description beside it as you consider it."""
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
                for m in HEG_METRICS.values()])

    def _motion(self) -> list:
        """Hold or carry on, and how much movement counts as movement.

        The number is only offered while the hold is on: with it off there is nothing for a threshold to
        be the threshold of, and an inert box beside a switched-off row invites the reading that the
        setting still applies.
        """
        session = self.session
        out = [pill("hold", session.motion_hold,
                    lambda _e: (setattr(session, "motion_hold", True), self.rebuild())),
               pill("keep training", not session.motion_hold,
                    lambda _e: (setattr(session, "motion_hold", False), self.rebuild()))]
        if session.motion_hold:
            out += [ft.Text("over", size=t.SMALL, color=t.FAINT),
                    num_box(session.motion_sensitivity,
                            lambda v: setattr(session, "motion_sensitivity", float(v)),
                            suffix="×", step=0.5, low=1.0, high=20.0)]
        # The note carries live figures, so it is a control this class keeps and writes on every draw
        # rather than a string baked in at rebuild time.
        return [*out, ft.Container(width=10), self.motion_note]

    def _refresh_motion(self) -> None:
        """A live figure and a number nobody can read off the screen — which is what earns a note.

        The multiple is of *your own stillness*, measured continuously in the accelerometer's own counts,
        and neither of those two facts is visible from a box reading "4 ×".
        """
        session = self.session
        held = session.heg.gate_fraction
        if not session.motion_hold:
            self.motion_note.value = (f"every frame trains, including movement · the hold condition "
                                      f"matched {held:.0%} of this session so far")
            return
        floor = session.heg.processor.stillness
        measured = ("still measuring how still you are" if floor is None else
                    f"your own stillness reads {floor:.0f} counts")
        self.motion_note.value = f"{held:.0%} held so far · {measured}, and this is the multiple of it"

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
        if raw is None:
            self.readout.value = "—"
            self.readout.color = t.FAINT
            self.note.value = f"waiting for {spec.label}"
        else:
            unit = f" {spec.unit}" if spec.unit else ""
            self.readout.value = (f"{raw:.3g}{unit}"
                                  + (f"  →  {signal:.0%}" if signal is not None else ""))
            self.readout.color = t.TEXT
            self.note.value = f"{spec.label} · {session.strategy.describe()}"
        self.warmup.width = 140 * session.strategy.warmup
        self.warmup.bgcolor = t.ACCENT if session.strategy.warmup >= 1.0 else t.WARN
        self._refresh_motion()


def session_for(protocol: Protocol, path=None, device=None, length_seconds=None,
                source=None, channels: int = 0, heart_choice=None, heart_runner=None,
                mendi_choice: MendiChoice | None = None,
                mendi_runner: mendi.MendiRunner | None = None, pupil_choice=None,
                pupil_runner=None) -> HEGSession:
    """Same signature as the other screens' factories, so `app.pick` need not special-case it. The EEG
    and heart arguments are accepted and ignored: this protocol uses neither the amplifier nor a strap."""
    _ = path, device, source, channels, heart_choice, heart_runner, pupil_choice, pupil_runner
    choice = mendi_choice if (mendi_choice is not None and mendi_choice.on) else None
    return HEGSession(protocol, choice=choice, length_seconds=length_seconds, runner=mendi_runner,
                      offered=mendi_choice)


def _instrument(session: HEGSession) -> tuple[str, str, str]:
    """What the device chip should say. The HRV screen's twin — see `hrv_view._instrument`, including
    why the chip gets a few words and the panel gets the runner's sentence."""
    runner = session.heg
    name = "simulated headband" if session.choice.kind == "simulator" else "Mendi headband"

    def said(note: str, tone: str) -> tuple[str, str, str]:
        """The charge on the chip's second line, where it is in front of you for the whole session.

        It was under the trace and nowhere else, which is the wrong place for it: a headband that dies
        mid-session ends the session, and below about 15% the firmware drops into low-power mode and
        slows the frame rate — so the number you want before starting a twenty-minute run is the one
        that was easiest to scroll past. On the second line rather than in the chip's name, because the
        name answers "what is this" and does not change.
        """
        pct = runner.battery_pct
        return name, (note if pct is None else f"{note} · battery {pct}%"), tone

    if runner.error:
        return name, "not connected · see the device screen", t.BAD
    if runner.stale:
        return said("connected, no frames for a while", t.WARN)
    if runner.ready:
        return said("frames arriving", t.GOOD)
    if runner.running:
        return said("connected, waiting for frames", t.ACCENT)
    return name, "not started", t.MUTED


def view(page: ft.Page, session: HEGSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width() -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - SIDE_W - 70 - heg_panel.AXIS_W)

    heg_ui = heg_panel.HEGPanel(width_of=plot_width)

    # The level, as an arc — the same instrument the HRV screen wraps around its breathing ring, without
    # a ring inside it. Imported rather than copied: it is one drawing of one number, and two of them
    # would drift apart for no reason.
    level_arc = cv.Canvas(shapes=[], width=ARC_SIDE, height=ARC_SIDE)
    level_note = ft.Text("", size=t.SMALL, color=t.FAINT)

    left = ft.Column([
        panel("BLOOD FLOW", *heg_ui.trace_rows()),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    right = ft.Column([
        panel("LEVEL",
              ft.Row([ft.Container(content=level_arc, width=ARC_SIDE, height=ARC_SIDE)],
                     alignment=ft.MainAxisAlignment.CENTER),
              ft.Container(height=10), level_note),
        ft.Container(height=12),
        panel("NOW", heg_ui.tiles),
        ft.Container(height=12),
        panel("SOURCE", heg_ui.source_note,
              ft.Container(height=8),
              ft.Text("nothing here has to be watched for the feedback to work — see the Feedback tab",
                      size=t.SMALL, color=t.FAINT)),
    ], spacing=0, width=SIDE_W, scroll=ft.ScrollMode.AUTO)

    train_tab = ft.Container(content=ft.Row([left, right], spacing=14, expand=True),
                             padding=ft.Padding(14, 12, 14, 14), expand=True)
    source_ui = SignalSource(page, session)
    # `inset=False`: it is not a tab here but the lower half of one, so it takes neither a tab's padding,
    # nor a tab's scroll region, nor a tab's hidden-until-selected state — the last of which once made
    # the whole OUTPUT panel invisible on the HRV screen.
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
        # The chip is the answer to "where are these numbers coming from", and here that is the headband,
        # never the amplifier — which is not connected, so the chip read "no device" beside a headband
        # that was streaming. Same fix as the HRV screen's.
        instrument = _instrument(session)
        if instrument != shown[0]:
            shown[0] = instrument
            toolbar.set_source(instrument[0], note=instrument[1], tone=instrument[2])
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        heg_ui.push(session.state)
        # Only while running: the outputs are the room, and a paused session that kept dimming the screen
        # is the failure `shadow`'s whole safety story is about.
        if session.running:
            session.observe()
        heg_ui.refresh(session)
        source_ui.refresh()
        signal.refresh()

        level_arc.shapes = _level_arc(ARC_SIDE, session.signal)
        if session.holding:
            level_note.value = "held — motion in the last two seconds"
        elif session.signal is None:
            level_note.value = "waiting for a signal to scale"
        else:
            headline, detail = session.level_source
            level_note.value = f"{headline} · {detail}"
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
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)
