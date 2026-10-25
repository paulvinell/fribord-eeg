"""Pupil: training voluntary control of pupil size.

**Passive** feeds back the pupil against its own last minute, with no trial and no aim. **Active** runs
the published protocol's trial: a cue names the direction, a baseline is measured while the person counts
backwards, fifteen seconds of effort are fed back against it, and the trial's mean is the verdict. The
engine is `trials.TrialRun` with a cue phase in front of the baseline, because a pupil dilates to the cue
tone itself and that answer must not sit inside the baseline.

**Nothing on this screen may change its light in step with the feedback**, or it trains the light reflex.
So there is no shadow, and the circle keeps a constant number of lit pixels: its line thins as it grows.
A blink holds the last kept diameter instead of stopping the sound, as the HEG screen holds through motion.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv

from . import chrome, cues, normalise, plan_pane, prefs, pupil, signal_pane, sysvolume, music
from . import theme as t
from .feedback import CONDITION, FeedbackSettings
from .inputs import PupilChoice
from .plan import SessionPlan
from .plots import dashed, grid, mark_shapes, place_value_labels, polyline, value_axis
from .protocols import Protocol
from .targets import Target
from .trials import (ACTIVE, BASELINE, CUE, CUED, ITI, NEGATIVE, POSITIVE, RESULT, VERDICT,
                     TrialRun)
from .widgets import button, help_button, labelled, num_box, panel, pill, test_button

HANDLES = ("PUPIL",)

SIDE_W = 360
CIRCLE_SIDE = 280
TRACE_H = 150
TRACE_SECONDS = 60.0
AXIS_W = 46
CELL_W, CELL_H = 9, 14

MODES = ("passive", "active")
DIRECTIONS = {
    "increase": (POSITIVE,),
    "decrease": (NEGATIVE,),
    "alternating": (POSITIVE, NEGATIVE),
    "random": (POSITIVE, NEGATIVE),
}
HIDDEN_DIRECTIONS = ("random",)
"""Directions that keep the aim out of sight until the effort starts. The cue names a trial, but not its
direction, so the person cannot set the pupil the wrong way in the baseline to make the shift look large."""

CUE_S = 2.0
BASELINE_S = 5.0
ACTIVE_S = 15.0
RESULT_S = 2.0
ITI_S = (4.0, 6.0)
"""The published protocol's trial: a 2 s cue, a baseline whose last 5 s are averaged, 15 s of
regulation, 2 s of verdict and a short rest. About 29 s a trial."""
TRIAL_S = CUE_S + BASELINE_S + ACTIVE_S + RESULT_S + sum(ITI_S) / 2

DEFAULT_TRIALS = 20
"""Two blocks of ten, the published block size. About ten minutes."""

DEFAULT_RANGE_MM = 0.3
"""The change from the baseline that takes the level to its end. Voluntary changes are 0.1 to 0.5 mm, so
0.3 mm puts a good trial near the top without pinning it there."""

MIN_RANGE_MM = 0.05
"""Below this the level shows measurement noise, not the person."""

PASSIVE_WINDOW_S = 60.0

OBSERVE_HZ = 10.0
"""How often a reading is taken from the stream, matched to the draw loop. The stream runs at 60 Hz and
is already smoothed; a voluntary change takes a second to show, so ten readings a second lose nothing."""

CUE_PITCH = {POSITIVE: 620.0, NEGATIVE: 310.0}
"""The slow-potential screen's two cue tones, for the same reason: an octave apart, so that the two
directions are told apart without looking. Here it matters more, since the eyes stay on the fixation
point."""
CUE_PITCH_NEUTRAL = 440.0
"""A tone between the two, for a hidden-direction trial. It starts the baseline without naming the aim,
so the directional tone can arrive later, at the start of the effort."""
CUE_VOLUME = 25

DIRECTION_WORD = {POSITIVE: "larger", NEGATIVE: "smaller"}


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60.0)
    return f"{minutes} min" if minutes >= 1 else f"{seconds:.0f} s"


class PupilSession:
    """The eye, the trials and the level. Not a `TrainingSession`, for the reason `HEGSession` is not:
    there are no sample blocks here, and the readings come from a runner the app owns."""

    def __init__(self, protocol: Protocol, choice: PupilChoice | None = None,
                 length_seconds: float | None = None, runner: pupil.PupilRunner | None = None,
                 offered: PupilChoice | None = None):
        self.protocol = protocol
        self.plan = SessionPlan(length_seconds)
        self.cue = cues.play
        # A simulated eye by default here only, as on the HEG screen, and handed back switched off by
        # `stop` — see `HEGSession._invented`.
        self._invented = choice is None
        self.choice = choice if choice is not None else PupilChoice(kind="simulator")
        self.offered = offered if offered is not None else choice
        self.eye = runner if runner is not None else pupil.PupilRunner(self.choice)
        self.finished = False

        self.feedback = FeedbackSettings()
        self.output_volume = sysvolume.SystemVolume()
        self.music = music.MusicOutput()
        for name, mech in self.feedback.mechanisms.items():
            if name not in self.offers:
                mech.enabled = False
        self.feedback.music.enabled = True
        self.feedback.music.source = "tone"

        self.mode = "active"
        self.direction = "alternating"
        self.n_trials = DEFAULT_TRIALS
        self.transfer_fraction = 0.0
        self.success_mm = 0.0
        self.range_mm = DEFAULT_RANGE_MM
        self.feedback_window_s = ACTIVE_S
        self.update_s = 0.0
        self.passive_window_s = PASSIVE_WINDOW_S
        self._scale = normalise.ZScore(window_s=self.passive_window_s)
        self.trials = self._new_run()
        self.count_from: int | None = None
        self.diameter: float | None = None
        # The trace's points, kept here rather than in the view: a trip to the device screen rebuilds the
        # view, and the last minute of the pupil is the session's.
        self.history: deque[tuple[float, float | None, bool]] = deque()

        self._signal: float | None = None
        self._level = 0.0
        self._level_at: float | None = None
        self._last_t: float | None = None
        self._last_phase: str | None = None
        self._block_at: float | None = None
        self._block_vals: list[float] = []
        self.shown_shift: float | None = None
        self.feedback_now = False
        self._driving = False
        self._last_click: float | None = None
        self.signal_target = Target(0, None, "increase", threshold=0.5, mode="fixed",
                                    name="pupil", unit="")
        self.signal_target._settled = True
        self.targets = [self.signal_target]
        self._accumulated = 0.0
        self._resumed_at: float | None = None
        self.paused = True
        self._stop = False
        self.redraw: Callable[[], None] = lambda: None

    # --- what the shared Feedback pane reads ------------------------------------------------
    # Declared one by one, as on the HEG screen, and for its reason.

    offers = ("sound", "click", "music")
    """No shadow. A darker screen enlarges the pupil, so the shadow would reward itself."""
    music_sources = ("tone", "noise", "shepard", "file")
    combines = ("weighted",)
    pinned_feedback = frozenset()
    arrangements: tuple[str, ...] = ()
    state_ends = None
    tone_bias = None
    unbiased = False
    gates = None
    scores_bands = False
    sustains_reward = False
    hold_choices = None
    hold_seconds = 0.0
    tunes_envelope = False
    conditions_signal = False

    def level_note(self) -> tuple[str, str] | None:
        if self.error is not None or self.eye.stale or not self.eye.ready:
            return None
        if self.mode == "active" and self._signal is None:
            return ("no level outside the effort window",
                    "the level runs for the 15 s of effort in each trial, and not in a trial without "
                    "feedback")
        return None

    def dead_channels(self) -> list[str]:
        if self.error is not None or self.eye.stale or not self.eye.ready:
            return ["the eye"]
        return []

    # --- surviving a stop -------------------------------------------------------------------

    def adopt(self, previous: "PupilSession") -> None:
        if previous is self or previous.protocol.code != self.protocol.code:
            return
        self.plan = previous.plan
        self.plan.rewind()
        self.feedback = previous.feedback
        for name in ("mode", "direction", "n_trials", "transfer_fraction", "success_mm", "range_mm",
                     "feedback_window_s", "update_s"):
            setattr(self, name, getattr(previous, name))
        self.set_passive_window(previous.passive_window_s)
        self.restart_trials()

    def calibration(self) -> dict:
        return prefs.capture(self.feedback)

    def restore_calibration(self, values: dict) -> int:
        return prefs.apply(self.feedback, values)

    # --- what the operator sets -------------------------------------------------------------

    def _new_run(self) -> TrialRun:
        return TrialRun(n_trials=self.n_trials, baseline_s=BASELINE_S, active_s=ACTIVE_S,
                        result_s=RESULT_S, iti_s=ITI_S, cue_s=CUE_S,
                        transfer_fraction=self.transfer_fraction,
                        directions=DIRECTIONS[self.direction],
                        alternate=self.direction == "alternating",
                        success_uv=self.success_mm, metric_hz=OBSERVE_HZ)

    def restart_trials(self) -> None:
        """A new run with the current settings. The results go: they were scored under the old ones."""
        self.trials = self._new_run()
        self.finished = False
        self.count_from = None
        self._last_phase = None
        self._go_quiet()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self.mode:
            return
        self.mode = mode
        self._scale.reset()
        self.restart_trials()

    def set_direction(self, direction: str) -> None:
        if direction not in DIRECTIONS or direction == self.direction:
            return
        self.direction = direction
        self.restart_trials()

    def set_range(self, mm: float) -> None:
        self.range_mm = max(MIN_RANGE_MM, float(mm))

    def set_passive_window(self, seconds: float) -> None:
        self.passive_window_s = max(10.0, float(seconds))
        self._scale = normalise.ZScore(window_s=self.passive_window_s)
        self._signal = None

    # --- the level --------------------------------------------------------------------------

    @property
    def level_source(self) -> tuple[str, str]:
        if self.mode == "passive":
            return ("pupil size against the last "
                    f"{self.passive_window_s:.0f} s", "half = your mean over that time")
        return ("the change from this trial's baseline, toward the cue",
                f"half = the baseline · full at ±{self.range_mm:g} mm")

    @property
    def level(self) -> float:
        return self._level

    @property
    def signal(self) -> float | None:
        return self._signal

    @property
    def active_targets(self) -> list[Target]:
        return self.targets if self._signal is not None else []

    def channel_targets(self, channel: str) -> list[Target]:
        return self.active_targets

    @property
    def state_balance(self) -> float | None:
        """None: only the crossfade reads a balance, and this screen does not offer it."""
        return None

    @property
    def required(self) -> str | None:
        run = self.trials
        if self.mode != "active" or run.phase not in (CUED, BASELINE, ACTIVE, RESULT):
            return None
        return run.required

    @property
    def hides_direction(self) -> bool:
        """Whether this direction keeps the aim out of sight until the effort starts."""
        return self.direction in HIDDEN_DIRECTIONS

    def direction_known(self) -> bool:
        """Whether the person can now see the aim. A hidden-direction trial shows it only once the effort
        starts, so the aim stays out of the cue and the baseline."""
        if not self.hides_direction:
            return True
        return self.trials.phase in (ACTIVE, RESULT)

    def trial_level(self, shift_mm: float | None) -> float | None:
        """A shift from the baseline as a level: the middle at the baseline, the top at `range_mm` toward
        the cue. Oriented here, because every mechanism reads one number and none knows the direction."""
        if shift_mm is None or self.trials.required is None:
            return None
        oriented = shift_mm if self.trials.required == POSITIVE else -shift_mm
        return min(1.0, max(0.0, 0.5 + oriented / (2 * max(MIN_RANGE_MM, self.range_mm))))

    def circle_level(self, shift_mm: float | None) -> float | None:
        """The feedback circle's size, **mirroring the pupil, not the goal**: bigger than the baseline
        when the pupil is bigger, smaller when it is smaller, whichever direction the trial asks for. So
        shrinking the pupil always shrinks the circle inside the baseline ring, which is what the eye
        expects — the colour, not the size, says whether that was the way the cue asked."""
        if shift_mm is None:
            return None
        return min(1.0, max(0.0, 0.5 + shift_mm / (2 * max(MIN_RANGE_MM, self.range_mm))))

    def observe(self) -> None:
        """One reading from the stream into the level, the trials and the outputs. Once per draw."""
        if self.error is not None or self.eye.stale:
            self.eye.steer(0.0)
            self._go_quiet()
            return
        state = self.state
        if state is None or state.t == self._last_t:
            return
        self._last_t = state.t
        self.diameter = state.diameter_mm
        self.history.append((state.t, state.diameter_mm, state.valid))
        while state.t - self.history[0][0] > TRACE_SECONDS:
            self.history.popleft()
        if self.mode == "passive":
            self._observe_passive(state)
        else:
            self._observe_active(state)

    def _observe_passive(self, state) -> None:
        self.eye.steer(0.0)
        if state.diameter_mm is None:
            return
        if not state.valid:
            return                      # a blink holds: no new value, and the outputs keep running
        self._move_level(self._scale.push(state.diameter_mm, state.t), state.t)

    def _observe_active(self, state) -> None:
        run = self.trials
        if run.done:
            return
        kept = state.valid and state.diameter_mm is not None
        for event in run.feed(state.t, state.diameter_mm if kept else 0.0, valid=kept):
            self._sound(event)
            if event.kind == CUE:
                self.count_from = random.randint(300, 999)
        if run.phase != self._last_phase:
            if run.phase == ACTIVE:
                self._start_block(state.t)
                # A hidden-direction trial names its aim here, at the effort start, not at the cue.
                if self.hides_direction and run.required is not None:
                    self.cue(cues.tone_at(CUE_PITCH[run.required]), CUE_VOLUME)
            self._last_phase = run.phase
        effort = 0.0
        if run.phase == ACTIVE and run.required is not None:
            effort = 1.0 if run.required == POSITIVE else -1.0
        self.eye.steer(effort)
        self._update_shown(run, state.t)
        self.feedback_now = run.feedback_live and self._feedback_unfaded(run, state.t)
        driving = self.feedback_now and self.shown_shift is not None
        if driving:
            self._move_level(self.trial_level(self.shown_shift), state.t)
        elif self._driving:
            self._go_quiet()
        self._driving = driving
        if run.done and not self.finished:
            self.finished = True
            self._accumulated = self.elapsed
            self._resumed_at = None
            self.paused = True
            self.eye.steer(0.0)
            self._go_quiet()

    def _start_block(self, t: float) -> None:
        """At the start of an effort window, begin a fresh block. The circle sits at the baseline until the
        first stepwise update, or follows the pupil at once when the update is live."""
        self._block_at = t
        self._block_vals = []
        self.shown_shift = 0.0 if self.update_s > 0 else None

    def _update_shown(self, run, t: float) -> None:
        """The shift the circle shows. Live, it is the shift now. Stepwise, it holds the last block's mean
        and refreshes every `update_s` seconds, so the circle moves in steps the person must bridge."""
        if run.phase != ACTIVE or run.transfer or run.shift is None:
            return
        if self.update_s <= 0:
            self.shown_shift = run.shift
            return
        self._block_vals.append(run.shift)
        if self._block_at is None:
            self._block_at = t
        if t - self._block_at >= self.update_s:
            self.shown_shift = sum(self._block_vals) / len(self._block_vals)
            self._block_vals = []
            self._block_at = t

    def _feedback_unfaded(self, run, t: float) -> bool:
        """Whether the circle is still on, for the within-trial fade. It shows for the first
        `feedback_window_s` of the effort, then goes off so the person holds the change without it."""
        if self.feedback_window_s >= run.active_s:
            return True
        return run.active_s - run.phase_remaining(t) < self.feedback_window_s

    def _move_level(self, signal: float, now: float) -> None:
        self._signal = signal
        self.signal_target.observe(signal, OBSERVE_HZ)
        tau = max(0.0, self.feedback.smoothing_seconds)
        if tau <= 0 or self._level_at is None:
            self._level = signal
        else:
            dt = now - self._level_at
            if dt > 0:
                self._level += (1.0 - math.exp(-dt / tau)) * (signal - self._level)
        self._level_at = now
        self._drive_outputs()

    def _go_quiet(self) -> None:
        self._signal = None
        self._driving = False
        self.feedback_now = False
        self._level_at = None
        self.release_outputs()

    def _drive_outputs(self) -> None:
        met = self.feedback.on_target(self.channel_targets(CONDITION))
        snd = self.feedback.sound
        self.output_volume.apply(snd.enabled, *snd.ends(), snd.map(self._level))
        self.music.apply(self.feedback.music, hold=self.feedback.music.holds(met),
                         balance=self.state_balance, level=self._level)
        click = self.feedback.click
        if not click.enabled or not met:
            if not click.enabled:
                self._last_click = None
            return
        now = time.monotonic()
        if self._last_click is None or now - self._last_click >= click.interval_seconds:
            self._last_click = now
            self.cue(click.sound, click.volume)

    def _sound(self, event) -> None:
        """The cue tone at the start of a trial, and the same tone twice on a success. A trial that did
        not succeed is silent, as on the slow-potential screen."""
        if event.kind == CUE:
            pitch = CUE_PITCH_NEUTRAL if self.hides_direction else CUE_PITCH[event.required]
            self.cue(cues.tone_at(pitch), CUE_VOLUME)
        elif event.kind == VERDICT and event.trial is not None and event.trial.success:
            self.cue(cues.held(cues.tone_at(CUE_PITCH[event.required])), CUE_VOLUME)

    def release_outputs(self) -> None:
        self.output_volume.release()
        self.music.release()

    @property
    def error(self) -> str | None:
        return self.eye.error

    @property
    def state(self):
        return self.eye.state

    # --- transport ------------------------------------------------------------------------

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
        """Pause lets go of every output, as on every screen. The trial in flight is lost on resume:
        the trial engine sees the gap in time and drops it, since its baseline no longer precedes it."""
        if self.paused:
            if self.finished:
                return
            self.paused = False
            self._resumed_at = time.monotonic()
        else:
            self._accumulated = self.elapsed
            self._resumed_at = None
            self.paused = True
            self.eye.steer(0.0)
            self._go_quiet()

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
            self.eye.steer(0.0)
            self._go_quiet()

    def adopt_offered(self) -> None:
        if self._invented and self.offered is not None and self.offered.on:
            self.choice = self.offered
            self._invented = False

    def run(self) -> None:
        self.adopt_offered()
        self.eye.attach(self.choice)

    def stop(self) -> None:
        self._stop = True
        self.paused = True
        prefs.remember(self.protocol.code, self)
        self.eye.steer(0.0)
        self.release_outputs()
        self.eye.release()
        if self._invented:
            self.eye.choice = PupilChoice(kind="off")
            self.eye.state = None
            self.eye.error = None
            self.eye.seen_at = None
            self.eye.processor.reset()

    def suspend(self) -> None:
        if not self.paused:
            self.toggle_pause()
        self.release_outputs()

    @property
    def suspended(self) -> bool:
        return not self._stop

    def rearm(self) -> None:
        """Nothing to rebuild: the eye kept running and the clock stayed where `suspend` left it."""


MODE_HELP = (
    "**What you see.** A solid circle around a fixation point, and a dashed ring. The dashed ring is "
    "your baseline. The solid circle is your pupil now: it grows past the ring when your pupil grows, "
    "and shrinks inside the ring when your pupil shrinks. The line of the circle becomes thinner as it "
    "grows, so the screen gives off the same light at every size. A change in light changes the pupil, "
    "and that change must not come from the feedback.\n\n"
    "**The two modes.** In **passive** mode, the circle follows your pupil size against your own mean. "
    "A row called \"scaled over\" appears and sets the time for that mean. There are no trials and no "
    "direction. Use it to find what moves your pupil.\n\n"
    "In **active** mode, the screen runs trials. Each trial has four parts. First, a tone gives the "
    "direction: a high tone for larger, a low tone for smaller. Then, for "
    f"{BASELINE_S:g} s, count back by fours from the number on the screen. This measures your baseline. "
    f"Then, for {ACTIVE_S:g} s, move your pupil. To make it smaller, pull the circle inside the ring; to "
    "make it larger, push it outside. Last, the circle turns green for a success or red for no success, "
    "and you hear the tone two times for a success.\n\n"
    "**The direction.** **increase** asks for a larger pupil in every trial. **decrease** asks for a "
    "smaller pupil in every trial. **alternating** asks for larger and smaller in turn. **random** asks "
    "for larger and smaller in a random order. **random** does not show the direction until the effort "
    "starts, and the cue tone is a middle tone. So you cannot move your pupil the wrong way in the "
    "baseline to make the shift look large.\n\n"
    "**What to do.** Look at the fixation point for the whole session. Keep the room light constant. To "
    "make the pupil larger, imagine a situation with strong emotion, such as fear or joy. To make it "
    "smaller, imagine a safe situation and relax, with your attention on your body and your breath. Do "
    "not look at a near point and do not squint. Both change the pupil without the skill that you "
    "train.\n\n"
    "**The settings.** **trials** sets the number of trials in a run. **without feedback** sets the "
    "share of trials that show no level. In those trials you hear the cue and the result only. The TRIALS "
    "panel counts these trials on their own. **succeeds at** sets how far past the baseline the mean of a "
    "trial must be. **full at** sets the change that moves the level to its end. **show circle for** sets "
    "how many seconds of the effort show the circle. After that the circle stops, and you hold the change "
    "without it. **update every** makes the circle refresh in steps: it waits that many seconds, then "
    "shows the average over them. Set it to 0 for a circle that moves all the time. Use these two rows to "
    "train with less help, before a trial without feedback. A change to **trials**, **without feedback**, "
    "**succeeds at** or **full at** starts a new run."
)


def session_for(protocol: Protocol, path=None, device=None, length_seconds=None, source=None,
                channels: int = 0, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice: PupilChoice | None = None,
                pupil_runner: pupil.PupilRunner | None = None) -> PupilSession:
    """The other screens' signature, so `app.pick` need not special-case it."""
    _ = path, device, source, channels, heart_choice, heart_runner, mendi_choice, mendi_runner
    choice = pupil_choice if (pupil_choice is not None and pupil_choice.on) else None
    return PupilSession(protocol, choice=choice, length_seconds=length_seconds, runner=pupil_runner,
                        offered=pupil_choice)


def _instrument(session: PupilSession) -> tuple[str, str, str]:
    runner = session.eye
    name = "simulated eye" if session.choice.kind == "simulator" else "eye tracker"
    if runner.error:
        return name, "not connected · see the device screen", t.BAD
    if runner.stale:
        return name, "connected, no readings for a while", t.WARN
    if runner.ready:
        return name, "readings arriving", t.GOOD
    if runner.running:
        return name, "connected, waiting for readings", t.ACCENT
    return name, "not started", t.MUTED


def _direction_arrows(c: float, r: float, inward: bool) -> list:
    """Four faint arrows at a fixed ring, pointing in to ask for a smaller pupil or out for a larger one.
    A hint of which way to move it, kept translucent so it never competes with the circle itself. The
    ring is fixed, not on the moving circle, so the arrows never leave the canvas."""
    colour = ft.Colors.with_opacity(0.35, t.ACCENT)
    size = 16.0
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        bx, by = c + dx * r, c + dy * r                      # base of the arrow, on the fixed ring
        px, py = -dy, dx                                     # perpendicular, for the two tail corners
        s = -1 if inward else 1                              # point toward the centre, or away from it
        tip = (bx + dx * size * s, by + dy * size * s)
        a = (bx - dx * size * s * 0.2 + px * size * 0.7, by - dy * size * s * 0.2 + py * size * 0.7)
        b = (bx - dx * size * s * 0.2 - px * size * 0.7, by - dy * size * s * 0.2 - py * size * 0.7)
        out.append(cv.Path(
            elements=[cv.Path.MoveTo(*tip), cv.Path.LineTo(*a), cv.Path.LineTo(*b), cv.Path.Close()],
            paint=ft.Paint(color=colour, style=ft.PaintingStyle.FILL)))
    return out


def circle_shapes(side: float, level: float | None, colour: str, baseline: bool = True,
                  arrows: str | None = None) -> list:
    """The feedback circle on a fixation point, with a **constant number of lit pixels**.

    The line's width is inversely proportional to the radius, so the ink is the same at every size. A
    circle that grew in a constant line would light more of the screen as the pupil enlarged, and the
    pupil would answer the light. `arrows` draws the faint direction hint: "shrink" or "expand".
    """
    c = side / 2
    r_min, r_max = 0.16 * side, 0.46 * side
    r_mid = (r_min + r_max) / 2
    ink = 2 * math.pi * r_mid * 4.0                 # a 4 px line at the middle size
    out = [cv.Circle(c, c, 3.0, paint=ft.Paint(color=t.TEXT, style=ft.PaintingStyle.FILL))]
    if baseline:
        out.append(cv.Circle(c, c, r_mid, paint=ft.Paint(
            color=t.FAINT, stroke_width=1.0, style=ft.PaintingStyle.STROKE,
            stroke_dash_pattern=[5, 5])))
    r = r_min + min(1.0, max(0.0, level)) * (r_max - r_min) if level is not None else r_mid
    if arrows in ("shrink", "expand"):
        out += _direction_arrows(c, 0.42 * side, inward=arrows == "shrink")
    if level is not None:
        out.append(cv.Circle(c, c, r, paint=ft.Paint(
            color=colour, stroke_width=ink / (2 * math.pi * r), style=ft.PaintingStyle.STROKE)))
    return out


def _direction_lines(pool: list, prefix: str) -> list[str]:
    """One line per direction that has trials: the success count and the signed mean shift, mm."""
    out = []
    for required, word in ((POSITIVE, "larger"), (NEGATIVE, "smaller")):
        trials = [r for r in pool if r.required == required]
        if trials:
            mean = sum(r.score_uv for r in trials) / len(trials)
            out.append(f"{prefix}{word}: {sum(r.success for r in trials)} of {len(trials)} "
                       f"· mean {mean:+.2f} mm")
    return out


def trial_summary(run: TrialRun) -> str:
    """The TRIALS panel text: per direction, how many trials succeeded and the mean shift from the
    baseline. When the run has no-feedback trials, they are counted on their own, because a shift held
    without the circle is the real test of control. The signed mean says more than the count: a pool that
    leans the asked-for way shows voluntary control building, before many trials cross the success margin."""
    if not run.results:
        return "no trials yet"
    valid = run.valid_results
    if any(r.transfer for r in run.results):
        lines = (_direction_lines([r for r in valid if not r.transfer], "feedback · ")
                 + _direction_lines([r for r in valid if r.transfer], "no feedback · "))
    else:
        lines = _direction_lines(valid, "")
    void = sum(1 for r in run.results if not r.valid)
    if void:
        lines.append(f"{void} with no result")
    return "\n".join(lines) if lines else "no result yet"


def view(page: ft.Page, session: PupilSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width() -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - SIDE_W - 76 - AXIS_W)

    # --- the fixation panel --------------------------------------------------------------------
    circle = cv.Canvas(shapes=[], width=CIRCLE_SIDE, height=CIRCLE_SIDE)
    direction_text = ft.Text("", size=t.VALUE, color=t.TEXT, weight=ft.FontWeight.W_300)
    phase_text = ft.Text("", size=t.SMALL, color=t.FAINT)
    count_text = ft.Text("", size=t.BODY, color=t.MUTED)
    verdict_text = ft.Text("", size=t.SMALL, color=t.MUTED)

    # --- the diameter trace --------------------------------------------------------------------
    history = session.history
    trace = cv.Canvas(shapes=[], expand=True, height=TRACE_H)
    trace_axis, trace_labels = value_axis(TRACE_H, count=4, width=AXIS_W)
    trace_caption = ft.Text("", size=t.SMALL, color=t.FAINT)

    # --- the tiles -----------------------------------------------------------------------------
    numbers: dict[str, ft.Text] = {}
    tiles = ft.Row(spacing=10, wrap=True)
    for label, unit in (("diameter", "mm"), ("rejected", "last 30 s"), ("blinks", "a minute")):
        value = ft.Text("—", size=17, color=t.TEXT, weight=ft.FontWeight.W_300)
        numbers[label] = value
        tiles.controls.append(ft.Container(
            content=ft.Column([ft.Text(label, size=t.SMALL, color=t.FAINT), value,
                               ft.Text(unit, size=t.SMALL, color=t.FAINT)], spacing=1),
            width=96, height=70, padding=ft.Padding(10, 9, 10, 9), border_radius=8,
            bgcolor=t.BG, border=ft.Border.all(1, t.BORDER)))
    source_note = ft.Text("", size=t.SMALL, color=t.MUTED)

    # --- the mode panel ------------------------------------------------------------------------
    mode_rows = ft.Column(spacing=10)
    history_cells = ft.Row(spacing=3, wrap=True)
    history_text = ft.Text("", size=t.SMALL, color=t.MUTED)
    history_panel = panel("TRIALS", history_cells, ft.Container(height=8), history_text)

    def changed(action) -> Callable:
        return lambda *_a: (action(*_a), rebuild(), draw())

    def set_trials(n: float) -> None:
        session.n_trials = int(n)
        session.restart_trials()

    def set_transfer(percent: float) -> None:
        session.transfer_fraction = max(0.0, min(1.0, percent / 100.0))
        session.restart_trials()

    def set_margin(mm: float) -> None:
        session.success_mm = max(0.0, float(mm))
        session.restart_trials()

    def set_feedback_window(s: float) -> None:
        session.feedback_window_s = max(1.0, min(ACTIVE_S, float(s)))

    def set_update(s: float) -> None:
        session.update_s = max(0.0, min(ACTIVE_S, float(s)))

    def audition(required: str, mark: bool = False) -> None:
        sound = cues.tone_at(CUE_PITCH[required])
        session.cue(cues.held(sound) if mark else sound, CUE_VOLUME)

    def rebuild() -> None:
        rows: list[ft.Control] = [
            labelled("train", *[pill(m, session.mode == m, changed(lambda _e, m_=m: session.set_mode(m_)))
                                for m in MODES], width=96),
        ]
        if session.mode == "passive":
            rows.append(labelled("scaled over",
                                 num_box(session.passive_window_s,
                                         changed(lambda v: session.set_passive_window(v)),
                                         suffix="s", step=15, low=15, high=600, width=52),
                                 note="your mean is the middle of the level", width=96))
        else:
            rows += [
                labelled("direction",
                         ft.Row([pill(d, session.direction == d,
                                      changed(lambda _e, d_=d: session.set_direction(d_)))
                                 for d in DIRECTIONS], spacing=6, wrap=True, expand=True),
                         width=96),
                labelled("trials", num_box(session.n_trials, changed(set_trials), step=5, low=1,
                                           high=200, width=52),
                         note=f"about {_duration(session.n_trials * TRIAL_S)}", width=96),
                labelled("without feedback",
                         num_box(round(session.transfer_fraction * 100), changed(set_transfer),
                                 suffix="%", step=10, low=0, high=100, width=52),
                         note="of the trials", width=96),
                labelled("show circle for",
                         num_box(session.feedback_window_s, changed(set_feedback_window),
                                 suffix="s", step=1, low=1, high=ACTIVE_S, width=58),
                         note=("the whole effort" if session.feedback_window_s >= ACTIVE_S
                               else f"then no circle for {ACTIVE_S - session.feedback_window_s:g} s"),
                         width=96),
                labelled("update every",
                         num_box(session.update_s, changed(set_update), suffix="s",
                                 step=0.5, low=0, high=ACTIVE_S, width=58),
                         note=("0 = live" if session.update_s <= 0
                               else f"steps of {session.update_s:g} s"), width=96),
                labelled("succeeds at", num_box(session.success_mm, changed(set_margin), suffix="mm",
                                                step=0.05, low=0, high=1.0, width=58),
                         note="past the baseline", width=96),
                labelled("full at", num_box(session.range_mm, changed(session.set_range),
                                            suffix="mm", step=0.05, low=MIN_RANGE_MM, high=2.0,
                                            width=58),
                         note="from the baseline", width=96),
                ft.Row([button("restart run", changed(lambda _e: session.restart_trials()))]),
                ft.Row([test_button("larger", lambda _e: audition(POSITIVE)),
                        test_button("smaller", lambda _e: audition(NEGATIVE)),
                        test_button("success", lambda _e: audition(POSITIVE, mark=True))],
                       spacing=6),
            ]
        mode_rows.controls = rows
        history_panel.visible = session.mode == "active"
        page.update()

    left = ft.Column([
        panel("FIXATION",
              ft.Row([ft.Container(content=circle, width=CIRCLE_SIDE, height=CIRCLE_SIDE)],
                     alignment=ft.MainAxisAlignment.CENTER),
              ft.Container(height=6),
              ft.Row([direction_text], alignment=ft.MainAxisAlignment.CENTER),
              ft.Row([count_text], alignment=ft.MainAxisAlignment.CENTER),
              ft.Container(height=4),
              ft.Row([phase_text], alignment=ft.MainAxisAlignment.CENTER),
              ft.Row([verdict_text], alignment=ft.MainAxisAlignment.CENTER)),
        ft.Container(height=12),
        panel("DIAMETER",
              ft.Row([trace_axis, ft.Container(content=trace, expand=True, height=TRACE_H)],
                     spacing=0),
              ft.Container(height=6), trace_caption),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    right = ft.Column([
        panel("MODE", mode_rows, help=help_button(page, "Pupil training", MODE_HELP)),
        ft.Container(height=12),
        history_panel,
        ft.Container(height=12),
        panel("NOW", tiles),
        ft.Container(height=12),
        panel("SOURCE", source_note),
    ], spacing=0, width=SIDE_W, scroll=ft.ScrollMode.AUTO)

    train_tab = ft.Container(content=ft.Row([left, right], spacing=14, expand=True),
                             padding=ft.Padding(14, 12, 14, 14), expand=True)
    signal = signal_pane.SignalPane(page, session, inset=False)
    feedback_tab = ft.Container(
        content=ft.Column([signal.control], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO),
        padding=ft.Padding(14, 12, 14, 14), expand=True, visible=False)
    panes = [train_tab, feedback_tab, plan_pane_.control]

    def select_tab(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs(["Train", "Feedback", "Session"], select_tab)
    shown: list = [None]

    def draw_fixation() -> None:
        run = session.trials
        verdict_text.value = ""
        if session.mode == "passive":
            direction_text.value = "passive"
            direction_text.color = t.MUTED
            count_text.value = ""
            phase_text.value = ("the circle follows your pupil · no trials"
                                if session.signal is not None else "waiting for the pupil")
            circle.shapes = circle_shapes(CIRCLE_SIDE, session.signal, t.ACCENT)
            return
        required = session.required
        if session.state is None:
            direction_text.value, direction_text.color = "waiting", t.FAINT
            count_text.value = ""
            phase_text.value = "no readings from the eye yet"
            circle.shapes = circle_shapes(CIRCLE_SIDE, None, t.ACCENT)
            return
        if run.done:
            direction_text.value, direction_text.color = f"done · {len(run.results)} trials", t.MUTED
        elif required is None:
            direction_text.value, direction_text.color = "rest", t.FAINT
        elif not session.direction_known():
            direction_text.value, direction_text.color = "?", t.FAINT
        else:
            direction_text.value, direction_text.color = DIRECTION_WORD[required], t.TEXT
        count_text.value = (f"count back by fours from {session.count_from}"
                            if run.phase in (CUED, BASELINE) and session.count_from else "")
        name = {CUED: "cue", BASELINE: "baseline", ACTIVE: "effort", RESULT: "result",
                ITI: "rest"}.get(run.phase, run.phase)
        phase_text.value = (f"trial {run.trial_number} of {run.n_trials} · {name} · "
                            f"{run.phase_remaining():.0f} s" if not run.done else
                            "the run is over · restart it in the mode panel")
        if not run.done and required is not None and not session.direction_known():
            phase_text.value += " · wait for the tone"
        level, colour = None, t.ACCENT
        if run.phase == ACTIVE and session.feedback_now:
            level = session.circle_level(session.shown_shift)
            goal = "inside the ring" if required == NEGATIVE else "outside the ring"
            phase_text.value += f" · pull the circle {goal}"
        elif run.phase == ACTIVE and run.transfer:
            phase_text.value += " · no feedback in this trial"
        elif run.phase == ACTIVE:
            phase_text.value += " · hold it · the circle is off"
        if run.phase == RESULT and run.last is not None:
            last = run.last
            asked = DIRECTION_WORD[last.required]
            if not last.valid:
                level, colour = None, t.MUTED
                direction_text.value, direction_text.color = "no result", t.MUTED
                verdict_text.value = "too many readings were rejected in this trial"
            else:
                level = session.circle_level(last.score_uv)
                colour = t.GOOD if last.success else t.BAD
                went = ("smaller" if last.score_uv < 0 else
                        "larger" if last.score_uv > 0 else "no change")
                direction_text.value = f"{'✓' if last.success else '✗'}  {asked}"
                direction_text.color = colour
                if went == "no change":
                    verdict_text.value = f"asked for {asked} · your pupil did not change"
                else:
                    verdict_text.value = (f"asked for {asked} · your pupil went "
                                          f"{abs(last.score_uv):.2f} mm {went}")
        arrows = None
        if (run.phase in (CUED, BASELINE, ACTIVE) and not run.transfer and required is not None
                and session.direction_known()
                and not (run.phase == ACTIVE and not session.feedback_now)):
            arrows = "shrink" if required == NEGATIVE else "expand"
        circle.shapes = circle_shapes(CIRCLE_SIDE, level, colour, arrows=arrows)

    def draw_trace() -> None:
        values = [d for _t, d, _k in history if d is not None]
        width = plot_width()
        if not values:
            trace.shapes = []
            for label in trace_labels:
                label.value = ""
            trace_caption.value = "no readings yet"
            return
        lo, hi = min(values), max(values)
        mid, span = (lo + hi) / 2, max(0.6, hi - lo) * 1.15
        top = math.ceil((mid + span / 2) * 10) / 10
        bottom = math.floor((mid - span / 2) * 10) / 10
        end = history[-1][0]

        def x_of(when: float) -> float:
            return width * (1.0 - (end - when) / TRACE_SECONDS)

        def y_of(mm: float) -> float:
            return TRACE_H * (top - mm) / (top - bottom)

        points = [ft.Offset(x_of(when), y_of(d)) for when, d, _k in history if d is not None]
        shapes = grid(width, TRACE_H, rows=3, cols=6) + [polyline(points, t.ACCENT, 1.4)]
        baseline = session.trials.baseline_uv
        if session.mode == "active" and baseline is not None and session.trials.phase in (ACTIVE, RESULT):
            shapes += dashed(y_of(baseline), width, t.MUTED)
        n = len(history)
        if n:
            shapes += mark_shapes([not k for _t, _d, k in history], width * (history[-1][0] - history[0][0])
                                  / TRACE_SECONDS if n > 1 else width, TRACE_H, t.WARN)
        trace.shapes = shapes
        place_value_labels(trace_labels, TRACE_H, top, bottom, unit="mm")
        trace_caption.value = (f"{TRACE_SECONDS:.0f} s across · the line holds through a blink · "
                               "amber below = rejected readings"
                               + (" · dashed = this trial's baseline" if session.mode == "active" else ""))

    def draw_side() -> None:
        state = session.state
        source_note.value = session.eye.describe()
        source_note.color = t.BAD if session.eye.error else t.MUTED
        if state is None:
            for value in numbers.values():
                value.value = "—"
        else:
            numbers["diameter"].value = "—" if state.diameter_mm is None else f"{state.diameter_mm:.2f}"
            numbers["rejected"].value = f"{state.rejected_fraction:.0%}"
            numbers["blinks"].value = ("—" if state.blinks_per_min is None
                                       else f"{state.blinks_per_min:.0f}")
        if session.mode != "active":
            return
        run = session.trials
        cells: list[ft.Control] = []
        for result in run.results:
            if not result.valid:
                colour, border = t.SURFACE_HI, t.BORDER
            elif result.success:
                colour, border = t.GOOD, t.GOOD
            else:
                colour, border = None, t.BAD
            cells.append(ft.Container(width=CELL_W, height=CELL_H, border_radius=3, bgcolor=colour,
                                      border=ft.Border.all(1, border)))
        history_cells.controls = cells
        history_text.value = trial_summary(run)

    def draw() -> None:
        session.tick()
        instrument = _instrument(session)
        if instrument != shown[0]:
            shown[0] = instrument
            toolbar.set_source(instrument[0], note=instrument[1], tone=instrument[2])
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        if session.running:
            session.observe()
        draw_fixation()
        draw_trace()
        draw_side()
        signal.refresh()
        page.update()

    rebuild()
    toolbar.set_channels_visible(False)
    session.redraw = draw
    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control, ft.Stack(panes, expand=True)],
                     spacing=0, expand=True)
