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
import time
from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv

from . import chrome, cues, heart_panel, normalise, plan_pane, prefs, pupil, signal_pane
from . import theme as t
from .feedback import FeedbackSettings
from .inputs import PupilChoice
from .outputs import LevelOutputs
from .plan import SessionPlan
from .plots import dashed, grid, mark_shapes, place_value_labels, polyline, value_axis
from .protocols import Protocol
from .targets import Target
from .trial_training import MODES, OBSERVE_HZ, TrialTiming, TrialTraining, TrialUnits, TrialView
from .trials import ACTIVE, NEGATIVE, POSITIVE, RESULT
from .widgets import help_button, labelled, num_box, panel, pill

HANDLES = ("PUPIL",)
EEG_SOURCE = False      # takes its own input; the replay EEG file is not opened for it

SIDE_W = 360
CIRCLE_SIDE = 280
TRACE_H = 150
TRACE_SECONDS = 60.0
AXIS_W = 46
CELL_W, CELL_H = 9, 14

CUE_S = 2.0
BASELINE_S = 5.0
ACTIVE_S = 15.0
RESULT_S = 2.0
ITI_S = (4.0, 6.0)
"""The published protocol's trial: a 2 s cue, a baseline whose last 5 s are averaged, 15 s of
regulation, 2 s of verdict and a short rest. About 29 s a trial."""
TRIAL_S = CUE_S + BASELINE_S + ACTIVE_S + RESULT_S + sum(ITI_S) / 2

TIMING = TrialTiming(cue_s=CUE_S, baseline_s=BASELINE_S, active_s=ACTIVE_S, result_s=RESULT_S, iti_s=ITI_S)

UNITS = TrialUnits(unit="mm", full_at=0.3, min_full_at=0.05, max_full_at=2.0, max_success=1.0, step=0.05)
"""Voluntary changes are 0.1 to 0.5 mm, so a level full at 0.3 mm puts a good trial near the top without
pinning it there. Below 0.05 mm the level shows measurement noise, not the person."""

PASSIVE_WINDOW_S = 60.0

DIRECTION_WORD = {POSITIVE: "larger", NEGATIVE: "smaller"}


class PupilSession(TrialTraining, LevelOutputs):
    """The eye, the trials and the level. Not a `TrainingSession`, for the reason `HEGSession` is not:
    there are no sample blocks here, and the readings come from a runner the app owns."""

    TIMING = TIMING
    UNITS = UNITS
    DIRECTION_WORD = DIRECTION_WORD
    COUNTS_IN_BASELINE = True
    """Counting backwards keeps the mind busy and level through the baseline, as the published protocol does."""
    trial_subject = "your pupil"
    trial_waiting = "no readings from the eye yet"

    def __init__(self, protocol: Protocol, choice: PupilChoice | None = None,
                 length_seconds: float | None = None, runner: pupil.PupilRunner | None = None,
                 offered: PupilChoice | None = None, heart_choice=None, heart_runner=None):
        self.protocol = protocol
        self.plan = SessionPlan(length_seconds)
        self.cue = cues.play
        # A simulated eye by default here only, as on the HEG screen, and handed back switched off by
        # `stop` — see `HEGSession._invented`.
        self._invented = choice is None
        self.choice = choice if choice is not None else PupilChoice(kind="simulator")
        self.offered = offered if offered is not None else choice
        self.eye = runner if runner is not None else pupil.PupilRunner(self.choice)
        # Borrowed, not owned: this screen never attaches or releases it, so the HRV screen can share it.
        self.heart_choice = heart_choice
        self.heart = heart_runner
        self.finished = False

        self.feedback = FeedbackSettings()
        self._init_outputs()
        for name, mech in self.feedback.mechanisms.items():
            if name not in self.offers:
                mech.enabled = False
        self.feedback.music.enabled = True
        self.feedback.music.source = "tone"

        self.passive_window_s = PASSIVE_WINDOW_S
        self._scale = normalise.ZScore(window_s=self.passive_window_s)
        self.diameter: float | None = None
        # The trace's points, kept here rather than in the view: a trip to the device screen rebuilds the
        # view, and the last minute of the pupil is the session's.
        self.history: deque[tuple[float, float | None, bool]] = deque()

        self._signal: float | None = None
        self._level = 0.0
        self._level_at: float | None = None
        self._init_trials("active")
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
        return self.no_level_note()

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
        self.set_passive_window(previous.passive_window_s)
        self.adopt_trials(previous)

    def calibration(self) -> dict:
        return prefs.capture(self.feedback)

    def restore_calibration(self, values: dict) -> int:
        return prefs.apply(self.feedback, values)

    # --- what the operator sets -------------------------------------------------------------

    def _mode_changed(self) -> None:
        self._scale.reset()

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
                f"half = the baseline · full at ±{self.full_at:g} mm")

    @property
    def level(self) -> float:
        return self._level

    @property
    def signal(self) -> float | None:
        return self._signal

    @property
    def heart_on(self) -> bool:
        """Whether a heart-rate source is chosen, so the heart panel and stats belong on the screen."""
        return self.heart is not None and self.heart_choice is not None and self.heart_choice.on

    def _heart_rate(self) -> float | None:
        """The latest instantaneous heart rate, or None when there is no live reading to trust. A stale
        runner holds its last reading, so a dead strap must not keep feeding one figure into the stats."""
        if not self.heart_on or self.heart.stale or self.heart.error:
            return None
        state = self.heart.state
        return state.hr_bpm if state is not None else None

    @property
    def active_targets(self) -> list[Target]:
        return self.targets if self._signal is not None else []

    def channel_targets(self, channel: str) -> list[Target]:
        return self.active_targets

    @property
    def state_balance(self) -> float | None:
        """None: only the crossfade reads a balance, and this screen does not offer it."""
        return None

    def observe(self) -> None:
        """The readings since the last draw, taken at `OBSERVE_HZ` of the eye's time, into the level, the
        trials and the outputs, in order. Once per draw."""
        if self.error is not None or self.eye.stale:
            self.eye.steer(0.0)
            self._go_quiet()
            return
        for state in self.due_readings(self.eye.readings):
            self._observe_one(state)

    def _observe_one(self, state) -> None:
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
        kept = state.valid and state.diameter_mm is not None
        self._observe_trial(state.t, state.diameter_mm if kept else None, kept)

    def _steer(self, effort: float) -> None:
        self.eye.steer(effort)

    def _move_level(self, signal: float, now: float, smoothed: bool = True) -> None:
        self._signal = signal
        self.signal_target.observe(signal, OBSERVE_HZ)
        tau = max(0.0, self.feedback.smoothing_seconds) if smoothed else 0.0
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
            # What arrived while paused is not part of the run. The first reading after it is then a gap.
            self.eye.readings.clear()
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
        self.start_outputs()

    def stop(self) -> None:
        self._stop = True
        self.paused = True
        prefs.remember(self.protocol.code, self)
        self.eye.steer(0.0)
        self.stop_outputs()
        if not self._invented:
            self.eye.release_aside()
            return
        # Cleared after the release, or the reader writes into it again.
        self.eye.choice = PupilChoice(kind="off")
        runner = self.eye

        def clear() -> None:
            runner.state = None
            runner.error = None
            runner.seen_at = None
            runner.processor.reset()

        runner.release_aside(clear)

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
    "without it. **update every** makes the feedback refresh in steps: it waits that many seconds, then "
    "shows the average over them. The circle and the sound step together. Set it to 0 for feedback that "
    "moves all the time. Use these two rows to "
    "train with less help, before a trial without feedback. A change to **trials**, **without feedback**, "
    "**succeeds at** or **full at** starts a new run.\n\n"
    "**The heart.** When you choose a heart-rate source on the device screen, a heart panel appears here. "
    "It shows the mean heart-rate change from baseline for each direction. It also shows the heart rate "
    "across the run, so you can see how the effort moves the heart."
)


def session_for(protocol: Protocol, path=None, device=None, length_seconds=None, source=None,
                channels: int = 0, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice: PupilChoice | None = None,
                pupil_runner: pupil.PupilRunner | None = None) -> PupilSession:
    """The other screens' signature, so `app.pick` need not special-case it."""
    _ = path, device, source, channels, mendi_choice, mendi_runner
    choice = pupil_choice if (pupil_choice is not None and pupil_choice.on) else None
    return PupilSession(protocol, choice=choice, length_seconds=length_seconds, runner=pupil_runner,
                        offered=pupil_choice, heart_choice=heart_choice, heart_runner=heart_runner)


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


def view(page: ft.Page, session: PupilSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width() -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - SIDE_W - 76 - AXIS_W)

    # --- the fixation panel, the trial settings, the trial history and the heart stats -----------
    trial_ui = TrialView(session, CIRCLE_SIDE, heart=session.heart_on)

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

    # --- the heart, when a source is chosen on the device screen -------------------------------
    heart_ui = (heart_panel.HeartPanel(compact=True, width_of=lambda: max(200.0, SIDE_W - 60))
                if session.heart_on else None)

    def changed(action) -> Callable:
        return lambda *_a: (action(*_a), rebuild(), draw())

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
            rows += trial_ui.settings_rows(changed)
        mode_rows.controls = rows
        trial_ui.history_panel.visible = session.mode == "active"
        page.update()

    left = ft.Column([
        trial_ui.fixation_panel(),
        ft.Container(height=12),
        panel("DIAMETER",
              ft.Row([trace_axis, ft.Container(content=trace, expand=True, height=TRACE_H)],
                     spacing=0),
              ft.Container(height=6), trace_caption),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    right_panels = [
        panel("MODE", mode_rows, help=help_button(page, "Pupil training", MODE_HELP)),
        ft.Container(height=12),
        trial_ui.history_panel,
    ]
    if heart_ui is not None:
        right_panels += [
            ft.Container(height=12),
            trial_ui.heart_stats_panel,
            ft.Container(height=12),
            panel("HEART", *heart_ui.spectrum_rows(),
                  ft.Container(height=10), heart_ui.tiles,
                  ft.Container(height=8), heart_ui.source_note),
        ]
    right_panels += [
        ft.Container(height=12),
        panel("NOW", tiles),
        ft.Container(height=12),
        panel("SOURCE", source_note),
    ]
    right = ft.Column(right_panels, spacing=0, width=SIDE_W, scroll=ft.ScrollMode.AUTO)

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
        trial_ui.draw_fixation("the circle follows your pupil · no trials" if session.signal is not None
                               else "waiting for the pupil")

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
        trial_ui.draw_history()

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
        if heart_ui is not None:
            heart_ui.refresh(session.heart)
            trial_ui.draw_heart_stats()
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
