"""Cued trials for a program that trains one signal up or down: the active mode of Pupil and HEG.

**Passive** feeds back the signal continuously, with no trial and no aim. **Active** runs trials: a cue
names the direction, a baseline is measured, the effort is fed back against that baseline, and the mean
shift over the effort is the verdict. The engine is `trials.TrialRun` with a cue phase in front of the
baseline, because a signal can answer the cue tone itself, and that answer must not sit inside the baseline.

A program supplies what differs: the durations (`TIMING`), the unit the shift is judged in (`UNITS`), the
words for the two directions, what the person does during the baseline, and how its readings become a value.
Everything else, from the settings to the circle, the trial history and the heart beside the trials, is here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import flet as ft
import flet.canvas as cv

from . import cues
from . import theme as t
from .trials import ACTIVE, BASELINE, CUE, CUED, ITI, NEGATIVE, POSITIVE, RESULT, VERDICT, TrialRun
from .widgets import button, labelled, num_box, panel, pill, stat_tile, test_button

MODES = ("passive", "active")
DIRECTIONS = {
    "increase": (POSITIVE,),
    "decrease": (NEGATIVE,),
    "alternating": (POSITIVE, NEGATIVE),
    "random": (POSITIVE, NEGATIVE),
}
HIDDEN_DIRECTIONS = ("random",)
"""Directions that keep the aim out of sight until the effort starts. The cue names a trial, but not its
direction, so the person cannot set the signal the wrong way in the baseline to make the shift look large."""

DEFAULT_TRIALS = 20

OBSERVE_HZ = 10.0
"""How many readings a second of the source's own time a trial takes. Counted in the source's time, not in
frames, so a slow frame takes the readings it missed."""

CUE_PITCH = {POSITIVE: 620.0, NEGATIVE: 310.0}
"""The slow-potential screen's two cue tones, for the same reason: an octave apart, so that the two
directions are told apart without looking."""
CUE_PITCH_NEUTRAL = 440.0
"""A tone between the two, for a hidden-direction trial. It starts the baseline without naming the aim,
so the directional tone can arrive later, at the start of the effort."""
CUE_VOLUME = 25

HEART_MIN_READINGS = 3
"""Fewest heart readings a trial window needs before its mean counts. Below it the window is too thin to
average, so the trial has no heart result rather than a number made from one or two beats."""

CELL_W, CELL_H = 9, 14


@dataclass(frozen=True)
class TrialTiming:
    """The parts of one trial, in seconds."""

    cue_s: float
    baseline_s: float
    active_s: float
    result_s: float
    iti_s: tuple[float, float]

    @property
    def trial_s(self) -> float:
        return self.cue_s + self.baseline_s + self.active_s + self.result_s + sum(self.iti_s) / 2


@dataclass(frozen=True)
class TrialUnits:
    """The unit a trial's shift is judged in, and the bounds of the two settings written in it."""

    unit: str
    full_at: float              # the default: the shift that takes the level to its end
    min_full_at: float          # below this the level shows measurement noise, not the person
    max_full_at: float
    max_success: float
    step: float
    decimals: int = 2           # how many decimals a shift is written with


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60.0)
    return f"{minutes} min" if minutes >= 1 else f"{seconds:.0f} s"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


class _Tally:
    """A running mean and range, so a whole session's heart rate needs no list kept."""

    __slots__ = ("total", "n", "lo", "hi")

    def __init__(self) -> None:
        self.total = 0.0
        self.n = 0
        self.lo: float | None = None
        self.hi: float | None = None

    def add(self, x: float) -> None:
        self.total += x
        self.n += 1
        self.lo = x if self.lo is None else min(self.lo, x)
        self.hi = x if self.hi is None else max(self.hi, x)

    @property
    def mean(self) -> float | None:
        return self.total / self.n if self.n else None


class HeartTrials:
    """Heart rate gathered against the trials, to show how the effort moves the heart.

    Fed one reading at a time with the trial phase, the trial number, the cue direction and the
    instantaneous heart rate, it keeps two views. Per trial: the mean rate over the baseline window and
    over the effort window, so each direction averages into one effort-against-baseline change. Per
    session: the rate over the whole run, and the rate during effort against the rate during the rest
    between trials, neither tied to a single trial.
    """

    def __init__(self, words: dict[str, str]) -> None:
        self.words = words
        self.reset()

    def reset(self) -> None:
        self._trial: int | None = None
        self._direction: str | None = None
        self._last_phase: str | None = None
        self._base: list[float] = []
        self._effort: list[float] = []
        self._scored = False
        self.by_direction: dict[str, list[float]] = {POSITIVE: [], NEGATIVE: []}
        self._all = _Tally()
        self._effort_all = _Tally()
        self._rest_all = _Tally()

    def feed(self, phase: str, trial: int, direction: str | None, hr: float | None) -> None:
        # Score when the effort ends, before the new-trial reset wipes the windows: the run advances its
        # trial number as the effort ends. A beat need not land on the result reading.
        if self._last_phase == ACTIVE and phase != ACTIVE and not self._scored:
            self._score()
        if trial != self._trial:
            self._trial = trial
            self._direction = None
            self._base = []
            self._effort = []
            self._scored = False
        if direction is not None:
            self._direction = direction
        self._last_phase = phase
        if hr is None:
            return
        self._all.add(hr)
        if phase == ACTIVE:
            self._effort_all.add(hr)
            self._effort.append(hr)
        elif phase == ITI:
            self._rest_all.add(hr)
        elif phase == BASELINE:
            self._base.append(hr)

    def _score(self) -> None:
        self._scored = True
        if (self._direction in self.by_direction
                and len(self._base) >= HEART_MIN_READINGS
                and len(self._effort) >= HEART_MIN_READINGS):
            self.by_direction[self._direction].append(_mean(self._effort) - _mean(self._base))

    @property
    def has_data(self) -> bool:
        return self._all.n > 0

    def direction_summary(self) -> list[tuple[str, float, int]]:
        """Per direction with scored trials: (the word on the screen, the mean change in bpm, the count)."""
        out = []
        for d in (POSITIVE, NEGATIVE):
            changes = self.by_direction[d]
            if changes:
                out.append((self.words[d], _mean(changes), len(changes)))
        return out

    def overall(self) -> tuple[float | None, float | None, float | None]:
        """The whole run: mean, low and high heart rate."""
        return self._all.mean, self._all.lo, self._all.hi

    def effort_vs_rest(self) -> tuple[float | None, float | None]:
        """Mean heart rate during the effort windows, and during the rest between trials."""
        return self._effort_all.mean, self._rest_all.mean


TRIAL_SETTINGS = ("mode", "trial_direction", "n_trials", "transfer_fraction", "success_at", "full_at",
                  "feedback_window_s", "update_s")
"""What `adopt` carries from a stopped session to the next: the operator's settings, not the results."""


class TrialTraining:
    """The trial half of a session. The session calls `_init_trials` once and `_observe_trial` with each
    reading in active mode, and supplies:

    - `TIMING`, `UNITS`, `DIRECTION_WORD` and `COUNTS_IN_BASELINE` (whether the baseline asks the person to
      count backwards, a task that keeps the mind busy and level);
    - `trial_subject` (what moved, in words: "your pupil") and `trial_waiting` (the line shown before the
      first reading);
    - `_move_level(signal, t, smoothed)`, `_go_quiet()`, `cue`, `elapsed` and `_heart_rate()`;
    - `_steer(effort)`, optionally, to tell a simulated source what the person is trying to do.
    """

    TIMING: TrialTiming
    UNITS: TrialUnits
    DIRECTION_WORD: dict[str, str]
    COUNTS_IN_BASELINE = False
    DIRECTIONS = DIRECTIONS
    """The direction choices this program offers, by name. A program can add its own."""
    DEFAULT_DIRECTION = "alternating"

    @property
    def trial_metric_hz(self) -> float:
        """How many readings a second reach the trial engine, for its check that a baseline has enough."""
        return OBSERVE_HZ

    def _init_trials(self, mode: str) -> None:
        self.mode = mode
        self.trial_direction = self.DEFAULT_DIRECTION
        self.n_trials = DEFAULT_TRIALS
        self.transfer_fraction = 0.0
        self.success_at = 0.0
        self.full_at = self.UNITS.full_at
        self.feedback_window_s = self.TIMING.active_s
        self.update_s = 0.0
        self.heart_trials = HeartTrials(self.DIRECTION_WORD)
        self.trials = self._new_run()
        self.count_from: int | None = None
        self.shown_shift: float | None = None
        self.feedback_now = False
        self._driving = False
        self._last_phase: str | None = None
        self._block_at: float | None = None
        self._block_vals: list[float] = []
        self._last_t: float | None = None
        self._due_t: float | None = None       # the source time from which the next reading is taken

    def _steer(self, effort: float) -> None:
        """Tell a simulated source what the person is trying to do. Nothing, by default."""

    # --- what the operator sets -------------------------------------------------------------

    def _new_run(self) -> TrialRun:
        timing = self.TIMING
        return TrialRun(n_trials=self.n_trials, baseline_s=timing.baseline_s, active_s=timing.active_s,
                        result_s=timing.result_s, iti_s=timing.iti_s, cue_s=timing.cue_s,
                        transfer_fraction=self.transfer_fraction,
                        directions=self.DIRECTIONS[self.trial_direction],
                        alternate=self.trial_direction == "alternating",
                        success_uv=self.success_at, metric_hz=self.trial_metric_hz)

    def restart_trials(self) -> None:
        """A new run with the current settings. The results go: they were scored under the old ones."""
        self.trials = self._new_run()
        self.finished = False
        self.count_from = None
        self._last_phase = None
        self.heart_trials.reset()
        self._go_quiet()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self.mode:
            return
        self.mode = mode
        self._mode_changed()
        self.restart_trials()

    def _mode_changed(self) -> None:
        """What a program resets when the mode changes, beyond the trials. Nothing, by default."""

    def set_trial_direction(self, direction: str) -> None:
        if direction not in self.DIRECTIONS or direction == self.trial_direction:
            return
        self.trial_direction = direction
        self.restart_trials()

    def set_full_at(self, value: float) -> None:
        self.full_at = max(self.UNITS.min_full_at, float(value))

    def adopt_trials(self, previous: "TrialTraining") -> None:
        for name in TRIAL_SETTINGS:
            setattr(self, name, getattr(previous, name))
        self.restart_trials()

    # --- where the trial is -----------------------------------------------------------------

    @property
    def required(self) -> str | None:
        run = self.trials
        if self.mode != "active" or run.phase not in (CUED, BASELINE, ACTIVE, RESULT):
            return None
        return run.required

    @property
    def hides_direction(self) -> bool:
        """Whether this direction keeps the aim out of sight until the effort starts."""
        return self.trial_direction in HIDDEN_DIRECTIONS

    def direction_known(self) -> bool:
        """Whether the person can now see the aim. A hidden-direction trial shows it only once the effort
        starts, so the aim stays out of the cue and the baseline."""
        if not self.hides_direction:
            return True
        return self.trials.phase in (ACTIVE, RESULT)

    def trial_level(self, shift: float | None) -> float | None:
        """A shift from the baseline as a level: the middle at the baseline, the top at `full_at` toward
        the cue. Oriented here, because every mechanism reads one number and none knows the direction."""
        if shift is None or self.trials.required is None:
            return None
        oriented = shift if self.trials.required == POSITIVE else -shift
        return min(1.0, max(0.0, 0.5 + oriented / (2 * max(self.UNITS.min_full_at, self.full_at))))

    def circle_level(self, shift: float | None) -> float | None:
        """The feedback circle's size, **mirroring the signal, not the goal**: bigger than the baseline
        when the signal is higher, smaller when it is lower, whichever direction the trial asks for. The
        colour, not the size, says whether that was the way the cue asked."""
        if shift is None:
            return None
        return min(1.0, max(0.0, 0.5 + shift / (2 * max(self.UNITS.min_full_at, self.full_at))))

    def no_level_note(self) -> tuple[str, str] | None:
        """Why there is no level between efforts, in active mode."""
        if self.mode == "active" and self._signal is None:
            return ("no level outside the effort window",
                    f"the level runs for the {self.TIMING.active_s:g} s of effort in each trial, and not in "
                    "a trial without feedback")
        return None

    # --- the readings -----------------------------------------------------------------------

    def due_readings(self, readings) -> list:
        """The waiting readings, thinned to one per `1 / OBSERVE_HZ` of source time. The trial engine
        judges a gap by the source's time between readings, so it must see every stretch of readings the
        source sent, however late the frame that hands them over."""
        step = 1.0 / OBSERVE_HZ
        due = []
        while readings:
            state = readings.popleft()
            # A step back in time is a new source or a clock that wrapped: taken, for the trial engine to
            # treat as the gap it is, and the grid starts again from it.
            back = self._last_t is not None and state.t < self._last_t
            if not back and self._due_t is not None and state.t < self._due_t:
                continue
            # On the grid while the readings keep up with it; from this reading after a gap.
            on_grid = not back and self._due_t is not None and state.t - self._due_t < step
            self._due_t = (self._due_t if on_grid else state.t) + step
            self._last_t = state.t
            due.append(state)
        return due

    def _observe_trial(self, t: float, value: float | None, valid: bool) -> None:
        """One reading into the trial in flight: its events, the heart beside it, and the level."""
        run = self.trials
        if run.done:
            return
        if not run.started and self.trial_hold() is not None:
            return
        kept = valid and value is not None
        for event in run.feed(t, value if kept else 0.0, valid=kept):
            self._trial_sound(event)
            if event.kind == CUE:
                self._trial_cued()
                if self.COUNTS_IN_BASELINE:
                    self.count_from = random.randint(300, 999)
        self.heart_trials.feed(run.phase, run.trial_number, run.required, self._heart_rate())
        if run.phase != self._last_phase:
            if run.phase == ACTIVE:
                self._start_block(t)
                # A hidden-direction trial names its aim here, at the effort start, not at the cue.
                if self.hides_direction and run.required is not None:
                    self.cue(cues.tone_at(CUE_PITCH[run.required]), CUE_VOLUME)
            self._last_phase = run.phase
        effort = 0.0
        if run.phase == ACTIVE and run.required is not None:
            effort = 1.0 if run.required == POSITIVE else -1.0
        self._steer(effort)
        self._update_shown(run, t)
        self.feedback_now = run.feedback_live and self._feedback_unfaded(run, t)
        driving = self.feedback_now and self.shown_shift is not None
        if driving:
            # Stepwise feedback is already quantised, so the continuous smoother must not run on top of it,
            # or the sound glides between the steps the circle jumps across.
            self._move_level(self.trial_level(self.shown_shift), t, smoothed=self.update_s <= 0)
        elif self._driving:
            self._go_quiet()
        self._driving = driving
        if run.done and not self.finished:
            self.finished = True
            self._accumulated = self.elapsed
            self._resumed_at = None
            self.paused = True
            self._steer(0.0)
            self._go_quiet()

    def _trial_cued(self) -> None:
        """A trial has just been cued. Nothing, by default."""

    def trial_hold(self) -> str | None:
        """Why the run cannot start yet, in words for the screen, or None when it can. The run waits for
        it before its first reading, so a trial is never cued that the program cannot judge. Never, by
        default."""
        return None

    def _start_block(self, t: float) -> None:
        """At the start of an effort window, begin a fresh block. The circle sits at the baseline until the
        first stepwise update, or follows the signal at once when the update is live."""
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

    def _trial_sound(self, event) -> None:
        """The cue tone at the start of a trial, and the same tone twice on a success. A trial that did
        not succeed is silent, as on the slow-potential screen."""
        if event.kind == CUE:
            pitch = CUE_PITCH_NEUTRAL if self.hides_direction else CUE_PITCH[event.required]
            self.cue(cues.tone_at(pitch), CUE_VOLUME)
        elif event.kind == VERDICT and event.trial is not None and event.trial.success:
            self.cue(cues.held(cues.tone_at(CUE_PITCH[event.required])), CUE_VOLUME)


# --- the view ---------------------------------------------------------------------------------------


def _direction_arrows(c: float, r: float, inward: bool) -> list:
    """Four faint arrows at a fixed ring, pointing in to ask for a smaller circle or out for a larger one.
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
    circle that grew in a constant line would light more of the screen as it grew, and a pupil would
    answer the light. `arrows` draws the faint direction hint: "shrink" or "expand".
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


def _direction_lines(pool: list, prefix: str, words: dict[str, str], unit: str,
                     decimals: int = 2) -> list[str]:
    """One line per direction that has trials: the success count and the signed mean shift."""
    out = []
    for required in (POSITIVE, NEGATIVE):
        trials = [r for r in pool if r.required == required]
        if trials:
            mean = sum(r.score_uv for r in trials) / len(trials)
            out.append(f"{prefix}{words[required]}: {sum(r.success for r in trials)} of {len(trials)} "
                       f"· mean {mean:+.{decimals}f} {unit}")
    return out


def trial_summary(run: TrialRun, words: dict[str, str], unit: str, decimals: int = 2) -> str:
    """The TRIALS panel text: per direction, how many trials succeeded and the mean shift from the
    baseline. When the run has no-feedback trials, they are counted on their own, because a shift held
    without the circle is the real test of control. The signed mean says more than the count: a pool that
    leans the asked-for way shows voluntary control building, before many trials cross the success margin."""
    if not run.results:
        return "no trials yet"
    valid = run.valid_results
    if any(r.transfer for r in run.results):
        lines = (_direction_lines([r for r in valid if not r.transfer], "feedback · ", words, unit, decimals)
                 + _direction_lines([r for r in valid if r.transfer], "no feedback · ", words, unit,
                                    decimals))
    else:
        lines = _direction_lines(valid, "", words, unit, decimals)
    void = sum(1 for r in run.results if not r.valid)
    if void:
        lines.append(f"{void} with no result")
    return "\n".join(lines) if lines else "no result yet"


class TrialView:
    """The trial controls a program puts on its screen: the fixation circle, the settings rows, the trial
    history and the heart stats. The program places them and calls the `draw_` methods each frame."""

    def __init__(self, session: TrialTraining, circle_side: int, heart: bool):
        self.session = session
        self.circle_side = circle_side
        self.circle = cv.Canvas(shapes=[], width=circle_side, height=circle_side)
        self.direction_text = ft.Text("", size=t.VALUE, color=t.TEXT, weight=ft.FontWeight.W_300)
        self.phase_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.count_text = ft.Text("", size=t.BODY, color=t.MUTED)
        self.verdict_text = ft.Text("", size=t.SMALL, color=t.MUTED)
        self.history_cells = ft.Row(spacing=3, wrap=True)
        self.history_text = ft.Text("", size=t.SMALL, color=t.MUTED)
        self.history_panel = panel("TRIALS", self.history_cells, ft.Container(height=8), self.history_text)
        self.heart = heart
        words = session.DIRECTION_WORD
        up, down = words[POSITIVE], words[NEGATIVE]
        self._up_box, self._up_val, _, self._up_sub = stat_tile(up, 158, 20, "bpm from baseline", sub=True)
        self._down_box, self._down_val, _, self._down_sub = stat_tile(down, 158, 20, "bpm from baseline",
                                                                      sub=True)
        self._run_box, self._run_val, self._run_unit, _ = stat_tile("run", 108, 17, "bpm")
        self._effort_box, self._effort_val, _, _ = stat_tile("effort", 108, 17, "bpm")
        self._rest_box, self._rest_val, _, _ = stat_tile("rest", 108, 17, "bpm")
        self._heart_no_data = ft.Text("no heart readings yet", size=t.SMALL, color=t.MUTED)
        self._heart_body = ft.Column([
            ft.Row([self._up_box, self._down_box], spacing=10),
            ft.Container(height=6),
            ft.Text("the mean change over each direction's trials", size=t.SMALL, color=t.FAINT),
            ft.Container(height=10), ft.Container(height=1, bgcolor=t.BORDER), ft.Container(height=10),
            ft.Row([self._run_box, self._effort_box, self._rest_box], spacing=8),
        ], spacing=0, visible=False)
        self.heart_stats_panel = panel("HEART STATS", self._heart_no_data, self._heart_body)

    def fixation_panel(self) -> ft.Control:
        side = self.circle_side
        return panel("FIXATION",
                     ft.Row([ft.Container(content=self.circle, width=side, height=side)],
                            alignment=ft.MainAxisAlignment.CENTER),
                     ft.Container(height=6),
                     ft.Row([self.direction_text], alignment=ft.MainAxisAlignment.CENTER),
                     ft.Row([self.count_text], alignment=ft.MainAxisAlignment.CENTER),
                     ft.Container(height=4),
                     ft.Row([self.phase_text], alignment=ft.MainAxisAlignment.CENTER),
                     ft.Row([self.verdict_text], alignment=ft.MainAxisAlignment.CENTER))

    def settings_rows(self, changed: Callable[[Callable], Callable]) -> list[ft.Control]:
        """The active mode's rows for the MODE panel. `changed` wraps a setter so the screen rebuilds."""
        session = self.session
        timing, units, words = session.TIMING, session.UNITS, session.DIRECTION_WORD
        active_s = timing.active_s

        def set_trials(n: float) -> None:
            session.n_trials = int(n)
            session.restart_trials()

        def set_transfer(percent: float) -> None:
            session.transfer_fraction = max(0.0, min(1.0, percent / 100.0))
            session.restart_trials()

        def set_margin(value: float) -> None:
            session.success_at = max(0.0, float(value))
            session.restart_trials()

        def set_feedback_window(s: float) -> None:
            session.feedback_window_s = max(1.0, min(active_s, float(s)))

        def set_update(s: float) -> None:
            session.update_s = max(0.0, min(active_s, float(s)))

        def audition(required: str, mark: bool = False) -> None:
            sound = cues.tone_at(CUE_PITCH[required])
            session.cue(cues.held(sound) if mark else sound, CUE_VOLUME)

        return [
            labelled("direction",
                     ft.Row([pill(d, session.trial_direction == d,
                                  changed(lambda _e, d_=d: session.set_trial_direction(d_)))
                             for d in session.DIRECTIONS], spacing=6, wrap=True, expand=True),
                     width=96),
            labelled("trials", num_box(session.n_trials, changed(set_trials), step=5, low=1, high=200,
                                       width=52),
                     note=f"about {_duration(session.n_trials * timing.trial_s)}", width=96),
            labelled("without feedback",
                     num_box(round(session.transfer_fraction * 100), changed(set_transfer),
                             suffix="%", step=10, low=0, high=100, width=52),
                     note="of the trials", width=96),
            labelled("show circle for",
                     num_box(session.feedback_window_s, changed(set_feedback_window),
                             suffix="s", step=1, low=1, high=active_s, width=58),
                     note=("the whole effort" if session.feedback_window_s >= active_s
                           else f"then no circle for {active_s - session.feedback_window_s:g} s"),
                     width=96),
            labelled("update every",
                     num_box(session.update_s, changed(set_update), suffix="s",
                             step=0.5, low=0, high=active_s, width=58),
                     note=("0 = live" if session.update_s <= 0 else f"steps of {session.update_s:g} s"),
                     width=96),
            labelled("succeeds at", num_box(session.success_at, changed(set_margin), suffix=units.unit,
                                            step=units.step, low=0, high=units.max_success, width=58),
                     note="past the baseline", width=96),
            labelled("full at", num_box(session.full_at, changed(session.set_full_at), suffix=units.unit,
                                        step=units.step, low=units.min_full_at, high=units.max_full_at,
                                        width=58),
                     note="from the baseline", width=96),
            ft.Row([button("restart run", changed(lambda _e: session.restart_trials()))]),
            ft.Row([test_button(words[POSITIVE], lambda _e: audition(POSITIVE)),
                    test_button(words[NEGATIVE], lambda _e: audition(NEGATIVE)),
                    test_button("success", lambda _e: audition(POSITIVE, mark=True))],
                   spacing=6),
        ]

    def draw_fixation(self, passive_line: str) -> None:
        """The circle and the lines under it. `passive_line` is what passive mode says under the circle."""
        session, run = self.session, self.session.trials
        words, unit = session.DIRECTION_WORD, session.UNITS.unit
        self.verdict_text.value = ""
        if session.mode == "passive":
            self.direction_text.value, self.direction_text.color = "passive", t.MUTED
            self.count_text.value = ""
            self.phase_text.value = passive_line
            self.circle.shapes = circle_shapes(self.circle_side, session.signal, t.ACCENT)
            return
        required = session.required
        if session.state is None:
            self.direction_text.value, self.direction_text.color = "waiting", t.FAINT
            self.count_text.value = ""
            self.phase_text.value = session.trial_waiting
            self.circle.shapes = circle_shapes(self.circle_side, None, t.ACCENT)
            return
        hold = None if run.started else session.trial_hold()
        if hold is not None:
            self.direction_text.value, self.direction_text.color = "waiting", t.FAINT
            self.count_text.value = ""
            self.phase_text.value = hold
            self.circle.shapes = circle_shapes(self.circle_side, None, t.ACCENT)
            return
        if run.done:
            self.direction_text.value, self.direction_text.color = f"done · {len(run.results)} trials", t.MUTED
        elif required is None:
            self.direction_text.value, self.direction_text.color = "rest", t.FAINT
        elif not session.direction_known():
            self.direction_text.value, self.direction_text.color = "?", t.FAINT
        else:
            self.direction_text.value, self.direction_text.color = words[required], t.TEXT
        self.count_text.value = (f"count back by fours from {session.count_from}"
                                 if run.phase in (CUED, BASELINE) and session.count_from else "")
        name = {CUED: "cue", BASELINE: "baseline", ACTIVE: "effort", RESULT: "result",
                ITI: "rest"}.get(run.phase, run.phase)
        phase = (f"trial {run.trial_number} of {run.n_trials} · {name} · {run.phase_remaining():.0f} s"
                 if not run.done else "the run is over · restart it in the mode panel")
        if not run.done and required is not None and not session.direction_known():
            phase += " · wait for the tone"
        level, colour = None, t.ACCENT
        if run.phase == ACTIVE and session.feedback_now:
            level = session.circle_level(session.shown_shift)
            goal = "inside the ring" if required == NEGATIVE else "outside the ring"
            phase += f" · pull the circle {goal}"
        elif run.phase == ACTIVE and run.transfer:
            phase += " · no feedback in this trial"
        elif run.phase == ACTIVE:
            phase += " · hold it · the circle is off"
        self.phase_text.value = phase
        if run.phase == RESULT and run.last is not None:
            last = run.last
            asked = words[last.required]
            if not last.valid:
                level, colour = None, t.MUTED
                self.direction_text.value, self.direction_text.color = "no result", t.MUTED
                self.verdict_text.value = "too many readings were rejected in this trial"
            else:
                level = session.circle_level(last.score_uv)
                colour = t.GOOD if last.success else t.BAD
                self.direction_text.value = f"{'✓' if last.success else '✗'}  {asked}"
                self.direction_text.color = colour
                if last.score_uv == 0:
                    self.verdict_text.value = f"asked for {asked} · {session.trial_subject} did not change"
                else:
                    went = words[POSITIVE] if last.score_uv > 0 else words[NEGATIVE]
                    self.verdict_text.value = (f"asked for {asked} · {session.trial_subject} went "
                                               f"{abs(last.score_uv):.2f} {unit} {went}")
        arrows = None
        if (run.phase in (CUED, BASELINE, ACTIVE) and not run.transfer and required is not None
                and session.direction_known()
                and not (run.phase == ACTIVE and not session.feedback_now)):
            arrows = "shrink" if required == NEGATIVE else "expand"
        self.circle.shapes = circle_shapes(self.circle_side, level, colour, arrows=arrows)

    def draw_history(self) -> None:
        session = self.session
        self.history_panel.visible = session.mode == "active"
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
        self.history_cells.controls = cells
        self.history_text.value = trial_summary(run, session.DIRECTION_WORD, session.UNITS.unit)

    def draw_heart_stats(self) -> None:
        stats = self.session.heart_trials
        self._heart_no_data.visible = not stats.has_data
        self._heart_body.visible = stats.has_data
        if not stats.has_data:
            return
        words = self.session.DIRECTION_WORD
        by = {word: (change, n) for word, change, n in stats.direction_summary()}
        for word, value, sub in ((words[POSITIVE], self._up_val, self._up_sub),
                                 (words[NEGATIVE], self._down_val, self._down_sub)):
            if word in by:
                change, n = by[word]
                value.value = f"{change:+.1f}"
                sub.value = f"{n} trial" if n == 1 else f"{n} trials"
            else:
                value.value, sub.value = "—", "no trials yet"
        mean, lo, hi = stats.overall()
        self._run_val.value = f"{mean:.0f}" if mean is not None else "—"
        self._run_unit.value = f"bpm · {lo:.0f}-{hi:.0f}" if lo is not None else "bpm"
        effort, rest = stats.effort_vs_rest()
        self._effort_val.value = f"{effort:.0f}" if effort is not None else "—"
        self._rest_val.value = f"{rest:.0f}" if rest is not None else "—"
