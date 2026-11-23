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

import dataclasses
import math
import random
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Callable, NamedTuple

import flet as ft
import flet.canvas as cv
from scipy import stats

from . import cues
from . import theme as t
from .trials import (ACTIVE, BASELINE, CUE, CUED, EXTRA, GAP, ITI, JUMP_S, LIMIT, LOST,
                     MIN_BASELINE_FRACTION, NEGATIVE, NO_SIGNAL,
                     PAUSED, POSITIVE, REJECTED, RESULT, SEAM, THIN_BASELINE, VERDICT, Trial, TrialRun,
                     succeeds)
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

ARROW_MARGIN = 36.0
"""The band outside the largest circle that holds the arrows: room for an arrow
pointing in or out, clear of the circle at any size."""

MAX_CUE_GAP_S = 5.0
"""Longest wait from the cue to the baseline that the operator can set."""

MIN_FEEDBACK_S = 2.0
"""The shortest feedback window **trial length** leaves: shorter, its mean rests on too few readings."""


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


CONDITION_AT = 0.5
"""Where each trial program's condition target is met. Inside a feedback window the target reads
`TrialTraining.condition_level`, which reaches this value at **succeeds at**."""

AUTOMATIC, FIXED = "automatic", "fixed"
"""How **full at** and **succeeds at** get their value: from the person's chance spread, or as typed."""

FULL_AT_SPREADS = 2.0
SUCCESS_SPREADS = 1.0
"""On *automatic*, **full at** is two chance spreads and **succeeds at** one: the user's decision. Floored at
the program's `min_full_at` and half of it, the resolution the level has; not clamped above, because the
box's range bounds a typed value and an automatic one follows the person's signal."""


class ScaleSetting(NamedTuple):
    """A mode and its typed value, replaced as one object, so the feed thread never reads half a change."""

    mode: str
    value: float


class TrialScale(NamedTuple):
    """The record a trial is shown and scored against, written once at its feedback start. `spread` is the
    chance spread of one trial it came from, None when there was none, and `seconds` the signal behind it."""

    full_at: float
    limit: float
    full_mode: str
    limit_mode: str
    spread: float | None
    seconds: float


LIMIT_LINE = 1.5
"""The ring at **succeeds at** is this wide at the middle size, the same as the ring of the mean."""

DIRECTION_LINE_H = 34
"""The height reserved for the direction line over the circle, whatever glyph it shows."""

VERDICT_LINE_H = 16
"""The height reserved for each of the two lines under the circle, so a verdict never moves the panel."""

VOID_WORDS = {
    LIMIT: ("no result · the signal reaches the input limit in this trial",
            "examine the electrode and its contact"),
    NO_SIGNAL: ("no result · the channel has no signal in this trial",
                "examine the electrode and its contact"),
    EXTRA: ("no result · an extra channel has no valid signal for the subtraction in this trial",
            "examine the electrodes of the extra channels"),
    LOST: ("no result · {share} of the readings do not arrive{limit}",
           "examine the connection to the device"),
    GAP: ("no result · the readings stop for more than {jump} in this trial",
          "keep the device connected through a trial"),
    PAUSED: ("no result · the session pauses in this trial",
             "keep the session running through a trial"),
    SEAM: ("no result · the looped file starts again in this trial",
           "no action is necessary"),
    THIN_BASELINE: ("no result · fewer than {baseline} of the baseline readings are valid",
                    "keep still in the baseline"),
    REJECTED: ("no result · the program rejects {share} of the readings{limit}",
               "keep still in the trial"),
}
"""The two lines under the circle for a void trial, by cause: what happened, then what to do. A program
changes the second line through `VOID_ACTIONS`. `{share}` is the share of the named cause alone, and
`{limit}` compares it with the 20% rule only where that share alone is past it."""


FULL_AT_PART = 0.6
"""How far from the baseline toward its limit the feedback picture is at `full_at`. The rest of the way is
for the shifts past it, which real sessions produce: a slow potential moves 20 µV against a 10 µV default."""


def picture_position(shift: float, full_at: float) -> float:
    """A shift as a signed position for the feedback picture, between -1 and 1 and never at either.

    In proportion up to `full_at`, which lands at `FULL_AT_PART`; past it, `over / (over + k)` toward the
    limit, with `k` chosen so the slope does not change at the knee. A clipped picture drew 15 and 30 µV
    the same at a 10 µV `full_at`; here they sit at 77% and 90%. The picture only: `trial_level`, which
    the sound, the shadow, the click and the music read, keeps its clip.
    """
    u = abs(shift) / full_at
    if u <= 1.0:
        part = FULL_AT_PART * u
    else:
        over, k = u - 1.0, (1.0 - FULL_AT_PART) / FULL_AT_PART
        part = FULL_AT_PART + (1.0 - FULL_AT_PART) * over / (over + k)
    return math.copysign(part, shift)


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


TRIAL_SETTINGS = ("mode", "trial_direction", "n_trials", "transfer_fraction", "success_at_setting",
                  "full_at_setting", "active_s", "feedback_window_s", "update_s", "cue_s")
"""What `adopt` carries from a stopped session to the next: the operator's settings, not the results. For
**full at** and **succeeds at** that is the mode with the typed value, never an automatic value, which is a
measurement of the old session."""


class TrialTraining:
    """The trial half of a session. The session calls `_init_trials` once and `_observe_trial` with each
    reading in active mode, and supplies:

    - `TIMING`, `UNITS`, `DIRECTION_WORD` and `COUNTS_IN_BASELINE` (whether the baseline asks the person to
      count backwards, a task that keeps the mind busy and level);
    - `trial_subject` (what moved, in words: "your pupil") and `trial_waiting` (the line shown before the
      first reading);
    - `_move_level(signal, t, smoothed)`, `_go_quiet()`, `cue` and `elapsed`;
    - `heart` and `heart_choice`, optionally, for the heart beside the trials;
    - `_steer(effort)`, optionally, to tell a simulated source what the person is trying to do;
    - `SCALE_MODES` and `chance_spread()`, optionally, for an automatic **full at** and **succeeds at**;
    - `VOID_ACTIONS` and `void_words(trial)`, optionally, for what a void trial asks the person to do.
    """

    TIMING: TrialTiming
    UNITS: TrialUnits
    DIRECTION_WORD: dict[str, str]
    COUNTS_IN_BASELINE = False
    DIRECTIONS = DIRECTIONS
    """The direction choices this program offers, by name. A program can add its own."""
    DEFAULT_DIRECTION = "alternating"
    PHASE_WORD = {CUED: "cue", BASELINE: "baseline", ACTIVE: "effort", RESULT: "result", ITI: "rest"}
    """What the line under the circle calls each phase, in the words of the program's own help."""
    RESTART_PLACE = "the mode panel"
    """Where the restart button is, for the line that says the run is over."""
    SCALE_MODES: tuple[str, ...] = (FIXED,)
    """The modes this program offers for **full at** and **succeeds at**, the first being the default. A
    program that measures a chance spread (`chance_spread`) can offer AUTOMATIC."""
    VOID_ACTIONS: dict[str, str] = {}
    """The second line under a void trial, by cause, where this program asks for something other than the
    shared `VOID_WORDS`."""
    MAX_TRIAL_S: float | None = None
    """The longest **trial length** this program offers, in seconds. None: twice the published length, a
    bound that keeps the box from running away, where the program has no physical one."""

    @property
    def trial_metric_hz(self) -> float:
        """How many readings a second reach the trial engine, for its check that a baseline has enough."""
        return OBSERVE_HZ

    def _init_trials(self, mode: str) -> None:
        self.mode = mode
        self.trial_direction = self.DEFAULT_DIRECTION
        self.n_trials = DEFAULT_TRIALS
        self.transfer_fraction = 0.0
        self.success_at_setting = ScaleSetting(self.SCALE_MODES[0], 0.0)
        self.full_at_setting = ScaleSetting(self.SCALE_MODES[0], self.UNITS.full_at)
        self.trial_scale: TrialScale | None = None
        """The record of the trial in flight, from its feedback start. Only `_freeze_scale` writes it."""
        self.trial_cut: str | None = None
        """Why the trial in flight ends at the next reading, or None: see `cut_trial`."""
        self.trial_condition: float | None = None
        """What the condition's target reads while a feedback window drives the outputs, else None. Only
        `_observe_trial` writes it; see `condition_level`."""
        self.active_s = self.TIMING.active_s
        """The feedback window, the trial length less the baseline. Everything that depends on the length of
        the feedback reads it here, never from `TIMING`."""
        self.feedback_window_s = self.TIMING.active_s
        self.update_s = 0.0
        self.cue_s = self.TIMING.cue_s
        """Time from the cue to the start of the baseline. A program that lets the operator set it shows a
        row for it; the others keep their `TIMING`."""
        self.heart_trials = HeartTrials(self.DIRECTION_WORD)
        self.trials = self._new_run()
        self.count_from: int | None = None
        self.shown_shift: float | None = None
        self.shown_mean: float | None = None
        self.feedback_now = False
        self._driving = False
        self._last_phase: str | None = None
        self._block_at: float | None = None
        self._block_vals: list[float] = []
        self._last_t: float | None = None
        self._due_t: float | None = None       # the source time from which the next reading is taken
        self._raw_t: float | None = None       # the newest reading's own time, before any re-stamp
        self._emitted_t: float | None = None   # the newest time handed on, after any re-stamp
        self._spacings: deque[float] = deque(maxlen=31)
        self._late: list | None = None         # a reading after a hole, and the batch behind it

    def _steer(self, effort: float) -> None:
        """Tell a simulated source what the person is trying to do. Nothing, by default."""

    heart = None
    """The app's heart-rate runner, borrowed and never owned, when the program was given one."""
    heart_choice = None

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

    # --- what the operator sets -------------------------------------------------------------

    def _new_run(self) -> TrialRun:
        timing = self.TIMING
        return TrialRun(n_trials=self.n_trials, baseline_s=timing.baseline_s, active_s=self.active_s,
                        result_s=timing.result_s, iti_s=timing.iti_s, cue_s=self.cue_s,
                        transfer_fraction=self.transfer_fraction,
                        directions=self.DIRECTIONS[self.trial_direction],
                        alternate=self.trial_direction == "alternating",
                        metric_hz=self.trial_metric_hz)

    def restart_trials(self) -> None:
        """A new run with the current settings. The results go: they were scored under the old ones."""
        self.trials = self._new_run()
        self.finished = False
        self.count_from = None
        self._last_phase = None
        self.trial_scale = None
        self.trial_condition = None
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

    @property
    def trial_seconds(self) -> float:
        """One trial with its rest, on average, as the run is set now."""
        return self.TIMING.trial_s - self.TIMING.cue_s + self.cue_s - self.TIMING.active_s + self.active_s

    @property
    def trial_length(self) -> float:
        """The baseline and the feedback window, in seconds: what **trial length** sets."""
        return self.TIMING.baseline_s + self.active_s

    @property
    def trial_length_range(self) -> tuple[float, float]:
        """The shortest length, the baseline with 2 s of feedback, and the longest this program offers."""
        published = self.TIMING.baseline_s + self.TIMING.active_s
        return self.TIMING.baseline_s + MIN_FEEDBACK_S, self.MAX_TRIAL_S or 2 * published

    def set_trial_length(self, seconds: float) -> None:
        """A new trial length starts the run again. A circle shown for the whole window stays so, and
        a shorter one, or a step, never outlasts the new window."""
        low, high = self.trial_length_range
        whole = self.feedback_window_s >= self.active_s
        self.active_s = max(low, min(high, float(seconds))) - self.TIMING.baseline_s
        self.feedback_window_s = self.active_s if whole else min(self.feedback_window_s, self.active_s)
        self.update_s = min(self.update_s, self.active_s)
        self._trial_length_changed()
        self.restart_trials()

    def _trial_length_changed(self) -> None:
        """The feedback window has a new length. Nothing more, by default."""

    def set_cue_gap(self, seconds: float) -> None:
        self.cue_s = max(0.0, min(MAX_CUE_GAP_S, float(seconds)))
        self.restart_trials()

    # --- full at and succeeds at -------------------------------------------------------------

    @property
    def full_at_mode(self) -> str:
        return self.full_at_setting.mode

    @property
    def success_at_mode(self) -> str:
        return self.success_at_setting.mode

    @property
    def full_at(self) -> float:
        """The **full at** the next trial takes: two chance spreads on *automatic*, the typed value on
        *fixed*. A trial in flight reads its own record instead (`scale_in_use`)."""
        return self._live_scale(self.full_at_setting, FULL_AT_SPREADS, self.UNITS.min_full_at)[0]

    @property
    def success_at(self) -> float:
        """The **succeeds at** the next trial takes: one chance spread on *automatic*, else as typed."""
        return self._live_scale(self.success_at_setting, SUCCESS_SPREADS, self.UNITS.min_full_at / 2)[0]

    def _live_scale(self, setting: ScaleSetting, spreads: float, floor: float,
                    measured: tuple[float, float] | None = None) -> tuple[float, float | None, float]:
        """(value, spread, seconds) for one setting. On *automatic* with no spread, the floor."""
        if setting.mode != AUTOMATIC:
            return setting.value, None, 0.0
        if measured is None:
            measured = self.chance_spread()
        if measured is None:
            return floor, None, 0.0
        spread, seconds = measured
        return max(floor, spreads * spread), spread, seconds

    def chance_spread(self) -> tuple[float, float] | None:
        """The chance spread of one trial's result, and the seconds of signal behind it, or None when there
        is none. A program that measures one overrides this. None, by default."""
        return None

    def set_full_at(self, value: float, typed: bool = False) -> None:
        """A **full at** for the next trial. A typed value sets *fixed*; otherwise the mode stays."""
        mode = FIXED if typed else self.full_at_setting.mode
        self.full_at_setting = ScaleSetting(mode, max(self.UNITS.min_full_at, float(value)))

    def set_success_at(self, value: float, typed: bool = False) -> None:
        """A **succeeds at** for the next trial. A typed value sets *fixed*; otherwise the mode stays."""
        mode = FIXED if typed else self.success_at_setting.mode
        self.success_at_setting = ScaleSetting(mode, max(0.0, float(value)))

    def set_full_at_mode(self, mode: str) -> None:
        if mode in self.SCALE_MODES:
            self.full_at_setting = ScaleSetting(mode, self.full_at_setting.value)

    def set_success_at_mode(self, mode: str) -> None:
        if mode in self.SCALE_MODES:
            self.success_at_setting = ScaleSetting(mode, self.success_at_setting.value)

    def _freeze_scale(self) -> None:
        """At a feedback start, the one record the trial is shown and scored against. Each setting and the
        spread are read once, so the record is one moment's values; it is written as one object."""
        run, units = self.trials, self.UNITS
        full, limit = self.full_at_setting, self.success_at_setting
        measured = self.chance_spread() if AUTOMATIC in (full.mode, limit.mode) else None
        full_at, spread, seconds = self._live_scale(full, FULL_AT_SPREADS, units.min_full_at, measured)
        limit_at, limit_spread, limit_seconds = self._live_scale(limit, SUCCESS_SPREADS,
                                                                 units.min_full_at / 2, measured)
        if spread is None:
            spread, seconds = limit_spread, limit_seconds
        record = TrialScale(full_at, limit_at, full.mode, limit.mode, spread, seconds)
        run.limit_uv = record.limit
        run.scale = record
        self.trial_scale = record

    def scale_in_use(self) -> tuple[float, float]:
        """(**full at**, **succeeds at**) as every reader takes them: the record of the trial from its
        feedback start to its verdict, and the live values outside a trial."""
        record, run = self.trial_scale, self.trials
        if record is not None and record is run.scale and run.phase in (ACTIVE, RESULT):
            return record.full_at, record.limit
        return self.full_at, self.success_at

    def adopt_trials(self, previous: "TrialTraining") -> None:
        for name in TRIAL_SETTINGS:
            setattr(self, name, getattr(previous, name))
        self._trial_length_changed()
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
        full_at = self.scale_in_use()[0]
        return min(1.0, max(0.0, 0.5 + oriented / (2 * max(self.UNITS.min_full_at, full_at))))

    def circle_level(self, shift: float | None) -> float | None:
        """The feedback circle's size, **mirroring the signal, not the goal**: bigger than the baseline
        when the signal is higher, smaller when it is lower, whichever direction the trial asks for. The
        colour, not the size, says whether that was the way the cue asked. **It never sticks**: past
        `full_at` it compresses (`picture_position`) where `trial_level` clips."""
        if shift is None:
            return None
        return 0.5 + 0.5 * picture_position(shift, max(self.UNITS.min_full_at, self.scale_in_use()[0]))

    def condition_level(self, shift: float) -> float:
        """The shift past **succeeds at**, on the level's scale and not clipped: at or above `CONDITION_AT`
        exactly where `succeeds` says the trial's rule is met. The click and the music read this, so a limit
        past **full at**, where the clipped level never reaches it, still says "good" past the limit."""
        full_at, limit = self.scale_in_use()
        required = self.trials.required
        oriented = shift if required == POSITIVE else -shift
        value = CONDITION_AT + (oriented - limit) / (2 * max(self.UNITS.min_full_at, full_at))
        return value if succeeds(shift, required, limit) else min(value, math.nextafter(CONDITION_AT, 0.0))

    def condition_for(self, level: float) -> float:
        """What the condition's target reads beside this level: the trial's condition while a feedback
        window drives the outputs, the level itself everywhere else."""
        return level if self.trial_condition is None else self.trial_condition

    def no_level_note(self) -> tuple[str, str] | None:
        """Why there is no level between efforts, in active mode."""
        if self.mode == "active" and self._signal is None:
            return ("no level outside the effort window",
                    f"the level runs for the {self.active_s:g} s of effort in each trial, and not in "
                    "a trial without feedback")
        return None

    def void_words(self, trial: Trial) -> tuple[str, str]:
        """The two lines under a void trial: its cause, then what to do, with `VOID_ACTIONS` in place of
        the shared action where this program has its own."""
        cause = trial.cause if trial.cause in VOID_WORDS else REJECTED
        what, action = VOID_WORDS[cause]
        limit = self.trials.reject_limit
        what = what.format(share=f"{trial.cause_share:.0%}",
                           limit=f" · the maximum is {limit:.0%}" if trial.cause_share > limit else "",
                           jump=f"{JUMP_S:g} s", baseline=f"{MIN_BASELINE_FRACTION:.0%}")
        return what, self.VOID_ACTIONS.get(cause, action)

    # --- the readings -----------------------------------------------------------------------

    def _on_time(self, readings) -> list:
        """The waiting readings, with a late batch put back in the hole it was made in.

        A source that stamps a frame at its arrival hands a stall's frames over together after it, each
        stamped well within one period of the stream after the one before. Those frames were made in the
        stall, so they go back over it one period apart, from the newest, and the trial does not count
        them as lost. The period is the median spacing of the stream itself. A reading after a hole waits
        for the next one to show whether a batch follows. Readings stamped by a device clock never arrive
        so, and pass unchanged."""
        out = []
        while readings:
            state = readings.popleft()
            prev, self._raw_t = self._raw_t, state.t
            gap = None if prev is None else state.t - prev
            period = statistics.median(self._spacings) if len(self._spacings) >= 5 else None
            if self._late is not None:
                if gap is not None and period is not None and 0 <= gap < period / 2:
                    self._late.append(state)
                    continue
                out += self._restamp(self._late, period)
                self._late = None
            if gap is not None and gap > 0:
                self._spacings.append(gap)
            if period is not None and gap is not None and 2 * period < gap < JUMP_S:
                self._late = [state]
                continue
            out.append(state)
            self._emitted_t = state.t
        return out

    def _restamp(self, batch: list, period: float | None) -> list:
        """A batch after a hole, one period apart back from its newest reading, and never at or before
        the reading handed on before it. A batch of one is a reading after a real hole, kept as it is."""
        newest = batch[-1].t
        if len(batch) > 1 and period is not None:
            lower = self._emitted_t if self._emitted_t is not None else newest - len(batch) * period
            spacing = min(period, (newest - lower) / len(batch))
            batch = [dataclasses.replace(s, t=newest - (len(batch) - 1 - k) * spacing)
                     for k, s in enumerate(batch)]
        self._emitted_t = newest
        return batch

    def due_readings(self, readings) -> list:
        """The waiting readings, thinned to one per `1 / OBSERVE_HZ` of source time. The trial engine
        judges a gap by the source's time between readings, so it must see every stretch of readings the
        source sent, however late the frame that hands them over (`_on_time`)."""
        step = 1.0 / OBSERVE_HZ
        due = []
        for state in self._on_time(readings):
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

    def _observe_trial(self, t: float, value: float | None, valid: bool, cause: str | None = None,
                       cut: str | None = None) -> None:
        """One reading into the trial in flight: its events, the heart beside it, and the level. `cause`
        and `cut` are as `TrialRun.feed` takes them."""
        run = self.trials
        cut, self.trial_cut = cut or self.trial_cut, None
        if run.done:
            return
        if not run.started and self.trial_hold() is not None:
            return
        kept = valid and value is not None
        events = run.feed(t, value if kept else 0.0, valid=kept, cause=cause, cut=cut)
        # The record is frozen as the feedback window opens, before any reader in it, transfer trials too.
        if run.phase == ACTIVE and run.scale is None:
            self._freeze_scale()
        for event in events:
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
        self.trial_condition = self.condition_level(self.shown_shift) if driving else None
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

    def cut_trial(self, cause: str) -> None:
        """End the trial in flight at the next reading, void with `cause`, because the signal before that
        reading no longer precedes it: a pause of any length (PAUSED), or a looped file's seam (SEAM)."""
        self.trial_cut = self.trial_cut or cause

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
        self.shown_mean = None

    def _update_shown(self, run, t: float) -> None:
        """The shift the circle shows, and the mean of the window so far. Live, they follow every reading.
        Stepwise, they hold and refresh every `update_s` seconds, so the circle moves in steps the person
        must bridge and the mean does not give away what the step withholds."""
        if run.phase != ACTIVE or run.transfer or run.shift is None:
            return
        if self.update_s <= 0:
            self.shown_shift = run.shift
            self.shown_mean = run.mean_shift
            return
        self._block_vals.append(run.shift)
        if self._block_at is None:
            self._block_at = t
        if t - self._block_at >= self.update_s:
            self.shown_shift = sum(self._block_vals) / len(self._block_vals)
            self.shown_mean = run.mean_shift
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


def _radii(side: float) -> tuple[float, float, float]:
    """The circle's smallest, largest and baseline radius on a canvas of this side."""
    r_min, r_max = 0.16 * side, 0.46 * side
    return r_min, r_max, (r_min + r_max) / 2


def circle_shapes(side: float, level: float | None, colour: str, baseline: bool = True,
                  arrows: str | None = None, mean: float | None = None, margin: float = 0.0,
                  limit: float | None = None) -> list:
    """The feedback circle on a fixation point, with a **constant number of lit pixels**.

    The line's width is inversely proportional to the radius, so the ink is the same at every size. A
    circle that grew in a constant line would light more of the screen as it grew, and a pupil would
    answer the light. `arrows` draws the faint direction hint: "shrink" or "expand". `mean` draws a thin
    faint ring at that level, for the mean of the window so far, in a line that keeps its ink in the same way.
    `margin` keeps a band of the canvas outside the largest circle, and the arrows go in it, so the circle
    never runs over them. `limit` draws a thin ring at that level, the size the circle must pass for
    the trial to succeed, in a line that keeps its ink as the ring of the mean does.
    """
    c = side / 2
    r_min, r_max, r_mid = _radii(side - 2 * margin)
    ink = 2 * math.pi * r_mid * 4.0                 # a 4 px line at the middle size
    out = [cv.Circle(c, c, 3.0, paint=ft.Paint(color=t.TEXT, style=ft.PaintingStyle.FILL))]
    if baseline:
        out.append(cv.Circle(c, c, r_mid, paint=ft.Paint(
            color=t.FAINT, stroke_width=1.0, style=ft.PaintingStyle.STROKE,
            stroke_dash_pattern=[5, 5])))
    r = r_min + min(1.0, max(0.0, level)) * (r_max - r_min) if level is not None else r_mid
    if arrows in ("shrink", "expand"):
        out += _direction_arrows(c, r_max + margin / 2 if margin else 0.42 * side, inward=arrows == "shrink")
    if mean is not None:
        mean_r = r_min + min(1.0, max(0.0, mean)) * (r_max - r_min)
        mean_ink = 2 * math.pi * r_mid * 1.5        # a 1.5 px line at the middle size
        out.append(cv.Circle(c, c, mean_r, paint=ft.Paint(
            color=ft.Colors.with_opacity(0.35, t.TEXT), stroke_width=mean_ink / (2 * math.pi * mean_r),
            style=ft.PaintingStyle.STROKE)))
    if limit is not None:
        r_lim = r_min + min(1.0, max(0.0, limit)) * (r_max - r_min)
        out.append(cv.Circle(c, c, r_lim, paint=ft.Paint(
            color=t.ACCENT, stroke_width=LIMIT_LINE * r_mid / r_lim, style=ft.PaintingStyle.STROKE)))
    if level is not None:
        out.append(cv.Circle(c, c, r, paint=ft.Paint(
            color=colour, stroke_width=ink / (2 * math.pi * r), style=ft.PaintingStyle.STROKE)))
    return out


def lead_in_shapes(side: float, part: float | None, margin: float = 0.0) -> list:
    """The cue and the baseline as their own picture: the fixation point, and the baseline ring drawing
    itself clockwise from the top, whole when the feedback starts. `part` is how much of it is drawn, and
    None draws the point alone, for the rest between trials. No circle and no arrows: nothing here moves
    with the signal, so nothing invites the task before its window."""
    c = side / 2
    r = _radii(side - 2 * margin)[2]
    out = [cv.Circle(c, c, 3.0, paint=ft.Paint(color=t.MUTED, style=ft.PaintingStyle.FILL))]
    if part is not None and part > 0:
        out.append(cv.Arc(c - r, c - r, 2 * r, 2 * r, -math.pi / 2, 2 * math.pi * min(1.0, part),
                          paint=ft.Paint(color=t.MUTED, stroke_width=2.0, style=ft.PaintingStyle.STROKE,
                                         stroke_dash_pattern=[6, 6])))
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


def verdict_line(last: Trial, session: TrialTraining) -> tuple[str, str]:
    """The two lines under a verdict, and the only place a trial becomes words there. A valid trial: what it
    asked for and what the signal did, then the comparison with **succeeds at** from the trial's record. A
    void trial: why it has no result, then what to do, in the program's words (`void_words`)."""
    if not last.valid:
        return session.void_words(last)
    words, units = session.DIRECTION_WORD, session.UNITS
    asked = words[last.required]
    if last.score_uv == 0:
        first = f"asked for {asked} · {session.trial_subject} did not change"
    else:
        went = words[POSITIVE] if last.score_uv > 0 else words[NEGATIVE]
        first = (f"asked for {asked} · {session.trial_subject} went "
                 f"{abs(last.score_uv):.{units.decimals}f} {units.unit} {went}")
    scale = last.scale
    if scale is None or (scale.limit_mode == FIXED and scale.limit <= 0):
        return first, ""
    limit = f"{scale.limit:.{units.decimals}f} {units.unit}"
    if scale.limit_mode == AUTOMATIC and scale.spread is None:
        return first, f"succeeds at {limit} · no noise value yet"
    side = "more" if last.success else "less"
    source = "fixed" if scale.limit_mode == FIXED else f"from {scale.seconds:.0f} s of signal"
    return first, f"{side} than the {limit} of succeeds at · {source}"


def run_verdict_help(words: dict[str, str]) -> str:
    """The help paragraph for the words `trial_summary` prints when a run is done, in a program's words."""
    return ("**The end of the run.** At the end of a run, the TRIALS panel compares the two directions "
            f"once. It shows *{words[POSITIVE]} minus {words[NEGATIVE]}*, the difference of their mean "
            "changes. Chance takes the difference past the *chance range* in the cued direction in 1 run of "
            "20. The next line tells how frequently chance gives the difference of this run. A run in one "
            "direction compares its mean change with zero. Past the chance range in the cued direction, the panel "
            "tells you to keep the strategy. Inside it, the run does not show control. Past it the wrong "
            "way, the strategy moves the signal the other way. The panel makes this comparison only at the "
            "end of the run, because a comparison after every trial gives chance many more ways to pass.")


RUN_ALPHA = 0.05
"""The chance level of the run's verdict, split over the groups it judges: chance alone passes the range
in the cued direction in one run of twenty."""

MIN_PER_DIRECTION = 2
"""Fewest scores per direction a group, or an extra channel, needs before it is compared."""


def _chance_test(up: list[float], down: list[float] | None, alpha: float) -> tuple[float, float, float]:
    """(difference, chance range, p) for the scores of one group: Welch's comparison of the two directions,
    or one direction against 0 when `down` is None. The chance range is the size chance passes in one
    direction `alpha` of the time, and p is how often chance gives a difference this large on this side
    of zero; a group with no scatter has a range of 0 and a p of 1, which judges nothing."""
    if down is None:
        diff, se, df = statistics.fmean(up), math.sqrt(statistics.variance(up) / len(up)), len(up) - 1
    else:
        vu, vd = statistics.variance(up) / len(up), statistics.variance(down) / len(down)
        diff, se = statistics.fmean(up) - statistics.fmean(down), math.sqrt(vu + vd)
        df = (vu + vd) ** 2 / (vu ** 2 / (len(up) - 1) + vd ** 2 / (len(down) - 1)) if se > 0 else 1.0
    if not se > 0:
        return diff, 0.0, 1.0
    return diff, float(stats.t.ppf(1 - alpha, df)) * se, float(stats.t.sf(abs(diff) / se, df))


MOST_RUNS = 1000
"""The largest count the verdict prints: past it, the scatter of one run cannot tell the counts apart."""


def _one_time_in(p: float) -> str:
    runs = round(1 / p) if p > 0 else math.inf
    return f"1 time in {runs}" if runs <= MOST_RUNS else f"less than 1 time in {MOST_RUNS}"


def _by_direction(pairs: list[tuple[str, float]], directions: tuple[str, ...]):
    """The scores of each direction, oriented so a shift the cued way is positive in a one-direction run:
    (up, down) for two directions, (oriented, None) for one, or None when a direction is too thin."""
    if len(directions) == 1:
        sign = 1.0 if directions[0] == POSITIVE else -1.0
        scores = [sign * s for _, s in pairs]
        return (scores, None) if len(scores) >= MIN_PER_DIRECTION else None
    up = [s for d, s in pairs if d == POSITIVE]
    down = [s for d, s in pairs if d == NEGATIVE]
    return (up, down) if min(len(up), len(down)) >= MIN_PER_DIRECTION else None


def _run_verdict(run: TrialRun, groups: list[tuple[str, list[Trial]]], words: dict[str, str], unit: str,
                 decimals: int, floor: float, checks: dict[int, list] | None) -> list[str]:
    """The words at the end of a run: per group the asked-for difference with its chance range, then
    whether chance explains it, then what each extra channel showed in the same windows."""
    directions = tuple(dict.fromkeys(run.directions))
    split = [(prefix, _by_direction([(r.required, r.score_uv) for r in pool], directions))
             for prefix, pool in groups]
    split = [(prefix, scores) for prefix, scores in split if scores is not None]
    if not split:
        return []
    alpha = RUN_ALPHA / len(split)
    lines, judged, wrong, any_scatter = [], False, False, False
    for prefix, (up, down) in split:
        diff, reach, p = _chance_test(up, down, alpha)
        if reach <= 0:
            continue
        any_scatter = True
        reach = max(floor, reach)
        name = f"{words[POSITIVE]} minus {words[NEGATIVE]}" if down is not None else "mean toward the cue"
        shown = round(diff, decimals) + 0.0             # no "-0.0" for a difference that rounds to zero
        lines.append(f"{prefix}{name} {shown:+.{decimals}f} {unit} · chance range ±{reach:.{decimals}f} {unit}")
        lines.append(f"{prefix}chance gives this much or more{'' if shown >= 0 else ' the wrong way'}: "
                     f"{_one_time_in(p)}")
        judged = judged or diff > reach
        wrong = wrong or diff < -reach
    if not any_scatter:
        return []
    extra = _extra_lines(run, directions, words, unit, decimals, checks or {})
    tripped = any(state == "followed" for state, _ in extra)
    vouched = any(state == "inside" for state, _ in extra)
    if judged:
        keep = not tripped and not vouched
        lines.append("past the chance range" + (" · keep this strategy" if keep else ""))
    elif wrong:
        lines.append("past the chance range the wrong way · the strategy gives the opposite shift · "
                     "try the opposite strategy in the next run")
    else:
        lines.append("inside the chance range · this run does not show control · start a new run")
    for state, texts in extra:
        if state == "inside" and judged and not tripped:
            texts = [texts[0] + " · keep this strategy"]
        lines += texts
    return lines


def _extra_lines(run: TrialRun, directions: tuple[str, ...], words: dict[str, str], unit: str,
                 decimals: int, checks: dict[int, list]) -> list[tuple[str, list[str]]]:
    """What each extra channel showed in the training trial's windows: ("followed", lines) when it also
    followed the cues, ("inside", lines) when it did not, ("no signal", lines) when it checked too few.
    Only a trial with a result has windows to check, so the counts are out of those."""
    out = []
    for channel in sorted(checks):
        scores = checks[channel]
        name = f"ch{channel}"
        windowed = [(r.required, s) for r, s in zip(run.results, scores) if r.valid]
        pairs = [(required, s) for required, s in windowed if s is not None]
        split = _by_direction(pairs, directions)
        if split is None:
            out.append(("no signal", [f"{name} had no signal in {len(windowed) - len(pairs)} of "
                                      f"{len(windowed)} trials · {name} did not check this run"]))
            continue
        # An extra channel that moves with the cues either way is a finding, so both sides count here.
        diff, reach, p = _chance_test(*split, RUN_ALPHA)
        if 2 * p < RUN_ALPHA:
            what = (f"{words[POSITIVE]} minus {words[NEGATIVE]}" if split[1] is not None
                    else "mean toward the cue")
            out.append(("followed", [f"{name} also followed the cues: {what} {diff:+.{decimals}f} {unit}",
                                     "an eye movement, an electrode movement or the trained potential can "
                                     "give this",
                                     f"keep the eyes and the head still in the next run, then compare {name}"]))
        else:
            out.append(("inside", [f"{name} did not follow the cues · checked in {len(pairs)} of "
                                   f"{len(windowed)} trials"]))
    return out


def trial_summary(run: TrialRun, words: dict[str, str], unit: str, decimals: int = 2,
                  checks: dict[int, list] | None = None, floor: float = 0.0) -> str:
    """The TRIALS panel text: per direction, how many trials succeeded and the mean shift from the
    baseline. When the run has no-feedback trials, they are counted on their own, because a shift held
    without the circle is the real test of control. The signed mean says more than the count: a pool that
    leans the asked-for way shows voluntary control building, before many trials cross the success margin.

    **The run is judged once, when it is done**, from its own scores: words after every trial would be a
    look after every trial, and chance passes one of many looks far more often than one in twenty. Each
    group is compared at the 5% level split over the groups judged, and its chance range is at least
    `floor`, the program's resolution. `checks` maps an extra channel's number to one score per trial,
    None where it did not vouch for the trial; one that followed the cues withholds "keep this strategy"."""
    if not run.results:
        return "no trials yet"
    valid = run.valid_results
    if any(r.transfer for r in run.results):
        groups = [("feedback · ", [r for r in valid if not r.transfer]),
                  ("no feedback · ", [r for r in valid if r.transfer])]
    else:
        groups = [("", valid)]
    lines = [line for prefix, pool in groups for line in _direction_lines(pool, prefix, words, unit, decimals)]
    void = sum(1 for r in run.results if not r.valid)
    if void:
        lines.append(f"{void} with no result")
    if run.done:
        # Drawn every frame once the run is done, and a done run does not change: worked out once.
        key = (len(run.results), tuple(words.items()), unit, decimals, floor,
               tuple((c, tuple(s)) for c, s in sorted((checks or {}).items())))
        cached = getattr(run, "_verdict", None)
        if cached is None or cached[0] != key:
            cached = run._verdict = (key, _run_verdict(run, groups, words, unit, decimals, floor, checks))
        lines += cached[1]
    return "\n".join(lines) if lines else "no result yet"


def trial_settings_rows(session: TrialTraining, changed: Callable[[Callable], Callable], display: str = "circle",
                        cue_gap: bool = False, help: ft.Control | None = None,
                        width: int = 96, boxes: dict | None = None) -> list[ft.Control]:
    """The rows that set a run: direction, length, trials without feedback, the fade, the steps,
    **succeeds at** and **full at**, then the restart and the three test sounds. `changed` wraps a setter
    so the screen rebuilds. `display` names what the feedback shows ("circle", "bar"). `cue_gap` adds the
    row for the wait from the cue to the baseline. `help` goes at the end of the restart row.

    Where the program offers AUTOMATIC, **succeeds at** and **full at** each get the pill and a second
    line that says where the value comes from. `boxes`, when given, receives their two boxes by setting
    name, for `show_scale` to keep the shown values current."""
    timing, units, words = session.TIMING, session.UNITS, session.DIRECTION_WORD
    active_s = session.active_s
    low_length, high_length = session.trial_length_range

    def set_trials(n: float) -> None:
        session.n_trials = int(n)
        session.restart_trials()

    def set_transfer(percent: float) -> None:
        session.transfer_fraction = max(0.0, min(1.0, percent / 100.0))
        session.restart_trials()

    def set_margin(value: float) -> None:
        session.set_success_at(value, typed=True)
        session.restart_trials()

    def shown(value: float) -> float:
        return round(value, units.decimals)

    def toggle_margin(_e=None) -> None:
        # A press on the lit pill keeps the value on screen as a fixed one, so the pill always acts.
        if session.success_at_mode == AUTOMATIC:
            session.set_success_at(shown(session.success_at), typed=True)
        else:
            session.set_success_at_mode(AUTOMATIC)
        session.restart_trials()

    def toggle_full(_e=None) -> None:
        if session.full_at_mode == AUTOMATIC:
            session.set_full_at(shown(session.full_at), typed=True)
        else:
            session.set_full_at_mode(AUTOMATIC)

    def set_full(value: float) -> None:
        session.set_full_at(value, typed=True)

    def scale_row(name: str, box: ft.Control, mode: str, note: str, automatic_note: str,
                  on_automatic: Callable, key: str) -> ft.Control:
        if AUTOMATIC not in session.SCALE_MODES:
            return labelled(name, box, note=note, width=width)
        choice = pill(AUTOMATIC, mode == AUTOMATIC, on_automatic)
        if boxes is not None:
            boxes[f"{key}{PILL_KEY}"] = choice
        line = ft.Row([box, choice], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        source = ft.Text(automatic_note if mode == AUTOMATIC else FIXED, size=t.SMALL, color=t.FAINT,
                         data=automatic_note)
        if boxes is not None:
            boxes[f"{key}{NOTE_KEY}"] = source
        return labelled(name, ft.Column([line, source], spacing=2), width=width)

    # An automatic value is a measurement, so its box shows it to the decimals a shift is written with.
    places = units.decimals if AUTOMATIC in session.SCALE_MODES else None
    margin_box = num_box(session.success_at, changed(set_margin), suffix=units.unit, step=units.step, low=0,
                         high=units.max_success, width=58, places=places)
    full_box = num_box(session.full_at, changed(set_full), suffix=units.unit, step=units.step,
                       low=units.min_full_at, high=units.max_full_at, width=58, places=places)
    if boxes is not None:
        boxes.update(success_at=margin_box, full_at=full_box)

    def set_feedback_window(s: float) -> None:
        session.feedback_window_s = max(1.0, min(active_s, float(s)))

    def set_update(s: float) -> None:
        session.update_s = max(0.0, min(active_s, float(s)))

    def audition(required: str, mark: bool = False) -> None:
        sound = cues.tone_at(CUE_PITCH[required])
        session.cue(cues.held(sound) if mark else sound, CUE_VOLUME)

    rows = [
        # One line for the choices with one direction, one for those with both, so a narrow column never
        # leaves one pill on a line of its own.
        labelled("direction",
                 ft.Column([ft.Row([pill(d, session.trial_direction == d,
                                         changed(lambda _e, d_=d: session.set_trial_direction(d_)))
                                    for d in session.DIRECTIONS if (len(session.DIRECTIONS[d]) == 1) == single],
                                   spacing=6, wrap=True)
                            for single in (True, False)], spacing=6, expand=True),
                 width=width),
        labelled("trials", num_box(session.n_trials, changed(set_trials), step=5, low=1, high=200,
                                   width=52),
                 note=f"about {_duration(session.n_trials * session.trial_seconds)}", width=width),
        labelled("trial length", num_box(session.trial_length, changed(session.set_trial_length), suffix="s",
                                         step=1, low=low_length, high=high_length, width=58),
                 note=f"{active_s:g} s {session.PHASE_WORD[ACTIVE]}", width=width),
    ]
    if cue_gap:
        rows.append(labelled("baseline starts",
                             num_box(session.cue_s, changed(session.set_cue_gap), suffix="s", step=0.5,
                                     low=0, high=MAX_CUE_GAP_S, width=58),
                             note=("at the tone" if session.cue_s <= 0 else "after the tone"),
                             width=width))
    rows += [
        labelled("without feedback",
                 num_box(round(session.transfer_fraction * 100), changed(set_transfer),
                         suffix="%", step=10, low=0, high=100, width=52),
                 note="of the trials", width=width),
        labelled(f"show {display} for",
                 num_box(session.feedback_window_s, changed(set_feedback_window),
                         suffix="s", step=1, low=1, high=active_s, width=58),
                 note=("the whole effort" if session.feedback_window_s >= active_s
                       else f"then off for {active_s - session.feedback_window_s:g} s"),
                 width=width),
        labelled("update every",
                 num_box(session.update_s, changed(set_update), suffix="s",
                         step=0.5, low=0, high=active_s, width=58),
                 note=("0 = live" if session.update_s <= 0 else f"steps of {session.update_s:g} s"),
                 width=width),
        scale_row("succeeds at", margin_box, session.success_at_mode, "past the baseline",
                  _spreads(SUCCESS_SPREADS), changed(toggle_margin), "success_at"),
        scale_row("full at", full_box, session.full_at_mode, "from the baseline",
                  _spreads(FULL_AT_SPREADS),
                  changed(toggle_full), "full_at"),
        ft.Row([button("restart run", changed(lambda _e: session.restart_trials()))]
               + ([ft.Container(expand=True), help] if help is not None else []),
               vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Row([test_button(words[POSITIVE], lambda _e: audition(POSITIVE)),
                test_button(words[NEGATIVE], lambda _e: audition(NEGATIVE)),
                test_button("success", lambda _e: audition(POSITIVE, mark=True))],
               spacing=6),
    ]
    return rows


def _spreads(n: float) -> str:
    return "equal to the noise" if n == 1.0 else f"{n:g} times the noise"


NOTE_KEY = "_note"
"""The suffix under which `trial_settings_rows` puts a scale row's second line in `boxes`."""

PILL_KEY = "_pill"
"""The suffix under which `trial_settings_rows` puts a scale row's *automatic* pill in `boxes`."""


def show_scale(session: TrialTraining, boxes: dict) -> None:
    """Keep the **succeeds at** and **full at** boxes on the value in use while their mode is automatic,
    and their second line on where it comes from: until there is a chance spread, the line says that none
    exists yet, so the floor in the box does not read as one. Call
    it on Flet's loop. A box with an edit in progress keeps the edit."""
    in_use = dict(zip(("full_at", "success_at"), session.scale_in_use()))
    record, run = session.trial_scale, session.trials
    if record is not None and record is run.scale and run.phase in (ACTIVE, RESULT):
        spread = record.spread
    else:
        measured = session.chance_spread()
        spread = None if measured is None else measured[0]
    for name, box in boxes.items():
        if name.endswith(PILL_KEY):
            continue
        if name.endswith(NOTE_KEY):
            key = name[:-len(NOTE_KEY)]
            if getattr(session, f"{key}_mode") == AUTOMATIC:
                box.value = box.data if spread is not None else f"{box.data} · none yet"
        elif getattr(session, f"{name}_mode") == AUTOMATIC and callable(box.data):
            box.data(in_use[name])


class TrialView:
    """The trial controls a program puts on its screen: the fixation circle, the settings rows, the trial
    history and the heart stats. The program places them and calls the `draw_` methods each frame.

    Each part of a trial is its own picture, so the person does not start the task before the feedback: the
    ground behind the circle is dim through the rest, the cue and the baseline, and bright through the
    feedback and the verdict; the lead-in draws `lead_in_shapes`; the circle is grey on the side the cue
    did not ask for until it passes **succeeds at**, which a thin ring marks; and a thin ring shows the
    mean of the window so far, which is what the trial is scored on. The two lines under the circle are
    reserved in every phase, so a verdict never changes the panel's height.

    `steady_light` is for a signal that answers light, which is the pupil: the ground keeps one brightness
    and the circle one colour through the feedback, because a change in either at the start of the window,
    or when the circle crosses the baseline, is a stimulus that the pupil follows.
    """

    def __init__(self, session: TrialTraining, circle_side: int, heart: bool, steady_light: bool = False):
        self.session = session
        self.circle_side = circle_side
        self.steady_light = steady_light
        self.margin = ARROW_MARGIN
        self.canvas_side = circle_side + 2 * self.margin
        self.circle = cv.Canvas(shapes=[], width=self.canvas_side, height=self.canvas_side)
        self.ground: ft.Container | None = None
        self.direction_text = ft.Text("", size=t.VALUE, color=t.TEXT, weight=ft.FontWeight.W_300)
        self.phase_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.count_text = ft.Text("", size=t.BODY, color=t.MUTED)
        self.verdict_text = ft.Text("", size=t.SMALL, color=t.MUTED, max_lines=1, no_wrap=True)
        self.verdict_detail = ft.Text("", size=t.SMALL, color=t.FAINT, max_lines=1, no_wrap=True)
        self.history_cells = ft.Row(spacing=3, wrap=True)
        self.history_text = ft.Text("", size=t.SMALL, color=t.MUTED)
        self.history_panel = panel("TRIALS", self.history_cells, ft.Container(height=8), self.history_text)
        self.heart = heart
        self.heart_stats = HeartStats(session)
        self.heart_stats_panel = self.heart_stats.panel

    def fixation_panel(self, title: str = "FIXATION", help: ft.Control | None = None) -> ft.Control:
        side = self.canvas_side
        circle = ft.Row([ft.Container(content=self.circle, width=side, height=side)],
                        alignment=ft.MainAxisAlignment.CENTER)
        if not self.steady_light:
            # The whole width of the panel, so the switch from dim to bright is the largest change on the
            # screen. The border is there in both states, in the ground's own colour when dim, so the
            # switch never moves anything.
            self.ground = ft.Container(content=circle, padding=ft.Padding(0, 6, 0, 6), border_radius=8,
                                       bgcolor=t.SURFACE_HI, border=ft.Border.all(2, t.SURFACE_HI))
            circle = self.ground
        return panel(title,
                     circle,
                     ft.Container(height=6),
                     # A fixed height: the verdict's mark comes from another font, and a line that grows with
                     # it would move the panel at every verdict.
                     ft.Container(content=ft.Row([self.direction_text], alignment=ft.MainAxisAlignment.CENTER),
                                  height=DIRECTION_LINE_H, alignment=ft.Alignment.CENTER),
                     ft.Row([self.count_text], alignment=ft.MainAxisAlignment.CENTER),
                     ft.Container(height=4),
                     ft.Row([self.phase_text], alignment=ft.MainAxisAlignment.CENTER),
                     *(ft.Container(content=ft.Row([line], alignment=ft.MainAxisAlignment.CENTER),
                                    height=VERDICT_LINE_H)
                       for line in (self.verdict_text, self.verdict_detail)),
                     help=help)

    def settings_rows(self, changed: Callable[[Callable], Callable]) -> list[ft.Control]:
        """The active mode's rows for the MODE panel. `changed` wraps a setter so the screen rebuilds."""
        return trial_settings_rows(self.session, changed)

    def _set_ground(self, bright: bool, edge: bool = True) -> None:
        if self.ground is None:
            return
        self.ground.bgcolor = t.SURFACE if bright else t.SURFACE_HI
        self.ground.border = ft.Border.all(2, t.ACCENT if bright and edge else t.SURFACE_HI)

    def _waiting(self, line: str) -> None:
        self.direction_text.value, self.direction_text.color = "waiting", t.FAINT
        self.count_text.value = ""
        self.phase_text.value = line
        self._set_ground(False)
        self.circle.shapes = lead_in_shapes(self.canvas_side, None, self.margin)

    def _lead_in_part(self) -> float:
        """How much of the cue and the baseline has passed, from 0 at the cue to 1 at the feedback."""
        run = self.session.trials
        total = run.cue_s + run.baseline_s
        if total <= 0:
            return 1.0
        to_go = run.phase_remaining() + (run.baseline_s if run.phase == CUED else 0.0)
        return min(1.0, max(0.0, 1.0 - to_go / total))

    def draw_fixation(self, passive_line: str) -> None:
        """The circle and the lines under it. `passive_line` is what passive mode says under the circle."""
        session, run = self.session, self.session.trials
        words = session.DIRECTION_WORD
        self.verdict_text.value = self.verdict_detail.value = ""
        if session.mode == "passive":
            self.direction_text.value, self.direction_text.color = "passive", t.MUTED
            self.count_text.value = ""
            self.phase_text.value = passive_line
            # No trial, so no phase to tell apart: the bright ground without the feedback window's edge.
            self._set_ground(True, edge=False)
            self.circle.shapes = circle_shapes(self.canvas_side, session.signal, t.ACCENT, margin=self.margin)
            return
        required = session.required
        if session.state is None:
            self._waiting(session.trial_waiting)
            return
        hold = None if run.started else session.trial_hold()
        if hold is not None:
            self._waiting(hold)
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
        name = session.PHASE_WORD.get(run.phase, run.phase)
        # At a verdict the run has already counted the trial, so the number is the one being judged.
        number = run.last.index if run.phase == RESULT and run.last is not None else run.trial_number
        phase = (f"trial {number} of {run.n_trials} · {name} · {run.phase_remaining():.0f} s"
                 if not run.done else f"the run is over · restart it in {session.RESTART_PLACE}")
        if not run.done and required is not None and not session.direction_known():
            phase += " · wait for the tone"
        level, colour, mean, marked = None, t.ACCENT, None, None
        limit = session.scale_in_use()[1]
        if run.phase == ACTIVE and session.feedback_now:
            level = session.circle_level(session.shown_shift)
            goal = "inside the ring" if required == NEGATIVE else "outside the ring"
            phase += f" · pull the circle {goal}"
            mean = session.circle_level(session.shown_mean)
            if session.direction_known():
                marked = required
            # The cued colour says the shift is past **succeeds at**, the size the trial is scored against.
            if (not self.steady_light and level is not None
                    and not succeeds(session.shown_shift, required, limit)):
                colour = t.FAINT
        elif run.phase == ACTIVE and run.transfer:
            phase += " · no feedback in this trial"
        elif run.phase == ACTIVE:
            phase += " · hold it · the circle is off"
        self.phase_text.value = phase
        # A trial cut by a gap, a pause or a seam has no verdict phase: the run rests at once. Its two lines show through that
        # rest instead, or its cause would be on the screen nowhere.
        if run.phase == ITI and not run.done and run.last is not None and run.last.cause in (GAP, PAUSED, SEAM):
            self.verdict_text.value, self.verdict_detail.value = verdict_line(run.last, session)
        if run.phase == RESULT and run.last is not None:
            last = run.last
            asked = words[last.required]
            self.verdict_text.value, self.verdict_detail.value = verdict_line(last, session)
            if not last.valid:
                level, colour = None, t.MUTED
                self.direction_text.value, self.direction_text.color = "no result", t.MUTED
            else:
                level = session.circle_level(last.score_uv)
                colour = t.GOOD if last.success else t.BAD
                marked = last.required
                self.direction_text.value = f"{'✓' if last.success else '✗'}  {asked}"
                self.direction_text.color = colour
        arrows = None
        lead_in = run.phase in (CUED, BASELINE)
        # The arrows are the goal, and the goal holds for the whole window: with the circle off, and in a
        # trial without feedback, they still say which way to go.
        if required is not None and session.direction_known() and run.phase == ACTIVE:
            arrows = "shrink" if required == NEGATIVE else "expand"
        bright = run.phase in (ACTIVE, RESULT)
        self._set_ground(bright)
        if bright:
            # The ring marks **succeeds at** on the cued side, from the trial's record: in a window with the
            # circle on, and at every valid verdict. At a limit of 0 the baseline ring is the limit.
            mark = (session.circle_level(limit if marked == POSITIVE else -limit)
                    if marked is not None and limit > 0 else None)
            self.circle.shapes = circle_shapes(self.canvas_side, level, colour, arrows=arrows, mean=mean,
                                               margin=self.margin, limit=mark)
        else:
            self.circle.shapes = lead_in_shapes(self.canvas_side,
                                                self._lead_in_part() if lead_in and not run.done else None,
                                                self.margin)

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
        self.history_text.value = trial_summary(run, session.DIRECTION_WORD, session.UNITS.unit,
                                                floor=session.UNITS.min_full_at)

    def draw_heart_stats(self) -> None:
        self.heart_stats.draw()


class HeartStats:
    """The HEART STATS panel: the mean heart-rate change from baseline for each direction, and the heart
    rate over the run, in the effort windows and in the rests."""

    def __init__(self, session: TrialTraining):
        self.session = session
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
        self.panel = panel("HEART STATS", self._heart_no_data, self._heart_body)

    def draw(self) -> None:
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
