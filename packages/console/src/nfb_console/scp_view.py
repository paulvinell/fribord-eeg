"""The slow-potential screen: a run of trials, each with its own baseline and a cued direction.

Every other protocol here is one undifferentiated stretch of signal judged against a threshold. This one is
a **sequence**: a cue names the direction, two seconds are averaged into that trial's baseline, the shift
against it is fed back for the rest of the trial length, and the trial is scored on the mean of that window. The state
machine is `nfb_console.trials`, which owns no signal and no screen; the metric is `nfb.dsp.scp.SlowShift`,
a lowpassed signed voltage with no drift follower. This screen is the third piece: it feeds one into the
other and draws the result.

**Why the trial is the unit and not a threshold.** What is being trained is *differentiation* — producing a
shift on demand, in the direction asked for — so the interesting quantity is a per-trial verdict rather
than a fraction of the session spent on one side of a line. That also decides where the de-trend lives:
each trial subtracts its own baseline, which removes standing offset and electrode drift at the moment they
matter, which is why the metric under it carries no high-pass of its own.

**The screen is watched, but the protocol is heard.** A trainee doing this has no reason to look at anything
during the cue, so the direction arrives as a tone — 620 Hz for up, 310 Hz for down, an octave apart for the
same reason alpha-theta's two reward tones are far apart: they have to be told from each other with the eyes
closed. The verdict is the same tone doubled on a success, and **silence otherwise**, which is the shape
every discrete mechanism in this repo takes: the absence is the message.

**One thing is deliberately absent from the feedback: the alert**, because the condition is crossed
constantly by construction — half a trial's readings are on the wrong side of a baseline by definition, so
an alert would fire on the ordinary behaviour of the signal. Music is offered, as on the other trial screens.

**A transfer trial has a cue and a verdict and no display.** That is what weans the skill off the screen,
and it is why the live circle is *hidden* rather than frozen during one: a circle that stopped moving reads
as a fault, where an absent one reads as the design — which the line under it says in words.

**The display is the shared fixation circle** (`trial_training.TrialView`), the one Pupil and HEG train
with, because a slow potential needs a still gaze and the circle gives the eyes a point to hold. It is
drawn one picture to a phase, so the baseline cannot be taken for the feedback: dim with a ring drawing
itself through the lead-in, bright with the moving circle through the feedback window.
"""

from __future__ import annotations

import dataclasses
import math
import time
from bisect import bisect_right
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import SlowShift, shift_survival
from nfb.dsp.artifact import input_limit_mask
from nfb.dsp.chance import PairSpread
from nfb.dsp.eyes import EyeFit, EyeSubtractor

from . import chrome, cues, eye_calibration, heart_panel, plan_pane, signal_pane, spectrum_pane, trace_rows, trends_pane
from . import theme as t
from .plots import place_value_labels, signed_shapes, value_axis
from .protocols import Protocol
from .targets import Target
from .traces import RawRing, TraceBuffer, WaveRing
from .training import METRIC_HZ, TrainingSession, true_metric_hz
from .trial_training import (AUTOMATIC, CONDITION_AT, FIXED, TrialTiming, TrialTraining, TrialUnits, TrialView,
                             show_scale, trial_settings_rows, trial_summary)
from .eye_calibration import EyeAligner, EyeCalibration
from .level_plot import LEVEL_SECONDS, LevelHistory, LevelPlot, limit_uv
from .trials import (ACTIVE, ACTIVE_S, BASELINE, BASELINE_S, CUED, DEFAULT_TRIALS, EXTRA, ITI_RANGE_S, JUMP_S, LIMIT,
                     MIN_BASELINE_FRACTION, NEGATIVE, NO_SIGNAL, PAUSED, POSITIVE, REJECT_LIMIT, REJECTED, RESULT,
                     RESULT_S, SEAM, THIN_BASELINE, TRANSFER_FRACTION, thin_baseline, too_many_invalid)
from .target_card import NOTE_H
from .widgets import button, help_button, labelled, num_box, panel, pill, test_button

HANDLES = ("SCP",)

from .trial_training import CUE_PITCH, CUE_VOLUME  # noqa: E402  (the cue tones are the trial piece's)

TIMING = TrialTiming(cue_s=0.0, baseline_s=BASELINE_S, active_s=ACTIVE_S, result_s=RESULT_S, iti_s=ITI_RANGE_S)
"""The published slow-potential trial: the baseline starts at the cue, two seconds of it, six of feedback. The
feedback is the session's `active_s`, which **trial length** sets; this is where it starts."""

DEFAULT_RANGE_UV = 10.0
"""The value **full at** takes on *fixed* until one is typed. On *automatic*, the default, it is two chance
spreads of the person's own signal and this number is not used."""

MIN_RANGE_UV = 1.0
"""A floor, for the reason `wave_view.MIN_RANGE_UV` gives: below it the level is a picture of electrode
noise."""

UNITS = TrialUnits(unit="µV", full_at=DEFAULT_RANGE_UV, min_full_at=MIN_RANGE_UV, max_full_at=100.0,
                   max_success=50.0, step=1.0, decimals=1)

BEST_CORNER_HZ = 0.01
SURVIVAL_FLOOR = 0.5
"""Below this fraction of a shift's mean surviving the amplifier's corner, the screen says so. Half is not
a delicate boundary — at the device's 0.5 Hz default the figure is about 5% — it is simply the point past
which the number on the card is more filter than trainee."""

MAX_FEEDBACK_S = max(t for t in range(1, 120) if shift_survival(BEST_CORNER_HZ, t) >= SURVIVAL_FLOOR)
"""The longest feedback window **trial length** offers: at the best corner, 0.01 Hz, a held shift decays with
a 16 s time constant, and past this window its mean keeps less than `SURVIVAL_FLOOR` of it (25 s)."""

WAVE_CENTRE_HZ, WAVE_HIGH_HZ = 0.5, 1.0
WAVE_CYCLES = 8
"""A ~16 s window on the slow signal, which is one trial plus its rest either side — the span an operator
reads to see whether a shift was produced and whether it came back."""

SPECTRUM_LO, SPECTRUM_HI = 1.0, 60.0
TRACE_SECONDS = 5.0
SESSION_MINUTES = 20.0

WAVE_GAINS = (5.0, 10.0, 20.0, 50.0)
WAVE_FLOOR_UV = 1.0

SIDE_W = 380
AXIS_W = 44
WAVE_H = 170
TRACE_H = 74
BAR_W = 200
CIRCLE_SIDE = 200
CELL_W, CELL_H = 11, 16
RUN_LABEL_W = 104

EXTRA_MISS = {LIMIT: "at the input limit", NO_SIGNAL: "with no signal", REJECTED: "with readings rejected"}
"""Why an extra channel checked nothing in a trial, in the words of its status line, first cause first."""

MAX_EXTRAS = 2
"""Extra channels that can be lit at once. The check reads each one on its own, and two cover the two
forehead sites an eye or a head movement reaches."""

TONE_HZ = 880.0
"""The calibration's blink tone: an octave over the high cue, so it is not taken for a trial."""

TRIAL_HELP = (
    "One trial runs in order: a cue, a baseline, a feedback window, a verdict, then a rest.\n\n"
    "The cue names the direction and sounds a tone: high for up, low for down. The two tones are an "
    "octave apart, so you can tell them apart with your eyes closed. With the *random* direction, the cue "
    "sounds a tone between the two and shows a question mark. The direction tone then sounds when the "
    "feedback starts.\n\n"
    "The baseline is the average of the first two seconds. Everything after it is measured against this "
    "baseline.\n\n"
    "Keep your eyes on the point in the middle of the circle for the whole run. Eye movements change the "
    "slow potential, and a still gaze keeps them out of the signal.\n\n"
    "During the cue and the baseline, the ground behind the point is grey. A dashed ring draws itself "
    "around the point. The ring is complete when the feedback starts. Do not do the task while the ring "
    "draws.\n\n"
    "When the feedback starts, the ground changes to white, and the circle and the arrows appear. The circle "
    "shows the slow potential against the baseline. It grows when the potential goes up and it shrinks "
    "when the potential goes down, on every trial. The arrows show the cued direction: out for up, in for "
    "down.\n\n"
    "A thin blue ring on the cued side of the dashed ring marks **succeeds at**. The circle is blue "
    "when it is past the blue ring, and grey at all other positions. At a **succeeds at** of 0 there is no "
    "blue ring, because the dashed ring is the limit.\n\n"
    "The circle moves in proportion to the shift up to **full at**. Past that value, it moves less and "
    "less, but it never stops. A shift of three times **full at** shows further out than a shift of two "
    "times **full at**.\n\n"
    "Each trial takes its **full at** and **succeeds at** when its feedback starts, and keeps them to its "
    "verdict. A change acts from the next trial.\n\n"
    "The trial is scored on the mean of the feedback window, not on one instant. A thin faint ring shows "
    "that mean so far. A brief spike does not earn the trial, and a brief lapse does not lose it. The "
    "mean must pass the blue ring for the trial to succeed.\n\n"
    "A success sounds the trial's cue tone twice. A failure and a trial with no result stay silent. The "
    "circle stays at the mean shift, green for a success and red for a failure. The first line under the "
    "circle says what the trial asked for, and how far and which way the slow potential went. The second "
    "line compares that shift with **succeeds at**, and says where that value comes from.\n\n"
    "A trial without feedback has the cue and the verdict, but no circle. The circle stays hidden for the "
    "whole feedback window instead of freezing. The arrows stay, because the direction still applies. The "
    "blue ring shows at the verdict.\n\n"
    "A trial has *no result* when its readings cannot give a fair score. The first line under the circle "
    "names the cause, and the second line says what to do. There are six causes in the readings. The "
    "signal reaches the input limit. The channel has no signal. Readings do not arrive. The readings stop "
    f"for more than {JUMP_S:g} s. Fewer than {MIN_BASELINE_FRACTION:.0%} of the baseline readings are "
    f"valid. The program rejects more than {REJECT_LIMIT:.0%} of the readings. When the program subtracts "
    "eye movements, an extra channel without valid signal is a seventh cause. A trial in "
    "which you pause the session, or in which a looped file starts again, also has no result. A trial with "
    "no result appears in the history, but no count includes it.\n\n"
    "When you choose a heart-rate source on the device screen, two heart panels appear beside the trials. "
    "They show the mean heart-rate change from baseline for each direction, and the heart rate across the "
    "run. A strategy that works through general arousal moves the heart in both directions."
)

HISTORY_HELP = (
    "Every completed trial, in order, oldest at the left.\n\n"
    "A filled cell is a success, and an outlined one is a failure. A muted cell is a trial with no result. "
    "A dot under a cell marks a trial without feedback.\n\n"
    "Under the cells, each direction has its own line: how many trials succeeded, and the mean shift from "
    "the baseline. The directions are kept apart because the skill is *differentiation*. Somebody who has "
    "learned to produce one direction, and drifts through the other, has a good overall figure. But that "
    "person has not learned the thing being trained.\n\n"
    "The mean shift says more than the count. A direction whose mean leans the cued way shows control "
    "that grows before many trials pass **succeeds at**. A mean the wrong way shows a strategy that "
    "works backwards.\n\n"
    "When the run has trials without feedback, they get their own lines. A shift held without the circle is "
    "the real test of control. Trials with no result are in no line and are counted on their own. Many of "
    "them show a problem with the electrodes or the connection, not with the training.\n\n"
    "When the run is done, the history compares the two directions once: *up minus down*, the difference "
    "of their mean shifts. Chance takes the difference past the *chance range* in the cued direction in 1 run of "
    "20. The next line tells how frequently chance gives the difference of this run. "
    "A run in one direction compares its mean shift with zero.\n\n"
    "Past the chance range in the cued direction, the history tells you to keep the strategy. Inside it, "
    "the run does not show control. Past it the wrong way, the strategy moves the potential the other way.\n\n"
    "The comparison is made only at the end of the run. A comparison after every trial gives chance many "
    "more ways to pass."
)

LEVEL_HELP = (
    f"The standing level of each channel over the last {LEVEL_SECONDS / 60:g} minutes, in µV. Each point is "
    "the mean of half a second of samples, so the EEG does not show. The faint band around a curve goes from "
    "the lowest to the highest sample, so a short spike shows. The dashed lines are the input limits.\n\n"
    "A curve or a band at a limit line means that the input is outside its range, and its readings do not "
    "count. The curve comes back by itself when the electrode settles, or after you correct the electrode."
)

SHIFT_HELP = (
    "The lowpassed, signed voltage that trials are scored on. This is the level itself, not a shift: a "
    "trial's baseline turns it into a shift, and between trials there is no baseline.\n\n"
    "The filter passes below 1 Hz, order 2, with no high-pass and no drift follower. It removes the "
    "ongoing EEG, so the trace moves like a level instead of like raw EEG. It keeps everything under "
    "1 Hz, including electrode drift; each trial's own baseline removes that drift instead.\n\n"
    "The window shown is about sixteen seconds: one trial, with its rest on either side."
)

SIGNAL_HELP = (
    "One channel, lowpassed, signed. The level sent to every output is the mean shift from this trial's "
    "baseline, oriented so the cued direction always reads as a rise.\n\n"
    "**full at** sets the sensitivity. The level reads 100% for a shift of **full at** the cued way, 0% "
    "for the same shift the wrong way, and 50% at the baseline. Narrow the range and small shifts swing "
    "the output further, but the output then stays at 0% or 100% more often. Widen the range and the "
    "output moves less.\n\n"
    "**noise** is how far the result of one trial moves by chance on your own signal, with no effort from "
    "you. On *automatic*, **succeeds at** is equal to the noise and **full at** is 2 times the noise. The program measures it from "
    "pairs of readings in the last 120 s of signal, and it has a value from the first readings. A typed "
    "value sets *fixed*. A press on *automatic* changes between *automatic* and *fixed*, and *fixed* "
    "keeps the value that the box shows.\n\n"
    "Each trial takes **full at** and **succeeds at** once, when its feedback starts. The circle, the blue ring, "
    "the outputs and the verdict use these values until the trial ends.\n\n"
    "The sound and the shadow follow the level. The condition starts at the level of **succeeds at**, so "
    "the click sounds only for a shift past **succeeds at**. Music that stops on the condition follows the "
    "same limit.\n\n"
    "The circle in the trial panel does not stop at **full at**. It moves in proportion up to that value, "
    "and less and less past it, so a large shift still shows.\n\n"
    "**full at** only sets how much of the output's travel a shift uses. **succeeds at**, judged on the "
    "mean of the whole feedback window, decides the trial.\n\n"
    "The zero of the level is the trial's baseline. You cannot change it, because there is no other value "
    "to measure the shift from.\n\n"
    "Changing the channel restarts the run, because a run measured on two different electrodes is not "
    "one run."
)

RUN_HELP = (
    "Changing **direction**, **trials**, **trial length**, **baseline starts**, **without feedback** or "
    "**succeeds at** starts the run again. Results measured under one setting are not results under another.\n\n"
    "**direction** sets which shifts the run asks for. *random* asks for up and down in an order that "
    "you cannot predict, with the same number of each. The tone at the cue does not name the direction. "
    "The direction tone sounds when the feedback starts. Then you cannot push the signal the other way "
    "during the baseline to make the shift look larger. *alternating* asks for up and down in turn. "
    "*increase* and *decrease* ask for one direction only. With these three, the tone at the cue names "
    "the direction.\n\n"
    "**trials** sets the block size. The note beside it says how long the run takes.\n\n"
    f"**trial length** sets the baseline and the feedback window together. The baseline stays "
    f"{BASELINE_S:g} s, and the feedback window is the rest. It starts at {BASELINE_S + ACTIVE_S:g} s, the "
    f"usual protocol. The longest is {BASELINE_S + MAX_FEEDBACK_S:g} s. The amplifier slowly removes a held "
    "shift, and past that length less than half of it is left in the mean. A change also measures the "
    "**noise** again.\n\n"
    "**baseline starts** sets the time from the tone to the start of the baseline. At 0 s the baseline "
    "starts at the tone, as in the usual protocol. A longer time keeps the brain's answer to the tone "
    "out of the baseline.\n\n"
    f"**without feedback** sets the share of trials that run without a display. It starts at "
    f"{TRANSFER_FRACTION:.0%}, a typical share for an experienced trainee. For a first session, set it "
    "to 0%.\n\n"
    "**show circle for** sets how long the circle shows in each feedback window. After that time, the "
    "circle goes off and you hold the shift without it. The trial is still scored on the whole window.\n\n"
    "**update every** sets how often the circle moves. At 0 s the circle follows the shift live. At 2 s "
    "the circle shows the mean of the last 2 s and moves in steps.\n\n"
    "**succeeds at** sets how far past the baseline the mean shift must go to count. On *automatic* it is "
    "equal to the **noise** of your own signal, so a shift that chance gives often does not count. Type a "
    "value to set *fixed*, and press *automatic* to go back.\n\n"
    "**full at** sets the shift that takes the level to 100%. On *automatic* it is 2 times the noise. A "
    "change to **full at** acts from the next trial and does not start the run again."
)

EXTRA_HELP = (
    "Extra channels near the eyes can remove the noise of eye movements from the trained signal. Select "
    "one or two, then press {calibrate}. A dot moves across the window. Follow it with your eyes, then "
    "blink once at each tone.\n\n"
    "{short} takes about {short_s} s, and {long} takes about {long_s} s with twice the eye movements. A press "
    "on one of them sets the next calibration and does not start it. Take {long} when the *with* numbers "
    "are near the *without* numbers, or when two short calibrations give different results.\n\n"
    "The program measures how much of each eye movement and each blink gets into the trained signal. It "
    "learns from the first half of the movements and blinks, and tests on the second half. It subtracts "
    "only what makes them smaller in the second half. The table shows their size on the trained signal, "
    "without and with the subtraction. The row *rest* is the signal while your eyes are on the dot at the "
    "centre.\n\n"
    "{subtract} is lit while the subtraction is in use. A press stops it or starts it again, and the run "
    "starts again. A change of the channel or of the extra channels removes the calibration.\n\n"
    "While the program subtracts, an extra channel that has no signal or is at the input limit makes the "
    "readings invalid. Too many invalid readings give a trial no result.\n\n"
    "The second use is a check. The program measures each extra channel in the same windows as the trial. "
    "If one also follows the cues, the history says so. An eye movement, an electrode movement or the "
    "trained potential can give that, so the history names all three. The check changes no output and "
    "makes no trial void. A channel that has no signal, or that is at the input limit, checks no trial, "
    "and the history says so. A change to **extra channels** does not start the run again, unless it "
    "removes a calibration in use."
)
"""The part of the SIGNAL help about the extra channels, shown only where there is another channel. A
label is bold only where it is on the screen."""

EYE_ROWS = ("up-down", "left-right", "blinks", "rest")


def eye_sentence(fit: EyeFit) -> str:
    """What the program does after a calibration, in one sentence, and a second for the blinks it left out."""
    left = fit.blinks_left_out
    tail = ("" if not left else
            f" The calibration leaves out {left} {'blink' if left == 1 else 'blinks'} that reach the input limit.")
    return _eye_use(fit) + tail


def _eye_use(fit: EyeFit) -> str:
    why = ("that subtraction makes the signal larger when the eyes are still" if fit.rest_larger
           else "they show no gain")
    if fit.rotations_used and fit.blinks_used:
        return "The program now subtracts the eye movements and the blinks."
    if fit.rotations_used:
        return "The program now subtracts the eye movements, but not the blinks, because they show no gain."
    if fit.blinks_used:
        return f"The program now subtracts the blinks, but not the eye movements, because {why}."
    if fit.rest_larger:
        return "The program subtracts nothing, because the subtraction makes the signal larger when the eyes are still."
    return "The program subtracts nothing, because the calibration shows no gain."


def extra_help(session: "SCPSession") -> str:
    bold = lambda word, shown: f"**{word}**" if shown else word          # noqa: E731
    lit = bool(session.extras)
    about = {name: 5 * round(plan.total_s / 5) for name, plan in eye_calibration.PLANS.items()}
    text = EXTRA_HELP.format(calibrate=bold("calibrate eyes", lit), subtract=bold("subtract", session.can_subtract),
                             short=bold("short", lit), long=bold("long", lit),
                             short_s=about["short"], long_s=about["long"])
    fit = session.eye_fit
    if fit is None:
        return text
    rows = " ".join(f"{name.capitalize()}: {fit.sizes[name][0]:.1f} µV without, {fit.sizes[name][1]:.1f} µV with."
                    for name in EYE_ROWS)
    return text + "\n\nThe last calibration, on the trained signal: " + rows + " " + eye_sentence(fit)


def signal_help(session: "SCPSession") -> str:
    """The SIGNAL help as this session can use it: with the extra channels only where one can be picked."""
    return SIGNAL_HELP + ("\n\n" + extra_help(session) if len(session.specs) > 1 else "")


class SCPSession(TrialTraining, TrainingSession):
    """One channel, one slow signal, and a run of trials over it.

    Everything generic is `TrainingSession`. What is here is the metric, the trial engine, the cues the
    engine's events sound, and a level that exists **only during a trial's feedback window** — which is
    the one thing that makes this session shape unlike every other on this screen's shelf.
    """

    TIMING = TIMING
    UNITS = UNITS
    DIRECTION_WORD = {POSITIVE: "up", NEGATIVE: "down"}
    DEFAULT_DIRECTION = "random"
    PHASE_WORD = TrialTraining.PHASE_WORD | {ACTIVE: "feedback", RESULT: "verdict"}
    RESTART_PLACE = "the training panel"
    trial_subject = "the slow potential"
    trial_waiting = "no readings yet"
    SCALE_MODES = (AUTOMATIC, FIXED)
    """**full at** and **succeeds at** start on *automatic*, from the chance spread this session measures."""
    VOID_ACTIONS = {
        LIMIT: "examine the electrode, or wait until its curve in LEVEL is between the limit lines",
        THIN_BASELINE: "keep still and look at the point in the baseline",
        REJECTED: "keep still and look at the point in the trial",
    }

    offers = ("sound", "shadow", "click", "music")
    """No alert: the condition here is a trial's own baseline, which the signal crosses constantly by
    construction, so an alert would fire on the ordinary behaviour of the signal rather than on an
    exception. Music is offered, as on Pupil and HEG, whose trials run under it."""

    music_sources = ("tone", "crossfade", "noise", "shepard", "file")
    """Every source but `eeg`, for the slow-wave screen's reason: it sonifies a fast band, and a slow
    potential has no rhythm in it for an ear to follow. `crossfade` has its two states here: the shift the
    cued way, and the shift the other way."""

    state_ends = ("other way", "cued way")

    combines = ("weighted",)
    """One target, so every rule reduces to the same thing. Stated rather than offered."""

    MAX_TRIAL_S = BASELINE_S + MAX_FEEDBACK_S

    reject_modes = ("hold",)
    """No `clamp`: it sets a rejected sample to zero, and on a slow level that stands hundreds of µV from
    zero, a zero is a step of the whole standing offset into the level and the trial. `hold` gives the level
    the last clean value instead."""

    threshold_modes = ("fixed",)
    """The threshold is the trial's own baseline, measured fresh two seconds before each feedback window.
    There is nothing for a tracker to track that the protocol is not already re-measuring per trial."""

    scores_bands = False
    """The level is a metric and a scaling, not targets scored against thresholds — see `level_source`."""

    tunes_envelope = False
    """`envelope_seconds` reaches no estimator here: there is no envelope in this measurement."""

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0, heart_choice=None, heart_runner=None):
        super().__init__(protocol, path=path, device=device, length_seconds=length_seconds,
                         source=source, channels=channels)
        # Borrowed, not owned, as on Pupil and HEG: the heart beside the trials, when one is chosen.
        self.heart_choice = heart_choice
        self.heart = heart_runner
        if length_seconds is None:
            self.plan.set_length(SESSION_MINUTES)

        self.channel = next(iter(self.specs), 0)
        self.gain_uv = 100.0
        self.wave_uv = WAVE_GAINS[1]
        self.trace_seconds = TRACE_SECONDS

        self.traces = {i: TraceBuffer(320, TRACE_SECONDS, s.rate_hz) for i, s in self.specs.items()}
        self.rings = {i: RawRing(2.0, s.rate_hz, SPECTRUM_LO, SPECTRUM_HI, 1.0)
                      for i, s in self.specs.items()}

        self.raw_shift: float | None = None
        self._signal: float | None = None
        self.metric: SlowShift | None = None
        self.wave: WaveRing | None = None
        self.spread: PairSpread | None = None
        self._demean: tuple | None = None       # (trials scored, direction, phase) and the mean it gave
        self._marks: list[tuple[float, str]] = []
        # The extra channels: every opened channel but the training one has a level and a reading ring,
        # lit or not, so a chip press loses nothing. `_windows` holds each scored trial's signal times.
        self._extras: tuple[int, ...] = ()
        self._extra_metrics: dict[int, SlowShift] = {}
        self._extra_rings: dict[int, list] = {}
        self._extra_t: dict[int, float] = {}
        # Per channel, whatever its role: its newest raw block railed or flat, so no signal there. Never
        # rebound, so a chip press between `observe_raw` and `consume` finds the block it judged.
        self._block_open: dict[int, bool] = {}
        self._train_t: float | None = None
        self._windows: list[tuple[int, float, float, float]] = []
        self._checked: dict[tuple, float | None] = {}
        # The subtraction of eye movements: the last calibration, the channels its weights belong to, and
        # while it runs or subtracts, the aligner that holds each training block for its extras.
        self.eye_cal: EyeCalibration | None = None
        self.eye_fit: EyeFit | None = None
        self._eye_set: tuple | None = None
        self.subtracting = False
        self._eyes: EyeAligner | None = None
        self._subtractor: EyeSubtractor | None = None
        self._raw_training: tuple[int, np.ndarray] | None = None
        self._eye_bad: dict[int, int] = {}
        self.eye_version = 0
        self.levels = LevelHistory()
        """The standing level of every opened channel, for the level plot on the screen and in the
        calibration."""
        self.eye_length = eye_calibration.SHORT
        """The version the next calibration takes."""
        """Counts each change of the calibration or its use, for the card to rebuild on the loop."""
        # The run and what the operator sets about it, from the shared trial piece. Only active mode: the
        # protocol is the trials. A run starts with a third of its trials without a display, and has the
        # forty trials of the clinical block.
        self._init_trials("active")
        self.transfer_fraction = TRANSFER_FRACTION
        self.n_trials = DEFAULT_TRIALS

        # One target, for the three jobs `wave_view` lists: the shared pane draws a contribution row for
        # it, it is how that pane knows data is arriving, and it is what the click asks. Its value is the
        # trial's condition (`value_for`), met at `CONDITION_AT`, and `_settled` says there is nothing left
        # to calibrate.
        self.signal_target = Target(self.channel, None, "increase", threshold=CONDITION_AT,
                                    mode="fixed", name="shift", unit="")
        self.signal_target._settled = True
        self.targets = [self.signal_target]

        self.set_channel(self.channel)

    # --- what is measured ------------------------------------------------------------------

    def set_channel(self, channel: int) -> None:
        """Point the instrument at an electrode, rebuilding the metric, the plot ring and the run.

        The run goes with it, and that is not tidiness: a run half-measured on one electrode and half on
        another is not a run, and its history would be a rate over two different measurements.
        """
        if channel in self.specs:
            if channel != self.channel:
                self._drop_eyes()
            self.channel = channel
            self.signal_target.source = channel
        spec = self.specs.get(self.channel)
        # The old training channel becomes an unlit chip, and a lit chip that is now trained goes out.
        # Each of these is replaced whole, never changed in place: the feed thread reads each one once
        # per block, so a press on the loop cannot pull an entry out from under it.
        self._extra_metrics = {c: SlowShift(s, output_hz=METRIC_HZ)
                               for c, s in self.specs.items() if c != self.channel}
        self._extras = tuple(c for c in self._extras if c in self._extra_metrics)
        self._extra_t = {}
        self._train_t = None
        self._new_level()

    def _new_level(self) -> None:
        """The level, the chance spread, the plot ring and the run, all new: the signal they measure has
        changed, by a channel or by the subtraction."""
        spec = self.specs.get(self.channel)
        self.metric = None if spec is None else SlowShift(spec, output_hz=METRIC_HZ)
        # The chance spread is of this signal, so it starts again with it.
        self.spread = (None if spec is None else
                       PairSpread(BASELINE_S, self.active_s, hz=true_metric_hz(spec.rate_hz)))
        self.wave = (None if spec is None else
                     WaveRing(WAVE_CENTRE_HZ, WAVE_HIGH_HZ, spec.rate_hz, (self.channel,),
                              cycles=WAVE_CYCLES))
        self.raw_shift = None
        self.amplitudes.pop(("scp",), None)
        self.restart_trials()

    def _trial_length_changed(self) -> None:
        """The noise is the chance spread of a trial of this length, so it is measured again from here."""
        spec = self.specs.get(self.channel)
        self.spread = (None if spec is None else
                       PairSpread(BASELINE_S, self.active_s, hz=true_metric_hz(spec.rate_hz)))

    def restart_trials(self) -> None:
        """A new run: the extra channels' readings and the trial windows go with the old results. The marks
        stay, with one at the start of each run, so the trial numbers of two runs are told apart."""
        self._windows = []
        self._extra_rings = {c: [] for c in self._extra_metrics}
        self._checked = {}
        super().restart_trials()
        self._mark(self._train_t or 0.0, f"run starts · {self.n_trials} trials")

    # --- the extra channels ------------------------------------------------------------------

    @property
    def extras(self) -> tuple[int, ...]:
        """The lit extra channels, in order."""
        return self._extras

    def set_extra(self, channel: int, lit: bool) -> None:
        """Light or unlight an extra channel. It changes only what the end of the run reads, so the run goes
        on. A third one, with two lit, is inert, and the card says why."""
        if channel not in self._extra_metrics:
            return
        others = [c for c in self._extras if c != channel]
        if lit and len(others) >= MAX_EXTRAS:
            return
        self._extras = tuple(sorted(others + [channel] if lit else others))
        if self._eye_set is not None and self._eye_set[1] != self._extras:
            # The calibration belongs to its set of channels; the level changes only where it subtracted.
            was = self.subtracting
            self._drop_eyes()
            if was:
                self._new_level()

    # --- the subtraction of eye movements ---------------------------------------------------------

    def start_calibration(self) -> None:
        """Start the led calibration on the lit extra channels. The old weights go and the run stops, so
        the calibration sees the signals as they are; a new run starts after it."""
        spec = self.specs.get(self.channel)
        if not self._extras or spec is None:
            return
        self._drop_eyes()
        self._new_level()
        self._go_quiet()
        self.eye_cal = EyeCalibration(spec.rate_hz, self.channel, self._extras,
                                      eye_calibration.PLANS[self.eye_length])
        # A channel outside its range waits with the level plot, not with the dot.
        for c in (self.channel, *self._extras):
            if self.levels.touched(c, self.LIMIT_CLEAR_SECONDS):
                self.eye_cal.hold(f"ch{c} is at the input limit")
                break
        self._eyes = EyeAligner(self._extras, spec.rate_hz)

    def stop_calibration(self, reason: str = "stopped") -> None:
        cal = self.eye_cal
        if cal is not None and cal.running:
            cal.stop(reason)
            self._calibration_ended()

    def continue_calibration(self) -> None:
        """Start, or go on from the start of the part that was held. Never waits for a channel."""
        if self.eye_cal is not None:
            self.eye_cal.resume()

    def close_calibration(self) -> None:
        """The overlay goes: the result or the reason has been read."""
        self.stop_calibration()
        self.eye_cal = None
        self.eye_version += 1

    def _calibration_ended(self) -> None:
        cal = self.eye_cal
        spec = self.specs.get(self.channel)
        self._eyes = None
        if cal is not None and cal.done and spec is not None:
            self.eye_fit = cal.result
            self._eye_set = (cal.training, cal.extras)
            if cal.result.weights.subtracts:
                self.subtracting = True
                self._subtractor = EyeSubtractor(cal.result.weights)
                self._eyes = EyeAligner(cal.extras, spec.rate_hz)
        self._new_level()
        self.eye_version += 1

    def set_eye_length(self, length: str) -> None:
        """Choose the version of the next calibration. It starts nothing and changes no calibration."""
        if length in eye_calibration.PLANS:
            self.eye_length = length

    @property
    def can_subtract(self) -> bool:
        """Whether the last calibration found something to subtract, so the pill acts."""
        return self.eye_fit is not None and self.eye_fit.weights.subtracts

    def set_subtract(self, on: bool) -> None:
        """Switch the subtraction off or on. The level changes, so the run starts again."""
        spec = self.specs.get(self.channel)
        if not self.can_subtract or spec is None or on == self.subtracting:
            return
        self.subtracting = on
        self._subtractor = EyeSubtractor(self.eye_fit.weights) if on else None
        self._eyes = EyeAligner(self._eye_set[1], spec.rate_hz) if on else None
        self._new_level()
        self.eye_version += 1

    def _drop_eyes(self) -> None:
        """No calibration and no subtraction: a change of channels makes the weights belong to nothing."""
        cal = self.eye_cal
        if cal is not None and cal.running:
            cal.stop("the channels change")
        if self.eye_fit is not None or self._eyes is not None or cal is not None:
            self.eye_version += 1
        self.eye_cal = None
        self.eye_fit = None
        self._eye_set = None
        self.subtracting = False
        self._eyes = None
        self._subtractor = None

    def _trial_cued(self) -> None:
        self._eye_bad = {}

    def void_words(self, trial) -> tuple[str, str]:
        """A trial lost to an extra channel names the channel that lost most of its readings."""
        if trial.cause != EXTRA:
            return super().void_words(trial)
        bad = self._eye_bad
        channels = self._eye_set[1] if self._eye_set else ()
        name = (f"ch{max(bad, key=bad.get)}" if bad else
                " or ".join(f"ch{c}" for c in channels) or "an extra channel")
        return (f"no result · {name} has no valid signal for the subtraction in this trial",
                f"examine the electrode of {name}, or press subtract to stop the subtraction")

    def extra_silent(self, channel: int) -> bool:
        """Whether this channel sends nothing while the training channel does: no block yet, or none
        within `JUMP_S` of the training channel's newest."""
        if self._train_t is None:
            return False
        last = self._extra_t.get(channel)
        return last is None or self._train_t - last > JUMP_S

    def extra_checks(self) -> dict[int, list]:
        """For each lit extra channel, one score per trial of the run: its mean shift in the trial's active
        window against its own baseline, or None where it did not carry valid signal for that trial."""
        run = self.trials
        windows = {i: w for i, *w in list(self._windows)}
        out = {}
        for ch in self._extras:
            out[ch] = [self._extra_score(run, ch, i, windows.get(i)) for i in range(len(run.results))]
        return out

    def _extra_score(self, run, channel: int, index: int, window, final: bool = False) -> float | None:
        """One trial's score on one extra channel, by the trial engine's own rules (`too_many_invalid`,
        `thin_baseline`) over the readings its windows expect.
        `final` keeps the score as it is now, because the readings it rests on are about to go."""
        if window is None:
            return None
        key = (id(run), channel, index)
        if key in self._checked:
            return self._checked[key]
        ring = self._extra_rings.get(channel, [])
        b0, a0, a1 = window
        ready = final or run.done or self.extra_silent(channel) or (ring and ring[-1][0] >= a1)
        spec = self.specs.get(channel)
        if spec is None:
            return None
        hz = true_metric_hz(spec.rate_hz)
        lo, mid, hi = (bisect_right(ring, edge, key=lambda e: e[0]) for edge in (b0, a0, a1))
        base = [v for _t, v, ok, _why in ring[lo:mid] if ok]
        active = [v for _t, v, ok, _why in ring[mid:hi] if ok]
        expected = round((a1 - b0) * hz)
        enough = (not too_many_invalid(expected - len(base) - len(active), expected)
                  and not thin_baseline(len(base), round((a0 - b0) * hz)) and active)
        score = sum(active) / len(active) - sum(base) / len(base) if enough else None
        if ready:
            self._checked[key] = score
        return score

    def extra_status(self, channel: int) -> tuple[str, bool]:
        """The status line of a lit extra channel, and whether it reports a problem."""
        name = f"ch{channel}"
        if self.at_input_limit(channel):
            return f"{name} · at the input limit · no check now", True
        if self._train_t is None:
            return f"{name} · no readings yet", False
        if self.state.get(channel) is False or self._block_open.get(channel) or self.extra_silent(channel):
            return f"{name} · no signal · no check now", True
        run = self.trials
        scores = self.extra_checks().get(channel, [])
        windowed = [(i, score) for i, (trial, score) in enumerate(zip(run.results, scores)) if trial.valid]
        if not windowed:
            return f"{name} · no trial yet", False
        checked = sum(1 for _i, score in windowed if score is not None)
        if checked == len(windowed):
            return f"{name} · checked in {checked} of {len(windowed)} trials", False
        # Why the others checked nothing, by the cause that took most of their readings, so a channel that
        # spent the run at the input limit says so after it has left it.
        windows = {i: w for i, *w in list(self._windows)}
        why = [self._extra_miss(channel, windows.get(i)) for i, score in windowed if score is None]
        cause = max(EXTRA_MISS, key=why.count)
        missed = why.count(cause)
        return (f"{name} · checked in {checked} of {len(windowed)} · {missed} {EXTRA_MISS[cause]}",
                missed > checked)

    def _extra_miss(self, channel: int, window) -> str:
        """Why an extra channel checked nothing in a trial: the cause of most of its invalid readings in
        the trial's windows, or no signal where none arrived."""
        if window is None:
            return NO_SIGNAL
        ring = self._extra_rings.get(channel, [])
        b0, _a0, a1 = window
        lo, hi = (bisect_right(ring, edge, key=lambda e: e[0]) for edge in (b0, a1))
        why = [w for _t, _v, ok, w in ring[lo:hi] if not ok and w is not None]
        return max(EXTRA_MISS, key=why.count) if why else NO_SIGNAL

    @property
    def trial_metric_hz(self) -> float:
        """Every reading of the metric reaches the engine, not a thinned stream."""
        return true_metric_hz(self.rate)

    def set_mode(self, mode: str) -> None:
        """One mode only: the protocol is the trials."""

    def toggle_pause(self) -> None:
        super().toggle_pause()
        if self.paused:
            self.cut_trial(PAUSED)
            self.stop_calibration(eye_calibration.PAUSED)

    def trial_hold(self) -> str | None:
        """Why the run cannot start: no channel, or no level yet, which exists from the first accepted
        sample. It never waits for the filter: the level is primed there, and every readout is provisional
        from that reading on."""
        if self.metric is None:
            return "no channel to measure"
        if math.isfinite(self.metric.level):
            return None
        return "no valid reading yet" if self._seen.get(self.channel) else self.trial_waiting

    def _go_quiet(self) -> None:
        """No level, and the outputs handed back: without that the sound and the shadow hold their last
        value through the whole rest interval, which says "you are still doing it" while nobody is."""
        self._signal = None
        self._driving = False
        self._outputs_off()

    def _move_level(self, signal: float, t: float, smoothed: bool = True) -> None:
        """The trial's level, oriented toward the cue by the trial piece. Stepwise feedback is already in
        steps, so it skips the session's smoother: the outputs jump with the circle instead of gliding."""
        self._signal = signal
        if not smoothed:
            self._level = signal

    @property
    def state_balance(self) -> float | None:
        """Where a crossfade sits: the oriented level, already 0…1, and nothing between feedback windows."""
        return self._signal

    @property
    def range_uv(self) -> float:
        """**full at** as the readers take it: the trial's record inside a trial, the live value outside."""
        return self.scale_in_use()[0]

    def set_range(self, half_range_uv: float) -> None:
        """A typed **full at**, which sets *fixed*."""
        self.set_full_at(half_range_uv, typed=True)

    def chance_spread(self) -> tuple[float, float] | None:
        """The chance spread of one trial on this channel, and the seconds of signal behind it."""
        spread = self.spread
        value = None if spread is None else spread.value
        return None if value is None else (value, spread.seconds)

    @property
    def colour(self) -> str:
        """The slow signal's own colour, shared by the trace, the swatch and the card's stripe — the
        repo's rule that a colour follows the frequency rather than a list position."""
        return t.band_colour(0.0, WAVE_HIGH_HZ)

    @property
    def trained_floor_hz(self) -> float:
        """Zero, and this screen is where that is most literally true — a slow potential *is* the DC shift,
        the metric carries no high-pass at all, and the device's corner is the only thing under it. See
        `TrainingSession.trained_floor_hz` for what the artifact panel does with it."""
        return 0.0

    # --- where the level comes from ---------------------------------------------------------

    @property
    def signal(self) -> float | None:
        """The oriented, normalised shift, or None whenever there is no feedback window running.
        **Not zero** — zero is a reading, and a mechanism handed it drives itself to its floor."""
        return self._signal

    @property
    def level_source(self) -> tuple[str, str]:
        return (f"the mean shift against this trial's own baseline, toward the cued direction · "
                f"±{self.range_uv:.{UNITS.decimals}f} µV",
                "half = the trial's baseline itself")

    @property
    def raw_level(self) -> float:
        return self._signal if self._signal is not None else 0.0

    @property
    def active_targets(self) -> list[Target]:
        if self._signal is None:
            return []
        return [x for x in self.targets if self.target_is_live(x)]

    def channel_targets(self, channel: str) -> list[Target]:
        return self.active_targets

    def key_for(self, tgt: Target):
        return ("scp",)

    def value_for(self, tgt: Target) -> float:
        """The condition the click and the music read: past its middle only for a shift past the trial's
        **succeeds at** (`condition_level`), which the clipped level cannot say for a limit past **full at**."""
        return self.condition_for(self._signal) if self._signal is not None else CONDITION_AT

    def latency_for(self, tgt: Target) -> float | None:
        return self.metric.latency_seconds if self.metric is not None else None

    # --- feed -------------------------------------------------------------------------------

    def observe_raw(self, block, judged) -> None:
        ch = block.channel
        # Judged on the samples as they arrived: a held block's stand-in is flat by construction.
        self._block_open[ch] = self._looks_open(block)
        values = np.asarray(block.values)
        self.levels.push(block)
        eyes = self._eyes
        if eyes is not None:
            if ch == self.channel:
                self._raw_training = (block.start_index, values)
            else:
                eyes.push_extra(ch, block.start_index, values, self._eye_code(block))
        if ch in self.traces:
            self.traces[ch].push(self.for_display(ch, values), block.valid, judged.accepted)
        if ch in self.rings:
            self.rings[ch].push(values)

    def consume(self, block) -> set:
        """Feed the metric, then the engine, the chance spread and the level, in that order.

        **Every reading reaches the engine, from the first one.** The level is primed at the first accepted
        block, so there is no transient to keep out of a baseline. **Not while the session is paused**:
        trials freeze with the transport, and on resume the engine's time-jump rule voids the trial in
        flight, however short the pause, since its baseline describes signal that no longer precedes its
        feedback window.

        Invalid readings *do* reach the engine, which counts them toward voiding the trial: a held block,
        a channel resolved dead and a block with a sample at the input limit, the last two named as the
        cause.
        """
        extra = self._extra_metrics.get(block.channel)
        if extra is not None:
            self._feed_extra(block, extra)
            return set()                # an extra channel never drives, never observes, never waits
        metric = self.metric
        if metric is None or block.channel != metric.spec.index:
            return set()
        # A held stand-in keeps a valid device state, so mark a rejected block invalid for the trial
        # engine, or a run of blinks scores as a valid baseline and shift.
        # A channel resolved dead takes `pass_through`, whose stand-in is accepted but is no sample: the
        # level holds over it, as over a held block. Resolved state (the operator override), not the raw
        # device flag, which ignores the override.
        usable = self.state.get(block.channel) is not False
        accepted = self._accepted.get(block.channel, True) and usable
        railed = self._railed.get(block.channel, False)
        eyes = self._eyes
        if eyes is None:
            return self._train(block, accepted, railed, usable)
        # The raw samples, not the gate's stand-in: the calibration needs the blinks the gate holds.
        raw = self._raw_training
        values = raw[1] if raw is not None and raw[0] == block.start_index else np.asarray(block.values)
        open_ = self._block_open.get(block.channel, False)
        eyes.push_training(block, (accepted, railed, usable and not open_, values))
        fresh: set = set()
        for held, (ok, limit, live, y), xs, codes in eyes.ready():
            fresh |= self._eye_block(held, ok, limit, live, y, xs, codes)
        return fresh

    def _eye_code(self, block) -> int:
        """What an extra block is to the subtraction: usable, at the input limit, or with no signal."""
        mask = input_limit_mask(block)
        if mask is not None and mask.any():
            return eye_calibration.LIMIT
        if self.state.get(block.channel) is False or self._block_open.get(block.channel):
            return eye_calibration.NO_SIGNAL
        return eye_calibration.OK

    def _eye_block(self, block, accepted: bool, railed: bool, usable: bool, y: np.ndarray, xs: np.ndarray,
                   codes: np.ndarray) -> set:
        """One training block with its extras: into the calibration while it runs, else corrected."""
        cal = self.eye_cal
        extras = self._eyes.extras if self._eyes is not None else ()
        if cal is not None and cal.running:
            # No signal holds the calibration; a sample at the input limit is the calibration's to judge, per
            # sample, because in the blinks part it costs only that blink.
            problem = None
            dead = [k for k in range(len(extras)) if (codes[k] == eye_calibration.NO_SIGNAL).any()]
            if not usable:
                problem = f"ch{block.channel} has no signal"
            elif dead:
                problem = f"ch{extras[dead[0]]} has no signal"
            lim_uv = limit_uv(block.spec.input_range)
            own = np.abs(y) >= lim_uv if lim_uv else np.zeros(len(y), dtype=bool)
            at = (codes == eye_calibration.LIMIT)
            limit = own | at.any(axis=0)
            named = block.channel if own.any() else next((extras[k] for k in range(len(extras)) if at[k].any()),
                                                          block.channel)
            for _ in range(cal.feed(block.start_index, y, xs, problem, limit=limit,
                                    limit_reason=f"ch{named} is at the input limit")):
                self.cue(cues.tone_at(TONE_HZ), CUE_VOLUME)
            if not cal.running:
                self._calibration_ended()
            return self._train(block, accepted, railed, usable, trial=False)
        sub = self._subtractor
        if sub is None:
            return self._train(block, accepted, railed, usable)
        if block.seam:
            sub.reset()
        corrected, ok = sub.process(y, xs, codes.max(axis=0) == eye_calibration.OK)
        extra_bad = not ok.all()
        for k, c in enumerate(extras):
            lost = int(np.count_nonzero(codes[k]))
            if lost:
                self._eye_bad[c] = self._eye_bad.get(c, 0) + lost
        corrected_block = dataclasses.replace(block, samples=tuple((corrected / block.spec.scale).tolist()))
        return self._train(corrected_block, accepted, railed, usable, extra_bad=extra_bad)

    def _train(self, block, accepted: bool, railed: bool, usable: bool, trial: bool = True,
               extra_bad: bool = False) -> set:
        """The training block into the level, then the engine, the chance spread and the outputs. With
        `trial` False, as during the calibration, only the level."""
        metric = self.metric
        if metric is None:
            return set()
        self._train_t = block.end_index / block.spec.rate_hz
        fresh: set = set()
        readings = metric.feed(block, accepted=accepted and not extra_bad)
        if math.isfinite(metric.level):
            self.raw_shift = metric.level
            self.amplitudes[("scp",)] = metric.level
        else:                           # no level yet, or none since a seam or a step back
            self.raw_shift = None
            self.amplitudes.pop(("scp",), None)
        # The level starts again at a seam, so the trial in flight and the spread's stretch end at the
        # first reading after it, which can come in a later block.
        if block.seam:
            self.cut_trial(SEAM)
        for r in readings:
            finite = math.isfinite(r.value)
            if self.paused or not trial:
                continue
            cut = self.trial_cut is not None
            ok = accepted and finite and not extra_bad
            run = self.trials
            before = (run.phase, run.required, len(run.results))
            # The trial piece orients the level toward the cue, sounds the cue and the verdict (a success
            # is the cue tone doubled; a failure and a void trial are silent), and ends the run. Its end
            # stops the clock, as `tick()` does for a timed plan: the run *is* the session's length here.
            cause = (None if ok else LIMIT if railed else NO_SIGNAL if not usable else EXTRA if extra_bad
                     else None)
            self._observe_trial(r.t, r.value if finite else None, ok, cause=cause)
            if run is self.trials:
                self._follow_trial(run, before, r.t)
            if self.spread is not None:
                self.spread.feed(r.t, r.value - self._own_shift(run, before) if finite else r.value, ok,
                                 fresh=cut)
            if self._driving:
                fresh.add(("scp",))
        wave = self.wave
        if wave is not None and metric.wave_start is not None and len(metric.waveform):
            wave.feed(metric.wave_start, {block.channel: metric.waveform})
        return fresh

    def _feed_extra(self, block, metric: SlowShift) -> None:
        """An extra channel's readings into its ring, for the check at the end of the run. A reading is
        valid when its block was accepted, the channel is not resolved dead, its block is not railed or
        flat, and it is not at the input limit. A flat line is no signal even where the device calls it
        valid, and must not vouch for a trial. Nothing here reaches the trial, the level or an output."""
        channel = block.channel
        self._extra_t[channel] = block.end_index / block.spec.rate_hz
        usable = self.state.get(channel) is not False and not self._block_open.get(channel, False)
        accepted = self._accepted.get(channel, True) and usable
        readings = metric.feed(block, accepted=accepted)
        if self.paused:
            return
        limit = self.at_input_limit(channel)
        ring = self._extra_rings.setdefault(channel, [])
        why = LIMIT if limit else NO_SIGNAL if not usable else REJECTED if not accepted else None
        for r in readings:
            ok = accepted and not limit and math.isfinite(r.value)
            if ring and r.t < ring[-1][0]:
                # A step back in time: what came before no longer precedes these readings. The windows
                # of the trials before it are scored on those readings now, or a later one with the same
                # times would score them on signal from after the step.
                run = self.trials
                for index, *window in list(self._windows):
                    self._extra_score(run, channel, index, tuple(window), final=True)
                ring = self._extra_rings[channel] = []
            ring.append((r.t, r.value if ok else math.nan, ok, None if ok else why))

    def _own_shift(self, run, before: tuple) -> float:
        """What the trainee's own shift adds to a reading of a feedback window or its verdict: the mean raw
        score of the earlier valid trials in the same direction, 0 with none. Taken off the reading the
        chance spread is fed, so the spread measures chance and not the skill it judges."""
        phase, required, scored = before
        if phase not in (ACTIVE, RESULT) or required is None:
            return 0.0
        key = (scored, required, phase)
        if self._demean is None or self._demean[0] != key:
            earlier = run.results[:scored - 1] if phase == RESULT else run.results[:scored]
            scores = [r.score_uv for r in earlier if r.valid and r.required == required]
            self._demean = (key, sum(scores) / len(scores) if scores else 0.0)
        return self._demean[1]

    def _follow_trial(self, run, before: tuple, t: float) -> None:
        """The windows of each scored trial, and a mark at its cue, its feedback start and its verdict. The
        times are the engine's own phase edges, which can fall on a lost reading inside a hole."""
        phase, _required, scored = before
        if run.phase == phase and len(run.results) == scored:
            return
        word = self.DIRECTION_WORD.get(run.required or "", "")
        edges = run.phase_t
        if run.phase in (CUED, BASELINE) and phase not in (CUED, BASELINE):
            self._mark(edges.get(run.phase, t), f"trial {run.trial_number} · cue · {word}"
                       + (" · without feedback" if run.transfer else ""))
        if run.phase == ACTIVE and phase == BASELINE:
            self._mark(edges.get(ACTIVE, t), f"trial {run.trial_number} · feedback starts · {word}")
        if len(run.results) > scored:
            last = run.results[-1]
            if phase == ACTIVE and run.phase == RESULT and {BASELINE, ACTIVE, RESULT} <= edges.keys():
                self._windows.append((len(run.results) - 1, edges[BASELINE], edges[ACTIVE], edges[RESULT]))
            verdict = (f"{'success' if last.success else 'failure'} · {last.score_uv:+.2f} µV" if last.valid
                       else f"no result · {last.cause}")
            self._mark(edges.get(RESULT, t), f"trial {last.index} · verdict · {verdict}")

    def _mark(self, t: float, note: str) -> None:
        spec = self.specs.get(self.channel)
        index = round(t * spec.rate_hz) if spec is not None else 0
        self._marks.append((time.time(), f"{note} · ch{self.channel} sample {index}"))

    def marks(self) -> list:
        """The trial events, each with the sample index of the training channel in this session's stream
        at which it fell. A recording started later, or a rearm, numbers its samples from another origin,
        so the wall time is what places a mark in a recording."""
        return list(self._marks)

    # --- surviving a stop -------------------------------------------------------------------

    def adopt_view(self, previous) -> None:
        super().adopt_view(previous)
        self.wave_uv = getattr(previous, "wave_uv", self.wave_uv)
        if isinstance(previous, TrialTraining):
            self.adopt_trials(previous)

    def adopt_targets(self, previous) -> None:
        """What an operator set is which electrode and how the run is shaped, and both are carried by
        *rebuilding* — a metric assigned as an object would have no filters behind it, and a run assigned
        as an object would bring the previous session's results with it. The results are the session."""
        self.set_channel(getattr(previous, "channel", self.channel))
        # A lit extra channel is a setting; one the new channel count does not open is dropped.
        self._extras = tuple(c for c in getattr(previous, "extras", ()) if c in self._extra_metrics)

    # --- what this protocol requires --------------------------------------------------------

    def wiring_problem(self) -> str | None:
        """Whether the amplifier's corner leaves enough of a shift to train.

        The metric has no high-pass of its own, so the device's corner is the only thing under the band —
        and what it costs is not a band edge but a *fraction of the mean*, since a mean over the feedback
        window is exactly what a trial is scored on. See `nfb.dsp.scp.shift_survival`.
        """
        if self.metric is None:
            return "no channel to measure"
        surviving = shift_survival(self.corner_hz, self.active_s)
        if surviving < SURVIVAL_FLOOR:
            return (f"the device's {self.corner_hz:g} Hz corner keeps {surviving:.0%} of a shift's "
                    f"mean over {self.active_s:g} s — set High-pass to {BEST_CORNER_HZ:g} Hz on the "
                    f"device screen")
        return None


def session_for(*args, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None,
                **kwargs) -> SCPSession:
    """A heart, when one is chosen on the device screen, is shown beside the trials."""
    _ = mendi_choice, mendi_runner, pupil_choice, pupil_runner
    return SCPSession(*args, heart_choice=heart_choice, heart_runner=heart_runner, **kwargs)


def _duration(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 90 else f"{seconds:.0f} s"


# --- the card ----------------------------------------------------------------------------------

class SCPCard:
    """The one signal, as a card, in the grammar `target_card.TargetCard` gives every screen: an identity
    row, the value beside what it is compared against, a bar that states its own range, the row where a
    threshold's mode would be, and a note that appears only when there is something to say.

    Not that class, for the reason the waveform screen's card is not either: the value is **signed**, so
    the bar is centre-zero and fills both ways where the shared one runs from zero to a ceiling. What is in
    the mode row here is the run's shape, since the threshold is a per-trial baseline and there is no mode
    to choose.
    """

    def __init__(self, page: ft.Page, session: SCPSession, on_change: Callable[[], None]):
        self.page = page
        self.session = session
        self.on_change = on_change

        self.identity = ft.Row(spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.value_text = ft.Text("—", size=19, color=t.FAINT)
        self.unit_text = ft.Text("µV", size=t.SMALL, color=t.FAINT)
        self.level_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.fill = ft.Container(width=0, height=8, border_radius=4, bgcolor=session.colour,
                                 left=BAR_W / 2, bottom=0)
        self.notch = ft.Container(width=2, height=14, bgcolor=t.TEXT, left=BAR_W / 2 - 1, bottom=-3)
        bar = ft.Stack([ft.Container(width=BAR_W, height=8, border_radius=4, bgcolor=t.SURFACE_HI,
                                     left=0, bottom=0),
                        self.fill, self.notch], width=BAR_W, height=14)
        self.scale_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.filter_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.spread_text = ft.Text("—", size=t.SMALL, color=t.MUTED, no_wrap=True)
        self.spread_row = labelled("noise", self.spread_text, width=RUN_LABEL_W)
        self.run_rows = ft.Column(spacing=9)
        self._scale_boxes: dict = {}
        # The extra channels: a chip for each other opened channel, and a line for each lit one. A line
        # shows from the press that lights its chip to the press that puts it out, whatever it says in
        # between, so a status never moves the card. None where only one channel is open.
        self.chips = ft.Row(spacing=6)
        self.extra_lines = [ft.Text("", size=t.SMALL, color=t.FAINT, no_wrap=True, max_lines=1)
                            for _ in range(MAX_EXTRAS + (len(session.specs) > MAX_EXTRAS + 1))]
        self.extra_boxes = [ft.Container(content=line, height=NOTE_H, visible=False)
                            for line in self.extra_lines]
        # The calibration's button and the subtraction's pill, under the lines of the lit chips.
        self.eye_row = ft.Column(spacing=6, visible=False)
        self._eye_key: tuple | None = None
        extras: list[ft.Control] = []
        if len(session.specs) > 1:
            extras = [labelled("extra channels", self.chips, width=RUN_LABEL_W), ft.Container(height=4),
                      *self.extra_boxes, self.eye_row, ft.Container(height=5)]
        self.note = ft.Text("", size=t.SMALL, color=t.WARN)
        # Two lines reserved whether or not there is a note — this card's notes run to two lines, and a
        # card whose height follows its note moves everything under it. See `target_card.NOTE_H`.
        self.note_box = ft.Container(content=self.note, height=2 * NOTE_H,
                                     alignment=ft.Alignment.BOTTOM_LEFT)

        self.control = ft.Container(
            content=ft.Column([
                self.identity,
                ft.Container(height=8),
                ft.Row([self.value_text, ft.Container(width=6), self.unit_text,
                        ft.Container(expand=True),
                        ft.Text("baseline", size=t.SMALL, color=t.FAINT),
                        ft.Text("this trial's", size=t.BODY, color=t.TEXT)],
                       spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=9),
                ft.Row([bar, ft.Container(width=12), self.level_text], spacing=0,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=4), self.scale_text,
                ft.Container(height=4), self.filter_text,
                ft.Container(height=10), self.spread_row,
                ft.Container(height=9), *extras, self.run_rows,
                self.note_box,
            ], spacing=0),
            bgcolor=t.BG, border=ft.Border.all(1, t.HAIRLINE), border_radius=8,
            padding=ft.Padding(12, 11, 12, 11))
        self.rebuild()

    def rebuild(self) -> None:
        """The rows that change when what is measured, or the shape of the run, changes.

        The channel is a menu rather than a row of pills for the reason the waveform card gives: this is a
        380 px column, and a control that is present and off the right-hand edge is worse than absent.
        """
        session = self.session
        self.identity.controls = [
            ft.Container(width=3, height=18, bgcolor=session.colour, border_radius=2),
            ft.PopupMenuButton(
                content=ft.Container(
                    content=ft.Row([ft.Text(f"ch{session.channel}", size=t.BODY, color=t.TEXT),
                                    ft.Text("▾", size=9, color=t.FAINT)], spacing=6),
                    padding=ft.Padding(9, 5, 9, 5), border_radius=6,
                    bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
                items=[ft.PopupMenuItem(
                    content=ft.Text(f"ch{c}", size=t.BODY),
                    on_click=(lambda c_=c: lambda _e: self._set_channel(c_))())
                    for c in session.specs]),
            ft.Container(expand=True),
            help_button(self.page, "What is measured", signal_help(session)),
        ]
        # A lit chip is pressed to unlight it; an unlit one, with two lit, is inert and the line under the
        # chips says why. The training channel has no chip.
        full = len(session.extras) >= MAX_EXTRAS
        self.chips.controls = [
            pill(f"ch{c}", c in session.extras,
                 (lambda c_=c: lambda _e: self._set_extra(c_, c_ not in self.session.extras))(),
                 enabled=c in session.extras or not full)
            for c in session.specs if c != session.channel]
        # Two rows, because the button, the two pills and `subtract` do not fit the card's width in one.
        rows: list[list[ft.Control]] = []
        if session.extras:
            rows.append([button("calibrate eyes", lambda _e: self._calibrate())]
                        + [pill(name, session.eye_length == name,
                                (lambda n=name: lambda _e: (session.set_eye_length(n), self.rebuild()))())
                           for name in eye_calibration.PLANS])
        if session.can_subtract:
            rows.append([pill("subtract", session.subtracting,
                              lambda _e: self._changed(lambda: session.set_subtract(not session.subtracting))())])
        self.eye_row.controls = [ft.Row([ft.Container(width=RUN_LABEL_W), *row], spacing=8,
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER) for row in rows]
        self.eye_row.visible = bool(rows)
        self._eye_key = (session.eye_version, session.extras)
        # Where a threshold's mode row goes on every other card. What is chosen here is the run's shape:
        # the shared trial rows, with the wait from the cue to the baseline, which only this screen sets.
        # The panel's `?` says that changing one starts the run again.
        self._scale_boxes = {}
        self.run_rows.controls = trial_settings_rows(
            session, self._changed, cue_gap=True,
            help=help_button(self.page, "The run", RUN_HELP), width=RUN_LABEL_W, boxes=self._scale_boxes)
        self.page.update()

    def _calibrate(self) -> None:
        self.session.start_calibration()
        self.rebuild()
        self.on_change()

    def _set_extra(self, channel: int, lit: bool) -> None:
        self.session.set_extra(channel, lit)
        self.rebuild()

    def _set_channel(self, channel: int) -> None:
        self.session.set_channel(channel)
        self.rebuild()
        self.on_change()

    def _changed(self, action) -> Callable:
        return lambda *a: (action(*a), self.rebuild(), self.on_change())

    def _refresh_extras(self) -> None:
        session, lit = self.session, self.session.extras
        lines = [session.extra_status(c) for c in lit]
        unlit = len(session.specs) - 1 - len(lit)
        if len(lit) >= MAX_EXTRAS and unlit:
            lines.append(("at most two extra channels", False))
        for i, (line, box) in enumerate(zip(self.extra_lines, self.extra_boxes)):
            text, problem = lines[i] if i < len(lines) else ("", False)
            line.value, line.color = text, t.WARN if problem else t.FAINT
            box.visible = bool(text)

    def refresh(self) -> None:
        session = self.session
        if (session.eye_version, session.extras) != self._eye_key:
            self.rebuild()
        run = session.trials
        # What the feedback shows, not the live shift: in a trial without a display, after the circle fades
        # and between steps, a live number here would be the feedback the trial withholds.
        shift = session.shown_shift if session.feedback_now else None
        # Inside a trial, the trial's record; outside one, the value the next trial takes.
        full = max(session.range_uv, MIN_RANGE_UV)
        show_scale(session, self._scale_boxes)
        measured = session.chance_spread()
        self.spread_text.value = ("—" if measured is None else
                                  f"±{measured[0]:.{UNITS.decimals}f} µV · from {measured[1]:.0f} s of signal")
        self._refresh_extras()
        if shift is None:
            self.value_text.value = "—"
            self.value_text.color = t.FAINT
            self.fill.width = 0
        else:
            self.value_text.value = f"{shift:+.1f}"
            self.value_text.color = t.TEXT
            half = BAR_W / 2
            span = half * min(1.0, abs(shift) / full)
            self.fill.width = span
            self.fill.left = half if shift >= 0 else half - span
        signal = session.signal
        self.level_text.value = "—" if signal is None else f"{signal:.0%}"
        level = session.raw_shift
        self.scale_text.value = (f"0% at −{full:.{UNITS.decimals}f} · 50% at the baseline · "
                                 f"100% at +{full:.{UNITS.decimals}f} µV, toward the cue")
        self.filter_text.value = ("" if session.metric is None else
                                  f"level {level:+.1f} µV · {session.metric.describe()}"
                                  if level is not None else session.metric.describe())
        # The input limit is not here: the level plot shows it, and a void trial says what to do.
        if session.metric is None:
            self.note.value = "no channel to measure"
        else:
            self.note.value = session.wiring_problem() or ""


# --- the calibration's overlay ----------------------------------------------------------------

DOT = 18
EDGE_GAP = 30
"""How far the dot stays from each edge of the window: near it, so an eye movement is as large as it can be."""
TABLE_W = (110, 90, 90)
HELD_W = 620
"""The level plot in the calibration: wide enough for minutes of a level, centred in the window."""


def held_line(cal: EyeCalibration, session: "SCPSession") -> str:
    """The one line under the level plot while the calibration waits for a channel. It follows the channels,
    so it says when they are inside their range again."""
    again = "press continue to do this part again" if cal.begun else "press start"
    if cal.held == eye_calibration.TOO_FEW:
        return f"{cal.held} · {again}"
    if "input limit" not in (cal.held or ""):
        return f"{cal.held} · examine the electrode, then {again}"
    outside = [c for c in (cal.training, *cal.extras)
               if session.levels.touched(c, session.LIMIT_CLEAR_SECONDS)]
    if not outside:
        return f"all channels are between the limit lines · {again}"
    go = "continue" if cal.begun else "start"
    return f"ch{outside[0]} is at the input limit · press {go} when its curve is between the limit lines"


class EyeOverlay:
    """The calibration over the whole window: the dot, one line under the centre and `stop`; then the
    result, or why it stopped, until the person closes it. While a channel it uses cannot be used, the
    level plot of its channels in place of the dot, with `start` (or `continue`) and `stop`."""

    def __init__(self, page: ft.Page, session: SCPSession):
        self.page = page
        self.session = session
        self.dot = ft.Container(width=DOT, height=DOT, border_radius=DOT / 2, bgcolor=t.TEXT, left=0, top=0)
        self.line = ft.Text("", size=t.BODY, color=t.MUTED, text_align=ft.TextAlign.CENTER)
        self.line_box = ft.Container(content=self.line, left=0, right=0, top=0, alignment=ft.Alignment.CENTER)
        self.stop_box = ft.Container(content=button("stop", lambda _e: self._close()), right=24, bottom=24)
        self.table = ft.Column(spacing=6, width=sum(TABLE_W))
        self.sentence = ft.Text("", size=t.BODY, color=t.TEXT, width=sum(TABLE_W), text_align=ft.TextAlign.CENTER)
        self.result = ft.Container(
            content=ft.Column([self.table, ft.Container(height=14), self.sentence, ft.Container(height=18),
                               button("close", lambda _e: self._close())],
                              spacing=0, tight=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            left=0, right=0, top=0, bottom=0, alignment=ft.Alignment.CENTER, visible=False)
        cal_channels = lambda: [] if session.eye_cal is None else [session.eye_cal.training,  # noqa: E731
                                                                     *session.eye_cal.extras]
        self.levels = LevelPlot(session.levels, cal_channels, lambda: HELD_W, height=220)
        self.held_line = ft.Text("", size=t.BODY, color=t.MUTED, text_align=ft.TextAlign.CENTER, width=HELD_W)
        self.go_text = ft.Text("start", size=t.SMALL, color=t.ACCENT)
        go = ft.Container(content=self.go_text, padding=ft.Padding(10, 5, 10, 5), border_radius=6,
                          border=ft.Border.all(1, t.BORDER), on_click=lambda _e: self._go())
        self.held = ft.Container(
            content=ft.Column([ft.Container(content=self.levels.control, width=HELD_W), ft.Container(height=16),
                               self.held_line, ft.Container(height=14),
                               ft.Row([go, button("stop", lambda _e: self._close())], spacing=12,
                                      alignment=ft.MainAxisAlignment.CENTER, width=HELD_W)],
                              spacing=0, tight=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            left=0, right=0, top=0, bottom=0, alignment=ft.Alignment.CENTER, visible=False)
        self.control = ft.Container(
            content=ft.Stack([self.dot, self.line_box, self.stop_box, self.result, self.held], expand=True),
            bgcolor=t.BG, expand=True, visible=False)

    def _go(self) -> None:
        self.session.continue_calibration()
        self.refresh()
        self.page.update()

    def _close(self) -> None:
        self.session.close_calibration()
        self.refresh()
        self.page.update()

    def _size(self) -> tuple[float, float]:
        return (getattr(self.page, "width", None) or 1020, getattr(self.page, "height", None) or 700)

    def refresh(self) -> None:
        cal = self.session.eye_cal
        self.control.visible = cal is not None
        if cal is None:
            return
        width, height = self._size()
        waiting = cal.running and cal.held is not None
        running = cal.running and not waiting
        self.dot.visible = self.line_box.visible = self.stop_box.visible = running
        self.held.visible = waiting
        self.result.visible = not cal.running
        if waiting:
            self.levels.refresh()
            self.held_line.value = held_line(cal, self.session)
            self.go_text.value = "continue" if cal.begun else "start"
            return
        if running:
            place = cal.dot[1]
            x = {"left": EDGE_GAP, "right": width - EDGE_GAP - DOT}.get(place, (width - DOT) / 2)
            y = {"up": EDGE_GAP, "down": height - EDGE_GAP - DOT}.get(place, (height - DOT) / 2)
            self.dot.left, self.dot.top = x, y
            self.line.value = cal.line
            self.line_box.top = height / 2 + DOT + 12
            return
        if cal.done:
            fit = cal.result
            header = ["", "without", "with"]
            rows = [[name, f"{fit.sizes[name][0]:.1f} µV", f"{fit.sizes[name][1]:.1f} µV"] for name in EYE_ROWS]
            self.table.controls = [
                ft.Row([ft.Text(cell, size=t.BODY if i else t.SMALL, width=w,
                                color=t.FAINT if r == 0 else t.TEXT if i else t.MUTED,
                                text_align=ft.TextAlign.RIGHT if i else ft.TextAlign.LEFT)
                        for i, (cell, w) in enumerate(zip(row, TABLE_W))], spacing=0)
                for r, row in enumerate([header, *rows])]
            self.sentence.value = eye_sentence(fit)
        else:
            self.table.controls = []
            self.sentence.value = f"the calibration stops · {cal.stopped} · nothing is kept"


# --- the screen --------------------------------------------------------------------------------

def view(page: ft.Page, session: SCPSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    """Build once, mutate on Flet's loop."""

    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width(reserve: int = SIDE_W) -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - reserve - 70)

    spectrum = spectrum_pane.SpectrumPane(page, session, plot_width)

    trends = trends_pane.TrendsPane(page, session, plot_width)

    # ---- the trial panel: the shared fixation circle, each phase its own picture ----
    trial_ui = TrialView(session, CIRCLE_SIDE, heart=session.heart_on)

    # ---- the history strip ----
    history_cells = ft.Row(spacing=3, wrap=True)
    history_text = ft.Text("", size=t.SMALL, color=t.MUTED)

    # ---- the slow signal ----
    wave_canvas = cv.Canvas(shapes=[], expand=True, height=WAVE_H)
    wave_axis, wave_labels = value_axis(WAVE_H, count=5, width=AXIS_W)
    wave_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    wave_state = ft.Text("", size=t.SMALL, color=t.WARN)
    wave_pills = ft.Row(spacing=6)
    wave_legend = ft.Row(spacing=12)

    traces = trace_rows.TraceRows(page, session, plot_width, height=TRACE_H)

    signal = signal_pane.SignalPane(page, session, inset=False)

    def after_change() -> None:
        signal.rebuild_contributions(session.targets)
        signal.rebuild()

    card = SCPCard(page, session, after_change)
    overlay = EyeOverlay(page, session)
    level = LevelPlot(session.levels, lambda: sorted(session.specs), lambda: SIDE_W - 34, height=110)

    def set_wave_gain(uv: float) -> None:
        session.wave_uv = uv
        build_wave_pills()

    def build_wave_pills() -> None:
        wave_pills.controls = [
            pill(f"±{g:g}", session.wave_uv == g, (lambda v=g: lambda _e: set_wave_gain(v))())
            for g in WAVE_GAINS]
        page.update()

    train_left = ft.Column([
        trial_ui.fixation_panel("TRIAL", help=help_button(page, "How a trial runs", TRIAL_HELP)),
        ft.Container(height=12),
        panel("HISTORY", history_cells, ft.Container(height=8), history_text,
              help=help_button(page, "The run so far", HISTORY_HELP)),
        ft.Container(height=12),
        panel("SHIFT",
              ft.Row([wave_axis, ft.Container(content=ft.Stack(
                  [ft.Container(content=wave_canvas, expand=True),
                   ft.Container(content=wave_state, alignment=ft.Alignment.CENTER, expand=True)],
                  expand=True, height=WAVE_H), expand=True, height=WAVE_H)], spacing=0),
              ft.Container(height=6),
              ft.Row([wave_legend, ft.Container(expand=True)]),
              ft.Container(height=4), wave_caption,
              trailing=wave_pills,
              help=help_button(page, "The slow signal", SHIFT_HELP)),
        ft.Container(height=12),
        panel("RAW", *traces.rows, ft.Container(height=6), traces.caption),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    # The heart, when a source is chosen on the device screen: how each direction's effort moves it.
    heart_ui = (heart_panel.HeartPanel(compact=True, width_of=lambda: max(200.0, SIDE_W - 60))
                if session.heart_on else None)
    heart_panels: list[ft.Control] = []
    if heart_ui is not None:
        heart_panels = [ft.Container(height=12), trial_ui.heart_stats_panel, ft.Container(height=12),
                        panel("HEART", *heart_ui.spectrum_rows(), ft.Container(height=10), heart_ui.tiles,
                              ft.Container(height=8), heart_ui.source_note)]

    train_right = ft.Column([
        # The corner warning is the card's note — it was also printed under the card, which was one
        # sentence twice in one panel.
        panel("TRAINING", card.control),
        ft.Container(height=12),
        panel("LEVEL", level.control, help=help_button(page, "The level of each input", LEVEL_HELP)),
        *heart_panels,
    ], spacing=0, width=SIDE_W, scroll=ft.ScrollMode.AUTO)

    train_tab = ft.Container(content=ft.Row([train_left, train_right], spacing=14, expand=True),
                             padding=ft.Padding(14, 12, 14, 14), expand=True)
    feedback_tab = ft.Container(
        content=ft.Column([signal.control], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO),
        padding=ft.Padding(14, 12, 14, 14), expand=True, visible=False)
    panes = [train_tab, spectrum.control, trends.control, feedback_tab, plan_pane_.control]

    def select_tab(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs(["Train", "Spectrum", "Trends", "Feedback", "Session"], select_tab)

    def draw_history() -> None:
        run = session.trials
        cells: list[ft.Control] = []
        for result in run.results:
            if not result.valid:
                colour, border = t.SURFACE_HI, t.BORDER
            elif result.success:
                colour, border = t.GOOD, t.GOOD
            else:
                colour, border = None, t.BAD
            cell = ft.Container(width=CELL_W, height=CELL_H, border_radius=3, bgcolor=colour,
                                border=ft.Border.all(1, border))
            # A fixed width, or each cell's column takes the whole row and the cells stack.
            cells.append(ft.Container(width=CELL_W, content=ft.Column(
                [cell, ft.Container(height=2),
                 ft.Container(width=5, height=5, border_radius=3,
                              bgcolor=t.MUTED if result.transfer else None)],
                spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER)))
        history_cells.controls = cells
        history_text.value = trial_summary(run, session.DIRECTION_WORD, session.UNITS.unit,
                                           session.UNITS.decimals, checks=session.extra_checks(),
                                           floor=session.UNITS.min_full_at)

    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        card.refresh()
        signal.refresh()
        spectrum.refresh()
        trends.refresh()
        trial_ui.draw_fixation("")
        overlay.refresh()
        level.refresh()
        draw_history()
        if heart_ui is not None:
            heart_ui.refresh(session.heart)
            trial_ui.draw_heart_stats()

        ring, width = session.wave, plot_width()
        full = max(WAVE_FLOOR_UV, session.wave_uv)
        colour = session.colour if session.raw_shift is not None else t.MUTED
        wave_legend.controls = [
            ft.Row([ft.Container(width=10, height=3, bgcolor=colour, border_radius=2),
                    ft.Text(f"ch{session.channel} · slow potential", size=t.SMALL, color=t.MUTED)],
                   spacing=6)]
        if ring is not None:
            wave_canvas.shapes = signed_shapes(
                [ring.series()[session.channel]], width, WAVE_H, full, [colour],
                pushed=ring.pushed)
            place_value_labels(wave_labels, WAVE_H, full, bottom=-full, unit="µV")
            wave_caption.value = (f"{_duration(ring.seconds)} across · ±{full:g} µV full scale, "
                                  f"this plot's own · middle line = zero, not a baseline")
        if session.metric is None:
            wave_state.value, wave_state.color = "no channel to measure", t.WARN
        elif session.raw_shift is None:
            wave_state.value, wave_state.color = "no readings yet", t.FAINT
        else:
            wave_state.value = ""

        traces.refresh(session.trace_seconds, levels_shown=True,
                       extra=("the trained signal is below this band and has its own pane",))

        page.update()

    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)

    build_wave_pills()
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    screen = ft.Column([toolbar.control, tabs.control, ft.Stack(panes, expand=True)], spacing=0, expand=True)
    return ft.Stack([screen, overlay.control], expand=True)
