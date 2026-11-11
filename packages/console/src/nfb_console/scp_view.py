"""The slow-potential screen: a run of trials, each with its own baseline and a cued direction.

Every other protocol here is one undifferentiated stretch of signal judged against a threshold. This one is
a **sequence**: a cue names the direction, two seconds are averaged into that trial's baseline, six seconds
of shift against it are fed back, and the trial is scored on the mean of those six seconds. The state
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
drawn with `distinct_phases`, so the baseline cannot be taken for the feedback: dim with a ring drawing
itself through the lead-in, bright with the moving circle through the feedback window.
"""

from __future__ import annotations

from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import SlowShift, shift_survival

from . import chrome, cues, heart_panel, plan_pane, signal_pane, spectrum_pane, trace_rows, trends_pane
from . import theme as t
from .plots import place_value_labels, signed_shapes, value_axis
from .protocols import Protocol
from .targets import Target
from .traces import RawRing, TraceBuffer, WaveRing
from .training import METRIC_HZ, TrainingSession, true_metric_hz
from .trial_training import (TrialTiming, TrialTraining, TrialUnits, TrialView, trial_settings_rows,
                             trial_summary)
from .trials import (ACTIVE, ACTIVE_S, BASELINE_S, DEFAULT_TRIALS, ITI_RANGE_S, NEGATIVE, POSITIVE,
                     RESULT, RESULT_S, TRANSFER_FRACTION)
from .target_card import NOTE_H
from .widgets import button, help_button, labelled, num_box, panel, pill, test_button

HANDLES = ("SCP",)

from .trial_training import CUE_PITCH, CUE_VOLUME  # noqa: E402  (the cue tones are the trial piece's)

TIMING = TrialTiming(cue_s=0.0, baseline_s=BASELINE_S, active_s=ACTIVE_S, result_s=RESULT_S, iti_s=ITI_RANGE_S)
"""The published slow-potential trial: the baseline starts at the cue, two seconds of it, six of feedback."""

DEFAULT_RANGE_UV = 10.0
"""The feedback sensitivity, ±µV of mean shift for the level's full travel. Ten is the size of shift a
trained trainee produces; the box is there because that is a person-by-person number."""

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
    "down. The circle is blue on the cued side of the dashed ring and grey on the other side.\n\n"
    "The circle moves in proportion to the shift up to **full at**. Past that value, it moves less and "
    "less, but it never stops. A shift of three times **full at** shows further out than a shift of two "
    "times **full at**.\n\n"
    "The trial is scored on the mean of the six seconds, not on one instant. A thin faint ring shows "
    "that mean so far. A brief spike does not earn the trial, and a brief lapse does not lose it.\n\n"
    "A success sounds the trial's cue tone twice. A failure and a trial with no result stay silent. The "
    "circle stays at the mean shift, green for a success and red for a failure. The line under the "
    "circle says what the trial asked for, and how far and which way the slow potential went.\n\n"
    "A trial without feedback has the cue and the verdict, but no circle. The circle stays hidden for the "
    "whole feedback window instead of freezing. The arrows stay, because the direction still applies.\n\n"
    "A trial has *no result* if too much of its signal was artifacted, or if its baseline was too thin "
    "to use. It appears in the history but is excluded from every count. A blink's deflection lies in the "
    "same band this protocol trains, so scoring a blink scores the blink instead of the trained signal.\n\n"
    "When you choose a heart-rate source on the device screen, two heart panels appear beside the trials. "
    "They show the mean heart-rate change from baseline for each direction, and the heart rate across the "
    "run. A strategy that works through general arousal moves the heart in both directions."
)

HISTORY_HELP = (
    "Every completed trial, in order, oldest at the left.\n\n"
    "A filled cell is a success, and an outlined one is a failure. A muted cell is a trial with no result. "
    "Its signal was artifacted, or its baseline was too thin to mean anything. A dot under a cell marks a "
    "trial without feedback.\n\n"
    "Under the cells, each direction has its own line: how many trials succeeded, and the mean shift from "
    "the baseline. The directions are kept apart because the skill is *differentiation*. Somebody who has "
    "learned to produce one direction, and drifts through the other, has a good overall figure. But that "
    "person has not learned the thing being trained.\n\n"
    "The mean shift says more than the count. A direction whose mean leans the cued way shows control "
    "that grows before many trials pass the success margin. A mean the wrong way shows a strategy that "
    "works backwards.\n\n"
    "When the run has trials without feedback, they get their own lines. A shift held without the circle is "
    "the real test of control. Trials with no result are in no line and are counted on their own. A run "
    "with many of them is a wiring problem rather than a poor session."
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
    "**full at** sets the sensitivity. At 10 µV, the level reads 100% for a 10 µV shift the cued way. "
    "It reads 0% for a 10 µV shift the wrong way, and 50% at the baseline. The condition, and so the "
    "click, starts at 50%. Narrow the range and small shifts swing the output further, but the output "
    "then stays at 0% or 100% more often. Widen the range and the output moves less.\n\n"
    "The circle in the trial panel does not stop at **full at**. It moves in proportion up to that value, "
    "and less and less past it, so a large shift still shows.\n\n"
    "This number is not the success criterion. The success margin, judged on the mean of the whole "
    "feedback window, decides the trial. This number only sets how much of the output's travel a given "
    "shift uses.\n\n"
    "The threshold is the trial's baseline. You cannot change it, because there is no other value to "
    "set it from.\n\n"
    "Changing the channel restarts the run, because a run measured on two different electrodes is not "
    "one run."
)

RUN_HELP = (
    "Changing **direction**, **trials**, **baseline starts**, **without feedback** or **succeeds at** "
    "starts the run again. Results measured under one setting are not results under another.\n\n"
    "**direction** sets which shifts the run asks for. *random* asks for up and down in an order that "
    "you cannot predict, with the same number of each. The tone at the cue does not name the direction. "
    "The direction tone sounds when the feedback starts. Then you cannot push the signal the other way "
    "during the baseline to make the shift look larger. *alternating* asks for up and down in turn. "
    "*increase* and *decrease* ask for one direction only. With these three, the tone at the cue names "
    "the direction.\n\n"
    f"**trials** sets the block size. A trial takes about 12 s, so {DEFAULT_TRIALS} trials take about "
    f"{round(DEFAULT_TRIALS * 12 / 60)} minutes.\n\n"
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
    "**succeeds at** sets how far past the baseline the mean shift must go to count. A run starts at "
    "0 µV, so any shift in the cued direction counts. Raise it when the direction is reliable."
)


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
        # The run and what the operator sets about it, from the shared trial piece. Only active mode: the
        # protocol is the trials. A run starts with a third of its trials without a display.
        self._init_trials("active")
        self.transfer_fraction = TRANSFER_FRACTION

        # One target, for the three jobs `wave_view` lists: the shared pane draws a contribution row for
        # it, it is how that pane knows data is arriving, and it is what the click asks. Its value is the
        # **already-oriented, already-normalised** signal, so its threshold is the middle of the scale and
        # `_settled` says there is nothing left to calibrate.
        self.signal_target = Target(self.channel, None, "increase", threshold=0.5,
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
            self.channel = channel
            self.signal_target.source = channel
        spec = self.specs.get(self.channel)
        self.metric = None if spec is None else SlowShift(spec, output_hz=METRIC_HZ)
        self.wave = (None if spec is None else
                     WaveRing(WAVE_CENTRE_HZ, WAVE_HIGH_HZ, spec.rate_hz, (self.channel,),
                              cycles=WAVE_CYCLES))
        self.raw_shift = None
        self.amplitudes.pop(("scp",), None)
        self.restart_trials()

    @property
    def trial_metric_hz(self) -> float:
        """Every reading of the metric reaches the engine, not a thinned stream."""
        return true_metric_hz(self.rate)

    def set_mode(self, mode: str) -> None:
        """One mode only: the protocol is the trials."""

    def trial_hold(self) -> str | None:
        """The run waits for the filter to settle, for the reason `consume` gives. `consume` holds the
        readings back; this says why under the circle."""
        if self.metric is None:
            return "no channel to measure"
        if not self.settled:
            return f"the filter is settling · {self.metric.warmup:.0%} through"
        return None

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
        return self.full_at

    def set_range(self, half_range_uv: float) -> None:
        self.set_full_at(half_range_uv)

    @property
    def settled(self) -> bool:
        return self.metric is not None and self.metric.warmup >= 1.0

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
                f"±{self.full_at:g} µV",
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
        return self._signal if self._signal is not None else 0.5

    def latency_for(self, tgt: Target) -> float | None:
        return self.metric.latency_seconds if self.metric is not None else None

    # --- feed -------------------------------------------------------------------------------

    def observe_raw(self, block, judged) -> None:
        ch = block.channel
        values = np.asarray(block.values)
        if ch in self.traces:
            self.traces[ch].push(self.for_display(ch, values), block.valid, judged.accepted)
        if ch in self.rings:
            self.rings[ch].push(values)

    def consume(self, block) -> set:
        """Feed the metric, then the engine, then the level — in that order, with a gate in front of each.

        **The engine is not fed while the filters are settling**, because a reading that is still a
        transient would go into a baseline and be subtracted from six seconds of real signal. **Nor while
        the session is paused**: trials freeze with the transport, and on resume the engine's own time-jump
        rule aborts whatever trial was in flight — which is the right outcome, since its baseline describes
        signal that no longer precedes its feedback window.

        Invalid readings *do* reach the engine. It counts them toward voiding the trial, which is the
        clinical arrangement and the only honest one: a blink's deflection lives in exactly the band this
        protocol trains, so a trial full of them is not a trial rather than a bad one.
        """
        metric = self.metric
        if metric is None or block.channel != self.channel:
            return set()
        fresh: set = set()
        # A held stand-in keeps a valid device state, so mark a rejected block invalid for the trial
        # engine, or a run of blinks scores as a valid baseline and shift.
        accepted = self._accepted.get(block.channel, True)
        usable = self.state.get(self.channel) is not False     # resolved state (the operator override), not
        for r in metric.feed(block):                           # the raw device flag, which ignores the override
            self.raw_shift = r.value
            self.amplitudes[("scp",)] = r.value
            if metric.warmup < 1.0 or self.paused:
                continue
            # The trial piece orients the level toward the cue, sounds the cue and the verdict (a success
            # is the cue tone doubled; a failure and a void trial are silent), and ends the run. Its end
            # stops the clock, as `tick()` does for a timed plan: the run *is* the session's length here.
            self._observe_trial(r.t, r.value, usable and accepted)
            if self._driving:
                fresh.add(("scp",))
        if self.wave is not None:
            self.wave.feed(metric.wave_start, {self.channel: metric.waveform})
        return fresh

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

    # --- what this protocol requires --------------------------------------------------------

    def wiring_problem(self) -> str | None:
        """Whether the amplifier's corner leaves enough of a shift to train.

        The metric has no high-pass of its own, so the device's corner is the only thing under the band —
        and what it costs is not a band edge but a *fraction of the mean*, since a mean over the feedback
        window is exactly what a trial is scored on. See `nfb.dsp.scp.shift_survival`.
        """
        if self.metric is None:
            return "no channel to measure"
        surviving = shift_survival(self.corner_hz, ACTIVE_S)
        if surviving < SURVIVAL_FLOOR:
            return (f"the device's {self.corner_hz:g} Hz corner keeps {surviving:.0%} of a shift's "
                    f"mean over {ACTIVE_S:g} s — set High-pass to {BEST_CORNER_HZ:g} Hz on the "
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
        self.run_rows = ft.Column(spacing=9)
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
                ft.Container(height=10), self.run_rows,
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
            help_button(self.page, "What is measured", SIGNAL_HELP),
        ]
        # Where a threshold's mode row goes on every other card. What is chosen here is the run's shape:
        # the shared trial rows, with the wait from the cue to the baseline, which only this screen sets.
        # The panel's `?` says that changing one starts the run again.
        self.run_rows.controls = trial_settings_rows(
            session, self._changed, cue_gap=True,
            help=help_button(self.page, "The run", RUN_HELP), width=RUN_LABEL_W)
        self.page.update()

    def _set_channel(self, channel: int) -> None:
        self.session.set_channel(channel)
        self.rebuild()
        self.on_change()

    def _changed(self, action) -> Callable:
        return lambda *a: (action(*a), self.rebuild(), self.on_change())

    def refresh(self) -> None:
        session = self.session
        run = session.trials
        # What the feedback shows, not the live shift: in a trial without a display, after the circle fades
        # and between steps, a live number here would be the feedback the trial withholds.
        shift = session.shown_shift if session.feedback_now else None
        full = max(session.full_at, MIN_RANGE_UV)
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
        self.scale_text.value = (f"0% at −{full:g} · 50% at the baseline · 100% at +{full:g} µV, "
                                 f"toward the cue")
        self.filter_text.value = ("" if session.metric is None else
                                  f"level {level:+.1f} µV · {session.metric.describe()}"
                                  if level is not None else session.metric.describe())
        if session.metric is None:
            self.note.value = "no channel to measure"
        elif not session.settled:
            self.note.value = (f"settling — {_duration(session.metric.settle_seconds)} of filter, "
                               f"{session.metric.warmup:.0%} through")
        else:
            self.note.value = session.wiring_problem() or ""


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
    trial_ui = TrialView(session, CIRCLE_SIDE, heart=session.heart_on, distinct_phases=True)

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
                                           session.UNITS.decimals)

    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        card.refresh()
        signal.refresh()
        spectrum.refresh()
        trends.refresh()
        trial_ui.draw_fixation("")
        draw_history()
        if heart_ui is not None:
            heart_ui.refresh(session.heart)
            trial_ui.draw_heart_stats()

        ring, width = session.wave, plot_width()
        full = max(WAVE_FLOOR_UV, session.wave_uv)
        colour = session.colour if session.settled else t.MUTED
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
            wave_state.value = "no channel to measure"
        elif not session.settled:
            wave_state.value = f"settling · {session.metric.warmup:.0%}"
        else:
            wave_state.value = ""

        traces.refresh(session.trace_seconds,
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

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)
