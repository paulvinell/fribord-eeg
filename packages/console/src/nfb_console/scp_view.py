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

**Two things are deliberately absent from the feedback.** There is no music, because continuous music under
discrete trials would blur the one boundary the protocol is made of; and no alert, because the condition is
crossed constantly by construction — half a trial's readings are on the wrong side of a baseline by
definition, so an alert would fire on the ordinary behaviour of the signal.

**A transfer trial has a cue and a verdict and no display.** That is what weans the skill off the screen,
and it is why the live bar is *hidden* rather than frozen during one: a bar that stopped moving reads as a
fault, where an absent one reads as the design — which the badge on the panel says in words.
"""

from __future__ import annotations

from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import SlowShift, shift_survival

from . import chrome, cues, plan_pane, signal_pane, spectrum_pane, trace_rows, trends_pane
from . import theme as t
from .plots import place_value_labels, signed_shapes, value_axis
from .protocols import Protocol
from .targets import Target
from .traces import RawRing, TraceBuffer, WaveRing
from .training import METRIC_HZ, TrainingSession, true_metric_hz
from .trials import (ACTIVE, ACTIVE_S, BASELINE, CUE, DEFAULT_TRIALS, ITI, NEGATIVE, POSITIVE,
                     RESULT, TRANSFER_FRACTION, VERDICT, TrialRun)
from .target_card import NOTE_H
from .widgets import button, help_button, labelled, num_box, panel, pill, test_button

HANDLES = ("SCP",)

CUE_PITCH = {POSITIVE: 620.0, NEGATIVE: 310.0}
"""The two cue tones, an octave apart.

Same reasoning as alpha-theta's reward tones, which are two and a half octaves apart for it: the whole job
of these is to be told from each other by somebody who is not looking at the screen, and an octave is the
interval the ear reads as "the other one" rather than as "the same one, slightly off"."""

CUE_VOLUME = 25
"""As the alert's, and above the click's 15: a cue that is missed is a trial that is not run. The scale is
linear amplitude, not loudness, so 25 is about −12 dB."""

DEFAULT_RANGE_UV = 10.0
"""The feedback sensitivity, ±µV of mean shift for the level's full travel. Ten is the size of shift a
trained trainee produces; the box is there because that is a person-by-person number."""

MIN_RANGE_UV = 1.0
"""A floor, for the reason `wave_view.MIN_RANGE_UV` gives: below it the level is a picture of electrode
noise."""

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
LEVEL_H = 22
CELL_W, CELL_H = 11, 16

TRIAL_HELP = (
    "One trial runs in order: a cue, a baseline, a feedback window, a verdict, then a rest.\n\n"
    "The cue names the direction and sounds a tone: high for up, low for down. The two tones are an "
    "octave apart, so you can tell them apart with your eyes closed.\n\n"
    "The baseline is the average of the first two seconds. Everything after it is measured against this "
    "baseline.\n\n"
    "Six seconds of feedback follow. The bar shows the shift against the baseline. Right is always the "
    "cued direction, so on a down trial, a fall fills the bar to the right. The trial is scored on the "
    "mean of those six seconds, not on any single instant. A brief spike does not earn the trial, and a "
    "brief lapse does not lose it.\n\n"
    "A success sounds the trial's cue tone twice. A failure and a void trial stay silent.\n\n"
    "A transfer trial has the cue and the verdict, but no display. The bar stays hidden for the whole "
    "trial instead of freezing.\n\n"
    "A trial is **void** if too much of its signal was artifacted, or if its baseline was too thin to "
    "use. A void trial appears in the history but is excluded from every rate. A blink's deflection lies "
    "in the same band this protocol trains, so scoring a blink scores the blink instead of the trained "
    "signal."
)

HISTORY_HELP = (
    "Every completed trial, in order, oldest at the left.\n\n"
    "A filled cell is a success, an outlined one a failure, a muted one a void trial — artifacted, or "
    "with too thin a baseline to mean anything. A dot under a cell marks a transfer trial.\n\n"
    "The two rates are kept apart because the skill is *differentiation*. Somebody who has learned to "
    "produce one direction, and drifts through the other, has a good overall figure. But that person has "
    "not learned the thing being trained. Void trials are in neither rate and are counted on their own. "
    "A run with many of them is a wiring problem rather than a poor session."
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
    "**full at ±** sets the sensitivity. At ±10 µV, the level reads 100% for a 10 µV shift the cued way. "
    "It reads 0% for a 10 µV shift the wrong way, and 50% at the baseline. The condition, and so the "
    "click, starts at 50%. Narrow the range and small shifts swing the output further, but the output "
    "then stays at 0% or 100% more often. Widen the range and the output moves less.\n\n"
    "This number is not the success criterion. The success margin, judged on the mean of the whole "
    "feedback window, decides the trial. This number only sets how much of the output's travel a given "
    "shift uses.\n\n"
    "The threshold is the trial's baseline. You cannot change it, because there is no other value to "
    "set it from.\n\n"
    "Changing the channel restarts the run, because a run measured on two different electrodes is not "
    "one run."
)

RUN_HELP = (
    "Changing any of these settings starts the run again. Results measured under one setting are not "
    "results under another.\n\n"
    f"**trials** sets the block size. A trial takes about 12 s, so {DEFAULT_TRIALS} trials take about "
    f"{round(DEFAULT_TRIALS * 12 / 60)} minutes.\n\n"
    f"**transfer** sets the share of trials that run without a display. It starts at "
    f"{TRANSFER_FRACTION:.0%}, a typical share for an experienced trainee. For a first session, set it "
    "to 0%.\n\n"
    "**succeeds at** sets how far past the baseline the mean shift must go to count. A run starts at "
    "0 µV, so any shift in the cued direction counts. Raise it when the direction is reliable."
)


class SCPSession(TrainingSession):
    """One channel, one slow signal, and a run of trials over it.

    Everything generic is `TrainingSession`. What is here is the metric, the trial engine, the cues the
    engine's events sound, and a level that exists **only during a trial's feedback window** — which is
    the one thing that makes this session shape unlike every other on this screen's shelf.
    """

    offers = ("sound", "shadow", "click")
    """No music: continuous music under discrete trials would blur the one boundary the protocol is made
    of, and the audio channel is already carrying the cue and the verdict. No alert: the condition here is
    a trial's own baseline, which the signal crosses constantly by construction, so an alert would fire on
    the ordinary behaviour of the signal rather than on an exception."""

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
                 source=None, channels: int = 0):
        super().__init__(protocol, path=path, device=device, length_seconds=length_seconds,
                         source=source, channels=channels)
        if length_seconds is None:
            self.plan.set_length(SESSION_MINUTES)

        self.channel = next(iter(self.specs), 0)
        self.gain_uv = 100.0
        self.wave_uv = WAVE_GAINS[1]
        self.trace_seconds = TRACE_SECONDS

        self.traces = {i: TraceBuffer(320, TRACE_SECONDS, s.rate_hz) for i, s in self.specs.items()}
        self.rings = {i: RawRing(2.0, s.rate_hz, SPECTRUM_LO, SPECTRUM_HI, 1.0)
                      for i, s in self.specs.items()}

        # What the operator sets about the run itself. Held on the session rather than read off the
        # engine, because the engine is *rebuilt* whenever one of them changes and a setting that lived
        # only inside it would be lost with it.
        self.n_trials = DEFAULT_TRIALS
        self.transfer_fraction = TRANSFER_FRACTION
        self.success_uv = 0.0
        self.range_uv = DEFAULT_RANGE_UV

        self.raw_shift: float | None = None
        self._signal: float | None = None
        # Whether the last reading actually drove the level. The transition out of it is what releases the
        # outputs: without that the sound and the shadow hold their last value through the whole rest
        # interval, which says "you are still doing it" while nobody is doing anything.
        self._driving = False

        self.metric: SlowShift | None = None
        self.wave: WaveRing | None = None
        self.trials = TrialRun(metric_hz=true_metric_hz(self.rate))

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

    def restart_trials(self) -> None:
        """Throw the run away and start a new one with the settings as they now stand.

        A fresh seed — the sequence of directions and transfer slots is drawn anew, because a second run
        with the same order would let a trainee learn the order rather than the cue.
        """
        self.trials = TrialRun(n_trials=max(1, int(self.n_trials)),
                               transfer_fraction=self.transfer_fraction,
                               success_uv=self.success_uv,
                               metric_hz=true_metric_hz(self.rate))
        self._signal = None
        self._driving = False
        self.finished = False
        self._outputs_off()

    def set_range(self, half_range_uv: float) -> None:
        self.range_uv = max(MIN_RANGE_UV, float(half_range_uv))

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
                f"±{self.range_uv:g} µV",
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
        for r in metric.feed(block):
            self.raw_shift = r.value
            self.amplitudes[("scp",)] = r.value
            if metric.warmup < 1.0 or self.paused:
                continue
            for event in self.trials.feed(r.t, r.value, valid=r.valid):
                self._sound(event)
            shift = self.trials.shift
            driving = self.trials.feedback_live and r.valid and shift is not None
            if driving:
                # Oriented, so that the level always rises toward the direction that was asked for. Every
                # mechanism reads one number and none of them knows which way this trial goes, so the
                # orientation has to happen here or a down trial would darken the screen for compliance.
                oriented = shift if self.trials.required == POSITIVE else -shift
                self._signal = min(1.0, max(0.0, 0.5 + oriented / (2 * self.range_uv)))
                fresh.add(("scp",))
            else:
                self._signal = None
                if self._driving:
                    self._outputs_off()
            self._driving = driving
        if self.wave is not None:
            self.wave.feed(metric.wave_start, {self.channel: metric.waveform})
        if self.trials.done and not self.finished:
            # The same completion path `tick()` takes for a timed plan: the run *is* the session's length
            # here, so its end stops the clock rather than leaving it running over a screen with nothing
            # left to happen on it.
            self.finished = True
            self._accumulated = self.elapsed
            self._resumed_at = None
            self.paused = True
            self._signal = None
            self._driving = False
            self._outputs_off()
        return fresh

    def _sound(self, event) -> None:
        """The two sounds this protocol has, both through `self.cue` so a test can silence them.

        A success is the trial's own cue tone **doubled** — "you did the thing you were cued", said with
        the sound that cued it, which needs no second pitch and cannot be confused with the cue itself.
        Failure and void are silent.
        """
        if event.kind == CUE:
            self.cue(cues.tone_at(CUE_PITCH[event.required]), CUE_VOLUME)
        elif event.kind == VERDICT and event.trial is not None and event.trial.success:
            self.cue(cues.held(cues.tone_at(CUE_PITCH[event.required])), CUE_VOLUME)

    # --- surviving a stop -------------------------------------------------------------------

    def adopt_view(self, previous) -> None:
        super().adopt_view(previous)
        self.wave_uv = getattr(previous, "wave_uv", self.wave_uv)
        self.range_uv = getattr(previous, "range_uv", self.range_uv)
        self.n_trials = getattr(previous, "n_trials", self.n_trials)
        self.transfer_fraction = getattr(previous, "transfer_fraction", self.transfer_fraction)
        self.success_uv = getattr(previous, "success_uv", self.success_uv)

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
    _ = heart_choice, heart_runner, mendi_choice, mendi_runner, pupil_choice, pupil_runner
    return SCPSession(*args, **kwargs)


# --- names for what is on screen ----------------------------------------------------------------

PHASE_NAME = {BASELINE: "baseline", ACTIVE: "feedback", RESULT: "verdict", ITI: "rest"}

DIRECTION_TEXT = {POSITIVE: "▲ shift up", NEGATIVE: "▼ shift down"}


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
        self.range_row = ft.Row(spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
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
                ft.Container(height=9), self.range_row,
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
        self.range_row.controls = [
            ft.Text("full at ±", size=t.SMALL, color=t.MUTED, width=76),
            num_box(session.range_uv, self._set_range, suffix="µV", step=1,
                    low=MIN_RANGE_UV, width=52),
            ft.Container(width=8),
            ft.Text("of mean shift", size=t.SMALL, color=t.FAINT),
        ]
        # Where a threshold's mode row goes on every other card. What is chosen here is the run's shape,
        # and the panel's `?` — not each row — says that changing one starts the run again: three notes
        # saying the same thing is the failure `labelled`'s own rule is about.
        self.run_rows.controls = [
            labelled("trials", num_box(session.n_trials, self._set_trials, step=5, low=1,
                                       high=200, width=52),
                     ft.Container(width=6),
                     ft.Text(f"≈ {_duration(session.n_trials * 12.0)}", size=t.SMALL,
                             color=t.FAINT), width=76),
            labelled("transfer", num_box(round(session.transfer_fraction * 100), self._set_transfer,
                                         suffix="%", step=10, low=0, high=100, width=52),
                     ft.Container(width=6),
                     ft.Text("no display", size=t.SMALL, color=t.FAINT), width=76),
            labelled("succeeds at", num_box(session.success_uv, self._set_margin, suffix="µV",
                                            step=1, low=0, high=50, width=52),
                     ft.Container(width=6),
                     ft.Text("past the baseline", size=t.SMALL, color=t.FAINT), width=76),
            ft.Row([button("restart run", lambda _e: self._restart()),
                    ft.Container(expand=True),
                    help_button(self.page, "The run", RUN_HELP)],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            # Both cue tones and the success mark, auditioned before trial one. This protocol is run
            # watching the trainee rather than the screen, so the sounds are the whole channel — and the
            # alternative is discovering after eight minutes that one of them was inaudible.
            ft.Row([test_button("up", lambda _e: self._audition(POSITIVE)),
                    test_button("down", lambda _e: self._audition(NEGATIVE)),
                    test_button("success", lambda _e: self._audition(POSITIVE, mark=True))],
                   spacing=6),
        ]
        self.page.update()

    def _set_channel(self, channel: int) -> None:
        self.session.set_channel(channel)
        self.rebuild()
        self.on_change()

    def _set_range(self, uv: float) -> None:
        self.session.set_range(uv)
        self.rebuild()
        self.on_change()

    def _set_trials(self, n: float) -> None:
        self.session.n_trials = int(n)
        self._restart()

    def _set_transfer(self, percent: float) -> None:
        self.session.transfer_fraction = max(0.0, min(1.0, percent / 100.0))
        self._restart()

    def _set_margin(self, uv: float) -> None:
        self.session.success_uv = max(0.0, float(uv))
        self._restart()

    def _restart(self) -> None:
        self.session.restart_trials()
        self.rebuild()
        self.on_change()

    def _audition(self, required: str, mark: bool = False) -> None:
        sound = cues.tone_at(CUE_PITCH[required])
        self.session.cue(cues.held(sound) if mark else sound, CUE_VOLUME)

    def refresh(self) -> None:
        session = self.session
        run = session.trials
        shift, full = run.shift, max(session.range_uv, MIN_RANGE_UV)
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

    # ---- the trial panel ----
    direction_text = ft.Text("", size=26, color=t.FAINT, weight=ft.FontWeight.W_500)
    phase_text = ft.Text("", size=t.BODY, color=t.TEXT)
    count_text = ft.Text("", size=t.SMALL, color=t.MUTED)
    transfer_badge = ft.Container(
        content=ft.Text("transfer", size=t.SMALL, color=t.MUTED),
        # Faded rather than hidden between transfer trials, so the row it sits in never reflows.
        padding=ft.Padding(7, 3, 7, 3), border_radius=5, bgcolor=t.SURFACE_HI, opacity=0.0)
    transfer_note = ft.Text("", size=t.SMALL, color=t.FAINT)

    trial_fill = ft.Container(width=0, height=LEVEL_H, border_radius=5, bgcolor=t.ACCENT_DIM,
                              left=0, bottom=0)
    trial_mark = ft.Container(width=1.5, height=LEVEL_H, bgcolor=t.MUTED, left=0, bottom=0)
    trial_track = ft.Stack([ft.Container(width=0, height=LEVEL_H, border_radius=5,
                                         bgcolor=t.SURFACE_HI, left=0, bottom=0),
                            trial_fill, trial_mark], height=LEVEL_H)
    # Always present: hiding it between trials changed the panel's height every twelve seconds.
    trial_bar = ft.Container(content=trial_track)
    trial_caption = ft.Text("", size=t.SMALL, color=t.FAINT)

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
        panel("TRIAL",
              ft.Row([direction_text, ft.Container(width=18),
                      ft.Column([ft.Row([phase_text, ft.Container(width=10), transfer_badge],
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                 ft.Container(height=3), count_text], spacing=0),
                      ft.Container(expand=True)],
                     vertical_alignment=ft.CrossAxisAlignment.CENTER),
              ft.Container(height=10), trial_bar,
              ft.Container(height=6), trial_caption, transfer_note,
              help=help_button(page, "How a trial runs", TRIAL_HELP)),
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

    train_right = ft.Column([
        # The corner warning is the card's note — it was also printed under the card, which was one
        # sentence twice in one panel.
        panel("TRAINING", card.control),
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

    def draw_trial() -> None:
        run = session.trials
        running = run.required is not None and run.phase in (BASELINE, ACTIVE, RESULT)
        if run.done:
            direction_text.value = f"done · {len(run.results)} trials"
            direction_text.color = t.MUTED
        elif running:
            direction_text.value = DIRECTION_TEXT[run.required]
            direction_text.color = t.ACCENT
        else:
            direction_text.value = "rest"
            direction_text.color = t.FAINT
        phase_text.value = (f"{PHASE_NAME.get(run.phase, run.phase)} · "
                            f"{run.phase_remaining():.1f} s")
        count_text.value = f"trial {run.trial_number}/{run.n_trials}"
        transfer = bool(running and run.transfer)
        transfer_badge.opacity = 1.0 if transfer else 0.0
        transfer_note.value = "no display — cue and verdict only" if transfer else ""

        signal_value = session.signal
        bar_w = max(120.0, plot_width() - 40)
        trial_track.width = bar_w
        trial_track.controls[0].width = bar_w
        trial_mark.left = bar_w * 0.5
        trial_fill.width = 0 if signal_value is None else bar_w * signal_value
        if signal_value is not None:
            trial_fill.bgcolor = t.ACCENT if signal_value >= 0.5 else t.ACCENT_DIM
            shift = run.shift or 0.0
            trial_caption.value = (f"{shift:+.1f} µV against this trial's baseline · full at "
                                   f"±{session.range_uv:g} µV · mark = the baseline")
        elif not session.settled:
            trial_caption.value = "the filter is still settling — nothing is fed back yet"
        elif run.done:
            trial_caption.value = "the run is over · restart it from the card"
        else:
            trial_caption.value = "no feedback in this window"

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
            cells.append(ft.Column([cell, ft.Container(height=2),
                                    ft.Container(width=5, height=5, border_radius=3,
                                                 bgcolor=t.MUTED if result.transfer else None)],
                                   spacing=0,
                                   horizontal_alignment=ft.CrossAxisAlignment.CENTER))
        history_cells.controls = cells
        if not run.results:
            history_text.value = "no trials yet"
            return
        up, down = run.success_rate(POSITIVE), run.success_rate(NEGATIVE)
        ups = [r for r in run.valid_results if r.required == POSITIVE]
        downs = [r for r in run.valid_results if r.required == NEGATIVE]
        void = sum(1 for r in run.results if not r.valid)
        parts = []
        if up is not None:
            parts.append(f"{sum(r.success for r in ups)}/{len(ups)} up")
        if down is not None:
            parts.append(f"{sum(r.success for r in downs)}/{len(downs)} down")
        parts.append(f"{void} void")
        history_text.value = " · ".join(parts)

    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        card.refresh()
        signal.refresh()
        spectrum.refresh()
        trends.refresh()
        draw_trial()
        draw_history()

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
