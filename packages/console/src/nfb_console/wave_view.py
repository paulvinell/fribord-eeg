"""The waveform screen: train where the filtered wave *is*, not how big it is.

The ILF slow-wave reward, which the synchrony screen next door draws the signal for and does not train.
What matters here is the shape.

**Why this is a screen of its own rather than a setting on the amplitude screen.** An amplitude needs a
rectifier averaged over several cycles before it means anything, so at 0.05 Hz — three cycles is a minute — it
cannot be trained at all. The signed value has no such requirement: it exists every sample and lags by the
filter alone. That is the whole argument, and it is why this reward is a level crossing
on a waveform rather than a band amplitude.

**It stops at 1 Hz, and above that there is nothing here rather than something else.** A wave at 10 Hz is
100 ms a cycle, unfollowable as a position — and the tempting way up is to feed back a fast band's
*amplitude*, whose swing is slow enough to look like the same quantity. It is not: an amplitude is a size
where this is a position. So the frequency box stops, and training a band's amplitude remains the amplitude
screen's job, which does it better than a slow reading of it here would. An earlier version of this screen
offered that as a second source and was wrong in the way that is hardest to see — feeding back a size while
every word on the screen said position.

**Where the level comes from is the three-row model the heart screen introduced**, and this is the second
customer for it, which is the test of whether that layer had the right shape: the de-trended wave in µV is the
metric, the aim is which side of its own baseline is being trained toward, and the scaling turns microvolts
into the 0…1 every mechanism follows. Everything from the level outward is the shared machinery, unchanged.

**One deliberate choice, the same one `music.ShepardPlayer` argues for.** Auto-ranging the feedback tells the
trainee about their own variance rather than their state. The other answer is slope compensation, so drift is
removed from the *signal* here (`nfb.dsp.wave.DETREND_POLES`) and the feedback scale is calibrated once and
then held for the session. A calibration is a phase, not a third mode, exactly as on the amplitude screen.

**Three scales, and keeping them apart is the point.** They were briefly one number, which is what made the
card strange to read: a control labelled "range" set the plot's height *and* what the feedback mapped over, so
a display choice silently changed what the mechanisms were doing.

| scale | where it is set | what it is |
|---|---|---|
| the plot's height | pills above the WAVE panel, and `fit` | how much of the pane a wave fills |
| the feedback range | calibrated, editable beside the scaling on the Feedback tab | what maps onto 0…1 |
| the raw trace's gain | pills above RAW | the wideband signal, which is 40× larger |

The card *reports* the second and does not set it, because the scaling it belongs to lives on the Feedback tab.

**What the drift is doing is a number and not a trace.** Measured on this hardware it runs to 305 µV against
20 µV of signal, so on the wave's own axis it is off-screen by a factor of fifteen — the same reason
`TrainingSession.DISPLAY_FLOOR_HZ` gives for high-passing the raw trace instead of offering a ±800 µV gain.
The caption says how much is being removed and above which frequency.

Four tabs, for the reasons `amplitude_view` gives. The slow spectrum is on **Train** rather than beside the
ordinary one, because it is read *during* the session — "is anything organised at the frequency I set?" — where
the 1–60 Hz spectrum is what says whether the channel is healthy at all.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import SignedWave, wave_edges
from nfb.dsp.wave import DETREND_RATIO, MAX_HZ

from . import chrome, normalise, plan_pane, signal_pane, spectrum_pane, trace_rows, trends_pane
from . import theme as t
from .feedback import CONDITION
from .plots import (bars, nice_ceiling, place_tick_labels, place_value_labels,
                    signed_shapes, tick_axis, value_axis)
from .protocols import Protocol
from .targets import RECENT_WINDOW, Target
from .traces import RawRing, SlowRing, TraceBuffer, WaveRing
from .training import METRIC_HZ, TrainingSession
from .target_card import NOTE_H
from .widgets import grid_hz, help_button, labelled, num_box, panel, pill, settings_button

HANDLES = ("WAVE",)

# The two places this instrument is pointed, and the frequency box is the actual control — same shape as the
# synchrony screen's presets and for the same reason: there is one instrument, and a pill moves it.
#
# 0.05 Hz is where infra-low work sits and is the slowest setting this amplifier reaches cleanly (see
# `wiring_problem`). 0.2 Hz is the slow-potential end of the same idea, still trainable as a position at five
# seconds a cycle. There is no third: the frequency stops at `MAX_HZ`, and a preset above it would be a preset
# for a quantity this screen does not train.
PRESETS: tuple[tuple[str, float], ...] = (
    ("infra-slow", 0.05),
    ("slow", 0.2),
)
DEFAULT_HZ = 0.05
HZ_RANGE = (0.01, MAX_HZ)
"""Down to the device's own best corner and up to where a position stops being followable.

The bottom is the amplifier: a 0.01 Hz corner
passes 71% *at* the corner, so anything below it is measuring the high-pass. The top is `nfb.dsp.wave.MAX_HZ`
rather than a number of this screen's own, so the metric refuses exactly what the box refuses and neither can
drift away from the other."""

ORDER_RANGE = (1, 2)
"""Filter order. Two by default: the conventional realtime choice."""

CALIBRATION_CYCLES = 3.0
CALIBRATION_MIN_SECONDS = 90.0
"""How long the range is measured over: three cycles or ninety seconds, whichever is longer. Three, because
one cycle of a wave is not an estimate of its size; ninety, because at 0.2 Hz three cycles is fifteen seconds
and a range set from fifteen seconds of a slow signal is a range set from one excursion.

**It is how long the range takes to stop moving, not how long there is nothing to see.** The range is in use
from the first reading, measured over whatever has arrived, and it is refined on every one after that. This
window is a *rolling* one, so when it has been full of settled signal the number in use is already the number
the old locked calibration would have produced — the same figure, arrived at without four and a half minutes
of an empty screen. See `_observe_calibration`."""

CALIBRATION_REFRESH_SECONDS = 1.0
"""How often the provisional range is recomputed while it is being refined. A median over the window costs
what the window holds, and a half-range measured over ninety seconds cannot meaningfully change in less than
a second — recomputing per reading would be twenty times the work for a number that moves at the same
speed."""

RANGE_SIGMAS = 2.0
"""The calibrated half-range, as a multiple of the RMS measured during the calibration. Two puts a sine's own
peaks at 70% of the range, which leaves room for a larger excursion without the scale being mostly empty —
where one would clip half of them and three would use a third of it."""

MIN_RANGE_UV = 1.0
"""A floor under the calibrated range, so a channel that was flat during its calibration does not produce a
scale on which electrode noise fills the screen."""

CONDITION_LEVEL = 0.5
"""Where the level counts as met, for the click and the music's stop: the middle of the scaled range, which on a
calibrated symmetric range is the wave's own baseline. The reward is a level crossing and this is the level.

There is nothing to derive from thirty seconds of signal here, unlike a band amplitude — the calibration
already set the scale, and a threshold on top of it would be a second calibration over the first."""

SESSION_MINUTES = 20.0

BEST_CORNER_HZ = 0.01
"""What to tell the operator to set on the device screen. `TrainingSession.corner_hz` reads the actual one."""

SLOW_LO_HZ, SLOW_HI_HZ = 0.005, 1.0
SLOW_WINDOW_SECONDS = 300.0
SLOW_BINS = 24

WAVE_CYCLES = 8
"""Cycles across the wave plot. Fewer than the synchrony screen's twelve, because this screen's window has to
fill before it says anything and at 0.05 Hz twelve cycles is four minutes."""

# µV full scale for the wave plot, and **its own set, not the raw trace's**: what reaches this pane is
# de-trended and band-limited to a tenth of a hertz, where the trace carries everything the amplifier passes.
# Measured on this hardware with electrodes on, the drift-removed slow wave runs about 20 µV where the
# wideband signal is 305 µV of drift with 13 µV of EEG inside it.
WAVE_GAINS = (5.0, 10.0, 20.0, 50.0, 100.0)
WAVE_HEADROOM = 1.6         # how much room above the recent peak `fit` leaves before it clips

WAVE_FLOOR_UV = 1.0
LEVEL_H = 26
WAVE_H = 210
SLOW_H = 150
TRACE_H = 74
TRACE_SECONDS = 5.0
SIDE_W = 380
AXIS_W = 44
SPECTRUM_LO, SPECTRUM_HI = 1.0, 60.0
BAR_W = 200


WAVE_HELP = (
    "This is the signal you train. It is not the size of the wave. It is which side of the baseline the "
    "wave is on, and which way it moves.\n\n"
    "The window is shown in cycles of the trained frequency, not in seconds. At 0.05 Hz, eight cycles span "
    "nearly three minutes. At 1 Hz, eight cycles span eight seconds.\n\n"
    "The middle line is zero. It is both the baseline and the threshold. Every discrete output requires "
    "the wave to be on the wanted side of this line.\n\n"
    "The pills above the plot set how much of the pane the wave fills. They do not set the feedback range. "
    "The feedback range is calibrated and shown on the card. You can edit it on the Feedback tab. Press "
    "fit to pick the smallest scale that still shows the full wave.\n\n"
    "Drift has been removed from the signal. The caption states how much was removed, and above which "
    "frequency."
)

SETTLING_HELP = (
    "Feedback starts with the first reading. Two numbers behind it settle over time: the band filter and "
    "the drift follower.\n\n"
    "The band filter settles in about 20 seconds at 0.05 Hz, and faster at a higher frequency. The drift "
    "follower settles in about two and a half minutes at 0.05 Hz.\n\n"
    "The feedback range is measured from the signal, over a rolling window of three cycles or ninety "
    "seconds, whichever is longer. Early in a session, this range is measured from an unsettled signal "
    "and is not accurate. It becomes accurate once the window holds only settled signal.\n\n"
    "Watch the numbers settle instead of waiting for a fixed time. The screen shows how much each number "
    "has settled."
)

SIGNAL_HELP = (
    "One channel, one filtered wave. Its position is the feedback.\n\n"
    "The filter is a lowpass at twice the trained frequency. At 0.05 Hz, this trains a signal between "
    "0 Hz and 0.1 Hz. The filter has no high-pass of its own, so the amplifier's high-pass corner is the "
    "only limit below the band. See the warning under this card for that limit.\n\n"
    "The screen stops at 1 Hz. Above 1 Hz, a cycle takes under a second, and no position is trainable at "
    "that speed. Training a band's amplitude above 1 Hz is a different protocol. Use the amplitude "
    "screen.\n\n"
    "The threshold is the baseline, and you cannot change it. The de-trender places the baseline at zero, "
    "so there is no other value to set.\n\n"
    "**full at ±** sets the sensitivity. At ±20 µV, the level reads 0% at −20 µV and below, 50% at the "
    "baseline, and 100% at +20 µV and above. The condition, and so the click, starts at 50%. Narrow the "
    "range and small movements swing the output further. The output then stays at 0% or 100% more often. "
    "Widen the range and the output moves less.\n\n"
    "The range is calibrated to twice the RMS of the signal, over a rolling window of three cycles or "
    "ninety seconds, whichever is longer. This places the wave's own peaks near 15% and 85%. Press recalibrate to measure the range again from the current "
    "signal. Feedback continues during recalibration.\n\n"
    "Changing the frequency, the channel or the filter order restarts settling and calibration, because "
    "each change alters what is measured."
)

SLOW_HELP = (
    "The spectrum below 1 Hz, on log-spaced bins, over a five-minute window.\n\n"
    "The signal is averaged down to a few samples a second before this spectrum is computed. Building it "
    "needs a window of minutes, so the plot fills slowly and shows how much of its window holds real "
    "data.\n\n"
    "Look for a peak at the trained frequency's bin. That peak shows the trained activity. A rising edge "
    "toward the left shows drift that the de-trender has not yet removed, or an electrode that is drying."
)


class WaveSession(TrainingSession):
    """One channel, one filtered waveform, and its position as the level.

    Everything generic is `TrainingSession`. What is here is the measurement (`nfb.dsp.wave`), the calibration
    phase that sets the feedback range, and the level: a normalised signed value rather than a score against a
    threshold, which is why `scores_bands` is False and `level_source` says so.
    """

    music_sources = ("shepard", "tone", "noise", "file")
    """**The Shepard tone is native here**, and this is the screen it was written for: it belongs
    to exactly this reward. The synchrony screen offers it disabled because there it is borrowed.

    `eeg` is absent, and so is anything else that carries a waveform into hearing: those sonify a *fast* band,
    and at a tenth of a hertz there is nothing in the signal itself for an ear to follow. What carries a slow
    position is a pitch, which is the wheel."""

    offers = ("sound", "shadow", "click", "music")
    """No alert. There is one condition and it is crossed twice a cycle by construction, so an alert would
    fire on the ordinary behaviour of the signal rather than on an exception — the same test
    `TrainingSession.offers` applies to the amplitude screen's inhibits."""

    combines = ("weighted",)
    """One target, so every rule reduces to the same thing. Stated rather than offered."""

    threshold_modes = ("fixed",)
    """The threshold is the baseline and there is nothing for a tracker to track: the de-trender has already
    put the wave's own baseline at zero. See `CONDITION_LEVEL`."""

    scores_bands = False
    """The level is a metric and a scaling, not targets against thresholds — see `level_source` and
    `TrainingSession.scores_bands`."""

    tunes_envelope = False
    """`envelope_seconds` reaches no estimator here: there is no envelope in this measurement, which is the
    whole point of it."""

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0):
        super().__init__(protocol, path=path, device=device, length_seconds=length_seconds,
                         source=source, channels=channels)
        if length_seconds is None:
            self.plan.set_length(SESSION_MINUTES)

        self.channel = next(iter(self.specs), 0)
        self.order = ORDER_RANGE[1]
        self.gain_uv = 50.0
        self.wave_uv = WAVE_GAINS[2]
        """The plot's own full scale. Separate from `gain_uv`, which is the raw trace's, and separate again
        from the feedback range — see the module docstring's table."""
        self.trace_seconds = TRACE_SECONDS

        self.traces = {i: TraceBuffer(320, TRACE_SECONDS, s.rate_hz) for i, s in self.specs.items()}
        self.rings = {i: RawRing(2.0, s.rate_hz, SPECTRUM_LO, SPECTRUM_HI, 1.0)
                      for i, s in self.specs.items()}
        self.slow = SlowRing(SLOW_WINDOW_SECONDS, self.rate, SLOW_LO_HZ, SLOW_HI_HZ)

        # **The three rows, in the order the screen puts them.** The metric is the wave; the direction is
        # which side of its baseline is being trained toward; the scaling turns microvolts into 0…1. See
        # `hrv_view`, which introduced this layer, and `normalise`, which says why a scaling is not a
        # direction.
        self.direction = "increase"
        self.strategy_id = "fixed_range"
        self.range_uv = 20.0
        self._strategy = normalise.build("fixed_range", low=-self.range_uv, high=self.range_uv)
        self.strategy_params: dict = {}
        self.raw_wave: float | None = None
        self._signal: float | None = None

        self.metric: SignedWave | None = None
        self.wave: WaveRing | None = None
        # (signal_seconds, value²) over a rolling `calibration_seconds`, and `_settled_at` is when the
        # filters finished — the window that *ends* the calibration is the one after that. See
        # `_observe_calibration`.
        self._cal_squares: deque[tuple[float, float]] = deque()
        self._settled_at: float | None = None
        self._cal_at = -1e9
        self.calibrated = False

        # One target, and it exists for three jobs rather than to hold a threshold: the shared pane draws a
        # contribution row for it, it is how that pane knows data is arriving, and it is what the click asks.
        # Its value is the **already-normalised** signal, so its threshold is the middle of the scale and
        # `_settled` says there is nothing to calibrate. Same construction as the heart screen's.
        self.signal_target = Target(self.channel, None, "increase", threshold=CONDITION_LEVEL,
                                    mode="fixed", name="position", unit="")
        self.signal_target._settled = True
        self.targets = [self.signal_target]

        self.feedback.music.enabled = True
        self.feedback.music.source = "shepard"
        self.set_frequency(DEFAULT_HZ)

    # --- what is measured ------------------------------------------------------------------

    @property
    def frequency_hz(self) -> float:
        return self.metric.centre_hz if self.metric is not None else DEFAULT_HZ

    def set_frequency(self, hz: float | None = None, order: int | None = None,
                      channel: int | None = None) -> bool:
        """Point the instrument somewhere, rebuilding it. False if the settings are not measurable.

        Rebuilt rather than edited, and the calibration goes with it: every one of these arguments changes
        what is being measured, so a range calibrated against the old quantity would be a scale for a
        different signal. The waveform ring goes too, because its stride depends on the frequency.
        """
        if channel is not None and channel in self.specs:
            self.channel = channel
            self.signal_target.source = channel
        if order is not None:
            self.order = min(ORDER_RANGE[1], max(ORDER_RANGE[0], int(order)))
        wanted = self.frequency_hz if hz is None else hz
        centre = round(min(HZ_RANGE[1], max(HZ_RANGE[0], wanted)), 6)
        spec = self.specs.get(self.channel)
        if spec is None:
            return False
        try:
            metric = SignedWave(spec, centre, order=self.order, output_hz=METRIC_HZ)
        except ValueError:
            return False
        self.metric = metric
        self.wave = WaveRing(centre, metric.high, spec.rate_hz, (self.channel,), cycles=WAVE_CYCLES)
        self.amplitudes.pop(("wave",), None)
        self.restart_calibration()
        return True

    @property
    def colour(self) -> str:
        """The trained frequency's own colour, shared by the trace, the card's stripe and the legend.

        The repo's rule everywhere else — a spectrum bin, a target card and a graph line all take their
        colour from the frequency — and it is what makes a legend work: the swatch and the wave are the same
        colour because they are the same number, not because two call sites happened to pick the same one.
        """
        return t.band_colour(*self.band_edges)

    @property
    def band_edges(self) -> tuple[float, float]:
        if self.metric is None:
            return wave_edges(DEFAULT_HZ)
        return self.metric.low, self.metric.high

    @property
    def trained_floor_hz(self) -> float:
        """This screen's target carries no band, so the default derivation has nothing to read — see
        `TrainingSession.trained_floor_hz`. The signed wave is a lowpass with nothing under it, and the
        device's corner is the only floor there is, which is the whole subject of `corner_warning`."""
        return self.band_edges[0]

    # --- the calibration phase --------------------------------------------------------------

    @property
    def calibration_seconds(self) -> float:
        return max(CALIBRATION_MIN_SECONDS, CALIBRATION_CYCLES / max(self.frequency_hz, 1e-9))

    def restart_calibration(self) -> None:
        """Forget the range and measure it again. Called on every change to what is measured, and by the
        button on the card — which is the same action, so it is the same code path.

        **The range in use is not forgotten with it.** Recalibrating starts a new measurement; it does not
        take the feedback away while that measurement runs. The old figure stands until the first reading of
        the new one replaces it, which is the same rule the whole phase now follows.
        """
        self._cal_squares.clear()
        self._settled_at = None
        self._cal_at = -1e9
        self.calibrated = False
        self._signal = None
        self.raw_wave = None
        self.signal_target.reset_stats()
        self.signal_target._settled = True

    @property
    def settled(self) -> bool:
        return self.metric is not None and self.metric.warmup >= 1.0

    @property
    def calibrating(self) -> bool:
        """Whether the range is still moving. **Not whether there is feedback** — there is, from the first
        reading, on whatever the range is at that moment."""
        return not self.calibrated

    @property
    def signal_seconds(self) -> float:
        """How much *signal* has been through the metric, which is not the session clock.

        Everything about settling and calibrating is measured on this, and that is deliberate: both are
        properties of the filters and of the signal, so they have to advance with samples rather than with
        wall clock. Measuring the calibration on `elapsed` also made it untestable against a recording played
        faster than real time, which is how every screen test here works.
        """
        if self.metric is None:
            return 0.0
        return self.metric.samples_seen / max(self.metric.rate_hz, 1e-9)

    @property
    def calibration_remaining(self) -> float:
        """How much longer the range will keep moving. The clock starts when the filters settle, because
        only then is the window full of the quantity being measured rather than of the drift follower
        finding the drift — but the range is *in use* throughout, so this is a statement about how settled
        a number is and not about when there will be one."""
        if self.calibrated:
            return 0.0
        if self._settled_at is None:
            return self.calibration_seconds
        return max(0.0, self.calibration_seconds - (self.signal_seconds - self._settled_at))

    def _observe_calibration(self, value: float) -> None:
        """Refine the range from everything that has arrived, and stop refining once it is settled data.

        **The range is in use from the first reading, and this used to be a wait.** The window was collected
        with the feedback withheld and the range set once at the end, so at 0.05 Hz nothing happened for
        about four and a half minutes: 2.9 of filter settling with the screen saying so, then 90 seconds
        more of "calibrating the range". Two consecutive progress bars in place of a protocol. The screen
        had every number it needed to draw a level within a second of starting and drew nothing, which is
        the same failure the infra-low screen had and had removed.

        So the window is **rolling** rather than collected. Every reading pushes a square in and drops
        anything older than `calibration_seconds`, and the range is the median of what is currently in
        there. That gives three things at once: a range from the first reading, a range that improves as
        the estimate does, and — once the window has been full of settled signal — *the identical number*
        the old locked calibration produced, since the window then holds exactly the samples it used to
        collect. Nothing was traded away for the immediacy.

        `_settled_at` is why the ending is still honest. The clock that decides when refining stops starts
        when the filters do, so the final number is measured on the wave rather than on the drift follower
        still finding the drift. Before then the range is whatever the drift-polluted readings say, which
        is wrong and visible and improving — which is the whole point, and is worth more than a correct
        number nobody is being shown.

        The RMS is the **median** of the squared readings, not their mean, and the difference is what the
        calibration survives. The artifact gate judges on 0.5–50 Hz, so an excursion in the band this
        protocol trains — a movement, an electrode pop, tens of seconds long — sails through it; a mean
        captured by one such excursion during the window set the range 5.3× wide (measured, one 300 µV /
        20 s event), after which the wave occupied a tenth of the level's travel and the feedback was all
        but motionless with only a number on a card saying why. For a sinusoid the median of the squares
        *is* the mean (sin² spends half its time above ½), so on the wave this protocol measures the two
        estimators agree — the median just refuses to be owned by any minority of the window.

        The median now also carries the settling transient out of the number rather than only an artifact,
        which is the same property doing a second job: the drift-follower's ramp is a minority of a full
        window by the time the window is full.
        """
        now = self.signal_seconds
        window = self.calibration_seconds
        self._cal_squares.append((now, value * value))
        while self._cal_squares and self._cal_squares[0][0] < now - window:
            self._cal_squares.popleft()
        if self._settled_at is None and self.settled:
            self._settled_at = now

        done = self._settled_at is not None and now - self._settled_at >= window
        if not done and now - self._cal_at < CALIBRATION_REFRESH_SECONDS:
            return                  # see `CALIBRATION_REFRESH_SECONDS`: a 90 s median does not move faster
        self._cal_at = now
        squares = [sq for _at, sq in self._cal_squares]
        rms = float(np.median(squares)) ** 0.5 if squares else 0.0
        self.set_range(max(MIN_RANGE_UV, RANGE_SIGMAS * rms))
        self.calibrated = done

    @property
    def feedback_range_uv(self) -> float:
        """What maps onto the level, ±µV, **read off the scaling** rather than held beside it.

        The calibration writes it and the Feedback tab's scaling can edit it, so a second stored copy could
        disagree with the one actually in use. It was worse than that while the *plot* read this number too:
        a display choice then silently changed what the mechanisms mapped over. The plot has `wave_uv` of its
        own now, and this is the feedback's alone.

        Only meaningful under a fixed range. The rolling scalings take their ends from the signal, so what
        this reports there is the calibrated figure they were switched away from.
        """
        strategy = self._strategy
        if isinstance(strategy, normalise.FixedRange):
            return max(abs(strategy.low), abs(strategy.high), MIN_RANGE_UV)
        return max(self.range_uv, MIN_RANGE_UV)

    def set_range(self, half_range_uv: float) -> None:
        """The feedback scale, ±µV. Rebuilds a fixed-range scaling in place; leaves a rolling one alone,
        since its ends come from the signal rather than from here."""
        self.range_uv = max(MIN_RANGE_UV, float(half_range_uv))
        if self.strategy_id == "fixed_range":
            self.set_strategy("fixed_range", low=-self.range_uv, high=self.range_uv)

    # --- where the level comes from ---------------------------------------------------------

    def set_direction(self, direction: str) -> None:
        """Which side of its own baseline is being trained toward.

        **On a signed wave this is polarity rather than a clinical aim**, and the row says so: in a bipolar
        montage which way is "up" is decided by which electrode went in which socket. It is offered because
        that makes it arbitrary, not because one direction is better.
        """
        if direction not in ("increase", "decrease") or direction == self.direction:
            return
        self.direction = direction
        self._signal = None
        self.signal_target.reset_stats()
        self.signal_target._settled = True

    @property
    def inverts(self) -> bool:
        return self.direction == "decrease"

    def set_strategy(self, strategy_id: str, **params) -> None:
        if strategy_id not in normalise.STRATEGIES:
            return
        if strategy_id == "fixed_range" and not params:
            params = {"low": -self.range_uv, "high": self.range_uv}
        self.strategy_id = strategy_id
        self.strategy_params = dict(params)
        self._strategy = normalise.build(strategy_id, **params)
        self._signal = None
        self.signal_target.reset_stats()
        self.signal_target._settled = True

    @property
    def strategy(self) -> normalise.Strategy:
        return self._strategy

    @property
    def signal(self) -> float | None:
        """The normalised position, or None while the filters are settling or the range is being measured.
        **Not zero** — zero is a reading, and a mechanism handed it would drive itself to its floor."""
        return self._signal

    @property
    def level_source(self) -> tuple[str, str]:
        scale = normalise.STRATEGIES[self.strategy_id].label
        aim = "below" if self.inverts else "above"
        return (f"the {self.frequency_hz:g} Hz wave on ch{self.channel}, {aim} its own baseline · "
                f"scaled by {scale}",
                f"half = the baseline itself — {self._strategy.describe()}")

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

    @property
    def state_balance(self) -> float | None:
        return self._signal

    def key_for(self, tgt: Target):
        return ("wave",)

    def value_for(self, tgt: Target) -> float:
        return self._signal if self._signal is not None else CONDITION_LEVEL

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
        if ch == self.channel:
            # **Unfiltered, and on purpose**: the slow spectrum's whole job is to show what is down there,
            # including the drift the de-trender removes from the measurement. Filtering it first would draw
            # a picture of our own filter.
            self.slow.push(values)

    def consume(self, block) -> set:
        """Feed the metric, then the scaling. The key is reported as soon as there is a reading at all.

        **It used to be withheld until the filters had settled and the range had been measured**, which at
        0.05 Hz is about four and a half minutes of a screen that says only how much longer it will be
        saying that. The reading exists from the first block; what did not exist yet was a range somebody
        had blessed. Withholding the key was how that was expressed to machinery — `TrainingSession.run`
        observes a target when its key is fresh — and it is the wrong thing to express. A number computed
        from what has arrived so far, shown while it improves, is worth more than a correct number nobody
        can see, and the filters settle in front of the operator either way. See `_observe_calibration`,
        which now refines the range instead of waiting to declare one.

        `valid` is still a gate, and it is a different kind: a railed reading is not a rough estimate of a
        wave, it is a reference electrode that has come off.

        Artifact rejection *holds* the level rather than zeroing it: a rejected block arrives here as the
        gate's stand-in at the last accepted value, so the filters advance and the wave coasts. On a
        position signal that is the only safe behaviour — dropping to zero is a jump the trainee hears as
        a plunge, where a stale reading for a few hundred milliseconds is not.

        Device-invalid readings are a different door and get the same holding treatment explicitly. The
        gate passes railed blocks through unjudged (rightly — see `ArtifactGates.pass_through`), so their
        readings land here marked `valid=False`: an 800 µV rail must not steer the level, and above all
        must not be *calibrated on* — a reference electrode lifting during the measurement window would
        otherwise set the session's range from the rail.
        """
        metric = self.metric
        if metric is None or block.channel != self.channel:
            return set()
        fresh: set = set()
        for r in metric.feed(block):
            self.raw_wave = r.value
            self.amplitudes[("wave",)] = r.value
            if not r.valid:
                continue
            if not self.calibrated:
                self._observe_calibration(r.value)
            signal = self._strategy.push(r.value, r.t)
            if self.inverts:
                signal = 1.0 - signal
            self._signal = signal
            fresh.add(("wave",))
        if self.wave is not None:
            # Fed from the metric rather than from the readings, because the waveform is decimated at its
            # own rate: the readings arrive at twenty a second and the plot's stride is its own.
            self.wave.feed(metric.wave_start, {self.channel: metric.waveform})
        return fresh

    # --- surviving a stop -------------------------------------------------------------------

    def adopt_targets(self, previous) -> None:
        """The one target is this screen's rather than an operator's list, so it is rebuilt. What an operator
        set is where the instrument points, which is carried across and applied by rebuilding — a metric
        assigned as an object would have no filters behind it."""
        self.set_frequency(getattr(previous, "frequency_hz", DEFAULT_HZ),
                           order=getattr(previous, "order", self.order),
                           channel=getattr(previous, "channel", self.channel))

    def adopt_view(self, previous) -> None:
        super().adopt_view(previous)
        self.wave_uv = getattr(previous, "wave_uv", self.wave_uv)
        self.direction = getattr(previous, "direction", self.direction)
        # **The scaling comes across; its calibrated *ends* do not.** A fixed range's ends were measured on
        # the session that ended — the same category as a threshold, which `prefs` refuses to carry for the
        # same reason — so this rebuilds them from the new session's own range and lets the calibration
        # measure it again.
        strategy_id = getattr(previous, "strategy_id", self.strategy_id)
        params = {} if strategy_id == "fixed_range" else getattr(previous, "strategy_params", {})
        self.set_strategy(strategy_id, **params)

    # --- what this protocol requires --------------------------------------------------------

    def wiring_problem(self) -> str | None:
        """Why the numbers would be wrong, worst first.

        The corner check is not a nicety here: the filter has no high-pass at all, so the device's corner *is*
        the band's lower edge, and at 0.5 Hz against a band stopping at 0.1 Hz there is no band left. A
        single-pole high-pass passes `f/sqrt(f² + fc²)`, and unlike the synchrony screen's product that is not
        squared — this is one channel, so the loss is first order.
        """
        if self.metric is None:
            return "no channel to measure"
        if self.corner_hz > self.metric.high / 3:
            top = self.metric.high
            kept = top / (top ** 2 + self.corner_hz ** 2) ** 0.5
            # Short, because it sits in a 380 px column and a six-line warning is read as noise. The
            # arithmetic and the reason the filtering cannot be fixed in software are behind the `?`.
            return (f"the device's {self.corner_hz:g} Hz corner is above this band's top edge of "
                    f"{top:g} Hz, so about {kept:.0%} of the wave survives — set High-pass to "
                    f"{BEST_CORNER_HZ:g} Hz on the device screen")
        if self.device is not None:
            reference = getattr(self.device.status.settings, "reference", "")
            if reference and reference != "bipolar":
                return (f"reference is '{reference}', and at this frequency a pair of nearby sites is "
                        f"safer — the dominant nuisance below 0.1 Hz is electrode drift, which a bipolar "
                        f"pair largely shares and cancels while a distant reference contributes its own in "
                        f"full")
        return None


def session_for(*args, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None,
                **kwargs) -> WaveSession:
    _ = heart_choice, heart_runner, mendi_choice, mendi_runner, pupil_choice, pupil_runner
    return WaveSession(*args, **kwargs)


def _duration(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 90 else f"{seconds:.0f} s"


# --- the card ----------------------------------------------------------------------------------

class WaveCard:
    """The one target, as a card — the same grammar `target_card.TargetCard` uses, not the same class.

    **What is borrowed** is the layout, top to bottom: an identity row, then the value beside what it is
    compared against, then a bar that states its own range, then the scale line with the reaction time at its
    right, then the row where a threshold's mode would be, then a note that appears only when there is
    something to say. Reading two protocol screens should not mean learning two layouts, and this was a
    column of `labelled` rows that looked like a settings pane rather than like the thing being trained.

    **Why not the class itself**, which would have been better if it fitted: three of its parts cannot work on
    this quantity.

    - Its bar runs 0 to a ceiling with the threshold as a notch. This value is **signed** — which side of the
      baseline it is on is the entire content — so the bar here is centre-zero and fills either way. On the
      shared bar a negative reading is a negative width.
    - Its threshold has −/+ nudges. The threshold here is the baseline, which the de-trender has already put
      at zero; nudging zero by 10% leaves it at zero, and there is nothing else it could mean.
    - Its mode row offers fixed/automatic and a redefine. What is derived from the signal here is the
      *range* rather than a threshold, so what belongs in that row is the calibration and its countdown.

    Adding two flags to the shared card to switch off half of it would have left the flags standing in for
    this explanation.
    """

    def __init__(self, page: ft.Page, session: WaveSession, on_change: Callable[[], None]):
        self.page = page
        self.session = session
        self.on_change = on_change

        self.identity = ft.Row(spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.value_text = ft.Text("—", size=19, color=t.TEXT, weight=ft.FontWeight.W_400)
        self.swing_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.pct_text = ft.Text("", size=t.SMALL, color=t.MUTED)
        # A **signed** bar: the middle is the baseline and the fill grows from it, either way. `left` moves
        # as well as `width`, which is what lets one container draw both directions.
        self.fill = ft.Container(width=0, height=8, border_radius=4, bgcolor=session.colour,
                                 left=BAR_W / 2, bottom=0)
        self.notch = ft.Container(width=2, height=14, bgcolor=t.TEXT, left=BAR_W / 2 - 1, bottom=-3)
        bar = ft.Stack([ft.Container(width=BAR_W, height=8, border_radius=4, bgcolor=t.SURFACE_HI,
                                     left=0, bottom=0),
                        self.fill, self.notch], width=BAR_W, height=14)
        self.scale_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.latency_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.state_row = ft.Row(spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.note = ft.Text("", size=t.SMALL, color=t.WARN)
        # Two lines reserved whether or not there is a note — this card's notes run to two lines, and a
        # card whose height follows its note moves everything under it. See `target_card.NOTE_H`.
        self.note_box = ft.Container(content=self.note, height=2 * NOTE_H,
                                     alignment=ft.Alignment.BOTTOM_LEFT)
        self.filter_text = ft.Text("", size=t.SMALL, color=t.FAINT)

        self.control = ft.Container(
            content=ft.Column([
                self.identity,
                ft.Container(height=8),
                ft.Row([self.value_text, ft.Container(width=6), self.swing_text,
                        ft.Container(expand=True),
                        ft.Text("threshold", size=t.SMALL, color=t.FAINT),
                        ft.Text("the baseline", size=t.BODY, color=t.TEXT)],
                       spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=9),
                ft.Row([bar, ft.Container(width=12), self.pct_text], spacing=0,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=4),
                ft.Row([ft.Container(content=self.scale_text, expand=True), self.latency_text],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=4), self.filter_text,
                ft.Container(height=7),
                self.state_row,
                self.note_box,
            ], spacing=0),
            bgcolor=t.BG, border=ft.Border.all(1, t.HAIRLINE), border_radius=8,
            padding=ft.Padding(12, 11, 12, 11))
        self.rebuild()

    def rebuild(self) -> None:
        """The identity row and the range row, which change when what is measured does.

        **The identity row has to fit a 380 px column**, and the first version did not: four channel pills
        and two order pills either side of the frequency box pushed the order off the right-hand edge, where
        a control that is present and invisible is worse than one that is absent. A channel is a menu, as it
        is on the amplitude screen, and the order goes behind a gear, which is where per-target filter
        settings go everywhere else — the same convention, and it takes the consequences of the choice with
        it rather than leaving two pills to speak for themselves.
        """
        session = self.session
        self.identity.controls = [
            # The frequency's own colour, which is also the trace's and the legend's — see
            # `WaveSession.colour`. Three places showing one signal have to agree about it.
            ft.Container(width=3, height=18, bgcolor=session.colour, border_radius=2),
            ft.PopupMenuButton(
                content=ft.Container(
                    content=ft.Row([ft.Text(f"ch{session.channel}", size=t.BODY, color=t.TEXT),
                                    ft.Text("▾", size=9, color=t.FAINT)], spacing=6),
                    padding=ft.Padding(9, 5, 9, 5), border_radius=6,
                    bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
                items=[ft.PopupMenuItem(
                    content=ft.Text(f"ch{c}", size=t.BODY),
                    on_click=(lambda c_=c: lambda _e: self._retune(channel=c_))())
                    for c in session.specs]),
            num_box(session.frequency_hz, lambda v: self._retune(hz=v), suffix="Hz",
                    step=grid_hz(session.frequency_hz), low=HZ_RANGE[0], high=HZ_RANGE[1],
                    width=64),
            ft.Container(expand=True),
            settings_button(self.page, "Filter", lambda redraw: self._filter_rows(redraw)),
        ]
        # Where the threshold's mode pills would be. What is derived from the signal here is the *range*, so
        # this row is the calibration: the number it produced, editable, and the button that measures it
        # again.
        #
        # **"full at ±", not "level range".** The old label named the quantity from the inside — a range, of
        # the level, in microvolts — which says what it *is* and not what it *does*, and what it does is
        # decide how large an excursion earns full output. The synchrony screen names its own output window
        # the same way for the same reason: "silent at 65% of threshold, full at 100%". A control whose name
        # needs a paragraph is the paragraph's fault.
        #
        # **And no note on the row**, which is `labelled`'s own rule and was learned again here: "calibrated
        # from the signal" wrapped to three lines beside a stepper, and the row became unreadable saying
        # something the `?` already says. The button is a word rather than a `↺` for the same reason.
        self.state_row.controls = [
            ft.Text("full at ±", size=t.SMALL, color=t.MUTED, width=76),
            num_box(session.feedback_range_uv, self._set_range, suffix="µV", step=1,
                    low=MIN_RANGE_UV, width=52),
            ft.Container(width=4),
            pill("recalibrate", False, lambda _e: (session.restart_calibration(), self.rebuild(),
                                                   self.on_change())),
            ft.Container(expand=True),
            help_button(self.page, "What is measured", SIGNAL_HELP),
        ]
        self.page.update()

    def _set_range(self, uv: float) -> None:
        """The one editor for the feedback range, and it is here because here is where the wave is.

        The scaling on the Feedback tab reports it instead of offering a second pair of boxes: it is the same
        quantity, and two editors for one number is how a screen comes to disagree with itself.
        """
        self.session.set_range(uv)
        self.rebuild()
        self.on_change()

    def _filter_rows(self, redraw: Callable[[], None]) -> list:
        """The filter gear, in the shape `target_card.filter_rows` gives every other screen: the parameter
        this filter actually has, and what choosing it costs.

        One parameter, because there is one: the band is the frequency doubled and the de-trender's corner is
        the frequency divided, so the only thing left to choose is how steep the lowpass is. Both
        consequences are printed, since the reason to raise the order is skirt steepness and the reason not to
        is that this screen already waits minutes before it can feed anything back.
        """
        session = self.session
        metric = session.metric

        def apply(order: int) -> None:
            session.set_frequency(order=order)
            self.rebuild()
            self.on_change()
            redraw()

        rows: list[ft.Control] = [
            labelled("measures",
                     ft.Text(f"below {session.band_edges[1]:g} Hz on ch{session.channel}",
                             size=t.BODY, color=t.TEXT), width=64),
            labelled("order",
                     *[pill(str(n), session.order == n, (lambda o=n: lambda _e: apply(o))())
                       for n in range(ORDER_RANGE[0], ORDER_RANGE[1] + 1)],
                     note="how steeply it falls above the band", width=64),
        ]
        if metric is not None:
            rows += [
                ft.Container(height=4),
                ft.Text(f"reacts in {metric.delay_seconds:.1f} s", size=t.BODY, color=t.TEXT),
                ft.Text(f"the filter's group delay. Fast protocols react within 250 ms; this protocol "
                        f"reacts more slowly, and that is normal here",
                        size=t.SMALL, color=t.MUTED),
                ft.Text(f"settles in {_duration(metric.settle_seconds)}. The drift follower needs "
                        f"{_duration(metric.settle_seconds - metric.band_settle_seconds)} of that time",
                        size=t.SMALL, color=t.MUTED),
            ]
        return rows

    def _retune(self, **kwargs) -> None:
        self.session.set_frequency(**kwargs)
        self.rebuild()
        self.on_change()

    def refresh(self) -> None:
        session = self.session
        metric, value = session.metric, session.raw_wave
        full = max(session.feedback_range_uv, MIN_RANGE_UV)
        if value is None:
            self.value_text.value = "—"
            self.value_text.color = t.FAINT
            self.swing_text.value = "µV"
            self.fill.width = 0
        else:
            met = session.feedback.on_target(session.channel_targets(CONDITION))
            self.value_text.value = f"{value:+.2f}"
            self.value_text.color = t.GOOD if met else t.TEXT
            self.swing_text.value = (f"µV · swing {metric.rms:.1f} rms" if metric is not None
                                     else "µV")
            half = BAR_W / 2
            span = half * min(1.0, abs(value) / full)
            self.fill.width = span
            self.fill.left = half if value >= 0 else half - span
        # **The mapping, not the bounds.** "−20 to +20 µV · middle = the baseline" describes the bar; what an
        # operator needs is what those ends *do*, which is the whole of what this range is for. Three numbers
        # and three percentages, in one line, on the row the bar already occupies.
        # Three significant figures: a calibrated range printed as 16.2177 is false precision, and it
        # was long enough to push the reaction time off the card.
        self.scale_text.value = (f"0% at −{full:.3g} · 50% at the baseline · 100% at +{full:.3g} µV")
        self.filter_text.value = metric.describe() if metric is not None else ""
        latency = session.latency_for(session.signal_target)
        self.latency_text.value = ("" if latency is None else
                                   f"reacts in {latency:.1f} s" if latency >= 1.0 else
                                   f"reacts in {latency * 1000:.0f} ms")
        self.latency_text.tooltip = (
            "The filter's group delay. Fast protocols react within 250 ms; this protocol reacts more "
            "slowly, and that is normal here.")
        recent = session.signal_target.recent_percent()
        self.pct_text.value = ("waiting" if recent is None else
                               f"{recent:.0f}% of the last {RECENT_WINDOW:g} s on the wanted side")
        # The settling and the calibration, in the place the shared card puts its own calibration note.
        # **Both of these describe a number that is in use, not a wait.** They used to be the two phases
        # nothing was fed back during, and the wording followed: "the filters need 2.9 min". The feedback
        # runs throughout now, so what is left to say is how much the range is still moving — which is
        # worth saying, and is a different sentence.
        if metric is None:
            self.note.value = "no channel to measure"
        elif not session.settled:
            self.note.value = (f"filters {metric.warmup:.0%} settled. The range updates as more signal "
                               f"arrives")
        elif session.calibrating:
            self.note.value = (f"refining the range: {_duration(session.calibration_remaining)} "
                               f"left of {_duration(session.calibration_seconds)} total")
        else:
            self.note.value = ""


# --- where the level comes from, as a panel ----------------------------------------------------

class LevelSource:
    """The two rows the shared Feedback pane cannot supply: which way, and on what scale.

    *What* is measured is on the Train tab, on the card, because it is what you change while looking at the
    wave. The scaling is here because this is where the level is configured on every other screen — and its
    ends are the calibrated range, which is why the card reports that number instead of setting it.
    """

    def __init__(self, page: ft.Page, session: WaveSession):
        self.page = page
        self.session = session
        self.rows = ft.Column(spacing=12)
        self.readout = ft.Text("", size=t.BODY, color=t.TEXT)
        self.note = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.control = panel(
            "LEVEL FROM",
            self.readout, ft.Container(height=4), self.note,
            ft.Container(height=14), self.rows,
            help=help_button(page, "Where the level comes from", SIGNAL_HELP))
        self.rebuild()

    def rebuild(self) -> None:
        session = self.session
        self.rows.controls = [
            labelled("aim",
                     *[pill(k, session.direction == k,
                            (lambda k_=k: lambda _e: (session.set_direction(k_), self.rebuild()))())
                       for k in ("increase", "decrease")],
                     note="which side of its own baseline the wave trains toward. In a bipolar pair, "
                          "the electrode wiring sets this direction",
                     width=76),
            labelled("scaled by",
                     *[pill(normalise.STRATEGIES[k].label, session.strategy_id == k,
                            (lambda k_=k: lambda _e: (session.set_strategy(k_), self.rebuild()))())
                       for k in normalise.STRATEGIES],
                     *self._params(),
                     note=normalise.STRATEGIES[session.strategy_id].description, width=76),
        ]
        self.page.update()

    def _params(self) -> list:
        """Only the knobs the selected scaling has, and only the ones this screen owns *here*: a fixed range's
        ends are the calibration's, so they are stated with a pointer at the card that sets them."""
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
            # **Reported, not offered.** Its ends are the calibrated range, and that has one editor: the card,
            # beside the wave it was measured from. A second pair of boxes here would be two controls for one
            # number, which is how a screen comes to disagree with itself — and it is also what made the
            # operator hunt for the range on the wrong tab.
            out += [ft.Text(f"{strategy.low:g} to {strategy.high:g} µV", size=t.BODY, color=t.TEXT),
                    ft.Container(width=6),
                    ft.Text("calibrated · set on the card", size=t.SMALL, color=t.FAINT)]
        if isinstance(strategy, (normalise.Saturating, normalise.Tanh)):
            out += [ft.Text("at", size=t.SMALL, color=t.FAINT),
                    num_box(strategy.scale, lambda v: self._set(scale=v), suffix="µV", step=1,
                            low=0.1)]
        return out

    def _set(self, **params) -> None:
        session = self.session
        session.set_strategy(session.strategy_id, **{**session.strategy_params, **params})
        self.rebuild()

    def refresh(self) -> None:
        session = self.session
        raw, signal = session.raw_wave, session.signal
        if raw is None:
            self.readout.value = "—"
            self.readout.color = t.FAINT
            self.note.value = "waiting for data"
            return
        self.readout.value = (f"{raw:+.2f} µV"
                             + (f"  →  {signal:.0%}" if signal is not None else "  →  —"))
        self.readout.color = t.TEXT if signal is not None else t.FAINT
        source, detail = session.level_source
        self.note.value = f"{source} · {detail}"


# --- the screen --------------------------------------------------------------------------------

def view(page: ft.Page, session: WaveSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    """Build once, mutate on Flet's loop."""

    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width(reserve: int = SIDE_W) -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - reserve - 70)

    spectrum = spectrum_pane.SpectrumPane(page, session, plot_width)

    trends = trends_pane.TrendsPane(page, session, plot_width)

    wave_canvas = cv.Canvas(shapes=[], expand=True, height=WAVE_H)
    wave_axis, wave_labels = value_axis(WAVE_H, count=5, width=AXIS_W)
    wave_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    wave_state = ft.Text("", size=t.SMALL, color=t.WARN)
    wave_pills = ft.Row(spacing=6)
    # **Which signal is on the plot**, the way the synchrony screen's SLOW SIGNAL panel says it: a swatch in
    # the trace's own colour and the name of what it is. The caption underneath says how it is drawn — the
    # window, the scale, the drift removed — and that is a different question from *what* is drawn. Without
    # this the pane was one anonymous line, on a screen where four channels and a frequency are selectable.
    wave_legend = ft.Row(spacing=12)

    level_fill = ft.Container(width=0, height=LEVEL_H, border_radius=5, bgcolor=t.ACCENT_DIM,
                             left=0, bottom=0)
    level_mark = ft.Container(width=1.5, height=LEVEL_H, bgcolor=t.MUTED, left=0, bottom=0)
    level_track = ft.Stack([ft.Container(width=0, height=LEVEL_H, border_radius=5,
                                        bgcolor=t.SURFACE_HI, left=0, bottom=0),
                            level_fill, level_mark], height=LEVEL_H)
    level_text = ft.Text("", size=t.BODY, color=t.TEXT)
    level_note = ft.Text("", size=t.SMALL, color=t.FAINT)

    slow_canvas = cv.Canvas(shapes=[], expand=True, height=SLOW_H)
    slow_axis, slow_labels = value_axis(SLOW_H, count=4, width=AXIS_W)
    slow_ticks_hz = (0.01, 0.05, 0.1, 0.5, 1.0)
    slow_tick_axis, slow_tick_labels = tick_axis(slow_ticks_hz, tick_width=34)
    slow_caption = ft.Text("", size=t.SMALL, color=t.FAINT)

    traces = trace_rows.TraceRows(page, session, plot_width, height=TRACE_H)
    wiring_note = ft.Text("", size=t.SMALL, color=t.WARN)

    source_ui = LevelSource(page, session)
    signal = signal_pane.SignalPane(page, session, inset=False)

    def after_retune() -> None:
        source_ui.rebuild()
        signal.rebuild_contributions(session.targets)
        signal.rebuild()

    card = WaveCard(page, session, after_retune)

    # ---- the plot's own scale, which is not the feedback's ----
    def set_wave_gain(uv: float) -> None:
        session.wave_uv = uv
        build_wave_pills()

    def fit_wave(_e=None) -> None:
        """Pick the smallest preset that holds what the wave is actually doing.

        Not autoscaling: it *selects a preset*, so the caption still names one fixed number and a picture can
        be compared with one from yesterday. Same control and same reasoning as the raw trace's `fit` — and
        here it is what replaced the plot borrowing the feedback range, which made a display choice change
        what the mechanisms mapped over.
        """
        peak = session.wave.peak() if session.wave is not None else 0.0
        wanted = peak * WAVE_HEADROOM
        session.wave_uv = next((g for g in WAVE_GAINS if g >= wanted), WAVE_GAINS[-1])
        build_wave_pills()

    def build_wave_pills() -> None:
        wave_pills.controls = [
            *[pill(f"±{g:g}", session.wave_uv == g, (lambda v=g: lambda _e: set_wave_gain(v))())
              for g in WAVE_GAINS],
            ft.Container(width=8),
            pill("fit", False, fit_wave),
        ]
        page.update()

    train_left = ft.Column([
        panel("WAVE",
              ft.Row([wave_axis, ft.Container(content=ft.Stack(
                  [ft.Container(content=wave_canvas, expand=True),
                   ft.Container(content=wave_state, alignment=ft.Alignment.CENTER, expand=True)],
                  expand=True, height=WAVE_H), expand=True, height=WAVE_H)], spacing=0),
              ft.Container(height=6),
              ft.Row([wave_legend, ft.Container(expand=True)]),
              ft.Container(height=4), wave_caption,
              trailing=wave_pills,
              help=help_button(page, "The wave", WAVE_HELP)),
        ft.Container(height=12),
        panel("LEVEL",
              ft.Row([level_text, ft.Container(width=14), level_note],
                     vertical_alignment=ft.CrossAxisAlignment.CENTER),
              ft.Container(height=8), level_track,
              help=help_button(page, "Settling and calibration", SETTLING_HELP)),
        ft.Container(height=12),
        panel("RAW", *traces.rows, ft.Container(height=6), traces.caption),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    preset_row = ft.Row(spacing=6)

    def build_presets() -> None:
        preset_row.controls = [
            pill(name, abs(np.log(max(session.frequency_hz, 1e-9) / hz)) < 0.02,
                 (lambda h=hz: lambda _e: (session.set_frequency(hz=h), card.rebuild(),
                                           build_presets(), after_retune()))())
            for name, hz in PRESETS]
        page.update()

    train_right = ft.Column([
        panel("TRAINING", card.control, ft.Container(height=8), wiring_note,
              trailing=preset_row),
        ft.Container(height=12),
        panel("SLOW SPECTRUM",
              ft.Row([slow_axis, ft.Container(content=slow_canvas, expand=True, height=SLOW_H)],
                     spacing=0),
              ft.Row([ft.Container(width=AXIS_W), ft.Container(content=slow_tick_axis, expand=True)],
                     spacing=0),
              ft.Container(height=4), slow_caption,
              help=help_button(page, "The slow spectrum", SLOW_HELP)),
    ], spacing=0, width=SIDE_W, scroll=ft.ScrollMode.AUTO)

    train_tab = ft.Container(content=ft.Row([train_left, train_right], spacing=14, expand=True),
                            padding=ft.Padding(14, 12, 14, 14), expand=True)
    feedback_tab = ft.Container(
        content=ft.Column([source_ui.control, ft.Container(height=12), signal.control],
                          spacing=0, expand=True, scroll=ft.ScrollMode.AUTO),
        padding=ft.Padding(14, 12, 14, 14), expand=True, visible=False)
    panes = [train_tab, spectrum.control, trends.control, feedback_tab, plan_pane_.control]

    def select_tab(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs(["Train", "Spectrum", "Trends", "Feedback", "Session"], select_tab)

    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        source_ui.refresh()
        card.refresh()
        signal.refresh()
        spectrum.refresh()
        trends.refresh()

        metric, ring = session.metric, session.wave
        width = plot_width()
        full = max(WAVE_FLOOR_UV, session.wave_uv)
        # **Its own colour throughout.** It was muted until the range was calibrated, which was honest then
        # — the trace really was a picture of a signal nothing was fed back from — and is a lie now. The
        # wave drives the level from the first reading, so greying it would say the opposite.
        colour = session.colour
        wave_legend.controls = [
            ft.Row([ft.Container(width=10, height=3, bgcolor=colour, border_radius=2),
                    ft.Text(f"ch{session.channel} · {session.frequency_hz:g} Hz wave"
                            + ("" if session.calibrated else ", on a range still being refined"),
                            size=t.SMALL, color=t.MUTED)], spacing=6)]
        if ring is not None:
            wave_canvas.shapes = signed_shapes(
                [ring.series()[session.channel]], width, WAVE_H, full,
                [colour], pushed=ring.pushed)
            place_value_labels(wave_labels, WAVE_H, full, bottom=-full, unit="µV")
            # What the de-trender is taking out, as a number: on the wave's own axis it is off-screen by a
            # factor of fifteen, so a trace of it would be a line along the top of the pane.
            drift = ""
            if metric is not None:
                corner = session.frequency_hz / DETREND_RATIO
                drift = f" · drift removed: {metric.trend:+.0f} µV above {corner:.3g} Hz"
            # **"full scale" here, "level range" on the card**, and both are µV: they are two
            # different scales and the words are the only thing that says so on a glance.
            wave_caption.value = (f"{ring.cycles} cycles = {_duration(ring.seconds)} across · "
                                  f"±{full:g} µV full scale, this plot's own · middle line = the "
                                  f"baseline and the threshold{drift}")
        # The empty state is the settling and the calibration, and it goes over the plot: a slow wave that
        # has not filled its window is a nearly blank pane, which is what a broken screen looks like.
        # Over the plot only while there is nothing on it. Once the wave is drawing, an overlay saying the
        # range is still moving is a sentence across a working instrument — the card's note says it.
        if metric is None:
            wave_state.value = "no channel to measure"
        elif ring is None or not ring.pushed:
            wave_state.value = "waiting for the first reading"
        else:
            wave_state.value = ""

        # the level, with the threshold marked on it
        bar_w = max(120.0, width - 40)
        level_track.width = bar_w
        level_track.controls[0].width = bar_w
        level_mark.left = bar_w * CONDITION_LEVEL
        signal_value = session.signal
        level_fill.width = bar_w * (signal_value if signal_value is not None else 0.0)
        level_fill.bgcolor = t.ACCENT if session.feedback.on_target(
            session.channel_targets(CONDITION)) else t.ACCENT_DIM
        if signal_value is None:
            level_text.value = "—"
            level_text.color = t.FAINT
            level_note.value = "waiting for the first reading from the channel"
        else:
            level_text.value = f"{session.level:.0%}"
            level_text.color = t.TEXT
            level_note.value = (f"{session.raw_wave:+.2f} µV, full at "
                                f"±{session.feedback_range_uv:.3g} µV · mark = the baseline")

        # the slow spectrum
        centres, amplitudes = session.slow.bins(SLOW_BINS)
        if amplitudes.size:
            peak = nice_ceiling(max(float(amplitudes.max()), 1e-3))
            x_of = _log_x(SLOW_LO_HZ, SLOW_HI_HZ, width, SLOW_BINS)
            slow_canvas.shapes = list(bars(
                amplitudes, width, SLOW_H,
                lambda i: t.ACCENT if _near(centres[i], session.frequency_hz) else t.MUTED,
                peak=peak))
            place_value_labels(slow_labels, SLOW_H, peak, unit="µV")
            place_tick_labels(slow_tick_labels, slow_ticks_hz, x_of, tick_width=34, width=width)
            # **The resolution is the honest part and it is stated, not waited for.** The bins below what
            # the fill currently resolves come back empty and simply are not drawn, so the picture fills in
            # from the right — see `SlowRing.MIN_SAMPLES`, which used to withhold all of it for 6.7 min.
            slow_caption.value = (
                f"{SLOW_LO_HZ:g}–{SLOW_HI_HZ:g} Hz · {_duration(session.slow.filled_seconds)} of "
                f"{_duration(session.slow.window_seconds)} filled · "
                f"{session.slow.resolution_hz:.3g} Hz bins so far · "
                f"lit bin = {session.frequency_hz:g} Hz")
        else:
            slow_canvas.shapes = []
            slow_caption.value = "filling · wait a few seconds for enough data"

        # raw traces. The extra clause is this screen's alone: what it trains is *below* the band drawn
        # here, so a reader who took this picture for the trained signal would be reading the wrong thing.
        traces.refresh(session.trace_seconds,
                       extra=("the trained wave is below this band and has its own pane",))

        problem = session.wiring_problem()
        wiring_note.value = problem or ""
        wiring_note.visible = bool(problem)
        page.update()

    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)

    build_wave_pills()
    build_presets()
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)


def _log_x(lo_hz: float, hi_hz: float, width: float, bins: int):
    """Where a frequency's gridline goes when the bins are log-spaced.

    The linear counterpart is `plots.bin_centre_x`, and the same rule applies: a bin is *centred* on its
    frequency, so the label for 0.1 Hz has to land in the middle of the cell that holds it rather than at its
    left edge. Off by half a cell reads as the spectrum being shifted.
    """
    span = np.log(hi_hz / lo_hz)
    step = width / max(1, bins)

    def x_of(hz: float) -> float | None:
        if hz < lo_hz or hz > hi_hz:
            return None
        fraction = np.log(hz / lo_hz) / span
        return float(fraction * (bins - 1) * step + step / 2)

    return x_of


def _near(centre_hz: float, target_hz: float) -> bool:
    """Whether a log-spaced bin is the one holding the trained frequency. Within a quarter of a decade,
    which is about one bin at this spacing."""
    return abs(np.log(max(centre_hz, 1e-9) / max(target_hz, 1e-9))) < 0.12
