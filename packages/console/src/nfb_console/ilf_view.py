"""ILF Synchrony.

Two bipolar sites and **one** reward: the product of what the two sites are doing, at one frequency you
retune. This is Othmer's infra-low-frequency synchrony method.

What matters here is the shape:

**One instrument, not three.** The measure takes a single reward frequency. So "ILF Synchrony trains 0.05 Hz,
alpha and gamma" describes three *settings* of one measure, not three concurrent rewards — you point it
somewhere and train that. `PRESETS` are those three places; the frequency box is the actual control.

**The measure is `nfb.dsp.ProductSynchrony`** and its own module docstring carries the arithmetic. The one
thing to know before reading anything else on this screen: the filter degrades to a **bare lowpass**
whenever `centre − bandwidth/2 ≤ 0`, so with the 3 Hz bandwidth a 0.05 Hz centre is a 0–1.55 Hz lowpass and
the reward is the product of two *slow waveforms*. That is what ILF on the raw waveform means, and it is why
nothing on this screen needs a special case for the slow end: a 0–1.55 Hz band is wide, fast to measure
(171 ms, against 8.2 s for a genuinely narrow band there) and gives an auto-threshold plenty to work with.

**What the reward drives.** The reward produces a continuous level, the fraction of time above threshold and
a pass/fail mark, and **no sound of its own**. So the arrangement here is the proportional master volume plus
a discrete mark. The Shepard tone in `music.ShepardPlayer` belongs to the slow-wave reward, a sibling screen
this one does not implement; it is offered here but off, because borrowing another instrument's feedback and
presenting it as this one's is the thing that made an earlier version of this screen wrong.

**Nine inhibit bands per channel**, ANDed in front of the reward — which is why there is no alert: eighteen
bands aimed at 95% put something over threshold most of the session. Gating is what `on_target` already does.

**The slow-wave reward is a screen of its own.** It is a level crossing on the *signed* slow waveform with
drift compensation attached and the Shepard tone hanging off it. That is a different computation from
synchrony and it is `wave_view`, not this file. The SLOW SIGNAL panel below still draws the signal it trains,
because that picture is worth having beside a synchrony reward; what it is *not* is a reward here, and the
Shepard tone stays off by default on this screen for the same reason.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import (Band, DEFAULT_BANDWIDTH_HZ, DEFAULT_POST_SECONDS, LockinBandPower,
                     PairAligner, ProductSynchrony, PRODUCT_ORDER, is_lowpass)
from nfb.dsp.product import DEFAULT_SLOW_BAND, SLOW_BANDS
from nfb.stream import NO_SIGNAL, OK, Block

from . import chrome, plan_pane, shadow, signal_pane, spectrum_pane, trace_rows, trends_pane
from . import theme as t
from .plots import (EnvelopePlot, StickyCeiling, nice_ceiling, plot_ceiling,
                    place_value_labels, signed_shapes, value_axis, xy_shapes)
from .feedback import LEVEL, STATE
from .protocols import Protocol
from .target_card import TargetCard
from .targets import PAIR, Target
from .traces import WAVE_CYCLES, RawRing, TraceBuffer, WaveRing
from .training import METRIC_HZ, TREND_SECONDS, TrainingSession
from .widgets import grid_hz, help_button, labelled, num_box, panel, pill, snap_hz

HANDLES = ("ILF",)

# The three places the method points this one instrument, and the frequency box is what actually sets it.
# 10 Hz is `SpinnerNumberModelSynch`'s own default; 40 Hz is where the `Reward Band` slider tops out; 0.05 Hz
# is the infra-slow setting reachable against our amplifier's 0.01 Hz corner — see `wiring_problem`.
PRESETS: tuple[tuple[str, float], ...] = (
    ("infra-slow", 0.05),
    ("alpha", 10.0),
    ("gamma", 40.0),
)
DEFAULT_HZ = 10.0           # alpha

SUBHERTZ_CORNER_HZ = 0.05
"""Where a fresh session points when the amplifier can actually reach it.

`DEFAULT_HZ` is a fixed default and cannot know how the machine is set up. The device's high-pass corner is in the config the source is
decoding with, so the screen can tell whether the infra-slow preset is *reachable* before offering it. At the
0.5 Hz corner a 0.05 Hz reward measures the filter's roll-off and `wiring_problem` says so, which is a poor
way to open a screen; at 0.01 Hz it is the setting the corner was chosen for.

So: sub-hertz when the corner allows it, alpha otherwise. Either way a preset is *selected* on entry, which
the fixed 10 Hz did not guarantee once a carried-over frequency could sit between the pills."""
HZ_RANGE = (0.001, 50.0)    # lowest and highest reward frequency, Hz; the box steps by 0.001

BANDWIDTH_RANGE = (1.0, 3.0)    # bandwidth of the synchrony filter pair, Hz
ORDER_RANGE = (1, 2)            # filter order

# A tenth of the time above threshold, where every other reward here asks for 81–85% and ours default to 90.
# It is what makes the reward a rare, distinct event rather than a background state.
REWARD_AIM = 0.10

# The aim of the inhibits: 50–100%, default 95.
INHIBIT_AIM = 0.95

# 50–100%, default **60** — the aim of the *second* reward, the band amplitude
# on the summed channels. Nothing like the product's 10%, because it is a different kind of target: an
# ordinary amplitude you are asked to hold up most of the time, where the product is a rare event.
AMPLITUDE_AIM = 0.60

# The amplitude reward's epoch: 10 s above 0.01 Hz, otherwise 30 s. It is **not** the synchrony measure's,
# which has no epoch input and uses `EPOCH_SECONDS`.
def amplitude_epoch(hz: float) -> float:
    return 10.0 if hz > 0.01 else 30.0

# The synchrony measure's epoch, 1–100 s, fixed at **30**. The amplitude reward's epoch is separate:
# see `amplitude_epoch`.
EPOCH_SECONDS = 30.0

# The nine contiguous bands. Two of them are filled in by contiguity: 8→12 and 18→22 are the only
# values that leave no gap.
#
# The lowest is **0.1–2 and not 0–2**, because a band whose lower edge is DC is not a band.
INHIBITS: tuple[tuple[float, float], ...] = (
    (0.1, 2.0), (2.0, 4.5), (4.5, 8.0), (8.0, 12.0), (12.0, 15.0),
    (15.0, 18.0), (18.0, 22.0), (22.0, 30.0), (30.0, 40.0),
)

# --- what the continuous output is --------------------------------------------------------------
#
# Where the value sits in a window set as percentages of the threshold: `top = RANGE_POSITION%`,
# `bottom = (RANGE_POSITION - RANGE_SIZE)%`, and the output is `(value - bottom) / (top - bottom)`.
# The binary reward is `value >= threshold`.
#
# **The window ends AT the threshold, and that is the part worth not guessing.** A window running from the
# threshold *up* to 1.5x would make the output a measure of how far past the threshold you are. This one is
# better: the number reports your *approach* to earning, is full the moment you earn it, and the discrete
# reward fires at exactly the point the continuous one saturates.
RANGE_POSITION = 100.0   # the top of the window, as a percent of the threshold. 0-200, step 1.
RANGE_SIZE = 35.0        # how far below the top the bottom sits, same units.
RANGE_LIMITS = (0.0, 200.0)     # both controls, step 1

# A sample-and-hold of 2.0 s sits between the post filter and everything downstream. It feeds both the
# threshold and the value, and so the continuous output.
#
# **So the reward updates once every two seconds, not continuously.** The output is a staircase with a two-second tread; the reward can change state at most thirty times a minute before the
# 500 ms interval gate applies at all; and the 30 s threshold window holds about *fifteen* values rather than
# the six hundred a 20 Hz feed puts in it. That last one is not a detail: at a 10% aim, fifteen values means
# the threshold sits near the highest one or two of the last half minute and moves in visible steps.
#
# `feedback.HOLD_FRACTION` records the same sample-and-hold from the alpha-theta side.
SAMPLE_HOLD_SECONDS = 2.0

GLIDE_SECONDS = 0.4
"""How long the Shepard comb takes to reach a new level here, against the 1.5 s default.

**Read against the tread, not against the ear.** The default was chosen for a level that flows — 1.5 s is
what stops a tone following a 20 Hz signal from reading as vibrato. This screen's level *steps*, once every
`SAMPLE_HOLD_SECONDS`, so there is no vibrato to suppress and a 1.5 s glide covers only 74% of each step
before the next one lands: the pitch never arrives, sits permanently a quarter of a step behind, and adds its
whole time constant to a chain that already carries seconds of filter delay.

0.4 s reaches `1 − e^(−2/0.4)` = **99%** inside the tread, so each step is heard as a quick slide that then
holds — which says both that the level moved and by how much. Not zero, because a jump in a Shepard comb is
heard as a different tone rather than as movement, and that percept is the whole reason the instrument is
here. See `feedback.Music.glide_seconds`."""

HOLD_CHOICES = {
    SAMPLE_HOLD_SECONDS: "stepped · a new value every 2 s",
    0.0: "continuous · twenty a second",
}
"""What the pills say. Each names a behaviour and not where a number came from, because an operator choosing
between two behaviours needs the behaviours named. The reasoning is in `SAMPLE_HOLD_SECONDS`.

**Which one you pick matters at alpha and gamma and barely at all infra-slow**, which is not obvious and is
worth knowing before touching it. Measured deterministically on a real capture, five passes, the same rolling
30 s 90th-percentile threshold either way:

| | threshold | met | reward on/off | level changes |
|---|---|---|---|---|
| 0.05 Hz stepped | 86.6 | 11.1% | 1.9/min | 18.5/min |
| 0.05 Hz continuous | 86.6 | 11.0% | 1.9/min | 449.9/min |
| 10 Hz stepped | 0.00262 | 7.9% | **3.8/min** | 8.6/min |
| 10 Hz continuous | 0.00274 | 10.8% | **53.2/min** | 162.0/min |

At the infra-slow preset the two are the *same instrument*: identical threshold, identical time on target,
identical reward rate, and only the smoothness of the level differs. That follows from the band — a 0–0.1 Hz
signal carries nothing a sample every 2 s cannot represent, so the hold is very nearly lossless there.

At alpha it is a different instrument. The product's envelope moves on the timescale of a 3 Hz band, so a
2 s sample is heavy undersampling, and the reward fires **fourteen times less often** — 3.8 a minute against
53. The 10% aim is meant to be "a rare, distinct event rather than a background state", and only the stepped
figure is that."""

# **These two are the sensitivity control.** The generic ramp controls are removed from this screen — see
# `scores_bands` — because there is no logistic to widen, so these two are what the operator adjusts rather
# than a spread multiplier.
# percent of the threshold: smaller means the output covers that distance faster.


def reward_output(value: float, threshold: float,
                  position: float = RANGE_POSITION, size: float = RANGE_SIZE) -> float:
    """The continuous reward output: where `value` sits in the window, 0 to 1.

    Linear, not a logistic — there is no soft ramp here, so the window itself is what
    decides how far a move travels. With the shipped sliders the window is 65% to 100% of the
    threshold, so the output is 0 at 0.65x, 1 at the threshold, and stays 1 above it.
    """
    top = threshold * position / 100.0
    bottom = threshold * (position - size) / 100.0
    if top <= bottom:
        return 0.0
    return float(min(1.0, max(0.0, (value - bottom) / (top - bottom))))


# The interval gate on the logical output, rising edges only. Much faster than
# alpha-theta's 3 s because the reward is aimed at 10% and a slow gate on a rare event swallows most of them.
REWARD_INTERVAL = 0.5

# A floor under the *value*, in µV², below which the reward is never met whatever the threshold says.
#
# **Measured failure, on the no-electrode bench recording.** Both channels rail to a constant ±800 µV, a
# bandpass of a constant is zero, so the product is zero — and a threshold derived from 30 s of zeros is
# `0.0`, at which point `value >= threshold` is `0 >= 0` and the reward fires **100% of the session on no
# signal at all**. That is the same hole `Target.signal_floor` was added for on alpha-theta, where the reward
# is gated on 1.0 µV upstream of the comparison.
#
# 0.5 µV², because a gate of 1.0 µV per channel and two channels at 1 µV in phase make a product of
# ½·1·1. So the number is that gate converted into this measure's own unit.
SIGNAL_FLOOR_UV2 = 0.5

# And the same gate for the amplitude reward, in µV: 1.0 upstream of the comparison, the same position.
SIGNAL_FLOOR_UV = 1.0

SESSION_MINUTES = 20.0

# --- what the state maps against -----------------------------------------------------------------
#
# **The scale the state is measured in is the summed wave's own recent spread, and the reason is a measured
# failure of the thing it replaced.** `state_balance` divided by the *waveform plot's* scale, which is
# `nice_ceiling` of a **maximum** over the ring — twelve cycles, four minutes at 0.05 Hz. The filter's
# settling transient is the largest thing that ever enters that ring, so it set the gain for its whole
# residency, and the recording that established this (5 min, bipolar, 0.01 Hz corner, 0.05 Hz) reads:
#
#     time   ring peak   auto scale   sum_wave
#       25s      387.1         500       222.72   ← the transient
#      150s      387.1         500         8.49
#      225s      387.1         500        -0.49
#      250s      129.3         200        30.68   ← it finally leaves the ring
#
# A 20.1 µV rms wave against a 500 µV scale is a state of p5 0.448, p50 0.497, p95 0.555 — **10.7% of the
# range**, 8% covered in a typical minute. On a Shepard tone spanning six octaves that is half an octave of
# glide per minute, which is what "so slow it is imperceptible" was reporting. It is not a startup cost
# either: a maximum is re-armed by any large excursion, so one movement buys another four minutes of it,
# and the window is in *cycles*, so at 0.01 Hz it would be twenty minutes.
#
# Measured on the same wave, the alternatives (p5–p95 span, and how much of it clamps at an end):
#
#     as shipped, 12-cycle maximum       0.107     0%
#     pinned 100 µV                      0.343     0%
#     pinned  50 µV                      0.687     4%
#     de-centred, 3σ over 6 cycles       0.538     0%
#     de-centred, trailing p99 of |x|    0.794     7%
#     de-centred, tanh at 1.5σ           0.774     0%
#
# The last is what is implemented. A σ moves a little when an artifact arrives where a maximum moves
# completely, which is the whole difference; the curve is what buys the width without the clipping that
# makes both ends stop carrying information.
#
# **This is not new auto-ranging, it is the same auto-ranging with an estimator that survives an artifact.**
# The objection written down in `music.ShepardPlayer` and `nfb.dsp.wave` — auto-range tells the trainee
# about their variance rather than their state — applies to what was already here. The absolute mapping is
# still available and is still one gesture: pinning the slow plot's scale pins the feedback with it, and
# that path stays linear, because "the wave at the top of the plot is the top of the range" is the whole
# point of pinning it.
STATE_WINDOW_CYCLES = 6.0
"""How much of the recent past the state's centre and spread are taken over, in cycles of the trained
frequency — the unit the waveform ring already uses, and the only one that works across three decades. Six
cycles is 120 s at 0.05 Hz: long enough that the estimate does not breathe with the wave it is measuring,
short enough to follow a session that changes. Measured at 6 and at 3 cycles, the state's spread differed by
0.002, so this is not a knife edge."""

STATE_SIGMA_SPAN = 1.5
"""How many σ of the summed wave reach 0.76 of the range, the value `tanh(1)` takes.

Not a hard end: the curve is asymptotic, so a wave three times its usual size still moves the output and
still reads as larger, where a linear map with a clamp would have stopped saying anything at 1.5σ. Measured
span at this setting is 0.774 of the range with nothing clamped."""

STATE_SETTLE_CYCLES = 1.0
"""How much wave the spread estimate wants before the state means anything, on top of the filter's own
settling. Without it the first seconds after warmup divide by a σ built from almost nothing, which reads as
a wildly swinging state exactly when the trainee is settling in. `state_balance` is None until then, and the
mechanisms fall back to the level, which is the same honest reading they fall back to elsewhere."""

BEST_CORNER_HZ = 0.01        # what to tell the operator to set; `TrainingSession.corner_hz` reads the actual

# The waveform ring is `traces.WaveRing`, in cycles of the trained frequency rather than seconds — the only
# unit that works when the frequency can be 0.05 or 40. Twelve cycles is enough to read the shape of a slow
# wave; the Lissajous takes the last two off the same ring, since a figure over more cycles is drawn on top of
# itself. Both counts are this screen's judgement, which is why they are arguments there and constants here.
LISSAJOUS_CYCLES = 2
WAVE_FLOOR_UV = 1.0
"""A floor under the waveform plots' full scale. Their peak is zero before data arrives and immediately after
a retune, and `nice_ceiling` of nothing prints "±1e-09 µV" — true about an empty buffer, unreadable as an
axis."""

REWARD_SECONDS = 600.0
REWARDS_H = 150
SLOW_H = 190
LISSAJOUS_H = 190
TRACE_H = 74
TRACE_SECONDS = 5.0

# --- what each plot can be set to ---------------------------------------------------------------
#
# **Every one of these was a constant, and a constant is a guess about a session you are not in.** Three
# things forced it. A high-pass costs settling, and the settling transient is the largest thing in the window
# for minutes — measured at 0.05 Hz it pinned the slow plot's auto-scale at ±10 µV while the wave under it
# was below ±0.5, so the picture was empty and stayed empty until the transient scrolled out four minutes
# later. The synchrony plot has the same auto-scale and the same problem. And the raw trace's ±100 µV is
# wrong in both directions at this corner: too small for drift, far too large for 8 µV of EEG.
#
# `0` means auto in each of the scale ladders — the behaviour that was there before, kept as the default so
# nothing changes until somebody chooses otherwise.
WAVE_CYCLE_CHOICES = (4, 8, 12, 24)
# **How hard the state drives what follows it.** The state is normalised against a measurement — see
# `STATE_WINDOW_CYCLES` — and a measurement is not a preference: how much movement you *want* out of it is.
# An earlier version left that to each mechanism's own range, on the grounds that the shadow has its
# brightness span and the tone has its climb, and it was wrong in the way that matters — a span sets where
# an output starts and stops, not how much of it a wave of this size uses, so a session that felt dead had
# nothing to turn.
#
# It multiplies the movement in both mappings, the σ span on auto and the pinned scale when the plot is
# pinned, so the number means the same thing either way. Offered through `state_sensitivity_range`, which is
# why it appears under the shadow's `follows` row on this screen and on no other: a state that is already
# bounded 0…1 by construction has nothing for a gain to fix.
#
# **The top of the range is measured rather than picked.** The output sits within 5% of an end once the wave
# exceeds 1.47 × the full-scale, which at sensitivity s is 1.47 × 1.5σ / s. For a wave with Gaussian spread
# that is 2.7% of the time at ×1, 27% at ×2 and **58% at ×4**; for a wave as sinusoidal as an infra-slow one,
# 75% at ×4. Past about ×2 the instrument spends most of the session pinned near an end, which is the failure
# the soft curve exists to avoid arriving through a different door — so 3 is the ceiling, not 4, and it is
# already past the point of diminishing returns rather than at it.
STATE_SENSITIVITY_RANGE = (0.25, 3.0)
WAVE_SCALE_CHOICES = (0.0, 1.0, 2.0, 5.0, 10.0, 25.0)
REWARD_MINUTE_CHOICES = (2.0, 5.0, 10.0, 20.0)
REWARD_SCALE_CHOICES = (0.0, 1.0, 5.0, 20.0, 100.0)
TRACE_SECOND_CHOICES = (2.0, 5.0, 10.0, 30.0)
TRACE_GAIN_CHOICES = (25.0, 50.0, 100.0, 200.0, 500.0, 800.0)
TARGETS_W = 440
AXIS_W = 44
SPECTRUM_LO, SPECTRUM_HI = 1.0, 45.0


def preset_name(hz: float) -> str:
    """The nearest named setting, or the number. Nearest in **log** space, because these span three decades
    and 0.1 Hz is far from 0.05 while 44 Hz is not far from 40."""
    best, distance = "", float("inf")
    for name, centre in PRESETS:
        gap = abs(np.log(max(hz, 1e-9) / centre))
        if gap < distance:
            best, distance = name, gap
    return best if distance < 0.35 else f"{hz:g} Hz"


REWARD_HELP = (
    "One reward, at one frequency. It multiplies the two channels' filtered waveforms together and "
    "removes the negative part. It is zero whenever the two sites sit on opposite sides of their own "
    "baselines, however loud they are. It is largest when they move the same way at the same time.\n\n"
    "This is not the coupling screen's reward, which adds the channels and measures the amplitude of "
    "the sum. A sum is an amplitude, in microvolts. This is a product, in microvolts squared. The two "
    "numbers do not compare, even at the same frequency on the same electrodes.\n\n"
    "The three presets point this one instrument at infra-slow, alpha or gamma. Only one frequency "
    "trains at a time: moving the frequency moves the same reward, it does not add a second one.\n\n"
    "**Silent at / full at** sets the sensitivity. The output stays silent until the value reaches the "
    "\"silent at\" percent of the threshold. It then rises in a straight line, and reaches full the "
    "instant the value crosses the threshold, which is also when the reward fires.\n\n"
    "A narrower gap between the two numbers makes a small change in synchrony move the output further. "
    "Move the bottom toward 0 to make the output change gently and never go silent. Raise the top above "
    "100% to leave headroom above the threshold. The output then does not reach full at the moment you "
    "earn the reward.\n\n"
    "There is no curve control here and no spread to scale against. This mapping is a straight line, so "
    "the ramp controls on the Feedback tab do not change it.\n\n"
    "The bandwidth is a fixed number of hertz. Below about 1.5 Hz the lower edge falls past zero, and "
    "the filter becomes a lowpass with no high-pass at all. At 0.05 Hz with a 3 Hz bandwidth, you train "
    "the product of two signals held between 0 Hz and 1.55 Hz. This is not a narrow band at 0.05 Hz. "
    "The card states which of the two filter types is running.\n\n"
    "The aim is 10%, not the 90% a reward usually gets, so this reward is a rare event, not a state you "
    "hold. The threshold tracks the value over the last 30 seconds. A tracking threshold holds the "
    "percentage near 10% at all times. A real change then shows in the value and the threshold, not in "
    "the percentage. Switch to fixed if you want the percentage itself to show a change."
)

INHIBITS_HELP = (
    "Nine bands per channel, side by side from 0.1 to 40 Hz. Each band stays below its own threshold "
    "95% of the time. This is a gate in front of the reward: the reward only counts while every band "
    "stays clear.\n\n"
    "This screen has no alert sound for the bands. Eighteen bands, each over threshold a twentieth of "
    "the time, create a sound through most of the session, and that sound carries no information. "
    "Stopping the reward is the signal instead.\n\n"
    "One wide inhibit cannot do this job. Set it high enough to avoid firing on habitual activity, and "
    "it misses a smaller rise elsewhere in the range."
)

SLOW_HELP = (
    "The two sites' filtered waveforms, on one axis and one scale. This is the infra-slow display; it "
    "shows the state, and does not train a reward.\n\n"
    "By default, the window holds twelve cycles of the trained frequency, not a fixed number of seconds. "
    "The cycle pills change the count. At 0.05 Hz, twelve cycles is four minutes. At 40 Hz it is a third of a second. A fixed number of seconds cannot show "
    "the same picture across three decades of frequency.\n\n"
    "Two slow waves rising and falling together show the state this protocol trains toward. Two waves "
    "drifting apart at the same amplitude show the failure this screen's other instruments report.\n\n"
    "**The summed wave also drives the state**, which a Shepard tone or a state-following brightness "
    "follows. On auto, the state centres on the recent mean and scales to the recent spread, measured "
    "over six cycles. The caption states the µV range that reaches the ends of the state. A 5 µV wave "
    "and a 50 µV wave each move the feedback through its full range. Pin the scale instead to map the "
    "wave linearly: the wave at the top of the plot becomes the top of the range. A pinned scale lets "
    "you compare feedback between sessions; an automatic scale does not."
)

LISSAJOUS_HELP = (
    "One channel plotted against the other. This picture shows the reward: it multiplies these two "
    "numbers together.\n\n"
    "A line along the rising diagonal means the two sites are in phase. A line along the falling "
    "diagonal means antiphase: the reward reads zero there, as far from the trained state as it gets. A "
    "circle means the two sites are a quarter cycle apart. A shapeless blob means no fixed relationship "
    "at all.\n\n"
    "An ellipse that leans off a diagonal can mean a phase shift or an amplitude imbalance between the "
    "two channels; the two look the same in this picture. Compare the two waveforms above to tell them "
    "apart.\n\n"
    "Both axes share one scale. A separate scale on each axis turns any amplitude imbalance into an "
    "apparent phase shift. This display must not allow that."
)


class ILFSession(TrainingSession):
    """One product-synchrony reward over a channel pair, nine inhibit bands per channel."""

    # **The synchrony reward makes no sound of its own.** It produces a level (a percent of threshold) and
    # a mark, so the arrangement is the proportional master volume plus the click — and the music is off.
    #
    # `shepard` is offered rather than removed, and the distinction matters: it is the *ILF reward element's*
    # feedback, not synchrony's, and an earlier version of this screen enabled it by default and thereby
    # presented another instrument's design as this one's. Off-by-default and unavailable are different
    # claims, and only the first is true — the level exists here, so the music works; it is simply not part of this method.
    music_sources = ("shepard",)

    # **No alert**: eighteen inhibit bands aimed at 95% means something is over threshold most of the time,
    # and the inhibits are an AND in front of the reward and nothing else.
    offers = ("sound", "shadow", "click", "music")

    # `automatic` first: the threshold is a rolling ratio over the
    # epoch, recomputed as data arrives. `fixed` stays available because it is the mode under which the aim
    # means what it says: a tracker holds the percentage at the aim whatever the trainee does.
    threshold_modes = ("automatic", "fixed")

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0):
        super().__init__(protocol, path=path, device=device, length_seconds=length_seconds,
                         source=source, channels=channels or 2)
        if length_seconds is None:
            self.plan.set_length(SESSION_MINUTES)

        chans = list(self.specs)
        self.pair = (chans[0], chans[1]) if len(chans) >= 2 else None
        self.bandwidth_hz = DEFAULT_BANDWIDTH_HZ
        self.order = PRODUCT_ORDER
        self.post_seconds = DEFAULT_POST_SECONDS

        self.trace_seconds = TRACE_SECONDS
        self.wave_cycles = WAVE_CYCLES
        self.wave_scale_uv = 0.0            # 0 = auto, as it always was
        self.reward_seconds = REWARD_SECONDS
        self.reward_scale = 0.0             # 0 = auto
        self.traces = {i: TraceBuffer(320, TRACE_SECONDS, s.rate_hz)
                       for i, s in self.specs.items()}
        self.rings = {i: RawRing(2.0, s.rate_hz, SPECTRUM_LO, SPECTRUM_HI, 1.0)
                      for i, s in self.specs.items()}
        self.sum_ring = RawRing(2.0, self.rate, SPECTRUM_LO, SPECTRUM_HI, 1.0)

        # **Everything below the product runs on the two channels summed**, with one set of inhibit bands for
        # the mode rather than one per channel.
        #
        # So there are nine inhibit bands, not eighteen, and only the product uses the channels separately.
        # **Two aligners, and they are not interchangeable.** An aligner is stateful — it hands each run over
        # exactly once — so one object cannot serve both the measurement and the display. They also see
        # different data on purpose: `consume` runs only on blocks the artifact gate accepted, while
        # `observe_raw` sees everything, and a spectrum that hid what was rejected would differ from the
        # per-channel spectra beside it. That was a real regression when the sum moved into `consume`.
        self.inhibit_aligner = PairAligner(*self.pair) if self.pair else None
        self.display_aligner = PairAligner(*self.pair) if self.pair else None
        self.site = chans[0] if chans else 0        # whose spec the summed signal borrows

        self.metric: ProductSynchrony | None = None
        self.wave: WaveRing | None = None
        self.reward: Target | None = None
        self.amplitude: Target | None = None
        self.bands: dict[tuple, LockinBandPower] = {}
        self.inhibit_targets: list[Target] = []
        self.reward_plot = EnvelopePlot(columns=int(REWARD_SECONDS / TREND_SECONDS), count=1)
        self.reward_ceiling = StickyCeiling(patience=6.0, floor=1.0)
        self._reward_plot_at = 0.0

        # The level is a percent-of-threshold number and the logical output is the mark.
        self.feedback.click.enabled = True
        self.feedback.click.interval_seconds = REWARD_INTERVAL
        # **The shadow is on when there is a trainee's screen for it**, which is a reversal, and the reason it
        # was off is the reason it is now conditional rather than absent. The trainee is usually watching a
        # game on another screen — so what the shadow must not do is dim
        # the operator's console, which is what it did when the default was unconditionally on. A second
        # display *is* the trainee's, the game is on it, and a brightness that follows the reward is
        # information added to that game rather than a screen taken from anybody.
        #
        # `shadow.subject_screen()` answers both halves at once: whether there is such a display, and which
        # number it is. With one display it returns None and this lands off, exactly as before.
        subject = shadow.subject_screen()
        self.feedback.shadow.enabled = subject is not None
        if subject is not None:
            self.feedback.shadow.screen = subject
        self.feedback.music.enabled = False
        # Off, but *this* sound if it is switched on. The dataclass default is pink noise, which has no
        # connection to this protocol at all — the Shepard tone is at least an infra-low sound, borrowed
        # from the slow-wave screen next door. See `music_sources`.
        self.feedback.music.source = "shepard"
        self.feedback.music.glide_seconds = GLIDE_SECONDS
        # **The pitch follows the summed slow wave, not the level.** In ILF training the pitch follows the
        # signed waveform's position, the slow signal itself, never an envelope or a score. The level here is the product's approach window, mostly
        # at its ends by design, which as a pitch would jump between two notes; the summed wave rises and
        # falls the way a pitch can follow. Loudness stays the synchrony channel (the master volume), so
        # each signal keeps its own perceptual dimension. Configurable — see `Music.follows`.
        self.feedback.music.follows = STATE
        # And when it plays, an inhibit violation pauses it — by default, not as an option someone has to
        # find. The inhibits are a feedback channel of their own, independent of every
        # reward, and an inhibit whose only
        # consequence is suppressing an event that fires 10% of the time barely acts on the trainee.
        # `music_hold` is what makes "failing" mean the inhibits here rather than the whole condition.
        self.feedback.music.stops_when = "failing"

        # **No smoothing of the level, and this is the one screen where that is right.** Everywhere else the
        # level is a blend of target scores that flicker several times a second, and 0.3 s earns its place.
        # Here the level is a straight line from 0.65× the threshold to the
        # threshold with nothing after it, and it arrives through a 2 s sample-and-hold. Smoothing a staircase with a 2 s tread using a 0.3 s pole rounds corners nobody
        # can see and costs 0.3 s of lag on a chain that already carries seconds of filter delay.
        self.feedback.smoothing_seconds = 0.0

        # The window's two controls, held per session so they can be set — see `RANGE_LIMITS`.
        self.range_position = RANGE_POSITION
        self.range_size = RANGE_SIZE
        # Stepped by default, because the rest of this screen is tuned for it — see `SAMPLE_HOLD_SECONDS`.
        # Which band the slow fork uses — `nfb.dsp.product.SLOW_BANDS` carries the reasoning.
        self.slow_band = DEFAULT_SLOW_BAND
        self.hold_seconds = SAMPLE_HOLD_SECONDS
        self.hold_choices = HOLD_CHOICES
        self._held_at: dict = {}
        self._held_value: dict = {}
        self._reset_state_scale()

        self.set_frequency(self.default_frequency_hz())
        for low, high in INHIBITS:
            self.add_inhibit(low, high)

    # --- the one reward --------------------------------------------------------------------

    @property
    def frequency_hz(self) -> float:
        return self.metric.centre_hz if self.metric is not None else DEFAULT_HZ

    @property
    def lowpass_only(self) -> bool:
        """Whether the filter has degraded to a bare lowpass at this centre — by design, not a fallback."""
        return is_lowpass(self.frequency_hz, self.bandwidth_hz)

    def set_frequency(self, hz: float, bandwidth_hz: float | None = None,
                      order: int | None = None, post_seconds: float | None = None,
                      slow_band: str | None = None) -> bool:
        """Point the instrument somewhere, rebuilding it. Returns False if the settings are not measurable.

        Rebuilt rather than edited, because every one of these changes what is being measured: the target's
        statistics were gathered against a different number and carrying them over would mix two quantities
        under one threshold. The waveform ring goes with it for the same reason, and because its own stride
        depends on both the centre and the upper edge.
        """
        if self.pair is None:
            return False
        if bandwidth_hz is not None:
            self.bandwidth_hz = min(BANDWIDTH_RANGE[1], max(BANDWIDTH_RANGE[0], bandwidth_hz))
        if order is not None:
            self.order = min(ORDER_RANGE[1], max(ORDER_RANGE[0], int(order)))
        if post_seconds is not None:
            self.post_seconds = max(0.001, min(1.0, post_seconds))
        if slow_band is not None and slow_band in SLOW_BANDS:
            self.slow_band = slow_band
        centre = snap_hz(min(HZ_RANGE[1], max(HZ_RANGE[0], hz)))
        try:
            metric = ProductSynchrony(self.specs, self.pair, centre, slow_band=self.slow_band,
                                      bandwidth_hz=self.bandwidth_hz, order=self.order,
                                      post_seconds=self.post_seconds, output_hz=METRIC_HZ)
        except ValueError:
            return False
        self.metric = metric
        self.wave = WaveRing(centre, metric.high, self.rate, self.pair,
                             cycles=self.wave_cycles, tail_cycles=LISSAJOUS_CYCLES)
        self._reset_state_scale()
        if self.reward is not None and self.reward in self.targets:
            self.targets.remove(self.reward)
        self.amplitudes.pop(("product",), None)
        # `band` is what it is measured over, so the card can print it and the colour follows the centre. The
        # lowpass case is honestly a 0–high band, and `Band` will not take a zero lower edge, so it is the
        # smallest positive number the screen would ever print — the caption says "below X Hz" either way.
        band = Band(max(metric.low, 1e-4), metric.high, "")
        self.reward = Target(PAIR, band, "increase", mode="automatic", pair=self.pair,
                             name=f"{centre:g} Hz", unit=metric.unit,
                             hit_rate_override=REWARD_AIM, window_seconds=EPOCH_SECONDS,
                             signal_floor=SIGNAL_FLOOR_UV2,
                             # A rectified product is not a narrowband envelope — about half its mass is an
                             # atom at exactly zero — so the Rayleigh fit that stands in for a percentile on
                             # narrow bands must not be reached. See `Target.narrowband_envelope`.
                             narrowband_envelope=False)
        self.targets.insert(0, self.reward)

        # The second reward is an amplitude reward in its own right, on the summed channels, at the same centre,
        # with its own aim of 60% and its own epoch. Its mark is ANDed with the negated inhibits
        # exactly as the product's is, so a Synchrony session trains **both**.
        if self.amplitude is not None and self.amplitude in self.targets:
            self.targets.remove(self.amplitude)
        self.amplitudes.pop(("amplitude",), None)
        self.amplitude = Target(PAIR, band, "increase", mode="automatic", pair=self.pair,
                                name=f"{centre:g} Hz amplitude", unit="µV",
                                hit_rate_override=AMPLITUDE_AIM,
                                window_seconds=amplitude_epoch(centre),
                                signal_floor=SIGNAL_FLOOR_UV)
        self.targets.insert(1, self.amplitude)
        self.reward_plot = EnvelopePlot(columns=int(REWARD_SECONDS / TREND_SECONDS), count=1)
        return True

    def add_inhibit(self, low: float, high: float) -> Target | None:
        """One band on the summed channels — see the note in `__init__` for why it is not one per channel."""
        band = Band(low, high, "")
        key = ("inhibit", self.band_key(band))
        if key not in self.bands:
            spec = self.specs.get(self.site)
            if spec is None:
                return None
            try:
                self.bands[key] = LockinBandPower(spec, band, output_hz=METRIC_HZ)
            except ValueError:
                return None
        # `PAIR` with both channels in it, so the label says `ch0+ch1` and `target_is_live` requires *both* —
        # a channel that is not delivering is still being added into the sum.
        tgt = Target(PAIR, band, "increase", role="inhibit", mode="fixed", pair=self.pair,
                     hit_rate_override=INHIBIT_AIM)
        self.inhibit_targets.append(tgt)
        self.targets.append(tgt)
        return tgt

    def set_view(self, *, trace_seconds: float | None = None, wave_cycles: int | None = None,
                 wave_scale_uv: float | None = None, reward_seconds: float | None = None,
                 reward_scale: float | None = None, state_sensitivity: float | None = None) -> None:
        """Change how a plot is *drawn*, and how hard the state drives what follows it.

        The sensitivity is here rather than with the measurement for the same reason the scales are: it
        changes nothing about what is measured, only how much of a mechanism's travel a given wave asks
        for. It sits beside the plot's own scale because those two are the pair an operator reasons about
        together — the picture and how hard it pushes.

        The trace and reward windows own buffers whose *column duration* changes with the window, so
        changing one rebuilds it and the history in it is gone — honest, because a buffer that kept its
        old samples would be drawing two different spans on one axis. The wave ring is different: its
        stride is set by the frequency, not by the cycle count, so the same points serve any width and
        `resize` keeps them — rebuilding it destroyed minutes of a slow wave for a display choice.
        The scales are read at draw time and cost nothing.
        """
        if trace_seconds is not None and trace_seconds != self.trace_seconds:
            self.trace_seconds = float(trace_seconds)
            self.traces = {i: TraceBuffer(320, self.trace_seconds, sp.rate_hz)
                           for i, sp in self.specs.items()}
        if wave_cycles is not None and wave_cycles != self.wave_cycles:
            self.wave_cycles = int(wave_cycles)
            if self.wave is not None:
                self.wave.resize(self.wave_cycles)
            elif self.metric is not None:
                self.wave = WaveRing(self.frequency_hz, self.metric.high, self.rate, self.pair,
                                     cycles=self.wave_cycles, tail_cycles=LISSAJOUS_CYCLES)
        if reward_seconds is not None and reward_seconds != self.reward_seconds:
            self.reward_seconds = float(reward_seconds)
            self.reward_plot = EnvelopePlot(columns=int(self.reward_seconds / TREND_SECONDS), count=1)
            self.reward_ceiling = StickyCeiling(patience=6.0, floor=1.0)
        if wave_scale_uv is not None:
            self.wave_scale_uv = max(0.0, float(wave_scale_uv))
        if reward_scale is not None:
            self.reward_scale = max(0.0, float(reward_scale))
        if state_sensitivity is not None:
            self.set_state_sensitivity(state_sensitivity)

    def default_frequency_hz(self) -> float:
        """The preset a fresh session opens on — see `SUBHERTZ_CORNER_HZ`."""
        return SUBHERTZ_CORNER_HZ if self.corner_hz <= 0.02 else DEFAULT_HZ

    def adopt_targets(self, previous) -> None:
        """**The inhibit ladder is the protocol and is rebuilt; the reward comes across by its settings.**

        Nine contiguous bands per channel at a fixed 95% aim are not a list anyone edits, so a fresh session
        builds them. What an operator sets here is where the one instrument points and how it is filtered,
        which is carried across and applied by rebuilding — a reward assigned as an object would have no
        metric behind it and would read zero for ever.
        """
        self.slow_band = getattr(previous, "slow_band", self.slow_band)
        self.set_frequency(getattr(previous, "frequency_hz", self.default_frequency_hz()),
                           bandwidth_hz=getattr(previous, "bandwidth_hz", self.bandwidth_hz),
                           order=getattr(previous, "order", self.order),
                           post_seconds=getattr(previous, "post_seconds", self.post_seconds))

    def key_for(self, tgt: Target):
        if tgt is self.reward:
            return ("product",)
        if tgt is self.amplitude:
            return ("amplitude",)
        return ("inhibit", self.band_key(tgt.band))

    def value_for(self, tgt: Target) -> float:
        return self.amplitudes.get(self.key_for(tgt), 0.0)

    def latency_for(self, tgt: Target) -> float | None:
        if tgt in (self.reward, self.amplitude):
            return self.metric.latency_seconds if self.metric is not None else None
        metric = self.bands.get(self.key_for(tgt))
        return metric.latency_seconds if metric is not None else None

    scores_bands = False
    """The level is not built by scoring targets against thresholds here — see `raw_level`. So the SIGNAL
    panel's first four steps have nothing to act on, exactly as on the heart screen: there is no soft ramp to
    widen, no spread to score against, and nothing to combine because the level reads one target."""

    state_ends = ("trough", "crest")
    state_sensitivity_range = STATE_SENSITIVITY_RANGE

    def _reset_state_scale(self) -> None:
        """Forget the state's centre and spread. Called on a retune, because both are properties of a band
        and carrying them across would map one band's wave against another band's amplitude."""
        self._state_mean = 0.0
        self._state_var = 0.0
        self._state_weight = 0.0
        self._state_seconds = 0.0
        self._state_samples: int | None = None

    def _advance_state_scale(self) -> None:
        """Fold the newest summed wave into the running centre and spread, once per aligned sample.

        Off the metric's own `samples_seen` rather than off blocks or a clock: `consume` is called once per
        channel per block, so counting calls would advance this at twice the rate on a pair and at a
        different rate again on a different block size. The metric counts what the aligner actually matched,
        which is the only clock the wave itself has.

        **Nothing accumulates until the filter has settled**, which is the whole point — the transient is
        exactly the excursion that must not set the scale. See the note above `STATE_WINDOW_CYCLES`.
        """
        metric = self.metric
        if metric is None or not self.lowpass_only:
            return
        seen = metric.samples_seen
        if self._state_samples is None or seen < self._state_samples:
            self._state_samples = seen                  # first block, or a rebuilt metric
            return
        new, self._state_samples = seen - self._state_samples, seen
        if new <= 0 or metric.warmup < 1.0:
            return
        dt = new / max(self.rate, 1e-9)
        window = STATE_WINDOW_CYCLES / max(self.frequency_hz, 1e-9)
        alpha = 1.0 - np.exp(-dt / max(window, 1e-9))
        value = float(metric.sum_wave)
        self._state_mean += alpha * (value - self._state_mean)
        self._state_var += alpha * ((value - self._state_mean) ** 2 - self._state_var)
        # Bias correction, the standard one: an average started at zero *is* zero for its first time
        # constant, and dividing by the weight accumulated so far is what stops a σ built from three
        # samples reading as a tiny σ and saturating the output.
        self._state_weight += alpha * (1.0 - self._state_weight)
        self._state_seconds += dt

    @property
    def state_scale_uv(self) -> float | None:
        """The σ the state is measured in, µV, or None while there is not yet enough wave to say.

        Public because the screen prints it: a mapping whose gain is derived has to show the number it
        derived, the same rule the threshold readouts follow.
        """
        if (self._state_weight <= 0.0
                or self._state_seconds * max(self.frequency_hz, 1e-9) < STATE_SETTLE_CYCLES):
            return None
        sigma = (max(self._state_var, 0.0) / self._state_weight) ** 0.5
        return sigma if sigma > 0 else None

    @property
    def state_full_uv(self) -> float | None:
        """How many µV either side of centre the state's travel is spent over, or None while settling.

        One expression for the mapping and the caption both, so a printed number can never describe a
        different gain from the one in force. `state_sensitivity` divides it: twice the sensitivity is half
        the microvolts, which is twice the movement out of the same wave.
        """
        if self.wave_scale_uv:
            return self.wave_scale_uv / self.state_sensitivity
        sigma = self.state_scale_uv
        return None if sigma is None else STATE_SIGMA_SPAN * sigma / self.state_sensitivity

    @property
    def state_balance(self) -> float | None:
        """Where the summed slow wave is sitting, 0…1 — this protocol's *state*.

        The signed waveform, not its envelope: ILF feedback follows the wave's actual
        position, the slow signal itself — and a
        rectified 0.05 Hz wave is not that idiom. What a state-following shadow, the crossfade and a
        state-following Shepard all read.

        **Two mappings, and which one is in force is the operator's own gesture.** Pinning the slow plot's
        scale maps the wave against it linearly, so the wave at the top of the plot is the top of the range
        and one pill calibrates the picture and the feedback together. On auto, the wave is centred and
        divided by its own recent spread, then put through a soft curve — see `STATE_WINDOW_CYCLES` for the
        measurement that made the plot's auto scale unusable here as a feedback gain.

        None outside the lowpass branch — at alpha or gamma the "summed wave" is a fast oscillation, not
        a state — while the cascade is still settling, and until the spread has enough wave behind it. The
        mechanisms then fall back to the level, which `shadow_input` documents as the honest reading.
        """
        metric = self.metric
        if metric is None or not self.lowpass_only or metric.warmup < 1.0:
            return None
        if self.wave_scale_uv:
            full = self.wave_scale_uv / self.state_sensitivity
            return float(min(1.0, max(0.0, 0.5 + metric.sum_wave / (2.0 * full))))
        sigma = self.state_scale_uv
        if sigma is None:
            return None
        # De-centred, because a standing offset is not a state: at a 0.01 Hz corner the drift that survives
        # into the band would otherwise sit the whole session at one end of the range and the wave would
        # ride on it. The waveform screen subtracts its own slow cascade for the same reason.
        centred = float(metric.sum_wave) - self._state_mean
        return float(0.5 + 0.5 * np.tanh(centred / self.state_full_uv))

    music_failing = ("an inhibit is over its threshold",
                     "any of the nine bands over its threshold pauses the sound until it clears")

    def music_hold(self, met: bool) -> bool:
        """"Stops while failing" here means **an inhibit is violated**, not the condition as a whole.

        The inhibits are a feedback channel of their own, independent of what any reward is doing, and the
        DVD-style feedback pauses on exactly that. The
        product, meanwhile, is met 10% of the time *by design*, so gating the playback on the full
        condition would silence it 90% of the session and the interruption would carry no information.
        Quiet Reward (`stops_when == "met"`) keeps the whole condition, because there "doing well" is
        what earns the silence and doing well is the product's rare event.
        """
        if self.feedback.music.stops_when == "failing":
            return any(t.total and not t.wanted for t in self.targets
                       if t.role == "inhibit" and self.target_is_live(t))
        return super().music_hold(met)

    @property
    def level_source(self) -> tuple[str, str]:
        """What the readout says the level is, in place of the combining rule's wording."""
        bottom = (self.range_position - self.range_size) / 100.0
        top = self.range_position / 100.0
        return ("the synchrony product's approach to its threshold",
                f"0 at {bottom:g}× the threshold, full at {top:g}×")

    @property
    def output_bottom(self) -> float:
        """Where the continuous output reaches zero, as a percent of the threshold.

        The window is stored as a *width* below a *top*; this is the bottom those two imply. Kept as a derived
        property rather than a third field so there is still one source of truth, and exposed because it is
        the end an operator reasons about — "it goes quiet at 65% of my threshold" is a sentence, where
        "the window is 35 wide" is a description of the storage.
        """
        return self.range_position - self.range_size

    @output_bottom.setter
    def output_bottom(self, value: float) -> None:
        self.range_size = max(0.0, self.range_position - value)

    @property
    def raw_level(self) -> float:
        """**The continuous output, and only the synchrony product's.**

        The output takes one value and one threshold — the synchrony product's — and returns where it
        sits in a window ending at the threshold. It is a straight line. See `reward_output`.

        What this replaces was the average of two rewards pushed through a logistic each, which
        could not work: the product is aimed at being met 10% of the time and the
        amplitude reward at 60%, so averaging them gives a number that hovers near the middle and reports
        mostly the common one. Measured on a real capture, the product's own score swung 0.07 to 0.95 while
        the average of the two only reached 0.32 at its low point — most of the movement in the trained
        quantity cancelled by a second measurement that was not about synchrony.

        **The amplitude reward is not being dropped.** It keeps its card, its threshold and its place in the
        condition, as its own bar, ANDed into the same
        reward event. It simply does not colour the continuous output.
        """
        tgt = self.reward
        if tgt is None or not self.target_is_live(tgt) or tgt.threshold <= 0:
            return 0.0
        return reward_output(self.value_for(tgt), tgt.threshold,
                             self.range_position, self.range_size)

    def channel_targets(self, channel: str) -> list[Target]:
        """**The inhibits gate the condition and do not touch the level**, which is not what
        `combine` would do.

        The continuous output comes from the value and the auto-threshold and nothing else. The inhibits
        reach only the internal AND, which produces the discrete mark. So the continuous output follows the rewards
        alone and the discrete one is rewards AND inhibits.

        This screen inherited `combine="scaled"` from cross-frequency coupling, where twelve bins *are* meant
        to pull the level down. Copying a level rule between protocols is exactly what `pinned_feedback`
        exists to stop, and it slipped through because the rule is a default rather than a control here.

        **And once the level stops seeing inhibits, the rule stops mattering at all**: `scaled`, `weighted`
        and `weakest` all reduce to the mean of the rewards when there is nothing on the other side. So the
        assignment was removed rather than corrected — a line that cannot change the output is worse than no
        line, because it reads as a decision somebody made.
        """
        live = self.active_targets
        if channel == LEVEL:
            # **The product alone**, because that is what `raw_level` reads, which takes one
            # value and one threshold. Returning both rewards here while the level ignored one of them made
            # the readout name a contributor that contributes nothing, which is the exact failure this hook's
            # docstring warns about from the other direction.
            return [x for x in live if x is self.reward]
        # **The condition is the product AND the inhibits — the amplitude reward is not in it.** The two
        # rewards are separate outputs and nothing combines them. Including the amplitude here let a background state aimed at 60% veto a
        # synchrony event designed to be rare at 10% — roughly a third of legitimate rewards eaten by a
        # condition that says nothing about synchrony. The amplitude keeps its card and its threshold; its
        # discrete channel simply is not this one.
        return [x for x in live if x is self.reward] + \
               [x for x in live if x.role == "inhibit"]

    # --- feed ------------------------------------------------------------------------------

    def observe_raw(self, block, judged) -> None:
        ch = block.channel
        values = np.asarray(block.values)
        if ch in self.traces:
            self.traces[ch].push(self.for_display(ch, values), block.valid, judged.accepted)
        if ch in self.rings:
            self.rings[ch].push(values)
        if self.display_aligner is not None:
            paired = self.display_aligner.push(block)
            if paired is not None:
                _start, a, b = paired
                self.sum_ring.push(a + b)

    def consume(self, block) -> set:
        fresh: set = set()
        if self.metric is not None:
            readings = self.metric.feed(block)
            # Fed from the metric rather than from the readings, because the waveform is decimated at its own
            # rate: a 40 Hz waveform needs hundreds of points a second and the readings arrive at twenty.
            if self.wave is not None:
                self.wave.feed(self.metric.wave_start, self.metric.waveforms)
            for r in readings:
                key = ("product",) if r.name == self.metric.name else ("amplitude",)
                self.amplitudes[key] = self._held(key, r.value, block)
                fresh.add(key)
            # After the metric has advanced and before anything reads the state. Here rather than inside
            # `state_balance` because that is a property, read by three mechanisms and by the panel — an
            # estimator that advanced on being *looked at* would run at a rate set by how many things
            # were watching.
            self._advance_state_scale()
        fresh |= self._consume_inhibits(block)
        return fresh

    def _held(self, key, value: float, block) -> float:
        """`value` through the sample-and-hold, if one is in force — see `SAMPLE_HOLD_SECONDS`.

        Held here, on the way into `amplitudes`, because the one hold feeds both the
        threshold and the value, so holding once reproduces both. The target still observes
        at `METRIC_HZ`; it simply sees the same number until the hold expires, which puts the same
        *distribution* in its window as fifteen distinct readings would and therefore the same percentile.

        Timed off the block's own sample index rather than a clock, so a replay holds exactly what a live
        session held and a paused session does not silently advance.
        """
        if self.hold_seconds <= 0:
            return value
        now = block.start_index / max(1e-6, block.spec.rate_hz)
        since = self._held_at.get(key)
        if since is None or now - since >= self.hold_seconds or now < since:
            self._held_at[key] = now
            self._held_value[key] = value
        return self._held_value.get(key, value)

    def _consume_inhibits(self, block) -> set:
        """The nine bands, on a synthetic block carrying the two channels summed.

        Built from **raw counts** rather than microvolts, so the sum is exact: every channel on this device
        shares one scale, and adding counts then scaling once is the same number without a round trip through
        floats. It carries the first channel's spec and number because that is what the estimators are keyed
        to — a metric checks the block's channel against its own spec, and there is no channel number meaning
        "the sum". Same construction as the alpha-theta screen's, for the same reason.
        """
        fresh: set = set()
        if self.inhibit_aligner is None:
            return fresh
        paired = self.inhibit_aligner.push(block)
        if paired is None:
            return fresh
        start, a, b = paired
        spec = self.specs.get(self.site)
        if spec is None:
            return fresh
        counts = (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / spec.scale
        # Usable only if *both* contributing channels are: one bad electrode is inside the sum, so the sum is
        # as bad as its worst input rather than as good as the block that happened to arrive.
        worst = OK if all(self.state.get(c) is not False for c in self.pair) else NO_SIGNAL
        summed = Block(self.site, start, tuple(int(round(v)) for v in counts), spec, state=worst)
        for key, metric in list(self.bands.items()):
            for r in metric.feed(summed):
                self.amplitudes[key] = r.value
                fresh.add(key)
        return fresh

    def push_displays(self, now: float) -> None:
        if now - self._reward_plot_at >= self.reward_seconds / self.reward_plot.columns:
            self._reward_plot_at = now
            # Not while the cascade is still settling. The step transient is µV² against a signal in the
            # ones — measured, it pinned this plot's auto scale at 50000 µV² for the whole window it took
            # to scroll out, so ten minutes of session were drawn as a flat line under one spike. The
            # transient is a property of the filters, not a measurement, so it does not enter the plot;
            # the caption says what is happening instead of the axis silently absorbing it.
            if self.metric is not None and self.metric.warmup >= 1.0:
                self.reward_plot.push([self.reward.value if self.reward is not None else 0.0])

    def spectrum_sum_ring(self):
        """Offered after the channels — `sum_spectrum_leads` stays false — and **that order is the claim**.
        On the coupling screen the sum is what every reward is measured on, so it leads; here nothing is ever
        measured on a sum, so putting it first would present a diagnostic as the instrument.

        And only where there is a pair to add, since the ring is filled from one."""
        return self.sum_ring if self.pair else None

    # --- what this protocol requires -------------------------------------------------------

    def wiring_problem(self) -> str | None:
        """Why the numbers would be wrong, worst first — only one is shown, because a shared reference makes
        the reward meaningless and anything said beside it would be noise."""
        if self.pair is None:
            return "needs two channels; only one is in use"
        if self.device is not None:
            reference = getattr(self.device.status.settings, "reference", "")
            if reference and reference != "bipolar":
                return (f"reference is '{reference}', and this protocol needs 'bipolar' — a shared "
                        f"reference appears in both channels at once, so it multiplies with itself and "
                        f"the reward reports the reference electrode's own noise as synchrony")
        # The *lower edge* against the corner, not the centre: in the lowpass case there is no lower edge at
        # all and the hardware high-pass becomes it, which is exactly the thing worth saying — a 0.05 Hz
        # setting against a 0.5 Hz corner is training 0.5–1.55 Hz and calling it infra-slow.
        if self.metric is not None and self.lowpass_only and self.corner_hz > self.metric.high / 3:
            # A single-pole high-pass passes `f/sqrt(f²+fc²)`, and the reward is a **product**, so what
            # survives is that squared. At a 0.5 Hz corner with the band stopping at 0.1 Hz that is 4% of the
            # scale, not a shifted band — the earlier wording said "you are training 0.5–1.55 Hz instead",
            # which was wrong twice: the corner is above the band, so there is no band left.
            top = self.metric.high
            kept = (top / (top ** 2 + self.corner_hz ** 2) ** 0.5) ** 2
            # The wording turns on whether *our* band has a lower edge. Under `0–2×` the device's corner
            # really does become the only one; under the others there is a software edge and the device's
            # corner is simply above the whole band. Either way the band is gone, and saying "no high-pass"
            # when there is one would send the operator looking for the wrong thing.
            lacking = ("the filter has no high-pass of its own" if self.metric.low <= 0 else
                       "this band's low edge comes from software, not from the device")
            return (f"At {self.frequency_hz:g} Hz, {lacking}. The device's {self.corner_hz:g} Hz "
                    f"high-pass corner sits above this band's top edge of {top:g} Hz. The reward "
                    f"keeps about {kept:.0%} of its scale. Set High-pass to {BEST_CORNER_HZ:g} Hz on "
                    f"the device screen. The filtering is in the hardware, so no software setting "
                    f"can fix it.")
        return None


def session_for(*args, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None,
                **kwargs) -> ILFSession:
    _ = heart_choice, heart_runner, mendi_choice, mendi_runner, pupil_choice, pupil_runner
    return ILFSession(*args, **kwargs)


def view(page: ft.Page, session: ILFSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    """Build once, mutate on Flet's loop."""

    signal = signal_pane.SignalPane(page, session)
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width(reserve: int = TARGETS_W) -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - reserve - 70)

    spectrum = spectrum_pane.SpectrumPane(page, session, plot_width)

    trends = trends_pane.TrendsPane(page, session, plot_width)

    slow_canvas = cv.Canvas(shapes=[], expand=True, height=SLOW_H)
    slow_axis, slow_labels = value_axis(SLOW_H, count=5, width=AXIS_W)
    slow_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    slow_legend = ft.Row(spacing=12)

    liss_canvas = cv.Canvas(shapes=[], expand=True, height=LISSAJOUS_H)
    liss_caption = ft.Text("", size=t.SMALL, color=t.FAINT)

    reward_canvas = cv.Canvas(shapes=[], expand=True, height=REWARDS_H)
    reward_axis, reward_labels = value_axis(REWARDS_H, count=4, width=AXIS_W)
    reward_caption = ft.Text("", size=t.SMALL, color=t.FAINT)

    # This screen is where the override was invented — the amplifier's below-range detector fires on a
    # perfectly good infra-low channel — and it is now every screen's, in `trace_rows`.
    traces = trace_rows.TraceRows(page, session, lambda: plot_width(reserve=TARGETS_W), height=TRACE_H)

    reward_column = ft.Column(spacing=10)
    reward_rows = ft.Column(spacing=12)
    preset_row = ft.Row(spacing=6)
    cards: dict[int, TargetCard] = {}

    def retune(**kwargs) -> None:
        # The two output-window sliders change the *mapping* and not the measurement, so they are set
        # straight on the session — pushing them through `set_frequency` would rebuild every estimator to
        # change a number the estimators never see.
        # The slow band changes what is measured, so it goes through `set_frequency` and rebuilds.
        for key in ("range_position", "range_size", "output_bottom"):
            if key in kwargs:
                setattr(session, key, max(RANGE_LIMITS[0], min(RANGE_LIMITS[1], float(kwargs.pop(key)))))
        if kwargs or "hz" in kwargs:
            session.set_frequency(kwargs.pop("hz", session.frequency_hz), **kwargs)
        rebuild_reward()
        signal.rebuild_contributions(session.targets)

    def rebuild_reward() -> None:
        cards.clear()
        tgt, metric = session.reward, session.metric
        if tgt is None or metric is None:
            reward_column.controls = [ft.Text("needs two channels", size=t.BODY, color=t.MUTED)]
            reward_rows.controls = []
            page.update()
            return
        identity = ft.Row(
            [ft.Container(width=3, height=18, bgcolor=tgt.colour, border_radius=2),
             ft.Text(preset_name(session.frequency_hz), size=t.BODY, color=t.TEXT, width=76),
             num_box(session.frequency_hz, lambda v: retune(hz=v), suffix="Hz",
                     step=grid_hz(session.frequency_hz), low=HZ_RANGE[0],
                     high=min(HZ_RANGE[1], session.rate / 2 - 1), width=64),
             ft.Container(expand=True)],
            spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        card = TargetCard(tgt, identity, rebuild_reward, page=page,
                          modes=session.threshold_modes)
        cards[id(tgt)] = card
        reward_column.controls = [card.control]

        # The three places the method points this instrument. Pills rather than three cards, because there is
        # one instrument: pressing one moves it, and the frequency box beside it is the same control.
        preset_row.controls = [
            pill(name, abs(np.log(max(session.frequency_hz, 1e-9) / hz)) < 0.02,
                 (lambda h=hz: lambda _e: retune(hz=h))())
            for name, hz in PRESETS]

        reward_rows.controls = [
            # **The bandwidth box only appears where the bandwidth does anything.** The bandwidth is a fixed
            # width in hertz spanning the whole frequency range, and below the fork at 1.8 Hz the filter
            # abandons it for a lowpass at twice the centre — so at the infra-slow preset every value from 1
            # to 3 gives the identical 0–0.1 Hz band. Measured across the whole slider range, not argued.
            # Drawing an editable "3 Hz" beside a 0.05 Hz centre is the thing that makes this screen look
            # deranged: a number that cannot apply at that frequency, offered as though it were in force.
            # The order still applies to the lowpass, so it stays.
            # **Only below the fork, because only there is there a choice.** Above 1.8 Hz the band is a
            # bandpass at centre ± bandwidth/2 and has a lower edge already; below it, the direct path has none at all and our amplifier cannot supply one — see
            # `nfb.dsp.product.SLOW_BANDS` for why that is a choice rather than a fix.
            *([labelled("slow band",
                        *[pill(name, session.slow_band == name,
                               (lambda n=name: lambda _e: retune(slow_band=n))())
                          for name in SLOW_BANDS],
                        note="of the centre · ½ is the ILF element's own low cut", width=62)]
              if session.lowpass_only else []),
            labelled("band",
                     *([ft.Text(f"{metric.low:g}–{metric.high:g} Hz", size=t.BODY, color=t.TEXT)]
                       if session.lowpass_only else
                       [num_box(session.bandwidth_hz, lambda v: retune(bandwidth_hz=v),
                                suffix="Hz", step=0.5, low=BANDWIDTH_RANGE[0],
                                high=BANDWIDTH_RANGE[1], width=54)]),
                     *[pill(f"order {n}", session.order == n,
                            (lambda o=n: lambda _e: retune(order=o))())
                       for n in range(ORDER_RANGE[0], ORDER_RANGE[1] + 1)],
                     note=metric.describe(), width=62),
            labelled("smoothing",
                     num_box(session.post_seconds * 1000,
                             lambda v: retune(post_seconds=v / 1000.0),
                             suffix="ms", step=10, low=1, high=1000, width=54),
                     note="on the product, before the threshold", width=62),
            # **No aim row here.** It was a read-only statement of `REWARD_AIM` back when the aim was fixed;
            # it is now a control, on the card's own gear, and a second copy on the panel would be a number
            # that does not change when you change it. Where it came from is still recorded, in
            # `REWARD_AIM` and in the reward card's `?`.
            # The window's size and position, **shown as the two ends of the window.** A width and a top —
            # "35% below 100% of threshold" — describes the storage and not the behaviour, and was unreadable on the screen.
            # The two numbers an operator wants are where the output is silent and where it is full.
            labelled("silent at",
                     num_box(session.output_bottom, lambda v: retune(output_bottom=v),
                             suffix="%", step=5, low=RANGE_LIMITS[0], high=RANGE_LIMITS[1], width=54),
                     ft.Text("of threshold, full at", size=t.SMALL, color=t.MUTED),
                     num_box(session.range_position, lambda v: retune(range_position=v),
                             suffix="%", step=5, low=RANGE_LIMITS[0], high=RANGE_LIMITS[1], width=54),
                     # **No note.** Two boxes and their connecting words fill the row, so a note gets the few
                     # pixels left and wraps to one character per line. It also had nothing to add that the
                     # row does not already say — which is `labelled`'s own rule for when a note earns its
                     # place. What the gap *means* is in the panel's `?`.
                     width=62),
        ]
        page.update()

    inhibit_widgets: dict[int, dict] = {}
    inhibit_grid = ft.Column(spacing=0)
    # **One value column, because there are nine inhibits on the sum and not eighteen across two channels.**
    # This grid was built for the eighteen-band reading and kept its per-channel columns when the ladder was
    # corrected: the inhibits are one set on the sum, per *mode*, so
    # `add_inhibit` builds `Target(PAIR, ...)`. The lookup here asked for `x.source == ch`, which no inhibit
    # target has ever matched — so every cell fell through to an empty `Container` and the panel rendered
    # nine band labels beside nine blank spaces, for a ladder that was computing values the whole time.
    # A grid keyed on something no row can satisfy fails *silently*, which is why it survived: nothing is
    # missing from the screen, the numbers simply are not on it.
    inhibit_scroll = ft.Row([inhibit_grid], scroll=ft.ScrollMode.AUTO, expand=True)
    blocking_text = ft.Text("—", size=22, color=t.TEXT, weight=ft.FontWeight.W_300)
    blocking_note = ft.Text("", size=t.SMALL, color=t.FAINT)

    # `wrap`, because these are the longest pill ladders on any panel head: window and scale together
    # reach eleven pills, which is wider than the head at any usable window size, and unwrapped they
    # were clipped at the panel edge — the larger scales were unreachable, not merely cramped.
    wave_view_row = ft.Row(spacing=10, wrap=True, alignment=ft.MainAxisAlignment.END)
    reward_view_row = ft.Row(spacing=10, wrap=True, alignment=ft.MainAxisAlignment.END)
    trace_view_row = ft.Row(spacing=10, wrap=True, alignment=ft.MainAxisAlignment.END)

    def view_pills(options, current, apply, label=lambda v: f"{v:g}", auto: str = "auto") -> ft.Row:
        """A row of pills that changes how a plot is drawn. `0` in a ladder means auto."""
        return ft.Row([pill(auto if v == 0 else label(v), current == v,
                           (lambda x=v: lambda _e: (apply(x), rebuild_view_rows(), redraw_now()))())
                       for v in options], spacing=4)

    def redraw_now() -> None:
        session.redraw()
        page.update()

    def rebuild_view_rows() -> None:
        """**Window and scale, on the panel they belong to.** Both were constants; see the ladders above for
        what forced the change. Auto is kept as an option and remains the default, so the scales behave as
        they always did until somebody chooses a number."""
        wave_view_row.controls = [
            view_pills(WAVE_CYCLE_CHOICES, session.wave_cycles,
                       lambda v: session.set_view(wave_cycles=v), lambda v: f"{v:g} cyc"),
            view_pills(WAVE_SCALE_CHOICES, session.wave_scale_uv,
                       lambda v: session.set_view(wave_scale_uv=v), lambda v: f"±{v:g}"),
        ]
        reward_view_row.controls = [
            view_pills(REWARD_MINUTE_CHOICES, session.reward_seconds / 60.0,
                       lambda v: session.set_view(reward_seconds=v * 60.0), lambda v: f"{v:g} min"),
            view_pills(REWARD_SCALE_CHOICES, session.reward_scale,
                       lambda v: session.set_view(reward_scale=v), lambda v: f"{v:g}"),
        ]
        trace_view_row.controls = [
            view_pills(TRACE_SECOND_CHOICES, session.trace_seconds,
                       lambda v: session.set_view(trace_seconds=v), lambda v: f"{v:g} s", auto=""),
            view_pills(TRACE_GAIN_CHOICES, session.gain_uv,
                       lambda v: setattr(session, "gain_uv", v), lambda v: f"±{v:g}", auto=""),
        ]

    def build_inhibit_grid() -> None:
        inhibit_widgets.clear()
        # The column is headed by what the bands are actually measured on, which is the *sum* of the pair.
        # `Target.source_label` is that string, so the header cannot drift from the measurement.
        measured_on = (session.inhibit_targets[0].source_label
                       if session.inhibit_targets else "ch0+ch1")
        header = ft.Row([ft.Container(width=74, content=ft.Text("band", size=t.SMALL, color=t.FAINT)),
                         ft.Container(width=150, content=ft.Text(measured_on, size=t.SMALL,
                                                                 color=t.FAINT))], spacing=8)
        rows: list[ft.Control] = [header, ft.Container(height=6)]
        for low, high in INHIBITS:
            cells: list[ft.Control] = [
                ft.Container(width=74, content=ft.Row(
                    [ft.Container(width=3, height=11, border_radius=2,
                                  bgcolor=t.band_colour(low, high)),
                     ft.Text(f"{low:g}–{high:g}", size=t.SMALL, color=t.TEXT)], spacing=5))]
            tgt = next((x for x in session.inhibit_targets
                        if x.band.low == low and x.band.high == high), None)
            if tgt is None:
                # A band the rate cannot carry — `add_inhibit` returns None when the estimator will not fit,
                # and saying so is the point: a blank cell is what this panel used to be made of.
                cells.append(ft.Container(width=150, content=ft.Text(
                    "not at this rate", size=t.SMALL, color=t.FAINT)))
            else:
                value = ft.Text("—", size=t.BODY, color=t.TEXT, width=44)
                pct = ft.Text("", size=t.SMALL, color=t.MUTED, width=32)
                fill = ft.Container(width=0, height=6, border_radius=3,
                                   bgcolor=t.band_colour(low, high), left=0, bottom=0)
                notch = ft.Container(width=2, height=10, bgcolor=t.TEXT, left=0, bottom=-2)
                bar = ft.Stack([ft.Container(width=50, height=6, border_radius=3,
                                             bgcolor=t.SURFACE_HI, left=0, bottom=0),
                                fill, notch], width=50, height=10)
                inhibit_widgets[id(tgt)] = dict(value=value, pct=pct, fill=fill, notch=notch)
                cells.append(ft.Container(width=150, content=ft.Row(
                    [value, bar, ft.Container(width=6), pct], spacing=2,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER)))
            rows.append(ft.Container(content=ft.Row(cells, spacing=8),
                                     padding=ft.Padding(0, 5, 0, 5)))
        inhibit_grid.controls = rows

    def redefine_all(_e=None) -> None:
        for tgt in session.inhibit_targets:
            tgt.redefine()

    warning = ft.Container(visible=False, bgcolor=t.SURFACE_HI, border_radius=8,
                           padding=ft.Padding(12, 9, 12, 9),
                           content=ft.Text("", size=t.SMALL, color=t.BAD))


    # **Only shown while the trained signal actually is a slow waveform**, and the test is the filter's own
    # fork: below 1.8 Hz it degrades to a lowpass, which is when there is a slow signal to read. At 10 Hz the
    # panel was a 1.2 s oscilloscope of a bandpassed sine under a heading claiming otherwise — the raw traces
    # already show that, and the Lissajous shows the alignment, so it was a third view of the same thing
    # wearing the wrong name.
    slow_panel = ft.Column([
        panel("SLOW SIGNAL",
              ft.Row([slow_axis, ft.Container(content=slow_canvas, expand=True, height=SLOW_H)],
                     spacing=0),
              ft.Container(height=6),
              ft.Row([slow_legend, ft.Container(expand=True)]),
              ft.Container(height=4), slow_caption,
              trailing=wave_view_row,
              help=help_button(page, "Slow signal", SLOW_HELP)),
        ft.Container(height=12),
    ], spacing=0)

    train_left = ft.Column([
        slow_panel,
        panel("SYNCHRONY",
              ft.Row([reward_axis,
                      ft.Container(content=reward_canvas, expand=True, height=REWARDS_H)],
                     spacing=0),
              ft.Container(height=6), reward_caption,
              trailing=reward_view_row,
              help=help_button(page, "Synchrony over time",
                               "The reward's value with its threshold drawn across it.\n\n"
                               "The threshold tracks the signal over the last 30 seconds, so it rises "
                               "and falls with it. The percentage of time above the threshold stays "
                               "near the aim at all times. A real change shows in the value and the "
                               "threshold, not in the percentage.")),
        ft.Container(height=12),
        panel("RAW", *traces.rows, ft.Container(height=6), traces.caption,
              trailing=trace_view_row,
              help=help_button(page, "Raw traces",
                               "Both channels, unfiltered apart from a display low-pass named in the "
                               "caption. An amber trace means the artifact gate is rejecting samples on "
                               "that channel; they are still drawn, because a rejection you cannot see is "
                               "indistinguishable from a dead electrode.\n\n"
                               "Nothing at the trained frequency is visible here when it is infra-slow — "
                               "five seconds of a 0.05 Hz wave is a quarter of a cycle. The slow signal "
                               "panel is where that lives.")),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    train_right = ft.Column([
        panel("LISSAJOUS",
              ft.Container(content=liss_canvas, height=LISSAJOUS_H),
              ft.Container(height=6), liss_caption,
              help=help_button(page, "Lissajous figure", LISSAJOUS_HELP)),
        ft.Container(height=12),
        panel("REWARD", reward_column, ft.Container(height=12), reward_rows,
              trailing=preset_row,
              help=help_button(page, "The synchrony reward", REWARD_HELP)),
        ft.Container(height=12),
        panel("INHIBIT BANDS",
              ft.Row([blocking_text, ft.Container(width=10),
                      ft.Column([blocking_note], spacing=0, expand=True)],
                     vertical_alignment=ft.CrossAxisAlignment.CENTER),
              ft.Container(height=12), inhibit_scroll,
              trailing=ft.Container(
                  content=ft.Text("redefine all", size=t.SMALL, color=t.ACCENT),
                  padding=ft.Padding(10, 5, 10, 5), border_radius=6,
                  bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE), on_click=redefine_all),
              help=help_button(page, "Inhibit bands", INHIBITS_HELP)),
    ], spacing=0, width=TARGETS_W, scroll=ft.ScrollMode.AUTO)

    train_tab = ft.Container(
        content=ft.Column([warning, ft.Container(height=10),
                           ft.Row([train_left, train_right], spacing=14, expand=True)],
                          spacing=0, expand=True),
        padding=ft.Padding(14, 12, 14, 14), expand=True)

    panes = [train_tab, spectrum.control, trends.control, signal.control, plan_pane_.control]

    def select_tab(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs(["Train", "Spectrum", "Trends", "Feedback", "Session"], select_tab)

    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        signal.refresh()
        spectrum.refresh()
        trends.refresh()
        session.push_displays(session.elapsed)

        trouble = session.wiring_problem()
        warning.visible = trouble is not None
        if trouble:
            warning.content.value = trouble

        ring, metric = session.wave, session.metric
        # Nothing is drawn into a hidden panel: the shapes cost a full pass over the ring, and a caption
        # written where nobody can read it is how a stale one survives being noticed.
        slow_panel.visible = session.lowpass_only

        if ring is not None and metric is not None and slow_panel.visible:
            width = plot_width(reserve=TARGETS_W + AXIS_W)
            series = ring.series()
            # The sum is drawn from the two channel series added pointwise, which *is* the filtered sum:
            # every stage in the chain is linear and each channel's filters share one design, so
            # filter(a) + filter(b) = filter(a + b) exactly. It is the signal the amplitude reward
            # measures and the state the wave-following mechanisms read, so it belongs on the same axis
            # as its two halves — two waves cresting together and their sum are one picture.
            a, b = (series[c] for c in session.pair)
            summed = a + b
            waves = [a, b, summed]
            # The sum reaches twice a channel's excursion when the sites align, so the auto scale has to
            # see it or alignment — the thing being trained — is exactly what clips.
            peak = (session.wave_scale_uv if session.wave_scale_uv
                    else nice_ceiling(max(ring.peak(), float(np.max(np.abs(summed))),
                                          WAVE_FLOOR_UV)))
            colours = (t.ACCENT, t.ACCENT_DIM, t.MUTED)
            slow_canvas.shapes = signed_shapes(
                waves, width, SLOW_H, peak, colours, pushed=ring.pushed)
            place_value_labels(slow_labels, SLOW_H, peak, bottom=-peak, unit="µV")
            names = [f"ch{c}" for c in session.pair] + ["ch{}+ch{}".format(*session.pair)]
            slow_legend.controls = [
                ft.Row([ft.Container(width=10, height=3, bgcolor=colour, border_radius=2),
                        ft.Text(name, size=t.SMALL, color=t.MUTED)], spacing=6)
                for name, colour in zip(names, colours)]
            filled = min(1.0, ring.pushed / ring.columns)
            slow_caption.value = (
                f"{session.frequency_hz:g} Hz · {metric.describe()} · "
                f"{ring.cycles} cycles = {_duration(ring.seconds)} across · ±{peak:g} µV full scale"
                + ("" if filled >= 1.0 else f" · {filled:.0%} of the window so far")
                + _state_scale_note(session))

        if ring is not None and metric is not None:
            x, y = ring.tail()
            side = min(TARGETS_W - 40, LISSAJOUS_H)
            liss_peak = nice_ceiling(max(
                float(np.max(np.abs(np.concatenate([x, y])))) if len(x) else 0.0, WAVE_FLOOR_UV))
            liss_canvas.shapes = xy_shapes(x, y, side, LISSAJOUS_H, liss_peak, t.ACCENT)
            shape = _lissajous_shape(x, y)
            liss_caption.value = (
                f"ch{session.pair[0]} across, ch{session.pair[1]} up · "
                f"{LISSAJOUS_CYCLES} cycles = {_duration(ring.tail_seconds)} · "
                f"±{liss_peak:g} µV" + (f" · {shape}" if shape else ""))
        else:
            liss_canvas.shapes = []
            liss_caption.value = "needs two channels"

        tgt = session.reward
        # Only the automatic branch: a scale the operator fixed is a choice, and lifting it to clear a
        # threshold would be overriding that choice to tidy a line.
        peak = (session.reward_scale if session.reward_scale
                else plot_ceiling(session.reward_ceiling.update(
                    max(session.reward_plot.peak(), 1e-9), session.elapsed),
                    tgt.threshold if tgt is not None else None))
        reward_canvas.shapes = session.reward_plot.shapes(
            plot_width(reserve=TARGETS_W + AXIS_W), REWARDS_H, peak,
            [tgt.colour if tgt is not None else t.ACCENT],
            thresholds=[tgt.threshold] if tgt is not None else [])
        unit = tgt.unit if tgt is not None else "µV²"
        place_value_labels(reward_labels, REWARDS_H, peak, unit=unit)
        settling = (metric is not None and metric.warmup < 1.0)
        reward_caption.value = (f"{_duration(session.reward_seconds)} across · 0–{peak:g} {unit} "
                                f"full scale{'' if session.reward_scale else ' (auto)'} · "
                                f"thin line = threshold"
                                + (f" · settling — plotted in "
                                   f"{_duration(metric.settle_seconds * (1.0 - metric.warmup))}"
                                   if settling else ""))

        traces.refresh(session.trace_seconds)

        if tgt is not None and id(tgt) in cards:
            card = cards[id(tgt)]
            card.refresh(session.value_for(tgt), tgt.unit, session.latency_for(tgt))
            # The one thing an operator can act on that is not a threshold: a product of two channels at very
            # different amplitudes is dominated by the louder one, and the fix is an electrode.
            if metric is not None and metric.balance > 3.0 and metric.louder_channel is not None:
                other = next(c for c in metric.parts if c != metric.louder_channel)
                card.scale_text.value += (
                    f" · ch{metric.louder_channel} is {metric.balance:.1f}× ch{other} here, so the "
                    f"product mostly follows it")

        blocking = 0
        for inh in session.inhibit_targets:
            wid = inhibit_widgets.get(id(inh))
            if wid is None:
                continue
            value = session.value_for(inh)
            over = inh.total and not inh.wanted
            blocking += int(bool(over))
            wid["value"].value = f"{value:.1f}"
            wid["value"].color = t.WARN if over else t.TEXT
            ceiling = max(inh.threshold * 2.0, value, 1e-9)
            wid["fill"].width = 50 * min(1.0, value / ceiling)
            wid["notch"].left = 50 * min(1.0, inh.threshold / ceiling) - 1
            recent = inh.recent_percent()
            wid["pct"].value = "—" if recent is None else f"{100 - recent:.0f}%"
            wid["pct"].color = t.WARN if (recent is not None and 100 - recent > 10) else t.MUTED
        total = len(session.inhibit_targets)
        ready = all(x.total for x in session.inhibit_targets) if total else False
        if ready:
            blocking_text.value = f"{blocking}"
            blocking_text.color = t.GOOD if blocking == 0 else t.WARN
            blocking_note.value = (
                f"of {total} bands over threshold right now · "
                f"{(1.0 - INHIBIT_AIM) * total:.1f} expected on average at a {INHIBIT_AIM:.0%} aim · "
                + ("the reward is gated" if blocking else "the reward is free to count"))
        else:
            blocking_text.value = "—"
            blocking_text.color = t.FAINT
            blocking_note.value = (f"each band aims to stay below threshold {INHIBIT_AIM:.0%} of the "
                                   f"time, so about {(1 - INHIBIT_AIM) * total:.1f} of {total} will be "
                                   f"over at any moment")

        page.update()

    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)

    build_inhibit_grid()
    rebuild_view_rows()
    rebuild_reward()
    signal.rebuild_contributions(session.targets)
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)


def _state_scale_note(session: ILFSession) -> str:
    """What the *feedback* is mapped against, printed beside what the *picture* is drawn against.

    They used to be the same number, and the caption saying one thing served both. They are no longer: on
    auto the state is measured in σ of the summed wave, which no axis shows. A derived gain that is not
    printed is a gain nobody can check — the same rule the derived thresholds and the derived click pitches
    already follow — and it is what makes "the sound is barely moving" a question with an answer on screen.
    """
    if not session.lowpass_only:
        return ""                       # no state at this frequency; nothing follows it
    full = session.state_full_uv
    if full is None:
        return " · feedback settling"
    return (f" · feedback ±{full:.0f} µV"
            + ("" if session.state_sensitivity == 1.0 else f" at ×{session.state_sensitivity:g}"))


def _duration(seconds: float) -> str:
    """A window length in the unit it is legible in. This screen's windows run from 300 ms to four minutes,
    so one format cannot serve them: "0.3 s" and "240 s" are both harder to read than they need to be."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 90.0:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"


def _lissajous_shape(x: np.ndarray, y: np.ndarray) -> str:
    """A word for what the figure looks like, from the correlation of the two traces.

    A caption rather than a measurement: the reward already reports the quantity, and the value of the
    picture is that a shape is recognised faster than a number is read. Naming what is on screen is what
    makes it teachable — but only when there is enough of it to name, which is why this is silent early.
    """
    n = min(len(x), len(y))
    if n < 20 or np.std(x[-n:]) < 1e-9 or np.std(y[-n:]) < 1e-9:
        return ""
    r = float(np.corrcoef(x[-n:], y[-n:])[0, 1])
    if r > 0.85:
        return "in phase"
    if r < -0.85:
        return "antiphase — the reward reads zero here"
    if abs(r) < 0.3:
        return "no fixed relationship"
    return "partly aligned"
