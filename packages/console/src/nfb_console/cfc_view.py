"""Cross-frequency coupling.

The method comes from its published operations manuals rather than from websites, which carry the theory
and no algorithm.

**No complexity metric drives the feedback, and that is the thing to understand first.** Not entropy, not
Lempel-Ziv, not a 1/f slope; the method's complexity document is a bibliography of such measures and the
protocol uses none of them. Complexity is what the *operator* reads off the cascading spectral display —
"often easy to see on screen as they occur", "more difficult to describe mathematically". What the
software trains is two things:

- **Two narrowband synchrony rewards**, up. `chA + chB` at a hailing frequency: in phase it adds, at 180°
  it cancels. Shipped at 4.8–5.2 and 9.8–10.2 Hz, and the narrowness is the point — "if you want two
  regions of the brain to synchronize briefly (like handshaking) it will not be possible if one part is
  operating around 8 Hz and the other around 12".
- **Twelve amplitude bins**, six per channel, down. 1-5-10-15-20-25-45, which are the lower and upper
  alpha skirts and beta, i.e. the fast-waves-riding-on-slow-waves signature.

**Theta-alpha-gamma is where the method started, not the boundary.** "For starts I investigated CFC primarily of
theta, alpha and gamma EEG… TAG Sync (as a methodology) carries the cross-frequency coupling all the way to
delta, to the 0.1 Hz Mayer wave, to the ultra-low frequencies around 0.01 Hz that are shared with resting
state networks." So `WAVES` spans four decades, and delta-gamma is a pairing the manuals discuss by name — the
coupling that resolves when depression lifts.

And the frequency is meant to be **found rather than picked**: "it was often necessary to find and 'open' a
narrow communications channel between two hubs. This frequency was often within 1-2 Hz of the frequency in
the spectral display showing the highest amplitude in a particular band." Which is why each reward card has
a typeable centre next to the named waves, and why the label says which band you are *in* rather than
claiming you are on a preset. The cascade above is where you read the peak off.

So the protocol is: *synchronise at two frequencies while spending less energy everywhere.* Complexity is
what appears when that succeeds.

**Each reward carries a second, higher threshold, and the moment both cross it at once is an event.**
That is the method's third instrument on the reward pair, and the one this screen went without for a long time:
"you can catch and announce these 'Phase Reset Candidates' using the second thresholds of the Rew 1 & 2
instruments … when Rew 1 and Rew 2 amplitudes simultaneously exceed the second threshold then a chime is
sounded and the top status bar light is illuminated." It is the coincidence the method is named for — the
brief window in which the faster frequency can nest inside the slower — and one reward past its own higher
line means nothing on its own. The manuals call it unnecessary for early training, which is why the sound here is
armed rather than shipped on; see `REACH_RATE` for the numbers and `CFCSession.coincident` for the gates.

**The interlock between the two halves is the elegant part and must not be tidied away.** A summed
amplitude conflates synchrony with amplitude — two channels both getting louder raise the sum without
aligning. The bin edges are what close that: each reward frequency sits *exactly on a bin boundary*, so
it is being pushed down by two bins while the sum is pushed up, and the only way to satisfy both is phase
alignment. The manuals say so outright: "the goal is not to increase the amplitude at that narrow frequency …
in fact that activity at 9.9 Hz and at 10.1 Hz will be inhibited by the inhibit bins."

**Our microvolts are not the manuals' microvolts, and no threshold they quote transfers.** Our bins report *peak*
amplitude (cFIR's analytic modulus) because that is the convention the rest of this codebase uses; a
conventional RMS reading of the same signal is 0.69× ours and a mean-absolute reading is 0.63× — measured, on
a 30 µV sine. Which of those the manuals' platforms report is not in the documents. It mostly does not matter,
because every threshold here is *calibrated from the trainee's own data* rather than typed in, and the
5-15% total is a percentage rather than a voltage. It matters for exactly two things: the bins' full-scale
values below, which came off the manuals' screenshots in their convention, and any number quoted elsewhere.

**This is a slow protocol, deliberately.** A 0.2 Hz band cannot produce a fast signal — its envelope
cannot move faster than a few seconds whatever estimates it, and our lock-in measures 2.05 s of delay.
The 250 ms end-to-end target was derived for alpha *spindles* and does not apply here. CFC
trains a state that persists over seconds, reads it off percentages and a cascade, and rewards it with a
change in the ambient sound. That is coherent rather than a compromise.

**The heart belongs on this screen, not only on its own.** The manuals: "those who make a habit of watching
both the raw and spectral EEG along with the raw and spectral HRV will see that sometimes a person appears
to make improvements in EEG while the HRV splits from its normal peak at 0.1 Hz into high frequency
parasympathetic and very low frequency sympathetic peaks" — one foot on the gas and one on the brake. That
is a failure **you can only see by having both in front of you at once**, which is why the method's own screens
carry a biofeedback panel next to the EEG instruments. So when a heart-rate source is chosen on the device
screen, the same panel appears here, and the split is called out where the EEG is being read rather than on
a screen nobody is looking at.

**It needs bipolar wiring**, and this is the first protocol whose requirement differs from the app
default. "The ear references are not linked together" — with a shared reference, the reference electrode's
own signal appears identically in both channels and therefore adds *in phase* into `chA + chB`, inflating
the reward directly. Linked referencing would manufacture the thing being trained, so the screen says so
rather than quietly training an artifact.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import Band, CFIRBandPower, PairAligner, Reward, SynchronyMetric, MEASURES

from . import chrome, cues, heart, heart_panel, plan_pane, shadow, signal_pane, spectrum_pane, trends_pane
from . import trace_rows
from . import theme as t
from .plots import (Chronospectrogram, EnvelopePlot, bin_centre_x, nice_ceiling, plot_ceiling,
                    place_tick_labels, place_value_labels, tick_axis, value_axis)
from .feedback import FEEDBACK_MODELS
from .protocols import Protocol
from .target_card import SLOW_RECOMPUTE_SECONDS, TargetCard
from .targets import CALIBRATION_SECONDS, PAIR, SPREAD_SAMPLES, Target
from .traces import RawRing, TraceBuffer
from .training import CFIR_DELAY, METRIC_HZ, TrainingSession
from .widgets import help_button, labelled, num_box, panel, pill, tiny

HANDLES = ("CFC",)

# The six bins, exact. Bands and full-scale values are legible on all three of the manuals' example screens,
# and the descending scales *are* the 1/f expectation — "notice as we move from left to right that the
# threshold bars in each bin get lower in amplitude as the frequency gets higher".
BINS = (
    (1.0, 5.0, 100.0),
    (5.0, 10.0, 50.0),       # the lower alpha skirt
    (10.0, 15.0, 50.0),      # the upper alpha skirt
    (15.0, 20.0, 25.0),
    (20.0, 25.0, 25.0),
    (25.0, 45.0, 25.0),
)
BIN_NAMES = {(5.0, 10.0): "lower alpha skirt", (10.0, 15.0): "upper alpha skirt"}

# The two hailing frequencies the method ships, and the pairs worth reaching for by name.
#
# **Two is the manuals' implementation limit, not the method's.** The frequency control panel exposes exactly two
# synchrony rewards — "one or both of the synchrony rewards – Channels 44 and 45" — and TAG Sync is named
# for a *phenomenon* across three bands rather than for three instruments. We are not limited to two, so
# rewards can be added and removed; the presets set a pair because that is the shape the manual describes,
# and a third is available for anyone who wants to see all of theta, alpha and gamma at once.
REWARDS = (5.0, 10.0)

# Every frequency the method reaches, with its own names. The list is short because it is the method's: "TAG Sync
# (as a methodology) carries the cross-frequency coupling all the way to delta, to the 0.1 Hz Mayer wave,
# to the ultra-low frequencies around 0.01 Hz that are shared with resting state networks."
#
# So delta belongs, the Mayer wave belongs, and 0.01 Hz belongs — theta-alpha-gamma is where the method *started*
# ("for starts I investigated CFC primarily of theta, alpha and gamma EEG"), not the boundary. Delta-gamma
# in particular is a pairing the manuals discuss by name, as the coupling that resolves when depression lifts.
WAVES = {
    "ultra-low": 0.01,      # "shared with resting state networks"
    "Mayer": 0.1,           # the 0.1 Hz cardiovascular/anterior-cingulate wave; the HRV target too
    "delta": 2.0,
    "theta": 5.0,           # what the screens ship with
    "alpha": 10.0,          # and the other one
    "gamma": 40.0,
}

# **These are starting points, not a menu, and the centre stays typeable for a reason.** The frequency is
# meant to be *found*: "my earliest 2010 observations included the fact that it was often necessary to find
# and 'open' a narrow communications channel between two hubs. This frequency was often within 1-2 Hz of
# the frequency in the spectral display showing the highest amplitude in a particular band." So you read the
# peak off the cascade and type it, and the named waves are where you start looking.
#
# "Ultra-low" is the manuals' word and has a stated range, from the *Ultra-Low Frequency Bipolar Protocol Guide*
# (2009): "the infra-slow frequencies such as 0.01 – 1.0 Hz", in practice "as low as 0.01 - 0.1 Hz". It is
# not delta, which the manuals list separately — "some practitioners have an aversion to training at a low delta
# frequency" — which is why both appear above.
INFRA_SLOW = (0.01, 1.0)


def wave_name(hz: float) -> str:
    """The nearest named wave, or the frequency itself when it is not near one.

    Nearest in log space, since these span four decades and 0.02 Hz is far from 0.01 while 41 Hz is not far
    from 40. A found hailing frequency is usually 1-2 Hz off a named one, so the label has to say which
    band you are in without pretending you are exactly on it.
    """
    import math

    best, distance = None, math.inf
    for name, centre in WAVES.items():
        gap = abs(math.log(max(hz, 1e-6) / centre))
        if gap < distance:
            best, distance = name, gap
    return best if distance < 0.25 else f"{hz:g} Hz"

# Below this the device's own high-pass is what you would be measuring. `build_eeg_config` can move the
# corner to 0.01 Hz, so this is a warning with a remedy rather than a limit. The corner itself comes from
# `TrainingSession.corner_hz`, which reads it from the config the source is decoding with.


SUBHERTZ_HZ = 1.0
"""Below this a reward gets its own width, because one number cannot serve both ends of four decades.

Not an arbitrary line: a hertz is where a width that is comfortable for the EEG bands stops being a width
at all. 0.2 Hz around 10 is a fifth of a hertz on a band that is four wide; 0.2 Hz around 0.1 is twice the
centre frequency, and it has to be clamped before it reaches DC."""

SUBHERTZ_WIDTH_HZ = 0.05
"""What a reward below a hertz opens at. 0.075–0.125 Hz on the Mayer wave: inside the clamp with room to
spare, and about 8 s of delay, since bandwidth is delay and there is no filter design that avoids it.

It replaces a clamp that nobody chose. The one width used to be shared across the whole range, so a
sub-hertz reward silently got `centre × 0.8` — 0.08 Hz on the Mayer wave, a number that came from the edge
of the arithmetic rather than from anyone's judgement, while the control still read 0.2."""


CALIBRATION_CAP_SECONDS = 600.0
"""The longest a reward's threshold may be derived from. Ten minutes.

Not a statistical figure — the arithmetic below would ask for half an hour at 0.01 Hz — but the length past
which a derivation stops being part of a session and becomes the session. Past the cap the threshold is
derived from fewer independent values than it wants and the card says so, which is the honest report of a
band that genuinely cannot be characterised inside a sitting."""


def calibration_for(freq_hz: float, width_hz: float) -> float:
    """How long a window this reward's threshold needs before it means anything.

    A band's amplitude cannot change faster than its own width, so the *information* in a window is
    `seconds × width` and nothing else — 30 s of a 0.05 Hz band holds about one and a half independent
    readings, which is not a distribution, it is where one slow cycle happened to be. `SPREAD_SAMPLES × 3`
    is the target: enough to measure how far the reading swings and not merely where it sits, with room
    above the point where that estimate starts to be usable, so the last part of the window refines a sane
    number rather than producing the first one.

    **Only below a hertz**, and the reason is what the extra time is *for*. It buys a measured spread, which
    is only wanted where the assumed one does not fit — see `narrowband_envelope` below and
    `Target.derivation`. A 0.2 Hz reward at 10 Hz holds six independent values in 30 s and derives from the
    one-parameter fit, which was measured to work on exactly that: doubling its wait would buy a better
    median and nothing else, on the arrangement this screen opens with.
    """
    if freq_hz >= SUBHERTZ_HZ:
        return CALIBRATION_SECONDS
    return min(CALIBRATION_CAP_SECONDS,
               max(CALIBRATION_SECONDS, SPREAD_SAMPLES * 3.0 / max(width_hz, 1e-6)))


def effective_width(freq_hz: float, wanted_hz: float) -> float:
    """The width actually used at this centre frequency: what was asked for, bounded so the band cannot
    reach DC.

    The bound still matters with a separate sub-hertz width, because that width is also one number over a
    decade: 0.05 Hz is right at the Mayer wave and impossible at 0.01, where it would span -0.015 to 0.035.
    The manuals hit the same wall from the other side and say so — "at such low frequencies, your only options for
    fine tuning the frequency are to narrow the reward bin or to change the filter characteristics".

    The consequence is a much slower instrument, because bandwidth *is* delay: 0.04 Hz wide is about 10 s
    of it. That is honest rather than avoidable, and the card prints it.
    """
    return min(wanted_hz, freq_hz * 0.8)

# What the twelve bins must total between them, over the session. The one hard number in the manual:
# an example totalling 30% is called "too often for most people", and "usually the ideal total for all
# the inhibit bins in both channels added together should be about 5-15%".
TOTAL_RANGE = (5.0, 15.0)
BIN_BUDGET = sum(TOTAL_RANGE) / 2 / 100.0          # aim at the middle of the stated range
# Which implies under 1% per bin, so the per-bin aim is derived from the total rather than reusing our own
# single-inhibit default of 25% — twelve bins at 25% would allow 300%.
BIN_HIT_RATE = 1.0 - BIN_BUDGET / (2 * len(BINS))

# Below this the summed reward has stopped being much of a phase measurement and reads mostly as the louder
# channel's amplitude. 0.5 means one channel has three times the other's amplitude *in the reward's band*.
# See `SynchronyMetric.sensitivity`. **The verdict leaves at a higher number than it arrives at**: the
# measurement wanders, and a note that came and went every frame around one line flickered on the card.
MIN_SENSITIVITY = 0.5
CLEAR_SENSITIVITY = 0.6

# The rewards sound too, and *that* number is not in the manual: "in the Quiet Reward screen the trainee
# will also produce audio alerts when synchrony between the two electrodes falls at either of the reward
# frequencies", with no rate given for it. So it is derived rather than inherited, because inheriting our
# own 90%-per-reward default would put two rewards below threshold ~19% of the time — on top of the bins'
# 10%, which is nearly 30% of the session making a noise. That is precisely the figure the manual calls "too often
# for most people", and it would wreck the one thing Quiet Reward depends on: that silence is reachable.
#
# So the rewards get their own budget, half the bins', and each reward's rate follows from how many there
# are. Two rewards at 97.5% each ≈ 5%, for about 15% total — the top of the stated range, and a deliberately
# conservative starting point, since a threshold that is too easy is invisible while one that is too hard
# is obvious in the first minute.
REWARD_BUDGET = BIN_BUDGET / 2


REACH_RATE = 0.10
"""Each reward's **higher line**, in the top tenth of its own readings.

The moment *every* reward is past its own at once is the event this protocol is named for: two
frequencies aligning across the pair at the same instant, which is the window in which the faster can
nest inside the slower. One reward past its higher line is nothing on its own.

**Not divided between the rewards, unlike the aims above.** `_rebalance_rewards` splits a fixed
interruption budget, so a third reward tightens each aim there. This line is not on that budget — its
rate is bounded by the gate below — and by that function's own argument a third frequency should make
the moment *rarer*, which a fixed share per reward gives."""
REACH_HOLD_SECONDS = 2.0
"""How long the coincidence must hold, as a duty cycle rather than an unbroken run.

Two seconds because the lock-in behind these rewards measures about 2.05 s of delay: anything shorter is
the two estimators' ramps overlapping rather than a state the instrument can resolve."""
REACH_INTERVAL_SECONDS = 10.0
"""At most one announcement this often, phase kept across the session. Ten rather than the click's couple
of seconds because this screen trains a state that persists over seconds — it is worth announcing as an
arrival, not narrating."""
REACH_PITCH_HZ = 1200.0
"""A pitch rather than a click, and deliberately not keyed to either band. Every other sound here is a
frequency-keyed noise burst, so a clear tone is unmistakable against all of them — and the moment belongs
to both rewards at once, so there is no one frequency to key it to."""
REACH_EVENT_LIMIT = 500
REACH_RESPONSES = ("silent", "tone")


def reward_hit_rate(count: int) -> float:
    """Per-reward aim, so that `count` rewards together sound about `REWARD_BUDGET` of the time."""
    return 1.0 - REWARD_BUDGET / max(1, count)

SYNCHRONY_HELP = (
    "Each reward measures the amplitude of the two channels added together, in a narrow band. When the "
    "two channels are in phase, the sum is large. When they are out of phase, the sum cancels toward "
    "zero. The reading rises when the two sites align at that frequency.\n\n"
    "The band is narrow on purpose. Two regions must match frequency closely to synchronise. A wide "
    "band cannot tell synchrony from a chance match. A narrow band takes longer to measure: a 0.2 Hz "
    "width adds about two seconds of delay, and no filter design removes that delay.\n\n"
    "Find the centre frequency instead of guessing it. Read the strongest peak off the cascade display, "
    "and type that frequency in. It usually falls within 1–2 Hz of a named wave.\n\n"
    "Two rewards is the usual number. A third reward does not add more interruptions, because each "
    "threshold's aim rises to keep the total the same.\n\n"
    "Set each threshold once, then leave it fixed. This protocol trains time spent above a threshold. "
    "An updating threshold holds that time near the same percentage always, so it hides any real "
    "improvement.\n\n"
    "Infra-slow means 0.01 to 1.0 Hz, in practice 0.01 to 0.1 Hz. This is not the same as delta. Below "
    "the amplifier's high-pass corner, this reward measures the filter instead of your signal. Move the "
    "corner on the device screen."
)

BINS_HELP = (
    "Six bands per channel, each held below its own threshold. A single wide inhibit cannot do this "
    "job. Set it high enough to avoid firing on habitual activity, and it misses a smaller rise "
    "elsewhere in the range.\n\n"
    "Thresholds must fall as frequency rises, following the 1/f power law. A bin that breaks this "
    "pattern is marked. Repeated violations are the pattern the protocol exists to reduce.\n\n"
    "Watch the total: the share of the session spent above threshold, summed across all bins. Too high, "
    "and the feedback interrupts too often to notice a change of state. Too low, and the bins show "
    "nothing useful.\n\n"
    "The bin edges sit on both sides of the reward frequencies on purpose. To raise a summed reward "
    "while both channels stay below their inhibit thresholds, the two channels must align in phase "
    "instead of growing louder."
)

SPECTROGRAM_LO, SPECTROGRAM_HI = 1.0, 40.0
SPECTROGRAM_H = 236
SPECTROGRAM_ROWS = 90
# **5 s is the manuals' choice**: "the spectral display shows the approximate voltage that was produced at every frequency
# from 1-40 Hz for the last 5 seconds", and it lines the cascade up with the 5 s raw sweep beside it. The
# longer windows are ours and earn their place: the manuals' before/after comparison is done by saving screenshots
# and putting them side by side, which a window that holds a minute does without the screenshots.
DEPTHS = (5.0, 20.0, 60.0)
SPECTROGRAM_TICKS = (5, 10, 15, 20, 25, 30, 35, 40)
CEILINGS = (5.0, 10.0, 20.0, 35.0, 60.0)
REWARDS_H = 132
REWARDS_SECONDS = 60.0
TRACE_H = 74
PACER_SECONDS = 5.0        # in for the sweep, out for the return: 6 breaths a minute, which is 0.1 Hz
TARGETS_W = 430
AXIS_W = 40
TICK_W = 26


class CFCSession(TrainingSession):
    """Two synchrony rewards over a channel pair, twelve amplitude bins, one cascading spectrogram."""

    # The level rule is load-bearing and measured: `scaled` tracks the rewards at r = 0.94 where
    # `weakest` let a bin decide 55% of the time, and the bins-against-rewards interlock is the whole
    # design. So it is locked, along with the mechanisms the arrangement owns.
    #
    # Unlike alpha-theta, the **arrangement stays a choice**, because it is one here: the method's screens ship in
    # variants and each is loaded deliberately — Quiet Reward, Audio-Proportional, or the pausing one. So
    # the three pills are live and what they set is locked to them, which is why a pinned row on this
    # screen says "set by the arrangement" rather than "set by the protocol".
    #
    # The shadow is deliberately *not* pinned: nothing in the method's description mentions it, so it is ours
    # to offer, and pinning a default the protocol never chose would present our choice as its own.
    # Both threshold modes; the rewards open automatic on a 90 s period, the bins stay fixed as the
    # method's screens have them. See `target_card.SLOW_RECOMPUTE_SECONDS`.
    pinned_feedback = frozenset({"combine", "sound", "click", "alert", "music"})
    arrangements = tuple(FEEDBACK_MODELS)
    # The alert is *this* protocol's, and it only means anything here: 5-15% of the session is the stated
    # budget for hearing something, and every threshold on this screen is derived from it. See `offers`.
    offers = ("sound", "shadow", "click", "alert", "music")

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0, heart_runner=None):
        super().__init__(protocol, path=path, device=device, length_seconds=length_seconds,
                         source=source, channels=channels or 2)
        # Borrowed rather than owned: one heart source, shared with the HRV screen, held by the app. A
        # second reader would fail — a strap notifies one subscriber — and inventing beats beside a real
        # EEG would be worse than showing none.
        self.heart = heart_runner
        chans = list(self.specs)
        self.pair = (chans[0], chans[1]) if len(chans) >= 2 else None
        self.width_hz = 0.2
        self.subhertz_width_hz = SUBHERTZ_WIDTH_HZ
        self.measure = "sum"

        self.traces = {i: TraceBuffer(320, PACER_SECONDS, s.rate_hz)
                       for i, s in self.specs.items()}
        # The spectral display is of the two channels *summed*, which is the method's arrangement and has a
        # consequence worth stating on the screen: "you will not be able to tell from the spectral
        # display whether the recurrent beta spindling is coming from channel A or channel B."
        self.sum_ring = RawRing(2.0, self.rate, SPECTROGRAM_LO, SPECTROGRAM_HI, 1.0)
        # Per channel as well as the sum, for the Spectrum tab. The cascade shows the sum, which is the method's
        # arrangement and carries a stated cost — "you will not be able to tell from the spectral display
        # whether the recurrent beta spindling is coming from channel A or channel B." That is exactly the
        # question a per-channel spectrum answers, and it matters here because a reward is only sensitive
        # to phase while the two channels are balanced.
        self.rings = {i: RawRing(2.0, s_.rate_hz, SPECTROGRAM_LO, SPECTROGRAM_HI, 1.0)
                      for i, s_ in self.specs.items()}
        self.sum_aligner = PairAligner(*self.pair) if self.pair else None
        self.spectrogram = Chronospectrogram(SPECTROGRAM_ROWS, SPECTROGRAM_LO, SPECTROGRAM_HI,
                                            depth_seconds=DEPTHS[0])
        self.reward_plot = EnvelopePlot(columns=300, count=2)
        # Appended from the feed thread, copied whole by the recorder's. The count is separate from the
        # bounded deque so the readout stays honest once the oldest have been dropped.
        self.reach_response = "silent"
        self.reaches: deque = deque(maxlen=REACH_EVENT_LIMIT)
        self.reach_count = 0
        self._reach_log: deque = deque()
        self._last_reach: float | None = None
        self._spectrogram_at = 0.0
        self._reward_plot_at = 0.0

        # Pink noise under Quiet Reward, which is the traditional arrangement and the one the manuals recommend
        # for learning to recognise a change of state. The two halves fit each other: 1/f noise is a
        # rendering of the target state — "appears random but is actually what is called pink noise …
        # This is complexity" — and Quiet Reward makes it *recede* as the trainee approaches that state,
        # so the reward is the music thinning out rather than a sound arriving.
        #
        self.feedback.apply_model("quiet")
        self.feedback.music.enabled = True
        self.feedback.music.source = "noise"
        # **A deeper floor than the shared one, and only the floor.** The ceiling this line used to set was
        # an argument against the old 100 — hear a change rather than be driven back in the chair — and
        # `feedback.SOUND_RANGE` now makes that argument for every screen, so restating it here would only
        # be a second copy to disagree with. What is this screen's own is the *bottom*: Quiet Reward
        # inverts the range, so the floor is where the trainee arrives when they are doing well, and the
        # reward is the noise receding further than an ordinary screen would ever take it.
        self.feedback.sound.low = 15
        # **The shadow is on here when there is a trainee's screen to put it on.** This screen is watched —
        # the cascade is read *by the operator*, but the trainee sits in front of a display and Quiet Reward
        # gives them nothing to look at, so a brightness that recedes as they approach the state is the
        # visual half of the same idea the music is the audible half of. `apply_model` does not touch the
        # shadow, deliberately, so this is not clobbered by switching arrangements.
        #
        # Conditional because the alternative is dimming the operator: see `shadow.subject_screen`, which
        # answers "is there a second display" and "which one is it" together, since a shadow switched on by
        # default has to point at the right screen or the safety argument is decoration.
        subject = shadow.subject_screen()
        self.feedback.shadow.enabled = subject is not None
        if subject is not None:
            self.feedback.shadow.screen = subject
        # **`scaled`, and getting here took a measurement.** The method has two different rules, for two
        # different kinds of output, and collapsing them into one is what went wrong first:
        #
        # - The **discrete** consequences are a strict AND. A sound plays, or the video pauses, when *any*
        #   instrument fails. That is `on_target`, which every discrete mechanism here asks directly, and it
        #   is not affected by this setting at all.
        # - The **continuous** one follows the *rewards*: "the Audio - Proportional screen increases Audio
        #   volume of an MP3 file as the rewards increase."
        #
        # `weakest` is the AND written continuously, so it looked like the faithful choice — but applied to
        # a continuous output it is dominated by whichever of twelve bins is worst. Measured on a real
        # two-channel recording: a bin was the deciding target **55% of the time**, and the level tracked
        # the rewards at only r = 0.48. `scaled` tracks them at r = 0.94 while still letting the inhibits
        # pull it down, which is what "as the rewards increase" plus a working inhibit means.
        self.feedback.combine = "scaled"

        # One setting for all twelve, unlike the amplitude screen's per-card control: the bins are a set
        # with one job, and twelve separate filter choices would be twelve chances to make them
        # incomparable to each other.
        self.bin_delay = CFIR_DELAY["inhibit"]
        self.sync: dict[float, SynchronyMetric] = {}
        self.bins: dict[tuple[int, str], CFIRBandPower] = {}
        self.rewards: list[Target] = []
        self.bin_targets: list[Target] = []
        self._build_targets()

    # --- targets ---------------------------------------------------------------------------

    def _build_targets(self) -> None:
        for freq in REWARDS:
            self.add_reward(freq)
        self._rebalance_rewards()
        for ch in self.specs:
            for low, high, full in BINS:
                self.add_bin(ch, low, high, full)

    def width_for(self, freq_hz: float) -> float:
        """This reward's width: the sub-hertz setting below a hertz, the ordinary one above it, either way
        bounded away from DC. One place, because the card, the caption and the estimator have to agree."""
        wanted = self.subhertz_width_hz if freq_hz < SUBHERTZ_HZ else self.width_hz
        return effective_width(freq_hz, wanted)

    def add_reward(self, freq_hz: float) -> Target | None:
        if self.pair is None:
            return None
        reward = Reward(freq_hz, self.width_for(freq_hz), self.measure)
        try:
            self.sync[round(freq_hz, 6)] = SynchronyMetric(self.specs, self.pair, reward,
                                                 output_hz=METRIC_HZ)
        except ValueError:
            return None
        # **`fixed`, not our automatic default, and this one is the method's goal rather than a taste.** The
        # trained quantity is *time above threshold*: "the goal of this type of training is to increase the
        # client's production (i.e., time above threshold) of reward 1 and reward 2." A threshold that
        # tracks a rolling window pins that percentage by construction, so production cannot increase and
        # the number being trained becomes unable to show it — which is the criticism that applies
        # to any automatic mode here, walked into from the other side.
        #
        # It also matches the method's screens: every instrument on them reads **Manual**, with an `A → M` button
        # that copies an auto-computed value into the manual one. Our calibration is that button, run once.
        metric = self.sync[round(freq_hz, 6)]
        tgt = Target(PAIR, Band(reward.low, reward.high, ""), "increase", mode="automatic",
                     recompute_seconds=SLOW_RECOMPUTE_SECONDS,
                     window_seconds=calibration_for(freq_hz, reward.high - reward.low),
                     pair=self.pair, name=f"{freq_hz:g} Hz sync",
                     unit="µV" if self.measure == "sum" else "",
                     # `sum` is a narrowband amplitude, which is what the Rayleigh fit is *for*. `plv`
                     # is a correlation bounded to [0,1] — fitting a Rayleigh to it put the threshold at
                     # 0.19 on a pair whose locked value is 1.0, so the reward was always met and the
                     # measure trained nothing. Same guard ILF carries for its rectified product.
                     #
                     # **And a sub-hertz sum is not a narrowband envelope either**, which took a
                     # measurement to see. On a real 0.1 Hz reward the readings ran 110–228 µV around a
                     # median of 175 — a 5th percentile at 0.70 of the median, where a Rayleigh puts it at
                     # 0.27. Down here the reading is a slowly drifting *level*: the lock-in's window is
                     # tens of seconds and the band is hundredths of a hertz wide, so nothing about it
                     # returns towards zero the way a 10 Hz envelope does between bursts. Under the
                     # Rayleigh fit the threshold sat at a fifth of the median, below anything the signal
                     # reached, and the reward was met 100% of the session. See `Target.derivation`, where
                     # this routes to a fit that measures the spread instead of assuming it.
                     narrowband_envelope=(self.measure == "sum" and freq_hz >= SUBHERTZ_HZ),
                     # The lock-in needs about a time constant per pole before its reading means
                     # anything, and at an infra-slow pairing that is most of a calibration window —
                     # the threshold derived ~30% low without this. Three time constants is where a
                     # one-pole sits within 5% of its answer, the same figure `SignedWave` uses.
                     settle_seconds=3.0 * metric.delay_seconds,
                     calibration_seconds=calibration_for(freq_hz, reward.high - reward.low),
                     hit_rate_override=reward_hit_rate(1),       # rebalanced once the set is complete
                     reach_rate=REACH_RATE)
        self.rewards.append(tgt)
        self.targets.append(tgt)
        return tgt

    def add_bin(self, ch: int, low: float, high: float, full: float) -> Target | None:
        band = Band(low, high, "")
        key = (ch, self.band_key(band))
        if key not in self.bins:
            try:
                self.bins[key] = CFIRBandPower(self.specs[ch], band,
                                               delay_seconds=self.bin_delay,
                                               output_hz=METRIC_HZ, drift_hz=self.drift_hz)
            except ValueError:
                return None         # the band reaches Nyquist on this channel
        tgt = Target(ch, band, "increase", role="inhibit", threshold=full / 4,
                     hit_rate_override=BIN_HIT_RATE, mode="fixed")     # Manual, as the method's screens are
        self.bin_targets.append(tgt)
        self.targets.append(tgt)
        return tgt

    def retune_bins(self, delay_seconds: float) -> None:
        """Rebuild every bin's estimator at a new delay. All of them, so they stay comparable."""
        self.bin_delay = round(max(0.05, delay_seconds), 3)
        self.bins.clear()
        self.amplitudes = {k: v for k, v in self.amplitudes.items() if k[0] == "sync"}
        for tgt in self.bin_targets:
            key = (tgt.source, self.band_key(tgt.band))
            if key not in self.bins:
                try:
                    self.bins[key] = CFIRBandPower(self.specs[tgt.source], tgt.band,
                                                   delay_seconds=self.bin_delay,
                                                   output_hz=METRIC_HZ, drift_hz=self.drift_hz)
                except ValueError:
                    pass

    def retune_rewards(self, freqs=None, width_hz: float | None = None,
                       measure: str | None = None, subhertz_width_hz: float | None = None) -> None:
        """Rebuild the rewards, keeping each one's history where the change was a **nudge**.

        A reward's estimator is built around its centre frequency, so moving the frequency is a new lock-in
        and a new target rather than an edit — and a new target used to mean an empty window and a
        calibration from zero. That is half a minute on a fast band and *minutes* on a sub-hertz one, where
        the frequency is precisely what you are hunting for, "shifts of 0.1 Hz at a time": the calibration
        would restart on every step and never finish. So a centre that moves by less than half the band's
        own width still describes the same signal, and the replacement inherits — see `Target.inherit`.

        **Changing the width or the measure does not inherit**, and that is not a technicality: a different
        width is a different filter with a different delay, and `plv` is not even in the same units as
        `sum`. Those readings are not comparable, so the targets restart, which is what this method has
        always done for everything.
        """
        rebuilt = width_hz is not None or subhertz_width_hz is not None or measure is not None
        if width_hz is not None:
            self.width_hz = max(0.05, width_hz)
        if subhertz_width_hz is not None:
            # A tenth of the floor the ordinary width has, because this one is set at frequencies a tenth
            # as high — 0.005 Hz is a real width at 0.01 Hz and a meaningless one at 10.
            self.subhertz_width_hz = max(0.005, subhertz_width_hz)
        if measure is not None:
            self.measure = measure
        wanted = list(freqs) if freqs is not None else [t_.band.centre for t_ in self.rewards]
        previous = list(self.rewards)
        for tgt in self.rewards:
            self.remove_target(tgt)
        self.rewards.clear()
        self.sync.clear()
        for freq in wanted:
            fresh = self.add_reward(freq)
            if fresh is None or rebuilt:
                continue
            near = self._continues(previous, fresh)
            if near is not None:
                previous.remove(near)
                fresh.inherit(near)
        self._rebalance_rewards()
        # rewards belong at the top of the list, before the bins
        self.targets = self.rewards + [x for x in self.targets if x not in self.rewards]

    @staticmethod
    def _continues(previous: list, fresh: Target) -> Target | None:
        """Which of the old rewards, if any, this new one is a nudged version of.

        **Half the band's width**, because that is where the two stop overlapping in the middle: nudge a
        0.05 Hz-wide reward from 0.100 to 0.115 and most of what it measures is what it measured before, so
        the readings behind it still describe the signal in front of it. Move it to 0.2 and they do not.

        Nearest first, so that swapping two rewards' frequencies does not hand one of them the other's
        history. Matched by centre alone: the caller has already established that nothing else changed.
        """
        if not previous:
            return None
        near = min(previous, key=lambda x: abs(x.band.centre - fresh.band.centre))
        width = fresh.band.high - fresh.band.low
        return near if abs(near.band.centre - fresh.band.centre) < width / 2 else None

    def _rebalance_rewards(self) -> None:
        """Every reward's aim depends on how many there are, so a change re-derives all of them.

        Otherwise the budget would grow with the count and silence would recede as you trained *more*
        coupling, which is backwards — the point of adding a third frequency is to require more, not to be
        interrupted more.

        **Except where the operator set one.** The budget is a default and a hand-set aim outranks it, the
        same way a nudged threshold outranks a calibration — otherwise adding a third reward would silently
        undo a number somebody typed on the other two, and the gear that sets it would look broken.
        """
        rate = reward_hit_rate(len(self.rewards))
        for tgt in self.rewards:
            if not tgt.aim_by_hand:
                tgt.hit_rate_override = rate

    def key_for(self, tgt: Target):
        if tgt.source == PAIR:
            return ("sync", round(tgt.band.centre, 4))
        return (tgt.source, self.band_key(tgt.band))

    def value_for(self, tgt: Target) -> float:
        return self.amplitudes.get(self.key_for(tgt), 0.0)

    @property
    def total_above(self) -> float:
        """The number the operator actually manages: percent above threshold, summed over all bins.

        Summed rather than averaged, because that is the figure the manual's 5-15% refers to — "when you
        total all the 'per cent above' values in figure 3 you get about 30%. This means that about 30% of
        the time the feedback will either pause or set off an auditory alert."
        """
        return sum(100.0 - (t_.recent_percent() or 100.0) for t_ in self.bin_targets)

    @property
    def total_ready(self) -> bool:
        return all(t_.recent_percent() is not None for t_ in self.bin_targets)

    def breaks_one_over_f(self) -> list[Target]:
        """Bins whose amplitude exceeds the one below them in frequency, on the same channel.

        "Repeated violation of this expectation in one or more inhibit bins may indicate sickness
        behavior such as inefficient, maladaptive or redundant network processing." The operator is told
        to look for this, so it is drawn rather than left to the eye.
        """
        out = []
        for ch in self.specs:
            per_channel = [x for x in self.bin_targets if x.source == ch]
            for lower, upper in zip(per_channel, per_channel[1:]):
                if upper.value > lower.value and upper.total:
                    out.append(upper)
        return out

    # --- feed ------------------------------------------------------------------------------

    def observe_raw(self, block, judged) -> None:
        ch = block.channel
        values = np.asarray(block.values)
        if ch in self.traces:
            self.traces[ch].push(self.for_display(ch, values), block.valid, judged.accepted)
        if ch in self.rings:
            self.rings[ch].push(values)
        if self.sum_aligner is not None:
            paired = self.sum_aligner.push(block)
            if paired is not None:
                _start, a, b = paired
                self.sum_ring.push(a + b)
        if self.feedback.music.source == "eeg" and ch == self.focus:
            self.music.feed(values)

    def adopt_targets(self, previous) -> None:
        """**The bins are not adopted, and the rewards are adopted by their settings rather than as
        objects.** Twelve bins at fixed edges with descending thresholds are the protocol, not a list
        anyone edits, so a fresh session rebuilds them; what an operator does set here is where the rewards
        listen and how wide, which is carried across. Rebuilding through `retune_rewards` also keeps the
        lock-ins and the target list consistent, which assigning a list would not.
        """
        freqs = [x.band.centre for x in getattr(previous, "rewards", [])]
        self.retune_bins(previous.bin_delay)
        self.retune_rewards(freqs or None, width_hz=previous.width_hz,
                            measure=previous.measure,
                            subhertz_width_hz=getattr(previous, "subhertz_width_hz", None))

    sum_spectrum_leads = True
    """The sum first, because that is the display the reward frequency is read off — "often within 1-2 Hz
    of the frequency in the spectral display showing the highest amplitude in a particular band" — then each
    channel, which is where an imbalance between the two shows up."""

    def spectrum_sum_ring(self):
        """Only with a pair to add. The ring itself is always built, because the cascading spectrogram reads
        from the same one, so "has a ring" does not answer "has a sum"."""
        return self.sum_ring if self.pair else None

    def consume(self, block) -> set:
        ch = block.channel
        fresh: set = set()
        for (mch, key), metric in list(self.bins.items()):
            if mch == ch:
                for r in metric.feed(block):
                    self.amplitudes[(mch, key)] = r.value
                    fresh.add((mch, key))
        for freq, metric in list(self.sync.items()):
            for r in metric.feed(block):
                key = ("sync", round(freq, 4))
                self.amplitudes[key] = r.value
                fresh.add(key)
        return fresh

    def push_displays(self, now: float) -> None:
        """The two slow instruments, advanced from the draw loop rather than from the feed.

        The cascade is a *time* axis, so its rows have to arrive at a fixed rate however the blocks do —
        driving it from the feed would make the scroll speed a function of the sample rate and the number
        of channels, and the axis would then be lying.
        """
        if now - self._spectrogram_at >= self.spectrogram.interval:
            self._spectrogram_at = now
            row = self.sum_ring.spectrum()
            if row.size:
                self.spectrogram.push(row)
        if now - self._reward_plot_at >= REWARDS_SECONDS / self.reward_plot.columns:
            self._reward_plot_at = now
            self.reward_plot.push([t_.value for t_ in self.rewards] or [0.0, 0.0],
                                  mark=self.coincident)

    # --- the coincidence -------------------------------------------------------------------

    @property
    def coincident(self) -> bool:
        """Every reward past its **higher** line at once, which is the moment this screen exists for.

        Four gates, each of them a failure this repo has already paid for once: a target that has never
        been fed reads zero, one still calibrating has a provisional line derived from a fraction of a
        window, a dead channel earns whatever it likes, and a reward with no higher line cannot be past
        it. Announcing the crossing of a guess would be worse than saying nothing.
        """
        rewards = self.rewards
        return bool(rewards) and all(
            t_.total and not t_.provisional and t_.reach_rate is not None
            and self.target_is_live(t_) and t_.reaches(t_.value)
            for t_ in rewards)

    def _maybe_click(self) -> None:
        """The reward click, then the coincidence on top of it.

        On the click's side of the house rather than the alert's, because this is something earned. It is
        the protocol's own cue rather than a shared mechanism: a coincidence between two rewards means
        nothing on a screen that has one reward, and the arrangement here is pinned.

        Held rather than instantaneous, and re-armed on a fixed interval whose phase is kept — two
        envelopes flickering across a line together is exactly the shape that produces a burst.
        """
        super()._maybe_click()
        now = time.monotonic()
        held = self.coincident
        self._reach_log.append((now, held))
        horizon = now - (REACH_HOLD_SECONDS + 1.0)
        while self._reach_log and self._reach_log[0][0] < horizon:
            self._reach_log.popleft()
        if not held or not self.duty_held(self._reach_log, REACH_HOLD_SECONDS, now):
            return
        if self._last_reach is not None and now - self._last_reach < REACH_INTERVAL_SECONDS:
            return
        self._last_reach = now
        self.reach_count += 1
        self.reaches.append((time.time(), " + ".join(f"{t_.band.centre:g} Hz" for t_ in self.rewards)))
        if self.reach_response == "tone":
            self.cue(cues.tone_at(REACH_PITCH_HZ), self.feedback.alert.volume)

    def set_reach_response(self, name: str) -> None:
        """Silent or a tone. Only the sound is switched — the marks, the count and the record are
        collected either way, the same rule a hidden pane follows."""
        self.reach_response = name if name in REACH_RESPONSES else "silent"
        self._last_reach = None

    def marks(self) -> list:
        return list(self.reaches)

    def toggle_pause(self) -> None:
        # Cleared, or a pause shorter than the hold window lets samples from before it count towards a
        # coincidence nobody was there for.
        super().toggle_pause()
        self._reach_log.clear()

    music_failing = ("a reward falls short",
                     "a bin over its threshold sounds the alert instead of pausing")

    def music_hold(self, met: bool) -> bool:
        """"Stops while failing" here means **a reward fell below threshold**, not the whole condition.

        The method's screens ship in two variants and the difference is exactly this: under `V-Inh` a bin over its
        threshold pauses the playback as well, under `A-Inh` it sounds an alert and the playback runs on.
        The manuals recommend the second — pausing on the bins "makes the video very jerky" and "will also not
        allow them to distinguish the event", since both failures then produce the same interruption. A
        reward below threshold pauses the playback in every variant of the method.

        The bins keep a consequence because this arrangement enables the alert in the same breath, so
        what changes is which of the two sounds they speak through.

        Quiet Reward (`stops_when == "met"`) keeps the whole condition, and must: there the silence is the
        reward, and it is earned only "when both Reward and Inhibit instruments meet criteria".
        """
        if self.feedback.music.stops_when == "failing":
            return any(t_.total and not t_.wanted for t_ in self.rewards
                       if self.target_is_live(t_))
        return super().music_hold(met)

    # --- what this protocol requires -------------------------------------------------------

    def wiring_problem(self) -> str | None:
        """Why the numbers would be wrong, if they would be. None when there is nothing to say.

        Ordered by how badly each one invalidates the screen, and only the worst is shown: a shared
        reference makes the reward meaningless, so saying anything else alongside it would be noise.
        """
        if self.pair is None:
            return "needs two channels; only one is in use"
        low = [t_ for t_ in self.rewards if t_.band.low < self.corner_hz]
        if low:
            names = ", ".join(t_.name for t_ in low)
            return (f"{names} sits below the device's {self.corner_hz:g} Hz high-pass. You then "
                    f"measure the filter's roll-off instead of the signal. Set High-pass to 0.01 Hz "
                    f"on the device screen. The filtering is in the hardware, so no software setting "
                    f"can fix it.")
        if self.device is None:
            return None                     # playing from a file: the wiring is whatever it was
        reference = getattr(self.device.status.settings, "reference", "")
        if reference and reference != "bipolar":
            return (f"reference is '{reference}', and this protocol needs 'bipolar' — a shared "
                    f"reference appears in both channels at once, so it adds in phase into the sum "
                    f"and inflates the reward")
        return None


def session_for(*args, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None,
                **kwargs) -> CFCSession:
    _ = heart_choice, mendi_choice, mendi_runner, pupil_choice, pupil_runner
    return CFCSession(*args, heart_runner=heart_runner, **kwargs)


def view(page: ft.Page, session: CFCSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    """Build once, mutate on Flet's loop."""

    signal = signal_pane.SignalPane(page, session)
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width(reserve: int = TARGETS_W) -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - reserve - 70)

    # The instantaneous spectrum, shared with every other screen that has one. It is not a duplicate of
    # the cascade beside it: the cascade shows *change over time* at a glance and this one has numbers on
    # both axes, which is what "find the frequency with the highest amplitude in a band" needs.
    spectrum = spectrum_pane.SpectrumPane(page, session, plot_width)
    trends = trends_pane.TrendsPane(page, session, plot_width)

    # ---- the cascading spectral display ----
    # `ft.Image.src` takes raw bytes, so the cascade goes straight across as a PNG with no base64 step.
    # FILL rather than CONTAIN because the aspect ratio here is meaningless — the axes are frequency and
    # time, both independently scaled, so preserving the pixel ratio of a 40×90 array would only letterbox
    # it.
    # NONE, not the default smoothing: a 40x90 array stretched to ~800x236 with bilinear filtering blurs
    # across *frequency*, which invents structure between 1 Hz bins that the transform never resolved.
    spec_image = ft.Image(src=None, height=SPECTROGRAM_H, expand=True, fit=ft.BoxFit.FILL,
                          filter_quality=ft.FilterQuality.NONE)
    spec_empty = ft.Text("waiting for a second of data", size=t.SMALL, color=t.FAINT)
    spec_scale, spec_labels = tick_axis(SPECTROGRAM_TICKS, tick_width=TICK_W)
    spec_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    ceiling_pills = ft.Row(spacing=6)
    ramp_strip = ft.Row(spacing=0)
    for i in range(24):
        ramp_strip.controls.append(
            ft.Container(width=9, height=8, bgcolor=t.amplitude_colour(i / 23)))

    def set_ceiling(uv: float) -> None:
        session.spectrogram.ceiling_uv = uv
        build_ceiling_pills()
        session.redraw()

    def set_depth(seconds: float) -> None:
        # Clearing is deliberate: the rows already held were sampled at the old interval, so keeping them
        # would put two different time scales in one picture — the axis would be lying about half of it.
        session.spectrogram.depth_seconds = seconds
        session.spectrogram.clear()
        build_ceiling_pills()
        session.redraw()

    def build_ceiling_pills() -> None:
        ceiling_pills.controls = [
            *[pill(f"{d:g} s", session.spectrogram.depth_seconds == d,
                   (lambda v=d: lambda _e: set_depth(v))())
              for d in DEPTHS],
            ft.Container(width=10),
            *[pill(f"{c:g} µV", session.spectrogram.ceiling_uv == c,
                   (lambda v=c: lambda _e: set_ceiling(v))())
              for c in CEILINGS]]

    # ---- the cross-frequency coupling instrument ----
    reward_canvas = cv.Canvas(shapes=[], expand=True, height=REWARDS_H)
    reward_axis, reward_labels = value_axis(REWARDS_H, count=4, width=AXIS_W)
    reward_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    # Always in the tree, never appearing and disappearing: a readout may not change the height of the
    # pane it sits in.
    reach_caption = ft.Text("", size=t.SMALL, color=t.FAINT, max_lines=1)
    reach_light = ft.Container(width=8, height=8, border_radius=4, bgcolor=t.SURFACE_HI,
                               tooltip="both rewards above their higher lines")

    # ---- raw traces, which double as the breathing pacer ----
    traces = trace_rows.TraceRows(page, session, lambda: plot_width(reserve=TARGETS_W), height=TRACE_H)

    # ---- the rewards ----
    reward_column = ft.Column(spacing=10)
    imbalanced: set = set()
    """Which rewards currently carry the amplitude verdict — the hysteresis state behind it."""
    reward_cards: dict[int, TargetCard] = {}
    reward_rows = ft.Column(spacing=12)

    def retune(freqs) -> None:
        session.retune_rewards(list(freqs))
        rebuild_rewards()
        signal.rebuild_contributions(session.targets)

    def set_freq(index: int, hz: float) -> None:
        freqs = [x.band.centre for x in session.rewards]
        if index < len(freqs):
            freqs[index] = hz
            retune(freqs)

    def rebuild_rewards() -> None:
        reward_cards.clear()
        cards = []
        for n, tgt in enumerate(session.rewards):
            # Same shape as an amplitude target card, because it is one: a colour bar, what it measures,
            # then the remove button at the far right — away from the threshold nudges, since one of those
            # discards the target and the others adjust it by 10%.
            wave = ft.PopupMenuButton(
                content=ft.Container(
                    content=ft.Row([ft.Text(wave_name(tgt.band.centre), size=t.BODY, color=t.TEXT),
                                    ft.Text("▾", size=9, color=t.FAINT)], spacing=6),
                    padding=ft.Padding(8, 4, 8, 4), border_radius=6,
                    bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
                items=[ft.PopupMenuItem(
                    content=ft.Text(f"{name} ({hz:g} Hz)", size=t.BODY),
                    on_click=(lambda i=n, h=hz: lambda _e: set_freq(i, h))())
                    for name, hz in WAVES.items()])
            identity = ft.Row(
                [ft.Container(width=3, height=18, bgcolor=tgt.colour, border_radius=2),
                 wave,
                 # The step sets how many decimals the box prints, so a sub-hertz card needs its own:
                 # 0.01 Hz shown to one decimal reads "0.0", and the ultra-low preset put exactly that on
                 # screen — a frequency the operator had just chosen from the menu, displayed as zero.
                 num_box(tgt.band.centre,
                         (lambda i=n: lambda v: set_freq(i, v))(),
                         suffix="Hz", step=0.01 if tgt.band.centre < SUBHERTZ_HZ else 0.1,
                         low=0.005, high=session.rate / 2 - 1, width=54),
                 # **The band this card is actually listening to, on the card.** The centre is what you
                 # set and the band is what is measured, and they came apart silently at the bottom of the
                 # range: a sub-hertz reward is bounded away from DC, so the width in force can be smaller
                 # than the width asked for, and the only mention of that was a line at the foot of the
                 # panel referring to "the lowest reward" — three cards away from the card it was about.
                 ft.Text(f"{tgt.band.low:g}–{tgt.band.high:g} Hz", size=t.SMALL, color=t.FAINT),
                 ft.Container(expand=True),
                 tiny("×", (lambda i=n: lambda _e: retune(
                     [x.band.centre for j, x in enumerate(session.rewards) if j != i]))())
                 if len(session.rewards) > 1 else ft.Container(width=24)],
                spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER)
            card = TargetCard(tgt, identity, rebuild_rewards, page=page,
                              modes=session.threshold_modes)
            reward_cards[id(tgt)] = card
            cards.append(card.control)
        reward_column.controls = cards

        # **A width the rewards on screen cannot use is not offered.** The sub-hertz row appears when a
        # sub-hertz reward exists and goes when the last one leaves, the same rule the shadow's state row
        # follows: a control nothing on this screen gives a meaning to should not be on this screen.
        subhertz = [x for x in session.rewards if x.band.centre < SUBHERTZ_HZ]
        reward_rows.controls = [
            labelled("width",
                     num_box(session.width_hz,
                             lambda v: (session.retune_rewards(width_hz=v), rebuild_rewards()),
                             suffix="Hz", step=0.05, low=0.01, high=4.0),
                     note=_delay_note(session, low=SUBHERTZ_HZ), width=54),
            *([labelled("subhertz width",
                        num_box(session.subhertz_width_hz,
                                lambda v: (session.retune_rewards(subhertz_width_hz=v),
                                           rebuild_rewards()),
                                suffix="Hz", step=0.01, low=0.005, high=0.8),
                        note=_subhertz_note(session, subhertz), width=54)]
              if subhertz else []),
            labelled("measure",
                     *[pill(k, session.measure == k,
                            (lambda k_=k: lambda _e: (session.retune_rewards(measure=k_),
                                                      rebuild_rewards()))())
                       for k in MEASURES],
                     note=MEASURES[session.measure], width=54),
            # Silent by default: under Quiet Reward every sound means something went wrong, so one that
            # means the opposite is armed deliberately or not at all.
            labelled("announce",
                     *[pill(k, session.reach_response == k,
                            (lambda k_=k: lambda _e: (session.set_reach_response(k_),
                                                      rebuild_rewards()))())
                       for k in REACH_RESPONSES],
                     note="when every reward crosses its higher line · at most one every 10 s",
                     width=64),
            # **No aim row.** Each card's gear sets its own now, so a panel row could only restate them —
            # first as one number that was a claim about the others, then as a range, which is a summary of
            # two figures you can see on the two cards above it. The total that went with it was arithmetic
            # anyone can do on those same two cards, and it was the *reason* for the aims rather than a
            # reading from the session: the bins have a total worth watching because there are twelve of
            # them and they move, which is why that one stays.
        ]
        page.update()

    # ---- the bin grid ----
    # Twelve full cards would be four screens tall, and the operator's question is a property of the
    # *set*: which bin breaks 1/f, and what do they total. That is a grid, not a list.
    bin_widgets: dict[int, dict] = {}
    bin_grid = ft.Column(spacing=0)
    total_text = ft.Text("—", size=22, color=t.TEXT, weight=ft.FontWeight.W_300)
    total_note = ft.Text("", size=t.SMALL, color=t.FAINT)

    def build_bin_grid() -> None:
        bin_widgets.clear()
        header = ft.Row([ft.Container(width=86, content=ft.Text("band", size=t.SMALL,
                                                               color=t.FAINT))]
                        + [ft.Container(width=132, content=ft.Text(f"ch{ch}", size=t.SMALL,
                                                                   color=t.FAINT))
                           for ch in session.specs],
                        spacing=8)
        rows: list[ft.Control] = [header, ft.Container(height=6)]
        for low, high, full in BINS:
            cells: list[ft.Control] = [
                ft.Container(width=86, content=ft.Column([
                    ft.Row([ft.Container(width=3, height=11, border_radius=2,
                                         bgcolor=t.band_colour(low, high)),
                            ft.Text(f"{low:g}–{high:g}", size=t.SMALL, color=t.TEXT)], spacing=5),
                    ft.Text(BIN_NAMES.get((low, high), ""), size=9, color=t.FAINT),
                ], spacing=1))]
            for ch in session.specs:
                tgt = next((x for x in session.bin_targets
                            if x.source == ch and x.band.low == low and x.band.high == high), None)
                if tgt is None:
                    cells.append(ft.Container(width=132))
                    continue
                value = ft.Text("—", size=t.BODY, color=t.TEXT, width=38)
                pct = ft.Text("", size=t.SMALL, color=t.MUTED, width=34)
                fill = ft.Container(width=0, height=6, border_radius=3,
                                   bgcolor=t.band_colour(low, high), left=0, bottom=0)
                notch = ft.Container(width=2, height=10, bgcolor=t.TEXT, left=0, bottom=-2)
                bar = ft.Stack([ft.Container(width=52, height=6, border_radius=3,
                                             bgcolor=t.SURFACE_HI, left=0, bottom=0),
                                fill, notch], width=52, height=10)
                bin_widgets[id(tgt)] = dict(value=value, pct=pct, fill=fill, notch=notch,
                                            full=full)
                cells.append(ft.Container(width=132, content=ft.Row(
                    [value, bar, ft.Container(width=6), pct], spacing=2,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER)))
            rows.append(ft.Container(content=ft.Row(cells, spacing=8),
                                     padding=ft.Padding(0, 5, 0, 5)))
        bin_grid.controls = rows

    bin_filter_row = ft.Row(spacing=6)

    def build_bin_filter() -> None:
        bin_filter_row.controls = [
            ft.Text("filter", size=t.SMALL, color=t.MUTED, width=54),
            num_box(session.bin_delay,
                    lambda v: (session.retune_bins(v), build_bin_filter(), page.update()),
                    suffix="s", step=0.05, low=0.05, high=1.0),
            ft.Text(f"delay on every bin · longer is sharper, and a veto can afford it",
                    size=t.SMALL, color=t.FAINT, expand=True),
        ]

    def redefine_all_bins(_e=None) -> None:
        for tgt in session.bin_targets:
            tgt.redefine()

    # ---- warnings: the wiring, and the heart splitting while the EEG looks fine ----
    warning = ft.Container(visible=False, bgcolor=t.SURFACE_HI, border_radius=8,
                           padding=ft.Padding(12, 9, 12, 9),
                           content=ft.Text("", size=t.SMALL, color=t.BAD))

    # ---- the heart, beside the EEG ----
    # The method's arrangement, and the reason for it is a specific failure: the EEG can improve while the HRV
    # splits, and a screen showing only one of the two cannot show that at all.
    # Built only when there is a source. Hiding an instrument that has nothing behind it would leave a
    # panel in the tree whose controls the draw loop keeps writing "—" into, and the first thing anyone
    # does when a hidden control misbehaves is wonder whether it is meant to be there.
    heart_ui = (heart_panel.HeartPanel(compact=True, width_of=lambda: max(200.0, TARGETS_W - 90))
                if session.heart is not None else None)
    heart_box = ft.Column(spacing=0, visible=heart_ui is not None)

    # ---- tabs ----

    # In the panel header, where the amplitude screen puts it. An octave above the highest reward is a
    # starting point that is usually somewhere useful, since the pairings the manuals name are wide apart.
    add_reward_button = ft.Container(
        content=ft.Text("+ add", size=t.SMALL, color=t.ACCENT),
        padding=ft.Padding(10, 5, 10, 5), border_radius=6,
        bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE),
        on_click=lambda _e: retune([x.band.centre for x in session.rewards]
                                   + [min(session.rate / 2 - 1,
                                          max(x.band.centre for x in session.rewards) * 2)]))

    train_left = ft.Column([
        panel("SPECTRAL DISPLAY",
              # **Positioned to fill, not `expand`.** `expand` is a Row/Column idea; a `Stack` lays an
              # unpositioned child out at its *intrinsic* size. The cascade's PNG is one pixel per bin by one
              # per row — about 44 wide by 72 tall — so the panel drew it at that shape: narrow, tall, and
              # nothing like the box reserved for it. Anchoring all four edges makes the child the size of the
              # Stack whatever the image happens to be.
              ft.Stack([ft.Container(content=spec_image, left=0, top=0, right=0, bottom=0),
                        ft.Container(content=spec_empty, alignment=ft.Alignment.CENTER,
                                     left=0, top=0, right=0, bottom=0)], height=SPECTROGRAM_H),
              ft.Container(content=spec_scale, padding=ft.Padding(0, 2, 0, 0)),
              ft.Container(height=4),
              ft.Row([ramp_strip, ft.Container(width=10),
                      ft.Container(content=spec_caption, expand=True)],
                     vertical_alignment=ft.CrossAxisAlignment.CENTER),
              trailing=ceiling_pills,
              help=help_button(page, "Spectral display",
                               "The cascade shows the two channels summed, scrolling down with the "
                               "newest row at the top. This is the display the state reads from.\n\n"
                               "Because the display sums the channels, it cannot show which channel a "
                               "feature comes from. Check the raw traces below for that.\n\n"
                               "The ceiling is a fixed setting, not automatic. This display must show "
                               "energy rising or falling. An automatic scale hides that change.")),
        ft.Container(height=12),
        panel("CROSS-FREQUENCY COUPLING",
              ft.Row([reward_axis, ft.Container(content=reward_canvas, expand=True, height=REWARDS_H)],
                     spacing=0),
              ft.Container(height=6), reward_caption, reach_caption,
              trailing=reach_light,
              help=help_button(page, "Cross-frequency coupling",
                               "The two rewards' amplitudes show on one shared scale. Coupling shows "
                               "as a shape: the two envelopes rise and fall together.\n\n"
                               "A separate scale for each reward makes any two signals look coupled. "
                               "One shared scale avoids this.\n\n"
                               "Each reward has a second, higher threshold, drawn as a dashed line in "
                               "its own colour.\n\n"
                               "The band along the bottom lights when every reward is above its higher "
                               "threshold at the same time. The dot beside the panel name does the "
                               "same. This moment is what the screen trains: two frequencies aligning "
                               "at the same instant, not one reward alone being loud.\n\n"
                               "If the tone is on, it marks the same moment. It sounds only after the "
                               "state holds for two seconds, and at most once every ten seconds. The "
                               "light therefore shows more often than the tone sounds.\n\n"
                               "The screen waits to announce a moment until both thresholds are final. "
                               "A threshold from a few seconds of a slow band is not reliable yet.")),
        ft.Container(height=12),
        panel("RAW", *traces.rows, ft.Container(height=6), traces.caption,
              help=help_button(page, "Raw traces and the pacer",
                               "The sweep lasts five seconds. Use it as a breathing pacer: breathe in "
                               "as the trace crosses, and out as it returns. This pace is six breaths "
                               "a minute, 0.1 Hz. The HRV screen's target starts at 0.1 Hz, though you "
                               "can set a lower target there.\n\n"
                               "An amber trace means the artifact gate rejects samples on that channel. "
                               "The screen still draws them, because a hidden rejection looks the same "
                               "as a dead electrode.")),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    if heart_ui is not None:
        heart_box.controls = [
            panel("HEART", *heart_ui.spectrum_rows(),
                  ft.Container(height=10), heart_ui.tiles,
                  ft.Container(height=8), heart_ui.source_note),
            ft.Container(height=12),
        ]

    train_right = ft.Column([
        panel("SYNCHRONY", reward_column, ft.Container(height=12), reward_rows,
              trailing=add_reward_button,
              help=help_button(page, "Synchrony rewards", SYNCHRONY_HELP)),
        ft.Container(height=12),
        panel("INHIBIT BINS",
              ft.Row([total_text, ft.Container(width=10),
                      ft.Column([total_note], spacing=0, expand=True)],
                     vertical_alignment=ft.CrossAxisAlignment.CENTER),
              ft.Container(height=12),
              bin_grid,
              trailing=ft.Container(
                  content=ft.Text("redefine all", size=t.SMALL, color=t.ACCENT),
                  padding=ft.Padding(10, 5, 10, 5), border_radius=6,
                  bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE), on_click=redefine_all_bins),
              help=help_button(page, "Inhibit bins", BINS_HELP)),
        ft.Container(height=8),
        bin_filter_row,
        ft.Container(height=12),
        heart_box,
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

    # ---- draw ----
    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        signal.refresh()
        spectrum.refresh()
        trends.refresh()
        session.push_displays(session.elapsed)

        # The wiring problem outranks the split, because with a shared reference the reward is wrong and
        # nothing else on the screen means what it says.
        trouble = session.wiring_problem()
        if heart_ui is not None:
            heart_ui.refresh(session.heart)
            if trouble is None and session.heart.split:
                trouble = ("the HRV has split — high-frequency and very-low-frequency peaks with "
                           "0.1 Hz missing, which is one foot on the gas and one on the brake. It can "
                           "happen while the EEG looks like it is improving.")
        warning.visible = trouble is not None
        if trouble:
            warning.content.value = trouble

        # the cascade
        image = session.spectrogram.image()
        spec_image.src = image
        spec_image.visible = image is not None
        spec_empty.visible = image is None
        sg = session.spectrogram
        x_of = bin_centre_x(sg.lo_hz, sg.bin_hz, sg.bins, plot_width(), top=sg.hi_hz)
        width = plot_width()
        # And an explicit width on the image itself, so the picture cannot fall back to its own pixel
        # dimensions if the Stack ever changes its mind about how it sizes children. It is the same width
        # the tick axis below is placed against, which is what keeps a label under its bin.
        spec_image.width = width
        place_tick_labels(spec_labels, SPECTROGRAM_TICKS, x_of, tick_width=TICK_W, width=width)
        so_far = sg.seconds_filled * sg.depth_seconds
        spec_caption.value = (
            f"ch{session.pair[0]}+ch{session.pair[1]} summed · "
            f"{sg.lo_hz:g}–{sg.hi_hz:g} Hz in {sg.bin_hz:g} Hz bins · "
            f"0–{sg.ceiling_uv:g} µV full scale · {sg.depth_seconds:g} s deep"
            + ("" if sg.seconds_filled >= 1.0 or so_far < 1.0
               else f" · {so_far:.0f} s of it so far")
            if session.pair else "needs two channels")

        # the coupling instrument
        peak = plot_ceiling(session.reward_plot.peak(),
                            *[x.threshold for x in session.rewards],
                            *[x.reach_threshold for x in session.rewards])
        colours = [x.colour for x in session.rewards] or [t.ACCENT, t.ACCENT_DIM]
        reward_canvas.shapes = session.reward_plot.shapes(
            plot_width(reserve=TARGETS_W + AXIS_W), REWARDS_H, peak, colours,
            thresholds=[x.threshold for x in session.rewards],
            higher=[x.reach_threshold for x in session.rewards],
            marks_colour=t.GOOD)
        place_value_labels(reward_labels, REWARDS_H, peak, unit="µV")
        reward_caption.value = (f"{REWARDS_SECONDS:g} s across · 0–{peak:g} µV full scale · "
                                f"thin line = threshold, dashed = the higher one")
        reach_light.bgcolor = t.GOOD if session.coincident else t.SURFACE_HI
        reach_caption.value = _reach_line(session)

        # the traces, which are also the pacer
        traces.refresh(PACER_SECONDS, extra=("in as it crosses, out as it returns",))

        # the rewards, each with how much of its range phase can actually move — a summed reward stops
        # being a phase measurement when the two channels are unbalanced, and the fix for that is an
        # electrode rather than a threshold
        for tgt in session.rewards:
            card = reward_cards.get(id(tgt))
            if card is None:
                continue
            metric = session.sync.get(round(tgt.band.centre, 6))
            card.refresh(session.value_for(tgt), tgt.unit,
                         metric.delay_seconds if metric is not None else None)
            if metric is None or not any(metric.parts.values()):
                continue
            # "of the reading", and the imbalance names both channels and its units, because "phase moves
            # 81%" and "ch0 is 1.5×" are both meaningless without saying 81% of what and 1.5 times what.
            sensitivity = metric.sensitivity
            loud = metric.louder_channel
            quiet = next((c for c in metric.parts if c != loud), None)
            card.scale_text.value += f" · phase moves {sensitivity:.0%} of it"
            # A verdict — this is not really a phase measurement — so the note, not the scale line. It
            # arrives below one line and leaves above a higher one, so a reading wandering across the
            # floor does not make it blink.
            if sensitivity < MIN_SENSITIVITY:
                imbalanced.add(id(tgt))
            elif sensitivity > CLEAR_SENSITIVITY:
                imbalanced.discard(id(tgt))
            if id(tgt) in imbalanced and loud is not None and not card.note.value:
                card.note.value = (f"mostly amplitude, not phase: ch{loud} {metric.parts[loud]:.1f} µV "
                                   f"against ch{quiet} {metric.parts[quiet]:.1f}")

        # the bins
        breaking = {id(x) for x in session.breaks_one_over_f()}
        for tgt in session.bin_targets:
            wid = bin_widgets.get(id(tgt))
            if wid is None:
                continue
            value = session.value_for(tgt)
            wid["value"].value = f"{value:.1f}"
            wid["value"].color = t.BAD if id(tgt) in breaking else (
                t.TEXT if tgt.wanted else t.WARN)
            ceiling = wid["full"]
            wid["fill"].width = 52 * min(1.0, value / ceiling)
            wid["notch"].left = 52 * min(1.0, tgt.threshold / ceiling) - 1
            over = tgt.recent_percent()
            wid["pct"].value = "—" if over is None else f"{100 - over:.0f}%"
            wid["pct"].color = t.WARN if (over is not None and 100 - over > 3) else t.MUTED

        if session.total_ready:
            total = session.total_above
            total_text.value = f"{total:.0f}%"
            lo, hi = TOTAL_RANGE
            total_text.color = t.GOOD if lo <= total <= hi else (
                t.WARN if total > hi else t.MUTED)
            total_note.value = (
                f"of the time above threshold, all {len(session.bin_targets)} bins added together · "
                f"aim {lo:g}–{hi:g}% · "
                + ("too often — raise the thresholds" if total > hi else
                   "below the range" if total < lo else "in range"))
        else:
            total_text.value = "—"
            total_text.color = t.FAINT
            total_note.value = f"each bin aims at {1 - BIN_HIT_RATE:.1%} above threshold"

        page.update()

    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)

    build_ceiling_pills()
    build_bin_filter()
    build_bin_grid()
    rebuild_rewards()
    signal.rebuild_contributions(session.targets)
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)


def _delay_note(session: CFCSession, low: float = 0.0, high: float = float("inf")) -> str:
    """The delay, as a range when the rewards this width governs differ.

    The band limits are what make each row describe its own rewards: the ordinary width governs everything
    at a hertz and up, the sub-hertz width everything below, and one figure covering both would read
    "0.5–8 s" under a control responsible for neither end of it.
    """
    delays = sorted(m.delay_seconds for f, m in session.sync.items() if low <= f < high)
    if not delays:
        return "no pair" if session.pair is None else "no reward at this width"
    if delays[-1] - delays[0] < 0.05:
        return f"{delays[0]:.1f} s of delay"
    return f"{delays[0]:.1f}–{delays[-1]:.0f} s of delay"


def _subhertz_note(session: CFCSession, rewards: list) -> str:
    """What the sub-hertz width comes to, including where it could not be granted.

    The clamp is stated **as a fact about a named reward**, not as a warning at the foot of a panel: at
    0.01 Hz nothing wider than 0.008 Hz can be measured, and an operator who types 0.05, watches the box
    keep saying 0.05, and reads 0.008–0.012 on the card is owed that sentence rather than left to derive it.
    """
    note = _delay_note(session, high=SUBHERTZ_HZ)
    bounded = [x for x in rewards
               if session.width_for(x.band.centre) < session.subhertz_width_hz - 1e-9]
    if not bounded:
        return note
    worst = min(bounded, key=lambda x: x.band.centre)
    return (f"{note} · {worst.name} is bounded to "
            f"{session.width_for(worst.band.centre):g} Hz, because a wider band here reaches 0 Hz")


def _reach_line(session: CFCSession) -> str:
    """The second caption line: how much of the drawn window was marked, how many moments were
    announced, and how well the two envelopes are moving together.

    Says what it is waiting for rather than blanking or counting down, because a caption that goes empty
    reads as a broken instrument and one that counts down describes a defect."""
    if any(t_.reach_rate is None or t_.provisional for t_ in session.rewards) or not session.rewards:
        return "both above the higher line — waiting for both thresholds"
    share = session.reward_plot.mark_fraction()
    if share is None:
        return "both above the higher line — waiting for the window"
    return (f"both above the higher line {share:.0%} of the window · "
            f"{session.reach_count} announced{_moving_together(session)}")


def _moving_together(session: CFCSession) -> str:
    """A number for what the instrument shows as a shape. Correlation of the two envelopes over the
    window, reported only when there is a window — it is a summary of the picture, not a substitute."""
    series = session.reward_plot.series()
    if len(series) < 2 or len(series[0]) < 40:
        return ""
    a, b = series[0], series[1]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return ""
    r = float(np.corrcoef(a, b)[0, 1])
    return f" · they are moving together at r = {r:+.2f}"
