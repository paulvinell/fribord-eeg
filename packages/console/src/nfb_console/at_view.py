"""Alpha-Theta.

Eyes closed, one channel at Pz, theta and alpha both rewarded, high beta held down. The state being
trained is the **crossover** — theta amplitude rising past alpha over twenty or thirty minutes — and the
whole screen exists to make that one comparison readable while the trainee cannot look at it.

**The shape of the protocol.** One session plan and one feedback design, so this screen is unusually well
pinned down:

- **The bands**: two raw display channels, theta and alpha as rewards, high beta as an inhibit. Theta is
  2-7 Hz, alpha 8-11 Hz and high beta 22-36 Hz, at site `PZ-A1`. A second variant splits theta as 5-8 with a
  2-5 inhibit under it, which is why the bands are editable here rather than fixed.
- **The aims**: 25% of the time above threshold for theta, 55% for alpha, 15% *over* for the inhibit, with a
  **6 µV floor** under the inhibit's threshold. Theta being the harder of the two is the point of the
  arrangement: it is the one you are trying to bring up.
- **The feedback**: two *state* sounds crossfaded, surf and a babbling brook, with a transition rate; two
  *reward* sounds, one per band, at different pitches; a minimum interval between rewards; and a reward
  lockout after a run of artifacts.

**More than one electrode is summed, and the bands are measured on the sum.** One site is the shipped
montage, and running four sites as four independent alpha-theta instruments would be a different protocol:
there is one crossover to read, not four, and four sets of reward tones sounding against each other is not
feedback. Two
consequences worth knowing: the summed amplitude scales with the channel count, so no threshold transfers
between two channels and four (they are all derived from data, so nothing breaks); and one bad electrode is
now *inside* the number rather than visible as one bad trace, which is why a channel not delivering signal
is named on the screen rather than left to the traces nobody is watching.

Three things follow from that last item, and they are what makes this protocol differ from every other
screen here.

**The rewards are alternatives, not conditions.** Theta and alpha are two states you can be in one of, so
combining them with an average would ask for both at once and report half a signal when the trainee is
firmly in one. The rule is an OR with a preference: theta wins when both are available, and a switch
turns that into "whichever is doing better". That is `combine="either"` here, and
it is the one combining rule under which a discrete output fires on one reward rather than on all of
them — see `feedback.combine`.

**The continuous feedback carries the state, not the score.** Nothing in the design gets louder as you do
better: the two soundscapes fade into each other as the balance moves, and that is the whole continuous
channel. So the music is a `crossfade` — brown noise at the theta end, pink at the alpha end — and the
system volume and the shadow are **off**. The shadow would be off anyway: the trainee's eyes are closed,
and a screen brightening at someone with their eyes shut is an output into a closed channel.

**The reward is a pitched tone per band**, low for theta and high for alpha, which is `Click.keyed` plus
`cues.reward_tone`. The two anchor pitches are **137 Hz and 738 Hz**, two and a half octaves apart, because
with the eyes closed the pitch is the only thing saying which of the two alternatives earned it. Those two
pitches are the anchors our mapping is built on; a third reward between them lands between the two.
This is the audible twin of `theme.frequency_colour` and of the alert, one direction over — the alert says
which band is in trouble, this says which one is earning — but *not* the same sound: the alert renders a
few milliseconds of band-limited noise, which is heard as a click and not as a pitch, and across theta and
alpha its two versions come out under an octave apart. That was what this screen played until it was
measured.

**Design choices worth stating.**

- The soundscapes are *generated* rather than played from recordings. The two ends differ mostly in
  spectral tilt, with surf-like energy near DC and brook-like energy around 800 Hz. So brown against pink is
  the same distinction, and it inherits `music.NoisePlayer`'s reason for existing: a loop of surf has a
  period you eventually start hearing, and generated noise does not.
- **The θ/α ratio and the crossover readout are ours.** The display is two amplitude trends and the
  operator's eye. Putting a number on the thing they are being compared for is the obvious next step
  rather than a different measurement.
- **The session has no ~3 minute periods with a score screen between them.** What stands in is the
  Session tab's length and reminder interval, which is a weaker thing and is why the length defaults to 30
  minutes here instead of being open-ended.

**Thresholds hold still by default, and `automatic` is available.** Autothresholding is off by default,
and a threshold tracking a rolling window follows theta up as it rises, pinning the hit rate where it
started — which matters more here than elsewhere because the pitched tone is the whole discrete channel, so
its density is the only thing a trainee with closed eyes has that says the session is going anywhere. Every
target therefore starts `fixed`, derived once from the first 30 seconds.

That is a default and not a requirement, and it took an argument to get right: automatic thresholds also
work on this protocol, at the cost of the tone's density. `threshold_modes` is for what a protocol *requires*;
CFC keeps its lock because the quantity it trains **is** time above threshold, which a tracker pins by
definition, while here the trend and the ratio are computed from the trends and survive either way.
"""

from __future__ import annotations

import time
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import Band, ChannelAligner
from nfb.stream import NO_SIGNAL, OK, Block

from . import chrome, cues, music, plan_pane, signal_pane, spectrum_pane, target_card, trace_rows, trends_pane
from . import theme as t
from . import plots
from .plots import (EnvelopePlot, nice_ceiling, place_value_labels, plot_ceiling,
                    value_axis)
from .protocols import Protocol
from .target_card import TargetCard
from .targets import PAIR, Target
from .traces import RawRing, TraceBuffer
from .feedback import CONDITION, LEVEL
from .training import TREND_SECONDS, TrainingSession
from .widgets import help_button, num_box, panel, pill, test_button, tiny

HANDLES = ("A-T",)

# Alpha is the same band in every variant, and it is the anchor rather than the thing being
# changed: `[8, 11]` at a 55% aim, reached most of the time.
ALPHA = (8.0, 11.0, 0.55)

# **Three axes.** The two coherent variants differ along exactly these lines: `wide + no guard + 22` and
# `narrow + guard + 15`. They are offered independently because the combinations in between are legitimate;
# what is *not* offered is a combination nobody is told about, which is what `overlap_warning` is for.
#
# Theta's aim is 25% against alpha's 55% either way. That asymmetry is the arrangement, not a property of
# the band: theta is the one being brought up, so its threshold sits where it is reached rarely.
THETA_BANDS = {
    # `[2.0, 7.0]`. Wide, and it reaches down into delta — which is why the guard below it
    # has nowhere to go.
    "wide": (2.0, 7.0),
    # `[5.0, 8.0]`. Abuts the guard exactly at 5 Hz, so no energy both earns and is punished.
    "narrow": (5.0, 8.0),
}

# The sleep guard. `[2, 5]` at `[15, 6, 250]` — 15% over threshold, 6 µV floor. It exists
# because the corridor has two walls: high beta says *too awake*, and this says *too far gone*, which
# under a protocol that rewards theta is otherwise rewarded. Off by default, because the plan we
# implemented first is the one without it.
GUARD_BAND = (2.0, 5.0, 0.85)

# The high inhibit's lower edge, and the aim moves with it because a wider band is over threshold more
# often: `[22, 36]` pairs with `[15, 6, 250]` (85% met) and `[15, 36]` pairs with
# `[20, 6, 250]` (80% met). Reading one aim onto the other band would be inventing a number.
HIGH_BANDS = {22.0: (22.0, 36.0, 0.85), 15.0: (15.0, 36.0, 0.80)}

# There is deliberately **no summary line naming the combination.** Two were tried and both were wrong in
# the same way: a layout code names a slot count that means nothing on the screen, and a
# phrase like "one wall" is a metaphor the screen then has to teach. The pills already say `2–7 wide`,
# `off`, `22–36` — that *is* the arrangement, stated in the unit it is set in. What the walls are for
# belongs in the `?`, which is read once. The only thing the line is kept for is saying when the bands have
# been set by hand, because derived pills then show nothing selected and that needs a word.

# **Two objectives, because there are two descriptions of the same state and they are not equivalent.**
#
# `bands`: theta and alpha scored separately against their own
# thresholds, as *alternatives* — be in one of the two states. It is what the aims 25% and 55% belong to.
#
# `difference` is θ−α as one quantity, trained upward (`In1-In2`, on a ±100 readout). It is a genuinely
# different claim: it cannot be satisfied by everything getting bigger, and it cannot be satisfied by
# strong alpha alone.
#
# They are chosen **per output channel**, because the two channels ask different questions. A continuous
# fade wants a quantity that moves smoothly and says *how far toward* the state you are, which is what a
# difference is; a discrete tone wants "you are in one of these two states now", which is what the
# alternatives are. Forcing one answer onto both was the arrangement until this existed.
BANDS_OBJECTIVE = "bands"
DIFFERENCE_OBJECTIVE = "difference"
OBJECTIVES = (BANDS_OBJECTIVE, DIFFERENCE_OBJECTIVE)

# The difference target's aim. Theta's 25% is the closest analogue, and for the same reason — this is the quantity
# being brought up, so its threshold belongs where it is reached rarely.
DIFFERENCE_AIM = 0.25

# **What the sleep guard *does* when it fires, and there are three answers.** The band is only half the
# design; the other half is what the trainee perceives, and with the eyes closed that is the whole of it.
#
# - `silence`: the guard is an ordinary inhibit, so its only effect is that the reward tones stop.
# - `knock`: a discrete, low, repeated sound, like a knock on the door. About 1.63 s, peaking near 266 Hz.
# - `knock + hush`: the knock, and all other sounds fade out while the guard is over.
#
# `silence` is ambiguous and that is the argument for offering the others: a tone that stops could mean
# neither reward is met *or* that a guard fired, and those call for opposite corrections — go deeper against
# come back. A sound that is only ever the guard cannot be confused with either.
#
# The knock is `cues.frequency_click` at the guard band's own centre, so it is a low thock by derivation
# rather than by a chosen asset — the same mapping the alert uses. `hush` drops the music while the guard
# is over, which makes the knock arrive into silence instead of over music.
GUARD_RESPONSES = ("silence", "knock", "knock + hush")

# How often the knock repeats while the guard is over. The knock is 1.63 s of sound, so a repeat
# faster than that would overlap itself; three seconds leaves a gap you can hear.
GUARD_KNOCK_SECONDS = 3.0

# A floor under every inhibit's threshold, matching the aims `[15, 6, 250]` and `[20, 6, 250]`.
# Without it one quiet stretch drives the threshold toward zero and the inhibit then fires for the rest of
# the session — which under this protocol means the reward tones stop and nothing says why.
INHIBIT_FLOOR_UV = 6.0

# A floor under the *signal*, not under the threshold. The reward is gated on a 1.0 µV signal level upstream
# of the comparison, and the failure it prevents is specific: a derived threshold on a
# channel with nothing on it calibrates to nearly nothing, so the target is met about as often as its aim
# asks and the tone sounds all session on noise. See `Target.signal_floor`.
SIGNAL_FLOOR_UV = 1.0

# Reward tones no closer together than this. The value is ours; it is slow enough that
# consecutive tones read as separate events.
REWARD_INTERVAL = 3.0

# How long the condition has to hold before the reward tone doubles. The criterion is a sustained
# reward and the number is three seconds. With the interval also at 3 s that means one single tone before
# the doubles start, which is
# brief — the alternative is two intervals, and it is a settable number either way.
HOLD_SECONDS = 3.0

# **The music alone is this protocol's, and it is written here because it has a reason.** Every other
# calibrated default now comes from `feedback` unchanged — see `CLICK_VOLUME`, where three disagreeing
# numbers became one. The tone is the shared mark at the shared gain; the argument that it had to be louder
# than everyone else's mark was really an argument about the *music* under it, and that is what this line
# sets.
#
# 20 rather than the shared 60 because on this protocol the music plays continuously for an hour with the
# eyes closed, and it is not the reward — the reward is the tone arriving over it. The music is also
# **generated well down** (see `music.HEADROOM`), which is what lets the control sit at 20 rather than the 5
# that broadband noise at full scale would need; 5 has nowhere to go, and the whole point of music here is
# that the trainee can have it set where they want it.
#
# **On the operator's scale, and it does not move when the generator's pad does.** The pad went three times
# deeper because the noise engine was three times too loud, and every screen that generates noise was
# getting that engine — so this music is now three times quieter at the same 20, which is the fix reaching
# here rather than a regression. A user-facing number set by ear stays where it was set.
MUSIC_VOLUME = 20.0

# The Spectrum tab's ring. 1-45 Hz: below is drift and above is mains, and the highest band on this screen
# is the 22-36 Hz inhibit, so nothing being measured sits near the edge.
SPECTRUM_LO, SPECTRUM_HI = 1.0, 45.0
SPECTRUM_BIN_HZ = 1.0

# Which end of the music is which. Theta is the slow, low end and alpha the brighter one, matching the pair
# of recordings it fades between — surf into a babbling brook.
MUSIC_ENDS = {"theta": "brown", "alpha": "pink"}

SESSION_MINUTES = 30.0      # eyes closed is the case a session limit exists for

TRACE_H = 92
TRACE_SECONDS = 5.0
"""How much of the raw signal is on screen. Stated once because the caption and the buffer have to agree —
they were a hardcoded 5.0 in the buffer and a hardcoded "5 s" in the words beside it."""
CROSSOVER_H = 200
CROSSOVER_SECONDS = 600.0   # ten minutes, because the crossover is a change over tens of minutes
TARGETS_W = 430
DIFF_BAR_W = 190        # the θ−α strip's bar; signed, so it is centred rather than filled from zero
AXIS_W = 40
RATIO_WINDOW = 30.0         # what the live ratio is averaged over; a raw ratio of two envelopes flickers

# **What the music's balance is averaged over, and it is not a cosmetic smoothing.** The two band envelopes
# flicker through an order of magnitude, so music following them instantaneously swaps ends several times a
# second while the state it stands for moves over half an hour. A crossfade period of 30 s is the
# same figure `RATIO_WINDOW` arrived at for the readout, and the music
# not using it was an oversight rather than a choice.
BALANCE_SECONDS = 30.0

# **There is deliberately no choice of quantity here, and a `share`/`difference` pair of pills was removed.**
# The crossfade could be driven from θ−α (`Out1=(In1-In2)*In3` into a 0…1 clamp) or from θ/(θ+α),
# which looked like two options worth offering. It is one option: normalising θ−α by the pair's own size — the
# only scale available that does not invent a constant — gives `(θ−α)/(θ+α)`, and that is identically
# `2·share − 1`. So the "difference" branch computed `2·share − 0.5`, an affine function of the share with a
# gain of 2. Verified against the running session: share 0.626, "difference" 0.751, and 2(0.626) − 0.5 = 0.752.
#
# Two pills naming the same quantity are worse than no pills: the label has to be cryptic because there is no
# real distinction to describe. What *is* real is the gain — how big a swing traverses the fade — and if that
# is wanted it is one number on the Music pane beside the music's ends, not a choice of arithmetic here.

CROSSOVER_HELP = (
    "Theta and alpha amplitude share one axis and one scale, so you can compare them directly.\n\n"
    "With more than one electrode, the channels are summed first. There is one crossover to read, not "
    "one per site. The channels in the sum are named beside the bands.\n\n"
    "A summed reading increases with the number of channels. Do not compare thresholds between sessions "
    "that use a different number of channels.\n\n"
    "The ratio shown is theta over alpha, averaged over 30 seconds. Above 1.0, theta is larger than "
    "alpha. The ratio usually starts below 1.0 and rises over twenty to thirty minutes. If a session "
    "crosses in the first two minutes, check that the eyes are closed and that the alpha band is placed "
    "correctly.\n\n"
    "Both theta and alpha train upward. Theta is asked for less often than alpha, because theta is the "
    "band you are raising. The high beta band is held down under both bands. It rises when you tense up "
    "or drift toward sleep.\n\n"
    "By default, thresholds are set once, from the first 30 seconds, and then held fixed. An automatic "
    "threshold follows a rolling window, so it moves up with theta and hides the change. This "
    "protocol exists to produce that change, so keep the threshold fixed."
)

# The two objectives as the screen names them, and the pairings spelled out. Labelled by what each one
# *asks for* rather than by "bands" and "difference", which are names for how they are computed.
OBJECTIVE_LABELS = {BANDS_OBJECTIVE: "either band", DIFFERENCE_OBJECTIVE: "θ−α"}

# One line per pairing, because the mixed cases are the point of having two rows and a reader who has just
# pressed one pill needs to see what the pair now means. Every combination has an entry: a description that
# does not depend on the mode is the failure `COMBINES` was keyed for.
OBJECTIVE_NOTES = {
    (BANDS_OBJECTIVE, BANDS_OBJECTIVE):
        "both read theta and alpha as alternatives",
    (DIFFERENCE_OBJECTIVE, DIFFERENCE_OBJECTIVE):
        "both read theta minus alpha, so nothing is earned by overall amplitude",
    (DIFFERENCE_OBJECTIVE, BANDS_OBJECTIVE):
        "the fade shows how far toward theta; the tone marks being in either state",
    (BANDS_OBJECTIVE, DIFFERENCE_OBJECTIVE):
        "the fade follows whichever band is doing better; the tone needs theta above alpha",
}

OBJECTIVE_HELP = (
    "Two ways to score the same training, and they are not the same claim.\n\n"
    "Either band scores theta and alpha against their own thresholds, and uses whichever is higher. This "
    "is what the 25% and 55% aims apply to.\n\n"
    "θ−α scores one quantity: theta's amplitude minus alpha's, trained upward. A rise in both bands "
    "together does not satisfy it, and a strong alpha alone does not satisfy it either. Its threshold is "
    "derived from the first 30 seconds, like every other threshold. Its aim is 25%, by analogy with "
    "theta's aim; no published figure exists for it.\n\n"
    "You set this per output, because the fade and the tone ask different questions. The fade shows how "
    "far toward the state you are. The tone fires when you are in the state now. The high beta band and "
    "the sleep guard apply to both outputs either way.\n\n"
    "Theta and alpha keep their own cards regardless of this choice. The crossover plot shows whichever "
    "objective is chosen. The tone's pitch is still read from the bands."
)

BANDS_HELP = (
    "This screen offers two preset band arrangements. 2-7 Hz theta, no sleep guard, and high beta from "
    "22 Hz is one. 5-8 Hz narrow theta, with the guard, and high beta from 15 Hz is the other. You can "
    "set anything between these.\n\n"
    "The sleep guard is a second limit. High beta rising shows you are too alert. 2-5 Hz rising shows you "
    "are too drowsy. The guard stops the drowsy state from earning the theta reward. It is off by "
    "default.\n\n"
    "The narrow theta band, 5-8 Hz, exists for the guard. The wide theta band, 2-7 Hz, reaches into the "
    "frequencies the guard watches. Combining the wide band with the guard makes the same energy earn a "
    "reward and trigger the guard at once. So the protocol cannot be satisfied. The narrow band instead "
    "sits next to the guard, at 5 Hz. The screen locks each combination that cannot work, and marks it "
    "in both directions.\n\n"
    "When **sleep** is on, a row called 'when it fires' appears. It sets what the guard does. 'silence' "
    "stops the reward tones. It is an ordinary inhibit, so a stopped tone can also mean a reward is not "
    "met. 'knock' plays a low, repeated sound that always means the guard has fired. 'knock + hush' also "
    "fades the music out while the guard is active, so the knock plays into silence.\n\n"
    "You can still edit every band edge. If you drag theta's lower edge under an active guard, the "
    "screen marks the overlap in red instead of blocking it.\n\n"
    "The high beta threshold recalculates when you move its lower edge. It targets 85% met at 22-36 Hz, "
    "and 80% met at 15-36 Hz, because a wider band crosses the threshold more often.\n\n"
    "The music's fade is theta divided by theta plus alpha, averaged over 30 seconds. Its midpoint lines "
    "up with the crossover. The 30-second average keeps the fade steady instead of switching several "
    "times a second.\n\n"
    "Every band edge on the cards below is a starting point. Editing a preset marks it as edited, and it "
    "no longer claims to be one of the two arrangements above."
)

FEEDBACK_HELP = (
    "This protocol trains with your eyes closed, so every output is a sound.\n\n"
    "The music fades between two characters of noise as the balance between theta and alpha shifts. It "
    "is deeper as theta rises, and brighter as alpha rises. This fade is the continuous feedback. No "
    "output gets louder as you improve.\n\n"
    "A tone also sounds whenever either band is above its threshold and the high beta band is not. The "
    "tone's pitch marks which band earned it: low for theta, higher for alpha.\n\n"
    "The screen shadow and the master volume are off. The shadow is a visual output, and your eyes are "
    "closed. The master volume stays fixed. The noise character already shows the state, so the volume "
    "does not need to show it as well."
)


def _mean_recent(tgt, seconds: float) -> float | None:
    """A target's mean value over the last `seconds`, from its own trend, or None before there is any.

    Read off `Target.trend` rather than kept as a second buffer, for the reason the ratio readout is: the
    trend already advances at `TREND_SECONDS` and a parallel history is a second thing to keep in step.
    """
    points = list(tgt.trend)[-max(1, int(seconds / TREND_SECONDS)):]
    return float(sum(points) / len(points)) if points else None


class ATSession(TrainingSession):
    """Theta and alpha as alternatives, high beta down, and the crossover between them."""

    sustains_reward = True
    """This protocol has a momentary reward event and a three-second sustained one beside it, so
    the distinction the Click panel's "doubles after" row configures belongs here — see
    `TrainingSession.sustains_reward` and `HOLD_SECONDS`."""

    tunes_envelope = True
    """`envelope_seconds` is part of this screen's estimator identity — `filter_for` keys on it
    and `retune_filter` rebuilds when it moves, which is what makes the SIGNAL panel's first step a
    live control here and absent on the lock-in screens."""

    # The crossfade is the *whole* continuous channel here, so it is the only music on offer — see the
    # module docstring. A fixed tilt would be the fade frozen at one end, a backing track carries nothing
    # at all, and the EEG as a beating tone is a different protocol's idea of what to listen to. Offering
    # them would be offering three ways to switch the feedback off.
    # Everything `_apply_feedback` sets, and nothing else. The arrangement *is* the protocol here: two
    # soundscapes fading between two states, a tone pitched by whichever band earned it, and no shadow or
    # master volume at all — each for a stated reason, none of them a preference. The volumes, the tone
    # interval and the auditions stay live, because those are the room rather than the design; the
    # session is run with the eyes closed, so the first hearing would otherwise be the one moment nothing
    # can be adjusted. `arrangements` is empty: the named arrangements are alternatives to *this*, not settings
    # within it.
    # The shadow is **off by default and not pinned.** Eyes closed is the reason it starts off — an
    # output into a closed channel — but it is not a reason nobody may ever turn it on: a dim room, a
    # trainee who opens their eyes at the end, an operator watching the screen themselves. Off-by-default
    # and unavailable are different claims, and only the first one is true here.
    # `either` and nothing else: its rewards *are* alternatives, so the other three rules would each be a
    # different claim about them. One entry, so the row states it rather than offering it.

    # **`fixed` is the default and `automatic` is available.** This was `("fixed",)` — locked — and the
    # lock was over-reach.
    #
    # The default is fixed for a real reason: a threshold tracking a
    # rolling window follows theta up as it rises, pinning the hit rate where it started. On this screen
    # the pitched tone is the *whole* discrete channel — the level drives nothing and the music carries state
    # rather than score — so its density is the only thing a trainee with closed eyes has that says the
    # session is going anywhere, and a tracker holds that density constant by construction.
    #
    # But that is an argument for a default, not for a lock: automatic thresholds also work on this protocol,
    # and `threshold_modes` exists for requirements — CFC keeps its lock because the quantity it trains *is*
    # time above threshold, which a tracker pins by definition. Here the trend and the ratio readouts are
    # computed from the trends, so a tracker costs the tone's density and nothing else measurable. Anyone
    # turning our tone off has no density to preserve and should have the tracker.
    threshold_modes = ("fixed", "automatic")
    combines = ("either",)
    # Theta first is the arrangement: theta is the band
    # being brought up, and the bias is what makes its tone the one you hear when you have got there.
    tone_bias = ("theta first", "whichever is better")
    pinned_feedback = frozenset({"combine", "sound", "click", "alert", "music"})
    music_sources = ("crossfade",)
    state_ends = ("alpha", "theta")           # low end first, matching `colour` / `second_colour` below

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0):
        super().__init__(protocol, path=path, device=device,
                         length_seconds=length_seconds, source=source, channels=channels or 1)
        # Eyes closed with nothing to watch is exactly the case a session length exists for, and the case
        # that also wants the cues — a limit nobody can see is not a limit. So this is the one protocol
        # that opens bounded rather than open-ended.
        if length_seconds is None:
            self.plan.set_length(SESSION_MINUTES)

        # **Every channel in use is summed, and the bands are measured on the sum.** One site is the
        # montage, and running two or four sites as independent alpha-theta instruments would be a
        # different protocol: there is one crossover to read, not four, and four sets of reward tones
        # sounding against each other is not feedback. Summing keeps it one instrument on a wider view of
        # the head, with the same reward streams.
        #
        # The sum is over *aligned* samples, which needs an aligner rather than an addition: blocks arrive
        # one channel at a time, and adding whatever has most recently landed would drift the channels
        # against each other by up to a frame.
        self.sites = tuple(self.specs)
        self.site = next(iter(self.specs), 0)         # whose spec the summed signal borrows
        self.aligner = ChannelAligner(self.sites) if self.sites else None
        self.traces = {i: TraceBuffer(320, TRACE_SECONDS, s.rate_hz)
                       for i, s in self.specs.items()}
        # Rings for the Spectrum tab: the **sum first**, because the sum is what every band here is
        # measured on, then each channel — a flat electrode is invisible inside a sum, and the crossover
        # is read off amplitudes that come from it.
        self.rings = {i: RawRing(2.0, s.rate_hz, SPECTRUM_LO, SPECTRUM_HI, SPECTRUM_BIN_HZ)
                      for i, s in self.specs.items()}
        self.sum_ring = RawRing(2.0, self.rate, SPECTRUM_LO, SPECTRUM_HI, SPECTRUM_BIN_HZ)
        # One point every `TREND_SECONDS`, the same cadence the targets' own trends use, so the plot and
        # the ratio above it are reading the same series at the same rate.
        self.crossover = EnvelopePlot(columns=int(CROSSOVER_SECONDS / TREND_SECONDS), count=2)
        self._crossover_at = 0.0

        # Which reward the tone marks when both are above threshold. False is a theta bias, which is
        # the sensible default — theta is the band being brought up.
        self.unbiased = False
        self.metrics: dict[tuple, object] = {}
        self.theta: Target | None = None
        self.alpha: Target | None = None
        self.inhibit: Target | None = None
        # The sleep guard, `None` until it is switched on. Held separately from `inhibit` rather than
        # found by direction, because the two are not interchangeable: one says too awake and the other
        # says too far gone, and the screen has to be able to name which is which.
        self.guard: Target | None = None
        # θ−α, built on demand — see `set_objective`. Held here so `value_for` and `key_for` can recognise
        # it before it exists.
        self.difference: Target | None = None
        # What the sleep guard does when it fires — see `GUARD_RESPONSES`. `silence` is what the guard
        # did with no code at all, so it stays the default.
        self.guard_response = "silence"
        self._last_knock: float | None = None
        # Which objective each output channel reads. Two descriptions of one state, and they can differ per
        # channel because the questions differ: see `OBJECTIVES` and `channel_targets`.
        self.objective = {LEVEL: BANDS_OBJECTIVE, CONDITION: BANDS_OBJECTIVE}
        self.theta_width = "wide"
        self.high_low = 22.0
        theta_low, theta_high = THETA_BANDS[self.theta_width]
        self.theta = self.add_band(theta_low, theta_high, "increase", 0.25)
        self.alpha = self.add_band(ALPHA[0], ALPHA[1], "increase", ALPHA[2])
        low, high, aim = HIGH_BANDS[self.high_low]
        self.inhibit = self.add_band(low, high, "increase", aim, role="inhibit")

        self._apply_feedback()

    # --- targets -------------------------------------------------------------------------

    def add_band(self, low: float, high: float, direction: str, aim: float,
                 role: str = "reward") -> Target | None:
        band = Band(low, high, "")
        # `PAIR` with every site in it, whatever the count. That makes the target's own label say what it
        # is measured on — "ch0" with one channel, "ch0+ch1" with two — and it is what makes
        # `target_is_live` require *all* of them, which is right: a channel that is not delivering is still
        # being added into the sum.
        # The two inhibits here are `increase` — rises in high-beta are what is being suppressed — and
        # they read `role` rather than the direction, which used to carry both meanings at once.
        tgt = Target(PAIR, band, direction, role=role, mode="fixed", hit_rate_override=aim,
                     pair=self.sites, signal_floor=SIGNAL_FLOOR_UV,
                     floor=INHIBIT_FLOOR_UV if role == "inhibit" else None)
        if not self._ensure_metric(tgt):
            return None
        self.targets.append(tgt)
        return tgt

    # --- variants ------------------------------------------------------------------------

    @property
    def variant(self) -> tuple[str, bool, float]:
        """Which combination is in force, **derived from the bands** rather than stored.

        Stored, it would go stale the moment an edge was nudged — and nudging an edge is the commonest
        thing done on this screen. Derived, a hand edit simply lands on no variant, which is a state the
        panel can name.
        """
        def edges(tgt: Target | None) -> tuple[float, float] | None:
            return None if tgt is None else (tgt.band.low, tgt.band.high)

        width = next((k for k, b in THETA_BANDS.items() if edges(self.theta) == b), "")
        high = next((k for k, b in HIGH_BANDS.items() if edges(self.inhibit) == b[:2]), 0.0)
        return width, self.guard is not None, high

    # --- what cannot be combined -------------------------------------------------------
    #
    # **The guard and the wide theta band are mutually exclusive, and the rule lives here as well as on
    # the controls.** A guard at 2-5 under a reward at 2-7 makes the same energy earn the tone and silence
    # it, so the protocol cannot be satisfied at all — which is not a combination worth offering with a
    # warning attached, it is one the screen should not let you reach.
    #
    # It is enforced in both directions, because a rule enforced on one control is a rule with a hole in
    # it: switching the guard on with theta wide and widening theta with the guard on reach the same
    # state. And it is enforced in the *model* rather than only by disabling a pill, since "not while
    # that" belongs where it takes effect.
    #
    # `overlap_warning` stays, and is not made redundant by this: band edges are editable on the cards, so
    # theta's low edge can still be dragged under a guard that was legitimately switched on.

    def guard_blocked(self) -> str:
        """Why the sleep guard cannot be switched on, or "" when it can."""
        if self.theta is None or self.theta.band.low >= GUARD_BAND[1] - 0.05:
            return ""
        return "needs the narrow theta band"

    def width_blocked(self, width: str) -> str:
        """Why a theta width cannot be chosen, or "" when it can."""
        if self.guard is None or width not in THETA_BANDS:
            return ""
        if THETA_BANDS[width][0] >= GUARD_BAND[1] - 0.05:
            return ""
        return "turn the sleep guard off first"

    def set_theta_width(self, width: str) -> None:
        if width not in THETA_BANDS or self.theta is None or self.width_blocked(width):
            return
        self.theta_width = width
        low, high = THETA_BANDS[width]
        self.retune(self.theta, low=low, high=high)

    def set_high_low(self, low_hz: float) -> None:
        """Move the high inhibit's lower edge, **and its aim with it.** A 15-36 band is over threshold more
        often than a 22-36 one, so keeping one aim across both would make the wider band a different
        stringency wearing the same number — and the two are set separately for that reason."""
        if low_hz not in HIGH_BANDS or self.inhibit is None:
            return
        self.high_low = low_hz
        low, high, aim = HIGH_BANDS[low_hz]
        moved = aim != self.inhibit.hit_rate_override
        self.inhibit.hit_rate_override = aim
        self.retune(self.inhibit, low=low, high=high)
        # A new aim with the old threshold under it is a card whose two numbers disagree, so the threshold
        # is re-derived from the history already collected. Nothing is thrown away — `redefine` reads the
        # last 30 s and falls back to calibrating if that much has not arrived yet.
        if moved:
            self.inhibit.redefine()

    def set_guard(self, on: bool) -> None:
        """Add or remove the sleep guard.

        Adding a target changes what is being trained, so it is deliberately the same kind of act as
        adding one on the amplitude screen rather than something hidden inside a preset — and removing it
        drops the target rather than disabling it, since an inhibit that is present and inert is exactly
        the swallowed-refusal shape.
        """
        if on and self.guard is None:
            if self.guard_blocked():
                return
            low, high, aim = GUARD_BAND
            self.guard = self.add_band(low, high, "increase", aim, role="inhibit")
        elif not on and self.guard is not None:
            if self.guard in self.targets:
                self.targets.remove(self.guard)
            self.guard = None
            self.retune_all()

    def adopt_targets(self, previous: "TrainingSession") -> None:
        """Take the previous session's targets **and its names for them.**

        The base class replaces `targets` wholesale, which leaves `theta` / `alpha` / `inhibit` / `guard`
        pointing at the objects this session built in `__init__` and then discarded. Those orphans are
        never fed, so the crossover plot and the ratio would read empty trends for ever, the tone bias
        would compare the wrong pair, and every card would be labelled by position. Bound from the
        previous session's own pointers rather than re-derived from directions, because with two inhibits
        there is no way to tell the guard from the high inhibit by direction alone.
        """
        super().adopt_targets(previous)
        self.theta = getattr(previous, "theta", None)
        self.alpha = getattr(previous, "alpha", None)
        self.inhibit = getattr(previous, "inhibit", None)
        self.guard = getattr(previous, "guard", None)
        self.guard_response = getattr(previous, "guard_response", self.guard_response)
        self.difference = getattr(previous, "difference", None)
        self.objective = dict(getattr(previous, "objective", self.objective))
        self.theta_width = getattr(previous, "theta_width", self.theta_width)
        self.high_low = getattr(previous, "high_low", self.high_low)

    # --- objectives ----------------------------------------------------------------------

    def set_objective(self, channel: str, kind: str) -> None:
        """Point one output channel at one of the two objectives.

        The difference target exists exactly while some channel reads it, rather than being built once and
        left inert when nothing does. A target that is measured, calibrated and drawn while contributing to
        no output is the state `channel_targets` warns about — visible on screen as a card with a threshold
        and a hit rate, and connected to nothing.
        """
        if channel not in self.objective or kind not in OBJECTIVES:
            return
        self.objective[channel] = kind
        wanted = DIFFERENCE_OBJECTIVE in self.objective.values()
        if wanted and self.difference is None:
            # **`band=None`, and that one word is the whole rendering.** No edges to nudge, no colour from a
            # centre frequency, no filter of its own — so `rebuild_targets` gives it no card and it appears
            # where it was chosen instead. It carried a nominal 2-11 Hz band for one commit and that put two
            # inert number boxes and a filter dialog beside a threshold that worked.
            self.difference = Target(PAIR, None, "increase", mode="fixed", name="θ−α", unit="µV",
                                     hit_rate_override=DIFFERENCE_AIM, pair=self.sites)
            self.targets.append(self.difference)
        elif not wanted and self.difference is not None:
            if self.difference in self.targets:
                self.targets.remove(self.difference)
            self.difference = None
            self.retune_all()

    def _due(self, tgt: Target, fresh: set) -> bool:
        """The difference advances on theta's reading, and that rate is the whole point.

        Nothing puts `("difference",)` in `fresh`, because nothing measures it — so left to the base class
        it would never be observed at all, and every window it has (its calibration, its trend, "last 30 s
        on target") would stay empty while its card showed a threshold. Theta and alpha are measured on the
        same sum at the same rate, so theta's cadence *is* the difference's cadence; keying off either one
        would be right and keying off both would double it, which is the rate bug this method exists for.
        """
        if tgt is self.difference:
            return self.theta is not None and self.key_for(self.theta) in fresh
        return super()._due(tgt, fresh)

    def channel_targets(self, channel: str) -> list[Target]:
        """The inhibits, plus whichever objective this channel reads.

        The inhibits are in **both** sets whichever objective is chosen: "too awake" and "too far gone" are
        conditions on the state itself, not on one way of describing it. Only the reward side forks.
        """
        live = self.active_targets
        inhibits = [x for x in live if x.role == "inhibit"]
        if self.objective.get(channel) == DIFFERENCE_OBJECTIVE:
            rewards = [x for x in live if x is self.difference]
        else:
            rewards = [x for x in live if x.role == "reward" and x is not self.difference]
        return rewards + inhibits

    @property
    def mixed(self) -> bool:
        """Whether the two channels are reading different objectives — which the screen has to say, since
        one signal panel describing "the objective" would then be describing only half of it."""
        return self.objective[LEVEL] != self.objective[CONDITION]

    # --- what the guard does -------------------------------------------------------------

    @property
    def guard_over(self) -> bool:
        """Whether the sleep guard is currently the wrong side of its threshold, with data behind it.

        `total` as well as `wanted`, because a target that has never been fed reads as failing — every value
        is zero and a `decrease` target wants zero — so without it the guard would hush the music and knock
        from the moment the screen opened.
        """
        g = self.guard
        return (g is not None and bool(g.total) and not g.wanted and self.target_is_live(g))

    def _maybe_alert(self) -> None:
        """The shared alert, plus the guard's own knock.

        Deliberately **not** routed through `FeedbackSettings.alert`, which fires for whichever target is
        worst over its threshold. That is right for twelve bins where the question is *which* one, and wrong
        here: this sound means one specific thing — you are falling asleep — and a mechanism that sometimes
        means that and sometimes means high beta would carry no information at all with the eyes closed.
        """
        super()._maybe_alert()
        if self.guard_response == "silence" or not self.guard_over:
            self._last_knock = None
            return
        now = time.monotonic()
        if self._last_knock is not None and now - self._last_knock < GUARD_KNOCK_SECONDS:
            return
        self._last_knock = now
        self.cue(cues.frequency_click(self.guard.band.centre), self.feedback.alert.volume)

    def set_guard_response(self, name: str) -> None:
        if name in GUARD_RESPONSES:
            self.guard_response = name
            self._last_knock = None     # so switching it on knocks at the next crossing, not one gap later

    def music_hold(self, met: bool) -> bool:
        """Hush the music while the guard is over, on top of whatever the music settings already say.

        The `knock + hush` half of the design — all other sounds fade out — and it is what
        makes the knock arrive into silence rather than over music. `or`, not a replacement: a protocol that
        also gates the music on the condition keeps doing so.
        """
        return super().music_hold(met) or (self.guard_response == "knock + hush" and self.guard_over)

    def retune_all(self) -> None:
        """Drop metrics nothing reads any more. `retune` does this for one target; removing one needs it
        without a target to pass."""
        wanted = {self.key_for(x) for x in self.targets}
        for key in [k for k in self.metrics if k not in wanted]:
            del self.metrics[key]
            self.amplitudes.pop(key, None)

    def overlap_warning(self) -> str:
        """Whether any reward and any inhibit share frequencies, and by how much.

        The combination that needs saying is a sleep guard under the *wide* theta band: 2-5 inside 2-7
        means the same energy both earns and is punished, so the protocol cannot be satisfied. It is
        allowed rather than refused — the bands are editable and the operator may be doing something
        deliberate — but it cannot be allowed *silently*, which is the whole reason this exists.
        """
        rewards = [x for x in (self.theta, self.alpha) if x is not None]
        inhibits = [x for x in (self.inhibit, self.guard) if x is not None]
        for rew in rewards:
            for inh in inhibits:
                lo = max(rew.band.low, inh.band.low)
                hi = min(rew.band.high, inh.band.high)
                if hi - lo > 0.05:
                    return (f"{lo:g}–{hi:g} Hz is rewarded and inhibited at once — "
                            f"energy there both earns the tone and silences it")
        return ""

    def _ensure_metric(self, tgt: Target) -> bool:
        if tgt is self.difference:
            return True             # nothing to build; its value is two other targets subtracted
        key = self.key_for(tgt)
        if key in self.metrics:
            return True
        spec = self.specs.get(self.site)
        if spec is None:
            return False
        try:
            self.metrics[key] = self.build_metric(spec, tgt.band, tgt)
        except ValueError:
            return False        # the band reaches Nyquist at this rate
        return True

    def retune(self, tgt: Target, low: float | None = None, high: float | None = None) -> None:
        tgt.set_band(low, high)
        self._ensure_metric(tgt)
        wanted = {self.key_for(x) for x in self.targets}
        for key in [k for k in self.metrics if k not in wanted]:
            del self.metrics[key]
            self.amplitudes.pop(key, None)

    def retune_filter(self, tgt: Target, kind: str | None = None, order: int | None = None,
                      cfir_delay: float | None = None, ripple: float | None = None) -> None:
        # The convention rescale, exactly as `AmplitudeSession.retune_filter` argues it: switching
        # between cFIR (peak) and the IIR kinds (rms / meanabs) changes what the numbers mean by a fixed
        # sine ratio, so a settled fixed threshold is carried across and the statistics restart.
        conventions = {"rms": 1 / np.sqrt(2.0), "meanabs": 2.0 / np.pi}
        before = self.filter_for(tgt)
        was = conventions.get("" if before[0] == "cfir" else before[-1], 1.0)
        if kind is not None:
            tgt.filter_kind = kind
        if order is not None:
            tgt.filter_order = max(1, min(10, int(order)))
        if cfir_delay is not None:
            tgt.cfir_delay = round(max(0.02, min(1.0, cfir_delay)), 3)
        if ripple is not None:
            tgt.filter_ripple = round(max(0.05, min(6.0, ripple)), 2)
        after = self.filter_for(tgt)
        now = conventions.get("" if after[0] == "cfir" else after[-1], 1.0)
        if was != now:
            tgt.threshold *= now / was
            tgt.reset_stats()
        self.retune(tgt)

    def key_for(self, tgt: Target):
        if tgt is self.difference:
            # Its own key, so it never shares one with a real band and `_ensure_metric` never builds an
            # estimator for the 2-11 Hz span its nominal band covers.
            return ("difference",)
        # "sum" rather than a channel, because there is one signal here however many electrodes feed it
        return ("sum", self.band_key(tgt.band), self.filter_for(tgt))

    @property
    def summed(self) -> bool:
        return len(self.sites) > 1

    @property
    def source_label(self) -> str:
        """What the bands are measured on, for the screen to say. Never inferred from the channel count in
        the toolbar: that says how many are in use, not which ones are added together."""
        if not self.sites:
            return "no channel"
        joined = "+".join(f"ch{c}" for c in self.sites)
        return f"{joined} summed" if self.summed else joined

    def value_for(self, tgt: Target) -> float:
        if tgt is self.difference:
            # Derived rather than measured: no estimator of its own, no band of its own. The two it is
            # built from are measured on the same sum through filters of the same design, so subtracting
            # their envelopes is subtracting two numbers in the same unit at the same latency.
            if self.theta is None or self.alpha is None:
                return 0.0
            return self.value_for(self.theta) - self.value_for(self.alpha)
        return self.amplitudes.get(self.key_for(tgt), 0.0)

    def latency_for(self, tgt: Target) -> float | None:
        if tgt is self.difference:
            # The slower of the two it is made of: a difference is no fresher than its stalest half.
            parts = [self.latency_for(x) for x in self.rewards]
            known = [x for x in parts if x is not None]
            return max(known) if known else None
        metric = self.metrics.get(self.key_for(tgt))
        return metric.latency_seconds if metric is not None else None

    @property
    def rewards(self) -> list[Target]:
        """Theta and alpha, always — they are measured and drawn whichever objective is in force, because
        the crossover plot and the tone's pitch are read off them either way."""
        return [x for x in (self.theta, self.alpha) if x is not None]

    # --- the crossover -------------------------------------------------------------------

    @property
    def ratio(self) -> float | None:
        """Theta over alpha, averaged over `RATIO_WINDOW`. None until there is a window of it.

        Averaged because the instantaneous ratio of two narrowband envelopes flickers through an order of
        magnitude second to second, and the quantity of interest changes over tens of minutes.

        From each target's own trend rather than from the plot beside it, because the plot's point rate
        follows its *window* — so reading 60 points off a ten-minute plot would be averaging two minutes
        while the label said thirty seconds.
        """
        if self.theta is None or self.alpha is None:
            return None
        points = int(RATIO_WINDOW / TREND_SECONDS)
        theta, alpha = list(self.theta.trend), list(self.alpha.trend)
        if min(len(theta), len(alpha)) < points:
            return None
        top, bottom = float(np.mean(theta[-points:])), float(np.mean(alpha[-points:]))
        if bottom <= 1e-9:
            return None
        return top / bottom

    @property
    def crossed(self) -> bool:
        ratio = self.ratio
        return ratio is not None and ratio >= 1.0

    @property
    def state_balance(self) -> float | None:
        """Where the music sits between its theta end and its alpha end, **averaged over `BALANCE_SECONDS`**.

        Two decisions here, and the second was a real defect.

        **A share rather than a ratio**, because a share is bounded and a ratio is not: θ/α runs from 0 to
        arbitrarily large, so it cannot be mapped onto a crossfade without choosing a ceiling that the fade
        would then sit against. `θ/(θ+α)` is 0.5 at the crossover, which is the middle of the fade, which is
        where it belongs. Driving the crossfade from θ−α instead is the *same*
        quantity once it is normalised by the only scale available — see the note above `BALANCE_SECONDS`.

        **And it is averaged, where it used to read the instantaneous values.** Two narrowband envelopes
        flicker through an order of magnitude, which is exactly why `RATIO_WINDOW` exists for the readout
        above the plot — and the music was not using it, so the sound swapped ends several times a second on a
        state that moves over half an hour, under a caption that was averaged. The window is **30 s**, which
        matches `RATIO_WINDOW`.
        """
        if self.theta is None or self.alpha is None:
            return None
        theta = _mean_recent(self.theta, BALANCE_SECONDS)
        alpha = _mean_recent(self.alpha, BALANCE_SECONDS)
        if theta is None or alpha is None:
            return None                 # nothing measured yet: leave the music where it is
        total = theta + alpha
        if total <= 1e-9:
            return None
        # 1 is the *theta* end, so the second colour is theta's; see `_apply_feedback`
        return float(min(1.0, max(0.0, theta / total)))

    def earning_target(self) -> Target | None:
        """Which reward the tone should sound like — theta when both are met.

        The default is a theta bias, with the unbiased switch choosing "the highest reward
        percentage". The bias is the default because theta is the band being brought up, so when both are
        available it is the one worth marking; `unbiased` here is the same switch.
        """
        met = [x for x in self.rewards if x.wanted and self.target_is_live(x)]
        if not met:
            return None
        if self.unbiased:
            return max(met, key=lambda x: x.score(self.feedback.normalisation))
        return met[0]

    # --- feedback ------------------------------------------------------------------------

    def _apply_feedback(self) -> None:
        fb = self.feedback
        # `either`, because the two rewards are alternatives — see the module docstring. This is also what
        # makes the reward tone fire on one of them rather than on both, since `on_target` follows the rule
        # in this one case.
        fb.combine = "either"
        fb.music.enabled = True
        fb.music.source = "crossfade"
        fb.music.colour = MUSIC_ENDS["alpha"]              # balance 0
        fb.music.second_colour = MUSIC_ENDS["theta"]       # balance 1
        fb.click.enabled = True
        fb.click.keyed = True
        fb.click.interval_seconds = REWARD_INTERVAL
        fb.click.hold_seconds = HOLD_SECONDS
        # **The tone's own gain is no longer written here.** It was 25, which is what `CLICK_VOLUME` is, so
        # the line said nothing except that this screen had decided separately — and a number restated is a
        # number that drifts. What this protocol actually needs is the tone *above the music*, and that is
        # the line below: the music comes down to 20 rather than the tone going up.
        fb.music.volume = MUSIC_VOLUME
        # Both off, for two different reasons. The shadow is an output into a closed channel. The master
        # volume would put a score onto the music's loudness, on top of the state its character already
        # carries — and nothing in this design gets louder as the trainee does better.
        fb.shadow.enabled = False
        fb.sound.enabled = False
        fb.alert.enabled = False

    # --- feed ----------------------------------------------------------------------------

    def observe_raw(self, block, judged) -> None:
        ch = block.channel
        values = np.asarray(block.values)
        if ch in self.traces:
            self.traces[ch].push(self.for_display(ch, values), block.valid, judged.accepted)
        if ch in self.rings:
            self.rings[ch].push(values)
        if self.feedback.music.source == "eeg" and ch == self.site:
            self.music.feed(values)

    def consume(self, block) -> set:
        """Sum the aligned channels, then measure every band once on that sum.

        The summed block is built from the **raw counts** rather than from the microvolts, so the sum is
        exact: every channel on this device shares one scale, so adding counts and scaling once is the same
        number as scaling and adding, without a round trip through floats. It carries the first channel's
        spec and channel number because that is what the estimators are keyed to — a metric checks the
        block's channel against its own spec, and there is no channel number that means "the sum".
        """
        fresh: set = set()
        if self.aligner is None:
            return fresh
        paired = self.aligner.push(block)
        if paired is None:
            return fresh                # not every channel covers this run yet
        start, runs = paired
        spec = self.specs[self.site]
        counts = np.zeros(len(runs[0]))
        for values in runs:
            counts += values / spec.scale
        # Usable only if every contributing channel is: one bad electrode is being added into the sum, so
        # the sum is as bad as its worst input rather than as good as the block that happened to arrive.
        worst = OK if all(self.state.get(c) is not False for c in self.sites) else NO_SIGNAL
        summed = Block(self.site, start, tuple(int(round(v)) for v in counts), spec, state=worst)
        self.sum_ring.push(np.asarray(summed.values))
        for key, metric in list(self.metrics.items()):
            for r in metric.feed(summed):
                self.amplitudes[key] = r.value
                fresh.add(key)
        return fresh

    sum_spectrum_leads = True
    """The sum leads, because it is what every band on this screen is measured on — the crossover is read
    off amplitudes that come from it. The channels follow, because a flat electrode is invisible inside a
    sum."""

    def spectrum_sum_ring(self):
        """Only with more than one electrode in use: with one, the "sum" is that channel and offering both
        would be one signal under two names."""
        return self.sum_ring if len(self.specs) > 1 else None

    def push_displays(self, now: float) -> None:
        """The crossover plot, advanced from the draw loop so its time axis is a time axis.

        Driven from the feed instead, its scroll rate would be a function of the sample rate and the
        channel count, and the axis would be lying about how long ago the left edge was.
        """
        if now - self._crossover_at >= TREND_SECONDS:
            self._crossover_at = now
            self.crossover.push([self.value_for(x) for x in self.rewards] or [0.0, 0.0])

    # --- what this protocol requires -----------------------------------------------------

    def wiring_problem(self) -> str | None:
        """Why the numbers would be wrong, if they would be.

        One channel at Pz referenced to an ear is the shipped montage. We have no montage model to check
        that against, so what can be checked is that there is a channel delivering signal at all — which is
        the failure that matters here, because with the eyes closed nobody is watching the trace.
        """
        if not self.specs:
            return "no channel in use"
        dead = [c for c in self.sites if self.state.get(c) is False]
        if dead:
            # Named individually, because with a summed signal a bad electrode does not show up as one bad
            # trace on the instrument being read — it is inside the number, and nobody is watching the
            # traces with their eyes closed.
            names = ", ".join(f"ch{c}" for c in dead)
            verb = "is" if len(dead) == 1 else "are"
            where = " and in the sum" if self.summed else ""
            return f"{names} {verb} not delivering signal{where} — check the electrodes before starting"
        return None


def session_for(*args, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None,
                **kwargs) -> ATSession:
    _ = heart_choice, heart_runner, mendi_choice, mendi_runner, pupil_choice, pupil_runner
    return ATSession(*args, **kwargs)


# --- the screen --------------------------------------------------------------------------

def view(page: ft.Page, session: ATSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    """Build once, mutate on Flet's loop."""

    signal = signal_pane.SignalPane(page, session)
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width(reserve: int = TARGETS_W) -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - reserve - 70)

    # The spectrum, shared with every other screen that has one. Needed here because the crossover is a
    # comparison of two band amplitudes and the shape underneath them is how you tell an alpha peak from
    # broadband drift — and because a band edge is worth moving once you can see where the energy is.
    spectrum = spectrum_pane.SpectrumPane(page, session, plot_width)
    trends = trends_pane.TrendsPane(page, session, plot_width)

    # ---- the crossover instrument ----
    ratio_text = ft.Text("—", size=30, color=t.TEXT, weight=ft.FontWeight.W_300)
    ratio_note = ft.Text("", size=t.SMALL, color=t.FAINT)
    cross_canvas = cv.Canvas(shapes=[], expand=True, height=CROSSOVER_H)
    cross_axis, cross_labels = value_axis(CROSSOVER_H, count=4, width=AXIS_W)
    cross_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    cross_legend = ft.Row(spacing=14)
    cross_empty = ft.Text("waiting for data", size=t.SMALL, color=t.FAINT)

    # ---- raw trace ----
    traces = trace_rows.TraceRows(page, session, plot_width, height=TRACE_H)

    # ---- the music, and how it is sitting ----
    music_fill = ft.Container(width=0, height=8, border_radius=4, bgcolor=t.ACCENT, left=0, bottom=0)
    music_bar = ft.Stack([ft.Container(width=200, height=8, border_radius=4, bgcolor=t.SURFACE_HI,
                                       left=0, bottom=0), music_fill], width=200, height=10)
    music_note = ft.Text("", size=t.SMALL, color=t.FAINT)
    # **No sound controls here.** Both music tilts and both reward tones were auditionable from this panel,
    # and which band the tone marks was set from it — all of which is *sound configuration* and belongs on
    # the Feedback tab with the mechanism that plays it. Here it was redundant twice over: the Music pane
    # previews each tilt and the Tone pane auditions each reward at its own pitch. What stays is the
    # readout: where the music is sitting, which is a thing you watch rather than a thing you set.

    # ---- the targets ----
    target_column = ft.Column(spacing=10)
    cards: dict[int, TargetCard] = {}

    def filter_rows(tgt, redraw) -> list:
        """The shared per-target filter gear, so every screen opens the same dialog."""
        return target_card.filter_rows(session, tgt, lambda: (rebuild_targets(), redraw()))

    def slide(tgt, delta: float) -> None:
        """Move a band by `delta` and keep its width — what the arrows on either edge do."""
        tgt.shift_band(delta)
        session.retune(tgt)
        rebuild_targets()

    def rebuild_targets() -> None:
        cards.clear()
        rows = []
        # **A card is for a band.** Everything on one is a property of having edges — two nudgeable
        # numbers, a swatch from the centre frequency, a filter gear, an unsigned µV bar — so a target with
        # no band is not a card with rows missing, it is not a card. `banded` is the one question asked.
        for tgt in [x for x in session.targets if x.banded]:
            role = ("theta" if tgt is session.theta else
                    "alpha" if tgt is session.alpha else
                    "sleep" if tgt is session.guard else "high beta")
            identity = ft.Row(
                [ft.Container(width=3, height=18, bgcolor=tgt.colour, border_radius=2),
                 ft.Text(role, size=t.BODY, color=t.TEXT, width=64),
                 # Arrows slide the band and keep its width; typing sets one edge. `Target.shift_band`.
                 num_box(tgt.band.low,
                         (lambda g=tgt: lambda v: (session.retune(g, low=v),
                                                   rebuild_targets()))(),
                         step=0.1, low=0.1, high=tgt.band.high - 0.1,
                         on_step=(lambda g=tgt: lambda d: slide(g, d))()),
                 ft.Text("–", size=t.BODY, color=t.FAINT),
                 num_box(tgt.band.high,
                         (lambda g=tgt: lambda v: (session.retune(g, high=v),
                                                   rebuild_targets()))(),
                         suffix="Hz", step=0.1, low=tgt.band.low + 0.1,
                         on_step=(lambda g=tgt: lambda d: slide(g, d))()),
                 ft.Container(expand=True),
                 # Role and direction, as on the amplitude screen — read-only here, because this
                 # protocol's arrangement is fixed and the whole screen says so. Two words rather than
                 # one because they are two facts: what it drives, and which way it is trained.
                 ft.Container(content=ft.Text(
                     f"{tgt.role_label}  {'↑' if tgt.direction == 'increase' else '↓'}",
                     size=t.SMALL,
                     color=t.ACCENT if tgt.role == "reward" else t.WARN,
                     weight=ft.FontWeight.W_600),
                     padding=ft.Padding(8, 4, 8, 4), border_radius=6,
                     bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE))],
                spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER)
            card = TargetCard(tgt, identity, rebuild_targets, page=page,
                              modes=session.threshold_modes,
                              settings=(lambda g=tgt: lambda redraw: filter_rows(g, redraw))())
            cards[id(tgt)] = card
            rows.append(card.control)
        target_column.controls = rows
        signal.rebuild_contributions(session.targets)
        rebuild_variant()
        page.update()

    # ---- warnings ----
    warning = ft.Container(visible=False, bgcolor=t.SURFACE_HI, border_radius=8,
                           padding=ft.Padding(12, 9, 12, 9),
                           content=ft.Text("", size=t.SMALL, color=t.BAD))

    # ---- the band arrangement ----
    # Three axes rather than a list of named plans, because the two variants differ along exactly
    # these lines and the combinations between them are legitimate. The pills are **derived** from the
    # bands (`session.variant`), so nudging an edge on a card cannot leave this row asserting something
    # untrue — it lands on no variant, and the line underneath says so.
    variant_rows = ft.Column(spacing=8)
    variant_note = ft.Text("", size=t.SMALL, color=t.MUTED, max_lines=2)
    overlap_note = ft.Text("", size=t.SMALL, color=t.BAD, max_lines=2, visible=False)
    objective_rows = ft.Column(spacing=8)
    objective_note = ft.Text("", size=t.SMALL, color=t.MUTED, max_lines=3)

    # ---- the θ−α strip ----
    #
    # **Not a card, and this is the shape a derived quantity gets instead.** It lives in the panel where the
    # objective was chosen, because that is where you would look for its threshold, and it is built for a
    # *signed* number from scratch rather than inheriting the card's unsigned µV bar: zero is the crossover
    # and the sign is the entire content, so a bar starting at zero cannot show the half of the range that
    # matters. Present only while a channel reads it — content absent rather than empty.
    diff_value = ft.Text("—", size=19, color=t.TEXT, weight=ft.FontWeight.W_400)
    diff_thresh = ft.Text("", size=t.BODY, color=t.TEXT)
    diff_pct = ft.Text("", size=t.SMALL, color=t.MUTED)
    diff_note = ft.Text("", size=t.SMALL, color=t.WARN, visible=False)
    diff_scale = ft.Text("", size=t.SMALL, color=t.FAINT)
    diff_fill = ft.Container(width=0, height=8, border_radius=4, bgcolor=t.ACCENT, left=0, bottom=0)
    diff_zero = ft.Container(width=1, height=14, bgcolor=t.FAINT, left=0, bottom=-3)
    diff_mark = ft.Container(width=2, height=14, bgcolor=t.TEXT, left=0, bottom=-3)
    diff_mode = ft.Row(spacing=6)
    diff_ceiling = plots.StickyCeiling(floor=2.0)
    diff_bar = ft.Stack([ft.Container(width=DIFF_BAR_W, height=8, border_radius=4,
                                      bgcolor=t.SURFACE_HI, left=0, bottom=0),
                         diff_fill, diff_zero, diff_mark], width=DIFF_BAR_W, height=14)

    def nudge_difference(factor: float) -> None:
        """A hand nudge, which on a signed quantity cannot be a multiplication.

        The card multiplies by 0.9 and 1.1, which is right for an amplitude and meaningless here: it would
        never cross zero, and it would step in the wrong direction below it. So the step is a fraction of the
        scale actually on screen, which is also what makes it the same size wherever the value sits.
        """
        tgt = session.difference
        if tgt is None:
            return
        step = max(0.1, round(diff_ceiling.value * 0.1, 2))
        tgt.threshold = round(tgt.threshold + (step if factor > 1 else -step), 2)
        tgt.mode = "fixed"          # a hand-set number is never silently overwritten by a tracker
        tgt._settled = True
        tgt.rescore()               # the comparison moved, the measurement did not
        refresh_difference()

    def redefine_difference() -> None:
        if session.difference is not None:
            session.difference.redefine()
            refresh_difference()

    def set_difference_mode(mode: str) -> None:
        tgt = session.difference
        if tgt is None or mode not in session.threshold_modes:
            return
        tgt.mode = mode
        # Switching *to* fixed has to derive a number, or the strip would show the tracker's last value
        # under a label saying it is being held. Switching to automatic needs nothing: the tracker
        # recomputes on the next observation.
        if mode == "fixed":
            tgt.redefine()
        refresh_difference()

    def refresh_difference() -> None:
        tgt = session.difference
        diff_strip.visible = tgt is not None
        if tgt is None:
            return
        ceiling = diff_ceiling.update(max(abs(tgt.value), abs(tgt.threshold)), session.elapsed)
        mid = DIFF_BAR_W / 2.0
        span = mid / ceiling
        diff_zero.left = mid - 0.5
        # From zero toward the value, so the bar reads as a displacement rather than as a quantity.
        offset = max(-mid, min(mid, tgt.value * span))
        diff_fill.left = mid + min(0.0, offset)
        diff_fill.width = abs(offset)
        diff_fill.bgcolor = t.GOOD if tgt.wanted else t.ACCENT
        diff_mark.left = mid + max(-mid, min(mid, tgt.threshold * span)) - 1
        diff_value.value = f"{tgt.value:+.1f}"
        diff_thresh.value = f"{tgt.threshold:+g}"
        pct = tgt.recent_percent()
        diff_pct.value = ("waiting" if pct is None
                          else f"{pct:.0f}% of the last 30 s · aim {tgt.hit_rate:.0%}")
        diff_scale.value = (f"±{ceiling:g} µV · 0 = crossover · mark = threshold "
                            f"{tgt.threshold:+g}")
        diff_note.value = ("" if not tgt.calibrating else
                           f"calibrating — {tgt.calibration_remaining:.0f} s left")
        diff_note.visible = bool(diff_note.value)
        # The same modes the cards offer, built the same way — one entry stated, several offered — because
        # this strip is a target's controls in a different shape, not a different kind of target. Nudging
        # already forces `fixed`, so a hand-set number is never silently overwritten by the tracker.
        modes = session.threshold_modes
        diff_mode.controls = (
            [ft.Text(modes[0], size=t.SMALL, color=t.MUTED)] if len(modes) == 1 else
            [pill(m, tgt.mode == m, (lambda m_=m: lambda _e: set_difference_mode(m_))())
             for m in modes])
        if tgt.can_redefine:
            diff_mode.controls += [ft.Container(width=6),
                                   pill("redefine", False, lambda _e: redefine_difference())]

    diff_strip = ft.Column([
        ft.Container(height=10),
        ft.Row([ft.Text("θ−α", size=t.BODY, color=t.TEXT, width=44), diff_value,
                ft.Container(expand=True),
                ft.Text("threshold", size=t.SMALL, color=t.FAINT), diff_thresh,
                tiny("−", lambda _e: nudge_difference(0.9)),
                tiny("+", lambda _e: nudge_difference(1.1))],
               spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Container(height=9),
        ft.Row([diff_bar, ft.Container(width=12), diff_pct], spacing=0,
               vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Container(height=4),
        # **The pills get their own row, because the one they shared could not hold them.** The scale
        # caption has no width limit and the spacer between collapses to nothing before anything wraps, so a
        # third pill pushed the row past the panel and clipped the last one off the edge — with only `fixed`
        # and `automatic` the row already reached the border, leaving no room for `redefine` to appear in.
        # The same failure the amplitude card's identity row had, and the same fix: shorten the row rather
        # than widen the card.
        diff_scale,
        ft.Container(height=6),
        ft.Row([diff_mode], alignment=ft.MainAxisAlignment.END),
        diff_note,
    ], spacing=0, visible=False)

    def rebuild_variant() -> None:
        width, guard_on, high = session.variant

        def row(label: str, choices, reason: str = "") -> ft.Control:
            """`choices` is `(text, selected, action, why_not)`. An inert pill keeps its place and its
            state — locked is not hidden — and the reason goes at the end of its own row rather than in the
            summary line, because an inert control with the reason somewhere else reads as a broken one."""
            cells = [pill(text, sel, act, enabled=not why) for text, sel, act, why in choices]
            if reason:
                cells += [ft.Container(width=2),
                          ft.Text(reason, size=t.SMALL, color=t.FAINT, max_lines=1)]
            return ft.Row([ft.Text(label, size=t.SMALL, color=t.MUTED, width=78), *cells],
                          spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        def choose(fn, *args):
            # Only `rebuild_targets`, which ends by calling this back — a band edit on a card has to
            # refresh these pills too, so the callback runs one way round and not both.
            def go(_):
                fn(*args)
                rebuild_targets()
            return go

        # The two mutually exclusive choices each disable the other's pill and say why. One direction alone
        # would be a rule with a hole in it: switching the guard on under wide theta and widening theta
        # under the guard reach the same unsatisfiable state.
        no_wide = session.width_blocked("wide")
        no_guard = session.guard_blocked()
        variant_rows.controls = [
            row("theta", [("2–7 wide", width == "wide",
                           choose(session.set_theta_width, "wide"), no_wide),
                          ("5–8 narrow", width == "narrow",
                           choose(session.set_theta_width, "narrow"), "")], no_wide),
            row("sleep", [("off", not guard_on, choose(session.set_guard, False), ""),
                          ("2–5 down", guard_on, choose(session.set_guard, True), no_guard)], no_guard),
            # **Only while there is a guard to respond to.** A row of choices about what a switched-off
            # inhibit would do is a control for a thing that is not happening, and hiding it takes its
            # spacing with it because the whole row is one control.
            *([row("when it fires",
                   [(name, session.guard_response == name,
                     choose(session.set_guard_response, name), "")
                    for name in GUARD_RESPONSES])] if guard_on else []),
            row("high beta", [("22–36", high == 22.0, choose(session.set_high_low, 22.0), ""),
                              ("15–36", high == 15.0, choose(session.set_high_low, 15.0), "")]),
        ]
        # Only the hand-edited case gets a word. Everything else is already on the pills.
        variant_note.value = "" if width and high else "these bands do not match a preset"
        variant_note.visible = bool(variant_note.value)
        overlap_note.value = session.overlap_warning()
        overlap_note.visible = bool(overlap_note.value)

        # Which objective each output channel reads. Two rows rather than one, because the whole point is
        # that they may differ — a fade wants a quantity that says how far toward the state you are, and a
        # tone wants "you are in one of them now".
        objective_rows.controls = [
            row("fade", [(text, session.objective[LEVEL] == kind,
                          choose(session.set_objective, LEVEL, kind), "")
                         for kind, text in OBJECTIVE_LABELS.items()]),
            row("tone", [(text, session.objective[CONDITION] == kind,
                          choose(session.set_objective, CONDITION, kind), "")
                         for kind, text in OBJECTIVE_LABELS.items()]),
        ]
        objective_note.value = OBJECTIVE_NOTES[(session.objective[LEVEL],
                                                session.objective[CONDITION])]
        refresh_difference()
        page.update()

    # ---- tabs ----

    train_left = ft.Column([
        panel("CROSSOVER",
              ft.Row([ratio_text, ft.Container(width=12),
                      ft.Column([ratio_note], spacing=0, expand=True)],
                     vertical_alignment=ft.CrossAxisAlignment.CENTER),
              ft.Container(height=10),
              ft.Container(content=cross_legend, padding=ft.Padding(0, 0, 0, 8)),
              ft.Row([cross_axis,
                      ft.Stack([ft.Container(content=cross_canvas, expand=True),
                                ft.Container(content=cross_empty, expand=True,
                                             alignment=ft.Alignment.CENTER)],
                               expand=True, height=CROSSOVER_H)],
                     spacing=0),
              ft.Container(height=6), cross_caption,
              help=help_button(page, "Crossover", CROSSOVER_HELP)),
        ft.Container(height=12),
        panel("RAW", *traces.rows, ft.Container(height=6), traces.caption),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    # Which electrodes the bands are measured on, in the panel header. The toolbar's channel count says
    # *how many* are in use and cannot say that they are added together — and with a summed signal that is
    # the difference between one instrument on a wide view and a number nobody can source.
    source_chip = ft.Container(
        content=ft.Text(session.source_label, size=t.SMALL, color=t.MUTED),
        padding=ft.Padding(8, 3, 8, 3), border_radius=5, bgcolor=t.SURFACE_HI)

    train_right = ft.Column([
        panel("BANDS", variant_rows, ft.Container(height=8), variant_note, overlap_note,
              help=help_button(page, "Bands", BANDS_HELP)),
        ft.Container(height=12),
        panel("OBJECTIVE", objective_rows, ft.Container(height=8), objective_note, diff_strip,
              help=help_button(page, "Objective", OBJECTIVE_HELP)),
        ft.Container(height=12),
        panel("TRAINING", target_column, trailing=source_chip),
        ft.Container(height=12),
        panel("SOUND",
              ft.Row([music_bar, ft.Container(width=12),
                      ft.Column([music_note], spacing=0, expand=True)],
                     vertical_alignment=ft.CrossAxisAlignment.CENTER),
              help=help_button(page, "Sound", FEEDBACK_HELP)),
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

        # Per frame, like every other live readout: it holds a value, a comparison and a calibration
        # countdown, none of which are static.
        refresh_difference()

        trouble = session.wiring_problem()
        warning.visible = trouble is not None
        if trouble:
            warning.content.value = trouble

        # the crossover: two envelopes, one scale, because which is bigger is the question
        measured = session.crossover.peak()
        # A floor under the ceiling, so an empty plot's caption reads as a scale rather than as
        # "0–1e-09 µV full scale", which looks like a fault in the axis rather than an absence of data.
        peak = plot_ceiling(max(measured, 1.0), *[x.threshold for x in session.rewards])
        colours = [x.colour for x in session.rewards] or [t.ACCENT, t.ACCENT_DIM]
        cross_canvas.shapes = session.crossover.shapes(
            plot_width(reserve=TARGETS_W + AXIS_W), CROSSOVER_H, peak, colours,
            thresholds=[x.threshold for x in session.rewards])
        place_value_labels(cross_labels, CROSSOVER_H, peak, unit="µV")
        cross_empty.visible = measured <= 0
        cross_caption.value = (f"{session.source_label} · {CROSSOVER_SECONDS / 60:g} min across · "
                               f"0–{peak:g} µV full scale · thin line = threshold")
        cross_legend.controls = [
            ft.Row([ft.Container(width=10, height=2, bgcolor=x.colour, border_radius=1),
                    ft.Text(f"{name} {x.band.low:g}–{x.band.high:g}", size=t.SMALL,
                            color=t.MUTED)], spacing=6)
            for name, x in zip(("theta", "alpha"), session.rewards)]

        ratio = session.ratio
        if ratio is None:
            ratio_text.value = "—"
            ratio_text.color = t.FAINT
            ratio_note.value = f"theta over alpha, once there is {RATIO_WINDOW:g} s of it"
        else:
            ratio_text.value = f"{ratio:.2f}"
            ratio_text.color = t.GOOD if session.crossed else t.TEXT
            ratio_note.value = (f"theta over alpha, last {RATIO_WINDOW:g} s · "
                                + ("theta is ahead" if session.crossed
                                   else "crosses over at 1.00"))

        # the trace, which nobody is watching — which is why the warning above exists
        traces.refresh(TRACE_SECONDS, extra=("500 ms/div",))

        # where the music is sitting. Named at both ends, since "62%" alone says nothing about which way.
        balance = session.state_balance
        if balance is None:
            music_fill.width = 0
            music_note.value = "waiting for data"
        else:
            music_fill.width = 200 * balance
            music_note.value = (f"{balance:.0%} toward theta · {MUSIC_ENDS['theta']} noise at that end, "
                                f"{MUSIC_ENDS['alpha']} at the other")

        for tgt in session.targets:
            card = cards.get(id(tgt))
            if card is not None:
                card.refresh(session.value_for(tgt), "µV", session.response_time(tgt))

        page.update()

    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)

    rebuild_targets()
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)
