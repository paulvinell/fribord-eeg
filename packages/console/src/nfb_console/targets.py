"""Training targets: one band, one source, one direction, one threshold.

The threshold is the part worth care, and the two modes are genuinely different arrangements:

- **fixed** — one number for the session. It still has to come from somewhere, so a session that starts
  with a fixed threshold spends its first 30 s *calibrating*: deriving a threshold from whatever has
  arrived so far, once a second, and settling on the full window at 30 s. That is not a mode you choose,
  it is what happens, so it is a label that counts down and then goes away. **Redefine** does the same
  thing on demand from the last 30 s, which needs no waiting because the history is already there.

  It used to derive *nothing* until the 30 s were up, which left the target thresholded at whatever it was
  constructed with — and a screen whose reward never fires for the first half minute is one where the
  operator cannot tell a settling session from a broken one. A provisional number that visibly moves under
  a countdown is better than a silent one that does not.
- **automatic** — tracks a rolling window continuously. Nothing to calibrate and nothing to redefine,
  because it is always doing exactly that.

Both windows are settable per target — see `window_seconds`, which is what lets one screen track a 40 Hz band
over 10 s and a 0.05 Hz one over 30.

Manual nudging switches to fixed and marks it settled: a number someone set by hand should never be
overwritten by a tracker or a recalibration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from nfb.dsp import Band

from . import theme
from .feedback import ABSOLUTE_STEP

PAIR = -1
"""Pseudo-source: one measurement spanning several channels at once — a synchrony over a pair, or a band
measured on the channels **added together**.

The summing is real, and it happens before anything filters: blocks arrive one channel at a time, so an
aligner hands back the runs covering the same samples and those are added as raw counts, then one estimator
measures the result (`amplitude_view.consume`, `at_view.consume`). One signal, one estimator, one number,
and phase relationships survive into it.

**The amplitude screen used to offer a `SUM` that was not one**, and the sentence it was described by is the
whole of the bug: "the mean amplitude across active channels, not the amplitude of the summed signal". Each
channel was filtered separately, each envelope taken separately, and the numbers averaged — which is blind to
phase by construction, so two channels carrying the same rhythm in antiphase read the same as two in phase.
The measure a summed signal exists to make was the one it could not make.

That average is also not worth keeping as a second option, which was the next thing tried and was wrong for a
simpler reason: a mean of N amplitudes *is* their sum over N, so against a threshold calibrated from the
trainee's own data — which every threshold here is — it is the same measure with a constant in front of it.
Two entries in a menu that differ by a scale factor is a choice with nothing on either side of it."""

MODES = ("fixed", "automatic")
"""Two, and a third was built and removed.

A `ratchet` mode that cleared its window on every hit, so the threshold would climb each time the trainee
reached it, was implemented here on a misreading of a reset condition. That condition's input is the
**success-rate setting**, not the signal, so it fires when an operator drags the aim to 100% and back: it is
a manual recalibrate, which `redefine` already is. The mode deadlocked whenever the target was permanently met — no hits
means no window, and being always met means only hits. Removed rather than fixed."""

ROLES = ("reward", "inhibit", "monitor")
"""What a target does to the feedback — **and it is not the same question as which way it is trained.**

For a long time it was: every consumer that wanted to know "reward or inhibit" asked `direction`, and
`increase` meant reward. That works for the protocols where the band you are training is the one you want
more of, and it has no room at all for the ones where it is not. Ros and Lanius' alpha-down protocol is the
plain case — 8–12 Hz at Pz, trained **down**, and it is the thing the session is *for*, so it has to
dominate the level rather than sit in the 20% the inhibits share. Under the old model it could not: it
landed in the inhibit group by virtue of pointing down, and adding one artifact inhibit beside it halved
its influence on the only output the person perceives.

So the two axes are separate, and both are named on the card:

| | |
|---|---|
| **role** | reward, inhibit or monitor — whether the movement is reinforced, suppressed, or only watched |
| **direction** | increase or decrease — **the movement in the signal this target is about** |

**Direction names the movement, not the wanted side of the line**, and that is the part worth reading
twice. A reward reinforces the movement it names; an inhibit suppresses it. So the four combinations
read as plain English and the expected effect on the band falls out:

| | met when | expect the band to |
|---|---|---|
| reward, increase | at or above the threshold | rise |
| reward, decrease | at or below | fall |
| **inhibit, increase** | at or **below** | fall |
| inhibit, decrease | at or above | rise |

"Inhibit increase on beta" is *inhibit rises in beta*, so it is met while beta stays low and you expect
beta to come down — which is what every inhibit in this codebase has always done, and it used to be
spelled "decrease" because direction meant the wanted side. That spelling made the two ordinary
arrangements read as opposites of each other ("reward increase alpha, decrease beta") when they are the
same sentence with one word changed. `wants_above` is where the two axes are resolved into the one thing
the arithmetic needs, and **nothing but `wants_above` may ask which side of the line is which**.

`monitor` is a role rather than the separate flag it used to be, so it has the same slot kinds as a
reward and an inhibit. It keeps a direction, a threshold and an aim; only the
wire to the output is cut."""

# How often the target should be met when a threshold is computed from data, per **role**. These are
# not the same number and should not be: a reward wants to be earned nearly all the time, while an
# inhibit is a condition you are asked to keep clear, so its threshold sits where most of your recent
# activity already falls. Stricter defaults exist (70% reward, 20% *over* threshold for an
# inhibit, i.e. 80% met) — these are deliberately more generous. A monitor takes the reward's.
#
# **Role and not direction**, and the split is what makes a down-trained reward possible: the aim says how
# often it should be met, and `direction` decides which tail of the distribution that lands the threshold
# in. Read off direction, a reward trained downward would have been aimed at an inhibit's 75%.
HIT_RATES = {"reward": 0.90, "inhibit": 0.75, "monitor": 0.90}

# Which mode a target opens in, keyed by **which side of the threshold it is met on** — `wants_above`,
# not the raw direction and not the role. A target met above tracks, so it keeps rewarding as amplitude
# drifts over a session; one met below holds still, because a tracker follows the trainee down and pins
# the percentage where it started, which erases the very change the training exists to produce.
#
# That reasoning is about where the line sits relative to the signal, which is exactly what `wants_above`
# resolves the two axes into: an ordinary inhibit (increase, suppressed) and an alpha-down reward
# (decrease, reinforced) are both held below a line and both correctly open fixed.
DEFAULT_MODES = {True: "automatic", False: "fixed"}

CALIBRATION_SECONDS = 30.0  # what a fixed threshold is derived from, at session start or on redefine

# How many values must be expected on the far side of the threshold before a percentile *of the data* is
# worth trusting. Below this the threshold comes from a one-parameter fit instead — see `Target.derivation`.
TAIL_EVENTS_NEEDED = 10.0
RAYLEIGH_MEDIAN = 1.1774100225154747   # sqrt(2*ln 2): a Rayleigh's median in units of its scale

SPREAD_SAMPLES = 4.0
"""Independent values needed before the *spread* of the readings can be measured rather than assumed — see
`Target.derivation`, and note that this route is only reached where the assumption has been declared
inapplicable.

**Low on purpose, because of what it is competing with.** A standard deviation from four values carries
about 35% error of its own, which would be a poor trade against a shape that fits — but where it is
reached, the alternative is a 2.5th percentile of four numbers, which is an extrapolation off the sample
minimum. Waiting for a precise spread would mean minutes of a threshold nothing can reach, on a screen
whose entire feedback is whether it is reached.

It also errs the better way. A spread estimated from too few values comes out *small*, which puts the
threshold nearer the middle and makes the reward too hard — and this protocol's own reasoning already says
which way to err: a threshold that is too easy is invisible, while one that is too hard is obvious in the
first minute."""

MEASURE_SPREAD_OVER_SHAPE = 20.0
"""Independent values above which a **measured** spread is preferred to the assumed Rayleigh one, even for
a band amplitude the shape is supposed to fit.

The Rayleigh is the envelope of narrowband *noise*, and a reward band is the one place in this program
where the signal is meant not to be noise. A sustained rhythm is a steady component plus noise — Rician,
not Rayleigh — and it is far less spread out, so a Rayleigh fitted to it puts the threshold much too low
and the target sits permanently met. Which is exactly what was reported: an 11.7–14.7 Hz reward at a 90%
aim reading 100% of the last 30 s immediately after `redefine`.

Measured on `sessions/20260812`, that band, 90% aim, nine consecutive 30 s windows: the Rayleigh fit
delivered **100%** both in its own window and in the next one, where the measured-spread fit delivered
91% and 84% and a plain percentile 90% and 82%. The band's log-envelope spread is 0.28 against the 0.64 a
Rayleigh implies — a signal a Rayleigh cannot describe, and 90 independent values with which to notice.

**Twenty, because that is where measuring stops costing more than assuming.** Simulated at a 90% aim,
4000 windows per cell, scoring what each estimator actually delivers: on genuinely Rayleigh data the
shape fit holds 89–90% at every window size while the measured spread runs 81% at 6 values, 84% at 12,
86% at 25 and 87% at 90 — so below about twenty the assumption is worth more than the measurement.
On Rician data the shape fit reads 94% at every size and does not improve with more data, because it is
not wrong by noise, it is wrong by shape.

The residual bias is real and it is the right way round: measuring the spread lands a few points *under*
the aim where assuming can land ten or more points over. Under-shooting shows up as a target that has to
be worked for; over-shooting shows up as feedback that says nothing at all, all session, and the older
constant above already argues for the same direction."""
AUTOMATIC_WINDOW = 60.0     # rolling window the automatic mode tracks
RECENT_WINDOW = 30.0        # what "on target" is reported over
TREND_POINTS = 1200
RECENT_POINTS = 4000        # ≈200 s at the 20 Hz metric rate


def _normal_quantile(p: float) -> float:
    """How many standard deviations below the middle `p` of a normal distribution falls.

    By bisection on the error function, which is exact to a millionth in forty cheap steps. The alternative
    is `scipy.special.ndtri`, and scipy is not otherwise imported by anything in this module's chain — a
    dependency the whole target layer would pay for at import time, for one number computed once a second
    during a calibration.
    """
    import math

    p = min(max(p, 1e-9), 1 - 1e-9)
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Target:
    """One trained quantity. Deliberately shared by every protocol.

    Thresholding, calibration, the percentile modes, time-on-target and the soft score are the same
    problem whichever screen is asking, and they are the parts with the subtle failures in them. So a
    protocol contributes *what* is measured and lets this hold the rest — a CFC synchrony reward over a
    channel pair is this class with `pair` set, not a parallel implementation with its own bugs.
    """

    source: int
    band: Band | None
    """The frequency band measured, or **None for a quantity that has no band** — theta minus alpha is one
    number derived from two other targets, so it has no edges, no filter and no frequency.

    `None` is not an absence to be worked around: it is what says this target gets no card, no colour and no
    filter gear. Giving it a nominal band instead — 2-11 Hz, spanning the two it is built from — was tried
    and it put two inert edge boxes and a filter dialog on the screen next to a threshold that worked, which
    is precisely the shape these conventions exist to catch. See `derivation`, which was silently right for
    the wrong reason under the fake band."""
    direction: str = "increase"
    """The movement in the signal this target is about — see `ROLES` for the table.

    **Increase unless something says otherwise**, and that is now true of inhibits too: the ordinary
    "keep beta down" inhibit is `increase` (rises in beta are what is being suppressed) where it used to
    be spelled `decrease`. A target names a rise and the role says what is done about it."""

    role: str = "reward"
    """`reward`, `inhibit` or `monitor`. See `ROLES`.

    **Reward outright, rather than derived from the direction**, which is what it was for one commit and
    which kept the old conflation alive in the constructor: an `increase` target would have been a reward
    and a `decrease` one an inhibit, so the two axes were still one until somebody said otherwise. Every
    screen that builds an inhibit says `role="inhibit"` at the point it builds it, which is a word in
    place of a rule nobody could see."""

    threshold: float = 10.0
    mode: str = ""                      # "fixed" | "automatic"; empty means take the direction default

    pair: tuple[int, ...] | None = None
    """The channels the measurement spans, with `source` = PAIR. Two for a synchrony reward; any number for
    a target measured on several channels summed, which is what alpha-theta does when more than one
    electrode is in use. Set even for one channel there, so that the label says which one and so that
    `target_is_live` requires every contributing channel rather than guessing."""
    signs: tuple[int, ...] | None = None
    """A sign per channel in `pair`, `+1` or `-1`, or None for the plain sum where every one is added.

    **A difference is a sum with a sign, and that is the whole of it** — the channels are lined up by
    sample index and combined before anything filters either way, so the estimator downstream cannot tell
    which it was given. What the sign buys is the montage: under a linked reference every channel is
    site-minus-reference, so `ch0 − ch1` is `(C4 − Pz) − (C3 − Pz)` and the reference cancels exactly,
    leaving C4 − C3 without rewiring anything. The plain sum has the opposite property — a shared
    reference appears in both channels and adds in phase.

    **A signed combination does not follow the live channels the way a sum does.** Dropping a contributor
    from a sum leaves a smaller sum; dropping one side of a difference leaves a different montage, not a
    degraded one. So `summed_sites` re-points the plain sum only, and a signed target goes inert through
    the ordinary `target_is_live` rule when any channel it names dies."""
    name: str = ""
    """Overrides the label, for a target that is not "chN low–high" — a 0.2 Hz synchrony reward reads as
    "5.0 Hz sync", because "4.9–5.1" states the implementation where the centre states the intent."""
    tone_hz: float | None = None
    """The audible pitch a keyed reward sounds at, in Hz. None derives it from the band — see
    `cues.tone_pitch`, whose anchors are two reward tones measured at 137 Hz and 738 Hz.

    Settable because the derived spread is a *default*, not a fact about hearing: the whole job of the pitch
    is to be told apart from the other reward's with the eyes closed, and how far apart two pitches need to
    be for that is a property of the listener and the room. The derivation puts theta and alpha two and a
    half octaves apart; a third reward between them lands between the
    two pitches and may want moving by hand."""
    unit: str = "µV"
    filter_kind: str = ""
    """`cfir`, `butterworth` or `elliptic`; empty takes the direction's default. A *per-target* choice
    because the trade is per target — a reward's amplitude is a proportional signal whose phase matters, an
    inhibit only has to answer whether this band crossed a line and not the neighbouring one — and because
    one band can be trained in both directions at once."""
    filter_order: int = 0
    """Per edge, for the IIR kinds; 0 takes the default. Not used by `cfir`, which is an FIR designed
    around a delay and so has `cfir_delay` instead — each kind exposes the parameter it actually has."""
    cfir_delay: float = 0.0
    """The delay cFIR is designed around, in seconds; 0 takes the direction's default."""
    filter_ripple: float = 0.0
    """Passband ripple in dB, for `elliptic` only; 0 takes the default. The trade that defines that shape —
    more unevenness inside the band buys steeper skirts at the same order.

    Envelope smoothing is deliberately **not** here. It is one constant for the whole session
    (`FeedbackSettings.envelope_seconds`), applied to every target, where the per-target entries are runtime state."""
    hit_rate_override: float | None = None
    """Per-target aim, where a protocol's own convention differs from ours. CFC's twelve inhibit bins
    have to total 5–15% above threshold between them, which is about 1% each — nothing like the 25% our
    own single-inhibit default allows, and using the default would put the whole budget in one bin.

    Also what the operator sets, through `set_aim` — see `aim_by_hand`."""
    reach_rate: float | None = None
    """A **second, higher aim**, or None for a target with one line — which is every target but a synchrony
    reward. The share of readings that should fall past the higher line, so for a target met above it is
    smaller than `hit_rate`: 0.10 against 0.975 puts the second line in the top tenth of the readings.

    It exists because one screen trains a *coincidence* rather than a level. Two rewards each individually
    past their thresholds is the ordinary state that screen is arranged to keep; both well past a second
    line at the same moment is the event it is named for, and a line for that needs its own aim rather than
    a multiple of the first — see `reaches` and `cfc_view.REACH_RATE`."""
    reach_by_hand: bool = False
    """The same promise `aim_by_hand` makes, for the second aim."""
    aim_by_hand: bool = False
    """Whether the aim was set by the operator rather than derived by the protocol.

    The same promise `nudge` makes about a threshold, one level up: a number somebody typed is not
    overwritten by a recalculation. CFC's rewards share a fixed interruption budget and re-derive every
    reward's aim whenever the set changes, so without this, adding a third reward would silently undo a
    hand-set aim on the other two."""
    window_seconds: float = 0.0
    """The rolling window an `automatic` threshold is derived from, in seconds. 0 takes
    `AUTOMATIC_WINDOW`.

    The window is 1–100 s, and on the ILF screen the value is picked *from the trained frequency*:
    10 s above 0.01 Hz and 30 s at or below it. That is the right shape — a window has
    to hold enough cycles of the thing being measured to describe it, and at 0.01 Hz a 10 s window holds a
    tenth of one — and it cannot be a constant here for the same reason the bandwidth cannot: this screen
    spans four decades.

    Per target rather than per session because the rewards on one ILF screen are at 0.05, 10 and 40 Hz and
    each needs its own."""
    recompute_seconds: float = 0.0
    """How often an `automatic` threshold is re-derived, in seconds. 0 means as often as is cheap — once a
    second — which is what it has always done.

    **Optional, and the reason to set it is not performance.** A tracker recomputing continuously holds
    percent-over-threshold at the aim *by construction*, so the quantity being trained cannot show a change;
    one that holds still for a period makes that percentage a real number within the period while still
    absorbing drift between them. The period is operator-chosen, 0–1000 seconds, and it is the instrument that sits between `fixed` and a continuous tracker rather than
    a compromise between them.

    Generic rather than per protocol: nothing about it depends on what is being measured."""
    narrowband_envelope: bool = True
    """Whether this target's value is the **envelope of a narrowband signal**, which is what makes the
    Rayleigh fit in `_apply_threshold` a legitimate estimator rather than an arbitrary one.

    True for every band amplitude, which is nearly everything here. False for a quantity with a different
    distribution, and ILF Synchrony's rectified product is the case that forced this to be said out loud: it
    is `max(0, x·y)`, so about half its mass sits in an atom at exactly zero and the rest is a product of two
    envelopes. A Rayleigh has neither property. Fitting one to it is not a conservative approximation — it is
    a one-parameter fit of the wrong shape, and it fails worst in the tail, which is the only part a threshold
    reads.

    **Separate from `bandwidth_hz`, and the two were conflated once already.** That one answers *how many
    independent values* a window holds; this one answers *which estimator shape is valid*. The derived-quantity
    target got the right route by returning an infinite bandwidth, which its own docstring admits was right
    for the wrong reason — this is the reason, stated where it belongs."""
    signal_floor: float | None = None
    """A floor under the *value*, in the target's own unit — below it the target is never met, whatever the
    threshold says. None for no floor, which is the default.

    **Not `floor`, which bounds the threshold.** That one stops a quiet stretch driving an inhibit's threshold
    to zero; this one stops a *dead or near-dead channel earning a reward*. A derived threshold on a channel
    with nothing on it calibrates to nearly nothing, so the target is met about as often as its aim asks and
    the reward sounds all session on noise. `target_is_live` catches the case where the device says so, which
    needs the device to report no-signal rather than merely a very quiet channel — this catches the rest.

    The alpha-theta screen gates its reward on 1.0 µV in exactly this position, upstream of
    the comparison."""
    floor: float | None = None
    ceiling: float | None = None
    """Bounds on the threshold itself, in the target's own unit. None for neither, which is the default.

    An autothresholded target has one failure in each direction and both are self-sustaining. Without a
    floor, one quiet stretch drives an inhibit's threshold toward zero and it then fires forever. Without a
    ceiling, one artifact-laden stretch pushes it up and it never fires again.

    A target therefore carries three numbers for this reason — a desired percentage and a minimum and
    maximum µV beside it. For example a high-beta inhibit can be 15% above threshold with a 6 µV floor
    under it. **This bounds a training threshold and
    is not artifact rejection**, though both are µV limits; see `nfb.dsp.artifact` for the other one.

    Applied wherever the threshold is set, including by hand: a nudge past the floor would otherwise
    survive until the next recalibration and be silently undone by it."""

    reach_threshold: float = 0.0
    """The higher line itself, derived in the same call as `threshold` and never on its own. Meaningless
    while `reach_rate` is None, which is why nothing reads it without checking that first."""

    value: float = 0.0
    hits: int = 0
    total: int = 0
    session_sum: float = 0.0
    trend: deque = field(default_factory=lambda: deque(maxlen=TREND_POINTS))
    _recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_POINTS))
    _settled: bool = False              # a fixed threshold that has been derived, or set by hand
    _rate_hz: float = 20.0
    _hits: deque = field(default_factory=lambda: deque(maxlen=RECENT_POINTS))
    _spread: float = 0.0
    _spread_at: int = -1        # the `total` the cached spread was computed at; see `spread`
    _derived_at: int = 0        # the `total` an automatic threshold was last recomputed at

    def __post_init__(self) -> None:
        if not self.mode:
            self.mode = DEFAULT_MODES[self.wants_above]

    @property
    def hit_rate(self) -> float:
        """The fraction of time this target is aimed at meeting. **From the role** — see `HIT_RATES`."""
        if self.hit_rate_override is not None:
            return self.hit_rate_override
        return HIT_RATES[self.role]

    def set_aim(self, rate: float) -> None:
        """Aim to be met `rate` of the time, and move the threshold to where that is true.

        **The aim is the threshold's cause, so setting one has to re-derive the other.** Every derivation
        here is "put the line where `hit_rate` of the readings fall on the wanted side" — change the aim and
        the old threshold is simply the answer to a question nobody is asking any more. A fixed threshold
        with its window already behind it therefore re-derives at once, which is what `redefine` is; one
        still calibrating picks the new aim up on its next provisional pass, a second later; an automatic
        one on its next recomputation.

        Bounded away from 0 and 1 because both are degenerate: an aim of 100% asks for a threshold below
        every reading the signal can produce, which is the state a sub-hertz reward was stuck in, and 0%
        asks for one above all of them. Either way the target stops being able to say anything, and the
        feedback riding it stops moving.
        """
        self.hit_rate_override = min(0.99, max(0.01, float(rate)))
        self.aim_by_hand = True
        if self.mode == "fixed":
            self.redefine()
        else:
            self.rescore()

    def set_reach(self, rate: float) -> None:
        """The same, for the higher line. Bounded away from 0 and 1 for the same reason."""
        if self.reach_rate is None:
            raise ValueError("this target has no higher line")
        self.reach_rate = min(0.99, max(0.01, float(rate)))
        self.reach_by_hand = True
        if self.mode == "fixed":
            self.redefine()

    # --- identity ------------------------------------------------------------------------

    @property
    def source_label(self) -> str:
        """What this target is measured on. A multi-channel target reads "ch0+ch1" or "ch0−ch1", which is
        the arithmetic: those channels combined, then measured."""
        if self.pair is None:
            return f"ch{self.source}"
        signs = self.signs or (1,) * len(self.pair)
        out = ""
        for n, (ch, sign) in enumerate(zip(self.pair, signs)):
            if n == 0:
                out = ("−" if sign < 0 else "") + f"ch{ch}"
            else:
                out += ("−" if sign < 0 else "+") + f"ch{ch}"
        return out

    @property
    def label(self) -> str:
        if self.name:
            return f"{self.source_label} {self.name}"
        if self.band is None:
            return self.source_label
        return f"{self.source_label} {self.band.low:g}–{self.band.high:g}"

    @property
    def banded(self) -> bool:
        """Whether this target measures a frequency band, and therefore whether it gets a card.

        The one question that decides the whole rendering: edges to nudge, a colour from its centre
        frequency, a filter to configure, an unsigned µV bar. A derived quantity has none of those.
        """
        return self.band is not None

    @property
    def monitor(self) -> bool:
        """Measured and drawn, and deliberately **not** driving the feedback.

        A fourth kind beside raw, reward and inhibit. Without it, watching a band means
        adding it as a target and thereby changing what is being trained, which is the opposite of
        watching.

        **A value of `role`, and it used to be a flag beside `direction`.** The flag's own reasoning —
        that a monitor still has a direction, a threshold and an aim, and only the wire to the output is
        cut — is the argument for it being a *role*: those are exactly the things role does not decide.
        It stayed a flag only because there was no role to be a value of. Cutting the wire happens in
        `TrainingSession.active_targets`, so one place decides and the card, the trace and the spectrum
        are untouched.
        """
        return self.role == "monitor"

    @property
    def wants_above(self) -> bool:
        """Whether this target is met **at or above** its threshold. The two axes, resolved.

        `direction` names a movement and `role` says whether that movement is reinforced or suppressed,
        so an inhibit inverts: "inhibit increase" is met while the band stays low. Every piece of
        arithmetic that needs a side — `meets`, `score`'s sign, which tail the threshold is derived from,
        the default mode — asks this and never asks `direction`, because asking `direction` is how the
        two became one field in the first place.
        """
        return (self.direction == "increase") != (self.role == "inhibit")

    @property
    def role_label(self) -> str:
        """REWARD, INHIBIT or MONITOR — the word on the card's left chip, and the colour it takes."""
        return self.role.upper()

    @property
    def verb(self) -> str:
        """INCREASE or DECREASE — the right chip. **It no longer says MONITOR**, because a monitor has a
        direction like anything else and printing the role here was the conflation in one property."""
        return "INCREASE" if self.direction == "increase" else "DECREASE"

    # --- state ---------------------------------------------------------------------------

    def meets(self, value: float) -> bool:
        """Whether `value` is on the wanted side. **The single rule, and it had two implementations.**

        `wanted` applied the signal floor and `rescore` did not, so a target with a floor answered "am I on
        target" one way and "what fraction of the last 30 s was I on target" another. Measured on ILF
        Synchrony against the no-electrode bench recording: the product is zero, the threshold derives to
        `0.0`, `wanted` correctly said no because of the floor, and `recent_percent` said **100%** because
        `rescore` compared `0 >= 0` and never saw the floor. One rule, called from both.

        **The floor follows the role, and it used to follow the direction.** It guards one failure: a
        reward calibrating onto noise and then paying out on it all session. That is a property of what
        the target *drives*, not of which way it points — so an inhibit is exempt (the region below the
        floor is the most inhibited state there is, and failing it inverted the protocol: on alpha-theta
        a high-beta band quieter than 1 µV failed its inhibit, and since every inhibit gates the reward
        tone, the tone cut out exactly when the trainee arrived where the training points — eyes closed,
        nothing else on, no explanation) and **a reward trained downward is not**.

        That last case is the one the role split creates, and it is the same hazard wearing the other
        sign: a dead electrode reads near zero, satisfies an alpha-down reward perfectly, and earns for
        the whole session. Reading the floor off `direction` would have exempted it silently. The
        device-validity path (`target_is_live`) catches a channel the device knows is dead; the floor
        catches the one it does not.
        """
        return self._past(value, self.threshold)

    def _past(self, value: float, line: float) -> bool:
        """`meets` and `reaches` differ only in which line they ask about, so the rule is written once.

        The docstring above is the record of what two implementations of this comparison cost the last
        time there were two. A second line is exactly the moment that recurs."""
        if (self.role != "inhibit" and self.signal_floor is not None
                and abs(value) < self.signal_floor):
            return False
        return value >= line if self.wants_above else value <= line

    def reaches(self, value: float) -> bool:
        """Whether `value` is past the **higher** line. False for a target that has none.

        There is deliberately no stored history of this and no `rescore` counterpart. What the higher line
        produces is moments, and a moment is a thing that happened rather than a judgement to be revised
        when a threshold moves — the record of them lives on the session, in the plot and in the
        recording's companion file."""
        if self.reach_rate is None:
            return False
        return self._past(value, self.reach_threshold)

    @property
    def wanted(self) -> bool:
        return self.meets(self.value)

    @property
    def percent(self) -> float:
        """Over the whole session. Kept for a session summary, not for the live card: a percentage over
        twenty minutes barely moves, so it cannot tell you whether the last minute went well."""
        return 100.0 * self.hits / self.total if self.total else 0.0

    def recent_percent(self, seconds: float = RECENT_WINDOW) -> float | None:
        """On target over the last `seconds`, or None until there is a window's worth."""
        need = int(seconds * self._rate_hz)
        if len(self._hits) < need or need <= 0:
            return None
        window = list(self._hits)[-need:]
        return 100.0 * sum(window) / len(window)

    @property
    def session_average(self) -> float:
        return self.session_sum / self.total if self.total else 0.0

    # --- calibration ---------------------------------------------------------------------

    @property
    def tracking(self) -> bool:
        """Whether the threshold is derived from a rolling window rather than held."""
        return self.mode == "automatic"

    @property
    def window(self) -> float:
        """The rolling window this target's tracker uses. See `window_seconds`."""
        return self.window_seconds or AUTOMATIC_WINDOW

    @property
    def provisional(self) -> bool:
        """Whether the line is still being derived from a window that has not filled once — a fixed
        target calibrating, or a tracker in its first window. **The gate anything downstream should ask**,
        rather than `calibrating`, which is the fixed half of it only."""
        return self.calibrating or (self.tracking and self.window_remaining > 0)

    @property
    def window_remaining(self) -> float:
        """How long until an automatic threshold's window has filled once — the tracker's own
        "calibrating": the line is provisional until then and the card says so."""
        rate = max(1.0, self._rate_hz)
        return max(0.0, self.window - self.total / rate)

    @property
    def calibrating(self) -> bool:
        """A fixed threshold with no history behind it yet. Not a mode — a phase."""
        return self.mode == "fixed" and not self._settled

    settle_seconds: float = 0.0
    """How long this target's estimator takes to mean anything, prepended to the calibration.

    Zero for a fast band, whose estimator settles inside the first second of the window. It stops being
    ignorable exactly where the bands get slow: an infra-slow synchrony lock-in carries ~10 s of delay,
    so most of a 30 s window was the estimator ramping up from zero — measured on a clean 0.05 Hz pair,
    the derived threshold sat about 30% low, held for the whole session because CFC's targets are
    `fixed`. The calibration therefore runs `settle_seconds + CALIBRATION_SECONDS` and derives from the
    last `CALIBRATION_SECONDS` of it, which is the settled part. The screen that builds the target sets
    this from the estimator it wires up, since only it knows which estimator that is."""

    calibration_seconds: float = CALIBRATION_SECONDS
    """How long a window this target's threshold is derived from. Thirty seconds unless a screen says
    otherwise, and one does.

    **Thirty seconds of a sub-hertz band is two readings.** A band's amplitude cannot change faster than its
    own width, so 30 s of a 0.05 Hz band holds about 1.5 independent values — and every estimator of spread
    over such a window is reporting the accident of which part of one slow cycle it landed on. Widening it is
    the only thing that adds information: at 300 s the same band offers about 15, which is enough to estimate
    a spread as well as a level, and that is what makes the difference between a threshold that is placed and
    one that is guessed. See `derivation`, where the two estimators divide on exactly that count.

    Nothing waits for it. The provisional derivation in `observe` runs from the first second and re-derives
    every second throughout, so a five-minute window is a threshold that keeps improving for five minutes,
    not a five-minute blank — `calibrating` stays true so nothing downstream mistakes it for final."""

    @property
    def calibration_remaining(self) -> float:
        need = int((self.settle_seconds + self.calibration_seconds) * self._rate_hz)
        return max(0.0, (need - len(self._recent)) / self._rate_hz)

    @property
    def can_redefine(self) -> bool:
        """Not applicable to automatic, which is already recomputing from a rolling window."""
        return self.mode == "fixed"

    def redefine(self) -> None:
        """Set a fixed threshold from the last 30 s. Immediate when that much history exists, which is
        the normal case mid-session; otherwise it calibrates as the data arrives.

        **"Enough history" is the same test the opening calibration uses, and it was not.** This used to
        accept any twenty readings — one second at the metric rate — and then mark the result final. On a
        slow reward that is a threshold derived from the estimator's ramp: measured on a 0.1 Hz synchrony
        reward, pressing redefine two seconds after retuning to it set the threshold to 19 µV while the
        reading at that moment was 306 and the settled signal ran 110–228. Fixed thresholds do not
        re-derive, so that number was then the whole session's.

        `calibration_remaining` already knows the answer — it counts `settle_seconds` of estimator ramp plus
        a full window — so the immediate path now asks it, and everything short of that falls through to the
        calibrating branch in `observe`, which counts down on the card and re-derives as data arrives.
        """
        self.mode = "fixed"
        if self.calibration_remaining <= 0:
            self._apply_threshold(self._window(self.calibration_seconds), self.calibration_seconds)
            self._settled = True
        else:
            self._settled = False

    # --- band edges ----------------------------------------------------------------------

    @property
    def colour(self) -> str | None:
        """Every view keys a band's colour off its centre frequency — so a band is
        the same colour on its card, on the graph and in the spectrum.

        **None where there is no band**, and that is the rule rather than a missing case: the whole scheme is
        one colour per frequency, so a quantity with no frequency has no colour to be given. Assigning one
        anyway would put a swatch on screen that agrees with nothing else on it — which is the failure
        `SERIES_COLOURS[n]` was removed for, arrived at from the other direction.
        """
        if self.band is None:
            return None
        return theme.band_colour(self.band.low, self.band.high)

    def set_band(self, low: float | None = None, high: float | None = None) -> None:
        if self.band is None:
            raise ValueError("this target has no band to set")
        lo = self.band.low if low is None else low
        hi = self.band.high if high is None else high
        if not 0 <= lo < hi:
            raise ValueError(f"band {lo}–{hi} is not a range")
        self.band = Band(lo, hi, "")
        # Not a reset: see `rescore`. A band edit is usually a 0.1 Hz nudge, and blanking the card for
        # thirty seconds each time is worse than a window that is briefly a blend.
        self.rescore()

    def shift_band(self, delta: float, lowest: float = 0.1) -> None:
        """Slide the band by `delta`, keeping its width.

        What the arrows on a band edge do, and typing into the box is what sets an edge on its own. Nudging
        is how you *hunt for a frequency* — "often within 1-2 Hz of the frequency in the spectral display
        showing the highest amplitude" — and that is a move, not a resize. Nudging one edge alone takes two
        presses in opposite boxes per step and silently changes the width if you lose count.

        Clamped at the bottom rather than refused, so holding the down arrow parks the band instead of
        raising out of a click handler; the width is preserved even there.
        """
        width = self.band.high - self.band.low
        low = max(lowest, self.band.low + delta)
        self.set_band(round(low, 6), round(low + width, 6))

    def rescore(self) -> None:
        """Recompute the hit record against the current threshold, keeping every value.

        **Moving a threshold does not change what was measured, only the comparison** — so throwing the
        history away and making the card read "waiting" for thirty seconds was destroying information it
        already had. Every value in `_recent` can be re-judged against the new number exactly, which gives
        the correct "last 30 s on target" *immediately* rather than after a full window.

        The same is done for a band edit. There the measurement really does change, so the window is briefly
        a blend of the old band and the new one — but it heals within one window on its own, and it is a much
        better answer than a blank card, especially since the commonest band edit is a 0.1 Hz nudge while
        hunting for a frequency. Assume continuity; a change of a tenth of a hertz is not a new session.

        Session counters are deliberately left alone: `hits` and `total` are session-long and cannot be
        recomputed from a 200 s window, so a band edit mixes them slightly. They feed a summary, not the live
        readout — see `percent` against `recent_percent`.
        """
        self._hits = deque((self.meets(v) for v in self._recent), maxlen=RECENT_POINTS)

    def inherit(self, previous: "Target") -> None:
        """Take over another target's history, for a retune that is a **nudge and not a new measurement**.

        Some screens cannot edit a band in place: a synchrony reward's whole estimator is built around its
        centre frequency, so moving the frequency means a new lock-in and, until now, a new `Target` with an
        empty window. On a fast band that costs half a minute of "calibrating". On a sub-hertz one it costs
        *minutes*, and hunting a hailing frequency is done in small steps — "shifts of 0.1 Hz at a time" —
        so the calibration could be restarted from zero ten times before the frequency was found, and never
        finish. Continuity is the same judgement `rescore` already makes for an in-place band edit: a nudge
        is not a new session.

        The caller decides whether it *is* a nudge, because only the screen knows what changed — see
        `cfc_view.retune_rewards`, where a moved centre inherits and a changed width or measure does not.
        What comes across is the window, the hit record, the session counters and the threshold with its
        settled flag; what does not is the band, which is the thing being changed.

        The estimator behind the new target still has to fill, so the first readings after a nudge are its
        ramp. They are a small fraction of a long window, and a settled threshold does not re-derive at all.
        """
        self._rate_hz = previous._rate_hz
        self._fit_window()
        self._recent.extend(previous._recent)
        self._hits.extend(previous._hits)
        self.trend.extend(previous.trend)
        self.hits, self.total = previous.hits, previous.total
        self.session_sum = previous.session_sum
        self.threshold, self.mode = previous.threshold, previous.mode
        self.reach_threshold = previous.reach_threshold
        self._settled = previous._settled
        if previous.aim_by_hand:                # a number somebody typed outlives a retune, as it does a stop
            self.hit_rate_override, self.aim_by_hand = previous.hit_rate_override, True
        if previous.reach_by_hand:
            self.reach_rate, self.reach_by_hand = previous.reach_rate, True
        self._derived_at = previous._derived_at
        self._spread_at = -1            # recomputed on demand; see `spread`

    def rescale_history(self, factor: float) -> None:
        """Every collected reading multiplied, for a change of *unit* rather than of quantity.

        Switching a target's filter between cFIR and the IIR kinds changes the amplitude convention — peak
        against RMS or mean-absolute — which is one constant ratio and nothing else: the same rhythm, in a
        different unit. That used to call `reset_stats`, on the argument that a window holding two
        conventions holds neither. True of the window; not true of the readings, which can each be
        converted. What it cost was the whole calibration: pressing a filter pill mid-session blanked the
        card and started a thirty-second countdown, reported as exactly that.

        The threshold is already rescaled by this factor, so scaling the readings with it keeps the two in
        the same unit and the hit record exactly as it was — the card answers on the next frame instead of
        going to "waiting". The approximation is the same one the threshold's rescale already accepts:
        the ratio is exact for a sine and about 11% off for narrowband noise, which is why the *derived*
        threshold is worth re-deriving at leisure but is not worth a blank card.
        """
        if not factor or factor <= 0 or factor == 1.0:
            return
        self._recent = deque((v * factor for v in self._recent), maxlen=self._recent.maxlen)
        self.trend = deque((v * factor for v in self.trend), maxlen=self.trend.maxlen)
        self.session_sum *= factor
        self._spread_at = -1        # the cached spread is in the old unit

    def reset_stats(self) -> None:
        """A genuine restart: the numbers are about something else now. For a *source* change, where the
        values came from a different electrode entirely, not for a threshold or a band edit."""
        self.hits = self.total = 0
        self.session_sum = 0.0
        self.trend.clear()
        self._recent.clear()
        self._hits.clear()
        self._settled = False
        self._spread_at = -1
        self._derived_at = 0

    # --- thresholding --------------------------------------------------------------------

    def _window(self, seconds: float) -> list[float]:
        return list(self._recent)[-int(seconds * self._rate_hz):]

    def _fit_window(self) -> None:
        """Make sure the recent-values buffer can hold what this target derives from.

        `RECENT_POINTS` is 200 s at the metric rate, which was more than any window needed until a
        sub-hertz target asked for a five-minute one — and a deque that cannot hold the window silently
        shortens it, so the calibration would have counted down to a number it could never reach. Grown
        once, on the first reading, since the rate is not known before then.
        """
        need = int((self.settle_seconds + self.calibration_seconds) * self._rate_hz) + 1
        if need > (self._recent.maxlen or 0):
            self._recent = deque(self._recent, maxlen=need)
            self._hits = deque(self._hits, maxlen=need)

    def observe(self, value: float, rate_hz: float) -> None:
        if rate_hz != self._rate_hz or self.calibration_seconds > CALIBRATION_SECONDS:
            self._rate_hz = rate_hz
            self._fit_window()
        self.value = value
        self.total += 1
        self.session_sum += value
        met = self.wanted
        self.hits += int(met)
        self._hits.append(met)
        self._recent.append(value)

        if self.mode == "automatic":
            # Once a second of data at the fastest, not on every value. `_apply_threshold` runs
            # `np.percentile` over the whole 60 s window — 1200 values at 20 Hz — and profiling a running
            # session put it among the heaviest things in the process. A threshold derived from 60 seconds
            # cannot meaningfully move in 50 ms, so recomputing twenty times a second bought nothing.
            #
            # `recompute_seconds` then makes the period an operator setting, which closes gap 2. The difference between one second and one *minute* is not an optimisation: a
            # tracker that recomputes continuously pins percent-over-threshold at the aim by construction,
            # while one that holds still for a period makes that percentage a real number **within** the
            # period — which is the only way an operator can see a change and the tracker can still absorb
            # slow drift. Default 0, meaning as often as is cheap, which is exactly the old behaviour.
            # **The period starts once the window has filled, not before.** Until then the line is
            # re-derived every second from whatever has arrived, as a fixed target's calibration is —
            # otherwise a 90 s period meant 90 s on the constructed placeholder, which on a screen whose
            # readings live between 0 and 5% is a reward never met, i.e. no feedback at all.
            period = (max(1.0, self.recompute_seconds)
                      if self.recompute_seconds and self.window_remaining <= 0 else 1.0)
            if self.total - self._derived_at >= period * max(1.0, self._rate_hz):
                window = self._window(self.window)
                if len(window) >= 20:
                    self._derived_at = self.total
                    self._apply_threshold(window, self.window)
        elif not self._settled:
            # **Provisional from the first second, final at 30 s.** This branch used to do nothing at all
            # until the full window had arrived, which meant a fixed target spent its first 30 s
            # thresholded at whatever it was *constructed* with — the class default of 10 µV for anything
            # that did not pass one. On a narrow synchrony reward that is a number the signal may never
            # reach, so the screen was not giving coarse feedback for half a minute, it was giving none.
            #
            # So the threshold is derived from whatever has arrived, from the moment there is enough of it
            # to derive one, and re-derived as more comes in. It moves — visibly, and it should: the card
            # already says "calibrating, N s left" beside it, so a number that settles under a countdown is
            # the honest picture of what is happening. `_settled` stays false throughout, so the 30 s
            # derivation still lands exactly as it did and nothing downstream can mistake a provisional
            # number for a final one.
            #
            # Thin windows are handled and not merely tolerated: `derivation` routes to the Rayleigh fit
            # until there are `TAIL_EVENTS_NEEDED` values expected in the tail, which is exactly the early
            # case, and is why a 1% aim on a narrow band gets a usable number from a few seconds instead
            # of an extrapolation off the sample minimum.
            #
            # Rate-limited for the reason the automatic branch is: `_apply_threshold` runs a percentile over
            # the window, and doing that on every one of 20 values a second was measured among the heaviest
            # things in the process.
            # The final derivation is **not** rate-limited, so "settled" happens the moment the window is
            # full rather than up to a second after it. Otherwise the countdown would reach zero and the
            # label would linger, which reads as a stuck screen.
            #
            # **The route is chosen from the window being aimed at, not from what has arrived**, and that
            # is deliberate: which estimator suits this target is a property of its bandwidth and its aim,
            # both known from the start, and re-deciding it every second would change the *kind* of number
            # on the card several times during one calibration. Routing on what had arrived was tried and
            # is worse where it shows: two seconds into an alpha target it picks the estimator meant for a
            # handful of independent values and halves the provisional threshold, on the very screen whose
            # first half-minute this branch exists to make usable.
            done = self.calibration_remaining <= 0
            if done or self.total - self._derived_at >= max(1.0, self._rate_hz):
                window = self._window(self.calibration_seconds)
                if len(window) >= 20:
                    self._derived_at = self.total
                    self._apply_threshold(window, self.calibration_seconds)
                    self._settled = done

    # --- how many samples are really behind a threshold -----------------------------------

    @property
    def bandwidth_hz(self) -> float:
        """A bandless target reports an infinite one, which routes `derivation` to the percentile.

        Not a fallback: it is the correct answer for the case. A difference of two envelopes is roughly
        symmetric about zero and can be negative, so the Rayleigh fit — a one-parameter *shape* for a
        narrowband envelope — does not describe it and must not be reached. Under the nominal 2-11 Hz band
        this target used to carry it took the percentile route anyway, because 9 Hz is wide; right answer,
        wrong reason, and it would have broken silently the moment the fake band narrowed.
        """
        if self.band is None:
            return float("inf")
        return max(1e-6, self.band.high - self.band.low)

    def independent_samples(self, seconds: float) -> float:
        """Roughly how many *independent* values a window of this length holds.

        A band's envelope cannot change faster than its own bandwidth, so 30 s of a 4 Hz band holds about
        120 independent values and 30 s of a 0.2 Hz band holds about **six**. First-order, and good enough
        for the one decision it makes: the point is that the two differ by more than an order of magnitude.
        """
        return max(1.0, seconds * self.bandwidth_hz)

    def derivation(self, seconds: float = CALIBRATION_SECONDS, rate: float | None = None) -> str:
        """How a line at aim `rate` can honestly be derived from `seconds` of this target's own data.

        `rate` defaults to this target's own aim. It is an argument because a target can have two lines at
        two aims off one window, and which estimator is honest depends on the aim rather than on the
        target — a thin tail is thin for the aim that asks for it.

        A percentile is a *tail* estimate and it needs values in that tail. Aiming at 97.5% on a 0.2 Hz band
        asks for the 2.5th percentile of six numbers — where the sample minimum is already the 17th
        percentile, so the answer is an extrapolation rather than an estimate. Measured: aiming at 97.5%
        from six samples delivered anywhere between 59% and 98%.

        So below `TAIL_EVENTS_NEEDED` expected tail values it fits a **shape** instead. The envelope of a
        narrowband signal is Rayleigh, which has one parameter, and estimating one parameter from the middle
        of the data is a different and much easier problem than estimating a quantile at its edge. Measured
        on the same case: 94.6-99.1% instead of 59.7-98.4%, an **8.6x** tighter spread. In the body of the
        distribution the two are equivalent — 0.9x at a 75% aim over 120 samples — so the percentile stays
        where it already works, and nothing about the amplitude screen's defaults changes.

        **Where the Rayleigh shape does not describe the quantity, there is now something between it and a
        percentile.** `narrowband_envelope` is the screen's declaration that the one-parameter fit does not
        apply, and it used to route straight back to the percentile — which is the estimator that was
        already established as unusable on a thin window. That left the worst case with the worst tool.
        Measuring the *spread* as well as the middle is the third option: two parameters, both from the body
        of the data, which needs `SPREAD_SAMPLES` independent values rather than a populated tail.

        The case that forced it is a sub-hertz synchrony reward. Measured on a real 0.1 Hz one: the readings
        ran 110–228 µV around a median of 175, with a 5th percentile at **0.70 of the median** where a
        Rayleigh puts it at 0.27 — a third as spread out as that shape predicts, because at these
        frequencies the reading is a slowly drifting level and not a fluctuating envelope. Under the
        Rayleigh fit the threshold landed at a fifth of the median, below anything the signal ever reached,
        so the reward sat permanently met and said nothing for a whole session. A longer window cannot fix
        that on its own — the median was already well estimated, and the error was in the shape.

        Below `SPREAD_SAMPLES` there is nothing to estimate a spread from and it falls back to the
        percentile, which on two or three readings is roughly their minimum: too strict rather than too
        loose, and that is the right way round — a threshold that is too easy is invisible, one that is too
        hard is obvious in the first minute.
        """
        rate = self.hit_rate if rate is None else rate
        independent = self.independent_samples(seconds)
        if independent * min(rate, 1.0 - rate) >= TAIL_EVENTS_NEEDED:
            return "percentile"
        # **A shape is what you use when you cannot measure, not what you use when the tail is thin.**
        # This asked only whether the Rayleigh was *applicable* and, if so, took it — so a target with 90
        # independent values and a tail of 9 got a one-parameter assumption instead of a two-parameter
        # measurement, and the assumption is wrong for precisely the signal a reward is trying to produce.
        # See `MEASURE_SPREAD_OVER_SHAPE`, where the reported case and the crossover are measured.
        if independent >= MEASURE_SPREAD_OVER_SHAPE:
            return "spread"
        if self.narrowband_envelope:
            return "median"
        return "spread" if independent >= SPREAD_SAMPLES else "percentile"

    def tail_events(self, seconds: float = CALIBRATION_SECONDS, rate: float | None = None) -> float:
        """How many *independent* values a window of this length is expected to put in the threshold's tail.

        The number that says whether a threshold is determined or merely computed. Below
        `TAIL_EVENTS_NEEDED` a percentile is an extrapolation rather than an estimate, and where the Rayleigh
        fit cannot stand in — see `narrowband_envelope` — there is nothing better available and the screen has
        to say so instead of printing a number that looks settled.
        """
        rate = self.hit_rate if rate is None else rate
        return self.independent_samples(seconds) * min(rate, 1.0 - rate)

    def describe_derivation(self, seconds: float = CALIBRATION_SECONDS,
                            rate: float | None = None) -> str:
        aim = self.hit_rate if rate is None else rate
        if self.derivation(seconds, aim) == "percentile":
            pct = (1.0 - aim) * 100 if self.wants_above else aim * 100
            note = f"the {pct:.0f}th percentile of {seconds:g} s"
            tail = self.tail_events(seconds, aim)
            if tail < TAIL_EVENTS_NEEDED:
                note += (f". About {tail:.1f} independent values are expected in that tail. "
                         f"The estimate needs {TAIL_EVENTS_NEEDED:g}. The threshold wanders")
            return note
        independent = self.independent_samples(seconds)
        if self.derivation(seconds, aim) == "spread":
            return (f"the middle and the spread of {seconds:g} s. There are {independent:.0f} independent "
                    f"values. This is too few for a {aim:.1%} percentile. It is enough to measure how "
                    f"far the signal swings")
        return (f"a median fit over {seconds:g} s. There are {independent:.0f} independent values. "
                f"This is too few for a {aim:.1%} percentile. The spread is assumed, not measured")

    # --- deriving one --------------------------------------------------------------------

    def clamp(self, value: float) -> float:
        """`value` held inside this target's bounds. Identity when it has none — see `floor`."""
        if self.floor is not None:
            value = max(self.floor, value)
        if self.ceiling is not None:
            value = min(self.ceiling, value)
        return value

    def _apply_threshold(self, samples: list[float], seconds: float) -> None:
        """Put the threshold where `hit_rate` of values should fall on the wanted side, and the higher
        line where `reach_rate` does.

        **Both lines, one call, one window.** The higher line is never derived anywhere else — deriving it
        on demand from whatever the last readings happen to be would make it an automatic threshold on the
        one screen that forbids them, and it would re-run a percentile per reward per frame.
        """
        self.threshold = self._threshold_from(samples, seconds, self.hit_rate)
        if self.reach_rate is not None:
            # Clamped because the two aims can route to *different* estimators off one window, and two
            # estimators can cross.
            self.reach_threshold = max(self.threshold,
                                       self._threshold_from(samples, seconds, self.reach_rate))
        self.rescore()

    def _threshold_from(self, samples: list[float], seconds: float, rate: float) -> float:
        """The line where `rate` of `samples` should fall on the wanted side. Pure: no state is written,
        which is what lets one window produce two lines."""
        how = self.derivation(seconds, rate)
        if how == "percentile":
            pct = (1.0 - rate) * 100 if self.wants_above else rate * 100
            return self.clamp(round(float(np.percentile(samples, pct)), 2))
        if how == "spread":
            # Two parameters, both from the middle of the data: where the readings sit and how far they
            # swing, then the quantile of a *log-normal* with those. Log-normal because the quantity is an
            # amplitude — positive, and spread multiplicatively rather than additively, so "a third smaller"
            # is the natural step at 3 µV and at 300. It is also what makes the estimate robust to the one
            # thing these windows always contain: a slow drift in level, which moves the middle and leaves
            # the shape alone.
            #
            # On the measured 0.1 Hz reward this lands at 0.65 of the median against the Rayleigh's 0.19,
            # and the signal's own 5th percentile is 0.70 — the difference between a reward that can fall
            # below its threshold and one that never does.
            values = np.log(np.maximum(np.asarray(samples, dtype=float), 1e-9))
            below = (1.0 - rate) if self.wants_above else rate
            return self.clamp(
                round(float(np.exp(np.median(values)
                                   + _normal_quantile(below) * np.std(values))), 2))
        # One parameter, from the middle of the data, which is why six values can support it and a
        # percentile cannot.
        scale = float(np.median(samples)) / RAYLEIGH_MEDIAN
        wanted = rate if self.wants_above else 1.0 - rate
        return self.clamp(round(scale * float(np.sqrt(-2.0 * np.log(max(wanted, 1e-12)))), 2))

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "fixed":
            self.redefine()             # derive one now if we can, else calibrate

    def nudge(self, factor: float) -> None:
        """Manual adjustment switches the mode to fixed and settles it: a number someone set by hand
        should not be silently overwritten by the tracker or by a calibration finishing."""
        self.mode = "fixed"
        self._settled = True
        self.threshold = self.clamp(max(0.01, round(self.threshold * factor, 2)))
        if self.reach_rate is not None:
            # Both lines together: a nudge is a judgement about the signal, not about one of two aims.
            self.reach_threshold = max(self.threshold,
                                       self.clamp(max(0.01, round(self.reach_threshold * factor, 2))))
        # Re-judged, not cleared. Nudging a threshold changes the comparison and not the measurement, so
        # every value already collected can be scored against the new number exactly — which is what makes
        # the card answer straight away instead of reading "waiting" for the next thirty seconds. Clearing
        # `hits`/`total` as well used to reset the *session* percentage on every nudge, so a hand
        # adjustment twenty minutes in threw away the whole session's summary.
        self.rescore()

    # --- contribution to the feedback signal ---------------------------------------------

    @property
    def spread(self) -> float:
        """Recent standard deviation — the scale on which this target's changes are meaningful.

        **Cached, and the reason is that it got slower the longer a session ran.** It reads a 100 s window
        out of a deque holding 200 s, so the copy is O(the deque) and the deque *fills over the first 200 s*
        — measured, 9.6 µs per call at 100 values against 60.3 at 4000, saturating a little over three
        minutes in. `score()` calls it, `combine()` calls `score()` for every target, and the feed and the
        draw loop both call `combine()`, so the cost lands several times per target per frame and grows
        under the operator exactly as the session gets going.

        A 100 s standard deviation cannot meaningfully change in 50 ms, so recomputing it on every read was
        never buying anything. It is recomputed once per second of data instead — keyed on `total` rather
        than on the clock, so it does not depend on how often anyone happens to ask.
        """
        if len(self._recent) < 20:
            return 0.0
        if self._spread_at != self.total:
            self._spread_at = self.total
            self._spread = float(np.std(self._window(100.0)))
        return self._spread

    def scale_for(self, scale_from: str = "sd", factor: float = 1.0) -> float:
        """How many microvolts from the threshold counts as a big move.

        This is what the feedback tab's "scale from" chooses, and it is the first step of the signal:
        microvolts in, a comparable 0…1 score out. `sd` follows the person — a channel that has been
        quiet needs a smaller move to count — while `threshold` and `absolute` are fixed, which is what
        you want if the scores have to mean the same thing across sessions.

        `factor` scales whatever the mode's own step is, and it exists because the step was **fixed and
        unstated**: one standard deviation, a quarter of the threshold, or five microvolts, with no way
        to say that a move of that size ought to count for more or less. Bigger means a bigger move is
        needed to shift the level by the same amount.

        A threshold alone gives a boolean crossing, and reward is that crossing. The soft score is a
        separate quantity, so both its shape and its steepness need justifying. See `SPAN_SIGMAS` for what the default
        works out to.
        """
        factor = max(0.05, factor)
        if scale_from == "threshold":
            return max(abs(self.threshold) * 0.25 * factor, 1e-9)
        if scale_from == "absolute":
            return ABSOLUTE_STEP * factor
        return (self.spread or max(abs(self.threshold) * 0.25, 1e-9)) * factor

    def score(self, scale_from: str = "sd", factor: float = 1.0) -> float:
        """How well this target is doing, in 0…1, with 0.5 exactly at the threshold.

        `wanted` is a cliff: one side is success and the other is failure, and a value a hair under
        the threshold looks identical to one at zero. That is fine for scoring time-on-target and
        wrong for driving continuous feedback, which needs to say *how far*. So this is a soft ramp
        whose steepness comes from `scale_for`, which makes scores from different bands and different
        channels comparable enough to combine.

        The ramp is a logistic of `delta / scale`, so at one step out it reads 73%, at two 88%, at three
        95%: it approaches the ends rather than reaching them, which is deliberate — an output pinned at
        its ceiling has stopped saying how well you are doing.
        """
        scale = self.scale_for(scale_from, factor)
        delta = self.value - self.threshold
        if not self.wants_above:
            delta = -delta
        return float(1.0 / (1.0 + np.exp(-np.clip(delta / scale, -30.0, 30.0))))

    def moving_average(self, points: int, span: int | None = None) -> list[float]:
        """Rolling mean of the plotted trend. Drawn on the graph, where the session mean is a number
        on the card — a mean over twenty minutes does not tell you which way the last minute went.

        `span` is how many points the caller is going to draw, and it is here because **this was the one
        thing in the amplitude screen that got slower the longer a session ran.** The graph shows a chosen
        window, but this averaged the whole `trend` and the caller then threw most of it away — and it did
        it in a Python loop, per target, per frame. Measured: 29 µs at 60 points, 111 at 300, 446 at 1200,
        which is where `TREND_POINTS` caps it. Three targets on the ten-minute window is 1.3 ms of the frame
        spent on points nobody sees, arriving gradually over the first ten minutes, which is exactly the
        shape of "it gets heavier as the session goes on".

        Both halves of the fix are needed and neither is sufficient. Taking the tail first bounds the work
        by what is drawn rather than by how long the session has run; doing the arithmetic in numpy is what
        makes the remaining work small. The tail keeps `points - 1` extra values of lead-in, so the first
        returned point still has its whole window behind it and the numbers are identical to averaging
        everything and slicing — not approximately, exactly.

        Without `span` it averages the whole trend, which is what the "session" window asks for and there is
        no honest way to shorten.
        """
        data = list(self.trend)
        if len(data) < 2:
            return []
        n = max(1, min(points, len(data)))
        if span is not None and span < len(data):
            data = data[-(span + n - 1):]
        values = np.asarray(data, dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        i = np.arange(len(values))
        lo = np.maximum(0, i - n + 1)
        out = (cumulative[i + 1] - cumulative[lo]) / (i + 1 - lo)
        return out[-span:].tolist() if span is not None else out.tolist()
