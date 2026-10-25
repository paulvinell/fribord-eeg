"""Feedback: two channels out of the targets, and the mechanisms that carry them.

**There are two, not one, and the screen has to say so.** Everything below used to be described as "one
signal", which is true of the continuous half and misleading about the rest — it made the `level from`
row look as though it governed every output, and left an operator unable to see why a click carried on
firing when the blend moved:

| channel | what it is | who follows it |
|---|---|---|
| **the level** | one 0…1 number, every target blended | sound, shadow — magnitude varying |
| **the condition** | every target on its wanted side, a plain AND | click, alert, the music's stop |

`Mechanism.reads` names which one, derived from the kind rather than stored, because a discrete
mechanism reading a blend is exactly the confusion this distinction exists to prevent. The fork is
printed under the SIGNAL panel with each channel's current value and its followers, so "why did that
not change" has an answer on the screen rather than in this docstring.

**The level** is a single 0…1 number computed from *all* targets at once — rewards and inhibits,
every channel. Nominating one target as the driver would mean the rest are measured, displayed, and
then ignored by the only output the person can perceive, so the screen and the sound would be training
different things. How the targets combine, what scale they are judged on, and how much the result is
smoothed are properties of the signal. They are chosen **once**, in one place.

**The mechanisms** each turn that number into something perceivable, and they are not
interchangeable — they differ in *kind*:

- **Sound** and **Shadow** are *continuous*: present the whole time, magnitude varying. Sound is the
  **system output volume**, so it modulates everything audible — the reward click, the music, whatever
  else is playing. Shadow is the subject's screen brightness.
- **Click** is *discrete*: a click at a **fixed** interval while every target is met, and nothing at
  all otherwise. It carries one bit — earned or not earned — and the absence of a click is the message.
  Its rate is a *presentation* choice (how often you want telling), never a reading of the signal:
  clicks that sped up and slowed down would be a continuous mechanism in disguise, and you could not
  tell a slow reward rate from a stopped one.

The kind is not a setting; it is what the mechanism *is*, and it determines the mapping. A continuous
mechanism has no on/off condition — being present is the whole idea, and a shadow that appeared only
while the targets were met would be a different mechanism (a flash) wearing a shadow's name. A discrete
one is the opposite: its whole content is the condition.

**The combining rule shapes the continuous mechanisms and does not touch the discrete one.** The
signal sits at 0.5 when every target is exactly on its threshold, since that is where each target's
score is 0.5, and above that:

| rule | signal ≥ 0.5 means |
|---|---|
| `weakest` | *every* target is better than its threshold — a strict AND |
| `weighted` | the 80/20 mix is favourable, so a strong reward can carry a failing inhibit |
| `scaled` | rewards are up *and* the inhibits are not dragging it down |
| `either` | the *best* reward is up and the inhibits are not dragging it down — an OR |

For sound and shadow that choice shapes the curve — how brightness responds as the targets move. The
click ignores it entirely and asks the targets directly: every one met, or not. That is deliberate. A
level is a blend and can sit at 60% with one target failing, which is the right thing for a continuous
display and the wrong thing for an event that means "earned".

**`either` is the one exception, and it has to be.** Under the other three rules the rewards are
conditions to satisfy together, so "earned" is unambiguously all of them. Under `either` they are
*alternatives* — two states you can be in one of, which is what alpha-theta training is — and an AND over
alternatives is unsatisfiable in principle. The level therefore takes the
**best** reward and ignores the other, which is what `max` means and what the row should say. Distinguish
that from **which reward the tone marks**: a separate preference the protocol sets
(`TrainingSession.tone_bias`), theta-first on the screen that has one. The two sound alike and are not —
one shapes a number, the other decides a pitch — and describing the first in the second's language is how
the row became unreadable. So `on_target` follows the rule there: one reward met, every
inhibit met. That is the only place the combining rule reaches a discrete mechanism, and the alternative
was a second setting saying the same thing in a place where it could disagree with this one.

None of the three rules is a switch — `scaled` in particular is a *product*, not a gate, which is what
it used to be called. On a continuous mechanism it means brightness follows the rewards scaled down by
how badly the inhibits are doing: a firing inhibit dims the screen even with a strong reward band, and
it sinks toward the floor rather than going out.

Continuous mechanisms have a floor and a ceiling. Output that reaches zero stops informing — the person
cannot tell "doing badly" from "it broke" — and output at full is unpleasant with nowhere left to go.
The click has neither: it is on at its interval, or it is off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Iterable, Protocol

# The two channels out of the targets. Named constants because they are printed, compared and keyed on,
# and three spellings of "condition" in three files is how a distinction stops being one.
HOLD_FRACTION = 0.70
"""How much of the hold window the condition has to have been met for the mark to double.

**Not a continuity requirement, and measuring is what settled it.** On a real two-channel export the raw
condition flickers at about 3 Hz — 186 on-target runs in 60 s, median **0.15 s**, longest **0.85 s**, not one
reaching a full second. So "met continuously for three seconds" can never fire, at any threshold, and the
first version of this required exactly that and was silently dead. The same flicker is why the click's
interval phase survives a gap: the interval hides it and a hold criterion is exposed to it directly. A
2 s sample-and-hold ahead of the comparison would also hide it.

**The figure is ours, and the second version got it wrong by borrowing a number of the right shape from the
wrong quantity.** A figure of **81.0** is in use elsewhere for this position in this
protocol — but it is the auto-threshold aim for *one band's amplitude*, where this is a duty cycle of an AND
over three targets. Those cannot share a constant. Measured on the real export, 3 s windows of our own
condition: p50 **57%**, p90 68%, p95 70%, p99 75%, max **78%** — so 81% is unreachable by construction, since
the condition is met about 56% of the time when its three aims (25%, 55%, 85% met) are being hit as designed.

0.70 sits at about the 95th percentile of what this arrangement produces, so a doubled mark lands on roughly
7% of marks — a few times a minute at best, which is what makes it a landmark rather than a texture. It is a
module constant rather than a setting because it is a property of the arrangement, not of the room."""

STATE = "state"
"""A third quantity, and the only one a *protocol* has to supply: where between two named states the trainee
is. Not a score — it has no better end — which is why `state_ends` names what the two ends *mean* and why
nothing here can compute it. See `TrainingSession.state_balance`."""

LEVEL = "level"
CONDITION = "condition"
READS = {LEVEL: "follows the level", CONDITION: "fires when every target is met"}
"""What a mechanism reads, on the chip beside its switch.

"Fires on the condition" named a variable, not a thing the operator can check. `CONDITION` is a good name
*in the code*, where it has to be distinguishable from `LEVEL` in three files — and a terrible one on screen,
where nobody has read those files and nothing else on the panel defines it. What it means is a plain AND over
every target the protocol put on that channel, so the chip says that instead."""

# How far from its threshold a target has to be before its score moves much. This is the first step of
# the pipeline: microvolts in, a 0…1 score out. It feeds `COMBINES`, which then reduces the scores to
# one number, and saying so in that order is the only way the two settings make sense together.
NORMALISATIONS = {
    "sd": "recent variability",
    "threshold": "a quarter of the threshold",
    "absolute": "a fixed 5 µV step",
}
NORMALISATION_HELP = (
    "The row named scale from is one step of turning microvolts into a signal. It sets how far a target "
    "must move from its threshold before its score changes much.\n\n"
    "recent variability follows the person. A channel that has been quiet needs a smaller move to count. "
    "The other two settings are fixed. Use a fixed setting when scores must mean the same thing across "
    "sessions.\n\n"
    "A score is a soft ramp through 50% at the threshold. One step out reads 73%, two steps read 88%, "
    "three steps read 95%. The score approaches 0% and 100% but never reaches them. This is deliberate. "
    "An output pinned at its ceiling no longer shows how well you are doing.\n\n"
    "The size of the step is fixed here. It is adjusted once, at the end of the chain. The sensitivity "
    "setting controls how much of an output's travel an ordinary swing uses. Sensitivity belongs to the "
    "finished level, not to each target's own ramp."
)
ABSOLUTE_STEP = 5.0        # µV, the step `absolute` uses
SPAN_SIGMAS = 2.94
"""Where the logistic reads 95%, in steps: `ln(19) ≈ 2.94`. So a target's score spans 5–95% across about
±3 σ of its own recent spread. Fixed: it is the *shape* of one target's ramp, and how much output a real
excursion produces is `FeedbackSettings.sensitivity`, one dial at the end of the chain."""

SENSITIVITY_RANGE = (0.125, 8.0)
"""What the one dial may be set to. The same bounds as `TrainingSession.state_sensitivity_range`, because
it is the same kind of quantity asked of a different signal, and two different ranges for "how hard does
this drive" would be a distinction with nothing behind it.

A range rather than a ladder of offered values, for the reason that field already gives: this is
continuous, and 1.4 is a legitimate answer to "a bit more than it does now" that no ladder short enough to
fit a panel contains."""

SENSITIVITY_HEADROOM = 2.0
"""How much wider than the calibrated swing the output's full travel is, at a sensitivity of 1.

This is what fixes the *unit* of the dial, so it decides what 1 means and therefore what everybody types.
At 1 the swing measured during calibration lands on the middle half of the output, 0.25 to 0.75, which is
the sentence the panel prints. Measured through this code on a real session — the level's 5th to 95th
percentile after the first 90 s, the system volume that produces on a 30–70 range, and how often it sat
against an end:

| sensitivity | level p5–p95 | its sd | volume | at an end |
|---|---|---|---|---|
| — (before this existed) | 0.58–0.81 | 0.068 | 53–62 | 0% |
| 0.5 | 0.45–0.64 | 0.058 | 48–56 | 0% |
| 1 | 0.40–0.79 | 0.117 | 46–61 | 0% |
| 2 | 0.31–0.90 | 0.179 | 42–66 | 0% |

So 1 is a session that uses about half of each output's travel and does not pin, which is the default a
person should not have to think about, and 2 is lively. Headroom rather than mapping the calibrated swing
straight onto the full range: a p5–p95 is not the widest the session will go, and without it the same
recording spends 17% of its time against an end, where an output that sits at a limit has stopped saying
how well anyone is doing."""

SHAPE_KNEE = 0.2
"""How much of each end of `shape`'s output is a soft knee rather than a straight line.

The middle 60% of the travel is linear and identical to what a hard clip gave, so ordinary operation is
untouched — at the default sensitivity a calibrated swing lives on 0.25–0.75 and never meets a knee. The
knees replace the clip at the ends, where a level past the calibrated range used to become a constant:
each bends away exponentially with its slope continuous at the join, reaching the end only in the limit.
At the point where the clip used to hit 1.0, the knee reads 0.926 — the price of the top 7% of travel
buys back a gradient over everything beyond it."""

# How the per-target scores become one number.
#
# An inhibit can be a veto rather than a weight, so that it never averages into the reward. `mean`
# below is the one that can hide a firing inhibit behind a strong reward.
COMBINES = {
    "weighted": "rewards and inhibits in a fixed ratio",
    "weakest": "whichever target is doing worst",
    "scaled": "rewards averaged, then scaled down by the inhibits",
    # `max` over the rewards where `scaled` takes their mean. Two things sound alike here and are not:
    # **the level** takes the best-scoring reward, while **which reward the tone marks** is a separate
    # protocol-set preference (`tone_bias`, theta-first on the screen that has one).
    "either": "whichever reward is doing best, scaled down by the inhibits",
}
# What half way up the level means, per rule — printed beside the level on the fork, because "50%" is
# otherwise a number with no referent. `weakest` is the only one where it coincides with the condition.
COMBINE_DETAIL = {
    "weighted": "50% means the mix is balanced",
    "weakest": "50% means every target is met",
    "scaled": "50% means the rewards are up and the inhibits are not lowering the level",
    "either": "50% means one reward is met and the inhibits are not lowering the level",
}
# The condition's two forms, as a row of its own. **Not a second setting**: which one is in force is
# decided by the rule above it, since `either` makes the rewards alternatives and an AND over alternatives
# is unsatisfiable. It is shown as a locked choice rather than as a sentence because the level's rule had a
# row of pills and the condition had a footnote, so the screen said there was one mixing strategy and a
# remark — and the remark is the rule that governs every discrete output.
CONDITIONS = {
    "all": "every target",
    "either": "one reward",
}
CONDITION_DETAIL = {
    "all": "every target must be on its wanted side",
    "either": "one reward must be met, and every inhibit must be met",
}

COMBINE_HELP = (
    "This sets how every target's score combines into one level, from 0 to 100%. Every continuous "
    "output follows that level.\n\n"
    "weighted — the rewards average together, the inhibits average together, and the two combine at the "
    "ratio you set. The rewards lead, and the inhibits pull on the level without controlling it. With "
    "only one kind of target, that group is the whole level.\n\n"
    "weakest — the worst target decides the level. The level is 50% when every target is at or better "
    "than its own threshold.\n\n"
    "scaled — the rewards average together, then the inhibits scale that average down. An inhibit at "
    "its threshold halves the level. A firing inhibit cuts the level further, even when the rewards are "
    "high.\n\n"
    "either — the best reward decides the level, then the inhibits scale it down. Use this when the "
    "rewards are alternatives instead of conditions to meet together. Averaging them asks for both at "
    "once, and shows half the level when only one is true. Under either, a discrete output fires on one "
    "reward instead of on all of them.\n\n"
    "Under weighted, weakest and scaled, a discrete mechanism does not follow the level. It checks the "
    "targets directly, because the level can sit at 60% with one target failing. Under either, a "
    "discrete mechanism fires on one reward, as described above."
)
COMBINE_SHARES = {
    "weighted": "each target's score: rewards and inhibits average separately, then mix",
    "weakest": "each target's score: the lowest score is the level",
    "scaled": "each target's score: rewards average together, then the inhibits scale it down",
    "either": "each target's score: the best reward, then the inhibits scale it down",
}
DEFAULT_REWARD_SHARE = 0.80      # of the signal; the inhibits get the rest
SHADOW_MECHANISMS = {
    "brightness": "the whole screen dims and lifts",
    "overlay": "not implemented — a shade drawn over the image",
}
SUPPORTED_SHADOW = ("brightness",)

# --- the calibrated defaults, in one place -------------------------------------------------------
#
# **Every one of these is a number set by ear or by eye in a room, and the room does not change when the
# protocol does.** They were spread across four dataclasses and three screens, each picked on its own and
# none of them agreeing: the click was 15 here and 25 inside alpha-theta, the alert 25 for an argument the
# click's docstring contradicted, the master volume ran 60–100 on most screens and 15–70 on the coupling
# one. So the same room needed re-balancing on every screen, and moving between protocols mid-evening
# changed how loud the machine was for reasons no protocol had asked for.
#
# What is *not* standardised is the deliberate override: a screen that has a written reason to differ still
# writes it, and alpha-theta's quieter music is the case that proves the rule — eyes closed, an hour long,
# and the music is the state rather than the reward. The rule is that differing needs a sentence.
CLICK_VOLUME = 25.0
"""Every discrete mark, 0-100. A click and an alert are both one event through one path in one room."""

MUSIC_VOLUME = 60.0
"""Continuous playback, 0-100. Above the marks, which sit under it — see `Click.volume`."""

SOUND_RANGE = (30.0, 70.0)
"""The master volume's ends, which is what the Sound mechanism sweeps between.

Neither end is at a limit, and both edges of that are the point. The floor is audible rather than silent,
because the bottom of a proportional channel means "doing least well", not "the machine is off" — and a
mechanism that goes quiet is indistinguishable from one that has stopped working. The ceiling is well under
100 because the top of the range arrives repeatedly all session and should not be the loudest the hardware
goes; it also leaves room above for anything the operator plays deliberately."""

SHADOW_RANGE = (40.0, 100.0)
"""The screen brightness ends. The floor is higher than a display can technically reach for the same reason
the volume's is: a screen dimmed to near-black is not feedback, it is a screen somebody will assume is
broken. The ceiling stays at full, because the reward here *is* the room brightening."""


class Scored(Protocol):
    """What combining needs from a target: a soft 0…1 score and a **role**.

    Role and not direction, and that is the whole of this change. Which group a target belongs to is a
    statement about what it drives; which way it points is a statement about where its threshold is. They
    were one field, so a band trained downward could only ever be an inhibit — see `targets.ROLES`.
    """

    role: str

    def score(self, scale_from: str = "sd", factor: float = 1.0) -> float: ...


def combine(targets: Iterable[Scored], mode: str = "weighted", scale_from: str = "sd",
            reward_share: float = DEFAULT_REWARD_SHARE, factor: float = 1.0) -> float:
    """Reduce every target to the single 0…1 quantity the mechanisms carry.

    `weighted` is the default: average the rewards, average the inhibits, mix at `reward_share`. The
    ratio is the point — a plain average over all targets gives each target an accidental weight that
    depends on how many of each kind you happen to have, so adding a second inhibit quietly doubles how
    much the inhibits matter. Stating the split fixes that, and 80/20 says what training usually means:
    the signal follows the band being trained, and the inhibits pull on it without taking it over.

    **The split is by role**, which is a field, not by `direction == "increase"`, which is what it used to
    be. A protocol that trains its subject band *downward* — Ros and Lanius' alpha-down is the standing
    example — had its subject in the inhibit group and got 20% of the mix for the one thing the session was
    about, with any artifact inhibit added beside it halving that again.

    Each group is averaged *within itself* first, so the ratio holds however many targets are in each.
    With only one kind present that group is the whole signal rather than being capped at its share —
    otherwise a reward-only setup could never exceed 80%.

    `either` is `scaled` with a **max** over the rewards instead of a mean, for rewards that are
    alternatives rather than simultaneous conditions. Averaging two alternatives asks for both at once and
    reports half the signal when you are firmly in one of them, which is exactly backwards; the inhibits
    still scale the result, because an inhibit is a veto whichever reward is earning.

    `scaled` is the traditional arrangement written continuously, and it is a product, not a gate: the
    rewards set the level and the inhibits scale it down, so artifact on any inhibit pulls the signal
    towards its floor however good the reward looks. Its inhibit factors combine as a geometric mean
    rather than a bare product, so it does not collapse simply for having several inhibits — three
    inhibits at their thresholds read the same as one. `weakest` is the same instinct without
    privileging direction, and is the one where "half signal" means exactly "every target met".
    """
    # No fallback to the direction, deliberately. `Scored` is a structural protocol, so one would be easy
    # to write — and it would say the opposite of what a real `Target` says, since an ordinary inhibit is
    # now `increase` and a `decrease` target is a reward trained downward. A caller without a role is a
    # reward, which is the field's own default.
    scores = [(getattr(x, "role", "reward"), x.score(scale_from, factor)) for x in targets]
    if not scores:
        return 0.0
    if mode == "weakest":
        return min(s for _r, s in scores)
    up = [s for r, s in scores if r == "reward"]
    down = [s for r, s in scores if r != "reward"]
    if mode in ("scaled", "either"):
        if not up:
            level = 1.0
        else:
            level = max(up) if mode == "either" else sum(up) / len(up)
        if down:
            product = 1.0
            for s in down:
                product *= s
            level *= product ** (1.0 / len(down))
        return level
    if not up:
        return sum(down) / len(down)
    if not down:
        return sum(up) / len(up)
    share = min(1.0, max(0.0, reward_share))
    return share * (sum(up) / len(up)) + (1.0 - share) * (sum(down) / len(down))


@dataclass
class Ramp:
    """A mechanism's output range. Sorted on read, so entering it backwards is not a trap, and
    clamped, so a metric excursion cannot drive brightness past 100."""

    low: float
    high: float

    @property
    def span(self) -> tuple[float, float]:
        return (self.low, self.high) if self.low <= self.high else (self.high, self.low)

    def at(self, level: float) -> float:
        lo, hi = self.span
        return lo + (hi - lo) * min(1.0, max(0.0, level))


@dataclass
class Mechanism:
    """One way of carrying the signal. The kind is a class fact, not a field: it is what the
    mechanism is, and no setting should be able to turn a shadow into a flash."""

    kind: ClassVar[str] = "continuous"      # continuous | discrete
    label: ClassVar[str] = "mechanism"
    quantity: ClassVar[str] = "level"       # what the range is a range *of*
    unit: ClassVar[str] = ""

    enabled: bool = False
    low: float = 0.0
    high: float = 100.0
    invert: bool = False
    """Doing better means *less* of this output. The Quiet Reward arrangement in Cross-frequency coupling:
    "when the trainee is slightly more awake, adaptive and/or balanced and graceful then the sounds of the
    software stop … listen to the silence and increase its duration".

    A flag rather than entering the range backwards, because `Ramp` deliberately *sorts* its ends so
    that a backwards range is not a trap — which means swapping them cannot express this. Both rules are
    worth keeping and they need separate mechanisms.
    """

    @property
    def reads(self) -> str:
        """Which of the two channels this mechanism follows — `LEVEL` or `CONDITION`.

        Derived from the kind rather than stored, because it *is* the kind seen from the other end: a
        continuous mechanism has a magnitude and therefore needs a number, a discrete one has only the
        question of whether to fire. Storing it would allow the two to disagree, and a discrete mechanism
        reading a blend is the specific confusion the distinction exists to prevent.
        """
        return LEVEL if self.kind == "continuous" else CONDITION

    @property
    def ramp(self) -> Ramp:
        return Ramp(self.low, self.high)

    def ends(self) -> tuple[float, float]:
        return self.low, self.high

    def map(self, level: float) -> float:
        """The 0…1 position this mechanism is at, which is the signal or its complement.

        Note what inversion does *not* change: the floor is still a floor. Under Quiet Reward the quiet
        end is genuine silence — that is the reward — so it is the *ceiling* that has to stay bounded, and
        `Ramp` clamping both ends is what keeps "doing badly" distinguishable from "it broke" either way
        round.
        """
        level = min(1.0, max(0.0, level))
        return 1.0 - level if self.invert else level

    def output(self, level: float, on_target: bool = False) -> float:
        """Continuous: always present, magnitude from the signal. There is no off, and `on_target` is
        ignored — only a discrete mechanism asks that question."""
        return round(self.ramp.at(self.map(level)), 2)

    def describe_output(self, value: float, level: float = 0.0) -> str:
        """`level` is accepted for the one mechanism whose *description* needs more than its own
        output value; the continuous ones ignore it."""
        return f"{value:g} {self.unit}".strip() if self.unit else f"{self.label} {value:g}"


@dataclass
class Sound(Mechanism):
    """The system output volume, following the signal.

    Not a tone of our own. Sound feedback means "make what you are hearing follow the signal", and the
    thing that does that for *everything* — click, music, anything — is the master volume. An earlier
    version gave the click its own signal-following volume instead, which was the wrong shape twice: it
    made a discrete event carry an amount, and it left every other sound unaffected.
    """

    label: ClassVar[str] = "sound"
    quantity: ClassVar[str] = "system volume"

    enabled: bool = True
    low: float = SOUND_RANGE[0]
    high: float = SOUND_RANGE[1]

    def describe_output(self, value: float, level: float = 0.0) -> str:
        return f"volume {value:.0f}"


@dataclass
class Shadow(Mechanism):
    """The subject's screen, dimming and lifting with the signal."""

    label: ClassVar[str] = "shadow"
    quantity: ClassVar[str] = "brightness"
    unit: ClassVar[str] = "%"

    enabled: bool = False
    """**Off by default, and this is the one mechanism where the default is a safety question.**

    It was on, and only alpha-theta and the infra-low screen turned it off, so the coupling screen opened
    with the whole display swinging 20-100% behind a level that flickers at about 3 Hz. `shadow.MAX_SLEW`
    now makes that impossible whatever the signal does, and the default still belongs at off: taking over
    the operator's display is the most invasive thing this app does, and a mechanism that invasive should be
    asked for rather than assumed. Every other continuous output is off by default for weaker reasons than
    this one.

    A screen that wants it can set it, as `at_view` and `ilf_view` already set it False — those lines are now
    redundant and kept, because they carry each screen's own reason for not wanting it, which is not this
    reason.

    **And the coupling and synchrony screens now set it True, conditionally.** Both feed back to somebody
    looking at a screen, so a brightness that follows the signal is wanted there — but "on by default" and
    "safe" only coincide when there is a display to dim that is not the operator's. `shadow.subject_screen()`
    is that test, and it also supplies the screen number, so the two screens ask it for both at once. With one
    display attached the default is still off, and the switch is still there to be thrown by hand: an
    operator who wants their own screen dimmed can say so, which is the difference between a choice and a
    surprise."""

    low: float = SHADOW_RANGE[0]
    high: float = SHADOW_RANGE[1]
    smoothing_seconds: float = 0.0
    """Extra smoothing on the brightness, on top of the fixed `shadow.APPROACH_SECONDS`, in seconds.

    **Zero by default on every protocol**, so nothing is smoothed unless someone asks for it. It exists
    because how fast the level moves is a property of the protocol rather than of the display — the
    amplitude screen's level moves at the pace of an envelope, while the coupling and infra-low screens
    combine a reward with nine or twelve threshold crossings — but a default above zero would be adding lag
    to every screen to pre-empt a problem no screen is currently known to have.

    It was briefly defaulted to 0.4 on those two screens, chasing a flicker that turned out not to be ours:
    a second program was driving the same backlight through the same private API. Nothing in this file could
    have fixed that, and the numbers picked to fix it had no evidence behind them once the cause was known.

    What a value buys, since the arithmetic is the only honest way to choose one: a one-pole passes
    `1/sqrt(1 + (2πf·tau)²)`, so against a 3 Hz flicker **0.1 s leaves 47%**, 0.2 s leaves 26%, 0.4 s leaves
    13% and 0.8 s leaves 6.6%. The control's note prints that figure for whatever is set, so the choice is
    made against the number rather than by feel.

    A slew-rate ceiling in `shadow.py` was tried first and was the wrong instrument: rate-limiting a square
    wave produces a triangle wave at *the same frequency*, so it shrinks the flicker without removing it.
    A low-pass attenuates the frequency itself."""

    mechanism: str = "brightness"
    screen: int = 1
    follows: str = LEVEL
    """`LEVEL` — how well — or `STATE`, where the protocol supplies one.

    Not a violation of `reads` being derived from `kind`: that rule prevents a *discrete* mechanism reading a
    blend, and both of these are continuous. This is a third quantity rather than the other channel — the
    same one the crossfade follows, `TrainingSession.state_balance` — and it is a legitimate thing to point
    a brightness at: the visuals darken as the dominant frequency moves toward
    theta, which is our music's logic applied to the display rather than a score being shown twice.

    Only offered where a protocol *has* a state, for the same reason a crossfade is: with no `state_balance`
    it would render as a brightness frozen at one end."""

    @property
    def supported(self) -> bool:
        return self.mechanism in SUPPORTED_SHADOW

    def describe_output(self, value: float, level: float = 0.0) -> str:
        return f"{value:.0f}% bright"


@dataclass
class Click(Mechanism):
    """One click every `interval_seconds`, while every target is met. Nothing otherwise.

    The interval is how often you want telling, not a reading of the signal. A click train that sped up
    and slowed down would be a continuous mechanism in disguise, and worse, it would be ambiguous:
    a slow rate and a stopped one are the same experience until you have waited long enough to be sure.
    Fixed-rate-or-silent has no such ambiguity — a click means the last moment was earned, and its
    absence means it was not.

    Below about half a second the clicks stop reading as separate events; a few seconds apart is the
    usual choice, so the default is two.
    """

    kind: ClassVar[str] = "discrete"
    quantity: ClassVar[str] = "interval"
    unit: ClassVar[str] = "s"

    enabled: bool = True
    interval_seconds: float = 2.0
    hold_seconds: float = 0.0
    """How long the condition must have held *continuously* before the mark changes character. 0 never
    changes it, which is what every screen did before this existed.

    **Two marks on one channel, not two mechanisms.** A momentary reward event and a
    three-second sustained one are two real messages: marking arrival and marking persistence. But the sustained one *replaces* the
    momentary one rather than sounding over it — two independent streams would overlap and muddy both, and
    this channel's whole design rests on being one bit at a fixed rate. So there is one `enabled`, one
    `volume` and one `interval_seconds`, and what changes after `hold_seconds` is the *sound*: one mark means
    you got there, two means you have held it. See `cues.held` for why rhythm carries it and not pitch.

    An earlier version of this was a `dwell_seconds` that *delayed* the first sound instead, which made the
    two messages mutually exclusive — you could have one or the other and never the pair, which is the thing
    the pairing exists to say."""
    sound: str = "tick"             # which click character; see cues.CLICK_PROFILES
    volume: float = CLICK_VOLUME
    """Its own gain, 0-100, **fixed** — exactly like `Music.volume` and for the same reason: you balance the
    click against the music once and let the Sound mechanism move the master over both.

    Low because the scale is linear amplitude, not loudness: `afplay -v` takes 0…1 as a straight multiplier,
    so 40 is −8 dB and still loud for a few milliseconds of near-full-scale noise, and 25 is −12 dB. Well
    under music's 60, because a click that fires every couple of seconds all session sits *under* the music.

    Distinguish this from the thing that was removed. A click volume that *followed the signal* was wrong,
    because it put an amount into the one channel whose entire content is that something happened. A volume
    the operator sets is not information at all, it is calibration — and without it the click plays at the
    system volume, which is to say as loud as the machine goes.

    The number itself is `CLICK_VOLUME`, one place, because it was three: 15 here, 25 on the alert and 25
    again written out inside alpha-theta, each arrived at by ear on its own screen and none of them agreeing
    with the others. See that constant for why a mark is a mark on every protocol."""
    keyed: bool = False
    """Take the sound from the frequency of the target that is *earning*, where the protocol has one — the
    mirror of `Alert.keyed`, and useful for the same reason in the opposite direction. Off by default,
    because with a single condition there is nothing for the pitch to distinguish: it is the protocols whose
    rewards are alternatives that need the click to say which one fired."""

    @property
    def label(self) -> str:
        """**A keyed reward is called a tone, because that is what it plays.**

        Keyed, the sound is `cues.reward_tone` — a pitched 160 ms tone identifying which reward earned it.
        Calling that a click on screen is the kind of copy that makes an interface untrustworthy: the tab
        said Click, the readout said "a click every 2 s", and what came out of the speaker was a tone.
        The mechanism is the same one either way, which is why this is a label and not a second class:
        one bit, at a fixed interval, on the condition.
        """
        return "tone" if self.keyed else "click"

    @property
    def verb(self) -> str:
        return "sound" if self.keyed else "click"

    @property
    def rate_hz(self) -> float:
        return 1.0 / max(0.05, self.interval_seconds)

    def output(self, level: float, on_target: bool = False) -> float:
        """The interval it is clicking at, or 0 for silence. Whether it clicks at all is the condition,
        never the level — see the module docstring."""
        return round(max(0.05, self.interval_seconds), 2) if on_target else 0.0

    def describe_output(self, value: float, level: float = 0.0) -> str:
        """The interval only. `volume` is a setting rather than news, so it is not printed here — and it is
        deliberately not a *reading*: a loudness that followed the signal would put an amount into the one
        channel whose entire content is that something happened."""
        return f"not {self.verb}ing" if value == 0 else f"a {self.label} every {value:g} s"


@dataclass
class Alert(Mechanism):
    """A sound when a target goes the wrong side of its threshold. **This is CFC's.**

    When an inhibit bin is exceeded there are two choices: the video pauses, or an audio alert sounds. The
    audio one is better, because pausing makes the video jerky and does not let the trainee tell the event
    apart. It applies to the rewards too: under Quiet Reward the trainee also hears an alert when
    synchrony between the two electrodes falls at either of the reward frequencies.

    So it is **not** artifact rejection and not a diagnostic. It is the thing that makes the 5-15% total
    across the bins meaningful: that figure is the fraction of the session you expect to *hear* something,
    and under Quiet Reward these alerts are the only sound there is.

    A distinct mechanism from `Click` rather than a polarity setting on it, because the two answer
    different questions and can be wanted at once: the click says a moment was earned, the alert says a
    specific condition broke.
    """

    kind: ClassVar[str] = "discrete"
    label: ClassVar[str] = "alert"
    quantity: ClassVar[str] = "interval"
    unit: ClassVar[str] = "s"

    interval_seconds: float = 1.5
    """No faster than this. A value resting on its threshold flickers, and an ungated alert on flicker is
    a buzz rather than an event — the same failure the click's phase discipline exists for."""
    volume: float = CLICK_VOLUME
    """Its own fixed gain, 0-100, like the click's, and **the same number**. It used to be higher, on the
    argument that an alert is the exception and has to be noticed where a click repeats all session — but
    both are single marks played through the same path at the same moment in the same room, so what
    distinguishes them is the sound and the rhythm, not the gain. See `CLICK_VOLUME`."""
    keyed: bool = True
    """Take the sound from the failing target's frequency, so a run of alerts says *where* in the spectrum
    the trouble is. With twelve bins, "something is over" is not actionable and "that one is" is. Deriving
    the sound from the band means a bin added
    later already sounds like where it sits. Off falls back to `sound` for every target."""
    sound: str = "clack"

    def output(self, level: float, on_target: bool = False) -> float:
        """Sounding, or silent — the opposite condition to the click's."""
        return 0.0 if on_target else round(max(0.05, self.interval_seconds), 2)

    def describe_output(self, value: float, level: float = 0.0) -> str:
        return "quiet" if value == 0 else f"alerting every {value:g} s"


@dataclass
class Music:
    """Generated noise, a crossfade between two tilts, a Shepard tone, a backing track, or the EEG as sound.

    **Not a mechanism in the sense above**, and the reason is worth stating precisely because it changed. It
    used to be that music carried no signal; three of the five sources now do — `crossfade` follows the
    protocol's *state*, `shepard` follows the *level*, and `eeg` carries the waveform itself. What makes this
    not a mechanism is that it has no `Ramp` and no `reads`: a mechanism maps the level through a range an
    operator sets, whereas each source here has its own fixed relationship to the signal, chosen along with
    the source. That is why `MusicOutput.apply` takes `balance` and `level` as separate arguments rather than
    one — collapsing them would make a screen with both drive them from the same quantity.

    It is here because it is audio the session owns and must stop when the session stops.

    Noise is the default source because it is what this kind of training traditionally uses, and it is the
    one source that needs nothing chosen before it will play.

    Its volume is a software gain on this playback alone, deliberately independent of the system volume:
    you set the backing track once, and the Sound mechanism moves the master over the top of it.
    """

    enabled: bool = False
    source: str = "noise"               # noise | crossfade | file | eeg
    colour: str = "pink"                # for the noise source; see music.NOISE_COLOURS
    second_colour: str = "brown"
    """The other end of a `crossfade`. Two tilts of generated noise with the balance between them
    driven by the protocol, so the music's *character* carries the signal while its loudness does not — which
    is how the two soundscapes of alpha-theta training work. Ignored by every other source."""
    path: str = ""
    pitch_hz: float = 0.0
    second_pitch_hz: float = 0.0
    """Where each end of the music **sits**, in Hz, as opposed to what it is made of. 0 for neither, which is
    the default and leaves the music exactly as it was.

    A tilt is a slope and the ear reads it as texture; a resonance is what makes filtered noise read as
    *pitched*. So a non-zero value replaces that end's flat high-pass with a band-pass an octave wide, and the
    two ends then differ along two axes instead of one — which matters because the entire job of a crossfade
    music is that the trainee can tell which end they are at with the eyes closed.

    A higher pitched reward pink noise plays when alpha is dominant, and a lower pitched one when theta
    is dominant. Ignored by every
    source but `noise` and `crossfade`."""
    volume: float = MUSIC_VOLUME
    low_hz: float = 8.0                 # the band that becomes audible; below is drift
    high_hz: float = 16.0               # above is mains and muscle
    """The band the `eeg` source plays, and **an octave of it by default rather than the whole spectrum**.
    It was 2–38 Hz, which was right while the band only had to be *audible*: a heterodyne renders whatever
    it is given as roughness, so width cost nothing. Now the band's own frequency is the pitch, and a
    frequency estimate over a 19:1 band is one amplitude-weighted average of everything in it — with alpha
    and beta both present it sits between them and wobbles, which is a sound with no rhythm in it rather
    than two voices. 8–16 Hz is one octave around the rhythm most often trained, and it maps to the
    default 220–440 as an exact transposition. See `music.eeg_pitch`."""
    # There is deliberately no loudness field for the `eeg` source. Its predecessor, `full_at_uv`, asked
    # how many microvolts of the band count as full loudness — a number that varies per person, per site
    # and per montage, that nothing on the screen helped anyone know, and whose fallback (the raw trace's
    # display gain, ±50 or ±100 µV against a band producing 2–5) parked the voice at its quiet floor for
    # whole sessions. The voice now hears the band against its own learned range — see
    # `music.Sonifier._learn_range` — and the learned pair is printed in the player's state line.
    glide_seconds: float = 0.0
    """How long the Shepard comb takes to reach a new level, in seconds. **0 takes
    `music.SHEPARD_GLIDE_SECONDS`**, which is 1.5. Ignored by every other source.

    A protocol setting rather than a constant, for the reason `Shadow.smoothing_seconds` is one: how fast the
    level moves is a property of the protocol, and this glide has to be read against it. The 1.5 s default was
    chosen against a level updating twenty times a second, where the glide is what stops the tone reading as
    vibrato instead of as travel.

    **On a level that steps rather than flows it does the opposite job.** ILF Synchrony's level moves once
    every two seconds — a sample-and-hold — and against that input a 1.5 s glide covers only
    `1 − e^(−2/1.5)` = **74%** of each step before the next arrives, so the pitch never gets there: it is
    permanently a quarter of a step behind, and the 1.5 s is pure lag on a measurement that already carries
    seconds of filter delay.

    It cannot go to zero. A jump in a Shepard comb is heard as a *different tone* rather than as movement,
    which is exactly the percept the instrument exists to produce, so 30 jumps a minute would destroy it. The
    value wants to be short against the tread instead — see `ilf_view.GLIDE_SECONDS`."""
    travel_octaves: float = 0.0
    """How far the comb slides across the whole range, in octaves. **0 takes `music.SHEPARD_OCTAVES`**, which
    is 6. Ignored by every other source.

    **This is where "I want more movement" belongs**, and it is the reason the state's own mapping is not a
    sensitivity control. Every continuous mechanism owns how far its output travels — the shadow has its
    brightness span, the bounded `tone` has its two frequencies — and this is that same setting for the one
    mechanism that had it welded shut as a module constant. A second gain upstream of these would multiply
    into the same effect with no way to tell which of the two to reach for.

    Bounded above at half the wheel, which is `SHEPARD_OCTAVES` and not a matter of taste: the wheel is 12
    octaves, so at more than 6 the top of the range starts sounding like the bottom, which is the failure a
    wheel with no ceiling has in place of saturation. Below it is preference — a slow protocol with a wide
    swing may want the whole half, and a screen whose state crosses often may want two octaves so the room
    is not a siren."""
    tone_base_hz: float = 220.0
    tone_top_hz: float = 660.0
    """The ends of the `tone` source's pitch range.

    It is a *bounded* pitch where `shepard` is an unbounded one — the two are different instruments rather
    than settings of each other. A bounded pitch carries a position and needs no reference point: you hear
    which way it moved without remembering where it was, which no volume can do. An unbounded one carries a
    direction of travel and suits a signal with no natural ends. 220 to 660 Hz is a range the ear resolves
    finely across its whole span."""
    eeg_pitch_lo_hz: float = 220.0
    eeg_pitch_hi_hz: float = 440.0
    """The ends of the `eeg` source's pitch range: what the bottom of its band sounds at, and the top.

    **A separate pair from `tone_base_hz`/`tone_top_hz`, and the defaults are the reason.** They look like
    one setting — two ends of a pitch — but what is being mapped differs and so does what a good pair is.
    The `tone` maps a 0…1 score, which has no intervals in it, so its range is chosen purely for
    legibility and 220–660 is a twelfth. This maps a *band*, whose ratios mean something, and the pair
    that leaves them alone is the one spanning as many octaves as the band does: 220–440 against 8–16 Hz
    is an exact transposition, ×27.5. Sharing the fields would have shipped a 1.6× stretch of everything
    the EEG does, chosen by accident.

    Wider is not simply better. A stretch exaggerates the wobble in the frequency estimate along with the
    signal, so a range far past the band's own span turns an unremarkable minute into a siren.
    `music.stretch_factor` computes the multiple and the panel prints it under the row."""
    follows: str = LEVEL
    """What the `shepard` source's pitch tracks: `LEVEL` — how well — or `STATE`, where the protocol
    supplies one (`TrainingSession.state_balance`, the same quantity the crossfade and a state-following
    shadow read).

    Exists because the two quantities can deserve different instruments on one screen. ILF Synchrony's
    level is the product's approach to its threshold — rare-event by design, mostly at its ends — while
    the protocol's state is the signed summed slow wave, which suits a Shepard tone that hangs off the signed waveform's position, not off a score. Pointing the
    pitch at the state and leaving the loudness on the level gives each signal its own perceptual
    dimension instead of blending two things with different dynamics into one number.

    Ignored by every source but `shepard`: the crossfade always follows the state and the tone always
    follows the level, each being the fixed relationship its source was chosen for. Only offered where a
    protocol *has* a state, for the same reason `Shadow.follows` is."""
    stops_when: str = "never"
    """When the playback goes silent, and there are two opposite answers because he uses both.

    - `"failing"` — the DVD model without a DVD: "the playback will pause whenever one of the Reward
      instruments falls below threshold". The information is in the interruption. He warns it can be "too
      jerky for some people", which is why the alert exists as the gentler alternative.
    - `"met"` — **Quiet Reward**, and the polarity is the whole point: "when the trainee is slightly more
      awake, adaptive and/or balanced and graceful then the sounds of the software **stop**." Silence is
      what is being trained toward, so it has to be a silence you can notice arriving. A proportional fade
      is not the same thing and cannot be — "the salience of silence" needs a discernible edge.

    A property of the playback rather than a mechanism, because what carries the information is whether it
    is there, not how much of it there is.
    """

    def holds(self, met: bool) -> bool:
        """Whether playback should be silent right now, given whether every target is met."""
        if self.stops_when == "failing":
            return not met
        if self.stops_when == "met":
            return met
        return False


# The two opposite polarities of silence, both of which he uses — see `Music.stops_when`.
STOPS_WHEN = {
    "never": "keeps playing",
    "failing": "stops while failing",
    "met": "stops when every target is met",
}
STOPS_WHEN_DETAIL = {
    "never": "plays without stopping; the mechanisms carry everything",
    "failing": "a pause tells you a target failed",
    "met": "silence is the reward, and it starts suddenly instead of fading in",
}

MUSIC_SOURCES = {
    "noise": "generated sound with a tilted spectrum",
    "crossfade": "two tilted sounds, with the signal fading between them",
    "tone": "one tone, its pitch tracking the signal between two frequencies",
    "shepard": "a pitch that rises with the signal and has no top",
    "file": "a file, or a folder played on repeat",
    "eeg": "the EEG's own rhythm, played as a pitch",
}

# Three named arrangements. Presets over the
# settings above rather than modes of their own — see `FeedbackSettings.apply_model`.
FEEDBACK_MODELS = {
    "quiet": "silence when every target is met",
    "proportional": "louder as the signal rises",
    "pause": "stops while a target fails",
}
FEEDBACK_MODELS_HELP = (
    "These are three arrangements of the mechanisms below. The line under the pills shows what the "
    "current one sets. Each arrangement also turns off what it does not use, so switching never leaves "
    "settings from the last one active. The mechanisms it sets are locked: use the pills above to "
    "change them.\n\n"
    "quiet — the music stops when every target is met. The master volume falls as the level rises, so "
    "the sound fades before it stops. Only the alerts remain, and silence is the reward.\n\n"
    "proportional — the music gets louder as the level rises. This is easy to read at once, but harder "
    "to use for noticing small changes.\n\n"
    "pause — the music plays while every target is met and stops when one fails. This is easy to read, "
    "but some people find the interruption uncomfortable."
)


@dataclass
class FeedbackSettings:
    """The signal, chosen once, plus one settings group per mechanism.

    **Sound and click start on; shadow, alert and music start off.** A session that opens with nothing
    connected to the person is a session that says "nothing is being fed back" until someone goes looking
    for the switch, and the two that are on are the ones every protocol wants at no cost. Alert is off
    because it is only meaningful with several targets to distinguish between, and music because it needs a
    choice.

    The shadow is the exception and `Shadow.enabled` carries the argument: it takes over a display's
    brightness the moment a session runs, so it is asked for rather than assumed. A protocol whose design
    wants it may still turn it on for itself, and `CFCSession` and `ILFSession` do — but only when
    `shadow.subject_screen()` finds a display that is not the operator's, since the alternative is that
    pressing play dims the person running the session.

    This docstring claimed the opposite for a while, and the claim outlived the code: the shadow was moved to
    off-by-default with its own written reason at `Shadow.enabled`, and the paragraph here went on promising
    that play would dim something. A default stated in two places is a default that will disagree with
    itself, so this one now points at the field instead of restating it.
    """

    # the signal — these live here and nowhere else
    combine: str = "weighted"          # weighted | weakest | scaled
    reward_share: float = DEFAULT_REWARD_SHARE
    normalisation: str = "sd"          # sd | threshold | absolute
    sensitivity: float = 1.0
    """How much of each output's travel an ordinary swing uses. **The one dial**, and it acts on the
    finished level rather than on any target's own ramp.

    It replaced a control that multiplied the per-target step — "level spans ±N σ" — and the replacement is
    not a rename. That control steepened each score around its *own* threshold, which on a level already
    sitting at 0.7 pushed it towards 1 and clipped instead of widening anything: measured, going from 1.0
    to 0.5 moved the level's 5th-to-95th percentile from 0.58–0.81 to 0.63–0.89, a swing 0.03 wider and
    shifted up. It also read backwards, since a bigger number meant flatter feedback, and it was reported
    exactly that way — turned up to 10 to get more movement, which removed what movement there was.

    The reason a steepness control cannot do this job is that the level is off-centre before it is narrow.
    Fixing that needs the level's own middle *and* its own spread, which is a measurement rather than a
    setting — see `TrainingSession.level_range` — and once those exist, the only thing left for an operator
    to choose is a gain. Hence one dial and not three.

    Bounded by `SENSITIVITY_RANGE`, defaulting to 1, and 1 is defined by `SENSITIVITY_HEADROOM`, which is
    where the measurements behind the default are."""
    # Averaging inside each IIR amplitude estimate; cFIR has no such stage. A "0.5 s" cutoff period
    # is a time constant of 0.5/pi = 0.159 s. Taking the 0.5 at face value smoothed three times as
    # hard: measured against a known envelope, r fell from 0.954 to 0.896 and the effective delay rose
    # from 252 ms to 420.
    envelope_seconds: float = 0.16
    envelope_convention: str = "rms"
    """Which amplitude the IIR estimators report: "rms", or "meanabs", which
    is `abs(bandpass(x))` into the smoother. The two differ by a fixed ratio on a sine (mean-abs reads
    0.900× the RMS figure), which is exactly why the choice exists: a threshold carried over from another program
    means what it meant only under the convention it was set in, and the whole reason the IIR kinds are
    offered is that thresholds do not transfer between designs. cFIR ignores this — it reports the peak
    (analytic modulus) and has no envelope stage at all."""
    # Smoothing of the finished 0-1 signal, after the targets are combined. The number is our own choice.
    smoothing_seconds: float = 0.3

    # the mechanisms
    sound: Sound = field(default_factory=Sound)
    shadow: Shadow = field(default_factory=Shadow)
    click: Click = field(default_factory=Click)
    alert: Alert = field(default_factory=Alert)

    # audio the session owns but which does not carry the level; see `Music`
    music: Music = field(default_factory=Music)

    arrangement: str = ""
    """Which named arrangement is in force, or empty where none was applied.

    Held so the pills can show *which one you are in* rather than reading as three buttons that do
    nothing visible. It cannot go stale, and the reason is the locking: a protocol that offers
    arrangements pins the fields they set, so a pill is the only way those fields change. Without that it
    would need a `custom` state the moment anything was nudged by hand — which is how the threshold modes
    work, and the right answer there for the same reason.
    """

    @property
    def mechanisms(self) -> dict[str, Mechanism]:
        return {"sound": self.sound, "shadow": self.shadow, "click": self.click,
                "alert": self.alert}

    def apply_model(self, name: str) -> None:
        """Set the mechanisms to one of the three named arrangements.

        These are presets over settings that already exist, not a fourth kind of thing — which is the
        test of whether the mechanism model was right. Each one also *clears* what it does not use, so
        switching to Quiet Reward cannot leave a proportional volume quietly still following the signal:
        a preset that only adds is not a preset, it is an accumulation.
        """
        if name not in FEEDBACK_MODELS:
            raise ValueError(f"unknown feedback model {name!r}")
        self.arrangement = name
        self.sound.enabled = self.click.enabled = self.alert.enabled = False
        self.sound.invert = False
        self.music.stops_when = "never"
        if name == "quiet":
            # Silence is the reward, so the music **stops** when every target is met — a
            # discernible edge rather than a fade. The inverted master volume on top of that is ours: it
            # makes the music recede *as* you approach, so the silence is arrived at rather than switched to.
            self.alert.enabled = True
            self.sound.enabled = True
            self.sound.invert = True
            self.music.stops_when = "met"
        elif name == "proportional":
            self.music.enabled = True
            self.sound.enabled = True
        elif name == "pause":
            self.music.enabled = True
            self.music.stops_when = "failing"
            self.alert.enabled = True

    def describe(self) -> str:
        on = [m.label for m in self.mechanisms.values() if m.enabled]
        return ", ".join(on) if on else "no output enabled"

    def level(self, targets: Iterable[Scored]) -> float:
        """The 0…1 signal. Purely the combined score: 0.5 when every target sits on its threshold.
        Whether that is enough to fire anything is a discrete mechanism's business."""
        targets = list(targets)
        if not targets:
            return 0.0
        return combine(targets, self.combine, self.normalisation, self.reward_share)

    def shape(self, level: float, centre: float, span: float) -> float:
        """The level moved onto the output's own range: `sensitivity` applied about a measured middle.

        **Two operations, and only one of them is the dial.** The level is first re-centred, because a
        combined score sits wherever the thresholds and the mix put it — 0.7 on the shipped amplitude
        arrangement — and a gain applied to an off-centre number saturates rather than widens. Then it is
        scaled, and that is the part an operator sets.

        `centre` and `span` are what the level actually did earlier in this session, held against
        re-centring but ratcheting outward; see `TrainingSession.level_range`. A `span` of zero
        means nothing has been measured yet, and then this is the identity — feedback from the first
        reading, unshaped, rather than a screen that waits.

        **The ends are approached, never hit** — the same rule `Target.score` already states for itself:
        an output pinned at its ceiling has stopped saying how well you are doing. This used to be a hard
        clip, and the clip is half of how a live session pinned volume and brightness at their maxima for
        good (the level range's docstring holds the other half and the measurements). The middle
        `1 − 2·SHAPE_KNEE` of the travel is exactly the straight line it always was; the outer knees bend
        away exponentially with the slope continuous at the joins, so however far past the calibrated range
        the level runs, 0.93 and 0.96 still read as two different brightnesses while the range ratchets out
        to meet it.
        """
        if span <= 0.0:
            return min(1.0, max(0.0, level))
        moved = 0.5 + self.sensitivity * (level - centre) / (SENSITIVITY_HEADROOM * span)
        if moved > 1.0 - SHAPE_KNEE:
            return 1.0 - SHAPE_KNEE * math.exp(-(moved - (1.0 - SHAPE_KNEE)) / SHAPE_KNEE)
        if moved < SHAPE_KNEE:
            return SHAPE_KNEE * math.exp(-(SHAPE_KNEE - moved) / SHAPE_KNEE)
        return moved

    def on_target(self, targets: Iterable[Scored]) -> bool:
        """Whether a discrete mechanism should be firing. Also shown in the readout.

        Every target on its wanted side — which is what `level() >= 0.5` amounts to under `weakest`, and
        does not under `weighted`. **Except under `either`**, where the rewards are alternatives and an AND
        over alternatives can never be satisfied: there it is one reward met and every inhibit met. See the
        module docstring for why the rule reaches the discrete mechanisms in that one case and nowhere else.
        """
        targets = list(targets)
        if not targets:
            return False
        if self.combine != "either":
            return all(getattr(x, "wanted", True) for x in targets)
        rewards = [x for x in targets if getattr(x, "role", "reward") == "reward"]
        inhibits = [x for x in targets if getattr(x, "role", "reward") != "reward"]
        if not all(getattr(x, "wanted", True) for x in inhibits):
            return False
        return any(getattr(x, "wanted", True) for x in rewards) if rewards else True

    def followers(self, channel: str) -> list[str]:
        """What is currently carrying one of the two channels, by name.

        The music appears under the condition when it is gated, because `stops_when` is asked of the
        condition and not of the level — the same channel the click is on, which is what makes Quiet
        Reward one arrangement rather than two coincidences. Its *loudness* is not here: it has a fixed
        gain of its own and the level reaches it only through the master volume, which is Sound.
        """
        names = [m.label for m in self.mechanisms.values()
                 if m.enabled and m.reads == channel]
        if channel == CONDITION and self.music.enabled and self.music.stops_when != "never":
            names.append("the music stops")
        return names

    def describe_combine(self) -> str:
        if self.combine == "weighted":
            return (f"{self.reward_share:.0%} rewards / {1 - self.reward_share:.0%} inhibits")
        return COMBINES[self.combine]

    @property
    def condition_rule(self) -> str:
        """Which form the condition takes, `all` or `either` — **derived, never stored.**

        It follows `combine` because there is only one place the two can disagree and this is it: under
        `either` the rewards are alternatives, so requiring all of them is requiring the impossible. A
        second setting saying the same thing is a second thing to keep in step.
        """
        return "either" if self.combine == "either" else "all"

    def describe_condition(self) -> str:
        """What the *condition* is, in words — printed beside it, because it is not always the same rule.

        Ordinarily it is the plain AND and the row above it has no bearing on it. Under `either` it does:
        the rewards are alternatives, so an AND over them is unsatisfiable and it becomes one reward plus
        every inhibit. That is the single place the level's rule reaches a discrete mechanism, and a
        channel whose definition changes silently under one setting is exactly what needs saying.
        """
        if self.combine == "either":
            return "one reward met, and every inhibit met"
        return "every target on its wanted side"

    def outputs(self, targets: Iterable[Scored]) -> dict[str, float]:
        """What each enabled mechanism would currently be doing. Nothing being trained means nothing
        happening, rather than every mechanism sitting at its floor."""
        targets = list(targets)
        if not targets:
            return {}
        return self.outputs_for(self.level(targets), self.on_target(targets))

    def outputs_for(self, level: float, on_target: bool = False) -> dict[str, float]:
        """Same, from an already-computed level — which is what the session passes in, since the level
        it feeds the mechanisms is smoothed and this must not recompute an unsmoothed one.

        `on_target` is passed separately because a discrete mechanism asks the targets directly rather
        than reading the level; see the module docstring.
        """
        return {name: m.output(level, on_target)
                for name, m in self.mechanisms.items() if m.enabled}

    def pipeline(self, targets: Iterable[Scored], envelope_active: bool = True) -> str:
        """The chain in order, so `scale from` and `driven by` are visibly two steps of one thing
        rather than two unrelated menus.

        The two smoothing constants are different stages and both appear when both run: `envelope_seconds`
        averages each band's amplitude before it is scored, `smoothing_seconds` averages the one number
        that comes out. Printing the chain is what keeps them from reading as a duplicated control.
        `envelope_active` is False when every estimator is cFIR — that stage then does not exist, and a
        chain describing a stage that is not running is a wrong picture of the signal path.
        """
        n = len(list(targets))
        first = (f"each band's amplitude, averaged over {self.envelope_seconds:g} s"
                 if envelope_active else "each band's amplitude")
        return (f"{first}  →  distance from its threshold, measured in "
                f"{NORMALISATIONS[self.normalisation]}  →  a 0–100% score per target  →  "
                f"all {n} reduced to one: {self.describe_combine()}  →  smoothed over "
                f"{self.smoothing_seconds:g} s  →  the level")

    def describe_outputs(self, targets: Iterable[Scored], level: float | None = None) -> str:
        if level is None:
            level = self.level(targets)
        got = self.outputs_for(level, self.on_target(targets))
        if not got:
            return "no output enabled"
        return " · ".join(self.mechanisms[name].describe_output(value, level)
                          for name, value in got.items())
