"""Coupling — slow phase against fast amplitude.

Built from a framework published as a series of essays, which proposes that the brain's stability is held the way a Fuller tensegrity holds itself up:
slow rhythms as a continuous tension network, fast bursts as compression struts that never touch each
other, and the coupling between them as the joint.

The short version of the provenance, because it decides what is on this screen and what is not. The
essays name four things a session could train:

- **Phase-amplitude coupling** — the "temporal cable network". The one claim in the framework with a large
  independent literature behind it, measurable from an ordinary recording, and not computed anywhere in
  this codebase before. The COUPLING panel.
- **Phase locking between regions** — the "struts". The LOCKING panel. It was nearly left out on a wrong
  argument: a reading of 0.73 on unrelated channels was taken as a
  verdict when it was a baseline, and it had been measured on a band twenty times narrower than this one.
  The bias is removed exactly rather than the measure abandoned.
- **Excitation-inhibition balance held at criticality** — the "pre-stress". Not trainable directly, but it
  has a published algorithm that estimates it from a band's amplitude envelope, with a target value in
  the *middle* rather than at either end. The BALANCE panel, as a readout.
- **Complexity, as a live reward.** Not built. The first reason given here was that entropy restates the
  power spectrum, and that was asserted rather than tested; tested against phase-randomised surrogates on
  real recordings it is **wrong** — sample entropy and multiscale entropy separate strongly. What the
  control surrogates then showed is that the separation lives at timescales longer than a feedback
  window: shuffling the signal into four-second blocks removes most of it. So it belongs beside the
  balance numbers, not on the reward path.

**Nobody's name and no product name is on this screen.** The provenance is here and in the protocol
document, which is the same line the cross-frequency coupling screen draws for the same reason.

**This is the slowest protocol here, and the number says so on the screen.** A coupling is a statistic
over many slow cycles: below about twelve seconds of signal every published estimator fails, and the two
computed here are the two most sensitive to running short. The window is twenty seconds, the reading is
drawn against the floor it has to clear, and the card states the window rather than implying immediacy.

The card's "reacts in" is the **half**-response, which is what that label means everywhere in this app,
and it is right: measured step response at the shipped window, half at 21.3 s against 20.1 s claimed. The
full response is much longer — 90% at 63 s — so the window pills carry the measured figures, because a
choice between 10, 20 and 40 seconds is a choice about that and nothing on screen said so.

**The reward has a floor drawn beside it, and that is the most important thing on the screen.** Every
phase-amplitude estimator reads above zero on unrelated signals, and worse, a slow rhythm that is merely
*sharp-edged* rather than sinusoidal produces a large reading with no coupling behind it at all — its own
harmonics land in the fast band already phase-locked. So the same statistic is computed against a
time-shifted copy of the fast amplitude, which breaks the phase relationship and preserves everything
else. Measured on a 6 Hz square wave in noise: the reading was 0.050, which would pass for strong
coupling, and the floor sat at 0.047. The gap is the signal. The reading alone is not.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import (
    BALANCED, BandPower, CFIRBandPower, COUPLING_WINDOW_SECONDS, Coupling, CouplingMetric, DFA_FLOOR,
    DFA_MIN_SECONDS, DFA_NOISE, Locking, LockingMetric, PHASE_BINS, dfa, functional_ei,
)
from nfb.dsp.filters import Band

from . import chrome, plan_pane, signal_pane, spectrum_pane, trace_rows, trends_pane
from .plots import EnvelopePlot, bars, place_value_labels, plot_ceiling, value_axis
from .protocols import Protocol
from .target_card import SLOW_RECOMPUTE_SECONDS, TargetCard
from .targets import PAIR, Target
from .traces import RawRing, TraceBuffer
from . import theme as t
from .training import METRIC_HZ, TrainingSession
from .widgets import help_button, labelled, panel, pill, tiny

HANDLES = ("PAC",)

SLOW = Band(4.0, 8.0, "theta")
"""The rhythm whose phase is read. Theta because that is the pairing the framework names first and the
one the independent literature is built on — gamma bursts nesting inside theta troughs, seven or so to a
cycle, which is the mechanism proposed for how many things working memory can hold at once."""

FAST = Band(30.0, 45.0, "low gamma")
"""The rhythm whose amplitude is measured.

**30–45 rather than the 30–80 the essays quote, and both halves of that have a reason.**

The bottom: amplitude modulated at the slow frequency puts its energy at `fast ± slow`, so a fast band
narrower than twice the slow centre cuts off the very thing being measured. At 6 Hz theta that is 12 Hz,
and 15 Hz clears it. `Coupling` refuses a band that does not, rather than reporting no coupling.

The top: above about 45 Hz scalp recordings are dominated by muscle, which is broadband, and broadband
activity gated by a slow rhythm is exactly what a coupling measure is looking for. Jaw tension would read
as a healthy brain. The band stops below that and `GUARD` watches what is above it."""

GUARD = Band(70.0, 90.0, "muscle")
"""An inhibit above the trained band, on by default, and it is not decoration.

The failure it prevents: the trainee clenches, the fast band's amplitude jumps in time with whatever they
were doing, the coupling reading climbs, the reward fires. That is training which is responsive,
immediate and teaching the wrong thing — worse than a protocol that does not work.

70–90 Hz is above the frequencies anyone trains and below the sample rate's own ceiling, and it clears
both mains frequencies and both of their second harmonics, so what is left in it is muscle."""

GUARD_HIT_RATE = 0.90

HELD_GUARD_TAU = 5.0
"""How long a locking channel's muscle guard averages over, in seconds.

**A different guard for a different failure.** A half-second burst of muscle does not move a phase
comparison — measured, a 100 µV burst on both channels left the locking at 0.035 against 0.034 clean,
because half a second is a small part of a five-second window averaged across a session. A *sustained*
clench does: 120 µV held throughout took two unrelated sites from 0.034 to **0.130**. So the guard on a
locking channel averages over five seconds, which a burst barely moves and a held jaw cannot hide from.
Guarding it as tightly as a coupling's channel would fail the trainee for something harmless."""
"""How much of the trainee's own muscle band stays under the line. Ordinary for an inhibit: this is a
guard against a specific artifact, not a second thing being trained."""

REWARD_HIT_RATE = 0.75
"""How much of the session a coupling reward is expected to sit below its threshold.

Looser than an amplitude reward's, because the quantity moves over twenty seconds rather than over one:
a tight aim on a slow signal produces a level that sits still for minutes at a time and tells the trainee
nothing about what they just did."""

CALIBRATION_SECONDS = 120.0
"""What a threshold is derived from. Six windows' worth, because a threshold set from fewer readings than
the estimator needs to settle is a threshold set on the estimator warming up."""

SLOW_BANDS = {"delta": Band(1.0, 4.0, "delta"),
              "theta": Band(4.0, 8.0, "theta"),
              "alpha": Band(8.0, 12.0, "alpha")}
"""The rhythms whose phase can be read — the framework's tension network, "theta and alpha rhythms in the
4-12 Hz range", with delta beside them because the same essays reach down to it."""

FAST_BANDS = {"beta": Band(15.0, 30.0, "beta"),
              "low gamma": Band(30.0, 45.0, "low gamma"),
              "wide gamma": Band(30.0, 55.0, "wide gamma")}
"""The rhythms whose amplitude can be measured.

**Three rather than one because the sideband rule makes the choice depend on the slow band.** A fast band
has to be at least twice the slow centre frequency wide or it cuts off the modulation it is meant to
show, so alpha's 10 Hz centre needs 20 Hz of fast band and neither of the 15 Hz-wide ones will do. The
menus say which combinations that leaves rather than letting one be picked and then refused."""

LOCK_BANDS = {"theta": Band(4.0, 8.0, "theta"),
              "alpha": Band(8.0, 12.0, "alpha"),
              "beta": Band(15.0, 30.0, "beta")}
"""Where two sites can be asked to hold a phase relationship. No gamma: at the scalp it is mostly muscle,
and a phase is only worth taking where there is a peak to have one."""

LOCK = Band(8.0, 12.0, "alpha")
"""Which band the two sites are asked to hold a steady phase relationship in.

The essays' second proposal — the struts, the fast relationships that a slow scaffold reconfigures. Alpha
rather than gamma because gamma at the scalp is mostly muscle, and because a phase is only worth taking
where there is a peak to have one, which at rest is here."""

LOCK_WINDOW_SECONDS = 5.0
"""How long the phase relationship is averaged over.

Short enough to train against and long enough to mean something: at alpha's 4 Hz width this is about
forty independent phase estimates, where an uncoupled pair separates from a half-locked one by `d' = 3.4`
— measured. The debias below makes the *number* independent of this setting, which is what lets it be a
control at all."""

LOCK_HIT_RATE = 0.75

OPENING_COUPLING, OPENING_LOCKING = 1.0, 20.0
"""Where each threshold sits before the calibration replaces it, in the units of its own card.

**The shared default is 10, and it is a number about microvolts.** On a modulation index scaled to a
percentage a 10 means nothing reachable, so for the two minutes of calibration the bar sat empty, the
plot's ceiling pinned to a line the trace could never approach, and the whole panel read as broken rather
than as waiting. These are the middle of each measure's real range instead: a modulation index of 0.01 is
a coupling worth having, and a debiased locking around 0.2 is a relationship.

They are placeholders and nothing is trained against them — every threshold here is derived from the
trainee's own first two minutes. What they buy is a screen that looks like it is working while it waits.
"""

TRAINING_HELP = """\
Each card's threshold updates on a timer. It uses the signal from the last 120 seconds, and a muscle guard \
uses the last 60 seconds. By default, it updates every 90 seconds. Feedback starts at once, against a provisional \
line that becomes accurate as the window fills. The card counts down while it fills.

A threshold that updates on every sample keeps the percentage of time above it fixed. You then cannot \
see a real change in your own signal. A threshold held for 90 seconds lets that percentage move within each \
period, so a real change shows. The "every" box sets this period. Set it to 0 for continuous updates.

Fixed holds one threshold for the whole session: the value from the first window, or a value you set \
again with redefine. If you change a threshold with − or +, the card switches to fixed, so your value \
stays in place."""
"""Why there is one threshold mode here, in the words of the reason rather than of the protocol.

**The articles say nothing about thresholds**, so a note implying they did would be putting our choice in
their mouths. The reason is arithmetic and it is ours: an automatic threshold derives a percentile from a
60 s rolling window, and a reading averaged over 20 s decorrelates in about that time, so the window holds
roughly *three* independent values. That is not a distribution; it is where the last minute happened to
be. The same objection the synchrony screen records against tracking a sub-hertz reward, arrived at from
the sample count rather than from a manual.
"""

LAGGED_SHARE = 0.4
"""How much of the locking has to survive discarding zero lag before the panel calls it a relationship.

Measured, as the ratio of `lagged` to `locked`: a genuine relationship scored 1.19 to 1.40 across every
coupling strength, a reference carrying 40 µV of alpha scored **0.33**, and a sustained clench scored
**0.00**. The line sits between them with room on both sides."""

LOCK_MARGIN = 2.5
"""How far above its own floor the drawn window has to sit before the panel calls it a relationship.

Both numbers are averaged over the whole drawn window first, or the comparison is between a noisy value
and a quiet one. With that, measured through the running screen at a 5 s window: unrelated channels put
the ratio at **1.2**, a quarter-locked pair at **1.8**, a half-locked one at **6.0** and a strongly
locked one at **13.0**. The line sits between the quarter and the half, which is where a relationship
stops being arguable — and this sentence is a description on a panel, not a threshold anything is fed
back from.
"""

CRITICALITY_BAND = Band(8.0, 12.0, "alpha")
"""Which band's envelope the balance readouts are computed on.

Alpha because it is where the published values come from and where the measure was characterised; it is
deliberately *not* the band being trained, so the readout is not a restatement of the reward."""

RHYTHM_RATIO = 1.5
"""The slow band's amplitude over its flanks', below which there is no rhythm to take a phase from.
Printed in the verdict, so the number on the readout and the line it is judged against sit together."""

SCALE = 100.0
"""What the modulation index is multiplied by before it reaches a threshold, a card or a plot.

The index is a fraction of the most uneven the fast amplitude could possibly be across the slow cycle, and
real values live between about 0.001 and 0.05 — a genuine scale rather than a small one, but not one a
card showing a decimal place can print. Times a hundred it is that same fraction as a percentage, which
is readable, carries no invented unit, and converts back to any number quoted in the literature by
dividing. The measure itself stays in its published units, so nothing downstream of `CouplingMetric` has
to know this happened.
"""

CRITICALITY_SECONDS = 900.0
CRITICALITY_EVERY = 5.0

CRITICALITY_FIRM = 300.0
"""How much envelope the balance numbers are called provisional until.

**Both numbers are computed from the first seconds that allow it, and this is only a label.** It was a
gate — nothing shown before five minutes — and that broke the rule every screen here lives by: feedback
from the first sample, provisional if it must be, never withheld because it is not yet accurate. The
reason the label exists is the measurement behind `DFA_NOISE`: at two minutes a *noise* envelope's memory
exponent came out as high as 0.67 across six seeds, over the inclusion floor, so a short run can show a
confident balance computed from nothing. At five minutes the worst of six was 0.64. So the caption says
how much the numbers are drawn from and calls them provisional under this; it does not hide them.
"""

TRACE_SECONDS = 6.0
SPECTRUM_LO, SPECTRUM_HI, SPECTRUM_BIN_HZ = 1.0, 95.0, 1.0
"""The Spectrum tab's range. **Up to 95 rather than the 60 the band screens use**, because this screen
trains a fast band and guards a faster one — a spectrum that stops at 60 cannot show either the muscle the
guard is watching for or whether the low gamma being trained is muscle too."""
PLOT_COLUMNS = 300
PLOT_SECONDS = 120.0
PLOT_H = 130
HIST_H = 86
AXIS_W = 44
SIDE_W = 430
"""The card column. **430, because that is what every screen with cards on it uses** — the amplitude,
alpha-theta and cross-frequency screens all set exactly this, and ILF 440. 330 is the width the screens
whose side panel holds *readouts* use, and taking it from one of those left the coupling card's control
strip about a hundred pixels short, so it wrapped one menu per line."""

COUPLING_HELP = """\
The top plot shows the coupling strength for the last two minutes. The solid line is the threshold. The \
dashed line is the floor: the same measurement with the fast amplitude's timing shifted, so it no \
longer matches the slow phase. A reading below the dashed line is not coupling.

The bars show the fast band's average amplitude at each phase of the slow wave's cycle. The cycle runs \
from the lowest point, up through the highest, and back down to the lowest. Flat bars mean the fast \
amplitude is the same at every phase. Tall bars at one point mean the fast band is strongest there. \
The line under the bars names that point, in words and in degrees (0° is the highest point, ±180° the \
lowest).

The number under the bars is the slow band's amplitude divided by the amplitude of the two \
neighbouring bands. Each neighbouring band is the same width as the slow band — for theta at 4–8 Hz, \
1–4 Hz and 8–12 Hz. The EEG spectrum always has energy at 4–8 Hz, because it slopes down through that \
range. An oscillation is a peak that stands above the slope, and a peak gives a value above 1. Below \
1.5 there is no peak: the band holds only the background slope, filtered to 4–8 Hz. The filter gives \
that background a phase, but the phase is the filter's, not the brain's. The coupling number is then \
noise, and the line turns amber.

If the line is amber, open the Spectrum tab for this channel. Move the slow band to a frequency that \
shows a peak, or choose a channel that shows one."""

LOCKING_HELP = """\
This panel shows how steadily the two sites hold a phase relationship in the alpha band. It does not \
show whether the two sites are both strong at once. It shows whether the offset between them stays the \
same.

The raw measurement reads high even on unrelated channels. How high depends on the averaging window: \
over two seconds it reads 0.22, over ten seconds 0.11. This screen removes that baseline, so a reading \
of 0 means no relationship beyond chance, at any window length.

The dashed line is the floor. It comes from the same measurement with one channel's phase delayed past \
the band's own memory, which removes any real relationship. The floor must stay near the bottom. If it \
does not, do not trust the reading above it.

The lag number shows which channel leads, and by how much. Signals that reach both electrodes at the \
same instant — from a noisy reference, a clenched jaw, or volume conduction through the scalp — produce \
a high reading at zero lag. The floor cannot detect this, because the two channels are genuinely \
locked. Check the lag when the reading is high. A high reading at zero lag can mean a wiring problem, \
not a true phase relationship between the two sites."""

BALANCE_HELP = """\
These two numbers describe the alpha envelope's fluctuations. Both show a reading from the first \
seconds and become more accurate as the run goes on. The caption marks them provisional until five \
minutes of envelope have passed, because a short run can show a false reading by chance.

Memory measures how much structure the envelope carries over tens of seconds. Compare it against 0.58, \
not 0.5: a band-filtered noise signal reads 0.58 on this measure. Published resting alpha falls between \
0.6 and 0.8, so the useful range is narrow. A reading only a little above 0.58 on a short run proves \
nothing yet.

Balance measures whether the envelope's fluctuations scale with its loudness. 1.0 is balanced. Below 1 \
means inhibition dominated, above 1 means excitation dominated. Balance is less reliable when Memory is \
below 0.6, because it is then a ratio between two noisy numbers. The caption states when this is the \
case."""


def coupling_label(c: Coupling) -> str:
    """Names the bands and the site, always.

    The site is not optional even when only one coupling is on screen: two within-channel couplings on
    different electrodes are the same two bands, so without it both cards read "theta→low gamma", the
    operator cannot tell them apart, and — found by adding a second one — the lookup that finds a
    target's metric by name hands back whichever was first, so removing one removed the other's guard.
    """
    where = (f"ch{c.phase_channel}" if c.within_channel
             else f"ch{c.phase_channel}→ch{c.amplitude_from}")
    return f"{c.slow.label}→{c.fast.label} {where}"


class CouplingSession(TrainingSession):
    """One or more phase-amplitude couplings, a muscle guard, and two balance readouts."""

    # Both modes; automatic on a period by default. See `target_card.SLOW_RECOMPUTE_SECONDS` for why
    # this screen stopped allowing only `fixed`.
    offers = ("sound", "shadow", "music", "click")
    """No alert and no keyed tones. Both speak in discrete events about an instant, and this screen has no
    instants — its quantity is an average over twenty seconds, so an event fired off it would be announcing
    a state that arrived some time ago and has been there a while."""
    music_sources = ("noise", "tone", "shepard", "file", "eeg")
    """The base list predates the tone and the Shepard voice, and both follow the level — which is what a
    reading averaged over twenty seconds suits best. No crossfade: that needs two named states."""
    normalisations = ("sd", "threshold")
    """No `absolute`: a 5 µV step on a reading that lives between 0 and 5 percent never moves."""
    pinned_feedback = frozenset({"combine"})
    """**The muscle guard has to veto the level, not average into it.** Under `weighted` a clenched jaw —
    which inflates the coupling threefold, the whole reason the guard exists — read as reward 1, inhibit 0,
    level 0.8, and the sound paid out on muscle while only the condition held. `scaled` lets the inhibit
    pull the level down, as the synchrony screen pins for the same reason."""

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0):
        super().__init__(protocol, path=path, device=device, length_seconds=length_seconds,
                         source=source, channels=channels or 1)
        self.window_seconds = COUPLING_WINDOW_SECONDS
        self.trace_seconds = TRACE_SECONDS
        self.feedback.combine = "scaled"
        # One click per averaging window, not one every two seconds: a quantity averaged over twenty
        # seconds does not change state faster than that, and a 2 s click on it is narration.
        self.feedback.click.interval_seconds = COUPLING_WINDOW_SECONDS
        self.traces = {i: TraceBuffer(320, TRACE_SECONDS, sp.rate_hz)
                       for i, sp in self.specs.items()}
        # **The Spectrum tab had a tab and no source.** It is not decoration here: the phase this whole
        # screen rests on is only interpretable where the slow band has a real peak, and the spectrum is
        # where an operator sees whether it does. The rhythm ratio on the coupling panel is the number;
        # this is the picture behind it.
        self.rings = {i: RawRing(2.0, sp.rate_hz, SPECTRUM_LO,
                                 min(SPECTRUM_HI, sp.rate_hz / 2 - 1), SPECTRUM_BIN_HZ)
                      for i, sp in self.specs.items()}
        self.couplings: dict[tuple, CouplingMetric] = {}
        self.lockings: dict[tuple, LockingMetric] = {}
        self.lock_window_seconds = LOCK_WINDOW_SECONDS
        self.lock_plot = EnvelopePlot(columns=PLOT_COLUMNS, count=1)
        self.lock_floor_plot = EnvelopePlot(columns=PLOT_COLUMNS, count=1)
        """The floor's own history, over the same window as the reading it qualifies.

        Kept because the *verdict* has to compare like with like. A single reading of a debiased phase
        statistic swings widely — measured, at a 5 s window an unrelated pair's reading reached 0.34 where
        its mean was 0.07 — and the drawn floor is already an average of four surrogates, so it sits much
        closer to the middle than any one reading does. Comparing one against the other asks whether a
        noisy number beats a quiet one, which it often does for no reason. Averaged over the same window,
        both settle and the comparison means something.
        """
        self.guards: dict[tuple, object] = {}
        self.guard_kinds: dict[int, str] = {}
        self.plot = EnvelopePlot(columns=PLOT_COLUMNS, count=1)
        self._plot_at = 0.0
        self._plot_step = PLOT_SECONDS / PLOT_COLUMNS

        # The envelope the balance readouts are computed from. Kept on the session rather than in the
        # panel because it is a history of the run, and the view is rebuilt whenever the operator visits
        # the device screen and comes back.
        self.balance_ring: deque = deque(maxlen=int(CRITICALITY_SECONDS * 10))
        self.balance_metric: CFIRBandPower | None = None
        self.balance_hz = 10.0
        self.memory: float | None = None
        self.balance: float | None = None
        self._balance_at = 0.0
        self._accepted: dict[int, bool] = {}
        self.focus_coupling: tuple | None = None
        """Which coupling the phase bars and the rhythm ratio are of.

        There is a selection rather than a colour per coupling because **hue means frequency everywhere in
        this app**, and several couplings on this screen are the same two bands on different electrodes —
        keyed to frequency they would all be one colour, and a new key would disagree with the spectrum,
        the cascade and every card. So the chosen one is drawn in its band's colour and the rest are
        faint, which is a statement about attention rather than about frequency.
        """

        # **One of each, on the first channels, and the rest is the operator's.** The screen opens with
        # the arrangement the framework describes — one nesting, one relationship — rather than one
        # instrument per electrode, because four couplings calibrating at once is four thresholds nobody
        # chose. Every other channel is reachable from the two "+ add" buttons.
        self.open_default()
        self.rebuild_balance()

    # --- what the protocol supplies ------------------------------------------------------

    def add_coupling(self, coupling: Coupling, role: str = "reward") -> Target | None:
        try:
            metric = CouplingMetric(self.specs, coupling, window_seconds=self.window_seconds)
        except ValueError:
            return None
        key = self.coupling_key(coupling)
        if key in self.couplings:
            return None
        self.couplings[key] = metric
        return self._coupling_target(coupling, role)

    def _coupling_target(self, coupling: Coupling, role: str = "reward") -> Target:
        cross = not coupling.within_channel
        tgt = Target(PAIR if cross else coupling.phase_channel, None, "increase", role=role,
                     mode="automatic", recompute_seconds=SLOW_RECOMPUTE_SECONDS,
                     window_seconds=CALIBRATION_SECONDS,
                     pair=coupling.channels if cross else None,
                     threshold=OPENING_COUPLING, name=coupling_label(coupling), unit="%",
                     hit_rate_override=REWARD_HIT_RATE,
                     # Six windows' worth. A threshold set from fewer readings than the estimator needs
                     # to settle is a threshold set on the estimator warming up.
                     calibration_seconds=CALIBRATION_SECONDS,
                     # A modulation index is a bounded ratio, not a narrowband envelope, so the
                     # one-parameter fit that assumes the latter would put the line somewhere unrelated
                     # to where the readings are. Same guard the synchrony screen carries for its PLV.
                     narrowband_envelope=False)
        self.targets.append(tgt)
        if not tgt.monitor:
            self.add_guard(coupling.amplitude_from)
        self._order_guards()
        self.resize_plots()
        return tgt

    def add_locking(self, locking: Locking) -> Target | None:
        try:
            metric = LockingMetric(self.specs, locking, window_seconds=self.lock_window_seconds)
        except ValueError:
            return None
        key = self.locking_key(locking)
        if key in self.lockings:
            return None
        self.lockings[key] = metric
        return self._locking_target(locking)

    def _locking_target(self, locking: Locking) -> Target:
        tgt = Target(PAIR, None, "increase", mode="automatic",
                     recompute_seconds=SLOW_RECOMPUTE_SECONDS, window_seconds=CALIBRATION_SECONDS,
                     pair=tuple(locking.pair),
                     threshold=OPENING_LOCKING, name=locking.label, unit="%",
                     hit_rate_override=LOCK_HIT_RATE,
                     # Bounded to 0-1 like a correlation, so the Rayleigh fit that assumes a narrowband
                     # envelope would put the line somewhere unrelated to where the readings are.
                     narrowband_envelope=False,
                     calibration_seconds=CALIBRATION_SECONDS)
        self.targets.append(tgt)
        for channel in locking.pair:
            self.add_guard(channel)
        self._order_guards()
        self.resize_plots()
        return tgt

    @staticmethod
    def locking_key(locking: Locking):
        return ("plv", locking.band.low, locking.band.high, tuple(locking.pair))

    def guard_kind(self, channel: int) -> str:
        """`fast` where a coupling reads its gamma from that channel, `held` where only a locking does.

        One guard per channel either way, and the fast one wins when a channel carries both: it is the
        stricter of the two, so a channel that has it needs nothing else.
        """
        if channel in self._earning_coupling_channels():
            return "fast"
        return "held"

    def _earning_coupling_channels(self) -> set:
        """Channels whose coupling can actually pay out — a monitor cannot, so it needs no guard.

        An inhibit on a channel nothing is trained on is something to fail for no reason, which is the
        same argument that gives a locking channel the gentler guard.
        """
        out = set()
        for key, metric in self.couplings.items():
            tgt = next((x for x in self.targets if self.key_for(x) == key), None)
            if tgt is not None and not tgt.monitor:
                out.add(metric.coupling.amplitude_from)
        return out

    def add_guard(self, channel: int) -> Target | None:
        """One muscle inhibit per channel any instrument reads, of the kind that channel needs.

        Per channel rather than one for the session, because the artifact is a property of an electrode:
        a clenched jaw raises the fast band under *that* site. A guard on a channel nothing is trained on
        would be an inhibit the trainee can fail for no reason.
        """
        if channel not in self.specs or GUARD.high >= self.specs[channel].rate_hz / 2:
            return None
        kind = self.guard_kind(channel)
        key = (channel, self.band_key(GUARD))
        existing = next((tg for tg in self.targets
                         if tg.band == GUARD and tg.source == channel), None)
        if existing is not None and self.guard_kinds.get(channel) == kind:
            return existing
        if existing is not None:
            self.remove_target(existing)
        self.guards[key] = (CFIRBandPower(self.specs[channel], GUARD) if kind == "fast"
                            else BandPower(self.specs[channel], GUARD,
                                           tau_seconds=HELD_GUARD_TAU, output_hz=METRIC_HZ))
        self.guard_kinds[channel] = kind
        tgt = Target(channel, GUARD, "increase", role="inhibit", mode="automatic",
                     recompute_seconds=SLOW_RECOMPUTE_SECONDS,
                     hit_rate_override=GUARD_HIT_RATE,
                     name=f"ch{channel} muscle" + ("" if kind == "fast" else ", held"))
        self.targets.append(tgt)
        return tgt

    def set_role(self, tgt: Target, role: str) -> None:
        """Train this coupling or merely watch it. The guard follows, since a monitor cannot pay out."""
        tgt.role = role
        tgt.hit_rate_override = REWARD_HIT_RATE
        for channel in self.instrument_channels(tgt):
            if role == "monitor":
                self.guards.pop((channel, self.band_key(GUARD)), None)
                self.guard_kinds.pop(channel, None)
                for x in [y for y in self.targets if y.band == GUARD and y.source == channel]:
                    self.remove_target(x)
            else:
                self.add_guard(channel)
        self._drop_orphan_guards()
        self._order_guards()

    def retune_coupling(self, tgt: Target, *, channel: int | None = None,
                        slow: Band | None = None, fast: Band | None = None) -> Target | None:
        """Point an existing coupling somewhere else, keeping its place in the list.

        **Rebuilt rather than edited, and its history is not carried.** The estimator is built around both
        bands and both channels, so changing any of them means new filters — and the readings already in
        the threshold's window are of a different quantity. `Target.inherit` exists for the case where
        they are not, which is a nudge within one band; none of these are that.
        """
        old = self.couplings.get(self.key_for(tgt))
        if old is None:
            return None
        try:
            # `Coupling` refuses a fast band too narrow for the sidebands, so the combination is
            # checked by building it rather than by a rule copied out here to disagree with it later.
            want = Coupling(slow or old.coupling.slow, fast or old.coupling.fast,
                            phase_channel=(old.coupling.phase_channel if channel is None
                                           else channel))
            metric = CouplingMetric(self.specs, want, window_seconds=self.window_seconds)
        except ValueError:
            return None
        at = self.targets.index(tgt)
        guarded = any(x.band == GUARD and x.source == old.coupling.amplitude_from
                      for x in self.targets)
        role = tgt.role
        self.drop(tgt)
        self.couplings[self.coupling_key(want)] = metric
        fresh = self._coupling_target(want, role)
        self.targets.remove(fresh)
        self.targets.insert(min(at, len(self.targets)), fresh)
        if guarded:
            self.add_guard(want.amplitude_from)
        self._order_guards()
        self.resize_plots()
        return fresh

    def retune_locking(self, tgt: Target, *, pair: tuple[int, int] | None = None,
                       band: Band | None = None) -> Target | None:
        old = self.lockings.get(self.key_for(tgt))
        if old is None:
            return None
        try:
            want = Locking(band or old.locking.band, tuple(pair or old.locking.pair))
            metric = LockingMetric(self.specs, want, window_seconds=self.lock_window_seconds)
        except ValueError:
            return None
        at = self.targets.index(tgt)
        self.drop(tgt)
        self.lockings[self.locking_key(want)] = metric
        fresh = self._locking_target(want)
        self.targets.remove(fresh)
        self.targets.insert(min(at, len(self.targets)), fresh)
        self._order_guards()
        self.resize_plots()
        return fresh

    def set_guarded(self, tgt: Target, on: bool) -> None:
        """Turn the muscle guard on a coupling's channel on or off.

        It is a switch rather than a card that appears by itself: the guard used to materialise beside
        the instruments and vanish when one was deleted, which is a card nobody asked for and then a card
        gone for no visible reason.
        """
        for channel in self.instrument_channels(tgt):
            if on:
                self.add_guard(channel)
            else:
                self.guards.pop((channel, self.band_key(GUARD)), None)
                self.guard_kinds.pop(channel, None)
                for x in [y for y in self.targets if y.band == GUARD and y.source == channel]:
                    self.remove_target(x)
        if on:
            self._order_guards()

    def instrument_channels(self, tgt: Target) -> tuple:
        """Which electrodes this instrument reads, and therefore which ones it wants guarded."""
        key = self.key_for(tgt)
        if key in self.couplings:
            return (self.couplings[key].coupling.amplitude_from,)
        if key in self.lockings:
            return tuple(self.lockings[key].locking.pair)
        return ()

    def guarded(self, tgt: Target) -> bool:
        channels = self.instrument_channels(tgt)
        return bool(channels) and all((c, self.band_key(GUARD)) in self.guards for c in channels)

    def _order_guards(self) -> None:
        """Each guard immediately after the coupling it belongs to.

        The relationship is the whole point and it was invisible: a guard sitting among the instruments
        as a peer reads as a third thing being trained rather than as a condition on the second.
        """
        guards = [x for x in self.targets if x.band == GUARD]
        rest = [x for x in self.targets if x.band != GUARD]
        out: list[Target] = []
        for tgt in rest:
            out.append(tgt)
            key = self.key_for(tgt)
            metric = self.couplings.get(key) or self.lockings.get(key)
            if metric is None:
                continue
            channels = ((metric.coupling.amplitude_from,) if key in self.couplings
                        else tuple(metric.locking.pair))
            for guard in guards:
                if guard.source in channels and guard not in out:
                    out.append(guard)
        self.targets = out + [g for g in guards if g not in out]

    def drop(self, tgt: Target) -> None:
        """Remove one instrument, and any guard that was only there for it."""
        key = self.key_for(tgt)
        if key == self.focus_coupling:
            self.focus_coupling = None
        self.couplings.pop(key, None)
        self.lockings.pop(key, None)
        self.amplitudes.pop(key, None)
        self.remove_target(tgt)
        self._drop_orphan_guards()
        self.resize_plots()

    @property
    def guarded_channels(self) -> set:
        """Every channel an instrument reads — each wants a guard of its own kind."""
        return self._earning_coupling_channels() | {
            c for m in self.lockings.values() for c in m.locking.pair}

    def _drop_orphan_guards(self) -> None:
        """Drop the guards nothing reads any more, and re-kind the ones whose owner changed.

        A channel that loses its coupling but keeps a locking still wants a guard — the *held* one, not
        the strict one it had. **Only channels that already carry a guard are re-kinded**, so a guard the
        operator switched off does not come back because an unrelated instrument was removed.
        """
        wanted = self.guarded_channels
        for key in [k for k in self.guards if k[0] not in wanted]:
            self.guards.pop(key)
            self.guard_kinds.pop(key[0], None)
            for tg in [x for x in self.targets if x.band == GUARD and x.source == key[0]]:
                self.remove_target(tg)
        for channel in [k[0] for k in list(self.guards) if k[0] in wanted]:
            self.add_guard(channel)

    def resize_plots(self) -> None:
        """One series per instrument. Rebuilt rather than resized, because a plot that grew a series
        mid-window would draw a line back to a zero that was never measured."""
        self.plot = EnvelopePlot(columns=PLOT_COLUMNS, count=max(1, len(self.couplings)))
        self.lock_plot = EnvelopePlot(columns=PLOT_COLUMNS, count=max(1, len(self.lockings)))
        self.lock_floor_plot = EnvelopePlot(columns=PLOT_COLUMNS, count=max(1, len(self.lockings)))

    @property
    def free_channels(self) -> list[int]:
        """Channels with no coupling on them yet — what "+ add" can still offer."""
        taken = {m.coupling.phase_channel for m in self.couplings.values()}
        return [c for c in self.specs if c not in taken]

    @property
    def free_pairs(self) -> list[tuple[int, int]]:
        taken = {tuple(sorted(m.locking.pair)) for m in self.lockings.values()}
        chans = list(self.specs)
        return [(a, b) for i, a in enumerate(chans) for b in chans[i + 1:]
                if (a, b) not in taken]

    def adopt_targets(self, previous) -> None:
        """**The instruments are adopted by their settings, not as objects, and this crashed first.**

        The base implementation hands the target list across and calls `retune` on each, which only the
        band-editing screens have — so changing the channel count on this screen raised
        `AttributeError: no attribute 'retune'` before a frame was drawn. Every protocol whose targets
        are not a freely edited band list overrides this; this one did not.

        Rebuilding rather than assigning is also what makes the channel count safe. A carried-over target
        would keep naming an electrode that no longer exists, with no metric behind it, and would read
        zero for the rest of the session. Here each instrument is offered to the new session and the ones
        whose channels are gone are simply refused — `add_coupling` and `add_locking` already return None
        when the metric cannot be built. If that leaves nothing, the default arrangement comes back,
        because a screen with no instruments is not a state an operator chose.
        """
        self.window_seconds = getattr(previous, "window_seconds", self.window_seconds)
        self.lock_window_seconds = getattr(previous, "lock_window_seconds",
                                           self.lock_window_seconds)
        self.targets = []
        self.couplings, self.lockings, self.guards = {}, {}, {}
        for key, metric in getattr(previous, "couplings", {}).items():
            # Which one is trained is a setting like any other, and losing it put four rewards on the
            # screen where the operator had left one.
            was = next((x for x in previous.targets if previous.key_for(x) == key), None)
            self.add_coupling(metric.coupling,
                              role="monitor" if (was is not None and was.monitor) else "reward")
        for metric in getattr(previous, "lockings", {}).values():
            self.add_locking(metric.locking)
        if not self.couplings and not self.lockings:
            self.open_default()
        self.focus_coupling = (previous.focus_coupling
                               if getattr(previous, "focus_coupling", None) in self.couplings
                               else None)
        self.resize_plots()

    def open_default(self) -> None:
        """A coupling on **every** channel, the first one trained and the rest watched.

        **One site drives the sound, all of them are on screen.** Those are different questions and the
        screen used to answer only the first, which made four electrodes buy one measurement and left
        nothing saying why ch0 was special — it is not, it is merely first.

        What should not be shared is the *output*: four slow indices averaged into one level barely move,
        and a trainee cannot tell which site they are affecting. So the others open as `monitor`, which
        this codebase already means precisely — measured, calibrated, drawn and scored, with only the
        wire to the output cut. Changing which one is trained is the role menu on its card.
        """
        chans = list(self.specs)
        for n, channel in enumerate(chans):
            self.add_coupling(Coupling(SLOW, FAST, phase_channel=channel),
                              role="reward" if n == 0 else "monitor")
        if len(chans) >= 2:
            self.add_locking(Locking(LOCK, (chans[0], chans[1])))

    @property
    def trained_floor_hz(self) -> float:
        """The lowest frequency anything here depends on — the slow bands', not the guard's.

        The base derives it from the targets' own bands, and a coupling target has none, so the only band
        it could see was the 70 Hz muscle guard. Harmless today, because the artifact panel only reads
        this to warn about training *under* the gate's 0.5 Hz floor and nothing here goes near that — but
        it is the wrong number, and a screen that reported 70 Hz would go on being wrong if a slow band
        ever moved down.
        """
        lows = [m.coupling.slow.low for m in self.couplings.values()]
        lows += [m.locking.band.low for m in self.lockings.values()]
        lows += [tgt.band.low for tgt in self.targets if tgt.band is not None]
        return min(lows) if lows else super().trained_floor_hz

    def rebuild_balance(self) -> None:
        first = next(iter(self.specs), None)
        self.balance_metric = (None if first is None
                               else CFIRBandPower(self.specs[first], CRITICALITY_BAND))
        self.balance_ring.clear()
        self.memory = self.balance = None

    @staticmethod
    def coupling_key(coupling: Coupling):
        return ("pac", coupling.slow.low, coupling.slow.high, coupling.fast.low,
                coupling.fast.high, coupling.channels)

    def key_for(self, tgt: Target):
        if tgt.band is None:
            for key, metric in self.couplings.items():
                if coupling_label(metric.coupling) == tgt.name:
                    return key
            for key, metric in self.lockings.items():
                if metric.locking.label == tgt.name:
                    return key
            return None
        return (tgt.source, self.band_key(tgt.band))

    def value_for(self, tgt: Target) -> float:
        return self.amplitudes.get(self.key_for(tgt), 0.0)

    def metric_for(self, tgt: Target):
        key = self.key_for(tgt)
        return self.couplings.get(key) or self.lockings.get(key)

    def latency_for(self, tgt: Target) -> float | None:
        metric = self.metric_for(tgt)
        return metric.latency_seconds if metric is not None else None

    def observe_raw(self, block, judged) -> None:
        ch = block.channel
        if ch in self.traces:
            self.traces[ch].push(self.for_display(ch, block.values), block.valid, judged.accepted)
        if ch in self.rings:
            self.rings[ch].push(np.asarray(block.values))
        # Remembered here because `consume` is handed the *fed* block, which for a rejected one is a
        # stand-in indistinguishable from real signal — and a window statistic must not count it.
        self._accepted[ch] = judged.accepted

    def consume(self, block) -> set:
        fresh = set()
        # Signal time, not the clock. The plot's axis is what the recording did, so a replay that runs
        # faster or slower than real time has to draw the same picture — and a headless test that drives
        # `feed_block` in a loop has no wall clock to advance at all.
        now = None
        counts = self._accepted.get(block.channel, True)
        for key, metric in self.couplings.items():
            readings = metric.feed(block, count=counts)
            if readings:
                self.amplitudes[key] = metric.value * SCALE
                fresh.add(key)
                now = readings[-1].t
        for key, lock in self.lockings.items():
            readings = lock.feed(block, count=counts)
            if readings:
                self.amplitudes[key] = lock.value * SCALE
                fresh.add(key)
                now = readings[-1].t
        for key, guard in self.guards.items():
            if key[0] != block.channel:
                continue
            readings = guard.feed(block)
            if readings:
                self.amplitudes[key] = readings[-1].value
                fresh.add(key)
        if (counts and self.balance_metric is not None
                and block.channel == next(iter(self.specs), 0)):
            for reading in self.balance_metric.feed(block):
                self.balance_ring.append(reading.value)
        if now is not None:
            self.push_displays(now)
        return fresh

    def push_displays(self, now: float) -> None:
        if now - self._plot_at >= self._plot_step:
            self._plot_at = now
            self.plot.push([m.modulation * SCALE for m in self.couplings.values()] or [0.0])
            self.lock_plot.push([m.locked * SCALE for m in self.lockings.values()] or [0.0])
            self.lock_floor_plot.push([m.floor * SCALE for m in self.lockings.values()] or [0.0])
        if now - self._balance_at >= CRITICALITY_EVERY:
            self._balance_at = now
            self.refresh_balance()

    @property
    def live_coupling(self) -> CouplingMetric | None:
        """The coupling the bars and the rhythm ratio are of — the chosen one, or the first."""
        if self.focus_coupling in self.couplings:
            return self.couplings[self.focus_coupling]
        return next(iter(self.couplings.values()), None)

    @property
    def live_locking(self) -> LockingMetric | None:
        return next(iter(self.lockings.values()), None)

    @property
    def balance_rate(self) -> float:
        """Envelope readings per second in `balance_ring` — the estimator's own stride, not a guess.

        The caption used to divide by ten while the estimator emitted twenty a second, so it reported
        twice the envelope actually collected and read "346 s of 300 s needed" with nothing shown.
        """
        if self.balance_metric is None:
            return 1.0
        return 1.0 / max(1e-6, self.balance_metric.stride / self.balance_metric.spec.rate_hz)

    @property
    def balance_seconds(self) -> float:
        return len(self.balance_ring) / self.balance_rate

    def refresh_balance(self) -> None:
        """Recompute the two balance numbers from the envelope so far.

        **From the first seconds that allow a fit, not from five minutes.** `None` means the estimator
        cannot run yet — DFA needs four of its shortest window — never that it is not yet trustworthy;
        how much is behind the numbers is the caption's to say. The balance is computed whether or not
        the memory exponent clears its inclusion floor, and the caption says when it does not.
        """
        if self.balance_metric is None:
            return
        rate = self.balance_rate
        envelope = np.asarray(self.balance_ring, dtype=float)
        try:
            self.memory = dfa(envelope, rate)
        except ValueError:
            self.memory = self.balance = None
            return
        try:
            self.balance = functional_ei(envelope, rate)
        except ValueError:
            self.balance = None


def session_for(*args, **kwargs) -> CouplingSession:
    kwargs.pop("heart_choice", None)
    kwargs.pop("heart_runner", None)
    kwargs.pop("mendi_choice", None)
    kwargs.pop("mendi_runner", None)
    kwargs.pop("pupil_choice", None)
    kwargs.pop("pupil_runner", None)
    return CouplingSession(*args, **kwargs)


def view(page: ft.Page, session: CouplingSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    """Build once, mutate on Flet's loop."""

    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width(reserve: int = SIDE_W) -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - reserve - 70)

    spectrum = spectrum_pane.SpectrumPane(page, session, plot_width)

    trends = trends_pane.TrendsPane(page, session, plot_width)
    traces = trace_rows.TraceRows(page, session, plot_width, height=54)
    signal = signal_pane.SignalPane(page, session, inset=False)

    strength_canvas = cv.Canvas(shapes=[], expand=True, height=PLOT_H)
    strength_axis, strength_labels = value_axis(PLOT_H, count=4, width=AXIS_W)
    strength_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    # Two lines, both always present, so the pane's height cannot change with what they say.
    strength_state = ft.Text("", size=t.SMALL, color=t.FAINT)

    hist_canvas = cv.Canvas(shapes=[], expand=True, height=HIST_H)
    hist_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    # The rhythm check is the second of two facts and gets its own line — the two together ran to 130
    # characters. Always present, so the pane holds its height.
    hist_rhythm = ft.Text("", size=t.SMALL, color=t.FAINT)
    hist_ticks = ft.Row([ft.Text(s, size=t.SMALL, color=t.FAINT, expand=True,
                                 text_align=ft.TextAlign.CENTER)
                         # Plain words for the four quarters of the slow wave — "trough" and "peak" are
                         # English a reader may not have, and "rising edge" is engineering.
                         for s in ("lowest", "going up", "highest", "going down", "lowest")])

    lock_canvas = cv.Canvas(shapes=[], expand=True, height=PLOT_H)
    lock_axis, lock_labels = value_axis(PLOT_H, count=4, width=AXIS_W)
    lock_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    lock_legend = ft.Row(spacing=14, wrap=True)
    lock_state = ft.Text("", size=t.SMALL, color=t.FAINT)

    memory_text = ft.Text("", size=20, color=t.TEXT, weight=ft.FontWeight.W_500)
    memory_note = ft.Text("", size=t.SMALL, color=t.FAINT)
    balance_text = ft.Text("", size=20, color=t.TEXT, weight=ft.FontWeight.W_500)
    balance_track = ft.Stack([], height=10)
    balance_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    # **Nothing to show is one line, not a pane of dashes.** For the first five minutes this panel held
    # two "—" readouts and an empty track over a caption counting the seconds; the caption is the whole
    # content until there is a number, so the readouts stay out of the tree until then.
    balance_body = ft.Column([], spacing=0, visible=False)

    focus_pills = ft.Row(spacing=6, wrap=True,
                         vertical_alignment=ft.CrossAxisAlignment.CENTER)
    cards: dict[int, TargetCard] = {}
    card_column = ft.Column(spacing=10)
    window_pills = ft.Row(spacing=6)

    def after_change() -> None:
        signal.rebuild_contributions(session.targets)
        signal.rebuild()
        build_cards()

    def set_window(seconds: float) -> None:
        """Rebuild every coupling at a new averaging window. All of them, so they stay comparable."""
        session.window_seconds = seconds
        for key, metric in list(session.couplings.items()):
            session.couplings[key] = CouplingMetric(session.specs, metric.coupling,
                                                    window_seconds=seconds)
        session.amplitudes = {k: v for k, v in session.amplitudes.items() if k[0] != "pac"}
        build_window_pills()

    def build_window_pills() -> None:
        # The measured half-response, in the tooltip rather than on the face: three pills reading
        # "10 s · half in 12 s" would be the same sentence three times, and the choice is between them.
        window_pills.controls = [
            pill(f"{v:g} s", session.window_seconds == v,
                 (lambda w=v: lambda _e: set_window(w))())
            for v in (10.0, 20.0, 40.0)]
        for control, (half, full) in zip(window_pills.controls,
                                         ((12, 21), (21, 63), (52, 102))):
            control.tooltip = (f"a change in the coupling is half shown after {half} s and "
                               f"nine tenths shown after {full} s — measured")
        page.update()

    def choose(key) -> None:
        session.focus_coupling = key
        page.update()

    def build_focus_pills() -> None:
        """Which coupling the bars are of. Absent while there is only one, because a lone pill offering
        the state it is already in is a control that does nothing."""
        keys = list(session.couplings)
        focus_pills.visible = len(keys) > 1
        if not focus_pills.visible:
            focus_pills.controls = []
            return
        chosen = session.live_coupling
        focus_pills.controls = [ft.Text("showing", size=t.SMALL, color=t.FAINT)] + [
            pill(f"ch{session.couplings[k].coupling.phase_channel}",
                 session.couplings[k] is chosen, (lambda kk=k: lambda _e: choose(kk))())
            for k in keys]

    def add_next_coupling(_e=None) -> None:
        """Another coupling, on the first electrode that has none.

        **Where it starts does not matter, which is the point of the rewrite.** This used to be the only
        way to say anything — the card was dead text, so wanting a coupling on the fourth electrode meant
        adding three and deleting two. Every part of the card is now a menu, so "+ add" only has to
        produce *an* instrument and the operator points it.
        """
        free = session.free_channels
        if free:
            session.add_coupling(Coupling(SLOW, FAST, phase_channel=free[0]))
            after_change()

    def add_next_locking(_e=None) -> None:
        free = session.free_pairs
        if free:
            session.add_locking(Locking(LOCK, free[0]))
            after_change()

    def band_menu(label: str, options: dict, current: Band, apply,
                  allowed=lambda _b: True) -> ft.Control:
        """A named band, as the menu that sets it. The card's title *is* the control.

        Invalid choices are shown and disabled rather than hidden, with the reason in the item: the
        sideband rule that stops alpha pairing with a 15 Hz fast band is worth learning once, and a menu
        whose contents change as you use it teaches nothing.
        """
        items = []
        for name, band in options.items():
            why = allowed(band)
            items.append(ft.PopupMenuItem(
                content=ft.Text(name if why is True else f"{name} — {why}", size=t.BODY),
                disabled=why is not True,
                on_click=(lambda b=band: lambda _e: apply(b))()))
        return ft.PopupMenuButton(
            content=ft.Container(
                content=ft.Row([ft.Text(label, size=t.BODY, color=t.TEXT),
                                ft.Text("▾", size=9, color=t.FAINT)], spacing=5),
                padding=ft.Padding(7, 3, 7, 3), border_radius=6, bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
            items=items)

    def site_menu(label: str, options, apply) -> ft.Control:
        return ft.PopupMenuButton(
            content=ft.Container(
                content=ft.Row([ft.Text(label, size=t.BODY, color=t.TEXT),
                                ft.Text("▾", size=9, color=t.FAINT)], spacing=5),
                padding=ft.Padding(7, 3, 7, 3), border_radius=6, bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
            items=[ft.PopupMenuItem(content=ft.Text(name, size=t.BODY),
                                    on_click=(lambda v=value: lambda _e: apply(v))())
                   for name, value in options])

    def identity_block(controls: list, edges: str, tgt: Target) -> ft.Control:
        """The menus on one line with the remove × at its right, and the bands they resolve to under it.

        **Two lines rather than one wrapping line, and the difference is not cosmetic.** A `wrap=True`
        row has no bounded main axis, so an `expand=True` spacer inside it has nothing to expand against
        — it rendered as a grey block the height of the card. No other wrapping row in this app contains
        one; the ones that push content right set `alignment=END` instead. Here the menus fit on one
        unwrapped line at this column width, so the spacer is back to doing the job it is for.
        """
        return ft.Column(
            [ft.Row(controls + [ft.Container(expand=True),
                                tiny("×", (lambda g=tgt: lambda _e: drop(g))())],
                    spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
             ft.Text(edges, size=t.SMALL, color=t.FAINT)],
            spacing=4, tight=True)

    def coupling_identity(tgt: Target) -> ft.Control:
        metric = session.couplings[session.key_for(tgt)]
        c = metric.coupling
        # **Taken by *this pairing*, not by any coupling.** Two couplings on one channel in different
        # bands are a perfectly ordinary thing to want — theta→gamma and delta→beta at the same site —
        # and once every channel opened with one, excluding any channel that had a coupling left every
        # card unable to move anywhere at all.
        taken = {m.coupling.phase_channel for m in session.couplings.values()
                 if m is not metric and (m.coupling.slow, m.coupling.fast) == (c.slow, c.fast)}

        def fast_ok(band: Band) -> object:
            needed = 2 * c.slow.centre
            if band.low <= c.slow.high:
                return f"not above {c.slow.label}"
            if band.high - band.low < needed:
                return f"needs {needed:g} Hz for {c.slow.label}"
            return True

        def slow_ok(band: Band) -> object:
            if c.fast.low <= band.high:
                return f"not under {c.fast.label}"
            if c.fast.high - c.fast.low < 2 * band.centre:
                return f"{c.fast.label} is too narrow"
            return True

        return identity_block(
            [site_menu(f"ch{c.phase_channel}",
                       [(f"ch{n}", n) for n in session.specs if n not in taken],
                       lambda n, g=tgt: retune(g, channel=n)),
             band_menu(c.slow.label, SLOW_BANDS, c.slow,
                       lambda b, g=tgt: retune(g, slow=b), allowed=slow_ok),
             ft.Text("→", size=t.BODY, color=t.FAINT),
             band_menu(c.fast.label, FAST_BANDS, c.fast,
                       lambda b, g=tgt: retune(g, fast=b), allowed=fast_ok),
             # **Which one drives the sound is a property of the instrument, so it sits with it.** One
             # site is trained and the rest are watched; promoting another is this menu, and the muscle
             # guard follows, because a monitor cannot pay out and so has nothing to be guarded from.
             site_menu(tgt.role, [("reward", "reward"), ("monitor", "monitor")],
                       lambda r, g=tgt: set_role(g, r))],
            f"{c.slow.low:g}–{c.slow.high:g} → {c.fast.low:g}–{c.fast.high:g} Hz", tgt)

    def locking_identity(tgt: Target) -> ft.Control:
        metric = session.lockings[session.key_for(tgt)]
        lk = metric.locking
        taken = {tuple(sorted(m.locking.pair)) for m in session.lockings.values()
                 if m is not metric}
        chans = list(session.specs)
        pairs = [(a, b) for i, a in enumerate(chans) for b in chans[i + 1:]
                 if (a, b) not in taken]
        return identity_block(
            [site_menu(f"ch{lk.pair[0]} ↔ ch{lk.pair[1]}",
                       [(f"ch{a} ↔ ch{b}", (a, b)) for a, b in pairs],
                       lambda pr, g=tgt: retune_lock(g, pair=pr)),
             band_menu(lk.band.label, LOCK_BANDS, lk.band,
                       lambda b, g=tgt: retune_lock(g, band=b))],
            f"{lk.band.low:g}–{lk.band.high:g} Hz", tgt)

    def guard_identity(tgt: Target) -> ft.Control:
        """Named for what it is *for*, so a card nobody added reads as a consequence of one they did."""
        held = session.guard_kinds.get(tgt.source) == "held"
        # **Parallel and short.** The held guard's caption used to explain *why* it averages slowly, and
        # it ran past the card's right edge and pushed the remove × off it. The reason is behind the
        # gear of the instrument it belongs to, where there is room for it.
        return ft.Row([ft.Text(tgt.name, size=t.BODY, color=t.TEXT),
                       ft.Text("guarding the comparison above" if held
                               else "guarding the coupling above", size=t.SMALL, color=t.FAINT),
                       ft.Container(expand=True),
                       tiny("×", (lambda c=tgt.source: lambda _e: unguard(c))())],
                      spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def guard_rows(tgt: Target):
        """Behind the coupling's gear: whether its channel carries a muscle inhibit.

        A switch rather than a card that appears by itself. It used to materialise beside the instruments
        whenever a coupling was added and vanish when one was deleted, which is a card nobody asked for
        and then a card gone for no visible reason — and it was the only card with no way to remove it.
        """
        def rows(refresh):
            on = session.guarded(tgt)

            def flip(_e=None):
                session.set_guarded(tgt, not on)
                after_change()

            channels = session.instrument_channels(tgt)
            if tgt.monitor:
                return [labelled("muscle guard", ft.Text("not needed", size=t.SMALL, color=t.MUTED),
                                 note="A monitored instrument produces no output, so a muscle guard "
                                      "protects nothing. Set it to reward, and the guard turns on "
                                      "with it.")]
            held = all(session.guard_kinds.get(c) == "held" for c in channels)
            return [labelled("muscle guard",
                             pill("on", on, flip), pill("off", not on, flip),
                             note=(f"An inhibit at {GUARD.low:g}–{GUARD.high:g} Hz on "
                                   + ("both channels, averaged over five seconds. A short clench "
                                      "does not change the phase reading. A sustained clench does "
                                      "change it."
                                      if held else
                                      "this channel. Above this band the signal is mostly muscle "
                                      "activity. Muscle activity timed to the slow rhythm can look "
                                      "like coupling. Without the guard, a clenched jaw can trigger "
                                      "the reward.")))]
        return rows

    def set_role(tgt: Target, role: str) -> None:
        session.set_role(tgt, role)
        after_change()

    def retune(tgt: Target, **kwargs) -> None:
        session.retune_coupling(tgt, **kwargs)
        after_change()

    def retune_lock(tgt: Target, **kwargs) -> None:
        session.retune_locking(tgt, **kwargs)
        after_change()

    def unguard(channel: int) -> None:
        owner = next((tg for tg in session.targets
                      if session.key_for(tg) in session.couplings
                      and session.couplings[session.key_for(tg)].coupling.amplitude_from == channel),
                     None)
        if owner is not None:
            session.set_guarded(owner, False)
            after_change()

    def drop(tgt: Target) -> None:
        session.drop(tgt)
        after_change()

    def build_cards() -> None:
        cards.clear()
        controls = []
        for tgt in session.targets:
            key = session.key_for(tgt)
            if key in session.couplings:
                identity = coupling_identity(tgt)
            elif key in session.lockings:
                identity = locking_identity(tgt)
            else:
                identity = guard_identity(tgt)
            card = TargetCard(tgt, identity, after_change, page=page,
                              modes=session.threshold_modes,
                              settings=(guard_rows(tgt) if session.key_for(tgt) in session.couplings
                                        else None))
            cards[id(tgt)] = card
            controls.append(card.control)
        card_column.controls = controls
        page.update()

    build_window_pills()
    build_cards()

    train_left = ft.Column([
        panel("COUPLING",
              ft.Row([strength_axis,
                      ft.Container(content=strength_canvas, expand=True, height=PLOT_H)], spacing=0),
              ft.Container(height=6), strength_caption, strength_state,
              ft.Container(height=12),
              # **Beside the bars, not in the header.** It is a control over this one drawing rather
              # than over the panel, and in the header it was the sixth thing there and pushed the
              # panel's own name onto a second line.
              ft.Row([ft.Container(width=AXIS_W), focus_pills], spacing=0),
              ft.Container(height=6),
              ft.Row([ft.Container(width=AXIS_W),
                      ft.Container(content=hist_canvas, expand=True, height=HIST_H)], spacing=0),
              ft.Row([ft.Container(width=AXIS_W), ft.Container(content=hist_ticks, expand=True)],
                     spacing=0),
              ft.Container(height=4), hist_caption, hist_rhythm,
              # **The rhythm ratio came out of here.** A header carrying a title, a help button, a
              # channel selector, a readout, three window pills and an add button wrapped onto three
              # lines and put the readout above the panel name. A readout is not a control: it moved to
              # the caption under the bars, which is the picture it qualifies.
              trailing=window_pills,
              help=help_button(page, "Reading the coupling", COUPLING_HELP)),
        ft.Container(height=12),
        panel("LOCKING",
              ft.Row([lock_axis,
                      ft.Container(content=lock_canvas, expand=True, height=PLOT_H)], spacing=0),
              ft.Container(height=6), ft.Row([lock_legend, ft.Container(expand=True)]),
              ft.Container(height=4), lock_caption, lock_state,
              help=help_button(page, "Reading the phase locking", LOCKING_HELP)),
        ft.Container(height=12),
        panel("BALANCE",
              # **Both columns hold three rows, or they do not line up.** The memory side has a note
              # under its number and the balance side does not, so centring the row put the two readouts
              # at different heights — two numbers meant to be read together, half a line apart. The
              # empty text is the spacer; it is there whether or not it says anything.
              balance_body, balance_caption,
              help=help_button(page, "The two balance numbers", BALANCE_HELP)),
        ft.Container(height=12),
        panel("RAW", *traces.rows, ft.Container(height=6), traces.caption),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)

    # **One add control, in the header of the panel the cards are in.** That is where both other screens
    # put theirs, and it was the other half of the confusion: two identical "+ add" buttons sat in the
    # plot panels on the left while the cards they made appeared in a column on the right.
    add_button = ft.PopupMenuButton(
        content=ft.Container(content=ft.Text("+ add", size=t.SMALL, color=t.ACCENT),
                             padding=ft.Padding(10, 5, 10, 5), border_radius=6,
                             bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
        items=[ft.PopupMenuItem(
                   content=ft.Text("coupling" if session.free_channels
                                   else "coupling — every channel has one", size=t.BODY),
                   disabled=not session.free_channels, on_click=add_next_coupling),
               ft.PopupMenuItem(
                   content=ft.Text("phase locking" if session.free_pairs
                                   else "phase locking — every pair has one", size=t.BODY),
                   disabled=not session.free_pairs, on_click=add_next_locking)])

    # The mode is on each card now — pills, and the period beside them — so the panel states nothing
    # about it; the reasoning is behind its `?`.
    train_right = ft.Column([panel("TRAINING", card_column, trailing=add_button,
                                   help=help_button(page, "Thresholds here", TRAINING_HELP))],
                            spacing=0, width=SIDE_W, scroll=ft.ScrollMode.AUTO)

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

    def target_named(name: str) -> Target | None:
        return next((tg for tg in session.targets if tg.name == name), None)

    def draw_strength() -> None:
        metric = session.live_coupling
        width = plot_width()
        keys = list(session.couplings)
        chosen = session.live_coupling
        colours, thresholds, floors = [], [], []
        for key in keys:
            m = session.couplings[key]
            on = m is chosen
            colours.append(t.band_colour(m.coupling.slow.low, m.coupling.slow.high)
                           if on else t.ACCENT_DIM)
            tg = target_named(coupling_label(m.coupling))
            thresholds.append(tg.threshold if (tg is not None and on) else None)
            floors.append(m.floor * SCALE if on else 0.0)
        floor = metric.floor * SCALE if metric is not None else 0.0
        target = target_named(coupling_label(metric.coupling)) if metric is not None else None
        top = plot_ceiling(max(session.plot.peak(), floor * 2, 0.1),
                           target.threshold if target else None)
        strength_canvas.shapes = session.plot.shapes(
            width, PLOT_H, top, colours or [t.ACCENT],
            thresholds=thresholds, higher=floors)
        build_focus_pills()
        # **Positioned, not just written.** Setting `label.value` leaves every label at `top=0`, so all
        # four stack in the corner and read as one unparsable number — which is exactly how it looked.
        # Every other plot in this app calls this; this one set the text by hand and never placed it.
        place_value_labels(strength_labels, PLOT_H, top, unit="%")

        strength_caption.value = (f"{PLOT_SECONDS:g} s across · solid = threshold, "
                                  f"dashed = floor · percent of the largest unevenness possible")
        if metric is None:
            strength_state.value = "no coupling is set up"
            return
        clears = metric.modulation * SCALE > floor * 1.5 and metric.settled
        strength_state.value = (
            f"averaged over {session.window_seconds:g} s · "
            + ("clear of the floor" if clears
               else "at the floor — nothing here is coupling yet" if metric.settled
               else "filling the first window"))

    def draw_histogram() -> None:
        metric = session.live_coupling
        width = plot_width()
        if metric is None:
            hist_canvas.shapes = []
            hist_caption.value = hist_rhythm.value = ""
            return
        means = metric.bin_means
        # Rolled so the drawing starts at the trough, which is what the tick labels say and where the
        # literature's own figures put it — a histogram whose axis starts at an arbitrary phase is a
        # picture you have to be told how to read.
        means = np.roll(means, PHASE_BINS // 2)
        hist_canvas.shapes = bars(means, width, HIST_H, t.ACCENT_DIM)
        if not means.any():
            hist_caption.value = "waiting for the slow wave to complete one cycle"
            hist_rhythm.value = ""
            return
        degrees = np.degrees(metric.preferred_phase)
        where = ("at its highest" if abs(degrees) < 45 else
                 "at its lowest" if abs(degrees) > 135 else
                 "going up" if degrees < 0 else "going down")
        # One line, and every number on it says what it is. It read "theta is 0.7× its neighbours — too
        # flat for a phase to mean much": no unit, no named comparison, and a conclusion with no
        # mechanism. The quantity is the slow band's amplitude against the bands either side of it; a
        # rhythm is a peak, so it stands above them, and the line it is judged against is printed. Why
        # that decides whether the phase means anything is the `?`'s to explain, not this line's.
        slow, site = metric.coupling.slow.label, f"ch{metric.coupling.phase_channel}"
        # Judged on the number as printed: 1.47 shown as "1.5×" and coloured for being under 1.5 is a
        # line disagreeing with itself.
        shown = round(metric.peak_ratio, 1)
        fast = metric.coupling.fast.label
        hist_caption.value = f"{site} · {fast} is strongest while {slow} is {where} ({degrees:+.0f}°)"
        hist_rhythm.color = t.WARN if shown < RHYTHM_RATIO else t.FAINT
        hist_rhythm.value = (f"{slow} amplitude {shown:.1f}× the bands either side · a peak is above "
                             f"{RHYTHM_RATIO:g}")

    def draw_locking() -> None:
        metric = session.live_locking
        width = plot_width()
        metrics = list(session.lockings.values())
        colours = [t.band_colour(m.locking.band.low, m.locking.band.high) for m in metrics]
        targets = [target_named(m.locking.label) for m in metrics]
        floor = metric.floor * SCALE if metric is not None else 0.0
        top = plot_ceiling(max(session.lock_plot.peak(), floor * 2, 10.0),
                           *[tg.threshold for tg in targets if tg])
        lock_canvas.shapes = session.lock_plot.shapes(
            width, PLOT_H, top, colours or [t.ACCENT],
            thresholds=[tg.threshold if tg else None for tg in targets],
            higher=[m.floor * SCALE for m in metrics])
        place_value_labels(lock_labels, PLOT_H, top, unit="%")
        lock_caption.value = (f"{PLOT_SECONDS:g} s across · solid = threshold, dashed = floor · "
                              f"chance already taken out, so 0 is nothing")
        if metric is None:
            lock_state.value = "one channel — phase locking needs two"
            return
        # The verdict is of the *drawn window*, not of the latest reading: a debiased phase statistic
        # swings by about 1/sqrt(N) sample to sample, so a sentence read off one value flips while the
        # plot under it plainly shows one thing.
        drawn = session.lock_plot.recent_mean()
        drawn_floor = session.lock_floor_plot.recent_mean()
        lock_legend.controls = [
            ft.Row([ft.Container(width=10, height=3, border_radius=2,
                                 bgcolor=t.band_colour(m.locking.band.low, m.locking.band.high)),
                    ft.Text(m.locking.label, size=t.SMALL, color=t.MUTED)], spacing=6)
            for m in session.lockings.values()]
        shared = metric.lagged < LAGGED_SHARE * max(metric.locked, 1e-9)
        lead = metric.locking.pair[0 if metric.lag_degrees >= 0 else 1]
        if not metric.settled or drawn is None or drawn_floor is None:
            verdict = "still filling"
        elif drawn <= LOCK_MARGIN * max(drawn_floor, 1.0):
            verdict = "nothing beyond chance"
        elif shared:
            # The case the floor cannot see: genuinely above chance, and the same at both electrodes.
            verdict = (f"at {abs(metric.lag_degrees):.0f}° — a shared reference or a clench "
                       f"looks like this")
        else:
            verdict = f"holding steady · ch{lead} leads by {abs(metric.lag_degrees):.0f}°"
        lock_state.color = t.WARN if (metric.settled and shared and drawn is not None
                                      and drawn_floor is not None
                                      and drawn > LOCK_MARGIN * max(drawn_floor, 1.0)) else t.FAINT
        lock_state.value = (f"{metric.locking.band.label} · averaged over "
                            f"{session.lock_window_seconds:g} s · {verdict}")

    def draw_balance() -> None:
        width = max(80.0, plot_width() - 300)
        if session.memory is not None and not balance_body.visible:
            balance_body.controls = [
                ft.Row([ft.Column([ft.Text("memory", size=t.SMALL, color=t.FAINT), memory_text,
                                   memory_note], spacing=2),
                        ft.Container(width=34),
                        ft.Column([ft.Text("balance", size=t.SMALL, color=t.FAINT), balance_text,
                                   ft.Text("", size=t.SMALL)], spacing=2),
                        ft.Container(width=20),
                        ft.Container(content=balance_track, expand=True)],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=8)]
            balance_body.visible = True
        memory_text.value = "—" if session.memory is None else f"{session.memory:.2f}"
        memory_text.color = (t.FAINT if session.memory is None
                             else t.TEXT if session.memory >= DFA_FLOOR else t.MUTED)
        memory_note.value = ("" if session.memory is None
                             else f"noise reads {DFA_NOISE:g}")
        balance_text.value = "—" if session.balance is None else f"{session.balance:.2f}"
        balance_text.color = t.FAINT if session.balance is None else t.TEXT

        shapes: list[ft.Control] = [
            ft.Container(width=width, height=4, bottom=3, left=0, border_radius=2,
                         bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
            ft.Container(width=1.5, height=10, bottom=0, left=width / 2, bgcolor=t.MUTED)]
        if session.balance is not None:
            # The bar spans half to one and a half, which is where every published value sits; a value
            # outside it pins to the end rather than rescaling the axis under the reader.
            place = min(1.0, max(0.0, (session.balance - 0.5)))
            shapes.append(ft.Container(width=7, height=7, bottom=1.5, border_radius=4,
                                       left=max(0.0, min(width - 7, place * width - 3.5)),
                                       bgcolor=t.ACCENT))
        balance_track.controls = shapes

        seconds = session.balance_seconds
        if session.memory is None:
            balance_caption.value = (f"the first reading needs {4 * DFA_MIN_SECONDS:g} s of "
                                     f"{CRITICALITY_BAND.label} envelope · {seconds:.0f} s so far")
        else:
            side = ("balanced" if abs(session.balance - BALANCED) < 0.1
                    else "inhibition dominated" if session.balance < BALANCED
                    else "excitation dominated") if session.balance is not None else "—"
            parts = [side, f"mark = {BALANCED:g}",
                     f"from {seconds:.0f} s of {CRITICALITY_BAND.label} envelope"]
            if seconds < CRITICALITY_FIRM:
                parts.append(f"provisional until {CRITICALITY_FIRM:g} s")
            if session.memory < DFA_FLOOR:
                parts.append(f"memory under {DFA_FLOOR:g}, so the balance is a ratio of noise to noise")
            balance_caption.value = " · ".join(parts)

    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        for tgt in session.targets:
            card = cards.get(id(tgt))
            if card is not None:
                card.refresh(session.value_for(tgt), tgt.unit, session.latency_for(tgt))
        signal.refresh()
        spectrum.refresh()
        trends.refresh()
        traces.refresh(session.trace_seconds)
        draw_strength()
        draw_histogram()
        draw_locking()
        draw_balance()
        page.update()

    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)

    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)
