"""Artifact rejection: keeping blinks and clenches out of the measurement.

Our only signal-quality
input before this module was the device's own `valid` flag, which means *no signal* — so a blink, a jaw clench or a cable
tug went straight into band amplitude and drove the feedback. On real electrodes that dominates every
other accuracy question, because the artifacts are an order of magnitude larger than the signal: a
blink is 100-500 µV against 10-40 µV of EEG.

The policy has two halves, and both are load-bearing:

- **Band-limit before judging.** 0.5-50 Hz, so neither mains hum and EMG hash above nor electrode
  drift below sets the threshold that decides what counts as an excursion. `JUDGE_FLOOR_HZ` says why
  the lower edge has to exist.
- **An adaptive threshold, not a fixed µV number.** A fixed number is wrong for
  everybody: 60 µV is unreachable for one person and constant for another, so it either rejects nothing
  or rejects the session.

Three more decisions, each for a reason:

- **The threshold adapts from accepted samples only.** Feeding rejected ones back in means a long
  artifact — a settling electrode, a person scratching their head — raises the threshold until the
  artifact is inside it, and rejection quietly switches itself off exactly when it is needed. This
  is what the arithmetic requires.
- **Absolute bounds around the adaptive number.** Without a floor, a very quiet stretch drags the
  threshold down until ordinary EEG is rejected; without a ceiling, an artifact-laden stretch pushes it
  past any real blink. See the note on `clamp` below, which is a different idea
  wearing a similar word.
- **`hold` rather than skipping each sample.** We work in blocks, so the unit of rejection is a
  block: 8 samples, which is 8 ms at 1000 sps and 32 ms at 250 — granularity below anything a metric
  resolves. But a held block must not simply vanish from the stream, and for a long time it did. Every
  estimator resets its filters when a block's index does not continue the last one, which is right for a
  real acquisition gap — and a skipped block is indistinguishable from one. So each blink re-ran every
  filter's settling time: measured on the waveform protocol, whose chain settles over ~173 s, a blink
  every 12 s left **11 µV RMS of filter transient on a 20 µV wave**, against 0.44 µV when state is
  carried across; the amplitude protocols read 6–30% low depending on the rejection rate. `hold` now
  keeps the stream contiguous instead: `Judgement.fed` carries a stand-in block whose samples are all
  the last accepted value, the filters advance through it, and the block is excluded from *training* —
  targets are not observed and no reward can fire on it. What `continue` protected (the measurement)
  stays protected; what it destroyed (filter state) survives.

`clamp` is **not** a fixed-number threshold, and the difference matters because the word invites the wrong
reading. The two modes keep the adaptive threshold either way and only change what happens to the
sample. That is the more useful pair: skipping against zeroing is a question about whether the filters
keep advancing, and choosing it should not also force you back to a fixed number.

The µV *bounds* above are unrelated to both. They bound an *artifact* threshold. They do not bound a
**training** threshold, and no per-target µV limits are implemented.

**Rejection has to be visible.** A gate silently eating 40% of the data looks exactly like a working
session with a quiet subject, so `rate` and `percent` exist to be put on the screen — beside the *trace*,
since rejection is a property of a channel and several targets can share one.

There is deliberately **no threshold log**. A log of the live threshold would let offline review re-apply
exactly what the session applied, but only a system with two implementations of the gate needs that, to
make them agree. This code has one implementation, so replaying a recording re-derives the identical thresholds — verified on a real file,
271 and 338 rejections reproduced exactly along with all 75 thresholds per channel. A log would be a second
copy of something already reproducible, and the failure mode of those is that they disagree.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..stream import Block, ChannelSpec
from .filters import highpass, lowpass

REJECT_MODES = {
    "hold": "the measurement is fed the last clean value and the block trains nothing",
    "clamp": "the sample is zeroed and measurement carries on",
}

JUDGE_FLOOR_HZ = 0.5
"""The **lower** edge of the band an excursion is judged on, and the reason it has to exist.

`limit_hz` gave the judged band an upper edge and nothing gave it a lower one, so for a long time the band
was "everything the amplifier passes, up to 50 Hz". That worked only because every protocol ran at the
amplifier's 0.5 Hz corner and the hardware supplied the edge — a borrowed constant, and a borrowed constant
breaks when the lender changes. Setting the device to its 0.01 Hz corner for infra-low or coupling work is
the lender changing: electrode drift then enters the judged signal at **hundreds of µV** where the EEG under
it is 7-8, and the gate rejects the session.

Measured on a real 60 s capture at the 0.01 Hz corner, two live channels: 305 and 311 µV mean absolute,
essentially all of it below 0.5 Hz, and the gate held **90.6% and 78.1%** of blocks. With this edge in place,
0.3% and 0.4% — with the same tolerance, floor and ceiling, which is the point. No number needed retuning;
the band was wrong, not the thresholds.

**Not taken from the device's corner**, which is the trap: following the amplifier down to 0.01 Hz would
reproduce the bug exactly. The judged band is a statement about what an artifact looks like, not about what
the amplifier passes, so the detector owns both its edges. 0.5 Hz because it is already this codebase's
boundary for "below this is not EEG" — the amplifier's own corner and the driver's default — and because
nothing the gate must catch lives below it: a blink or a clench is a 100 ms to 1 s event with most of its
energy well above. Verified against eight 300 µV blinks riding on a 400 µV slow wander: all eight still
caught, while the blocks held fell from 5187 to 249. Detection unchanged, collateral damage gone.

**Order 1**, where everything else here is order 2, and it is the one place a single pole is the better
filter. A high-pass smears a large transient over its own time constant, so a blink costs *more* rejected
data than it used to: measured on a 400 µV blink 100 ms long, order 2 held 87 samples over 448 ms against
order 1's 54 over 320, with no difference at all on the real drift (0.3% and 0.5% either way). A single pole
also decays monotonically instead of ringing, and it is what the corner it stands in for physically is — the
amplifier's high-pass is an RC, not a Butterworth.

A constant rather than a field on the policy on purpose. There is no legitimate reason for an operator to
move it, and another box on the signal pane is another thing to set wrong; what was missing was that the
assumption be *stated*, not that it be adjustable.
"""

@dataclass
class ArtifactPolicy:
    """One set of rules, shared by every channel of a session.

    Mutable and read on every block, so the toggle on the screen takes effect immediately rather than
    at the next session — which matters because the moment you want to switch it off is the moment you
    are watching it reject something you think is real.
    """

    enabled: bool = True
    mode: str = "hold"
    limit_hz: float = 50.0          # band-limit before judging
    tolerance: float = 4.0          # multiples of the running mean-absolute level
    tau_seconds: float = 8.0        # how fast the threshold adapts
    floor_uv: float = 30.0          # never reject below this, however quiet the channel
    ceiling_uv: float = 250.0       # always reject above this, however loud it has been

    def describe(self) -> str:
        if not self.enabled:
            return "off — every sample reaches the measurement, artifacts included"
        return (f"{self.tolerance:g}× the running level, {self.floor_uv:g}–{self.ceiling_uv:g} µV, "
                f"judged on {JUDGE_FLOOR_HZ:g}–{self.limit_hz:g} Hz · {REJECT_MODES[self.mode]}")


@dataclass
class Judgement:
    """What the gate decided about one block.

    Two blocks, because a trace and a filter need different things. `block` is always the signal as it
    arrived — an operator who cannot see what was rejected cannot tell a blink from a lost electrode.
    `fed` is what the measurement is given: the same block when accepted, the zeroed rewrite for clamp,
    and for hold a stand-in at the last accepted value so the sample indices stay contiguous and no
    filter mistakes a rejection for an acquisition gap.
    """

    block: Block                # always as it arrived — what a trace shows
    accepted: bool              # False when the block must not train anything
    rejected: int               # samples over threshold
    total: int
    threshold: float            # µV, as applied to this block
    level: float                # µV, the running mean-absolute level behind it
    fed: Block | None = None    # what the measurement is given; defaults to the block itself

    def __post_init__(self) -> None:
        if self.fed is None:
            self.fed = self.block

    @property
    def clean(self) -> bool:
        return self.rejected == 0


class ArtifactGate:
    """One channel's gate. Stateful — it owns a filter and a running level.

    Cost is one `sosfilt` call and a handful of array operations per block, and at 8-sample blocks it is
    almost entirely per-call overhead: **measured 24 µs per block per channel**, so four channels at
    1000 sps spend about 1.2% of a core against an 8 ms budget. That is more than the whole cFIR
    pipeline (0.070 ms per block for sixteen estimators) and for a simple reason: six
    small array calls on eight elements cost more than one large one on two thousand. Worth knowing
    before adding a seventh; not worth optimising, since 1.2% of a core is not the constraint.
    """

    def __init__(self, spec: ChannelSpec, policy: ArtifactPolicy | None = None):
        self.spec = spec
        self.policy = policy if policy is not None else ArtifactPolicy()
        self.level = 0.0
        self.threshold = 0.0
        self.rate = 0.0                     # recent fraction rejected, for the live indicator
        self.rejected = 0                   # session totals, for the summary
        self.seen = 0
        self._primed = False
        self._filters: list = []
        self._filter_hz: float | None = None
        self._held = 0                      # last accepted raw sample, the fill for a held block

    def reset(self) -> None:
        self.level = self.threshold = self.rate = 0.0
        self.rejected = self.seen = 0
        self._primed = False
        self._held = 0
        for filt in self._filters:
            filt.reset()

    @property
    def percent(self) -> float:
        """Fraction of the session rejected on this channel."""
        return self.rejected / self.seen if self.seen else 0.0

    def _band_limited(self, values: np.ndarray) -> np.ndarray:
        """Both edges of the judged band — see `JUDGE_FLOOR_HZ` for why the lower one has to be here.

        Two filters in series rather than one `bandpass`, because the upper edge is conditional and the
        lower one is not: `lowpass` returns None on a channel the device decimated below 100 sps, where
        there is nothing above 50 Hz to remove, while a `bandpass` reaching past Nyquist would raise. So a
        policy that is right at 1000 sps would fail on a 62.5 sps channel — a difference the caller should
        not have to know about, which is the same reasoning `lowpass` documents for returning None.
        """
        if self._filter_hz != self.policy.limit_hz:
            self._filters = [f for f in (highpass(JUDGE_FLOOR_HZ, self.spec.rate_hz, order=1),
                                         lowpass(self.policy.limit_hz, self.spec.rate_hz))
                             if f is not None]
            self._filter_hz = self.policy.limit_hz
        for filt in self._filters:
            values = filt.process(values)
        return values

    def feed(self, block: Block) -> Judgement:
        policy = self.policy
        values = np.asarray(block.values, dtype=float)
        if not policy.enabled or not values.size:
            if values.size:
                self._held = block.samples[-1]
            return Judgement(block, True, 0, values.size, 0.0, self.level)

        judged = np.abs(self._band_limited(values))
        if not self._primed:
            # Start from what the channel is actually doing rather than from zero. Warming up from zero
            # means the threshold sits on its floor for the first several seconds, which rejects
            # ordinary EEG on anyone whose amplitude is above it.
            self.level = float(judged.mean())
            self._primed = True

        threshold = min(policy.ceiling_uv, max(policy.floor_uv, policy.tolerance * self.level))
        self.threshold = threshold
        over = judged > threshold
        rejected = int(np.count_nonzero(over))

        self.seen += values.size
        self.rejected += rejected
        # A one-pole on the per-block fraction: what the indicator shows is "am I rejecting *now*",
        # which the session total stops being able to answer within a minute or two.
        block_seconds = values.size / self.spec.rate_hz
        alpha = min(1.0, block_seconds / 5.0)
        self.rate += alpha * ((rejected / values.size) - self.rate)

        # Accepted samples only — see the module docstring. The clean case avoids the boolean index and
        # its allocation, which is most of the difference on an 8-element array; the block is clean
        # almost always, and at this size a spared array call is worth more than tidier code.
        if not rejected:
            kept_n, kept_mean = values.size, float(judged.mean())
        elif rejected < values.size:
            kept = judged[~over]
            kept_n, kept_mean = kept.size, float(kept.mean())
        else:
            kept_n, kept_mean = 0, 0.0
        if kept_n:
            tau = max(1e-6, policy.tau_seconds)
            step = min(1.0, (kept_n / self.spec.rate_hz) / tau)
            self.level += step * (kept_mean - self.level)

        if not rejected:
            self._held = block.samples[-1]
            return Judgement(block, True, 0, values.size, threshold, self.level)
        if policy.mode == "clamp":
            samples = tuple(0 if bad else s for s, bad in zip(block.samples, over))
            self._held = samples[-1]
            return Judgement(block, True, rejected, values.size, threshold, self.level,
                             fed=dataclasses.replace(block, samples=samples))
        # Held: excluded from training, but the measurement is fed a stand-in at the last accepted
        # value so the stream stays contiguous — a skipped block reads as an acquisition gap
        # downstream and resets every filter, which is the expensive failure the module docstring
        # describes. `_held` is deliberately not updated here.
        stand_in = dataclasses.replace(block, samples=(self._held,) * len(block.samples))
        return Judgement(block, False, rejected, values.size, threshold, self.level, fed=stand_in)


@dataclass
class ArtifactGates:
    """The gates for a session's channels, plus the summary numbers a screen wants.

    Exists so a screen can ask one object "how much is being thrown away" without knowing how many
    channels there are, and so the policy has exactly one home per session.
    """

    policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    gates: dict[int, ArtifactGate] = field(default_factory=dict)

    def for_channel(self, spec: ChannelSpec) -> ArtifactGate:
        gate = self.gates.get(spec.index)
        if gate is None:
            gate = self.gates[spec.index] = ArtifactGate(spec, self.policy)
        return gate

    def feed(self, block: Block) -> Judgement:
        return self.for_channel(block.spec).feed(block)

    def pass_through(self, block: Block) -> Judgement:
        """Accept a block without judging it, and without disturbing the running level.

        For blocks from a channel already resolved unusable — the device's word plus the operator's
        override, which is the caller's to settle. Feeding them to the gate would be worse than
        useless: a railed channel fails the ceiling on every sample, so the whole measurement is held and
        the traces stop, which hides the very thing the operator needs to see. It must not update the
        level either — the point of adapting from accepted samples only is that garbage does not move the
        threshold, and 800 µV of rail is garbage whichever door it comes in by.

        **`fed` is a stand-in here too, for the same reason `hold`'s is.** The raw samples used to go to
        the metrics — the readings were marked invalid and every screen ignored them, but the *filters*
        had still eaten an 800 µV rail, and a slow filter remembers: a reference electrode lifting for
        ten seconds put a rail-sized step into the waveform screen's 25 s-per-pole drift follower, so the
        reading was transient for minutes after the device said the channel was fine again, under a
        `warmup` that said settled (indices stayed contiguous, so no reset fired). The stand-in holds the
        last accepted sample — from when the channel was last valid — so the filters idle through the
        outage and meet the recovered signal with only the signal's own step. Traces still show the rail:
        `block` is untouched, and the readings stay `valid=False` throughout.
        """
        gate = self.for_channel(block.spec)
        stand_in = (dataclasses.replace(block, samples=(gate._held,) * len(block.samples))
                    if block.samples else block)
        return Judgement(block, True, 0, len(block.samples), gate.threshold, gate.level,
                         fed=stand_in)

    def reset(self) -> None:
        for gate in self.gates.values():
            gate.reset()

    @property
    def rate(self) -> float:
        """The worst channel's recent rejection rate — the one that decides whether to intervene."""
        return max((g.rate for g in self.gates.values()), default=0.0)

    @property
    def percent(self) -> float:
        seen = sum(g.seen for g in self.gates.values())
        return sum(g.rejected for g in self.gates.values()) / seen if seen else 0.0

    def describe(self) -> str:
        if not self.policy.enabled:
            return "not rejecting anything"
        if not self.gates:
            return "waiting for data"
        worst = max(self.gates.items(), key=lambda kv: kv[1].rate)
        return (f"rejecting {self.rate:.0%} right now, {self.percent:.1%} of the session · "
                f"worst ch{worst[0]} at {worst[1].threshold:.0f} µV")
