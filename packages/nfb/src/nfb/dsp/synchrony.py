"""Narrowband between-channel synchrony, for Cross-frequency coupling.

CFC's reward is defined by summation: the instantaneous voltages in each channel are added together.
Activity that is synchronous at a particular frequency combines additively and the reward amplitude
increases. Activity 180 degrees out of phase combines subtractively and cancels. So the reward at frequency *f* is the narrowband amplitude of `chA + chB`, and
two equal-amplitude sinusoids differing by Δφ sum to `2A·cos(Δφ/2)` — which is the mechanism and also
the test.

**Why a lock-in and not a bandpass.** The bands are 0.2 Hz wide (4.8-5.2 and 9.8-10.2 Hz by default, and
narrower bands are in use). At 1000 sps that is a bandwidth-to-rate ratio of 2e-4, which is an
ill-conditioned biquad — the kind of filter that works in a notebook and rings or drifts in a long
session. A lock-in instead multiplies by `exp(-j2πft)` and lowpasses the result, which is the same
measurement with none of the conditioning problem: the arithmetic is a complex multiply and a one-pole,
and the bandwidth is set by the one-pole's time constant. It also hands back **phase**, which a real
bandpass cannot, so both amplitude measures below come out of one pass.

**The bandwidth is a delay, and no filter design changes that.** A one-pole with time constant τ has a
two-sided -3 dB bandwidth of `1/(πτ)`, so a 0.2 Hz band means τ = 1.6 s and a group delay of about the
same. Nothing recovers that: the envelope of a 0.2 Hz-wide process cannot move faster than a few seconds
whatever estimates it. cFIR does not help here, because the limit is bandwidth rather than causality.
The 250 ms end-to-end target in AGENTS.md was derived for alpha *spindles* and does not apply to this
protocol — CFC trains a state that persists over seconds and reads it off percentages and slow
instruments, which is consistent rather than a compromise.

**Two measures, and the difference is not cosmetic.**

- `sum` is the summed-voltage measure, and it conflates synchrony with amplitude twice over. Two channels both getting
  louder raise the sum without aligning at all — and, less obviously, an *imbalance* between the two
  channels' amplitudes **in the reward's own band** makes it stop responding to phase at all: see
  `SynchronyMetric.sensitivity`. The protocol closes that hole with the inhibit bins — the bin edges
  are 1-5-10-15-20-25-45, so each reward frequency sits exactly on a bin boundary and is being pushed
  *down* by two bins while the sum is pushed up, and the only way to satisfy both at once is phase
  alignment. That interlock is the design; it is not an accident to be tidied away.
- `plv` is the phase-locking value, amplitude-invariant by construction, so it needs no interlock and is
  the measure to reach for when comparing across sessions. It is *not* what the usual thresholds and the
  5-15% conventions refer to, so `sum` stays the default and the screen says which is running.

**`sum` is what CFC trains. `plv` is an addition, and measuring it confirmed the default rather than
challenging it.** A phase-locking value is a mean over independent phase estimates, and a 0.2 Hz-wide
band produces 0.2 of those per second. Averaged over the band's own coherence time it reads near 1 for
*anything* — measured: two independent white-noise channels scored 0.73 at a 1 Hz band, and 0.15 once the
window was widened to the shipped `PLV_CYCLES`. The bias of an exponential mean goes as
`1/sqrt(2·tau·bandwidth)`, so an honest PLV on a 0.2 Hz band needs tens of seconds of averaging, which is
slower than the state being trained. `PLV_CYCLES` sets that window in
coherence times and the screen prints the figure, because a PLV whose window is not stated is not a
number. The sum has no such problem — it averages amplitude, not phase, and phase enters
instantaneously through the cancellation — so it is the better choice on its own terms and stays
the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stream import Block, ChannelSpec
from .filters import Band
from .metrics import Reading

MEASURES = {
    "sum": "the amplitude of the two channels added together — in phase adds, antiphase cancels",
    "plv": "phase locking, independent of how large either channel is",
}
DEFAULT_WIDTH_HZ = 0.2
DEFAULT_POLES = 2
PLV_CYCLES = 8.0        # coherence times to average phase over; see the note in the module docstring


def bandwidth_tau(width_hz: float, poles: int = DEFAULT_POLES) -> float:
    """Per-pole time constant for `poles` cascaded one-poles at a two-sided -3 dB bandwidth of
    `width_hz`.

    `(1 + (ωτ)²)^poles = 2` at the corner, so `ωτ = sqrt(2**(1/poles) - 1)`.
    """
    corner = np.sqrt(2.0 ** (1.0 / max(1, poles)) - 1.0)
    return float(corner / (np.pi * max(width_hz, 1e-6)))


class Lockin:
    """Amplitude and phase at one frequency, by complex heterodyne.

    `process` returns one complex value per input sample, scaled so that `abs(z)` is the **peak**
    amplitude of a sinusoid at the centre frequency — the same convention cFIR reports, and not the RMS
    one, so a threshold set against `CFIRBandPower` transfers and one set against
    `BandPower(envelope="rms")` does not.

    **Two poles, not one, and this was measured rather than assumed.** This is a choice about the
    *filter*, not about the protocol: the quantity is still the narrowband amplitude of the sum, which is
    what CFC trains. Only the skirt steepness and the settling time are open to choice, because any bandpass
    reaches the same quantity.

    A single one-pole rolls off at 6 dB/octave, which is far too shallow for a band this narrow: at
    250 sps a 0.2 Hz lock-in at 10 Hz still passed **5%** of a 50 µV tone sitting at 12 Hz — so a strong
    alpha would have leaked into the theta reward and been rewarded as synchrony. Two cascaded poles at
    12 dB/octave cut that to 0.6%, and the cost is only **29% more delay** (2.05 s against 1.59 s at
    0.2 Hz), because holding the bandwidth fixed shortens each pole's time constant. It also all but
    removes the 2f ripple — the mixer's image at twice the centre frequency, which at one pole showed up
    as a 2.5% wobble on a settled reading. Cheap for what it buys, so it is the default.

    The cascade runs as an explicit loop over the block. That is deliberate and measured: at our
    8-sample blocks a `lfilter` call costs 3.07 µs of entry to do 8 multiply-accumulates, against
    ~150 ns/sample for the loop. See the note in AGENTS.md before "optimising" it.
    """

    def __init__(self, freq_hz: float, width_hz: float, rate_hz: float,
                 poles: int = DEFAULT_POLES):
        if freq_hz <= 0 or freq_hz >= rate_hz / 2:
            raise ValueError(f"{freq_hz} Hz is not measurable at {rate_hz:g} sps")
        self.freq_hz = float(freq_hz)
        self.width_hz = float(width_hz)
        self.rate_hz = float(rate_hz)
        self.poles = max(1, int(poles))
        self.tau = bandwidth_tau(width_hz, self.poles)
        dt = 1.0 / self.rate_hz
        self.alpha = dt / (self.tau + dt)
        self._increment = 2 * np.pi * self.freq_hz / self.rate_hz
        self._phase = 0.0
        self._z = [0.0 + 0.0j] * self.poles

    @property
    def delay_seconds(self) -> float:
        """Group delay at the centre frequency: the cascade's time constants, summed."""
        return self.tau * self.poles

    def reset(self) -> None:
        self._z = [0.0 + 0.0j] * self.poles
        # the local oscillator's phase is deliberately *not* reset: it is a clock, and restarting it
        # would rotate every subsequent phase difference by an arbitrary amount

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if not n:
            return np.zeros(0, dtype=complex)
        angles = self._phase + self._increment * np.arange(1, n + 1)
        self._phase = float(angles[-1] % (2 * np.pi))
        mixed = np.asarray(x, dtype=float) * np.exp(-1j * angles)
        out = np.empty(n, dtype=complex)
        z, a = self._z, self.alpha
        for i in range(n):
            value = mixed[i]
            for p in range(self.poles):
                z[p] += a * (value - z[p])
                value = z[p]
            out[i] = value
        self._z = z
        return 2.0 * out           # (A/2)·e^{jφ} after mixing, so double it back to peak amplitude


class ChannelAligner:
    """Lines up any number of channels' blocks into equal-length runs of the same sample indices.

    Blocks arrive one channel at a time, so anything that combines channels — a sum, a difference, a
    between-channel measurement — has to line them up by absolute index first. Anything not yet overlapped
    on every channel is held.

    A gap on any one channel **re-anchors that channel** rather than pairing samples that are not
    simultaneous. That is the whole reason this is a class rather than three lines at the call site: the
    alternative silently turns a dropped sample into a permanent misalignment, which for a summed signal
    reads as a change in the thing being measured.
    """

    def __init__(self, channels):
        self.channels = tuple(channels)
        self._start: dict[int, int | None] = {ch: None for ch in self.channels}
        self._values: dict[int, list[float]] = {ch: [] for ch in self.channels}

    def reset(self) -> None:
        for ch in self.channels:
            self._start[ch] = None
            self._values[ch] = []

    def push(self, block: Block) -> tuple[int, list[np.ndarray]] | None:
        """The next run every channel now covers, as `(start_index, [values per channel])`."""
        ch = block.channel
        if ch not in self._start:
            return None
        start, values = self._start[ch], self._values[ch]
        if start is None or block.start_index != start + len(values):
            start, values = block.start_index, []      # first block, or a discontinuity
        self._start[ch] = start
        self._values[ch] = values + list(block.values)

        if any(self._start[c] is None for c in self.channels):
            return None
        lo = max(self._start[c] for c in self.channels)
        hi = min(self._start[c] + len(self._values[c]) for c in self.channels)
        if hi <= lo:
            return None
        out = []
        for c in self.channels:
            s = self._start[c]
            out.append(np.asarray(self._values[c][lo - s:hi - s], dtype=float))
            self._start[c], self._values[c] = hi, self._values[c][hi - s:]
        return lo, out


class PairAligner(ChannelAligner):
    """Pairs up two channels' blocks into equal-length runs of the same sample indices.

    The two-channel case of `ChannelAligner`, kept because it returns the two runs unpacked — every
    between-channel measure here takes an `a` and a `b`, and `(start, [a, b])` at those call sites reads
    worse than the tuple.
    """

    def __init__(self, a: int, b: int):
        super().__init__((a, b))

    def push(self, block: Block) -> tuple[int, np.ndarray, np.ndarray] | None:
        got = super().push(block)
        if got is None:
            return None
        start, (a, b) = got
        return start, a, b


@dataclass(frozen=True)
class Reward:
    """One CFC synchrony reward: where it listens and how it judges.

    The band must not reach zero. That is not pedantry: a reward at 0.1 Hz with the default 0.2 Hz width
    spans `0.0-0.2`, and a band whose lower edge is DC is not a band — the lock-in would be measuring the
    mean. It matters because he does train that low ("don't be surprised if you train as low as 0.01 -
    0.1 Hz for some cases"), so this is a case that arises rather than a theoretical one, and the caller has
    to narrow the width with the centre.
    """

    freq_hz: float
    width_hz: float = DEFAULT_WIDTH_HZ
    measure: str = "sum"
    name: str = ""

    def __post_init__(self) -> None:
        if self.freq_hz <= 0:
            raise ValueError(f"a reward frequency must be positive, got {self.freq_hz}")
        if self.low <= 0:
            raise ValueError(
                f"a {self.width_hz:g} Hz band at {self.freq_hz:g} Hz reaches DC — narrow the width to "
                f"less than {2 * self.freq_hz:g} Hz")

    @property
    def label(self) -> str:
        return self.name or f"{self.freq_hz:g} Hz sync"

    @property
    def low(self) -> float:
        return self.freq_hz - self.width_hz / 2

    @property
    def high(self) -> float:
        return self.freq_hz + self.width_hz / 2

    def describe(self) -> str:
        return f"{self.low:g}–{self.high:g} Hz · {MEASURES[self.measure]}"


class SynchronyMetric:
    """One reward frequency over one channel pair. Satisfies the `Metric` protocol.

    Both channels are locked in separately, which costs nothing extra and buys three things: the sum is
    `|z_a + z_b|` because a lock-in is linear, the PLV is available from the two phases, and the
    *per-channel* narrowband amplitudes are there for the display — which matters because the operator
    needs to see whether a rising sum is alignment or just more power.
    """

    def __init__(self, specs: dict[int, ChannelSpec], pair: tuple[int, int], reward: Reward,
                 output_hz: float = 20.0):
        if pair[0] not in specs or pair[1] not in specs:
            raise ValueError(f"channels {pair} are not both in this session")
        rates = {specs[c].rate_hz for c in pair}
        if len(rates) != 1:
            raise ValueError("a synchrony pair needs both channels at the same sample rate")
        self.specs = specs
        self.pair = pair
        self.reward = reward
        self.rate_hz = rates.pop()
        self.aligner = PairAligner(*pair)
        self.lockins = {c: Lockin(reward.freq_hz, reward.width_hz, self.rate_hz) for c in pair}
        self.stride = max(1, int(round(self.rate_hz / output_hz)))
        self._phasor = 0.0 + 0.0j        # running mean of exp(j·Δφ), for the PLV
        # Phase averages over coherence times, not over the band's own time constant — at the band's own
        # constant the PLV reads near 1 for independent channels. See the module docstring.
        self.plv_tau = PLV_CYCLES / max(reward.width_hz, 1e-6)
        dt = 1.0 / self.rate_hz
        self._alpha = dt / (self.plv_tau + dt)
        self._next_index: int | None = None
        self.parts: dict[int, float] = dict.fromkeys(pair, 0.0)
        self.value = 0.0
        self.plv = 0.0

    @property
    def name(self) -> str:
        return f"sync_{self.reward.freq_hz:g}"

    @property
    def sensitivity(self) -> float:
        """What fraction of this reward's reading phase alignment actually controls, 0…1.

        The sum measure responds to phase only to the extent that the two channels are *equally loud in
        this reward's band*. Write `A_a` and `A_b` for the two per-channel narrowband amplitudes at the
        reward frequency — `self.parts`, in µV, and **not** the channels' broadband amplitude and nothing
        to do with sample rate. In phase the sum reads `A_a + A_b`; antiphase it reads `|A_a - A_b|`. So
        the part of the reading that phase moves is

            2·min(A_a, A_b) / (A_a + A_b)

        1.0 when the two are equal, 0.4 when one is four times the other, 0.22 at eight times. Closed form
        verified against a numeric sweep.

        This is a property of the measure rather than of this implementation, but it is the one thing
        about it an operator can act on: at four to one the reward is mostly reporting the louder channel's
        amplitude, and the fix for that is an electrode rather than a threshold.
        """
        values = [v for v in self.parts.values()]
        total = sum(values)
        if total <= 0 or len(values) != 2:
            return 0.0
        return 2.0 * min(values) / total

    @property
    def balance(self) -> float:
        """How many times louder the louder channel is *in this reward's band*, 1.0 when they match.

        A ratio of the two `parts` — narrowband µV against narrowband µV. Not a rate, not a ratio of
        anything per second.
        """
        values = sorted(v for v in self.parts.values())
        if len(values) != 2 or values[0] <= 0:
            return float("inf") if any(values) else 1.0
        return values[1] / values[0]

    @property
    def louder_channel(self) -> int | None:
        """Which channel is carrying more of this band, for a screen to name."""
        if not any(self.parts.values()):
            return None
        return max(self.parts, key=self.parts.get)

    @property
    def delay_seconds(self) -> float:
        return self.lockins[self.pair[0]].delay_seconds

    @property
    def smoothing_seconds(self) -> float:
        """For `sum`, none: the bandwidth *is* the averaging. For `plv`, the phase window, which is
        long and has to be admitted to rather than buried."""
        return self.plv_tau if self.reward.measure == "plv" else 0.0

    def describe(self) -> str:
        detail = (f"phase averaged over {self.plv_tau:.0f} s" if self.reward.measure == "plv"
                  else f"{self.delay_seconds:.1f} s delay")
        return f"{self.reward.describe()} · {detail}"

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds

    def feed(self, block: Block) -> list[Reading]:
        paired = self.aligner.push(block)
        if paired is None:
            return []
        start, a_values, b_values = paired
        za = self.lockins[self.pair[0]].process(a_values)
        zb = self.lockins[self.pair[1]].process(b_values)
        summed = np.abs(za + zb)
        self.parts = {self.pair[0]: float(np.abs(za[-1])), self.pair[1]: float(np.abs(zb[-1]))}

        # PLV: the running mean of the unit phasor, over `PLV_CYCLES` coherence times — deliberately
        # much longer than the band's own constant, or it reads near 1 for anything (see the module
        # docstring). One control still sets both timescales, since the width is in the denominator.
        unit = np.exp(1j * (np.angle(za) - np.angle(zb)))
        p, alpha = self._phasor, self._alpha
        for value in unit:
            p += alpha * (value - p)
        self._phasor = p
        self.plv = float(min(1.0, abs(p)))

        if self._next_index is None:
            self._next_index = start
        out: list[Reading] = []
        for offset in range(len(a_values)):
            index = start + offset
            if index < self._next_index:
                continue
            self._next_index = index + self.stride
            self.value = (float(summed[offset]) if self.reward.measure == "sum"
                          else float(min(1.0, abs(p))))
            out.append(Reading(name=self.name, value=self.value,
                               t=index / self.rate_hz, channel=self.pair,
                               unit="µV" if self.reward.measure == "sum" else "",
                               valid=block.valid))
        return out


class LockinBandPower:
    """One channel's amplitude in one band, by lock-in. Satisfies the `Metric` protocol.

    **Why not the two estimators that already exist**, for ILF Synchrony's 0.1–40 Hz inhibit ladder:

    - `CFIRBandPower` designs 250 taps, which at 1000 sps is 250 ms of history. The lowest band is 0.1–2 Hz
      and the filter has never seen a quarter of a cycle of its bottom edge. That one is impossible, not
      merely awkward.
    - `BandPower`'s biquads would in fact work here — a 0.1–2 Hz bandpass at 1000 sps has a bandwidth-to-rate
      ratio of 1.9e-3, an order of magnitude better conditioned than the 2e-4 `Lockin`'s docstring calls
      ill-conditioned. So this is **not** a case of impossibility, and claiming it was would be overreach.

    What a lock-in buys is uniform conditioning across the whole ladder: it is as well behaved at 0.1 Hz as
    at 40, because the bandwidth is a one-pole's time constant rather than a pole placement near the unit
    circle. Nine bands that have to be comparable *to each other* get one estimator rather than a
    well-conditioned one at the top and a marginal one at the bottom.

    The cost is the same one CFC pays: about a time constant per pole of delay, 108 ms for a 1.9 Hz-wide
    band. Bandwidth is delay; nothing here is a workaround for that.

    Amplitude is `|z|`, the **peak** convention, so it matches `CFIRBandPower` and `Lockin` and does not
    match an RMS reading of the same signal. See `Lockin`.
    """

    def __init__(self, spec: ChannelSpec, band: Band, output_hz: float = 20.0,
                 poles: int = DEFAULT_POLES):
        centre = (band.low + band.high) / 2
        width = band.high - band.low
        if width <= 0:
            raise ValueError(f"{band.low:g}–{band.high:g} Hz is not a band")
        self.spec = spec
        self.band = band
        self.rate_hz = spec.rate_hz
        self.lockin = Lockin(centre, width, self.rate_hz, poles=poles)
        self.stride = max(1, int(round(self.rate_hz / output_hz)))
        self._next_index: int | None = None
        self.value = 0.0

    @property
    def name(self) -> str:
        return f"lockin_{self.band.low:g}-{self.band.high:g}"

    @property
    def delay_seconds(self) -> float:
        return self.lockin.delay_seconds

    @property
    def smoothing_seconds(self) -> float:
        """None. The bandwidth *is* the averaging, as it is for `sum` — there is no envelope stage to set."""
        return 0.0

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds

    def feed(self, block: Block) -> list[Reading]:
        # `Block.values` are already scaled to µV — every estimator scales exactly once, either here
        # via `values` or from `samples` with an explicit `* spec.scale`, never both.
        values = np.asarray(block.values, dtype=float)
        z = np.abs(self.lockin.process(values))
        if self._next_index is None:
            self._next_index = block.start_index
        out: list[Reading] = []
        for offset in range(len(values)):
            index = block.start_index + offset
            if index < self._next_index:
                continue
            self._next_index = index + self.stride
            self.value = float(z[offset])
            out.append(Reading(name=self.name, value=self.value, t=index / self.rate_hz,
                               channel=block.channel, unit="µV", valid=block.valid))
        return out
