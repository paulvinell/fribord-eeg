"""Two channels multiplied together: ILF Synchrony's reward.

**One instrument, one frequency at a time.** The measure takes a single reward frequency, so this is not a
family of rewards at different bands — it is one measure you retune. Infra-slow, alpha and gamma are three settings of it, not three instances.

The measure:

```
Ch1, Ch2 ──→ one Butterworth each, same centre / bandwidth / order
         ──→ pos(x * y):  (In1*In2)>0 ? (In1*In2) : 0        ← the whole measure
         ──→ Average (Post Filter, 100 ms)
```

The three traces on screen are `Ch1`, `Ch2` and `Ch1 x Ch2 if >0`.

**This is a different quantity from `synchrony.SynchronyMetric`, not a rearrangement of it.** That one adds
the channels and measures the amplitude of the sum: µV, largest when both are loud and in phase. This
multiplies the two *waveforms* and discards the negative part: **µV²**, and signed before rectification, so it
is zero whenever the two sites are on opposite sides of their own baselines however loud they are. Two
antiphase channels at 40 and 10 µV leave a summed reward reporting 30 µV of standing signal and leave this at
exactly zero. Separate module because they share nothing but the aligner.

**The filter forks to a bare lowpass, and that fork is the whole protocol at ILF frequencies.** The rule:

```
lowpass  when (centre - bw/2 - 0.3) <= 0, with corner = centre * 2
bandpass otherwise, from centre - bw/2 to centre + bw/2
```

Two details that are easy to miss and both matter. The fork carries a **0.3 Hz margin**, so with a 3 Hz
bandwidth it trips at 1.8 Hz rather than 1.5. And the lowpass corner is **twice the centre**, not
`centre + bw/2`. With a bandwidth of 3 Hz:

| centre | band | what it is |
|---|---|---|
| 40 Hz | 38.5–41.5 | bandpass |
| 10 Hz | 8.5–11.5 | bandpass |
| 1.8 Hz | — | where the fork trips |
| 0.05 Hz | 0–0.1 | lowpass at twice the centre |

So "synchrony at 0.05 Hz" is the product of two slow *waveforms* band-limited from DC to 0.1 Hz — ILF training
on the raw waveform, and constant-Q rather than constant-width at the slow end. It is a **narrow** band all the
same: 0.1 Hz yields about six independent values a minute, so a 30 s epoch against a 10% aim is thin down
there, and that is a property of the measure rather than something to work around.

**A Butterworth, not a lock-in**, with an order of 1 or 2. It also hands back the waveform
directly rather than needing an analytic signal remodulated.

Conditioning is not the reason and it would be wrong to claim it: measured, a 0.1 Hz order-2 lowpass has its
worst pole at `1-|p|` = 1.8e-3 at 250 sps and 2.2e-4 at 2000 sps, against float64's 2.2e-16 — nine orders of
margin either way. The gain at 0.05 Hz (0.9701) and the settling time (17.0 s) are **identical** at 125, 250,
1000 and 2000 sps, as a rate-independent response should be. So the sample rate does not help or hurt this
measurement, and 250 sps is right for the screen only because the gamma preset needs 46 Hz, which rules out
125.

**Rectify *then* average, and the order is the measure.** Averaging first and rectifying after gives
`max(0, mean)`, which is a much less selective number: a pair that spends half its time aligned and half
anti-aligned averages to nearly zero and then rectifies to nearly zero, where rectify-first credits the
aligned half. The label on screen says which is meant.
"""

from __future__ import annotations

import numpy as np

from ..stream import Block, ChannelSpec
from .filters import bandpass, bandstop, highpass, lowpass, Band
from .metrics import Reading
from .synchrony import PairAligner

DEFAULT_BANDWIDTH_HZ = 3.0
"""The bandwidth of the synchrony filter pair, in hertz: adjustable from 1 to 3, set to 3.0. **A fixed number of
hertz, not a fraction of the centre** — which is what makes the lowpass fork happen at low frequencies rather
than being a case to design around."""

DEFAULT_ORDER = 2
"""The filter order, 1 to 2, set to 2."""

DEFAULT_POST_SECONDS = 0.1
"""The post filter, 1–1000 ms, set to 100. It sits between the rectified product and everything
downstream, and it belongs to the measurement rather than to a screen: at 10 Hz it spans one cycle of the
product's own 2f ripple and removes most of it, and a measure whose smoothing is set somewhere else is not
one number."""

FORK_MARGIN_HZ = 0.3
"""The margin in the fork test: `(centre - bandwidth/2 - 0.3) <= 0` selects lowpass, not
`(centre - bandwidth/2) <= 0`. With a 3 Hz bandwidth that trips at **1.8 Hz**, not 1.5."""

THETA_NOTCH = Band(4.0, 6.0, "theta notch")
"""The theta notch is a toggle, **on by default**, driving Butterworth order-2 bandstops at 4–6 Hz. The
product is computed on the notched channels, which means an unnotched product measures something
different at any centre near theta."""


SLOW_BANDS: dict[str, tuple[float, float]] = {
    "0–2×": (0.0, 2.0),
    "¼–2×": (0.25, 2.0),
    "½–2×": (0.5, 2.0),
    "½–1½×": (0.5, 1.5),
}
"""What the band becomes below the fork, as multiples of the centre. **A choice, because our amplifier cannot supply a lower edge by itself.**

`0–2×` is the lowpass corner (`centre * 2`) with no lower edge, which suits an amplifier whose hardware
supplies the lower edge.

**That shape assumes the hardware supplies the lower edge, and ours cannot.** The Optima offers 0.5 Hz, which
annihilates a 0.05 Hz band, and 0.01 Hz, which lets electrode drift through at full amplitude — nothing in
between, and in between is where the edge for a 0.01–0.05 Hz band belongs. Measured on a real capture: both
channels sat near −7.4 µV, 99% of the reward's variance was the offset-driven linear terms, and the product
went negative **0%** of the time where the two waves were on opposite sides of their baselines 16% of it. The
rectifier that is the whole reason to use a product never fired.

The `½` in the two `½` options is a low cut of `centre − bandwidth/2` with bandwidth = centre, chosen for
frequencies this low. `½–1½×` is that band entire, high cut included (`centre + bandwidth/2`); `½–2×`
keeps the lowpass corner at twice the centre and takes only the low cut.

Nothing is removed: `0–2×` remains available.

**`¼–2×` is the default — it is where the group delay is.** The edge is the biggest
single term in this screen's latency: at `centre/2` its delay at the trained frequency equals the whole
lowpass's, so the reward arrived about 96° into a 0.05 Hz cycle rather than 48°. The instinct is to drop the
*order*, and a first-order lower edge looks like a saving.

Computed on the designs themselves at 250 sps, centre 0.05 Hz. **Every delay is the group delay at the
centre** — the edge's own contribution and the whole cascade's total — because a sum of delays quoted at
different frequencies is not a number, and an earlier version of this table quoted each edge at its own
corner-derived point, which inverted the ranking `delay_seconds` reported to the screen. "Keep" is the
cascade's gain at the centre; the drift columns are the *edge's* leakage at drift frequencies:

| lower edge | edge delay | total | keep | drift 0.005 Hz | drift 0.001 Hz |
|---|---|---|---|---|---|
| order 2 @ centre/2 *(was the default)* | 2.65 s | 5.30 s | 94.1% | 4.0% | 0.2% |
| order 1 @ centre/2 | 1.27 s | 3.92 s | 86.8% | 19.6% | 4.0% |
| **order 2 @ centre/4** *(the default)* | **1.19 s** | **3.84 s** | **96.8%** | 15.8% | 0.6% |
| order 1 @ centre/4 | 0.75 s | 3.40 s | 94.1% | 37.1% | 8.0% |
| order 1 @ 0.056 Hz | 1.58 s | 4.23 s | 64.6% | 8.9% | 1.8% |

**Order 1 at centre/2 is dominated by order 2 at centre/4 on every column**, including the two the order
change was supposed to win: less delay *and* several times better rejection of slow drift, because halving
the corner buys back more than the second pole costs. First order was implemented, measured against this
table and reverted; the order stays at `order` and the corner is the knob.

**It costs two things, and both are real trades rather than free savings.**

*Drift rejection.* Against `centre/2` it gives up a factor of four at 0.001 Hz — 0.16% through becomes 0.64% —
which is the timescale electrode drift actually lives on. With the 305 µV measured on this amplifier at the
0.01 Hz corner, that is 0.5 µV of residual becoming 2 µV, against 7–8 µV of signal. Chosen deliberately: 1.5 s
on a 20 s cycle is a quarter of the reward's phase, and the residual is still under a third of the signal.

*Settling, which doubles.* Measured on the built cascade at 0.05 Hz, the step transient is gone by **66 s**
against 35 s at `centre/2` — a corner further from the band has a longer time constant, so the same move that
buys delay costs warm-up. The 30 s threshold window sits on top of that, so the first minute and a half of a
session is the instruments arriving rather than a session. `ilf_view` already carries the consequence for the
plots: the transient is the largest thing in every window while it lasts, which is why the scales are settable.

If a capture ever shows the rectifier failing, `½–2×` is one pill away and these are the numbers to come back
to."""

DEFAULT_SLOW_BAND = "¼–2×"
"""Which of `SLOW_BANDS` a session starts on. Not `0–2×`, because a band reaching DC on this amplifier makes
the product measure something other than what it is for — see above. Keeps **96.8%** of the trained wave
through the whole cascade, puts drift a decade below at about 16%, and carries **3.84 s** of group delay at
the centre against the 5.30 s of `½–2×` — the fastest of the lower-edged options, not the slowest, which is
what an earlier delay figure quoted at the wrong frequency made it look like."""


def edges(centre_hz: float, bandwidth_hz: float,
          slow_band: str = DEFAULT_SLOW_BAND) -> tuple[float, float]:
    """`(low, high)` of the band the design actually builds:

    ```
    lowpass corner = centre * 2
    bandpass       = centre - bw/2  to  centre + bw/2
    ```

    **The lowpass corner is twice the centre, not `centre + bandwidth/2`.** So a 0.05 Hz setting is a
    0–0.1 Hz lowpass, and getting that wrong is what made an earlier version claim the slow case was a wide
    0–1.55 Hz band with plenty of independent values behind its threshold. It is not: 0.1 Hz of bandwidth is
    about six independent values a minute, so a 30 s epoch against a 10% aim really is thin down there.
    The rule is constant-Q rather than constant-width, which is the sensible thing for a frequency that spans
    decades — and it is a *third* bandwidth rule in the same product.
    """
    if is_lowpass(centre_hz, bandwidth_hz):
        lo, hi = SLOW_BANDS.get(slow_band, SLOW_BANDS[DEFAULT_SLOW_BAND])
        return centre_hz * lo, centre_hz * hi
    return centre_hz - bandwidth_hz / 2.0, centre_hz + bandwidth_hz / 2.0


def is_lowpass(centre_hz: float, bandwidth_hz: float) -> bool:
    """Whether the design degrades to a bare lowpass — `(centre - bandwidth/2 - 0.3) <= 0`.

    An amplifier with no hardware corner would need a software high-pass in the lowpass path. Ours has a
    hardware corner, so the lowpass path takes the raw signal — and `ILFSession.wiring_problem` is where that corner has to be checked, because in this
    branch it becomes the band's *only* lower edge.
    """
    return centre_hz - bandwidth_hz / 2.0 - FORK_MARGIN_HZ <= 0.0


class ProductSynchrony:
    """`pos(ch1 × ch2)` over one channel pair at one frequency. Satisfies the `Metric` protocol.

    Exposes the two filtered waveforms and the absolute index they start at, because on this protocol they
    are not a debugging aid: the slow signal is what an ILF state is read off, and a Lissajous figure of the
    two is the picture of the reward. They come from the *same* filters the reward uses, so the display and
    the reward cannot disagree. A second filter pair with its own bandwidth and
    order controls for the Lissajous would let them disagree.
    """

    def __init__(self, specs: dict[int, ChannelSpec], pair: tuple[int, int], centre_hz: float,
                 bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ, order: int = DEFAULT_ORDER,
                 post_seconds: float = DEFAULT_POST_SECONDS, output_hz: float = 20.0,
                 theta_notch: bool = True, slow_band: str = DEFAULT_SLOW_BAND):
        if pair[0] not in specs or pair[1] not in specs:
            raise ValueError(f"channels {pair} are not both in this session")
        rates = {specs[c].rate_hz for c in pair}
        if len(rates) != 1:
            raise ValueError("a synchrony pair needs both channels at the same sample rate")
        self.specs = specs
        self.pair = pair
        self.centre_hz = float(centre_hz)
        self.bandwidth_hz = float(bandwidth_hz)
        self.order = max(1, min(2, int(order)))
        self.rate_hz = rates.pop()
        self.slow_band = slow_band if slow_band in SLOW_BANDS else DEFAULT_SLOW_BAND
        self.low, self.high = edges(self.centre_hz, self.bandwidth_hz, self.slow_band)
        self.lowpass_only = is_lowpass(self.centre_hz, self.bandwidth_hz)
        if self.high >= self.rate_hz / 2:
            raise ValueError(f"{self.high:g} Hz is above Nyquist at {self.rate_hz:g} sps")
        self.filters = {c: self._design() for c in pair}
        if any(not f for f in self.filters.values()):
            raise ValueError(f"no filter for {self.describe()} at {self.rate_hz:g} sps")
        # Ahead of the band filter, on each channel, exactly where `Ch1notch`/`ch2notch` sit — the synchrony
        # module receives notched signals, so the product is computed on them. `None` when the rate cannot
        # represent 6 Hz, which is not a case this protocol reaches but is not worth crashing on.
        # **Never notch out the band being trained.** The notch removes 4–6 Hz, and the frequency control
        # reaches into it — set the centre to 5 Hz and the band 3.5–6.5 Hz is 99% deleted, so the reward reads
        # near zero with nothing on screen explaining it. Measured: a 5 Hz tone through this bandstop comes out
        # at 1.0% of its amplitude, and the reward is a *product*, so 0.01% of its scale.
        #
        # Suppressed rather than warned about, because there is no reading of "notch out what you are
        # measuring" that anybody wants, and `describe` says which case is running either way.
        self.notch_suppressed = bool(theta_notch) and self.high > THETA_NOTCH.low and \
            self.low < THETA_NOTCH.high
        self.theta_notch = bool(theta_notch) and not self.notch_suppressed
        self.notches = ({c: bandstop(THETA_NOTCH, self.rate_hz, order=2) for c in pair}
                        if self.theta_notch else {c: None for c in pair})
        # The **second** reward of the synchrony mode, on the two channels *summed*, with its own success
        # percentage (60), epoch and bar. Same band, so it shares the design.
        self.sum_filter = self._design()
        self._sum_post = 0.0
        self.amplitude = 0.0
        self.aligner = PairAligner(*pair)
        self.stride = max(1, int(round(self.rate_hz / output_hz)))
        self._next_index: int | None = None
        self.post_seconds = max(0.0, float(post_seconds))
        dt = 1.0 / self.rate_hz
        self._post_alpha = 1.0 if self.post_seconds <= 0 else dt / (self.post_seconds + dt)
        self._post = 0.0
        self.value = 0.0
        self.waveforms: dict[int, np.ndarray] = {c: np.zeros(0) for c in pair}
        self.wave_start: int | None = None
        """Absolute sample index of `waveforms[…][0]`, so a consumer can decimate at its own rate. Anything
        that decimated once per call would make its point rate a function of the block size and the channel
        count, which is the discipline every estimator here follows."""
        self.parts: dict[int, float] = dict.fromkeys(pair, 0.0)
        """Per-channel RMS of the filtered waveform. Not part of the measure — it is what says a product is
        being dominated by one channel, which is a fault an electrode fixes and a threshold cannot."""
        self.samples_seen = 0
        """Aligned samples fed so far, for `warmup`. Counts pairs the aligner has matched, because settling
        is a property of the filters and only matched samples reach them."""
        self.sum_wave = 0.0
        """The signed summed slow wave, µV — the latest sample of `filter(ch_a + ch_b)`, before the
        rectifier that turns it into `amplitude`. This is the feedback quantity for the sum: the
        Shepard tone follows the signed waveform's position, not an
        envelope, and a rectified 0.05 Hz wave is not that idiom. Computed anyway for the amplitude
        reward; kept here instead of discarded."""

    def _design(self):
        """The band, as stages applied in order.

        A list because the slow branch can be two filters — a high-pass at `low` and a lowpass at `high` —
        where `0–2×` has no lower edge and stays one. `bandpass` is not used for the slow case even when
        both edges exist, because the two corners can sit two octaves apart and a cascaded pair suits it.

        Both edges carry `order`. Dropping the lower edge to first order was measured and **rejected**:
        it is dominated by halving the *corner* instead. See `SLOW_BANDS`.
        """
        if not self.lowpass_only:
            return [bandpass(Band(self.low, self.high, ""), self.rate_hz, order=self.order)]
        # `at_hz=centre` on both stages: the delay these report is summed into `delay_seconds`, and a
        # sum only means something when every term is quoted at the same frequency — the one being
        # trained. Left to its default, each stage quoted its delay at its own corner-derived guess,
        # and the summed figure inverted the ranking of the slow-band options: `¼–2×` was reported as
        # the slowest lower-edged choice when at the centre it is the fastest. See `SLOW_BANDS`.
        stages = []
        if self.low > 0:
            stages.append(highpass(self.low, self.rate_hz, order=self.order,
                                   at_hz=self.centre_hz))
        lp = lowpass(self.high, self.rate_hz, order=self.order, at_hz=self.centre_hz)
        if lp is not None:
            stages.append(lp)
        return stages

    @property
    def name(self) -> str:
        return f"product_{self.centre_hz:g}"

    @property
    def amplitude_name(self) -> str:
        return f"amplitude_{self.centre_hz:g}"

    @property
    def unit(self) -> str:
        """µV². A product of two voltages is not an amplitude, and a card printing µV over it would lie —
        every threshold here is derived from data so nothing breaks, but the number is a square."""
        return "µV²"

    @property
    def delay_seconds(self) -> float:
        # Summed across a channel's stages: the slow branch can be a high-pass and a lowpass in series, and
        # "how late is this reading" wants both rather than the worse of the two.
        return max(sum(f.latency_seconds for f in stages) for stages in self.filters.values())

    @property
    def smoothing_seconds(self) -> float:
        return self.post_seconds

    @property
    def settle_seconds(self) -> float:
        """How long before the reading means anything — the step transient's lifetime, not the group delay.

        The two differ by an order of magnitude on the slow branch and conflating them was a real error:
        `delay_seconds` at the ¼–2× default is 3.84 s while the measured step transient lasts **66 s** (see
        `SLOW_BANDS`), because settling is set by the *corners'* time constants and delay by the response at
        the centre. Per stage: three time constants for the high-pass — `1/(0.707·2π·fc)` each for an
        order-2 Butterworth, whose pole pair decays at 0.707·ωc — and `2/fc` for the lowpass, the same
        conservative figure `SignedWave.band_settle_seconds` carries against its own measurement. At 0.05 Hz
        ¼–2× that sums to 74 s against the measured 66: deliberately the far side, since what it gates is
        whether a number is shown as meaning anything.
        """
        if not self.lowpass_only:
            return 2.0 / max(self.high - self.low, 1e-9)
        settle = 2.0 / max(self.high, 1e-9)
        if self.low > 0:
            damping = 0.707 if self.order >= 2 else 1.0
            settle += 3.0 / (damping * 2.0 * np.pi * self.low)
        return settle

    @property
    def warmup(self) -> float:
        """0…1 of the way through settling. Public because a screen must be able to say "not yet"."""
        seconds = self.samples_seen / max(self.rate_hz, 1e-9)
        return min(1.0, seconds / max(self.settle_seconds, 1e-9))

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds + self.post_seconds

    @property
    def balance(self) -> float:
        """How many times louder the louder channel is in this band, 1.0 when they match."""
        values = sorted(self.parts.values())
        if len(values) != 2 or values[0] <= 0:
            return float("inf") if any(values) else 1.0
        return values[1] / values[0]

    @property
    def louder_channel(self) -> int | None:
        if not any(self.parts.values()):
            return None
        return max(self.parts, key=self.parts.get)

    def describe(self) -> str:
        notch = (" · 4–6 Hz notched" if self.theta_notch else
                 " · notch off, it would remove this band" if self.notch_suppressed else "")
        if self.lowpass_only and self.low <= 0:
            return (f"below {self.high:g} Hz · order {self.order} lowpass at twice the centre, "
                    f"no high-pass{notch}")
        if self.lowpass_only:
            return (f"{self.low:g}–{self.high:g} Hz · order {self.order}, {self.slow_band} of the "
                    f"centre{notch}")
        return f"{self.low:g}–{self.high:g} Hz · order {self.order}{notch}"

    def feed(self, block: Block) -> list[Reading]:
        paired = self.aligner.push(block)
        if paired is None:
            return []
        start, a_raw, b_raw = paired
        # The aligner accumulates `Block.values`, which are already in µV — scaling here again would
        # shrink every reading by the scale (41× on this device) and, since the product is a square,
        # the reward by its square. That happened.
        raw_a = np.asarray(a_raw, dtype=float)
        raw_b = np.asarray(b_raw, dtype=float)
        notch_a, notch_b = self.notches[self.pair[0]], self.notches[self.pair[1]]
        if notch_a is not None:
            raw_a = notch_a.process(raw_a)
        if notch_b is not None:
            raw_b = notch_b.process(raw_b)
        wave_a, wave_b = raw_a, raw_b
        for stage in self.filters[self.pair[0]]:
            wave_a = stage.process(wave_a)
        for stage in self.filters[self.pair[1]]:
            wave_b = stage.process(wave_b)
        # The amplitude reward, on the sum. Mean-absolute, smoothed
        # by the same post filter so one control sets the timescale of both rewards rather than two that can
        # disagree. **Not** the peak convention the rest of this codebase reports; every
        # threshold here is derived from data, and the screen names the convention.
        summed = raw_a + raw_b
        for stage in self.sum_filter:
            summed = stage.process(summed)
        if len(summed):
            self.sum_wave = float(summed[-1])
        sum_env = np.abs(summed)
        self.waveforms = {self.pair[0]: wave_a, self.pair[1]: wave_b}
        self.wave_start = start
        self.samples_seen += len(wave_a)
        if len(wave_a):
            self.parts = {self.pair[0]: float(np.sqrt(np.mean(wave_a ** 2))),
                          self.pair[1]: float(np.sqrt(np.mean(wave_b ** 2)))}

        # Rectify, then smooth. See the module docstring for why that order is the measure.
        raw = np.clip(wave_a * wave_b, 0.0, None)
        value, beta = self._post, self._post_alpha
        smoothed = np.empty(len(raw))
        amp = np.empty(len(raw))
        sv = self._sum_post
        for i in range(len(raw)):
            value += beta * (raw[i] - value)
            smoothed[i] = value
            sv += beta * (sum_env[i] - sv)
            amp[i] = sv
        self._post = float(value)
        self._sum_post = float(sv)

        if self._next_index is None:
            self._next_index = start
        out: list[Reading] = []
        for offset in range(len(raw)):
            index = start + offset
            if index < self._next_index:
                continue
            self._next_index = index + self.stride
            self.value = float(smoothed[offset])
            self.amplitude = float(amp[offset])
            t = index / self.rate_hz
            out.append(Reading(name=self.name, value=self.value, t=t,
                               channel=self.pair, unit=self.unit, valid=block.valid))
            out.append(Reading(name=self.amplitude_name, value=self.amplitude, t=t,
                               channel=self.pair, unit="µV", valid=block.valid))
        return out
