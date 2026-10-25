"""The filtered waveform itself, signed and de-trended: what the Waveform screen trains.

**This is a different quantity from every other metric here, and the difference is a response time.** A band
amplitude needs a rectifier averaged over several cycles before it means anything, so at 0.05 Hz — where three
cycles is a minute — an amplitude reward cannot be trained at all. The *signed instantaneous value* has no such
requirement: it exists every sample and its only lag is the filter's group delay. So the observable that
survives downward is the wave, which is why an infra-low reward is a level crossing on
the signed slow waveform rather than a band amplitude.

**`MAX_HZ` is 1 Hz, and above it there is nothing here rather than something else.** A wave at 10 Hz is 100 ms
a cycle: nobody can follow its position or act on it, so the value stops being feedback. The tempting way up is
to feed back a fast band's *amplitude* instead — its swing is slow, so it looks like the same thing — and it is
not: an amplitude is a size where this is a position, which is the exact distinction the screen above this
exists to make. So this module refuses the frequency rather than quietly changing what it measures. Training a
band's amplitude is the amplitude screen, and it is a better version of that than a slow reading of it here
would be.

An earlier version did offer that as a second source, and it was wrong in the way that is hardest to see: the
screen would have been feeding back a size while every word on it said position.

Two decisions worth stating, because each is a place this could have been built differently:

**The band is a lowpass at twice the centre**, which is constant-Q. It has no lower edge at all: the device's high-pass
corner is the only thing under it, which is what `WaveSession.wiring_problem` checks and why `wave_edges`
returns a zero rather than a small positive number that would read as a band.

**Drift is removed from the signal, not absorbed by the scale.** An `Auto Range` on the
output would be the other way to absorb it, and the objection to that is already written down in `music.ShepardPlayer`: auto-ranging the feedback tells
the trainee about their own variance rather than about their state. The other answer is slope compensation, and that is the one
implemented here. A cascade of one-poles at `centre / DETREND_RATIO` is subtracted from the signal, so drift
leaves the *signal* and the display scale can then hold still for the whole session.

That is not a refinement but the only lower edge there is: the band reaches DC, so electrode drift passes
through it at full amplitude. Measured on this hardware, 305 µV of drift sits on top of 20 µV of signal.

**Settling is long and is reported rather than hidden.** A 0.1 Hz order-2 lowpass takes 17 s to settle and the
de-trender takes three of its own time constants per pole on top — 173 s at 0.05 Hz, minutes at 0.01. `warmup`
is what a screen shows while that happens, and feeding back a number the filters have not settled into is the
failure it exists to prevent.

**The latency budget does not apply to this metric, and that is not a compromise.** 250 ms was
derived for alpha spindles. A 0.05 Hz bandpass has seconds of group delay by construction, and two seconds on a
twenty-second cycle is a tenth of a cycle — perceptually immediate. Latency has to be read against the period of
the thing being trained.
"""

from __future__ import annotations

import numpy as np

from ..stream import Block, ChannelSpec
from .filters import Band, OnePole, lowpass
from .metrics import Reading

MAX_HZ = 1.0
"""The highest frequency a wave's *position* can be trained at, and it is a statement about perception rather
than about filters.

At 1 Hz a cycle is a second, which is about the limit of following where in a wave you are. Above it the
position is unusable and the only thing left to feed back would be a size — see the module docstring for why
that is a different protocol and not an extension of this one.

A fork between a bandpass and a lowpass at 1.8 Hz would be a 0.3 Hz margin under half a
fixed 3 Hz bandwidth: an artefact of a width, not a judgement about what can be perceived. So this number is
chosen on its own, and it is round because the thing it describes is."""

DEFAULT_ORDER = 2
"""Two biquads, which is the conventional realtime choice."""

DETREND_RATIO = 8.0
"""The de-trender's corner, as a fraction of the trained frequency: `centre / 8`.

Two things it has to satisfy at once. It must pass the wave — a single-pole high-pass keeps
`f/sqrt(f² + fc²)`, which at a corner an eighth of the frequency is **99.2%** — and it must settle inside a
session, which rules out the much lower corner a purist would pick: at `centre/32` the loss is 0.05% and the
settling is four times as long. At 0.05 Hz this is a 25 s time constant and 173 s of total settling, against
0.002 Hz and ten minutes."""

DETREND_POLES = 2
"""How many one-poles the drift follower cascades, and **one is not enough** — slope compensation
is a different thing from a refinement of a minimum-drift cut.

Measured, because the first version had one pole and looked fine on a stationary offset: a first-order
high-pass removes a *constant* completely and leaves `τ × slope` of a **ramp**, which is what electrode gel
drying actually produces. On a 300 µV drift across 400 s at 0.05 Hz that is a standing **19 µV** bias — half
of a calibrated ±40 µV range, on a signal of 20 µV. A second pole removes it, because the residual the first
stage leaves is a constant and a high-pass has no DC gain: the same 400 s run comes out at 0.4 µV of bias.

Two *cascaded first-order* stages rather than one second-order Butterworth, for the reason
`TrainingSession.DISPLAY_FLOOR_HZ` gives about its own lower edge: a high-order high-pass rings on a step, and
at this corner the ringing would have a period of minutes. Each one-pole decays monotonically."""

RMS_CYCLES = 4.0
"""Cycles of the trained frequency the running RMS averages over. Four, because it is what a calibration reads
its range from and one cycle of a wave is not an estimate of its size."""


def wave_edges(centre_hz: float) -> tuple[float, float]:
    """`(low, high)` of the band actually built: a lowpass at twice the centre.

    The zero lower edge is the truth about it rather than a convenience — the filter has no high-pass, so the
    amplifier's corner is the only thing beneath the band, and a screen has to say so.

    **The unconditional lowpass is a consequence of `MAX_HZ`.** A bandpass with a fixed 3 Hz bandwidth
    would collapse to a lowpass when `centre − bw/2 − 0.3 ≤ 0`, which is centre ≤ 1.8 Hz, so at every
    frequency this metric accepts a bare lowpass is the right shape. The de-trender below is what
    supplies the lower edge, as a software high-pass, for an amplifier whose corner is not in hardware.
    """
    return 0.0, centre_hz * 2.0


def detrend_tau(centre_hz: float) -> float:
    """Time constant of the drift follower, from `DETREND_RATIO`. A one-pole's corner is `1/(2πτ)`."""
    corner = max(1e-6, centre_hz / DETREND_RATIO)
    return 1.0 / (2.0 * np.pi * corner)


class SignedWave:
    """One channel's low-pass-filtered waveform, de-trended. Satisfies the `Metric` protocol.

    What an infra-low reward measures, without auto-ranging: a level crossing on the signed
    waveform, where the level sits at the middle of a scale. Here the scale is a screen's business
    (`nfb_console.normalise`) and this hands back microvolts.

    It also exposes the waveform itself with the absolute index it starts at, because on that screen the
    waveform is not a debugging aid: it is the instrument the state is read off, and `traces.WaveRing`
    decimates it at its own rate. Same contract as `ProductSynchrony`, for the same reason, and it is what
    lets one ring serve both screens.
    """

    unit = "µV"

    def __init__(self, spec: ChannelSpec, centre_hz: float, order: int = DEFAULT_ORDER,
                 output_hz: float = 20.0, detrend: bool = True):
        if not 0 < centre_hz <= MAX_HZ:
            raise ValueError(
                f"{centre_hz:g} Hz is not a position anybody can follow — this metric stops at "
                f"{MAX_HZ:g} Hz; see MAX_HZ")
        self.spec = spec
        self.centre_hz = float(centre_hz)
        self.rate_hz = spec.rate_hz
        self.order = max(1, min(4, int(order)))
        self.low, self.high = wave_edges(self.centre_hz)
        self.filter = lowpass(self.high, self.rate_hz, order=self.order)
        if self.filter is None:
            raise ValueError(f"no filter for {self.describe()} at {self.rate_hz:g} sps")
        self.detrend = bool(detrend)
        self.trend_filters = ([OnePole(detrend_tau(self.centre_hz), self.rate_hz)
                               for _ in range(DETREND_POLES)] if detrend else [])
        self.rms_filter = OnePole(max(0.5, RMS_CYCLES / self.centre_hz), self.rate_hz)
        self.stride = max(1, int(round(self.rate_hz / output_hz)))
        self._next_index: int | None = None
        self._expected: int | None = None
        self.value = 0.0
        """The last signed reading, in µV. **Signed** — which side of its own baseline the wave is on is the
        whole content of this metric."""
        self.trend = 0.0
        """How much drift is currently being subtracted, in µV. Reported rather than drawn: measured on this
        hardware it runs to 305 µV against a 20 µV signal, so on the wave's own axis it is off-screen by a
        factor of fifteen and the only honest place for it is a caption."""
        self.rms = 0.0
        self.waveform: np.ndarray = np.zeros(0)
        self.wave_start: int | None = None
        self.samples_seen = 0

    @property
    def name(self) -> str:
        return f"wave_{self.centre_hz:g}"

    @property
    def band(self) -> Band:
        """What it is measured over, for a card to print. `Band` will not take a zero lower edge, so this
        reports the smallest number a screen would ever show and the caption says "below X"."""
        return Band(max(self.low, 1e-4), self.high, "")

    # --- timings ----------------------------------------------------------------------------

    @property
    def delay_seconds(self) -> float:
        return self.filter.latency_seconds

    @property
    def smoothing_seconds(self) -> float:
        return 0.0

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds + self.smoothing_seconds

    @property
    def band_settle_seconds(self) -> float:
        """Roughly two over the upper corner for an order-2 Butterworth — 20 s at a 0.1 Hz corner, against
        the 17 s measured on that exact filter. Deliberately the
        conservative side of the measurement, since what it gates is whether feedback starts at all."""
        return 2.0 / max(self.high, 1e-9)

    @property
    def settle_seconds(self) -> float:
        """How long before the reading means anything: the band's settling plus the de-trender's.

        Three time constants per pole, which is where a one-pole is within 5% of its answer. Both terms are
        needed and the second dominates — a screen that waited only for the band would feed back the drift it
        has not finished measuring yet.
        """
        return self.band_settle_seconds + sum(3.0 * f.tau for f in self.trend_filters)

    @property
    def warmup(self) -> float:
        """0…1 of the way through settling. Public because a screen must be able to say "not yet"."""
        seconds = self.samples_seen / max(self.rate_hz, 1e-9)
        return min(1.0, seconds / max(self.settle_seconds, 1e-9))

    def describe(self) -> str:
        drift = (f" · drift removed above {self.centre_hz / DETREND_RATIO:.3g} Hz" if self.detrend
                 else " · drift left in")
        return (f"below {self.high:g} Hz · order {self.order} lowpass at twice the centre, "
                f"no high-pass{drift}")

    # --- the feed ---------------------------------------------------------------------------

    def feed(self, block: Block) -> list[Reading]:
        if block.channel != self.spec.index:
            return []
        if self._expected is not None and block.start_index != self._expected:
            # A gap: every filter's state describes samples that no longer precede these, and the drift
            # follower in particular would step by whatever the signal moved while nothing arrived.
            # This is a *device* gap and nothing else — the artifact gate keeps rejected blocks
            # contiguous precisely so a blink does not land here (see `nfb.dsp.artifact` on `hold`).
            self.filter.reset()
            for stage in self.trend_filters:
                stage.reset()
            self.rms_filter.reset()
            self._next_index = block.start_index
            # The settling clock restarts with the filters, or `warmup` would go on saying "settled"
            # while the reading is mostly transient — which is exactly what it exists to prevent.
            self.samples_seen = 0
        self._expected = block.end_index

        raw = np.asarray(block.samples, dtype=np.float64) * self.spec.scale
        x = raw
        for stage in self.trend_filters:
            x = x - stage.process(x)
        if self.trend_filters and len(x):
            self.trend = float(raw[-1] - x[-1])
        wave = self.filter.process(x)
        self.waveform = wave
        self.wave_start = block.start_index
        self.samples_seen += len(wave)
        if len(wave):
            self.rms = float(np.sqrt(max(0.0, self.rms_filter.process(wave ** 2)[-1])))

        if self._next_index is None:
            self._next_index = block.start_index
        out: list[Reading] = []
        while self._next_index < block.end_index:
            i = self._next_index - block.start_index
            self.value = float(wave[i])
            out.append(Reading(name=self.name, value=self.value,
                               t=self._next_index / self.rate_hz, channel=self.spec.index,
                               unit=self.unit, valid=block.valid))
            self._next_index += self.stride
        return out
