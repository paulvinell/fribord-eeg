"""The slow cortical potential: a lowpassed signed voltage, with **no de-trender on purpose**.

What the SCP screen measures. It looks like `wave.SignedWave` with the drift follower switched off, and the
resemblance is real — same filter shape, same gap handling, same readings — but the *reason* the de-trender
is absent is what makes this a different metric rather than a configuration of that one.

**SCP training is trial-based, and the per-trial baseline is the de-trend.** Every trial subtracts the mean
of its own two-second baseline window from everything that follows, so a standing offset and slow electrode
drift are removed trial by trial, at the moment they matter, by the protocol's own arithmetic
(`nfb_console.trials`). A drift follower on top of that would not be redundant — it would be destructive:
the shift being trained *is* a sustained deviation over about eight seconds, exactly what a high-pass fast
enough to matter within a session removes. `wave`'s follower at `centre/8` would sit at 0.0625 Hz for an
equivalent band and eat most of an eight-second shift. So the only thing under this band is the amplifier's
own corner, and `shift_survival` below is the arithmetic for what that corner costs.

**The lowpass is at 1 Hz, order 2.** Its job is to remove the ongoing EEG — alpha, beta, muscle — so what
remains is the quasi-DC level the trial engine reads. Clinical SCP feedback smooths to about this order;
the corner is not delicate, because everything the protocol computes is a *mean over seconds* on top of it.
What the corner buys is a display that moves like a level rather than like EEG. Group delay is ~0.3 s,
which against an eight-second trial is nothing (the same argument `wave` makes about its own latency).

**The amplifier's corner is the protocol's floor, and 0.01 Hz is workable.** A single-pole high-pass with
corner 0.01 Hz has a time constant of ~16 s. A sustained shift held for 6 s through it keeps ~83% of its
mean (see `shift_survival`); at the 0.5 Hz default corner it keeps ~5%, which is no protocol at all. This is
the same class of wiring check `wave_view.wiring_problem` does, with the window replacing the band edge —
clinical SCP with AC amplifiers at time constants near 10-16 s is an established arrangement, so 0.01 Hz
puts this device inside it. True DC would need the device's voltage mode, which this driver does not use.
"""

from __future__ import annotations

import math

import numpy as np

from ..stream import Block, ChannelSpec
from .filters import lowpass
from .metrics import Reading

SCP_LOWPASS_HZ = 1.0
"""The corner above which the ongoing EEG is removed. Everything the trial engine computes is a mean over
seconds, so the corner's precision does not matter; what it decides is whether the on-screen level moves
like a level or like EEG."""

SCP_ORDER = 2
"""Two biquads, the conventional realtime choice, and the same order the waveform metric uses."""


def shift_survival(corner_hz: float, seconds: float) -> float:
    """What fraction of a sustained shift's *mean* survives the amplifier's high-pass, over a window.

    A single-pole high-pass turns a step into a decaying exponential with `tau = 1/(2*pi*corner)`. The mean
    of that decay over the first `seconds` is `(tau/T) * (1 - exp(-T/tau))`, and the mean is the right
    quantity because it is exactly what the trial engine scores. At 0.01 Hz over 6 s this is ~0.83; at
    0.5 Hz it is ~0.05, which is why the screen refuses to pretend the default corner can run this
    protocol.
    """
    if corner_hz <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * corner_hz)
    t = max(1e-9, seconds)
    return (tau / t) * (1.0 - math.exp(-t / tau))


class SlowShift:
    """One channel's lowpassed signed voltage, in µV. Satisfies the `Metric` protocol.

    The value is **not baseline-corrected** — that is the trial engine's job, per trial, against each
    trial's own baseline window. What this hands over is the settled quasi-DC level for the engine to
    difference, plus the waveform itself for a trace ring, on the same contract as `SignedWave`.
    """

    unit = "µV"

    def __init__(self, spec: ChannelSpec, output_hz: float = 20.0):
        self.spec = spec
        self.rate_hz = spec.rate_hz
        self.low, self.high = 0.0, SCP_LOWPASS_HZ
        self.filter = lowpass(SCP_LOWPASS_HZ, self.rate_hz, order=SCP_ORDER)
        if self.filter is None:
            raise ValueError(f"no {SCP_LOWPASS_HZ:g} Hz lowpass at {self.rate_hz:g} sps")
        self.stride = max(1, int(round(self.rate_hz / output_hz)))
        self._next_index: int | None = None
        self._expected: int | None = None
        self.value = 0.0
        """The last signed reading, in µV — the level itself, not a shift. Signed, because which way it
        moves against a trial's baseline is the whole content of the protocol."""
        self.waveform: np.ndarray = np.zeros(0)
        self.wave_start: int | None = None
        self.samples_seen = 0

    name = "scp"

    # --- timings ----------------------------------------------------------------------------

    @property
    def delay_seconds(self) -> float:
        return self.filter.latency_seconds

    @property
    def smoothing_seconds(self) -> float:
        return 0.0

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds

    @property
    def settle_seconds(self) -> float:
        """Two over the corner — the same conservative rule `SignedWave.band_settle_seconds` uses, and with
        no drift follower it is the whole answer: two seconds. The long wait the waveform screen has does
        not exist here, because the thing that made it long is the thing this metric deliberately lacks."""
        return 2.0 / SCP_LOWPASS_HZ

    @property
    def warmup(self) -> float:
        """0…1 of the way through settling. Public because a screen must be able to say "not yet"."""
        seconds = self.samples_seen / max(self.rate_hz, 1e-9)
        return min(1.0, seconds / max(self.settle_seconds, 1e-9))

    def describe(self) -> str:
        return (f"below {SCP_LOWPASS_HZ:g} Hz · order {SCP_ORDER} lowpass, no high-pass, no de-trender — "
                f"each trial's baseline is the de-trend")

    # --- the feed ---------------------------------------------------------------------------

    def feed(self, block: Block) -> list[Reading]:
        if block.channel != self.spec.index:
            return []
        if self._expected is not None and block.start_index != self._expected:
            # A device gap: the filter's state describes samples that no longer precede these. The trial
            # engine aborts its trial on the time jump; this only has to stop feeding it a transient and
            # say, through `warmup`, that it is settling again.
            self.filter.reset()
            self._next_index = block.start_index
            self.samples_seen = 0
        self._expected = block.end_index

        raw = np.asarray(block.samples, dtype=np.float64) * self.spec.scale
        wave = self.filter.process(raw)
        self.waveform = wave
        self.wave_start = block.start_index
        self.samples_seen += len(wave)

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
