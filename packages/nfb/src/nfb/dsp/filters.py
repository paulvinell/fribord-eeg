"""Streaming IIR filters, with their latency stated rather than implied.

Filters here are causal and stateful: they process blocks as they arrive and carry state across
them, so filtering a stream in pieces gives the same result as filtering it whole. That property
is tested, because it is the one that breaks silently if state handling is wrong.

Every filter reports `latency_seconds`. Feedback latency is the sum of the pipeline's delays, and
a delay nobody wrote down is a delay nobody can budget for.

What NOT to do here: the device already band-limits each channel at 0.4×sample rate (its
"frequency characteristic") and applies a mains notch in hardware when
`power_interference_filter` is set. Re-applying a notch in software costs latency and buys
nothing. Check the config before adding one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Band:
    """A frequency band, in Hz. Named so metric output is self-describing."""

    low: float
    high: float
    name: str = ""

    def __post_init__(self):
        if not 0 <= self.low < self.high:
            raise ValueError(f"invalid band {self.low}..{self.high}")

    @property
    def label(self) -> str:
        return self.name or f"{self.low:g}-{self.high:g}Hz"

    @property
    def centre(self) -> float:
        return (self.low + self.high) / 2


# The conventional EEG bands. Provided as a convenience, not a commitment: neurofeedback
# protocols routinely use their own edges, and `Band(...)` takes anything.
DELTA = Band(1.0, 4.0, "delta")
THETA = Band(4.0, 8.0, "theta")
ALPHA = Band(8.0, 12.0, "alpha")
SMR = Band(12.0, 15.0, "smr")
BETA = Band(15.0, 30.0, "beta")
GAMMA = Band(30.0, 45.0, "gamma")


try:                                    # the compiled kernel under `signal.sosfilt`
    from scipy.signal._sosfilt import _sosfilt as _sos_kernel
except Exception:                       # a scipy that moved it: the public call still works
    _sos_kernel = None

"""**`signal.sosfilt` spends 18.7 of its 18.8 µs validating arguments when the block is eight samples.**

Measured on this machine: 17.8 µs per call through the public function against 0.94 µs calling the kernel
it wraps — 19× — for output that matches sample for sample. The wrapper's cost is fixed per call, so it is
invisible on a long array and is nearly all of the cost at the size this device delivers: eight samples,
about 125 times a second, times every filter on every channel.

It is a private name, so the import is guarded and the public function is the fallback, and
`test_dsp.py` checks the two against each other on random data and random block splits. What it buys in
production is small (a few milliseconds per second); what it buys in the test suite is most of a minute,
because every DSP test feeds device-sized blocks."""


class StreamFilter:
    """A cascaded second-order-section IIR filter that keeps state between calls."""

    def __init__(self, sos: np.ndarray, rate_hz: float, at_hz: float | None = None):
        self.sos = np.ascontiguousarray(sos, dtype=np.float64)
        self.rate_hz = rate_hz
        self._zi = signal.sosfilt_zi(self.sos) * 0.0
        # The kernel wants its state as (signals, sections, 2) and writes through it in place.
        self._state = np.zeros((1, self.sos.shape[0], 2), dtype=np.float64)
        self._at_hz = at_hz

    def reset(self) -> None:
        """Drop filter state. Call after a gap — carrying state across missing samples would
        smear the discontinuity into everything that follows."""
        self._zi = signal.sosfilt_zi(self.sos) * 0.0
        self._state[:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not len(x):
            return x
        if _sos_kernel is None:
            y, self._zi = signal.sosfilt(self.sos, x, zi=self._zi)
            return y
        # **A copy, always.** The kernel filters in place, and `ascontiguousarray` on an already-contiguous
        # float64 array returns the array itself — so the fast path was overwriting the caller's samples.
        # It presented as a raw waveform buffer holding filtered values, three tests away from here.
        buf = np.array(x, dtype=np.float64).reshape(1, -1)
        _sos_kernel(self.sos, buf, self._state)
        return buf[0]

    @property
    def latency_seconds(self) -> float:
        """Group delay at the frequency of interest.

        IIR group delay varies with frequency, so this is the delay where it matters — band
        centre for a bandpass — not a worst case across the spectrum.

        Summed per second-order section rather than computed from the equivalent transfer
        function: the cascade's group delay is the sum of its sections' delays, and each biquad
        is well conditioned where `sos2tf` is not. Converting first gives nonsense at order 6
        and above — it reported a *negative* delay, which is what prompted doing it this way.
        """
        at = self._at_hz if self._at_hz is not None else self.rate_hz / 8
        w = np.array([2 * np.pi * at / self.rate_hz])
        total = 0.0
        for section in self.sos:
            _, gd = signal.group_delay((section[:3], section[3:]), w=w)
            total += float(gd[0])
        return total / self.rate_hz


FILTER_KINDS = ("butterworth", "elliptic")
# Passband ripple, elliptic only. 0.5 dB with the order-2 default and the 60 dB stopband below makes one
# coherent design. The ripple was 1.0 dB earlier, which mixed a loose ripple with a tight order. Anyone
# who wants 1.0 dB should raise the order too.
ELLIPTIC_RIPPLE_DB = 0.5
# Stopband attenuation. `ellip(2, rp=0.5, rs=60)` is the design this pair of defaults gives, so 60
# is a chosen design value rather than a round number.
ELLIPTIC_STOP_DB = 60.0


def bandpass(band: Band, rate_hz: float, order: int = 2,
             kind: str = "butterworth", ripple_db: float = ELLIPTIC_RIPPLE_DB) -> StreamFilter:
    """Bandpass. `order` is per edge, and `kind` is the shape of the response.

    **Butterworth** is monotonic with no ripple and the flattest group delay of the two. **Elliptic** is
    far sharper for the same order at high orders — 50 dB down where Butterworth is 30 — at the cost of
    passband ripple and a group delay that varies by more than 3× across the band. Butterworth suits
    rewards and elliptic suits inhibits at order 8: a reward's amplitude is a
    proportional signal and its phase matters, an inhibit only has to answer "over the line, in this band
    and not the neighbour's".

    At order **2** that story does not hold — elliptic there
    is *faster* than Butterworth (99 ms against 150 at the centre of a 12-15 Hz band) with a **shallower**
    immediate skirt (-4.8 dB one Hz out against -8.7), because the stopband budget goes into notches placed
    further out rather than into a steep edge. Do not assume elliptic means selective; measure the band you
    are actually using.

    `ripple_db` applies to elliptic only, and it is the trade that defines the shape: allowing more
    unevenness inside the band buys steeper skirts at the same order. 1.0 dB is about ±6% on amplitude.

    Defaults to 2 — two biquads — and the reason is
    latency: group delay is close to linear in order, so order 4 costs about twice
    as long. Measured at 250 sps, alpha 8-12 Hz: 111 ms at order 2, 205 ms at order 4, 302 ms at
    order 6. Raise it only if stopband rejection actually matters more than responsiveness.

    Raises if the band reaches the Nyquist frequency, rather than silently clipping it — a
    metric on a band the sample rate cannot represent is meaningless, and at 62.5 sps a gamma
    band is exactly that.
    """
    nyq = rate_hz / 2
    if band.high >= nyq:
        raise ValueError(
            f"band {band.label} needs {band.high} Hz but Nyquist is {nyq:g} Hz at "
            f"{rate_hz:g} sps — raise the sample rate or narrow the band"
        )
    edges = [band.low / nyq, band.high / nyq]
    if kind == "elliptic":
        sos = signal.ellip(order, max(0.01, ripple_db), ELLIPTIC_STOP_DB, edges,
                           btype="bandpass", output="sos")
    elif kind == "butterworth":
        sos = signal.butter(order, edges, btype="bandpass", output="sos")
    else:
        raise ValueError(f"unknown filter kind {kind!r}; expected one of {FILTER_KINDS}")
    return StreamFilter(sos, rate_hz, at_hz=band.centre)


def highpass(cutoff_hz: float, rate_hz: float, order: int = 2,
             at_hz: float | None = None) -> StreamFilter:
    """Butterworth highpass. `at_hz` is where `latency_seconds` is quoted — pass the frequency the
    *caller* cares about. The default of twice the corner is only a generic guess, and summing stage
    delays each quoted at its own guess produced a ranking of ILF's slow-band options that was simply
    wrong: the default `¼–2×` was reported as the slowest lower-edged option when at the trained centre
    it is the fastest. A delay is a property of the filter *at a frequency*; when the frequency of
    interest is known, say so."""
    sos = signal.butter(order, cutoff_hz / (rate_hz / 2), btype="highpass", output="sos")
    return StreamFilter(sos, rate_hz, at_hz=at_hz if at_hz is not None else cutoff_hz * 2)


def lowpass(cutoff_hz: float, rate_hz: float, order: int = 2,
            at_hz: float | None = None) -> StreamFilter | None:
    """Butterworth lowpass, or **None** when the rate already band-limits below the cutoff.

    None rather than raising, because the callers that want this — artifact detection band-limits to
    0-50 Hz before judging amplitude — are asking "remove anything above 50 Hz" and the correct answer
    on a 62.5 sps channel is "there is nothing above 50 Hz to remove". Raising would make a policy that
    is right at 250 sps fail on a channel the device decimated, which is a difference the caller should
    not have to know about.

    `at_hz` as in `highpass`: where the reported delay is quoted, defaulting to half the corner.
    """
    nyq = rate_hz / 2
    if cutoff_hz >= nyq * 0.9:
        return None
    sos = signal.butter(order, cutoff_hz / nyq, btype="lowpass", output="sos")
    return StreamFilter(sos, rate_hz, at_hz=at_hz if at_hz is not None else cutoff_hz / 2)


def bandstop(band: Band, rate_hz: float, order: int = 2) -> StreamFilter | None:
    """Butterworth bandstop — a *wide* notch, where `notch` below is a narrow one at a single frequency.

    Both exist because they do different jobs. `notch` kills mains at 50 Hz and is defined by a Q; this
    removes a whole band, and the caller that wants it is ILF Synchrony's theta notch: a 4–6 Hz bandstop that
    is applied to both channels before the synchrony product, behind a toggle that is on by default.

    None when the rate cannot represent the upper edge, matching `lowpass`: on a decimated channel the answer
    to "remove 4–6 Hz" is still meaningful but "remove up to 6 Hz of a 5 Hz Nyquist" is not.
    """
    nyq = rate_hz / 2
    if band.high >= nyq * 0.9 or band.low <= 0:
        return None
    sos = signal.butter(order, [band.low / nyq, band.high / nyq], btype="bandstop", output="sos")
    return StreamFilter(sos, rate_hz, at_hz=band.centre)


def notch(freq_hz: float, rate_hz: float, q: float = 30.0) -> StreamFilter:
    """Mains notch. Usually unnecessary — the device does this in hardware."""
    b, a = signal.iirnotch(freq_hz, q, rate_hz)
    return StreamFilter(signal.tf2sos(b, a), rate_hz, at_hz=freq_hz)


class OnePole:
    """Single-pole lowpass smoother, the classic envelope follower.

    Time constant rather than cutoff, because that is how smoothing is reasoned about in
    practice: "half a second of averaging" is the useful unit.
    """

    def __init__(self, tau_seconds: float, rate_hz: float):
        if tau_seconds <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau_seconds
        self.rate_hz = rate_hz
        self.alpha = 1.0 - np.exp(-1.0 / (tau_seconds * rate_hz))
        self._y = 0.0
        self._primed = False

    def reset(self) -> None:
        self._y = 0.0
        self._primed = False

    def process(self, x: np.ndarray) -> np.ndarray:
        """An explicit Python loop, deliberately.

        This recursion is expressible as `lfilter([a], [1, -(1-a)])`, and that was tried. At our
        block size it is *slower*: measured 187 ns/sample for this loop against 388 ns/sample for
        lfilter on 8-sample blocks, because per-call overhead dominates such short calls. lfilter
        only wins from about 64 samples up (51 ns/sample) and decisively at 512 (10 ns/sample).
        Revisit if blocks ever get longer; the two are numerically identical to 7e-15.
        """
        x = np.asarray(x, dtype=np.float64)
        if not len(x):
            return x
        if not self._primed:
            self._y = float(x[0])       # start at the signal, not at zero, to avoid a ramp
            self._primed = True
        out = np.empty_like(x)
        y, a = self._y, self.alpha
        for i, v in enumerate(x):
            y += a * (v - y)
            out[i] = y
        self._y = y
        return out

    @property
    def latency_seconds(self) -> float:
        return self.tau
