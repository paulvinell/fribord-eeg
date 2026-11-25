"""RR intervals in, tachogram and spectrum out.

The spectrum is its own object with a peak structure rather than a flat bag of summary floats, and a
split detector flags a spectrum with no peak at the target.

The chain, which is the standard one (Task Force 1996):

    RR intervals, one per beat, irregularly spaced
      → instantaneous HR = 60/RR per beat
      → linear interpolation onto a uniform grid
      → mean-detrend, Hann window, rFFT
      → power against frequency

**Why report peak structure rather than a coherence score.** A score cannot distinguish the two ways of
being wrong. Power concentrated at 0.1 Hz is the goal; power at 0.03 Hz and power at 0.25 Hz with nothing
between them is the *specific failure* to catch — sympathetic and parasympathetic drive at once —
and "fraction of power in the resonance band" is low in both a flat spectrum and a split one. So the
peaks are found and named, and the split is flagged.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .artifacts import ArtifactFilter, RANGE_MAX_S, RANGE_MIN_S

# The resonance band, where a coherence number lives. Wide enough to span normal
# resonance breathing (5–8 breaths a minute).
RESONANCE_BAND = (0.04, 0.26)
VLF_BAND = (0.0033, 0.04)       # sympathetic drive, in the split failure

MIN_BEATS_FOR_SPECTRUM = 3
MIN_GRID_SAMPLES = 4
"""The only floors on building a spectrum, and both are arithmetic: two points to interpolate between and
a handful of samples to transform.

**"Cannot" and "will not be accurate" are different things**, and every previous floor here was the second
wearing the first's clothes — first about eighty seconds, then twenty. The transform is defined on any
span. Refusing to produce one meant a blank plot, a dash where the peak-share number goes, and nothing
driving the outputs, through the opening of every session; and the twenty-second version was the same
mistake made smaller.

What a span decides is *which parts of the curve mean anything*, which is `resolves`, asked per claim by
whoever makes it. The curve is drawn from the first beats and the reader is told the span and the bin
width; the sentence naming a peak waits for four cycles of it; the split waits for a window that can see
the very-low-frequency band. A screen showing a two-bin spectrum labelled "8 s of beats, 0.125 Hz per bin"
is telling the truth. A blank panel for the first minute is telling the truth too, and telling it about
nothing."""

RESOLVE_CYCLES = 4.0
"""How many cycles of a frequency a window must contain before a claim about it is a measurement. Four
puts 0.1 Hz at forty seconds: the curve is drawn from twenty, and the sentence naming a peak waits."""

SPLIT_MIN_SPAN_S = 1.0 / VLF_BAND[0] / 4
"""About 76 s, and unchanged: the split is a claim about the very-low-frequency band, and a window that
cannot see VLF will make one out of its own edges — measured, a 70 s dropout printed the clinical warning
on a clean heart. A quarter cycle rather than four, because this is a band-power comparison rather than a
peak location; it is the figure the whole spectrum used to wait for."""
HF_BAND = (0.15, 0.40)          # respiratory, parasympathetic
COHERENCE_TOP_HZ = 0.4          # where the "above the peak" region stops; see `Spectrum.coherence`

TARGET_HZ = 0.1
"""The frequency the spectrum is scored against, **by default and not by definition**.

0.1 Hz is the goal, a single spectral peak at precisely 0.1 Hz, and it is where the
Mayer–Traube–Hering wave sits in the population. It is not where it sits in any particular person: the
resonant frequency of the baroreflex loop depends on blood volume and on height, published resonance
protocols find people anywhere from 4.5 to 7 breaths a minute, and lower is common in tall adults. So
this is where the screen starts, and `HRVProcessor(target_hz=...)` is how it moves. Every claim scored
against it — `sharpness`, `distance_hz`, `peak_proximity`, the split, the marked band on the spectrum
— reads the *spectrum's* target rather than this constant, so one setting moves all of them together.

**Nothing else moves with it.** `RESONANCE_BAND`, `LF_BAND`, `HF_BAND` and `VLF_BAND` are standard
definitions that a published figure is quoted against, and re-cutting them around a personal target
would produce numbers that look like the conventional ones and are not."""

NARROW_HALF_HZ = 0.02
"""Half the width of the band around the target that `sharpness` integrates.

±0.02 Hz tolerates a breath a cycle a minute either side of the target and absorbs Hann leakage;
narrower would reward landing on the bin grid over being at the right frequency. Adjustable alongside
the target because the two are one judgement — how tightly you are asking somebody to hold a rate."""

TARGET_RANGE_HZ = (0.04, 0.15)
"""How far the target may be moved. The bottom is `RESONANCE_BAND`'s and `VLF_BAND`'s shared edge and
the top is `LF_BAND`'s: outside these the target would sit where the standard bands call the power
something else, and `peak_proximity`'s search range would not contain its own target."""

NARROW_HALF_RANGE_HZ = (0.005, 0.05)
"""And how tight or loose the band may be. The floor is about one bin of a three-minute window, so a
narrower one would be measuring the grid; the ceiling keeps it inside `RESONANCE_BAND` at any target."""


def narrow_band(target_hz: float = TARGET_HZ,
                half_hz: float = NARROW_HALF_HZ) -> tuple[float, float]:
    """The band `sharpness` integrates, around a target."""
    return (target_hz - half_hz, target_hz + half_hz)


NARROW_BAND = narrow_band()
"""The default band, for callers that have no spectrum in hand. Anything holding a `Spectrum` should
ask it for `spectrum.narrow_band` instead, which is the one the numbers on the screen were computed
with."""

SPLIT_RATIO = 1.5
"""How much stronger both flanking peaks must be than the target band to call it a split. Not a
literature number — chosen so that a clean resonance spectrum never trips it and the simulated
gas-and-brake case does. Recorded here rather than buried, because it is a judgement."""

GAP_FACTOR = 5.0
GAP_MIN_S = 5.0
"""When a hole between accepted beats stops being a withheld artifact and becomes a dropout.

The artifact filter *withholds* rejected beats on purpose, so one- and two-beat holes are ordinary and
the tachogram must interpolate across them — that is the design (see `ArtifactFilter`). A strap going
silent is a different event wearing the same shape: interpolating across it draws a long straight ramp,
and a ramp is broadband power piled at the low end of the spectrum — squarely in `VLF_BAND`, the
"sympathetic" half of the split detector. Measured on a perfect 0.1 Hz signal: a 60 s dropout took
sharpness from 0.998 to 0.258 and put 44% of total power into VLF, and with ordinary respiratory content
present a ~70 s dropout made `split` fire on a clean heart — the one clinical warning this protocol
exists to deliver, manufactured by a flat battery, with the reject fraction reading zero throughout.

So a hole longer than `GAP_FACTOR` local median intervals (never less than `GAP_MIN_S`) ends the
spectrum's window: only the contiguous run *after* the last such hole is analysed, and the spectrum
reports not-ready again until that run is long enough. Withheld pairs at resting rates leave ~3 s holes
and pass; five seconds of silence is not a heartbeat phenomenon at any plausible rate."""


@dataclass(frozen=True)
class Beat:
    """One heart-rate notification, decoded.

    `rr_intervals` may be empty (some devices send HR-only packets between beats), hold one (the usual
    case), or several (when a device batches — rare but legal).
    """

    t: float                        # epoch seconds, arrival
    hr_bpm: float
    rr_intervals: list[float]       # seconds, oldest first
    contact: bool | None = None     # None when the device does not report it
    arrived: float | None = None    # when the source gave the beat, on the clock of `t`, if that is later


RR_SCALE_SPEC = 1024.0      # the Bluetooth Heart Rate Service's mandated unit: 1/1024 s
RR_SCALE_MS = 1000.0        # plain milliseconds, which is what at least one shipping device sends


class RrScaleResolver:
    """Works out whether a device's RR integers are 1/1024 s or milliseconds.

    **This exists because we had a silent 2.3% error in every interval**, and the docstring below used to
    say why without noticing: "the HeartMath Coherence Plus advertises the standard service, so this is the
    same code path as a Polar strap." It is — and that device reports **milliseconds**, against the spec.
    Established two independent ways on hardware: the device's own HR field, and 500 Hz PPG beat
    detection.

    2.3% is `1000/1024`, so it never looks like a fault: heart rate reads a little low, RMSSD a little
    high, and the spectral peak sits 2.3% off where it belongs. At a nominal 0.1 Hz that is 0.0977, which
    is well inside the band nobody would question.

    Hardcoding milliseconds is not the fix, because a Polar strap really is compliant. So it is detected:
    every notification carries an HR in bpm *alongside* the RR integers, and only one interpretation
    reproduces it. The two differ by 2.3%, which one smoothed packet cannot settle, so error is accumulated
    for both hypotheses over `min_packets` notifications and then committed for the rest of the session.
    """

    def __init__(self, mode: str = "auto", min_packets: int = 8):
        mode = (mode or "auto").lower()
        if mode not in ("auto", "spec", "ms"):
            raise ValueError(f"rr_units must be 'auto', 'spec' or 'ms'; got {mode!r}")
        self.mode = mode
        self.min_packets = max(1, min_packets)
        self.decided: str | None = None if mode == "auto" else mode
        self._err = {"spec": 0.0, "ms": 0.0}
        self._n = 0

    @property
    def scale(self) -> float:
        """The divisor to apply now. Defaults to the spec until the evidence says otherwise."""
        return RR_SCALE_MS if self.decided == "ms" else RR_SCALE_SPEC

    def observe(self, hr_bpm: float, raw: list[int]) -> float:
        """Note one packet's evidence and return the divisor to use for it.

        The device's own HR is the reference. It is usually smoothed over several beats, which is why this
        accumulates rather than deciding on one packet — but smoothing biases both hypotheses equally, so
        the *difference* between their errors is still informative.
        """
        if self.decided is not None or not raw or sum(raw) <= 0 or hr_bpm <= 0:
            return self.scale
        implied = 60.0 / (sum(raw) / len(raw))
        self._err["spec"] += abs(implied * RR_SCALE_SPEC - hr_bpm)
        self._err["ms"] += abs(implied * RR_SCALE_MS - hr_bpm)
        self._n += 1
        if self._n >= self.min_packets:
            self.decided = "ms" if self._err["ms"] < self._err["spec"] else "spec"
        return self.scale

    def describe(self) -> str:
        if self.decided is None:
            return f"resolving RR units ({self._n}/{self.min_packets} packets)"
        unit = "milliseconds" if self.decided == "ms" else "1/1024 s per the spec"
        return f"RR in {unit}" + (" (set by hand)" if self.mode != "auto" else " (detected)")


def parse_hr_measurement(data: bytes,
                         scale: float = RR_SCALE_SPEC) -> tuple[float, list[float], bool | None]:
    """Decode a BLE Heart Rate Measurement per the SIG spec.

        byte 0   flags: bit 0 HR format (0=uint8, 1=uint16), bits 1-2 contact status,
                 bit 3 energy expended present, bit 4 RR intervals present
        byte 1+  HR value
        then     optional uint16 energy expended
        then     optional uint16 array of RR intervals

    `scale` is the divisor for those intervals, and it is a parameter rather than the spec's constant
    because at least one shipping device ignores the spec — see `RrScaleResolver`. Use
    `parse_hr_raw` when you want the integers to feed the resolver.
    """
    if len(data) < 2:
        return 0.0, [], None

    flags = data[0]
    hr_is_uint16 = bool(flags & 0x01)
    contact_status = (flags & 0x06) >> 1
    contact = bool(contact_status == 3) if contact_status in (2, 3) else None
    energy_present = bool(flags & 0x08)
    rr_present = bool(flags & 0x10)

    idx = 1
    if hr_is_uint16:
        hr = float(int.from_bytes(data[idx:idx + 2], "little"))
        idx += 2
    else:
        hr = float(data[idx])
        idx += 1
    if energy_present:
        idx += 2

    rr: list[float] = []
    if rr_present:
        while idx + 1 < len(data):
            rr.append(int.from_bytes(data[idx:idx + 2], "little") / scale)
            idx += 2
    return hr, rr, contact


def parse_hr_raw(data: bytes) -> tuple[float, list[int], bool | None]:
    """The same decode, with the RR intervals left as the integers the device sent.

    What `RrScaleResolver` needs: it compares the two candidate scalings against the device's own HR, and
    it cannot do that once one of them has already been applied.
    """
    hr, scaled, contact = parse_hr_measurement(data, scale=1.0)
    return hr, [int(round(v)) for v in scaled], contact


@dataclass(frozen=True)
class Peak:
    hz: float
    power: float
    label: str


@dataclass
class Spectrum:
    """The power spectrum of the tachogram, plus what is worth saying about its shape."""

    freqs: np.ndarray
    power: np.ndarray
    span_s: float = 0.0
    """How many seconds of beats this was computed from. It is what decides which parts of the curve
    mean anything — see `resolves` — and it is drawn on the screen, because a spectrum with no span
    beside it invites being read as if every part of it were equally trustworthy."""

    target_hz: float = TARGET_HZ
    narrow_half_hz: float = NARROW_HALF_HZ
    """The frequency this spectrum is scored against, and the half-width of the band around it.

    **Carried on the spectrum rather than read from the module** so that a number and the target it was
    computed against cannot come apart. A screen draws one spectrum and reads several quantities off it;
    if `sharpness` asked a global while the marked band on the plot asked a session setting, moving the
    target mid-session would put the percentage and the shaded region a beat out of step, which is the
    one thing a person adjusting their breathing would read as the measurement being wrong."""

    @property
    def narrow_band(self) -> tuple[float, float]:
        """The band `sharpness` integrates: the target plus and minus the half-width."""
        return narrow_band(self.target_hz, self.narrow_half_hz)

    @property
    def ready(self) -> bool:
        return bool(self.freqs.size)

    @property
    def resolution_hz(self) -> float:
        """The bin spacing, which is one over the span. 0.05 Hz from twenty seconds."""
        return 1.0 / self.span_s if self.span_s > 0 else float("inf")

    def resolves(self, hz: float) -> bool:
        """Whether `hz` has had enough cycles in this window to be a measurement rather than a guess.

        **Four cycles**, which is the same rule the window floor is set by: at fewer than that a "peak"
        is one or two bins wide and its height says as much about where the window happened to start as
        about the signal. 0.1 Hz needs 40 s, and the bottom of the very-low-frequency band needs 20
        minutes — which is why the split detector below can be honest about resonance long before it can
        say anything at all about VLF.
        """
        return self.ready and self.span_s * hz >= RESOLVE_CYCLES

    @property
    def total(self) -> float:
        return float(self.power.sum()) if self.ready else 0.0

    def band_power(self, band: tuple[float, float]) -> float:
        if not self.ready:
            return 0.0
        sel = (self.freqs >= band[0]) & (self.freqs <= band[1])
        return float(self.power[sel].sum())

    def band_peak(self, band: tuple[float, float]) -> Peak | None:
        if not self.ready:
            return None
        sel = (self.freqs >= band[0]) & (self.freqs <= band[1])
        if not sel.any() or self.power[sel].max() <= 0:
            return None
        i = int(np.argmax(self.power[sel]))
        return Peak(float(self.freqs[sel][i]), float(self.power[sel][i]), "")

    @property
    def sharpness(self) -> float:
        """Fraction of total power inside the narrow band around the target. 1.0 is the goal, reached."""
        total = self.total
        return self.band_power(self.narrow_band) / total if total > 0 else 0.0

    @property
    def resonance_ratio(self) -> float:
        total = self.total
        return self.band_power(RESONANCE_BAND) / total if total > 0 else 0.0

    @property
    def coherence(self) -> float:
        """The coherence ratio, which is more selective than `resonance_ratio`.

        Two differences from `resonance_ratio`:

        - **It locates the peak; `resonance_ratio` integrates a fixed band.** The peak is the largest *local maximum*
          anywhere in 0.031–0.266 Hz, so somebody whose resonant frequency sits at 0.08 or 0.12 Hz is not
          penalised for it. `resonance_ratio` sums all in-band power regardless of shape, and `sharpness`
          is narrow but pinned to 0.1 Hz.
        - **It multiplies two ratios; `resonance_ratio` takes one.** Power on *both* sides of the peak is punished, and
          multiplicatively. That rewards entrainment — one dominant narrow peak — rather than in-band
          energy: a diffuse band-filling signal and a sharp peak of equal integrated power score the same
          under `resonance_ratio` and very differently here.

        ```
        P_peak  = the peak bin and its two neighbours
        P_below = everything from the band's bottom up to P_peak
        P_above = everything from P_peak up to 0.4 Hz
        ratio   = (P_peak / P_below) * (P_peak / P_above)
        ```

        `P_below` drops out when the peak is at the bottom of the band. The value is
        dimensionless-squared and unbounded, which is why the displayed number is `ln(1 + ratio)` —
        `coherence_score` is that log.

        `resonance_ratio` stays as well rather than being replaced: it is a fraction of total power and is
        the number `split` and the band readouts are written against, and a screen that silently swapped one
        for the other would make every stored threshold mean something else.
        """
        if not self.ready or self.freqs.size < 5:
            return 0.0
        lo, hi = RESONANCE_BAND
        band = (self.freqs >= lo) & (self.freqs <= hi)
        if not band.any():
            return 0.0
        idx = np.flatnonzero(band)
        # A local maximum, not the largest bin: an endpoint that happens to be high is the shoulder of
        # something outside the band, and treating it as a peak measures the wrong thing.
        inner = idx[(idx > 0) & (idx < self.power.size - 1)]
        local = inner[(self.power[inner] > self.power[inner - 1])
                      & (self.power[inner] >= self.power[inner + 1])]
        if not local.size:
            return 0.0
        p = int(local[np.argmax(self.power[local])])
        top = float(min(COHERENCE_TOP_HZ, self.freqs[-1]))
        peak = float(self.power[max(p - 1, 0):p + 2].sum())
        if peak <= 0:
            return 0.0
        below = float(self.power[(self.freqs >= lo) & (self.freqs < self.freqs[max(p - 1, 0)])].sum())
        above = float(self.power[(self.freqs > self.freqs[min(p + 1, self.power.size - 1)])
                                 & (self.freqs <= top)].sum())
        # The floors are divide-by-zero guards relative to the peak, in units that do not depend on how
        # the tachogram happens to be scaled.
        guard = peak * 1e-9
        ratio = (peak / max(below, guard)) * (peak / max(above, guard))
        return float(ratio) if below > 0 else float(peak / max(above, guard))

    @property
    def coherence_score(self) -> float:
        """`ln(1 + coherence)`, which is the number put on screen and used to score sessions."""
        return float(np.log1p(max(0.0, self.coherence)))

    @property
    def distance_hz(self) -> float:
        """Wasserstein-1 distance from the normalised spectrum to a spike at the target, in Hz.

        0 is the ideal. Better than "power in a band" for training, because it is continuous in *how far
        off* the energy is rather than in whether it crossed a boundary — a peak at 0.13 Hz scores worse
        than one at 0.11 and better than one at 0.3, which is what a person adjusting their breathing
        needs the number to do.
        """
        if not self.ready or self.total <= 0 or self.freqs.size < 2:
            return 0.0
        normalised = self.power / self.total
        cdf = np.cumsum(normalised)
        step = float(self.freqs[1] - self.freqs[0])
        return float(np.abs(cdf - (self.freqs >= self.target_hz).astype(float)).sum() * step)

    @property
    def target_label(self) -> str:
        """What to call the band around the target, in the three places that name it: the peak list,
        the split sentence, and the screen. `0.1 Hz` when the target is where it starts."""
        return f"{self.target_hz:g} Hz"

    @property
    def peaks(self) -> list[Peak]:
        found = []
        for band, label in ((VLF_BAND, "very low"), (self.narrow_band, self.target_label),
                            (HF_BAND, "high")):
            peak = self.band_peak(band)
            if peak is not None:
                found.append(Peak(peak.hz, peak.power, label))
        return found

    @property
    def split(self) -> bool:
        """The gas-and-brake failure: energy at very low *and* high frequency, with 0.1 Hz missing.

        The reason a single coherence number is not enough: a person can appear to make improvements in
        EEG while the HRV splits from its normal peak at 0.1 Hz into high frequency parasympathetic and
        very low frequency sympathetic (fight or flight) peaks.
        """
        # **The one claim on this screen that needs the long window.** It is a statement about the
        # very-low-frequency band, whose bottom is 0.0033 Hz, and a window that cannot resolve VLF will
        # happily produce a VLF "peak" out of the window's own edges — which is exactly how a 70 s
        # dropout once printed this warning on a clean heart. Everything else here is about the
        # resonance band and is measurable from far less.
        if not self.ready or self.span_s < SPLIT_MIN_SPAN_S:
            return False
        target = self.band_peak(self.narrow_band)
        low = self.band_peak(VLF_BAND)
        high = self.band_peak(HF_BAND)
        if target is None or low is None or high is None:
            return False
        floor = max(target.power, 1e-15)
        return low.power > floor * SPLIT_RATIO and high.power > floor * SPLIT_RATIO

    def describe(self) -> str:
        if not self.ready:
            # Three beats, which is what the transform needs; a *claim* about the target then waits for
            # `RESOLVE_CYCLES` of it, which is the branch below. (This used to name a constant that no
            # longer exists, so the one branch reached before any beats arrived raised `NameError`.)
            return "waiting for beats"
        if not self.resolves(self.target_hz):
            # Drawn but not yet claimed: the curve exists, the numbers under it are real, and calling a
            # peak at the target from two cycles of it would not be.
            return (f"{self.span_s:.0f} s of beats, resolving {self.resolution_hz:.3f} Hz — "
                    f"the shape is early, keep breathing")
        if self.split:
            return (f"split — energy above and below with {self.target_label} missing: one foot on the "
                    f"gas and one on the brake")
        target = self.band_peak(self.narrow_band)
        strongest = max(self.peaks, key=lambda p: p.power, default=None)
        if strongest is None:
            return "no structure yet"
        if target is not None and strongest.label == self.target_label:
            return f"one peak at {target.hz:.3f} Hz · {self.sharpness:.0%} of the power is in it"
        return (f"strongest peak at {strongest.hz:.3f} Hz ({strongest.label}) — "
                f"aim for one at {self.target_hz:g}")


@dataclass
class BeatState:
    """A snapshot after each beat. Everything a screen draws comes from here."""

    t: float = 0.0
    arrived: float | None = None    # when the source gave the newest beat, if later than `t`
    hr_bpm: float = 0.0
    hr_smoothed: float = 0.0
    rr_s: float = 0.0
    rmssd_ms: float = 0.0
    sdnn_ms: float = 0.0
    pnn50_pct: float = 0.0
    beats: int = 0
    contact: bool | None = None
    age_s: list = field(default_factory=list)
    tachogram: list = field(default_factory=list)
    spectrum: Spectrum = field(default_factory=lambda: Spectrum(np.zeros(0), np.zeros(0)))
    reject_fraction: float = 0.0
    """Share of beats the artifact filter threw away. On the state rather than only on the processor because
    a screen showing a *spectrum* has no way to tell a clean tachogram from a heavily filtered one by eye."""


class BeatTimeline:
    """When the beats of a packet happened, from the intervals that the sensor measured and the times at
    which its packets arrived.

    **A packet arrives after its beats, by a time that changes from packet to packet.** A strap that
    notifies once a second delivers a beat between no time and a second after it happened, so the arrival
    is half a second late in the mean and scatters by a quarter of a second around that. The intervals
    carry no such error: the sensor measured them. So the times of the beats are the running sum of the
    intervals, and the arrivals only place that sum: no beat can arrive before it happens, so the packet
    that arrived soonest after its last beat, over `WINDOW_S`, says where the sum stands. A source that
    stamps each beat itself arrives equally late every time and keeps its own stamps.
    """

    WINDOW_S = 60.0
    """How long the soonest arrival is remembered. A notification each second gives 60 arrivals, of which the
    soonest is a sixtieth of a second late in the mean, and a clock of the sensor that runs a little fast or
    slow moves the sum against the arrivals by far less than that in a minute."""

    LOST_S = 1.3
    """Intervals that never arrived leave the sum behind the beats by their length, and every later packet
    then looks later by that much. A packet is at most one notification late, a second for a strap, and its
    delivery scatters by a little more; an arrival later than the soonest remembered by more than this has
    lost beats before it, and the sum starts again there. A loss too short to pass this leaves the times
    early by its length until the soonest arrivals from before it leave the window."""

    BACK_S = 3.0
    """An arrival this much sooner than the soonest remembered is a wall clock that was set back, and the
    sum starts again. A little sooner is only a packet that was delivered faster than any before it."""

    SLEW = 0.2
    """The most that the place of the sum moves at one packet, as a share of the intervals in it, so that a
    correction never puts a beat before the one that it follows."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start again: the next packet's last beat is put at its arrival, as a single packet must be."""
        self._sum: float | None = None
        self._late: deque[tuple[float, float]] = deque()
        self._place = 0.0

    def last_beat(self, arrived: float, intervals: list[float]) -> float:
        """The time of the last beat of a packet that `arrived` with these `intervals`, in seconds."""
        total = float(sum(intervals))
        if not intervals or not total > 0.0:
            return arrived              # no beat to place, and nothing learned about the sum
        if self._sum is None:
            return self._begin(arrived)
        at = self._sum + total
        late = arrived - at
        while self._late and self._late[0][0] < arrived - self.WINDOW_S:
            self._late.popleft()
        soonest = min((v for _, v in self._late), default=late)
        if late - soonest > self.LOST_S or soonest - late > self.BACK_S:
            return self._begin(arrived)
        self._late.append((arrived, late))
        self._sum = at
        limit = self.SLEW * total
        self._place += max(-limit, min(limit, min(soonest, late) - self._place))
        return at + self._place

    def _begin(self, arrived: float) -> float:
        self._sum = arrived
        self._late = deque([(arrived, 0.0)])
        self._place = 0.0
        return arrived


class HRVProcessor:
    """A rolling RR buffer, and the tachogram and spectrum derived from it.

    Time-based rather than beat-based, so every window means the same thing whether the person is at 50
    or 90 bpm.
    """

    def __init__(self, buffer_seconds: float = 180.0, spectrum_seconds: float = 120.0,
                 resample_hz: float = 4.0, smoothing_beats: int = 5,
                 reject_artifacts: bool = True, target_hz: float = TARGET_HZ,
                 narrow_half_hz: float = NARROW_HALF_HZ):
        self.buffer_seconds = buffer_seconds
        self.spectrum_seconds = spectrum_seconds
        self.resample_hz = resample_hz
        self.smoothing_beats = smoothing_beats
        self.target_hz = TARGET_HZ
        self.narrow_half_hz = NARROW_HALF_HZ
        self.set_target(target_hz, narrow_half_hz)
        # (beat time, rr seconds, contiguous), accepted only. `contiguous` rides with the beat so it
        # evicts in the same popleft and cannot desync from the array. See push() for what sets it.
        self._beats: deque[tuple[float, float, bool]] = deque()
        self._gap = False     # a beat has been dropped since the last accepted one
        # **The live readouts do not wait for the artifact filter.** The filter holds each beat until its
        # successor arrives, because the ectopic test needs it — about a second at resting rates. That latency
        # is invisible in a spectrum and shows in the heart rate now, which reads this.
        #
        # So the windowed things — the tachogram, the spectrum, RMSSD, SDNN, pNN50 — read accepted beats,
        # and the instantaneous ones read this. An artifact can therefore glitch the rate for one beat.
        self._live: deque[tuple[float, float]] = deque(maxlen=8)
        # **We had no artifact rejection at all**, and a single ectopic beat moves the spectral peak.
        # Measured before rejection existed: on a corrupted synthetic series the resonance ratio
        # went 0.63 against a clean 1.00, and the spectral peak was reported at 0.251 Hz instead of 0.101.
        #
        # A rejected beat is **withheld**, not deleted: the tachogram is interpolated over absolute beat
        # times, so leaving a gap makes the interpolation span it in a straight line. Dropping the beat
        # instead would compress the time axis and zero-filling would inject broadband power — both corrupt
        # the spectrum, which is the only thing this protocol reads.
        self.artifacts = ArtifactFilter() if reject_artifacts else None
        self._last_t: float | None = None    # newest beat time handed to any buffer; see push()
        self._timeline = BeatTimeline()

    def set_target(self, hz: float, half_hz: float | None = None) -> None:
        """Move the frequency every spectral claim is scored against, clamped to `TARGET_RANGE_HZ`.

        **Nothing is recomputed and nothing needs to be.** The spectrum is rebuilt from the buffered
        beats on the next one to arrive — about a second — and it is stamped with whatever the target
        is then. Clamped rather than refused, for the same reason `set_pacer_period` clamps: a control
        that silently ignores what you typed is worse than one that shows you the nearest thing it can
        do. A value that is not finite has no nearest thing, so it leaves the old one.
        """
        if math.isfinite(hz):
            lo, hi = TARGET_RANGE_HZ
            self.target_hz = float(min(hi, max(lo, float(hz))))
        if half_hz is not None and math.isfinite(half_hz):
            lo, hi = NARROW_HALF_RANGE_HZ
            self.narrow_half_hz = float(min(hi, max(lo, float(half_hz))))

    def _empty_spectrum(self) -> Spectrum:
        """An unready spectrum that still knows the target, so a screen can draw the marked band and
        say what it is aiming at before the first beat arrives."""
        return Spectrum(np.zeros(0), np.zeros(0), target_hz=self.target_hz,
                        narrow_half_hz=self.narrow_half_hz)

    def reset(self) -> None:
        self._beats.clear()
        self._gap = False
        self._live.clear()
        self._last_t = None
        self._timeline.reset()
        if self.artifacts is not None:
            self.artifacts.reset()

    def replay(self, step_s: float = 2.0) -> list["BeatState"]:
        """The states this processor would have produced over the beats it still holds, sampled every
        `step_s` seconds and oldest first.

        **So that a fresh scale does not start empty.** Changing the metric on the HRV screen rebuilds
        the normaliser — it must, since a rolling window holds values in the old metric's units — and the
        rebuilt one then had no history, so the feedback went silent for a quarter of its window. Every
        number it needs is already here: the beats are kept for three minutes, and a metric is a pure
        function of a state. Recomputing costs one FFT per sampled point, a few milliseconds for a full
        buffer, against thirty seconds of silence.

        The beats are replayed through a clone with artifact rejection off, because these are the beats
        that already passed it. Nothing here touches this processor's own state.
        """
        beats = list(self._beats)
        if len(beats) < 2:
            return []
        clone = HRVProcessor(buffer_seconds=self.buffer_seconds,
                             spectrum_seconds=self.spectrum_seconds,
                             resample_hz=self.resample_hz, smoothing_beats=self.smoothing_beats,
                             reject_artifacts=False, target_hz=self.target_hz,
                             narrow_half_hz=self.narrow_half_hz)
        out: list[BeatState] = []
        sampled_at = beats[0][0] - step_s
        for bt, rr, _ in beats:
            state = clone.push(Beat(t=bt, hr_bpm=60.0 / rr, rr_intervals=[rr]))
            if bt - sampled_at >= step_s:
                sampled_at = bt
                out.append(state)
        return out

    @property
    def reject_fraction(self) -> float:
        """Share of beats thrown away, 0…1. A signal-quality figure a screen can show: this protocol reads
        a *spectrum*, so a reader has no way to tell a clean tachogram from a filtered one by eye."""
        return self.artifacts.reject_fraction if self.artifacts is not None else 0.0

    def push(self, beat: Beat) -> BeatState:
        t = beat.t
        if not math.isfinite(t) or not all(math.isfinite(rr) for rr in beat.rr_intervals):
            # Stamps count back from the arrival and the intervals, so one non-finite value spoils them all;
            # an infinite stamp would stay newest and drop every later beat.
            if self._last_t is None:
                return BeatState(spectrum=self._empty_spectrum())
            return self._state(self._last_t, beat)
        # Timestamp each RR in the packet backwards from its last beat, so a batched packet does not stack
        # several beats at one instant and manufacture a spike in the tachogram.
        # An RR interval is the time *between* beats, so the beat it timestamps is the one that ends it —
        # the last interval in a packet ends at the packet's last beat, not one interval before it. Getting
        # this backwards shifts the whole tachogram by one beat, which is invisible in the numbers and wrong
        # in the spectrum. Where that last beat lies is the timeline's to say: the arrival is late by a
        # time that scatters from packet to packet.
        offset = sum(beat.rr_intervals)
        # A strap reporting no skin contact is reporting that its intervals are whatever the detector
        # latched onto — the state still carries `contact` so the screen can say so, but the buffers
        # must not train on it. The hole this leaves is a dropout, which the spectrum's gap handling
        # (`GAP_FACTOR`) already knows how to refuse.
        intervals = beat.rr_intervals if beat.contact is not False else []
        if beat.contact is False:
            self._timeline.reset()      # its intervals are not beats, so the sum of the beats ends here
        last = self._timeline.last_beat(t, intervals)
        if not intervals:
            self._gap = True        # a no-contact or HR-only packet is a hole between accepted beats
        for rr in intervals:
            offset -= rr                            # decremented first, and outside any validity check
            bt = last - offset
            # Arrival time is wall clock and BLE delivery jitter is unbounded, so a delayed packet
            # followed by a prompt one can compute a beat *before* its predecessor — and `np.interp`
            # returns silent garbage on a non-monotonic grid. A beat that would go backwards is a
            # transport artifact, not a heartbeat; it is skipped, and the interval it carried is simply
            # a slightly longer hole for the interpolation to span.
            if self._last_t is not None and bt <= self._last_t:
                self._gap = True
                continue
            # One range gate, the artifact filter's own (333-2000 ms). There used to be a second,
            # wider one here (250-2500 ms), and the seam between them was a hole: a 4 s missed-beat
            # interval — the most common real artifact — fell outside the wide gate, never reached the
            # filter, and so never counted in `reject_fraction`, while the filter's pending beat was
            # left unsettled across the skip.
            if RANGE_MIN_S <= rr <= RANGE_MAX_S:
                self._last_t = bt
                self._live.append((bt, rr))
                if self.artifacts is None:
                    self._beats.append((bt, rr, not self._gap))
                    self._gap = False
                    continue
            elif self.artifacts is None:
                self._gap = True
                continue
            # One beat of latency: the ectopic test needs the following interval. Out-of-range intervals
            # go through the filter too, so it counts them rather than a pre-gate dropping them unseen.
            for verdict in self.artifacts.push(bt, rr):
                if verdict.accepted:
                    self._beats.append((verdict.t, verdict.rr, not self._gap))
                    self._gap = False
                else:
                    self._gap = True        # a withheld beat breaks the next pair's adjacency
        while self._beats and self._beats[0][0] < t - self.buffer_seconds:
            self._beats.popleft()
        return self._state(t, beat)

    def _state(self, t: float, beat: Beat) -> BeatState:
        if not self._live:
            return BeatState(t=t, hr_bpm=beat.hr_bpm, hr_smoothed=beat.hr_bpm,
                             contact=beat.contact, spectrum=self._empty_spectrum())
        # Instantaneous, from every plausible beat: the readout cannot wait a beat. See `_live`.
        live_hr = np.asarray([60.0 / r for _bt, r in self._live])
        window = live_hr[-self.smoothing_beats:]

        # Windowed, from accepted beats only: an ectopic moves the spectral peak and wrecks RMSSD.
        times = np.asarray([bt for bt, _, _ in self._beats])
        rr = np.asarray([r for _, r, _ in self._beats])
        hr = 60.0 / rr if rr.size else np.zeros(0)
        if rr.size > 1:
            # Successive differences between adjacent beats only. `contiguous` marks each beat that
            # followed its predecessor directly; arrival times cannot tell on a BLE strap. See push().
            contiguous = np.asarray([c for _, _, c in self._beats], dtype=bool)
            diffs = (np.diff(rr) * 1000.0)[contiguous[1:]]
        else:
            diffs = np.zeros(0)
        return BeatState(
            t=t,
            arrived=beat.arrived,
            hr_bpm=float(live_hr[-1]),
            hr_smoothed=float(window.mean()),
            rr_s=float(self._live[-1][1]),
            rmssd_ms=float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else 0.0,
            sdnn_ms=float(np.std(rr) * 1000.0) if rr.size else 0.0,
            pnn50_pct=float(100.0 * np.mean(np.abs(diffs) > 50.0)) if diffs.size else 0.0,
            beats=len(self._live) if len(self._live) < self._live.maxlen else len(self._beats) + 1,
            contact=beat.contact,
            age_s=[float(t - bt) for bt in times],
            tachogram=[float(v) for v in hr],
            spectrum=self._spectrum(t, times, hr) if rr.size else self._empty_spectrum(),
            reject_fraction=self.reject_fraction,
        )

    def _spectrum(self, t: float, times: np.ndarray, hr: np.ndarray) -> Spectrum:
        """Welch-ish: one Hann-windowed rFFT over the resampled tachogram.

        **Computed from the first beats it is defined on**, which is three of them. The only floors left
        are arithmetic ones — two points to interpolate between, four grid samples to transform — because
        every other floor this had was a claim about accuracy dressed up as a claim about possibility.
        Accuracy is the caller's question and `Spectrum.resolves` is where it is asked, per frequency:
        the curve is drawn from the start, the sentence naming a peak waits for four cycles of it, and the
        split waits for a window that can see the very-low-frequency band at all.
        """
        empty = self._empty_spectrum()
        keep = times >= t - self.spectrum_seconds
        bt, values = times[keep], hr[keep]
        if bt.size < MIN_BEATS_FOR_SPECTRUM:
            return empty
        # Only the contiguous run since the last dropout — see GAP_FACTOR for why a ramp across a
        # silent stretch is worse than an honest "not ready".
        spacing = np.diff(bt)
        threshold = max(GAP_MIN_S, GAP_FACTOR * float(np.median(spacing)))
        gaps = np.flatnonzero(spacing > threshold)
        if gaps.size:
            start = int(gaps[-1]) + 1
            bt, values = bt[start:], values[start:]
        if bt.size < MIN_BEATS_FOR_SPECTRUM or bt[-1] <= bt[0]:
            return empty
        grid = np.arange(bt[0], bt[-1], 1.0 / self.resample_hz)
        if grid.size < MIN_GRID_SAMPLES:
            return empty
        series = np.interp(grid, bt, values)
        series = series - series.mean()
        win = np.hanning(series.size)
        fft = np.fft.rfft(series * win)
        freqs = np.fft.rfftfreq(series.size, d=1.0 / self.resample_hz)
        denom = (win ** 2).sum() * self.resample_hz
        power = (np.abs(fft) ** 2) / max(denom, 1e-12)
        keep_hz = freqs <= 0.5
        return Spectrum(freqs[keep_hz], power[keep_hz], span_s=float(bt[-1] - bt[0]),
                        target_hz=self.target_hz, narrow_half_hz=self.narrow_half_hz)
