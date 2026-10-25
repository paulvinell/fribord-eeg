"""Low-latency envelope estimation: complex-valued FIR (cFIR).

Why this exists. Belinskaia, Smetanin, Lebedev & Ossadtchi (J. Neural Eng. 2020,
doi:10.1088/1741-2552/abc8d7) trained parietal alpha with feedback at 244 ms, 494 ms and 744 ms
end-to-end latency. Only the shortest worked: at +250 ms the effect was much weaker, at +500 ms
sustained change was absent entirely. Their explanation is temporal specificity — feedback has to
arrive while the alpha spindle is still ongoing to reinforce it, and what changes is the spindle
*incidence rate*, not its amplitude or duration. They report that fielded NFB systems sit at
300–1000 ms, and call for work at 50 ms and below.

The conventional envelope — bandpass, rectify, smooth — is what they call "Rect", and it is the
method their comparison beats at *every* latency. Its problem is structural: delay is a
side-effect of filter order plus smoothing constant, so buying accuracy costs delay with no way to
choose the trade directly.

cFIR inverts that. A single complex-valued FIR maps the raw signal to the *analytic* signal of the
narrowband component, and the filter is designed by least squares against an ideal response that
has the delay you asked for built into its phase. Delay becomes a parameter, not a consequence.
The envelope is then just the modulus, with no smoothing stage at all — which is where the 500 ms
in a Rect pipeline actually goes.

At 100 ms delay the paper reports r = 0.85 ± 0.1 against the true (non-causal) envelope.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..stream import Block, ChannelSpec
from .filters import Band
from .metrics import Reading


def band_centre_gain(taps: np.ndarray, band: Band, rate_hz: float) -> float:
    """Magnitude response of complex taps at the band centre."""
    k = np.arange(len(taps))
    return float(np.abs(np.sum(taps * np.exp(-2j * np.pi * band.centre * k / rate_hz))))


def design_cfir(band: Band, rate_hz: float, delay_seconds: float, n_taps: int = 250,
                n_fft: int = 2000, weights: np.ndarray | None = None,
                normalise: bool = True) -> np.ndarray:
    """Least-squares complex FIR whose output approximates the band's analytic signal.

    `delay_seconds` is the delay the filter is designed for and may be **negative**, which asks
    for prediction — the design will simply fit it less well. Longer `n_taps` fits the ideal
    response more closely at no cost in delay, since the delay lives in the target's phase rather
    than in the filter length; it costs only arithmetic.

    With `normalise` (the default) the taps are scaled so a sine at the band centre reads its true
    amplitude. This matters: the least-squares fit attenuates in-band, and *by an amount that
    depends on the requested delay* — measured 68% of true amplitude at 50 ms and 87% at 100 ms
    before normalising against the ideal in-band response of 2. Without correction, an amplitude
    threshold would silently stop meaning the
    same thing whenever the delay setting changed. The correction is exact at the band centre and
    approximate towards the edges, which is the honest compromise; `band_centre_gain()` exposes
    the raw figure.

    Returns complex taps to be applied with `lfilter`. The envelope is the modulus of the result.
    """
    if n_taps < 8:
        raise ValueError("n_taps too small to fit a band")
    delay_samples = delay_seconds * rate_hz
    k = np.arange(n_fft)
    freqs = k / n_fft * rate_hz
    # Ideal target: analytic (positive frequencies only, doubled) band-limited to the band, with
    # linear phase corresponding to the requested delay.
    target = 2 * np.exp(-2j * np.pi * k / n_fft * delay_samples) * (
        (freqs >= band.low) & (freqs <= band.high)
    )
    # DFT matrix restricted to n_taps: rows are frequencies, columns are tap indices.
    basis = np.exp(-2j * np.pi / n_fft * np.outer(k, np.arange(n_taps)))
    if weights is None:
        taps = basis.conj().T @ target / n_fft
    else:
        # The conjugate transpose on both sides, B^H W B t = B^H W target — the plain transpose was
        # used here for a long time, which solves a different (and wrong) problem for a complex basis:
        # with unit weights it disagreed with the unweighted branch by more than the largest tap, a
        # different filter rather than a rescaled one, and `normalise` hid the magnitude while leaving
        # the response wrong. Latent because nothing passed weights; a test now pins the identity.
        w = np.diag(weights)
        taps = np.linalg.solve(basis.conj().T @ w @ basis, basis.conj().T @ w @ target)
    if normalise:
        gain = band_centre_gain(taps, band, rate_hz)
        if gain > 0:
            # normalise to 2, not 1: the ideal analytic response is 2 in-band, which is what makes
            # the modulus equal the true amplitude of a real sine rather than half of it
            taps = taps * (2.0 / gain)
    return taps


def true_envelope(x: np.ndarray, band: Band, rate_hz: float, order: int = 4) -> np.ndarray:
    """Non-causal reference envelope: zero-phase bandpass, then Hilbert modulus.

    Only valid offline — it uses future samples. This is the ground truth any causal estimator is
    scored against, and the reason accuracy can be measured at all.
    """
    nyq = rate_hz / 2
    sos = signal.butter(order, [band.low / nyq, band.high / nyq], btype="bandpass", output="sos")
    return np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))


DRIFT_CORNER_MAX_HZ = 0.5
"""Highest corner the drift high-pass in `CFIREnvelope` will use — see its `_drift` note. Never above the
0.5 Hz the artifact gate and the device's own coupling already treat as the bottom of EEG, and what a
caller passes as `drift_hz` when the amplifier is not doing the job itself."""

DRIFT_CORNER_FLOOR_HZ = 0.05
"""Below this the high-pass is not applied at all. A band whose bottom edge is down here *is* the slow
signal — infra-low training, the waveform screen's territory — and removing drift there would remove the
thing being trained. The rule below puts the corner two octaves under the band, so this only excludes
bands starting below 0.2 Hz, which no cFIR target in this program uses."""


class CFIREnvelope:
    """Streaming cFIR envelope estimator: blocks of samples in, envelope out.

    Stateful and causal, so processing a stream in pieces matches processing it whole.
    """

    def __init__(self, band: Band, rate_hz: float, delay_seconds: float = 0.1,
                 n_taps: int = 250, weights: np.ndarray | None = None, normalise: bool = True,
                 drift_hz: float | None = None):
        if band.high >= rate_hz / 2:
            raise ValueError(
                f"band {band.label} needs {band.high} Hz but Nyquist is {rate_hz / 2:g} Hz"
            )
        self.band = band
        self.rate_hz = rate_hz
        self.delay_seconds = delay_seconds
        self.taps = design_cfir(band, rate_hz, delay_seconds, n_taps, weights=weights,
                                normalise=normalise)
        self._zi = np.zeros(len(self.taps) - 1, dtype=complex)
        # **The drift high-pass, off unless the caller says the input still carries drift.** A cFIR is a
        # least-squares fit over the whole spectrum, so its stopband is whatever the fit achieved: about 6%
        # of the in-band gain at every frequency, DC included. That is nothing against neighbouring EEG and
        # everything against an electrode's standing offset. Measured on a capture made at the amplifier's
        # 0.01 Hz corner, an 11.7–14.7 Hz band whose true amplitude is 2.11 µV: this estimator reported
        # **12.30 µV**, and the offset accounts for all but 0.36 of the excess — 6% of −174 µV of DC is
        # five times the rhythm, so the reward follows the electrodes settling.
        #
        # `drift_hz` is the caller stating that nothing upstream has dealt with it, and it is a parameter
        # rather than a permanent stage because at the amplifier's *other* corner there is nothing to deal
        # with: through a 0.5 Hz pole the same capture reads 2.13 µV, an error of 1%. Whatever is passed is
        # capped two octaves under the band, so a 1–5 Hz bin keeps its own bottom edge, and ignored below
        # `DRIFT_CORNER_FLOOR_HZ`, where the drift is the signal being trained.
        self._drift = None
        self._drift_zi = None
        if drift_hz:
            corner = min(float(drift_hz), band.low / 4.0)
            if corner >= DRIFT_CORNER_FLOOR_HZ:
                self._drift = signal.butter(2, corner, btype="highpass", fs=rate_hz, output="sos")
                self._drift_zi = signal.sosfilt_zi(self._drift) * 0.0

    def reset(self) -> None:
        self._zi = np.zeros(len(self.taps) - 1, dtype=complex)
        if self._drift is not None:
            self._drift_zi = signal.sosfilt_zi(self._drift) * 0.0

    def analytic(self, x: np.ndarray) -> np.ndarray:
        """The complex narrowband estimate. Modulus is amplitude, angle is instantaneous phase."""
        x = np.asarray(x, dtype=np.float64)
        if not len(x):
            return np.zeros(0, dtype=complex)
        if self._drift is not None:
            x, self._drift_zi = signal.sosfilt(self._drift, x, zi=self._drift_zi)
        y, self._zi = signal.lfilter(self.taps, [1.0], x, zi=self._zi)
        return y

    def process(self, x: np.ndarray) -> np.ndarray:
        return np.abs(self.analytic(x))

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds


class CFIRBandPower:
    """Band amplitude via cFIR — the low-latency alternative to `BandPower`.

    Same `Metric` contract, but `delay_seconds` is whatever you asked for and
    `smoothing_seconds` is zero, because there is no smoothing stage. Emits amplitude in the
    channel's units, in the **peak** convention — the analytic modulus. **Not** comparable to
    `BandPower(envelope="rms")`, which this docstring claimed for a long time: for a steady sine this
    reads √2 larger (10.00 against 7.07 for a 10 µV tone, verified numerically), and "meanabs" reads
    0.900× the RMS figure again. A threshold moved between estimators without that factor silently
    stops meaning the same thing.
    """

    def __init__(self, spec: ChannelSpec, band: Band, delay_seconds: float = 0.1,
                 output_hz: float = 20.0, n_taps: int = 250, drift_hz: float | None = None):
        self.spec = spec
        self.band = band
        self.estimator = CFIREnvelope(band, spec.rate_hz, delay_seconds, n_taps,
                                      drift_hz=drift_hz)
        self.stride = max(1, int(round(spec.rate_hz / output_hz)))
        self._next_index: int | None = None
        self._expected: int | None = None

    @property
    def name(self) -> str:
        return f"{self.band.label}_cfir"

    @property
    def delay_seconds(self) -> float:
        return self.estimator.delay_seconds

    @property
    def smoothing_seconds(self) -> float:
        return 0.0

    @property
    def latency_seconds(self) -> float:
        return self.delay_seconds

    def feed(self, block: Block) -> list[Reading]:
        if block.channel != self.spec.index:
            return []
        if self._expected is not None and block.start_index != self._expected:
            self.estimator.reset()
            self._next_index = block.start_index
        self._expected = block.end_index

        x = np.asarray(block.samples, dtype=np.float64) * self.spec.scale
        amplitude = self.estimator.process(x)

        if self._next_index is None:
            self._next_index = block.start_index
        out = []
        while self._next_index < block.end_index:
            i = self._next_index - block.start_index
            out.append(Reading(
                name=self.name,
                value=float(amplitude[i]),
                t=self._next_index / self.spec.rate_hz,
                channel=self.spec.index,
                unit=self.spec.unit,
                valid=block.valid,
            ))
            self._next_index += self.stride
        return out
