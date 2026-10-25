"""DSP tests against signals whose answers are known in advance.

A metric that merely runs is worthless; these check values. A 10 µV peak sine has an RMS of
10/√2 ≈ 7.07 µV, white noise has near-maximal complexity where a sine has little, identical
signals cohere at 1. Where a tolerance is loose it is because the estimator is genuinely noisy,
not because the expected value is vague.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import counts

from nfb.dsp import (
    ALPHA, BETA, THETA, AlignedWindower, Band, BandPower, OnePole, PairMetric, Pipeline,
    Windower, WindowMetric, band_amplitude, band_ratio, bandpass, coherence, lempel_ziv,
    phase_locking_value, sample_entropy,
)
from nfb.stream import Block, ChannelSpec, Gap

RATE = 250.0
SPEC = ChannelSpec(index=0, rate_hz=RATE, unit="uV", scale=1.0)


def sine(freq, seconds=4.0, amp=10.0, rate=RATE, phase=0.0):
    t = np.arange(int(seconds * rate)) / rate
    return amp * np.sin(2 * np.pi * freq * t + phase)


def blocks(x, spec=SPEC, size=8, state=1, start=0):
    """Chop a signal into Blocks the way the device delivers it."""
    out = []
    for i in range(0, len(x) - size + 1, size):
        out.append(Block(spec.index, start + i, tuple(int(v) for v in x[i:i + size]),
                         spec, state=state))
    return out


# --- filters ------------------------------------------------------------------------------

def test_streaming_filter_matches_filtering_the_whole_signal():
    """The property that breaks silently if filter state is mishandled."""
    x = sine(10) + sine(40, amp=5)
    whole = bandpass(ALPHA, RATE).process(x)
    f = bandpass(ALPHA, RATE)
    pieces = np.concatenate([f.process(x[i:i + 8]) for i in range(0, len(x), 8)])
    assert np.allclose(whole, pieces, atol=1e-12)


def test_bandpass_keeps_its_band_and_rejects_outside_it():
    f_in = bandpass(ALPHA, RATE)
    f_out = bandpass(ALPHA, RATE)
    passed = f_in.process(sine(10))[500:]      # skip the filter's start-up transient
    blocked = f_out.process(sine(35))[500:]
    assert passed.std() > 0.9 * (10 / np.sqrt(2))
    assert blocked.std() < 0.02 * (10 / np.sqrt(2))


def test_band_above_nyquist_is_refused():
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass(Band(30, 45, "gamma"), 62.5)      # a real config: GSR-rate channel


def test_filter_reports_a_plausible_latency():
    """A narrow band at a high rate is slow even at the default order 2: the 8-12 Hz bandpass
    costs ~110 ms on its own, which dominates every stage except the smoothing constant."""
    lat = bandpass(ALPHA, RATE).latency_seconds
    assert 0.08 < lat < 0.15
    # corroborate the analytic group delay against the impulse response peak
    imp = bandpass(ALPHA, RATE).process(np.concatenate(([1.0], np.zeros(400))))
    assert abs(np.argmax(np.abs(imp)) / RATE - lat) < 0.06
    assert OnePole(0.5, RATE).latency_seconds == 0.5


def test_onepole_converges_to_a_constant():
    s = OnePole(0.1, RATE)
    out = s.process(np.full(int(RATE), 5.0))
    assert out[-1] == pytest.approx(5.0, rel=1e-3)


# --- windowing ----------------------------------------------------------------------------

def test_windower_emits_on_the_absolute_grid():
    w = Windower(SPEC, size=64, hop=32)
    got = [win for b in blocks(sine(10, seconds=2.0)) for win in w.push(b)]
    assert [win.start_index for win in got] == list(range(0, len(got) * 32, 32))
    assert all(len(win.samples) == 64 for win in got)


def test_no_window_spans_a_gap():
    w = Windower(SPEC, size=64, hop=64)
    x = sine(10, seconds=2.0)
    before = blocks(x)[:4]                                  # 32 samples, not a full window
    after = blocks(x, start=10_000)[:16]                    # jumps forward: a gap
    got = [win for b in before + after for win in w.push(b)]
    assert w.buffer.discontinuities == 1
    for win in got:
        assert win.start_index >= 10_000, "window straddled the gap"


def test_windows_inherit_invalidity_from_the_device_state():
    w = Windower(SPEC, size=64, hop=64)
    got = [win for b in blocks(sine(10), state=2) for win in w.push(b)]
    assert got and not any(win.valid for win in got)


def test_aligned_windows_cover_identical_index_ranges():
    specs = {0: SPEC, 1: ChannelSpec(1, RATE, "uV", 1.0)}
    aw = AlignedWindower(specs, size=64, hop=64)
    x = sine(10, seconds=2.0)
    groups = []
    for b0, b1 in zip(blocks(x, specs[0]), blocks(x, specs[1])):
        aw.push(b0)
        groups.extend(aw.push(b1))
    assert groups
    for g in groups:
        assert len({w.start_index for w in g.values()}) == 1


def test_aligned_windows_reject_mixed_rates():
    specs = {0: SPEC, 1: ChannelSpec(1, 62.5, "uV", 1.0)}
    with pytest.raises(ValueError, match="one sample rate"):
        AlignedWindower(specs, size=64)


# --- continuous band power ----------------------------------------------------------------

def test_band_power_recovers_the_rms_of_a_sine_in_band():
    m = BandPower(SPEC, ALPHA, tau_seconds=0.2, output_hz=10)
    readings = [r for b in blocks(sine(10, seconds=6.0, amp=100.0)) for r in m.feed(b)]
    settled = [r.value for r in readings if r.t > 2.0]
    assert np.mean(settled) == pytest.approx(100 / np.sqrt(2), rel=0.05)
    assert readings[0].unit == "uV" and readings[0].name == "alpha_rms"


def test_band_power_ignores_a_sine_outside_its_band():
    m = BandPower(SPEC, ALPHA, tau_seconds=0.2, output_hz=10)
    readings = [r for b in blocks(sine(35, seconds=6.0, amp=100.0)) for r in m.feed(b)]
    settled = [r.value for r in readings if r.t > 2.0]
    assert np.mean(settled) < 0.05 * (100 / np.sqrt(2))


def test_band_power_output_rate_and_latency():
    m = BandPower(SPEC, ALPHA, tau_seconds=0.5, output_hz=10)
    readings = [r for b in blocks(sine(10, seconds=4.0)) for r in m.feed(b)]
    dt = np.diff([r.t for r in readings])
    assert np.allclose(dt, 0.1)
    # smoothing (0.5 s) plus the order-2 bandpass group delay (~0.11 s)
    assert 0.55 < m.latency_seconds < 0.70


def test_band_power_marks_unusable_data_rather_than_dropping_it():
    m = BandPower(SPEC, ALPHA)
    readings = [r for b in blocks(sine(10), state=2) for r in m.feed(b)]
    assert readings and not any(r.valid for r in readings)


# --- spectral and ratio metrics -----------------------------------------------------------

def test_band_amplitude_matches_a_known_sine():
    assert band_amplitude(sine(10, amp=50.0), RATE, ALPHA) == pytest.approx(
        50 / np.sqrt(2), rel=0.1)


def test_band_ratio_behaves_as_the_ratio_of_amplitudes():
    x = sine(6, amp=20.0) + sine(20, amp=10.0)      # theta 20µV, beta 10µV
    assert band_ratio(x, RATE, THETA, BETA) == pytest.approx(2.0, rel=0.2)


# --- cross-channel ------------------------------------------------------------------------

def test_identical_signals_are_fully_coherent_and_phase_locked():
    x = sine(10) + 0.1 * np.random.default_rng(0).standard_normal(int(4 * RATE))
    assert coherence(x, x.copy(), RATE, ALPHA) == pytest.approx(1.0, abs=1e-6)
    assert phase_locking_value(x, x.copy(), RATE, ALPHA) == pytest.approx(1.0, abs=1e-6)


def test_independent_noise_is_incoherent():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(int(8 * RATE))
    b = rng.standard_normal(int(8 * RATE))
    assert coherence(a, b, RATE, ALPHA) < 0.3


def test_phase_locking_survives_a_phase_offset_that_coherence_also_sees():
    a = sine(10)
    b = sine(10, phase=np.pi / 3)
    # not exactly 1: the two signals differ at t=0, so their filter transients differ, and the
    # Hilbert transform has edge effects on a finite window
    assert phase_locking_value(a, b, RATE, ALPHA) == pytest.approx(1.0, abs=0.05)


def test_pair_metric_emits_for_the_channel_pair():
    specs = {0: SPEC, 1: ChannelSpec(1, RATE, "uV", 1.0)}
    m = PairMetric(specs, "alpha_coh",
                   lambda g: coherence(g[0].samples, g[1].samples, RATE, ALPHA),
                   window_seconds=2.0)
    x = sine(10, seconds=6.0, amp=100.0)
    readings = []
    for b0, b1 in zip(blocks(x, specs[0]), blocks(x, specs[1])):
        m.feed(b0)
        readings.extend(m.feed(b1))
    assert readings and readings[0].channel == (0, 1)
    assert all(r.value > 0.9 for r in readings)
    assert m.latency_seconds == pytest.approx(2.0)


# --- nonlinear ----------------------------------------------------------------------------

def test_noise_is_more_complex_than_a_sine():
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(500)
    wave = sine(10, seconds=2.0)
    assert sample_entropy(noise) > sample_entropy(wave)
    assert lempel_ziv(noise) > lempel_ziv(wave)


def test_lempel_ziv_is_near_zero_for_a_constant_and_near_one_for_noise():
    rng = np.random.default_rng(3)
    assert lempel_ziv(np.ones(512)) < 0.1
    assert 0.7 < lempel_ziv(rng.standard_normal(512)) < 1.3


def test_sample_entropy_of_a_constant_is_zero():
    assert sample_entropy(np.ones(100)) == 0.0


def test_window_metric_reports_complexity_over_windows():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(int(6 * RATE)) * 50
    m = WindowMetric(SPEC, "lziv", lambda w: lempel_ziv(w.samples), window_seconds=2.0)
    readings = [r for b in blocks(x) for r in m.feed(b)]
    assert len(readings) >= 2
    assert all(0.5 < r.value < 1.5 for r in readings)
    assert m.latency_seconds == pytest.approx(2.0)


# --- pipeline -----------------------------------------------------------------------------

def test_pipeline_reports_worst_case_latency_and_counts_gaps():
    p = Pipeline().add(
        BandPower(SPEC, ALPHA, tau_seconds=0.3),
        WindowMetric(SPEC, "lziv", lambda w: lempel_ziv(w.samples), window_seconds=2.0),
    )
    assert p.latency_seconds == pytest.approx(2.0)      # the window metric dominates
    assert p.feed(Gap(0, 8)) == []
    assert p.gaps == 1
    readings = [r for b in blocks(sine(10, seconds=4.0, amp=80.0)) for r in p.feed(b)]
    assert {r.name for r in readings} == {"alpha_rms", "lziv"}


def test_pipeline_resets_filter_state_across_a_gap():
    """A gap must not smear across the discontinuity: after it, output reflects new data."""
    m = BandPower(SPEC, ALPHA, tau_seconds=0.1, output_hz=50)
    loud = blocks(sine(10, seconds=2.0, amp=200.0))
    quiet = blocks(np.zeros(int(2 * RATE)), start=100_000)
    for b in loud:
        m.feed(b)
    after = [r.value for b in quiet for r in m.feed(b)]
    assert after[-1] < 1.0, "energy from before the gap persisted"


# --- amplitude conventions ----------------------------------------------------------------

def test_meanabs_envelope_is_two_over_pi_of_peak():
    """The meanabs envelope smooths abs(bandpass(x)); for a sine that is 2/pi of peak against RMS's 1/sqrt(2),
    so the two conventions differ by a known factor and thresholds must not be mixed."""
    x = sine(10, seconds=6.0, amp=100.0)
    rms = BandPower(SPEC, ALPHA, tau_seconds=0.2, envelope="rms")
    mab = BandPower(SPEC, ALPHA, tau_seconds=0.2, envelope="meanabs")
    got = {}
    for name, m in (("rms", rms), ("meanabs", mab)):
        vals = [r.value for b in blocks(x) for r in m.feed(b) if r.t > 2.0]
        got[name] = float(np.mean(vals))
    assert got["rms"] == pytest.approx(100 / np.sqrt(2), rel=0.05)
    assert got["meanabs"] == pytest.approx(100 * 2 / np.pi, rel=0.05)
    assert got["meanabs"] / got["rms"] == pytest.approx(2 * np.sqrt(2) / np.pi, rel=0.05)


def test_unknown_envelope_is_refused():
    with pytest.raises(ValueError, match="envelope must be"):
        BandPower(SPEC, ALPHA, envelope="peak")


def test_lower_filter_order_costs_less_latency():
    """The main latency lever, and why order 2 is the default."""
    lat = {o: bandpass(ALPHA, RATE, order=o).latency_seconds for o in (2, 4, 6)}
    assert lat[2] < lat[4] < lat[6]
    assert lat[4] / lat[2] == pytest.approx(2.0, rel=0.2)
    # order 6 used to report a negative delay via sos2tf; per-section summing fixed it
    assert all(v > 0 for v in lat.values())


def test_delay_and_smoothing_are_reported_separately():
    """650ms of "latency" was mostly a chosen smoothing constant, not an imposed lag."""
    m = BandPower(SPEC, ALPHA, tau_seconds=0.5)
    assert m.delay_seconds == pytest.approx(0.111, abs=0.02)    # the irreducible part
    assert m.smoothing_seconds == 0.5                           # the knob
    assert m.latency_seconds == pytest.approx(m.delay_seconds + 0.5)
    fast = BandPower(SPEC, ALPHA, tau_seconds=0.05)
    assert fast.delay_seconds == m.delay_seconds                # unchanged by smoothing
    assert fast.latency_seconds < 0.2

    w = WindowMetric(SPEC, "lziv", lambda win: lempel_ziv(win.samples), window_seconds=2.0)
    assert w.delay_seconds == pytest.approx(2.0) and w.smoothing_seconds == 0.0

    p = Pipeline().add(m, w)
    assert p.delay_seconds == pytest.approx(2.0)
    assert p.smoothing_seconds == 0.5


def test_measured_step_response_beats_the_nominal_sum():
    """A step reaches half its final value well before `latency_seconds` elapses."""
    t = np.arange(int(6 * RATE)) / RATE
    x = np.where(t < 2.0, 0.0, 50 * np.sin(2 * np.pi * 10 * t))
    m = BandPower(SPEC, ALPHA, tau_seconds=0.5, output_hz=RATE)
    rs = [r for b in blocks(x) for r in m.feed(b)]
    ts = np.array([r.t for r in rs])
    vs = np.array([r.value for r in rs])
    final = np.median(vs[ts > 5.0])
    assert final == pytest.approx(50 / np.sqrt(2), rel=0.05)
    half = ts[np.argmax(vs >= 0.5 * final)] - 2.0
    assert 0.2 < half < 0.4, half
    assert half < m.latency_seconds


# --- cFIR envelope (low-latency) ----------------------------------------------------------

def alpha_process(seconds=60.0, rate=RATE, alpha_uv=30.0, noise_uv=15.0, seed=0):
    """A narrowband alpha-like process with a wandering envelope, plus broadband noise."""
    from scipy import signal as sg
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    nyq = rate / 2
    sos = sg.butter(4, [8 / nyq, 12 / nyq], btype="bandpass", output="sos")
    a = sg.sosfilt(sos, rng.standard_normal(n))
    a *= alpha_uv / a.std()
    return a + rng.standard_normal(n) * noise_uv


def effective_delay_and_r(est, truth, rate=RATE, maxlag=1.0):
    """Score a causal estimate at the lag that suits it best — its real effective delay."""
    warm = int(5 * rate)
    best = (-1.0, 0)
    for d in range(int(maxlag * rate)):
        a = est[d:]
        b = truth[:len(truth) - d] if d else truth
        m = min(len(a), len(b))
        r = float(np.corrcoef(a[warm:m], b[warm:m])[0, 1])
        if r > best[0]:
            best = (r, d)
    return best[0], best[1] / rate


def test_cfir_recovers_the_amplitude_of_a_sine():
    from nfb.dsp import CFIREnvelope
    est = CFIREnvelope(ALPHA, RATE, delay_seconds=0.1)
    out = est.process(sine(10, seconds=6.0, amp=100.0))
    assert np.mean(out[int(2 * RATE):]) == pytest.approx(100.0, rel=0.1)


def test_cfir_gives_phase_as_well_as_amplitude():
    """The analytic signal is the point of the method: phase comes free, which Rect cannot give."""
    from nfb.dsp import CFIREnvelope
    est = CFIREnvelope(ALPHA, RATE, delay_seconds=0.1)
    y = est.analytic(sine(10, seconds=6.0, amp=100.0))
    phase = np.unwrap(np.angle(y[int(3 * RATE):]))
    measured_hz = np.mean(np.diff(phase)) * RATE / (2 * np.pi)
    assert measured_hz == pytest.approx(10.0, rel=0.02)


def test_cfir_streams_identically_to_processing_the_whole_signal():
    from nfb.dsp import CFIREnvelope
    x = alpha_process(seconds=20.0)
    whole = CFIREnvelope(ALPHA, RATE, 0.1).process(x)
    chunked = CFIREnvelope(ALPHA, RATE, 0.1)
    pieces = np.concatenate([chunked.process(x[i:i + 8]) for i in range(0, len(x), 8)])
    assert np.allclose(whole, pieces[:len(whole)], atol=1e-9)


def test_cfir_beats_bandpass_and_smooth_at_matched_delay():
    """The paper's central comparison, on our own pipeline: cFIR reaches a given accuracy at
    roughly half the delay of bandpass-rectify-smooth."""
    from nfb.dsp import CFIREnvelope, OnePole, true_envelope
    x = alpha_process(seconds=60.0)
    truth = true_envelope(x, ALPHA, RATE)

    r_cfir, d_cfir = effective_delay_and_r(
        CFIREnvelope(ALPHA, RATE, delay_seconds=0.1).process(x), truth)
    f, sm = bandpass(ALPHA, RATE), OnePole(0.5, RATE)
    rect = np.sqrt(np.maximum(sm.process(f.process(x) ** 2), 0))
    r_rect, d_rect = effective_delay_and_r(rect, truth)

    assert r_cfir > 0.9, r_cfir            # paper reports 0.85 +- 0.1 on real EEG at 100 ms
    assert r_cfir > r_rect                 # more accurate...
    assert d_cfir < d_rect                 # ...and sooner
    assert d_cfir < 0.15 and d_rect > 0.2


def test_cfir_metric_reports_zero_smoothing():
    from nfb.dsp import CFIRBandPower
    m = CFIRBandPower(SPEC, ALPHA, delay_seconds=0.05)
    assert m.smoothing_seconds == 0.0
    assert m.delay_seconds == 0.05
    readings = [r for b in blocks(sine(10, seconds=6.0, amp=100.0)) for r in m.feed(b)]
    settled = [r.value for r in readings if r.t > 3.0]
    assert np.mean(settled) == pytest.approx(100.0, rel=0.1)
    assert readings[0].name == "alpha_cfir" and readings[0].unit == "uV"


def test_cfir_reads_the_band_and_not_the_drift_under_it():
    """**A cFIR's stopband is whatever a least-squares fit happened to achieve**, which is fine against
    neighbouring EEG and nowhere near enough against this amplifier's drift.

    Measured on two recordings of a real hookup — an 11.7–14.7 Hz band whose true amplitude is about
    2.1 µV — the estimator fed raw samples reported **12.3 µV and 22.1 µV**, six and eleven times the
    signal, because 99.8% of the raw power sits below 1 Hz. A reward on it therefore tracked the electrode
    settling rather than the rhythm, and the IIR filter kinds did not have the problem because a real
    band-pass rejects what the fit only attenuates.

    Here that is a 2 µV rhythm sitting on the offset and slow swing a recording actually carries.

    **Asked for and not assumed**, because at the amplifier's 0.5 Hz corner the offset never arrives and
    the same capture reads within 1% without any of this — so the caller says which case it is in, and
    `training.drift_for` is the one place that decides.
    """
    from nfb.dsp import CFIREnvelope

    rate, seconds = 250.0, 40.0
    t = np.arange(int(rate * seconds)) / rate
    rhythm = 2.0 * np.sin(2 * np.pi * 13.0 * t)
    drift = -174.0 + 60.0 * np.sin(2 * np.pi * 0.05 * t) + 25.0 * np.sin(2 * np.pi * 0.3 * t)

    band = Band(11.7, 14.7, "smr")
    clean = CFIREnvelope(band, rate).process(rhythm)[int(5 * rate):]
    unasked = CFIREnvelope(band, rate).process(rhythm + drift)[int(5 * rate):]
    fixed = CFIREnvelope(band, rate, drift_hz=0.5).process(rhythm + drift)[int(5 * rate):]
    assert np.median(clean) == pytest.approx(2.0, rel=0.05), "the rhythm alone reads its own amplitude"
    assert np.median(unasked) > 5 * np.median(clean), \
        f"the leak, left in when nobody asks: {np.median(unasked):.2f} against {np.median(clean):.2f}"
    assert np.median(fixed) == pytest.approx(np.median(clean), rel=0.1), \
        f"and gone when they do: {np.median(fixed):.2f} against {np.median(clean):.2f}"

    # The corner sits two octaves under the band, so CFC's lowest bin keeps its own bottom edge...
    low = CFIREnvelope(Band(1.0, 5.0, "delta"), rate, drift_hz=0.5)
    assert low._drift is not None
    slow = 10.0 * np.sin(2 * np.pi * 1.2 * t)
    assert np.median(low.process(slow)[int(10 * rate):]) == pytest.approx(10.0, rel=0.1)
    # ...and a band that *is* the slow signal is left alone whatever the caller asks for.
    assert CFIREnvelope(Band(0.05, 0.2, "ilf"), rate, drift_hz=0.5)._drift is None


def test_the_drift_high_pass_follows_the_device_corner_and_nothing_else():
    """One place decides, and it decides on the amplifier's own setting.

    At 0.5 Hz the offset is gone before the samples arrive and a high-pass here would be answering a
    question nobody asked; at 0.01 Hz it is the difference between measuring a rhythm and measuring an
    electrode settling. The corner is read from what is streaming — a recording carries it in its header —
    so this holds on a replay as well as on a live device.
    """
    from nfb_console.training import DEFAULT_CORNER_HZ, drift_for

    assert drift_for(0.5) is None and drift_for(DEFAULT_CORNER_HZ) is None
    assert drift_for(0.01) == 0.5
    assert drift_for(0.0) == 0.5, "DC coupling is the same case, more so"


def test_cfir_refuses_a_band_above_nyquist():
    from nfb.dsp import CFIREnvelope
    with pytest.raises(ValueError, match="Nyquist"):
        CFIREnvelope(Band(30, 45, "gamma"), 62.5)


def test_the_three_amplitude_conventions_relate_as_expected():
    """cFIR reports peak amplitude (analytic modulus), BandPower reports RMS or mean-abs.
    For a sine: peak : rms : meanabs = 1 : 0.707 : 0.637. Thresholds do not transfer."""
    from nfb.dsp import CFIRBandPower
    x = sine(10, seconds=8.0, amp=100.0)
    got = {}
    for name, m in (
        ("peak", CFIRBandPower(SPEC, ALPHA, delay_seconds=0.1)),
        ("rms", BandPower(SPEC, ALPHA, tau_seconds=0.2, envelope="rms")),
        ("meanabs", BandPower(SPEC, ALPHA, tau_seconds=0.2, envelope="meanabs")),
    ):
        vals = [r.value for b in blocks(x) for r in m.feed(b) if r.t > 3.0]
        got[name] = float(np.mean(vals))
    assert got["peak"] == pytest.approx(100.0, rel=0.05)
    assert got["rms"] / got["peak"] == pytest.approx(1 / np.sqrt(2), rel=0.05)
    assert got["meanabs"] / got["peak"] == pytest.approx(2 / np.pi, rel=0.05)


def test_our_elliptic_at_order_two_has_the_chosen_design():
    """The elliptic design is 2 biquads, 0.5 dB passband ripple, 60 dB stopband, which is
    `scipy.signal.ellip(2, 0.5, 60, ...)`.

    This pins three properties that separate that design from a Butterworth of the same order.
    """
    import numpy as np
    from scipy import signal

    from nfb.dsp import Band, bandpass
    from nfb.dsp.filters import ELLIPTIC_STOP_DB

    assert ELLIPTIC_STOP_DB == 60.0, "60 dB is the chosen stopband; 50 was a guess"

    band = (12.0, 15.0)
    rate = 256.0
    filt = bandpass(Band(*band, ""), rate, order=2, kind="elliptic", ripple_db=0.5)
    w, h = signal.sosfreqz(filt.sos, worN=200000, fs=rate)
    db = 20 * np.log10(np.abs(h) + 1e-300)
    inside = (w >= band[0]) & (w <= band[1])

    # 1. the passband ripples by the stated amount, and the *edges sit at the ripple bound* — not at
    #    -3 dB, which is what a Butterworth band edge means by definition
    assert db[inside].max() - db[inside].min() == pytest.approx(0.5, abs=0.02)
    for edge in band:
        assert np.interp(edge, w, db) - db[inside].max() == pytest.approx(-0.5, abs=0.02)

    # 2. finite transmission zeros on the unit circle, one below the band and one above. A Butterworth
    #    bandpass puts all of its zeros at DC and Nyquist, so a notch inside the audible range is the
    #    signature that settles it.
    zeros = np.roots(signal.sos2tf(filt.sos)[0])
    on_circle = [z for z in zeros if abs(abs(z) - 1.0) < 1e-6 and 1e-3 < abs(np.angle(z)) < np.pi - 1e-3]
    assert len(on_circle) >= 4, f"expected two conjugate pairs of transmission zeros, got {zeros}"
    notches = sorted(abs(np.angle(z)) / (2 * np.pi) * rate for z in on_circle)
    assert notches[0] < band[0] and notches[-1] > band[1], f"a notch each side: {notches}"

    # 3. and at this order it is *faster* than the Butterworth, which is the opposite of the usual story
    bw = bandpass(Band(*band, ""), rate, order=2, kind="butterworth")
    def group_delay_ms(f):
        ww, hh = signal.sosfreqz(f.sos, worN=200000, fs=rate)
        return np.interp(13.5, ww, -np.gradient(np.unwrap(np.angle(hh)), ww * 2 * np.pi)) * 1000
    assert group_delay_ms(filt) < group_delay_ms(bw) - 30


def _bin_spill(design, seconds=40.0, rate=250.0):
    """Worst neighbouring-bin reading as a fraction of the owning bin's, on a realistic background.

    A 25 µV peak at 12.5 Hz sits inside a 10–15 Hz bin, on genuine 1/f noise high-passed at 0.5 Hz the way
    the device does it. Normalising to the owning bin makes the answer independent of the amplitude
    convention, which matters because cFIR reports peak and `BandPower` reports RMS.
    """
    import numpy as np

    from nfb.dsp import Band, BandPower, CFIRBandPower, highpass
    from nfb.stream import Block, ChannelSpec

    spec = ChannelSpec(index=0, rate_hz=rate, unit="uV", scale=0.01)
    n = int(seconds * rate)
    rng = np.random.default_rng(9)
    freqs = np.fft.rfftfreq(n, 1 / rate)
    freqs[0] = freqs[1]
    pink = np.fft.irfft(np.fft.rfft(rng.normal(0, 1, n)) / np.sqrt(freqs), n)
    pink = highpass(0.5, rate).process(pink / np.std(pink) * 6.0)
    x = pink + 25 * np.sin(2 * np.pi * 12.5 * np.arange(n) / rate)

    def read(band):
        kind = design["kind"]
        if kind == "cfir":
            metric = CFIRBandPower(spec, band, delay_seconds=design["delay"], output_hz=20.0)
        else:
            metric = BandPower(spec, band, output_hz=20.0, order=design["order"], kind=kind,
                               envelope="rms", tau_seconds=0.16, ripple_db=design.get("rp", 0.5))
        out = []
        ints = counts(x, spec.scale)
        for i in range(0, n - 7, 8):
            block = Block(0, i, tuple(ints[i:i + 8].tolist()), spec)
            out += [r.value for r in metric.feed(block)]
        return float(np.median(out[len(out) // 3:]))

    owner = read(Band(10.0, 15.0, ""))
    return max(read(Band(5.0, 10.0, "")), read(Band(15.0, 20.0, ""))) / owner


def test_the_defaults_split_by_role_and_the_reward_is_the_steady_one():
    """**The reward went to the elliptic, and the argument that had kept it on cFIR was a bound.**

    cFIR's case is that it estimates amplitude directly and so carries no envelope smoother. That is also
    what its reading *feels* like: measured on a real 11.7–14.7 Hz session, both estimators fed the same
    signal with thresholds derived at the same 90% aim, cFIR crossed its threshold 172 times a minute and
    moved 32% of its own median between updates 50 ms apart, against the elliptic's 31 and 7%.

    What the steadiness costs is 48 ms — 168 ms against 216 to report half of a burst, measured through
    both shipping estimators. The figure this decision had previously been argued against, 450 ms, is
    `latency_seconds`: group delay plus the smoother's whole time constant, a conservative bound and not a
    response. Choosing a default off a bound is choosing it off the wrong number.

    The inhibit stays on cFIR at 300 ms, which is unaffected: it is judged on separation from its
    neighbours rather than on how it feels, and it wins that on measurement below.
    """
    from nfb_console.training import CFIR_DELAY, DEFAULT_FILTER

    # Keyed by **role**: the arguments are about what the estimate is used for, and neither mentions which
    # way the band is trained. A reward trained downward is still moment-locked.
    assert DEFAULT_FILTER == {"reward": "elliptic", "inhibit": "cfir", "monitor": "elliptic"}
    assert CFIR_DELAY == {"reward": 0.1, "inhibit": 0.3, "monitor": 0.1}

    ours = _bin_spill(dict(kind="cfir", delay=0.3))
    assert ours < 0.10, f"an inhibit that leaks a tenth of its neighbour is not a separate bin: {ours:.1%}"
    for label, design in (("elliptic 4", dict(kind="elliptic", order=4, rp=0.5)),
                          ("butterworth 4", dict(kind="butterworth", order=4)),
                          ("elliptic 2", dict(kind="elliptic", order=2, rp=0.5))):
        other = _bin_spill(design)
        assert ours <= other, f"{label} separates adjacent bins better than cFIR 300 ms: {other:.1%}"


def test_the_elliptic_default_is_one_coherent_design():
    """Order 2 with a 60 dB stopband and 0.5 dB of ripple is one design. A 1.0 dB ripple with those
    two was a loose ripple on a tight order, so the ripple default and the stopband default are pinned
    together."""
    from nfb.dsp.filters import ELLIPTIC_RIPPLE_DB, ELLIPTIC_STOP_DB
    from nfb_console.training import DEFAULT_ORDER, DEFAULT_RIPPLE

    assert (DEFAULT_ORDER, ELLIPTIC_RIPPLE_DB, ELLIPTIC_STOP_DB) == (2, 0.5, 60.0)
    assert DEFAULT_RIPPLE is ELLIPTIC_RIPPLE_DB, "one constant, not two that can drift apart"


def test_the_display_filter_leaves_cfcs_top_bin_alone():
    """The raw trace is drawn band-limited, and the corner has to clear the highest band anything measures.
    CFC's top bin is 25–45 Hz, so a corner at 40 would sit inside a band being measured — and the
    trace is what an operator reads to check what the bins are doing.

    Order 4 rather than 2 because a picture has no latency budget: it is flatter at 40 Hz *and* rejects the
    100 Hz mains harmonic that a 1000 sps hardware band limit lets through.
    """
    import numpy as np
    from scipy import signal

    from nfb_console.training import TrainingSession as TS

    assert TS.DISPLAY_LIMIT_HZ > 45.0 - 1e-9, "must clear CFC's 25-45 Hz bin"
    assert TS.DISPLAY_LIMIT_HZ < 50.0, "must stay below mains; the notch is the hardware's job"

    rate = 1000.0
    def response(order):
        sos = signal.butter(order, TS.DISPLAY_LIMIT_HZ / (rate / 2), btype="lowpass", output="sos")
        w, h = signal.sosfreqz(sos, worN=8000, fs=rate)
        db = 20 * np.log10(np.abs(h) + 1e-12)
        return lambda f: float(db[np.argmin(abs(w - f))])

    ours, plain = response(TS.DISPLAY_ORDER), response(2)
    assert ours(40.0) > plain(40.0), "the chosen order must be flatter at 40 Hz, not steeper"
    assert ours(40.0) > -2.0, f"40 Hz is attenuated by {ours(40.0):.1f} dB"
    assert ours(10.0) > -0.2, "alpha must be untouched"
    assert ours(100.0) < -25.0, f"the mains harmonic survives at {ours(100.0):.1f} dB"


def test_the_weighted_cfir_design_agrees_with_the_unweighted_one_at_unit_weights():
    """The weighted branch solved the wrong normal equations — plain transpose where a complex basis
    needs the conjugate transpose on both sides — so with all-ones weights, where the two branches must
    coincide by definition, it produced a *different filter*: the disagreement exceeded the largest tap,
    and `normalise` hid the magnitude error while leaving the response wrong. Latent only because nothing
    passed weights; the identity is the regression test."""
    from nfb.dsp.envelope import design_cfir

    band = Band(8.0, 12.0, "alpha")
    plain = design_cfir(band, 250.0, 0.1)
    weighted = design_cfir(band, 250.0, 0.1, weights=np.ones(2000))
    assert np.abs(plain - weighted).max() < 1e-9


def test_the_fast_filter_path_matches_the_reference_one_sample_for_sample(monkeypatch):
    """**`StreamFilter` calls the kernel under `scipy.signal.sosfilt` rather than the function.**

    The wrapper spends 18.7 of its 18.8 µs validating arguments when the block is eight samples — the size
    this device delivers, 125 times a second, per filter per channel — so the public call is 19× the cost
    of the work. It is a private name, so this pins the two together: same random data, same random block
    splits, and the state has to carry across blocks identically or a filter reset would go unnoticed.
    """
    import numpy as np
    from scipy import signal

    from nfb.dsp import filters

    assert filters._sos_kernel is not None, "the fast path is what production runs"
    rng = np.random.default_rng(4)
    x = rng.standard_normal(4096) * 40.0
    sos = signal.butter(4, [8.0, 12.0], btype="bandpass", fs=250.0, output="sos")

    fast = filters.StreamFilter(sos, 250.0)
    monkeypatch.setattr(filters, "_sos_kernel", None)
    reference = filters.StreamFilter(sos, 250.0)

    before = x.copy()
    cuts = np.cumsum(rng.integers(1, 40, size=400))
    cuts = [0, *[int(c) for c in cuts if c < len(x)], len(x)]
    a = np.concatenate([fast.process(x[i:j]) for i, j in zip(cuts, cuts[1:])])
    b = np.concatenate([reference.process(x[i:j]) for i, j in zip(cuts, cuts[1:])])
    assert np.allclose(a, b, rtol=0, atol=1e-12), f"max difference {np.abs(a - b).max():.3g}"
    # The kernel filters in place, so the one thing the wrapper was doing for us is a copy: without it
    # the caller's own array comes back filtered, which showed up as a raw trace holding filtered values.
    assert np.array_equal(x, before), "the input must not be written through"

    # and a reset has to clear both the same way
    fast.reset(); reference.reset()
    assert np.allclose(fast.process(x[:64]), reference.process(x[:64]), rtol=0, atol=1e-12)
