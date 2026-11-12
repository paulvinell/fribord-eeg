"""Wavelets: the properties every design holds, the burst detector, the fit and the muscle reading."""

import numpy as np
import pytest

from nfb.dsp.planted import Planted, pink, plant
from nfb.dsp.wavelet import BurstDetector, Wavelet, WaveletEnvelope, bursts, design_wavelet, outline

RATE = 250.0


def noise(n, seed=0):
    return pink(n, RATE, np.random.default_rng(seed))


DESIGNS = [Wavelet(10.0, 2), Wavelet(13.5, 5), Wavelet(20.0, 8, rise=0.9), Wavelet(40.0, 6, rise=0.75)]


@pytest.mark.parametrize("wavelet", DESIGNS)
def test_a_steady_sine_reads_its_amplitude_and_its_phase_now(wavelet):
    est = WaveletEnvelope(wavelet, RATE)
    t = np.arange(int(3 * RATE)) / RATE
    y = est.analytic(7.0 * np.cos(2 * np.pi * wavelet.frequency_hz * t + 0.4))
    # A short wavelet also passes a little of the sine's mirror at −f, which shows as a ripple.
    settled = y[len(est.taps):]
    assert np.abs(settled).mean() == pytest.approx(7.0, rel=0.02)
    phase = 2 * np.pi * wavelet.frequency_hz * t[len(est.taps):] + 0.4
    assert np.abs(np.angle(settled * np.exp(-1j * phase))).mean() < 0.1


@pytest.mark.parametrize("wavelet", DESIGNS)
def test_a_standing_offset_reads_nothing(wavelet):
    assert abs(design_wavelet(wavelet, RATE).sum()) < 1e-9


def test_the_length_is_a_number_of_cycles():
    assert Wavelet(40.0, 5).taps(RATE) * 4 == pytest.approx(Wavelet(10.0, 5).taps(RATE), abs=2)
    slow, fast = WaveletEnvelope(Wavelet(10.0, 8), RATE), WaveletEnvelope(Wavelet(10.0, 2), RATE)
    assert fast.width_hz > 3 * slow.width_hz        # fewer cycles: sooner, and less exact about frequency


def test_streaming_in_pieces_matches_the_whole():
    x = noise(5000, seed=3)
    whole = WaveletEnvelope(Wavelet(13.5, 5, rise=0.75), RATE).process(x)
    pieces = WaveletEnvelope(Wavelet(13.5, 5, rise=0.75), RATE)
    assert np.allclose(np.concatenate([pieces.process(x[i:i + 8]) for i in range(0, len(x), 8)]), whole)


def test_the_outline_peaks_where_rise_says_and_ends_at_zero():
    for rise in (0.1, 0.5, 0.9):
        env = outline(101, rise)
        assert env[0] == pytest.approx(0.0) and env[-1] == pytest.approx(0.0)
        assert int(np.argmax(env)) == pytest.approx(rise * 100, abs=1)


def test_a_burst_needs_its_length_and_one_that_dips_is_one():
    amplitude = np.zeros(1000)
    amplitude[100:160] = 5.0            # long enough
    amplitude[130:135] = 4.0            # a dip above the release level
    amplitude[400:410] = 5.0            # too short
    assert bursts(amplitude, RATE, level=4.5, min_seconds=0.1) == [(100, 125, 160)]


def test_a_gated_burst_is_confirmed_when_the_gate_holds():
    amplitude = np.zeros(1000)
    amplitude[100:200] = 5.0
    amplitude[400:500] = 5.0
    gate = np.zeros(1000, dtype=bool)
    gate[150:] = True
    gate[380:600] = False
    assert bursts(amplitude, RATE, level=4.5, min_seconds=0.1, gate=gate) == [(100, 150, 200)]


def test_the_detector_streams_as_it_runs_whole():
    rng = np.random.default_rng(4)
    amplitude = np.abs(np.convolve(rng.standard_normal(4000), np.ones(25) / 5, "same"))
    whole = bursts(amplitude, RATE, level=1.2, min_seconds=0.05)
    assert len(whole) > 5
    detector = BurstDetector(RATE, level=1.2, min_seconds=0.05)
    on = np.concatenate([detector.process(amplitude[i:i + 8]) for i in range(0, len(amplitude), 8)])
    assert detector.finish() == whole
    for start, confirmed, end in whole:
        assert not on[start:confirmed].any() and on[confirmed:end].all()


def fitted(spec, low, high, seed=0):
    from nfb.dsp.fit import fit_design
    rng = np.random.default_rng(seed)
    x = pink(int(10 * 60 * RATE), RATE, rng)
    plant(x, RATE, rng, spec)
    return fit_design(x, RATE, low, high)


def test_a_fit_reads_the_frequency_and_the_length_of_planted_bursts():
    fit = fitted(Planted(10.5, (0.5, 2.0), 12.0, 8, 0.3), 8, 13)
    assert fit.wavelet.frequency_hz == pytest.approx(10.5, abs=0.3)
    assert 8 < fit.wavelet.cycles < 16
    assert fit.bursts >= 8


def test_a_fit_refuses_a_background_with_no_bursts():
    with pytest.raises(ValueError, match="too few"):
        fitted(Planted(10.0, (0.5, 2.0), 0.0, 8), 8, 13)


@pytest.mark.parametrize("rate", [250.0, 1000.0])
def test_the_muscle_reading_rises_for_muscle_and_not_for_a_rhythm(rate):
    from nfb.dsp import MuscleEnvelope
    rng = np.random.default_rng(0)
    quiet = pink(int(20 * rate), rate, rng)
    rhythm, muscle = quiet.copy(), quiet.copy()
    t = np.arange(int(2 * rate)) / rate
    at = slice(int(8 * rate), int(10 * rate))
    rhythm[at] += 15.0 * np.cos(2 * np.pi * 40.0 * t)
    muscle[at] += 15.0 * rng.standard_normal(len(t))
    usual = np.median(MuscleEnvelope(rate).process(quiet)[int(rate):])
    assert MuscleEnvelope(rate).process(rhythm)[at][int(rate / 2):].mean() < 1.3 * usual
    assert MuscleEnvelope(rate).process(muscle)[at][int(rate / 2):].mean() > 3 * usual


def test_the_muscle_reading_streams_as_it_runs_whole():
    from nfb.dsp import MuscleEnvelope
    x = noise(3000, seed=5)
    pieces = MuscleEnvelope(RATE)
    assert np.allclose(np.concatenate([pieces.process(x[i:i + 8]) for i in range(0, len(x), 8)]),
                       MuscleEnvelope(RATE).process(x))
