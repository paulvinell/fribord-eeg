"""The same heartbeats at two sites: what the comparison recovers from them.

One synthetic heartbeat sequence is drawn at a forehead and at a reference sensor with a known delay
between them, known pulse heights, one beat missing at the forehead and one false beat there, so every
number the comparison reports has a right answer. Values pushed in; no device, no clock.
"""

from __future__ import annotations

import math
import random

import pytest

from nfb.hrv.compare import compare, delay, match_beats, rescue

FOREHEAD_AT, REFERENCE_AT = 0.02, 0.13
"""Where in each heartbeat the two sites stamp it: the reference later, by pulse travel and radio."""


def _pulse(d: float) -> float:
    """A pulse with a smooth rise, steepest halfway up, and a slow fall."""
    if 0.0 <= d < 0.12:
        return 0.5 * (1.0 - math.cos(math.pi * d / 0.12))
    if 0.12 <= d < 0.9:
        return math.exp(-(d - 0.12) / 0.25)
    return 0.0


def _wave(beats, at, rate, seconds, height=lambda t: 1.0, noise=0.03, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(int(seconds * rate)):
        t = i / rate
        v = sum(height(b) * _pulse(t - b - at) for b in beats if -0.1 < t - b - at < 0.9)
        out.append((t, v + rng.gauss(0.0, noise)))
    return out


def _heartbeats(seconds=90.0, seed=2, interval=0.92):
    rng = random.Random(seed)
    beats, t = [], 1.0
    while t < seconds - 1.0:
        beats.append(t)
        t += interval + 0.08 * math.sin(2 * math.pi * 0.1 * t) + rng.gauss(0.0, 0.01)
    return beats


def _scene(forehead_noise=0.03, drift_s=0.0, offset_s=0.0, interval=0.92):
    beats = _heartbeats(interval=interval)
    rng = random.Random(5)
    span = beats[-1] - beats[0]
    forehead = [b + FOREHEAD_AT + rng.gauss(0.0, 0.004) for b in beats]
    reference = [b + REFERENCE_AT + offset_s + drift_s * (b - beats[0]) / span + rng.gauss(0.0, 0.004)
                 for b in beats]
    missing = forehead.pop(20)
    false_beat = (forehead[40] + forehead[41]) / 2.0
    forehead = sorted(forehead + [false_beat])
    f_wave = _wave(beats, FOREHEAD_AT, 30.0, 90.0, noise=forehead_noise, seed=1)
    r_wave = _wave(beats, REFERENCE_AT + offset_s, 100.0, 90.0, height=lambda b: 1.0 if b < 45.0 else 0.6, seed=2)
    return beats, forehead, reference, f_wave, r_wave, missing, false_beat


def test_beats_from_the_same_heartbeat_are_paired_and_the_odd_ones_flagged():
    beats, forehead, reference, *_, missing, false_beat = _scene()
    agreement = match_beats(forehead, reference)
    assert len(agreement.pairs) == len(beats) - 1
    assert agreement.forehead_only == [false_beat]
    assert len(agreement.reference_only) == 1, "the heartbeat the forehead missed"
    assert agreement.usual_delay_s == pytest.approx(REFERENCE_AT - FOREHEAD_AT, abs=0.01)


def test_beats_are_paired_when_the_two_clocks_are_half_a_second_apart():
    """Two devices stamped on arrival share no clock. At 56 bpm, stamps 0.47 s apart are further than a
    search for the nearest beat within 0.4 s can reach, and still within half a heartbeat."""
    beats, forehead, reference, *_ = _scene(offset_s=-0.58, interval=1.07)
    agreement = match_beats(forehead, reference)
    assert agreement.usual_delay_s == pytest.approx(-0.47, abs=0.01)
    assert len(agreement.pairs) == len(beats) - 1


def test_beyond_half_a_heartbeat_the_delay_comes_out_one_heartbeat_over():
    """Beat times cannot tell a delay from that delay plus a whole heartbeat. At 65 bpm, 0.47 s is more
    than half a heartbeat, so it is reported as the equivalent delay on the other side."""
    _, forehead, reference, *_ = _scene(offset_s=-0.58)
    assert match_beats(forehead, reference).usual_delay_s == pytest.approx(-0.47 + 0.92, abs=0.03)


def test_the_delay_between_sites_is_measured_on_the_same_point_of_each_pulse():
    _, forehead, reference, f_wave, r_wave, *_ = _scene()
    result = compare(forehead, reference, f_wave, r_wave)
    assert result.delay.median_s == pytest.approx(REFERENCE_AT - FOREHEAD_AT, abs=0.015)
    assert abs(result.delay.change_s) < 0.01


def test_a_delay_that_grows_across_the_recording_shows_as_change():
    _, forehead, reference, *_ = _scene(drift_s=0.03)
    pairs = match_beats(forehead, reference).pairs
    assert delay(pairs).change_s == pytest.approx(0.02, abs=0.006), "two thirds of a 30 ms drift"


def test_a_shrinking_reference_pulse_shows_as_a_falling_contrast():
    _, forehead, reference, f_wave, r_wave, *_ = _scene()
    result = compare(forehead, reference, f_wave, r_wave)
    assert result.contrast.late == pytest.approx(0.6, abs=0.08)


def test_averaging_around_the_reference_beats_recovers_a_buried_forehead_pulse():
    """Noise twice the pulse's height hides every single beat; ninety-odd beats averaged leave the pulse
    standing several times above what remains of the noise, in the shape the noise-free average has."""
    _, _, reference, f_wave, *_ = _scene(forehead_noise=2.0)
    recovered = rescue(f_wave, reference)
    assert recovered.height / recovered.noise > 4.0
    _, _, _, clean, *_ = _scene(forehead_noise=0.0)
    truth = rescue(clean, reference).shape
    mean_t, mean_s = float(truth.mean()), float(recovered.shape.mean())
    cov = sum((a - mean_t) * (b - mean_s) for a, b in zip(truth, recovered.shape))
    spread = math.sqrt(sum((a - mean_t) ** 2 for a in truth) * sum((b - mean_s) ** 2 for b in recovered.shape))
    assert cov / spread > 0.8, "the recovered shape is the forehead pulse, at the right lag"


def test_without_waveforms_only_the_beat_times_are_compared():
    _, forehead, reference, *_ = _scene()
    result = compare(forehead, reference)
    assert result.delay is not None
    assert result.rescue is None and result.contrast is None


def test_a_beat_only_counts_as_one_site_where_both_were_recording():
    """The tool waits for both devices before its clock starts, so one site can have beats from before the
    other began. Those are not artifacts at either site."""
    beats, forehead, reference, *_ = _scene()
    late_start = [f for f in forehead if f > 30.0]
    agreement = match_beats(late_start, reference)
    assert all(r > 29.5 for r in agreement.reference_only), "nothing from before the forehead began"
