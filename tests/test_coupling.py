"""Phase-amplitude coupling, and the two balance readouts beside it.

Every signal here is built rather than recorded, because the point of each test is that the measure says
the right thing about a *known* coupling — and a known coupling is something you can only have by
constructing it. The construction matters in one specific way: the slow rhythm is narrowband noise rather
than a sine, so its phase wanders the way a real rhythm's does. With a perfect sine the surrogate floor
cannot work, since shifting a periodic signal in time only rotates its phase and leaves the coupling
exactly where it was. That is a property of the test signal and not of the measure, and it cost an hour
to see, so it is written down here.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp

from nfb.dsp import (
    Coupling, CouplingMetric, DFA_FLOOR, DFA_NOISE, Locking, LockingMetric, PHASE_BINS,
    dfa, functional_ei, modulation_index,
    true_envelope,
)
from nfb.dsp.filters import Band
from nfb.stream import Block, ChannelSpec

RATE = 250.0
SLOW = Band(4.0, 8.0, "theta")
FAST = Band(30.0, 45.0, "low gamma")
SPEC = ChannelSpec(index=0, rate_hz=RATE, scale=1.0, unit="µV")


def narrowband(n, band, rng):
    sos = sp.butter(4, [band.low, band.high], btype="bandpass", fs=RATE, output="sos")
    y = sp.sosfiltfilt(sos, rng.standard_normal(n))
    return y / y.std()


def coupled(seconds: float, depth: float, seed: int = 0, noise: float = 4.0):
    """A slow rhythm whose phase gates a fast one, loudest at the slow trough.

    `depth` 0 is no coupling at all; the fast band is still there and still the same size, so the two
    cases differ in nothing but whether the amplitude follows the phase.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    slow = 30 * narrowband(n, SLOW, rng)
    gate = 1.0 - depth * np.cos(np.angle(sp.hilbert(slow)))
    fast = 5 * narrowband(n, FAST, rng)
    return slow + gate * fast + noise * rng.standard_normal(n)


def feed(x, *, window: float = 20.0, size: int = 250, coupling: Coupling | None = None):
    metric = CouplingMetric({0: SPEC}, coupling or Coupling(SLOW, FAST), window_seconds=window)
    for i in range(0, len(x), size):
        metric.feed(Block(channel=0, start_index=i, samples=np.asarray(x[i:i + size]), spec=SPEC))
    return metric


def test_a_coupled_signal_reads_far_above_its_own_floor_and_an_uncoupled_one_sits_on_it():
    """The floor is the whole claim: a reading only means something as a distance above it."""
    strong = feed(coupled(120, 0.9))
    none = feed(coupled(120, 0.0))

    assert strong.modulation > 10 * strong.floor, (
        f"a strong coupling read {strong.modulation:.4f} against a floor of {strong.floor:.4f}")
    assert none.modulation <= 2 * none.floor, (
        f"an uncoupled signal read {none.modulation:.4f} against a floor of {none.floor:.4f}")
    assert strong.modulation > 20 * none.modulation


def test_the_index_falls_as_the_coupling_weakens():
    readings = [feed(coupled(120, depth)).modulation for depth in (0.9, 0.5, 0.2, 0.0)]
    assert readings == sorted(readings, reverse=True), readings


def test_the_preferred_phase_points_at_the_part_of_the_cycle_the_modulation_was_put_on():
    """A strength with no preferred phase is not a description of anything, so the angle is checked.

    The gate is `1 - depth·cos(phase)`, which is largest at ±180° — the trough of the slow wave, which is
    also where the literature this comes from puts the nesting.
    """
    metric = feed(coupled(120, 0.9))
    assert abs(abs(np.degrees(metric.preferred_phase)) - 180.0) < 20.0, (
        f"read {np.degrees(metric.preferred_phase):.0f}°, expected the trough")

    peak = int(np.argmax(metric.bin_means))
    assert peak in (0, 1, PHASE_BINS - 2, PHASE_BINS - 1), (
        "the fullest phase bin has to agree with the angle, or the picture and the number disagree")


def test_a_square_wave_looks_strongly_coupled_and_the_floor_says_it_is_not():
    """The failure this measure is most often caught by, and the reason the floor is drawn on screen.

    A slow wave that is not a sine has harmonics; the harmonics land in the fast band; they are locked to
    the slow wave by construction. There is no coupling, there is one oscillation that is not sinusoidal —
    and a bare modulation index cannot tell the difference. Shifting the amplitude in time can, because
    the harmonics move with it.
    """
    rng = np.random.default_rng(3)
    n = int(120 * RATE)
    t = np.arange(n) / RATE
    square = 40 * np.sign(np.sin(2 * np.pi * 6 * t)) + 8 * rng.standard_normal(n)
    metric = feed(square)

    assert metric.modulation > 0.02, (
        "the point of this test is that the raw reading is large; if it is not, it proves nothing")
    assert metric.modulation < 1.5 * metric.floor, (
        f"the floor has to absorb it: read {metric.modulation:.4f} against {metric.floor:.4f}")


def test_a_fast_band_too_narrow_to_hold_the_sidebands_is_refused_with_the_width_it_needs():
    """Amplitude modulated at the slow frequency puts energy at `fast ± slow`. A fast band narrower than
    twice the slow centre cuts that off and reports no coupling however much there is — which is the most
    commonly violated rule in this literature, so it is a refusal rather than a note."""
    with pytest.raises(ValueError) as caught:
        Coupling(SLOW, Band(30.0, 38.0, "narrow"))
    assert "12 Hz" in str(caught.value), str(caught.value)

    Coupling(SLOW, Band(30.0, 42.0, "just wide enough"))        # exactly 2 × 6 Hz


def test_a_fast_band_under_the_slow_one_is_refused():
    with pytest.raises(ValueError):
        Coupling(Band(30.0, 45.0), Band(4.0, 20.0))


def test_feeding_in_pieces_gives_the_same_answer_as_feeding_whole():
    """The property that breaks silently if the forgetting window is applied per block rather than per
    sample, and the one every streaming estimator here is required to have."""
    x = coupled(40, 0.6, seed=7)
    small, large = feed(x, size=8), feed(x, size=2000)
    assert small.modulation == pytest.approx(large.modulation, abs=1e-12)
    assert small.floor == pytest.approx(large.floor, rel=1e-9)
    assert small.peak_ratio == pytest.approx(large.peak_ratio, rel=1e-9)


def test_the_rhythm_ratio_says_whether_there_is_a_phase_worth_taking():
    """An instantaneous phase is only interpretable where there is a peak in the spectrum. On flat noise
    the phase is the filter's own ringing, which is perfectly periodic and will couple to anything."""
    rng = np.random.default_rng(11)
    n = int(90 * RATE)

    flat = feed(20 * rng.standard_normal(n))
    assert flat.peak_ratio == pytest.approx(1.0, abs=0.2), (
        "white noise has no rhythm anywhere, so the band and its neighbours have to read alike")

    rhythmic = feed(coupled(90, 0.0))
    assert rhythmic.peak_ratio > 2.0, "a real slow rhythm has to stand above its neighbours"

    alpha = 40 * narrowband(n, Band(8.0, 12.0), rng) + 10 * rng.standard_normal(n)
    assert feed(alpha).peak_ratio < 1.0, (
        "with the neighbour louder than the band, taking the band's phase is taking the neighbour's")


def test_the_reading_is_held_at_zero_until_every_part_of_the_slow_cycle_has_been_seen():
    metric = feed(coupled(120, 0.9), size=8)
    early = CouplingMetric({0: SPEC}, Coupling(SLOW, FAST))
    x = coupled(120, 0.9)[:20]
    early.feed(Block(channel=0, start_index=0, samples=np.asarray(x), spec=SPEC))
    assert early.modulation == 0.0
    assert not early.settled and metric.settled


def test_the_index_itself_is_flat_at_zero_and_one_when_everything_is_in_one_bin():
    assert modulation_index(np.ones(PHASE_BINS)) == 0.0
    one = np.zeros(PHASE_BINS)
    one[4] = 1.0
    assert modulation_index(one) == pytest.approx(1.0)
    assert modulation_index(np.zeros(PHASE_BINS)) == 0.0


def test_the_memory_exponent_has_the_known_answer_on_the_two_signals_that_have_one():
    """The estimator itself, checked where theory gives an exact number: 0.5 for a memoryless signal and
    1.5 for a random walk. Anything about *envelopes* is a claim about the envelope, not about this."""
    rng = np.random.default_rng(5)
    n = int(300 * RATE)
    assert dfa(rng.standard_normal(n), RATE) == pytest.approx(0.5, abs=0.05)
    assert dfa(np.cumsum(rng.standard_normal(n)), RATE) == pytest.approx(1.5, abs=0.05)


def test_a_narrowband_envelope_of_pure_noise_reads_near_the_measured_noise_level_not_near_a_half():
    """The finding that decides what the screen may claim, pinned so it cannot drift.

    Filtering to a band makes the envelope correlated over the filter's own time constant, so an envelope
    with no long-range structure behind it still scores around 0.58 — close enough to the 0.6 inclusion
    floor that on a short run, clearing the floor is not evidence of anything. The screen draws
    `DFA_NOISE` beside the reading for this reason.
    """
    rng = np.random.default_rng(5)
    envelope = true_envelope(rng.standard_normal(int(300 * RATE)), Band(8.0, 12.0), RATE)
    assert dfa(envelope, RATE) == pytest.approx(DFA_NOISE, abs=0.09)


def test_a_long_memory_envelope_scores_above_the_inclusion_floor_and_white_noise_does_not():
    rng = np.random.default_rng(6)
    n = int(300 * RATE)
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    freqs[0] = freqs[1]
    slow = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(freqs), n)
    carrier = narrowband(n, Band(8.0, 12.0), rng)
    signal = np.clip(1 + 0.8 * slow / slow.std(), 0.05, None) * carrier

    envelope = true_envelope(signal, Band(8.0, 12.0), RATE)
    assert dfa(envelope, RATE) > DFA_FLOOR
    assert functional_ei(envelope, RATE) > 0


def test_both_balance_measures_refuse_a_window_too_short_to_answer_from():
    short = np.abs(np.random.default_rng(1).standard_normal(int(3 * RATE)))
    with pytest.raises(ValueError):
        dfa(short, RATE)
    with pytest.raises(ValueError):
        functional_ei(short, RATE, window_seconds=5.0)


def test_a_held_artifact_block_is_filtered_but_not_counted():
    """The defect a whole-screen test found, pinned where the measure lives.

    A rejected block reaches every estimator as a stand-in held at the last accepted value, so the
    filters do not read a blink as an acquisition gap. A held stretch is a flat step, and a flat step in
    the middle of a window statistic is a non-stationarity — the textbook way to manufacture
    cross-frequency coupling out of nothing. Counting them made an uncoupled signal read three times its
    own floor.
    """
    x = coupled(90, 0.0, seed=4)
    n = int(90 * RATE)

    def run(count_held: bool):
        metric = CouplingMetric({0: SPEC}, Coupling(SLOW, FAST))
        held = 0.0
        for i in range(0, n - 249, 250):
            chunk = np.asarray(x[i:i + 250])
            # Every twentieth block stands in at a constant, which is what the gate feeds on a reject.
            rejected = (i // 250) % 20 == 0
            if rejected:
                chunk = np.full(250, held)
            else:
                held = chunk[-1]
            metric.feed(Block(channel=0, start_index=i, samples=chunk, spec=SPEC),
                        count=not rejected or count_held)
        return metric

    counted, excluded = run(True), run(False)
    assert counted.modulation > 2 * excluded.modulation, (
        f"held samples read as coupling: {counted.modulation:.5f} against "
        f"{excluded.modulation:.5f} with them excluded")
    assert excluded.modulation < 2 * excluded.floor, "and excluding them puts it back on the floor"


# --- phase locking ------------------------------------------------------------------------

LOCK_SPECS = {0: SPEC, 1: ChannelSpec(index=1, rate_hz=RATE, scale=1.0, unit="µV")}
ALPHA = Band(8.0, 12.0, "alpha")


def locked_pair(seconds: float, strength: float, seed: int = 0):
    """Two channels sharing a phase relationship in alpha to a controlled degree.

    `strength` 0 is two independent rhythms; 1 is one rhythm appearing in both. In between, the second
    channel mixes the shared component with its own, which is what a partial relationship looks like.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    sos = sp.butter(4, [ALPHA.low, ALPHA.high], btype="bandpass", fs=RATE, output="sos")

    def band():
        y = sp.sosfiltfilt(sos, rng.standard_normal(n))
        return y / y.std()

    a = band()
    shared = sp.sosfiltfilt(sos, np.imag(sp.hilbert(a)))       # a, a quarter cycle along
    shared /= shared.std()
    b = strength * shared + np.sqrt(max(0.0, 1 - strength ** 2)) * band()
    return (20 * a + 3 * rng.standard_normal(n),
            20 * b + 3 * rng.standard_normal(n))


def feed_locking(a, b, window: float = 5.0, size: int = 250):
    metric = LockingMetric(LOCK_SPECS, Locking(ALPHA, (0, 1)), window_seconds=window)
    for i in range(0, len(a) - size + 1, size):
        metric.feed(Block(channel=0, start_index=i, samples=np.asarray(a[i:i + size]), spec=SPEC))
        metric.feed(Block(channel=1, start_index=i, samples=np.asarray(b[i:i + size]),
                          spec=LOCK_SPECS[1]))
    return metric


def test_the_plain_locking_value_is_biased_and_the_bias_is_the_window_not_the_signal():
    """The measurement that overturned a wrong conclusion, pinned so it cannot be forgotten again.

    An earlier reading of 0.73 on unrelated channels was taken as a reason not to train this at all. It
    was measured on a 0.2 Hz band; at ordinary EEG bandwidths the same null is a fraction of that, and it
    falls further the longer you average — which is the signature of a bias, not of a measure that cannot
    see anything.
    """
    nulls = {w: feed_locking(*locked_pair(120, 0.0), window=w).raw for w in (2.0, 5.0, 10.0)}
    assert nulls[2.0] < 0.35, f"nowhere near 0.73 at a 4 Hz band: {nulls[2.0]:.3f}"
    assert nulls[10.0] < nulls[5.0] < nulls[2.0], nulls


def test_the_debiased_locking_reads_nothing_on_unrelated_channels_at_every_window():
    """What the debias buys: zero means zero, whatever the averaging is set to."""
    for window in (2.0, 5.0, 10.0):
        metric = feed_locking(*locked_pair(120, 0.0), window=window)
        assert metric.locked < 0.08, f"{window:g} s window read {metric.locked:.3f}"
        assert abs(metric.consistency) < 0.08


def test_the_debiased_locking_means_the_same_thing_whatever_the_window_is():
    """The property that makes the window safe to expose as a control.

    The raw value for a fixed relationship drifts as the window changes, so an operator who widened it
    would see the number fall and read that as the trainee losing the relationship. The debiased one
    does not move.
    """
    raw = [feed_locking(*locked_pair(120, 0.5), window=w).raw for w in (2.0, 5.0, 10.0)]
    fixed = [feed_locking(*locked_pair(120, 0.5), window=w).locked for w in (2.0, 5.0, 10.0)]
    assert max(raw) - min(raw) > 2 * (max(fixed) - min(fixed)), (raw, fixed)


def test_the_locking_rises_with_the_relationship_and_reaches_one_when_it_is_total():
    readings = [feed_locking(*locked_pair(120, s)).locked for s in (0.0, 0.3, 0.5, 0.8, 1.0)]
    assert readings == sorted(readings), readings
    assert readings[-1] > 0.9, "one rhythm in both channels is total locking"


def test_the_locking_floor_stays_near_zero_however_locked_the_pair_is():
    """Two things about this surrogate are easy to get wrong and both were.

    It shifts one channel's *phase* before the two are combined — shifting the phase difference leaves a
    steady relationship exactly as steady. And the shift clears the band's coherence time rather than a
    few of its periods: at a half-period shift a half-locked pair's floor read 0.40 against a true 0.13.
    """
    for strength in (0.0, 0.5, 1.0):
        metric = feed_locking(*locked_pair(120, strength))
        assert metric.floor < 0.12, (
            f"at strength {strength} the floor read {metric.floor:.3f} against a value of "
            f"{metric.locked:.3f}")


def test_locking_refuses_a_channel_it_does_not_have_and_a_band_above_nyquist():
    with pytest.raises(ValueError):
        Locking(ALPHA, (0, 0))
    with pytest.raises(ValueError):
        LockingMetric(LOCK_SPECS, Locking(ALPHA, (0, 7)))
    with pytest.raises(ValueError):
        LockingMetric(LOCK_SPECS, Locking(Band(100.0, 140.0), (0, 1)))


def test_locking_feeds_in_pieces_the_same_as_whole():
    """Exactly, for the reading. The floor is allowed a start-up difference and only that.

    A surrogate cannot begin until it has a whole lag of phase behind it, and where that moment falls
    depends on the block size — a 2000-sample block covers the lag on its first arrival where 10-sample
    blocks take two hundred of them. What differs is a fraction of a second at the very beginning, which
    the window then forgets; it shows up as a few parts per million and it does not grow.
    """
    a, b = locked_pair(40, 0.6, seed=9)
    small, large = feed_locking(a, b, size=10), feed_locking(a, b, size=2000)
    assert small.raw == pytest.approx(large.raw, rel=1e-9)
    assert small.locked == pytest.approx(large.locked, rel=1e-9)
    assert small.floor == pytest.approx(large.floor, rel=1e-4)


def test_a_relationship_carried_by_something_both_channels_share_reads_at_zero_lag():
    """**The failure the floor cannot see, and the reason `lagged` exists.**

    Under a common reference every channel is a site minus one electrode, so whatever that electrode
    picks up arrives at both at the same instant. The locking it produces is genuinely above chance —
    the two channels really are locked — so a surrogate that breaks the timing correctly reports a large
    real effect, and says nothing about it being wiring rather than physiology.

    Measured, with two *independent* sites and alpha put into the shared reference: 10 µV takes the
    locking to 0.22, 40 µV to 0.78, while the floor stays under 0.06 throughout. What moves is the lag,
    which collapses towards zero, and `lagged` — which discards zero lag — stays far below.
    """
    rng = np.random.default_rng(5)
    n = int(120 * RATE)

    def with_reference(uv: float):
        a = 20 * narrowband(n, ALPHA, rng) + 10 * narrowband(n, Band(1.0, 45.0), rng)
        b = 20 * narrowband(n, ALPHA, rng) + 10 * narrowband(n, Band(1.0, 45.0), rng)
        ref = uv * narrowband(n, ALPHA, rng) if uv else 0.0
        return feed_locking(a - ref, b - ref)

    quiet, hot = with_reference(0.0), with_reference(40.0)
    assert quiet.locked < 0.15, "two unrelated sites and a clean reference"
    assert hot.locked > 0.5, (
        f"a reference carrying alpha makes unrelated sites look locked: {hot.locked:.3f}")
    assert hot.floor < 0.2, "and the floor cannot tell, because the locking is real"

    assert abs(hot.lag_degrees) < 45.0, (
        f"a shared source arrives at both at once: {hot.lag_degrees:.0f}°")
    assert hot.lagged < 0.4 * hot.locked, (
        f"discarding zero lag has to take it apart: {hot.lagged:.3f} of {hot.locked:.3f}")


def test_a_genuine_relationship_keeps_its_lag_and_survives_the_same_test():
    """The other side of it, or the rule above would just be a rule against reading anything."""
    for strength in (0.3, 0.5, 0.8):
        metric = feed_locking(*locked_pair(120, strength))
        assert 60.0 < abs(metric.lag_degrees) < 120.0, (
            f"the built-in relationship is a quarter cycle: {metric.lag_degrees:.0f}°")
        assert metric.lagged > 0.4 * metric.locked, (
            f"at strength {strength}: {metric.lagged:.3f} of {metric.locked:.3f}")


def test_a_sustained_clench_moves_the_locking_and_a_brief_one_does_not():
    """Which is why a locking channel's guard averages over seconds rather than reacting.

    A half-second burst is a small part of a five-second window averaged across a session. A held jaw is
    not: measured, 120 µV sustained took two unrelated sites from 0.03 to 0.13 — and `lagged` stayed at
    zero throughout, because a shared muscle is a shared source like any other.
    """
    rng = np.random.default_rng(7)
    n = int(120 * RATE)

    def with_muscle(uv: float, duty: float):
        a = 15 * narrowband(n, ALPHA, rng) + 10 * narrowband(n, Band(1.0, 45.0), rng)
        b = 15 * narrowband(n, ALPHA, rng) + 10 * narrowband(n, Band(1.0, 45.0), rng)
        if not uv:
            return feed_locking(a, b)
        gate = np.ones(n) if duty >= 1.0 else np.zeros(n)
        if duty < 1.0:
            for start in rng.choice(n - int(RATE), int(120 * duty), replace=False):
                gate[start:start + int(0.5 * RATE)] = 1.0
        emg = gate * uv * narrowband(n, Band(5.0, 110.0), rng)
        return feed_locking(a + emg, b + 0.8 * emg)

    clean = with_muscle(0.0, 0.0).locked
    bursts = with_muscle(120.0, 0.12).locked
    held = with_muscle(120.0, 1.0)

    assert abs(bursts - clean) < 0.03, (
        f"bursts must not move it, or the guard would fail the trainee for nothing: "
        f"{bursts:.3f} against {clean:.3f}")
    assert held.locked > 2 * clean, f"a held clench does: {held.locked:.3f} against {clean:.3f}"
    assert held.lagged < 0.4 * held.locked, "and it is a shared source, so it has no lag"
