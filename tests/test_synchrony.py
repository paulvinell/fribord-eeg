"""Narrowband synchrony, against the closed form.

Two equal-amplitude sinusoids differing in phase by Δφ sum to `2A·cos(Δφ/2)`. That identity is the whole
of the reward, so it is what these check — not that the code runs, but that it reproduces the
arithmetic the protocol is built on.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import counts

from nfb.dsp import Lockin, PairAligner, Reward, SynchronyMetric, bandwidth_tau
from nfb.stream import Block, ChannelSpec

RATE = 250.0
SCALE = 0.001


def spec(index):
    return ChannelSpec(index=index, rate_hz=RATE, unit="uV", scale=SCALE)


SPECS = {0: spec(0), 1: spec(1)}


def sine(freq, seconds, amp=10.0, phase=0.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return amp * np.sin(2 * np.pi * freq * t + phase)


def blocks(x, index, size=8):
    s = SPECS[index] if index in SPECS else spec(index)
    ints = counts(x, SCALE)
    return [Block(index, i, tuple(ints[i:i + size].tolist()), s)
            for i in range(0, len(x) - size + 1, size)]


def interleave(xa, xb):
    """The two channels' blocks, arriving alternately as they do from the device."""
    out = []
    for a, b in zip(blocks(xa, 0), blocks(xb, 1)):
        out += [a, b]
    return out


def final(reward, xa, xb, output_hz=20.0):
    metric = SynchronyMetric(SPECS, (0, 1), reward, output_hz=output_hz)
    readings = []
    for block in interleave(xa, xb):
        readings += metric.feed(block)
    return metric, readings


# --- the lock-in itself ---------------------------------------------------------------------

def test_the_lockin_recovers_peak_amplitude():
    lock = Lockin(10.0, 1.0, RATE)
    z = lock.process(sine(10.0, 20.0, amp=25.0))
    assert abs(abs(z[-1]) - 25.0) < 0.5


def test_the_lockin_recovers_the_phase_difference():
    a, b = Lockin(10.0, 1.0, RATE), Lockin(10.0, 1.0, RATE)
    za = a.process(sine(10.0, 20.0, phase=0.0))
    zb = b.process(sine(10.0, 20.0, phase=np.pi / 3))
    measured = (np.angle(za[-1]) - np.angle(zb[-1])) % (2 * np.pi)
    assert abs(measured - (2 * np.pi - np.pi / 3)) < 0.05 or abs(measured - np.pi / 3) < 0.05


def test_the_lockin_rejects_a_neighbouring_frequency():
    """A 0.2 Hz-wide measurement at 10 Hz must not see 12 Hz. This is the point of narrow bands:
    "it will not be possible if one part of the brain is operating around 8 Hz and the other around
    12" — a 4 Hz-wide band cannot tell synchrony from coincidence."""
    lock = Lockin(10.0, 0.2, RATE)
    z = lock.process(sine(12.0, 30.0, amp=50.0))
    assert abs(z[-1]) < 2.0


def test_bandwidth_sets_the_delay_and_the_relationship_is_stated():
    """A 0.2 Hz band costs about 2 s of delay, and no filter design changes that.

    The second pole is worth its 29%: one pole is 1.59 s and leaks 5% from 2 Hz away.
    """
    assert bandwidth_tau(0.2, poles=1) == pytest.approx(1.0 / (np.pi * 0.2))
    assert Lockin(10.0, 0.2, RATE, poles=1).delay_seconds == pytest.approx(1.59, abs=0.01)
    assert Lockin(10.0, 0.2, RATE).delay_seconds == pytest.approx(2.05, abs=0.02)
    assert Lockin(10.0, 2.0, RATE).delay_seconds < 0.25


# --- the reward -----------------------------------------------------------------------------

@pytest.mark.parametrize("offset,factor", [
    (0.0, 2.0),                     # in phase: adds
    (np.pi / 2, np.sqrt(2)),        # quarter cycle: 2·cos(π/4)
    (2 * np.pi / 3, 1.0),           # 2·cos(π/3)
    (np.pi, 0.0),                   # antiphase: cancels
])
def test_the_sum_reward_follows_two_a_cos_half_delta(offset, factor):
    amp = 12.0
    reward = Reward(10.0, width_hz=1.0)
    metric, readings = final(reward, sine(10.0, 25.0, amp), sine(10.0, 25.0, amp, phase=offset))
    assert readings
    assert metric.value == pytest.approx(amp * factor, abs=0.6)


def test_the_sum_conflates_synchrony_with_amplitude_which_is_why_the_bins_exist():
    """Two channels louder but no better aligned raise the reward. The protocol closes this with
    inhibit bins straddling the reward frequency; the measurement itself cannot."""
    reward = Reward(10.0, width_hz=1.0)
    quiet, _ = final(reward, sine(10.0, 25.0, 10.0, phase=np.pi / 2),
                     sine(10.0, 25.0, 10.0))
    loud, _ = final(reward, sine(10.0, 25.0, 25.0, phase=np.pi / 2),
                    sine(10.0, 25.0, 25.0))
    assert loud.value > quiet.value * 2


def test_the_plv_is_amplitude_invariant_where_the_sum_is_not():
    same = (sine(10.0, 60.0, 10.0), sine(10.0, 60.0, 10.0, phase=0.7))
    louder = (sine(10.0, 60.0, 10.0), sine(10.0, 60.0, 40.0, phase=0.7))
    plv_same, _ = final(Reward(10.0, 1.0, "plv"), *same)
    plv_loud, _ = final(Reward(10.0, 1.0, "plv"), *louder)
    assert plv_same.value == pytest.approx(plv_loud.value, abs=0.02)
    assert plv_same.value > 0.9
    sum_same, _ = final(Reward(10.0, 1.0, "sum"), *same)
    sum_loud, _ = final(Reward(10.0, 1.0, "sum"), *louder)
    assert sum_loud.value > sum_same.value * 1.5


def test_unrelated_channels_do_not_lock():
    """And the residual is not zero, by construction: an exponential mean over `2·tau·bandwidth`
    independent phase estimates has a floor of about `1/sqrt` of that. Here, ~0.25."""
    rng = np.random.default_rng(3)
    n = int(90 * RATE)
    metric, _ = final(Reward(10.0, 1.0, "plv"),
                      rng.normal(0, 20.0, n), rng.normal(0, 20.0, n))
    assert metric.value < 0.5


def test_the_plv_window_is_reported_because_it_is_long():
    metric = SynchronyMetric(SPECS, (0, 1), Reward(10.0, 0.2, "plv"))
    assert metric.smoothing_seconds == pytest.approx(40.0)
    assert "40 s" in metric.describe()
    # the sum measure has no phase window at all, and should not claim one
    assert SynchronyMetric(SPECS, (0, 1), Reward(10.0, 0.2)).smoothing_seconds == 0.0


def test_the_per_channel_parts_are_reported():
    """So the operator can see whether a rising sum is alignment or just more power."""
    metric, _ = final(Reward(10.0, 1.0), sine(10.0, 25.0, 10.0), sine(10.0, 25.0, 30.0))
    assert metric.parts[0] == pytest.approx(10.0, abs=0.6)
    assert metric.parts[1] == pytest.approx(30.0, abs=0.9)


def test_readings_come_out_at_the_requested_rate():
    _, readings = final(Reward(10.0, 1.0), sine(10.0, 10.0, 10.0), sine(10.0, 10.0, 10.0),
                        output_hz=20.0)
    span = readings[-1].t - readings[0].t
    assert len(readings) / span == pytest.approx(20.0, rel=0.1)
    assert readings[0].channel == (0, 1)


def test_a_pair_needs_two_channels_at_one_rate():
    mixed = {0: spec(0), 1: ChannelSpec(index=1, rate_hz=125.0, unit="uV", scale=SCALE)}
    with pytest.raises(ValueError, match="same sample rate"):
        SynchronyMetric(mixed, (0, 1), Reward(10.0))
    with pytest.raises(ValueError, match="not both in this session"):
        SynchronyMetric(SPECS, (0, 2), Reward(10.0))


# --- alignment ------------------------------------------------------------------------------

def test_the_aligner_pairs_only_simultaneous_samples():
    aligner = PairAligner(0, 1)
    a = Block(0, 0, tuple(range(8)), SPECS[0])
    assert aligner.push(a) is None, "nothing to pair with yet"
    b = Block(1, 0, tuple(range(100, 108)), SPECS[1])
    start, va, vb = aligner.push(b)
    assert start == 0
    assert len(va) == len(vb) == 8
    assert vb[0] == pytest.approx(100 * SCALE)


def test_the_aligner_holds_the_leftover_rather_than_pairing_across_time():
    aligner = PairAligner(0, 1)
    aligner.push(Block(0, 0, tuple(range(16)), SPECS[0]))
    start, va, vb = aligner.push(Block(1, 0, tuple(range(8)), SPECS[1]))
    assert len(va) == 8
    start, va, vb = aligner.push(Block(1, 8, tuple(range(8)), SPECS[1]))
    assert start == 8, "the second half of channel 0 was held, not discarded"
    assert len(va) == 8


def test_a_discontinuity_reanchors_instead_of_misaligning():
    """Pairing samples that are not simultaneous would turn a dropped block into a phase error, which
    is invisible in the output and fatal to a synchrony measurement."""
    aligner = PairAligner(0, 1)
    aligner.push(Block(0, 0, tuple(range(8)), SPECS[0]))
    aligner.push(Block(1, 0, tuple(range(8)), SPECS[1]))
    aligner.push(Block(0, 8000, tuple(range(8)), SPECS[0]))       # channel 0 jumped
    assert aligner.push(Block(1, 8, tuple(range(8)), SPECS[1])) is None
    paired = aligner.push(Block(1, 8000, tuple(range(8)), SPECS[1]))
    assert paired is not None and paired[0] == 8000


def test_a_band_that_reaches_dc_is_refused():
    """He trains "as low as 0.01 - 0.1 Hz for some cases", so this case arises rather than being
    theoretical: 0.2 Hz wide at 0.1 Hz spans 0.0-0.2, and a band whose lower edge is DC is not a band."""
    with pytest.raises(ValueError, match="reaches DC"):
        Reward(0.1, 0.2)
    with pytest.raises(ValueError, match="positive"):
        Reward(0.0)
    ok = Reward(0.1, 0.08)
    assert ok.low > 0


def test_the_sum_stops_measuring_phase_when_the_channels_are_unbalanced():
    """A property of the measure, not of this implementation, and the one thing about it an operator can
    act on: at 4:1 the reward is mostly reporting the louder channel and the fix is an electrode.

    In phase it reads `A_a + A_b` and antiphase `|A_a - A_b|`, so phase controls
    `2·min / (A_a + A_b)` of the reading. Checked against a numeric sweep rather than asserted.
    """
    for ratio, expected in ((1.0, 1.0), (2.0, 2 / 3), (4.0, 0.4), (8.0, 2 / 9)):
        metric, _ = final(Reward(10.0, 0.5), sine(10.0, 40.0, 12.0),
                          sine(10.0, 40.0, 12.0 * ratio))
        assert metric.sensitivity == pytest.approx(expected, abs=0.01), ratio
        assert metric.balance == pytest.approx(ratio, rel=0.02)

        # and the closed form has to match what the measurement actually swings
        in_phase, _ = final(Reward(10.0, 0.5), sine(10.0, 40.0, 12.0),
                            sine(10.0, 40.0, 12.0 * ratio))
        anti, _ = final(Reward(10.0, 0.5), sine(10.0, 40.0, 12.0),
                        sine(10.0, 40.0, 12.0 * ratio, phase=np.pi))
        swing = (in_phase.value - anti.value) / in_phase.value
        assert swing == pytest.approx(expected, abs=0.02)


def test_filtering_each_channel_and_summing_is_summing_then_filtering():
    """The reward is the instantaneous voltages in each channel added together and then filtered. We
    lock in each channel and add the phasors, which is the same thing because a lock-in is linear — and
    that is worth *checking* rather than reasoning about, since it is the claim the whole reward rests on."""
    for offset in (0.0, np.pi / 3, np.pi / 2, np.pi):
        a = sine(10.0, 30.0, 12.0)
        b = sine(10.0, 30.0, 12.0, phase=offset)
        summed = Lockin(10.0, 0.5, RATE).process(a + b)
        ours = Lockin(10.0, 0.5, RATE).process(a) + Lockin(10.0, 0.5, RATE).process(b)
        assert np.allclose(summed, ours, atol=1e-9), offset


def test_the_two_channels_oscillators_stay_in_lockstep():
    """Each channel has its own lock-in, so their local oscillators must not drift apart — a relative
    phase error between them would appear as a synchrony change that is not in the signal."""
    a, b = Lockin(10.0, 0.5, RATE), Lockin(10.0, 0.5, RATE)
    x = sine(10.0, 8.0, 10.0)
    for i in range(0, len(x) - 7, 8):
        a.process(x[i:i + 8])
        b.process(x[i:i + 8])
    assert a._phase == b._phase
