"""The test heart that answers the paced breath through a resonance. It stands in for a person in the tests of
a pacer that the heart corrects, so its own behaviour is checked here: the gain, the lag and what noise and
drift do.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfb.hrv.sweep import paced_swing
from resonant_heart import ResonantHeart, response

RESONANCE = 5.5
RATES = (4.0, 5.5, 7.0)


def _breaths(heart, rate, share, minutes=6.0, skip=3):
    """The swing and the in-step value of each whole breath that the heart gives at a fixed rate of the pacer,
    after `skip` breaths for the filter to settle. A breath is given as the stamps of its beats are cut."""
    period = 60.0 / rate
    heart.advance(minutes * 60.0, lambda t: t / period, share)
    stamps, rr = np.array(heart.stamps), np.array(heart.rr_ms)
    out = []
    for k in range(skip, int(minutes * 60.0 / period) - 1):
        lo, hi = k * period, (k + 1) * period
        sel = (stamps >= lo) & (stamps < hi)
        out.append(paced_swing(stamps[sel], rr[sel], lo, hi, share))
    return out


@pytest.mark.parametrize("share", [0.3, 0.5])
def test_the_heart_swings_most_and_is_in_step_at_its_own_rate_late_above_and_early_below(share):
    """With no noise, the swing of the beat interval is largest at the heart's own rate, where the heart
    rate is in step with the breath. Above that rate it is late, below it early."""
    swing, late = {}, {}
    for rate in RATES:
        got = _breaths(ResonantHeart(RESONANCE, noise_ms=0.0, seed=1), rate, share)
        assert all(g.swing_ms is not None and g.in_step_s is not None for g in got)
        swing[rate] = float(np.median([g.swing_ms for g in got]))
        late[rate] = float(np.median([g.in_step_s for g in got]))
    assert swing[RESONANCE] > swing[4.0] and swing[RESONANCE] > swing[7.0]
    assert abs(late[RESONANCE]) < 0.15
    assert late[7.0] > 0.5 and late[4.0] < -0.5


def test_the_heart_follows_the_response_that_it_is_given():
    """The measured swing is in proportion to the gain of the band-pass and the measured lag is its lag, at
    three rates, so that a test that plants a resonance finds the resonance it planted."""
    heart_q = 1.5
    ratios, lags = [], []
    for rate in RATES:
        got = _breaths(ResonantHeart(RESONANCE, q=heart_q, noise_ms=0.0, seed=2), rate, 0.5)
        gain, share = response(rate, RESONANCE, heart_q)
        ratios.append(np.median([g.swing_ms for g in got]) / gain)
        lags.append((np.median([g.in_step_s for g in got]), share * 60.0 / rate))
    assert max(ratios) / min(ratios) < 1.15
    for measured, planted in lags:
        assert measured == pytest.approx(planted, abs=0.2)


def test_noise_and_drift_move_the_beats_and_leave_the_response_in_place():
    """With 15 ms of noise and a drift of 4 bpm, the median lag over many breaths stays near the planted one
    where the heart is above its rate, and the same seed gives the same beats."""
    got = _breaths(ResonantHeart(RESONANCE, drift_bpm=4.0, seed=3), 7.0, 0.5, minutes=10.0)
    _, share = response(7.0, RESONANCE, 1.5)
    assert len(got) > 50
    late = [g.in_step_s for g in got if g.in_step_s is not None]
    assert len(late) > len(got) // 2
    assert np.median(late) == pytest.approx(share * 60.0 / 7.0, abs=0.3)
    first, second = (ResonantHeart(RESONANCE, drift_bpm=4.0, seed=3) for _ in range(2))
    for heart in (first, second):
        heart.advance(20.0, lambda t: t / 8.0)
    assert first.rr_ms == second.rr_ms and len(first.rr_ms) > 15
