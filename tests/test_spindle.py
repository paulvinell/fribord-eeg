"""The spindle detector and the search for a person's spindle frequency, on spindles planted at known times."""

import numpy as np
import pytest

from nfb.dsp import Planted, SpindleDetector, SpindleSearch, pink, plant, score_found

RATE = 1000.0


def planted(seconds, frequency, seed, rate=RATE):
    rng = np.random.default_rng(seed)
    x = pink(int(seconds * rate), rate, rng)
    truth = plant(x, rate, rng, Planted(frequency, (0.25, 0.4), 14.0, per_minute=8, jitter_hz=2.0))
    return x, truth


def blocks(x, n=100):
    return (x[i:i + n] for i in range(0, len(x), n))


@pytest.mark.parametrize("rate", [250.0, 1000.0])
def test_the_detector_finds_the_spindles_and_little_else(rate):
    x, truth = planted(240, 25.0, 1, rate)
    detector = SpindleDetector(rate)
    confirmed = sum(detector.process(b) for b in blocks(x))
    result = score_found(detector.found(), truth, len(x), rate)
    assert result.found > 0.8 and result.false_per_minute < 2.0
    assert result.delay_seconds < 0.25
    assert confirmed == detector.count == len(detector.found())


def test_a_clench_reads_as_spindles_unless_the_muscle_is_rejected():
    rng = np.random.default_rng(2)
    x = pink(int(120 * RATE), RATE, rng)
    clenches = [slice(int(t * RATE), int((t + 2) * RATE)) for t in (70, 80, 90, 100)]
    for at in clenches:
        x[at] += 40.0 * rng.standard_normal(at.stop - at.start)
    plain, rejecting = SpindleDetector(RATE), SpindleDetector(RATE, reject_muscle=True)
    for b in blocks(x):
        plain.process(b)
        rejecting.process(b)
    late = lambda d: [s for s in d.found() if s[1] >= 65 * RATE]
    assert len(late(plain)) >= 8 and not late(rejecting)
    assert rejecting.rejected >= 4


def test_the_search_answers_from_the_first_block_and_settles_on_the_frequency():
    x, _ = planted(150, 22.0, 3)
    search = SpindleSearch(RATE, start_hz=30.0)
    assert search.estimate().frequency_hz == 30.0
    seen = {}
    for k, b in enumerate(blocks(x, 250)):
        search.push(b)
        if k in (0, 39):
            seen[k] = search.estimate()
    found = search.estimate()
    assert 13.0 <= seen[0].frequency_hz <= 35.0 and seen[0].seconds == pytest.approx(0.25, abs=0.01)
    assert seen[39].spindles < 8 and seen[39].frequency_hz == pytest.approx(22.0, abs=2.5)
    assert found.frequency_hz == pytest.approx(22.0, abs=0.7) and found.spindles >= 8


def test_the_search_keeps_only_its_last_minutes():
    search = SpindleSearch(RATE, seconds=20.0)
    for b in blocks(planted(60, 25.0, 4)[0], 1000):
        search.push(b)
    assert search.seconds == pytest.approx(20.0)


def test_a_higher_level_counts_fewer_spindles_and_keeps_what_was_measured():
    x, _ = planted(240, 25.0, 5)
    low, changed = SpindleDetector(RATE, level=3.0), SpindleDetector(RATE, level=3.0)
    half = len(x) // 2
    for b in blocks(x[:half]):
        low.process(b)
        changed.process(b)
    usual = changed.usual
    changed.set_level(8.0)
    assert changed.level == 8.0 and changed.usual == usual
    assert changed.level_uv == pytest.approx(8.0 * usual)
    before = changed.count
    for b in blocks(x[half:]):
        low.process(b)
        changed.process(b)
    assert before == pytest.approx(low.count / 2, rel=0.5)
    assert changed.count - before < (low.count - before) / 2
