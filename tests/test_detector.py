"""A wavelet watching a signal live: the level as a multiple of the usual size, and the veto."""

import numpy as np
import pytest

from nfb.dsp import Planted, Wavelet, pink, plant
from nfb.dsp.detector import LiveDetector, RunningMedian

RATE = 250.0


def run(detector, x, block=8):
    for i in range(0, len(x), block):
        detector.process(x[i:i + block])


def test_the_usual_size_is_the_median_of_the_last_minute_from_the_first_sample():
    usual = RunningMedian(RATE, seconds=60.0, step_seconds=0.02, every_seconds=0.02)
    assert usual.value is None
    assert usual.update(np.array([3.0])) == 3.0
    x = np.concatenate([np.full(int(90 * RATE), 1.0), np.full(int(40 * RATE), 5.0)])
    for i in range(0, len(x), 7):
        usual.update(x[i:i + 7])
    # 60 s back from the end holds 20 s of ones and 40 s of fives.
    assert usual.value == 5.0
    assert usual.seconds == pytest.approx(60.0)


def test_the_level_is_the_same_bursts_at_any_signal_size():
    """A detector has to work at any site: a signal ten times larger finds the same bursts."""
    rng = np.random.default_rng(2)
    x = pink(int(180 * RATE), RATE, rng)
    plant(x, RATE, rng, Planted(10.0, (0.5, 1.5), 12.0))
    small, large = LiveDetector(Wavelet(10.0, 5), RATE), LiveDetector(Wavelet(10.0, 5), RATE)
    run(small, x)
    run(large, 10.0 * x)
    assert len(small.found()) >= 5 and small.found() == large.found()
    assert large.usual.value == pytest.approx(10.0 * small.usual.value)


def test_a_veto_lasts_past_the_clench_it_read():
    """The wavelet's output outlasts a clench by the wavelet's length. A veto that ended with the clench
    would let that tail through as a burst."""
    rng = np.random.default_rng(3)
    x = pink(int(40 * RATE), RATE, rng)
    clench = slice(int(20 * RATE), int(21 * RATE))
    x[clench] += 60.0 * rng.standard_normal(int(RATE))
    veto = np.zeros(len(x), dtype=bool)
    veto[clench] = True
    held, plain = LiveDetector(Wavelet(40.0, 6), RATE), LiveDetector(Wavelet(40.0, 6), RATE)
    for i in range(0, len(x), 50):
        held.process(x[i:i + 50], veto[i:i + 50])
        plain.process(x[i:i + 50])
    near = lambda found: [b for b in found if clench.start <= b[1] < clench.stop + int(RATE)]
    assert near(plain.found()) and not near(held.found())
