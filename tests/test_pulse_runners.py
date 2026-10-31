"""The two runners that read a pulse hand it out in the same form.

The HeartMath's runner and the Mendi's each keep their recent beat times, give the last minute of their
pulse pointing up, and keep a heart-rate state under one name, so a program comparing them needs nothing
device-specific. Driven with synthetic sources; no device, no clock.
"""

from __future__ import annotations

import statistics
from collections import deque

from nfb.hrv.beats import Beat
from nfb.hrv.ppg import PpgDecoder

from nfb_console.heart import HeartRunner
from nfb_console.inputs import HeartChoice, MendiChoice
from nfb_console.mendi import MendiRunner

from test_heg_pulse import _Frames
from test_ppg import FS, heartmath_waveform


class _Cable:
    """A HeartMath cable as the runner sees it: beats, and a decoder holding the raw waveform."""

    error = None

    def __init__(self):
        samples, truth = heartmath_waveform(seconds=20.0)
        self.decoder = PpgDecoder()
        self.decoder.raw.extend((i / FS, s) for i, s in enumerate(samples))
        self._truth = truth

    def beats(self):
        for a, b in zip(self._truth, self._truth[1:]):
            yield Beat(t=b, hr_bpm=60.0 / (b - a), rr_intervals=[b - a], contact=None)

    def close(self):
        pass

    def describe(self):
        return "synthetic cable"


def _heart() -> HeartRunner:
    runner = HeartRunner(HeartChoice(kind="usb"))
    runner.source = _Cable()
    runner._read(runner._generation)
    return runner


def _mendi() -> MendiRunner:
    runner = MendiRunner(MendiChoice(kind="simulator"))
    runner.source = _Frames()
    runner._read(runner._generation)
    return runner


def test_both_runners_keep_their_beat_times_and_a_heart_rate():
    for runner in (_heart(), _mendi()):
        assert isinstance(runner.beat_times, deque) and len(runner.beat_times) > 10
        assert list(runner.beat_times) == sorted(runner.beat_times)
        assert runner.heart_state is not None and 50.0 < runner.heart_state.hr_smoothed < 75.0


def test_both_runners_give_their_pulse_pointing_up():
    """The HeartMath's raw pulse points down; its runner hands it out the other way up, as the Mendi's
    does, so a pulse rises steeply at each beat in both."""
    for runner in (_heart(), _mendi()):
        wave = runner.waveform
        times = [t for t, _ in wave]
        steps = [b - a for (_, a), (_, b) in zip(wave, wave[1:])]
        rises = sorted(steps)[-len(steps) // 20:]          # the steepest twentieth of the steps
        falls = sorted(steps)[:len(steps) // 20]
        assert times == sorted(times)
        assert statistics.fmean(rises) > -statistics.fmean(falls), "the fast edge of each pulse goes up"


def test_a_heart_source_without_a_waveform_says_so():
    runner = HeartRunner(HeartChoice(kind="strap"))
    assert runner.waveform is None, "the Bluetooth heart-rate service sends beats only"


def test_a_fresh_source_starts_without_the_previous_beats():
    for runner in (_heart(), _mendi()):
        runner._reset()
        assert not runner.beat_times
