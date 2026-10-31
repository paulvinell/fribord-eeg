"""The two runners that read a pulse hand it out in the same form.

The HeartMath's runner and the Mendi's each keep their recent beat times, give the last minute of their
pulse pointing up, and keep a heart-rate state under one name, so a program comparing them needs nothing
device-specific. Driven with synthetic sources; no device, no clock.
"""

from __future__ import annotations

import statistics

import pytest
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


def test_a_waveform_can_be_read_while_the_runner_keeps_appending():
    """The comparison reads on its own thread while each runner's thread appends. Walking the live deque
    item by item failed 199 times in 200 under this load; the waveform is copied in one step instead."""
    import threading

    heart, mendi = _heart(), _mendi()
    targets = [heart.source.decoder.raw, mendi.pulse.waveform]
    stop = threading.Event()

    def append():
        i = 0
        while not stop.is_set():
            for target in targets:
                target.append((1e6 + i / FS, 1.0))
            i += 1

    writer = threading.Thread(target=append)
    writer.start()
    try:
        for _ in range(200):
            assert heart.waveform and mendi.waveform
    finally:
        stop.set()
        writer.join()


def test_a_slow_drift_at_the_ear_does_not_count_as_pulse_height():
    """The headband's pulse has its slow level taken off before anything measures it, and the ear's did
    not, so a level sliding within one beat's window counted as height at the ear only. Both now pass
    through the same one-second centred average."""
    from nfb.hrv.compare import heights

    steady, drifting = _heart(), _heart()
    raw = drifting.source.decoder.raw
    drifted = [(t, v + 4000.0 * t) for t, v in raw]     # a steep slide, far larger than a pulse per second
    raw.clear()
    raw.extend(drifted)
    beats = list(steady.beat_times)[3:-3]
    plain, slid = heights(steady.waveform, beats), heights(drifting.waveform, beats)
    assert all(a is not None and b is not None for a, b in zip(plain, slid))
    assert max(abs(b - a) / a for a, b in zip(plain, slid)) < 0.02


def test_taking_off_the_slow_level_keeps_the_pulse_where_it_was():
    from nfb.hrv.ppg import without_slow_level

    points = [(i / 500, (i % 500 == 250) * 1.0 + 0.001 * i) for i in range(5000)]
    cleaned = without_slow_level(points, 1.0)
    assert cleaned[0][0] == pytest.approx(0.5, abs=0.002) and cleaned[-1][0] == pytest.approx(9.998 - 0.5, abs=0.002)
    peaks = [t for t, v in cleaned if v > 0.5]
    assert peaks == pytest.approx([0.5 + k for k in range(9)], abs=1e-9), "no beat moves"
    assert abs(sum(v for _, v in cleaned) / len(cleaned)) < 0.01, "the slide is gone"
