"""Heart rate from the Mendi's optical frames.

Held to two kinds of evidence: synthetic frames with known beat times and the room-light flicker a real
recording showed, which always run, and that real recording of a resting forehead, which is kept on the
machine it was made on and not committed, so its tests run only where it exists. Frames are values pushed
in; no headband, no clock.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import statistics
from pathlib import Path

import pytest

from nfb.heg.pulse import PulseExtractor, _FrameClock
from nfb.heg.source import Channel, Frame

FIXTURE = Path(__file__).parent / "fixtures" / "mendi-pulse.jsonl.gz"
"""40 s from a real headband, put on during the first few seconds, with room light flickering."""

recorded = pytest.mark.skipif(not FIXTURE.exists(), reason="needs the local headband recording")


def _recorded() -> list[Frame]:
    frames = []
    with gzip.open(FIXTURE, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            frames.append(Frame(t=r["t"], l=Channel(r["l_ir"], r["l_red"], r["l_amb"]),
                                r=Channel(r["r_ir"], r["r_red"], r["r_amb"]),
                                p=Channel(r["p_ir"], r["p_red"], r["p_amb"]),
                                acc=tuple(r["acc"]), gyro=tuple(r["gyro"]), temp_c=r["temp"]))
    return frames


def _successive_rms_ms(intervals: list[float]) -> float:
    steps = [b - a for a, b in zip(intervals, intervals[1:])]
    return 1000.0 * math.sqrt(sum(s * s for s in steps) / len(steps))


def _synthetic(seconds: float = 60.0, flicker: bool = True, pulse: bool = True, seed: int = 3,
               drift: float = 0.0):
    """Frames at 31 a second with radio jitter on their arrival, a pulse with a fast rise and a slow
    fall at known beat times, and optionally room light flickering into infrared and ambient alike, with
    occasional frames where the two do not match."""
    rng = random.Random(seed)
    beats, at = [], 0.5
    while at < seconds:
        at += 1.0 + 0.08 * math.sin(2 * math.pi * 0.1 * at)
        beats.append(at)
    frames, fs = [], 31.2
    for i in range(int(seconds * fs)):
        t = i / fs
        blood = 0.0
        if pulse:
            for b in beats:
                d = t - b
                if 0.0 <= d < 0.12:
                    blood += d / 0.12
                elif 0.12 <= d < 0.9:
                    blood += math.exp(-(d - 0.12) / 0.25)
        light = 60000.0 * (1.0 if (t * 1.95) % 1.0 < 0.2 else 0.0) if flicker else 0.0
        spike = 40000.0 if flicker and rng.random() < 0.03 else 0.0
        ir = (300000.0 - 4000.0 * blood + drift * math.sin(2 * math.pi * 0.1 * t) + light + spike
              + rng.gauss(0.0, 150.0))
        amb = 15000.0 + light + rng.gauss(0.0, 150.0)
        arrival = t + abs(rng.gauss(0.0, 0.013))
        frames.append(Frame(t=arrival, l=Channel(ir, ir, amb), r=Channel(ir, ir, amb), p=Channel(ir, ir, amb),
                            acc=(0, 0, 0), gyro=(0, 0, 0), temp_c=33.0))
    return frames, beats


@recorded
def test_the_recorded_forehead_gives_a_resting_heart_rate():
    frames = _recorded()
    extractor = PulseExtractor()
    beats = [b for f in frames for b in extractor.push(f)]
    settled = [b.rr_intervals[0] for b in beats if b.t - frames[0].t > 8.0]
    assert len(settled) >= 25
    rate = 60.0 / statistics.median(settled)
    assert 52.0 < rate < 60.0, f"{rate:.1f} bpm; the recording's spectrum peaks at 58"


@recorded
def test_beats_are_stamped_on_the_edge_so_their_intervals_stay_steady():
    """Stamped on the broad top of the pulse, the recording's beat-to-beat variation read 154 ms;
    stamped on the steepest point of the rise, about 90."""
    frames = _recorded()
    extractor = PulseExtractor()
    beats = [b for f in frames for b in extractor.push(f)]
    settled = [b.rr_intervals[0] for b in beats if b.t - frames[0].t > 8.0]
    assert _successive_rms_ms(settled) < 120.0


def test_room_light_flicker_does_not_become_a_heart_rate():
    frames, truth = _synthetic()
    extractor = PulseExtractor()
    beats = [b for f in frames for b in extractor.push(f)]
    assert abs(len(beats) - (len(truth) - 1)) <= 3, f"{len(beats)} beats for {len(truth) - 1} intervals"
    true_intervals = [b - a for a, b in zip(truth, truth[1:])]
    found = [b.rr_intervals[0] for b in beats[2:]]
    errors = [min(abs(f - t) for t in true_intervals) for f in found]
    assert statistics.median(errors) < 0.015, "intervals follow the heart, not the flicker"


def test_a_slow_drift_under_the_pulse_does_not_hide_beats():
    """The forehead signal drifts over seconds. Without the drift taken out first, the peak detector's
    slower baseline left whole seconds of pulses below it; on this drift, every beat."""
    frames, truth = _synthetic(drift=20000.0)
    extractor = PulseExtractor()
    beats = [b for f in frames for b in extractor.push(f)]
    assert len(beats) >= len(truth) - 4, f"{len(beats)} beats for {len(truth) - 1} intervals"


def _broken(frames, start_s, end_s):
    """The frames with the headband's out-of-range state between two times: every reading far below zero."""
    from dataclasses import replace

    t0 = frames[0].t
    bad = Channel(ir=-600000.0, red=-600000.0, amb=-936000.0)
    return [replace(f, l=bad, r=bad, p=bad) if start_s <= f.t - t0 < end_s else f for f in frames]


def test_out_of_range_readings_give_no_beats_and_say_so():
    """Light readings below zero mean nothing; once the headband sent them until it was restarted."""
    frames, _ = _synthetic(seconds=60.0)
    frames = _broken(frames, 20.0, 40.0)
    extractor = PulseExtractor()
    t0 = frames[0].t
    beats, troubled = [], []
    for f in frames:
        beats += extractor.push(f)
        if extractor.trouble:
            troubled.append(f.t - t0)
    assert troubled and min(troubled) < 21.0 and max(troubled) > 39.0, "the whole broken stretch is flagged"
    assert extractor.trouble is None, "readings back in range clear it"
    assert not [b for b in beats if 21.0 < b.t - t0 < 40.0], "no beat from readings that carry none"
    assert len([b for b in beats if b.t - t0 > 42.0]) >= 10, "beats resume once the readings come back"


def test_a_signal_with_no_pulse_gives_no_beats():
    frames, _ = _synthetic(seconds=30.0, pulse=False, flicker=False)
    extractor = PulseExtractor()
    assert not [b for f in frames for b in extractor.push(f)], "a headband on the table finds no heart"


def test_the_clock_spaces_frames_evenly_through_radio_jitter():
    rng = random.Random(1)
    fs = 31.2
    truth = [i / fs for i in range(2000)]
    clock = _FrameClock(30.0)
    stamps = [clock.stamp(t + abs(rng.gauss(0.0, 0.013))) for t in truth]
    errors = [s - t for s, t in zip(stamps[400:], truth[400:])]
    assert statistics.pstdev(errors) < 0.004, "jitter is averaged out of the stamps"


def test_reset_starts_a_fresh_waveform():
    frames, _ = _synthetic(seconds=10.0)
    extractor = PulseExtractor()
    for f in frames:
        extractor.push(f)
    assert extractor.waveform
    extractor.reset()
    assert not extractor.waveform and extractor.beats_found == 0


# --- in the Mendi runner -------------------------------------------------------------------------

class _Frames:
    """Synthetic frames as a source the runner reads, ending when they do."""

    error = None

    def frames(self):
        yield from _synthetic()[0]

    def close(self):
        pass

    def describe(self):
        return "synthetic headband"


def test_the_runner_keeps_a_heart_rate_beside_the_hemoencephalography_signal():
    from nfb_console.inputs import MendiChoice
    from nfb_console.mendi import MendiRunner

    runner = MendiRunner(MendiChoice(kind="simulator"))
    runner.source = _Frames()
    runner._read(runner._generation)
    assert runner.state is not None, "the HEG signal still runs"
    assert runner.heart_state is not None
    assert 55.0 < runner.heart_state.hr_smoothed < 65.0, f"{runner.heart_state.hr_smoothed:.1f} bpm"


def test_the_runner_drops_its_heart_rate_when_the_headband_reads_out_of_range():
    from nfb_console.inputs import MendiChoice
    from nfb_console.mendi import MendiRunner

    class _EndsBroken(_Frames):
        def frames(self):
            frames, _ = _synthetic()
            yield from _broken(frames, 50.0, 60.0)

    runner = MendiRunner(MendiChoice(kind="simulator"))
    runner.source = _EndsBroken()
    runner._read(runner._generation)
    assert runner.heart_state is None, "the last rate is not held over readings that carry none"


def test_a_fresh_source_starts_without_the_previous_heart_rate():
    from nfb_console.inputs import MendiChoice
    from nfb_console.mendi import MendiRunner

    runner = MendiRunner(MendiChoice(kind="simulator"))
    runner.source = _Frames()
    runner._read(runner._generation)
    assert runner.heart_state is not None and runner.pulse.waveform
    runner._reset()
    assert runner.heart_state is None and not runner.pulse.waveform
