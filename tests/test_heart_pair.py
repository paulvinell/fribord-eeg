"""The heart-pair tool from a saved recording to its printed report, with no devices.

A recording is built from the synthetic headband frames and pulses the pulse and comparison tests use,
saved and loaded as the tool does, so what is checked here is the tool's own wiring: frames through the
pulse extractor, both sites into the comparison, and a report that says what it found and what it cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import heart_pair                                                       # noqa: E402

from test_heg_pulse import _synthetic                                   # noqa: E402


def _heartmath_raw(onsets, seconds, rate=500.0):
    """The HeartMath's raw waveform for these heartbeats, as the device sends it: in sensor counts, each
    pulse a fast fall for about 0.45 s and then a slow climb back."""
    import math
    import random

    rng = random.Random(4)

    def upright(d):
        if 0.0 <= d < 0.45:
            return 0.5 * (1.0 - math.cos(math.pi * d / 0.45))
        return math.exp(-(d - 0.45) / 0.35) if d >= 0.45 else 0.0

    return [[i / rate, int(3000 - 600 * sum(upright(i / rate - b) for b in onsets if -0.05 < i / rate - b < 2.5)
                            + rng.gauss(0.0, 6.0))] for i in range(int(seconds * rate))]


def _recording(heart: str = "usb") -> dict:
    frames, beats = _synthetic(seconds=90.0)
    reference = [b + 0.10 for b in beats]
    wave = _heartmath_raw([b + 0.10 for b in beats], 90.0) if heart == "usb" else None
    return {"seconds": 90.0, "heart": heart, "frames": [heart_pair._row(f) for f in frames],
            "reference_beats": reference, "reference_wave": wave, "errors": {}}


def test_a_saved_recording_is_compared_and_reported(tmp_path):
    path = tmp_path / "pair.json.gz"
    heart_pair.save(_recording(), path)
    recording = heart_pair.load(path)
    analysis = heart_pair.analyse(recording)
    result = analysis.comparison
    assert len(result.agreement.pairs) > 0.9 * len(recording["reference_beats"])
    assert result.contrast.late == pytest.approx(1.0, abs=0.1), "nothing tightened"
    text = heart_pair.report(recording, analysis)
    for heading in ("Agreement", "Forehead pulse", "Pulse height ratio", "Delay"):
        assert heading in text
    assert "about the same as at the start" in text
    assert "only the change can mean anything" in text


def test_over_bluetooth_the_report_says_what_it_cannot_measure():
    recording = _recording(heart="ble")
    text = heart_pair.report(recording, heart_pair.analyse(recording))
    assert "only the USB cable sends" in text
    assert "the sensor's own processing time" in text


def test_replaying_a_recording_needs_no_devices(tmp_path, capsys):
    path = tmp_path / "pair.json.gz"
    heart_pair.save(_recording(), path)
    assert heart_pair.main(["--replay", str(path)]) == 0
    assert "heartbeats seen at both sites" in capsys.readouterr().out


PAIRED = Path(__file__).parent / "fixtures" / "mendi-heart-pair.json.gz"
"""40 s of the headband and the HeartMath together, kept on the machine it was made on."""


@pytest.mark.skipif(not PAIRED.exists(), reason="needs the local paired recording")
def test_the_recorded_pair_agrees_beat_for_beat():
    """A forehead signal that drifted, and a HeartMath that stamps the foot of its pulse."""
    recording = heart_pair.load(PAIRED)
    result = heart_pair.analyse(recording).comparison
    a = result.agreement
    assert len(a.pairs) >= 30
    assert len(a.forehead_only) <= 5 and len(a.reference_only) <= 5
    assert abs(result.delay.median_s) < 0.25, "steepest rise to steepest rise, the sites are close"


BROKEN = Path(__file__).parent / "fixtures" / "mendi-out-of-range.json.gz"
"""40 s with the headband stuck reading near -900000, kept on the machine it was made on."""


@pytest.mark.skipif(not BROKEN.exists(), reason="needs the local out-of-range recording")
def test_a_recording_out_of_range_says_to_restart_the_headband():
    recording = heart_pair.load(BROKEN)
    analysis = heart_pair.analyse(recording)
    assert analysis.out_of_range_s > 35.0 and not analysis.forehead_beats
    assert "Restart the headband" in heart_pair.report(recording, analysis)


LONG = Path(__file__).parent / "fixtures" / "mendi-heart-pair-3min.json.gz"
"""Three minutes of the headband and the HeartMath together, kept on the machine it was made on."""


def _intervals(analysis):
    pairs = analysis.comparison.agreement.pairs
    f, r = [], []
    for (f1, r1), (f2, r2) in zip(pairs, pairs[1:]):
        if 0.6 < r2 - r1 < 1.6 and 0.6 < f2 - f1 < 1.6:
            f.append(f2 - f1)
            r.append(r2 - r1)
    return f, r


@pytest.mark.parametrize("path", [PAIRED, LONG], ids=["40s", "3min"])
def test_the_two_devices_agree_on_every_interval_and_on_the_variation(path):
    """Timed at half height of its rise, the HeartMath's intervals follow the headband's beat for beat. On
    its old stamp they correlated at 0.69 and 0.33, and its beat-to-beat variation read 104 and 168 ms
    where the headband read 66."""
    if not path.exists():
        pytest.skip("needs the local paired recording")
    import statistics

    analysis = heart_pair.analyse(heart_pair.load(path))
    f, r = _intervals(analysis)
    mf, mr = statistics.fmean(f), statistics.fmean(r)
    cov = sum((a - mf) * (b - mr) for a, b in zip(f, r))
    corr = cov / (sum((a - mf) ** 2 for a in f) * sum((b - mr) ** 2 for b in r)) ** 0.5
    rmssd = lambda xs: (sum((b - a) ** 2 for a, b in zip(xs, xs[1:])) / (len(xs) - 1)) ** 0.5
    assert corr > 0.95
    assert abs(rmssd(f) - rmssd(r)) < 0.15 * rmssd(f)
