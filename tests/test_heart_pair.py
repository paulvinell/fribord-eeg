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
from test_hrv_compare import _wave                                      # noqa: E402


def _recording(heart: str = "usb") -> dict:
    frames, beats = _synthetic(seconds=90.0)
    reference = [b + 0.10 for b in beats]
    wave = [list(p) for p in _wave(beats, 0.10, 100.0, 90.0)] if heart == "usb" else None
    return {"seconds": 90.0, "heart": heart, "frames": [heart_pair._row(f) for f in frames],
            "reference_beats": reference, "reference_wave": wave, "errors": {}}


def test_a_saved_recording_is_compared_and_reported(tmp_path):
    path = tmp_path / "pair.json.gz"
    heart_pair.save(_recording(), path)
    recording = heart_pair.load(path)
    result, forehead = heart_pair.analyse(recording)
    assert len(result.agreement.pairs) > 0.9 * len(recording["reference_beats"])
    assert result.contrast.late == pytest.approx(1.0, abs=0.1), "nothing tightened"
    text = heart_pair.report(recording, result, forehead)
    for heading in ("Agreement", "Forehead pulse", "Pulse height ratio", "Delay"):
        assert heading in text
    assert "about the same as at the start" in text
    assert "Only the change can mean anything" in text


def test_over_bluetooth_the_report_says_what_it_cannot_measure():
    recording = _recording(heart="ble")
    result, forehead = heart_pair.analyse(recording)
    text = heart_pair.report(recording, result, forehead)
    assert "only the USB cable sends" in text
    assert "mostly radio" in text


def test_replaying_a_recording_needs_no_devices(tmp_path, capsys):
    path = tmp_path / "pair.json.gz"
    heart_pair.save(_recording(), path)
    assert heart_pair.main(["--replay", str(path)]) == 0
    assert "heartbeats seen at both sites" in capsys.readouterr().out
