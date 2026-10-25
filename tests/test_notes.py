"""The companion file: the signal states travelling beside a recording that cannot carry them.

This is what makes EDF a real choice rather than a lossy one — the states, the per-second quality and
the config blob live in `<stem>.notes.jsonl` whatever the recording's own format is.
"""

from __future__ import annotations

import math

import pytest

from conftest import waitfor
from test_device import connected

from nfb import Recorder
from nfb.notes import NotesWriter, decode_cells, encode_cells, notes_path, read_notes


def test_a_good_night_costs_almost_nothing_and_a_bad_minute_survives():
    """Run-length encoded, because that is the shape of the data: hours of clean seconds and a few
    minutes that matter. The encoding must not smooth the minutes away to achieve it."""
    cells = ([(0.0, 0.0)] * 3600                       # an hour of clean
             + [(0.9, 0.0)] * 30                       # half a minute of heavy rejection
             + [(float("nan"), float("nan"))] * 5      # then the feed stops for five seconds
             + [(0.0, 1.0)] * 60)                      # and the channel is condemned for a minute
    runs = encode_cells(cells)
    assert len(runs) == 4, "four runs for four states, not 3695 numbers"
    assert runs[0] == [3600, 0.0, 0.0]
    assert runs[2] == [5, None, None], "missing is written as null, never as a clean zero"

    back = decode_cells(runs)
    assert len(back) == len(cells)
    assert back[3600] == (0.9, 0.0)
    assert math.isnan(back[3631][0]), "and comes back as missing, not as data"
    assert back[-1] == (0.0, 1.0)


def test_the_file_names_itself_after_the_recording_whatever_its_suffix(tmp_path):
    assert notes_path(tmp_path / "tuesday.edf").name == "tuesday.notes.jsonl"
    assert notes_path(tmp_path / "tuesday.jsonl.gz").name == "tuesday.notes.jsonl", \
        "the whole suffix chain, not just the last one"


def test_a_rewrite_that_is_killed_leaves_the_previous_file_whole(tmp_path):
    """Written to a temporary and moved into place. A night's companion is rewritten every minute, so
    a process killed during one must not leave half a file where a whole one was."""
    path = tmp_path / "n.notes.jsonl"
    writer = NotesWriter(path)
    writer.write({"protocol": "SLEEP"}, [], {0: [(0.0, 0.0)] * 10})
    first = path.read_text()
    assert not list(tmp_path.glob("*.part")), "no debris left behind"
    writer.write({"protocol": "SLEEP"}, [("2026-08-24T02:00:00+00:00", "ch1 unusable")],
                 {0: [(0.0, 0.0)] * 20})
    assert path.read_text() != first
    assert read_notes(path)["alarms"][0][1] == "ch1 unusable"


def test_an_edf_recording_gets_its_states_beside_it(monkeypatch, tmp_path):
    """The point of the whole file. EDF carries decoded samples and nothing else by construction; the
    states, the gaps, the alarms and the config blob ride alongside."""
    device, _fake = connected(monkeypatch)
    device.start_measurement()
    snapshot = {
        "session": {"protocol": "SLEEP", "config": device.config.hex(), "cell_seconds": 1.0,
                    "corner_hz": 0.5, "gaps": 2,
                    "channels": {"0": {"rate_hz": 250.0, "scale": 0.024, "unit": "uV"}}},
        "alarms": [("2026-08-24T02:13:00+00:00", "ch1 unusable for 61 s")],
        "quality": {0: [(0.0, 0.0)] * 50 + [(0.0, 1.0)] * 10},
    }
    rec = Recorder(device, tmp_path / "night.edf", poll_seconds=0.02, fmt="edf",
                   companion=lambda: snapshot)
    rec.start()
    waitfor(lambda: rec.frames >= 1, what="an EDF record written")
    device.stop()
    waitfor(lambda: rec.closed, what="recorder finalised")

    assert rec.notes_path.exists(), "written at close even though a minute has not passed"
    notes = read_notes(rec.notes_path)
    assert notes["session"]["recording"] == "night.edf"
    assert notes["session"]["format_written"] == "edf"
    assert notes["session"]["config"] == device.config.hex(), \
        "the config blob, which EDF has nowhere to put"
    assert notes["session"]["gaps"] == 2
    assert notes["alarms"] == [("2026-08-24T02:13:00+00:00", "ch1 unusable for 61 s")]
    assert len(notes["quality"][0]) == 60
    assert notes["quality"][0][55] == (0.0, 1.0), "the second-by-second states, recovered"
    device.close()


def test_a_session_hands_over_what_its_recording_cannot_hold():
    """`TrainingSession.companion` is the snapshot the recorder writes; it has to describe the
    channels well enough that an EDF file beside it can be interpreted."""
    from test_screens import replay

    from nfb_console import amplitude_view as av
    from nfb_console.protocols import BY_CODE

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        for n, item in enumerate(session.source.blocks()):
            session.feed_block(item)
            if n > 400:
                break
        snapshot = session.companion()
        head = snapshot["session"]
        assert head["protocol"] == "AMP"
        assert head["config"], "the blob every scale and the corner are decoded from"
        assert head["corner_hz"] in (0.5, 0.01)
        assert set(head["channels"]) == {"0", "1"}
        assert head["channels"]["0"]["rate_hz"] > 0 and head["channels"]["0"]["scale"] != 0
        assert snapshot["quality"][0], "and the per-second states the timeline is drawn from"
        rejected, unusable = snapshot["quality"][0][0]
        assert 0.0 <= unusable <= 1.0 and 0.0 <= rejected <= 1.0
    finally:
        session.stop()


def test_a_marked_moment_travels_beside_the_recording(tmp_path):
    """The companion carries what the protocol judged worth keeping, not only what went wrong — same
    line shape as an alarm, opposite news."""
    path = tmp_path / "s.notes.jsonl"
    writer = NotesWriter(path)
    writer.write({"protocol": "CFC"}, [], {},
                 marks=[("2026-09-18T20:14:00+00:00", "5 Hz + 10 Hz")])
    back = read_notes(path)
    assert back["marks"] == [("2026-09-18T20:14:00+00:00", "5 Hz + 10 Hz")]
    assert back["alarms"] == []


def test_a_companion_written_without_marks_still_reads(tmp_path):
    """Old files, and every screen that marks nothing."""
    path = tmp_path / "s.notes.jsonl"
    NotesWriter(path).write({"protocol": "SLEEP"}, [], {0: [(0.0, 0.0)] * 5})
    assert read_notes(path)["marks"] == []
