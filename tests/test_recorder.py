"""The console-side recorder: raw frames teed off the shared device, one file per epoch.

Against the same fake amplifier the device tests use, because the rules under test are the device's:
the recorder is a subscriber like any other, the epoch ending is what finalises its file, and the file
it leaves behind is a plain `nfb-session/1` recording that the replay path already reads.
"""

from __future__ import annotations

import pytest

import optima

from conftest import waitfor
from test_device import connected

from nfb import Recorder, SessionReader
from nfb.recorder import LATEST


def recording(monkeypatch, tmp_path, **kw):
    device, fake = connected(monkeypatch)
    device.start_measurement()
    rec = Recorder(device, tmp_path / "capture.jsonl.gz", poll_seconds=0.02, **kw)
    rec.start()
    return device, fake, rec


def test_frames_are_recorded_as_they_arrived_and_the_epoch_closes_the_file(monkeypatch, tmp_path):
    device, _fake, rec = recording(monkeypatch, tmp_path, note="SLEEP")
    waitfor(lambda: rec.frames >= 10, what="ten frames written")
    assert rec.active and not rec.closed
    device.stop()                       # the epoch ends; the recorder finalises itself
    waitfor(lambda: rec.closed, what="recorder finalised on EpochEnd")

    reader = SessionReader(rec.path)
    assert reader.header.note == "SLEEP"
    assert reader.header.serial == "00000000"
    assert reader.header.config == device.config
    frames = list(reader.frames())
    assert len(frames) >= 10
    assert {code for code, _p, _t in frames} == {optima.NOT_DATA}
    stamps = [t for _c, _p, t in frames]
    assert all(b >= a for a, b in zip(stamps, stamps[1:])), "elapsed never runs backwards"
    assert (tmp_path / LATEST).resolve() == rec.path.resolve(), "latest points at the new capture"
    device.close()


def test_disarming_mid_session_finalises_the_file_and_leaves_the_stream_alone(monkeypatch, tmp_path):
    device, _fake, rec = recording(monkeypatch, tmp_path)
    waitfor(lambda: rec.frames >= 5, what="five frames written")
    rec.stop()
    assert rec.closed and not rec.active
    assert device.status.mode.value == "measuring", "the session's stream outlives its recording"
    assert len(list(SessionReader(rec.path).frames())) >= 5
    device.close()


def test_a_recording_is_never_overwritten_and_never_left_empty(monkeypatch, tmp_path):
    device, _fake = connected(monkeypatch)
    taken = tmp_path / "capture.jsonl.gz"
    taken.write_text("someone's head")
    with pytest.raises(FileExistsError):
        Recorder(device, taken).start()
    assert taken.read_text() == "someone's head"

    # Armed while idle: nothing ever streams, the iterator returns, and the header-only file is removed
    # rather than listed forever beside real recordings.
    rec = Recorder(device, tmp_path / "empty.jsonl.gz", poll_seconds=0.02)
    rec.start()
    waitfor(lambda: rec.closed, what="idle recorder gave up")
    assert not rec.path.exists()
    assert not rec.error
    device.close()


def test_edf_is_an_export_and_the_native_format_stays_the_replayable_one(monkeypatch, tmp_path):
    from nfb.edf import read_header
    from nfb.recorder import LATEST

    device, _fake, rec = recording(monkeypatch, tmp_path, fmt="edf")
    waitfor(lambda: rec.frames >= 2, what="two EDF records written")
    device.stop()
    waitfor(lambda: rec.closed, what="recorder finalised")

    head = read_header(rec.path)
    assert head["signals"] == 4 and head["records"] >= 2
    assert head["prefilter"][0].startswith("HP:"), "the device's own corner travels with the export"
    assert not (tmp_path / LATEST).exists(), \
        "latest points at replayable recordings only — an export is not one"
    device.close()


def test_a_name_is_kept_a_format_names_the_suffix_and_a_clash_is_never_overwritten(tmp_path):
    from nfb.recorder import timestamped_path, unique_path

    assert timestamped_path(tmp_path, "nfb").name.endswith(".jsonl.gz")
    assert timestamped_path(tmp_path, "edf").name.endswith(".edf")
    assert timestamped_path(tmp_path, "edf", "tuesday").name == "tuesday.edf"
    assert timestamped_path(tmp_path, "edf", "tuesday.edf").name == "tuesday.edf", \
        "a typed suffix is not doubled"
    assert timestamped_path(tmp_path, "nfb", "  ").name.endswith(".jsonl.gz"), "blank means timestamp"

    taken = tmp_path / "tuesday.edf"
    taken.write_text("last week")
    assert unique_path(taken).name == "tuesday-2.edf"
    (tmp_path / "tuesday-2.edf").write_text("also taken")
    assert unique_path(taken).name == "tuesday-3.edf"
    assert taken.read_text() == "last week", "nothing was touched to find that out"


def test_the_record_default_is_per_protocol_and_the_protocol_speaks_until_the_operator_does():
    from nfb_console import prefs

    assert prefs.record_default("AMP") is False
    assert prefs.record_default("SLEEP", fallback=True) is True, "the protocol's own default holds"
    prefs.set_record_default("SLEEP", False)
    assert prefs.record_default("SLEEP", fallback=True) is False, "until the operator touches it"
    prefs.set_record_default("AMP", True)
    assert prefs.record_default("AMP") is True
    assert prefs.record_default("A-T") is False, "one protocol's choice stays its own"


def test_the_record_options_resolve_to_the_file_you_will_actually_get(tmp_path):
    """The dialog's whole job: three fields whose only real question is "what file am I about to
    get", answered with the resolved path — clash suffix included, since a reused name must keep
    tonight rather than refuse it."""
    from nfb_console import prefs
    from nfb_console.record_dialog import RecordSettings

    s = RecordSettings(directory=str(tmp_path), name="", fmt="nfb")
    assert s.resolve().name.endswith(".jsonl.gz")
    s.fmt = "edf"
    assert s.resolve().name.endswith(".edf"), "the format names the suffix"
    s.name = "tuesday"
    assert s.resolve().name == "tuesday.edf"
    s.resolve().write_text("last week")
    assert s.resolve().name == "tuesday-2.edf", "a name reused does not overwrite or refuse"

    prefs.set_record_where(vars(s))
    assert RecordSettings(**prefs.record_where()).resolve().name == "tuesday-2.edf", \
        "and where recordings go outlives the launch"


def test_the_toolbar_paints_the_recorder_it_is_shown(tmp_path):
    from types import SimpleNamespace

    from test_screens import StubDevice

    from nfb_console import chrome
    from nfb_console import theme as t

    bar = chrome.Toolbar(StubDevice(), lambda: None, lambda: None)
    bar.set_transport_visible(True)
    bar.set_record_supported(True)
    path = tmp_path / "20260822-010203.jsonl.gz"

    bar.recording = SimpleNamespace(active=True, closed=False, frames=90, dropped=0,
                                    seconds=61.0, bytes_written=2_500_000, path=path)
    bar.refresh(running=True)
    assert bar.record_glyph.color == t.BAD
    assert "rec 01:01" in bar.rec_note.value and "2.5 MB" in bar.rec_note.value

    bar.recording = SimpleNamespace(active=False, closed=True, frames=90, dropped=3,
                                    seconds=61.0, bytes_written=2_500_000, path=path)
    bar.refresh(running=False)
    assert bar.record_glyph.color != t.BAD
    assert bar.rec_note.value == f"saved {path.name}"

    bar.recording = None
    bar.set_record_supported(False)
    bar.refresh(running=False)
    assert bar.rec_note.value == ""
    assert bar.record.disabled and bar.record_split.opacity < 1.0, \
        "both halves of the split button grey together"


def test_a_screen_with_no_sentry_can_still_be_opened(monkeypatch):
    """Arming a recorder must not assume the session is an amplifier session.

    The HEG and HRV screens drive their own Bluetooth devices and are not `TrainingSession` at all, so
    they have no signal sentry to hand a recorder to. Picking one raised `AttributeError` on the way in
    and the launcher card did nothing.
    """
    from nfb_console.app import Console, build_parser
    from nfb_console.protocols import BY_CODE

    class StubPage:
        width = 1240

        def __init__(self):
            self.controls, self.services, self.tasks, self.dialogs = [], [], [], []
            self.window = type("W", (), {})()

        def show_dialog(self, d): self.dialogs.append(d)
        def pop_dialog(self): self.dialogs.pop()
        def add(self, c): self.controls.append(c)
        def update(self, *a): pass
        def run_task(self, fn, *a): self.tasks.append(fn)

    def refuse(*a, **k):
        raise FileNotFoundError("no FTDI device found")

    monkeypatch.setattr(optima, "Optima", refuse)
    console = Console(StubPage(), build_parser().parse_args(["--live"]))
    for code in ("HEG", "HRV"):
        console.pick(BY_CODE[code])              # must not raise
        assert console.session is not None
        assert getattr(console.session, "sentry", None) is None, \
            "the point of the test: this screen has none"
    console.show_launcher()
