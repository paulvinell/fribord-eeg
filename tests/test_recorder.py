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


def test_the_latest_pointer_never_deletes_a_real_recording(tmp_path):
    """Finalisation repoints `latest` at the finished capture. A capture named `latest` IS the pointer, so
    repointing it used to unlink the recording that just finished and leave a dangling self-link; and a
    real file already at the reserved name was deleted to make way for the link. Both are data loss."""
    from nfb.recorder import point_latest

    # The capture is itself the pointer: it must survive, not be unlinked to point at itself.
    itself = tmp_path / LATEST
    itself.write_text("a real recording that happens to be called latest")
    point_latest(itself)
    assert itself.exists() and itself.read_text().startswith("a real recording")

    # A real (non-symlink) file squatting at the reserved name is someone's recording: leave it.
    squatter = tmp_path / LATEST
    squatter.write_text("yesterday's capture")
    point_latest(tmp_path / "tonight.jsonl.gz")
    assert squatter.read_text() == "yesterday's capture", "a real file at latest is never deleted"

    # The ordinary case still works: a stale symlink is repointed.
    squatter.unlink()
    (tmp_path / "tonight.jsonl.gz").write_text("tonight")
    point_latest(tmp_path / "tonight.jsonl.gz")
    assert (tmp_path / LATEST).is_symlink()
    assert (tmp_path / LATEST).resolve() == (tmp_path / "tonight.jsonl.gz").resolve()


def test_a_close_failure_marks_the_recorder_errored_not_silently_dead(monkeypatch, tmp_path):
    """A flush failure on a full disk used to raise straight out of the finally, leaving `closed` False —
    so `active` read True for a dead thread with no error, and the sentry's recording alarm stayed blind.
    Now the close is guarded: the fault is recorded, the recorder reads closed-and-errored, and a
    half-flushed capture is never promoted to `latest`."""
    device, _fake, rec = recording(monkeypatch, tmp_path)
    waitfor(lambda: rec.frames >= 5, what="five frames written")

    def boom():
        raise OSError("disk full")

    rec._writer.close = boom
    rec.stop()
    waitfor(lambda: rec.closed, what="recorder finalised despite the close failure")
    assert not rec.active, "a recorder whose thread has died is not active"
    assert rec.error and "disk full" in rec.error, "the fault is recorded for the sentry to see"
    assert not (tmp_path / LATEST).exists(), "a half-flushed capture does not become latest"
    device.close()


def test_a_failed_start_releases_the_subscription(monkeypatch, tmp_path):
    """Registration happens before the directory and writer are made. An open failure used to leave the
    subscription behind, so the device kept fanning frames into a queue no thread would ever drain. The
    subscription is now released on a failed start."""
    device, _fake = connected(monkeypatch)
    device.start_measurement()
    before = device.subscribers

    def boom(*a, **kw):
        raise OSError("read-only file system")

    monkeypatch.setattr("nfb.recorder.SessionWriter", boom)
    rec = Recorder(device, tmp_path / "capture.jsonl.gz")
    with pytest.raises(OSError):
        rec.start()
    assert device.subscribers == before, "a failed arm leaves no orphan subscription"
    device.close()


def test_the_companion_is_bound_to_the_recorded_session_not_the_live_one(monkeypatch, tmp_path):
    """The companion is written at finalize, after the recorder thread joins. On a protocol switch the new
    session is installed before the old recorder is joined, so reading the live `self.session` wrote the
    new session's empty snapshot onto the old recording — deterministic companion loss on every switch.
    The companion is now bound to the session that was recorded, captured when the recorder is armed."""
    import types

    from nfb_console import app as app_mod

    captured = {}

    class FakeRec:
        def __init__(self, device, path, note="", fmt="nfb", companion=None):
            captured["companion"] = companion

        def start(self):
            pass

    monkeypatch.setattr(app_mod, "Recorder", FakeRec)
    recorded = types.SimpleNamespace(companion=lambda: {"session": {"who": "recorded"}})
    console = types.SimpleNamespace(
        device=object(),
        record_settings=types.SimpleNamespace(resolve=lambda: tmp_path / "c.jsonl.gz", fmt="nfb"),
        protocol=types.SimpleNamespace(code="AMP"),
        session=recorded, recorder=None,
        toolbar=types.SimpleNamespace(recording=None))

    app_mod.Console._start_recording(console, recorded)
    console.session = types.SimpleNamespace(companion=lambda: {"session": {"who": "next"}})   # a switch
    assert captured["companion"]()["session"]["who"] == "recorded", \
        "the companion follows the recorded session, not whatever is current at finalize"


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


def test_a_malformed_preferences_file_does_not_take_startup_down():
    """Pref loading handled bad JSON syntax but not valid JSON of the wrong shape: a list where a dict was
    expected crashed every reader. It is now tolerated as if the file were absent."""
    from nfb_console import prefs

    prefs.PATH.write_text("[1, 2, 3]")              # valid JSON, wrong shape
    assert prefs._load() == {}
    assert prefs.record_where() == {}
    assert prefs.record_default("AMP", fallback=True) is True
    prefs.set_record_default("AMP", False)          # must not raise on the list-shaped file
    assert prefs.record_default("AMP") is False


def test_an_unexpected_recording_key_does_not_crash_record_settings():
    """An older or hand-edited file with a stray recording-settings key used to TypeError the dataclass at
    startup. The console keeps only the fields the dataclass declares."""
    import dataclasses

    from nfb_console import prefs, record_dialog

    prefs.set_record_where({"directory": "nights", "fmt": "edf", "legacy": "drop me"})
    known = {f.name for f in dataclasses.fields(record_dialog.RecordSettings)}
    settings = record_dialog.RecordSettings(
        **{k: v for k, v in prefs.record_where().items() if k in known})
    assert settings.directory == "nights" and settings.fmt == "edf"


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
