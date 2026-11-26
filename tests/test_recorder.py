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


def test_a_recording_at_a_wide_range_replays_with_the_same_microvolts(monkeypatch, tmp_path):
    """The header's config carries the range, so a replay decodes the counts to the µV the live
    session read, outside the narrow range as well."""
    import struct

    from nfb import Device, ReplaySource, Settings
    from nfb.stream import Block
    from test_device import FakeOptima

    class Loud(FakeOptima):
        def read_message(self):
            msg = super().read_message()
            counts = struct.pack("<32h", *range(-16000, 16000, 1000))
            return optima.Message(msg.code, msg.payload[:4] + counts)

    fake = Loud()
    monkeypatch.setattr(optima, "Optima", lambda *a, **k: fake)
    device = Device(Settings(range_uv=6000.0))
    assert device.connect()
    live = device.subscribe("live")
    rec = Recorder(device, tmp_path / "wide.jsonl.gz", poll_seconds=0.02)
    rec.start()
    device.start_measurement()

    def first(blocks, n=16):
        out = []
        for item in blocks:
            if isinstance(item, Block):
                out.append((item.channel, item.start_index, tuple(item.values), item.spec.input_range))
            if len(out) >= n:
                return out
        return out

    seen = first(live.blocks(timeout=1.0))
    waitfor(lambda: rec.frames >= 8, what="eight frames written")
    device.stop()
    waitfor(lambda: rec.closed, what="recorder finalised")
    assert max(abs(v) for _c, _i, values, _r in seen for v in values) > 800.0
    assert first(ReplaySource(rec.path).blocks()) == seen
    device.close()
