"""Recording teed off the shared device, for the console.

`nfb record` records by owning the port; the console's `Device` already holds it, so recording there
is a *subscriber*: `Recorder` consumes the raw frame stream and writes the same `nfb-session/1` file
`SessionWriter` has always written. The writer runs behind the subscription's bounded queue on its own
thread, so a disk stall drops recorder frames (counted, visible on replay as a `Gap`) rather than
stalling acquisition — and the stream's `EpochEnd` finalises the file, one file per epoch, because a
config change invalidates the header the frames are decoded with.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .device import Device, Mode
from .edf import EDFWriter
from .notes import NotesWriter, notes_path
from .record import SessionHeader, SessionWriter
from .stream import Block

LATEST = "latest.jsonl.gz"
"""The symlink `make console` opens with no arguments. Repointed after every finished recording, the
same convention the Makefile's `record` target keeps."""

SUFFIXES = {"nfb": ".jsonl.gz", "edf": ".edf"}
"""What each format is called on disk. `nfb` is the native frame recording — replayable, lossless,
the one this program reads back; `edf` is the interchange export, decoded samples for other tools.
See `nfb.edf` for what EDF cannot carry."""


def timestamped_path(directory: str | Path, fmt: str = "nfb", name: str = "") -> Path:
    """Where a recording goes. A stated name keeps its own stem; otherwise the timestamp names it."""
    stem = (name or "").strip() or time.strftime("%Y%m%d-%H%M%S")
    suffix = SUFFIXES.get(fmt, SUFFIXES["nfb"])
    for known in SUFFIXES.values():                  # a typed ".edf" must not become ".edf.edf"
        if stem.endswith(known):
            stem = stem[:-len(known)]
    return Path(directory) / f"{stem}{suffix}"


def unique_path(path: Path) -> Path:
    """`path`, or the first `-2`, `-3` … that does not exist. **A recording is never overwritten**, and
    an operator who typed a name they used yesterday wants tonight's night kept, not an error at 11 pm."""
    if not path.exists():
        return path
    suffix = next((s for s in SUFFIXES.values() if path.name.endswith(s)), path.suffix)
    stem = path.name[:-len(suffix)] if suffix else path.name
    for n in range(2, 1000):
        candidate = path.with_name(f"{stem}-{n}{suffix}")
        if not candidate.exists():
            return candidate
    return path


def point_latest(path: Path) -> None:
    """Repoint the `latest` symlink beside a recording at it. A convenience pointer, so failures are
    swallowed — never take a finished recording down over its shortcut."""
    link = path.parent / LATEST
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.name)
    except OSError:
        pass


class Recorder:
    """One recording: subscribe raw, write until the epoch ends or `stop()` is called.

    Build it, `start()` it, and read `frames`, `dropped`, `seconds` and `bytes_written` for a status
    line. It finalises itself when the device stops or reconfigures — `closed` says so — and `stop()`
    is for the operator disarming mid-session.
    """

    QUEUE_FRAMES = 65536
    """About 35 minutes of four-channel 250 sps data frames (~30/s). A recorder exists to survive disk
    hiccups, so its queue is deep where a display consumer's is shallow — dropping is still possible,
    counted, and visible on replay, but it should take a stall measured in minutes to get there."""

    COMPANION_SECONDS = 60.0
    """How often the companion file is rewritten while recording. It describes a night that may not
    survive to have an end — a crash, a flat battery, a lid closing — so it cannot only be written at
    close, and it is small enough (run-length encoded) that a minute's granularity costs nothing."""

    def __init__(self, device: Device, path: str | Path, note: str = "",
                 poll_seconds: float = 1.0, fmt: str = "nfb", companion=None):
        self.device = device
        self.path = Path(path)
        self.note = note
        self.poll_seconds = poll_seconds
        self.fmt = fmt if fmt in SUFFIXES else "nfb"
        self.companion = companion
        """Called for a `{"session": …, "alarms": …, "quality": …}` snapshot to write beside the
        recording — the signal states and gaps an EDF export cannot carry. See `nfb.notes`."""
        self.error = ""
        self.closed = False
        self._sub = None
        self._writer = None
        self._notes: NotesWriter | None = None
        self._notes_at = 0.0
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None

    @property
    def notes_path(self) -> Path:
        return notes_path(self.path)

    def start(self) -> None:
        """Subscribe and start writing. Raises rather than overwrites: a recording is never destroyed,
        which is the same refusal `nfb record` makes at the command line."""
        if self._thread is not None:
            raise RuntimeError("already started; a Recorder is one file, arm a new one")
        if self.path.exists():
            raise FileExistsError(f"{self.path} exists — a recording is never overwritten")
        self._sub = self.device.subscribe("recorder", queue_size=self.QUEUE_FRAMES)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = self._open_edf() if self.fmt == "edf" else self._open_native()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True, name="recorder")
        self._thread.start()

    def _open_native(self) -> SessionWriter:
        st = self.device.status
        header = SessionHeader(
            config=self.device.config, device=st.model, serial=st.serial, firmware=st.firmware,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            note=self.note, test_mode=st.mode is Mode.TESTING)
        return SessionWriter(self.path, header)

    def _open_edf(self) -> EDFWriter:
        st = self.device.status
        settings = st.settings
        notch = f" N:{settings.mains_hz:g}Hz" if settings.mains_hz else ""
        return EDFWriter(
            self.path, self._sub.channels(),
            recording=f"Startdate X X X {st.model.replace(' ', '_') or 'Optima'}",
            prefilter=f"HP:{settings.lower_limit_hz:g}Hz{notch}")

    def _run(self) -> None:
        try:
            if self.fmt == "edf":
                for item in self._sub.blocks(timeout=self.poll_seconds):
                    if isinstance(item, Block):
                        self._writer.feed(item)
                    self._write_notes()
            else:
                t0: float | None = None
                for code, payload in self._sub.frames(timeout=self.poll_seconds):
                    if t0 is None:
                        t0 = time.monotonic()
                    self._writer.write(code, payload, time.monotonic() - t0)
                    self._write_notes()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._writer.close()
            self._sub.close()
            self.closed = True
            if self.frames:
                self._write_notes(force=True)
                if self.fmt == "nfb":
                    point_latest(self.path)
            elif not self.error:
                # Armed but nothing ever streamed: a header-only file in sessions/ is a recording that
                # never happened, listed forever beside real ones.
                try:
                    self.path.unlink()
                except OSError:
                    pass

    def _write_notes(self, force: bool = False) -> None:
        if self.companion is None:
            return
        now = time.monotonic()
        if not force and now - self._notes_at < self.COMPANION_SECONDS:
            return
        self._notes_at = now
        try:
            snapshot = self.companion()
        except Exception:
            return                  # the session is the recording's guest, not its keeper
        if not snapshot:
            return
        if self._notes is None:
            self._notes = NotesWriter(self.notes_path)
        self._notes.write({"recording": self.path.name, "format_written": self.fmt,
                           **snapshot.get("session", {})},
                          snapshot.get("alarms", ()), snapshot.get("quality"),
                          snapshot.get("marks", ()))

    @property
    def frames(self) -> int:
        """Frames written, or EDF records — either way, "has anything been recorded"."""
        if self._writer is None:
            return 0
        return getattr(self._writer, "frames", None) or getattr(self._writer, "records", 0)

    @property
    def dropped(self) -> int:
        return self._sub.dropped if self._sub is not None else 0

    @property
    def active(self) -> bool:
        return self._thread is not None and not self.closed

    @property
    def seconds(self) -> float:
        """Wall time since arming, for a status line — not signal time, which lives in the frames."""
        return time.monotonic() - self._started_at if self._started_at is not None else 0.0

    @property
    def bytes_written(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def stop(self, timeout: float = 3.0) -> None:
        """Disarm mid-session: close the subscription, wait for the file to finalise. Idempotent."""
        if self._sub is not None:
            self._sub.close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
