"""Session recording: raw device frames on disk, replayable byte for byte.

Records frames as they arrived, not decoded samples. That is the whole point — a recording made
today stays useful after the decoder changes, and a decoder bug found next month can be
re-tested against real data instead of being reproduced live. It also means processing can be
developed with no hardware attached and regression-tested against fixed expected output.

Format is JSON Lines, gzipped when the filename ends in .gz. First line is a header holding
everything needed to decode the rest — crucially the config blob, since channel rates and
scaling cannot be recovered from the frames alone. Every later line is one frame. Text was
chosen over a packed binary format because these files are small (a four-channel 250 sps session
is about 6 kB/s gzipped) and being able to read, diff and truncate one by hand is worth more
than the bytes.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

FORMAT = "nfb-session/1"


@dataclass(frozen=True)
class SessionHeader:
    """Everything needed to decode a recording, plus provenance for it."""

    config: bytes
    device: str = ""
    serial: str = ""
    firmware: str = ""
    started_at: str = ""          # ISO 8601, caller-supplied
    note: str = ""
    test_mode: bool = False

    def to_json(self) -> dict:
        return {
            "format": FORMAT,
            "config": self.config.hex(),
            "device": self.device,
            "serial": self.serial,
            "firmware": self.firmware,
            "started_at": self.started_at,
            "note": self.note,
            "test_mode": self.test_mode,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SessionHeader":
        fmt = d.get("format")
        if fmt != FORMAT:
            raise ValueError(f"unsupported session format {fmt!r}, expected {FORMAT!r}")
        return cls(
            config=bytes.fromhex(d["config"]),
            device=d.get("device", ""),
            serial=d.get("serial", ""),
            firmware=d.get("firmware", ""),
            started_at=d.get("started_at", ""),
            note=d.get("note", ""),
            test_mode=d.get("test_mode", False),
        )


def _open(path: Path, mode: str) -> IO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class SessionWriter:
    """Append frames to a session file. Use as a context manager."""

    def __init__(self, path: str | Path, header: SessionHeader):
        self.path = Path(path)
        self._fh = _open(self.path, "w")
        self._fh.write(json.dumps(header.to_json()) + "\n")
        self.frames = 0

    def write(self, code: int, payload: bytes, elapsed: float | None = None) -> None:
        """Record one frame. `elapsed` is host arrival time in seconds since capture start.

        Arrival time is recorded for diagnostics only. Sample timing comes from the phase
        counter, never from this — host timestamps carry the USB and scheduler jitter that the
        device's own counter does not.
        """
        rec: dict = {"c": code, "p": payload.hex()}
        if elapsed is not None:
            rec["t"] = round(elapsed, 6)
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.frames += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SessionReader:
    """Read a session file: its header, then its frames."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with _open(self.path, "r") as fh:
            first = fh.readline()
        if not first:
            raise ValueError(f"{self.path} is empty")
        self.header = SessionHeader.from_json(json.loads(first))

    def frames(self) -> Iterator[tuple[int, bytes, float | None]]:
        """Yield (code, payload, elapsed) in recorded order.

        Truncated trailing lines are skipped rather than raising: a recording interrupted by a
        crash or a pulled cable is still worth reading up to the cut.
        """
        with _open(self.path, "r") as fh:
            fh.readline()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return
                yield rec["c"], bytes.fromhex(rec["p"]), rec.get("t")
