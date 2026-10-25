"""EDF output, so a recording can be read by something other than this program.

The native format records raw frames and is the one to keep: it replays byte for byte, survives
decoder changes, and carries the config blob, the per-channel signal states and the phase counter that
makes a gap visible. EDF carries none of that — it is decoded samples on a fixed grid — so it is an
*export*, offered because it is what every other tool reads (EDFbrowser, MNE, Polyman, sleep scorers).

Two details are where an EDF writer usually goes wrong, and both are handled by writing from the
blocks' **absolute sample indices** rather than from arrival order: channels at different rates stay
aligned to the same record grid, and a dropped frame becomes a zero-filled hole at the right instant
instead of shifting everything after it earlier in time.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

from .stream import Block, ChannelSpec

DIGITAL_MIN = -32768
DIGITAL_MAX = 32767
"""The device's own 16-bit counts are written unchanged; EDF's physical scaling carries the µV. So an
EDF export is lossless in amplitude — no requantisation — which a float export would not be."""

RECORD_CHOICES = (1, 2, 4, 8)
"""Record durations to try, in seconds. EDF stores samples-per-record as an integer, so the duration
has to make every channel's rate integral: 62.5 sps (the device's deepest decimation) needs 2 s."""


def record_seconds(rates) -> int:
    """The shortest record duration that gives every rate a whole number of samples."""
    for duration in RECORD_CHOICES:
        if all(abs(rate * duration - round(rate * duration)) < 1e-9 for rate in rates):
            return duration
    return RECORD_CHOICES[-1]


def _text(value: str, width: int) -> bytes:
    """EDF is a fixed-width ASCII header: every field left-justified and space-padded, no exceptions."""
    return value.encode("ascii", "replace")[:width].ljust(width)[:width]


def _number(value: float, width: int) -> bytes:
    """A number in `width` characters, as much precision as fits — EDF has no other way to say it."""
    for form in (f"{value:.8f}", f"{value:.6f}", f"{value:.4f}", f"{value:.2f}", f"{value:.0f}",
                 f"{value:g}"):
        if len(form) <= width:
            return _text(form, width)
    return _text(f"{value:.0f}"[:width], width)


class EDFWriter:
    """Write decoded blocks as EDF. Use as a context manager, or call `close()`.

    `specs` fixes the signal set for the file's lifetime — EDF's header is written once and the signal
    count is in it — which is the same rule the epoch already enforces on a recording.
    """

    def __init__(self, path: str | Path, specs: dict[int, ChannelSpec], patient: str = "",
                 recording: str = "", prefilter: str = "", started_at: float | None = None):
        self.path = Path(path)
        self.specs = dict(specs)
        self.channels = sorted(self.specs)
        rates = [self.specs[c].rate_hz for c in self.channels]
        self.record_seconds = record_seconds(rates)
        self.per_record = {c: int(round(self.specs[c].rate_hz * self.record_seconds))
                           for c in self.channels}
        self.records = 0
        self._buf: dict[int, list[int]] = {c: [] for c in self.channels}
        self._base: dict[int, int] = {c: 0 for c in self.channels}
        self._seen: dict[int, int] = {c: 0 for c in self.channels}
        self._fh = open(self.path, "wb")
        self._write_header(patient, recording, prefilter,
                           time.time() if started_at is None else started_at)

    # --- header ---------------------------------------------------------------------------

    def _write_header(self, patient: str, recording: str, prefilter: str, when: float) -> None:
        ns = len(self.channels)
        stamp = time.localtime(when)
        head = b"".join((
            _text("0", 8),
            _text(patient or "X X X X", 80),
            _text(recording or "Startdate X X X X", 80),
            _text(time.strftime("%d.%m.%y", stamp), 8),
            _text(time.strftime("%H.%M.%S", stamp), 8),
            _text(str(256 * (ns + 1)), 8),
            _text("", 44),
            # -1 while writing: EDF's way of saying "still going", patched at close so a file left by
            # a crash is still readable up to its last whole record.
            _text("-1", 8),
            _text(str(self.record_seconds), 8),
            _text(str(ns), 4),
        ))
        scales = [abs(self.specs[c].scale) for c in self.channels]
        fields = [
            [_text(f"EEG ch{c}", 16) for c in self.channels],
            [_text("AgAgCl electrode", 80) for _c in self.channels],
            [_text(self.specs[c].unit or "uV", 8) for c in self.channels],
            [_number(DIGITAL_MIN * s, 8) for s in scales],
            [_number(DIGITAL_MAX * s, 8) for s in scales],
            [_text(str(DIGITAL_MIN), 8) for _c in self.channels],
            [_text(str(DIGITAL_MAX), 8) for _c in self.channels],
            [_text(prefilter, 80) for _c in self.channels],
            [_text(str(self.per_record[c]), 8) for c in self.channels],
            [_text("", 32) for _c in self.channels],
        ]
        self._fh.write(head + b"".join(b"".join(group) for group in fields))

    # --- data -----------------------------------------------------------------------------

    def feed(self, block: Block) -> None:
        """One decoded block. Anything that is not a `Block` — a `Gap` — is ignored: the hole it
        reports is already implied by the next block's absolute index."""
        channel = block.channel
        if channel not in self._buf:
            return
        buf = self._buf[channel]
        start = block.start_index
        end = self._base[channel] + len(buf)
        if start < end:
            # Overlap: keep what is already placed and take only the tail. Re-writing a sample would
            # move every later one, which is the failure this whole index-based placement avoids.
            samples = block.samples[end - start:]
        else:
            buf.extend([0] * (start - end))          # a hole, at the instant it happened
            samples = block.samples
        buf.extend(samples)
        self._seen[channel] = max(self._seen[channel], start + len(block.samples))
        self._flush()

    def _flush(self) -> None:
        """Write every record that all channels have now covered."""
        while all(self._seen[c] >= (self.records + 1) * self.per_record[c]
                  for c in self.channels):
            for channel in self.channels:
                n = self.per_record[channel]
                want = (self.records + 1) * n - self._base[channel]
                buf = self._buf[channel]
                if len(buf) < want:
                    buf.extend([0] * (want - len(buf)))
                take = buf[want - n:want]
                self._fh.write(np.asarray(take, dtype="<i2").tobytes())
                del buf[:want]
                self._base[channel] += want
            self.records += 1

    def close(self) -> None:
        if self._fh.closed:
            return
        try:
            self._fh.flush()
            self._fh.seek(236)                       # the number-of-records field
            self._fh.write(_text(str(self.records), 8))
        except OSError:
            pass                                     # -1 is still a readable file
        self._fh.close()

    def __enter__(self) -> "EDFWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_header(path: str | Path) -> dict:
    """The header back as a dict, for tests and for anything checking what was written."""
    with open(path, "rb") as fh:
        raw = fh.read(256)
        ns = int(raw[252:256])
        rest = fh.read(256 * ns)

    def field(offset: int, width: int) -> list[str]:
        return [rest[offset * ns + i * width:offset * ns + (i + 1) * width].decode().strip()
                for i in range(ns)]

    return {
        "version": raw[0:8].decode().strip(),
        "patient": raw[8:88].decode().strip(),
        "recording": raw[88:168].decode().strip(),
        "start_date": raw[168:176].decode().strip(),
        "start_time": raw[176:184].decode().strip(),
        "header_bytes": int(raw[184:192]),
        "records": int(raw[236:244]),
        "record_seconds": float(raw[244:252]),
        "signals": ns,
        "labels": field(0, 16),
        "units": field(96, 8),
        "physical_min": [float(x) for x in field(104, 8)],
        "physical_max": [float(x) for x in field(112, 8)],
        "digital_min": [int(x) for x in field(120, 8)],
        "digital_max": [int(x) for x in field(128, 8)],
        "prefilter": field(136, 80),
        "samples_per_record": [int(x) for x in field(216, 8)],
    }


def read_samples(path: str | Path) -> dict[int, np.ndarray]:
    """Every signal's digital samples, concatenated across records. For tests and quick checks."""
    header = read_header(path)
    counts = header["samples_per_record"]
    per_record = sum(counts)
    with open(path, "rb") as fh:
        fh.seek(header["header_bytes"])
        data = np.frombuffer(fh.read(), dtype="<i2")
    records = len(data) // per_record if per_record else 0
    data = data[:records * per_record].reshape(records, per_record) if records else \
        np.zeros((0, per_record), dtype="<i2")
    out, at = {}, 0
    for i, n in enumerate(counts):
        out[i] = data[:, at:at + n].reshape(-1)
        at += n
    return out
