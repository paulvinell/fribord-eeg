"""The companion file: what a recording's own format cannot carry, kept beside it.

A native frame recording carries everything — the config blob, each frame's per-channel signal states,
the phase counter that makes a gap visible. An EDF export carries none of that by construction: it is
decoded samples on a fixed grid, which is exactly why every other tool can read it. That is not a
reason to treat EDF as second class; it is a reason for the states to travel in their own file.

So every recording gets `<stem>.notes.jsonl` beside it: JSON Lines, one object per line, `kind`
discriminating. A `session` line with the config blob and each channel's rate and scale, `alarm` lines
with wall timestamps, `mark` lines for a moment the protocol judged worth keeping — the same shape as an
alarm, and the opposite news — and one `quality` line per channel holding every second of the recording as
rejected/unusable fractions — run-length encoded, because a good night is one long run and a bad
minute is the only part worth the bytes.

Written repeatedly while recording rather than once at the end, since the thing it describes is a
night that may not survive to have an end.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

FORMAT = "nfb-notes/1"
SUFFIX = ".notes.jsonl"

MISSING = None
"""How a second the feed never delivered is written. Distinct from zero on purpose: no data is not
clean data, and it is the case an EDF file cannot express at all — its samples are zeros like any
other."""


def notes_path(recording: Path | str) -> Path:
    """`<stem>.notes.jsonl` beside the recording, whatever the recording's own suffix chain is."""
    path = Path(recording)
    stem = path.name
    for suffix in (".jsonl.gz", ".edf", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    else:
        stem = path.stem
    return path.with_name(stem + SUFFIX)


def encode_cells(cells) -> list[list]:
    """Per-second `(rejected, unusable)` fractions as `[count, rejected, unusable]` runs.

    Rounded to a thousandth before comparing, so a run is not broken by noise below what anyone reads.
    """
    out: list[list] = []
    for rejected, unusable in cells:
        if rejected is None or (isinstance(rejected, float) and math.isnan(rejected)):
            pair = [MISSING, MISSING]
        else:
            pair = [round(float(rejected), 3), round(float(unusable), 3)]
        if out and out[-1][1:] == pair:
            out[-1][0] += 1
        else:
            out.append([1] + pair)
    return out


def decode_cells(runs) -> list[tuple[float, float]]:
    """The inverse, for reading a night back. Missing seconds come back as NaN, as the log holds them."""
    out: list[tuple[float, float]] = []
    for run in runs:
        count, rejected, unusable = run[0], run[1], run[2]
        pair = ((float("nan"), float("nan")) if rejected is None
                else (float(rejected), float(unusable)))
        out.extend([pair] * int(count))
    return out


class NotesWriter:
    """Rewrite the companion file from a snapshot. Cheap enough to do repeatedly: a night of four
    channels is a few hundred runs once the clean stretches are collapsed."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.lines = 0

    def write(self, session: dict, alarms=(), quality=None, marks=()) -> None:
        """`quality` is `{channel: [(rejected, unusable), ...]}` — raw cells, encoded here.

        Written to a temporary file and moved into place, so a process killed mid-write leaves the
        previous complete file rather than half of a new one.
        """
        rows = [{"kind": "session", "format": FORMAT, **session}]
        rows += [{"kind": "alarm", "at": at, "event": event} for at, event in alarms]
        rows += [{"kind": "mark", "at": at, "event": event} for at, event in marks]
        for channel, cells in sorted((quality or {}).items()):
            rows.append({"kind": "quality", "channel": int(channel),
                         "runs": encode_cells(cells)})
        body = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
        tmp = self.path.with_name(self.path.name + ".part")
        try:
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, self.path)
            self.lines = len(rows)
        except OSError:
            # A companion is a convenience; never take a recording down for one.
            try:
                tmp.unlink()
            except OSError:
                pass


def read_notes(path: Path | str) -> dict:
    """The companion file back as `{"session": {...}, "alarms": [...], "marks": [...],
    "quality": {ch: [...]}}`."""
    out: dict = {"session": {}, "alarms": [], "marks": [], "quality": {}}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break                      # truncated tail: everything before it still stands
            kind = row.get("kind")
            if kind == "session":
                out["session"] = row
            elif kind == "alarm":
                out["alarms"].append((row.get("at"), row.get("event")))
            elif kind == "mark":
                out["marks"].append((row.get("at"), row.get("event")))
            elif kind == "quality":
                out["quality"][int(row["channel"])] = decode_cells(row.get("runs", ()))
    return out
