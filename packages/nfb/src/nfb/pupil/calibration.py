"""Calibration for the eye tracker: follow a dot, so the tracker learns where the eyes look.

Calibration fits a model of the eyes from a few points of known position, which makes the gaze point
accurate. Pupil size does not need it, so it is optional; where it earns its place is the fixation
check, where an accurate gaze point tells the pupil screen that the eyes stay on the fixation point. It
belongs to the device, not a protocol, so it runs from the device screen.

The sequence is the tracker's: start, clear, show and sample each point while the eyes rest on it,
compute, stop, then read the model back as opaque bytes. The bytes are applied again on every
connection, because the tracker forgets the model when a session closes, and kept per tracker in the
operator home directory, outside this tree — personal data, not program material.

Capture is gaze-verified: the tracker streams a rough gaze point before any calibration, so each point
is sampled once the live gaze has rested near it; a blink is tolerated, and a point that does not settle
is shown again. Not run against a tracker yet; the ops and payloads are pinned by `tests/test_tracker.py`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import tracker as tr

OP_START = 0x3F2
OP_STOP = 0x3FC
OP_CLEAR = 0x424
OP_ADD_POINT = 0x406
OP_DISCARD_POINT = 0x438
OP_COMPUTE = 0x42E
OP_RETRIEVE = 0x44C
OP_APPLY = 0x456

EYE_BOTH = 3            # the eye mask the tracker takes: 1 left, 2 right, 3 both

POINTS = ((0.5, 0.5), (0.1, 0.9), (0.5, 0.1), (0.9, 0.9), (0.1, 0.1), (0.5, 0.9), (0.9, 0.1))
"""Seven points on the screen, in fractions of its width and height, centre first. The spread a tracker
needs to fit a model over the whole screen."""

SETTLE_S = 0.7
"""A pause after the eyes land on a point before measuring begins, so the point is not sampled while the
eyes are still moving to it. The ring stays empty during this; it starts filling only after."""
DWELL_S = 1.0
"""How long an eye must stay tracked on a point, after the settle, before it is sampled. Capture is a
timer on a tracked eye, not a test of where the gaze points: an uncalibrated tracker's reported position
can be far from the truth, so neither nearness to the dot nor steadiness of that position can be required.
The person is told to look at the dot and hold, and a tracked eye for settle-plus-dwell is that look."""
GAP_TOLERANCE_S = 0.25
"""A loss shorter than this, such as a blink, does not reset the hold."""
POINT_TIMEOUT_S = 12.0
"""How long one point waits for the eyes before it is given up for this attempt."""
MAX_ATTEMPTS = 3
"""How many times the whole run is shown before it fails."""


def session_payload() -> bytes:
    return b"\x00\x00"


def add_point_payload(x: float, y: float, eye: int = EYE_BOTH) -> bytes:
    """A point to sample: two plain fixed-point numbers, then the eye mask. Not a point structure."""
    return b"\x00\x00" + tr.q42(x) + tr.q42(y) + tr.u32(tr.T_U32, eye)


def discard_point_payload(x: float, y: float) -> bytes:
    return b"\x00\x00" + tr.q42(x) + tr.q42(y)


def apply_payload(blob: bytes) -> bytes:
    """Send a saved model back to the tracker: the prefix, then the block of bytes as it was read."""
    return b"\x00\x00" + blob


def model_path(serial: str | None) -> Path:
    """Where a tracker's model is kept, named by its serial. Under the operator's home, never in this
    tree: it is personal data, not program material."""
    name = "".join(c for c in (serial or "default") if c.isalnum() or c in "-_") or "default"
    return Path(os.path.expanduser("~/.nfb/tobii")) / f"{name}.cal"


def save_model(serial: str | None, blob: bytes) -> Path:
    path = model_path(serial)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def load_model(serial: str | None) -> bytes | None:
    path = model_path(serial)
    return path.read_bytes() if path.exists() else None


def apply_model(session: "tr.TrackerSession", blob: bytes) -> None:
    """Send a saved model to the tracker on connect. Best effort: a model the tracker rejects leaves it
    on its own stored one, which is no worse than before."""
    try:
        session.ask(OP_APPLY, apply_payload(blob))
    except tr.TrackerError:
        pass


PENDING, RESTING, SAMPLED, REFUSED = "pending", "resting", "sampled", "refused"


@dataclass
class Calibrator:
    """Drives the calibration sequence over a connected session, one step per `step()`.

    The screen owns the clock and the drawing. It calls `step()` on every frame; the calibrator reads
    the live gaze, decides whether the current point has been held long enough, sends the tracker what
    the point needs, and reports where it is through the fields below. The screen draws the dot at
    `POINTS[self.index]` and shows `self.done`, `self.failed` and `self.message`.
    """

    session: "tr.TrackerSession"
    serial: str | None = None
    points: tuple[tuple[float, float], ...] = POINTS
    index: int = 0
    attempt: int = 1
    state: str = PENDING
    done: bool = False
    failed: bool = False
    applied: bool = False       # the model is computed and live on the device, saved or not
    message: str = ""
    blob: bytes | None = None
    _held_since: float | None = field(default=None, repr=False)
    _lost_at: float | None = field(default=None, repr=False)
    _point_since: float | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    last_gaze: tuple[float, float] | None = field(default=None, repr=False)
    last_screen_gaze: tuple[float, float] | None = field(default=None, repr=False)
    probe: dict = field(default_factory=dict, repr=False)
    gaze_seen: bool = field(default=False, repr=False)

    def start(self, now: float | None = None) -> None:
        self.session.ask(OP_START, session_payload())
        self.session.ask(OP_CLEAR, session_payload())
        self._started = True
        self._begin_point(time.monotonic() if now is None else now)

    def _begin_point(self, now: float) -> None:
        self.state = PENDING
        self._held_since = None
        self._lost_at = None
        self._point_since = now

    def _gaze(self):
        """The latest fixation signal the session has seen, draining the frames it has queued.

        `gaze_seen` records whether any frame at all arrived, and `probe` keeps the raw gaze columns of
        the last frame, so the screen can tell "the tracker sends no gaze" from "it sends no usable eye
        signal" and show what it does send — the two reasons a ring never fills, told apart."""
        point = None
        for frame in self.session.drain_gaze():
            self.gaze_seen = True
            self.probe = frame.probe
            if frame.gaze_2d is not None:
                self.last_screen_gaze = frame.gaze_2d
            if frame.fixation is not None:
                point = frame.fixation
        if point is not None:
            self.last_gaze = point
        return point

    def step(self, now: float | None = None) -> None:
        """One tick. Samples the point once the eyes have been tracked on it for the dwell.

        Capture is a timer that runs while an eye is tracked, not a test of where the gaze points: before
        calibration the tracker's reported position can be far from the truth, so nearness and even
        steadiness of that position cannot be required. The person is told to look at the dot and hold; a
        continuously tracked eye for the dwell is taken as that look. A short loss, such as a blink, does
        not reset the timer; a longer one does."""
        if self.done or self.failed:
            return
        now = time.monotonic() if now is None else now
        if not self._started:
            self.start(now)
        x, y = self.points[self.index]
        tracked = self._gaze() is not None
        if tracked:
            self._lost_at = None
            if self._held_since is None:
                self._held_since = now
            self.state = RESTING
            if now - self._held_since >= SETTLE_S + DWELL_S:
                self._sample(x, y, now)
                return
        elif self._held_since is not None:
            if self._lost_at is None:
                self._lost_at = now
            if now - self._lost_at > GAP_TOLERANCE_S:
                self._held_since = None
                self.state = PENDING
        if self._point_since is not None and now - self._point_since > POINT_TIMEOUT_S:
            self._give_up(now)

    def dwell_progress(self, now: float | None = None) -> float:
        """How far the current point is toward being sampled, 0 to 1, while the eyes hold still. The
        screen draws this as a ring that fills, so the hold is visible and the person knows when it is
        taken."""
        if self._held_since is None or self.state != RESTING:
            return 0.0
        now = time.monotonic() if now is None else now
        held = now - self._held_since - SETTLE_S
        return 0.0 if held < 0 else min(1.0, held / DWELL_S)

    def settling(self, now: float | None = None) -> bool:
        """Whether the current point has been shown a while without the eyes holding still — a cue to the
        person to look at the dot and keep their eyes on it."""
        if self._point_since is None or self.state == RESTING:
            return False
        now = time.monotonic() if now is None else now
        return now - self._point_since > 2.0

    def _sample(self, x: float, y: float, now: float) -> None:
        reply = self.session.ask(OP_ADD_POINT, add_point_payload(x, y), optional=True)
        if reply is None:
            self.session.ask(OP_DISCARD_POINT, discard_point_payload(x, y))
            self._give_up(now)
            return
        self.state = SAMPLED
        self.index += 1
        if self.index >= len(self.points):
            self._finish()
        else:
            self._begin_point(now)

    def _give_up(self, now: float) -> None:
        if self.attempt >= MAX_ATTEMPTS:
            self.failed = True
            self.message = "the eyes did not settle on enough points — try again, in steady light"
            self._leave()
            return
        self.attempt += 1
        self.index = 0
        self.session.ask(OP_CLEAR, session_payload())
        self._begin_point(now)

    def _finish(self) -> None:
        # Compute applies the model, so it is live from here. Reading it back to save is best effort —
        # a failed read-back still leaves the calibration working for this session.
        self.session.ask(OP_COMPUTE, session_payload())
        self._leave()
        self.applied = True
        self.blob = self.session.ask(OP_RETRIEVE, session_payload(), optional=True, timeout_s=5.0)
        if self.blob:
            save_model(self.serial, self.blob)
            self.message = "calibration done and saved — look around, the dot follows your eyes"
        else:
            self.message = ("calibration done — look around, the dot follows your eyes. It was not "
                            "saved, so it must be run again next time")
        self.done = True

    def refresh_gaze(self) -> None:
        """Drain gaze without advancing a point, so the live gaze dot keeps moving after calibration."""
        self._gaze()

    def _leave(self) -> None:
        if self._started:
            self.session.ask(OP_STOP, session_payload())
            self._started = False

    def cancel(self) -> None:
        """Leave calibration without saving. The tracker keeps whatever model it had."""
        self._leave()
        self.failed = True
        self.message = "calibration stopped"
