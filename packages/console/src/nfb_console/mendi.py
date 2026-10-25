"""Frames from a Mendi headband, running beside whatever else the app is doing.

The twin of `heart.HeartRunner`, and deliberately the same shape: a source and its processing owned by
one object on its own thread, with `attach` and `release` as the whole lifecycle. The HRV runner earned
every one of those decisions the hard way — a frozen snapshot driving the room after the strap died, a
generator ending in silence, a session inheriting the previous one's buffers — and an fNIRS stream has
exactly the same failure modes, so it arrives with the answers rather than rediscovering them.

Two things are stacked here that `HeartRunner` keeps in one object: `FrameProcessor` (filters, motion
gate, projection) and `SignalProcessor` (the rolling windows). That is how `nfb.heg` divides them —
"what happened to this frame" and "what does the signal look like now" — and the runner just chains
them, so nothing above has to know there are two.
"""

from __future__ import annotations

import threading
import time

from nfb.heg import FrameProcessor, SignalProcessor

SCAN_SECONDS = 8.0


def scan(seconds: float = SCAN_SECONDS) -> tuple[list[tuple[str, str]], str | None]:
    """Look for headbands now, synchronously. Returns (name, address) pairs and a problem.

    Blocking on purpose, and therefore called from a worker thread by the screen — `heart.scan`'s twin,
    for the reason written there: a scan that reports nothing leaves you unable to tell "no headband"
    from "did not run".

    Matched the way `LiveMendi` matches when it connects — the vendor service UUID first, the name as a
    fallback — because a list that used a different rule would offer devices the connection then refused,
    or hide one it would have taken.
    """
    from nfb.heg import problem
    from nfb.heg.source import SERVICE_UUID

    trouble = problem()
    if trouble:
        return [], trouble
    try:
        import asyncio

        from bleak import BleakScanner

        async def go():
            return await BleakScanner.discover(timeout=seconds, return_adv=True)

        found = asyncio.run(go())
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    out: list[tuple[str, str]] = []
    for device, adv in found.values():
        name = device.name or adv.local_name or "unnamed"
        uuids = [str(u).lower() for u in (adv.service_uuids or [])]
        if SERVICE_UUID in uuids or "mendi" in name.lower():
            out.append((name, device.address))
    return out, None


class MendiRunner:
    """Owns a frame source and its processing for as long as it is running."""

    STALE_S = 3.0
    """No frame for this long, with the reader still alive, means the headband has gone quiet.

    Much tighter than the heart's five seconds, and it should be: the device streams at ~30 Hz, so three
    seconds of silence is hundreds of missed frames rather than a few missed beats. The reason for having
    the check at all is the same — `state` freezes on whatever arrived last and nothing about the object
    says when that was, which is how a dead source keeps driving the outputs from a snapshot.
    """

    def __init__(self, choice=None):
        self.choice = choice
        self.processor = FrameProcessor()
        self.signal = SignalProcessor()
        self.source = None
        self.state = None
        self.error: str | None = None
        self.seen_at: float | None = None    # monotonic arrival of the last frame, written by _run only
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def stale(self) -> bool:
        """Whether the snapshot in `state` has stopped being news.

        Arrival time rather than the frame's own timestamp, because the simulator and the tests carry
        synthetic clocks; and only while the reader is alive, because a dead reader sets `error` and that
        is the louder signal.
        """
        return (self.running and self.seen_at is not None
                and time.monotonic() - self.seen_at > self.STALE_S)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ready(self) -> bool:
        """Enough frames for the windowed numbers to be arithmetic rather than a single reading.

        A low bar on purpose: it says the stream is alive, not that the signal means anything. The
        settling clock in `FrameProcessor` holds quality at 0.5 for eleven seconds and each metric carries
        its own `ready` predicate — this is only the check that anything is arriving at all.
        """
        return self.state is not None and self.state.samples_seen > 5

    def attach(self, choice=None) -> bool:
        """Start reading, if a source is chosen. Idempotent."""
        if choice is not None:
            self.choice = choice
        if self.choice is None or not self.choice.on:
            self.release()
            return False
        if self.running:
            return True
        trouble = self.choice.problem()
        if trouble:
            self.error = trouble
            return False
        self.source = self.choice.open()
        if self.source is None:
            return False
        # A fresh source starts a fresh record — the lesson `HeartRunner.attach` records, and it bites
        # harder here: the low-pass filters hold the *previous* source's DC level, the settling clock
        # would arrive pre-satisfied, and the motion gate's window would carry another headband's
        # accelerometer. Same-source continuity is not lost, since an attach onto a running reader
        # returns above without passing here.
        self.processor.reset()
        self.signal.reset()
        self.state = None
        self.seen_at = None
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mendi")
        self._thread.start()
        return True

    def _run(self) -> None:
        try:
            for frame in self.source.frames():
                if self._stop.is_set():
                    return
                self.state = self.signal.push(self.processor.process(frame),
                                              self.processor.gate_fraction)
                self.seen_at = time.monotonic()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return
        # The generator ending is how `LiveMendi` reports its BLE worker dying — the connection dropped,
        # the headband switched off, bleak raised inside the asyncio thread. An end nobody asked for is a
        # fault and has to say so, or `state` freezes on the last frame and the feedback runs on it
        # indefinitely. The source's own `error` is preferred because it knows what actually happened.
        if not self._stop.is_set() and self.error is None:
            source = self.source
            self.error = (getattr(source, "error", None) if source is not None else None) \
                or "the headband stopped delivering frames"

    def release(self) -> None:
        self._stop.set()
        source = self.source
        self.source = None
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def describe(self) -> str:
        if self.error:
            return self.error
        if self.choice is not None and not self.choice.on:
            return "no headband — choose one on the device screen"
        if self.source is None:
            return "not started"
        return self.source.describe()

    @property
    def battery_pct(self) -> int | None:
        """What the headband last said about its charge, when it is the kind of source that knows."""
        return getattr(self.source, "battery_pct", None)

    @property
    def firmware_version(self) -> str | None:
        return getattr(self.source, "firmware_version", None)

    @property
    def gate_fraction(self) -> float:
        """Share of frames held for motion — the screen's honesty figure."""
        return self.processor.gate_fraction
