"""Beats, running alongside whatever else the session is doing.

Separate from `TrainingSession` and attachable to it, because the heart is watched *while* the EEG is
trained, not instead of it. A person can appear to improve in the EEG while the heart-rate spectrum
splits, and that is a failure you can only see by having both in front of you at once. So the heart
instruments sit beside the EEG instruments on more than one screen.

So this is one object, owned by the app like the device is, with two consumers: the HRV screen trains on
it, and the CFC screen shows the same instruments beside the EEG and warns when the spectrum splits.
"""

from __future__ import annotations

import threading
import time

from nfb.hrv import HRVProcessor

SCAN_SECONDS = 8.0


class HeartRunner:
    """Owns a beat source and its processor for as long as it is running.

    Its own thread rather than the session's, because the two streams are independent: a strap keeps
    beating whether or not the amplifier is streaming, and a session that ends should not have to decide
    what happens to the heart. `attach` and `release` are the whole lifecycle.
    """

    RETRY_SECONDS = 3.0
    """How long to wait before opening a chosen source again. Long enough that a cable that is not there
    is not being asked for thirty times a minute, short enough that plugging one in feels like it just
    works."""

    STALE_S = 5.0
    """No beat for this long, with the reader still alive, means the strap has gone quiet. Several
    missed beats at any plausible rate, and the same figure the spectrum uses to call a hole a dropout
    (`nfb.hrv.beats.GAP_MIN_S`)."""

    def __init__(self, choice=None):
        self.choice = choice
        self.processor = HRVProcessor()
        self.source = None
        self.state = None
        self.error: str | None = None
        self.seen_at: float | None = None    # monotonic arrival of the last beat, written by _run only
        self._reading: tuple | None = None   # what the live reader was opened for, if any
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def stale(self) -> bool:
        """Whether the snapshot in `state` has stopped being news.

        `state` freezes on whatever beat came last, and nothing about the object says when that was —
        which is how a dead strap kept driving the volume, the shadow and the clicks from a snapshot
        for as long as anyone let it. Arrival time rather than the beat's own timestamp, because replays
        and tests carry synthetic clocks; and only while the reader is alive, because a dead reader sets
        `error` and that is the louder signal.
        """
        return (self.running and self.seen_at is not None
                and time.monotonic() - self.seen_at > self.STALE_S)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ready(self) -> bool:
        return self.state is not None and self.state.beats > 2

    def _wanted(self) -> tuple:
        """What the current choice asks for, in the terms that decide whether a reader is the right one:
        the kind, and whichever device it names."""
        c = self.choice
        return (getattr(c, "kind", None), getattr(c, "address", ""), getattr(c, "port", ""))

    def attach(self, choice=None) -> bool:
        """Start reading, if a source is chosen. Idempotent."""
        if choice is not None:
            self.choice = choice
        if self.choice is None or not self.choice.on:
            self.release()
            return False
        if self.running:
            # A reader that is retrying is still running, so "already going" is no longer the same
            # question as "already going on what you just asked for". Attaching a different source onto
            # a live reader used to be answered yes while the old one kept reading.
            if self._reading == self._wanted():
                return True
            self.release()
        trouble = self.choice.problem()
        if trouble:
            self.error = trouble
            return False
        self.source = self.choice.open()
        if self.source is None:
            return False
        # A fresh source starts a fresh record. Nothing ever reset the processor, so a session started
        # within the buffer's three minutes of the last built its first spectrum largely from the
        # previous session's beats, the artifact filter's 20 s warm-up arrived pre-satisfied, and its
        # pending beat was ectopic-tested against a beat from another session — or another *device*,
        # since attach is also the path that switches simulator to strap. Same-source continuity is not
        # lost: an attach onto a running reader returns above without passing here.
        self.processor.reset()
        self.state = None
        self.seen_at = None
        self.error = None
        self._reading = self._wanted()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="heart")
        self._thread.start()
        return True

    def _read_once(self) -> str | None:
        """One attempt at the current source: read until it ends, and say why it did.

        None means it was told to stop, which is the only ending that is not a fault. The generator
        ending on its own is how a reader reports its worker dying, and it used to end in silence:
        `error` stayed None, `state` froze on the last beat, and the feedback ran on the frozen snapshot
        indefinitely.
        """
        try:
            for beat in self.source.beats():
                if self._stop.is_set():
                    return None
                self.state = self.processor.push(beat)
                self.seen_at = time.monotonic()
            why = getattr(self.source, "error", None) or "the beat source stopped delivering"
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
        return None if self._stop.is_set() else why

    def _run(self) -> None:
        """Read beats, and keep trying to.

        **A chosen source connects itself.** Choosing one is the operator saying "this is what is on my
        body"; a cable not plugged in yet, a strap switched on a minute later, a lead pulled out and put
        back are all ordinary, and none of them is a reason to make somebody go back to the device screen
        and click the same button again. So the reader outlives one attempt: a source that will not open,
        or that stops delivering, is recorded as the fault it is and tried again a few seconds later,
        until `release` — which is the operator choosing something else, or the session ending.

        The processor is deliberately *not* reset between attempts. A reconnection leaves a hole in the
        record, and the spectrum already ends its window at a hole rather than analysing across it, so
        throwing away the beats either side of the gap would cost the history for nothing.
        """
        while not self._stop.is_set():
            why = self._read_once()
            if why is None:
                return
            self.error = f"{why} · trying again"
            if self._stop.wait(self.RETRY_SECONDS) or self.choice is None:
                return
            source, self.source = self.source, self.choice.open()
            try:
                source.close()
            except Exception:
                pass
            if self.source is None:
                return
            self.error = None

    def release(self) -> None:
        self._stop.set()
        self._reading = None
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
            return "no heart-rate source — choose one on the device screen"
        if self.source is None:
            return "not started"
        return self.source.describe()

    @property
    def spectrum(self):
        return self.state.spectrum if self.state is not None else None

    @property
    def split(self) -> bool:
        """The warning that justifies having this next to the EEG at all."""
        spectrum = self.spectrum
        return bool(spectrum is not None and spectrum.ready and spectrum.split)


def scan(seconds: float = SCAN_SECONDS) -> tuple[list[tuple[str, str]], str | None]:
    """Look for heart-rate straps now, synchronously. Returns (name, address) pairs and a problem.

    Blocking on purpose, and therefore called from a worker thread by the screen: the alternative is a
    scan that reports nothing and leaves the operator unsure whether it ran.
    """
    from nfb.hrv import sources

    trouble = sources.problem()
    if trouble:
        return [], trouble
    try:
        import asyncio

        from bleak import BleakScanner

        async def go():
            return await BleakScanner.discover(timeout=seconds,
                                               service_uuids=[sources.HR_SERVICE])

        found = asyncio.run(go())
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return [((d.name or "unnamed"), d.address) for d in found], None
