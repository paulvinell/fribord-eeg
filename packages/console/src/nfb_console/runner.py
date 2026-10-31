"""The shared lifecycle of a sensor the app runs beside a session — the heart strap, the HEG headband,
the eye tracker.

Each owns one source on its own thread, with `attach`/`release` as the whole lifecycle, a fresh record for
every fresh source, a stale flag so a frozen snapshot cannot drive the room after the source dies, and a
generation tag so a reader stuck past its join cannot write onto its healthy replacement. All three were
once separate classes with that logic pasted into each — which is how a fix for the stale-reader race landed
in two of them and was missed in the third. Now there is one body, and the three differ only in what they
read and how they judge a reading, below.

Not the EEG amplifier (`nfb.Device`): that owns an exclusive port and fans out to many subscribers, a
different shape. These are one-consumer readers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Protocol


class SensorRunner:
    """Owns a source and its processing for as long as it is running.

    A subclass sets `STALE_S`, `_THREAD_NAME`, `NO_SOURCE` and (for a source that reconnects on its own)
    `RETRY_SECONDS`, and implements `_reset`, `_ready` and `_read`. Everything about the thread, the
    generation tag and the lifecycle lives here.
    """

    STALE_S = 3.0
    """No reading for this long, with the reader alive, means the source has gone quiet. `state` freezes on
    whatever arrived last and nothing about the object says when, which is how a dead source keeps driving
    the outputs from a snapshot. Arrival time, not the reading's own stamp, since simulators carry their own
    clock; and only while the reader lives, because a dead reader sets `error`, the louder signal."""

    RETRY_SECONDS: float | None = None
    """Seconds to wait before opening the source again after it ends. `None` means a source that ends has
    failed for good (the headband, the tracker); a number means one that reconnects on its own (the strap:
    a cable plugged in late is not a reason to send the operator back to the device screen)."""

    _THREAD_NAME = "sensor"
    NO_SOURCE = "no source — choose one on the device screen"

    def __init__(self, choice=None):
        self.choice = choice
        self.source = None
        self.state = None
        self.error: str | None = None
        self.seen_at: float | None = None    # monotonic arrival of the last reading, written by _read only
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Each reader carries its generation, so a stale one that outlived its join cannot write onto a
        # healthy replacement — a handshake can block well past the stop event.
        self._generation = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stale(self) -> bool:
        return (self.running and self.seen_at is not None
                and time.monotonic() - self.seen_at > self.STALE_S)

    @property
    def ready(self) -> bool:
        """Enough data that the numbers are arithmetic, not a single reading. A low bar: it says the stream
        is alive, not that the signal means anything — each metric carries its own readiness besides."""
        return self.state is not None and self._ready(self.state)

    def attach(self, choice=None) -> bool:
        """Start reading, if a source is chosen. Idempotent."""
        if choice is not None:
            self.choice = choice
        if self.choice is None or not self.choice.on:
            self.release()
            return False
        if self._already_serving():
            return True
        trouble = self.choice.problem()
        if trouble:
            self.error = trouble
            return False
        self.source = self.choice.open()
        if self.source is None:
            return False
        # A fresh source starts a fresh record, or the processor carries the previous source's state. Same-
        # source continuity is not lost: an attach onto a running reader returns above.
        self._reset()
        self.state = None
        self.seen_at = None
        self.error = None
        self._on_attach()
        self._stop.clear()
        self._generation += 1
        self._thread = threading.Thread(target=self._run, args=(self._generation,),
                                        daemon=True, name=self._THREAD_NAME)
        self._thread.start()
        return True

    def _run(self, gen: int) -> None:
        while True:
            try:
                why = self._read(gen)       # read until the source ends; None means we were told to stop
            except Exception as exc:
                why = f"{type(exc).__name__}: {exc}"
            if why is None or self._stop.is_set() or gen != self._generation:
                return                      # stopped, or a newer reader took over — a clean end, not a fault
            if self.RETRY_SECONDS is None:
                if not self._stop.is_set() and self.error is None:
                    self.error = why        # the source ended on its own: a fault, so the room stops on it
                return
            # A source that reconnects: record the fault, wait, open it again, and keep reading.
            self.error = f"{why} · trying again"
            if self._stop.wait(self.RETRY_SECONDS) or self.choice is None or gen != self._generation:
                return
            if not self._reopen():
                return

    def release(self) -> None:
        self._stop.set()
        self._on_release()
        source, self.source = self.source, None
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def describe(self) -> str:
        if self.error:
            return self.error
        if self.choice is not None and not self.choice.on:
            return self.NO_SOURCE
        if self.source is None:
            return "not started"
        return self.source.describe()

    # --- what a subclass fills in --------------------------------------------------------------

    def _ready(self, state) -> bool:
        """Whether `state` holds enough to count as a live reading."""
        raise NotImplementedError

    def _reset(self) -> None:
        """Clear the processing so a fresh source starts a fresh record."""
        raise NotImplementedError

    def _read(self, gen: int) -> str | None:
        """Read from `self.source` until it ends, writing `state`/`seen_at` as readings arrive. Return the
        reason it ended, or `None` when `self._stop` is set or `gen` is no longer current (both mean: do not
        touch shared state). The per-reading loop must check `self._stop.is_set() or gen != self._generation`
        and return `None` — that is what keeps a superseded reader from writing over its replacement."""
        raise NotImplementedError

    def _already_serving(self) -> bool:
        """Whether a running reader already satisfies the current choice, so attach is a no-op."""
        return self.running

    def _on_attach(self) -> None:
        """Extra state to record at the moment of a successful attach."""

    def _on_release(self) -> None:
        """Extra state to clear on release."""

    def _reopen(self) -> bool:
        """For a reconnecting source: open it again. Return False to give up retrying."""
        return True


BEATS_KEPT = 256
"""How many raw beat times a pulse runner keeps: four minutes at a resting rate."""


class PulseRunner(Protocol):
    """What a runner that reads a pulse hands out, whichever device it reads.

    The HeartMath's runner and the Mendi's both provide it, so a program that compares them treats the
    two alike and neither device learns about the other.
    """

    heart_state: object | None
    """The heart-rate state from `nfb.hrv.HRVProcessor`, or None before beats arrive."""

    beat_times: deque
    """When each recent beat happened, in seconds, as the device's own processing stamped it."""

    @property
    def waveform(self) -> list[tuple[float, float]] | None:
        """The last minute of the pulse as `(time, value)` pairs, pulse pointing up, or None when the
        source sends no waveform."""
