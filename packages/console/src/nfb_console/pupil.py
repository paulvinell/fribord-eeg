"""Pupil readings, running beside whatever else the app is doing.

`mendi.MendiRunner`'s twin, with the same lifecycle and for the same reasons: one source owned by the
app on its own thread, `attach` and `release` as the whole lifecycle, a fresh record for every fresh
source, and a stale flag so a frozen snapshot cannot drive the room after the source has died.
"""

from __future__ import annotations

import threading
import time

from nfb.pupil import PupilProcessor


class PupilRunner:
    STALE_S = 2.0
    """No reading for this long, with the reader alive, means the source has gone quiet. Readings arrive
    at 60 Hz, so two seconds is more than a hundred missed ones."""

    def __init__(self, choice=None):
        self.choice = choice
        self.processor = PupilProcessor()
        self.source = None
        self.state = None
        self.sample = None                  # the last reading as the source gave it, both eyes included
        self.error: str | None = None
        self.seen_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stale(self) -> bool:
        return (self.running and self.seen_at is not None
                and time.monotonic() - self.seen_at > self.STALE_S)

    @property
    def ready(self) -> bool:
        return self.state is not None and self.state.samples_seen > 5

    def steer(self, effort: float) -> None:
        """Tell a simulated eye what its person is trying to do. A real eye has no such attribute, and
        this then does nothing — which is right, since a real person needs no telling."""
        source = self.source
        if source is not None and hasattr(source, "effort"):
            source.effort = effort

    def attach(self, choice=None) -> bool:
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
        self.processor.reset()
        self.state = None
        self.sample = None
        self.seen_at = None
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pupil")
        self._thread.start()
        return True

    def _run(self) -> None:
        try:
            for sample in self.source.samples():
                if self._stop.is_set():
                    return
                self.sample = sample
                self.state = self.processor.push(sample)
                self.seen_at = time.monotonic()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return
        if not self._stop.is_set() and self.error is None:
            self.error = getattr(self.source, "error", None) or "the source stopped sending readings"

    def release(self) -> None:
        self._stop.set()
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
            return "no pupil source — choose one on the device screen"
        if self.source is None:
            return "not started"
        return self.source.describe()

    def status(self) -> tuple[str, str]:
        """The connection state as (word, tone-name), for a clear line on the device screen. The tone
        names map to theme colours there."""
        if self.choice is None or not self.choice.on:
            return "off", "muted"
        if self.error:
            return self.error, "bad"
        if self.ready:
            return "connected — readings arriving", "good"
        if self.running:
            return "opening…", "accent"
        return "not started", "muted"

    @property
    def handshake_hex(self) -> dict:
        """The raw handshake bytes from the source, for diagnosing a tracker that will not open."""
        return getattr(self.source, "handshake_hex", {}) or {}

    @property
    def position(self):
        """Where the eyes sit, for the positioning guide, or None when the source does not report it."""
        return getattr(self.source, "position", None)
