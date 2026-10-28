"""Pupil readings, running beside whatever else the app is doing.

`mendi.MendiRunner`'s twin, with the same lifecycle and for the same reasons: one source owned by the
app on its own thread, `attach` and `release` as the whole lifecycle, a fresh record for every fresh
source, and a stale flag so a frozen snapshot cannot drive the room after the source has died.
"""

from __future__ import annotations

import time

from nfb.pupil import PupilProcessor

from .runner import SensorRunner


class PupilRunner(SensorRunner):
    STALE_S = 2.0
    """No reading for this long, with the reader alive, means the source has gone quiet. Readings arrive
    at 60 Hz, so two seconds is more than a hundred missed ones."""
    _THREAD_NAME = "pupil"
    NO_SOURCE = "no pupil source — choose one on the device screen"

    def __init__(self, choice=None):
        super().__init__(choice)
        self.processor = PupilProcessor()
        self.sample = None                  # the last reading as the source gave it, both eyes included

    def _ready(self, state) -> bool:
        return state.samples_seen > 5

    def _reset(self) -> None:
        self.processor.reset()
        self.sample = None

    def _read(self, gen: int) -> str | None:
        for sample in self.source.samples():
            if self._stop.is_set() or gen != self._generation:
                return None
            self.sample = sample
            self.state = self.processor.push(sample)
            self.seen_at = time.monotonic()
        return getattr(self.source, "error", None) or "the source stopped sending readings"

    def steer(self, effort: float) -> None:
        """Tell a simulated eye what its person is trying to do. A real eye has no such attribute, and
        this then does nothing — which is right, since a real person needs no telling."""
        source = self.source
        if source is not None and hasattr(source, "effort"):
            source.effort = effort

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
