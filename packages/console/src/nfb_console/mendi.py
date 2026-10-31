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

import time
from collections import deque

from nfb.heg import FrameProcessor, PulseExtractor, SignalProcessor
from nfb.hrv import HRVProcessor

from .runner import BEATS_KEPT, SensorRunner

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


class MendiRunner(SensorRunner):
    """Owns a frame source and its processing for as long as it is running.

    Two things are stacked here: `FrameProcessor` (filters, motion gate, projection) and `SignalProcessor`
    (the rolling windows), chained in `_read` so nothing above knows there are two. Beside them the same
    frames feed `PulseExtractor`, whose beats become `heart_state`, the headband's own heart rate in the
    form `HeartRunner.state` takes.
    """

    STALE_S = 3.0
    """Tighter than the heart's five seconds: the headband streams at ~30 Hz, so three seconds of silence
    is hundreds of missed frames rather than a few missed beats."""
    _THREAD_NAME = "mendi"
    NO_SOURCE = "no headband — choose one on the device screen"

    def __init__(self, choice=None):
        super().__init__(choice)
        self.processor = FrameProcessor()
        self.signal = SignalProcessor()
        self.pulse = PulseExtractor()
        self.heart_processor = HRVProcessor()
        self.heart_state = None             # the heart rate from the headband's own frames
        self.beat_times: deque[float] = deque(maxlen=BEATS_KEPT)

    def _ready(self, state) -> bool:
        return state.samples_seen > 5

    def _reset(self) -> None:
        self.processor.reset()
        self.signal.reset()
        self.pulse.reset()
        self.heart_processor.reset()
        self.heart_state = None
        self.beat_times.clear()

    def _read(self, gen: int) -> str | None:
        for frame in self.source.frames():
            if self._stop.is_set() or gen != self._generation:
                return None
            self.state = self.signal.push(self.processor.process(frame), self.processor.gate_fraction)
            for beat in self.pulse.push(frame):
                self.heart_state = self.heart_processor.push(beat)
                self.beat_times.append(beat.t)
            if self.pulse.trouble:
                self.heart_state = None     # no heart rate from readings that carry none
            self.seen_at = time.monotonic()
        # The generator ending is how `LiveMendi` reports its BLE worker dying; its own `error` knows why.
        source = self.source
        return (getattr(source, "error", None) if source is not None else None) \
            or "the headband stopped delivering frames"

    def describe(self) -> str:
        # The out-of-range state outranks the source's own line, though not a dead reader: the headband
        # connects and streams, and its readings carry no pulse until it is restarted.
        return self.error or self.pulse.trouble or super().describe()

    @property
    def waveform(self) -> list[tuple[float, float]]:
        """The last minute of the forehead pulse, pointing up; see `runner.PulseRunner`."""
        return list(self.pulse.waveform.copy())    # one step; the reader thread appends meanwhile

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
