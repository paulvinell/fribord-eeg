"""Beats, running alongside whatever else the session is doing.

Separate from `TrainingSession` and attachable to it, because the heart is watched *while* the EEG is
trained, not instead of it. A person can appear to improve in the EEG while the heart-rate spectrum
splits, and that is a failure you can only see by having both in front of you at once. So the heart
instruments sit beside the EEG instruments on more than one screen.

So this is one object, owned by the app like the device is, with two consumers: the HRV screen trains on
it, and the CFC screen shows the same instruments beside the EEG and warns when the spectrum splits.
"""

from __future__ import annotations

import time

from nfb.hrv import HRVProcessor

from .runner import SensorRunner

SCAN_SECONDS = 8.0


class HeartRunner(SensorRunner):
    """Owns a beat source and its processor for as long as it is running.

    Its own thread rather than the session's, because the two streams are independent: a strap keeps
    beating whether or not the amplifier is streaming, and a session that ends should not have to decide
    what happens to the heart. The one runner that reconnects on its own (`RETRY_SECONDS`): a chosen strap
    is the operator saying "this is on my body", so a cable plugged in late just works, no return trip to
    the device screen.
    """

    RETRY_SECONDS = 3.0
    """How long to wait before opening a chosen source again. Long enough that a cable that is not there
    is not being asked for thirty times a minute, short enough that plugging one in feels like it just
    works."""

    STALE_S = 5.0
    """No beat for this long, with the reader still alive, means the strap has gone quiet. Several
    missed beats at any plausible rate, and the same figure the spectrum uses to call a hole a dropout
    (`nfb.hrv.beats.GAP_MIN_S`)."""
    _THREAD_NAME = "heart"
    NO_SOURCE = "no heart-rate source — choose one on the device screen"

    def __init__(self, choice=None):
        super().__init__(choice)
        self.processor = HRVProcessor()
        self._reading: tuple | None = None   # what the live reader was opened for, if any

    def _ready(self, state) -> bool:
        return state.beats > 2

    def _reset(self) -> None:
        # On a new source only, not between reconnect attempts: a gap's beats are worth keeping (`_reopen`
        # does not call this).
        self.processor.reset()

    def _wanted(self) -> tuple:
        """What the current choice asks for, in the terms that decide whether a reader is the right one:
        the kind, and whichever device it names."""
        c = self.choice
        return (getattr(c, "kind", None), getattr(c, "address", ""), getattr(c, "port", ""))

    def _already_serving(self) -> bool:
        # A retrying reader is still running, so keep the live one only when it is reading what is now asked;
        # a different choice replaces it.
        if self.running:
            if self._reading == self._wanted():
                return True
            self.release()
        return False

    def _on_attach(self) -> None:
        self._reading = self._wanted()

    def _on_release(self) -> None:
        self._reading = None

    def _read(self, gen: int) -> str | None:
        for beat in self.source.beats():
            if self._stop.is_set() or gen != self._generation:
                return None
            self.state = self.processor.push(beat)
            # Only a beat with usable data is news: a strap that lost contact keeps notifying with intervals
            # the processor discards, and refreshing on those stops `stale` ever firing.
            if beat.contact is not False and beat.rr_intervals:
                self.seen_at = time.monotonic()
        return getattr(self.source, "error", None) or "the beat source stopped delivering"

    def _reopen(self) -> bool:
        source, self.source = self.source, self.choice.open()
        try:
            source.close()
        except Exception:
            pass
        if self.source is None:
            return False
        self.error = None
        return True

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
