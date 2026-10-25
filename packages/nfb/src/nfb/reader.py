"""A dedicated reader thread, so processing cannot delay acquisition.

Measured motivation: with reading and processing in one loop at 1000 sps, 0.24% of reads fell more
than one frame behind, and because a frame covers only 8 ms at that rate a brief OS scheduling
hiccup queued eight frames — 64 ms of extra latency. It was not compute (2.8 ms p99 against an 8 ms
budget) and not garbage collection (disabling it made it marginally worse). It was simply that
anything stalling the loop also stalled the reader.

Separating them works because `serial.read()` releases the GIL while it waits, so the reader thread
can be woken to drain the USB buffer while the consumer is busy in numpy. The thread does nothing
but read and enqueue — no decoding, no file I/O — because every line in it is a line that can delay
the next read.

Timestamps are taken here, at the moment of arrival, which is the earliest the host can observe a
frame.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import optima


@dataclass(frozen=True)
class Arrival:
    """One frame as it arrived, with the host time it arrived at."""

    code: int
    payload: bytes
    at: float                 # time.monotonic() when the frame was read
    queued: int               # frames waiting behind it, for backlog diagnostics


class FrameReader:
    """Reads frames from a device on its own thread into a bounded queue.

    The queue is deliberately large: dropping a frame loses signal irrecoverably, so the default
    holds about 30 s of frames even at 2000 sps. `dropped` counts anything lost anyway, which means
    the consumer is not merely late but fundamentally too slow.
    """

    def __init__(self, dev: optima.Optima, queue_size: int = 8192):
        self.dev = dev
        self.queue: queue.Queue[Arrival | BaseException] = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="optima-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self.dev.read_message()
            except TimeoutError:
                continue                      # idle link; keep waiting
            except BaseException as exc:      # noqa: BLE001 - must reach the consumer
                try:
                    self.queue.put_nowait(exc)
                except queue.Full:
                    pass
                return
            try:
                self.queue.put_nowait(
                    Arrival(msg.code, msg.payload, time.monotonic(), self.queue.qsize())
                )
            except queue.Full:
                self.dropped += 1             # consumer is too slow, not just late

    def arrivals(self, timeout: float = 5.0) -> Iterator[Arrival]:
        """Yield frames as they arrive. Re-raises anything the reader thread hit."""
        while True:
            try:
                item = self.queue.get(timeout=timeout)
            except queue.Empty:
                if self._stop.is_set() or not self._thread.is_alive():
                    return
                raise TimeoutError(f"no frame for {timeout}s")
            if isinstance(item, BaseException):
                raise item
            yield item

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
