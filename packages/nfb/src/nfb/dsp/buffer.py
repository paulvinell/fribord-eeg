"""Buffering blocks into windows, on an absolute sample grid.

Windowed metrics need contiguous runs of samples. Blocks arrive in frame-sized pieces, and the
stream can contain gaps, so the buffer's real job is refusing to hand out a window that spans
lost samples: a metric computed across a gap is wrong rather than noisy, and there is no way to
tell afterwards. Windows are addressed by absolute sample index, which is what lets several
channels be sliced on the same grid for cross-channel measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stream import Block, ChannelSpec


@dataclass(frozen=True)
class Window:
    """A contiguous run of samples in physical units, ready to measure."""

    channel: int
    start_index: int
    samples: np.ndarray          # float, physical units (spec.unit)
    spec: ChannelSpec
    valid: bool                  # device signal state was ok for every sample

    @property
    def start_seconds(self) -> float:
        return self.start_index / self.spec.rate_hz

    @property
    def duration(self) -> float:
        return len(self.samples) / self.spec.rate_hz

    @property
    def centre_seconds(self) -> float:
        return self.start_seconds + self.duration / 2


class ChannelBuffer:
    """Accumulates one channel's samples, keyed by absolute index.

    A discontinuity discards what came before it: buffered samples on the far side of a gap can
    never form a valid window with samples on this side, so keeping them would only invite a
    metric to straddle the hole.
    """

    def __init__(self, spec: ChannelSpec, capacity_seconds: float = 60.0):
        self.spec = spec
        self.capacity = max(1, int(capacity_seconds * spec.rate_hz))
        self._samples = np.zeros(0, dtype=np.float64)
        self._valid = np.zeros(0, dtype=bool)
        self._start = 0                     # absolute index of _samples[0]
        self.discontinuities = 0

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._start + len(self._samples)

    def push(self, block: Block) -> None:
        if block.channel != self.spec.index:
            raise ValueError(f"block for ch{block.channel} pushed to ch{self.spec.index}")
        if len(self._samples) and block.start_index != self.end:
            self.discontinuities += 1
            self._samples = np.zeros(0, dtype=np.float64)
            self._valid = np.zeros(0, dtype=bool)
        if not len(self._samples):
            self._start = block.start_index
        vals = np.asarray(block.samples, dtype=np.float64) * self.spec.scale
        self._samples = np.concatenate((self._samples, vals))
        self._valid = np.concatenate((self._valid, np.full(len(vals), block.valid)))
        if len(self._samples) > self.capacity:
            drop = len(self._samples) - self.capacity
            self._discard(drop)

    def _discard(self, n: int) -> None:
        self._samples = self._samples[n:]
        self._valid = self._valid[n:]
        self._start += n

    def discard_before(self, index: int) -> None:
        if index > self._start:
            self._discard(min(index - self._start, len(self._samples)))

    def take(self, start: int, size: int) -> Window | None:
        """The window at absolute `start`, or None if not fully buffered."""
        if start < self._start or start + size > self.end:
            return None
        off = start - self._start
        return Window(
            channel=self.spec.index,
            start_index=start,
            samples=self._samples[off:off + size].copy(),
            spec=self.spec,
            valid=bool(self._valid[off:off + size].all()),
        )


class Windower:
    """Emits fixed-size windows for one channel at a fixed hop.

    Window starts are anchored to the first sample seen and advance by `hop` regardless of
    gaps, so a given window always covers the same absolute indices no matter how the data
    arrived — two runs over the same recording produce identical windows.
    """

    def __init__(self, spec: ChannelSpec, size: int, hop: int | None = None,
                 capacity_seconds: float = 60.0):
        if size <= 0:
            raise ValueError("window size must be positive")
        self.buffer = ChannelBuffer(spec, capacity_seconds)
        self.size = size
        self.hop = hop if hop is not None else size
        self._next: int | None = None

    @property
    def spec(self) -> ChannelSpec:
        return self.buffer.spec

    def push(self, block: Block) -> list[Window]:
        self.buffer.push(block)
        if self._next is None:
            self._next = self.buffer.start
        out = []
        while True:
            # a discontinuity can leave _next behind the buffer; skip forward on the grid
            if self._next < self.buffer.start:
                behind = self.buffer.start - self._next
                self._next += ((behind + self.hop - 1) // self.hop) * self.hop
            w = self.buffer.take(self._next, self.size)
            if w is None:
                break
            out.append(w)
            self._next += self.hop
            self.buffer.discard_before(self._next)
        return out


class AlignedWindower:
    """Emits windows covering the same absolute index range across several channels.

    Cross-channel measures — coherence, phase synchrony, ratios between sites — are only
    meaningful on simultaneous samples, so this refuses to emit until every channel has the
    same span. All channels must share a sample rate; different rates would put them on
    different index grids and the comparison would be silently wrong.
    """

    def __init__(self, specs: dict[int, ChannelSpec], size: int, hop: int | None = None,
                 capacity_seconds: float = 60.0):
        rates = {s.rate_hz for s in specs.values()}
        if len(rates) > 1:
            raise ValueError(f"aligned windows need one sample rate, got {sorted(rates)}")
        self.buffers = {i: ChannelBuffer(s, capacity_seconds) for i, s in specs.items()}
        self.size = size
        self.hop = hop if hop is not None else size
        self._next: int | None = None

    def push(self, block: Block) -> list[dict[int, Window]]:
        buf = self.buffers.get(block.channel)
        if buf is None:
            return []
        buf.push(block)
        if self._next is None:
            self._next = max(b.start for b in self.buffers.values())
        out = []
        while True:
            low = max(b.start for b in self.buffers.values())
            if self._next < low:
                behind = low - self._next
                self._next += ((behind + self.hop - 1) // self.hop) * self.hop
            windows = {i: b.take(self._next, self.size) for i, b in self.buffers.items()}
            if any(w is None for w in windows.values()):
                break
            out.append(windows)
            self._next += self.hop
            for b in self.buffers.values():
                b.discard_before(self._next)
        return out
