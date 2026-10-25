"""Composing metrics over a source.

Deliberately thin: it dispatches blocks to metrics, forwards gaps, and reports the pipeline's
worst-case latency. The reason it exists at all is that latency should be answerable without
reading every metric's source, and gaps should reach metrics rather than being swallowed by
whoever is iterating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from ..stream import Block, Gap, SampleSource
from .metrics import Metric, Reading


@dataclass
class Pipeline:
    """Feeds a source's blocks through a set of metrics."""

    metrics: list[Metric] = field(default_factory=list)
    gaps: int = 0

    def add(self, *metrics: Metric) -> "Pipeline":
        self.metrics.extend(metrics)
        return self

    @property
    def delay_seconds(self) -> float:
        """Worst irreducible lag across metrics — how fast this could possibly respond.

        Add the transport's own delay for an end-to-end figure: the FTDI VCP driver on macOS
        buffers about 16 ms before handing bytes over.
        """
        return max((m.delay_seconds for m in self.metrics), default=0.0)

    @property
    def smoothing_seconds(self) -> float:
        """Worst averaging constant across metrics. A choice, not a cost."""
        return max((m.smoothing_seconds for m in self.metrics), default=0.0)

    @property
    def latency_seconds(self) -> float:
        """Conservative upper bound: delay plus smoothing. See delay_seconds first."""
        return max((m.latency_seconds for m in self.metrics), default=0.0)

    def feed(self, item: Block | Gap) -> list[Reading]:
        if isinstance(item, Gap):
            self.gaps += 1
            return []
        out = []
        for m in self.metrics:
            out.extend(m.feed(item))
        return out

    def run(self, source: SampleSource) -> Iterator[Reading]:
        for item in source.blocks():
            yield from self.feed(item)
