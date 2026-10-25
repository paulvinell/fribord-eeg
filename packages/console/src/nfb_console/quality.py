"""Where the session went bad, kept per channel for as long as it runs.

Everything else about signal quality describes the instant — the trace goes amber while the gate is
firing, the dot goes red while a channel is condemned, the captions carry running percentages. None of
it survives the ten seconds of the raw window, so "1.2 h of the night was lost" could never be followed
by *when*. That is the question a rejection policy is actually tuned against, and on a replay the whole
timeline is deterministic, so a proposed threshold can be judged against a night that really happened.

**Rejected and unusable are kept apart rather than blended into one "bad".** They fail differently and
they are tuned by different knobs: rejection is the artifact gate's judgement about a channel that is
delivering, and unusable is the channel verdict — railed, flat, or condemned by the operator. A channel
known unusable bypasses the gate entirely (`ArtifactGates.pass_through`), so the two are nearly
exclusive by construction, and a display that mixed them would hide which of the two knobs to reach for.

One cell per `CELL_SECONDS` of **signal time**, so a replay reproduces a live session exactly.
"""

from __future__ import annotations

import numpy as np

from . import theme as t

CELL_SECONDS = 1.0
"""How much signal each cell covers. A second is the artifact gate's own natural grain — its rate is a
5 s average and its judgements are per block — and a night is then 28800 cells per channel, which is
nothing to hold and downsamples cleanly to any width a screen has."""

LIMIT_CELLS = int(14 * 3600 / CELL_SECONDS)
"""Fourteen hours, so a long night cannot silently lose its beginning."""

CLEAN = "#cfd8cd"
"""What a good cell looks like: quiet, closer to the panel than to a signal colour. A timeline is read
for its *marks*, so the unmarked part must not compete with them — a bright green field of health is
the same mistake as a graph with no empty state, in the other direction."""


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


class QualityLog:
    """Per channel, per second: how much was rejected and how much was unusable.

    Fed from the feed thread with the gate's own verdict, which is already computed there — nothing
    here re-judges anything, it only remembers what was decided and when.
    """

    def __init__(self) -> None:
        self._cells: dict[int, list[tuple[float, float]]] = {}
        self._acc: dict[int, list[float]] = {}          # channel -> [slot, total, rejected, unusable]

    def observe(self, channel: int, at_seconds: float, samples: int, rejected: int,
                unusable: bool) -> None:
        if samples <= 0:
            return
        slot = int(at_seconds / CELL_SECONDS)
        acc = self._acc.get(channel)
        if acc is None or slot != int(acc[0]):
            if acc is not None:
                self._flush(channel, acc, slot)
            acc = [float(slot), 0.0, 0.0, 0.0]
            self._acc[channel] = acc
        acc[1] += samples
        acc[2] += rejected
        acc[3] += samples if unusable else 0

    def _flush(self, channel: int, acc: list[float], slot: int) -> None:
        cells = self._cells.setdefault(channel, [])
        total = max(acc[1], 1.0)
        cells.append((acc[2] / total, acc[3] / total))
        # A slot the feed skipped is a hole in the *recording*, not a clean second — an acquisition gap
        # or a stalled feed leaves no blocks at all, and drawing it as healthy would hide it.
        for _ in range(min(int(slot - acc[0]) - 1, LIMIT_CELLS)):
            cells.append((float("nan"), float("nan")))
        del cells[:-LIMIT_CELLS]

    def channels(self) -> list[int]:
        return sorted(set(self._cells) | set(self._acc))

    def series(self, channel: int) -> list[tuple[float, float]]:
        """Finished cells, plus the one in progress so a fresh session is not blank for a second."""
        out = list(self._cells.get(channel, ()))
        acc = self._acc.get(channel)
        if acc is not None and acc[1] > 0:
            out.append((acc[2] / acc[1], acc[3] / acc[1]))
        return out

    def seconds(self, channel: int) -> float:
        return len(self.series(channel)) * CELL_SECONDS

    def totals(self, channel: int, span_seconds: float = 0.0) -> tuple[float, float, float]:
        """`(rejected, unusable, missing)` as fractions of the span — the whole log for 0."""
        cells = self.series(channel)
        if span_seconds:
            cells = cells[-int(span_seconds / CELL_SECONDS):]
        if not cells:
            return 0.0, 0.0, 0.0
        arr = np.asarray(cells, dtype=float)
        missing = np.isnan(arr[:, 0])
        n = len(arr)
        good = ~missing
        return (float(arr[good, 0].sum() / n) if good.any() else 0.0,
                float(arr[good, 1].sum() / n) if good.any() else 0.0,
                float(missing.sum() / n))

    def worst_minute(self, channel: int) -> tuple[float, float] | None:
        """`(at_seconds, bad_fraction)` for the worst 60 s — where to look first, as a number the
        health panel can print beside the picture."""
        cells = self.series(channel)
        if len(cells) < 2:
            return None
        arr = np.nan_to_num(np.asarray(cells, dtype=float), nan=1.0)
        bad = arr[:, 0] + arr[:, 1]
        width = min(len(bad), int(60.0 / CELL_SECONDS))
        window = np.convolve(bad, np.ones(width), mode="valid") / width
        at = int(np.argmax(window))
        return at * CELL_SECONDS, float(window[at])

    def image(self, channels: list[int], width: int, row_height: int = 14,
              gap: int = 3) -> bytes | None:
        """The timeline as a PNG: one band per channel, time left to right.

        An image rather than shapes because that is what makes a whole night affordable — 28800 cells
        per channel would be 28800 canvas objects serialised on every frame, against one `src` here.
        """
        from .plots import png_bytes

        rows = [self.series(c) for c in channels]
        if not rows or not any(rows):
            return None
        width = max(16, int(width))
        height = len(rows) * row_height + max(0, len(rows) - 1) * gap
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = _rgb(t.SURFACE)
        for i, cells in enumerate(rows):
            band = self._band(cells, width)
            top = i * (row_height + gap)
            frame[top:top + row_height] = band
        return png_bytes(frame)

    @staticmethod
    def _band(cells: list[tuple[float, float]], width: int) -> np.ndarray:
        """One channel's row of pixels, `width` wide.

        Downsampled by the **worst** cell in each pixel rather than the mean: a timeline is read to find
        trouble, and a five-second dropout inside a two-minute pixel has to survive being averaged with
        its neighbours or the picture cannot be trusted at any zoom.
        """
        out = np.zeros((width, 3), dtype=float)
        out[:] = _rgb(t.BG)
        if not cells:
            return np.repeat(out[None, :, :], 1, axis=0).astype(np.uint8)
        arr = np.asarray(cells, dtype=float)
        n = len(arr)
        edges = (np.arange(width + 1) * n / width).astype(int)
        for x in range(width):
            lo, hi = edges[x], max(edges[x + 1], edges[x] + 1)
            if lo >= n:
                break
            chunk = arr[lo:min(hi, n)]
            missing = np.isnan(chunk[:, 0])
            if missing.all():
                out[x] = _rgb(t.BORDER)
                continue
            rejected = float(np.nanmax(chunk[:, 0]))
            unusable = float(np.nanmax(chunk[:, 1]))
            if missing.any():
                unusable = max(unusable, float(missing.mean()))
            if unusable > 0.5:
                out[x] = _rgb(t.BAD)
            elif unusable > 0:
                out[x] = _rgb(CLEAN) + (_rgb(t.BAD) - _rgb(CLEAN)) * unusable
            elif rejected > 0:
                # Floored well above nothing: a single rejected block in a second is 12% of it, and the
                # whole point is to see where the gate fired at all.
                weight = min(1.0, 0.35 + 0.65 * rejected)
                out[x] = _rgb(CLEAN) + (_rgb(t.WARN) - _rgb(CLEAN)) * weight
            else:
                out[x] = _rgb(CLEAN)
        return np.repeat(out[None, :, :], 1, axis=0).astype(np.uint8)
