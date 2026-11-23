"""The standing level of each input over the last minutes, against the input's own limits.

The RAW panel draws a band from 0.5 Hz, so it cannot show the one thing that takes an input past its range:
the standing offset of the electrode, which drifts for minutes after it is put on. This plot draws the mean
of each half second of samples, on an axis that spans the input range, with the two limits as lines. A
curve that settles away from a line is an electrode that is ready; one that sits on a line is outside the
range. A source with no input range has no lines, and the axis scales to the data.

Two parts, for the build-once rule: `LevelHistory` is fed on the feed thread with every raw block, and
`LevelPlot` reads it on Flet's loop. Several plots can read one history, since a control can be in one place
only.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb.dsp.artifact import INPUT_LIMIT_FRACTION

from . import theme as t
from .plots import dashed, filled, nice_ceiling, place_value_labels, polyline, value_axis

LEVEL_SECONDS = 180.0
"""The window: an electrode's offset settles over minutes, so seconds would show only a flat line."""

STEP_S = 0.5
"""One point per half second of samples: the standing level, with the EEG averaged out."""

AXIS_W = 52
HEADROOM = 1.1
BAND_OPACITY = 0.3


def channel_colour(channel: int) -> str:
    return t.CHANNEL_COLOURS[channel % len(t.CHANNEL_COLOURS)]


class LevelHistory:
    """Per channel, the mean, the lowest and the highest of each `STEP_S` of raw samples in µV, at the signal
    time of its end. A step back in time starts the channel again. The band the plot draws and the answer to
    `touched` come from the same lowest and highest samples, so the plot shows what a line beside it says."""

    def __init__(self, seconds: float = LEVEL_SECONDS):
        self.seconds = seconds
        self._lock = threading.Lock()
        self._points: dict[int, deque] = {}
        self._sum: dict[int, tuple[float, int]] = {}
        self._end: dict[int, float] = {}
        self.ranges: dict[int, float | None] = {}

    def push(self, block) -> None:
        values = np.asarray(block.values, dtype=float)
        if not len(values):
            return
        ch, rate = block.channel, block.spec.rate_hz
        end = block.end_index / rate
        with self._lock:
            self.ranges[ch] = block.spec.input_range
            points = self._points.setdefault(ch, deque(maxlen=int(self.seconds / STEP_S) + 2))
            if ch in self._end and end < self._end[ch]:
                points.clear()
                self._sum.pop(ch, None)
            self._end[ch] = end
            total, n, lo, hi = self._sum.get(ch, (0.0, 0, math.inf, -math.inf))
            total, n = total + float(values.sum()), n + len(values)
            lo, hi = min(lo, float(values.min())), max(hi, float(values.max()))
            if n >= STEP_S * rate or not points:
                points.append((end, total / n, lo, hi))
                total, n, lo, hi = 0.0, 0, math.inf, -math.inf
            self._sum[ch] = (total, n, lo, hi)

    def series(self) -> dict[int, list[tuple[float, float, float, float]]]:
        """Per channel, `(end, mean, lowest, highest)` of each step, oldest first."""
        with self._lock:
            return {ch: list(points) for ch, points in self._points.items()}

    def touched(self, channel: int, seconds: float) -> bool:
        """Whether a sample of this channel reached its input limit in the steps that end in its last
        `seconds`: the band touches a limit line there."""
        with self._lock:
            lim = limit_uv(self.ranges.get(channel))
            points = self._points.get(channel)
            if lim is None or not points:
                return False
            newest = points[-1][0]
            return any(lo <= -lim or hi >= lim for at, _mean, lo, hi in points if newest - at < seconds)

    def latest(self) -> float | None:
        with self._lock:
            return max(self._end.values()) if self._end else None


def limit_uv(input_range: float | None) -> float | None:
    """Where a sample is at the input limit, in µV: the same rule the gate and the session use."""
    return INPUT_LIMIT_FRACTION * input_range if input_range else None


class LevelPlot:
    """The plot, its axis, one name per curve and a caption that states the window and the scale. Its size
    is fixed, so nothing in it moves the panel around it."""

    def __init__(self, history: LevelHistory, channels: Callable[[], list[int]], width_of: Callable[[], float],
                 height: int = 150):
        self.history = history
        self.channels = channels
        self.width_of = width_of
        self.height = height
        self.canvas = cv.Canvas(shapes=[], expand=True, height=height)
        self.axis, self.labels = value_axis(height, count=3, width=AXIS_W)
        # The limits are named on their own lines at the left, clear of the newest points at the right.
        self.limit_labels = [ft.Text("", size=t.SMALL, color=t.WARN, left=4, top=0, visible=False)
                             for _ in range(2)]
        self.legend = ft.Row(spacing=12, height=16)
        self.caption = ft.Text("", size=t.SMALL, color=t.FAINT, max_lines=1, no_wrap=True)
        self.empty = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.control = ft.Column([
            ft.Row([self.axis, ft.Container(content=ft.Stack(
                [ft.Container(content=self.canvas, expand=True),
                 ft.Container(content=self.empty, alignment=ft.Alignment.CENTER, expand=True),
                 *self.limit_labels],
                expand=True, height=height), expand=True, height=height)], spacing=0),
            ft.Container(height=6), self.legend, ft.Container(height=4), self.caption], spacing=0)

    def refresh(self) -> None:
        w, h = max(40.0, self.width_of() - AXIS_W), float(self.height)
        series = self.history.series()
        shown = [ch for ch in self.channels() if series.get(ch)]
        ranges = [self.history.ranges.get(ch) for ch in shown]
        limits = sorted({lim for r in ranges if (lim := limit_uv(r))})
        if limits and all(ranges):
            # A tenth past the range, so the limit lines sit inside the plot and not on its edge.
            top = HEADROOM * max(ranges)
            scale = f"input range ±{max(ranges):g} µV"
        else:
            peak = max((max(abs(lo), abs(hi)) for ch in shown for _t, _v, lo, hi in series[ch]), default=0.0)
            top = nice_ceiling(max(peak * 1.15, 1.0))
            scale = "this plot's own"
        place_value_labels(self.labels, h, top, bottom=-top, unit="µV")
        for label, value in zip(self.labels, (top, 0.0, -top)):
            label.value = f"{value:.0f} µV"                                      # one form for all three
        y_of = lambda v: h / 2 - (max(-top, min(top, v)) / top) * (h / 2)        # noqa: E731
        shapes = [polyline([ft.Offset(0, h / 2), ft.Offset(w, h / 2)], t.BORDER, 0.6)]
        for lim in limits:
            for sign in (1, -1):
                shapes += dashed(y_of(sign * lim), w, t.WARN, 1.0)
        newest = self.history.latest()
        seconds = self.history.seconds
        for ch in shown:
            kept = [p for p in series[ch] if newest - p[0] <= seconds]
            xs = [w * (1 - (newest - at) / seconds) for at, *_rest in kept]
            if len(kept) == 1:
                kept, xs = kept * 2, [xs[0] - 2, xs[0]]                            # one reading is a short dash
            colour = channel_colour(ch)
            # The band from the lowest to the highest sample of each step, so a spike the mean hides shows.
            shapes.append(filled([ft.Offset(x, y_of(p[3])) for x, p in zip(xs, kept)]
                                 + [ft.Offset(x, y_of(p[2])) for x, p in zip(reversed(xs), reversed(kept))],
                                 ft.Colors.with_opacity(BAND_OPACITY, colour)))
            shapes.append(polyline([ft.Offset(x, y_of(p[1])) for x, p in zip(xs, kept)], colour, 1.6))
        self.canvas.shapes = shapes
        self.legend.controls = [
            ft.Row([ft.Container(width=10, height=3, bgcolor=channel_colour(ch), border_radius=2),
                    ft.Text(f"ch{ch}", size=t.SMALL, color=t.MUTED)], spacing=6) for ch in shown]
        self.empty.value = "" if shown else "no readings yet"
        outer = limits[-1] if limits else None
        for label, sign in zip(self.limit_labels, (1, -1)):
            label.visible = outer is not None
            if outer is not None:
                label.value = f"limit {sign * outer:+.0f} µV"
                label.top = y_of(sign * outer) + (2 if sign > 0 else -16)
        self.caption.value = f"{seconds / 60:g} min across · ±{top:g} µV full scale · {scale}"
