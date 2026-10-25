"""The HEG instruments: two rolling traces, a headline, and the honesty line under them.

`heart_panel`'s counterpart, and built the same way — controls made once, mutated per frame. What it
draws is deliberately modest, because there is very little an fNIRS screen can honestly show. The
projection in `nfb.heg.processor` is not Beer-Lambert and its absolute scale is arbitrary, so a number
with units on it would be a claim nobody can cash. What is real is the *shape*: which way HbO is going,
whether HbR is moving against it, and how steady both are.

**Both series share one vertical scale.** They live at different offsets, so a per-series scale would
draw two traces that look equally large and hide the one thing worth seeing — HbR moving the other way,
by less. One scale keeps the comparison honest and costs some of each trace's height.

**The caption carries the things that explain a bad trace** rather than a legend nobody needs: what the
source is, its battery and firmware when it knows them, the quality word, and the share of frames held
for motion. That last figure is this protocol's equivalent of the EEG artifact rejection rate — a
motion-ridden stretch and a calm one look identical once the gate has held the channels flat, so the
number has to be beside the trace it applies to.
"""

from __future__ import annotations

from collections import deque

import flet as ft
import flet.canvas as cv

from nfb.heg import QUALITY_GOOD, QUALITY_MOTION

from . import theme as t
from .plots import grid, place_value_labels, polyline, value_axis

TRACE_SECONDS = 120.0
"""Two minutes across. The hemodynamic response runs over tens of seconds, so a window shorter than
this shows a slope with nothing to judge it against."""

AXIS_W = 46


def quality_word(quality: float) -> str:
    if quality >= QUALITY_GOOD:
        return "good"
    if quality <= QUALITY_MOTION:
        return "motion"
    return "settling"


class HEGPanel:
    """Build once, `push(state)` per frame and `refresh(session)` per draw.

    It takes the *session* rather than the runner, unlike `HeartPanel`: the headline here is the metric
    being trained and its current value, which is a property of what the operator chose rather than of
    the stream.
    """

    def __init__(self, trace_height: int = 200, width_of=lambda: 520.0):
        self.trace_h = trace_height
        self.width_of = width_of
        self.history: deque[tuple[float, float, float]] = deque()
        self._last_t: float | None = None

        self.headline = ft.Text("—", size=20, color=t.TEXT, weight=ft.FontWeight.W_300)
        self.trace_canvas = cv.Canvas(shapes=[], expand=True, height=self.trace_h)
        self.trace_axis, self._trace_labels = value_axis(self.trace_h, count=4, width=AXIS_W)
        self.trace_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.source_note = ft.Text("", size=t.SMALL, color=t.MUTED)

        self.numbers: dict[str, ft.Text] = {}
        tiles = ft.Row(spacing=10, wrap=True)
        for label, unit in (("HbO", "smoothed"), ("HbR", "smoothed"), ("steadiness", "over 5 s")):
            value = ft.Text("—", size=17, color=t.TEXT, weight=ft.FontWeight.W_300)
            self.numbers[label] = value
            tiles.controls.append(ft.Container(
                content=ft.Column([ft.Text(label, size=t.SMALL, color=t.FAINT), value,
                                   ft.Text(unit, size=t.SMALL, color=t.FAINT)], spacing=1),
                width=94, height=70, padding=ft.Padding(10, 9, 10, 9), border_radius=8,
                bgcolor=t.BG, border=ft.Border.all(1, t.BORDER)))
        self.tiles = tiles

    # --- the pieces a screen arranges --------------------------------------------------------

    def trace_rows(self) -> list[ft.Control]:
        return [
            self.headline,
            ft.Container(height=8),
            ft.Row([self.trace_axis,
                    ft.Container(content=self.trace_canvas, expand=True, height=self.trace_h)],
                   spacing=0),
            ft.Container(height=6), self.trace_caption,
        ]

    # --- per frame ---------------------------------------------------------------------------

    def push(self, state) -> None:
        """Record a point, if this is a frame we have not already drawn.

        The draw loop runs at 10 Hz and the device at 30, so this samples the stream rather than
        capturing it — which is the right trade for a two-minute window on a signal whose fastest
        component has already been low-passed away at 0.5 Hz. Keyed on the frame's own timestamp so a
        paused or stalled session does not lay repeats on top of each other.
        """
        if state is None or state.t == self._last_t:
            return
        self._last_t = state.t
        self.history.append((state.t, state.smoothed_hbo, state.smoothed_hbr))
        while self.history and state.t - self.history[0][0] > TRACE_SECONDS:
            self.history.popleft()

    def reset(self) -> None:
        self.history.clear()
        self._last_t = None

    def refresh(self, session) -> None:
        runner = session.heg
        self.source_note.value = runner.describe()
        self.source_note.color = t.BAD if runner.error else t.MUTED
        state = runner.state

        if state is None:
            self.headline.value = "no headband" if runner.error else "—"
            self.headline.color = t.BAD if runner.error else t.TEXT
            self.trace_caption.value = self._caption(session, None)
            self.trace_canvas.shapes = []
            for label in self.numbers.values():
                label.value = "—"
            for label in self._trace_labels:
                label.value = ""
            return

        self.numbers["HbO"].value = f"{state.smoothed_hbo:.0f}"
        self.numbers["HbR"].value = f"{state.smoothed_hbr:.0f}"
        self.numbers["steadiness"].value = f"{state.hbo_std_short:.1f}"
        self._headline(session, state)
        self._draw_traces()
        self.trace_caption.value = self._caption(session, state)

    def _headline(self, session, state) -> None:
        spec = session.metric_spec
        if session.error is not None or session.heg.stale:
            self.headline.value = "no headband"
            self.headline.color = t.BAD
            return
        if state.quality <= QUALITY_MOTION:
            # Named rather than left to a gap in the trace: during the cooldown the channels are held
            # flat, so the picture is *calm*, which is the opposite of what is happening to the person.
            self.headline.value = "motion — holding"
            self.headline.color = t.WARN
            return
        value = session.raw_metric
        if value is None:
            self.headline.value = "settling"
            self.headline.color = t.MUTED
            return
        unit = f" {spec.unit}" if spec.unit else ""
        self.headline.value = f"{spec.label} {value:.3g}{unit}"
        self.headline.color = t.TEXT

    def _draw_traces(self) -> None:
        if len(self.history) < 2:
            self.trace_canvas.shapes = []
            for label in self._trace_labels:
                label.value = ""
            return
        w, h = self.width_of(), self.trace_h
        newest = self.history[-1][0]
        values = [v for _, hbo, hbr in self.history for v in (hbo, hbr)]
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1e-9)

        def points(index: int) -> list[ft.Offset]:
            return [ft.Offset(w * (1.0 - (newest - point[0]) / TRACE_SECONDS),
                              h - (point[index] - lo) / span * (h - 2))
                    for point in self.history]

        self.trace_canvas.shapes = grid(w, h, rows=4, cols=8) + [
            polyline(points(1), t.ACCENT, 1.8),
            polyline(points(2), t.frequency_colour(3.0), 1.4),
        ]
        place_value_labels(self._trace_labels, h, hi, bottom=lo, unit="")

    def _caption(self, session, state) -> str:
        runner = session.heg
        parts = [f"{TRACE_SECONDS:g} s across · HbO in the brighter line, HbR under it, on one scale"]
        if state is not None:
            parts.append(quality_word(state.quality))
        else:
            parts.append("waiting for frames")
        gate = runner.gate_fraction
        if gate > 0:
            parts.append(f"{gate:.0%} held for motion")
        battery = runner.battery_pct
        if battery is not None:
            parts.append(f"battery {battery}%")
        firmware = runner.firmware_version
        if firmware:
            parts.append(f"fw {firmware}")
        choice = getattr(runner, "choice", None)
        if choice is not None:
            parts.append(choice.describe())
        return " · ".join(parts)
