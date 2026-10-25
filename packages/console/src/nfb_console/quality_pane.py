"""The quality timeline: where the session went bad, per channel, as one picture.

Shared rather than per-screen because the question belongs to two different jobs — tuning the artifact
policy on the Feedback tab, and reading a night's recording on the sleep screen — and both want the
same strip. See `quality.QualityLog` for what is kept and why rejected and unusable stay apart.
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from . import theme as t
from .quality import CELL_SECONDS
from .widgets import panel

ROW_H = 14
GAP = 3

TIMELINE_HELP = (
    "This shows every second of the session, per channel, left to right. A percentage tells you how "
    "much was lost. This strip shows where it was lost.\n\n"
    "Amber means the artifact gate is rejecting: the signal arrives, and the gate throws some of it "
    "away. Red means the channel is unusable: railed, flat, or turned off by you. This is a different "
    "failure with a different fix, so the two colours never mix. Grey is a gap where no samples "
    "arrived.\n\n"
    "Each pixel takes the colour of its worst second, not the average. This way, a five-second dropout "
    "still shows, even when the strip spans two hours.\n\n"
    "On a replayed recording, the timeline is identical every run. Use it to judge an artifact setting: "
    "change the tolerance, replay the same night, and compare the strips."
)


class QualityPane:
    """Build once, mutate on Flet's loop. `refresh()` per frame; redraws only when a cell lands."""

    def __init__(self, page: ft.Page, session, width: Callable[[int], float],
                 titled: bool = True):
        self.page = page
        self.session = session
        self.width = width
        self._drawn: tuple = ()

        self.image = ft.Image(src=None, fit=ft.BoxFit.FILL, filter_quality=ft.FilterQuality.NONE)
        self.empty = ft.Text("nothing recorded yet — the first second lands in a moment",
                             size=t.SMALL, color=t.FAINT)
        self.labels = ft.Column(spacing=GAP)
        self.caption = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.legend = ft.Row([
            _key(t.WARN, "rejected"), _key(t.BAD, "unusable"), _key(t.BORDER, "no data"),
        ], spacing=14)

        body = ft.Column([
            ft.Row([self.labels, ft.Container(width=8),
                    ft.Container(content=ft.Stack([self.image, self.empty]), expand=True)],
                   spacing=0, vertical_alignment=ft.CrossAxisAlignment.START),
            ft.Container(height=8),
            ft.Row([self.legend, ft.Container(expand=True)]),
            ft.Container(height=4), self.caption,
        ], spacing=0)
        # Titled where it stands alone; bare where it sits inside a panel that already has a heading —
        # the ARTIFACTS panel is one, and a panel inside a panel reads as a boxed-in mistake.
        self.control = (panel("QUALITY", body, help=_help(page)) if titled
                        else ft.Column([body], spacing=0))

    def refresh(self) -> None:
        log = self.session.quality
        channels = [c for c in self.session.specs if c in log.channels()]
        width = int(self.width(90))
        # Rebuilt when a cell lands, not per frame — what makes this affordable at 10 Hz.
        key = (tuple(int(log.seconds(c)) for c in channels), width)
        if key == self._drawn:
            return
        self._drawn = key
        if len(self.labels.controls) != len(channels):
            self.labels.controls = [
                ft.Container(content=ft.Text(f"ch{c}", size=t.SMALL, color=t.MUTED),
                             height=ROW_H, width=32,
                             alignment=ft.Alignment.CENTER_RIGHT) for c in channels]
        src = log.image(channels, width, row_height=ROW_H, gap=GAP) if channels else None
        self.image.src = src
        self.image.visible = src is not None
        self.image.width = width
        self.image.height = len(channels) * ROW_H + max(0, len(channels) - 1) * GAP
        self.empty.visible = src is None
        if src is None:
            self.caption.value = ""
            return
        span = max((log.seconds(c) for c in channels), default=0.0)
        per_pixel = span / max(1, width)
        worst = ""
        marks = [(c, log.worst_minute(c)) for c in channels]
        marks = [(c, at, bad) for c, m in marks if m for at, bad in (m,) if bad > 0.05]
        if marks:
            c, at, bad = max(marks, key=lambda m: m[2])
            worst = f" · worst minute ch{c} at {_clock(at)}, {bad:.0%} bad"
        self.caption.value = (f"{_duration(span)} · one pixel = {_duration(per_pixel)}, coloured by "
                              f"its worst {CELL_SECONDS:g} s{worst}")


def _key(colour: str, label: str) -> ft.Control:
    return ft.Row([ft.Container(width=10, height=10, bgcolor=colour, border_radius=2),
                   ft.Text(label, size=t.SMALL, color=t.MUTED)], spacing=6)


def _help(page: ft.Page) -> ft.Control:
    from .widgets import help_button

    return help_button(page, "The quality timeline", TIMELINE_HELP)


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 3600:d}:{int(seconds) % 3600 // 60:02d}:{int(seconds) % 60:02d}"


def _duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 90:
        return f"{seconds / 60:.0f} min"
    if seconds >= 1:
        return f"{seconds:.0f} s"
    return f"{seconds * 1000:.0f} ms"
