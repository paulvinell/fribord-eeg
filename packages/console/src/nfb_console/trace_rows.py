"""The RAW panel: one row per channel, shared by every protocol screen.

Five screens each built this out of the same parts — a state dot, a channel label, a min/max envelope on a
canvas, a caption saying what the window and the scale are — and each drew it with its own copy of the same
dozen lines. The copies had drifted, which is the argument for this module rather than a tidiness one:

- **Only the infra-low screen let you overrule the device about a channel**, though `TrainingSession.trusted`
  and `trust_channel` have always been shared and every screen has the same problem. The amplifier's
  below-range detector fires on a perfectly good channel — 20 µV of EEG is 2.5% of the narrowest range it
  offers — and the only thing that can adjudicate that is somebody looking at the waveform. That person is
  looking at *this* row, on whichever screen they happen to be on, so the override belongs to the row.
- **Only that screen showed the standing offset**, which is the other thing that kills a channel and the one
  the picture cannot show: the trace is drawn high-passed so it stays readable at the 0.01 Hz corner, which
  removes the very offset that rails the input. Contact impedance does not predict it — 14 and 26 kOhm beside
  a 311 µV offset on one capture — so a screen without it is a screen where a dying channel looks fine.
- Three screens coloured a not-yet-heard-from channel as dead, one as unknown. Unknown is right: `None` means
  no block has arrived, which is neither good nor bad.

What stays with each screen is what each screen actually knows: how many seconds its window is, and any
clause it wants to add to the caption ("500 ms/div", "in as it crosses, out as it returns"). Those are passed
in per refresh rather than configured once, because a screen whose window is an operator setting changes it
while running.
"""

from __future__ import annotations

from typing import Callable, Sequence

import flet as ft
import flet.canvas as cv

from . import theme as t
from .plots import envelope_shapes, mark_shapes, polyline

TRACE_H = 92
LABEL_W = 64
HEADROOM_VISIBLE_UV = 200.0
"""How close to the rail is worth saying, in µV. Below this the channel is in real danger of railing and the
operator has to know; above it, printing the number every frame would be noise on a healthy channel."""

FORCED_HELP = (
    "Click a channel to set it yourself: the amplifier's own word, forced on, or forced off.\n\n"
    "The amplifier can report a good channel as dead. 20 µV of EEG is only 2.5% of its narrowest range, "
    "so the amplifier cannot always tell a live channel from a dead one. The waveform beside it can.\n\n"
    "Force a channel on to measure it no matter what the amplifier reports. Force a channel off to "
    "remove it from every measurement, for an electrode that is off the head."
)


class TraceRows:
    """Build once, mutate on Flet's loop — `rows` into the layout, `refresh()` per frame.

    `width` is a callable rather than a number because the canvas is `expand` while the shapes inside it are
    placed in absolute coordinates, so the drawing code needs the post-layout width that Flet does not
    expose; every screen derives it from the window and its own reserved columns. See `plot_width` in any of
    them.
    """

    def __init__(self, page: ft.Page, session, width: Callable[[], float],
                 height: int = TRACE_H, notes: bool = False):
        self.page = page
        self.session = session
        self.width = width
        self.height = height

        self.canvas = {i: cv.Canvas(shapes=[], expand=True, height=height) for i in session.specs}
        self.dots = {i: ft.Container(width=7, height=7, border_radius=4, bgcolor=t.BORDER)
                     for i in session.specs}
        self.labels = {i: ft.Text(f"ch{i}", size=t.SMALL, color=t.MUTED) for i in session.specs}
        self.caption = ft.Text("", size=t.SMALL, color=t.FAINT)

        # A channel the device calls unusable draws a flat grey line, which is indistinguishable from a
        # broken app — the "explicit empty states" rule. The words go *over* the trace, because that is what
        # is being explained. Off by default: only the screens with room for it ask.
        self.notes = ({i: ft.Text("", size=t.SMALL, color=t.WARN) for i in session.specs}
                      if notes else {})

        self.rows = [ft.Row([
            ft.Container(width=LABEL_W, on_click=self._toggle(i), tooltip=FORCED_HELP,
                         content=ft.Row([self.dots[i], self.labels[i]], spacing=7)),
            ft.Container(content=self._plot(i), expand=True, height=height)],
            spacing=8) for i in session.specs]

    def _plot(self, channel: int) -> ft.Control:
        if channel not in self.notes:
            return self.canvas[channel]
        return ft.Stack([ft.Container(content=self.canvas[channel], expand=True),
                         ft.Container(content=self.notes[channel],
                                      alignment=ft.Alignment.CENTER, expand=True)],
                        expand=True, height=self.height)

    def _toggle(self, channel: int):
        """**Click the channel to overrule the device about it.** Three states, cycled: the device's word,
        forced live, forced dead.

        Here rather than in a settings pane because the evidence is here — see the module docstring and
        `TrainingSession.trust_channel`.
        """
        def handler(_e=None) -> None:
            nxt = {None: True, True: False, False: None}[self.session.trusted.get(channel)]
            self.session.trust_channel(channel, nxt)
            self.session.redraw()
        return handler

    def refresh(self, window_seconds: float, extra: Sequence[str] = ()) -> None:
        """Draw every row and rebuild the caption. `extra` clauses go after the scale and before the
        diagnostics, which is where a screen's own note about its window belongs."""
        session = self.session
        width = self.width()
        mid = self.height / 2
        for i, buf in session.traces.items():
            if i not in self.canvas:
                continue                # a channel the toolbar trimmed after these rows were built
            lo, hi = buf.envelope()
            ok = session.state.get(i)
            rejecting = session.rejecting(i)
            self.canvas[i].shapes = [
                polyline([ft.Offset(0, mid), ft.Offset(width, mid)], t.BORDER, 0.6),
                *envelope_shapes(lo, hi, width, self.height, session.gain_uv,
                                 t.WARN if rejecting else (t.ACCENT if ok else t.BORDER)),
                # *which* part was thrown away — the trace colour above only says the rate is up
                *mark_shapes(buf.marks(), width, self.height, t.WARN),
            ]
            # `None` is its own colour: not heard from yet is neither good nor bad, and three of the five
            # screens used to paint it as dead — which reads as a broken electrode for the first second of
            # every session.
            self.dots[i].bgcolor = (t.BORDER if ok is None else
                                    t.BAD if not ok else t.WARN if rejecting else t.GOOD)
            # The label carries the override, because the dot only has colour and colour already means
            # live/rejecting/dead. A forced channel has to look forced or the operator cannot tell their own
            # decision from the amplifier's.
            forced = session.trusted.get(i)
            self.labels[i].value = f"ch{i}" + ("" if forced is None else (" ✓" if forced else " ✕"))
            self.labels[i].color = t.MUTED if forced is None else t.ACCENT
            # Three states in words, because `ok` is None until the channel has been heard from and saying
            # "no signal" then is a different claim from the device's.
            if i in self.notes:
                self.notes[i].value = ("waiting for data" if ok is None else
                                       "" if ok else "no signal — check the electrode")
                self.notes[i].color = t.FAINT if ok is None else t.WARN

        hot = [f"ch{i} {session.rejection(i):.0%}" for i in session.traces if session.rejecting(i)]
        forced = [f"ch{c} forced {'on' if v else 'off'}" for c, v in sorted(session.trusted.items())]
        # **The offset, because the trace cannot show it and it is what kills a channel.** The picture is
        # drawn high-passed so it stays readable at the 0.01 Hz corner, which also removes the standing
        # offset from view — and the offset is what rails a channel and makes the device report no signal.
        # Shown as the closest approach to the rail, since that is the quantity that decides whether the
        # channel survives, and only when it is worth acting on.
        tight = [(c, room) for c in sorted(session.specs)
                 if (room := session.headroom_uv(c)) is not None and room < HEADROOM_VISIBLE_UV]
        self.caption.value = (
            f"{window_seconds:g} s across · ±{session.gain_uv:g} µV full scale · "
            f"drawn {session.DISPLAY_FLOOR_HZ:g}–{session.DISPLAY_LIMIT_HZ:g} Hz"
            + "".join(f" · {clause}" for clause in extra if clause)
            + (" · " + ", ".join(forced) if forced else "")
            + (f" · rejecting {', '.join(hot)}" if hot else "")
            + (" · " + ", ".join(f"ch{c} within {room:.0f} µV of the rail" for c, room in tight)
               if tight else ""))
        self.caption.color = t.WARN if (hot or tight) else t.FAINT
