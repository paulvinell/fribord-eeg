"""The settings menu: choices that apply to every program, reached from the launcher's toolbar.

A dialog rather than a screen, like the record options, because these are decided rarely and there is
nothing on them to watch. Today it holds one section, the developer settings, and the section is closed
by default: with the master switch off, its rows are not on the dialog at all, so it holds one choice and
not three. The master switch's own label heads the section, so there is no separate heading to say
"developer" twice.

Every change is applied and stored as it is made — there is no "apply" — because each one takes effect
at once: the outputs and the logs are read on every call (see `developer`), and the launcher re-lists
its programs through `on_change`.
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from . import developer
from . import theme as t
from .widgets import labelled, pill

LABEL_W = 130


def show(page: ft.Page, on_change: Callable[[], None]) -> None:
    """Open the dialog. `on_change` is called after anything is changed, to store it and repaint."""
    if getattr(page, "show_dialog", None) is None:
        return

    # Tight, so the dialog is as tall as its rows: a loose column fills the window from top to bottom.
    body = ft.Column(spacing=14, width=520, tight=True)
    s = developer.settings

    def choose(attr: str, value: bool) -> None:
        setattr(s, attr, value)
        on_change()
        rebuild()

    def on_off(attr: str, invert: bool = False, locked: bool | None = None) -> list[ft.Control]:
        """The `on` and `off` pills for one setting. `invert` is for a setting stored as its opposite:
        the row says "outputs on", the field is `outputs_off`. `locked` is what an environment variable
        decided, shown inert, so the row still says which way it went."""
        lit = (getattr(s, attr) != invert) if locked is None else locked
        free = locked is None
        return [pill("on", lit, lambda _e: choose(attr, not invert), enabled=free),
                pill("off", not lit, lambda _e: choose(attr, invert), enabled=free)]

    def rebuild() -> None:
        rows: list[ft.Control] = [labelled("developer settings", *on_off("on"), width=LABEL_W)]
        if s.on:
            # The variables win, and an inert pill with the variable named beside it is the row saying
            # so. See `developer` for why the menu cannot turn outputs back on under `NFB_NO_OUTPUTS`.
            debug_set = developer.debug_by_variable()
            forced = developer.outputs_off_by() == developer.OUTPUTS_VARIABLE
            rows += [
                labelled("debug log", *on_off("debug", locked=True if debug_set else None),
                         width=LABEL_W,
                         note=(f"{developer.DEBUG_VARIABLE} is set" if debug_set
                               else "the console writes its logs to /tmp")),
                labelled("outputs", *on_off("outputs_off", invert=True,
                                            locked=False if forced else None),
                         width=LABEL_W,
                         note=(f"{developer.OUTPUTS_VARIABLE} is set" if forced
                               else "the shadow, the volume, the sounds and the sleep hold")),
            ]
        body.controls = rows
        try:
            page.update()
        except Exception:
            pass

    rebuild()
    page.show_dialog(ft.AlertDialog(
        title=ft.Text("Settings", size=t.H2, color=t.TEXT, weight=ft.FontWeight.W_500),
        content=ft.Container(width=520, content=body),
        bgcolor=t.SURFACE,
        actions=[ft.Container(content=ft.Text("close", size=t.BODY, color=t.ACCENT),
                              padding=ft.Padding(12, 7, 12, 7), border_radius=6,
                              on_click=lambda _e: page.pop_dialog())]))
