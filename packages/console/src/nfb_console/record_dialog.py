"""Where a recording goes and in what format — the caret half of the toolbar's record button.

A dialog rather than more chrome, because these are decided rarely and read never: the toolbar already
carries the one thing worth watching, which is whether it is writing. What belongs behind the caret is
the folder, the name and the format, together with the *resolved path*, since the only question these
three settings really answer is "what file am I about to get".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import flet as ft

from nfb.recorder import SUFFIXES, timestamped_path, unique_path

from . import theme as t
from .widgets import pill

FORMATS = (
    ("native", "nfb", "raw frames, replayable here, keeps signal states and gaps"),
    ("EDF", "edf", "decoded samples other tools read — EDFbrowser, MNE, sleep scorers"),
)


@dataclass
class RecordSettings:
    """The operator's standing answer to where recordings go. Persisted; see `prefs`."""

    directory: str = "sessions"
    name: str = ""
    """A stem, or empty for a timestamp. Kept as typed: a name that is reused gets `-2` appended when
    the file is opened, rather than being refused or silently overwriting a night."""
    fmt: str = "nfb"

    def resolve(self) -> Path:
        return unique_path(timestamped_path(self.directory or "sessions", self.fmt, self.name))


def show(page: ft.Page, settings: RecordSettings, on_change: Callable[[], None],
         picker: ft.FilePicker | None = None) -> None:
    """Open the dialog. `on_change` is called after anything is edited, to persist and repaint."""
    if getattr(page, "show_dialog", None) is None:
        return

    body = ft.Column(spacing=14, width=520)
    folder = ft.TextField(value=settings.directory, height=34, text_size=t.BODY, color=t.TEXT,
                          bgcolor=t.SURFACE, border_color=t.BORDER, expand=True,
                          content_padding=ft.Padding(10, 4, 10, 4))
    name = ft.TextField(value=settings.name, hint_text="leave empty for a timestamp", height=34,
                        text_size=t.BODY, color=t.TEXT, bgcolor=t.SURFACE, border_color=t.BORDER,
                        expand=True, content_padding=ft.Padding(10, 4, 10, 4))
    formats = ft.Row(spacing=6)
    preview = ft.Text("", size=t.SMALL, color=t.MUTED, selectable=True)
    note = ft.Text("", size=t.SMALL, color=t.FAINT)

    def commit(_e=None) -> None:
        settings.directory = str(folder.value or "").strip() or "sessions"
        settings.name = str(name.value or "").strip()
        on_change()
        rebuild()

    def set_format(code: str) -> None:
        settings.fmt = code
        on_change()
        rebuild()

    def browse(_e) -> None:
        async def run() -> None:
            try:
                chosen = await picker.get_directory_path(dialog_title="Recordings folder")
            except Exception:
                return
            if chosen:
                folder.value = chosen
                commit()
        page.run_task(run)

    def rebuild() -> None:
        formats.controls = [
            pill(label, settings.fmt == code, (lambda c=code: lambda _e: set_format(c))())
            for label, code, _why in FORMATS]
        note.value = next(why for _l, code, why in FORMATS if code == settings.fmt)
        # The resolved path, because three separate fields do not answer the only question being
        # asked here — and it is the *next* file, clash suffix included.
        preview.value = f"next recording → {settings.resolve()}"
        body.controls = [
            ft.Text("Recording", size=t.BODY, color=t.MUTED),
            _row("folder", folder,
                 *([_button("browse…", browse)] if picker is not None else [])),
            _row("name", name, ft.Text(SUFFIXES.get(settings.fmt, ""), size=t.SMALL,
                                       color=t.FAINT)),
            _row("format", formats),
            ft.Container(content=note, padding=ft.Padding(104, 0, 0, 0)),
            ft.Container(height=4),
            preview,
        ]
        try:
            page.update()
        except Exception:
            pass

    folder.on_submit = folder.on_blur = commit
    name.on_submit = name.on_blur = commit
    rebuild()
    page.show_dialog(ft.AlertDialog(
        title=ft.Text("Where recordings go", size=t.H2, color=t.TEXT,
                      weight=ft.FontWeight.W_500),
        content=ft.Container(width=520, content=body),
        bgcolor=t.SURFACE,
        actions=[ft.Container(content=ft.Text("close", size=t.BODY, color=t.ACCENT),
                              padding=ft.Padding(12, 7, 12, 7), border_radius=6,
                              on_click=lambda _e: (commit(), page.pop_dialog()))]))


def _row(label: str, *controls: ft.Control) -> ft.Control:
    return ft.Row([ft.Container(content=ft.Text(label, size=t.SMALL, color=t.FAINT), width=96),
                   *controls], spacing=8,
                  vertical_alignment=ft.CrossAxisAlignment.CENTER)


def _button(label: str, on_click) -> ft.Control:
    return ft.Container(content=ft.Text(label, size=t.SMALL, color=t.ACCENT),
                        padding=ft.Padding(9, 5, 9, 5), border_radius=6,
                        border=ft.Border.all(1, t.BORDER), on_click=on_click)
