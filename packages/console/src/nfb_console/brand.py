"""The name and the mark.

The mark is a hull riding above a line of raw EEG: fribord is the freeboard, the height of a hull above
its waterline, and keeping a signal on the right side of a line is what every screen here trains. It is
drawn on a canvas rather than shipped as a picture, so it is sharp at every size and takes its colours
from the theme.
"""

from __future__ import annotations

import logging
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import flet as ft
import flet.canvas as cv

from . import theme as t

log = logging.getLogger(__name__)

NAME = "Fribord EEG"

ASSETS = str(Path(__file__).parent / "assets")

# Two weights as two families: Flet maps one family to one file, so it cannot pick a weight out of a set.
FONTS = {
    "Brand": "fonts/FamiljenGrotesk-SemiBold.ttf",
    "Brand Regular": "fonts/FamiljenGrotesk-Regular.ttf",
}

# The drawing, on a 64-unit square. The hull's keel (y 36) clears the trace's highest point (y 42.5) by
# a visible gap at every size; that gap is the freeboard.
HULL = ((12, 20), (52, 20), (45, 36), (19, 36))
TRACE = ((10, 46), (14, 44.5), (17, 47), (20, 45), (22, 48.5), (25, 43), (28, 47.5), (31, 45.5),
         (34, 46.5), (37, 42.5), (40, 49.5), (43, 45), (46, 47), (49, 45.5), (54, 46))


def mark(size: float = 28) -> ft.Control:
    s = size / 64
    # Below about 20 px a 2.6-unit stroke is under one screen pixel and the trace disappears.
    stroke = (3.4 if size <= 20 else 2.6) * s
    hull = [cv.Path.MoveTo(HULL[0][0] * s, HULL[0][1] * s)]
    hull += [cv.Path.LineTo(x * s, y * s) for x, y in HULL[1:]]
    hull.append(cv.Path.Close())
    trace = [cv.Path.MoveTo(TRACE[0][0] * s, TRACE[0][1] * s)]
    trace += [cv.Path.LineTo(x * s, y * s) for x, y in TRACE[1:]]
    canvas = cv.Canvas(
        shapes=[
            cv.Rect(0, 0, size, size, border_radius=14 * s, paint=ft.Paint(color=t.ACCENT)),
            cv.Path(elements=hull, paint=ft.Paint(color=t.SURFACE, style=ft.PaintingStyle.FILL)),
            cv.Path(elements=trace, paint=ft.Paint(
                color=t.ACCENT_DIM, stroke_width=stroke, style=ft.PaintingStyle.STROKE,
                stroke_cap=ft.StrokeCap.ROUND, stroke_join=ft.StrokeJoin.ROUND)),
        ],
        width=size, height=size)
    return canvas


def wordmark(size: float = 16) -> ft.Control:
    name, suffix = NAME.split(" ")
    return ft.Row(
        [ft.Text(name, size=size, font_family="Brand", color=t.TEXT),
         ft.Text(suffix, size=size, font_family="Brand Regular", color=t.MUTED)],
        spacing=size * 0.28, vertical_alignment=ft.CrossAxisAlignment.CENTER, tight=True)


def window_app() -> None:
    """Make the Dock show this app's name and icon, not Flet's.

    The window belongs to Flet's own desktop app, which `ft.run` starts, so the Dock shows whatever that
    bundle says it is. Flet can be pointed at another bundle with `FLET_VIEW_PATH`, so the first run on
    a machine copies Flet's, renames it and gives it this icon, and every run after uses the copy. One
    copy per Flet version, because the copy is that version's window app. On any failure the stock
    window opens, which costs only the icon.
    """
    if sys.platform != "darwin" or "FLET_VIEW_PATH" in os.environ:
        return
    try:
        import flet_desktop

        stock = flet_desktop.ensure_client_cached()
        home = stock.parent / f"fribord-{stock.name}"
        if not home.exists():
            work = home.parent / f"{home.name}.partial"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            app = work / f"{NAME}.app"
            shutil.copytree(stock / "Flet.app", app, symlinks=True)
            shutil.copy(Path(ASSETS) / "AppIcon.icns", app / "Contents/Resources/AppIcon.icns")
            info = app / "Contents/Info.plist"
            with info.open("rb") as f:
                plist = plistlib.load(f)
            # Without the icon *name* macOS reads the icon *file*; with it, the compiled catalogue
            # inside the bundle wins and that still holds Flet's icon.
            plist.pop("CFBundleIconName", None)
            plist["CFBundleName"] = plist["CFBundleDisplayName"] = NAME
            # A new identifier, so the system's icon cache cannot hand back Flet's icon for it.
            plist["CFBundleIdentifier"] = "eeg.fribord.console"
            with info.open("wb") as f:
                plistlib.dump(plist, f)
            # The edits break the copied signature, and macOS will not open an unsigned app.
            subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)],
                           check=True, capture_output=True)
            work.rename(home)
        os.environ["FLET_VIEW_PATH"] = str(home)
    except Exception:  # noqa: BLE001 - the stock window is a working fallback
        log.exception("could not brand the window app; using Flet's")
