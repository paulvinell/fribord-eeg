"""The eye tracker's infrared camera, full screen, for lining the eyes up.

The camera and the gaze reader cannot run at once on the one device, so this takes the tracker over while
it is open, the way calibration does, and hands it back on the way out. The picture is the tracker's own
560×560 infrared view; it is a thing to look at, not to measure.

The frames are read and turned into pictures on a worker thread; the draw loop only shows the newest one.
"""

from __future__ import annotations

import threading
from typing import Callable

import flet as ft
import numpy as np

from nfb.pupil.camera import HEIGHT, TrackerCamera, WIDTH

from . import theme as t
from .plots import png_bytes

STEP = 2        # show every second pixel: 280×280 is plenty for lining up, and encodes quickly
TARGET_BRIGHTNESS = 110.0
"""The tracker strobes its infrared lamps, so raw frames alternate bright and dark. Each frame is scaled
to this average brightness, which levels that flicker and lifts the generally dark infrared picture."""


def level(grey: "np.ndarray") -> "np.ndarray":
    """Scale a frame to a steady brightness, so the lamp's on/off flicker does not show."""
    mean = float(grey.mean())
    if mean < 1.0:
        return grey
    return np.clip(grey.astype(np.float32) * (TARGET_BRIGHTNESS / mean), 0, 255).astype(np.uint8)


def view(page: ft.Page, on_done: Callable[[], None]) -> ft.Control:
    state = {"png": None, "error": None, "stop": False, "seen": 0}

    image = ft.Image(src=None, width=520, height=520, fit=ft.BoxFit.CONTAIN,
                     filter_quality=ft.FilterQuality.NONE, border_radius=10, visible=False)
    headline = ft.Text("Camera", size=18, color=t.TEXT, weight=ft.FontWeight.W_400)
    detail = ft.Text("the tracker's infrared view — line your eyes up in the middle",
                     size=t.BODY, color=t.MUTED)
    done = {"left": False}

    def leave() -> None:
        if done["left"]:
            return
        done["left"] = True
        state["stop"] = True
        on_done()

    back = ft.Container(content=ft.Text("Back", size=t.BODY, color=t.TEXT),
                        padding=ft.Padding(16, 9, 16, 9), border_radius=7,
                        border=ft.Border.all(1, t.BORDER), on_click=lambda _e: leave())

    def worker() -> None:
        cam = TrackerCamera()
        try:
            for frame in cam.frames():
                if state["stop"]:
                    break
                grey = np.frombuffer(frame, dtype=np.uint8).reshape(HEIGHT, WIDTH)[::STEP, ::STEP]
                grey = level(grey)
                rgb = np.ascontiguousarray(np.repeat(grey[:, :, None], 3, axis=2))
                state["png"] = png_bytes(rgb)
                state["seen"] += 1
            if cam.error and not state["stop"]:
                state["error"] = cam.error
        finally:
            cam.close()

    threading.Thread(target=worker, daemon=True, name="pupil-camera").start()

    def draw() -> None:
        if state["error"] is not None:
            headline.value = "The camera did not open"
            detail.value = state["error"]
        elif state["png"] is None:
            headline.value = "Opening the camera"
            detail.value = "a moment…"
        else:
            headline.value = "Camera"
            detail.value = "the tracker's infrared view — line your eyes up in the middle"
            image.src = state["png"]
            image.visible = True
        page.update()

    page.run_task(_loop, page, state, draw)

    return ft.Container(
        content=ft.Column(
            [headline, ft.Container(height=2), detail, ft.Container(height=14),
             image, ft.Container(height=14), back],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0,
            alignment=ft.MainAxisAlignment.CENTER, expand=True),
        expand=True, bgcolor=t.BG, alignment=ft.Alignment.CENTER)


async def _loop(page: ft.Page, state: dict, draw: Callable[[], None]) -> None:
    import asyncio

    while not state["stop"]:
        try:
            draw()
        except Exception:
            pass
        await asyncio.sleep(0.08)
    try:
        draw()
    except Exception:
        pass
