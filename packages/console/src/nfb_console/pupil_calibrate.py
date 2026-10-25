"""Follow-the-dot calibration for the eye tracker, full screen.

It belongs to the device, not the training, so it opens from the device screen. It needs the whole
window, because the dots must sit at known places on the real screen. While it runs, it holds the
tracker itself, so the app stops the pupil reader first and starts it again when this returns.

The work runs on a worker thread: opening the tracker and sending it points are blocking calls, and
nothing blocking may sit on the drawing loop. The thread drives `calibration.Calibrator`; the draw loop
reads its fields and moves the dot. The person looks at each dot until it is taken, then the next
appears.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import flet as ft
import flet.canvas as cv

from nfb.pupil import calibration, tracker

from . import chrome
from . import theme as t
from .inputs import PupilChoice

DOT_R = 16


def view(page: ft.Page, choice: PupilChoice, on_done: Callable[[], None]) -> ft.Control:
    width, height, _why = choice.screen_size()
    state = {"cal": None, "error": None, "stop": False}

    canvas = cv.Canvas(shapes=[], expand=True)
    headline = ft.Text("Look at the dot", size=18, color=t.TEXT, weight=ft.FontWeight.W_400)
    detail = ft.Text("Look at each dot and hold until its ring fills. Then the next dot appears.",
                     size=t.BODY, color=t.MUTED)
    finish_btn = ft.Container(
        content=ft.Text("Stop", size=t.BODY, color=t.TEXT),
        padding=ft.Padding(16, 9, 16, 9), border_radius=7, border=ft.Border.all(1, t.BORDER),
        on_click=lambda _e: leave())

    done = {"left": False}

    # Full screen: a dot's window fraction must equal its screen fraction, which is what each point tells
    # the tracker.
    try:
        page.window.full_screen = True
        page.update()
    except Exception:
        pass

    def leave() -> None:
        if done["left"]:
            return
        done["left"] = True
        state["stop"] = True
        try:
            page.window.full_screen = False
            page.update()
        except Exception:
            pass
        on_done()

    def worker() -> None:
        link = None
        try:
            link = tracker.UsbLink()
            session = tracker.TrackerSession(link, (width, height))
            session.connect()
            cal = calibration.Calibrator(session, serial=link.serial)
            state["cal"] = cal
            while not state["stop"] and not cal.done and not cal.failed:
                cal.step()              # its gaze read paces the loop; no extra sleep needed
            if state["stop"] and not cal.done:
                cal.cancel()
            # Keep the session open after a good calibration so the live gaze dot moves — this is the
            # check that it worked. The connection holds the applied model; closing it loses it.
            while not state["stop"] and cal.done and not cal.failed:
                cal.refresh_gaze()
        except tracker.TrackerError as exc:
            state["error"] = str(exc)
        finally:
            if link is not None:
                try:
                    link.close()
                except Exception:
                    pass

    threading.Thread(target=worker, daemon=True, name="pupil-calibrate").start()

    def draw() -> None:
        cal = state["cal"]
        w = getattr(page, "width", None) or 1000
        h = getattr(page, "height", None) or 700
        shapes = []
        if state["error"] is not None:
            headline.value = "The tracker did not calibrate"
            detail.value = state["error"]
            finish_btn.content.value = "Back"
        elif cal is None:
            headline.value = "Opening the tracker"
            detail.value = "This takes a moment."
        elif cal.done or cal.failed:
            headline.value = "Done" if cal.done else "Stopped"
            detail.value = cal.message
            finish_btn.content.value = "Back"
            # The check that it worked: a dot where the tracker now thinks the eyes look.
            if cal.done and cal.last_screen_gaze is not None:
                gx, gy = cal.last_screen_gaze
                shapes = [cv.Circle(gx * w, gy * h, 14, paint=ft.Paint(
                              color=t.ACCENT, style=ft.PaintingStyle.FILL)),
                          cv.Circle(gx * w, gy * h, 22, paint=ft.Paint(
                              color=t.ACCENT, style=ft.PaintingStyle.STROKE, stroke_width=2))]
        elif cal.index >= len(cal.points):
            # Every point captured; the device is computing and saving the model, which takes a moment.
            headline.value = "Saving the calibration"
            detail.value = "a moment…"
        else:
            import math

            x, y = cal.points[cal.index]
            px, py = x * w, y * h
            progress = cal.dwell_progress()
            ring_r = DOT_R + 12
            shapes = [cv.Circle(px, py, ring_r, paint=ft.Paint(
                          color=t.SURFACE_HI, style=ft.PaintingStyle.STROKE, stroke_width=4))]
            if progress > 0:
                # The ring fills from the top as the gaze is held, so "keep looking" and "done" are visible.
                shapes.append(cv.Arc(px - ring_r, py - ring_r, ring_r * 2, ring_r * 2,
                                     -math.pi / 2, 2 * math.pi * progress,
                                     paint=ft.Paint(color=t.ACCENT, style=ft.PaintingStyle.STROKE,
                                                    stroke_width=4, stroke_cap=ft.StrokeCap.ROUND)))
            shapes.append(cv.Circle(px, py, DOT_R, paint=ft.Paint(color=t.ACCENT,
                                                                  style=ft.PaintingStyle.FILL)))
            headline.value = "Look at the dot and hold still"
            if cal.last_gaze is None:
                detail.value = ("the tracker does not see your eyes — sit about an arm's length away, "
                                "with the tracker below the screen")
            else:
                detail.value = f"point {cal.index + 1} of {len(cal.points)} · hold until the ring fills"
        canvas.shapes = shapes
        page.update()

    page.run_task(_loop, page, state, draw)

    # The text sits in a thin band at the very top and the button in the top-left corner, clear of the
    # dots, which never come nearer the edges than a tenth of the screen.
    body = ft.Stack([
        ft.Container(content=canvas, expand=True),
        ft.Container(content=ft.Column([headline, detail],
                                       horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                     alignment=ft.Alignment.TOP_CENTER, top=8, left=0, right=0),
        ft.Container(content=finish_btn, top=10, left=16),
    ], expand=True)
    return ft.Container(content=body, expand=True, bgcolor=t.BG)


async def _loop(page: ft.Page, state: dict, draw: Callable[[], None]) -> None:
    import asyncio

    while not state["stop"]:
        try:
            draw()
        except Exception:
            pass                # a bad frame must not end the screen; the next frame is usually fine
        await asyncio.sleep(0.1)
    try:
        draw()
    except Exception:
        pass
