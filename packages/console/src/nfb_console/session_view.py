"""The session screen: traces, contact state, the protocol's metric, and frame timing.

The frame-time readout is not decoration. The display path is the one term in the latency budget that
was never measured, so the screen that draws the feedback reports its own cost.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device, Mode, ReplaySource
from nfb.dsp import ALPHA, THETA, CFIRBandPower, Pipeline
from nfb.stream import Gap

from . import theme as t
from .training import corner_of, drift_for
from .protocols import Protocol
from .traces import TraceBuffer

COLUMNS = 420
WINDOW_SECONDS = 6.0
TRACE_HEIGHT = 74


class Session:
    """Owns the source, the DSP and the trace buffers, and feeds itself on a thread."""

    conditions_signal = True
    scores_bands = True
    level_source = None

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, columns: int = COLUMNS, source=None,
                 channels: int = 0):
        self.protocol = protocol
        self.device = device
        if source is not None:
            self.source = source          # already-built file source; see inputs.InputChoice
        elif device is not None:
            # subscribe; the device is owned by the app and outlives this session
            if device.status.mode is Mode.IDLE:
                device.start_measurement()
            self.source = device.subscribe(f"session:{protocol.code}")
        else:
            # loop by default: a 20 s capture has to stand in for a device indefinitely
            self.source = ReplaySource(path, realtime=True, loop=True)
        self.specs = self.source.channels()
        if channels:
            self.specs = {i: spec for i, spec in list(self.specs.items())[:channels]}
        self.traces = {
            i: TraceBuffer(columns, WINDOW_SECONDS, s.rate_hz) for i, s in self.specs.items()
        }
        self.pipeline = Pipeline()
        self.readout = "alpha µV"
        # Drift removed here for the same reason it is in a session: this readout is a cFIR on live
        # samples, and at the 0.01 Hz corner an electrode's standing offset leaks into it — the number
        # under a launcher card would read a few times its true size, on the screen somebody uses to
        # decide whether the hookup is good. See `training.drift_for`.
        drift = drift_for(corner_of(self.source, self.specs))
        for spec in self.specs.values():
            if protocol.code == "A-T":
                self.pipeline.add(CFIRBandPower(spec, ALPHA, delay_seconds=0.1, output_hz=20,
                                                drift_hz=drift))
                self.pipeline.add(CFIRBandPower(spec, THETA, delay_seconds=0.1, output_hz=20,
                                                drift_hz=drift))
                self.readout = "alpha / theta"
            else:
                self.pipeline.add(CFIRBandPower(spec, ALPHA, delay_seconds=0.1, output_hz=20,
                                                drift_hz=drift))
        self.latest: dict[int, float] = dict.fromkeys(self.specs, 0.0)
        self.state: dict[int, bool] = dict.fromkeys(self.specs, False)
        self.gaps = 0
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="session-feed")

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            for item in self.source.blocks():
                if self._stop.is_set():
                    return
                if isinstance(item, Gap):
                    self.gaps += 1
                    continue
                self.traces[item.channel].push(np.asarray(item.values), item.valid)
                self.state[item.channel] = item.valid
                self.samples += len(item.samples)
                for r in self.pipeline.feed(item):
                    if isinstance(r.channel, int):
                        self.latest[r.channel] = r.value
        except Exception as exc:                      # surface it instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        """Detach. Stops the device's stream but leaves the connection open for the next protocol."""
        self._stop.set()
        try:
            self.source.close()
        except Exception:
            pass
        if self.device is not None:
            self.device.stop()


def device_chip(session: Session, on_device: Callable[[], None] | None) -> ft.Control:
    """Reaching the device screen from inside a protocol.

    Opening it ends the session, because the impedance test and a measurement are exclusive device
    modes and because two screens must not both drive the render loop. Coming back re-enters the
    protocol with a fresh session.
    """
    label = "device" if session.device is not None else "replay"
    chip = ft.Container(
        content=ft.Row([ft.Text(label, size=t.SMALL, color=t.MUTED),
                        ft.Text("›", size=13, color=t.FAINT)], spacing=7),
        padding=ft.Padding(11, 6, 11, 6), border_radius=7,
        border=ft.Border.all(1, t.BORDER),
    )
    if on_device is not None:
        chip.on_click = lambda _e: on_device()
    else:
        chip.opacity = 0.5
    return chip


def view(page: ft.Page, session: Session, on_back: Callable[[], None],
         on_device: Callable[[], None] | None = None, benchmark: float = 0.0) -> ft.Control:
    p = session.protocol
    canvases = {i: cv.Canvas(shapes=[], expand=True) for i in session.specs}
    values = {i: ft.Text("—", size=16, color=t.ACCENT, weight=ft.FontWeight.W_500)
              for i in session.specs}
    dots = {i: ft.Container(width=7, height=7, bgcolor=t.BORDER, border_radius=4)
            for i in session.specs}
    frame_text = ft.Text("—", size=t.SMALL, color=t.FAINT)
    stat_text = ft.Text("—", size=t.SMALL, color=t.MUTED)

    back = ft.Container(
        content=ft.Row([ft.Text("←", size=15, color=t.MUTED),
                        ft.Text("protocols", size=t.BODY, color=t.MUTED)], spacing=8),
        padding=ft.Padding(10, 6, 12, 6), border_radius=6, on_click=lambda _e: on_back(),
    )

    header = ft.Container(
        content=ft.Row(
            [
                back,
                ft.Container(width=10),
                ft.Container(width=3, height=22, bgcolor=t.ACCENT, border_radius=2),
                ft.Container(width=10),
                ft.Text(p.code, size=19, weight=ft.FontWeight.W_400, color=t.TEXT),
                ft.Container(width=8),
                ft.Text(p.name, size=t.BODY, color=t.MUTED),
                ft.Container(expand=True),
                stat_text,
                ft.Container(width=16),
                frame_text,
                ft.Container(width=14),
                device_chip(session, on_device),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding(18, 14, 24, 14),
        border=ft.Border(bottom=ft.BorderSide(1, t.BORDER)),
    )

    rows = []
    for i in session.specs:
        rows.append(
            ft.Container(
                content=ft.Row(
                    [
                        ft.Container(
                            width=58,
                            content=ft.Row([dots[i], ft.Text(f"ch{i}", size=t.BODY, color=t.TEXT)],
                                           spacing=8),
                        ),
                        ft.Container(content=canvases[i], expand=True, height=TRACE_HEIGHT),
                        ft.Container(width=84, content=values[i],
                                     alignment=ft.Alignment.CENTER_RIGHT),
                    ],
                    spacing=14,
                ),
                padding=ft.Padding(22, 6, 22, 6),
            )
        )

    panel = ft.Container(
        content=ft.Column(
            rows + [
                ft.Container(
                    content=ft.Row([ft.Container(expand=True),
                                    ft.Text(session.readout, size=t.SMALL, color=t.FAINT)]),
                    padding=ft.Padding(22, 0, 26, 10),
                )
            ],
            spacing=0,
        ),
        bgcolor=t.SURFACE, border=ft.Border.all(1, t.BORDER), border_radius=10,
        margin=ft.Margin(18, 16, 18, 18), padding=ft.Padding(0, 10, 0, 0),
    )

    async def render_loop() -> None:
        """Runs on Flet's event loop, not a thread.

        A thread can mutate these controls and call page.update() without the client ever seeing it
        — see device_panel.poll for the evidence. Drawing on the loop also means the frame rate this
        reports is the rate Flet can actually deliver.
        """
        recent = deque(maxlen=60)
        builds = deque(maxlen=60)
        started = time.perf_counter()
        while not session._stop.is_set():
            t0 = time.perf_counter()
            for i, buf in session.traces.items():
                lo, hi = buf.envelope()
                half = buf.scale()
                mid = TRACE_HEIGHT / 2
                gain = (TRACE_HEIGHT / 2 - 3) / half
                pts = [ft.Offset(x * 2.0, mid - lo[x] * gain) for x in range(len(lo))]
                pts += [ft.Offset(x * 2.0, mid - hi[x] * gain) for x in range(len(hi) - 1, -1, -1)]
                ok = session.state.get(i)
                canvases[i].shapes = [
                    cv.Points(points=pts, point_mode=cv.PointMode.POLYGON,
                              paint=ft.Paint(color=t.ACCENT if ok else t.BORDER,
                                             stroke_width=1.0, style=ft.PaintingStyle.STROKE))
                ]
                values[i].value = f"{session.latest.get(i, 0.0):.1f}"
                dots[i].bgcolor = t.GOOD if ok else t.BAD
            builds.append((time.perf_counter() - t0) * 1000)
            page.update()
            now = time.perf_counter()
            recent.append(now)
            if len(recent) > 3:
                fps = (len(recent) - 1) / (recent[-1] - recent[0])
                frame_text.value = f"{fps:.0f} fps · {np.mean(builds):.1f} ms build"
            stat_text.value = (
                session.error or
                f"{session.samples // 1000}k samples · {session.gaps} gaps · "
                f"{session.pipeline.delay_seconds * 1000:.0f} ms delay"
            )
            if session.error:
                stat_text.color = t.BAD
            if benchmark and now - started > benchmark:
                d = np.diff(list(recent)) * 1000
                print(f"RESULT {1000 / np.median(d):.1f} fps median, "
                      f"build {np.mean(builds):.2f} ms", flush=True)
                import os
                os._exit(0)
            await asyncio.sleep(0.005)

    page.run_task(render_loop)
    return ft.Column([header, ft.Container(content=panel, expand=True)], spacing=0, expand=True)
