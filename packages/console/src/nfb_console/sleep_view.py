"""The sleep screen: record the night, show its structure, and wake somebody if the electrodes fail.

A monitoring screen, not a feedback one — it offers no mechanisms, drives no outputs, and its product
is the recording (armed by default, see `records_by_default`). What it shows is the night: a cascade
of the whole spectrum over hours, the delta share and delta/beta ratio that sleep's structure is
visible in — deliberately *not* labelled sleep stages, since nothing here implements staging — and a
health tally per channel, because "was my recording any good" should be answerable while it can still
be fixed. The audible tier of the `sentry` is armed here and nowhere else: an electrode that comes
off at 2 a.m. costs the rest of the night, and a warning nobody hears is not a warning.
"""

from __future__ import annotations

import time
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb import Device
from nfb.dsp import Band

from . import chrome, cues, plan_pane, quality_pane, spectrum_pane, trace_rows, trends_pane
from . import theme as t
from .plots import (Chronospectrogram, StickyCeiling, bin_centre_x, grid, place_tick_labels,
                    place_value_labels, polyline, tick_axis, value_axis)
from .protocols import Protocol
from .traces import RawRing, TraceBuffer
from .training import TrainingSession
from .widgets import help_button, num_box, panel, pill, test_button

HANDLES = ("SLEEP",)

SLEEP_BANDS = (Band(0.5, 4, "delta"), Band(4, 8, "theta"), Band(8, 12, "alpha"),
               Band(12, 15, "sigma"), Band(15, 30, "beta"))
"""The conventional sleep bands. Sigma is the spindle band, which is what distinguishes this set from
the training screens' — spindles are stage-2 sleep's signature and no training protocol here has a
tile for 12–15 under that name."""

SPECTRUM_LO, SPECTRUM_HI, SPECTRUM_BIN_HZ = 1.0, 45.0, 1.0

TREND_UPDATE_SECONDS = 10.0
TREND_RETAIN = int(12 * 3600 / TREND_UPDATE_SECONDS)
"""A night, with margin: twelve hours at a ten-second cadence. The training screens sample every 2 s
because a 2-minute span has to feel continuous; a night is read in hours, and five times fewer samples
is what lets the retention cover it at the same cost."""

CASCADE_ROWS = 160
CASCADE_LO, CASCADE_HI = 1.0, 30.0
CASCADE_DEPTH_SECONDS = 8 * 3600.0
CASCADE_H = 236
CASCADE_TICKS = (5, 10, 15, 20, 25, 30)
CASCADE_CEILING_UV = 30.0
"""Higher than the CFC cascade's 20: slow-wave sleep's delta is the largest signal any screen here
draws, and a ceiling it saturates would flatten exactly the structure the picture exists to show."""

STRUCTURE_H = 150
TRACE_H = 64
TRACE_SECONDS = 10.0

NIGHT_HELP = (
    "**What you see.** The whole night, newest row at the top. Each row shows the amplitude of every "
    f"frequency over {CASCADE_DEPTH_SECONDS / CASCADE_ROWS / 60:.0f} minutes. Below it, two lines: the "
    "delta share of the signal, and the delta/beta ratio on its own scale. Further below, a health score "
    "for each channel.\n\n"
    "**What it means.** Deep sleep shows heavy low-frequency rows. Waking shows the whole row lighting "
    "up. Delta share rises in deep sleep. These two lines are not sleep stages, because this screen does "
    "not detect stages. They are the raw numbers that stages are built from. Each channel's health score "
    "shows how much of the night it was usable, the total artifact rejection, and the gaps. It also "
    "shows how close the signal came to the rail.\n\n"
    "**What to do.** Set the alarm in the ALARM panel. If a channel stays unusable past the time you "
    "set, a red banner names the channel and the fix. The beep repeats at the set interval and volume "
    "until the signal recovers. Reattach the electrode to stop the beep. Choose \"silent\" to keep the "
    "banner and drop the sound. Heavy artifact rejection alone never sounds the alarm, so moving in bed "
    "does not wake you."
)


class SleepSession(TrainingSession):
    """A session with no targets and no outputs: the recording and the watching are the product."""

    offers: tuple[str, ...] = ()
    music_sources: tuple[str, ...] = ()
    records_by_default = True
    """The one screen where recording is the point rather than a paper trail — the toolbar's record
    toggle arms itself here unless the operator has said otherwise."""

    def __init__(self, protocol: Protocol, path: str | None = None,
                 device: Device | None = None, length_seconds: float | None = None,
                 source=None, channels: int = 0):
        super().__init__(protocol, path=path, device=device, length_seconds=length_seconds,
                         source=source, channels=channels)
        self.sentry.audible = True
        self.trace_seconds = TRACE_SECONDS
        self.gain_uv = 100.0
        self.traces = {i: TraceBuffer(320, self.trace_seconds, s.rate_hz)
                       for i, s in self.specs.items()}
        self.rings = {i: RawRing(2.0, s.rate_hz, SPECTRUM_LO, SPECTRUM_HI, SPECTRUM_BIN_HZ)
                      for i, s in self.specs.items()}
        self.spectrogram = Chronospectrogram(CASCADE_ROWS, CASCADE_LO, CASCADE_HI, 1.0,
                                             depth_seconds=CASCADE_DEPTH_SECONDS)
        self.spectrogram.ceiling_uv = CASCADE_CEILING_UV
        self._spectrogram_at = 0.0
        # Signal time, not the session clock: the night runs whether or not anyone pressed play, and
        # the trends, the cascade and the tally must not depend on the transport.
        self.night_seconds = 0.0
        self._signal_seconds: dict[int, float] = {}
        self._lost_seconds: dict[int, float] = {}

    def consume(self, block) -> set:
        return set()                         # no metrics: nothing here is trained

    def observe_raw(self, block, judged) -> None:
        ch = block.channel
        values = np.asarray(block.values)
        self.traces[ch].push(self.for_display(ch, values), block.valid, judged.accepted)
        self.rings[ch].push(values)
        seconds = len(values) / block.spec.rate_hz
        self._signal_seconds[ch] = self._signal_seconds.get(ch, 0.0) + seconds
        if self.state.get(ch) is False:
            self._lost_seconds[ch] = self._lost_seconds.get(ch, 0.0) + seconds
        if ch == self.focus:
            self.night_seconds += seconds

    def usable_fraction(self, channel: int) -> float | None:
        """How much of the night this channel was usable, 0…1, or None before anything arrived."""
        total = self._signal_seconds.get(channel)
        if not total:
            return None
        return 1.0 - self._lost_seconds.get(channel, 0.0) / total

    def calibration(self) -> dict:
        """The alarm's numbers persist with the calibration: how long before it sounds, how often it
        repeats and how loud are the same kind of number as a volume — the person's and the room's,
        set once by ear, changing nothing about what is recorded."""
        out = super().calibration()
        s = self.sentry
        out.update({"sentry.audible": s.audible, "sentry.alarm_seconds": s.alarm_seconds,
                    "sentry.repeat_seconds": s.repeat_seconds, "sentry.volume": s.volume})
        return out

    def restore_calibration(self, values: dict) -> int:
        n = super().restore_calibration(values)
        s = self.sentry
        for attr in ("alarm_seconds", "repeat_seconds", "volume"):
            value = values.get(f"sentry.{attr}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(s, attr, float(value))
                n += 1
        audible = values.get("sentry.audible")
        if isinstance(audible, bool):
            s.audible = audible
            n += 1
        return n

    def push_display(self, now: float) -> None:
        """One cascade row per interval, from the draw loop — a time axis has to advance at its own
        rate however the blocks arrive, the same rule the CFC cascade states."""
        if now - self._spectrogram_at < self.spectrogram.interval:
            return
        self._spectrogram_at = now
        ring = self.rings.get(self.focus)
        if ring is None:
            return
        row = ring.spectrum()
        if row.size:
            self.spectrogram.push(row[:self.spectrogram.bins])


def session_for(*args, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None,
                **kwargs) -> SleepSession:
    _ = heart_choice, heart_runner, mendi_choice, mendi_runner, pupil_choice, pupil_runner
    return SleepSession(*args, **kwargs)


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def view(page: ft.Page, session: SleepSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    """Build once, mutate on Flet's loop."""

    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width(reserve: int = 0) -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - reserve - 70)

    spectrum = spectrum_pane.SpectrumPane(page, session, plot_width)
    trends = trends_pane.TrendsPane(page, session, plot_width, bands=SLEEP_BANDS,
                                    update_seconds=TREND_UPDATE_SECONDS, retain=TREND_RETAIN)

    # ---- the alarm, over everything ----
    alarm_text = ft.Text("", size=t.BODY, color=t.ON_ACCENT, weight=ft.FontWeight.W_600,
                         expand=True)
    alarm = ft.Container(
        content=ft.Row([ft.Text("⚠", size=16, color=t.ON_ACCENT), alarm_text], spacing=10,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.Padding(16, 12, 16, 12), bgcolor=t.BAD, border_radius=8, visible=False)
    advice_text = ft.Text("", size=t.SMALL, color=t.WARN)

    # ---- the alarm's own controls: the numbers are the operator's, not the module's ----
    sentry_ = session.sentry
    alarm_pills = ft.Row(spacing=6)

    def set_audible(on: bool) -> None:
        sentry_.audible = on
        build_alarm_pills()

    def build_alarm_pills() -> None:
        alarm_pills.controls = [
            pill("sound on", sentry_.audible, lambda _e: set_audible(True)),
            pill("silent", not sentry_.audible, lambda _e: set_audible(False)),
            ft.Container(width=8),
            # the shared audition control, as on the Session tab — same glyph, same disable rule
            test_button("hear it", lambda _e: session.cue("beep", sentry_.volume),
                        cues.problem() is None),
        ]
        page.update()

    def set_number(attr: str):
        def apply(value: float) -> None:
            setattr(sentry_, attr, float(value))
        return apply

    alarm_row = ft.Row([
        alarm_pills, ft.Container(width=14),
        ft.Text("after", size=t.SMALL, color=t.FAINT),
        num_box(sentry_.alarm_seconds, set_number("alarm_seconds"), suffix="s", step=10,
                low=10, high=600),
        ft.Container(width=10),
        ft.Text("repeat every", size=t.SMALL, color=t.FAINT),
        num_box(sentry_.repeat_seconds, set_number("repeat_seconds"), suffix="s", step=5,
                low=5, high=300),
        ft.Container(width=10),
        ft.Text("volume", size=t.SMALL, color=t.FAINT),
        num_box(sentry_.volume, set_number("volume"), step=10, low=0, high=100),
        # no wrap: a wrapped child with no stated width takes a whole line, one control per row
    ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    alarm_caption = ft.Text(
        "beeps once a channel has been unusable for the whole \"after\" time, then repeats until it "
        "recovers · silent keeps the banner and drops the sound · advisories never sound",
        size=t.SMALL, color=t.FAINT)

    # ---- the cascade ----
    cascade_image = ft.Image(src=None, height=CASCADE_H, expand=True, fit=ft.BoxFit.FILL,
                             filter_quality=ft.FilterQuality.NONE)
    cascade_empty = ft.Text("waiting for a second of data", size=t.SMALL, color=t.FAINT)
    cascade_scale, cascade_labels = tick_axis(CASCADE_TICKS, tick_width=30)
    cascade_caption = ft.Text("", size=t.SMALL, color=t.FAINT)

    # ---- the night's structure: delta share, and delta/beta on its own scale ----
    structure_canvas = cv.Canvas(shapes=[], expand=True, height=STRUCTURE_H)
    structure_axis, structure_labels = value_axis(STRUCTURE_H, count=4, width=40)
    structure_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
    structure_empty = ft.Text("waiting for data", size=t.SMALL, color=t.FAINT)
    structure_legend = ft.Row(spacing=12)
    ratio_ceiling = StickyCeiling()

    # ---- health ----
    health_rows = {i: ft.Text("", size=t.SMALL, color=t.MUTED) for i in session.specs}
    health_line = ft.Text("", size=t.SMALL, color=t.FAINT)
    events_text = ft.Text("", size=t.SMALL, color=t.MUTED)
    timeline = quality_pane.QualityPane(page, session, plot_width, titled=False)

    traces = trace_rows.TraceRows(page, session, plot_width, height=TRACE_H)

    night_tab = ft.Container(content=ft.Column([
        alarm,
        panel("ALARM", alarm_row, ft.Container(height=4), alarm_caption),
        ft.Container(height=12),
        panel("NIGHT",
              ft.Row([ft.Container(width=40),
                      ft.Stack([ft.Container(content=cascade_image, expand=True),
                                ft.Container(content=cascade_empty, expand=True,
                                             alignment=ft.Alignment.CENTER)],
                               expand=True, height=CASCADE_H)], spacing=0),
              ft.Row([ft.Container(width=40),
                      ft.Container(content=cascade_scale, expand=True)], spacing=0),
              ft.Container(height=4), cascade_caption,
              help=help_button(page, "The night", NIGHT_HELP)),
        ft.Container(height=12),
        panel("STRUCTURE",
              ft.Container(content=structure_legend, padding=ft.Padding(0, 0, 0, 8)),
              ft.Row([structure_axis,
                      ft.Stack([ft.Container(content=structure_canvas, expand=True),
                                ft.Container(content=structure_empty, expand=True,
                                             alignment=ft.Alignment.CENTER)],
                               expand=True, height=STRUCTURE_H)], spacing=0),
              ft.Container(height=4), structure_caption),
        ft.Container(height=12),
        panel("HEALTH",
              timeline.control, ft.Container(height=12),
              *health_rows.values(), ft.Container(height=4), health_line, advice_text,
              events_text),
        ft.Container(height=12),
        panel("RAW", *traces.rows, ft.Container(height=6), traces.caption),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO),
        padding=ft.Padding(14, 12, 14, 14), expand=True)

    panes = [night_tab, trends.control, spectrum.control, plan_pane_.control]

    def select_tab(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs(["Night", "Trends", "Spectrum", "Session"], select_tab)

    delta_colour = t.band_colour(0.5, 4)

    def draw() -> None:
        session.tick()
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()

        now = time.monotonic()
        # Watching and collecting are never gated on visibility — the beep must fire and the history
        # must accumulate whichever tab is up. Painting is; see SpectrumPane.refresh.
        session.sentry.tick(session, now)
        session.push_display(now)
        trends.refresh()        # collection is on the feed — see TrainingSession.feed_block
        spectrum.refresh()
        if not night_tab.visible:
            page.update()
            return

        problem = session.sentry.problem
        alarm.visible = problem is not None
        if problem:
            alarm_text.value = problem
        advice = session.sentry.advice
        advice_text.value = advice or ""
        advice_text.visible = bool(advice)

        image = session.spectrogram.image()
        cascade_image.src = image
        cascade_image.visible = image is not None
        cascade_empty.visible = image is None
        width = plot_width(40)
        sg = session.spectrogram
        cascade_image.width = width
        x_of = bin_centre_x(sg.lo_hz, sg.bin_hz, sg.bins, width, top=sg.hi_hz)
        place_tick_labels(cascade_labels, CASCADE_TICKS, x_of, tick_width=30, width=width)
        so_far = sg.seconds_filled * sg.depth_seconds
        cascade_caption.value = (
            f"ch{session.focus} · {sg.lo_hz:g}–{sg.hi_hz:g} Hz in {sg.bin_hz:g} Hz bins · "
            f"0–{sg.ceiling_uv:g} µV full scale · {_hours(sg.depth_seconds)} deep, one row per "
            f"{sg.interval / 60:.0f} min"
            + (f" · {_hours(so_far)} of it so far" if 0 < sg.seconds_filled < 1.0 else ""))

        label = f"ch{session.focus}"
        window = trends.trends.window(label, 0.0)
        have = len(window) >= 2
        structure_empty.visible = not have
        if have:
            h = float(STRUCTURE_H)
            t0, t1 = window[0][0], window[-1][0]
            spread = max(t1 - t0, 1e-9)
            shares, ratios = [], []
            for at, absolute, _peaks in window:
                total = sum(absolute.values())
                beta = absolute.get("beta", 0.0)
                shares.append((at, 100.0 * absolute.get("delta", 0.0) / total if total else 0.0))
                ratios.append((at, absolute.get("delta", 0.0) / beta if beta > 0 else 0.0))
            ratio_top = ratio_ceiling.update(max(r for _at, r in ratios), now)
            shapes = grid(width, h)
            for pts, top, colour in ((shares, 100.0, delta_colour), (ratios, ratio_top, t.ACCENT)):
                if len(pts) > width:
                    take = np.linspace(0, len(pts) - 1, int(width)).astype(int)
                    pts = [pts[i] for i in take]
                shapes.append(polyline(
                    [ft.Offset((at - t0) / spread * width, h - min(1.0, v / top) * h)
                     for at, v in pts], colour, 1.4))
            structure_canvas.shapes = shapes
            place_value_labels(structure_labels, h, 100.0, unit="%")
            structure_legend.controls = [
                ft.Row([ft.Container(width=10, height=2, bgcolor=delta_colour, border_radius=1),
                        ft.Text("delta share, 0–100 %", size=t.SMALL, color=t.MUTED)], spacing=6),
                ft.Row([ft.Container(width=10, height=2, bgcolor=t.ACCENT, border_radius=1),
                        ft.Text(f"delta/beta, 0–{ratio_top:g} on its own scale", size=t.SMALL,
                                color=t.MUTED)], spacing=6)]
            structure_caption.value = (
                f"{label} · the whole night ({_hours(spread)}) · axis is the delta share; the ratio "
                f"line carries its own 0–{ratio_top:g} scale · not sleep stages — the quantities "
                f"staging is built from")
        else:
            structure_canvas.shapes = []
            structure_caption.value = "waiting for data — the first sample lands a second in"

        for i in session.specs:
            usable = session.usable_fraction(i)
            if usable is None:
                health_rows[i].value = f"ch{i} · waiting for data"
                health_rows[i].color = t.FAINT
                continue
            room = session.headroom_uv(i)
            lost = session._lost_seconds.get(i, 0.0)
            gate = session.gates.gates.get(i)
            health_rows[i].value = (
                f"ch{i} · usable {usable:.1%} of {_hours(session._signal_seconds[i])}"
                + (f" · {_hours(lost)} lost" if lost else "")
                + (f" · rejecting {gate.percent:.1%} overall" if gate is not None else "")
                + (f" · worst headroom {room:.0f} µV" if room is not None else ""))
            health_rows[i].color = t.MUTED if usable > 0.98 else t.WARN
        timeline.refresh()
        health_line.value = f"{session.gaps} gaps in acquisition · {_hours(session.night_seconds)} of signal"
        recent = session.sentry.events[-3:]
        events_text.value = "\n".join(
            f"{time.strftime('%H:%M:%S', time.localtime(at))} — {msg}" for at, msg in recent)
        events_text.visible = bool(recent)

        traces.refresh(session.trace_seconds)
        page.update()

    session.redraw = draw

    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)

    build_alarm_pills()
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control,
                      ft.Stack(panes, expand=True)], spacing=0, expand=True)
