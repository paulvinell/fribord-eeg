"""The Mendi's pulse against the HeartMath's, live: a program that compares two heart sensors.

It borrows the two runners the app already owns, the headband's and the heart-rate sensor's, and never
starts or stops either, so the HEG and HRV programs can use them too. Both hand out their beats and their
pulse in one form (`runner.PulseRunner`), so nothing here knows which device is which beyond its name.
Every couple of seconds it compares the last minute of the two pulses with `nfb.hrv.compare`. It trains
nothing: the height ratio and the delay are not validated as things to train, so they are shown, not fed
back.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import flet as ft
import flet.canvas as cv

from nfb.hrv.compare import compare, heights

from . import chrome, cues, plan_pane, prefs
from . import theme as t
from .plan import SessionPlan
from .plots import StickyCeiling, dashed, grid, place_value_labels, polyline, value_axis
from .protocols import Protocol
from .runner import PulseRunner
from .widgets import help_button, panel, stat_tile

HANDLES = ("PAIR",)
EEG_SOURCE = False

WINDOW_S = 60.0
"""How much of each pulse is compared at a time: long enough for the pairing to find the usual delay and
for the averaged pulse to stand clear of the noise, short enough to follow a change within a minute."""

UPDATE_S = 1.0
"""How often the comparison is worked out again, and so how often the training signal can move."""

SERIES_KEPT = 3600
"""Points kept for the session trends: an hour at one point a second."""

SMOOTH_BEATS = 8
"""How many heartbeats the training signal is the median of: about eight seconds at rest, enough to steady
one beat's measurement and short enough to follow the skin's vessels, which react over several seconds. A
starting value, not a measured one."""

START_BEATS = 20
"""How many of the session's first heartbeats set the ratio's starting value, the half-way point of the
level. Until that many have arrived the start is provisional, so feedback runs from the first beats."""

SETTLE_S = 1.2
"""How long after a heartbeat both sensors' pulses around it have arrived: the headband's cleaned waveform
runs about half a second behind, and a height is read up to 0.4 s past the beat."""


@dataclass(frozen=True)
class Snapshot:
    """One comparison of the last minute, handed whole from the session's thread to the view."""

    forehead_rate: float | None             # bpm, from each runner's heart-rate state
    ear_rate: float | None
    forehead_rmssd: float | None            # ms
    ear_rmssd: float | None
    paired: int                             # heartbeats seen at both sites, in the window
    forehead_only: int
    ear_only: int
    agreement: float | None                 # correlation of the two devices' paired intervals
    intervals: list[tuple[float, float, float]] = field(default_factory=list)   # (time, forehead, ear) s
    delay_ms: float | None = None           # forehead to ear, median over the window
    delay_spread_ms: float | None = None
    height_ratio: float | None = None       # ear pulse height over forehead pulse height, 1 = the start
    pulse_lags: list[float] = field(default_factory=list)       # s from the ear's beat
    pulse_shape: list[float] = field(default_factory=list)      # the forehead pulse averaged around them


class PairSession:
    """The two borrowed runners, the comparison and the session clock. No outputs: nothing is trained."""

    def __init__(self, protocol: Protocol, length_seconds: float | None = None,
                 heart_runner: PulseRunner | None = None, mendi_runner: PulseRunner | None = None,
                 heart_choice=None, mendi_choice=None):
        self.protocol = protocol
        self.plan = SessionPlan(length_seconds)
        self.cue = cues.play
        self.heart = heart_runner
        self.mendi = mendi_runner
        self.heart_choice = heart_choice
        self.mendi_choice = mendi_choice
        self.snapshot: Snapshot | None = None
        self.fault: str | None = None           # the last comparison's failure, if it failed
        self.delay_series: deque[tuple[float, float]] = deque(maxlen=SERIES_KEPT)
        self.ratio_series: deque[tuple[float, float]] = deque(maxlen=SERIES_KEPT)
        self.beat_ratios: deque[tuple[float, float]] = deque(maxlen=4096)  # (ear beat, ear over forehead)
        self._start_ratios: list[float] = []
        self._last_ratio_at = float("-inf")
        self.training: tuple[float, float] | None = None      # (ear beat, ratio against the start)
        self.finished = False
        self.paused = True
        self._stop = False
        self._accumulated = 0.0
        self._resumed_at: float | None = None
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self.redraw: Callable[[], None] = lambda: None

    # --- what is missing ----------------------------------------------------------------------

    def device_lines(self) -> list[str]:
        """One line per sensor, headband first: its trouble and what to do, or its own status."""
        out = []
        for name, runner, choice, where in (
                ("headband", self.mendi, self.mendi_choice, "choose the Mendi headband on the device screen"),
                ("heart sensor", self.heart, self.heart_choice, "choose the HeartMath on the device screen")):
            if runner is None or choice is None or not choice.on:
                out.append(f"no {name}: {where}")
            elif runner.error:
                out.append(f"{name}: {runner.error}")
            elif runner.stale:
                out.append(f"{name}: no readings for a while")
            else:
                out.append(f"{name}: {runner.describe()}")
        return out

    def missing(self) -> list[str]:
        """Why a sensor gives nothing to compare, one line each, or nothing when both are reading."""
        lines = self.device_lines()
        runners = (self.mendi, self.heart)
        return [line for line, runner in zip(lines, runners)
                if runner is None or line.startswith("no ") or runner.error or runner.stale
                or runner.heart_state is None]

    # --- the comparison -----------------------------------------------------------------------

    def update(self, now: float | None = None) -> None:
        """Compare the last `WINDOW_S` of both pulses and keep the result. `now` is in the beats' own
        seconds; a test passes it, the session's thread uses the wall clock the runners stamp with."""
        if self.missing():
            self.snapshot = None
            return
        now = time.time() if now is None else now
        since = now - WINDOW_S
        forehead = [t for t in self.mendi.beat_times.copy() if since <= t <= now]
        ear = [t for t in self.heart.beat_times.copy() if since <= t <= now]
        f_wave = [p for p in (self.mendi.waveform or []) if since <= p[0] <= now] or None
        e_wave = [p for p in (self.heart.waveform or []) if since <= p[0] <= now] or None
        result = compare(forehead, ear, f_wave, e_wave)
        a = result.agreement
        f_int, e_int, intervals = [], [], []
        for (f1, e1), (f2, e2) in zip(a.pairs, a.pairs[1:]):
            if 0.3 < f2 - f1 < 2.0 and 0.3 < e2 - e1 < 2.0:
                f_int.append(f2 - f1)
                e_int.append(e2 - e1)
                intervals.append((f2, f2 - f1, e2 - e1))
        ratio = self._per_beat(f_wave, e_wave, a.pairs, now)
        at = self.elapsed
        if result.delay is not None:
            self.delay_series.append((at, result.delay.median_s * 1000.0))
        if ratio is not None:
            self.ratio_series.append((at, ratio))
        f_state, e_state = self.mendi.heart_state, self.heart.heart_state
        self.snapshot = Snapshot(
            forehead_rate=f_state.hr_smoothed if f_state else None,
            ear_rate=e_state.hr_smoothed if e_state else None,
            forehead_rmssd=f_state.rmssd_ms if f_state else None,
            ear_rmssd=e_state.rmssd_ms if e_state else None,
            paired=len(a.pairs), forehead_only=len(a.forehead_only), ear_only=len(a.reference_only),
            agreement=_correlation(f_int, e_int), intervals=intervals,
            delay_ms=result.delay.median_s * 1000.0 if result.delay else None,
            delay_spread_ms=result.delay.spread_s * 1000.0 if result.delay else None,
            height_ratio=self.training[1] if self.training else None,
            pulse_lags=list(result.rescue.lags_s) if result.rescue else [],
            pulse_shape=list(result.rescue.shape) if result.rescue else [])

    def _per_beat(self, f_wave, e_wave, pairs, now: float) -> float | None:
        """The training signal: the ear pulse's height over the forehead's for each heartbeat both sensors
        saw, the median of the last `SMOOTH_BEATS`, against the median of the session's first `START_BEATS`.
        Each heartbeat is measured once, when its pulses have arrived at both sensors."""
        if not (f_wave and e_wave):
            return None
        settled = [(f, e) for f, e in pairs if self._last_ratio_at < e <= now - SETTLE_S]
        if settled:
            f_h = heights(f_wave, [f for f, _ in settled])
            e_h = heights(e_wave, [e for _, e in settled])
            for (_, e), fh, eh in zip(settled, f_h, e_h):
                self._last_ratio_at = e
                if fh and eh:
                    self.beat_ratios.append((e, eh / fh))
                    if len(self._start_ratios) < START_BEATS:
                        self._start_ratios.append(eh / fh)
        if not self.beat_ratios or not self._start_ratios:
            return None
        recent = [r for _, r in list(self.beat_ratios)[-SMOOTH_BEATS:]]
        value = statistics.median(recent) / statistics.median(self._start_ratios)
        self.training = (self.beat_ratios[-1][0], value)
        return value

    # --- the thread ---------------------------------------------------------------------------

    def run(self) -> None:
        """Start the comparison's own thread if it is not already running; the app calls this again on
        resume. The runners read their devices on their own threads, so nothing else needs starting."""
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, daemon=True, name="pair-compare")
            self._worker.start()

    def _loop(self) -> None:
        while not self._stop:
            if self.running:
                try:
                    self.update()
                    self.fault = None
                except Exception as exc:        # said in the toolbar; the comparison carries on
                    self.fault = f"{type(exc).__name__}: {exc}"
                    self.snapshot = None
            self._wake.wait(UPDATE_S)

    # --- transport, the contract the toolbar expects -----------------------------------------

    @property
    def elapsed(self) -> float:
        running = (time.monotonic() - self._resumed_at) if self._resumed_at is not None else 0.0
        return self._accumulated + running

    @property
    def running(self) -> bool:
        return not self.paused

    @property
    def length(self) -> float | None:
        return self.plan.length_seconds

    def toggle_pause(self) -> None:
        if self.paused:
            self.paused = False
            self._resumed_at = time.monotonic()
        else:
            self._accumulated = self.elapsed
            self._resumed_at = None
            self.paused = True

    def tick(self) -> None:
        if not self.running:
            return
        elapsed = self.elapsed
        for cue in self.plan.due(elapsed):
            self.cue(cue)
        if self.plan.complete(elapsed):
            self.finished = True
            self._accumulated = self.plan.length_seconds
            self._resumed_at = None
            self.paused = True

    def stop(self) -> None:
        """End the session. The runners are borrowed and keep running for whatever program is next."""
        self._stop = True
        self.paused = True
        self._wake.set()
        prefs.remember(self.protocol.code, self)

    # --- going to the device screen and coming back ------------------------------------------

    def suspend(self) -> None:
        if not self.paused:
            self.toggle_pause()

    @property
    def suspended(self) -> bool:
        return not self._stop

    def rearm(self) -> None:
        """Nothing to rebuild: the runners never stopped and the clock stayed where `suspend` left it."""

    # --- surviving a stop -------------------------------------------------------------------

    def adopt(self, previous: "PairSession") -> None:
        if previous is self or previous.protocol.code != self.protocol.code:
            return
        self.plan = previous.plan
        self.plan.rewind()

    def calibration(self) -> dict:
        return {}

    def restore_calibration(self, values: dict) -> int:
        return 0


def _correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 5:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def session_for(protocol: Protocol, path=None, device=None, length_seconds=None, source=None,
                channels: int = 0, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None) -> PairSession:
    """The other programs' signature, so `app.pick` need not special-case it."""
    _ = path, device, source, channels, pupil_choice, pupil_runner
    return PairSession(protocol, length_seconds=length_seconds, heart_runner=heart_runner,
                       mendi_runner=mendi_runner, heart_choice=heart_choice, mendi_choice=mendi_choice)


# --- the view -----------------------------------------------------------------------------------

SIDE_W = 330
AXIS_W = 58
INTERVAL_H, PULSE_H, TREND_H = 170, 120, 90
FOREHEAD, EAR = t.ACCENT, t.MUTED

PAIR_HELP = (
    "**What you see.** The program compares two sensors on one heart: the Mendi headband on the forehead "
    "and the HeartMath sensor on the ear. **intervals** shows the time between beats from each sensor over "
    "the last minute. The two lines must follow each other closely. **averaged pulse** shows the forehead "
    "pulse, averaged around each beat of the ear sensor. **over the session** shows the delay and the "
    "height ratio from the start of the session.\n\n"
    "**The numbers.** **agreement** shows how closely the two sensors' beat intervals follow each other, "
    "from 0 to 1. **delay** is the time from a beat at the forehead to the same beat at the ear. It "
    "includes the time each sensor takes to send its data, so only a change during a session tells you "
    "something. **height ratio** is the ear pulse height divided by the forehead pulse height, against its "
    "value at the start. Small vessels in the skin that tighten make it smaller, and so does a sensor "
    "that moves.\n\n"
    "**What you need.** Choose both sensors on the device screen. Connect the HeartMath with its cable for "
    "the height ratio and the averaged pulse shape at the ear. Over Bluetooth it sends only beats. The "
    "program trains nothing and gives no feedback."
)


def _instrument(session: PairSession) -> tuple[str, str, str]:
    """The toolbar chip: a short state. Which sensor is missing, and why, is said in the device panel."""
    if session.fault:
        return "two pulses", f"comparison failed: {session.fault}", t.BAD
    if any(r is not None and r.error for r in (session.mendi, session.heart)):
        return "two pulses", "a sensor reports a fault", t.BAD
    if session.missing():
        return "two pulses", "needs both sensors", t.MUTED
    if session.snapshot is None:
        return "two pulses", "comparing", t.ACCENT
    return "two pulses", "both sensors reading", t.GOOD


def view(page: ft.Page, session: PairSession, toolbar: chrome.Toolbar,
         on_stop: Callable[[], None]) -> ft.Control:
    plan_pane_ = plan_pane.PlanPane(page, session)

    def plot_width() -> float:
        width = getattr(page, "width", None) or 1020
        return max(260.0, width - SIDE_W - 76 - AXIS_W)

    def plot(height: int):
        canvas = cv.Canvas(shapes=[], expand=True, height=height)
        axis, labels = value_axis(height, count=3, width=AXIS_W)
        caption = ft.Text("waiting for beats", size=t.SMALL, color=t.FAINT)
        row = ft.Row([axis, ft.Container(content=canvas, expand=True, height=height)], spacing=0)
        return canvas, labels, caption, row

    iv_canvas, iv_labels, iv_caption, iv_row = plot(INTERVAL_H)
    pulse_canvas = cv.Canvas(shapes=[], expand=True, height=PULSE_H)
    pulse_caption = ft.Text("waiting for beats", size=t.SMALL, color=t.FAINT)
    delay_canvas, delay_labels, delay_caption, delay_row = plot(TREND_H)
    ratio_canvas, ratio_labels, ratio_caption, ratio_row = plot(TREND_H)
    iv_span, delay_span, ratio_span = StickyCeiling(floor=100.0), StickyCeiling(floor=20.0), StickyCeiling(floor=0.2)

    values: dict[str, ft.Text] = {}
    units: dict[str, ft.Text] = {}

    def tile(key: str, label: str, unit: str) -> ft.Control:
        box, value, unit_text, _ = stat_tile(label, 138, t.VALUE, unit)
        values[key], units[key] = value, unit_text
        return box

    tiles = ft.Column([
        ft.Row([tile("f_rate", "forehead", "bpm"), tile("e_rate", "ear", "bpm")], spacing=10),
        ft.Row([tile("f_rmssd", "forehead RMSSD", "ms"), tile("e_rmssd", "ear RMSSD", "ms")], spacing=10),
        ft.Row([tile("agreement", "agreement", "paired intervals"), tile("paired", "paired", "beats, last minute")],
               spacing=10),
        ft.Row([tile("delay", "delay", "ms, forehead to ear"), tile("ratio", "height ratio", "of the start")],
               spacing=10),
    ], spacing=10)
    # Two lines reserved for each sensor, so a long status cannot change the panel's height.
    device_texts = [ft.Text("", size=t.SMALL, color=t.MUTED, max_lines=2) for _ in range(2)]

    left = ft.Column([
        panel("INTERVALS", iv_row, ft.Container(height=6), iv_caption,
              help=help_button(page, "Pulse pair", PAIR_HELP)),
        ft.Container(height=12),
        panel("AVERAGED PULSE", ft.Row([ft.Container(width=AXIS_W),
                                        ft.Container(content=pulse_canvas, expand=True, height=PULSE_H)], spacing=0),
              ft.Container(height=6), pulse_caption),
        ft.Container(height=12),
        panel("OVER THE SESSION", delay_row, ft.Container(height=6), delay_caption, ft.Container(height=12),
              ratio_row, ft.Container(height=6), ratio_caption),
    ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)
    right = ft.Column([
        panel("NOW", tiles),
        ft.Container(height=12),
        panel("DEVICES", ft.Container(content=device_texts[0], height=32), ft.Container(height=6),
              ft.Container(content=device_texts[1], height=32)),
    ], spacing=0, width=SIDE_W, scroll=ft.ScrollMode.AUTO)
    main_tab = ft.Container(content=ft.Row([left, right], spacing=14, expand=True),
                            padding=ft.Padding(14, 12, 14, 14), expand=True)
    panes = [main_tab, plan_pane_.control]

    def select_tab(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs(["Pulses", "Session"], select_tab)
    shown: list = [None]
    drawn: list = [object()]

    def paint_tiles(snap: Snapshot | None) -> None:
        def num(v, fmt):
            return "—" if v is None else format(v, fmt)
        values["f_rate"].value = num(snap and snap.forehead_rate, ".0f")
        values["e_rate"].value = num(snap and snap.ear_rate, ".0f")
        values["f_rmssd"].value = num(snap and snap.forehead_rmssd, ".0f")
        values["e_rmssd"].value = num(snap and snap.ear_rmssd, ".0f")
        values["agreement"].value = num(snap and snap.agreement, ".2f")
        values["paired"].value = "—" if snap is None else str(snap.paired)
        units["paired"].value = ("beats, last minute" if snap is None else
                                 f"one site: {snap.forehead_only} + {snap.ear_only}")
        values["delay"].value = num(snap and snap.delay_ms, ".0f")
        values["ratio"].value = num(snap and snap.height_ratio, ".2f")

    def paint_intervals(snap: Snapshot | None) -> None:
        width = plot_width()
        if snap is None or len(snap.intervals) < 2:
            iv_canvas.shapes = []
            for label in iv_labels:
                label.value = ""
            iv_caption.value = "waiting for beats"
            return
        ms = [v * 1000.0 for _, f, e in snap.intervals for v in (f, e)]
        centre = round(statistics.fmean(ms) / 50.0) * 50.0
        half = iv_span.update(max(abs(v - centre) for v in ms), time.monotonic())
        top, bottom = centre + half, centre - half
        end = snap.intervals[-1][0]

        def x(at):
            return width * (1.0 - (end - at) / WINDOW_S)

        def y(v):
            return INTERVAL_H * (top - v * 1000.0) / (top - bottom)

        iv_canvas.shapes = grid(width, INTERVAL_H, rows=2, cols=6) + [
            polyline([ft.Offset(x(at), y(f)) for at, f, _ in snap.intervals], FOREHEAD, 1.4),
            polyline([ft.Offset(x(at), y(e)) for at, _, e in snap.intervals], EAR, 1.4)]
        place_value_labels(iv_labels, INTERVAL_H, top, bottom, unit="ms")
        iv_caption.value = "60 s across · blue = forehead, grey = ear · one point per beat interval"

    def paint_pulse(snap: Snapshot | None) -> None:
        width = plot_width()
        if snap is None or len(snap.pulse_shape) < 3:
            pulse_canvas.shapes = []
            pulse_caption.value = "waiting for beats"
            return
        lo, hi = min(snap.pulse_shape), max(snap.pulse_shape)
        span = (hi - lo) or 1.0
        first, last = snap.pulse_lags[0], snap.pulse_lags[-1]

        def x(lag):
            return width * (lag - first) / (last - first)

        zero = x(0.0)
        pulse_canvas.shapes = grid(width, PULSE_H, rows=2, cols=6) + [
            cv.Line(zero, 0, zero, PULSE_H, paint=ft.Paint(color=EAR, stroke_width=1.0)),
            polyline([ft.Offset(x(lag), PULSE_H * (1.0 - (v - lo) / span))
                      for lag, v in zip(snap.pulse_lags, snap.pulse_shape)], FOREHEAD, 1.6)]
        pulse_caption.value = (f"forehead pulse, averaged around each ear beat · {first:+.1f} s to {last:+.1f} s"
                               " · height fits the panel · grey = ear beat")

    def paint_trend(canvas, labels, caption, series, span_ceiling, unit, around, name) -> None:
        width = plot_width()
        points = list(series.copy())
        if len(points) < 2:
            canvas.shapes = []
            for label in labels:
                label.value = ""
            caption.value = f"{name} · waiting"
            return
        vals = [v for _, v in points]
        centre = around if around is not None else statistics.median(vals)
        half = span_ceiling.update(max(abs(v - centre) for v in vals), time.monotonic())
        top, bottom = centre + half, centre - half
        length = max(points[-1][0], 60.0)

        def y(v):
            return TREND_H * (top - v) / (top - bottom)

        shapes = grid(width, TREND_H, rows=2, cols=6)
        if around is not None:
            shapes += dashed(y(around), width, t.FAINT)
        shapes.append(polyline([ft.Offset(width * at / length, y(v)) for at, v in points], FOREHEAD, 1.4))
        canvas.shapes = shapes
        place_value_labels(labels, TREND_H, top, bottom, unit=unit)
        caption.value = f"{name} · {length / 60.0:.0f} min across"

    def draw() -> None:
        session.tick()
        instrument = _instrument(session)
        if instrument != shown[0]:
            shown[0] = instrument
            toolbar.set_source(instrument[0], note=instrument[1], tone=instrument[2])
        toolbar.refresh(session.running, session.elapsed, session.length)
        plan_pane_.refresh()
        missing = session.missing()
        for text, line in zip(device_texts, session.device_lines()):
            text.value = line
            text.color = t.TEXT if line in missing else t.MUTED
        snap = session.snapshot
        if snap is not drawn[0]:
            drawn[0] = snap
            paint_tiles(snap)
            paint_intervals(snap)
            paint_pulse(snap)
            paint_trend(delay_canvas, delay_labels, delay_caption, session.delay_series, delay_span, "ms",
                        None, "delay, forehead to ear")
            paint_trend(ratio_canvas, ratio_labels, ratio_caption, session.ratio_series, ratio_span, "",
                        1.0, "height ratio, dashed = the start")
        page.update()

    toolbar.set_channels_visible(False)
    session.redraw = draw
    toolbar.bind_transport(lambda _e: (session.toggle_pause(), draw()),
                           lambda _e: (session.stop(), on_stop()))
    toolbar.set_transport_visible(True)
    toolbar.set_transport_enabled(True)
    toolbar.set_crumb(session.protocol.name)
    page.run_task(chrome.render_loop, page, session, draw, toolbar)

    return ft.Column([toolbar.control, tabs.control, ft.Stack(panes, expand=True)], spacing=0, expand=True)
