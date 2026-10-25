"""The Trends tab: where each band has been, averaged over a span the operator chooses.

The Spectrum tab answers "what is the signal doing *now*" — one second, live tiles. Nothing answered
"what has it been doing", which is the question a session is actually run for: did alpha come up, did
the peak drift, was the second half different from the first. A statistics window answers it: band trends as
coloured lines, absolute or relative, numbers beside them. The span is the operator's choice, the numbers
are averages rather than instantaneous values, and the session averages sit in one place.

So this pane is that window with the three gaps closed:

- **One span for everything on it**, typed in minutes, 0 meaning the whole session. Every number and
  every line on the pane answers over the same stretch of time, and the caption says which.
- **The numbers are averages over that span** — the live values are already on the Spectrum tiles, and
  printing them twice would add nothing. Amplitude in the chosen unit, and each band's mean dominant
  frequency beside it, which is "average peak frequency" per band rather than one overall number that is
  nearly always the slowest thing present (see `RawRing.band_peaks` for why per-band).
- **Absolute and relative are one switch**, shared with the Spectrum tab through `session.relative`, so
  the two tabs can never disagree about what a percentage means.

The collector samples the same `RawRing` the spectrum draws from — every `UPDATE_SECONDS`, per source —
so history accumulates from the session's first second whether or not this tab is ever opened. Sampling
lives in `BandTrends`, display in `TrendsPane`, and the split is what makes the collector testable by
pushing samples and ticking, without a page.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from . import theme as t
from .plots import StickyCeiling, fmt_amplitude, grid, place_value_labels, polyline, value_axis
from .spectrum_pane import TILE_BANDS
from .widgets import help_button, num_box, panel, pill

TREND_H = 200
AXIS_W = 40

UPDATE_SECONDS = 2.0
"""How often a sample is taken from each source's ring. The ring's spectral window is one second, so
two-second sampling makes consecutive samples nearly independent; faster would average the same window
twice, slower would leave holes a short "last 2 minutes" span can feel."""

RETAIN_SAMPLES = int(4 * 3600 / UPDATE_SECONDS)
"""Four hours of history per source, which is longer than any session. Bounded because unbounded per-frame
appends are how a screen gets slower the longer it runs — the lesson `Target.spread` records."""

TRENDS_HELP = (
    "This shows where each band has been over the chosen span, and its average. The Spectrum tab shows "
    "the last second. This pane shows the session.\n\n"
    "span sets how many minutes of the most recent data to show. 0 means the whole session. Every line, "
    "average and peak frequency on this pane uses the same span.\n\n"
    "Each tile's large number is the band's average amplitude over the span, in the unit chosen above. "
    "The peak x̄ number beneath it is the average of the band's dominant frequency: where inside the "
    "band the energy sat, on average. Pick your reward frequency from this peak.\n\n"
    "relative % divides each band by the total across these bands, before averaging. This stays steady "
    "through the overall amplitude drift that microvolts do not. A band can hold its share even as the "
    "gel dries and everything gets quieter.\n\n"
    "peak Hz redraws the lines as each band's dominant frequency, instead of its amplitude. This is the "
    "trend behind the peak x̄ numbers: it shows the alpha peak move across a session, instead of a "
    "single number. Each line stays inside its own band's edges, so seven lines share one axis without "
    "crossing.\n\n"
    "A sample is taken every 2 s, from the same one-second spectral window the Spectrum tab draws. "
    "Sampling starts at the session's first second, so opening this tab late loses nothing."
)


class BandTrends:
    """History of band amplitudes and dominant frequencies, per spectrum source.

    One sample is `(t, absolute, peaks)` — relative shares are derived from the absolute dict at read
    time, so they cost nothing to store and cannot drift out of step with it.
    """

    def __init__(self, bands=TILE_BANDS, update_seconds: float = UPDATE_SECONDS,
                 retain: int = RETAIN_SAMPLES):
        self.bands = bands
        self.update_seconds = update_seconds
        self.retain = retain
        self._hist: dict[str, deque] = {}
        self._last: float | None = None

    def tick(self, sources, now: float) -> None:
        """Sample every ring that has a second of data. `now` is signal time, so a replay reproduces a
        live session exactly. Rate-limited here rather than by the caller, because the caller is a draw
        loop and how often screens draw is not a property of the history."""
        if self._last is not None and now - self._last < self.update_seconds:
            return
        self._last = now
        for source in sources:
            absolute = source.ring.band_values(self.bands, relative=False)
            if not absolute:
                continue                     # not a second of data yet
            peaks = source.ring.band_peaks(self.bands)
            self._hist.setdefault(source.label, deque(maxlen=self.retain)).append(
                (now, absolute, peaks))

    def labels(self) -> list[str]:
        return list(self._hist)

    def window(self, label: str, span_seconds: float) -> list:
        """The samples the span covers: the most recent `span_seconds`, or everything for 0."""
        hist = self._hist.get(label)
        if not hist:
            return []
        if not span_seconds:
            return list(hist)
        cutoff = hist[-1][0] - span_seconds
        return [s for s in hist if s[0] >= cutoff]

    @staticmethod
    def relative_of(absolute: dict) -> dict:
        total = sum(absolute.values())
        if total <= 0:
            return {k: 0.0 for k in absolute}
        return {k: 100.0 * v / total for k, v in absolute.items()}

    def series(self, label: str, span_seconds: float, relative: bool) -> dict[str, list]:
        """Per band, the trend the graph draws: `(t, value)` pairs in the chosen unit."""
        out: dict[str, list] = {band.label: [] for band in self.bands}
        for now, absolute, _peaks in self.window(label, span_seconds):
            values = self.relative_of(absolute) if relative else absolute
            for key, value in values.items():
                out[key].append((now, value))
        return out

    def peak_series(self, label: str, span_seconds: float) -> dict[str, list]:
        """Per band, the dominant frequency's own trend: `(t, hz)` pairs.

        A first-class series and not a by-product, because it is half of what the pane is for: "did the
        alpha peak drift" is a question about a *line*, and an average alone answers it the way a mean
        temperature answers whether it rained. Each line lives inside its band's own edges, so seven of
        them share one Hz axis without crossing."""
        out: dict[str, list] = {band.label: [] for band in self.bands}
        for now, _absolute, peaks in self.window(label, span_seconds):
            for key, hz in peaks.items():
                out[key].append((now, hz))
        return out

    def averages(self, label: str, span_seconds: float, relative: bool) -> dict[str, float]:
        """Each band's mean amplitude over the span. Relative is the mean of per-sample shares, not the
        share of the means — a share is what each moment looked like, and averaging afterwards keeps a
        quiet minute from being outvoted by a loud one."""
        window = self.window(label, span_seconds)
        if not window:
            return {}
        sums: dict[str, float] = {band.label: 0.0 for band in self.bands}
        for _now, absolute, _peaks in window:
            values = self.relative_of(absolute) if relative else absolute
            for key, value in values.items():
                sums[key] += value
        return {k: v / len(window) for k, v in sums.items()}

    def peak_averages(self, label: str, span_seconds: float) -> dict[str, float]:
        """Each band's mean dominant frequency over the span, in Hz."""
        window = self.window(label, span_seconds)
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for _now, _absolute, peaks in window:
            for key, hz in peaks.items():
                sums[key] = sums.get(key, 0.0) + hz
                counts[key] = counts.get(key, 0) + 1
        return {k: sums[k] / counts[k] for k in sums}


class TrendsPane:
    """Build once, mutate on Flet's loop. `rebuild()` after a setting changes, `refresh()` per frame."""

    def __init__(self, page: ft.Page, session, width: Callable[[int], float],
                 bands=TILE_BANDS, update_seconds: float = UPDATE_SECONDS,
                 retain: int = RETAIN_SAMPLES):
        self.page = page
        self.session = session
        self.width = width
        # Bands and cadence are the screen's to choose: the sleep screen watches sleep's bands over a
        # whole night, which wants a coarser sample and a longer memory than a training hour.
        # **Held by the session, not by this pane**, because a trip to the device screen rebuilds every
        # view against the same session — and an hour of band history is exactly the kind of thing that
        # survives one. See `TrainingSession.band_trends`.
        kept = getattr(session, "band_trends", None)
        if kept is None or kept.bands is not bands:
            kept = BandTrends(bands, update_seconds, retain)
            try:
                session.band_trends = kept
            except AttributeError:
                pass                # a screen whose session is not ours keeps it on the pane
        self.trends = kept
        self.key = ""
        self.span_minutes = 0.0              # 0 = the whole session; the ask names session first
        self.graph_shows = "amplitude"       # or "peak" — which trend the lines draw; see the pills

        self.canvas = cv.Canvas(shapes=[], expand=True, height=TREND_H)
        self.axis, self.axis_labels = value_axis(TREND_H, count=4, width=AXIS_W)
        self.ceiling = StickyCeiling()
        self.caption = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.empty = ft.Text("waiting for data", size=t.SMALL, color=t.FAINT)
        self.source_pills = ft.Row(spacing=6)
        self.unit_pills = ft.Row(spacing=6)
        # no wrap: a wrapped child with no stated width takes a whole line, one band per row
        self.legend = ft.Row(spacing=14)
        self.tiles = ft.Row(spacing=10, wrap=True)
        self.tile_values: dict[str, ft.Text] = {}
        self.tile_peaks: dict[str, ft.Text] = {}

        for band in self.trends.bands:
            self.tile_values[band.label] = ft.Text("—", size=22, color=t.TEXT,
                                                   weight=ft.FontWeight.W_300)
            self.tile_peaks[band.label] = ft.Text("", size=t.SMALL,
                                                  color=t.band_colour(band.low, band.high))
            self.tiles.controls.append(ft.Container(
                content=ft.Row([
                    ft.Container(width=4, bgcolor=t.band_colour(band.low, band.high),
                                 border_radius=2),
                    ft.Container(width=8),
                    ft.Column([ft.Text(band.label, size=t.SMALL, color=t.FAINT),
                               self.tile_values[band.label],
                               self.tile_peaks[band.label],
                               ft.Text(f"{band.low:g}–{band.high:g} Hz", size=t.SMALL,
                                       color=t.FAINT)], spacing=2)], spacing=0),
                width=118, height=104, padding=ft.Padding(10, 10, 10, 12), border_radius=8,
                bgcolor=t.BG, border=ft.Border.all(1, t.BORDER)))
            self.legend.controls.append(
                ft.Row([ft.Container(width=10, height=2, border_radius=1,
                                     bgcolor=t.band_colour(band.low, band.high)),
                        ft.Text(band.label, size=t.SMALL, color=t.MUTED)], spacing=6))

        span_row = ft.Row([
            ft.Text("span", size=t.SMALL, color=t.FAINT),
            num_box(self.span_minutes, self._set_span, suffix="min", step=1, low=0, high=240),
            ft.Text("0 = the whole session", size=t.SMALL, color=t.FAINT)],
            spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        self.control = ft.Container(
            content=panel("TRENDS",
                          ft.Row([self.source_pills, ft.Container(expand=True), span_row,
                                  ft.Container(width=10), self.unit_pills],
                                 vertical_alignment=ft.CrossAxisAlignment.CENTER),
                          ft.Container(height=8),
                          ft.Container(content=self.legend, padding=ft.Padding(0, 0, 0, 8)),
                          ft.Row([self.axis,
                                  ft.Stack([ft.Container(content=self.canvas, expand=True),
                                            ft.Container(content=self.empty, expand=True,
                                                         alignment=ft.Alignment.CENTER)],
                                           expand=True, height=TREND_H)],
                                 spacing=0),
                          ft.Container(height=8), self.caption,
                          ft.Container(height=16), self.tiles, ft.Container(height=6),
                          help=help_button(page, "Trends", TRENDS_HELP)),
            padding=ft.Padding(14, 12, 14, 14), expand=True, visible=False)

        self.rebuild()

    # --- state ------------------------------------------------------------------------------

    def current(self) -> str:
        found = self.trends.labels() or [s.label for s in self.session.spectrum_sources()]
        if not found:
            return ""
        if self.key not in found:
            self.key = found[0]
        return self.key

    def _set_span(self, minutes: float) -> None:
        self.span_minutes = max(0.0, float(minutes))
        self.session.redraw()

    def _pick(self, key: str) -> None:
        self.key = key
        self.rebuild()
        self.session.redraw()

    def _set_relative(self, relative: bool) -> None:
        self.graph_shows = "amplitude"
        self.session.relative = relative
        self.rebuild()
        self.session.redraw()

    def _show_peaks(self) -> None:
        self.graph_shows = "peak"
        self.rebuild()
        self.session.redraw()

    def rebuild(self) -> None:
        found = self.trends.labels() or [s.label for s in self.session.spectrum_sources()]
        self.current()
        self.source_pills.controls = (
            [ft.Text(found[0], size=t.BODY, color=t.TEXT)] if len(found) == 1 else
            [pill(label, label == self.key, (lambda k=label: lambda _e: self._pick(k))())
             for label in found])
        amplitude = self.graph_shows == "amplitude"
        self.unit_pills.controls = [
            pill("absolute µV", amplitude and not self.session.relative,
                 lambda _e: self._set_relative(False)),
            pill("relative %", amplitude and self.session.relative,
                 lambda _e: self._set_relative(True)),
            pill("peak Hz", not amplitude, lambda _e: self._show_peaks())]

    # --- per frame --------------------------------------------------------------------------

    def tick(self, now: float) -> None:
        """Collect, whether or not this tab is showing — a session average must not depend on when the
        operator first opened the pane."""
        self.trends.tick(self.session.spectrum_sources(), now)

    def refresh(self) -> None:
        # Painting is skipped while hidden — collection is `tick`, which never is. See the same guard
        # in `SpectrumPane.refresh` for the cost this avoids.
        if not self.control.visible:
            return
        session = self.session
        relative = session.relative
        peaks_mode = self.graph_shows == "peak"
        unit = "Hz" if peaks_mode else ("%" if relative else "µV")
        label = self.current()
        span = self.span_minutes * 60.0
        series = ({} if not label else
                  self.trends.peak_series(label, span) if peaks_mode else
                  self.trends.series(label, span, relative))
        window = self.trends.window(label, span) if label else []

        w, h = self.width(AXIS_W), float(TREND_H)
        have = len(window) >= 2
        self.empty.visible = not have
        if have:
            t0, t1 = window[0][0], window[-1][0]
            spread = max(t1 - t0, 1e-9)
            peak = max((max(v for _t, v in pts) for pts in series.values() if pts), default=1e-9)
            if peaks_mode:
                # A dominant frequency is clamped inside its band, so the axis's true top is the highest
                # band edge — fixed, no auto ceiling to round 45 Hz of data up to a 100 Hz axis.
                top = max(band.high for band in self.trends.bands)
            else:
                top = self.ceiling.update(peak, time.monotonic())
            shapes = grid(w, h)
            for band in self.trends.bands:
                pts = series.get(band.label) or []
                if len(pts) < 2:
                    continue
                # No more points than pixels — two points on one pixel is one pixel, and a session
                # window holds thousands of samples. Same rule as the targets graph.
                if len(pts) > w:
                    take = np.linspace(0, len(pts) - 1, int(w)).astype(int)
                    pts = [pts[i] for i in take]
                shapes.append(polyline(
                    [ft.Offset((at - t0) / spread * w, h - min(1.0, v / top) * h)
                     for at, v in pts],
                    t.band_colour(band.low, band.high), 1.4))
            self.canvas.shapes = shapes
            place_value_labels(self.axis_labels, h, top, unit=unit)
            minutes = spread / 60.0
            described = (f"the whole session ({minutes:.0f} min)" if not self.span_minutes
                         else f"the last {self.span_minutes:g} min")
            drawn = ("each line is where its band's energy sits, in Hz" if peaks_mode
                     else "each line is a band's amplitude")
            self.caption.value = (
                f"{label} · {described} · 0–{fmt_amplitude(top, unit, True)} · {drawn} · "
                f"averages below are over the same span · sampled every "
                f"{self.trends.update_seconds:g} s from a 1 s window")
        else:
            self.canvas.shapes = []
            for axis_label in self.axis_labels:
                axis_label.value = ""
            self.caption.value = "waiting for data — the first sample lands a second in"

        averages = self.trends.averages(label, span, relative) if label else {}
        peaks = self.trends.peak_averages(label, span) if label else {}
        for band in self.trends.bands:
            value = averages.get(band.label)
            self.tile_values[band.label].value = "—" if value is None else (
                f"{value:.0f}%" if relative else f"{value:.1f}")
            hz = peaks.get(band.label)
            self.tile_peaks[band.label].value = "" if hz is None else f"peak x̄ {hz:.1f} Hz"
