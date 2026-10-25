"""The Spectrum tab, shared by every protocol screen that has one.

Lifted out of `amplitude_view` when the second and third screens needed it. Nothing here knows what is
being trained — it takes a list of `SpectrumSource`s from the session and draws whichever one is selected —
which is the test for belonging in the shared layer.

**Every screen with a band on it needs this, and two did not have it.** Alpha-theta is read by watching
two amplitudes cross; cross-frequency coupling is explicit that the reward frequency is *found* on the
spectral display — "often within 1-2 Hz of the frequency in the spectral display showing the highest
amplitude in a particular band" — and neither screen offered a spectrum to find it on. The cascade on the
CFC screen shows *change over time* at one glance; this shows the instantaneous shape with numbers on both
axes, and they answer different questions.

Three details are decisions rather than layout:

- **A bin is centred on its frequency**, and the gridlines pass through bin centres. Binning `[f, f+1)`
  and drawing from f's gridline rightward puts every feature half a bin late, which is visible: a notch at
  50 Hz looks as though it starts there and continues past it.
- **Axis labels are controls placed absolutely**, centred on the line they name, not a row of cells — a
  right-aligned cell puts each number half its own width to the left of its gridline, and the eye reads
  that as the *bars* being shifted right.
- **The auto scale is sticky** (`plots.StickyCeiling`), and a fixed scale sits beside it. Auto is what you
  want before you know the range; a fixed axis is the only way to compare this minute with the last.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb.dsp import ALPHA, DELTA, SMR, THETA, Band

from . import theme as t
from .plots import (StickyCeiling, bars, bin_centre_x, fmt_amplitude, gridlines, place_tick_labels,
                    place_value_labels, polyline, tick_axis, value_axis)
from .traces import RawRing
from .widgets import help_button, panel, pill

SPECTRUM_H = 250
AXIS_W = 40                                        # room for amplitude labels left of the plot
TICK_W = 30                                        # a frequency label, centred on its gridline
TICK_STEP_HZ = 10.0

# Full-scale choices, µV. `auto` is a *sticky* auto — see `plots.StickyCeiling` — and the fixed values
# exist because an axis that holds still is the only way to compare one minute with the next, or this
# session with yesterday's.
SCALES = (0.0, 5.0, 10.0, 20.0, 50.0)

# One tile per conventional band. Gamma is 38–42 Hz, clear of 50 Hz mains and its notch's residue.
TILE_BANDS = (DELTA, THETA, ALPHA, SMR, Band(15, 20, "beta1"), Band(20, 30, "beta2"),
              Band(38, 42, "gamma"))

SPECTRUM_HELP = (
    "This shows the frequency content of the last second, in 1 Hz bins. Both axes have a number.\n\n"
    "Each bin is centred on its frequency and takes its colour from that frequency. A band shows the "
    "same colour here as on its card and in the tiles below.\n\n"
    "auto follows the signal. Use it before you know the range. It falls slowly on purpose, because an "
    "axis that flips between two steps while the peak moves is hard to read. A fixed scale lets you "
    "compare one minute with the next.\n\n"
    "relative % divides each bin by the total. This stays steady through the overall amplitude drift "
    "that microvolts do not. A band can hold its share even while everything gets quieter.\n\n"
    "Each tile shows two numbers that answer different questions. The large number is how much of that "
    "band is present, averaged across it. The \"peak\" number beneath it shows where inside the band "
    "the energy sits: the dominant frequency. Pick your reward frequency from this peak, instead of "
    "assuming the middle of the band. The peak is interpolated between bins, so it moves in tenths of a "
    "hertz, even though the bars are 1 Hz wide."
)


@dataclass
class SpectrumSource:
    """One thing whose spectrum can be looked at.

    `channel` is set where the source *is* a channel, so selecting it can move the session's focus — on
    the amplitude screen that also decides which channel a new target lands on. A sum has no channel and
    must not move it, which is why this is a field rather than a label to be parsed.
    """

    label: str
    ring: RawRing
    channel: int | None = None


class SpectrumPane:
    """Build once, mutate on Flet's loop. `rebuild()` after a setting changes, `refresh()` per frame."""

    def __init__(self, page: ft.Page, session, width: Callable[[int], float]):
        self.page = page
        self.session = session
        self.width = width
        self.key = ""

        self.canvas = cv.Canvas(shapes=[], expand=True, height=SPECTRUM_H)
        self.axis, self.amp_labels = value_axis(SPECTRUM_H, count=5, width=AXIS_W)
        self.ticks = self._ticks()
        self.scale_row, self.hz_labels = tick_axis(self.ticks, tick_width=TICK_W)
        self.caption = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.source_pills = ft.Row(spacing=6)
        self.scale_pills = ft.Row(spacing=6)
        self.unit_pills = ft.Row(spacing=6)
        self.ceiling = StickyCeiling()
        self.tiles = ft.Row(spacing=10, wrap=True)
        self.tile_values: dict[str, ft.Text] = {}
        # Where the energy sits inside the band, under how much of it there is. See `RawRing.band_peaks`
        # and `PEAK_HELP`: the amplitude answers "how much alpha", this answers "which alpha", and the
        # second question is the one a reward frequency is chosen from.
        self.tile_peaks: dict[str, ft.Text] = {}

        for band in TILE_BANDS:
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

        self.control = ft.Container(
            content=panel("SPECTRUM",
                          ft.Row([self.source_pills, ft.Container(expand=True), self.scale_pills,
                                  ft.Container(width=10), self.unit_pills],
                                 vertical_alignment=ft.CrossAxisAlignment.CENTER),
                          ft.Container(height=12),
                          ft.Row([self.axis,
                                  ft.Container(content=self.canvas, expand=True, height=SPECTRUM_H)],
                                 spacing=0),
                          ft.Container(content=self.scale_row, padding=ft.Padding(AXIS_W, 2, 0, 0)),
                          ft.Container(height=8), self.caption,
                          ft.Container(height=16), self.tiles, ft.Container(height=6),
                          help=help_button(page, "Spectrum", SPECTRUM_HELP)),
            padding=ft.Padding(14, 12, 14, 14), expand=True, visible=False)

        self.rebuild()

    # --- sources ----------------------------------------------------------------------------

    def sources(self) -> list[SpectrumSource]:
        return list(self.session.spectrum_sources())

    def _ticks(self) -> tuple[int, ...]:
        """Every 10 Hz up to whatever the widest source reaches.

        Derived rather than a per-screen constant: CFC reads to 40 Hz and the amplitude screen to 60 so
        that mains is visible, and a screen that had to state its own tick list would be one more place
        for the axis to disagree with the data.
        """
        top = max((s.ring.top_hz for s in self.sources()), default=60.0)
        return tuple(int(hz) for hz in np.arange(TICK_STEP_HZ, top + 1e-9, TICK_STEP_HZ))

    def current(self) -> SpectrumSource | None:
        found = self.sources()
        if not found:
            return None
        for source in found:
            if source.label == self.key:
                return source
        self.key = found[0].label
        return found[0]

    # --- controls ---------------------------------------------------------------------------

    def rebuild(self) -> None:
        session, found = self.session, self.sources()
        self.current()
        # One source is *stated* rather than offered: a row containing a single pill reads as a setting
        # whose other options have gone missing.
        self.source_pills.controls = (
            [ft.Text(found[0].label, size=t.BODY, color=t.TEXT)] if len(found) == 1 else
            [pill(s.label, s.label == self.key, (lambda k=s.label: lambda _e: self._pick(k))())
             for s in found])
        self.scale_pills.controls = [
            pill("auto" if uv == 0 else f"{uv:g}", session.spectrum_scale == uv,
                 (lambda v=uv: lambda _e: self._set_scale(v))())
            for uv in SCALES]
        self.unit_pills.controls = [
            pill("absolute µV", not session.relative, lambda _e: self._set_relative(False)),
            pill("relative %", session.relative, lambda _e: self._set_relative(True))]

    def _pick(self, key: str) -> None:
        self.key = key
        source = self.current()
        # Selecting a channel moves the session's focus, since on the amplitude screen that is also which
        # channel a new target lands on. A sum is not a channel and leaves it alone.
        if source is not None and source.channel is not None:
            self.session.focus = source.channel
        self.rebuild()
        self.session.redraw()

    def _set_scale(self, uv: float) -> None:
        self.session.spectrum_scale = uv
        self.rebuild()
        self.session.redraw()

    def _set_relative(self, relative: bool) -> None:
        self.session.relative = relative
        self.rebuild()
        self.session.redraw()

    # --- drawing ----------------------------------------------------------------------------

    def refresh(self) -> None:
        # A hidden pane must cost nothing: mutated controls are serialised whether or not they show,
        # and invisible panes repainted at 10 Hz starved the event loop that delivers clicks.
        if not self.control.visible:
            return
        session = self.session
        unit = "%" if session.relative else "µV"
        source = self.current()
        ring = source.ring if source is not None else None
        values = ring.spectrum(session.relative) if ring is not None else np.zeros(0)
        if values.size and ring is not None:
            sw = self.width(AXIS_W)
            top = (session.spectrum_scale if session.spectrum_scale
                   else self.ceiling.update(float(values.max()), time.monotonic()))
            # every bin coloured by its own frequency, so a band is the same colour here as on its card
            # and in the tiles — colour follows centre frequency everywhere
            shapes = bars(values, sw, SPECTRUM_H,
                          lambda i: t.frequency_colour(ring.lo_hz + i * ring.bin_hz),
                          peak=top)
            for frac in (0.25, 0.5, 0.75):
                y = SPECTRUM_H * (1 - frac)
                shapes.append(polyline([ft.Offset(0, y), ft.Offset(sw, y)], t.BORDER, 0.6))
            x_of = bin_centre_x(ring.lo_hz, ring.bin_hz, len(values), sw, top=ring.top_hz)
            shapes += gridlines(self.ticks, x_of, SPECTRUM_H)
            self.canvas.shapes = shapes
            place_value_labels(self.amp_labels, SPECTRUM_H, top, unit=unit)
            place_tick_labels(self.hz_labels, self.ticks, x_of, tick_width=TICK_W)
            self.caption.value = (
                f"{source.label} · {ring.lo_hz:g}–{ring.top_hz:g} Hz in {ring.bin_hz:g} Hz bins · "
                f"0–{fmt_amplitude(top, unit, True)} "
                f"{'full scale' if session.spectrum_scale else 'auto scale'} · "
                f"peak {values.max():.1f} {unit} at "
                f"{ring.lo_hz + values.argmax() * ring.bin_hz:.0f} Hz · 1 s window"
                + ("" if ring.top_hz >= ring.hi_hz
                   else f" · capped at Nyquist for {ring.rate_hz:g} sps"))
        else:
            self.canvas.shapes = []
            for label in self.hz_labels:
                label.visible = False
            for label in self.amp_labels:
                label.value = ""
            self.caption.value = "waiting for a second of data"
        band_values = ring.band_values(TILE_BANDS, session.relative) if ring is not None else {}
        band_peaks = ring.band_peaks(TILE_BANDS) if ring is not None else {}
        for band in TILE_BANDS:
            value = band_values.get(band.label)
            self.tile_values[band.label].value = "—" if value is None else (
                f"{value:.0f}%" if session.relative else f"{value:.1f}")
            peak = band_peaks.get(band.label)
            # Two decimals below 1 Hz, one above: the infra-slow bands are whole decades narrower than
            # the fast ones, so a single figure there would report every frequency in the band as the
            # same number.
            self.tile_peaks[band.label].value = "" if peak is None else (
                f"peak {peak:.2f} Hz" if peak < 1.0 else f"peak {peak:.1f} Hz")
