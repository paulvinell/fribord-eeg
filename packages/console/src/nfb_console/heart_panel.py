"""The heart instruments: a tachogram, a spectrum with the target marked, and the split warning.

One implementation, two sizes, because the same things are watched whether HRV is the session or is
running beside the EEG. `compact=True` is what sits on the CFC screen — the spectrum, the numbers and the
warning, without the tachogram taking a third of the height of a screen whose subject is the EEG.

Two decisions worth stating:

- **The target is a band, not a line.** ±0.02 Hz by default, drawn as a region. A hairline would claim a
  precision nobody can breathe to, and the tolerance is what makes "in it" achievable.
- **And the target is read off the spectrum, never from the module.** It is the operator's setting — 0.1 Hz
  is the population figure, not a given person's resonance — so the marked band, the line, the tile's
  caption and the percentage in it all come from the one `Spectrum` being drawn. Taken from a constant
  instead, they would disagree with each other the moment the setting moved, and a tile reading "at 0.1 Hz"
  over a band drawn at 0.065 is the sort of thing that reads as the instrument being broken.
- **The split is named in words**, not left to the shape. It is the failure that looks like success on
  every scalar measure, so it gets a sentence and a colour rather than a subtle asymmetry in a curve —
  and it is the *only* sentence here. Everything else this panel used to say above and below the plot
  restated what the curve was already showing, changing ten times a second while it did so. A spectrum
  with an axis, a marked band and a line at the target is the reading.
"""

from __future__ import annotations

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb.hrv.beats import TARGET_HZ

from . import theme as t
from .plots import grid, nice_ceiling, place_tick_labels, place_value_labels, polyline, tick_axis, \
    value_axis

SPECTRUM_HI_HZ = 0.4
SPECTRUM_TICKS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
TACHO_SECONDS = 120.0
AXIS_W = 46
TICK_W = 34


class HeartPanel:
    """Build once, `refresh(runner)` per frame."""

    def __init__(self, compact: bool = False, spectrum_height: int = 210,
                 tacho_height: int = 150, width_of=lambda: 520.0):
        self.compact = compact
        self.spectrum_h = 120 if compact else spectrum_height
        self.tacho_h = tacho_height
        self.width_of = width_of

        # Empty and hidden unless the split fires: an empty Text still occupies its line, and a blank
        # gap above the plot is the sort of thing that reads as a control that failed to load.
        self.headline = ft.Text("", size=17 if compact else 20, color=t.BAD,
                                weight=ft.FontWeight.W_300, visible=False)
        self.spec_canvas = cv.Canvas(shapes=[], expand=True, height=self.spectrum_h)
        self.spec_axis, self._spec_labels = value_axis(self.spectrum_h, count=3 if compact else 4,
                                                      width=AXIS_W)
        self.spec_scale, self._spec_ticks = tick_axis(SPECTRUM_TICKS, tick_width=TICK_W)
        self.spec_caption = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.source_note = ft.Text("", size=t.SMALL, color=t.MUTED)

        self.numbers: dict[str, ft.Text] = {}
        # The third tile's own title moves with the target, so it is kept rather than built and
        # forgotten: "at 0.1 Hz" over a percentage measured at 0.065 would be a wrong instrument.
        self.target_title = ft.Text(f"at {TARGET_HZ:g} Hz", size=t.SMALL, color=t.FAINT)
        tiles = ft.Row(spacing=10, wrap=True)
        for label, unit, title in (("heart rate", "bpm", None), ("RMSSD", "ms", None),
                                   ("target", "of the power", self.target_title)):
            value = ft.Text("—", size=17, color=t.TEXT, weight=ft.FontWeight.W_300)
            self.numbers[label] = value
            head = title if title is not None else ft.Text(label, size=t.SMALL, color=t.FAINT)
            tiles.controls.append(ft.Container(
                content=ft.Column([head, value, ft.Text(unit, size=t.SMALL, color=t.FAINT)],
                                  spacing=1),
                width=94, height=70, padding=ft.Padding(10, 9, 10, 9), border_radius=8,
                bgcolor=t.BG, border=ft.Border.all(1, t.BORDER)))
        self.tiles = tiles

        self.tacho_canvas = cv.Canvas(shapes=[], expand=True, height=tacho_height)
        self.tacho_axis, self._tacho_labels = value_axis(tacho_height, count=4, width=AXIS_W)
        self.tacho_caption = ft.Text("", size=t.SMALL, color=t.FAINT)

    # --- the pieces a screen arranges --------------------------------------------------------

    def spectrum_rows(self) -> list[ft.Control]:
        return [
            self.headline,
            ft.Container(height=8),
            ft.Row([self.spec_axis,
                    ft.Container(content=self.spec_canvas, expand=True, height=self.spectrum_h)],
                   spacing=0),
            ft.Container(content=self.spec_scale, padding=ft.Padding(AXIS_W, 2, 0, 0)),
            ft.Container(height=6), self.spec_caption,
        ]

    def tacho_rows(self) -> list[ft.Control]:
        return [
            ft.Row([self.tacho_axis,
                    ft.Container(content=self.tacho_canvas, expand=True, height=self.tacho_h)],
                   spacing=0),
            ft.Container(height=6), self.tacho_caption,
        ]

    # --- per frame ---------------------------------------------------------------------------

    def refresh(self, runner) -> None:
        self.source_note.value = runner.describe()
        self.source_note.color = t.BAD if runner.error else t.MUTED
        state = runner.state
        if state is None:
            self.headline.value = ""
            self.headline.visible = False
            self.headline.visible = False
            self.spec_caption.value = "waiting for beats"
            self.tacho_caption.value = "waiting for beats"
            self.spec_canvas.shapes = []
            self.tacho_canvas.shapes = []
            for label in self.numbers.values():
                label.value = "—"
            return

        self.numbers["heart rate"].value = f"{state.hr_smoothed:.0f}"
        self.numbers["RMSSD"].value = f"{state.rmssd_ms:.0f}"
        # **From the first spectrum, and a dash only when there is none.** This showed a dash until the
        # window could resolve the target — forty seconds, more at a lower one — on the argument that the
        # share reads near zero while the target falls between two bins. It does, and that is a real
        # reading of a window this short: it climbs off the floor as the bins arrive, which is visible
        # and instructive, where a dash for the first minute of every session is neither. The span and
        # the bin width are under the curve throughout, which is where the caveat belongs.
        self.target_title.value = f"at {state.spectrum.target_hz:g} Hz"
        self.numbers["target"].value = (f"{state.spectrum.sharpness:.0%}"
                                        if state.spectrum.ready else "—")
        self._draw_spectrum(state)
        if not self.compact:
            self._draw_tachogram(state)

    def _draw_spectrum(self, state) -> None:
        spectrum = state.spectrum
        w = self.width_of()
        h = self.spectrum_h
        if not spectrum.ready:
            self.spec_canvas.shapes = []
            place_tick_labels(self._spec_ticks, SPECTRUM_TICKS, lambda _hz: None,
                              tick_width=TICK_W)
            for label in self._spec_labels:
                label.value = ""
            self.headline.value = ""
            self.headline.visible = False
            self.spec_caption.value = "waiting for beats"
            return

        keep = spectrum.freqs <= SPECTRUM_HI_HZ
        freqs, power = spectrum.freqs[keep], spectrum.power[keep]
        top = nice_ceiling(float(power.max()) or 1e-9)

        def x_of(hz: float):
            if hz < 0 or hz > SPECTRUM_HI_HZ:
                return None
            return hz / SPECTRUM_HI_HZ * w

        band = spectrum.narrow_band
        lo, hi = (x_of(band[0]) or 0.0), (x_of(band[1]) or 0.0)
        shapes: list = [cv.Rect(lo, 0, hi - lo, h, paint=ft.Paint(color=t.SURFACE_HI))]
        shapes += grid(w, h, rows=3, cols=8)
        gain = (h - 2) / top
        shapes.append(polyline(
            [ft.Offset(x_of(f) or 0.0, h - min(h - 1, p * gain)) for f, p in zip(freqs, power)],
            t.BAD if spectrum.split else t.ACCENT, 1.8))
        target = x_of(spectrum.target_hz) or 0.0
        shapes.append(polyline([ft.Offset(target, 0), ft.Offset(target, h)], t.GOOD, 1.0))
        self.spec_canvas.shapes = shapes
        place_value_labels(self._spec_labels, h, top, unit="")
        place_tick_labels(self._spec_ticks, SPECTRUM_TICKS, x_of, tick_width=TICK_W)

        # **A spectrum is a spectrum.** This carried a sentence of commentary above it and a line of
        # findings below — the peak and its frequency, the share of power in the band, the distance to an
        # ideal, an instruction to keep breathing — all of it restating what the curve already shows to
        # anyone looking at it, and all of it changing every tenth of a second. The plot has an axis, a
        # marked target band and a line at 0.1 Hz; that is the reading. The numbers that are worth having
        # as numbers are in the tiles beside it, once each.
        #
        # Three things survive. The span, because a curve from ten seconds and one from three minutes look
        # identical and are not. The target, because the green line and the shaded band are the only marks
        # here that are a *setting* rather than the data, and an unlabelled line invites being read as a
        # property of the signal. And the split warning, in red, because it is the one thing on this screen
        # a person cannot see by looking — a shape that reads as success on every scalar measure.
        self.headline.value = spectrum.describe() if spectrum.split else ""
        self.headline.visible = bool(self.headline.value)
        self.spec_caption.value = (
            f"0–{SPECTRUM_HI_HZ:g} Hz · {spectrum.span_s:.0f} s of beats, "
            f"{spectrum.resolution_hz:.3f} Hz per bin · aiming at {spectrum.target_hz:g} Hz "
            f"±{spectrum.narrow_half_hz:g}")

    def _draw_tachogram(self, state) -> None:
        ages = np.asarray(state.age_s)
        values = np.asarray(state.tachogram)
        keep = ages <= TACHO_SECONDS
        ages, values = ages[keep], values[keep]
        if values.size < 2:
            self.tacho_canvas.shapes = []
            self.tacho_caption.value = "waiting for beats"
            return
        w, h = self.width_of(), self.tacho_h
        lo_bpm = float(values.min()) - 2
        hi_bpm = float(values.max()) + 2
        span = max(hi_bpm - lo_bpm, 1e-9)
        points = [ft.Offset(w * (1.0 - a / TACHO_SECONDS),
                            h - (v - lo_bpm) / span * (h - 2))
                  for a, v in zip(ages, values)]
        self.tacho_canvas.shapes = grid(w, h, rows=4, cols=8) + [
            polyline(sorted(points, key=lambda p: p.x), t.frequency_colour(10.0), 1.6)]
        place_value_labels(self._tacho_labels, h, hi_bpm, bottom=lo_bpm, unit="bpm")
        swing = float(values.max() - values.min())
        # The reject figure is the one signal-quality number this protocol has, and two docstrings and
        # the protocol doc promised it here for as long as nothing rendered it. It matters because this
        # screen reads a *spectrum*: a heavily filtered tachogram and a clean one look identical by eye,
        # and a spectrum degraded by dropouts arrives with rejection at zero — so the figure is drawn
        # beside the signal it applies to, where a reader can weigh one against the other.
        rejected = (f" · {state.reject_fraction:.0%} of beats rejected"
                    if state.reject_fraction > 0 else "")
        self.tacho_caption.value = (
            f"{TACHO_SECONDS:g} s across · {lo_bpm:.0f}–{hi_bpm:.0f} bpm · "
            f"{swing:.0f} bpm of swing, which is the depth of the modulation · "
            f"{state.beats} beats held{rejected}")
