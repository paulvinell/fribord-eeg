"""The heart panel: the spectrum of the beats and the heart rate, drawn on the canvas they are read from."""

from __future__ import annotations

import flet.canvas as cv
import numpy as np

from nfb.hrv import Beat, HRVProcessor
from nfb_console import theme
from nfb_console.heart import HeartRunner
from nfb_console.heart_panel import HeartPanel

WIDTH = 400.0


def _drawn_spectrum():
    proc = HRVProcessor()
    t, state = 1000.0, None
    while t < 1200.0:
        rr = 1.0 + 0.09 * np.sin(2 * np.pi * 0.1 * t)
        t += rr
        state = proc.push(Beat(t, 60.0 / rr, [rr]))
    runner = HeartRunner()
    runner.state = state
    panel = HeartPanel(width_of=lambda: WIDTH)
    panel.refresh(runner)
    return panel


def test_the_spectrum_curve_spans_the_plot_and_its_peak_stands_inside_it():
    """The curve runs across the width of the plot, rises and falls inside its height, and its peak is in
    the upper half."""
    panel = _drawn_spectrum()
    h = panel.spectrum_h
    curve = max((sh for sh in panel.spec_canvas.shapes if isinstance(sh, cv.Points)), key=lambda sh: len(sh.points))
    xs, ys = [pt.x for pt in curve.points], [pt.y for pt in curve.points]
    assert xs == sorted(xs) and xs[-1] - xs[0] > WIDTH / 2
    assert all(0.0 <= y <= h for y in ys)
    assert min(ys) < h / 2 and max(ys) - min(ys) > h / 4


def test_the_shaded_band_is_centred_on_the_target_line():
    """The band around the target is drawn on both sides of the target line, at equal distance."""
    panel = _drawn_spectrum()
    band = next(sh for sh in panel.spec_canvas.shapes if isinstance(sh, cv.Rect))
    line = next(sh for sh in panel.spec_canvas.shapes if isinstance(sh, cv.Points)
                and sh.paint.color == theme.GOOD and len(sh.points) == 2)
    x = line.points[0].x
    assert band.x < x < band.x + band.width
    assert abs(band.x + band.width / 2 - x) < 1.0
