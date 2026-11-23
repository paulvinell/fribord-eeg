"""The level plot: each input's standing level against its own limits, from the first reading."""

from __future__ import annotations

import numpy as np

from nfb.stream import Block, ChannelSpec
from nfb_console.level_plot import LevelHistory, LevelPlot, limit_uv

from conftest import counts

RATE = 250.0
W, H = 300.0, 120


def _spec(channel: int, input_range: float | None = 800.0) -> ChannelSpec:
    return ChannelSpec(index=channel, rate_hz=RATE, unit="µV", scale=800.0 / 32767, input_range=input_range)


def _push(history: LevelHistory, spec: ChannelSpec, start: int, uv: np.ndarray) -> None:
    history.push(Block(spec.index, start, tuple(counts(uv, spec.scale).tolist()), spec))


def _plot(history: LevelHistory, channels=(0, 1)) -> LevelPlot:
    plot = LevelPlot(history, lambda: list(channels), lambda: W + 52, height=H)
    plot.refresh()
    return plot


def _y(plot: LevelPlot, uv: float) -> float:
    top = float(plot.labels[0].value.split()[0])
    return H / 2 - uv / top * (H / 2)


def _strokes(plot: LevelPlot) -> list:
    return [s for s in plot.canvas.shapes if hasattr(s, "points")]


def _bands(plot: LevelPlot) -> list[list]:
    """The filled bands, each as its outline's points."""
    return [[(e.x, e.y) for e in s.elements if hasattr(e, "x")] for s in plot.canvas.shapes
            if not hasattr(s, "points")]


def _lines_at(plot: LevelPlot) -> set[int]:
    """The heights of the dashed lines, rounded to the pixel."""
    return {round(s.points[0].y) for s in _strokes(plot)
            if len(s.points) == 2 and s.points[0].y == s.points[1].y and s.points[1].x - s.points[0].x < W / 2}


def test_one_reading_draws_a_curve_and_the_limits_are_lines_named_in_microvolts():
    history = LevelHistory()
    _push(history, _spec(0), 0, np.full(8, -300.0))
    plot = _plot(history, (0,))
    curves = [s for s in _strokes(plot) if s.paint.color != "#a66306" and len(s.points) >= 2
              and s.points[0].y != H / 2]
    assert curves, "a curve from the first block"
    assert abs(curves[-1].points[-1].y - _y(plot, -300.0)) < 1.0
    limit = limit_uv(800.0)
    assert {round(_y(plot, limit)), round(_y(plot, -limit))} <= _lines_at(plot)
    assert all(0 < y < H for y in _lines_at(plot)), "the limit lines sit inside the plot"
    named = {label.value for label in plot.limit_labels if label.visible}
    assert named == {f"limit +{limit:.0f} µV", f"limit -{limit:.0f} µV"}, named


def test_a_source_with_no_input_range_has_no_limit_lines_and_scales_to_the_data():
    history = LevelHistory()
    _push(history, _spec(0, None), 0, np.full(int(RATE), 40.0))
    plot = _plot(history, (0,))
    assert not any(label.visible for label in plot.limit_labels)
    top = float(plot.labels[0].value.split()[0])
    assert 40.0 < top <= 100.0, top


def test_each_channel_has_its_own_curve_and_name_and_a_step_back_starts_it_again():
    history = LevelHistory()
    for ch in (0, 1):
        _push(history, _spec(ch), 0, np.full(int(4 * RATE), 100.0 * (ch + 1)))
    plot = _plot(history)
    names = [row.controls[1].value for row in plot.legend.controls]
    assert names == ["ch0", "ch1"]
    colours = {row.controls[0].bgcolor for row in plot.legend.controls}
    assert len(colours) == 2
    _push(history, _spec(0), 0, np.full(int(RATE), -50.0))
    points = history.series()[0]
    assert len(points) == 1 and abs(points[0][1] + 50.0) < 0.1, "a step back starts the channel again"


def test_the_axis_labels_have_one_form():
    history = LevelHistory()
    _push(history, _spec(0), 0, np.full(8, 10.0))
    values = [label.value for label in _plot(history, (0,)).labels]
    assert all(v.endswith(" µV") and "." not in v for v in values), values


def _held(history: LevelHistory) -> str:
    """The line under the plot while the calibration waits on ch0 at the input limit."""
    from types import SimpleNamespace

    from nfb_console.eye_calibration import EyeCalibration
    from nfb_console.scp_view import held_line

    cal = EyeCalibration(RATE, 0, ())
    cal.held = "ch0 is at the input limit"
    return held_line(cal, SimpleNamespace(levels=history, LIMIT_CLEAR_SECONDS=5.0))


def test_one_sample_at_the_limit_shows_in_the_band_and_names_the_channel_for_5_s_of_samples():
    """A level at -450 µV with one sample at the limit: the band reaches the limit line where it fell, and the
    line names the channel; 5 s of samples later it says every channel is between the lines."""
    history = LevelHistory()
    spec = _spec(0)
    level = np.full(int(4 * RATE), -450.0)
    level[int(2 * RATE)] = -800.0
    _push(history, spec, 0, level)
    plot = _plot(history, (0,))
    lowest = max(y for band in _bands(plot) for _x, y in band)
    assert lowest >= _y(plot, -limit_uv(800.0)) - 0.5, "the band reaches the limit line"
    assert "ch0 is at the input limit" in _held(history)
    _push(history, spec, len(level), np.full(int(5 * RATE), -450.0))
    assert _held(history).startswith("all channels are between the limit lines")


def test_without_a_sample_at_the_limit_the_line_never_names_the_channel():
    history = LevelHistory()
    for k in range(20):
        _push(history, _spec(0), k * int(RATE), np.full(int(RATE), -450.0 - 10 * k))
        assert _held(history).startswith("all channels are between the limit lines")
