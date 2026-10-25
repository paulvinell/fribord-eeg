"""The quality timeline: where a session went bad, kept apart by *how* it went bad.

The collector is the subject here rather than the picture — what a cell means, that rejected and
unusable never blend, that a hole in the feed is not a clean second, and that downsampling a night to
a few hundred pixels cannot average a dropout away.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfb_console.quality import CELL_SECONDS, CLEAN, QualityLog, _rgb
from nfb_console import theme as t


def feed(log: QualityLog, channel: int, seconds: int, rejected: float = 0.0,
         unusable: bool = False, start: int = 0, samples: int = 250) -> None:
    """One second per call, as the feed would deliver it: `rejected` is the fraction of the second."""
    for second in range(start, start + seconds):
        log.observe(channel, second + 0.999, samples, int(rejected * samples), unusable)


def test_a_cell_carries_the_two_failures_separately():
    log = QualityLog()
    feed(log, 0, 3)                                   # clean
    feed(log, 0, 2, rejected=0.5, start=3)            # the gate firing on half of it
    feed(log, 0, 4, unusable=True, start=5)           # the channel condemned
    cells = log.series(0)
    assert len(cells) == 9
    assert cells[0] == (0.0, 0.0)
    assert cells[3] == pytest.approx((0.5, 0.0)), "rejection does not leak into unusable"
    assert cells[6] == pytest.approx((0.0, 1.0)), "or the other way"

    rejected, unusable, missing = log.totals(0)
    assert rejected == pytest.approx(2 * 0.5 / 9)
    assert unusable == pytest.approx(4 / 9)
    assert missing == 0.0


def test_a_second_the_feed_never_delivered_is_a_hole_not_a_clean_second():
    """An acquisition gap or a stalled feed leaves no blocks at all. Drawing that as healthy would
    hide exactly the failure the timeline exists to show."""
    log = QualityLog()
    feed(log, 0, 2)
    feed(log, 0, 2, start=10)                          # eight seconds simply never arrived
    cells = log.series(0)
    assert np.isnan(cells[3][0]), "the hole is marked, not filled in"
    _rejected, _unusable, missing = log.totals(0)
    assert missing == pytest.approx(8 / 12)


def test_the_span_is_a_worst_case_summary_not_an_average_when_it_is_drawn():
    """A five-second dropout has to survive being drawn two hours wide, so a pixel takes the worst
    cell inside it rather than the mean."""
    log = QualityLog()
    feed(log, 0, 600)                                  # ten clean minutes...
    feed(log, 0, 5, unusable=True, start=600)          # ...with five bad seconds in them
    band = QualityLog._band(log.series(0), width=60)   # one pixel per ten seconds
    bad = [tuple(px) for px in band[0]]
    assert tuple(_rgb(t.BAD)) in bad, "the dropout survives a 10x downsample"
    assert bad.count(tuple(_rgb(CLEAN))) > 50, "and the clean minutes still read clean"


def test_the_worst_minute_says_where_to_look():
    log = QualityLog()
    feed(log, 0, 300)
    feed(log, 0, 60, rejected=0.9, start=300)
    feed(log, 0, 300, start=360)
    at, bad = log.worst_minute(0)
    assert 295 <= at <= 305, "named by when it started, in seconds"
    assert bad > 0.8


def test_the_picture_is_one_image_however_long_the_night():
    """The whole reason this is a PNG: a night is 28800 cells per channel, and cells as canvas shapes
    would be serialised to the client on every frame."""
    log = QualityLog()
    feed(log, 0, 3600)
    feed(log, 1, 3600)
    png = log.image([0, 1], width=400)
    assert png is not None and png.startswith(b"\x89PNG")
    assert len(png) < 20_000, "an hour of two channels is a few kilobytes"
    assert log.image([], width=400) is None
    assert QualityLog().image([0], width=400) is None, "nothing recorded yet draws nothing"


def test_the_trace_marks_which_columns_were_thrown_away_and_costs_one_shape_per_run():
    """The live-window half of the same question. A column is marked if any block feeding it was
    rejected — a blink is shorter than a column — and the marks are drawn as runs, so a trace with two
    blinks in it costs two shapes rather than 320."""
    from nfb_console.plots import mark_shapes
    from nfb_console.traces import TraceBuffer

    buf = TraceBuffer(columns=20, seconds=1.0, rate_hz=200.0)      # 10 samples per column
    quiet = np.zeros(10)
    for column in range(20):
        buf.push(quiet, accepted=column not in (5, 6, 15))
    marks = buf.marks()
    assert marks.sum() == 3 and marks[5] and marks[6] and marks[15]

    shapes = mark_shapes(marks, w=200.0, h=90.0, colour="#a66306")
    assert len(shapes) == 2, "two contiguous runs, not three columns"
    assert mark_shapes(np.zeros(20, dtype=bool), 200.0, 90.0, "#a66306") == [], \
        "a clean trace draws nothing at all"


def test_the_log_runs_on_signal_time_so_a_replay_repeats_exactly():
    log, other = QualityLog(), QualityLog()
    for run in (log, other):
        feed(run, 0, 5, rejected=0.25)
    assert log.series(0) == other.series(0)
    assert log.seconds(0) == 5 * CELL_SECONDS
