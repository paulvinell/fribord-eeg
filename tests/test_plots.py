"""The drawing helpers the screens share: axes, the cascade, the raw trace, the spectrum scale."""

from __future__ import annotations



from harness import struct_unpack


def test_the_cascade_does_not_compress_while_it_fills():
    """The image is stretched to a fixed-height box, so emitting only the rows that had arrived meant the
    picture compressed vertically for the first twenty seconds — every row you had already read kept moving
    and shrinking as the next arrived, which is the one thing a time axis must not do."""

    import numpy as np

    from nfb_console.plots import Chronospectrogram

    sg = Chronospectrogram(rows=90, lo_hz=1.0, hi_hz=40.0)
    sizes = set()
    for step in range(120):
        sg.push(np.full(40, 5.0))
        if step % 17 == 0:
            head = sg.image()[:26]
            width, height = struct_unpack(head)
            sizes.add((width, height))
    assert sizes == {(40, 90)}, f"the picture changed shape as it filled: {sizes}"

    # and the unfilled part reads as the panel's own surface, not as silence at every frequency
    fresh = Chronospectrogram(rows=8, lo_hz=1.0, hi_hz=40.0)
    fresh.push(np.full(40, 5.0))
    assert fresh.seconds_filled == 1 / 8


def test_the_frequency_axis_keeps_its_last_label_on_the_plot():
    """A label centred on the last gridline hangs half its width past the edge and gets clipped by the
    parent, so "40" rendered as "4"."""
    from nfb_console.plots import bin_centre_x, place_tick_labels, tick_axis

    ticks = (5, 10, 20, 40)
    _stack, labels = tick_axis(ticks, tick_width=26)
    width = 740.0
    place_tick_labels(labels, ticks, bin_centre_x(1.0, 1.0, 40, width, top=40.0),
                      tick_width=26, width=width)
    assert all(label.visible for label in labels)
    for label in labels:
        assert 0 <= label.left <= width - 26 + 0.01, (label.value, label.left)


def test_the_raw_trace_is_a_waveform_not_a_filled_block():
    """A filled min/max envelope inks every column's whole excursion, so real EEG drew as one solid mass —
    the information is present and the waveform is not, which is the only thing a raw trace is read for.

    Stroked as a zigzag instead: one polyline that walks down one column and up the next, so it still visits
    both extremes of every column (no spike is decimated away) while the ink follows the signal rather than
    the area under it. Same point count, one shape.
    """
    import flet as ft
    import numpy as np

    from nfb_console.plots import envelope_shapes

    rng = np.random.default_rng(0)
    lo = -20 - rng.random(64) * 5
    hi = 20 + rng.random(64) * 5
    shapes = envelope_shapes(lo, hi, 320.0, 92.0, 100.0, "#000000")
    assert len(shapes) == 1
    paint = shapes[0].paint
    assert paint.style == ft.PaintingStyle.STROKE, "a filled envelope is the fuzzy-worm bug"

    pts = shapes[0].points
    assert len(pts) == 2 * len(lo), "both extremes of every column are still drawn"
    # consecutive points share an x, and the walk alternates direction so it is one continuous stroke
    assert pts[0].x == pts[1].x and pts[2].x == pts[3].x
    assert (pts[0].y < pts[1].y) != (pts[2].y < pts[3].y), "the zigzag does not retrace itself"

    # a quiet channel still draws something rather than vanishing sub-pixel
    flat = np.zeros(64)
    thin = envelope_shapes(flat, flat, 320.0, 92.0, 100.0, "#000000")[0].points
    assert abs(thin[1].y - thin[0].y) >= 1.0


def test_the_spectrum_scale_does_not_strobe():
    """`nice_ceiling(peak)` recomputed every frame is unusable live: a peak wobbling either side of 5 µV
    flips the axis between 5 and 10 several times a second, the bars halve and double, and the picture
    strobes. Right numbers, unreadable display.

    Rises at once, because a clipped bar is information lost. Falls only after the peak has stayed
    *comfortably* below for a few seconds — hysteresis and a delay, since either alone still flickers.
    """
    from nfb_console.plots import StickyCeiling, nice_ceiling

    # the pathological case: a peak sitting right on a step boundary
    naive = {nice_ceiling(v) for v in (4.9, 5.1, 4.8, 5.2)}
    assert len(naive) > 1, "the premise: the naive ceiling moves here"

    sticky = StickyCeiling(patience=4.0)
    now = 0.0
    for i in range(100):                      # let it settle out of its initial guess
        now += 0.1
        sticky.update(5.0 + (0.3 if i % 2 else -0.3), now)
    seen = set()
    for i in range(400):
        now += 0.1
        seen.add(sticky.update(5.0 + (0.3 if i % 2 else -0.3), now))
    assert len(seen) == 1, f"it strobed anyway: {sorted(seen)}"
    assert seen.pop() >= 5.3, "and whatever it settled on must still hold the peak"

    # rises immediately, because clipping loses the peak you are looking for
    assert sticky.update(40.0, now) >= 40.0
    risen = sticky.value

    # and does not fall until the peak has been below for the full patience
    now += 1.0
    assert sticky.update(2.0, now) == risen, "falling instantly is the other half of the flicker"
    for _ in range(60):
        now += 0.1
        sticky.update(2.0, now)
    assert sticky.value < risen, "it has to come back down eventually"
    assert sticky.value >= 2.0, "but never below the peak it is showing"

    # a peak that keeps touching the ceiling holds it there indefinitely
    high = StickyCeiling(patience=1.0)
    high.update(9.0, 0.0)
    held = high.value
    for i in range(50):
        assert high.update(9.0, 1.0 + i) == held, "a peak at the top must not trip a step down"
