"""The shapes every protocol screen draws, and the axes that make them readable.

Moved out of `amplitude_view.py` alongside `widgets.py`. Two rules are enforced here
rather than restated at each call site:

- **A filled shape needs `cv.Path`.** `cv.Points` with `PointMode.POLYGON` strokes and silently ignores
  `PaintingStyle.FILL`, so a min/max envelope came out as its own outline — two parallel lines that read
  as two traces.
- **An axis label is centred on the line it names, positioned absolutely.** Laying labels out as a Row
  of right-aligned cells puts each number half its own width left of its gridline, and the eye reads
  that as the bars being shifted right.

The chronospectrogram is rendered as a **PNG rather than as canvas rectangles**, and that is a
consequence of arithmetic rather than taste: 60 frequency bins over 64 time rows is 3840 cells, and
rebuilding that many `cv.Rect` shapes several times a second is not something to ask of a UI toolkit
when the same picture is one `ft.Image` whose bytes we can write directly. The encoder is 30 lines of
`zlib` and `struct` because a spectrogram is not worth a Pillow dependency.
"""

from __future__ import annotations

import struct
import zlib
from collections import deque

import flet as ft
import flet.canvas as cv
import numpy as np

from . import theme as t

MIN_TRACE_PX = 1.4                  # thinnest an envelope may draw, or a quiet channel vanishes


# --- shapes -------------------------------------------------------------------------------

def polyline(points, colour: str, width: float = 1.2) -> cv.Points:
    return cv.Points(points=points, point_mode=cv.PointMode.POLYGON,
                     paint=ft.Paint(color=colour, stroke_width=width,
                                    style=ft.PaintingStyle.STROKE))


def filled(points, colour: str) -> cv.Path:
    """A closed shape, actually filled. See the module docstring for why this is not `cv.Points`."""
    if not points:
        return cv.Path(elements=[], paint=ft.Paint(color=colour))
    elements = [cv.Path.MoveTo(points[0].x, points[0].y)]
    elements += [cv.Path.LineTo(pt.x, pt.y) for pt in points[1:]]
    elements.append(cv.Path.Close())
    return cv.Path(elements=elements,
                   paint=ft.Paint(color=colour, style=ft.PaintingStyle.FILL))


def signed_shapes(series, w: float, h: float, full_scale: float, colours,
                  pushed: int | None = None) -> list:
    """Several **signed** series on one centred axis, right-anchored like `EnvelopePlot.shapes`.

    Separate from `EnvelopePlot` because that one draws from the bottom, which is right for an amplitude and
    wrong for a waveform: an infra-slow signal swings either side of its own baseline and *which* side is
    the information. Drawing it from the bottom would put the baseline at the bottom of the plot and make a
    negative excursion indistinguishable from a quiet one.

    `pushed` is how many columns hold real data; the rest are left undrawn, for the reason `EnvelopePlot`
    gives — a flat line along the middle is a measurement of zero, not an empty window.
    """
    out: list = list(grid(w, h, rows=4, cols=6))
    mid = h / 2
    out.append(polyline([ft.Offset(0, mid), ft.Offset(w, mid)], t.MUTED, 0.8))
    gain = (h / 2 - 1) / max(full_scale, 1e-9)
    for values, colour in zip(series, colours):
        data = np.asarray(values, dtype=float)
        if len(data) < 2:
            continue
        first = 0 if pushed is None else max(0, len(data) - pushed)
        if len(data) - first < 2:
            continue
        step = w / max(1, len(data) - 1)
        ys = mid - np.clip(data[first:] * gain, -(h / 2 - 1), h / 2 - 1)
        xs = (np.arange(first, len(data))) * step
        out.append(polyline([ft.Offset(float(x), float(y)) for x, y in zip(xs, ys)], colour, 1.4))
    return out


def mark_shapes(marks, w: float, h: float, colour: str, thickness: float = 3.0) -> list:
    """A ribbon along the bottom of a trace, lit where the gate rejected.

    **Runs, not columns.** One shape per contiguous marked stretch rather than per marked column keeps
    this at a handful of objects on a real trace, where per-column would put 320 of them on every
    frame — the cost that makes a marking like this worth having at all.

    A ribbon rather than recolouring the waveform, for the same reason the trace is stroked and not
    filled: the waveform's colour already carries whether the channel is live, and a second meaning on
    the same ink is unreadable exactly where it matters, in the middle of an excursion.
    """
    flags = np.asarray(marks, dtype=bool)
    n = len(flags)
    if not n or not flags.any():
        return []
    step = w / n
    edges = np.diff(np.concatenate(([0], flags.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    top = h - thickness
    return [filled([ft.Offset(a * step, top), ft.Offset(b * step, top),
                    ft.Offset(b * step, h), ft.Offset(a * step, h)], colour)
            for a, b in zip(starts, ends)]


def dashed(y: float, w: float, colour: str, width: float = 0.8,
           dash: float = 6.0, gap: float = 5.0) -> list:
    """A horizontal line drawn in dashes, for a second line that must not compete with the first.

    Style rather than colour, because colour here is keyed to frequency everywhere — a second hue for
    the same band would put a swatch on screen that agrees with nothing else on it."""
    out, x = [], 0.0
    while x < w:
        out.append(polyline([ft.Offset(x, y), ft.Offset(min(x + dash, w), y)], colour, width))
        x += dash + gap
    return out


def xy_shapes(x, y, w: float, h: float, full_scale: float, colour: str,
              guides: bool = True) -> list:
    """A Lissajous figure: one channel against the other, both signed, on a square-ish box.

    An optional window on the synchrony screen, and it is the only display that shows phase alignment *directly* rather than through a derived number: a
    diagonal line means the two sites are in phase, the other diagonal means antiphase, a circle means a
    quarter cycle apart, and a shapeless blob means no fixed relationship at all.

    The two diagonals are drawn as guides because the figure is read *against* them — without something to
    compare to, a tilted ellipse is just a shape. **Both axes share one scale**, for the reason the
    crossover plot does: scaling them separately would turn any amplitude imbalance into an apparent phase
    shift, which is the one thing this is for.
    """
    out: list = []
    cx, cy = w / 2, h / 2
    radius = min(cx, cy) - 1
    if guides:
        out.append(polyline([ft.Offset(cx - radius, cy), ft.Offset(cx + radius, cy)], t.BORDER, 0.6))
        out.append(polyline([ft.Offset(cx, cy - radius), ft.Offset(cx, cy + radius)], t.BORDER, 0.6))
        out.append(polyline([ft.Offset(cx - radius, cy + radius),
                             ft.Offset(cx + radius, cy - radius)], t.SURFACE_HI, 0.8))
        out.append(polyline([ft.Offset(cx - radius, cy - radius),
                             ft.Offset(cx + radius, cy + radius)], t.SURFACE_HI, 0.8))
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    n = min(len(xs), len(ys))
    if n < 2:
        return out
    gain = radius / max(full_scale, 1e-9)
    px = cx + np.clip(xs[-n:] * gain, -radius, radius)
    py = cy - np.clip(ys[-n:] * gain, -radius, radius)
    out.append(polyline([ft.Offset(float(a), float(b)) for a, b in zip(px, py)], colour, 1.2))
    return out


def grid(w: float, h: float, rows: int = 4, cols: int = 6) -> list:
    out = []
    for i in range(1, rows):
        y = h * i / rows
        out.append(polyline([ft.Offset(0, y), ft.Offset(w, y)], t.BORDER, 0.6))
    for i in range(1, cols):
        x = w * i / cols
        out.append(polyline([ft.Offset(x, 0), ft.Offset(x, h)], t.BORDER, 0.6))
    return out


def bars(values: np.ndarray, w: float, h: float, colour, peak: float | None = None) -> list:
    """`colour` is either one colour for every bar or a function of the bar's index."""
    if not values.size:
        return []
    top = peak or (float(values.max()) or 1.0)
    step = w / len(values)
    pick = colour if callable(colour) else (lambda _i: colour)
    return [cv.Rect(i * step + 0.5, h - max(1.0, (v / top) * (h - 2)),
                    max(1.0, step - 1.8), max(1.0, (v / top) * (h - 2)),
                    border_radius=0.7, paint=ft.Paint(color=pick(i)))
            for i, v in enumerate(values)]


def envelope_shapes(lo: np.ndarray, hi: np.ndarray, w: float, h: float, full_scale: float,
                    colour: str) -> list:
    """A min/max trace as one stroked waveform, with a floor on its thickness.

    Min/max per column rather than every Nth sample, because decimating by sampling aliases spikes away
    entirely — but *stroked* rather than filled, because a filled envelope of real EEG is a solid block.
    See the loop below.
    """
    n = len(lo)
    if not n:
        return []
    limit = h / 2 - 1
    mid = h / 2
    gain = limit / max(full_scale, 1e-9)
    # **All of the geometry in numpy, and every `ft.Offset` constructed exactly once.** Profiled: this
    # function was 75% of a draw on the amplitude screen, and almost all of it was Flet's typed-property
    # machinery inside `Offset.__init__` — 5794 of them per draw where 2560 are needed, because the
    # minimum-thickness pass below used to *replace* the ones it had just built. Doing the arithmetic on
    # arrays and building the objects at the end removes the throwaways and the per-element Python.
    xs = np.arange(n) * (w / max(1, n - 1))
    tops = mid - np.minimum(limit, np.asarray(hi, dtype=float) * gain)
    bots = mid - np.maximum(-limit, np.asarray(lo, dtype=float) * gain)
    thin = (bots - tops) < MIN_TRACE_PX
    if thin.any():
        # A quiet channel would otherwise draw sub-pixel and vanish.
        centre = (bots + tops) / 2
        tops = np.where(thin, centre - MIN_TRACE_PX / 2, tops)
        bots = np.where(thin, centre + MIN_TRACE_PX / 2, bots)
    # **Stroked as a zigzag, not filled.** Filling between the two edges inks every column's whole
    # excursion, so real EEG came out as one solid mass — the information is there and the waveform is
    # not, which is no use for the thing a raw trace is read for: recognising what the signal is doing.
    # A single polyline that walks down one column and up the next visits both extremes of every column,
    # so nothing is thrown away, and it reads as a trace because the ink follows the signal instead of
    # the area under it. One shape, two points per column.
    firsts = np.where(np.arange(n) % 2 == 0, tops, bots)
    seconds = np.where(np.arange(n) % 2 == 0, bots, tops)
    ys = np.empty(2 * n)
    ys[0::2], ys[1::2] = firsts, seconds
    x2 = np.repeat(xs, 2)
    return [polyline([ft.Offset(float(x), float(y)) for x, y in zip(x2, ys)], colour, 1.0)]


# --- numbers on axes ----------------------------------------------------------------------

THRESHOLD_HEADROOM = 2.0
"""How much room a plot leaves above its own threshold line, as a multiple of it.

**A ceiling taken from the data alone draws every line the data has not reached along the top frame**,
where it reads as the edge of the plot rather than as a mark on it — and before any data arrives that is
every line on the screen. Two, because the cards' own bars have always scaled to twice their mark, so a
threshold sits at the same place on a card and on the plot beside it. It costs nothing when the data is
higher: the data wins the max.
"""


def plot_ceiling(peak: float, *lines: float | None) -> float:
    """A full scale that contains the data *and* leaves its threshold lines visibly inside."""
    wanted = [peak] + [line * THRESHOLD_HEADROOM for line in lines if line]
    return nice_ceiling(max(max(wanted), 1e-9))


def nice_ceiling(value: float) -> float:
    """Round a peak up to something worth printing on an axis: 1, 2, 5 × a power of ten.

    A full scale of "13.7 µV" changes every frame and tells you nothing you can compare against
    tomorrow; 20 µV holds still for minutes and can be read at a glance.
    """
    if value <= 0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(value))
    for step in (1.0, 2.0, 5.0, 10.0):
        if value <= step * magnitude:
            return step * magnitude
    return 10.0 * magnitude


class StickyCeiling:
    """An auto scale that does not flicker.

    `nice_ceiling(peak)` recomputed every frame is unusable on a live spectrum: a peak wobbling either side
    of 5 µV flips the axis between 5 and 10 several times a second, so the bars halve and double in height
    and the whole picture strobes. The numbers were right and the display was unreadable.

    Three things are needed, and the first two alone are not enough — a test walks the case that proves it:

    - **Headroom.** The ceiling is chosen for `peak × headroom`, not for the peak, so a signal that just
      reached the top does not sit exactly on the boundary it will re-cross next frame.
    - **A delay before falling.** Rising is immediate, because a clipped bar is information lost and the peak
      is the thing being looked for; falling waits `patience` seconds.
    - **Falling on the window's *maximum*, not on the current value.** This is the one that was missing and
      it made the first two useless: lowering to fit the instantaneous peak lands on a ceiling that the very
      next sample clips, so it rose again immediately and the strobe came back with extra steps in it. What
      decides a lower ceiling has to be the largest thing seen over the whole waiting period.
    """

    def __init__(self, patience: float = 4.0, headroom: float = 1.15, floor: float = 1.0):
        self.patience = patience
        self.headroom = headroom
        self.floor = floor
        self.value = floor
        self._recent: deque = deque()        # (time, peak) over the last `patience` seconds
        self._changed_at = 0.0

    def _wanted(self, peak: float) -> float:
        return max(self.floor, nice_ceiling(max(peak, 1e-9) * self.headroom))

    def update(self, peak: float, now: float) -> float:
        self._recent.append((now, peak))
        while len(self._recent) > 1 and now - self._recent[0][0] > self.patience:
            self._recent.popleft()

        wanted = self._wanted(peak)
        if wanted > self.value:
            self.value = wanted
            self._changed_at = now
            return self.value
        # Lower only on the window's high-water mark, and only once the window has actually had time to
        # fill since the last change — otherwise a single quiet frame decides the axis.
        settled = self._wanted(max(p for _, p in self._recent))
        if settled < self.value and now - self._changed_at >= self.patience:
            self.value = settled
            self._changed_at = now
        return self.value


def fmt_amplitude(value: float, unit: str, with_unit: bool = False) -> str:
    """One number for an axis or a caption, at a precision the value can actually carry.

    **Percentages used to be whole numbers unconditionally**, which is right for the ones this was written
    for — a 0-to-100 scale where a decimal place says nothing. It is wrong for a percentage that lives
    between 0 and 5: a modulation index axis came out "1%, 1%, 0%, 0%" across four gridlines that were
    genuinely 1.00, 0.67, 0.33 and 0.00, so the axis named one value four times. The rule is the same one
    the unit branch already used — keep a decimal while the number is small — applied to both.
    """
    if unit == "%":
        return f"{value:.0f}%" if value >= 10 or value == 0 else f"{value:.2g}%"
    text = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
    return f"{text} {unit}" if with_unit else text


def value_axis(height: float, count: int = 5, width: int = 40) -> tuple[ft.Stack, list[ft.Text]]:
    """A vertical scale beside a plot: `count` labels, absolutely placed by the caller each frame.

    Returns the stack and the labels, because the labels are mutated per frame and the stack is built
    once — the same build-once-mutate rule as everything else on a screen.
    """
    labels = [ft.Text("", size=t.SMALL, color=t.FAINT, width=width - 6,
                      text_align=ft.TextAlign.RIGHT, left=0, top=0) for _ in range(count)]
    return ft.Stack(labels, width=width, height=height), labels


def tick_axis(ticks, tick_width: int = 30) -> tuple[ft.Stack, list[ft.Text]]:
    """A horizontal scale under a plot, one label per named gridline."""
    labels = [ft.Text(f"{v:g}", size=t.SMALL, color=t.FAINT, width=tick_width,
                      text_align=ft.TextAlign.CENTER, left=0, top=0, visible=False)
              for v in ticks]
    return ft.Stack(labels, height=16), labels


def place_value_labels(labels: list[ft.Text], height: float, top: float, bottom: float = 0.0,
                       unit: str = "µV") -> None:
    """Write and position a value axis so each number sits on the line it names."""
    n = len(labels)
    for i, label in enumerate(labels):
        fraction = i / (n - 1) if n > 1 else 0.0
        label.value = fmt_amplitude(top - fraction * (top - bottom), unit,
                                    with_unit=(i == 0))
        # clamp the ends inside the plot: they name its edges, and half a line above the top reads as
        # naming nothing
        label.top = min(height - 12, max(0.0, fraction * height - 6))


def bin_centre_x(lo: float, bin_hz: float, bins: int, width: float, top: float | None = None):
    """Where the gridline for a frequency goes, when the plot is drawn in bins.

    **A bin is centred on its frequency**, so bin *k* covers `[f-half, f+half)` and its gridline passes
    through the middle of the drawn cell — `((hz - lo) / bin_hz + 0.5) * step`, not `(hz - lo) / span`.
    This lives here because getting it wrong is subtle and was fixed twice: binning `[f, f+1)` and
    drawing from f's gridline rightward puts every feature half a bin late, which reads as a notch at
    50 Hz starting at 50 and continuing past it. Both the histogram and the spectrogram use it, so there
    is one implementation to be right.

    Returns a function of frequency giving an x, or None when that frequency is off the plot.
    """
    step = width / max(1, bins)
    ceiling = top if top is not None else lo + (bins - 1) * bin_hz

    def x_of(hz: float) -> float | None:
        if hz < lo or hz > ceiling:
            return None
        return ((hz - lo) / bin_hz + 0.5) * step

    return x_of


def place_tick_labels(labels: list[ft.Text], ticks, x_of, tick_width: int = 30,
                      left_offset: float = 0.0, width: float | None = None) -> None:
    """Centre each label on the gridline it names. `x_of` comes from `bin_centre_x` or is any
    value → x mapping returning None for off-plot.

    Pass `width` to keep the end labels inside the plot, the same clamp `place_value_labels` applies for the
    same reason: a label centred on the last gridline hangs half its width past the edge and gets clipped by
    the parent, so "40" renders as "4".
    """
    for label, value in zip(labels, ticks):
        x = x_of(value)
        label.visible = x is not None
        if x is None:
            continue
        left = left_offset + x - tick_width / 2
        if width is not None:
            left = min(left, left_offset + width - tick_width)
            left = max(left, left_offset)
        label.left = left


def gridlines(ticks, x_of, height: float) -> list:
    """Vertical rules at the same places the labels name — so the two can never disagree."""
    out = []
    for value in ticks:
        x = x_of(value)
        if x is not None:
            out.append(polyline([ft.Offset(x, 0), ft.Offset(x, height)], t.BORDER, 0.6))
    return out


# --- the cascading spectral display -------------------------------------------------------

def png_bytes(rgb: np.ndarray) -> bytes:
    """PNG of an (h, w, 3) uint8 array, for `ft.Image(src=...)`, which takes raw bytes.

    Hand-rolled because the only alternative is a Pillow dependency for one image, and the format is
    four chunks: a signature, IHDR, one zlib-compressed IDAT of filter-prefixed scanlines, and IEND.
    """
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


class Chronospectrogram:
    """The instrument CFC is actually read off: power against frequency, cascading down over time.

    This is where the state is seen: it shows the approximate voltage produced at every frequency from
    1-40 Hz for the last 5 seconds, and the two channels are **summed** before the transform, so which channel a beta spindle came from
    is deliberately not visible here and has to be read off the raw traces.

    Rows arrive newest-first at the top and scroll downward, which is the direction all three commercial
    implementations use. The ceiling is a setting rather than an autoscale, because the whole clinical
    read is "the energy in the skirts fell from 25 µV to under 5" and an autoscaling display makes
    exactly that change invisible.
    """

    def __init__(self, rows: int = 72, lo_hz: float = 1.0, hi_hz: float = 40.0,
                 bin_hz: float = 1.0, depth_seconds: float = 5.0):
        self.rows = rows
        self.depth_seconds = depth_seconds
        self.lo_hz = lo_hz
        self.hi_hz = hi_hz
        self.bin_hz = bin_hz
        # 20, because our bins are a per-Hz amplitude density from a
        # Welch estimate, the same convention as the amplitude screen's histogram, and in that convention
        # a 30 µV sine reads about 14. Verified against a synthetic signal before choosing the number.
        self.ceiling_uv = 20.0
        self._rows: deque[np.ndarray] = deque(maxlen=rows)

    @property
    def bins(self) -> int:
        return max(1, int(round((self.hi_hz - self.lo_hz) / self.bin_hz)) + 1)

    def centres(self) -> np.ndarray:
        """Bin centres, on integers when `bin_hz` is 1 — a bin is centred on its frequency, so a notch
        at 50 Hz looks like it is at 50 Hz."""
        return self.lo_hz + np.arange(self.bins) * self.bin_hz

    def push(self, spectrum: np.ndarray) -> None:
        self._rows.appendleft(np.asarray(spectrum, dtype=float))

    def clear(self) -> None:
        self._rows.clear()

    @property
    def filled(self) -> bool:
        return bool(self._rows)

    def image(self) -> bytes | None:
        """The cascade as a PNG, newest row at the top, **always `rows` rows tall**.

        The padding is not cosmetic. The image is drawn stretched to a fixed-height box, so emitting only the
        rows that have arrived meant the picture *compressed vertically* for the first twenty seconds —
        every row you had already read kept moving and shrinking as the next one arrived, which is the one
        thing a time axis must not do. Padding to full height fixes the geometry from the first frame: the
        cascade fills downward at a constant rate and then scrolls.

        Zero pads to the palette's first stop, which is the panel's own surface colour, so the not-yet-drawn
        part reads as empty rather than as silence at every frequency.
        """
        if not self._rows:
            return None
        w = len(self._rows[0])
        frame = np.zeros((self.rows, w, 3), dtype=np.uint8)
        frame[:] = t.amplitude_ramp(np.zeros(w))
        ceiling = max(self.ceiling_uv, 1e-6)
        for y, row in enumerate(self._rows):
            if len(row) != w:
                row = np.resize(row, w)
            frame[y] = t.amplitude_ramp(np.clip(row / ceiling, 0.0, 1.0))
        return png_bytes(frame)

    @property
    def interval(self) -> float:
        """Seconds between rows, derived from the depth so the row *rate* follows the window rather than
        the window following a fixed rate. Changing the depth must not change how fast the picture scrolls
        relative to the axis, which is the only thing a time axis promises."""
        return self.depth_seconds / max(1, self.rows)

    @property
    def seconds_filled(self) -> float:
        """How much of the window has arrived, as a fraction of its rows. For a caption to be honest about
        the part of the plot that is not data yet."""
        return len(self._rows) / self.rows if self.rows else 0.0


class EnvelopePlot:
    """Two amplitude envelopes on one time axis — the cross-frequency coupling instrument.

    The judgement it supports is a shape, not a number: the training aims at self-regulation of
    cross-frequency coupling, that is, the amplitude envelopes moving together. So the two traces must share one axis and one scale — scaling
    them independently would make any two signals look coupled, which is the one thing this instrument
    exists to distinguish.
    """

    def __init__(self, columns: int = 300, count: int = 2):
        self.columns = columns
        self._series = [deque([0.0] * columns, maxlen=columns) for _ in range(count)]
        self._marks = deque([False] * columns, maxlen=columns)
        self._pushed = 0

    def push(self, values, mark: bool = False) -> None:
        for series, value in zip(self._series, values):
            series.append(float(value))
        self._marks.append(bool(mark))
        self._pushed += 1

    def mark_fraction(self) -> float | None:
        """How much of what is *drawn* is marked, 0…1, or None before anything has been pushed.

        Of the filled part of the window and not of the window, so the number answers the picture the
        reader is looking at rather than counting the zeros nobody drew."""
        drawn = min(self._pushed, self.columns)
        if drawn <= 0:
            return None
        return sum(list(self._marks)[-drawn:]) / drawn

    def series(self) -> list[np.ndarray]:
        return [np.asarray(s, dtype=float) for s in self._series]

    @property
    def filled(self) -> float:
        """How much of the window holds real data, 0…1."""
        return min(1.0, self._pushed / max(1, self.columns))

    def recent_mean(self, index: int = 0) -> float | None:
        """The mean of what is actually drawn for one series, or None before anything arrives.

        For a verdict about a noisy reading. A single sample of a debiased phase statistic swings by
        about `1/sqrt(N)` around its true value, so a sentence that reads it off the latest value alone
        flips between "holding" and "nothing there" while the picture below it plainly shows one or the
        other. Of the filled part of the window, like `mark_fraction`, so it answers the picture.
        """
        drawn = min(self._pushed, self.columns)
        if drawn <= 0 or index >= len(self._series):
            return None
        return float(np.mean(list(self._series[index])[-drawn:]))

    def peak(self) -> float:
        arrays = [s[-self._pushed:] if self._pushed else s[:0] for s in self.series()]
        return max((float(np.max(s)) for s in arrays if len(s)), default=0.0)

    def shapes(self, w: float, h: float, full_scale: float, colours,
               thresholds=(), higher=(), marks_colour: str | None = None) -> list:
        """Right-anchored, at a fixed step, drawing only what has arrived.

        The columns are pre-filled with zeros so that the time axis is fixed from the first frame, and
        those are **not drawn**: a flat line along the bottom is a measurement of zero, which is a claim
        about the signal rather than about the window not being full yet. Anchoring right and keeping the
        step constant is the other half of it — stretching a short series across the whole width would
        make every point you had already read keep moving as the next arrived.

        `higher` is a second line per series, drawn dashed in the same colour: one instrument with two
        bars rather than four unrelated lines. `marks_colour` lights the ribbon along the bottom for the
        columns pushed with `mark=True`.
        """
        out = list(grid(w, h, rows=4, cols=6))
        gain = (h - 2) / max(full_scale, 1e-9)
        for level, colour in zip(thresholds, colours):
            if level is None:
                continue
            y = h - min(h - 1, level * gain)
            out.append(polyline([ft.Offset(0, y), ft.Offset(w, y)], colour, 0.8))
        for level, colour in zip(higher, colours):
            if not level:
                continue
            out += dashed(h - min(h - 1, level * gain), w, colour)
        if marks_colour is not None:
            out += mark_shapes(self._marks, w, h, marks_colour)
        step = w / max(1, self.columns - 1)
        for series, colour in zip(self.series(), colours):
            first = max(0, len(series) - self._pushed)
            if len(series) - first < 2:
                continue
            out.append(polyline([ft.Offset(i * step, h - min(h - 1, series[i] * gain))
                                 for i in range(first, len(series))], colour, 1.6))
        return out
