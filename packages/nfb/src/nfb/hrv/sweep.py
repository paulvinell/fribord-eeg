"""The sliding rate search: a pacer that changes its rate at every breath, and the breathing rate at which
the heart follows it most.

**Pure, and in processing's layer.** Nothing here reads a clock, a device or a screen: the schedule is a
table, a breath is measured from the beats that were put into it, and the clock that the series runs on is
the one that `Sweep.feed` is given.

**The swing is read from the beat interval, in milliseconds.** The heart rate in bpm leans with the mean
rate (a fall from 80 to 55 bpm moved the found peak by +0.21 breaths a minute); the interval does not
(+0.02). The fit also takes off the median of what noise alone gives, for the number of beats that the breath
holds, so that noise alone reads zero in half of the short breaths, as it does in half of the long ones.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import accumulate
from typing import NamedTuple, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

START_PER_MIN = 6.75
END_PER_MIN = 4.25
"""The default sweep: 78 breaths from 6.75 down to 4.25 breaths a minute, so the first breaths are short and
the last are long, and the whole sweep takes about 15 minutes. This project's choice. Both rates are
settings, in either order."""

STEP_S = (60.0 / END_PER_MIN - 60.0 / START_PER_MIN) / 77
"""The change of the period between two neighbouring breaths, 67.9 ms. It is constant in seconds, so the
count of breaths and the time of a sweep follow from its range."""

WINDOW = 9
"""Neighbouring breath numbers in the window that the curve, the best rate and the target figure read. Nine
spans 0.3 breaths a minute of the swept range and takes a breath that lost the pace: three lost at the peak
cost 0.05 breaths a minute of rms error and no bias. A window is a span of breath numbers and never the
nine nearest measured breaths, which can lie on the two sides of a stretch that was not measured."""

SUPPORT = WINDOW // 2 + 1
"""Measured breaths that a window needs, a majority of it. While fewer breaths are measured, or they lie
too far apart for any window to hold that many, the most that a window holds is enough."""

MIN_BEATS = 5
"""Beats a breath needs: three fitted values (the level, a sine and a cosine) and two degrees of freedom
that the noise term is estimated from."""

MIN_FOR_VERDICT = 27
"""Measured breaths the verdict needs: three windows, so that a window and half a window either side
still leave breaths outside to compare with. Half as many second differences between three measured
neighbours must exist for the test to know how alike neighbours are."""

TRIES = 1000
CLEAR_SHARE = 0.05
"""The peak is clear when at most this share of `TRIES` random series give a rise as large as the real
one, where the series are as smooth from breath to breath as the real one (`_neighbour_noise`). On 500 runs
of 78 breaths with no peak and 8 ms of scatter it called 0.6, 2.8, 5.8 and 9.2 percent clear when
neighbours resembled each other by 0, 0.3, 0.6 and 0.8, and an 80 over 20 ms peak that is a third of the
range wide 64 percent (a narrow one, 78)."""

EDGE_WINDOWS = 3
"""A window with fewer than this many supported windows beside it on one side is at an end of the swept
range, or beside a stretch that was not measured."""

EDGE_SCATTER = 2.0
"""The best window is at an edge when a window at an end reaches within this many scatters of a window
median (as noise alone gives it) of the best one. A steady slope has its best windows at an end, and
noise puts the best window a few places away from it."""

IN_STEP_LEVEL = 0.95
"""How far above the noise term a breath's sine and cosine must stand before the breath gives an in-step value:
the squared amplitude must reach this quantile of what noise alone gives. The phase of noise is random, so a
value from a breath that does not stand clear would move the pacer by chance."""

REASONS = ("few", "gap")
"""Why a breath gives no swing: too few beats, or beats that do not cover the breath."""

CAUSES = ("error", "paused", "silence", "stall", "clock")
"""Why the sweep holds: the source reports a fault, the session is paused, no beat arrives, the looks are
more than `STALL_S` apart, or the clock of the stamps moved against the clock of the looks."""

END_REASONS = ("done", "stop", "plan", "source")

STALL_S = 1.0
"""The longest step between two looks that the clock follows. The screen draws many times a second, so a
longer step means that it did not run, and the pacer cannot have kept to the clock."""

CLOCK_STEP_S = 2.0
"""How far the stamps of two new snapshots can drift from the looks that saw them before the sweep holds.
Only looks that follow their predecessor within `STALL_S` are compared, and a new snapshot arrived after the
look before it, so each is seen less than `STALL_S` late and two of them differ by less than that however
far apart the beats are. A wall clock that is set moves the stamps by more."""

_SEED = 20240611

def _noise_median(dof: int) -> float:
    """The median of the squared sine-and-cosine amplitude that noise alone gives, in units of the noise
    term that is estimated from `dof` degrees of freedom: the median of an F distribution with 2 and `dof`
    degrees of freedom, `dof (2^(2/dof) - 1) / 2`. It is 1 at 2 degrees and tends to ln 2 as they grow, so
    a fixed ln 2 leaves a short breath with a median swing above zero. The mean of the reads still falls
    with the number of beats, because the amplitude of noise does (about 1.5 ms between a breath of 9 s
    and one of 14 s, against 1.9 ms with ln 2), which no offset that needs no reference beat count can undo."""
    return _noise_quantile(dof, 0.5)


def _noise_quantile(dof: int, level: float) -> float:
    """The `level` quantile of the same distribution: `dof ((1 - level)^(-2/dof) - 1) / 2`."""
    return dof * ((1.0 - level) ** (-2.0 / dof) - 1.0) / 2.0


@dataclass(frozen=True)
class SweepSchedule:
    """Where the pacer is at any second of the sweep. The first breath has the period of `start_per_min`
    and each next one differs from it by `step_s`, towards `end_per_min`, in either direction. The last
    breath is the last whose period does not pass the end, so every rate of the sweep lies between the two. `breath_at` is the one function that
    bins a time into a breath, and it and `start_s` read the same table of running sums."""

    start_per_min: float = START_PER_MIN
    end_per_min: float = END_PER_MIN
    step_s: float = STEP_S
    breaths: int = field(init=False)
    rates: tuple = field(init=False, repr=False, compare=False)
    """Breaths a minute of each breath, in sweep order."""
    _periods: tuple = field(init=False, repr=False, compare=False)
    _starts: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rates = (self.start_per_min, self.end_per_min)
        if not (all(math.isfinite(r) and r > 0.0 for r in rates)
                and math.isfinite(self.step_s) and self.step_s > 0.0):
            raise ValueError("the rates and the step must be finite and above 0")
        if self.start_per_min == self.end_per_min:
            raise ValueError("the sweep needs two different rates")
        first, last = 60.0 / self.start_per_min, 60.0 / self.end_per_min
        count = math.floor(abs(last - first) / self.step_s + 1e-9) + 1
        step = math.copysign(self.step_s, last - first)
        periods = [first + step * k for k in range(count)]
        object.__setattr__(self, "breaths", count)
        object.__setattr__(self, "_periods", tuple(periods))
        object.__setattr__(self, "_starts", tuple(accumulate(periods, initial=0.0)))
        object.__setattr__(self, "rates", tuple(60.0 / p for p in periods))

    def with_end(self, end_per_min: float) -> "SweepSchedule":
        """The schedule with another end rate and the same start and step, so the breaths that both have
        keep their numbers and their start times."""
        return SweepSchedule(self.start_per_min, end_per_min, self.step_s)

    @property
    def low_per_min(self) -> float:
        """The slower of the two ends: no breath of the sweep is slower."""
        return min(self.start_per_min, self.end_per_min)

    @property
    def high_per_min(self) -> float:
        return max(self.start_per_min, self.end_per_min)

    def period_s(self, k: int) -> float:
        return self._periods[k]

    def start_s(self, k: int) -> float:
        """Seconds from the start of the sweep to the start of breath `k`; `k == breaths` is the end."""
        return self._starts[k]

    @property
    def total_s(self) -> float:
        return self._starts[-1]

    def breath_at(self, s: float) -> int | None:
        """The breath that owns second `s`: half open, so a breath owns its start and not its end."""
        if not 0.0 <= s < self.total_s:
            return None                 # also NaN, for which every comparison is false
        return bisect_right(self._starts, s) - 1

    def phase_at(self, s: float) -> tuple[int, float] | None:
        """`(breath, part of it that has passed)`, the part in `[0, 1)`; None outside the sweep."""
        k = self.breath_at(s)
        if k is None:
            return None
        return k, min((s - self._starts[k]) / self._periods[k], math.nextafter(1.0, 0.0))


def breath_turn(frac: float, share: float) -> tuple[bool, float]:
    """In or out, and how far through that half, when the part `frac` of a breath has passed and the first
    `share` of each breath is the inhale. The metronome and the sweep both turn here, so the two pacers
    cannot differ."""
    if not 0.0 < share < 1.0:
        raise ValueError(f"the inhale share must lie between 0 and 1, not {share!r}")
    if frac < share:
        return True, min(1.0, max(0.0, frac / share))
    return False, min(1.0, max(0.0, (frac - share) / (1.0 - share)))


def volume_peak(share: float) -> float:
    """The part of a breath, in `[0, 1)`, at which the fundamental of the paced lung volume peaks, when the
    first `share` of each breath is the inhale. The volume rises linearly through the inhale and falls
    linearly through the exhale, as the ring does. Its first Fourier coefficient has the phase
    `-(pi share + pi / 2)`, so the peak lies at `1/4 + share/2`: at the end of the inhale for an even
    breath, and at a quarter of the breath for an inhale that is nearly none (a sawtooth). It is defined for
    every share, where the end of the inhale is the peak of no single wave."""
    if not 0.0 < share < 1.0:
        raise ValueError(f"the inhale share must lie between 0 and 1, not {share!r}")
    return 0.25 + share / 2.0


class Measure(NamedTuple):
    """One breath: `swing_ms`, and `in_step_s`, how late the wave of the heart rate is against the paced
    breath. Positive is late and negative is early, in seconds: the heart rate peaks that long after the
    fundamental of the paced lung volume (`volume_peak`) peaks, or that long before it. It is None where
    the paced swing does not stand `IN_STEP_LEVEL` above its noise term."""

    swing_ms: float | None
    reason: str | None
    in_step_s: float | None = None


def paced_swing(stamps_s: Sequence[float], rr_ms: Sequence[float], start_s: float, end_s: float,
                inhale_share: float = 0.5) -> Measure:
    """The peak-to-peak swing of the beat interval that follows one breath, in milliseconds, and how late
    the heart rate is against the breath.

    `stamps_s` are the times of the beats of this breath and `rr_ms` the intervals that end at them; the
    caller has put them into the breath, and `start_s` and `end_s` are its edges. The interval is fitted by
    a constant and one cycle of sine and cosine over the breath, and the median of what noise alone gives
    is taken off the squared amplitude, so noise reads near zero in the middle of the range. An interval ends
    at its beat and is the rate at its middle, so it is put half an interval before its stamp.

    The fit also gives the phase. The heart rate peaks where the interval is shortest, half a cycle after
    the interval peaks. `Measure.in_step_s` is that time minus `volume_peak(inhale_share)`, as a share of
    the breath turned into seconds and wrapped into half a breath either way, so it reads the same for
    every share of the inhale and every length of breath. The interval is the fitted quantity because
    its noise is the same at every rate, and the sign is that of the heart rate.

    Not measured, with a reason, when the breath has fewer than `MIN_BEATS` beats, or a stamp or an
    interval that is not finite, or stamps that do not rise or leave their breath, or a quarter of the
    breath with no beat, or a fit of rank below 3. Four beats that crowd a third of a breath fit 252 ms
    against 20 ms with full cover, so the quarter rule is what keeps them out.
    """
    peak = volume_peak(inhale_share)
    stamps = np.asarray(stamps_s, dtype=float)
    rr = np.asarray(rr_ms, dtype=float)
    if stamps.ndim != 1 or stamps.shape != rr.shape or stamps.size < MIN_BEATS:
        return Measure(None, "few")
    if not (np.all(np.isfinite(stamps)) and np.all(np.isfinite(rr))
            and math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
        return Measure(None, "gap")
    if np.any(np.diff(stamps) <= 0.0) or stamps[0] < start_s or stamps[-1] > end_s:
        return Measure(None, "gap")
    phase = (stamps - start_s) / (end_s - start_s)
    quarter = np.minimum((phase * 4.0).astype(int), 3)
    if np.unique(quarter).size < 4:
        return Measure(None, "gap")
    angle = 2.0 * np.pi * (stamps - rr / 2000.0 - start_s) / (end_s - start_s)     # an interval is the rate
    design = np.column_stack([np.ones_like(phase), np.sin(angle), np.cos(angle)])
    coef, _, rank, _ = np.linalg.lstsq(design, rr, rcond=None)
    if rank < 3:
        return Measure(None, "gap")
    left = float(np.sum((rr - design @ coef) ** 2))
    sigma2 = left / (stamps.size - 3)
    cov = np.linalg.inv(design.T @ design)
    noise = sigma2 * float(cov[1, 1] + cov[2, 2])
    dof = stamps.size - 3
    amplitude2 = float(coef[1] ** 2 + coef[2] ** 2)
    power = amplitude2 - _noise_median(dof) * noise
    in_step = None
    if amplitude2 > 0.0 and amplitude2 >= _noise_quantile(dof, IN_STEP_LEVEL) * noise:
        heart_peak = math.atan2(coef[1], coef[2]) / (2.0 * math.pi) + 0.5
        late = (heart_peak - peak + 0.5) % 1.0 - 0.5
        in_step = late * (end_s - start_s)
    return Measure(2.0 * math.sqrt(max(0.0, power)), None, in_step)


class Point(NamedTuple):
    """One supported window of the curve: the number of its first breath, the mean rate of the breaths
    measured in it and their median swing."""

    first: int
    window_rate: float
    median_ms: float


class Best(NamedTuple):
    rate: float
    swing_ms: float
    breaths: tuple


class Verdict(NamedTuple):
    kind: str
    measured: int
    rise_ms: float
    outside_ms: float
    reached: int
    tries: int
    edge: bool


def _series(rates: Sequence[float], values: Sequence[float | None]) -> np.ndarray:
    """The swing of every breath of the sweep in breath order, with NaN where a breath is not measured.
    Breath numbers stay breath numbers, so a hole stays a hole."""
    if len(rates) != len(values):
        raise ValueError("one rate for each value")
    return np.array([math.nan if v is None else v for v in values], dtype=float)


def _median(a: np.ndarray) -> np.ndarray:
    """The median over the last axis of the values that are there, NaN where there are none."""
    count = np.sum(~np.isnan(a), axis=-1)
    ordered = np.sort(a, axis=-1)           # NaN sorts last
    low = np.take_along_axis(ordered, np.maximum(count - 1, 0)[..., None] // 2, axis=-1)[..., 0]
    high = np.take_along_axis(ordered, (count // 2)[..., None], axis=-1)[..., 0]
    return np.where(count > 0, (low + high) / 2.0, math.nan)


def _width(count: int) -> int:
    return min(WINDOW, count)


def _windows(rows: np.ndarray, need: int) -> np.ndarray:
    """The median of every window of `WINDOW` neighbouring breath numbers, for each row of swings in breath
    order. NaN for a window that holds fewer than `need` measured breaths."""
    win = sliding_window_view(rows, _width(rows.shape[1]), axis=1)
    return np.where(np.sum(~np.isnan(win), axis=-1) >= need, _median(win), math.nan)


def _need(row: np.ndarray) -> int:
    """How many measured breaths a window needs: `SUPPORT`, or the most that any window holds when the
    breaths are so far apart that none holds that many, so that a best rate exists from the first measured
    breath on."""
    win = sliding_window_view(row, _width(row.size))
    return min(SUPPORT, int(np.max(np.sum(~np.isnan(win), axis=-1))))


def curve(rates: Sequence[float], values: Sequence[float | None]) -> list[Point]:
    """One point for each supported window of `WINDOW` neighbouring breath numbers, in breath order. The
    line through them never joins breaths that lie on the two sides of a stretch that was not measured,
    because a window there holds too few measured breaths and gives no point."""
    row = _series(rates, values)
    if np.all(np.isnan(row)):
        return []
    width = _width(row.size)
    points = []
    for s, median in enumerate(_windows(row[None, :], _need(row))[0]):
        if not math.isnan(median):
            near = np.flatnonzero(~np.isnan(row[s:s + width])) + s
            points.append(Point(s, float(np.mean([rates[i] for i in near])), float(median)))
    return points


def best(rates: Sequence[float], values: Sequence[float | None]) -> Best | None:
    """The window of `WINDOW` neighbouring breath numbers with the highest median, from the first measured
    breath on. Its rate is the mean rate of the breaths measured in it, so it is a rate at which breaths
    were measured. The first window wins a tie."""
    row = _series(rates, values)
    if np.all(np.isnan(row)):
        return None
    medians = _windows(row[None, :], _need(row))[0]
    s = int(np.argmax(np.where(np.isnan(medians), -math.inf, medians)))
    near = np.flatnonzero(~np.isnan(row[s:s + _width(row.size)])) + s
    return Best(float(np.mean([rates[i] for i in near])), float(medians[s]), tuple(int(k) for k in near))


def swing_at(rates: Sequence[float], values: Sequence[float | None], rate: float) -> float | None:
    """The median swing of the window that is centred on `rate`, in milliseconds, or None when `rate` lies
    outside the swept rates or the window around it holds too few measured breaths. The same window as the
    curve and the best rate use, so a figure at a rate always comes from breaths that were paced near it."""
    row = _series(rates, values)
    lo, hi = min(rates), max(rates)
    if np.all(np.isnan(row)) or not lo <= rate <= hi:
        return None
    order = np.argsort(rates)           # the rates fall or rise along the sweep, and interp needs them rising
    k = int(round(float(np.interp(rate, np.asarray(rates)[order], order))))
    width = _width(row.size)
    median = _windows(row[None, :], _need(row))[0][int(np.clip(k - width // 2, 0, row.size - width))]
    return None if math.isnan(median) else float(median)


def _rise(rows: np.ndarray, need: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each row of swings in breath order: the medians of its windows, where its best window starts,
    the median of that window and the median of the measured breaths outside the window and half a window
    either side of it."""
    medians = _windows(rows, need)
    at = np.argmax(np.where(np.isnan(medians), -math.inf, medians), axis=1)
    outside = _median(np.where(_near(at, rows.shape[1]), np.nan, rows))
    return medians, at, medians[np.arange(rows.shape[0]), at], outside


def _near(at: np.ndarray, count: int) -> np.ndarray:
    """For each window start in `at`: which of `count` breath numbers lie in that window or within half a
    window of it."""
    width, pos = _width(count), np.arange(count)
    return (pos >= (at - width // 2)[:, None]) & (pos < (at + width + width // 2)[:, None])


def _neighbour_noise(row: np.ndarray) -> tuple[float, float] | None:
    """The correlation `phi` between neighbouring breaths and the scatter of one breath, for the random
    series that the peak is compared with. None when too few breaths have two measured neighbours.

    Both come from differences between neighbours. The variance of the first differences is 2 s^2
    (1 - phi) and that of the second differences is that times (3 - phi). A smooth drift hardly changes
    differences, so the series that results has the drift that a series of such neighbours has by chance.
    The estimate of phi is moved up by its bias, (1 + 3 phi) / n, and by one standard error,
    sqrt((1 - phi^2) / n), so that slow drift does not read as a peak."""
    first = (row[1:] - row[:-1])[~np.isnan(row[1:] - row[:-1])]
    second = (row[2:] - 2.0 * row[1:-1] + row[:-2])
    second = second[~np.isnan(second)]
    if second.size < MIN_FOR_VERDICT // 2:
        return None
    n = int(np.sum(~np.isnan(row)))
    v1, v2 = float(np.var(first)), float(np.var(second))
    if v1 <= 1e-12:
        return 0.0, 0.0
    phi = float(np.clip(3.0 - v2 / v1, 0.0, 0.97))
    phi = min(0.97, phi + (1.0 + 3.0 * phi) / n + math.sqrt((1.0 - phi ** 2) / n))
    return phi, math.sqrt(v1 / (2.0 * (1.0 - phi)))


def _random_series(phi: float, sd: float, like: np.ndarray, count: int, rng) -> np.ndarray:
    """`count` series of random swings in which each breath resembles the one before it by `phi`, with the
    hole pattern of `like`."""
    e = rng.normal(size=(count, like.size))
    x = np.zeros_like(e)
    x[:, 0] = e[:, 0]
    root = math.sqrt(1.0 - phi ** 2)
    for i in range(1, like.size):
        x[:, i] = phi * x[:, i - 1] + root * e[:, i]
    return np.where(np.isnan(like)[None, :], math.nan, sd * x)


def _end_windows(supported: np.ndarray) -> np.ndarray:
    """The supported windows with fewer than `EDGE_WINDOWS` supported windows next to them on one side,
    which are those at an end of the swept range or beside a stretch that was not measured."""
    out = np.zeros(supported.size, dtype=bool)
    for p in np.flatnonzero(supported):
        left = right = 0
        while left < EDGE_WINDOWS and p - left - 1 >= 0 and supported[p - left - 1]:
            left += 1
        while right < EDGE_WINDOWS and p + right + 1 < supported.size and supported[p + right + 1]:
            right += 1
        out[p] = left < EDGE_WINDOWS or right < EDGE_WINDOWS
    return out


def verdict(rates: Sequence[float], values: Sequence[float | None]) -> Verdict:
    """Whether the best window stands out from the other breaths.

    The rise is the median of the best window minus the median outside it and half a window either side.
    It is compared with the rise of `TRIES` random series in which neighbouring breaths are as alike as
    in these (`_neighbour_noise`), with the same breaths missing, so slow drift in the swings gives large
    rises by chance and is not called a peak. A fixed seed gives the same answer each time and moves no
    global random state. The peak is clear when at most `CLEAR_SHARE` of the tries give a rise as large.
    It is at an edge when a window at an end of the swept range, or beside a stretch that was not measured,
    is within `EDGE_SCATTER` scatters of a window median of the best one, as it is for a steady slope.
    It costs a few milliseconds, so it is for the end of a sweep and not for every draw.
    """
    row = _series(rates, values)
    count = int(np.sum(~np.isnan(row)))
    if count < MIN_FOR_VERDICT:
        return Verdict("few", count, 0.0, 0.0, 0, 0, False)
    need = _need(row)
    medians, at, top, outside = _rise(row[None, :], need)
    noise = (_neighbour_noise(np.where(_near(at, row.size)[0], np.nan, row))
             or _neighbour_noise(row))      # with too few breaths away from the peak, all of them: the
    if noise is None:                       # peak then makes the null wider, and the verdict more careful
        return Verdict("few", count, 0.0, 0.0, 0, 0, False)
    rise = float(top[0] - outside[0])
    tries = _random_series(*noise, row, TRIES, np.random.default_rng(_SEED))
    medians_t, _, top_t, outside_t = _rise(tries, need)
    reached = int(np.sum(top_t - outside_t >= rise))
    scatter = float(np.nanstd(medians_t))
    ends = _end_windows(~np.isnan(medians[0]))
    edge = bool(np.nanmax(np.where(ends, medians[0], np.nan)) >= top[0] - EDGE_SCATTER * scatter)
    return Verdict("clear" if reached <= CLEAR_SHARE * TRIES else "not_clear", count, rise,
                   float(outside[0]), reached, TRIES, edge)


def heart_rate_ends(means_ms: Sequence[float | None]) -> tuple[float, float] | None:
    """The mean heart rate in bpm of the first and of the last `WINDOW` measured breaths, from the mean beat
    interval of each, or None when fewer than two sets of `WINDOW` breaths are measured. The two sets share
    no breath, so a fall of the rate between them is a fall and not an overlap."""
    got = [v for v in means_ms if v is not None]
    if len(got) < 2 * WINDOW:
        return None
    return 60000.0 / float(np.mean(got[:WINDOW])), 60000.0 / float(np.mean(got[-WINDOW:]))


class InStep(NamedTuple):
    """Where the heart rate and the breath are in step: `rate` in breaths a minute, or None with `side` saying
    why. `side` is "late" when the heart rate is late at every measured rate, so the rate lies below the
    measured range, "early" when it is early at every one, so the rate lies above it, and None otherwise."""

    rate: float | None
    side: str | None


def in_step(rates: Sequence[float], values: Sequence[float | None], near: float | None = None) -> InStep:
    """The rate at which the windowed in-step value changes sign, from late at the faster rates to early at
    the slower ones. It reads the windows of the curve (`curve`), so a hole in the breaths is never joined
    across, and it places the change between the two windows that hold it by a straight line between their
    medians. Where several windows change sign, the one nearest `near` counts; where none does, the rate is
    None and `side` says whether every window is late or every window is early, once `MIN_FOR_VERDICT`
    breaths give a value. The `median_ms` of a point is here in seconds."""
    points = curve(rates, values)
    found = []
    for a, b in zip(points, points[1:]):
        if b.first != a.first + 1:
            continue
        (r0, v0), (r1, v1) = ((a.window_rate, a.median_ms), (b.window_rate, b.median_ms))
        if (v0 < 0.0) != (v1 < 0.0) and (v1 - v0) * (r1 - r0) > 0.0:
            found.append(r0 + (0.0 - v0) * (r1 - r0) / (v1 - v0))
    if found:
        return InStep(min(found, key=lambda r: abs(r - near)) if near is not None else found[0], None)
    medians = [p.median_ms for p in points]
    if sum(v is not None for v in values) < MIN_FOR_VERDICT:
        return InStep(None, None)       # a claim about the whole range needs breaths across it: noise alone
    if medians and all(v >= 0.0 for v in medians):      # gives a value in one breath in twenty
        return InStep(None, "late")
    if medians and all(v < 0.0 for v in medians):
        return InStep(None, "early")
    return InStep(None, None)


class Outcome(NamedTuple):
    """One attempt at a breath that ended: its swing and no reason, or no swing and why."""

    k: int
    swing_ms: float | None
    reason: str | None


@dataclass(frozen=True)
class Result:
    """What a sweep found, fixed at its end."""

    best: Best | None
    in_step: InStep
    verdict: Verdict
    heart_bpm: tuple[float, float] | None
    measured: int
    unmeasured: int
    reason: str
    source_key: object


def _check_cause(cause: str | None) -> None:
    if cause is not None and cause not in CAUSES:
        raise ValueError(f"cause must be one of {CAUSES}, not {cause!r}")


class Look(NamedTuple):
    """What one look at a snapshot gave: the new beats, the cause that holds the run (None to run on), the
    step of the run's own clock since the look before, whether this is the first look, the stamp of the
    snapshot with whether it is new, and when the source gave the snapshot, on the clock of the stamp."""

    beats: list
    cause: str | None
    step: float
    first: bool
    stamp: float
    fresh: bool
    given: float


class BeatRun:
    """The part of a run that measures breaths from the beats which does not depend on its schedule.

    It owns the beats that were taken, the clock that the run keeps (the subclass says how it grows) and an
    anchor that turns the stamp of a beat into a time on that clock. A look is `_observe`, which takes the
    new beats and judges the holds that only the stamps show: a step of the looks that is too long, and a
    step of the clock of the stamps. The inhale share is noted for each breath in progress, so that a breath
    that ran with two shares gives no in-step value.
    """

    def __init__(self, source_key: object = None):
        self._key = source_key
        self._begun = False
        self._clock = 0.0
        self._last_now = 0.0
        self._mark = -math.inf          # the stamp of the newest beat taken, whatever became of it
        self._anchor: float | None = None
        self._stamp: float | None = None    # the stamp of the newest snapshot that a look has seen
        self._base: tuple[float, float] | None = None   # that stamp and the look that saw it, if prompt
        self._holding = False
        self._cause: str | None = None
        self._shares: dict[int, float] = {}     # the inhale share that each breath in progress began with
        self._mixed: set[int] = set()           # the breaths in which it changed
        self._share = 0.5
        self._count = dict.fromkeys(("taken", "before", "measured", "unmeasured", "voided", "held", "after"), 0)

    def _observe(self, now: float, state, cause: str | None) -> Look:
        """The part of a look that does not depend on the schedule. `cause` is what the session saw; a step of the looks
        that is too long (`stall`) and a step of the stamps (`clock`) are added here."""
        first = not self._begun
        self._begun = True
        step = now - self._last_now if not first else 0.0
        if math.isfinite(now):
            self._last_now = now
        if cause is None and not 0.0 <= step <= STALL_S:
            cause = "stall"
        stamp = state.t if state is not None else math.nan
        fresh = math.isfinite(stamp) and stamp != self._stamp
        if fresh:
            self._stamp = stamp
            if self._moved(stamp, now, not first and 0.0 <= step <= STALL_S):
                self._rebase(state)
                cause = cause or "clock"
        beats = self.take(state)
        self._cause = cause
        given = getattr(state, "arrived", None)
        if given is None or not given >= stamp:      # a source that gives a beat as it happens, or a spoiled value
            given = stamp
        return Look(beats, cause, step, first, stamp, fresh, given)

    def _anchored(self, look: Look) -> None:
        """Keep the anchor: each look is late, so the smallest lateness is the best. A look sees a snapshot
        no sooner than the source gave it, which is later than its stamp for a source that times a beat
        after the fact, so the lateness counts from then."""
        if math.isfinite(look.given) and (self._anchor is None or look.fresh):
            late = self._clock - look.given
            self._anchor = late if self._anchor is None else min(self._anchor, late)

    def _note_share(self, k: int | None, share: float) -> None:
        """A look in breath `k` ran with `share`."""
        self._share = share
        if k is not None and self._shares.setdefault(k, share) != share:
            self._mixed.add(k)

    def _close_share(self, k: int) -> tuple[float, bool]:
        """The share that breath `k` began with, and whether it changed during it."""
        lost = k in self._mixed
        self._mixed.discard(k)
        return self._shares.pop(k, self._share), lost

    def _forget_share(self, k: int) -> None:
        self._shares.pop(k, None)
        self._mixed.discard(k)

    def take(self, state) -> list[tuple[float, float]]:
        """The beats of `state` that are new, oldest first, as `(stamp, interval in ms)`.

        A beat is `(state.t - age_s[i], 60000 / tachogram[i])`. It is taken only when the stamp and the
        rate are finite, the rate is above 0 and the stamp is above that of the newest beat taken, so the
        same snapshot fed again, an older one and a spoiled value add nothing. It walks back from the
        newest beat and stops at the first old one, so a call costs what is new.
        """
        if state is None:
            return []
        t, ages, rates = state.t, state.age_s, state.tachogram
        found = []
        for i in range(min(len(ages), len(rates)) - 1, -1, -1):
            stamp = t - ages[i]
            if not math.isfinite(stamp):
                continue
            if stamp <= self._mark:
                break
            if math.isfinite(rates[i]) and rates[i] > 0.0:
                found.append((stamp, 60000.0 / rates[i]))
        taken, top = [], self._mark
        for stamp, rr in reversed(found):
            if stamp > top:
                taken.append((stamp, rr))
                top = stamp
        if taken:
            self._mark = top
            self._count["taken"] += len(taken)
        return taken

    def _moved(self, stamp: float, now: float, prompt: bool) -> bool:
        """Whether the stamps of two new snapshots are farther apart than the looks that saw them, by more
        than `CLOCK_STEP_S`. A snapshot is a baseline only when its look `prompt`ly followed the one before:
        the first look sees a snapshot of unknown age, and a late look sees one that is nearly as old as
        the lateness, which says nothing about the beat after it. A gap between beats, a dropout or a
        withheld beat moves a stamp and a look by the same amount, so none of them is a step of the clock."""
        base = self._base
        self._base = (stamp, now) if prompt and math.isfinite(now) else None
        return base is not None and abs((stamp - base[0]) - (now - base[1])) > CLOCK_STEP_S

    def _rebase(self, state) -> None:
        """Put the mark at the newest stamp of `state`, even if that is lower than the mark. After a clock
        that was set back, the old mark would refuse every beat for as long as the step is long."""
        for age in reversed(state.age_s):
            if math.isfinite(state.t - age):
                self._mark = state.t - age
                return


class Sweep(BeatRun):
    """The series of one sweep: the beats put into breaths, the swing of each breath, and the clock.

    `feed` runs once per draw. It owns a clock that grows only while nothing holds the sweep, so the pacer
    and the beats stay together, and an anchor that turns the stamp of a beat into a time on that clock.
    A beat is put into a breath once, when it is taken, and a breath is measured once, when a beat arrives
    that lies after its end.
    """

    def __init__(self, schedule: SweepSchedule, source_key: object = None):
        super().__init__(source_key)
        self.schedule = schedule
        self._reason: str | None = None
        self._result: Result | None = None
        self._breath = 0
        self._values: list[float | None] = [None] * schedule.breaths
        self._means: list[float | None] = [None] * schedule.breaths     # the mean interval of a measured breath
        self._steps: list[float | None] = [None] * schedule.breaths     # how late the heart rate is
        self._outcomes: list[Outcome] = []
        self._newest: float | None = None   # the latest time on the clock that a taken beat maps to
        self._pending: list[tuple[int, float, float]] = []      # (breath, time, interval in ms)

    @property
    def active(self) -> bool:
        return self._begun and self._reason is None

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def cause(self) -> str | None:
        """What held the sweep at the latest look, or None when that look ran it on."""
        return self._cause if self.active else None

    @property
    def clock(self) -> float:
        return self._clock

    @property
    def breath(self) -> int:
        """The earliest breath that is not final."""
        return self._breath

    @property
    def values(self) -> list:
        return list(self._values)

    @property
    def steps(self) -> list:
        """How late the heart rate is in each breath of the sweep, in seconds, None where a breath gives none."""
        return list(self._steps)

    @property
    def outcomes(self) -> list[Outcome]:
        return list(self._outcomes)

    @property
    def result(self) -> Result | None:
        return self._result

    def ledger(self) -> dict:
        """Every beat taken, by what became of it. The parts add up to `taken` at every look."""
        return {**self._count, "pending": len(self._pending)}

    def retarget(self, schedule: SweepSchedule) -> None:
        """Run on over a copy of the schedule with another end (`SweepSchedule.with_end`). Every measured
        breath that the new schedule has keeps its value. When the pacer is already past the new end, the sweep ends now through its normal end,
        with the breath in progress dropped, as a hold at the last breath does."""
        old = self.schedule
        if (schedule.start_per_min, schedule.step_s) != (old.start_per_min, old.step_s):
            raise ValueError("a running sweep takes a schedule with its own start and step")
        if self._reason is not None:
            return
        count = schedule.breaths
        self.schedule = schedule
        self._values = (self._values + [None] * count)[:count]
        self._means = (self._means + [None] * count)[:count]
        self._steps = (self._steps + [None] * count)[:count]
        self._breath = min(self._breath, count)
        self._finish_breaths()
        if self._reason is None and self._clock >= schedule.total_s:
            self.end("done")

    def end(self, reason: str) -> None:
        """Stop with `reason`. The measured breaths stay and the one in progress is dropped. The first
        reason stays."""
        if reason not in END_REASONS:
            raise ValueError(f"reason must be one of {END_REASONS}, not {reason!r}")
        if self._reason is not None:
            return
        self._reason = reason
        self._void_pending()
        measured = sum(v is not None for v in self._values)
        top = best(self.schedule.rates, self._values)
        self._result = Result(top, in_step(self.schedule.rates, self._steps, top.rate if top else None),
                              verdict(self.schedule.rates, self._values), heart_rate_ends(self._means),
                              measured,
                              self.schedule.breaths - measured, reason, self._key)

    def feed(self, now: float, state, cause: str | None = None, source_key: object = None,
             inhale_share: float = 0.5) -> None:
        """One look. `now` is seconds on the session's monotonic clock, in any origin, and `state` is the
        newest `BeatState` or None. `cause` holds the sweep: its breath in progress is voided and the clock
        goes back to its start. Another `source_key` than the sweep began with ends it. `inhale_share` is the
        share that the pacer turns at; a breath in which it changes gives a swing and no in-step value."""
        if self._reason is not None:
            return
        _check_cause(cause)
        if source_key != self._key:
            self.end("source")
            return
        look = self._observe(now, state, cause)
        if look.cause is not None:
            self._hold(look.cause, len(look.beats))
            return
        self._holding = False
        self._clock += look.step
        self._note_share(self.schedule.breath_at(self._clock), inhale_share)
        self._anchored(look)
        if look.first:
            self._count["before"] += len(look.beats)    # a snapshot that is there at the start is history
        else:
            for beat_stamp, rr in look.beats:
                self._place(beat_stamp + self._anchor, rr)
        self._finish_breaths()

    def _place(self, s: float, rr: float) -> None:
        """Put one beat at time `s` into the breath that owns it, or count it as before or after."""
        k = self.schedule.breath_at(s)
        if self._newest is None or s > self._newest:
            self._newest = s
        if k is None:
            self._count["before" if s < 0.0 else "after"] += 1
        elif k < self._breath:
            self._count["before"] += 1      # the breath it belongs to is already final
        else:
            self._pending.append((k, s, rr))

    def _finish_breaths(self) -> None:
        """Measure every breath that a taken beat has passed, in order, and end after the last."""
        n = self.schedule.breaths
        while (self._breath < n and self._newest is not None
               and self._newest >= self.schedule.start_s(self._breath + 1)):
            k = self._breath
            mine = [p for p in self._pending if p[0] == k]
            self._pending = [p for p in self._pending if p[0] != k]
            share, lost = self._close_share(k)
            got = paced_swing([p[1] for p in mine], [p[2] for p in mine],
                              self.schedule.start_s(k), self.schedule.start_s(k + 1), share)
            self._values[k] = got.swing_ms
            self._steps[k] = None if lost else got.in_step_s
            self._means[k] = float(np.mean([p[2] for p in mine])) if got.swing_ms is not None else None
            self._outcomes.append(Outcome(k, got.swing_ms, got.reason))
            self._count["measured" if got.swing_ms is not None else "unmeasured"] += len(mine)
            self._breath += 1
        if self._breath == n:
            self.end("done")

    def _hold(self, cause: str, held: int) -> None:
        """A look under a cause: the beats are taken and belong to no breath. The first look of a hold
        voids the breath in progress and any that waits for its last beat, puts the clock back to the start
        of the earliest of them, and asks for a new anchor. A hold after the end of the last breath, while
        that breath waits for a beat, ends the sweep with it not measured."""
        self._count["held"] += held
        if self._holding:
            return
        self._holding = True
        k = self._breath
        self._outcomes.append(Outcome(k, None, cause))
        self._void_pending()
        self._forget_share(k)
        self._anchor = None
        if k == self.schedule.breaths - 1 and self._clock >= self.schedule.total_s:
            self.end("done")
        else:
            self._clock = self.schedule.start_s(k)

    def _void_pending(self) -> None:
        self._count["voided"] += len(self._pending)
        self._pending = []
