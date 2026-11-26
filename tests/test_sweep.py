"""The sliding rate search, as a pure module: the schedule, the swing of one breath, the curve over the
breaths, and the series that a run of beats builds. No screen and no clock: `Sweep.feed` takes the time.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from conftest import SWEEP_MONO0, beat_state, pacer_beats, run_sweep, schedule_of

def _mod():
    import nfb.hrv.sweep as module
    return module


def _short(breaths=10):
    return schedule_of(breaths)


def _flat(ms):
    return lambda rate: ms


def _peaked(at, base=20.0, top=80.0, width=0.35):
    return lambda rate: base + (top - base) * math.exp(-0.5 * ((rate - at) / width) ** 2)


def _breath_beats(schedule, stamps, rr, k):
    lo = int(np.searchsorted(stamps, schedule.start_s(k), side="left"))
    hi = int(np.searchsorted(stamps, schedule.start_s(k) + schedule.period_s(k), side="left"))
    return stamps[lo:hi], rr[lo:hi]


def _swings(schedule, stamps, rr):
    m = _mod()
    out = []
    for k in range(schedule.breaths):
        s, r = _breath_beats(schedule, stamps, rr, k)
        out.append(m.paced_swing(s, r, schedule.start_s(k), schedule.start_s(k) + schedule.period_s(k)).swing_ms)
    return out


# --- the turn of a breath ------------------------------------------------------------------------

@pytest.mark.parametrize("share", [0.01, 0.5, 0.99])
def test_a_breath_turns_to_the_exhale_exactly_at_the_share_and_each_half_runs_from_zero_to_one(share):
    """The inhale owns the start of the breath and not the share itself, and the ring phase of a half
    is 0 where the half begins and 1 where it ends."""
    m = _mod()
    assert m.breath_turn(0.0, share) == (True, 0.0)
    assert m.breath_turn(share, share) == (False, 0.0)
    inhaling, phase = m.breath_turn(math.nextafter(share, 0.0), share)
    assert inhaling and phase == pytest.approx(1.0)
    assert m.breath_turn(math.nextafter(1.0, 0.0), share)[1] == pytest.approx(1.0)


@pytest.mark.parametrize("share", [0.0, 1.0, -0.2, 1.5, math.nan, math.inf])
def test_a_share_that_leaves_no_inhale_or_no_exhale_is_refused(share):
    """With no inhale or no exhale there is no turn to find, so the call raises and does not guess."""
    with pytest.raises(ValueError):
        _mod().breath_turn(0.3, share)


# --- the schedule ------------------------------------------------------------------------------

RANGES = [(6.75, 4.25), (4.25, 6.75), (8.0, 3.0), (3.5, 7.5)]
"""Start and end in breaths a minute: slowing and quickening over the default range, and a wider range
in each direction."""


@pytest.fixture(params=RANGES, ids=lambda r: f"{r[0]}-to-{r[1]}")
def sch(request):
    return _mod().SweepSchedule(*request.param)


def test_the_period_changes_by_the_same_step_at_every_breath_in_the_direction_of_the_range(sch):
    """Between two ends, the period of breath k+1 differs from the period of breath k by one constant
    step, up when the sweep slows and down when it quickens, and it starts at the start rate."""
    steps = np.diff([sch.period_s(k) for k in range(sch.breaths)])
    assert np.allclose(steps, steps[0], atol=1e-9)
    assert abs(steps[0]) == pytest.approx(sch.step_s)
    assert (steps[0] > 0) == (sch.end_per_min < sch.start_per_min)
    assert sch.rates[0] == pytest.approx(sch.start_per_min)
    past = (sch.period_s(sch.breaths - 1) - 60.0 / sch.end_per_min) * (1.0 if steps[0] > 0 else -1.0)
    assert -sch.step_s < past <= 1e-9, "the last breath is the last that does not pass the end"
    assert sch.low_per_min <= min(sch.rates) and max(sch.rates) <= sch.high_per_min + 1e-9


def test_the_count_and_the_total_time_follow_from_the_range():
    """The step is constant in seconds, so a wider range has more breaths and takes longer, and the same
    range in the other direction has as many breaths and takes as long."""
    m = _mod()
    default, wide = m.SweepSchedule(6.75, 4.25), m.SweepSchedule(8.0, 3.0)
    assert wide.breaths > default.breaths and wide.total_s > default.total_s
    assert m.SweepSchedule(4.25, 6.75).breaths == default.breaths
    assert m.SweepSchedule(4.25, 6.75).total_s == pytest.approx(default.total_s)
    narrow = m.SweepSchedule(6.0, 5.0)
    assert 2 <= narrow.breaths < default.breaths


def test_breath_starts_add_up_to_the_total(sch):
    """Each breath starts when the one before it ends, and the last ends at `total_s`."""
    for k in range(sch.breaths - 1):
        assert sch.start_s(k + 1) == pytest.approx(sch.start_s(k) + sch.period_s(k))
    assert sch.start_s(0) == 0.0
    assert sch.start_s(sch.breaths - 1) + sch.period_s(sch.breaths - 1) == pytest.approx(sch.total_s)


def test_a_stamp_belongs_to_exactly_one_breath(sch):
    """`breath_at` is half open: a breath owns its start and not its end, and nothing outside the sweep
    belongs to any breath."""
    for k in range(sch.breaths):
        start = sch.start_s(k)
        assert sch.breath_at(start) == k
        assert sch.breath_at(start + sch.period_s(k) / 2) == k
        end = sch.start_s(k + 1) if k + 1 < sch.breaths else sch.total_s
        assert sch.breath_at(end - 1e-6) == k
        assert sch.breath_at(end) == (k + 1 if k + 1 < sch.breaths else None)
    assert sch.breath_at(-1e-9) is None
    assert sch.breath_at(sch.total_s) is None
    assert sch.breath_at(sch.total_s + 100.0) is None
    assert sch.breath_at(math.nan) is None


@pytest.mark.parametrize("start,end,other,step", [(6.75, 4.25, 3.0, 0.0679), (6.75, 4.25, 5.5, 0.0679),
                                                   (4.25, 6.75, 8.0, 0.0679), (4.25, 6.75, 5.5, 0.0679),
                                                   (8.0, 3.0, 4.0, 0.2), (6.0, 5.0, 5.5, 0.2)])
def test_a_copy_with_another_end_keeps_the_numbers_and_start_times_of_the_breaths_that_exist_in_both(start, end, other, step):
    """Moving the end changes how many breaths follow and nothing of the breaths before: a breath that
    both schedules have has the same period and starts at the same second."""
    first = _mod().SweepSchedule(start, end, step)
    moved = first.with_end(other)
    assert (moved.start_per_min, moved.step_s, moved.end_per_min) == (start, first.step_s, other)
    assert moved.breaths != first.breaths
    for k in range(min(first.breaths, moved.breaths)):
        assert moved.period_s(k) == first.period_s(k) and moved.start_s(k) == first.start_s(k)


@pytest.mark.parametrize("kw", [dict(start_per_min=math.nan), dict(end_per_min=math.inf), dict(end_per_min=0.0),
                                dict(start_per_min=-1.0), dict(end_per_min=6.75), dict(step_s=0.0),
                                dict(step_s=math.nan), dict(step_s=-0.1)])
def test_a_schedule_that_cannot_be_run_is_refused(kw):
    """A rate or a step that is not finite or not above 0, and a start equal to the end, all raise."""
    with pytest.raises(ValueError):
        _mod().SweepSchedule(**kw)


def test_the_phase_runs_forward_without_a_jump():
    """Over the sweep, breaths and the part of the breath that has passed grow without a step back, and
    the part is at least 0 and under 1."""
    sch = _short()
    grid = np.linspace(0.0, sch.total_s, 4000, endpoint=False)
    cycles = []
    for s in grid:
        k, frac = sch.phase_at(float(s))
        assert 0.0 <= frac < 1.0
        cycles.append(k + frac)
    assert np.all(np.diff(cycles) > 0)
    assert np.max(np.diff(cycles)) < 0.01, "no breath is skipped between two near stamps"


# --- the swing of one breath -------------------------------------------------------------------

def test_a_planted_swing_is_read_back_at_both_ends_of_the_sweep():
    """With no noise, the swing that `paced_swing` reads in the first and in the last breath is the swing
    that the heart carries, whatever the length of the breath."""
    sch = _mod().SweepSchedule()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0))
    for k in (0, sch.breaths - 1):
        s, r = _breath_beats(sch, stamps, rr, k)
        got = _mod().paced_swing(s, r, sch.start_s(k), sch.start_s(k) + sch.period_s(k))
        assert got.reason is None
        assert got.swing_ms == pytest.approx(40.0, rel=0.05)


# --- how early or late the heart rate is, against the paced breath -------------------------------

SHARES = (0.01, 0.3, 0.5, 0.7, 0.99)
PERIODS_S = (8.9, 11.0, 14.1)


@functools.lru_cache(maxsize=None)
def _ring_peak(share):
    """Where in the breath the fundamental of the ring's size peaks, read from the ring's own turn function
    and a transform, so it does not repeat the formula under test."""
    frac = (np.arange(4000) + 0.5) / 4000
    size = np.array([ph if up else 1.0 - ph for up, ph in (_mod().breath_turn(f, share) for f in frac)])
    return float(np.angle(np.fft.rfft(size)[1].conjugate()) / (2.0 * np.pi)) % 1.0


def _heart_wave(period_s, share, shift_s, *, amp_bpm=3.0, noise_ms=0.0, seed=0, spacing_s=0.85, at=None):
    """The beats of one breath, 0 to `period_s`, whose heart rate has one cycle that peaks `shift_s` after
    the fundamental of the paced volume peaks, or at the part `at` of the breath when that is given. Returns
    stamps and intervals in ms."""
    rng = np.random.default_rng(seed)
    peak = (_ring_peak(share) + shift_s / period_s) if at is None else at
    stamps = np.arange(0.1, period_s, spacing_s)
    rr = np.full(stamps.size, 60000.0 / 66.0)
    for _ in range(4):      # an interval ends at its beat and is the rate at its middle
        hr = 66.0 + amp_bpm * np.cos(2.0 * np.pi * ((stamps - rr / 2000.0) / period_s - peak))
        rr = 60000.0 / hr
    return stamps, rr + rng.normal(0.0, noise_ms, stamps.size)


@pytest.mark.parametrize("share", SHARES)
def test_the_peak_of_the_paced_volume_is_the_peak_of_its_fundamental_at_every_share(share):
    """The reference that in-step is read against is where the fundamental of the ring's size peaks, from
    1 to 99 percent of inhale. It is the end of the inhale for an even breath only."""
    assert _mod().volume_peak(share) == pytest.approx(_ring_peak(share), abs=2e-3)
    assert _mod().volume_peak(0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("period_s", PERIODS_S)
@pytest.mark.parametrize("share", SHARES)
@pytest.mark.parametrize("shift_s", [-1.5, 0.0, 2.0])
def test_a_heart_rate_wave_that_peaks_later_than_the_breath_is_late_by_that_many_seconds(period_s, share, shift_s):
    """In step is a heart rate that peaks with the fundamental of the paced volume: it reads 0. A wave that
    peaks later reads positive and one that peaks earlier reads negative, and the value in seconds is the
    shift, for every length of breath and every share of the inhale."""
    stamps, rr = _heart_wave(period_s, share, shift_s)
    got = _mod().paced_swing(stamps, rr, 0.0, period_s, share)
    assert got.reason is None and got.in_step_s is not None
    assert got.in_step_s == pytest.approx(shift_s, abs=0.05)


def test_a_heart_rate_that_peaks_at_the_end_of_an_even_inhale_is_in_step():
    """For the even breath the end of the inhale is half the breath, and a wave that peaks there is in step."""
    stamps, rr = _heart_wave(10.0, 0.5, 0.0, at=0.5)
    assert _mod().paced_swing(stamps, rr, 0.0, 10.0, 0.5).in_step_s == pytest.approx(0.0, abs=0.05)


def test_the_in_step_value_wraps_into_half_a_breath_either_way():
    """A wave that peaks 0.6 of a breath late is 0.4 of a breath early: the value stays between plus and
    minus half a breath."""
    stamps, rr = _heart_wave(10.0, 0.5, 6.0)
    assert _mod().paced_swing(stamps, rr, 0.0, 10.0).in_step_s == pytest.approx(-4.0, abs=0.05)
    stamps, rr = _heart_wave(10.0, 0.99, 3.5)       # the reference is late in the breath, the wave passes the end
    assert _mod().paced_swing(stamps, rr, 0.0, 10.0, 0.99).in_step_s == pytest.approx(3.5, abs=0.05)


def test_the_paced_swing_is_the_same_at_every_share_of_the_inhale():
    """The share only moves the reference of the phase, so the swing of one planted wave does not change."""
    stamps, rr = _heart_wave(11.0, 0.5, 0.0)
    reads = [_mod().paced_swing(stamps, rr, 0.0, 11.0, share).swing_ms for share in SHARES]
    assert max(reads) - min(reads) < 1e-9


@pytest.mark.parametrize("share", [0.0, 1.0, -0.2, 1.5, math.nan, math.inf])
def test_a_breath_with_no_inhale_or_no_exhale_has_no_reference_and_raises(share):
    """The reference of the phase needs an inhale and an exhale, as the turn of a breath does."""
    stamps, rr = _heart_wave(10.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        _mod().paced_swing(stamps, rr, 0.0, 10.0, share)


def test_a_breath_whose_swing_is_not_above_its_noise_gives_no_in_step_value():
    """With only noise, the phase is random, so it must be given in at most one breath in ten of a
    fixed set of runs. With a swing far above the noise, every breath gives one. A swing is read in both."""
    m = _mod()
    noise_only = [m.paced_swing(*_heart_wave(11.0, 0.5, 0.0, amp_bpm=0.0, noise_ms=20.0, seed=seed), 0.0, 11.0).in_step_s
                  for seed in range(200)]
    assert sum(v is not None for v in noise_only) <= 20
    strong = [m.paced_swing(*_heart_wave(11.0, 0.5, 0.5, amp_bpm=4.0, noise_ms=20.0, seed=seed), 0.0, 11.0)
              for seed in range(50)]
    late = [r.in_step_s for r in strong]
    assert None not in late and abs(np.mean(late) - 0.5) < 0.15 and np.std(late) < 0.5
    assert all(r.swing_ms > 0.0 for r in strong)


def test_noise_alone_reads_near_zero_at_both_ends_of_the_sweep():
    """With no swing and 20 ms of noise in the interval, the middle of what `paced_swing` reads is under
    half of a real swing of 40 ms, and a fit that does not take the noise off reads about twice that. It
    holds at the end of the short breaths and at the end of the long ones."""
    sch = _mod().SweepSchedule()
    ends = (0, sch.breaths - 1)
    reads = {k: [] for k in ends}
    for seed in range(40):
        stamps, rr, _ = pacer_beats(sch, _flat(0.0), noise_ms=20.0, seed=seed, lead_s=0.0, tail_s=0.0)
        for k in ends:
            s, r = _breath_beats(sch, stamps, rr, k)
            reads[k].append(_mod().paced_swing(s, r, sch.start_s(k), sch.start_s(k) + sch.period_s(k)).swing_ms)
    for k in ends:
        assert None not in reads[k]
        assert np.median(reads[k]) < 18.0


@pytest.mark.parametrize("mean_ms", [923.0, 1200.0])
def test_noise_alone_reads_the_same_in_a_short_breath_and_in_a_long_one(mean_ms):
    """With no swing at all, the median of what `paced_swing` reads must not lean toward the short
    breaths, because the curve is made of medians and a lean at the fast end reads as a peak there.
    (The mean of the reads falls with the number of beats whatever the offset, so it is not compared.)"""
    rng = np.random.default_rng(0)

    def median_read(period_s, count=1000):
        reads = []
        for _ in range(count):
            at = float(rng.uniform(0.0, mean_ms / 1000.0))
            stamps, rr = [], []
            while at < period_s:
                stamps.append(at)
                rr.append(mean_ms + float(rng.normal(0.0, 14.0)))
                at += rr[-1] / 1000.0
            swing = _mod().paced_swing(stamps, rr, 0.0, period_s).swing_ms
            reads.append(0.0 if swing is None else swing)
        return float(np.median(reads))

    short, long_ = median_read(60.0 / 6.75), median_read(60.0 / 4.25)
    assert abs(short - long_) < 1.0 and max(short, long_) < 1.0


def test_four_beats_with_one_in_each_quarter_are_still_too_few_to_measure():
    """A breath needs more beats than the fit has values to find and the noise has to be estimated from,
    so four beats that cover every quarter give no swing, and a fifth gives one."""
    m = _mod()
    _, _, lo, hi = _good_breath()
    stamps, rr = np.array([0.5, 3.0, 6.0, 8.5, 9.0]), np.array([900.0, 950.0, 850.0, 900.0, 920.0])
    assert m.paced_swing(stamps, rr, lo, hi).swing_ms is not None
    got = m.paced_swing(stamps[:4], rr[:4], lo, hi)
    assert got.swing_ms is None and got.reason in m.REASONS


def _good_breath():
    stamps = np.array([0.5, 2.5, 4.5, 6.5, 8.5])
    return stamps, np.array([900.0, 950.0, 900.0, 850.0, 900.0]), 0.0, 10.0


@pytest.mark.parametrize("case", ["four beats", "three quarters", "equal stamps", "stamps fall",
                                  "one stamp", "nan stamp", "inf stamp", "nan interval", "inf interval"])
def test_a_breath_that_cannot_be_fitted_is_not_measured_and_says_why(case):
    """Too few beats, a breath with a quarter that holds none, stamps that do not strictly rise and
    values that are not finite give no swing and one of the named reasons. A good breath gives a swing."""
    m = _mod()
    stamps, rr, lo, hi = _good_breath()
    assert m.paced_swing(stamps, rr, lo, hi).swing_ms is not None
    if case == "four beats":
        stamps, rr = stamps[:4], rr[:4]
    elif case == "three quarters":
        stamps, rr = np.array([0.5, 1.0, 1.5, 2.0, 7.0, 9.0]), np.array([900.0, 950, 900, 850, 900, 880])
    elif case == "equal stamps":
        stamps = np.array([0.5, 2.5, 4.5, 4.5, 8.5])
    elif case == "stamps fall":
        stamps = stamps[::-1].copy()
    elif case == "one stamp":
        stamps = np.full(5, 3.0)
    elif case == "nan stamp":
        stamps[2] = math.nan
    elif case == "inf stamp":
        stamps[2] = math.inf
    elif case == "nan interval":
        rr[2] = math.nan
    elif case == "inf interval":
        rr[2] = math.inf
    got = m.paced_swing(stamps, rr, lo, hi)
    assert got.swing_ms is None
    assert got.reason in m.REASONS


def test_a_few_beats_in_a_part_of_the_breath_are_refused_where_they_would_fit_a_huge_swing():
    """Four beats that crowd a third of the breath fit any swing, so a breath with a quarter that has no
    beat is not measured, and the same breath with beats in every quarter is."""
    m = _mod()
    crowded = np.array([0.2, 1.0, 1.8, 2.6, 3.0, 3.2])
    spread = np.array([0.5, 2.5, 5.0, 7.5, 9.0, 9.5])
    rr = np.array([900.0, 1100.0, 800.0, 1000.0, 700.0, 900.0])
    assert m.paced_swing(crowded, rr, 0.0, 10.0).swing_ms is None
    assert m.paced_swing(spread, rr, 0.0, 10.0).swing_ms is not None


def test_the_best_rate_follows_a_planted_peak_while_the_mean_rate_falls():
    """A peak planted at one rate is found there within a small error, over several seeds, when the mean
    heart rate falls through the sweep by a quarter."""
    sch = _mod().SweepSchedule()
    plant = 5.4
    assert min(sch.rates) < plant < max(sch.rates)
    errors = []
    for seed in range(5):
        stamps, rr, _ = pacer_beats(sch, _peaked(plant), mean_ms=750.0, mean_end_ms=1090.0,
                                    noise_ms=10.0, seed=seed)
        got = _mod().best(sch.rates, _swings(sch, stamps, rr))
        errors.append(got.rate - plant)
    assert abs(np.mean(errors)) < 0.15
    assert np.sqrt(np.mean(np.square(errors))) < 0.35


# --- the curve, the best rate and the verdict --------------------------------------------------

def _values(rng, count, measured):
    values = [None] * count
    for k in measured:
        values[k] = float(rng.uniform(10.0, 90.0))
    return values


def test_the_curve_has_a_point_for_each_supported_window_and_reads_neighbours_only():
    """A point is made for each window of neighbouring breath numbers that holds enough measured breaths,
    and for no other, in breath order. Its median and rate are those of the measured breaths in that window,
    never of breaths that lie on the far side of a stretch that was not measured."""
    m = _mod()
    rng = np.random.default_rng(3)
    count = 60
    rates = [6.0 - 0.01 * k for k in range(count)]
    for measured_count in (1, 2, m.WINDOW - 1, m.WINDOW, m.WINDOW + 1, 40):
        measured = sorted(rng.choice(count, measured_count, replace=False).tolist())
        values = _values(rng, count, measured)
        points = m.curve(rates, values)
        assert points, "something is measured, so something is shown"
        assert [p.first for p in points] == sorted({p.first for p in points})
        most = max(sum(v is not None for v in values[s:s + m.WINDOW]) for s in range(count - m.WINDOW + 1))
        for p in points:
            near = [k for k in range(p.first, p.first + min(m.WINDOW, count)) if values[k] is not None]
            assert len(near) >= min(m.SUPPORT, most)
            assert p.median_ms == pytest.approx(float(np.median([values[k] for k in near])))
            assert p.window_rate == pytest.approx(float(np.mean([rates[k] for k in near])))
    one = [None] * count
    one[7] = 55.0
    assert [(p.first, p.median_ms) for p in m.curve(rates, one)] == [(s, 55.0) for s in range(8)]


def test_the_best_rate_exists_from_the_first_measured_breath_and_is_the_highest_supported_window():
    """With nothing measured there is no best rate; with one measured breath there is one. Whatever is
    measured, the best window is a span of neighbouring breath numbers, its rate is the mean rate of the
    breaths measured in it, and no supported window of that span has a higher median."""
    m = _mod()
    rates = [6.5 - 0.02 * k for k in range(30)]
    assert m.best(rates, [None] * 30) is None
    one = [None] * 30
    one[7] = 55.0
    got = m.best(rates, one)
    assert got is not None and got.breaths == (7,) and got.swing_ms == 55.0
    assert got.rate == pytest.approx(rates[7])
    rng = np.random.default_rng(5)
    for measured_count in (2, 5, m.WINDOW, 17, 30):
        values = _values(rng, 30, sorted(rng.choice(30, measured_count, replace=False).tolist()))
        got = m.best(rates, values)
        assert got.breaths[-1] - got.breaths[0] < m.WINDOW, "one span of neighbouring breath numbers"
        assert all(values[k] is not None for k in got.breaths)
        assert got.rate == pytest.approx(np.mean([rates[k] for k in got.breaths]))
        assert got.swing_ms == pytest.approx(np.median([values[k] for k in got.breaths]))
        assert got.swing_ms == pytest.approx(max(p.median_ms for p in m.curve(rates, values)))


@pytest.mark.parametrize("start,end", RANGES)
def test_the_figure_at_a_rate_is_the_swing_of_the_breaths_paced_near_that_rate_in_either_direction(start, end):
    """Each breath's swing is a function of its own rate. The figure at a rate is then that function near the
    rate, whichever way the sweep runs."""
    m = _mod()
    rates = m.SweepSchedule(start, end).rates
    values = [100.0 * r for r in rates]
    lo, hi = min(rates), max(rates)
    for rate in np.linspace(lo + 0.3, hi - 0.3, 7):
        assert m.swing_at(rates, values, float(rate)) == pytest.approx(100.0 * rate, abs=15.0), rate


def test_a_stretch_that_was_not_measured_joins_nothing_in_the_curve_the_best_rate_or_the_figure_at_a_rate():
    """Breaths 10 to 13 and 50 to 54 are measured and nothing between them. No window reaches across the
    hole, so the best rate is a rate within one window of a measured breath, and a rate in the hole has no
    figure while a rate beside the measured breaths has one."""
    m = _mod()
    rates = m.SweepSchedule().rates
    keep = list(range(10, 14)) + list(range(50, 55))
    values = [60.0 + k if k in keep else None for k in range(len(rates))]
    got = m.best(rates, values)
    assert got.breaths[-1] - got.breaths[0] < m.WINDOW
    assert min(abs(got.rate - rates[k]) for k in keep) < 0.5 * m.WINDOW * (rates[0] - rates[-1]) / len(rates)
    assert all(p.first + m.WINDOW > 10 and p.first <= 54 for p in m.curve(rates, values))
    assert all(p.first + m.WINDOW <= 14 or p.first >= 46 for p in m.curve(rates, values))
    assert m.swing_at(rates, values, rates[30]) is None
    assert m.swing_at(rates, values, rates[52]) is not None
    assert m.swing_at(rates, values, max(rates) + 0.5) is None and m.swing_at(rates, values, 1.0) is None
    assert m.swing_at(rates, [None] * len(rates), rates[30]) is None


def test_the_verdict_needs_enough_measured_breaths_and_gives_the_same_answer_twice():
    """Under the number of measured breaths that the verdict needs, it says that there are too few and
    gives no rise. At that number it judges, and the same breaths give the same answer however the global
    random state stands."""
    m = _mod()
    rng = np.random.default_rng(1)
    rates = [6.7 - 0.03 * k for k in range(78)]
    for count in (0, m.MIN_FOR_VERDICT - 1):
        values = _values(rng, 78, list(range(count)))
        assert m.verdict(rates, values).kind == "few"
    values = _values(rng, 78, list(range(m.MIN_FOR_VERDICT)))
    first = m.verdict(rates, values)
    assert first.kind != "few" and first.measured == m.MIN_FOR_VERDICT
    np.random.seed(0)
    import random
    random.seed(0)
    assert m.verdict(rates, values) == first
    assert (first.kind == "clear") == (first.reached <= m.CLEAR_SHARE * first.tries)
    assert first.rise_ms == pytest.approx(m.best(rates, values).swing_ms - first.outside_ms)


def test_flat_curves_are_rarely_clear_and_a_planted_peak_usually_is():
    """Over fixed seeds, white scatter around one level is called clear in a small share of runs, and the
    same scatter around a narrow planted peak is called clear in most."""
    m = _mod()
    rates = _mod().SweepSchedule().rates
    flat_clear = peak_clear = 0
    runs = 30
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        flat = list(40.0 + rng.normal(0.0, 14.0, len(rates)))
        peak = [v + _peaked(5.4, 0.0, 60.0, 0.3)(r) for v, r in zip(flat, rates)]
        flat_clear += m.verdict(rates, flat).kind == "clear"
        peak_clear += m.verdict(rates, peak).kind == "clear"
    assert flat_clear <= 0.2 * runs
    assert peak_clear >= 0.75 * runs


def _alike(rng, count, phi):
    """Scatter of sd 8 ms in which each breath resembles the one before it by `phi`."""
    e = rng.normal(size=count)
    x = np.zeros(count)
    x[0] = e[0]
    for i in range(1, count):
        x[i] = phi * x[i - 1] + math.sqrt(1.0 - phi ** 2) * e[i]
    return 8.0 * x


@pytest.mark.parametrize("phi", [0.0, 0.3, 0.6, 0.8])
def test_a_series_with_no_peak_is_rarely_clear_also_when_neighbouring_breaths_are_alike(phi):
    """Slow drift in the swings is not a peak: with no planted peak and neighbours that are alike, at most
    about one run in ten says that the peak is clear."""
    m = _mod()
    rates = m.SweepSchedule().rates
    runs = 40
    clear = 0
    for seed in range(runs):
        scatter = _alike(np.random.default_rng(seed), len(rates), phi)
        clear += m.verdict(rates, list(np.maximum(0.0, 20.0 + scatter))).kind == "clear"
    assert clear <= 0.2 * runs


def test_a_planted_peak_is_clear_in_most_runs_also_when_neighbouring_breaths_are_alike():
    """The test that refuses drift must still find an 80 ms peak over a 20 ms floor."""
    m = _mod()
    rates = m.SweepSchedule().rates
    runs = 40
    clear = 0
    for seed in range(runs):
        scatter = _alike(np.random.default_rng(seed), len(rates), 0.6)
        values = [max(0.0, _peaked(5.4, 20.0, 80.0, 0.35)(r) + x) for r, x in zip(rates, scatter)]
        clear += m.verdict(rates, values).kind == "clear"
    assert clear > 0.5 * runs


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["rising", "falling"])
def test_a_steady_slope_across_the_range_ends_as_the_edge_verdict(sign):
    """A swing that rises or falls steadily over the whole sweep has its best breaths at one end, so the
    verdict marks an edge and does not place a peak inside the range."""
    m = _mod()
    rates = m.SweepSchedule().rates
    runs = 40
    edge = 0
    for seed in range(runs):
        noise = np.random.default_rng(seed).normal(0.0, 8.0, len(rates))
        values = list(np.maximum(0.0, 20.0 + sign * np.linspace(-15.0, 15.0, len(rates)) + noise))
        edge += m.verdict(rates, values).edge
    assert edge >= 0.9 * runs


def test_the_shoulders_of_a_peak_are_not_part_of_the_breaths_it_is_compared_with():
    """The rise is measured against the breaths that are farther from the best window than half a window,
    so breaths that are raised next to the peak do not lift the level that the peak is compared with."""
    m = _mod()
    count = m.MIN_FOR_VERDICT
    rates = [6.7 - 0.05 * k for k in range(count)]
    values = [float(k % 5) for k in range(count)]
    mid = count // 2
    for k in range(mid - m.WINDOW, mid + m.WINDOW + 1):
        values[k] = 100.0
    for k in range(mid - m.WINDOW // 2, mid + m.WINDOW // 2 + 1):
        values[k] = 200.0
    window = m.best(rates, values).breaths
    reach = m.WINDOW // 2
    far = [values[k] for k in range(count) if k < window[0] - reach or k > window[-1] + reach]
    beside = [values[k] for k in range(count) if k < window[0] or k > window[-1]]
    assert far and float(np.median(far)) != float(np.median(beside))
    assert m.verdict(rates, values).outside_ms == pytest.approx(float(np.median(far)))


def test_a_clear_peak_at_either_end_of_the_sweep_is_marked_as_an_edge_and_one_inside_is_not():
    """The best window that sits at the first or the last breaths is an edge, so the screen can say that
    the true peak can lie beyond the range."""
    m = _mod()
    rates = _mod().SweepSchedule().rates
    n = len(rates)

    def with_high(where):
        values = list(20.0 + np.random.default_rng(2).normal(0.0, 1.0, n))
        for k in where:
            values[k] += 60.0
        return values

    first = m.verdict(rates, with_high(range(0, 9)))
    last = m.verdict(rates, with_high(range(n - 9, n)))
    middle = m.verdict(rates, with_high(range(35, 46)))
    assert first.kind == last.kind == middle.kind == "clear"
    assert first.edge and last.edge and not middle.edge


def test_a_best_window_among_the_first_or_last_windows_is_an_edge_even_when_it_is_not_the_very_first():
    """The ends of the range are the first few windows and the last few, so a peak that sits a few
    breaths in from an end is still reported at the end. The best window is the earliest one that holds
    the whole raised block, here the third from either end."""
    m = _mod()
    rates = m.SweepSchedule().rates
    n = len(rates)
    for block in (range(6, 11), range(n - 7, n - 2)):
        values = [80.0 if k in block else 20.0 for k in range(n)]
        got = m.verdict(rates, values)
        assert m.best(rates, values).breaths[0] in (2, n - 11)
        assert got.edge, block


def test_a_series_that_reaches_the_share_of_tries_that_is_allowed_exactly_is_clear(monkeypatch):
    """The share that the peak may reach is allowed, not just under it: with the share set to the count of
    tries that one series reaches, that series is clear, and with one try less it is not."""
    m = _mod()
    rates = m.SweepSchedule().rates
    values = list(20.0 + np.random.default_rng(0).normal(0.0, 8.0, len(rates)))
    reached = m.verdict(rates, values).reached
    assert 0 < reached < m.TRIES
    monkeypatch.setattr(m, "CLEAR_SHARE", reached / m.TRIES)
    assert (reached / m.TRIES) * m.TRIES == reached
    assert m.verdict(rates, values).kind == "clear"
    monkeypatch.setattr(m, "CLEAR_SHARE", (reached - 1) / m.TRIES)
    assert m.verdict(rates, values).kind == "not_clear"


# --- the series that a run of beats builds -----------------------------------------------------

def _run(breaths=10, **kw):
    m = _mod()
    sch = _short(breaths)
    stamps, rr, _ = pacer_beats(sch, kw.pop("plant", _flat(40.0)), noise_ms=kw.pop("noise_ms", 4.0),
                                seed=kw.pop("seed", 0), rests=kw.pop("rests", ()))
    sw = m.Sweep(sch)
    run_sweep(sw, stamps, rr, until_s=sch.total_s + 40.0, **kw)
    return sch, stamps, rr, sw


def _measured(sw):
    return [v for v in sw.values if v is not None]


def test_the_same_snapshot_fed_again_changes_nothing():
    """A snapshot that the sweep has seen adds no beat: ten feeds of one state give the series of one,
    and a missing state is accepted."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    once, ten = m.Sweep(sch), m.Sweep(sch)
    once.feed(SWEEP_MONO0, None)
    ten.feed(SWEEP_MONO0, None)
    for at in np.arange(0.5, sch.total_s + 12.0, 0.5):
        state = beat_state(stamps, rr, float(at))
        once.feed(SWEEP_MONO0 + float(at), state)
        for _ in range(10):
            ten.feed(SWEEP_MONO0 + float(at), state)
    assert ten.values == once.values and ten.ledger() == once.ledger()
    assert ten.outcomes == once.outcomes and _measured(once)


@pytest.mark.parametrize("field, bad", [
    ("age_s", math.nan), ("age_s", math.inf), ("age_s", -math.inf),
    ("tachogram", math.nan), ("tachogram", math.inf), ("tachogram", -math.inf),
    ("tachogram", 0.0), ("tachogram", -50.0)])
def test_a_beat_that_is_not_finite_or_not_a_rate_is_never_taken_and_later_beats_flow(field, bad):
    """A newest beat that comes with a non-finite stamp or rate, or a rate not above zero, is not taken
    as it stands, and the beats after it flow: the series ends as it does when every beat is clean."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    clean, spoiled = m.Sweep(sch), m.Sweep(sch)
    for at in np.arange(0.0, sch.total_s + 12.0, 0.5):
        state = beat_state(stamps, rr, float(at))
        if 28.0 <= at <= 32.0 and state.age_s:
            getattr(state, field)[-1] = bad
        spoiled.feed(SWEEP_MONO0 + float(at), state)
        clean.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at)))
    assert spoiled.values == clean.values and _measured(clean)
    assert spoiled.ledger() == clean.ledger()


def test_a_beat_that_is_not_newer_than_the_newest_taken_is_not_taken_again():
    """A snapshot that holds an older beat than the newest one taken, or the same one, adds nothing."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0))
    sw = m.Sweep(sch)
    sw.feed(SWEEP_MONO0, beat_state(stamps, rr, 0.0))
    taken = sw.ledger()["taken"]
    sw.feed(SWEEP_MONO0 + 0.5, beat_state(stamps, rr, 0.0))
    assert sw.ledger()["taken"] == taken
    sw.feed(SWEEP_MONO0 + 1.0, beat_state(stamps, rr, 0.0, window_s=5.0))
    assert sw.ledger()["taken"] == taken


@pytest.mark.parametrize("trouble", ["late link", "old first look"])
def test_the_swings_do_not_depend_on_how_late_the_beats_come_or_on_an_old_first_look(trouble):
    """A link that delays every stamp by up to seconds, and a first look at a snapshot that is seconds
    old, leave the breaths measured and the swings that they read where a clean run puts them."""
    _, _, _, base = _run()
    kw = {"lag_s": 4.0} if trouble == "late link" else {"first_old_s": 4.0}
    _, _, _, other = _run(**kw)
    assert len(_measured(other)) >= len(_measured(base)) - 1
    assert abs(np.median(_measured(other)) - np.median(_measured(base))) < 3.0
    assert 33.0 < np.median(_measured(other)) < 47.0


def test_jittered_and_bunched_looks_measure_the_same_swings():
    """Looks that come late by a third of a second, and some that bunch after a late one, give the same
    measured breaths and the same swings as evenly spaced looks."""
    _, _, _, base = _run()
    _, _, _, other = _run(step_s=0.1, jitter_s=0.3, burst_s=0.45, seed=1)
    assert len(_measured(other)) >= len(_measured(base)) - 1
    assert abs(np.median(_measured(other)) - np.median(_measured(base))) < 3.0
    assert other.reason == "done"


def test_a_step_of_the_stamps_that_the_clock_does_not_share_voids_one_breath_and_the_sweep_goes_on():
    """When the stamps move by seconds between two new snapshots and the clock does not, the breath in
    progress is voided with the cause `clock`, once, and the sweep keeps running."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    jump_at = sch.start_s(4) + 0.4 * sch.period_s(4)
    for at in np.arange(0.0, jump_at, 0.5):
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at)))
    assert sw.breath == 4 and sw.active
    sw.feed(SWEEP_MONO0 + jump_at + 0.5, beat_state(stamps, rr, jump_at + 0.5, lag_s=5.0))
    assert [o.reason for o in sw.outcomes if o.reason in m.CAUSES] == ["clock"]
    assert sw.active and sw.breath == 4 and sw.values[4] is None
    assert sw.clock == pytest.approx(sch.start_s(4))


@pytest.mark.parametrize("step_s", [5.0, -5.0], ids=["set forward", "set back"])
def test_a_wall_clock_that_is_set_by_five_seconds_in_either_direction_voids_one_breath(step_s):
    """A step of the stamps that the looks do not share, forward or back, is one hold with the cause
    `clock` and one outcome, and the sweep goes on."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    at_step = sch.start_s(4) + 0.4 * sch.period_s(4)
    for at in np.arange(0.0, sch.total_s + 20.0, 0.5):
        lag = step_s * (at >= at_step) if step_s > 0 else -step_s * (at < at_step)
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at), lag_s=lag))
        if not sw.active:
            break
    assert [o.reason for o in sw.outcomes if o.reason in m.CAUSES] == ["clock"]
    assert sw.reason == "done"


def test_a_stall_in_the_looks_that_meets_a_gap_in_the_beats_is_one_hold_and_never_a_clock_step():
    """The beats come three seconds apart and one look is three seconds late. The late look sees a beat
    that is nearly three seconds old, and a late look is no yardstick for the next beat: the hold is a
    stall, once, and the clock of the stamps is never blamed."""
    m = _mod()
    sch = _short()
    stamps = np.arange(-30.0, sch.total_s + 30.0, 3.2)
    rr = np.full(stamps.shape, 3200.0)
    late = float(stamps[stamps > sch.start_s(3)][0]) + 2.95
    sw = m.Sweep(sch)
    causes = []
    looks = np.concatenate([np.arange(0.0, late - 3.0, 0.5), [late], late + 0.5 * np.arange(1, 30)])
    for at in looks:
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at)))
        causes.append(sw.cause)
    assert {c for c in causes if c} == {"stall"}
    assert [o.reason for o in sw.outcomes if o.reason in m.CAUSES] == ["stall"]


def test_a_gap_between_two_looks_voids_one_breath_and_the_sweep_goes_on():
    """A gap of seconds between two looks voids the breath in progress with the cause `stall`, and the
    clock does not leap over the gap: the pacer starts that breath again."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    gap_at = sch.start_s(4) + 0.4 * sch.period_s(4)
    for at in np.arange(0.0, gap_at, 0.5):
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at)))
    sw.feed(SWEEP_MONO0 + gap_at + 20.0, beat_state(stamps, rr, gap_at + 20.0))
    assert [o.reason for o in sw.outcomes if o.reason in m.CAUSES] == ["stall"]
    assert sw.active and sw.breath == 4 and sw.values[4] is None
    assert sw.clock == pytest.approx(sch.start_s(4))


def test_a_look_that_is_one_and_a_half_seconds_late_voids_the_breath_and_a_prompt_one_does_not():
    """The pacer is kept to the looks, so a look that comes a second and a half after the one before
    it voids the breath in progress with the cause `stall`, and looks half a second apart never do."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    gap_at = sch.start_s(4) + 0.4 * sch.period_s(4)
    for at in np.arange(0.0, gap_at, 0.5):
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at)))
    assert [o.reason for o in sw.outcomes if o.reason in m.CAUSES] == []
    sw.feed(SWEEP_MONO0 + gap_at + 1.5, beat_state(stamps, rr, gap_at + 1.5))
    assert [o.reason for o in sw.outcomes if o.reason in m.CAUSES] == ["stall"]
    assert sw.values[4] is None and sw.clock == pytest.approx(sch.start_s(4))


def test_stamps_that_step_by_three_seconds_against_the_looks_void_a_breath_and_one_second_does_not():
    """A wall clock that is set by a few seconds is a hold with the cause `clock`: three seconds is
    one, and a step of one second, which a late link can give, is not."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    holds = {}
    for step_s in (1.0, 3.0):
        sw = m.Sweep(sch)
        at_step = sch.start_s(4) + 0.4 * sch.period_s(4)
        for at in np.arange(0.0, sch.total_s + 20.0, 0.5):
            sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at), lag_s=step_s * (at >= at_step)))
            if not sw.active:
                break
        holds[step_s] = [o.reason for o in sw.outcomes if o.reason in m.CAUSES]
    assert holds == {1.0: [], 3.0: ["clock"]}


def test_a_look_at_a_time_that_is_not_a_number_holds_one_look_and_the_sweep_goes_on():
    """A time that is not finite holds that look and nothing after it: the look that follows measures
    its step from the last good look and runs on, and the sweep ends by itself."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    bad_at = sch.start_s(4) + 0.4 * sch.period_s(4)
    for at in np.arange(0.0, sch.total_s + 40.0, 0.5):
        if abs(at - bad_at) < 0.25:
            sw.feed(math.nan, beat_state(stamps, rr, float(at)))
            assert sw.cause == "stall"
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at)))
        if abs(at - bad_at) < 0.25:
            assert sw.cause is None, "the look after a time that is not finite is not held"
        if not sw.active:
            break
    assert sw.reason == "done"
    assert len([o for o in sw.outcomes if o.reason in m.CAUSES]) == 1


def test_every_beat_is_counted_once_in_one_place():
    """Beats taken equal the beats before the first breath, in measured breaths, in breaths that gave no
    swing, in voided breaths, held, after the last and still waiting, for several delays and look spacings."""
    for lag, step in ((0.0, 0.5), (1.3, 0.5), (2.9, 1.0), (0.4, 0.25), (4.0, 0.7)):
        _, _, _, sw = _run(lag_s=lag, step_s=step)
        led = sw.ledger()
        assert led["taken"] == sum(v for k, v in led.items() if k != "taken")
        assert led["taken"] > 0 and led["before"] > 0 and led["measured"] > 0


def test_a_measured_breath_never_changes_when_beats_come_one_at_a_time():
    """A value that was shown once stays: over a run that gives one beat per look, no breath changes its
    value after it was measured, and no breath is measured twice."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    seen = {}
    for stamp in stamps[stamps > 0.0]:
        sw.feed(SWEEP_MONO0 + float(stamp), beat_state(stamps, rr, float(stamp)))
        for k, v in enumerate(sw.values):
            if v is not None:
                assert seen.setdefault(k, v) == v
        if not sw.active:
            break
    measured_k = [o.k for o in sw.outcomes if o.reason is None]
    assert len(measured_k) == len(set(measured_k)) and seen


def test_beats_that_come_during_a_hold_belong_to_no_later_breath():
    """While a cause holds the sweep, the beats are taken as held and stored for no breath. Beats of a
    different swing that come during a pause leave no mark on the breaths measured after it."""
    sch = _short()
    pause = (sch.start_s(4) + 0.4 * sch.period_s(4), 20.0)
    held = lambda t: "paused" if pause[0] <= t < pause[0] + pause[1] else None
    _, _, _, sw = _run(rests=(pause,), cause_at=held)
    assert sw.ledger()["held"] >= 10
    assert sw.reason == "done" and sw.values[4] is not None
    assert max(_measured(sw)) < 80.0


def test_a_hold_after_the_last_breath_ended_ends_the_sweep_with_that_breath_not_measured():
    """If the cause comes while the last breath has run its time and waits for a later beat, the sweep
    ends at once and that breath is not measured; the others keep their values."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    cut = stamps[stamps < sch.total_s - 0.3]
    cut_rr = rr[:len(cut)]
    t = 0.0
    while sw.clock < sch.total_s and t < sch.total_s + 3.0:
        sw.feed(SWEEP_MONO0 + t, beat_state(cut, cut_rr, t))
        t += 0.5
    assert sw.active and sw.values[-1] is None
    sw.feed(SWEEP_MONO0 + t, beat_state(cut, cut_rr, t), cause="paused")
    assert not sw.active and sw.values[-1] is None
    assert all(v is not None for v in sw.values[1:-1])


@pytest.mark.parametrize("rate_range", RANGES, ids=lambda r: f"{r[0]}-to-{r[1]}")
def test_the_edge_verdict_marks_a_peak_at_the_slow_end_and_at_the_fast_end_and_not_one_inside(rate_range):
    """Whichever way the sweep runs, a clear peak among the slowest breaths or among the fastest is at an
    edge, and one inside the range is not."""
    m = _mod()
    rates = list(m.SweepSchedule(*rate_range).rates)
    n = len(rates)
    order = np.argsort(rates)           # breath numbers from the slowest rate to the fastest

    def with_high(breaths):
        values = list(20.0 + np.random.default_rng(2).normal(0.0, 1.0, n))
        for k in breaths:
            values[int(k)] += 60.0
        return values

    slow, fast = m.verdict(rates, with_high(order[:9])), m.verdict(rates, with_high(order[-9:]))
    inside = m.verdict(rates, with_high(order[n // 2 - 5:n // 2 + 6]))
    assert slow.kind == fast.kind == inside.kind == "clear"
    assert slow.edge and fast.edge and not inside.edge


@pytest.mark.parametrize("rate_range", RANGES, ids=lambda r: f"{r[0]}-to-{r[1]}")
def test_a_whole_sweep_in_either_direction_finds_a_planted_peak_and_counts_every_beat_once(rate_range):
    """With a peak planted inside the range, a sweep that runs over all of it ends by itself, measures
    nearly every breath, finds the peak within a small error and puts every beat into one place."""
    m = _mod()
    schedule = m.SweepSchedule(*rate_range)
    plant = 0.55 * min(rate_range) + 0.45 * max(rate_range)
    stamps, rr, _ = pacer_beats(schedule, _peaked(plant), noise_ms=6.0, tail_s=60.0)
    sw = m.Sweep(schedule)
    run_sweep(sw, stamps, rr, until_s=schedule.total_s + 40.0)
    assert sw.reason == "done" and len(sw.values) == schedule.breaths
    assert len(_measured(sw)) >= schedule.breaths - 2
    assert abs(sw.result.best.rate - plant) < 0.5
    led = sw.ledger()
    assert led["taken"] == sum(v for k, v in led.items() if k != "taken")
    assert not sw.result.verdict.edge


@pytest.mark.parametrize("rate_range", RANGES, ids=lambda r: f"{r[0]}-to-{r[1]}")
def test_the_result_gives_the_mean_heart_rate_at_the_two_ends_of_a_series_whose_rate_falls(rate_range):
    """Over a sweep in which the heart rate falls from 80 to 55 bpm, the result holds the mean rate of the
    first measured breaths and of the last, each close to the planted rate where those breaths lie, and the
    first is the higher."""
    m = _mod()
    schedule = m.SweepSchedule(*rate_range)
    tail = 60.0
    stamps, rr, _ = pacer_beats(schedule, _flat(40.0), mean_ms=750.0, mean_end_ms=1090.0, noise_ms=4.0,
                                tail_s=tail)
    sw = m.Sweep(schedule)
    run_sweep(sw, stamps, rr, until_s=schedule.total_s + 40.0)
    first, last = sw.result.heart_bpm
    end_true = schedule.total_s + tail

    def planted(breaths):
        t = np.mean([schedule.start_s(k) + schedule.period_s(k) / 2 for k in breaths])
        return 60000.0 / (750.0 + 340.0 * t / end_true)

    assert first > last + 10.0
    assert first == pytest.approx(planted(range(m.WINDOW)), abs=3.0)
    assert last == pytest.approx(planted(range(schedule.breaths - m.WINDOW, schedule.breaths)), abs=3.0)


def test_the_two_ends_of_the_heart_rate_use_breaths_that_share_none_and_are_left_out_when_too_few_are_measured():
    """The first and the last `WINDOW` measured breaths are two sets with no breath in both, so a series
    with fewer measured breaths than two sets gives no figure and a breath that was not measured is not
    counted."""
    m = _mod()
    means = [900.0 + 10.0 * k for k in range(2 * m.WINDOW)]
    first, last = m.heart_rate_ends(means)
    assert first == pytest.approx(60000.0 / np.mean(means[:m.WINDOW]))
    assert last == pytest.approx(60000.0 / np.mean(means[-m.WINDOW:]))
    assert m.heart_rate_ends(means[:-1]) is None
    holes = [None] + means[:m.WINDOW] + [None, None] + means[m.WINDOW:] + [None]
    assert m.heart_rate_ends(holes) == m.heart_rate_ends(means)


def _retarget_run(start, end, other, *, at_fraction):
    """A sweep over the range `start` to `end`, a heart that follows the pacer for the range `other`, a
    look every half second, and the end moved to `other` at a fraction of the shorter total."""
    m = _mod()
    first = m.SweepSchedule(start, end)
    moved = first.with_end(other)
    stamps, rr, _ = pacer_beats(moved, _flat(40.0), noise_ms=4.0, tail_s=60.0)
    sw = m.Sweep(first)
    at = at_fraction * min(first.total_s, moved.total_s)
    run_sweep(sw, stamps, rr, until_s=at)
    return sw, first, moved, stamps, rr, at


@pytest.mark.parametrize("start,end,other", [(6.75, 4.25, 3.0), (6.75, 4.25, 5.5), (4.25, 6.75, 8.0),
                                              (4.25, 6.75, 5.5), (8.0, 3.0, 4.0)])
def test_moving_the_end_of_a_sweep_that_runs_keeps_every_measured_value_and_runs_on_to_the_new_end(start, end, other):
    """The breaths measured before the end moved keep their values, the sweep then runs over the breaths
    of the new schedule only, and it ends by itself after the last of them."""
    sw, first, moved, stamps, rr, at = _retarget_run(start, end, other, at_fraction=0.5)
    before = sw.values
    assert sw.active and any(v is not None for v in before)
    sw.retarget(moved)
    assert sw.schedule is moved and sw.active
    assert len(sw.values) == moved.breaths
    assert sw.values[:min(first.breaths, moved.breaths)] == before[:min(first.breaths, moved.breaths)]
    run_sweep(sw, stamps, rr, until_s=moved.total_s + 40.0, looks_from_s=at + 0.5)
    assert sw.reason == "done" and sw.result.measured + sw.result.unmeasured == moved.breaths
    assert len(_measured(sw)) >= moved.breaths - 2
    led = sw.ledger()
    assert led["taken"] == sum(v for k, v in led.items() if k != "taken")


@pytest.mark.parametrize("start,end", [(6.75, 4.25), (4.25, 6.75), (8.0, 3.0)])
def test_moving_the_end_behind_the_pacer_ends_the_sweep_at_once_with_what_it_has(start, end):
    """When the pacer is already past the new end, the sweep ends through its normal end: no more breath
    is awaited, the values that lie inside the new schedule stay, and the result is built from them."""
    m = _mod()
    first = m.SweepSchedule(start, end)
    stamps, rr, _ = pacer_beats(first, _flat(40.0), noise_ms=4.0, tail_s=60.0)
    sw = m.Sweep(first)
    run_sweep(sw, stamps, rr, until_s=0.7 * first.total_s)
    before = sw.values
    near = start + (end - start) * 0.15       # an end a short way from the start
    moved = first.with_end(near)
    assert moved.total_s < sw.clock
    sw.retarget(moved)
    assert sw.reason == "done" and not sw.active
    assert sw.values == before[:moved.breaths]
    assert sw.result.measured == sum(v is not None for v in before[:moved.breaths])
    assert sw.result.measured + sw.result.unmeasured == moved.breaths


def test_moving_the_end_behind_the_pacer_ends_a_sweep_that_still_waits_for_the_beat_that_closes_its_last_breath():
    """A breath that the pacer has passed but that no later beat has closed is not measured and not
    awaited: the sweep ends at once with what it has, as a hold at the last breath does."""
    m = _mod()
    sw = m.Sweep(m.SweepSchedule(6.0, 3.0, step_s=10.0))
    _empty_looks(sw, 24)
    assert sw.active
    sw.retarget(sw.schedule.with_end(5.0))
    assert sw.reason == "done" and sw.result.measured == 0 and sw.result.unmeasured == sw.schedule.breaths


def test_a_sweep_takes_only_a_copy_of_its_own_schedule_with_another_end():
    """The measured breaths belong to one start and one step: another start or step is refused."""
    m = _mod()
    sw = m.Sweep(m.SweepSchedule(6.75, 4.25))
    with pytest.raises(ValueError):
        sw.retarget(m.SweepSchedule(6.5, 4.25))
    with pytest.raises(ValueError):
        sw.retarget(m.SweepSchedule(6.75, 4.25, step_s=0.1))


def test_a_whole_sweep_ends_by_itself_and_later_beats_change_nothing():
    """A sweep over the whole schedule ends when its last breath is measured, with all breaths measured
    for a clean heart; beats that come after the end change no value, no outcome and no count."""
    m = _mod()
    sch = m.SweepSchedule()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0, tail_s=120.0)
    sw = m.Sweep(sch)
    run_sweep(sw, stamps, rr, until_s=sch.total_s + 40.0)
    assert not sw.active and sw.reason == "done" and sw.values[-1] is not None
    assert len(_measured(sw)) >= sch.breaths - 2
    before = (list(sw.values), list(sw.outcomes), sw.ledger(), m.best(sch.rates, sw.values))
    for at in np.arange(sch.total_s + 10.0, sch.total_s + 70.0, 0.6):
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at)))
    assert (list(sw.values), list(sw.outcomes), sw.ledger(), m.best(sch.rates, sw.values)) == before


@pytest.mark.parametrize("reason", ["stop", "plan", "source", "done"])
def test_ending_keeps_what_was_measured_and_the_first_reason(reason):
    """`end` keeps the breaths that are measured, drops the one in progress, and the first reason that
    it is given stays."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    run_sweep(sw, stamps, rr, until_s=sch.start_s(5) + 0.5 * sch.period_s(5))
    kept = list(sw.values)
    assert sw.active and kept[0] is not None and all(v is None for v in kept[5:])
    sw.end(reason)
    sw.end("stop" if reason != "stop" else "plan")
    assert not sw.active and sw.reason == reason and list(sw.values) == kept


def test_another_source_key_ends_the_sweep_at_the_next_feed_also_during_a_hold():
    """A feed with a key other than the one that the sweep began with ends it with the reason `source`,
    whether or not a cause holds it at that look."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    for cause in (None, "paused"):
        sw = m.Sweep(sch, source_key=("strap", 1))
        for at in np.arange(0.0, 20.0, 0.5):
            sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at)), source_key=("strap", 1))
        assert sw.active
        sw.feed(SWEEP_MONO0 + 20.5, beat_state(stamps, rr, 20.5), cause=cause, source_key=("strap", 2))
        assert not sw.active and sw.reason == "source"


def test_a_new_sweep_starts_empty_and_takes_nothing_from_the_one_before():
    """Two sweeps in one process share nothing: the second one, fed the same snapshot after the first has
    run, has no value, no outcome and a clock at the start."""
    m = _mod()
    sch = _short()
    _, stamps, rr, first = _run()
    second = m.Sweep(sch)
    second.feed(SWEEP_MONO0 + 900.0, beat_state(stamps, rr, sch.total_s + 5.0))
    assert _measured(first) and not _measured(second)
    assert second.outcomes == [] and second.clock == 0.0 and second.active
    assert second.ledger()["measured"] == 0


def _look(t, ages=(), rates=None):
    from types import SimpleNamespace

    return SimpleNamespace(t=t, age_s=list(ages), tachogram=[60.0] * len(ages) if rates is None else rates)


def _two_breaths():
    """Two breaths of 10 s and 20 s, looks every half second from the epoch 1024, and a beat that is placed
    by hand: all the numbers are exact in binary, so a beat can land on the edge of a breath."""
    m = _mod()
    return m.SweepSchedule(6.0, 3.0, step_s=10.0)


def _empty_looks(sw, count):
    for i in range(count):
        sw.feed(SWEEP_MONO0 + 0.5 * i, _look(1024.0 + 0.5 * i))


def test_a_beat_on_the_edge_of_a_breath_belongs_to_the_next_one_and_ends_this_one():
    """A beat that lands on the second where a breath ends is in the next breath, so this breath is
    final at that look and not a look later."""
    sw = _mod().Sweep(_two_breaths())
    _empty_looks(sw, 20)
    assert sw.outcomes == []
    sw.feed(SWEEP_MONO0 + 10.0, _look(1034.0, [0.0]))
    assert [o.k for o in sw.outcomes] == [0]
    assert sw.breath == 1


def test_a_beat_that_a_better_anchor_puts_into_a_breath_that_is_final_is_counted_as_before():
    """The anchor can move back by a part of a second between two looks, so a beat that is newer than the
    newest taken can fall into a breath that is already final. It is counted and not kept."""
    sw = _mod().Sweep(_two_breaths())
    _empty_looks(sw, 20)
    sw.feed(SWEEP_MONO0 + 10.0, _look(1034.0, [0.0]))
    was = sw.ledger()
    sw.feed(SWEEP_MONO0 + 10.5, _look(1034.8, [0.6]))
    led = sw.ledger()
    assert led["before"] == was["before"] + 1 and led["pending"] == was["pending"]
    assert led["taken"] == sum(v for k, v in led.items() if k != "taken")


def test_beats_in_the_first_snapshot_belong_to_no_breath():
    """What the first snapshot already holds came before the sweep began, whatever its stamps say."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    sw.feed(SWEEP_MONO0, beat_state(stamps, rr, 0.0))
    led = sw.ledger()
    assert led["taken"] > 0 and led["before"] == led["taken"] and led["pending"] == 0


def test_a_beat_that_is_older_than_one_taken_in_the_same_snapshot_is_not_taken():
    """Inside one snapshot, a beat whose stamp is not above that of the newest beat before it is not
    taken, so the stamps that a breath is fitted to always rise."""
    sw = _mod().Sweep(_two_breaths())
    sw.feed(SWEEP_MONO0, _look(1024.0))
    sw.feed(SWEEP_MONO0 + 0.5, _look(1030.0, [4.0, 1.0, 2.0]))
    assert sw.ledger()["taken"] == 2


def test_the_cost_of_a_look_does_not_grow_with_the_beats_that_the_snapshot_holds():
    """A look at a snapshot with nothing new reads the newest beat and no more, however many beats the
    snapshot holds, because the screen looks many times a second for a quarter of an hour."""
    class Counted(list):
        reads = 0

        def __getitem__(self, i):
            Counted.reads += 1
            return super().__getitem__(i)

    sw = _mod().Sweep(_two_breaths())
    ages = Counted(range(5000, 0, -1))
    state = _look(10000.0, ages, Counted([60.0] * len(ages)))
    sw.feed(SWEEP_MONO0, state)
    Counted.reads = 0
    sw.feed(SWEEP_MONO0 + 0.5, state)
    assert Counted.reads <= 4


def test_a_hold_voids_one_breath_for_each_hold_and_not_for_each_look():
    """Looks that keep the same cause are one hold and give one outcome, and a second hold later gives a
    second."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    sw = m.Sweep(sch)
    holds = ((sch.start_s(3) + 1.0, 6.0), (sch.start_s(6) + 1.0, 6.0))
    held = lambda t: "paused" if any(a <= t < a + b for a, b in holds) else None
    run_sweep(sw, stamps, rr, until_s=sch.total_s + 40.0, cause_at=held)
    assert [o.reason for o in sw.outcomes if o.reason in m.CAUSES] == ["paused", "paused"]


@pytest.mark.parametrize("trouble", ["old first look", "stamps step up", "stamps step down"])
def test_a_breath_is_final_when_the_pacer_has_run_its_time_and_not_much_before_or_after(trouble):
    """However the stamps and the looks line up, the breath that the sweep calls final has run its time on
    the clock to within a look, and a breath that ran its time some seconds ago is final. The anchor must
    follow a first look at an old snapshot, and it must be taken again after the stamps step either way."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0)
    step_at = sch.start_s(4) + 0.4 * sch.period_s(4)
    sw = m.Sweep(sch)
    for i, at in enumerate(np.arange(0.0, sch.total_s + 30.0, 0.5)):
        old = 4.0 if trouble == "old first look" and i == 0 else 0.0
        lag = {"stamps step up": 20.0 * (at >= step_at), "stamps step down": 20.0 * (at < step_at)}.get(trouble, 0.0)
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at) - old, lag_s=lag))
        assert all(sw.clock >= sch.start_s(k + 1) - 1.0 for k in range(sw.breath)), (at, sw.breath)
        assert all(k < sw.breath for k in range(sch.breaths) if sw.clock >= sch.start_s(k + 1) + 3.0), (at, sw.breath)
        if not sw.active:
            break
    assert sw.reason == "done"
    if trouble == "old first look":
        assert not [o for o in sw.outcomes if o.reason in m.CAUSES], "an old first snapshot holds nothing"


def test_a_clock_that_is_set_back_by_a_minute_does_not_stop_the_beats():
    """After the stamps step back by much more than a breath, the beats that come after the step are
    still taken, and the sweep measures the breaths that follow and ends by itself."""
    m = _mod()
    sch = _short()
    stamps, rr, _ = pacer_beats(sch, _flat(40.0), noise_ms=4.0, tail_s=100.0)
    step_at = sch.start_s(4) + 0.4 * sch.period_s(4)
    sw = m.Sweep(sch)
    for at in np.arange(0.0, sch.total_s + 60.0, 0.5):
        sw.feed(SWEEP_MONO0 + float(at), beat_state(stamps, rr, float(at), lag_s=60.0 * (at < step_at)))
        if not sw.active:
            break
    assert sw.reason == "done" and sw.values[-1] is not None


def test_a_cause_or_an_end_reason_that_is_not_named_is_refused():
    """A name that the module does not know is a fault of the caller, and it is refused at once."""
    m = _mod()
    sw = m.Sweep(_two_breaths())
    with pytest.raises(ValueError):
        sw.feed(SWEEP_MONO0, _look(1024.0), cause="tired")
    with pytest.raises(ValueError):
        sw.end("tired")
    assert sw.reason is None


# --- what the module is allowed to depend on ---------------------------------------------------

def test_the_sweep_module_loads_no_screen_and_no_device_module():
    """The module is processing: importing it must not bring in the console, the toolkit of the screen or
    a sound device."""
    import subprocess
    import sys

    code = ("import sys, nfb.hrv.sweep; "
            "print(sorted(n for n in sys.modules if n.split('.')[0] in ('nfb_console', 'flet', 'sounddevice')))")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "NFB_NO_OUTPUTS": "1"})
    assert out.stdout.strip() == "[]"


def test_a_best_window_beside_a_stretch_that_was_not_measured_is_an_edge_and_one_inside_a_run_is_not():
    """The true peak can lie in the stretch that was not measured, so a peak that ends at its rim is at an
    edge as much as one at the end of the sweep, and a peak in the middle of a measured run is not."""
    m = _mod()
    rates = m.SweepSchedule().rates
    n = len(rates)

    def values(high):
        series = list(20.0 + np.random.default_rng(2).normal(0.0, 1.0, n))
        for k in high:
            series[k] += 60.0
        return [None if 30 <= k < 48 else v for k, v in enumerate(series)]

    assert m.verdict(rates, values(range(20, 30))).edge
    assert m.verdict(rates, values(range(60, 70))).edge is False
    assert m.verdict(rates, values(range(8, 17))).edge is False


def test_a_series_with_another_length_than_the_schedule_is_refused():
    """One swing for each rate: a series of another length is a fault of the caller."""
    m = _mod()
    with pytest.raises(ValueError):
        m.best([6.0, 5.5], [1.0])
    with pytest.raises(ValueError):
        m.curve([6.0], [1.0, 2.0])


# --- the second mark: where the heart rate and the breath are in step -----------------------------

def _line(rates, at, slope=0.5):
    """In-step values that fall by `slope` seconds for each breath a minute below `at`."""
    return [slope * (r - at) for r in rates]


@pytest.mark.parametrize("share", [0.3, 0.7])
def test_both_marks_of_a_sweep_land_near_the_rate_of_a_heart_with_a_resonance(share):
    """On a heart that swings most and is in step at its own rate, over a few seeds, the rate of the largest
    paced swing and the rate in step both land near it, whatever the share of the inhale. The in-step rate
    is the closer of the two."""
    from resonant_heart import sweep_heart

    best_err, step_err = [], []
    for resonance in (4.75, 5.5, 6.0):
        for seed in range(3):
            result = sweep_heart(resonance, seed, share=share).result
            best_err.append(abs(result.best.rate - resonance))
            assert result.in_step.rate is not None, (resonance, seed, result.in_step)
            step_err.append(abs(result.in_step.rate - resonance))
    assert max(best_err) < 0.8 and max(step_err) < 0.4
    assert np.mean(step_err) < np.mean(best_err)


def test_a_heart_that_does_not_respond_to_the_breath_gives_no_in_step_rate_and_names_no_side():
    """With no response there is no phase to read, and the few breaths that noise lets through do not
    make a claim about the whole range: no in-step rate and no side."""
    from resonant_heart import sweep_heart

    for seed in range(5):
        result = sweep_heart(5.5, seed, swing_bpm=0.0).result
        assert result.in_step.rate is None and result.in_step.side is None, seed


def test_the_rate_in_step_is_where_the_windowed_value_changes_sign_and_only_there():
    """A series that is late at the fast end and early at the slow end changes sign once, at the rate where
    it crosses zero. The line between two windows places it, so it is between the measured rates."""
    m = _mod()
    rates = _mod().SweepSchedule().rates
    for at in (4.8, 5.5, 6.2):
        got = m.in_step(rates, _line(rates, at))
        assert got.rate == pytest.approx(at, abs=0.05) and got.side is None


@pytest.mark.parametrize("sign, side", [(1.0, "late"), (-1.0, "early")])
def test_a_series_that_never_changes_sign_has_no_rate_and_names_the_side_of_the_range_that_it_lies_on(sign, side):
    """Late at every measured rate means that the heart rate is in step below the measured range, and early
    at every one means above it; neither is given as a rate."""
    rates = _mod().SweepSchedule().rates
    got = _mod().in_step(rates, [sign * 0.3 + 0.01 * (r - 5.5) for r in rates])
    assert got.rate is None and got.side == side


def test_a_series_that_is_early_at_the_fast_end_and_late_at_the_slow_end_has_no_rate():
    """The heart rate is late above its rate and early below it. A change of sign the other way round
    is not that, so it is not given as a rate."""
    rates = _mod().SweepSchedule().rates
    got = _mod().in_step(rates, _line(rates, 5.5, slope=-0.5))
    assert got.rate is None


def test_the_in_step_rate_is_not_found_across_a_stretch_of_breaths_that_were_not_measured():
    """The sign changes inside a stretch with no measured breath, which is wider than a window. The
    two sides are not joined, so there is no rate: the best rate treats a hole the same way."""
    rates = _mod().SweepSchedule().rates
    values = _line(rates, 5.5)
    centre = min(range(len(rates)), key=lambda k: abs(rates[k] - 5.5))
    for k in range(centre - 8, centre + 9):
        values[k] = None
    assert _mod().in_step(rates, values).rate is None
    values = _line(rates, 5.5)
    for k in range(centre - 1, centre + 2):
        values[k] = None            # a stretch that leaves windows with enough breaths to be supported
    assert _mod().in_step(rates, values).rate == pytest.approx(5.5, abs=0.15)


def test_where_several_windows_change_sign_the_one_nearest_the_best_rate_counts():
    """A series that crosses zero twice, rising with the rate both times, gives the crossing that is
    nearer to the rate that is asked for, and the first when none is asked for."""
    rates = list(_mod().SweepSchedule().rates)
    values = [0.5 * (r - 6.2) if r > 5.0 else 0.5 * (r - 4.6) for r in rates]
    assert any(v is not None for v in values)
    near_low = _mod().in_step(rates, values, near=4.6).rate
    near_high = _mod().in_step(rates, values, near=6.2).rate
    assert near_low == pytest.approx(4.6, abs=0.2) and near_high == pytest.approx(6.2, abs=0.2)


def test_the_result_of_a_sweep_keeps_the_in_step_value_of_every_breath_that_gave_one():
    """The sweep keeps one value for each breath, in seconds, and the result is made from them with the
    best rate nearest, so the two marks of a result come from the same breaths."""
    from resonant_heart import sweep_heart

    sweep = sweep_heart(5.5, 1)
    steps = sweep.steps
    assert len(steps) == sweep.schedule.breaths and sum(v is not None for v in steps) > 60
    assert all(abs(v) < sweep.schedule.period_s(0) / 2 for v in steps if v is not None)
    assert sweep.result.in_step == _mod().in_step(sweep.schedule.rates, steps, sweep.result.best.rate)


def test_a_breath_in_which_the_inhale_share_changes_keeps_its_swing_and_gives_no_in_step_value():
    """The reference of the phase is that of one share, so a breath that ran with two has no phase to
    compare. The other breaths keep theirs."""
    m = _mod()
    schedule = schedule_of(10)
    stamps, rr, _ = pacer_beats(schedule, _flat(40.0))
    sweep = m.Sweep(schedule)
    changed = 4
    for t in np.arange(0.0, schedule.total_s + 5.0, 0.5):
        share = 0.3 if schedule.start_s(changed) + 1.0 < t < schedule.start_s(changed) + 2.0 else 0.5
        sweep.feed(SWEEP_MONO0 + float(t), beat_state(stamps, rr, float(t)), None, None, share)
        if not sweep.active:
            break
    assert sweep.values[changed] is not None and sweep.steps[changed] is None
    assert all(v is not None for k, v in enumerate(sweep.steps) if k != changed and sweep.values[k] is not None)
