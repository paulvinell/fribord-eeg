"""The chance spread of one trial, from pairs of readings, against wandering levels whose answer is known.

A trial's result is the mean of its feedback window minus the mean of its baseline. On a level that
wanders, with no task at all, that difference has a spread, and the spread is what a result must beat
before it says anything. The estimator measures it from pairs of valid readings at the trial's own lags,
so it needs no task, no trial and no settling time, and it follows the person's own signal.

Every signal here is `conftest.wandering`: power in f^-β with β across 1 to 2, drawn at three scales
fifteen times apart. White noise is never used, because its trial spread says nothing about a level.
"""

from __future__ import annotations

import math
import time

import numpy as np

from nfb_console.trials import ACTIVE_S, BASELINE_S, ITI_RANGE_S, LEADIN_S, RESULT_S

from conftest import BETAS, STEPS_UV, wandering

HZ = 20.0
"""Readings a second, as the slow level gives them."""


def _spread():
    from nfb.dsp.chance import PairSpread

    return PairSpread(baseline_s=BASELINE_S, active_s=ACTIVE_S)


def _feed(spread, x, start: float = 0.0, valid=None) -> None:
    for i, v in enumerate(x):
        spread.feed(start + i / HZ, float(v), True if valid is None else bool(valid[i]))


def _trial_rms(x) -> float:
    """The root mean square of the trial result at every position of `x`, a quarter second apart."""
    nb, na = int(BASELINE_S * HZ), int(ACTIVE_S * HZ)
    c = np.r_[0.0, np.cumsum(x)]
    starts = np.arange(0, len(x) - nb - na, int(HZ / 4))
    base = (c[starts + nb] - c[starts]) / nb
    active = (c[starts + nb + na] - c[starts + nb]) / na
    return float(np.sqrt(np.mean((active - base) ** 2)))


def _case(seed: int) -> tuple[float, float]:
    return BETAS[seed % len(BETAS)], STEPS_UV[seed % len(STEPS_UV)]


def test_a_spread_exists_at_the_first_feedback_start():
    """The lead-in and one baseline are 4 s of signal, less than the 8 s of one trial. The spread still has
    a value there, from the lags seen so far and a random walk for the rest, in every one of 300 starts."""
    lead = int((LEADIN_S + BASELINE_S) * HZ)
    missing = []
    for seed in range(300):
        beta, step = _case(seed)
        spread = _spread()
        _feed(spread, wandering(beta, 60.0, HZ, seed, step)[:lead])
        value = spread.value
        if value is None or not math.isfinite(value) or value <= 0:
            missing.append((seed, beta, value))
    assert not missing, f"no spread at the first feedback start in {len(missing)} of 300: {missing[:5]}"


def test_the_spread_matches_the_trial_arithmetic_on_wandering_signals():
    """After two minutes the spread is the spread of the trial result itself, measured at every position
    of the same two minutes, at each slope and each scale."""
    ratios = []
    for seed in range(8):
        beta, step = _case(seed)
        x = wandering(beta, 150.0, HZ, 100 + seed, step)
        spread = _spread()
        _feed(spread, x)
        ratios.append(spread.value / _trial_rms(x[-int(120 * HZ):]))
    assert all(0.7 < r < 1.4 for r in ratios), np.round(ratios, 2)
    assert 0.85 < float(np.median(ratios)) < 1.18, np.round(ratios, 2)


def test_a_doubled_noise_is_followed_within_the_memory():
    """The memory is 120 s of signal. When the level starts to move twice as much, the spread has most of
    the new size 120 s later."""
    ratios = []
    for seed in range(4):
        beta, step = _case(2 * seed)
        a = wandering(beta, 300.0, HZ, 200 + seed, step)
        at = int(180 * HZ)
        x = a.copy()
        x[at:] = a[at:] + (a[at:] - a[at])                      # each change doubled, with no step
        spread = _spread()
        _feed(spread, x[:at])
        before = spread.value
        _feed(spread, x[at:], start=at / HZ)
        ratios.append(spread.value / (2 * before))
    assert float(np.median(ratios)) > 0.75 and min(ratios) > 0.5, np.round(ratios, 2)


def test_a_step_across_a_gap_does_not_enter_the_spread():
    """The level restarts after a gap of more than a second, and an electrode can come back 300 µV away.
    No pair spans the gap, so the step is not a change of the level."""
    x = wandering(1.5, 150.0, HZ, 7, 8.0)
    at = int(75 * HZ)
    values = []
    for offset in (0.0, 300.0):
        spread = _spread()
        _feed(spread, x[:at])
        _feed(spread, x[at:] + offset, start=at / HZ + 2.0)
        values.append(spread.value)
    assert 0.85 < values[1] / values[0] < 1.18, values


def test_a_step_back_in_time_starts_a_new_stretch():
    """A block fed twice or a new source steps back in time. No pair joins the readings on either side."""
    x = wandering(1.5, 150.0, HZ, 8, 8.0)
    at = int(90 * HZ)
    values = []
    for offset in (0.0, 300.0):
        spread = _spread()
        _feed(spread, x[:at])
        _feed(spread, x[at:] + offset, start=at / HZ - 1.0)
        values.append(spread.value)
    assert 0.85 < values[1] / values[0] < 1.18, values


def test_a_non_finite_reading_is_invalid():
    """A NaN or an infinity marked valid counts as an invalid reading, and leaves no trace in the spread."""
    x = wandering(1.25, 100.0, HZ, 9, 8.0)
    bad = np.zeros(len(x), dtype=bool)
    bad[::37] = True
    marked, poisoned = _spread(), _spread()
    _feed(marked, x, valid=~bad)
    y = x.copy()
    y[bad] = [math.nan if i % 2 else math.inf for i in range(int(bad.sum()))]
    _feed(poisoned, y)
    assert poisoned.value is not None and math.isfinite(poisoned.value)
    assert poisoned.value == marked.value


def _schedule(seed: int, trials: int) -> list[tuple[float, float, float, int]]:
    """(baseline start, feedback start, feedback end, direction) for each trial, at the program's timing."""
    rng = np.random.default_rng(10_000 + seed)
    t, out = LEADIN_S, []
    for _ in range(trials):
        out.append((t, t + BASELINE_S, t + BASELINE_S + ACTIVE_S, 1 if rng.random() < 0.5 else -1))
        t += BASELINE_S + ACTIVE_S + RESULT_S + rng.uniform(*ITI_RANGE_S)
    return out


def test_a_noise_burst_raises_chance_success_while_it_lasts_and_not_much_after():
    """**succeeds at** is one chance spread, frozen at each feedback start. A level that moves three times
    as much for 30 s outruns the spread, and then holds it high: both effects stay inside a band."""
    during, after = [], []
    for seed in range(20):
        beta, step = _case(seed)
        plan = _schedule(seed, 19)
        x = wandering(beta, plan[-1][2] + 2.0, HZ, 300 + seed, step)
        b0, b1 = int(plan[8][0] * HZ), int((plan[8][0] + 30.0) * HZ)
        change = np.diff(x, prepend=x[0])
        change[:b0], change[b1:] = 0.0, 0.0
        y = x + 2.0 * np.cumsum(change)                         # each change three times, with no step
        spread, k, seen_after = _spread(), 0, 0
        for base, start, end, direction in plan:
            i0, i1, i2 = int(base * HZ), int(start * HZ), int(end * HZ)
            _feed(spread, y[k:i1], start=k / HZ)
            limit = spread.value
            _feed(spread, y[i1:i2], start=i1 / HZ)
            k = i2
            score = y[i1:i2].mean() - y[i0:i1].mean()
            success = limit is not None and direction * score >= limit
            if b0 <= i1 < b1:
                during.append(success)
            elif i1 >= b1 and seen_after < 9:
                after.append(success)
                seen_after += 1
    assert 0.15 < np.mean(during) < 0.55, f"during the burst: {np.mean(during):.2f} of {len(during)}"
    assert 0.015 < np.mean(after) < 0.14, f"in the 9 trials after it: {np.mean(after):.3f} of {len(after)}"


def test_thirty_minutes_of_readings_cost_little():
    """The feed thread pays for every reading, and the screen reads the value once a frame. Half an hour of
    both is well under the time the half hour takes, by three orders of magnitude."""
    x = wandering(1.5, 1800.0, HZ, 11, 8.0)
    spread = _spread()
    began = time.perf_counter()
    for i, v in enumerate(x):
        spread.feed(i / HZ, float(v), True)
        if i % int(HZ) == 0:
            _ = spread.value, spread.seconds
    cost = time.perf_counter() - began
    assert spread.value is not None and 0 < spread.seconds <= 120.0 + 1.0
    assert cost < 1.0, f"30 min of readings took {cost:.2f} s"


def test_a_missing_reading_is_a_hole_and_not_a_step():
    """A reading that never came leaves its place on the grid empty, exactly as an invalid one does, so
    the pairs after it keep their true lags."""
    x = wandering(1.5, 100.0, HZ, 12, 8.0)
    gone = np.zeros(len(x), dtype=bool)
    gone[5::23] = True
    marked, missing = _spread(), _spread()
    _feed(marked, x, valid=~gone)
    for i, v in enumerate(x):
        if not gone[i]:
            missing.feed(i / HZ, float(v), True)
    assert missing.value == marked.value


def test_the_first_second_of_a_stretch_is_left_out():
    """A new stretch starts with the step response of the level's lowpass. Its first second does not
    enter the spread."""
    x = wandering(1.5, 150.0, HZ, 13, 8.0)
    at = int(75 * HZ)
    values = []
    for ring in (0.0, 300.0):
        y = x.copy()
        y[at:at + int(0.9 * HZ)] += ring
        spread = _spread()
        _feed(spread, y[:at])
        _feed(spread, y[at:], start=at / HZ + 2.0)
        values.append(spread.value)
    assert values[0] == values[1]
