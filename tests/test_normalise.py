"""The scalings that map a metric to the 0 to 1 level: each one rises, stays in range and says what it is."""

from __future__ import annotations

import numpy as np
import pytest

from nfb_console import normalise


def test_every_strategy_maps_into_the_unit_interval_and_is_documented():
    for strategy_id, spec in normalise.STRATEGIES.items():
        assert spec.label and spec.description and spec.detail
        strategy = normalise.build(strategy_id)
        for t, value in enumerate([-1e6, -1.0, 0.0, 0.5, 1.0, 1e6]):
            out = strategy.push(value, float(t))
            assert 0.0 <= out <= 1.0, f"{strategy_id} produced {out}"
        assert strategy.describe()


def test_an_unknown_strategy_raises_rather_than_falling_back():
    """A silent fallback would normalise by a rule nobody chose, and everything downstream would look fine."""
    with pytest.raises(ValueError):
        normalise.build("vibes")


def test_a_rolling_strategy_reports_the_middle_and_says_it_is_not_ready():
    """It has to return *something*, and 0.5 is the least wrong something — but a screen that showed that as
    a real reading would be showing 'average' for a signal it knows nothing about. Hence `warmup`."""
    strategy = normalise.build("rolling_minmax", window_s=60.0)
    assert strategy.warmup == 0.0
    assert strategy.push(5.0, 0.0) == 0.5, "one value cannot have a range"
    assert strategy.warmup < 1.0
    for i in range(1, 61):
        strategy.push(float(i), float(i))
    assert strategy.warmup == 1.0
    assert strategy.push(1000.0, 61.0) == 1.0, "a new maximum is the top of the range"

    # the stateless ones are ready immediately, which is the whole reason to prefer them
    assert normalise.build("fixed_range").warmup == 1.0
    assert normalise.build("tanh").warmup == 1.0


def test_a_scaling_only_scales_and_the_direction_belongs_to_the_metric():
    """**Every strategy rises.** Which end is good is the metric's property — `MetricSpec.invert` — applied
    after normalisation, and a scaling that also flipped the sign would be a second way to say the same thing.
    One of them was: `1/(1 + value/scale)`, offered in the "scaled by" row under the label "small is good",
    where it read as a claim about the metric rather than about the scale. It also let a metric that already
    inverts be inverted twice, which lands back where it started with nothing on screen saying so.
    """
    for sid in normalise.STRATEGIES:
        strategy = normalise.build(sid)
        seen = [strategy.push(v, float(i)) for i, v in enumerate(np.linspace(0.0, 4.0, 40))]
        assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:])), f"{sid} falls as the metric rises"

    # and no label or description claims a direction, since the pill sits under "scaled by"
    for spec in normalise.STRATEGIES.values():
        words = f"{spec.label} {spec.description}".lower()
        for claim in ("small is good", "large is good", "high is better", "zero is the goal"):
            assert claim not in words, f"{spec.id} describes a direction, not a scale"


def test_the_strategies_are_not_interchangeable():
    """Each one exists because it answers a different question about the number's shape."""
    fixed = normalise.build("fixed_range", low=0.0, high=1.0)
    assert fixed.push(0.25, 0.0) == pytest.approx(0.25), "a straight map, comparable across sessions"
    assert fixed.push(50.0, 1.0) == 1.0, "and pinned for a metric outside its range"

    open_ended = normalise.build("saturating", scale=2.0)
    assert open_ended.push(0.0, 0.0) == pytest.approx(0.0)
    assert open_ended.push(2.0, 1.0) == pytest.approx(0.5), "the knob is where it crosses the middle"
    assert open_ended.push(100.0, 2.0) > 0.95, "and it approaches the top without reaching it"
    assert open_ended.push(1e12, 3.0) < 1.0

    centred = normalise.build("tanh", scale=1.0)
    assert centred.push(0.0, 0.0) == pytest.approx(0.5), "zero is neutral"
    assert centred.push(1.0, 1.0) == pytest.approx(0.88, abs=0.01)
    assert centred.push(-1.0, 2.0) == pytest.approx(0.12, abs=0.01)
    # It saturates gracefully rather than hitting a wall: a clamped strategy gives the same answer for
    # everything past its limit, where this keeps its ordering over any range a real metric reaches.
    # (Far enough out, float itself saturates — tanh(1e9) is exactly 1.0. That is arithmetic, not the shape.)
    assert centred.push(5.0, 3.0) < centred.push(8.0, 4.0) < 1.0
    assert 0.0 < centred.push(-8.0, 5.0) < centred.push(-5.0, 6.0)
    clamped = normalise.build("zscore", window_s=60.0, clamp_k=2.0)
    for i in range(60):
        clamped.push(float(i % 3), float(i))            # a small, stable spread
    assert clamped.push(1e3, 61.0) == clamped.push(1e6, 62.0) == 1.0, "a clamp stops distinguishing"
