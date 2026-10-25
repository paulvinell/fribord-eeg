"""Turning a metric into a 0…1 feedback signal — the second of three layers, and the one we did not have.

**The split here is between *what* to measure and *how to make it
comparable*, and the reason it is worth having as a layer is that the same metric wants different treatment
depending on what you know about it. A fraction that is already 0…1 needs a straight map. Milliseconds of
RMSSD need to be judged against this person on this day. A quantity with no upper end needs a knee rather
than a range. None of those is a special case of the others.

**A scaling says nothing about which end is good.** That belongs to the metric — `MetricSpec.invert` — and is
applied to the normalised value afterwards, so a rolling minimum stays a minimum and the pane's description of
what its window holds stays true. This file had one strategy that broke the rule: `1/(1 + value/scale)`,
offered under the label "small is good", which is a *direction* wearing a scaling's clothes. Two ways to say
the same thing is bad enough; combined with `invert` it also let a metric be inverted twice. It is
`Saturating` now, increasing like every other one.

**How this differs from what the EEG screens use, and why both exist.** `Target.score` is a logistic through
0.5 at a *threshold*, and the threshold is derived from the trainee's own first thirty seconds. That is the
right shape when the question is "are you above your own baseline" — which is what a band amplitude is
trained against. It is the wrong shape when the metric is already a bounded fraction (a logistic through a
calibrated midpoint throws away the fact that 1.0 means something absolute), and it has no way to express
"low is good" without inverting the target's direction, which then inverts the threshold derivation too.

So the EEG path keeps `Target`, and the HRV path gets these. They are not competing implementations of one
idea; they answer different questions about different kinds of number.

Every strategy is stateful only in the rolling ones, and `warmup` is public because a normaliser that has not
seen enough history yet is reporting 0.5 — the exact middle — and a screen that showed that as a real reading
would be showing "average" for a signal it knows nothing about.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Strategy:
    """A metric in, a 0…1 signal out. Stateful in the rolling ones."""

    def push(self, value: float, t: float) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    @property
    def warmup(self) -> float:
        """0…1, how much of the history this needs it has. 1.0 for the stateless ones."""
        return 1.0

    def describe(self) -> str:
        return ""


class ZScore(Strategy):
    """Standard deviations from a rolling mean, clamped to ±k and mapped linearly.

    The universal one: it fits any metric because it asks nothing about the units, only about the spread. Its
    cost is the one every rolling normaliser has — it makes *your own recent range* the scale, so improving
    steadily reads as average once the window catches up. That is right for a session and wrong across
    sessions, and it is why `fixed_range` exists.
    """

    def __init__(self, window_s: float = 60.0, clamp_k: float = 2.5):
        if window_s <= 0 or clamp_k <= 0:
            raise ValueError("window_s and clamp_k must both be positive")
        self.window_s = float(window_s)
        self.clamp_k = float(clamp_k)
        self._buf: deque[tuple[float, float]] = deque()

    def push(self, value: float, t: float) -> float:
        self._buf.append((t, value))
        while self._buf and self._buf[0][0] < t - self.window_s:
            self._buf.popleft()
        if len(self._buf) < 2:
            return 0.5
        values = np.fromiter((v for _, v in self._buf), dtype=float, count=len(self._buf))
        sd = float(values.std())
        if sd < 1e-9:
            return 0.5
        z = (value - float(values.mean())) / sd
        return clamp01((z + self.clamp_k) / (2.0 * self.clamp_k))

    def reset(self) -> None:
        self._buf.clear()

    @property
    def warmup(self) -> float:
        if not self._buf:
            return 0.0
        return min(1.0, (self._buf[-1][0] - self._buf[0][0]) / self.window_s)

    def describe(self) -> str:
        return f"±{self.clamp_k:g} sd of the last {self.window_s:g} s"


class FixedRange(Strategy):
    """A straight map from a range you name. No history, no calibration, no drift.

    The one to use when the metric's range is already meaningful — a fraction over [0, 1] — because it is
    then the only strategy under which the number means the same thing today as last week. Everything else
    here re-scales to recent history by design.
    """

    def __init__(self, low: float = 0.0, high: float = 1.0):
        if high <= low:
            raise ValueError("high must exceed low")
        self.low = float(low)
        self.high = float(high)

    def push(self, value: float, t: float) -> float:
        return clamp01((value - self.low) / (self.high - self.low))

    def describe(self) -> str:
        return f"{self.low:g} to {self.high:g}, fixed"


class RollingMinMax(Strategy):
    """A straight map from the extremes actually hit recently.

    Self-calibrating like `zscore` and more aggressive: the ends track real extremes rather than assuming a
    Gaussian, so it uses the whole output range whatever the distribution's shape. It is the right first
    choice for a metric whose units mean nothing to you yet — and it is twitchy, because one outlier
    redefines an end until it ages out of the window.
    """

    def __init__(self, window_s: float = 120.0):
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = float(window_s)
        self._buf: deque[tuple[float, float]] = deque()

    def push(self, value: float, t: float) -> float:
        self._buf.append((t, value))
        while self._buf and self._buf[0][0] < t - self.window_s:
            self._buf.popleft()
        if len(self._buf) < 2:
            return 0.5
        values = [v for _, v in self._buf]
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return 0.5
        return clamp01((value - lo) / (hi - lo))

    def reset(self) -> None:
        self._buf.clear()

    @property
    def warmup(self) -> float:
        if not self._buf:
            return 0.0
        return min(1.0, (self._buf[-1][0] - self._buf[0][0]) / self.window_s)

    def describe(self) -> str:
        return f"the range hit in the last {self.window_s:g} s"


class Saturating(Strategy):
    """`value / (value + scale)` — maps [0, ∞) onto [0, 1), rising and never arriving.

    For a positive metric with no upper end, where a fixed range would need a top you do not know and a
    rolling one would keep moving under you. `scale` is where it crosses 0.5, which makes it the one knob and
    a meaningful one: the value you consider middling, in the metric's own units.

    It **rises**, like every strategy here. Its predecessor was the same curve upside down, offered as "small
    is good" — see the module docstring for why that is not a scaling.
    """

    def __init__(self, scale: float = 1.0):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)

    def push(self, value: float, t: float) -> float:
        v = max(0.0, value)
        return clamp01(v / (v + self.scale))

    def describe(self) -> str:
        return f"0.5 at {self.scale:g}, no top"


class Tanh(Strategy):
    """`(tanh(value/scale) + 1) / 2` — rising, zero at the middle of the output, never clipping.

    For a signed metric, where zero is the middle of the *scale* rather than a good or a bad place to be. It
    saturates gracefully instead of hitting a wall, which matters for anything that occasionally spikes: a
    clamped strategy pins at the end and stops saying anything, where this keeps its ordering all the way out.
    """

    def __init__(self, scale: float = 1.0):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)

    def push(self, value: float, t: float) -> float:
        return clamp01((math.tanh(value / self.scale) + 1.0) / 2.0)

    def describe(self) -> str:
        return f"0.88 at {self.scale:g}, soft ends"


@dataclass(frozen=True)
class StrategySpec:
    id: str
    label: str
    description: str
    build: Callable[..., Strategy]
    detail: str = ""
    defaults: dict = field(default_factory=dict)


STRATEGIES: dict[str, StrategySpec] = {
    "rolling_minmax": StrategySpec(
        id="rolling_minmax", label="recent range",
        description="the highest and lowest you have actually reached lately become the ends",
        detail=("Uses the whole output range whatever the metric's units or distribution, which makes it "
                "the one to reach for when you do not yet know what good looks like for this person.\n\n"
                "Twitchy by construction: a single outlier defines an end until it ages out of the window, "
                "so the same reading can mean different things a minute apart. And like every rolling "
                "strategy it re-scales to recent history, so steady improvement reads as average."),
        build=RollingMinMax, defaults={"window_s": 120.0},
    ),
    "zscore": StrategySpec(
        id="zscore", label="standard deviations",
        description="how far from your own recent average, in standard deviations",
        detail=("Fits any metric, because it asks about spread rather than units. Steadier than the recent "
                "range since one outlier moves a mean far less than it moves a maximum.\n\n"
                "Assumes the spread is roughly symmetric, which several of these metrics are not — pNN50 "
                "sits against zero at low variability, and a z-score there reports 0.5 for a signal that "
                "is doing nothing."),
        build=ZScore, defaults={"window_s": 60.0, "clamp_k": 2.5},
    ),
    "fixed_range": StrategySpec(
        id="fixed_range", label="a fixed range",
        description="a range you name, and it never moves",
        detail=("The only strategy under which today's reading is comparable with last week's — everything "
                "else here re-scales to recent history on purpose. Use it for a metric that is already a "
                "meaningful fraction, where 1.0 means something absolute.\n\n"
                "It needs the range to be right. Set it too wide and the signal barely moves; too narrow "
                "and it sits pinned at an end saying nothing."),
        build=FixedRange, defaults={"low": 0.0, "high": 1.0},
    ),
    "saturating": StrategySpec(
        id="saturating", label="no upper end",
        description="half way up at a value you name, and it never quite reaches the top",
        detail=("For a positive quantity with no ceiling, where a fixed range would need a top you do not "
                "know and a rolling one would keep moving under you. The knob is where the output crosses "
                "the middle, set in the metric's own units: the value you would call middling.\n\n"
                "Stateless, so it does not drift — but it also does not adapt, which means a badly chosen "
                "scale leaves the signal parked near an end.\n\n"
                "Which end is good is not this setting's business; that comes from the metric and is shown "
                "on the row above."),
        build=Saturating, defaults={"scale": 1.0},
    ),
    "tanh": StrategySpec(
        id="tanh", label="soft ends",
        description="a signed metric, with zero in the middle of the scale",
        detail=("Never clips: extremes saturate gracefully and keep their ordering, which matters for a "
                "metric that spikes. A clamped strategy pins at the end and stops distinguishing.\n\n"
                "Only sensible where the metric goes negative — on one that is always positive it uses half "
                "its range and nothing else. Zero landing in the middle is a statement about the scale, not "
                "about zero being a good place to be."),
        build=Tanh, defaults={"scale": 1.0},
    ),
}

DEFAULT_STRATEGY = "rolling_minmax"
"""Because it is the one that needs nothing known in advance. Their default is a fixed range over [0, 1],
which is right for their default metric and wrong for most of the others — RMSSD in milliseconds against a
[0, 1] range is pinned at 1.0 for ever."""


def build(strategy_id: str, **overrides) -> Strategy:
    """A strategy from its id, with the catalogue's defaults filled in.

    Unknown ids raise rather than falling back, because a silent fallback would normalise the signal by a
    rule nobody chose and everything downstream would look plausible.
    """
    if strategy_id not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy_id!r}; expected one of {sorted(STRATEGIES)}")
    spec = STRATEGIES[strategy_id]
    return spec.build(**{**spec.defaults, **overrides})
