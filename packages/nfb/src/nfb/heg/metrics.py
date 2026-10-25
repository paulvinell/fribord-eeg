"""What to feed back on the HEG channel — a catalogue, for the reason the HRV one is one.

Six quantities, expressed on the `MetricSpec` this repo already has. Two design points:

- **Inversion is `invert=True`, not a strategy.** A `softplus_inverse` strategy would flip
  "low variability is good" into a rising signal inside the scaling — precisely the conflation the
  HRV screen's direction row was built to undo. Here every strategy rises, the metric names the
  quantity, and the direction row owns the aim.
- **Every metric carries a `ready` predicate from day one.** The HRV screen shipped without them and
  its time-domain metrics fed literal zeros into the rolling scale for the whole first window; this
  catalogue does not get to relearn that.

`MetricSpec` is imported from `nfb.hrv.metrics`, where it already says of itself that it is
sensor-agnostic. If a third modality ever imports it too, it should move up a level rather than be
copied — the same rule the console applies to its output mechanisms.
"""

from __future__ import annotations

from typing import Callable

from ..hrv.metrics import MetricSpec

READY_SECONDS_MIN_SAMPLES = 60
"""Frames before the windowed numbers mean anything: two seconds at the device's ~30 Hz — the
smoothing window's worth. The settling tracker upstream already holds quality at 0.5 for eleven
seconds; this is only the floor under the arithmetic itself."""


def _ready(n: int) -> Callable[[object], bool]:
    return lambda s: bool(getattr(s, "samples_seen", 0) >= n)


HEG_METRICS: dict[str, MetricSpec] = {
    "hbo": MetricSpec(
        id="hbo", label="HbO (smoothed)", unit="",
        description="oxygenated hemoglobin reaching the prefrontal cortex",
        detail=("The mainline activation signal: rises when oxygen-rich blood reaches the "
                "prefrontal cortex, which usually means more neural activity there. Smoothed over "
                "two seconds; the projection's absolute scale is arbitrary, so pair it with a "
                "self-calibrating scaling rather than reading the number itself.\n\n"
                "Note the sign convention: the projection is IR intensity, and more arterial HbO "
                "absorbs more IR — the direction row is where polarity lives, as everywhere else."),
        extract=lambda s: s.smoothed_hbo, ready=_ready(READY_SECONDS_MIN_SAMPLES),
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
    "hbr": MetricSpec(
        id="hbr", label="HbR (smoothed)", unit="",
        description="deoxygenated hemoglobin — usually anticorrelated with HbO",
        detail=("What is left after the tissue has consumed the oxygen. During activation it "
                "usually falls slightly, because oxygen-rich blood arrives faster than it is used. "
                "More interesting as part of HbO − HbR than on its own; offered for diagnostics and "
                "for slicing the signal apart."),
        extract=lambda s: s.smoothed_hbr, ready=_ready(READY_SECONDS_MIN_SAMPLES),
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
    "oxy_index": MetricSpec(
        id="oxy_index", label="HbO − HbR", unit="",
        description="the classical oxygenation difference — amplifies activation",
        detail=("During activation HbO rises and HbR usually falls, so the difference swings "
                "further than either alone — and it cancels much of the shared cardiac, respiratory "
                "and scalp noise, which makes it the closest thing here to a brain-specific number. "
                "The cost is summing the noise of two streams."),
        extract=lambda s: s.smoothed_hbo - s.smoothed_hbr,
        ready=_ready(READY_SECONDS_MIN_SAMPLES),
        strategies=("zscore", "rolling_minmax", "tanh", "fixed_range"),
    ),
    "hbo_variability": MetricSpec(
        id="hbo_variability", label="HbO variability", unit="",
        description="how jittery the activation is over the last five seconds",
        detail=("Standard deviation of smoothed HbO over the stability window: independent of the "
                "level — you can be highly activated and steady, or lightly activated and bouncing. "
                "The meditation, flow and deep-work metric. **Low is the good end**, so it defaults "
                "to decrease; motion artifacts spike it, which is what the motion gate is for."),
        extract=lambda s: s.hbo_std_short, invert=True,
        ready=_ready(150),
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
    "oxy_variability": MetricSpec(
        id="oxy_variability", label="HbO − HbR variability", unit="",
        description="jitter of the brain-specific component",
        detail=("The same steadiness reading on HbO − HbR, whose common-mode cancellation makes the "
                "jitter that remains harder to dismiss as breathing or posture. Higher per-sample "
                "noise than plain HbO variability — two streams' worth — but the noise kept is more "
                "diagnostic. **Low is the good end.**"),
        extract=lambda s: s.oxy_std_short, invert=True,
        ready=_ready(150),
        strategies=("zscore", "rolling_minmax", "fixed_range"),
    ),
    "hbo_trend": MetricSpec(
        id="hbo_trend", label="HbO trend", unit="/s",
        description="the slope of activation — positive while it is rising",
        detail=("Least-squares slope of smoothed HbO over the trend window: direction of change, "
                "not level. Catches the moment of re-engaging after zoning out, and the start of a "
                "slow drift down. Under z-score, the middle means flat. The noisiest metric here — "
                "slopes amplify whatever jitter the smoothing left."),
        extract=lambda s: s.hbo_slope, ready=_ready(150),
        strategies=("zscore", "tanh", "rolling_minmax", "fixed_range"),
    ),
}

DEFAULT_HEG_METRIC = "oxy_index"
"""`oxy_index` is the default for the reason its
catalogue entry gives — it is the closest to a brain-specific quantity, and the common-mode noise it
cancels is exactly what a headband on a forehead is full of."""
