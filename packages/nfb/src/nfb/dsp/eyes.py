"""Subtraction of eye movements and blinks from a slow channel, with weights from a led calibration.

Eye rotations and blinks reach the trained channel `y` and each extra channel `x_k`, each source in its own
proportion. The corrected sample is `y - sum r_k x_k`, and inside a blink also `- sum (b_k - r_k)(x_k - x_k0)`,
with `x_k0` the extra sample where the blink term starts, so a change of weight adds no step. A standing
offset times a fixed weight is a constant, which each trial's baseline removes.

The weights come from movements whose times the calibration knows: least squares on the first half, a test
of the size of the movements on the trained channel on the second half. A weight set that does not make
the test movements smaller is not used. Nothing here knows a device or a screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .filters import lowpass
from .scp import SCP_LOWPASS_HZ, SCP_ORDER

SETTLE_S = 0.7
"""How long after a move its level is left out: the eyes start about 0.2 s after the dot, and the 1 Hz order-2
lowpass reaches a step's new level about 0.5 s after it."""

BLINK_BAND_HZ = 10.0
"""A blink takes 200 ms to 400 ms, so its shape is below 10 Hz; the band keeps it and takes off muscle."""

BLINK_BEFORE_S = 0.5
"""The level on each side of a blink's window, the one before the tone and the one after the window."""
BLINK_WINDOW_S = 1.0
"""A blink to a tone starts within the reaction time, well inside 1 s."""

BLINK_MARGIN_S = 0.1
"""Added to each side of the half-height span: the rise and the tail of a blink outside it."""

RISE_S = 0.15
"""A lid closes in about 100 ms, so a blink rises past its threshold inside this; slow drift never does."""

END_FRACTION = 0.3
TIMEOUT_DURATIONS = 2.0
"""A blink that does not come back below 30% of its peak in 2 calibrated durations is a rotation."""


@dataclass(frozen=True)
class EyeWeights:
    """The fixed weights of one calibration, per extra channel in order, and the blink detector's settings.
    `blink_channel` is None when the blinks are not subtracted: the rotation weights then apply always."""

    rotation: tuple[float, ...]
    blink: tuple[float, ...]
    blink_channel: int | None
    blink_sign: float
    blink_threshold: float
    blink_samples: int
    rate_hz: float

    @property
    def subtracts(self) -> bool:
        return any(self.rotation) or self.blink_channel is not None


@dataclass(frozen=True)
class EyeFit:
    """What a calibration found: the weights in use, and the sizes on the training channel in µV of the
    half it did not learn from, without and with the subtraction, by row."""

    weights: EyeWeights
    sizes: dict[str, tuple[float, float]]
    rotations_used: bool
    blinks_used: bool
    blinks_left_out: int = 0
    """Blinks the calibration did not fit from, because a channel touched its input limit in them."""
    rest_larger: bool = False
    """The eye movements have a gain on their own rows, and are not subtracted because the rest row grows."""


def slow_level(x: np.ndarray, rate_hz: float) -> np.ndarray:
    """The trained signal's own lowpass, primed at the first sample, over a whole array."""
    return _filtered(x, SCP_LOWPASS_HZ, rate_hz, SCP_ORDER)


def blink_band(x: np.ndarray, rate_hz: float) -> np.ndarray:
    return _filtered(x, BLINK_BAND_HZ, rate_hz, 2)


def _filtered(x: np.ndarray, hz: float, rate_hz: float, order: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f = lowpass(hz, rate_hz, order=order)
    if f is None or not len(x):
        return x.copy()
    f.prime(x[0])
    return f.process(x)


def trip_deltas(slow: np.ndarray, rate_hz: float, trips) -> np.ndarray:
    """Per trip `(out, back)` of the dot, per channel: the mean slow level at the edge, from `SETTLE_S` after the
    move out to the move back, minus the mean of two centre windows of the same width, one ending at the move
    out and one starting `SETTLE_S` after the move back. The two centre windows sit at equal distances before
    and after the edge window, so a level that drifts in a straight line adds the same to the edge and to the
    mean of the centres, and cancels. `slow` is (channels, samples); the result is (trips, channels)."""
    settle = round(SETTLE_S * rate_hz)
    out = []
    for m, r in trips:
        width = r - m - settle
        edge = slow[:, m + settle:r].mean(axis=1)
        before = slow[:, m - width:m].mean(axis=1)
        after = slow[:, r + settle:r + settle + width].mean(axis=1)
        out.append(edge - (before + after) / 2)
    return np.asarray(out).reshape(len(out), slow.shape[0])


def least_squares(dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """The weights that take `dx @ w` closest to `dy`. `dx` is (events, extras)."""
    w, *_ = np.linalg.lstsq(np.asarray(dx, dtype=float), np.asarray(dy, dtype=float), rcond=None)
    return w


@dataclass(frozen=True)
class Blink:
    """One blink to a tone: the excursions of every channel over its span, training channel first, and the
    signed peak of each extra channel."""

    excursions: np.ndarray
    peaks: np.ndarray


def blink_spans(band: np.ndarray, rate_hz: float, tones) -> list[Blink]:
    """The blink after each tone. `band` is (channels, samples) in the blink band, training channel first.
    The excursion is measured against a straight line through the level before the tone and the level
    after the window, so a drift under the blink is not part of it. The span runs while the largest extra
    excursion is above half its peak, widened by the margin."""
    before, window, margin = (round(s * rate_hz) for s in (BLINK_BEFORE_S, BLINK_WINDOW_S, BLINK_MARGIN_S))
    out = []
    for tone in tones:
        first = band[:, tone - before:tone].mean(axis=1)
        last = band[:, tone + window:tone + window + before].mean(axis=1)
        # The two levels stand at the centres of their windows, half a window outside the blink's window.
        at = (np.arange(window) + before / 2) / (window + before)
        e = band[:, tone:tone + window] - (first[:, None] + (last - first)[:, None] * at[None, :])
        extras = e[1:]
        at = np.abs(extras).argmax(axis=1)
        peaks = extras[np.arange(len(extras)), at]
        j = int(np.abs(peaks).argmax())
        half = abs(peaks[j]) / 2
        lo = hi = int(at[j])
        while lo > 0 and abs(extras[j, lo - 1]) > half:
            lo -= 1
        while hi < e.shape[1] - 1 and abs(extras[j, hi + 1]) > half:
            hi += 1
        lo, hi = max(0, lo - margin), min(e.shape[1], hi + 1 + margin)
        out.append(Blink(e[:, lo:hi], peaks))
    return out


def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else math.nan


SHOWN_UV = 0.1
"""The table's resolution. A gain counts only where the table shows it, so a weight of zero plus rounding
never reads as one."""


def _spread(slow: np.ndarray, stretches) -> float:
    """The standard deviation of the level inside the stretches, each about its own straight line, so neither
    the drift from one stretch to the next nor the drift inside one is in it."""
    parts = []
    for lo, hi in stretches:
        if hi - lo >= 2:
            x = np.arange(hi - lo, dtype=float)
            parts.append(slow[lo:hi] - np.polyval(np.polyfit(x, slow[lo:hi], 1), x))
    return float(np.sqrt(np.mean(np.concatenate(parts) ** 2))) if parts else math.nan


def _smaller(corrected: float, uncorrected: float) -> bool:
    return round(corrected / SHOWN_UV) < round(uncorrected / SHOWN_UV)


def fit(y: np.ndarray, xs: np.ndarray, rate_hz: float, learn_trips, test_trips: dict[str, list],
        learn_tones, test_tones, rest: list[tuple[int, int]]) -> EyeFit:
    """The weights from one calibration's raw arrays, in µV: `y` the training channel, `xs` (extras, samples).
    Trips are `(out, back)` sample indices and tones sample indices; `test_trips` groups the test half's
    trips by row name. `rest` holds the stretches with the eyes still at the centre."""
    xs = np.atleast_2d(np.asarray(xs, dtype=float))
    raw = np.vstack([np.asarray(y, dtype=float), xs])
    slow = np.vstack([slow_level(c, rate_hz) for c in raw])
    learn = trip_deltas(slow, rate_hz, learn_trips)
    r = least_squares(learn[:, 0], learn[:, 1:])

    sizes: dict[str, tuple[float, float]] = {}
    test_all = []
    for name, trips in test_trips.items():
        d = trip_deltas(slow, rate_hz, trips)
        corrected = d[:, 0] - d[:, 1:] @ r
        sizes[name] = (_rms(d[:, 0]), _rms(corrected))
        test_all.append((d[:, 0], corrected))
    rotations_used = _smaller(_rms(np.concatenate([c for _, c in test_all])),
                              _rms(np.concatenate([u for u, _ in test_all])))
    if not rotations_used:
        r = np.zeros_like(r)

    band = np.vstack([blink_band(c, rate_hz) for c in raw])
    learned = blink_spans(band, rate_hz, learn_tones)
    ey = np.concatenate([b.excursions[0] for b in learned])
    ex = np.hstack([b.excursions[1:] for b in learned]).T
    b = least_squares(ey, ex)
    tested = blink_spans(band, rate_hz, test_tones)
    ty = np.concatenate([t.excursions[0] for t in tested])
    tx = np.hstack([t.excursions[1:] for t in tested]).T
    sizes["blinks"] = (_rms(ty), _rms(ty - tx @ b))
    blinks_used = _smaller(*reversed(sizes["blinks"]))

    peaks = np.array([bl.peaks for bl in learned])                  # (blinks, extras)
    channel = int(np.median(np.abs(peaks), axis=0).argmax())
    median_peak = float(np.median(peaks[:, channel]))

    def weights(rotation, blinks: bool) -> EyeWeights:
        return EyeWeights(
            rotation=tuple(float(v) for v in rotation),
            blink=tuple(float(v) for v in (b if blinks else rotation)),
            blink_channel=channel if blinks else None,
            blink_sign=1.0 if median_peak >= 0 else -1.0,
            blink_threshold=abs(median_peak) / 2,
            blink_samples=int(np.median([bl.excursions.shape[1] for bl in learned])),
            rate_hz=rate_hz)

    def at_rest(w: EyeWeights) -> float:
        corrected, _ok = EyeSubtractor(w).process(raw[0], raw[1:])
        return _spread(slow_level(corrected, rate_hz), rest)

    # The rotation term acts on every sample, and an extra channel brings its own drift and noise with it.
    # The rest has no eye movement to remove, so weights that make the rest larger only add, and are not used.
    without = _spread(slow[0], rest)
    sizes["rest"] = (without, at_rest(weights(r, blinks_used)))
    rest_larger = bool(rotations_used and _smaller(without, at_rest(weights(r, False))))
    if rest_larger:
        rotations_used, r = False, np.zeros_like(r)
    return EyeFit(weights(r, blinks_used), sizes, bool(rotations_used), bool(blinks_used), rest_larger=rest_larger)


class EyeSubtractor:
    """The corrected training samples, fed in order with aligned raw blocks in µV. Causal: the blink term
    starts at the sample where the detector sees the blink, and ends where the blink returns below
    `END_FRACTION` of its peak, or after `TIMEOUT_DURATIONS` durations, where it was a rotation."""

    def __init__(self, weights: EyeWeights):
        self.weights = weights
        self.r = np.asarray(weights.rotation, dtype=float)
        self.extra = np.asarray(weights.blink, dtype=float) - self.r
        self.rise = max(1, round(RISE_S * weights.rate_hz))
        self.timeout = max(1, round(TIMEOUT_DURATIONS * weights.blink_samples))
        self.blinks = 0
        """Blinks started since the subtractor was made, for a test to count."""
        self.reset()

    def reset(self) -> None:
        """A discontinuity: the band filter and the detector start again."""
        self._band = lowpass(BLINK_BAND_HZ, self.weights.rate_hz, order=2)
        self._primed = False
        self._history: list[float] = []         # the band of the last `rise` samples
        self._columns: list[np.ndarray] = []    # and the raw extra samples beside them
        self._x0: np.ndarray | None = None
        self._ref = 0.0
        self._peak = 0.0
        self._since = 0

    @property
    def in_blink(self) -> bool:
        return self._x0 is not None

    def process(self, y: np.ndarray, xs: np.ndarray, valid: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Corrected `y`, and which samples are valid: those where every extra sample is valid. An invalid
        sample stops a blink and starts the detector again."""
        y = np.asarray(y, dtype=float)
        xs = np.atleast_2d(np.asarray(xs, dtype=float))
        ok = np.ones(len(y), dtype=bool) if valid is None else np.asarray(valid, dtype=bool).copy()
        out = y - self.r @ xs
        channel = self.weights.blink_channel
        if channel is None or not len(y):
            return out, ok
        if not self._primed:
            self._band.prime(xs[channel, 0])
            self._primed = True
        band = self._band.process(xs[channel]) * self.weights.blink_sign
        threshold = self.weights.blink_threshold
        for i, v in enumerate(band):
            if not ok[i]:
                self._x0, self._history, self._columns = None, [], []
                continue
            history = self._history
            if self._x0 is None and len(history) >= self.rise and v - history[0] >= threshold:
                # The blink started where the rise did: the term removes all of it from here, and is 0
                # again where the blink comes back to that level.
                start = int(np.argmin(history))
                self._x0, self._ref = self._columns[start], history[start]
                self._peak, self._since = v - self._ref, 0
                self.blinks += 1
            elif self._x0 is not None:
                self._since += 1
                self._peak = max(self._peak, v - self._ref)
                if v - self._ref < END_FRACTION * self._peak or self._since > self.timeout:
                    self._x0 = None
            if self._x0 is not None:
                out[i] -= self.extra @ (xs[:, i] - self._x0)
            history.append(float(v))
            self._columns.append(xs[:, i].copy())
            if len(history) > self.rise:
                del history[0], self._columns[0]
        return out, ok
