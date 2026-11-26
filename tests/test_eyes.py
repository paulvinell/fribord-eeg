"""The subtraction of eye movements: the fit, the subtractor and the blink detector, on artificial arrays."""

from __future__ import annotations

import numpy as np

from nfb.dsp.eyes import EyeSubtractor, EyeWeights, fit, slow_level

from conftest import eye_source, plausible_eeg, wandering

RATE = 250.0
DIRS = {"up": (1.0, 0.0), "down": (-1.0, 0.0), "left": (0.0, -1.0), "right": (0.0, 1.0)}
ORDER = ("up", "down", "left", "right")
TRIPS, HOLD_S = 16, 2.0                 # 2 s at the centre, then 2 s at the edge
ROT_S = TRIPS * 2 * HOLD_S
TONES = [ROT_S + 2.0 + 1.5 * k for k in range(12)]
END_S = TONES[-1] + 1.5


def _gaze(t: float) -> tuple[float, float]:
    if t >= ROT_S:
        return (0.0, 0.0)
    trip, into = divmod(t, 2 * HOLD_S)
    return DIRS[ORDER[int(trip) % 4]] if into >= HOLD_S else (0.0, 0.0)


def _channels(rot: dict, blink: dict, seed: int = 3, eye_uv: float = 100.0, level_uv: float = 2.0,
              drift: dict | None = None):
    """Training channel 'y' and extras, each a wandering level, some EEG, and the eye in its proportions:
    `rot[name] = (vertical, horizontal)`, `blink[name]` a number. `drift[name]` adds a straight line, in µV
    each second."""
    v, h, b = eye_source(_gaze, TONES, END_S, RATE)
    out = {}
    for k, name in enumerate(rot):
        level = wandering(1.5, END_S, RATE, seed + k, level_uv)
        eeg = 0.15 * level_uv * plausible_eeg(END_S, RATE, 1, seed=seed + 50 + k)         # a seed no level uses
        rv, rh = rot[name]
        line = (drift or {}).get(name, 0.0) * np.arange(len(v)) / RATE
        out[name] = level + eeg + line + eye_uv * (rv * v + rh * h + blink[name] * b)
    return out


def _schedule():
    trips = [(int((4.0 * j + 2.0) * RATE), int((4.0 * j + 4.0) * RATE)) for j in range(TRIPS)]
    learn = trips[:8]
    test = {"up-down": [m for j, m in enumerate(trips[8:]) if j % 4 < 2],
            "left-right": [m for j, m in enumerate(trips[8:]) if j % 4 >= 2]}
    tones = [int(s * RATE) for s in TONES]
    rest = [(int((4.0 * j + (0.7 if j else 0.0)) * RATE), int((4.0 * j + 2.0) * RATE)) for j in range(TRIPS + 1)]
    return learn, test, tones[:6], tones[6:], rest


def _fit(ch: dict):
    names = list(ch)
    return fit(ch[names[0]], np.vstack([ch[n] for n in names[1:]]), RATE, *_schedule())


def test_two_extras_recover_the_rotation_weights_and_shrink_the_test_half():
    rot = {"y": (0.10, 0.05), "x1": (1.0, 0.3), "x2": (-0.2, 1.0)}
    blink = {"y": 0.2, "x1": 2.0, "x2": 0.4}
    result = _fit(_channels(rot, blink, level_uv=0.5))
    m = np.array([rot["x1"], rot["x2"]]).T
    expected = np.linalg.solve(m, np.array(rot["y"]))
    assert np.allclose(result.weights.rotation, expected, atol=0.015), (result.weights.rotation, expected)
    assert result.rotations_used and result.blinks_used
    for row in ("up-down", "left-right", "blinks"):
        without, with_ = result.sizes[row]
        assert with_ < 0.35 * without, (row, result.sizes[row])
    assert result.weights.blink_channel == 0 and result.weights.blink_sign == 1.0


def test_one_extra_with_opposite_proportions_on_the_two_axes_gains_nothing():
    """One weight cannot serve an axis that reaches the training channel up and another that reaches it
    down: the result says no gain, and nothing is subtracted for the rotations. Only the eye is on the
    channels, so the weight is zero to the last digit and the table shows no change."""
    rot = {"y": (0.10, -0.10), "x1": (1.0, 1.0)}
    blink = {"y": 0.0, "x1": 2.0}
    result = _fit(_channels(rot, blink, level_uv=0.0))
    assert not result.rotations_used, result.sizes
    assert result.weights.rotation == (0.0,)


def test_a_weight_that_makes_the_untested_half_larger_is_not_used():
    """The training channel takes the eye one way in the first half and the other way in the second, as an
    electrode that moved would: the weight from the first half is not used."""
    blink = {"y": 0.0, "x1": 2.0}
    first = _channels({"y": (0.1, 0.1), "x1": (1.0, 1.0)}, blink, level_uv=0.5)
    second = _channels({"y": (-0.1, -0.1), "x1": (1.0, 1.0)}, blink, level_uv=0.5)
    half = int(48.0 * RATE)
    spliced = {name: np.r_[first[name][:half], second[name][half:]] for name in first}
    result = _fit(spliced)
    assert not result.rotations_used, result.sizes
    assert result.weights.rotation == (0.0,)


def test_blinks_that_do_not_reach_the_training_channel_are_not_subtracted():
    """Blinks on an extra channel alone give no blink term; the rotation weight is still used."""
    rot = {"y": (0.10, 0.0), "x1": (1.0, 0.0)}
    blink = {"y": 0.0, "x1": 2.0}
    result = _fit(_channels(rot, blink, seed=8, level_uv=0.5))
    assert result.rotations_used
    assert result.weights.rotation[0] > 0.08
    assert not result.blinks_used and result.weights.blink_channel is None, result.sizes


def test_weights_that_make_the_rest_larger_are_not_used():
    """The extra channel removes the eye movements and has a large wander of its own. The wander comes
    with the weight into every trial, so the eye movements are not subtracted."""
    rot = {"y": (0.10, 0.10), "x1": (1.0, 1.0)}
    blink = {"y": 0.0, "x1": 2.0}
    ch = _channels(rot, blink, level_uv=0.5)
    quiet = _fit(ch)
    assert quiet.rotations_used and not quiet.rest_larger, quiet.sizes
    ch["x1"] = ch["x1"] + wandering(1.5, END_S, RATE, 40, 30.0)
    result = _fit(ch)
    for row in ("up-down", "left-right"):
        assert result.sizes[row][1] < result.sizes[row][0], ("the eye movements still gain", result.sizes)
    assert result.sizes["rest"][1] > result.sizes["rest"][0], result.sizes
    assert result.rest_larger and not result.rotations_used
    assert not any(result.weights.rotation)


def _weights(r=0.1, b=0.3, threshold=50.0, samples=100) -> EyeWeights:
    return EyeWeights(rotation=(r,), blink=(b,), blink_channel=0, blink_sign=1.0, blink_threshold=threshold,
                      blink_samples=samples, rate_hz=RATE)


def _blocks(sub: EyeSubtractor, y, x, block: int = 50):
    out = []
    for i in range(0, len(y), block):
        corrected, _ok = sub.process(y[i:i + block], x[None, i:i + block])
        out.append(corrected)
    return np.concatenate(out)


def test_a_blink_is_removed_with_the_blink_weight():
    n = int(4 * RATE)
    _v, _h, b = eye_source(lambda _t: (0.0, 0.0), [1.0], 4.0, RATE)
    x = 200.0 * b
    y = 0.3 * x + 20.0
    sub = EyeSubtractor(_weights(r=0.1, b=0.3, threshold=60.0, samples=int(0.4 * RATE)))
    corrected = _blocks(sub, y, x)
    assert sub.blinks == 1
    plain = y - 0.1 * x
    assert np.ptp(slow_level(plain, RATE)) > 10.0
    # Before the detector sees it, the rise carries the rotation weight; what is left is that part only.
    assert np.ptp(slow_level(corrected, RATE)) < 0.25 * np.ptp(slow_level(plain, RATE)), np.ptp(corrected)
    assert len(corrected) == n


def test_a_glance_that_does_not_return_is_handed_to_the_rotation_weight():
    """A fast step that stays: the detector takes it for a blink, and after two calibrated durations the
    blink term stops and the rotation weight applies alone."""
    n = int(6 * RATE)
    x = np.where(np.arange(n) >= int(1.0 * RATE), 200.0, 0.0)
    y = 0.1 * x
    duration = int(0.4 * RATE)
    sub = EyeSubtractor(_weights(r=0.1, b=0.3, threshold=60.0, samples=duration))
    corrected = _blocks(sub, y, x)
    assert sub.blinks == 1 and not sub.in_blink
    late = corrected[int(1.0 * RATE) + 3 * duration:]
    assert np.allclose(late, late[0]), "the rotation weight alone, so the term is constant"
    assert abs(late[0]) < 0.25 * 200.0 * 0.2


def test_an_invalid_extra_sample_is_reported_and_stops_a_blink():
    """The samples reported valid are those where every extra sample is valid, and an invalid one ends the
    blink in progress."""
    sub = EyeSubtractor(_weights(threshold=60.0, samples=int(0.4 * RATE)))
    x = np.where(np.arange(int(2 * RATE)) >= int(RATE), 200.0, 0.0)
    at = int(RATE) + 20
    sub.process(0.1 * x[:at], x[None, :at])
    assert sub.in_blink, "the step is taken for a blink"
    valid = np.ones(10, dtype=bool)
    valid[4] = False
    _out, ok = sub.process(0.1 * x[at:at + 10], x[None, at:at + 10], valid)
    assert list(ok) == list(valid)
    assert not sub.in_blink


def test_without_a_blink_channel_the_rotation_weight_applies_always():
    weights = EyeWeights(rotation=(0.5,), blink=(0.5,), blink_channel=None, blink_sign=1.0,
                         blink_threshold=1.0, blink_samples=10, rate_hz=RATE)
    sub = EyeSubtractor(weights)
    out, _ok = sub.process(np.full(20, 10.0), np.full((1, 20), 4.0))
    assert np.allclose(out, 8.0)


def test_a_trip_is_its_edge_against_both_centres_so_a_straight_drift_cancels():
    from nfb.dsp.eyes import trip_deltas

    m, r = int(3 * RATE), int(5 * RATE)
    n = int(8 * RATE)
    level = 37.0 * np.arange(n) / RATE                        # a straight drift of 37 µV each second
    level[m:r] += 10.0                                        # the eye at the edge
    level[m:m + int(0.7 * RATE)] += 500.0                     # while the filter arrives: not in it
    assert abs(trip_deltas(level[None, :], RATE, [(m, r)])[0, 0] - 10.0) < 1e-9


DRIFT_UV_S = 100.0
"""A drift of this size each second on one input, the size seen on a real head."""


def test_a_steady_drift_on_an_extra_or_on_the_training_channel_leaves_the_weights_and_the_gain():
    rot = {"y": (0.10, 0.05), "x1": (1.0, 0.3), "x2": (-0.2, 1.0)}
    blink = {"y": 0.2, "x1": 2.0, "x2": 0.4}
    expected = np.linalg.solve(np.array([rot["x1"], rot["x2"]]).T, np.array(rot["y"]))
    for name in ("x1", "y"):
        result = _fit(_channels(rot, blink, level_uv=0.5, drift={name: DRIFT_UV_S}))
        assert np.allclose(result.weights.rotation, expected, atol=0.015), (name, result.weights.rotation)
        assert result.rotations_used and result.blinks_used, (name, result.sizes)
        for row in ("up-down", "left-right", "blinks"):
            without, with_ = result.sizes[row]
            assert with_ < 0.35 * without, (name, row, result.sizes)
        assert result.sizes["rest"][0] < 5.0, ("the rest leaves the drift out", result.sizes)
