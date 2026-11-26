"""The calibration of the eye subtraction as a state machine on signal time, fed artificial readings."""

from __future__ import annotations

import numpy as np
import pytest

from nfb_console.eye_calibration import (BACK, BLINKS, CENTRE, GAP, LINES, LONG, PLANS, ROTATIONS, SHORT,
                                         TOO_FEW, EyeCalibration)

from nfb_console.trials import JUMP_S

from conftest import eye_source, wandering

RATE = 250.0
PLACE = {CENTRE: (0.0, 0.0), "up": (1.0, 0.0), "down": (-1.0, 0.0), "left": (0.0, -1.0), "right": (0.0, 1.0)}


def eye_channels(rot: dict, blink: dict, seconds: float | None = None, seed: int = 4, eye_uv: float = 100.0,
                 level_uv: float = 0.5, plan: str = SHORT) -> list[np.ndarray]:
    """Channels in order, each a wandering level and the eye that follows the calibration's own dot."""
    version = PLANS[plan]
    seconds = version.total_s + 2.0 if seconds is None else seconds
    v, h, b = eye_source(lambda t: PLACE[version.dot_at(t)[1]], version.tones, seconds, RATE)
    out = []
    for k, (rv, rh) in enumerate(rot):
        out.append(wandering(1.5, seconds, RATE, seed + k, level_uv) + eye_uv * (rv * v + rh * h + blink[k] * b))
    return out


def _feed(cal: EyeCalibration, channels: list[np.ndarray], block: int = 50, problem_at=None, skip=None) -> int:
    tones = 0
    n = len(channels[0])
    i = 0
    while i + block <= n and cal.running:
        if skip is not None and skip[0] <= i < skip[1]:
            i += block
            continue
        spans = problem_at if problem_at is None or isinstance(problem_at[0], (list, tuple)) else [problem_at]
        index = np.arange(i, i + block)
        limit = np.zeros(block, dtype=bool)
        for lo, hi in spans or ():
            limit |= (index >= lo) & (index < hi)
        tones += cal.feed(i, channels[0][i:i + block], np.vstack([c[i:i + block] for c in channels[1:]]),
                          limit=limit, limit_reason="ch1 is at the input limit")
        i += block
    return tones


def test_the_dot_is_at_an_edge_only_in_a_trip_and_the_blinks_follow_the_eye_movements():
    """The dot is at a direction exactly inside that direction's trips and at the centre between them; the
    blinks part starts where the eye movements end. The long version has more of both, and each part has
    its own line."""
    for plan in PLANS.values():
        trips = plan.trip_times()
        for t in np.arange(0.0, plan.total_s, 0.25):
            part, place = plan.dot_at(float(t))
            assert part == (ROTATIONS if t < plan.rotations_s else BLINKS), t
            inside = [d for d, spans in trips.items() if any(a <= t < b for a, b in spans)]
            assert place == (inside[0] if inside else CENTRE), (t, place, inside)
        assert all(trips.values()), "every direction has a trip"
        assert all(plan.rotations_s < tone < plan.total_s for tone in plan.tones)
    short, long = PLANS[SHORT], PLANS[LONG]
    assert short.trips < long.trips and len(short.tones) < len(long.tones)
    assert len(set(LINES.values())) == 2


def test_a_calibration_fed_to_its_end_recovers_the_weights_and_sounds_each_tone():
    rot = [(0.10, 0.05), (1.0, 0.3), (-0.2, 1.0)]
    cal = EyeCalibration(RATE, 0, (1, 2))
    tones = _feed(cal, eye_channels(rot, [0.2, 2.0, 0.4]))
    assert cal.done and cal.stopped is None
    assert tones == 8
    expected = np.linalg.solve(np.array(rot[1:]).T, np.array(rot[0]))
    assert np.allclose(cal.result.weights.rotation, expected, atol=0.015), cal.result.weights
    assert cal.result.rotations_used and cal.result.blinks_used
    for row in ("up-down", "left-right", "blinks"):
        without, with_ = cal.result.sizes[row]
        assert with_ < 0.35 * without, (row, cal.result.sizes)


def test_the_long_calibration_learns_from_twice_the_events():
    rot = [(0.10, 0.05), (1.0, 0.3), (-0.2, 1.0)]
    cal = EyeCalibration(RATE, 0, (1, 2), PLANS[LONG])
    tones = _feed(cal, eye_channels(rot, [0.2, 2.0, 0.4], plan=LONG))
    assert cal.done and tones == 12
    expected = np.linalg.solve(np.array(rot[1:]).T, np.array(rot[0]))
    assert np.allclose(cal.result.weights.rotation, expected, atol=0.015), cal.result.weights


def test_the_rest_is_the_time_at_the_centre_after_the_return_has_arrived():
    """Each rest is time with the dot at the centre. After a return it starts once the level after the move
    has settled, and the last one is the lead before the first tone."""
    from nfb.dsp.eyes import SETTLE_S

    plan = PLANS[SHORT]
    returns = sorted(b for spans in plan.trip_times().values() for _a, b in spans)
    rest = plan.rest()
    for a, b in rest:
        assert all(plan.dot_at(float(t))[1] == CENTRE for t in np.linspace(a, b, 20, endpoint=False)), (a, b)
    assert rest[0][0] == 0.0, "nothing returns before the first trip"
    assert [a for a, _b in rest[1:]] == pytest.approx([r + SETTLE_S for r in returns])
    assert plan.rotations_s < rest[-1][1] <= plan.tones[0], "the lead before the first tone"


def test_a_replay_in_other_blocks_gives_the_same_result():
    channels = eye_channels([(0.1, 0.0), (1.0, 0.2)], [0.1, 2.0])
    results = []
    for block in (8, 50, 250):
        cal = EyeCalibration(RATE, 0, (1,))
        _feed(cal, channels, block=block)
        results.append(cal.result.weights.rotation)
    assert np.allclose(results[0], results[1]) and np.allclose(results[1], results[2]), results


def _go_on_from(cal: EyeCalibration, channels: list[np.ndarray], seconds: float, block: int = 50) -> int:
    """Resume, then feed the channels from the calibration's own time `seconds`, at sample indices far
    past the first feed, as the signal after a wait arrives."""
    cal.resume()
    tones, base = 0, 10_000_000
    i = round(seconds * RATE)
    while i + block <= len(channels[0]) and cal.running:
        tones += cal.feed(base + i, channels[0][i:i + block], np.vstack([c[i:i + block] for c in channels[1:]]))
        i += block
    return tones


ROT1, BLINK1 = [(0.1, 0.0), (1.0, 0.2)], [0.1, 2.0]


def test_a_channel_at_the_input_limit_in_the_rotations_holds_it_and_it_goes_on_from_their_start():
    channels = eye_channels(ROT1, BLINK1)
    cal = EyeCalibration(RATE, 0, (1,))
    _feed(cal, channels, problem_at=(int(20 * RATE), int(21 * RATE)))
    assert cal.held == "ch1 is at the input limit" and cal.running and cal.stopped is None
    assert cal.seconds == 0.0, "the part in progress is dropped"
    _go_on_from(cal, channels, 0.0)
    assert cal.done
    assert abs(cal.result.weights.rotation[0] - 0.1) < 0.015, cal.result.weights


def test_a_channel_at_the_input_limit_in_the_blinks_keeps_the_rotations():
    channels = eye_channels(ROT1, BLINK1)
    plan = PLANS[SHORT]
    cal = EyeCalibration(RATE, 0, (1,))
    _feed(cal, channels, problem_at=(int(38 * RATE), int(40 * RATE)))
    assert cal.held and cal.seconds == plan.rotations_s, "2 s at the limit holds, and the complete part is kept"
    tones = _go_on_from(cal, channels, plan.rotations_s)
    assert cal.done and tones == plan.blinks, "every tone of the part sounds again"
    assert abs(cal.result.weights.rotation[0] - 0.1) < 0.015, cal.result.weights


def _spikes(tone_numbers) -> list:
    """0.1 s at the limit inside each named blink's window, as a blink that reaches the limit gives."""
    plan = PLANS[SHORT]
    return [(int((plan.tones[k] + 0.4) * RATE), int((plan.tones[k] + 0.5) * RATE)) for k in tone_numbers]


def test_blinks_that_reach_the_limit_are_left_out_and_the_calibration_completes():
    channels = eye_channels(ROT1, BLINK1)
    cal = EyeCalibration(RATE, 0, (1,))
    _feed(cal, channels, problem_at=_spikes([1, 4, 6]))
    assert cal.done and cal.held is None, cal.held
    assert cal.result.blinks_left_out == 3
    assert abs(cal.result.weights.rotation[0] - 0.1) < 0.015


def test_too_few_blinks_left_in_one_half_asks_for_the_blinks_again():
    channels = eye_channels(ROT1, BLINK1)
    plan = PLANS[SHORT]
    cal = EyeCalibration(RATE, 0, (1,))
    _feed(cal, channels, problem_at=_spikes([0, 1, 2]))
    assert cal.held == TOO_FEW and cal.result is None
    assert cal.seconds == plan.rotations_s, "the eye movements are kept"


def test_a_held_calibration_takes_no_sample_until_it_goes_on():
    cal = EyeCalibration(RATE, 0, (1,))
    cal.hold("ch1 is at the input limit")
    assert cal.feed(0, np.zeros(50), np.zeros((1, 50))) == 0 and cal.seconds == 0.0 and not cal.begun
    cal.resume()
    cal.feed(5000, np.zeros(50), np.zeros((1, 50)))
    assert cal.begun and cal.seconds == 50 / RATE


def test_a_gap_of_more_than_a_second_stops_it():
    cal = EyeCalibration(RATE, 0, (1,))
    _feed(cal, eye_channels([(0.1, 0.0), (1.0, 0.2)], [0.1, 2.0]), skip=(int(30 * RATE), int(31.5 * RATE)))
    assert cal.stopped == GAP and cal.result is None


def test_a_hole_up_to_a_second_is_held_and_the_calibration_goes_on():
    """A hole of the jump limit or less is held at the last values; only a longer one stops it."""
    for hole_s in (0.4, JUMP_S):
        cal = EyeCalibration(RATE, 0, (1,))
        _feed(cal, eye_channels([(0.1, 0.0), (1.0, 0.2)], [0.1, 2.0]),
              skip=(int(30 * RATE), int((30 + hole_s) * RATE)))
        assert cal.done, (hole_s, cal.stopped)


def test_a_step_back_stops_it():
    cal = EyeCalibration(RATE, 0, (1,))
    y, x = np.zeros(50), np.zeros((1, 50))
    cal.feed(1000, y, x)
    cal.feed(500, y, x)
    assert cal.stopped == BACK


def test_a_stop_keeps_nothing_and_a_later_feed_does_nothing():
    cal = EyeCalibration(RATE, 0, (1,))
    cal.feed(0, np.zeros(50), np.zeros((1, 50)))
    cal.stop("the session pauses")
    assert cal.feed(50, np.zeros(50), np.zeros((1, 50))) == 0
    assert cal.stopped == "the session pauses" and cal.result is None


def test_extras_sent_ahead_are_kept_and_a_training_block_goes_on_as_soon_as_they_are_there():
    """The device sends each channel on its own. Extra blocks that arrive before their training blocks are
    all kept, and each training block goes on the moment its extras are there, with no wait."""
    from nfb.stream import Block, ChannelSpec

    from nfb_console.eye_calibration import OK, EyeAligner

    spec = ChannelSpec(0, RATE, "µV", 1.0)
    aligner = EyeAligner((1,), RATE)
    x = np.arange(100, dtype=float)
    aligner.push_extra(1, 0, x[:50], OK)
    aligner.push_extra(1, 50, x[50:], OK)
    out = []
    for start in (0, 50):
        aligner.push_training(Block(0, start, (0,) * 50, spec), None)
        ready = aligner.ready()
        assert len(ready) == 1, f"the block at {start} waits with its extras there"
        out.append(ready[0])
    assert np.array_equal(np.hstack([xs for _b, _m, xs, _c in out])[0], x)
    assert all((codes == OK).all() for _b, _m, _xs, codes in out)
