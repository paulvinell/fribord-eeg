"""The Mendi path, against answers known by construction.

The codec is pinned against the documented wire format; the processing chain against
signals whose right answer is arithmetic. The seams — motion, settling, held frames — get the same
scrutiny the HRV path only got after shipping, because they are the same seams.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nfb.heg import (Frame, Channel, FrameProcessor, HEG_METRICS, QUALITY_GOOD, QUALITY_MOTION,
                     QUALITY_SETTLING, SignalProcessor, SimulatedMendi)
from nfb.heg import proto
from nfb.hrv.metrics import metric_value

RATE = 30.0


def frames(seconds: float, hbo_drift=lambda t: 0.0, acc=(0, 0, 1000), start_t: float = 1000.0):
    """A synthetic stream: flat channels plus a controllable IR drift on L/R, quiet P."""
    for i in range(int(seconds * RATE)):
        t = start_t + i / RATE
        ir = 20000.0 + hbo_drift(i / RATE)
        yield Frame(t=t, l=Channel(ir, 15000.0, 200.0), r=Channel(ir, 15000.0, 200.0),
                    p=Channel(19000.0, 14000.0, 200.0), acc=acc, gyro=(0, 0, 0), temp_c=36.5)


# --- the codec --------------------------------------------------------------------------------

def test_the_sensor_write_matches_the_apps_wire_bytes():
    """Field 2 (address) as varint, field 3 (data) as little-endian fixed32, field 1 only when
    true — proto3 omits defaults, and a stop that omitted data would be indistinguishable from no
    command at all, which is why the encoder always emits it."""
    start = proto.encode_sensor_write(30, 0x0100)
    assert start == bytes([0x10, 30, 0x1D, 0x00, 0x01, 0x00, 0x00])
    stop = proto.encode_sensor_write(30, 0x0000)
    assert stop == bytes([0x10, 30, 0x1D, 0x00, 0x00, 0x00, 0x00])
    read = proto.encode_sensor_write(30, 0, read=True)
    assert read[0:2] == bytes([0x08, 0x01]), "the read flag is emitted only when set"


def test_a_frame_round_trips_including_negative_imu_values():
    """Protobuf encodes negative int32 as a 10-byte varint; decoding it unsigned gives an enormous
    number, and the decoder must sign-extend. A tilted head produces negative accelerometer values
    constantly, so this is the common case, not the edge."""
    def field(num: int, value: int) -> bytes:
        # varint field, encoding value as int64 two's complement the way protobuf does
        out = bytearray([num << 3])
        v = value & 0xFFFFFFFFFFFFFFFF
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                return bytes(out)

    buf = field(1, -512) + field(8, 20000) + field(11, 19000)
    decoded = proto.decode_frame(buf)
    assert decoded["acc_x"] == -512
    assert decoded["ir_l"] == 20000 and decoded["ir_r"] == 19000
    assert decoded["ir_p"] == 0, "missing fields default to zero, as proto3 does"


def test_the_battery_curve_is_monotonic_and_honest_at_the_ends():
    """The app's own (mV-3600)/5 calls a cell with charge left "0%" and sags across that cliff under
    streaming load; the OCV curve reports what the cell actually has."""
    values = [proto.adc_voltage_to_percent(mv) for mv in range(3200, 4300, 10)]
    assert values == sorted(values)
    assert proto.adc_voltage_to_percent(3200) == 0
    assert proto.adc_voltage_to_percent(4250) == 100
    assert proto.adc_voltage_to_percent(3600) > 0, "3.6 V is not an empty cell"


# --- the processing chain ---------------------------------------------------------------------

def test_settling_then_good_and_the_projection_arithmetic():
    proc = FrameProcessor()
    out = [proc.process(f) for f in frames(15.0)]
    assert out[0].quality == QUALITY_SETTLING
    assert out[-1].quality == QUALITY_GOOD
    # flat 20000 L/R against 19000 P: the projection is the difference, filtered to the same DC
    assert out[-1].hbo == pytest.approx(1000.0, abs=1.0)
    assert out[-1].raw_ir == pytest.approx(20000.0, abs=1.0)


def test_a_jolt_gates_and_holds_the_optics_flat_while_the_filters_advance():
    proc = FrameProcessor()
    stream = list(frames(20.0)) + list(
        frames(0.3, acc=(4000, -3000, 5000), start_t=1020.0)) + list(
        frames(10.0, start_t=1020.3))
    out = [proc.process(f) for f in stream]
    jolt_at = int(20.0 * RATE)
    during = out[jolt_at + 2]
    assert during.quality == QUALITY_MOTION
    before = out[jolt_at - 1]
    assert during.hbo == before.hbo, "held flat at the last good value"
    assert proc.gate_fraction > 0.0, "and the screen can say how much was held"
    # the sticky cooldown outlasts the jolt itself
    assert out[jolt_at + int(1.0 * RATE)].quality == QUALITY_MOTION
    assert out[-1].quality == QUALITY_GOOD, "and it lets go afterwards"


def test_an_ordinarily_noisy_head_is_not_motion_at_any_sensor_scale():
    """The gate compared the accelerometer's wobble against a fixed 80 counts, and 80 counts is not a
    quantity anybody here can convert: the IMU's scaling was never
    published. On the bench headband the constant held **half of all frames** while somebody sat still,
    which both reports a fidgeting session for a calm one and throws away half the training data.

    So the comparison is to the person: this asserts a still-but-real head passes, and asserts it twice at
    scales a factor of forty apart, because scale-freeness is the whole point of the change."""
    rng = np.random.default_rng(7)

    def still(counts_sd: float) -> float:
        proc = FrameProcessor()
        for f in frames(120.0):
            noise = rng.normal(0.0, counts_sd, size=3)
            proc.process(Frame(t=f.t, l=f.l, r=f.r, p=f.p, temp_c=f.temp_c, gyro=f.gyro,
                               acc=(noise[0], noise[1], 1000.0 + noise[2])))
        return proc.gate_fraction

    assert still(1.0) == 0.0
    assert still(40.0) == 0.0, "the same stillness in bigger counts is the same stillness"


def test_the_multiple_is_of_the_wearers_own_stillness_and_the_hold_can_be_switched_off():
    """Two settings, one row on the screen: how much movement counts, and whether it holds at all.

    The hold is off-switchable because the person wearing it is the one who knows whether they were
    moving, and a gate that is wrong about that is unarguable from inside the screen. Both are live —
    read per frame rather than at construction — so the answer arrives on the next frame instead of
    costing a session and eleven seconds of settling to find out."""
    rng = np.random.default_rng(11)

    def run(sensitivity: float, hold: bool = True, shove: float = 300.0) -> float:
        proc = FrameProcessor(motion_sensitivity=sensitivity, motion_gate=hold)
        for i, f in enumerate(frames(120.0)):
            # Still throughout, apart from half a second of movement every twenty.
            sd = shove if i >= 600 and i % 600 < 15 else 3.0
            noise = rng.normal(0.0, sd, size=3)
            proc.process(Frame(t=f.t, l=f.l, r=f.r, p=f.p, temp_c=f.temp_c, gyro=f.gyro,
                               acc=(noise[0], noise[1], 1000.0 + noise[2])))
        return proc.gate_fraction

    caught = run(4.0)
    assert 0.0 < caught < 0.35, f"the shoves and their cooldowns, not the sitting still ({caught:.0%})"
    assert run(20.0) == caught, "a shove a hundred times the noise floor is movement at either setting"
    # A fidget rather than a shove — ten times the noise floor — is where the dial does its work.
    assert run(4.0, shove=30.0) > 0.0
    assert run(20.0, shove=30.0) == 0.0, "and a higher multiple lets it through"
    assert run(4.0, hold=False) == 0.0, "and off is off"


def test_switching_the_hold_off_mid_session_takes_effect_on_the_next_frame():
    """Not at the next session: rebuilding the processor to change this would restart the low-pass
    filters and the eleven-second settling clock, so the question 'was it the gate?' would cost the
    answer it was asked to get."""
    proc = FrameProcessor()
    jolt = dict(acc=(4000, -3000, 5000), start_t=1020.0)
    for f in frames(20.0):
        proc.process(f)
    during = proc.process(next(iter(frames(0.1, **jolt))))
    assert during.quality == QUALITY_MOTION
    proc.motion_gate = False
    after = proc.process(next(iter(frames(0.1, acc=(4000, -3000, 5000), start_t=1020.1))))
    assert after.quality == QUALITY_GOOD, "the same movement, no longer held"


def test_a_steady_tilt_is_not_motion():
    """Stddev of the magnitude, not the level: a tilted head is a DC offset, and gating on it would
    hold the screen flat for anyone who does not sit like a statue."""
    proc = FrameProcessor()
    out = [proc.process(f) for f in frames(20.0, acc=(700, 700, 700))]
    assert out[-1].quality == QUALITY_GOOD
    assert proc.gate_fraction == 0.0


def test_motion_frames_advance_the_clock_and_not_the_statistics():
    """The processor holds optics flat during a cooldown; two seconds of repeated constants entering
    a five-second stability window would read as sudden calm — variability trained by head-stillness.
    The signal layer skips them, and the time-based windows span the hole."""
    proc, sig = FrameProcessor(), SignalProcessor()
    state = None
    wobble = lambda t: 300.0 * np.sin(2 * np.pi * 0.2 * t)
    for f in frames(30.0, hbo_drift=wobble):
        state = sig.push(proc.process(f), proc.gate_fraction)
    steady_std = state.hbo_std_short
    proc2, sig2 = FrameProcessor(), SignalProcessor()
    stream = list(frames(30.0, hbo_drift=wobble)) + list(
        frames(2.0, acc=(4000, -3000, 5000), start_t=1030.0)) + list(
        frames(1.0, hbo_drift=lambda t: wobble(t + 33.0), start_t=1032.0))
    for f in stream:
        state = sig2.push(proc2.process(f), proc2.gate_fraction)
    assert state.hbo_std_short == pytest.approx(steady_std, rel=0.5), \
        "a motion hold must not read as a collapse in variability"


def test_the_metrics_wait_for_their_windows():
    proc, sig = FrameProcessor(), SignalProcessor()
    early = sig.push(proc.process(next(iter(frames(0.1)))), 0.0)
    for metric_id in ("hbo", "oxy_index", "hbo_variability", "hbo_trend"):
        assert metric_value(HEG_METRICS[metric_id], early) is None, metric_id
    state = None
    for f in frames(20.0):
        state = sig.push(proc.process(f), proc.gate_fraction)
    for metric_id, spec in HEG_METRICS.items():
        assert metric_value(spec, state) is not None, metric_id


def test_the_variability_metrics_default_to_decrease():
    assert HEG_METRICS["hbo_variability"].invert
    assert HEG_METRICS["oxy_variability"].invert
    assert HEG_METRICS["hbo_variability"].default_direction == "decrease"
    assert not HEG_METRICS["oxy_index"].invert


def test_the_simulator_reaches_good_quality_and_plausible_numbers():
    sim = SimulatedMendi(seed=3, realtime=False)
    proc, sig = FrameProcessor(), SignalProcessor()
    state = None
    for frame in itertools.islice(sim.frames(), int(30 * RATE)):
        state = sig.push(proc.process(frame), proc.gate_fraction)
    assert state.quality == QUALITY_GOOD
    assert state.samples_seen == int(30 * RATE)
    assert abs(state.smoothed_hbo) < 5000, "the projection cancels most of the common mode"
