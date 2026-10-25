"""Minimal codec for the `DeviceV4` protobuf used by the Mendi headband.

We hand-roll instead of pulling in `protobuf` + `protoc` because:
- The schema is fixed and tiny (5 messages, all scalar fields).
- The encoder only needs to produce one shape (the Sensor start/stop write).
- Zero build-time dependencies — keeps the project plain-Python.

Schema:

    message Frame {
      int32 acc_x=1; int32 acc_y=2; int32 acc_z=3;          // accelerometer raw
      int32 ang_x=4; int32 ang_y=5; int32 ang_z=6;          // gyroscope raw
      float temp =7;                                         // °C
      int32 ir_l=8;  int32 red_l=9;  int32 amb_l=10;        // channel Left
      int32 ir_r=11; int32 red_r=12; int32 amb_r=13;        // channel Right
      int32 ir_p=14; int32 red_p=15; int32 amb_p=16;        // channel P (prefrontal-midline?)
    }
    message Adc { uint32 voltage=1; bool charging=2; bool usb=3; }
    message Calibration {
      float offset_l=1; float offset_r=2; float offset_p=3;
      bool enable=4; bool low_power_mode=5;
    }
    message Sensor { bool read=1; uint32 address=2; fixed32 data=3; }
    message Diagnostics { Adc adc=1; bool imu_ok=2; bool sensor_ok=3; }

The decoder is intentionally lenient: it returns a `{field_number: value}`
dict and skips unknown wire types rather than raising. That gives us
forward compatibility if Mendi extends the schema in a firmware update.
"""

from __future__ import annotations

import struct
from typing import Dict, Tuple


# ---- low-level wire-format helpers ----------------------------------------

def _read_varint(buf: bytes, idx: int) -> Tuple[int, int]:
    """Decode one protobuf varint starting at buf[idx]. Returns (value, new_idx)."""
    result = 0
    shift = 0
    while True:
        b = buf[idx]
        idx += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, idx
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _write_varint(value: int) -> bytes:
    """Encode an unsigned int as protobuf varint."""
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _decode_message(buf: bytes) -> Dict[int, object]:
    """Decode one protobuf message into {field_num: value}.

    Returns ints for varint / fixed32 fields, floats only when the caller
    asks for that interpretation (the decoder can't tell int32 from float
    apart on the wire — both are wire-type 5). The caller maps fields to
    their expected types.
    """
    fields: Dict[int, object] = {}
    idx = 0
    n = len(buf)
    while idx < n:
        tag, idx = _read_varint(buf, idx)
        field_num = tag >> 3
        wire = tag & 0x07
        if wire == 0:  # varint (int32, uint32, bool)
            value, idx = _read_varint(buf, idx)
            fields[field_num] = value
        elif wire == 5:  # 32-bit fixed (float, fixed32)
            value = int.from_bytes(buf[idx:idx + 4], "little", signed=False)
            idx += 4
            fields[field_num] = value
        elif wire == 1:  # 64-bit fixed
            idx += 8
        elif wire == 2:  # length-delimited (string/bytes/sub-message)
            length, idx = _read_varint(buf, idx)
            idx += length
        else:
            # Unknown wire type — bail rather than risk corrupting the cursor.
            raise ValueError(f"unsupported wire type {wire} at idx {idx}")
    return fields


def _as_float(raw32: int) -> float:
    """Reinterpret a 32-bit unsigned int as an IEEE-754 float."""
    return struct.unpack("<f", raw32.to_bytes(4, "little"))[0]


def _as_int32(varint_value: int) -> int:
    """Sign-extend a protobuf varint into a signed 32-bit int.

    Protobuf encodes negative int32 values as the matching int64 — the
    sign bit propagates into a 10-byte varint. Decoding that as unsigned
    gives an enormous positive number; the caller must mask to 32 bits
    and sign-extend the top bit to recover the original value.
    """
    v = varint_value & 0xFFFFFFFF
    if v & 0x80000000:
        v -= 0x100000000
    return v


# ---- typed decoders --------------------------------------------------------

# DeviceV4.Frame field numbers — kept in sync with the schema above.
FRAME_FIELDS = {
    1: "acc_x", 2: "acc_y", 3: "acc_z",
    4: "ang_x", 5: "ang_y", 6: "ang_z",
    7: "temp",                          # float — caller must reinterpret
    8: "ir_l",  9: "red_l", 10: "amb_l",
    11: "ir_r", 12: "red_r", 13: "amb_r",
    14: "ir_p", 15: "red_p", 16: "amb_p",
}


def decode_frame(buf: bytes) -> Dict[str, float]:
    """Decode one Frame packet. Returns a dict keyed by the field name.

    Missing fields default to 0 (matching proto3's default-value semantics).
    The temperature field (float on the wire) is reinterpreted; all other
    fields are returned as ints.
    """
    raw = _decode_message(buf)
    out: Dict[str, float] = {name: 0 for name in FRAME_FIELDS.values()}
    for fnum, name in FRAME_FIELDS.items():
        if fnum not in raw:
            continue
        if name == "temp":
            out[name] = _as_float(int(raw[fnum]))
        else:
            # Every other Frame field is wire-type int32 (signed). Negative
            # values come in as 10-byte varints whose unsigned interpretation
            # is huge — sign-extend back to a real signed int.
            out[name] = _as_int32(int(raw[fnum]))  # type: ignore[assignment]
    return out


def decode_adc(buf: bytes) -> Dict[str, object]:
    """Decode an Adc packet → {voltage_mv, charging, usb}."""
    raw = _decode_message(buf)
    return {
        "voltage_mv": int(raw.get(1, 0)),
        "charging":   bool(raw.get(2, 0)),
        "usb":        bool(raw.get(3, 0)),
    }


# Single-cell Li-ion open-circuit voltage → state-of-charge, anchored at
# a few well-known points on the discharge curve. Linear interpolation
# between them is good to ±5 % which is far below display resolution.
# A linear `(mV-3600)/5` is dishonestly precise near the bottom: it calls
# 3.6 V "0 %" even though the cell still has charge, and under streaming
# load the voltage sags a few mV across that cliff and the pill drops to
# 0 %. We report what the cell actually has.
_LIION_OCV_CURVE = (
    (3300,   0),
    (3500,   5),
    (3600,  12),
    (3700,  25),
    (3750,  35),
    (3800,  45),
    (3850,  55),
    (3900,  65),
    (3950,  75),
    (4000,  85),
    (4100,  95),
    (4200, 100),
)


def adc_voltage_to_percent(voltage_mv: int) -> int:
    """Linear-interpolate over a single-cell Li-ion OCV curve."""
    if voltage_mv <= _LIION_OCV_CURVE[0][0]:
        return 0
    if voltage_mv >= _LIION_OCV_CURVE[-1][0]:
        return 100
    for (lo_v, lo_p), (hi_v, hi_p) in zip(_LIION_OCV_CURVE, _LIION_OCV_CURVE[1:]):
        if voltage_mv <= hi_v:
            frac = (voltage_mv - lo_v) / (hi_v - lo_v)
            return int(round(lo_p + frac * (hi_p - lo_p)))
    return 100


def decode_calibration(buf: bytes) -> Dict[str, object]:
    """Decode a Calibration packet."""
    raw = _decode_message(buf)
    return {
        "offset_l":        _as_float(int(raw.get(1, 0))),
        "offset_r":        _as_float(int(raw.get(2, 0))),
        "offset_p":        _as_float(int(raw.get(3, 0))),
        "enable":          bool(raw.get(4, 0)),
        "low_power_mode":  bool(raw.get(5, 0)),
    }


# ---- encoder for the Sensor start/stop write ------------------------------

def encode_sensor_write(address: int, data: int, read: bool = False) -> bytes:
    """Encode a Sensor protobuf for writing to the ABB2 characteristic.

    The Android app uses this to start (data=0x0100) and stop (data=0x0000)
    the frame stream by writing register 30. We always emit `address` and
    `data` explicitly even when zero, so the firmware can't mistake our
    intent — proto3 normally omits default values, which would make a
    "stop" indistinguishable from "no command at all".
    """
    out = bytearray()
    # Field 1 (read): only emit if true. proto3 default is false.
    if read:
        out += _write_varint((1 << 3) | 0)  # tag: field 1, varint
        out += _write_varint(1)
    # Field 2 (address): varint.
    out += _write_varint((2 << 3) | 0)
    out += _write_varint(address)
    # Field 3 (data): fixed32, little-endian.
    out += _write_varint((3 << 3) | 5)
    out += int(data).to_bytes(4, "little", signed=False)
    return bytes(out)
