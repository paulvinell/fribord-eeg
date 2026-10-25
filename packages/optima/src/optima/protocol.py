"""Wire protocol: framing, opcodes, replies and notifications.

No knowledge of device configuration or transport.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


# Serial parameters, from FT_SetBaudRate/FT_SetDataCharacteristics/FT_SetFlowControl
# in NeurobitServ.exe. Hardware flow control is not optional.
BAUD = 921600


FLOW_RTSCTS = True


# Message classes, from the top two bits of the code byte.
CLASS_CMD = 0x00


CLASS_REP = 0x80


CLASS_NOT = 0xC0


CMD = {
    "GET_DEV_INFO": 0x00,
    "GET_FIRM_INFO": 0x01,
    "GET_DEV_STATE": 0x03,
    "SET_NOTIF": 0x04,
    "SET_RTC": 0x05,
    "READ_CFG": 0x06,
    "WRITE_CFG": 0x07,
    "START_MES": 0x08,
    "TURN_OFF": 0x09,
    "STOP_MES": 0x0A,
    "START_TEST": 0x0B,
    "STOP_TEST": 0x0C,
    "RESTART": 0x0D,
    "SET_GEN": 0x0E,
    "CALIB": 0x0F,
}


CMD_NAME = {v: k for k, v in CMD.items()}


NOT_EVENT = 0xC0


NOT_DATA = 0xC1


NOT_EXT_DATA = 0xC2


NOT_NAME = {NOT_EVENT: "NOT_EVENT", NOT_DATA: "NOT_DATA", NOT_EXT_DATA: "NOT_EXT_DATA"}


# Reply status byte, paraphrasing the driver's own error strings.
STATUS = {
    0x00: "ok",
    0x01: "unknown or unimplemented command",
    0x02: "invalid parameter(s)",
    0x03: "not valid in the current device state",
    0x04: "device resources unavailable or insufficient",
    0x05: "invalid data format",
}


# Minimum reply length the driver accepts (code byte included), which is a useful
# sanity check on anything we parse.
REP_MINLEN = {
    0x80: 69, 0x81: 32, 0x83: 3, 0x84: 2, 0x85: 2, 0x86: 22, 0x87: 2,
    0x88: 2, 0x89: 2, 0x8A: 2, 0x8B: 2, 0x8C: 2, 0x8D: 2, 0x8E: 2,
}


MAX_PAYLOAD = 0xFE  # the driver's own send path clamps here


# CMD_SET_NOTIF values. The driver only ever sends these two. 0x40 is sent once after
# WRITE_CFG to enable notifications for a session (both measurement and test); 0xE0 is sent
# on every stop, error and teardown path, and transitions the driver to APP_CLEAR_NOTIF /
# APP_RESET_NOTIF. Treat them as opaque enable/disable values: the bit semantics are not
# established, only what the driver does with them.
NOTIF_ENABLE = 0x40


NOTIF_CLEAR = 0xE0


# NOT_EVENT payload[0] codes, from the dispatcher at 0x10006639's caller.
EVENT_BATTERY = 0x40      # payload[1] = level, reported to the app as indication type 1


EVENT_LINK_ERROR = 0x60


EVENT_STOPPED = 0x61      # device ended the measurement by itself


EVENT_HARDWARE = 0x80     # "Hardware problem has been detected in the device"


EVENT_NAME = {
    EVENT_BATTERY: "battery",
    EVENT_LINK_ERROR: "link error",
    EVENT_STOPPED: "measurement stopped in device",
    EVENT_HARDWARE: "hardware problem",
}


# Signal state, the 2-bit per-entity field in every NOT_DATA frame. Value 2 is confirmed on
# hardware (every entity reads 2 with nothing attached); 1 and 3 are mapped from the manual's
# indicator colours (green = correct, orange = no signal, red = signal exceeds range) and are
# *unverified* individually.
SIGNAL_STATE = {0: "inactive", 1: "ok", 2: "no signal", 3: "over range"}


# The device powers itself off after 5 minutes idle with no host connection, to save batteries
# (vendor manual). It does not need a keepalive while streaming — 90 s of continuous capture
# showed zero drops — but an open port left idle will eventually find the device gone.
IDLE_POWEROFF_SECONDS = 300


# Device state, from REP_GET_DEV_STATE. Recovered from the per-APP-state expectation table at
# 0x10022892: idle for setup and teardown, 2 during the impedance test, 5 while measuring.
DEV_STATE = {0: "off", 1: "idle", 2: "testing", 5: "measuring"}


class ProtocolError(Exception):
    pass


class DeviceError(Exception):
    def __init__(self, code: int, status: int):
        self.code, self.status = code, status
        name = CMD_NAME.get(code & 0x3F, f"0x{code:02x}")
        super().__init__(f"{name}: {STATUS.get(status, f'error 0x{status:02x}')}")


@dataclass
class Message:
    code: int
    payload: bytes

    @property
    def is_reply(self) -> bool:
        return self.code & 0xC0 == CLASS_REP

    @property
    def is_notification(self) -> bool:
        return self.code & 0xC0 == CLASS_NOT

    @property
    def name(self) -> str:
        if self.is_notification:
            return NOT_NAME.get(self.code, f"NOT_0x{self.code:02x}")
        base = CMD_NAME.get(self.code & 0x3F, f"0x{self.code & 0x3F:02x}")
        return ("REP_" if self.is_reply else "CMD_") + base

    def __repr__(self) -> str:
        return f"<{self.name} {self.payload.hex(' ')}>"


def encode_varint(n: int) -> bytes:
    """Length prefix: 7 bits per byte, low group first, bit 7 = continuation."""
    if n < 0:
        raise ValueError(n)
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode(code: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload {len(payload)} exceeds {MAX_PAYLOAD}")
    body = bytes([code]) + payload
    return encode_varint(len(body)) + body
