"""Serial transport and the device session."""

from __future__ import annotations

import glob
import time

import serial
from serial.tools import list_ports

from .protocol import BAUD, CLASS_REP, CMD, CMD_NAME, DEV_STATE, DeviceError, FLOW_RTSCTS, Message, NOTIF_ENABLE, ProtocolError, REP_MINLEN, encode


def find_port() -> str:
    """Locate the amplifier's serial port.

    It is a plain FTDI USB-serial bridge, so the glob below is what identifies it, and for a long time
    that was the whole function. It stopped being enough once the HeartMath pulse sensor's cable arrived
    in the same program: that is also an FTDI bridge, it also appears as /dev/cu.usbserial-*, and taking
    the first match alphabetically would sometimes hand the EEG side a heart-rate monitor.

    The device does say who it is — its USB product string reads "Neurobit Optima*" — so a port that
    names itself wins over one that does not. When nothing names itself, this is the old behaviour, and
    `--port` remains the answer for anything ambiguous.
    """
    named = [p.device for p in list_ports.comports()
             if "neurobit" in " ".join(str(x) for x in (p.description, p.manufacturer, p.product)
                                       if x).lower()]
    if named:
        return sorted(named)[0]
    ports = sorted(glob.glob("/dev/cu.usbserial-*")) + sorted(glob.glob("/dev/cu.usbmodem*"))
    if not ports:
        raise RuntimeError(
            "no FTDI serial port found — is the Optima plugged in and switched on?"
        )
    return ports[0]


class Optima:
    def __init__(self, port: str | None = None, timeout: float = 2.0, log: bool = False):
        self.port = port or find_port()
        self.log = log
        self.ser = serial.Serial(
            self.port,
            baudrate=BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            rtscts=FLOW_RTSCTS,
            timeout=timeout,
        )
        self.events: list[Message] = []

    def close(self) -> None:
        self.ser.close()

    def __enter__(self) -> "Optima":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- framing -------------------------------------------------------------

    def _read_exactly(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise TimeoutError(f"wanted {n} bytes, got {len(buf)}")
            buf += chunk
        return buf

    def read_message(self) -> Message:
        length, shift = 0, 0
        while True:
            b = self._read_exactly(1)[0]
            length |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
            if shift > 21:
                raise ProtocolError("length field runaway — stream out of sync")
        if length < 1:
            raise ProtocolError("zero-length message")
        body = self._read_exactly(length)
        msg = Message(body[0], body[1:])
        if self.log:
            direction = ">" if msg.code & 0x80 else "<"
            print(f"{direction} {msg.name} {msg.payload.hex(' ')}")
        return msg

    def send(self, code: int, payload: bytes = b"") -> None:
        frame = encode(code, payload)
        if self.log:
            print(f"< {CMD_NAME.get(code, hex(code))} {frame.hex(' ')}")
        self.ser.write(frame)
        self.ser.flush()

    def request(self, code: int, payload: bytes = b"", retries: int = 0) -> bytes:
        """Send a command and return its reply payload, minus the status byte.

        Notifications arriving while we wait are collected into self.events rather than
        discarded — the device can emit them at any time.
        """
        want = code | CLASS_REP
        for attempt in range(retries + 1):
            self.send(code, payload)
            deadline = time.monotonic() + (self.ser.timeout or 2.0) * 2
            while time.monotonic() < deadline:
                try:
                    msg = self.read_message()
                except TimeoutError:
                    break
                if msg.is_notification:
                    self.events.append(msg)
                    continue
                if msg.code != want:
                    raise ProtocolError(f"expected {hex(want)}, got {msg!r}")
                if not msg.payload:
                    raise ProtocolError(f"{msg.name} has no status byte")
                if msg.payload[0] != 0:
                    raise DeviceError(code, msg.payload[0])
                minlen = REP_MINLEN.get(want)
                if minlen and len(msg.payload) + 1 < minlen:
                    raise ProtocolError(
                        f"{msg.name} is {len(msg.payload) + 1} bytes, driver requires {minlen}"
                    )
                return msg.payload[1:]
            if attempt < retries:
                self.ser.reset_input_buffer()
        raise TimeoutError(f"no reply to {CMD_NAME.get(code, hex(code))}")

    # --- commands ------------------------------------------------------------

    def device_info(self) -> bytes:
        """68+ bytes: model, serial number, capabilities. Field layout unknown."""
        return self.request(CMD["GET_DEV_INFO"])

    def firmware_info(self) -> bytes:
        return self.request(CMD["GET_FIRM_INFO"])

    def device_state(self) -> tuple[int, str]:
        """Device state: 0 off, 1 idle, 2 testing, 5 measuring.

        Note the vendor driver never sends GET_DEV_STATE, so firmware support is unproven —
        but it does handle the reply and check it against these values.
        """
        data = self.request(CMD["GET_DEV_STATE"])
        return data[0], DEV_STATE.get(data[0], f"unknown 0x{data[0]:02x}")

    def read_config(self) -> bytes:
        """The packed config blob, at least 20 bytes — the driver's table gives a minimum,
        not a fixed size, so the real blob is probably longer and may vary with channel
        count. Bit layout unknown; diff it by changing one setting at a time in the
        Windows driver."""
        return self.request(CMD["READ_CFG"])

    def write_config(self, blob: bytes) -> None:
        if len(blob) < 20:
            raise ValueError(f"config blob is at least 20 bytes, got {len(blob)}")
        self.request(CMD["WRITE_CFG"], blob)

    def set_notifications(self, mask: int = NOTIF_ENABLE) -> None:
        """Enable or clear notifications. Defaults to NOTIF_ENABLE, which is what the driver
        sends after WRITE_CFG; send NOTIF_CLEAR when stopping."""
        self.request(CMD["SET_NOTIF"], bytes([mask]))

    def start_measurement(self) -> None:
        self.request(CMD["START_MES"])

    def stop_measurement(self) -> None:
        self.request(CMD["STOP_MES"])

    def stream(self):
        """Yield notifications until interrupted. Sample decoding is not implemented —
        NOT_DATA payloads come out raw, which is what you want while working out the
        layout."""
        while True:
            msg = self.read_message()
            if msg.is_notification:
                yield msg
