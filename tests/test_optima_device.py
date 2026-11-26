"""The amplifier's serial transport: framing on the wire, and a command matched to its reply.

Against a fake port that hands back prepared bytes, because what is under test is how the bytes are read.
"""

from __future__ import annotations

import optima
import pytest
from optima import device as transport
from optima.protocol import CLASS_REP, REP_MINLEN, encode

from harness import ListedPort


class FakePort:
    """Serves prepared bytes to `read` and keeps what was written."""

    def __init__(self, incoming: bytes):
        self.incoming = bytearray(incoming)
        self.written = bytearray()
        self.timeout = 0.05

    def read(self, n: int) -> bytes:
        out = bytes(self.incoming[:n])
        del self.incoming[:n]
        return out

    def write(self, data: bytes) -> None:
        self.written += data

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        pass


def amplifier(monkeypatch, incoming: bytes) -> optima.Optima:
    port = FakePort(incoming)
    monkeypatch.setattr(transport.serial, "Serial", lambda *a, **k: port)
    return optima.Optima(port="/dev/fake")


def test_a_message_with_no_payload_is_read_as_one(monkeypatch):
    """A frame that holds only a code is a whole message with an empty payload, not a framing error."""
    msg = amplifier(monkeypatch, encode(optima.NOT_EVENT)).read_message()
    assert (msg.code, msg.payload) == (optima.NOT_EVENT, b"")


def test_a_reply_of_the_minimum_length_is_accepted(monkeypatch):
    """A reply exactly as long as the device must send is a good reply, and a notification that arrives
    before it is kept, not taken for the reply."""
    code = optima.CMD["GET_DEV_INFO"]
    want = code | CLASS_REP
    body = bytes([0]) + bytes(range(REP_MINLEN[want] - 2))      # the status byte, then the payload
    event = encode(optima.NOT_EVENT, bytes([optima.EVENT_BATTERY, 3]))
    dev = amplifier(monkeypatch, event + encode(want, body))
    assert dev.device_info() == body[1:]
    assert [m.code for m in dev.events] == [optima.NOT_EVENT]


def test_a_reply_shorter_than_the_minimum_is_refused(monkeypatch):
    """A reply shorter than the device must send is a framing fault, not identity bytes."""
    code = optima.CMD["GET_DEV_INFO"]
    want = code | CLASS_REP
    dev = amplifier(monkeypatch, encode(want, bytes([0]) + bytes(REP_MINLEN[want] - 3)))
    with pytest.raises(optima.ProtocolError, match="driver requires"):
        dev.device_info()


def test_the_amplifier_prefers_the_port_that_says_it_is_one(monkeypatch):
    """The other half of the same collision: the amplifier used to take the first
    /dev/cu.usbserial-* alphabetically, which is the pulse sensor's cable often enough to matter."""
    import optima.device as device

    monkeypatch.setattr(device.list_ports, "comports", lambda: [
        ListedPort("/dev/cu.usbserial-AAAA", 0x0403, "FT230X Basic UART"),
        ListedPort("/dev/cu.usbserial-FT000001", 0x0403, "Neurobit Optima+ 4 USB"),
    ])
    assert device.find_port() == "/dev/cu.usbserial-FT000001"
