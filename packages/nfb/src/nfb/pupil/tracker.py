"""The Tobii Eye Tracker 5 over USB, reduced to what pupil training needs.

The tracker measures each pupil on the device, in millimetres, and sends it in its gaze stream about 33
times a second. The wire protocol was recovered so this software can talk to the device. Layers, each
testable without it: the wire encoding (pure functions), `Reassembler` (transfers to frames),
`decode_gaze`, `TrackerSession` (the connect sequence over any `Link`), and `TrackerPupil`, a
`PupilSource` the pupil screen takes in place of the simulator.

**Authentication.** The device says what authentication it needs; for the gaze stream it has been seen
to need none, and that is the only answer accepted. This module holds no key. **The display area.** The
tracker forgets the screen at every session close and tracks nothing until told again, so every
connection sends a screen size; a rough one is enough for pupil data.

Not yet run against a tracker. Open: does the pupil value answer light, how noisy is it over a second,
and does it arrive without a calibration made by the vendor's software.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import struct
import time
from dataclasses import dataclass
from typing import Iterator, Protocol

from .source import PupilSample

VENDOR_ID = 0x2104
PRODUCT_ID = 0x0313
INTERFACE = 0
EP_OUT = 0x05
EP_IN = 0x83
CTRL_TYPE = 0x41            # vendor, host to device, interface recipient
CTRL_OPEN = 0x41
CTRL_CLOSE = 0x42

DIR_OUT, DIR_IN = 0x00, 0x01
ENVELOPE = 8
HEADER = 24
MAGIC_REQUEST, MAGIC_RESPONSE, MAGIC_NOTIFY = 0x51, 0x52, 0x53

OP_HELLO = 0x3E8
OP_QUERY_AUTH = 0x640
OP_OPEN_AUTH = 0x76C
OP_ANSWER_AUTH = 0x776
OP_CLOSE_AUTH = 0x77B
OP_SUBSCRIBE = 0x4C4
OP_SET_DISPLAY = 0x5A0
STREAM_GAZE = 0x500

T_ENUM, T_U32, T_FIXED16, T_Q42, T_TAG, T_S64 = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06
TAG_POINT3D = 0x00031F41
TAG_COLUMN = 0x00020BB9
ROW_TYPE = 0x0BB8

COL_TIME = 0x01
COL_ORIGIN = {"left": 0x02, "right": 0x08}
COL_TRACKBOX = {"left": 0x03, "right": 0x09}     # eye position in the trackbox, 0..1 each axis
COL_PUPIL = {"left": 0x06, "right": 0x0C}
COL_VALID = {"left": 0x07, "right": 0x0D}
COL_GAZE_2D = 0x1C          # combined, filtered gaze on the screen, 0..1; (-1, -1) when invalid
COL_GAZE_2D_EYE = {"left": 0x05, "right": 0x0B}     # per-eye gaze on the screen, 0..1
COL_GAZE_2D_RAW = 0x20      # combined, unfiltered screen gaze — present when the filtered one is not
COL_DIR = {"left": 0x04, "right": 0x0A}             # per-eye gaze direction (yaw, pitch), pre-screen
VALID_TRACKED = 0
GAZE_PROBE = {0x1C: "combined", 0x05: "left-2d", 0x0B: "right-2d", 0x20: "unfiltered",
              0x04: "left-dir", 0x0A: "right-dir"}
"""The gaze-bearing columns, for the on-screen probe when no screen gaze is valid — so what a real
tracker actually sends can be read off rather than guessed."""

HELLO_FEATURES = bytes.fromhex(
    "00000009" "00010000" "00010001" "00010002" "00010003"
    "00010004" "00010005" "00010006" "00010007" "00010008")
"""The capability list the hello request carries: a count, then nine version pairs. Sent as it is."""
HELLO_TYPE = 0x17
SUBSCRIBE_TRAILER_TYPE = 0x17

DISPLAY_TRAILER_TAG = 0x00010100
DISPLAY_TRAILER_VALUE = 0x3039
SCREEN_LIFT_MM = 10.0
"""The screen's bottom edge above the tracker. The tracker sits on the bottom bezel."""

WRITE_TIMEOUT_MS = 2000
REPLY_TIMEOUT_S = 2.0
READ_TIMEOUT_MS = 1000
DRAIN_TIMEOUT_MS = 30   # one read while draining gaze during calibration
DRAIN_BUDGET_S = 0.12   # keep reading up to here until a whole gaze frame assembles
READ_SIZE = 65536
CONTROL_TIMEOUT_MS = 1000
CHUNK = 8192    # the tracker's frame-size limit; a larger request goes as several transfers
FRAME_CAP = 2 * 1024 * 1024


class TrackerError(RuntimeError):
    """A fault the operator can act on. The message is shown on the device screen as it is."""


# --- wire -------------------------------------------------------------------------------------

def field(kind: int, body: bytes) -> bytes:
    return struct.pack(">BI", kind, len(body)) + body


def u32(kind: int, value: int) -> bytes:
    return field(kind, struct.pack(">I", value))


def q42(value: float) -> bytes:
    return field(T_Q42, struct.pack(">q", round(value * 2 ** 42)))


def point3d(x: float, y: float, z: float) -> bytes:
    return u32(T_TAG, TAG_POINT3D) + q42(x) + q42(y) + q42(z)


def request(seq: int, op: int, payload: bytes) -> bytes:
    """One request, in its outbound envelope. The envelope length counts the bytes after it."""
    body = struct.pack(">IIIIII", MAGIC_REQUEST, seq, 0, op, 0, len(payload)) + payload
    return bytes([DIR_OUT, 0, 0, 0]) + struct.pack("<I", len(body)) + body


def hello_payload() -> bytes:
    return b"\x00\x00" + field(HELLO_TYPE, HELLO_FEATURES)


def subscribe_payload(stream: int = STREAM_GAZE) -> bytes:
    return b"\x00\x00" + u32(T_U32, stream) + u32(SUBSCRIBE_TRAILER_TYPE, 0)


def open_auth_payload(kind: int) -> bytes:
    return b"\x00\x00" + u32(T_U32, kind) + b"\x00"


KEY_ENV = "NFB_TOBII_AUTH_KEY"
"""Where the authentication key comes from: this environment variable, as hex. It is the operator's to
supply, for a tracker they are entitled to unlock. It is never stored in this tree — the committed code
carries the exchange, never the key, so a checkout holds no material taken from another program."""


def auth_key() -> bytes | None:
    """The key for the challenge, or None when the operator has set none. Hex, or plain text as a
    fallback. A file path can stand in through the same variable, so the key need not sit in the shell
    history."""
    raw = os.environ.get(KEY_ENV)
    if not raw:
        return None
    candidate = raw.strip()
    path = os.path.expanduser(candidate)
    if os.path.sep in candidate and os.path.exists(path):
        with open(path) as fh:
            candidate = fh.read().strip()
    try:
        return bytes.fromhex(candidate)
    except ValueError:
        return candidate.encode()


def answer_auth_payload(realm_id: int, field: int, digest: bytes) -> bytes:
    """The reply to the tracker's challenge: two identifiers it sent, then the digest, raw."""
    return b"\x00\x00" + u32(T_U32, realm_id) + u32(T_U32, field) + digest


def close_auth_payload(realm_id: int) -> bytes:
    return b"\x00\x00" + u32(T_U32, realm_id)


def answer_challenge(key: bytes, challenge: bytes) -> bytes:
    """The 16-byte answer to a challenge: a keyed MD5 digest over the bytes the tracker sent. MD5 is the
    algorithm the tracker checks against; it is a fixed property of the device, not a security choice
    here."""
    return hmac.new(key, challenge, hashlib.md5).digest()


def resp_fields(payload: bytes) -> Iterator[tuple[int, bytes]]:
    """Fields of a handshake *response*, which are framed more loosely than a request: a one-byte type, a
    pad byte, a two-byte length, then the body. The two-byte prefix is skipped. Stops at a short buffer."""
    pos = 2
    while pos + 4 <= len(payload):
        kind = payload[pos]
        size = struct.unpack_from(">H", payload, pos + 2)[0]
        pos += 4
        if pos + size > len(payload):
            return
        yield kind, payload[pos:pos + size]
        pos += size


def resp_u32_at(payload: bytes, index: int) -> int | None:
    """The `index`-th four-byte response field, as an unsigned integer."""
    found = [b for _k, b in resp_fields(payload) if len(b) == 4]
    return struct.unpack(">I", found[index])[0] if index < len(found) else None


def resp_challenge(payload: bytes) -> bytes | None:
    """The first response field longer than four bytes — the challenge the tracker wants answered."""
    return next((b for _k, b in resp_fields(payload) if len(b) > 4), None)


def display_payload(width_mm: float, height_mm: float) -> bytes:
    """Three corners of the screen in the tracker's frame, in millimetres: top left, top right, bottom
    left. x is to the right, y is up, and the screen is upright above the tracker, centred on it."""
    half = width_mm / 2
    bottom, top = SCREEN_LIFT_MM, SCREEN_LIFT_MM + height_mm
    return (b"\x00\x00" + point3d(-half, top, 0.0) + point3d(half, top, 0.0)
            + point3d(-half, bottom, 0.0)
            + u32(T_TAG, DISPLAY_TRAILER_TAG) + u32(T_U32, DISPLAY_TRAILER_VALUE))


def fields(payload: bytes, start: int = 2) -> Iterator[tuple[int, bytes]]:
    """`(type, body)` for each field after the two-byte prefix. Stops at a field the buffer cannot hold."""
    pos = start
    while pos + 5 <= len(payload):
        kind, size = struct.unpack_from(">BI", payload, pos)
        pos += 5
        if pos + size > len(payload):
            return
        yield kind, payload[pos:pos + size]
        pos += size


def _value(kind: int, body: bytes):
    if kind in (T_U32, T_ENUM) and len(body) == 4:
        return struct.unpack(">I", body)[0]
    if kind == T_FIXED16 and len(body) == 4:
        return struct.unpack(">i", body)[0] / 65536.0
    if kind == T_Q42 and len(body) == 8:
        return struct.unpack(">q", body)[0] / 2 ** 42
    if kind == T_S64 and len(body) == 8:
        return struct.unpack(">q", body)[0]
    return body


def columns(payload: bytes) -> dict[int, object]:
    """A stream payload as `{column id: value}`.

    The payload is a row tag, then per column a column tag, the column id, and a value: one field, or a
    point tag followed by its coordinates. A point is returned as a tuple of floats. Reading is by
    structure, never by byte offset, so a firmware that adds or reorders columns is read the same way.
    """
    items = list(fields(payload))
    out: dict[int, object] = {}
    i = 0
    if items and items[0][0] == T_TAG and (struct.unpack(">I", items[0][1])[0] & 0xFFFF) == ROW_TYPE:
        i = 1
    while i + 2 < len(items):
        kind, body = items[i]
        if kind != T_TAG or struct.unpack(">I", body)[0] != TAG_COLUMN:
            i += 1
            continue
        _id_kind, id_body = items[i + 1]
        column = struct.unpack(">I", id_body)[0]
        kind, body = items[i + 2]
        if kind == T_TAG:
            count = struct.unpack(">I", body)[0] >> 16
            coords = items[i + 3:i + 3 + count]
            out[column] = tuple(_value(k, b) for k, b in coords)
            i += 3 + count
        else:
            out[column] = _value(kind, body)
            i += 3
    return out


@dataclass(frozen=True)
class Frame:
    magic: int
    seq: int
    op: int
    payload: bytes


class Reassembler:
    """USB transfers in, whole frames out.

    Every transfer starts with its own eight-byte envelope. A frame larger than one transfer continues in
    the next ones, each behind an envelope of its own, and the frame is complete when the payload length
    in its header is reached. An envelope's length is trusted only as far as the transfer goes.
    """

    def __init__(self):
        self._acc = bytearray()

    def push(self, data: bytes) -> list[Frame]:
        out: list[Frame] = []
        pos = 0
        while pos + ENVELOPE <= len(data):
            if data[pos] != DIR_IN:
                self._acc.clear()
                return out
            length = struct.unpack_from("<I", data, pos + 4)[0]
            end = min(len(data), pos + length) if length > ENVELOPE else len(data)
            self._acc += data[pos + ENVELOPE:end]
            pos = end
            while len(self._acc) >= HEADER:
                magic, seq, _flag, op, _res, plen = struct.unpack_from(">IIIIII", self._acc, 0)
                if magic not in (MAGIC_RESPONSE, MAGIC_NOTIFY) or plen > FRAME_CAP:
                    self._acc.clear()
                    break
                if len(self._acc) < HEADER + plen:
                    break
                out.append(Frame(magic, seq, op, bytes(self._acc[HEADER:HEADER + plen])))
                # Bytes after a complete frame belong to no frame we can find the start of: the next
                # frame starts behind an envelope, at a transfer boundary.
                self._acc.clear()
        if len(self._acc) > FRAME_CAP:
            self._acc.clear()
        return out


# --- gaze -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Eye:
    tracked: bool
    pupil_mm: float | None
    origin_mm: tuple[float, float, float] | None
    box: tuple[float, float] | None = None       # position in the trackbox, 0..1 x and y


@dataclass(frozen=True)
class Position:
    """Where the eyes sit relative to the tracker, for the positioning guide. Each eye is an (x, y) in the
    trackbox, 0..1, or None when that eye is not found; `distance_mm` is the mean distance to the sensor."""

    left: tuple[float, float] | None
    right: tuple[float, float] | None
    distance_mm: float | None

    @property
    def seen(self) -> bool:
        """Whether the tracker has the eyes — by a trackbox position, or at least a distance."""
        return self.left is not None or self.right is not None or self.distance_mm is not None


def _screen_point(value) -> tuple[float, float] | None:
    """A screen-gaze column as a point, or None for the (-1, -1) sentinel or a missing one."""
    return (tuple(value) if isinstance(value, tuple) and len(value) == 2
            and value[0] >= 0 and value[1] >= 0 else None)


@dataclass(frozen=True)
class GazeFrame:
    timestamp_us: int | None
    left: Eye
    right: Eye
    gaze_2d: tuple[float, float] | None     # where on the screen, 0..1; None when not valid
    fixation: tuple[float, float] | None    # a steady-gaze signal for calibration; see `decode_gaze`
    probe: dict                             # raw gaze-column values, for the on-screen diagnostic

    def distance_mm(self) -> float | None:
        """The mean distance of the tracked eyes from the tracker."""
        found = [math.dist((0, 0, 0), e.origin_mm) for e in (self.left, self.right)
                 if e.tracked and e.origin_mm is not None and any(e.origin_mm)]
        return sum(found) / len(found) if found else None


def decode_gaze(payload: bytes) -> GazeFrame:
    cols = columns(payload)

    def eye(side: str) -> Eye:
        validity = cols.get(COL_VALID[side])
        tracked = validity == VALID_TRACKED
        pupil = cols.get(COL_PUPIL[side])
        # The device sends −1 for a pupil it did not measure. Zero is not a pupil either.
        pupil_mm = float(pupil) if tracked and isinstance(pupil, float) and pupil > 0 else None
        origin = cols.get(COL_ORIGIN[side])
        origin_mm = tuple(origin) if isinstance(origin, tuple) and len(origin) == 3 else None
        box = cols.get(COL_TRACKBOX[side])
        box_xy = (box[0], box[1]) if tracked and isinstance(box, tuple) and len(box) == 3 else None
        return Eye(tracked, pupil_mm, origin_mm, box_xy)

    stamp = cols.get(COL_TIME)
    left, right = eye("left"), eye("right")
    probe = {GAZE_PROBE[c]: cols[c] for c in GAZE_PROBE if c in cols}

    # With no eye tracked the device zeroes every eye column; (0, 0) is not a look at the corner, so no
    # gaze is read unless an eye is tracked.
    if not (left.tracked or right.tracked):
        return GazeFrame(stamp if isinstance(stamp, int) else None, left, right, None, None, probe)

    gaze_2d = _screen_point(cols.get(COL_GAZE_2D))
    # Fixation falls back across gaze columns; units are immaterial (capture pairs the dot with the
    # device's own raw eye data).
    per_eye = [_screen_point(cols.get(COL_GAZE_2D_EYE[s])) for s in ("left", "right")]
    per_eye = [p for p in per_eye if p is not None]
    directions = [cols.get(COL_DIR[s]) for s in ("left", "right")]
    directions = [d for d in directions if isinstance(d, tuple) and len(d) == 3 and any(d)]
    fixation = gaze_2d or _screen_point(cols.get(COL_GAZE_2D_RAW))
    if fixation is None and per_eye:
        fixation = (sum(p[0] for p in per_eye) / len(per_eye),
                    sum(p[1] for p in per_eye) / len(per_eye))
    if fixation is None and directions:
        fixation = (sum(d[0] for d in directions) / len(directions),
                    sum(d[1] for d in directions) / len(directions))

    return GazeFrame(stamp if isinstance(stamp, int) else None, left, right, gaze_2d, fixation, probe)


# --- the session ------------------------------------------------------------------------------

class Link(Protocol):
    def write(self, data: bytes) -> None: ...
    def read(self, timeout_ms: int) -> bytes | None: ...
    def close(self) -> None: ...


def _usb():
    """`usb.core` and a libusb backend, or the reason there is none. libusb comes bundled with the
    package that ships it, so nothing has to be installed on the machine."""
    try:
        import usb.core
    except Exception as exc:
        raise TrackerError(f"USB support is unavailable ({exc}) — check `uv sync`") from exc
    backend = None
    try:
        import libusb_package
        backend = libusb_package.get_libusb1_backend()
    except Exception:
        backend = None
    return usb.core, backend


class UsbLink:
    """Interface 0 of the tracker, its two bulk endpoints and the session control requests."""

    def __init__(self):
        core, backend = _usb()
        self._core = core
        device = core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID, backend=backend)
        if device is None:
            raise TrackerError("no eye tracker on USB — connect it, and close any other program that "
                               "uses it")
        self.device = device
        try:
            try:
                if device.is_kernel_driver_active(INTERFACE):
                    device.detach_kernel_driver(INTERFACE)
            except (NotImplementedError, core.USBError):
                pass                # macOS has no driver on a vendor interface to detach
            try:
                device.set_configuration()
            except core.USBError:
                pass                # already configured, which is the usual case
            import usb.util
            usb.util.claim_interface(device, INTERFACE)
            device.ctrl_transfer(CTRL_TYPE, CTRL_OPEN, 0, INTERFACE, None, timeout=CONTROL_TIMEOUT_MS)
        except core.USBError as exc:
            raise TrackerError(f"the eye tracker refused the connection ({exc}) — another program can "
                               f"hold it; close it and try again") from exc
        self.serial = None
        try:
            self.serial = device.serial_number
        except Exception:
            pass

    def write(self, data: bytes) -> None:
        try:
            self.device.write(EP_OUT, data, timeout=WRITE_TIMEOUT_MS)
        except self._core.USBError as exc:
            raise TrackerError(f"the eye tracker stopped answering ({exc})") from exc

    def read(self, timeout_ms: int) -> bytes | None:
        try:
            data = self.device.read(EP_IN, READ_SIZE, timeout=timeout_ms)
        except self._core.USBTimeoutError:
            return None
        except self._core.USBError as exc:
            if "timeout" in str(exc).lower():
                return None
            raise TrackerError(f"the eye tracker stopped answering ({exc})") from exc
        return bytes(data) if len(data) else None

    def close(self) -> None:
        try:
            self.device.ctrl_transfer(CTRL_TYPE, CTRL_CLOSE, 0, INTERFACE, None,
                                      timeout=CONTROL_TIMEOUT_MS)
        except Exception:
            pass
        try:
            import usb.util
            usb.util.release_interface(self.device, INTERFACE)
            usb.util.dispose_resources(self.device)
        except Exception:
            pass


class TrackerSession:
    """The connect sequence and the gaze stream, over any `Link`.

    Requests are matched to their responses by operation and sequence number together. A gaze frame that
    arrives while a request waits is not a response and is skipped.
    """

    def __init__(self, link: Link, screen_mm: tuple[float, float] = (600.0, 340.0),
                 serial: str | None = None):
        self.link = link
        self.screen_mm = screen_mm
        self.serial = serial if serial is not None else getattr(link, "serial", None)
        self._seq = 0
        self._frames = Reassembler()
        self._pending: list[Frame] = []
        self._replies: dict[int, bytes] = {}

    @property
    def handshake_hex(self) -> dict[str, str]:
        """The raw bytes of the replies this program had to guess the shape of, as hex, named by step.
        For reading what a real tracker actually sends when a step does not parse as expected."""
        return {name: self._replies[op].hex() for name, op in
                (("query-auth", OP_QUERY_AUTH), ("open-auth", OP_OPEN_AUTH)) if op in self._replies}

    def _next_seq(self) -> int:
        self._seq = self._seq % 0xFFFFFFFF + 1
        return self._seq

    def ask(self, op: int, payload: bytes, optional: bool = False,
            timeout_s: float = REPLY_TIMEOUT_S) -> bytes | None:
        """Send a request, wait for its matching response, and return its payload. Gaze frames that
        arrive meanwhile are queued for `drain_gaze`. `optional` returns None on no answer rather than
        raising — for a request the tracker can decline, such as a calibration point it will not take.
        `timeout_s` is longer for a reply that is big or slow, such as the calibration model read-back."""
        seq = self._next_seq()
        frame = request(seq, op, payload)
        if len(frame) <= CHUNK + ENVELOPE:
            self.link.write(frame)
        else:
            # Over the tracker's size limit (a calibration model): one envelope per slice, which it
            # joins. Sent whole, the apply is refused.
            ttp = frame[ENVELOPE:]
            for i in range(0, len(ttp), CHUNK):
                piece = ttp[i:i + CHUNK]
                self.link.write(bytes([DIR_OUT, 0, 0, 0]) + struct.pack("<I", len(piece)) + piece)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            data = self.link.read(100)
            if data is None:
                continue
            for frame in self._frames.push(data):
                if frame.magic == MAGIC_RESPONSE and frame.op == op and frame.seq == seq:
                    return frame.payload
                if frame.magic == MAGIC_NOTIFY:
                    self._pending.append(frame)
        if optional:
            return None
        raise TrackerError(f"the eye tracker did not answer request 0x{op:x} — unplug it, plug it in "
                           f"again and wait ten seconds")

    def drain_gaze(self) -> list[GazeFrame]:
        """The gaze frames that have arrived since the last call, decoded. Used during calibration, where
        the live gaze and the request replies share one stream.

        Reads until at least one frame assembles or a short budget passes, because one gaze frame spans
        several USB transfers and a single read often returns only a piece of one — one read per call was
        why the calibration ring never filled."""
        pending, self._pending = self._pending, []
        out = [decode_gaze(f.payload) for f in pending if f.op == STREAM_GAZE]
        deadline = time.monotonic() + DRAIN_BUDGET_S
        while not out and time.monotonic() < deadline:
            data = self.link.read(DRAIN_TIMEOUT_MS)
            if data is None:
                continue
            out += [decode_gaze(f.payload) for f in self._frames.push(data)
                    if f.magic == MAGIC_NOTIFY and f.op == STREAM_GAZE]
        return out

    def connect(self) -> None:
        self.ask(OP_HELLO, hello_payload())
        self.open_auth()
        self.apply_saved_calibration()
        self.ask(OP_SUBSCRIBE, subscribe_payload(STREAM_GAZE))
        self.ask(OP_SET_DISPLAY, display_payload(*self.screen_mm))

    def apply_saved_calibration(self) -> None:
        """Send this tracker's saved calibration, if one was made. The tracker forgets it at every
        session close, so it is applied on every connect. Imported here to keep calibration out of the
        gaze path for a tracker that was never calibrated."""
        from . import calibration

        blob = calibration.load_model(self.serial)
        if blob:
            calibration.apply_model(self, blob)

    def open_auth(self) -> int:
        """Open the session, answering the tracker's challenge if it sets one. Returns the identifier the
        tracker gives the open session, which `close_auth` hands back.

        The tracker says what kind of authentication it wants. Kind zero is none — the gaze stream has
        been seen to want none — and the session opens with nothing more. A non-zero kind is a challenge:
        the tracker sends bytes, and the operator's key turns them into the answer it checks.

        **An answer that will not parse is read as kind zero, not as a fault.** The query reply has only
        one shape in hand, so a reply that does not match it is far more likely this program's gap than a
        tracker demanding a key — and no authentication is the tracker's own answer for the gaze stream.
        Treating it as zero opens the session and lets the stream itself be the test, where a hard stop
        here would hide a tracker that was about to work. The raw replies are kept for `handshake_hex`.
        """
        query = self.ask(OP_QUERY_AUTH, b"\x00\x00")
        self._replies[OP_QUERY_AUTH] = query or b""
        kind = resp_u32_at(query, 0) or 0
        reply = self.ask(OP_OPEN_AUTH, open_auth_payload(kind)) or b""
        self._replies[OP_OPEN_AUTH] = reply
        if kind == 0:
            return resp_u32_at(reply, 0) or 0
        key = auth_key()
        if key is None:
            raise TrackerError("the eye tracker asks for a key to open a session — set it in the "
                               f"{KEY_ENV} environment variable, or turn the tracker off")
        realm_id = resp_u32_at(reply, 0)
        field = resp_u32_at(reply, 1)
        challenge = resp_challenge(reply)
        if realm_id is None or field is None or challenge is None:
            raise TrackerError("the eye tracker asked for a key but sent no challenge to answer")
        self.ask(OP_ANSWER_AUTH, answer_auth_payload(realm_id, field, answer_challenge(key, challenge)))
        return realm_id

    def close_auth(self, realm_id: int) -> None:
        if realm_id:
            try:
                self.ask(OP_CLOSE_AUTH, close_auth_payload(realm_id))
            except TrackerError:
                pass

    def gaze(self, stop=lambda: False) -> Iterator[GazeFrame]:
        while not stop():
            pending, self._pending = self._pending, []
            for frame in pending:
                if frame.op == STREAM_GAZE:
                    yield decode_gaze(frame.payload)
            data = self.link.read(READ_TIMEOUT_MS)
            if data is None:
                continue
            for frame in self._frames.push(data):
                if frame.magic == MAGIC_NOTIFY and frame.op == STREAM_GAZE:
                    yield decode_gaze(frame.payload)


# --- the source ---------------------------------------------------------------------------------

EYES = ("both", "left", "right")


def sample_of(frame: GazeFrame, eye: str, t: float) -> PupilSample:
    """One eye's pupil, or the mean of both. **Both or nothing** in `both`: the two pupils differ by a
    few tenths of a millimetre, so falling back to one eye when the other is lost would put a step in the
    trace each time, of the size this protocol trains."""
    if eye == "both":
        pair = (frame.left.pupil_mm, frame.right.pupil_mm)
        diameter = sum(pair) / 2 if None not in pair else None
    else:
        diameter = getattr(frame, eye).pupil_mm
    return PupilSample(t=t, diameter_mm=diameter, confidence=0.95 if diameter is not None else 0.0,
                       distance_mm=frame.distance_mm(), left_mm=frame.left.pupil_mm,
                       right_mm=frame.right.pupil_mm)


class TrackerPupil:
    """The eye tracker as a `PupilSource`. Connects when `samples()` starts, on the caller's thread.

    Time is the device's own clock, anchored to the host clock at the first frame, so USB scheduling
    does not put jitter into the intervals that the processor's speed check divides by.
    """

    def __init__(self, eye: str = "both", screen_mm: tuple[float, float] = (600.0, 340.0),
                 link: Link | None = None):
        if eye not in EYES:
            raise ValueError(f"eye must be one of {EYES}")
        self.eye = eye
        self.screen_mm = screen_mm
        self._link = link
        self._stop = False
        self.error: str | None = None
        self.serial: str | None = None
        self.rate_hz: float | None = None
        self.handshake_hex: dict[str, str] = {}
        self.position: Position | None = None       # where the eyes sit, for the positioning guide

    def samples(self) -> Iterator[PupilSample]:
        session = None
        try:
            link = self._link if self._link is not None else UsbLink()
            self._link = link
            self.serial = getattr(link, "serial", None)
            session = TrackerSession(link, self.screen_mm)
            session.connect()
            anchor: tuple[float, int] | None = None
            arrivals: list[float] = []
            for frame in session.gaze(stop=lambda: self._stop):
                now = time.time()
                arrivals = (arrivals + [now])[-34:]
                if len(arrivals) > 1:
                    self.rate_hz = (len(arrivals) - 1) / max(1e-6, arrivals[-1] - arrivals[0])
                if frame.timestamp_us is None:
                    t = now
                else:
                    if anchor is None:
                        anchor = (now, frame.timestamp_us)
                    t = anchor[0] + (frame.timestamp_us - anchor[1]) / 1e6
                self.position = Position(frame.left.box, frame.right.box, frame.distance_mm())
                yield sample_of(frame, self.eye, t)
        except TrackerError as exc:
            if not self._stop:
                self.error = str(exc)
        finally:
            if session is not None:
                self.handshake_hex = session.handshake_hex
            link, self._link = self._link, None
            if link is not None:
                try:
                    link.close()
                except Exception:
                    pass

    def close(self) -> None:
        """Ask the reading loop to end. The link is closed by that loop, on its own thread, within one
        read timeout: closing it from here would race a transfer in flight."""
        self._stop = True

    def describe(self) -> str:
        parts = ["eye tracker"]
        if self.serial:
            parts.append(f"serial {self.serial}")
        if self.rate_hz:
            parts.append(f"{self.rate_hz:.0f} readings/s")
        parts.append({"both": "mean of both eyes", "left": "left eye", "right": "right eye"}[self.eye])
        return " · ".join(parts)


def problem() -> str | None:
    """Why the tracker cannot be used on this machine, before anything is tried."""
    try:
        _usb()
    except TrackerError as exc:
        return str(exc)
    return None
