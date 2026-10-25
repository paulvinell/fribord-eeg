"""The eye tracker driver, without the eye tracker.

Every byte layout is pinned here: the requests against bytes known from the device, the stream decoder
against the opening of a real gaze frame, and the connect sequence against a scripted link that answers
the way the device answers. What these tests cannot show is listed in `nfb.pupil.tracker`.
"""

from __future__ import annotations

import struct

import pytest

from nfb.pupil import tracker as tr


# --- a scripted device ------------------------------------------------------------------------

def _inbound(magic: int, seq: int, op: int, payload: bytes) -> bytes:
    body = struct.pack(">IIIIII", magic, seq, 0, op, 0, len(payload)) + payload
    return bytes([tr.DIR_IN, 0, 0, 0]) + struct.pack("<I", len(body) + tr.ENVELOPE) + body


def _column(column: int, value: bytes) -> bytes:
    return tr.u32(tr.T_TAG, tr.TAG_COLUMN) + tr.u32(tr.T_U32, column) + value


def _fixed(value: float) -> bytes:
    return tr.field(tr.T_FIXED16, struct.pack(">i", round(value * 65536)))


def gaze_payload(stamp_us: int = 1_000_000, left: float = 4.2, right: float = 4.4,
                 valid=(0, 0), origin=(30.0, 0.0, 600.0)) -> bytes:
    cols = [
        _column(0x01, tr.field(tr.T_S64, struct.pack(">q", stamp_us))),
        _column(0x02, tr.point3d(*origin)),
        _column(0x06, _fixed(left)),
        _column(0x07, tr.u32(tr.T_U32, valid[0])),
        _column(0x08, tr.point3d(-origin[0], origin[1], origin[2])),
        _column(0x0C, _fixed(right)),
        _column(0x0D, tr.u32(tr.T_U32, valid[1])),
    ]
    row = tr.u32(tr.T_TAG, (len(cols) << 16) | tr.ROW_TYPE)
    return b"\x00\x00" + row + b"".join(cols)


def _resp_field(value: bytes) -> bytes:
    """A handshake response field: type, pad, two-byte length, body — the looser frame the device uses."""
    return struct.pack(">BBH", tr.T_U32, 0, len(value)) + value


class FakeLink:
    """Answers each request with an empty response of the same operation and sequence number. The
    session-open query answers `auth_kind`; when that is non-zero, the open reply carries a realm id, a
    second field and a challenge, and the answer is recorded in `digest`."""

    REALM_ID = 42
    FIELD = 7
    CHALLENGE = bytes(range(16))

    def __init__(self, auth_kind: int = 0, gaze: list[bytes] | None = None, answer: bool = True,
                 notify_during_handshake: bool = False):
        self.auth_kind = auth_kind
        self.gaze = list(gaze or [])
        self.answer = answer
        self.notify_during_handshake = notify_during_handshake
        self.queue: list[bytes] = []
        self.sent: list[tuple[int, int, bytes]] = []
        self.digest: bytes | None = None
        self.closed = False

    def write(self, data: bytes) -> None:
        assert data[0] == tr.DIR_OUT
        assert struct.unpack("<I", data[4:8])[0] == len(data) - tr.ENVELOPE
        magic, seq, _flag, op, _res, plen = struct.unpack(">IIIIII", data[8:32])
        assert magic == tr.MAGIC_REQUEST and plen == len(data) - 32
        self.sent.append((seq, op, data[32:]))
        if not self.answer:
            return
        if self.notify_during_handshake and op == tr.OP_HELLO:
            self.queue.append(_inbound(tr.MAGIC_NOTIFY, 0, tr.STREAM_GAZE, gaze_payload()))
        reply = b""
        if op == tr.OP_QUERY_AUTH:
            reply = b"\x00\x00" + _resp_field(struct.pack(">I", self.auth_kind))
        elif op == tr.OP_OPEN_AUTH and self.auth_kind != 0:
            reply = (b"\x00\x00" + _resp_field(struct.pack(">I", self.REALM_ID))
                     + _resp_field(struct.pack(">I", self.FIELD)) + _resp_field(self.CHALLENGE))
        elif op == tr.OP_OPEN_AUTH:
            reply = b"\x00\x00" + _resp_field(struct.pack(">I", self.REALM_ID))
        elif op == tr.OP_ANSWER_AUTH:
            self.digest = self.sent[-1][2][-16:]
        self.queue.append(_inbound(tr.MAGIC_RESPONSE, seq, op, reply))

    def read(self, timeout_ms: int) -> bytes | None:
        if self.queue:
            return self.queue.pop(0)
        if self.gaze and self.sent and self.sent[-1][1] == tr.OP_SET_DISPLAY:
            return _inbound(tr.MAGIC_NOTIFY, 0, tr.STREAM_GAZE, self.gaze.pop(0))
        return None

    def close(self) -> None:
        self.closed = True


# --- the requests -----------------------------------------------------------------------------

def test_the_hello_request_is_the_79_bytes_the_device_expects():
    frame = tr.request(1, tr.OP_HELLO, tr.hello_payload())
    assert len(frame) == 79
    assert frame[:8] == bytes.fromhex("0000000047000000"), "envelope: outbound, 71 bytes after it"
    assert frame[8:32] == bytes.fromhex("00000051" "00000001" "00000000" "000003e8" "00000000" "0000002f")
    assert frame[32:] == bytes.fromhex(
        "00 00 17 00 00 00 28 00 00 00 09 00 01 00 00 00"
        "01 00 01 00 01 00 02 00 01 00 03 00 01 00 04 00"
        "01 00 05 00 01 00 06 00 01 00 07 00 01 00 08")


def test_the_subscribe_request_carries_the_stream_id():
    payload = tr.subscribe_payload(tr.STREAM_GAZE)
    assert len(payload) == 20
    assert payload[9:11] == b"\x05\x00"


def test_the_display_area_is_three_corners_and_a_trailer():
    payload = tr.display_payload(600.0, 340.0)
    assert len(payload) == 164
    assert tr.q42(200.0)[5:] == (0x0003_2000_0000_0000).to_bytes(8, "big")
    values = [tr._value(k, b) for k, b in tr.fields(payload) if k == tr.T_Q42]
    assert values[:3] == pytest.approx([-300.0, 350.0, 0.0]), "top left: 10 mm above the tracker"
    assert values[6:9] == pytest.approx([-300.0, 10.0, 0.0]), "bottom left"


# --- the stream ---------------------------------------------------------------------------------

def test_the_opening_of_a_real_gaze_frame_decodes():
    """The first bytes of a gaze frame captured from the device: a 39-column row, the timestamp, then a
    column 0x11 holding 4."""
    head = bytes.fromhex(
        "0000"
        "05000000040027 0bb8"
        "0500000004 00020bb9" "0200000004 00000001" "0600000008 0000000045e13a79"
        "0500000004 00020bb9" "0200000004 00000011" "0200000004 00000004".replace(" ", ""))
    cols = tr.columns(head)
    assert cols[0x01] == 0x45E13A79
    assert cols[0x11] == 4


def test_a_gaze_frame_gives_each_pupil_and_the_distance():
    frame = tr.decode_gaze(gaze_payload(left=4.25, right=4.5, origin=(30.0, 0.0, 600.0)))
    assert frame.left.pupil_mm == pytest.approx(4.25, abs=1e-4)
    assert frame.right.pupil_mm == pytest.approx(4.5, abs=1e-4)
    assert frame.timestamp_us == 1_000_000
    assert frame.distance_mm() == pytest.approx((30.0 ** 2 + 600.0 ** 2) ** 0.5)


def test_an_eye_that_is_not_tracked_or_says_minus_one_has_no_pupil():
    lost = tr.decode_gaze(gaze_payload(valid=(4, 0)))
    assert lost.left.pupil_mm is None and lost.right.pupil_mm is not None
    sentinel = tr.decode_gaze(gaze_payload(right=-1.0))
    assert sentinel.right.pupil_mm is None


def test_a_frame_split_across_transfers_is_put_back_together():
    whole = _inbound(tr.MAGIC_NOTIFY, 0, tr.STREAM_GAZE, gaze_payload())
    body = whole[tr.ENVELOPE:]
    parts = [body[:40], body[40:90], body[90:]]
    transfers = [bytes([1, 0, 0, 0]) + struct.pack("<I", len(whole)) + parts[0]]
    transfers += [bytes([1, 0, 0, 0]) + struct.pack("<I", len(p) + 8) + p for p in parts[1:]]
    r = tr.Reassembler()
    frames = [f for t in transfers for f in r.push(t)]
    assert len(frames) == 1
    assert tr.decode_gaze(frames[0].payload).left.pupil_mm == pytest.approx(4.2, abs=1e-4)


def test_two_frames_in_one_transfer_are_both_read():
    one = _inbound(tr.MAGIC_RESPONSE, 3, tr.OP_HELLO, b"")
    two = _inbound(tr.MAGIC_NOTIFY, 0, tr.STREAM_GAZE, gaze_payload())
    frames = tr.Reassembler().push(one + two)
    assert [(f.magic, f.op) for f in frames] == [(tr.MAGIC_RESPONSE, tr.OP_HELLO),
                                                 (tr.MAGIC_NOTIFY, tr.STREAM_GAZE)]


# --- the connect sequence -------------------------------------------------------------------------

def test_the_connect_sequence_and_then_gaze():
    link = FakeLink(gaze=[gaze_payload(stamp_us=i * 30_000) for i in range(3)])
    session = tr.TrackerSession(link, (600.0, 340.0))
    session.connect()
    assert [op for _seq, op, _p in link.sent] == [
        tr.OP_HELLO, tr.OP_QUERY_AUTH, tr.OP_OPEN_AUTH, tr.OP_SUBSCRIBE, tr.OP_SET_DISPLAY]
    assert [seq for seq, _op, _p in link.sent] == [1, 2, 3, 4, 5]
    frames = []
    for frame in session.gaze():
        frames.append(frame)
        if len(frames) == 3:
            break
    assert [f.timestamp_us for f in frames] == [0, 30_000, 60_000]


def test_a_gaze_frame_during_the_handshake_is_kept_not_taken_for_an_answer():
    link = FakeLink(notify_during_handshake=True)
    session = tr.TrackerSession(link)
    session.connect()
    assert next(session.gaze()).left.pupil_mm is not None


def test_a_tracker_that_asks_for_a_key_is_answered_when_one_is_set(monkeypatch):
    monkeypatch.setenv(tr.KEY_ENV, b"a secret key".hex())
    link = FakeLink(auth_kind=10002, gaze=[gaze_payload()])
    session = tr.TrackerSession(link)
    realm = session.open_auth()
    assert realm == FakeLink.REALM_ID
    assert link.digest == tr.answer_challenge(b"a secret key", FakeLink.CHALLENGE)
    assert [op for _s, op, _p in link.sent] == [
        tr.OP_QUERY_AUTH, tr.OP_OPEN_AUTH, tr.OP_ANSWER_AUTH]


def test_a_tracker_that_asks_for_a_key_with_none_set_stops_with_a_message(monkeypatch):
    monkeypatch.delenv(tr.KEY_ENV, raising=False)
    with pytest.raises(tr.TrackerError, match="asks for a key"):
        tr.TrackerSession(FakeLink(auth_kind=10002)).connect()


def test_the_key_is_read_as_hex_then_as_text_then_from_a_file(monkeypatch, tmp_path):
    monkeypatch.setenv(tr.KEY_ENV, "49 53")      # hex ignores the space
    assert tr.auth_key() == b"\x49\x53"
    monkeypatch.setenv(tr.KEY_ENV, "not-hex-key")
    assert tr.auth_key() == b"not-hex-key"
    key_file = tmp_path / "key"
    key_file.write_text("abcd\n")
    monkeypatch.setenv(tr.KEY_ENV, str(key_file))
    assert tr.auth_key() == b"\xab\xcd"
    monkeypatch.delenv(tr.KEY_ENV)
    assert tr.auth_key() is None


def test_a_tracker_that_needs_no_key_opens_with_none_set(monkeypatch):
    monkeypatch.delenv(tr.KEY_ENV, raising=False)
    link = FakeLink(auth_kind=0, gaze=[gaze_payload()])
    session = tr.TrackerSession(link)
    session.connect()
    assert tr.OP_ANSWER_AUTH not in [op for _s, op, _p in link.sent]


def test_a_tracker_that_does_not_answer_is_reported(monkeypatch):
    monkeypatch.setattr(tr, "REPLY_TIMEOUT_S", 0.05)
    with pytest.raises(tr.TrackerError, match="did not answer"):
        tr.TrackerSession(FakeLink(answer=False)).connect()


# --- the source ---------------------------------------------------------------------------------

def _samples(eye: str, payloads: list[bytes]):
    source = tr.TrackerPupil(eye=eye, link=FakeLink(gaze=payloads))
    out = []
    for sample in source.samples():
        out.append(sample)
        if len(out) == len(payloads):
            source.close()
            break
    return source, out


def test_both_eyes_is_the_mean_and_nothing_when_one_is_lost():
    _src, out = _samples("both", [gaze_payload(left=4.0, right=4.4),
                                  gaze_payload(stamp_us=1_030_000, valid=(4, 0))])
    assert out[0].diameter_mm == pytest.approx(4.2, abs=1e-4)
    assert out[1].diameter_mm is None and out[1].right_mm is not None, "no step to one eye"


def test_one_eye_reads_that_eye_and_time_follows_the_device_clock():
    _src, out = _samples("right", [gaze_payload(stamp_us=1_000_000, right=4.4),
                                   gaze_payload(stamp_us=1_500_000, right=4.6)])
    assert [s.diameter_mm for s in out] == pytest.approx([4.4, 4.6], abs=1e-4)
    assert out[1].t - out[0].t == pytest.approx(0.5)


def test_a_failed_connection_becomes_the_source_error_and_closes_the_link(monkeypatch):
    monkeypatch.delenv(tr.KEY_ENV, raising=False)
    link = FakeLink(auth_kind=7)
    source = tr.TrackerPupil(link=link)
    assert list(source.samples()) == []
    assert "key" in source.error
    assert link.closed


def test_the_device_screen_says_why_the_tracker_did_not_connect(monkeypatch):
    from conftest import waitfor
    from test_layout import walk
    from test_screens import StubPage

    import flet as ft

    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import PupilChoice
    from nfb_console.pupil import PupilRunner

    def no_tracker():
        raise tr.TrackerError("no eye tracker on USB — connect it")

    monkeypatch.setattr(tr, "UsbLink", no_tracker)
    device = Device(Settings())
    choice = PupilChoice(kind="off")
    runner = PupilRunner(choice)
    try:
        control = device_panel.view(StubPage(), device, pupil_choice=choice, pupil_runner=runner)
        found = [c for c in walk(control) if isinstance(c, ft.Container)
                 and isinstance(c.content, ft.Text) and c.content.value == "tracker"]
        found[-1].on_click(None)
        waitfor(lambda: runner.error is not None, what="the connection fails")
        assert "no eye tracker on USB" in runner.describe()
    finally:
        runner.release()
        device.close()


# --- the screen size ------------------------------------------------------------------------------

def _screens():
    from nfb_console.displays import Display

    return [Display(1, True, False, 301.0, 196.0), Display(2, False, True, 597.0, 336.0)]


def test_the_tracker_screen_is_the_external_one_and_falls_back_in_order():
    from nfb_console.displays import Display, likely_tracker_display

    built_in, external = _screens()
    assert likely_tracker_display([built_in, external])[0] is external
    assert likely_tracker_display([built_in])[0] is built_in
    second = Display(3, False, False, 531.0, 299.0)
    assert likely_tracker_display([built_in, second, external])[0] is external, "the main external"
    assert likely_tracker_display([])[0] is None


def test_the_size_is_read_not_typed_and_a_lost_screen_falls_back(monkeypatch):
    from nfb_console import displays
    from nfb_console.inputs import PupilChoice

    monkeypatch.setattr(displays, "displays", _screens)
    choice = PupilChoice(kind="tracker")
    assert choice.screen_size()[:2] == (597.0, 336.0)
    choice.screen = "1"
    assert choice.screen_size()[:2] == (301.0, 196.0)
    choice.screen = "99"
    assert choice.screen_size()[:2] == (597.0, 336.0), "a screen that is gone: the automatic one"
    choice.screen = "manual"
    choice.screen_width_mm, choice.screen_height_mm = 800.0, 335.0
    assert choice.screen_size()[:2] == (800.0, 335.0)
    monkeypatch.setattr(displays, "displays", lambda: [])
    assert PupilChoice(kind="tracker").screen_size()[2] == "the size you typed"


def test_reading_the_screens_from_macos_does_not_raise():
    from nfb_console.displays import displays

    for d in displays():
        assert d.width_mm >= 100 and d.height_mm >= 100


def test_the_device_screen_offers_each_reported_screen(monkeypatch):
    from test_screens import StubPage, _texts

    from nfb import Device, Settings
    from nfb_console import device_panel, displays
    from nfb_console.inputs import PupilChoice

    monkeypatch.setattr(displays, "displays", _screens)
    device = Device(Settings())
    try:
        control = device_panel.view(StubPage(), device, pupil_choice=PupilChoice(kind="tracker"))
        shown = " | ".join(x.value for x in _texts(control) if x.value)
        assert "external · 597 × 336 mm" in shown and "built-in · 301 × 196 mm" in shown
        assert "chosen because it is the only external screen" in shown
    finally:
        device.close()


# --- calibration ----------------------------------------------------------------------------------

from nfb.pupil import calibration as cal                           # noqa: E402

TAG_POINT2D = 0x00021F40




def _gaze_frame_with_point(x: float, y: float) -> bytes:
    cols = [
        _column(0x01, tr.field(tr.T_S64, struct.pack(">q", 0))),
        _column(0x07, tr.u32(tr.T_U32, 0)),
        _column(0x0D, tr.u32(tr.T_U32, 0)),
        _column(tr.COL_GAZE_2D, tr.u32(tr.T_TAG, TAG_POINT2D) + tr.q42(x) + tr.q42(y)),
    ]
    row = tr.u32(tr.T_TAG, (len(cols) << 16) | tr.ROW_TYPE)
    return b"\x00\x00" + row + b"".join(cols)


def test_a_gaze_frame_carries_where_the_eyes_look_on_the_screen():
    frame = tr.decode_gaze(_gaze_frame_with_point(0.3, 0.7))
    assert frame.gaze_2d == pytest.approx((0.3, 0.7), abs=1e-4)
    away = tr.decode_gaze(_gaze_frame_with_point(-1.0, -1.0))
    assert away.gaze_2d is None


def test_the_add_point_payload_is_two_numbers_and_an_eye_mask():
    payload = cal.add_point_payload(0.25, 0.75, cal.EYE_BOTH)
    frame = tr.request(1, cal.OP_ADD_POINT, payload)
    assert len(frame) == 69, "8 envelope + 24 header + 37 payload"
    assert payload[-9] == tr.T_U32 and payload[-1] == cal.EYE_BOTH


class CalLink:
    """A tracker that reports a settable gaze point and acks calibration ops. `refuse_add` makes it
    decline every point, and `retrieve` is the model it hands back."""

    def __init__(self, retrieve: bytes = b"MODEL-BYTES", refuse_add: bool = False):
        self.point = (0.5, 0.5)
        self.retrieve = retrieve
        self.refuse_add = refuse_add
        self.ops: list[int] = []
        self.applied: bytes | None = None
        self._queue: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        # Transfers of one request may arrive over several write calls; join them as the device does.
        length = struct.unpack_from("<I", data, 4)[0]
        self._acc = getattr(self, "_acc", bytearray()) + data[tr.ENVELOPE:tr.ENVELOPE + length]
        if len(self._acc) < tr.HEADER:
            return
        _m, seq, _f, op, _r, plen = struct.unpack_from(">IIIIII", self._acc, 0)
        if len(self._acc) < tr.HEADER + plen:
            return
        payload = bytes(self._acc[tr.HEADER:tr.HEADER + plen])
        self._acc = bytearray()
        for seq, op, payload in [(seq, op, payload)]:
            self.ops.append(op)
            if op == cal.OP_ADD_POINT and self.refuse_add:
                continue                      # no answer: a refused point
            if op == cal.OP_APPLY:
                self.applied = payload[2:]
            if op in (tr.OP_QUERY_AUTH, tr.OP_OPEN_AUTH):
                reply = b"\x00\x00" + _resp_field(struct.pack(">I", 0))   # no key needed
            elif op == cal.OP_RETRIEVE:
                reply = self.retrieve
            else:
                reply = b""
            self._queue.append(_inbound(tr.MAGIC_RESPONSE, seq, op, reply))

    def read(self, timeout_ms: int) -> bytes | None:
        if self._queue:
            return self._queue.pop(0)
        return _inbound(tr.MAGIC_NOTIFY, 0, tr.STREAM_GAZE,
                        _gaze_frame_with_point(*self.point))

    def close(self) -> None:
        self.closed = True




def _run_calibration(link: CalLink):
    session = tr.TrackerSession(link)
    c = cal.Calibrator(session, serial="UNIT1")
    now = 0.0
    for _ in range(4000):
        link.point = c.points[min(c.index, len(c.points) - 1)]
        c.step(now=now)
        now += 0.1
        if c.done or c.failed:
            break
    return c


def test_a_full_calibration_samples_every_point_and_saves_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    link = CalLink(retrieve=b"A-REAL-MODEL")
    c = _run_calibration(link)
    assert c.done and not c.failed
    assert link.ops.count(cal.OP_ADD_POINT) == len(cal.POINTS), "one point each"
    assert cal.OP_COMPUTE in link.ops and link.ops[-1] in (cal.OP_STOP, cal.OP_RETRIEVE)
    assert cal.load_model("UNIT1") == b"A-REAL-MODEL", "kept under the home directory"


def test_a_calibration_with_no_tracked_eye_fails_after_its_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    class NoEye(CalLink):
        def read(self, timeout_ms):
            if self._queue:
                return self._queue.pop(0)
            cols = [_column(0x07, tr.u32(tr.T_U32, 4)), _column(0x0D, tr.u32(tr.T_U32, 4))]
            row = tr.u32(tr.T_TAG, (len(cols) << 16) | tr.ROW_TYPE)
            return _inbound(tr.MAGIC_NOTIFY, 0, tr.STREAM_GAZE, b"\x00\x00" + row + b"".join(cols))

    link = NoEye()
    c = cal.Calibrator(tr.TrackerSession(link))
    now = 0.0
    for _ in range(6000):
        c.step(now=now)
        now += 0.5
        if c.done or c.failed:
            break
    assert c.failed and c.attempt == cal.MAX_ATTEMPTS
    assert cal.OP_STOP in link.ops, "it leaves calibration mode on failure"


def test_a_saved_model_is_applied_on_connect(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cal.save_model("UNIT1", b"X" * 20000)      # larger than one transfer
    link = CalLink()
    session = tr.TrackerSession(link, serial="UNIT1")
    session.connect()
    assert link.applied == b"X" * 20000, "the whole model arrives, rejoined from its transfers"
    assert cal.OP_APPLY in link.ops


def test_a_tracker_with_no_saved_model_applies_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    link = CalLink()
    tr.TrackerSession(link, serial="NEVER-CALIBRATED").connect()
    assert cal.OP_APPLY not in link.ops


def test_an_unreadable_open_answer_is_treated_as_no_key_not_a_fault(monkeypatch):
    """The real tracker's reply may not match the one shape this program knows. A reply it cannot read is
    read as no authentication, so the session opens and the stream itself is the test — the gaze stream
    needs no key. The raw bytes are kept for diagnosis."""
    monkeypatch.delenv(tr.KEY_ENV, raising=False)

    class OddLink(FakeLink):
        def write(self, data: bytes) -> None:
            magic, seq, _f, op, _r, _p = struct.unpack(">IIIIII", data[8:32])
            self.sent.append((seq, op, data[32:]))
            reply = b"\xde\xad\xbe\xef" if op in (tr.OP_QUERY_AUTH, tr.OP_OPEN_AUTH) else b""
            self.queue.append(_inbound(tr.MAGIC_RESPONSE, seq, op, reply))

    link = OddLink(gaze=[gaze_payload()])
    session = tr.TrackerSession(link)
    session.connect()                               # does not raise
    assert tr.OP_SUBSCRIBE in [op for _s, op, _p in link.sent]
    assert session.handshake_hex["query-auth"] == "deadbeef"


def test_the_runner_reports_a_connection_state():
    from nfb_console.inputs import PupilChoice
    from nfb_console.pupil import PupilRunner

    off = PupilRunner(PupilChoice(kind="off"))
    assert off.status() == ("off", "muted")
    on = PupilRunner(PupilChoice(kind="tracker"))
    on.error = "no eye tracker on USB"
    assert on.status() == ("no eye tracker on USB", "bad")


def test_a_tracked_eye_fills_progress_and_samples_an_untracked_one_does_not():
    link = CalLink()
    c = cal.Calibrator(tr.TrackerSession(link))
    c.start(now=0.0)
    assert c.dwell_progress(0.0) == 0.0, "nothing held yet"
    link.point = c.points[0]                          # eyes tracked on the dot
    progressed = False
    for k in range(1, 40):
        c.step(now=k * 0.1)
        if 0.0 < c.dwell_progress(k * 0.1) < 1.0:
            progressed = True
        if c.index == 1:                             # point 0 sampled, moved on
            break
    assert progressed, "the ring fills while the eyes are tracked"
    assert c.index == 1, "a tracked eye is captured"

    # Frames arrive but no eye is tracked: the timer never runs.
    class Untracked(CalLink):
        def read(self, timeout_ms):
            if self._queue:
                return self._queue.pop(0)
            cols = [_column(0x07, tr.u32(tr.T_U32, 4)), _column(0x0D, tr.u32(tr.T_U32, 4))]
            row = tr.u32(tr.T_TAG, (len(cols) << 16) | tr.ROW_TYPE)
            return _inbound(tr.MAGIC_NOTIFY, 0, tr.STREAM_GAZE, b"\x00\x00" + row + b"".join(cols))

    u = cal.Calibrator(tr.TrackerSession(Untracked()))
    u.start(now=0.0)
    for k in range(1, 40):
        u.step(now=k * 0.1)
    assert u.index == 0 and u.dwell_progress(4.0) == 0.0, "no tracked eye, nothing captured"





def _frame_cols(cols: list[bytes]) -> bytes:
    row = tr.u32(tr.T_TAG, (len(cols) << 16) | tr.ROW_TYPE)
    return b"\x00\x00" + row + b"".join(cols)


def test_fixation_falls_back_from_combined_to_per_eye_to_direction():
    def pt2(x, y):
        return tr.u32(tr.T_TAG, TAG_POINT2D) + tr.q42(x) + tr.q42(y)

    # combined invalid, per-eye valid → fixation is the per-eye mean
    f = tr.decode_gaze(_frame_cols([
        _column(0x07, tr.u32(tr.T_U32, 0)),        # left eye tracked
        _column(tr.COL_GAZE_2D, pt2(-1.0, -1.0)),
        _column(tr.COL_GAZE_2D_EYE["left"], pt2(0.40, 0.60)),
        _column(tr.COL_GAZE_2D_EYE["right"], pt2(0.50, 0.70)),
    ]))
    assert f.gaze_2d is None
    assert f.fixation == pytest.approx((0.45, 0.65), abs=1e-3)
    assert "left-2d" in f.probe

    # all screen gaze invalid, only a direction → fixation is the direction mean, gaze_2d still None
    d = tr.decode_gaze(_frame_cols([
        _column(0x07, tr.u32(tr.T_U32, 0)),        # left eye tracked
        _column(tr.COL_GAZE_2D, pt2(-1.0, -1.0)),
        _column(tr.COL_DIR["left"], tr.point3d(0.1, 0.2, -0.9)),
        _column(tr.COL_DIR["right"], tr.point3d(0.3, 0.2, -0.9)),
    ]))
    assert d.gaze_2d is None and d.fixation == pytest.approx((0.2, 0.2), abs=1e-3)


def test_the_calibrator_captures_a_steady_gaze_from_a_fallback_signal():
    """Combined screen gaze invalid, but a steady per-eye gaze still lets a point be held."""
    class EyeOnlyLink(CalLink):
        def read(self, timeout_ms):
            if self._queue:
                return self._queue.pop(0)
            px, py = self.point
            cols = [
                _column(0x01, tr.field(tr.T_S64, struct.pack(">q", 0))),
                _column(0x07, tr.u32(tr.T_U32, 0)),        # left eye tracked
                _column(tr.COL_GAZE_2D, tr.u32(tr.T_TAG, TAG_POINT2D) + tr.q42(-1.0) + tr.q42(-1.0)),
                _column(tr.COL_GAZE_2D_EYE["left"], tr.u32(tr.T_TAG, TAG_POINT2D) + tr.q42(px) + tr.q42(py)),
                _column(tr.COL_GAZE_2D_EYE["right"], tr.u32(tr.T_TAG, TAG_POINT2D) + tr.q42(px) + tr.q42(py)),
            ]
            return _inbound(tr.MAGIC_NOTIFY, 0, tr.STREAM_GAZE, _frame_cols(cols))

    link = EyeOnlyLink()
    c = cal.Calibrator(tr.TrackerSession(link))
    c.start(now=0.0)
    link.point = c.points[0]
    for k in range(1, 40):
        c.step(now=k * 0.1)
        if c.index == 1:
            break
    assert c.index == 1, "a steady per-eye gaze is captured even when combined screen gaze is invalid"
    assert c.last_screen_gaze is None, "no combined screen gaze was ever valid"


# --- the positioning guide ------------------------------------------------------------------------

def test_the_positioning_guide_reads_eye_position_and_distance():
    from nfb_console import pupil_guide as pg
    from nfb.pupil.tracker import Position

    none = Position(None, None, None)
    assert not none.seen
    assert pg.guide_status(none)[1] == "bad"
    assert len(pg.guide_shapes(none, 240, 150)) == 2, "box and centre target, no eye dots"

    far = Position((0.5, 0.5), (0.5, 0.5), 1100.0)
    assert "move back" in pg.guide_status(far)[0] and pg.guide_status(far)[1] == "accent"
    near = Position((0.5, 0.5), (0.5, 0.5), 300.0)
    assert "closer" in pg.guide_status(near)[0]

    off = Position((0.9, 0.5), (0.85, 0.5), 650.0)
    assert "centre" in pg.guide_status(off)[0].lower()

    good = Position((0.52, 0.49), (0.48, 0.51), 650.0)
    assert pg.guide_status(good)[1] == "good"
    assert len(pg.guide_shapes(good, 240, 150)) == 4, "box, target, and two eye dots"


def test_a_gaze_frame_carries_the_trackbox_position():
    cols = [
        _column(0x07, tr.u32(tr.T_U32, 0)),
        _column(tr.COL_TRACKBOX["left"], tr.point3d(0.54, 0.65, 0.23)),
    ]
    row = tr.u32(tr.T_TAG, (len(cols) << 16) | tr.ROW_TYPE)
    f = tr.decode_gaze(b"\x00\x00" + row + b"".join(cols))
    assert f.left.box == pytest.approx((0.54, 0.65), abs=1e-3)


def test_a_point_is_not_sampled_during_the_settle_pause():
    link = CalLink()
    c = cal.Calibrator(tr.TrackerSession(link))
    c.start(now=0.0)
    link.point = c.points[0]
    c.step(now=0.1)                                   # eyes land, settle begins
    c.step(now=0.1 + cal.SETTLE_S - 0.1)             # still within the settle
    assert c.index == 0 and c.dwell_progress(0.1 + cal.SETTLE_S - 0.1) == 0.0, "ring empty while settling"
    # after settle + dwell, the point is taken
    for k in range(1, 60):
        c.step(now=k * 0.1)
        if c.index == 1:
            break
    assert c.index == 1


def test_the_guide_reports_eyes_seen_from_distance_alone():
    from nfb_console import pupil_guide as pg
    from nfb.pupil.tracker import Position

    # tracked, distance known, but no trackbox position: still "seen", not "does not see your eyes"
    p = Position(None, None, 590.0)
    assert p.seen
    word, tone = pg.guide_status(p)
    assert tone == "good" and "does not see" not in word


# --- the camera -----------------------------------------------------------------------------------

def test_camera_emits_clean_single_frames_and_drops_the_rest():
    from nfb.pupil import camera as cam

    img = (bytes(range(256)) * (cam.FRAME_BYTES // 256 + 1))[:cam.FRAME_BYTES]

    transfers = [
        bytes([0x02, 0x03]) + img + b"\x00\x00\x00",   # 2-byte header + a frame + a few trailing bytes
        bytes([0x0c]) + b"\x00" * 11 + img,              # 12-byte header + a frame
        b"",                                              # timeout tick, ignored
        bytes([0x02, 0x03]) + img + img,                 # two frames merged — dropped, not torn
        bytes([0x40]) + img[:1000],                      # no valid header — dropped
        bytes([0x02, 0x03]) + img[:100000],              # short — dropped
    ]
    frames = list(cam.assemble_frames(iter(transfers)))
    assert len(frames) == 2, "only the two clean single-frame transfers"
    assert all(len(f) == cam.FRAME_BYTES for f in frames)
    assert frames[0] == img and frames[1] == img
