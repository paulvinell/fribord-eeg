"""Device manager tests, against a fake amplifier.

The point of these is the rules we agreed rather than the plumbing: modes are exclusive, settings
cannot change mid-stream, and a dead device is reported with a remedy instead of retried.
"""

from __future__ import annotations

import struct
import time

import optima
import pytest

from conftest import waitfor

from nfb.device import Device, Mode, Settings
from nfb.sources import WaitingSource
from nfb.stream import Gap


class FakeOptima:
    """Answers the handful of commands Device issues, and streams frames on demand."""

    def __init__(self, fail_read: bool = False):
        self.port = "/dev/fake"
        self.written: list[bytes] = []
        self.sent: list[int] = []
        self.notif: list[int] = []
        self.closed = False
        self.fail_read = fail_read
        self._phase = 0
        self._streaming = False
        self._testing = False
        # Every entity reports 0, which is what the real amplifier sends for a channel that is *fine*.
        # This fake used to send 1 on every entity, matching a wrong assumption in `Block.valid` — so the
        # suite agreed with itself and no test could see that live sessions marked every channel unusable.
        self.state_word = 0

    # identity
    def device_info(self) -> bytes:
        buf = bytearray(90)
        buf[0x02 - 2:0x02 - 2 + 16] = b"Neurobit Systems"
        buf[0x23 - 2:0x23 - 2 + 22] = b"Neurobit Optima+ 4 USB"
        buf[0x44 - 2:0x44 - 2 + 8] = b"00000000"
        return bytes(buf)

    def firmware_info(self) -> bytes:
        return bytes([2, 9, 5])

    def write_config(self, blob: bytes) -> None:
        self.written.append(blob)

    def set_notifications(self, mask: int = optima.NOTIF_ENABLE) -> None:
        self.notif.append(mask)

    def start_measurement(self) -> None:
        self.sent.append(optima.CMD["START_MES"])
        self._streaming = True

    def stop_measurement(self) -> None:
        self.sent.append(optima.CMD["STOP_MES"])
        self._streaming = False

    def request(self, code: int, payload: bytes = b"") -> bytes:
        self.sent.append(code)
        if code == optima.CMD["START_TEST"]:
            self._streaming = self._testing = True
            self._phase = 0             # the real device restarts at 0, which is what makes parity absolute
        if code == optima.CMD["STOP_TEST"]:
            self._streaming = self._testing = False
        return b"\x00"

    def read_message(self):
        if self.fail_read:
            raise optima.ProtocolError("link lost")
        if not self._streaming:
            raise TimeoutError("idle")
        time.sleep(0.002)
        if self._testing:
            # Shaped like the real test stream: one value per channel per frame, phase advancing by one,
            # and the two electrodes alternating with its parity. Even is "+", which is the whole point.
            counts = 30000 if self._phase % 2 == 0 else 15000
            payload = (struct.pack("<HH", self._phase, self.state_word)
                       + struct.pack("<4h", *(counts + c for c in range(4))))
            self._phase = (self._phase + 1) % 65536
            return optima.Message(optima.NOT_DATA, payload)
        payload = struct.pack("<HH", self._phase, self.state_word) + bytes(64)
        self._phase = (self._phase + 8) % 65536
        return optima.Message(optima.NOT_DATA, payload)

    def close(self) -> None:
        self.closed = True


def connected(monkeypatch, **kw) -> tuple[Device, FakeOptima]:
    fake = FakeOptima(**kw)
    monkeypatch.setattr(optima, "Optima", lambda *a, **k: fake)
    d = Device()
    assert d.connect() is True
    return d, fake


def test_connect_reads_identity_and_writes_a_config(monkeypatch):
    d, fake = connected(monkeypatch)
    assert d.status.mode is Mode.IDLE
    assert d.status.serial == "00000000"
    assert d.status.model == "Neurobit Optima+ 4 USB"
    assert d.status.firmware == "2.9.5"
    assert len(fake.written) == 1 and optima.validate_config(fake.written[0]) == []
    d.close()
    assert fake.closed


def test_modes_are_exclusive(monkeypatch):
    d, _ = connected(monkeypatch)
    d.start_measurement()
    assert d.status.mode is Mode.MEASURING
    with pytest.raises(RuntimeError, match="already measuring"):
        d.start_impedance()
    d.stop()
    assert d.status.mode is Mode.IDLE
    d.start_impedance()
    assert d.status.mode is Mode.TESTING
    d.stop()
    d.close()


def test_the_command_sequence_matches_the_protocol(monkeypatch):
    d, fake = connected(monkeypatch)
    d.start_measurement()
    d.stop()
    assert fake.notif == [optima.NOTIF_ENABLE, optima.NOTIF_CLEAR]
    assert fake.sent == [optima.CMD["START_MES"], optima.CMD["STOP_MES"]]
    fake.notif.clear(); fake.sent.clear()
    d.start_impedance()
    d.stop()
    assert fake.notif == [optima.NOTIF_ENABLE, optima.NOTIF_CLEAR]
    assert fake.sent == [optima.CMD["START_TEST"], optima.CMD["STOP_TEST"]]
    d.close()


def test_settings_cannot_change_mid_stream(monkeypatch):
    d, _ = connected(monkeypatch)
    d.start_measurement()
    with pytest.raises(RuntimeError, match="cannot change settings while measuring"):
        d.reconfigure(hz=250.0)
    d.stop()
    d.reconfigure(hz=250.0, reference="bipolar")
    assert d.status.settings.hz == 250.0 and d.status.settings.reference == "bipolar"
    d.close()


def test_reconfigure_rejects_an_illegal_combination(monkeypatch):
    d, _ = connected(monkeypatch)
    with pytest.raises(ValueError, match="exceeds the device limit"):
        d.reconfigure(hz=2000.0, channels=4)      # 8000 sps against the 5000 limit
    assert d.status.settings.hz == 1000.0, "settings must not change on a rejected reconfigure"
    d.close()


def test_reconfigure_ends_the_epoch_so_subscribers_rebuild(monkeypatch):
    d, _ = connected(monkeypatch)
    sub = d.subscribe("traces")
    before = sub.channels()[0].rate_hz
    epoch = d.status.epoch
    d.reconfigure(hz=250.0)
    assert d.status.epoch > epoch
    assert sub.channels()[0].rate_hz == 250.0 != before, "spec must follow the new epoch"
    d.close()


def test_subscribers_each_get_the_frames(monkeypatch):
    d, _ = connected(monkeypatch)
    a, b = d.subscribe("a"), d.subscribe("b")
    assert d.subscribers == ("a", "b")
    d.start_measurement()
    got = {"a": 0, "b": 0}
    deadline = time.monotonic() + 2.0
    for name, sub in (("a", a), ("b", b)):
        for item in sub.blocks(timeout=1.0):
            if not isinstance(item, Gap):
                got[name] += 1
            if got[name] >= 8 or time.monotonic() > deadline:
                break
    d.stop()
    assert got["a"] >= 8 and got["b"] >= 8
    a.close(); b.close()
    assert d.subscribers == ()
    d.close()


def test_a_slow_subscriber_drops_its_own_frames_only(monkeypatch):
    d, _ = connected(monkeypatch)
    slow = d.subscribe("slow", queue_size=4)
    fast = d.subscribe("fast")
    d.start_measurement()
    # Nobody consumes `slow`, so what this waits for is its queue overflowing — the thing under test.
    waitfor(lambda: slow.dropped > 0, what="the slow subscriber never dropped anything")
    d.stop()
    assert slow.dropped > 0, "a full queue must drop rather than block the reader"
    assert fast.queue.qsize() > slow.queue.qsize()
    d.close()


def test_a_dead_device_reports_a_remedy_and_does_not_retry(monkeypatch):
    fake = FakeOptima()

    def explode(*a, **k):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(optima, "Optima", explode)
    d = Device()
    assert d.connect() is False
    assert d.status.mode is Mode.DISCONNECTED
    assert "Invalid argument" in d.status.failure
    assert "replug" in d.status.remedy


def test_a_read_failure_stops_cleanly_rather_than_spinning(monkeypatch):
    d, fake = connected(monkeypatch)
    sub = d.subscribe("s")
    fake.fail_read = True
    d.start_measurement()
    waitfor(lambda: d.status.mode is Mode.DISCONNECTED, what="the dead link was never noticed")
    assert d.status.mode is Mode.DISCONNECTED
    assert "link lost" in d.status.failure
    assert list(sub.blocks(timeout=0.2)) == [], "epoch must end so the consumer stops"


def test_impedance_readings_are_available_without_subscribing(monkeypatch):
    d, _ = connected(monkeypatch)
    d.start_impedance()
    waitfor(lambda: len(d.contact) == 4, what="no contact readings arrived")
    d.stop()
    assert d.contact, "device should expose the newest contact reading per channel"
    assert all(s.worst >= 0 for s in d.contact.values())
    d.close()


def test_a_channel_reports_two_electrodes_on_alternating_frames(monkeypatch):
    """A voltage channel measures a difference, so it has two electrodes and two contacts to check. The
    device sends one value per channel per frame and alternates between them, which collapsed into a single
    newest reading looked like one number flickering twice a second — a fault in the instrument rather than
    two different electrodes.

    Which electrode a frame carries is decided by the parity of its phase: **even is "+"**. Verified on
    hardware with one electrode wired to ch0 — on even frames the unwired channels read exactly full scale
    while ch0 read its own electrode, and on odd frames all four agreed within 0.3 kOhm because linked
    referencing joins the "−" inputs inside the device. Three fresh starts each began at phase 0, which is
    what makes the parity absolute rather than relative to when the test began.
    """
    d, _ = connected(monkeypatch)
    d.start_impedance()
    waitfor(lambda: all(c.plus is not None and c.minus is not None for c in d.contact.values())
            and len(d.contact) == 4, what="not every channel reported both electrodes")
    d.stop()
    scale = optima.test_scale()
    for ch, spot in d.contact.items():
        assert spot.plus is not None and spot.minus is not None, f"ch{ch} lost an electrode"
        # the fake sends 30000 counts on even phases and 15000 on odd ones, so the split is checkable
        assert spot.plus == pytest.approx((30000 + ch) * scale)
        assert spot.minus == pytest.approx((15000 + ch) * scale)
        assert spot.worst == spot.plus, "contact is only as good as its worse electrode"
    d.close()


def test_an_empty_channel_is_dropped_from_the_contact_check(monkeypatch):
    """The device drives the test on every *enabled* channel at once, and a channel with no electrode
    disturbs the ones that have them. Measured on hardware with one electrode at C4 and one reference:

        1 channel  linked  18.25 / 12.60      4 channels  linked  28.44 / 12.60
        1 channel bipolar  18.31 / 12.85      4 channels bipolar  69.21 / 49.19

    So the true figures are ~18 and ~13 kOhm and the four-channel bipolar reading is out by 3.8x — enough
    to condemn a good electrode or pass a bad one. That the two reference modes agree to 0.25 kOhm on one
    channel is what says the scale factor and the wiring are both right and the empty channels are the whole
    story. Hence this is a correction, not tidying.
    """
    from nfb.device import OPEN_KOHM, Contact

    d, _ = connected(monkeypatch)
    d.start_impedance()
    waitfor(lambda: bool(d.contact), what="the impedance stream never started")
    # ch0 wired, the rest open — what one electrode on a four-channel amplifier looks like
    d.contact.update({0: Contact(plus=18.3, minus=12.8),
                      **{c: Contact(plus=OPEN_KOHM, minus=OPEN_KOHM) for c in (1, 2, 3)}})
    assert d.unwired() == [1, 2, 3]

    assert d.narrow_to_wired() == [1, 2, 3]
    assert d.status.settings.channels == 1, "the test now runs on the wired channel alone"
    assert d.status.mode is Mode.TESTING, "and it is still running"
    assert d.narrow_to_wired() == [], "narrowing twice would undo the restore"

    d.stop()
    assert d.status.settings.channels == 4, "the operator's channel count is handed back"
    d.close()


def test_the_contact_check_leaves_the_bench_case_alone(monkeypatch):
    """Nothing attached anywhere is the ordinary smoke test, and there is no good reading to protect — so
    narrowing to zero channels, which would leave nothing to test at all, must not happen. A channel that
    reads open on "+" is only *excluded* when some other channel does not."""
    from nfb.device import OPEN_KOHM, Contact

    d, _ = connected(monkeypatch)
    d.start_impedance()
    waitfor(lambda: bool(d.contact), what="the impedance stream never started")
    d.contact.update({c: Contact(plus=OPEN_KOHM, minus=OPEN_KOHM) for c in range(4)})
    assert d.unwired() == [0, 1, 2, 3]
    assert d.narrow_to_wired() == []
    assert d.status.settings.channels == 4
    d.stop()
    d.close()


def test_an_empty_channel_below_a_wired_one_cannot_be_dropped(monkeypatch):
    """Only a *prefix* of channels can be enabled, so ch1 empty with ch2 wired cannot be excluded. It is
    still named by `unwired()`, because a number that cannot be corrected should at least not be trusted
    silently — the panel warns rather than pretending."""
    from nfb.device import OPEN_KOHM, Contact

    d, _ = connected(monkeypatch)
    d.start_impedance()
    waitfor(lambda: bool(d.contact), what="the impedance stream never started")
    d.contact.update({0: Contact(plus=18.3, minus=12.8),
                      1: Contact(plus=OPEN_KOHM, minus=OPEN_KOHM),
                      2: Contact(plus=21.0, minus=12.8),
                      3: Contact(plus=OPEN_KOHM, minus=OPEN_KOHM)})
    assert d.unwired() == [1, 3]
    assert d.narrow_to_wired() == [3], "the trailing one goes; ch1 cannot"
    assert d.status.settings.channels == 3
    # Restarting the check cleared the readings, so re-state what the second pass will find: ch1 is still
    # empty, still enabled, and still named — which is what the panel turns into a warning.
    d.contact.update({0: Contact(plus=18.3, minus=12.8),
                      1: Contact(plus=OPEN_KOHM, minus=OPEN_KOHM),
                      2: Contact(plus=21.0, minus=12.8)})
    assert d.unwired() == [1]
    d.stop()
    d.close()


def test_mains_frequency_is_settable_and_reaches_the_config(monkeypatch):
    """The notch runs in the device's hardware, so this bit has to actually be written."""
    d, fake = connected(monkeypatch)
    field = optima.HEADER_FIELDS["mains_frequency"]
    assert optima.get_field(fake.written[-1], field) == 0, "50 Hz is the default"
    d.reconfigure(mains_hz=60)
    assert optima.get_field(fake.written[-1], field) == 1
    d.reconfigure(mains_hz=50)
    assert optima.get_field(fake.written[-1], field) == 0
    with pytest.raises(ValueError):
        d.reconfigure(mains_hz=55)
    assert d.status.settings.mains_hz == 50
    d.close()


def test_the_high_pass_corner_is_a_real_setting_not_a_label():
    """It was referenced by a warning before it existed, which is the exact failure the UI conventions are meant to
    prevent: copy that tells you to use a control there is no control for.

    The filtering is in the device's hardware, so there is no software equivalent — a band below the corner
    measures the filter's roll-off and nothing downstream can recover it.
    """
    import optima

    from nfb import Settings

    default = Settings()
    assert default.lower_limit_hz == 0.5, "the vendor's own EEG choice"

    slow = Settings(lower_limit_hz=0.01)
    assert slow.build() != default.build(), "it has to reach the config, not just the dataclass"
    assert not optima.validate_config(slow.build())
    assert not optima.validate_config(default.build())


def test_the_device_screen_offers_the_high_pass_corner():
    from nfb import Device, Settings
    from nfb_console import device_panel

    class Page:
        def __init__(self):
            self.services, self.tasks = [], []

        def update(self, *a):
            pass

        def run_task(self, fn, *a):
            self.tasks.append(fn)

    device = Device(Settings())
    control = device_panel.view(Page(), device)

    def texts(ctrl, depth=0):
        import flet as ft
        if isinstance(ctrl, ft.Text):
            return [ctrl.value or ""]
        out = []
        for child in getattr(ctrl, "controls", None) or []:
            out += texts(child, depth + 1)
        content = getattr(ctrl, "content", None)
        if content is not None:
            out += texts(content, depth + 1)
        return out

    shown = texts(control)
    assert "High-pass" in shown
    for corner in device_panel.CORNERS:
        assert f"{corner:g} Hz" in shown

    device.reconfigure(lower_limit_hz=0.01)
    assert device.status.settings.lower_limit_hz == 0.01
    device.close()


def test_the_contact_check_takes_the_channels_off_the_sum(monkeypatch):
    """The SUM entity ties the connected channels together inside the device, and during the impedance test
    that node couples them: a channel with a floating input drags the reading of a channel that has an
    electrode. Measured on hardware, one electrode at C4 against one reference, four channels enabled,
    bipolar wiring — ch0 read +25.41/-41.56 with every channel summed, +5.52/-10.41 with the empty ones off
    the sum, against +5.52/-10.56 with only ch0 enabled. Off-the-sum is exact; summed was out by 4x.

    Applied to the test alone, so the acquisition config stays byte-identical — which this checks in both
    directions, because a test that silently changed what a session measures would be a worse bug than the
    one it fixes.
    """
    field = optima.CHANNEL_FIELDS["sum_disconnected"]
    d, fake = connected(monkeypatch)
    measuring = fake.written[-1]
    assert all(optima.get_field(measuring, field, c) == 0 for c in range(4)), "summed for measurement"

    d.start_impedance()
    waitfor(lambda: fake.written[-1] != measuring, what="the test config was never written")
    testing = fake.written[-1]
    assert all(optima.get_field(testing, field, c) == 1 for c in range(4)), "off the sum for the test"
    assert d.config == measuring, "the decoder still holds the measurement config"

    d.stop()
    assert fake.written[-1] == measuring, "and the channels go back on the sum afterwards"
    d.close()


def test_a_channel_the_device_calls_fine_is_usable(monkeypatch):
    """**The usable signal state is 0, not 1**, and the old `state == 1` made every live channel unusable.

    Measured on hardware with a working electrode at C4 against a linked reference: 19960 samples of real
    EEG arrived with state 0 and one frame with state 2, while three channels with nothing attached reported
    2 throughout. The protocol notes only ever established 2 from hardware — an open input — and the "1 ok"
    written beside it was the vendor manual's prose ordering mistaken for a wire value.

    Nothing in the suite could catch it: this fake sent 1 on every entity, matching the same assumption, and
    the file sources synthesise their own states. So every session from a recording read valid and every
    session from the amplifier read invalid — a screen that works from a file and shows a dead grey trace
    from the device, which is precisely how it was reported.
    """
    from nfb.stream import NO_SIGNAL, OK, OVER_RANGE, Block, ChannelSpec

    spec = ChannelSpec(0, 250.0, "uV", 0.024)
    assert Block(0, 0, (1, 2), spec, state=OK).valid, "0 is a working channel"
    assert not Block(0, 0, (1, 2), spec, state=NO_SIGNAL).valid
    assert not Block(0, 0, (1, 2), spec, state=OVER_RANGE).valid
    assert Block(0, 0, (1, 2), spec, state=1).valid, "unobserved, but not one of the device's complaints"
    assert Block(0, 0, (1, 2), spec).valid, "the default must be a usable channel"

    # and end to end: the fake now sends what the amplifier sends, so a session sees usable channels
    d, fake = connected(monkeypatch)
    sub = d.subscribe("s")
    d.start_measurement()
    seen = []
    for item in sub.blocks(timeout=1.0):
        if not isinstance(item, Gap):
            seen.append(item.valid)
        if len(seen) >= 8:
            break
    d.stop(); sub.close(); d.close()
    assert seen and all(seen), "a live-shaped stream must not read as four dead channels"


def test_a_padded_file_column_reports_no_signal_not_a_fine_channel(tmp_path):
    """`SampleFileSource` marked an all-zero column with state 0 to mean "inactive". That is the device's
    code for a channel that is **fine**, so once `valid` was corrected a padded column would have read as a
    working channel sitting at a perfectly flat 0 µV — which is the exact confusion the marking exists to
    prevent. It now uses the device's own code for an open input."""
    from nfb.sources import SampleFileSource
    from nfb.stream import NO_SIGNAL, OK

    path = tmp_path / "export.txt"
    rows = ["Sampling rate: 256"]
    for i in range(600):
        rows.append(f"{i % 7 - 3}\t0")          # one live column, one padded with zeros
    path.write_text("\n".join(rows) + "\n")

    # `realtime=False`: this is about what the source *marks* a padded column, not about how fast it
    # replays one. Left at the default it paced 600 rows at 256 Hz and spent 2.3 s asleep.
    src = SampleFileSource(str(path), loop=False, realtime=False)
    states = {}
    for item in src.blocks():
        states.setdefault(item.channel, set()).add(item.state)
    assert states[0] == {OK}, "a column with signal is a usable channel"
    assert states[1] == {NO_SIGNAL}, "a padded column is an open input, not a flat working one"


def test_a_session_will_not_subscribe_to_the_impedance_stream(monkeypatch):
    """Impedance testing is an exclusive mode, and a session that assumed the device was idle silently
    subscribed to the *test* stream when it was not. Reproduced on hardware: the session came up with one
    channel at 3.90625 Hz, 22 of 320 trace columns filled in six seconds, and not one target ever fed — a
    flat trace with an occasional spike and no feedback, which is how it was reported. Reaching that state
    is ordinary: leave the device screen for a protocol without stopping the contact check.
    """
    from nfb_console import amplitude_view
    from nfb_console.protocols import BY_CODE

    d, _ = connected(monkeypatch)
    d.start_impedance()
    assert d.status.mode is Mode.TESTING
    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], device=d, channels=0)
    try:
        assert d.status.mode is Mode.MEASURING, "the session must end the test, not join it"
        rates = {spec.rate_hz for spec in session.specs.values()}
        assert rates == {1000.0}, f"subscribed at the test rate: {rates}"
    finally:
        session.stop()
        d.stop()
        d.close()


def test_a_channel_not_heard_from_yet_is_neither_good_nor_bad(monkeypatch):
    """`state` was `False` for every channel before the first block, so "not known yet" was indistinguishable
    from "known bad" — a screen said "no signal, check the electrode" before any data existed. It also
    contradicted `target_is_live`, whose rule is that exclusion needs *evidence*: with every key present and
    False, every target was dropped from the feedback until its first block arrived."""
    from nfb_console import amplitude_view
    from nfb_console.protocols import BY_CODE

    d, _ = connected(monkeypatch)
    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], device=d, channels=2)
    try:
        assert set(session.state.values()) == {None}, "unknown, not bad"
        assert session.active_targets == session.targets, "an unheard channel still counts as live"
        session.state[0] = False
        assert all(not session.target_is_live(t) for t in session.targets if t.source == 0)
    finally:
        session.stop()
        d.stop()
        d.close()


def test_an_unplug_while_idle_is_noticed(monkeypatch, tmp_path):
    """Nothing touched the port while idle — the read loop is the only thing that does, and it runs only
    while streaming. So the device screen kept offering "Disconnect" beside the model and serial of a device
    that was no longer plugged in, and the first sign was a session that would not start.

    Two checks, because neither catches both failures: the port node disappears on an unplug, and it
    *survives* the device powering itself off, since the FTDI module is USB-powered and stays enumerated —
    only asking the device something catches that one.
    """
    node = tmp_path / "cu.usbserial-FAKE"
    node.write_text("")
    d, fake = connected(monkeypatch)
    d.status.port = str(node)

    assert d.check_alive() is True
    assert optima.CMD["GET_DEV_STATE"] in fake.sent, "it asks the device, it does not only stat the node"

    node.unlink()                       # the cable comes out
    assert d.check_alive() is False
    assert d.status.mode is Mode.DISCONNECTED
    assert "is gone" in d.status.failure
    assert d.status.remedy == "plug the USB cable back in"


def test_a_powered_off_device_whose_node_remains_is_noticed(monkeypatch, tmp_path):
    """The switch-off case: `/dev/cu.usbserial-*` is still there because the FTDI module is USB-powered, so
    the node check passes and only the round trip fails. This is the replug case,
    and its remedy has to say so."""
    node = tmp_path / "cu.usbserial-FAKE"
    node.write_text("")
    d, fake = connected(monkeypatch)
    d.status.port = str(node)

    def dead(code, payload=b""):
        raise OSError(22, "Invalid argument")

    fake.request = dead
    assert d.check_alive() is False
    assert d.status.mode is Mode.DISCONNECTED
    assert "replug" in d.status.remedy and "switch the device on" in d.status.remedy


def test_liveness_is_not_checked_mid_stream(monkeypatch):
    """A command injected into a running stream would desync the framing, and while streaming the read loop
    is already the detector — it calls `_fail` and ends every subscriber's epoch."""
    d, fake = connected(monkeypatch)
    d.start_measurement()
    before = list(fake.sent)
    assert d.check_alive() is True
    assert fake.sent == before, "it must not send anything while streaming"
    d.stop()
    d.close()


def test_the_channel_count_reaches_the_device_and_is_not_swallowed(monkeypatch):
    """Marking "1 ch" has to reconfigure the *device*, not only trim what the screen draws.

    It did not. A session holds the device measuring even while paused, so `reconfigure` was refused and
    the refusal was caught and discarded — the device went on streaming four channels while the screen
    showed one. Verified on hardware: with four enabled and one electrode, channels 1-3 report state 2 and
    the device's own indicator goes red; with one enabled, nothing reports no signal at all.
    """
    import flet as ft

    from nfb_console.app import Console, build_parser
    from nfb_console.protocols import BY_CODE

    class StubPage:
        width = 1240

        def __init__(self):
            self.controls, self.services, self.tasks, self.dialogs = [], [], [], []
            self.window = type("W", (), {})()

        def show_dialog(self, d): self.dialogs.append(d)
        def pop_dialog(self): self.dialogs.pop()
        def add(self, c): self.controls.append(c)
        def update(self, *a): pass
        def run_task(self, fn, *a): self.tasks.append(fn)

    fake = FakeOptima()
    monkeypatch.setattr(optima, "Optima", lambda *a, **k: fake)
    console = Console(StubPage(), build_parser().parse_args(["--live"]))
    assert console.device.status.connected

    console.pick(BY_CODE["AMP"])
    assert console.device.status.mode is Mode.MEASURING, "a session holds the device measuring"
    console.session.toggle_pause()               # paused, and still holding it

    console.set_channels(1)
    assert console.device.status.settings.channels == 1, "the count never reached the device"
    written = fake.written[-1]
    enabled = [c.index for c in optima.plan_channels(written) if c.enabled]
    assert enabled == [0], f"the device is still streaming {enabled}"
    console.session.stop()
    console.device.close()


def test_a_protocol_screen_opens_with_no_amplifier_and_waits(monkeypatch):
    """`--live` with the amplifier off: the screen opens, waiting, and the red chip says why.

    Pressing a protocol card used to raise `config blob too short: 0 bytes` out of the framing layer,
    because `Device.config` is empty until a connection writes one and nothing between the card and the
    decoder checked. Two things fix it and a third was tried and rejected: `Device.subscribe` refuses
    with the failure and its remedy, and the session falls back to a `WaitingSource` that knows the
    channel layout from the settings and emits nothing. Rejected: refusing the press, whether by
    redirecting to the device screen or greying the cards out — the chip is already red and already
    carries the remedy, so that removes the screen in exchange for what was on it.
    """
    from nfb_console.app import Console, build_parser
    from nfb_console.protocols import BY_CODE

    class StubPage:
        width = 1240

        def __init__(self):
            self.controls, self.services, self.tasks, self.dialogs = [], [], [], []
            self.window = type("W", (), {})()

        def show_dialog(self, d): self.dialogs.append(d)
        def pop_dialog(self): self.dialogs.pop()
        def add(self, c): self.controls.append(c)
        def update(self, *a): pass
        def run_task(self, fn, *a): self.tasks.append(fn)

    def refuse(*a, **k):
        raise FileNotFoundError("no FTDI device found")

    monkeypatch.setattr(optima, "Optima", refuse)
    console = Console(StubPage(), build_parser().parse_args(["--live"]))
    assert not console.device.status.connected
    assert console.device.status.remedy, "a failure has to carry its remedy"

    with pytest.raises(RuntimeError) as exc:
        console.device.subscribe("amp")
    assert console.device.status.remedy in str(exc.value)

    console.pick(BY_CODE["AMP"])                 # must not raise
    assert console.session is not None, "the screen opens; the chip says there is no device"
    assert console.toolbar.crumb.value == "Amplitude", "and it must not send you somewhere else"
    assert isinstance(console.session.source, WaitingSource)
    assert list(console.session.specs) == [0, 1, 2, 3], "the layout is known from the settings"
    assert all(v is None for v in console.session.state.values()), (
        "not heard from, which is neither good nor bad")
    console.session.redraw()                     # a waiting screen has to draw
    console.session.stop()


def test_recording_can_be_made_at_the_infra_low_corner_and_carries_it():
    """`nfb record` could set the rate, the reference and the notch but not the corner, so **every
    recording in this repo was captured at 0.5 Hz** and nothing depending on that could be tested.

    Two claims, and the second is the one that makes the flag worth having: the flag reaches the config,
    and the corner survives into the session header — the blob is what a replay decodes from, so a file
    captured at 0.01 Hz replays at 0.01 Hz rather than at whatever the reader assumes.
    """
    import optima

    from nfb.cli import main
    from nfb.record import SessionHeader

    field = optima.CHANNEL_FIELDS["lower_limit_frequency"]
    seen: dict[str, bytes] = {}

    class Capture:
        def __init__(self, cfg, **kwargs):
            seen["cfg"] = cfg
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def channels(self): return {}
        def blocks(self): return iter(())

    import nfb.cli as cli
    for corner, code in ((0.5, 0), (0.01, 2)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cli, "OptimaSource", Capture)
            main(["record", "out.jsonl", "--high-pass", str(corner), "--channels", "2"])
        cfg = seen["cfg"]
        assert optima.get_field(cfg, field, 0) == code, f"--high-pass {corner} has to reach the config"
        assert not optima.validate_config(cfg)
        # Through a header and back, because that is the path a replay actually takes.
        back = SessionHeader.from_json(SessionHeader(config=cfg).to_json()).config
        assert optima.get_field(back, field, 0) == code, "and survive being written to a session file"


def test_the_devices_own_beeper_is_switched_off():
    """`silent_mode` was recovered with the rest of the header's General group and then never written, so
    the amplifier kept beeping on its own — at every mode change, which means stopping a session beeped.

    Pinned because that is the failure class: a field decoded, documented and left at zero looks finished
    from the code and is not. The host owns every sound a session makes (a beep,
    a click and an alert are three sounds with three jobs) and a beeper inside the amplifier is a fourth
    that no screen describes and `NFB_NO_OUTPUTS` cannot reach.
    """
    import optima

    from nfb import Settings

    field = optima.HEADER_FIELDS["silent_mode"]
    assert optima.get_field(optima.build_eeg_config(), field) == 1, "silent by default"
    assert optima.get_field(optima.build_eeg_config(silent=False), field) == 0, "and switchable"
    # Both are legal configs — the polarity of the bit is inferred from the vendor's field name and is the
    # one thing here no measurement settles, so the escape hatch has to actually work.
    assert optima.validate_config(optima.build_eeg_config(silent=False)) == []
    # It has to survive the paths that rebuild a config from settings, or a live session beeps anyway.
    assert optima.get_field(Settings().build(), field) == 1
    assert optima.get_field(optima.contact_check_config(Settings().build()), field) == 1, \
        "the impedance check changes channels, not the header"


def test_a_backed_up_queue_serves_the_present_not_the_past(monkeypatch):
    """Overflow evicts the *oldest* frame. Dropping the newest — which this did — meant a consumer that
    once fell behind was served an ever-older stream for the rest of the session, up to the queue's whole
    depth in the past, with nothing on screen saying so. The eviction leaves a hole in the phase counter,
    so the loss surfaces as a Gap through the machinery that already exists for one. And the EpochEnd
    sentinel is never the thing evicted: losing it left an iterator waiting out its timeout on an epoch
    that had already ended."""
    d, _ = connected(monkeypatch)
    sub = d.subscribe("slow", queue_size=4)
    for n in range(10):
        sub._offer(0x40, bytes([n]))
    assert sub.dropped == 6
    queued = [sub.queue.get_nowait() for _ in range(sub.queue.qsize())]
    assert [p[0] for _, p in queued] == [6, 7, 8, 9], "the newest frames survive, not the oldest"

    for n in range(4):
        sub._offer(0x40, bytes([n]))
    sub._end_epoch()
    queued = [sub.queue.get_nowait() for _ in range(sub.queue.qsize())]
    assert any(isinstance(item, type(queued[-1])) for item in queued)
    from nfb.device import EpochEnd

    assert isinstance(queued[-1], EpochEnd), "the sentinel survives a full queue"
    d.close()
