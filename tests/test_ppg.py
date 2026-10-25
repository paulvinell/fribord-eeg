"""The HeartMath device's own protocol, and the beats we find in it.

The Bluetooth path takes intervals the device has already worked out. This path takes the raw pulse
waveform and finds the beats here, so the detector is ours and has to be held to a number: a synthetic
waveform with known beat times, scored for misses and false positives. The two failure modes below —
counting the dicrotic notch, and locking out when the pulse shrinks — are the ones that were measured
and fixed, and each has a test so a tuning change cannot quietly undo it.
"""

from __future__ import annotations

import math
import random
import threading
import time

from nfb.hrv.beats import Beat
from nfb.hrv.ppg import (
    ALLON_BODY_LEN, BIOMETRICS_ALLON, HEADER, PACKET_BIOMETRICS, PPG_RATE_HZ, QUERY_BATTERY,
    START_PPG, STOP_PPG, FrameSplitter, PeakDetector, PpgDecoder, battery_level, command, crc,
    parse_ppg,
)

FS = PPG_RATE_HZ


# --- the wire format -------------------------------------------------------------------------

def test_the_commands_are_the_bytes_confirmed_on_hardware():
    """These three were confirmed against hardware. If a
    change to the frame builder moves any of them, it has broken the protocol, not the tests."""
    assert START_PPG == bytes.fromhex("AB020208F4")
    assert STOP_PPG == bytes.fromhex("AB020205F7")
    assert QUERY_BATTERY == bytes.fromhex("AB020701F6")


def ppg_frame(samples: list[int]) -> bytes:
    body = bytes([PACKET_BIOMETRICS, BIOMETRICS_ALLON])
    for s in samples:
        body += bytes([(s >> 8) & 0xFF, s & 0xFF])
    with_len = bytes([ALLON_BODY_LEN]) + body
    return bytes([HEADER]) + with_len + bytes([crc(with_len)])


def test_a_pulse_frame_decodes_big_endian_and_keeps_every_sample():
    """Reading one sample of the sixteen little-endian one byte along gives
    the same number for a pair in the middle of the buffer and a different one at the end — where the
    little-endian read would take in the CRC. Sixteen samples, big-endian, is the reading that holds
    everywhere."""
    values = [1362 + 7 * i for i in range(16)]
    frame = ppg_frame(values)
    assert parse_ppg(frame) == values
    assert len(frame) == 3 + ALLON_BODY_LEN


def test_anything_that_is_not_a_pulse_frame_comes_back_as_none():
    assert parse_ppg(command(0x03, 0xA1, b"\x00")) is None      # an acknowledgement
    assert parse_ppg(START_PPG) is None
    assert parse_ppg(b"") is None


def test_a_battery_reply_is_read_and_a_nonsense_one_is_not():
    assert battery_level(command(0x07, 0x01, bytes([100]))) == 100
    assert battery_level(command(0x07, 0x01, bytes([200]))) is None
    assert battery_level(START_PPG) is None


def test_the_splitter_survives_arbitrary_chunking_and_a_corrupt_frame():
    """A serial read returns whatever was in the operating system's buffer, so frames arrive split and
    glued. A bad CRC costs its own frame and nothing else."""
    good = [ppg_frame([i] * 16) for i in range(3)]
    corrupt = bytearray(ppg_frame([9] * 16))
    corrupt[-1] ^= 0xFF
    stream = good[0] + bytes(corrupt) + good[1] + good[2]

    whole = FrameSplitter()
    assert whole.feed(stream) == [good[0], good[1], good[2]]
    assert whole.bad_crc == 1

    byte_at_a_time = FrameSplitter()
    out = []
    for i in range(len(stream)):
        out += byte_at_a_time.feed(stream[i:i + 1])
    assert out == [good[0], good[1], good[2]]
    assert byte_at_a_time.bad_crc == 1


def test_junk_before_a_header_is_dropped_rather_than_ending_the_stream():
    splitter = FrameSplitter()
    frame = ppg_frame([100] * 16)
    assert splitter.feed(b"\x00\x11\x22" + frame) == [frame]
    assert splitter.dropped_bytes == 3


# --- the detector ----------------------------------------------------------------------------

def waveform(seconds: float = 60.0, hr: float = 60.0, rsa_s: float = 0.06, amp: float = 400.0,
             amp_at=None, dicrotic: float = 0.0, drift: float = 0.0, noise: float = 2.0,
             seed: int = 1) -> tuple[list[int], list[float]]:
    """A synthetic pulse waveform and the beat times that generated it.

    `dicrotic` adds the secondary pressure peak 0.35 s after each beat, which is the thing a fixed
    refractory period counts as a second beat. `amp_at(t)` scales the pulse height, for the clip coming
    loose partway through.
    """
    rng = random.Random(seed)
    beats: list[float] = []
    t = phase = 0.0
    while t < seconds:
        rr = max(0.3, 60.0 / hr + rsa_s * math.sin(2 * math.pi * 0.1 * phase))
        phase += rr
        t += rr
        beats.append(t)

    out: list[int] = []
    for i in range(int(seconds * FS)):
        t = i / FS
        v = 2000.0 + (drift * math.sin(2 * math.pi * 0.05 * t) if drift else 0.0)
        a = amp * (amp_at(t) if amp_at else 1.0)
        for b in beats:
            d = t - b
            if -0.05 < d < 0.6:
                v += a * math.exp(-(d ** 2) / (2 * 0.045 ** 2))
                if dicrotic:
                    v += a * dicrotic * math.exp(-((d - 0.35) ** 2) / (2 * 0.06 ** 2))
        out.append(int(max(0, min(65535, v + rng.gauss(0.0, noise)))))
    return out, beats


def score(samples: list[int], truth: list[float], detector=None,
          tol: float = 0.15) -> tuple[float, float]:
    """(missed, false positive) as fractions of the true beats."""
    detector = detector or PeakDetector(rate_hz=FS)
    found = [r[0] for i, s in enumerate(samples) if (r := detector.push(i / FS, s)) is not None]
    remaining = list(truth)
    matched = spurious = 0
    for at in found:
        near = min(remaining, key=lambda b: abs(b - at), default=None)
        if near is not None and abs(near - at) <= tol:
            remaining.remove(near)
            matched += 1
        else:
            spurious += 1
    return 1.0 - matched / len(truth), spurious / len(truth)


def test_a_clean_pulse_waveform_gives_back_the_beats_that_made_it():
    samples, truth = waveform(seconds=60.0, hr=62.0)
    missed, spurious = score(samples, truth)
    assert missed < 0.03, f"missed {missed:.1%}"
    assert spurious < 0.03, f"{spurious:.1%} spurious"


def test_the_dicrotic_notch_is_not_counted_as_a_beat():
    """The pulse waveform's secondary peak lands about 0.35 s after the main one — just outside a fixed
    0.30 s refractory period, so every beat gets counted twice. And the error sustains itself: doubling
    the rate halves the mean interval, which shrinks a proportional refractory, which keeps admitting the
    notch. Measured, the detector locked onto exactly double the true rate and never recovered."""
    samples, truth = waveform(seconds=60.0, hr=60.0, dicrotic=0.55)
    missed, spurious = score(samples, truth)
    assert spurious < 0.05, f"{spurious:.1%} spurious — the notch is being counted"
    assert missed < 0.05, f"missed {missed:.1%}"


def test_the_detector_recovers_when_the_pulse_gets_smaller():
    """The amplitude floor is a fraction of the height of recently accepted beats, and rejected peaks
    never update it — so a clip coming loose could lock the detector out permanently. Measured, a step
    down to 30% lost every subsequent beat. The floor decays once a beat is overdue, so it re-acquires."""
    samples, truth = waveform(seconds=60.0, hr=60.0, amp_at=lambda t: 1.0 if t < 30.0 else 0.3)
    late = [b for b in truth if b > 35.0]
    detector = PeakDetector(rate_hz=FS)
    found = [r[0] for i, s in enumerate(samples) if (r := detector.push(i / FS, s)) is not None]
    caught = sum(1 for b in late if any(abs(f - b) <= 0.15 for f in found))
    assert caught / len(late) > 0.9, f"only {caught}/{len(late)} beats after the drop"


def test_baseline_drift_does_not_move_the_beats():
    """300 counts of 0.05 Hz wander against a 400-count pulse, which is the figure the detector was
    tuned against. It is not unbounded: the baseline average lags by roughly its own time constant
    times the slope, so drift steep enough to move the baseline by a pulse height within 1.5 s does
    start costing beats."""
    samples, truth = waveform(seconds=60.0, hr=60.0, drift=300.0)
    missed, spurious = score(samples, truth)
    assert missed < 0.05 and spurious < 0.05


# --- the decoder -----------------------------------------------------------------------------

def test_beats_come_out_of_a_byte_stream_at_the_rate_that_went_in():
    """End to end without a device: build the waveform, frame it as the device would, and feed it in
    chunks that do not respect frame boundaries."""
    samples, truth = waveform(seconds=40.0, hr=66.0)
    decoder = PpgDecoder(warm_up_samples=0)
    stream = b"".join(ppg_frame(samples[i:i + 16]) for i in range(0, len(samples) - 15, 16))

    beats: list[Beat] = []
    t = 1000.0
    step = 700
    for i in range(0, len(stream), step):
        chunk = stream[i:i + step]
        # Sixteen samples per frame at 500 Hz; a chunk of `step` bytes is that many frames' worth.
        t += len(chunk) / (3 + ALLON_BODY_LEN) * 16 / FS
        beats += decoder.feed(chunk, t)

    assert decoder.bad_crc == 0
    rate = 60.0 / (sum(b.rr_intervals[0] for b in beats) / len(beats))
    assert abs(len(beats) - len(truth)) <= 2, f"{len(beats)} beats against {len(truth)}"
    assert abs(rate - 66.0) < 1.5, f"{rate:.1f} bpm"
    assert all(b.contact is None for b in beats), "a pulse sensor has no contact flag to report"


def test_sample_times_come_from_arrivals_so_a_wrong_rate_cannot_drift():
    """The spacing is derived from when chunks actually arrive, not from the nominal rate — which is what
    keeps a slightly wrong constant from accumulating. The old 312 Hz figure stretched the timeline and
    reported 37 bpm for a 60 bpm signal, so this feeds a real 500 Hz stream to a decoder told 400."""
    samples, _ = waveform(seconds=40.0, hr=60.0)
    decoder = PpgDecoder(rate_hz=400.0, warm_up_samples=0)
    beats: list[Beat] = []
    t = 1000.0
    for i in range(0, len(samples) - 15, 16):
        t += 16 / FS
        beats += decoder.feed(ppg_frame(samples[i:i + 16]), t)
    rate = 60.0 / (sum(b.rr_intervals[0] for b in beats) / len(beats))
    assert abs(rate - 60.0) < 1.5, f"{rate:.1f} bpm from a decoder told the wrong rate"


def test_a_gap_in_the_stream_re_anchors_instead_of_smearing_samples_across_it():
    """A stall or a reconnect leaves a hole no spacing bridges honestly. Sixteen samples must not be
    spread across ten seconds of silence, because the detector would then see intervals no heart has."""
    decoder = PpgDecoder(warm_up_samples=0)
    frame = ppg_frame(list(range(16)))
    decoder.feed(frame, 1000.0)
    decoder.feed(frame, 1010.0)                 # ten seconds later
    stamps = [t for t, _ in decoder.raw][-16:]
    assert stamps[-1] == 1010.0
    assert abs((stamps[-1] - stamps[0]) - 15 / FS) < 1e-9, "the run is anchored at the arrival"


def test_the_warm_up_samples_are_dropped():
    decoder = PpgDecoder(warm_up_samples=32)
    for i in range(4):
        decoder.feed(ppg_frame([500] * 16), 1000.0 + i * 16 / FS)
    assert len(decoder.raw) == 2 * 16, "the first two frames are settle time"


# --- finding the cable -----------------------------------------------------------------------

class FakePort:
    def __init__(self, device, vid, description="", manufacturer=None, product=None,
                 serial_number=None):
        self.device, self.vid = device, vid
        self.description, self.manufacturer = description, manufacturer
        self.product, self.serial_number = product, serial_number


def test_the_amplifier_is_never_offered_as_a_pulse_sensor(monkeypatch):
    """**Both devices are FTDI serial bridges on /dev/cu.usbserial-***, so a vendor-id match alone would
    sometimes hand the heart screen the amplifier — and opening a serial port is exclusive, so that is
    the console losing its EEG device and writing pulse-sensor commands at it. The product string is the
    one thing that separates them."""
    from nfb.hrv import sources

    ports = [FakePort("/dev/cu.usbserial-FT000001", 0x0403, "Neurobit Optima+ 4 USB"),
             FakePort("/dev/cu.usbserial-HM01", 0x0403, "FT230X Basic UART"),
             FakePort("/dev/cu.Bluetooth-Incoming-Port", None, "n/a")]
    monkeypatch.setattr(sources, "usb_candidates",
                        lambda: [p for p in ports if p.vid == 0x0403 and not sources.is_amplifier(p)])
    port, trouble = sources.find_usb_port()
    assert port == "/dev/cu.usbserial-HM01" and trouble is None


def test_the_missing_cable_message_names_what_is_on_usb_and_nothing_else(monkeypatch):
    """A Mac lists serial ports that are nothing of the kind — a debug console, an incoming-Bluetooth
    port, every set of headphones ever paired. The first version of this message recited all four, which
    wrapped to two lines and said nothing about whether a cable was plugged in."""
    from nfb.hrv import sources

    monkeypatch.setattr(sources, "usb_candidates", lambda: [])
    from serial.tools import list_ports

    monkeypatch.setattr(list_ports, "comports",
                        lambda: [FakePort("/dev/cu.Bluetooth-Incoming-Port", None, "n/a"),
                                 FakePort("/dev/cu.JBLTUNE570BT", None, "n/a"),
                                 FakePort("/dev/cu.usbserial-FT000001", 0x0403,
                                          "Neurobit Optima+ 4 USB")])
    port, trouble = sources.find_usb_port()
    assert port is None
    assert "JBL" not in trouble and "Bluetooth-Incoming" not in trouble
    assert "Neurobit Optima+ 4 USB" in trouble, "what is on USB is worth naming"
    assert "skipped on purpose" in trouble, "including why the one that is there was not taken"


def test_choosing_the_cable_does_not_report_the_same_fault_twice():
    """The screen has two places a heart-rate fault can appear: a note under the source buttons, and a
    status line under the port field. Both were being told there was no cable — the choice by going and
    looking, the runner by trying — so the same two-line sentence appeared above and below the field.
    Looking for a cable is not something to know in advance; the runner finds out and says so."""
    from nfb_console.inputs import HeartChoice

    assert HeartChoice(kind="usb").problem() is None
    assert HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01").problem() is None


def test_the_missing_cable_is_reported_by_whoever_tried_to_open_it(monkeypatch):
    """The other half of not saying it twice: with the choice staying quiet, the fault still has to
    arrive — from the runner, on its own thread, and be the sentence the status line shows."""
    from nfb.hrv import sources
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    monkeypatch.setattr(sources, "usb_candidates", lambda: [])
    runner = HeartRunner(HeartChoice(kind="usb"))
    assert runner.attach() is True, "nothing is known to be wrong until it is tried"
    deadline = time.monotonic() + 2.0
    while runner.error is None and time.monotonic() < deadline:
        time.sleep(0.02)
    runner.release()
    assert runner.error and "HeartMath cable" in runner.error
    assert runner.describe() == runner.error, "and it is what the screen shows"


def test_switching_source_does_not_wait_for_the_old_one_on_the_event_loop(monkeypatch):
    """Releasing a source waits up to two seconds for its reader thread, and a reader halfway through a
    twenty-second Bluetooth scan takes all of it. Doing that in the click handler froze every control on
    the page, which is the Flet rule in reverse: the event loop must not be blocked, not merely written
    to from elsewhere."""
    import threading

    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import HeartChoice

    class SlowRunner:
        """Releases only when the test says so — which is the point: the click must return while this is
        still blocked. A `time.sleep` here would have tested the same thing at the cost of half a second,
        and would have been a race on a loaded machine rather than a fact."""

        error = None

        def __init__(self):
            self.entered = threading.Event()
            self.may_finish = threading.Event()
            self.released = threading.Event()

        def release(self):
            self.entered.set()
            self.may_finish.wait(5.0)
            self.released.set()

        def attach(self, _choice=None):
            return True

        def describe(self):
            return "…"

    class Page:
        def __init__(self):
            self.services, self.tasks = [], []

        def update(self, *a):
            pass

        def run_task(self, fn, *a):
            self.tasks.append(fn)

    device = Device(Settings())
    runner = SlowRunner()
    choice = HeartChoice(kind="off")
    control = device_panel.view(Page(), device, heart_choice=choice, heart_runner=runner)

    _press(control, "usb")
    assert runner.entered.wait(2.0), "the swap started"
    assert choice.kind == "usb", "and the click had already returned, with the release still blocked"
    runner.may_finish.set()
    assert runner.released.wait(2.0), "the swap completed, just not on the event loop"
    device.close()


def _press(control, label: str) -> None:
    """Click the source button carrying `label`, the way the operator does."""
    import flet as ft

    def walk(ctrl):
        if (isinstance(ctrl, ft.Container) and isinstance(ctrl.content, ft.Text)
                and ctrl.content.value == label and ctrl.on_click is not None):
            return ctrl
        for child in list(getattr(ctrl, "controls", None) or []) + [getattr(ctrl, "content", None)]:
            if child is not None and (hit := walk(child)) is not None:
                return hit
        return None

    button = walk(control)
    assert button is not None, f"no button reading {label!r}"
    button.on_click(None)


def test_a_chosen_source_keeps_trying_until_the_cable_is_plugged_in(monkeypatch):
    """Choosing a source is the operator saying what is on their body, not what is plugged in this
    second. A cable connected a minute later, or pulled out and put back, must not need a trip to the
    device screen to click the same button again."""
    from nfb.hrv import sources
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    plugged: list = []
    opened: list = []
    monkeypatch.setattr(sources, "usb_candidates", lambda: list(plugged))
    monkeypatch.setattr(HeartRunner, "RETRY_SECONDS", 0.1)

    class Nothing:
        """A port that is not there: the reader records why and ends, as the real one does."""

        def __init__(self, port=None):
            self.error = None
            self.port = port

        def beats(self):
            port, trouble = sources.find_usb_port()
            if port is None:
                self.error = trouble
                return iter(())
            opened.append(port)
            return iter((Beat(t=time.time(), hr_bpm=60.0, rr_intervals=[1.0]),))

        def close(self):
            pass

        def describe(self):
            return self.error or "reading"

    monkeypatch.setattr(sources, "UsbPpgBeats", Nothing)
    monkeypatch.setattr("nfb.hrv.beat_source",
                        lambda kind, **kw: Nothing(**kw) if kind == "usb" else None)

    runner = HeartRunner(HeartChoice(kind="usb"))
    runner.attach()
    deadline = time.monotonic() + 2.0
    while runner.error is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert "trying again" in (runner.error or ""), "it says it has not given up"

    plugged.append(FakePort("/dev/cu.usbserial-HM01", 0x0403, "FT230X Basic UART"))
    deadline = time.monotonic() + 2.0
    while not opened and time.monotonic() < deadline:
        time.sleep(0.02)
    runner.release()
    assert opened == ["/dev/cu.usbserial-HM01"], "it connected itself once the cable was there"


def test_the_source_row_keeps_up_with_a_connection_it_did_not_click(monkeypatch):
    """The row is repainted when something is clicked, and connecting happens on another thread — so
    after the click that *starts* a connection it read "not started", the state of a millisecond
    earlier, and stayed that way because nothing on the device screen refreshes on a timer unless the
    contact check is running."""
    import asyncio

    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import HeartChoice

    class LateRunner:
        error = None

        def __init__(self):
            self.said = "not started"

        def release(self):
            pass

        def attach(self, _choice=None):
            return True

        def describe(self):
            return self.said

    class Page:
        def __init__(self):
            self.services, self.tasks, self.updates = [], [], 0

        def update(self, *a):
            self.updates += 1

        def run_task(self, fn, *a):
            self.tasks.append((fn, a))

    page = Page()
    device = Device(Settings())
    runner = LateRunner()
    control = device_panel.view(page, device, heart_choice=HeartChoice(kind="usb", port="/dev/x"),
                                heart_runner=runner)
    assert any("not started" in x for x in _texts(control))

    # The tick interval is the loop's, not the test's: with it at zero the loop runs as fast as the event
    # loop will schedule it, and a hundredth of a second is dozens of iterations. Waiting a real second
    # for a real 0.25 s tick was measuring the sleep.
    monkeypatch.setattr(device_panel, "POLL_SECONDS", 0)

    async def briefly():
        poll = next(fn for fn, _ in page.tasks)
        runner.said = "reading the pulse waveform on /dev/x"
        try:
            await asyncio.wait_for(poll(), 0.01)
        except (asyncio.TimeoutError, TimeoutError):
            pass

    asyncio.run(briefly())
    assert any("reading the pulse waveform" in x for x in _texts(control)), \
        "the row followed the connection with nothing clicked"
    device.close()


def test_the_hrv_screen_trains_on_the_cable_the_device_screen_chose(monkeypatch):
    """The whole path, from the source picked on the device screen to a running session: the screen asks
    the shared runner for the chosen source, the beats arrive, and the transport starts."""
    from nfb.hrv import SimulatedBeats
    from nfb_console import hrv_view
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    opened: list = []

    def fake_source(kind, **kw):
        opened.append((kind, kw))
        return SimulatedBeats(hr_bpm=60.0, realtime=False, seed=3)

    monkeypatch.setattr("nfb.hrv.beat_source", fake_source)

    choice = HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01")
    runner = HeartRunner(choice)
    session = hrv_view.session_for(BY_CODE["HRV"], heart_choice=choice, heart_runner=runner)
    assert session.choice is choice, "the screen trains on what was chosen, not an invented simulator"

    # **Two claims, and only one of them is about a thread.** That the screen opens the chosen source is
    # `run()`'s doing and is checked the moment it returns. That beats reach the session is the
    # processor's, and pushing them is instant where waiting for a real-time simulator to produce three
    # was three seconds of sleeping — for a fact about wiring, not about timing.
    session.run()
    assert opened and opened[0][0] == "usb" and opened[0][1]["port"] == "/dev/cu.usbserial-HM01"
    runner.release()
    beats = SimulatedBeats(hr_bpm=60.0, realtime=False, seed=3).beats()
    while not runner.ready:
        runner.state = runner.processor.push(next(beats))
    runner.seen_at = time.monotonic()
    assert session.dead_channels() == [], "so the screen has its source"

    assert not session.running, "a session opens paused"
    session.toggle_pause()
    assert session.running, "and the transport starts it"
    session.stop()


def _texts(ctrl) -> list[str]:
    import flet as ft

    if isinstance(ctrl, ft.Text):
        return [ctrl.value or ""]
    out: list[str] = []
    for child in getattr(ctrl, "controls", None) or []:
        out += _texts(child)
    content = getattr(ctrl, "content", None)
    if content is not None:
        out += _texts(content)
    return out


def test_two_cables_it_could_be_means_asking_rather_than_guessing(monkeypatch):
    from nfb.hrv import sources

    ports = [FakePort("/dev/cu.usbserial-A", 0x0403, "FT230X Basic UART"),
             FakePort("/dev/cu.usbserial-B", 0x0403, "FT231X USB UART")]
    monkeypatch.setattr(sources, "usb_candidates", lambda: ports)
    port, trouble = sources.find_usb_port()
    assert port is None
    assert "name the port" in trouble and "/dev/cu.usbserial-B" in trouble


def test_the_amplifier_prefers_the_port_that_says_it_is_one(monkeypatch):
    """The other half of the same collision: the amplifier used to take the first
    /dev/cu.usbserial-* alphabetically, which is the pulse sensor's cable often enough to matter."""
    import optima.device as device

    monkeypatch.setattr(device.list_ports, "comports", lambda: [
        FakePort("/dev/cu.usbserial-AAAA", 0x0403, "FT230X Basic UART"),
        FakePort("/dev/cu.usbserial-FT000001", 0x0403, "Neurobit Optima+ 4 USB"),
    ])
    assert device.find_port() == "/dev/cu.usbserial-FT000001"


# --- the console's own view of it ------------------------------------------------------------

def test_the_heart_source_choice_offers_the_cable():
    from nfb_console.inputs import HeartChoice

    choice = HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01")
    assert choice.on
    assert "/dev/cu.usbserial-HM01" in choice.describe()
    assert choice.problem() is None, "a named port needs nothing found"
    source = choice.open()
    assert type(source).__name__ == "UsbPpgBeats" and source.port == "/dev/cu.usbserial-HM01"


def test_the_device_screen_offers_the_cable_beside_the_strap():
    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import HeartChoice

    class Page:
        def __init__(self):
            self.services, self.tasks = [], []

        def update(self, *a):
            pass

        def run_task(self, fn, *a):
            self.tasks.append(fn)

    device = Device(Settings())
    # A named port, because with none named the note under the source row is replaced by the reason
    # nothing was found — which is the right thing for it to say, and not what this is checking.
    control = device_panel.view(Page(), device,
                                heart_choice=HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01"))

    def texts(ctrl):
        import flet as ft
        if isinstance(ctrl, ft.Text):
            return [ctrl.value or ""]
        out = []
        for child in getattr(ctrl, "controls", None) or []:
            out += texts(child)
        content = getattr(ctrl, "content", None)
        if content is not None:
            out += texts(content)
        return out

    shown = texts(control)
    assert "usb" in shown, "the source row names it"
    assert any("USB-C cable" in x for x in shown), "and says what it is"
    device.close()


def test_the_device_chip_names_the_pulse_sensor_rather_than_the_absent_amplifier():
    """The chip answers "where are these numbers coming from", and on the HRV screen the answer is never
    the amplifier — which is not connected, and said "no device" in grey beside a working pulse sensor.
    That is true of the amplifier, useless as an answer, and reads as a fault."""
    from nfb_console import hrv_view
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    choice = HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01")
    session = hrv_view.session_for(BY_CODE["HRV"], heart_choice=choice,
                                   heart_runner=HeartRunner(choice))
    label, note, _tone = hrv_view._instrument(session)
    assert label == "HeartMath · USB"
    assert note == "not started"

    session.heart.error = "no HeartMath cable — plug it in · trying again"
    label, note, _tone = hrv_view._instrument(session)
    assert label == "HeartMath · USB", "the name does not become the fault"
    assert "device screen" in note, "and the fault points at where it is fixed"
    assert len(note) < 60, "the chip has room for a few words, not for the runner's sentence"


def test_the_toolbar_shows_a_named_instrument_where_it_shows_the_amplifier():
    from nfb import Device, Settings
    from nfb_console import chrome

    device = Device(Settings())
    bar = chrome.Toolbar(device, lambda: None, lambda: None)
    assert bar.dev_name.value == "no device"

    bar.set_source("HeartMath · USB", note="62 bpm · not the amplifier")
    assert bar.dev_name.value == "HeartMath · USB"
    assert bar.channels_text.value == "62 bpm · not the amplifier", \
        "the second line is the instrument's state, not the amplifier's settings"

    bar.set_source("file: yesterday.jsonl.gz")
    assert bar.channels_text.value == "replay · not the amplifier", "replay keeps its own wording"
    device.close()


def test_a_screen_with_no_amplifier_channels_does_not_offer_to_reconfigure_it():
    """The count in the toolbar reconfigures the EEG amplifier, and on the HRV screen there is no
    amplifier in the path at all — so "1 ch" beside a pulse sensor is a control for a device this screen
    never touches. Hidden rather than greyed: disabled means "not now", and this is "not here"."""
    from nfb import Device, Settings
    from nfb_console import chrome

    device = Device(Settings())
    bar = chrome.Toolbar(device, lambda: None, lambda: None)
    assert bar.channel_menu.visible is not False, "the amplifier screens keep it"
    bar.set_channels_visible(False)
    assert bar.channel_menu.visible is False
    device.close()


def test_connecting_a_sensor_mid_session_replaces_the_simulator_the_screen_invented(monkeypatch):
    """**The HRV screen invents a simulator when nothing is chosen**, so that picking the protocol does
    not open an empty screen. Connecting a real sensor on the device screen and coming back then found
    the session still holding that simulator — and `run` hands its choice to the shared runner, so
    returning actively switched the connected sensor off again.

    An invented stand-in loses to anything the operator actually chose, and the memo that hands the
    runner back switched off goes with it, since it is no longer ours to hand back.
    """
    from nfb.hrv import SimulatedBeats
    from nfb_console import hrv_view
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    opened: list = []
    monkeypatch.setattr("nfb.hrv.beat_source",
                        lambda kind, **kw: (opened.append(kind),
                                            SimulatedBeats(realtime=False, seed=1))[1])

    app_choice = HeartChoice(kind="off")            # nothing chosen yet, as at launch
    runner = HeartRunner(app_choice)
    session = hrv_view.session_for(BY_CODE["HRV"], heart_choice=app_choice, heart_runner=runner)
    session.run()
    assert session.choice.kind == "simulator" and session._invented
    assert hrv_view._instrument(session)[0] == "simulated beats"

    # the device screen: a cable is chosen and connected, which is what set_heart_kind does
    app_choice.kind, app_choice.port = "usb", "/dev/cu.usbserial-HM01"
    runner.release()
    runner.attach(app_choice)

    session.rearm()                                  # coming back from the device screen
    session.run()
    try:
        assert session.choice is app_choice, "the session trains on what was actually chosen"
        assert not session._invented
        assert hrv_view._instrument(session)[0] == "HeartMath · USB", "and the chip says so"
        assert opened[-1] == "usb", "and the runner is reading the cable, not the simulator"
    finally:
        session.stop()
        assert app_choice.kind == "usb", "stopping must not switch the operator's own source off"


def test_a_paired_headband_replaces_the_heg_screens_invented_one(monkeypatch):
    """The same defect and the same fix on the HEG screen, whose simulator is invented for the same
    reason."""
    from nfb_console import heg_view
    from nfb_console.inputs import MendiChoice
    from nfb_console.protocols import BY_CODE

    app_choice = MendiChoice(kind="off")
    session = heg_view.session_for(BY_CODE["HEG"], mendi_choice=app_choice)
    assert session.choice.kind == "simulator" and session._invented

    app_choice.kind, app_choice.address = "headband", "AA:BB"
    session.adopt_offered()
    assert session.choice is app_choice and not session._invented


def test_a_cable_that_does_not_start_is_asked_again_and_said_to_be_silent(monkeypatch):
    """**Sometimes the device simply does not start.** It acknowledges the start command and then takes
    about 1.8 s to begin streaming, so anything disturbing it in that window leaves an open port, a live
    reader and no data — which read as an indefinite "connected, waiting for beats". Asking again costs
    five bytes. And the three states behind that one phrase are now told apart, because they want
    different things: nothing on the cable, a waveform with no pulse in it, and beats."""
    from nfb.hrv import sources
    from nfb.hrv.ppg import START_PPG

    written: list[bytes] = []

    class Silent:
        def __init__(self, *a, **kw):
            pass

        def reset_input_buffer(self):
            pass

        def write(self, data):
            written.append(bytes(data))

        def flush(self):
            pass

        def close(self):
            pass

        def read(self, _n):
            reads.append(1)
            if len(reads) == 6:                  # six empty reads is enough to have asked again
                said.append(source.progress())
                source.close()
            return b""

    reads: list[int] = []
    said: list[str] = []
    monkeypatch.setattr("serial.Serial", Silent)
    source = sources.UsbPpgBeats(port="/dev/null-ish")
    # Retry on every empty read, so the loop's own iterations drive this rather than a wall clock: the
    # thing under test is "it asks again", not "it asks again after two seconds".
    monkeypatch.setattr(type(source), "START_RETRY_S", 0.0)
    assert list(source.beats()) == [], "a silent cable yields nothing"
    assert written.count(START_PPG) > 2, "it kept asking the device to start"
    assert said and "no data on the cable" in said[0], "and said so rather than 'waiting for beats'"


def test_the_three_silences_are_told_apart():
    from nfb.hrv import sources

    source = sources.UsbPpgBeats(port="/dev/x")
    assert "opening" in source.progress()
    source.connected = True
    assert "no data on the cable" in source.progress()
    source.decoder._seen = 500
    assert "no pulse yet" in source.progress(), "streaming, but nothing is clipped on"
    source.beats_found = 3
    assert source.progress() == "reading the pulse waveform"
