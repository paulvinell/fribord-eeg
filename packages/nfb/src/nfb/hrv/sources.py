"""Where beats come from: a heart-rate strap over Bluetooth, the HeartMath device over its USB cable,
or a simulator.

**Synchronous iterators, deliberately.** This is a Flet app whose feeds already run on threads, and the EEG side's contract is
`for block in source.blocks()`. Matching that means an HRV session is the same shape as every other
session — a thread running `session.run()` — rather than a second concurrency model inside one process.
`BleBeats` keeps its asyncio loop private, in its own thread, and hands beats over through a queue.

**bleak is imported lazily** and `problem()` reports a failure to load it as a message rather than an
exception, because the EEG path must keep working on a machine whose Bluetooth stack does not. It is an
ordinary dependency — installed always, imported only here. Nothing in `optima` or the EEG side imports
this.
"""

from __future__ import annotations

import math
import queue
import random
import threading
import time
from typing import Iterator, Protocol

from .beats import Beat, RrScaleResolver, parse_hr_measurement, parse_hr_raw
from .ppg import QUERY_BATTERY, START_PPG, STOP_PPG, USB_BAUD, PpgDecoder

HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"

FTDI_VID = 0x0403
"""Both this device and the Optima are FTDI USB-serial bridges, which is the whole difficulty in
`find_usb_port` below."""

NOT_THE_HEART = ("neurobit", "optima")
"""Words in a port's description that mean it is the amplifier, not a pulse sensor."""


class BeatSource(Protocol):
    """What a strap and a simulator both look like from above. Not a `SampleSource`: beats are not
    samples, and pretending otherwise is what forces a fake sample rate onto irregular events."""

    def beats(self) -> Iterator[Beat]: ...
    def close(self) -> None: ...
    def describe(self) -> str: ...


class SimulatedBeats:
    """A clean resting heart with respiratory modulation, in real time.

    Exists for the same reason the file sources do: developing a screen should not require strapping
    something on. The modulation frequency is a parameter so the spectrum's peak lands somewhere known,
    which makes it a fixture as well as a stand-in.

    `split` generates the split-spectrum failure — very-low-frequency and high-frequency modulation
    at once with nothing at 0.1 Hz — because a detector for a state nobody can produce on demand is a
    detector nobody has tested.
    """

    def __init__(self, hr_bpm: float = 65.0, rsa_ms: float = 80.0, rsa_hz: float = 0.1,
                 noise_ms: float = 8.0, seed: int | None = None, realtime: bool = True,
                 split: bool = False):
        self.hr_bpm = hr_bpm
        self.rsa_s = rsa_ms / 1000.0
        self.rsa_hz = rsa_hz
        self.noise_s = noise_ms / 1000.0
        self.realtime = realtime
        self.split = split
        self._rng = random.Random(seed)
        self._stop = False

    def beats(self) -> Iterator[Beat]:
        elapsed = 0.0
        started = time.time()
        while not self._stop:
            base = 60.0 / self.hr_bpm
            if self.split:
                swing = (self.rsa_s * 0.8 * math.sin(2 * math.pi * 0.02 * elapsed)
                         + self.rsa_s * 0.6 * math.sin(2 * math.pi * 0.28 * elapsed))
            else:
                swing = self.rsa_s * math.sin(2 * math.pi * self.rsa_hz * elapsed)
            rr = max(0.3, base + swing + self._rng.gauss(0.0, self.noise_s))
            elapsed += rr
            yield Beat(t=started + elapsed, hr_bpm=60.0 / rr, rr_intervals=[rr], contact=True)
            if self.realtime:
                time.sleep(rr)

    def close(self) -> None:
        self._stop = True

    def describe(self) -> str:
        if self.split:
            return "simulated — deliberately split, for testing the warning"
        return f"simulated — {self.hr_bpm:g} bpm with {self.rsa_hz:g} Hz modulation"


def problem() -> str | None:
    """Why BLE would not work, if it would not."""
    try:
        import bleak                # noqa: F401
    except Exception as exc:
        return f"Bluetooth is unavailable ({exc}) — check `uv sync`, or use the simulator"
    return None


class BleBeats:
    """Any device advertising the standard Heart Rate Service, which the HeartMath Coherence Plus does.

    Generic rather than HeartMath-specific: the same code reads a Polar strap.
    """

    def __init__(self, address: str | None = None, name_hint: str = "", timeout: float = 20.0,
                 rr_units: str = "auto"):
        self.address = address
        self.name_hint = name_hint
        self.timeout = timeout
        # Which unit this device's RR integers are in. `auto` decides from the first few packets — see
        # `RrScaleResolver`, which exists because the HeartMath device sends milliseconds where the spec
        # says 1/1024 s, and we divided by 1024 regardless for a silent 2.3% error in every interval.
        self.rr_scale = RrScaleResolver(rr_units)
        self.error: str | None = None
        self.connected = False
        self._queue: queue.Queue = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def beats(self) -> Iterator[Beat]:
        trouble = problem()
        if trouble:
            self.error = trouble
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="hrv-ble")
        self._thread.start()
        while not self._stop.is_set():
            try:
                beat = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._thread is not None and not self._thread.is_alive():
                    return
                continue
            if beat is None:
                return
            yield beat

    def _run(self) -> None:
        import asyncio

        async def go() -> None:
            from bleak import BleakClient, BleakScanner

            address = self.address
            if not address:
                found = await BleakScanner.discover(timeout=self.timeout,
                                                    service_uuids=[HR_SERVICE])
                match = next((d for d in found
                              if not self.name_hint or self.name_hint.lower()
                              in (d.name or "").lower()), None)
                if match is None:
                    self.error = ("no heart-rate device found · switch it on and make sure it is not "
                                  "paired to a phone")
                    return
                address = match.address
            async with BleakClient(address) as client:
                self.connected = True
                # Packets held back until the unit resolver commits. Without this the first
                # `min_packets` intervals from a milliseconds device enter the record scaled by the
                # spec's 1024 — 2.3% short, never revisited: exactly the silent error the resolver was
                # written to eliminate, surviving in its own warm-up. A few seconds of held beats at
                # session start is what `describe()`'s "resolving RR units" was already saying.
                held: list[tuple[float, bytes]] = []

                def emit(at: float, data: bytes) -> None:
                    hr, rr, contact = parse_hr_measurement(data, scale=self.rr_scale.scale)
                    self._push(Beat(at, hr, rr, contact))

                def on_notify(_sender, data: bytearray) -> None:
                    packet = bytes(data)
                    hr, raw, _contact = parse_hr_raw(packet)
                    self.rr_scale.observe(hr, raw)
                    if self.rr_scale.decided is None:
                        held.append((time.time(), packet))
                        return
                    while held:
                        emit(*held.pop(0))
                    emit(time.time(), packet)

                await client.start_notify(HR_MEASUREMENT, on_notify)
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)
                await client.stop_notify(HR_MEASUREMENT)

        try:
            asyncio.run(go())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.connected = False
            self._push(None)

    def _push(self, item) -> None:
        """Queue a beat, evicting the *oldest* on overflow — the policy `nfb.Device`'s subscription
        queue settled on, for the same reason: dropping the newest serves an ever-staler stream to a
        consumer that once fell behind, silently. The end-of-stream `None` sentinel is never the thing
        evicted, and never lost to a full queue either, which it used to be."""
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    oldest = self._queue.get_nowait()
                except queue.Empty:
                    continue
                if oldest is None:
                    item = oldest       # the sentinel outranks whatever displaced it

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def describe(self) -> str:
        if self.error:
            return self.error
        if self.connected:
            return f"connected to {self.address or 'a heart-rate monitor'}"
        return "looking for a heart-rate monitor…"


def describe_port(port) -> str:
    """What a `serial.tools.list_ports` entry looks like to a person, for the "I saw these" messages."""
    return " ".join(str(x) for x in (port.description, port.manufacturer, port.product,
                                     port.serial_number) if x and str(x) != "n/a")


def is_amplifier(port) -> bool:
    """Whether an FTDI port is the Optima rather than a pulse sensor.

    **This exists because the two are indistinguishable by vendor id**, and the amplifier is the more
    dangerous of the two to open by mistake: it is a serial port, so opening it is exclusive, and writing
    a pulse-sensor start command at it would be writing nonsense into the device the whole console
    depends on. It advertises itself as a Neurobit, which is the one thing that separates them.
    """
    return any(word in describe_port(port).lower() for word in NOT_THE_HEART)


def usb_candidates() -> list:
    """FTDI serial ports that are not the amplifier."""
    from serial.tools import list_ports

    return [p for p in list_ports.comports() if p.vid == FTDI_VID and not is_amplifier(p)]


def find_usb_port() -> tuple[str | None, str | None]:
    """The pulse sensor's serial port, or a sentence saying what was there instead.

    The sentence names only what is on USB. A Mac lists a handful of serial ports that are nothing of
    the kind — a debug console, an incoming-Bluetooth port, every audio device ever paired — and
    reciting those says nothing about a cable that is or is not plugged in.
    """
    from serial.tools import list_ports

    candidates = usb_candidates()
    if len(candidates) == 1:
        return candidates[0].device, None
    if len(candidates) > 1:
        listed = ", ".join(f"{p.device} ({describe_port(p) or 'unnamed'})" for p in candidates)
        return None, f"more than one cable it could be, so name the port: {listed}"

    on_usb = [p for p in list_ports.comports() if p.vid is not None]
    if not on_usb:
        return None, "no HeartMath cable — nothing at all is plugged into USB"
    seen = ", ".join(f"{p.device} ({describe_port(p) or 'unnamed'})" for p in on_usb)
    excluded = (" The amplifier is skipped on purpose; it is the same kind of chip."
                if any(is_amplifier(p) for p in on_usb) else "")
    return None, f"no HeartMath cable — plug it in, or name the port. On USB: {seen}.{excluded}"


class UsbPpgBeats:
    """The HeartMath device over its USB-C cable, reading the raw pulse waveform.

    A different **stream**, not just a different transport, and that is the reason to prefer it. The
    Bluetooth path takes the device's own beat detections through the standard Heart Rate Service, one
    interval at a time. This one takes the sensor's 500 Hz waveform and finds the beats here, in
    `PpgDecoder`, where the detector is ours and its failures are visible.

    Three practical things a cable buys, all established on hardware:

    - **No keepalive.** Over Bluetooth the firmware powers the sensor's LED down after about a minute
      unless something stays subscribed to the Heart Rate characteristic. There is no GATT here and a
      100 s capture streamed throughout.
    - **No scanning, no pairing, and nothing else can claim it.** A phone cannot take the cable.
    - **The device's own beat detections arrive too**, in frames only the serial stream carries. We do
      not read them yet; they are a free cross-check when someone wants one.

    The sample rate is *not* one of the differences — both carry the full 500 Hz. Battery is not one
    either, since `QUERY_BATTERY` works here. The cost is the cable.

    No reader thread, unlike `BleBeats`: the read is blocking with a short timeout, so the generator is
    the loop. `close()` is called from another thread and only sets the flag; the port is opened and
    closed on the thread that iterates.
    """

    START_RETRY_S = 2.0
    """Re-send the start command if nothing has arrived for this long.

    **The device sometimes does not start.** It acknowledges the command and then takes about 1.8 s to
    begin streaming, so anything that disturbs it in that window — a previous reader's stop command, a
    session ended mid-handshake, the cable seated while the firmware was busy — leaves a port that is
    open, a reader that is alive, and no data at all, which on screen was an indefinite "waiting for
    beats". Asking again costs five bytes and is harmless while it is already streaming."""

    def __init__(self, port: str | None = None, baud: int = USB_BAUD, read_size: int = 4096,
                 read_timeout: float = 0.1, battery_poll_s: float = 60.0):
        self.port = port
        self.baud = baud
        self.read_size = read_size
        self.read_timeout = read_timeout
        self.battery_poll_s = battery_poll_s
        self.error: str | None = None
        self.connected = False
        self.decoder = PpgDecoder()
        self.beats_found = 0
        self._stop = threading.Event()

    def beats(self) -> Iterator[Beat]:
        import serial

        port = self.port
        if not port:
            port, trouble = find_usb_port()
            if port is None:
                self.error = trouble
                return
        try:
            link = serial.Serial(port, self.baud, timeout=self.read_timeout)
        except Exception as exc:
            self.error = f"{port} would not open ({type(exc).__name__}: {exc})"
            return
        self.port = port
        try:
            # Drop whatever the device buffered before we arrived, so the splitter starts on a frame
            # boundary rather than spending its first frames re-syncing.
            link.reset_input_buffer()
            link.write(START_PPG)
            link.flush()
            self.connected = True
            asked_battery = 0.0
            asked_start = time.time()
            while not self._stop.is_set():
                chunk = link.read(self.read_size)
                now = time.time()
                if chunk:
                    asked_start = now
                    for beat in self.decoder.feed(bytes(chunk), now):
                        self.beats_found += 1
                        yield beat
                elif now - asked_start >= self.START_RETRY_S:
                    # Nothing at all on the cable. Ask again — see START_RETRY_S.
                    asked_start = now
                    link.write(START_PPG)
                    link.flush()
                if self.battery_poll_s > 0 and now - asked_battery >= self.battery_poll_s:
                    asked_battery = now
                    link.write(QUERY_BATTERY)
                    link.flush()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.connected = False
            # Tell it to stop before dropping the port, or it keeps the sensor lit and streams into a
            # closed buffer.
            try:
                link.write(STOP_PPG)
                link.flush()
            except Exception:
                pass
            try:
                link.close()
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()

    def progress(self) -> str:
        """Which of the three states this is in, in a few words.

        **"Waiting for beats" covered three different situations** and the screen could not tell them
        apart: nothing arriving on the cable at all, a waveform with no pulse in it — the sensor not on an
        ear — and a waveform we are failing to find beats in. They need different actions from the
        operator, so they get different words.
        """
        if not self.connected:
            return "opening the cable"
        if self.decoder._seen == 0:
            return "no data on the cable · asking it to start"
        if self.beats_found == 0:
            return "waveform arriving, no pulse yet · is the sensor on?"
        return "reading the pulse waveform"

    def describe(self) -> str:
        if self.error:
            return self.error
        battery = (f" · battery {self.decoder.battery_pct}%"
                   if self.decoder.battery_pct is not None else "")
        if self.connected:
            return f"{self.progress()} on {self.port}{battery}"
        return "opening the USB cable…"


def beat_source(kind: str, **kwargs) -> BeatSource:
    if kind == "simulator":
        return SimulatedBeats(**kwargs)
    if kind == "ble":
        return BleBeats(**kwargs)
    if kind == "usb":
        return UsbPpgBeats(**kwargs)
    raise ValueError(f"unknown beat source {kind!r}")
