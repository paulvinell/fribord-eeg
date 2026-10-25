"""Where fNIRS frames come from: a Mendi headband, or a simulator.

**Synchronous iterators, deliberately** — the same shape as `nfb.hrv.sources` and for the same reason:
this app's feeds are threads running `for x in source`, and a second concurrency model inside one
process is a cost with no buyer. `LiveMendi` keeps its asyncio loop private, in its own thread, and
hands frames over through a queue.

The device protocol is the `DeviceV4` protobuf — see `proto.py`. It was
checked against a real headband.

Stream lifecycle (matches the Android app):
    1. Connect via bleak; read firmware (0x2A26) and hardware (0x2A27) revisions, best-effort.
    2. Subscribe ABB1 (frames), ABB4 (battery), ABB6 (calibration).
    3. Write Sensor(addr=30, data=0x0100) to ABB2 → the device starts streaming. The write must be
       **with response**: the firmware ignores Write Command (0x52) on this characteristic even though
       the descriptor advertises WRITE_NO_RESPONSE — established empirically against firmware 1.0.2.
       And it must come *after* the frame subscription, or the first frames are lost.
    4. On exit: write Sensor(addr=30, data=0x0000), unsubscribe, disconnect — all best-effort.

`bleak` is the same optional extra the HRV strap uses; `problem()` says so when it is missing.
"""

from __future__ import annotations

import math
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Protocol

from . import proto

HEG_RATE_HZ = 30.0
"""What the headband actually streams at, nominal. Frames carry their own timestamps and every window
downstream is time-based, so this is documentation and a simulator default rather than a contract."""


@dataclass(frozen=True)
class Channel:
    """One optical channel: IR and red are signal, ambient is the LEDs-off baseline in the same packet."""

    ir: float
    red: float
    amb: float


@dataclass(frozen=True)
class Frame:
    """One decoded sensor frame.

    Channels L and R sit over Fp1/Fp2; P sits between them and moves much less — the working reading
    is a short-separation/reference pickup, which is what makes it the common-mode term in
    `processor.project`. Accelerometer and gyroscope are raw counts with no known scaling. They are relative-motion signals and the motion gate treats them
    as nothing more.
    """

    t: float                            # epoch seconds, arrival
    l: Channel
    r: Channel
    p: Channel
    acc: tuple[int, int, int]
    gyro: tuple[int, int, int]
    temp_c: float


class FrameSource(Protocol):
    """What a headband and a simulator both look like from above."""

    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...
    def describe(self) -> str: ...


def problem() -> str | None:
    """Why the headband would not work, if it would not."""
    try:
        import bleak                # noqa: F401
    except Exception as exc:
        return f"Bluetooth is unavailable ({exc}) — check `uv sync`, or use the simulator"
    return None


def _mendi_uuid(short: str) -> str:
    return f"fc3e{short}-c6c4-49e6-922a-6e551c455af5"


SERVICE_UUID = _mendi_uuid("abb0")
CHAR_FRAME = _mendi_uuid("abb1")        # notify: DeviceV4.Frame
CHAR_SENSOR = _mendi_uuid("abb2")       # write : DeviceV4.Sensor (register R/W)
CHAR_BATTERY = _mendi_uuid("abb4")      # notify: DeviceV4.Adc
CHAR_CALIBRATION = _mendi_uuid("abb6")  # notify: DeviceV4.Calibration
FIRMWARE_REVISION_UUID = "00002a26-0000-1000-8000-00805f9b34fb"
HARDWARE_REVISION_UUID = "00002a27-0000-1000-8000-00805f9b34fb"

STREAM_REG_ADDR = 30
STREAM_ON_DATA = 0x0100
STREAM_OFF_DATA = 0x0000
"""The sensor-IC register that gates streaming, bit 8 high = on."""


class SimulatedMendi:
    """Synthetic frames: slow anticorrelated IR/red drift, ~1 Hz cardiac ripple, quiet IMU.

    Good enough to exercise the screen and the per-channel arithmetic; not a model of a brain. The
    per-channel phase offsets keep L, R and P from being identical, and P drifts least — matching the
    empirical observation the projection rests on.
    """

    def __init__(self, sample_rate_hz: float = HEG_RATE_HZ, seed: int | None = None,
                 realtime: bool = True):
        self._dt = 1.0 / max(0.1, sample_rate_hz)
        self._rng = random.Random(seed)
        self._drift_ir = [0.0, 0.0, 0.0]
        self._drift_red = [0.0, 0.0, 0.0]
        self.realtime = realtime
        self._stop = False

    def _channel(self, idx: int, pulse_phase: float) -> Channel:
        # P (idx 2) drifts a fifth as much: it is the reference pickup the projection subtracts.
        gain = 0.2 if idx == 2 else 1.0
        self._drift_ir[idx] = 0.995 * self._drift_ir[idx] + gain * self._rng.gauss(0, 30.0)
        self._drift_red[idx] = 0.995 * self._drift_red[idx] - 0.4 * gain * self._rng.gauss(0, 30.0)
        cardiac = 800.0 * math.sin(pulse_phase + idx * 0.4)
        ir = 20000 + self._drift_ir[idx] + cardiac + self._rng.gauss(0, 40)
        red = 15000 + self._drift_red[idx] + 0.7 * cardiac + self._rng.gauss(0, 40)
        amb = 200 + self._rng.gauss(0, 20)
        return Channel(ir=max(0.0, ir), red=max(0.0, red), amb=max(0.0, amb))

    def frames(self) -> Iterator[Frame]:
        t0, elapsed, pulse_hz = time.time(), 0.0, 1.05
        while not self._stop:
            phase = 2.0 * math.pi * pulse_hz * elapsed
            yield Frame(
                t=t0 + elapsed,
                l=self._channel(0, phase),
                r=self._channel(1, phase + 0.1),
                p=self._channel(2, phase + 0.2),
                acc=(self._rng.randint(-100, 100), self._rng.randint(-100, 100),
                     self._rng.randint(900, 1100)),
                gyro=(self._rng.randint(-20, 20), self._rng.randint(-20, 20),
                      self._rng.randint(-20, 20)),
                temp_c=36.5 + self._rng.gauss(0, 0.05),
            )
            elapsed += self._dt
            if self.realtime:
                time.sleep(self._dt)

    def close(self) -> None:
        self._stop = True

    def describe(self) -> str:
        return f"simulated headband — {1.0 / self._dt:g} frames/s"


class LiveMendi:
    """A Mendi headband over BLE, frames handed across a thread boundary.

    Discovery prefers advertisers of the Mendi service UUID and falls back to a name match; the
    failure message lists what *was* visible, so picking an address by hand is a copy-paste.
    """

    def __init__(self, address: str | None = None, name_hint: str = "Mendi",
                 timeout: float = 10.0):
        self.address = address
        self.name_hint = name_hint
        self.timeout = timeout
        self.error: str | None = None
        self.connected = False
        self.battery_pct: int | None = None
        self.battery_charging: bool | None = None
        self.calibration: dict | None = None
        self.firmware_version: str | None = None
        self.hardware_version: str | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def frames(self) -> Iterator[Frame]:
        trouble = problem()
        if trouble:
            self.error = trouble
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="mendi-ble")
        self._thread.start()
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._thread is not None and not self._thread.is_alive():
                    return
                continue
            if frame is None:
                return
            yield frame

    def _push(self, item) -> None:
        """Evicting push, oldest first, sentinel never lost — the policy every queue in this repo
        converged on; see `nfb.Device._push` for the argument."""
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
                    item = oldest

    def _run(self) -> None:
        import asyncio

        async def go() -> None:
            from bleak import BleakClient, BleakScanner

            address = self.address
            if not address:
                found = await BleakScanner.discover(timeout=self.timeout, return_adv=True)
                match = None
                for d, adv in found.values():
                    uuids = [str(u).lower() for u in (adv.service_uuids or [])]
                    if SERVICE_UUID in uuids:
                        match = d
                        break
                if match is None:
                    hint = self.name_hint.lower()
                    for d, adv in found.values():
                        name = (d.name or adv.local_name or "").lower()
                        if hint and hint in name:
                            match = d
                            break
                if match is None:
                    nearby = ", ".join(sorted({(d.name or adv.local_name or "?")
                                               for d, adv in found.values()})) or "nothing"
                    self.error = (f"no Mendi found — switch it on and make sure it is not paired "
                                  f"to a phone · nearby: {nearby}")
                    return
                address = match.address
            async with BleakClient(address) as client:
                self.connected = True
                for uuid, attr in ((FIRMWARE_REVISION_UUID, "firmware_version"),
                                   (HARDWARE_REVISION_UUID, "hardware_version")):
                    try:
                        raw = await client.read_gatt_char(uuid)
                        setattr(self, attr, raw.decode("ascii", errors="replace").strip())
                    except Exception:
                        pass            # DIS access can be revoked in low-power mode

                def on_frame(_sender, data: bytearray) -> None:
                    try:
                        f = proto.decode_frame(bytes(data))
                    except Exception:
                        return
                    self._push(Frame(
                        t=time.time(),
                        l=Channel(f["ir_l"], f["red_l"], f["amb_l"]),
                        r=Channel(f["ir_r"], f["red_r"], f["amb_r"]),
                        p=Channel(f["ir_p"], f["red_p"], f["amb_p"]),
                        acc=(int(f["acc_x"]), int(f["acc_y"]), int(f["acc_z"])),
                        gyro=(int(f["ang_x"]), int(f["ang_y"]), int(f["ang_z"])),
                        temp_c=float(f["temp"]),
                    ))

                def on_battery(_sender, data: bytearray) -> None:
                    try:
                        adc = proto.decode_adc(bytes(data))
                    except Exception:
                        return
                    self.battery_pct = proto.adc_voltage_to_percent(int(adc["voltage_mv"]))
                    self.battery_charging = bool(adc["charging"])

                def on_calibration(_sender, data: bytearray) -> None:
                    try:
                        self.calibration = proto.decode_calibration(bytes(data))
                    except Exception:
                        pass

                await client.start_notify(CHAR_BATTERY, on_battery)
                await client.start_notify(CHAR_CALIBRATION, on_calibration)
                await client.start_notify(CHAR_FRAME, on_frame)
                # After the subscription, with response — see the module docstring; both halves of
                # that sentence were established the hard way.
                await client.write_gatt_char(
                    CHAR_SENSOR, proto.encode_sensor_write(STREAM_REG_ADDR, STREAM_ON_DATA),
                    response=True)
                try:
                    while not self._stop.is_set():
                        await asyncio.sleep(0.2)
                finally:
                    try:
                        await client.write_gatt_char(
                            CHAR_SENSOR,
                            proto.encode_sensor_write(STREAM_REG_ADDR, STREAM_OFF_DATA),
                            response=True)
                    except Exception:
                        pass
                    for char in (CHAR_FRAME, CHAR_BATTERY, CHAR_CALIBRATION):
                        try:
                            await client.stop_notify(char)
                        except Exception:
                            pass

        try:
            asyncio.run(go())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.connected = False
            self._push(None)

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
            extras = []
            if self.firmware_version:
                extras.append(f"fw {self.firmware_version}")
            if self.battery_pct is not None:
                extras.append(f"battery {self.battery_pct}%")
            joined = f" · {' · '.join(extras)}" if extras else ""
            return f"connected to {self.address or 'a Mendi headband'}{joined}"
        return "looking for a Mendi headband…"


def frame_source(kind: str, **kwargs) -> FrameSource:
    if kind == "simulator":
        return SimulatedMendi(**kwargs)
    if kind == "headband":
        return LiveMendi(**kwargs)
    raise ValueError(f"unknown frame source {kind!r}")
