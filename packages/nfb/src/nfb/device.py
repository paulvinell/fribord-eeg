"""The app-scope device: one owner of the port, many subscribers to its samples.

A serial port is exclusive, so exactly one object may hold it. Everything that wants samples
subscribes instead of connecting — protocols, traces, recorders, the feedback path. That means the
connection outlives whatever is running on top of it, and the status shown in a header is read from
state rather than by probing the port, which would fight the holder.

Three rules this encodes, all of them learned the hard way and documented in AGENTS.md:

- **Modes are exclusive.** Idle, measuring and impedance testing are separate device states (1, 5 and
  2), reached by different commands. Transitions are serialised here so the
  `SET_NOTIF` → `START_*` → `STOP_*` → `SET_NOTIF` sequence exists in one place.
- **Reconfiguring ends an epoch.** Changing rate or reference means writing a new config, which
  changes every channel's scale and decimation and restarts sample indices at zero. Subscribers are
  told the stream ended rather than having specs mutated underneath them mid-window, and
  reconfiguring is refused outright while streaming.
- **No auto-reconnect.** When the device stops responding it needs the USB cable replugged, not a
  retry loop — the FTDI module stays enumerated after the device powers off, so retrying just fails
  in a tight loop. The failure is recorded with its remedy and left for a human.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import optima

from .decode import FrameDecoder
from .stream import Block, ChannelSpec, Gap


class Mode(Enum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    MEASURING = "measuring"
    TESTING = "testing"          # electrode impedance / input continuity


@dataclass(frozen=True)
class Settings:
    """Everything a reconfigure needs. Immutable so an epoch's settings cannot drift."""

    channels: int = 4
    hz: float = 1000.0
    reference: str = "linked"
    mains_hz: int = 50
    """Which mains frequency the device notches out. The filter is in the device's hardware, so
    setting this wrongly means the hum is not filtered at all — it is not a cosmetic label."""
    lower_limit_hz: float = 0.5
    """The high-pass corner, in the device's hardware. 0.5 Hz is the vendor's own EEG choice; 0.01 Hz is
    the only other option a voltage channel offers short of DC, and it is what ultra-low-frequency work
    needs — a reward band below the corner measures the filter's roll-off rather than the brain. Like the
    mains setting, this is not a label: the filtering happens before we see a sample."""
    silent: bool = True
    """Whether the amplifier's own beeper is switched off. On by default because the host owns every sound a
    session makes; see `optima.build_eeg_config`, which also records that the bit's polarity is inferred from
    the vendor's field name and is the one thing about it nothing here has measured. A setting rather than a
    constant precisely because of that: if it turns out backwards, the device screen is where you find out
    and where you fix it, without editing the driver."""

    def build(self) -> bytes:
        return optima.build_eeg_config(self.channels, hz=self.hz, reference=self.reference,
                                       mains_hz=self.mains_hz,
                                       lower_limit_hz=self.lower_limit_hz, silent=self.silent)


@dataclass
class Status:
    """What a UI needs, without touching the port."""

    mode: Mode = Mode.DISCONNECTED
    port: str = ""
    model: str = ""
    serial: str = ""
    firmware: str = ""
    settings: Settings = field(default_factory=Settings)
    epoch: int = 0
    failure: str = ""
    remedy: str = ""

    @property
    def connected(self) -> bool:
        return self.mode is not Mode.DISCONNECTED


class EpochEnd:
    """Pushed to subscribers when the stream stops or is reconfigured, so they rebuild."""


class Subscription:
    """One consumer's view of the sample stream.

    Owns its own decoder, because decoding carries state — the phase tracker and every filter — and
    two consumers must not share it. Owns its own queue too, so a slow consumer drops its own frames
    instead of stalling acquisition for everyone.
    """

    def __init__(self, device: "Device", name: str, queue_size: int = 4096):
        self.device = device
        self.name = name
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self._decoder: FrameDecoder | None = None
        self._epoch = -1
        self._closed = False

    @property
    def config(self) -> bytes:
        """The config in force on the device, which is what this subscription decodes with.

        Same name as the file sources carry, so a consumer can read the corner or anything else that
        lives only in the blob without knowing whether it is holding a device or a recording.
        """
        return self.device.config

    def _offer(self, code: int, payload: bytes) -> None:
        """Queue a frame, evicting the *oldest* on overflow.

        The choice of end matters more than it looks. Dropping the newest — which this did — means a
        consumer that once fell behind is served an ever-older stream for the rest of the session: the
        queue holds up to 131 s of frames at 250 sps, all of it stale, with fresh frames dying at the
        door and nothing on screen distinguishing the result from a live session. Dropping the oldest
        keeps the consumer near the present, and the loss is *visible*: the evicted frames leave a hole
        in the phase counter, the decoder reports it as a Gap, and everything downstream already knows
        what a Gap means. An `EpochEnd` sentinel is never evicted — losing it would leave an iterator
        waiting on an epoch that already ended.
        """
        self._push((code, payload))

    def _push(self, item: object) -> None:
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    oldest = self.queue.get_nowait()
                except queue.Empty:
                    continue
                self.dropped += 1
                if isinstance(oldest, EpochEnd):
                    item = oldest          # the frame it displaces belonged to the dead epoch anyway

    def _end_epoch(self) -> None:
        """Mark the end of a stream, which terminates any iteration in progress.

        Through the same evicting push as frames: this sentinel used to be dropped outright when the
        queue was full — precisely the slow-consumer case — leaving the iterator waiting out its
        timeout on an epoch that had already ended.
        """
        self._push(EpochEnd())

    def _drain(self) -> None:
        """Discard anything queued from a previous epoch.

        Used when a stream *starts*, where pushing an EpochEnd would be wrong: a consumer that
        begins iterating after the start would meet the stale sentinel and stop immediately.
        """
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self._decoder = None

    def channels(self) -> dict[int, ChannelSpec]:
        self._ensure_decoder()
        return self._decoder.channels()

    def _ensure_decoder(self) -> None:
        if self._decoder is None or self._epoch != self.device.status.epoch:
            self._decoder = FrameDecoder(self.device.config,
                                         test_mode=self.device.status.mode is Mode.TESTING)
            self._epoch = self.device.status.epoch

    def frames(self, timeout: float = 5.0) -> Iterator[tuple[int, bytes]]:
        """Raw frames until the epoch ends or the subscription is closed — what a recorder consumes.

        The same loop as `blocks` without the decoder, because a recording's whole point is the frames
        exactly as they arrived (see `nfb.record`): decoding survives driver changes only if nothing here
        interpreted anything. A frame this subscription's queue evicted is simply absent from the file,
        which leaves a hole in the phase counter — so the loss is visible on replay as a `Gap`, the same
        way a live consumer sees it.
        """
        while not self._closed:
            try:
                item = self.queue.get(timeout=timeout)
            except queue.Empty:
                if self.device.status.mode in (Mode.IDLE, Mode.DISCONNECTED):
                    return
                continue
            if isinstance(item, EpochEnd):
                return
            yield item

    def blocks(self, timeout: float = 5.0) -> Iterator[Block | Gap]:
        """Decoded blocks until the epoch ends or the subscription is closed.

        Subscribe first, start the device, then iterate. Iterating while idle returns at once.
        """
        self._ensure_decoder()
        while not self._closed:
            try:
                item = self.queue.get(timeout=timeout)
            except queue.Empty:
                if self.device.status.mode in (Mode.IDLE, Mode.DISCONNECTED):
                    return
                continue
            if isinstance(item, EpochEnd):
                self._decoder = None
                return
            code, payload = item
            yield from self._decoder.feed(code, payload)

    def close(self) -> None:
        self._closed = True
        self.device._unsubscribe(self)


# An open input reads the top of the scale: the largest positive count times the kOhm-per-count factor, so
# just under the range rather than exactly on it. Compared with a margin, since it is the *saturation* that
# means "no electrode" and one count either way is not the point.
OPEN_KOHM = optima.TEST_RANGE_DEFAULT * 0.999


@dataclass
class Contact:
    """The **two** electrode impedances a voltage channel reports, in kOhm.

    A voltage channel measures a difference, so it has two electrodes and two contacts to check — and the
    device sends one value per channel per frame, alternating between them. Collapsing that into one number
    is what made the reading appear to flicker between two values twice a second, which reads as a fault in
    the instrument rather than as two different electrodes.

    Which is which is settled: **even phases carry "+", odd phases carry "−"**, verified on hardware with
    one electrode wired. The unwired channels' "+" read exactly full scale (open circuit) on even frames,
    while every channel's odd frame agreed to within 0.3 kOhm — because under `reference="linked"` the "−"
    inputs are joined inside the device, so all four are reading the *same* reference electrode. Across
    three fresh `START_TEST` runs the phase counter began at 0 every time, which is what makes the parity
    meaningful rather than relative.

    That shared "−" is also why the panel shows it once rather than four times: under linked or external
    referencing it is one electrode, and one bad reference contaminating all four channels is this wiring's
    named failure mode.
    """

    plus: float | None = None
    minus: float | None = None

    @property
    def worst(self) -> float | None:
        """The higher of the two, since contact is only as good as its worse electrode."""
        both = [v for v in (self.plus, self.minus) if v is not None]
        return max(both) if both else None

    @property
    def open_circuit(self) -> bool:
        """No electrode on the "+" input: it reads the top of the scale.

        The "+" input decides it because it is the only one that can. Under linked referencing the "−" is
        the shared reference electrode, so it reads a real value on a channel with nothing else attached.
        """
        return self.plus is not None and self.plus >= OPEN_KOHM


class Device:
    """Owns the connection for the lifetime of the application."""

    def __init__(self, settings: Settings | None = None):
        self.status = Status(settings=settings or Settings())
        self.config: bytes = b""
        self._dev: optima.Optima | None = None
        self._subs: list[Subscription] = []
        self._lock = threading.RLock()
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self.contact: dict[int, Contact] = {}
        self.excluded: list[int] = []
        """Channels left out of the running contact check for having no electrode. Public because a panel
        has to keep listing them: a row that vanished when the check narrowed would read as a channel that
        stopped existing rather than as one with nothing plugged into it."""
        self._restore_channels: int | None = None

    # --- connection ----------------------------------------------------------------------

    def connect(self, port: str | None = None) -> bool:
        """Open the port and read identity. Returns False and records why on failure."""
        with self._lock:
            if self.status.connected:
                return True
            try:
                dev = optima.Optima(port)
            except Exception as exc:
                self._fail(exc)
                return False
            try:
                info = dev.device_info()
                self.status.model = self._text(info, 0x23)
                self.status.serial = self._text(info, 0x44, 13)
                self.status.firmware = ".".join(str(b) for b in dev.firmware_info()[:3])
                self.status.port = dev.port
                self._dev = dev
                self.status.mode = Mode.IDLE
                self.status.failure = self.status.remedy = ""
                self._write_config()
                return True
            except Exception as exc:
                dev.close()
                self._fail(exc)
                return False

    def check_alive(self) -> bool:
        """Is the device still there? Call periodically while connected and **idle**.

        **Nothing noticed an unplug while idle**, because the only thing that ever touches the port is the
        read loop, and that runs only while streaming. So the device screen went on saying "Disconnect"
        next to a model and a serial number for a device that was no longer plugged in, and the first sign
        was a session that would not start.

        Two checks, because they catch different failures and neither catches both:

        - **the port node is gone** — unplugging the cable removes `/dev/cu.usbserial-*`, so this is free
          and unambiguous;
        - **a `GET_DEV_STATE` round trip** — the node *survives* the device powering itself off, since the
          FTDI module is USB-powered and stays enumerated. That is the replug case, and only asking the device something detects it.

        Refuses to run while streaming, where the read loop is already the detector and a command injected
        mid-stream would desync the framing.
        """
        with self._lock:
            if not self.status.connected or self._dev is None:
                return False
            if self.status.mode in (Mode.MEASURING, Mode.TESTING):
                return True
            port = self.status.port
            if port and not os.path.exists(port):
                self._fail(FileNotFoundError(f"{port} is gone"))
                return False
            try:
                self._dev.request(optima.CMD["GET_DEV_STATE"])
            except Exception as exc:
                self._fail(exc)
                return False
            return True

    def _fail(self, exc: BaseException) -> None:
        self._dev = None
        self.status.mode = Mode.DISCONNECTED
        text = str(exc)
        self.status.failure = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
        # EINVAL on open is the signature of a powered-off device whose FTDI module is still
        # enumerated; retrying cannot fix it, so say what does.
        if isinstance(exc, FileNotFoundError):
            self.status.remedy = "plug the USB cable back in"
        elif "Invalid argument" in text or "22" in text:
            self.status.remedy = "replug the USB cable, then switch the device on"
        elif "no FTDI" in text:
            self.status.remedy = "plug the device in and switch it on"
        else:
            self.status.remedy = "check the device is on; replug the cable if it persists"

    @staticmethod
    def _text(info: bytes, frame_offset: int, length: int = 24) -> str:
        chunk = info[frame_offset - 2:frame_offset - 2 + length]
        return chunk.split(b"\0")[0].decode("latin-1")

    def close(self) -> None:
        with self._lock:
            self.stop()
            if self._dev is not None:
                self._dev.close()
                self._dev = None
            self.status.mode = Mode.DISCONNECTED

    # --- configuration -------------------------------------------------------------------

    def reconfigure(self, **changes) -> None:
        """Change settings. Refused while streaming, because the data format would change mid-run."""
        with self._lock:
            if self.status.mode in (Mode.MEASURING, Mode.TESTING):
                raise RuntimeError(
                    f"cannot change settings while {self.status.mode.value}; stop the session first"
                )
            merged = Settings(**{**vars(self.status.settings), **changes})
            problems = optima.validate_config(merged.build())
            if problems:
                raise ValueError("; ".join(problems))
            self.status.settings = merged
            if self._dev is not None:
                self._write_config()

    def _write_config(self, for_contact_check: bool = False) -> None:
        self.config = self.status.settings.build()
        if for_contact_check:
            # Off the SUM entity for the duration of the test only, because that node couples the channels
            # and corrupts the reading — see `optima.contact_check_config`. `self.config` keeps the
            # measurement blob, since it is what the decoder is built from and the test's plan is derived
            # separately.
            self._dev.write_config(optima.contact_check_config(self.config))
            return
        self._dev.write_config(self.config)
        self.status.epoch += 1
        for sub in list(self._subs):
            sub._end_epoch()

    # --- modes ---------------------------------------------------------------------------

    def start_measurement(self) -> None:
        self._start(Mode.MEASURING)

    def start_impedance(self) -> None:
        self._start(Mode.TESTING)

    def unwired(self) -> list[int]:
        """Channels in use whose "+" input reads open circuit, so they have no electrode."""
        return sorted(c for c in range(self.status.settings.channels)
                      if c in self.contact and self.contact[c].open_circuit)

    def narrow_to_wired(self) -> list[int]:
        """Re-run the contact check on the wired channels alone, and return the ones dropped.

        **This is the fix for a reading that is simply wrong, not a tidying step.** The device drives the
        test on every *enabled* channel at once, and a channel with nothing attached disturbs the ones that
        do. Measured on hardware, one electrode at C4 against one reference, four channels enabled:

        | enabled | ch0 "+" | ch0 "−" |
        |---|---|---|
        | 1 channel, linked | 18.25 | 12.60 |
        | 1 channel, bipolar | 18.31 | 12.85 |
        | 4 channels, linked | 28.44 | 12.60 |
        | 4 channels, bipolar | **69.21** | **49.19** |

        So the true figures are ~18 and ~13 kOhm and the four-channel bipolar reading is out by **3.8×** —
        enough to condemn a good electrode or pass a bad one. With one channel enabled the two reference
        modes agree to 0.25 kOhm, which is what says the scale factor and the reference wiring are both
        right and the open channels are the whole story.

        Only a **prefix** of channels can be enabled, so a gap — ch0 and ch2 wired, ch1 not — cannot be
        excluded; `unwired()` still names it, and the panel says so, because a number that cannot be fixed
        should at least not be trusted silently.

        Restores the channel count when the test stops, since it is the operator's setting and a contact
        check must not quietly change what the next session runs.
        """
        with self._lock:
            if self.status.mode is not Mode.TESTING:
                raise RuntimeError("not testing")
            if self._restore_channels is not None:
                return []           # already narrowed; narrowing again would undo the restore
            open_ = self.unwired()
            wired = [c for c in range(self.status.settings.channels) if c not in open_]
            # Nothing attached anywhere is the bench case: there is no good reading to protect, and
            # narrowing to zero channels would leave nothing to test at all.
            if not open_ or not wired:
                return []
            keep = max(wired) + 1          # a prefix is all the config can express
            if keep >= self.status.settings.channels:
                return []
            was = self.status.settings.channels
            self.stop()
            self.reconfigure(channels=keep)
            self._restore_channels = was
            self.start_impedance()
            self.excluded = [c for c in range(was) if c >= keep]
            return list(self.excluded)

    def _start(self, mode: Mode) -> None:
        with self._lock:
            if self._dev is None:
                raise RuntimeError("not connected")
            if self.status.mode is mode:
                return
            if self.status.mode in (Mode.MEASURING, Mode.TESTING):
                raise RuntimeError(f"already {self.status.mode.value}; stop it first")
            self.contact.clear()
            if self._restore_channels is None:
                self.excluded = []      # a fresh check, not the second pass of a narrowing one
            self.status.mode = mode
            self.status.epoch += 1
            for sub in list(self._subs):
                sub._drain()
            if mode is Mode.TESTING:
                self._write_config(for_contact_check=True)
            self._dev.set_notifications(optima.NOTIF_ENABLE)
            if mode is Mode.TESTING:
                self._dev.request(optima.CMD["START_TEST"])
            else:
                self._dev.start_measurement()
            self._stop.clear()
            self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                            name="device-reader")
            self._reader.start()

    def stop(self) -> None:
        with self._lock:
            if self.status.mode not in (Mode.MEASURING, Mode.TESTING):
                return
            was = self.status.mode
            self._stop.set()
            if self._reader is not None:
                self._reader.join(timeout=2.0)
                self._reader = None
            try:
                if was is Mode.TESTING:
                    self._dev.request(optima.CMD["STOP_TEST"])
                else:
                    self._dev.stop_measurement()
                self._dev.set_notifications(optima.NOTIF_CLEAR)
            except Exception:
                pass                       # teardown must not mask why we stopped
            self.status.mode = Mode.IDLE
            for sub in list(self._subs):
                sub._end_epoch()
            # Hand the operator's channel count back. `narrow_to_wired` borrowed it for the duration of a
            # contact check, and a check that quietly left the device running one channel would change what
            # the next session trains.
            self.excluded = []
            restore, self._restore_channels = self._restore_channels, None
            if restore is not None and restore != self.status.settings.channels:
                self.reconfigure(channels=restore)
            elif was is Mode.TESTING and self._dev is not None:
                # Put the channels back on the SUM entity. The reconfigure above already rewrites the
                # measurement config, so this is only for the path that did not need it.
                try:
                    self._write_config()
                except Exception:
                    pass                   # teardown must not mask why we stopped

    # --- reading -------------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Reads and fans out. Deliberately does nothing else — every line here can delay a read."""
        testing = self.status.mode is Mode.TESTING
        plan = optima.plan_channels(self.config, test_mode=testing) if testing else None
        while not self._stop.is_set():
            try:
                msg = self._dev.read_message()
            except TimeoutError:
                continue
            except Exception as exc:
                self._fail(exc)
                for sub in list(self._subs):
                    sub._end_epoch()
                return
            for sub in list(self._subs):
                sub._offer(msg.code, msg.payload)
            if testing and msg.code == optima.NOT_DATA:
                self._note_impedance(msg.payload, plan)

    def _note_impedance(self, payload: bytes, plan) -> None:
        """Keep the newest reading per electrode so a panel can show contact without subscribing.

        Two electrodes per channel, one per frame, decided by the parity of the phase the sample sits at —
        see `Contact`. The parity is taken from `next_phase - 1` rather than from the frame's own header
        because that is the phase the *last* sample belongs to, which stays right if a frame ever carries
        more than one value per channel; in test mode it carries exactly one.
        """
        try:
            frame = optima.decode_data(payload, plan)
        except Exception:
            return
        scale = optima.test_scale()
        plus = (frame.next_phase - 1) % 2 == 0
        for ch, samples in frame.samples.items():
            if not samples:
                continue
            reading = samples[-1] * scale
            spot = self.contact.setdefault(ch, Contact())
            if plus:
                spot.plus = reading
            else:
                spot.minus = reading

    # --- subscription --------------------------------------------------------------------

    def subscribe(self, name: str, queue_size: int = 4096) -> Subscription:
        """Subscribing while idle is fine; subscribing to a device that is not there is not.

        A subscription's decoder is built from `self.config`, which does not exist until a successful
        `connect()` writes one — so a caller that skipped the connection check got `config blob too
        short: 0 bytes` out of the framing layer, several frames below anything that knows what a
        device is. Refuse here, with the failure and its remedy, because those are the two things the
        caller has to put on screen.
        """
        with self._lock:
            if not self.status.connected:
                why = self.status.failure or "no device"
                raise RuntimeError(f"not connected: {why}"
                                   + (f" — {self.status.remedy}" if self.status.remedy else ""))
            sub = Subscription(self, name, queue_size)
            self._subs.append(sub)
            return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    @property
    def subscribers(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._subs)

    def __enter__(self) -> "Device":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
