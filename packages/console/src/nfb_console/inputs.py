"""Where samples come from: the amplifier, or a file played on loop.

This is a device-screen concern rather than a protocol one, for the same reason the sample rate is:
what you are recording from outlives any one session, and every protocol wants the same answer.

Playing from a file is not a demo feature. Working on a protocol screen against a live head means
wiring up electrodes for every code change, and the whole architecture is arranged so that a source
is interchangeable — so the honest thing is to make the substitution reachable from the UI rather
than only from a command-line flag.

Two file formats, dispatched on extension, and they are not equivalent:

- **`.jsonl` / `.jsonl.gz`** — a session recorded by this program. Raw device frames, replayed through
  the identical decoder, so the result is provably what a live run would have produced.
- **anything else** — a plain text export: one row per sample, one column per channel. Already-decoded
  microvolts from some other program, with no decoder in the path and no device to be faithful to.
  Fine for exercising a protocol screen; not evidence about the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nfb import Device, ReplaySource, SampleFileSource

FRAME_SUFFIXES = (".jsonl", ".gz")


def is_frame_recording(path: str | Path) -> bool:
    return Path(path).suffix in FRAME_SUFFIXES


@dataclass
class InputChoice:
    """Which source the next session will open."""

    kind: str = "amplifier"        # amplifier | file
    path: str = ""
    loop: bool = True

    @property
    def from_file(self) -> bool:
        return self.kind == "file" and bool(self.path)

    def problem(self) -> str | None:
        """Why this choice would not work, for the screen to say before a session is started."""
        if not self.from_file:
            return None
        p = Path(self.path).expanduser()
        if not p.exists():
            return f"no such file: {p}"
        if p.is_dir():
            return f"{p} is a directory"
        return None

    def describe(self) -> str:
        if not self.from_file:
            return "live amplifier"
        name = Path(self.path).name
        kind = "recorded frames" if is_frame_recording(self.path) else "text samples"
        return f"{name} · {kind}{' · looping' if self.loop else ''}"

    def open(self, device: Device | None = None, realtime: bool = True):
        """Build the source. `device` is used only for the amplifier choice, and is subscribed to
        rather than opened, since the app owns the port."""
        if self.from_file:
            path = str(Path(self.path).expanduser())
            if is_frame_recording(path):
                return ReplaySource(path, realtime=realtime, loop=self.loop)
            return SampleFileSource(path, loop=self.loop, realtime=realtime)
        if device is None:
            raise RuntimeError("no device to read from")
        return None                # the caller subscribes; see AmplitudeSession


@dataclass
class HeartChoice:
    """Which heart-rate source, if any. The device-screen twin of `InputChoice`.

    Lives here for the same reason: which strap is on your chest outlives any one session, and it is not
    a property of the protocol you happen to be running. It has to be *one* object owned by the app,
    because two screens want it at once — the HRV screen trains on it, and the CFC screen shows the same
    instruments beside the EEG.
    """

    kind: str = "off"              # off | strap | usb | simulator
    address: str = ""              # a specific device, once one has been found
    name_hint: str = ""            # or a name filter, e.g. "HeartMath"
    port: str = ""                 # a serial port, when the HeartMath is on its cable

    @property
    def on(self) -> bool:
        return self.kind in ("strap", "usb", "simulator")

    def problem(self) -> str | None:
        """What is wrong before anything is tried. Bluetooth can be missing from the machine itself,
        which is knowable in advance; a cable cannot, and looking for one here would say the same thing
        the runner says the moment it tries, in two places at once. One message, from whichever of the
        two actually found it out."""
        if self.kind == "strap":
            from nfb.hrv import sources
            return sources.problem()
        return None

    def describe(self) -> str:
        if self.kind == "off":
            return "no heart-rate source"
        if self.kind == "simulator":
            return "simulated beats — 65 bpm with 0.1 Hz modulation"
        if self.kind == "usb":
            return f"HeartMath on {self.port}" if self.port else "HeartMath on its USB cable"
        if self.address:
            return f"strap at {self.address}"
        return f"any strap advertising the Heart Rate Service{f' named ~{self.name_hint}' if self.name_hint else ''}"

    def open(self):
        """Build the beat source, or None when this is off."""
        from nfb.hrv import beat_source
        if self.kind == "simulator":
            return beat_source("simulator")
        if self.kind == "strap":
            return beat_source("ble", address=self.address or None, name_hint=self.name_hint)
        if self.kind == "usb":
            return beat_source("usb", port=self.port or None)
        return None


@dataclass
class MendiChoice:
    """Which fNIRS headband, if any. `HeartChoice`'s twin, and for the same reasons.

    Which headband is on your forehead outlives any one session and is not a property of the protocol
    you happen to be running, so it lives beside the other input choices rather than inside the HEG
    screen. One object owned by the app, because BLE is exclusive: two readers of one headband is the
    same failure as two openers of the serial port.
    """

    kind: str = "off"              # off | headband | simulator
    address: str = ""              # a specific device, once one has been found
    name_hint: str = "Mendi"       # or a name filter, for the discovery fallback

    @property
    def on(self) -> bool:
        return self.kind in ("headband", "simulator")

    def problem(self) -> str | None:
        if self.kind != "headband":
            return None
        from nfb.heg import problem
        return problem()

    def describe(self) -> str:
        if self.kind == "off":
            return "no headband"
        if self.kind == "simulator":
            return "simulated headband — 30 frames/s"
        if self.address:
            return f"headband at {self.address}"
        return f"any Mendi headband{f' named ~{self.name_hint}' if self.name_hint else ''}"

    def open(self):
        """Build the frame source, or None when this is off."""
        from nfb.heg import frame_source
        if self.kind == "simulator":
            return frame_source("simulator")
        if self.kind == "headband":
            return frame_source("headband", address=self.address or None,
                                name_hint=self.name_hint)
        return None


SIMULATED_RESPONSE_MM = {"none": 0.0, "weak": 0.12, "strong": 0.35}
"""How far the simulated eye moves for a full effort, once settled. `none` shows what the screen does for
a person with no control yet; `strong` is near the top of what trained people reach."""


@dataclass
class PupilChoice:
    """Which pupil source, if any. `MendiChoice`'s twin: the eye being watched outlives any one session,
    so the choice lives with the other inputs and the app owns one reader."""

    kind: str = "off"              # off | tracker | simulator
    response: str = "weak"         # the simulated eye's answer to an effort, a key of SIMULATED_RESPONSE_MM
    eye: str = "both"              # the tracker's: both | left | right
    screen: str = "auto"           # auto | manual | the id of a display that macOS reports
    screen_width_mm: float = 600.0     # used when `screen` is manual
    screen_height_mm: float = 340.0

    def screen_size(self) -> tuple[float, float, str]:
        """The tracker's screen as (width, height, why). Read from macOS when it reports one, so the size
        is never typed in. A display that was chosen and is gone falls back to the automatic choice."""
        from .displays import displays, likely_tracker_display

        if self.screen != "manual":
            found = displays()
            chosen = next((d for d in found if str(d.id) == self.screen), None)
            if chosen is not None:
                return chosen.width_mm, chosen.height_mm, "the screen you selected"
            auto, why = likely_tracker_display(found)
            if auto is not None:
                return auto.width_mm, auto.height_mm, why
        return self.screen_width_mm, self.screen_height_mm, "the size you typed"

    @property
    def on(self) -> bool:
        return self.kind in ("tracker", "simulator")

    def problem(self) -> str | None:
        if not self.on:
            return None
        from nfb.pupil import problem
        return problem(self.kind)

    def describe(self) -> str:
        if self.kind == "off":
            return "no pupil source"
        if self.kind == "tracker":
            return f"eye tracker on USB · {self.eye}"
        return f"simulated eye · {self.response} response to effort"

    def open(self):
        from nfb.pupil import pupil_source
        if self.kind == "simulator":
            return pupil_source("simulator", response_mm=SIMULATED_RESPONSE_MM[self.response])
        if self.kind == "tracker":
            width, height, _why = self.screen_size()
            return pupil_source("tracker", eye=self.eye, screen_mm=(width, height))
        return None
