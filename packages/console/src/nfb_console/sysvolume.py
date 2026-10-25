"""The system output volume, as a feedback channel.

Reached through CoreAudio by ctypes: the default output device's `kAudioHardwareServiceDeviceProperty_
VirtualMainVolume`, which is the same number the volume keys move. Public framework, so unlike the
shadow's `DisplayServices` this is not private API.

**Why the system volume and not our own tone.** Sound feedback here means "make everything audible
follow the signal", and that deliberately includes the reward click, any music, and anything else
playing. Giving the click its own signal-following volume was the wrong shape: it made the *event*
carry an amount, when the event's whole content is that it happened, and it left every other sound
unaffected. One knob on the master is both simpler and more honest about what is being modulated.

The safety rules mirror `shadow.py`, for the same reasons and with the same shape:

- **`HARD_FLOOR` in code**, below any configured minimum. Feedback that reaches silence stops being
  feedback, and the operator loses the ability to hear anything at all — including the beep that says
  the session ended.
- **The operator's volume is captured and restored**, `atexit` included, so a session cannot leave the
  machine muted.
- **The target is approached, not jumped to**, because a backlight and a volume both flicker if stepped
  twenty times a second.

Measured here: a set costs ~300 µs, which is 0.6% of a 50 ms tick, so it runs inline from the level
update like the shadow does.
"""

from __future__ import annotations

import atexit
import os
import struct
import sys
from ctypes import CDLL, POINTER, Structure, byref, c_float, c_uint32, c_void_p

HARD_FLOOR = 0.05       # never below this, whatever the settings say

APPROACH_SECONDS = 0.15
"""How long the volume takes to cover most of a step, in **seconds of wall clock**.

It was `APPROACH = 0.25` per call, "per 20 Hz step" — the identical bug `shadow.APPROACH_SECONDS`
documents having been found and fixed there, left behind in this file. `_drive_outputs` runs once per
arriving block group, so the real call rate is whatever the sample rate, the channel count and the
target layout produce: measured 20.2 Hz with three targets on one channel and **60.7 Hz with the same
three spread over three channels** — the documented 150 ms time constant silently becoming 50 ms because
someone moved a target to another electrode. Time-based from a monotonic clock, so a constant in seconds
means seconds at any call rate."""

MIN_DT = 1e-4
"""Floor under the measured interval, as in `shadow`: two calls in the same instant must not divide by
zero, and a suspiciously tiny interval is more likely a double-call than real elapsed time."""


def clock() -> float:
    """The monotonic clock, as a module attribute so a test can advance time without sleeping — the same
    injection seam `shadow.clock` provides, for the same reason."""
    import time

    return time.monotonic()


def _alpha(dt: float, tau_seconds: float) -> float:
    """One-pole coefficient for a step of `dt`, exactly rate-independent — see `shadow._alpha` for why
    the exact form and not `dt / (tau + dt)`."""
    import math

    if tau_seconds <= 0:
        return 1.0
    return 1.0 - math.exp(-dt / tau_seconds)

_CA = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
_SYSTEM_OBJECT = 1


def _fourcc(text: str) -> int:
    return struct.unpack(">I", text.encode())[0]


class _Address(Structure):
    _fields_ = [("selector", c_uint32), ("scope", c_uint32), ("element", c_uint32)]


_ca = None
_error: str | None = None

if sys.platform != "darwin":
    _error = f"no system volume control on {sys.platform}"
else:
    try:
        _ca = CDLL(_CA)
        _ca.AudioObjectGetPropertyData.argtypes = [c_uint32, POINTER(_Address), c_uint32, c_void_p,
                                                   POINTER(c_uint32), c_void_p]
        _ca.AudioObjectSetPropertyData.argtypes = [c_uint32, POINTER(_Address), c_uint32, c_void_p,
                                                   c_uint32, c_void_p]
        _ca.AudioObjectHasProperty.argtypes = [c_uint32, POINTER(_Address)]
    except OSError as exc:
        _error = f"CoreAudio unavailable: {exc}"

_DEFAULT_OUTPUT = _Address(_fourcc("dOut"), _fourcc("glob"), 0) if _ca else None
_MAIN_VOLUME = _Address(_fourcc("vmvc"), _fourcc("outp"), 0) if _ca else None


def _device() -> int | None:
    if _ca is None:
        return None
    device = c_uint32()
    size = c_uint32(4)
    if _ca.AudioObjectGetPropertyData(_SYSTEM_OBJECT, byref(_DEFAULT_OUTPUT), 0, None,
                                      byref(size), byref(device)) != 0:
        return None
    return device.value


def available() -> bool:
    if _error or muted():
        return False
    device = _device()
    return device is not None and bool(_ca.AudioObjectHasProperty(device, byref(_MAIN_VOLUME)))


def muted() -> bool:
    """Whether outputs are switched off process-wide by `NFB_NO_OUTPUTS`.

    A session takes over the machine's display or volume the moment it runs, which is correct for a
    session and wrong for anything else. A test suite, a benchmark or a scratch script that imports a
    session should not be able to dim the operator's screen — and one did, which is how this exists.
    Checked on every call rather than cached, so setting it mid-process works.
    """
    return bool(os.environ.get("NFB_NO_OUTPUTS"))


def problem() -> str | None:
    """Why the volume would not move, if it would not."""
    if muted():
        return "the volume is switched off by NFB_NO_OUTPUTS"
    if _error:
        return _error
    device = _device()
    if device is None:
        return "no default output device"
    if not _ca.AudioObjectHasProperty(device, byref(_MAIN_VOLUME)):
        return "this output has no master volume (aggregate or digital device?)"
    return None


def get() -> float | None:
    device = _device()
    if device is None or _ca is None:
        return None
    value = c_float()
    size = c_uint32(4)
    if _ca.AudioObjectGetPropertyData(device, byref(_MAIN_VOLUME), 0, None,
                                      byref(size), byref(value)) != 0:
        return None
    return float(value.value)


def set(level: float) -> None:
    """Set the output volume, clamped to [HARD_FLOOR, 1]. The clamp lives here so no caller can route
    around it."""
    device = _device()
    if device is None or _ca is None:
        return
    safe = c_float(min(1.0, max(HARD_FLOOR, level)))
    _ca.AudioObjectSetPropertyData(device, byref(_MAIN_VOLUME), 0, None, 4, byref(safe))


class SystemVolume:
    """Holds the output volume for the life of a session; hands it back on release.

    Same contract as `shadow.DisplayShadow`: `apply()` owns take, follow and give-back, so a caller
    cannot forget to restore.
    """

    def __init__(self) -> None:
        self.initial: float | None = None
        self.current: float = 1.0
        self.held = False
        self._registered = False
        self._last: float | None = None

    def apply(self, enabled: bool, low: float, high: float, level: float) -> None:
        """`low`/`high` are percentages, `level` is 0…1."""
        if not enabled or not available():
            self.release()
            return
        if not self.held:
            self.initial = get()
            self.current = self.initial if self.initial is not None else 0.5
            self.held = True
            if not self._registered:
                atexit.register(self.release)
                self._registered = True
        lo, hi = sorted((low / 100.0, high / 100.0))
        target = lo + (hi - lo) * min(1.0, max(0.0, level))
        now = clock()
        dt = MIN_DT if self._last is None else max(MIN_DT, now - self._last)
        self._last = now
        self.current += _alpha(dt, APPROACH_SECONDS) * (target - self.current)
        set(self.current)

    def release(self) -> None:
        if self.held and self.initial is not None:
            set(self.initial)
        self.held = False
        self.initial = None

    def describe(self) -> str:
        if not available():
            return problem() or "unavailable"
        if not self.held:
            return "not holding the output volume"
        return f"holding the system volume at {self.current * 100:.0f}%"
