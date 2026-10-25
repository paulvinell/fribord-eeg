"""The shadow: the subject's display dimming and lifting with the signal.

macOS exposes brightness through the **private** `DisplayServices` framework, reached by ctypes. It
ships with the OS, so there is nothing to install, and it is the same door the `brightness` CLI uses.
Being private, it can change between releases — hence `available()` and `problem()`, so a pane can say
why the screen is not moving instead of presenting a control that quietly does nothing.

Three things here are safety rather than features, and they are the reason this is not four lines
inline:

- **`HARD_FLOOR` is enforced in code**, below whatever anyone configures. A screen driven to black is
  harder to recover from than any training gain is worth — you cannot see the button that stops it.
- **The operator's brightness is captured and restored.** A session must not leave the machine dimmed,
  including when the process dies, which is why the restore is also registered with `atexit`.
- **The target is approached, not jumped to.** Even a smoothed signal moves in steps, and stepping the
  backlight twenty times a second is visible flicker; a ~150 ms approach reads as one continuous change.

Measured on this machine: `DisplayServicesSetBrightness` costs **6 µs** and the getter 37 µs, so this is
driven straight from the 20 Hz level update rather than from a loop of its own. That is a deliberate
difference from the version in `mendi-ble` this was taken from, which needed its own interpolation task
because its signal arrived at ~2 Hz.
"""

from __future__ import annotations

import atexit
import os
import ctypes
import math
import sys
import time
from ctypes import POINTER, byref, c_float, c_int, c_uint32


# Never below this, whatever the settings say.
HARD_FLOOR = 0.10
MAX_DISPLAYS = 16

APPROACH_SECONDS = 0.15
"""How long the backlight takes to cover most of a step, in **seconds of wall clock**.

**It was `APPROACH = 0.25` per call, and per-call is the bug.** The module claimed to be driven from the
20 Hz level update, so 0.25 per step was a 150 ms time constant. It is really driven from
`TrainingSession._drive_outputs`, which runs whenever any target advanced — once per arriving block in the
worst case. Measured 21 Hz on a two-channel 250 sps replay, which is why it looked right, and a four-channel
1000 sps session delivers blocks at 500 a second. Every time constant in this file was therefore wrong by
whatever ratio the sample rate, the channel count and the number of targets happened to produce, and the
faster the session the *less* smoothing it got — so the shadow flickered worst exactly where the signal
arrived fastest.

Time-based from a monotonic clock, so a constant in seconds means seconds at any call rate."""

def clock() -> float:
    """The monotonic clock, as a module attribute so a test can advance time without sleeping.

    The smoothing here is in seconds of wall clock — that is the whole point of it, since the call rate is
    whatever the sample rate and channel count produce — so a test that drives `apply` in a tight loop passes
    no time at all and nothing moves. Injecting the clock is the only way to exercise a time constant
    deterministically, and a `sleep`-based test of a 0.4 s constant would take real seconds and be flaky.
    """
    return time.monotonic()


MIN_DT = 1e-4
"""Floor under the measured interval. Two calls in the same instant would otherwise divide by zero, and a
suspiciously tiny interval is more likely a double-call than real elapsed time."""


def _alpha(dt: float, tau_seconds: float) -> float:
    """One-pole coefficient for a step of `dt`, **exactly** rate-independent.

    `dt / (tau + dt)` is the usual form and it is only an approximation: it lags at coarse steps, so the same
    time constant reached different brightnesses at 20 Hz and 200 Hz — 90% against 99% over 0.3 s, which is
    the very rate-dependence this module is being fixed to remove. `1 - exp(-dt/tau)` is the exact
    discretisation and costs one `exp` per call against a 6 µs write to the display.
    """
    if tau_seconds <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau_seconds)
_CG = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
_DS = "/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices"

_cg = _ds = None
_error: str | None = None

if sys.platform != "darwin":
    _error = f"no display brightness control on {sys.platform}"
else:
    try:
        _cg = ctypes.CDLL(_CG)
        _ds = ctypes.CDLL(_DS)
        _cg.CGMainDisplayID.restype = c_uint32
        _cg.CGGetActiveDisplayList.argtypes = [c_uint32, POINTER(c_uint32), POINTER(c_uint32)]
        _cg.CGGetActiveDisplayList.restype = c_int
        _ds.DisplayServicesGetBrightness.argtypes = [c_uint32, POINTER(c_float)]
        _ds.DisplayServicesGetBrightness.restype = c_int
        _ds.DisplayServicesSetBrightness.argtypes = [c_uint32, c_float]
        _ds.DisplayServicesSetBrightness.restype = c_int
    except OSError as exc:
        _error = f"DisplayServices unavailable: {exc}"


def available() -> bool:
    return _error is None and not muted()


def muted() -> bool:
    """Whether outputs are switched off process-wide by `NFB_NO_OUTPUTS`.

    A session takes over the machine's display or volume the moment it runs, which is correct for a
    session and wrong for anything else. A test suite, a benchmark or a scratch script that imports a
    session should not be able to dim the operator's screen — and one did, which is how this exists.
    Checked on every call rather than cached, so setting it mid-process works.
    """
    return bool(os.environ.get("NFB_NO_OUTPUTS"))


def problem() -> str | None:
    """Why the screen would not move, if it would not."""
    if muted():
        return "the shadow is switched off by NFB_NO_OUTPUTS"
    if _error:
        return _error
    if not displays():
        return "no active display"
    return None


def displays() -> list[int]:
    """Active display ids, in the order macOS reports them. The first is the main one.

    **Empty when outputs are muted**, so a test run makes no display call at all. `NFB_NO_OUTPUTS` promises
    that nothing but a session touches the operator's screen, and the suite was still asking WindowServer for
    the display list a few dozen times a run — through the Feedback pane's "N attached" note, which is drawn
    on every screen. The measured brightness never moved, so this is not the dim-the-screen failure the flag
    was written for; it is the same promise held one call earlier, where it is cheap to hold. A muted process
    has no business opening a connection to the window server to count monitors it may not touch.
    """
    if _cg is None or muted():
        return []
    count = c_uint32()
    ids = (c_uint32 * MAX_DISPLAYS)()
    if _cg.CGGetActiveDisplayList(MAX_DISPLAYS, ids, byref(count)) != 0:
        return []
    return [ids[i] for i in range(count.value)]


def subject_screen() -> int | None:
    """The 1-based screen number of a display that is **not** the operator's, or None if there is only one.

    What this answers is "is there a screen the trainee is looking at", which is the question that decides
    whether a protocol may switch its shadow on by itself. The first display macOS reports is the main one
    and that is the console's own, so a second display is the first thing that can be dimmed without dimming
    the person running the session.

    Returns a screen *number* rather than a boolean because the two decisions are one decision: a shadow
    enabled by default has to point somewhere, and pointing it at `screen = 1` while claiming the second
    display made it safe would dim the operator anyway. Muted processes see no displays at all, so this is
    None under `NFB_NO_OUTPUTS` and a protocol's default lands off — which is what a test run wants.
    """
    return 2 if len(displays()) > 1 else None


def display_for(screen: int) -> int | None:
    """Resolve a 1-based screen number. Out of range falls back to the main display rather than doing
    nothing — a shadow on the wrong screen is a visible mistake, a shadow on no screen is silence."""
    found = displays()
    if not found:
        return None
    index = max(1, screen) - 1
    return found[index] if index < len(found) else found[0]


def get(display: int) -> float | None:
    if _ds is None or muted():
        return None
    value = c_float()
    if _ds.DisplayServicesGetBrightness(display, byref(value)) != 0:
        return None
    return float(value.value)


LOG = "/tmp/nfb-shadow.log"
"""Where `NFB_DEBUG=1` records every brightness write, with the thread and the caller that made it.

**Because a shadow symptom cannot be reproduced from a recording.** A replay through the same session
objects writes one smooth series from one thread — measured, 309 distinct values over 15 s from
`_drive_outputs` alone — while the live app was reported flickering between two fixed points, which is the
signature of two writers rather than of any time constant. The shape of
this was seen before: right values, right objects, wrong thing on screen, and two wrong diagnoses made by guessing before
`NFB_DEBUG` was consulted. This is that switch for the display.

    NFB_DEBUG=1 make live      # then read /tmp/nfb-shadow.log

Each line is `value  [thread]  caller < caller < caller`. One repeating thread with a slowly changing value
is healthy; two threads, or one thread alternating between two values, names the second writer."""


def _log(display: int, level: float) -> None:
    if not os.environ.get("NFB_DEBUG"):
        return
    import threading
    import traceback
    try:
        stack = traceback.extract_stack()[-5:-2]
        who = " < ".join(f"{f.name}:{f.lineno}" for f in reversed(stack))
        with open(LOG, "a") as fh:
            fh.write(f"{time.monotonic():10.3f} d{display} {level:.4f}  "
                     f"[{threading.current_thread().name}]  {who}\n")
    except OSError:
        pass                    # diagnostics must never take a session down


def set(display: int, level: float) -> None:
    """Set brightness, clamped to [HARD_FLOOR, 1]. The clamp is here, at the boundary, so no caller can
    route around it."""
    _log(display, level)
    if _ds is None or muted():
        return
    _ds.DisplayServicesSetBrightness(display, c_float(min(1.0, max(HARD_FLOOR, level))))


class DisplayShadow:
    """Holds one display's brightness for the life of a session.

    `apply()` is called on every level update and is responsible for the whole lifecycle — taking the
    display when the mechanism is switched on, following the signal, and giving it back when it is
    switched off. Callers therefore cannot forget to restore, which is the failure that matters.
    """

    def __init__(self) -> None:
        self.display: int | None = None
        self.initial: float | None = None
        self.current: float = 1.0
        self.smoothed: float = 1.0
        self._last: float | None = None
        self._registered = False

    @property
    def active(self) -> bool:
        return self.display is not None

    def apply(self, enabled: bool, screen: int, low: float, high: float, level: float,
              smoothing: float = 0.0) -> None:
        """`low`/`high` are percentages, `level` is 0…1, `smoothing` is the protocol's own time constant."""
        if not enabled or not available():
            self.release()
            return
        wanted = display_for(screen)
        if wanted is None:
            self.release()
            return
        if wanted != self.display:
            self.release()          # moving to another screen gives the first one back
            self.display = wanted
            self.initial = get(wanted)
            self.current = self.initial if self.initial is not None else 1.0
            self.smoothed = self.current
            self._last = None
            if not self._registered:
                atexit.register(self.release)
                self._registered = True

        lo, hi = sorted((low / 100.0, high / 100.0))
        target = lo + (hi - lo) * min(1.0, max(0.0, level))

        # Two one-poles in series, both in seconds of wall clock: the fixed approach that keeps a step from
        # being a jump, and the protocol's own `smoothing_seconds` on top of it. Separate stages because they
        # answer different questions — the first is about the backlight, the second about the signal — and
        # collapsing them would make a screen that wants no extra smoothing lose the approach as well.
        now = clock()
        dt = MIN_DT if self._last is None else max(MIN_DT, now - self._last)
        self._last = now

        self.smoothed += _alpha(dt, smoothing) * (target - self.smoothed)
        self.current += _alpha(dt, APPROACH_SECONDS) * (self.smoothed - self.current)
        set(self.display, self.current)

    def release(self) -> None:
        """Give the display back at the brightness we found it. Safe to call repeatedly."""
        if self.display is not None and self.initial is not None:
            set(self.display, self.initial)
        self.display = None
        self.initial = None

    def describe(self) -> str:
        if not available():
            return problem() or "unavailable"
        if not self.active:
            return "not holding the display"
        which = displays().index(self.display) + 1 if self.display in displays() else 0
        return f"holding display {which} at {self.current * 100:.0f}%"
