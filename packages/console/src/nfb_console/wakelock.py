"""Keep the subject's display awake while a session runs.

A person doing neurofeedback sits still and gives the machine no keyboard or mouse input, so macOS runs
down its idle timer and dims, then sleeps, the display — in the middle of training, with nothing wrong.
This holds a power assertion for the life of a session that stops the idle-sleep timer, and drops it the
moment the session ends, pauses to the device screen, or goes home, so the machine behaves normally the
rest of the time.

The assertion is IOKit's `PreventUserIdleDisplaySleep`, a public API (unlike the shadow's private
`DisplayServices`), reached by ctypes. It stops the idle dimming only: it does not set the brightness, so
the shadow still drives the backlight as feedback, and it does not stop a manual sleep or the lid closing.

The same `NFB_NO_OUTPUTS` switch that silences the shadow and the volume disables this too: a test or a
scratch script that starts a session must not pin the operator's display awake. The assertion is released
at `atexit` as well, so a process that dies holding one cannot leave the machine unable to sleep.
"""

from __future__ import annotations

import atexit
import sys
from ctypes import CDLL, POINTER, byref, c_char_p, c_int32, c_uint32, c_void_p

from . import developer

ASSERTION_NAME = "Fribord neurofeedback session"
"""The name the assertion carries, as `pmset -g assertions` lists it, so the reason the display is awake
is legible from outside the program."""

_ASSERT_TYPE = "PreventUserIdleDisplaySleep"    # kIOPMAssertionTypePreventUserIdleDisplaySleep
_LEVEL_ON = 255                                 # kIOPMAssertionLevelOn
_UTF8 = 0x08000100                              # kCFStringEncodingUTF8

_IOKIT = "/System/Library/Frameworks/IOKit.framework/IOKit"
_CF = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"

_iokit = None
_cf = None
_error: str | None = None

if sys.platform != "darwin":
    _error = f"no display-sleep control on {sys.platform}"
else:
    try:
        _iokit = CDLL(_IOKIT)
        _cf = CDLL(_CF)
        _cf.CFStringCreateWithCString.restype = c_void_p
        _cf.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
        _cf.CFRelease.argtypes = [c_void_p]
        _iokit.IOPMAssertionCreateWithName.argtypes = [c_void_p, c_uint32, c_void_p, POINTER(c_uint32)]
        _iokit.IOPMAssertionCreateWithName.restype = c_int32
        _iokit.IOPMAssertionRelease.argtypes = [c_uint32]
        _iokit.IOPMAssertionRelease.restype = c_int32
    except OSError as exc:
        _error = f"IOKit unavailable: {exc}"

_held_id: int | None = None
_registered = False


def muted() -> bool:
    """Whether the idle-sleep hold is switched off process-wide. See `developer.outputs_off` for why it
    exists: a session takes over the machine the moment it runs, and a test must not."""
    return developer.outputs_off()


def available() -> bool:
    return not muted() and _error is None and _iokit is not None


def problem() -> str | None:
    """Why the display would still dim, if it would."""
    if muted():
        return f"the display-sleep hold is switched off by {developer.outputs_off_by()}"
    return _error


def _cfstr(text: str) -> c_void_p:
    return c_void_p(_cf.CFStringCreateWithCString(None, text.encode("utf-8"), _UTF8))


def _create(assertion_type: str, name: str) -> int | None:
    """Create a power assertion, returning its id, or None on failure. The one boundary a test gates."""
    type_ref = _cfstr(assertion_type)
    name_ref = _cfstr(name)
    out = c_uint32(0)
    try:
        rc = _iokit.IOPMAssertionCreateWithName(type_ref, _LEVEL_ON, name_ref, byref(out))
    finally:
        _cf.CFRelease(type_ref)
        _cf.CFRelease(name_ref)
    return out.value if rc == 0 else None


def _release(assertion_id: int) -> None:
    _iokit.IOPMAssertionRelease(assertion_id)


def hold() -> None:
    """Stop the display dimming or sleeping on its idle timer. Idempotent: a second call while already
    holding does nothing, so every session-start path can call it without keeping a count."""
    global _held_id, _registered
    if _held_id is not None or not available():
        return
    _held_id = _create(_ASSERT_TYPE, ASSERTION_NAME)
    if _held_id is not None and not _registered:
        atexit.register(release)
        _registered = True


def release() -> None:
    """Let the display dim and sleep normally again. Idempotent."""
    global _held_id
    if _held_id is None:
        return
    _release(_held_id)
    _held_id = None


def describe() -> str:
    if not available():
        return problem() or "unavailable"
    return "holding the display awake" if _held_id is not None else "not holding the display awake"
