"""The physical size of each screen, as macOS reports it.

The eye tracker must be told the size of the screen that it is mounted on, or it sends no data. A
display reports its own size to the computer, and macOS passes it on, so nobody has to measure it.
Read through CoreGraphics with ctypes: one system call, no extra library.

A display that reports no size, or a size no screen has, is left out; the device screen then offers
the number boxes. Anything but macOS gives an empty list, for the same result.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from dataclasses import dataclass

MIN_MM = 100.0
"""Below this a reported size is a placeholder, which some televisions and projectors send."""


@dataclass(frozen=True)
class Display:
    id: int
    builtin: bool
    main: bool
    width_mm: float
    height_mm: float

    @property
    def label(self) -> str:
        kind = "built-in" if self.builtin else "external"
        return f"{kind} · {self.width_mm:.0f} × {self.height_mm:.0f} mm"


class _Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


def displays() -> list[Display]:
    if sys.platform != "darwin":
        return []
    try:
        cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
        cg.CGGetActiveDisplayList.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                              ctypes.POINTER(ctypes.c_uint32)]
        cg.CGDisplayScreenSize.argtypes = [ctypes.c_uint32]
        cg.CGDisplayScreenSize.restype = _Size
        cg.CGDisplayIsBuiltin.argtypes = [ctypes.c_uint32]
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        ids = (ctypes.c_uint32 * 16)()
        count = ctypes.c_uint32()
        if cg.CGGetActiveDisplayList(16, ids, ctypes.byref(count)) != 0:
            return []
        main = cg.CGMainDisplayID()
        out = []
        for i in range(count.value):
            size = cg.CGDisplayScreenSize(ids[i])
            if size.width >= MIN_MM and size.height >= MIN_MM:
                out.append(Display(int(ids[i]), bool(cg.CGDisplayIsBuiltin(ids[i])), ids[i] == main,
                                   float(size.width), float(size.height)))
        return out
    except Exception:
        return []


def likely_tracker_display(found: list[Display]) -> tuple[Display | None, str]:
    """The screen the tracker is most probably on, and why. The tracker is a desk device, so an external
    screen wins over the built-in one; among several external screens, the main one."""
    if not found:
        return None, "macOS reports no screen size"
    external = [d for d in found if not d.builtin]
    if len(external) == 1:
        return external[0], "the only external screen"
    if external:
        main = next((d for d in external if d.main), external[0])
        return main, "the main external screen"
    return found[0], "the only screen"
