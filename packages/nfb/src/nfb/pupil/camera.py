"""The eye tracker's own infrared camera, over its USB video interface.

The tracker carries a standard USB video camera on a second interface, which the operating system does
not bind, so this drives it directly: the small video setup exchange (probe then commit of the one format
the device offers), then the video bulk endpoint. The picture is 560 by 560, 8 bits a pixel, grey. It is
a wide view of the face in infrared, for lining the eyes up — a thing to look at, not to measure.

Separate from the gaze path in `tracker`: that speaks interface 0, this speaks interface 2, and the two
can run at once on the one device.
"""

from __future__ import annotations

import struct
import time
from typing import Iterator

VENDOR_ID = 0x2104
PRODUCT_ID = 0x0313
VS_INTERFACE = 2
EP_VIDEO = 0x82

WIDTH = HEIGHT = 560
FRAME_BYTES = WIDTH * HEIGHT

SET_CUR, GET_CUR = 0x01, 0x81
PROBE, COMMIT = 0x0100, 0x0200      # control selector << 8
DEFAULT_INTERVAL = 416667           # 100 ns units → ~24 frames a second
READ_BUF = 700_000                  # over one frame, so a frame rarely splits across reads
READ_TIMEOUT_MS = 500


def problem() -> str | None:
    """Why the camera cannot be read, before trying — the USB libraries, and the device being present."""
    try:
        import usb.core
    except Exception as exc:
        return f"USB support is unavailable ({exc}) — check `uv sync`"
    try:
        import libusb_package
        backend = libusb_package.get_libusb1_backend()
    except Exception:
        backend = None
    if usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID, backend=backend) is None:
        return "no eye tracker on USB — connect it"
    return None


def _probe_payload(interval: int = DEFAULT_INTERVAL) -> bytes:
    """The video format request: the one format and frame the device offers, at its default rate."""
    b = bytearray(34)
    struct.pack_into("<H", b, 0, 0x0001)        # bmHint: hold the frame interval
    b[2] = 1                                     # bFormatIndex
    b[3] = 1                                     # bFrameIndex
    struct.pack_into("<I", b, 4, interval)
    return bytes(b)


HEADER_LENGTHS = (2, 12)
"""The video-payload header sizes this device uses. Anything else at the start of a transfer means the
transfer is not one clean payload, and it is dropped rather than risk a misaligned picture."""
SIZE_SLACK = 256
"""A clean payload is the header plus one frame, give or take a few trailing bytes; more than this means
two frames ran together and the transfer is dropped."""


def assemble_frames(transfers: Iterator[bytes]) -> Iterator[bytes]:
    """Turn a stream of video-bulk transfers into whole 560×560 frames.

    The device delivers one picture per payload, each as one bulk transfer beginning with a short header.
    So a clean transfer is a known header length plus one frame's bytes; it yields the frame. A transfer
    that starts with no header, is short, or carries two merged frames is dropped — a dropped frame only
    costs a little smoothness, while a misaligned one tears the picture across the screen.
    """
    for data in transfers:
        if len(data) < HEADER_LENGTHS[0] + 1:
            continue
        hlen = data[0]
        if hlen not in HEADER_LENGTHS:
            continue
        payload = data[hlen:]
        if FRAME_BYTES <= len(payload) <= FRAME_BYTES + SIZE_SLACK:
            yield bytes(payload[:FRAME_BYTES])


class TrackerCamera:
    """The infrared camera as a source of 560×560 grey frames, each a `bytes` of `FRAME_BYTES`.

    Frames are assembled from the video stream's payloads: each carries a short header whose end-of-frame
    bit closes a frame, and the frame-id bit flips between frames. A frame that runs long is cut to one
    picture — enough for a live view, which is all this is for.
    """

    def __init__(self):
        self._device = None
        self._usb = None
        self._stop = False
        self.error: str | None = None

    def _open(self) -> None:
        import usb.core
        import usb.util

        self._usb = usb.util
        backend = None
        try:
            import libusb_package
            backend = libusb_package.get_libusb1_backend()
        except Exception:
            backend = None
        device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID, backend=backend)
        if device is None:
            raise RuntimeError("no eye tracker on USB")
        self._device = device
        usb.util.claim_interface(device, VS_INTERFACE)
        device.ctrl_transfer(0x21, SET_CUR, PROBE, VS_INTERFACE, _probe_payload())
        negotiated = bytes(device.ctrl_transfer(0xA1, GET_CUR, PROBE, VS_INTERFACE, 34))
        device.ctrl_transfer(0x21, SET_CUR, COMMIT, VS_INTERFACE, negotiated)

    def _transfers(self) -> Iterator[bytes]:
        import usb.core

        while not self._stop:
            try:
                yield bytes(self._device.read(EP_VIDEO, READ_BUF, timeout=READ_TIMEOUT_MS))
            except usb.core.USBError as exc:
                if "timeout" in str(exc).lower():
                    yield b""
                    continue
                self.error = str(exc)
                return

    def frames(self) -> Iterator[bytes]:
        try:
            self._open()
        except Exception as exc:
            self.error = str(exc)
            return
        yield from assemble_frames(self._transfers())

    def close(self) -> None:
        self._stop = True
        device, self._device = self._device, None
        if device is not None and self._usb is not None:
            try:
                self._usb.release_interface(device, VS_INTERFACE)
                self._usb.dispose_resources(device)
            except Exception:
                pass


def grab(seconds: float = 1.5) -> bytes | None:
    """One frame, for a quick check. Opens, reads briefly, returns the last whole frame or None."""
    cam = TrackerCamera()
    last = None
    deadline = time.monotonic() + seconds
    try:
        for f in cam.frames():
            last = f
            if time.monotonic() > deadline:
                break
    finally:
        cam.close()
    return last
