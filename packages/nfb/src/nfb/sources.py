"""Sample sources: the live device, and a recording played back through the same decoder.

Both satisfy stream.SampleSource, so anything downstream — a metric, a session, a plot — cannot
tell them apart. That is the property that lets processing be built and regression-tested with
no hardware attached.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import optima

from .decode import FrameDecoder
from .reader import FrameReader
from .record import SessionHeader, SessionReader, SessionWriter
from .stream import NO_SIGNAL, OK, Block, ChannelSpec, Gap


class OptimaSource:
    """A live Optima session: configure, start, and stream blocks until closed.

    Optionally records every frame as it arrives. Recording is done here rather than higher up
    so that what lands on disk is exactly what the device sent, before any interpretation.
    """

    def __init__(self, config: bytes | None = None, port: str | None = None,
                 record_to: str | Path | None = None, test_mode: bool = False,
                 note: str = "", threaded: bool = True):
        problems = optima.validate_config(config) if config is not None else []
        if problems:
            raise ValueError("invalid config: " + "; ".join(problems))
        self.config = config if config is not None else optima.build_eeg_config()
        self.test_mode = test_mode
        self.threaded = threaded
        self.dev = optima.Optima(port)
        self.reader: FrameReader | None = None
        self.decoder = FrameDecoder(self.config, test_mode=test_mode)
        self._writer: SessionWriter | None = None
        self._t0: float | None = None
        self._started = False

        info = self.dev.device_info()
        model = self._text(info, 0x23)
        serial = self._text(info, 0x44)
        firmware = ".".join(str(b) for b in self.dev.firmware_info()[:3])
        self.dev.write_config(self.config)
        if record_to is not None:
            header = SessionHeader(
                config=self.config, device=model, serial=serial, firmware=firmware,
                started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                note=note, test_mode=test_mode,
            )
            self._writer = SessionWriter(record_to, header)

    @staticmethod
    def _text(info: bytes, frame_offset: int, length: int = 24) -> str:
        """Read a NUL-terminated string from a reply. Offsets in the protocol doc are
        frame-relative; `info` starts after the code and status bytes."""
        chunk = info[frame_offset - 2:frame_offset - 2 + length]
        return chunk.split(b"\0")[0].decode("latin-1")

    def channels(self) -> dict[int, ChannelSpec]:
        return self.decoder.channels()

    def start(self) -> None:
        if self._started:
            return
        self.dev.set_notifications(optima.NOTIF_ENABLE)
        if self.test_mode:
            self.dev.request(optima.CMD["START_TEST"])
        else:
            self.dev.start_measurement()
        self._t0 = time.monotonic()
        self._started = True
        if self.threaded:
            # started only after the device is streaming, so the thread never spins on an idle link
            self.reader = FrameReader(self.dev)
            self.reader.start()

    def blocks(self) -> Iterator[Block | Gap]:
        """Yield decoded blocks.

        With `threaded` (the default) a reader thread owns the port, so time spent here decoding or
        downstream computing cannot delay acquisition. Without it, reading and processing share one
        loop and a scheduling hiccup shows up as latency — see reader.py for the measurement.
        """
        self.start()
        if self.reader is not None:
            for arrival in self.reader.arrivals():
                if self._writer is not None:
                    self._writer.write(arrival.code, arrival.payload, arrival.at - self._t0)
                yield from self.decoder.feed(arrival.code, arrival.payload)
        else:
            while True:
                msg = self.dev.read_message()
                if self._writer is not None:
                    self._writer.write(msg.code, msg.payload, time.monotonic() - self._t0)
                yield from self.decoder.feed(msg.code, msg.payload)

    def close(self) -> None:
        if self.reader is not None:
            self.reader.stop()          # stop reading before issuing teardown commands
            self.reader = None
        try:
            if self._started:
                if self.test_mode:
                    self.dev.request(optima.CMD["STOP_TEST"])
                else:
                    self.dev.stop_measurement()
                self.dev.set_notifications(optima.NOTIF_CLEAR)
        except Exception:
            pass          # teardown must not mask whatever ended the session
        finally:
            if self._writer is not None:
                self._writer.close()
            self.dev.close()

    def __enter__(self) -> "OptimaSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ReplaySource:
    """Replays a recorded session through the identical decoder.

    Runs as fast as it can read by default. `realtime=True` paces playback using the recorded
    arrival times, for exercising anything that cares about wall-clock behaviour.

    `loop=True` starts over when the file runs out, which turns a short capture into an indefinite
    stand-in for a device. The **same decoder** is kept across the seam, deliberately: recreating it
    would restart sample indices at zero and they are monotonic by contract. Keeping it means the phase
    counter appears to jump, so a `Gap` is emitted at each seam and every filter downstream resets —
    which is the truthful description of what happened, since the signal really is discontinuous there.
    """

    def __init__(self, path: str | Path, realtime: bool = False, loop: bool = False):
        self.path = Path(path)
        self.reader = SessionReader(path)
        self.header = self.reader.header
        self.realtime = realtime
        self.loop = loop
        self.decoder = FrameDecoder(self.header.config, test_mode=self.header.test_mode)
        self._stop = False

    @property
    def config(self) -> bytes:
        """The config the frames were recorded under, which is the one they decode with.

        Named to match `OptimaSource`, `WaitingSource` and `Subscription` so a caller can ask any
        frame-based source what it is decoding with and never branch on which kind it holds. Settings
        that are only in the blob — the high-pass corner especially — are otherwise unreachable
        offline, which is how a screen came to assume 0.5 Hz while replaying a 0.01 Hz capture.
        """
        return self.header.config

    def channels(self) -> dict[int, ChannelSpec]:
        return self.decoder.channels()

    def blocks(self) -> Iterator[Block | Gap]:
        t0 = time.monotonic()
        played = 0.0                     # seconds of material already played, across loops
        while not self._stop:
            last = 0.0
            for code, payload, elapsed in self.reader.frames():
                if self._stop:
                    return
                if elapsed is not None:
                    last = elapsed
                    if self.realtime:
                        behind = played + elapsed - (time.monotonic() - t0)
                        if behind > 0:
                            time.sleep(behind)
                yield from self.decoder.feed(code, payload)
            if not self.loop:
                return
            played += last
            # A loop is a splice, not a hole: without the rebase the recording's phase counter
            # restarting reads as a gap of ~63k groups, and every filter downstream resets and
            # re-runs its settling time once per loop. `SampleFileSource` documents the same
            # contract for itself below.
            self.decoder.rebase()
            self.reader = SessionReader(self.path)

    def close(self) -> None:
        self._stop = True

    def __enter__(self) -> "ReplaySource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SampleFileSource:
    """Plays back a plain text file of samples — one row per sample, one column per channel.

    Not the same thing as `ReplaySource`. A recorded session holds raw device frames and is replayed
    through the identical decoder, which is what makes an offline result provably equal to a live one.
    This reads *already decoded* microvolts from something else's export, so there is no decoder in the
    path and no device to be faithful to. Both still satisfy `stream.SampleSource`, so a metric, a
    session or a plot cannot tell any of the three apart.

    Format, as produced by the emotional-n-back recorder:

        Sampling rate: 256
        -6.66455<TAB>-11.6064<TAB>0<TAB>0
        ...

    A leading `Sampling rate:` line is used when present; otherwise pass `rate_hz`. Values are
    microvolts. `Block.samples` is integral by contract, so they are stored as millivolt-thousandths
    (1 nV resolution) with the scale set to match — precision far below anything an EEG amplifier
    resolves, and it keeps the block contract honest rather than widening it for one file format.

    **A column that is zero throughout is reported as having no signal** (`stream.NO_SIGNAL`, so `valid`
    is False) rather than as a channel reading a perfectly flat 0 µV. Exports pad unused channels with
    zeros, and a flat line is otherwise indistinguishable from a working channel on a dead subject. It uses
    the device's own code for an open input, which is what a padded column stands for; it previously used 0,
    and 0 is what the amplifier reports for a channel that is **fine**.

    `loop=True` restarts at the top when the file runs out, which is what makes a short capture usable
    as a stand-in for a live device. Sample indices keep counting across the seam — they are contiguous
    and monotonic by contract, and a loop is not a dropped sample, so no `Gap` is emitted. The signal
    itself does jump at the seam, and a filter will ring for its group delay afterwards; with 74 s of
    material that is a transient every 74 s and worth knowing about before reading a spectrum.
    """

    NANOVOLT = 1e-3        # microvolts per stored count

    def __init__(self, path: str | Path, rate_hz: float | None = None, loop: bool = True,
                 realtime: bool = True, block_size: int = 8):
        self.path = Path(path)
        self.loop = loop
        self.realtime = realtime
        self.block_size = block_size
        self._stop = False

        rows: list[list[float]] = []
        header_rate: float | None = None
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                fields = line.replace(",", ".").split("\t")
                try:
                    rows.append([float(x) for x in fields])
                except ValueError:
                    if header_rate is None and "rate" in line.lower():
                        header_rate = self._rate_from(line)
                    continue        # any other prose is a header we do not need
        if not rows:
            raise ValueError(f"{self.path} holds no sample rows")

        self.rate_hz = rate_hz or header_rate
        if self.rate_hz is None:
            raise ValueError(f"{self.path} has no sampling-rate line; pass rate_hz")

        width = max(len(r) for r in rows)
        self.data = [[(r[c] if c < len(r) else 0.0) for r in rows] for c in range(width)]
        self.active = [any(v != 0.0 for v in col) for col in self.data]
        self.samples = len(rows)
        self.specs = {
            c: ChannelSpec(index=c, rate_hz=self.rate_hz, unit="uV", scale=self.NANOVOLT,
                           function="voltage")
            for c in range(width)
        }
        self.counts = [[int(round(v / self.NANOVOLT)) for v in col] for col in self.data]

    @staticmethod
    def _rate_from(line: str) -> float | None:
        for token in line.replace(":", " ").split():
            try:
                return float(token)
            except ValueError:
                continue
        return None

    @property
    def seconds(self) -> float:
        return self.samples / self.rate_hz

    def channels(self) -> dict[int, ChannelSpec]:
        return dict(self.specs)

    def blocks(self) -> Iterator[Block | Gap]:
        """Yield one block per channel per group of `block_size` samples.

        Interleaved channel by channel within each group, in the order the device does it, so a
        consumer that assumes channels advance together is exercised the same way.
        """
        t0 = time.monotonic()
        emitted = 0                      # absolute samples emitted, across loops
        cursor = 0
        while not self._stop:
            take = min(self.block_size, self.samples - cursor)
            for c, counts in enumerate(self.counts):
                yield Block(channel=c, start_index=emitted,
                            samples=tuple(counts[cursor:cursor + take]),
                            spec=self.specs[c],
                            state=OK if self.active[c] else NO_SIGNAL)
            cursor += take
            emitted += take
            if cursor >= self.samples:
                if not self.loop:
                    return
                cursor = 0
            if self.realtime:
                behind = emitted / self.rate_hz - (time.monotonic() - t0)
                if behind > 0:
                    time.sleep(behind)

    def close(self) -> None:
        self._stop = True

    def __enter__(self) -> "SampleFileSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class WaitingSource:
    """The channel layout with no samples in it: a live session whose amplifier is not connected.

    Opening a protocol screen must not require the device. `--live` with the amplifier switched off used
    to raise out of the framing layer — `config blob too short: 0 bytes`, because `Device.config` is
    empty until a connection writes one — and the fix that suggests itself, refusing to open the screen,
    is worse than the crash: the status chip is already red and already says what to do, so blocking the
    press adds nothing and takes the screen away.

    The layout is known without a device, since it comes from `Settings` rather than from the amplifier,
    so every trace and card draws its own **"waiting for data"** — which is the truth, and which the
    screens already have a state for. No block is ever emitted, so nothing is measured, no target is fed
    and the state of every channel stays `None`: not heard from, as against known bad.

    It does not become live when the device connects. A session built on this is replaced by one built on
    a subscription, which is what going to the device screen and coming back already does.
    """

    def __init__(self, config: bytes | None = None):
        self.config = config if config is not None else optima.build_eeg_config()
        self._specs = FrameDecoder(self.config).channels()
        self._stop = False

    def channels(self) -> dict[int, ChannelSpec]:
        return dict(self._specs)

    def blocks(self) -> Iterator[Block | Gap]:
        # Not a `return`: the feed thread has to stay alive so that `feeding` stays true and the session
        # is not mistaken for one that has ended. It sleeps, which is what waiting is.
        while not self._stop:
            time.sleep(0.2)
            yield from ()

    def close(self) -> None:
        self._stop = True

    def __enter__(self) -> "WaitingSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
