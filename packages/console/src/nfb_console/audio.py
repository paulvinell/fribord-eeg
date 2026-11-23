"""Sound output: the one path every generated sample takes to the speaker, and the rules that keep it clean.

Crackle is a class here, fixed once. Nothing else opens an output stream or plays a generated sound:

- a step in the waveform: `finish` starts and ends every short sound at zero; `Stream` fades a start, a stop
  and a late producer, and `ramp` spreads a gain change across a block;
- a starved stream: no callback, a writer thread fills a buffer the device drains without Python;
- a hard clip: short sounds peak at `PEAK`, and every block passes `soft_limit`;
- a cone at its excursion limit: `roll_off` and `partial_roll_off` at the one measured corner;
- a device opened per sound: short sounds share one open stream, `CueStream`;
- a rate converted elsewhere: `CueStream` runs at the device's rate, and `resample` is polyphase.

The one exception is a music file the operator chooses, which `music.TrackPlayer` plays through `afplay`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import queue
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import developer

SPEAKER_CORNER_HZ = 70.0
"""The small-speaker corner. Below it a laptop driver only travels, and what is heard is the crackle of a
cone at its excursion limit. Measured on pink noise: 28% of its power sat below 80 Hz before this corner,
4.2% after it, at order 2."""
CORNER_ORDER = 2

PEAK = 0.45
"""The peak of every short sound, as a fraction of full scale: -6.9 dBFS. A pure tone near full scale makes a
small speaker buzz, and at this peak two sounds at once still sum under full scale."""

ONSET_MS = 0.3
"""Every short sound rises from exactly zero over at least this long. Short enough that a click stays hard."""

RELEASE_MS = 8.0
"""Every short sound falls to exactly zero over its last few milliseconds. Half a cycle of the lowest tone
rendered is 4.5 ms, so the ramp always spans a full cycle or more."""

FADE_MS = 5.0
"""A stop, a start and a late producer are faded over this long: long enough that a full-scale step is
inaudible, far shorter than anything a listener reads as a change in the sound."""

KNEE = 2 * PEAK
"""Where `soft_limit` starts to bend: two short sounds at their peak pass as they are."""

STALL_SECONDS = 2.0
"""How long a stream may go without finishing a write, or without the device taking what it was given,
before it is treated as dead. Generous next to the buffers underneath."""
REOPEN_TRIES = 3
REOPEN_PAUSE_SECONDS = 0.25
SHUT_WAIT_SECONDS = 2.0
"""How long `shut_aside` waits for a stream's writer to finish its last write before it shuts the stream."""

WAV_RATE = 44100
"""The rate of the files written for the `afplay` fallback."""

_last_error: str | None = None


def muted() -> bool:
    """Outputs off silences audio for the process. See `developer.outputs_off` for why it exists."""
    return developer.outputs_off()


# --- the finishing rules ------------------------------------------------------------------------

def corner_sos(rate: float):
    """The speaker-corner high-pass, designed once per rate."""
    from scipy.signal import butter

    return butter(CORNER_ORDER, SPEAKER_CORNER_HZ, btype="highpass", fs=rate, output="sos")


def roll_off(x: np.ndarray, rate: float) -> np.ndarray:
    """Content below the speaker corner taken out, from a filter at rest."""
    from scipy.signal import sosfilt

    return sosfilt(corner_sos(rate), np.asarray(x, dtype=float))


def partial_roll_off(freqs) -> np.ndarray:
    """The corner for a generator that makes its partials itself: a gain per frequency, smooth, so a partial
    that slides through the corner fades rather than steps. Fourth power below the corner, 24 dB an octave."""
    return 1.0 / (1.0 + (SPEAKER_CORNER_HZ / np.maximum(np.asarray(freqs, dtype=float), 1e-9)) ** 4)


def fade_in(x: np.ndarray, ms: float, rate: float) -> np.ndarray:
    """Raised-cosine rise over the first `ms`, starting at exactly zero. In place, and returned."""
    n = min(len(x), max(1, int(rate * ms / 1000.0)))
    x[:n] *= 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))
    return x


def fade_out(x: np.ndarray, ms: float, rate: float) -> np.ndarray:
    """Raised-cosine fall over the last `ms`, ending at exactly zero. In place, and returned."""
    n = min(len(x), max(1, int(rate * ms / 1000.0)))
    x[-n:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(1, n + 1) / n))
    return x


def ramp(start: float, end: float, frames: int) -> np.ndarray:
    """A gain change spread across a block. Held for a block and then changed, a gain is a step."""
    return np.linspace(start, end, frames)


def soft_limit(mix: np.ndarray) -> np.ndarray:
    """The sum bent smoothly above `KNEE` so that it never reaches full scale. The bend has the slope of the
    line where it starts, so it adds no step. In place, and returned."""
    over = np.abs(mix) > KNEE
    if over.any():
        size = np.abs(mix[over])
        mix[over] = np.sign(mix[over]) * (KNEE + (1.0 - KNEE) * np.tanh((size - KNEE) / (1.0 - KNEE)))
    return mix


def resample(x: np.ndarray, source: float, rate: float) -> np.ndarray:
    """`x` at `rate`, by a polyphase filter. Unchanged when the rates agree."""
    from scipy.signal import resample_poly

    src, dst = int(round(source)), int(round(rate))
    if src == dst:
        return np.asarray(x, dtype=float)
    common = int(np.gcd(src, dst))
    return resample_poly(np.asarray(x, dtype=float), dst // common, src // common)


def finish(x: np.ndarray, rate: float, peak: float | None = PEAK, onset_ms: float = ONSET_MS,
           release_ms: float = RELEASE_MS) -> np.ndarray:
    """The one way a short sound is made ready for the speaker.

    In order: the speaker corner, a rise from exactly zero, a fall to exactly zero, then the level. With a
    `peak` the sound is scaled to it; with None it keeps its level, under `PEAK` at most. The result is float
    in -1..1 at `rate`."""
    out = roll_off(x, rate)
    fade_out(fade_in(out, onset_ms, rate), release_ms, rate)
    top = float(np.abs(out).max()) if len(out) else 0.0
    if top > 0.0:
        out *= (peak / top) if peak is not None else min(1.0, PEAK / top)
    return out


def counts(x: np.ndarray) -> np.ndarray:
    """A finished sound as 16-bit counts, for a file."""
    return np.round(np.asarray(x) * 32767).astype(np.int16)


# --- streams ------------------------------------------------------------------------------------

def shut_aside(stream, writer: threading.Thread | None = None) -> None:
    """Stop and close an audio stream on a thread of its own, after its writer, if given, has finished.

    Never on the thread that asked for a sound. That is the screen's thread, and closing there hung the
    whole program: the screen's thread waited inside CoreAudio for a lock held by CoreAudio's own thread,
    which was running PortAudio's start-and-stop callback and waited for a lock the screen's thread held.
    If the close deadlocks here, only this thread is lost.
    """
    if stream is None:
        return

    def shut() -> None:
        if writer is not None and writer is not threading.current_thread():
            writer.join(timeout=SHUT_WAIT_SECONDS)
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass                        # a stream that will not shut has nothing more to give anyone

    threading.Thread(target=shut, daemon=True, name="audio-shut").start()


def _device_rate() -> int:
    import sounddevice as sd

    return int(round(sd.query_devices(kind="output")["default_samplerate"]))


def granted_latency(samplerate: float | None, blocksize: int, latency) -> float:
    """What the device grants for a stream of these settings, in seconds. Opens a stream, writes nothing,
    and shuts it. For the latency bench, so that no other module opens a stream."""
    import sounddevice as sd

    stream = sd.OutputStream(samplerate=samplerate or _device_rate(), channels=1, dtype="float32",
                             blocksize=blocksize, latency=latency)
    stream.start()
    try:
        return float(stream.latency)
    finally:
        stream.stop()
        stream.close()


class Stream:
    """An output stream fed by a writer thread. A subclass supplies `_block(frames)`: the next block, or None
    when it has nothing ready. The buffer, the fades, the limit and the recovery are here.

    No callback: a callback must take the GIL on CoreAudio's thread, and any stall elsewhere is a dropout.
    The writer measures the device, because a device change leaves a stream that takes everything and plays
    nothing. The stream's `gain` ramps across a block, from 0 at a start and to 0 at a stop. With
    `tail_frames` it holds samples back, so a late producer fades out over them and back in after.
    """

    stream_name = "audio-writer"
    samplerate: int | None = 48000
    """None for the device's own rate."""
    block_frames = 2048
    latency = "low"
    tail_frames = 0
    held = False
    """Silenced without stopping the stream, for a subclass that reads it."""

    def _init_stream(self) -> None:
        self.error: str | None = None
        self.rate = self.samplerate
        self.gain = 1.0
        """The stream's own gain, ramped across the next block when it changes."""
        self._gain_now = 0.0            # where the last block left the gain: a start rises from silence
        self._tail = np.zeros(self.tail_frames)
        self._quiet = True              # the producer has nothing sounding: the next block fades in
        self._stream = None
        self._writer = None
        self._stop = threading.Event()
        self._swap = threading.Lock()   # `stop` and a writer rebuilding its stream must not cross
        self._last_write = 0.0
        self.dropouts = 0
        """Blocks the device ran out of data for. **Counted rather than ignored**, because a dropout and a
        speaker at its excursion limit both sound like crackle and the fix differs."""
        self.starved = 0
        self.late = 0
        """Blocks the producer had nothing ready for."""
        self.restarts = 0

    def _block(self, frames: int):
        raise NotImplementedError

    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def block_seconds(self) -> float:
        return self.block_frames / float(self.rate or 48000)

    def _open(self):
        """A started output stream, or None with `error` saying why."""
        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = f"sounddevice unavailable: {exc}"
            return None
        try:
            rate = self.samplerate or _device_rate()
            stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32",
                                     blocksize=self.block_frames, latency=self.latency)
            stream.start()
        except Exception as exc:
            self.error = f"no audio output: {exc}"
            return None
        self.rate = rate
        return stream

    def silenced(self) -> bool:
        """Outputs off, for this stream. A subclass can name its own module's switch."""
        return muted()

    def start(self) -> bool:
        if self.silenced():
            return False
        if self._stream is not None:
            if time.monotonic() - self._last_write < STALL_SECONDS:
                return True
            self.stop()                 # a writer stuck in a write that will never return
            self.restarts += 1
        stream = self._open()
        if stream is None:
            return False
        stop = threading.Event()
        writer = threading.Thread(target=self._run, args=(stream, stop), daemon=True, name=self.stream_name)
        with self._swap:
            # Started before it is published, both under the lock: a `stop` can only ever find a writer that
            # runs, and `shut_aside` cannot be handed one it is not allowed to join.
            self._last_write = time.monotonic()
            self.held = False
            self._gain_now = 0.0
            self._tail = np.zeros(self.tail_frames)
            self._quiet = True
            writer.start()
            self._stream, self._stop, self._writer = stream, stop, writer
        self.error = None
        return True

    def _shut(self, stream) -> None:
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    def _reopen(self, stream, stop: threading.Event, why: str):
        """A fresh stream in place of one that has stopped playing, or None once stopped or given up."""
        self._shut(stream)
        for _ in range(REOPEN_TRIES):
            if stop.wait(REOPEN_PAUSE_SECONDS):
                return None
            fresh = self._open()
            if fresh is None:
                continue
            with self._swap:
                current = not stop.is_set()     # stopped while it opened: then this stream is nobody's
                if current:
                    self._stream = fresh
                    self.restarts += 1
                    self.error = None
                    self._gain_now = 0.0
            if not current:
                self._shut(fresh)
                return None
            return fresh
        self.error = why
        return None

    def next_block(self, frames: int, closing: bool = False) -> np.ndarray:
        """The producer's next block, finished: held-back tail, fades, the stream's gain and the limit."""
        raw = self._block(frames)
        late = raw is None
        data = np.zeros(frames) if late else np.asarray(raw, dtype=float).reshape(-1)[:frames].copy()
        if len(data) < frames:
            data = np.concatenate([data, np.zeros(frames - len(data))])
        held = len(self._tail)
        if late:
            self.late += 1
            if not self._quiet and held:
                fade_out(self._tail, FADE_MS, self.rate or 48000)
            elif not self._quiet:
                self.dropouts += 1      # nothing held back to fade: the step could not be avoided
            self._quiet = True
        elif self._quiet:
            if held:
                fade_in(data, FADE_MS, self.rate or 48000)
            self._quiet = False
        if held:
            full = np.concatenate([self._tail, data])
            out, self._tail = full[:frames], full[frames:].copy()
            if late:
                self._tail[:] = 0.0
        else:
            out = data
        target = 0.0 if closing else self.gain
        out = out * ramp(self._gain_now, target, frames)
        self._gain_now = target
        return soft_limit(out).astype(np.float32).reshape(-1, 1)

    def _run(self, stream, stop: threading.Event) -> None:
        """Fill the buffer for as long as the stream lives, then write one block that fades to zero. `write`
        blocks when the buffer is full, so this paces itself and needs no clock of its own — and how far it
        runs ahead is the health of the device.

        The writer is handed its stream and its own stop flag and reads neither from the player, so a stop
        returns at once and a fresh start can open the next stream while this one winds down."""
        frames = self.block_frames
        written, began = 0, time.monotonic()
        while not stop.is_set():
            try:
                block = self.next_block(frames)   # inside the try: a render-time error must reopen, not die
                if stream.write(block):
                    self.dropouts += 1
            except Exception as exc:
                stream = self._reopen(stream, stop, f"audio write failed: {exc}")
                if stream is None:
                    return
                written, began = 0, time.monotonic()
                continue
            now = time.monotonic()
            self._last_write = now
            written += len(block)
            verdict = self._health(written, now - began, self.rate or 48000, frames)
            if verdict == "dead":
                stream = self._reopen(stream, stop, "the audio device stopped playing what it was given")
                if stream is None:
                    return
                written, began = 0, time.monotonic()
            elif verdict == "starved":
                self.starved += 1
        try:
            stream.write(self.next_block(frames, closing=True))
        except Exception:
            pass                        # a stream that will not take its last block is being shut anyway

    @staticmethod
    def _health(written: int, elapsed: float, rate: float = 48000, frames: int = 2048) -> str:
        """What the lead says about the device — audio handed over that it has not played yet.

        A healthy stream holds the lead at the buffer's depth: it drains in real time, so `write` blocks
        and the writer paces itself. A lead that keeps growing is a device consuming nothing, and a lead
        under one block is a buffer about to run dry, which is a crackle whether or not PortAudio says so.
        """
        lead = written / rate - elapsed
        if lead > STALL_SECONDS:
            return "dead"
        if written > rate and lead < frames / rate:
            return "starved"
        return ""

    def stop(self) -> None:
        """Fade the stream out and return at once; the writer writes its last block and the device winds
        down on a thread of their own (`shut_aside`). Waited for here, the two took 0.85 s on the screen's
        loop at the end of a Pupil effort, long enough for the trial engine to drop the result."""
        with self._swap:
            stream, writer = self._stream, self._writer
            self._stream = self._writer = None
            self._stop.set()            # a held player stays held through its last block: silent to the end
        shut_aside(stream, writer)


class Mixer:
    """Sums queued sounds into blocks, sample for sample, with silence between them.

    `add` can come from any thread and never blocks: it is a put on a queue. `block` runs on the writer's
    thread alone. A sound added starts at the first sample of the next block."""

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._voices: list[list] = []           # [float32 samples, gain, position]
        self._lock = threading.Lock()           # `block` against `clear`, never against `add`

    def add(self, samples: np.ndarray, gain: float = 1.0) -> None:
        self._queue.put((samples, gain))

    def clear(self) -> None:
        with self._lock:
            self._voices = []
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    @property
    def busy(self) -> bool:
        return bool(self._voices) or not self._queue.empty()

    def block(self, frames: int) -> np.ndarray:
        out = np.zeros(frames)
        with self._lock:
            while True:
                try:
                    samples, gain = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._voices.append([samples, gain, 0])
            keep = []
            for voice in self._voices:
                data, gain, pos = voice
                n = min(frames, len(data) - pos)
                if n > 0:
                    out[:n] += data[pos:pos + n] * gain
                    voice[2] = pos + n
                if voice[2] < len(data):
                    keep.append(voice)
            self._voices = keep
        return out


class CueStream(Stream):
    """One stream for every short sound, open from the first need until the console exits.

    At the device's own rate, with a small block: a cue is one short sound at a moment already decided, and
    waits at most one block plus the buffer. Its producer is a `Mixer`, which always has a block, so it needs
    no held-back tail and its latency carries none. It opens on a thread of its own, because the caller is
    the screen's thread or the feed thread, where a call into CoreAudio that blocks holds up the program.
    """

    stream_name = "cue-writer"
    samplerate = None
    block_frames = 512
    latency = "low"
    tail_frames = 0

    def __init__(self, opener: Callable[[], tuple] | None = None) -> None:
        self._opener = opener
        self.mixer = Mixer()
        self._init_stream()
        self._opening = False
        self._tried = False

    def _block(self, frames: int) -> np.ndarray:
        return self.mixer.block(frames)

    def _open(self):
        if self._opener is None:
            return super()._open()
        try:
            stream, rate = self._opener()
        except Exception as exc:
            self.error = f"no audio output: {exc}"
            return None
        self.rate = rate
        return stream

    def ready(self) -> bool:
        """Whether a sound added now is heard: an opening that has finished, and a writer that has written
        within `STALL_SECONDS`."""
        return (not self._opening and self._stream is not None
                and time.monotonic() - self._last_write < STALL_SECONDS)

    def ensure(self) -> bool:
        """Whether the stream can play now. When it cannot yet, it opens on a thread of its own and this says
        no, so the caller plays that one sound another way."""
        if self.ready():
            return True
        if self._stream is not None:
            self.stop()                 # went silent (device changed): shut aside, and reopen
            self.mixer.clear()          # a dead stream never played them; do not carry them over
            self._tried = False
        if self._opening or self._tried:    # one failure per process, not one per sound
            return False
        self._opening = self._tried = True

        def open_() -> None:
            try:
                Stream.start(self)
            finally:
                self._opening = False

        threading.Thread(target=open_, daemon=True, name="cue-stream-open").start()
        return False

    def release(self) -> None:
        """For the exit: the writer stops after its current block. Nothing here waits for it."""
        self._stop.set()


_families: list[Callable[[], Iterable[tuple[str, Callable[[int], np.ndarray], float | None]]]] = []


def family(enumerate_: Callable) -> Callable:
    """Register a module's sounds, as `(key, render, highest_hz)` across their whole domain, so that one test
    can hold every sound the program renders to the rules. `highest_hz` is None for broadband content."""
    _families.append(enumerate_)
    return enumerate_


def every_sound() -> list[tuple[str, Callable[[int], np.ndarray], float | None]]:
    return [entry for enumerate_ in _families for entry in enumerate_()]


_cues = CueStream()


def glitches(stream=None) -> str:
    """What a crackle looks like from inside a stream, as text to append to a status line, or "" when it
    has none. The cue stream when no stream is given. Silent at zero: "0 dropouts" on every screen is noise."""
    stream = _cues if stream is None else stream
    said = []
    for n, what in ((getattr(stream, "dropouts", 0), "dropout"),
                    (getattr(stream, "starved", 0), "starved block"),
                    (getattr(stream, "restarts", 0), "restart")):
        if n:
            said.append(f"{n} {what}{'s' if n > 1 else ''}")
    return f" · {', '.join(said)}" if said else ""
_sounds: dict[tuple[str, int], np.ndarray] = {}
_files: dict[str, Path] = {}
_file_dir: Path | None = None


def warm() -> None:
    """Open the cue stream ahead of the first sound, so that one too goes through it. Returns at once."""
    if not muted():
        _cues.ensure()


def release() -> None:
    _cues.release()


def sound(key: str, render: Callable[[int], np.ndarray], rate: int) -> np.ndarray:
    """A finished short sound at `rate`, rendered once per key and rate."""
    if (key, rate) not in _sounds:
        _sounds[(key, rate)] = np.asarray(render(rate), dtype=np.float32)
    return _sounds[(key, rate)]


def file_for(key: str, render: Callable[[int], np.ndarray]) -> Path:
    """The sound as a WAV file at `WAV_RATE`, written once per key, for `afplay`."""
    global _file_dir
    if key not in _files:
        if _file_dir is None:
            _file_dir = Path(tempfile.mkdtemp(prefix="nfb-sounds-"))
        path = _file_dir / f"{''.join(c if c.isalnum() else '_' for c in key)}.wav"
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(WAV_RATE)
            fh.writeframes(counts(sound(key, render, WAV_RATE)).tobytes())
        _files[key] = path
    return _files[key]


def afplay_available() -> bool:
    return sys.platform == "darwin" and shutil.which("afplay") is not None


def afplay_argv(path: Path, volume: float | None) -> list[str]:
    """The fallback's command line. `afplay -v` takes 0…1; with no volume it plays at full level."""
    argv = ["afplay"]
    if volume is not None:
        argv += ["-v", f"{min(100.0, max(0.0, volume)) / 100:.3f}"]
    return argv + [str(path)]


def _afplay(path: Path, volume: float | None) -> bool:
    global _last_error
    if muted() or not afplay_available():
        return False
    try:
        subprocess.Popen(afplay_argv(path, volume), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        return False


def play(key: str, render: Callable[[int], np.ndarray], volume: float | None = None,
         file: Path | None = None) -> bool:
    """Fire and forget one short sound. Never raises and never blocks the screen or the feed.

    `render(rate)` returns the finished sound at `rate`, from `finish`; it runs once per key and rate.
    Through the cue stream when it is open; before that, and when it cannot open, through `afplay` of
    `file`, or of a file written from `render`. `volume` is 0–100, or None for full level, and holds for
    the whole sound, so it never steps inside one.
    """
    if muted():
        return False
    if _cues.ensure():
        gain = 1.0 if volume is None else min(100.0, max(0.0, volume)) / 100.0
        _cues.mixer.add(sound(key, render, _cues.rate), gain)
        return True
    return _afplay(file if file is not None else file_for(key, render), volume)


def last_error() -> str | None:
    return _last_error
