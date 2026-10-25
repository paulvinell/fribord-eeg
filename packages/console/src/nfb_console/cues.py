"""Short sounds: the session's reminder beep and finishing tone, and one click of a click train.

Deliberately not the feedback engine. Continuous feedback has to modulate a tone smoothly at low
latency and belongs in a real audio path; a cue is one short sound at a moment already decided, so
`afplay` in a detached process is the whole implementation. A reminder half a second late is still a
reminder.

The click is the same kind of thing: one short sound at a moment already decided — every target met —
so it plays through here too. What it is *not* is a train whose rate carries the signal; see
`feedback.Click`. It is also the one sound here that is **generated**: macOS ships nothing short enough
to read as a click, so `click_path` renders a few milliseconds of band-limited noise to a WAV instead.

Volume is per-play, because a click and a session-end tone want different loudness and the click's is a
setting the operator adjusts. `afplay -v` takes 0…1, so a 0–100 setting is divided here rather than at
every call site.

macOS only, matching the rest of this project. `problem()` reports why a sound would not be heard,
so a pane can say so rather than presenting a silent control as working.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

SYSTEM = Path("/System/Library/Sounds")
RATE = 44100

# The session's own two cues are system sounds, which is fine: they mark a moment and can afford to be
# a few hundred milliseconds long.
SOUNDS = {
    "beep": SYSTEM / "Ping.aiff",       # the reminder: clear, not startling
    "finish": SYSTEM / "Glass.aiff",    # clearly an ending
}
LABELS = {"beep": "reminder beep", "finish": "finishing tone"}

# The reward click is **synthesised**, because macOS does not ship one. Every system sound is either
# too long or has the wrong character: `Tink` is the error chirp, `Purr` rings for a quarter of a
# second, and `Pop` reads as a hollow knock. A click has to be a few milliseconds of noise with a hard
# attack and an immediate decay — that is what makes it read as an event rather than a tone — so it is
# cheaper to make one than to hunt for one.
#
# (centre-band low, high, decay in ms). The decay is the whole character: 2 ms is a Geiger tick, 9 ms
# has body. All of them finish inside 40 ms, which they must — a reward every couple of seconds cannot
# still be sounding when the next is due.
CLICK_PROFILES = {
    "tick": (2500.0, 7000.0, 1.8),
    "clack": (900.0, 3000.0, 4.0),
    "thock": (250.0, 1400.0, 8.0),
}
DEFAULT_CLICK = "tick"

# Profiles derived from a frequency rather than chosen by name — see `frequency_click`. Kept apart from
# CLICK_PROFILES on purpose: that dict is the *menu*, and generating into it would have put "alert@0.58"
# in the list of click characters the Click pane offers. Caught by a test that pins the menu's contents.
_DERIVED: dict[str, tuple[float, float, float]] = {}

_TONES: dict[str, tuple[float, float]] = {}
"""Rendered reward tones, `name -> (pitch, ms)`. Apart from `CLICK_PROFILES` and `_DERIVED` for the same
reason those two are apart: that dict is the *menu* of click characters, and a tone is not one."""

HOLD_GAP_MS = 90.0
"""Silence between the two marks of a *held* reward. Long enough that they are heard as two events rather
than as one longer sound, short enough that the pair still reads as one message."""

_DOUBLES: dict[str, str] = {}
"""Doubled forms, `doubled name -> base name`. **The distinction between arriving and having held has to be
carried by rhythm, not by pitch**, because on a keyed screen the pitch is already saying *which band* earned
it and cannot carry two things at once. One mark means you got there, two means you have held it — learnable
in a session, needing no calibration, and impossible to confuse with the alert, which is noise rather than a
pitch.

Its own table for the same reason `_DERIVED` has one: `CLICK_PROFILES` is the *menu* the Click pane lists, and
generating into it would put "tick+held" in the list of characters to choose from."""

_cache: dict[str, Path] = {}
_cache_dir: Path | None = None

_last_error: str | None = None


def muted() -> bool:
    """`NFB_NO_OUTPUTS` silences audio for the process. See `shadow.muted` for why it exists."""
    return bool(os.environ.get("NFB_NO_OUTPUTS"))


def available() -> bool:
    return (not muted() and sys.platform == "darwin" and shutil.which("afplay") is not None
            and all(p.exists() for p in SOUNDS.values()))


def click_choices() -> list[str]:
    """The click characters on offer, for a pane to list. All generated, so all always available."""
    return list(CLICK_PROFILES)


def _profile(name: str) -> tuple[float, float, float] | None:
    return CLICK_PROFILES.get(name) or _DERIVED.get(name)


def held(name: str) -> str:
    """The doubled form of a click or tone — two marks instead of one.

    Idempotent, and it returns `name` unchanged for anything it does not recognise, so a caller can hand it
    whatever the click's sound happens to be without first asking what kind it is.
    """
    if name in _DOUBLES or (_profile(name) is None and name not in _TONES):
        return name
    doubled = name + "+held"
    _DOUBLES[doubled] = name
    return doubled


RELEASE_MS = 8.0
"""Every rendered sound is faded to **exactly zero** over its last few milliseconds.

**This is a crackle fix, and it is the third family of them in this codebase.** The other two are in
`music`: a stream starved of blocks, and a laptop cone driven past its excursion limit by low frequencies.
This one is neither — it is a step in the waveform. The pacer tone's envelope is an exponential over
160 ms with a time constant of 160/2.5, so it was still at **8% of full scale when the buffer ended** and
the speaker was asked to go from there to silence in one sample. A discontinuity is broadband: what comes
out is a click on the end of every breath cue, and on a small speaker it reads as crackle.

Eight milliseconds is inaudible as a fade and long enough to be gentle at any pitch here — half a cycle
of the lowest tone this renders is 4.5 ms, so the ramp always spans a full cycle or more."""


def _fade_out(wave_data: "np.ndarray", ms: float = RELEASE_MS) -> "np.ndarray":
    """Raised-cosine fade over the last `ms`, ending at exactly zero. See RELEASE_MS."""
    import numpy as np

    n = min(len(wave_data), max(1, int(RATE * ms / 1000.0)))
    ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, n + 1) / n))
    wave_data[-n:] *= ramp
    return wave_data


def _render(low: float, high: float, decay_ms: float) -> "np.ndarray":
    """A band-limited noise burst under an exponential decay: hard attack, immediate fall.

    Noise rather than a tone, because a damped sine at these lengths is heard as a pitched blip while
    filtered noise is heard as a *click*. Seeded, so the same profile always renders the same file.
    """
    import numpy as np
    from scipy import signal

    tau = decay_ms / 1000.0
    samples = int(RATE * tau * 5)                    # five time constants: silent by the end
    times = np.arange(samples) / RATE
    noise = np.random.default_rng(0).standard_normal(samples)
    sos = signal.butter(2, [low, high], btype="bandpass", fs=RATE, output="sos")
    wave_data = signal.sosfilt(sos, noise) * np.exp(-times / tau)
    peak = float(np.abs(wave_data).max()) or 1.0
    return (_fade_out(wave_data / peak * 0.9) * 32767).astype(np.int16)


def click_samples(name: str) -> "np.ndarray | None":
    """The rendered int16 samples of a click or tone, doubled form included. None for unknown names.

    Split out of `click_path` so the persistent stream below can play the samples directly — the whole
    point of that stream is not touching the filesystem or a child process at the moment of a reward.
    """
    base = _DOUBLES.get(name, name)
    profile = _profile(base)
    tone = _TONES.get(base)
    if profile is None and tone is None:
        return None
    data = _render_tone(*tone) if tone is not None else _render(*profile)
    if base != name:
        import numpy as np

        gap = np.zeros(int(RATE * HOLD_GAP_MS / 1000.0), dtype=data.dtype)
        data = np.concatenate([data, gap, data])
    return data


def click_path(name: str) -> Path | None:
    """Render `name` to a WAV once per process and return its path.

    Clicks and reward tones both come through here — they are the same kind of thing to a caller, a short
    synthesised sound identified by name — while staying different kinds of thing to the ear and in the
    tables above.
    """
    global _cache_dir
    if name in _cache:
        return _cache[name]
    data = click_samples(name)
    if data is None:
        return None
    if _cache_dir is None:
        _cache_dir = Path(tempfile.mkdtemp(prefix="nfb-clicks-"))
    path = _cache_dir / f"{name.replace('@', '-').replace('.', '_').replace('+', '_')}.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes(data.tobytes())
    _cache[name] = path
    return path


def duration_ms(name: str) -> float:
    """How long a click or a tone lasts, for a pane that wants to say so. A doubled one is both marks and
    the gap, since that is what the ear is given."""
    base = _DOUBLES.get(name, name)
    tone = _TONES.get(base)
    one = tone[1] if tone is not None else (_profile(base)[2] * 5 if _profile(base) else 0.0)
    return one * 2 + HOLD_GAP_MS if base != name else one


# --- one timbre per frequency ---------------------------------------------------------------

# EEG 1-45 Hz mapped onto audible centre frequencies, logarithmically in both, so equal musical steps
# correspond to equal ratios of EEG frequency. This is the audible twin of `theme.frequency_colour`: with
# twelve inhibit bins, "something is over threshold" is not actionable and "*that* one is" is, so the
# alert has to say which — and it has to do so without twelve separate settings to configure and remember.
#
# Deriving it means a bin added later already sounds like where it sits, with no per-bin setting.
ALERT_LO_HZ, ALERT_HI_HZ = 1.0, 45.0
ALERT_TONE_LO, ALERT_TONE_HI = 180.0, 5000.0
ALERT_DECAY_LO, ALERT_DECAY_HI = 9.0, 2.5      # ms: slow activity thocks, fast activity ticks


def frequency_click(hz: float) -> str:
    """Name of a click whose character encodes `hz`, rendering it on first use.

    Low EEG frequencies come out as a low, slow thock and high ones as a high, quick tick, so a run of
    alerts is legible as *where* in the spectrum the trouble is without reading anything.
    """
    import numpy as np

    hz = min(ALERT_HI_HZ, max(ALERT_LO_HZ, float(hz)))
    # rounded to a third of an octave: enough steps to tell twelve bins apart, few enough that the
    # rendered files stay a handful rather than one per distinct threshold crossing
    span = np.log2(ALERT_HI_HZ / ALERT_LO_HZ)
    step = round(np.log2(hz / ALERT_LO_HZ) / span * 12) / 12
    name = f"alert@{step:.2f}"
    if name not in _DERIVED:
        centre = ALERT_TONE_LO * (ALERT_TONE_HI / ALERT_TONE_LO) ** step
        decay = ALERT_DECAY_LO + (ALERT_DECAY_HI - ALERT_DECAY_LO) * step
        _DERIVED[name] = (centre * 0.7, min(centre * 1.7, RATE / 2 * 0.9), decay)
    return name


def profile_of(name: str) -> tuple[float, float, float] | None:
    """The band and decay behind a click, named or derived. For tests and for a pane that says so."""
    return _profile(name)


# --- reward tones: a pitch per band, not a click ---------------------------------------------

# **A reward that says which of two alternatives earned it has to be a tone, and this is not a
# preference.** The two reward pitches are **137 Hz and 738 Hz**, two and a half
# octaves apart. `frequency_click` cannot express that: it renders a few milliseconds of band-limited
# noise, which is heard as a *click* rather than as a pitch by design, and across theta and alpha its
# centres come out 719 Hz and 1251 Hz — under an octave, on two sounds whose whole job is to be told
# apart while the trainee's eyes are closed.
#
# So the anchors below are the two pitches, mapped to the centres of the theta (2-7 Hz) and alpha
# (8-11 Hz) bands, and the interpolation between and beyond them is ours: a straight line in log-log, so
# a third reward between the two lands between the two pitches. Clamped, because the mapping is steep and
# a reward at 30 Hz would otherwise shriek.
TONE_ANCHORS = ((4.5, 137.0), (9.5, 738.0))
TONE_LO_HZ, TONE_HI_HZ = 110.0, 1500.0
TONE_MS = 160.0
"""Long enough to be heard as a pitch — 137 Hz needs about 50 ms for the ear to find a note — and short
enough to finish well inside a reward interval. A click is 2-8 ms and deliberately too short for this."""
TONE_ATTACK_MS = 12.0
"""A raised-cosine attack rather than the click's hard edge. A step onto a sine is heard as a click
followed by a tone, which would put the two mechanisms' characters back on top of each other."""

def tone_pitch(hz: float) -> float:
    """The audible pitch a reward at `hz` sounds at."""
    (f1, p1), (f2, p2) = TONE_ANCHORS
    import numpy as np

    hz = max(0.1, float(hz))
    slope = np.log(p2 / p1) / np.log(f2 / f1)
    return float(min(TONE_HI_HZ, max(TONE_LO_HZ, p1 * (hz / f1) ** slope)))


def target_pitch(tgt) -> float:
    """The pitch a keyed reward sounds at: its own `tone_hz` if set, otherwise derived from its band.

    One place, so the audition button and the sound the session plays cannot disagree — which is exactly how
    the audition row and the reward tone drifted apart once before, and a pitch you can set by hand is a
    second chance to make that mistake."""
    explicit = getattr(tgt, "tone_hz", None)
    if explicit:
        return float(min(TONE_HI_HZ, max(TONE_LO_HZ, explicit)))
    band = getattr(tgt, "band", None)
    return tone_pitch(band.centre if band is not None else 10.0)


def reward_tone(hz: float) -> str:
    """Name of a tone whose pitch stands for the band at `hz`, rendering it on first use."""
    pitch = round(tone_pitch(hz))
    name = f"tone@{pitch:d}"
    if name not in _TONES:
        _TONES[name] = (float(pitch), TONE_MS)
    return name


def tone_at(pitch_hz: float) -> str:
    """Name of a tone at an audible pitch given directly, rather than derived from a band."""
    pitch = round(min(TONE_HI_HZ, max(TONE_LO_HZ, float(pitch_hz))))
    name = f"tone@{pitch:d}"
    if name not in _TONES:
        _TONES[name] = (float(pitch), TONE_MS)
    return name


def _render_tone(pitch: float, ms: float) -> "np.ndarray":
    """A sine under a raised-cosine attack and an exponential decay."""
    import numpy as np

    samples = int(RATE * ms / 1000.0)
    times = np.arange(samples) / RATE
    tone = np.sin(2 * np.pi * pitch * times)
    attack = max(1, int(RATE * TONE_ATTACK_MS / 1000.0))
    envelope = np.exp(-times / (ms / 1000.0 / 2.5))
    envelope[:attack] *= 0.5 * (1 - np.cos(np.pi * np.arange(attack) / attack))
    return (_fade_out(tone * envelope * 0.9) * 32767).astype(np.int16)


def problem() -> str | None:
    """Why a sound would not be heard, if it would not be."""
    if sys.platform != "darwin":
        return f"no sound player on {sys.platform}"
    if shutil.which("afplay") is None:
        return "afplay not found"
    missing = [p.name for p in SOUNDS.values() if not p.exists()]
    if missing:
        return f"missing system sound {missing[0]}"
    return _last_error


def command(name: str, volume: float | None = None) -> list[str] | None:
    """The argv `play` would run, or None if there is nothing to play. Separated so the volume
    arithmetic is testable without making a noise."""
    path = SOUNDS.get(name) or click_path(name)
    if path is None:
        return None
    argv = ["afplay"]
    if volume is not None:
        argv += ["-v", f"{min(100.0, max(0.0, volume)) / 100:.3f}"]
    return argv + [str(path)]


def play_file(path: "Path | str | None", volume: float | None = None) -> bool:
    """Fire and forget an already-rendered file. For auditions of things this module does not own —
    the noise's tilts, which `music.preview_path` renders. Same guards as `play`."""
    global _last_error
    if path is None or muted() or not available():
        return False
    argv = ["afplay"]
    if volume is not None:
        argv += ["-v", f"{min(100.0, max(0.0, volume)) / 100:.3f}"]
    try:
        subprocess.Popen(argv + [str(path)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except OSError as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        return False


class _ClickStream:
    """A pre-opened output stream that mixes already-rendered buffers on demand.

    Why this exists, measured: a fresh `afplay` returns from `Popen` in ~3 ms, but the audio hardware does
    not go active until **74–117 ms** later — CoreAudio session setup, paid again on every click. The click
    is the most moment-locked sound in the app; it is the one mechanism near the 250 ms budget, and a third of that budget was being spent launching a process to play 9 ms of noise.
    Keeping one stream open makes a click cost the stream's output latency and nothing else — **71 ms
    measured** (`make latency`), CoreAudio's floor for a 'low'-latency stream rather than anything the
    block size buys back — and `subprocess.Popen` leaves the feed thread, where its 2.7 ms was a stall
    like any other. Against afplay that is 71 ms flat instead of 74–117 ms of session setup plus the
    child's own buffering, and constant instead of per-click.

    Small block and `latency='low'` on purpose — the opposite trade from the music streams' deep buffer. A
    dropout in a 9 ms click is inaudible where a dropout in continuous music is a crackle, so this stream
    buys immediacy where the music streams buy slack. Voices are mixed additively: a reward click landing on a
    reminder beep's tail must not silence either.
    """

    def __init__(self) -> None:
        self._stream = None
        self._lock = threading.Lock()
        self._voices: list[list] = []           # [float32 samples, gain, position]
        self._tried = False
        self.error: str | None = None

    def start(self) -> bool:
        if self._stream is not None:
            return True
        if self._tried:                          # one failure per process, not one per click
            return False
        self._tried = True
        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = f"sounddevice unavailable: {exc}"
            return False
        try:
            self._stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="float32",
                                           blocksize=256, latency="low",
                                           callback=self._callback)
            self._stream.start()
        except Exception as exc:
            self.error = f"no audio output: {exc}"
            self._stream = None
            return False
        return True

    def trigger(self, name: str, volume: float | None) -> bool:
        if not self.start():
            return False
        import numpy as np

        data = click_samples(name)
        if data is None:
            return False
        gain = 1.0 if volume is None else min(100.0, max(0.0, volume)) / 100.0
        with self._lock:
            self._voices.append([data.astype(np.float32) / 32767.0, gain, 0])
        return True

    def _callback(self, outdata, frames, time_info, status) -> None:
        import numpy as np

        outdata[:] = 0.0
        with self._lock:
            keep = []
            for voice in self._voices:
                data, gain, pos = voice
                n = min(frames, len(data) - pos)
                if n > 0:
                    outdata[:n, 0] += data[pos:pos + n] * gain
                    voice[2] = pos + n
                if voice[2] < len(data):
                    keep.append(voice)
            self._voices = keep
        # **Mixing is addition, and addition overflows.** Each voice is rendered to 0.9 of full scale, so
        # a reward click landing on a breath cue asks for 1.8 — and what a device does past 1.0 is clip
        # the samples flat, which is a square wave, which is crackle. Additive mixing is still right (a
        # click on a cue's tail must not silence either), so the sum is limited here instead: transparent
        # while nothing overlaps, and a soft ceiling rather than a broken waveform when something does.
        np.clip(outdata, -1.0, 1.0, out=outdata)


_stream = _ClickStream()


def play(name: str, volume: float | None = None) -> bool:
    if muted():
        return False
    """Fire and forget, one sound. Never raises and never blocks the draw loop.

    `volume` is 0–100, or None for the system volume.

    Synthesised sounds — the click and the reward tones — go through the persistent stream when it can
    open, because they are moment-locked; see `_ClickStream`. The two system cues, and everything when
    the stream cannot open, still take `afplay`: a reminder half a second late is still a reminder.
    """
    global _last_error
    if name not in SOUNDS and available() and _stream.trigger(name, volume):
        return True
    argv = command(name, volume)
    if argv is None or not available():
        return False
    try:
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        return False
