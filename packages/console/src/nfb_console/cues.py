"""Short sounds: the session's reminder beep and finishing tone, the clicks, and the reward tones.

What each sound is made of lives here; how it reaches the speaker does not. Every sound is finished by
`audio.finish` and played by `audio.play`, which hold the rules against crackle for every sound at once.

The click is one short sound at a moment already decided — every target met. What it is *not* is a train
whose rate carries the signal; see `feedback.Click`. It is generated: macOS ships nothing short enough to
read as a click, so it is a few milliseconds of band-limited noise.

Volume is per-play, because a click and a session-end tone want different loudness and the click's is a
setting the operator adjusts.

macOS only, matching the rest of this project. `problem()` reports why a sound would not be heard,
so a pane can say so rather than presenting a silent control as working.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import audio, developer

SYSTEM = Path("/System/Library/Sounds")
RATE = audio.WAV_RATE

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

def muted() -> bool:
    """Outputs off silences audio for the process. See `developer.outputs_off` for why it exists."""
    return developer.outputs_off()


def available() -> bool:
    return not muted() and audio.afplay_available() and all(p.exists() for p in SOUNDS.values())


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


def _click_shape(low: float, high: float, decay_ms: float, rate: int) -> np.ndarray:
    """A band-limited noise burst under an exponential decay: hard attack, immediate fall.

    Noise rather than a tone, because a damped sine at these lengths is heard as a pitched blip while
    filtered noise is heard as a *click*. Seeded, so the same profile always renders the same sound.
    """
    from scipy import signal

    tau = decay_ms / 1000.0
    samples = int(rate * tau * 5)                    # five time constants: silent by the end
    times = np.arange(samples) / rate
    noise = np.random.default_rng(0).standard_normal(samples)
    sos = signal.butter(2, [low, min(high, rate / 2 * 0.9)], btype="bandpass", fs=rate, output="sos")
    return signal.sosfilt(sos, noise) * np.exp(-times / tau)


def _tone_shape(pitch: float, ms: float, rate: int) -> np.ndarray:
    """A sine under a raised-cosine attack and an exponential decay."""
    samples = int(rate * ms / 1000.0)
    times = np.arange(samples) / rate
    tone = np.sin(2 * np.pi * pitch * times)
    attack = max(1, int(rate * TONE_ATTACK_MS / 1000.0))
    envelope = np.exp(-times / (ms / 1000.0 / 2.5))
    envelope[:attack] *= 0.5 * (1 - np.cos(np.pi * np.arange(attack) / attack))
    return tone * envelope


def _read_aiff(path: Path) -> tuple[np.ndarray, float]:
    """The samples of an uncompressed AIFF as mono floats in -1..1, and its rate. Enough for the two
    system sounds, which are 24-bit stereo PCM: the standard library's reader leaves Python in 3.13."""
    data = path.read_bytes()
    pos, channels, bits, rate, pcm = 12, 1, 16, float(RATE), b""
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"COMM":
            channels, _frames, bits = struct.unpack(">hIh", body[:8])
            exponent, mantissa = struct.unpack(">HQ", body[8:18])
            rate = mantissa * 2.0 ** ((exponent & 0x7FFF) - 16383 - 63)
        elif cid == b"SSND":
            pcm = body[8 + struct.unpack(">I", body[:4])[0]:]
        pos += 8 + size + (size & 1)
    width = bits // 8
    raw = np.frombuffer(pcm[:len(pcm) // width * width], dtype=np.uint8).reshape(-1, width)
    value = np.zeros(len(raw), dtype=np.int64)
    for byte in range(width):
        value = (value << 8) | raw[:, byte]
    value = np.where(value >= 1 << (bits - 1), value - (1 << bits), value)
    frames = value[:len(value) // channels * channels].reshape(-1, channels)
    return frames.mean(axis=1) / float(1 << (bits - 1)), rate


def sound_samples(name: str, rate: int = RATE) -> np.ndarray | None:
    """Any sound here, finished, as floats at `rate`. None for a name that is not a sound.

    A system sound keeps its own level and is resampled to `rate`; a click and a tone are made at `rate`.
    Every one goes through `audio.finish`, and a doubled one is two finished marks with silence between."""
    if name in SOUNDS:
        data, source = _read_aiff(SOUNDS[name])
        return audio.finish(audio.resample(data, source, rate), rate, peak=None)
    base = _DOUBLES.get(name, name)
    profile, tone = _profile(base), _TONES.get(base)
    if profile is None and tone is None:
        return None
    one = audio.finish(_tone_shape(*tone, rate) if tone is not None else _click_shape(*profile, rate), rate)
    if base == name:
        return one
    return np.concatenate([one, np.zeros(int(rate * HOLD_GAP_MS / 1000.0)), one])


def click_samples(name: str, rate: int = RATE) -> np.ndarray | None:
    """A sound as 16-bit counts, for a file. None for a name that is not a sound."""
    data = sound_samples(name, rate)
    return None if data is None else audio.counts(data)


def click_path(name: str) -> Path | None:
    """The sound as a WAV file, written once per process."""
    if sound_samples(name, RATE) is None:
        return None
    return audio.file_for(name, lambda rate: sound_samples(name, rate))


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


def _highest_hz(name: str) -> float | None:
    """The highest frequency a sound is meant to hold: a tone's pitch, or an octave over a click's band,
    where its second-order band-pass is 12 dB down. None for a system sound."""
    base = _DOUBLES.get(name, name)
    if base in _TONES:
        return _TONES[base][0]
    profile = _profile(base)
    return 2.0 * profile[1] if profile is not None else None


@audio.family
def every_sound():
    """Every sound here across its domain: the system sounds, the click menu, each alert step, tones across
    the pitch range at a sixth of an octave and at both ends, and the doubled form of each."""
    names = list(SOUNDS) + list(CLICK_PROFILES)
    names += [frequency_click(ALERT_LO_HZ * (ALERT_HI_HZ / ALERT_LO_HZ) ** (k / 12)) for k in range(13)]
    sixths = int(np.log2(TONE_HI_HZ / TONE_LO_HZ) * 6)
    names += [tone_at(TONE_LO_HZ * 2 ** (k / 6)) for k in range(sixths + 1)] + [tone_at(TONE_HI_HZ)]
    names += [held(n) for n in names if n not in SOUNDS]
    return [(n, (lambda rate, n=n: sound_samples(n, rate)), _highest_hz(n)) for n in dict.fromkeys(names)]


def problem() -> str | None:
    """Why a sound would not be heard, if it would not be."""
    if not audio.afplay_available():
        return "no sound player on this machine"
    missing = [p.name for p in SOUNDS.values() if not p.exists()]
    if missing:
        return f"missing system sound {missing[0]}"
    return audio.last_error()


def play(name: str, volume: float | None = None) -> bool:
    """Fire and forget one sound, through `audio.play`. Never raises and never blocks the screen or the feed.

    `volume` is 0–100, or None for full level."""
    if muted() or not available():
        return False
    if name not in SOUNDS and _profile(_DOUBLES.get(name, name)) is None \
            and _DOUBLES.get(name, name) not in _TONES:
        return False
    return audio.play(name, lambda rate: sound_samples(name, rate), volume, file=SOUNDS.get(name))
