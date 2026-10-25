"""Music: noise, a track, a folder, or the EEG itself.

Four sources. The last is the interesting one technically; the first is the one that matters for CFC.

**Noise at a spectral tilt**, pink by default, generated rather than played. Pink noise is the traditional
feedback music for this kind of training, and in CFC the reason is not habit: 1/f noise is *literally the
thing being trained*. A healthy signal "appears random but is actually what is called pink
noise, or 1/f noise." So the music is a
rendering of the target state, and under Quiet Reward it recedes as the trainee approaches it.

The choice to make pink noise the CFC default is ours.

**A file or a folder** plays through `afplay`, one track at a time on a worker thread, advancing when a
track ends and looping when the folder runs out. `afplay -v` is a *software* gain applied to that
playback only, which is what makes this volume independent of the system volume the Sound mechanism
drives — you want to set the backing track once and let the feedback move the master.

**The EEG as music** is a live stream through `sounddevice`, and the method is worth stating because the
two obvious ones do not work. *Audification* — replaying the samples faster until they are audible —
cannot run live: playing 250 samples per second of signal at 48 kHz consumes data 192× faster than it
arrives. Interpolating up to 48 kHz instead keeps the frequencies where they were, and 2–38 Hz is
inaudible.

**The third way was a heterodyne, and it is what this instrument used to be**: a fixed carrier whose
amplitude was the band-passed signal, so the band landed as sidebands at `carrier ± 2…38 Hz`. Timing was
exact and the render was cheap, and it is recorded here because the failure was structural rather than a
matter of tuning. **A heterodyne can only add a fixed offset, and pitch is a ratio.** 2–38 Hz against a
220 Hz carrier is 182–258 Hz — the tone never moves anywhere the ear reads as *height*, so alpha and beta
were the same note with different roughness, and 10 → 20 Hz, an octave in EEG terms, arrived as about a
semitone. Nothing on the panel could change that. Reported exactly that way: the signal did not feel alive,
and turning the sensitivity up only made it louder, because loudness was the only dimension it had.

**What it does now: track the band and resynthesise it.** A complex FIR (`nfb.dsp.envelope`, the cFIR the
low-latency envelope work is built on) turns the band into its analytic signal, whose modulus is the
band's amplitude and whose phase advance per sample is its instantaneous frequency. One oscillator then
plays that frequency, mapped **by ratio** onto a pitch range the operator chooses, at a loudness that
follows the amplitude. A fixed interval in EEG is a fixed interval in what you hear: with the band 8–16 Hz
sounding across 220–440 Hz the mapping is an exact transposition, ×27.5, and any other pair of ends
stretches or compresses that interval by a stated amount — `eeg_pitch` and `stretch_factor` below.

**What that costs, stated because it is not free.** Timing is still exact — no time scaling, no window —
but the *waveform* is gone: what you hear is one voice at the band's dominant frequency, not the shape of
the trace. A wide band with two rhythms in it does not arrive as two notes; the frequency estimate is one
amplitude-weighted average and it wobbles between them, which is why this instrument wants a band chosen
around the rhythm being trained rather than the full 2–38. Two voices would need a filter bank with one
oscillator per bin, which is a different instrument and is not built. The cFIR also has a design delay,
`EEG_TRACK_DELAY_SECONDS` below, which is added to the audio path.

The default band is 2–38 Hz: below 2 is drift and above 38 is mains and muscle.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
# NOT `from scipy import signal`: this module already imports the stdlib `signal` for SIGSTOP, and the
# shadow was silent — `hold()` would have raised AttributeError at the moment a target failed.
from scipy.signal import lfilter

from nfb.dsp import Band, CFIREnvelope, bandpass

AUDIO_SUFFIXES = (".mp3", ".m4a", ".aac", ".wav", ".aiff", ".aif", ".flac", ".ogg")
SAMPLE_RATE = 48000
BLOCK_FRAMES = 2048        # 43 ms per block; see NoisePlayer.start
BUFFER_LATENCY = "low"
"""How far ahead the music is written — PortAudio's suggested output latency. **This is the crackle fix that
a bigger block was not, and its value is the latency fix that the crackle fix overpaid for.**

The generator is Python, so it needs the GIL — and this process also runs a draw loop that rebuilds canvases
and a feed thread that filters EEG. With a callback, *any* stall elsewhere longer than one block is a
dropout, and a 43 ms block only moved the threshold. Writing ahead into a deep buffer from a thread of its
own instead means the device is never waiting on Python: a couple of hundred milliseconds of slack absorbs
a garbage collection, a heavy frame and a scheduling hiccup together.

This was `0.25`, meant as a quarter second of slack — but the number is only a *suggestion* to PortAudio,
and CoreAudio inflates it: measured with this exact stream configuration, asking for 0.25 yields **0.785 s**
of actual output latency, asking for 0.1 yields 0.444, and asking for `'low'` yields **0.231 s at the same
2048-frame block size** — the slack the 0.25 was aiming for, for free. Every continuous sound here reads
`level`, the fast feedback signal, so the difference is not cosmetic: 785 ms sits past the latency at which
trained effects disappear entirely, while ~230 ms of buffer plus the glide is at least in
the argued-about range for slow protocols and honest about being the wrong sound for fast ones.

The cost is that a change to the sound takes up to this long to be heard. It would still be the wrong trade
for anything moment-locked, and this is why the reward click does not go through here.

**Every player here obeys this, and the last exception was the last crackle.** `Sonifier` kept a callback
long after the others gave theirs up, on the argument that the live EEG envelope should not sit behind a
deep buffer. That argument is about the *ring's* read cursor — how far behind the EEG the render is — which
is a different quantity from the output buffer's depth and is untouched by writing ahead. Measured with real
EEG, the ring holds a 43–79 ms lead on its own; the callback bought nothing and cost the paragraph above."""
RING_SECONDS = 2.0

BLOCK_SECONDS = BLOCK_FRAMES / SAMPLE_RATE
STALL_SECONDS = 2.0
"""How long a stream may go without finishing a write, or without the device taking any of what it was
given, before it is treated as dead. Both are generous next to the ~230 ms of buffer underneath."""
REOPEN_TRIES = 3
REOPEN_PAUSE_SECONDS = 0.25

# Spectral tilts, as IIR filters over white noise. Pink is the standard -3 dB/octave approximation from the
# DSP literature, accurate to a fraction of a dB across the audible range; brown is a leaky integrator, a
# further -6 dB/octave. The third number scales each to **unit RMS**, measured rather than guessed —
# without it, switching colour changes the loudness by 6× and reads as the control being broken.
# Filtered in one `lfilter` call per audio block rather than per sample, because the callback runs at
# 48 kHz and the per-call-overhead argument that governs 8-sample EEG blocks inverts completely at this
# size.
NOISE_FILTERS = {
    "white": (np.array([1.0]), np.array([1.0]), 1.0),
    "pink": (np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786]),
             np.array([1.0, -2.494956002, 2.017265875, -0.522189400]), 13.7),
    "brown": (np.array([0.02]), np.array([1.0, -0.98]), 11.8),
}

# **The music is high-passed, and this is not tidiness — it is what a small speaker can physically do.**
# A 1/f tilt keeps rising toward DC and brown is a leaky integrator, so both pile energy up where nothing
# can reproduce it: measured, **28% of pink's and of brown's total power sat below 80 Hz, 8-11% of it below
# 20**, and brown's random walk wanders in DC offset on top of that. A laptop driver cannot move air at
# those frequencies; it just travels, and what you hear is the crackle of a cone at its excursion limit,
# with intermodulation on top of it — music chosen for having no structure making a noise of its own.
#
# 70 Hz at order 2 takes pink to 4.2% and brown to 6.9% below 80 Hz while leaving the tilt intact (brown
# still holds 12× the energy at 150-300 Hz that it does at 3-6 kHz, against pink's 0.9 and white's 0.1).
# The gains above are re-measured *through* this filter, or the colours would no longer match in loudness —
# they were 12.0 and 10.3 before it existed.
#
# Nothing is being made audible here, and that is the difference from the EEG sonifier: 2-38 Hz has to be
# moved up before it can be heard at all, where noise is already broadband audio and band-limiting is the
# whole job.
HIGHPASS_HZ = 70.0
HIGHPASS_ORDER = 2
PEAK = 0.19                 # full scale at 5.3 sigma; see NoisePlayer._callback
HEADROOM = 0.0833
"""A deliberate **21.6 dB** pad on generated noise, so the operator's own control has room *below*.

Broadband noise at a given full-scale amplitude is far louder than the number suggests — it carries energy
at every frequency at once, where a tone or a track does not — so music rendered at full scale needed its
volume set to about 5 to be a background, and 5 is a control with nowhere left to go: one step down is 0.
Padding the *source* instead puts usable music well up the range, where it can be halved twice and still be
heard.

**It was 0.25, a 12 dB pad, and that was still three times too loud in use** — reported from a session at
volume 60, which is the default and should stay the default. The pad is the right place to fix it: the
alternative is telling the operator to run at 20, which is the same sound with less control left underneath
it. Measured: 0.0473 RMS at volume 100 and 0.0280 at 60 before, 0.0158 and 0.0093 after.

It is on the noise alone, not on `Music.volume`: a file's loudness is whatever was mastered into it, and
padding a track the operator supplied would be second-guessing their own mix. The other two *generated* sounds
are calibrated against this one — see `TONE_PEAK` and `SHEPARD_PEAK` — so they move with it, or switching
source would become the unmarked volume change that calibration exists to prevent.

**And nothing on the operator's scale moves with it.** Every screen that selects noise gets this generator, so
a change here reaches all of them at once, which is the point — the engine was too loud, not one screen's
setting. `at_view.MUSIC_VOLUME` is a number somebody set by ear and it stays at 20; raising it to keep that
screen's music at its old loudness was tried and is exactly the fix undoing itself."""
NOISE_COLOURS = {
    "pink": "1/f, equal energy per octave",
    "white": "flat, equal energy per hertz",
    "brown": "1/f², deeper again",
}


# **Pitch, for music whose two ends have to be told apart.** A tilt already makes brown darker than pink, but
# a tilt is a slope and the ear reads it as *texture*; what makes filtered noise read as **pitched** is a
# resonance. So a non-zero pitch replaces the flat high-pass on that end with a band-pass an octave wide
# centred on it, and the two ends then differ in pitch as well as in tilt.
#
# The idea: a higher pitched reward pink noise when alpha is dominant, and a lower pitched theta reward
# noise when theta is dominant.
# It is a *second* axis rather than a replacement for the tilt, because the two are perceptually different —
# tilt says what the music is made of, pitch says where it sits — and the whole point is that the trainee can
# tell which end they are at with the eyes closed.
#
# An octave wide, and that is a decision. Narrower reads as a tone and starts to ring, which is what
# `CLICK_PROFILES` is careful to avoid and would put an event into music chosen for having none; wider stops
# being a pitch at all and is just a tilt again.
PITCH_OCTAVES = 1.0
PITCH_ORDER = 2
PITCH_LO_HZ = HIGHPASS_HZ
"""Never below the high-pass corner, because that corner is not cosmetic: 28% of pink's power sat below 80 Hz
and drove a laptop speaker into its excursion limit. A pitch that reached under it would put the crackle back
by another door."""
PITCH_HI_HZ = 4000.0


# --- the pitch-tracking tone --------------------------------------------------------------
#
# The `tone` mode maps the signal between `base_hz` and `base_hz + range_hz`. 220 to 660 Hz is two octaves
# a fifth apart from a low A, which is a range the ear resolves finely across its whole span — below about
# 150 Hz pitch discrimination falls off, and above ~1 kHz a sustained sine becomes tiring rather than
# informative.
TONE_BASE_HZ = 220.0
TONE_RANGE_HZ = 440.0
TONE_GLIDE_SECONDS = 0.35
"""Shorter than the Shepard tone's 1.5 s: a bounded pitch is read as a *position* and wants to feel
responsive, where an unbounded one is read as a direction of travel and wants to feel slow."""
TONE_PEAK = 0.06
"""Calibrated against the pink noise, because `Music.volume` is shared across sources and a mode that was twice
as loud at the same number would make switching an unmarked volume change — and the calibration is by *ear*,
not by RMS, because for a sine against broadband the two disagree by about a factor of two.

It sat a little *under* the noise for a long time, on the claim that a sine concentrates its energy in one
critical band and is therefore perceptually louder at the same RMS. That has loudness summation backwards:
loudness adds across critical bands, so the broadband sounds — the noise, the Shepard's twelve partials —
are the ones that come out ahead at equal RMS, by roughly 10 phons at background levels. The equal-RMS
calibration therefore pinned both one-sine instruments (this tone and the EEG voice) at about half the
perceived loudness of every other source, and it was measured the only way loudness can be: reported from
sessions, twice — "EEG music at 60 is about half as loud as the Shepard at 60", and still half after the
loudness *range* was fixed, which isolated the calibration as the remainder.

So the sine sits at 3× the noise's RMS (+9.5 dB, about the 10 phons the reports ask for): 0.06 gives
0.0424 RMS at volume 100 against the pink noise's 0.0158. The old value was 0.02 — the 0.9×-by-RMS
figure — and before `HEADROOM` divided the noise by three it was 0.06 against 0.047, the same wrong ratio
at a louder level."""


# --- the Shepard tone ---------------------------------------------------------------------
#
# The constants: `Subtones` 12, `Minimum Frequency` 1 Hz, `Maximum Frequency` 4000 Hz. Those three are
# consistent rather than three separate choices — twelve octaves above 1 Hz is 4096 — so the wheel is
# exactly 12 octaves wide and the top and bottom partials are both outside hearing, which is what makes
# the wrap inaudible.
SHEPARD_SUBTONES = 12
SHEPARD_MIN_HZ = 1.0
SHEPARD_SIGMA_OCTAVES = 2.6
"""Width of the amplitude bell, in octaves. **Ours.**

It is what trades brightness against the seam. Wide, and the partials at both ends of the wheel are still
audible when they wrap, which puts a soft click into the music once per octave of travel. Narrow, and the tone
becomes a single sine and stops being a Shepard tone at all — there is no ambiguity of octave left to make
the rise endless. 2.6 puts the end partials at exactly exp(-0.5·(6/2.6)²) = 7% of centre amplitude, about
23 dB down, which is under the noise of anything else in the room while leaving five partials clearly
audible."""
SHEPARD_OCTAVES = 6.0
"""How far the comb slides between level 0 and level 1, in octaves.

Half the wheel, so the ends of the range are as far apart as they can be *without* wrapping — one more
octave in either direction and level 1 would sound like level 0, which is the failure mode a design with no
ceiling has instead of saturation. A `value/change` toggle with an offset would sidestep the question;
this is the answer for the `value` half of it."""
SHEPARD_GLIDE_SECONDS = 1.5
"""Time constant of the glide toward the level. A Shepard tone is heard as a *direction of travel*, so it
has to move slowly enough that the direction is legible — following a level that updates twenty times a
second reads as vibrato rather than as rising."""
SHEPARD_FLOOR_HZ = HIGHPASS_HZ
"""Below this the partials are rolled off, and it is the noise's crackle fix arriving at the Shepard tone.

The wheel is 1–4096 Hz, and its amplitude bell is centred six octaves up — on **64 Hz** — so the
comb's loudest partials sit at the bottom of what a laptop driver can reproduce and its lower half is pure
cone excursion: a 4 Hz partial at 31% of centre amplitude moves the cone without making a sound, and what is
heard instead is the crackle of the cone at its travel limit, per partial, which is why it reads as
crackling *notes*. Same physics as `HIGHPASS_HZ`, whose corner was measured for this exact speaker, so the
same number rather than a second guess. Measured mid-wheel: **60.8% of the rendered power sat below 70 Hz
and 20.4% below 20** — worse than the pink noise (28% below 80) that forced the original fix — against 9.2%
below 70 with this floor in place.

A smooth roll-off (fourth power below the corner, ~24 dB/octave) rather than a cut, because a partial slides
*through* the floor as the comb moves — a hard edge would step its amplitude between blocks, which is a
click, the very thing the envelope bell exists to avoid at the wrap. The normalisation in `_block` divides
by the post-floor amplitude sum, so the surviving partials carry the tone at the same peak."""
SHEPARD_PEAK = 0.0507
"""Full scale, **chosen so that one volume setting means the same thing on this source as on the noise.**

Not a headroom argument: twelve sines have no long tail of peaks to leave room for, so any value up to 1
would be safe. It is a *calibration*. `Music.volume` is persisted per protocol and shared across sources, so
music rendered 1.75× louder than the other one at the same number turns switching source into an unmarked
volume change.

Measured at volume 100 over 2.5 s: 0.0507 gives 0.0158 RMS against the pink noise's 0.0158, re-measured
*through* `SHEPARD_FLOOR_HZ` exactly as the noise gains were re-measured through `HIGHPASS_HZ` — the floor
thins the comb, and the normalisation then concentrates the peak into fewer partials, so the RMS rose to
0.0208 at the old 0.0667 and the calibration had silently drifted a third loud. Before the floor it was
0.0667, and 0.20 before `HEADROOM` divided the noise by three — the number moves whenever its reference
does, because the calibration is to the noise and not to a number."""


def pitch_sos(centre_hz: float):
    """A band-pass an octave wide around `centre_hz`, or None for no pitch at all."""
    from scipy.signal import butter

    if not centre_hz:
        return None
    centre = min(PITCH_HI_HZ, max(PITCH_LO_HZ, float(centre_hz)))
    half = 2.0 ** (PITCH_OCTAVES / 2.0)
    lo = max(20.0, centre / half)
    hi = min(SAMPLE_RATE / 2 - 100.0, centre * half)
    return butter(PITCH_ORDER, [lo, hi], btype="bandpass", fs=SAMPLE_RATE, output="sos")


RESPONSE_POINTS = 8192
"""Grid for the analytic gains below. Enough that a one-octave band at 85 Hz is covered by tens of points."""


def _response(colour: str, centre_hz: float) -> np.ndarray:
    """The complex frequency response of one end's whole chain: tilt, pitch, and the shared high-pass."""
    from scipy.signal import freqz, sosfreqz

    w = np.linspace(0.0, np.pi, RESPONSE_POINTS)
    b, a, gain = NOISE_FILTERS[colour]
    tilt = np.full(RESPONSE_POINTS, gain, dtype=complex) if (len(a) == 1 and len(b) == 1) \
        else freqz(b, a, worN=w)[1] * gain
    total = tilt * sosfreqz(highpass_sos(), worN=w)[1]
    sos = pitch_sos(centre_hz)
    if sos is not None:
        total = total * sosfreqz(sos, worN=w)[1]
    return total


def pitch_gain(colour: str, centre_hz: float) -> float:
    """Unit RMS through one end's whole chain — its tilt, its pitch, and the high-pass after them.

    The per-colour gains in `NOISE_FILTERS` exist because changing colour otherwise changed the loudness
    sixfold and read as a broken control. A pitch has the same problem and worse: a band-pass at 200 Hz keeps
    far more of pink's power than one at 2 kHz, so moving the pitch would move the volume. The centre is a
    free number rather than one of three, so it cannot be a table.

    **Computed from the filter response rather than measured on a probe**, and that is not a tidying. White
    noise through a one-octave band at 85 Hz yields only tens of independent values in a half-second probe, so
    the RMS *estimate* carried about 12% of noise — which came out as the two ends of the fade differing in
    loudness by up to 22%, a slow swell from one end to the other on music chosen for having no events in it.
    The integral has no estimator error at all.

    The high-pass is in it because that runs after the mix and is not gentle where a low pitch lives.
    Normalising against part of the chain is normalising against the wrong chain.
    """
    if not centre_hz:
        return 1.0
    power = float(np.mean(np.abs(_response(colour, centre_hz)) ** 2))
    return 1.0 / np.sqrt(power) if power > 1e-12 else 1.0


def highpass_sos():
    """The music's band limit, designed once. Both the live stream and the preview use it, or an audition
    would be a different sound from the thing it previews."""
    from scipy.signal import butter

    return butter(HIGHPASS_ORDER, HIGHPASS_HZ, btype="highpass", fs=SAMPLE_RATE, output="sos")


PREVIEW_SECONDS = 1.6       # long enough to hear a tilt, short enough that nobody waits for it to end
_preview_cache: dict[str, Path] = {}
_preview_dir: Path | None = None


def preview_path(colour: str, pitch_hz: float = 0.0) -> Path | None:
    """A short burst of one tilt, rendered to a WAV so it can be auditioned.

    Rendered and handed to `afplay` rather than opened on the audio device, for the same reason the click
    is: a live session may already hold that device, and a preview that silenced the music it is previewing
    would be worse than no preview. Cached per colour per process.

    A noise colour is the one setting here whose name tells you nothing about the sound — "1/f²" does not
    predict how deep brown is — so a preview is not a convenience, it is the only way the control says what
    it does.
    """
    global _preview_dir
    if colour not in NOISE_FILTERS:
        return None
    # Keyed by the pitch as well, or an audition would be the sound of a *different* setting — the same rule
    # that made the preview carry the 12 dB pad rather than previewing four times the music.
    key = f"{colour}@{round(pitch_hz)}"
    if key in _preview_cache:
        return _preview_cache[key]
    from scipy.signal import sosfilt

    b, a, gain = NOISE_FILTERS[colour]
    samples = int(SAMPLE_RATE * PREVIEW_SECONDS)
    white = np.random.default_rng(0).standard_normal(samples)
    shaped = (white * gain) if len(a) == 1 and len(b) == 1 else lfilter(b, a, white) * gain
    # The pitch band-pass and its unit-RMS gain, exactly as the live music applies them — an audition at a
    # different loudness or a different centre is an audition of something else.
    sos = pitch_sos(pitch_hz)
    if sos is not None:
        shaped = sosfilt(sos, shaped) * pitch_gain(colour, pitch_hz)
    shaped = sosfilt(highpass_sos(), shaped)
    # A fade at each end, or the burst starts and stops with a click — which would be a second sound on
    # top of the one being auditioned, and the wrong one to be listening to.
    edge = int(SAMPLE_RATE * 0.05)
    ramp = np.ones(samples)
    ramp[:edge] = np.linspace(0.0, 1.0, edge)
    ramp[-edge:] = np.linspace(1.0, 0.0, edge)
    # Same full-scale convention *and the same headroom* as the live music, so the preview is as loud as the
    # thing it previews — the volume is applied by the player on top, from the same setting.
    data = np.clip(shaped * ramp * PEAK * HEADROOM, -1.0, 1.0)
    if _preview_dir is None:
        _preview_dir = Path(tempfile.mkdtemp(prefix="nfb-noise-"))
    path = _preview_dir / f"{key}.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes((data * 32767).astype(np.int16).tobytes())
    _preview_cache[key] = path
    return path


def muted() -> bool:
    """`NFB_NO_OUTPUTS` silences audio for the process. See `shadow.muted` for why it exists.

    This module was missing from that guard, which was a real hole rather than an omission of principle:
    every protocol that enables music by default opened a live audio stream from the test suite, and the
    only reason it was not obvious is that generated noise under a test run sounds like nothing in particular.

    Checked by the two paths that open an audio *device*. The track player is not gated here, because
    playing a file needs a path that nothing sets by default, and gating it would defeat the tests that
    exercise it deliberately with a stubbed player.
    """
    return bool(os.environ.get("NFB_NO_OUTPUTS"))


def available() -> bool:
    return not muted()


def tracks(path: str | Path) -> list[Path]:
    """The playable files at `path`: itself if it is one, or its audio files sorted if it is a folder."""
    p = Path(path).expanduser()
    if p.is_file():
        return [p] if p.suffix.lower() in AUDIO_SUFFIXES else []
    if p.is_dir():
        return sorted(f for f in p.iterdir()
                      if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES)
    return []


class TrackPlayer:
    """Plays a list of files in order, looping, at its own volume.

    One `afplay` per track on a worker thread. Volume changes take effect at the next track rather than
    mid-track, which is a real limitation and the price of not decoding audio ourselves — `afplay` reads
    mp3 and m4a without help, and matching that would mean a decoder dependency for a backing track.
    """

    def __init__(self) -> None:
        self.files: list[Path] = []
        self.volume = 60.0
        self.index = 0
        self.held = False
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def now_playing(self) -> str:
        if not self.playing or not self.files:
            return ""
        return self.files[self.index % len(self.files)].name

    def start(self, files: list[Path], volume: float) -> None:
        if self.playing and files == self.files:
            self.volume = volume
            return
        self.stop()
        self.files = list(files)
        self.volume = volume
        if not self.files:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="music")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set() and self.files:
            track = self.files[self.index % len(self.files)]
            gain = f"{min(100.0, max(0.0, self.volume)) / 100:.3f}"
            try:
                self._proc = subprocess.Popen(["afplay", "-v", gain, str(track)],
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
            except OSError:
                return
            self._proc.wait()
            self._proc = None
            if self._stop.is_set():
                return
            self.index += 1

    def hold(self, paused: bool) -> None:
        """Pause or resume playback in place, for the pause-on-violation feedback model.

        `SIGSTOP`/`SIGCONT` on the `afplay` child rather than stopping and restarting it, because
        restarting would begin the track again — and a backing track that restarts every time an inhibit
        fires is a different, much worse experience than one that stops mid-phrase and carries on. The
        information is in *when* it stopped, which needs the position kept.
        """
        if paused == self.held:
            return
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self.held = paused
            return
        try:
            proc.send_signal(signal.SIGSTOP if paused else signal.SIGCONT)
        except OSError:
            return          # the track ended between the check and the signal; the next one starts fresh
        self.held = paused

    def stop(self) -> None:
        self._stop.set()
        self.held = False
        proc = self._proc
        if proc is not None:
            # Shutdown must not raise, whatever state the child is in — this runs from `session.stop()`,
            # which is also what restores the operator's brightness and volume, so an exception here
            # would leave the machine dimmed. Broader than OSError on purpose.
            try:
                # a stopped process ignores terminate until it is allowed to run again
                proc.send_signal(signal.SIGCONT)
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


class _AudioStream:
    """The output stream every generated sound writes into, and what keeps it alive.

    One copy of open, write, watch and stop for the four players below, because the recovery is the part
    that has to exist once. A player supplies `_block(frames)` and a `stream_name`.

    **A stream can stop playing without failing.** The output device changing under a running stream —
    a screenshot's shutter sound, headphones going in, a Bluetooth speaker switching profile — leaves an
    object that still accepts everything written to it and plays none of it, and the console went silent
    for the rest of the session with the panel still reading "music on". So the writer measures the device
    instead of trusting it: audio handed over should be consumed at real time, and a lead that grows past
    `STALL_SECONDS` means nothing is listening. `start` also re-checks a writer that has not completed a
    write in that long, for the shape of the same failure where the write blocks forever instead.

    `starved` is the underrun PortAudio does not flag: the buffer down to under one block, which is a
    crackle whether or not the host reports it.
    """

    stream_name = "audio-writer"
    held = False
    """Silenced without stopping the stream. The sonifier is never held — see `MusicOutput.apply`."""

    def _init_stream(self) -> None:
        self.error: str | None = None
        self._stream = None
        self._writer = None
        self._stop = threading.Event()
        self._last_write = 0.0
        self.dropouts = 0
        """Blocks the device ran out of data for. **Counted rather than ignored**, because a dropout and a
        speaker at its excursion limit both sound like crackle and the fix differs."""
        self.starved = 0
        self.restarts = 0

    @property
    def running(self) -> bool:
        return self._stream is not None

    def _open(self) -> bool:
        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = f"sounddevice unavailable: {exc}"
            return False
        try:
            # **No callback: a writer thread and a deep buffer.** See `BUFFER_LATENCY` — with a callback,
            # the device waits on Python every block and every stall in this process is audible.
            stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                     blocksize=BLOCK_FRAMES, latency=BUFFER_LATENCY)
            stream.start()
        except Exception as exc:
            self.error = f"no audio output: {exc}"
            return False
        self._stream = stream
        self._last_write = time.monotonic()
        return True

    def start(self) -> bool:
        if muted():
            return False
        if self._stream is not None:
            if time.monotonic() - self._last_write < STALL_SECONDS:
                return True
            self.stop()                 # a writer stuck in a write that will never return
            self.restarts += 1
        if not self._open():
            self._stream = None
            return False
        self._stop.clear()
        self._writer = threading.Thread(target=self._run, daemon=True, name=self.stream_name)
        self._writer.start()
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

    def _reopen(self, why: str) -> bool:
        """Build a fresh stream in place of one that has stopped playing. False once it has given up."""
        self._shut(self._stream)
        self._stream = None
        for _ in range(REOPEN_TRIES):
            if self._stop.wait(REOPEN_PAUSE_SECONDS):
                return False
            if self._open():
                self.restarts += 1
                self.error = None
                return True
        self.error = why
        return False

    def _run(self) -> None:
        """Fill the buffer for as long as the stream lives. `write` blocks when it is full, so this paces
        itself and needs no clock of its own — and how far it runs ahead is the health of the device."""
        written, began = 0, time.monotonic()
        while not self._stop.is_set():
            stream = self._stream
            if stream is None:
                return
            block = self._block(BLOCK_FRAMES)
            try:
                if stream.write(block):
                    self.dropouts += 1
            except Exception as exc:
                if not self._reopen(f"audio write failed: {exc}"):
                    return
                written, began = 0, time.monotonic()
                continue
            now = time.monotonic()
            self._last_write = now
            written += len(block)
            verdict = self._health(written, now - began)
            if verdict == "dead":
                if not self._reopen("the audio device stopped playing what it was given"):
                    return
                written, began = 0, time.monotonic()
            elif verdict == "starved":
                self.starved += 1

    @staticmethod
    def _health(written: int, elapsed: float) -> str:
        """What the lead says about the device — audio handed over that it has not played yet.

        A healthy stream holds the lead at the buffer's depth: it drains in real time, so `write` blocks
        and this thread paces itself. A lead that keeps growing is a device consuming nothing, and a lead
        under one block is a buffer about to run dry, which is a crackle whether or not PortAudio says so.
        """
        lead = written / SAMPLE_RATE - elapsed
        if lead > STALL_SECONDS:
            return "dead"
        if written > SAMPLE_RATE and lead < BLOCK_SECONDS:
            return "starved"
        return ""

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self.held = False
        self._stop.set()
        writer = self._writer
        self._writer = None
        if writer is not None and writer is not threading.current_thread():
            writer.join(timeout=1.0)
        self._shut(stream)


class NoisePlayer(_AudioStream):
    """Continuous noise at a chosen tilt, streamed. Optionally two tilts, crossfaded by a signal.

    Generated rather than played from a file, which is not just convenience: a file loops, and a loop of
    noise has a period you eventually start hearing, which is exactly the kind of structure this is meant
    not to have.

    **The crossfade is how alpha-theta training's two soundscapes are rendered here.** The reference
    implementation plays two recordings — surf and a babbling brook — and fades between them as the state
    moves, one continuous output whose *character* rather than its loudness carries the information. Those
    two differ mostly in spectral tilt: measured, the surf loop's energy peaks near DC and the brook's
    around 800 Hz. So brown against pink is the same distinction generated, and it inherits the reason
    `NoisePlayer` exists at all — neither end has a period you can learn.

    Both tilts filter the **same** white noise, so the crossfade interpolates one source into the other rather
    than mixing two independent sources. That keeps the mid-fade loudness in line: at balance 0.5 the RMS is
    within a few percent of either end, where two independent music sources would add in power and bulge.

    `held` silences it without stopping the stream — pause-on-violation, and under Quiet Reward the same
    thing read the other way round. Restarting the stream instead would click, and a click at the moment a
    target fails would be a second, unintended discrete signal.
    """

    stream_name = "noise-player"

    def __init__(self) -> None:
        self.colour = "pink"
        self.second_colour: str | None = None
        self.balance = 0.0              # 0 = all `colour`, 1 = all `second_colour`
        self.volume = 60.0
        self.held = False
        self._init_stream()
        self._zi = None
        self._zi2 = None
        # Per end: the pitch in force, the colour it was measured against, its filter, its state and its
        # unit-RMS gain. Two of each rather than one, because the two ends are pitched separately — that is
        # the whole feature — and each needs its own filter state or they would share a ringing.
        self._pitch_hz = [0.0, 0.0]
        self._pitch_colour: list[str | None] = [None, None]
        self._pitch_sos: list = [None, None]
        self._pitch_zi: list = [None, None]
        self._pitch_gain = [1.0, 1.0]
        self._fade_rho: float | None = 1.0
        """Correlation between the two ends, measured — and it is what keeps the fade's loudness flat.

        The crossfade filters **one** white source through both ends, so with only a tilt between them the two
        outputs are nearly identical and an amplitude fade interpolates correctly: that is why the RMS held
        within 5% across the sweep and why the fade was written that way. Pitch breaks it. Two band-passes an
        octave apart barely overlap, so at mid-fade they add in *power* rather than amplitude and the music dips
        — measured, 28% down at balance 0.5, which on music chosen for having no events in it is a slow swell
        in the middle of the range.

        `E[mix²] = (1-f)² + f² + 2f(1-f)ρ` with both ends at unit RMS, so dividing by its square root holds
        the loudness flat for *any* ρ. ρ=1 gives a scale of exactly 1, so the unpitched case is bit-identical
        to before rather than merely close — which is the property that makes this safe to apply always."""
        self._hp = None
        self._hp_zi = None
        self._last_balance = 0.0
        self._last_level = 0.0
        self._rng = np.random.default_rng()

    def configure(self, colour: str, volume: float, second_colour: str | None = None,
                  balance: float = 0.0, pitch_hz: float = 0.0,
                  second_pitch_hz: float = 0.0) -> None:
        if colour not in NOISE_FILTERS:
            colour = "pink"
        if second_colour is not None and second_colour not in NOISE_FILTERS:
            second_colour = None
        if colour != self.colour:
            self.colour = colour
            self._zi = None            # a new filter starts from rest rather than another's state
        if second_colour != self.second_colour:
            self.second_colour = second_colour
            self._zi2 = None
        # Redesigned only when the number actually changes, because `pitch_gain` runs a half-second probe
        # through the chain to measure it and this is called twenty times a second.
        self._set_pitch(0, colour, pitch_hz)
        self._set_pitch(1, second_colour or colour, second_pitch_hz)
        self.balance = min(1.0, max(0.0, float(balance)))
        self.volume = volume
        if self._fade_rho is None:
            self._measure_rho()

    def _measure_rho(self) -> None:
        """Correlation between the two ends, from their responses — exact, and for the same reason as the
        gains: driven by one white source, the two outputs' correlation is the overlap integral of their
        responses, so there is nothing to estimate."""
        if self.second_colour is None:
            self._fade_rho = 1.0
            return
        ha = _response(self.colour, self._pitch_hz[0]) * self._pitch_gain[0]
        hb = _response(self.second_colour, self._pitch_hz[1]) * self._pitch_gain[1]
        na = np.sqrt(np.mean(np.abs(ha) ** 2))
        nb = np.sqrt(np.mean(np.abs(hb) ** 2))
        if na < 1e-9 or nb < 1e-9:
            self._fade_rho = 1.0
            return
        self._fade_rho = float(np.clip(np.mean(np.real(ha * np.conj(hb))) / (na * nb), -0.999, 1.0))

    def _set_pitch(self, end: int, colour: str, centre_hz: float) -> None:
        centre = 0.0 if not centre_hz else min(PITCH_HI_HZ, max(PITCH_LO_HZ, float(centre_hz)))
        if self._pitch_hz[end] == centre and self._pitch_colour[end] == colour:
            return
        self._pitch_hz[end] = centre
        self._pitch_colour[end] = colour
        self._pitch_sos[end] = pitch_sos(centre)
        self._pitch_gain[end] = pitch_gain(colour, centre)
        self._pitch_zi[end] = None      # a new filter starts from rest rather than another's state
        self._fade_rho = None           # recomputed lazily: `_set_pitch` runs before the colours settle

    def _band_limit(self, x: np.ndarray) -> np.ndarray:
        """Take out what no speaker can render, keeping the filter state across callbacks."""
        from scipy.signal import sosfilt

        if self._hp is None:
            self._hp = highpass_sos()
            self._hp_zi = np.zeros((self._hp.shape[0], 2))
        out, self._hp_zi = sosfilt(self._hp, x, zi=self._hp_zi)
        return out

    def _shape(self, white: np.ndarray, colour: str, state: str,
               end: int | None = None) -> np.ndarray:
        """One end of the music: its tilt, then its pitch, keeping both filter states across blocks."""
        b, a, gain = NOISE_FILTERS[colour]
        if len(a) == 1 and len(b) == 1:
            shaped = white * gain
        else:
            zi = getattr(self, state)
            if zi is None or len(zi) != max(len(a), len(b)) - 1:
                zi = np.zeros(max(len(a), len(b)) - 1)
            shaped, zi = lfilter(b, a, white, zi=zi)
            setattr(self, state, zi)
            shaped = shaped * gain
        return shaped if end is None else self._pitched(shaped, end)

    def _pitched(self, x: np.ndarray, end: int) -> np.ndarray:
        """The pitch band-pass for one end, normalised back to unit RMS so moving it is not a volume change."""
        from scipy.signal import sosfilt

        sos = self._pitch_sos[end]
        if sos is None:
            return x
        if self._pitch_zi[end] is None:
            self._pitch_zi[end] = np.zeros((sos.shape[0], 2))
        out, self._pitch_zi[end] = sosfilt(sos, x, zi=self._pitch_zi[end])
        return out * self._pitch_gain[end]

    def _block(self, frames: int) -> np.ndarray:
        """One block of the music, ready to write.

        **The fade and the loudness are ramped *across* the block, not applied to it.** Both are read from
        a session that updates twenty times a second, so holding either constant for a block and then
        changing it puts a step in the waveform every 43 ms — which is a click, quietly, on top of music
        chosen for having no events in it. Interpolating from the last block's value costs one `linspace`.
        """
        # A hold fades over one block rather than cutting — the same ramp the resume already rides; the
        # reasoning, and the flickering gate that makes it matter, is at the Shepard tone's copy of this.
        if self.held and self._last_level <= 0.0:
            return np.zeros((frames, 1), dtype=np.float32)
        white = self._rng.standard_normal(frames)
        shaped = self._shape(white, self.colour, "_zi", end=0)
        if self.second_colour is not None:
            other = self._shape(white, self.second_colour, "_zi2", end=1)
            fade = np.linspace(self._last_balance, self.balance, frames)
            shaped = (1.0 - fade) * shaped + fade * other
            # Flat loudness across the fade for any overlap between the two ends — see `_fade_rho`. One
            # `sqrt` on an array we already have; at ρ=1 every element is exactly 1.0.
            rho = self._fade_rho if self._fade_rho is not None else 1.0
            power = (1.0 - fade) ** 2 + fade ** 2 + 2.0 * fade * (1.0 - fade) * rho
            shaped = shaped / np.sqrt(np.maximum(power, 1e-6))
        self._last_balance = self.balance
        shaped = self._band_limit(shaped)
        level = 0.0 if self.held else min(100.0, max(0.0, self.volume)) / 100.0
        gain = np.linspace(self._last_level, level, frames)
        self._last_level = level
        # `_shape` has already applied each tilt's unit-RMS scaling, and 0.19 puts full scale at 5.3
        # standard deviations.
        # Gaussian noise has no bounded peak, so "never clips" is not a thing that can be arranged — the
        # question is how often, and beyond 5.3σ that is one sample every few minutes, which is inaudible.
        # It matters because clipped noise is harsh in exactly the way this music is chosen not to be, and
        # it is audible long before it shows up in a level meter.
        out = np.clip(shaped * gain * PEAK * HEADROOM, -1.0, 1.0)
        return out.astype(np.float32).reshape(-1, 1)

class TonePlayer(_AudioStream):
    """One sine whose **pitch** tracks the signal. The `tone` feedback mode.

    Not the Shepard tone with the partials removed, and not the reward tone either. `ShepardPlayer` is unbounded
    by construction, which is what an infra-slow drifting signal needs; `cues.tone_at` is a 160 ms event whose
    pitch says *which* reward earned it. This is a continuous carrier that rises and falls with a 0…1 signal
    between two named frequencies, which is the plainest possible audio mapping and the one a listener
    interprets without being taught.

    A pure sine rather than anything richer, deliberately. A harmonically complex tone changes *timbre* as it
    moves, which the ear reads as a second dimension — and there is only one number here. The cost is that a
    sine is easy to stop noticing, which is the argument for the noise mask instead when the session is long.

    Glided, for the same reason the Shepard tone is: the signal updates several times a second and a pitch that
    followed it exactly would read as vibrato rather than as height.
    """

    stream_name = "tone-player"

    def __init__(self) -> None:
        self.volume = 40.0
        self.base_hz = TONE_BASE_HZ
        self.range_hz = TONE_RANGE_HZ
        self.level = 0.0
        self.held = False
        self._init_stream()
        self._pitch = TONE_BASE_HZ
        self._phase = 0.0
        self._last_level = 0.0

    def configure(self, volume: float, level: float, base_hz: float = TONE_BASE_HZ,
                  range_hz: float = TONE_RANGE_HZ) -> None:
        self.volume = volume
        self.base_hz = max(40.0, float(base_hz))
        self.range_hz = max(10.0, float(range_hz))
        self.level = min(1.0, max(0.0, float(level)))

    @property
    def pitch_hz(self) -> float:
        return self._pitch

    def _block(self, frames: int) -> np.ndarray:
        # A hold fades over one block rather than cutting — see the Shepard tone's copy of this comment.
        if self.held and self._last_level <= 0.0:
            return np.zeros((frames, 1), dtype=np.float32)
        target = self.base_hz + self.range_hz * self.level
        alpha = (frames / SAMPLE_RATE) / (TONE_GLIDE_SECONDS + frames / SAMPLE_RATE)
        self._pitch += alpha * (target - self._pitch)
        # Phase integrated, never recomputed from a clock: the frequency moves every block, and
        # `sin(2*pi*f*t)` with a shared `t` would step in phase each time it did.
        increment = 2 * np.pi * self._pitch / SAMPLE_RATE
        angles = self._phase + increment * np.arange(1, frames + 1)
        self._phase = float(angles[-1] % (2 * np.pi))
        level = 0.0 if self.held else min(100.0, max(0.0, self.volume)) / 100.0
        gain = np.linspace(self._last_level, level, frames)
        self._last_level = level
        out = np.clip(np.sin(angles) * gain * TONE_PEAK, -1.0, 1.0)
        return out.astype(np.float32).reshape(-1, 1)

class ShepardPlayer(_AudioStream):
    """A Shepard tone: a pitch that rises without ever getting higher, driven by the level.

    **This answers a problem every other source here dodges.** An infra-slow signal has no natural
    bounds and drifts, so any bounded
    sonification either saturates or has to be auto-ranged, and auto-ranging the feedback means the trainee
    is told about their variance rather than their state.

    A Shepard tone has no ceiling. Twelve sine partials an octave apart, with a bell-shaped amplitude
    envelope over log-frequency: as the whole comb slides upward, the top partial fades out as the bottom
    one fades in, so the pitch is heard as rising continuously and can rise for ever. Slide it far enough and
    it is back where it started with no audible seam. So an unbounded input maps onto an unbounded percept,
    which no volume, no brightness and no crossfade can do.

    **One design choice.** A `value/change` toggle (absolute position, or driven by the rate of change)
    is possible. Only `value` is implemented, because the level reaching this is already
    a 0…1 score against a threshold and its *derivative* would be a second-order feedback signal nothing
    else in this codebase produces. `SHEPARD_OCTAVES` then says how far the full range slides.

    The wrap is inaudible for a reason worth stating rather than trusting: the envelope's tails are what a
    partial is at when it crosses either end, so what wraps is always a partial at close to zero amplitude.
    A comb with a flat envelope would click once per octave.
    """

    stream_name = "shepard-player"

    def __init__(self) -> None:
        self.volume = 40.0
        self.level = 0.0
        self.held = False
        self._init_stream()
        self._position = 0.0        # octaves above the bottom of the wheel, glided toward the level
        self._phase = np.zeros(SHEPARD_SUBTONES)
        self._last_level = 0.0
        self.glide_seconds = SHEPARD_GLIDE_SECONDS
        self.travel_octaves = SHEPARD_OCTAVES

    def configure(self, volume: float, level: float, glide_seconds: float = 0.0,
                  travel_octaves: float = 0.0) -> None:
        self.volume = volume
        self.level = min(1.0, max(0.0, float(level)))
        # 0 takes the module default, the idiom every other "or the default" setting here uses. Held on the
        # music rather than read from the settings inside the audio callback, which runs on the sound card's
        # thread and must not reach into a dataclass the UI is editing.
        self.glide_seconds = max(0.01, glide_seconds or SHEPARD_GLIDE_SECONDS)
        # Capped at half the wheel rather than trusted: past that the top of the range wraps into sounding
        # like the bottom, which is the one thing this instrument is chosen for not doing. See
        # `feedback.Music.travel_octaves`.
        self.travel_octaves = min(SHEPARD_OCTAVES, max(0.25, travel_octaves or SHEPARD_OCTAVES))

    @property
    def octaves(self) -> float:
        """Where the comb currently sits, in octaves above the bottom of the wheel."""
        return self._position

    def partials(self, position: float) -> tuple[np.ndarray, np.ndarray]:
        """The twelve partials' frequencies and amplitudes at a given position on the wheel.

        Separate from the audio path so it can be checked without a sound card: the two properties that
        make this a Shepard tone rather than a comb — that every partial's amplitude is small at both ends
        of the wheel, and that the set of frequencies is periodic in the position with period one octave —
        are testable statements about this function.
        """
        k = np.arange(SHEPARD_SUBTONES)
        # Each partial's position on the wheel, wrapped. `position` slides all of them together.
        p = np.mod(position + k, SHEPARD_SUBTONES)
        freqs = SHEPARD_MIN_HZ * 2.0 ** p
        # A Gaussian in log-frequency, centred on the middle of the wheel. Width in octaves rather than in
        # hertz, because that is the axis the ear uses and the axis the partials are spaced on.
        centre = SHEPARD_SUBTONES / 2.0
        amps = np.exp(-0.5 * ((p - centre) / SHEPARD_SIGMA_OCTAVES) ** 2)
        return freqs, amps

    def _block(self, frames: int) -> np.ndarray:
        """One block, with the position glided and every partial's phase carried across.

        **Phase is integrated per partial and never recomputed from a clock.** Each partial's frequency
        changes as the comb slides, so `sin(2πft)` with a shared `t` would jump in phase every time `f`
        moved — twelve clicks per block, on music whose whole job is to be continuous. Accumulating
        `2πf/SR` per frame is the only formulation that stays continuous through a frequency change.
        """
        # A hold is a *fade*, not a cut. Returning zeros the moment `held` went true truncated the
        # waveform mid-cycle — a step, which is a click — and a condition sitting near its threshold
        # flickers, so the gate clicked each time it closed. The resume was always ramped (`gain` runs
        # from `_last_level`); the entry now rides the same ramp down, one block long, by rendering with
        # a target gain of zero until it gets there. Only then does the shortcut return silence.
        if self.held and self._last_level <= 0.0:
            return np.zeros((frames, 1), dtype=np.float32)
        # Glide toward the level rather than following it: a Shepard tone is read as a *direction* of travel,
        # so following a level that flows reads as vibrato and following one that steps reads as a jump to a
        # different tone. Which of those is the risk depends on the protocol, so the constant is a default —
        # see `feedback.Music.glide_seconds`.
        target = self.level * self.travel_octaves
        alpha = (frames / SAMPLE_RATE) / (self.glide_seconds + frames / SAMPLE_RATE)
        self._position += alpha * (target - self._position)
        freqs, amps = self.partials(self._position)
        # Above Nyquist there is nothing to render; the envelope has these at ~0 anyway, and this keeps a
        # small sample rate from aliasing the top of the wheel down into the middle of it.
        amps = np.where(freqs < SAMPLE_RATE / 2 - 100.0, amps, 0.0)
        # And below the speaker floor there is nothing to render either — see `SHEPARD_FLOOR_HZ`, where
        # the crackle this removes is explained. Smooth in frequency, so a partial sliding through the
        # corner fades rather than steps.
        amps = amps / (1.0 + (SHEPARD_FLOOR_HZ / np.maximum(freqs, 1e-9)) ** 4)
        increments = 2 * np.pi * freqs / SAMPLE_RATE
        steps = np.arange(1, frames + 1)
        angles = self._phase[:, None] + increments[:, None] * steps[None, :]
        self._phase = np.mod(angles[:, -1], 2 * np.pi)
        mixed = (amps[:, None] * np.sin(angles)).sum(axis=0)
        # Unit-ish peak: the envelope's amplitudes sum to a known total, so dividing by it makes the worst
        # case (every partial momentarily in phase) reach 1.0 rather than 12.
        mixed = mixed / max(float(amps.sum()), 1e-9)
        level = 0.0 if self.held else min(100.0, max(0.0, self.volume)) / 100.0
        gain = np.linspace(self._last_level, level, frames)
        self._last_level = level
        out = np.clip(mixed * gain * SHEPARD_PEAK, -1.0, 1.0)
        return out.astype(np.float32).reshape(-1, 1)

# --- the EEG's own frequency, made audible ------------------------------------------------
#
# The two ends of the pitch the band is played across. 220–440 Hz is one octave from a low A, which is
# what a one-octave band (8–16 Hz, say) fills exactly — see `stretch_factor`. Both ends are the
# operator's to set, because how far the tone travels is a property of every continuous output here and
# this one had it welded to a carrier constant.
EEG_PITCH_LO_HZ = 220.0
EEG_PITCH_HI_HZ = 440.0

EEG_TRACK_DELAY_SECONDS = 0.05
"""The delay the cFIR is designed for. Short, because this is the one output whose whole claim is that it
arrives while the rhythm is still happening — see `nfb.dsp.envelope`, where the delay is a design
parameter rather than a consequence of filter order. The fit is worse at 50 ms than at 100, and what a
worse fit costs here is a slightly noisier frequency estimate, which the smoothing below absorbs."""

EEG_TRACK_TAPS_SECONDS = 1.0
"""Filter length, in seconds of signal rather than in taps, because the sample rate is a session setting:
250 taps at 250 Hz and 125 at 125 Hz are the same filter. Length costs arithmetic and *not* delay."""

EEG_FREQ_SMOOTH_SECONDS = 0.20
"""One-pole on the instantaneous frequency before it becomes a pitch. The phase advance of a narrowband
estimate is a jittery quantity sample to sample — and where the band is quiet it is nearly random, since
there is no rhythm for it to be the frequency *of*. Unsmoothed it is heard as a rasp rather than as a
note. 200 ms is short against how fast a peak frequency really moves and long enough to hold a pitch
still."""

EEG_QUIET_GAIN = 0.12
"""What the tone sounds at when the band is silent, as a fraction of full. Not zero: this output is the
EEG, so a silence is indistinguishable from a lost channel — the same argument that stops `apply` from
holding the sonifier when a target fails. A twelfth of full is plainly quiet without being gone."""

# **Loudness calibrates itself against the band's own recent range — it replaced a typed µV number.**
# `full_at_uv` asked the operator how many microvolts of the band count as full loudness, and that is a
# number nobody has: it varies per person, per site, per montage, and nothing on the screen said what
# today's band was actually doing. Left unset it fell back to the raw trace's display gain — ±50 or ±100
# µV against a band that produces 2–5 — so the voice spent whole sessions within a few percent of its
# quiet floor, reported as "about half as loud as the Shepard at the same volume". The idiom of the fix is
# the codebase's own: a threshold is derived from the person, the level's range is measured from the
# session, and the loudness of this voice now is too. The learned pair is printed in the player's state
# line, which is also the answer to "how many microvolts is reasonable" — the screen now knows.
EEG_LOUD_TRACK_SECONDS = 5.0
"""During the first `EEG_LOUD_CAL_SECONDS`, how fast both ends of the amplitude range follow the band —
symmetric, so the range converges onto whatever today's signal is doing within a few time constants."""

EEG_LOUD_CAL_SECONDS = 30.0
"""When calibration ends and the asymmetric regime below takes over. The thresholds' own window."""

EEG_LOUD_ATTACK_SECONDS = 3.0
"""After calibration, how fast an end moves *outward* — a genuinely louder rhythm reaches full scale at
once (the mapping clips at 1 while the end catches up) and the range then owns it. Slow enough that a
sub-second artifact moves the top only fractionally toward its spike."""

EEG_LOUD_RELEASE_SECONDS = 300.0
"""And how slowly an end lets go inward. Not held rigid like a threshold, deliberately: the sonifier is
fed every sample *including what the artifact gate rejected*, so a single flailing electrode would
otherwise own the top of the range for the rest of the hour. Five minutes forgives that without being a
compressor — loudness still means something across any stretch a person can hear as one."""

EEG_LOUD_SPAN_FLOOR = 0.1
"""The narrowest believable range, as a fraction of its top. A perfectly steady amplitude would otherwise
collapse the span to nothing and turn the loudness into noise around a divide-by-almost-zero — the same
argument as `LevelRange.FLOOR`, in microvolts."""


def eeg_pitch(f_hz, band: Band, pitch_lo: float, pitch_hi: float):
    """Map a frequency in the band onto the pitch range **by ratio**, not by offset.

    Log-to-log, which is the whole point of the instrument: equal intervals in the EEG come out as equal
    intervals in what is heard, so a rhythm speeding from 9 to 10 Hz is the same musical step wherever in
    the band it happens. Accepts an array or a scalar and clamps to the band's ends, since the frequency
    estimate can wander outside a band it was band-limited to at the moment the band is quiet.
    """
    lo, hi = float(band.low), float(band.high)
    f = np.clip(f_hz, lo, hi)
    if hi <= lo or pitch_lo <= 0 or pitch_hi <= 0:
        return np.full_like(np.asarray(f, dtype=float), pitch_lo)
    frac = np.log(f / lo) / np.log(hi / lo)
    return pitch_lo * (pitch_hi / pitch_lo) ** frac


def stretch_factor(band: Band, pitch_lo: float, pitch_hi: float) -> float:
    """How much the band's own intervals are stretched by this pair of ends.

    1.0 means an exact transposition — the band and the pitch range span the same number of octaves, so
    the sound is the EEG played at a fixed multiple of its speed and every interval in it is intact. Above
    1 the intervals are exaggerated, which is legibility bought with honesty: 2 makes a semitone of EEG
    sound like a whole tone. It is reported on the panel rather than being a control, because the ends are
    what an operator actually chooses.
    """
    if band.high <= band.low or pitch_hi <= pitch_lo or pitch_lo <= 0:
        return 1.0
    return float(np.log(pitch_hi / pitch_lo) / np.log(band.high / band.low))


class Sonifier(_AudioStream):
    """The EEG's band, resynthesised as one moving voice: its frequency is the pitch, its amplitude the
    loudness.

    The session pushes samples in and the feed thread does the analysis: one pass of the cFIR turns a block
    of microvolts into a frequency and an amplitude per sample, which go into two rings. The writer thread
    pulls them out, holding each for as many output frames as the rate ratio calls for, and integrates the
    phase. Sample-and-hold rather than interpolation because at a 192:1 ratio the stair-steps are far above
    the band being listened to and inaudible — and because the smoothing that matters has already been
    applied to the frequency itself, at the rate it was measured.

    **This crackled on a laptop for three rounds of fixes, and the last of them is the one that mattered:
    it was the only player here still running Python on the sound card's thread.** `BUFFER_LATENCY` is the
    argument, written when the music streams were moved off callbacks — a `sounddevice` callback runs on
    CoreAudio's real-time thread and must take the GIL to enter Python, so *any* other thread holding the
    GIL past the device's deadline is a dropout: the draw loop rebuilding canvases, the feed thread
    filtering EEG, a garbage collection. The sonifier kept its callback on the argument that live EEG
    should not sit behind a deep output buffer. That argument is about the *ring's* read cursor, which
    bounds how far behind the EEG the render is and is a different quantity from the output buffer's
    depth — so it never justified the callback, and the callback is gone.

    The two rounds before it were real defects and both are kept; neither was the crackle, and the honest
    record is that each was found by reading and asserted to be the cause without being measured:

    - **The callback took a lock the feed thread held across a per-sample Python loop.** `feed` walked its
      block one element at a time inside `self._lock`, and the audio thread waited on that lock while the
      GIL was somewhere else. There is one writer and one reader, so there is no need for a lock at all:
      the ring is written first and the write cursor published afterwards, and the reader takes one
      snapshot of it. The writes are vectorised, so the producer holds the GIL for two slice assignments
      instead of a thousand iterations.
    - **An underrun cut the output to zero**, which breaks the rule that a sound stopping before
      it reaches zero is a crackle. The voice now runs through a dropout at the pitch and loudness it was
      left at, so a late block is a held note rather than a hole, and every seam is crossfaded over
      `SEAM_MS`. On a continuous seam the crossfade is arithmetically a no-op, so it costs nothing to apply
      always rather than detecting the cases.

    Measured on this recording at 250 Hz, with real EEG and realistic block timing — 8-sample device blocks
    against 2048-frame audio blocks, nothing synchronised — the render was clean: no callback ran dry, none
    ran out mid-block, and the largest sample-to-sample step was 5× a carrier cycle's own. Which is the
    evidence that the render was never what anybody was hearing. Those figures are the heterodyne's; the
    ring, the cursors and the seams they exercised are the same code, and the voice replaced only what is
    read out of the rings.
    """

    stream_name = "sonifier"

    SEAM_MS = 5.0
    """Crossfade across a discontinuity in pitch or loudness. Long enough that a full-scale step is
    inaudible, far shorter than a cycle of anything in the band being listened to, so nothing the signal
    does is smoothed by it."""

    def __init__(self) -> None:
        self.rate_hz = 0.0
        self.band = Band(2.0, 38.0, "music")
        self.pitch_lo_hz = EEG_PITCH_LO_HZ
        self.pitch_hi_hz = EEG_PITCH_HI_HZ
        self.volume = 60.0
        self.loud_lo = 0.0                  # the band's learned quiet, µV — see EEG_LOUD_TRACK_SECONDS
        self.loud_hi = 0.0                  # and its learned loud; 0.0 until the first block arrives
        self._fed_seconds = 0.0             # how much signal the range has learned from
        self.eeg_hz = 0.0                   # what the band is doing, for the panel to say
        self.pitch_hz = 0.0                 # and where that put the voice
        self._filter = None                 # the Butterworth that goes *before* the tracker; see `feed`
        self._tracker: CFIREnvelope | None = None
        self._ring = np.zeros(1)            # amplitude, µV
        self._ring_f = np.zeros(1)          # instantaneous frequency, Hz, already smoothed
        self._write = 0
        self._read = 0.0
        self._last_angle = 0.0          # the tracker's phase, for the next block's frequency estimate
        self._smooth_zi = np.zeros(1)   # the one-pole's state, so a block boundary is not a step
        self._phase = 0.0
        self._last_amp = 0.0            # what the voice was left at, so a dropout holds instead of cuts
        self._last_pitch = EEG_PITCH_LO_HZ
        self._primed = False            # whether anything has ever arrived; before that, silence
        self._init_stream()
        # `dropouts` counts one more thing here than it does for the other players: a block that found no
        # fresh EEG waiting, which is the same crackle arriving from the other side. See `_block`.

    def configure(self, rate_hz: float, band: Band) -> None:
        """(Re)build the tracker when the channel rate or the band changes."""
        if rate_hz <= 0:
            return
        if self._tracker is not None and self.rate_hz == rate_hz and self.band == band:
            return
        self.rate_hz = rate_hz
        self.band = band
        self._ring = np.zeros(max(64, int(RING_SECONDS * rate_hz)))
        self._ring_f = np.full(len(self._ring), band.centre)
        self._write = 0
        self._read = 0.0
        self._primed = False
        self.loud_lo = self.loud_hi = 0.0   # a new band or rate is a new person as far as loudness knows
        self._fed_seconds = 0.0
        self._last_angle = 0.0
        # The one-pole starts *at* the band centre rather than at zero, so the opening block glides down
        # from a plausible pitch instead of up from below the bottom of the range.
        self._smooth_zi = np.array([band.centre * self._smooth_pole(rate_hz)])
        try:
            self._filter = bandpass(band, rate_hz)
            taps = int(round(EEG_TRACK_TAPS_SECONDS * rate_hz))
            self._tracker = CFIREnvelope(band, rate_hz, delay_seconds=EEG_TRACK_DELAY_SECONDS,
                                         n_taps=max(64, taps))
        except ValueError as exc:
            self._filter = self._tracker = None
            self.error = str(exc)      # e.g. the band reaches Nyquist on this channel

    @staticmethod
    def _smooth_pole(rate_hz: float) -> float:
        """The one-pole coefficient for `EEG_FREQ_SMOOTH_SECONDS` at this rate."""
        return float(np.exp(-1.0 / max(1e-9, EEG_FREQ_SMOOTH_SECONDS * rate_hz)))

    def feed(self, values: np.ndarray) -> None:
        """Track a block of raw µV — amplitude and frequency — and add both to the rings.

        **Band-passed first, then tracked, and the second filter does not make the first one redundant.**
        The cFIR is a least-squares fit over the whole spectrum, so its stopband rejection is modest, and
        the raw signal off this amplifier carries a DC offset of a couple of hundred microvolts with 43% of
        its power below 4 Hz. Measured on a recording: fed raw, the analytic estimate is dominated by that
        leak, which is a nearly *stationary* complex offset — so the phase stops advancing and the tracked
        frequency reads 0.06 Hz, pinning the pitch at the bottom of the range for a whole session. Through
        the Butterworth first, the same minute reads a median of 10.9 Hz with a sane spread, and the band's
        amplitude reads 3.5 µV rather than the 43 µV the drift was contributing.

        The frequency is then the phase advance of the analytic signal, wrapped into ±π so a cycle boundary
        is not read as a leap, converted to Hz and smoothed by a one-pole carrying its state across blocks.
        Clamping to the band happens where the pitch is computed rather than here, so the panel can still
        report an estimate that has wandered out of it — which is what a quiet band looks like from inside.

        **Two slice assignments per ring, and the cursor published last.** This walked the block a sample
        at a time holding a lock the audio callback also wanted, which is a stall of the device for as long
        as the producer needs the GIL. Writing the samples before advancing `_write` is what makes the lock
        unnecessary: the reader only ever looks at indices below the cursor it read, so it cannot see a
        half-written block. One writer, one reader, and an integer store the GIL makes atomic.
        """
        if self._tracker is None or self._filter is None or not len(values):
            return
        filtered = self._filter.process(np.asarray(values, dtype=float))
        analytic = self._tracker.analytic(filtered)
        amplitude = np.abs(analytic)
        self._learn_range(amplitude)
        angles = np.angle(analytic)
        steps = np.diff(np.concatenate(([self._last_angle], angles)))
        self._last_angle = float(angles[-1])
        steps = (steps + np.pi) % (2 * np.pi) - np.pi
        raw_hz = steps * self.rate_hz / (2 * np.pi)
        pole = self._smooth_pole(self.rate_hz)
        smoothed, self._smooth_zi = lfilter([1.0 - pole], [1.0, -pole], raw_hz, zi=self._smooth_zi)

        ring, ring_f = self._ring, self._ring_f
        size = len(ring)
        if not size:
            return
        # A block longer than the ring: only the newest samples can survive it, but the cursor still
        # advances by everything fed. It is a monotonic count of samples, not an index — the reader takes
        # differences of it to decide how far behind it is, so under-counting here would read as the feed
        # thread having stalled.
        total = len(amplitude)
        n = min(total, size)
        start = (self._write + total - n) % size
        for target, source in ((ring, amplitude), (ring_f, smoothed)):
            tail = source[-n:]
            if start + n <= size:
                target[start:start + n] = tail
            else:
                cut = size - start
                target[start:] = tail[:cut]
                target[:n - cut] = tail[cut:]
        self._write += total                # published last: everything below it is written

    def _learn_range(self, amplitude: np.ndarray) -> None:
        """Follow the band's own quiet and loud, so the loudness needs no setting. See the constants.

        Two one-pole followers on each block's extremes: symmetric and quick while calibrating, then
        asymmetric — outward at `EEG_LOUD_ATTACK_SECONDS`, inward at `EEG_LOUD_RELEASE_SECONDS`. Runs on
        the feed thread; the writer reads `loud_lo`/`loud_hi` as two plain float loads, which is the same
        one-writer-one-reader bargain the rings already live by.
        """
        dt = len(amplitude) / self.rate_hz
        m_lo, m_hi = float(amplitude.min()), float(amplitude.max())
        if self.loud_hi <= 0.0:
            self.loud_lo, self.loud_hi = m_lo, m_hi
        else:
            calibrating = self._fed_seconds < EEG_LOUD_CAL_SECONDS
            out_gain = 1.0 - np.exp(-dt / (EEG_LOUD_TRACK_SECONDS if calibrating
                                           else EEG_LOUD_ATTACK_SECONDS))
            in_gain = 1.0 - np.exp(-dt / (EEG_LOUD_TRACK_SECONDS if calibrating
                                          else EEG_LOUD_RELEASE_SECONDS))
            self.loud_hi += (out_gain if m_hi > self.loud_hi else in_gain) * (m_hi - self.loud_hi)
            self.loud_lo += (out_gain if m_lo < self.loud_lo else in_gain) * (m_lo - self.loud_lo)
        self._fed_seconds += dt

    def start(self) -> bool:
        if self._filter is None or self.rate_hz <= 0:
            return False
        return super().start()

    def _block(self, frames: int) -> np.ndarray:
        """One block of voice: the band's frequency as pitch, its amplitude as loudness."""
        step = self.rate_hz / SAMPLE_RATE            # EEG samples consumed per output frame
        gain = min(100.0, max(0.0, self.volume)) / 100.0
        # One snapshot each, and no lock — see the class docstring. `_write` is read once so the arithmetic
        # below is consistent even if the feed thread advances it mid-callback; a sample that arrives during
        # this block is simply heard in the next one.
        ring, ring_f, write, read = self._ring, self._ring_f, self._write, self._read
        size = len(ring)

        available = write - read
        if read > write:
            # `configure` restarted the cursors under us. Re-anchor rather than index backwards.
            self._read = read = float(write)
            available = 0
        if available < 1 or size < 1:
            if not self._primed:
                # **Nothing has ever arrived, so there is nothing to hold.** Silence is right here and
                # only here: no sound has started, so no sound is being interrupted. Holding a never-set
                # amplitude would start the voice at its quiet floor — audible, and indistinguishable from
                # a channel delivering a flat line.
                return np.zeros((frames, 1), dtype=np.float32)
            # **Held, not cut.** Once the sound has started, silence is a step to zero and back a few
            # milliseconds later, which is a click at both ends — the fourth instance of the
            # rule that a sound stopping before it reaches zero crackles. A late block now sounds like a
            # note held slightly too long.
            amplitude = np.full(frames, self._last_amp)
            pitches = np.full(frames, self._last_pitch)
            self.dropouts += 1
        else:
            if available > 0.5 * self.rate_hz:
                # Fell behind: skip to a fifth of a second behind the freshest sample rather than
                # drift forever. This used to re-anchor at half the ring — a full second of standing
                # lag on top of the output buffer, held for the rest of the session. 200 ms still
                # rides out the feed thread's 32 ms bursts with a wide margin, and it is feedback
                # delay, so every spare millisecond is worth having.
                read = write - 0.2 * self.rate_hz
            positions = read + step * np.arange(frames)
            positions = np.minimum(positions, write - 1)
            taken = (positions.astype(np.int64)) % size
            amplitude = ring[taken]
            pitches = eeg_pitch(ring_f[taken], self.band, self.pitch_lo_hz, self.pitch_hi_hz)
            self.eeg_hz = float(ring_f[taken[-1]])
            self._read = float(positions[-1] + step)
            self._primed = True

        # **Every seam crossfaded, whether or not there was one.** Recovering from a dropout and
        # re-anchoring after falling behind are both steps — in the loudness, and now in the pitch as well,
        # where a step is heard as a click and as a different note. Where they are already continuous the
        # ramp is arithmetically a no-op, so testing for the discontinuity would only be a way of getting
        # the test wrong.
        seam = min(frames, max(1, int(SAMPLE_RATE * self.SEAM_MS / 1000.0)))
        blend = np.linspace(0.0, 1.0, seam)
        amplitude = amplitude.astype(float, copy=True)
        pitches = pitches.astype(float, copy=True)
        amplitude[:seam] = self._last_amp * (1.0 - blend) + amplitude[:seam] * blend
        pitches[:seam] = self._last_pitch * (1.0 - blend) + pitches[:seam] * blend
        self._last_amp = float(amplitude[-1])
        self._last_pitch = float(pitches[-1])
        self.pitch_hz = self._last_pitch

        # The phase is integrated rather than stepped, because the pitch changes *within* a block: a
        # constant increment per block would quantise every glide to 43 ms and put a seam at each boundary.
        increment = 2 * np.pi * pitches / SAMPLE_RATE
        angles = self._phase + np.cumsum(increment)
        self._phase = float(angles[-1] % (2 * np.pi))

        # Loudness is where the band sits in its own learned range — see `_learn_range` and the constants
        # for why there is no setting — floored rather than taken to silence (`EEG_QUIET_GAIN`), and on a
        # square root rather than straight. The root is the calibration against the other sources: the
        # band spends most of a session near the middle of its range, and mapped straight that is half
        # amplitude, a voice sitting 6 dB under the Shepard at the same volume number — which is the
        # unmarked-per-source loudness change `TONE_PEAK` and `SHEPARD_PEAK` exist to prevent, arriving by
        # another door. The root lifts the middle to 0.71 and moves neither end, so quiet is still quiet
        # and a burst is still a burst. `TONE_PEAK` because this is the same instrument as the `tone`
        # source — one sine — and `Music.volume` is shared across sources.
        lo, hi = self.loud_lo, self.loud_hi
        span = max(hi - lo, EEG_LOUD_SPAN_FLOOR * hi, 1e-12)
        frac = np.clip((amplitude - lo) / span, 0.0, 1.0)
        loud = EEG_QUIET_GAIN + (1.0 - EEG_QUIET_GAIN) * np.sqrt(frac)
        out = np.clip(loud * np.sin(angles) * gain * TONE_PEAK, -1.0, 1.0)
        return out.astype(np.float32).reshape(-1, 1)

class MusicOutput:
    """One object owning both paths, with the same take/follow/release contract as the shadow."""

    def __init__(self) -> None:
        self.player = TrackPlayer()
        self.sonifier = Sonifier()
        self.noise = NoisePlayer()
        self.shepard = ShepardPlayer()
        self.tone = TonePlayer()

    def apply(self, settings, rate_hz: float = 0.0,
              hold: bool = False, balance: float | None = None,
              level: float = 0.0) -> None:
        """Called on every level update. `settings` is a `feedback.Music`.

        `hold` pauses playback in place — the pause-on-violation model. The sonifier is not held, and
        that is deliberate: it *is* the EEG, so stopping it would replace a signal with a silence that
        looks identical to a lost channel.

        `balance` is where the crossfade sits, 0…1, supplied by the protocol because only a protocol knows
        what the two ends of it *are*. None leaves the music at whichever end it was.

        `level` is the feedback signal itself, and only the Shepard tone reads it. **It is a separate
        argument from `balance` on purpose**: a crossfade follows a *state* the protocol names and a Shepard
        tone follows the *score*, so a screen with both would drive them from two different quantities. That
        distinction is `Shadow.follows` one level up, and collapsing it here would undo it.
        """
        if not settings.enabled:
            self.release()
            return
        if settings.source == "shepard":
            self.player.stop()
            self.sonifier.stop()
            self.noise.stop()
            self.tone.stop()
            self.shepard.configure(settings.volume, level, settings.glide_seconds,
                                   settings.travel_octaves)
            self.shepard.start()
            self.shepard.held = hold
            return
        self.shepard.stop()
        if settings.source == "tone":
            self.player.stop()
            self.sonifier.stop()
            self.noise.stop()
            self.tone.configure(settings.volume, level, base_hz=settings.tone_base_hz,
                                range_hz=max(10.0, settings.tone_top_hz - settings.tone_base_hz))
            self.tone.start()
            self.tone.held = hold
            return
        self.tone.stop()
        if settings.source in ("noise", "crossfade"):
            self.player.stop()
            self.sonifier.stop()
            second = settings.second_colour if settings.source == "crossfade" else None
            self.noise.configure(settings.colour, settings.volume, second_colour=second,
                                 balance=self.noise.balance if balance is None else balance,
                                 pitch_hz=settings.pitch_hz,
                                 second_pitch_hz=(settings.second_pitch_hz if second else
                                                  settings.pitch_hz))
            self.noise.start()
            self.noise.held = hold
            return
        self.noise.stop()
        if settings.source == "eeg":
            self.player.stop()
            self.sonifier.configure(rate_hz, Band(settings.low_hz, settings.high_hz, "music"))
            self.sonifier.pitch_lo_hz = settings.eeg_pitch_lo_hz
            self.sonifier.pitch_hi_hz = settings.eeg_pitch_hi_hz
            self.sonifier.volume = settings.volume
            # No loudness scale is passed, because there is none to pass: the voice is heard against the
            # band's own learned range — see `Sonifier._learn_range` for what replaced `full_at_uv`.
            self.sonifier.start()
            return
        self.sonifier.stop()
        self.player.start(tracks(settings.path), settings.volume)
        self.player.hold(hold)

    def feed(self, values) -> None:
        if self.sonifier.running:
            self.sonifier.feed(values)

    def release(self) -> None:
        self.player.stop()
        self.sonifier.stop()
        self.noise.stop()
        self.shepard.stop()
        self.tone.stop()

    @staticmethod
    def _glitches(player) -> str:
        """What a crackle looks like from inside, appended to whatever the player is doing.

        **Every player has counted this since it was written and not one of them showed it.** A crackle was
        therefore diagnosed by ear and by theory — three separate rounds of it — when the program already
        knew the answer and had nowhere to put it. Same lesson as `NFB_DEBUG`: if a live
        symptom cannot be reproduced in a test, make the program say what it sees.

        Silent at zero, because a counter that reads "0 dropouts" on every screen forever is noise.
        """
        said = []
        for n, what in ((getattr(player, "dropouts", 0), "dropout"),
                        (getattr(player, "starved", 0), "starved block"),
                        (getattr(player, "restarts", 0), "restart")):
            if n:
                said.append(f"{n} {what}{'s' if n > 1 else ''}")
        return f" · {', '.join(said)}" if said else ""

    def describe(self, settings) -> str:
        if not settings.enabled:
            return "off"
        if settings.source == "tone":
            if self.tone.error:
                return self.tone.error
            if not self.tone.running:
                return "not started"
            if self.tone.held:
                return "silent — a target is failing"
            return (f"{self.tone.pitch_hz:.0f} Hz of "
                    f"{settings.tone_base_hz:g}–{settings.tone_top_hz:g}"
                    f"{self._glitches(self.tone)}")
        if settings.source == "shepard":
            if self.shepard.error:
                return self.shepard.error
            if not self.shepard.running:
                return "not started"
            if self.shepard.held:
                return "silent — a target is failing"
            freqs, amps = self.shepard.partials(self.shepard.octaves)
            loudest = float(freqs[int(np.argmax(amps))])
            return (f"{SHEPARD_SUBTONES} partials · {self.shepard.octaves:.2f} of "
                    f"{self.shepard.travel_octaves:g} octaves up · loudest near {loudest:.0f} Hz"
                    f"{self._glitches(self.shepard)}")
        if settings.source in ("noise", "crossfade"):
            if self.noise.error:
                return self.noise.error
            if not self.noise.running:
                return "not started"
            state = "silent — a target is failing" if self.noise.held else "playing"
            glitches = self._glitches(self.noise)
            if settings.source == "crossfade":
                return (f"{settings.colour} to {settings.second_colour} noise · "
                        f"{self.noise.balance:.0%} across · {state}{glitches}")
            return f"{settings.colour} noise · {state}{glitches}"
        if settings.source == "eeg":
            if self.sonifier.error:
                return self.sonifier.error
            if not self.sonifier.running:
                return "waiting for a channel"
            # The live pair, because the whole claim of this instrument is that one is the other: an
            # operator who cannot see the EEG frequency next to the pitch cannot tell a tracker that has
            # locked onto the rhythm from one wandering in a quiet band.
            band = Band(settings.low_hz, settings.high_hz, "music")
            stretch = stretch_factor(band, settings.eeg_pitch_lo_hz, settings.eeg_pitch_hi_hz)
            shape = ("an exact transposition" if abs(stretch - 1.0) < 0.05 else
                     f"intervals ×{stretch:.1f}")
            # The learned loudness range, printed because it answers the question the removed `full at`
            # box used to ask the operator: what does this band actually do, in microvolts. Nobody knows
            # that number; the tracker does.
            loud = (f" · quiet {self.sonifier.loud_lo:.1f}, full {self.sonifier.loud_hi:.1f} µV, learned"
                    if self.sonifier.loud_hi > 0 else "")
            return (f"{self.sonifier.eeg_hz:.1f} Hz sounding at {self.sonifier.pitch_hz:.0f} Hz of "
                    f"{settings.eeg_pitch_lo_hz:g}–{settings.eeg_pitch_hi_hz:g} · {shape}{loud}"
                    f"{self._glitches(self.sonifier)}")
        found = tracks(settings.path)
        if not found:
            return "no audio files there" if settings.path else "no file or folder chosen"
        playing = self.player.now_playing
        if not playing:
            return f"{len(found)} track(s) ready"
        state = "paused — a target is failing" if self.player.held else "playing"
        return f"{playing} · {len(found)} track(s) · {state}"
