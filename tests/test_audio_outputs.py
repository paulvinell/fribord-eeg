"""Sound as the system volume, and music as a track, a folder, or the EEG.

Same shape of risk as the shadow: these hold something that belongs to the operator's machine, so the
floor and the restore are tested rather than trusted. The real CoreAudio and the real audio device are
faked throughout, so the suite is silent and leaves the volume alone.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import waitfor

from nfb_console import music, sysvolume
from nfb_console.feedback import Music


# --- the system volume -------------------------------------------------------------------------

@pytest.fixture
def fake_volume(monkeypatch):
    # The clock is part of the fake: the approach smooths in seconds of wall clock, exactly as the
    # shadow does, so a test driving `apply` in a tight loop passes no time and nothing would move.
    state = {"level": 0.5, "writes": [], "t": [0.0]}

    def fake_write(level, device=None):
        level = min(1.0, max(0.0, level))       # the real _write clamps only to [0, 1]; set() floors on top
        state["level"] = level
        state["writes"].append(round(level, 4))

    monkeypatch.setattr(sysvolume, "available", lambda: True)
    monkeypatch.setattr(sysvolume, "problem", lambda: None)
    monkeypatch.setattr(sysvolume, "_device", lambda: 1)
    monkeypatch.setattr(sysvolume, "get", lambda device=None: state["level"])
    monkeypatch.setattr(sysvolume, "_write", fake_write)    # the write boundary; real set() floors into it
    monkeypatch.setattr(sysvolume, "clock", lambda: state["t"][0])
    return state


def settle(control, state, level, low=30, high=90, steps=60):
    for _ in range(steps):
        state["t"][0] += 0.05                # 20 Hz of fake wall clock
        control.apply(True, low, high, level)


def test_the_volume_floor_is_enforced_below_any_setting(fake_volume):
    """Feedback that reaches silence stops being feedback, and the operator can no longer hear the
    tone that says the session ended."""
    control = sysvolume.SystemVolume()
    settle(control, fake_volume, level=0.0, low=0, high=0)
    assert fake_volume["level"] == pytest.approx(sysvolume.HARD_FLOOR)
    assert min(fake_volume["writes"]) >= sysvolume.HARD_FLOOR


def test_the_volume_follows_the_level_inside_its_range(fake_volume):
    control = sysvolume.SystemVolume()
    settle(control, fake_volume, level=0.0)
    assert fake_volume["level"] == pytest.approx(0.30, abs=0.01)
    settle(control, fake_volume, level=1.0)
    assert fake_volume["level"] == pytest.approx(0.90, abs=0.01)
    settle(control, fake_volume, level=0.5)
    assert fake_volume["level"] == pytest.approx(0.60, abs=0.01)


def test_the_operators_volume_is_restored(fake_volume):
    control = sysvolume.SystemVolume()
    settle(control, fake_volume, level=0.0, low=10, high=100)
    assert fake_volume["level"] < 0.3, "it really did turn down"
    control.release()
    assert fake_volume["level"] == 0.5
    assert not control.held


def test_restore_gives_back_a_low_level_but_floors_a_true_mute(fake_volume):
    """Restore routed through the floored set(), so a muted (0.0) start came back at 5% and a low level was
    pushed up to the floor. Restore now writes what it found, flooring only a true mute to keep the
    'cannot leave it silent' rail."""
    fake_volume["level"] = 0.0                 # the operator started muted
    control = sysvolume.SystemVolume()
    settle(control, fake_volume, level=1.0, low=10, high=100)
    control.release()
    assert fake_volume["level"] == pytest.approx(sysvolume.HARD_FLOOR), "a true mute is floored, not left at 0"

    fake_volume["level"] = 0.03                # a low-but-audible start, below the floor
    control = sysvolume.SystemVolume()
    settle(control, fake_volume, level=1.0, low=10, high=100)
    control.release()
    assert fake_volume["level"] == pytest.approx(0.03), "a low-but-audible level comes back as found"


def test_restore_targets_the_device_it_took_not_the_new_default(monkeypatch):
    """The controller saved only the level, so after an output switch it restored to the new default while
    the original kept the training volume. It now captures the device at take and restores to that one."""
    levels = {1: 0.5, 2: 0.9}
    current = {"id": 1}
    monkeypatch.setattr(sysvolume, "available", lambda: True)
    monkeypatch.setattr(sysvolume, "_device", lambda: current["id"])
    monkeypatch.setattr(sysvolume, "get", lambda device=None: levels[current["id"] if device is None else device])
    monkeypatch.setattr(sysvolume, "_write",
                        lambda level, device=None: levels.__setitem__(current["id"] if device is None else device,
                                                                      min(1.0, max(0.0, level))))
    monkeypatch.setattr(sysvolume, "clock", lambda: 0.0)

    control = sysvolume.SystemVolume()
    control.apply(True, 10, 100, 1.0)          # takes device 1 at 0.5
    current["id"] = 2                          # the default output switches to device 2
    control.release()
    assert levels[1] == pytest.approx(0.5), "the device we took is restored to what it was"
    assert levels[2] == pytest.approx(0.9), "the new default is left untouched"


def test_resuming_after_a_pause_does_not_jump_the_volume(fake_volume):
    """release() left _last at the pre-pause time, so the first resumed update spanned the whole pause and
    jumped straight to target. _last is reset on re-take now, so the resume approaches like any step."""
    control = sysvolume.SystemVolume()
    settle(control, fake_volume, level=0.0, low=10, high=100)     # steady low
    control.release()
    fake_volume["t"][0] += 600.0                                   # a ten-minute pause
    control.apply(True, 10, 100, 1.0)                             # resume at a high target
    assert fake_volume["level"] < 0.9, "the first resumed step approaches, it does not jump to the target"


def test_switching_sound_off_restores_without_a_separate_call(fake_volume):
    control = sysvolume.SystemVolume()
    settle(control, fake_volume, level=0.0, low=10, high=100)
    control.apply(False, 10, 100, 0.0)
    assert fake_volume["level"] == 0.5 and not control.held


def test_the_volume_is_approached_not_jumped_to(fake_volume):
    control = sysvolume.SystemVolume()
    control.apply(True, 10, 100, 1.0)
    fake_volume["t"][0] += 0.05
    control.apply(True, 10, 100, 1.0)
    assert 0.5 < fake_volume["level"] < 1.0, "a step moves toward the target, not onto it"


def test_nothing_happens_when_the_platform_cannot_set_the_volume(monkeypatch):
    monkeypatch.setattr(sysvolume, "available", lambda: False)
    control = sysvolume.SystemVolume()
    control.apply(True, 10, 100, 1.0)
    assert not control.held


# --- music: tracks ----------------------------------------------------------------------------

def tone(path: Path, freq: float = 330.0, seconds: float = 0.2) -> Path:
    rate = 22050
    times = np.arange(int(rate * seconds)) / rate
    data = (0.2 * np.sin(2 * np.pi * freq * times) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(data.tobytes())
    return path


def test_a_file_and_a_folder_both_resolve_to_a_track_list(tmp_path):
    first = tone(tmp_path / "a.wav")
    tone(tmp_path / "b.wav", 440)
    (tmp_path / "notes.txt").write_text("not audio")

    assert music.tracks(first) == [first]
    assert [p.name for p in music.tracks(tmp_path)] == ["a.wav", "b.wav"], "sorted, and audio only"
    assert music.tracks(tmp_path / "nope") == []
    assert music.tracks(tmp_path / "notes.txt") == [], "a text file is not a track"


def test_the_file_player_stays_silent_under_the_output_guard(tmp_path, monkeypatch):
    """NFB_NO_OUTPUTS silences the process, and afplay is a real output like the generated streams. The
    player launched it anyway. conftest sets the flag for the whole suite, so muted() is True here."""
    launched: list = []
    monkeypatch.setattr(music.subprocess, "Popen", lambda *a, **k: launched.append(a))
    track = tmp_path / "t.wav"
    track.write_bytes(b"\x00")
    player = music.TrackPlayer()
    player.start([track], 80.0)             # returns at once under the guard; no worker thread, no afplay
    try:
        assert not player.playing, "the player does not start under NFB_NO_OUTPUTS"
        assert launched == [], "and no afplay is launched"
    finally:
        player.stop()


def test_the_player_reports_what_it_would_run_and_stops_cleanly(tmp_path, monkeypatch):
    """The subprocess is faked: the point is the track order, the volume argument and the shutdown."""
    monkeypatch.setattr(music, "muted", lambda: False)  # faked subprocess, so run the player
    launched: list[list[str]] = []

    class FakeProc:
        """Implements the whole interface the player uses, signals included — an incomplete fake made
        `stop()` raise from a shutdown path, intermittently, depending on whether the child was still
        set when it ran."""

        def __init__(self, argv, **kw):
            launched.append(argv)
            self.terminated = False
            self.signals: list[int] = []

        def wait(self):
            return 0

        def poll(self):
            return None

        def send_signal(self, sig):
            self.signals.append(sig)

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(music.subprocess, "Popen", FakeProc)
    tone(tmp_path / "a.wav")
    tone(tmp_path / "b.wav", 440)

    player = music.TrackPlayer()
    player.start(music.tracks(tmp_path), volume=35)
    for _ in range(200):
        if len(launched) >= 3:
            break
        time.sleep(0.005)
    player.stop()

    assert launched, "it has to actually launch something"
    assert launched[0][:3] == ["afplay", "-v", "0.350"], "its own gain, not the system volume"
    names = [Path(argv[-1]).name for argv in launched]
    assert names[0] == "a.wav" and "b.wav" in names, "in order"
    assert names.count("a.wav") >= 2 or len(names) >= 3, "and looping round"
    assert not player.playing


# --- music: the EEG ---------------------------------------------------------------------------

def test_drift_does_not_pin_the_pitch_at_the_bottom_of_the_range():
    """The Butterworth in front of the tracker, which looks redundant and is not.

    A complex FIR is a least-squares fit over the whole spectrum, so its stopband is modest, and this
    amplifier's raw signal carries a couple of hundred microvolts of offset and drift. Fed raw, that leak
    dominates the analytic estimate — and it is nearly *stationary*, so the phase stops advancing and the
    tracked frequency reads near zero: a session's worth of the lowest note in the range while the EEG
    does whatever it likes. Measured on `sessions/latest`, 0.06 Hz raw against a 10.9 Hz median through
    the filter.
    """
    rate = 250.0
    times = np.arange(int(rate * 12)) / rate
    signal = 8.0 * np.sin(2 * np.pi * 11.0 * times) + 250.0 + 60.0 * np.sin(2 * np.pi * 0.2 * times)

    son = music.Sonifier()
    son.configure(rate, music.Band(8.0, 16.0, "music"))
    son.pitch_lo_hz, son.pitch_hi_hz = 220.0, 440.0
    son.volume = 100.0
    for i in range(0, len(signal) - 7, 8):
        son.feed(signal[i:i + 8])
        son._block(1500)
    assert abs(son.eeg_hz - 11.0) < 0.5, f"tracked {son.eeg_hz:.2f} Hz through the drift"
    assert son.pitch_hz > 240.0, "and it is not sitting on the bottom of the pitch range"


def test_a_band_the_channel_cannot_carry_is_reported_not_crashed():
    son = music.Sonifier()
    son.configure(62.5, music.Band(2.0, 38.0, "music"))     # 38 Hz is past this Nyquist
    assert son.error
    son.feed(np.ones(10))                                   # must not raise
    assert not son.start(), "and it must refuse to open a stream it cannot fill"


def test_the_bands_own_frequency_becomes_the_pitch_by_ratio():
    """The instrument's whole claim, checked against the arithmetic rather than against itself.

    A band of 8–16 Hz played across 220–440 Hz is an exact transposition — the two span one octave each —
    so every EEG frequency has an answer known in advance: ×27.5. That the *ends* land right would be true
    of any monotonic mapping; that 10 Hz lands on 275 and not on 330 is what makes it a transposition and
    not a stretch, which is why the middle is the case worth asserting.
    """
    for eeg_hz, expected in ((8.0, 220.0), (10.0, 275.0), (12.0, 330.0), (16.0, 440.0)):
        son = music.Sonifier()
        son.configure(250.0, music.Band(8.0, 16.0, "music"))
        son.pitch_lo_hz, son.pitch_hi_hz = 220.0, 440.0
        son.volume = 100.0
        # Fed and rendered interleaved at live pacing, for the reason the rhythm test below gives.
        wave = 20 * np.sin(2 * np.pi * eeg_hz * np.arange(250 * 12) / 250.0)
        rendered = []
        for i in range(0, len(wave) - 7, 8):
            son.feed(wave[i:i + 8])
            rendered.append(son._block(1500)[:, 0].astype(float).copy())
        audio = np.concatenate(rendered)[-(1 << 17):]

        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        freqs = np.fft.rfftfreq(len(audio), 1 / music.SAMPLE_RATE)
        heard = freqs[int(np.argmax(spectrum))]
        assert abs(heard - expected) < 6.0, f"{eeg_hz} Hz came out at {heard:.0f}, not {expected:.0f}"
        assert abs(son.eeg_hz - eeg_hz) < 0.3, "and the panel's figure is the one being played"


def test_the_ends_of_the_pitch_are_how_far_it_travels():
    """The row that replaced the carrier, and the reason it replaced it: a carrier could only be *added*
    to, so nothing on the panel changed how far the sound moved. Doubling the top end doubles the interval
    a given change in the EEG produces — the same 8→12 Hz is a fifth across 220–440 and an octave and a
    fifth across 220–880."""
    def pitch_at(eeg_hz: float, top_hz: float) -> float:
        son = music.Sonifier()
        son.configure(250.0, music.Band(8.0, 16.0, "music"))
        son.pitch_lo_hz, son.pitch_hi_hz = 220.0, top_hz
        son.volume = 100.0
        wave = 20 * np.sin(2 * np.pi * eeg_hz * np.arange(250 * 8) / 250.0)
        for i in range(0, len(wave) - 7, 8):
            son.feed(wave[i:i + 8])
            son._block(1500)
        return son.pitch_hz

    narrow = np.log2(pitch_at(12.0, 440.0) / pitch_at(8.0, 440.0))
    wide = np.log2(pitch_at(12.0, 880.0) / pitch_at(8.0, 880.0))
    assert abs(narrow - np.log2(1.5)) < 0.05, f"a fifth, measured {narrow:.2f} octaves"
    assert abs(wide - 2 * narrow) < 0.05, f"twice as far, measured {wide:.2f} against {narrow:.2f}"
    assert music.stretch_factor(music.Band(8.0, 16.0, "m"), 220.0, 440.0) == pytest.approx(1.0)
    assert music.stretch_factor(music.Band(8.0, 16.0, "m"), 220.0, 880.0) == pytest.approx(2.0)


def test_a_burst_arrives_as_loudness_while_it_is_still_happening():
    """What survives from the heterodyne, and the reason this instrument is worth having at all: the
    timing is real. A rhythm appearing has to be heard within the delay this path admits to — the
    Butterworth, the cFIR's design delay, the frequency smoothing and one audio block — not seconds later.

    Fed and rendered interleaved, at live pacing: 8 EEG samples per 1500 audio frames. Rendering seconds
    in one offline call is a shape live never takes — the render deliberately stays ~200 ms behind the
    freshest sample, so a giant call would exhaust the material and measure starvation instead.

    Loudness is relative since `_learn_range`: the quiet phase waxes and wanes so it earns a real range
    of its own — its crests sit near the top of that range, its troughs near the floor — and the burst
    then pins the voice at full for as long as it outruns what the range has learned. A *steady* quiet
    sine would sit at the top of its own tiny range too, which is not a defect but the design: every
    loudness here means "against what this band has been doing", the way every threshold does.
    """
    rate, eeg_hz = 250.0, 10.0
    t_quiet = np.arange(int(rate * 6)) / rate
    quiet = (0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * t_quiet)) * np.sin(2 * np.pi * eeg_hz * t_quiet)
    loud = 20.0 * np.sin(2 * np.pi * eeg_hz * np.arange(int(rate * 4)) / rate)
    wave = np.concatenate([quiet, loud])

    son = music.Sonifier()
    son.configure(rate, music.Band(8.0, 16.0, "music"))
    son.pitch_lo_hz, son.pitch_hi_hz = 220.0, 440.0
    son.volume = 100.0
    rendered, marks = [], []
    for i in range(0, len(wave) - 7, 8):
        son.feed(wave[i:i + 8])
        block = son._block(1500)[:, 0].astype(float)
        rendered.append(block.copy())
        marks.append((i / rate, np.abs(block).max()))          # EEG time of this block, its loudness

    peak = max(amp for _t, amp in marks)
    before = [amp for t, amp in marks if 3.0 < t < 5.5]
    after = [amp for t, amp in marks if 6.3 < t < 9.0]
    assert min(before) < 0.45 * peak, "the quiet phase's troughs are plainly quiet"
    assert np.mean(after) > 1.25 * np.mean(before), "the burst holds the voice at full"
    assert np.mean(after) > 0.92 * peak
    assert max(amp for t, amp in marks if 6.0 < t < 6.6) > 0.95 * peak, \
        "and it is heard within the delay this path admits to, not seconds later"

    # One clean tone: the zero crossings count the pitch and nothing else. The old modulator was signed,
    # which inverted the carrier's phase every half cycle and put a few hundred flips a second of buzz on
    # top; the loudness envelope here cannot do that because it never goes negative.
    audio = np.concatenate(rendered)[-(1 << 17):]
    crossings = np.sum(np.diff(np.signbit(audio)) != 0) / (len(audio) / music.SAMPLE_RATE) / 2
    assert abs(crossings - son.pitch_hz) < 20, f"{crossings:.0f} crossings/s against {son.pitch_hz:.0f} Hz"


def test_the_render_resumes_close_behind_the_live_signal():
    """Falling behind is re-anchored 200 ms back, not half the ring. The ring's two seconds are
    *capacity*; treating half of it as the working lag put a full standing second between the trainee's
    EEG and their ears, on top of the output buffer, for the rest of the session."""
    son = music.Sonifier()
    son.configure(256.0, music.Band(2.0, 38.0, "music"))
    son.feed(np.zeros(256 * 10))          # ten seconds pile up while the callback is not running
    out = np.zeros((256, 1), dtype=np.float32)
    son._block(256)
    behind = (son._write - son._read) / 256.0
    assert behind < 0.25, f"resumed {behind:.2f} s behind the live signal"


def test_an_empty_ring_outputs_silence_rather_than_noise():
    son = music.Sonifier()
    son.configure(256.0, music.Band(2.0, 38.0, "music"))
    assert not son._block(512).any(), "no data means silence, not whatever was in the buffer"


def test_music_off_releases_both_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(music.subprocess, "Popen",
                        lambda argv, **kw: type("P", (), {"wait": lambda self: 0,
                                                          "terminate": lambda self: None})())
    tone(tmp_path / "a.wav")
    out = music.MusicOutput()
    out.apply(Music(enabled=True, source="file", path=str(tmp_path)))
    out.apply(Music(enabled=False))
    assert not out.player.playing and not out.sonifier.running


def test_switching_source_stops_the_other_path(tmp_path, monkeypatch):
    monkeypatch.setattr(music.subprocess, "Popen",
                        lambda argv, **kw: type("P", (), {"wait": lambda self: 0,
                                                          "terminate": lambda self: None})())
    monkeypatch.setattr(music.Sonifier, "start", lambda self: True)
    tone(tmp_path / "a.wav")
    out = music.MusicOutput()
    out.apply(Music(enabled=True, source="file", path=str(tmp_path)))
    out.apply(Music(enabled=True, source="eeg"), rate_hz=256.0)
    assert not out.player.playing, "two sources at once would just be noise"
    out.release()


def test_pausing_keeps_the_position_rather_than_restarting_the_track(tmp_path, monkeypatch):
    """The pause-on-violation model. A backing track that restarted on every violation would be a much
    worse experience than one that stops mid-phrase and carries on, so this is SIGSTOP, not a relaunch."""
    monkeypatch.setattr(music, "muted", lambda: False)  # faked subprocess, so run the player
    import signal as sig

    launched: list[list[str]] = []
    procs: list[object] = []

    class FakeProc:
        def __init__(self, argv, **kw):
            launched.append(argv)
            self.signals: list[int] = []
            self.stopped = False
            procs.append(self)

        def wait(self):
            while not self.stopped:
                time.sleep(0.01)
            return 0

        def poll(self):
            return None

        def send_signal(self, s):
            self.signals.append(s)

        def terminate(self):
            self.stopped = True

    monkeypatch.setattr(music.subprocess, "Popen", FakeProc)
    tone(tmp_path / "a.wav")
    player = music.TrackPlayer()
    player.start(music.tracks(tmp_path), volume=50)
    for _ in range(200):
        if procs:
            break
        time.sleep(0.005)

    player.hold(True)
    assert player.held
    assert sig.SIGSTOP in procs[0].signals
    player.hold(True)
    assert procs[0].signals.count(sig.SIGSTOP) == 1, "idempotent: no signal storm from a held signal"
    player.hold(False)
    assert not player.held
    assert sig.SIGCONT in procs[0].signals
    assert len(launched) == 1, "one launch — pausing is not a relaunch"
    player.stop()


def test_the_pitch_source_tracks_the_level_between_its_two_frequencies():
    from nfb_console import music
    from nfb_console.feedback import Music

    out = music.MusicOutput()
    settings = Music(enabled=True, source="tone")
    try:
        for level in (0.0, 0.5, 1.0):
            out.apply(settings, level=level)
            out.tone._block(2048)
        # the glide means it approaches rather than jumps, so the check is direction and bounds
        assert settings.tone_base_hz <= out.tone.pitch_hz <= settings.tone_top_hz
        out.apply(settings, level=0.0)
        for _ in range(80):
            out.tone._block(2048)
        low = out.tone.pitch_hz
        out.apply(settings, level=1.0)
        for _ in range(80):
            out.tone._block(2048)
        assert out.tone.pitch_hz > low + 100.0, "the pitch has to follow the signal"
    finally:
        out.release()


# --- the noise ---------------------------------------------------------------------------

def test_the_noise_colours_are_level_matched_and_actually_tilted():
    """Measured rather than guessed, and measured **through the band limit**, since that is what reaches a
    speaker. Without the per-colour gain, switching colour changed the loudness sixfold and read as the
    control being broken; the gains were re-measured when the high-pass arrived, or it would have done so
    again."""
    from scipy import signal as sps

    rng = np.random.default_rng(5)
    white = rng.standard_normal(48000 * 4)
    energy = {}
    for name, (b, a, gain) in music.NOISE_FILTERS.items():
        y = sps.sosfilt(music.highpass_sos(), sps.lfilter(b, a, white) * gain)
        assert abs(y.std() - 1.0) < 0.1, f"{name} is not at unit RMS: {y.std():.3f}"
        # Gaussian noise has no bounded peak, so the question is how often it clips, not whether.
        clipped = float(np.mean(np.abs(y) * music.PEAK > 1.0))
        assert clipped < 1e-4, f"{name} clips {clipped:.2%} of samples at full volume"
        f, p = sps.welch(y, fs=48000, nperseg=8192)
        low = 10 * np.log10(p[(f >= 100) & (f < 200)].mean())
        high = 10 * np.log10(p[(f >= 3200) & (f < 6400)].mean())
        energy[name] = low - high

    assert abs(energy["white"]) < 1.0, "white is flat"
    # pink is -3 dB/octave, so five octaves of separation is about 15 dB
    assert 12 < energy["pink"] < 18, energy["pink"]
    assert energy["brown"] > energy["pink"] + 8, "brown falls faster still"


def test_the_music_is_band_limited_to_what_a_speaker_can_render():
    """**A 1/f tilt keeps rising toward DC and brown is a leaky integrator**, so both piled energy where
    nothing can reproduce it: 28% of each below 80 Hz, 8-11% below 20, plus brown's random walk wandering
    in DC offset. A laptop driver does not move air down there, it just travels — and what you hear is a
    cone at its excursion limit crackling, with intermodulation over it. Music chosen for having no
    structure was making a noise of its own.

    The tilt has to survive, which is why this checks both halves.
    """
    from scipy import signal as sps

    rng = np.random.default_rng(11)
    white = rng.standard_normal(48000 * 8)
    for name, (b, a, gain) in music.NOISE_FILTERS.items():
        shaped = sps.lfilter(b, a, white) * gain
        limited = sps.sosfilt(music.highpass_sos(), shaped)
        f, before = sps.welch(shaped, fs=music.SAMPLE_RATE, nperseg=8192)
        _f, after = sps.welch(limited, fs=music.SAMPLE_RATE, nperseg=8192)
        share = lambda p: p[f < 80].sum() / p.sum()
        assert share(after) < 0.08, f"{name} still puts {share(after):.0%} below 80 Hz"
        if name != "white":
            assert share(before) > 0.2, f"{name}: the measurement this exists for has changed"
        assert abs(limited.mean()) < 0.02, f"{name} carries a DC offset"
        # and it is still the same colour: the tilt above the corner is untouched
        band = lambda p, lo, hi: p[(f >= lo) & (f < hi)].sum()
        assert band(after, 150, 300) / band(after, 3000, 6000) == pytest.approx(
            band(before, 150, 300) / band(before, 3000, 6000), rel=0.15), f"{name} lost its tilt"


def test_a_dropout_is_counted_rather_than_ignored():
    """A dropout and a speaker at its excursion limit both sound like crackle, and the fix differs, so the
    difference has to be knowable rather than guessed. `write` reports an under-run per block."""
    class Stream:
        def __init__(self): self.written, self.starved = 0, True
        def write(self, block):
            self.written += 1
            return self.starved
        def stop(self): pass
        def close(self): pass

    player = music.NoisePlayer()
    assert player.dropouts == 0
    stream, stop = Stream(), threading.Event()
    stop.set()                               # one pass and out
    player._run(stream, stop)
    assert stream.written == 1, "a stopped music writes one block, the fade to silence, and no more"

    stop.clear()
    original = player._block
    calls = []

    def once(frames):
        calls.append(frames)
        if len(calls) >= 3:
            stop.set()
        return original(frames)

    player._block = once
    player._run(stream, stop)
    assert stream.written == 5 and player.dropouts == 3, "three blocks, then the fade, are written"
    stream.starved = False
    stop.clear()
    calls.clear()
    player._run(stream, stop)
    assert player.dropouts == 3, "and a block that arrived in time is not counted"


def test_holding_the_noise_silences_it_without_stopping_the_stream():
    """Restarting the stream would click, and a click when a target fails would be a second, unintended
    discrete signal. Cutting to zeros mid-block was the same click by another door — the hold now fades
    over one block and only then goes silent, mirroring the ramp the resume always had."""
    player = music.NoisePlayer()
    player.configure("pink", 80.0)
    assert np.abs(player._block(2048)).max() > 0.0
    player.held = True
    fade = player._block(2048)[:, 0]           # the ramp down: still sounding, ending in silence
    assert np.abs(fade).max() > 0.0 and abs(float(fade[-1])) < 1e-5
    assert np.abs(player._block(256)).max() == 0.0
    player.held = False
    player._block(2048)                        # the level ramps back up from silence
    assert np.abs(player._block(2048)).max() > 0.0


def test_changing_colour_restarts_the_filter_rather_than_inheriting_state():
    player = music.NoisePlayer()
    player.configure("pink", 60.0)
    player._block(256)
    assert player._zi is not None
    player.configure("brown", 60.0)
    assert player._zi is None, "another filter's state is not this filter's state"


def test_volume_scales_the_music_and_zero_is_silent():
    player = music.NoisePlayer()
    levels = []
    for volume in (100.0, 50.0, 0.0):
        player.configure("pink", volume)
        player._block(4096)                    # one block for the ramp to arrive, one to measure
        levels.append(float(np.abs(player._block(4096)).std()))
    assert levels[0] > levels[1] > 0
    assert levels[2] == 0.0


def test_the_sonifier_writes_from_a_thread_and_never_from_the_audio_thread(monkeypatch):
    """**No Python on the sound card's thread, which is the rule every other player here already follows.**

    A `sounddevice` callback runs on CoreAudio's real-time thread and has to take the GIL to enter Python,
    so any other thread holding it past the device's deadline is a dropout — the draw loop rebuilding
    canvases, the feed thread filtering EEG, a garbage collection. `BUFFER_LATENCY` is that argument,
    written when the music streams were moved off callbacks; the sonifier stayed behind on the grounds that
    live EEG should not sit behind a deep buffer, which is a statement about the *ring's* read cursor and
    not about the output buffer at all.

    Also the regression this test was originally written for: `start()` was once a copy of a player's body
    over a class with none of its attributes, so selecting "eeg" raised AttributeError inside the feed
    thread and took the whole session down. It now really is that pattern, and this runs it for real.
    """
    import sys

    opened = {}
    written = []

    class Stream:
        def __init__(self, **kw):
            opened.update(kw)

        def start(self):
            opened["started"] = True

        def write(self, block):
            written.append(block)
            return 0

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", type("SD", (), {"OutputStream": Stream}))
    monkeypatch.setattr(music, "muted", lambda: False)
    son = music.Sonifier()
    son.configure(256.0, music.Band(2.0, 38.0, "music"))
    assert son.start(), son.error
    assert opened.get("started")
    assert "callback" not in opened, "Python on the audio thread is the defect, not the design"
    waitfor(lambda: written, what="the writer thread to produce a block")
    son.stop()
    assert not son.running
    assert written[0].shape == (music.BLOCK_FRAMES, 1)


def test_a_late_block_holds_the_note_instead_of_cutting_to_silence():
    """**The fourth distinct crackle in this repo, and the same one every time**: the rule
    is that a sound which stops before it reaches zero is a crackle. A callback that found no fresh EEG
    dropped the output to zero and picked up again a few milliseconds later — a step down and a step up,
    one click each — and on a laptop that is exactly what an occasional stall sounds like.

    Silence is still right before anything has ever arrived, and only there: no sound has started, so no
    sound is being interrupted.
    """
    son = music.Sonifier()
    son.configure(256.0, music.Band(2.0, 38.0, "music"))

    assert not son._block(512).any(), "nothing has ever arrived: there is no note to hold"

    times = np.arange(1024) / 256.0
    son.feed(40.0 * np.sin(2 * np.pi * 10 * times))
    fed = son._block(512)
    assert np.abs(fed).max() > 0

    before = son.dropouts
    son._read = float(son._write)                   # the reader has caught up: the feed thread stalled
    starved = son._block(512)
    assert son.dropouts > before, "a dropout is counted, so the next report is diagnosable"
    assert np.abs(starved).max() > 0, "the carrier keeps sounding rather than cutting out"

    # The join between the last good block and the first starved one is where the click used to be: the
    # output fell to zero there and climbed back when data resumed. Asserted at the join itself rather
    # than over the whole pair, because the envelope's own sample-and-hold staircase is inside both blocks
    # and is a much bigger step than the seam — it is above the band and inaudible, and it is not what
    # this test is about.
    join = abs(float(starved[0, 0]) - float(fed[-1, 0]))
    within = np.abs(np.diff(starved[:, 0]))
    assert join <= within.max() + 1e-6, \
        f"the join steps by {join:.3f}, more than the carrier does on its own — that is the click"


def test_the_ring_needs_no_lock_between_the_feed_thread_and_the_audio_thread():
    """The write cursor counts every sample fed, across the wrap and past a block longer than the ring, so
    the reader can tell from it alone how far behind it is."""
    son = music.Sonifier()
    son.configure(256.0, music.Band(2.0, 38.0, "music"))

    ring_size = len(son._ring)
    son.feed(np.ones(ring_size * 3))
    assert son._write == ring_size * 3, "the cursor counts what was fed, not what fitted"

    son2 = music.Sonifier()
    son2.configure(256.0, music.Band(2.0, 38.0, "music"))
    n = len(son2._ring)
    son2.feed(np.zeros(n))
    son2.feed(np.zeros(n // 2))
    assert son2._write == n + n // 2, "and it keeps counting across the wrap"


def test_the_status_line_says_which_frequency_is_making_which_pitch():
    """Two numbers and what the pair of ranges does to the intervals between them.

    The EEG frequency has to be beside the pitch, because that is the one thing an operator cannot check
    by listening: a tracker locked onto a rhythm and a tracker wandering in a quiet band sound similar and
    the pair of numbers tells them apart. And whether the ends chosen are a transposition or a stretch is
    arithmetic over two ranges, which nobody should be doing in their head while running a session.
    """
    from nfb_console.feedback import Music

    out = music.MusicOutput()
    out.sonifier.rate_hz = 250.0
    out.sonifier.eeg_hz, out.sonifier.pitch_hz = 10.4, 291.0
    out.sonifier._stream = object()                  # pretend it opened; describe reads `running`
    try:
        said = out.describe(Music(enabled=True, source="eeg"))
        assert "10.4 Hz sounding at 291 Hz" in said
        assert "220–440" in said and "exact transposition" in said
        stretched = out.describe(Music(enabled=True, source="eeg", eeg_pitch_hi_hz=880.0))
        assert "intervals ×2.0" in stretched
    finally:
        out.sonifier._stream = None


def test_every_player_reports_the_dropouts_it_has_been_counting_all_along():
    """**All four counted this and not one showed it**, so a crackle was diagnosed by ear and by theory
    three separate times while the program already knew. Same lesson as `NFB_DEBUG`.

    Silent at zero: a counter reading "0 dropout(s)" on every screen forever is noise, not instrumentation.
    """
    from nfb_console.feedback import Music

    out = music.MusicOutput()
    settings = Music(enabled=True, source="eeg")
    out.sonifier.rate_hz = 256.0
    out.sonifier._stream = object()                 # pretend it opened; describe reads `running`
    try:
        assert "dropout" not in out.describe(settings)
        assert "learned" not in out.describe(settings), "no range is claimed before one is learned"
        out.sonifier.dropouts = 7
        assert "7 dropouts" in out.describe(settings)
        out.sonifier.starved = 1
        out.sonifier.restarts = 2
        said = out.describe(settings)
        assert "1 starved block" in said and "2 restarts" in said, \
            "the buffer running dry and the device being rebuilt are different faults"
    finally:
        out.sonifier._stream = None


def test_loudness_calibrates_itself_against_the_bands_own_range():
    """**There is no `full at` setting any more, because it asked a question nobody can answer.** How many
    microvolts of the band count as full loudness varies per person, per site and per montage, nothing on
    the screen said what today's band was doing, and the unset fallback — the raw trace's ±50/±100 display
    gain against a band producing 2–5 µV — parked the voice within a few percent of its quiet floor for
    whole sessions, reported as "about half as loud as the Shepard at the same volume". The voice now
    hears the band against its own learned quiet and loud, the way a threshold is derived from the person
    — so the same waxing rhythm fills the same loudness range whoever it belongs to.
    """
    rate = 250.0
    times = np.arange(int(rate * 40)) / rate
    # A rhythm that waxes and wanes the way a real band does: amplitude swinging 1:4 over ~4 s cycles.
    waxing = (1.0 + 0.6 * np.sin(2 * np.pi * 0.25 * times)) * np.sin(2 * np.pi * 10 * times)

    def travel(scale_uv: float) -> tuple[float, float, float, float]:
        """(quietest, loudest, median, learned top) of the rendered blocks after calibration, one person."""
        son = music.Sonifier()
        son.configure(rate, music.Band(8.0, 16.0, "music"))
        son.volume = 100.0
        peaks = []
        for i in range(0, len(waxing) - 7, 8):
            son.feed(scale_uv * waxing[i:i + 8])
            block = son._block(1500)[:, 0].astype(float)
            if i / rate > music.EEG_LOUD_CAL_SECONDS:
                peaks.append(float(np.abs(block).max()))
        return min(peaks), max(peaks), float(np.median(peaks)), son.loud_hi

    lo_small, hi_small, mid_small, learned_small = travel(3.0)
    lo_big, hi_big, _mid, learned_big = travel(30.0)

    # The loudness spans a real range with no setting: crests near full scale, troughs well under half.
    assert hi_small > 0.8 * music.TONE_PEAK, f"crests at {hi_small:.4f} of {music.TONE_PEAK}"
    assert lo_small < 0.5 * hi_small, "and the troughs are plainly quieter than the crests"
    # A band in the middle of its range is well over half loud, so the voice sits level with the other
    # sources at the same volume number rather than 6 dB under them.
    assert mid_small > 0.5 * hi_small

    # Self-calibration is per person: ten times the microvolts is the same music, not ten times louder.
    assert hi_big == pytest.approx(hi_small, rel=0.15)
    assert lo_big == pytest.approx(lo_small, rel=0.3)

    # And the learned range is real microvolts, exposed for the state line to print — the answer to
    # "fuck if I know how many microvolts is reasonable" is that nobody has to.
    assert learned_small == pytest.approx(1.6 * 3.0, rel=0.2), "the modulation's own crest, in µV"
    assert learned_big == pytest.approx(1.6 * 30.0, rel=0.2)


def test_the_noise_preview_carries_the_generators_pad():
    """The noise preview carries the generator's pad, so an audition is as loud as the music it previews."""
    import wave as wavelib

    import numpy as np

    from nfb_console import music

    with wavelib.open(str(music.preview_path("pink"))) as fh:
        data = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(float) / 32767
    body = data[len(data) // 4:-len(data) // 4]
    rms = float(np.sqrt((body ** 2).mean()))
    assert rms == pytest.approx(music.PEAK * music.HEADROOM, rel=0.1), "the preview carries the pad"


# --- a stream that stops playing -------------------------------------------------------------

class FakeAudio:
    """A `sounddevice` stand-in whose streams can be told to fail."""

    def __init__(self) -> None:
        self.opened = []
        self.raise_on_write = 0

    def OutputStream(self, **kw):       # noqa: N802 — the name `sounddevice` uses
        stream = FakeStream(self, kw)
        self.opened.append(stream)
        return stream


class FakeStream:
    def __init__(self, audio, kw):
        self.audio, self.kw = audio, kw
        self.written = 0
        self.closed = False

    def start(self):
        pass

    def write(self, block):
        if self.audio.raise_on_write > 0:
            self.audio.raise_on_write -= 1
            raise RuntimeError("PortAudioError: Unanticipated host error")
        self.written += len(block)
        # a real device drains in real time, so the writer blocks once it is ahead
        time.sleep(min(0.005, music.BLOCK_SECONDS))
        return 0


def _tone(monkeypatch, audio):
    monkeypatch.setitem(sys.modules, "sounddevice", audio)
    monkeypatch.setattr(music, "muted", lambda: False)
    monkeypatch.setattr("nfb_console.audio.REOPEN_PAUSE_SECONDS", 0.01)
    player = music.TonePlayer()
    player.configure(volume=30, level=0.5)
    return player


def test_a_stream_that_fails_is_rebuilt_rather_than_leaving_the_session_silent(monkeypatch):
    """The writer thread used to `return` on any exception from `write`, so one hiccup from CoreAudio —
    an output device changing under a running stream — ended the sound for the rest of the session with
    the panel still reading "music on"."""
    audio = FakeAudio()
    player = _tone(monkeypatch, audio)
    assert player.start(), player.error
    try:
        waitfor(lambda: audio.opened[0].written, what="the first block")
        audio.raise_on_write = 1
        waitfor(lambda: player.restarts >= 1, what="the stream to be rebuilt")
        assert len(audio.opened) == 2, "a fresh stream, not the dead one retried"
        waitfor(lambda: audio.opened[1].written, what="sound again on the new stream")
        assert player.error is None, "and nothing left over to report"
    finally:
        player.stop()


def test_stopping_a_sound_returns_at_once_and_the_device_shuts_behind_it(monkeypatch):
    """A program stops its sound on the screen's own loop, at the end of every effort and on every pause.
    Waiting there for the writer and the device took 0.85 s, long enough for the Pupil trial engine to read
    a gap and drop the result, so the stop returns and the device shuts on a thread of its own."""
    audio = FakeAudio()
    shutting = threading.Event()
    shut = []

    def slow_stop(stream):
        shutting.wait(5.0)
        shut.append(stream)

    monkeypatch.setattr(FakeStream, "stop", slow_stop, raising=False)
    monkeypatch.setattr(FakeStream, "close", lambda stream: None, raising=False)
    player = _tone(monkeypatch, audio)
    assert player.start(), player.error
    waitfor(lambda: audio.opened[0].written, what="the first block")
    began = time.monotonic()
    player.stop()
    assert time.monotonic() - began < 0.05, "the caller does not wait for the device"
    assert not player.running
    assert player.start(), "and a fresh start does not wait for the old stream either"
    try:
        shutting.set()
        waitfor(lambda: audio.opened[0] in shut, what="the old stream to be shut")
        first = audio.opened[0].written
        waitfor(lambda: audio.opened[1].written, what="sound on the new stream")
        assert audio.opened[0].written == first, "the old writer never writes again"
    finally:
        player.stop()


# --- session output lifecycle -------------------------------------------------------------

def _amp_session_with_recorders():
    """An amplitude session whose three continuous outputs record apply/release instead of touching the
    machine, so the lifecycle (not the device) is what a test observes."""
    from nfb.stream import Block
    from nfb_console.amplitude_view import AmplitudeSession
    from nfb_console.protocols import BY_CODE
    from conftest import BENCH

    session = AmplitudeSession(BY_CODE["AMP"], path=str(BENCH), channels=2)
    rec = {"released": 0, "applied": 0}
    for name in ("display", "output_volume", "music"):
        obj = getattr(session, name)
        obj.release = lambda *a, r=rec, **k: r.__setitem__("released", r["released"] + 1)
        obj.apply = lambda *a, r=rec, **k: r.__setitem__("applied", r["applied"] + 1)
    return session, rec, Block


def test_a_completed_session_hands_the_outputs_back():
    """Auto-completion set paused and returned without releasing, so the shadow/volume/music held the last
    training value after the session ended. tick() now releases on completion, like pause and suspend."""
    session, rec, _Block = _amp_session_with_recorders()
    try:
        session.toggle_pause()                     # running
        session.plan.complete = lambda _elapsed: True   # as if the configured length was reached
        session.tick()
        assert session.finished and session.paused
        assert rec["released"] > 0, "a finished session releases the outputs"
    finally:
        session.stop()


def test_losing_every_channel_releases_rather_than_holding_the_last_value():
    """When every channel dies no live target advances, so the output-driving block was skipped and the
    machine held the last usable value. A block with no live channel now hands the outputs back."""
    session, rec, Block = _amp_session_with_recorders()
    try:
        session.toggle_pause()
        for i in range(0, 64, 8):                   # several all-no-signal blocks on every channel
            for ch in sorted(session.specs):
                session.feed_block(Block(ch, i, tuple([30000] * 8), session.specs[ch], state=2))
        assert not session.active_targets, "every channel is judged dead"
        assert rec["released"] > 0, "no live channel hands the outputs back"
    finally:
        session.stop()


def test_a_straggling_update_does_not_take_the_outputs_after_stop():
    """A feed-thread straggler could apply outputs inline after stop() released. _apply_outputs now no-ops
    once _stop is set, which stop() does before releasing."""
    session, rec, _Block = _amp_session_with_recorders()
    try:
        session._stop = True
        session._apply_outputs(0.5, True, 0.5, None)
        assert rec["applied"] == 0, "a stopped session takes no outputs"
    finally:
        session.stop()


def test_a_hold_set_while_no_child_exists_is_applied_to_the_next_one(tmp_path, monkeypatch):
    """hold(True) can land while _proc is None — between tracks, or in the window after start() spawns the
    worker before its first Popen. It then only set the flag, with nothing to signal, and the child spawned
    next played on un-paused while the player reported itself held. The worker now stops that fresh child."""
    monkeypatch.setattr(music, "muted", lambda: False)  # faked subprocess, so run the player
    import signal as stdlib
    import threading

    procs = []
    gate = threading.Event()

    class FakeProc:
        def __init__(self, argv, **kw):
            self.signals = []
            procs.append(self)

        def wait(self):
            if len(procs) == 1:
                gate.wait(timeout=2.0)      # hold the first child so the test can set a hold between tracks
            return 0

        def poll(self):
            return None

        def send_signal(self, sig):
            self.signals.append(sig)

        def terminate(self):
            pass

    monkeypatch.setattr(music.subprocess, "Popen", FakeProc)
    tone(tmp_path / "a.wav")
    player = music.TrackPlayer()
    player.start(music.tracks(tmp_path), volume=35)
    for _ in range(200):                    # wait for the first child to spawn and park in wait()
        if procs:
            break
        time.sleep(0.005)
    player.hold(True)                        # a hold while the first child plays, carried on the flag
    gate.set()                               # first child ends; the worker spawns the next under the hold
    for _ in range(200):
        if len(procs) >= 2:
            break
        time.sleep(0.005)
    player.stop()
    assert len(procs) >= 2 and stdlib.SIGSTOP in procs[1].signals, \
        "the child spawned while held is stopped, not left playing"


def test_the_sonifier_clears_a_stale_error_on_a_good_rebuild():
    """A bad band set an error that the next, valid configure never cleared, so the Feedback tab kept
    reporting failure after recovery."""
    son = music.Sonifier()
    son.configure(256.0, music.Band(100.0, 130.0, "past Nyquist"))   # 130 Hz > 128 Hz Nyquist → error
    assert son.error is not None
    son.configure(256.0, music.Band(2.0, 38.0, "music"))             # valid rebuild
    assert son.error is None, "a good rebuild clears the stale error"


def test_stopping_the_file_player_does_not_wait_for_its_player_to_exit(monkeypatch, tmp_path):
    """The stop waited up to a second for the player's thread, on whichever thread asked. A player
    process that is slow to exit must not hold up the caller."""
    monkeypatch.setattr(music, "muted", lambda: False)
    wedged = threading.Event()

    class SlowProc:
        def __init__(self, argv, **kw):
            pass

        def wait(self):
            wedged.wait(5.0)

        def poll(self):
            return None

        def send_signal(self, sig):
            pass

        def terminate(self):
            pass                            # it takes its time

    monkeypatch.setattr(music.subprocess, "Popen", SlowProc)
    track = tmp_path / "t.wav"
    track.write_bytes(b"\x00")
    player = music.TrackPlayer()
    player.start([track], volume=40)
    waitfor(lambda: player._proc is not None, what="the player process")
    began = time.monotonic()
    try:
        player.stop()
        assert time.monotonic() - began < 0.05
    finally:
        wedged.set()


def test_a_held_mark_is_two_of_the_same_sound_rather_than_a_new_one():
    """Rhythm and not pitch, because the pitch is already carrying which band earned it. So the doubled form
    is the base sound twice with a gap, which is why it needs no second pitch and no calibration."""
    import wave as wavelib

    from nfb_console import cues

    base = cues.tone_at(137)
    doubled = cues.held(base)
    assert doubled != base
    assert cues.held(doubled) == doubled, "idempotent"
    assert cues.held("nonesuch") == "nonesuch", "unknown names pass through untouched"
    assert doubled not in cues.click_choices(), "the menu of characters is not where these go"

    one = cues.duration_ms(base)
    both = cues.duration_ms(doubled)
    assert both == pytest.approx(one * 2 + cues.HOLD_GAP_MS)
    with wavelib.open(str(cues.click_path(doubled))) as fh:
        assert fh.getnframes() == pytest.approx(cues.RATE * both / 1000, rel=0.02)

    # It works for a click as well as a tone, since a protocol without an earning target still has one.
    assert cues.duration_ms(cues.held("tick")) > cues.duration_ms("tick")


def test_every_noise_colour_can_be_heard_before_the_session_starts():
    """"1/f²" does not tell anybody how deep brown is, so the preview is the only way the control says what
    it does. Rendered to a file and handed to the system player rather than opened on the audio device: a
    live session may already hold that device, and a preview that silenced the music it previews would be
    worse than none."""
    import wave

    import numpy as np

    from nfb_console import music

    energy = {}
    for colour in music.NOISE_COLOURS:
        path = music.preview_path(colour)
        assert path is not None and path.exists(), colour
        with wave.open(str(path)) as fh:
            frames = fh.getnframes()
            assert fh.getframerate() == music.SAMPLE_RATE
            data = np.frombuffer(fh.readframes(frames), dtype=np.int16).astype(float)
        assert frames == pytest.approx(music.SAMPLE_RATE * music.PREVIEW_SECONDS, rel=0.01)
        assert abs(data).max() < 32767, "clipped noise is harsh in the way this music is chosen not to be"
        assert abs(data[:8]).max() < abs(data).max() * 0.2, "no click at the start"
        # the spectral tilt is the whole content of the control, so it is what the test checks
        spectrum = np.abs(np.fft.rfft(data * np.hanning(len(data))))
        freqs = np.fft.rfftfreq(len(data), 1 / music.SAMPLE_RATE)
        low = spectrum[(freqs > 100) & (freqs < 300)].mean()
        high = spectrum[(freqs > 3000) & (freqs < 6000)].mean()
        energy[colour] = low / high
    assert energy["brown"] > energy["pink"] > energy["white"], f"the tilts are wrong: {energy}"
    assert music.preview_path(colour) is music.preview_path(colour), "rendered once per process"
    assert music.preview_path("chartreuse") is None


def test_the_crossfade_does_not_change_loudness_as_it_fades():
    """Music whose volume dipped mid-fade would read as a broken control rather than as a change of
    character. Both tilts filter the *same* white noise, so the fade interpolates one into the other
    instead of adding two independent sources in power. Measured over four seconds per position: the RMS
    holds within 5% across the sweep, under half a decibel."""
    import numpy as np

    from nfb_console.music import NoisePlayer

    levels = []
    for balance in (0.0, 0.25, 0.5, 0.75, 1.0):
        player = NoisePlayer()
        player.configure("pink", 100.0, second_colour="brown", balance=balance)
        rms = []
        for i in range(400):
            buf = player._block(512)
            if i > 20:                          # let the filter states settle
                rms.append(float(np.sqrt(np.mean(buf.astype(float) ** 2))))
        levels.append(float(np.mean(rms)))
    assert min(levels) > 0.9 * max(levels), f"the fade changes loudness: {levels}"
    assert max(abs(v) for v in buf.ravel()) < 1.0, "and it never reaches the clip point"


def test_a_new_pitch_moves_the_music_and_a_new_colour_at_one_pitch_keeps_its_loudness():
    """A pitch change on one colour moves where the music sits, and a colour change at one pitch stays
    level-matched, so neither control does nothing or jumps the volume."""
    def centroid(player: music.NoisePlayer) -> float:
        audio = np.concatenate([player._block(4096)[:, 0].astype(float) for _ in range(24)])[4096 * 4:]
        power = np.abs(np.fft.rfft(audio)) ** 2
        return float((np.fft.rfftfreq(len(audio), 1 / music.SAMPLE_RATE) * power).sum() / power.sum())

    def rms(player: music.NoisePlayer) -> float:
        return float(np.concatenate([player._block(4096)[:, 0].astype(float)
                                     for _ in range(24)])[4096 * 4:].std())

    player = music.NoisePlayer()
    player.configure("pink", 100.0, pitch_hz=300.0)
    low = centroid(player)
    player.configure("pink", 100.0, pitch_hz=1200.0)
    assert centroid(player) > 1.5 * low

    player.configure("pink", 100.0, pitch_hz=600.0)
    pink = rms(player)
    player.configure("brown", 100.0, pitch_hz=600.0)
    assert rms(player) == pytest.approx(pink, rel=0.25)


def test_a_new_band_is_the_band_the_voice_tracks():
    """Reconfiguring the voice to another band takes effect: of two rhythms, the frequency it reports is
    the one inside the new band."""
    son = music.Sonifier()
    son.configure(250.0, music.Band(8.0, 16.0, "music"))
    son.configure(250.0, music.Band(20.0, 30.0, "music"))
    times = np.arange(250 * 4) / 250.0
    wave = 20 * np.sin(2 * np.pi * 12.0 * times) + 20 * np.sin(2 * np.pi * 25.0 * times)
    for i in range(0, len(wave) - 7, 8):
        son.feed(wave[i:i + 8])
        son._block(1500)
    assert abs(son.eeg_hz - 25.0) < 1.0, son.eeg_hz


def test_choosing_the_eeg_source_starts_the_voice_and_no_other_source_does(monkeypatch):
    """The `eeg` source plays through the voice, and a noise source does not start it."""
    started = []
    monkeypatch.setattr(music.Sonifier, "start", lambda self: started.append("voice") or True)
    monkeypatch.setattr(music.NoisePlayer, "start", lambda self: started.append("noise") or True)
    out = music.MusicOutput()
    try:
        out.apply(Music(enabled=True, source="eeg"), rate_hz=250.0)
        assert started == ["voice"]
        started.clear()
        out.apply(Music(enabled=True, source="noise"), rate_hz=250.0)
        assert started == ["noise"]
    finally:
        out.release()
