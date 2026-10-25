"""Sound as the system volume, and music as a track, a folder, or the EEG.

Same shape of risk as the shadow: these hold something that belongs to the operator's machine, so the
floor and the restore are tested rather than trusted. The real CoreAudio and the real audio device are
faked throughout, so the suite is silent and leaves the volume alone.
"""

from __future__ import annotations

import math
import sys
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

    def fake_set(level):
        level = min(1.0, max(sysvolume.HARD_FLOOR, level))
        state["level"] = level
        state["writes"].append(round(level, 4))

    monkeypatch.setattr(sysvolume, "available", lambda: True)
    monkeypatch.setattr(sysvolume, "problem", lambda: None)
    monkeypatch.setattr(sysvolume, "get", lambda: state["level"])
    monkeypatch.setattr(sysvolume, "set", fake_set)
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


def test_the_player_reports_what_it_would_run_and_stops_cleanly(tmp_path, monkeypatch):
    """The subprocess is faked: the point is the track order, the volume argument and the shutdown."""
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


def test_music_volume_is_independent_of_the_system_volume():
    """You set the backing track once and let the feedback move the master over the top of it."""
    settings = Music(enabled=True, source="file", volume=35)
    assert settings.volume == 35
    assert not hasattr(settings, "system_volume")


# --- music: the EEG ---------------------------------------------------------------------------

def test_the_sonifier_bandpasses_what_it_is_fed():
    son = music.Sonifier()
    son.configure(256.0, music.Band(2.0, 38.0, "music"))
    assert son._filter is not None

    times = np.arange(2560) / 256.0
    in_band = 20 * np.sin(2 * np.pi * 10 * times)          # 10 Hz: passes
    son.feed(in_band)
    written = son._write
    assert written == len(in_band)
    kept = np.abs(son._ring[: son._write]).max()

    son2 = music.Sonifier()
    son2.configure(256.0, music.Band(2.0, 38.0, "music"))
    drift = 20 * np.sin(2 * np.pi * 0.3 * times)           # 0.3 Hz: rejected
    son2.feed(drift)
    rejected = np.abs(son2._ring[: son2._write]).max()
    assert rejected < kept / 10, "drift below the band must not become audible"


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
    assert son._filter is None and son.error
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
    stream = Stream()
    player._stream = stream
    player._stop.set()                       # one pass and out
    player._run()
    assert stream.written == 0, "the flag is checked before writing, so a stopped music writes nothing"

    player._stop.clear()
    original = player._block
    calls = []

    def once(frames):
        calls.append(frames)
        if len(calls) >= 3:
            player._stop.set()
        return original(frames)

    player._block = once
    player._run()
    assert stream.written == 3 and player.dropouts == 3
    stream.starved = False
    player._stop.clear()
    calls.clear()
    player._run()
    assert player.dropouts == 3, "and a block that arrived in time is not counted"
    # The slack is PortAudio's 'low' preset — measured 231 ms at this block size, ample for a GC pause
    # plus a heavy frame. A float here is only a *suggestion*, and CoreAudio inflates it: asking for
    # 0.25 s, as this constant used to, yields 785 ms of real output latency on the feedback path.
    assert music.BUFFER_LATENCY == "low"


def test_the_stdlib_signal_module_is_not_shadowed():
    """`from scipy import signal` would have silently replaced the stdlib one, and `hold()` would have
    raised AttributeError at the moment a target failed."""
    import signal as stdlib

    assert music.signal is stdlib
    assert hasattr(music.signal, "SIGSTOP")


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


# --- the click's persistent path ---------------------------------------------------------------

def test_a_click_plays_through_the_open_stream_not_a_fresh_process(monkeypatch):
    """A fresh `afplay` costs 74-117 ms of CoreAudio session setup per click — measured — and the click
    is the most moment-locked sound in the app. Through a pre-opened stream it costs only the stream's
    own output latency. The two system cues stay on `afplay`: a reminder half a second late is still a
    reminder."""
    import sys

    from nfb_console import cues

    class Stream:
        def __init__(self, **kw):
            self.kw = kw

        def start(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", type("SD", (), {"OutputStream": Stream}))
    monkeypatch.setattr(cues, "muted", lambda: False)
    monkeypatch.setattr(cues, "available", lambda: True)
    launched = []
    monkeypatch.setattr(cues.subprocess, "Popen", lambda argv, **kw: launched.append(argv))
    monkeypatch.setattr(cues, "_stream", cues._ClickStream())

    assert cues.play("tick", volume=50)
    assert not launched, "no child process for a synthesised sound"
    assert cues._stream._voices, "the click is queued on the open stream"
    data, gain, pos = cues._stream._voices[0]
    assert gain == 0.5 and pos == 0 and len(data)

    assert cues.play("beep")
    assert launched and launched[0][0] == "afplay", "system cues still go the slow, simple way"


def test_the_click_falls_back_to_afplay_when_no_stream_opens(monkeypatch):
    import sys

    from nfb_console import cues

    class Broken:
        def __init__(self, **kw):
            raise RuntimeError("no device")

    monkeypatch.setitem(sys.modules, "sounddevice", type("SD", (), {"OutputStream": Broken}))
    monkeypatch.setattr(cues, "muted", lambda: False)
    monkeypatch.setattr(cues, "available", lambda: True)
    launched = []
    monkeypatch.setattr(cues.subprocess, "Popen", lambda argv, **kw: launched.append(argv))
    monkeypatch.setattr(cues, "_stream", cues._ClickStream())

    assert cues.play("tick")
    assert launched and launched[0][0] == "afplay", "the click still sounds, just the slow way"
    assert not cues._stream._tried or cues._stream.error


def test_the_stream_mixes_overlapping_sounds_and_retires_finished_ones():
    import numpy as np

    from nfb_console import cues

    stream = cues._ClickStream()
    short = np.full(100, 8192, dtype=np.int16)      # below half scale, so the sum is not limited
    stream._voices.append([short.astype(np.float32) / 32767.0, 1.0, 0])
    stream._voices.append([short.astype(np.float32) / 32767.0, 1.0, 0])
    out = np.zeros((256, 1), dtype=np.float32)
    stream._callback(out, 256, None, None)
    assert out[:100, 0] == pytest.approx(2 * 8192 / 32767.0), "voices add, they do not replace"
    assert not out[100:].any()
    assert not stream._voices, "finished voices are retired"


def test_every_rendered_sound_ends_at_silence():
    """**A step in the waveform is broadband**, and a speaker asked to jump from 7% of full scale to
    silence in one sample makes a click — on a small laptop speaker, a crackle. The pacer's breath cue was
    an exponential truncated at 2.5 time constants, so it ended at 0.074 of full scale: a 2422-count step
    on the end of every breath. This is the third crackle family in this codebase and the other two are in
    `music` — a starved stream, and a cone driven past its excursion limit by low frequencies. The fix for
    this one is neither: it is a release ramp.
    """
    import numpy as np

    from nfb_console import cues

    names = [cues.tone_at(880), cues.tone_at(660), *cues.click_choices()]
    names += [cues.held(n) for n in names]
    for name in names:
        data = cues.click_samples(name)
        assert data is not None, name
        assert data[-1] == 0, f"{name} ends at {data[-1]} rather than at silence"
        # And it arrives there by fading, not by being cut: the last millisecond is a fraction of the
        # peak. Asserting only the final sample would pass a waveform that jumps to zero from anywhere.
        last_ms = np.abs(data[-int(cues.RATE / 1000.0):].astype(float)).max()
        assert last_ms < 0.05 * np.abs(data.astype(float)).max(), \
            f"{name} is still at {last_ms:.0f} counts a millisecond from its end"


def test_two_sounds_at_once_cannot_ask_for_more_than_full_scale():
    """Mixing is addition and every voice is rendered near full scale, so a reward click landing on a
    breath cue can ask for 1.8 — and what the device does past 1.0 is flatten the samples, which is a
    square wave, which is crackle. Additive mixing is still right; the sum is limited instead."""
    import numpy as np

    from nfb_console import cues

    stream = cues._ClickStream()
    loud = np.ones(512, dtype=np.float32) * 0.9
    stream._voices = [[loud, 1.0, 0], [loud, 1.0, 0], [loud, 1.0, 0]]
    out = np.zeros((256, 1), dtype=np.float32)
    stream._callback(out, 256, None, None)
    assert float(np.abs(out).max()) <= 1.0, "the mix is limited rather than left to the device"
    assert float(out.max()) == 1.0, "and it is a ceiling, not an attenuation of everything below it"


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
    """**The audio callback used to wait on a lock the feed thread held across a per-sample Python loop.**

    That is a stall of the sound card for however long the producer needs the GIL, which is the shape of
    "it crackles occasionally on my laptop". There is one writer and one reader, so the ring is written
    first and the cursor published afterwards: the reader only looks below the cursor it read, and cannot
    see a half-written block.

    The cursor counts *samples fed*, not ring positions — the reader takes differences of it to decide how
    far behind it is, so a block longer than the ring must still advance it by everything fed even though
    only the newest samples survive.
    """
    son = music.Sonifier()
    son.configure(256.0, music.Band(2.0, 38.0, "music"))
    assert not hasattr(son, "_lock"), "a lock here is a stall of the audio device"

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

    def travel(scale_uv: float) -> tuple[float, float, float]:
        """(quietest, loudest, learned top) of the rendered blocks after calibration, one person."""
        son = music.Sonifier()
        son.configure(rate, music.Band(8.0, 16.0, "music"))
        son.volume = 100.0
        peaks = []
        for i in range(0, len(waxing) - 7, 8):
            son.feed(scale_uv * waxing[i:i + 8])
            block = son._block(1500)[:, 0].astype(float)
            if i / rate > music.EEG_LOUD_CAL_SECONDS:
                peaks.append(float(np.abs(block).max()))
        return min(peaks), max(peaks), son.loud_hi

    lo_small, hi_small, learned_small = travel(3.0)
    lo_big, hi_big, learned_big = travel(30.0)

    # The loudness spans a real range with no setting: crests near full scale, troughs well under half.
    assert hi_small > 0.8 * music.TONE_PEAK, f"crests at {hi_small:.4f} of {music.TONE_PEAK}"
    assert lo_small < 0.5 * hi_small, "and the troughs are plainly quieter than the crests"

    # Self-calibration is per person: ten times the microvolts is the same music, not ten times louder.
    assert hi_big == pytest.approx(hi_small, rel=0.15)
    assert lo_big == pytest.approx(lo_small, rel=0.3)

    # And the learned range is real microvolts, exposed for the state line to print — the answer to
    # "fuck if I know how many microvolts is reasonable" is that nobody has to.
    assert learned_small == pytest.approx(1.6 * 3.0, rel=0.2), "the modulation's own crest, in µV"
    assert learned_big == pytest.approx(1.6 * 30.0, rel=0.2)


def test_a_steady_voice_sits_near_the_other_sources_not_six_decibels_under():
    """The loudness curve is a square root, and this is what it is for: the band spends most of a session
    mid-range, and mapped straight that is half amplitude — a voice 6 dB under the Shepard at the same
    volume number, which is the unmarked-per-source loudness change `TONE_PEAK` exists to prevent. The
    root lifts the middle to 0.71 of full and moves neither end."""
    assert music.EEG_QUIET_GAIN + (1 - music.EEG_QUIET_GAIN) * math.sqrt(0.5) == pytest.approx(0.74, abs=0.01)
    assert music.EEG_QUIET_GAIN + (1 - music.EEG_QUIET_GAIN) * math.sqrt(1.0) == 1.0
    assert music.EEG_QUIET_GAIN + (1 - music.EEG_QUIET_GAIN) * math.sqrt(0.0) == music.EEG_QUIET_GAIN


# --- a stream that stops playing -------------------------------------------------------------

class FakeAudio:
    """A `sounddevice` stand-in whose streams can be told to fail, or to stop playing without failing."""

    def __init__(self) -> None:
        self.opened = []
        self.raise_on_write = 0
        self.deaf = False               # accepts every write instantly and plays none of it

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
        if not self.audio.deaf:
            # a real device drains in real time, so the writer blocks once it is ahead
            time.sleep(min(0.005, music.BLOCK_SECONDS))
        return 0


def _tone(monkeypatch, audio):
    monkeypatch.setitem(sys.modules, "sounddevice", audio)
    monkeypatch.setattr(music, "muted", lambda: False)
    monkeypatch.setattr(music, "REOPEN_PAUSE_SECONDS", 0.01)
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


def test_a_device_that_accepts_everything_and_plays_none_of_it_is_noticed(monkeypatch):
    """The failure with no exception in it: the stream stays open, `write` keeps returning, and nothing
    comes out. Audio handed over has to be consumed at real time, so a lead that grows past
    `STALL_SECONDS` is the device saying nothing is listening."""
    audio = FakeAudio()
    audio.deaf = True
    player = _tone(monkeypatch, audio)
    monkeypatch.setattr(music, "STALL_SECONDS", 0.5)
    assert player.start(), player.error
    try:
        waitfor(lambda: player.restarts >= 1, what="the silent death to be caught")
    finally:
        player.stop()


def test_the_buffer_running_dry_is_counted_even_when_the_host_does_not_flag_it():
    """`write` reports an underflow only when PortAudio noticed one. Running less than a block ahead of
    the device is a crackle either way, and the count is what says the fault is this process rather than
    the speaker. Judged from the lead alone, so no thread and no clock are needed to test it."""
    health = music._AudioStream._health
    rate, block = music.SAMPLE_RATE, music.BLOCK_SECONDS
    assert health(int(2.0 * rate), 2.0 - 0.2) == "", "a fifth of a second of buffer is the healthy state"
    assert health(int(2.0 * rate), 2.0 - block / 2) == "starved", "less than one block left to play"
    assert health(int(0.2 * rate), 0.2 - block / 2) == "", "and not while the buffer is still filling"
    assert health(int(9.0 * rate), 9.0 - music.STALL_SECONDS - 1) == "dead"
