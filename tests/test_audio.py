"""The one path to the speaker: every rendered sound held to the rules, the stream that plays them, and the
rule that nothing else plays a generated sound.

No sound is played: the output device is a fake sink throughout, and `NFB_NO_OUTPUTS` is on for the suite.
"""

from __future__ import annotations

import ast
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import waitfor

from nfb_console import audio, cues, music

RATES = (44100, 48000)
LOW_SHARE = 0.05
"""The most of a sound's power that may sit below the speaker corner: the share the corner's own filter
leaves of brown noise, the deepest content rendered here."""
ENVELOPE_ALLOWANCE = 1.1
"""A sine at its peak steps by up to 2π·f·peak/rate between samples; its envelope adds a little to that."""


def _below_corner(x: np.ndarray, rate: int) -> float:
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    return float(power[freqs < audio.SPEAKER_CORNER_HZ].sum() / max(power.sum(), 1e-30))


def test_every_sound_family_is_enumerated():
    keys = [key for key, _render, _hz in audio.every_sound()]
    assert set(cues.SOUNDS) <= set(keys) and set(cues.CLICK_PROFILES) <= set(keys)
    assert any(k.startswith("tone@") for k in keys) and any(k.startswith("alert@") for k in keys)
    assert any(k.endswith("+held") for k in keys) and any(k.startswith("preview-") for k in keys)


@pytest.mark.parametrize("rate", RATES)
def test_every_sound_the_program_renders_obeys_the_rules(rate):
    """A step in the waveform is broadband and a small speaker turns it into a crackle; a sound too near
    full scale buzzes and clips when it overlaps another; content under the speaker corner moves the cone
    without making a sound. So each sound starts and ends at exactly zero, no step between two samples is
    larger than its highest frequency allows, no sample is over `PEAK`, and little of its power is under
    the corner."""
    for key, render, highest in audio.every_sound():
        if key in cues.SOUNDS and not cues.SOUNDS[key].exists():
            continue
        data = np.asarray(render(rate), dtype=float)
        assert data[0] == 0.0 and data[-1] == 0.0, key
        assert np.abs(data).max() <= audio.PEAK + 1e-9, key
        if highest is not None:
            allowed = ENVELOPE_ALLOWANCE * 2 * math.pi * highest * audio.PEAK / rate
            assert np.abs(np.diff(data)).max() <= allowed, key
        assert _below_corner(data, rate) <= LOW_SHARE, key


@pytest.mark.parametrize("rate", RATES)
def test_the_system_sounds_are_resampled_not_decimated(rate):
    for name, path in cues.SOUNDS.items():
        if not path.exists():
            pytest.skip("no system sounds on this machine")
        source, source_rate = cues._read_aiff(path)
        assert len(cues.sound_samples(name, rate)) == pytest.approx(len(source) * rate / source_rate, abs=2)


# --- the stream engine, with a fake sink --------------------------------------------------------

class _Producer(audio.Stream):
    """A stream whose blocks come from a list: an array, or None for a block it has nothing ready for."""

    samplerate = 48000
    block_frames = 256

    def __init__(self, blocks, tail_frames: int = 0):
        self.tail_frames = tail_frames
        self._init_stream()
        self.rate = 48000
        self._blocks = list(blocks)

    def _block(self, frames):
        return self._blocks.pop(0) if self._blocks else np.zeros(frames)


def _sine(frames: int, hz: float = 440.0, amp: float = audio.PEAK, start: int = 0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * hz * (start + np.arange(frames)) / 48000)


def test_a_start_rises_from_silence_and_a_stop_falls_to_it():
    stream = _Producer([_sine(256), _sine(256, start=256), _sine(256, start=512)])
    first = stream.next_block(256)[:, 0]
    assert first[0] == 0.0 and np.allclose(first, _sine(256) * np.linspace(0, 1, 256), atol=1e-6)
    stream.next_block(256)
    last = stream.next_block(256, closing=True)[:, 0]
    assert last[-1] == 0.0
    assert np.allclose(last, _sine(256, start=512) * np.linspace(1, 0, 256), atol=1e-6), "a fade, not a cut"


def test_a_held_sound_stays_silent_through_the_last_block_of_its_stop():
    player = music.NoisePlayer()
    player.configure("pink", 60.0)
    player.held = True
    player.next_block(music.BLOCK_FRAMES)       # running, held: silent, with the stream's gain up
    player.stop()                               # the writer writes its last block after this returns
    assert not player.next_block(music.BLOCK_FRAMES, closing=True).any()


def test_a_continuous_tone_under_the_speaker_corner_is_rolled_off():
    def loudness(base_hz: float) -> float:
        player = music.TonePlayer()
        player.configure(volume=100, level=0.0, base_hz=base_hz)
        player._pitch = player.base_hz
        player._last_level = 1.0
        player._last_corner = float(audio.partial_roll_off(player.base_hz))
        return float(np.sqrt(np.mean(player._block(4096) ** 2)))

    expected = float(audio.partial_roll_off(40.0) / audio.partial_roll_off(220.0))
    assert loudness(40.0) / loudness(220.0) == pytest.approx(expected, rel=0.02)


def test_a_gain_change_is_ramped_across_a_block():
    stream = _Producer([_sine(256), _sine(256, start=256)])
    stream.next_block(256)
    stream.gain = 0.5
    block = stream.next_block(256)[:, 0]
    assert np.allclose(block, _sine(256, start=256) * np.linspace(1, 0.5, 256), atol=1e-6)


def test_a_late_producer_fades_out_and_back_in_rather_than_cutting():
    tail = int(48000 * audio.FADE_MS / 1000)
    blocks = [_sine(256, start=0), _sine(256, start=256), None, None, _sine(256, start=1024),
              _sine(256, start=1280)]
    stream = _Producer(blocks, tail_frames=tail)
    out = np.concatenate([stream.next_block(256)[:, 0] for _ in blocks])
    allowed = ENVELOPE_ALLOWANCE * 2 * math.pi * 440.0 * audio.PEAK / 48000
    assert np.abs(np.diff(out)).max() <= allowed, "no step anywhere: into the silence or out of it"
    assert not out[512 + tail:1024 + tail].any(), "and silence while the producer has nothing"
    assert stream.late == 2


def test_overlapping_sounds_never_reach_full_scale_and_are_never_clipped_flat():
    mixer = audio.Mixer()
    tones = [cues.sound_samples(cues.tone_at(p), 48000) for p in (cues.TONE_LO_HZ, 440.0, 738.0)]
    for offset, tone in [(0, t) for t in tones] + [(37, tones[1]), (211, tones[2])]:
        mixer.add(np.concatenate([np.zeros(offset), tone]))
    stream = _Producer([])
    stream._block = mixer.block
    stream.next_block(256)                      # the start's rise, before the sounds arrive
    raw = np.zeros(40 * 256)
    for offset, tone in [(0, t) for t in tones] + [(37, tones[1]), (211, tones[2])]:
        raw[offset:offset + len(tone)] += tone
    out = np.concatenate([stream.next_block(256)[:, 0] for _ in range(40)])
    assert np.abs(raw).max() > audio.KNEE, "the test reaches the limit"
    assert np.abs(out).max() < 1.0
    assert np.abs(np.diff(out)).max() <= np.abs(np.diff(raw)).max() + 1e-6, "the limit adds no step"


def test_two_sounds_at_the_peak_pass_unchanged():
    stream = _Producer([np.full(256, 2 * audio.PEAK)] * 2)
    stream.next_block(256)
    assert stream.next_block(256)[:, 0] == pytest.approx(2 * audio.PEAK)


def test_one_queued_sound_comes_out_sample_exact_at_the_next_block():
    stream = audio.CueStream()
    stream.rate = 48000
    tone = cues.sound_samples(cues.tone_at(440.0), 48000)
    first = stream.next_block(512)
    stream.mixer.add(tone, 0.5)
    after = np.concatenate([stream.next_block(512)[:, 0] for _ in range(len(tone) // 512 + 2)])
    assert not first.any(), "silence before the sound"
    assert np.allclose(after[:len(tone)], tone * 0.5, atol=1e-7)
    assert not after[len(tone):].any(), "and silence after it"


class _Sink:
    """A fake output: keeps each block written, and stops the writer after `limit` of them."""

    def __init__(self, stop: threading.Event, limit: int, on_write=None):
        self.blocks: list[np.ndarray] = []
        self.stop, self.limit, self.on_write = stop, limit, on_write

    def write(self, block):
        self.blocks.append(block[:, 0].copy())
        if self.on_write is not None:
            self.on_write(len(self.blocks))
        if len(self.blocks) >= self.limit:
            self.stop.set()
        return False


def test_the_writer_writes_silence_between_sounds_and_fades_out_at_the_end():
    stream = audio.CueStream()
    stream.rate = 48000
    tone = cues.sound_samples(cues.tone_at(738.0), 48000)
    stop = threading.Event()
    sink = _Sink(stop, limit=40, on_write=lambda n: stream.mixer.add(tone) if n == 10 else None)
    stream._run(sink, stop)                     # this thread is the writer: no clock, no wait
    assert len(sink.blocks) == 41, "a block whether or not a sound is due, and one last block"
    sounding = [i for i, b in enumerate(sink.blocks) if b.any()]
    assert sounding and sounding[0] == 10
    assert not any(b.any() for b in sink.blocks[:10]) and sink.blocks[-1][-1] == 0.0


def test_a_device_that_takes_everything_and_plays_nothing_is_noticed(monkeypatch):
    monkeypatch.setattr(audio, "REOPEN_PAUSE_SECONDS", 0.0)
    stream = audio.CueStream(opener=lambda: (_ for _ in ()).throw(RuntimeError("no device")))
    stream.rate = 48000
    stop = threading.Event()
    stream._run(_Sink(stop, limit=10_000), stop)   # writes return at once: the lead outruns real time
    assert stream.error and "stopped playing" in stream.error


# --- the cue stream in use ----------------------------------------------------------------------

def _live(monkeypatch, opener):
    monkeypatch.setattr(audio, "muted", lambda: False)
    monkeypatch.setattr(cues, "muted", lambda: False)
    monkeypatch.setattr(cues, "available", lambda: True)
    monkeypatch.setattr(audio, "afplay_available", lambda: True)
    launched = []
    monkeypatch.setattr(audio.subprocess, "Popen", lambda argv, **kw: launched.append(argv))
    stream = audio.CueStream(opener=opener)
    monkeypatch.setattr(audio, "_cues", stream)
    return stream, launched


class _Holding:
    """A device whose buffer is full: each write waits until the test ends."""

    def __init__(self):
        self.release = threading.Event()

    def write(self, block):
        self.release.wait(5.0)
        return False

    def stop(self):
        pass

    def close(self):
        pass


def test_every_short_sound_goes_through_the_open_stream_and_not_a_process(monkeypatch):
    device = _Holding()
    stream, launched = _live(monkeypatch, lambda: (device, 48000))
    try:
        assert not stream.ensure(), "the stream opens on its own thread, not the caller's"
        waitfor(stream.ready, what="the cue stream to open")
        assert cues.play(cues.tone_at(440.0), volume=50) and cues.play("tick")
        if all(p.exists() for p in cues.SOUNDS.values()):
            assert cues.play("beep"), "the system sounds too"
        assert music.play_preview("pink", 0.0, 60), "and the music preview"
        assert not launched and stream.mixer.busy
    finally:
        device.release.set()


def test_with_no_stream_a_sound_falls_back_to_afplay(monkeypatch):
    """A sound plays the slow way when the stream cannot open, and a failed open is not tried again for
    every sound."""
    tries = []

    def broken():
        tries.append(1)
        raise RuntimeError("no device")

    stream, launched = _live(monkeypatch, broken)
    assert cues.play("tick")
    assert launched and launched[0][0] == "afplay", "the sound still plays, the slow way"
    waitfor(lambda: stream.error and not any(t.name == "cue-stream-open" for t in threading.enumerate()),
            what="the failed open to be reported")
    assert cues.play(cues.tone_at(440.0)) and len(launched) == 2
    assert len(tries) == 1, "one failed open per process, not one per sound"


def test_outputs_off_plays_nothing_on_either_path(monkeypatch):
    opened, launched = [], []
    monkeypatch.setattr(audio, "_cues", audio.CueStream(opener=lambda: opened.append(1) or (None, 48000)))
    monkeypatch.setattr(audio.subprocess, "Popen", lambda argv, **kw: launched.append(argv))
    assert audio.muted(), "the suite runs with NFB_NO_OUTPUTS"
    audio.warm()
    assert not cues.play("tick") and not cues.play("beep") and not cues.play(cues.tone_at(440.0))
    assert not music.play_preview("pink", 0.0, 60)
    assert not launched and not opened and not audio._cues.mixer.busy


def test_queueing_a_sound_never_waits_on_the_writer(monkeypatch):
    device = _Holding()
    stream, _ = _live(monkeypatch, lambda: (device, 48000))
    try:
        stream.ensure()
        waitfor(stream.ready, what="the cue stream to open")
        tone = cues.tone_at(440.0)
        audio.sound(tone, lambda rate: cues.sound_samples(tone, rate), 48000)
        done = []
        with stream.mixer._lock:                # the writer in the middle of a mix
            worker = threading.Thread(target=lambda: done.append(cues.play(tone)), daemon=True)
            worker.start()
            waitfor(lambda: done, timeout=1.0, what="the sound to be queued while the writer mixes")
        assert done == [True]
    finally:
        device.release.set()


def test_the_exit_never_waits_on_the_stream(monkeypatch):
    device = _Holding()
    stream, _ = _live(monkeypatch, lambda: (device, 48000))
    try:
        stream.ensure()
        waitfor(stream.ready, what="the cue stream to open")
        began = time.monotonic()
        audio.release()                         # the writer is stuck inside a write
        assert time.monotonic() - began < 0.05
    finally:
        device.release.set()


def test_opening_the_stream_never_holds_up_the_caller(monkeypatch):
    never = threading.Event()

    def slow():
        never.wait(5.0)
        raise RuntimeError("gave up")

    _stream, launched = _live(monkeypatch, slow)
    began = time.monotonic()
    try:
        assert cues.play("tick"), "this sound plays the other way"
        assert time.monotonic() - began < 0.05 and launched
    finally:
        never.set()


def test_a_stalled_stream_is_rebuilt_and_shut_away_from_the_caller(monkeypatch):
    """A device change leaves a stream open and unserviced. A sound then must not be queued unheard, and
    the old stream is shut on a thread of its own: a close inside CoreAudio once hung the program."""
    opened, never = [], threading.Event()

    class Wedged(_Holding):
        def stop(self):
            never.wait(5.0)

        def close(self):
            never.wait(5.0)

    def opener():
        device = Wedged()
        opened.append(device)
        return device, 48000

    stream, launched = _live(monkeypatch, opener)
    try:
        stream.ensure()
        waitfor(stream.ready, what="the cue stream to open")
        stream._last_write = time.monotonic() - audio.STALL_SECONDS - 1.0     # the writer went quiet
        began = time.monotonic()
        assert cues.play("tick") and launched, "not queued unheard: played the other way"
        assert time.monotonic() - began < 0.05, "the caller does not wait for the old stream to shut"
        waitfor(lambda: len(opened) == 2 and stream.ready(), what="a fresh stream")
        assert cues.play("tick") and len(launched) == 1, "and the fresh stream plays"
    finally:
        never.set()
        for device in opened:
            device.release.set()


# --- nothing else plays a generated sound -------------------------------------------------------

SOURCES = sorted(Path(audio.__file__).parent.glob("*.py")) + sorted(
    (Path(audio.__file__).parents[3] / "nfb" / "src" / "nfb").rglob("*.py"))


def _where(path: Path):
    """Each node of a module with the name of the class it sits in, or None."""
    tree = ast.parse(path.read_text())
    out = []

    def walk(node, cls):
        for child in ast.iter_child_nodes(node):
            out.append((child, cls))
            walk(child, child.name if isinstance(child, ast.ClassDef) else cls)

    walk(tree, None)
    return out


def test_only_the_audio_module_opens_a_stream_and_only_it_and_the_file_player_start_afplay():
    assert len(SOURCES) > 20, "the scan found the sources"
    for path in SOURCES:
        for node, cls in _where(path):
            own = path.name == "audio.py"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                assert own or "sounddevice" not in names, f"{path.name} imports sounddevice"
            if isinstance(node, ast.Attribute) and node.attr == "OutputStream":
                assert own, f"{path.name} opens an output stream"
            if isinstance(node, ast.Constant) and node.value == "afplay":
                assert own or (path.name == "music.py" and cls == "TrackPlayer"), \
                    f"{path.name} starts afplay"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "play" and getattr(node.func.value, "id", "") == "audio":
                assert path.name in ("cues.py", "music.py"), \
                    f"{path.name} plays a sound that no family registers"


def test_the_counts_a_crackle_leaves_are_said_and_silent_at_zero():
    stream = _Producer([])
    assert audio.glitches(stream) == ""
    stream.dropouts, stream.starved = 3, 1
    assert audio.glitches(stream) == " · 3 dropouts, 1 starved block"


def test_a_stream_still_opening_is_not_ready():
    """Ready the instant the stream was assigned, a stall noticed before the opening ended was never
    reopened, because the opening still looked under way."""
    stream = audio.CueStream()
    stream._stream, stream._last_write, stream._opening = object(), time.monotonic(), True
    assert not stream.ready()
    stream._opening = False
    assert stream.ready()


def test_a_writer_is_running_before_anything_can_find_it(monkeypatch):
    """A writer published before it was started could be handed to `shut_aside` by a stop in between,
    whose join then raised and left the old stream unshut and no fresh one opened."""
    device = _Holding()
    stream = audio.CueStream(opener=lambda: (device, 48000))
    found = []

    class Watched(threading.Thread):
        def start(self):
            if self.name == stream.stream_name:
                found.append(stream._writer is self)        # what a `stop` would see at this instant
            super().start()

    monkeypatch.setattr(audio.threading, "Thread", Watched)
    monkeypatch.setattr(audio, "muted", lambda: False)
    try:
        assert audio.Stream.start(stream)
        assert found == [False], "published only once it runs"
        assert stream._writer.is_alive()
    finally:
        device.release.set()
        stream.stop()


def test_the_buffer_running_dry_is_counted_even_when_the_host_does_not_flag_it():
    """`write` reports an underflow only when PortAudio noticed one. Running less than a block ahead of
    the device is a crackle either way, and the count is what says the fault is this process rather than
    the speaker. Judged from the lead alone, so no thread and no clock are needed to test it."""
    health = audio.Stream._health
    rate, block = music.SAMPLE_RATE, music.BLOCK_SECONDS
    assert health(int(2.0 * rate), 2.0 - 0.2) == "", "a fifth of a second of buffer is the healthy state"
    assert health(int(2.0 * rate), 2.0 - block / 2) == "starved", "less than one block left to play"
    assert health(int(0.2 * rate), 0.2 - block / 2) == "", "and not while the buffer is still filling"
    assert health(int(9.0 * rate), 9.0 - audio.STALL_SECONDS - 1) == "dead"
