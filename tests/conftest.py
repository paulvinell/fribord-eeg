"""Keep the test suite off the machine's outputs.

Sound, shadow and click are on by default, which is right for a session and wrong for a test run: without
this, every test that starts a session dims the operator's display, takes over the system volume and plays
clicks through the speakers. It also made a test *fail* — driving the real outputs on every block of a
non-realtime replay is slow enough to miss a deadline — which is the good kind of warning, because the
same thing on a slower machine would have looked like a flaky test rather than like a side effect.

Autouse, so it cannot be forgotten by a test added later. Anything that genuinely wants to exercise an
output does so through its own fake, as `test_shadow.py` does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal as _sp


@pytest.fixture(autouse=True)
def _no_real_outputs(monkeypatch):
    # The same switch any script should use: `NFB_NO_OUTPUTS=1`. Honoured inside `shadow.available`,
    # `sysvolume.available` and `cues.available`, so the lifecycle code under test still runs — it just
    # reaches a device that is not there, which is a state it has to handle anyway.
    monkeypatch.setenv("NFB_NO_OUTPUTS", "1")

    from nfb_console import prefs, shadow, sysvolume

    # belt and braces: even a direct call to a setter must not reach the hardware
    monkeypatch.setattr(shadow, "set", lambda *a, **k: None)
    monkeypatch.setattr(sysvolume, "set", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _prefs_off_the_real_file(tmp_path, monkeypatch):
    """A session writes its calibration on `stop`, and every test here stops one.

    Without this the suite would rewrite the operator's own `~/.nfb/prefs.json` from whatever a test happened
    to set — the file-side twin of what `NFB_NO_OUTPUTS` prevents, and quieter, because nothing about a test
    run would look wrong until the next real session opened with a stranger's volumes.
    """
    from nfb_console import prefs

    monkeypatch.setenv("NFB_PREFS", str(tmp_path / "prefs.json"))
    monkeypatch.setattr(prefs, "PATH", tmp_path / "prefs.json")


# --- the bench recording the screen tests replay ---------------------------------------------

BENCH = Path(__file__).parent / "fixtures" / "bench.jsonl.gz"
"""Open inputs on every channel: railed to full scale, signal state 2 on all four and on SUM.

**This used to be `sessions/latest.jsonl.gz`, which is gitignored and is whatever the operator recorded
last.** So `make record` — the documented way to capture a session — silently replaced the fixture that
four screen tests assert against, and two of them failed with real electrodes attached because a live
channel reports state 0 where the assertion wanted "no signal". A test that a normal day's work breaks is
not testing what it claims to.

Synthetic rather than captured, and it is the honest choice here: what these tests need is real *frames*
through the real decoder, not real EEG. The old file was no-electrode bench data, so a constant at the
rail reproduces it — a bandpass of a constant is zero either way. Anything wanting genuine EEG should use
`SampleFileSource` on an export, as `test_file_sources.py` does.
"""

_RAIL = 32767            # full scale; +800.0 µV on the 800 µV range, as an open input reads
_GROUPS = 8              # the device packs exactly 8 sample groups per frame at every rate


def _bench_state() -> int:
    """Signal state 2 on SUM and on all four channels.

    Two bits per entity, SUM in bits 0-1 and channel N at bit 2*(N+1) — `optima.decode_data` is the
    authority. Built rather than written as a literal because 682 explains nothing.
    """
    from nfb.stream import NO_SIGNAL

    state = NO_SIGNAL
    for channel in range(4):
        state |= NO_SIGNAL << (2 * (channel + 1))
    return state


def _write_bench(path: Path, seconds: float = 8.0, hz: float = 250.0,
                 state: int | None = None) -> None:
    import struct

    import optima
    from nfb.record import SessionHeader, SessionWriter
    from nfb.stream import PHASE_MODULUS

    cfg = optima.build_eeg_config(4, hz=hz)
    plan = [c for c in optima.plan_channels(cfg) if c.enabled]
    state = _bench_state() if state is None else state
    interval = _GROUPS / hz
    path.parent.mkdir(parents=True, exist_ok=True)
    header = SessionHeader(config=cfg, device="Neurobit Optima+ 4 USB", serial="00000000",
                           firmware="2.9.5", note="synthetic bench recording: open inputs")
    with SessionWriter(path, header) as writer:
        for index in range(int(seconds * hz / _GROUPS)):
            phase = (index * _GROUPS) % PHASE_MODULUS
            values = [_RAIL for step in range(_GROUPS)
                      for c in plan if not (c.mask & (((phase + step) % PHASE_MODULUS) + 1))]
            payload = struct.pack("<HH", phase, state) + struct.pack(f"<{len(values)}h", *values)
            writer.write(optima.NOT_DATA, payload, elapsed=index * interval)


BENCH_OK = Path(__file__).parent / "fixtures" / "bench-ok.jsonl.gz"
"""The same synthetic frames with **signal state 0 on everything** — a device that says its channels are
usable. Needed by tests of machinery that (rightly) refuses device-invalid data: the waveform screen's
calibration will not measure a range from readings the device already called unusable, so against `BENCH`
it correctly never calibrates and a test of the calibrated path never gets there."""

if not BENCH.exists():
    _write_bench(BENCH)
if not BENCH_OK.exists():
    _write_bench(BENCH_OK, state=0)


# --- no test waits on the clock --------------------------------------------------------------

SLEEP_CEILING_S = 0.05
"""The longest a test may ask to sleep for. Anything above this raises.

**A test that waits a fixed span is measuring the scheduler, and it is slow by construction.** The suite
had twenty-seven such waits and they were most of its runtime: sixty seconds of a seventy-second run went
on polling with 50 ms sleeps, waiting for real-time simulators, and joining threads whose work the test
could have done itself. The cure is not a shorter sleep, it is not sleeping: the DSP takes the samples,
`TrainingSession.feed_block` takes a block, the sources take `realtime=False`, and a test drives them
directly and stops when it has what it needs.

What is left below the ceiling is the one honest use — yielding so another thread can make progress while
this one waits for a *condition* it names. Use `waitfor` for that: it costs what it needs and not a
millisecond more, and it names the condition, so a failure says what never happened rather than that a
deadline passed.
"""


def waitfor(condition, timeout: float = 2.0, what: str = "") -> None:
    """Wait until `condition()` is true. Raises with `what` if it never becomes true.

    The only sanctioned way for a test to wait, and it is for genuinely concurrent behaviour — a reader
    thread pumping frames, a queue filling — where the thing under test is that another thread makes
    progress. Everything else drives the code directly.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if condition():
            return
        _time.sleep(0.001)
    raise AssertionError(what or "condition never became true")


@pytest.fixture(autouse=True)
def _no_long_sleeps(monkeypatch):
    """Make the wrong thing impossible rather than discouraged. See SLEEP_CEILING_S.

    **Only the test's own sleeps.** The code under test sleeps for perfectly good reasons — a reader
    thread pacing itself, a replay honouring a sample rate, the draw loop's tick — and a guard that
    raised in those turned them into busy loops and tripled the suite's runtime, which is a nice
    demonstration of why a blunt instrument is the wrong one. The caller's file is what decides.
    """
    import sys
    import time as _time

    real_sleep = _time.sleep
    here = str(Path(__file__).parent)

    def guarded(seconds):
        caller = sys._getframe(1).f_code.co_filename
        if caller.startswith(here) and seconds > SLEEP_CEILING_S:
            raise AssertionError(
                f"a test asked to sleep for {seconds:g}s. Tests do not wait on the clock: drive the "
                f"code directly (feed_block, push, realtime=False), or wait on a named condition with "
                f"conftest.waitfor.")
        return real_sleep(seconds)

    monkeypatch.setattr(_time, "sleep", guarded)


def counts(values, per_count: float, truncate: bool = False):
    """µV to the integer counts the device delivers, for a whole array at once.

    **The conversion, not the signal, is what makes a DSP test slow.** Eight minutes at 250 Hz is
    120,000 samples; generating them with numpy costs about a millisecond, and turning them into blocks
    one `int(round(v / scale))` at a time costs a hundred. Done here in one vectorised pass it is
    seventeen, for byte-identical blocks — and the per-sample version was half the runtime of the
    slowest test in the suite.

    `truncate` picks between the two conventions already in use: rounding to nearest (numpy's `rint`
    and Python's `round` are both half-to-even, so they agree exactly) and truncating toward zero, which
    is what `int(v * scale)` does. They differ by a count, which is 24 nV here and matters to nothing —
    but a test's numbers should not move because a helper arrived.
    """
    scaled = np.asarray(values, dtype=float) / per_count
    return (scaled.astype(np.int64) if truncate else np.rint(scaled).astype(np.int64))


# --- how wide a piece of text renders ------------------------------------------------------

_NARROW = set("iljItfr|!.,';:()[]{}-` ")
_WIDE = set("mwMW@%")


def text_width(value: str, size: float, bold: bool = False) -> float:
    """Roughly how many pixels a string occupies in the UI font, for "does it fit" checks.

    **Approximate on purpose, and calibrated rather than guessed.** Flet gives no way to measure text
    from Python, so this is a per-character advance model: narrow letters and punctuation, wide ones, and
    a default. It was fitted against a known clip point — the launcher card's 203-pixel slot cut
    "Two synchrony rewards, twelve inhibit bins" after "inhibi", and this scores that prefix at 201.

    Use it to catch text that plainly overflows, not to lay anything out. It errs on the generous side of
    reality for mixed-case prose, so something it calls too long is too long; something it passes by a
    hair may still clip, which is why callers leave a margin.
    """
    total = 0.0
    for ch in value:
        if ch in _NARROW:
            total += 0.30
        elif ch in _WIDE:
            total += 0.92
        elif ch.isupper() or ch.isdigit():
            total += 0.62
        else:
            total += 0.53
    return total * size * (1.06 if bold else 1.0)


def plausible_eeg(seconds: float, rate: float, channel: int, seed: int = 0) -> np.ndarray:
    """Something a screen can be photographed against.

    **Not the bench recording**, which is open inputs railed to full scale: a bandpass of a constant is
    zero, so every plot in a "fed" shot would be as empty as a cold one and the state would be a lie.
    Alpha over a 1/f background, and on the first channel a theta rhythm whose phase gates the gamma —
    so the coupling screen has a coupling to draw rather than a flat line.
    """
    rng = np.random.default_rng(seed + channel)
    n = int(seconds * rate)
    freqs = np.fft.rfftfreq(n, 1 / rate)
    freqs[0] = freqs[1]
    background = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / freqs ** 0.6, n)
    background *= 18.0 / background.std()

    def band(low, high, amp):
        sos = _sp.butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
        y = _sp.sosfiltfilt(sos, rng.standard_normal(n))
        return amp * y / y.std()

    out = background + band(8.0, 12.0, 14.0)
    if channel == 0:
        slow = band(4.0, 8.0, 20.0)
        gate = 1.0 - 0.8 * np.cos(np.angle(_sp.hilbert(slow)))
        out = out + slow + gate * band(30.0, 45.0, 6.0)
    return out
