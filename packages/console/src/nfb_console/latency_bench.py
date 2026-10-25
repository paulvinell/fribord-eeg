"""Measure the output half of the latency budget, on this machine.

The latency budget had every DSP stage measured and the output stages estimated — and the estimates
were the wrong half to trust: the music's "0.25 s" was 785 ms measured, and the reward click's
"fire and forget" was 74–117 ms of CoreAudio session setup per click. This script measures what can be
measured *in software*: per-block compute on a real recording, the audio streams' actual reported
output latency, and the cost of one system-volume write and one backlight write.

What it deliberately cannot measure is the **physical** half — the time from a brightness write to
photons leaving the panel, and from an audio write to pressure leaving the speaker. That needs a
photosensor (the Belinskaia paper's own method) and a microphone against a loopback. Until someone does
that, the physical tail is an unknown *added on top* of every number printed here, and the budget says
so rather than pretending this script is the whole answer.

Run via `make latency`. The device-touching sections are harmless by construction — the audio streams
play silence and are closed again, the volume is written *at its current value*, the backlight is
written at its current brightness — so nothing visible or audible changes. They still touch real
devices, which is why this module ignores `NFB_NO_OUTPUTS` only for those reads-in-disguise and takes
nothing it has to give back. A session recording is required for the compute section; any
`sessions/*.jsonl.gz` does.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path


def _quantiles(samples: list[float]) -> str:
    if not samples:
        return "no samples"
    p50 = statistics.median(samples)
    p99 = sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.99))]
    return f"p50 {p50 * 1000:7.3f} ms   p99 {p99 * 1000:7.3f} ms   max {max(samples) * 1000:7.3f} ms"


def bench_compute(recording: Path) -> None:
    """Per-block cost of the full measurement path: gate, three cFIR estimators, target scoring."""
    import numpy as np  # noqa: F401  (imported for parity with the live process' working set)

    from nfb import ReplaySource
    from nfb.dsp import ArtifactGates, Band, CFIRBandPower
    from nfb.stream import Block

    src = ReplaySource(str(recording), realtime=False)
    specs = src.channels()
    spec = next(iter(specs.values()))
    gates = ArtifactGates()
    estimators = [CFIRBandPower(spec, band) for band in
                  (Band(8.0, 12.0, "alpha"), Band(4.0, 8.0, "theta"), Band(15.0, 30.0, "beta"))]
    times: list[float] = []
    for item in src.blocks():
        if not isinstance(item, Block) or item.channel != spec.index:
            continue
        t0 = time.perf_counter()
        judged = gates.feed(item) if item.valid else gates.pass_through(item)
        for est in estimators:
            est.feed(judged.fed)
        times.append(time.perf_counter() - t0)
        if len(times) >= 4000:
            break
    print(f"  gate + 3 cFIR estimators, per block:      {_quantiles(times)}")


def bench_audio() -> None:
    """What the audio device actually grants, against what the code asks for.

    `stream.latency` is PortAudio's own figure for output — the delay between writing a sample and the
    hardware consuming it. It is the number the `latency=` parameter *suggests*, and the gap between
    the two is the 785 ms lesson.
    """
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"  sounddevice unavailable: {exc}")
        return

    from . import music

    cases = [
        ("music stream, as shipped", dict(samplerate=music.SAMPLE_RATE, blocksize=music.BLOCK_FRAMES,
                                        latency=music.BUFFER_LATENCY)),
        ("music stream, asking for 0.25", dict(samplerate=music.SAMPLE_RATE,
                                             blocksize=music.BLOCK_FRAMES, latency=0.25)),
        ("click stream, as shipped", dict(samplerate=44100, blocksize=256, latency="low")),
    ]
    for label, kwargs in cases:
        try:
            stream = sd.OutputStream(channels=1, dtype="float32", **kwargs)
            stream.start()
            granted = float(stream.latency)
            stream.stop()
            stream.close()
            print(f"  {label:34s}  asked {kwargs['latency']!r:8}  granted {granted * 1000:6.1f} ms")
        except Exception as exc:
            print(f"  {label:34s}  failed: {exc}")


def bench_volume() -> None:
    """One CoreAudio volume write, at the volume's current value so nothing audible changes."""
    from . import sysvolume

    if not sysvolume.available():
        print(f"  unavailable: {sysvolume.problem()}")
        return
    current = sysvolume.get()
    if current is None or current < sysvolume.HARD_FLOOR:
        print("  current volume under the floor; only timing the read")
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            sysvolume.get()
            times.append(time.perf_counter() - t0)
        print(f"  volume read:                              {_quantiles(times)}")
        return
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        sysvolume.set(current)
        times.append(time.perf_counter() - t0)
    print(f"  volume write (at its current value):      {_quantiles(times)}")


def bench_shadow() -> None:
    """The display lookup and one backlight write, at the current brightness so nothing visibly
    changes. The lookup is a window-server round trip and was measured at 15 ms worst — the reason
    outputs no longer run on the feed thread."""
    from . import shadow

    if not shadow.available():
        print(f"  unavailable: {shadow.problem()}")
        return
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        found = shadow.displays()
        times.append(time.perf_counter() - t0)
    print(f"  display lookup:                           {_quantiles(times)}")
    if not found:
        print("  no display to write to")
        return
    display = found[0]
    current = shadow.get(display)
    if current is None:
        print("  cannot read the current brightness; skipping the write")
        return
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        shadow.set(display, current)
        times.append(time.perf_counter() - t0)
    print(f"  backlight write (at current brightness):  {_quantiles(times)}")


def main(argv: list[str] | None = None) -> int:
    args = set(argv if argv is not None else sys.argv[1:])
    everything = "--all" in args or not args
    recording = Path("sessions/latest.jsonl.gz")

    print("output-path latency, software half — the physical panel and speaker are NOT in these")
    print("numbers; measuring those needs a photosensor and a microphone\n")
    if everything or "--compute" in args:
        print("compute (replaying a recording):")
        if recording.exists():
            bench_compute(recording)
        else:
            print(f"  no recording at {recording} — run `make record` first")
        print()
    if everything or "--audio" in args:
        print("audio streams (silence, opened and closed):")
        bench_audio()
        print()
    if everything or "--volume" in args:
        print("system volume (written at its current value):")
        bench_volume()
        print()
    if everything or "--shadow" in args:
        print("backlight (written at its current brightness):")
        bench_shadow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
