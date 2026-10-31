"""Record the headband and the HeartMath together, and compare the two pulses.

Usage: `make heartpair SECONDS=120`, or `uv run python tools/heart_pair.py --seconds 120`. `--replay FILE`
runs the comparison again on a saved recording, with no devices.

This file only opens the two devices, keeps what they send, saves it to `sessions/` and prints the
result. The pulse comes out of the headband's frames through `nfb.heg.PulseExtractor`, and the comparison
is `nfb.hrv.compare`. Over the USB cable the HeartMath sends its raw pulse waveform as well as its beats,
which the height ratio needs; over Bluetooth only beat times arrive.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
from pathlib import Path

from nfb.heg.pulse import PulseExtractor
from nfb.heg.source import Channel, Frame
from nfb.hrv.compare import Comparison, compare

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RATE_HZ = 500.0
CONNECT_S = 30.0
"""How long to wait for both devices to start sending before recording begins."""


def _row(f: Frame) -> dict:
    return {"t": f.t, **{f"{c}_{k}": getattr(getattr(f, c), k) for c in "lrp" for k in ("ir", "red", "amb")},
            "acc": list(f.acc), "gyro": list(f.gyro), "temp": f.temp_c}


def _frame(r: dict) -> Frame:
    return Frame(t=r["t"], l=Channel(r["l_ir"], r["l_red"], r["l_amb"]),
                 r=Channel(r["r_ir"], r["r_red"], r["r_amb"]), p=Channel(r["p_ir"], r["p_red"], r["p_amb"]),
                 acc=tuple(r["acc"]), gyro=tuple(r["gyro"]), temp_c=r["temp"])


def record(seconds: float, heart: str, address: str | None = None) -> dict:
    """Read both devices for `seconds` once both are sending, on a thread each."""
    from nfb.heg.source import LiveMendi
    from nfb.hrv.ppg import PpgDecoder
    from nfb.hrv.sources import BleBeats, UsbPpgBeats

    mendi = LiveMendi()
    reference = UsbPpgBeats() if heart == "usb" else BleBeats(address=address)
    if heart == "usb":
        # The decoder keeps the waveform for a caller; by default only the last eight seconds of it.
        reference.decoder = PpgDecoder(raw_keep=int((seconds + CONNECT_S + 10.0) * REFERENCE_RATE_HZ))
    frames: list[dict] = []
    beats: list[float] = []
    stop = threading.Event()

    def read_mendi() -> None:
        for f in mendi.frames():
            frames.append(_row(f))
            if stop.is_set():
                return

    def read_reference() -> None:
        for b in reference.beats():
            beats.append(b.t)
            if stop.is_set():
                return

    threads = [threading.Thread(target=read_mendi, daemon=True),
               threading.Thread(target=read_reference, daemon=True)]
    for thread in threads:
        thread.start()
    waited = time.time()
    while not (frames and beats) and time.time() - waited < CONNECT_S:
        time.sleep(0.25)
    if not (frames and beats):
        print(f"waiting for both devices: headband {'sending' if frames else 'silent'}, "
              f"heart sensor {'sending' if beats else 'silent'}")
    started = time.time()
    while time.time() - started < seconds:
        time.sleep(1.0)
        left = seconds - (time.time() - started)
        if int(left) % 10 == 0:
            print(f"  {left:3.0f} s left · headband {len(frames)} frames · heart sensor {len(beats)} beats",
                  flush=True)
    stop.set()
    mendi.close()
    reference.close()
    for thread in threads:
        thread.join(timeout=3.0)
    wave = [list(p) for p in reference.decoder.raw] if heart == "usb" else None
    return {"seconds": seconds, "heart": heart, "frames": frames, "reference_beats": beats,
            "reference_wave": wave, "errors": {"headband": mendi.error, "heart sensor": reference.error}}


def analyse(recording: dict) -> tuple[Comparison, list[float]]:
    extractor = PulseExtractor()
    forehead_beats: list[float] = []
    forehead_wave: list[tuple[float, float]] = []
    for row in recording["frames"]:
        forehead_beats += [b.t for b in extractor.push(_frame(row))]
        if extractor.waveform and (not forehead_wave or extractor.waveform[-1][0] != forehead_wave[-1][0]):
            forehead_wave.append(extractor.waveform[-1])
    # The HeartMath's raw pulse points down (averaged, it rises slowly and falls fast); flipped upright here.
    reference_wave = ([(t, -v) for t, v in recording["reference_wave"]]
                      if recording.get("reference_wave") else None)
    return (compare(forehead_beats, recording["reference_beats"], forehead_wave, reference_wave),
            forehead_beats)


def _rate(beats: list[float]) -> str:
    gaps = sorted(b - a for a, b in zip(beats, beats[1:]) if 0.3 < b - a < 2.0)
    return f"{60.0 / gaps[len(gaps) // 2]:.0f} bpm" if gaps else "no rate"


def _sparkline(values) -> str:
    bars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(bars[min(7, int((v - lo) / span * 8))] for v in values)


def report(recording: dict, result: Comparison, forehead_beats: list[float]) -> str:
    reference_beats = recording["reference_beats"]
    a = result.agreement
    lines = [f"Recorded {recording['seconds']:.0f} s. Forehead: {len(forehead_beats)} beats, "
             f"{_rate(forehead_beats)}. Heart sensor: {len(reference_beats)} beats, {_rate(reference_beats)}.",
             "",
             "Agreement",
             f"  {len(a.pairs)} heartbeats seen at both sites.",
             f"  {len(a.forehead_only)} at the forehead only and {len(a.reference_only)} at the heart sensor "
             "only. A beat at one site only is most likely an artifact there.",
             ""]
    r = result.rescue
    lines.append("Forehead pulse, averaged around the heart sensor's beats")
    if r is None:
        lines.append("  Not enough beats to average.")
    else:
        lines += [f"  {_sparkline(r.shape)}   from {r.lags_s[0]:+.1f} s to {r.lags_s[-1]:+.1f} s around each beat",
                  f"  Averaged over {r.beats} beats, the pulse stands {r.height / r.noise:.0f} times above "
                  "what is left of the noise."]
    lines += ["", "Pulse height ratio, heart sensor over forehead, 1 = as at the start"]
    c = result.contrast
    if c is None:
        lines.append("  Needs the heart sensor's waveform, which only the USB cable sends.")
    else:
        verdict = ("lower: the pulse shrank at the heart sensor against the forehead. Small vessels in the "
                   "skin tightening would do that, and so would a sensor that moved"
                   if c.late < 0.9 else
                   "higher: the pulse grew at the heart sensor against the forehead. Small vessels in the "
                   "skin opening would do that, and so would a sensor that moved"
                   if c.late > 1.1 else "about the same as at the start")
        lines.append(f"  Over the last third: {c.late:.2f}, {verdict}.")
    lines += ["", "Delay, forehead to heart sensor"]
    d = result.delay
    if d is None:
        lines.append("  Not enough paired beats.")
    else:
        lines += [f"  Median {d.median_s * 1000:.0f} ms, spread {d.spread_s * 1000:.0f} ms, change across "
                  f"the recording {d.change_s * 1000:+.0f} ms.",
                  "  Each device is timed when its data arrives: the headband over Bluetooth, the heart "
                  f"sensor over {'Bluetooth' if recording['heart'] == 'ble' else 'its cable'}. The median "
                  "includes the difference in those delays, so only the change can mean anything."]
        if recording["heart"] == "ble":
            lines.append("  Over Bluetooth the heart sensor reports each beat after it has found it, so the "
                         "delay also carries the sensor's own processing time.")
    lines += ["", "The heart sensor clips to the ear, so this compares two sites on the head. The height ratio "
              "and the delay move less than they would between a finger and the head."]
    return "\n".join(lines)


def save(recording: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(recording, fh)


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--heart", choices=("usb", "ble"), default="usb",
                        help="the HeartMath over its USB cable (with waveform) or over Bluetooth")
    parser.add_argument("--address", help="the HeartMath's Bluetooth address, for --heart ble")
    parser.add_argument("--replay", type=Path, help="compare a saved recording instead of recording")
    parser.add_argument("--out", type=Path, help="where to save the recording")
    args = parser.parse_args(argv)

    if args.replay:
        recording = load(args.replay)
    else:
        recording = record(args.seconds, args.heart, args.address)
        out = args.out or ROOT / "sessions" / f"heart-pair-{time.strftime('%Y%m%d-%H%M%S')}.json.gz"
        save(recording, out)
        print(f"saved to {out}")
        for device, error in recording["errors"].items():
            if error:
                print(f"{device}: {error}")
    if not recording["frames"] or not recording["reference_beats"]:
        print("Nothing to compare: one of the devices sent nothing.")
        return 1
    result, forehead_beats = analyse(recording)
    print()
    print(report(recording, result, forehead_beats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
