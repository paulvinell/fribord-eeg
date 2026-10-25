"""Command line: record a session, or summarise/replay a recorded one."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import optima

from .sources import OptimaSource, ReplaySource
from .stream import Block, Gap


def _summarise(source, seconds: float | None = None) -> None:
    chans = source.channels()
    for spec in chans.values():
        print(f"  ch{spec.index}: {spec.function} {spec.rate_hz:g}Hz "
              f"{spec.scale:.6g} {spec.unit}/count")
    counts: dict[int, int] = {}
    first: dict[int, int] = {}
    gaps = 0
    t0 = time.monotonic()
    for item in source.blocks():
        if isinstance(item, Gap):
            gaps += 1
            print(f"  GAP: {item.groups_lost} groups lost")
            continue
        counts[item.channel] = counts.get(item.channel, 0) + len(item.samples)
        first.setdefault(item.channel, item.start_index)
        if seconds is not None and time.monotonic() - t0 > seconds:
            break
    span = time.monotonic() - t0
    print(f"\n{sum(counts.values())} samples in {span:.1f}s, {gaps} gaps")
    for ch in sorted(counts):
        spec = chans[ch]
        print(f"  ch{ch}: {counts[ch]} samples from index {first[ch]} "
              f"({counts[ch] / span:.1f} Hz observed, {spec.rate_hz:g} configured)")


def _port_holder(port: str) -> str:
    """The pid of another process holding `port`, or "" — best effort, via lsof."""
    import os
    import subprocess

    try:
        out = subprocess.run(["lsof", "-t", port], capture_output=True, text=True, timeout=5.0)
    except (OSError, subprocess.SubprocessError):
        return ""
    mine = str(os.getpid())
    others = [line for line in out.stdout.split() if line and line != mine]
    return others[0] if others else ""


def _probe_heartmath(seconds: float, port: str | None = None) -> None:
    """Say what the pulse sensor is actually doing, in the three layers it can fail in.

    "Connected, waiting for beats" is one message covering three very different states, and the screen
    cannot tell them apart from where it stands: nothing on the wire, a waveform with no pulse in it
    (the clip not on an ear), or a waveform we are failing to find beats in. This separates them —
    frames, sample range, beats — so the answer is a line of output rather than a guess.
    """
    from .hrv.sources import UsbPpgBeats, find_usb_port

    if port is None:
        port, trouble = find_usb_port()
        if port is None:
            raise SystemExit(f"nfb: {trouble}")
    # **Refuse to open a port somebody else is reading**, and this is not politeness. macOS lets a second
    # process open a /dev/cu.* device that is already open: the reads then return nothing ("multiple
    # access on port"), which looks exactly like a dead device — but the *writes* go through, so this
    # command's own start and stop commands land on a sensor the console is mid-session on, and stop its
    # stream. Measured, doing precisely that to a running console.
    holder = _port_holder(port)
    if holder:
        raise SystemExit(
            f"nfb: {port} is open in another process (pid {holder}) — almost certainly the console.\n"
            f"Quit it first: two readers of one serial port means neither gets the data, and the "
            f"commands this sends would interrupt the other one's session.")
    source = UsbPpgBeats(port=port)
    print(f"opening {port} · reading for {seconds:g}s · put the sensor on an ear or a finger")
    started = time.monotonic()
    beats: list = []
    # On a timer rather than in the loop: `beats()` yields only when a beat is found, and a sensor with
    # nothing clipped to it never yields at all — which is the case this command exists to diagnose.
    threading.Timer(seconds, source.close).start()
    for beat in source.beats():
        beats.append(beat)
        print(f"  beat at +{time.monotonic() - started:5.1f}s  "
              f"{beat.hr_bpm:5.1f} bpm  interval {beat.rr_intervals[0]:.3f}s")
    source.close()
    span = max(1e-9, time.monotonic() - started)
    decoder = source.decoder
    samples = decoder._seen
    print(f"\n{samples} samples in {span:.1f}s ({samples / span:.0f} Hz), "
          f"{decoder.bad_crc} frames with a bad checksum, {len(beats)} beats")
    if source.error:
        print(f"  {source.error}")
    if not samples:
        print("  nothing arrived — the cable is open but the device is not streaming. Switch it on,\n"
              "  and check no other program has it.")
        return
    values = [v for _t, v in decoder.raw]
    if values:
        low, high = min(values), max(values)
        print(f"  raw pulse {low}-{high} (swing {high - low})")
        if high - low < 50:
            print("  a flat waveform: the sensor is streaming but nothing is on it. Clip it on.")
    if beats:
        rates = [b.hr_bpm for b in beats]
        print(f"  {sum(rates) / len(rates):.1f} bpm mean, {min(rates):.0f}-{max(rates):.0f} range")
    elif values and high - low >= 50:
        print("  a moving waveform with no beats found — that is a detector problem, not a cable one.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nfb", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="record a live session to a file")
    rec.add_argument("out", help="session file (.jsonl or .jsonl.gz)")
    rec.add_argument("--seconds", type=float, default=30.0)
    rec.add_argument("--port")
    rec.add_argument("--channels", type=int, default=4)
    rec.add_argument("--mains", type=int, default=50, choices=(50, 60),
                     help="mains frequency the device notches out")
    rec.add_argument("--reference", default="linked",
                     choices=sorted(optima.REFERENCE_MODES),
                     help="linked: one shared reference (default). bipolar: independent pairs.")
    rec.add_argument("--hz", type=float, default=1000.0,
                 help="sample rate; 1000 is the default for low latency")
    rec.add_argument("--high-pass", dest="high_pass", type=float, default=0.5, choices=(0.5, 0.01),
                     help="the device's high-pass corner; 0.01 for ultra-low-frequency work")
    rec.add_argument("--note", default="")
    rec.add_argument("--force", action="store_true",
                     help="overwrite the output file if it already exists")

    rep = sub.add_parser("replay", help="replay a recorded session")
    rep.add_argument("path")
    rep.add_argument("--realtime", action="store_true")

    hm = sub.add_parser("heartmath", help="probe the HeartMath pulse sensor over its USB cable")
    hm.add_argument("--seconds", type=float, default=15.0)
    hm.add_argument("--port", help="serial port; found automatically if omitted")

    args = ap.parse_args(argv)
    if args.cmd == "heartmath":
        _probe_heartmath(args.seconds, args.port)
        return
    if args.cmd == "record":
        # **Refuse to destroy a recording.** A capture is minutes of somebody's head that cannot be taken
        # again, and the default output has always been one fixed path — so every `make record` silently
        # overwrote the last one, including a 60 s infra-low capture that several measurements depended on.
        # Erroring costs one flag; the alternative costs the data.
        out = Path(args.out)
        if out.exists() and not args.force:
            raise SystemExit(
                f"{out} already exists ({out.stat().st_size / 1024:.0f} kB, "
                f"modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(out.stat().st_mtime))}).\n"
                f"Recording would destroy it. Choose another name, or pass --force to overwrite."
            )
        cfg = optima.build_eeg_config(args.channels, hz=args.hz, reference=args.reference,
                                      mains_hz=args.mains, lower_limit_hz=args.high_pass)
        with OptimaSource(cfg, port=args.port, record_to=args.out, note=args.note) as src:
            print(f"recording to {args.out}")
            _summarise(src, args.seconds)
    else:
        with ReplaySource(args.path, realtime=args.realtime) as src:
            h = src.header
            print(f"{h.device} serial {h.serial} firmware {h.firmware} started {h.started_at}")
            if h.note:
                print(f"note: {h.note}")
            _summarise(src)
