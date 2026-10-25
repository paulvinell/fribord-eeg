"""Command-line entry points: probe, capture, impedance."""

from __future__ import annotations

import argparse
import time

from .protocol import BAUD, CMD, DeviceError, EVENT_NAME, FLOW_RTSCTS, NOTIF_CLEAR, NOTIF_ENABLE, NOT_DATA, ProtocolError

from .tables import TEST_RANGE_DEFAULT, TEST_RATE_HZ, rate_hz, test_scale

from .config import build_eeg_config, channel_scale, describe_config

from .stream import decode_data, plan_channels

from .device import Optima


def probe(port: str | None = None) -> None:
    """Smallest useful test: confirm transport, framing and opcodes in one shot.

    A well-formed REP_GET_DEV_INFO validates the whole reconstruction at once. If this
    times out, suspect the serial parameters before the framing — try flipping rtscts,
    then other baud rates.
    """
    with Optima(port, log=True) as dev:
        print(f"port {dev.port} at {BAUD} 8N1 rtscts={FLOW_RTSCTS}\n")
        try:
            code, name = dev.device_state()
            print(f"device state: {code} ({name})")
        except (TimeoutError, ProtocolError, DeviceError) as e:
            print(f"device state: FAILED — {e}")
        for label, fn in (
            ("device info", dev.device_info),
            ("firmware info", dev.firmware_info),
            ("config blob", dev.read_config),
        ):
            try:
                data = fn()
            except (TimeoutError, ProtocolError, DeviceError) as e:
                note = ""
                if fn is dev.read_config:
                    note = ("  (note: the vendor driver never sends READ_CFG, so firmware "
                            "support for it is unproven — build a config instead)")
                print(f"{label}: FAILED — {e}{note}")
                continue
            print(f"{label}: {len(data)} bytes")
            print(f"  hex   {data.hex(' ')}")
            printable = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
            print(f"  ascii {printable}")
        if dev.events:
            print(f"\nunsolicited: {dev.events}")


def capture(port: str | None = None, seconds: float = 10.0, out: str | None = None,
            cfg: bytes | None = None) -> None:
    """Start measurement and dump notification frames, decoding NOT_DATA where possible.

    Reads the device's config to learn the channel interleave. READ_CFG is never sent by the
    vendor's driver, so if the firmware rejects it, pass a config built with set_field()
    instead — decoding is impossible without knowing each channel's rate.
    """
    with Optima(port) as dev:
        if cfg is None:
            try:
                cfg = dev.read_config()
            except (TimeoutError, ProtocolError, DeviceError) as e:
                print(f"READ_CFG failed ({e}) — the vendor driver never issues it, so "
                      "firmware support is unproven; writing a default EEG config instead")
                cfg = build_eeg_config()
                dev.write_config(cfg)
        plan = plan_channels(cfg)
        print(describe_config(cfg))
        for c in plan:
            if c.enabled:
                print(f"  ch{c.index} interleave mask 0x{c.mask:02x} "
                      f"({rate_hz(c.rate_exponent):g} Hz)")
        dev.set_notifications(NOTIF_ENABLE)
        dev.start_measurement()
        print(f"streaming for {seconds}s — Ctrl-C to stop early")
        frames, deadline, first = [], time.monotonic() + seconds, None
        try:
            for msg in dev.stream():
                if first is None:
                    first = time.monotonic()   # exclude start-up latency from the rate
                frames.append(msg)
                if time.monotonic() > deadline:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            try:
                dev.stop_measurement()
                dev.set_notifications(NOTIF_CLEAR)
            except Exception as e:
                print(f"warning: teardown failed — {e}")

    data = [f for f in frames if f.code == NOT_DATA]
    print(f"{len(frames)} frames, {len(data)} NOT_DATA")
    if data:
        sizes = sorted({len(f.payload) for f in data})
        print(f"payload sizes: {sizes}")
        phase = None
        totals: dict[int, int] = {}
        gaps = 0
        for f in data:
            try:
                df = decode_data(f.payload, plan, phase)
            except ProtocolError as e:
                print(f"  decode failed: {e}")
                break
            if phase is not None and df.phase != phase:
                gaps += 1
            phase = df.next_phase
            for ch, vals in df.samples.items():
                totals[ch] = totals.get(ch, 0) + len(vals)
        span = (time.monotonic() - first) if first else 0
        for ch, n in sorted(totals.items()):
            rate = n / span if span else 0
            print(f"  ch{ch}: {n} samples, {rate:.1f} Hz")
        print(f"phase discontinuities: {gaps}")
        first = decode_data(data[0].payload, plan)
        print(f"first frame: phase={first.phase} sum_state={first.sum_state} "
              f"channel_states={first.channel_states}")
        for ch, vals in first.samples.items():
            try:
                factor, unit = channel_scale(cfg, ch)
                shown = ", ".join(f"{v * factor:.3g}" for v in vals[:8])
                print(f"  ch{ch}: {shown} {unit}")
            except ValueError:
                print(f"  ch{ch}: {vals[:8]} (raw counts)")
    if out:
        with open(out, "w") as fh:
            for f in frames:
                fh.write(f"{f.code:02x} {f.payload.hex()}\n")
        print(f"wrote {out}")


def _kohm(value: float | None) -> str:
    return "   ?  " if value is None else f"{value:6.1f}"


def impedance(port: str | None = None, seconds: float = 10.0,
              range_kohm: float = TEST_RANGE_DEFAULT) -> None:
    """Run the electrode impedance / input-continuity test and report kOhm per channel.

    Same NOT_DATA stream as a measurement, but at TEST_RATE_HZ with no decimation and only
    channels whose function is < 4 participating. Each frame carries exactly one value per
    channel and advances the phase by one.

    A *voltage* channel has two electrode impedances to report (the "+" and "-" inputs), while a
    non-voltage channel has one (the resistance between its inputs). Only one value per channel
    arrives per frame, and the two electrodes alternate — **verified on hardware**, and which is
    which along with it: even phases carry "+", odd phases carry "-". Reported as a pair here
    rather than as a single newest reading, which is what made a stable pair of electrodes look
    like one flickering number.

    How it was established, with one electrode on ch0's "+" and a single reference: on even frames
    the three unwired channels read exactly full scale (open circuit) while ch0 read its own
    electrode; on odd frames all four agreed to within 0.3 kOhm, because `reference="linked"`
    joins the "-" inputs inside the device so they are all reading the same reference electrode.
    Three fresh START_TEST runs each began at phase 0, which is what makes the parity absolute
    rather than relative to whenever the test happened to start.

    Readings take a second or two to settle, and the manual notes the test injects a 31.25 Hz
    signal that can appear in voltage measurements.
    """
    with Optima(port) as dev:
        cfg = dev.read_config()
        plan = plan_channels(cfg, test_mode=True)
        active = [c.index for c in plan if c.enabled]
        print(f"testing channels {active} at {TEST_RATE_HZ} Hz, "
              f"{range_kohm:g} kOhm range ({test_scale(range_kohm):.6g} kOhm/count)")
        dev.set_notifications(NOTIF_ENABLE)
        dev.request(CMD["START_TEST"])
        latest: dict[int, list[float | None]] = {}
        deadline = time.monotonic() + seconds
        try:
            for msg in dev.stream():
                if msg.code != NOT_DATA:
                    print(f"  event: {EVENT_NAME.get(msg.payload[0], hex(msg.payload[0]))}")
                    continue
                df = decode_data(msg.payload, plan)
                # the phase the last sample sits at decides which electrode it is; even is "+"
                side = 0 if (df.next_phase - 1) % 2 == 0 else 1
                for ch, vals in df.samples.items():
                    if vals:
                        latest.setdefault(ch, [None, None])[side] = vals[-1] * test_scale(range_kohm)
                if latest:
                    print("  " + "  ".join(
                        f"ch{c}=+{_kohm(v[0])}/-{_kohm(v[1])}" for c, v in sorted(latest.items())))
                if time.monotonic() > deadline:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            try:
                dev.request(CMD["STOP_TEST"])
                dev.set_notifications(NOTIF_CLEAR)
            except Exception as e:
                print(f"warning: teardown failed — {e}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="optima", description=__doc__)
    ap.add_argument("action", choices=("probe", "capture", "impedance"))
    ap.add_argument("--port", help="serial port (default: first FTDI port found)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", help="write captured frames here")
    args = ap.parse_args(argv)
    if args.action == "probe":
        probe(args.port)
    elif args.action == "impedance":
        impedance(args.port, args.seconds)
    else:
        capture(args.port, args.seconds, args.out)
