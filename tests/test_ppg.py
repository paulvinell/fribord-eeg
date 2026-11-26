"""The HeartMath device's own protocol, and the beats we find in it.

The Bluetooth path takes intervals the device has already worked out. This path takes the raw pulse
waveform and finds the beats here, so the detector is ours and has to be held to a number: a synthetic
waveform with known beat times, scored for misses and false positives. The two failure modes below —
counting the dicrotic notch, and locking out when the pulse shrinks — are the ones that were measured
and fixed, and each has a test so a tuning change cannot quietly undo it.
"""

from __future__ import annotations

import math
import random

from nfb.hrv.beats import Beat
from nfb.hrv.ppg import (
    ALLON_BODY_LEN, BeatFinder, BIOMETRICS_ALLON, HEADER, PACKET_BIOMETRICS, PPG_RATE_HZ, QUERY_BATTERY,
    START_PPG, STOP_PPG, FrameSplitter, PeakDetector, PpgDecoder, battery_level, command, crc,
    parse_ppg,
)

from harness import ListedPort

FS = PPG_RATE_HZ


# --- the wire format -------------------------------------------------------------------------

def test_the_commands_are_the_bytes_confirmed_on_hardware():
    """These three were confirmed against hardware. If a
    change to the frame builder moves any of them, it has broken the protocol, not the tests."""
    assert START_PPG == bytes.fromhex("AB020208F4")
    assert STOP_PPG == bytes.fromhex("AB020205F7")
    assert QUERY_BATTERY == bytes.fromhex("AB020701F6")


def ppg_frame(samples: list[int]) -> bytes:
    body = bytes([PACKET_BIOMETRICS, BIOMETRICS_ALLON])
    for s in samples:
        body += bytes([(s >> 8) & 0xFF, s & 0xFF])
    with_len = bytes([ALLON_BODY_LEN]) + body
    return bytes([HEADER]) + with_len + bytes([crc(with_len)])


def test_a_pulse_frame_decodes_big_endian_and_keeps_every_sample():
    """Reading one sample of the sixteen little-endian one byte along gives
    the same number for a pair in the middle of the buffer and a different one at the end — where the
    little-endian read would take in the CRC. Sixteen samples, big-endian, is the reading that holds
    everywhere."""
    values = [1362 + 7 * i for i in range(16)]
    frame = ppg_frame(values)
    assert parse_ppg(frame) == values


def test_anything_that_is_not_a_pulse_frame_comes_back_as_none():
    assert parse_ppg(command(0x03, 0xA1, b"\x00")) is None      # an acknowledgement
    assert parse_ppg(START_PPG) is None
    assert parse_ppg(b"") is None


def test_a_battery_reply_is_read_and_a_nonsense_one_is_not():
    assert battery_level(command(0x07, 0x01, bytes([100]))) == 100
    assert battery_level(command(0x07, 0x01, bytes([200]))) is None
    assert battery_level(START_PPG) is None


def test_the_splitter_survives_arbitrary_chunking_and_a_corrupt_frame():
    """A serial read returns whatever was in the operating system's buffer, so frames arrive split and
    glued. A bad CRC costs its own frame and nothing else."""
    good = [ppg_frame([i] * 16) for i in range(3)]
    corrupt = bytearray(ppg_frame([9] * 16))
    corrupt[-1] ^= 0xFF
    stream = good[0] + bytes(corrupt) + good[1] + good[2]

    whole = FrameSplitter()
    assert whole.feed(stream) == [good[0], good[1], good[2]]
    assert whole.bad_crc == 1

    byte_at_a_time = FrameSplitter()
    out = []
    for i in range(len(stream)):
        out += byte_at_a_time.feed(stream[i:i + 1])
    assert out == [good[0], good[1], good[2]]
    assert byte_at_a_time.bad_crc == 1


def test_junk_before_a_header_is_dropped_rather_than_ending_the_stream():
    splitter = FrameSplitter()
    frame = ppg_frame([100] * 16)
    assert splitter.feed(b"\x00\x11\x22" + frame) == [frame]
    assert splitter.dropped_bytes == 3


# --- the detector ----------------------------------------------------------------------------

def waveform(seconds: float = 60.0, hr: float = 60.0, rsa_s: float = 0.06, amp: float = 400.0,
             amp_at=None, dicrotic: float = 0.0, drift: float = 0.0, noise: float = 2.0,
             seed: int = 1) -> tuple[list[int], list[float]]:
    """A synthetic pulse waveform and the beat times that generated it.

    `dicrotic` adds the secondary pressure peak 0.35 s after each beat, which is the thing a fixed
    refractory period counts as a second beat. `amp_at(t)` scales the pulse height, for the clip coming
    loose partway through.
    """
    rng = random.Random(seed)
    beats: list[float] = []
    t = phase = 0.0
    while t < seconds:
        rr = max(0.3, 60.0 / hr + rsa_s * math.sin(2 * math.pi * 0.1 * phase))
        phase += rr
        t += rr
        beats.append(t)

    out: list[int] = []
    for i in range(int(seconds * FS)):
        t = i / FS
        v = 2000.0 + (drift * math.sin(2 * math.pi * 0.05 * t) if drift else 0.0)
        a = amp * (amp_at(t) if amp_at else 1.0)
        for b in beats:
            d = t - b
            if -0.05 < d < 0.6:
                v += a * math.exp(-(d ** 2) / (2 * 0.045 ** 2))
                if dicrotic:
                    v += a * dicrotic * math.exp(-((d - 0.35) ** 2) / (2 * 0.06 ** 2))
        out.append(int(max(0, min(65535, v + rng.gauss(0.0, noise)))))
    return out, beats


def heartmath_waveform(seconds: float = 60.0, hr: float = 62.0, noise: float = 6.0, seed: int = 1,
                       rate: float = FS) -> tuple[list[int], list[float]]:
    """A raw waveform shaped like the HeartMath's: each beat falls fast for about 0.45 s, then climbs
    slowly back to a rounded top, as the device's own pulse does when averaged around its beats. Returns
    the samples and, for each beat, the moment the fall crosses half its depth, which is when a beat
    timed at half height of its upright rise should be stamped."""
    rng = random.Random(seed)
    onsets, t = [], 0.6
    while t < seconds:
        onsets.append(t)
        t += 60.0 / hr + 0.05 * math.sin(2 * math.pi * 0.1 * t)

    def upright(d: float) -> float:
        if 0.0 <= d < 0.45:
            return 0.5 * (1.0 - math.cos(math.pi * d / 0.45))
        if d >= 0.45:
            return math.exp(-(d - 0.45) / 0.35)
        return 0.0

    clean = [sum(upright(i / rate - b) for b in onsets if -0.05 < i / rate - b < 2.5)
             for i in range(int(seconds * rate))]
    out = [int(round(3000.0 - 600.0 * v + rng.gauss(0.0, noise))) for v in clean]
    # The truth from the noise-free waveform itself, since each beat's decay is still fading when the next
    # starts: lowest point near the onset, highest within 0.6 s after it, the crossing between them.
    truth = []
    for b in onsets:
        lo, hi = int((b - 0.1) * rate), int((b + 0.2) * rate)
        if lo < 0 or hi + int(0.6 * rate) > len(clean):
            continue                    # too near either end to have its whole rise
        foot = min(range(lo, hi), key=lambda i: clean[i])
        top = max(range(foot, min(len(clean), foot + int(0.6 * rate))), key=lambda i: clean[i])
        half = (clean[foot] + clean[top]) / 2.0
        k = next(i for i in range(foot, top + 1) if clean[i] >= half)
        truth.append((k - 1 + (half - clean[k - 1]) / (clean[k] - clean[k - 1])) / rate)
    return out, truth


def score(samples: list[int], truth: list[float], detector=None,
          tol: float = 0.15) -> tuple[float, float]:
    """(missed, false positive) as fractions of the true beats."""
    detector = detector or PeakDetector(rate_hz=FS)
    found = [r[0] for i, s in enumerate(samples) if (r := detector.push(i / FS, s)) is not None]
    remaining = list(truth)
    matched = spurious = 0
    for at in found:
        near = min(remaining, key=lambda b: abs(b - at), default=None)
        if near is not None and abs(near - at) <= tol:
            remaining.remove(near)
            matched += 1
        else:
            spurious += 1
    return 1.0 - matched / len(truth), spurious / len(truth)


def test_a_clean_pulse_waveform_gives_back_the_beats_that_made_it():
    samples, truth = waveform(seconds=60.0, hr=62.0)
    missed, spurious = score(samples, truth)
    assert missed < 0.03, f"missed {missed:.1%}"
    assert spurious < 0.03, f"{spurious:.1%} spurious"


def test_the_dicrotic_notch_is_not_counted_as_a_beat():
    """The pulse waveform's secondary peak lands about 0.35 s after the main one — just outside a fixed
    0.30 s refractory period, so every beat gets counted twice. And the error sustains itself: doubling
    the rate halves the mean interval, which shrinks a proportional refractory, which keeps admitting the
    notch. Measured, the detector locked onto exactly double the true rate and never recovered."""
    samples, truth = waveform(seconds=60.0, hr=60.0, dicrotic=0.55)
    missed, spurious = score(samples, truth)
    assert spurious < 0.05, f"{spurious:.1%} spurious — the notch is being counted"
    assert missed < 0.05, f"missed {missed:.1%}"


def test_a_beat_is_timed_between_samples_at_a_low_sample_rate():
    """At 30 samples a second the highest sample can be 16 ms from the true peak, and beat intervals
    come out as multiples of 33 ms. A parabola through the top three samples places each beat to a few
    milliseconds, so a slow optical sensor still gives intervals the heart actually produced."""
    fs = 30.0
    rng = random.Random(7)
    truth, at = [], 1.0
    while at < 60.0:
        at += 0.9 + 0.2 * rng.random()
        truth.append(at)
    samples = [sum(1000.0 * math.exp(-((i / fs - b) ** 2) / (2 * 0.12 ** 2)) for b in truth
                   if abs(i / fs - b) < 0.6) for i in range(int(60.0 * fs))]
    detector = PeakDetector(rate_hz=fs)
    found = [r[0] for i, s in enumerate(samples) if (r := detector.push(i / fs, s)) is not None]
    errors = [min(abs(f - b) for b in truth) for f in found]
    assert len(found) >= len(truth) - 2
    assert sum(errors) / len(errors) < 0.004, "beats are placed between samples, not on the grid"


def test_the_detector_recovers_when_the_pulse_gets_smaller():
    """The amplitude floor is a fraction of the height of recently accepted beats, and rejected peaks
    never update it — so a clip coming loose could lock the detector out permanently. Measured, a step
    down to 30% lost every subsequent beat. The floor decays once a beat is overdue, so it re-acquires."""
    samples, truth = waveform(seconds=60.0, hr=60.0, amp_at=lambda t: 1.0 if t < 30.0 else 0.3)
    late = [b for b in truth if b > 35.0]
    detector = PeakDetector(rate_hz=FS)
    found = [r[0] for i, s in enumerate(samples) if (r := detector.push(i / FS, s)) is not None]
    caught = sum(1 for b in late if any(abs(f - b) <= 0.15 for f in found))
    assert caught / len(late) > 0.9, f"only {caught}/{len(late)} beats after the drop"


def test_baseline_drift_does_not_move_the_beats():
    """300 counts of 0.05 Hz wander against a 400-count pulse, which is the figure the detector was
    tuned against. It is not unbounded: the baseline average lags by roughly its own time constant
    times the slope, so drift steep enough to move the baseline by a pulse height within 1.5 s does
    start costing beats."""
    samples, truth = waveform(seconds=60.0, hr=60.0, drift=300.0)
    missed, spurious = score(samples, truth)
    assert missed < 0.05 and spurious < 0.05


def test_a_flat_or_noise_only_trace_makes_no_beats():
    """A sensor off the body, disconnected, or saturated gives a constant or near-constant trace. The
    relative gates calibrate to it and manufacture beats at the refractory cadence — 10 s of flat input
    gave about 26 — which then enter heart-rate training. The signal-to-noise gate rejects a trace whose
    peaks do not stand clear of its own jitter, so there is no pulse and no beat."""
    detector = PeakDetector(rate_hz=FS)
    flat = [detector.push(i / FS, 2000) for i in range(int(10 * FS))]
    assert all(r is None for r in flat), "a constant trace has no beats"

    rng = random.Random(7)
    detector = PeakDetector(rate_hz=FS)
    noise = [detector.push(i / FS, 2000 + rng.randint(-1, 1)) for i in range(int(10 * FS))]
    assert all(r is None for r in noise), "a trace of one-count noise has no beats"


def test_a_weak_but_real_pulse_still_gives_beats():
    """The noise gate must not reject a living pulse. A quarter-height pulse (100 counts against the
    detector's 400-count reference) stands far above the noise floor, so every beat is kept."""
    samples, truth = waveform(seconds=60.0, hr=60.0, amp=100.0)
    missed, spurious = score(samples, truth)
    assert missed < 0.05 and spurious < 0.05, f"missed {missed:.1%}, {spurious:.1%} spurious"


# --- timing each beat --------------------------------------------------------------------------

def _found(samples, finder_or_detector):
    out = []
    for i, x in enumerate(samples):
        got = finder_or_detector.push(i / FS, x)
        out += [b.t for b in got] if isinstance(got, list) else ([got[0]] if got else [])
    return out


def test_a_heartmath_beat_is_timed_where_its_fall_crosses_half_its_depth():
    samples, truth = heartmath_waveform(seconds=40.0)
    errors = sorted(min(abs(b - at) for at in truth) for b in _found(samples, BeatFinder(FS)))
    assert errors[len(errors) // 2] < 0.003


def test_half_height_intervals_are_steadier_than_the_raw_maximum():
    """The detector's own stamp is the highest raw sample, which on the HeartMath's pulse is a rounded
    top that noise moves; on two real recordings it roughly doubled the beat-to-beat variation the
    intervals showed. Half height sits on the steep fall, where noise moves the crossing least."""
    samples, truth = heartmath_waveform(seconds=90.0, noise=25.0)
    true_gaps = dict(zip(truth[1:], [b - a for a, b in zip(truth, truth[1:])]))

    def interval_errors(times):
        offset = sorted(t - min(truth, key=lambda a: abs(t - a)) for t in times)[len(times) // 2]
        out = []
        for a, b in zip(times, times[1:]):
            at = min(truth, key=lambda x: abs(b - offset - x))
            if at in true_gaps and abs((b - a) - true_gaps[at]) < 0.3:
                out.append(abs((b - a) - true_gaps[at]))
        return sorted(out)[len(out) // 2]

    finder = interval_errors(_found(samples, BeatFinder(FS)))
    stamps = interval_errors(_found(samples, PeakDetector(rate_hz=FS)))
    assert finder < stamps / 2, f"half height {finder * 1000:.1f} ms against the raw maximum {stamps * 1000:.1f} ms"


# --- the decoder -----------------------------------------------------------------------------

def test_beats_come_out_of_a_byte_stream_at_the_rate_that_went_in():
    """End to end without a device: build the waveform, frame it as the device would, and feed it in
    chunks that do not respect frame boundaries."""
    samples, truth = heartmath_waveform(seconds=60.0, hr=66.0)
    decoder = PpgDecoder(warm_up_samples=0)
    stream = b"".join(ppg_frame(samples[i:i + 16]) for i in range(0, len(samples) - 15, 16))

    beats: list[Beat] = []
    t = 1000.0
    step = 700
    for i in range(0, len(stream), step):
        chunk = stream[i:i + step]
        # Sixteen samples per frame at 500 Hz; a chunk of `step` bytes is that many frames' worth.
        t += len(chunk) / (3 + ALLON_BODY_LEN) * 16 / FS
        beats += decoder.feed(chunk, t)

    assert decoder.bad_crc == 0
    rate = 60.0 / (sum(b.rr_intervals[0] for b in beats) / len(beats))
    # Two beats while the detector settles, one with no interval before it, one still rising at the end.
    assert len(beats) >= len(truth) - 5, f"{len(beats)} beats against {len(truth)}"
    assert abs(rate - 66.0) < 1.5, f"{rate:.1f} bpm"
    assert all(b.contact is None for b in beats), "a pulse sensor has no contact flag to report"
    # A beat is timed after its rise has passed, so it is given later than it happened, and it says when.
    assert all(b.arrived is not None and 0.3 < b.arrived - b.t < 1.5 for b in beats)


def test_sample_times_come_from_arrivals_so_a_wrong_rate_cannot_drift():
    """The spacing is derived from when chunks actually arrive, not from the nominal rate — which is what
    keeps a slightly wrong constant from accumulating. The old 312 Hz figure stretched the timeline and
    reported 37 bpm for a 60 bpm signal, so this feeds a real 500 Hz stream to a decoder told 400."""
    samples, _ = heartmath_waveform(seconds=40.0, hr=60.0)
    decoder = PpgDecoder(rate_hz=400.0, warm_up_samples=0)
    beats: list[Beat] = []
    t = 1000.0
    for i in range(0, len(samples) - 15, 16):
        t += 16 / FS
        beats += decoder.feed(ppg_frame(samples[i:i + 16]), t)
    rate = 60.0 / (sum(b.rr_intervals[0] for b in beats) / len(beats))
    assert abs(rate - 60.0) < 1.5, f"{rate:.1f} bpm from a decoder told the wrong rate"


def test_a_gap_in_the_stream_re_anchors_instead_of_smearing_samples_across_it():
    """A stall or a reconnect leaves a hole no spacing bridges honestly. Sixteen samples must not be
    spread across ten seconds of silence, because the detector would then see intervals no heart has."""
    decoder = PpgDecoder(warm_up_samples=0)
    frame = ppg_frame(list(range(16)))
    decoder.feed(frame, 1000.0)
    decoder.feed(frame, 1010.0)                 # ten seconds later
    stamps = [t for t, _ in decoder.raw][-16:]
    assert stamps[-1] == 1010.0
    assert abs((stamps[-1] - stamps[0]) - 15 / FS) < 1e-9, "the run is anchored at the arrival"


def test_the_warm_up_samples_are_dropped():
    decoder = PpgDecoder(warm_up_samples=32)
    for i in range(4):
        decoder.feed(ppg_frame([500] * 16), 1000.0 + i * 16 / FS)
    assert len(decoder.raw) == 2 * 16, "the first two frames are settle time"


# --- finding the cable -----------------------------------------------------------------------

def test_the_amplifier_is_never_offered_as_a_pulse_sensor(monkeypatch):
    """**Both devices are FTDI serial bridges on /dev/cu.usbserial-***, so a vendor-id match alone would
    sometimes hand the heart screen the amplifier — and opening a serial port is exclusive, so that is
    the console losing its EEG device and writing pulse-sensor commands at it. The product string is the
    one thing that separates them."""
    from nfb.hrv import sources

    ports = [ListedPort("/dev/cu.usbserial-FT000001", 0x0403, "Neurobit Optima+ 4 USB"),
             ListedPort("/dev/cu.usbserial-HM01", 0x0403, "FT230X Basic UART"),
             ListedPort("/dev/cu.Bluetooth-Incoming-Port", None, "n/a")]
    monkeypatch.setattr(sources, "usb_candidates",
                        lambda: [p for p in ports if p.vid == 0x0403 and not sources.is_amplifier(p)])
    port, trouble = sources.find_usb_port()
    assert port == "/dev/cu.usbserial-HM01" and trouble is None


def test_the_missing_cable_message_names_what_is_on_usb_and_nothing_else(monkeypatch):
    """A Mac lists serial ports that are nothing of the kind — a debug console, an incoming-Bluetooth
    port, every set of headphones ever paired. The first version of this message recited all four, which
    wrapped to two lines and said nothing about whether a cable was plugged in."""
    from nfb.hrv import sources

    monkeypatch.setattr(sources, "usb_candidates", lambda: [])
    from serial.tools import list_ports

    monkeypatch.setattr(list_ports, "comports",
                        lambda: [ListedPort("/dev/cu.Bluetooth-Incoming-Port", None, "n/a"),
                                 ListedPort("/dev/cu.JBLTUNE570BT", None, "n/a"),
                                 ListedPort("/dev/cu.usbserial-FT000001", 0x0403,
                                          "Neurobit Optima+ 4 USB")])
    port, trouble = sources.find_usb_port()
    assert port is None
    assert "JBL" not in trouble and "Bluetooth-Incoming" not in trouble
    assert "Neurobit Optima+ 4 USB" in trouble, "what is on USB is worth naming"
    assert "skipped on purpose" in trouble, "including why the one that is there was not taken"


def test_two_cables_it_could_be_means_asking_rather_than_guessing(monkeypatch):
    from nfb.hrv import sources

    ports = [ListedPort("/dev/cu.usbserial-A", 0x0403, "FT230X Basic UART"),
             ListedPort("/dev/cu.usbserial-B", 0x0403, "FT231X USB UART")]
    monkeypatch.setattr(sources, "usb_candidates", lambda: ports)
    port, trouble = sources.find_usb_port()
    assert port is None
    assert "name the port" in trouble and "/dev/cu.usbserial-B" in trouble


# --- starting the cable ----------------------------------------------------------------------

def test_a_cable_that_does_not_start_is_asked_again_and_said_to_be_silent(monkeypatch):
    """**Sometimes the device simply does not start.** It acknowledges the start command and then takes
    about 1.8 s to begin streaming, so anything disturbing it in that window leaves an open port, a live
    reader and no data — which read as an indefinite "connected, waiting for beats". Asking again costs
    five bytes. And the three states behind that one phrase are now told apart, because they want
    different things: nothing on the cable, a waveform with no pulse in it, and beats."""
    from nfb.hrv import sources
    from nfb.hrv.ppg import START_PPG

    written: list[bytes] = []

    class Silent:
        def __init__(self, *a, **kw):
            pass

        def reset_input_buffer(self):
            pass

        def write(self, data):
            written.append(bytes(data))

        def flush(self):
            pass

        def close(self):
            pass

        def read(self, _n):
            reads.append(1)
            if len(reads) == 6:                  # six empty reads is enough to have asked again
                said.append(source.progress())
                source.close()
            return b""

    reads: list[int] = []
    said: list[str] = []
    monkeypatch.setattr("serial.Serial", Silent)
    source = sources.UsbPpgBeats(port="/dev/null-ish")
    # Retry on every empty read, so the loop's own iterations drive this rather than a wall clock: the
    # thing under test is "it asks again", not "it asks again after two seconds".
    monkeypatch.setattr(type(source), "START_RETRY_S", 0.0)
    assert list(source.beats()) == [], "a silent cable yields nothing"
    assert written.count(START_PPG) > 2, "it kept asking the device to start"
    assert said and "no data on the cable" in said[0], "and said so rather than 'waiting for beats'"


def test_the_three_silences_are_told_apart():
    """A port that is opening, a cable that sends nothing, a waveform with no beats in it and a waveform
    with beats each have their own words."""
    from nfb.hrv import sources

    source = sources.UsbPpgBeats(port="/dev/x")
    said = [source.progress()]
    source.connected = True
    said.append(source.progress())
    source.decoder.feed(ppg_frame([2000] * 16), 1000.0)
    said.append(source.progress())
    source.beats_found = 3
    said.append(source.progress())
    assert all(said) and len(set(said)) == len(said), said
    assert "no pulse" in said[2] and "no pulse" not in said[3], "the beats found end the no-pulse words"
