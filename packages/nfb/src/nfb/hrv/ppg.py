"""The HeartMath device's own protocol: raw PPG in, beats out.

`sources.BleBeats` reads the standard Bluetooth Heart Rate Service, where the device has already found
the beats and sends one interval at a time. This module reads the **vendor** stream instead — the same
0xAB-framed protocol the Inner Balance app uses — which carries the sensor's raw pulse waveform at
500 Hz, sixteen samples per frame, about thirty frames a second. Finding the beats in it is then our
job, which is what `PeakDetector` is for.

Every claim here was checked against a real HM-FK424.

Decoding only. The transport that pumps bytes into `PpgDecoder` lives in `sources.py`, the same split
as `nfb.heg.proto` against `nfb.heg.source`, and for the same reason: everything in this file can be
tested to the byte with no device attached.
"""

from __future__ import annotations

import math
from collections import deque

from .beats import Beat

HEADER = 0xAB
"""Every frame starts with this, in both directions."""

PACKET_BIOMETRICS = 0x02
BIOMETRICS_ALLON = 0x08         # start streaming raw PPG, all channels
BIOMETRICS_STOPALL = 0x05
PACKET_BATTERY = 0x07
BATTERY_LEVEL = 0x01

ALLON_BODY_LEN = 34
"""The length byte of a raw-PPG frame: two bytes of type, thirty-two of samples. Total wire length is
this plus three — the header, the length byte itself and the CRC."""

SAMPLES_PER_FRAME = 16

PPG_RATE_HZ = 500.0
"""What the sensor samples at, and what both transports deliver.

Nominal rather than measured: captures give 487–507 Hz. An earlier figure was 312 Hz, from a five
second capture that counted the device's ~1.8 s startup delay as streaming time. That is not a cosmetic
error. The rate sets the spacing used to place sample timestamps, so too low a figure stretches the
timeline and drags the reported heart rate down with it: a synthetic 60 bpm run reported 37."""

USB_BAUD = 57600
"""The USB-C port is an FTDI UART bridge at 57600 8N1 carrying the identical framing."""


def crc(data: bytes) -> int:
    """Two's complement of the byte sum, over the length byte and body but not the header."""
    return (-sum(data)) & 0xFF


def command(packet_type: int, subtype: int, payload: bytes = b"") -> bytes:
    body = bytes([packet_type, subtype]) + payload
    with_len = bytes([len(body)]) + body
    return bytes([HEADER]) + with_len + bytes([crc(with_len)])


START_PPG = command(PACKET_BIOMETRICS, BIOMETRICS_ALLON)
STOP_PPG = command(PACKET_BIOMETRICS, BIOMETRICS_STOPALL)

QUERY_BATTERY = command(PACKET_BATTERY, BATTERY_LEVEL)
"""The official app never sends this — it reads battery from the standard Bluetooth service, which a
cable does not have — but the firmware answers it, whether or not the pulse stream is running. Do not
confuse the subtype with 0x02/0x03, which switch the charge path rather than reading the level."""


class FrameSplitter:
    """0xAB-delimited, CRC-checked framing over a byte stream.

    Stateful, and it never gives up: a bad CRC or a partial tail costs the frame and it re-syncs on the
    next header byte. A serial read returns whatever happened to be in the OS buffer, so assuming whole
    frames per chunk would be wrong on this transport even though it holds on Bluetooth.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self.bad_crc = 0
        self.dropped_bytes = 0

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        out: list[bytes] = []
        while True:
            start = self._buf.find(HEADER)
            if start < 0:
                self.dropped_bytes += len(self._buf)
                self._buf.clear()
                return out
            if start > 0:
                self.dropped_bytes += start
                del self._buf[:start]
            if len(self._buf) < 2:
                return out
            need = 1 + 1 + self._buf[1] + 1        # header, length byte, body, CRC
            if len(self._buf) < need:
                return out
            frame = bytes(self._buf[:need])
            del self._buf[:need]
            if crc(frame[1:-1]) == frame[-1]:
                out.append(frame)
            else:
                self.bad_crc += 1


def parse_ppg(frame: bytes) -> list[int] | None:
    """The sixteen samples of a raw-PPG frame, or None for any other frame.

    Everything else on the wire — acknowledgements, configuration read-backs, battery replies, and the
    device's own beat detections — comes back None so the caller can route it elsewhere.

    **The samples are big-endian.** Reading one sample little-endian at an offset one byte along gives
    the same number for a pair in the middle of the buffer, which hides the mistake. The equivalence
    breaks at the last sample, where the little-endian reading would step into the CRC.
    """
    if len(frame) < 3 + ALLON_BODY_LEN or frame[0] != HEADER or frame[1] != ALLON_BODY_LEN:
        return None
    if frame[2] != PACKET_BIOMETRICS or frame[3] != BIOMETRICS_ALLON:
        return None
    return [(frame[4 + 2 * i] << 8) | frame[5 + 2 * i] for i in range(SAMPLES_PER_FRAME)]


def battery_level(frame: bytes) -> int | None:
    """The percentage in a battery reply (`AB 03 07 01 <pct> <crc>`), or None for anything else."""
    if len(frame) >= 6 and frame[2] == PACKET_BATTERY and frame[3] == BATTERY_LEVEL:
        pct = frame[4]
        return pct if 0 <= pct <= 100 else None
    return None


class PeakDetector:
    """Beats out of a streaming pulse waveform.

    A slow exponential average tracks the baseline — the sensor's DC level and its breathing-rate drift
    — and is subtracted, leaving a signal whose dominant feature is the pulse. A sample is a beat when it
    is the largest in a window centred on it, which costs half a window of latency and is well under a
    beat. Two guards then reject the things that are not beats, and each of them earned its shape:

    **The refractory period scales with the measured interval.** A fixed 0.30 s is the worst available
    choice at resting rates, because the dicrotic notch — the pulse waveform's secondary pressure peak —
    lands about 0.35 s after the main one, just outside the window, so every beat is counted twice. Worse,
    the error is self-sustaining: counting the notch halves the mean interval, which shrinks the
    refractory, which keeps admitting the notch. On synthetic signals with a strong notch the detector
    locked onto exactly double the true rate and never recovered. Hence ~60% of the running mean, a floor
    of 0.35 s so a burst of doublets cannot drive it down there, and a fallback of 0.45 s before any rate
    is known. The cost is real and
    worth knowing: a 0.35 s floor caps the detectable rate at about 170 bpm. Fine for resting practice,
    wrong for exercise.

    **The amplitude floor relaxes once a beat is overdue.** It is a fraction of the mean height of
    recently *accepted* peaks, which on its own deadlocks: if the clip loosens and the pulse shrinks,
    every peak falls under the floor, and since rejected peaks never update the reference the floor stays
    high forever. Measured on synthetic data, an amplitude step down to 30% lost every subsequent beat. So
    past `overdue_factor` mean intervals the floor halves every `floor_relax_halflife_s`, and the detector
    re-acquires within a couple of seconds and then re-learns the new height.

    The mean is over four beats, not eight: at eight the reference follows a falling amplitude too slowly
    and worst-case misses measured 13.2% against 1.7%. A mean specifically — a minimum over the same
    window admits noise catastrophically, and a median of eight is worse than the mean of four.

    Simpler than the textbook detectors (Pan-Tompkins and its relatives) on purpose. Those were built for
    ECG, where the QRS complex is sharp enough to reward an explicit derivative and integrator. A pulse
    wave is smooth, and a local maximum with a refractory period is enough.
    """

    def __init__(self, rate_hz: float = PPG_RATE_HZ, baseline_tau_s: float = 1.5,
                 dwell_s: float = 0.18, refractory_s: float = 0.45, floor_frac: float = 0.35,
                 refractory_frac: float = 0.6, refractory_min_s: float = 0.35,
                 refractory_max_s: float = 0.60, overdue_factor: float = 1.4,
                 floor_relax_halflife_s: float = 0.4, amp_beats: int = 4):
        self.rate_hz = rate_hz
        dt = 1.0 / rate_hz
        self.baseline_alpha = 1.0 - math.exp(-dt / baseline_tau_s)
        self.dwell = max(3, int(round(dwell_s * rate_hz)) | 1)     # odd, so there is one centre
        self.refractory_s = refractory_s
        self.floor_frac = floor_frac
        self.refractory_frac = refractory_frac
        self.refractory_min_s = refractory_min_s
        self.refractory_max_s = refractory_max_s
        self.overdue_factor = overdue_factor
        self.floor_relax_halflife_s = floor_relax_halflife_s

        self._baseline: float | None = None
        self._window: deque[tuple[float, float]] = deque()
        self._last_beat: float | None = None
        self._amps: deque[float] = deque(maxlen=max(1, amp_beats))
        self._intervals: deque[float] = deque(maxlen=8)

    def push(self, t: float, sample: int) -> tuple[float, float | None] | None:
        """One sample in; `(beat_time, interval_seconds)` on a beat, and the interval is None on the
        first one, which has nothing to be measured from."""
        if self._baseline is None:
            self._baseline = float(sample)
            return None
        self._baseline += self.baseline_alpha * (sample - self._baseline)
        self._window.append((t, sample - self._baseline))
        while len(self._window) > self.dwell:
            self._window.popleft()
        if len(self._window) < self.dwell:
            return None

        centre_t, centre = self._window[self.dwell // 2]
        if any(centre < x for _, x in self._window):
            return None

        elapsed = None if self._last_beat is None else centre_t - self._last_beat
        if elapsed is not None and elapsed < self._refractory():
            return None
        if self._amps:
            mean_amp = sum(self._amps) / len(self._amps)
            if centre < mean_amp * self.floor_frac * self._relax(elapsed):
                return None

        self._last_beat = centre_t
        self._amps.append(centre)
        if elapsed is not None:
            self._intervals.append(elapsed)
        return centre_t, elapsed

    def _mean_interval(self) -> float | None:
        return sum(self._intervals) / len(self._intervals) if self._intervals else None

    def _refractory(self) -> float:
        mean = self._mean_interval()
        if mean is None:
            return self.refractory_s
        return min(max(self.refractory_frac * mean, self.refractory_min_s), self.refractory_max_s)

    def _relax(self, elapsed: float | None) -> float:
        mean = self._mean_interval()
        if elapsed is None or mean is None:
            return 1.0
        overdue_at = self.overdue_factor * mean
        if elapsed <= overdue_at:
            return 1.0
        return 0.5 ** ((elapsed - overdue_at) / self.floor_relax_halflife_s)


class PpgDecoder:
    """Transport bytes in, `Beat`s out. Everything between the two, and nothing about the wire.

    `feed()` takes a chunk and the time it arrived, and returns the beats it completed. The raw waveform
    is kept for a caller that wants to draw it; nothing does yet.
    """

    def __init__(self, rate_hz: float = PPG_RATE_HZ, warm_up_samples: int = 150,
                 raw_keep: int = 4096):
        self.rate_hz = rate_hz
        self._dt = 1.0 / rate_hz
        self._splitter = FrameSplitter()
        self._detector = PeakDetector(rate_hz=rate_hz)
        self._warm_up = max(0, warm_up_samples)
        self._seen = 0
        self._last_stamp: float | None = None
        self.raw: deque[tuple[float, int]] = deque(maxlen=raw_keep)
        self.battery_pct: int | None = None

    @property
    def bad_crc(self) -> int:
        return self._splitter.bad_crc

    def feed(self, data: bytes, arrival_t: float) -> list[Beat]:
        """Decode `data`, taking `arrival_t` as the moment its last sample was taken.

        **Sample times are spaced from arrivals, not from the nominal rate.** A chunk's samples are laid
        out evenly across the interval since the previous chunk ended, landing the last one on
        `arrival_t`, so a slightly wrong `rate_hz` cannot accumulate into drift — which is exactly what
        the old 312 Hz constant did. They stay strictly increasing by construction, so the detector can
        never see a negative interval.

        A chunk that spans far more than its own sample count would explain is a stall or a reconnect,
        and no spacing bridges a hole honestly, so that case re-anchors on the arrival time instead of
        smearing samples across it. Scaling that test by the chunk's own sample count is what keeps it
        transport-agnostic: Bluetooth delivers sixteen samples a notification, while a serial read returns
        whatever accumulated, so any fixed-duration threshold would fire constantly on one and never on
        the other.
        """
        pooled: list[int] = []
        for frame in self._splitter.feed(data):
            samples = parse_ppg(frame)
            if samples is None:
                pct = battery_level(frame)
                if pct is not None:
                    self.battery_pct = pct
                continue
            pooled += samples
        if not pooled:
            return []

        n = len(pooled)
        span = None if self._last_stamp is None else arrival_t - self._last_stamp
        if span is None or span > 4.0 * n * self._dt:
            step, t0 = self._dt, arrival_t - (n - 1) * self._dt
        else:
            # Floored only: the branch above already covers the long tail, so all that is left to guard
            # against is a burst compressing samples into implausibly little time.
            step = max(span / n, 0.25 * self._dt)
            t0 = self._last_stamp + step

        out: list[Beat] = []
        for i, sample in enumerate(pooled):
            t = t0 + i * step
            self._last_stamp = t
            self._seen += 1
            if self._seen <= self._warm_up:
                continue
            self.raw.append((t, sample))
            found = self._detector.push(t, sample)
            if found is None or found[1] is None:
                continue                        # the first beat has no interval to report
            beat_t, interval = found
            # `contact=None`, not False: a pulse sensor has no contact flag, and False is the strap
            # saying its intervals are whatever its detector latched onto, which the processor drops.
            out.append(Beat(t=beat_t, hr_bpm=60.0 / interval, rr_intervals=[interval], contact=None))
        return out
