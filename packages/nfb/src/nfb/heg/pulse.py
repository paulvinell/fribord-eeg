"""Heart rate from the headband's own optical frames.

Each heartbeat pushes a little blood into the forehead, the blood absorbs the headband's infrared light,
and the light that comes back dips once per beat. The HEG processing filters this out on purpose, since
it trains a change over seconds; this module keeps it and turns it into `Beat`s, the form the HeartMath
sources produce, so the heart-rate processing downstream takes either device. Each choice here was
measured on a 40 s recording of a resting forehead in a room with flickering light.
"""

from __future__ import annotations

from collections import deque
from statistics import median

from nfb.hrv.beats import Beat
from nfb.hrv.ppg import PeakDetector

from .source import HEG_RATE_HZ, Frame

DESPIKE_SAMPLES = 5
"""The width of the running median over infrared minus ambient. Room light flickering at about 2 Hz put
a false 117 bpm rhythm into the infrared; the ambient reading carries the same flicker, so subtracting
it removes the rhythm, and the frames where the two do not cancel exactly become spikes, which a
five-sample median removes. It delays the signal two frames, which the time stamps account for."""

EDGE_SEARCH_S = 0.40
"""How far before a detected peak to look for its edge. The edge of a resting pulse comes 0.1 to
0.25 s before its top; 0.4 s reaches it without reaching back into the previous beat, whose falling side
has the opposite slope."""

MIN_PULSE_TO_NOISE = 3.5
"""How far a beat's rise must stand above the noise, as a multiple of the frame-to-frame jitter of the
signal before the median. The detector's own flat-line guard measures jitter on what it is fed, and the
median smooths that jitter away, so plain noise would pass it. On the recording real beats rose 3.6 to
18 times this jitter, median 11; synthetic noise alone and flicker alone never passed 3.0."""

NOISE_SECONDS = 2.0
"""How much of the signal before the median the noise is measured over."""

WAVEFORM_SECONDS = 64.0
"""How much of the cleaned waveform to keep, for a caller that averages it around beats."""


class _FrameClock:
    """Even sample times from the times frames arrived.

    The headband sends no clock of its own, only one frame per Bluetooth notification, and the
    notifications arrive with about 13 ms of jitter. Stamping beats with arrival times would put that
    jitter into every beat interval. This clock steps by its estimate of the frame period and moves
    toward each arrival by a small share of the difference, so the jitter averages out over about
    twenty frames while a slow drift in the true rate is still followed.
    """

    FOLLOW = 0.05           # share of each arrival's error the stamp moves by
    PERIOD_FOLLOW = 0.01    # share of each step's error the period estimate moves by
    STALL_PERIODS = 4.0     # a gap this many periods long is a stall, and the clock restarts there

    def __init__(self, rate_hz: float):
        self.period = 1.0 / rate_hz
        self._stamp: float | None = None
        self._arrival: float | None = None

    def stamp(self, arrival: float) -> float:
        if self._stamp is None or arrival - self._arrival > self.STALL_PERIODS * self.period:
            self._stamp = self._arrival = arrival
            return arrival
        step = arrival - self._arrival
        if step > 0:
            self.period += self.PERIOD_FOLLOW * (step - self.period)
        self._arrival = arrival
        predicted = self._stamp + self.period
        self._stamp = predicted + self.FOLLOW * (arrival - predicted)
        return self._stamp


class PulseExtractor:
    """Mendi frames in, `Beat`s out. `push(frame)` returns the beats that frame completed.

    It reads the P channel, which carried the strongest pulse of the three and is the channel the HEG
    projection uses as its reference, so the pulse and the HEG signal come from different channels. The
    signal is negated, because the pulse is a dip in light and the detector looks for peaks. The cleaned
    waveform is kept in `waveform` as `(time, value)` pairs, pulse pointing up, for a caller that averages
    it around beats.
    """

    def __init__(self, rate_hz: float = HEG_RATE_HZ):
        self.rate_hz = rate_hz
        self.reset()

    def reset(self) -> None:
        self._clock = _FrameClock(self.rate_hz)
        self._raw: deque[tuple[float, float]] = deque(maxlen=DESPIKE_SAMPLES)
        self._recent: deque[float] = deque(maxlen=int(NOISE_SECONDS * self.rate_hz))
        self._detector = PeakDetector(rate_hz=self.rate_hz)
        self._last_edge: float | None = None
        self.waveform: deque[tuple[float, float]] = deque(maxlen=int(WAVEFORM_SECONDS * self.rate_hz))
        self.beats_found = 0

    def push(self, frame: Frame) -> list[Beat]:
        sample = frame.p.ir - frame.p.amb
        self._raw.append((self._clock.stamp(frame.t), sample))
        self._recent.append(sample)
        if len(self._raw) < DESPIKE_SAMPLES:
            return []
        t = self._raw[DESPIKE_SAMPLES // 2][0]
        value = -median(v for _, v in self._raw)
        self.waveform.append((t, value))
        found = self._detector.push(t, value)
        if found is None:
            return []
        rise = [(at, v) for at, v in self.waveform if found[0] - EDGE_SEARCH_S <= at <= found[0]]
        if len(rise) < 5 or not self._stands_clear(rise):
            self._last_edge = None      # the next interval would span the missing beat
            return []
        edge = self._edge_time(rise)
        previous, self._last_edge = self._last_edge, edge
        if previous is None or edge <= previous:
            return []
        interval = edge - previous
        self.beats_found += 1
        return [Beat(t=edge, hr_bpm=60.0 / interval, rr_intervals=[interval], contact=None)]

    def _stands_clear(self, rise: list[tuple[float, float]]) -> bool:
        values = [v for _, v in rise]
        recent = list(self._recent)
        jitter = median(abs(b - a) for a, b in zip(recent, recent[1:])) if len(recent) > 2 else 0.0
        return max(values) - min(values) >= MIN_PULSE_TO_NOISE * max(jitter, 0.5)

    def _edge_time(self, points: list[tuple[float, float]]) -> float:
        """The steepest point of the rise before a peak, between samples.

        The top of a forehead pulse is broad and noise moves it; on the recording, stamping beats on the
        steepest point of the rise instead halved their beat-to-beat variation. The waveform is smoothed
        over three samples first, because a difference of neighbouring samples doubles their noise, and
        the steepest step is placed by a parabola through it and its neighbours.
        """
        smooth =[(points[i][0], (points[i - 1][1] + points[i][1] + points[i + 1][1]) / 3.0)
                  for i in range(1, len(points) - 1)]
        slopes = [((smooth[i][0] + smooth[i + 1][0]) / 2.0, smooth[i + 1][1] - smooth[i][1])
                  for i in range(len(smooth) - 1)]
        k = max(range(len(slopes)), key=lambda i: slopes[i][1])
        if not 0 < k < len(slopes) - 1:
            return slopes[k][0]
        (t_m, s_m), (t_0, s_0), (t_p, s_p) = slopes[k - 1], slopes[k], slopes[k + 1]
        bend = s_m - 2.0 * s_0 + s_p
        if bend >= 0.0:
            return t_0
        shift = max(-0.5, min(0.5, 0.5 * (s_m - s_p) / bend))
        return t_0 + shift * (t_p - t_m) / 2.0
