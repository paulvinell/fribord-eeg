"""The same heartbeats seen at two sites: a reference pulse sensor and the forehead.

It takes plain timestamped beats and waveforms and knows nothing about the devices that made them. The
reference is the sensor with the clean beats (the HeartMath ear sensor); the forehead is the headband,
whose pulse is weaker. Four things come out: which beats both sites agree on, the forehead pulse shape
recovered by averaging around the reference beats, the ratio of the two pulse heights over the
recording, and the delay between the sites. Waveforms are `(time, value)` pairs, pulse pointing up.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from statistics import median

import numpy as np

from .ppg import steepest_rise

Wave = list[tuple[float, float]]

MATCH_TOLERANCE_S = 0.15
"""How far, once the usual delay is known, a pair may stray from it and still be the same heartbeat."""

RISE_SEARCH_S = 0.5
"""How far either side of the stamps to look for where a device's pulses rise steepest. Short of the
neighbouring heartbeats' rises below 120 bpm."""

EDGE_WINDOW_S = 0.15
"""How far either side of its device's usual place a beat's own steepest rise is looked for."""


@dataclass(frozen=True)
class Agreement:
    pairs: list[tuple[float, float]]        # (forehead beat, reference beat) from the same heartbeat
    forehead_only: list[float]
    reference_only: list[float]
    usual_delay_s: float                    # reference minus forehead, as used for pairing


@dataclass(frozen=True)
class Delay:
    median_s: float
    spread_s: float                         # robust standard deviation of the pairs around the median
    change_s: float                         # median over the last third minus the first third


@dataclass(frozen=True)
class Rescue:
    lags_s: np.ndarray                      # time from the reference beat
    shape: np.ndarray                       # the averaged forehead waveform
    height: float                           # top to bottom of the averaged shape
    noise: float                            # what is left of the beat-to-beat noise after averaging
    beats: int


@dataclass(frozen=True)
class Contrast:
    series: list[tuple[float, float]]       # (time, reference over forehead height, 1 = the start)
    late: float                             # the median over the last third of the recording


@dataclass(frozen=True)
class Comparison:
    agreement: Agreement
    delay: Delay | None
    rescue: Rescue | None
    contrast: Contrast | None


def match_beats(forehead: list[float], reference: list[float]) -> Agreement:
    """Pair beats from the same heartbeat; a beat seen at one site only is most likely an artifact there.
    The usual delay is found first, so pairing works however far apart the two devices' clocks are."""
    forehead, reference = sorted(forehead), sorted(reference)
    usual = _usual_delay(forehead, reference)
    used: set[int] = set()
    pairs: list[tuple[float, float]] = []
    for r in reference:
        target = r - usual
        i = bisect_left(forehead, target)
        best = min((j for j in (i - 1, i) if 0 <= j < len(forehead) and j not in used),
                   key=lambda j: abs(forehead[j] - target), default=None)
        if best is not None and abs(forehead[best] - target) <= MATCH_TOLERANCE_S:
            used.add(best)
            pairs.append((forehead[best], r))
    paired_reference = {r for _, r in pairs}
    # A beat counts as seen at one site only if the other site was recording when it happened.
    covered_f = (reference[0] - usual - MATCH_TOLERANCE_S, reference[-1] - usual + MATCH_TOLERANCE_S) \
        if reference else (0.0, -1.0)
    covered_r = (forehead[0] + usual - MATCH_TOLERANCE_S, forehead[-1] + usual + MATCH_TOLERANCE_S) \
        if forehead else (0.0, -1.0)
    return Agreement(pairs=pairs,
                     forehead_only=[f for j, f in enumerate(forehead)
                                    if j not in used and covered_f[0] <= f <= covered_f[1]],
                     reference_only=[r for r in reference
                                     if r not in paired_reference and covered_r[0] <= r <= covered_r[1]],
                     usual_delay_s=usual)


def edge_times(wave: Wave, beats: list[float]) -> list[float]:
    """Each beat moved to the steepest point of its pulse's rise, so two detectors that stamp beats at
    different points of the pulse are compared at the same point.

    Where that point lies relative to a stamp depends on the detector: on the rise for the headband,
    0.28 s after the stamp for the HeartMath, which stamps the foot of its pulse. So it is found first in
    the pulse averaged around all the stamps, and each beat is then searched for near it.
    """
    typical = rescue(wave, beats, before_s=RISE_SEARCH_S, after_s=RISE_SEARCH_S, rate_hz=100.0)
    if typical is None:
        return list(beats)
    lag = float(typical.lags_s[int(np.argmax(np.diff(typical.shape)))])
    times = [p[0] for p in wave]
    out = []
    for b in beats:
        lo, hi = bisect_left(times, b + lag - EDGE_WINDOW_S), bisect_left(times, b + lag + EDGE_WINDOW_S)
        out.append(steepest_rise(wave[lo:hi]) if hi - lo >= 5 else b + lag)
    return out


def delay(pairs: list[tuple[float, float]]) -> Delay | None:
    """Reference minus forehead time for each pair. The devices are stamped on arrival over different
    links, so the median carries the difference in their transport delays as well as the pulse's travel
    time; only its change across a recording can mean anything."""
    if len(pairs) < 6:
        return None
    d = np.array([r - f for f, r in pairs])
    mid = float(np.median(d))
    third = len(d) // 3
    return Delay(median_s=mid, spread_s=float(1.4826 * np.median(np.abs(d - mid))),
                 change_s=float(np.median(d[-third:]) - np.median(d[:third])))


def rescue(wave: Wave, beats: list[float], before_s: float = 0.3, after_s: float = 0.9,
           rate_hz: float = 30.0) -> Rescue | None:
    """The waveform averaged around each beat time on a common grid of lags. Noise averages away and the
    pulse does not, so this recovers the forehead pulse shape even when no single beat stands out."""
    if len(wave) < 10 or len(beats) < 5:
        return None
    t = np.array([p[0] for p in wave])
    v = np.array([p[1] for p in wave])
    lags = np.arange(-before_s, after_s, 1.0 / rate_hz)
    rows = [np.interp(b + lags, t, v) for b in beats if b - before_s >= t[0] and b + after_s <= t[-1]]
    rows = [row - row.mean() for row in rows]     # each slice's own level, not one noisy sample's
    if len(rows) < 5:
        return None
    stack = np.array(rows)
    shape = stack.mean(axis=0)
    noise = float(np.median(stack.std(axis=0)) / np.sqrt(len(rows)))
    return Rescue(lags_s=lags, shape=shape, height=float(np.ptp(shape)), noise=noise, beats=len(rows))


def heights(wave: Wave, beats: list[float], before_s: float = 0.4, after_s: float = 0.4) -> list[float | None]:
    """The pulse height around each beat: the top of the waveform less its bottom, within the window."""
    t = np.array([p[0] for p in wave])
    v = np.array([p[1] for p in wave])
    out: list[float | None] = []
    for b in beats:
        lo, hi = np.searchsorted(t, b - before_s), np.searchsorted(t, b + after_s)
        out.append(float(np.ptp(v[lo:hi])) if hi - lo >= 3 else None)
    return out


def contrast(forehead_wave: Wave, reference_wave: Wave, pairs: list[tuple[float, float]],
             start_s: float = 30.0) -> Contrast | None:
    """Reference height over forehead height for each paired heartbeat, divided by its median over the
    first `start_s` seconds (or the first third, if the recording is shorter), so 1 means as at the start.
    Small vessels in the skin tightening shrink the pulse at a sensor on the skin, so a falling ratio can
    mean that, though a sensor that moved changes it as well."""
    if len(pairs) < 6:
        return None
    f_h = heights(forehead_wave, [f for f, _ in pairs])
    r_h = heights(reference_wave, [r for _, r in pairs])
    raw = [(r, rh / fh) for (_, r), fh, rh in zip(pairs, f_h, r_h) if fh and rh]
    if len(raw) < 6:
        return None
    t0 = raw[0][0]
    early = [x for t, x in raw if t - t0 <= start_s]
    if len(early) < 3 or len(early) > 2 * len(raw) // 3:
        early = [x for _, x in raw[:len(raw) // 3]]
    base = median(early)
    series = [(t, x / base) for t, x in raw]
    last = [x for _, x in series[-(len(series) // 3):]]
    return Contrast(series=series, late=median(last))


def compare(forehead_beats: list[float], reference_beats: list[float],
            forehead_wave: Wave | None = None, reference_wave: Wave | None = None) -> Comparison:
    agreement = match_beats(forehead_beats, reference_beats)
    pairs = agreement.pairs
    if forehead_wave and reference_wave:
        f_edges = edge_times(forehead_wave, [f for f, _ in pairs])
        r_edges = edge_times(reference_wave, [r for _, r in pairs])
        timed = list(zip(f_edges, r_edges))
    else:
        timed = pairs
    return Comparison(
        agreement=agreement,
        delay=delay(timed),
        rescue=rescue(forehead_wave, reference_beats) if forehead_wave else None,
        contrast=contrast(forehead_wave, reference_wave, pairs) if forehead_wave and reference_wave else None,
    )


def _usual_delay(forehead: list[float], reference: list[float]) -> float:
    """Reference minus forehead, from where the forehead beats sit within the reference beat train.

    Each forehead beat's place between the two reference beats around it is a phase of the heartbeat,
    and the phases are averaged on a circle, so a delay near half a heartbeat, which splits between "just
    before" and "just after", still averages to one answer. Two devices stamped on arrival over different
    links share no clock, and nothing bounds how far apart their stamps are. Beat times alone cannot tell
    a delay from that delay plus a whole heartbeat, so the answer lies within half a heartbeat of zero.
    """
    angles = []
    for f in forehead:
        i = bisect_left(reference, f)
        if 0 < i < len(reference) and 0.3 < reference[i] - reference[i - 1] < 2.0:
            angles.append(2.0 * math.pi * (f - reference[i - 1]) / (reference[i] - reference[i - 1]))
    if not angles:
        return 0.0
    phase = math.atan2(sum(map(math.sin, angles)), sum(map(math.cos, angles))) / (2.0 * math.pi)
    period = median(b - a for a, b in zip(reference, reference[1:]))
    return -phase * period
