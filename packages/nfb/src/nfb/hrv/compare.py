"""The same heartbeats seen at two sites: a reference pulse sensor and the forehead.

It takes plain timestamped beats and waveforms and knows nothing about the devices that made them. The
reference is the sensor with the clean beats (the HeartMath ear sensor); the forehead is the headband,
whose pulse is weaker. Four things come out: which beats both sites agree on, the forehead pulse shape
recovered by averaging around the reference beats, the ratio of the two pulse heights over the
recording, and the delay between the sites. Waveforms are `(time, value)` pairs, pulse pointing up.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from statistics import median

import numpy as np

from .ppg import steepest_rise

Wave = list[tuple[float, float]]

MAX_LAG_S = 0.40
"""The largest delay between sites considered when looking for it. Below half the interval of a heart
at 75 bpm, so a beat cannot be paired with its neighbour."""

MATCH_TOLERANCE_S = 0.15
"""How far, once the usual delay is known, a pair may stray from it and still be the same heartbeat."""

EDGE_BEFORE_S, EDGE_AFTER_S = 0.40, 0.20
"""Where to look for the steepest rise around a beat: back far enough for a beat stamped on its top, and
on far enough for one stamped at the start of its rise, which at 30 samples a second needs the whole
rise in view. The next heartbeat's rise is further on than that even at 120 bpm."""


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
    The usual delay is found first, from each reference beat's nearest forehead beat, so pairing works
    whatever the radio latency is."""
    forehead, reference = sorted(forehead), sorted(reference)
    nearest = [d for r in reference if (d := _nearest_gap(r, forehead)) is not None and abs(d) <= MAX_LAG_S]
    usual = median(nearest) if nearest else 0.0
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
    return Agreement(pairs=pairs,
                     forehead_only=[f for j, f in enumerate(forehead) if j not in used],
                     reference_only=[r for r in reference if r not in paired_reference],
                     usual_delay_s=usual)


def edge_times(wave: Wave, beats: list[float]) -> list[float]:
    """Each beat moved to the steepest point of the rise near it, so two detectors that stamp beats at
    different points of the pulse are compared at the same point."""
    times = [p[0] for p in wave]
    out = []
    for b in beats:
        lo, hi = bisect_left(times, b - EDGE_BEFORE_S), bisect_left(times, b + EDGE_AFTER_S)
        out.append(steepest_rise(wave[lo:hi]) if hi - lo >= 5 else b)
    return out


def delay(pairs: list[tuple[float, float]]) -> Delay | None:
    """Reference minus forehead time for each pair. The devices are stamped on arrival over different
    links, so the median carries radio latency as well as the pulse's travel time; only its change across
    a recording can mean anything."""
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
    Under stress the body tightens the small vessels in the skin and the pulse at the sensor shrinks, so
    a falling ratio reads as rising arousal."""
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


def _nearest_gap(r: float, sorted_times: list[float]) -> float | None:
    i = bisect_left(sorted_times, r)
    gaps = [r - sorted_times[j] for j in (i - 1, i) if 0 <= j < len(sorted_times)]
    return min(gaps, key=abs) if gaps else None
