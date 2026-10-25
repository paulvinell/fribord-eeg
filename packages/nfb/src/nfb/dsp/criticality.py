"""How close a rhythm's fluctuations sit to the critical point between order and noise.

Two measures over a band's **amplitude envelope**, not over the signal itself. Both come from the same
literature and answer different halves of one question.

- `dfa` is the detrended fluctuation analysis exponent, which says whether the envelope's ups and downs
  are correlated over long stretches or independent from moment to moment. 0.5 is a memoryless signal;
  above 0.5 the envelope carries structure over tens of seconds, which is the signature that a network
  sitting near criticality produces and a network away from it does not. Published resting alpha lands
  around 0.6-0.8.
- `functional_ei` is the functional excitation-inhibition ratio: **1.0 balanced, below 1 inhibition
  dominated, above 1 excitation dominated.** It works by asking whether the envelope's fluctuation grows
  in proportion to its amplitude. In a balanced network it does, so the two are uncorrelated once scaled;
  when excitation dominates, big moments fluctuate relatively *more* and the correlation goes negative.

**The optimum is a value, not a direction, and that is the whole reason this module exists.** Every other
measure in this codebase is trained up or down. This one has a published target in the middle, so a
screen can reward *proximity* to it rather than more of it — which is what the framework these came from
asks for and what a maximised complexity number cannot give.

**Neither is feedback and neither pretends to be.** `dfa` fits a line across window sizes up to tens of
seconds, so it needs minutes of signal; `functional_ei` correlates across 5 s windows and its own authors
used three to five minutes. A number that settles over minutes is something an operator reads at the end
of a run, and this module is imported by the screen's trends rather than by its targets.

`functional_ei` is only interpretable where there are long-range correlations to speak of, which is what
`DFA_FLOOR` is: below it the measure is reported as undefined rather than as a number, because a ratio
computed from an envelope with no memory is a ratio of noise to noise.
"""

from __future__ import annotations

import numpy as np

DFA_FLOOR = 0.6
"""Below this exponent there are no long-range correlations to base an E/I ratio on.

The inclusion criterion its authors applied before reporting any functional E/I value. Kept as a gate
rather than a note, because the alternative is a screen showing a confident 1.03 computed from noise.
"""

DFA_NOISE = 0.58
"""**What this exponent reads on a narrowband envelope of pure noise — and it is not 0.5.**

Measured here, six seeds per length, alpha envelope of white noise: 0.572 mean at 120 s, 0.585 at 300 s,
0.581 at 600 s. The level does not fall with more data; only the spread does, and the worst of six went
0.668 at 120 s, 0.641 at 300 s, 0.598 at 600 s.

Two things follow, both of which change what a screen may claim. A band-limited envelope is *already*
correlated over the filter's own time constant, so half is the wrong thing to compare against — the
estimator is unbiased, which is checked separately against white noise at 0.51 and a random walk at 1.50,
and it is the envelope that carries the memory. And the 0.6 inclusion floor sits barely above this, so
below about ten minutes a reading that clears the floor is inside what noise produces. That is why this
number is drawn beside the reading rather than kept here.
"""

FEI_WINDOW_SECONDS = 5.0
FEI_OVERLAP = 0.8
BALANCED = 1.0
"""What `functional_ei` reads for a network at the critical point."""

DFA_MIN_SECONDS = 1.0
DFA_MAX_SECONDS = 25.0
DFA_WINDOWS = 20


def _profile(amplitude: np.ndarray) -> np.ndarray:
    """The cumulative sum of the de-meaned envelope — the random walk whose spread is being measured."""
    a = np.asarray(amplitude, dtype=float)
    return np.cumsum(a - a.mean())


def _detrended_std(segment: np.ndarray) -> float:
    """Standard deviation of a segment with its linear trend removed."""
    n = len(segment)
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, segment, 1)
    return float(np.std(segment - (slope * t + intercept)))


def dfa(amplitude: np.ndarray, rate_hz: float, min_seconds: float = DFA_MIN_SECONDS,
        max_seconds: float = DFA_MAX_SECONDS, windows: int = DFA_WINDOWS) -> float:
    """Detrended fluctuation analysis exponent of an amplitude envelope.

    Window sizes are spaced logarithmically, which is what makes the fit a fit: the exponent is the slope
    of log fluctuation against log window size, and linearly spaced sizes would weight the long windows —
    the ones with fewest samples behind them — by sheer count.

    Raises when the signal is too short to span the requested range, rather than quietly fitting a line
    through two points.
    """
    a = np.asarray(amplitude, dtype=float)
    profile = _profile(a)
    n = len(profile)
    lo, hi = int(min_seconds * rate_hz), int(max_seconds * rate_hz)
    hi = min(hi, n // 4)
    if hi <= lo or n < 4 * lo:
        raise ValueError(
            f"need at least {4 * min_seconds:g} s of envelope to fit from {min_seconds:g} s, "
            f"got {n / rate_hz:.1f} s")
    sizes = np.unique(np.geomspace(lo, hi, windows).astype(int))
    fluctuations = []
    for size in sizes:
        count = n // size
        segments = profile[:count * size].reshape(count, size)
        rms = np.sqrt(np.mean([_detrended_std(s) ** 2 for s in segments]))
        fluctuations.append(max(rms, 1e-12))
    slope = np.polyfit(np.log(sizes), np.log(fluctuations), 1)[0]
    return float(slope)


def functional_ei(amplitude: np.ndarray, rate_hz: float,
                  window_seconds: float = FEI_WINDOW_SECONDS,
                  overlap: float = FEI_OVERLAP) -> float:
    """Functional excitation-inhibition ratio of an amplitude envelope. 1.0 is balanced.

    The steps are the published ones and the order matters:

    1. build the profile — the cumulative sum of the de-meaned envelope;
    2. cut both the envelope and the profile into overlapping windows;
    3. divide each profile window by that window's own mean amplitude, which is the step that makes the
       fluctuation a *relative* one and is what lets this be estimated over 5 s rather than over minutes;
    4. detrend each normalised window and take its standard deviation;
    5. correlate those against the windows' mean amplitudes, and subtract from one.

    A balanced network's relative fluctuation does not depend on how loud it happens to be, so the
    correlation is zero and the ratio is one. Callers check `dfa` first: this returns a number for any
    input, and the number means nothing without long-range correlations behind it.
    """
    a = np.asarray(amplitude, dtype=float)
    size = max(3, int(round(window_seconds * rate_hz)))
    step = max(1, int(round(size * (1.0 - overlap))))
    if len(a) < size * 2:
        raise ValueError(f"need at least {2 * window_seconds:g} s of envelope, "
                         f"got {len(a) / rate_hz:.1f} s")
    profile = _profile(a)
    amps, fluctuations = [], []
    for start in range(0, len(a) - size + 1, step):
        window = a[start:start + size]
        mean = float(window.mean())
        if mean <= 0:
            continue
        amps.append(mean)
        fluctuations.append(_detrended_std(profile[start:start + size] / mean))
    if len(amps) < 3:
        raise ValueError("too few usable windows for a correlation")
    amps_arr, fluc = np.asarray(amps), np.asarray(fluctuations)
    if amps_arr.std() == 0 or fluc.std() == 0:
        raise ValueError("no variation to correlate")
    return float(1.0 - np.corrcoef(amps_arr, fluc)[0, 1])
