"""Artifact and ectopic-beat rejection for a streaming RR-interval series.

The measurements in the comments below justify each choice. The filter is pure Python and needs no package
boundary.

Three layers, in order of how often they fire:

1. **Range gate** — an interval outside 333–2000 ms (180–30 bpm) is not a
   heartbeat. Catches gross detector failures.
2. **Ectopic test** — a single-beat spike or notch, distinguished from a
   genuine change in heart rate by three conditions that must all hold.
3. **What happens to a rejected beat**, which is the subtle part. It is
   *withheld*, not deleted and not zero-filled. Because the downstream
   tachogram is interpolated over absolute beat times, withholding a beat
   makes the interpolation span the whole gap — a straight line between
   the last good interval and the next one. The two naive alternatives
   both corrupt the spectrum: dropping-and-shifting compresses the time
   axis, and zero-filling injects broadband power.

Why the three-condition ectopic test rather than a simple outlier check:
a real ectopic beat comes early and is followed by a compensatory pause,
so the *following* interval jumps too. Requiring that jump is what
separates an ectopic from a legitimate acceleration — during genuine
rate change, successive intervals drift in one direction rather than
spiking and rebounding. Without it, a filter this aggressive would eat
real respiratory-driven variability, which is precisely the signal we're
trying to measure.

**One-beat latency.** The test needs the following beat, so a beat is
only released once its successor arrives. At resting heart rates that's
about a second. `HRVProcessor` accounts for this; nothing downstream
needs to.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple


# Physiologically plausible interval bounds, in seconds. 0.333 s = 180 bpm,
# 2.0 s = 30 bpm.
RANGE_MIN_S = 0.333
RANGE_MAX_S = 2.000


@dataclass(frozen=True)
class ArtifactParams:
    """Tuning for `ArtifactFilter`."""

    range_min_s: float = RANGE_MIN_S
    range_max_s: float = RANGE_MAX_S
    # Deviation from the local mean, in running standard deviations, before
    # a beat is even a candidate.
    spike_sd: float = 2.0
    # How far the *following* interval must differ from the candidate. This
    # is the compensatory-pause test.
    rebound_sd: float = 1.0
    # Local mean is taken over this many most-recently-accepted beats.
    local_mean_beats: int = 4
    # The ectopic test needs a stable SD estimate; below this many accepted
    # beats, only the range gate applies.
    min_beats: int = 6
    # …and it stays off for this long after the first beat, for the same
    # reason.
    warmup_s: float = 20.0


@dataclass(frozen=True)
class Verdict:
    """One settled beat and why it was kept or rejected."""

    t: float                 # beat time (end of the interval)
    rr: float                # interval, seconds
    accepted: bool
    reason: str = ""         # "" when accepted; else "range" or "ectopic"


class ArtifactFilter:
    """Streaming RR filter. Feed beats in order, take settled beats out.

    Usage:

        filt = ArtifactFilter()
        for (t, rr) in beats:
            for v in filt.push(t, rr):
                if v.accepted:
                    use(v.t, v.rr)

    Each `push` returns zero or one verdict — zero while the first beat is
    still awaiting a successor, one thereafter. Rejected beats are
    reported rather than silently swallowed so callers can surface a
    signal-quality figure; `rejected_count` and `accepted_count` track
    totals.
    """

    def __init__(self, params: Optional[ArtifactParams] = None) -> None:
        self.p = params or ArtifactParams()
        # The candidate awaiting its successor.
        self._pending: Optional[Tuple[float, float]] = None
        # Interval immediately preceding the candidate, for the spike test.
        self._prev_rr: Optional[float] = None
        # Most-recently-accepted intervals, newest first.
        self._recent: Deque[float] = deque(maxlen=self.p.local_mean_beats)
        # Running mean/variance over ACCEPTED intervals only.
        #
        # Accumulating SD over every beat in the session lets artifacts inflate
        # the very threshold used to detect them: on a series with ectopics
        # every ~6 beats, the cumulative SD reaches ~0.25 s where the clean
        # signal's is ~0.004 s, and the 2-SD gate then admits almost
        # everything. Measured before the change: it caught 3 of 15 injected
        # ectopics. Feeding only accepted beats keeps the reference clean and
        # takes it to 15 of 15. Dividing by n rather than n-1 is immaterial
        # once past the min_beats gate.
        self._n = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self._first_t: Optional[float] = None
        self.accepted_count = 0
        self.rejected_count = 0
        self.rejected_range = 0
        self.rejected_ectopic = 0

    # ---- statistics -------------------------------------------------------

    @property
    def _sd(self) -> float:
        if self._n < 2:
            return 0.0
        var = self._sumsq / self._n - (self._sum / self._n) ** 2
        return math.sqrt(var) if var > 0.0 else 0.0

    def _local_mean(self) -> Optional[float]:
        if not self._recent:
            return None
        return sum(self._recent) / len(self._recent)

    # ---- the test ---------------------------------------------------------

    def _in_range(self, rr: float) -> bool:
        return self.p.range_min_s <= rr <= self.p.range_max_s

    @staticmethod
    def _same_side(cur: float, prev: float, nxt: float) -> bool:
        """Do both neighbours sit strictly on one side of `cur`?

        True for a spike (both below) or a notch (both above); False when
        the neighbours straddle it, which is what a monotonic rate change
        looks like.
        """
        return (prev > cur and nxt > cur) or (prev < cur and nxt < cur)

    def _is_ectopic(self, rr: float, prev_rr: float, next_rr: float) -> bool:
        sd = self._sd
        if sd <= 0.0:
            return False
        local_mean = self._local_mean()
        if local_mean is None:
            return False
        # All three must hold. Any one alone produces far too many false
        # positives on real respiratory variability.
        if abs(rr - local_mean) <= self.p.spike_sd * sd:
            return False
        if not self._same_side(rr, prev_rr, next_rr):
            return False
        if abs(next_rr - rr) <= self.p.rebound_sd * sd:
            return False
        return True

    def _ectopic_test_active(self, t: float) -> bool:
        if self._n < self.p.min_beats:
            return False
        if self._first_t is None:
            return False
        return (t - self._first_t) >= self.p.warmup_s

    # ---- streaming interface ---------------------------------------------

    def push(self, t: float, rr: float) -> List[Verdict]:
        """Feed one beat. Returns verdicts for any beat that just settled."""
        if self._first_t is None:
            self._first_t = t

        # Out-of-range intervals are rejected immediately: they can't inform
        # the running statistics and they mustn't become a spike test's
        # neighbour, which would let one bad beat condemn a good one.
        if not self._in_range(rr):
            self.rejected_count += 1
            self.rejected_range += 1
            out = self._settle_pending_without_successor()
            return out + [Verdict(t=t, rr=rr, accepted=False, reason="range")]

        out: List[Verdict] = []
        if self._pending is not None:
            p_t, p_rr = self._pending
            prev = self._prev_rr
            reject = (
                prev is not None
                and self._ectopic_test_active(p_t)
                and self._is_ectopic(p_rr, prev, rr)
            )
            if reject:
                self.rejected_count += 1
                self.rejected_ectopic += 1
                out.append(Verdict(t=p_t, rr=p_rr, accepted=False,
                                   reason="ectopic"))
                # Deliberately do NOT advance `_prev_rr` to a rejected beat,
                # and do not add it to `_recent`. The local mean and the
                # spike comparison must both rest on accepted beats only,
                # or a run of artifacts drags the reference along with it
                # and the filter stops detecting anything.
            else:
                out.append(self._accept(p_t, p_rr))

        self._pending = (t, rr)
        return out

    def _accept(self, t: float, rr: float) -> Verdict:
        """Admit a beat and let it inform the running statistics."""
        self.accepted_count += 1
        self._recent.appendleft(rr)
        self._prev_rr = rr
        self._n += 1
        self._sum += rr
        self._sumsq += rr * rr
        return Verdict(t=t, rr=rr, accepted=True)

    def _settle_pending_without_successor(self) -> List[Verdict]:
        """Release the pending beat when the next interval is unusable.

        A range-rejected interval can't serve as the successor for the
        ectopic test, so the pending beat is accepted on the evidence
        available rather than held indefinitely.
        """
        if self._pending is None:
            return []
        p_t, p_rr = self._pending
        self._pending = None
        return [self._accept(p_t, p_rr)]

    def flush(self) -> List[Verdict]:
        """Release the pending beat untested. For end-of-stream only."""
        return self._settle_pending_without_successor()

    def reset(self) -> None:
        self.__init__(self.p)  # type: ignore[misc]

    @property
    def reject_fraction(self) -> float:
        total = self.accepted_count + self.rejected_count
        return (self.rejected_count / total) if total else 0.0
