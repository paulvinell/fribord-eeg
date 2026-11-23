"""The calibration of the eye subtraction: a sequence the program leads, on signal time.

A dot moves to each edge of the window and back, then the person blinks at tones. The times of every move
and tone are known here, so the fit (`nfb.dsp.eyes.fit`) needs no detector to find them. The clock is the
samples fed in, as in `trials.TrialRun`: a test drives it with arrays, and a replay faster than real time
gives the same result. Owns no device and no screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import dataclasses

import numpy as np

from nfb.dsp.eyes import BLINK_BEFORE_S, BLINK_WINDOW_S, SETTLE_S, EyeFit, fit

from .trials import JUMP_S

ROTATIONS, BLINKS = "rotations", "blinks"
CENTRE = "centre"
DIRECTIONS = ("up", "down", "left", "right")
CENTRE_S = EDGE_S = 2.0
TONE_EVERY_S = 1.5
"""A blink to a tone is over within 1 s, and the next tone's level is the 0.5 s before it."""
BLINK_LEAD_S = 2.0
"""The dot holds at the centre after the last return before the first tone: the return is a move too, and
its level after it is measured up to 2 s."""

LINES = {ROTATIONS: "follow the dot with your eyes and keep your head still",
         BLINKS: "blink once at each tone"}
"""The one line under the dot, by part."""

GAP = f"the readings stop for more than {JUMP_S:g} s"
BACK = "the readings start again from an earlier point"
PAUSED = "the session pauses"
TOO_FEW = "too many blinks reach the input limit"

LIMIT_HOLD_S = 1.0
"""In the blinks part, a channel at the input limit holds the calibration only when it stays there longer than
this: a blink is over in under 0.4 s, so a longer stay is the electrode, not the blink."""


@dataclass(frozen=True)
class Plan:
    """One version of the calibration: how many trips of the dot, and how many blinks."""

    trips: int
    blinks: int

    @property
    def rotations_s(self) -> float:
        return self.trips * (CENTRE_S + EDGE_S)

    @property
    def tones(self) -> list[float]:
        return [self.rotations_s + BLINK_LEAD_S + k * TONE_EVERY_S for k in range(self.blinks)]

    @property
    def total_s(self) -> float:
        """To the end of the level after the last blink's window, which the blink is measured against."""
        return self.tones[-1] + BLINK_WINDOW_S + BLINK_BEFORE_S

    def trip_times(self) -> dict[str, list[tuple[float, float]]]:
        """The trips of each direction, in order: the time the dot moves out to the edge and the time it
        moves back to the centre."""
        out: dict[str, list[tuple[float, float]]] = {d: [] for d in DIRECTIONS}
        for k in range(self.trips):
            edge = k * (CENTRE_S + EDGE_S) + CENTRE_S
            out[DIRECTIONS[k % len(DIRECTIONS)]].append((edge, edge + EDGE_S))
        return out

    def rest(self) -> list[tuple[float, float]]:
        """The stretches with the eyes still at the centre: from the arrival of each return to the next move,
        and the lead before the first tone."""
        trip = CENTRE_S + EDGE_S
        return [(k * trip + (SETTLE_S if k else 0.0), k * trip + CENTRE_S) for k in range(self.trips + 1)]

    def dot_at(self, t: float) -> tuple[str, str]:
        """The part and the place of the dot at `t` seconds into the calibration."""
        if t < self.rotations_s:
            trip, into = divmod(t, CENTRE_S + EDGE_S)
            return ROTATIONS, DIRECTIONS[int(trip) % len(DIRECTIONS)] if into >= CENTRE_S else CENTRE
        return BLINKS, CENTRE


SHORT, LONG = "short", "long"
PLANS = {SHORT: Plan(trips=8, blinks=8), LONG: Plan(trips=16, blinks=12)}


class EyeCalibration:
    """One calibration, fed aligned raw runs of the training channel and the extras, in µV.

    `feed` returns how many tones fell due, for the session to sound. It ends `done` with `result`, or
    `stopped` with the reason; a stopped calibration keeps nothing. A channel that cannot be used holds it
    (`held` says why) and drops the part in progress; `resume` goes on from the start of that part, so the
    parts already complete are kept."""

    def __init__(self, rate_hz: float, training: int, extras: tuple[int, ...], plan: Plan = PLANS[SHORT]):
        self.rate_hz = rate_hz
        self.plan = plan
        self.training = training
        self.extras = tuple(extras)
        self._next: int | None = None
        self._y: list[np.ndarray] = []
        self._xs: list[np.ndarray] = []
        self._lim: list[np.ndarray] = []        # per sample, whether a channel was at the input limit
        self._run = 0                           # samples at the limit in a row, up to the newest
        self._n = 0
        self._tones = 0
        self.stopped: str | None = None
        self.result: EyeFit | None = None
        self.held: str | None = None
        self.begun = False
        """Whether a sample has been taken, so the screen can say `start` before and `continue` after."""

    @property
    def seconds(self) -> float:
        return self._n / self.rate_hz

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def running(self) -> bool:
        return self.result is None and self.stopped is None

    @property
    def dot(self) -> tuple[str, str]:
        return self.plan.dot_at(self.seconds)

    @property
    def line(self) -> str:
        return LINES[self.dot[0]]

    def hold(self, reason: str) -> None:
        """Wait for `resume`, keeping only the parts that are complete."""
        if not self.running:
            return
        self.held = reason
        keep = round(self.plan.rotations_s * self.rate_hz) if self.seconds >= self.plan.rotations_s else 0
        if keep < self._n:
            y, xs, lim = np.concatenate(self._y), np.hstack(self._xs), np.concatenate(self._lim)
            self._y, self._xs, self._lim = ([y[:keep]], [xs[:, :keep]], [lim[:keep]]) if keep else ([], [], [])
            self._n = keep
        self._run = 0
        self._tones = sum(1 for at in self.plan.tones if at <= self.seconds)

    def resume(self) -> None:
        """Go on from where the kept parts end, with the next sample fed, whatever its index."""
        self.held = None
        self._next = None

    def stop(self, reason: str) -> None:
        """End it here, keeping nothing."""
        if self.running:
            self.stopped = reason
            self._y, self._xs, self._lim = [], [], []

    def feed(self, start: int, y: np.ndarray, xs: np.ndarray, problem: str | None = None,
             limit: np.ndarray | None = None, limit_reason: str | None = None) -> int:
        """One aligned run from sample `start`. `problem` names a channel that cannot be used, which holds
        the calibration. `limit` says per sample whether a channel is at the input limit, which `limit_reason`
        names: in the eye movements it holds the calibration; in the blinks it leaves the blink out, and holds
        only past `LIMIT_HOLD_S` in a row. A hole up to `JUMP_S` is held at the last values; a longer one stops
        it."""
        if not self.running or self.held is not None or not len(y):
            return 0
        if problem is not None:
            self.hold(problem)
            return 0
        lim = np.zeros(len(y), dtype=bool) if limit is None else np.asarray(limit, dtype=bool)
        reason = limit_reason or "a channel is at the input limit"
        if lim.any() and self.seconds < self.plan.rotations_s:
            self.hold(reason)
            return 0
        for at in lim:
            self._run = self._run + 1 if at else 0
            if self._run > LIMIT_HOLD_S * self.rate_hz:
                self.hold(reason)
                return 0
        self.begun = True
        y = np.asarray(y, dtype=float)
        xs = np.atleast_2d(np.asarray(xs, dtype=float))
        if self._next is not None and start != self._next:
            hole = start - self._next
            if hole < 0 or hole > JUMP_S * self.rate_hz:
                self.stop(BACK if hole < 0 else GAP)
                return 0
            self._y.append(np.full(hole, self._y[-1][-1]))
            self._xs.append(np.repeat(self._xs[-1][:, -1:], hole, axis=1))
            self._lim.append(np.zeros(hole, dtype=bool))
            self._n += hole
        self._y.append(y)
        self._xs.append(xs)
        self._lim.append(lim)
        self._n += len(y)
        self._next = start + len(y)
        due = sum(1 for at in self.plan.tones if at <= self.seconds) - self._tones
        self._tones += due
        if self.seconds >= self.plan.total_s:
            self._finish()
        return due

    def _finish(self) -> None:
        """Each trip gives one event, its edge against the centres on both sides. Of each direction's trips
        and of the blinks, the first half is learned from and the second half tests."""
        rate, plan = self.rate_hz, self.plan
        y = np.concatenate(self._y)
        xs = np.hstack(self._xs)
        trips = {d: [(round(a * rate), round(b * rate)) for a, b in times] for d, times in plan.trip_times().items()}
        learn = [m for d in DIRECTIONS for m in trips[d][:len(trips[d]) // 2]]
        test = {"up-down": [m for d in ("up", "down") for m in trips[d][len(trips[d]) // 2:]],
                "left-right": [m for d in ("left", "right") for m in trips[d][len(trips[d]) // 2:]]}
        tones = [round(s * rate) for s in plan.tones]
        half = plan.blinks // 2
        # A blink is left out where a channel touched the limit in what it is measured on: its window and
        # the levels on both sides of it.
        lim = np.concatenate(self._lim)
        side, window = round(BLINK_BEFORE_S * rate), round(BLINK_WINDOW_S * rate)
        clean = [not lim[t - side:t + window + side].any() for t in tones]
        learn_tones = [t for t, ok in zip(tones[:half], clean[:half]) if ok]
        test_tones = [t for t, ok in zip(tones[half:], clean[half:]) if ok]
        if 2 * len(learn_tones) < half or 2 * len(test_tones) < plan.blinks - half:
            self.hold(TOO_FEW)
            return
        rest = [(round(lo * rate), round(hi * rate)) for lo, hi in plan.rest()]
        result = fit(y, xs, rate, learn, test, learn_tones, test_tones, rest)
        self.result = dataclasses.replace(result, blinks_left_out=clean.count(False))
        self._y, self._xs, self._lim = [], [], []

OK, LIMIT, NO_SIGNAL = 0, 1, 2
"""What each extra sample is: usable, at the input limit, or with no signal. The larger code wins."""


class EyeAligner:
    """Training blocks held until every extra channel has sent the same samples, then handed on with them.

    The device sends each channel's block on its own, so the training block can come before or after its
    extras. A training block waits for them, and goes on without them where an extra has sent nothing
    for `JUMP_S`: those samples are then marked `NO_SIGNAL`, as a hole in an extra is."""

    def __init__(self, extras: tuple[int, ...], rate_hz: float):
        self.extras = tuple(extras)
        self.rate_hz = rate_hz
        self._start: dict[int, int] = {}
        self._values: dict[int, np.ndarray] = {}
        self._codes: dict[int, np.ndarray] = {}
        self._pending: list[tuple] = []

    def push_extra(self, channel: int, start: int, values: np.ndarray, code: int) -> None:
        if channel not in self.extras:
            return
        values = np.asarray(values, dtype=float)
        codes = np.full(len(values), code, dtype=np.int8)
        end = self._start[channel] + len(self._values[channel]) if channel in self._start else None
        if end is None or start < end or start - end > JUMP_S * self.rate_hz:
            self._start[channel], self._values[channel], self._codes[channel] = start, values, codes
            return
        hole = start - end
        self._values[channel] = np.r_[self._values[channel], np.zeros(hole), values]
        self._codes[channel] = np.r_[self._codes[channel], np.full(hole, NO_SIGNAL, dtype=np.int8), codes]

    def push_training(self, block, meta) -> None:
        """A training block to hold. One that steps back drops those held before it: they precede nothing."""
        if self._pending and block.start_index < self._pending[-1][0].end_index:
            self._pending.clear()
        self._pending.append((block, meta))

    def _end(self, channel: int) -> int | None:
        return self._start[channel] + len(self._values[channel]) if channel in self._start else None

    def ready(self) -> list[tuple]:
        """The training blocks that can go on, oldest first: `(block, meta, xs, codes)`, with `xs` the
        extras' raw µV over the block's samples and `codes` the code of each of those samples."""
        out = []
        newest = self._pending[-1][0].end_index if self._pending else 0
        while self._pending:
            block, meta = self._pending[0]
            s, e = block.start_index, block.end_index
            behind = any(self._end(c) is None or self._end(c) < e for c in self.extras)
            if behind and newest - e < JUMP_S * self.rate_hz:
                break                                   # the extras of this block are still to come
            self._pending.pop(0)
            n = e - s
            xs = np.zeros((len(self.extras), n))
            codes = np.full((len(self.extras), n), NO_SIGNAL, dtype=np.int8)
            for k, c in enumerate(self.extras):
                if c in self._start:
                    lo, hi = max(s, self._start[c]), min(e, self._end(c))
                    if hi > lo:
                        part = slice(lo - self._start[c], hi - self._start[c])
                        xs[k, lo - s:hi - s] = self._values[c][part]
                        codes[k, lo - s:hi - s] = self._codes[c][part]
                    cut = max(0, min(e, self._end(c)) - self._start[c])
                    self._start[c] += cut
                    self._values[c], self._codes[c] = self._values[c][cut:], self._codes[c][cut:]
            out.append((block, meta, xs, codes))
        return out
