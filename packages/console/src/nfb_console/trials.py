"""Trial structure: the session shape SCP needs and no other protocol here has.

Every other protocol in this repo is one undifferentiated run against a threshold. SCP training is a
**sequence of trials**, each with a required direction, its own baseline, and a per-trial verdict — the
session-structure gap, closed here as a pure state machine so a screen can
render it and a test can drive it without either owning it.

The shape of a trial is the clinical one (Strehl's protocol line, the best-evidenced arrangement in the
field): a cue announces the required direction, a **baseline** window is averaged, an **active** window
follows in which the feedback shows the shift against that baseline, and the trial is scored on the *mean*
shift over the active window. Trials alternate direction pseudorandomly and a configurable fraction are
**transfer** trials — cue and verdict but no feedback while it runs — which is how the skill is weaned off
the display. The numbers below are defaults in that tradition, not measurements of anything; each is a
field on `TrialRun` precisely because a clinic would tune them.

Three decisions worth stating:

- **The clock is signal time**, the `t` of the readings fed in, never the wall clock — the same rule the
  waveform screen's calibration follows and for the same reason: phase lengths are properties of the
  signal, and a replay driven faster than real time must produce the same trials. A jump in `t` (a device
  gap, a suspend) **aborts the trial in flight** rather than scoring it: its baseline describes signal
  that no longer precedes its active window. The person did that trial, so it stays in the history as a
  void one. The same clock says how many readings a phase should hold: a reading that never arrived is
  counted as a lost one, not left out of the count.

- **The per-trial baseline is the de-trend.** Subtracting each trial's own baseline mean removes drift and
  standing offset trial by trial, which is why `nfb.dsp.scp.SlowShift` carries no drift follower — one
  fast enough to matter would eat the eight-second shift being trained. See that module's docstring.

- **An artifacted trial is discarded, not scored.** Readings arriving `valid=False` (the gate's stand-ins,
  device-invalid blocks) are counted rather than used, and a trial where they exceed `reject_limit` is
  marked invalid: it appears in the history, drawn as such, and is excluded from every success rate. This
  is the clinical arrangement — EOG-contaminated trials are voided — and it is also the only honest one,
  since a blink's deflection lives in exactly the band this protocol trains. A void trial keeps one
  cause, the first of `CAUSES` that its own readings show, so the screen can say why and what to do.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

BASELINE_S = 2.0
"""The window averaged into the trial's baseline. Two seconds is the standard clinical pre-phase."""

ACTIVE_S = 6.0
"""The feedback window the trial is scored on. Six seconds of shift against the baseline."""

RESULT_S = 1.5
"""How long the verdict is shown before the rest interval. Display time, not measurement."""

ITI_RANGE_S = (2.0, 4.0)
"""The rest between trials, drawn uniformly. Randomised so the next cue cannot be timed rather than
attended to."""

LEADIN_S = 2.0
"""Rest before the first trial, so the first cue does not land on the first settled sample."""

DEFAULT_TRIALS = 40
"""Trials per run — the clinical block size. At ~12 s a trial this is about eight minutes."""

TRANSFER_FRACTION = 1.0 / 3.0
"""What share of trials run without feedback. A third is the usual mature-protocol mix; 0 is a
reasonable first-sessions setting and the screen offers it."""

REJECT_LIMIT = 0.2
"""Above this fraction of invalid readings across baseline + active, the trial is void."""

MIN_BASELINE_FRACTION = 0.5
"""A baseline needs at least this share of its expected readings to mean anything; below it the trial is
void for the same reason an artifacted one is."""

JUMP_S = 1.0
"""A step in signal time larger than this is a gap — readings arrive at ~20 Hz, so anything near a second
means the signal was interrupted — and the trial in flight is aborted rather than scored. Under it, the
readings the clock expected and did not get are counted as lost."""

LIMIT, NO_SIGNAL, LOST, GAP, THIN_BASELINE, REJECTED = ("limit", "no signal", "lost", "gap", "baseline",
                                                       "rejected")
EXTRA = "extra"
"""An extra channel that the subtraction of eye movements uses has no signal or is at the input limit, so the
program cannot correct the reading."""
CAUSES = (LIMIT, NO_SIGNAL, EXTRA, LOST, GAP, THIN_BASELINE, REJECTED)
"""Why a trial is void. The first that its own readings show of the causes that are not counts: a sample at
the input limit, a channel with no signal, an extra channel the subtraction cannot use, a jump in time, a
baseline too thin to measure against. After them, of the two counts (a reading that never arrived, a
rejected reading), the one with more readings, the missing one on a tie. The void decision itself stays
the 20% rule and the baseline rule; the cause only names it."""

COUNTS = (LOST, REJECTED)

PAUSED, SEAM = "paused", "seam"
"""The causes of a cut that the signal's time does not show: the session paused, however briefly, or a
looped file started again. Like GAP, each ends the trial in flight at once."""

BLOCK = 6
"""Directions and transfer slots are balanced within blocks of this many trials, so no long run of one
direction and no front-loading of transfers can happen by draw."""

POSITIVE, NEGATIVE = "positive", "negative"
"""The required sign of the shift, in µV as measured. Deliberately not "negativity"/"positivity" in the
cortical sense: which physical polarity that is depends on which electrode went in which socket, the same
honesty `wave_view.set_direction` states. The trained skill is *differentiation* — producing each on
demand — which survives either wiring."""

BASELINE, ACTIVE, RESULT, ITI = "baseline", "active", "result", "iti"
CUED = "cued"
"""The optional phase between the cue and the baseline, when `cue_s` is not zero. Nothing is averaged in it:
it exists for a signal that answers the cue itself, as a pupil dilates to a tone, so the answer settles
before the baseline that is meant to precede the effort."""

CUE, FEEDBACK_ON, VERDICT = "cue", "feedback_on", "verdict"
"""Event kinds `feed` returns, in the order a trial emits them: the direction cue at baseline start, the
active window opening (absent on transfer trials — that is what a transfer trial is), and the trial's
verdict."""


def thin_baseline(valid: float, expected: float) -> bool:
    """The baseline rule: fewer than `MIN_BASELINE_FRACTION` of the readings a baseline expects are valid."""
    return valid < MIN_BASELINE_FRACTION * expected


def too_many_invalid(invalid: float, total: float, limit: float = REJECT_LIMIT) -> bool:
    """The 20% rule: more than `limit` of a trial's readings are missing or not valid."""
    return invalid > limit * total


def succeeds(shift: float, required: str | None, limit: float) -> bool:
    """Whether a shift is past `limit` on the required side: the verdict's rule, and the one every output
    that says "good" during a trial asks. A zero shift is no shift in either direction, even at a limit of 0."""
    oriented = shift if required == POSITIVE else -shift
    return oriented >= limit and shift != 0.0


@dataclass(frozen=True)
class Trial:
    """One completed trial, as history."""
    index: int
    required: str               # POSITIVE or NEGATIVE
    transfer: bool
    score_uv: float             # mean shift over the active window, µV
    valid: bool                 # False: artifacted or truncated, excluded from every rate
    success: bool               # valid and the mean shift past the limit on the required side
    cause: str | None = None    # one of CAUSES when void, None when valid
    invalid_share: float = 0.0  # the part of the trial's expected readings that were not valid
    cause_share: float = 0.0    # the part of them that `cause` alone accounts for
    scale: Any = None           # the session's record of the limit and the range, from the feedback start


@dataclass(frozen=True)
class Event:
    kind: str                   # CUE, FEEDBACK_ON or VERDICT
    required: str
    transfer: bool
    trial: Trial | None = None  # the verdict's payload


@dataclass
class TrialRun:
    """The trial sequence, fed one reading at a time. Owns no signal and no screen.

    `feed(t, value, valid)` with `t` in seconds of signal and `value` the metric's µV level returns the
    events that fell due. Everything a screen draws is a property; everything a session feeds back is
    `shift` and `feedback_live`.
    """

    n_trials: int = DEFAULT_TRIALS
    baseline_s: float = BASELINE_S
    active_s: float = ACTIVE_S
    result_s: float = RESULT_S
    iti_s: tuple[float, float] = ITI_RANGE_S
    transfer_fraction: float = TRANSFER_FRACTION
    directions: tuple[str, ...] = (POSITIVE, NEGATIVE)
    """Which directions the run asks for. One direction trains it alone; two are balanced in each block."""
    alternate: bool = False
    """Two directions strictly in turn, instead of shuffled in each block."""
    cue_s: float = 0.0
    """Time from the cue to the start of the baseline. Zero puts the cue on the baseline's first reading."""
    limit_uv: float = 0.0
    """Beyond-the-baseline margin the mean shift must clear, µV. Zero: any shift on the required side. A
    session writes it at each feedback start, so a trial is scored against the limit it was shown."""
    scale: Any = None
    """The session's record of the limit and the range, written with `limit_uv` and kept on the trial."""
    reject_limit: float = REJECT_LIMIT
    metric_hz: float = 20.0
    """What rate the readings arrive at, for sizing the baseline-sufficiency check. The session passes
    `true_metric_hz`, which is not quite the asked-for 20 — see that function's docstring."""
    seed: int | None = None
    """Fixed for a reproducible sequence — a test, a replay. None draws fresh, which is a session."""

    phase: str = field(default=ITI, init=False)
    results: list[Trial] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._plan: list[tuple[str, bool]] = []
        self._transfer_owed = 0.0
        self._t: float | None = None
        self._phase_until: float | None = None
        self.required: str | None = None
        self.transfer: bool = False
        self._baseline_sum = 0.0
        self._baseline_n = 0
        self.baseline_uv: float | None = None
        self._shift_sum = 0.0
        self._shift_n = 0
        self._invalid = 0
        self._counts: dict[str, int] = {}       # this trial's invalid readings, by cause
        self.shift: float | None = None
        self.last: Trial | None = None
        self.phase_t: dict[str, float] = {}
        """When each phase of the trial in flight began: the time of the reading, real or lost, after
        which its readings count. Replaced whole at each cue."""

    # --- the sequence -----------------------------------------------------------------------

    def _next_assignment(self) -> tuple[str, bool]:
        """Direction and transfer for the next trial, balanced within blocks of `BLOCK`.

        Directions come half-and-half per block, shuffled, so the longest possible run is one block's
        tail into the next block's head. Transfer slots accumulate fractionally per block and are drawn
        without replacement, so a third means a third of every six rather than a coin per trial that can
        cluster them all at the start of a run.
        """
        if not self._plan:
            half = BLOCK // 2
            if len(set(self.directions)) == 1:
                directions = [self.directions[0]] * BLOCK
            elif self.alternate:
                directions = [POSITIVE, NEGATIVE] * half
            else:
                directions = [POSITIVE] * half + [NEGATIVE] * (BLOCK - half)
                self._rng.shuffle(directions)
            self._transfer_owed += self.transfer_fraction * BLOCK
            owed = min(BLOCK, int(self._transfer_owed))
            self._transfer_owed -= owed
            slots = self._rng.sample(range(BLOCK), owed)
            self._plan = [(d, i in slots) for i, d in enumerate(directions)]
        return self._plan.pop(0)

    # --- state a screen reads ---------------------------------------------------------------

    @property
    def trial_number(self) -> int:
        """1-based number of the trial in progress or about to start, capped at the run length."""
        return min(len(self.results) + 1, self.n_trials)

    @property
    def done(self) -> bool:
        return len(self.results) >= self.n_trials

    @property
    def started(self) -> bool:
        """Whether the run has had its first reading, which starts the lead-in to the first cue."""
        return self._t is not None

    @property
    def feedback_live(self) -> bool:
        """Whether the level should be driven right now: an active window, and not a transfer trial."""
        return self.phase == ACTIVE and not self.transfer

    @property
    def mean_shift(self) -> float | None:
        """The mean shift of the active window so far, which is what the trial will be scored on. None
        outside an active window, and before its first valid reading."""
        if self.phase != ACTIVE or not self._shift_n:
            return None
        return self._shift_sum / self._shift_n

    def phase_remaining(self, t: float | None = None) -> float:
        at = self._t if t is None else t
        if at is None or self._phase_until is None:
            return 0.0
        return max(0.0, self._phase_until - at)

    @property
    def valid_results(self) -> list[Trial]:
        return [r for r in self.results if r.valid]

    def success_rate(self, required: str | None = None) -> float | None:
        """Fraction of valid trials that met their direction — overall, or for one direction. None before
        there is anything to rate, because 0% of nothing reads as failure."""
        pool = [r for r in self.valid_results if required is None or r.required == required]
        if not pool:
            return None
        return sum(r.success for r in pool) / len(pool)

    # --- the feed ---------------------------------------------------------------------------

    def feed(self, t: float, value: float, valid: bool = True, cause: str | None = None,
             cut: str | None = None) -> list[Event]:
        """One reading of the metric. Returns the events that fell due, oldest first. `cause` says why an
        invalid reading is invalid, LIMIT, NO_SIGNAL, EXTRA or REJECTED; an invalid reading with none counts
        as REJECTED.
        `cut` says the signal before this reading does not precede it, and why: PAUSED or SEAM."""
        events: list[Event] = []
        last = self._t
        # A backward step is a discontinuity like a forward gap (a device clock can wrap or glitch), and it
        # would stall the phase clock, whose deadline is then never reached again. A cut can leave the
        # clock continuous and is a discontinuity all the same.
        if last is not None and (cut is not None or t - last > JUMP_S or t < last):
            events += self._abort(cut or GAP)
            last = None
        if last is None:
            self._phase_until = t + LEADIN_S
            self.phase = ITI
        else:
            # Under a jump, the readings the clock expected and did not get go in at their own times.
            missing = int((t - last) * self.metric_hz + 0.5) - 1
            for k in range(1, missing + 1):
                events += self._step(last + k / self.metric_hz, 0.0, False, LOST)
        events += self._step(t, value, valid, cause)
        return events

    def _invalid_reading(self, cause: str) -> None:
        self._invalid += 1
        self._counts[cause] = self._counts.get(cause, 0) + 1

    def _step(self, t: float, value: float, valid: bool, cause: str | None) -> list[Event]:
        events: list[Event] = []
        self._t = t
        if self.done:
            return events

        # Accumulate into the phase we are in *before* checking whether it ends at this reading, so the
        # reading that lands exactly on a boundary belongs to the window it closes.
        if self.phase == BASELINE:
            if valid:
                self._baseline_sum += value
                self._baseline_n += 1
            else:
                self._invalid_reading(cause or REJECTED)
        elif self.phase == ACTIVE:
            if not valid:
                self._invalid_reading(cause or REJECTED)
            elif self.baseline_uv is None:
                self._invalid += 1      # a good reading with no baseline to measure it against
            else:
                self.shift = value - self.baseline_uv
                self._shift_sum += self.shift
                self._shift_n += 1

        while self._phase_until is not None and t >= self._phase_until and not self.done:
            if self.phase == ITI:
                self.phase_t = {}
                self.required, self.transfer = self._next_assignment()
                self.phase = CUED if self.cue_s > 0 else BASELINE
                self._phase_until += self.cue_s if self.cue_s > 0 else self.baseline_s
                self._baseline_sum = 0.0
                self._baseline_n = 0
                self._invalid = 0
                self._counts = {}
                self.scale = None
                self.baseline_uv = None
                self.shift = None
                events.append(Event(CUE, self.required, self.transfer))
            elif self.phase == CUED:
                self.phase = BASELINE
                self._phase_until += self.baseline_s
            elif self.phase == BASELINE:
                thin = thin_baseline(self._baseline_n, self.baseline_s * self.metric_hz)
                self.baseline_uv = (self._baseline_sum / self._baseline_n
                                    if self._baseline_n and not thin else None)
                self.phase = ACTIVE
                self._phase_until += self.active_s
                self._shift_sum = 0.0
                self._shift_n = 0
                if not self.transfer:
                    events.append(Event(FEEDBACK_ON, self.required, self.transfer))
            elif self.phase == ACTIVE:
                events.append(Event(VERDICT, self.required, self.transfer, self._score()))
                self.phase = RESULT
                self._phase_until += self.result_s
                self.shift = None
            elif self.phase == RESULT:
                self.phase = ITI
                self._phase_until += self._rng.uniform(*self.iti_s)
            self.phase_t[self.phase] = t
        return events

    def _score(self) -> Trial:
        total = self._baseline_n + self._shift_n + self._invalid
        share = self._invalid / total if total else 0.0
        artifacted = too_many_invalid(self._invalid, total, self.reject_limit)
        score = self._shift_sum / self._shift_n if self._shift_n else 0.0
        valid = (self.baseline_uv is not None and self._shift_n > 0 and not artifacted)
        cause = None if valid else self._cause()
        return self._record(score, valid, valid and succeeds(score, self.required, self.limit_uv), cause,
                            share, self._counts.get(cause, 0) / total if total else 0.0)

    def _cause(self) -> str:
        """The first cause that is not a count, else the count with the most readings: see `CAUSES`."""
        shown = {cause for cause, n in self._counts.items() if n}
        if self.baseline_uv is None:
            shown.add(THIN_BASELINE)
        fixed = next((cause for cause in CAUSES if cause in shown and cause not in COUNTS), None)
        if fixed is not None:
            return fixed
        counted = [cause for cause in COUNTS if cause in shown]
        return max(counted, key=lambda cause: self._counts[cause]) if counted else REJECTED

    def _record(self, score: float, valid: bool, success: bool, cause: str | None, share: float,
                cause_share: float = 0.0) -> Trial:
        trial = Trial(index=len(self.results) + 1, required=self.required or POSITIVE,
                      transfer=self.transfer, score_uv=score, valid=valid, success=success, cause=cause,
                      invalid_share=share, cause_share=cause_share, scale=self.scale)
        self.results.append(trial)
        self.last = trial
        return trial

    def _abort(self, cause: str = GAP) -> list[Event]:
        """A gap mid-trial: the trial in flight is void, because its baseline describes signal that no longer
        precedes it. It stays in the history, since the person did it, and the run rests before the next
        cue. Returns its verdict, as the end of any other trial does."""
        events: list[Event] = []
        if self.phase in (CUED, BASELINE, ACTIVE):
            if self.required is not None:
                trial = self._record(0.0, False, False, cause, 1.0)
                events.append(Event(VERDICT, trial.required, trial.transfer, trial))
            self.phase = ITI
            self.required = None
            self.transfer = False
            self.shift = None
            self.baseline_uv = None
        # Whatever phase we were in, the clock restarts from the new time.
        self._phase_until = None
        self._t = None
        return events
