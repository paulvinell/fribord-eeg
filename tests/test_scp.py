"""The SCP metric and the trial engine, each against inputs whose answer is known.

Two halves, mirroring `test_wave.py`'s split. The metric half checks `nfb.dsp.scp.SlowShift` recovers a
sustained shift and that `shift_survival` tells the truth about the amplifier's corner — the number the
screen's wiring check stands on. The engine half drives `nfb_console.trials.TrialRun` with synthetic
readings, because trial structure is exactly the kind of logic that regresses silently: a mis-scored trial
looks like a trainee having a bad day.
"""

from __future__ import annotations

import numpy as np

from nfb.dsp import SlowShift, shift_survival
from nfb.dsp.scp import SCP_LOWPASS_HZ
from nfb.stream import Block, ChannelSpec
from nfb_console import trials
from nfb_console.trials import (ACTIVE, BASELINE, CUE, FEEDBACK_ON, ITI, NEGATIVE, POSITIVE,
                                VERDICT, TrialRun)

RATE = 250.0
SPEC = ChannelSpec(0, RATE, "µV", 800.0 / 32767)


def _feed(metric, x: np.ndarray, start: int = 0) -> np.ndarray:
    scale = 1.0 / SPEC.scale
    out: list[tuple[float, float]] = []
    for i in range(0, len(x) - 7, 8):
        block = Block(SPEC.index, start + i, tuple(int(v * scale) for v in x[i:i + 8]), SPEC)
        for r in metric.feed(block):
            out.append((r.t, r.value))
    return np.asarray(out) if out else np.zeros((0, 2))


# --- the metric ----------------------------------------------------------------------------------

def test_a_sustained_shift_is_recovered_under_the_ongoing_eeg():
    """A 10 µV step held for six seconds, with 20 µV of alpha on top: the reading settles on the step and
    the alpha is gone — which is the lowpass's whole job."""
    metric = SlowShift(SPEC)
    t = np.arange(int(12 * RATE)) / RATE
    step = np.where(t >= 4.0, 10.0, 0.0)
    x = step + 20.0 * np.sin(2 * np.pi * 10.0 * t)
    readings = _feed(metric, x)
    before = readings[(readings[:, 0] > 2.0) & (readings[:, 0] < 3.8), 1]
    after = readings[readings[:, 0] > 7.0, 1]
    assert abs(float(np.mean(before))) < 1.0
    assert abs(float(np.mean(after)) - 10.0) < 1.0
    # the alpha does not survive: the settled readings barely wobble
    assert float(np.std(after)) < 1.0


def test_a_gap_resets_the_filter_and_the_settling_clock():
    metric = SlowShift(SPEC)
    _feed(metric, np.full(int(4 * RATE), 5.0))
    assert metric.warmup >= 1.0
    _feed(metric, np.full(80, 5.0), start=int(20 * RATE))     # a jump in sample indices
    assert metric.warmup < 1.0, "settling must restart with the filters"


def test_shift_survival_says_the_corner_verdict():
    """The numbers the screen's wiring check stands on: 0.01 Hz keeps most of a six-second shift's mean,
    the 0.5 Hz default keeps almost none of it."""
    good = shift_survival(0.01, 6.0)
    bad = shift_survival(0.5, 6.0)
    assert 0.78 < good < 0.88
    assert bad < 0.08
    assert shift_survival(0.0, 6.0) == 1.0


def test_the_metric_states_its_band_and_its_reason():
    text = SlowShift(SPEC).describe()
    assert f"{SCP_LOWPASS_HZ:g}" in text and "de-trender" in text


# --- the engine: driving it ----------------------------------------------------------------------

def _run(run: TrialRun, seconds: float, value=lambda t, run: 0.0, valid=lambda t: True,
         start: float = 0.0, hz: float = 20.0) -> list:
    """Feed a run for a stretch of signal time, collecting events. `value` may read the run's own state,
    which is how a compliant trainee is simulated."""
    events = []
    n = int(seconds * hz)
    for i in range(n):
        t = start + i / hz
        events.extend(run.feed(t, value(t, run), valid(t)))
    return events


def _compliant(t, run: TrialRun) -> float:
    """Produces +8 µV when asked positive, −8 when asked negative, 0 otherwise."""
    if run.phase == ACTIVE and run.required == POSITIVE:
        return 8.0
    if run.phase == ACTIVE and run.required == NEGATIVE:
        return -8.0
    return 0.0


def test_a_compliant_trainee_succeeds_on_every_trial():
    run = TrialRun(n_trials=12, transfer_fraction=0.0, seed=7)
    events = _run(run, 240.0, value=_compliant)
    assert run.done
    verdicts = [e for e in events if e.kind == VERDICT]
    assert len(verdicts) == 12
    assert all(v.trial.valid and v.trial.success for v in verdicts)
    assert run.success_rate() == 1.0
    # scored in µV, on the required side, at about the produced size
    for v in verdicts:
        expected = 8.0 if v.required == POSITIVE else -8.0
        assert abs(v.trial.score_uv - expected) < 1.0


def test_directions_are_balanced_and_never_run_long():
    run = TrialRun(n_trials=36, transfer_fraction=0.0, seed=3)
    _run(run, 720.0, value=_compliant)
    required = [r.required for r in run.results]
    assert abs(required.count(POSITIVE) - required.count(NEGATIVE)) <= trials.BLOCK // 2
    longest = run_len = 1
    for a, b in zip(required, required[1:]):
        run_len = run_len + 1 if a == b else 1
        longest = max(longest, run_len)
    assert longest <= trials.BLOCK, "a block's tail into the next block's head is the worst case"


def test_a_wrong_way_shift_fails_and_a_flat_one_is_not_a_success():
    run = TrialRun(n_trials=6, transfer_fraction=0.0, seed=1)
    _run(run, 120.0, value=lambda t, r: 8.0 if r.phase == ACTIVE else 0.0)
    assert run.done
    positive = [r for r in run.results if r.required == POSITIVE]
    negative = [r for r in run.results if r.required == NEGATIVE]
    assert all(r.success for r in positive)
    assert not any(r.success for r in negative), "an upward shift is a failure on a downward trial"

    flat = TrialRun(n_trials=4, transfer_fraction=0.0, seed=1)
    _run(flat, 90.0, value=lambda t, r: 0.0)
    assert flat.done and not any(r.success for r in flat.results), \
        "no shift is not a shift in either direction"


def test_transfer_trials_exist_withhold_feedback_and_still_score():
    run = TrialRun(n_trials=18, transfer_fraction=1.0 / 3.0, seed=5)
    events = _run(run, 400.0, value=_compliant)
    transfers = [r for r in run.results if r.transfer]
    assert len(transfers) == 6, "a third of every block, not a coin per trial"
    assert all(r.success for r in transfers), "the verdict still lands"
    # FEEDBACK_ON never fires for a transfer trial, and feedback_live was never true during one
    assert all(not e.transfer for e in events if e.kind == FEEDBACK_ON)
    cues = [e for e in events if e.kind == CUE]
    assert sum(e.transfer for e in cues) == 6


def test_an_artifacted_trial_is_void_and_out_of_every_rate():
    run = TrialRun(n_trials=4, transfer_fraction=0.0, seed=2)
    # everything invalid from 30 s on: whichever trials that stretch covers must be void
    _run(run, 90.0, value=_compliant, valid=lambda t: t < 30.0)
    assert run.done
    void = [r for r in run.results if not r.valid]
    assert void, "a fully-rejected stretch has to void at least one trial"
    assert all(not r.success for r in void)
    rate = run.success_rate()
    assert rate is None or rate == 1.0, "void trials must not drag the rate down"


def test_a_time_jump_aborts_the_trial_in_flight():
    run = TrialRun(n_trials=8, transfer_fraction=0.0, seed=4)
    _run(run, 5.0, value=_compliant)                 # lead-in (2 s) + baseline, into the first active
    assert run.phase in (BASELINE, ACTIVE)
    before = len(run.results)
    _run(run, 30.0, value=_compliant, start=100.0)   # a suspend: signal time leaps
    assert len(run.results) >= before + 1
    aborted = run.results[0]
    assert aborted.index == 1
    # the interrupted trial was discarded, not scored: the first *recorded* trial began after the jump,
    # so its verdict is a success like every other compliant one
    assert all(r.valid and r.success for r in run.results)


def test_rates_are_none_before_there_is_anything_to_rate():
    run = TrialRun(seed=0)
    assert run.success_rate() is None
    assert run.success_rate(POSITIVE) is None
    assert run.phase == ITI and not run.done
