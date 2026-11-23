"""The trial engine's record of each trial, the one line that prints it, and the run's verdict at its end.

`test_scp.py` drives `TrialRun` for the shape of a run. This file is about what a trial records and who
may say what about it: why a void trial has no result, counted from the readings of that trial; the one
limit a trial is scored against, written once at its feedback start; the one function that turns a trial
into words under the circle; and the words at the end of a run, which say what chance does with the
run's own scores and nothing before the run is done.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from nfb_console.trials import (ACTIVE, ACTIVE_S, BASELINE, BASELINE_S, FEEDBACK_ON, ITI_RANGE_S, LEADIN_S,
                                NEGATIVE, POSITIVE, RESULT, RESULT_S, Trial, TrialRun)

from conftest import BETAS, STEPS_UV, wandering

HZ = 20.0
WORDS = {POSITIVE: "up", NEGATIVE: "down"}


# --- one trial, reading by reading -------------------------------------------------------------

def _trial(baseline=lambda i: "ok", active=lambda i: "ok", lead=lambda i: "ok", n_trials: int = 1):
    """One trial of a fresh run. Each pattern names, per reading of its phase: "ok", "rejected",
    "limit", or "hole" (no reading arrives). Returns the run once the trial has its verdict."""
    run = TrialRun(n_trials=n_trials, transfer_fraction=0.0, seed=1, directions=(POSITIVE,), metric_hz=HZ)
    counts = {None: 0}
    t, last = 0.0, None
    while not run.results:
        phase = run.phase if run.started else None
        key = phase if phase in (BASELINE, ACTIVE) else None
        i = counts.setdefault(key, 0)
        counts[key] = i + 1
        kind = (baseline if key == BASELINE else active if key == ACTIVE else lead)(i)
        if kind != "hole":
            if kind == "ok":
                run.feed(t, 0.0)
            elif kind == "rejected":
                run.feed(t, 0.0, valid=False)
            else:
                run.feed(t, 0.0, valid=False, cause=kind)
            last = t
        t += 1.0 / HZ
        assert t < 60.0 and (last is None or t - last < 5.0), "the trial never reached its verdict"
    return run


def test_a_hole_under_a_second_counts_its_readings_invalid():
    """Readings that never arrive are not readings that were fine. Two holes of 0.9 s are 22% of a trial."""
    hole = lambda i: "hole" if 20 <= i < 38 or 60 <= i < 78 else "ok"       # noqa: E731
    trial = _trial(active=hole).results[-1]
    assert not trial.valid, "missing readings count against the trial"
    assert trial.cause == "lost"


def test_a_pause_over_a_second_leaves_a_void_cell():
    """A trial cut by a jump in time is a trial the person did. It stays in the history as void."""
    run = TrialRun(n_trials=3, transfer_fraction=0.0, seed=1, directions=(POSITIVE,), metric_hz=HZ)
    t = 0.0
    while run.phase != ACTIVE:
        run.feed(t, 0.0)
        t += 1.0 / HZ
    for _ in range(30):
        run.feed(t, 0.0)
        t += 1.0 / HZ
    run.feed(t + 3.0, 0.0)
    assert len(run.results) == 1, "the cut trial has a cell"
    assert not run.results[0].valid and run.results[0].cause == "gap"


def test_each_void_cause_alone():
    limit = _trial(active=lambda i: "limit" if i < 40 else "ok").results[-1]
    rejected = _trial(active=lambda i: "rejected" if i < 40 else "ok").results[-1]
    thin = _trial(baseline=lambda i: "rejected" if i < 25 else "ok").results[-1]
    assert (limit.valid, limit.cause) == (False, "limit")
    assert (rejected.valid, rejected.cause) == (False, "rejected")
    assert (thin.valid, thin.cause) == (False, "baseline")
    clean = _trial().results[-1]
    assert clean.valid and clean.cause is None


def test_thin_baseline_wins_over_20_percent():
    """A thin baseline also puts the trial over 20%, because no reading after it has a baseline to be
    measured against. The cause is the baseline, so no percentage of rejected readings is printed."""
    trial = _trial(baseline=lambda i: "rejected" if i < 30 else "ok",
                   active=lambda i: "rejected" if i < 10 else "ok").results[-1]
    assert (trial.valid, trial.cause) == (False, "baseline")


def test_the_input_limit_wins_over_rejected_readings():
    trial = _trial(active=lambda i: "limit" if i < 20 else "rejected" if i < 40 else "ok").results[-1]
    assert (trial.valid, trial.cause) == (False, "limit")


def test_rail_only_before_the_cue_reads_rejected():
    """The cause comes from the readings of this trial. A channel at its input limit before the cue, and
    clear by the baseline, does not name the limit for a trial lost to other rejections."""
    trial = _trial(lead=lambda i: "limit", active=lambda i: "rejected" if i < 40 else "ok").results[-1]
    assert (trial.valid, trial.cause) == (False, "rejected")


def test_limit_uv_is_a_field_that_the_score_reads():
    """The session writes the limit at each feedback start. The engine scores against that field and no
    other, and a run built by hand scores against 0 µV."""
    run = TrialRun(n_trials=2, transfer_fraction=0.0, seed=1, directions=(POSITIVE,), metric_hz=HZ)
    assert run.limit_uv == 0.0
    limits = iter([5.0, 0.0])
    t = 0.0
    while not run.done:
        value = 3.0 if run.phase == ACTIVE else 0.0
        for event in run.feed(t, value):
            if event.kind == FEEDBACK_ON:
                run.limit_uv = next(limits)
        t += 1.0 / HZ
    first, second = run.results
    assert first.valid and first.score_uv == pytest.approx(3.0)
    assert not first.success, "3 µV is short of a 5 µV limit"
    assert second.success, "and past a 0 µV one"


def test_only_freeze_scale_writes_the_frozen_record():
    """One record per trial, written at one point. Every reader in the trial takes it from there."""
    import nfb_console

    writers: dict[str, list[str]] = {}
    for path in Path(nfb_console.__file__).parent.glob("*.py"):
        source = path.read_text()
        if "trial_scale" not in source and "limit_uv" not in source:
            continue
        tree = ast.parse(source)
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
                value = getattr(node, "value", None)
                for target in targets:
                    for leaf in (target.elts if isinstance(target, ast.Tuple) else [target]):
                        if (isinstance(leaf, ast.Attribute) and leaf.attr in ("trial_scale", "limit_uv")
                                and not (isinstance(value, ast.Constant) and value.value in (None, 0.0))):
                            writers.setdefault(leaf.attr, []).append(f"{path.name}:{func.name}")
    assert writers.get("trial_scale") and writers.get("limit_uv"), "the record is written somewhere"
    strays = [w for ws in writers.values() for w in ws if not w.endswith(":_freeze_scale")]
    assert not strays, f"written outside _freeze_scale: {strays}"


# --- the line under the circle ------------------------------------------------------------------

def _void_run(active=lambda i: "rejected" if i < 40 else "ok"):
    """A run in its verdict phase, its one trial void."""
    run = _trial(active=active, n_trials=2)
    assert run.phase == RESULT and not run.last.valid
    return run


def _ui(run):
    from nfb import ReplaySource
    from nfb_console import scp_view
    from nfb_console.protocols import BY_CODE
    from nfb_console.trial_training import TrialView

    from conftest import BENCH_OK

    session = scp_view.SCPSession(BY_CODE["SCP"], source=ReplaySource(str(BENCH_OK), realtime=False),
                                  channels=2)
    session.trials = run
    ui = TrialView(session, scp_view.CIRCLE_SIDE, heart=False)
    ui.fixation_panel()
    return session, ui


def test_draw_fixation_prints_a_void_trial_through_verdict_line(monkeypatch):
    from nfb_console import trial_training

    seen = []
    original = trial_training.verdict_line

    def spy(last, *args, **kwargs):
        seen.append(last)
        return original(last, *args, **kwargs)

    monkeypatch.setattr(trial_training, "verdict_line", spy)
    session, ui = _ui(_void_run())
    try:
        ui.draw_fixation("")
        assert [t for t in seen if not t.valid], "a void trial reaches the screen through verdict_line"
    finally:
        session.stop()


def test_a_void_trial_shows_its_cause_and_an_action_under_the_circle():
    session, ui = _ui(_void_run())
    try:
        ui.draw_fixation("")
        assert ui.verdict_text.value == "no result · the program rejects 25% of the readings · the maximum is 20%"
        assert ui.verdict_detail.value and ui.verdict_detail.value != ui.verdict_text.value, \
            "the second line says what to do"
    finally:
        session.stop()


def test_only_verdict_line_prints_a_verdict():
    """Every write to the two lines under the circle is a blank or comes from `verdict_line`."""
    import nfb_console

    strays = []
    for path in Path(nfb_console.__file__).parent.glob("*.py"):
        source = path.read_text()
        if "verdict_text" not in source and "verdict_detail" not in source:
            continue
        tree = ast.parse(source)
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = "verdict_line(" in (ast.get_source_segment(source, func) or "")
            for node in ast.walk(func):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    for leaf in (target.elts if isinstance(target, ast.Tuple) else [target]):
                        if not (isinstance(leaf, ast.Attribute) and leaf.attr == "value"
                                and isinstance(leaf.value, ast.Attribute)
                                and leaf.value.attr in ("verdict_text", "verdict_detail")):
                            continue
                        blank = isinstance(node.value, ast.Constant) and node.value.value == ""
                        if not blank and not calls:
                            strays.append(f"{path.name}:{func.name}:{node.lineno}")
                        if isinstance(node.value, ast.Constant) and node.value.value:
                            strays.append(f"{path.name}:{func.name}:{node.lineno} writes a literal")
    assert not strays, strays


# --- the run's verdict, at its end ----------------------------------------------------------------

def _plan(seed: int, n: int, directions=(POSITIVE, NEGATIVE), transfer: float = 1 / 3):
    run = TrialRun(n_trials=n, transfer_fraction=transfer, seed=seed, directions=directions,
                   alternate=False)
    rng = np.random.default_rng(20_000 + seed)
    t, out = LEADIN_S, []
    for _ in range(n):
        required, is_transfer = run._next_assignment()
        out.append((t, required, is_transfer))
        t += BASELINE_S + ACTIVE_S + RESULT_S + rng.uniform(*ITI_RANGE_S)
    return out, t


def _scores(x, plan) -> np.ndarray:
    nb, na = int(BASELINE_S * HZ), int(ACTIVE_S * HZ)
    starts = np.array([int(t * HZ) for t, _, _ in plan])
    c = np.r_[0.0, np.cumsum(x)]
    return (c[starts + nb + na] - c[starts + nb]) / na - (c[starts + nb] - c[starts]) / nb


def _run_of(seed: int, skill: float = 0.0, n: int = 40, keep: int | None = None,
            directions=(POSITIVE, NEGATIVE), extra=None):
    """A run of `n` trials on a wandering level, `skill` chance spreads toward each cue, as the engine
    would have scored it. `keep` keeps only the first trials, as a run not yet done. `extra` adds the scores
    of an extra channel in the same windows: (its noise relative to the training channel, its share of
    the cued shift)."""
    beta, step = BETAS[seed % len(BETAS)], STEPS_UV[seed % len(STEPS_UV)]
    plan, end = _plan(seed, n, directions)
    raw = _scores(wandering(beta, end + 2.0, HZ, 40_000 + seed, step), plan)
    spread = float(np.std(raw))
    sign = np.array([1.0 if r == POSITIVE else -1.0 for _, r, _ in plan])
    scores = raw + skill * spread * sign
    run = TrialRun(n_trials=n, directions=directions)
    run.results = [Trial(index=i + 1, required=r, transfer=tr, score_uv=float(s), valid=True,
                         success=bool(s * g > 0))
                   for i, ((_, r, tr), s, g) in enumerate(zip(plan, scores, sign))][:keep]
    if extra is None:
        return run, None
    noise, share = extra
    own = _scores(wandering(beta, end + 2.0, HZ, 60_000 + seed, step * noise), plan)
    return run, {1: list(own + share * skill * spread * sign)[:keep]}


def _summary(run, checks=None) -> str:
    from nfb_console.trial_training import trial_summary

    return trial_summary(run, WORDS, "µV", 1, checks=checks)


def test_run_verdict_words_only_when_the_run_is_done():
    """The words at the end judge the run once, when it is done. Words after every trial would be a
    look after every trial, and chance would pass one of them in about one run of four."""
    partial, _ = _run_of(1, skill=3.0, keep=20)
    assert not partial.done
    assert "keep this strategy" not in _summary(partial) and "chance gives this" not in _summary(partial)
    done, _ = _run_of(1, skill=3.0)
    text = _summary(done)
    assert "past the chance range · keep this strategy" in text, text
    flat = TrialRun(n_trials=4)
    flat.results = [Trial(index=i + 1, required=r, transfer=False, score_uv=0.0, valid=True, success=False)
                    for i, r in enumerate((POSITIVE, NEGATIVE, POSITIVE, NEGATIVE))]
    assert "keep this strategy" not in _summary(flat), "a flat signal earns no words"


def test_chance_words_stay_rare_on_wandering_signals_without_skill():
    """500 runs with no skill at all, across the slopes and the scales: the words that tell the person to
    keep a strategy come in at most 8% of them. With skill of one and a half chance spreads, they come in
    nearly every run, so the 8% is the comparison working and not the words missing."""
    chance = sum("keep this strategy" in _summary(_run_of(seed)[0]) for seed in range(500))
    assert chance <= 40, f"chance words in {chance} of 500 runs with no skill"
    skilled = sum("keep this strategy" in _summary(_run_of(seed, skill=1.5)[0]) for seed in range(40))
    assert skilled >= 36, f"words in {skilled} of 40 runs with skill"


def test_chance_passes_the_range_in_the_cued_direction_in_one_run_of_twenty():
    """Scores with no skill in them, for runs of several lengths: the difference is past the chance range
    in the cued direction in 5% of the runs, and as often the wrong way. The printed frequency of a run is
    below 1 in 20 exactly where the run is past the range."""
    import numpy as np

    from nfb_console.trial_training import RUN_ALPHA, _chance_test

    rng = np.random.default_rng(5)
    runs, cued, wrong = 3000, 0, 0
    for k in range(runs):
        n = (4, 10, 30)[k % 3]
        diff, reach, p = _chance_test(list(rng.normal(0, 3.0, n)), list(rng.normal(0, 5.0, n)), RUN_ALPHA)
        cued += diff > reach
        wrong += diff < -reach
        assert (abs(diff) > reach) == (p < RUN_ALPHA), (diff, reach, p)
    for name, count in (("cued", cued), ("wrong", wrong)):
        assert 0.038 < count / runs < 0.062, f"{name}: past the range in {count} of {runs} runs with no skill"


def test_one_direction_is_judged_against_zero():
    run, _ = _run_of(3, skill=1.5, directions=(POSITIVE,))
    assert "keep this strategy" in _summary(run), "a run in one direction has its own comparison"


def test_an_extra_channel_that_followed_the_cues_withholds_keep():
    """A cue-locked shift that also reaches an extra channel, larger there: an eye or an electrode can give
    that, and so can the trained potential. The end says what was measured, and not "keep"."""
    for seed in (1, 2, 3):
        run, checks = _run_of(seed, skill=2.0, extra=(2.0, 4.0))
        text = _summary(run, checks)
        assert "ch1 also followed the cues: up minus down" in text, text
        assert "an eye movement, an electrode movement or the trained potential can give this" in text
        assert "keep this strategy" not in text, text


def test_an_extra_channel_inside_the_chance_range_vouches():
    vouched = 0
    for seed in range(10):
        run, checks = _run_of(seed, skill=2.0, extra=(2.0, 0.0))
        vouched += "ch1 did not follow the cues · checked in 40 of 40 trials · keep this strategy" in \
            _summary(run, checks)
    assert vouched >= 8, f"an extra channel with no cue-locked shift vouched in {vouched} of 10 runs"


def test_an_extra_channel_with_no_signal_does_not_check():
    """A channel that was dead or at its input limit vouches for nothing, and the end says so."""
    run, _ = _run_of(4, skill=2.0)
    text = _summary(run, {2: [None] * len(run.results)})
    assert "ch2 had no signal in 40 of 40 trials · ch2 did not check this run" in text, text
    assert "ch2 did not follow the cues" not in text


def test_trial_summary_keeps_its_old_callers():
    """`trial_summary(run, words, unit)` stays a call that works, for Pupil and HEG."""
    from nfb_console.trial_training import trial_summary

    assert "checks" in inspect.signature(trial_summary).parameters
    assert trial_summary(TrialRun(), WORDS, "µV") == "no trials yet"


def _void_with(lost: int, rejected: int, thin: bool = False):
    """One trial at 10 readings a second, with `lost` single readings that never arrive and `rejected`
    rejected readings in its feedback window, and a baseline too thin to use where `thin`."""
    from nfb_console.trials import BASELINE, REJECTED

    run = TrialRun(n_trials=1, transfer_fraction=0.0, seed=1, metric_hz=10.0)
    t, j, skipped, bad_n = 0.0, 0, 0, 0
    while not run.results:
        if run.phase == ACTIVE:
            j += 1
            if j % 2 == 0 and skipped < lost:
                t, skipped = round(t + 0.1, 6), skipped + 1      # one reading that never arrives
                continue
        bad = (run.phase == ACTIVE and bad_n < rejected) or (thin and run.phase == BASELINE)
        bad_n += run.phase == ACTIVE and bad
        run.feed(t, 5.0, valid=not bad, cause=REJECTED)
        t = round(t + 0.1, 6)
    return run.results[0]


def test_a_void_trial_names_the_count_with_the_most_readings():
    from nfb_console.trials import LOST, REJECTED, THIN_BASELINE

    more_rejected = _void_with(lost=3, rejected=20)
    assert more_rejected.cause == REJECTED, more_rejected
    more_lost = _void_with(lost=20, rejected=3)
    assert more_lost.cause == LOST, more_lost
    assert more_lost.cause_share < more_lost.invalid_share, "the share of the named cause alone"
    assert _void_with(lost=3, rejected=20, thin=True).cause == THIN_BASELINE, "a cause that is not a count first"
