"""HEG's active mode: Pupil's cued trials on the headband's signal, with the shift judged in standard
deviations of the minute before each trial. Readings are pushed in as states; no device, no clock."""

from __future__ import annotations

import math
import random

import pytest

from nfb.heg import HegState
from nfb.heg.processor import QUALITY_GOOD, QUALITY_MOTION

from nfb_console.trials import ACTIVE, BASELINE, CUED, POSITIVE, RESULT


def _session(**settings):
    from nfb_console.heg_view import HEGSession
    from nfb_console.inputs import MendiChoice
    from nfb_console.protocols import BY_CODE

    session = HEGSession(BY_CODE["HEG"], choice=MendiChoice(kind="simulator"))
    session.cue = lambda *_a, **_k: True
    session.set_mode("active")
    for name, value in settings.items():
        setter = getattr(session, f"set_{name}", None)
        setter(value) if setter is not None else setattr(session, name, value)
    session.restart_trials()
    session.paused = False
    return session


def _push(session, seconds: float, t0: float, rise: float = 0.0, spread: float = 0.5,
          motion_in_effort: bool = False, seed: int = 2) -> float:
    """States at 30 a second, HbO − HbR around 10 with noise of `spread`. During an effort the signal
    moves `rise` the asked way. Observed every third state, the screen's ten a second."""
    rng = random.Random(seed)
    t = t0
    for i in range(int(seconds * 30)):
        run = session.trials
        effort = run.phase == ACTIVE and session.mode == "active"
        sign = 1.0 if run.required == POSITIVE else -1.0
        value = 10.0 + rng.gauss(0.0, spread) + (sign * rise if effort else 0.0)
        quality = QUALITY_MOTION if (effort and motion_in_effort) else QUALITY_GOOD
        state = HegState(t=t, smoothed_hbo=value, smoothed_hbr=0.0, quality=quality, samples_seen=1000,
                         windowed_ready=True)
        session.heg.state = state
        session.heg.readings.append(state)
        if i % 3 == 0:
            session.observe()
        t += 1 / 30
    return t


def _first_result(session, t: float, **kwargs):
    while not session.trials.results:
        t = _push(session, 1.0, t, **kwargs)
    return session.trials.results[0], t


def test_a_rise_of_two_spreads_in_the_effort_succeeds_and_reads_about_two_sd():
    session = _session(trial_direction="increase", n_trials=1)
    t = _push(session, 70.0, 1000.0)                     # a minute of signal, so the spread exists
    session.restart_trials()                             # the trial starts now, with a minute behind it
    result, _ = _first_result(session, t, rise=1.0, spread=0.5)
    assert result.valid and result.success
    assert result.score_uv == pytest.approx(2.0, abs=0.4), "1.0 over a spread of 0.5 is 2 sd"
    assert session.trial_spread == pytest.approx(0.5, rel=0.15)


def test_a_signal_that_does_not_move_does_not_succeed_past_a_margin():
    session = _session(trial_direction="increase", n_trials=1, success_at=1.0)
    t = _push(session, 70.0, 1000.0)
    session.restart_trials()
    result, _ = _first_result(session, t, rise=0.0)
    assert result.valid and not result.success


def test_the_first_trial_waits_for_the_signal_its_spread_needs():
    """Opened straight into active mode, the run holds its first cue until there are 20 s of signal, and
    the screen says so. The first trial then has a spread and a result."""
    from nfb_console.heg_view import MIN_SPREAD_S

    session = _session(trial_direction="increase", n_trials=1)
    t = _push(session, MIN_SPREAD_S / 2, 1000.0)
    assert not session.trials.started
    assert "s more" in session.trial_hold()
    result, _ = _first_result(session, t, rise=1.0)
    assert session.trial_spread is not None
    assert result.valid and result.success


def test_signal_in_passive_mode_counts_toward_the_first_trial():
    """A minute in passive mode, then a switch to active: the first trial starts at once, with its spread
    measured in the passive minute."""
    session = _session(trial_direction="increase", n_trials=1)
    session.set_mode("passive")
    t = _push(session, 70.0, 1000.0)
    session.set_mode("active")
    assert session.trial_hold() is None
    result, _ = _first_result(session, t, rise=1.0, spread=0.5)
    assert result.valid
    assert session.trial_spread == pytest.approx(0.5, rel=0.15)


def test_motion_in_the_effort_rejects_its_frames_and_leaves_the_trial_without_a_result():
    session = _session(trial_direction="increase", n_trials=1)
    t = _push(session, 70.0, 1000.0)
    session.restart_trials()
    result, _ = _first_result(session, t, rise=1.0, motion_in_effort=True)
    assert not result.valid


def test_the_baseline_has_no_counting_task():
    """Counting works the prefrontal cortex, which is what HEG measures."""
    session = _session(trial_direction="increase", n_trials=1)
    t = _push(session, 70.0, 1000.0)
    session.restart_trials()
    while session.trials.phase not in (CUED, BASELINE):
        t = _push(session, 0.5, t)
    assert session.count_from is None


def test_the_level_runs_only_in_the_effort_and_rises_with_the_signal():
    session = _session(trial_direction="increase", n_trials=1)
    t = _push(session, 70.0, 1000.0)
    session.restart_trials()
    while session.trials.phase != ACTIVE:
        t = _push(session, 0.5, t)
        assert session.signal is None or session.trials.phase == ACTIVE
    t = _push(session, 10.0, t, rise=1.0)
    assert session.signal is not None and session.signal > 0.7
    while session.trials.phase == ACTIVE:
        t = _push(session, 0.5, t, rise=1.0)
    assert session.trials.phase in (RESULT, "iti") and session.signal is None


def test_the_screen_draws_the_trials_in_active_mode():
    import sys

    sys.path.insert(0, "tests")
    from harness import StubPage, _texts, toolbar

    from nfb_console import heg_view

    session = _session(trial_direction="increase", n_trials=2)
    control = heg_view.view(StubPage(), session, toolbar(), lambda: None)
    t = _push(session, 70.0, 1000.0)
    session.restart_trials()
    _first_result(session, t, rise=1.0)
    session.redraw()
    texts = " | ".join(c.value for c in _texts(control) if c.value)
    for shown in ("FIXATION", "TRIALS", "succeeds at", "full at", "sd", "higher: 1 of 1"):
        assert shown in texts, shown
    assert "scaled by" not in texts, "the trials judge the shift in their own unit, so the scaling row goes"


# --- the limit on the circle, and the void line ------------------------------------------------------

def _drawn(session):
    from nfb_console.heg_view import CIRCLE_SIDE
    from nfb_console.trial_training import TrialView

    ui = TrialView(session, CIRCLE_SIDE, heart=False)
    ui.fixation_panel()
    ui.draw_fixation("")
    return ui


def test_a_thin_ring_marks_a_typed_limit_on_heg():
    """One thin ring at the radius of **succeeds at** on the cued side, from the trial's own record."""
    import flet.canvas as cv
    from nfb_console import theme
    from nfb_console.heg_view import CIRCLE_SIDE
    from nfb_console.trial_training import _radii

    session = _session(trial_direction="increase", n_trials=2, success_at=1.0)
    t = _push(session, 70.0, 1000.0)
    session.restart_trials()
    while not (session.trials.phase == ACTIVE and session.feedback_now):
        t = _push(session, 0.2, t, rise=0.5)
    t = _push(session, 1.0, t, rise=0.5)
    shapes = _drawn(session).circle.shapes
    assert not [s for s in shapes if isinstance(s, cv.Arc)], "no blocks on the circle"
    r_min, r_max, r_mid = _radii(CIRCLE_SIDE)
    want = r_min + session.circle_level(1.0) * (r_max - r_min)
    rings = [s for s in shapes if isinstance(s, cv.Circle) and s.radius == pytest.approx(want)]
    assert len(rings) == 1, "one ring at the typed limit"
    assert rings[0].paint.color == theme.ACCENT
    assert rings[0].paint.stroke_width == pytest.approx(1.5 * r_mid / want), "a thin line"


def test_a_heg_trial_with_no_level_spread_says_so():
    """A level that does not move has no spread to judge a trial in, so the trial has no result. That is
    not a rejection, and the line under the circle does not call it one."""
    session = _session(trial_direction="increase", n_trials=2)
    result, _ = _first_result(session, 1000.0, spread=0.0)
    assert not result.valid
    ui = _drawn(session)
    lines = f"{ui.verdict_text.value} | {ui.verdict_detail.value}"
    assert "spread" in lines and "rejected" not in lines, lines


def test_the_trial_length_is_the_run_s_baseline_and_feedback_and_the_run_follows_it():
    """The trial length shown is the run's baseline and feedback window together, and a new length moves
    the feedback window of the run."""
    session = _session()
    try:
        run = session.trials
        assert session.trial_length == pytest.approx(run.baseline_s + run.active_s)
        before = run.active_s
        session.set_trial_length(session.trial_length + 10.0)
        assert session.trials.active_s == pytest.approx(before + 10.0)
        assert session.trial_length == pytest.approx(session.trials.baseline_s + session.trials.active_s)
    finally:
        session.stop()


def test_passive_mode_has_no_note_about_the_effort_window():
    """The note that the level runs only in the effort is for active mode; passive mode has no effort."""
    session = _session(trial_direction="increase", n_trials=1)
    try:
        session.set_mode("passive")
        assert session.no_level_note() is None
        session.set_mode("active")
        session.restart_trials()
        assert session.no_level_note() is not None, "active mode, no level yet"
    finally:
        session.stop()


def test_a_stall_whose_frames_arrive_later_loses_no_reading():
    from nfb.heg import HegState
    from nfb.heg.processor import QUALITY_GOOD

    session = _session(n_trials=1)
    try:
        frame = 0
        while not session.trials.results:
            made = frame / 30.0
            phase = made % 2.0
            arrival = made if phase < 1.6 else made - phase + 1.998 + 0.0025 * (phase - 1.6)
            state = HegState(t=arrival, smoothed_hbo=10.0 + 0.3 * math.sin(made), smoothed_hbr=0.0,
                             quality=QUALITY_GOOD, samples_seen=1000, windowed_ready=True)
            session.heg.state = state
            session.heg.readings.append(state)
            if frame % 3 == 0:
                session.observe()
            frame += 1
            assert made < 400.0, "no trial came"
        trial = session.trials.results[0]
        assert trial.valid, trial
        assert trial.invalid_share == pytest.approx(0.0), f"{trial.invalid_share:.0%} of the readings lost"
    finally:
        session.stop()


def test_frames_that_never_arrive_are_still_lost_readings():
    """The control: a stall whose frames never come is a hole, and its readings count as lost."""
    from nfb.heg import HegState
    from nfb.heg.processor import QUALITY_GOOD

    session = _session(n_trials=1)
    try:
        frame = 0
        while not session.trials.results:
            made = frame / 30.0
            frame += 1
            if made % 2.0 >= 1.6:
                continue                                # this frame is lost on the way
            state = HegState(t=made, smoothed_hbo=10.0 + 0.3 * math.sin(made), smoothed_hbr=0.0,
                             quality=QUALITY_GOOD, samples_seen=1000, windowed_ready=True)
            session.heg.state = state
            session.heg.readings.append(state)
            if frame % 3 == 0:
                session.observe()
            assert made < 400.0, "no trial came"
        assert session.trials.results[0].invalid_share > 0.1
    finally:
        session.stop()
