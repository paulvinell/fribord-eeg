"""HEG's active mode: Pupil's cued trials on the headband's signal, with the shift judged in standard
deviations of the minute before each trial. Readings are pushed in as states; no device, no clock."""

from __future__ import annotations

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


def test_heg_opens_in_passive_mode_as_it_always_has():
    from nfb_console.heg_view import HEGSession
    from nfb_console.inputs import MendiChoice
    from nfb_console.protocols import BY_CODE

    assert HEGSession(BY_CODE["HEG"], choice=MendiChoice(kind="simulator")).mode == "passive"


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


def test_a_trial_without_a_minute_of_signal_before_it_has_no_result():
    """The spread needs signal to be measured in. Cued with too little of it, the trial cannot be judged,
    and says so, rather than dividing by a spread made of a few readings."""
    session = _session(trial_direction="increase", n_trials=1)
    result, _ = _first_result(session, 1000.0, rise=1.0)
    assert session.trial_spread is None
    assert not result.valid


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
    from test_screens import StubPage, _texts, toolbar

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
