"""The pupil screen: which readings are kept, how a trial is run and scored, and what the room hears.

Driven like the HEG tests: a simulated eye with `realtime=False`, its readings pushed through the
processor by hand, and the session observed at its own rate. No thread, no clock.
"""

from __future__ import annotations

import pytest

from nfb.pupil import PupilProcessor, PupilSample, SimulatedPupil
from nfb.pupil.processor import AFTER_BLINK_S, RESYNC_S

from nfb_console.inputs import PupilChoice
from nfb_console.protocols import BY_CODE
from nfb_console.trials import ACTIVE, NEGATIVE, POSITIVE, TrialRun

from test_screens import StubPage, toolbar
from test_ste import _help_faults


def _session(response: str = "strong", **settings):
    from nfb_console.pupil_view import PupilSession

    session = PupilSession(BY_CODE["PUPIL"], choice=PupilChoice(kind="simulator", response=response))
    session.cue = lambda *_a, **_k: True
    for name, value in settings.items():
        setter = getattr(session, f"set_{name}", None)
        if setter is not None:
            setter(value)
        else:
            setattr(session, name, value)
    session.restart_trials()
    session.paused = False
    return session


def _drive(session, seconds: float, seed: int = 3, response_mm: float | None = None) -> None:
    """Simulated readings at 60 Hz through the processor, the session observing every sixth one — its
    own 10 Hz. The simulated eye is the runner's source, so the session's steer reaches it."""
    from nfb_console.inputs import SIMULATED_RESPONSE_MM

    eye = SimulatedPupil(seed=seed, realtime=False,
                         response_mm=(SIMULATED_RESPONSE_MM[session.choice.response]
                                      if response_mm is None else response_mm))
    runner = session.eye
    runner.source = eye
    start = None
    for i, sample in enumerate(eye.samples()):
        start = sample.t if start is None else start
        if sample.t - start > seconds or session.finished:
            break
        runner.state = runner.processor.push(sample)
        if i % 6 == 0:
            session.observe()


# --- which readings are kept ------------------------------------------------------------------

def test_a_blink_is_rejected_and_the_diameter_holds_through_it():
    proc = PupilProcessor()
    states = [proc.push(PupilSample(i / 60, 4.0, 0.95)) for i in range(30)]
    assert states[-1].valid and states[-1].diameter_mm == pytest.approx(4.0)
    blink = [proc.push(PupilSample(0.5 + i / 60, None, 0.0)) for i in range(12)]
    assert not any(s.valid for s in blink)
    assert all(s.diameter_mm == pytest.approx(4.0) for s in blink), "the last kept value holds"
    after = proc.push(PupilSample(0.5 + 12 / 60, 4.1, 0.95))
    assert not after.valid, "the lid is still moving just after the pupil reappears"
    later = proc.push(PupilSample(0.5 + 12 / 60 + AFTER_BLINK_S + 0.02, 4.1, 0.95))
    assert later.valid


def test_a_low_confidence_an_impossible_size_and_a_jump_are_all_rejected():
    proc = PupilProcessor()
    for i in range(10):
        proc.push(PupilSample(i / 60, 4.0, 0.95))
    assert not proc.push(PupilSample(11 / 60, 3.9, 0.4)).valid, "a lid edge: the detector is unsure"
    assert not proc.push(PupilSample(12 / 60, 12.0, 0.95)).valid, "no pupil is 12 mm"
    assert not proc.push(PupilSample(13 / 60, 5.0, 0.95)).valid, "1 mm in a frame is not a pupil"
    resync = proc.push(PupilSample(13 / 60 + RESYNC_S + 0.05, 5.0, 0.95))
    assert resync.valid, "after a long rejection the reading is kept, or a real change locks out"


def test_the_simulated_eye_moves_toward_the_effort_and_less_for_smaller():
    def mean_after(effort: float) -> float:
        eye = SimulatedPupil(seed=1, realtime=False, blinks=False, response_mm=0.4)
        eye.effort = effort
        values = []
        for i, s in enumerate(eye.samples()):
            if i > 60 * 20:
                break
            if i > 60 * 10:
                values.append(s.diameter_mm)
        return sum(values) / len(values)

    rest, up, down = mean_after(0.0), mean_after(1.0), mean_after(-1.0)
    assert up - rest == pytest.approx(0.4, abs=0.05)
    assert rest - down == pytest.approx(0.4 * SimulatedPupil.DOWN_SHARE, abs=0.05)


# --- the trial engine -------------------------------------------------------------------------

def _directions(run: TrialRun, n: int) -> list[str]:
    return [run._next_assignment()[0] for _ in range(n)]


def test_one_direction_asks_for_that_direction_only_and_alternating_takes_turns():
    assert set(_directions(TrialRun(directions=(POSITIVE,), seed=1), 12)) == {POSITIVE}
    assert set(_directions(TrialRun(directions=(NEGATIVE,), seed=1), 12)) == {NEGATIVE}
    turns = _directions(TrialRun(alternate=True, seed=1), 12)
    assert turns == [POSITIVE, NEGATIVE] * 6


def test_the_cue_phase_keeps_the_answer_to_the_tone_out_of_the_baseline():
    run = TrialRun(cue_s=2.0, baseline_s=5.0, active_s=15.0, metric_hz=10.0, seed=1,
                   directions=(POSITIVE,))
    t = 0.0
    while run.phase != "cued":
        run.feed(t, 4.0)
        t += 0.1
    while run.phase == "cued":
        run.feed(t, 9.0)               # a large answer to the cue tone
        t += 0.1
    while run.phase == "baseline":
        run.feed(t, 4.0)
        t += 0.1
    assert run.baseline_uv == pytest.approx(4.0), "nothing from the cue phase is in the baseline"


# --- the session ------------------------------------------------------------------------------

def test_a_strong_eye_succeeds_in_most_trials_of_both_directions():
    session = _session("strong", direction="alternating", n_trials=6)
    try:
        _drive(session, 6 * 32.0)
        run = session.trials
        assert run.done and session.finished, "the run ends the session"
        assert {r.required for r in run.results} == {POSITIVE, NEGATIVE}
        assert sum(r.success for r in run.results) >= 5
    finally:
        session.stop()


def test_the_trials_panel_shows_the_mean_shift_per_direction():
    from nfb_console.pupil_view import trial_summary
    from nfb_console.trials import Trial

    def trial(required, score, valid=True):
        return Trial(index=0, required=required, transfer=False, score_uv=score, valid=valid,
                     success=valid and (score > 0 if required == POSITIVE else score < 0))

    run = TrialRun()
    assert trial_summary(run) == "no trials yet"
    run.results = [trial(POSITIVE, 0.20), trial(POSITIVE, 0.10),
                   trial(NEGATIVE, -0.08), trial(NEGATIVE, 0.02),
                   trial(POSITIVE, 0.0, valid=False)]
    text = trial_summary(run)
    assert "larger: 2 of 2 · mean +0.15 mm" in text, "the larger mean is the signed average shift"
    assert "smaller: 1 of 2 · mean -0.03 mm" in text, "the smaller mean leans negative"
    assert "1 with no result" in text, "a void trial is counted, not averaged"


def test_the_within_trial_fade_turns_the_circle_off_after_the_set_time():
    session = _session("strong", direction="increase", n_trials=1)
    session.feedback_window_s = 5.0
    on_at, off_at = [], []
    original = session.observe

    def watch():
        original()
        run = session.trials
        if run.phase == ACTIVE and not run.transfer:
            elapsed = run.active_s - run.phase_remaining()
            (on_at if session.feedback_now else off_at).append(elapsed)

    session.observe = watch
    try:
        _drive(session, 40.0)
        assert on_at and max(on_at) <= 5.5, "the circle is on only for the first seconds"
        assert off_at and min(off_at) >= 4.5, "the circle is off after that"
    finally:
        session.stop()


def test_the_stepwise_update_holds_the_circle_between_blocks():
    session = _session("strong", direction="increase", n_trials=1)
    session.update_s = 3.0
    shown = []
    original = session.observe

    def watch():
        original()
        run = session.trials
        if run.phase == ACTIVE and not run.transfer and session.feedback_now:
            shown.append(session.shown_shift)

    session.observe = watch
    try:
        _drive(session, 40.0)
        assert len(shown) > 20, "the effort window has many readings"
        changes = sum(1 for a, b in zip(shown, shown[1:]) if a != b)
        assert changes <= 6, "the circle updates in steps, not every reading"
    finally:
        session.stop()


def test_no_feedback_trials_are_counted_on_their_own():
    from nfb_console.pupil_view import trial_summary
    from nfb_console.trials import Trial

    def trial(required, score, transfer):
        return Trial(index=0, required=required, transfer=transfer, score_uv=score, valid=True,
                     success=(score > 0 if required == POSITIVE else score < 0))

    run = TrialRun()
    run.results = [trial(POSITIVE, 0.10, False), trial(POSITIVE, 0.04, True),
                   trial(NEGATIVE, -0.06, False), trial(NEGATIVE, -0.02, True)]
    text = trial_summary(run)
    assert "feedback · larger: 1 of 1 · mean +0.10 mm" in text
    assert "no feedback · larger: 1 of 1 · mean +0.04 mm" in text
    assert "feedback · smaller: 1 of 1 · mean -0.06 mm" in text
    assert "no feedback · smaller: 1 of 1 · mean -0.02 mm" in text


def test_random_direction_hides_the_aim_until_the_effort_starts():
    import nfb_console.pupil_view as pv
    from nfb_console.trials import ACTIVE, BASELINE, CUED, RESULT

    session = _session("strong", direction="random", n_trials=4)
    pitches: list[float] = []
    real_tone = pv.cues.tone_at
    pv.cues.tone_at = lambda hz: pitches.append(hz) or real_tone(hz)
    known_in: dict[str, set] = {}
    original = session.observe

    def watch():
        original()
        known_in.setdefault(session.trials.phase, set()).add(session.direction_known())

    session.observe = watch
    try:
        _drive(session, 4 * 32.0)
        assert known_in.get(BASELINE) == {False}, "the baseline never shows the aim"
        assert known_in.get(CUED, {False}) == {False}, "the cue phase never shows the aim"
        assert known_in.get(ACTIVE) == {True}, "the effort shows the aim"
        assert {r.required for r in session.trials.results} == {POSITIVE, NEGATIVE}
        assert pv.CUE_PITCH_NEUTRAL in pitches, "the cue tone is the middle tone, not a direction"
    finally:
        pv.cues.tone_at = real_tone
        session.stop()


def test_an_eye_with_no_control_does_not_succeed_by_chance_in_most_trials():
    session = _session("none", direction="increase", n_trials=8)
    session.success_mm = 0.1
    session.restart_trials()
    try:
        _drive(session, 8 * 32.0)
        assert sum(r.success for r in session.trials.results) <= 2
    finally:
        session.stop()


def test_the_level_runs_only_in_the_effort_window_and_rises_toward_the_cue():
    session = _session("strong", direction="decrease", n_trials=3)
    seen_outside, seen_inside = [], []
    original = session.observe

    def watch():
        original()
        run = session.trials
        (seen_inside if run.phase == ACTIVE else seen_outside).append(session.signal)

    session.observe = watch
    try:
        _drive(session, 3 * 32.0)
        assert all(s is None for s in seen_outside), "no level outside the effort window"
        inside = [s for s in seen_inside if s is not None]
        assert inside and sum(inside) / len(inside) > 0.55, "a smaller pupil raises the level here"
    finally:
        session.stop()


def test_a_trial_without_feedback_has_no_level_but_is_scored():
    session = _session("strong", direction="increase", n_trials=3, transfer_fraction=1.0)
    try:
        _drive(session, 3 * 32.0)
        assert session.signal is None
        assert len(session.trials.results) == 3 and all(r.transfer for r in session.trials.results)
    finally:
        session.stop()


def test_passive_mode_follows_the_pupil_with_no_trials():
    session = _session("strong", mode="passive")
    try:
        session.eye.source = None
        _drive(session, 30.0)
        assert session.signal is not None
        assert not session.trials.results
    finally:
        session.stop()


def test_a_pause_releases_the_outputs_and_stops_the_effort():
    session = _session("strong", direction="increase")
    released = []
    session.release_outputs = lambda: released.append(True)
    try:
        _drive(session, 20.0)
        session.toggle_pause()
        assert released, "a pause hands the room back"
        assert session.eye.source.effort == 0.0
    finally:
        session.stop()


def test_the_screen_offers_no_shadow():
    from nfb_console.pupil_view import PupilSession

    assert "shadow" not in PupilSession.offers


def test_an_invented_simulator_is_handed_back_switched_off():
    from nfb_console import pupil_view
    from nfb_console.pupil import PupilRunner

    runner = PupilRunner(PupilChoice(kind="off"))
    session = pupil_view.session_for(BY_CODE["PUPIL"], pupil_choice=runner.choice,
                                     pupil_runner=runner)
    session.stop()
    assert runner.choice.kind == "off"


# --- the screens ------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["passive", "active"])
def test_the_screen_draws_cold_and_with_readings(mode):
    from nfb_console import pupil_view

    session = _session("strong", mode=mode)
    page = StubPage()
    try:
        control = pupil_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        _drive(session, 40.0)
        session.redraw()
        assert control is not None
    finally:
        session.stop()


def test_the_help_dialog_is_simplified_technical_english_in_both_modes():
    from nfb_console import pupil_view

    for mode in ("active", "passive"):
        session = _session(mode=mode)
        try:
            page = StubPage()
            control = pupil_view.view(page, session, toolbar(), lambda: None)
            faults = [f for f in _help_faults(page, control)
                      if mode == "active" or "names no label" not in f]
            assert not faults, "\n".join(faults)
        finally:
            session.stop()


def test_the_device_screen_starts_and_stops_the_simulated_eye():
    import flet as ft

    from conftest import waitfor
    from test_layout import walk

    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.pupil import PupilRunner

    device = Device(Settings())
    choice = PupilChoice(kind="off")
    runner = PupilRunner(choice)

    def press(label: str) -> None:
        """The last choice with this label: the Pupil tab is the last tab, and the others have an
        "off" and a "simulator" too."""
        found = [c for c in walk(control) if isinstance(c, ft.Container)
                 and isinstance(c.content, ft.Text) and c.content.value == label
                 and c.on_click is not None]
        found[-1].on_click(None)

    try:
        control = device_panel.view(StubPage(), device, pupil_choice=choice, pupil_runner=runner)
        press("simulator")
        assert choice.kind == "simulator"
        waitfor(lambda: runner.ready, what="the simulated eye sends readings")
        press("strong")
        assert runner.source.response_mm == pytest.approx(0.35), "changed on the running eye"
        press("off")
        waitfor(lambda: not runner.running, what="the simulated eye stops")
    finally:
        runner.release()
        device.close()


def test_the_calibration_and_camera_overlays_build(monkeypatch):
    """They use Flet widgets directly, so a wrong attribute only shows when the view is built. The USB
    openers are stubbed so the worker threads touch no hardware."""
    from nfb.pupil import tracker
    from nfb_console import pupil_calibrate, pupil_camera
    from nfb_console.inputs import PupilChoice

    def no_usb(*_a, **_k):
        raise tracker.TrackerError("no device in tests")

    class _NoCamera:
        error = None
        def frames(self):
            return iter(())
        def close(self):
            pass

    monkeypatch.setattr(pupil_calibrate.tracker, "UsbLink", no_usb)
    monkeypatch.setattr(pupil_camera, "TrackerCamera", _NoCamera)

    assert pupil_calibrate.view(StubPage(), PupilChoice(kind="tracker"), lambda: None) is not None
    assert pupil_camera.view(StubPage(), lambda: None) is not None


def test_the_camera_view_levels_frame_brightness():
    import numpy as np
    from nfb_console.pupil_camera import level, TARGET_BRIGHTNESS

    dark = np.full((8, 8), 20, dtype=np.uint8)
    bright = np.full((8, 8), 200, dtype=np.uint8)
    assert abs(level(dark).mean() - TARGET_BRIGHTNESS) < 2
    assert abs(level(bright).mean() - TARGET_BRIGHTNESS) < 2, "both end near the same brightness"
    assert level(np.zeros((4, 4), dtype=np.uint8)).mean() == 0, "an all-black frame is left alone"


def test_the_circle_mirrors_the_pupil_not_the_goal():
    """circle_level is direction-independent: a larger pupil is a bigger circle whichever way the trial
    asks, unlike trial_level which is oriented toward the cue."""
    from nfb_console.pupil_view import PupilSession
    from nfb_console.protocols import BY_CODE
    from nfb_console.inputs import PupilChoice

    s = PupilSession(BY_CODE["PUPIL"], choice=PupilChoice(kind="simulator"))
    try:
        s.set_range(0.3)
        assert s.circle_level(+0.3) > 0.9 and s.circle_level(-0.3) < 0.1, "bigger pupil, bigger circle"
        assert s.circle_level(0.0) == pytest.approx(0.5)
        # same pupil shift gives the same circle size regardless of the trial's direction
        s.trials.required = "negative"
        down = s.circle_level(+0.15)
        s.trials.required = "positive"
        assert s.circle_level(+0.15) == down
    finally:
        s.stop()


def test_the_feedback_circle_draws_direction_arrows():
    from nfb_console.pupil_view import circle_shapes

    plain = circle_shapes(280, 0.5, "#2e6bc2")
    shrink = circle_shapes(280, 0.5, "#2e6bc2", arrows="shrink")
    assert len(shrink) == len(plain) + 4, "four arrows added"
    # all arrow points stay inside the canvas
    import flet.canvas as cv
    pts = [e for sh in shrink if isinstance(sh, cv.Path)
           for e in sh.elements if hasattr(e, "x")]
    assert pts and all(0 <= e.x <= 280 and 0 <= e.y <= 280 for e in pts)
