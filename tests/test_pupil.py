"""The pupil screen: which readings are kept, how a trial is run and scored, and what the room hears.

Driven like the HEG tests: a simulated eye with `realtime=False`, its readings pushed through the
processor by hand, and the session observed at its own rate. No thread, no clock.
"""

from __future__ import annotations

import pytest

from nfb.pupil import PupilProcessor, PupilSample, SimulatedPupil
from nfb.pupil.processor import AFTER_BLINK_S, RESYNC_S

from nfb_console.inputs import HeartChoice, PupilChoice
from nfb_console.protocols import BY_CODE
from nfb_console.trials import ACTIVE, BASELINE, ITI, LEADIN_S, NEGATIVE, POSITIVE, RESULT, TrialRun
from nfb_console.pupil_view import ITI_S, OBSERVE_HZ, RESULT_S
from nfb_console.trial_training import CUE_PITCH_NEUTRAL

from test_screens import StubPage, toolbar
from test_ste import _help_faults, _texts


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
        runner.push(sample)
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
    session = _session("strong", trial_direction="alternating", n_trials=6)
    try:
        _drive(session, 6 * 32.0)
        run = session.trials
        assert run.done and session.finished, "the run ends the session"
        assert {r.required for r in run.results} == {POSITIVE, NEGATIVE}
        assert sum(r.success for r in run.results) >= 5
    finally:
        session.stop()


def test_the_trials_panel_shows_the_mean_shift_per_direction():
    from nfb_console.trial_training import trial_summary
    from nfb_console.trials import Trial

    def trial(required, score, valid=True):
        return Trial(index=0, required=required, transfer=False, score_uv=score, valid=valid,
                     success=valid and (score > 0 if required == POSITIVE else score < 0))

    run = TrialRun()
    assert trial_summary(run, {POSITIVE: "larger", NEGATIVE: "smaller"}, "mm") == "no trials yet"
    run.results = [trial(POSITIVE, 0.20), trial(POSITIVE, 0.10),
                   trial(NEGATIVE, -0.08), trial(NEGATIVE, 0.02),
                   trial(POSITIVE, 0.0, valid=False)]
    text = trial_summary(run, {POSITIVE: "larger", NEGATIVE: "smaller"}, "mm")
    assert "larger: 2 of 2 · mean +0.15 mm" in text, "the larger mean is the signed average shift"
    assert "smaller: 1 of 2 · mean -0.03 mm" in text, "the smaller mean leans negative"
    assert "1 with no result" in text, "a void trial is counted, not averaged"


def test_the_within_trial_fade_turns_the_circle_off_after_the_set_time():
    session = _session("strong", trial_direction="increase", n_trials=1)
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
    session = _session("strong", trial_direction="increase", n_trials=1)
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


def test_the_sound_steps_with_the_circle_and_does_not_glide_between_blocks():
    """The bug: the circle stepped, but the smoother ran on the sound's level too, so the sound glided
    across the steps the circle jumped. Stepwise feedback is already quantised; the continuous smoother
    must not run on top of it, whatever the smoothing is set to."""
    session = _session("strong", trial_direction="increase", n_trials=1)
    session.update_s = 3.0
    session.feedback.smoothing_seconds = 0.5      # on: the smoother is what dissolves the steps
    levels = []
    original = session.observe

    def watch():
        original()
        run = session.trials
        if run.phase == ACTIVE and not run.transfer and session.feedback_now:
            levels.append(round(session.level, 4))

    session.observe = watch
    try:
        _drive(session, 40.0)
        assert len(levels) > 20, "the effort window has many readings"
        changes = sum(1 for a, b in zip(levels, levels[1:]) if a != b)
        assert changes <= 6, "the sound's level steps with the blocks, it does not glide between them"
    finally:
        session.stop()


def test_live_feedback_still_smooths_the_level():
    """The other side of it: with no stepping, the smoother still runs, so the level moves every reading."""
    session = _session("strong", trial_direction="increase", n_trials=1)
    session.update_s = 0.0
    session.feedback.smoothing_seconds = 0.5
    levels = []
    original = session.observe

    def watch():
        original()
        run = session.trials
        if run.phase == ACTIVE and not run.transfer and session.feedback_now:
            levels.append(round(session.level, 4))

    session.observe = watch
    try:
        _drive(session, 40.0)
        changes = sum(1 for a, b in zip(levels, levels[1:]) if a != b)
        assert changes > 10, "live feedback moves continuously, not in a handful of steps"
    finally:
        session.stop()


def test_no_feedback_trials_are_counted_on_their_own():
    from nfb_console.trial_training import trial_summary
    from nfb_console.trials import Trial

    def trial(required, score, transfer):
        return Trial(index=0, required=required, transfer=transfer, score_uv=score, valid=True,
                     success=(score > 0 if required == POSITIVE else score < 0))

    run = TrialRun()
    run.results = [trial(POSITIVE, 0.10, False), trial(POSITIVE, 0.04, True),
                   trial(NEGATIVE, -0.06, False), trial(NEGATIVE, -0.02, True)]
    text = trial_summary(run, {POSITIVE: "larger", NEGATIVE: "smaller"}, "mm")
    assert "feedback · larger: 1 of 1 · mean +0.10 mm" in text
    assert "no feedback · larger: 1 of 1 · mean +0.04 mm" in text
    assert "feedback · smaller: 1 of 1 · mean -0.06 mm" in text
    assert "no feedback · smaller: 1 of 1 · mean -0.02 mm" in text


def test_random_direction_hides_the_aim_until_the_effort_starts():
    import nfb_console.pupil_view as pv
    from nfb_console.trials import ACTIVE, BASELINE, CUED, RESULT

    session = _session("strong", trial_direction="random", n_trials=4)
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
        assert CUE_PITCH_NEUTRAL in pitches, "the cue tone is the middle tone, not a direction"
    finally:
        pv.cues.tone_at = real_tone
        session.stop()


def test_an_eye_with_no_control_does_not_succeed_by_chance_in_most_trials():
    session = _session("none", trial_direction="increase", n_trials=8)
    session.success_at = 0.1
    session.restart_trials()
    try:
        _drive(session, 8 * 32.0)
        assert sum(r.success for r in session.trials.results) <= 2
    finally:
        session.stop()


def test_the_level_runs_only_in_the_effort_window_and_rises_toward_the_cue():
    session = _session("strong", trial_direction="decrease", n_trials=3)
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
    session = _session("strong", trial_direction="increase", n_trials=3, transfer_fraction=1.0)
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
    session = _session("strong", trial_direction="increase")
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


def _steady(session, start: float, seconds: float, observe_every: int | None = 6) -> float:
    """Steady 60 Hz readings of a 4 mm pupil into the session's runner from `start`, observed every
    `observe_every` readings, or not at all. Returns the time after the last reading."""
    t = start
    for i in range(int(round(seconds * 60))):
        session.eye.push(PupilSample(t, 4.0, 0.95))
        if observe_every and i % observe_every == 0:
            session.observe()
        t += 1 / 60
    return t


def _to_result(session) -> float:
    t = 1000.0
    while session.trials.phase != RESULT:
        t = _steady(session, t, 0.1)
    return t


def test_a_slow_frame_does_not_cut_the_result_short():
    """Stopping the sound at the end of an effort once held the screen for 0.85 s. The session took only
    the newest reading per frame, so the trial engine saw a gap and dropped the result after a split
    second. It takes every reading the frame missed, so the result runs its full time."""
    session = _session()
    t = _to_result(session)
    began = session._last_t
    t = _steady(session, t, 1.2, observe_every=None)        # the screen stands still for 1.2 s
    session.observe()
    assert session.trials.phase == RESULT, "the readings the frame missed are taken, not a gap"
    while session.trials.phase == RESULT:
        t = _steady(session, t, 0.1)
    assert session._last_t - began == pytest.approx(RESULT_S, abs=0.11)


def test_a_real_gap_in_the_readings_still_drops_the_trial():
    session = _session()
    t = _to_result(session)
    _steady(session, t + 1.5, 0.2)                           # the eye sends nothing for 1.5 s
    assert session.trials.phase == ITI
    assert session.trials.phase_remaining() <= LEADIN_S < ITI_S[0], \
        "the engine starts again from its lead-in, not from the rest that follows a result"


def test_readings_that_arrive_while_paused_are_not_taken():
    session = _session()
    t = _steady(session, 1000.0, 1.0)
    session.toggle_pause()
    t = _steady(session, t, 0.5, observe_every=None)
    session.toggle_pause()
    resumed = t
    _steady(session, t, 0.5)
    assert all(not (resumed - 0.5 <= at < resumed) for at, _d, _k in session.history)


def test_the_session_takes_ten_readings_a_second_of_eye_time_however_often_it_draws():
    session = _session()
    _steady(session, 1000.0, 10.0, observe_every=45)        # a frame every 0.75 s
    times = [at for at, _d, _k in session.history]
    assert len(times) == pytest.approx(10 * OBSERVE_HZ, abs=2)
    assert max(b - a for a, b in zip(times, times[1:])) < 0.12


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
        s.set_full_at(0.3)
        assert s.circle_level(+0.3) > 0.75 and s.circle_level(-0.3) < 0.25, "bigger pupil, bigger circle"
        assert s.circle_level(0.0) == pytest.approx(0.5)
        # same pupil shift gives the same circle size regardless of the trial's direction
        s.trials.required = "negative"
        down = s.circle_level(+0.15)
        s.trials.required = "positive"
        assert s.circle_level(+0.15) == down
    finally:
        s.stop()


def test_the_circle_never_sticks_at_its_limit():
    """The picture is in proportion up to `full_at`, then compresses toward its limit without reaching it,
    so a shift three times `full_at` still draws further out than one and a half times it. The level the
    outputs read keeps its clip."""
    from nfb_console.trial_training import FULL_AT_PART, picture_position
    from nfb_console.pupil_view import PupilSession
    from nfb_console.protocols import BY_CODE
    from nfb_console.inputs import PupilChoice

    assert picture_position(10.0, 10.0) == pytest.approx(FULL_AT_PART), "full at lands at a stated part"
    assert picture_position(5.0, 10.0) == pytest.approx(FULL_AT_PART / 2), "in proportion below it"
    shifts = [0.5 * i for i in range(-200, 201)]
    positions = [picture_position(x, 10.0) for x in shifts]
    assert all(b > a for a, b in zip(positions, positions[1:])), "monotonic"
    assert all(-1.0 < p < 1.0 for p in positions), "never at the limit"
    assert picture_position(30.0, 10.0) - picture_position(15.0, 10.0) > 0.1, "30 is visibly past 15"
    knee = [picture_position(10.0 + d, 10.0) - picture_position(10.0, 10.0) for d in (-1e-4, 1e-4)]
    assert knee[1] == pytest.approx(-knee[0], rel=1e-2), "no kink at the knee"

    s = PupilSession(BY_CODE["PUPIL"], choice=PupilChoice(kind="simulator"))
    try:
        s.set_full_at(0.3)
        s.trials.required = "positive"
        assert 0.0 < s.circle_level(-0.9) < s.circle_level(-0.45) < 0.5
        assert s.trial_level(-0.9) == s.trial_level(-0.45) == 0.0, "the outputs still clip"
    finally:
        s.stop()


def test_the_feedback_circle_draws_direction_arrows():
    from nfb_console.trial_training import circle_shapes

    plain = circle_shapes(280, 0.5, "#2e6bc2")
    shrink = circle_shapes(280, 0.5, "#2e6bc2", arrows="shrink")
    assert len(shrink) == len(plain) + 4, "four arrows added"
    # all arrow points stay inside the canvas
    import flet.canvas as cv
    pts = [e for sh in shrink if isinstance(sh, cv.Path)
           for e in sh.elements if hasattr(e, "x")]
    assert pts and all(0 <= e.x <= 280 and 0 <= e.y <= 280 for e in pts)


def test_a_stale_pupil_reader_does_not_silence_its_replacement():
    """An old handshake can outlive its 2 s join and fail after a healthy reader has taken over. Its error
    must not land on the shared runner and silence feedback. Each reader carries the generation it began in."""
    from nfb_console.pupil import PupilRunner

    runner = PupilRunner()
    stale_gen = 1
    runner._generation = 2          # a newer reader has since started

    class Boom:
        def samples(self):
            raise RuntimeError("handshake timed out")

        def close(self):
            pass

    runner.source = Boom()
    runner._stop.clear()
    runner._run(stale_gen)          # the stale reader finally fails, after the generation moved on
    assert runner.error is None, "a stale reader's failure does not overwrite the live runner's state"


def test_the_camera_shows_lit_frames_and_holds_through_the_strobe_dark_ones():
    """The tracker strobes its lamp, so frames alternate lit and dark; a dark frame carries no picture, only
    sensor bias the leveller amplifies into rings. The view shows the lit frames and holds through the dark."""
    from nfb_console.pupil_camera import MAX_HELD, _LitSelector

    sel = _LitSelector()
    shown = [sel(m) for m in (100, 15, 100, 15, 100, 15)]     # lamp on / off / on / off ...
    assert shown == [True, False, True, False, True, False], "lit frames show, dark ones are held"

    sel = _LitSelector()
    sel(100)                                                  # a lit frame sets the peak
    run = [sel(10) for _ in range(MAX_HELD + 2)]              # then a long dark run
    assert run[MAX_HELD - 1] is False and run[MAX_HELD] is True, \
        "a genuinely dim scene is not frozen out — the next frame shows after MAX_HELD dark ones"


# --- the heart beside the pupil trials --------------------------------------------------------

def _feed_trial(h, trial, direction, base_bpm, effort_bpm, base_n=5, effort_n=5, result_beat=True):
    """One trial through the heart aggregator: a baseline window, an effort window, then the result."""
    from nfb_console.trial_training import HeartTrials
    for _ in range(base_n):
        h.feed(BASELINE, trial, direction, base_bpm)
    for _ in range(effort_n):
        h.feed(ACTIVE, trial, direction, effort_bpm)
    h.feed(RESULT, trial, direction, effort_bpm if result_beat else None)


def test_heart_stats_are_the_change_from_baseline_split_by_direction():
    from nfb_console.trial_training import HeartTrials
    h = HeartTrials({POSITIVE: "larger", NEGATIVE: "smaller"})
    _feed_trial(h, 1, POSITIVE, 60.0, 66.0)        # larger: +6 bpm
    _feed_trial(h, 2, POSITIVE, 62.0, 66.0)        # larger: +4 bpm  -> mean +5
    _feed_trial(h, 3, NEGATIVE, 70.0, 66.0)        # smaller: -4 bpm
    summary = {word: (change, n) for word, change, n in h.direction_summary()}
    assert summary["larger"][0] == pytest.approx(5.0) and summary["larger"][1] == 2
    assert summary["smaller"][0] == pytest.approx(-4.0) and summary["smaller"][1] == 1


def test_a_trial_is_scored_even_with_no_beat_at_the_result_instant():
    from nfb_console.trial_training import HeartTrials
    h = HeartTrials({POSITIVE: "larger", NEGATIVE: "smaller"})
    _feed_trial(h, 1, POSITIVE, 60.0, 66.0, result_beat=False)
    assert h.direction_summary()[0][1] == pytest.approx(6.0), "the window is scored when it ends"


def test_a_thin_heart_window_has_no_result():
    from nfb_console.trial_training import HeartTrials, HEART_MIN_READINGS
    h = HeartTrials({POSITIVE: "larger", NEGATIVE: "smaller"})
    _feed_trial(h, 1, POSITIVE, 60.0, 66.0, base_n=HEART_MIN_READINGS - 1)
    assert h.direction_summary() == [], "too few baseline readings means no heart result for the trial"


def test_the_session_heart_view_is_overall_and_effort_against_rest():
    from nfb_console.trial_training import HeartTrials
    h = HeartTrials({POSITIVE: "larger", NEGATIVE: "smaller"})
    for _ in range(5):
        h.feed(BASELINE, 1, POSITIVE, 60.0)
    for _ in range(5):
        h.feed(ACTIVE, 1, POSITIVE, 70.0)
    for _ in range(5):
        h.feed(ITI, 1, POSITIVE, 62.0)
    mean, lo, hi = h.overall()
    assert lo == 60.0 and hi == 70.0 and mean == pytest.approx(64.0)
    effort, rest = h.effort_vs_rest()
    assert effort == pytest.approx(70.0) and rest == pytest.approx(62.0)


def test_a_missing_reading_does_not_enter_the_stats():
    from nfb_console.trial_training import HeartTrials
    h = HeartTrials({POSITIVE: "larger", NEGATIVE: "smaller"})
    for _ in range(5):
        h.feed(ACTIVE, 1, POSITIVE, 70.0)
    h.feed(ACTIVE, 1, POSITIVE, None)              # a dropped beat: skipped, not counted as zero
    mean, lo, hi = h.overall()
    assert mean == 70.0 and lo == 70.0 and hi == 70.0


class _PhaseHeart:
    """A stand-in heart runner whose rate rises during the effort window, so a direction effect appears."""

    def __init__(self, session, base=70.0, lift=6.0):
        self.session = session
        self.base = base
        self.lift = lift
        self.stale = False
        self.error = None

    @property
    def state(self):
        import types
        bpm = self.base + (self.lift if self.session.trials.phase == ACTIVE else 0.0)
        return types.SimpleNamespace(hr_bpm=bpm)

    def describe(self):
        return "simulated strap"


def test_driving_with_a_heart_source_fills_the_stats_from_the_effort():
    session = _session("strong", trial_direction="increase", n_trials=1)
    session.heart_choice = HeartChoice(kind="simulator")
    session.heart = _PhaseHeart(session)
    session.heart_trials.reset()        # the heart runner was set after the session was built
    assert session.heart_on
    _drive(session, 40.0)
    assert session.heart_trials.has_data
    summary = session.heart_trials.direction_summary()
    assert summary and summary[0][1] == pytest.approx(6.0, abs=0.5), "effort runs 6 bpm above baseline"
    effort, rest = session.heart_trials.effort_vs_rest()
    assert effort > rest, "the effort window runs hotter than the rest between trials"


def _labels_of(session):
    page = StubPage()
    control = pupil_view_view(page, session)
    return " | ".join(_texts(control)).upper()


def pupil_view_view(page, session):
    from nfb_console import pupil_view
    return pupil_view.view(page, session, toolbar(), lambda: None)


def test_the_heart_panel_appears_only_when_a_source_is_chosen():
    from nfb_console import pupil_view
    bare = pupil_view.session_for(BY_CODE["PUPIL"], channels=0)
    assert "HEART" not in _labels_of(bare), "no heart panel without a chosen source"

    hc = HeartChoice(kind="simulator")
    from nfb_console import heart as heart_mod
    with_heart = pupil_view.session_for(BY_CODE["PUPIL"], channels=0,
                                        heart_choice=hc, heart_runner=heart_mod.HeartRunner(hc))
    labels = _labels_of(with_heart)
    assert "HEART STATS" in labels and "HEART" in labels, "the heart panels show with a chosen source"


def test_the_pupil_baseline_keeps_its_picture(monkeypatch):
    """The distinct phases the slow-potential screen draws are off here: a pupil answers light, so the
    baseline keeps the same ground, the same canvas and the same shapes, the ring and the arrows."""
    import flet.canvas as cv
    from nfb_console.inputs import PupilChoice
    from nfb_console.protocols import BY_CODE
    from nfb_console.pupil_view import CIRCLE_SIDE, PupilSession
    from nfb_console.trial_training import TrialView, circle_shapes
    from nfb_console.trials import BASELINE, POSITIVE

    s = PupilSession(BY_CODE["PUPIL"], choice=PupilChoice(kind="simulator"))
    try:
        s.set_mode("active")
        s.set_trial_direction("alternating")
        monkeypatch.setattr(PupilSession, "state", property(lambda _self: object()))
        run = s.trials
        run._t, run._phase_until, run.phase, run.required = 0.0, 1.0, BASELINE, POSITIVE
        ui = TrialView(s, CIRCLE_SIDE, heart=False)
        ui.fixation_panel()
        ui.draw_fixation("")
        assert ui.ground is None and ui.circle.width == CIRCLE_SIDE

        def described(shapes):
            return [(type(x).__name__, getattr(x, "radius", None), x.paint.color, x.paint.stroke_width,
                     x.paint.stroke_dash_pattern) for x in shapes]

        assert described(ui.circle.shapes) == described(
            circle_shapes(CIRCLE_SIDE, None, "unused", arrows="expand")), "point, dashed ring, four arrows"
        assert len([x for x in ui.circle.shapes if isinstance(x, cv.Path)]) == 4
    finally:
        s.stop()
