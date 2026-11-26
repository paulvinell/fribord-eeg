"""Where the HEG level comes from, and what happens to the room when the headband misbehaves.

The HRV screen is the template, so these are deliberately the same tests with a different signal in
them: a session opens paused, every path out of running hands the outputs back, a dead source stops the
feedback rather than freezing it, and a screen-invented simulator is handed back switched off. Each of
those was a real bug on the heart side before it was a rule here.

The one shape that is new is **motion**. On the heart a bad stretch is a dropout, and dropping the
outputs is the honest response. Here a jolt is a two-second hold: the processor keeps the optical
channels flat while the skin settles, and the screen has to hold the level rather than release it —
releasing would strobe the room every time somebody shifted in their chair, and pushing the held
constants into a rolling scale would read as sudden perfect steadiness, which a variability metric
rewards.
"""

from __future__ import annotations

import time

import pytest

from nfb.heg import HegState, Processed, QUALITY_GOOD, QUALITY_MOTION, QUALITY_SETTLING, SimulatedMendi


def _session(choice_kind: str = "simulator"):
    from nfb_console.heg_view import HEGSession
    from nfb_console.inputs import MendiChoice
    from nfb_console.protocols import BY_CODE

    return HEGSession(BY_CODE["HEG"], choice=MendiChoice(kind=choice_kind))


def _feed(session, seconds: float = 60.0, seed: int = 5) -> None:
    """Run simulated frames through the session with its clock advancing.

    The clock has to advance or the rolling strategies see every value at one instant, never warm up,
    and report the middle for ever — the trap the HRV version of this helper records. `realtime=False`
    so the frames cost no wall-clock time.

    **Sixty seconds, not the hundred and forty it was.** Thirty is what any of these assertions actually
    need — a rolling window's worth of warm-up — and sixty is the margin; the rest was 4200 frames of
    processing bought for nothing.
    """
    source = SimulatedMendi(seed=seed, realtime=False)
    session._resumed_at = None
    start = None
    for frame in source.frames():
        start = frame.t if start is None else start
        clock = frame.t - start
        if clock > seconds:
            break
        session._accumulated = clock
        runner = session.heg
        runner.state = runner.signal.push(runner.processor.process(frame),
                                          runner.processor.gate_fraction)
        runner.seen_at = time.monotonic()
        session.observe()
    source.close()


def _spy_outputs(session) -> tuple[list[str], list[str]]:
    """Replace the three output mechanisms' entry points with recorders.

    `NFB_NO_OUTPUTS=1` (autouse in conftest) already makes them inert; this says *which* was asked and
    when, which is what every lifecycle rule here is about.
    """
    applied: list[str] = []
    released: list[str] = []
    for name in ("display", "output_volume", "music"):
        obj = getattr(session, name)
        setattr(obj, "apply", (lambda n=name: lambda *a, **kw: applied.append(n))())
        setattr(obj, "release", (lambda n=name: lambda: released.append(n))())
    return applied, released


# --- the level -------------------------------------------------------------------------------

def test_the_metric_and_scaling_become_the_level_the_mechanisms_follow():
    """The whole composition: metric, scaling, level, then the same four mechanisms every screen has."""
    session = _session()
    try:
        session.toggle_pause()
        session.feedback.sound.enabled = True
        session.feedback.shadow.enabled = True
        assert session.signal is None, "nothing before the metric can speak"
        assert session.active_targets == []
        _feed(session)
        assert session.raw_metric is not None
        assert 0.0 <= session.signal <= 1.0
        assert session.strategy.warmup > 0.9
        assert len(session.targets) == 1
        target = session.targets[0]
        assert target.band is None, "it measures no band, so it gets no card and no filter gear"
        assert not target.calibrating, "the scaling already did that job"
        assert target.value == pytest.approx(session.signal)
        outputs = session.feedback.outputs_for(session.level, True)
        assert outputs["sound"] >= session.feedback.sound.low
    finally:
        session.stop()


def test_the_room_follows_a_scale_that_is_still_filling_rather_than_waiting_for_it():
    """This screen used to withhold the level until the scaling had a quarter of its window — thirty
    seconds in which somebody sits there and nothing answers. The HRV screen already went the other way
    and wrote down why; this is that decision applied here.

    A few seconds in, the scale is genuinely rough. It is also the only honest answer to "how am I doing
    compared with myself", and it firms up while being used."""
    session = _session()
    try:
        session.toggle_pause()
        applied, _ = _spy_outputs(session)
        _feed(session, seconds=8.0)
        assert session.strategy.warmup < 0.25, "a fraction of the window, as it was before"
        assert session.signal is not None, "and the level exists anyway"
        assert applied, "the room is following it"
    finally:
        session.stop()


def test_changing_the_metric_rebuilds_the_scaling_and_resets_the_aim():
    """A rolling window holds values in the *old* metric's units — a slope in units per second against a
    history of raw projection levels would sit pinned at an end until the window aged out."""
    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        before = session.strategy
        assert before.warmup > 0.9
        session.set_metric("hbo_variability")
        assert session.strategy is not before
        assert session.strategy.warmup == 0.0, "the new metric starts calibrating from scratch"
        assert session.signal is None
        assert session.direction == "decrease", "low is the good end for steadiness"
        session.set_metric("hbo")
        assert session.direction == "increase", "and the aim goes back with the metric"
    finally:
        session.stop()


def test_the_inversion_is_applied_to_the_scaled_value_and_not_to_the_metric():
    """Inverting first would turn a rolling minimum into a rolling maximum and make the pane's
    description of what the window holds wrong."""
    session = _session()
    try:
        session.toggle_pause()
        session.set_strategy("fixed_range", low=-1000.0, high=1000.0)
        _feed(session, seconds=40.0)
        up = session.signal
        assert up is not None
        window = session.strategy
        session.set_direction("decrease")
        assert session.strategy is window, "the direction is not a rescaling"
        assert session.signal is None, "the level it already reported was under the other aim"
        _feed(session, seconds=40.0, seed=5)
        assert session.inverts
        assert session.signal == pytest.approx(1.0 - up, abs=0.05)
    finally:
        session.stop()


# --- motion: a hold, not a dropout ------------------------------------------------------------

def test_motion_holds_the_level_instead_of_dropping_the_room():
    """The distinguishing rule of this screen.

    A motion cooldown is two seconds long and an ordinary session has many of them. Releasing the
    outputs on each would strobe the room; feeding the held-flat values into the rolling scale would
    read as sudden perfect steadiness, which the variability metrics *reward*. So the level stays put,
    the outputs keep running on it, and nothing enters the strategy.
    """
    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        held_level, held_signal = session.level, session.signal
        pushed = session.strategy.warmup
        assert held_signal is not None

        applied, released = _spy_outputs(session)
        session.heg.state = HegState(t=session.state.t + 0.03, quality=QUALITY_MOTION,
                                     samples_seen=session.state.samples_seen + 1,
                                     smoothed_hbo=1e6, smoothed_hbr=-1e6)
        session.heg.seen_at = time.monotonic()
        session.observe()

        assert session.holding
        assert released == [], "a two-second cooldown is not a dropout"
        assert session.level == held_level and session.signal == held_signal
        assert session.strategy.warmup == pushed, "nothing entered the scale"
    finally:
        session.stop()


def test_the_gate_fraction_is_reported_because_a_held_stretch_looks_calm():
    """Once the channels are held flat a motion-ridden minute and a still one draw the same trace, so
    the share of frames held has to be printed beside the trace — the same argument as the EEG artifact
    rejection figure."""
    from nfb_console import heg_panel

    session = _session()
    try:
        _feed(session, seconds=20.0)
        session.heg.processor.gated = 30
        session.heg.processor.seen = 100
        panel = heg_panel.HEGPanel()
        panel.push(session.state)
        panel.refresh(session)
        assert "30% held for motion" in panel.trace_caption.value
        words = {heg_panel.quality_word(q) for q in (QUALITY_GOOD, QUALITY_SETTLING, QUALITY_MOTION)}
        assert len(words) == 3, "good, settling and motion each have their own word"
    finally:
        session.stop()


# --- the lifecycle every screen has to get right ------------------------------------------------

def test_the_toolbar_lifecycle_exists_and_pausing_hands_the_room_back():
    """The device chip on the shared toolbar calls suspend()/suspended/rearm() on whatever session is
    up; a session without them is an AttributeError inside a Flet event handler — a click that silently
    does nothing. And a pause has to *release*, not merely stop updating: skipping observe() leaves the
    display dimmed and the music singing."""
    session = _session()
    try:
        _, released = _spy_outputs(session)

        assert session.paused, "a session opens paused"
        session.toggle_pause()
        assert not session.paused and released == [], "resuming releases nothing"
        session.toggle_pause()
        assert session.paused
        assert released == ["display", "output_volume", "music"], "pausing hands all three back"

        released.clear()
        session.suspend()                    # already paused: still releases, never raises
        assert released == ["display", "output_volume", "music"]
        assert session.suspended
        session.rearm()                      # and coming back is a no-op, not a crash
    finally:
        session.stop()


def test_a_finished_plan_releases_the_outputs_too():
    session = _session()
    try:
        _, released = _spy_outputs(session)
        session.plan.length_seconds = 0.01
        session.toggle_pause()               # start the clock
        # Rewinding the resume stamp *is* the passage of time here. Sleeping for it tests the sleep.
        session._resumed_at -= 1.0
        session.tick()
        assert session.finished and session.paused
        assert released == ["display", "output_volume", "music"], \
            "a timed session that ends must not leave the room held"
    finally:
        session.stop()


def test_a_dead_or_silent_headband_stops_the_feedback_and_says_so():
    """Two flavours, both of which ran the room on a frozen snapshot on the heart side: a source that
    raises sets an error but freezes `state` with no gate downstream, and a source whose generator
    simply *ends* — which is how the BLE reader reports its worker dying — set no error at all."""
    session = _session()
    try:
        runner = session.heg

        class Ends:
            error = None

            def frames(self):
                source = SimulatedMendi(seed=1, realtime=False)
                for i, frame in enumerate(source.frames()):
                    if i >= 3:
                        return              # the worker died; the generator just stops
                    yield frame

            def close(self):
                pass

            def describe(self):
                return "test source"

        runner.source = Ends()
        runner._stop.clear()
        runner._run(runner._generation)      # the generator ends on its own
        assert runner.error, "an end nobody asked for is a fault, not a silence"

        applied, _ = _spy_outputs(session)
        session.toggle_pause()
        session.observe()
        assert session.signal is None and applied == [], \
            "an errored source must not keep driving the outputs"
        assert session.dead_channels() == ["the headband"]
    finally:
        session.stop()


def test_a_silent_headband_goes_stale_much_sooner_than_a_silent_strap():
    """A silence shorter than the one that makes a heart strap stale already stops the feedback: at 30 Hz
    it is hundreds of missed frames, where the heart's limit has to allow for a slow pulse."""
    from nfb_console.heart import HeartRunner

    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        assert session.signal is not None
        applied, released = _spy_outputs(session)

        class Alive:
            def is_alive(self):
                return True

        session.heg._thread = Alive()
        session.heg.seen_at = time.monotonic() - (HeartRunner.STALE_S - 1.0)
        assert session.heg.stale
        session.observe()
        assert session.signal is None and applied == []
        assert released == ["display", "output_volume", "music"]
    finally:
        session.heg._thread = None
        session.stop()


def test_the_invented_simulator_is_handed_back_switched_off():
    """The simulator default is this screen's, not the app's: the runner is owned by the app, and one
    visit to HEG must not leave the whole application reading invented frames. A choice the operator
    actually made is kept."""
    from nfb_console.heg_view import HEGSession
    from nfb_console.protocols import BY_CODE

    explicit = _session()
    try:
        assert not explicit._invented, "an explicit choice is the operator's and is kept"
    finally:
        explicit.stop()

    invented = HEGSession(BY_CODE["HEG"], choice=None)
    try:
        assert invented._invented and invented.choice.kind == "simulator"
        invented.heg.state = object()        # pretend the simulator ran
        invented.heg.error = "leftover"
    finally:
        invented.stop()
    assert invented.heg.choice.kind == "off", "the shared runner goes back the way it was found"
    assert invented.heg.state is None and invented.heg.error is None
    assert invented.heg.seen_at is None


def test_a_paired_headband_replaces_the_heg_screens_invented_one(monkeypatch):
    """The same defect and the same fix on the HEG screen, whose simulator is invented for the same
    reason."""
    from nfb_console import heg_view
    from nfb_console.inputs import MendiChoice
    from nfb_console.protocols import BY_CODE

    app_choice = MendiChoice(kind="off")
    session = heg_view.session_for(BY_CODE["HEG"], mendi_choice=app_choice)
    assert session.choice.kind == "simulator" and session._invented

    app_choice.kind, app_choice.address = "headband", "AA:BB"
    session.adopt_offered()
    assert session.choice is app_choice and not session._invented


def test_a_fresh_source_starts_a_fresh_record():
    """attach() resets both processors. Without it the low-pass filters hold the previous source's DC
    level, the eleven-second settling clock arrives pre-satisfied, and the motion gate's window carries
    another headband's accelerometer — the same class of bug as the heart runner inheriting beats."""
    from nfb_console.inputs import MendiChoice
    from nfb_console.mendi import MendiRunner

    runner = MendiRunner(MendiChoice(kind="simulator"))
    runner.signal.push(Processed(t=1000.0, hbo=1.0, hbr=1.0, raw_ir=1.0, quality=QUALITY_GOOD,
                                 temp_c=36.0))
    runner.signal.push(Processed(t=1001.0, hbo=2.0, hbr=2.0, raw_ir=2.0, quality=QUALITY_GOOD,
                                 temp_c=36.0))
    runner.state = object()
    assert runner.signal._hbo, "old samples are in the buffers"

    # The reset is `attach`'s own work and happens before it starts the reader, so it is checked the
    # moment it returns. Sleeping for a fifth of a second afterwards checked the same thing through a
    # thread, which is the scheduler's business rather than this test's.
    assert runner.attach()
    try:
        # Nothing from before the attach survives — true the instant it returns, whatever the reader has
        # added since. The old version slept for a fifth of a second so there would be something to
        # compare against, which put the scheduler inside the assertion.
        assert all(ts > 1e6 for ts, _ in runner.signal._hbo), "old samples survived the attach"
        assert runner.error is None
    finally:
        runner.release()


# --- the screen draws -----------------------------------------------------------------------

def test_the_heg_screen_draws_before_and_after_frames():
    """The one thing the object-level tests cannot see: whether a frame can be produced at all. An
    extraction that leaves a stale name in the draw path passes every test above."""
    import sys

    sys.path.insert(0, "tests")
    from harness import StubPage, _texts, toolbar

    from nfb_console import heg_view

    page = StubPage()
    session = _session()
    try:
        control = heg_view.view(page, session, toolbar(), lambda: None)
        session.redraw()                     # no frames at all
        assert session.error is None

        session.toggle_pause()
        _feed(session)
        session.redraw()
        assert session.error is None, session.error
        captions = " | ".join(c.value for c in _texts(control) if c.value)
        assert "HbO" in captions
        assert "held for motion" in captions or "s across" in captions
    finally:
        session.stop()


def test_the_screen_has_one_place_to_configure_the_feedback():
    """The structural check the HRV screen's four-tab version would have failed: two tabs configuring
    the feedback means two writers to the display brightness and two music streams able to sound at once."""
    import sys

    sys.path.insert(0, "tests")
    from harness import StubPage, _texts, toolbar

    from nfb_console import heg_view

    page = StubPage()
    session = _session()
    try:
        control = heg_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        panes = control.controls[-1].controls
        assert len(panes) == 3, "Train, Feedback, Session"
        for i, pane in enumerate(panes):
            pane.visible = (i == 1)
        texts = [x.value for x in _texts(control) if x.value]
        shown = " ".join(texts)
        assert texts.count("OUTPUT") == 1, "one panel, on one tab — the shared one"
        assert "SIGNAL SOURCE" in shown
        assert "feed back" in shown and "scaled by" in shown and "aim to" in shown
        for mechanism in ("sound", "shadow", "click", "music"):
            assert mechanism in shown.lower(), f"{mechanism} is configurable here"
        # and none of the rows that belong to a level built by scoring bands: absent rather than
        # emptied, which is what `conditions_signal` and `scores_bands` are for
        for gone in ("ARTIFACTS", "rejection", "band averaging", "3 · level from", "4 · condition"):
            assert gone not in shown, f"{gone!r} has nothing to act on here"
    finally:
        session.stop()


def test_the_protocol_is_offered_now_that_it_has_a_source():
    from nfb_console.app import Console
    from nfb_console.protocols import BY_CODE

    from nfb_console import heg_view

    assert BY_CODE["HEG"].available, "the signal it needs can be computed"
    assert Console.screen_for(BY_CODE["HEG"]) is heg_view


def test_the_motion_hold_is_a_row_on_the_screen_and_the_chip_carries_the_charge():
    """Both from the same sitting: the hold was a constant nobody could see or change, and the battery
    was under the trace and nowhere else — the number you want before starting twenty minutes was the
    one that was easiest to scroll past."""
    import sys

    sys.path.insert(0, "tests")
    from harness import StubPage, _texts, toolbar

    from nfb_console import heg_view

    page = StubPage()
    session = _session()
    try:
        control = heg_view.view(page, session, toolbar(), lambda: None)
        session.toggle_pause()
        _feed(session, seconds=20.0)
        session.redraw()
        panes = control.controls[-1].controls
        for i, pane in enumerate(panes):
            pane.visible = (i == 1)                  # the Feedback tab, where the signal is configured
        shown = " ".join(x.value for x in _texts(control) if x.value)
        assert "when you move" in shown
        assert "keep training" in shown, "the hold is off-switchable from here"
        assert "held so far" in shown, "and it says what it is costing"

        session.motion_hold = False
        assert session.heg.processor.motion_gate is False, "straight through to the processor"

        session.heg.source = type("S", (), {"battery_pct": 41})()
        assert "battery 41%" in heg_view._instrument(session)[1]
    finally:
        session.stop()


def test_a_fresh_session_keeps_the_metric_the_scaling_and_the_aim():
    """A stop starts a fresh session with what the operator chose: the metric, its scaling and the aim."""
    from nfb_console.heg_view import HEGSession
    from nfb_console.protocols import BY_CODE

    old = _session()
    new = None
    try:
        old.set_metric("hbo_trend")
        old.set_strategy("fixed_range", low=-5.0, high=5.0)
        old.set_direction("decrease")
        new = HEGSession(BY_CODE["HEG"], choice=old.choice)
        new.adopt(old)
        assert new.metric == "hbo_trend" and new.direction == "decrease"
        assert new.strategy_id == "fixed_range" and new.strategy_params == old.strategy_params
    finally:
        old.stop()
        if new is not None:
            new.stop()


def test_frames_without_a_metric_yet_name_the_metric_that_waits():
    """While frames arrive and the metric cannot speak yet, the note under the level says which one waits."""
    session = _session()
    try:
        session.toggle_pause()
        _feed(session, seconds=1.0)
        assert session.raw_metric is None, "the premise: too few frames for the metric"
        note = session.level_note()
        assert note is not None and session.metric_spec.label in note[0]
    finally:
        session.stop()


def test_a_level_with_no_smoothing_follows_the_signal_exactly():
    """Smoothing set to zero is a level equal to the scaled signal, not a fault."""
    session = _session()
    try:
        session.toggle_pause()
        session.feedback.smoothing_seconds = 0.0
        _feed(session, seconds=20.0)
        assert session.signal is not None and session.level == session.signal
    finally:
        session.stop()


def test_play_does_not_resume_a_trial_run_that_is_over():
    """When every trial of an active run has a result, play keeps the session paused."""
    session = _session()
    try:
        session.set_mode("active")
        run = session.trials
        run.results = [object()] * run.n_trials
        assert run.done, "the premise: the run is over"
        session.toggle_pause()
        assert session.paused
    finally:
        session.stop()
