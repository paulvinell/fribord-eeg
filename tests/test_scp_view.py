"""The SCP screen: what it feeds back, when it feeds back nothing, and what it makes a noise about.

The metric and the trial engine are checked in `test_scp.py` against inputs whose answer is known. What is
left — and it is the whole of this file — is the *session*: that a level exists during a trial's feedback
window and nowhere else, that it is oriented toward the direction the trial cued, that a transfer trial
produces none, that the outputs are handed back between trials rather than held at their last value, and
that the two sounds this protocol has fire exactly when they should.

Signals are fed as `Block`s straight through `consume`, as `test_wave_view.py` does, because these need a
*known* waveform: the bench recording is a constant at the rail and a trial engine driven by it would run
forty trials of nothing.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from nfb import ReplaySource
from nfb.stream import Block, Gap
from nfb_console import scp_view
from nfb_console.protocols import BY_CODE
from nfb_console.trials import (ACTIVE, BASELINE, ITI, NEGATIVE, POSITIVE, REJECTED, RESULT,
                                TrialRun)

from conftest import BENCH_OK, counts, wandering
from harness import _block

RATE = 250.0
COMPLIANT_UV = 8.0
"""What the simulated trainee produces during a feedback window, in the cued direction."""


def _session(channels: int = 2, source=None, **kwargs) -> scp_view.SCPSession:
    return scp_view.SCPSession(BY_CODE["SCP"],
                               source=source or ReplaySource(str(BENCH_OK), realtime=False,
                                                             loop=True),
                               channels=channels, **kwargs)


def _run(session, n_trials: int = 6, transfer: float = 0.0, seed: int = 7) -> TrialRun:
    """Replace the engine with a seeded one, so a test reads a fixed sequence of directions."""
    from nfb_console.training import true_metric_hz

    session.trials = TrialRun(n_trials=n_trials, transfer_fraction=transfer, seed=seed,
                              metric_hz=true_metric_hz(session.rate))
    return session.trials


def _drive(session, seconds: float, uv=None, start: int = 0, watch=None) -> int:
    """Push blocks through `consume` for a stretch of signal, returning the next sample index.

    `uv` is called with the session before each block, so a *compliant* trainee can be simulated: it reads
    the engine's own phase and produces a shift in whichever direction the trial asked for. `watch` is
    called after each block, which is where a test collects what it is asserting about.
    """
    spec = session.specs[session.channel]
    scale = 1.0 / spec.scale
    n = int(seconds * RATE)
    for i in range(0, n - 7, 8):
        value = 0.0 if uv is None else uv(session)
        samples = tuple(int(value * scale) for _ in range(8))
        session.consume(Block(spec.index, start + i, samples, spec))
        if watch is not None:
            watch()
    return start + (n - n % 8)


def _compliant(session) -> float:
    run = session.trials
    if run.phase != ACTIVE or run.required is None:
        return 0.0
    return COMPLIANT_UV if run.required == POSITIVE else -COMPLIANT_UV


def _releases(session) -> list[str]:
    """Record what the session hands back, the way `test_hrv_view` does. Returns the live list."""
    released: list[str] = []
    for name in ("display", "output_volume", "music"):
        obj = getattr(session, name)
        setattr(obj, "release", (lambda n=name: lambda: released.append(n))())
    return released


# --- the level -------------------------------------------------------------------------------

def test_a_transfer_trial_shows_nothing_while_it_runs():
    """That is what a transfer trial *is* — cue and verdict, no display — so the level has to be absent
    through the whole of its feedback window, not frozen at its last value."""
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=4, transfer=1.0)
        during: list[float | None] = []

        def watch() -> None:
            if run.phase == ACTIVE:
                during.append(session.signal)

        _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 90.0, uv=_compliant, start=int(5 * RATE), watch=watch)

        assert during, "at least one feedback window has to have run"
        assert all(x is None for x in during), "a transfer trial fed something back"
        assert all(r.transfer for r in run.results)
        assert run.results, "and it still scores — the verdict is not what a transfer withholds"
    finally:
        session.stop()


def test_between_trials_there_is_no_level_and_the_room_is_handed_back():
    """A rest interval with the sound and the shadow held at their last value says "you are still doing
    it" while nobody is doing anything, so the transition out of a feedback window releases them."""
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=3)
        released = _releases(session)
        _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 60.0, uv=_compliant, start=int(5 * RATE))

        assert run.results, "a trial has to have finished for there to be a transition"
        assert run.phase != ACTIVE or session.signal is None
        assert "display" in released and "output_volume" in released and "music" in released
    finally:
        session.stop()


# --- the sounds ------------------------------------------------------------------------------

def test_each_trial_is_cued_and_only_a_success_is_marked():
    """One tone per trial start, at that direction's own pitch, and the same tone doubled on a success.

    Failure and void are silent, which is the rule every discrete output here follows: the absence is the
    message. The recorder is `session.cue`, which is also what keeps the test silent.
    """
    session = _session()
    try:
        session.set_trial_direction("alternating")      # a direction named at the cue: one tone per trial
        session.toggle_pause()
        run = _run(session, n_trials=6)
        played: list[str] = []
        session.cue = lambda name, volume=None: played.append(name)

        _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 120.0, uv=_compliant, start=int(5 * RATE))

        cues_played = [x for x in played if "+held" not in x]
        marks = [x for x in played if "+held" in x]
        successes = sum(1 for r in run.results if r.success)
        assert successes > 0, "a compliant trainee should be succeeding"
        assert len(cues_played) == len(run.results) + (1 if run.required is not None and not run.done else 0)
        assert len(marks) == successes
        # the two cue pitches are an octave apart, and both were used
        assert len(set(cues_played)) == 2, "both directions were cued, at their own pitches"
    finally:
        session.stop()


# --- the transport ----------------------------------------------------------------------------

def test_pausing_mid_trial_abandons_it_and_hands_the_room_back():
    """A trial's baseline describes the signal immediately before its feedback window. Look away for
    twenty seconds and it no longer does, so the engine's time-jump rule discards the trial in flight
    rather than scoring it — and the transport releases the outputs on the way out."""
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=6)
        _drive(session, 5.0, uv=lambda _s: 0.0)
        at = int(5 * RATE)
        while run.phase != ACTIVE:
            at = _drive(session, 0.5, uv=_compliant, start=at)
            assert at < int(90 * RATE), "no trial ever started"
        scored = len(run.results)

        released = _releases(session)
        session.toggle_pause()
        assert released == ["display", "output_volume", "music"], "pausing hands all three back"
        assert session.paused

        # time passes with nobody watching: the blocks keep arriving, the engine is not fed
        at = _drive(session, 20.0, uv=_compliant, start=at)
        assert len(run.results) == scored, "a paused session must not score trials"
        session.toggle_pause()
        _drive(session, 1.0, uv=_compliant, start=at)
        assert run.phase == ITI, "the trial in flight was abandoned, not resumed"
        assert len(run.results) == scored + 1, "the person did that trial, so it stays as a void cell"
        assert run.results[-1].cause == "paused" and not run.results[-1].valid, \
            "and it was not scored on its stale baseline"
        assert session.signal is None
    finally:
        session.stop()


def test_the_end_of_the_run_ends_the_session():
    """The run *is* the session's length on this screen, so the last verdict stops the clock — otherwise
    the transport goes on counting over a screen where nothing further can happen."""
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=1)
        _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 40.0, uv=_compliant, start=int(5 * RATE))
        assert run.done and len(run.results) == 1
        assert session.finished and session.paused
        assert session.signal is None
    finally:
        session.stop()


def test_a_run_carries_its_settings_across_a_stop_and_not_its_results():
    """`adopt`'s line, on this screen: the configuration is what an afternoon of setting up produced, the
    results are what this session produced. A fresh run inheriting the last one's history would report a
    rate over trials nobody in the room has run."""
    first = _session()
    try:
        first.toggle_pause()
        first.set_channel(1)
        first.set_range(25.0)
        first.n_trials = 12
        first.transfer_fraction = 0.5
        first.set_success_at(3.0, typed=True)
        first.restart_trials()
        _drive(first, 5.0, uv=lambda _s: 0.0)
        _drive(first, 40.0, uv=_compliant, start=int(5 * RATE))
        assert first.trials.results, "there has to be a history for clearing it to mean anything"

        second = _session()
        try:
            second.adopt(first)
            assert second.channel == 1
            assert second.range_uv == 25.0
            assert (second.n_trials, second.transfer_fraction, second.success_at) == (12, 0.5, 3.0)
            assert (second.full_at_mode, second.success_at_mode) == ("fixed", "fixed")
            assert second.trials.n_trials == 12
            assert second.trials.results == [], "the run's results are the session's, not the setup's"
            assert second.metric is not None and second.metric.spec.index == 1
        finally:
            second.stop()
    finally:
        first.stop()


def test_changing_the_channel_starts_the_run_again():
    """A run half-measured on one electrode and half on another is not a run, and its success rate would
    be a rate over two different measurements."""
    session = _session()
    try:
        session.toggle_pause()
        _run(session, n_trials=6)
        _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 40.0, uv=_compliant, start=int(5 * RATE))
        assert session.trials.results
        session.set_channel(1)
        assert session.trials.results == []
        assert session.signal is None
    finally:
        session.stop()


# --- the wiring check --------------------------------------------------------------------------

def test_the_corner_check_says_how_much_of_a_shift_survives():
    """The metric has no high-pass of its own, so the amplifier's corner is the only thing under the band
    — and what it costs is a fraction of the *mean* over the feedback window, which is exactly what a
    trial is scored on."""
    from nfb import Settings, WaitingSource

    bad = _session(source=WaitingSource(Settings(channels=2, lower_limit_hz=0.5).build()))
    try:
        problem = bad.wiring_problem()
        assert problem and "0.01 Hz" in problem and "%" in problem
        assert len(problem) < 220, "it sits in a narrow column; the arithmetic is behind the ?"
    finally:
        bad.stop()
    good = _session(source=WaitingSource(Settings(channels=2, lower_limit_hz=0.01).build()))
    try:
        assert good.wiring_problem() is None
    finally:
        good.stop()


# --- the screen -------------------------------------------------------------------------------

def _page():
    from harness import StubPage

    return StubPage()


def _toolbar():
    from harness import toolbar

    return toolbar()


def _texts(control, depth: int = 0):
    from harness import _texts as walk

    return walk(control, depth)


def test_the_screen_draws_before_any_data_and_says_what_is_missing():
    session = _session()
    try:
        control = scp_view.view(_page(), session, _toolbar(), lambda: None)
        session.redraw()
        assert session.error is None
        joined = " | ".join(c.value for c in _texts(control) if c.value)
        assert "no trials yet" in joined, "an explicit empty state, not a blank strip"
        assert "no readings yet" in joined
        assert "across" in joined and "full scale" in joined, "every graph states its window and scale"
    finally:
        session.stop()


def test_every_tab_draws_with_trials_running_through_it():
    """A press on a tab shows that tab's pane and hides every other, and each pane draws a frame with
    trials running through it."""
    from test_scp_eyes import _press

    page, session = _page(), _session()
    try:
        control = scp_view.view(page, session, _toolbar(), lambda: None)
        session.toggle_pause()
        _run(session, n_trials=6)
        _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 60.0, uv=_compliant, start=int(5 * RATE))
        panes = control.controls[0].controls[2].controls           # the screen, under the eye overlay
        for i, label in reversed(list(enumerate(("Train", "Spectrum", "Trends", "Feedback", "Session")))):
            _press(control, label)
            assert [pane.visible is not False for pane in panes] == [j == i for j in range(len(panes))], label
            session.redraw()
        assert session.error is None, session.error
        joined = " | ".join(c.value for c in _texts(control) if c.value)
        assert "up" in joined and "down" in joined
        assert "mean +" in joined or "mean -" in joined, "each direction's line carries its mean shift"
        assert "settling" not in joined, "it settled; the empty state should be gone"
    finally:
        session.stop()


def test_every_control_on_the_screen_can_be_pressed():
    """Every control here is a lambda closing over a session, and a wrong argument name or a stale
    attribute raises **out of a Flet event handler** — a traceback in the terminal and nothing on
    screen. Pressing them is the only way an object test can see it."""
    page, session = _page(), _session()
    try:
        control = scp_view.view(page, session, _toolbar(), lambda: None)
        from test_wave_view import _clickables

        handlers = _clickables(control)
        assert len(handlers) > 15, "the screen is supposed to have controls on it"
        for label, handler in handlers:
            if label in (None, "?", "browse…", "▶", "▲", "▼", "×"):
                continue
            handler(None)
        session.redraw()
        assert session.error is None
        assert isinstance(session.trials, TrialRun), "still a working run"
    finally:
        session.stop()


def test_the_signal_help_explains_the_extra_channels_only_where_one_can_be_picked():
    """With one open channel there is no extra channel to pick, and the help does not explain one."""
    one, two = _session(channels=1), _session(channels=2)
    try:
        assert scp_view.extra_help(one) not in scp_view.signal_help(one)
        assert scp_view.extra_help(two) in scp_view.signal_help(two)
    finally:
        one.stop()
        two.stop()


def test_the_screen_does_not_lecture_and_names_nobody():
    """The two conventions a protocol screen is held to: no permanent sentence over 110 characters, and
    never a person's name in the interface."""
    page, session = _page(), _session()
    try:
        control = scp_view.view(page, session, _toolbar(), lambda: None)
        session.redraw()
        lines = [c.value for c in _texts(control) if c.value]
        assert not [x for x in lines if len(x) > 110], "it lectures"
        joined = " ".join(lines)
        for word in (" he ", " his ", "He ", "His "):
            assert word not in joined, f"the screen names a person: {word!r}"
    finally:
        session.stop()


def test_session_for_ignores_the_other_sources():
    """`app.pick` hands every screen the heart and Mendi runners, because two screens need them. A screen
    that raised on the arguments would fail at the moment it was chosen."""
    session = scp_view.session_for(BY_CODE["SCP"],
                                   source=ReplaySource(str(BENCH_OK), realtime=False, loop=True),
                                   channels=2, heart_choice=object(), heart_runner=object(),
                                   mendi_choice=object(), mendi_runner=object())
    try:
        assert isinstance(session, scp_view.SCPSession)
    finally:
        session.stop()


def test_a_fresh_run_is_the_published_protocol():
    """A fresh run is the published protocol: both directions in a balanced order, the baseline at the cue,
    some trials without a display, feedback for the whole window, live, the trials of the clinical block,
    and **full at** and **succeeds at** from the person's own noise. The one departure is that the
    direction is hidden until the feedback, so the baseline cannot be pushed the other way. The choices are
    the ones every trial program offers."""
    from nfb_console.trial_training import DIRECTIONS
    from nfb_console.trials import DEFAULT_TRIALS

    session, control = _screen()
    try:
        assert session.DIRECTIONS is DIRECTIONS
        assert session.hides_direction
        run = session.trials
        assert set(run.directions) == {POSITIVE, NEGATIVE} and not run.alternate
        assert run.cue_s == 0.0, "the baseline starts at the cue"
        assert 0.0 < session.transfer_fraction < 1.0 and run.transfer_fraction == session.transfer_fraction
        assert session.n_trials == run.n_trials == DEFAULT_TRIALS
        assert session.feedback_window_s == run.active_s and session.update_s == 0.0
        assert (session.full_at_mode, session.success_at_mode) == ("automatic", "automatic")
        text = _joined(session, control)
        assert "automatic" in text and "noise" in text, text
    finally:
        session.stop()


def test_the_cue_tones_are_an_octave_apart_and_a_success_doubles_its_tone():
    """Up and down have their own tones an octave apart, told apart with the eyes closed. The verdict on a
    success is the cue tone held twice, and a failure is silent. Every cue sounds at one volume."""
    from nfb_console import cues

    session = _session()
    try:
        session.set_trial_direction("alternating")      # a direction named at the cue: no middle tone
        heard: list = []
        session.cue = lambda sound, volume=None: heard.append((sound, volume))
        session.toggle_pause()
        _run(session, n_trials=2)
        nxt = _drive(session, 3.0, uv=lambda _s: 0.0)
        _drive(session, 30.0, uv=_compliant, start=nxt)
        pitch = lambda name: float(name.split("@")[1])     # noqa: E731
        plain = sorted({sound for sound, _ in heard if cues.held(sound) != sound}, key=pitch)
        assert len(plain) == 2, heard
        low, high = (pitch(name) for name in plain)
        assert high == pytest.approx(2 * low, abs=1.0), "an octave apart"
        assert {sound for sound, _ in heard} <= set(plain) | {cues.held(name) for name in plain}
        assert any(cues.held(name) in {sound for sound, _ in heard} for name in plain), \
            "a compliant trial succeeds and doubles its tone"
        assert len({volume for _, volume in heard}) == 1
    finally:
        session.stop()


def test_the_random_default_sounds_a_neutral_cue_and_names_the_direction_as_the_feedback_starts():
    """The cue tone is the one between the two, and the screen does not know the direction through the cue
    and the baseline. The direction tone sounds as the feedback starts, so the baseline cannot be pushed the
    other way to make the shift look larger."""
    from nfb_console import cues
    from nfb_console.trial_training import CUE_PITCH_NEUTRAL
    from nfb_console.trials import BASELINE, CUED

    session = _session()
    try:
        heard: list = []
        session.cue = lambda sound, volume=None: heard.append(sound)
        session.toggle_pause()
        _run(session, n_trials=2)
        known: dict[str, set] = {}
        watch = lambda: known.setdefault(session.trials.phase, set()).add(session.direction_known())  # noqa: E731
        nxt = _drive(session, 3.0, uv=lambda _s: 0.0)
        _drive(session, 30.0, uv=_compliant, start=nxt, watch=watch)
        assert heard[0] == cues.tone_at(CUE_PITCH_NEUTRAL), "the cue does not name the direction"
        assert heard[1] in (cues.tone_at(620.0), cues.tone_at(310.0)), "the direction tone comes next"
        assert known.get(CUED, set()) | known[BASELINE] == {False} and known[ACTIVE] == {True}
    finally:
        session.stop()


def test_the_verdict_says_what_was_asked_and_how_far_the_signal_went():
    """A tone says success or nothing; the line under the bar says how much, and which way, so a near miss
    and a large shift the wrong way read differently."""
    from nfb_console.trials import RESULT

    page, session = _page(), _session()
    try:
        control = scp_view.view(page, session, _toolbar(), lambda: None)
        session.toggle_pause()
        _run(session, n_trials=3)
        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        seen: list[str] = []

        def watch():
            if session.trials.phase == RESULT and not seen:
                session.redraw()
                seen.append(" | ".join(c.value for c in _texts(control) if c.value))

        _drive(session, 30.0, uv=_compliant, start=nxt, watch=watch)
        assert seen, "a trial reached its verdict"
        assert "asked for up · the slow potential went" in seen[0] or \
            "asked for down · the slow potential went" in seen[0]
        assert " µV " in seen[0]
    finally:
        session.stop()


def test_music_is_offered_with_every_source_but_the_eeg_rhythm():
    """Music runs under the trials, as on Pupil and HEG. The EEG rhythm is the one source left out: it
    sonifies a fast band, and a slow potential has no rhythm in it for an ear to follow."""
    session = _session()
    try:
        from nfb_console.feedback import MUSIC_SOURCES

        assert "music" in session.offers
        assert set(session.music_sources) == set(MUSIC_SOURCES) - {"eeg"}
        assert session.state_balance is None, "no crossfade position between feedback windows"
    finally:
        session.stop()


def test_a_chosen_heart_fills_the_heart_stats_beside_the_trials():
    """As on Pupil and HEG: a heart chosen on the device screen reaches the screen, and how each direction's
    effort moves it is gathered trial by trial. Without one, no heart panel appears."""
    import types

    from nfb_console.inputs import HeartChoice

    class _Heart:
        stale, error = False, None

        def __init__(self, session):
            self.session = session

        @property
        def state(self):
            lift = 6.0 if self.session.trials.phase == ACTIVE else 0.0
            return types.SimpleNamespace(hr_bpm=70.0 + lift)

    plain = _session()
    try:
        assert not plain.heart_on
    finally:
        plain.stop()

    session = scp_view.session_for(BY_CODE["SCP"], source=ReplaySource(str(BENCH_OK), realtime=False, loop=True),
                                   channels=2, heart_choice=HeartChoice(kind="simulator"), heart_runner=None)
    try:
        session.heart = _Heart(session)
        assert session.heart_on
        session.toggle_pause()
        _run(session, n_trials=4)
        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 60.0, uv=_compliant, start=nxt)
        summary = session.heart_trials.direction_summary()
        assert summary and all(change == pytest.approx(6.0, abs=0.5) for _, change, _ in summary)
    finally:
        session.stop()


# --- the trial panel: the shared fixation circle, each phase its own picture --------------------------


def _ring_view(session):
    from nfb_console.trial_training import TrialView

    ui = TrialView(session, scp_view.CIRCLE_SIDE, heart=False)
    ui.fixation_panel()
    return ui


def _solid_circles(shapes) -> list:
    """The circles drawn in a solid line: the feedback circle and the mean ring, not the point or the
    dashed baseline ring."""
    import flet.canvas as cv

    return [s for s in shapes if isinstance(s, cv.Circle) and s.radius > 3.0
            and not s.paint.stroke_dash_pattern]


def _wrong_way(uv: float):
    def produce(session) -> float:
        run = session.trials
        if run.phase != ACTIVE or run.required is None:
            return 0.0
        return -uv if run.required == POSITIVE else uv
    return produce


def test_the_ring_shows_a_shift_far_the_wrong_way():
    """The bar this replaced was empty for any shift of full at or more the wrong way, so the screen went
    blank when the trainee did it wrong. The circle mirrors the signal and never sticks at its limit, so a
    shift three times full at the wrong way draws a circle, grey, inside its limits and away from the ring."""
    from nfb_console import theme as t
    from nfb_console.trial_training import _radii

    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=4)
        drawn: list = []

        def watch() -> None:
            if run.phase == ACTIVE and session.feedback_now and run.phase_remaining() < 1.0:
                ui.draw_fixation("")
                drawn.append((run.required, list(ui.circle.shapes), ui.ground.bgcolor))

        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 30.0, uv=_wrong_way(3 * session.full_at), start=nxt, watch=watch)
        assert drawn, "a feedback window ran"
        r_min, r_max, r_mid = _radii(scp_view.CIRCLE_SIDE)
        for required, shapes, ground in drawn:
            circles = _solid_circles(shapes)
            assert circles, "a circle is drawn for a shift far the wrong way"
            moving = circles[-1]
            assert moving.paint.color == t.FAINT, "grey on the side the cue did not ask for"
            assert r_min < moving.radius < r_max, "never at the limit"
            assert (moving.radius > r_mid) == (required == NEGATIVE), "it mirrors the signal: up is larger"
            assert ground == t.SURFACE, "the feedback window is bright"
    finally:
        session.stop()


def test_the_baseline_is_dim_and_draws_its_ring_with_no_circle_and_no_arrows():
    """The trainee started the task in the baseline because it looked like the feedback. With the direction
    hidden, the lead-in is dim, the dashed ring draws itself and is part drawn, and nothing that moves
    with the signal or names the aim is on the screen."""
    import math

    import flet.canvas as cv
    from nfb_console import theme as t
    from nfb_console.trials import BASELINE

    session = _session()
    try:
        assert session.hides_direction, "random is the default"
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=3)
        seen: list = []

        def watch() -> None:
            if run.phase == BASELINE:
                ui.draw_fixation("")
                seen.append((run.trial_number, list(ui.circle.shapes), ui.ground.bgcolor,
                             ui.direction_text.value))
                assert not ui.count_text.value, "this program has no counting task"

        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 25.0, uv=_compliant, start=nxt, watch=watch)
        assert seen, "a baseline ran"
        sweeps: dict[int, list[float]] = {}
        for trial, shapes, ground, word in seen:
            assert ground == t.SURFACE_HI, "dim"
            assert not _solid_circles(shapes), "no circle"
            assert not [s for s in shapes if isinstance(s, cv.Path)], "no arrows"
            assert word == "?", "the direction stays hidden"
            sweeps.setdefault(trial, []).extend(s.sweep_angle for s in shapes if isinstance(s, cv.Arc))
        whole = sweeps[2]                   # the first trial's baseline began while the filter settled
        assert all(b >= a for a, b in zip(whole, whole[1:])), "the ring draws itself"
        assert whole[0] < 0.25 * math.pi and 1.75 * math.pi < whole[-1] < 2 * math.pi + 1e-9, \
            "part drawn from the start of the baseline, and close to whole at its end"
    finally:
        session.stop()


def test_the_arrows_stay_through_a_feedback_window_with_no_circle():
    """The arrows are the goal. A trial without feedback still asks for a direction, so its window shows
    the arrows and no circle, and the line says there is no feedback in that window only."""
    import flet.canvas as cv
    from nfb_console.trials import BASELINE

    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=3, transfer=1.0)
        seen: list = []

        lead: list[str] = []

        def watch() -> None:
            if run.phase == ACTIVE:
                ui.draw_fixation("")
                seen.append(list(ui.circle.shapes))
                assert "no feedback" in ui.phase_text.value, ui.phase_text.value
            elif run.phase == BASELINE:
                ui.draw_fixation("")
                lead.append(ui.phase_text.value)

        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 25.0, uv=_compliant, start=nxt, watch=watch)
        assert seen and lead, "a baseline and a feedback window ran"
        assert not any("no feedback" in text for text in lead), "the baseline does not say it"
        for shapes in seen:
            assert not _solid_circles(shapes), "no circle"
            assert len([s for s in shapes if isinstance(s, cv.Path)]) == 4, "the four arrows"
    finally:
        session.stop()


def test_the_feedback_shows_the_mean_so_far_as_a_thin_ring():
    """The trial is scored on the mean of the window, so a thin ring shows that mean as the window runs."""
    from nfb_console.trial_training import _radii

    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=3)
        found: list = []

        def watch() -> None:
            if run.phase == ACTIVE and session.feedback_now and session.shown_mean is not None:
                ui.draw_fixation("")
                level = session.circle_level(session.shown_mean)
                r_min, r_max, _ = _radii(scp_view.CIRCLE_SIDE)
                want = r_min + level * (r_max - r_min)
                found.append([c for c in _solid_circles(ui.circle.shapes)
                              if abs(c.radius - want) < 1e-6
                              and abs(c.paint.stroke_width * c.radius - 1.5 * _radii(scp_view.CIRCLE_SIDE)[2]) < 1e-6])

        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 15.0, uv=_compliant, start=nxt, watch=watch)
        assert found and all(found), "the mean ring is drawn at the mean so far"
    finally:
        session.stop()


def test_the_lines_under_the_circle_say_why_there_is_no_trial():
    """Before the first reading the line says so, and the end of the run says where its restart button is."""
    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        ui.draw_fixation("")
        assert "no readings" in ui.phase_text.value, ui.phase_text.value
        run = _run(session, n_trials=1)
        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 20.0, uv=_compliant, start=nxt)
        assert run.done
        ui.draw_fixation("")
        assert "restart" in ui.phase_text.value, ui.phase_text.value
    finally:
        session.stop()


# --- data quality: driven through feed_block, on levels that wander -----------------------------------

BLOCK = 50
"""Samples per block where blocking is not the subject: a fifth of a second at 250 sps."""


def _feed(session, waves: dict, start: int = 0, add=None, until=None, watch=None,
          block: int = BLOCK) -> int:
    """Every channel in `waves`, one block each in turn, through `feed_block`, as the device sends them.

    `waves` maps a channel to its signal in µV, indexed by sample. `add(session, channel)` is added to that
    channel's block, which is how a source locked to the cues is made. Returns the next sample index."""
    from conftest import counts

    n = min(len(w) for w in waves.values())
    i = start
    while i + block <= n:
        for ch, wave in waves.items():
            spec = session.specs[ch]
            segment = wave[i:i + block] + (add(session, ch) if add is not None else 0.0)
            session.feed_block(Block(ch, i, tuple(counts(segment, spec.scale).tolist()), spec))
        i += block
        if watch is not None:
            watch()
        if until is not None and until(session):
            break
    return i


def _rail(seconds: float, uv: float = -800.0) -> np.ndarray:
    return np.full(int(seconds * RATE), uv)


def _oriented(trial) -> float:
    return trial.score_uv if trial.required == POSITIVE else -trial.score_uv


def test_level_starts_at_the_first_accepted_block():
    """Three railed blocks, then the electrode at −300 µV. The gate holds the first blocks of the step; the
    level starts at the first block it accepts, not from zero and not from a held stand-in."""
    from conftest import counts, wandering

    session = _session()
    try:
        session.toggle_pause()
        spec = session.specs[0]
        rail = int(round(-800.0 / spec.scale))
        for i in range(3):
            session.feed_block(Block(0, i * 8, (rail,) * 8, spec, state=3))
        x = -300.0 + wandering(1.5, 4.0, RATE, 1, 2.0)
        first = None
        for i in range(0, len(x) - 7, 8):
            session.feed_block(Block(0, 24 + i, tuple(counts(x[i:i + 8], spec.scale).tolist()), spec))
            if session.raw_shift is not None:
                first = session.raw_shift
                break
        assert first is not None, "the gate accepted a block of the step"
        assert abs(first + 300.0) < 10.0, f"the level starts at {first:.1f} µV"
    finally:
        session.stop()


def test_each_trial_event_is_marked_with_its_sample_index():
    """The cue, the feedback start and the verdict are marked with the training channel's sample index in
    the session's stream, beside the wall time."""
    import re

    from conftest import wandering

    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=2, transfer=0.0, seed=14)
        starts: list[int] = []
        last = [run.phase]

        def watch():
            if run.phase == ACTIVE and last[0] != ACTIVE:
                starts.append(int(session.stream_seconds * RATE))
            last[0] = run.phase

        _feed(session, {0: wandering(1.5, 40.0, RATE, 31, 4.0)}, watch=watch, until=_until_done(session))
        notes = [note for _at, note in session.marks() if note.startswith("trial ")]
        assert [n.split(" · ")[1] for n in notes] == ["cue", "feedback starts", "verdict"] * 2, notes
        marked = [int(re.search(r"ch0 sample (\d+)", n).group(1)) for n in notes if "feedback starts" in n]
        assert len(marked) == len(starts) == 2
        assert all(0 <= at - mark <= BLOCK for mark, at in zip(marked, starts)), (marked, starts)
    finally:
        session.stop()


def test_the_raw_caption_leaves_the_input_limit_to_the_level_plot():
    """One sample at the limit on ch0, and ch1 wholly on it. The SCP screen shows that in its level plot,
    so its RAW caption does not; on a screen without the plot the caption names both, and says "rail"
    nowhere."""
    from harness import _trace_rows
    from conftest import wandering

    session = _session()
    try:
        page = _page()
        control = scp_view.view(page, session, _toolbar(), lambda: None)
        session.toggle_pause()
        x = wandering(1.5, 4.0, RATE, 30, 2.0)[:int(2 * RATE)]
        x[-1] = -800.0
        _feed(session, {0: x, 1: _rail(2.0)})
        session.redraw()
        caption = [c for c in _texts(control) if c.value and "s across" in c.value and "drawn" in c.value]
        assert caption and "input limit" not in caption[0].value, caption[0].value
        _trace_rows(page, session).refresh(session.trace_seconds)
        assert "ch0, ch1 at the input limit" in caption[0].value, caption[0].value
        assert "rail" not in caption[0].value
    finally:
        session.stop()


def _screen(channels: int = 2):
    session = _session(channels=channels)
    control = scp_view.view(_page(), session, _toolbar(), lambda: None)
    session.toggle_pause()
    return session, control


def _joined(session, control) -> str:
    session.redraw()
    return " | ".join(c.value for c in _texts(control) if c.value)


def test_an_extra_at_the_input_limit_shows_its_status_line():
    """Whole blocks on the rail resolve the channel dead, and it is still *at the input limit*: the line
    says so, whichever path the blocks took."""
    from conftest import wandering

    session, control = _screen()
    try:
        session.set_extra(1, True)
        _feed(session, {0: wandering(1.5, 4.0, RATE, 3, 2.0)[:int(3 * RATE)], 1: _rail(3.0)})
        assert "ch1 · at the input limit · no check now" in _joined(session, control)
    finally:
        session.stop()


def test_an_extra_that_is_dead_says_no_signal():
    from conftest import wandering

    session, control = _screen()
    try:
        session.set_extra(1, True)
        _feed(session, {0: wandering(1.5, 4.0, RATE, 4, 2.0)[:int(3 * RATE)], 1: np.zeros(int(3 * RATE))})
        assert "ch1 · no signal · no check now" in _joined(session, control)
    finally:
        session.stop()


def test_the_extra_channels_row_shows_only_with_two_open_channels():
    """At one open channel there is nothing to pick and no row. With four, two can be lit and the third
    chip is inert, with the reason on the status line."""
    one, control = _screen(channels=1)
    try:
        assert "extra channels" not in _joined(one, control)
    finally:
        one.stop()
    four, control = _screen(channels=4)
    try:
        text = _joined(four, control)
        assert "extra channels" in text and "ch1" in text
        four.set_extra(1, True)
        four.set_extra(2, True)
        four.set_extra(3, True)
        assert tuple(four.extras) == (1, 2), "at most two are lit"
        assert "at most two extra channels" in _joined(four, control)
    finally:
        four.stop()


def _until_done(session):
    return lambda _s: session.trials.done


def test_a_cue_locked_shift_on_the_training_and_an_extra_channel_withholds_keep():
    """A shift in every feedback window that also reaches the extra channel, larger there. The end of the
    run names what was measured and does not say "keep this strategy"."""
    from conftest import cue_locked, wandering

    session, control = _screen()
    try:
        session.set_extra(1, True)
        _run(session, n_trials=8, transfer=0.0, seed=11)
        waves = {0: wandering(1.5, 140.0, RATE, 5, 2.0), 1: wandering(1.5, 140.0, RATE, 6, 4.0)}
        size = {0: 20.0, 1: 60.0}
        _feed(session, waves, add=lambda s, ch: cue_locked(s.trials, size[ch]), until=_until_done(session))
        assert session.trials.done
        text = _joined(session, control)
        assert "ch1 also followed the cues" in text, text
        assert "keep this strategy" not in text
    finally:
        session.stop()


def test_an_extra_railed_all_run_does_not_check():
    from conftest import cue_locked, wandering

    session, control = _screen()
    try:
        session.set_extra(1, True)
        _run(session, n_trials=6, transfer=0.0, seed=12)
        waves = {0: wandering(1.5, 110.0, RATE, 7, 2.0), 1: _rail(110.0)}
        _feed(session, waves, add=lambda s, ch: cue_locked(s.trials, 20.0) if ch == 0 else 0.0,
              until=_until_done(session))
        text = _joined(session, control)
        assert "ch1 did not check this run" in text, text
        assert "ch1 did not follow the cues" not in text
    finally:
        session.stop()


def _outputs_of(training: int, extra: int, kind: str, seed: int = 5) -> tuple[list, list]:
    """What reaches the outputs on a short run, and the trials it scored, with the extra channel `kind`:
    "quiet", "cued" (a large cue-locked shift) or "rejected" (a spike in every block)."""
    from conftest import cue_locked, wandering

    session = _session()
    try:
        session.set_channel(training)
        session.set_extra(extra, True)
        session.toggle_pause()
        _run(session, n_trials=3, transfer=0.0, seed=seed)
        moved: list = []
        original = session._move_level
        session._move_level = lambda level, t, smoothed=True: (moved.append((round(t, 6), level)),
                                                               original(level, t, smoothed))
        waves = {0: wandering(1.5, 60.0, RATE, 8, 2.0), 1: wandering(1.5, 60.0, RATE, 9, 2.0)}
        if kind == "rejected":
            waves[extra] = waves[extra].copy()
            waves[extra][BLOCK // 2::BLOCK] += 400.0

        def add(s, ch):
            if ch == training:
                return cue_locked(s.trials, 6.0)
            return cue_locked(s.trials, 80.0) if kind == "cued" else 0.0

        _feed(session, waves, add=add, until=_until_done(session))
        scored = [(r.score_uv, r.valid, r.success) for r in session.trials.results]
        return moved, scored
    finally:
        session.stop()


def test_a_rejected_extra_block_never_stops_the_outputs():
    """Blocks come one channel at a time. An extra channel whose every block is rejected changes nothing
    on the training channel, with the training channel first or second."""
    for training, extra in ((0, 1), (1, 0)):
        quiet = _outputs_of(training, extra, "quiet")
        rejected = _outputs_of(training, extra, "rejected")
        assert quiet[0], "the level moved in the feedback windows"
        assert rejected == quiet, f"training ch{training}: a rejected extra block changed the outputs"


def test_an_extra_channels_results_never_reach_the_output_path():
    """An extra channel is measured for the end of the run and for nothing else: a large shift on it in
    every window leaves the level, the outputs and every score exactly as they are with it quiet."""
    assert _outputs_of(0, 1, "cued") == _outputs_of(0, 1, "quiet")


def test_an_extra_that_stops_sending_does_not_stall_the_training_channel():
    """A lit extra channel whose blocks stop: nothing waits for it. The trials run on, the level moves, and
    its line says it has no signal; it vouches for no trial it did not carry."""
    from conftest import cue_locked, wandering

    session, control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=4, transfer=0.0, seed=16)
        x = {0: wandering(1.5, 80.0, RATE, 33, 2.0), 1: wandering(1.5, 80.0, RATE, 34, 2.0)}
        at = _feed(session, {0: x[0][:int(10 * RATE)], 1: x[1][:int(10 * RATE)]})
        assert "ch1 · " in _joined(session, control)
        moved: list = []
        original = session._move_level
        session._move_level = lambda level, t, smoothed=True: (moved.append(level), original(level, t, smoothed))
        _feed(session, {0: x[0]}, start=at, add=lambda s, _ch: cue_locked(s.trials, 6.0),
              until=_until_done(session))
        assert run.done and moved, "the training channel ran the whole run alone"
        assert "ch1 · no signal · no check now" in _joined(session, control)
        late = [score for trial, score in zip(run.results, session.extra_checks()[1]) if trial.index > 1]
        assert late and all(score is None for score in late), "it carried none of those trials"
    finally:
        session.stop()


def test_an_extra_with_too_few_valid_readings_vouches_for_nothing():
    """An extra channel with a spike in every other block loses about half of its readings in each trial,
    more than the 20% a trial of the training channel may lose. It vouches for none of them."""
    from conftest import wandering

    session, _control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=3, transfer=0.0, seed=19)
        x = {0: wandering(1.5, 60.0, RATE, 38, 2.0), 1: wandering(1.5, 60.0, RATE, 39, 2.0)}
        x[1][BLOCK // 2::2 * BLOCK] += 400.0
        _feed(session, x, until=_until_done(session))
        assert run.done and all(trial.valid for trial in run.results)
        assert session.extra_checks()[1] == [None] * len(run.results)
    finally:
        session.stop()


def test_a_held_extra_block_is_not_a_channel_with_no_signal():
    """A rejected block reaches the measurement as a flat stand-in. The status line judges the samples as
    they arrived, so an extra channel with a blink in its newest block still has a signal."""
    from conftest import wandering

    session, control = _screen()
    try:
        session.set_extra(1, True)
        x = {0: wandering(1.5, 4.0, RATE, 41, 2.0), 1: wandering(1.5, 4.0, RATE, 42, 2.0)}
        x[1][-BLOCK // 2] += 400.0
        _feed(session, x)
        assert session._accepted[1] is False, "the newest block of ch1 was held"
        text = _joined(session, control)
        assert "ch1 · no signal" not in text and "ch1 · " in text
    finally:
        session.stop()


def test_a_chip_pressed_mid_run_changes_nothing_in_flight():
    """Lighting an extra channel during a feedback window does not start the run again and moves neither
    the trial's record nor the level."""
    from conftest import wandering

    session, control = _screen()
    try:
        run = _run(session, n_trials=3, transfer=0.0, seed=17)
        x = wandering(1.5, 60.0, RATE, 35, 4.0)
        at = _feed(session, {0: x, 1: x[::-1].copy()}, until=lambda _s: run.phase == ACTIVE
                   and run.phase_remaining() < 3.0)
        record, level, scored = session.trial_scale, session.signal, list(run.results)
        session.set_extra(1, True)
        assert session.trials is run and run.phase == ACTIVE, "the run goes on"
        assert (session.trial_scale, session.signal, run.results) == (record, level, scored)
        assert session.extras == (1,)
        _feed(session, {0: x, 1: x[::-1].copy()}, start=at, until=_until_done(session))
        assert run.done and len(session.extra_checks()[1]) == len(run.results)
    finally:
        session.stop()


def test_the_training_channel_moved_to_a_lit_chip_starts_again():
    """The training channel becomes ch1 while ch1 is lit: the run starts again, that chip goes out, and
    the old training channel is an unlit chip."""
    session, control = _screen(channels=4)
    try:
        session.set_extra(1, True)
        session.set_extra(2, True)
        old = session.trials
        session.set_channel(1)
        assert session.trials is not old and session.extras == (2,)
        session.set_extra(0, True)
        assert session.extras == (0, 2), "the old training channel is a chip now"
        text = _joined(session, control)
        assert "ch0 · no readings yet" in text and "ch2 · no readings yet" in text
        assert "ch1 · no readings yet" not in text, "the training channel has no status line"
    finally:
        session.stop()


def test_adopt_keeps_only_the_lit_channels_that_exist():
    four = _session(channels=4)
    try:
        four.set_extra(1, True)
        four.set_extra(3, True)
        two = _session(channels=2)
        try:
            two.adopt(four)
            assert two.extras == (1,)
        finally:
            two.stop()
    finally:
        four.stop()


def test_with_no_extra_channel_lit_nothing_asks_for_one():
    """Two channels open and no chip lit: every part works, the run is judged on its own scores, and no
    line asks for an extra channel or treats one channel as a fault."""
    from conftest import cue_locked, wandering

    session, control = _screen()
    try:
        _run(session, n_trials=8, transfer=0.0, seed=18)
        waves = {0: wandering(1.5, 150.0, RATE, 36, 2.0), 1: wandering(1.5, 150.0, RATE, 37, 2.0)}
        _feed(session, waves, add=lambda s, ch: cue_locked(s.trials, 30.0) if ch == 0 else 0.0,
              until=_until_done(session))
        text = _joined(session, control)
        assert "past the chance range · keep this strategy" in text, text
        for asking in ("ch1 ·", "extra channel ", "no check", "select", "one channel"):
            assert asking not in text, asking
    finally:
        session.stop()


# --- the record of each trial: automatic succeeds at and full at --------------------------------------

def _record(scale) -> tuple:
    return (scale.full_at, scale.limit, scale.full_mode, scale.limit_mode, scale.spread, scale.seconds)


def test_the_limit_is_frozen_at_the_feedback_start():
    """**succeeds at** is one chance spread and **full at** two, each written once at the feedback start.
    The live spread moves with every reading; the trial in flight is scored against its own record."""
    from conftest import wandering

    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=5, transfer=0.0, seed=3)
        seen: dict[int, set] = {}
        moved: list[bool] = []

        def watch():
            if run.phase == ACTIVE and session.trial_scale is not None:
                seen.setdefault(run.trial_number, set()).add(_record(session.trial_scale))
                moved.append(session.full_at != session.trial_scale.full_at)

        _feed(session, {0: wandering(1.5, 90.0, RATE, 21, 8.0)}, watch=watch, until=_until_done(session))
        assert run.done and all(len(records) == 1 for records in seen.values()), "one record per trial"
        assert any(moved), "the live value moved while the record held"
        for trial in run.valid_results:
            scale = trial.scale
            assert (scale.full_mode, scale.limit_mode) == ("automatic", "automatic")
            floor = scp_view.UNITS.min_full_at
            assert scale.full_at == pytest.approx(max(floor, 2.0 * scale.spread))
            assert scale.limit == pytest.approx(max(floor / 2, scale.spread))
            assert trial.success == (_oriented(trial) >= scale.limit and trial.score_uv != 0.0)
    finally:
        session.stop()


def test_a_typed_value_sets_fixed():
    from conftest import wandering
    from nfb_console.trials import RESULT

    session = _session()
    try:
        ui = _ring_view(session)
        session.set_full_at(25.0, typed=True)
        session.set_success_at(4.0, typed=True)
        assert (session.full_at_mode, session.success_at_mode) == ("fixed", "fixed")
        session.toggle_pause()
        run = _run(session, n_trials=2, transfer=0.0, seed=4)
        lines: list[str] = []

        def watch():
            if run.phase == RESULT and run.last.valid and not lines:
                ui.draw_fixation("")
                lines.append(ui.verdict_detail.value)

        _feed(session, {0: wandering(1.5, 40.0, RATE, 22, 8.0)}, watch=watch, until=_until_done(session))
        for trial in run.results:
            assert (trial.scale.full_at, trial.scale.limit) == (25.0, 4.0)
            assert (trial.scale.full_mode, trial.scale.limit_mode) == ("fixed", "fixed")
        assert lines and lines[0].endswith("· fixed"), lines
    finally:
        session.stop()


def test_no_spread_uses_the_floors_and_says_so():
    """With no chance spread at a feedback start, the record takes the program's resolution floors. The
    trial is scored, not voided, and the second line under the circle says why the limit is small."""
    from conftest import wandering
    from nfb_console.trials import RESULT

    session = _session()
    try:
        ui = _ring_view(session)
        session.chance_spread = lambda: None
        session.toggle_pause()
        run = _run(session, n_trials=2, transfer=0.0, seed=5)
        lines: list[str] = []

        def watch():
            if run.phase == RESULT and not lines:
                ui.draw_fixation("")
                lines.append(ui.verdict_detail.value)

        _feed(session, {0: wandering(1.5, 30.0, RATE, 23, 8.0)}, watch=watch,
              until=lambda _s: bool(run.results))
        trial = run.results[0]
        assert trial.valid, "no spread is no reason to void a trial"
        floor = scp_view.UNITS.min_full_at
        assert (trial.scale.full_at, trial.scale.limit, trial.scale.spread) == (floor, floor / 2, None)
        assert lines and lines[0].startswith(f"succeeds at {floor / 2:.1f} µV") and "noise" in lines[0], lines
    finally:
        session.stop()


def test_a_flat_signal_sets_the_floors():
    """A channel with almost no slow movement has a spread far under a microvolt. **full at** stays at its
    1 µV floor and **succeeds at** at half of it, so no trial succeeds on quantisation."""
    from conftest import wandering

    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=3, transfer=0.0, seed=6)
        tt = np.arange(int(60 * RATE)) / RATE
        x = wandering(1.5, 60.0, RATE, 24, 0.01) + np.sin(2 * np.pi * 10.0 * tt)
        _feed(session, {0: x}, until=_until_done(session))
        assert run.done and all(trial.valid for trial in run.results)
        floor = scp_view.UNITS.min_full_at
        assert all((trial.scale.full_at, trial.scale.limit) == (floor, floor / 2) for trial in run.results)
    finally:
        session.stop()


def test_a_10x_signal_is_not_clamped():
    """The box's range bounds a typed value. An automatic one follows the person's signal past it."""
    from conftest import wandering

    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=6, transfer=0.0, seed=7)
        x = wandering(1.8, 120.0, RATE, 31, 50.0)
        assert np.abs(x).max() < 700.0, "the fixture stays inside the input range"
        _feed(session, {0: x}, until=_until_done(session))
        scales = [trial.scale for trial in run.valid_results if trial.scale.spread]
        assert scales
        assert all(s.full_at == pytest.approx(2.0 * s.spread) and s.limit == pytest.approx(s.spread)
                   for s in scales)
        assert max(s.full_at for s in scales) > scp_view.UNITS.max_full_at
        assert max(s.limit for s in scales) > scp_view.UNITS.max_success
    finally:
        session.stop()


def test_an_automatic_box_shows_the_value_in_use():
    """On *automatic* the **full at** and **succeeds at** boxes show what the trial uses, as it moves,
    without a commit: the value stays automatic."""
    from conftest import wandering

    session = _session()
    try:
        card = scp_view.SCPCard(_page(), session, lambda: None)
        session.toggle_pause()
        _run(session, n_trials=2, transfer=0.0, seed=15)
        _feed(session, {0: wandering(1.5, 30.0, RATE, 32, 8.0)}, until=lambda _s: bool(session.trials.results))
        card.refresh()
        full, limit = session.scale_in_use()
        assert full > scp_view.UNITS.min_full_at, "a spread of the person's own"
        shown = {name: box.controls[0].value for name, box in card._scale_boxes.items()
                 if not name.endswith(("_note", "_pill"))}
        assert shown == {"full_at": f"{full:.1f}", "success_at": f"{limit:.1f}"}, shown
        assert (session.full_at_mode, session.success_at_mode) == ("automatic", "automatic")
    finally:
        session.stop()


def test_adopt_carries_the_mode_not_the_automatic_value():
    from conftest import wandering

    first = _session()
    try:
        first.toggle_pause()
        _run(first, n_trials=2, transfer=0.0, seed=8)
        _feed(first, {0: wandering(1.5, 40.0, RATE, 25, 8.0)}, until=_until_done(first))
        assert first.full_at > first.UNITS.min_full_at, "the first session has a spread of its own"
        second = _session()
        try:
            second.adopt(first)
            assert second.full_at_mode == "automatic"
            assert second.full_at == second.UNITS.min_full_at, "not the old session's value"
        finally:
            second.stop()
        first.set_full_at(25.0, typed=True)
        third = _session()
        try:
            third.adopt(first)
            assert (third.full_at_mode, third.full_at) == ("fixed", 25.0)
        finally:
            third.stop()
    finally:
        first.stop()


def test_every_reader_takes_the_frozen_record_inside_a_trial():
    """The level for the outputs, the circle, its colour, the arc and the card's scale line all read the
    record of the trial's feedback start, while the live value moves under them."""
    import math

    import flet.canvas as cv
    from conftest import wandering
    from nfb_console import theme as t
    from nfb_console.trial_training import _radii, picture_position

    session = _session()
    try:
        ui = _ring_view(session)
        card = scp_view.SCPCard(_page(), session, lambda: None)
        session.toggle_pause()
        run = _run(session, n_trials=4, transfer=0.0, seed=9)
        r_min, r_max, _ = _radii(scp_view.CIRCLE_SIDE)
        checked: list[bool] = []

        def watch():
            if not (run.phase == ACTIVE and session.feedback_now and session.shown_shift is not None):
                return
            scale, sign = session.trial_scale, (1.0 if run.required == POSITIVE else -1.0)
            for shift in (-7.0, 3.0, 12.0):
                level = min(1.0, max(0.0, 0.5 + sign * shift / (2 * scale.full_at)))
                assert session.trial_level(shift) == pytest.approx(level)
                assert session.circle_level(shift) == pytest.approx(
                    0.5 + 0.5 * picture_position(shift, scale.full_at))
            card.refresh()
            assert f"{scale.full_at:.1f}" in card.scale_text.value
            ui.draw_fixation("")
            assert not [s for s in ui.circle.shapes if isinstance(s, cv.Arc)], "no blocks on the circle"
            want = r_min + session.circle_level(sign * scale.limit) * (r_max - r_min)
            moving = _solid_circles(ui.circle.shapes)[-1]
            rings = [s for s in _solid_circles(ui.circle.shapes)[:-1] if s.radius == pytest.approx(want)]
            assert len(rings) == 1 and rings[0].paint.stroke_width < moving.paint.stroke_width, \
                "one ring at the limit, thinner than the circle"
            past = sign * session.shown_shift >= scale.limit
            assert (moving.paint.color == t.ACCENT) == past, "the cued colour only past the limit"
            checked.append(not math.isclose(session.full_at, scale.full_at))

        _feed(session, {0: wandering(1.5, 70.0, RATE, 26, 8.0)}, watch=watch, until=_until_done(session))
        assert checked and any(checked), "the live value moved while the readers held the record"
    finally:
        session.stop()


def test_a_shift_short_of_the_limit_does_not_click():
    """The click and the music reward a shift past **succeeds at**, not every shift the cued way: in a
    feedback window the condition is met exactly where the shown shift passes the frozen limit."""
    from conftest import wandering

    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=3, transfer=0.0, seed=10)
        target = session.signal_target
        seen: list[tuple[bool, bool]] = []

        def watch():
            if run.phase == ACTIVE and session.active_targets and session.shown_shift is not None:
                limit = session.trial_scale.limit
                oriented = session.shown_shift if run.required == "positive" else -session.shown_shift
                seen.append((target.wanted, oriented >= limit))

        _feed(session, {0: wandering(1.5, 50.0, RATE, 27, 8.0)}, block=25, watch=watch,
              until=_until_done(session))
        assert seen and all(met == past for met, past in seen)
        assert any(past for _, past in seen) and not all(past for _, past in seen)
    finally:
        session.stop()


def test_a_verdict_names_the_trial_it_judges():
    """The run counts a trial as its feedback ends, so at the verdict the line under the circle names the
    trial being judged, not the next one."""
    from conftest import wandering
    from nfb_console.trials import RESULT

    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=3, transfer=0.0, seed=20)
        seen: list[tuple[int, str]] = []

        def watch():
            if run.phase == RESULT and not run.done:
                ui.draw_fixation("")
                seen.append((run.last.index, ui.phase_text.value))

        _feed(session, {0: wandering(1.5, 60.0, RATE, 40, 4.0)}, watch=watch, until=_until_done(session))
        assert seen and all(text.startswith(f"trial {index} of 3 · verdict") for index, text in seen), seen[:3]
    finally:
        session.stop()


def test_a_trial_with_a_blink_in_every_block_has_no_result():
    """A spike in every block of the feedback window is rejected, and the trial counts those readings
    against it: it has no result, and its cause is the rejected readings."""
    from conftest import wandering
    from nfb_console.trials import REJECTED

    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=2, transfer=0.0, seed=13)
        x = wandering(1.5, 40.0, RATE, 28, 4.0)
        at = _feed(session, {0: x}, until=lambda _s: run.phase == ACTIVE)
        x[at + BLOCK // 2:at + int(run.active_s * RATE):BLOCK] += 400.0
        _feed(session, {0: x}, start=at, until=lambda _s: bool(run.results))
        trial = run.results[0]
        assert not trial.valid and trial.cause == REJECTED, trial
    finally:
        session.stop()


def test_a_trial_lost_to_the_input_limit_says_so():
    """Three seconds of the feedback window with a sample at the rail in every block: the trial has no
    result, and the line under the circle names the input limit."""
    from conftest import wandering
    from nfb_console.trials import RESULT

    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=2, transfer=0.0, seed=13)
        x = wandering(1.5, 40.0, RATE, 28, 4.0)
        at = _feed(session, {0: x}, block=25, until=lambda _s: run.phase == ACTIVE)
        x[at + 24:at + int(3 * RATE):25] = 800.0
        lines: list[str] = []

        def watch():
            if run.phase == RESULT and not lines:
                ui.draw_fixation("")
                lines.append(ui.verdict_text.value)

        _feed(session, {0: x}, start=at, block=25, watch=watch, until=lambda _s: bool(lines))
        assert not run.results[0].valid and run.results[0].cause == "limit"
        assert "input limit" in lines[0], lines
    finally:
        session.stop()


# --- the demeaning: the trainee's own shift stays out of the chance spread ---------------------------

_BASE: dict[int, list] = {}


def _frozen_spreads(seed: int, size: float = 0.0, shape: str = "step", same_sign: bool = False,
                    from_trial: int = 0) -> list:
    """The chance spread frozen at each feedback start of a 12-trial run, with a cue-locked skill of
    `size` µV from trial `from_trial` on. The same seed gives the same signal and the same trials."""
    from conftest import BETAS, STEPS_UV, cue_locked, wandering

    session = _session(channels=1)
    try:
        session.toggle_pause()
        run = _run(session, n_trials=12, transfer=1 / 3, seed=100 + seed)
        beta, step = BETAS[(2 * seed) % len(BETAS)], STEPS_UV[seed % len(STEPS_UV)]

        def add(s, _ch):
            if not size or len(run.results) < from_trial:
                return 0.0
            return cue_locked(run, size, shape=shape, same_sign=same_sign)

        _feed(session, {0: wandering(beta, 180.0, RATE, 80 + seed, step)}, add=add,
              until=_until_done(session), block=125)
        return [trial.scale.spread for trial in run.results]
    finally:
        session.stop()


def _rise(spreads: int = 1, **skill) -> float:
    """The median over three seeds of the late spread with skill against the same run without it."""
    ratios = []
    for seed in range(3):
        if seed not in _BASE:
            _BASE[seed] = _frozen_spreads(seed)
        base = _BASE[seed]
        size = spreads * float(np.median(base[3:]))
        late = _frozen_spreads(seed, size=size, **skill)[8:]
        ratios.append(float(np.median(late)) / float(np.median(base[8:])))
    return float(np.median(ratios))


def test_the_limit_at_trial_ten_stays_near_the_skill_free_limit():
    """A compliant trainee with one chance spread of skill. Without the demeaning the limit rises 22%;
    with it, a few percent."""
    assert _rise(1) < 1.12


def test_a_ramped_skill_raises_the_limit_less_than_without_the_demeaning():
    """A shift that builds through the window is not removed by its mean. It still costs much less than
    the ×2.1 it costs with no demeaning."""
    assert _rise(2, shape="ramp") < 1.8


def test_a_skill_that_starts_mid_run_raises_the_limit_within_its_stated_cost():
    assert _rise(2, from_trial=6) < 1.6


def test_a_same_sign_shift_is_removed_by_the_raw_score_mean():
    """The mean taken off is of the raw `score_uv`, not of the score toward the cue. A shift up in every
    window whatever the cue is removed, where a mean toward the cue would double it."""
    assert _rise(2, same_sign=True) < 1.35


def test_the_run_starts_at_the_first_reading():
    """No settling gate: the lead-in to the first cue starts with the first reading, and nothing on the
    screen says the filter is settling."""
    from conftest import wandering

    session, control = _screen()
    try:
        _feed(session, {0: wandering(1.5, 10.0, RATE, 29, 8.0)[:int(0.5 * RATE)]})
        assert session.trials.started, "the run waits for nothing"
        assert "settling" not in _joined(session, control)
    finally:
        session.stop()


def test_a_difference_that_rounds_to_zero_has_no_minus_sign():
    """A difference that rounds to zero prints as +0.0 and is not called the wrong way."""
    from nfb_console.trial_training import trial_summary
    from nfb_console.trials import Trial

    run = TrialRun(n_trials=6, transfer_fraction=0.0)
    scores = [(POSITIVE, 1.0), (POSITIVE, -1.0), (POSITIVE, 0.92), (NEGATIVE, 1.0), (NEGATIVE, -1.0),
              (NEGATIVE, 0.96)]
    run.results = [Trial(index=i + 1, required=r, transfer=False, score_uv=s, valid=True, success=False)
                   for i, (r, s) in enumerate(scores)]
    text = trial_summary(run, scp_view.SCPSession.DIRECTION_WORD, "µV", 1, floor=1.0)
    assert "up minus down +0.0 µV" in text, text
    assert "the wrong way" not in text, "a difference of zero goes no way"


def test_a_trial_cut_by_a_pause_says_so_in_the_rest_after_it():
    """A pause voids the trial in flight and the run rests at once, with no verdict phase. The two lines
    under the circle say why through that rest."""
    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=6)
        at = _drive(session, 5.0, uv=lambda _s: 0.0)
        while run.phase != ACTIVE:
            at = _drive(session, 0.5, uv=_compliant, start=at)
        session.toggle_pause()
        at = _drive(session, 5.0, uv=_compliant, start=at)
        session.toggle_pause()
        _drive(session, 0.5, uv=_compliant, start=at)
        assert run.phase == ITI and run.last.cause == "paused"
        ui.draw_fixation("")
        assert "paus" in ui.verdict_text.value, ui.verdict_text.value
        assert ui.verdict_detail.value
    finally:
        session.stop()


def test_an_automatic_row_with_no_spread_says_so_and_not_chance_spreads():
    """Before the first chance spread an automatic box holds its floor, and the line under it says there is
    none yet, so the floor does not read as one or two of them."""
    session = _session()
    try:
        card = scp_view.SCPCard(_page(), session, lambda: None)
        card.refresh()
        notes = {name: box.value for name, box in card._scale_boxes.items() if name.endswith("_note")}
        assert set(notes) == {"success_at_note", "full_at_note"}
        assert all("none yet" in note for note in notes.values()), notes
        session.chance_spread = lambda: (4.0, 30.0)
        card.refresh()
        after = {name: box.value for name, box in card._scale_boxes.items() if name.endswith("_note")}
        assert not any("none yet" in note for note in after.values()), after
        assert after["success_at_note"] != after["full_at_note"], "each note says how many spreads"
    finally:
        session.stop()


# --- trial length -------------------------------------------------------------------------------------

def test_at_a_trial_length_of_12_s_the_feedback_lasts_10_s_and_the_score_is_its_mean():
    """A shift toward the cue in the first 5 s of the feedback and none after: the score is the mean over
    the whole 10 s window, about half the shift, where a 6 s window would give about five sixths."""
    import dataclasses

    from conftest import wandering

    session = _session(channels=1)
    try:
        session.toggle_pause()
        session.set_trial_length(12.0)
        assert session.trials.active_s == 10.0
        session.trials = dataclasses.replace(session.trials, n_trials=3, transfer_fraction=0.0, seed=4)

        def first_half(s, _ch):
            run = s.trials
            if run.phase != ACTIVE or run.required is None or run.active_s - run.phase_remaining() >= 5.0:
                return 0.0
            return 20.0 if run.required == POSITIVE else -20.0
        _feed(session, {0: wandering(1.5, 80.0, RATE, 41, 0.2)}, add=first_half,
              until=lambda s: s.trials.done)
        scores = [abs(r.score_uv) for r in session.trials.results]
        assert len(scores) == 3 and all(abs(x - 10.0) < 2.0 for x in scores), scores
    finally:
        session.stop()


def test_a_new_trial_length_restarts_the_run_and_measures_the_noise_again():
    session = _session(channels=1)
    try:
        session.toggle_pause()
        _drive(session, 30.0, uv=_compliant)
        before = session.trials
        assert session.chance_spread() is not None
        session.set_trial_length(10.0)
        assert session.trials is not before and not session.trials.results
        assert session.spread.active_s == session.active_s and session.chance_spread() is None
    finally:
        session.stop()


def test_show_circle_for_never_outlasts_the_feedback_window():
    """The circle shows for at most the feedback window, and the trial length stays inside its range."""
    session = _session(channels=1)
    try:
        session.set_trial_length(14.0)
        assert session.feedback_window_s == session.active_s, "a circle shown for the whole window stays so"
        session.feedback_window_s = session.active_s - 3.0
        session.set_trial_length(6.0)
        assert session.feedback_window_s == session.active_s, "cut to the shorter window"
        low, high = session.trial_length_range
        session.set_trial_length(1000.0)
        assert session.trial_length == high and scp_view.shift_survival(0.01, session.active_s) >= 0.5
        session.set_trial_length(0.0)
        assert session.trial_length == low and session.active_s > 0
    finally:
        session.stop()


def _held(run, size: float) -> float:
    """A shift of `size` toward the cue that rises over the first second of the effort and falls over the
    first second of the verdict, so the gate never sees a step."""
    if run.required is None or run.phase not in (ACTIVE, RESULT):
        return 0.0
    sign = 1.0 if run.required == POSITIVE else -1.0
    if run.phase == ACTIVE:
        part = min(1.0, (run.active_s - run.phase_remaining()) / 1.0)
    else:
        part = max(0.0, 1.0 - (run.result_s - run.phase_remaining()) / 1.0)
    return sign * size * part


@pytest.mark.parametrize("fed, pressed", [(0, 1), (1, 0)], ids=["training-block", "extra-block"])
def test_a_channel_press_while_a_block_is_in_flight_does_not_stop_the_session(fed, pressed):
    """A press between `observe_raw` and `consume`: on a block of the training channel that the press makes
    an extra, and on an extra's block while the press rebuilds the channels."""
    session = _session(channels=2)
    try:
        session.toggle_pause()
        original = session.observe_raw

        def press(block, judged):
            original(block, judged)
            session.set_channel(pressed)

        session.observe_raw = press
        spec = session.specs[fed]
        try:
            session.feed_block(_block(spec, 0, wandering(1.5, 4.0, RATE, 1, 2.0)[:BLOCK]))
        except KeyError as exc:
            pytest.fail(f"the feed raises KeyError {exc}, and the session stops")
    finally:
        session.stop()


def test_a_success_said_good_during_its_trial_and_a_short_shift_never_did():
    """"Good" is what the click and the music ask: the condition target, met past the frozen limit."""
    session = _session()
    try:
        session.toggle_pause()
        session.set_success_at(30.0, typed=True)
        session.set_full_at(20.0, typed=True)
        run = _run(session, n_trials=2, transfer=0.0, seed=21)
        size = {1: 48.0, 2: 24.0}
        good: dict[int, bool] = {1: False, 2: False}

        def watch():
            if run.phase == ACTIVE and session.active_targets and session.signal_target.wanted:
                good[run.trial_number] = True

        x = wandering(1.5, 60.0, RATE, 51, 2.0)
        _feed(session, {0: x}, block=25, watch=watch, until=_until_done(session),
              add=lambda s, _ch: _held(s.trials, size[s.trials.trial_number]))
        first, second = run.results
        assert first.valid and first.success, first
        assert second.valid and not second.success, second
        assert not good[2], "a shift short of succeeds at said good"
        assert good[1], "a trial that succeeds never said good while it ran"
    finally:
        session.stop()


def test_the_share_on_a_void_line_is_the_share_of_the_named_cause():
    session = _session(channels=1)
    try:
        run = TrialRun(n_trials=1, transfer_fraction=0.0, seed=1, metric_hz=10.0)
        k, t = 0, 0.0
        skipped = False
        while not run.results:
            in_trial = run.phase in (BASELINE, ACTIVE)
            if run.phase == ACTIVE and run.phase_remaining() < 3.0 and not skipped:
                t += 0.1                                # one reading never arrives
                skipped = True
            run.feed(t, 5.0, valid=not (in_trial and k % 4 == 0), cause=REJECTED)
            k += 1
            t = round(t + 0.1, 6)
        session.trials = run
        trial = run.results[0]
        assert trial.cause == REJECTED, "the count with the most readings is named"
        what, _action = session.void_words(trial)
        expected = run.baseline_s * 10 + run.active_s * 10
        rejected = trial.invalid_share - 1 / expected          # all but the one that did not arrive
        shares = [int(p) for p in re.findall(r"(\d+)%", what)]
        assert shares, what
        assert abs(shares[0] - 100 * rejected) <= 1, f"{what!r}: the rejected share alone, not the lost one"
    finally:
        session.stop()


def test_an_extra_channel_of_a_file_with_a_standing_offset_has_a_signal(tmp_path):
    from nfb.sources import SampleFileSource

    n = int(10 * RATE)
    rng = np.random.default_rng(4)
    ch0 = wandering(1.5, 10.0, RATE, 61, 2.0)
    ch1 = 100.0 + 3.0 * rng.standard_normal(n)
    path = tmp_path / "export.txt"
    path.write_text("Sampling rate: 250\n" + "".join(f"{a:.5f}\t{b:.5f}\n" for a, b in zip(ch0, ch1)))
    source = SampleFileSource(path, realtime=False, loop=False)
    session = _session(channels=2, source=source)
    try:
        session.set_extra(1, True)
        for block in source.blocks():
            session.feed_block(block)
        line, _problem = session.extra_status(1)
        assert "no signal" not in line, line
        assert session.headroom_uv(1) is None and not session.at_input_limit(1), "an export has no input limit"
    finally:
        session.stop()


def test_a_trial_across_the_seam_of_a_looped_file_has_no_result(tmp_path):
    from nfb.sources import SampleFileSource

    seconds = 30.0
    tt = np.arange(int(seconds * RATE)) / RATE
    rise = 0.5 - 0.5 * np.cos(np.pi * np.clip((tt - 15.0) / 10.0, 0.0, 1.0))
    x = -300.0 + 200.0 * rise + wandering(1.5, seconds, RATE, 71, 2.0)
    path = tmp_path / "export.txt"
    path.write_text("Sampling rate: 250\n" + "".join(f"{v:.5f}\n" for v in x))
    source = SampleFileSource(path, realtime=False, loop=True)
    session = _session(channels=1, source=source)
    try:
        session.toggle_pause()
        run = _run(session, n_trials=12, transfer=0.0, seed=22)
        across: set[int] = set()
        for block in source.blocks():
            if block.seam and run.phase in (BASELINE, ACTIVE):
                across.add(run.trial_number)
            session.feed_block(block)
            if run.done or (across and len(run.results) >= max(across)):
                break
        assert across, "no trial ran across a seam"
        scored = [run.results[i - 1] for i in sorted(across)]
        assert not any(trial.valid for trial in scored), [(t.index, t.score_uv) for t in scored]
        assert all(trial.cause == "seam" for trial in scored), [t.cause for t in scored]
    finally:
        session.stop()


def test_the_slow_level_is_offered_no_mode_that_feeds_it_zeros():
    """Clamp sets a rejected sample to zero, a step of the whole standing offset on a slow level. SCP does
    not offer it, with the reason stated on the session; the Feedback tab draws only what is offered."""
    session, control = _screen(channels=1)
    try:
        assert "clamp" not in session.reject_modes and session.reject_modes
        joined = _joined(session, control)
        assert "hold" in joined.split(" | ") and "clamp" not in joined.split(" | "), joined
    finally:
        session.stop()


def _write_recording(path, start_group: int, seconds: float, hz: float = 250.0) -> None:
    """A recording that starts `start_group` sample groups into a session: frames of eight groups whose
    phase is the wrapped counter, as the device sends them, with a wandering level on every channel."""
    import struct

    import optima
    from nfb.record import SessionHeader, SessionWriter
    from nfb.stream import PHASE_MODULUS

    cfg = optima.build_eeg_config(4, hz=hz)
    plan = [c for c in optima.plan_channels(cfg) if c.enabled]
    n = int(seconds * hz)
    waves = [counts(wandering(1.5, seconds, hz, 80 + p, 2.0), 800.0 / 32767) for p in range(len(plan))]
    header = SessionHeader(config=cfg, device="Neurobit Optima+ 4 USB", serial="00000000",
                           firmware="2.9.5", note="synthetic recording started mid-session")
    with SessionWriter(path, header) as writer:
        for frame in range(n // 8):
            phase = (start_group + frame * 8) % PHASE_MODULUS
            values = [int(waves[p][frame * 8 + step]) for step in range(8) for p in range(len(plan))]
            payload = struct.pack("<HH", phase, 0) + struct.pack(f"<{len(values)}h", *values)
            writer.write(optima.NOT_DATA, payload, elapsed=frame * 8 / hz)


def test_a_marks_sample_index_is_the_live_streams_and_its_wall_time_places_it(tmp_path):
    """A mark's sample index is the live stream's, also for a recording that starts mid-session, and the
    mark keeps the wall time that places it in a recording."""
    from nfb import ReplaySource

    live_start = 70000
    path = tmp_path / "mid-session.jsonl.gz"
    _write_recording(path, live_start, 12.0)
    replayed = [b for b in ReplaySource(str(path), realtime=False, loop=False).blocks()
                if isinstance(b, Block)]
    offset = live_start - min(b.start_index for b in replayed)
    live = set()
    for b in replayed:
        if b.channel == 0:
            live.update(range(b.start_index + offset, b.end_index + offset))
    session = _session(channels=1, source=ReplaySource(str(path), realtime=False, loop=False))
    try:
        session.toggle_pause()
        _run(session, n_trials=1, transfer=0.0, seed=24)
        for b in replayed:
            if b.channel in session.specs:
                session.feed_block(Block(b.channel, b.start_index + offset, b.samples, b.spec, b.state))
        cues = [(at, note) for at, note in session.marks() if " · cue · " in note]
        assert cues, session.marks()
        at, note = cues[0]
        assert int(re.search(r"ch0 sample (\d+)", note).group(1)) in live
        assert at > 1e9, "a wall time, which places the mark in a recording"
    finally:
        session.stop()


def test_a_trial_before_a_rearm_is_not_checked_on_signal_from_after_it():
    session, _control = _screen()
    try:
        run = _run(session, n_trials=14, transfer=0.0, seed=25)
        flat = np.zeros(int(200 * RATE))
        _feed(session, {0: wandering(1.5, 200.0, RATE, 90, 2.0), 1: flat},
              until=lambda _s: len(run.results) >= 5)
        before = len(run.results)
        session.feed_block(Gap(0, 0))
        _feed(session, {0: wandering(1.5, 200.0, RATE, 91, 2.0), 1: wandering(1.5, 200.0, RATE, 92, 2.0)},
              until=_until_done(session))
        assert run.done
        session.set_extra(1, True)
        checks = session.extra_checks()[1][:before]
        assert checks == [None] * before, f"ch1 was flat before the rearm, and its scores are {checks}"
    finally:
        session.stop()


def test_a_trial_lost_to_a_channel_forced_off_does_not_ask_the_person_to_keep_still():
    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=1, transfer=0.0, seed=26)
        x = wandering(1.5, 40.0, RATE, 93, 2.0)
        at = _feed(session, {0: x}, until=lambda _s: run.phase == ACTIVE)
        session.trust_channel(0, False)
        _feed(session, {0: x}, start=at, until=lambda _s: bool(run.results))
        trial = run.results[0]
        assert not trial.valid
        ui.draw_fixation("")
        assert "keep still" not in ui.verdict_detail.value, (trial.cause, ui.verdict_detail.value)
    finally:
        session.stop()


def test_the_no_signal_count_leaves_out_trials_cut_by_a_pause():
    session, control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=9, transfer=0.0, seed=27)
        n = int(300 * RATE)
        ch0, clean = wandering(1.5, 300.0, RATE, 94, 2.0), wandering(1.5, 300.0, RATE, 95, 2.0)
        cut = (1, 2, 3)
        at = 0
        while not run.done and at + BLOCK <= n:
            number = run.trial_number
            ch1 = clean if number in cut and len(run.results) < number else np.zeros(n)
            at = _feed(session, {0: ch0[:at + BLOCK], 1: ch1[:at + BLOCK]}, start=at)
            if number in cut and run.phase == ACTIVE and len(run.results) == number - 1:
                session.toggle_pause()
                at = _feed(session, {0: ch0[:at + int(1.5 * RATE)], 1: clean[:at + int(1.5 * RATE)]},
                           start=at)
                session.toggle_pause()
        assert run.done and [r.cause for r in run.results[:3]] == ["paused"] * 3
        text = _joined(session, control)
        said = re.search(r"ch1 had no signal in (\d+) of (\d+) trials", text)
        assert said, text
        windowed = sum(1 for r in run.results if r.cause != "paused")
        assert int(said.group(1)) <= windowed, f"{said.group(0)}; 3 of them were cut by a pause, with a " \
                                               f"clean signal on ch1"
    finally:
        session.stop()


def test_a_short_pause_in_the_effort_voids_the_trial_in_flight():
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=1, transfer=0.0, seed=28)
        x = wandering(1.5, 40.0, RATE, 96, 2.0)
        at = _feed(session, {0: x}, until=lambda _s: run.phase == ACTIVE and run.phase_remaining() < 4.0)
        session.toggle_pause()
        at = _feed(session, {0: x[:at + int(0.8 * RATE)]}, start=at)
        session.toggle_pause()
        _feed(session, {0: x}, start=at, until=lambda _s: bool(run.results))
        trial = run.results[0]
        assert not trial.valid and trial.cause == "paused", trial
    finally:
        session.stop()


def test_the_extra_channel_is_checked_in_the_windows_of_the_training_trial():
    from nfb_console.training import true_metric_hz

    session, _control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=1, transfer=0.0, seed=29)
        x = {0: wandering(1.5, 60.0, RATE, 97, 2.0), 1: wandering(1.5, 60.0, RATE, 98, 2.0)}
        at = _feed(session, x, block=25, until=lambda _s: run.phase == BASELINE
                   and run.phase_remaining() < 0.3)
        at += int(0.6 * RATE)                           # a hole of 0.6 s across the feedback start
        _feed(session, x, start=at, block=25, until=lambda _s: bool(run.results))
        assert run.results[0].valid
        (_index, b0, a0, _a1), = session._windows
        step = 1.0 / true_metric_hz(RATE)
        assert abs(a0 - (b0 + run.baseline_s)) <= 1.5 * step, \
            f"the extra window starts {a0 - b0 - run.baseline_s:.2f} s after the feedback start"
    finally:
        session.stop()


def test_an_emptied_box_changes_neither_the_mode_nor_the_run():
    session = _session()
    try:
        card = scp_view.SCPCard(_page(), session, lambda: None)
        session.chance_spread = lambda: (1.2, 30.0)
        card.refresh()
        field = card._scale_boxes["success_at"].controls[0]
        assert field.value == "1.2", field.value
        run = session.trials
        field.value = ""
        field.on_blur(None)
        assert session.success_at_mode == "automatic", (session.success_at_mode, session.success_at)
        assert session.trials is run, "the run started again"
        assert field.value == "1.2", "the box shows its last value again"
        field.value = "abc"
        field.on_submit(None)
        assert (session.success_at_mode, field.value) == ("automatic", "1.2") and session.trials is run
    finally:
        session.stop()


def test_the_automatic_pill_acts_on_each_press():
    """A press on the lit pill fixes the value on screen; a second press goes back to automatic."""
    session = _session()
    try:
        card = scp_view.SCPCard(_page(), session, lambda: None)
        session.chance_spread = lambda: (1.23, 30.0)
        card.refresh()
        for name, spreads in (("success_at", 1.0), ("full_at", 2.0)):
            press = card._scale_boxes[f"{name}_pill"].on_click
            press(None)
            assert getattr(session, f"{name}_mode") == "fixed", name
            assert getattr(session, name) == pytest.approx(round(spreads * 1.23, 1)), "the value on screen"
            press(None)
            assert getattr(session, f"{name}_mode") == "automatic", name
    finally:
        session.stop()


def test_the_output_level_is_above_the_middle_toward_the_cue_and_below_it_away():
    """A shift toward the cue drives every output above the middle, and a shift away drives it below."""
    for toward in (True, False):
        session = _session()
        try:
            session.toggle_pause()
            run = _run(session, n_trials=2, transfer=0.0, seed=30)
            seen: list[float] = []

            def watch():
                if run.phase == ACTIVE and run.active_s - run.phase_remaining() > 2.0 \
                        and session.signal is not None:
                    seen.append(session.signal)

            _feed(session, {0: wandering(1.5, 40.0, RATE, 100, 1.0)}, block=25, watch=watch,
                  until=_until_done(session), add=lambda s, _ch: (1 if toward else -1) * _held(s.trials, 12.0))
            assert seen
            assert all((level > 0.5) == toward for level in seen), (toward, min(seen), max(seen))
        finally:
            session.stop()


def test_an_extra_channel_is_not_checked_for_5_s_after_it_touches_the_input_limit():
    session, _control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=1, transfer=0.0, seed=31)
        x = {0: wandering(1.5, 40.0, RATE, 101, 2.0), 1: wandering(1.5, 40.0, RATE, 102, 2.0)}
        at = _feed(session, x, until=lambda _s: run.phase == BASELINE)
        x[1][at + 10] = 800.0                           # one sample at the limit as the baseline starts
        _feed(session, x, start=at, until=lambda _s: bool(run.results))
        assert run.results[0].valid
        assert session.extra_checks()[1] == [None], "5 s of the window came after a touch of the limit"
    finally:
        session.stop()


def test_an_extra_with_a_thin_baseline_vouches_for_nothing():
    """The extra channel is flat for the first 1.2 s of the baseline: 15% of the trial's readings, under the
    20% rule, but less than half of its baseline. By the trial engine's own baseline rule it checks nothing."""
    session, _control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=1, transfer=0.0, seed=33)
        x = {0: wandering(1.5, 40.0, RATE, 103, 2.0), 1: wandering(1.5, 40.0, RATE, 104, 2.0)}
        at = _feed(session, x, until=lambda _s: run.phase == BASELINE)
        x[1][at:at + int(1.2 * RATE)] = 5.0                 # flat: no signal there
        _feed(session, x, start=at, until=lambda _s: bool(run.results))
        assert run.results[0].valid
        assert session.extra_checks()[1] == [None], "a thin baseline on the extra vouched for the trial"
    finally:
        session.stop()


def test_at_a_verdict_every_reader_still_takes_the_trials_record():
    session = _session()
    try:
        session.toggle_pause()
        session.set_full_at(20.0, typed=True)
        session.set_success_at(4.0, typed=True)
        run = _run(session, n_trials=2, transfer=0.0, seed=34)
        _feed(session, {0: wandering(1.5, 40.0, RATE, 105, 2.0)}, until=lambda _s: run.phase == RESULT)
        session.set_full_at(50.0, typed=True)
        session.success_at_setting = session.success_at_setting._replace(value=9.0)
        assert session.scale_in_use() == (20.0, 4.0), session.scale_in_use()
    finally:
        session.stop()


def test_a_fixed_box_keeps_the_value_typed_for_the_next_trial():
    """Inside a trial the readers take the record, but a box on *fixed* shows what was typed, which acts
    from the next trial: the frame does not write the record over it."""
    session = _session()
    try:
        card = scp_view.SCPCard(_page(), session, lambda: None)
        session.toggle_pause()
        session.set_full_at(20.0, typed=True)
        card.rebuild()
        run = _run(session, n_trials=2, transfer=0.0, seed=35)
        _feed(session, {0: wandering(1.5, 40.0, RATE, 106, 2.0)}, until=lambda _s: run.phase == ACTIVE)
        field = card._scale_boxes["full_at"].controls[0]
        field.value = "30"
        field.on_submit(None)
        card.refresh()
        shown = card._scale_boxes["full_at"].controls[0].value
        assert session.scale_in_use()[0] == 20.0 and shown == "30.0", shown
    finally:
        session.stop()


def test_at_a_verdict_the_spread_is_demeaned_without_the_trial_it_judges():
    from nfb_console.trials import Trial

    session = _session()
    try:
        run = TrialRun(n_trials=5)
        run.results.extend(Trial(index=i + 1, required=POSITIVE, transfer=False, score_uv=s, valid=True,
                                 success=False) for i, s in enumerate((2.0, 4.0, 30.0)))
        assert session._own_shift(run, (RESULT, POSITIVE, 3)) == pytest.approx(3.0)
        assert session._own_shift(run, (ACTIVE, POSITIVE, 3)) == pytest.approx(12.0)
    finally:
        session.stop()


def test_the_input_limit_clears_five_seconds_of_samples_after_the_last_one_there():
    """*at the input limit* holds for 5 s of samples after the last sample at the limit, to the sample, counted
    from the blocks and not from a clock, so a replay at any speed clears at the same sample."""
    session = _session()
    try:
        spec = session.specs[0]
        uv = np.full(8, 5.0)
        uv[0] = 800.0                                   # the last sample at the limit opens the block
        session.feed_block(_block(spec, 0, uv))
        left = round(session.LIMIT_CLEAR_SECONDS * spec.rate_hz) - 7
        session.feed_block(_block(spec, 8, np.full(left - 1, 5.0)))
        assert session.at_input_limit(0)
        session.feed_block(_block(spec, 8 + left - 1, np.full(1, 5.0)))
        assert not session.at_input_limit(0), "5 s of samples after the last one at the limit"
    finally:
        session.stop()


def test_an_extra_that_spent_the_run_at_the_input_limit_says_why_it_checked_nothing():
    session, control = _screen()
    try:
        session.set_extra(1, True)
        run = _run(session, n_trials=3, transfer=0.0, seed=36)
        x = {0: wandering(1.5, 80.0, RATE, 107, 2.0), 1: np.full(int(80 * RATE), -800.0)}
        at = _feed(session, x, until=lambda _s: run.done)
        x[1][at:] = wandering(1.5, 80.0, RATE, 108, 2.0)[at:]
        _feed(session, x, start=at, until=lambda s: s.stream_seconds * RATE > at + 6 * RATE)
        assert not session.at_input_limit(1), "the channel has left the input limit"
        line, problem = session.extra_status(1)
        assert "checked in 0 of 3" in line and "3 at the input limit" in line and problem, (line, problem)
        assert line in _joined(session, control)
    finally:
        session.stop()


def test_each_run_is_marked_and_a_void_trial_mark_has_no_score():
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=1, transfer=0.0, seed=37)
        x = wandering(1.5, 40.0, RATE, 109, 2.0)
        at = _feed(session, {0: x}, until=lambda _s: run.phase == ACTIVE)
        session.trust_channel(0, False)
        _feed(session, {0: x}, start=at, until=lambda _s: bool(run.results))
        session.restart_trials()
        notes = [note for _at, note in session.marks()]
        verdict = next(n for n in notes if " · verdict · " in n)
        assert "no result · no signal" in verdict and "µV" not in verdict, verdict
        assert notes[-1].startswith("run starts · "), notes
    finally:
        session.stop()
