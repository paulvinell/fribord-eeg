"""The SCP screen: what it feeds back, when it feeds back nothing, and what it makes a noise about.

The metric and the trial engine are checked in `test_scp.py` against inputs whose answer is known. What is
left — and it is the whole of this file — is the *session*: that a level exists during a trial's feedback
window and nowhere else, that it is oriented toward the direction the trial cued, that a transfer trial
produces none, that the outputs are handed back between trials rather than held at their last value, and
that the two sounds this protocol has fire exactly when they should.

Signals are fed as `Block`s straight through `consume`, as `test_wave.py` does, because these need a
*known* waveform: the bench recording is a constant at the rail and a trial engine driven by it would run
forty trials of nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfb import ReplaySource
from nfb.stream import Block
from nfb_console import scp_view
from nfb_console.protocols import BY_CODE
from nfb_console.trials import ACTIVE, ITI, NEGATIVE, POSITIVE, TrialRun

from conftest import BENCH_OK

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
    """Record what the session hands back, the way `test_hrv_level` does. Returns the live list."""
    released: list[str] = []
    for name in ("display", "output_volume", "music"):
        obj = getattr(session, name)
        setattr(obj, "release", (lambda n=name: lambda: released.append(n))())
    return released


# --- the level -------------------------------------------------------------------------------

def test_a_rejected_block_is_marked_invalid_to_the_trial_engine():
    """An artifact-rejected block arrives as a held stand-in whose device state is still valid, so SCP used
    to feed it to the trial engine as a genuine reading — a run of blinks (which land exactly in the SCP
    band) scored as a valid baseline and shift. The reading is now marked invalid when the block was not
    accepted, so the engine counts it toward voiding the trial rather than scoring it."""
    session = _session()
    try:
        session.toggle_pause()
        nxt = _drive(session, 3.0, uv=lambda _s: 0.0)        # warm the metric past the 2 s settle
        assert session.settled

        captured: list[bool] = []
        original = session.trials.feed

        def spy(t, value, valid=True):
            captured.append(valid)
            return original(t, value, valid=valid)

        session.trials.feed = spy
        spec = session.specs[session.channel]
        session._accepted[session.channel] = False           # as feed_block records for a held block
        session.consume(Block(spec.index, nxt, tuple([0] * 8), spec))
        assert captured and all(v is False for v in captured), \
            "a rejected block feeds the trial engine an invalid reading, not a valid one"
    finally:
        session.stop()


def test_nothing_is_fed_back_before_the_metric_settles():
    """The filter needs two seconds, and a reading that is still a transient would go into a baseline and
    then be subtracted from six seconds of real signal. **Nothing**, not zero: a mechanism handed zero
    drives itself to its floor, which reads as a bad result rather than as no result."""
    session = _session()
    try:
        session.toggle_pause()
        assert session.signal is None and session.active_targets == []
        _drive(session, 1.0, uv=lambda _s: 5.0)
        assert not session.settled
        assert session.signal is None and session.active_targets == []
        assert not session.trials.results, "and the engine has not been fed either"
    finally:
        session.stop()


def test_the_level_is_the_shift_oriented_toward_the_cue():
    """The claim the screen rests on, and the orientation is the half that is easy to get wrong.

    Every mechanism reads one number and none of them knows which way this trial goes, so a down trial has
    to raise the level for a *fall*. Otherwise compliance on half the trials would darken the screen.
    """
    session = _session()
    try:
        session.toggle_pause()
        run = _run(session, n_trials=8)
        seen: list[tuple[str, float, float]] = []

        def watch() -> None:
            if session.signal is not None:
                seen.append((run.required, run.shift, session.signal))

        _drive(session, 5.0, uv=lambda _s: 0.0)          # settle the filter
        _drive(session, 120.0, uv=_compliant, start=int(5 * RATE), watch=watch)

        assert seen, "some feedback window should have driven the level"
        for required, shift, signal in seen:
            oriented = shift if required == POSITIVE else -shift
            expected = min(1.0, max(0.0, 0.5 + oriented / (2 * session.range_uv)))
            assert signal == pytest.approx(expected), (required, shift, signal)

        # both directions ran, and compliance reads high on both — which is what "oriented" means
        for direction in (POSITIVE, NEGATIVE):
            settled = [s for d, _shift, s in seen if d == direction][-5:]
            assert settled, f"no {direction} trial ran"
            assert min(settled) > 0.5, f"compliance on a {direction} trial read low"
    finally:
        session.stop()


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
        assert len(cues_played) == len(run.results) + (1 if run.required is not None else 0)
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
        assert len(run.results) == scored, "and it was not scored on its stale baseline"
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
        first.success_at = 3.0
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
            assert second.trials.n_trials == 12 and second.trials.success_uv == 3.0
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
    from test_screens import StubPage

    return StubPage()


def _toolbar():
    from test_screens import toolbar

    return toolbar()


def _texts(control, depth: int = 0):
    from test_screens import _texts as walk

    return walk(control, depth)


def test_the_screen_draws_before_any_data_and_says_what_is_missing():
    session = _session()
    try:
        control = scp_view.view(_page(), session, _toolbar(), lambda: None)
        session.redraw()
        assert session.error is None
        joined = " | ".join(c.value for c in _texts(control) if c.value)
        assert "no trials yet" in joined, "an explicit empty state, not a blank strip"
        assert "settling" in joined
        assert "across" in joined and "full scale" in joined, "every graph states its window and scale"
    finally:
        session.stop()


def test_every_tab_draws_with_trials_running_through_it():
    """The one thing an object test cannot do: produce a frame. Panes are held in a `Stack` with `visible`
    toggled, so the draw loop keeps writing to the hidden ones and a stale name in any of them throws."""
    page, session = _page(), _session()
    try:
        control = scp_view.view(page, session, _toolbar(), lambda: None)
        session.toggle_pause()
        _run(session, n_trials=6)
        _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 60.0, uv=_compliant, start=int(5 * RATE))
        panes = control.controls[2].controls
        for i in range(len(panes)):
            for j, pane in enumerate(panes):
                pane.visible = i == j
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
        from test_wave import _clickables

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


def test_the_catalogue_offers_it_and_a_screen_handles_it():
    from nfb_console.app import Console

    protocol = BY_CODE["SCP"]
    assert protocol.available and not protocol.note
    assert Console.screen_for(protocol) is scp_view


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
    """The shared trial piece brings settings the published protocol does not use. Each starts where that
    protocol is: both directions in a balanced order, the baseline at the cue, a third of the trials without
    a display, feedback for the whole window, live, full at 10 µV, any shift succeeds. The one departure is
    that the direction is random, hidden until the feedback, so the baseline cannot be pushed the other way.
    The choices are the four every trial program offers, and nothing more."""
    from nfb_console.trial_training import DIRECTIONS

    session = _session()
    try:
        assert list(session.DIRECTIONS) == ["increase", "decrease", "alternating", "random"]
        assert session.DIRECTIONS is DIRECTIONS and "shuffled" not in session.DIRECTIONS
        assert session.trial_direction == "random" and session.hides_direction
        assert session.transfer_fraction == pytest.approx(1 / 3)
        assert session.feedback_window_s == session.TIMING.active_s and session.update_s == 0.0
        assert (session.full_at, session.success_at) == (10.0, 0.0)
        run = session.trials
        assert set(run.directions) == {POSITIVE, NEGATIVE} and not run.alternate
        assert run.baseline_s == 2.0 and run.active_s == 6.0 and run.cue_s == 0.0
    finally:
        session.stop()


def test_the_cue_tones_are_an_octave_apart_and_a_success_doubles_its_tone():
    """620 Hz for up and 310 Hz for down, told apart with the eyes closed; the verdict on a success is the
    cue tone held twice, and a failure is silent."""
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
        expected = {cues.tone_at(620.0), cues.tone_at(310.0),
                    cues.held(cues.tone_at(620.0)), cues.held(cues.tone_at(310.0))}
        assert heard and {sound for sound, _ in heard} <= expected
        assert all(volume == 25 for _, volume in heard)
        assert any(sound in (cues.held(cues.tone_at(620.0)), cues.held(cues.tone_at(310.0)))
                   for sound, _ in heard), "a compliant trial succeeds and doubles its tone"
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


def test_no_trial_starts_while_the_filter_settles():
    """The run's first reading, which starts the lead-in to the first cue, waits for the metric to settle,
    so no trial is cued, baselined or scored on a filter that is still a transient."""
    session = _session()
    try:
        session.toggle_pause()
        _drive(session, 1.0, uv=lambda _s: 5.0)
        assert not session.settled and not session.trials.started
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
        assert "music" in session.offers
        assert set(session.music_sources) == {"tone", "crossfade", "noise", "shepard", "file"}
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
    the arrows and no circle."""
    import flet.canvas as cv

    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        run = _run(session, n_trials=3, transfer=1.0)
        seen: list = []

        def watch() -> None:
            if run.phase == ACTIVE:
                ui.draw_fixation("")
                seen.append(list(ui.circle.shapes))

        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 25.0, uv=_compliant, start=nxt, watch=watch)
        assert seen, "a feedback window ran"
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
    """The settling filter holds the run, and the end of the run says where its restart button is."""
    session = _session()
    try:
        ui = _ring_view(session)
        session.toggle_pause()
        ui.draw_fixation("")
        assert "the filter is settling" in ui.phase_text.value
        run = _run(session, n_trials=1)
        nxt = _drive(session, 5.0, uv=lambda _s: 0.0)
        _drive(session, 20.0, uv=_compliant, start=nxt)
        assert run.done
        ui.draw_fixation("")
        assert ui.phase_text.value == "the run is over · restart it in the training panel"
    finally:
        session.stop()
