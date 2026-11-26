"""Where the HRV level comes from: a metric, and a scaling.

Two layers: a metric and a scaling. A third — a set of "channels" rendering the result — was built
and deleted, because it reimplemented the mechanisms this codebase already shares across every protocol: a
second `DisplayShadow`, a noise mask that is `music` on noise plus `sound.invert`, and no master volume at all.
The mechanisms are the general, calibrated, persisted machinery; what was missing was the layer *underneath*
them, which on an EEG screen is a threshold and here cannot be.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from nfb.hrv import Beat, DEFAULT_METRIC, HRV_METRICS, metric_value
from nfb_console.protocols import BY_CODE

from harness import StubPage, _texts, coherent_state, toolbar


# --- and what carries it: the shared mechanisms, not a parallel set ----------------------------


def test_the_pitch_source_gets_its_own_rows_and_not_the_file_browser():
    """Its two ends are a real choice, because unlike the Shepard wheel it is bounded — and a source with no
    branch of its own falls through to the file browser and offers to play an mp3, which is what the Shepard
    music did before it got one."""
    import flet as ft
    import sys
    sys.path.insert(0, "tests")
    from harness import StubPage

    from nfb_console import signal_pane

    session = _session()
    try:
        pane = signal_pane.SignalPane(StubPage(), session)
        pane._select(session.offers.index("music"))
        pane.control.visible = True

        def visible(x, d: int = 0) -> str:
            if x is None or d > 30 or getattr(x, "visible", True) is False:
                return ""
            if isinstance(x, ft.Text):
                return x.value or ""
            out = [visible(c, d + 1) for c in (getattr(x, "controls", None) or [])]
            content = getattr(x, "content", None)
            if content is not None:
                out.append(visible(content, d + 1))
            return " ".join(v for v in out if v)

        shown = visible(pane.control)
        assert "pitch" in shown and "glide" in shown
        assert "file or folder" not in shown and "browse" not in shown
    finally:
        session.stop()


# --- end to end on the screen -----------------------------------------------------------------

def _session():
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    return HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))


def _feed(session, seconds: float = 250.0, depth: float = 0.09):
    """Run a coherent tachogram through the session with its clock advancing.

    The clock has to advance or the rolling strategies see every value at one instant, never warm up, and
    report the middle for ever — which is how a first check of this looked like it was working and was not.
    """
    t, clock = 1000.0, 0.0
    session._resumed_at = None
    while clock < seconds:
        rr = 1.0 + depth * np.sin(2 * np.pi * 0.1 * t)
        t += rr
        clock += rr
        session._accumulated = clock
        session.heart.state = session.heart.processor.push(Beat(t, 60.0 / rr, [rr]))
        session.observe()


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
        assert session.level > 0.5, "a clean 0.1 Hz peak should read high"
        assert session.strategy.warmup > 0.9
        outputs = session.feedback.outputs_for(session.level, True)
        assert outputs["sound"] > session.feedback.sound.low
        assert outputs["shadow"] > session.feedback.shadow.low
    finally:
        session.stop()


def test_the_level_comes_from_the_scaling_and_not_from_a_threshold():
    """The one target on this screen exists for the pane's contribution row and for the click to ask; it is
    fed the *already normalised* signal with its threshold at the middle of that range, so nothing is
    calibrated twice and the level is the scaling's own output."""
    from nfb_console.hrv_view import CONDITION_LEVEL

    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        assert len(session.targets) == 1
        target = session.targets[0]
        assert target.band is None, "it measures no band, so it gets no card and no filter gear"
        assert target.threshold == CONDITION_LEVEL == 0.5
        assert not target.calibrating, "the scaling already did that job"
        assert target.value == pytest.approx(session.signal)
        # the level is the signal, smoothed — not a logistic of it
        assert session.level == pytest.approx(session.signal, abs=0.35)
    finally:
        session.stop()


def test_changing_the_metric_rebuilds_the_scaling_and_refills_it_from_the_beats_in_hand():
    """A rolling window holds values in the *old* metric's units. Carrying it across would normalise
    milliseconds against a history of dimensionless ratios, and the signal would sit pinned at an end until
    the window aged out — so it is rebuilt.

    **Rebuilt is not the same as empty.** An empty rolling scale is gated until a quarter of its window
    exists, which meant changing the metric stopped the feedback for thirty seconds; and there was never
    anything to wait for, since the beats are still in hand and a metric is a pure function of a state.
    The new scale is handed the values it would have had if it had been running all along.
    """
    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        assert session.strategy.warmup > 0.9
        before = session.strategy
        session.set_metric("rmssd")
        assert session.strategy is not before, "the old window was in the old metric's units"
        assert session.strategy.warmup > 0.9, "and the new one arrives full, from the same beats"

        # The values in it are the new metric's, not the old one's: RMSSD is in milliseconds and runs to
        # tens, a coherence score does not.
        held = [v for _t, v in session.strategy._buf]
        assert held and max(held) > 5.0, f"the window holds {max(held):.3g}, which is not milliseconds"
        session.observe()
        assert session.signal is not None, "so the feedback continues rather than stopping"
        # and the strategy follows the metric's own recommendation when the current one does not suit it
        session.set_strategy("fixed_range", low=0.0, high=1.0)
        session.set_metric("coherence")
        assert session.strategy_id in HRV_METRICS["coherence"].strategies
    finally:
        session.stop()


def test_the_direction_is_the_operators_and_every_metric_arrives_with_one():
    """**Which way you are training it is a setting, not a property of the number.**

    It was `MetricSpec.invert` and nothing else — unreachable from the screen, so the most obvious thing an
    operator wants to say about a quantity could not be said. LF/HF is the case that makes it plain: inverted
    in the catalogue because sympathetic drive is the usual concern, and exactly backwards for somebody
    training a flat, over-parasympathetic profile. The metric supplies the default, the way a protocol's bands
    arrive as increase or decrease on the amplitude screen and can be cycled there.
    """
    session = _session()
    try:
        # every metric ships an aim, so there is never a direction nobody chose
        for spec in HRV_METRICS.values():
            assert spec.default_direction in ("increase", "decrease")

        session.set_metric("coherence")
        session.toggle_pause()
        assert session.direction == "increase" and not session.inverts
        _feed(session)
        up = session.signal
        assert up > 0.5
        session.set_direction("increase")
        assert session.signal == up, "the aim that is already chosen changes nothing"

        # flipping it inverts the level and nothing else: the scaling keeps its window and its warm-up
        window = session.strategy
        session.set_direction("decrease")
        assert session.strategy is window, "the direction is not a rescaling"
        assert session.signal is None, "and the level it already reported was under the other aim"

        # the same coherent tachogram, trained the other way, reads low
        other = _session()
        try:
            other.toggle_pause()
            other.set_direction("decrease")
            _feed(other)
            assert other.inverts
            assert other.signal == pytest.approx(1.0 - up, abs=0.05)
            assert other.signal < 0.5
        finally:
            other.stop()

        # changing the metric puts the aim back to that metric's own — carrying "decrease" onto RMSSD would
        # train somebody toward *less* variability with only the metric's name having changed
        ups = [m for m, spec in HRV_METRICS.items() if spec.default_direction == "increase"]
        downs = [m for m, spec in HRV_METRICS.items() if spec.default_direction == "decrease"]
        assert ups and downs
        session.set_metric(ups[-1])
        assert session.direction == "increase"
        session.set_metric(downs[0])
        assert session.direction == "decrease", "and it takes the inverted default where there is one"

        # it survives a stop, like the metric and the scaling
        session.set_direction("increase")
        second = _session()
        try:
            second.adopt(session)
            assert second.direction == "increase"
            assert second.metric == downs[0]
        finally:
            second.stop()
    finally:
        session.stop()


def test_the_screen_puts_the_direction_in_its_own_row_and_says_what_is_usual():
    import flet as ft
    import sys
    sys.path.insert(0, "tests")
    from harness import StubPage

    from nfb_console import hrv_view

    def visible(x, d: int = 0) -> str:
        if x is None or d > 30 or getattr(x, "visible", True) is False:
            return ""
        if isinstance(x, ft.Text):
            return x.value or ""
        out = [visible(c, d + 1) for c in (getattr(x, "controls", None) or [])]
        content = getattr(x, "content", None)
        if content is not None:
            out.append(visible(content, d + 1))
        return " ".join(v for v in out if v)

    session = _session()
    try:
        source = hrv_view.SignalSource(StubPage(), session)
        usual = visible(source.control)
        assert "aim to" in usual and "increase" in usual and "decrease" in usual

        def chosen():
            from nfb_console import theme

            return [k for k in ("increase", "decrease") if _by_label(source.control, k).content.color == theme.TEXT]

        def note():
            """The line under the pills: the text that names the metric."""
            label = session.metric_spec.label
            return next(x.value for x in walk(source.control)
                        if isinstance(x, ft.Text) and label in (x.value or "") and "usual" in x.value)

        usual_aim = session.metric_spec.default_direction
        assert chosen() == [session.direction] == [usual_aim]
        assert usual_aim not in note(), "at the usual aim the note names no other"

        other = "decrease" if usual_aim == "increase" else "increase"
        session.set_direction(other)
        source.rebuild()
        off = visible(source.control)
        assert chosen() == [other]
        assert off != usual and usual_aim in note(), \
            "off the usual aim, the row names the usual one rather than looking like the default"
    finally:
        session.stop()


def test_an_inverted_metric_inverts_the_level_and_not_the_scaling():
    """The inversion is applied after normalisation, so a rolling *minimum* stays a minimum and the
    description on the pane keeps telling the truth about what the window holds."""
    session = _session()
    try:
        session.toggle_pause()
        session.set_metric("spectrum_distance")         # arrives set to decrease
        session.set_strategy("fixed_range", low=0.0, high=1.0)
        _feed(session, seconds=200.0)
        raw, signal = session.raw_metric, session.signal
        assert raw is not None and raw < 0.5, "a coherent spectrum is close to the ideal"
        assert signal == pytest.approx(1.0 - raw, abs=1e-6), "inverted, so close-to-ideal reads high"
    finally:
        session.stop()


def test_the_screen_has_one_feedback_tab_carrying_the_shared_mechanisms():
    import sys
    sys.path.insert(0, "tests")
    from harness import StubPage, toolbar

    import flet as ft

    from nfb_console import hrv_view

    def visible(x, d: int = 0) -> str:
        if x is None or d > 20 or getattr(x, "visible", True) is False:
            return ""
        if isinstance(x, ft.Text):
            return x.value or ""
        out = [visible(c, d + 1) for c in (getattr(x, "controls", None) or [])]
        content = getattr(x, "content", None)
        if content is not None:
            out.append(visible(content, d + 1))
        return " ".join(v for v in out if v)

    page = StubPage()
    session = _session()
    try:
        control = hrv_view.view(page, session, toolbar(), lambda: None)
        session.toggle_pause()
        _feed(session)
        session.redraw()
        panes = control.controls[-1].controls
        assert len(panes) == 3, "Train, Feedback, Session"
        for i, pane in enumerate(panes):
            pane.visible = (i == 1)
        shown = visible(control)
        # the two rows that are ours
        assert "SIGNAL SOURCE" in shown
        assert "feed back" in shown and "scaled by" in shown
        # and the shared mechanisms below them
        assert shown.split().count("OUTPUT") == 1, "one panel of the mechanisms, the one every protocol uses"
        for mechanism in ("sound", "shadow", "click", "music"):
            assert mechanism in shown.lower(), f"{mechanism} is configurable here"
        # and none of the panels or rows that belong to a level built by scoring bands: absent rather than
        # emptied, which is what `conditions_signal` and `scores_bands` are for
        for gone in ("ARTIFACTS", "rejection", "band averaging", "2 · scale from", "how far counts",
                     "3 · level from", "4 · condition"):
            assert gone not in shown, f"{gone!r} has nothing to act on here"
        assert "output smoothing" in shown, "the one step that applies to any level"
        # and no second set of channels
        assert "ambient tint" not in shown and "the room" not in shown
    finally:
        session.stop()


def test_the_toolbar_lifecycle_exists_and_pausing_hands_the_room_back():
    """The device chip on the shared toolbar calls suspend()/suspended/rearm() on whatever session is
    up; this session not having them was an AttributeError inside a Flet event handler — a click that
    silently did nothing. And toggle_pause only flipped a flag, which stops the outputs being *updated*
    rather than being *held*: the display stayed dimmed and the music sang on, the exact bug
    `TrainingSession.toggle_pause` documents having been fixed on every other screen."""
    session = _session()
    try:
        released = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "release", (lambda n=name: lambda: released.append(n))())

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
        released = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "release", (lambda n=name: lambda: released.append(n))())
        clock = [100.0]
        session.now = lambda: clock[0]
        session.plan.length_seconds = 10.0
        session.toggle_pause()               # start the clock
        clock[0] += 10.5
        session.tick()
        assert session.finished and session.paused
        assert released == ["display", "output_volume", "music"], \
            "a timed session that ends must not leave the room held"
    finally:
        session.stop()


def test_a_dead_or_silent_beat_source_stops_the_feedback_and_says_so():
    """Two flavours, both of which used to run the room on a frozen snapshot: a source that raises sets
    an error but froze `state` with no gate downstream, and a source whose generator simply *ends* — how
    BleBeats reports its worker dying — set no error at all. Now an unrequested end is a fault, and
    observe() refuses to feed outputs from an errored or stale runner."""
    import time as _time

    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    class Ends:
        def beats(self):
            yield Beat(1000.0, 60.0, [1.0])
            yield Beat(1001.0, 60.0, [1.0])

        def close(self):
            pass

        def describe(self):
            return "test source"

    session = _session()
    try:
        runner = session.heart
        runner.source = Ends()
        runner._stop.clear()
        # One attempt, not the retry loop the reader thread runs: what is under test is that an ending
        # nobody asked for is reported as a fault rather than passing in silence.
        runner.error = runner._read(runner._generation)   # the generator ends on its own
        assert runner.error, "an end nobody asked for is a fault, not a silence"
        runner.state = coherent_state()                   # the beats held before the end are plenty
        runner.seen_at = time.monotonic()                 # and the last one is fresh

        applies = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "apply", (lambda n=name: lambda *a, **kw: applies.append(n))())
            setattr(obj, "release", lambda: None)
        session.toggle_pause()
        session.observe()
        assert session.signal is None and applies == [], \
            "an errored source must not keep driving the outputs"
    finally:
        session.stop()


def test_a_paused_scale_does_not_grow_and_warms_on_resume():
    """A paused or pre-start session has a frozen clock, so feeding the rolling scale piles every tick at
    one instant: the window never evicts, grows without bound, and its warm-up can never advance. The scale
    is now fed only while running, and warmed from the buffered beats on resume — which is where the
    'warm before you start' the screen promises actually happens."""
    session = _session()
    try:
        session.set_strategy("rolling_minmax")          # a strategy with a rolling buffer to watch
        # Beats accrue in the heart while the operator sets up, before play is pressed.
        t = 1000.0
        for _ in range(120):
            t += 1.0
            session.heart.state = session.heart.processor.push(Beat(t, 60.0, [1.0]))
        session._strategy.reset()
        assert session.paused, "a session opens paused"

        for _ in range(200):
            session.observe()
        assert len(session._strategy._buf) == 0, "a paused scale is not fed, so it cannot grow unbounded"

        session.toggle_pause()                          # resume: the clock advances again
        assert len(session._strategy._buf) > 0, "resume warms the scale from the buffered beats"
        assert session._strategy.warmup > 0.0, "the warmed scale spans real time, so warm-up advances"
    finally:
        session.stop()


def test_a_strap_that_loses_contact_goes_stale_rather_than_running_the_room():
    """A Bluetooth strap keeps sending notifications after skin contact drops, with the contact bit false.
    The processor discards their intervals, so no new usable data arrives — but `seen_at` used to be
    refreshed on every notification, so `stale` never fired and the room ran on the frozen snapshot. Now
    only a beat with usable data refreshes it, so a run of no-contact beats lets the runner go stale."""
    from nfb_console.heart import HeartRunner

    class Source:
        def __init__(self, beats):
            self._beats = beats

        def beats(self):
            yield from self._beats

        def close(self):
            pass

        def describe(self):
            return "test strap"

    runner = HeartRunner()
    runner._stop.clear()
    # A good beat is news; its arrival time is recorded.
    runner.source = Source([Beat(1000.0, 60.0, [1.0], contact=True)])
    runner._read(runner._generation)
    assert runner.seen_at is not None, "a beat with a real interval is fresh news"

    # A run of no-contact beats carries no usable interval, so it must not refresh the freshness clock.
    was = runner.seen_at
    runner.source = Source([Beat(1001.0 + i, 60.0, [], contact=False) for i in range(5)])
    runner._read(runner._generation)
    assert runner.seen_at == was, "no-contact beats are not news and must not hold off staleness"


def test_the_invented_simulator_is_handed_back_switched_off():
    """The simulator default is this screen's, not the app's: the runner is shared with CFC, and one
    visit to HRV used to leave the app-wide choice pointing at the simulator — CFC then drew invented
    beats, and could raise the split warning on them, beside a real EEG. Stopping now restores the
    runner to off with the simulated state cleared. A choice the operator actually made is kept."""
    from nfb_console.inputs import HeartChoice

    session = _session()                     # _session passes an explicit simulator choice
    assert not session._invented, "an explicit choice is the operator's and is kept"

    from nfb_console.hrv_view import HRVSession
    from nfb_console.protocols import BY_CODE

    invented = HRVSession(BY_CODE["HRV"], choice=None)
    try:
        assert invented._invented and invented.choice.kind == "simulator"
        invented.heart.state = object()      # pretend the simulator ran
        invented.heart.error = "leftover"
    finally:
        invented.stop()
    assert invented.heart.choice.kind == "off", "the shared runner goes back the way it was found"
    assert invented.heart.state is None and invented.heart.error is None


def test_a_fresh_source_starts_a_fresh_record(monkeypatch):
    """attach() resets the processor: a session started within the buffer's three minutes of the last
    used to inherit its beats, its pre-satisfied warm-up, and a pending beat from another session — or
    another device, since attach is also how simulator switches to strap.

    The reset is `attach`'s own work, done before it starts the reader, so it is checked the moment it
    returns rather than by sleeping until beats have arrived through a thread."""
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    # A source that ends at once, so the reader thread neither sleeps between beats nor has to be waited
    # for: what is under test is the reset, and a real simulator's 0.9 s beat interval was the cost of
    # releasing it afterwards.
    class Empty:
        def beats(self):
            return iter(())

        def close(self):
            pass

        def describe(self):
            return "nothing"

    monkeypatch.setattr("nfb.hrv.beat_source", lambda kind, **kw: Empty())
    runner = HeartRunner(HeartChoice(kind="simulator"))
    for i in range(10):
        runner.processor.push(Beat(1000.0 + i, 60.0, [1.0]))
    runner.state = object()
    assert runner.processor.replay(step_s=1.0), "old beats are in the buffers"
    assert runner.attach()
    try:
        # The source gives no beat, so whatever the processor still holds came from before the attach.
        assert runner.processor.replay(step_s=1.0) == [], "old beats survived the attach"
    finally:
        runner.release()


def test_the_room_does_something_from_the_first_beats_rather_than_after_two_minutes():
    """**The opening minute fed back nothing at all**, in three successive versions of this screen: the
    spectrum refused to exist below eighty seconds, then twenty; the spectral metrics waited four cycles
    of the target; and the rolling scale wanted a quarter of its window on top. Every one of them bought
    accuracy with a minute in which somebody breathes and nothing answers.

    Now nothing waits. Fourteen beats is a real spectrum — coarse, and labelled coarse by the span and
    bin width under the curve — so the chosen metric has a value and the room runs on it."""
    from nfb.hrv import HRVProcessor
    from nfb.hrv.beats import TARGET_HZ
    from nfb_console.hrv_view import STANDIN_METRIC

    session = _session()
    try:
        session.toggle_pause()
        proc = HRVProcessor()
        t0 = 1000.0
        state = None
        for i in range(14):                    # fourteen beats: no spectrum, plenty for a swing
            rr = 1.0 + 0.12 * np.sin(2 * np.pi * 0.1 * (t0 + i))
            t0 += rr
            state = proc.push(Beat(t0, 60.0 / rr, [rr]))
        session.heart.state = state
        session.heart.seen_at = time.monotonic()
        # Fourteen beats cannot *resolve* 0.1 Hz — four cycles of it is forty seconds — and that is no
        # longer a reason to withhold the number. The spectrum exists, so the metric has a value.
        assert not state.spectrum.resolves(TARGET_HZ), "fourteen beats is a coarse window, and says so"
        assert state.spectrum.ready, "but it is a spectrum, and the transform is defined on it"

        session.observe()
        assert session.signal is not None, "and the room has a level to run on"
        assert 0.0 <= session.signal <= 1.0
        assert not session.standing_in, \
            f"the chosen metric speaks from the first spectrum, so {STANDIN_METRIC} is not needed"
        assert session.raw_metric == metric_value(session.metric_spec, state)
    finally:
        session.stop()


def test_the_stand_in_covers_a_metric_with_nothing_to_say_and_not_a_scale_that_is_still_settling():
    """**The stand-in's job shrank to the honest one.** It used to hold the room until the chosen
    metric's scale was a quarter full, which is a settling time, and the same trade this screen made
    three times over — an opening minute spent on accuracy nobody asked for. The chosen metric now
    drives from its first value, however rough, and the warm-up bar says it is still firming up.

    What remains is real: a metric with *no* value — the first seconds, before enough beats exist for a
    difference between beats — where the swing of the heart rate is the first thing that can be measured.
    """
    session = _session()
    try:
        session.toggle_pause()
        session.heart.state = coherent_state()          # a full spectrum: coherence has a value
        session.heart.seen_at = time.monotonic()

        session.observe()
        assert not session.standing_in, "one value is a value; the scale reports the middle and widens"
        assert session.signal is not None
        assert len(session.strategy._buf) == 1, "and the scale is being fed the whole time"
        assert session.raw_metric == metric_value(session.metric_spec, session.heart.state)

        # A window's worth of readings later, nothing about who is driving has changed.
        _feed(session)
        assert not session.standing_in
        assert session.raw_metric == metric_value(session.metric_spec, session.heart.state)
        assert session.signal is not None, "and the level never stopped on the way"
    finally:
        session.stop()


def test_the_level_keeps_running_through_a_metric_change_and_is_warm_when_you_press_play():
    """Two delays that were never necessary, both from a scale that starts empty.

    Changing the metric rebuilds the normaliser — it has to, the units differ — and the rebuilt one was
    empty, so the feedback stopped for a quarter of its window. And the scale was only fed while the
    session ran, so pressing play began a further thirty seconds of silence. Both are answered from the
    beats already in hand: the metric is a pure function of a state, and the states can be recomputed.
    """
    session = _session()
    try:
        session.toggle_pause()
        _feed(session)
        session.toggle_pause()                       # paused again, beats still arriving
        assert session.strategy.warmup > 0.9

        before = session.signal
        session.set_metric("rmssd")
        session.observe()
        assert session.signal is not None and before is not None, "the level did not stop"

        # And while paused the scale keeps learning, so play starts on a warm one rather than on nothing.
        session.set_strategy("rolling_minmax")
        assert session.strategy.warmup > 0.9, "refilled from history the moment it was built"
        assert not session.running
        session.observe()
        assert session.raw_metric is not None, "measured while paused"
    finally:
        session.stop()


def test_measuring_while_paused_never_touches_the_room():
    """The reason the measurement was gated on running in the first place, and it is still the rule —
    just applied to the one place that drives anything instead of to the whole path."""
    session = _session()
    try:
        _feed(session, seconds=60.0)
        applied: list[str] = []
        for name in ("display", "output_volume", "music"):
            obj = getattr(session, name)
            setattr(obj, "apply", (lambda n=name: lambda *a, **kw: applied.append(n))())
            setattr(obj, "release", lambda: None)
        assert not session.running
        for _ in range(5):
            session.observe()
        assert applied == [], "a paused session measures and drives nothing"
        assert session.raw_metric is not None
    finally:
        session.stop()


def test_a_missing_level_says_why_instead_of_blaming_a_channel_that_does_not_exist():
    """**"no channel is delivering signal", in red, while the beats were arriving and visible.** The
    shared pane's only vocabulary for an unavailable level was about amplifier channels, and on this
    screen there are none.

    The situation it used to describe — a scale that had not filled a quarter of its window — is gone,
    because nothing waits for that any more. What is left is the honest case: not enough beats yet for a
    difference between beats to exist. The wording is checked here; the counting is checked below."""
    from nfb.hrv import Beat, HRVProcessor
    from nfb_console import signal_pane

    session = _session()
    try:
        session.toggle_pause()
        # Three beats: the source is plainly alive, and the swing still needs a fourth to exist.
        proc = HRVProcessor()
        state = None
        for i in range(3):
            state = proc.push(Beat(1000.0 + i, 60.0, [1.0]))
        session.set_metric("rsa_amplitude")
        session.heart.state = state
        session.heart.seen_at = time.monotonic()
        session.observe()

        assert session.dead_channels() == [], "the source is fine and must not be blamed"
        holding = signal_pane._holding(session)
        assert holding is not None
        headline, note = holding
        assert "counting beats" in headline
        assert "channel" not in headline and "channel" not in note

        # and a barely-started rolling scale is *not* one of these situations any more
        _feed(session, seconds=20.0)
        session.set_strategy("rolling_minmax", window_s=600.0)
        session._strategy.reset()                      # as if freshly built with nothing behind it
        session.observe()
        assert signal_pane._holding(session) is None, \
            "an empty scale reports the middle and widens; it does not withhold the level"
        assert session.signal is not None
    finally:
        session.stop()


def test_the_opening_seconds_say_what_they_are_counting():
    """"waiting for data", on a screen whose chip reads "reading the pulse waveform" two inches above,
    is a screen contradicting itself. Variability cannot be measured without beats — four of them, which
    is the arithmetic floor for a swing and not a settling time — but the wait can at least be counted
    out."""
    from nfb.hrv import Beat, HRVProcessor
    from nfb.hrv.metrics import RSA_MIN_BEATS
    from nfb_console import signal_pane

    session = _session()
    try:
        session.toggle_pause()
        proc = HRVProcessor()
        t = 1000.0
        for i in range(3):                       # three beats: not yet a swing
            t += 1.0
            session.heart.state = proc.push(Beat(t, 60.0, [1.0]))
        session.heart.seen_at = time.monotonic()
        session.observe()

        assert session.signal is None, "three beats is not a measurement of variability"
        holding = signal_pane._holding(session)
        assert holding is not None
        assert f"of {RSA_MIN_BEATS}" in holding[0], holding
        assert "waiting for data" not in holding[1]

        for i in range(4):                       # five in total, and the level starts
            t += 1.0
            session.heart.state = proc.push(Beat(t, 60.0 + 4.0 * (i % 2), [1.0]))
        session.observe()
        assert session.signal is not None, "the level runs from the first beats that can carry it"
    finally:
        session.stop()


def test_the_hrv_screen_draws_before_and_after_beats():
    from nfb_console import hrv_view
    from nfb.hrv import SimulatedBeats

    page = StubPage()
    session = hrv_view.HRVSession(BY_CODE["HRV"])
    try:
        control = hrv_view.view(page, session, toolbar(), lambda: None)
        session.redraw()                        # no beats at all
        assert session.error is None

        # **The beats are pushed, not awaited.** This used to assign `session.source`, which `HRVSession`
        # does not have and nothing reads, then start `run()` on a thread — so it opened the invented
        # *realtime* simulator and slept through it, and its duration tracked whatever the spectrum
        # happened to need that week. The processor is the thing under the screen; hand it beats.
        source = SimulatedBeats(rsa_hz=0.1, seed=11, realtime=False)
        beats = source.beats()
        while not (session.state is not None and session.state.spectrum.ready):
            session.heart.state = session.heart.processor.push(next(beats))
        source.close()
        session.heart.seen_at = time.monotonic()

        session.redraw()
        assert session.error is None, session.error
        captions = " | ".join(c.value for c in _texts(control))
        assert "0.1 Hz" in captions
        assert "bpm" in captions
        # the ring around the pacer is the level, and BREATHE names the metric and the number it has
        assert session.signal is not None
        labels = [spec.label for spec in HRV_METRICS.values()]
        assert any(f"{session.signal:.0%}" in t.value and any(label in t.value for label in labels)
                   for t in _texts(_breathe(control)) if t.value)
    finally:
        session.stop()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_pacer_period_that_is_not_finite_is_ignored_and_a_far_one_is_clamped(bad):
    """A cycle that is not a number leaves the cycle where it was and does not raise; a finite one far
    out of range goes to the nearest end."""
    from nfb_console.hrv_view import PACER_RANGE_S

    session = _session()
    try:
        session.set_pacer_period(12.0)
        session.set_pacer_period(bad)
        assert session.pacer_period == 12.0
        session.set_pacer_period(1e-9)
        assert session.pacer_period == PACER_RANGE_S[0]
        session.set_pacer_period(1e9)
        assert session.pacer_period == PACER_RANGE_S[1]
    finally:
        session.stop()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_target_that_is_not_finite_leaves_the_cycle_where_it_was(bad):
    """A target that is not a number changes neither the target nor the cycle that follows it."""
    session = _session()
    try:
        session.set_target_hz(0.075)
        session.set_pacer_period(20.0)
        before = (session.target_hz, session.pacer_period)
        session.set_target_hz(bad)
        assert (session.target_hz, session.pacer_period) == before
    finally:
        session.stop()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_saved_setting_that_is_not_finite_changes_nothing(bad):
    """A stored number that is not finite, for any setting that the session keeps, does not raise and
    leaves every kept setting as it was."""
    session = _session()
    try:
        session.set_target_hz(0.075, 0.015)
        before = session.calibration()
        numbers = [key for key, value in before.items() if isinstance(value, float)]
        assert {"target_hz", "target_half_hz", "pacer_period"} <= set(numbers)
        session.restore_calibration({key: bad for key in numbers})
        assert session.calibration() == before
    finally:
        session.stop()


@pytest.mark.parametrize("huge", [10 ** 400, -10 ** 400], ids=["large", "small"])
def test_a_saved_integer_too_large_for_a_float_does_not_stop_the_screen_from_opening(huge):
    """A preferences file that holds an integer with hundreds of digits, in any setting that the console
    restores, leaves the settings finite and in range and does not raise."""
    import math

    from nfb_console import prefs

    session = _session()
    try:
        numbers = ["target_hz", "target_half_hz", "pacer_period", "pacer_inhale"]
        session.restore_calibration({key: huge for key in numbers})
        assert all(math.isfinite(v) for v in session.calibration().values() if isinstance(v, float))
        prefs.apply(session.feedback, {f"{owner}.{attr}": huge for owner, attr in prefs.CALIBRATED})
    finally:
        session.stop()


@pytest.mark.parametrize("far", [1e-9, 1e9])
def test_a_saved_target_and_cycle_far_out_of_range_are_clamped_into_it(far):
    """A stored target or cycle that is finite and far outside the range restores to the nearest end of
    its range, and everything that the session keeps is finite."""
    import math

    from nfb.hrv import NARROW_HALF_RANGE_HZ, TARGET_RANGE_HZ
    from nfb_console.hrv_view import PACER_RANGE_S

    session = _session()
    try:
        session.restore_calibration({"target_hz": far, "target_half_hz": far, "pacer_period": far})
        kept = session.calibration()
        assert TARGET_RANGE_HZ[0] <= kept["target_hz"] <= TARGET_RANGE_HZ[1]
        assert NARROW_HALF_RANGE_HZ[0] <= kept["target_half_hz"] <= NARROW_HALF_RANGE_HZ[1]
        assert PACER_RANGE_S[0] <= kept["pacer_period"] <= PACER_RANGE_S[1]
        assert all(math.isfinite(v) for v in kept.values() if isinstance(v, float))
    finally:
        session.stop()


def test_the_session_clock_is_the_one_the_session_is_given():
    """Elapsed time counts on `now` alone: it grows while the session runs, stands still while it is
    paused, and takes nothing from the wall clock."""
    session = _session()
    clock = [100.0]
    session.now = lambda: clock[0]
    try:
        assert session.elapsed == 0.0
        session.toggle_pause()
        clock[0] += 7.5
        assert session.elapsed == pytest.approx(7.5)
        session.toggle_pause()
        clock[0] += 30.0
        assert session.elapsed == pytest.approx(7.5)
        session.toggle_pause()
        clock[0] += 2.0
        assert session.elapsed == pytest.approx(9.5)
    finally:
        session.stop()


def test_the_heart_source_key_changes_when_another_choice_is_attached(monkeypatch):
    """The key stays the same while one reader runs and the same choice is attached again, and it
    differs after a different choice replaces that reader."""
    import threading

    from conftest import waitfor
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    class Quiet:
        def __init__(self):
            self.closed = threading.Event()

        def beats(self):
            reading.append(self)
            self.closed.wait()
            return iter(())

        def close(self):
            self.closed.set()

        def describe(self):
            return "quiet"

    reading = []
    monkeypatch.setattr("nfb.hrv.beat_source", lambda kind, **kw: Quiet())
    runner = HeartRunner()
    before = runner.source_key
    try:
        assert runner.attach(HeartChoice(kind="simulator"))
        waitfor(lambda: len(reading) == 1, what="the reader to start")
        first = runner.source_key
        assert first != before
        assert runner.attach(HeartChoice(kind="simulator"))
        assert runner.source_key == first
        assert runner.attach(HeartChoice(kind="usb", port="/dev/none"))
        assert runner.source_key != first
        waitfor(lambda: len(reading) == 2, what="the other reader to start")
        second = runner.source_key
        runner.choice.port = "/dev/other"
        assert runner.source_key != second, "a choice edited in place names another device at once"
        third = runner.source_key
        runner.release()
        assert runner.attach()
        waitfor(lambda: runner.source_key != third, what="a new reader of the same device to be a new source")
    finally:
        runner.release()


# === the sliding rate search on the screen ====================================================
#
# The pure series is tested in `test_sweep.py`. These drive the session by its own tick with a clock the
# test owns, and read the panel only where the fact is a panel's.

import functools
import math
import re

import flet as ft
import flet.canvas as cv

from conftest import SWEEP_MONO0, beat_state, pacer_beats
from harness import _by_label
from test_layout import RULES, walk
from test_ste import _help_faults, faults as ste_faults


def _peak(at, base=20.0, top=80.0, width=0.35):
    return lambda rate: base + (top - base) * math.exp(-0.5 * ((rate - at) / width) ** 2)


_PLANTS = {                     # name -> (swing at a rate, noise ms, interval ms, interval at the end)
    "peak": (_peak(5.4), 0.0, 900.0, None),
    "fast_peak": (_peak(6.7), 0.0, 900.0, None),
    "flat": (lambda rate: 20.0, 25.0, 900.0, None),
    "slow_heart": (lambda rate: 40.0, 0.0, 2600.0, None),
    "falling_heart": (_peak(5.4), 4.0, 750.0, 1090.0),
    "slow_peak": (_peak(4.3), 0.0, 900.0, None),
    "resonant": (None, 0.0, 0.0, None),
}


@functools.lru_cache(maxsize=None)
def _plant(name: str, seed: int = 0, rate_range=None):
    from nfb.hrv.sweep import SweepSchedule

    swing, noise, mean, mean_end = _PLANTS[name]
    schedule = SweepSchedule(*rate_range) if rate_range else SweepSchedule()
    if name == "resonant":      # a heart with a resonance: in step at 5.4 breaths a minute
        from resonant_heart import sweep_beats

        return sweep_beats(5.4, seed, schedule, noise_ms=12.0)
    stamps, rr, _axis = pacer_beats(schedule, swing, noise_ms=noise, mean_ms=mean, mean_end_ms=mean_end,
                                    seed=seed)
    return stamps, rr


class Rig:
    """A session on a clock that the test owns, beside a heart that breathes with the pacer.

    `look(t)` is one draw at second `t` of the true axis: it sets the clock and the beats that exist then
    and runs `tick`, the first thing a draw does. The sweep begins at `t` = 0."""

    def __init__(self, kind="strap", plant="peak", seed=0, length_seconds=None, quiet_from=None,
                 hole=None, rate_range=None):
        from nfb_console.hrv_view import HRVSession
        from nfb_console.inputs import HeartChoice

        self.session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind=kind),
                                  length_seconds=length_seconds)
        self.t = 0.0
        self.skew = 0.0
        self.frozen = False
        self.played: list[str] = []
        self.when: list[float] = []
        self.session.now = lambda: SWEEP_MONO0 + self.t
        self.session.cue = lambda name, *a, **k: (self.played.append(name), self.when.append(self.t))
        if rate_range is not None:
            self.session.set_sweep_range(*rate_range)
            assert (self.session.sweep_start, self.session.sweep_end) == rate_range
        self.stamps, self.rr = _plant(plant, seed, rate_range)
        if hole is not None:
            keep = (self.stamps < hole[0]) | (self.stamps > hole[1])
            self.stamps, self.rr = self.stamps[keep], self.rr[keep]
        self.sweep_started = False

    def look(self, t: float) -> None:
        self.t = t
        if not self.frozen:
            state = beat_state(self.stamps, self.rr, t)
            state.t += self.skew
            state.beats = len(state.tachogram)
            self.session.heart.state = state
        self.session.heart.seen_at = time.monotonic()
        self.session.tick()

    def begin(self) -> bool:
        """Choose the sweep on a session that runs, and look once: breath 1 starts with the first beats."""
        self.session.choose_pacer("sweep")
        if self.session.paused:
            self.session.toggle_pause()
        self.look(0.0)
        return self.session.sweeping

    def drive(self, until: float, step: float = 0.8, stop=None) -> None:
        while self.t + step <= until + 1e-9:
            self.look(self.t + step)
            if stop is not None and stop():
                return
            if self.session.sweep is not None and not self.session.sweep.active and stop is None:
                return

    @property
    def sweep(self):
        return self.session.sweep

    def close(self) -> None:
        self.session.stop()


@pytest.fixture
def rig():
    made = []

    def make(**kw):
        made.append(Rig(**kw))
        return made[-1]

    yield make
    for r in made:
        r.close()


def _finished(rig_, until=1400.0):
    assert rig_.begin()
    rig_.drive(until)
    return rig_


def test_the_metronome_period_is_configurable_within_its_range():
    """A cycle inside the range is kept as given and the note names it with its breaths a minute; a cycle
    outside the range is clamped to the nearest end."""
    from nfb_console.hrv_view import HRVSession, PACER_RANGE_S
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        session.set_pacer_period(12)
        assert session.pacer_period == 12.0
        assert "12 s" in session.pacer_note() and "5.0 breaths a minute" in session.pacer_note()
        assert "metronome" in session.pacer_note(), "it must not pass for the heart-locked cue"
        # clamped *into* the range rather than out of it
        session.set_pacer_period(1.0)
        assert session.pacer_period == PACER_RANGE_S[0]
        session.set_pacer_period(999.0)
        assert session.pacer_period == PACER_RANGE_S[1]
        # a cycle between two steps of the arrows runs as it was given, and the note names it
        session.set_pacer_period(10.37)
        assert session.pacer_period == 10.37
        assert "10.37 s" in session.pacer_note() and "5.8 breaths a minute" in session.pacer_note()
    finally:
        session.stop()


def test_the_target_frequency_is_settable_from_the_screen_and_takes_the_metronome_with_it():
    """Moving the target moves the metronome to the same frequency and the processor scores against it. A
    cycle set later does not move the target."""
    from nfb.hrv import TARGET_RANGE_HZ
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        session.set_target_hz(0.065)
        assert session.target_hz == 0.065
        assert session.heart.processor.target_hz == 0.065, "one copy, and it lives on the processor"
        assert session.pacer_period == pytest.approx(1 / 0.065), "the metronome breathes at the target"
        assert "3.9 breaths a minute" in session.target_note()

        # the band moves independently, and both are clamped
        session.set_target_hz(0.065, 0.01)
        assert session.target_half_hz == 0.01
        session.set_target_hz(0.0)
        assert session.target_hz == TARGET_RANGE_HZ[0], "clamped into the range, not refused"

        # nudging the cycle afterwards does not drag the target with it
        session.set_target_hz(0.065)
        session.set_pacer_period(10.0)
        assert session.target_hz == 0.065
    finally:
        session.stop()


def test_the_target_survives_a_relaunch_because_it_is_the_persons_own_frequency():
    """Same category as a volume: found by hunting for it over a session or two, and asking somebody to
    re-find it every launch is the failure `prefs` exists to prevent. Restored before the pacer, since
    setting the target sets the pacer from it."""
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        session.set_target_hz(0.065, 0.015)
        stored = session.calibration()
        assert stored["target_hz"] == 0.065 and stored["target_half_hz"] == 0.015
    finally:
        session.stop()

    fresh = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        assert fresh.target_hz != 0.065, "a new session does not start at the stored target"
        fresh.restore_calibration(stored)
        assert fresh.target_hz == 0.065 and fresh.target_half_hz == 0.015
        assert fresh.pacer_period == stored["pacer_period"], \
            "and the stored cycle survives, rather than being overwritten by one derived from the target"
    finally:
        fresh.stop()


def test_the_pacer_period_actually_changes_the_cue(monkeypatch):
    """The metronome's cue turns at the share of the cycle that is set, and the cycle repeats at the
    period that is set."""
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    now = [0.0]
    monkeypatch.setattr(HRVSession, "elapsed", property(lambda _s: now[0]))
    try:
        session.set_pacer_period(20)
        for at, inhale in ((0.0, True), (9.0, True), (11.0, False), (21.0, True)):
            now[0] = at
            assert session.breathe_in() is inhale, at
    finally:
        session.stop()


def test_opening_hrv_as_the_first_protocol_does_not_crash():
    """Preferences restore into an HRV session, which is not a `TrainingSession`, and a cycle set after
    that is in its calibration."""
    from nfb_console import prefs
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    session = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        # It reaches the mechanisms through the session's own hook rather than an attribute it assumed was
        # there, which is what makes this work on a screen whose settings are not only a `FeedbackSettings`.
        assert prefs.restore("HRV", session) >= 0
        session.set_pacer_period(12.5)
        assert session.calibration()["pacer_period"] == 12.5
    finally:
        session.stop()


def test_stopping_hrv_and_re_entering_it_keeps_the_setup():
    """`Console.pick` calls `adopt` whenever there is a previous session, and this screen had none — so the
    stop button raised too. Same line as the EEG screens draw: configuration comes across, history does not.
    """
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    first = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        first.set_pacer_period(13.0)
        first.set_metric("rmssd")
        first.set_strategy("zscore", window_s=90.0)
        first.feedback.music.source = "shepard"
        first.feedback.sound.high = 45.0

        second = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
        try:
            second.adopt(first)
            assert second.pacer_period == 13.0
            assert second.metric == "rmssd"
            assert second.strategy_id == "zscore" and second.strategy.window_s == 90.0
            assert second.feedback.music.source == "shepard"
            assert second.feedback.sound.high == 45.0
            # the history does not come across: a fresh normaliser has nothing in its window
            assert second.strategy.warmup == 0.0
            assert second.signal is None
            # and adopting a different protocol's session is refused rather than half-applied
            other = HRVSession(BY_CODE["AMP"], choice=HeartChoice(kind="simulator"))
            try:
                second.pacer_period = 7.0
                second.adopt(other)
                assert second.pacer_period == 7.0
            finally:
                other.stop()
        finally:
            second.stop()
    finally:
        first.stop()


def test_hrv_persists_what_was_set_by_ear_and_not_what_is_being_trained():
    """`prefs`' own line: calibration survives a relaunch, arrangement does not. The pacer's period is the
    person's own resonant rate and the volumes are the room; the metric and the scaling are what is being
    trained, which is the same category as a band or a threshold."""
    import json

    from nfb_console import prefs
    from nfb_console.hrv_view import HRVSession
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    prefs.PATH.unlink(missing_ok=True)
    first = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    first.set_pacer_period(11.5)
    first.feedback.sound.high = 30.0
    first.feedback.shadow.high = 80.0
    first.set_metric("pnn50")
    first.feedback.music.source = "shepard"
    first.stop()

    # One set for every protocol — see `prefs.SHARED`. This screen pins nothing, so everything it stores is
    # the room's and is what the next screen opens with.
    stored = json.loads(prefs.PATH.read_text())[prefs.SHARED]
    assert stored["pacer_period"] == 11.5
    assert stored["sound.high"] == 30.0 and stored["shadow.high"] == 80.0
    assert "metric" not in stored and "strategy" not in stored
    assert "shepard" not in str(stored), "which music is playing is the arrangement, not the calibration"

    second = HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        assert prefs.restore("HRV", second) > 0
        assert second.pacer_period == 11.5
        assert second.feedback.sound.high == 30.0
        assert second.feedback.shadow.high == 80.0
        assert second.metric == DEFAULT_METRIC, "what is trained comes from the protocol"
    finally:
        second.stop()


def test_the_hrv_screen_trains_on_the_cable_the_device_screen_chose(monkeypatch):
    """The whole path, from the source picked on the device screen to a running session: the screen asks
    the shared runner for the chosen source, the beats arrive, and the transport starts."""
    from nfb.hrv import SimulatedBeats
    from nfb_console import hrv_view
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    opened: list = []

    def fake_source(kind, **kw):
        opened.append((kind, kw))
        return SimulatedBeats(hr_bpm=60.0, realtime=False, seed=3)

    monkeypatch.setattr("nfb.hrv.beat_source", fake_source)

    choice = HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01")
    runner = HeartRunner(choice)
    session = hrv_view.session_for(BY_CODE["HRV"], heart_choice=choice, heart_runner=runner)
    assert session.choice is choice, "the screen trains on what was chosen, not an invented simulator"

    # **Two claims, and only one of them is about a thread.** That the screen opens the chosen source is
    # `run()`'s doing and is checked the moment it returns. That beats reach the session is the
    # processor's, and pushing them is instant where waiting for a real-time simulator to produce three
    # was three seconds of sleeping — for a fact about wiring, not about timing.
    session.run()
    assert opened and opened[0][0] == "usb" and opened[0][1]["port"] == "/dev/cu.usbserial-HM01"
    runner.release()
    beats = SimulatedBeats(hr_bpm=60.0, realtime=False, seed=3).beats()
    while not runner.ready:
        runner.state = runner.processor.push(next(beats))
    runner.seen_at = time.monotonic()
    assert session.dead_channels() == [], "so the screen has its source"

    assert not session.running, "a session opens paused"
    session.toggle_pause()
    assert session.running, "and the transport starts it"
    session.stop()


def test_the_device_chip_names_the_pulse_sensor_rather_than_the_absent_amplifier():
    """The chip answers "where are these numbers coming from", and on the HRV screen the answer is never
    the amplifier — which is not connected, and said "no device" in grey beside a working pulse sensor.
    That is true of the amplifier, useless as an answer, and reads as a fault."""
    from nfb_console import hrv_view
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    choice = HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01")
    session = hrv_view.session_for(BY_CODE["HRV"], heart_choice=choice,
                                   heart_runner=HeartRunner(choice))
    label, note, _tone = hrv_view._instrument(session)
    assert label == "HeartMath · USB"
    assert note == "not started"

    session.heart.error = "no HeartMath cable — plug it in · trying again"
    label, note, _tone = hrv_view._instrument(session)
    assert label == "HeartMath · USB", "the name does not become the fault"
    assert "device screen" in note, "and the fault points at where it is fixed"
    assert len(note) < 60, "the chip has room for a few words, not for the runner's sentence"


def test_connecting_a_sensor_mid_session_replaces_the_simulator_the_screen_invented(monkeypatch):
    """**The HRV screen invents a simulator when nothing is chosen**, so that picking the protocol does
    not open an empty screen. Connecting a real sensor on the device screen and coming back then found
    the session still holding that simulator — and `run` hands its choice to the shared runner, so
    returning actively switched the connected sensor off again.

    An invented stand-in loses to anything the operator actually chose, and the memo that hands the
    runner back switched off goes with it, since it is no longer ours to hand back.
    """
    from nfb.hrv import SimulatedBeats
    from nfb_console import hrv_view
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice
    from nfb_console.protocols import BY_CODE

    opened: list = []
    monkeypatch.setattr("nfb.hrv.beat_source",
                        lambda kind, **kw: (opened.append(kind),
                                            SimulatedBeats(realtime=False, seed=1))[1])

    app_choice = HeartChoice(kind="off")            # nothing chosen yet, as at launch
    runner = HeartRunner(app_choice)
    session = hrv_view.session_for(BY_CODE["HRV"], heart_choice=app_choice, heart_runner=runner)
    session.run()
    assert session.choice.kind == "simulator" and session._invented
    assert hrv_view._instrument(session)[0] == "simulated beats"

    # the device screen: a cable is chosen and connected, which is what set_heart_kind does
    app_choice.kind, app_choice.port = "usb", "/dev/cu.usbserial-HM01"
    runner.release()
    runner.attach(app_choice)

    session.rearm()                                  # coming back from the device screen
    session.run()
    try:
        assert session.choice is app_choice, "the session trains on what was actually chosen"
        assert not session._invented
        assert hrv_view._instrument(session)[0] == "HeartMath · USB", "and the chip says so"
        assert opened[-1] == "usb", "and the runner is reading the cable, not the simulator"
    finally:
        session.stop()
        assert app_choice.kind == "usb", "stopping must not switch the operator's own source off"


# --- the inhale setting --------------------------------------------------------------------------


def test_the_metronome_turns_at_the_inhale_share_and_the_ring_follows_it():
    """With an inhale of 30 percent of a 10 s cycle the turn is at 3 s, and the ring is half way through
    each half at 1.5 s and 6.5 s."""
    s = _session()
    clock = [100.0]
    s.now = lambda: clock[0]
    s.set_pacer_period(10.0)
    s.set_inhale(30)
    try:
        s.toggle_pause()
        for at, inhaling in ((0.1, True), (2.9, True), (3.1, False), (9.9, False), (10.1, True)):
            clock[0] = 100.0 + at
            assert s.breathe_in() is inhaling, at
        clock[0] = 101.5
        assert s.breath_phase() == pytest.approx(0.5, abs=0.02)
        clock[0] = 106.5
        assert s.breath_phase() == pytest.approx(0.5, abs=0.02)
    finally:
        s.stop()


@pytest.mark.parametrize("pct", [50, 30])
def test_the_pacer_follows_the_sweep_before_the_metronome(rig, pct):
    """While a sweep runs, in and out and the ring come from the sweep's own clock at the inhale share,
    whatever the heart does."""
    from nfb.hrv.sweep import breath_turn

    r = rig()
    s = r.session
    s.set_inhale(pct)
    assert r.begin()
    for i in range(60):
        r.look(0.8 * (i + 1))
        k, frac = s.sweep.schedule.phase_at(s.sweep.clock)
        inhaling, phase = breath_turn(frac, pct / 100.0)
        assert s.breathe_in() is inhaling
        ring = s.breath_phase()
        assert ring is not None and 0.0 <= ring <= 1.0
        assert ring == pytest.approx(phase, abs=0.05)


def test_a_change_of_the_inhale_share_acts_at_the_next_draw_also_during_a_sweep(rig):
    """The phase reads the live value: the same second of the same breath is an exhale after the share
    is cut."""
    r = rig()
    s = r.session
    assert r.begin()
    r.drive(4.0)                                       # 4 s into a breath of 8.9 s: under half
    assert s.breathe_in()
    s.set_inhale(30)
    r.look(r.t + 0.1)
    assert not s.breathe_in()
    s.set_inhale(90)
    r.look(r.t + 0.1)
    assert s.breathe_in()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_an_inhale_that_is_not_finite_is_ignored(bad):
    """No nearest value exists, so the setting stays where it was."""
    s = _session()
    try:
        s.set_inhale(40)
        s.set_inhale(bad)
        assert s.inhale_pct == 40
    finally:
        s.stop()


def test_the_inhale_is_kept_inside_its_limits_and_each_limit_leaves_an_exhale_and_an_inhale():
    """A far value is clamped, not refused, to a limit that leaves both halves of the breath, and a value
    inside the limits is kept."""
    from nfb_console.hrv_view import INHALE_RANGE

    lo, hi = INHALE_RANGE
    assert 0 < lo < hi < 100
    s = _session()
    try:
        for far, kept in ((0, lo), (-40, lo), (100, hi), (1e9, hi), (40, 40)):
            s.set_inhale(far)
            assert s.inhale_pct == kept
    finally:
        s.stop()


def test_the_inhale_is_saved_and_restored_with_the_pacer_and_a_bad_value_changes_nothing():
    """It travels in `calibration()` beside the pacer settings, restores clamped, and a value that is not
    finite or not a number leaves it."""
    s, other = _session(), _session()
    try:
        s.set_inhale(35)
        saved = s.calibration()
        assert "pacer_inhale" in saved
        other.restore_calibration(saved)
        assert other.inhale_pct == 35
        for bad in (float("nan"), float("inf"), "40", None, True):
            other.restore_calibration({"pacer_inhale": bad})
            assert other.inhale_pct == 35
        other.restore_calibration({"pacer_inhale": 1e9})
        s.set_inhale(1e9)
        assert other.inhale_pct == s.inhale_pct, "restored as the box would keep it"
    finally:
        s.stop()
        other.stop()


def test_a_session_that_takes_over_keeps_the_inhale_share():
    """The share is a setting of the pacer, so it comes across with the cycle and the sound."""
    old, new = _session(), _session()
    try:
        old.set_inhale(35)
        new.adopt(old)
        assert new.inhale_pct == 35
    finally:
        old.stop()
        new.stop()


def _built(name, make):
    """A screen built once for the tests that read it, kept where the shared builds are stopped."""
    import harness

    def build():
        from nfb_console import hrv_view

        r = make()
        page = StubPage()
        control = hrv_view.view(page, r.session, toolbar(), lambda: None)
        r.session.redraw()
        return page, control, r.session, r

    _page, control, _session_, r = harness.built_once(("sweep", name), build)
    return control, r


def _labels(control):
    return {t.value for t in _texts(control)}


def _row(control, label):
    """The row of a setting, found by its label."""
    return next((c for c in walk(control) if isinstance(c, ft.Row) and c.controls
                 and isinstance(c.controls[0], ft.Text) and c.controls[0].value == label), None)


def test_the_inhale_row_is_there_under_every_pacer(rig):
    """Every pacer runs on a clock, so the share acts under both and a row that acts is shown."""
    from nfb_console import hrv_view

    r = rig()
    s = r.session
    page = StubPage()
    control = hrv_view.view(page, s, toolbar(), lambda: None)
    s.redraw()
    assert s.pacer_choice == "metronome" and "inhale" in _labels(control)
    r.look(0.0)
    assert r.begin()
    s.redraw()
    assert "inhale" in _labels(control), "the share acts during a sweep, whatever the pill"


def test_what_the_inhale_box_shows_is_what_is_kept_and_it_adds_to_the_exhale_in_the_note(rig):
    """A share that is typed with a decimal is shown, kept and named in the note as the same number, and
    the inhale and the exhale on the screen add to one hundred."""
    from nfb_console import hrv_view

    r = rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    s.redraw()

    def inhale_row():
        for ctrl in walk(control):
            if isinstance(ctrl, ft.Row) and ctrl.controls and isinstance(ctrl.controls[0], ft.Text) \
                    and ctrl.controls[0].value == "inhale":
                return ctrl
        raise AssertionError("no inhale row")

    def shown():
        row = inhale_row()
        field = next(c for c in walk(row) if isinstance(c, ft.TextField))
        note = next(c.value for c in walk(row) if isinstance(c, ft.Text) and "exhale" in (c.value or ""))
        return field, float(field.value), float(re.search(r"exhale ([\d.]+)%", note).group(1))

    field, _, _ = shown()
    field.value = "37.5"
    field.on_submit(None)
    _, inhale, exhale = shown()
    assert inhale == pytest.approx(s.inhale_pct)
    assert inhale + exhale == pytest.approx(100.0)


@pytest.mark.parametrize("pct", [1, 50, 99])
def test_the_paced_swing_of_a_planted_sine_does_not_depend_on_the_inhale_share(rig, pct):
    """The swing is fitted against time inside the breath, so the same beats give the same swings at any
    share."""
    r = rig(plant="peak")
    s = r.session
    s.set_inhale(pct)
    assert r.begin()
    r.drive(260.0)
    values = s.sweep.values
    got = [(k, v) for k, v in enumerate(values) if v is not None]
    assert len(got) >= 10
    for k, v in got:
        assert v == pytest.approx(_PLANTS["peak"][0](s.sweep.schedule.rates[k]), rel=0.1)


@pytest.mark.parametrize("pct", [1, 99])
def test_ring_word_and_tone_still_work_at_the_ends_of_the_inhale_range(pct):
    """At 1 and at 99 percent the turn exists, the ring stays inside its range, the word changes and
    each phase sounds when a draw sees it."""
    s = _session()
    clock = [0.0]
    played = []
    s.now = lambda: clock[0]
    s.cue = lambda name, *a, **k: played.append(name)
    s.set_pacer_period(10.0)
    s.set_inhale(pct)
    try:
        s.toggle_pause()
        seen = []
        for at in np.arange(0.0, 20.0, 0.05):
            clock[0] = float(at)
            s.tick()
            seen.append(s.breathe_in())
            assert 0.0 <= s.breath_phase() <= 1.0
        assert True in seen and False in seen
        assert len(set(played)) == 2, "both pitches sound"
        assert played[0] != played[1]
    finally:
        s.stop()


# --- choosing the sweep, ending it, and the one sweep ------------------------------------------------


def test_choosing_sweep_on_a_running_session_leaves_it_running(rig):
    """The transport belongs to the person, as for every pacer."""
    r = rig()
    s = r.session
    s.toggle_pause()
    s.choose_pacer("sweep")
    r.look(0.0)
    r.look(0.8)
    assert s.running and s.sweeping
    s.choose_pacer("metronome")
    assert s.running


def test_breath_one_starts_with_the_first_beats_of_a_session_that_runs(rig):
    """Armed and paused, nothing runs. The first draw after the session runs starts breath 1, sounds its
    turn and starts the sweep's clock."""
    r = rig()
    s = r.session
    s.choose_pacer("sweep")
    r.look(0.0)
    assert not s.sweeping
    s.toggle_pause()
    r.look(0.0)
    assert s.sweeping and s.sweep.breath == 0 and s.sweep.clock == 0.0
    assert len(r.played) == 1, "the turn of breath 1 sounds at once"
    r.look(0.8)
    assert s.sweep.clock == pytest.approx(0.8)


WAITS = ["source", "error", "silence", "paused", "finished"]


def _waiting(rig_, cause):
    """A rig whose session has the sweep armed and is held by `cause`, and the way to lift the cause."""
    r = rig_(kind="off" if cause == "source" else "strap", length_seconds=5.0 if cause == "finished" else None)
    s = r.session
    if cause == "finished":
        s.toggle_pause()
        r.drive(8.0, stop=lambda: False)
        assert s.finished
    elif cause != "paused":
        s.toggle_pause()
    r.played.clear()
    s.choose_pacer("sweep")
    if cause == "error":
        s.heart.error = "the strap is gone"
    elif cause == "silence":
        s.heart.state = None
        r.frozen = True
    r.look(r.t + 0.8)

    def lift():
        if cause == "source":
            from nfb_console.inputs import HeartChoice

            s.heart.choice = HeartChoice(kind="strap")
        elif cause == "error":
            s.heart.error = None
        elif cause == "silence":
            r.frozen = False
        elif cause == "paused":
            s.toggle_pause()
    return r, lift


@pytest.mark.parametrize("cause", WAITS)
def test_an_armed_sweep_waits_for_each_cause_and_begins_when_it_is_gone(rig, cause):
    """It names what it waits for, it begins when that is gone, and the wait voids no breath. A session
    that is over never begins one."""
    r, lift = _waiting(rig, cause)
    s = r.session
    assert s.pacer_choice == "sweep" and not s.sweeping
    assert s.sweep_hold() == cause
    assert r.played == [], "nothing sounds while the sweep waits"
    lift()
    r.look(r.t + 0.8)
    r.look(r.t + 0.8)
    if cause == "finished":
        assert not s.sweeping and s.sweep_hold() == "finished"
    else:
        assert s.sweeping and s.sweep_hold() is None
        assert s.sweep.outcomes == [], "the wait is no breath that was not measured"


def test_choosing_the_metronome_stops_the_sweep_and_keeps_the_measured_part(rig, label="metronome"):
    """The measured breaths stay as a result, the pacer is the one that was chosen, and the ring follows it."""
    r = rig()
    s = r.session
    assert r.begin()
    r.drive(150.0)
    measured = [v for v in s.sweep.values if v is not None]
    assert measured
    s.choose_pacer(label)
    assert not s.sweeping and s.sweep.reason == "stop"
    assert [v for v in s.sweep.values if v is not None] == measured
    assert s.sweep.result.measured == len(measured)
    assert s.pacer_choice == label
    assert 0.0 <= s.breath_phase() <= 1.0


def test_choosing_another_pacer_while_the_sweep_waits_cancels_it_and_leaves_no_result(rig):
    """Nothing was measured, so nothing stays."""
    r = rig()
    s = r.session
    s.choose_pacer("sweep")
    s.choose_pacer("metronome")
    assert s.pacer_choice == "metronome" and s.sweep is None
    s.toggle_pause()
    r.look(0.0)
    r.look(0.8)
    assert not s.sweeping and s.sweep is None


@pytest.mark.parametrize("end", ["done", "plan", "source"])
def test_when_the_sweep_ends_the_pacer_is_the_metronome_again(rig, end):
    """Whatever ends it, the row shows the metronome again."""
    from nfb_console.inputs import HeartChoice

    r = rig(length_seconds=200.0 if end == "plan" else None)
    s = r.session
    before = s.pacer_choice
    assert r.begin()
    assert s.pacer_choice == "sweep"
    if end == "source":
        r.drive(30.0)
        s.heart.choice = HeartChoice(kind="usb", port="/dev/none")
        r.look(r.t + 0.8)
    else:
        r.drive(1400.0 if end == "done" else 220.0, stop=None if end == "done" else (lambda: not s.sweeping))
    assert s.sweep.reason == end
    assert s.pacer_choice == before != "sweep"


def test_choosing_sweep_again_after_a_result_replaces_it(rig):
    """The old curve, result and clock go; the new sweep starts empty."""
    r = rig()
    s = r.session
    assert r.begin()
    r.drive(120.0)
    s.choose_pacer("metronome")
    old = s.sweep
    assert old.result is not None and any(v is not None for v in old.values)
    s.choose_pacer("sweep")
    assert s.sweep is None, "the old result goes at once"
    r.look(r.t + 0.8)
    assert s.sweeping and s.sweep is not old and s.sweep.result is None
    assert all(v is None for v in s.sweep.values)


def test_choosing_sweep_while_one_runs_changes_nothing(rig):
    """One sweep, one clock: the choice again is a no-op, and it does not sound the turn again."""
    r = rig(kind="simulator")
    s = r.session
    assert r.begin()
    first = s.sweep
    r.drive(2.0)
    sounded = len(r.played)
    s.choose_pacer("sweep")
    r.look(r.t + 0.4)
    assert s.sweep is first and first.active and s.pacer_choice == "sweep"
    assert first.clock == pytest.approx(r.t)
    assert len(r.played) == sounded


def test_an_unknown_pacer_is_refused(rig):
    """A typed error, and the choice stays."""
    s = rig().session
    with pytest.raises(ValueError):
        s.choose_pacer("find my rate")
    assert s.pacer_choice != "sweep"


def test_a_source_change_ends_the_sweep_also_during_a_hold(rig):
    """Beats of another source cannot join the curve: the sweep ends at the next draw, paused or not."""
    from nfb_console.inputs import HeartChoice

    for paused in (False, True):
        r = rig()
        assert r.begin()
        r.drive(30.0)
        if paused:
            r.session.toggle_pause()
        r.session.heart.choice = HeartChoice(kind="usb", port="/dev/none")
        r.look(r.t + 0.8)
        assert not r.sweep.active and r.sweep.reason == "source"
        assert r.session.usable_rate() is None


def test_a_strap_that_loses_contact_holds_a_running_sweep_with_the_cause_silence(rig):
    """A strap without skin contact keeps notifying but gives no beat. A sweep on its beats waits with the
    cause `silence` once no beat has come for the time that the runner calls stale, however many
    notifications arrive."""
    from conftest import SWEEP_EPOCH
    from nfb_console.heart import HeartRunner

    class Notifies:
        def __init__(self, beat):
            self._beat = beat

        def beats(self):
            yield self._beat

        def close(self):
            pass

        def describe(self):
            return "test strap"

    r = rig()
    s = r.session
    assert r.begin()
    r.drive(60.0)
    assert s.sweep_hold() is None
    r.frozen = True
    for _ in range(int(HeartRunner.STALE_S) + 4):
        r.t += 1.0
        s.heart.source = Notifies(Beat(SWEEP_EPOCH + r.t, 0.0, [], contact=False))
        s.heart._stop.clear()
        s.heart._read(s.heart._generation)
        r.look(r.t)
    assert s.sweep_hold() == "silence"


@pytest.mark.parametrize("cause", ["error", "paused", "silence", "stall", "clock"])
def test_each_cause_holds_the_sweep_once_and_the_hold_ends_when_the_cause_does(rig, cause):
    """A hold is named, voids one breath whatever its length, and lets go when the cause is gone."""
    r = rig()
    s = r.session
    assert r.begin()
    r.drive(60.0)
    assert s.sweep_hold() is None
    voided = len(s.sweep.outcomes)
    breath = s.sweep.breath
    if cause == "error":
        s.heart.error = "the strap is gone"
        r.drive(r.t + 2.4, stop=lambda: False)
        held = s.sweep_hold()
        s.heart.error = None
    elif cause == "paused":
        s.toggle_pause()
        r.drive(r.t + 2.4, stop=lambda: False)
        held = s.sweep_hold()
        s.toggle_pause()
    elif cause == "silence":
        from nfb_console.heart import HeartRunner

        r.frozen = True
        r.drive(r.t + HeartRunner.STALE_S - 2.0, stop=lambda: False)
        assert s.sweep_hold() is None, "a short quiet is not a hold"
        r.drive(r.t + 4.0, stop=lambda: False)
        held = s.sweep_hold()
        r.frozen = False
    elif cause == "stall":
        r.look(r.t + 3.0)
        held = s.sweep_hold()
    else:
        r.skew = 10.0
        r.look(r.t + 0.8)
        held = s.sweep_hold()
    assert held == cause
    r.look(r.t + 0.8)
    r.look(r.t + 0.8)
    assert s.sweep_hold() is None
    new = s.sweep.outcomes[voided:]
    assert [o for o in new if o.reason == cause] == [(breath, None, cause)], "one void for the hold"
    assert s.sweep.breath == breath, "the breath comes round again"


def test_a_hold_freezes_the_ring_and_the_tone_and_the_way_back_sounds_the_turn(rig):
    """While held nothing moves and nothing sounds. The pacer goes back to the start of its breath, so
    the first draw after the hold sounds that turn."""
    from nfb_console import cues, hrv_view

    r = rig()
    s = r.session
    assert r.begin()
    sch = s.sweep.schedule
    r.drive(sch.start_s(3) + 0.7 * sch.period_s(3))
    assert not s.breathe_in(), "in the exhale of the fourth breath"
    s.toggle_pause()
    r.look(r.t + 0.8)
    sounded, ring, word = len(r.played), s.breath_phase(), s.breathe_in()
    for _ in range(5):
        r.look(r.t + 0.8)
    assert len(r.played) == sounded
    assert (s.breath_phase(), s.breathe_in()) == (ring, word)
    s.toggle_pause()
    r.look(r.t + 0.8)
    assert r.played[sounded:] == [cues.tone_at(hrv_view.PACER_IN_HZ)]
    assert s.sweep.breath == 3


def test_a_fault_that_holds_a_running_sweep_stops_the_tone_and_the_way_back_sounds_the_turn_again(rig):
    """A hold that begins in an exhale puts the pacer back at the start of its breath, an inhale. That turn
    does not sound during the hold, and it sounds when the hold ends, even if the last tone was an inhale."""
    from nfb_console import cues, hrv_view

    r = rig()
    s = r.session
    assert r.begin()
    sch = s.sweep.schedule
    r.drive(sch.start_s(3) + 0.7 * sch.period_s(3))
    assert not s.breathe_in()
    s.heart.error = "the strap is gone"
    r.look(r.t + 0.8)
    assert s.sweep_hold() == "error"
    sounded = len(r.played)
    for _ in range(5):
        r.look(r.t + 0.8)
    assert len(r.played) == sounded, "nothing sounds while the sweep is held"
    s.heart.error = None
    r.look(r.t + 0.8)
    assert r.played[sounded:] == [cues.tone_at(hrv_view.PACER_IN_HZ)]
    s.heart.error = "the strap is gone"
    r.look(r.t + 0.8)
    sounded = len(r.played)
    s.heart.error = None
    r.look(r.t + 0.8)
    assert r.played[sounded:] == [cues.tone_at(hrv_view.PACER_IN_HZ)], "also when it was an inhale before"


def test_a_sweep_sounds_its_first_turn_even_when_the_metronome_has_just_sounded_the_same_one(rig):
    """The person starts the sweep a moment into an inhale of the metronome: the sweep begins with its own
    inhale, and its tone is not taken for the metronome's."""
    from nfb_console import cues, hrv_view

    r = rig()
    s = r.session
    s.toggle_pause()
    r.look(0.0)
    inhale = cues.tone_at(hrv_view.PACER_IN_HZ)
    assert r.played == [inhale]
    s.choose_pacer("sweep")
    r.look(0.0)
    assert s.sweeping
    assert r.played == [inhale, inhale]


def test_the_tones_over_a_sweep_are_the_two_pitches_one_for_each_turn_and_one_cue_at_the_end(rig):
    """In and out alternate from the first draw, each at its turn in the schedule, and a sweep that
    ends by itself plays one cue that is no pitch."""
    from nfb_console import cues, hrv_view

    r = rig()
    s = r.session
    assert r.begin()
    r.drive(1400.0)
    assert s.sweep.reason == "done"
    pitches = [cues.tone_at(hrv_view.PACER_IN_HZ), cues.tone_at(hrv_view.PACER_OUT_HZ)]
    end = next(i for i, name in enumerate(r.played) if name not in pitches)
    assert [n for n in r.played if n not in pitches] == [r.played[end]], "one cue that is no pitch"
    tones = r.played[:end]
    assert tones[0] == pitches[0] and r.when[0] == 0.0
    assert all(a != b for a, b in zip(tones, tones[1:]))
    assert tones.count(pitches[0]) == s.sweep.schedule.breaths
    sch = s.sweep.schedule
    turns = sorted([sch.start_s(k) for k in range(sch.breaths)]
                   + [sch.start_s(k) + 0.5 * sch.period_s(k) for k in range(sch.breaths)])
    for turn, at in zip(turns, r.when[:end]):
        assert turn <= at < turn + 0.8 + 1e-6, "each tone sounds at its turn, not on a metronome"


@pytest.mark.parametrize("end", ["stop", "source"])
def test_a_sweep_that_is_stopped_before_its_end_plays_no_finish_cue(rig, end):
    """The finish cue marks a sweep that ran its course and nothing else."""
    from nfb_console import cues, hrv_view
    from nfb_console.inputs import HeartChoice

    r = rig()
    assert r.begin()
    r.drive(30.0)
    if end == "stop":
        r.session.choose_pacer("metronome")
    elif end == "source":
        r.session.heart.choice = HeartChoice(kind="usb", port="/dev/none")
    r.look(r.t + 0.8)
    r.look(r.t + 0.8)
    assert r.sweep.reason == end
    pitches = {cues.tone_at(hrv_view.PACER_IN_HZ), cues.tone_at(hrv_view.PACER_OUT_HZ)}
    assert set(r.played) <= pitches


def test_after_a_sweep_the_metronome_paces_as_if_no_sweep_had_run(rig):
    """The ring, the word and the note are the metronome's again."""
    from nfb_console.inputs import HeartChoice

    r = rig()
    s = r.session
    s.set_pacer_period(10.0)
    r.look(0.0)
    before = s.pacer_note()
    assert r.begin()
    r.drive(60.0)
    during = s.pacer_note()
    s.heart.choice = HeartChoice(kind="usb", port="/dev/none")
    r.look(r.t + 0.8)
    assert not s.sweeping and s.pacer_note() == before and during != before
    from nfb.hrv.sweep import breath_turn

    inhaling, phase = breath_turn((s.elapsed % s.pacer_period) / s.pacer_period, 0.5)
    assert s.breathe_in() is inhaling and s.breath_phase() == pytest.approx(phase)


def test_the_choices_fit_the_pacer_row_on_one_line():
    """The row does not wrap, and its label, choices and gaps are no wider than the inside of the panel."""
    from conftest import text_width
    from nfb_console import hrv_view, theme as theme_

    control, _r = _built("cold", _idle)
    row = _row(control, "pacer")
    group = row.controls[1]
    assert not getattr(row, "wrap", False) and not getattr(group, "wrap", False)
    spent = (row.controls[0].width or 0) + (row.spacing or 0) * (len(row.controls) - 1)
    spent += (group.spacing or 0) * (len(group.controls) - 1)
    for pill_ in group.controls:
        pad = pill_.padding
        chosen = pill_.content.weight is not None and pill_.content.weight != ft.FontWeight.W_400
        spent += text_width(pill_.content.value, theme_.SMALL, bold=chosen) + pad.left + pad.right + 2
    assert spent <= hrv_view.SIDE_W - 34, f"{spent:.0f}px of {hrv_view.SIDE_W - 34}"


@pytest.mark.parametrize("armed_only", [True, False], ids=["armed", "running"])
def test_under_the_sweep_the_cycle_row_is_absent_and_the_inhale_target_and_band_rows_act(rig, armed_only):
    """The sweep sets the breath length, so no cycle is offered. The inhale share, the target and the band
    stay, and a change in each one takes effect."""
    from nfb_console import hrv_view

    r = rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    if armed_only:
        s.choose_pacer("sweep")
    else:
        assert r.begin()
    s.redraw()
    assert s.pacer_choice == "sweep" and s.sweeping is not armed_only
    labels = _labels(control)
    assert "cycle" not in labels
    assert {"pacer", "inhale", "sound", "target", "band"} <= labels

    def submit(label, text):
        field = next(c for c in walk(_row(control, label)) if isinstance(c, ft.TextField))
        field.value = text
        field.on_submit(None)

    submit("inhale", "40")
    assert s.inhale_pct == 40
    submit("target", "0.08")
    assert s.target_hz == pytest.approx(0.08, abs=0.003)
    submit("band", "0.03")
    assert s.target_half_hz == pytest.approx(0.03, abs=0.003)
    assert s.pacer_choice == "sweep" and s.sweeping is not armed_only, "none of them ends the sweep"


def test_the_cycle_row_is_back_under_the_metronome_after_a_sweep(rig):
    """The row that the sweep removed returns when the choice does."""
    from nfb_console import hrv_view

    r = rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    assert r.begin()
    s.redraw()
    assert "cycle" not in _labels(control)
    _by_label(control, "metronome").on_click(None)
    s.redraw()
    assert "cycle" in _labels(control)


def test_the_sound_switch_acts_under_the_sweep(rig):
    """With the sound off a sweep sounds no turn."""
    r = rig()
    s = r.session
    s.pacer_sound = False
    assert r.begin()
    r.drive(30.0)
    assert r.played == []


@pytest.mark.parametrize("path", ["armed", "running", "held", "done", "stop", "plan", "source"])
def test_the_sweep_leaves_the_saved_settings_as_they_were(rig, path):
    """On every path, `calibration()` is equal to what it was, so a sweep is never saved as the pacer
    and a new session never starts in one."""
    from nfb_console.inputs import HeartChoice

    r = rig(length_seconds=300.0 if path == "plan" else None)
    s = r.session
    s.set_inhale(40)
    r.look(0.0)
    before = s.calibration()
    if path == "armed":
        s.choose_pacer("sweep")
    else:
        assert r.begin()
    if path == "done":
        r.drive(1400.0)
        assert s.sweep.reason == "done"
    elif path == "held":
        r.drive(100.0)
        s.toggle_pause()
        r.drive(r.t + 20.0, stop=lambda: False)
        assert s.sweep_hold() == "paused"
    elif path == "stop":
        r.drive(100.0)
        s.choose_pacer("metronome")
    elif path == "plan":
        r.drive(320.0)
        r.look(r.t + 0.8)
        assert s.sweep.reason == "plan"
    elif path == "source":
        r.drive(30.0)
        s.heart.choice = HeartChoice(kind="usb", port="/dev/none")
        r.look(r.t + 0.8)
        assert s.sweep.reason == "source"
    elif path == "running":
        r.drive(100.0)
    assert s.calibration() == before
    assert "sweep" not in str(s.calibration().values())


def test_a_session_that_restores_or_takes_over_never_starts_in_a_sweep(rig):
    """The saved settings and a session that is replaced carry the choice before the sweep and no sweep."""
    r = rig()
    s = r.session
    assert r.begin()
    r.drive(60.0)
    saved = s.calibration()
    fresh, again = _session(), _session()
    try:
        fresh.restore_calibration(saved)
        assert fresh.pacer_choice == "metronome" and fresh.sweep is None
        again.adopt(s)
        assert again.pacer_choice == "metronome" and again.sweep is None
    finally:
        fresh.stop()
        again.stop()


# --- the range of the sweep ----------------------------------------------------------------------

RANGES = [(6.75, 4.25), (4.25, 6.75), (8.0, 3.0), (3.5, 7.5)]
"""Start and end of a sweep: slowing and quickening over the default range, and a wider range of each."""


def _range_id(rate_range):
    return f"{rate_range[0]}-to-{rate_range[1]}"


def test_the_limits_of_the_range_are_rates_that_the_target_can_take_on_the_step_of_the_box():
    """The two limits lie inside the target's range and on the step, with the next step beyond each outside
    it, so every rate that the boxes take is one that `use this rate` can give the target."""
    from nfb.hrv import TARGET_RANGE_HZ
    from nfb_console import hrv_view

    lo, hi = hrv_view.SWEEP_RATE_RANGE
    step = hrv_view.SWEEP_RATE_STEP
    target_lo, target_hi = (60.0 * hz for hz in TARGET_RANGE_HZ)
    assert target_lo <= lo < target_lo + step and target_hi - step < hi <= target_hi
    assert lo / step == pytest.approx(round(lo / step)) and hi / step == pytest.approx(round(hi / step))
    s = _session()
    try:
        for rate in (lo, hi):
            s.set_target_hz(rate / 60.0)
            assert s.target_breaths_per_min == pytest.approx(rate)
    finally:
        s.stop()


def test_a_rate_is_kept_as_given_and_clamped_into_the_limits():
    """A typed rate between two steps of the arrows is the rate of the sweep, and one beyond a limit is kept
    at the limit, from either side."""
    from nfb_console import hrv_view

    lo, hi = hrv_view.SWEEP_RATE_RANGE
    s = _session()
    try:
        s.set_sweep_range(6.0, 5.0)
        assert s.set_sweep_start(5.3)
        assert s.sweep_start == 5.3 and s.sweep_schedule.start_per_min == 5.3
        assert s.set_sweep_start(1e9) and s.sweep_start == hi
        assert s.set_sweep_end(-3.0) and s.sweep_end == lo
    finally:
        s.stop()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_rate_that_is_not_finite_is_refused_and_nothing_changes(bad):
    """No nearest rate exists: both boxes keep their values and the call says that it changed nothing."""
    s = _session()
    try:
        before = (s.sweep_start, s.sweep_end, s.sweep_schedule)
        assert s.set_sweep_start(bad) is False and s.set_sweep_end(bad) is False
        assert s.set_sweep_range(bad, 5.0) is False and s.set_sweep_range(5.0, bad) is False
        assert (s.sweep_start, s.sweep_end, s.sweep_schedule) == before
    finally:
        s.stop()


def test_a_start_equal_to_the_end_is_refused():
    """With equal rates there is no sweep: the box keeps its value."""
    s = _session()
    try:
        before = (s.sweep_start, s.sweep_end)
        assert s.set_sweep_start(s.sweep_end) is False
        assert s.set_sweep_end(s.sweep_start) is False
        assert s.set_sweep_range(5.0, 5.0) is False
        assert (s.sweep_start, s.sweep_end) == before
    finally:
        s.stop()


@pytest.mark.parametrize("rate_range", RANGES, ids=_range_id)
def test_the_session_schedule_follows_the_two_settings_in_either_direction(rate_range):
    """The one schedule that the sweep runs and the screen reads is built from the start and the end that
    are set, so the direction is the order of the two."""
    s = _session()
    try:
        s.set_sweep_range(*rate_range)
        sch = s.sweep_schedule
        assert (sch.start_per_min, sch.end_per_min) == rate_range
        assert (sch.rates[0] > sch.rates[-1]) == (rate_range[0] > rate_range[1])
    finally:
        s.stop()


def test_a_change_of_the_range_while_the_sweep_waits_acts_at_once(rig):
    """Before breath 1, the sweep that begins runs the range that is set at that moment."""
    r = rig()
    s = r.session
    s.choose_pacer("sweep")
    s.set_sweep_range(4.5, 7.0)
    assert s.sweep is None and s.pacer_choice == "sweep"
    s.toggle_pause()
    r.look(0.0)
    assert s.sweeping
    assert (s.sweep.schedule.start_per_min, s.sweep.schedule.end_per_min) == (4.5, 7.0)


def test_moving_the_end_of_a_sweep_that_runs_keeps_the_sweep_and_every_measured_breath(rig):
    """The same series goes on over the new end: its breaths that were measured keep their values and it
    measures on."""
    r = rig()
    s = r.session
    assert r.begin()
    r.drive(300.0)
    sweep, before = s.sweep, s.sweep.values
    assert any(v is not None for v in before)
    assert s.set_sweep_end(3.0)
    assert s.sweep is sweep and s.sweeping and sweep.schedule.end_per_min == 3.0
    assert sweep.schedule.breaths > len(before)
    assert sweep.values[:len(before)] == before
    r.drive(r.t + 60.0)
    assert sum(v is not None for v in sweep.values) > sum(v is not None for v in before)


def test_moving_the_end_behind_the_pacer_ends_the_sweep_at_once_with_what_it_has_and_sounds_the_tone(rig):
    """The pacer is past an end that is near the start, so the sweep ends through its normal end: the
    pacer goes back, the result holds the measured breaths of the new range and the finishing tone sounds
    once."""
    r = rig()
    s = r.session
    assert r.begin()
    r.drive(300.0)
    before = s.sweep.values
    assert s.set_sweep_end(6.0)
    assert not s.sweeping and s.sweep.reason == "done" and s.pacer_choice != "sweep"
    result = s.sweep.result
    assert result.measured == sum(v is not None for v in before[:s.sweep.schedule.breaths])
    assert r.played.count("finish") == 1
    r.look(r.t + 0.8)
    assert r.played.count("finish") == 1


@pytest.mark.parametrize("rate_range", RANGES, ids=_range_id)
def test_changing_the_start_of_a_sweep_that_runs_starts_it_again_from_the_new_start(rig, rate_range):
    """The breaths so far belong to another schedule, so none stays: the sweep is armed again and, when the
    next draw begins it, it runs from the new start with nothing measured."""
    r = rig(rate_range=rate_range)
    s = r.session
    assert r.begin()
    r.drive(300.0)
    old = s.sweep
    assert any(v is not None for v in old.values)
    new_start = rate_range[0] + (0.5 if rate_range[0] < rate_range[1] else -0.5)
    assert s.set_sweep_start(new_start)
    assert s.sweep is None and not s.sweeping and s.pacer_choice == "sweep"
    r.look(r.t + 0.8)
    assert s.sweeping and s.sweep is not old
    assert s.sweep.schedule.start_per_min == new_start and s.sweep.schedule.end_per_min == rate_range[1]
    assert all(v is None for v in s.sweep.values)


def test_a_change_of_the_range_after_a_sweep_ended_leaves_its_result_and_acts_on_the_next_sweep(rig):
    """The result keeps the range that ran. The next sweep runs the range that is set."""
    r = _usable(rig)
    s = r.session
    done = s.sweep
    ran = (done.schedule.start_per_min, done.schedule.end_per_min)
    assert s.set_sweep_range(4.5, 7.0) and s.set_sweep_end(8.0)
    assert s.sweep is done and done.result is not None
    assert (done.schedule.start_per_min, done.schedule.end_per_min) == ran
    s.choose_pacer("sweep")
    r.look(r.t + 0.8)
    assert (s.sweep_schedule.start_per_min, s.sweep_schedule.end_per_min) == (4.5, 8.0)


@pytest.mark.parametrize("rate_range", RANGES, ids=_range_id)
def test_the_range_is_saved_and_restored_in_either_order_with_the_other_pacer_settings(rate_range):
    """Both rates travel in `calibration()`, so a sweep that quickens is restored as one that quickens,
    whichever of the two is set first."""
    s, other = _session(), _session()
    try:
        s.set_sweep_range(*rate_range)
        saved = s.calibration()
        assert other.restore_calibration(saved) >= 2
        assert (other.sweep_start, other.sweep_end) == rate_range
    finally:
        s.stop()
        other.stop()


def test_a_saved_range_that_is_bad_changes_nothing_and_a_far_one_is_clamped():
    """A value that is not a finite number, a pair that is equal, and nothing at all leave the range. One
    rate that is good and one that is missing set the good one."""
    from nfb_console import hrv_view

    s = _session()
    try:
        s.set_sweep_range(6.0, 5.0)
        for bad in (float("nan"), float("inf"), "5", None, True, [5.0], 10 ** 400):
            for key in ("sweep_start", "sweep_end"):
                assert s.restore_calibration({key: bad}) == 0
                assert (s.sweep_start, s.sweep_end) == (6.0, 5.0)
        assert s.restore_calibration({"sweep_start": 5.0, "sweep_end": 5.0}) == 0
        assert s.restore_calibration({"sweep_end": 6.0}) == 0
        assert (s.sweep_start, s.sweep_end) == (6.0, 5.0)
        assert s.restore_calibration({"sweep_start": 7.0}) == 1 and (s.sweep_start, s.sweep_end) == (7.0, 5.0)
        assert s.restore_calibration({"sweep_start": 1e9, "sweep_end": -1e9}) == 2
        assert (s.sweep_start, s.sweep_end) == tuple(reversed(hrv_view.SWEEP_RATE_RANGE))
    finally:
        s.stop()


def test_a_session_that_takes_over_keeps_the_range():
    """The range is a setting of the pacer, so it comes across with the cycle and the inhale."""
    old, new = _session(), _session()
    try:
        old.set_sweep_range(4.5, 7.0)
        new.adopt(old)
        assert (new.sweep_start, new.sweep_end) == (4.5, 7.0)
    finally:
        old.stop()
        new.stop()


@pytest.mark.parametrize("rate_range", RANGES, ids=_range_id)
def test_the_best_rate_and_its_use_hold_in_either_direction_and_over_any_range(rig, rate_range):
    """A peak planted inside the range is found there, the rate lies in the swept range, and using it moves
    the target to it and the metronome cycle with it."""
    r = _usable(rig, rate_range=rate_range)
    s = r.session
    best = s.sweep.result.best
    plant = 5.4
    assert min(rate_range) < plant < max(rate_range)
    assert abs(best.rate - plant) < 0.5
    rate = s.usable_rate()
    assert rate == pytest.approx(best.rate) and min(rate_range) <= rate <= max(rate_range)
    assert s.use_rate()
    assert s.target_breaths_per_min == pytest.approx(rate, abs=0.02)
    assert s.pacer_period == s.cycle_for_rate(rate)


# --- using the rate ----------------------------------------------------------------------------

def _usable(rig_, plant="peak", stop_at=None, **kw):
    r = rig_(plant=plant, **kw)
    assert r.begin()
    if stop_at is None:
        r.drive(r.session.sweep_schedule.total_s + 60.0)
    else:
        r.drive(stop_at)
        r.session.choose_pacer("metronome")
    return r


def test_a_rate_can_be_used_for_every_verdict_from_real_beats_and_it_lies_in_the_swept_range(rig):
    """Clear, not clear and too few breaths all keep the rate: a best rate exists, so the person decides."""
    seen = {}
    for name, kw in (("clear", dict()), ("not_clear", dict(plant="flat")), ("few", dict(stop_at=150.0))):
        r = _usable(rig, **kw)
        seen[name] = r
        rate = r.session.usable_rate()
        sch = r.sweep.schedule
        assert rate is not None and math.isfinite(rate)
        assert sch.low_per_min <= rate <= sch.high_per_min
        assert rate == pytest.approx(r.sweep.result.best.rate)
    assert [seen[k].sweep.result.verdict.kind for k in ("clear", "not_clear", "few")] \
        == ["clear", "not_clear", "few"]


def test_no_rate_can_be_used_from_simulated_beats_nor_without_a_result_nor_with_nothing_measured(rig):
    """Simulated beats never move the target; a running sweep and a sweep that measured nothing have no rate."""
    sim = _usable(rig, kind="simulator")
    assert sim.sweep.result.best is not None and sim.session.usable_rate() is None
    run = rig()
    assert run.session.usable_rate() is None
    assert run.begin()
    run.drive(60.0)
    assert run.session.usable_rate() is None
    run.session.choose_pacer("metronome")
    empty = rig()
    assert empty.begin()
    empty.session.choose_pacer("metronome")
    assert empty.sweep.result.best is None and empty.session.usable_rate() is None


def test_a_rate_cannot_be_used_after_the_source_changed(rig):
    """The key at press time must equal the key of the result."""
    from nfb_console.inputs import HeartChoice

    r = _usable(rig)
    before = r.session.calibration()
    assert r.session.usable_rate() is not None
    r.session.heart.choice = HeartChoice(kind="usb", port="/dev/none")
    assert r.session.usable_rate() is None
    assert r.session.use_rate() is False
    assert r.session.calibration() == before


def test_using_the_rate_moves_the_target_and_the_cycle_together_and_the_band_keeps_its_width(rig):
    """One press: the target is the rate, the cycle is its period, and the band width stays."""
    r = _usable(rig)
    s = r.session
    s.set_target_hz(0.1)
    rate, half = s.usable_rate(), s.target_half_hz
    assert abs(rate - 6.0) > 0.2
    assert s.use_rate() is True
    assert s.target_breaths_per_min == pytest.approx(rate, abs=0.01)
    assert s.pacer_period == pytest.approx(60.0 / rate, abs=0.26)
    assert s.target_half_hz == half


def test_using_the_rate_changes_nothing_while_the_target_is_already_there(rig):
    """The target at the rate: no change, and the settings are equal after the press."""
    r = _usable(rig)
    s = r.session
    s.set_target_hz(s.usable_rate() / 60.0)
    before = s.calibration()
    s.use_rate()
    assert s.calibration() == before


@pytest.mark.parametrize("rate", [math.nan, math.inf, 3.0, 8.0], ids=["nan", "inf", "below", "above"])
def test_a_best_rate_that_is_not_a_number_or_lies_outside_the_swept_range_cannot_be_used(rig, rate):
    """Whatever a result holds, the target moves only to a rate inside the range that the sweep covered."""
    from dataclasses import replace
    from types import SimpleNamespace

    from nfb.hrv.sweep import Best

    r = _usable(rig)
    s = r.session
    done = s.sweep
    stood = replace(done.result, best=Best(rate, 50.0, (1,)))
    s.sweep = SimpleNamespace(result=stood, schedule=done.schedule, reason="stop", active=False, cause=None)
    before = s.calibration()
    assert s.usable_rate() is None
    assert s.use_rate() is False
    assert s.calibration() == before


def test_a_rate_cannot_be_used_while_the_sweep_runs(rig):
    """No result exists yet, so a press changes nothing."""
    r = rig()
    assert r.begin()
    r.drive(100.0)
    before = r.session.calibration()
    assert r.session.use_rate() is False
    assert r.session.calibration() == before


# --- the sweep in BREATHE ------------------------------------------------------------------------------

def _breathe(control):
    """The BREATHE panel: the ring, the pacer rows, and under them the sweep's curve and result."""
    for c in walk(control):
        if isinstance(c, ft.Container) and isinstance(c.content, ft.Column) and c.content.controls:
            head = c.content.controls[0]
            if isinstance(head, ft.Row) and head.controls and isinstance(head.controls[0], ft.Text) \
                    and head.controls[0].value == "BREATHE":
                return c
    raise AssertionError("the HRV screen has no BREATHE panel")


def _said(control) -> str:
    return " | ".join(x.value for x in _texts(_breathe(control)))


def _curves(control):
    """The canvases that hold a curve of the sweep, not the ring's arc."""
    from nfb_console import sweep_view

    return [c for c in walk(_breathe(control)) if isinstance(c, cv.Canvas) and c.width == sweep_view.CURVE_W
            and c.shapes]


def _button(control, label):
    return _by_label(_breathe(control), label)


def _pressable(control, label) -> bool:
    found = _button(control, label)
    return found is not None and found.on_click is not None


def _full(**kw):
    r = Rig(**kw)
    assert r.begin()
    r.drive(r.session.sweep_schedule.total_s + 60.0)
    return r


def _cause_then_clear(cause, **kw):
    r = Rig(**kw)
    s = r.session
    assert r.begin()
    r.drive(100.0)
    if cause == "paused":
        s.toggle_pause(); r.drive(r.t + 2.4, stop=lambda: False); s.toggle_pause(); r.look(r.t + 0.8)
    elif cause == "error":
        s.heart.error = "gone"; r.drive(r.t + 2.4, stop=lambda: False); s.heart.error = None
        r.look(r.t + 0.8)
    elif cause == "silence":
        r.frozen = True; r.drive(r.t + 7.2, stop=lambda: False); r.frozen = False; r.look(r.t + 0.8)
    elif cause == "stall":
        r.look(r.t + 3.0); r.look(r.t + 0.8)
    else:
        r.skew = 10.0; r.look(r.t + 0.8); r.look(r.t + 0.8)
    assert r.sweep.outcomes[-1].reason == cause
    return r


def _breath_without_enough(reason):
    kw = dict(plant="slow_heart") if reason == "few" else dict(hole=(2.0, 4.7))
    r = Rig(**kw)
    assert r.begin()
    r.drive(60.0, stop=lambda: bool(r.sweep.outcomes))
    assert r.sweep.outcomes[-1].reason == reason
    return r


def _held(cause):
    r = Rig()
    s = r.session
    assert r.begin()
    r.drive(100.0)
    if cause == "paused":
        s.toggle_pause()
    elif cause == "error":
        s.heart.error = "gone"
    else:
        r.frozen = True
    r.drive(r.t + 7.2, stop=lambda: False)
    assert s.sweep_hold() == cause
    return r


def _ended(how):
    from nfb_console.inputs import HeartChoice

    r = Rig(length_seconds=200.0 if how == "plan" else None)
    r.session.set_target_hz(5.0 / 60.0)      # away from the rates that the first 200 s of the sweep cover
    assert r.begin()
    r.drive(210.0 if how == "plan" else 200.0)
    if how == "stop":
        r.session.choose_pacer("metronome")
    elif how == "source":
        r.session.heart.choice = HeartChoice(kind="usb", port="/dev/none")
    r.look(r.t + 0.8)
    r.look(r.t + 0.8)
    assert r.sweep.reason == how
    return r


def _inhale40():
    r = Rig()
    r.session.set_inhale(40)
    assert r.begin()
    r.drive(1400.0)
    return r


def _end_moved():
    r = _running()
    assert r.session.set_sweep_end(3.5)
    r.look(r.t + 0.8)
    return r


def _armed(cause, **kw):
    r, _lift = _waiting(Rig, cause)
    return r


def _armed_with(kind="strap", **kw):
    r = Rig(kind=kind, **kw)
    r.session.choose_pacer("sweep")
    r.look(0.0)
    return r


def _running(kind="strap", **kw):
    r = Rig(kind=kind, **kw)
    assert r.begin()
    r.drive(300.0)
    return r


def _idle(kind="strap"):
    r = Rig(kind=kind); r.look(0.0); return r


def _metronome():
    r = _idle()
    return r


SCENARIOS = {
    "cold": _idle, "metronome": _metronome,
    "armed paused": lambda: _armed("paused"), "armed no source": lambda: _armed("source"),
    "armed no beats": lambda: _armed("silence"), "armed error": lambda: _armed("error"),
    "armed session over": lambda: _armed("finished"),
    "armed simulated": lambda: _armed_with("simulator"),
    "armed maximum": lambda: _armed_with(length_seconds=420.0),
    "running": _running, "running simulated": lambda: _running("simulator"),
    "armed quickening": lambda: _armed_with(rate_range=(4.25, 6.75)),
    "armed wide": lambda: _armed_with(rate_range=(8.0, 3.0)),
    "running quickening": lambda: _running(rate_range=(4.25, 6.75)),
    "running wide": lambda: _running(rate_range=(3.5, 7.5)),
    "running end moved": _end_moved,
    "not measured few": lambda: _breath_without_enough("few"),
    "not measured gap": lambda: _breath_without_enough("gap"),
    "not measured paused": lambda: _cause_then_clear("paused"),
    "not measured error": lambda: _cause_then_clear("error"),
    "not measured silence": lambda: _cause_then_clear("silence"),
    "not measured stall": lambda: _cause_then_clear("stall"),
    "not measured clock": lambda: _cause_then_clear("clock"),
    "held paused": lambda: _held("paused"), "held error": lambda: _held("error"),
    "held silence": lambda: _held("silence"),
    "ended done": _full, "ended stop": lambda: _ended("stop"), "ended plan": lambda: _ended("plan"),
    "ended source": lambda: _ended("source"),
    "result not clear": lambda: _full(plant="flat"), "result edge": lambda: _full(plant="fast_peak"),
    "result simulated": lambda: _full(kind="simulator"), "result inhale 40": _inhale40,
    "result quickening": lambda: _full(rate_range=(4.25, 6.75)),
    "result wide": lambda: _full(rate_range=(3.5, 7.5)),
    "result falling heart": lambda: _full(plant="falling_heart"),
    "result quickening edge": lambda: _full(plant="fast_peak", rate_range=(4.25, 6.75)),
    "result quickening slow edge": lambda: _full(plant="slow_peak", rate_range=(4.25, 6.75)),
    "result slow edge": lambda: _full(plant="slow_peak"),
    "result in step": lambda: _full(plant="resonant"),
}
GROUPS = [pytest.param(name, id=name, marks=pytest.mark.xdist_group(f"sweep-{name}")) for name in SCENARIOS]


def _state(name):
    return _built(name, SCENARIOS[name])


def _hrv_texts(control):
    out = []
    for ctrl in walk(control):
        if isinstance(ctrl, ft.Text):
            out += [ctrl.value or ""] + [span.text or "" for span in (ctrl.spans or [])]
        if isinstance(getattr(ctrl, "tooltip", None), str):
            out.append(ctrl.tooltip)
    return [x for x in out if x]


@pytest.mark.parametrize("name", GROUPS)
def test_every_state_of_breathe_with_a_sweep_keeps_the_layout_rules(name):
    """The rules that run over every screen find no fault in it."""
    control, _r = _state(name)
    found = [f"{rule.__name__}: {fault}" for rule in RULES for fault in rule(control)]
    assert not found, "\n".join(found)


@pytest.mark.parametrize("name", GROUPS)
def test_every_state_of_breathe_with_a_sweep_reads_as_simplified_technical_english(name):
    """Every text of the screen in this state passes the mechanical rules."""
    control, _r = _state(name)
    found = []
    for text in dict.fromkeys(_hrv_texts(control)):
        found += ste_faults(text)
    assert not found, "\n".join(found)


@pytest.mark.parametrize("name", GROUPS)
def test_no_state_of_breathe_shows_a_number_that_is_not_one(name):
    """A figure that cannot be computed is left out, not printed as nan or inf."""
    control, _r = _state(name)
    bad = [x for x in _hrv_texts(control) if re.search(r"\b(nan|inf|-inf)\b", x, re.I)]
    assert not bad, bad


@pytest.mark.parametrize("name", GROUPS)
def test_no_text_names_a_button_that_the_screen_does_not_have(name):
    """The sweep has no button to start or stop it, and no text says to press one. Where a text names an
    action, the control for it is on the screen and acts."""
    control, _r = _state(name)
    said = _said(control)
    assert not re.search(r"find my rate|\bpress stop\b|\bthe stop button\b", said, re.I)
    assert _button(control, "find my rate") is None and _button(control, "stop") is None
    for label in ("use this rate", "close"):
        if f"Press {label}" in said:
            assert _pressable(control, label), label
    if re.search(r"\bChoose sweep\b", said):
        assert _pressable(control, "sweep")


def test_the_screen_has_one_help_for_breathe_and_no_sweep_panel():
    """The three pacers, the sweep and the inhale share are explained in the one help of BREATHE. The
    sweep has no panel of its own, so no help of its own."""
    import harness

    page, control = harness.built_screen("hrv_view", "HRV", 0, False)
    titles = [c.controls[0].value for c in walk(control) if isinstance(c, ft.Row) and c.controls
              and isinstance(c.controls[0], ft.Text) and c.controls[0].value in ("BREATHE", "SWEEP")]
    assert titles == ["BREATHE"]
    head = next(c for c in walk(_breathe(control)) if isinstance(c, ft.Row) and c.controls
                and isinstance(c.controls[0], ft.Text) and c.controls[0].value == "BREATHE")
    helps = [c for c in walk(head) if isinstance(c, ft.Container) and c.on_click is not None
             and isinstance(c.content, ft.Text) and c.content.value == "?"]
    assert len(helps) == 1
    helps[0].on_click(None)
    body = "".join(span.text for span in page.dialogs[-1].content.content.spans).lower()
    for word in ("metronome", "sweep", "inhale"):
        assert word in body, word
    assert not re.search(r"find my rate|\bpress stop\b", body)


def test_breathe_help_bolds_only_labels_that_are_on_the_cold_screen_and_names_the_sweep():
    """`sweep` is a pacer choice on the cold screen, so help can bold it, and every bold word is a label."""
    import harness

    page, control = harness.built_screen("hrv_view", "HRV", 0, False)
    bold = set()
    for ctrl in walk(control):
        glyph = getattr(ctrl, "content", None)
        if isinstance(ctrl, ft.Container) and ctrl.on_click is not None and isinstance(glyph, ft.Text) \
                and glyph.value == "?":
            ctrl.on_click(None)
            for span in page.dialogs[-1].content.content.spans:
                if span.style is not None and span.style.weight is not None:
                    bold.add(span.text.strip(" .,:;").lower())
    assert "sweep" in bold
    assert not _help_faults(page, control)


def _schedule_of_state(name):
    r = _state(name)[1]
    return r.session.sweep.schedule if r.session.sweep is not None else r.session.sweep_schedule


@pytest.mark.parametrize("name", ["armed paused", "armed quickening", "armed wide", "running",
                                  "running quickening", "running wide", "ended done", "result quickening",
                                  "result wide"])
def test_every_text_that_states_the_range_the_count_or_the_minutes_reads_the_schedule(name):
    """Both ends, the count of breaths and the minutes on the screen are those of the schedule that the
    state is in, and a range that is not the default shows none of the default numbers."""
    from nfb.hrv.sweep import END_PER_MIN, START_PER_MIN

    control, _r = _state(name)
    sch = _schedule_of_state(name)
    said = _said(control)
    assert f"{sch.start_per_min:g}" in said and f"{sch.end_per_min:g}" in said
    assert f"{sch.breaths} breaths" in said or f"of {sch.breaths} breaths" in said or f"of {sch.breaths}." in said
    if name.startswith(("armed", "running")):
        assert f"{round(sch.total_s / 60.0)} minutes" in said
    if {sch.start_per_min, sch.end_per_min}.isdisjoint({START_PER_MIN, END_PER_MIN}) or \
            (sch.start_per_min, sch.end_per_min) not in ((START_PER_MIN, END_PER_MIN), (END_PER_MIN, START_PER_MIN)):
        for default in (START_PER_MIN, END_PER_MIN):
            if default not in (sch.start_per_min, sch.end_per_min):
                assert f"{default:g}" not in said, default


def test_the_verb_of_the_pacer_note_follows_the_direction_of_the_sweep():
    """A start above the end decreases the rate and a start below it increases it."""
    slowing, quickening = _said(_state("armed paused")[0]), _said(_state("armed quickening")[0])
    assert "decreases" in slowing and "increases" not in slowing
    assert "increases" in quickening and "decreases" not in quickening


def test_the_range_row_is_there_with_a_sweep_chosen_and_nowhere_else():
    """One row, `range`, with its two boxes and the word between them; the cold screen, the metronome and a
    result after the pacer went back have none."""
    for name in ("armed paused", "running", "armed quickening", "running wide"):
        control, _r = _state(name)
        row = _row(_breathe(control), "range")
        assert row is not None, name
        kinds = [type(c).__name__ + (":" + c.value if isinstance(c, ft.Text) else "") for c in row.controls]
        assert kinds == ["Text:range", "Row", "Text:to", "Row"], kinds
        assert [len([x for x in walk(box) if isinstance(x, ft.TextField)]) for box in row.controls[1::2]] == [1, 1]
    for name in ("cold", "metronome", "ended done", "result quickening"):
        assert _row(_breathe(_state(name)[0]), "range") is None, name


def test_the_range_row_fits_the_column_on_one_line():
    """The label, both boxes and the word between them add up to less than the width inside the panel, so the
    row does not wrap."""
    from nfb_console import hrv_view
    from conftest import text_width

    row = _row(_breathe(_state("armed paused")[0]), "range")
    widths = []
    for c in row.controls:
        if isinstance(c, ft.Text):
            widths.append(c.width or text_width(c.value, float(c.size or 11)))
        else:
            widths.append(sum(x.width or (16 if isinstance(x, ft.Column) else text_width(x.value, 11))
                              for x in c.controls) + 4 * (len(c.controls) - 1))
    assert sum(widths) + 6 * (len(widths) - 1) < hrv_view.SIDE_W - 2 * 17


def test_typing_into_the_range_boxes_sets_the_ends_and_an_equal_value_is_refused_with_the_box_kept():
    """A typed start moves the start. A typed end that equals the start changes nothing, and the box shows
    the end that is kept."""
    from nfb_console import hrv_view

    r = Rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    s.choose_pacer("sweep")
    s.redraw()

    def boxes():
        return [next(x for x in walk(box) if isinstance(x, ft.TextField)) for box in _row(control, "range").controls[1::2]]

    first, _second = boxes()
    first.value = "5.5"
    first.on_submit(None)
    assert s.sweep_start == 5.5 and float(boxes()[0].value) == 5.5
    end = s.sweep_end
    _first, second = boxes()
    second.value = "5.5"
    second.on_submit(None)
    assert s.sweep_end == end and float(boxes()[1].value) == end


def test_moving_the_end_by_the_box_while_the_sweep_runs_moves_the_axis_and_keeps_the_curve():
    """The curve is drawn over the range of the sweep that it shows, so after the end moved its axis note
    names the new range, the tick labels stay apart and the dots that were measured are still drawn."""
    from nfb_console import sweep_view

    r = _running()
    from nfb_console import hrv_view

    page = StubPage()
    control = hrv_view.view(page, r.session, toolbar(), lambda: None)
    r.session.redraw()
    before = len(_curves(control)[0].shapes)
    r.session.set_sweep_end(3.5)
    r.look(r.t + 0.8)
    r.session.redraw()
    sch = r.session.sweep.schedule
    assert sch.low_per_min == 3.5
    assert f"{sch.low_per_min:g} to {sch.high_per_min:g} breaths a minute" in _said(control)
    assert len(_curves(control)[0].shapes) >= before - 3
    _assert_ticks_apart(control, sweep_view.TICK_W)


def _assert_ticks_apart(control, width):
    for stack in walk(_breathe(control)):
        if isinstance(stack, ft.Stack):
            lefts = sorted(c.left for c in stack.controls if isinstance(c, ft.Text) and c.visible
                           and c.left is not None and c.top == 0)
            assert all(b - a >= width for a, b in zip(lefts, lefts[1:])), lefts


@pytest.mark.parametrize("name", ["running quickening", "running wide", "result quickening", "result wide",
                                  "result falling heart", "result slow edge"])
def test_the_curve_axis_follows_the_range_and_its_tick_labels_do_not_overlap(name):
    """The note under the curve names the two ends of the sweep, and the labels of the scale never come
    closer than their own width, however wide the range."""
    from nfb_console import sweep_view

    control, _r = _state(name)
    sch = _schedule_of_state(name)
    assert f"{sch.low_per_min:g} to {sch.high_per_min:g} breaths a minute" in _said(control)
    _assert_ticks_apart(control, sweep_view.TICK_W)


def test_the_result_gives_the_mean_heart_rate_at_both_ends_of_a_sweep_whose_rate_falls():
    """The line names two rates in bpm, the first above the last, and they are the ones that the result
    holds. A sweep that was stopped early with too few breaths has no such line."""
    control, r = _state("result falling heart")
    first, last = r.session.sweep.result.heart_bpm
    said = _said(control)
    match = re.search(r"Mean heart rate: (\d+) bpm in the first breaths, (\d+) bpm in the last", said)
    assert match and int(match.group(1)) == round(first) and int(match.group(2)) == round(last)
    assert int(match.group(1)) > int(match.group(2)) + 10
    assert "Mean heart rate" not in _said(_state("running")[0])
    from nfb_console import hrv_view

    early = Rig()
    assert early.begin()
    early.drive(150.0)
    early.session.choose_pacer("metronome")
    page_control = hrv_view.view(StubPage(), early.session, toolbar(), lambda: None)
    early.session.redraw()
    assert early.session.sweep.result.heart_bpm is None
    assert "Mean heart rate" not in _said(page_control)


def _marks(control):
    """The vertical lines of the curve that are grey, by the x of each: the mark of the rate in step."""
    from nfb_console import theme as t

    return [round(shape.points[0].x, 3) for c in _curves(control) for shape in c.shapes
            if isinstance(shape, cv.Points) and shape.paint.color == t.MUTED
            and len(shape.points) == 2 and shape.points[0].x == shape.points[1].x]


def _x_at(rate, schedule):
    from nfb_console import sweep_view

    lo, hi = schedule.low_per_min, schedule.high_per_min
    return round(sweep_view.MARGIN + (rate - lo) / (hi - lo) * (sweep_view.CURVE_W - 2 * sweep_view.MARGIN), 3)


def test_the_result_names_the_largest_paced_swing_and_the_rate_in_step_in_two_sentences():
    """The two marks are two sentences that give two rates, each from its own part of the result. They
    are not merged into one rate."""
    control, r = _state("result in step")
    result = r.session.sweep.result
    said = _said(control)
    assert result.in_step.rate is not None
    swing = re.search(r"The paced swing is largest at (\d\.\d) breaths a minute", said)
    step = re.search(r"The heart rate and the breath are in step at (\d\.\d) breaths a minute\.", said)
    assert swing and step and float(swing.group(1)) == pytest.approx(result.best.rate, abs=0.05)
    assert float(step.group(1)) == pytest.approx(result.in_step.rate, abs=0.05)
    assert result.in_step.rate == pytest.approx(5.4, abs=0.4)


@pytest.mark.parametrize("name", ["result in step", "result falling heart", "result not clear"])
def test_the_curve_marks_the_rate_in_step_when_the_result_has_one_and_marks_nothing_when_it_has_none(name):
    """A grey line stands at the rate in step, over the same axis as the best rate. A result with no such
    rate draws no grey line."""
    control, r = _state(name)
    result = r.session.sweep.result
    expected = [] if result.in_step.rate is None else [_x_at(result.in_step.rate, r.session.sweep.schedule)]
    assert _marks(control) == expected
    if name == "result in step":
        assert expected


@pytest.mark.parametrize("name,side", [("result falling heart", "slower"), ("result not clear", None)])
def test_a_result_with_no_rate_in_step_says_so_and_where_the_rate_lies_when_the_heart_is_late_at_all_rates(name, side):
    """The planted heart of these results is late at every rate, so the rate in step lies below the range;
    a heart that follows nothing has no side to name."""
    control, r = _state(name)
    said = _said(control)
    assert "in step at no measured rate" in said
    assert ("is slower than the range" in said) == (side is not None)


def test_a_sweep_that_runs_shows_the_rate_in_step_so_far_once_a_crossing_exists():
    """No crossing, no line. Once the sweep has passed the rate in step, the line gives it, in the same
    unit as the result."""
    early, late = _running(plant="resonant"), _running(plant="resonant")
    late.drive(800.0)
    from nfb_console import hrv_view

    def line(r):
        control = hrv_view.view(StubPage(), r.session, toolbar(), lambda: None)
        r.session.redraw()
        return [c.value for c in walk(_breathe(control))
                if isinstance(c, ft.Text) and c.visible and (c.value or "").startswith("In step so far")]

    assert line(early) == []
    got = line(late)
    assert len(got) == 1 and re.fullmatch(r"In step so far: \d\.\d breaths a minute\.", got[0])


@pytest.mark.parametrize("name,where", [("result quickening edge", "fast"), ("result quickening slow edge", "slow"),
                                         ("result slow edge", "slow"), ("result edge", "fast")])
def test_the_edge_sentence_is_given_at_either_end_of_the_range_in_either_direction(name, where):
    """A peak at the slow end or at the fast end is called a peak at the end of the swept range, and the
    best rate lies at that end."""
    control, r = _state(name)
    sch = r.session.sweep.schedule
    result = r.session.sweep.result
    assert result.verdict.edge and "end of the swept range" in _said(control)
    mid = (sch.low_per_min + sch.high_per_min) / 2
    assert (result.best.rate > mid) == (where == "fast")


def test_a_quickening_result_names_the_best_rate_the_target_and_a_button_that_moves_it():
    """The sentences and `use this rate` hold for a sweep that quickens: the best rate is named, the target
    is compared, and the button moves the target to the best rate."""
    control, r = _state("result quickening")
    said = _said(control)
    best = r.session.sweep.result.best
    assert f"{best.rate:.1f} breaths a minute" in said and "At the target" in said
    assert _pressable(control, "use this rate")
    s = r.session
    rate = s.usable_rate()
    assert rate is not None


def test_a_state_of_the_range_does_not_show_the_default_numbers_after_the_range_changed():
    """Texts read the schedule and keep no copy: after the range of a waiting sweep changed, the screen
    shows the new ends at once."""
    from nfb_console import hrv_view

    r = Rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    s.choose_pacer("sweep")
    s.redraw()
    first = (s.sweep_start, s.sweep_end)
    assert f"{first[0]:g}" in _said(control)
    s.set_sweep_range(8.0, 3.0)
    s.redraw()
    said = _said(control)
    assert all(f"{rate:g}" not in said for rate in first) and "8 to 3" in said
    assert f"{s.sweep_schedule.breaths} breaths" in said
    shown = [float(x.value) for x in walk(_row(control, "range")) if isinstance(x, ft.TextField)]
    assert shown == [8.0, 3.0]


def test_the_help_says_that_a_second_sweep_the_other_way_checks_a_peak_from_the_order_of_the_breaths():
    """The help of BREATHE names the check for a peak that the order of the breaths makes."""
    import harness

    page, control = harness.built_screen("hrv_view", "HRV", 0, False)
    for ctrl in walk(control):
        glyph = getattr(ctrl, "content", None)
        if isinstance(ctrl, ft.Container) and ctrl.on_click is not None and isinstance(glyph, ft.Text) \
                and glyph.value == "?":
            ctrl.on_click(None)
            break
    body = " ".join("".join(span.text for span in page.dialogs[-1].content.content.spans).split())
    assert re.search(r"second sweep in the other direction checks a peak that comes from the order of the "
                     r"breaths", body)


def test_an_armed_sweep_that_waits_says_for_what_and_what_to_do():
    """Each wait has its own line. A pause points at the play button, a missing or faulty source at the
    device screen, no beats at the contact of the source, and a session that is over at the Session tab."""
    said = {n: _said(_state(f"armed {n}")[0]) for n in ("paused", "no source", "no beats", "error", "session over")}
    assert len(set(said.values())) == 5
    assert "play" in said["paused"].lower()
    assert "device" in said["no source"].lower() and "device" in said["error"].lower()
    assert "contact" in said["no beats"].lower()
    assert "Session tab" in said["session over"]
    assert not _pressable(_state("armed paused")[0], "use this rate")


def test_the_session_maximum_is_named_when_it_ends_the_sweep_first():
    """With a maximum that is shorter than the sweep, the note names the minutes that are left and the
    Session tab, before the sweep begins and while it waits. With no maximum it names neither."""
    said = {}
    for length in (420.0, None):
        control, _r = _state("armed maximum" if length else "armed paused")
        said[length] = _said(control)
    assert f"{round(420.0 / 60.0)} minutes" in said[420.0] and "Session tab" in said[420.0]
    assert "Session tab" not in said[None]


def test_simulated_beats_are_named_from_the_first_frame_and_real_beats_are_not():
    """The screen says that the beats are simulated before, during and after a sweep, and never says it of
    a strap. A sweep can run on them."""
    for name in ("armed simulated", "running simulated", "result simulated"):
        assert "simulated" in _said(_state(name)[0]).lower(), name
    for name in ("armed paused", "running", "ended done"):
        assert "simulated" not in _said(_state(name)[0]).lower(), name
    assert _state("running simulated")[1].session.sweeping


def test_a_running_sweep_says_which_breath_and_at_what_rate():
    """The breath out of the total, and the pacer's rate with its unit."""
    control, r = _state("running")
    said = _said(control)
    sch = r.sweep.schedule
    k = sch.breath_at(r.sweep.clock)
    assert f"{k + 1} of {sch.breaths}" in said
    assert f"{sch.rates[k]:.1f}" in said and "a minute" in said


def test_each_reason_that_a_breath_is_not_measured_has_its_own_line():
    """Seven reasons, seven different texts, each saying that the breath is not measured."""
    reasons = ("few", "gap", "paused", "error", "silence", "stall", "clock")
    lines = {}
    for reason in reasons:
        control, r = _state(f"not measured {reason}")
        assert r.sweep.outcomes[-1].reason == reason
        lines[reason] = _said(control)
        assert "measured" in lines[reason].lower(), reason
    assert len(set(lines.values())) == len(reasons)


def test_a_held_sweep_says_why_it_waits_and_what_to_do():
    """Three holds, three lines. A pause points at the play button and a fault at the device screen."""
    said = {c: _said(_state(f"held {c}")[0]) for c in ("paused", "error", "silence")}
    assert len(set(said.values())) == 3 and said["paused"] != _said(_state("running")[0])
    assert "play" in said["paused"].lower()
    assert "device" in said["error"].lower()


def test_each_end_says_how_far_the_sweep_went():
    """An end before the last breath says where it stopped. The sweep that ran its course does not claim
    to have stopped."""
    said = {n: _said(_state(f"ended {n}")[0]) for n in ("done", "stop", "plan", "source")}
    breaths = _state("ended done")[1].sweep.schedule.breaths
    for n in ("stop", "plan", "source"):
        assert re.search(rf"breath \d+ of {breaths}", said[n]), n
    assert not re.search(rf"breath \d+ of {breaths}", said["done"])
    assert len({said[n] for n in ("stop", "plan", "source")}) == 3
    assert not _pressable(_state("ended source")[0], "use this rate")


def test_the_use_this_rate_button_is_there_for_every_verdict_from_real_beats_and_not_otherwise():
    """Clear, not clear, too few and an edge keep it (a best rate exists). Simulated beats and a changed
    source never have it. Each result can be closed."""
    for name in ("ended done", "result not clear", "result edge", "ended stop", "ended plan"):
        control, r = _state(name)
        assert _pressable(control, "use this rate"), name
        assert _pressable(control, "close"), name
    kinds = {n: _state(n)[1].sweep.result.verdict for n in ("ended done", "result not clear",
                                                             "result edge", "ended stop")}
    assert [kinds[n].kind for n in ("ended done", "result not clear", "result edge", "ended stop")] \
        == ["clear", "not_clear", "clear", "few"]
    assert kinds["result edge"].edge and not kinds["ended done"].edge
    for name in ("result simulated", "ended source"):
        control, _r = _state(name)
        assert _button(control, "use this rate") is None, name
        assert _pressable(control, "close")


def test_a_result_that_is_not_clear_or_too_short_says_how_to_run_it_again_and_a_clear_one_does_not():
    """The action is a choice of the pacer, which is always there."""
    for name in ("result not clear", "ended stop"):
        control, _r = _state(name)
        assert "Choose sweep" in _said(control) and _pressable(control, "sweep"), name
    assert "Choose sweep" not in _said(_state("ended done")[0])


def test_a_clear_result_names_the_rate_the_swing_and_the_verdict_with_their_units():
    """Breaths a minute, milliseconds, and a verdict in words. A result that is not clear says so."""
    control, r = _state("ended done")
    said = _said(control)
    best = r.sweep.result.best
    assert f"{best.rate:.1f}" in said and "a minute" in said
    assert f"{best.swing_ms:.0f}" in said and "ms" in said
    assert "clear" in said.lower()
    not_clear = _said(_state("result not clear")[0]).lower()
    assert "not clear" in not_clear and "cannot move" not in not_clear


def test_the_result_says_the_comparison_once():
    """The size of the peak and the count of tries each appear once, so the lines add up and no line
    repeats a figure that another one or the curve already gives."""
    control, r = _state("ended done")
    said = _said(control)
    verdict = r.sweep.result.verdict
    assert said.count(f"{verdict.reached} of {verdict.tries}") == 1
    assert said.count(f"{r.sweep.result.best.swing_ms:.0f} ms") == 1


def test_the_result_states_the_inhale_share_when_it_is_not_half():
    """The rate was found at the share in force, so the view says which share that was, and only then."""
    assert re.search(r"\b40 ?%", _said(_state("result inhale 40")[0]))
    assert not re.search(r"sweep[^|]*inhale", _said(_state("ended done")[0]), re.I)


def test_a_result_says_so_when_the_inhale_setting_changed_during_the_sweep_and_names_no_share():
    """No single share was in force, so the view says that the setting changed and gives no figure."""
    from nfb_console import hrv_view

    r = Rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    assert r.begin()
    r.drive(120.0)
    s.set_inhale(30)
    r.drive(r.t + 20.0, stop=lambda: False)
    s.choose_pacer("metronome")
    s.redraw()
    said = _said(control)
    assert s.sweep_inhale_pct is None
    assert re.search(r"inhale setting changes", said)
    result_part = said.split("inhale setting changes")[1]
    assert not re.search(r"\b30 ?%", result_part.split("|")[0])
    r.close()


def test_the_target_sentence_names_no_swing_at_a_rate_where_no_breath_was_measured():
    """The curve covers the rates of the measured breaths. A target inside the swept range but outside
    them gets no figure of the paced swing, because the figure would come from another rate."""
    from nfb.hrv.sweep import curve
    from nfb_console import hrv_view

    r = Rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    assert r.begin()
    r.drive(150.0)
    s.choose_pacer("metronome")
    covered = [p.window_rate for p in curve(s.sweep.schedule.rates, s.sweep.values)]
    inside, outside = float(np.median(covered)), 5.0
    assert min(covered) < inside < max(covered) and not min(covered) <= outside <= max(covered)
    for target, shown in ((inside, True), (outside, False)):
        s.set_target_hz(target / 60.0)
        s.redraw()
        assert bool(re.search(r"paced swing is \d+ ms", _said(control))) is shown, target
    r.close()


def test_a_rate_that_no_breath_was_paced_near_cannot_be_used():
    """Breaths that were measured on either side of a long run that was not make one window, but the mean
    rate of that window is a rate that the pacer never ran near, so it must not move the target."""
    from nfb.hrv.sweep import SweepSchedule

    schedule = SweepSchedule()
    keep = set(range(10, 14)) | set(range(50, 55))
    r = Rig()
    mask = np.ones(len(r.stamps), dtype=bool)
    for k in range(schedule.breaths):
        if k not in keep:
            quarter = schedule.period_s(k) / 4.0
            lo, hi = schedule.start_s(k) + quarter - 0.02, schedule.start_s(k) + 2.0 * quarter + 0.02
            mask &= ~((r.stamps >= lo) & (r.stamps <= hi))
    r.stamps, r.rr = r.stamps[mask], r.rr[mask]
    try:
        assert r.begin()
        r.drive(1400.0, step=0.4)
        sweep = r.sweep
        assert sweep.reason == "done" and not [o for o in sweep.outcomes if o.reason in ("silence", "stall")]
        measured = [k for k, v in enumerate(sweep.values) if v is not None]
        assert measured == sorted(keep)
        usable = r.session.usable_rate()
        assert usable is None or min(abs(usable - schedule.rates[k]) for k in measured) < 0.15
    finally:
        r.close()


def test_the_target_sentence_follows_the_live_target_and_says_when_it_is_outside_the_swept_range():
    """Written at each draw from the target now: the rate with its unit, 'no change' with no button when
    the target is at the rate, and an outside target named as outside."""
    r = _full()
    s = r.session
    from nfb_console import hrv_view

    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    rate = s.usable_rate()
    texts = {}
    for name, target in (("six", 6.0), ("edited", 5.0), ("outside", 7.5), ("there", rate)):
        s.set_target_hz(target / 60.0)
        s.redraw()
        texts[name] = (_said(control), _pressable(control, "use this rate"))
    assert len(set(t for t, _ in texts.values())) == 4
    assert "6.0" in texts["six"][0] and "5.0" in texts["edited"][0] and "7.5" in texts["outside"][0]
    assert texts["six"][1] and texts["edited"][1] and texts["outside"][1]
    assert not texts["there"][1], "the target is at the rate: no button"
    r.close()


def test_the_screen_reads_the_schedule_of_the_sweep_that_runs(rig):
    """The breath count and the rate on the screen are those of the sweep that the session runs, so a
    session that runs another range is shown as it is."""
    from nfb_console import hrv_view

    r = rig(rate_range=(6.0, 5.0))
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    s.choose_pacer("sweep")
    s.redraw()
    assert f"{s.sweep_schedule.breaths}" in _said(control)
    assert r.begin()
    r.drive(40.0)
    s.redraw()
    said = _said(control)
    assert s.sweeping and f"of {s.sweep.schedule.breaths}." in said
    r.close()


def test_the_end_of_a_sweep_sounds_the_finishing_tone_only_when_that_tone_is_on(rig):
    """One switch governs the tone that marks an end: with the finishing tone off, a sweep that ends by
    itself plays none, and with it on it plays one."""
    played = {}
    for on in (True, False):
        r = rig()
        r.session.plan.finish_cue = on
        assert r.begin()
        r.drive(1400.0)
        assert r.sweep.reason == "done"
        played[on] = r.played.count("finish")
    assert played == {True: 1, False: 0}


def test_the_pills_and_buttons_act_on_the_session():
    """sweep arms, the transport runs it, a pacer pill stops it, use this rate moves the target and close
    drops the result, after which the sweep can be chosen again."""
    from nfb_console import hrv_view

    r = Rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    r.look(0.0)
    s.redraw()
    _button(control, "sweep").on_click(None)
    assert s.pacer_choice == "sweep" and not s.sweeping
    s.toggle_pause()
    r.look(0.0)
    assert s.sweep.active
    r.drive(300.0)
    s.redraw()
    _button(control, "metronome").on_click(None)
    assert not s.sweep.active
    s.set_target_hz(0.1)
    s.redraw()
    rate = s.usable_rate()
    _button(control, "use this rate").on_click(None)
    assert s.target_breaths_per_min == pytest.approx(rate, abs=0.01)
    s.redraw()
    _button(control, "close").on_click(None)
    s.redraw()
    assert _button(control, "close") is None and _button(control, "use this rate") is None
    assert not _curves(control)
    assert _pressable(control, "sweep")
    r.close()


def test_choosing_sweep_after_a_result_replaces_the_result_on_the_screen():
    """The curve, the verdict and the buttons of the old result are gone when the sweep is chosen again."""
    from nfb_console import hrv_view

    r = Rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    assert r.begin()
    r.drive(150.0)
    s.choose_pacer("metronome")
    s.redraw()
    assert _button(control, "close") is not None
    _button(control, "sweep").on_click(None)
    s.redraw()
    assert _button(control, "close") is None and _button(control, "use this rate") is None
    assert "Measured:" not in _said(control)
    r.close()


def test_the_rows_and_the_curve_are_built_again_only_when_something_that_they_show_changes():
    """A draw with nothing new keeps the same controls and the same shapes. A measured breath redraws the
    curve and not the rows, a target that moves redraws the curve, a change of the pacer gives new rows, and
    a target that reaches the rate takes the button away."""
    from nfb_console import hrv_view

    r = Rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    assert r.begin()
    r.drive(300.0)
    s.redraw()

    def shapes():
        return [id(x) for c in _curves(control) for x in c.shapes]

    row, drawn = _row(control, "target"), shapes()
    assert row is not None and drawn
    for _ in range(20):
        s.redraw()
    assert _row(control, "target") is row and shapes() == drawn
    r.drive(r.t + 14.0, stop=lambda: False)
    s.redraw()
    assert shapes() != drawn and _row(control, "target") is row, "a measured breath redraws the curve only"
    drawn = shapes()
    s.set_target_hz(s.target_hz + 0.01)
    s.redraw()
    assert shapes() != drawn, "the target moved"
    s.choose_pacer("metronome")
    s.redraw()
    assert _row(control, "target") is not row, "a new pacer gives new rows"
    close, row, drawn = _button(control, "close"), _row(control, "target"), shapes()
    assert close is not None
    for _ in range(20):
        s.redraw()
    assert _button(control, "close") is close and _row(control, "target") is row and shapes() == drawn
    s.set_target_hz(s.usable_rate() / 60.0)
    s.redraw()
    assert _button(control, "use this rate") is None and _button(control, "close") is not close
    r.close()


def test_the_median_line_is_broken_where_a_stretch_was_not_measured():
    """Breaths 20 to 51 are not measured. The line of medians is drawn in two pieces, one for each side, and
    never straight across the stretch that has no measure."""
    import flet.canvas as cv

    from nfb.hrv.sweep import SweepSchedule
    from nfb_console import hrv_view, theme

    schedule = SweepSchedule()
    r = Rig()
    mask = np.ones(len(r.stamps), dtype=bool)
    for k in range(20, 52):             # no beat in the second quarter of a breath: the breath is not measured
        quarter = schedule.period_s(k) / 4.0
        lo, hi = schedule.start_s(k) + quarter - 0.02, schedule.start_s(k) + 2.0 * quarter + 0.02
        mask &= ~((r.stamps >= lo) & (r.stamps <= hi))
    r.stamps, r.rr = r.stamps[mask], r.rr[mask]
    control = hrv_view.view(StubPage(), r.session, toolbar(), lambda: None)
    assert r.begin()
    r.drive(1400.0, step=0.4)
    assert r.sweep.reason == "done"
    r.session.redraw()
    canvases = [c for c in walk(control) if isinstance(c, cv.Canvas) and c.shapes]
    lines = [sh for c in canvases for sh in c.shapes
             if isinstance(sh, cv.Points) and sh.paint.color == theme.ACCENT and len(sh.points) > 2]
    assert len(lines) == 2
    r.close()


def test_a_sweep_that_starts_when_only_packets_without_a_beat_come_waits_with_the_cause_silence(rig):
    """The wait for the first beat counts from the newest accepted beat and not from the newest packet, so
    packets that carry no beat do not keep a sweep running that has seen no beat since it started."""
    from nfb_console.heart import HeartRunner

    r = rig()
    s = r.session
    r.drive(10.0, stop=lambda: False)
    s.choose_pacer("sweep")
    s.toggle_pause()
    r.look(r.t)
    assert s.sweeping
    base = s.heart.state
    r.frozen = True
    for _ in range(int(HeartRunner.STALE_S) + 3):
        r.t += 1.0
        state = beat_state(r.stamps, r.rr, 10.0)
        state.t = base.t + (r.t - 10.0)          # a packet later, and the same beats, a little older
        state.age_s = [a + (r.t - 10.0) for a in state.age_s]
        s.heart.state = state
        s.heart.seen_at = time.monotonic()
        s.tick()
    assert s.sweep_hold() == "silence"


def test_an_arrow_of_the_inhale_box_moves_the_shown_share_to_the_next_step_and_the_setting_with_it(rig):
    """After a share between two steps is typed, the up arrow raises the share that the box shows to the
    next multiple of the step, and the setting is that number."""
    from nfb_console import hrv_view

    r = rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    s.redraw()

    def row():
        return next(c for c in walk(control) if isinstance(c, ft.Row) and c.controls
                    and isinstance(c.controls[0], ft.Text) and c.controls[0].value == "inhale")

    def field():
        return next(c for c in walk(row()) if isinstance(c, ft.TextField))

    field().value = "37.5"
    field().on_submit(None)
    before = float(field().value)
    up = next(c for c in walk(row()) if isinstance(c, ft.Container) and isinstance(c.content, ft.Text)
              and c.content.value == "▲")
    up.on_click(None)
    step = hrv_view.INHALE_STEP_PCT
    assert 0.0 < float(field().value) - before <= step and float(field().value) % step == 0.0
    assert s.inhale_pct == pytest.approx(float(field().value))


def test_a_typed_cycle_of_the_metronome_is_the_cycle_that_runs(rig):
    """A cycle typed between two steps of the arrows is shown as typed and is the metronome's period."""
    from nfb_console import hrv_view

    r = rig()
    s = r.session
    control = hrv_view.view(StubPage(), s, toolbar(), lambda: None)
    s.redraw()

    def field():
        row = next(c for c in walk(control) if isinstance(c, ft.Row) and c.controls
                   and isinstance(c.controls[0], ft.Text) and c.controls[0].value == "cycle")
        return next(c for c in walk(row) if isinstance(c, ft.TextField))

    field().value = "13.37"
    field().on_submit(None)
    assert s.pacer_period == 13.37 and float(field().value) == 13.37


def test_the_heart_plots_take_the_width_that_the_window_gives_them():
    """A wider window draws the spectrum and the heart rate wider."""
    from nfb_console import hrv_view

    def extent(width: float) -> float:
        page = StubPage()
        page.width = width
        session = _session()
        try:
            control = hrv_view.view(page, session, toolbar(), lambda: None)
            session.heart.state = coherent_state()
            session.heart.seen_at = time.monotonic()
            session.redraw()
            xs = [pt.x for c in walk(control) if isinstance(c, cv.Canvas) for sh in c.shapes
                  if isinstance(sh, cv.Points) for pt in sh.points]
            return max(xs)
        finally:
            session.stop()

    assert extent(1700.0) > extent(1100.0) + 300.0
