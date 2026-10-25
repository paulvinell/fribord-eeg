"""Render every protocol screen headlessly, with real data through it.

This exists because of a specific failure: an extraction moved the spectrum axis into `plots.py`, left a
stale variable name behind in the draw path, and **200 tests passed**. Every test in this repo inspected
Python objects, and the objects were fine — the screen would have thrown on its first frame.

So these tests do the one thing the others do not: build the screen, push a real recording through the
session, and call `draw()`. They assert very little about appearance, because that is not what they are
for. They assert that a frame can be produced at all, that it can be produced before any data arrives
and after, and that the captions which state a graph's scale actually got written — which is the closest
a headless test can get to "the screen is not blank".

A stub `Page` stands in for Flet. That is sound for the draw path — `page.update()` is a no-op in both
cases — but it does *not* prove anything about updates reaching a client. That failure mode is invisible
to any test here, and the rule that prevents it is that every UI mutation runs on Flet's own loop.
"""

from __future__ import annotations

import re
import time

import flet as ft
import pytest

from nfb import ReplaySource, Settings, WaitingSource
from nfb_console import chrome
from nfb_console.feedback import COMBINE_DETAIL, LEVEL as LEVEL_CH
from nfb_console.protocols import BY_CODE

from conftest import BENCH, counts as to_counts, waitfor

RECORDING = str(BENCH)      # see conftest: a generated bench recording, not sessions/


class StubPage:
    width = 1240

    def __init__(self):
        self.controls, self.services, self.tasks, self.dialogs = [], [], [], []
        self.window = type("W", (), {})()
        self.updates = 0

    def show_dialog(self, dialog):
        self.dialogs.append(dialog)

    def pop_dialog(self):
        self.dialogs.pop()

    def add(self, control):
        self.controls.append(control)

    def update(self, *a):
        self.updates += 1

    def run_task(self, fn, *a):
        self.tasks.append((fn, a))     # never awaited here: a draw loop would run for the whole session


class StubDevice:
    """Just enough for `chrome.Toolbar`, which reads device status every refresh."""

    class _Settings:
        hz = 250.0
        reference = "linked"
        mains_hz = 50

    class _Status:
        from nfb import Mode
        mode = Mode.DISCONNECTED
        failure = None
        remedy = ""              # the launcher's status chip reads it; the toolbar does not
        model = ""
        connected = False
        settings = None

    def __init__(self):
        self.status = self._Status()
        self.status.settings = self._Settings()


def shown(dialog) -> str:
    """A help dialog's text as it reads on screen: its runs joined, the emphasis markers gone."""
    return "".join(span.text for span in dialog.content.content.spans)


def toolbar():
    return chrome.Toolbar(StubDevice(), lambda: None, lambda: None)


def pump(session, blocks: int = 200_000, until=None) -> None:
    """Drive the session's feed **on this thread**, until it has what the test needs.

    **No thread and no clock.** This started the feed on a thread and then polled with a 50 ms sleep
    until a deadline — which measures the scheduler, costs whatever the sleep granularity happens to be,
    and fails under load rather than when the code is wrong. `session.feed_block` is the body of that
    thread's loop, so a test can simply be the loop: exactly the blocks it needs, in order, and stop.

    The default `until` is "enough for the spectrum, not merely enough for a band amplitude" — the
    spectrum branch only runs once a second of samples exists, so stopping earlier would leave the whole
    of it (axes, gridlines, bin colouring) undrawn and unchecked.
    """
    def ready() -> bool:
        ring = session.rings.get(session.focus) if hasattr(session, "rings") else None
        if ring is not None:
            return ring.psd() is not None
        return bool(session.amplitudes)

    done = until or ready
    for n, item in enumerate(session.source.blocks()):
        session.feed_block(item)
        if done() or n >= blocks:
            break
    session._stop = True

def replay(loop: bool = False):
    return ReplaySource(RECORDING, realtime=False, loop=loop)


def at_corner(corner_hz: float, channels: int = 2):
    """A source whose high-pass corner is **stated rather than inherited from `sessions/`**.

    `replay()` reads a gitignored file that is whatever was recorded last, so a capture made at 0.01 Hz
    changes the corner under any test that reads one — which it did, once `corner_hz` began reading the
    corner from the source rather than from a device that only exists live.
    """
    return WaitingSource(Settings(channels=channels, lower_limit_hz=corner_hz).build())


def test_only_the_newest_view_keeps_drawing_and_a_bad_frame_is_reported_once():
    """**Two failures behind one report: the screen kept animating while the buttons went dead.**

    Going to the device screen suspends the session without stopping it — contact is checked mid-session and
    the session holds the clock, the calibrations and the trends — so a loop that only watches `_stop` runs
    on against controls that are no longer on the page. Coming back builds a second view with a second loop
    on the same session, and every further trip adds one more. Each costs a full redraw and a page update
    ten times a second on the same event loop that delivers clicks, so the screen goes on repainting while
    input starves. `session.redraw` is the newest view's draw function, so identity against it is the test.

    And a frame that raises used to end the loop and write nothing anywhere, taking the transport with it —
    the play button calls the same `draw`. Now it is reported once, however many times it recurs, and
    drawing continues.
    """
    import asyncio

    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    bar = toolbar()
    try:
        av.view(page, session, bar, lambda: None)
        first_loop, first_args = page.tasks[-1]
        av.view(page, session, bar, lambda: None)               # the trip back from the device screen
        second_loop, second_args = page.tasks[-1]

        asyncio.run(asyncio.wait_for(first_loop(*first_args), timeout=2.0))
        assert session._stop is False, "the session is alive; it is the old *view* that ended"

        # The live one keeps going, so it has to be stopped rather than awaited to completion.
        calls = []
        boom = [True]

        def bad_draw():
            calls.append(1)
            if boom[0]:
                raise ValueError("no frame today")

        session.redraw = bad_draw

        async def run_briefly():
            task = asyncio.ensure_future(chrome.render_loop(page, session, bad_draw, bar,
                                                            seconds=0.001))
            await asyncio.sleep(0.05)
            session._stop = True
            await task

        asyncio.run(run_briefly())
        assert len(calls) > 3, "one failed frame does not end the screen"
        assert bar.fault.visible and "no frame today" in bar.fault_text.value
        assert len(bar._seen_faults) == 1, "the same fault ten times a second is one report"

        # Dismissing clears the strip and does not re-open it, or closing it would be undone in 100 ms.
        already = bar.fault_text.value
        bar.clear_fault()
        bar.report(already)
        assert not bar.fault.visible, "a fault the operator has closed does not come back"

        # A genuinely different one still gets through, which is the distinction worth keeping.
        bar.report("KeyError: something else — while drawing AMP, at x.py:1")
        assert bar.fault.visible and "KeyError" in bar.fault_text.value
    finally:
        session.stop()


# --- the amplitude screen -------------------------------------------------------------------

def test_the_amplitude_screen_draws_before_any_data_arrives():
    """"waiting for data" rather than a blank axis — and the empty state has to survive a real frame."""
    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        control = av.view(page, session, toolbar(), lambda: None)
        assert control is not None
        session.redraw()                    # no blocks fed at all
        assert session.error is None
    finally:
        session.stop()


def _feed_two_tones(session, antiphase: bool, seconds: float = 4.0) -> None:
    """Put the same 10 Hz tone on both channels, in phase or in antiphase, and measure it.

    Hand-built blocks rather than a recording, because the whole question is what happens to a *phase*
    relationship and a recording has whatever phase it has. Blocks are handed over one channel at a time in
    the order the device sends them, so this also exercises the aligner rather than stepping around it.
    """
    import numpy as np

    from nfb.stream import Block

    specs = session.specs
    chans = list(specs)
    rate = specs[chans[0]].rate_hz
    scale = specs[chans[0]].scale
    n = int(seconds * rate)
    times = np.arange(n) / rate
    wave = 30.0 * np.sin(2 * np.pi * 10.0 * times)
    other = -wave if antiphase else wave
    ints = {ch: to_counts(values, scale) for ch, values in zip(chans, (wave, other))}
    for start in range(0, n - 8, 8):
        for ch in chans:
            block = Block(ch, start, tuple(ints[ch][start:start + 8].tolist()), specs[ch], state=0)
            # The gate's real verdict, as the feed hands it over — the traces read it to mark which
            # columns were thrown away. `pass_through` is the unjudged one, which is what this is.
            session.observe_raw(block, session.gates.pass_through(block))
            fresh = session.consume(block)
            for tgt in session.targets:
                if session._due(tgt, fresh):
                    tgt.observe(session.value_for(tgt), 20.0)


def test_the_summed_source_adds_the_signals_rather_than_their_amplitudes():
    """The point of the source, and the thing this screen could not do before.

    Adding the *signals* and filtering once keeps the phase relationship: two channels carrying the same
    rhythm reinforce when they agree and cancel when they oppose. Measuring each channel and combining the
    numbers afterwards — which is what a control labelled `SUM` used to do here — cannot see that at all, and
    reports the same amplitude either way.

    The numbers below are the physics, and they are worth pinning as numbers: two 30 µV channels give 60 µV
    peak summed in phase and nothing at all in antiphase — read here as **42.4 µV RMS**, since the default
    reward estimator is an elliptic filter and reports RMS. The ratio is what the test is about; the
    convention is why the number is not 60, and a threshold does not transfer across it.
    """
    from nfb.dsp import ALPHA
    from nfb_console import amplitude_view as av
    from nfb_console.targets import PAIR

    def summed_amplitude(antiphase: bool) -> float:
        session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
        try:
            session.targets.clear()
            session.metrics.clear()
            tgt = session.add_target(PAIR, ALPHA, "increase")
            _feed_two_tones(session, antiphase)
            return session.value_for(tgt)
        finally:
            session.stop()

    together = summed_amplitude(antiphase=False)
    opposed = summed_amplitude(antiphase=True)

    assert together == pytest.approx(42.4, rel=0.1), "two 30 µV channels in phase add, at 1/√2 for RMS"
    assert opposed < 0.02 * together, "and in antiphase they cancel — which an amplitude sum cannot show"


def test_a_summed_target_says_what_it_is_measured_on():
    """"ch0+ch1" rather than "SUM", because the arithmetic is the label: those channels, added. And it has
    to survive being moved back — a `pair` left behind by switching to one channel would keep claiming two
    electrodes are contributing when one is.

    This is the *target's* label, and the one place it is shortened is the card's source button, where
    spelling it out on four channels pushed the remove button off the card — see the test below."""
    from nfb.dsp import ALPHA
    from nfb_console import amplitude_view as av
    from nfb_console.targets import PAIR

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        tgt = session.add_target(PAIR, ALPHA, "increase")
        assert tgt.source_label == "ch0+ch1"

        session.retune(tgt, source=0)
        assert tgt.pair is None, "a stale pair would outlive the source that needed it"
        assert tgt.source_label == "ch0"
    finally:
        session.stop()


def test_the_source_button_is_the_same_width_whatever_it_is_measured_on():
    """**The card is 430 px wide and its identity row was allowed to grow with the channel count.**

    That row is a colour chip, this button, two band edges with steppers, the direction and the remove — and
    the only part of it whose width depends on the session is the source label. Spelled out on four channels
    it is `ch0+ch1+ch2+ch3`, fifteen characters, and the remove button ended up outside the card's border:
    a control you cannot reach, on the widest arrangement this screen offers.

    So the button says SUM and the arithmetic is in its tooltip. Nothing that has room loses it — the target
    still calls itself `ch0+ch1+ch2+ch3`, and the filter dialog still prints "measures … on" it.
    """
    import flet as ft

    from nfb.dsp import ALPHA
    from nfb_console import amplitude_view as av
    from nfb_console.targets import PAIR

    def source_button(session):
        found = []

        def walk(control, depth=0):
            if control is None or depth > 20:
                return
            if isinstance(control, ft.PopupMenuButton) and isinstance(control.content, ft.Container) \
                    and isinstance(control.content.content, ft.Text):
                found.append(control.content)
            for attr in ("controls", "shapes"):
                for child in getattr(control, attr, None) or []:
                    walk(child, depth + 1)
            for attr in ("content", "control"):
                walk(getattr(control, attr, None), depth + 1)

        walk(av.view(StubPage(), session, toolbar(), lambda: None))
        return found[0]

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=4)
    try:
        session.targets.clear()
        session.add_target(PAIR, ALPHA, "decrease")
        button = source_button(session)
        assert button.content.value == "SUM"
        assert button.tooltip == "ch0+ch1+ch2+ch3 — added, then measured"
        assert session.targets[0].source_label == "ch0+ch1+ch2+ch3", "the target keeps the arithmetic"
    finally:
        session.stop()

    # A channel names itself: there is nothing to shorten and no tooltip to add.
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=4)
    try:
        button = source_button(session)
        assert button.content.value == "ch0" and button.tooltip is None
    finally:
        session.stop()


def test_the_amplitude_screen_says_when_the_corner_makes_its_bands_meaningless():
    """**At the 0.01 Hz corner this screen measures electrode drift, and nothing on it could show that.**

    The band estimator leaks about 6% of whatever is outside the band, which is nothing against EEG and
    everything against a few hundred microvolts of drift: measured on the captures in `sessions/`, the
    alpha reward read 20-44 µV where the true alpha was 3, following the drift at 0.77-0.98 and the alpha
    at 0.00-0.06. The raw trace and the artifact gate are both drawn through their own 0.5 Hz floor, so
    both looked healthy throughout.

    The corner belongs to the device and stays where the last screen left it — the infra-low, waveform and
    SCP screens all need 0.01 Hz — so arriving here with it set is the ordinary route and the screen has to
    say so rather than assume a mistake.
    """
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=at_corner(0.01, channels=2))
    try:
        control = av.view(StubPage(), session, toolbar(), lambda: None)
        session.redraw()
        said = [c.value for c in _texts(control) if c.value and "high-pass" in c.value]
        assert said, "the screen says nothing about a corner that invalidates every band on it"
        assert "0.5 Hz" in said[0], "and it names the setting to change"
        assert len(said[0]) <= 110, "it lectures"
    finally:
        session.stop()

    # At the corner the bands are actually measured on, there is nothing to say.
    session = av.AmplitudeSession(BY_CODE["AMP"], source=at_corner(0.5, channels=2))
    try:
        assert session.wiring_problem() is None
        control = av.view(StubPage(), session, toolbar(), lambda: None)
        session.redraw()
        assert not [c.value for c in _texts(control) if c.value and "high-pass" in c.value]
    finally:
        session.stop()

    # And a target deliberately edited below the floor wants that content, so the warning goes away.
    session = av.AmplitudeSession(BY_CODE["AMP"], source=at_corner(0.01, channels=2))
    try:
        session.retune(session.targets[0], low=0.05, high=0.2)
        assert session.wiring_problem() is None
    finally:
        session.stop()


def test_an_unwired_channel_leaves_the_summed_inhibits_measuring_the_rest():
    """**A dead electrode used to take every summed target down with it, in silence.**

    The default inhibits open on the sum, and a summed target requires every channel it names — which is
    right for a measure *between* channels and wrong for a sum, where one contributor dropping out leaves a
    smaller sum rather than no measurement. Four channels enabled with three wired, which is an ordinary
    evening, and both inhibits sat inert at 0.00 µV for the whole session: never calibrated, never in the
    level, never in the click, with nothing on the card saying so.

    An unwired channel is what the device flags *and* what the samples show — railed to full scale — which
    is both halves of `_judge_channel`'s rule, so this feeds three channels a signal and the fourth its rail
    with the amplifier's no-signal bit set. It then asserts the two things that were broken: the target
    still measures, and its card names the electrodes it is actually measuring.
    """
    import numpy as np

    from nfb.stream import NO_SIGNAL, Block
    from nfb_console import amplitude_view as av
    from nfb_console.targets import PAIR

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=4)
    try:
        session.paused = False              # a paused session observes no target, by design
        spec = session.specs[0]
        rate = spec.rate_hz
        rail = int(round(800.0 / spec.scale))
        t = np.arange(int(40 * rate)) / rate
        signal = 20.0 * np.sin(2 * np.pi * 4.0 * t)          # inside the 2-6 Hz inhibit's band
        for i in range(0, signal.size - 7, 8):
            live = tuple(int(round(v / spec.scale)) for v in signal[i:i + 8])
            for ch in (0, 1, 2):
                session.feed_block(Block(ch, i, live, session.specs[ch]))
            session.feed_block(Block(3, i, (rail,) * 8, session.specs[3], state=NO_SIGNAL))

        assert session.state[3] is False, "a railed input is a disconnected one"
        summed = [t for t in session.targets if t.source == PAIR]
        assert len(summed) == 2, "the two default inhibits"
        for tgt in summed:
            assert tgt.pair == (0, 1, 2), "the card names the electrodes actually being measured"
            assert session.target_is_live(tgt), "three live channels are a sum"
            assert tgt.value > 0.0, "and it is being measured, not left at zero"
        assert all(t in session.active_targets for t in summed), "so they reach the feedback"
    finally:
        session.stop()


def test_the_amplitude_screen_offers_a_spectrum_of_the_sum():
    """Every other multi-channel screen has one; this one did not, and it is the same signal a summed
    target is measured on — so the spectrum and the reward are looking at one thing.

    The channels lead and the sum comes last, which is the opposite of the coupling screen's order and is
    the claim each screen is making: there every reward is measured on the sum, here a target sits on one
    channel unless somebody moves it.
    """
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        labels = [s.label for s in session.spectrum_sources()]
        assert labels == ["ch0", "ch1", "sum"]
        # A sum is not a channel and must not move the focus that decides where a new target lands.
        assert session.spectrum_sources()[-1].channel is None
    finally:
        session.stop()

    # With one channel there is nothing to add, and offering "sum" would name a second electrode that is
    # not there — a row of one pill reading "sum" over a single channel's spectrum.
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        assert [s.label for s in session.spectrum_sources()] == ["ch0"]
    finally:
        session.stop()


def test_every_screen_lets_you_overrule_the_device_about_a_channel():
    """It was on one screen and the problem is on all of them.

    The amplifier's below-range detector fires on a perfectly good channel — 20 µV of EEG is 2.5% of the
    narrowest range it offers — and nothing in software can tell that apart from an electrode that has come
    off. The person looking at the waveform can, and they are looking at whichever screen they are on. The
    state was always shared (`TrainingSession.trust_channel`); only the way to reach it was not.
    """
    import importlib

    for mod, code in (("amplitude_view", "AMP"), ("cfc_view", "CFC"), ("at_view", "A-T"),
                      ("ilf_view", "ILF"), ("pac_view", "PAC"), ("scp_view", "SCP"),
                      ("sleep_view", "SLEEP"), ("wave_view", "WAVE")):
        module = importlib.import_module(f"nfb_console.{mod}")
        session = module.session_for(BY_CODE[code], source=replay(), channels=2)
        try:
            page = StubPage()
            module.view(page, session, toolbar(), lambda: None)
            rows = _trace_rows(page, session)
            assert rows is not None, f"{code} has no shared trace rows"

            # Three states, cycled by clicking: the device's word, forced live, forced dead.
            rows._toggle(0)()
            assert session.trusted.get(0) is True and session.state.get(0) is True
            assert " ✓" in rows.labels[0].value, f"{code}: a forced channel has to look forced"
            rows._toggle(0)()
            assert session.trusted.get(0) is False and session.state.get(0) is False
            rows._toggle(0)()
            assert 0 not in session.trusted, f"{code}: the third click hands the decision back"
        finally:
            session.stop()


def show_all_panes(control) -> None:
    """Open every tab at once. A hidden pane deliberately paints nothing — repainting four invisible
    panes at 10 Hz is what starved the event loop that delivers clicks — so a test that reads a pane's
    text has to open it first, exactly as a person would."""
    for pane in control.controls[2].controls:
        pane.visible = True


def _trace_rows(page, session):
    """The `TraceRows` a screen built, found by asking the session's own redraw closure for it.

    Reaching into the view is the point: what is being checked is that the *screen* wired the shared rows up,
    not that the class works — which is a separate test and would pass with the rows left out of the layout.
    """
    from nfb_console.trace_rows import TraceRows

    for cell in getattr(session.redraw, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, TraceRows):
            return value
    return None


def test_the_amplitude_screen_draws_with_real_data_through_it():
    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        av.view(page, session, toolbar(), lambda: None)
        pump(session)
        for _ in range(3):
            session.redraw()
        assert session.error is None, session.error
    finally:
        session.stop()


def test_every_graph_states_its_own_window_and_scale():
    """The convention, checked rather than trusted: silent autoscaling turns a trace into decoration."""
    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        control = av.view(page, session, toolbar(), lambda: None)
        pump(session)
        session.redraw()
        captions = [c.value for c in _texts(control)]
        joined = " | ".join(captions)
        assert "full scale" in joined
        assert "across" in joined
        assert "µV" in joined or "%" in joined
    finally:
        session.stop()


def test_switching_every_tab_and_mechanism_pane_draws():
    """Panes are held in a Stack with `visible` toggled, so the draw loop keeps writing to hidden ones.
    That is the arrangement; this checks it does not throw when the visible one changes."""
    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        av.view(page, session, toolbar(), lambda: None)
        pump(session)
        for mech in (session.feedback.sound, session.feedback.shadow, session.feedback.click):
            mech.enabled = True
        session.redraw()
        assert session.error is None
    finally:
        session.stop()


def test_the_add_button_adds_a_target_on_every_screen_that_offers_one():
    """Pressing "+ add" on the amplitude screen raised `too many values to unpack` and no card appeared.

    The preset rows grew a fifth column when a target learned which channel it is measured on, and the
    two places that read them were updated one apiece. Nothing caught it because a view's own click
    handlers are closures nothing else calls — so this presses the button rather than reading the table.
    """
    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        view = av.view(page, session, toolbar(), lambda: None)
        before = len(session.targets)
        button = _by_label(view, "+ add")
        assert button is not None, "the button has to be findable to be pressed"
        button.on_click(None)
        assert len(session.targets) == before + 1
        added = session.targets[-1]
        assert added.source == session.focus, "added where you are looking, not on the preset's channel"
        assert (added.band.low, added.band.high) == av.DEFAULT_TARGETS[0][:2]
        assert added.direction == "increase" and added.role == "reward", \
            "a card added by hand is a reward on the way up, whatever entry zero says"
    finally:
        session.stop()

    from nfb_console import cfc_view

    page = StubPage()
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        view = cfc_view.view(page, session, toolbar(), lambda: None)
        before = len(session.rewards)
        _by_label(view, "+ add").on_click(None)
        assert len(session.rewards) == before + 1, "the same button, a different mechanism behind it"
    finally:
        session.stop()


def _numbers(control, depth: int = 0):
    """Every editable number on a control tree, as a list of strings.

    Separate from `_texts` because a typeable value is a TextField and not a Text — which is the whole point
    of it being typeable, and is how two tests that asserted a *rendered* pitch stopped seeing it the moment
    the pitch became settable.
    """
    import flet as ft

    if control is None or depth > 20:
        return []
    if isinstance(control, ft.TextField):
        return [control.value] if control.value else []
    out = []
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            out += _numbers(child, depth + 1)
    for attr in ("content", "control"):
        out += _numbers(getattr(control, attr, None), depth + 1)
    return out


def _by_label(control, label: str, depth: int = 0):
    """The container whose only Text is `label` — i.e. a pill, found by what it says.

    For checking that an inert control is *on the screen and inert*, which is the claim the convention
    makes: locked is not hidden. Asking the model whether it thinks the choice is blocked would pass with
    nothing rendered.
    """
    import flet as ft

    if control is None or depth > 20:
        return None
    if isinstance(control, ft.Container) and isinstance(control.content, ft.Text) \
            and control.content.value == label:
        return control
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            found = _by_label(child, label, depth + 1)
            if found is not None:
                return found
    for attr in ("content", "control"):
        found = _by_label(getattr(control, attr, None), label, depth + 1)
        if found is not None:
            return found
    return None


def _texts(control, depth: int = 0):
    """Every Text in a control tree, for checking that captions were written.

    The depth limit is a runaway guard and nothing else, so it is set well clear of what any screen
    actually nests: at 14 it sat one level above the target cards, and wrapping a Train tab in a column to
    hang a warning above it silently emptied this list rather than failing the walk.
    """
    import flet as ft

    if control is None or depth > 20:
        return []
    if isinstance(control, ft.Text):
        return [control] if control.value else []
    out = []
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            out += _texts(child, depth + 1)
    content = getattr(control, "content", None)
    if content is not None:
        out += _texts(content, depth + 1)
    return out


def _menu_texts(control, depth: int = 0):
    """Every Text inside a popup menu's items. Separate from `_texts` deliberately: a menu's options are
    not on the screen until it is opened, and folding them in would make every caption assertion count
    things nobody can see."""
    import flet as ft

    if control is None or depth > 20:
        return []
    out = []
    for item in getattr(control, "items", None) or []:
        out += _texts(getattr(item, "content", None), depth + 1)
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            out += _menu_texts(child, depth + 1)
    out += _menu_texts(getattr(control, "content", None), depth + 1)
    return out


# --- the CFC screen -------------------------------------------------------------------------

def test_the_cfc_screen_draws_before_any_data_arrives():
    from nfb_console import cfc_view

    page = StubPage()
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        assert control is not None
        session.redraw()
        assert session.error is None
    finally:
        session.stop()


def test_the_cfc_screen_draws_with_real_data_through_it():
    from nfb_console import cfc_view

    page = StubPage()
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        pump(session)
        for _ in range(3):
            session.redraw()
        assert session.error is None, session.error
        captions = " | ".join(c.value for c in _texts(control))
        assert "full scale" in captions
        assert "in as it crosses" in captions, "the 5 s trace is a breathing pacer and should say so"
        # the arithmetic behind that — six breaths a minute is 0.1 Hz — is behind the panel's `?`, not
        # standing on a screen watched for twenty minutes
        helper = _find_help(control, "Raw traces and the pacer")
        assert helper is not None
        helper.on_click(None)
        assert "0.1 Hz" in shown(page.dialogs[-1])
    finally:
        session.stop()


def test_cfc_opens_with_two_rewards_and_twelve_bins():
    """The protocol, as a shape: two hailing frequencies up, six bins per channel down."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert [t.band.centre for t in session.rewards] == [5.0, 10.0]
        assert len(session.bin_targets) == 12
        assert all(t.direction == "increase" for t in session.rewards)
        assert all(t.role == "inhibit" and t.direction == "increase" for t in session.bin_targets)
        edges = sorted({(t.band.low, t.band.high) for t in session.bin_targets})
        assert edges == [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 45)]
        # every reward frequency sits on a bin boundary — the interlock, not a coincidence
        boundaries = {e for pair in edges for e in pair}
        for reward in session.rewards:
            assert reward.band.centre in boundaries
    finally:
        session.stop()


def test_cfc_thresholds_are_derived_from_his_total_not_from_our_defaults():
    """5-15% across twelve bins is under 1% each; our own inhibit default of 25% would allow 300%.

    And the rewards sound too. Inheriting our 90%-per-reward default would have put two rewards below
    threshold ~19% of the time on top of the bins' 10% — nearly 30% of the session making a noise, which
    is exactly the figure he calls "too often for most people", and fatal to Quiet Reward.
    """
    from nfb_console import cfc_view
    from nfb_console.targets import HIT_RATES

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        for tgt in session.bin_targets:
            assert tgt.hit_rate != HIT_RATES["inhibit"]
        bins_over = sum(1 - t.hit_rate for t in session.bin_targets)
        assert bins_over == pytest.approx(0.10, abs=0.001), "the middle of a 5-15% range"

        for tgt in session.rewards:
            assert tgt.hit_rate != HIT_RATES["reward"], "our default would be far too loose here"
        rewards_under = sum(1 - t.hit_rate for t in session.rewards)
        assert rewards_under == pytest.approx(0.05, abs=0.001)
        assert bins_over + rewards_under <= 0.15, "the top of the range, all sounds together"
    finally:
        session.stop()


def test_adding_a_reward_does_not_buy_more_interruption():
    """The point of a third frequency is to require more, not to be interrupted more — so the budget is
    fixed and each reward's aim rises with the count."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        two = sum(1 - t.hit_rate for t in session.rewards)
        session.retune_rewards([5.0, 10.0, 40.0])
        assert len(session.rewards) == 3
        three = sum(1 - t.hit_rate for t in session.rewards)
        assert three == pytest.approx(two, abs=0.001)
        assert len({t.hit_rate for t in session.rewards}) == 1, "and they share it evenly"
    finally:
        session.stop()


def test_cfc_uses_his_two_rules_in_the_right_places():
    """He has two, for two kinds of output, and collapsing them into one was the first mistake.

    The discrete consequences are a strict AND — a sound plays when *any* instrument fails — and the
    continuous one follows the rewards: "increases Audio volume of an MP3 file as the rewards increase".
    `weakest` is the AND written continuously and was measurably wrong for the continuous output: a bin
    decided the level 55% of the time and it tracked the rewards at r = 0.48.
    """
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert session.feedback.combine == "scaled", "the continuous output follows the rewards"
        # and the discrete side is still the strict AND, which no combining rule can weaken
        assert session.feedback.on_target([]) is False
        assert session.feedback.describe_condition() == "every target on its wanted side"
        assert "rewards are up" in COMBINE_DETAIL[session.feedback.combine]
    finally:
        session.stop()


def test_a_channel_with_no_signal_cannot_hold_the_feedback_down():
    """Its value is zero, its threshold calibrates to zero, so its score is exactly 0.5 forever — which
    under `weakest` caps the output at half permanently. Measured on a four-column export whose last two
    columns are padding: twelve of twenty-four bins sat at 0.500."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        session.state = {0: True, 1: False}
        live = session.active_targets
        assert live, "channel 0 is still delivering"
        assert all(t.source != 1 for t in live if t.pair is None)
        assert not any(t.pair for t in live), "a pair needs both channels"
        # and it is still on the screen: an unsatisfiable condition you cannot see is worse than one you can
        assert len(session.targets) > len(live)
    finally:
        session.stop()


def test_a_reward_below_the_devices_high_pass_is_refused_not_measured():
    """"Ultra-low frequencies" are in the manual, and the device high-passes at 0.5 Hz by default — a
    0.1 Hz reward against that corner measures the filter's roll-off."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=at_corner(0.5), channels=2)
    try:
        assert session.corner_hz == 0.5
        assert session.wiring_problem() is None
        session.retune_rewards([0.1, 40.0])
        trouble = session.wiring_problem()
        assert trouble is not None and "high-pass" in trouble and "0.01 Hz" in trouble
    finally:
        session.stop()

    # And nothing to refuse once the corner is actually below the reward — read from the source, so this
    # holds on a replay of a 0.01 Hz capture and not only on a live device.
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=at_corner(0.01), channels=2)
    try:
        session.retune_rewards([0.1, 40.0])
        assert session.wiring_problem() is None
    finally:
        session.stop()


def test_cfc_refuses_to_pretend_about_linked_referencing():
    """A shared reference appears in both channels and adds in phase into the sum, so it inflates the
    reward. The screen has to say so rather than train an artifact."""
    from nfb_console import cfc_view

    class Linked(StubDevice):
        def __init__(self):
            super().__init__()
            self.status.settings.reference = "linked"

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert session.wiring_problem() is None, "no device: the wiring is whatever it was"
        session.device = Linked()
        trouble = session.wiring_problem()
        assert trouble is not None and "bipolar" in trouble
        session.device.status.settings.reference = "bipolar"
        assert session.wiring_problem() is None
    finally:
        session.device = None
        session.stop()


def test_the_spectrogram_is_of_the_summed_pair():
    """The arrangement has a stated consequence: you cannot tell from the cascade which
    channel a spindle came from."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        pump(session)
        session.push_displays(1.0)
        assert session.sum_ring.filled > 0
        centres = session.spectrogram.centres()
        assert centres[0] == 1.0 and centres[-1] == 40.0
    finally:
        session.stop()


# --- the HRV screen -------------------------------------------------------------------------

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
    finally:
        session.stop()


def test_the_hrv_screen_has_exactly_one_place_to_configure_the_feedback():
    """**The structural test the four-tab version would have failed.**

    It briefly had Train, Passive, Feedback and Session — two tabs configuring the feedback, two writers to
    the display brightness, two noise generators able to sound at once, and a metric selector on one tab that
    the other tab ignored. Per-layer tests cannot see that by construction, so this looks at the whole.
    """
    from nfb_console import hrv_view
    from nfb_console.inputs import HeartChoice

    page = StubPage()
    session = hrv_view.HRVSession(BY_CODE["HRV"], choice=HeartChoice(kind="simulator"))
    try:
        control = hrv_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        texts = [x.value for x in _texts(control)]
        shown = " ".join(texts)
        assert texts.count("OUTPUT") == 1, "one panel, on one tab — the shared one"
        assert "Passive" not in shown, "and not a second tab beside it"
        # one owner for every effector it drives, and it is the same object every other screen uses
        assert not hasattr(session, "passive_outputs")
        assert session.display is not None and session.output_volume is not None
        assert session.music is not None
        # what this screen replaces is where the level comes from, and those two rows sit above that panel
        assert "feed back" in shown and "scaled by" in shown
        # not a parallel set of renderers
        for gone in ("look at", "listen to", "the room", "ambient tint"):
            assert gone not in shown, f"{gone!r} was a mechanism reimplemented"
    finally:
        session.stop()


def test_the_hrv_pacer_says_which_kind_it_is():
    """A metronome and a cardiac-phase-locked cue are different instruments, and the screen must not let
    one pass for the other."""
    from nfb_console import hrv_view

    session = hrv_view.HRVSession(BY_CODE["HRV"])
    try:
        assert "waiting for beats" in session.pacer_note()
        session.pacer_locked = False
        assert "metronome" in session.pacer_note()
        # with no beats, the metronome still paces — that is the point of the fallback
        assert isinstance(session.breathe_in(), bool)
    finally:
        session.stop()


# --- the heart, on both screens -------------------------------------------------------------

def _heart_with(split: bool, seconds: float = 300.0):
    """A runner already fed a whole simulated session, without waiting for one."""
    from nfb.hrv import SimulatedBeats
    from nfb_console import heart

    runner = heart.HeartRunner()
    source = SimulatedBeats(rsa_hz=0.1, seed=21, realtime=False, split=split)
    start = None
    for beat in source.beats():
        runner.state = runner.processor.push(beat)
        start = start if start is not None else beat.t
        if beat.t - start > seconds:
            break
    source.close()
    return runner


def test_the_cfc_screen_shows_the_heart_beside_the_eeg():
    """The arrangement exists for one failure: the EEG can look like it is improving
    while the HRV splits. You can only see that with both in front of you."""
    from nfb_console import cfc_view

    page = StubPage()
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2,
                                  heart_runner=_heart_with(split=False))
    try:
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        pump(session)
        session.redraw()
        captions = " | ".join(c.value for c in _texts(control))
        assert "HEART" in captions
        assert "0.1 Hz" in captions
        assert session.error is None
    finally:
        session.stop()


def test_a_split_heart_warns_on_the_eeg_screen():
    from nfb_console import cfc_view

    page = StubPage()
    runner = _heart_with(split=True)
    assert runner.split, "the fixture has to actually be split"
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2,
                                  heart_runner=runner)
    try:
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        captions = " | ".join(c.value for c in _texts(control))
        assert "gas" in captions, "the split has to be said in words, not left to the curve's shape"
    finally:
        session.stop()


def test_without_a_heart_source_the_cfc_screen_says_nothing_about_it():
    """An absent instrument must not leave an empty panel implying a broken one."""
    from nfb_console import cfc_view

    page = StubPage()
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        assert session.heart is None
        assert "HEART" not in " | ".join(c.value for c in _texts(control))
    finally:
        session.stop()


def test_the_device_screen_offers_a_heart_source_and_scrolls():
    """Both of these were reported: no way to connect an HRV device, and the contact panel off-screen."""
    import flet as ft

    from nfb import Device, Settings
    from nfb_console import device_panel, heart
    from nfb_console.inputs import HeartChoice

    page = StubPage()
    choice = HeartChoice()
    # a real, unconnected Device: the panel reads more of its status than a stub is worth faking
    control = device_panel.view(page, Device(Settings()), None, choice,
                                heart.HeartRunner(choice))
    body = control.controls[1]           # tabs, then the scrolling body; no header of its own
    assert getattr(body.content, "scroll", None) == ft.ScrollMode.AUTO, "the contact panel was off-screen"

    text = " | ".join(c.value for c in _texts(control))
    for kind, _note in device_panel.HEART_KINDS:
        assert kind in text, f"no way to choose {kind}"
    assert "0.1 Hz" in text, "the pane should say what the source is for"
    assert "both in front of you" in text, "and why it also appears beside the EEG"


def test_cfc_opens_with_pink_noise_under_quiet_reward():
    """The traditional music, and the two halves fit: 1/f noise is a rendering of the target state, and
    Quiet Reward makes it recede as the trainee approaches that state."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        fb = session.feedback
        assert fb.music.enabled and fb.music.source == "noise" and fb.music.colour == "pink"
        assert fb.sound.enabled and fb.sound.invert, "the music recedes as you do better"
        assert fb.alert.enabled, "under Quiet Reward the alerts are the only arriving sound"
        assert fb.sound.high < 100, "an inverted range sits at its top when the signal is zero"
    finally:
        session.stop()


def test_a_session_opens_with_something_connected_to_the_person():
    """A screen that opens with every mechanism off says "nothing is being fed back" until someone goes
    looking for the switch. Sound and click are what every protocol wants; alert needs several targets to be
    worth distinguishing between, and music needs a choice.

    **The shadow is deliberately not in that set**, though it used to be: it takes over the operator's
    display, which is the most invasive thing this app does, and a mechanism that invasive is asked for
    rather than assumed. See `Shadow.enabled` and `shadow.MAX_SLEW`."""
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        fb = session.feedback
        assert fb.sound.enabled and fb.click.enabled
        assert not fb.shadow.enabled, "the display is asked for, not assumed"
        assert not fb.alert.enabled and not fb.music.enabled
        # a protocol default is still not an app default: the pink noise belongs to CFC
        assert fb.music.source == "noise" and not fb.sound.invert
    finally:
        session.stop()


def test_the_watched_screens_take_a_shadow_only_when_it_is_not_the_operators(monkeypatch):
    """Coupling and synchrony both feed back to somebody looking at a display, so both want a shadow — and
    "on by default" is only safe when there is a display to dim that the operator is not using.

    The trap this pins is that a shadow enabled by default has to *point* somewhere. Turning it on because a
    second monitor exists and then leaving `screen` at 1 dims the console anyway, which is the failure the
    off-by-default rule was written for, reintroduced with a safety check bolted on the front.
    """
    from nfb_console import cfc_view, ilf_view, shadow

    for code, factory in (("CFC", cfc_view.CFCSession), ("ILF", ilf_view.ILFSession)):
        # one display: the only screen is the operator's, so nothing is taken
        monkeypatch.setattr(shadow, "displays", lambda: [1])
        session = factory(BY_CODE[code], source=replay(), channels=2)
        try:
            assert not session.feedback.shadow.enabled, f"{code} dimmed the operator"
        finally:
            session.stop()

        # two: the second one is the trainee's, and that is the one the shadow lands on
        monkeypatch.setattr(shadow, "displays", lambda: [1, 2])
        session = factory(BY_CODE[code], source=replay(), channels=2)
        try:
            fb = session.feedback
            assert fb.shadow.enabled, f"{code} left a trainee's screen unused"
            assert fb.shadow.screen == 2, "pointed at the trainee's display, not the console"
        finally:
            session.stop()


def test_an_infra_slow_reward_narrows_its_own_band_and_says_what_it_cost():
    """"Infra-slow" means 0.01-1.0 Hz here, and it is not delta, which is listed
    separately. The width has to come down with the centre or the band reaches DC, and that costs delay,
    because bandwidth *is* delay."""
    from nfb.dsp import bandwidth_tau
    from nfb_console import cfc_view

    assert cfc_view.INFRA_SLOW == (0.01, 1.0)
    waves = cfc_view.WAVES
    assert waves["delta"] > waves["ultra-low"] * 10, \
        "delta and ultra-low are different frequencies, not two names for one"
    # every frequency he names as reachable, and no more
    assert set(waves) == {"ultra-low", "Mayer", "delta", "theta", "alpha", "gamma"}
    assert waves["Mayer"] == 0.1, "the 0.1 Hz Mayer wave, named in the CFC page"

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        session.retune_rewards([waves["ultra-low"] * 5, waves["gamma"]])
        slow = min(session.rewards, key=lambda t: t.band.centre)
        assert slow.band.low > 0, "a band starting at DC would be measuring the mean"
        width = slow.band.high - slow.band.low
        assert width < session.width_hz, "narrowed with the centre"
        assert 2 * bandwidth_tau(width) > 5.0, "and it is a much slower instrument for it"
    finally:
        session.stop()


def test_a_subhertz_reward_has_its_own_width_and_the_card_says_what_it_listens_to():
    """**One width could not serve four decades, and the screen was covering for that with a warning.**

    0.2 Hz around 10 is a fifth of a hertz on a band four wide; 0.2 Hz around 0.1 is twice the centre
    frequency, so it was silently clamped to `centre × 0.8` — a number from the edge of the arithmetic
    rather than anyone's judgement — while the control still read 0.2 and the only mention of it was a line
    at the foot of the panel, referring to "the lowest reward", three cards away from the card it was about.

    So a sub-hertz reward gets its own width, offered only while one exists, and every card states the band
    it is actually listening to. The clamp survives, because 0.05 Hz is impossible at 0.01 Hz too — but it
    is now a sentence about a named reward rather than a warning about an unnamed one.
    """
    import flet as ft

    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        page = StubPage()
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        shown = [x.value for x in _texts(control)]
        assert "9.9–10.1 Hz" in shown, "an ordinary reward states its band too"
        assert "subhertz width" not in shown, "no sub-hertz reward, so no sub-hertz width"

        session.retune_rewards(freqs=[cfc_view.WAVES["Mayer"], 10.0])
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        shown = [x.value for x in _texts(control)]
        assert "subhertz width" in shown
        assert "0.075–0.125 Hz" in shown, "the Mayer card says what it listens to"
        assert not any("narrowed to" in x for x in shown), "the warning it replaces is gone"
        assert session.width_for(0.1) == pytest.approx(0.05), "its own width, not the clamp"
        assert session.width_for(10.0) == pytest.approx(0.2), "and the fast reward is untouched"

        # The clamp still applies where the width cannot be granted, and it names the reward it bounded.
        session.retune_rewards(freqs=[cfc_view.WAVES["ultra-low"], 10.0])
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        shown = " ".join(x.value for x in _texts(control))
        assert session.width_for(0.01) == pytest.approx(0.008)
        assert "0.01 Hz sync is bounded to 0.008 Hz" in shown
        assert "0.006–0.014 Hz" in shown

        # And the centre box can print the number it holds: a step of 0.1 rendered 0.01 as "0.0", which
        # showed a frequency the operator had just chosen from the menu as zero.
        assert "0.01" in _numbers(control)
    finally:
        session.stop()


def _mayer_readings(seconds: float, rate_hz: float = 20.0, cycle: float = 40.0):
    """A sub-hertz synchrony reading: a level that drifts and never returns towards zero.

    Not a fitted curve — this is the shape the real one has, and the shape is the whole point. Measured on
    a 0.1 Hz reward from a real capture: 110–228 µV around a median of 175, so it swings by a third of
    itself over tens of seconds and its 5th percentile sits at 0.70 of its middle, where a Rayleigh
    envelope would put it at 0.27.
    """
    import numpy as np

    n = int(seconds * rate_hz)
    times = np.arange(n) / rate_hz
    return 175.0 * (1.0 + 0.25 * np.sin(2 * np.pi * times / cycle)
                    + 0.08 * np.sin(2 * np.pi * times / (cycle * 2.7)))


def test_a_subhertz_threshold_gets_a_long_window_and_a_live_number_throughout_it():
    """**Thirty seconds of a 0.05 Hz band is one and a half independent readings.** That is not a
    distribution, it is where one slow cycle happened to be, so the window has to be as long as the band is
    slow: `calibration_for` asks for enough readings to measure the *spread* and not merely the level.

    What must not follow from a longer window is waiting for it. The provisional derivation re-runs every
    second from the first second, so a four-minute window is a threshold that keeps improving for four
    minutes rather than four minutes of the constructed placeholder — which on this screen would be 10 µV
    against a signal in the tens, i.e. a reward met from the start whatever happens.
    """
    from nfb_console import cfc_view
    from nfb_console.targets import CALIBRATION_SECONDS

    assert cfc_view.calibration_for(10.0, 0.2) == CALIBRATION_SECONDS, "an ordinary reward is unchanged"
    assert cfc_view.calibration_for(0.1, 0.05) == pytest.approx(240.0)
    assert cfc_view.calibration_for(0.01, 0.008) == cfc_view.CALIBRATION_CAP_SECONDS, "and capped"

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        session.retune_rewards(freqs=[cfc_view.WAVES["Mayer"], 10.0])
        slow, fast = session.rewards
        assert slow.calibration_seconds == pytest.approx(240.0)
        assert fast.calibration_seconds == CALIBRATION_SECONDS
        assert slow.derivation(slow.calibration_seconds) == "spread"

        # Readings by hand: the bench recording is eight seconds of a railed constant, which is the right
        # fixture for "can this screen draw" and no use at all for a band that needs minutes.
        placeholder = slow.threshold
        for value in _mayer_readings(seconds=30.0):
            slow.observe(value, 20.0)
        assert slow.threshold != placeholder, "a threshold derived from the readings, thirty seconds in"
        assert slow.threshold > 20.0, "and near the signal rather than the constructed 10 µV"
        # A tracker now, on a 90 s period — and the period does not start until its window has filled,
        # so the first minutes are re-derived every second exactly as a fixed calibration is.
        assert slow.tracking and slow.window_remaining > 120, "provisional, with minutes left to go"
        # The buffer has to be able to hold the window it is counting down to, or the calibration would
        # count towards a number it could never reach: 200 s was more than anything needed until this.
        assert slow._recent.maxlen >= (slow.settle_seconds + slow.calibration_seconds) * 20
    finally:
        session.stop()


def test_nudging_a_reward_frequency_keeps_the_readings_it_already_has():
    """**Hunting a hailing frequency is done in small steps** — "shifts of 0.1 Hz at a time" — and each step
    rebuilds the lock-in, because a synchrony reward's estimator is built around its centre. That used to
    mean a new target with an empty window, so the calibration restarted from zero on every nudge: half a
    minute lost on a fast band, and on a sub-hertz one a four-minute derivation that could never finish.

    A centre that moves by less than half the band's width still describes the same signal, so the
    replacement inherits the window, the hit record and the threshold. A move further than that, or a change
    of width or measure, does not — those readings are not about the same thing.
    """
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        session.retune_rewards(freqs=[cfc_view.WAVES["Mayer"], 10.0])
        before = session.rewards[0]
        for value in _mayer_readings(seconds=60.0):
            before.observe(value, 20.0)
        samples, threshold, left = len(before._recent), before.threshold, before.calibration_remaining
        assert samples > 100 and threshold > 0

        session.retune_rewards(freqs=[0.11, 10.0])       # a nudge: bands overlap
        after = session.rewards[0]
        assert after is not before, "the estimator really was rebuilt"
        assert len(after._recent) == samples, "and it kept what it had measured"
        assert after.threshold == threshold
        assert after.calibration_remaining == pytest.approx(left, abs=0.1), "the countdown does not restart"

        session.retune_rewards(freqs=[0.4, 10.0])        # not a nudge: a different signal
        moved = session.rewards[0]
        assert len(moved._recent) == 0 and moved.provisional

        # A width change is not a nudge either: a different filter with a different delay.
        for value in _mayer_readings(seconds=60.0):
            session.rewards[0].observe(value, 20.0)
        assert len(session.rewards[0]._recent) > 100
        session.retune_rewards(subhertz_width_hz=0.08)
        assert len(session.rewards[0]._recent) == 0
    finally:
        session.stop()


def test_a_reward_is_labelled_by_the_band_it_is_in_not_by_a_preset():
    """The frequency is meant to be found — "often within 1-2 Hz of the frequency in the spectral display
    showing the highest amplitude in a particular band" — so the label has to say which band you are in
    without pretending you are exactly on a named value."""
    from nfb_console.cfc_view import WAVES, wave_name

    for name, hz in WAVES.items():
        assert wave_name(hz) == name
    assert wave_name(10.8) == "alpha", "a found hailing frequency is still alpha"
    assert wave_name(5.4) == "theta"
    assert wave_name(24.0) == "24 Hz", "and one nowhere near a named wave says its number"


# --- navigation -----------------------------------------------------------------------------

def _console(channels: int = 2, protocol: str = "AMP"):
    import argparse

    from nfb_console.app import Console

    # Through the real parser, never a hand-built Namespace: a hand-built one is how the console's own
    # command line drifted out of step with the Makefile and `make live` broke unnoticed.
    from nfb_console.app import build_parser
    args = build_parser().parse_args([RECORDING, "--channels", str(channels)])
    console = Console(StubPage(), args)
    console.pick(BY_CODE[protocol])
    return console


def test_stopping_a_session_leaves_you_where_you_were():
    """Stopping is something you do *to the session*, not a way of leaving the screen. The logo is the way
    out, from here as from everywhere else."""
    console = _console()
    try:
        assert console.toolbar.crumb.value == "Amplitude"
        first = console.session

        console.toolbar.stop.on_click(None)

        assert console.toolbar.crumb.value == "Amplitude", "still on the protocol screen"
        assert console.protocol is not None and console.protocol.code == "AMP"
        assert console.session is not first, "on a fresh session, since the old source is closed"
        assert console.session.paused, "and paused, ready to start"
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_back_from_the_device_screen_returns_to_the_screen_you_opened_it_from():
    """The device chip is on every screen, so "back" has to mean the screen it was pressed on. It went to
    the launcher, which is not where you came from."""
    console = _console()
    try:
        assert console.toolbar.crumb.value == "Amplitude"

        console.toolbar.dev_chip.on_click(None)         # the chip, as pressed from the session
        assert console.toolbar.crumb.value == "Device"
        # the trail names where you came from rather than saying "back" and meaning the launcher
        assert console.toolbar.parent_crumb.value == "Amplitude"
        assert console.toolbar.parent_group.visible

        console.toolbar.parent_box.on_click(None)
        assert console.toolbar.crumb.value == "Amplitude", "back to where the chip was pressed"
        assert not console.toolbar.parent_group.visible, "and the trail shortens again"
        assert console.session is not None
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_the_device_chip_from_the_launcher_still_goes_back_to_the_launcher():
    """The same control, and it has to remember two different origins."""
    import argparse

    from nfb_console.app import Console

    from nfb_console.app import build_parser
    args = build_parser().parse_args([RECORDING, "--channels", "2"])
    console = Console(StubPage(), args)
    try:
        console.show_launcher()
        assert console.toolbar.crumb.value == ""
        console.toolbar.dev_chip.on_click(None)
        assert console.toolbar.crumb.value == "Device"
        assert not console.toolbar.parent_group.visible, "the launcher *is* the location; it has no crumb"
        console.toolbar.logo.on_click(None)
        assert console.toolbar.crumb.value == "", "the logo is the way home, as everywhere else"
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_the_device_screen_has_no_header_of_its_own():
    """It carried a second row with the word "Device" in it and a back button, while the toolbar above
    already said where you were. Same mistake as the `FRE ›` row: a whole row of height for one control and
    one redundant word."""
    from nfb import Device, Settings
    from nfb_console import device_panel

    control = device_panel.view(StubPage(), Device(Settings()))
    shown = [c.value for c in _texts(control)]
    assert "Device" not in shown, "the crumb says that"
    assert not any("back" in (v or "") for v in shown), "the trail is the way back"


def test_the_contact_check_shows_the_shared_reference_once():
    """Two electrodes per channel, and under linked or external wiring every channel's "−" is the *same*
    electrode. Shown per channel that is one number repeated four times; shown once it is the thing whose
    failure mode this wiring is named for — one bad reference contaminating all four channels at once, which
    presents as everything being noisy with no channel obviously at fault.

    Bipolar gives each channel its own pair, so there both numbers belong on the channel's own row.
    """
    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb.device import Contact

    device = Device(Settings(reference="linked"))
    device.contact.update({0: Contact(plus=108.3, minus=55.6),
                           1: Contact(plus=125.0, minus=55.4)})
    control = device_panel.view(StubPage(), device)
    shown = " ".join(x.value for x in _texts(control))
    assert "108.3 kΩ" in shown and "125.0 kΩ" in shown, "each channel's own electrode"
    assert shown.count("55.") == 1, "the shared reference is one electrode, not one per channel"
    assert "shared by every channel" in shown
    assert "ref" in shown

    device = Device(Settings(reference="bipolar"))
    device.contact.update({0: Contact(plus=108.3, minus=55.6)})
    control = device_panel.view(StubPage(), device)
    shown = " ".join(x.value for x in _texts(control))
    assert "108.3" in shown and "55.6" in shown, "bipolar has two electrodes of its own per channel"
    assert "shared by every channel" not in shown
    assert "bipolar wiring" in shown, "the same electrodes read differently under each wiring"


# --- the amplitude screen is not collateral damage ------------------------------------------

def test_amplitude_defaults_are_untouched_by_the_cfc_work():
    """A fence. Everything CFC needed went into the *shared* layer, so every one of those changes was a
    change to this screen too — and the ones that are protocol opinions must not have leaked here.

    Checked rather than assumed because the failure mode is silent: the amplitude screen would still run,
    just trained differently than it did before, and nothing on it would say so.
    """
    from nfb_console import amplitude_view as av
    from nfb_console.targets import DEFAULT_MODES, HIT_RATES

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        fb = session.feedback

        # The signal: still `weighted` at 80/20. CFC *selects* `scaled`, which predates it — no combining
        # rule was added or reshaped for that protocol, which is why this fence is cheap to hold.
        assert fb.combine == "weighted"
        assert fb.reward_share == 0.80
        assert fb.normalisation == "sd"

        # thresholds: our own rates, not a budget derived from someone's twelve-bin total
        assert [t.hit_rate for t in session.targets] == [HIT_RATES[t.role]
                                                         for t in session.targets]
        assert all(t.hit_rate_override is None for t in session.targets)
        assert [t.mode for t in session.targets] == [DEFAULT_MODES[t.wants_above]
                                                     for t in session.targets]

        # the targets themselves: reward rises in alpha, inhibit rises in slow and fast
        assert [(t.band.low, t.band.high, t.direction, t.role) for t in session.targets] == [
            (8.0, 12.0, "increase", "reward"), (2.0, 6.0, "increase", "inhibit"),
            (19.0, 38.0, "increase", "inhibit")]
        assert all(t.unit == "µV" for t in session.targets)
        # **The reward is on a site and the inhibits are on the sum**, which is the asymmetry the two are
        # for: a reward is a claim about an electrode, and contamination is not site-specific, so a veto
        # watching one channel is blind to whichever input the operator is not looking at.
        reward, *inhibits = session.targets
        assert reward.pair is None and reward.source == 0
        assert all(t.source == av.PAIR and t.pair == session.sites for t in inhibits)

        # feedback: the app's own defaults, not CFC's arrangement
        assert fb.sound.enabled and fb.click.enabled
        assert not fb.shadow.enabled, "the display is asked for, not assumed"
        assert not fb.alert.enabled and not fb.music.enabled
        assert not fb.sound.invert and not fb.shadow.invert
        assert fb.music.stops_when == "never"

        # artifact rejection *is* on, which was asked for and applies everywhere
        assert session.gates.policy.enabled
    finally:
        session.stop()


def test_excluding_dead_channels_changes_nothing_while_they_are_alive():
    """The one shared change with teeth. Measured on the real two-channel export with the feed stopped so
    nothing moves between reads: identical on every read, because `active_targets` *is* `targets` when every
    channel reports signal. It only bites when a channel does not, which is the case it exists for."""
    import threading
    import time

    from nfb.sources import SampleFileSource
    from nfb_console import amplitude_view as av

    export = "/Users/paul/personal/emotional-n-back/data/eeg/Amplitude.recording"
    if not __import__("pathlib").Path(export).exists():
        pytest.skip("the sample export is not on this machine")

    session = av.AmplitudeSession(BY_CODE["AMP"],
                                  source=SampleFileSource(export, realtime=False, loop=False),
                                  channels=2)
    try:
        session.toggle_pause()
        # Both parts of what the assertions want, named: readings on every target *and* a verdict on
        # every channel. The thread version waited a tenth of a second at a time and got the second
        # condition by overshooting the first, which is luck rather than a wait.
        pump(session, until=lambda: (all(t.total for t in session.targets)
                                     and len(session.state) == 2
                                     and all(v is not None for v in session.state.values())))

        assert all(session.state.values()), "both channels should be delivering"
        assert session.active_targets == session.targets
        for _ in range(50):
            assert session.feedback.level(session.targets) == \
                session.feedback.level(session.active_targets)
    finally:
        session.stop()


def test_no_signal_reads_as_no_signal_rather_than_as_doing_badly():
    """Excluding dead channels from the output is right, and it created a new ambiguity: a level of 0% with
    nothing being output looks like failure when it means nothing is connected."""
    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        control = av.view(page, session, toolbar(), lambda: None)
        session.toggle_pause()          # targets are only observed while running
        pump(session)
        show_all_panes(control)
        session.redraw()
        # the bench recording is open inputs: every channel reports "no signal"
        assert not any(session.state.values())
        assert session.active_targets == []
        captions = " | ".join(c.value for c in _texts(control))
        # **Named, not "no channel".** Every target on the infra-low and coupling screens is measured on a
        # pair, so one bad channel makes every target dead — and the headline then claimed no channel was
        # delivering while the other drew EEG beside it. Here both really are open, so both are named.
        assert "ch0 and ch1 not delivering signal" in captions
        assert "no channel is delivering signal" not in captions
        # And it points at the device's own meaning for the flag rather than only at contact: "signal far
        # below range for a long time, or the two inputs shorted together". On the report that prompted this
        # the contact check read 14 and 26 kOhm — fine — so contact alone was the wrong place to look.
        assert "far below range" in captions and "bridged" in captions
        assert "check contact" in captions
    finally:
        session.stop()


def test_cfc_thresholds_hold_still_because_time_above_them_is_what_is_trained():
    """"The goal of this type of training is to increase the client's production (i.e., time above
    threshold) of reward 1 and reward 2." A threshold tracking a rolling window pins that percentage by
    construction, so production cannot increase and the number being trained cannot show it — the same
    criticism already recorded against our automatic mode, reached from the other direction.

    Every instrument reads Manual, with an `A → M` button that copies an
    auto-computed value into the manual one. Our calibration is that button, run once.
    """
    from nfb_console import cfc_view
    from nfb_console.targets import DEFAULT_MODES

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert all(t.mode == "fixed" for t in session.bin_targets), "the bins are Manual"
        assert all(t.mode == "automatic" and t.recompute_seconds == 90.0 for t in session.rewards), (
            "the rewards re-derive on a period and hold between")
        assert DEFAULT_MODES[True] == "automatic", \
            "and this is why it had to be said explicitly rather than inherited"
        # the first window is a phase on both: every line is provisional until it has filled once
        assert all(t.provisional for t in session.targets)
        assert all(t.can_redefine for t in session.bin_targets)

        # Both modes are on the reward cards; the bins stay Manual.
        assert "automatic" in session.threshold_modes
        control = cfc_view.view(StubPage(), session, toolbar(), lambda: None)
        assert "automatic" in dict(_pills(control)) and "fixed" in dict(_pills(control))
        assert "redefine" in " ".join(x.value for x in _texts(control)), (
            "re-deriving the bins from the last 30 s is an operator action and stays")
    finally:
        session.stop()


def test_the_amplitude_screen_still_tracks_a_reward_and_holds_an_inhibit():
    """Ours and stays ours: a target met *above* its line tracks, so it keeps rewarding as amplitude
    drifts; one met below holds still, so crossing the line means the same at minute 20 as at minute 1.

    Keyed on `wants_above` rather than on the direction, which is what makes the three AMP targets — all
    of them `increase` now, one reward and two inhibits — come out `automatic`, `fixed`, `fixed`."""
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        assert [t.direction for t in session.targets] == ["increase"] * 3
        assert {t.role: t.mode for t in session.targets} == {"reward": "automatic",
                                                             "inhibit": "fixed"}
    finally:
        session.stop()


def test_the_cascade_does_not_compress_while_it_fills():
    """The image is stretched to a fixed-height box, so emitting only the rows that had arrived meant the
    picture compressed vertically for the first twenty seconds — every row you had already read kept moving
    and shrinking as the next arrived, which is the one thing a time axis must not do."""
    from io import BytesIO

    import numpy as np

    from nfb_console.plots import Chronospectrogram

    sg = Chronospectrogram(rows=90, lo_hz=1.0, hi_hz=40.0)
    sizes = set()
    for step in range(120):
        sg.push(np.full(40, 5.0))
        if step % 17 == 0:
            head = sg.image()[:26]
            width, height = struct_unpack(head)
            sizes.add((width, height))
    assert sizes == {(40, 90)}, f"the picture changed shape as it filled: {sizes}"

    # and the unfilled part reads as the panel's own surface, not as silence at every frequency
    fresh = Chronospectrogram(rows=8, lo_hz=1.0, hi_hz=40.0)
    fresh.push(np.full(40, 5.0))
    assert fresh.seconds_filled == 1 / 8


def struct_unpack(png_head: bytes) -> tuple[int, int]:
    """Width and height out of a PNG's IHDR."""
    import struct

    return struct.unpack(">II", png_head[16:24])


def test_the_depth_is_his_five_seconds_and_changing_it_keeps_the_axis_honest():
    """"The spectral display shows the approximate voltage that was produced at every frequency from
    1-40 Hz for the last 5 seconds." The longer windows are ours; changing depth must move the row *rate*,
    not leave two time scales in one picture."""
    import numpy as np

    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        sg = session.spectrogram
        assert sg.depth_seconds == 5.0 == cfc_view.DEPTHS[0]
        assert sg.interval == pytest.approx(5.0 / 90)

        sg.push(np.full(40, 5.0))
        assert sg.filled
        sg.depth_seconds = 60.0
        sg.clear()
        assert not sg.filled, "rows sampled at the old interval cannot stay in a picture with a new axis"
        assert sg.interval == pytest.approx(60.0 / 90)
    finally:
        session.stop()


def test_the_frequency_axis_keeps_its_last_label_on_the_plot():
    """A label centred on the last gridline hangs half its width past the edge and gets clipped by the
    parent, so "40" rendered as "4"."""
    from nfb_console.plots import bin_centre_x, place_tick_labels, tick_axis

    ticks = (5, 10, 20, 40)
    _stack, labels = tick_axis(ticks, tick_width=26)
    width = 740.0
    place_tick_labels(labels, ticks, bin_centre_x(1.0, 1.0, 40, width, top=40.0),
                      tick_width=26, width=width)
    assert all(label.visible for label in labels)
    for label in labels:
        assert 0 <= label.left <= width - 26 + 0.01, (label.value, label.left)


def test_an_inhibit_is_measured_with_a_sharper_slower_filter_than_a_reward():
    """A reward drives moment-to-moment reinforcement and every millisecond comes out of a budget the
    literature puts at ~250 ms. An inhibit is a veto — 300 ms late to notice a spindle costs nothing, and
    what it needs instead is selectivity, or adjacent bins are not really separate bins.

    One elliptic design (2 biquads, 0.5 dB ripple, 60 dB stop) for every band would ignore this, so the
    design follows the role.
    """
    from nfb_console import amplitude_view as av, cfc_view
    from nfb_console.training import CFIR_DELAY

    assert CFIR_DELAY["reward"] < CFIR_DELAY["inhibit"]

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        # **By role, not by direction.** The argument above is about what the estimate is used for — a
        # reward is moment-locked reinforcement, an inhibit is a veto — and neither sentence mentions
        # which way the band is trained. A reward trained downward is still moment-locked.
        by_role = {t.role: session.latency_for(t) for t in session.targets}
        assert by_role["reward"] < by_role["inhibit"]
        assert by_role["inhibit"] == pytest.approx(0.3, abs=0.02)
    finally:
        session.stop()

    cfc = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        bins = cfc.bins[(0, cfc.band_key(cfc.bin_targets[0].band))]
        assert bins.latency_seconds == pytest.approx(0.3, abs=0.02)
    finally:
        cfc.stop()


def test_a_card_says_how_quickly_its_number_reacts():
    """Measured for the exact filter, not group delay plus smoothing — that sum is a bound, and it counts
    a smoother's whole time constant when the output is halfway there in rather less."""
    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        control = av.view(page, session, toolbar(), lambda: None)
        session.redraw()
        reacts = [c for c in _texts(control) if c.value.startswith("reacts in ")]
        assert len(reacts) == len(session.targets)
        # a reward is measured faster than an inhibit, and the numbers are the measured ones
        fast = session.response_time(session.targets[0])
        slow = session.response_time(session.targets[1])
        assert fast < slow
        # The reward is the elliptic design at order 2 with the session's envelope constant; the inhibit is
        # cFIR at a 300 ms design. Both are measured through the estimator itself rather than summed from
        # a filter's group delay and a smoother's time constant, which is the bound and reads 234 ms here.
        assert fast == pytest.approx(0.180, abs=0.03)
        assert all("half of it" in (c.tooltip or "") for c in reacts)
    finally:
        session.stop()


def _find_help(control, tooltip, depth: int = 0):
    """A panel's `?` button, by the title it opens."""
    import flet as ft

    if control is None or depth > 16:
        return None
    if isinstance(control, ft.Container) and getattr(control, "tooltip", None) == tooltip:
        return control
    for child in getattr(control, "controls", None) or []:
        found = _find_help(child, tooltip, depth + 1)
        if found is not None:
            return found
    content = getattr(control, "content", None)
    if content is not None:
        return _find_help(content, tooltip, depth + 1)
    return None


def test_the_screens_do_not_lecture():
    """A protocol screen is read for twenty minutes at a stretch, so every permanent sentence is rent
    charged to a tenant with nothing new to say after the first reading. Reasoning goes behind a `?`."""
    from nfb_console import amplitude_view as av, at_view, cfc_view, pac_view

    for module, code in ((av, "AMP"), (cfc_view, "CFC"), (at_view, "A-T"), (pac_view, "PAC")):
        page = StubPage()
        session = module.session_for(BY_CODE[code], source=replay(), channels=2)
        try:
            control = module.view(page, session, toolbar(), lambda: None)
            session.redraw()
            lines = [c.value for c in _texts(control)]
            long_ones = [x for x in lines if len(x) > 110]
            assert not long_ones, f"{code} still lectures: {long_ones}"
            assert max(len(x) for x in lines) <= 110

            joined = " ".join(lines)
            for word in (" he ", " his ", "He ", "His "):
                assert word not in joined, f"{code} names a person in the interface: {word!r}"
        finally:
            session.stop()


def test_a_rejecting_channel_says_so_where_the_channel_is():
    """Rejection is a property of the channel, so it belongs beside the trace — one of those per channel —
    and not on the cards, where several targets share a channel and it would appear three times looking
    like a property of the band."""
    import numpy as np

    from nfb.stream import Block, ChannelSpec
    from nfb_console import amplitude_view as av

    class Bursty:
        def __init__(self):
            self.spec = {0: ChannelSpec(index=0, rate_hz=250.0, unit="uV", scale=0.001)}

        def channels(self):
            return self.spec

        def close(self):
            pass

        def blocks(self):
            rng = np.random.default_rng(4)
            n = int(30 * 250)
            t = np.arange(n) / 250
            x = 25 * np.sin(2 * np.pi * 10 * t) + 3 * rng.normal(0, 1, n)
            for start in rng.integers(0, n - 60, 20):
                x[start:start + 60] += rng.normal(0, 400)
            for i in range(0, n - 7, 8):
                yield Block(0, i, tuple(int(v / 0.001) for v in x[i:i + 8]), self.spec[0])

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=Bursty(), channels=1)
    try:
        control = av.view(page, session, toolbar(), lambda: None)
        session.toggle_pause()
        pump(session)
        session.redraw()

        assert session.rejecting(0), "the fixture has to actually trip the gate"
        traces = [c.value for c in _texts(control) if "ms/div" in c.value]
        assert traces and "rejecting ch0" in traces[0], traces
        # once, beside the channel — not once per target reading that channel
        assert sum("rejecting ch" in c.value for c in _texts(control)) == 1
    finally:
        session.stop()


# --- per-target filter settings ---------------------------------------------------------------

def _find_gear(control, tooltip="Target", depth: int = 0):
    import flet as ft

    if control is None or depth > 16:
        return None
    if isinstance(control, ft.Container) and getattr(control, "tooltip", None) == tooltip:
        return control
    for child in getattr(control, "controls", None) or []:
        found = _find_gear(child, tooltip, depth + 1)
        if found is not None:
            return found
    content = getattr(control, "content", None)
    if content is not None:
        return _find_gear(content, tooltip, depth + 1)
    return None


def test_every_card_can_set_its_aim_and_the_threshold_follows_it():
    """**The aim is the threshold's cause, and it was the one number nobody could change.**

    Every derivation on every screen puts the line where `hit_rate` of the readings fall on the wanted side,
    so the aim decides the threshold — and it came from a table of defaults, or from a protocol's own
    budget, with no way in. Where that goes wrong is visible: a sub-hertz synchrony reward aiming at 97.5%
    puts its threshold below anything the signal produces, so it is met permanently and every output riding
    it stops moving.

    So it lives behind each card's gear, on every screen that has cards — with the filter rows beneath it
    where a screen has any, and alone where it does not, which is why the card owns the gear now rather than
    being handed one.
    """
    from nfb_console import amplitude_view, at_view, cfc_view, ilf_view, pac_view

    import numpy as np

    for module, code in ((amplitude_view, "AMP"), (at_view, "A-T"),
                         (cfc_view, "CFC"), (ilf_view, "ILF"), (pac_view, "PAC")):
        page = StubPage()
        session = module.session_for(BY_CODE[code], None, source=replay(), channels=2)
        try:
            control = module.view(page, session, toolbar(), lambda: None)
            gear = _find_gear(control)
            assert gear is not None, f"{code} has cards and therefore needs the aim on them"
            gear.on_click(None)
            body = " ".join(c.value for c in _texts(page.dialogs[-1].content) if c.value)
            assert "the threshold" in body and "aim" in " ".join(
                c.value for c in _texts(page.dialogs[-1].content))
        finally:
            session.stop()

    # And setting it re-derives: the old threshold is the answer to a question nobody is asking now.
    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        tgt = session.targets[0]
        for value in 20.0 + 5.0 * np.sin(np.arange(1200) / 7.0):
            tgt.observe(float(value), 20.0)
        tgt.mode = "fixed"
        tgt.redefine()
        lenient = tgt.threshold

        tgt.set_aim(0.5)
        assert tgt.hit_rate == 0.5 and tgt.aim_by_hand
        assert tgt.threshold > lenient, "asking to be met half the time raises the line"

        tgt.set_aim(2.0)
        assert tgt.hit_rate == 0.99, "both ends are degenerate and are refused"
        tgt.set_aim(0.0)
        assert tgt.hit_rate == 0.01
    finally:
        session.stop()


def test_a_hand_set_aim_survives_the_protocol_recomputing_the_others():
    """The coupling screen's rewards share the part of a session that is allowed to make a sound, so
    pressing `+ add` on its SYNCHRONY panel re-derives every reward's aim. A number somebody typed outranks
    that, the same way a nudged threshold outranks a calibration — otherwise adding a reward would silently
    undo the aim set on the others, and the gear that set it would look broken."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        first, second = session.rewards
        shared = first.hit_rate
        first.set_aim(0.80)

        session.add_reward(20.0)
        session._rebalance_rewards()
        assert first.hit_rate == 0.80, "the hand-set one is left alone"
        assert second.hit_rate != shared, "and the others still share the budget between them"

        # It also survives a nudge, which rebuilds the target around a new lock-in.
        session.retune_rewards(freqs=[first.band.centre + 0.01,
                                      second.band.centre, 20.0])
        assert session.rewards[0].hit_rate == 0.80

        # And the panel does not restate any of it: the aims are on the cards, one gear each.
        page = StubPage()
        shown = [x.value for x in _texts(cfc_view.view(page, session, toolbar(), lambda: None))]
        assert not any("met each" in x for x in shown)
    finally:
        session.stop()


def test_the_filter_is_settable_per_target_and_says_what_it_costs():
    """The trade is per target — a reward spends latency on being fast, an inhibit on being selective —
    so the setting has to be too. Behind a gear rather than on the card, which would turn a card into a
    form, and not on the panel, which would make it global to every target on it."""
    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        control = av.view(page, session, toolbar(), lambda: None)
        gear = _find_gear(control)
        assert gear is not None, "every card needs a way in"

        gear.on_click(None)
        body = " ".join(c.value for c in _texts(page.dialogs[-1].content) if c.value)
        # The aim is in here too, on every card on every screen — see the test below.
        assert "of the time at or above the threshold" in body
        # each kind exposes the parameter it actually has, not an invented "how late may this be"
        assert "butterworth" in body and "elliptic" in body
        assert "order" in body and "per edge" in body, "the IIR kinds' own parameter"
        assert "may be late" not in body, "the invented knob is gone; each kind shows its own parameter"
        # and the consequences are shown rather than chosen — measured, not a bound
        # stated as an event, not as an abstract "halfway to a change" that names no change
        assert "shows half its size after" in body, "measured, not group delay plus smoothing"
        assert "group delay" in body, "and where that time goes, or the two numbers look inconsistent"
        # dB is spelled out as the factor it is, because dB is jargon
        assert "across the band" in body and "down at" in body and "just outside it" in body
        assert "×" in body
        # envelope smoothing is one constant for the session, so it is not a per-target control here
        labels = [c.value for c in _texts(page.dialogs[-1].content) if c.value]
        assert not any(v.strip() == "smoothing" for v in labels)

        reward = session.targets[0]
        assert session.filter_for(reward) == ("elliptic", 2, 0.16, 0.5, "rms"), \
            "the default design is two biquads at 0.5 dB of ripple"
        # and the gear opens on cFIR's own parameter as soon as that kind is chosen
        session.retune_filter(reward, kind="cfir")
        page2 = StubPage()
        _find_gear(av.view(page2, session, toolbar(), lambda: None)).on_click(None)
        cfir_body = " ".join(c.value for c in _texts(page2.dialogs[-1].content) if c.value)
        assert "delay" in cfir_body and "what it is designed around" in cfir_body
        session.retune_filter(reward, kind="butterworth", order=4)
        assert session.filter_for(reward) == ("butterworth", 4, 0.16, 0.0, "rms"), \
            "0.16 s is the default band averaging once its pi scaling is undone; ripple is elliptic's " \
            "only; the amplitude convention is part of the design because thresholds do not transfer " \
            "across it"
        session.retune_filter(reward, kind="elliptic", ripple=2.0)
        assert session.filter_for(reward) == ("elliptic", 4, 0.16, 2.0, "rms"), "ripple is elliptic's own parameter"
        session.retune_filter(reward, kind="butterworth")
        assert session.latency_for(reward) > 0.15, "order 4 is slower than cFIR, and says so"
        assert len(session.metrics) == len(session.targets), "the old estimator is gone"
    finally:
        session.stop()


def test_changing_a_filter_converts_the_history_rather_than_starting_a_new_calibration():
    """Pressing a filter pill used to blank the card and count down thirty seconds.

    The reason given was that the amplitude convention changes — peak against RMS — so a window holding
    both holds neither. That is true of the *window* and false of the *readings*: they differ by one
    constant, the same one the threshold is already multiplied by, so both can be converted and stay in
    step. Losing them cost exactly the thing the pill is pressed for, which is hearing what the other
    filter does to a signal you are in the middle of.
    """
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        tgt = session.targets[0]
        for _ in range(int(35 * 20)):                      # a full calibration window at the metric rate
            tgt.observe(10.0, 20.0)
        tgt.redefine()
        assert not tgt.calibrating and tgt.recent_percent() is not None
        settled, threshold = tgt.recent_percent(), tgt.threshold

        session.retune_filter(tgt, kind="cfir")            # elliptic RMS → cFIR peak: a change of unit
        assert not tgt.calibrating, "the card must not go back to waiting"
        assert tgt.calibration_remaining == 0
        assert tgt.threshold > threshold, "the threshold is in the new convention"
        assert tgt.recent_percent() == settled, "and the readings came with it, so the score is unchanged"
        assert tgt.session_average == pytest.approx(10.0 * tgt.threshold / threshold, rel=1e-6)
    finally:
        session.stop()


def test_one_band_trained_both_ways_keeps_two_filters():
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        band = session.targets[0].band
        session.add_target(0, band, "decrease")
        up, down = session.targets[0], session.targets[-1]
        session.retune_filter(down, kind="elliptic", order=8)
        assert session.filter_for(up) != session.filter_for(down)
        assert session.key_for(up) != session.key_for(down)
        assert session.latency_for(up) != session.latency_for(down)
    finally:
        session.stop()


def test_cfc_sets_one_filter_for_all_twelve_bins():
    """A set with one job. Twelve separate choices would be twelve chances to make them incomparable."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert session.bin_delay == 0.3
        # hold the objects, not their ids: CPython reuses the address of a freed object, so comparing
        # id() after a rebuild is a coin toss
        before = dict(session.bins)
        assert all(m.delay_seconds == pytest.approx(0.3, abs=0.02) for m in before.values())

        session.retune_bins(0.5)
        assert session.bin_delay == 0.5
        assert set(session.bins) == set(before), "same bins"
        assert all(m.delay_seconds == pytest.approx(0.5, abs=0.02) for m in session.bins.values())
        assert all(session.bins[k] is not v for k, v in before.items()), "all rebuilt, none stale"
    finally:
        session.stop()


def test_switching_filters_back_and_forth_lands_where_it_started():
    """It did not. An unset order fell through as 0, so choosing butterworth built a zeroth-order filter
    with no group delay; setting an order and coming back to 2 then gave a different number for the same
    label. The reported latency also mixed the filter's group delay with an envelope smoother cFIR does
    not have, so a single figure could not be compared across kinds."""
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        tgt = session.targets[0]
        session.retune_filter(tgt, kind="butterworth")
        first = session.latency_for(tgt)
        assert session.filter_for(tgt)[1] == 2, "an unset order is 2, not 0"
        assert session.metrics[session.key_for(tgt)].delay_seconds > 0.05, "a real filter"

        session.retune_filter(tgt, order=4)
        assert session.latency_for(tgt) > first
        session.retune_filter(tgt, order=2)
        assert session.latency_for(tgt) == pytest.approx(first, abs=1e-9), "same label, same number"

        # and the two halves of that number are separable, since one of them does not exist for cFIR
        metric = session.metrics[session.key_for(tgt)]
        assert metric.delay_seconds > 0 and metric.smoothing_seconds > 0
        session.retune_filter(tgt, kind="cfir")
        assert session.metrics[session.key_for(tgt)].smoothing_seconds == 0
    finally:
        session.stop()


# --- the alpha-theta screen -----------------------------------------------------------------

def test_the_at_screen_draws_before_any_data_arrives():
    from nfb_console import at_view

    page = StubPage()
    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        control = at_view.view(page, session, toolbar(), lambda: None)
        assert control is not None
        session.redraw()
        assert session.error is None
        captions = " | ".join(c.value for c in _texts(control))
        assert "waiting for data" in captions, "an empty axis looks identical to a broken one"
    finally:
        session.stop()


def test_the_theta_minus_alpha_controls_all_fit_on_the_screen():
    """**Redefine had nowhere to appear.** The strip's last row was the scale caption, an expanding spacer,
    and the mode pills — and the caption has no width limit, so the spacer collapsed to nothing before
    anything wrapped. With `fixed` and `automatic` alone the row already reached the panel border; the third
    pill pushed it past and clipped off the edge, so a control that is meant to be there was invisible.

    Asserted through the built screen rather than on the model, because a control that exists and never
    renders is precisely the failure: `can_redefine` was right the whole time.
    """
    from nfb_console import at_view
    from nfb_console.at_view import DIFFERENCE_OBJECTIVE
    from nfb_console.feedback import CONDITION, LEVEL

    page = StubPage()
    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        control = at_view.view(page, session, toolbar(), lambda: None)
        # The strip exists only while a channel reads the difference — see `set_objective`.
        session.set_objective(LEVEL, DIFFERENCE_OBJECTIVE)
        session.set_objective(CONDITION, DIFFERENCE_OBJECTIVE)
        session.redraw()
        tgt = session.difference
        assert tgt is not None

        # Counted rather than looked for, because the theta and alpha cards carry a redefine of their own
        # and this is about the one in the objective panel.
        tgt.mode = "automatic"
        session.redraw()
        without = [x.value for x in _texts(control)].count("redefine")

        tgt.mode = "fixed"
        session.redraw()
        shown = [x.value for x in _texts(control)]
        assert shown.count("redefine") == without + 1, \
            "a fixed threshold is the one you would want to re-derive, and it has to be on the screen"
        assert "fixed" in shown and "automatic" in shown
    finally:
        session.stop()


def test_the_at_band_pills_are_on_screen_and_switching_them_redraws():
    """Read off a built panel rather than asked of the model, because the whole point of deriving the pill
    state from the bands is that the *screen* cannot go stale — and a model that can describe itself while
    nothing renders the description is the failure these tests exist for."""
    from nfb_console import at_view

    page = StubPage()
    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        control = at_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        shown = [c.value for c in _texts(control)]
        for label in ("2–7 wide", "5–8 narrow", "off", "2–5 down", "22–36", "15–36"):
            assert label in shown, f"{label} is not on the screen"

        session.set_theta_width("narrow")
        session.set_guard(True)
        session.set_high_low(15.0)
        control = at_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        shown = [c.value for c in _texts(control)]
        assert "sleep" in shown, "the guard gets its own card, named for what it watches"

        # The pill that would make the arrangement unsatisfiable is inert and carries its reason, on the
        # screen, next to itself — an inert control whose reason is somewhere else reads as a broken one.
        assert any("turn the sleep guard off first" in x for x in shown)
        wide = _by_label(control, "2–7 wide")
        assert wide.on_click is None and wide.opacity < 1.0, "inert, and visibly so"
        assert _by_label(control, "5–8 narrow").on_click is not None

        # Locked out from the other side as well, and still visible rather than hidden.
        session.set_guard(False)
        session.set_theta_width("wide")
        control = at_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        shown = [c.value for c in _texts(control)]
        assert "2–5 down" in shown, "locked is not hidden"
        assert any("needs the narrow theta band" in x for x in shown)
        assert _by_label(control, "2–5 down").on_click is None
    finally:
        session.stop()


def test_the_at_screen_draws_with_real_data_through_it():
    from nfb_console import at_view

    page = StubPage()
    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        control = at_view.view(page, session, toolbar(), lambda: None)
        session.toggle_pause()
        pump(session)
        session.redraw()
        assert session.error is None
        assert session.amplitudes, "the estimators produced nothing"
        captions = [c.value for c in _texts(control)]
        assert any("full scale" in x for x in captions), "every graph states its own scale"
        assert any("toward theta" in x for x in captions), "the music says which way it is sitting"
    finally:
        session.stop()


def test_at_opens_with_the_shipped_bands_aims_and_floor():
    """Everything here is off one data file: `fmt=CCRRI` with
    `filter=[[0,40],[0,40],[2,7],[8,11],[22,36]]` and `values=[[1,0,250],[1,0,250],[25,0,250],
    [55,0,250],[15,6,250]]` — the two raw channels are traces rather than targets, so what is left is
    theta up at 25%, alpha up at 55%, high beta down at 15% over with a 6 µV floor.

    Theta being the *harder* of the two rewards is the arrangement, not a typo: it is the band being
    brought up.
    """
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert [(t.band.low, t.band.high, t.direction) for t in session.targets] == [
            (2.0, 7.0, "increase"), (8.0, 11.0, "increase"), (22.0, 36.0, "increase")]
        assert session.theta.hit_rate == 0.25
        assert session.alpha.hit_rate == 0.55
        assert session.inhibit.hit_rate == 0.85, "15% over threshold is 85% met"
        assert session.theta.hit_rate < session.alpha.hit_rate, "theta is the one being brought up"
        # the floor, which is the one clamp the shipped plan actually uses
        assert session.inhibit.floor == 6.0
        assert session.theta.floor is None and session.alpha.floor is None
        assert session.guard is None, "the plan without a sleep guard is the default"
    finally:
        session.stop()


def test_the_band_variants_are_both_shipped_plans():
    """Its two coherent stage files differ along three axes — theta width, whether there is a guard below
    it, and where the high inhibit starts — so both ends of each choice are its numbers.

    `Stages.5` is `[[2,7],[8,11],[22,36]]` with aims `25/55/15-over`. `Stages.6i` is
    `[[5,8],[8,11],[2,5],[15,36]]` with aims `25/55/15-over/20-over`. The aim moving with the high band's
    lower edge is its own doing and is the part that would be easiest to get wrong by carrying one number
    across both.
    """
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert session.variant == ("wide", False, 22.0)

        session.set_theta_width("narrow")
        session.set_guard(True)
        session.set_high_low(15.0)
        assert session.variant == ("narrow", True, 15.0)
        assert [(t.band.low, t.band.high, t.direction, t.hit_rate) for t in session.targets] == [
            (5.0, 8.0, "increase", 0.25), (8.0, 11.0, "increase", 0.55),
            (15.0, 36.0, "increase", 0.80), (2.0, 5.0, "increase", 0.85)]
        assert session.guard.floor == 6.0, "both inhibits carry its floor"
        assert not session.overlap_warning(), "5-8 abuts the guard at 5 Hz, which is why it is narrow"

        # **The guard and the wide band are mutually exclusive, in both directions.** One direction alone is
        # a rule with a hole in it — switching the guard on under wide theta and widening theta under the
        # guard reach the same unsatisfiable state — and the refusal lives in the model, not only on the
        # pill, because "not while that" belongs where it takes effect.
        assert session.width_blocked("wide") == "turn the sleep guard off first"
        session.set_theta_width("wide")
        assert session.variant == ("narrow", True, 15.0), "refused, and it did not half-apply"

        session.set_guard(False)
        assert session.guard_blocked() == ""
        session.set_theta_width("wide")
        assert session.guard_blocked() == "needs the narrow theta band"
        session.set_guard(True)
        assert session.guard is None, "refused from the other side too"

        # The overlap check is *not* made redundant by that, because the edges stay editable: theta's lower
        # edge can be dragged under a guard that was switched on legitimately.
        session.set_theta_width("narrow")
        session.set_guard(True)
        session.retune(session.theta, low=2.0)
        assert "rewarded and inhibited at once" in session.overlap_warning()

        # Removing the guard removes the target rather than leaving it inert, and takes its estimator with
        # it — an inhibit that is present and not counted is the swallowed-refusal shape.
        session.set_guard(False)
        session.set_theta_width("wide")
        session.set_high_low(22.0)
        assert session.variant == ("wide", False, 22.0)
        assert len(session.targets) == 3
        assert len(session.metrics) == 3, "the guard's estimator is dropped, not orphaned"

        # The pills are derived from the bands, so a hand edit cannot leave them asserting a plan that is
        # no longer in force.
        session.retune(session.theta, high=6.5)
        assert session.variant[0] == "", "a hand edit lands on no variant"
    finally:
        session.stop()


def test_a_card_says_what_it_drives_and_which_way_in_one_menu():
    """**Two questions, one menu.** The card cycled increase → decrease → monitor, which made "reward" a
    synonym for "increase" and left no way to say what an alpha-down arrangement needs. Two worded chips
    said it and overran a 430 px identity row, clipping the remove button; a menu says it in the width of
    one chip, spells every combination out, and offers **monitor once, with no direction** — because a
    monitor is not trained in one, and a rule that falls out of the menu needs no control hidden.
    """
    from nfb_console import amplitude_view

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        control = amplitude_view.view(StubPage(), session, toolbar(), lambda: None)
        session.redraw()
        shown = [c.value for c in _texts(control)]
        # All three AMP targets are `increase` now; the roles are what differ. That is the shape of the
        # model — reward rises in alpha, inhibit rises in delta and in high beta.
        assert "REWARD ↑" in shown and "INHIBIT ↑" in shown, "the role is named on the card at last"

        options = [c.value for c in _menu_texts(control)
                   if c.value.startswith(("REWARD", "INHIBIT", "MONITOR"))]
        assert any(x.startswith("REWARD   ↓ decrease") for x in options), \
            "the combination the old model could not express"
        # and the one whose plain reading and whose threshold point opposite ways says so in the menu
        assert any(x.startswith("INHIBIT   ↑ increase") and "expect it to fall" in x for x in options)
        assert any(x.startswith("MONITOR") for x in options)
        assert not any(x.startswith("MONITOR") and ("increase" in x or "decrease" in x)
                       for x in options), "a monitor is offered no direction"

        tgt = session.targets[0]
        assert (tgt.role, tgt.direction) == ("reward", "increase")

        # A monitor keeps the direction it had, so coming back out of it restores rather than invents —
        # and its chip drops the arrow, because it is not being trained in a direction.
        tgt.role, tgt.direction = "reward", "decrease"
        tgt.role = "monitor"
        assert tgt.monitor and tgt.direction == "decrease"
        again = [c.value for c in _texts(amplitude_view.view(StubPage(), session, toolbar(),
                                                             lambda: None))]
        assert "MONITOR" in again and "MONITOR ↓" not in again
    finally:
        session.stop()


def test_a_down_trained_reward_is_aimed_and_filtered_as_a_reward():
    """Ros et al.'s alpha-down, built the way it is meant to be built: on the amplitude screen, by
    changing one card. It is an arrangement rather than a protocol — same screen, same estimators, same
    cards — so it has no preset and no launcher entry, and what it needed was for the arrangement to be
    expressible at all.

    Everything that follows the role follows it here, and the one thing that follows the direction still
    does: it opens `fixed`, because a tracker would follow the trainee down and erase the change.
    """
    from nfb_console import amplitude_view
    from nfb_console.targets import HIT_RATES

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        alpha, slow, fast = session.targets
        assert (alpha.band.low, alpha.band.high) == (8.0, 12.0)

        alpha.direction = "decrease"          # the one change: train the subject band downward
        assert alpha.role == "reward", "and it stays the reward, which is the whole point"
        assert alpha.hit_rate == HIT_RATES["reward"] == 0.90
        assert session.latency_for(alpha) < session.latency_for(slow), "filtered as a reward"

        fresh = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
        try:
            down = fresh.add_target(0, alpha.band, "decrease", "reward")
            assert down.mode == "fixed", "trained downward, so the threshold holds still"
            assert down.hit_rate == 0.90, "and is aimed as a reward"
        finally:
            fresh.stop()
    finally:
        session.stop()


def test_a_monitor_is_measured_and_drawn_and_reaches_no_output():
    """A fourth slot kind — `C` raw, `R` reward, `I` inhibit, `M` monitor — and alpha-theta ships a
    whole layout for it. Without it, watching a band means adding it as a target and thereby changing what is
    being trained, which is the opposite of watching.

    **A value of `role`, and it used to be a flag beside `direction`** — because a monitor still has a
    direction, a threshold and an aim, and those are exactly the things a role does not decide. Only the
    wire to the output is cut, and it is cut in one place.
    """
    from nfb_console import amplitude_view

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        watched = session.targets[0]
        assert not watched.monitor and watched.role == "reward"
        assert watched in session.active_targets

        watched.role = "monitor"
        assert watched.monitor and watched.role_label == "MONITOR"
        assert watched.verb == "INCREASE", "and it keeps its direction, which is the point of the split"
        assert watched not in session.active_targets, "it reaches no output"
        assert watched in session.targets, "and is still measured and drawn"
        # Still a full target in every other respect — this is the part a third `direction` would have lost.
        assert watched.hit_rate and watched.mode and watched.threshold

        control = amplitude_view.view(StubPage(), session, toolbar(), lambda: None)
        session.redraw()
        assert "MONITOR" in [c.value for c in _texts(control)], "and says so on its card"
    finally:
        session.stop()


def test_an_automatic_threshold_can_be_told_to_hold_still_between_re_derivations():
    """The re-derivation period, and the reason is not performance. A tracker recomputing continuously holds
    percent-over-threshold at the aim *by construction*, so the trained quantity cannot show a change; one
    that holds still for a period makes the percentage real within the period while still absorbing drift
    between them. Optional, and generic across protocols — 0 keeps exactly the old behaviour."""
    import numpy as np

    from nfb.dsp import Band
    from nfb_console.targets import Target

    def thresholds(period: float) -> list[float]:
        rng = np.random.default_rng(3)
        tgt = Target(0, Band(8.0, 12.0, ""), "increase", mode="automatic")
        tgt.recompute_seconds = period
        seen = []
        for i in range(20 * 240):                       # four minutes at 20 Hz, rising steadily
            tgt.observe(float(10 + i / (20 * 240) * 10 + rng.normal(0, 2)), 20.0)
            if i % (20 * 30) == 0:
                seen.append(round(tgt.threshold, 2))    # at 0, 30, 60 … 210 s
        return seen

    continuous = thresholds(0.0)
    held = thresholds(120.0)
    assert len(set(continuous)) == len(continuous), "a continuous tracker moves every time you look"
    # **The period starts once the window has filled.** Before that the line is re-derived every second
    # like a fixed calibration, or a 90 s period would mean 90 s on the constructed placeholder.
    assert held[1] != held[0], "provisional and moving inside the first window"
    assert held[2] == held[3] == held[4] == held[5], "held still inside the period, from 60 s"
    assert held[7] != held[2], "and re-derived across it"


def test_the_shadow_can_follow_the_state_instead_of_the_score():
    """The shadow can darken the visuals as the dominant frequency moves toward theta, which
    is our music's logic applied to the display rather than a score shown twice.

    Offered only where the protocol *has* a state, the same gate the crossfade uses: with no `state_balance` it
    would render as a brightness frozen at one end.
    """
    from nfb_console import amplitude_view, at_view, signal_pane
    from nfb_console.feedback import LEVEL as LVL, STATE

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    plain = None
    try:
        assert session.feedback.shadow.follows == LVL, "the level is the default"
        session._level = 0.9
        assert session.shadow_input() == 0.9, "the level, by default"

        # `state_balance` is None until something has been measured — the music stays where it is rather than
        # jumping to a made-up middle — so give the two rewards a reading.
        n = int(at_view.BALANCE_SECONDS / at_view.TREND_SECONDS)
        for _ in range(n):
            session.theta.trend.append(3.0)
            session.alpha.trend.append(1.0)
        session.feedback.shadow.follows = STATE
        assert session.state_balance == 0.75
        assert session.shadow_input() == 0.75, "the state, not the score"

        pane = signal_pane.SignalPane(StubPage(), session)
        shown = " ".join(x.value for x in _texts(pane.shadow_rows))
        assert "how well" in shown and "toward theta" in shown,             "the ends are named by the protocol, since only it knows what they mean"

        # A protocol with no state does not offer it, and falls back rather than freezing at one end.
        plain = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
        assert plain.state_ends is None
        flat = signal_pane.SignalPane(StubPage(), plain)
        assert "how well" not in " ".join(x.value for x in _texts(flat.shadow_rows))
        plain.feedback.shadow.follows = STATE
        plain._level = 0.4
        assert plain.shadow_input() == 0.4, "no state to follow, so the level"
    finally:
        session.stop()
        if plain is not None:
            plain.stop()


def test_the_reward_mark_doubles_once_the_condition_has_held():
    """Two messages on one channel, not two mechanisms.

    A momentary reward event and a three-second sustained one can sit side by side, so marking arrival
    and marking persistence are both real. But the sustained one *replaces* the momentary one — two
    independent streams would overlap and muddy both, and this channel's design rests on being one bit at a
    fixed rate. So one enabled, one volume, one interval, and what changes after `hold_seconds` is the sound.

    **And the criterion is a duty cycle, not an unbroken run**, which measurement forced: on a real
    two-channel export the raw condition flickers at about 3 Hz — median run 0.15 s, longest 0.85 s — so a
    continuity test can never fire. See `HOLD_FRACTION`.
    """
    from nfb_console import amplitude_view, cues
    from nfb_console.feedback import HOLD_FRACTION

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    played: list = []
    session.cue = lambda name, vol=None: played.append(name)
    try:
        for tgt in session.targets:
            tgt.total, tgt.threshold = 100, 1.0
            tgt.value = 5.0 if tgt.wants_above else 0.0
        session.state = {0: True}
        assert session.feedback.on_target(session.channel_targets("condition"))

        click = session.feedback.click
        click.enabled = True
        click.interval_seconds = 0.0

        def feed(now: float, met: bool) -> None:
            """One observation at a chosen time, so the window can be filled without waiting for it."""
            session._condition_log.append((now, met))

        # Zero never changes character, which is what every screen did before this existed.
        click.hold_seconds = 0.0
        session._maybe_click()
        assert played == [click.sound] and not session.click_held()

        # A full window is required before it can be true, so it cannot appear in the first two samples.
        click.hold_seconds = 3.0
        session._condition_log.clear()
        feed(100.0, True)
        feed(100.5, True)
        assert not session.click_held(now=101.0), "not enough history to say"

        # A window that is mostly met doubles it — a brief drop does not reset it.
        session._condition_log.clear()
        for i in range(30):
            feed(100.0 + i * 0.1, i not in (7, 19))
        assert session.click_held(now=103.0), "27 of 30 met is well over the fraction"

        # A window that is mostly missed does not.
        session._condition_log.clear()
        for i in range(30):
            feed(100.0 + i * 0.1, i % 2 == 0)
        assert not session.click_held(now=103.0), "half met is under it"
        assert 0.5 < HOLD_FRACTION < 1.0

        # And the sound that comes out is the doubled one, on the same stream. Fed from *before* the window
        # so the fullness check can see history older than it — the trim keeps a margin for exactly this, and
        # trimming to the window alone made the criterion permanently unreachable.
        import time as clock

        now = clock.monotonic()
        session._condition_log.clear()
        for i in range(40):
            feed(now - 3.9 + i * 0.1, True)
        played.clear()
        session._last_click = None
        session._maybe_click()
        assert played and played[0] == cues.held(click.sound), played
        assert cues.click_path(played[0]) is not None
    finally:
        session.stop()


def test_a_screen_that_can_double_the_mark_says_so_and_one_that_cannot_carries_no_hold():
    """**The doubling arithmetic runs on every session; the row that configures it does not.** So a screen
    that leaves `sustains_reward` unset must not be able to hold a non-zero `hold_seconds` — otherwise the
    mark changes character with nothing on the Feedback tab saying why or letting anyone turn it off, which
    is what a preferences file written by an older build would have restored.

    The amplitude screen is on the other side of that line, and deliberately: its condition is three targets
    met at once, so arriving and having stayed are different things and both are worth a sound.
    """
    from nfb_console import amplitude_view, cfc_view, signal_pane

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        assert session.sustains_reward
        pane = signal_pane.SignalPane(StubPage(), session)
        assert "doubles after" in " ".join(x.value for x in _texts(pane.click_rows))

        session.restore_calibration({"click.hold_seconds": 3.0})
        assert session.feedback.click.hold_seconds == 3.0, "settable here, so it is restored here"
    finally:
        session.stop()

    other = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert not other.sustains_reward
        pane = signal_pane.SignalPane(StubPage(), other)
        assert "doubles after" not in " ".join(x.value for x in _texts(pane.click_rows))

        other.restore_calibration({"click.hold_seconds": 3.0})
        assert other.feedback.click.hold_seconds == 0.0, "no control for it, so no state behind it"
    finally:
        other.stop()


def test_a_held_mark_is_two_of_the_same_sound_rather_than_a_new_one():
    """Rhythm and not pitch, because the pitch is already carrying which band earned it. So the doubled form
    is the base sound twice with a gap, which is why it needs no second pitch and no calibration."""
    import wave as wavelib

    from nfb_console import cues

    base = cues.tone_at(137)
    doubled = cues.held(base)
    assert doubled != base
    assert cues.held(doubled) == doubled, "idempotent"
    assert cues.held("nonesuch") == "nonesuch", "unknown names pass through untouched"
    assert doubled not in cues.click_choices(), "the menu of characters is not where these go"

    one = cues.duration_ms(base)
    both = cues.duration_ms(doubled)
    assert both == pytest.approx(one * 2 + cues.HOLD_GAP_MS)
    with wavelib.open(str(cues.click_path(doubled))) as fh:
        assert fh.getnframes() == pytest.approx(cues.RATE * both / 1000, rel=0.02)

    # It works for a click as well as a tone, since a protocol without an earning target still has one.
    assert cues.duration_ms(cues.held("tick")) > cues.duration_ms("tick")


def test_the_sleep_guard_has_three_responses_and_silence_is_the_ambiguous_one():
    """The band is half the design; what the trainee perceives is the other half, and with the eyes closed
    it is the whole of it. There are three ways to answer it — make it an ordinary
    inhibit so the tones merely stop, play "a knock on the door", or play noise and
    fade everything else out.

    Silence is the default because it is what the guard did with no extra code, and it is also the one that
    cannot be told from "neither reward is met" — which calls for the opposite correction.
    """
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    played: list = []
    session.cue = lambda name, vol=None: played.append(name)
    try:
        session.set_theta_width("narrow")
        session.set_guard(True)
        assert session.guard_response == "silence"
        assert at_view.GUARD_RESPONSES == ("silence", "knock", "knock + hush")

        # Not over its threshold yet, and never fed: `guard_over` needs *data*, or the music would hush and
        # the knock would start the moment the screen opened, since a `decrease` target wants zero.
        assert not session.guard_over
        session.guard.total = 100
        session.guard.threshold, session.guard.value = 5.0, 99.0
        session.state = {0: True}
        assert session.guard_over

        session._maybe_alert()
        assert not played, "silence plays nothing"
        assert not session.music_hold(False), "and leaves the music alone"

        session.set_guard_response("knock")
        session._maybe_alert()
        assert played, "the knock is the sound that can only ever be the guard"
        assert not session.music_hold(False), "but it does not touch the music"
        # Derived from the guard's own band, so it is a low thock rather than a chosen asset — and audibly
        # different from what a fast inhibit would sound.
        from nfb_console import cues
        assert played[-1] == cues.frequency_click(session.guard.band.centre)
        assert played[-1] != cues.frequency_click(28.0)

        session.set_guard_response("knock + hush")
        assert session.music_hold(False), "the music drops so the knock arrives into silence"
        session.guard.value = 1.0
        assert not session.guard_over and not session.music_hold(False), "and comes back"
    finally:
        session.stop()


def test_the_two_objectives_can_be_mixed_across_the_two_output_channels():
    """θ−α is a different claim from theta-or-alpha, not a rearrangement of it: it cannot be satisfied by
    everything getting bigger, and it cannot be satisfied by strong alpha alone.

    They are chosen per output channel because the channels ask different questions — a fade wants how far
    toward the state, a tone wants being in one of them now. The inhibits are in both sets either way, since
    "too awake" is a condition on the state rather than on a description of it.
    """
    from nfb_console import at_view
    from nfb_console.feedback import CONDITION, LEVEL

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert session.difference is None, "built only while something reads it"
        assert session.channel_targets(LEVEL) == session.channel_targets(CONDITION)

        session.set_objective(LEVEL, "difference")
        assert session.mixed
        assert session.channel_targets(LEVEL) == [session.difference, session.inhibit]
        assert session.channel_targets(CONDITION) == [session.theta, session.alpha, session.inhibit]

        # **No band, and that is what says it gets no card.** It carried a nominal 2-11 Hz span for one
        # commit, which put two inert edge boxes and a filter dialog beside a threshold that worked.
        assert session.difference.band is None
        assert not session.difference.banded
        assert session.difference.colour is None, "colour is per frequency; this has none"
        assert session.difference.label == "ch0 θ−α"
        # The percentile route, now for the right reason rather than because the fake band was 9 Hz wide.
        assert session.difference.derivation() == "percentile"

        # Derived, not measured: its own key, no estimator, and its value is the two subtracted.
        assert session.key_for(session.difference) == ("difference",)
        assert len(session.metrics) == 3, "no estimator is built for it"
        session.amplitudes[session.key_for(session.theta)] = 12.0
        session.amplitudes[session.key_for(session.alpha)] = 7.0
        assert session.value_for(session.difference) == 5.0
        # No fresher than its stalest half.
        assert session.latency_for(session.difference) == max(
            session.latency_for(session.theta), session.latency_for(session.alpha))

        # It is observed on theta's cadence, because nothing puts its key in `fresh` — left to the base
        # class every window it has would stay empty while its card showed a threshold.
        assert session._due(session.difference, {session.key_for(session.theta)}) is True
        assert session._due(session.difference, set()) is False

        # Switching both channels back drops the target rather than leaving it measured and unread.
        session.set_objective(LEVEL, "bands")
        assert session.difference is None and len(session.targets) == 3
    finally:
        session.stop()


def test_the_objective_is_on_screen_and_says_what_the_pairing_means():
    """Four pairings, four different sentences. A description that does not depend on the mode is the
    failure the combining-rule captions were keyed for, and the mixed cases are the whole reason there are
    two rows rather than one."""
    from nfb_console import at_view
    from nfb_console.feedback import LEVEL

    page = StubPage()
    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        control = at_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        shown = [c.value for c in _texts(control)]
        assert "either band" in shown and "θ−α" in shown
        assert any("alternatives" in x for x in shown)

        session.set_objective(LEVEL, "difference")
        control = at_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        shown = [c.value for c in _texts(control)]
        assert any("how far toward theta" in x for x in shown), "the mixed pairing has its own line"

        # **A strip in the panel where the objective was chosen, not a card in the target list.** The cards
        # are the four bands and nothing else; the difference appears where its threshold is set.
        # Counted by the role-and-direction chip, which appears on a card's identity row and nowhere
        # else — the role words themselves are also the BANDS row labels.
        chips = [c.value for c in _texts(control)
                 if c.value in ("REWARD  ↑", "REWARD  ↓", "INHIBIT  ↑", "INHIBIT  ↓")]
        assert chips == ["REWARD  ↑", "REWARD  ↑", "INHIBIT  ↑"], "three cards, no fourth for θ−α"
        assert "θ−α" in shown, "it is on screen, in the objective panel"
        # Built for a signed quantity: zero is the crossover, and the scale says so.
        assert any("0 = crossover" in x for x in shown)
        assert any(x.startswith("±") and "µV" in x for x in shown), "a symmetric range, not 0-20"

        assert len(set(at_view.OBJECTIVE_NOTES.values())) == 4, "every pairing reads differently"
    finally:
        session.stop()


def test_alpha_theta_keeps_its_variant_and_its_names_across_a_stop():
    """`adopt_targets` replaces the target list wholesale, which left `theta` / `alpha` / `inhibit` / `guard`
    pointing at the objects the new session built and discarded. Those orphans are never fed, so the
    crossover plot and the ratio would read empty trends for ever and the tone bias would compare the wrong
    pair — all while the cards looked right, because they are drawn from the list rather than the names."""
    from nfb_console import at_view

    first = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    second = None
    try:
        first.set_theta_width("narrow")
        first.set_guard(True)
        first.set_high_low(15.0)
        first.theta.threshold = 9.0

        second = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
        second.adopt(first)

        assert second.variant == ("narrow", True, 15.0)
        assert second.theta in second.targets and second.guard in second.targets
        assert second.alpha in second.targets and second.inhibit in second.targets
        assert second.theta.threshold == 9.0, "the configuration survives; only the history is cleared"
        assert not second.theta.total
        assert len(second.metrics) == 4, "every adopted target has an estimator behind it"
    finally:
        first.stop()
        if second is not None:
            second.stop()


def test_a_threshold_floor_survives_calibration_and_a_nudge():
    """Without it one quiet stretch drives an inhibit's threshold toward zero and it then fires for the
    rest of the session. It has to hold wherever the threshold is set, including by hand — a nudge past
    the floor would otherwise survive until the next recalibration and be silently undone by it."""
    import numpy as np

    from nfb_console.targets import Target
    from nfb.dsp import Band

    tgt = Target(0, Band(22.0, 36.0, ""), "decrease", mode="fixed", floor=6.0, ceiling=250.0)
    rng = np.random.default_rng(7)
    for _ in range(700):                        # a very quiet channel
        tgt.observe(abs(rng.normal(0, 0.3)), 20.0)
    assert tgt.threshold == 6.0, "the derived number was below the floor"

    tgt.nudge(0.5)
    assert tgt.threshold == 6.0, "and a hand-set number cannot go under it either"

    unbounded = Target(0, Band(22.0, 36.0, ""), "decrease", mode="fixed")
    for _ in range(700):
        unbounded.observe(abs(rng.normal(0, 0.3)), 20.0)
    assert unbounded.threshold < 1.0, "the default is still no clamp at all"


def test_at_rewards_are_alternatives_rather_than_conditions():
    """Theta and alpha are two states you can be in one of, so an AND over them is unsatisfiable in
    principle and an average asks for both at once. `either` is the max over rewards, still scaled down by
    the inhibits — and it is the one rule that reaches the discrete mechanisms, because "earned" under it
    means one reward rather than all of them."""
    from nfb_console import at_view
    from nfb_console.feedback import FeedbackSettings, combine

    class Fake:
        def __init__(self, role, value, wanted):
            self.role, self._v, self.wanted = role, value, wanted
            self.direction = "increase" if role == "reward" else "decrease"

        def score(self, scale_from="sd", factor=1.0):
            return self._v

    theta, alpha = Fake("reward", 0.95, True), Fake("reward", 0.05, False)
    clear = Fake("inhibit", 0.9, True)
    assert combine([theta, alpha, clear], "either") > combine([theta, alpha, clear], "scaled")
    assert combine([theta, alpha, clear], "either") > 0.5, "one reward firmly met is not half a signal"
    # the inhibit is still a veto on the level, whichever reward is earning
    firing = Fake("inhibit", 0.02, False)
    assert combine([theta, alpha, firing], "either") < 0.2

    assert FeedbackSettings(combine="either").on_target([theta, alpha, clear]), "one is enough"
    assert not FeedbackSettings(combine="either").on_target([theta, alpha, firing])
    assert not FeedbackSettings(combine="weakest").on_target([theta, alpha, clear]), \
        "and every other rule still wants all of them"

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert session.feedback.combine == "either"
    finally:
        session.stop()


def test_the_at_reward_tone_says_which_band_earned_it():
    """Its two reward files are named for the frequencies they stand for and differ in pitch by more than
    two octaves — measured, 137 Hz against 738. So the tone is keyed to the earning band, and theta wins
    when both are available, which is its own default with `unbiasedreward` off."""
    from nfb_console import at_view
    from nfb_console.cues import reward_tone, tone_pitch

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert session.feedback.click.keyed
        assert session.earning_target() is None, "nothing met, nothing to sound like"

        session.state[session.site] = True
        session.theta.threshold = session.alpha.threshold = 10.0
        session.theta.value, session.alpha.value = 20.0, 20.0
        assert session.earning_target() is session.theta, "theta bias, when both are available"

        session.theta.value = 1.0
        assert session.earning_target() is session.alpha, "and alpha when only alpha is met"

        session.theta.value, session.alpha.value = 12.0, 30.0
        assert session.earning_target() is session.theta, "still theta, because the bias ignores by how much"
        session.unbiased = True
        assert session.earning_target() is session.alpha, "unbiased takes the better-scoring one"

        # A **tone** per band, not a click, and far enough apart to be told from each other with the
        # eyes closed: 137 Hz and 738 Hz are our anchors.
        low = tone_pitch(session.theta.band.centre)
        high = tone_pitch(session.alpha.band.centre)
        assert (low, high) == pytest.approx((137.0, 738.0)), "the anchors are the two measured files"
        assert high / low > 4, "under two octaves apart is what the alert's noise bursts gave"
        assert reward_tone(session.theta.band.centre) == "tone@137"
    finally:
        session.stop()


def test_at_thresholds_hold_still_by_default_but_automatic_is_offered():
    """`fixed` is the default for a real reason and `automatic` is available anyway, which is the line
    between a default and a requirement.

    The reason for the default: its own plan ships `autoset=0`, and a tracker follows theta up as it rises
    and pins the hit rate where it started — which costs more here than elsewhere, because the pitched tone
    is the whole discrete channel and its density is the only progress signal a trainee with closed eyes
    gets. The reason it is not locked: automatic thresholds are a legitimate choice on this protocol,
    including a θ−α design at 75%, so locking one mode would be a judgement. `threshold_modes` is for what a protocol requires — CFC keeps its lock because
    the quantity it trains *is* time above threshold, which a tracker pins by definition.
    """
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert all(x.mode == "fixed" for x in session.targets), "the default"
        assert session.threshold_modes == ("fixed", "automatic")
        control = at_view.view(StubPage(), session, toolbar(), lambda: None)
        pills = dict(_pills(control))
        assert "automatic" in pills and "fixed" in pills, "offered, not stated"
        assert "redefine" in " ".join(x.value for x in _texts(control))

        # It reaches the θ−α strip too, which is a target's controls in a different shape rather than a
        # different kind of target.
        session.set_objective(LEVEL_CH, "difference")
        control = at_view.view(StubPage(), session, toolbar(), lambda: None)
        session.redraw()
        assert session.difference.mode == "fixed"
        session.difference.mode = "automatic"
        session.redraw()
        assert "automatic" in dict(_pills(control))
    finally:
        session.stop()


def test_the_music_is_generated_with_headroom_so_its_control_has_somewhere_to_go():
    """Broadband noise is far louder than its full-scale number suggests — energy at every frequency at
    once — so music rendered at full scale needed a volume of about 5 to sit in the background, and 5 is a
    control with nowhere left below it.

    Padding the *source* is what buys the control its range back. The pad is on the noise and not on
    `Music.volume`, because a track's loudness is whatever was mastered into it. The preview carries the same
    pad, or the audition would be louder than the music it previews.

    **The pad is the generator's and the volume is the operator's, and they do not move together.** The pad
    went three times deeper because the noise engine was three times too loud in use; every screen that
    generates noise was getting that engine, so every generated sound got quieter at the same setting. A volume somebody
    set by ear stays where they set it — "compensating" it here would have undone the fix on this screen.
    """
    import wave as wavelib

    import numpy as np

    from nfb_console import at_view, music

    with wavelib.open(str(music.preview_path("pink"))) as fh:
        data = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(float) / 32767
    body = data[len(data) // 4:-len(data) // 4]
    rms = float(np.sqrt((body ** 2).mean()))
    assert rms == pytest.approx(music.PEAK * music.HEADROOM, rel=0.1), "the preview carries the pad"
    # The control still has room under it, which is the whole reason to pad the source: at 20 the music is well
    # above the bottom of the range, so it can be halved twice and still be heard.
    assert 5.0 < at_view.MUSIC_VOLUME < 50.0

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert session.feedback.music.volume == at_view.MUSIC_VOLUME == 20.0
        # The tone is a mastered file scaled by its own number, where the music is a pad we control, so the two
        # volumes are not on one scale. Pinned because they *were* compared as though they were.
        assert session.feedback.click.volume == 25.0
    finally:
        session.stop()


def test_the_at_music_fades_between_two_characters_of_noise():
    """The continuous feedback carries the *state*, not a score: two soundscapes fading into each other as
    the balance moves. Nothing in its design gets louder as the trainee does better, which is why the
    master volume and the shadow are off here."""
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        music = session.feedback.music
        assert music.enabled and music.source == "crossfade"
        assert {music.colour, music.second_colour} == {"pink", "brown"}
        assert not session.feedback.sound.enabled, "a score on the music's loudness is not in the design"
        assert not session.feedback.shadow.enabled, "eyes closed: a closed channel"
        assert not session.feedback.sound.invert, "and not Quiet Reward either, which is another protocol"

        # **The balance is averaged over `BALANCE_SECONDS`, not read off the instantaneous envelopes.** Two
        # narrowband envelopes flicker through an order of magnitude, so music following them swaps ends
        # several times a second on a state that moves over half an hour — under a caption that was already
        # averaged. A 30 s fade-over period suits the crossfade.
        assert session.state_balance is None, "nothing measured yet leaves the music where it is"

        def hold(theta: float, alpha: float) -> None:
            """Fill both trends, since the music now reads a window rather than a sample."""
            n = int(at_view.BALANCE_SECONDS / at_view.TREND_SECONDS)
            session.theta.trend.clear()
            session.alpha.trend.clear()
            for _ in range(n):
                session.theta.trend.append(theta)
                session.alpha.trend.append(alpha)

        hold(0.0, 10.0)
        assert session.state_balance == 0.0
        hold(10.0, 10.0)
        assert session.state_balance == pytest.approx(0.5), "the crossover is the middle of the fade"
        hold(10.0, 0.0)
        assert session.state_balance == 1.0

        # A single spike cannot move it far, which is the whole point of the window.
        hold(10.0, 10.0)
        session.theta.trend.append(200.0)
        assert session.state_balance == pytest.approx(0.5, abs=0.08)

        # **There is one quantity, not a choice of two.** A `share`/`difference` pair of pills was removed
        # because θ−α normalised by the only scale available is `(θ−α)/(θ+α)`, which is identically
        # `2·share − 1` — so the second option computed an affine function of the first and its label had to be
        # cryptic because there was no real distinction to name.
        assert not hasattr(session, "music_source")
        hold(15.0, 5.0)
        share = session.state_balance
        assert share == pytest.approx(0.75)
        assert (15.0 - 5.0) / (15.0 + 5.0) == pytest.approx(2 * share - 1)
    finally:
        session.stop()


def test_music_is_offered_only_where_it_means_something():
    """A control that cannot work on the screen it is on is worse than a missing one: it invites a choice
    and leaves the operator to find out. `crossfade` needs two ends standing for two states and a protocol
    that knows which state you are in, so it belongs to alpha-theta alone — and alpha-theta's whole
    continuous channel *is* that fade, so a fixed tilt or a backing track would be three ways to switch the
    feedback off."""
    from nfb_console import amplitude_view, at_view, cfc_view
    from nfb_console.feedback import MUSIC_SOURCES

    at = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert at.music_sources == ("crossfade",)
        assert at.feedback.music.source in at.music_sources
        assert at.state_ends == ("alpha", "theta"), "low end first, matching colour/second_colour"
        assert at.state_ends[1] == "theta" and at.feedback.music.second_colour == "brown"
    finally:
        at.stop()

    for module, code in ((amplitude_view, "AMP"), (cfc_view, "CFC")):
        session = module.session_for(BY_CODE[code], source=replay(), channels=1)
        try:
            assert "crossfade" not in session.music_sources, f"{code} has no state to fade between"
            assert session.state_ends is None
            assert set(session.music_sources) <= set(MUSIC_SOURCES)
            assert session.feedback.music.source in session.music_sources
        finally:
            session.stop()


def test_the_signal_panel_names_both_channels_and_their_followers():
    """The fork has to be *on the screen*, not only in the model.

    Two things here were documented as printed and were rendered nowhere — `pipeline()` and the old
    `means_at()` — so a test that only asks the model whether it can describe itself is not enough. This
    one builds the panel, pushes real data through and reads the text off it. It also checks the level
    row's label is not "driven by": that wording is what made one channel look like all of them.
    """
    from nfb_console import amplitude_view, signal_pane

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        pane = signal_pane.SignalPane(StubPage(), session)
        session.feedback.sound.enabled = session.feedback.click.enabled = True
        session.feedback.shadow.enabled = session.feedback.alert.enabled = False
        pane.rebuild()
        rows = " ".join(x.value for x in _texts(pane.signal_rows))
        assert "level from" in rows and "driven by" not in rows
        assert "→ level" in rows and "→ condition" in rows

        # The bench recording rails and reports no signal on every channel, so drive the two lines
        # directly: what is being pinned is that each channel prints its own value and its own
        # followers, which is the thing that was missing.
        pane._refresh_fork(live=True, level=0.62, met=True)
        assert pane._level_line.value == "62% → sound"
        assert pane._condition_line.value == "met → click"

        session.feedback.click.enabled = False
        pane._refresh_fork(live=True, level=0.62, met=False)
        assert pane._condition_line.value == "no output on this channel", (
            "a channel nothing follows is the state worth naming")
        session.feedback.click.enabled = True
        pane._refresh_fork(live=False, level=0.0, met=False)
        assert pane._level_line.value == "— → sound", (
            "no data is not a level of zero — but who follows the channel still shows, since a paused "
            "screen is exactly when this is read")
    finally:
        session.stop()


def _pills(control, depth: int = 0):
    """Every pill in a tree, as (label, live). A pill is a Container holding one Text; a live one has a
    click handler and a dimmed one does not."""
    import flet as ft

    if control is None or depth > 20:
        return []
    out = []
    if isinstance(control, ft.Container) and isinstance(control.content, ft.Text) \
            and control.border_radius == 6 and control.padding is not None:
        out.append((control.content.value, control.on_click is not None))
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            out += _pills(child, depth + 1)
    if not isinstance(control, ft.Container) or not isinstance(control.content, ft.Text):
        out += _pills(getattr(control, "content", None), depth + 1)
    return out


def _chosen(control) -> set:
    """Which pills read as selected, by the accent background `widgets.pill` gives the chosen one."""
    from nfb_console import theme

    out = set()
    for label, _live in _pills(control):
        out.add(label)
    picked = set()

    def walk(node, depth=0):
        import flet as ft
        if node is None or depth > 14:
            return
        if isinstance(node, ft.Container) and isinstance(node.content, ft.Text) \
                and node.bgcolor == theme.SURFACE_HI and node.border_radius == 6:
            picked.add(node.content.value)
        for attr in ("controls", "shapes"):
            for child in getattr(node, attr, None) or []:
                walk(child, depth + 1)
        if not isinstance(node, ft.Container) or not isinstance(node.content, ft.Text):
            walk(getattr(node, "content", None), depth + 1)

    walk(control)
    return picked & out


def test_stopping_ends_the_session_and_keeps_the_setting_up():
    """**Stop is not a factory reset.** It ends the session and re-enters the protocol on a fresh one,
    which is right — but a fresh one was built from the protocol's defaults, so training SMR and pressing
    stop put the band back to 8-12 Hz, and with it every volume, the level rule, the session length and the
    artifact policy.

    The line is the one the suspend/rearm work drew: the session *is* the history, the configuration is
    not. So the targets come across with their statistics cleared, and each rebuilds its estimator —
    carried across with no metric behind it, a target would read zero for ever.
    """
    from nfb_console.app import Console, build_parser
    from nfb_console.protocols import BY_CODE

    console = Console(StubPage(), build_parser().parse_args([RECORDING]))
    console.pick(BY_CODE["AMP"])
    session = console.session
    tgt = session.targets[0]
    tgt.set_band(12.0, 15.0)
    tgt.nudge(1.3)
    session.feedback.click.volume = 42.0
    session.feedback.combine = "weakest"
    session.plan.set_length(20)
    session.gates.policy.enabled = False
    session.gain_uv = 25.0
    threshold = tgt.threshold
    for _ in range(40):
        tgt.observe(14.0, 20.0)
    assert tgt.total

    console.restart()
    fresh = console.session
    assert fresh is not session, "it really is a new session"
    kept = fresh.targets[0]
    assert (kept.band.low, kept.band.high) == (12.0, 15.0), "the band an operator set is not a default"
    assert kept.threshold == threshold and kept.mode == "fixed"
    assert fresh.feedback.click.volume == 42.0 and fresh.feedback.combine == "weakest"
    assert fresh.plan.length_seconds == 1200.0
    assert fresh.gates.policy.enabled is False
    assert fresh.gain_uv == 25.0
    assert kept.total == 0 and not kept.trend, "the history belonged to the session that ended"
    assert fresh.key_for(kept) in fresh.metrics, "and it is measured again"
    fresh.stop()


def test_a_protocol_owns_its_arrangement_and_the_operator_owns_the_calibration():
    """The Feedback tab was the wrong tenant on a specialty screen: it offered every choice on every
    protocol, including the ones the protocol depends on. Switching CFC's level rule from `scaled` to
    `weighted` breaks the interlock between its bins and its rewards, and nothing said so.

    The cut is by *ownership*: conditioning and calibration stay live everywhere, arrangement locks.
    **Locked is not hidden** — every pinned row still shows what was chosen, that alternatives exist,
    and why it cannot be changed, because an inert control with no reason reads as a broken one.
    """
    from nfb_console import at_view, signal_pane

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        pane = signal_pane.SignalPane(StubPage(), session)
        rows = " ".join(x.value for x in _texts(pane.signal_rows))
        assert "either" in rows, "the rule it chose still shows"
        assert "set by the protocol" in rows, "and an inert row has to say why"
        live = dict(_pills(pane.signal_rows))
        assert "either" not in live, "one option is stated rather than offered, as everywhere else"
        assert "scaled" not in live, "and the rules it does not offer are not on the screen"
        assert live["sd"] is True, "scale from is conditioning, and stays the operator's"

        for rows_control in (pane.sound_rows, pane.click_rows, pane.music_rows):
            pills = dict(_pills(rows_control))
            for label in ("sound on", "click on", "music on"):
                if label in pills:
                    assert pills[label] is False, f"{label} is the arrangement's"

        # Calibration is untouched: the volumes, the interval and the auditions are the room, not the
        # design, and this screen is run with the eyes closed — the first hearing would otherwise be the
        # one moment nothing can be adjusted.
        # And it is called a **tone** here, because that is what it plays: keyed, the sound comes from
        # `cues.reward_tone`, and a tab reading Click over a speaker playing a tone is the kind of copy
        # that makes a screen untrustworthy. The character menu is replaced by one audition per reward,
        # since with the pitch coming from the band there is no character left to choose.
        click = " ".join(x.value for x in _texts(pane.click_rows))
        assert "one tone every" in click and "volume" in click
        assert "click" not in click, click
        assert click.count("▶") == 3, "both tones, plus the held mark, each auditionable"
        assert "Tone" in [x.value for x in _texts(pane.tabs.control)]

        # And the `?` explains the tone rather than the click. It was one static string written for
        # clicks, under a heading reading Tone, which is the reader's first clue that a screen is not to
        # be trusted.
        helper = _find_help(pane.control, "Tone")
        assert helper is not None
        helper.on_click(None)
        body = shown(pane.page.dialogs[-1])
        assert body.startswith("A tone sounds at a fixed interval")
        assert "A click sounds at a fixed interval" not in body
        assert "pitch says which reward earned it" in body
    finally:
        session.stop()

    # And the amplitude screen is unchanged: nothing is pinned, so everything is live.
    from nfb_console import amplitude_view

    amp = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        pane = signal_pane.SignalPane(StubPage(), amp)
        assert amp.pinned_feedback == frozenset()
        live = dict(_pills(pane.signal_rows))
        assert all(live[k] for k in ("weighted", "weakest", "scaled"))
        assert "set by the protocol" not in " ".join(x.value for x in _texts(pane.signal_rows))
    finally:
        amp.stop()


def test_the_alert_is_offered_only_where_the_thresholds_were_chosen_for_it():
    """It fires when a target crosses the wrong way, so whether it is an event or a nuisance is decided
    by how often that happens — a property of the protocol's aims, not of the mechanism.

    CFC budgets 5-15% of the session for hearing something and derives every threshold from it. The
    amplitude screen aims an inhibit at 75% met, so it is over the line a quarter of the time and the
    alert would fire about as often as its interval gate allows, all session.
    """
    from nfb_console import amplitude_view, at_view, cfc_view, signal_pane

    for maker in (lambda: amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2),
                  lambda: at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)):
        session = maker()
        try:
            assert "alert" not in session.offers, session.protocol.code
            assert not session.feedback.alert.enabled, "and unoffered means off, not merely unlisted"
            pane = signal_pane.SignalPane(StubPage(), session)
            assert "alert" not in pane.offered
            assert "Alert" not in [x.value for x in _texts(pane.tabs.control)]
        finally:
            session.stop()

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert "alert" in session.offers and session.feedback.alert.enabled
        pane = signal_pane.SignalPane(StubPage(), session)
        assert "Alert" in [x.value for x in _texts(pane.tabs.control)]
    finally:
        session.stop()


def test_the_shadow_is_off_for_alpha_theta_but_not_forbidden():
    """Eyes closed is why it starts off — an output into a closed channel. It is not a reason nobody may
    turn it on: a dim room, or an operator watching the screen themselves. Off by default and unavailable
    are different claims and only the first is true here."""
    from nfb_console import at_view, signal_pane

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert not session.feedback.shadow.enabled
        assert "shadow" in session.offers and "shadow" not in session.pinned_feedback
        pane = signal_pane.SignalPane(StubPage(), session)
        assert dict(_pills(pane.shadow_rows))["shadow on"] is True, "it has to be pressable"
    finally:
        session.stop()


def test_an_arrangement_is_offered_only_where_it_means_something_and_shows_what_it_set():
    """Two faults in one control. The three arrangements were offered on the amplitude screen,
    where Quiet Reward is a setting that belongs to another protocol. And pressing one changed five fields spread across nested tabs while showing *nothing*: a
    control whose whole effect is off-screen, which is how it reads as a button that does not work.
    """
    from nfb_console import amplitude_view, cfc_view, signal_pane

    amp = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        assert amp.arrangements == (), "not this protocol's, so not on this screen"
        pane = signal_pane.SignalPane(StubPage(), amp)
        assert "ARRANGEMENT" not in " ".join(x.value for x in _texts(pane.control)), (
            "absent rather than empty — three pills that mean nothing here is the crossfade mistake")
        assert pane.model_rows.controls == []
    finally:
        amp.stop()

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        pane = signal_pane.SignalPane(StubPage(), session)
        assert session.feedback.arrangement == "quiet", "and this one is in force from the start"
        assert dict(_pills(pane.model_rows))["quiet"] is True, "so the pill says which you are in"

        before = " ".join(x.value for x in _texts(pane.model_rows))
        assert "sound on · inverted" in before and "stops when every target is met" in before
        pane._apply_model("proportional")
        after = " ".join(x.value for x in _texts(pane.model_rows))
        assert after != before, "pressing it has to change something you can see"
        assert "inverted" not in after and "keeps playing" in after
        assert session.feedback.arrangement == "proportional"
    finally:
        session.stop()


def test_the_crossfade_pane_shows_both_ends_and_not_a_file_browser():
    """Every branch of the music pane has to depend on the source it is drawn for. `crossfade` fell through
    to the file branch, so choosing it presented a path field and a browse button and never said it runs
    brown to pink — the default nobody could see. Both ends are now named by the state they stand for."""
    from nfb_console import at_view, signal_pane

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        pane = signal_pane.SignalPane(StubPage(), session)
        on_screen = " ".join(x.value for x in _texts(pane.music_rows))
        assert "brown" in on_screen and "pink" in on_screen, "the default is invisible again"
        assert "theta end" in on_screen and "alpha end" in on_screen
        assert "browse" not in on_screen, "the file branch is drawing for a crossfade"
        assert "hear it" in on_screen, "a noise colour's name does not say what it sounds like"
    finally:
        session.stop()


def test_every_noise_colour_can_be_heard_before_the_session_starts():
    """"1/f²" does not tell anybody how deep brown is, so the preview is the only way the control says what
    it does. Rendered to a file and handed to the system player rather than opened on the audio device: a
    live session may already hold that device, and a preview that silenced the music it previews would be
    worse than none."""
    import wave

    import numpy as np

    from nfb_console import music

    energy = {}
    for colour in music.NOISE_COLOURS:
        path = music.preview_path(colour)
        assert path is not None and path.exists(), colour
        with wave.open(str(path)) as fh:
            frames = fh.getnframes()
            assert fh.getframerate() == music.SAMPLE_RATE
            data = np.frombuffer(fh.readframes(frames), dtype=np.int16).astype(float)
        assert frames == pytest.approx(music.SAMPLE_RATE * music.PREVIEW_SECONDS, rel=0.01)
        assert abs(data).max() < 32767, "clipped noise is harsh in the way this music is chosen not to be"
        assert abs(data[:8]).max() < abs(data).max() * 0.2, "no click at the start"
        # the spectral tilt is the whole content of the control, so it is what the test checks
        spectrum = np.abs(np.fft.rfft(data * np.hanning(len(data))))
        freqs = np.fft.rfftfreq(len(data), 1 / music.SAMPLE_RATE)
        low = spectrum[(freqs > 100) & (freqs < 300)].mean()
        high = spectrum[(freqs > 3000) & (freqs < 6000)].mean()
        energy[colour] = low / high
    assert energy["brown"] > energy["pink"] > energy["white"], f"the tilts are wrong: {energy}"
    assert music.preview_path(colour) is music.preview_path(colour), "rendered once per process"
    assert music.preview_path("chartreuse") is None


def test_the_at_screen_can_audition_both_tilts_and_both_tones_from_the_feedback_tab():
    """Four sounds nobody has heard, in a protocol run with the eyes closed — so without auditions the
    first time either is heard is the one moment nothing can be adjusted.

    They live **with the mechanism that plays them**, not on Train. Train had all four plus the choice of
    which band the tone marks, which is sound configuration in the wrong place and redundant besides: the
    Music pane previews each tilt and the Tone pane auditions each reward at its own pitch.
    """
    from nfb_console import at_view, signal_pane

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        pane = signal_pane.SignalPane(StubPage(), session)
        music_text = " ".join(x.value for x in _texts(pane.music_rows))
        assert "theta end" in music_text and "alpha end" in music_text
        assert music_text.count("hear it") == 2, "both tilts, each with its own preview"

        tones = " ".join(x.value for x in _texts(pane.click_rows))
        # The pitch is a *settable* number now, so it is a field rather than a caption — derived by default
        # and pinned by typing, which is the same shape as a threshold.
        pitches = _numbers(pane.click_rows)
        assert "137" in pitches and "738" in pitches, f"both tones at their own pitches, got {pitches}"
        assert tones.count("▶") == 3, "each tone, plus the held mark, beside the number that sets it"
        assert "theta first" in tones, "and which band the tone marks, beside the tone"

        # Typing pins it; typing the derived value back leaves it derived, so the box is not a one-way door.
        pane._set_tone_hz(session.theta, 300)
        assert session.theta.tone_hz == 300.0
        assert "300" in _numbers(pane.click_rows)
        pane._set_tone_hz(session.theta, 137)
        assert session.theta.tone_hz is None, "back to derived"

        control = at_view.view(StubPage(), session, toolbar(), lambda: None)
        train = " ".join(x.value for x in _texts(control.controls[-1].controls[0]))
        assert "hear it" not in train, "no auditions on Train"
        assert "theta first" not in train, "and no sound configuration either"
        assert "toward theta" in train or "waiting for data" in train, "the music readout stays"
    finally:
        session.stop()


def test_the_crossfade_does_not_change_loudness_as_it_fades():
    """Music whose volume dipped mid-fade would read as a broken control rather than as a change of
    character. Both tilts filter the *same* white noise, so the fade interpolates one into the other
    instead of adding two independent sources in power. Measured over four seconds per position: the RMS
    holds within 5% across the sweep, under half a decibel."""
    import numpy as np

    from nfb_console.music import NoisePlayer

    levels = []
    for balance in (0.0, 0.25, 0.5, 0.75, 1.0):
        player = NoisePlayer()
        player.configure("pink", 100.0, second_colour="brown", balance=balance)
        rms = []
        for i in range(400):
            buf = player._block(512)
            if i > 20:                          # let the filter states settle
                rms.append(float(np.sqrt(np.mean(buf.astype(float) ** 2))))
        levels.append(float(np.mean(rms)))
    assert min(levels) > 0.9 * max(levels), f"the fade changes loudness: {levels}"
    assert max(abs(v) for v in buf.ravel()) < 1.0, "and it never reaches the clip point"


def test_the_at_ratio_is_averaged_over_the_window_it_names():
    """Read off each target's own trend rather than off the plot beside it, because the plot's point rate
    follows its *window* — 60 points of a ten-minute plot is two minutes, not the thirty seconds the label
    claims."""
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert session.ratio is None, "nothing until there is a window of it"
        need = int(at_view.RATIO_WINDOW / 0.5)
        for _ in range(need):
            session.theta.trend.append(4.0)
            session.alpha.trend.append(8.0)
        assert session.ratio == pytest.approx(0.5) and not session.crossed
        for _ in range(need):
            session.theta.trend.append(12.0)
            session.alpha.trend.append(8.0)
        assert session.ratio == pytest.approx(1.5) and session.crossed, "theta ahead is the crossover"
    finally:
        session.stop()


def test_at_thresholds_hold_still_so_the_crossover_can_show():
    """A threshold tracking a rolling window would follow theta up as it rose and pin the percentage where
    it started, erasing the one change the protocol exists to produce. Its own plan ships with
    autothresholding off (`autoset=0`) for the same end."""
    from nfb_console import at_view
    from nfb_console.targets import DEFAULT_MODES

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert all(t.mode == "fixed" for t in session.targets)
        assert DEFAULT_MODES[True] == "automatic", "so it had to be said rather than inherited"
        assert all(t.calibrating for t in session.targets), "calibration is still a phase"
    finally:
        session.stop()


def test_at_opens_bounded_because_nobody_is_watching_the_clock():
    """Open-ended is the app default, and eyes closed is the named exception — a limit nobody can see is
    not a limit, which is why the length and the audible cues live together."""
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert session.plan.bounded and session.length == 30 * 60
        assert session.plan.finish_cue
    finally:
        session.stop()

    given = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1, length_seconds=600)
    try:
        assert given.length == 600, "an explicit length is not overwritten"
    finally:
        given.stop()


def test_at_measures_the_summed_signal_when_several_channels_are_in_use():
    """One crossover to read, not one per site. Its own model list has a `Sum` variant of every one-channel
    feedback model — "sum of two channels of data input", `bandpass,in=0-1` into the same reward streams.

    Checked against the arithmetic rather than argued from it: two channels carrying the same tone read
    twice one channel, four read four times, and antiphase cancels. Cancelling is what proves the sum is
    *sample-aligned* — adding whatever block had most recently landed would leave a residue, which is why
    the blocks here stay the device's eight samples: more boundaries, more chances to misalign.
    """
    import numpy as np

    from nfb.stream import Block, ChannelSpec
    from nfb_console import at_view

    class Tones:
        # Fifteen seconds, not forty: the ratios below are identical to four decimal places at both, and
        # the estimator is settled long before. The forty was a guess with a factor of three in it.
        def __init__(self, phases, seconds=15.0, rate=250.0):
            self.phases, self.rate, self.seconds = phases, rate, seconds
            self.spec = {i: ChannelSpec(index=i, rate_hz=rate, unit="uV", scale=0.01)
                         for i in range(len(phases))}

        def channels(self):
            return self.spec

        def close(self):
            pass

        def blocks(self):
            n = int(self.seconds * self.rate)
            t = np.arange(n) / self.rate
            waves = [20 * np.sin(2 * np.pi * 10 * t + p) for p in self.phases]
            ints = [to_counts(x, 0.01) for x in waves]
            for i in range(0, n - 7, 8):
                for ch, x in enumerate(ints):
                    yield Block(ch, i, tuple(x[i:i + 8].tolist()), self.spec[ch])

    def alpha_of(phases):
        session = at_view.ATSession(BY_CODE["A-T"], source=Tones(phases), channels=len(phases))
        try:
            session.toggle_pause()
            # The fixture's blocks, straight into the loop body. Threading this and joining with a 25 s
            # timeout tested the scheduler's ability to finish in time, which is not the subject.
            for item in session.source.blocks():
                session.feed_block(item)
            session._stop = True
            return session.value_for(session.alpha), session.source_label
        finally:
            session.stop()

    one, one_label = alpha_of([0.0])
    two, two_label = alpha_of([0.0, 0.0])
    four, _ = alpha_of([0.0] * 4)
    cancelled, _ = alpha_of([0.0, np.pi])

    # 20 µV peak per channel in the fixture, read in the RMS convention the elliptic default reports.
    assert one > 10.0, "the fixture has to produce a real reading first"
    assert two == pytest.approx(2 * one, rel=0.02)
    assert four == pytest.approx(4 * one, rel=0.02)
    assert cancelled < 0.05 * one, "antiphase cancels, which is what says the sum is aligned"

    # and the screen has to say which electrodes those were: the toolbar's channel count says how many
    # are in use and cannot say that they are added together
    assert one_label == "ch0"
    assert two_label == "ch0+ch1 summed"


def test_the_at_screen_names_the_channels_it_is_summing():
    """A summed reading with no statement of its inputs is a number nobody can source — and a bad electrode
    inside a sum does not show up as one bad trace on the instrument being read."""
    from nfb_console import at_view

    page = StubPage()
    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=2)
    try:
        control = at_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        lines = [c.value for c in _texts(control)]
        assert "ch0+ch1 summed" in lines, "beside the bands it is measured on"
        assert any("ch0+ch1 summed ·" in x for x in lines), "and on the graph, with its window and scale"

        # every contributing channel has to be delivering, because all of them are inside the number
        assert all(t.pair == (0, 1) for t in session.targets)
        session.state = {0: True, 1: False}
        assert not session.target_is_live(session.theta)
        assert "ch1" in session.wiring_problem() and "sum" in session.wiring_problem()
    finally:
        session.stop()


def test_an_excluded_channel_stays_on_the_contact_check():
    """Narrowing the check drops the device's channel count, and the rows follow it — so an excluded channel
    would silently vanish, which reads as a channel that stopped existing rather than as one with no
    electrode. The rows a check started with stay for as long as it runs, and the empty ones say so."""
    from nfb import Device, Settings
    from nfb.device import Contact, Mode

    from nfb_console import device_panel

    # mid-check, after ch1-3 were dropped: the device is down to one channel but still names the three
    device = Device(Settings(channels=1))
    device.status.mode = Mode.TESTING
    device.contact.update({0: Contact(plus=5.3, minus=13.1)})
    device.excluded = [1, 2, 3]

    control = device_panel.view(StubPage(), device)
    rows = [r for r in _contact_rows(control) if r.visible]
    assert len(rows) == 4, f"a dropped channel must keep its row, got {len(rows)}"
    shown = " ".join(x.value for x in _texts(control))
    assert shown.count("no electrode") == 3, "every dropped channel is named"
    assert "5.3" in shown, "and the wired one still reads"
    assert "still listed" in shown


def _contact_rows(control):
    """The per-channel rows of the ELECTRODE CONTACT panel, found by the label they carry."""
    import flet as ft

    found = []

    def walk(node, depth=0):
        if node is None or depth > 14:
            return
        if isinstance(node, ft.Row):
            labels = [x.value for x in _texts(node)]
            if any(v in (f"ch{c}" for c in range(4)) for v in labels) and len(labels) <= 3:
                found.append(node)
        for attr in ("controls", "shapes"):
            for child in getattr(node, attr, None) or []:
                walk(child, depth + 1)
        walk(getattr(node, "content", None), depth + 1)

    walk(control)
    return found


def test_a_channel_with_no_signal_says_so_on_the_trace():
    """A flat grey line with a small red dot beside it is what a broken app looks like. The device already
    knows the input is open — it reports it in every frame — so there is no reason to leave it a guess.
    The words go over the trace, because that is the thing being explained."""
    from nfb_console import amplitude_view

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        page = StubPage()
        control = amplitude_view.view(page, session, toolbar(), lambda: None)
        session.state = {0: True, 1: False}
        session.redraw()
        shown = [x.value for x in _texts(control)]
        assert sum("no signal" in v for v in shown) == 1, "the open channel, and only it, is named"
        session.state = {0: True, 1: True}
        session.redraw()
        assert not any("no signal" in x.value for x in _texts(control)), "and it clears again"
    finally:
        session.stop()


def test_the_raw_trace_default_scale_suits_real_eeg():
    """Resting EEG measured on this device runs 12–20 µV peak-to-peak. On ±100 µV that is under 6 px of a
    92 px trace and reads as a flat line — the screen looked broken on a first session while the signal was
    fine. `fit` picks a preset rather than autoscaling, so the caption still names one fixed number."""
    from nfb_console import amplitude_view

    assert amplitude_view.GAINS[2] == 50.0, "the default is the third pill"
    assert 10.0 in amplitude_view.GAINS, "a quiet channel is the case that needs the smallest scale"
    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        assert session.gain_uv == 50.0
        page = StubPage()
        control = amplitude_view.view(page, session, toolbar(), lambda: None)
        assert any(x.value == "fit" for x in _texts(control)), "no way to choose a scale from the signal"
    finally:
        session.stop()


def test_nudging_a_band_edge_slides_the_whole_band():
    """The arrows move the band and keep its width; typing an edge still sets that edge alone.

    Nudging a band is hunting for a frequency — "often within 1-2 Hz of the frequency in the spectral
    display showing the highest amplitude" — which is a move, not a resize. Edge-at-a-time meant two presses
    in opposite boxes per step, and a width that drifted if you lost count.
    """
    from nfb_console import amplitude_view

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        tgt = session.targets[0]
        low, high = tgt.band.low, tgt.band.high
        tgt.shift_band(0.1)
        assert (tgt.band.low, tgt.band.high) == pytest.approx((low + 0.1, high + 0.1))
        assert tgt.band.high - tgt.band.low == pytest.approx(high - low), "width is preserved"

        tgt.shift_band(-0.1)
        assert (tgt.band.low, tgt.band.high) == pytest.approx((low, high))

        # clamped at the bottom rather than raising out of a click handler, width still preserved
        tgt.shift_band(-1000.0)
        assert tgt.band.low == pytest.approx(0.1)
        assert tgt.band.high - tgt.band.low == pytest.approx(high - low)

        # typing is unchanged: it sets one edge
        session.retune(tgt, low=4.0)
        assert tgt.band.low == 4.0 and tgt.band.high == pytest.approx(0.1 + (high - low))
    finally:
        session.stop()


def test_the_band_arrows_are_wired_to_the_slide():
    """A regression fence on the wiring rather than the arithmetic: `num_box` takes `on_step` so the arrows
    and the typed value can do different things, and a screen that forgot to pass it would silently go back
    to moving one edge."""
    import inspect

    from nfb_console import amplitude_view, at_view, widgets

    assert "on_step" in inspect.signature(widgets.num_box).parameters
    for module in (amplitude_view, at_view):
        src = inspect.getsource(module)
        assert src.count("on_step=") == 2, f"{module.__name__} must wire both edges"
        assert "def slide(" in src


def test_checking_contact_mid_session_comes_back_to_the_same_session():
    """Pause, look at the electrodes, come back and carry on. The feed has to stop — impedance testing and
    measuring are exclusive device modes and a paused session still holds the device measuring — but the
    *session* must survive: the elapsed clock, every target's trend and calibration, the gate's running level
    and the on-target history are the session, and a fresh one with the same settings is a different session
    that merely looks alike."""
    from nfb_console import amplitude_view

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        session.paused = False
        session.gain_uv = 10.0
        tgt = session.targets[0]
        session.retune(tgt, low=9.5, high=11.5)
        tgt.threshold = 7.25
        tgt.mode = "fixed"
        for v in (1.0, 2.0, 3.0):
            tgt.trend.append(v)
        trend_before = list(tgt.trend)

        session.suspend()
        assert session.paused, "the clock freezes: looking at electrodes is not time on task"
        assert not session._stop, "suspend is not stop"
        assert session.suspended

        # what `resume` re-applies: nothing. The same object carries on.
        assert session.gain_uv == 10.0
        assert (tgt.band.low, tgt.band.high) == (9.5, 11.5)
        assert tgt.threshold == 7.25 and tgt.mode == "fixed"
        assert list(tgt.trend) == trend_before
        assert session.targets[0] is tgt, "the same target object, not a copy of its settings"
    finally:
        session.stop()


def test_coming_back_from_a_contact_check_puts_the_device_back_in_measuring_mode():
    """**Leaving the device screen does not stop a running contact check** — only pressing the button again
    does — so coming back from an impedance measurement found the device in TESTING, and `rearm` acted only
    on IDLE. The mode never changed, the surviving subscription was handed *impedance* frames, and the screen
    came back with a running clock, flat traces and not one target ever fed.

    The identical guard is in `TrainingSession.__init__`, for the identical reason. This is the route
    somebody actually takes: pause, check contact, come back.

    Stopping is also what hands the channel count back, since `narrow_to_wired` borrows it for the check and
    `Device.stop` restores it.
    """
    from nfb.device import Mode
    from nfb_console import amplitude_view

    class ModeStub:
        def __init__(self, mode):
            self.status = type("S", (), {"mode": mode, "connected": True})()
            self.calls = []

        def stop(self):
            self.calls.append("stop")
            self.status.mode = Mode.IDLE

        def start_measurement(self):
            self.calls.append("start")
            self.status.mode = Mode.MEASURING

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        session.device = ModeStub(Mode.TESTING)
        session.rearm()
        assert session.device.calls == ["stop", "start"], "a contact check has to end before measuring can"
        assert session.device.status.mode is Mode.MEASURING

        session.device = ModeStub(Mode.IDLE)         # the ordinary way back, unchanged
        session.rearm()
        assert session.device.calls == ["start"]

        session.device = ModeStub(Mode.MEASURING)    # already measuring: nothing to do, and no restart
        session.rearm()
        assert session.device.calls == []
    finally:
        session.device = None
        session.stop()


def test_stopping_is_still_destructive():
    """`suspend` keeps a session alive; the stop button must not. Two paths, and conflating them would mean
    the stop button handed you the session you just ended."""
    from nfb_console import amplitude_view

    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    session.stop()
    assert session._stop
    assert not session.suspended, "a stopped session must not look re-armable"


def test_the_documented_command_line_parses():
    """`make live` passed `--mains` and the parser did not accept it, so the command the Makefile
    documents simply failed. Every flag the Makefile passes is checked here, because a broken
    entry point is invisible to a test suite that constructs sessions directly."""
    import re

    from nfb_console.app import build_parser

    flags = set(re.findall(r"--[a-z-]+", open("Makefile").read()))
    parser = build_parser()
    known = {s for action in parser._actions for s in action.option_strings}
    # only the console's own flags: `nfb record` has its own parser
    console_line = [ln for ln in open("Makefile").read().splitlines() if "nfb-console" in ln]
    used = {f for ln in console_line for f in re.findall(r"--[a-z-]+", ln)}
    assert used, "no nfb-console invocation found in the Makefile"
    assert used <= known, f"the Makefile passes flags the console rejects: {sorted(used - known)}"
    assert flags, "sanity: the Makefile does pass flags"

    args = parser.parse_args(["--live", "--hz", "1000", "--reference", "linked", "--mains", "50"])
    assert args.mains == 50 and args.hz == 1000.0


def test_the_raw_trace_is_a_waveform_not_a_filled_block():
    """A filled min/max envelope inks every column's whole excursion, so real EEG drew as one solid mass —
    the information is present and the waveform is not, which is the only thing a raw trace is read for.

    Stroked as a zigzag instead: one polyline that walks down one column and up the next, so it still visits
    both extremes of every column (no spike is decimated away) while the ink follows the signal rather than
    the area under it. Same point count, one shape.
    """
    import flet as ft
    import numpy as np

    from nfb_console.plots import envelope_shapes

    rng = np.random.default_rng(0)
    lo = -20 - rng.random(64) * 5
    hi = 20 + rng.random(64) * 5
    shapes = envelope_shapes(lo, hi, 320.0, 92.0, 100.0, "#000000")
    assert len(shapes) == 1
    paint = shapes[0].paint
    assert paint.style == ft.PaintingStyle.STROKE, "a filled envelope is the fuzzy-worm bug"

    pts = shapes[0].points
    assert len(pts) == 2 * len(lo), "both extremes of every column are still drawn"
    # consecutive points share an x, and the walk alternates direction so it is one continuous stroke
    assert pts[0].x == pts[1].x and pts[2].x == pts[3].x
    assert (pts[0].y < pts[1].y) != (pts[2].y < pts[3].y), "the zigzag does not retrace itself"

    # a quiet channel still draws something rather than vanishing sub-pixel
    flat = np.zeros(64)
    thin = envelope_shapes(flat, flat, 320.0, 92.0, 100.0, "#000000")[0].points
    assert abs(thin[1].y - thin[0].y) >= 1.0


def test_the_spectrum_scale_does_not_strobe():
    """`nice_ceiling(peak)` recomputed every frame is unusable live: a peak wobbling either side of 5 µV
    flips the axis between 5 and 10 several times a second, the bars halve and double, and the picture
    strobes. Right numbers, unreadable display.

    Rises at once, because a clipped bar is information lost. Falls only after the peak has stayed
    *comfortably* below for a few seconds — hysteresis and a delay, since either alone still flickers.
    """
    from nfb_console.plots import StickyCeiling, nice_ceiling

    # the pathological case: a peak sitting right on a step boundary
    naive = {nice_ceiling(v) for v in (4.9, 5.1, 4.8, 5.2)}
    assert len(naive) > 1, "the premise: the naive ceiling moves here"

    sticky = StickyCeiling(patience=4.0)
    now = 0.0
    for i in range(100):                      # let it settle out of its initial guess
        now += 0.1
        sticky.update(5.0 + (0.3 if i % 2 else -0.3), now)
    seen = set()
    for i in range(400):
        now += 0.1
        seen.add(sticky.update(5.0 + (0.3 if i % 2 else -0.3), now))
    assert len(seen) == 1, f"it strobed anyway: {sorted(seen)}"
    assert seen.pop() >= 5.3, "and whatever it settled on must still hold the peak"

    # rises immediately, because clipping loses the peak you are looking for
    assert sticky.update(40.0, now) >= 40.0
    risen = sticky.value

    # and does not fall until the peak has been below for the full patience
    now += 1.0
    assert sticky.update(2.0, now) == risen, "falling instantly is the other half of the flicker"
    for _ in range(60):
        now += 0.1
        sticky.update(2.0, now)
    assert sticky.value < risen, "it has to come back down eventually"
    assert sticky.value >= 2.0, "but never below the peak it is showing"

    # a peak that keeps touching the ceiling holds it there indefinitely
    high = StickyCeiling(patience=1.0)
    high.update(9.0, 0.0)
    held = high.value
    for i in range(50):
        assert high.update(9.0, 1.0 + i) == held, "a peak at the top must not trip a step down"


def test_every_screen_with_a_band_on_it_has_a_spectrum():
    """A-T and CFC had none, and both are read against the spectrum.

    The reward frequency in CFC is *found* there — "within 1-2 Hz of the frequency in the spectral display
    showing the highest amplitude in a particular band" — and A-T's crossover is a comparison of two band
    amplitudes, where the shape underneath says whether an alpha peak or broadband drift is moving.

    The sum comes first on both, because that is what their bands are measured on; the channels follow,
    because a flat electrode is invisible inside a sum, and CFC says so outright — "you will not be able
    to tell from the spectral display whether the recurrent beta spindling is coming from channel A or
    channel B".
    """
    from nfb_console import amplitude_view, at_view, cfc_view

    for mod, code, kwargs in ((amplitude_view, "AMP", {}),
                              (at_view, "A-T", {}),
                              (cfc_view, "CFC", {})):
        session = mod.session_for(BY_CODE[code], source=replay(), channels=2, **kwargs)
        try:
            page = StubPage()
            control = mod.view(page, session, toolbar(), lambda: None)
            labels = [x.value for x in _texts(control)]
            assert "Spectrum" in labels, code
            sources = [s.label for s in session.spectrum_sources()]
            assert sources, code
            # Every multi-channel screen offers the sum; *where* in the row is the claim each screen makes,
            # declared once as `sum_spectrum_leads` rather than five hand-built lists in three orders.
            assert "sum" in sources, f"{code}: no spectrum of the summed signal"
            if session.sum_spectrum_leads:
                assert sources[0] == "sum", f"{code}: the sum is what the bands are measured on"
            else:
                assert sources[-1] == "sum", f"{code}: a target sits on a channel, so the channels lead"
            assert [s for s in sources if s.startswith("ch")], code
            pump(session)
            session.redraw()
            assert session.error is None, (code, session.error)
        finally:
            session.stop()


def test_both_mixing_rules_are_rows_and_neither_is_a_footnote():
    """Two strategies run at once and the panel showed one of them as a control.

    `level from` was a row of pills and the condition's rule was a remark under a readout, so a locked
    `scaled` read as *the* mixing strategy — while the rule that actually governs every discrete output sat
    in small text. It is now its own numbered row with its own pills, locked everywhere because it is
    derived rather than chosen: `either` makes the rewards alternatives, and an AND over alternatives cannot
    be satisfied. Two pills rather than a sentence, so the alternative and the dependency are both visible.
    """
    from nfb_console import cfc_view, signal_pane

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        pane = signal_pane.SignalPane(StubPage(), session)
        rows = " ".join(x.value for x in _texts(pane.signal_rows))
        # **The numbers are positions in this screen's chain, not fixed labels.** CFC's bands are measured
        # by a lock-in with its own post-filter, so `envelope_seconds` reaches nothing here and the
        # band-averaging step is absent — see `TrainingSession.tunes_envelope`. What must hold is that the
        # numbering stays contiguous from 1, because a panel starting at "2 ·" reads as one missing a row.
        assert "1 · scale from" in rows, "the chain starts at 1 whichever steps this screen has"
        assert "2 · level from" in rows and "3 · condition" in rows
        assert "4 · sensitivity" in rows, "the one gain, on the finished level, after both rules"
        assert "5 · output smoothing" in rows, "and the smoothing is last, on the shaped level"
        assert "band averaging" not in rows, "no estimator here takes it; offering it crashed on ILF"

        # Stated rather than offered: it has one form per protocol, because it is derived from the rule
        # above and `either` is only on offer where the rewards really are alternatives. Two inert pills
        # was the first attempt and read as a broken control beside a live one.
        assert session.feedback.condition_rule == "all"
        assert "every target" in rows and "one reward" not in rows
        assert "one reward" not in dict(_pills(pane.signal_rows))
        assert "follows the rule above" in rows, "and it says what decides it"
        assert _chosen(pane.signal_rows) >= {"scaled"}

        # it tracks the rule rather than being a second setting that could disagree with it
        session.feedback.combine = "either"
        assert session.feedback.condition_rule == "either"
        pane.rebuild()
        rows = " ".join(x.value for x in _texts(pane.signal_rows))
        assert "one reward" in rows and "every target" not in rows
    finally:
        session.stop()


def test_picking_a_channel_spectrum_moves_the_focus_and_a_sum_does_not():
    """On the amplitude screen the focus also decides which channel a new target lands on, so selecting a
    channel's spectrum has to move it — and a sum is not a channel, so it must not."""
    from nfb_console import cfc_view, spectrum_pane

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        pane = spectrum_pane.SpectrumPane(StubPage(), session, lambda reserve=0: 600.0)
        assert pane.key == "sum", "the default is the thing being trained"
        before = session.focus
        pane._pick("ch1")
        assert session.focus == 1
        pane._pick("sum")
        assert session.focus == 1, "a sum leaves it where it was rather than resetting it"
        assert before in (0, 1)
    finally:
        session.stop()


def test_the_spectrum_can_be_pinned_to_a_fixed_scale():
    """Auto is right before you know the range; a fixed axis is the only way to compare one minute with the
    next, or this session with yesterday's."""
    from nfb_console import amplitude_view, spectrum_pane

    assert 0.0 in spectrum_pane.SCALES, "auto is one of the choices"
    session = amplitude_view.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        assert session.spectrum_scale == 0.0, "auto by default"
        page = StubPage()
        control = amplitude_view.view(page, session, toolbar(), lambda: None)
        shown = " ".join(x.value for x in _texts(control))
        assert "auto" in shown
        session.spectrum_scale = 10.0
        pump(session)
        show_all_panes(control)
        session.redraw()
        caption = [x.value for x in _texts(control) if "Hz bins" in (x.value or "")]
        assert caption and "full scale" in caption[0], "a pinned axis says so rather than 'auto scale'"
    finally:
        session.stop()


# --- ILF Synchrony --------------------------------------------------------------------------

def test_the_ilf_screen_draws_before_any_data_arrives():
    from nfb_console import ilf_view

    page = StubPage()
    session = ilf_view.ILFSession(BY_CODE["ILF"], source=replay(), channels=2)
    try:
        control = ilf_view.view(page, session, toolbar(), lambda: None)
        assert control is not None
        session.redraw()                    # no blocks fed at all
        assert session.error is None
    finally:
        session.stop()


def test_the_ilf_screen_draws_with_real_data_through_it():
    """Four instruments on one screen whose windows run from 300 ms to four minutes, so every caption has a
    different unit in it — which is exactly the kind of thing that throws on a live frame and passes every
    object-inspecting test."""
    from nfb_console import ilf_view

    page = StubPage()
    session = ilf_view.ILFSession(BY_CODE["ILF"], source=replay(), channels=2)
    try:
        control = ilf_view.view(page, session, toolbar(), lambda: None)
        session.toggle_pause()
        pump(session)
        for _ in range(3):
            session.redraw()
        assert session.error is None, session.error
        captions = " ".join(x.value for x in _texts(control))
        assert "cycles" in captions, "the slow signal and the Lissajous state their window in cycles"
        assert "µV²" in captions, "and the reward axis says it is a product"
        assert "across" in captions and "full scale" in captions
    finally:
        session.stop()


def test_the_ilf_screen_says_which_band_the_reward_is_measuring():
    """And crucially says when the filter has become a bare lowpass, because that is the difference between
    training a narrow band at 0.05 Hz and training the slow waveform — the same control, two instruments."""
    from nfb_console import ilf_view

    page = StubPage()
    session = ilf_view.ILFSession(BY_CODE["ILF"], source=replay(), channels=2)
    try:
        control = ilf_view.view(page, session, toolbar(), lambda: None)
        shown = " ".join(x.value for x in _texts(control))
        assert "8.5–11.5 Hz" in shown, "their 3 Hz bandwidth at their default centre"
        # The default slow band has a lower edge — their `direct` path assumes a hardware edge this amplifier
        # cannot supply — and it sits at `centre/4` rather than the ILF element's `centre/2`, which is a group
        # delay decision and ours. See `nfb.dsp.product.SLOW_BANDS`.
        session.set_frequency(0.05)
        session.redraw()
        shown = " ".join(x.value for x in _texts(control))
        assert "0.0125–0.1 Hz" in shown, "centre/4 to twice the centre"

        # And their path is still there, saying exactly what it always said.
        session.set_frequency(0.05, slow_band="0–2×")
        session.redraw()
        shown = " ".join(x.value for x in _texts(control))
        assert "no high-pass" in shown and "below 0.1 Hz" in shown, \
            "the lowpass corner is twice the centre, not centre + bandwidth/2"
    finally:
        session.stop()


def test_the_ilf_screen_presents_one_instrument_with_three_places_to_point_it():
    """Their `Synchrony Module` appears once and takes one `RewardFrequency`, so the three named frequencies
    are presets on one card rather than three cards. An earlier version built three concurrent rewards."""
    from nfb_console import ilf_view

    page = StubPage()
    session = ilf_view.ILFSession(BY_CODE["ILF"], source=replay(), channels=2)
    try:
        control = ilf_view.view(page, session, toolbar(), lambda: None)
        session.redraw()            # the axis units are written on the first frame
        shown = " ".join(x.value for x in _texts(control))
        for name, _hz in ilf_view.PRESETS:
            assert name in shown
        assert shown.count("threshold") >= 1
        assert "µV²" in shown, "and it says the reward is a product"
    finally:
        session.stop()


def test_the_feedback_tab_offers_no_file_browser_for_the_shepard_tone():
    """It fell through to the file branch and offered to play an mp3 as a Shepard tone. What makes it a
    Shepard tone is stated — the wheel — and how far it travels is set, because that is this mechanism's
    output range and every other continuous mechanism has one."""
    from nfb_console import ilf_view, signal_pane

    page = StubPage()
    session = ilf_view.ILFSession(BY_CODE["ILF"], source=replay(), channels=2)
    try:
        pane = signal_pane.SignalPane(page, session)
        shown = " ".join(x.value for x in _texts(pane.control))
        # the music is off on this screen, so switch to it to render its row
        session.feedback.music.enabled = True
        session.feedback.music.source = "shepard"
        pane = signal_pane.SignalPane(page, session)
        shown = " ".join(x.value for x in _texts(pane.control))
        assert "partials an octave apart" in shown
        assert "how far it climbs" in shown and "from the bottom of the signal to the top" in shown
        assert "file or folder" not in shown
    finally:
        session.stop()


def test_a_channel_the_device_calls_dead_while_delivering_signal_stays_live():
    """**The device asserts no-signal on channels that are working, and one frame was fatal.**

    `Block.valid` describes a block; `state` describes a channel. The feed assigned one to the other, so a
    single frame with the bit set flipped the channel to dead. That happens on real hardware — the measurement
    in `Block.state`'s docstring is 19960 samples of EEG at state 0 from a good electrode at C4 **and one
    frame at state 2** — and on an infra-low channel the condition stays asserted far longer than a flicker.

    Reported as: raw trace fine on both channels, contact fine on both, no feedback at all. Every target on
    the infra-low screen is measured on the *pair*, so `target_is_live` needs both channels and one condemned
    channel took the reward down; the trace kept drawing because `pass_through` shows rejected blocks on
    purpose.

    A 30 s grace period fixed the flicker and left the sustained case, which is the one reported next: good
    impedance, the 0.5 Hz corner, a signal that looks fine, and the channel dead anyway. The timer only ever
    ran down — nothing could acquit a channel, so once the amplifier asserted its bit for half a minute the
    verdict stood until the amplifier withdrew it, and it is under no obligation to. Its own wording is
    "signal far below range **for a long time**", and 20 µV of EEG sits at 2.5% of an 800 µV range, so a
    conservative detector latching on a good quiet channel is not a fault in the device. Treating it as
    unappealable was ours.

    So the samples decide, in both directions, on every block.
    """
    from nfb.stream import NO_SIGNAL, OK, Block
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        spec = session.specs[0]
        moving = tuple(int(20 * (-1) ** i / spec.scale) for i in range(8))       # ±20 µV, plainly a signal
        railed = tuple(int(800 / spec.scale) for _ in range(8))                  # an open input
        good = Block(0, 0, moving, spec, state=OK)
        bad = Block(0, 8, moving, spec, state=NO_SIGNAL)

        session._judge_channel(good)
        assert session.state[0] is True
        for _ in range(2000):
            session._judge_channel(bad)
        assert session.state[0] is True, \
            "no amount of complaining condemns a channel whose samples are a signal"

        # It is not a licence either: samples that corroborate the verdict condemn it on the first block.
        session._judge_channel(Block(0, 16, railed, spec, state=NO_SIGNAL))
        assert session.state[0] is False
        # And recovery is the next block, not the next time the amplifier changes its mind. This is the
        # half that was missing: `_looks_open` was consulted only where it could convict.
        session._judge_channel(bad)
        assert session.state[0] is True, "a live channel recovers from its own samples"
    finally:
        session.stop()


def test_a_verdict_the_samples_corroborate_is_taken_at_once():
    """Which is what lets the grace period be long: a real disconnection is caught on the first block.

    An open input rails to exactly full scale and a shorted pair reads dead flat, and both are visible in the
    block. Waiting 30 s on either would be the old failure — a dead screen with nothing said — wearing a
    different hat.
    """
    from nfb.stream import NO_SIGNAL, OK, Block
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        spec = session.specs[0]
        moving = tuple(int(20 * (-1) ** i / spec.scale) for i in range(8))
        session._judge_channel(Block(0, 0, moving, spec, state=OK))
        assert session.state[0] is True

        railed = (32767,) * 8                                   # an unconnected input, at the rail
        session._judge_channel(Block(0, 8, railed, spec, state=NO_SIGNAL))
        assert session.state[0] is False, "railed samples agree with the device; no reason to wait"

        session._judge_channel(Block(0, 16, moving, spec, state=OK))
        assert session.state[0] is True
        flat = (500,) * 8                                       # two inputs shorted together
        session._judge_channel(Block(0, 24, flat, spec, state=NO_SIGNAL))
        assert session.state[0] is False, "a dead flat line agrees too"
    finally:
        session.stop()


def test_the_operator_can_overrule_the_device_about_a_channel():
    """**The last word belongs to the person looking at the waveform**, from the trace where they can see it.

    The amplifier's below-range detector fires on good EEG and no amount of arithmetic here can adjudicate
    that better than an operator with the trace in front of them. `trust_channel` is that override, and it has
    to survive the next arriving block — which is why the device's own reading is kept apart from `state`.
    """
    from nfb.stream import NO_SIGNAL, OK, Block
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        spec = session.specs[0]
        railed = (32767,) * 8
        session._judge_channel(Block(0, 0, railed, spec, state=NO_SIGNAL))
        assert session.state[0] is False

        session.trust_channel(0, True)
        assert session.state[0] is True, "the operator says it is fine"
        for _ in range(20):
            session._judge_channel(Block(0, 8, railed, spec, state=NO_SIGNAL))
        assert session.state[0] is True, "and arriving blocks must not quietly undo that"

        # It goes both ways: a channel the device likes can still be dismissed.
        session.trust_channel(0, False)
        session._judge_channel(Block(0, 16, railed, spec, state=OK))
        assert session.state[0] is False

        session.trust_channel(0, None)                  # handed back to the device
        session._judge_channel(Block(0, 24, railed, spec, state=OK))
        assert session.state[0] is True
    finally:
        session.stop()


def test_a_forced_on_channel_is_measured_whatever_the_device_says():
    """**The override has to reach the measurement, not just the indicators.**

    The feed loop's gate-or-bypass decision read the block's own `valid` flag, which the override never
    touched — so a channel the device flagged "no signal" continuously (which it does on good infra-low
    EEG at 2.5% of its narrowest range) could be forced on, go green, count for every target, and the
    metrics were still fed `pass_through`'s stand-in: a constant at the last accepted sample, zero on a
    channel never once accepted. Reported as a slow trace pinned flat beside a healthy raw trace and a
    caption reading "ch1 forced on". The decision now follows the resolved state, where the operator wins.
    """
    from nfb.stream import NO_SIGNAL, Block, ChannelSpec
    from nfb_console import amplitude_view as av

    spec = ChannelSpec(index=0, rate_hz=250.0, unit="µV", scale=0.05)
    moving = tuple(int(20 * (-1) ** i / spec.scale) for i in range(8))     # ±20 µV, plainly a signal

    # Dead flat under the no-signal verdict: the one case the samples cannot rescue, because two shorted
    # inputs look exactly like this and so does an amplifier saying nothing. `_judge_channel` condemns it,
    # correctly, and the override is what is left when the operator can see it is wrong.
    flat = (int(20 / spec.scale),) * 8

    class FlaggedSource:
        def __init__(self, samples):
            self.samples = samples

        def channels(self):
            return {0: spec}

        def blocks(self):
            for n in range(50):
                yield Block(0, n * 8, self.samples, spec, state=NO_SIGNAL)

    def fed_samples(samples, force: bool) -> list[tuple[int, ...]]:
        session = av.AmplitudeSession(BY_CODE["AMP"], source=FlaggedSource(samples), channels=1)
        try:
            if force:
                session.trust_channel(0, True)
            fed: list[tuple[int, ...]] = []
            original = session.consume
            session.consume = lambda block: (fed.append(block.samples), original(block))[1]
            session.run()                       # finite source, runs to exhaustion synchronously
            return fed
        finally:
            session.stop()

    forced = fed_samples(flat, force=True)
    assert forced and all(s == flat for s in forced), \
        "forced on means the metrics see the channel's own samples, not the stand-in"

    # Left alone it is the stand-in that arrives — held at zero, since nothing was ever accepted — which is
    # the flat slow trace the override exists to cure.
    left_alone = fed_samples(flat, force=False)
    assert left_alone and all(s == (0,) * 8 for s in left_alone)

    # And the case that needed no override in the first place: samples that are plainly a signal are
    # believed whatever the amplifier says about them, so this reaches the metrics untouched.
    assert all(s == moving for s in fed_samples(moving, force=False))


def test_the_cascade_is_sized_to_its_box_and_not_to_its_pixels():
    """**A Stack lays an unpositioned child out at its intrinsic size**, and `expand` is a Row/Column idea.

    The cascade's PNG is one pixel per frequency bin by one per time row — about 44 by 72 — so with only a
    height set and `expand=True` inside a Stack, the panel drew it at roughly the shape of the array: about
    half the width it had been given and twice the height. Reported as "rotated 90 degrees", which is what a
    picture that should be wide and is instead tall looks like.

    Both halves are pinned: the image carries an explicit width matching the axis below it, and the Stack's
    children are anchored on all four edges.
    """
    import flet as ft

    from nfb_console import cfc_view

    page = StubPage()
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        pump(session)
        session.redraw()

        def stacks(x, d=0):
            out = []
            if x is None or d > 25:
                return out
            if isinstance(x, ft.Stack):
                out.append(x)
            for c in (getattr(x, "controls", None) or []):
                out += stacks(c, d + 1)
            if getattr(x, "content", None) is not None:
                out += stacks(x.content, d + 1)
            return out

        images = [c for st in stacks(control) for c in st.controls
                  if isinstance(getattr(c, "content", None), ft.Image)]
        assert images, "the cascade lives in a Stack"
        holder = images[0]
        assert (holder.left, holder.top, holder.right, holder.bottom) == (0, 0, 0, 0), \
            "anchored on all four edges, or the Stack sizes it to the PNG"

        image = holder.content
        assert image.width and image.width > 200, "an explicit width, not the array's bin count"
        assert image.height and image.height > 0
        assert image.fit == ft.BoxFit.FILL, "stretched to the box rather than letterboxed into it"
        # Wide, not tall: frequency runs across and time runs down.
        assert image.width > image.height
    finally:
        session.stop()


def test_the_offset_that_rails_a_channel_is_visible_somewhere():
    """**A dead channel with good contact had nowhere to look, and that was partly this repo's doing.**

    At the device's 0.01 Hz corner an electrode pair's standing potential passes straight through — measured
    305 and 311 µV on a real capture, one channel reaching 652 µV against an 800 µV rail. Rail, and the
    device reports no signal. Contact impedance neither predicts nor explains it: the same capture read 14
    and 26 kOhm, which is fine. So the operator checks contact, finds nothing, and is stuck.

    And the raw trace cannot show it, because it is high-passed at `DISPLAY_FLOOR_HZ` to stay readable at
    that corner — a fix that removed the offset from the only place it was visible. Hence a number.
    """
    from nfb.stream import OK, Block
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        spec = session.specs[0]
        assert session.offset_uv(0) is None and session.headroom_uv(0) is None, "nothing until data"

        # A channel sitting on a -600 µV standing offset, well inside range but close to the 800 µV rail.
        counts = tuple(int((-600.0 + 5 * (-1) ** i) / spec.scale) for i in range(8))
        for i in range(400):
            session._watch_offset(Block(0, i * 8, counts, spec, state=OK))
        assert session.offset_uv(0) == pytest.approx(-600.0, abs=5.0)
        assert session.headroom_uv(0) == pytest.approx(800.0 - 605.0, abs=5.0)

        # Headroom is the worst of the session, not the latest: a channel that grazed the rail once will do
        # it again, and the operator needs to know that after it has recovered.
        calm = tuple(int(10.0 / spec.scale) for _ in range(8))
        # Long enough to settle: 8 samples at 250 Hz is 32 ms a block, against `OFFSET_TAU_SECONDS` of 5.
        for i in range(1200):
            session._watch_offset(Block(0, 9999 + i * 8, calm, spec, state=OK))
        assert session.headroom_uv(0) < 200.0, "the close approach is remembered"
        assert session.offset_uv(0) == pytest.approx(10.0, abs=5.0), "the offset itself follows"
    finally:
        session.stop()


def test_continuous_outputs_apply_on_their_own_thread_not_the_feed():
    """The output calls are the slowest things in the process — a CoreAudio write at 1.1 ms p99, a
    window-server display lookup at 13 ms worst, a subprocess spawn for the music — and they ran inline
    on the thread that decodes and filters against a 32 ms budget. `run` now starts an output pump;
    `_drive_outputs` snapshots and wakes it, so the feed thread never blocks on an output device."""
    import threading as th

    from nfb import ReplaySource
    from nfb_console import amplitude_view as av

    from conftest import BENCH_OK

    session = av.AmplitudeSession(BY_CODE["AMP"],
                                  source=ReplaySource(str(BENCH_OK), realtime=False, loop=True),
                                  channels=2)
    calls: list[str] = []

    def out():
        return type("Out", (), {
            "apply": lambda self, *a, **kw: calls.append(th.current_thread().name),
            "release": lambda self: None,
            "feed": lambda self, *a, **kw: None,
        })()

    session.display = out()
    session.output_volume = out()
    session.music = out()
    session.toggle_pause()                  # a session opens paused
    feeder = th.Thread(target=session.run, name="feed", daemon=True)
    feeder.start()
    waitfor(lambda: len(calls) >= 6, what="six output applies")
    session.stop()
    feeder.join(timeout=2.0)
    assert calls, "the outputs were never driven"
    # **Including the last one.** `stop()` ends the pump's loop while the feed thread is still inside
    # `source.blocks()`, so the fallback used to fire on the way out and the final apply of every session
    # landed on the feed thread. This assertion failed about one run in six, which is what an intermittent
    # assertion usually is: a real race, seen from the outside.
    assert set(calls) == {"outputs"}, f"output calls ran on {sorted(set(calls))}"


def test_pausing_hands_the_outputs_back_even_when_a_block_was_in_flight():
    """Reported live: the music was set to a tone, the session was paused, and the tone went on playing.

    The pause asks for a release by publishing no snapshot, but the feed thread can already be inside
    `_drive_outputs` when that lands, and its snapshot overwrites the request. The pump then saw a
    snapshot, saw `paused`, and skipped — applying nothing and releasing nothing, so whatever was
    sounding kept sounding until the session resumed or stopped. A paused session feeds nothing back, so
    a snapshot that arrives while paused is a release like any other."""
    from nfb_console import amplitude_view as av

    log: list[str] = []
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    session.music = type("M", (), {
        "apply": lambda self, *a, **kw: log.append("apply"),
        "release": lambda self: log.append("release"),
        "feed": lambda self, *a, **kw: None,
    })()
    try:
        pump(session)
        session._stop = False
        session._output_thread = None
        session.paused = False
        session.toggle_pause()
        assert session.paused and log and log[-1] == "release", "the pause itself hands them back"

        log.clear()
        session._drive_outputs()
        assert log == [], "a block still in flight must not re-take the outputs"

        log.clear()
        session._apply_snapshot((0.8, True, 0.0, None))
        assert log == ["release"], "and a snapshot that lands anyway releases rather than being skipped"

        session.paused = False
        log.clear()
        session._apply_snapshot((0.8, True, 0.0, None))
        assert log == ["apply"], "a running session still applies"
    finally:
        session.stop()


def test_the_heart_and_headband_screens_hide_the_amplifier_channel_count():
    """**Building the screen is what has to hide it**, not a rule somewhere about which protocols use the
    amplifier. On the HRV screen there is one instrument and it is not the amplifier, so a "1 ch ▾" menu
    beside it offers to reconfigure a device the screen never touches — and picking a number from it would
    rewrite the amplifier's config and restart a session that is reading a heart."""
    from nfb_console import heg_view, hrv_view
    from nfb_console.inputs import HeartChoice, MendiChoice
    from nfb_console.protocols import BY_CODE

    for screen, code, kwargs in (
        (hrv_view, "HRV", {"heart_choice": HeartChoice(kind="simulator")}),
        (heg_view, "HEG", {"mendi_choice": MendiChoice(kind="simulator")}),
    ):
        bar = toolbar()
        assert bar.channel_menu.visible is not False, "it starts visible, as the EEG screens need it"
        session = screen.session_for(BY_CODE[code], **kwargs)
        try:
            screen.view(StubPage(), session, bar, lambda: None)
            assert bar.channel_menu.visible is False, f"{code} still offers the channel count"
        finally:
            session.stop()


def test_calibration_set_on_one_screen_is_there_on_the_next():
    """**The room is one room.** Preferences were keyed by protocol code, so a click volume balanced by ear
    on the amplitude screen did not exist on the infra-low one — the same person, the same chair, the same
    speakers, and the same three numbers set again on every protocol.

    The one real exception is expressed rather than generalised from: `pinned_feedback` names what a
    protocol owns, which is already why those rows are inert on screen, and a pinned mechanism is neither
    stored nor restored. So alpha-theta's quieter music cannot leak out to a screen that never asked for it.
    """
    from nfb_console import at_view, ilf_view, prefs
    import nfb_console.amplitude_view as av

    prefs.PATH.unlink(missing_ok=True)

    amp = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    amp.feedback.click.volume = 42.0
    amp.stop()                                  # `stop` is what writes them

    ilf = ilf_view.ILFSession(BY_CODE["ILF"], source=replay(), channels=2)
    try:
        assert prefs.restore("ILF", ilf) > 0
        assert ilf.feedback.click.volume == 42.0, "set on one screen, there on the next"
    finally:
        ilf.stop()

    # Alpha-theta pins the click, so it neither takes the shared number nor overwrites it.
    at = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=2)
    try:
        assert "click" in at.pinned_feedback
        prefs.restore("A-T", at)
        assert at.feedback.click.volume != 42.0, "a pinned mechanism is the protocol's, not the room's"
        at.feedback.click.volume = 7.0
    finally:
        at.stop()

    again = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        prefs.restore("AMP", again)
        assert again.feedback.click.volume == 42.0, "and what it pins never reaches anybody else"
    finally:
        again.stop()


def test_a_screen_training_below_the_judged_band_says_the_gate_cannot_see_it():
    """**"rejecting 2% of the session" is true and reads as cover the measurement does not have.**

    The gate judges excursions on 0.5–50 Hz, and that floor is right — following the amplifier down to its
    0.01 Hz corner let drift set the threshold and held 78–91% of every block. The cost is a blind spot
    exactly where the infra-low screens live: at a 0.05 Hz reward, drift and movement pass straight into the
    measurement while the panel reports rejection is on and working.

    Asserted on the amplitude screen too, in the negative, because a warning that appears everywhere says
    nothing: there is no band under 0.5 Hz on that screen and nothing for it to warn about.
    """
    from nfb_console import ilf_view, signal_pane
    import nfb_console.amplitude_view as av

    def artifact_text(session) -> str:
        pane = signal_pane.SignalPane(StubPage(), session, lambda: None)
        return " ".join(x.value for x in _texts(pane.artifact_rows))

    ilf = ilf_view.ILFSession(BY_CODE["ILF"], source=at_corner(0.01), channels=2)
    try:
        ilf.set_frequency(0.05)
        assert ilf.trained_floor_hz < 0.5
        text = artifact_text(ilf)
        assert "trains below 0.5 Hz" in text, "the panel claims cover it does not have"
        assert "not rejected" in text
    finally:
        ilf.stop()

    # The two screens whose targets carry no band at all, and which are the extreme case: both are
    # lowpasses from DC, so the default derivation would have read nothing and said nothing.
    from nfb_console import scp_view, wave_view

    for screen, code in ((wave_view.WaveSession, "WAVE"), (scp_view.SCPSession, "SCP")):
        session = screen(BY_CODE[code], source=at_corner(0.01), channels=2)
        try:
            assert session.trained_floor_hz < 0.5
            assert "trains below 0.5 Hz" in artifact_text(session), code
        finally:
            session.stop()

    amp = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        assert amp.trained_floor_hz >= 0.5, "the lowest band on this screen is theta"
        assert "trains below" not in artifact_text(amp), "a warning shown everywhere says nothing"
    finally:
        amp.stop()


def test_the_eeg_music_offers_the_pitch_it_travels_over_and_says_what_it_does():
    """The row that replaced the carrier, and the note that does the arithmetic.

    A carrier frequency was a control that could not change what it was asked to change: adding 220 to a
    2–38 Hz band gives 182–258 Hz however it is set, so the sound stayed the same note and the panel had
    no answer to "make it move". The two pitch ends are that answer, and the note under them says whether
    the pair leaves the EEG's own intervals alone or exaggerates them — which is a ratio of two ranges,
    not something to work out while running a session.
    """
    from nfb_console import signal_pane
    import nfb_console.amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        session.feedback.music.enabled = True
        session.feedback.music.source = "eeg"
        pane = signal_pane.SignalPane(StubPage(), session, lambda: None)
        pane.rebuild()
        shown = " ".join(text.value for text in _texts(pane.control))
        assert "pitch" in shown and "carrier" not in shown
        assert "an exact transposition, ×27.5" in shown, "8–16 Hz across 220–440 is one octave each way"

        session.feedback.music.eeg_pitch_hi_hz = 880.0
        pane.rebuild()
        widened = " ".join(text.value for text in _texts(pane.control))
        assert "intervals ×2.0" in widened and "wider apart than they are" in widened
    finally:
        session.stop()


def test_the_music_status_line_says_what_the_player_is_doing_now():
    """**It was written once, when the panel was built, and never again.**

    Choosing a source rebuilds this pane immediately and the stream starts on the next output apply a
    fraction of a second later — so the line captured the instant before the sound began and held it. On
    screen: "waiting for a channel" under the EEG source, and "not started" under noise, for the whole
    session, with the music audibly playing. A player starts, stops, is held by a failing target and can
    fail outright; none of that is expressible in a string written at build time.
    """
    from nfb_console import signal_pane
    import nfb_console.amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        session.feedback.music.enabled = True
        session.feedback.music.source = "eeg"
        pane = signal_pane.SignalPane(StubPage(), session, lambda: None)
        pane.rebuild()
        assert "waiting for a channel" in pane.music_state.value, "nothing is playing yet"

        # The player starts. Nothing about the panel changed, so only a refresh can notice.
        session.music.sonifier.rate_hz = 250.0
        session.music.sonifier._stream = object()
        try:
            pane.control.visible = True     # the tab is open; hidden panes skip painting
            pane.refresh()
            assert "waiting" not in pane.music_state.value
            assert "sounding at" in pane.music_state.value
        finally:
            session.music.sonifier._stream = None
    finally:
        session.stop()


def test_the_default_inhibits_measure_something_on_a_single_channel():
    """**The sum of one channel is that channel**, and gating it on a second would have made this the
    change that broke the commonest setup.

    The inhibits open on the summed signal, because contamination is not site-specific — a clench, a
    drifting electrode or mains pickup arrives on whatever is wired, so a veto watching one channel is
    blind to the input nobody is looking at. `ChannelAligner` already handles one channel by handing back
    each block's own run, so the measuring aligner exists whatever is wired and a one-electrode session
    gets two inhibits that read rather than two cards that never move.

    The *spectrum's* aligner still needs two, and that is the opposite call for the opposite reason: a sum
    spectrum beside a single channel's is the same picture twice, offered as though it were a choice.
    """
    import nfb_console.amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        assert session.aligner is not None, "the sum has to exist for the inhibits to read"
        assert session.display_aligner is None and session.sum_ring is None

        inhibits = [t for t in session.targets if t.role == "inhibit"]
        assert inhibits and all(t.source == av.PAIR for t in inhibits)
        # The estimator behind them has to actually run, which is the thing a missing aligner silently
        # prevented. The bench fixture is open inputs, so the readings are zero and the channel reads dead
        # — that is not the subject here; that a summed target on one channel is measured at all is.
        pump(session, until=lambda: any(k[0] == av.SUM_KEY for k in session.amplitudes))
        assert any(k[0] == av.SUM_KEY for k in session.amplitudes), \
            "the sum estimator never produced a reading"
    finally:
        session.stop()


def test_a_second_click_on_stop_does_not_reset_the_cards():
    """**Two clicks on Stop threw away the afternoon's setting up**, intermittently, which is the shape it
    was reported as: "sometimes when pressing Stop the settings will be reset".

    Building a screen is slow — it opens the device, constructs the session and lays out every control —
    and Flet dispatches `on_click` on a thread pool, so a second click lands *inside* the first call.
    `self.session` was cleared for the duration of the build, so the second call found nothing to adopt
    from, took the `prefs.restore` branch, and came up on the protocol's defaults: bands back to 8-12 Hz,
    every threshold, filter and role gone.

    Reproduced without threads by re-entering `pick` from inside the build, which is exactly the
    interleaving that made it happen and is deterministic. The second entry is *dropped* rather than
    queued — running it afterwards would stop the session the first click just started — and dropping is
    also what keeps a re-entry on one thread from deadlocking on the guard.
    """
    from nfb_console import amplitude_view as av
    from nfb_console.app import Console, build_parser

    console = Console(StubPage(), build_parser().parse_args([RECORDING]))
    console.pick(BY_CODE["AMP"])
    tgt = console.session.targets[0]
    tgt.set_band(12.0, 15.0)
    console.session.feedback.click.volume = 42.0

    real = av.session_for
    reentered = []

    def slow_build(*a, **kw):
        session = real(*a, **kw)
        if not reentered:                 # the second click, landing inside the first build
            reentered.append(True)
            console.restart()
        return session

    av.session_for = slow_build
    try:
        console.restart()
    finally:
        av.session_for = real

    assert reentered, "the interleaving under test never happened"
    assert (console.session.targets[0].band.low,
            console.session.targets[0].band.high) == (12.0, 15.0), "the band went back to the default"
    assert console.session.feedback.click.volume == 42.0
    console.session.stop()


def test_changing_the_channel_count_keeps_the_cards():
    """The same reset, arrived at deterministically. `set_channels` stopped the session **and cleared it**,
    so the `pick` that follows had nothing to adopt from and rebuilt the protocol from its defaults.
    Stopping is what it needed; clearing is what threw the configuration away.
    """
    from nfb_console.app import Console, build_parser

    console = Console(StubPage(), build_parser().parse_args([RECORDING]))
    console.pick(BY_CODE["AMP"])
    console.session.targets[0].set_band(12.0, 15.0)
    console.session.feedback.click.volume = 42.0

    console.set_channels(1)

    assert (console.session.targets[0].band.low,
            console.session.targets[0].band.high) == (12.0, 15.0)
    assert console.session.feedback.click.volume == 42.0
    console.session.stop()



def _armed_rewards(session, value: float = 10.0, reach: float = 8.0):
    """Two settled rewards reading `value` against a higher line at `reach`, which is the state the
    detector is supposed to fire on. Set rather than trained, because what is under test is the gate.

    The channels are declared live because the bench recording is a no-electrode capture and the device
    says so — which the detector correctly refuses to announce anything on."""
    session.state = dict.fromkeys(session.specs, True)
    for tgt in session.rewards:
        tgt.threshold, tgt.reach_threshold = reach / 2.0, reach
        # Past its first window, whichever mode it is in: a tracker is provisional until the window has
        # filled once, and the detector refuses to announce on a provisional line.
        tgt._rate_hz = 20.0
        tgt.value, tgt.total, tgt._settled = value, int(tgt.window * 20.0) + 100, True
    return session.rewards


def test_a_reward_carries_a_higher_line_derived_from_the_same_window():
    """One window, two lines, and the higher one cannot fall below the first — the two aims can route to
    different estimators, and two estimators can cross."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        pump(session)
        for tgt in session.rewards:
            assert tgt.reach_rate == cfc_view.REACH_RATE
            tgt.redefine()
            if tgt._settled:
                assert tgt.reach_threshold >= tgt.threshold
                assert tgt.reaches(tgt.reach_threshold + 1.0)
                assert not tgt.reaches(tgt.threshold)
    finally:
        session.stop()


def test_a_bin_has_no_higher_line_and_no_second_mark():
    """Every other target in the app keeps exactly one threshold. The second line is one screen's."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert all(t.reach_rate is None for t in session.bin_targets)
        assert not any(t.reaches(1e6) for t in session.bin_targets)
    finally:
        session.stop()


def test_a_nudge_carries_the_higher_line_with_it():
    """A nudge is a judgement about the signal, not about one of two aims, so both lines move and the
    order between them survives."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        tgt = session.rewards[0]
        tgt.threshold, tgt.reach_threshold = 10.0, 16.0
        tgt.nudge(1.1)
        assert tgt.threshold == pytest.approx(11.0)
        assert tgt.reach_threshold == pytest.approx(17.6)
        tgt.nudge(0.9)
        assert tgt.reach_threshold > tgt.threshold
    finally:
        session.stop()


def test_the_coincidence_needs_every_reward_and_a_held_window():
    """Both past the higher line, held as a duty cycle, and announced once per crossing."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    played: list[str] = []
    session.cue = lambda name, volume=None: played.append(name)
    try:
        pump(session)
        rewards = _armed_rewards(session)
        session.set_reach_response("tone")
        assert session.coincident

        rewards[0].value = 0.0
        assert not session.coincident, "one reward past its line is not the event"
        rewards[0].value = 10.0

        now = time.monotonic()
        session._reach_log.extend((now - cfc_view.REACH_HOLD_SECONDS - 0.5 + i * 0.1, True)
                                  for i in range(30))
        session._maybe_click()
        assert len(played) == 1, "the coincidence announced once it had held"
        session._maybe_click()
        assert len(played) == 1, "and not again inside the re-arm interval"
        assert session.reach_count == 1
        assert session.marks() and "Hz" in session.marks()[0][1]
    finally:
        session.stop()


def test_a_flickering_coincidence_is_not_announced():
    """The condition flickers across a line; an unbroken-run test would miss real states and an
    ungated one would machine-gun. Neither happens: a mostly-false window says nothing."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    played: list[str] = []
    session.cue = lambda name, volume=None: played.append(name)
    try:
        pump(session)
        _armed_rewards(session)
        session.set_reach_response("tone")
        now = time.monotonic()
        session._reach_log.extend((now - cfc_view.REACH_HOLD_SECONDS - 0.5 + i * 0.1, i % 2 == 0)
                                  for i in range(30))
        session._maybe_click()
        assert played == [], "half a window is not a held state"
    finally:
        session.stop()


def test_a_provisional_threshold_announces_nothing():
    """A line derived from a fraction of a window is a guess, and announcing the crossing of a guess is
    worse than saying nothing."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        pump(session)
        _armed_rewards(session)
        for tgt in session.rewards:
            tgt.total, tgt._settled = 100, False      # back inside the first window, in either mode
        assert all(t.provisional for t in session.rewards)
        assert not session.coincident
    finally:
        session.stop()


def test_the_tone_is_silent_until_it_is_asked_for_but_the_record_is_not():
    """Silence is the reward on this screen, so the sound is armed deliberately. Everything else — the
    marks, the count, the moments kept beside a recording — is collected either way."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    played: list[str] = []
    session.cue = lambda name, volume=None: played.append(name)
    try:
        pump(session)
        assert session.reach_response == "silent", "off unless somebody asks for it"
        _armed_rewards(session)
        now = time.monotonic()
        session._reach_log.extend((now - cfc_view.REACH_HOLD_SECONDS - 0.5 + i * 0.1, True)
                                  for i in range(30))
        session._maybe_click()
        assert played == []
        assert session.reach_count == 1 and session.marks(), "counted and kept, just not sounded"
    finally:
        session.stop()


def test_a_pause_cannot_be_half_of_a_coincidence():
    """Samples from before a pause must not count towards a state nobody was present for."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        pump(session)
        session._reach_log.append((time.monotonic(), True))
        session.toggle_pause()
        assert not session._reach_log
    finally:
        session.stop()


def test_the_coupling_panel_says_what_the_second_line_is():
    """The legend names both lines, and the count line is present before there is anything to count —
    saying what it is waiting for rather than going blank."""
    from nfb_console import cfc_view

    page = StubPage()
    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        captions = " | ".join(c.value for c in _texts(control))
        assert "dashed = the higher one" in captions
        assert "waiting for both thresholds" in captions

        pump(session)
        _armed_rewards(session)
        # the plot is advanced from the draw loop, so fill it the way the draw loop would
        for n in range(320):
            session.push_displays(n * (cfc_view.REWARDS_SECONDS / 300.0))
        session.redraw()
        captions = " | ".join(c.value for c in _texts(control))
        assert "both above the higher line 100% of the window" in captions, captions
        assert "announced" in captions
        for text in _texts(control):
            value = text.value or ""
            if "higher line" in value or "announced" in value:
                assert len(value) <= 110, value
    finally:
        session.stop()


def test_a_bin_sounds_the_alert_rather_than_pausing_the_playback():
    """A bin over its threshold either pauses the playback or sounds an
    alert. The alert is the default, because pausing on the bins makes playback jerky and leaves the two
    kinds of failure indistinguishable. A reward below threshold pauses it in either variant, and Quiet
    Reward keeps the whole condition because there the silence has to be earned by both."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        pump(session)
        session.state = dict.fromkeys(session.specs, True)
        session.feedback.apply_model("pause")
        assert session.feedback.music.stops_when == "failing"
        assert session.feedback.alert.enabled, "the bins keep a consequence in the same arrangement"

        for tgt in session.rewards:
            tgt.threshold, tgt.value, tgt.total = 1.0, 10.0, 100
        for tgt in session.bin_targets:
            tgt.threshold, tgt.value, tgt.total = 1.0, 10.0, 100     # every bin over its threshold
        assert not session.music_hold(met=False), "twelve bins over threshold must not pause the playback"

        session.rewards[0].value = 0.0
        assert session.music_hold(met=False), "a reward below threshold still pauses it"

        session.feedback.apply_model("quiet")
        assert session.feedback.music.stops_when == "met"
        assert not session.music_hold(met=False), "silence is earned by the whole condition, bins included"
        assert session.music_hold(met=True)
    finally:
        session.stop()


def test_the_music_row_says_what_it_actually_watches():
    """A control whose behaviour differs from its label is a defect however good the reason, so the
    screen states the narrowed condition where the operator reads it."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        phrase, note = session.music_failing
        assert "reward" in phrase
        assert "alert" in note and "pausing" in note
        assert len(f"stops while {phrase}") <= 110 and len(note) <= 110
    finally:
        session.stop()


# --- a target measured on a signed combination of channels -------------------------------------

def test_a_difference_of_two_channels_is_measured_as_the_difference():
    """Two channels carrying the same rhythm in phase cancel; the same two in antiphase add.

    That is the whole content of the Custom source. With a linked reference every channel is
    site-minus-reference, so ch0 − ch1 is (C4 − Pz) − (C3 − Pz) — the reference cancels and what is left
    is C4 − C3, a montage that would otherwise need rewiring.

    Asserted as a ratio against the same signal measured on one channel, because the absolute number
    depends on the amplitude convention the shipped filter reports and the ratio does not."""
    import numpy as np

    from nfb.dsp import ALPHA
    from nfb.stream import Block
    from nfb_console import amplitude_view as av
    from nfb_console.targets import PAIR

    def amplitude(phase: float | None) -> float:
        """`phase` None measures ch0 alone; otherwise ch0 − ch1 with ch1 shifted by it."""
        session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
        try:
            session.paused = False
            session.targets.clear()
            tgt = session.add_target(0, ALPHA)
            if phase is not None:
                session.set_combination(tgt, [0, 1], [1, -1])
                assert tgt.source == PAIR and tgt.source_label == "ch0−ch1"

            spec = session.specs[0]
            rate = spec.rate_hz
            ticks = np.arange(int(20 * rate)) / rate
            a = 20.0 * np.sin(2 * np.pi * 10.0 * ticks)
            b = 20.0 * np.sin(2 * np.pi * 10.0 * ticks + (phase or 0.0))
            for i in range(0, a.size - 7, 8):
                session.feed_block(Block(0, i, to_counts(a[i:i + 8], spec.scale), session.specs[0]))
                session.feed_block(Block(1, i, to_counts(b[i:i + 8], spec.scale), session.specs[1]))
            return session.value_for(tgt)
        finally:
            session.stop()

    one = amplitude(None)
    assert one > 1.0, "the single-channel reading this is measured against"
    assert amplitude(0.0) / one < 0.05, "identical channels subtract to nothing"
    assert 1.8 < amplitude(np.pi) / one < 2.2, "opposed channels subtract to twice one channel"


def test_a_difference_and_a_sum_of_the_same_channels_are_two_measurements():
    """They cannot share an estimator or a reading: on the same two channels they are two montages, and
    the keys are what keep them apart."""
    from nfb.dsp import ALPHA
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        session.targets.clear()
        diff = session.add_target(0, ALPHA)
        session.set_combination(diff, [0, 1], [1, -1])
        added = session.add_target(0, ALPHA)
        session.set_combination(added, [0, 1], [1, 1])
        assert session.key_for(diff) != session.key_for(added)
        assert diff.source_label == "ch0−ch1" and added.source_label == "ch0+ch1"
    finally:
        session.stop()


def test_a_signed_combination_does_not_quietly_follow_the_live_channels():
    """A sum that loses a contributor is a smaller sum. A difference that loses one is a different
    montage, so it goes inert instead — the card still names both channels and the target stops being
    live, which is what `target_is_live` already says about a measurement spanning channels."""
    import numpy as np

    from nfb.dsp import ALPHA
    from nfb.stream import NO_SIGNAL, Block
    from nfb_console import amplitude_view as av
    from nfb_console.targets import PAIR

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=3)
    try:
        session.paused = False
        session.targets.clear()
        diff = session.add_target(0, ALPHA)
        session.set_combination(diff, [0, 1], [1, -1])
        summed = session.add_target(PAIR, ALPHA)

        spec = session.specs[0]
        rate = spec.rate_hz
        rail = int(round(800.0 / spec.scale))
        ticks = np.arange(int(4 * rate)) / rate
        wave = 20.0 * np.sin(2 * np.pi * 10.0 * ticks)
        for i in range(0, wave.size - 7, 8):
            live = to_counts(wave[i:i + 8], spec.scale)
            session.feed_block(Block(0, i, live, session.specs[0]))
            session.feed_block(Block(1, i, (rail,) * 8, session.specs[1], state=NO_SIGNAL))
            session.feed_block(Block(2, i, live, session.specs[2]))

        assert session.state[1] is False
        assert summed.pair == (0, 2), "the sum drops the dead channel and carries on"
        assert diff.pair == (0, 1), "the difference keeps naming both — it is that montage or none"
        assert not session.target_is_live(diff)
    finally:
        session.stop()


def test_the_custom_source_dialog_hugs_its_four_rows():
    """Shipped once as a dialog the height of the window with four short rows stranded at the top: a Flet
    column fills the main axis unless it is told not to. Also one segmented control per channel rather
    than three loose pills — three boxes in a row read as three independent switches, which is the thing
    a sign is not."""
    import flet as ft

    from nfb_console import amplitude_view as av

    page = StubPage()
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=4)
    try:
        control = av.view(page, session, toolbar(), lambda: None)

        menus = []

        def walk(node, depth=0):
            if node is None or depth > 22:
                return
            if isinstance(node, ft.PopupMenuButton):
                menus.append(node)
            for attr in ("controls", "items", "actions"):
                for child in getattr(node, attr, None) or []:
                    walk(child, depth + 1)
            for attr in ("content", "control"):
                walk(getattr(node, attr, None), depth + 1)

        walk(control)
        custom = None
        for menu in menus:
            for item in menu.items or []:
                text = getattr(item.content, "value", "")
                if text.startswith("Custom"):
                    custom = item
        assert custom is not None, "the source menu offers a custom combination"

        custom.on_click(None)
        dialog = page.dialogs[-1]
        body = dialog.content.content
        assert body.tight, "the column hugs its rows instead of taking the whole dialog"
        assert body.width <= 320 and dialog.content.width <= 320, "four short rows need no more"

        rows = [c for c in body.controls if isinstance(c, ft.Row) and getattr(c, "data", None) is not None]
        assert len(rows) == 4, "one row per channel"
        group = rows[0].controls[1]
        assert isinstance(group, ft.Container) and group.border is not None, \
            "the three signs are one bordered control, not three"
        labels = [c.content.value for c in group.content.controls if getattr(c, "content", None) is not None]
        assert labels == ["+", "−", "off"]
    finally:
        session.stop()


# --- Coupling -----------------------------------------------------------------------------

def _coupled_signal(seconds: float, depth: float, rate: float, seed: int = 0):
    """A slow rhythm whose phase decides when a fast one is loud, plus noise.

    Narrowband noise rather than sines, so the slow phase wanders as a real rhythm's does — with a
    perfect sine the surrogate floor cannot work, because shifting a periodic signal only rotates it.
    See the note at the top of `tests/test_coupling.py`.
    """
    import numpy as np
    from scipy import signal as sp

    rng = np.random.default_rng(seed)
    n = int(seconds * rate)

    def band(low, high):
        sos = sp.butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
        y = sp.sosfiltfilt(sos, rng.standard_normal(n))
        return y / y.std()

    slow = 30 * band(4.0, 8.0)
    gate = 1.0 - depth * np.cos(np.angle(sp.hilbert(slow)))
    return slow + gate * 5 * band(30.0, 45.0) + 4 * rng.standard_normal(n)


def _run_coupling(depth: float, seconds: float = 90.0):
    from nfb.stream import Block
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=1)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.paused = False
    session.state = dict.fromkeys(session.specs, True)
    spec = session.specs[0]
    x = _coupled_signal(seconds, depth, spec.rate_hz)
    for i in range(0, x.size - 249, 250):
        session.feed_block(Block(0, i, to_counts(x[i:i + 250], spec.scale), spec))
    session.redraw()
    return session, page, control


def test_the_coupling_screen_separates_a_real_coupling_from_none_at_all():
    """The whole screen, end to end, on a signal whose coupling is known because it was built in.

    Not a unit test of the measure — that is `tests/test_coupling.py`. This one exists because the
    measure being right is not the same as the *screen* being right: the value has to reach the target,
    the target has to be the one with no band on it, and the plot has to be of the same number.
    """
    strong, _, _ = _run_coupling(0.9)
    none, _, _ = _run_coupling(0.0)
    try:
        reward = next(tg for tg in strong.targets if tg.band is None)
        quiet = next(tg for tg in none.targets if tg.band is None)
        assert strong.value_for(reward) > 5 * none.value_for(quiet), (
            f"coupled read {strong.value_for(reward):.3f}, uncoupled "
            f"{none.value_for(quiet):.3f}")
        assert strong.value_for(reward) == strong.live_coupling.modulation * 100
        assert strong.plot.peak() > 0, "the plot draws the same number the card does"
    finally:
        strong.stop()
        none.stop()


def test_the_coupling_screen_says_when_the_reading_is_only_the_floor():
    """A reading that has not cleared its floor is not coupling, and the caption has to say so rather
    than leaving a number on screen for the operator to over-read."""
    session, _, control = _run_coupling(0.0)
    try:
        said = " ".join(c.value for c in _texts(control) if c.value)
        assert "at the floor" in said, said
    finally:
        session.stop()

    session, _, control = _run_coupling(0.9)
    try:
        said = " ".join(c.value for c in _texts(control) if c.value)
        assert "clear of the floor" in said, said
        assert "at its lowest" in said, "the preferred phase is named in plain words, not only degrees"
    finally:
        session.stop()


def test_the_coupling_reward_has_no_band_and_the_muscle_guard_does():
    """A coupling is one number derived from two bands, so it has no edges of its own — which is what
    keeps a filter gear and two edge boxes off a card that has nothing to put in them. The guard beside
    it is an ordinary band inhibit and does have all of that."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=1)
    try:
        reward = next(tg for tg in session.targets if tg.band is None)
        guard = next(tg for tg in session.targets if tg.band is not None)
        assert reward.role == "reward" and reward.unit == "%"
        assert guard.role == "inhibit" and guard.band == pac_view.GUARD
        # Both modes, automatic on a period by default — a tracker re-derived continuously would pin
        # the percentage being trained, and the period is what stops that without forbidding tracking.
        assert "automatic" in session.threshold_modes and "fixed" in session.threshold_modes
        assert reward.mode == "automatic" and reward.recompute_seconds == 90.0
        assert reward.window == reward.calibration_seconds == 120.0, (
            "re-derived from as much signal as the opening calibration used")
        assert guard.mode == "automatic" and guard.recompute_seconds == 90.0
    finally:
        session.stop()


def test_the_coupling_screen_refuses_a_fast_band_that_cannot_show_the_modulation():
    """Adding a coupling whose fast band is narrower than twice the slow centre frequency returns
    nothing rather than adding a target that would read zero for ever."""
    from nfb.dsp.filters import Band
    from nfb.dsp import Coupling
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=1)
    try:
        before = len(session.targets)
        with pytest.raises(ValueError):
            Coupling(pac_view.SLOW, Band(30.0, 36.0))
        assert session.add_coupling(Coupling(pac_view.SLOW, Band(100.0, 140.0))) is None, (
            "140 Hz is above this session's Nyquist at 250 sps")
        assert len(session.targets) == before
    finally:
        session.stop()


def test_the_balance_numbers_show_from_the_first_seconds_and_say_how_much_they_rest_on():
    """**Feedback from the first sample, provisional if it must be.** Both numbers used to be withheld
    for five minutes, and the caption counting down to that divided by ten where the estimator emits
    twenty readings a second — so it read "346 s of 300 s needed" with nothing shown. Now the numbers
    appear as soon as the estimator can fit, the caption says how much envelope is behind them from the
    estimator's own rate, and calls them provisional under five minutes rather than hiding them."""
    from nfb_console import pac_view

    session, _, control = _run_coupling(0.5, seconds=40.0)
    try:
        assert session.memory is not None and session.balance is not None
        assert abs(session.balance_seconds - 40.0) < 3.0, session.balance_seconds
        said = " ".join(c.value for c in _texts(control) if c.value)
        assert "provisional until 300 s" in said, said
        assert "needed" not in said and "not fed back" not in said, said
        assert f"from {session.balance_seconds:.0f} s" in said, said
    finally:
        session.stop()


def test_changing_the_averaging_window_rebuilds_every_coupling_together():
    """All of them, so they stay comparable — the same rule the synchrony screen applies to its bins."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=1)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    try:
        pills = _find_pills(control, "40 s")
        assert pills is not None, "the window is a control, not a constant"
        pills.on_click(None)
        assert session.window_seconds == 40.0
        assert all(m.window_seconds == 40.0 for m in session.couplings.values())
        assert not [k for k in session.amplitudes if k[0] == "pac"], (
            "readings taken over the old window are not readings of the new one")
    finally:
        session.stop()


def _find_pills(control, label, depth: int = 0):
    import flet as ft

    if depth > 14 or control is None:
        return None
    if isinstance(control, ft.Container) and getattr(control, "on_click", None) is not None:
        for text in _texts(control):
            if text.value == label:
                return control
    for child in (getattr(control, "controls", None) or []):
        found = _find_pills(child, label, depth + 1)
        if found is not None:
            return found
    return _find_pills(getattr(control, "content", None), label, depth + 1)


def _locked_pair(seconds, strength, rate, seed=0):
    """Two channels sharing an alpha phase relationship to a controlled degree."""
    import numpy as np
    from scipy import signal as sp

    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    sos = sp.butter(4, [8.0, 12.0], btype="bandpass", fs=rate, output="sos")

    def band():
        y = sp.sosfiltfilt(sos, rng.standard_normal(n))
        return y / y.std()

    a = band()
    shared = sp.sosfiltfilt(sos, np.imag(sp.hilbert(a)))
    shared /= shared.std()
    b = strength * shared + np.sqrt(max(0.0, 1 - strength ** 2)) * band()
    return 20 * a + 3 * rng.standard_normal(n), 20 * b + 3 * rng.standard_normal(n)


def _run_locking(strength: float, seconds: float = 90.0):
    from nfb.stream import Block
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.paused = False
    session.state = dict.fromkeys(session.specs, True)
    spec = session.specs[0]
    a, b = _locked_pair(seconds, strength, spec.rate_hz)
    for i in range(0, a.size - 249, 250):
        session.feed_block(Block(0, i, to_counts(a[i:i + 250], spec.scale), session.specs[0]))
        session.feed_block(Block(1, i, to_counts(b[i:i + 250], spec.scale), session.specs[1]))
    session.redraw()
    return session, control


def test_the_screen_separates_a_held_phase_relationship_from_none():
    """The second half of what the source framework proposes, end to end.

    It is on the screen at all because the earlier reason for leaving it off was wrong: a reading of
    0.73 on unrelated channels was measured on a band twenty times narrower than this one, and at
    ordinary widths the same null is a fraction of that and is removed exactly.
    """
    held, held_control = _run_locking(0.8)
    apart, apart_control = _run_locking(0.0)
    try:
        # Over the drawn window rather than the latest value: one sample of a debiased phase statistic
        # swings by about 1/sqrt(N), so a single reading is not what either the screen or a test should
        # be judging.
        assert held.lock_plot.recent_mean() > 3 * apart.lock_plot.recent_mean()
        assert apart.lock_plot.recent_mean() < 20.0, "unrelated channels, on a 0-100 scale"
        tgt = next(tg for tg in held.targets if tg.name == held.live_locking.locking.label)
        assert held.value_for(tgt) == held.live_locking.locked * 100

        assert "holding steady" in " ".join(c.value for c in _texts(held_control) if c.value)
        assert "nothing beyond chance" in " ".join(
            c.value for c in _texts(apart_control) if c.value)
    finally:
        held.stop()
        apart.stop()


def test_the_locking_floor_stays_at_the_bottom_while_the_reading_climbs():
    """The dashed line is what says the correction is describing this signal, so it is asserted against
    a real locked pair rather than only against noise."""
    session, _ = _run_locking(0.8)
    try:
        metric = session.live_locking
        assert metric.floor < 0.15 < metric.locked, (
            f"floor {metric.floor:.3f} against a reading of {metric.locked:.3f}")
    finally:
        session.stop()


def test_a_one_channel_coupling_session_has_no_locking_target_and_says_why():
    """Phase locking is between two sites, so on one channel the panel states that rather than drawing
    a flat line that looks like a measurement of nothing."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=1)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.redraw()
    try:
        assert not session.lockings
        assert session.live_locking is None
        said = " ".join(c.value for c in _texts(control) if c.value)
        assert "phase locking needs two" in said, said
    finally:
        session.stop()


def test_the_coupling_screen_uses_every_channel_it_was_given():
    """**One site drives the sound; all of them are on screen.**

    It used to open with one coupling whatever the channel count was, so four electrodes bought one
    measurement and nothing said why the first channel was special — it is not, it is merely first.

    The rest open as monitors, which this codebase already means precisely: measured, calibrated, drawn
    and scored, with only the wire to the output cut. Four slow indices averaged into one level barely
    move, and a trainee cannot tell which site they are affecting.
    """
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=4)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.redraw()
    try:
        couplings = [tg for tg in session.targets if session.key_for(tg) in session.couplings]
        assert len(couplings) == 4, [tg.name for tg in couplings]
        assert [tg.monitor for tg in couplings] == [False, True, True, True]
        assert len(session.plot.series()) == 4, "one drawn series per coupling"

        # Only what can pay out is guarded: ch0 carries the trained coupling, ch1 is in the locking.
        guards = {tg.name for tg in session.targets if tg.band == pac_view.GUARD}
        assert guards == {"ch0 muscle", "ch1 muscle, held"}, guards

        # And with every channel carrying the default pairing, the add menu says so rather than
        # silently doing nothing.
        assert not session.free_channels
        assert (0, 2) in session.free_pairs
        _find_menu_items(control, ("phase locking",))["phase locking"](None)
        assert any(tg.name.startswith("alpha ch0 ↔ ch2") for tg in session.targets)
    finally:
        session.stop()


def test_promoting_a_watched_coupling_brings_its_guard_with_it():
    """The role decides whether a site drives the sound, so the muscle inhibit follows it: a monitor
    cannot pay out, and an inhibit on a channel nothing is trained on is something to fail for no
    reason."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=4)
    try:
        watched = [tg for tg in session.targets
                   if session.key_for(tg) in session.couplings and tg.monitor]
        assert "ch2 muscle" not in [tg.name for tg in session.targets]

        session.set_role(watched[1], "reward")                 # ch2
        assert "ch2 muscle" in [tg.name for tg in session.targets]
        assert session.guard_kinds[2] == "fast"

        session.set_role(watched[1], "monitor")
        assert "ch2 muscle" not in [tg.name for tg in session.targets]
    finally:
        session.stop()


def _menu_clicks(control, label, depth: int = 0, found=None):
    """Every `on_click` for a menu item with this label, in the order they appear.

    A list rather than one, because a label is not unique across a screen — "beta" is in the coupling
    card's fast-band menu *and* in the locking card's band menu, which is correct on screen and
    ambiguous to a test. Callers take the one belonging to the card they mean.
    """
    import flet as ft

    found = [] if found is None else found
    if depth > 16 or control is None:
        return found
    if isinstance(control, ft.PopupMenuItem):
        if any(x.value == label for x in _texts(control)) and control.on_click is not None:
            found.append(control.on_click)
    for attr in ("controls", "items"):
        for child in (getattr(control, attr, None) or []):
            _menu_clicks(child, label, depth + 1, found)
    _menu_clicks(getattr(control, "content", None), label, depth + 1, found)
    return found


def _find_menu_items(control, labels):
    return {label: clicks[0] for label in labels
            for clicks in [_menu_clicks(control, label)] if clicks}


def test_two_couplings_on_different_channels_are_told_apart_and_removed_separately():
    """Both are the same two bands, so without the channel in the name both cards read alike — and the
    lookup that finds a target's metric by name returned whichever came first, so removing one took the
    other's muscle guard with it."""
    from nfb.dsp import Coupling
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        session.add_coupling(Coupling(pac_view.SLOW, pac_view.FAST, phase_channel=1))
        names = [tg.name for tg in session.targets]
        assert len(set(names)) == len(names), f"two cards with one name: {names}"

        session.drop(next(tg for tg in session.targets if tg.name == "theta→low gamma ch1"))
        left = [tg.name for tg in session.targets]
        assert "theta→low gamma ch0" in left and "ch0 muscle" in left, left
        # ch1 is still in the default locking, so it keeps a guard — but the *held* one now, since the
        # coupling that wanted the strict one is gone.
        assert "ch1 muscle, held" in left and "ch1 muscle" not in left, left
    finally:
        session.stop()


def test_each_coupling_card_shows_its_mode_and_period_without_claiming_a_source():
    """The articles say nothing about thresholds, so nothing on the screen implies they did. Each card
    carries its own mode pills with the period beside them — the one place the answer to "is this
    computed once or tracked" is, since it is per card — and the reasoning is behind the panel's ?."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=4)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.redraw()
    try:
        values = [c.value for c in _texts(control) if c.value]
        trained = [tg for tg in session.targets if not tg.monitor]
        assert values.count("automatic") == len(trained), (values.count("automatic"), len(trained))
        assert values.count("fixed") == len(trained)
        assert "protocol" not in pac_view.TRAINING_HELP
        assert "90 s" in pac_view.TRAINING_HELP
    finally:
        session.stop()


def _find_all_clickable(control, label, depth: int = 0, found=None):
    import flet as ft

    found = [] if found is None else found
    if depth > 14 or control is None:
        return found
    if getattr(control, "on_click", None) is not None:
        for text in _texts(control):
            if text.value == label:
                found.append(control)
                break
    for child in (getattr(control, "controls", None) or []):
        _find_all_clickable(child, label, depth + 1, found)
    _find_all_clickable(getattr(control, "content", None), label, depth + 1, found)
    return found


def test_changing_the_channel_count_on_the_coupling_screen_does_not_crash():
    """**It did, before a frame was drawn.** The base `adopt_targets` hands the target list over and calls
    `retune` on each, which only the band-editing screens have — so picking a different channel count
    raised `AttributeError: 'CouplingSession' object has no attribute 'retune'`.

    Rebuilding by settings rather than carrying objects is also what makes the count safe: an instrument
    whose electrode is gone has to be dropped, not kept naming a channel that no longer exists with no
    metric behind it.
    """
    from nfb.dsp import Coupling, Locking
    from nfb_console import pac_view

    old = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=4)
    old.add_coupling(Coupling(pac_view.SLOW, pac_view.FAST, phase_channel=2))
    old.add_locking(Locking(pac_view.LOCK, (2, 3)))
    old.window_seconds = 40.0
    try:
        for count, expect in ((1, ["theta→low gamma ch0", "ch0 muscle"]),
                              (2, ["theta→low gamma ch0", "ch0 muscle",
                                   "theta→low gamma ch1", "ch1 muscle, held",
                                   "alpha ch0 ↔ ch1"])):
            new = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=count)
            try:
                new.adopt(old)
                assert [tg.name for tg in new.targets] == expect, count
                assert new.window_seconds == 40.0, "a setting is not a statistic"
                page = StubPage()
                pac_view.view(page, new, toolbar(), lambda: None)
                new.redraw()
            finally:
                new.stop()

        # And the round trip keeps everything, so nothing is lost by passing through the device screen.
        same = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=4)
        try:
            same.adopt(old)
            assert len(same.couplings) == 4 and len(same.lockings) == 2
            # And which one was trained is a setting, not a statistic: losing it put four rewards on
            # the screen where the operator had left one.
            trained = [tg for tg in same.targets
                       if same.key_for(tg) in same.couplings and not tg.monitor]
            assert len(trained) == 1, [tg.name for tg in trained]
            assert trained[0].name == "theta→low gamma ch0"
        finally:
            same.stop()
    finally:
        old.stop()


def test_a_target_with_no_band_has_no_response_time_rather_than_raising():
    """`Target.band` documents None as a state — a coupling index, a difference between two targets — so
    the shared answer has to be "no burst at a centre frequency to answer about", not an AttributeError
    from inside the estimator builder."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        for tgt in session.targets:
            answer = session.response_time(tgt)
            assert answer is None if tgt.band is None else answer > 0
    finally:
        session.stop()


def test_the_coupling_screen_has_a_spectrum_to_show():
    """The tab was there with no source behind it. It is not decoration on this screen: the phase the
    whole measurement rests on is only interpretable where the slow band has a real peak, and this is
    where an operator sees whether it does."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        pump(session)
        assert [s.label for s in session.spectrum_sources()] == ["ch0", "ch1", "sum"]
        assert session.rings[0].psd() is not None
        assert session.trained_floor_hz == 4.0, (
            "the slow band's floor, not the 70 Hz muscle guard's — the only band the base could see")
    finally:
        session.stop()
def test_a_threshold_is_a_mark_inside_the_plot_and_not_its_top_edge():
    """A ceiling taken from the threshold draws that line along the frame, where it reads as the edge of
    the plot rather than as a mark on it — which is how it looked for the whole calibration."""
    from nfb_console import pac_view

    from nfb_console.targets import Target

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        shared = Target(0, None).threshold
        opening = {tgt.name: tgt.threshold for tgt in session.targets if tgt.band is None}
        assert opening, "the two instruments"
        for name, value in opening.items():
            assert value != shared, (
                f"{name} opened on the shared default, which is a number about microvolts")
        # Each in its own measure's range: a modulation index scaled to a percentage lives near 1, a
        # debiased locking near 20. Both well inside what the signal can reach, so the line is a mark
        # rather than the ceiling.
        assert opening["theta→low gamma ch0"] <= 2.0
        assert 5.0 <= opening["alpha ch0 ↔ ch1"] <= 50.0
    finally:
        session.stop()


def test_a_card_says_its_own_calibration_length_not_the_shared_default():
    """It read "set from the first 30 s, 120 s left" — a sentence contradicting itself — on every screen
    that gives a target its own calibration window, which is both of the slow ones.

    **The card now says only what is left**, and the window it is drawn from is the `?`'s to state: the
    explanation was printed under every card, and on a column of four it was the same sentence four
    times. So the countdown is checked against the target's own window, and the help text is checked
    for the window."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.paused = False
    session.state = dict.fromkeys(session.specs, True)
    pump(session)
    session.redraw()
    try:
        notes = [c.value for c in _texts(control) if c.value
                 and ("calibrating" in c.value or "provisional" in c.value)]
        assert notes, "the cards are still in their first window at this point"
        windows = {max(tg.window, tg.calibration_seconds) for tg in session.targets if not tg.monitor}
        assert 120.0 in windows, "the instruments' first window is two minutes"
        for note in notes:
            (left,) = re.findall(r"(\d+) s", note)
            assert int(left) <= max(windows), f"more time left than any window: {note}"
            assert "first" not in note, "the explanation belongs behind the ?, not on every card"
        coupling = next(tg for tg in session.targets if tg.calibration_seconds == 120.0)
        from nfb_console.target_card import TargetCard
        card = TargetCard(coupling, ft.Text(""), lambda: None, page=page)
        assert "120 s" in card._help_text()
    finally:
        session.stop()


def test_a_coupling_is_pointed_from_its_own_card():
    """**The fix for a clunky add-and-delete.** The card used to be dead text, so the only way to say
    "a coupling on the fourth electrode" was to add three and remove two. Every part of the name is the
    menu that sets it now, which is what the synchrony screen's reward cards have always done — and it is
    why the add button no longer has to guess what you meant.
    """
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=4)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.redraw()
    try:
        # Every channel already carries the default pairing, so moving ch0's coupling needs a band
        # change first — a channel is taken by *this* pairing, not by any coupling at all.
        _menu_clicks(control, "delta")[0](None)
        assert session.targets[0].name == "delta→low gamma ch0"

        control = pac_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        _menu_clicks(control, "ch3")[0](None)
        assert session.targets[0].name == "delta→low gamma ch3", (
            "one click from the card, once the pairing is free there")

        control = pac_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        _find_menu_items(control, ("delta",))["delta"](None)
        assert session.targets[0].name == "delta→low gamma ch3"

        # The first "beta" on the screen is the coupling card's fast-band menu; the locking card below
        # has one too, which is correct and is why the helper hands back every match in order.
        control = pac_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        _menu_clicks(control, "beta")[0](None)
        assert session.targets[0].name == "delta→beta ch3"
    finally:
        session.stop()


def test_a_band_pairing_the_measure_cannot_show_is_offered_and_disabled_with_the_reason():
    """The sideband rule — a fast band must be twice the slow centre wide — is worth learning once, so an
    impossible pairing is shown greyed with why rather than quietly missing from the menu."""
    import flet as ft
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        session.retune_coupling(session.targets[0], slow=pac_view.SLOW_BANDS["alpha"],
                                fast=pac_view.FAST_BANDS["wide gamma"])
        page = StubPage()
        control = pac_view.view(page, session, toolbar(), lambda: None)
        session.redraw()

        items = []

        def walk(ctrl, depth=0):
            if depth > 16 or ctrl is None:
                return
            if isinstance(ctrl, ft.PopupMenuItem):
                items.append((" ".join(x.value for x in _texts(ctrl)), bool(ctrl.disabled)))
            for attr in ("controls", "items"):
                for child in (getattr(ctrl, attr, None) or []):
                    walk(child, depth + 1)
            walk(getattr(ctrl, "content", None), depth + 1)

        walk(control)
        blocked = [label for label, off in items if off and "low gamma" in label]
        assert blocked, [label for label, off in items if off]
        assert "20 Hz" in blocked[0], (
            f"the reason has to be the rule, not just a greyed row: {blocked[0]}")
    finally:
        session.stop()


def test_the_muscle_guard_is_a_switch_on_its_coupling_rather_than_a_card_that_appears_alone():
    """It materialised beside the instruments when a coupling was added and vanished when one was
    deleted — a card nobody asked for, then gone for no visible reason, and the only card with no way to
    remove it. It is behind its coupling's gear now, and its own × turns the same switch off."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        coupling = session.targets[0]
        assert session.guarded(coupling)
        assert session.targets[1].name == "ch0 muscle", "directly under the coupling it guards"

        session.set_guarded(coupling, False)
        assert "ch0 muscle" not in [tg.name for tg in session.targets]
        session.set_guarded(coupling, True)
        assert session.targets[1].name == "ch0 muscle"

        # And it follows the coupling rather than the session: move the coupling, move the guard.
        moved = session.retune_coupling(coupling, channel=1)
        assert [tg.name for tg in session.targets][:2] == ["theta→low gamma ch1", "ch1 muscle"]
        session.set_guarded(moved, False)
        # The locking still guards both its channels; only the coupling's own guard went.
        assert "ch1 muscle" not in [tg.name for tg in session.targets]
    finally:
        session.stop()
def test_the_card_column_is_as_wide_as_every_other_screen_with_cards():
    """330 is the width the screens whose side panel holds readouts use. Taking it for a panel of cards
    that carry control strips left them about a hundred pixels short, so every menu wrapped onto a line
    of its own."""
    import importlib

    from nfb_console import pac_view

    widths = []
    for mod, attr in (("amplitude_view", "TARGETS_W"), ("at_view", "TARGETS_W"),
                      ("cfc_view", "TARGETS_W")):
        widths.append(getattr(importlib.import_module(f"nfb_console.{mod}"), attr))
    assert pac_view.SIDE_W == min(widths), (
        f"cards need {min(widths)} like the others, not {pac_view.SIDE_W}")


def test_every_launcher_card_fits_the_text_it_is_given():
    """**Four blurbs were clipped mid-word** — "twelve inhibi", "from 0.05", "if electrodes".

    The card keeps every text slot a fixed height on purpose, so a description that would wrap overflows
    visibly instead of shoving the big code letters to a different height than the card beside it. That
    is the right trade and it leaves the width a hard budget, which nothing was measuring. Clipping also
    has no ellipsis, so it reads as a rendering fault rather than as text that is too long.
    """
    from nfb_console import splash, theme
    from nfb_console.protocols import CATALOGUE

    from conftest import text_width

    slot = splash.CARD_W - splash.BAR_W - 16 - 14      # the card's padding, from `protocol_card`
    for p in CATALOGUE:
        line = p.blurb if p.available else p.note
        assert line, f"{p.code} has nothing to say in its subtitle"
        assert text_width(line, theme.SMALL) <= slot - 8, (
            f"{p.code}: {line!r} needs {text_width(line, theme.SMALL):.0f}px of {slot}")
        if p.name.upper() != p.code.upper():
            assert text_width(p.name, theme.BODY, bold=True) <= slot - 8, (
                f"{p.code}: the name {p.name!r} does not fit either")


def test_no_catalogue_card_opens_a_screen_that_is_not_its_protocol():
    """A card that is not greyed has to lead somewhere that does what the card says.

    Coherence was live with nothing declaring `HANDLES = ("COH",)`, so it fell through to the generic
    traces view — a button that fails after you click it, which is the exact thing this catalogue's own
    rule exists to prevent.
    """
    from nfb_console.app import Console
    from nfb_console.protocols import CATALOGUE

    for p in CATALOGUE:
        screen = Console.screen_for(p)
        if p.available:
            assert screen is not None, f"{p.code} is offered and has no screen"
            assert p.code in screen.HANDLES
        else:
            assert p.note, f"{p.code} is greyed and does not say why"


def test_the_launcher_scrolls_so_no_protocol_is_below_the_fold():
    """Eleven protocols is four rows of cards, and at the height the window opens at the fourth row is
    not on screen — HEG and HRV were unreachable with nothing saying so."""
    import flet as ft

    from nfb_console import splash

    page = StubPage()
    control = splash.view(page, StubDevice(), lambda _p: None, lambda: None)
    body = control.content
    assert isinstance(body, ft.Column)
    assert body.scroll is not None, "a launcher that hides protocols is worse than one that scrolls"


def test_a_monitor_card_is_a_reading_with_nothing_to_press():
    """A monitor cannot pay out, so a threshold, its nudges, a percentage met and a redefine button on its
    card are controls whose behaviour differs from their label. What is left is the identity row, the
    number, and a bar with a scale and no notch — and the number is never green, because a monitor is
    not wanted or unwanted."""
    from nfb_console import pac_view
    from nfb_console.target_card import TargetCard

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    page = StubPage()
    try:
        watched = next(tg for tg in session.targets if tg.monitor)
        trained = next(tg for tg in session.targets if tg.role == "reward")
        card = TargetCard(watched, ft.Text("id"), lambda: None, page=page)
        card.refresh(0.5, "%")
        said = [c.value for c in _texts(card.control) if c.value]
        assert "threshold" not in said and "redefine" not in said and "waiting" not in said, said
        assert not any("mark = threshold" in s for s in said), said
        assert card.notch.visible is False and card.value_text.color != "#248647"
        full = TargetCard(trained, ft.Text("id"), lambda: None, page=page)
        full.refresh(0.5, "%")
        assert "threshold" in [c.value for c in _texts(full.control)]
    finally:
        session.stop()


def test_the_coupling_feedback_tab_offers_only_what_works_on_it():
    """Four things found by reading the tab against the screen: the tone and the Shepard voice follow the
    level and were missing only because the base list predates them; the muscle guard averaged into the
    level under `weighted` so a clenched jaw paid out; `absolute` is a 5 µV step on a percent reading; and
    a click every 2 s on a 20 s average is narration. And "(soon)" is not a control, on any screen."""
    from nfb_console import pac_view
    from nfb_console.signal_pane import SignalPane

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    try:
        assert "tone" in session.music_sources and "shepard" in session.music_sources
        assert "crossfade" not in session.music_sources, "no two states to fade between"
        assert session.feedback.combine == "scaled" and "combine" in session.pinned_feedback
        assert "absolute" not in session.normalisations
        assert session.feedback.click.interval_seconds == pac_view.COUPLING_WINDOW_SECONDS
        pills = dict(_pills(control))
        assert "tone" in pills and "shepard" in pills, sorted(pills)
        assert "absolute" not in pills
        assert not any("(soon)" in c.value for c in _texts(control) if c.value)
    finally:
        session.stop()


def test_every_screen_with_a_spectrum_has_trends_and_offers_the_sum():
    """Two things that were true on some screens for no reason that survives being stated. Trends reads
    the same rings the Spectrum tab draws, so it belongs wherever Spectrum does; it was wired into two
    screens. And "sum" was offered only where a protocol had plumbed its own summed ring — four screens —
    while three with the same channel rings offered nothing. The sum is derived from the channel rings
    now, on any screen with two or more at one rate."""
    import importlib

    for module, code in (("cfc_view", "CFC"), ("at_view", "A-T"), ("pac_view", "PAC"),
                         ("scp_view", "SCP"), ("ilf_view", "ILF"), ("wave_view", "WAVE"),
                         ("amplitude_view", "AMP")):
        screen = importlib.import_module(f"nfb_console.{module}")
        session = screen.session_for(BY_CODE[code], source=replay(), channels=2)
        page = StubPage()
        control = screen.view(page, session, toolbar(), lambda: None)
        try:
            labels = [c.value for c in _texts(control) if c.value in ("Spectrum", "Trends")]
            assert labels.count("Trends") == 1, f"{code}: {labels}"
            sources = [s.label for s in session.spectrum_sources()]
            assert "sum" in sources, f"{code} offers {sources}"
        finally:
            session.stop()

    session = importlib.import_module("nfb_console.pac_view").session_for(
        BY_CODE["PAC"], source=replay(), channels=1)
    try:
        assert "sum" not in [s.label for s in session.spectrum_sources()], "one channel is nothing to add"
    finally:
        session.stop()


def test_the_rhythm_line_says_what_its_number_is_and_the_help_says_why():
    """It read "theta is 0.7× its neighbours — too flat for a phase to mean much": no unit, no named
    comparison, and a conclusion with no mechanism. One line now, every number with what it is and the
    line it is judged against; the reasoning — why a band with no peak has a phase that means nothing —
    is behind the panel's ?, where there is room to say it properly."""
    from nfb_console import pac_view

    session, _, control = _run_coupling(0.0, seconds=40.0)
    try:
        texts = [c.value for c in _texts(control) if c.value]
        line = next(v for v in texts if "is strongest while" in v)
        rhythm = next(v for v in texts if "the bands either side" in v)
        assert "amplitude" in rhythm and f"a peak is above {pac_view.RHYTHM_RATIO:g}" in rhythm, rhythm
        assert max(len(line), len(rhythm)) <= 110
        assert "too flat" not in rhythm and "neighbours" not in rhythm and "trough" not in line
        help_text = pac_view.COUPLING_HELP
        for phrase in ("always", "slope", "peak", "the filter's", "Spectrum tab"):
            assert phrase in help_text, phrase
    finally:
        session.stop()
