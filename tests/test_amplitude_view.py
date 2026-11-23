"""The amplitude screen: built, fed and drawn, and what its cards, sources and filters do."""

from __future__ import annotations

import math

import pytest

from nfb_console import chrome
from nfb_console.protocols import BY_CODE

from conftest import counts as to_counts, waitfor

from harness import (
    _feed_two_tones,
    _find_gear,
    _menu_texts,
    _texts,
    at_corner,
    pump,
    replay,
    show_all_panes,
    StubPage,
    toolbar,
)


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


def test_excluding_dead_channels_changes_nothing_while_they_are_alive():
    """The one shared change with teeth. Measured on the real two-channel export with the feed stopped so
    nothing moves between reads: identical on every read, because `active_targets` *is* `targets` when every
    channel reports signal. It only bites when a channel does not, which is the case it exists for."""

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


def test_the_convention_pill_carries_the_threshold_across():
    """Changing the amplitude convention from the signal pane used to set the field first and rebuild the
    estimators after, so `retune_filter`'s before/after snapshot read the same convention and the rescale
    never ran. A settled reward then sat ~10% wrong against meanabs until someone pressed redefine."""
    from nfb_console import amplitude_view as av

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=2)
    try:
        tgt = session.targets[0]
        assert session.filter_for(tgt)[0] != "cfir", "the default target runs an IIR estimator"
        for _ in range(int(35 * 20)):
            tgt.observe(10.0, 20.0)
        tgt.redefine()
        settled, threshold = tgt.recent_percent(), tgt.threshold

        session.set_convention("meanabs")
        ratio = (2.0 / math.pi) / (1.0 / math.sqrt(2.0))
        assert session.feedback.envelope_convention == "meanabs"
        assert tgt.threshold == pytest.approx(threshold * ratio, rel=1e-6), "the threshold moved to the unit"
        assert tgt.recent_percent() == settled, "and the readings came with it, so the score is unchanged"
        assert not tgt.calibrating, "the card does not go back to waiting"
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


def test_continuous_outputs_apply_on_their_own_thread_not_the_feed(tmp_path):
    """The output calls are the slowest things in the process — a CoreAudio write at 1.1 ms p99, a
    window-server display lookup at 13 ms worst, a subprocess spawn for the music — and they ran inline
    on the thread that decodes and filters against a 32 ms budget. `run` now starts an output pump;
    `_drive_outputs` snapshots and wakes it, so the feed thread never blocks on an output device.

    Fed a signal, not the bench recording: its open inputs sit at the input limit, which trains nothing."""
    import threading as th

    from nfb.sources import SampleFileSource
    from nfb_console import amplitude_view as av

    from conftest import plausible_eeg

    export = tmp_path / "eeg.txt"
    columns = [plausible_eeg(8.0, 250.0, c) for c in range(2)]
    export.write_text("Sampling rate: 250\n" + "".join(f"{a:.4f}\t{b:.4f}\n" for a, b in zip(*columns)))
    session = av.AmplitudeSession(BY_CODE["AMP"],
                                  source=SampleFileSource(export, realtime=False, loop=True),
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
        session._outputs = type(session._outputs)(session._apply_snapshot, session._release_outputs)
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
