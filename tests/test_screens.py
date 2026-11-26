"""Behaviour that two or more screens must share. One screen's own behaviour is in that screen's file."""

from __future__ import annotations


import pytest

from nfb_console.protocols import BY_CODE
from nfb_console.trials import ACTIVE

from harness import (
    _by_label,
    _find_gear,
    _find_help,
    _pills,
    _texts,
    _trace_rows,
    at_corner,
    pump,
    replay,
    OTHER_SCREENS,
    SCREENS,
    screen_params,
    shown,
    StubPage,
    toolbar,
)


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


def test_the_trend_goes_on_when_the_stream_starts_again_from_an_earlier_index():
    """A replay that loops, or a device that restarts, steps the stream time back; the trend keeps
    taking points from the new start."""
    from nfb.stream import Block
    from nfb_console import amplitude_view as av
    from nfb_console.training import TREND_SECONDS

    from conftest import counts, plausible_eeg

    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        session.paused = False
        session.state = {0: True}
        spec = session.specs[0]
        wave = plausible_eeg(20.0, session.rate, 0)

        def feed(seconds: float) -> None:
            for i in range(0, int(seconds * session.rate) - 249, 250):
                session.feed_block(Block(0, i, counts(wave[i:i + 250], spec.scale), spec))

        feed(20.0)
        before = len(session.targets[0].trend)
        feed(4 * TREND_SECONDS)
        assert len(session.targets[0].trend) > before, "the trend froze at the step back"
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


def test_the_arrows_on_either_band_edge_slide_the_whole_band():
    """On every screen whose cards have band edges, an arrow on either edge moves the band and keeps its
    width."""
    import flet as ft

    from nfb_console import amplitude_view, at_view
    from test_layout import walk

    def boxes(control):
        """Each number box as (its text field, its up arrow), the smallest control that holds both."""
        found = {}
        for c in walk(control):
            fields = [x for x in walk(c) if isinstance(x, ft.TextField)]
            ups = [x for x in walk(c) if isinstance(x, ft.Container) and isinstance(x.content, ft.Text)
                   and x.content.value == "\u25b2"]
            if len(fields) == 1 and len(ups) == 1:
                found[id(fields[0])] = (fields[0], ups[0])
        return list(found.values())

    for module, code in ((amplitude_view, "AMP"), (at_view, "A-T")):
        session = module.session_for(BY_CODE[code], source=replay(), channels=1)
        try:
            control = module.view(StubPage(), session, toolbar(), lambda: None)
            tgt = session.targets[0]
            for edge in ("low", "high"):
                before = (tgt.band.low, tgt.band.high)
                shown = getattr(tgt.band, edge)
                box = next((up for field, up in boxes(control)
                            if field.value and float(field.value) == pytest.approx(shown)), None)
                assert box is not None, f"{code}: no box shows the {edge} edge"
                box.on_click(None)
                after = (tgt.band.low, tgt.band.high)
                assert after[0] > before[0] and after[1] > before[1], f"{code}: the {edge} arrow moved one edge"
                assert after[1] - after[0] == pytest.approx(before[1] - before[0]), f"{code}: width kept"
                control = module.view(StubPage(), session, toolbar(), lambda: None)
        finally:
            session.stop()


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
            if code in ("A-T", "CFC"):
                assert sources[0] == "sum", f"{code}: the sum is what the bands are measured on"
            else:
                assert sources[-1] == "sum", f"{code}: a target sits on a channel, so the channels lead"
            assert [s for s in sources if s.startswith("ch")], code
            pump(session)
            session.redraw()
            assert session.error is None, (code, session.error)
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


@pytest.mark.parametrize("module,code,channels", screen_params(SCREENS + OTHER_SCREENS))
def test_pressing_a_tab_shows_that_pane_and_only_that_pane(module, code, channels):
    """On every screen, each tab label shows its own pane and hides every other one."""
    import importlib

    import flet as ft

    from harness import _find_pills

    screen = importlib.import_module(f"nfb_console.{module}")
    session = screen.session_for(BY_CODE[code], source=replay(), channels=channels)
    try:
        control = screen.view(StubPage(), session, toolbar(), lambda: None)
        root = control.controls[0] if isinstance(control, ft.Stack) else control
        labels = [x.value for x in _texts(root.controls[1])]
        panes = root.controls[2].controls
        assert len(labels) == len(panes), code
        for index in reversed(range(len(labels))):
            _find_pills(root.controls[1], labels[index]).on_click(None)
            assert [p.visible is not False for p in panes] == [i == index for i in range(len(panes))], \
                f"{code}: {labels[index]}"
    finally:
        session.stop()


@pytest.mark.parametrize("module,code,channels", screen_params(SCREENS + OTHER_SCREENS))
def test_pressing_a_choice_lights_that_choice_and_only_that_one(module, code, channels):
    """On every screen, pressing a choice that is not lit lights it, and in a row with one lit choice it is then
    the only one lit. **fit** is the exception that proves the rule: it picks another choice of its row, which
    is then the one lit."""
    import importlib

    import flet as ft

    from nfb_console import theme

    picks_another = {"fit"}

    def is_pill(node):
        """A choice as `widgets.pill` draws it. An action such as **redefine** has the same fill, in the
        accent colour."""
        return isinstance(node, ft.Container) and isinstance(node.content, ft.Text) \
            and node.border_radius == 6 and node.content.color in (theme.TEXT, theme.MUTED)

    def rows(node, path=()):
        """Each row of two or more pills, by its place in the tree: a press rebuilds a row, and a row can
        gain a pill, so neither the object nor its labels identify it afterwards."""
        if node is None or len(path) > 30:
            return
        children = getattr(node, "controls", None) or []
        pills = [c for c in children if is_pill(c)]
        if len(pills) >= 2:
            yield path, pills
        for i, child in enumerate(children):
            yield from rows(child, path + (i,))
        if not is_pill(node):
            yield from rows(getattr(node, "content", None), path + ("content",))

    def lit(row):
        return [p.content.value for p in row if p.bgcolor == theme.SURFACE_HI]

    pressed = 0
    screen = importlib.import_module(f"nfb_console.{module}")
    session = screen.session_for(BY_CODE[code], source=replay(), channels=channels)
    try:
        control = screen.view(StubPage(), session, toolbar(), lambda: None)
        choices = [(path, [p.content.value for p in row]) for path, row in rows(control)]
        # A row of one lit choice as the screen opens is a row of exclusive choices; a row of switches, or
        # of channels to add, can light none or several.
        exclusive = {path for path, row in rows(control) if len(lit(row)) == 1}
        for path, names in choices:
            for label in names:
                row = dict(rows(control)).get(path)
                pill = next((p for p in row or [] if p.content.value == label), None)
                if pill is None or pill.on_click is None or label in lit(row):
                    continue
                pill.on_click(None)
                pressed += 1
                after = dict(rows(control)).get(path)
                if after is None or label not in [p.content.value for p in after]:
                    continue
                if label in picks_another:
                    assert len(lit(after)) == 1, f"{code}: {label} in {names} lit {lit(after)}"
                elif path in exclusive:
                    assert lit(after) == [label], f"{code}: pressed {label} in {names}, lit {lit(after)}"
                else:
                    assert label in lit(after), f"{code}: pressed {label} in {names}, lit {lit(after)}"
    finally:
        session.stop()
    assert pressed, "the walk found no row of choices to press"


def test_the_level_line_names_the_full_at_of_the_trial_in_flight():
    from test_heg_trials import _push
    from test_heg_trials import _session as heg_session
    from test_pupil_view import _eye, _to_feedback
    from test_pupil_view import _session as pupil_session

    pupil = pupil_session("strong", trial_direction="increase", n_trials=3)
    try:
        _to_feedback(pupil, _eye(pupil))
        in_use = pupil.scale_in_use()[0]
        pupil.set_full_at(0.5, typed=True)
        assert f"±{pupil.scale_in_use()[0]:g} mm" in pupil.level_source[1], \
            (pupil.level_source[1], in_use)
    finally:
        pupil.stop()
    heg = heg_session(n_trials=2)
    try:
        t = 0.0
        while heg.trials.phase != ACTIVE:
            t = _push(heg, 1.0, t)
            assert t < 300.0, "no feedback window came"
        heg.set_full_at(5.0, typed=True)
        assert f"±{heg.scale_in_use()[0]:g} sd" in heg.level_source[1], heg.level_source[1]
    finally:
        heg.stop()


def test_pupil_and_heg_help_explains_the_words_of_the_run_verdict():
    from nfb_console import heg_view, pupil_view

    for name, text in (("Pupil", pupil_view.MODE_HELP), ("HEG", heg_view.MODE_HELP)):
        assert "chance range" in text, f"the {name} help does not explain the chance range"
