"""Moving between the launcher, a screen and the device screen, and what a session keeps across it."""

from __future__ import annotations



from nfb import WaitingSource
from nfb_console.protocols import BY_CODE

from conftest import BENCH

from harness import (
    _console,
    _contact_rows,
    _live_console,
    _texts,
    RECORDING,
    replay,
    StubPage,
    toolbar,
)


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


def test_selecting_the_amplifier_does_not_reopen_the_recording():
    """`_pick` branched session creation on the startup `--live` flag, which never changes, not on the live
    input choice. So after starting in replay mode and selecting the amplifier, picking a protocol reopened
    the old recording instead of going live."""
    from nfb import WaitingSource

    console = _console()                            # started from a file
    try:
        assert console.source_label().startswith("file:")
        console.inputs.kind, console.inputs.path = "amplifier", ""
        console.pick(BY_CODE["AMP"])
        assert console.source_label() == "", "the amplifier announces no file"
        assert isinstance(console.session.source, WaitingSource), \
            "a live session waits on the device, not the recording"
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_reconfiguring_on_the_device_screen_rebuilds_the_session(monkeypatch):
    """`resume` rebuilt only the view, never the estimators. After changing the rate on the device screen the
    session fed a 1000 Hz stream into filters built for the old rate, so rebuild on a real settings mismatch."""
    console, _ = _live_console(monkeypatch)
    try:
        first = console.session
        assert first is not None and first.rate == 1000.0
        console.show_device(return_to=BY_CODE["AMP"])       # suspends, stops the device
        assert console.session is first and first.suspended
        console.device.reconfigure(hz=250.0)
        console.resume(BY_CODE["AMP"])
        assert console.session is not first, "a new rate must rebuild, not reuse 1000 Hz estimators"
        assert console.session.rate == 250.0
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_choosing_a_file_on_the_device_screen_rebuilds_the_session(monkeypatch):
    """`resume` came back to the session built on the amplifier, which read nothing of the file chosen on
    the device screen: the screen waited on a device that was no longer the input."""
    console, _ = _live_console(monkeypatch)
    try:
        first = console.session
        console.show_device(return_to=BY_CODE["AMP"])
        console.inputs.kind, console.inputs.path = "file", str(BENCH)
        console.resume(BY_CODE["AMP"])
        assert console.session is not first, "another input must rebuild"
        assert not isinstance(console.session.source, WaitingSource), "the session reads the file"
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_a_contact_check_with_no_reconfigure_returns_to_the_same_session(monkeypatch):
    """The rebuild is only on a real mismatch: looking at electrodes and coming back must keep the session,
    its clock and its calibration, exactly as leaving it does today."""
    console, _ = _live_console(monkeypatch)
    try:
        first = console.session
        console.show_device(return_to=BY_CODE["AMP"])
        console.resume(BY_CODE["AMP"])
        assert console.session is first, "no reconfigure, so the session carries on"
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_returning_from_the_device_screen_keeps_a_replay_session():
    """A replay session has no device to stop, so it keeps feeding (paused) through the device-screen visit
    and `resume` rebuilt it — resetting every target. It is retained now, like a live suspended one."""
    console = _console(protocol="AMP")          # file-backed, from the bench recording
    try:
        first = console.session
        console.show_device(return_to=BY_CODE["AMP"])
        console.resume(BY_CODE["AMP"])
        assert console.session is first, "a replay session is retained across the device screen, not rebuilt"
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_resuming_hrv_heg_or_pupil_does_not_crash():
    """`resume` reached for `session.device` — the EEG amplifier, which only the EEG screens hold. HRV, HEG
    and Pupil run their own hardware and are not TrainingSessions, so that attribute does not exist and the
    back-crumb handler crashed. `resume` now keys on `suspended`, which every screen answers. The camera
    round trip on the Pupil screen is this exact path."""
    for code in ("PUPIL", "HRV", "HEG"):
        console = _console(protocol=code)
        try:
            console.show_device(return_to=BY_CODE[code])
            console.resume(BY_CODE[code])       # the back-crumb path; must not raise
            assert console.session is not None and console.protocol.code == code
        finally:
            if console.session is not None:
                console.session.stop()
            console.device.close()


def test_a_missing_recording_does_not_block_the_hrv_screen():
    """Protocol selection opened the replay file before choosing the screen, so an absent recording crashed
    HRV, HEG and Pupil — which read their own inputs and never touch it."""
    from nfb_console.app import Console, build_parser

    args = build_parser().parse_args(["/does/not/exist.jsonl.gz"])
    console = Console(StubPage(), args)
    try:
        console.pick(BY_CODE["HRV"])
        assert console.session is not None, "HRV takes its own input; the missing EEG file must not block it"
    finally:
        if console.session is not None:
            console.session.stop()
        console.device.close()


def test_the_device_chip_from_the_launcher_still_goes_back_to_the_launcher():
    """The same control, and it has to remember two different origins."""

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


def test_a_bin_has_no_higher_line_and_no_second_mark():
    """Every other target in the app keeps exactly one threshold. The second line is one screen's."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert all(t.reach_rate is None for t in session.bin_targets)
        assert not any(t.reaches(1e6) for t in session.bin_targets)
    finally:
        session.stop()


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
