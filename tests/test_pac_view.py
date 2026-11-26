"""The coupling screen: amplitude coupling, phase locking, and the cards that point them."""

from __future__ import annotations

import re

import flet as ft
import pytest

from nfb_console.protocols import BY_CODE

from harness import (
    _find_menu_items,
    _find_pills,
    _menu_clicks,
    _pills,
    _run_coupling,
    _run_locking,
    _texts,
    pump,
    replay,
    StubPage,
    toolbar,
)


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
    """A coupling reward has no band, so its card has no edges; its muscle guard is a banded inhibit.
    Both re-derive on a period from a window as long as the opening calibration."""
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
        assert reward.mode == "automatic" and reward.recompute_seconds > 0
        assert reward.window == reward.calibration_seconds, (
            "re-derived from as much signal as the opening calibration used")
        assert guard.mode == "automatic" and guard.recompute_seconds > 0
    finally:
        session.stop()


def test_the_coupling_screen_refuses_a_coupling_it_cannot_measure():
    """A coupling the session cannot measure is refused: adding it returns nothing and adds no target."""
    from nfb.dsp.filters import Band
    from nfb.dsp import Coupling
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=1)
    try:
        before = len(session.targets)
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


def test_each_coupling_card_shows_its_mode_and_the_help_states_its_period():
    """Each trained card carries its own mode pills, and the help states the window and the period the
    cards use."""
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
        coupling = next(tg for tg in trained if tg.band is None)
        assert f"every {coupling.recompute_seconds:g} seconds" in pac_view.TRAINING_HELP
        assert f"last {coupling.calibration_seconds:g} seconds" in pac_view.TRAINING_HELP
    finally:
        session.stop()


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


def test_a_pac_instrument_keeps_its_settings_across_a_channel_change():
    """`adopt_targets` rebuilds the couplings, so without carrying the per-target settings a reward whose
    aim was typed by hand, or whose threshold was fixed, went back to the automatic default the moment the
    channel count changed. The readings do not come across — the estimator is new — but the choices do."""
    from nfb_console import pac_view

    old = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        tgt = next(t for t in old.targets if old.key_for(t) in old.couplings and not t.monitor)
        tgt.set_aim(0.2)
        tgt.mode, tgt.threshold, tgt._settled = "fixed", 0.42, True

        new = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
        try:
            new.adopt_targets(old)
            moved = next(t for t in new.targets if new.key_for(t) == old.key_for(tgt))
            assert moved.aim_by_hand and moved.hit_rate_override == pytest.approx(0.2)
            assert moved.mode == "fixed" and moved.threshold == pytest.approx(0.42) and moved._settled
        finally:
            new.stop()
    finally:
        old.stop()


def test_setting_a_pac_coupling_to_monitor_keeps_a_guard_a_locking_still_needs():
    """`set_role` used to pop the guard for every channel the coupling read, then rebuild the orphans from
    what was left — so a channel a locking still guards lost its guard, because the pop had already removed
    it from the set the rebuild scans. Now the pop is gone and `_drop_orphan_guards` alone decides."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        coupling = next(t for t in session.targets
                        if session.key_for(t) in session.couplings and t.source == 0)
        assert not coupling.monitor
        assert session.guard_kinds.get(0) == "fast", "an earning coupling takes the strict guard"

        session.set_role(coupling, "monitor")
        assert any(t.band == pac_view.GUARD and t.source == 0 for t in session.targets), \
            "the locking on ch0 still needs ch0 guarded"
        assert session.guard_kinds.get(0) == "held", "and the gentler kind, now nothing earns on it"
    finally:
        session.stop()


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
    """The screen offers a spectrum per channel and of the sum, and its lowest trained frequency is the slow
    band's floor, not the muscle guard's."""
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    try:
        pump(session)
        assert [s.label for s in session.spectrum_sources()] == ["ch0", "ch1", "sum"]
        assert session.rings[0].psd() is not None
        assert session.trained_floor_hz == pac_view.SLOW.low, (
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
    """A card counts down no more than its own calibration window, and its help states that window."""
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
        for note in notes:
            (left,) = re.findall(r"(\d+) s", note)
            assert int(left) <= max(windows), f"more time left than any window: {note}"
            assert "first" not in note, "the explanation belongs behind the ?, not on every card"
        coupling = next(tg for tg in session.targets if tg.band is None)
        from nfb_console.target_card import TargetCard
        card = TargetCard(coupling, ft.Text(""), lambda: None, page=page)
        assert f"{coupling.calibration_seconds:g} s" in card._help_text()
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
        assert not session.guarded(coupling), "the switch reads off once it is off"
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


def test_a_monitor_card_is_a_reading_with_nothing_to_press():
    """A monitor cannot pay out, so a threshold, its nudges, a percentage met and a redefine button on its
    card are controls whose behaviour differs from their label. What is left is the identity row, the
    number, and a bar with a scale and no notch — and the number is never green, because a monitor is
    not wanted or unwanted."""
    from nfb_console import pac_view
    from nfb_console import theme
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
        assert card.notch.visible is False and card.value_text.color != theme.GOOD
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


def test_the_rhythm_line_says_what_its_number_is():
    """The rhythm line names its quantity and the ratio it is judged against, on one line that fits."""
    from nfb_console import pac_view

    session, _, control = _run_coupling(0.0, seconds=40.0)
    try:
        texts = [c.value for c in _texts(control) if c.value]
        line = next(v for v in texts if "is strongest while" in v)
        rhythm = next(v for v in texts if "the bands either side" in v)
        assert "amplitude" in rhythm and f"a peak is above {pac_view.RHYTHM_RATIO:g}" in rhythm, rhythm
        assert max(len(line), len(rhythm)) <= 110
    finally:
        session.stop()
