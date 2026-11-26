"""The CFC screen: its rewards and bins, their thresholds, and the heart beside the EEG."""

from __future__ import annotations

import time

import pytest

from conftest import counts as to_counts
from nfb_console.protocols import BY_CODE

from harness import (
    _armed_rewards,
    _chosen,
    _find_help,
    _heart_with,
    _mayer_readings,
    _numbers,
    _pills,
    _texts,
    at_corner,
    pump,
    replay,
    shown,
    StubDevice,
    StubPage,
    toolbar,
)


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


def test_each_cfc_reward_sits_on_a_boundary_between_two_bins():
    """The interlock: a reward frequency is the edge that two inhibit bins share, on every channel."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert all(t.role == "inhibit" and t.direction == "increase" for t in session.bin_targets)
        boundaries = {edge for t in session.bin_targets for edge in (t.band.low, t.band.high)}
        assert session.rewards
        for reward in session.rewards:
            assert reward.direction == "increase"
            assert reward.band.centre in boundaries
    finally:
        session.stop()


def test_cfc_thresholds_are_derived_from_his_total_not_from_our_defaults():
    """Every sound on the screen together, the twelve bins and the rewards, is aimed inside the stated
    5-15% of the session, so silence stays reachable."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        bins_over = sum(1 - t.hit_rate for t in session.bin_targets)
        rewards_under = sum(1 - t.hit_rate for t in session.rewards)
        assert bins_over > 0 and rewards_under > 0
        assert 0.05 <= bins_over + rewards_under <= 0.15, "all sounds together, inside the stated range"
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
    """The cascade draws the sum of the two channels: a rhythm on either channel appears in its rows."""
    import numpy as np

    from nfb.stream import Block
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    rows: list = []
    session.spectrogram.push = rows.append
    try:
        rate = session.rate
        t = np.arange(int(4 * rate)) / rate
        waves = {0: 20 * np.sin(2 * np.pi * 10.0 * t), 1: 20 * np.sin(2 * np.pi * 25.0 * t)}
        for i in range(0, len(t) - 7, 8):
            for ch, spec in session.specs.items():
                session.feed_block(Block(ch, i, tuple(to_counts(waves[ch][i:i + 8], spec.scale).tolist()), spec))
        session.push_displays(1e9)
        assert rows, "a row once the ring holds a spectrum"
        centres = session.spectrogram.centres()
        row = rows[-1]
        quiet = np.median(row)
        for hz in (10.0, 25.0):
            assert row[int(np.argmin(abs(centres - hz)))] > 5 * quiet, f"{hz:g} Hz is missing from the sum"
    finally:
        session.stop()


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


def test_a_subhertz_reward_has_its_own_width_and_the_card_says_what_it_listens_to():
    """A sub-hertz reward gets its own width, offered only while one exists, and every card states the band
    it measures. Where that width would reach 0 Hz it is bounded, and the note names the reward."""
    from nfb_console import cfc_view

    def band_text(tgt):
        return f"{tgt.band.low:g}–{tgt.band.high:g} Hz"

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        page = StubPage()
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        shown = [x.value for x in _texts(control)]
        assert all(band_text(x) in shown for x in session.rewards), "an ordinary reward states its band too"
        assert "subhertz width" not in shown, "no sub-hertz reward, so no sub-hertz width"

        session.retune_rewards(freqs=[cfc_view.WAVES["Mayer"], 10.0])
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        shown = [x.value for x in _texts(control)]
        slow, fast = session.rewards
        assert "subhertz width" in shown
        assert band_text(slow) in shown, "the slow card says what it listens to"
        assert slow.band.high - slow.band.low == pytest.approx(session.subhertz_width_hz), "its own width"
        assert fast.band.high - fast.band.low == pytest.approx(session.width_hz), "the fast one is untouched"
        assert not any("bounded" in x for x in shown), "a width that fits needs no note"

        # The bound still applies where the width cannot be granted, and the note names the reward.
        session.retune_rewards(freqs=[cfc_view.WAVES["ultra-low"], 10.0])
        control = cfc_view.view(page, session, toolbar(), lambda: None)
        shown = " ".join(x.value for x in _texts(control))
        slow = session.rewards[0]
        assert slow.band.low > 0, "a band that reaches 0 Hz measures the mean"
        assert slow.band.high - slow.band.low < session.subhertz_width_hz
        assert f"{slow.name} is bounded to" in shown
        assert band_text(slow) in shown

        # And the centre box can print the number it holds, not round it to zero.
        assert f"{cfc_view.WAVES['ultra-low']:g}" in _numbers(control)
    finally:
        session.stop()


def test_a_subhertz_threshold_gets_a_long_window_and_a_live_number_throughout_it():
    """A sub-hertz reward calibrates over minutes, capped, and its threshold is derived from the readings
    every second of that window rather than held at the placeholder."""
    from nfb_console import cfc_view
    from nfb_console.targets import CALIBRATION_SECONDS

    assert cfc_view.calibration_for(10.0, 0.2) == CALIBRATION_SECONDS, "an ordinary reward is unchanged"
    assert cfc_view.calibration_for(0.1, 0.05) > 4 * CALIBRATION_SECONDS, "a slow band waits for minutes"
    assert cfc_view.calibration_for(0.01, 0.001) == cfc_view.calibration_for(0.01, 0.0001), "and is capped"

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        session.retune_rewards(freqs=[cfc_view.WAVES["Mayer"], 10.0])
        slow, fast = session.rewards
        assert slow.calibration_seconds > fast.calibration_seconds == CALIBRATION_SECONDS
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


def test_cfc_thresholds_hold_still_because_time_above_them_is_what_is_trained():
    """The bins hold a fixed threshold and the rewards re-derive on a period, so time above threshold can
    change; the reward cards offer both modes."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        assert all(t.mode == "fixed" for t in session.bin_targets), "the bins are Manual"
        assert all(t.mode == "automatic" and t.recompute_seconds > 0 for t in session.rewards), (
            "the rewards re-derive on a period and hold between")
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


def test_changing_the_depth_moves_the_row_rate_and_clears_the_cascade():
    """A new depth changes the seconds between rows in proportion, and clears the rows drawn at the old
    interval, so one picture never holds two time scales."""
    import numpy as np

    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        sg = session.spectrogram
        assert sg.depth_seconds == cfc_view.DEPTHS[0], "it opens at the shallowest depth"
        before = sg.interval

        sg.push(np.full(40, 5.0))
        assert sg.filled
        sg.depth_seconds = 4 * cfc_view.DEPTHS[0]
        sg.clear()
        assert not sg.filled, "rows sampled at the old interval cannot stay in a picture with a new axis"
        assert sg.interval == pytest.approx(4 * before)
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


def test_cfc_sets_one_filter_for_all_twelve_bins():
    """Every bin's estimator has the one delay the session states, before and after a change, and a change
    rebuilds all of them."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        # hold the objects, not their ids: CPython reuses the address of a freed object, so comparing
        # id() after a rebuild is a coin toss
        before = dict(session.bins)
        opening = session.bin_delay
        assert all(m.delay_seconds == pytest.approx(opening, abs=0.02) for m in before.values())

        wanted = opening + 0.2
        session.retune_bins(wanted)
        assert session.bin_delay == pytest.approx(wanted)
        assert set(session.bins) == set(before), "same bins"
        assert all(m.delay_seconds == pytest.approx(wanted, abs=0.02) for m in session.bins.values())
        assert all(session.bins[k] is not v for k, v in before.items()), "all rebuilt, none stale"
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


def test_a_reward_carries_a_higher_line_derived_from_the_same_window():
    """Each settled reward has a second line above its threshold, from the same window, and only a
    reading past that line reaches it."""
    import numpy as np

    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        rng = np.random.default_rng(0)
        for tgt in session.rewards:
            for value in 10.0 + 3.0 * rng.standard_normal(int((tgt.settle_seconds + tgt.window) * 20) + 20):
                tgt.observe(float(value), 20.0)
            tgt.redefine()
            assert tgt.reach_threshold > tgt.threshold
            assert tgt.reaches(tgt.reach_threshold + 1.0)
            assert not tgt.reaches(tgt.threshold)
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


def test_each_bin_reads_its_own_channel():
    """A rhythm on one channel raises that channel's bin and not the same bin on the other channel."""
    import numpy as np

    from nfb.stream import Block
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        rate = session.rate
        t = np.arange(int(4 * rate)) / rate
        waves = {0: 30 * np.sin(2 * np.pi * 10.0 * t), 1: np.zeros(len(t))}
        for i in range(0, len(t) - 7, 8):
            for ch, spec in session.specs.items():
                session.feed_block(Block(ch, i, tuple(to_counts(waves[ch][i:i + 8], spec.scale).tolist()), spec))
        loud, quiet = ({tgt.band.low: session.value_for(tgt) for tgt in session.bin_targets if tgt.source == ch}
                       for ch in (0, 1))
        band = next(low for low in loud if low <= 10.0 < low + 4)
        assert loud[band] > 10 * max(quiet[band], 0.1), (loud[band], quiet[band])
    finally:
        session.stop()


def test_a_bin_louder_than_the_bin_below_it_is_named():
    """A bin is named as breaking the falling spectrum only when it reads above the bin below it."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        bins = [tgt for tgt in session.bin_targets if tgt.source == 0]
        for n, tgt in enumerate(bins):
            tgt.value, tgt.total = 100.0 - n, 100
        assert session.breaks_one_over_f() == [], "a falling spectrum breaks nothing"
        bins[3].value = bins[2].value + 5.0
        assert session.breaks_one_over_f() == [bins[3]]
    finally:
        session.stop()


def test_the_reward_plot_draws_the_rewards():
    """The coupling plot advances with each reward's own reading."""
    from nfb_console import cfc_view

    session = cfc_view.CFCSession(BY_CODE["CFC"], source=replay(), channels=2)
    try:
        for n, tgt in enumerate(session.rewards):
            tgt.value = 7.0 + n
        session.push_displays(1e9)
        series = session.reward_plot.series()
        assert [s[-1] for s in series] == [7.0 + n for n in range(len(session.rewards))]
    finally:
        session.stop()


def test_cfc_total_above_does_not_hide_an_always_hot_bin():
    """total_above treated a bin met 0% of the time (sustained above threshold — the worst) the same as a
    bin with no window yet, because `0.0 or 100.0` is 100.0. None still counts as fully met; 0.0 does not."""
    import types

    from nfb_console.cfc_view import CFCSession

    class _Bin:
        def __init__(self, pct):
            self._pct = pct

        def recent_percent(self):
            return self._pct

    stub = types.SimpleNamespace(bin_targets=[_Bin(0.0), _Bin(None), _Bin(100.0)])
    total = CFCSession.total_above.fget(stub)
    # met 0% → 100 above (the hot bin, now counted); no window → 0; met 100% → 0
    assert total == pytest.approx(100.0)
