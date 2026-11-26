"""The alpha-theta screen: its bands, objectives, crossfade and thresholds."""

from __future__ import annotations

import math

import pytest

from nfb_console.feedback import LEVEL as LEVEL_CH
from nfb_console.protocols import BY_CODE

from conftest import counts as to_counts

from harness import _by_label, _numbers, _pills, _texts, pump, replay, StubPage, toolbar


def test_the_alpha_theta_convention_pill_carries_the_threshold_and_restarts():
    """Alpha-theta carries the threshold like the amplitude screen, but restarts its statistics rather than
    rescaling them, because the difference window is too short to hold a converted history."""
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        tgt = next(t for t in session.targets
                   if t is not session.difference and session.filter_for(t)[0] != "cfir")
        for _ in range(int(35 * 20)):
            tgt.observe(10.0, 20.0)
        tgt.redefine()
        threshold = tgt.threshold
        assert tgt.total > 0

        session.set_convention("meanabs")
        ratio = (2.0 / math.pi) / (1.0 / math.sqrt(2.0))
        assert tgt.threshold == pytest.approx(threshold * ratio, rel=1e-6), "the threshold moved to the unit"
        assert tgt.total == 0, "the collected readings restart"
    finally:
        session.stop()


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


def test_the_band_variants_are_derived_from_the_bands_and_refuse_an_unsatisfiable_pair():
    """The variant is read from the bands, the sleep guard and the wide theta band refuse each other from
    both sides, and removing the guard removes its target and its estimator."""
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        assert session.variant == ("wide", False, 22.0)

        session.set_theta_width("narrow")
        session.set_guard(True)
        session.set_high_low(15.0)
        assert session.variant == ("narrow", True, 15.0)
        assert session.guard.floor is not None and session.guard.floor == session.inhibit.floor, \
            "both inhibits carry the floor"
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


def test_the_sleep_guard_has_three_responses_and_silence_is_the_ambiguous_one():
    """The guard fires only on data past its threshold; silence plays nothing, the knock sounds a tone keyed
    to the guard's band, and only the knock with hush holds the music."""
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    played: list = []
    session.cue = lambda name, vol=None: played.append(name)
    try:
        session.set_theta_width("narrow")
        session.set_guard(True)
        session.set_guard_response("silence")

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
    """The reward tone is keyed to the band that earned it, theta first unless unbiased, and the two bands'
    tones are more than two octaves apart."""
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
        assert high / low > 4, "under two octaves apart is what the alert's noise bursts gave"
        assert reward_tone(session.theta.band.centre) != reward_tone(session.alpha.band.centre)
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


def test_the_at_music_fades_between_two_characters_of_noise():
    """The music fades between two noise colours by the theta share, averaged over a window, and nothing in
    it grows louder with a score."""
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        music = session.feedback.music
        assert music.enabled and music.source == "crossfade"
        assert music.colour != music.second_colour, "two characters, one at each end"
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

        hold(15.0, 5.0)
        assert session.state_balance == pytest.approx(0.75), "the theta share of the two"
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


def test_the_at_screen_can_audition_both_tilts_and_both_tones_from_the_feedback_tab():
    """Four sounds nobody has heard, in a protocol run with the eyes closed — so without auditions the
    first time either is heard is the one moment nothing can be adjusted.

    They live **with the mechanism that plays them**, not on Train. Train had all four plus the choice of
    which band the tone marks, which is sound configuration in the wrong place and redundant besides: the
    Music pane previews each tilt and the Tone pane auditions each reward at its own pitch.
    """
    from nfb_console import at_view, signal_pane
    from nfb_console.cues import tone_pitch

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
        derived = [f"{tone_pitch(x.band.centre):g}" for x in (session.theta, session.alpha)]
        assert all(x in pitches for x in derived), f"both tones at their own pitches, got {pitches}"
        assert tones.count("▶") == 3, "each tone, plus the held mark, beside the number that sets it"
        assert "theta first" in tones, "and which band the tone marks, beside the tone"

        # Typing pins it; typing the derived value back leaves it derived, so the box is not a one-way door.
        pane._set_tone_hz(session.theta, 300)
        assert session.theta.tone_hz == 300.0
        assert "300" in _numbers(pane.click_rows)
        pane._set_tone_hz(session.theta, float(derived[0]))
        assert session.theta.tone_hz is None, "back to derived"

        control = at_view.view(StubPage(), session, toolbar(), lambda: None)
        train = " ".join(x.value for x in _texts(control.controls[-1].controls[0]))
        assert "hear it" not in train, "no auditions on Train"
        assert "theta first" not in train, "and no sound configuration either"
        assert "toward theta" in train or "waiting for data" in train, "the music readout stays"
    finally:
        session.stop()


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


def test_switching_the_sleep_guard_on_again_keeps_it():
    """Pressing the guard on while it is already on leaves it on."""
    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        session.set_theta_width("narrow")
        session.set_guard(True)
        guard = session.guard
        session.set_guard(True)
        assert session.guard is guard and guard in session.targets
    finally:
        session.stop()


def test_moving_the_high_inhibit_re_derives_its_threshold_for_the_new_aim():
    """A new aim on the high inhibit re-derives its threshold from the readings already collected."""
    import numpy as np

    from nfb_console import at_view

    session = at_view.ATSession(BY_CODE["A-T"], source=replay(), channels=1)
    try:
        tgt = session.inhibit
        for value in 10.0 + 3.0 * np.random.default_rng(0).standard_normal(int(40 * 20)):
            tgt.observe(float(value), 20.0)
        tgt.redefine()
        before = tgt.threshold
        session.set_high_low(15.0)
        assert tgt.threshold != pytest.approx(before), "not the old threshold under the new aim"
    finally:
        session.stop()
