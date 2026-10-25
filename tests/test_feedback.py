"""The feedback signal: one 0…1 level, from every target at once.

The property worth protecting by test is that no target is ignored. It is easy to write a UI that
displays four targets and drives sound from one of them, and the mistake is invisible — the screen
looks right and the sound is wrong.
"""

from __future__ import annotations

import numpy as np
import math

import pytest

from nfb.dsp import SMR, THETA
from nfb_console.feedback import (CLICK_VOLUME, MUSIC_VOLUME, SHADOW_MECHANISMS, SHADOW_RANGE,
                                  SOUND_RANGE, Click, FeedbackSettings, Ramp, Shadow, Sound, combine)
from nfb_console.targets import Target


def settled(role: str, threshold: float, sd: float, seed: int = 0,
            direction: str = "increase") -> Target:
    """A target with enough history to have a spread, sitting at its threshold.

    The threshold is **stated after the history is fed**, and it has to be: a fixed target now derives a
    provisional threshold from as little as a second of data, so a number passed to the constructor is
    overwritten by the time 300 values have arrived. `nudge(1.0)` is the operator's own path to "this
    number, and stop deriving it", which is exactly what this fixture wants — a target whose threshold is
    the one the test names, with a real spread behind it.
    """
    tgt = Target(0, SMR if role == "reward" else THETA, direction, role=role,
                 mode="fixed", threshold=threshold)
    rng = np.random.default_rng(seed)
    for _ in range(300):
        tgt.observe(threshold + rng.normal(0, sd), 20.0)
    tgt.threshold = threshold
    tgt.nudge(1.0)
    tgt.value = threshold
    return tgt


def test_score_is_a_half_at_the_threshold_and_ordered_around_it():
    up = settled("reward", 10.0, 2.0)
    assert up.score() == pytest.approx(0.5, abs=1e-6)
    up.value = 16.0
    high = up.score()
    up.value = 4.0
    assert up.score() < 0.5 < high

    down = settled("inhibit", 20.0, 4.0)
    assert down.score() == pytest.approx(0.5, abs=1e-6)
    down.value = 8.0                      # low is good for an inhibit
    assert down.score() > 0.5
    down.value = 32.0
    assert down.score() < 0.5


def test_every_target_moves_the_level():
    """Change any one target and the signal must move. This is the anti-'driver' test."""
    targets = [settled("reward", 10.0, 2.0, 1), settled("inhibit", 20.0, 4.0, 2),
               settled("reward", 10.0, 2.0, 3)]
    for mode in ("mean", "weakest", "scaled"):
        base = combine(targets, mode)
        for tgt in targets:
            keep = tgt.value
            # "Worse" is away from the wanted side, and which side that is comes from `wants_above` —
            # the one place the role and the direction are resolved into a side of the line.
            tgt.value = keep * (0.5 if tgt.wants_above else 1.5)
            assert combine(targets, mode) < base, f"{mode} ignored a target"
            tgt.value = keep


def test_scaled_lets_an_inhibit_hold_back_a_good_reward():
    up, down = settled("reward", 10.0, 2.0, 4), settled("inhibit", 20.0, 4.0, 5)
    up.value, down.value = 16.0, 32.0          # reward high, artifact firing
    assert combine([up, down], "scaled") < 0.2
    assert combine([up, down], "weakest") < 0.2
    assert combine([up, down], "mean") > 0.4   # a blend masks it, by construction


def test_scaled_does_not_collapse_with_several_inhibits():
    """Three inhibits at their thresholds must read the same as one, or adding an inhibit would
    silently turn the feedback down."""
    up = settled("reward", 10.0, 2.0, 6)
    one = combine([up, settled("inhibit", 20.0, 4.0, 7)], "scaled")
    three = combine([up] + [settled("inhibit", 20.0, 4.0, 7 + i) for i in range(3)], "scaled")
    assert three == pytest.approx(one, abs=1e-6)


def test_a_continuous_mechanism_has_no_off_state():
    """A shadow that vanished when the targets were not met would be a flash wearing a shadow's name.
    Continuous means present; only the magnitude moves."""
    fb = FeedbackSettings()
    fb.sound.enabled = fb.shadow.enabled = True
    up, down = settled("reward", 10.0, 2.0, 8), settled("inhibit", 20.0, 4.0, 9)
    up.value, down.value = 2.0, 40.0            # about as bad as it gets
    assert not fb.on_target([up, down])
    got = fb.outputs([up, down])
    assert got["shadow"] == pytest.approx(fb.shadow.low, abs=2) and fb.shadow.low > 0
    assert got["sound"] == pytest.approx(fb.sound.low, abs=2) and fb.sound.low > 0
    assert not hasattr(fb.shadow, "presence"), "no setting may turn a continuous mechanism off"


def test_the_click_is_a_fixed_interval_gated_on_every_target():
    """It carries one bit — earned or not — so its rate must not vary. A rate that moved would be a
    continuous mechanism in disguise, and a slow rate would be indistinguishable from a stopped one
    until you had waited long enough to be sure."""
    click = Click(enabled=True)
    assert click.interval_seconds == 2.0
    assert click.output(0.99, on_target=True) == 2.0
    assert click.output(0.01, on_target=True) == 2.0, "the level must not change the interval"
    assert click.output(0.99, on_target=False) == 0.0, "and not-earned is silence"
    assert click.rate_hz == 0.5

    click.interval_seconds = 5.0
    assert click.output(0.5, on_target=True) == 5.0
    assert click.rate_hz == 0.2


def test_the_click_asks_the_targets_and_not_the_combining_rule():
    """A blend can sit high with one target failing, which is right for a display and wrong for an
    event that means 'earned'. So the click is gated on every target, whatever the rule."""
    up, down = settled("reward", 10.0, 2.0, 20), settled("inhibit", 20.0, 4.0, 21)
    up.value, down.value = 22.0, 32.0           # reward well met, inhibit firing → not on target
    assert not FeedbackSettings().on_target([up, down])
    for rule in ("weighted", "weakest", "scaled"):
        fb = FeedbackSettings(combine=rule)
        fb.click.enabled = fb.shadow.enabled = True
        got = fb.outputs([up, down])
        assert got["click"] == 0, f"{rule}: a failing target has to stop the clicks"
        assert got["shadow"] > 0, f"{rule}: the continuous mechanism keeps informing"

    down.value = 8.0                             # everything met
    for rule in ("weighted", "weakest", "scaled"):
        fb = FeedbackSettings(combine=rule)
        fb.click.enabled = True
        assert fb.outputs([up, down])["click"] == fb.click.interval_seconds


def test_each_mechanism_names_the_channel_it_reads():
    """The two channels have to be legible, or the level's rule looks as though it governs everything.

    This is the model half of it: `reads` is derived from the kind, so a mechanism cannot be enabled on
    one channel and described as being on the other, and `followers` is what the fork under the SIGNAL
    panel prints. The music appears under the condition when it is gated and nowhere when it is not — its
    loudness has a fixed gain and reaches the level only through the master volume.
    """
    from nfb_console.feedback import CONDITION, LEVEL, READS

    fb = FeedbackSettings()
    assert fb.sound.reads == LEVEL and fb.shadow.reads == LEVEL
    assert fb.click.reads == CONDITION and fb.alert.reads == CONDITION
    # **Not "the condition".** That names a variable, and the chip is read by somebody who has not read the
    # code and has nothing else on the panel defining it. It says what the thing is instead.
    assert READS[fb.click.reads] == "fires when every target is met"
    assert "condition" not in READS[fb.click.reads]

    fb.sound.enabled = fb.click.enabled = True
    fb.shadow.enabled = fb.alert.enabled = False
    assert fb.followers(LEVEL) == ["sound"]
    assert fb.followers(CONDITION) == ["click"]

    fb.music.enabled = True
    assert "the music stops" not in fb.followers(CONDITION), "not gated, so it follows neither"
    fb.music.stops_when = "met"
    assert fb.followers(CONDITION) == ["click", "the music stops"]
    assert "the music stops" not in fb.followers(LEVEL), "its loudness is a fixed gain, not the level"


def test_a_targets_ramp_has_one_fixed_shape_and_the_gain_lives_at_the_end_of_the_chain():
    """**The per-target ramp is a shape, not a dial.** Its steepness was an operator setting for a while,
    and it could not do the job it was reached for: steepening each score around its own threshold pushes
    an already off-centre level towards an end rather than widening it. What an operator actually wants —
    how much of an output's travel an ordinary swing uses — is a gain on the *finished* level, and there is
    exactly one of those, `FeedbackSettings.sensitivity`.

    So this pins the shape. A plain crossing of a threshold would need no ramp, so the soft ramp is
    a decision of this project and its steepness has to be justified here.
    """
    from nfb_console.feedback import SPAN_SIGMAS

    assert FeedbackSettings().sensitivity == 1.0
    up = settled("reward", 10.0, 2.0, 40)
    sigma = up.spread
    assert sigma > 0

    up.value = 10.0 + sigma
    assert up.score("sd") == pytest.approx(0.731, abs=0.005), "one step out is 73%"
    up.value = 10.0 + 2 * sigma
    assert up.score("sd") == pytest.approx(0.881, abs=0.005)
    up.value = 10.0 + SPAN_SIGMAS * sigma
    assert up.score("sd") == pytest.approx(0.95, abs=0.005), "SPAN_SIGMAS is where it reads 95%"

    # The factor still exists on `score` itself, because a caller measuring one target's ramp needs to be
    # able to ask; nothing wires a session setting to it any more.
    up.value = 10.0 + sigma
    assert up.score("sd", 2.0) < up.score("sd", 1.0) < up.score("sd", 0.5)
    assert up.score("sd", 2.0) == pytest.approx(0.622, abs=0.005)
    # and it never reaches the ends, which is deliberate: a pinned output has stopped informing
    up.value = 10.0 + 50 * sigma
    assert up.score("sd") < 1.0


def test_the_conditions_definition_changes_only_under_either():
    """`either` is the one rule that reaches a discrete mechanism, so it is the one place the condition
    is not the plain AND — and the screen has to say which of the two is in force."""
    assert FeedbackSettings().describe_condition() == "every target on its wanted side"
    assert FeedbackSettings(combine="either").describe_condition() == (
        "one reward met, and every inhibit met")


def test_rewards_and_inhibits_mix_at_a_stated_ratio():
    """80/20 by default. A plain average would give each target a weight that depends on how many of
    each kind happen to exist, so adding a second inhibit would quietly double the inhibits' say."""
    class Fake:
        def __init__(self, role, value):
            self.role, self._v = role, value

        def score(self, scale_from="sd", factor=1.0):
            return self._v

    best_reward, worst_inhibit = Fake("reward", 1.0), Fake("inhibit", 0.0)
    assert combine([best_reward, worst_inhibit], "weighted") == pytest.approx(0.8)
    assert combine([Fake("reward", 0.0), Fake("inhibit", 1.0)],
                   "weighted") == pytest.approx(0.2)

    with_two = combine([best_reward, worst_inhibit, Fake("inhibit", 0.0)], "weighted")
    assert with_two == pytest.approx(0.8), "the ratio must not shift with the number of inhibits"

    assert combine([best_reward], "weighted") == pytest.approx(1.0), \
        "rewards alone are the whole signal, not capped at their share"
    assert combine([worst_inhibit], "weighted") == pytest.approx(0.0)

    half = combine([best_reward, worst_inhibit], "weighted", reward_share=0.5)
    assert half == pytest.approx(0.5), "an even split is the plain average of the two groups"


def test_the_ratio_is_reported_where_it_is_chosen():
    fb = FeedbackSettings(combine="weighted", reward_share=0.7)
    assert "70%" in fb.describe_combine() and "30%" in fb.describe_combine()
    fb.combine = "weakest"
    assert "worst" in fb.describe_combine(), "the ratio is meaningless for the other rules"


def test_each_continuous_mechanism_stays_inside_its_own_stated_range():
    fb = FeedbackSettings()
    fb.sound.enabled = fb.shadow.enabled = True
    up = settled("reward", 10.0, 2.0, 10)
    for value in (0.0, 5.0, 10.0, 1e6):
        up.value = value
        got = fb.outputs([up])
        assert SOUND_RANGE[0] <= got["sound"] <= SOUND_RANGE[1]
        assert SHADOW_RANGE[0] <= got["shadow"] <= SHADOW_RANGE[1]


def test_the_kind_says_what_the_numbers_on_a_pane_mean():
    """Sound's 60-100 and a click's 2 look alike and are not: loudness, against seconds between
    events. The kind is what the screen has to say."""
    fb = FeedbackSettings()
    assert fb.sound.kind == "continuous" and fb.sound.quantity == "system volume"
    assert fb.shadow.kind == "continuous" and fb.shadow.quantity == "brightness"
    assert fb.click.kind == "discrete" and fb.click.quantity == "interval"
    assert "kind" not in FeedbackSettings.__dataclass_fields__, "kind is a class fact, not a setting"
    assert fb.click.unit == "s" and fb.shadow.unit == "%"


def test_only_enabled_mechanisms_report_anything():
    fb = FeedbackSettings()
    up = settled("reward", 10.0, 2.0, 11)
    # **Two, not three: the shadow starts off.** It is the only mechanism whose default is a safety
    # question — see `Shadow.enabled` — because it takes over the operator's display, and it was on by
    # default behind a level that flickers, which is what `shadow.MAX_SLEW` now bounds.
    assert set(fb.outputs([up])) == {"sound", "click"}, "the two that start on"
    for mech in fb.mechanisms.values():
        mech.enabled = False
    assert fb.outputs([up]) == {}, "nothing enabled means nothing happening"
    fb.click.enabled = True
    assert set(fb.outputs([up])) == {"click"}
    assert fb.outputs([]) == {}, "nothing being trained is silence, not every floor value"


def test_the_signal_itself_is_never_gated():
    """Gating belongs to a mechanism, since one may be gated while another runs continuously. If
    level() applied it, a gated click would silence the shadow too."""
    fb = FeedbackSettings()
    up = settled("reward", 10.0, 2.0, 13)
    up.value = 2.0
    assert fb.level([up]) > 0.0
    assert not fb.on_target([up])


def test_a_range_entered_backwards_is_not_a_trap():
    assert Ramp(90, 40).span == (40, 90)
    assert Ramp(90, 40).at(0.0) == 40
    assert Ramp(90, 40).at(1.0) == 90
    assert Sound(low=90, high=40).output(0.0) == 40


def test_a_level_outside_zero_to_one_is_clamped_not_extrapolated():
    ramp = Ramp(20, 80)
    assert ramp.at(-1.0) == 20
    assert ramp.at(2.0) == 80


def test_the_shadow_names_the_mechanism_it_cannot_do_yet():
    shadow = Shadow()
    assert shadow.mechanism == "brightness" and shadow.supported
    shadow.mechanism = "overlay"
    assert not shadow.supported, "an unimplemented mechanism must not claim to work"
    assert "overlay" in SHADOW_MECHANISMS


def test_describe_says_which_mechanisms_are_on():
    fb = FeedbackSettings()
    assert "sound" in fb.describe() and "click" in fb.describe()
    for mech in fb.mechanisms.values():
        mech.enabled = False
    assert fb.describe() == "no output enabled"
    fb.sound.enabled = fb.click.enabled = True
    assert "sound" in fb.describe() and "click" in fb.describe()


def test_not_clicking_reads_as_a_state_not_a_number():
    """"0 s" is an interval; "not clicking" is what is actually happening."""
    fb = FeedbackSettings()
    fb.sound.enabled = fb.click.enabled = True
    up = settled("reward", 10.0, 2.0, 14)
    up.value = 2.0
    text = fb.describe_outputs([up])
    assert "not clicking" in text
    assert "volume " in text, "the tone keeps informing while the clicks are stopped"
    up.value = 16.0
    assert "a click every 2 s" in fb.describe_outputs([up])

def test_scale_from_actually_changes_the_score():
    """It was a setting no code read. If it does nothing, it must not be on the screen."""
    tgt = settled("reward", 10.0, 2.0, 30)
    tgt.value = 11.0                               # 1 µV above threshold
    by_sd = tgt.score("sd")                        # sd ≈ 2, so half a step
    by_threshold = tgt.score("threshold")          # 2.5 µV step, so a smaller move
    by_absolute = tgt.score("absolute")            # 5 µV step, smaller still
    assert by_sd > by_threshold > by_absolute > 0.5, (by_sd, by_threshold, by_absolute)

    fb = FeedbackSettings()
    fb.normalisation = "sd"
    loose = fb.level([tgt])
    fb.normalisation = "absolute"
    assert fb.level([tgt]) < loose, "the signal has to follow the scale, not just the label"


def test_the_scale_is_the_step_it_advertises():
    tgt = settled("reward", 10.0, 2.0, 31)
    assert tgt.scale_for("absolute") == 5.0
    assert tgt.scale_for("threshold") == pytest.approx(2.5)
    assert tgt.scale_for("sd") == pytest.approx(tgt.spread)


def test_each_combining_rule_has_its_own_wording_everywhere_it_is_described():
    """The share caption used to be written for `weakest` and shown for all three, so two thirds of
    the time it described a calculation that was not happening."""
    from nfb_console.feedback import COMBINE_DETAIL, COMBINE_SHARES, COMBINES
    assert set(COMBINES) == set(COMBINE_DETAIL) == set(COMBINE_SHARES)
    assert len({*COMBINE_SHARES.values()}) == len(COMBINES), "one description per rule"
    assert len({*COMBINE_DETAIL.values()}) == len(COMBINES)
    assert len({*COMBINES.values()}) == len(COMBINES)


def test_the_pipeline_names_all_three_steps_in_order():
    fb = FeedbackSettings(normalisation="absolute", combine="weakest", smoothing_seconds=0.4)
    text = fb.pipeline([settled("reward", 10.0, 2.0, 32)])
    assert text.index("5 µV") < text.index("worst") < text.index("0.4 s"), text


def test_a_discrete_cue_has_a_fixed_volume_but_never_a_following_one():
    """The distinction is the whole point, and an earlier version of this test had it too coarse: it asserted
    the click had *no* volume at all, which is why it played at whatever the system was set to — as loud as
    the machine goes.

    A volume that **follows the signal** is wrong: it puts an amount into the one channel whose entire
    content is that something happened. A volume the **operator sets** is not information at all, it is
    calibration — the same thing `Music.volume` is for, so you can balance a click against the music once and
    let Sound move the master over both.
    """
    from nfb_console.feedback import Alert

    click = Click(enabled=True)
    assert 0 <= click.volume <= 100
    # Well under the music, and low in absolute terms because the scale is linear amplitude rather than
    # loudness: `afplay -v` is a straight multiplier, so 25 is -12 dB and 40 would be -8.
    assert click.volume < MUSIC_VOLUME, "a click repeats all session; it sits under the music"
    for following in ("quiet_volume", "volume_follows", "volume_at"):
        assert not hasattr(click, following), f"{following} would make loudness a reading"
    # not printed in the readout: it is a setting, not news
    assert click.describe_output(2.0, 0.8) == "a click every 2 s"
    assert click.describe_output(0.0, 0.8) == "not clicking"

    assert Alert().volume == click.volume, \
        "both are one mark through one path in one room; the sound and the rhythm tell them apart"


def test_sound_is_the_system_volume():
    fb = FeedbackSettings()
    assert fb.sound.quantity == "system volume"
    assert (fb.sound.low, fb.sound.high) == SOUND_RANGE


def test_every_calibrated_default_comes_from_one_place():
    """**The room is one room, and it was being re-balanced on every screen.** The click was 15 here and 25
    written out again inside alpha-theta; the alert was 25 for an argument the click's own docstring
    contradicted; the master volume ran 60-100 everywhere except the coupling screen, which said 15-70. Four
    numbers set by ear, none of them agreeing, and nothing to say which was right.

    So the assertion is not on the values — those are somebody's ear and will move — but on there being one
    of each. A screen may still differ, and `pinned_feedback` is how it says so.
    """
    fb = FeedbackSettings()
    assert fb.click.volume == fb.alert.volume == CLICK_VOLUME
    assert fb.music.volume == MUSIC_VOLUME
    assert (fb.sound.low, fb.sound.high) == SOUND_RANGE
    assert (fb.shadow.low, fb.shadow.high) == SHADOW_RANGE


def test_music_is_owned_by_the_session_and_is_not_one_of_the_mechanisms():
    """It was named "carries no level", and that stopped being true when the Shepard tone arrived.

    Two of the five sources now carry the signal — `crossfade` follows the protocol's *state* and `shepard`
    follows the *level* — so what makes `Music` not a mechanism is no longer that it carries nothing. It is
    that it has no `Ramp` and no `reads`: a mechanism maps the level through a range an operator sets, and
    every music source instead has its own relationship to the signal, chosen with the source. Hence the
    separate `balance` and `level` arguments on `MusicOutput.apply` rather than one.
    """
    from nfb_console.feedback import MUSIC_SOURCES, Music

    fb = FeedbackSettings()
    assert isinstance(fb.music, Music)
    assert not fb.music.enabled
    assert fb.music.source == "noise", "the one source that needs nothing chosen before it will play"
    assert set(MUSIC_SOURCES) == {"noise", "crossfade", "tone", "shepard", "eeg", "file"}
    # An octave around the rhythm most often trained, and it maps onto the default 220–440 Hz pitch range
    # as an exact transposition. It was the whole 2–38 Hz while the band only had to be *audible*; now its
    # frequency is the pitch, and a frequency estimate over a 19:1 band is an average of everything in it.
    assert (fb.music.low_hz, fb.music.high_hz) == (8.0, 16.0)
    assert "music" not in fb.mechanisms, "it is audio the session owns, not a carrier of the signal"
    assert fb.music.volume == 60.0


# --- the three arrangements, and the two mechanisms CFC needed --------------------------------


def test_inversion_is_a_flag_because_a_backwards_range_is_deliberately_not_one():
    """`Ramp` sorts its ends so entering a range backwards is not a trap, which means swapping low and
    high cannot express Quiet Reward. Both rules are worth having."""
    from nfb_console.feedback import Sound

    normal = Sound(enabled=True, low=60, high=100)
    quiet = Sound(enabled=True, low=60, high=100, invert=True)
    assert normal.output(1.0) == 100 and normal.output(0.0) == 60
    assert quiet.output(1.0) == 60 and quiet.output(0.0) == 100
    # entering it backwards still does not invert it
    backwards = Sound(enabled=True, low=100, high=60)
    assert backwards.output(1.0) == 100


def test_the_alert_is_the_mirror_of_the_click():
    from nfb_console.feedback import Alert, Click

    click, alert = Click(enabled=True), Alert(enabled=True)
    assert click.output(0.9, on_target=True) > 0 and click.output(0.9, on_target=False) == 0
    assert alert.output(0.9, on_target=True) == 0 and alert.output(0.9, on_target=False) > 0
    assert alert.kind == "discrete"
    assert "quiet" in alert.describe_output(0.0)


def test_a_feedback_model_clears_what_it_does_not_use():
    """A preset that only adds is not a preset, it is an accumulation."""
    from nfb_console.feedback import FeedbackSettings

    fb = FeedbackSettings()
    fb.apply_model("proportional")
    assert fb.sound.enabled and not fb.sound.invert and not fb.alert.enabled

    fb.apply_model("quiet")
    assert fb.sound.enabled and fb.sound.invert, "silence is the reward"
    assert fb.alert.enabled, "the alerts are the only audio"
    assert fb.music.stops_when == "met", "Quiet Reward: silence is the reward, and it stops rather than fades"

    fb.apply_model("pause")
    assert fb.music.stops_when == "failing" and fb.alert.enabled
    assert not fb.sound.invert, "the previous arrangement must not survive"

    with pytest.raises(ValueError):
        fb.apply_model("nonsense")


def test_the_alert_explains_itself_behind_a_question_mark_not_on_the_screen():
    """It is the one mechanism whose purpose is not guessable from its controls, so the explanation has to
    exist — but standing prose next to a control is rent charged to a tenant with nothing new to say after
    the first reading. The pane carries labels; the `?` carries the reasoning."""
    from nfb_console import signal_pane
    from nfb_console.feedback import FeedbackSettings
    from nfb_console.music import MusicOutput

    class Page:
        def __init__(self):
            self.services = []
            self.dialogs = []

        def update(self, *a):
            pass

        def run_task(self, fn, *a):
            pass

        def show_dialog(self, dialog):
            self.dialogs.append(dialog)

        def pop_dialog(self):
            self.dialogs.pop()

    class Session:
        conditions_signal = True
        scores_bands = True
        tunes_envelope = True
        hold_choices = None
        hold_seconds = 0.0
        level_source = None
        feedback = FeedbackSettings()
        targets: list = []
        music = MusicOutput()
        music_sources = ("noise", "file", "eeg")
        state_ends = None
        state_balance = None
        pinned_feedback = frozenset()
        arrangements = ()
        offers = ("sound", "shadow", "click", "alert", "music")
        combines = ("weighted", "weakest", "scaled")
        tone_bias = None
        unbiased = False

        class gates:
            policy = type("P", (), {"enabled": False})()

            @staticmethod
            def describe():
                return ""

            rate = 0.0

    page = Page()
    pane = signal_pane.SignalPane(page, Session())

    on_screen = " ".join(_texts(pane.alert_rows))
    assert len(on_screen) < 400, f"the pane is prose again: {len(on_screen)} chars"
    assert " he " not in on_screen and " his " not in on_screen

    helper = _find_help(pane.control, "Alert")
    assert helper is not None, "the explanation has to be reachable"
    helper.on_click(None)
    body = "".join(span.text for span in page.dialogs[-1].content.content.spans)
    assert "mirror of the click" in body
    assert "jerky" in body, "why a sound rather than a pause"
    assert " he " not in body and " his " not in body, "no people in the interface"


def _texts(control, depth=0):
    import flet as ft
    if control is None or depth > 12:
        return []
    if isinstance(control, ft.Text):
        return [control.value or ""]
    out = []
    for child in getattr(control, "controls", None) or []:
        out += _texts(child, depth + 1)
    content = getattr(control, "content", None)
    if content is not None:
        out += _texts(content, depth + 1)
    return out


def _find_help(control, tooltip, depth=0):
    import flet as ft
    if control is None or depth > 14:
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


def test_a_keyed_alert_sounds_like_where_the_trouble_is():
    """Twelve bins mean "something is over" is not actionable. Configuring each bin's sound by
    hand is tedious; deriving it from the band means a thirteenth already sounds right."""
    from nfb_console import cues

    slow = cues.frequency_click(3.0)
    alpha = cues.frequency_click(10.0)
    fast = cues.frequency_click(35.0)
    assert len({slow, alpha, fast}) == 3, "the six bin centres must be distinguishable"
    assert not set(cues.click_choices()) & {slow, alpha, fast}, \
        "derived profiles must not leak into the menu of click characters"
    centres = [sum(cues.profile_of(n)[:2]) / 2 for n in (slow, alpha, fast)]
    assert centres[0] < centres[1] < centres[2], "higher band, higher sound"
    decays = [cues.profile_of(n)[2] for n in (slow, alpha, fast)]
    assert decays[0] > decays[1] > decays[2], "slow activity thocks, fast activity ticks"
    # and it renders, rather than naming a profile nothing can play
    assert cues.click_path(alpha) is not None


def test_the_alert_picks_the_worst_offender_not_the_first_in_the_list():
    """List order would let a bin barely over threshold mask one far over."""
    from nfb.dsp import Band
    from nfb_console.feedback import FeedbackSettings
    from nfb_console.targets import Target
    from nfb_console.training import TrainingSession

    session = TrainingSession.__new__(TrainingSession)
    session.feedback = FeedbackSettings()
    session.feedback.alert.enabled = True
    session.state = {}          # no channel status known: every target counts as live
    session._last_alert = {}
    session._alerting = set()
    played: list[str] = []
    session.cue = lambda name, volume=None: played.append(name)

    barely = Target(0, Band(2, 6, ""), role="inhibit", threshold=10.0)
    barely.observe(10.5, 20.0)
    far = Target(0, Band(20, 30, ""), role="inhibit", threshold=10.0)
    far.observe(90.0, 20.0)
    session.targets = [barely, far]

    session._maybe_alert()
    assert played, "something should have sounded"
    assert played[-1] == cues_name(far), "the one that is far over"


def cues_name(target):
    from nfb_console import cues
    return cues.frequency_click(target.band.centre)


def test_the_output_readout_cannot_change_its_own_height():
    """Its lines are rewritten every frame from live values, so when one grew long enough to wrap the panel
    grew a row — and every pane below it on the Feedback tab jumped by that much and back again on the next
    frame. A readout that changes the layout of the screen it is on is unusable however correct its numbers
    are; the fix is to reserve the worst-case height and cap the lines, so content varies and geometry
    cannot. Same rule as the chronospectrogram not changing shape while it fills.
    """
    import flet as ft

    from nfb_console import signal_pane
    from nfb_console.music import MusicOutput
    from nfb_console.targets import Target

    class Page:
        def __init__(self): self.services, self.dialogs = [], []
        def show_dialog(self, d): self.dialogs.append(d)
        def pop_dialog(self): self.dialogs.pop()
        def update(self, *a): pass

    class Session:
        conditions_signal = True
        scores_bands = True
        tunes_envelope = True
        hold_choices = None
        hold_seconds = 0.0
        level_source = None
        feedback = FeedbackSettings()
        targets = [Target(0, SMR, "increase")]
        music = MusicOutput()
        music_sources = ("noise", "file", "eeg")
        state_ends = None
        state_balance = None
        pinned_feedback = frozenset()
        arrangements = ()
        offers = ("sound", "shadow", "click", "alert", "music")
        combines = ("weighted", "weakest", "scaled")
        tone_bias = None
        unbiased = False
        level = 0.5

        class gates:
            policy = type("P", (), {"enabled": False})()

            @staticmethod
            def describe():
                return ""

            rate = 0.0

    pane = signal_pane.SignalPane(Page(), Session())
    strip = pane._strips[0]

    def reserved(control):
        """The fixed-height box a text sits in, found by walking the built strip."""
        found = []

        def walk(node, depth=0):
            if node is None or depth > 12:
                return
            if isinstance(node, ft.Container) and node.content is control:
                found.append(node)
            for child in getattr(node, "controls", None) or []:
                walk(child, depth + 1)
            walk(getattr(node, "content", None), depth + 1)

        walk(pane.control)
        return found[0] if found else None

    for name in ("headline", "note"):
        text = strip[name]
        assert text.max_lines, f"{name} may not grow without bound"
        assert text.overflow == ft.TextOverflow.ELLIPSIS
        box = reserved(text)
        assert box is not None and box.height, f"{name} has no reserved height"

    # and the height does not depend on what the frame happened to say
    before = (reserved(strip["headline"]).height, reserved(strip["note"]).height)
    pane.refresh()
    Session.feedback.combine = "weakest"
    pane.refresh()
    assert (reserved(strip["headline"]).height, reserved(strip["note"]).height) == before


def test_one_dial_widens_the_level_where_the_old_per_target_one_could_not():
    """**The level is off-centre before it is narrow, which is why a steepness control could not fix it.**

    A combined score sits wherever the thresholds and the mix put it — 0.7 on the default amplitude
    arrangement — so steepening each target's ramp pushes it towards 1 and clips. Measured on a real
    session, halving the old per-target span moved the level's 5th-to-95th percentile from 0.58-0.81 to
    0.63-0.89: 0.03 wider, and shifted up. The dial that replaced it re-centres first and then scales, and
    both halves are needed.

    It is also the direction that was wrong before: bigger meant flatter, and it was reported turned up to
    10 to get more movement.
    """
    from nfb_console.feedback import SENSITIVITY_HEADROOM, SENSITIVITY_RANGE, FeedbackSettings

    fb = FeedbackSettings()
    centre, span = 0.66, 0.20               # what a real session measured over its first minute

    # Uncalibrated is the identity, so a session feeds back from its first reading rather than waiting.
    assert fb.shape(0.81, 0.0, 0.0) == pytest.approx(0.81)

    # The middle of the measured swing is the middle of the output, whatever the combining left it at.
    assert fb.shape(centre, centre, span) == pytest.approx(0.5)

    # At 1, the calibrated swing uses half the travel — the top of it lands at 0.75, not at the ceiling.
    top = fb.shape(centre + span / 2, centre, span)
    assert top == pytest.approx(0.5 + 1.0 / (2 * SENSITIVITY_HEADROOM), abs=0.001)
    assert 0.7 < top < 0.8

    # Up is more. The old control's whole defect was that this read the other way.
    fb.sensitivity = 2.0
    assert fb.shape(0.75, centre, span) > top
    fb.sensitivity = 0.5
    assert fb.shape(0.75, centre, span) < top

    # And it cannot leave the range, however far the signal goes or however hard the dial is turned —
    # though the ends are approached rather than hit (see SHAPE_KNEE), so "at" an end means within a
    # rounding error of it rather than exactly on it.
    fb.sensitivity = SENSITIVITY_RANGE[1]
    assert fb.shape(5.0, centre, span) == pytest.approx(1.0, abs=1e-12)
    assert fb.shape(-5.0, centre, span) == pytest.approx(0.0, abs=1e-12)


def test_the_levels_range_is_measured_from_the_session_and_then_held():
    """**Held against re-centring, not rolling.** A rolling stretch re-normalises the trainee every
    window, so a genuinely better state fades back to the middle within a minute — which is the trap an
    automatic threshold is already in, arrived at from the other end. Once the range settles, doing
    better than you did during calibration reads as more output and stays there: the middle and the
    bottom never chase the person. Only an end the level lives *beyond* moves, outward — that is
    `test_a_level_living_beyond_its_range_gets_its_gradient_back`'s subject.

    It settles at 60 s rather than 30 because the level is built from scores against thresholds, and those
    are themselves provisional for their own first 30 s: a range derived from that stretch would be
    measuring the thresholds settling as much as the person.
    """
    from nfb_console.training import LevelRange

    rng = LevelRange()
    assert rng.span == 0.0, "nothing measured yet, so `shape` is the identity"

    # A minute of a level swinging between 0.6 and 0.8, at the metric rate.
    now = 0.0
    for i in range(int(LevelRange.SETTLE_SECONDS * 20) + 1):
        now = i / 20.0
        rng.observe(0.7 + 0.1 * math.sin(2 * math.pi * 0.2 * now), now)

    assert rng.settled and rng.remaining(now) == 0.0
    assert rng.centre == pytest.approx(0.7, abs=0.02)
    assert rng.span == pytest.approx(0.19, abs=0.03), "a 5th-to-95th percentile of a ±0.1 swing"

    # Held: a trainee who now sits high, but still inside the travel, moves nothing at all — the range
    # is a calibration, not a follower. 0.85 is under centre + span, where `shape` still has room.
    centre, span = rng.centre, rng.span
    for i in range(int(300 * 20)):
        now += 1 / 20.0
        rng.observe(0.85, now)
    assert (rng.centre, rng.span) == (centre, span)

    # And sitting *low* inside the travel is equally not a reason to move: no end re-approaches, so a
    # better-than-calibration state reached later in the session still reads high.
    for i in range(int(300 * 20)):
        now += 1 / 20.0
        rng.observe(0.55, now)
    assert (rng.centre, rng.span) == (centre, span)


def test_a_level_living_beyond_its_range_gets_its_gradient_back():
    """**The ceiling is a knee, not a wall — and the wall was experienced, not hypothesised.** The range
    is calibrated from the first minute and a training session exists to push the level past its own
    first minute; with a hard clip half a span above the calibrated high, an SMR session on the elliptic
    reward default parked volume and brightness at their maxima for good, which the person in the chair
    reported as no feedback at all. (The old cFIR default hid the wall: its score visited the top only in
    ~100 ms flickers the level smoothing averaged away. Replayed on `sessions/20260812`, elliptic sat
    hard at 1.0 for 6.5% of a *resting* recording; cFIR 0%.) Two halves to the fix, tested in order:
    `shape`'s ends bend away instead of clipping, and the range's ends ratchet out to meet a level living
    beyond them — outward only, so nothing here re-normalises a better state back to the middle.
    """
    from nfb_console.feedback import FeedbackSettings
    from nfb_console.training import LevelRange

    fb = FeedbackSettings()
    rng = LevelRange()
    now = 0.0
    for i in range(int(LevelRange.SETTLE_SECONDS * 20) + 1):
        now = i / 20.0
        rng.observe(0.55 + 0.1 * math.sin(2 * math.pi * 0.2 * now), now)
    assert rng.settled
    top = rng.centre + rng.span                 # where `shape` used to run out of travel

    # The knee, before the range has moved at all: past the old clip point, more is still more.
    at_wall = fb.shape(top, rng.centre, rng.span)
    assert at_wall < 1.0
    assert fb.shape(0.96, rng.centre, rng.span) > fb.shape(0.93, rng.centre, rng.span) > at_wall

    # A level living above the travel drags the top end out to meet it — RATCHET_SECONDS is 10, so a
    # minute is six time constants and the end has essentially arrived.
    bottom = rng.centre - rng.span
    for i in range(int(60 * 20)):
        now += 1 / 20.0
        rng.observe(0.95, now)
    assert rng.centre + rng.span == pytest.approx(0.95, abs=0.01)
    assert rng.centre - rng.span == pytest.approx(bottom, abs=1e-9), "the bottom never chased anything"

    # Outward only: coming back inside the widened range pulls nothing back in. Doing better than
    # calibration still reads high for the rest of the session — the held promise, kept.
    grown = (rng.centre, rng.span)
    for i in range(int(120 * 20)):
        now += 1 / 20.0
        rng.observe(0.7, now)
    assert (rng.centre, rng.span) == grown
    assert fb.shape(0.95, rng.centre, rng.span) > 0.9, "the state that was pinned is still near the top"

    # And a 100 ms flicker past an end is a nudge, not a recalibration.
    before = rng.centre + rng.span
    for i in range(2):
        now += 1 / 20.0
        rng.observe(1.0, now)
    assert rng.centre + rng.span - before < 0.002


def test_the_level_splits_by_role_so_a_down_trained_reward_dominates_it():
    """**The whole point of separating the two axes.** Ros and Lanius train 8–12 Hz downward, and that
    band is what the session is about — so it has to lead the level, with the artifact bands pulling on
    it. Split by direction, it landed in the inhibit group and got 20% of the mix, and every artifact
    inhibit added beside it halved that again.
    """
    from nfb_console.feedback import DEFAULT_REWARD_SHARE, combine

    alpha = settled("reward", 10.0, 2.0, direction="decrease")
    emg = settled("inhibit", 10.0, 2.0, seed=1)
    eyes = settled("inhibit", 10.0, 2.0, seed=2)

    alpha.value, emg.value, eyes.value = 4.0, 16.0, 16.0     # alpha quiet, artifacts firing
    good, bad = alpha.score(), emg.score()
    assert good > 0.8 and bad < 0.2, "the setup is what the test claims it is"

    level = combine([alpha, emg, eyes], "weighted")
    assert level == pytest.approx(DEFAULT_REWARD_SHARE * good
                                  + (1 - DEFAULT_REWARD_SHARE) * (bad + eyes.score()) / 2, abs=1e-6)
    assert level > 0.5, "the band being trained leads the level"

    # What it used to do: the same "keep alpha low" written as an inhibit, pooled with its own artifacts.
    as_inhibit = settled("inhibit", 10.0, 2.0)
    as_inhibit.value = 4.0
    assert as_inhibit.score() == pytest.approx(good, abs=1e-6), "same reading, same side of the line"
    pooled = combine([as_inhibit, emg, eyes], "weighted")
    assert pooled == pytest.approx((good + bad + eyes.score()) / 3, abs=1e-6)
    assert pooled < 0.5, "and the session's subject counted for a third"


def test_the_condition_under_either_takes_its_alternatives_from_the_role():
    """`either` makes the rewards alternatives and the inhibits vetoes. Which targets are which is a
    statement about role, so a down-trained reward is one of the alternatives rather than a veto."""
    from nfb_console.feedback import FeedbackSettings

    up = settled("reward", 10.0, 2.0)
    down = settled("reward", 10.0, 2.0, seed=1, direction="decrease")
    inhibit = settled("inhibit", 10.0, 2.0, seed=2)

    up.value, down.value, inhibit.value = 4.0, 4.0, 4.0    # up failing, down earning, inhibit clear
    rule = FeedbackSettings(combine="either")
    assert not up.wanted and down.wanted and inhibit.wanted, "the setup is what it claims"
    assert rule.on_target([up, down, inhibit]), "one of the two rewards is met, which is enough"

    inhibit.value = 16.0
    assert not rule.on_target([up, down, inhibit]), "and the inhibit is still a veto"
