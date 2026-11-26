"""The shadow: the subject's display following the signal.

The tests that matter here are the safety ones. A brightness bug is not a wrong number on a screen —
it is a machine someone cannot see well enough to stop, so the floor and the restore are pinned rather
than trusted, and both are exercised through a fake display so the suite never touches the real one.
"""

from __future__ import annotations

import pytest

from nfb_console import shadow


@pytest.fixture
def fake(monkeypatch):
    """A display that records what it was set to, in place of DisplayServices."""
    state = {"brightness": {1: 0.6, 2: 0.4}, "writes": []}

    def fake_write(display, level):
        level = min(1.0, max(0.0, level))       # the real _write clamps only to [0, 1]; set() floors on top
        state["brightness"][display] = level
        state["writes"].append((display, round(level, 4)))

    # **The clock is part of the fake, because the module smooths in seconds of wall clock.** A test that
    # drives `apply` in a loop passes no time and nothing moves; `drive` advances this instead of sleeping.
    state["t"] = [1000.0]
    monkeypatch.setattr(shadow, "clock", lambda: state["t"][0])
    monkeypatch.setattr(shadow, "displays", lambda: [1, 2])
    monkeypatch.setattr(shadow, "available", lambda: True)
    monkeypatch.setattr(shadow, "problem", lambda: None)
    monkeypatch.setattr(shadow, "get", lambda d: state["brightness"].get(d))
    monkeypatch.setattr(shadow, "_write", fake_write)   # the write boundary; real set() floors into it
    return state


def drive(sh, fake, level, low=20, high=100, screen=1, seconds=6.0, hz=20.0, smoothing=0.0):
    """Call `apply` at `hz` for `seconds` of the fake clock, as a session would."""
    for _ in range(int(seconds * hz)):
        fake["t"][0] += 1.0 / hz
        sh.apply(True, screen, low, high, level, smoothing=smoothing)
    return fake["brightness"][sh.display]


def settle(sh, fake, level, low=20, high=100, screen=1, seconds=6.0):
    return drive(sh, fake, level, low, high, screen, seconds=seconds)


def test_the_smoothing_is_in_seconds_and_not_per_call(fake):
    """**The bug this file did not catch, because it drove `apply` and never looked at the clock.**

    The constants were per call and the module assumed 20 Hz. Measured, a two-channel 250 sps replay drives
    it at 21 — which is why it looked right — while `_drive_outputs` runs whenever any target advanced, so a
    four-channel 1000 sps session drives it far faster and every time constant shrank by that ratio. The
    shadow was least smoothed exactly where the signal arrived fastest.
    """
    reached = {}
    for hz in (20.0, 200.0):
        # Reset the display, or the second run starts from where the first left it and the comparison is
        # between two different starting points rather than between two call rates.
        fake["brightness"][1] = 0.6
        sh = shadow.DisplayShadow()
        reached[hz] = drive(sh, fake, level=1.0, low=10, high=100, seconds=0.3, hz=hz)
    assert reached[20.0] == pytest.approx(reached[200.0], abs=0.02), (
        f"ten times the call rate changed the brightness reached: {reached}")


def test_extra_smoothing_attenuates_a_flickering_level(fake):
    """A level crossing thresholds several times a second strobes the whole backlight — measured at about
    3 Hz on a real multi-target condition, see `feedback.HOLD_FRACTION`.

    A **low-pass** rather than the slew ceiling this first shipped with, and the difference is the point:
    rate-limiting a square wave yields a triangle wave at the same frequency, so it shrinks the flicker
    without removing it. `Shadow.smoothing_seconds` is the control, per protocol.
    """
    def swing(smoothing: float) -> float:
        sh = shadow.DisplayShadow()
        drive(sh, fake, level=0.5, low=20, high=100, seconds=5.0, smoothing=smoothing)
        before = len(fake["writes"])
        # 3 Hz square wave between the ends of the range, driven at 60 Hz
        for i in range(300):
            fake["t"][0] += 1 / 60.0
            sh.apply(True, 1, 20, 100, 1.0 if (i // 10) % 2 else 0.0, smoothing=smoothing)
        seen = [b for _d, b in fake["writes"][before:]]
        return max(seen) - min(seen)

    none = swing(0.0)
    some = swing(0.4)
    assert none > 0.05, f"without it the backlight really does swing ({none:.0%})"
    assert some < none / 2, f"0.4 s has to cut it hard: {none:.0%} -> {some:.0%}"


def test_the_floor_is_enforced_below_whatever_is_configured(fake):
    """Configuring 0% and asking for a level of zero must not black the screen out: you cannot find
    the button that stops it."""
    sh = shadow.DisplayShadow()
    assert settle(sh, fake, level=0.0, low=0, high=0) == pytest.approx(shadow.HARD_FLOOR)
    assert min(level for _d, level in fake["writes"]) >= shadow.HARD_FLOOR


def test_the_range_maps_the_level_and_nothing_exceeds_it(fake):
    sh = shadow.DisplayShadow()
    assert settle(sh, fake, level=0.0, low=30, high=90) == pytest.approx(0.30, abs=0.01)
    assert settle(sh, fake, level=1.0, low=30, high=90) == pytest.approx(0.90, abs=0.01)
    assert settle(sh, fake, level=0.5, low=30, high=90) == pytest.approx(0.60, abs=0.01)
    assert settle(sh, fake, level=5.0, low=30, high=90) == pytest.approx(0.90, abs=0.01), "clamped"


def test_a_range_entered_backwards_still_works(fake):
    sh = shadow.DisplayShadow()
    assert settle(sh, fake, level=1.0, low=90, high=30) == pytest.approx(0.90, abs=0.01)


def test_the_operators_brightness_is_restored(fake):
    sh = shadow.DisplayShadow()
    settle(sh, fake, level=0.0, low=15, high=100)
    assert fake["brightness"][1] < 0.5, "it really did dim"
    sh.release()
    assert fake["brightness"][1] == 0.6, "and it came back to what it was"
    assert not sh.active


def test_restore_gives_back_a_dim_display_but_floors_a_true_black(fake):
    """Restore routed through the floored set(), so a 2% (dark sleep room) start came back at 10%. Restore
    now gives back what it found, flooring only a true black (0.0) so a session cannot leave the screen dark."""
    fake["brightness"][1] = 0.02               # the operator's dark-room brightness
    sh = shadow.DisplayShadow()
    settle(sh, fake, level=1.0, low=20, high=100)
    sh.release()
    assert fake["brightness"][1] == pytest.approx(0.02), "a low-but-lit brightness comes back as found"

    fake["brightness"][2] = 0.0                # a display found fully off
    sh2 = shadow.DisplayShadow()
    settle(sh2, fake, level=1.0, low=20, high=100, screen=2)
    sh2.release()
    assert fake["brightness"][2] == pytest.approx(shadow.HARD_FLOOR), "a true black is floored, not left dark"


def test_switching_the_mechanism_off_restores_without_a_separate_call(fake):
    """`apply` owns the lifecycle, so a caller cannot forget to hand the display back."""
    sh = shadow.DisplayShadow()
    settle(sh, fake, level=0.0, low=15, high=100)
    sh.apply(False, 1, 15, 100, 0.0)
    assert fake["brightness"][1] == 0.6
    assert not sh.active


def test_moving_to_another_screen_hands_the_first_one_back(fake):
    sh = shadow.DisplayShadow()
    settle(sh, fake, level=0.0, low=15, high=100, screen=1)
    settle(sh, fake, level=0.0, low=15, high=100, screen=2)
    assert fake["brightness"][1] == 0.6, "screen 1 restored when we left it"
    assert fake["brightness"][2] < 0.4, "screen 2 now being driven"
    sh.release()
    assert fake["brightness"][2] == 0.4


def test_a_screen_number_past_the_end_falls_back_to_the_main_display(fake):
    """A shadow on the wrong screen is a visible mistake; a shadow on no screen is silence, and the
    operator would conclude the feature is broken."""
    sh = shadow.DisplayShadow()
    settle(sh, fake, level=0.0, low=15, high=100, screen=9)
    assert sh.display == 1


def test_the_target_is_approached_rather_than_jumped_to(fake):
    """Stepping the backlight at 20 Hz is visible flicker even from a smooth signal."""
    sh = shadow.DisplayShadow()
    started = fake["brightness"][1]
    sh.apply(True, 1, 20, 100, 1.0)
    first = fake["brightness"][1]
    assert first > started, "it must move toward the target"
    assert first < 1.0, "one step must not arrive at the target"
    assert settle(sh, fake, level=1.0) == pytest.approx(1.0, abs=0.01), "and it must get there"


def test_release_is_safe_before_anything_was_taken(fake):
    sh = shadow.DisplayShadow()
    sh.release()
    sh.release()
    assert fake["writes"] == []


def test_nothing_happens_when_the_platform_cannot_do_it(monkeypatch):
    monkeypatch.setattr(shadow, "available", lambda: False)
    sh = shadow.DisplayShadow()
    sh.apply(True, 1, 20, 100, 1.0)
    assert not sh.active
    assert shadow.problem() and shadow.problem() in sh.describe(), "the screen says why"


def test_the_session_gives_the_display_back_when_it_stops(fake):
    from nfb_console import amplitude_view as av
    from nfb_console.feedback import FeedbackSettings
    from nfb_console.protocols import BY_CODE

    session = av.AmplitudeSession.__new__(av.AmplitudeSession)
    session.protocol = BY_CODE["AMP"]
    session.feedback = FeedbackSettings()
    session.feedback.shadow.enabled = True
    session.display = shadow.DisplayShadow()
    session.output_volume = type("V", (), {"release": lambda self: None})()
    session.music = type("M", (), {"release": lambda self: None})()
    session._level = 0.0
    session._stop = False
    from nfb_console.outputs import OutputPump

    session._outputs = OutputPump(session._apply_snapshot, session._release_outputs)   # never started: inline
    session.paused = False
    session.device = None
    session.source = type("S", (), {"close": lambda self: None})()

    # Time has to pass: the smoothing is in seconds of wall clock, not per call — see `APPROACH_SECONDS`.
    for _ in range(120):
        fake["t"][0] += 0.05
        session._drive_shadow()
    assert fake["brightness"][1] < 0.5 and session.display.active

    session.stop()
    assert fake["brightness"][1] == 0.6, "stopping a session must not leave the machine dimmed"
    assert not session.display.active


def test_an_unsupported_shadow_mechanism_does_not_touch_the_display(fake):
    from nfb_console import amplitude_view as av
    from nfb_console.feedback import FeedbackSettings
    from nfb_console.protocols import BY_CODE

    session = av.AmplitudeSession.__new__(av.AmplitudeSession)
    session.protocol = BY_CODE["AMP"]
    session.feedback = FeedbackSettings()
    session.feedback.shadow.enabled = True
    session.feedback.shadow.mechanism = "overlay"        # named but not implemented
    session.display = shadow.DisplayShadow()
    session._level = 0.0

    session._drive_shadow()
    assert fake["writes"] == [], "an unimplemented mechanism must stay inert"


def test_outputs_can_be_switched_off_for_a_whole_process(monkeypatch):
    """A session takes over the display and the volume the moment it runs, which is right for a session
    and wrong for a benchmark, a scratch script or anything else that merely imports one. `NFB_NO_OUTPUTS`
    is the switch, and it exists because a measurement script dimmed the operator's screen while it ran."""
    from nfb_console import cues, shadow, sysvolume

    monkeypatch.setenv("NFB_NO_OUTPUTS", "1")
    assert shadow.muted() and sysvolume.muted() and cues.muted()
    assert not shadow.available() and not sysvolume.available() and not cues.available()
    assert "NFB_NO_OUTPUTS" in (shadow.problem() or "")
    assert "NFB_NO_OUTPUTS" in (sysvolume.problem() or "")
    assert cues.play("beep") is False

    # and a session that thinks the shadow is on still takes nothing
    sh = shadow.DisplayShadow()
    sh.apply(True, 1, 20, 100, 0.0)
    assert not sh.active

    # **Not one call to the display, not even to look.** The suite was asking WindowServer for the display
    # list a few dozen times a run, through the Feedback pane's "N attached" note. Nothing measurable came of
    # it — the brightness never moved — but "outputs are off" is easier to keep as "the module does not open
    # that door" than as "it opens it only to read", and a promise about the operator's hardware is worth
    # holding at the boundary.
    assert shadow.displays() == [] and shadow.get(1) is None

    monkeypatch.delenv("NFB_NO_OUTPUTS")
    assert not shadow.muted()
