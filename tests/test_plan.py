"""Session length and the cues that mark time inside it.

The cases worth pinning are the ones that misbehave silently: a session that ends when nobody asked
for an ending, a finishing tone that never sounds because the length was edited after it fired, and
reminders that queue up and then play four at once.
"""

from __future__ import annotations

import pytest

from nfb_console.plan import SessionPlan


def test_open_ended_is_the_default_and_never_completes():
    plan = SessionPlan()
    assert not plan.bounded
    assert plan.remaining(0) is None
    assert plan.remaining(86400) is None
    assert not plan.complete(86400)
    assert plan.due(86400) == []


def test_a_limit_completes_once_and_announces_once():
    plan = SessionPlan()
    plan.set_length(10)
    assert plan.remaining(0) == 600
    assert plan.remaining(700) == 0, "remaining should floor at zero, not go negative"
    assert plan.due(599) == []
    assert plan.due(600) == ["finish"]
    assert plan.due(601) == [], "the finishing tone must not repeat every frame"


def test_extending_a_finished_session_rearms_the_finishing_tone():
    plan = SessionPlan()
    plan.set_length(10)
    assert plan.due(600) == ["finish"]
    plan.set_length(20)
    assert plan.due(700) == []
    assert plan.due(1200) == ["finish"]


def test_reminders_fire_on_the_interval_and_do_not_queue_up():
    plan = SessionPlan()
    plan.set_reminder(1)
    assert plan.due(59) == []
    assert plan.due(60) == ["beep"]
    assert plan.due(119) == []
    # a gap: four intervals passed while nothing was drawing. One beep, not four.
    assert plan.due(400) == ["beep"]
    assert plan.due(401) == []


def test_reminders_do_not_land_on_top_of_the_ending():
    """A beep and a finishing tone in the same instant would be heard as one muddled sound, and
    the ending is the informative one."""
    plan = SessionPlan()
    plan.set_length(5)
    plan.set_reminder(1)
    assert plan.due(60) == ["beep"]
    assert plan.due(300) == ["finish"]


def test_next_reminder_counts_down_and_stops_at_the_end():
    plan = SessionPlan()
    plan.set_reminder(2)
    assert plan.next_reminder(0) == 120
    assert plan.next_reminder(90) == 30
    plan.due(120)
    assert plan.next_reminder(120) == 120

    plan.set_length(3)
    plan.set_reminder(2)
    assert plan.next_reminder(0) == 120
    plan.due(120)
    assert plan.next_reminder(121) is None, "no reminder is due past the end of the session"

    assert SessionPlan().next_reminder(10) is None


def test_cues_can_be_silenced_without_losing_the_schedule():
    plan = SessionPlan(reminder_cue=False, finish_cue=False)
    plan.set_length(2)
    plan.set_reminder(1)
    assert plan.due(60) == []
    assert plan.due(120) == []
    assert plan.complete(120), "the session still ends; it just does so quietly"


def test_a_length_is_never_shorter_than_a_minute():
    plan = SessionPlan()
    plan.set_length(0.1)
    assert plan.length_seconds == 60
    plan.set_length(None)
    assert plan.length_seconds is None


def test_the_session_fires_cues_through_a_swappable_player(monkeypatch):
    """Wired end to end: the model, not the view, is what makes a cue happen."""
    import nfb_console.amplitude_view as av
    from nfb_console.protocols import BY_CODE

    session = av.AmplitudeSession.__new__(av.AmplitudeSession)
    session.plan = SessionPlan()
    session.protocol = BY_CODE["AMP"]
    session.finished = False
    session.paused = False
    session._accumulated = 0.0
    session._resumed_at = None
    played: list[str] = []
    session.cue = played.append

    session.plan.set_length(1)
    session._accumulated = 30.0
    session.tick()
    assert played == [] and not session.finished

    session._accumulated = 60.0
    session.tick()
    assert played == ["finish"]
    assert session.finished and session.paused
    assert session.elapsed == 60.0, "the clock should stop exactly on the limit, not past it"


def test_a_paused_session_does_not_advance_towards_its_cues():
    import nfb_console.amplitude_view as av
    from nfb_console.protocols import BY_CODE

    session = av.AmplitudeSession.__new__(av.AmplitudeSession)
    session.plan = SessionPlan()
    session.protocol = BY_CODE["AMP"]
    session.finished = False
    session.paused = True
    session._accumulated = 600.0
    session._resumed_at = None
    played: list[str] = []
    session.cue = played.append

    session.plan.set_length(1)
    session.tick()
    assert played == [] and not session.finished


def test_a_cue_plays_once_at_the_volume_it_was_given():
    """The click audition used to fire three clicks so a *rate* could be judged; there is no rate any
    more, and three sounds cannot tell you what one will sound like."""
    from nfb_console import cues

    tick = str(cues.click_path("tick"))
    assert cues.command("tick") == ["afplay", tick], "no -v means the system volume"
    assert cues.command("tick", 75) == ["afplay", "-v", "0.750", tick]
    assert cues.command("tick", 0) == ["afplay", "-v", "0.000", tick]
    assert cues.command("tick", 100) == ["afplay", "-v", "1.000", tick]
    assert cues.command("tick", 250)[2] == "1.000", "clamped, not passed through"
    assert cues.command("tick", -5)[2] == "0.000"
    assert cues.command("beep", 50)[3] == str(cues.SOUNDS["beep"])
    assert cues.command("nope") is None
    assert not hasattr(cues, "burst"), "one click means there is nothing left to repeat"


def test_the_click_carries_its_own_sound():
    from nfb_console import cues
    from nfb_console.feedback import Click

    assert Click().sound == cues.DEFAULT_CLICK == "tick"
    assert Click().sound in cues.click_choices()
    assert cues.command(Click().sound) is not None, "the default has to be playable"


def test_a_click_is_generated_short_and_actually_a_click():
    """macOS ships nothing short enough, so these are rendered. The length is the property that makes
    a click a click, and it also has to fit inside the reward interval."""
    import wave as wavelib

    from nfb_console import cues

    assert set(cues.click_choices()) == {"tick", "clack", "thock"}
    for name in cues.click_choices():
        assert cues.duration_ms(name) <= 40, f"{name} is too long to read as one event"
        path = cues.click_path(name)
        assert path.exists() and path.suffix == ".wav"
        with wavelib.open(str(path)) as fh:
            assert fh.getframerate() == cues.RATE
            assert fh.getnchannels() == 1
            frames = fh.getnframes()
        assert frames == pytest.approx(cues.RATE * cues.duration_ms(name) / 1000, rel=0.01)

    assert cues.click_path("tick") is cues.click_path("tick"), "rendered once, then cached"
    assert cues.click_path("nope") is None
    assert cues.duration_ms("tick") < cues.duration_ms("thock"), "the decay is the character"


def test_a_reward_tone_is_a_pitch_and_not_a_click():
    """A reward that says *which* of two alternatives earned it has to be heard as a pitch.

    The alert's `frequency_click` cannot do that job — it is a few milliseconds of band-limited noise,
    which is heard as a click by design, and across theta and alpha its two versions land under an octave
    apart. Two alpha-theta reward sounds of 137 Hz and 738 Hz are two and a half octaves apart, and
    those are the anchors here.
    """
    import wave as wavelib

    import numpy as np
    from scipy import signal

    from nfb_console import cues

    assert cues.tone_pitch(4.5) == pytest.approx(137.0)
    assert cues.tone_pitch(9.5) == pytest.approx(738.0)
    # between the two bands, between the two pitches; outside them, clamped rather than shrieking
    assert 137.0 < cues.tone_pitch(7.0) < 738.0
    assert cues.tone_pitch(0.5) == cues.TONE_LO_HZ and cues.tone_pitch(40.0) == cues.TONE_HI_HZ

    name = cues.reward_tone(9.5)
    assert name not in cues.click_choices(), "a tone is not one of the click characters"
    path = cues.click_path(name)
    with wavelib.open(str(path)) as fh:
        assert fh.getframerate() == cues.RATE
        data = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(float)
    assert len(data) == pytest.approx(cues.RATE * cues.TONE_MS / 1000, rel=0.01)

    # it really is a tone at that pitch: the spectrum peaks there
    freqs, psd = signal.welch(data, fs=cues.RATE, nperseg=min(2048, len(data)))
    assert freqs[psd.argmax()] == pytest.approx(738.0, rel=0.05)
    # and it is long enough to be a pitch rather than a click: several cycles at 137 Hz
    assert cues.TONE_MS / 1000 * 137 > 15
    assert cues.duration_ms(name) > 3 * max(cues.duration_ms(c) for c in cues.click_choices())


def test_a_generated_click_decays_to_silence():
    """A click that did not decay would be a burst of noise, and the next one would overlap it."""
    import wave as wavelib

    import numpy as np

    from nfb_console import cues

    with wavelib.open(str(cues.click_path("tick"))) as fh:
        data = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(float)
    third = len(data) // 3
    assert np.abs(data[:third]).max() > 10 * np.abs(data[-third:]).max(), \
        "the attack has to dominate the tail, or it reads as a tone"
    assert np.abs(data).max() > 20000, "and it has to be loud enough to hear at all"


def test_clicks_cannot_beat_their_interval_however_the_targets_flicker():
    """A signal sitting near its threshold crosses it constantly. Clearing the click phase on every
    exit meant re-entry fired at once, which machine-gunned — measured gaps of 0.15 s at a 1 s
    interval. The interval holds across the session, not per stretch of being on target.
    """
    from nfb_console import amplitude_view as av
    from nfb_console.feedback import FeedbackSettings
    from nfb_console.protocols import BY_CODE

    session = av.AmplitudeSession.__new__(av.AmplitudeSession)
    session.protocol = BY_CODE["AMP"]
    session.feedback = FeedbackSettings()
    session.feedback.click.enabled = True
    session.feedback.click.interval_seconds = 1.0
    session._last_click = None
    # the hold criterion reads a window of (time, met) samples; a stub without it is not a stub of a session
    from collections import deque
    session._condition_log = deque()
    session._level = 1.0
    session.output_volume = type("V", (), {"apply": lambda *a, **k: None,
                                           "release": lambda self: None})()

    met = [True]
    # `pair` and `source` are part of what a target is — the session asks whether the channel behind a
    # target is delivering signal before letting it drive the output, so a stub without them is not a stub
    # of a target.
    session.state = {0: True}
    session.targets = [type("T", (), {"wanted": property(lambda self: met[0]),
                                      "direction": "increase",
                                      "pair": None,
                                      "source": 0,
                                      "score": lambda self, s="sd": 1.0})()]
    fired: list[float] = []
    session.cue = lambda name, volume=None: fired.append(name)

    clock = [1000.0]
    import nfb_console.amplitude_view as module
    real = module.time.monotonic
    module.time.monotonic = lambda: clock[0]
    try:
        session._maybe_click()
        assert len(fired) == 1, "the first click lands as soon as the targets are met"

        for _ in range(20):                 # flicker on and off every 50 ms for a second
            clock[0] += 0.05
            met[0] = not met[0]
            session._maybe_click()
        assert len(fired) == 1, "flicker must not produce a click"

        clock[0] += 0.1                     # now past the interval
        met[0] = True
        session._maybe_click()
        assert len(fired) == 2

        session.feedback.click.enabled = False
        clock[0] += 5.0
        session._maybe_click()
        assert len(fired) == 2, "disabled means silent"
    finally:
        module.time.monotonic = real
