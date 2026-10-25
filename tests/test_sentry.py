"""The signal sentry: sustained failures alarm, transients never do, and the beep repeats.

Driven entirely by injected clocks — the sentry's whole subject is duration, so every test hands it
`now` and asserts what a night would have heard.
"""

from __future__ import annotations

from types import SimpleNamespace

from nfb_console.sentry import (ADVISORY_SECONDS, ALARM_SECONDS, GARBAGE_SECONDS, REPEAT_SECONDS,
                                STARVED_SECONDS, SignalSentry)


def session_like(channels=(0, 1), device=None):
    beeps: list[str] = []
    volumes: list[float | None] = []
    s = SimpleNamespace(
        specs={c: object() for c in channels},
        state={c: True for c in channels},
        device=device,
        feeding=True,
        running=True,
        error=None,
        beeps=beeps,
        volumes=volumes,
        cue=lambda name, volume=None: (beeps.append(name), volumes.append(volume)),
    )
    s.rejections = {c: 0.0 for c in channels}
    s.rejection = lambda c: s.rejections.get(c, 0.0)
    return s


def test_a_transient_never_alarms_and_a_sustained_failure_does():
    sentry, s = SignalSentry(), session_like()
    sentry.audible = True

    s.state[1] = False                                  # a turn in bed: five bad seconds
    for at in (0.0, 2.0, 4.5):
        assert sentry.tick(s, at) is None
    s.state[1] = True
    assert sentry.tick(s, 5.0) is None
    assert sentry.tick(s, 24.5) is None

    s.state[1] = False                                  # then the electrode really goes
    sentry.tick(s, 25.0)
    assert sentry.tick(s, 84.0) is None, "59 s of badness is not yet the alarm's 60"
    problem = sentry.tick(s, 86.5)
    assert problem is not None and "ch1" in problem and "reattach" in problem
    assert s.beeps == ["beep"], "a fresh alarm sounds at once"

    s.state[1] = True                                   # electrode reattached
    assert sentry.tick(s, 87.0) is not None, "one good moment does not clear it — flaps would re-beep"
    assert sentry.tick(s, 93.0) is None, "a few good seconds do"
    assert [m for _t, m in sentry.events] == [problem, "signal recovered"]


def test_a_flapping_contact_alarms_even_though_it_never_stays_bad():
    """The commonest death of a taped electrode: railed most of the time, moments of real signal in
    between. A continuity test never fires — every good block resets it — so unusable time has to
    accumulate."""
    sentry, s = SignalSentry(), session_like()
    problem = None
    for second in range(0, 95):
        s.state[1] = second % 10 == 9                   # bad nine seconds of every ten
        problem = sentry.tick(s, float(second)) or problem
    assert problem is not None and "ch1" in problem
    assert "most of the last" in problem, "named as accumulated, not as one unbroken spell"

    gentle = SignalSentry()                             # 30% bad is a rough night, not a failure
    g = session_like()
    for second in range(0, 95):
        g.state[1] = second % 10 >= 3
        assert gentle.tick(g, float(second)) is None


def test_every_channel_gone_names_the_reference_and_the_beep_repeats_until_fixed():
    sentry, s = SignalSentry(), session_like()
    sentry.audible = True
    s.state = {0: False, 1: False}
    sentry.tick(s, 0.0)
    problem = sentry.tick(s, ALARM_SECONDS + 1)
    assert "every channel" in problem and "reference electrode" in problem
    for i in range(3):
        sentry.tick(s, ALARM_SECONDS + 1 + (i + 1) * REPEAT_SECONDS)
    assert len(s.beeps) == 4, "one beep per repeat interval for as long as it holds"

    quiet = SignalSentry()                              # audible unarmed: same verdicts, no sound
    q = session_like()
    q.state = {0: False, 1: False}
    quiet.tick(q, 0.0)
    assert quiet.tick(q, ALARM_SECONDS + 1) is not None
    assert q.beeps == []


def test_starvation_alarms_only_with_a_device_attached_and_a_feed_running():
    sentry, s = SignalSentry(), session_like(device=object())
    sentry.saw_block(0.0)
    assert sentry.tick(s, STARVED_SECONDS - 1) is None
    problem = sentry.tick(s, STARVED_SECONDS + 1)
    assert problem is not None and "no samples" in problem

    replay = SignalSentry()                             # no device: a file source cannot starve
    r = session_like(device=None)
    replay.saw_block(0.0)
    assert replay.tick(r, STARVED_SECONDS + 5) is None


def test_the_feed_dying_is_an_alarm_not_a_silent_disarm():
    """`feeding` going False used to gate the starvation alarm off — which silenced the watchdog in
    exactly the case where the software itself broke."""
    sentry, s = SignalSentry(), session_like(device=object())
    sentry.audible = True
    assert sentry.tick(s, 0.0) is None
    s.error = "RuntimeError: boom"
    s.feeding = False
    problem = sentry.tick(s, 1.0)
    assert problem is not None and "the feed died" in problem and "boom" in problem
    assert s.beeps == ["beep"]


def test_a_dead_recording_is_an_alarm_and_drops_are_an_advisory():
    sentry, s = SignalSentry(), session_like()
    sentry.audible = True
    sentry.recorder = SimpleNamespace(error="OSError: disk full", closed=True, dropped=0)
    problem = sentry.tick(s, 0.0)
    assert problem is not None and "recording has stopped" in problem and "disk full" in problem
    assert s.beeps == ["beep"]

    sentry.recorder = SimpleNamespace(error="", closed=True, dropped=0)
    assert "still running" in sentry.tick(s, 1.0), "closing early is a failure even without an error"

    sentry.recorder = SimpleNamespace(error="", closed=False, dropped=7, active=True)
    assert sentry.tick(s, 2.0) is None, "a healthy recorder is no alarm"
    assert sentry.advice is not None and "dropped 7 frames" in sentry.advice

    sentry.recorder = None                              # the app disarmed it on purpose
    assert sentry.tick(s, 3.0) is None


def test_a_channel_delivering_nothing_but_artifact_becomes_audible_eventually():
    """The electrode that almost popped off, or lies against nothing: signal arrives, none of it is
    usable, and no absence detector will ever fire. Stricter and far slower than the advisory, so
    movement in bed has long cleared before it can sound."""
    sentry, s = SignalSentry(), session_like()
    sentry.audible = True
    s.rejections[0] = 0.9
    sentry.tick(s, 0.0)
    assert sentry.tick(s, GARBAGE_SECONDS - 1) is None, "ten minutes means ten minutes"
    assert sentry.advice is not None, "the advisory said so five minutes earlier"
    problem = sentry.tick(s, GARBAGE_SECONDS + 1)
    assert problem is not None and "nothing but artifact" in problem and "ch0" in problem
    assert s.beeps == ["beep"]
    s.rejections[0] = 0.1
    assert sentry.tick(s, GARBAGE_SECONDS + 2) is None


def test_the_alarm_runs_on_the_operators_numbers_not_the_modules():
    sentry, s = SignalSentry(), session_like()
    sentry.audible = True
    sentry.alarm_seconds, sentry.repeat_seconds, sentry.volume = 20.0, 10.0, 55.0
    s.state[0] = False
    sentry.tick(s, 0.0)
    assert sentry.tick(s, 19.0) is None, "the configured patience, not the default's"
    assert sentry.tick(s, 21.0) is not None
    sentry.tick(s, 26.0)                        # inside the configured repeat: quiet
    sentry.tick(s, 32.0)                        # past it: the second beep
    assert s.beeps == ["beep", "beep"]
    assert s.volumes == [55.0, 55.0], "the beep carries the configured volume"


def test_heavy_rejection_is_an_advisory_not_an_alarm():
    sentry, s = SignalSentry(), session_like()
    sentry.audible = True
    s.rejections[0] = 0.7                       # heavy, but under the audible garbage tier
    sentry.tick(s, 0.0)
    assert sentry.advice is None
    sentry.tick(s, ADVISORY_SECONDS + 1)
    assert sentry.advice is not None and "ch0" in sentry.advice
    assert sentry.problem is None and s.beeps == [], "degrading is shown, never sounded"
    s.rejections[0] = 0.1
    sentry.tick(s, ADVISORY_SECONDS + 2)
    assert sentry.advice is None
