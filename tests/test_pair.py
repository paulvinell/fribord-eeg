"""The program comparing the Mendi's pulse with the HeartMath's: its session.

One synthetic heartbeat sequence is seen at a forehead and an ear sensor, with a known delay between them
and an ear pulse that shrinks halfway, and handed to the session through two runners that keep the
shared pulse form. Values pushed in, `now` passed explicitly; no device, no clock.
"""

from __future__ import annotations

import types
from collections import deque

import pytest

from nfb_console.pair_view import PairSession
from nfb_console.protocols import Protocol

from test_hrv_compare import FOREHEAD_AT, REFERENCE_AT, _scene

PAIR = Protocol("PAIR", "Pulse pair", "Forehead against ear, beat for beat")


class _Runner:
    """A borrowed runner as the program sees it: beats, a pulse pointing up, and a heart-rate state."""

    def __init__(self, beats, wave, rate=62.0, rmssd=45.0):
        self.beat_times = deque(beats)
        self._wave = wave
        self.heart_state = types.SimpleNamespace(hr_smoothed=rate, rmssd_ms=rmssd)
        self.error = None
        self.stale = False

    @property
    def waveform(self):
        return self._wave

    def describe(self):
        return "reading"


ON = types.SimpleNamespace(on=True)


def _session(**kw):
    _, forehead, reference, f_wave, r_wave, *_ = _scene(**kw)
    return PairSession(PAIR, heart_runner=_Runner(reference, r_wave), mendi_runner=_Runner(forehead, f_wave),
                       heart_choice=ON, mendi_choice=ON)


def test_the_last_minute_of_both_pulses_is_compared():
    session = _session()
    session.update(now=90.0)
    s = session.snapshot
    assert s.paired >= 60 and s.forehead_only <= 2 and s.ear_only <= 2
    assert s.agreement > 0.9, "the two sites see the same heart, beat for beat"
    assert s.delay_ms == pytest.approx((REFERENCE_AT - FOREHEAD_AT) * 1000, abs=15)
    assert s.forehead_rate == 62.0 and s.ear_rmssd == 45.0
    assert s.pulse_shape and len(s.pulse_shape) == len(s.pulse_lags)


def test_the_height_ratio_is_against_the_start_of_the_session():
    """The ear pulse shrinks to 60% after 45 s. The first minute sets 1; the last shows the shrink."""
    session = _session()
    session.update(now=46.0)
    session.update(now=90.0)
    first, last = session.ratio_series[0][1], session.ratio_series[-1][1]
    assert first == pytest.approx(1.0, abs=0.05)
    assert last == pytest.approx(0.6, abs=0.12)


def test_the_training_signal_follows_a_change_within_seconds():
    """Each heartbeat is measured once its pulses have arrived; the signal is the median of the last eight.
    So when the ear pulse shrinks at 45 s, the signal has moved most of the way ten seconds later, not a
    minute later as a one-minute median would."""
    session = _session()
    readings = {}
    for now in range(20, 70):
        session.update(now=float(now))
        readings[now] = session.training[1]
    assert readings[44] > 0.9, "before the change, about as at the start"
    assert readings[56] < 0.75, "ten seconds after it, most of the way down"


def test_each_heartbeat_is_measured_once():
    session = _session()
    for now in (60.0, 61.0, 61.0, 62.0):
        session.update(now=now)
    times = [t for t, _ in session.beat_ratios]
    assert times == sorted(set(times)), "a heartbeat seen in two comparisons is counted once"


def test_a_device_not_chosen_is_named_and_nothing_is_compared():
    session = _session()
    session.heart_choice = types.SimpleNamespace(on=False)
    assert session.missing() == ["no heart sensor: choose the HeartMath on the device screen"]
    session.update(now=90.0)
    assert session.snapshot is None


def test_a_device_gone_quiet_is_named():
    session = _session()
    session.mendi.stale = True
    assert session.missing() == ["headband: no readings for a while"]


def test_a_heart_sensor_without_a_waveform_still_compares_beats():
    """Over Bluetooth the HeartMath sends beats only: pairing and delay work, the height ratio does not."""
    session = _session()
    session.heart._wave = None
    session.update(now=90.0)
    assert session.snapshot.paired >= 60 and session.snapshot.height_ratio is None


def test_running_twice_starts_one_comparison_and_stopping_ends_it():
    session = _session()
    session.run()
    worker = session._worker
    session.run()
    assert session._worker is worker, "resume calls run again; one comparison thread is enough"
    session.stop()
    worker.join(timeout=1.0)
    assert not worker.is_alive()


def test_a_fresh_session_keeps_the_plan_and_nothing_else():
    old = _session()
    old.update(now=90.0)
    new = _session()
    new.adopt(old)
    assert new.plan is old.plan and new.snapshot is None and not new.delay_series


# --- the view ---------------------------------------------------------------------------------

def _drawn(session):
    from nfb_console import pair_view
    from test_screens import StubPage, toolbar
    from test_ste import _texts

    page = StubPage()
    control = pair_view.view(page, session, toolbar(), lambda: None)
    session.redraw()
    return " | ".join(_texts(control))


def test_with_no_devices_the_program_says_what_to_choose():
    from nfb_console import pair_view
    from nfb_console.protocols import BY_CODE

    texts = _drawn(pair_view.session_for(BY_CODE["PAIR"], channels=0))
    assert "no headband: choose the Mendi headband on the device screen" in texts
    assert "no heart sensor: choose the HeartMath on the device screen" in texts
    assert "waiting for beats" in texts


def test_fed_the_program_shows_both_sensors_and_how_they_compare():
    session = _session()
    session.update(now=46.0)
    session.update(now=90.0)
    texts = _drawn(session)
    for shown in ("INTERVALS", "AVERAGED PULSE", "OVER THE SESSION", "agreement", "ear pulse / forehead pulse", "62"):
        assert shown in texts
    assert "blue = forehead, grey = ear" in texts
    assert "waiting for beats" not in texts
    assert "height ratio" not in texts, "the number is named by what it divides"
    assert "scaled by" in texts and "recent range" in texts, "the shared scaling, on the Feedback tab"


def test_the_height_ratio_axis_stops_at_zero():
    """The axis is centred on the start, so a ratio that climbed to 2.2 drew down to -1.0: a height ratio
    that cannot exist."""
    session = _session()
    session.update(now=90.0)
    session.ratio_series.clear()
    session.ratio_series.extend((float(i), 1.0 + 1.2 * i / 480) for i in range(0, 481, 5))
    texts = _drawn(session).split(" | ")
    assert not any(text.strip().startswith("-") for text in texts), "no negative ratio on the axis"


def test_a_comparison_that_fails_is_said_and_the_next_one_still_runs():
    """A fault inside one comparison once killed the comparison's thread for the rest of the session."""
    session = _session()

    class Broken(_Runner):
        @property
        def waveform(self):
            raise RuntimeError("deque mutated during iteration")

    good = session.heart
    session.heart = Broken(list(good.beat_times), good.waveform)
    session.paused = False
    session._wake.set()                 # every wait returns at once, so the loop runs back to back
    session.run()
    from conftest import waitfor
    waitfor(lambda: session.fault is not None, what="the failure is recorded")
    session.heart = good
    waitfor(lambda: session.fault is None, what="the next comparison runs and clears it")
    session.stop()


# --- the training -------------------------------------------------------------------------------

def _spied(session):
    calls = {"volume": [], "music": [], "released": 0}
    session.output_volume.apply = lambda *a, **k: calls["volume"].append(a)
    session.music.apply = lambda *a, **k: calls["music"].append(k.get("level"))
    session.output_volume.release = lambda: calls.__setitem__("released", calls["released"] + 1)
    session.music.release = lambda: None
    session.display.apply = lambda *a, **k: calls.setdefault("shadow", []).append(a)
    session.display.release = lambda: None
    session.cue = lambda *a, **k: True
    return calls


def test_the_level_comes_from_the_scaling_chosen_on_the_feedback_tab():
    """The program once had a scale of its own, half at the start and full 30% above it, and none of the
    shared scalings. A ratio that drifted up held that scale at full for minutes."""
    session = _session()
    assert session.strategy_id == "rolling_minmax", "the shared default"
    session.set_strategy("fixed_range", low=0.5, high=1.5)
    _spied(session)
    session.toggle_pause()
    session.update(now=40.0)
    session.observe()
    assert session.signal == pytest.approx(session.training[1] - 0.5)


def test_every_shared_mechanism_is_offered_and_the_shadow_follows_the_level():
    session = _session()
    assert set(session.offers) == {"sound", "shadow", "click", "music"}
    calls = _spied(session)
    session.feedback.shadow.enabled = True
    session.toggle_pause()
    session.update(now=40.0)
    session.observe()
    assert calls["shadow"], "the screen is driven with the other outputs"


def test_each_new_reading_drives_the_outputs_once():
    session = _session()
    calls = _spied(session)
    session.toggle_pause()
    session.update(now=40.0)
    session.observe()
    session.observe()                   # no new reading since: nothing is driven again
    assert len(calls["volume"]) == 1 and len(calls["music"]) == 1


def test_a_shrinking_ear_pulse_lowers_the_level():
    session = _session()
    _spied(session)
    session.toggle_pause()
    session.feedback.smoothing_seconds = 0.0
    levels = {}
    for now in range(30, 70):
        session.update(now=float(now))
        session.observe()
        levels[now] = session.level
    assert levels[44] > 0.4 and levels[66] < 0.2


def test_pause_suspend_and_stop_let_go_of_the_outputs():
    session = _session()
    calls = _spied(session)
    session.toggle_pause()
    session.update(now=40.0)
    session.observe()
    session.toggle_pause()
    assert calls["released"] >= 1 and session.signal is None
    session.suspend()
    session.stop()
    assert calls["released"] >= 3


def test_a_sensor_dropping_out_silences_the_feedback():
    session = _session()
    calls = _spied(session)
    session.toggle_pause()
    session.update(now=40.0)
    session.observe()
    session.mendi.stale = True
    session.observe()
    assert session.signal is None and calls["released"] >= 1


def test_settings_survive_a_stop():
    old = _session()
    old.set_strategy("zscore", window_s=90.0)
    old.feedback.click.enabled = False
    new = _session()
    new.adopt(old)
    assert new.strategy_id == "zscore" and new.strategy.window_s == 90.0
    assert not new.feedback.click.enabled
    assert new.restore_calibration(old.calibration()) > 0
