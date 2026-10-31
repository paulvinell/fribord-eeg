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
    assert first == pytest.approx(1.0)
    assert last == pytest.approx(0.6, abs=0.12)


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
    for shown in ("INTERVALS", "AVERAGED PULSE", "OVER THE SESSION", "agreement", "height ratio", "62"):
        assert shown in texts
    assert "blue = forehead, grey = ear" in texts
    assert "waiting for beats" not in texts


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
