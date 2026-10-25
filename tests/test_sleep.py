"""The sleep screen: a monitoring session whose product is the recording.

What is pinned here is the screen's three claims: it draws before and after data like every screen,
it trains nothing and records by default, and its health tally counts the night a channel actually
lost rather than only saying a channel is bad right now.
"""

from __future__ import annotations

from test_screens import StubPage, at_corner, pump, replay, toolbar

from nfb_console import sleep_view
from nfb_console.protocols import BY_CODE


def sleep_session(source=None, channels: int = 2) -> sleep_view.SleepSession:
    return sleep_view.session_for(BY_CODE["SLEEP"], source=source or replay(), channels=channels)


def test_the_sleep_screen_draws_before_any_data_arrives():
    session = sleep_session(source=at_corner(0.5, channels=2))
    try:
        page = StubPage()
        sleep_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        assert page.updates >= 1
    finally:
        session.stop()


def test_the_sleep_screen_draws_with_real_data_and_states_its_scales():
    session = sleep_session()
    try:
        pump(session)
        page = StubPage()
        control = sleep_view.view(page, session, toolbar(), lambda: None)
        session.push_display(1e9)               # a cascade row lands whatever the draw cadence was
        session.redraw()
        text = _all_text(control)
        assert "µV full scale" in text, "the cascade states its ceiling"
        assert "not sleep stages" in text or "waiting for data" in text
        assert "usable" in text or "waiting" in text, "the health tally has a row per channel"
        assert "gaps in acquisition" in text
    finally:
        session.stop()


def test_the_night_carries_a_quality_timeline_showing_where_it_went_bad():
    """The health tally says how much was lost; this says when. Drawn from the gate's own verdicts as
    they pass through the feed, so it costs one image rather than a shape per second."""
    session = sleep_session()
    try:
        pump(session, until=lambda: session.quality.seconds(0) >= 3.0)
        page = StubPage()
        control = sleep_view.view(page, session, toolbar(), lambda: None)
        session.redraw()
        text = _all_text(control)
        assert "one pixel =" in text and "worst" in text
        assert session.quality.image([0], width=200) is not None
        rejected, unusable, _missing = session.quality.totals(0)
        assert unusable > 0.9, "the bench capture is railed on every channel: unusable, not rejected"
        assert rejected == 0.0, "a condemned channel bypasses the gate rather than failing it"
    finally:
        session.stop()


def test_sleep_trains_nothing_records_by_default_and_is_loud_about_failures():
    session = sleep_session()
    try:
        assert session.records_by_default is True
        assert session.sentry.audible is True, "the audible tier is armed here and nowhere else"
        assert session.targets == [] and session.offers == ()
        assert all(not m.enabled for m in session.feedback.mechanisms.values()), \
            "no output can take the volume or the display from a sleeping operator"
    finally:
        session.stop()


def test_the_tally_counts_the_night_a_channel_lost():
    """Driven with hand-made healthy blocks: the bench fixture is railed on every channel by design,
    and this test needs a channel with nothing wrong to prove the tally does not blame it."""
    import numpy as np

    from nfb.stream import Block

    session = sleep_session(source=at_corner(0.5, channels=2))
    try:
        rate = session.rate
        times = np.arange(int(rate)) / rate
        counts = tuple(int(v) for v in 2000 * np.sin(2 * np.pi * 10.0 * times))
        step = len(counts) // 4

        def second(start: int) -> None:
            for channel, spec in session.specs.items():
                for i in range(0, len(counts), step):
                    session.feed_block(Block(channel, start + i, counts[i:i + step], spec))

        second(0)
        assert session.usable_fraction(0) == 1.0, "a healthy channel loses nothing"

        session.trust_channel(0, False)          # the operator declares it off the head
        second(len(counts))
        usable = session.usable_fraction(0)
        assert usable is not None and 0.0 < usable < 1.0, "time while unusable is counted as lost"
        assert session.usable_fraction(1) == 1.0, "and only on the channel that lost it"
        assert session.night_seconds == pytest_approx(2.0)
    finally:
        session.stop()


def pytest_approx(value):
    import pytest

    return pytest.approx(value, rel=0.05)


def test_the_alarm_settings_are_calibration_and_survive_a_stop():
    """The ALARM panel's numbers are the person's and the room's, so they ride the same channel every
    volume does: `calibration()` out, `restore_calibration()` in."""
    first = sleep_session()
    try:
        first.sentry.alarm_seconds = 120.0
        first.sentry.repeat_seconds = 60.0
        first.sentry.volume = 40.0
        first.sentry.audible = False
        stored = first.calibration()
    finally:
        first.stop()

    fresh = sleep_session()
    try:
        assert fresh.sentry.alarm_seconds != 120.0, "the default, before the restore"
        fresh.restore_calibration(stored)
        assert fresh.sentry.alarm_seconds == 120.0
        assert fresh.sentry.repeat_seconds == 60.0
        assert fresh.sentry.volume == 40.0
        assert fresh.sentry.audible is False
    finally:
        fresh.stop()


def _all_text(root) -> str:
    out: list[str] = []

    def walk(control) -> None:
        value = getattr(control, "value", None)
        if isinstance(value, str):
            out.append(value)
        content = getattr(control, "content", None)
        if content is not None:
            walk(content)
        for child in getattr(control, "controls", None) or ():
            walk(child)

    walk(root)
    return " · ".join(out)
