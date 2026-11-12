"""The spindle card on the amplitude program: counted on the feed, heard as its own cue, searched for, and
carried across a stop. Driven block by block through `feed_block`, on spindles planted at known times."""

from __future__ import annotations

import numpy as np
import pytest

from nfb.dsp import Planted, pink, plant
from nfb.stream import Block
from nfb_console import amplitude_view as av
from nfb_console import spindle_card as sc
from nfb_console.protocols import BY_CODE

from conftest import counts, plausible_eeg, waitfor
from test_layout import RULES, walk
from test_screens import StubPage, replay, toolbar
from test_ste import _help_faults, faults


def planted(seconds, rate, frequency=25.0, seed=1, per_minute=8):
    rng = np.random.default_rng(seed)
    x = pink(int(seconds * rate), rate, rng)
    plant(x, rate, rng, Planted(frequency, (0.25, 0.4), 14.0, per_minute=per_minute, jitter_hz=2.0))
    return x


def feed(session, waves, start=0):
    """Every channel's wave through the feed loop's body, a second at a time. Returns the next index."""
    n = min(len(w) for w in waves.values())
    step = int(session.rate)
    for i in range(0, n - step + 1, step):
        for ch, spec in session.specs.items():
            session.feed_block(Block(ch, start + i, counts(waves[ch][i:i + step], spec.scale), spec))
    return start + n - n % step


def make(spindles=True):
    s = av.session_for(BY_CODE["AMP"], source=replay(), channels=2)
    s.paused = False
    s.state = dict.fromkeys(s.specs, True)
    s.played = []
    s.cue = lambda name, volume=None: s.played.append((name, volume))
    if spindles:
        s.add_spindles(0)
    return s


@pytest.fixture
def session():
    s = make()
    yield s
    s.stop()


def test_a_spindle_is_counted_and_heard_as_its_own_cue(session):
    rate = session.rate
    feed(session, {0: planted(120, rate), 1: plausible_eeg(120, rate, 1)})
    watch = session.spindles
    assert watch.count >= 8
    assert watch.per_minute == pytest.approx(len(watch._times), rel=0.01)
    heard = [p for p in session.played if p[0] == sc.sound()]
    assert len(heard) == watch.count
    assert all(volume == session.feedback.click.volume for _, volume in heard)


def test_the_spindles_change_nothing_the_targets_drive():
    rate = 250.0
    waves = {0: planted(90, rate), 1: plausible_eeg(90, rate, 1)}
    plain, watched = make(spindles=False), make()
    try:
        feed(plain, waves)
        feed(watched, waves)
        assert watched.spindles.count > 0
        assert watched._level == plain._level
        assert [t.threshold for t in watched.targets] == [t.threshold for t in plain.targets]
        assert watched.outputs == plain.outputs
    finally:
        plain.stop()
        watched.stop()


def test_a_paused_session_counts_nothing_and_stays_silent(session):
    session.paused = True
    feed(session, {0: planted(60, session.rate), 1: plausible_eeg(60, session.rate, 1)})
    assert session.spindles.count == 0 and not session.played
    assert session.spindles.per_minute is None


def test_another_channel_starts_the_count_again(session):
    rate = session.rate
    at = feed(session, {0: planted(60, rate), 1: plausible_eeg(60, rate, 1)})
    assert session.spindles.count > 0
    session.spindles.channel = 1
    feed(session, {0: plausible_eeg(5, rate, 0), 1: plausible_eeg(5, rate, 1)}, at)
    assert session.spindles.count == 0 and 0 < session.spindles.seconds <= 5.0


def test_the_search_has_a_frequency_from_the_start_and_on_stop_sets_it(session):
    rate = session.rate
    watch = session.spindles
    watch.frequency_hz = 24.0
    watch.start_search()
    assert sc.search_line(watch) == "searching 0 s · 24.0 Hz from 0 spindles"
    at = feed(session, {0: planted(10, rate, 22.0, seed=3), 1: plausible_eeg(10, rate, 1)})
    early = watch.estimate()
    assert sc.search_line(watch) == (f"searching 10 s · {early.frequency_hz:.1f} Hz from "
                                     f"{early.spindles} spindle{'' if early.spindles == 1 else 's'}")
    feed(session, {0: planted(140, rate, 22.0, seed=4, per_minute=10),
                   1: plausible_eeg(140, rate, 1)}, at)
    found = watch.estimate()
    assert found.frequency_hz == pytest.approx(22.0, abs=0.7) and found.spindles >= 8
    watch.stop_search()
    assert not watch.searching
    assert watch.frequency_hz == round(found.frequency_hz, 1)
    assert sc.search_line(watch).startswith(f"found {found.frequency_hz:.1f} Hz")


def test_a_search_stopped_after_one_second_still_sets_the_frequency(session):
    watch = session.spindles
    watch.start_search()
    feed(session, {0: planted(1, session.rate, 30.0), 1: plausible_eeg(1, session.rate, 1)})
    shown = watch.estimate().frequency_hz
    watch.stop_search()
    assert watch.frequency_hz == round(shown, 1)


def test_an_estimate_runs_on_its_own_thread_from_the_first_moment(session):
    watch = session.spindles
    watch.start_search()
    watch.estimate_due(0.0)
    first = watch._worker
    assert first is not None
    waitfor(lambda: not first.is_alive(), what="the search's first estimate")
    watch.estimate_due(0.2)
    assert watch._worker is first                 # not due again for ESTIMATE_SECONDS
    watch.estimate_due(0.6)
    assert watch._worker is not first


def test_a_stop_carries_the_settings_and_not_the_counts(session):
    feed(session, {0: planted(60, session.rate), 1: plausible_eeg(60, session.rate, 1)})
    watch = session.spindles
    watch.channel, watch.frequency_hz, watch.reject_muscle, watch.sound = 1, 21.5, True, False
    watch.level = 5.5
    watch.start_search()
    fresh = make(spindles=False)
    try:
        fresh.adopt(session)
        carried = fresh.spindles
        assert carried.settings() == watch.settings() and carried.level == 5.5
        assert carried.count == 0 and not carried.searching
    finally:
        fresh.stop()


def test_a_new_level_keeps_the_detector_and_the_count(session):
    rate = session.rate
    at = feed(session, {0: planted(60, rate), 1: plausible_eeg(60, rate, 1)})
    watch = session.spindles
    detector, count = watch._detector, watch.count
    assert count > 0 and watch.level_uv == pytest.approx(4.0 * watch.usual)
    watch.level = 3.0
    assert watch.level_uv == pytest.approx(3.0 * watch.usual)     # before the feed applies it
    feed(session, {0: planted(30, rate, seed=2), 1: plausible_eeg(30, rate, 1)}, at)
    assert watch._detector is detector and detector.level == 3.0
    assert watch.count >= count


# --- the card on the screen -------------------------------------------------------------------

def test_minus_and_plus_move_the_level_within_its_range():
    session = make()
    try:
        control = av.view(StubPage(), session, toolbar(), lambda: None)
        watch = session.spindles

        def press(glyph, times=1):
            for _ in range(times):
                row = next(r for r in (c for c in walk(control) if getattr(c, "controls", None))
                           if any(getattr(k, "value", None) == f"{watch.level:.1f} ×" for k in r.controls))
                next(k for k in row.controls
                     if getattr(getattr(k, "content", None), "value", None) == glyph).on_click(None)

        press("+")
        assert watch.level == 4.5
        press("−", 2)
        assert watch.level == 3.5
        press("−", 10)
        assert watch.level == sc.LEVELS[0]
        press("+", 30)
        assert watch.level == sc.LEVELS[1]
        assert any(getattr(c, "value", None) == f"{sc.LEVELS[1]:.1f} ×" for c in walk(control))
    finally:
        session.stop()


def test_the_level_line_says_the_threshold_in_microvolts_from_the_first_sample(session):
    watch = session.spindles
    assert "not known yet" in sc.level_line(watch)
    rate = session.rate
    session.feed_block(Block(0, 0, counts(planted(1, rate)[:250], session.specs[0].scale), session.specs[0]))
    assert watch.usual is not None
    assert sc.level_line(watch) == (f"threshold now {4.0 * watch.usual:.1f} µV: 4.0 × the usual "
                                    f"{watch.usual:.1f} µV")


@pytest.mark.parametrize("state", ["cold", "fed", "searching", "muscle"])
def test_the_card_passes_the_screen_rules(state):
    session = make()
    page = StubPage()
    try:
        control = av.view(page, session, toolbar(), lambda: None)
        if state == "muscle":
            session.spindles.reject_muscle = True
        if state == "searching":
            session.spindles.start_search()
        if state != "cold":
            feed(session, {0: planted(70, session.rate), 1: plausible_eeg(70, session.rate, 1)})
        if state == "searching":
            session.spindles.estimate()
        session.redraw()
        texts = [c.value for c in walk(control) if hasattr(c, "value") and isinstance(c.value, str)]
        assert ("stop search" in texts) == (state == "searching")
        found = [f"{rule.__name__}: {fault}" for rule in RULES for fault in rule(control)]
        found += [fault for text in dict.fromkeys(texts) for fault in faults(text)]
        found += _help_faults(page, control)
        assert not found, "\n".join(found)
    finally:
        session.stop()


def test_the_add_button_goes_while_the_card_is_on_the_list():
    session = make(spindles=False)
    try:
        control = av.view(StubPage(), session, toolbar(), lambda: None)
        button = next(c for c in walk(control)
                      if getattr(getattr(c, "content", None), "value", None) == "+ spindle")
        assert button.visible is not False
        button.on_click(None)
        assert session.spindles is not None and button.visible is False
    finally:
        session.stop()


def test_the_remove_button_drops_the_card_and_brings_back_the_add_button():
    session = make()
    try:
        control = av.view(StubPage(), session, toolbar(), lambda: None)
        chip = next(c for c in walk(control) if getattr(c, "value", None) == "SPINDLE")
        row = next(c for c in walk(control) if chip in (getattr(c, "controls", None) or [])
                   or any(getattr(k, "content", None) is chip for k in (getattr(c, "controls", None) or [])))
        remove = next(c for c in row.controls if getattr(getattr(c, "content", None), "value", None) == "×")
        remove.on_click(None)
        assert session.spindles is None
        assert not any(getattr(c, "value", None) == "SPINDLE" for c in walk(control))
        button = next(c for c in walk(control)
                      if getattr(getattr(c, "content", None), "value", None) == "+ spindle")
        assert button.visible is not False
        feed(session, {0: planted(60, session.rate), 1: plausible_eeg(60, session.rate, 1)})
        assert not [p for p in session.played if p[0] == sc.sound()]
    finally:
        session.stop()
