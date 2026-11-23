"""The subtraction of eye movements on the SCP screen, driven through `feed_block` on artificial signals."""

from __future__ import annotations

import dataclasses

import numpy as np

from nfb_console import eye_calibration
from nfb_console.trials import ACTIVE, EXTRA

from conftest import wandering
from test_eye_calibration import eye_channels
from test_scp_view import RATE, _feed, _run, _session

CAL_S = eye_calibration.PLANS[eye_calibration.SHORT].total_s + 2.0
EYE_UV = 100.0
ROT = [(0.10, 0.03), (1.0, 0.3)]
BLINK = [0.2, 2.0]                     # the same proportion as the rotations, so one weight serves both


def _calibrated(session, rot=ROT, blink=BLINK) -> int:
    """The calibration fed from sample 0 to its end. Returns the next sample index."""
    session.toggle_pause()
    session.set_extra(1, True)
    session.start_calibration()
    channels = eye_channels(rot, blink, seconds=CAL_S, eye_uv=EYE_UV)
    return _feed(session, {0: channels[0], 1: channels[1]}, until=lambda s: not s.eye_cal.running)


def _looks_up_in_feedback(session, ch) -> float:
    """An eye that looks up through every feedback window, whatever the cue asks."""
    run = session.trials
    return EYE_UV * ROT[ch][0] if run.phase == ACTIVE else 0.0


def _trials(session, start: int, add, seconds: float = 100.0, extra=None) -> None:
    n = start + int(seconds * RATE)
    waves = {0: wandering(1.5, n / RATE + 1, RATE, 21, 0.5), 1: wandering(1.5, n / RATE + 1, RATE, 22, 0.5)}
    if extra is not None:
        waves[1] = extra(waves[1])
    _run(session, n_trials=5, transfer=0.0, seed=3)
    _feed(session, waves, start=start, add=add, until=lambda s: s.trials.done)


def test_the_calibration_through_feed_block_finds_the_weight_and_subtracts():
    session = _session()
    try:
        _calibrated(session)
        cal = session.eye_cal
        assert cal.done, cal.stopped
        assert session.subtracting and session.can_subtract
        assert abs(session.eye_fit.weights.rotation[0] - 0.1) < 0.015, session.eye_fit.weights
        assert session.trials.results == [], "a new run starts after the calibration"
    finally:
        session.stop()


def test_an_eye_movement_in_the_feedback_moves_the_plain_shift_and_not_the_corrected_one():
    scores = {}
    for subtract in (True, False):
        session = _session()
        try:
            at = _calibrated(session)
            session.set_subtract(subtract)
            assert session.subtracting == subtract
            _trials(session, at, _looks_up_in_feedback)
            results = session.trials.results
            assert len(results) == 5 and all(r.valid for r in results), results
            scores[subtract] = float(np.mean([r.score_uv for r in results]))
        finally:
            session.stop()
    assert scores[False] > 7.0, scores
    assert abs(scores[True]) < 2.0, scores


def test_an_extra_at_the_input_limit_with_the_subtraction_on_voids_the_trial_and_names_the_channel():
    session = _session()
    try:
        at = _calibrated(session)
        _trials(session, at, None, extra=lambda w: np.full(len(w), -800.0))
        last = session.trials.results[-1]
        assert not last.valid and last.cause == EXTRA, last
        what, action = session.void_words(last)
        assert "ch1" in what and "ch1" in action and "subtract" in action, (what, action)
    finally:
        session.stop()


def test_without_the_subtraction_the_same_extra_at_the_limit_voids_nothing():
    session = _session()
    try:
        at = _calibrated(session)
        session.set_subtract(False)
        _trials(session, at, None, extra=lambda w: np.full(len(w), -800.0))
        assert all(r.valid for r in session.trials.results)
    finally:
        session.stop()


def test_a_pause_stops_the_calibration_and_keeps_nothing():
    session = _session()
    try:
        session.toggle_pause()
        session.set_extra(1, True)
        session.start_calibration()
        channels = eye_channels(ROT, BLINK, seconds=CAL_S)
        _feed(session, {0: channels[0][:int(20 * RATE)], 1: channels[1][:int(20 * RATE)]})
        session.toggle_pause()
        assert session.eye_cal.stopped == eye_calibration.PAUSED
        assert session.eye_fit is None and not session.subtracting
    finally:
        session.stop()


def test_a_gap_stops_the_calibration():
    session = _session()
    try:
        session.toggle_pause()
        session.set_extra(1, True)
        session.start_calibration()
        channels = eye_channels(ROT, BLINK, seconds=CAL_S)
        at = _feed(session, {0: channels[0][:int(20 * RATE)], 1: channels[1][:int(20 * RATE)]})
        _feed(session, {0: channels[0][:int(30 * RATE)], 1: channels[1][:int(30 * RATE)]}, start=at + int(2 * RATE))
        assert session.eye_cal.stopped == eye_calibration.GAP
        assert session.eye_fit is None
    finally:
        session.stop()


def test_a_change_of_the_channels_drops_the_weights():
    session = _session(channels=3)
    try:
        _calibrated(session)
        assert session.subtracting
        session.set_extra(2, True)
        assert session.eye_fit is None and not session.subtracting
    finally:
        session.stop()
    session = _session()
    try:
        _calibrated(session)
        session.set_channel(1)
        assert session.eye_fit is None and not session.subtracting
    finally:
        session.stop()


# --- the screen -------------------------------------------------------------------------------------

def _press(control, label: str) -> None:
    """Press the one visible control named `label`."""
    import flet as ft

    found = []

    def walk(c):
        if c is None or getattr(c, "visible", True) is False:
            return
        content = getattr(c, "content", None)
        if getattr(c, "on_click", None) is not None and isinstance(content, ft.Text) and content.value == label:
            found.append(c.on_click)
        for attr in ("content", "controls"):
            value = getattr(c, attr, None)
            for item in value if isinstance(value, list) else [value]:
                walk(item)
    walk(control)
    assert len(found) == 1, f"{len(found)} visible controls named {label!r}"
    found[0](None)


def _shown(session, control) -> list[str]:
    from harness import _texts

    session.redraw()
    return [c.value for c in _texts(control) if c.value and c.visible is not False]


def _visible_texts(control, out=None, hidden=False) -> list[str]:
    """The texts a person sees: none under a control whose `visible` is False."""
    import flet as ft

    out = [] if out is None else out
    if control is None or hidden or getattr(control, "visible", True) is False:
        return out
    if isinstance(control, ft.Text) and control.value:
        out.append(control.value)
    for attr in ("content", "controls"):
        value = getattr(control, attr, None)
        for item in value if isinstance(value, list) else [value]:
            _visible_texts(item, out)
    return out


def test_the_button_shows_with_a_lit_extra_and_a_press_covers_the_window_with_the_dot():
    from nfb_console import scp_view
    from test_scp_view import _screen

    session, control = _screen()
    try:
        session.redraw()
        assert "calibrate eyes" not in _visible_texts(control)
        session.set_extra(1, True)
        session.redraw()
        assert "calibrate eyes" in _visible_texts(control)
        _press(control, "calibrate eyes")
        session.redraw()
        overlay = control.controls[1]
        assert overlay.visible
        assert eye_calibration.LINES[eye_calibration.ROTATIONS] in _visible_texts(overlay)
        _press(overlay, "stop")
        session.redraw()
        assert not overlay.visible and session.eye_cal is None and session.eye_fit is None
        assert scp_view.EyeOverlay
    finally:
        session.stop()


def test_after_a_calibration_the_result_shows_until_closed_and_the_pill_switches_the_subtraction():
    from test_scp_view import _screen

    session, control = _screen()
    try:
        session.toggle_pause()                  # `_calibrated` starts it again
        _calibrated(session)
        session.redraw()
        overlay = control.controls[1]
        seen = _visible_texts(overlay)
        for word in ("up-down", "left-right", "blinks", "rest", "without", "with"):
            assert word in seen, (word, seen)
        assert "The program now subtracts the eye movements and the blinks." in seen, seen
        _press(overlay, "close")
        session.redraw()
        assert not overlay.visible
        assert "subtract" in _visible_texts(control.controls[0])
        _press(control, "subtract")
        assert not session.subtracting
        _press(control, "subtract")
        assert session.subtracting
    finally:
        session.stop()


def test_a_press_on_short_or_long_sets_the_next_calibration_and_starts_nothing():
    from test_scp_view import _screen

    session, control = _screen()
    try:
        session.set_extra(1, True)
        session.redraw()
        shown = _visible_texts(control)
        assert "short" in shown and "long" in shown and session.eye_length == "short"
        _press(control, "long")
        assert session.eye_length == "long" and session.eye_cal is None, "a press starts nothing"
        _press(control, "calibrate eyes")
        assert session.eye_cal.plan == eye_calibration.PLANS["long"]
        session.close_calibration()
        session.set_extra(1, False)
        session.redraw()
        assert "long" not in _visible_texts(control), "the pills show only with the button"
    finally:
        session.stop()


def test_a_press_on_short_keeps_the_calibration_that_exists():
    session = _session()
    try:
        _calibrated(session)
        fit = session.eye_fit
        session.set_eye_length("long")
        assert session.eye_fit is fit and session.subtracting
    finally:
        session.stop()


def test_the_new_texts_and_the_help_are_simplified_technical_english():
    """The card, the result, the three lines and the stop lines, and every help with the result in it. A
    bold label in the help is one the screen shows."""
    from nfb_console import scp_view
    from test_scp_view import _page, _toolbar
    from test_ste import _help_faults, faults

    page, session = _page(), _session()
    try:
        control = scp_view.view(page, session, _toolbar(), lambda: None)
        _calibrated(session)
        session.close_calibration()
        session.redraw()
        found = [f for text in dict.fromkeys(_visible_texts(control)) for f in faults(text)]
        found += [f for line in eye_calibration.LINES.values() for f in faults(line)]
        for reason in (eye_calibration.GAP, eye_calibration.PAUSED, eye_calibration.BACK, "ch1 has no signal"):
            found += faults(f"the calibration stops · {reason} · nothing is kept")
        for rotations in (True, False):
            for blinks in (True, False):
                fit = session.eye_fit.__class__(session.eye_fit.weights, session.eye_fit.sizes, rotations, blinks)
                found += faults(scp_view.eye_sentence(fit))
                for left in (1, 3):
                    said = scp_view.eye_sentence(dataclasses.replace(fit, blinks_left_out=left))
                    found += faults(said)
                    assert f"leaves out {left} blink" in said, said
                assert "leaves out" not in scp_view.eye_sentence(fit), "nothing said for none left out"
        cal = eye_calibration.EyeCalibration(RATE, 0, (1,))
        for reason in ("ch1 is at the input limit", "ch1 has no signal"):
            for begun in (False, True):
                cal.held, cal.begun = reason, begun
                found += faults(scp_view.held_line(cal, session))
        found += _help_faults(page, control)
        assert not found, "\n".join(found)
        assert "The last calibration" in scp_view.signal_help(session)
    finally:
        session.stop()


# --- a channel at the input limit ---------------------------------------------------------------------

def test_a_press_with_a_channel_at_the_limit_shows_the_level_plot_and_start_goes_at_once():
    from test_scp_view import _rail, _screen

    session, control = _screen()
    try:
        session.set_extra(1, True)
        _feed(session, {0: wandering(1.5, 4.0, RATE, 3, 2.0)[:int(2 * RATE)], 1: _rail(2.0)})
        session.redraw()
        _press(control, "calibrate eyes")
        session.redraw()
        overlay = control.controls[1]
        seen = _visible_texts(overlay)
        assert session.eye_cal.held == "ch1 is at the input limit"
        assert any("ch1 is at the input limit" in text and "start" in text for text in seen), seen
        assert "ch0" in seen and "ch1" in seen, "the level plot names each channel of the calibration"
        assert eye_calibration.LINES[eye_calibration.ROTATIONS] not in seen, "no dot while it waits"
        _press(overlay, "start")
        session.redraw()
        assert session.eye_cal.held is None
        assert eye_calibration.LINES[eye_calibration.ROTATIONS] in _visible_texts(overlay)
    finally:
        session.stop()


def test_a_channel_at_the_limit_in_the_middle_stops_the_dot_and_continue_goes_on():
    from test_scp_view import _screen

    session, control = _screen()
    try:
        session.set_extra(1, True)
        session.redraw()
        _press(control, "calibrate eyes")
        channels = eye_channels(ROT, BLINK, seconds=CAL_S, eye_uv=EYE_UV)
        n = int(10 * RATE)
        x1 = channels[1].copy()
        x1[n - 100:n] = -800.0
        _feed(session, {0: channels[0][:n], 1: x1[:n]})
        session.redraw()
        overlay = control.controls[1]
        seen = _visible_texts(overlay)
        assert session.eye_cal.held and session.eye_cal.running
        assert not any("nothing is kept" in text for text in seen), seen
        assert any("continue" in text for text in seen), seen
        _press(overlay, "continue")
        assert session.eye_cal.held is None and session.eye_cal.running
    finally:
        session.stop()
