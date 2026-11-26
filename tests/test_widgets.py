"""The controls every screen is built from, held to what pressing them means.

A number box commits what was typed. A click into the box and a click away types nothing, so it commits
nothing: the box was built from a value that may have moved since, and writing that value back as a
choice would turn an automatic setting into a fixed one, or start a run again, with no edit at all.
"""

from __future__ import annotations

import importlib

import flet as ft
import pytest

from nfb_console.protocols import BY_CODE
from nfb_console.widgets import num_box

from test_layout import walk
from test_screens import StubPage, replay, toolbar


def _field(box) -> ft.TextField:
    return next(c for c in walk(box) if isinstance(c, ft.TextField))


def test_blur_without_edit_calls_nothing():
    calls: list[float] = []
    field = _field(num_box(5.0, calls.append, step=0.5, low=0.0, high=10.0))
    field.on_blur(None)
    field.on_submit(None)
    assert calls == [], "a box that nobody typed in commits nothing"
    field.value = "7"
    field.on_blur(None)
    assert calls == [7.0], "an edited box commits what was typed"
    field.on_blur(None)
    assert calls == [7.0], "and only once: the text has not changed since"


def test_a_typed_number_is_kept_and_the_arrows_go_to_the_next_multiple_of_the_step():
    """Typing sets the number that was typed. An arrow moves one step from a multiple of the step, and from
    a number between two multiples to the nearer one in its direction."""
    calls: list[float] = []
    box = num_box(5.0, calls.append, step=0.5, low=0.0, high=10.0, places=2)
    field = _field(box)
    up, down = (next(c for c in walk(box) if isinstance(c, ft.Container) and isinstance(c.content, ft.Text)
                     and c.content.value == glyph) for glyph in "▲▼")
    up.on_click(None)
    assert calls[-1] == 5.5
    field.value = "7.3"
    field.on_submit(None)
    assert calls[-1] == 7.3 and float(field.value) == 7.3
    up.on_click(None)
    assert calls[-1] == 7.5
    field.value = "7.3"
    field.on_submit(None)
    down.on_click(None)
    assert calls[-1] == 7.0
    down.on_click(None)
    assert calls[-1] == 6.5


def test_no_number_box_on_a_trial_screen_commits_an_unedited_text():
    """Every box on the three trial screens, each blurred with no edit: the run is the same run and every
    setting has the value it had."""
    from nfb_console.trial_training import TRIAL_SETTINGS

    for module, code, channels in (("scp_view", "SCP", 2), ("pupil_view", "PUPIL", 0),
                                   ("heg_view", "HEG", 0)):
        screen = importlib.import_module(f"nfb_console.{module}")
        session = screen.session_for(BY_CODE[code], source=replay(), channels=channels)
        try:
            session.set_mode("active")
            control = screen.view(StubPage(), session, toolbar(), lambda: None)
            session.redraw()
            run = session.trials
            before = {name: getattr(session, name) for name in TRIAL_SETTINGS}
            fields = [c for c in walk(control) if isinstance(c, ft.TextField)]
            assert len(fields) >= 5, f"{code}: the trial rows have their boxes"
            for field in fields:
                field.on_blur(None)
            assert session.trials is run, f"{code}: a blur with no edit started the run again"
            after = {name: getattr(session, name) for name in TRIAL_SETTINGS}
            assert after == before, f"{code}: a blur with no edit changed a setting"
        finally:
            session.stop()


def test_a_box_with_an_edit_in_progress_keeps_the_edit():
    from nfb_console.widgets import num_box

    box = num_box(1.0, lambda _v: None, step=0.1)
    field, show = box.controls[0], box.data
    show(2.0)
    assert field.value == "2.0", "a value that moves on its own reaches an untouched box"
    field.value = "7"
    show(3.0)
    assert field.value == "7", "the next frame does not write over a value being typed"


def test_a_number_field_clamps_and_survives_its_own_rebuild():
    """Two failures this covers. Typing a value past a bound used to be rejected outright, and the
    field's trailing `update()` raised out of the blur handler once `on_set` had rebuilt the panel the
    field lived in — "Control must be added to the page first"."""
    from nfb_console.amplitude_view import num_box

    seen: list[float] = []
    # This field is never added to a page, which is exactly the state a rebuilt panel leaves the old
    # field in — so every commit below exercises the guard that used to raise from `field.update()`.
    row = num_box(5.0, seen.append, step=0.1, low=1.0, high=10.0)
    field, stepper = row.controls[0], row.controls[1]
    up, down = stepper.controls

    assert field.value == "5.0", "formatted to the step's precision"
    up.on_click(None)
    assert seen[-1] == pytest.approx(5.1)
    up.on_click(None)
    assert seen[-1] == pytest.approx(5.2), "the arrows step from what is shown, not from the initial"
    down.on_click(None)
    assert seen[-1] == pytest.approx(5.1)

    field.value = "99"
    field.on_blur(None)
    assert seen[-1] == 10.0, "clamped to the bound rather than refused"
    field.value = "-4"
    field.on_blur(None)
    assert seen[-1] == 1.0

    assert field.value == "1.0", "the box shows the value that was accepted, not what was typed"

    field.value = "not a number"
    field.on_blur(None)
    assert seen[-1] == 1.0, "unparseable input snaps back to the last accepted value"
    assert field.value == "1.0"
