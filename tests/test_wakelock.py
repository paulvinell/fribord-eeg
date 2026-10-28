"""The display-sleep hold: keep the subject's screen awake through a still session.

The properties that matter are the safety rails, as with the shadow: the hold must be released, must be
idempotent so every start path can call it, and must stay off under `NFB_NO_OUTPUTS` so a test run cannot
pin the operator's display awake. A fake stands in for the real IOKit call so the suite never touches it.
"""

from __future__ import annotations

import pytest

from nfb_console import wakelock


@pytest.fixture
def fake(monkeypatch):
    """A stand-in for the IOKit assertion that records creates and releases."""
    state = {"created": [], "released": [], "next_id": 100}

    def fake_create(assertion_type, name):
        state["next_id"] += 1
        state["created"].append((assertion_type, name, state["next_id"]))
        return state["next_id"]

    def fake_release(assertion_id):
        state["released"].append(assertion_id)

    monkeypatch.setattr(wakelock, "available", lambda: True)
    monkeypatch.setattr(wakelock, "_create", fake_create)
    monkeypatch.setattr(wakelock, "_release", fake_release)
    monkeypatch.setattr(wakelock, "_held_id", None)
    monkeypatch.setattr(wakelock, "_registered", True)      # do not touch the real atexit list
    yield state
    wakelock._held_id = None


def test_hold_creates_one_assertion_and_release_drops_it(fake):
    wakelock.hold()
    assert len(fake["created"]) == 1
    held = fake["created"][0][2]
    wakelock.release()
    assert fake["released"] == [held]


def test_hold_is_idempotent(fake):
    """Every session-start path calls `hold`, so a second call must not stack a second assertion."""
    wakelock.hold()
    wakelock.hold()
    wakelock.hold()
    assert len(fake["created"]) == 1


def test_release_without_a_hold_is_safe(fake):
    wakelock.release()
    assert fake["released"] == []


def test_re_hold_after_release_takes_a_fresh_assertion(fake):
    wakelock.hold()
    wakelock.release()
    wakelock.hold()
    assert len(fake["created"]) == 2


def test_the_output_guard_keeps_it_off(monkeypatch):
    """`NFB_NO_OUTPUTS` is set for the whole suite, so `hold` must do nothing and say why."""
    calls = []
    monkeypatch.setattr(wakelock, "_create", lambda *a, **k: calls.append(a) or 1)
    assert wakelock.muted() is True
    assert wakelock.available() is False
    wakelock.hold()
    assert calls == []
    assert wakelock._held_id is None
    assert "NFB_NO_OUTPUTS" in wakelock.problem()
