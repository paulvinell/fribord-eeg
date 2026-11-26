"""The source choices: what a heart choice offers and when it reports a fault."""

from __future__ import annotations

def test_choosing_the_cable_does_not_report_the_same_fault_twice():
    """The screen has two places a heart-rate fault can appear: a note under the source buttons, and a
    status line under the port field. Both were being told there was no cable — the choice by going and
    looking, the runner by trying — so the same two-line sentence appeared above and below the field.
    Looking for a cable is not something to know in advance; the runner finds out and says so."""
    from nfb_console.inputs import HeartChoice

    assert HeartChoice(kind="usb").problem() is None
    assert HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01").problem() is None


def test_the_heart_source_choice_offers_the_cable():
    from nfb_console.inputs import HeartChoice

    choice = HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01")
    assert choice.on
    assert "/dev/cu.usbserial-HM01" in choice.describe()
    assert choice.problem() is None, "a named port needs nothing found"
    source = choice.open()
    assert type(source).__name__ == "UsbPpgBeats" and source.port == "/dev/cu.usbserial-HM01"
