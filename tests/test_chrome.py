"""The toolbar: what it names as the instrument, what it offers, and how it shows the recorder."""

from __future__ import annotations

def test_the_toolbar_shows_a_named_instrument_where_it_shows_the_amplifier():
    from nfb import Device, Settings
    from nfb_console import chrome

    device = Device(Settings())
    bar = chrome.Toolbar(device, lambda: None, lambda: None)
    assert bar.dev_name.value == "no device"

    bar.set_source("HeartMath · USB", note="62 bpm · not the amplifier")
    assert bar.dev_name.value == "HeartMath · USB"
    assert bar.channels_text.value == "62 bpm · not the amplifier", \
        "the second line is the instrument's state, not the amplifier's settings"

    bar.set_source("file: yesterday.jsonl.gz")
    assert bar.channels_text.value == "replay · not the amplifier", "replay keeps its own wording"
    device.close()


def test_a_screen_with_no_amplifier_channels_does_not_offer_to_reconfigure_it():
    """The count in the toolbar reconfigures the EEG amplifier, and on the HRV screen there is no
    amplifier in the path at all — so "1 ch" beside a pulse sensor is a control for a device this screen
    never touches. Hidden rather than greyed: disabled means "not now", and this is "not here"."""
    from nfb import Device, Settings
    from nfb_console import chrome

    device = Device(Settings())
    bar = chrome.Toolbar(device, lambda: None, lambda: None)
    assert bar.channel_menu.visible is not False, "the amplifier screens keep it"
    bar.set_channels_visible(False)
    assert bar.channel_menu.visible is False
    device.close()


def test_the_toolbar_paints_the_recorder_it_is_shown(tmp_path):
    from types import SimpleNamespace

    from harness import StubDevice

    from nfb_console import chrome
    from nfb_console import theme as t

    bar = chrome.Toolbar(StubDevice(), lambda: None, lambda: None)
    bar.set_transport_visible(True)
    bar.set_record_supported(True)
    path = tmp_path / "20260822-010203.jsonl.gz"

    bar.recording = SimpleNamespace(active=True, closed=False, frames=90, dropped=0,
                                    seconds=61.0, bytes_written=2_500_000, path=path)
    bar.refresh(running=True)
    assert bar.record_glyph.color == t.BAD
    assert "rec 01:01" in bar.rec_note.value and "2.5 MB" in bar.rec_note.value

    bar.recording = SimpleNamespace(active=False, closed=True, frames=90, dropped=3,
                                    seconds=61.0, bytes_written=2_500_000, path=path)
    bar.refresh(running=False)
    assert bar.record_glyph.color != t.BAD
    assert bar.rec_note.value == f"saved {path.name}"

    bar.recording = None
    bar.set_record_supported(False)
    bar.refresh(running=False)
    assert bar.rec_note.value == ""
    assert bar.record.disabled and bar.record_split.opacity < 1.0, \
        "both halves of the split button grey together"
