"""The launcher."""

from __future__ import annotations


from harness import StubDevice, StubPage


def test_the_launcher_scrolls_so_no_protocol_is_below_the_fold():
    """Eleven protocols is four rows of cards, and at the height the window opens at the fourth row is
    not on screen — HEG and HRV were unreachable with nothing saying so."""
    import flet as ft

    from nfb_console import splash

    page = StubPage()
    control = splash.view(page, StubDevice(), lambda _p: None, lambda: None)
    body = control.content
    assert isinstance(body, ft.Column)
    assert body.scroll is not None, "a launcher that hides protocols is worse than one that scrolls"
