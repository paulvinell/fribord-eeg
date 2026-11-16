"""The launcher."""

from __future__ import annotations


from harness import StubDevice, StubPage


def test_every_launcher_card_fits_the_text_it_is_given():
    """**Four blurbs were clipped mid-word** — "twelve inhibi", "from 0.05", "if electrodes".

    The card keeps every text slot a fixed height on purpose, so a description that would wrap overflows
    visibly instead of shoving the big code letters to a different height than the card beside it. That
    is the right trade and it leaves the width a hard budget, which nothing was measuring. Clipping also
    has no ellipsis, so it reads as a rendering fault rather than as text that is too long.
    """
    from nfb_console import splash, theme
    from nfb_console.protocols import CATALOGUE

    from conftest import text_width

    slot = splash.CARD_W - splash.BAR_W - 16 - 14      # the card's padding, from `protocol_card`
    for p in CATALOGUE:
        line = p.blurb if p.available else p.note
        assert line, f"{p.code} has nothing to say in its subtitle"
        assert text_width(line, theme.SMALL) <= slot - 8, (
            f"{p.code}: {line!r} needs {text_width(line, theme.SMALL):.0f}px of {slot}")
        if p.name.upper() != p.code.upper():
            assert text_width(p.name, theme.BODY, bold=True) <= slot - 8, (
                f"{p.code}: the name {p.name!r} does not fit either")


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
