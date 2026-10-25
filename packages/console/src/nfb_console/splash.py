"""The launcher: pick a protocol.

A grid of cards, each a short code over a one-line description, with an
accent bar down the left — because it reads instantly and the codes are what you end up saying out
loud. Departures: unavailable cards say *why* rather than just going grey.
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from nfb import Device

from . import theme as t
from .protocols import CATALOGUE, Protocol

CARD_W = 236
CARD_H = 108
CODE_SLOT = 50     # fixed so the code sits at the same height on every card
SUB_SLOT = 20      # fixed so a wrapping description cannot shift the code

# The accent runs the full height of the card, flush to its left edge, rather than trying to line up
# with the code text. Three attempts at the latter all looked wrong — a full-slot bar left the letters
# low inside it, a cap-height bar was too short, centring both put the bar high — because where the
# ink sits inside a text line box depends on font metrics that are not worth chasing. A card edge has
# nothing to align to and cannot be off.
BAR_W = 3


def protocol_card(p: Protocol, on_pick: Callable[[Protocol], None]) -> ft.Control:
    """One card. Every text slot has a fixed height.

    That is the whole trick: if the description is allowed to wrap, the content grows and the big
    code text sits at a different height from the card next to it. Fixed slots mean a two-line
    description would overflow visibly rather than silently shove the layout around, which is the
    failure you want.
    """
    enabled = p.available
    size = t.CODE if len(p.code) <= 3 else 30
    code = ft.Text(p.code, size=size, weight=ft.FontWeight.W_300,
                   color=t.TEXT if enabled else t.FAINT)

    # **The code and the blurb; the name is not on the card.** "A-T" over "Alpha-Theta" over "Alpha
    # against theta" said the same thing three times, and the codes are the vocabulary of the people
    # this launcher is for. The name is the crumb once the screen is open.
    subtitle: list[ft.Control] = [
        ft.Text(p.blurb if enabled else p.note, size=t.SMALL,
                color=t.MUTED if enabled else t.FAINT, no_wrap=True)]

    text = ft.Column(
        [
            ft.Container(height=CODE_SLOT, content=code,
                         alignment=ft.Alignment.CENTER_LEFT),
            # bottom-aligned so the description shares a baseline across every card, whether or
            # not a name line sits above it
            ft.Container(height=SUB_SLOT, content=ft.Column(subtitle, spacing=3),
                         alignment=ft.Alignment.BOTTOM_LEFT),
        ],
        spacing=6,
    )

    edge = ft.Container(width=BAR_W, bgcolor=t.ACCENT if enabled else t.BORDER)

    card = ft.Container(
        content=ft.Row(
            [
                edge,
                ft.Container(content=text, expand=True,
                             padding=ft.Padding(16, 16, 14, 14)),
            ],
            spacing=0,
        ),
        width=CARD_W, height=CARD_H,
        bgcolor=t.SURFACE,
        border=ft.Border.all(1, t.BORDER),
        border_radius=8,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,   # so the edge follows the rounded corners
        animate_scale=ft.Animation(90, ft.AnimationCurve.EASE_OUT),
        animate=ft.Animation(120, ft.AnimationCurve.EASE_OUT),
    )

    if enabled:
        def hover(e: ft.ControlEvent) -> None:
            on = e.data == "true"
            card.bgcolor = t.SURFACE_HI if on else t.SURFACE
            card.border = ft.Border.all(1, t.ACCENT_DIM if on else t.BORDER)
            card.scale = 1.012 if on else 1.0
            card.update()

        card.on_hover = hover
        card.on_click = lambda _e: on_pick(p)
        card.ink = False
    else:
        card.opacity = 0.5

    return card


def view(page: ft.Page, device: Device, on_pick: Callable[[Protocol], None],
         on_device: Callable[[], None]) -> ft.Control:
    # **A short last row is padded to full width, not centred.** Each row centres itself, so eleven
    # cards put the last two in the middle of the window with the three columns above them running past
    # on both sides — a grid that stops being a grid on its final line. The spacers are the missing
    # cards' width, so every row is the same width and every column keeps its edge.
    rows = []
    cards = [protocol_card(p, on_pick) for p in CATALOGUE]
    for i in range(0, len(cards), 3):
        row = cards[i:i + 3]
        row += [ft.Container(width=CARD_W) for _ in range(3 - len(row))]
        rows.append(ft.Row(row, spacing=16, alignment=ft.MainAxisAlignment.CENTER))

    body = ft.Column(
        [
            ft.Container(height=44),
            ft.Text("Choose a protocol", size=t.H1, weight=ft.FontWeight.W_400, color=t.TEXT),
            ft.Container(height=6),
            ft.Text("Nothing is recorded until you start a session.",
                    size=t.BODY, color=t.MUTED),
            ft.Container(height=34),
            ft.Column(rows, spacing=16),
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=0,
        # **Scrolls, because the catalogue outgrew the window.** Eleven protocols is four rows, and at
        # the height this opens at the fourth is simply not there — HEG and HRV were unreachable with
        # nothing on screen saying so. A launcher that hides protocols is worse than one that scrolls.
        scroll=ft.ScrollMode.AUTO,
    )

    # no header of its own: chrome.Toolbar sits above every screen now
    return ft.Container(content=body, expand=True)
