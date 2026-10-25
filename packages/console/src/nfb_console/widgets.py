"""The controls every protocol screen is built from.

Moved out of `amplitude_view.py` when a second protocol screen arrived. None of these knows what is
being trained, which is the test for whether something belongs here: if it mentions a band, a target or
a channel, it is not a widget.

`widgets` and `plots` are the two halves of the view kit — controls you operate, and shapes you read.
"""

from __future__ import annotations

import re
from typing import Callable

import flet as ft

from . import theme as t
from .feedback import READS


def num_box(value: float, on_set: Callable[[float], None], width: int = 46,
            suffix: str = "", step: float = 1.0, low: float | None = None,
            high: float | None = None,
            on_step: Callable[[float], None] | None = None) -> ft.Control:
    """A number with a stepper: type it, or click the arrows to move by `step`.

    `on_step` splits those two apart, and a band edge is why. Typing an edge sets *that* edge, because that
    is what typing a number into a box means. Nudging one should **slide the whole band** and keep its
    width: what you are doing when you reach for the arrows is hunting for a frequency, not redefining how
    wide the band is — and the alternative is two nudges per step, in opposite boxes, with a band that
    changes width if you lose count. Given `on_step`, the arrows call it with the signed delta and leave
    `on_set` to the typed path.

    An entry plus stacked arrows, with an increment and optional
    bounds, because almost every number here is being *adjusted* rather than entered
    from scratch, and adjusting a band edge by 0.5 Hz or a ratio by a point should not require selecting
    text and retyping it.

    Commits on Enter or on losing focus, and clamps rather than rejecting. The trailing `field.update()`
    is guarded because `on_set` usually rebuilds the panel this field lives in, which detaches it — an
    unguarded update then raises "Control must be added to the page first" out of the blur handler.
    """
    places = len(f"{step:g}".split(".")[1]) if "." in f"{step:g}" else 0
    fmt = f"{{:.{places}f}}"
    field = ft.TextField(value=fmt.format(value), width=width, height=30, text_size=t.BODY,
                         text_align=ft.TextAlign.CENTER, color=t.TEXT, bgcolor=t.SURFACE,
                         border_color=t.BORDER, content_padding=ft.Padding(2, 2, 2, 2))

    accepted = [float(value)]           # the last value that got through, for unparseable input

    def apply(raw: float) -> None:
        if low is not None:
            raw = max(low, raw)
        if high is not None:
            raw = min(high, raw)
        raw = round(raw, 6)
        accepted[0] = raw
        # write the clamped value back, so the box shows what was accepted even when nothing rebuilds
        field.value = fmt.format(raw)
        on_set(raw)

    def current() -> float:
        try:
            return float(str(field.value).replace(",", "."))
        except (ValueError, TypeError):
            return accepted[0]          # not the initial value: a clamp may have moved it since

    def commit(_e=None) -> None:
        apply(current())
        try:
            field.update()
        except Exception:
            pass            # the commit rebuilt the panel; this field is no longer on the page

    field.on_submit = commit
    field.on_blur = commit

    def arrow(glyph: str, delta: float) -> ft.Control:
        return ft.Container(content=ft.Text(glyph, size=8, color=t.MUTED),
                            width=16, height=15, alignment=ft.Alignment.CENTER,
                            border=ft.Border.all(1, t.BORDER),
                            on_click=lambda _e: (on_step(delta) if on_step is not None
                                                 else apply(current() + delta)))

    stepper = ft.Column([arrow("▲", step), arrow("▼", -step)], spacing=0)
    parts: list[ft.Control] = [field, stepper]
    if suffix:
        parts.append(ft.Text(suffix, size=t.SMALL, color=t.FAINT))
    return ft.Row(parts, spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def pill(label: str, selected: bool, on_click, enabled: bool = True) -> ft.Control:
    """A choice. `enabled=False` shows the same row, inert.

    Inert rather than absent, and rather than replaced by a sentence, because it still says *that this
    was a choice* and which way it went — a sentence loses the alternatives, and hiding it turns a
    decision the protocol made into one nobody made. Whoever disables a row is responsible for putting
    the reason next to it, or an inert control reads as a broken one.
    """
    # **The chosen one is filled; the others are words.** Every option used to carry its own outline,
    # and the chosen one an accent outline on top, so a row of three choices was three boxes and the
    # screen a grid of them. A fill with no outline is enough to say which is lit, and the rest reading
    # as plain text is what makes the lit one read at all.
    return ft.Container(
        content=ft.Text(label, size=t.SMALL, color=t.TEXT if selected else t.MUTED,
                        weight=ft.FontWeight.W_500 if selected else ft.FontWeight.W_400),
        padding=ft.Padding(9, 5, 9, 5), border_radius=6,
        bgcolor=t.SURFACE_HI if selected else None,
        # A transparent edge on the unchosen ones, so choosing does not move the row by a pixel.
        border=ft.Border.all(1, t.HAIRLINE if selected else ft.Colors.TRANSPARENT),
        opacity=1.0 if enabled else 0.45,
        on_click=on_click if enabled else None)


def tiny(label: str, on_click, width: int = 24) -> ft.Control:
    """A one-glyph button: nudge, remove. Filled rather than outlined, like every other control here."""
    return ft.Container(content=ft.Text(label, size=t.SMALL, color=t.MUTED),
                        width=width, height=22, border_radius=5,
                        alignment=ft.Alignment.CENTER,
                        bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE), on_click=on_click)


# `**word**` is bold and `*word*` is italic in help text, and nothing else is markup. A whole Markdown
# renderer was the alternative and was turned down: it joins single line breaks, which some help lays
# out lines with, it takes neither the theme's size nor its colour, and it turns a line that happens to
# start with "-" or "#" into a list or a heading.
EMPHASIS = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*", re.S)


def emphasis(text: str) -> list[ft.TextSpan]:
    """Help text as spans: plain, bold and italic runs. Size and colour stay on the parent `ft.Text`."""
    spans, at = [], 0
    for m in EMPHASIS.finditer(text):
        if m.start() > at:
            spans.append(ft.TextSpan(text[at:m.start()]))
        if m.group(1) is not None:
            spans.append(ft.TextSpan(m.group(1), style=ft.TextStyle(weight=ft.FontWeight.W_600,
                                                                    color=t.TEXT)))
        else:
            spans.append(ft.TextSpan(m.group(2), style=ft.TextStyle(italic=True)))
        at = m.end()
    if at < len(text):
        spans.append(ft.TextSpan(text[at:]))
    return spans


def help_button(page: ft.Page, title: str, body: str) -> ft.Control:
    """A `?` that opens an explanation, instead of printing one.

    **The screen says what a control does; this says why.** Rationale, provenance and the reasoning behind
    a default are worth having and are not worth a paragraph of standing text next to the control — a
    protocol screen is read for twenty minutes at a stretch and every permanent sentence is rent charged
    to a tenant that has nothing new to say after the first reading. Anything longer than a phrase goes
    behind one of these.
    """
    def open_it(_e) -> None:
        show = getattr(page, "show_dialog", None)
        if show is None:
            return
        dialog = ft.AlertDialog(
            title=ft.Text(title, size=t.H2, color=t.TEXT, weight=ft.FontWeight.W_500),
            content=ft.Container(width=460, content=ft.Text(spans=emphasis(body), size=t.BODY,
                                                            color=t.MUTED)),
            bgcolor=t.SURFACE,
            actions=[ft.Container(content=ft.Text("close", size=t.BODY, color=t.ACCENT),
                                  padding=ft.Padding(12, 7, 12, 7), border_radius=6,
                                  on_click=lambda _e: page.pop_dialog())])
        show(dialog)

    return ft.Container(content=ft.Text("?", size=t.SMALL, color=t.FAINT),
                        width=18, height=18, border_radius=9,
                        alignment=ft.Alignment.CENTER,
                        bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE), on_click=open_it,
                        tooltip=title)


def settings_button(page: ft.Page, title: str, rows: Callable[[Callable[[], None]], list],
                    glyph: str = "⚙") -> ft.Control:
    """A gear that opens editable settings, where a `?` opens an explanation.

    Settings that belong to *one* target — which filter measures it, how sharp, how late — cannot go on the
    card itself without turning a card into a form, and cannot go on a panel without becoming global to
    every target on it. A gear is the third option and the right one: per-target, out of the way, and
    obviously a place where something can be changed rather than read.

    `rows` is called with a `redraw` callback so the dialog can rebuild itself when a value changes and
    show the consequence of the change immediately — which is the whole reason to open it.
    """
    def open_it(_e) -> None:
        show = getattr(page, "show_dialog", None)
        if show is None:
            return
        body = ft.Column(spacing=14, width=460)

        def rebuild() -> None:
            body.controls = rows(rebuild)
            try:
                page.update()
            except Exception:
                pass

        rebuild()
        page.show_dialog(ft.AlertDialog(
            title=ft.Text(title, size=t.H2, color=t.TEXT, weight=ft.FontWeight.W_500),
            content=ft.Container(width=460, content=body),
            bgcolor=t.SURFACE,
            actions=[ft.Container(content=ft.Text("close", size=t.BODY, color=t.ACCENT),
                                  padding=ft.Padding(12, 7, 12, 7), border_radius=6,
                                  on_click=lambda _e: page.pop_dialog())]))

    return ft.Container(content=ft.Text(glyph, size=t.SMALL, color=t.FAINT),
                        width=18, height=18, border_radius=9,
                        alignment=ft.Alignment.CENTER,
                        bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE), on_click=open_it, tooltip=title)


def panel(title: str, *content: ft.Control, expand: bool = False,
          trailing: ft.Control | None = None, help: ft.Control | None = None) -> ft.Control:
    # The trailing controls are given the head's remaining width rather than riding beside a spacer with
    # unbounded constraints. Unbounded, a long pill ladder painted past the panel's right edge and was
    # clipped there — the last scale pills simply were not on the screen, with nothing saying so. Bounded,
    # a trailing row that sets `wrap=True` can flow onto a second line instead; one that fits keeps
    # exactly the old right-aligned look.
    head = ft.Row([ft.Text(title, size=t.SMALL, color=t.FAINT)]
                  + ([ft.Container(width=7), help] if help else [])
                  + ([ft.Container(content=trailing, expand=True,
                                   alignment=ft.Alignment.CENTER_RIGHT)]
                     if trailing else [ft.Container(expand=True)]),
                  vertical_alignment=ft.CrossAxisAlignment.CENTER)
    # The surface colour lifts the panel off the ground on its own; an outline on top of that was the
    # first of three nested frames on every screen, with the cards' outlines the third.
    return ft.Container(
        content=ft.Column([head, ft.Container(height=10), *content], spacing=0, expand=expand),
        bgcolor=t.SURFACE, border=ft.Border.all(1, t.HAIRLINE), border_radius=10,
        padding=ft.Padding(16, 13, 16, 14), expand=expand)


def labelled(name: str, *controls: ft.Control, note: str = "", width: int = 96) -> ft.Control:
    """One setting: what it is on the left, the controls, and only then a note if there is one.

    **Most rows do not need a note.** Offering the argument invited a remark on every row, and a remark on
    a row whose controls already say everything is worse than nothing — "never silent, never blasting" next
    to a 60-to-100 range is a comment on the defaults, not an explanation of the control. A note earns its
    place by saying something the controls cannot: a live value, a device count, a reason it will not work.
    Everything else belongs behind the panel's `?`.
    """
    parts: list[ft.Control] = [ft.Text(name, size=t.SMALL, color=t.MUTED, width=width), *controls]
    if note:
        parts += [ft.Container(width=10),
                  ft.Text(note, size=t.SMALL, color=t.FAINT, expand=True)]
    return ft.Row(parts, spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def range_row(name: str, mech, on_change: Callable[[object, str, float], None],
              note: str) -> ft.Control:
    """Every mechanism has a floor and a ceiling, for the same two reasons each time: output that
    reaches zero stops informing, and output at full has nowhere left to go."""
    return labelled(
        name,
        num_box(mech.low, lambda v: on_change(mech, "low", v), step=1, low=0, high=100),
        ft.Text("to", size=t.SMALL, color=t.FAINT),
        num_box(mech.high, lambda v: on_change(mech, "high", v), suffix=mech.unit,
                step=1, low=0, high=100),
        note=note)


def test_button(label: str, on_click, enabled: bool = True) -> ft.Control:
    """Anything audible can be auditioned; the alternative is waiting five minutes to discover the
    beep was inaudible."""
    return ft.Container(
        content=ft.Row([ft.Text("▶", size=9, color=t.ACCENT if enabled else t.FAINT),
                        ft.Text(label, size=t.SMALL,
                                color=t.ACCENT if enabled else t.FAINT)], spacing=6),
        padding=ft.Padding(9, 5, 9, 5), border_radius=6,
        border=ft.Border.all(1, t.BORDER), on_click=on_click if enabled else None)


def reads_chip(mech) -> ft.Control:
    """Which of the two channels this mechanism follows: the level, or the condition.

    It used to print the *kind* — "continuous" or "discrete" — which is a word from the code and says
    nothing to anyone who has not read it. What the reader needs is which number moves this output, since
    that is the answer to "the signal changed and this did not". A magnitude implies continuous anyway;
    the reverse is not legible.
    """
    return ft.Container(content=ft.Text(READS[mech.reads], size=t.SMALL, color=t.MUTED),
                        padding=ft.Padding(7, 3, 7, 3), border_radius=5, bgcolor=t.SURFACE_HI)


def button(label: str, on_click, colour: str | None = None) -> ft.Control:
    return ft.Container(content=ft.Text(label, size=t.SMALL, color=colour or t.ACCENT),
                        padding=ft.Padding(10, 5, 10, 5), border_radius=6,
                        border=ft.Border.all(1, t.BORDER), on_click=on_click)


# --- the frequency ladder ---------------------------------------------------------------------
#
# Here rather than on a screen because **two screens have a frequency box now** — ILF Synchrony and Waveform
# — and this ladder is what their steppers step by and what their typed values snap to. Two copies of it
# would be two copies of a rounding rule, which is the shape of duplication
# this repo has already paid for once: the moment one is corrected the other is wrong and nothing says so.
#
# It belongs beside `num_box` in particular, since a box whose arrow moves by a finer step than its snapping
# grid moves and springs back, which is the failure `grid_hz` is written to prevent.

COARSE_GRID_HZ = 0.1
"""Our step above 1 Hz: a tenth of a hertz. **The one rung of this ladder that is a judgement.**

A quarter of a hertz is too coarse to place a reward at alpha or gamma. At 10 Hz it offers 9.75, 10.00, 10.25
and nothing between, which is a fifth of the band's own 3 Hz width per step and cannot follow an individual
peak. Reported from real use, and it is a judgement about the instrument, hence a named constant.

The other three rungs are the conventional ladder, so infra-slow work still lands on round numbers."""


def grid_hz(hz: float) -> float:
    """The frequency grid at this magnitude.

    Free below 0.01 Hz, hundredths below 0.1, twentieths below 1 and `COARSE_GRID_HZ` above. It is also the
    step ladder, so one function serves the arrows and the snapping and they cannot disagree, which two
    functions eventually would. That is why the step could not be made finer on its own: a finer arrow
    against a coarser snap would move the box and have the value spring back.
    """
    if hz < 0.01:
        return 0.001
    if hz < 0.1:
        return 0.01
    if hz < 1.0:
        return 0.05
    return COARSE_GRID_HZ


def snap_hz(hz: float) -> float:
    """`hz` on its own grid. Below 0.01 Hz it is left alone, which is the convention: at four significant figures
    of a millihertz there is no grid worth imposing."""
    if hz < 0.01:
        return round(hz, 6)
    step = grid_hz(hz)
    return round(round(hz / step) * step, 6)
