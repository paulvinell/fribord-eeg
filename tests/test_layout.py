"""Layout faults that every other test in this repo is blind to.

`test_screens` proves a frame can be produced. It cannot tell whether the frame is any good, because it
inspects Python objects and the objects are correct in all of the ways that matter to it. Every UI bug
this screen family has shipped was still *in* those objects — labels sharing a position, a spacer that
cannot expand, text wider than the box holding it, a threshold drawn along the frame of its own plot —
and nobody had written the checks.

So: one module, every screen, a list of rules. Each rule is a function over the control tree, each is
named for the fault rather than for the property, and each carries the failure that produced it. Adding a
rule when a new fault turns up is the point; that is how this stops being a list of things that went
wrong once.

**What this is not.** It is not a substitute for looking. A rule can say two labels sit on top of each
other; it cannot say three pills were used where the design wanted one segmented control. See
`tools/shots.py` for the half that renders pixels.
"""

from __future__ import annotations

import importlib

import flet as ft
import flet.canvas as cv
import pytest

from nfb_console.protocols import BY_CODE

from nfb.stream import Block

from conftest import counts, plausible_eeg, text_width
from test_screens import StubDevice, StubPage, replay, toolbar

FED_SECONDS = 30.0

# Screen, protocol code, channels, and whether to push a recording through it first. Cold *and* fed,
# because most of these faults only show in one of the two: a threshold pinned to the top of its plot
# needs no data at all, and an axis whose labels collide needs some.
SCREENS = [
    ("amplitude_view", "AMP", 2),
    ("at_view", "A-T", 2),
    ("cfc_view", "CFC", 2),
    ("ilf_view", "ILF", 2),
    ("pac_view", "PAC", 4),
    ("pac_view", "PAC", 1),
    ("scp_view", "SCP", 1),
    ("sleep_view", "SLEEP", 2),
    ("wave_view", "WAVE", 1),
]

BOLD = {ft.FontWeight.W_500, ft.FontWeight.W_600, ft.FontWeight.W_700, ft.FontWeight.BOLD}


def walk(control, depth: int = 0):
    """Every control in the tree, once."""
    if depth > 22 or control is None:
        return
    yield control
    for attr in ("controls", "items", "shapes"):
        for child in (getattr(control, attr, None) or []):
            yield from walk(child, depth + 1)
    content = getattr(control, "content", None)
    if content is not None and not isinstance(content, str):
        yield from walk(content, depth + 1)


def _pad(control) -> tuple[float, float]:
    """(left, right) padding of a control, however it was spelled."""
    padding = getattr(control, "padding", None)
    if padding is None:
        return 0.0, 0.0
    if isinstance(padding, (int, float)):
        return float(padding), float(padding)
    return float(getattr(padding, "left", 0) or 0), float(getattr(padding, "right", 0) or 0)


def widths(control, budget: float | None, depth: int = 0):
    """Yield `(text, budget)` for every `no_wrap` text whose available width can be worked out.

    Only where it *can* be: a Row splits its width among its children and there is no general way to
    know the split, so a row that is not the one shape we can solve hands its children `None` and they
    are skipped. Silent where it cannot answer, rather than guessing and crying wolf.
    """
    if depth > 22 or control is None:
        return
    own = getattr(control, "width", None)
    if own:
        budget = float(own)
    left, right = _pad(control)
    budget = None if budget is None else budget - left - right

    if isinstance(control, ft.Text):
        if getattr(control, "no_wrap", False) and control.value and budget is not None:
            yield control, budget
        return

    if isinstance(control, ft.Row):
        kids = list(control.controls or [])
        expanding = [k for k in kids if getattr(k, "expand", None)]
        known = [getattr(k, "width", None) for k in kids if not getattr(k, "expand", None)]
        spacing = float(getattr(control, "spacing", 0) or 0) * max(0, len(kids) - 1)
        solvable = (len(expanding) == 1 and budget is not None and all(w for w in known))
        for kid in kids:
            if getattr(kid, "expand", None) and solvable:
                yield from widths(kid, budget - sum(known) - spacing, depth + 1)
            else:
                yield from widths(kid, None, depth + 1)
        return

    for attr in ("controls", "items"):
        for kid in (getattr(control, attr, None) or []):
            yield from widths(kid, budget, depth + 1)
    inner = getattr(control, "content", None)
    if inner is not None and not isinstance(inner, str):
        yield from widths(inner, budget, depth + 1)


# --- the rules ------------------------------------------------------------------------------

def no_expanding_child_in_a_wrapping_row(control) -> list[str]:
    """A grey block the height of the container, with no error anywhere.

    `expand` needs a bounded main axis and `wrap` removes it, so the spacer has nothing to expand
    against and paints as a solid rectangle. It reads as a broken image rather than as a layout fault.
    """
    out = []
    for ctrl in walk(control):
        if isinstance(ctrl, ft.Row) and getattr(ctrl, "wrap", False):
            for kid in (ctrl.controls or []):
                if getattr(kid, "expand", None):
                    out.append(f"{type(kid).__name__} expands inside a wrapping row")
    return out


def no_two_axis_labels_in_one_place(control) -> list[str]:
    """Four gridline values piled into the corner as one unparsable smear.

    `value_axis` hands back labels the caller has to *place*; writing `label.value` and stopping leaves
    every one of them at `top=0`. And a label that names the same number as its neighbour is the other
    half of the same fault — an axis that says one value four times, from rounding.
    """
    out = []
    for ctrl in walk(control):
        if not isinstance(ctrl, ft.Stack):
            continue
        placed = [c for c in (ctrl.controls or [])
                  if isinstance(c, ft.Text) and c.value and c.visible is not False
                  and (c.top is not None or c.left is not None)]
        if len(placed) < 2:
            continue
        # **Which coordinate carries the axis is the axis's business.** A value axis runs down the plot
        # and places by `top`; a tick axis runs along it and places by `left`, so every label on one
        # shares a `top` of zero and is perfectly correct. The fault is when *neither* separates them.
        tops = {c.top for c in placed}
        lefts = {c.left for c in placed}
        vertical = len(tops) == len(placed)
        if not vertical and len(lefts) != len(placed):
            out.append(f"axis labels share a position: {[c.value for c in placed]}")
        elif vertical:
            # Only down the plot, where the fault was: a tick axis may legitimately repeat a word —
            # "trough … peak … trough" names one cycle twice and means it.
            values = [c.value for c in placed]
            if len(set(values)) != len(values):
                out.append(f"axis names one value more than once: {values}")
    return out


def no_text_wider_than_the_box_holding_it(control) -> list[str]:
    """Blurbs clipped mid-word — "twelve inhibi", "if electrodes" — with no ellipsis to say why.

    A fixed slot is the right trade: text that wraps would shove everything under it. It leaves the
    width a hard budget, and nothing was measuring it.
    """
    out = []
    for text, budget in widths(control, None):
        size = float(getattr(text, "size", None) or 14)
        bold = getattr(text, "weight", None) in BOLD
        need = text_width(text.value, size, bold=bold)
        if need > budget - 6:
            out.append(f"{text.value!r} needs {need:.0f}px of {budget:.0f}")
    return out


def no_line_drawn_along_the_edge_of_its_own_plot(control) -> list[str]:
    """A threshold whose plot ceiling was taken *from* that threshold draws along the very top, where it
    reads as the frame rather than as a mark on it.

    Gridlines are excluded by construction: `plots.grid` draws them at interior fractions only, never at
    0 or at the full height. Dashes are summed by row, so a dashed line pinned to an edge counts too.
    """
    out = []
    for ctrl in walk(control):
        if not isinstance(ctrl, cv.Canvas):
            continue
        height = float(getattr(ctrl, "height", 0) or 0)
        if height <= 0:
            continue
        spans: dict[float, float] = {}
        for shape in (ctrl.shapes or []):
            pts = getattr(shape, "points", None)
            if not pts or len(pts) != 2 or abs(pts[0].y - pts[1].y) > 0.01:
                continue
            spans[round(pts[0].y, 2)] = spans.get(round(pts[0].y, 2), 0.0) + abs(pts[1].x - pts[0].x)
        reach = max((abs(p.x) for s in (ctrl.shapes or [])
                     for p in (getattr(s, "points", None) or [])), default=0.0)
        for y, span in spans.items():
            if reach and span >= reach * 0.4 and y <= 2.0:
                out.append(f"a full-width line sits at y={y:g} of a {height:g}px plot")
    return out


SENTENCE = 30
"""Text at least this long is a sentence, and a sentence is said once per screen.

Shorter strings repeat for good reasons — "waiting", "threshold", a unit, a channel name on four cards —
and so does a line that opens with a number, which is a reading, and two readings may coincide. A
sentence that appears twice is a screen-level fact printed per card, which is how the coupling screen
came to carry "held still — a tracker's window holds too few readings of a signal this slow" four times
in one column, with "calibrating — threshold set from the first 120 s, 114 s left" under each one. The
day it was written it also found a shadow row stating its problem twice, a corner warning printed on a
card and again under it, and two outputs sharing one sentence about being switched off.
"""


def no_sentence_printed_twice(control) -> list[str]:
    """The same long text on one screen more than once. A fact true of every card belongs above them."""
    seen: dict[str, int] = {}
    for ctrl in on_screen(control):
        if isinstance(ctrl, ft.Text) and ctrl.value and ctrl.visible is not False \
                and len(ctrl.value) >= SENTENCE and not ctrl.value[0].isdigit():
            seen[ctrl.value] = seen.get(ctrl.value, 0) + 1
    return [f"{text!r} is printed {n} times" for text, n in seen.items() if n > 1]


def on_screen(control, depth: int = 0):
    """`walk`, minus two things that are not prose.

    What a menu holds until it is opened: three cards each offering the same six choices is three menus,
    not the same sentence three times. And the label of a chip or a pill — a Text that is the whole
    content of a filled container — which is a control's name and may repeat like a unit does.
    """
    if depth > 22 or control is None:
        return
    yield control
    for attr in ("controls", "shapes"):
        for child in (getattr(control, attr, None) or []):
            yield from on_screen(child, depth + 1)
    content = getattr(control, "content", None)
    if content is not None and not isinstance(content, str):
        if isinstance(content, ft.Text) and getattr(control, "bgcolor", None):
            return
        yield from on_screen(content, depth + 1)


RULES = (no_expanding_child_in_a_wrapping_row,
         no_two_axis_labels_in_one_place,
         no_text_wider_than_the_box_holding_it,
         no_line_drawn_along_the_edge_of_its_own_plot,
         no_sentence_printed_twice)


def test_launcher_layout():
    """The launcher goes through the same rules. It is where four blurbs were clipped mid-word, and it
    is not a protocol screen, so it needs its own build rather than a row in the table above."""
    from nfb_console import splash

    control = splash.view(StubPage(), StubDevice(), lambda _p: None, lambda: None)
    faults = [f"{rule.__name__}: {fault}" for rule in RULES for fault in rule(control)]
    assert not faults, "\n".join(faults)


@pytest.mark.parametrize("module,code,channels", SCREENS,
                         ids=[f"{c}-{n}ch" for _, c, n in SCREENS])
@pytest.mark.parametrize("fed", [False, True], ids=["cold", "fed"])
def test_screen_layout(module: str, code: str, channels: int, fed: bool):
    screen = importlib.import_module(f"nfb_console.{module}")
    session = screen.session_for(BY_CODE[code], source=replay(), channels=channels)
    try:
        page = StubPage()
        control = screen.view(page, session, toolbar(), lambda: None)
        if fed:
            # **Plausible signal, not the bench recording.** The fixture is open inputs railed to full
            # scale, so every band reads zero and a "fed" screen is a cold one wearing a different name
            # — which meant the rules below were checking one state twice. The same generator feeds
            # `tools/shots.py`, so an audit failure and a screenshot of it are of the same picture.
            session.paused = False
            session.state = dict.fromkeys(session.specs, True)
            waves = {ch: plausible_eeg(FED_SECONDS, session.rate, ch) for ch in session.specs}
            for i in range(0, int(FED_SECONDS * session.rate) - 249, 250):
                for ch, spec in session.specs.items():
                    session.feed_block(Block(ch, i, counts(waves[ch][i:i + 250], spec.scale), spec))
        session.redraw()

        faults = [f"{rule.__name__}: {fault}" for rule in RULES for fault in rule(control)]
        assert not faults, "\n".join(faults)
    finally:
        session.stop()


# --- nothing flickers -------------------------------------------------------------------------

FLICKER_SECONDS = 60.0
FLIPS_ALLOWED = 2
"""A control may appear once and go once over a minute of signal. More than that is a flicker.

The synchrony screen's "mostly amplitude, not phase" note sat on a measurement that wanders across its
floor, so it came and went every frame — and because the note row was hidden when empty, every card below
it jumped each time. A verdict that blinks is worse than no verdict, and a card whose height is a readout
moves the whole column. This drives a minute of signal through every screen a second at a time, redraws
after each, and counts how often each control's visibility changes.
"""


@pytest.mark.parametrize("module,code,channels", SCREENS,
                         ids=[f"{c}-{n}ch" for _, c, n in SCREENS])
def test_nothing_flickers(module: str, code: str, channels: int):
    screen = importlib.import_module(f"nfb_console.{module}")
    session = screen.session_for(BY_CODE[code], source=replay(), channels=channels)
    try:
        page = StubPage()
        control = screen.view(page, session, toolbar(), lambda: None)
        session.paused = False
        session.state = dict.fromkeys(session.specs, True)
        waves = {ch: plausible_eeg(FLICKER_SECONDS, session.rate, ch) for ch in session.specs}
        flips: dict[int, int] = {}
        names: dict[int, str] = {}
        last: dict[int, bool] = {}
        step = int(session.rate)
        for start in range(0, int(FLICKER_SECONDS * session.rate) - step + 1, step):
            for i in range(start, start + step, 250):
                for ch, spec in session.specs.items():
                    session.feed_block(Block(ch, i, counts(waves[ch][i:i + 250], spec.scale), spec))
            session.redraw()
            for ctrl in walk(control):
                shown = getattr(ctrl, "visible", True) is not False
                key = id(ctrl)
                if key in last and last[key] != shown:
                    flips[key] = flips.get(key, 0) + 1
                    inner = getattr(ctrl, "content", None)
                    names[key] = (f"{type(ctrl).__name__} "
                                  f"{getattr(ctrl, 'value', None) or getattr(ctrl, 'tooltip', None) or ''}"
                                  f"{' holding ' + repr(inner.value) if isinstance(inner, ft.Text) else ''}")
                last[key] = shown
        blinking = [f"{names[k]!r} changed visibility {n} times in {FLICKER_SECONDS:g} s"
                    for k, n in flips.items() if n > FLIPS_ALLOWED]
        assert not blinking, "\n".join(blinking)
    finally:
        session.stop()
