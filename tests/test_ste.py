"""Every string a user can read follows ASD-STE100 Simplified Technical English.

Set as the standard for all on-screen text. The rules a test can check mechanically are checked here over
every screen's rendered text, cold and fed, and over every help dialog as it opens — every `?` is pressed,
so help written at the button or built from a target is read as well as the constants. Word choice
against the approved dictionary is a reading job, since the dictionary is ASD's and not vendored.

What is checked:

- a sentence is at most 25 words (STE: 20 in a procedure, 25 in a description — the longer limit, since a
  caption is a description);
- no contractions ("don't", "it's", "you're");
- no "may", "might", "shall", "should" — STE writes "can" for ability and "must" for a requirement;
- no Latin abbreviations ("e.g.", "i.e.", "etc.", "vs");
- no "and/or";
- no "would", "could", "simply", "whatever", "just", and none of a short list of idioms and project
  history ("used to", "a bump", "does not care") — the rest of word choice is a reading job.
- in a help dialog, no asterisk survives the emphasis markers, and a bold word in lower case is a label
  that the screen shows.

Code comments and docs are out of scope; this is about what the person in the chair reads.
"""

from __future__ import annotations

import importlib
import re

import flet as ft
import pytest



from nfb.stream import Block
from nfb_console.protocols import BY_CODE

from conftest import counts, plausible_eeg
from harness import (OTHER_SCREENS, SCREENS, StubDevice, StubPage, built_screen, replay, screen_params,
                     toolbar)
from test_layout import walk

MAX_WORDS = 25
BANNED = {"may", "might", "shall", "should", "would", "could", "simply", "whatever", "just"}
IDIOM = (
    # phrases that are not STE however short the sentence is: history, hedging, figures of speech
    "used to ", "a couple of", "come round", "comes round", "worth ", "a bump", "the whole point",
    "does not care", "pays out", "pay out", "in other words", "so to speak", "rule of thumb",
    "a handful", "by hand", "on the fly", "plainly", "honest", "merely", "nice",
)
LATIN = re.compile(r"\b(e\.g\.|i\.e\.|etc\.?|vs\.?|viz\.)", re.I)
CONTRACTION = re.compile(r"\b\w+n't\b|\b\w+'(re|ll|ve|d|m)\b|\b(it|that|there|what|who|he|she)'s\b", re.I)

SENTENCE_END = re.compile(r"[.!?]+(\s|$)|\s·\s|\n| — |; ")


def sentences(text: str):
    for part in SENTENCE_END.split(text):
        if part and part.strip() and len(part.strip()) > 1:
            yield part.strip()


def faults(text: str) -> list[str]:
    out = []
    for sentence in sentences(text):
        words = [w for w in re.split(r"\s+", sentence) if re.search(r"[A-Za-z]", w)]
        if len(words) > MAX_WORDS:
            out.append(f"{len(words)} words: {sentence!r}")
        low = {w.strip(".,;:()").lower() for w in words}
        hit = BANNED & low
        if hit:
            out.append(f"{sorted(hit)}: {sentence!r}")
    if CONTRACTION.search(text):
        out.append(f"contraction: {text!r}")
    if LATIN.search(text):
        out.append(f"latin abbreviation: {text!r}")
    if "and/or" in text:
        out.append(f"and/or: {text!r}")
    low_text = text.lower()
    for phrase in IDIOM:
        if phrase in low_text:
            out.append(f"idiom {phrase.strip()!r}: {text[:90]!r}")
    return out


def _texts(control) -> list[str]:
    out = []
    for ctrl in walk(control):
        if isinstance(ctrl, ft.Text):
            if ctrl.value:
                out.append(ctrl.value)
            out.extend(span.text for span in (ctrl.spans or []) if span.text)
        tip = getattr(ctrl, "tooltip", None)
        if isinstance(tip, str) and tip:
            out.append(tip)
    return out


def open_every_help(page, control) -> tuple[list[ft.AlertDialog], list[str]]:
    """Click every `?` under `control`, and every gear, for the `?` inside its dialog.

    Returns the help dialogs, and every label the operator can see on the screen and in its gear
    dialogs. Help text is checked here as the dialog shows it, because a constant is only part of it:
    some help is written at the button, and some is built by a method from the target it explains.
    """
    helps, labels, seen = [], _texts(control), set()
    trees = [control]
    while trees:
        for ctrl in walk(trees.pop()):
            glyph = getattr(ctrl, "content", None)
            if (not isinstance(ctrl, ft.Container) or ctrl.on_click is None
                    or not isinstance(glyph, ft.Text) or glyph.value not in ("?", "⚙")
                    or id(ctrl) in seen):
                continue
            seen.add(id(ctrl))
            before = len(page.dialogs)
            ctrl.on_click(None)
            if len(page.dialogs) == before:
                continue
            dialog = page.dialogs[-1]
            if glyph.value == "?":
                helps.append(dialog)
            else:
                labels += _texts(dialog.content)
                trees.append(dialog.content)
    return helps, labels


def _help_faults(page, control) -> list[str]:
    helps, labels = open_every_help(page, control)
    assert helps, "no help dialog opened: the walk no longer finds the ? buttons"
    on_screen = " | ".join(labels).lower()
    found = []
    for dialog in helps:
        title = dialog.title.value
        body = dialog.content.content
        shown = "".join(span.text for span in body.spans)
        found += [f"{title}: {fault}" for fault in faults(shown)]
        if "*" in shown:
            found.append(f"{title}: an asterisk is shown as it is (an unmatched emphasis marker)")
        for span in body.spans:
            weight = span.style.weight if span.style else None
            word = span.text.strip(" .,:;")
            # a bold word in lower case names a control: it has to be one the operator can find
            if (weight is not None and word[:1].islower() and word not in LABEL_EXEMPT
                    and word.lower() not in on_screen):
                found.append(f"{title}: **{word}** names no label on this screen")
    return found


LABEL_EXEMPT = {"void"}     # a trial verdict the history draws as a mark, not a word


def help_constants() -> list[tuple[str, str]]:
    """Every module-level string constant whose name says it is read on screen."""
    import pkgutil

    import nfb_console

    out = []
    for info in pkgutil.iter_modules(nfb_console.__path__):
        mod = importlib.import_module(f"nfb_console.{info.name}")
        for name, value in vars(mod).items():
            if not name.isupper() or not re.search(r"HELP|NOTE", name):
                continue
            if isinstance(value, str):
                out.append((f"{info.name}.{name}", value))
            elif isinstance(value, dict):
                out.extend((f"{info.name}.{name}[{k!r}]", v) for k, v in value.items()
                           if isinstance(v, str))
    return out


@pytest.mark.parametrize("module,code,channels", screen_params(SCREENS))
@pytest.mark.parametrize("fed", [False, True], ids=["cold", "fed"])
def test_screen_text_is_simplified_technical_english(module, code, channels, fed):
    _page, control = built_screen(module, code, channels, fed)
    found = []
    for text in dict.fromkeys(_texts(control)):
        found += faults(text)
    assert not found, "\n".join(found)


@pytest.mark.parametrize("module,code,channels", screen_params(OTHER_SCREENS))
def test_other_screen_text_is_simplified_technical_english(module, code, channels):
    """The screens that read another device, as they open."""
    _page, control = built_screen(module, code, channels, False)
    found = [fault for text in dict.fromkeys(_texts(control)) for fault in faults(text)]
    assert not found, "\n".join(found)


def test_launcher_text_is_simplified_technical_english():
    from nfb_console import splash

    control = splash.view(StubPage(), StubDevice(), lambda _p: None, lambda: None)
    found = []
    for ctrl in walk(control):
        if isinstance(ctrl, ft.Text) and ctrl.value:
            found += faults(ctrl.value)
    assert not found, "\n".join(found)


def test_help_text_is_simplified_technical_english():
    found = []
    for name, text in help_constants():
        for fault in faults(text):
            found.append(f"{name}: {fault}")
    assert not found, "\n".join(found)


@pytest.mark.parametrize("module,code,channels", screen_params(SCREENS))
def test_no_screen_shows_an_emphasis_marker(module, code, channels):
    _page, control = built_screen(module, code, channels, True)
    shown = [text for text in _texts(control) if "*" in text]
    assert not shown, "\n".join(shown)


# Last in the file: it presses every `?` and every gear on the shared screens, and the tests above read
# those screens as they were built.
@pytest.mark.parametrize("module,code,channels", screen_params(SCREENS + OTHER_SCREENS))
@pytest.mark.parametrize("fed", [False, True], ids=["cold", "fed"])
def test_every_help_dialog_is_simplified_technical_english(module, code, channels, fed):
    if fed and not channels:
        pytest.skip("fed with EEG blocks; this screen reads another device")
    page, control = built_screen(module, code, channels, fed)
    found = _help_faults(page, control)
    assert not found, "\n".join(dict.fromkeys(found))


def test_scp_text_and_help_at_one_two_and_four_channels():
    """The **extra channels** row and its help paragraph exist only where there is a channel to pick. At
    one channel neither shows, and nothing asks for a second channel."""
    from nfb_console import scp_view

    for channels in (1, 2, 4):
        session = scp_view.session_for(BY_CODE["SCP"], source=replay(), channels=channels)
        try:
            page = StubPage()
            control = scp_view.view(page, session, toolbar(), lambda: None)
            if channels > 1:
                session.set_extra(1, True)
            session.paused = False
            waves = {ch: plausible_eeg(10.0, session.rate, ch) for ch in session.specs}
            if channels == 4:
                session.set_extra(2, True)
                waves[2] = [-800.0] * len(waves[2])
            for i in range(0, int(10.0 * session.rate) - 249, 250):
                for ch, spec in session.specs.items():
                    session.feed_block(Block(ch, i, counts(waves[ch][i:i + 250], spec.scale), spec))
            session.redraw()
            shown = _texts(control)
            found = [f for text in dict.fromkeys(shown) for f in faults(text)]
            found += _help_faults(page, control)
            assert not found, "\n".join(dict.fromkeys(found))
            helps, _labels = open_every_help(page, control)
            said = " ".join("".join(s.text for s in d.content.content.spans) for d in helps)
            row = any("extra channels" in text for text in shown)
            assert row == (channels > 1), f"{channels} ch: the extra channels row"
            assert ("extra channels" in said) == (channels > 1), f"{channels} ch: the help paragraph"
        finally:
            session.stop()
