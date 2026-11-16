"""What the screen tests share: a page and a device that stand in for the real ones, sources that do not
wait, and the walks that find a control or a text on a built screen.

A stub `Page` stands in for Flet. That is sound for the draw path, where `page.update()` does nothing in
both cases. It proves nothing about an update that reaches a client: no test here can see that failure,
and the rule that prevents it is that every UI change runs on Flet's own loop.
"""

from __future__ import annotations



from nfb import ReplaySource, Settings, WaitingSource
from nfb_console import chrome
from nfb_console.protocols import BY_CODE

from conftest import BENCH, counts as to_counts, plausible_eeg

RECORDING = str(BENCH)      # see conftest: a generated bench recording, not sessions/


class StubPage:
    width = 1240

    def __init__(self):
        self.controls, self.services, self.tasks, self.dialogs = [], [], [], []
        self.window = type("W", (), {})()
        self.updates = 0

    def show_dialog(self, dialog):
        self.dialogs.append(dialog)

    def pop_dialog(self):
        self.dialogs.pop()

    def add(self, control):
        self.controls.append(control)

    def update(self, *a):
        self.updates += 1

    def run_task(self, fn, *a):
        self.tasks.append((fn, a))     # never awaited here: a draw loop would run for the whole session


class StubDevice:
    """Just enough for `chrome.Toolbar`, which reads device status every refresh."""

    class _Settings:
        hz = 250.0
        reference = "linked"
        mains_hz = 50

    class _Status:
        from nfb import Mode
        mode = Mode.DISCONNECTED
        failure = None
        remedy = ""              # the launcher's status chip reads it; the toolbar does not
        model = ""
        connected = False
        settings = None

    def __init__(self):
        self.status = self._Status()
        self.status.settings = self._Settings()


def shown(dialog) -> str:
    """A help dialog's text as it reads on screen: its runs joined, the emphasis markers gone."""
    return "".join(span.text for span in dialog.content.content.spans)


def toolbar():
    return chrome.Toolbar(StubDevice(), lambda: None, lambda: None)


def pump(session, blocks: int = 200_000, until=None) -> None:
    """Drive the session's feed **on this thread**, until it has what the test needs.

    **No thread and no clock.** This started the feed on a thread and then polled with a 50 ms sleep
    until a deadline — which measures the scheduler, costs whatever the sleep granularity happens to be,
    and fails under load rather than when the code is wrong. `session.feed_block` is the body of that
    thread's loop, so a test can simply be the loop: exactly the blocks it needs, in order, and stop.

    The default `until` is "enough for the spectrum, not merely enough for a band amplitude" — the
    spectrum branch only runs once a second of samples exists, so stopping earlier would leave the whole
    of it (axes, gridlines, bin colouring) undrawn and unchecked.
    """
    def ready() -> bool:
        ring = session.rings.get(session.focus) if hasattr(session, "rings") else None
        if ring is not None:
            return ring.psd() is not None
        return bool(session.amplitudes)

    done = until or ready
    for n, item in enumerate(session.source.blocks()):
        session.feed_block(item)
        if done() or n >= blocks:
            break
    session._stop = True


def replay(loop: bool = False):
    return ReplaySource(RECORDING, realtime=False, loop=loop)


def at_corner(corner_hz: float, channels: int = 2):
    """A source whose high-pass corner is **stated rather than inherited from `sessions/`**.

    `replay()` reads a gitignored file that is whatever was recorded last, so a capture made at 0.01 Hz
    changes the corner under any test that reads one — which it did, once `corner_hz` began reading the
    corner from the source rather than from a device that only exists live.
    """
    return WaitingSource(Settings(channels=channels, lower_limit_hz=corner_hz).build())


def _feed_two_tones(session, antiphase: bool, seconds: float = 4.0) -> None:
    """Put the same 10 Hz tone on both channels, in phase or in antiphase, and measure it.

    Hand-built blocks rather than a recording, because the whole question is what happens to a *phase*
    relationship and a recording has whatever phase it has. Blocks are handed over one channel at a time in
    the order the device sends them, so this also exercises the aligner rather than stepping around it.
    """
    import numpy as np

    from nfb.stream import Block

    specs = session.specs
    chans = list(specs)
    rate = specs[chans[0]].rate_hz
    scale = specs[chans[0]].scale
    n = int(seconds * rate)
    times = np.arange(n) / rate
    wave = 30.0 * np.sin(2 * np.pi * 10.0 * times)
    other = -wave if antiphase else wave
    ints = {ch: to_counts(values, scale) for ch, values in zip(chans, (wave, other))}
    for start in range(0, n - 8, 8):
        for ch in chans:
            block = Block(ch, start, tuple(ints[ch][start:start + 8].tolist()), specs[ch], state=0)
            # The gate's real verdict, as the feed hands it over — the traces read it to mark which
            # columns were thrown away. `pass_through` is the unjudged one, which is what this is.
            session.observe_raw(block, session.gates.pass_through(block))
            fresh = session.consume(block)
            for tgt in session.targets:
                if session._due(tgt, fresh):
                    tgt.observe(session.value_for(tgt), 20.0)


def show_all_panes(control) -> None:
    """Open every tab at once. A hidden pane deliberately paints nothing — repainting four invisible
    panes at 10 Hz is what starved the event loop that delivers clicks — so a test that reads a pane's
    text has to open it first, exactly as a person would."""
    for pane in control.controls[2].controls:
        pane.visible = True


def _trace_rows(page, session):
    """The `TraceRows` a screen built, found by asking the session's own redraw closure for it.

    Reaching into the view is the point: what is being checked is that the *screen* wired the shared rows up,
    not that the class works — which is a separate test and would pass with the rows left out of the layout.
    """
    from nfb_console.trace_rows import TraceRows

    for cell in getattr(session.redraw, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, TraceRows):
            return value
    return None


def _numbers(control, depth: int = 0):
    """Every editable number on a control tree, as a list of strings.

    Separate from `_texts` because a typeable value is a TextField and not a Text — which is the whole point
    of it being typeable, and is how two tests that asserted a *rendered* pitch stopped seeing it the moment
    the pitch became settable.
    """
    import flet as ft

    if control is None or depth > 20:
        return []
    if isinstance(control, ft.TextField):
        return [control.value] if control.value else []
    out = []
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            out += _numbers(child, depth + 1)
    for attr in ("content", "control"):
        out += _numbers(getattr(control, attr, None), depth + 1)
    return out


def _by_label(control, label: str, depth: int = 0):
    """The container whose only Text is `label` — i.e. a pill, found by what it says.

    For checking that an inert control is *on the screen and inert*, which is the claim the convention
    makes: locked is not hidden. Asking the model whether it thinks the choice is blocked would pass with
    nothing rendered.
    """
    import flet as ft

    if control is None or depth > 20:
        return None
    if isinstance(control, ft.Container) and isinstance(control.content, ft.Text) \
            and control.content.value == label:
        return control
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            found = _by_label(child, label, depth + 1)
            if found is not None:
                return found
    for attr in ("content", "control"):
        found = _by_label(getattr(control, attr, None), label, depth + 1)
        if found is not None:
            return found
    return None


def _texts(control, depth: int = 0):
    """Every Text in a control tree, for checking that captions were written.

    The depth limit is a runaway guard and nothing else, so it is set well clear of what any screen
    actually nests: at 14 it sat one level above the target cards, and wrapping a Train tab in a column to
    hang a warning above it silently emptied this list rather than failing the walk.
    """
    import flet as ft

    if control is None or depth > 20:
        return []
    if isinstance(control, ft.Text):
        return [control] if control.value else []
    out = []
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            out += _texts(child, depth + 1)
    content = getattr(control, "content", None)
    if content is not None:
        out += _texts(content, depth + 1)
    return out


def _menu_texts(control, depth: int = 0):
    """Every Text inside a popup menu's items. Separate from `_texts` deliberately: a menu's options are
    not on the screen until it is opened, and folding them in would make every caption assertion count
    things nobody can see."""

    if control is None or depth > 20:
        return []
    out = []
    for item in getattr(control, "items", None) or []:
        out += _texts(getattr(item, "content", None), depth + 1)
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            out += _menu_texts(child, depth + 1)
    out += _menu_texts(getattr(control, "content", None), depth + 1)
    return out


def _heart_with(split: bool, seconds: float = 300.0):
    """A runner already fed a whole simulated session, without waiting for one."""
    from nfb.hrv import SimulatedBeats
    from nfb_console import heart

    runner = heart.HeartRunner()
    source = SimulatedBeats(rsa_hz=0.1, seed=21, realtime=False, split=split)
    start = None
    for beat in source.beats():
        runner.state = runner.processor.push(beat)
        start = start if start is not None else beat.t
        if beat.t - start > seconds:
            break
    source.close()
    return runner


def _mayer_readings(seconds: float, rate_hz: float = 20.0, cycle: float = 40.0):
    """A sub-hertz synchrony reading: a level that drifts and never returns towards zero.

    Not a fitted curve — this is the shape the real one has, and the shape is the whole point. Measured on
    a 0.1 Hz reward from a real capture: 110–228 µV around a median of 175, so it swings by a third of
    itself over tens of seconds and its 5th percentile sits at 0.70 of its middle, where a Rayleigh
    envelope would put it at 0.27.
    """
    import numpy as np

    n = int(seconds * rate_hz)
    times = np.arange(n) / rate_hz
    return 175.0 * (1.0 + 0.25 * np.sin(2 * np.pi * times / cycle)
                    + 0.08 * np.sin(2 * np.pi * times / (cycle * 2.7)))


def _console(channels: int = 2, protocol: str = "AMP"):
    pass

    from nfb_console.app import Console

    # Through the real parser, never a hand-built Namespace: a hand-built one is how the console's own
    # command line drifted out of step with the Makefile and `make live` broke unnoticed.
    from nfb_console.app import build_parser
    args = build_parser().parse_args([RECORDING, "--channels", str(channels)])
    console = Console(StubPage(), args)
    console.pick(BY_CODE[protocol])
    return console


def _live_console(monkeypatch, channels: int = 2, protocol: str = "AMP"):
    import optima

    from test_device import FakeOptima
    from nfb_console.app import Console, build_parser

    fake = FakeOptima()
    monkeypatch.setattr(optima, "Optima", lambda *a, **k: fake)
    args = build_parser().parse_args(["--live", "--channels", str(channels)])
    console = Console(StubPage(), args)
    console.pick(BY_CODE[protocol])
    return console, fake


def struct_unpack(png_head: bytes) -> tuple[int, int]:
    """Width and height out of a PNG's IHDR."""
    import struct

    return struct.unpack(">II", png_head[16:24])


def _find_help(control, tooltip, depth: int = 0):
    """A panel's `?` button, by the title it opens."""
    import flet as ft

    if control is None or depth > 16:
        return None
    if isinstance(control, ft.Container) and getattr(control, "tooltip", None) == tooltip:
        return control
    for child in getattr(control, "controls", None) or []:
        found = _find_help(child, tooltip, depth + 1)
        if found is not None:
            return found
    content = getattr(control, "content", None)
    if content is not None:
        return _find_help(content, tooltip, depth + 1)
    return None


def _find_gear(control, tooltip="Target", depth: int = 0):
    import flet as ft

    if control is None or depth > 16:
        return None
    if isinstance(control, ft.Container) and getattr(control, "tooltip", None) == tooltip:
        return control
    for child in getattr(control, "controls", None) or []:
        found = _find_gear(child, tooltip, depth + 1)
        if found is not None:
            return found
    content = getattr(control, "content", None)
    if content is not None:
        return _find_gear(content, tooltip, depth + 1)
    return None


def _pills(control, depth: int = 0):
    """Every pill in a tree, as (label, live). A pill is a Container holding one Text; a live one has a
    click handler and a dimmed one does not."""
    import flet as ft

    if control is None or depth > 20:
        return []
    out = []
    if isinstance(control, ft.Container) and isinstance(control.content, ft.Text) \
            and control.border_radius == 6 and control.padding is not None:
        out.append((control.content.value, control.on_click is not None))
    for attr in ("controls", "shapes"):
        for child in getattr(control, attr, None) or []:
            out += _pills(child, depth + 1)
    if not isinstance(control, ft.Container) or not isinstance(control.content, ft.Text):
        out += _pills(getattr(control, "content", None), depth + 1)
    return out


def _chosen(control) -> set:
    """Which pills read as selected, by the accent background `widgets.pill` gives the chosen one."""
    from nfb_console import theme

    out = set()
    for label, _live in _pills(control):
        out.add(label)
    picked = set()

    def walk(node, depth=0):
        import flet as ft
        if node is None or depth > 14:
            return
        if isinstance(node, ft.Container) and isinstance(node.content, ft.Text) \
                and node.bgcolor == theme.SURFACE_HI and node.border_radius == 6:
            picked.add(node.content.value)
        for attr in ("controls", "shapes"):
            for child in getattr(node, attr, None) or []:
                walk(child, depth + 1)
        if not isinstance(node, ft.Container) or not isinstance(node.content, ft.Text):
            walk(getattr(node, "content", None), depth + 1)

    walk(control)
    return picked & out


def _contact_rows(control):
    """The per-channel rows of the ELECTRODE CONTACT panel, found by the label they carry."""
    import flet as ft

    found = []

    def walk(node, depth=0):
        if node is None or depth > 14:
            return
        if isinstance(node, ft.Row):
            labels = [x.value for x in _texts(node)]
            if any(v in (f"ch{c}" for c in range(4)) for v in labels) and len(labels) <= 3:
                found.append(node)
        for attr in ("controls", "shapes"):
            for child in getattr(node, attr, None) or []:
                walk(child, depth + 1)
        walk(getattr(node, "content", None), depth + 1)

    walk(control)
    return found


def _armed_rewards(session, value: float = 10.0, reach: float = 8.0):
    """Two settled rewards reading `value` against a higher line at `reach`, which is the state the
    detector is supposed to fire on. Set rather than trained, because what is under test is the gate.

    The channels are declared live because the bench recording is a no-electrode capture and the device
    says so — which the detector correctly refuses to announce anything on."""
    session.state = dict.fromkeys(session.specs, True)
    for tgt in session.rewards:
        tgt.threshold, tgt.reach_threshold = reach / 2.0, reach
        # Past its first window, whichever mode it is in: a tracker is provisional until the window has
        # filled once, and the detector refuses to announce on a provisional line.
        tgt._rate_hz = 20.0
        tgt.value, tgt.total, tgt._settled = value, int(tgt.window * 20.0) + 100, True
    return session.rewards


def _coupled_signal(seconds: float, depth: float, rate: float, seed: int = 0):
    """A slow rhythm whose phase decides when a fast one is loud, plus noise.

    Narrowband noise rather than sines, so the slow phase wanders as a real rhythm's does — with a
    perfect sine the surrogate floor cannot work, because shifting a periodic signal only rotates it.
    See the note at the top of `tests/test_coupling.py`.
    """
    import numpy as np
    from scipy import signal as sp

    rng = np.random.default_rng(seed)
    n = int(seconds * rate)

    def band(low, high):
        sos = sp.butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
        y = sp.sosfiltfilt(sos, rng.standard_normal(n))
        return y / y.std()

    slow = 30 * band(4.0, 8.0)
    gate = 1.0 - depth * np.cos(np.angle(sp.hilbert(slow)))
    return slow + gate * 5 * band(30.0, 45.0) + 4 * rng.standard_normal(n)


def _run_coupling(depth: float, seconds: float = 90.0):
    from nfb.stream import Block
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=1)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.paused = False
    session.state = dict.fromkeys(session.specs, True)
    spec = session.specs[0]
    x = _coupled_signal(seconds, depth, spec.rate_hz)
    for i in range(0, x.size - 249, 250):
        session.feed_block(Block(0, i, to_counts(x[i:i + 250], spec.scale), spec))
    session.redraw()
    return session, page, control


def _find_pills(control, label, depth: int = 0):
    import flet as ft

    if depth > 14 or control is None:
        return None
    if isinstance(control, ft.Container) and getattr(control, "on_click", None) is not None:
        for text in _texts(control):
            if text.value == label:
                return control
    for child in (getattr(control, "controls", None) or []):
        found = _find_pills(child, label, depth + 1)
        if found is not None:
            return found
    return _find_pills(getattr(control, "content", None), label, depth + 1)


def _locked_pair(seconds, strength, rate, seed=0):
    """Two channels sharing an alpha phase relationship to a controlled degree."""
    import numpy as np
    from scipy import signal as sp

    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    sos = sp.butter(4, [8.0, 12.0], btype="bandpass", fs=rate, output="sos")

    def band():
        y = sp.sosfiltfilt(sos, rng.standard_normal(n))
        return y / y.std()

    a = band()
    shared = sp.sosfiltfilt(sos, np.imag(sp.hilbert(a)))
    shared /= shared.std()
    b = strength * shared + np.sqrt(max(0.0, 1 - strength ** 2)) * band()
    return 20 * a + 3 * rng.standard_normal(n), 20 * b + 3 * rng.standard_normal(n)


def _run_locking(strength: float, seconds: float = 90.0):
    from nfb.stream import Block
    from nfb_console import pac_view

    session = pac_view.session_for(BY_CODE["PAC"], source=replay(), channels=2)
    page = StubPage()
    control = pac_view.view(page, session, toolbar(), lambda: None)
    session.paused = False
    session.state = dict.fromkeys(session.specs, True)
    spec = session.specs[0]
    a, b = _locked_pair(seconds, strength, spec.rate_hz)
    for i in range(0, a.size - 249, 250):
        session.feed_block(Block(0, i, to_counts(a[i:i + 250], spec.scale), session.specs[0]))
        session.feed_block(Block(1, i, to_counts(b[i:i + 250], spec.scale), session.specs[1]))
    session.redraw()
    return session, control


def _menu_clicks(control, label, depth: int = 0, found=None):
    """Every `on_click` for a menu item with this label, in the order they appear.

    A list rather than one, because a label is not unique across a screen — "beta" is in the coupling
    card's fast-band menu *and* in the locking card's band menu, which is correct on screen and
    ambiguous to a test. Callers take the one belonging to the card they mean.
    """
    import flet as ft

    found = [] if found is None else found
    if depth > 16 or control is None:
        return found
    if isinstance(control, ft.PopupMenuItem):
        if any(x.value == label for x in _texts(control)) and control.on_click is not None:
            found.append(control.on_click)
    for attr in ("controls", "items"):
        for child in (getattr(control, attr, None) or []):
            _menu_clicks(child, label, depth + 1, found)
    _menu_clicks(getattr(control, "content", None), label, depth + 1, found)
    return found


def _find_menu_items(control, labels):
    return {label: clicks[0] for label in labels
            for clicks in [_menu_clicks(control, label)] if clicks}


# --- every screen, built once for the tests that only read it -------------------------------

FED_SECONDS = 30.0

# Screen module, protocol code, channels. Cold *and* fed, because most layout faults only show in one of
# the two: a threshold pinned to the top of its plot needs no data at all, and an axis whose labels
# collide needs some.
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
# The screens that read another device. They are built cold only: the feed below is EEG blocks.
OTHER_SCREENS = [("hrv_view", "HRV", 0), ("heg_view", "HEG", 0), ("pupil_view", "PUPIL", 0),
                 ("pair_view", "PAIR", 0)]


def screen_params(screens):
    """The screens as test parameters. The tests of one screen run in one worker process (`xdist_group`),
    so they share its build."""
    import pytest

    return [pytest.param(module, code, channels, id=f"{code}-{channels}ch",
                         marks=pytest.mark.xdist_group(f"{code}-{channels}ch"))
            for module, code, channels in screens]


_built: dict[tuple, tuple] = {}


def built_screen(module: str, code: str, channels: int, fed: bool):
    """A screen as the operator meets it: built, fed if asked, drawn. Returns `(page, control)`.

    **Built once and shared by every test that asks for it, so a test must only read it.** A screen costs
    a quarter of a second to build and feed, and the layout rules and the text rules read the same nine
    screens in the same two states. A test that presses something that changes the screen builds its own.

    **Fed with plausible signal, not the bench recording.** The recording is open inputs railed to full
    scale, so every band reads zero and a fed screen is a cold one with a different name. The same
    generator feeds `tools/shots.py`, so a failure here and a picture of it are of the same screen.
    """
    import importlib

    from nfb.stream import Block

    key = (module, code, channels, fed)
    if key not in _built:
        screen = importlib.import_module(f"nfb_console.{module}")
        session = screen.session_for(BY_CODE[code], source=replay(), channels=channels)
        page = StubPage()
        control = screen.view(page, session, toolbar(), lambda: None)
        if fed:
            session.paused = False
            session.state = dict.fromkeys(session.specs, True)
            waves = {ch: plausible_eeg(FED_SECONDS, session.rate, ch) for ch in session.specs}
            for i in range(0, int(FED_SECONDS * session.rate) - 249, 250):
                for ch, spec in session.specs.items():
                    session.feed_block(Block(ch, i, to_counts(waves[ch][i:i + 250], spec.scale), spec))
        session.redraw()
        _built[key] = (page, control, session)
    return _built[key][:2]


def stop_built_screens() -> None:
    while _built:
        _built.popitem()[1][2].stop()
