"""Serve every screen as a web page, so something can take its picture.

`tests/test_layout` checks the control tree, which catches the faults that *are* in the objects — and
most of them are. This is the other half: the ones that are only visible, like three pills used where
the design wanted one segmented control. No rule will ever catch that; somebody has to look.

**How it works.** `ft.run` serves the console over HTTP, so it renders through the same Skia engine the
desktop build uses at a viewport a script can set. Nothing opens a window — see the note above
`webbrowser` — because the capture script brings its own headless browser. Each shot is a *route* — `/PAC/fed/4` — and one
server answers all of them, because paying Flet's startup per image would make a full sweep a minute of
waiting. `capture.py` drives it.

**Readiness is signalled, not slept on.** Flutter paints to a canvas, so there is no element to wait for
and a fixed delay is a guess that fails on a loaded machine. The page sets its *title* to `ready:<route>`
once it has drawn, and the capture script waits for exactly that. A shot that never arrives fails loudly
instead of photographing a half-painted frame.

**What it is not.** The web build is the same engine but not the same host: fonts come from the browser
rather than from the system, and text metrics moving by a few percent moves exactly the wrapping
thresholds worth hunting. Treat a shot as "does this look right", not as "this is what Paul sees".
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("NFB_NO_OUTPUTS", "1")     # a screenshot must never take the volume or the screen

import webbrowser
from urllib.parse import unquote

# **Nothing opens a window.** `AppView.WEB_BROWSER` is what makes Flet serve over HTTP rather than open a
# desktop app, and it also means what it says — `flet.utils.browser.open_in_browser` calls
# `webbrowser.open` on the way up. The capture script brings its own headless browser, so the only thing
# that did was throw a window in the operator's face every time a shot was taken. Suppressed here rather
# than through the `BROWSER` environment variable, which is a global setting to solve a local problem.
webbrowser.open = lambda *_a, **_k: True

import flet as ft

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from nfb.stream import Block                                          # noqa: E402
from nfb_console import brand, chrome, splash                         # noqa: E402
from nfb_console.protocols import BY_CODE                             # noqa: E402

from conftest import BENCH_OK, counts, plausible_eeg                  # noqa: E402

FED_SECONDS = 150.0

SCREENS = {"AMP": "amplitude_view", "A-T": "at_view", "CFC": "cfc_view", "ILF": "ilf_view",
           "PAC": "pac_view", "SCP": "scp_view", "SLEEP": "sleep_view", "WAVE": "wave_view",
           "HEG": "heg_view", "HRV": "hrv_view", "PUPIL": "pupil_view"}

# Cold is not an edge case here, it is the first twenty seconds of every session — and it is where most
# of the faults have been, because it is the state nobody opens the app to look at.
STATES = ("cold", "fed")

SHOTS = [("launcher", "", 0)] + [
    (code, state, channels)
    for code in ("AMP", "CFC", "PAC", "ILF", "A-T", "SCP", "WAVE")
    for state in STATES
    for channels in ((1, 4) if code == "PAC" else (2,))
] + [("HRV", "cold", 0), ("HEG", "cold", 0)] + [     # no EEG to feed them; their own devices drive them
    ("PUPIL", state, 0) for state in STATES] + [
    ("device", kind, 3) for kind in ("simulator", "tracker")]     # the Pupil tab, with each source


def route_of(shot) -> str:
    code, state, channels = shot
    if code == "device":
        return f"/device/{channels}/{state}"
    return "/launcher" if code == "launcher" else f"/{code}/{state}/{channels}"


class _Device:
    """Enough of a device for the toolbar and the launcher chip. Never opens a port."""

    class _Settings:
        hz, reference, mains_hz = 250.0, "linked", 50

    class _Status:
        from nfb import Mode
        mode, failure, remedy, model, connected = Mode.DISCONNECTED, None, "", "", False
        settings = None

    def __init__(self):
        self._Status.settings = self._Settings()
        self.status = self._Status()

    def subscribe(self, _name):
        raise RuntimeError("shots never use the device")


def build_screen(code: str, state: str, channels: int, page: ft.Page):
    module = importlib.import_module(f"nfb_console.{SCREENS[code]}")
    from nfb import ReplaySource
    session = module.session_for(BY_CODE[code], source=ReplaySource(str(BENCH_OK), realtime=False),
                                 channels=channels)
    bar = chrome.Toolbar(_Device(), lambda: None, lambda: None, lambda _n: None)
    control = module.view(page, session, bar, lambda: None)
    if state == "fed" and code == "PUPIL":
        feed_pupil(session)
        session.redraw()
    elif state == "fed":
        # Fed here rather than by the session's own thread: the whole run in a moment, so a settled
        # screen costs no wall clock and the shot of one is reproducible.
        session.paused = False
        session.state = dict.fromkeys(session.specs, True)
        rate = session.rate
        waves = {ch: plausible_eeg(FED_SECONDS, rate, ch) for ch in session.specs}
        for i in range(0, int(FED_SECONDS * rate) - 249, 250):
            for ch, spec in session.specs.items():
                session.feed_block(Block(ch, i, counts(waves[ch][i:i + 250], spec.scale), spec))
        session._stop = True
        session.redraw()
    control._shot_redraw = session.redraw
    return control


def feed_pupil(session, seconds: float = 75.0) -> None:
    """A simulated eye run to the middle of a trial's effort window, at once. The session observes every
    sixth reading, its own 10 Hz, as it does on screen."""
    from nfb.pupil import SimulatedPupil

    eye = SimulatedPupil(seed=4, realtime=False, response_mm=0.35)
    session.eye.source = eye
    session.paused = False
    start = None
    for i, sample in enumerate(eye.samples()):
        start = sample.t if start is None else start
        if sample.t - start > seconds:
            break
        session.eye.state = session.eye.processor.push(sample)
        if i % 6 == 0:
            session.observe()
    session.paused = True       # the shot is a still: nothing may move the frame after it is drawn
    session.eye.source = None


_runners: list = []


def build_device(page: ft.Page, tab: int, pupil_kind: str):
    """The device screen on one tab. The pupil source is started for real, so its tab shows what it says:
    the simulator's readings, or the tracker's reason for not connecting."""
    import time

    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import PupilChoice
    from nfb_console.pupil import PupilRunner

    for runner in _runners:
        runner.release()
    _runners.clear()
    choice = PupilChoice(kind=pupil_kind)
    runner = PupilRunner(choice)
    _runners.append(runner)
    runner.attach(choice)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (runner.ready or runner.error or not choice.on):
        time.sleep(0.05)
    control = device_panel.view(page, Device(Settings()), pupil_choice=choice, pupil_runner=runner)
    stack = [control]
    while stack:
        ctrl = stack.pop()
        if isinstance(ctrl, ft.Stack) and len(ctrl.controls) == len(device_panel.MODALITIES):
            for i, pane in enumerate(ctrl.controls):
                pane.visible = i == tab
            break
        stack += list(getattr(ctrl, "controls", None) or [])
        content = getattr(ctrl, "content", None)
        if content is not None and not isinstance(content, str):
            stack.append(content)
    return control


def open_help(control, title: str) -> None:
    """Press the `?` whose dialog is called `title`. A help button carries its title as its tooltip."""
    stack = [control]
    while stack:
        ctrl = stack.pop()
        if (getattr(ctrl, "tooltip", None) == title and getattr(ctrl, "on_click", None)
                and getattr(getattr(ctrl, "content", None), "value", None) == "?"):
            ctrl.on_click(None)
            return
        stack += list(getattr(ctrl, "controls", None) or [])
        content = getattr(ctrl, "content", None)
        if content is not None and not isinstance(content, str):
            stack.append(content)
    raise LookupError(f"no help button called {title!r} on this screen")


def main() -> None:
    def build(page: ft.Page) -> None:
        page.padding = 0
        page.bgcolor = "#f1f1ee"
        page.fonts = brand.FONTS

        def show(_e=None) -> None:
            route = page.route or "/launcher"
            parts = [p for p in route.split("/") if p]
            page.controls.clear()
            if parts[:1] == ["launcher"]:
                page.controls.append(splash.view(page, _Device(), lambda _p: None, lambda: None))
            elif parts[:1] == ["device"]:
                page.controls.append(build_device(page, int(parts[1]), parts[2] if len(parts) > 2 else "off"))
            else:
                code, state, channels = parts[0], parts[1], int(parts[2])
                control = build_screen(code, state, channels, page)
                extra = parts[3:]
                help_title = next((unquote(p[5:]) for p in extra if p.startswith("help:")), None)
                tabs = [p for p in extra if not p.startswith("help:")]
                # An optional fourth part names a tab by index: `/PAC/fed/4/3` is the Feedback tab. The
                # screen's own tab bar is a closure, so the panes are switched directly — they are the
                # third child of every protocol screen, and visibility is all the tab bar sets.
                if tabs:
                    panes = control.controls[2].controls
                    for i, pane in enumerate(panes):
                        pane.visible = i == int(tabs[0])
                    # A hidden pane costs nothing, which means it drew nothing: redraw now that it shows.
                    session_redraw = getattr(control, "_shot_redraw", None)
                    if session_redraw is not None:
                        session_redraw()
                page.controls.append(control)
                if help_title is not None:
                    open_help(control, help_title)
            page.update()
            page.title = f"ready:{route}"
            page.update()

        page.on_route_change = show
        show()

    ft.run(build, view=ft.AppView.WEB_BROWSER, assets_dir=brand.ASSETS,
           port=int(os.environ.get("SHOTS_PORT", "8553")), web_renderer=ft.WebRenderer.CANVAS_KIT)


if __name__ == "__main__":
    main()
