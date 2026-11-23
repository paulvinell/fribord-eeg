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

import numpy as np                                                    # noqa: E402

from nfb.dsp import Planted, plant                                    # noqa: E402
from nfb.stream import Block                                          # noqa: E402
from nfb_console import brand, chrome, splash                         # noqa: E402
from nfb_console.protocols import BY_CODE                             # noqa: E402
from nfb_console.trials import ACTIVE, BASELINE, ITI, RESULT          # noqa: E402

from conftest import BENCH_OK, counts, plausible_eeg                  # noqa: E402

FED_SECONDS = 150.0
EARLY_SECONDS = 3.0            # how far into its search the early spindle shot is caught

SCREENS = {"AMP": "amplitude_view", "A-T": "at_view", "CFC": "cfc_view", "ILF": "ilf_view",
           "PAC": "pac_view", "SCP": "scp_view", "SLEEP": "sleep_view", "WAVE": "wave_view",
           "HEG": "heg_view", "HRV": "hrv_view", "PUPIL": "pupil_view", "PAIR": "pair_view"}

# Cold is not an edge case here, it is the first twenty seconds of every session — and it is where most
# of the faults have been, because it is the state nobody opens the app to look at.
STATES = ("cold", "fed")
SPINDLE_STATES = ("spindle", "search", "search-early")

SHOTS = [("launcher", "", 0)] + [
    (code, state, channels)
    for code in ("AMP", "CFC", "PAC", "ILF", "A-T", "SCP", "WAVE")
    for state in STATES
    for channels in ((1, 4) if code == "PAC" else (2,))
] + [
    # The spindle card, fed with spindles planted on ch0; a search over the whole run; a search 3 s old.
    ("AMP", state, 2) for state in SPINDLE_STATES] + [("HRV", "cold", 0)] + [("HEG", state, 0) for state in STATES] + [   # their own devices drive them
    ("PUPIL", state, 0) for state in STATES] + [
    # The arcs at a typed **succeeds at**, in an effort and at a verdict, on the two other trial programs.
    (code, state, 0) for code in ("PUPIL", "HEG") for state in ("arc", "verdict")] + [
    ("PAIR", state, 0) for state in STATES] + [
    ("device", "off", 0)] + [   # the EEG tab: input, connection, acquisition, contact
    ("device", kind, 3) for kind in ("simulator", "tracker")] + [   # the Pupil tab, with each source
    # The settings menu over the launcher: the developer section closed, open with the outputs row
    # locked by NFB_NO_OUTPUTS as this server runs, and open as the console shows it without the variable.
    ("settings", state, 0) for state in ("closed", "open", "free")]


def route_of(shot) -> str:
    code, state, channels = shot
    if code == "device":
        return f"/device/{channels}/{state}"
    if code == "settings":
        return f"/settings/{state}"
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
    heart_kwargs = {}
    if code in ("PUPIL", "SCP") and state == "fed":
        # Give the fed pupil and slow-potential shots a heart source, so the heart panel and its stats appear in the frame.
        from nfb_console import heart as heart_mod
        from nfb_console.inputs import HeartChoice
        hc = HeartChoice(kind="simulator")
        heart_kwargs = dict(heart_choice=hc, heart_runner=heart_mod.HeartRunner(hc))
    if code == "PAIR" and state == "fed":
        session = _paired_session(module)
    elif code == "HEG" and state in ("fed", "arc", "verdict"):
        session = _heg_trials(module, state)
    elif code == "SCP" and state in _SCP:
        # The SCP frames are fed through `feed_block`, after the shots that close this file register them.
        try:
            session = _scp_feed(module, state, channels, **heart_kwargs)
        except ScpWaits as e:
            return _waits_control(state, channels, str(e))
    else:
        session = module.session_for(BY_CODE[code], source=ReplaySource(str(BENCH_OK), realtime=False),
                                     channels=channels, **heart_kwargs)
    if code == "AMP" and state in SPINDLE_STATES:
        session.add_spindles(0)
        if state == "search":
            session.spindles.start_search()
    if code == "PUPIL" and state in ("arc", "verdict"):
        # A typed **succeeds at** of 0.15 mm, so the arcs show; set before the view builds its boxes.
        session.set_mode("active")
        session.set_success_at(0.15, typed=True)
        session.restart_trials()
    bar = chrome.Toolbar(_Device(), lambda: None, lambda: None, lambda _n: None)
    control = module.view(page, session, bar, lambda: None)
    if state == "fed" and code == "PUPIL":
        feed_pupil(session)
        session.redraw()
    elif code == "PUPIL" and state in ("arc", "verdict"):
        # In the third trial's effort, or at its verdict.
        run = lambda: session.trials                                  # noqa: E731
        feed_pupil(session, until=(
            (lambda s: len(run().results) >= 2 and run().phase == ACTIVE
             and run().active_s - run().phase_remaining() > 4.0) if state == "arc"
            else (lambda s: len(run().results) >= 3 and run().phase == RESULT)))
        session.redraw()
    elif code == "SCP" and state in _SCP:
        session.redraw()
    elif code == "HEG" and state in ("fed", "arc", "verdict"):
        session.redraw()
    elif state == "fed" and code == "PAIR":
        session.redraw()
    elif state in ("fed", *SPINDLE_STATES):
        # Fed here rather than by the session's own thread: the whole run in a moment, so a settled
        # screen costs no wall clock and the shot of one is reproducible.
        session.paused = False
        session.state = dict.fromkeys(session.specs, True)
        rate = session.rate
        waves = {ch: plausible_eeg(FED_SECONDS, rate, ch) for ch in session.specs}
        if state in SPINDLE_STATES:
            plant(waves[0], rate, np.random.default_rng(5),
                  Planted(23.0, (0.25, 0.4), 30.0, per_minute=12, jitter_hz=2.0))
        for i in range(0, int(FED_SECONDS * rate) - 249, 250):
            for ch, spec in session.specs.items():
                session.feed_block(Block(ch, i, counts(waves[ch][i:i + 250], spec.scale), spec))
        if state == "search-early":
            # Started after the fed run, so the few spindles on its line are its own.
            session.spindles.start_search()
            start = int(FED_SECONDS * rate) - 250
            more = {ch: plausible_eeg(EARLY_SECONDS, rate, ch, seed=1) for ch in session.specs}
            plant(more[0], rate, np.random.default_rng(6),
                  Planted(23.0, (0.25, 0.4), 30.0, per_minute=40, jitter_hz=2.0))
            for i in range(0, int(EARLY_SECONDS * rate) - 249, 250):
                for ch, spec in session.specs.items():
                    session.feed_block(Block(ch, start + i, counts(more[ch][i:i + 250], spec.scale), spec))
        if state.startswith("search"):
            session.spindles.estimate()
        session._stop = True
        session.redraw()
    control._shot_redraw = session.redraw
    return control


def _paired_session(module):
    """Both heart sensors on one synthetic heart, compared every 3 s over a minute, at once: the same
    two-site heart the comparison's tests use."""
    import types

    from test_hrv_compare import _scene
    from test_pair_view import _Runner

    _, forehead, reference, f_wave, r_wave, *_ = _scene()
    on = types.SimpleNamespace(on=True)
    session = module.session_for(BY_CODE["PAIR"], heart_choice=on, mendi_choice=on,
                                 heart_runner=_Runner(reference, r_wave, rmssd=68.0),
                                 mendi_runner=_Runner(forehead, f_wave, rmssd=66.0))
    session.paused = False
    for now in range(36, 91):
        session._accumulated = now - 36.0
        session.update(now=float(now))
        session.observe()
    session.paused = True       # set, not toggled: a still keeps the level it reached
    return session


def _heg_trials(module, state: str = "fed"):
    """HEG in active mode, a minute of signal and then trials until the third is under way, at once.
    `arc` types a **succeeds at** of 1 sd, so its arcs show in the window; `verdict` stops at the third
    trial's verdict, with the arcs and the two lines under the circle."""
    from nfb_console.inputs import MendiChoice
    from test_heg_trials import _push

    session = module.session_for(BY_CODE["HEG"], mendi_choice=MendiChoice(kind="simulator"))
    session.cue = lambda *_a, **_k: True
    session.set_mode("active")
    if state in ("arc", "verdict"):
        session.set_success_at(1.0, typed=True)
    session.paused = False
    t = _push(session, 70.0, 1000.0)
    session.restart_trials()
    while len(session.trials.results) < 2 or session.trials.phase != "active":
        t = _push(session, 0.5, t, rise=0.6 if len(session.trials.results) else 1.2)
    if state == "verdict":
        while session.trials.phase != "result":
            t = _push(session, 0.2, t, rise=1.4)
        t = _push(session, 0.5, t)
    else:
        t = _push(session, 8.0, t, rise=1.0)
    session.paused = True       # the shot is a still
    return session


# --- SCP, driven through `feed_block` -------------------------------------------------------------

class ScpWaits(Exception):
    """A state that needs a name the product code does not have yet, or a run that missed its state."""


_SCP: dict = {}


def _scp_state(name: str, *requires: str, channels: int = 2):
    """Register a state's driver with the names it needs: `Class.attr`, or `Class.method(keyword)`."""
    def register(fn):
        _SCP[name] = (requires, fn, channels)
        return fn
    return register


def _present(name: str) -> bool:
    import inspect

    from nfb_console import scp_view, trial_training, trials

    where = {"SCPSession": scp_view.SCPSession, "TrialTraining": trial_training.TrialTraining,
             "Trial": trials.Trial, "trial_summary": trial_training.trial_summary}
    path, _, keyword = name.partition("(")
    head, *attrs = path.split(".")
    obj = where[head]
    for attr in attrs:
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return not keyword or keyword.rstrip(")") in inspect.signature(obj).parameters


def waiting(shot) -> list[str]:
    """The names an SCP state needs and does not find. Empty when the state can be drawn."""
    code, state, _ = shot
    if code != "SCP" or state not in _SCP:
        return []
    return [name for name in _SCP[state][0] if not _present(name)]


class ScpDrive:
    """One SCP session fed through `feed_block`, 50 samples a channel in turn, on wandering slow levels
    under plausible EEG. Seeded, so a state lands on the same sample every time; no clock."""

    BLOCK = 50
    STEP_UV = 2.5              # the 1 s change of each channel's slow level
    TRIAL_S = 16.0             # the longest trial and its rest, with room

    def __init__(self, module, channels: int, seed: int = 5, **heart_kwargs):
        from nfb import ReplaySource

        self.session = module.session_for(BY_CODE["SCP"], source=ReplaySource(str(BENCH_OK), realtime=False),
                                          channels=channels, **heart_kwargs)
        self.session.cue = lambda *_a, **_k: True
        self.session.paused = False
        self.seed = seed
        self.at = 0
        self.slow: dict = {}
        self.fast: dict = {}
        self.skill = lambda _d, _ch: 0.0                  # µV the trainee adds to a channel's block now
        self.shapes: dict = {}                            # channel -> fn(drive, µV) -> µV
        self.steps: list = []                             # fn(drive) before each round of blocks
        self.gone = False                                 # a step sets it: this round of blocks never arrives
        self.marks: dict = {}                             # sample indices a step remembers
        self.reseed()

    @property
    def run(self):
        return self.session.trials

    def reseed(self) -> None:
        """A seeded copy of the run the session built, so every setting it carries is kept."""
        import dataclasses

        self.session.trials = dataclasses.replace(self.session.trials, seed=self.seed)

    def toward(self, size_uv: float, rise_s: float = 1.2, follow: dict | None = None):
        """The trainee: a shift toward the cue on the training channel, rising over `rise_s`. `follow` gives
        other channels the same shift at their own size, as a source that reaches them does."""
        import math

        from conftest import cue_locked

        sizes = {**(follow or {}), "training": size_uv}

        def skill(d, ch):
            size = sizes["training"] if ch == d.session.channel else sizes.get(ch)
            if size is None or d.run.phase != ACTIVE:
                return 0.0
            tau = d.run.active_s - d.run.phase_remaining()
            return cue_locked(d.run, size) * (1.0 - math.exp(-tau / rise_s))
        self.skill = skill

    def _signal(self) -> None:
        from conftest import wandering

        rate = self.session.rate
        seconds = self.session.n_trials * max(self.TRIAL_S, self.session.trial_seconds + 4.0) + 40.0
        for ch in self.session.specs:
            self.slow[ch] = wandering(1.5, seconds, rate, 100 * self.seed + ch, self.STEP_UV)
            self.fast[ch] = 0.5 * plausible_eeg(seconds, rate, ch, seed=self.seed)

    def feed(self, until) -> None:
        """Feed until `until(drive)` holds. Raises if the signal ends first."""
        import optima

        if not self.slow:
            self._signal()
        session, n = self.session, len(next(iter(self.slow.values())))
        while self.at + self.BLOCK <= n:
            if until(self):
                return
            for step in self.steps:
                step(self)
            for ch, spec in ({} if self.gone else session.specs).items():
                part = slice(self.at, self.at + self.BLOCK)
                uv = self.slow[ch][part] + self.fast[ch][part] + self.skill(self, ch)
                if ch in self.shapes:
                    uv = self.shapes[ch](self, uv)
                full = abs(spec.scale) * optima.ADC_FULL_SCALE
                session.feed_block(Block(ch, self.at, tuple(counts(np.clip(uv, -full, full), spec.scale)
                                                            .tolist()), spec))
            self.at += self.BLOCK
        raise ScpWaits("the signal ended before the state was reached")

    def into(self, phase: str, seconds: float, results: int):
        """A condition: `results` trials done, and `seconds` into `phase`."""
        def reached(d) -> bool:
            run = d.run
            length = {ITI: 2.0, BASELINE: run.baseline_s, ACTIVE: run.active_s, RESULT: run.result_s}[phase]
            return (len(run.results) >= results and run.phase == phase
                    and length - run.phase_remaining() >= seconds)
        return reached

    def trial(self, k: int, phase: str, seconds: float) -> None:
        """Stop `seconds` into `phase` of trial `k`, counted from 1."""
        self.feed(self.into(phase, seconds, k if phase == RESULT else k - 1))

    def to_end(self) -> None:
        self.feed(lambda d: d.run.done)

    def expect(self, holds: bool, what: str) -> None:
        if not holds:
            raise ScpWaits(f"the run did not land: {what}")

    def in_trial(self, k: int, phase: str, start: float = 0.0, stop: float = 99.0):
        """A condition on the block: inside `phase` of trial `k`, from `start` to `stop` seconds in."""
        def inside(d) -> bool:
            run = d.run
            if run.phase != phase or len(run.results) != k - 1:
                return False
            length = {BASELINE: run.baseline_s, ACTIVE: run.active_s}[phase]
            return start <= length - run.phase_remaining() < stop
        return inside


def _scp_feed(module, state: str, channels: int, **heart_kwargs):
    """The session in `state`, or `ScpWaits` naming what is missing."""
    missing = waiting(("SCP", state, channels))
    if missing:
        raise ScpWaits(f"waits for {', '.join(missing)}")
    drive = ScpDrive(module, channels, **heart_kwargs)
    _SCP[state][1](drive)             # a product error raises, so the capture fails instead of drawing it
    drive.session.paused = True       # the shot is a still
    return drive.session


def _waits_control(state: str, channels: int, reason: str) -> ft.Control:
    return ft.Container(content=ft.Text(f"SCP {state} at {channels} channels: {reason}", size=16,
                                        color="#a33a2a", selectable=True), padding=40)


def _phases(d) -> None:
    """The run of the trial frames: no trials without feedback, so each picture is the one with a circle."""
    d.session.transfer_fraction = 0.0
    d.session.restart_trials()
    d.reseed()


def _spikes(_d, uv):
    """A sharp artifact in the middle of the block, which the gate rejects."""
    uv = uv.copy()
    uv[20:24] += 400.0
    return uv


def _touch_limit(_d, uv):
    """The block's last sample at the input limit, the rest a signal."""
    uv = uv.copy()
    uv[-1] = -800.0
    return uv


def _rail(_d, uv):
    return np.full_like(uv, -800.0)


def _flat(_d, uv):
    return np.zeros_like(uv)


def _when(condition, shape):
    return lambda d, uv: shape(d, uv) if condition(d) else uv


def _since_active(d, k: int = 3) -> float | None:
    """Seconds of samples since trial `k`'s feedback window opened, by sample index, so a hole or a pause
    in the readings does not stop the count."""
    if "active" not in d.marks and d.in_trial(k, ACTIVE)(d):
        d.marks["active"] = d.at
    start = d.marks.get("active")
    return None if start is None else (d.at - start) / d.session.rate


def _holes(d) -> None:
    """No blocks for three stretches of 0.8 s in the third feedback window: lost, each under a second with
    the reading stride on top, and 30% of the trial together."""
    gone = _since_active(d)
    d.gone = gone is not None and any(start <= gone < start + 0.8 for start in (1.1, 2.7, 4.3))


def _pause(d) -> None:
    """The operator pauses 2 s into the third feedback window, and resumes 1.5 s later."""
    gone = _since_active(d)
    if d.session.paused != (gone is not None and 2.0 <= gone < 3.5):
        d.session.toggle_pause()        # the transport's path, which cuts the trial in flight


def _hold(k: int):
    """From trial `k`'s cue the training channel's slow level stays where it was, so the trial's result is
    the trainee's shift alone and lands where the frame needs it."""
    def shape(d, uv):
        if "cue" not in d.marks and len(d.run.results) == k - 1 and d.run.phase == BASELINE:
            d.marks["cue"] = d.at
        if "cue" not in d.marks or len(d.run.results) >= k:
            return uv
        slow = d.slow[d.session.channel]
        return uv - slow[d.at:d.at + len(uv)] + slow[d.marks["cue"]]
    return shape


def _toward_limit(d, factor: float) -> None:
    """The trainee holds `factor` times the trial's frozen **succeeds at**, toward the cue."""
    from conftest import cue_locked

    def skill(d_, ch):
        scale = d_.session.trial_scale
        if ch != d_.session.channel or d_.run.phase != ACTIVE or scale is None:
            return 0.0
        return cue_locked(d_.run, factor * scale.limit)
    d.skill = skill


for _name, _phase, _seconds, _size in (
        ("rest", ITI, 1.5, 15.0), ("baseline", BASELINE, 1.0, 15.0), ("feedback", ACTIVE, 4.5, 15.0),
        ("wrong", ACTIVE, 4.5, -30.0), ("off", ACTIVE, 4.5, 15.0), ("success", RESULT, 0.5, 15.0),
        ("failure", RESULT, 0.5, -30.0)):
    def _trial_frame(d, name=_name, phase=_phase, seconds=_seconds, size=_size) -> None:
        """Two trials of history, then the third the cued way, or 30 µV the wrong way."""
        _phases(d)
        if name == "off":
            d.session.feedback_window_s = 3.0
        d.toward(15.0)
        d.feed(lambda d_: len(d_.run.results) >= 2)
        if name == "success":
            d.shapes[d.session.channel] = _hold(3)     # the slow level still, so the shift is the trainee's
        d.toward(size)
        d.trial(3, phase, seconds)
        last = d.run.results[-1]
        if name == "success":
            d.expect(last.success, "the third trial succeeds")
        if name == "failure":
            d.expect(last.valid and not last.success, "the third trial fails")
    _scp_state(_name)(_trial_frame)


@_scp_state("fed")
def _fed(d) -> None:
    """The run as it opens, a trainee with some control, stopped where it is after FED_SECONDS."""
    if d.session.heart is not None:
        _warm_heart(d.session)
    d.toward(8.0)
    d.feed(lambda d_: d_.at >= FED_SECONDS * d_.session.rate)


@_scp_state("history")
def _history(d) -> None:
    d.toward(8.0)
    d.feed(d.into(ITI, 1.5, 12))


@_scp_state("one-direction")
def _one_direction(d) -> None:
    d.session.set_trial_direction("increase")
    d.reseed()
    d.toward(8.0)
    d.trial(6, ACTIVE, 4.5)


@_scp_state("extras-1-open", channels=1)
def _one_open(d) -> None:
    _phases(d)
    d.toward(15.0)
    d.trial(3, ACTIVE, 4.5)


@_scp_state("limit", "SCPSession.at_input_limit")
def _limit(d) -> None:
    """The training channel touches the input limit in the rest after two trials."""
    _phases(d)
    d.toward(15.0)
    ch = d.session.channel
    d.shapes[ch] = _when(lambda d_: len(d_.run.results) >= 2, _touch_limit)
    d.feed(d.into(ITI, 1.5, 2))
    d.expect(d.session.at_input_limit(ch), "the training channel is at the input limit")


def _void(cause: str, shape=None, step=None, *more: str) -> None:
    """The third trial is void for `cause` alone, and the frame is its verdict."""
    def frame(d) -> None:
        _phases(d)
        d.toward(15.0)
        if shape is not None:
            d.shapes[d.session.channel] = shape
        if step is not None:
            d.steps.append(step)
        d.feed(lambda d_: len(d_.run.results) >= 3)
        if d.run.phase == RESULT:
            d.feed(d.into(RESULT, 0.5, 3))
        d.expect(d.run.results[-1].cause == cause, f"the third trial is void for {cause}")
    _scp_state(f"void-{cause}", "Trial.cause", "TrialTraining.chance_spread", *more)(frame)


_void("limit", _when(lambda d: d.in_trial(3, ACTIVE, 0.0, 2.5)(d), _touch_limit), None,
      "SCPSession.at_input_limit")
_void("lost", None, _holes)
_void("paused", None, _pause)
_void("baseline", _when(lambda d: d.in_trial(3, BASELINE, 0.0, 1.3)(d), _spikes))
_void("rejected", _when(lambda d: d.in_trial(3, ACTIVE, 0.0, 2.4)(d), _spikes))


def _verdict(name: str, size: float, extra=None, follow: float = 0.0) -> None:
    """A whole run to its end verdict. `extra` lights ch1 with that shape (None: ch1 as it is), and
    `follow` gives ch1 the trainee's shift at that size."""
    def frame(d) -> None:
        if extra is not False:
            d.session.set_extra(1, True)
            if extra is not None:
                d.shapes[1] = extra
        d.toward(size, follow={1: follow} if follow else None)
        d.to_end()
    needs = ("trial_summary(checks)",) + (() if extra is False else ("SCPSession.set_extra",))
    _scp_state(name, *needs)(frame)


_verdict("history-chance", 0.0, extra=False)
_verdict("history-no-extra", 10.0, extra=False)
_verdict("history-against", -10.0, extra=False)
_verdict("history-extra-inside", 10.0)
_verdict("history-extra-followed", 10.0, follow=30.0)
_verdict("history-extra-no-signal", 10.0, extra=_flat)


@_scp_state("full-at-fixed", "TrialTraining.set_full_at(typed)")
def _full_at_fixed(d) -> None:
    _phases(d)
    d.session.set_full_at(25.0, typed=True)
    d.toward(15.0)
    d.trial(3, ACTIVE, 4.5)


@_scp_state("succeeds-fixed", "TrialTraining.set_success_at(typed)")
def _succeeds_fixed(d) -> None:
    _phases(d)
    d.session.set_success_at(4.0, typed=True)
    d.reseed()
    d.toward(15.0)
    d.trial(3, RESULT, 0.5)
    d.expect(d.run.results[-1].success, "the third trial succeeds")


for _name, _phase, _seconds, _factor in (("feedback-limit", ACTIVE, 4.5, 0.5),
                                         ("success-limit", RESULT, 0.5, 1.6),
                                         ("failure-limit", RESULT, 0.5, 0.5)):
    def _limit_frame(d, name=_name, phase=_phase, seconds=_seconds, factor=_factor) -> None:
        """The third trial holds a shift short of **succeeds at**, or past it, with the arc drawn there."""
        _phases(d)
        d.toward(15.0)
        d.feed(lambda d_: len(d_.run.results) >= 2)
        d.shapes[d.session.channel] = _hold(3)
        _toward_limit(d, factor)
        d.trial(3, phase, seconds)
        last = d.run.results[-1]
        if name == "success-limit":
            d.expect(last.success, "the third trial succeeds")
        if name == "failure-limit":
            d.expect(last.valid and not last.success, "the third trial falls short")
    _scp_state(_name, "TrialTraining.chance_spread")(_limit_frame)


@_scp_state("no-spread-first-trial", "TrialTraining.chance_spread")
def _no_spread(d) -> None:
    _phases(d)
    d.session.chance_spread = lambda: None
    d.toward(15.0)
    d.trial(1, ACTIVE, 3.0)


@_scp_state("arc-transfer-verdict", "TrialTraining.chance_spread")
def _arc_transfer(d) -> None:
    d.session.transfer_fraction = 1.0
    d.session.restart_trials()
    d.reseed()
    d.toward(15.0)
    d.trial(3, RESULT, 0.5)
    d.expect(d.run.results[-1].transfer, "the third trial has no feedback")


@_scp_state("arc-circle-off", "TrialTraining.chance_spread")
def _arc_off(d) -> None:
    _phases(d)
    d.session.feedback_window_s = 3.0
    d.toward(15.0)
    d.trial(3, ACTIVE, 4.5)


@_scp_state("arc-limit-above-full-at", "TrialTraining.set_full_at(typed)", "TrialTraining.set_success_at(typed)")
def _arc_above(d) -> None:
    _phases(d)
    d.session.set_full_at(2.0, typed=True)
    d.session.set_success_at(4.0, typed=True)
    d.reseed()
    d.toward(6.0)
    d.trial(3, ACTIVE, 4.5)


def _extras(name: str, channels: int, lit: tuple, shape=None, train=None, *more: str) -> None:
    """Trial 3's feedback with `lit` pressed as extra channels; `shape` for ch1, `train` moves training."""
    def frame(d) -> None:
        for ch in lit:
            d.session.set_extra(ch, True)
        if train is not None:
            d.session.set_channel(train)
        if shape is not None:
            d.shapes[1] = shape
        _phases(d)
        d.toward(15.0)
        d.trial(3, ACTIVE, 4.5)
    _scp_state(name, "SCPSession.set_extra", *more, channels=channels)(frame)


_extras("extras-2-open", 2, (1,))
_extras("extras-4-open-two-lit", 4, (1, 2, 3))
_extras("extra-at-limit", 2, (1,), _rail, None, "SCPSession.at_input_limit")


def _eyes(d, until_s: float | None) -> None:
    """ch1 lit, the calibration pressed, and an artificial eye that follows its dot on both channels, fed to
    `until_s` seconds into it, or to its end with None."""
    from conftest import eye_source

    from nfb_console import eye_calibration as ec

    place = {ec.CENTRE: (0.0, 0.0), "up": (1.0, 0.0), "down": (-1.0, 0.0), "left": (0.0, -1.0), "right": (0.0, 1.0)}
    session = d.session
    session.set_extra(1, True)
    d.feed(lambda d_: d_.at >= 2.0 * session.rate)
    session.start_calibration()
    start = d.at
    plan = ec.PLANS[session.eye_length]
    v, h, b = eye_source(lambda t: place[plan.dot_at(t)[1]], plan.tones, plan.total_s + 5.0, session.rate)
    share = {0: (6.0, 2.0, 12.0), 1: (60.0, 20.0, 150.0)}        # µV of each source on each channel

    def eye(ch):
        def shape(d_, uv):
            part = slice(d_.at - start, d_.at - start + len(uv))
            sv, sh, sb = share[ch]
            return uv + sv * v[part] + sh * h[part] + sb * b[part]
        return shape
    d.shapes = {0: eye(0), 1: eye(1)}
    if until_s is None:
        d.feed(lambda d_: not session.eye_cal.running)
        d.expect(session.eye_cal.done, f"the calibration ends, not {session.eye_cal.stopped}")
    else:
        d.feed(lambda d_: d_.at - start >= until_s * session.rate)
        d.expect(session.eye_cal.running, "the calibration runs")


def _no_transfer(d) -> None:
    """Every trial with its circle, so the frame of a feedback window shows one."""
    d.session.transfer_fraction = 0.0
    d.session.restart_trials()
    d.reseed()


@_scp_state("trial-length-12", "TrialTraining.set_trial_length")
def _trial_length_12(d) -> None:
    """A trial length of 12 s: the third trial 8 s into its 10 s feedback window."""
    d.session.set_trial_length(12.0)
    _no_transfer(d)
    d.toward(8.0)
    d.trial(3, ACTIVE, 8.0)


@_scp_state("eyes-button", "SCPSession.start_calibration")
def _eyes_button(d) -> None:
    d.session.set_extra(1, True)
    _no_transfer(d)
    d.toward(8.0)
    d.trial(2, ACTIVE, 3.0)


def _settling(rail_until_s: float):
    """An electrode whose standing level sits past the input limit, then settles to -150 µV over a minute."""
    import math

    def shape(d, uv):
        at = d.at / d.session.rate - rail_until_s
        return uv + (-1000.0 if at < 0 else -150.0 - 600.0 * math.exp(-at / 12.0))
    return shape


def _level_held(until_s: float, mid: bool = False):
    """ch1 lit and settling. The calibration is pressed at 20 s, while ch1 is past the limit, or, with
    `mid`, pressed at 2 s with ch1 inside and taken past the limit 6 s into it for 2 s."""
    def frame(d) -> None:
        session = d.session
        session.set_extra(1, True)
        if mid:
            d.feed(lambda d_: d_.at >= 2.0 * session.rate)
            session.start_calibration()
            start = d.at
            d.shapes[1] = lambda d_, uv: uv - (1000.0 if 6.0 <= (d_.at - start) / session.rate < 8.0 else 0.0)
        else:
            d.shapes[1] = _settling(40.0)
            d.feed(lambda d_: d_.at >= 20.0 * session.rate)
            session.start_calibration()
        d.feed(lambda d_: d_.at >= until_s * session.rate)
        d.expect(session.eye_cal is not None and session.eye_cal.held is not None, "the calibration waits")
    return frame


_scp_state("level-press", "SCPSession.continue_calibration")(_level_held(30.0))


@_scp_state("level-spike", "SCPSession.continue_calibration")
def _level_spike(d) -> None:
    """ch1 settled at -450 µV with one sample at the limit at 18 s; the calibration pressed at 19 s."""
    session = d.session
    session.set_extra(1, True)
    spike = int(18.0 * session.rate)

    def shape(d_, uv):
        uv = uv - 450.0
        if d_.at <= spike < d_.at + len(uv):
            uv = uv.copy()
            uv[spike - d_.at] = -1000.0
        return uv
    d.shapes[1] = shape
    d.feed(lambda d_: d_.at >= 19.0 * session.rate)
    session.start_calibration()
    d.feed(lambda d_: d_.at >= 21.0 * session.rate)
    d.expect(session.eye_cal.held is not None, "the calibration waits")
_scp_state("level-back", "SCPSession.continue_calibration")(_level_held(75.0))
_scp_state("level-continue", "SCPSession.continue_calibration")(_level_held(14.0, mid=True))


def _level_screen(channels: int):
    """The screen with the level plot, the last channel past the limit for its first 50 s, the frame at 45 s."""
    def frame(d) -> None:
        d.shapes[channels - 1] = _settling(50.0)
        _no_transfer(d)
        d.toward(8.0)
        d.feed(lambda d_: d_.at >= 45.0 * d_.session.rate)
        d.expect(d.session.at_input_limit(channels - 1), "the last channel is at the input limit")
    return frame


for _n in (1, 2, 4):
    _scp_state(f"level-{_n}ch", "SCPSession.continue_calibration", channels=_n)(_level_screen(_n))


_scp_state("eyes-rotations", "SCPSession.start_calibration")(lambda d: _eyes(d, 15.0))
_scp_state("eyes-blinks", "SCPSession.start_calibration")(lambda d: _eyes(d, 36.0))


@_scp_state("eyes-long", "SCPSession.set_eye_length")
def _eyes_long(d) -> None:
    """The card with `long` chosen for the next calibration, and none yet."""
    d.session.set_extra(1, True)
    d.session.set_eye_length("long")
    _no_transfer(d)
    d.toward(8.0)
    d.trial(2, ACTIVE, 3.0)
_scp_state("eyes-result", "SCPSession.start_calibration")(lambda d: _eyes(d, None))


@_scp_state("eyes-card", "SCPSession.start_calibration")
def _eyes_card(d) -> None:
    """After the result is closed: the card with the button and the lit pill, two trials into the new run."""
    _eyes(d, None)
    d.session.close_calibration()
    d.shapes = {}
    _no_transfer(d)
    d.toward(8.0)
    d.trial(2, ACTIVE, 3.0)
_extras("extra-dead", 2, (1,), _flat)
_extras("training-moved-to-lit-chip", 4, (1, 2), None, 1)

SHOTS += [("SCP", "fed", 4)] + [("SCP", name, channels) for name, (_r, _f, channels) in _SCP.items()
                                if name != "fed"]


def feed_pupil(session, seconds: float = 75.0, until=None) -> None:
    """A simulated eye run to the middle of a trial's effort window, at once, or until `until(session)`.
    The session observes every sixth reading, its own 10 Hz, as it does on screen."""
    from nfb.pupil import SimulatedPupil

    if session.heart is not None:
        _warm_heart(session)
    eye = SimulatedPupil(seed=4, realtime=False, response_mm=0.35)
    session.eye.source = eye
    session.paused = False
    start = None
    for i, sample in enumerate(eye.samples()):
        start = sample.t if start is None else start
        if (until is None and sample.t - start > seconds) or (until is not None and until(session)):
            break
        session.eye.push(sample)
        if i % 6 == 0:
            session.observe()
    session.paused = True       # the shot is a still: nothing may move the frame after it is drawn
    session.eye.source = None


def _warm_heart(session, beats: int = 120) -> None:
    """Fill the borrowed heart runner's state with simulated beats, so the heart panel has a spectrum and a
    rate to draw. No thread and no clock: the beats are pushed straight through the processor."""
    import time

    from nfb.hrv import SimulatedBeats

    source = SimulatedBeats(seed=4, realtime=False)
    for i, beat in zip(range(beats), source.beats()):
        session.heart.state = session.heart.processor.push(beat)
    session.heart.seen_at = time.monotonic()


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


def build_settings(page: ft.Page, state: str):
    """The launcher with its toolbar, and the settings menu open over it."""
    from nfb_console import developer, settings_dialog

    bar = chrome.Toolbar(_Device(), lambda: None, lambda: None, lambda _n: None)
    bar.bind_settings(lambda: None)
    bar.set_crumb(None)
    bar.set_transport_visible(False)      # as `Console.show_launcher` leaves it
    developer.settings.on = state != "closed"
    control = ft.Column([bar.control, splash.view(page, _Device(), lambda _p: None, lambda: None)],
                        spacing=0, expand=True)
    # **Without the variable only while the dialog is built**, which reads it once and draws static pills.
    # Nothing else runs on this server in that moment, and it is back before anything can.
    held = os.environ.pop("NFB_NO_OUTPUTS", None) if state == "free" else None
    try:
        settings_dialog.show(page, lambda: None)
    finally:
        if held is not None:
            os.environ["NFB_NO_OUTPUTS"] = held
    return control


def open_help(control, title: str) -> None:
    """Press the `?` whose dialog is called `title`, or the gear for `gear:title`. Both carry their title
    as their tooltip."""
    glyph = "?"
    if title.startswith("gear:"):
        glyph, title = "⚙", title[5:]
    stack = [control]
    while stack:
        ctrl = stack.pop()
        if (getattr(ctrl, "tooltip", None) == title and getattr(ctrl, "on_click", None)
                and getattr(getattr(ctrl, "content", None), "value", None) == glyph):
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
            while page.pop_dialog() is not None:    # one route's dialog must not sit over the next shot
                pass
            parts = [p for p in route.split("/") if p]
            page.controls.clear()
            if parts[:1] == ["launcher"]:
                page.controls.append(splash.view(page, _Device(), lambda _p: None, lambda: None))
            elif parts[:1] == ["settings"]:
                page.controls.append(build_settings(page, parts[1]))
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
                    screen = control.controls[0] if isinstance(control, ft.Stack) else control  # an overlay's
                    panes = screen.controls[2].controls
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
