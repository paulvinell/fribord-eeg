"""Where every program's continuous outputs are applied: the screen's brightness, the system volume and the
music, on a thread of their own.

These calls go into CoreAudio and the window server. They are the slowest things in the process (a volume
write at 1.1 ms p99, a display lookup at 13 ms worst, a music player that may spawn a process), and the
audio system can block in them: closing a stream on the screen's thread once hung the whole console in a
deadlock inside CoreAudio. So no program applies them where it computes the level. It publishes what the
outputs should follow, and the pump applies the freshest snapshot and drops any it missed, which is right
for outputs whose smoothing runs in wall-clock seconds.

Release goes through the same thread, so a release asked for mid-apply lands after that apply instead of
interleaving two threads' calls into CoreAudio. Without a running pump (a session never started, which is
what a test that drives a session directly gets) both run inline on the caller.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from . import music, shadow, sysvolume
from .feedback import CONDITION


class OutputPump:
    """One thread that applies a session's output snapshots, one at a time, freshest wins."""

    def __init__(self, apply: Callable[[object], None], release: Callable[[], None],
                 name: str = "outputs"):
        self._apply = apply
        self._release = release
        self._name = name
        self._state: object | None = None
        self._wake = threading.Event()
        self._closing = False
        self._stopped = False
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._closing

    def start(self) -> None:
        """Start the thread, if it is not already running. A session calls this when it starts to run."""
        if self.running:
            return
        self._closing = self._stopped = False
        self._state = None
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._wake,), daemon=True, name=self._name)
        self._thread.start()

    def publish(self, state: object) -> None:
        """What the outputs should follow now. A later snapshot replaces one the pump has not reached."""
        if self.running:
            self._state = state
            self._wake.set()
        elif not self._stopped:
            self._apply(state)          # never started: a test driving the session directly

    def release(self) -> None:
        """Hand the outputs back: the screen, the volume and the music."""
        if self.running:
            self._state = None
            self._wake.set()
        elif not self._closing:
            self._release()

    def stop(self) -> None:
        """Release the outputs and end the thread, without waiting for it. A thread stuck in an output call
        costs only itself; the caller, usually the screen's thread, goes on."""
        self._stopped = True
        if not self.running:
            if not self._closing:
                self._release()
            return
        self._closing = True
        self._wake.set()

    def _run(self, wake: threading.Event) -> None:
        while True:
            wake.wait()
            wake.clear()
            if self._closing or self._thread is not threading.current_thread():
                self._call(self._release)
                return
            state = self._state
            self._call(self._release if state is None else (lambda: self._apply(state)))

    def _call(self, step: Callable[[], None]) -> None:
        try:
            step()
        except Exception as exc:
            from .training import slog          # here, not at the top: `training` imports this module

            slog(f"{self._name}: {type(exc).__name__}: {exc}")


class LevelOutputs:
    """The outputs of a program that computes one level and is not a `TrainingSession`: HEG, HRV, Pupil and
    Pulse pair. The screen, the volume and the music go through an `OutputPump`; the click is a one-shot
    sound and plays where the level is computed.

    The session supplies `feedback`, `_level`, `state_balance`, `channel_targets` and `cue`, calls
    `_init_outputs` once, `_drive_outputs` on every new level, `release_outputs` when the level stops,
    `start_outputs` when it starts to run and `stop_outputs` when it ends.
    """

    def _init_outputs(self) -> None:
        self.display = shadow.DisplayShadow()
        self.output_volume = sysvolume.SystemVolume()
        self.music = music.MusicOutput()
        self._last_click: float | None = None
        self._outputs = OutputPump(self._apply_outputs, self._release_now)

    def start_outputs(self) -> None:
        self._outputs.start()

    def stop_outputs(self) -> None:
        self._outputs.stop()

    def _met(self) -> bool:
        """Whether the condition holds, for the click and the music's hold."""
        return self.feedback.on_target(self.channel_targets(CONDITION))

    def _drive_outputs(self) -> None:
        met = self._met()
        self._outputs.publish((self._level, met, self.state_balance))
        self._maybe_click(met)

    def _apply_outputs(self, state: tuple) -> None:
        level, met, balance = state
        sh = self.feedback.shadow
        follows = balance if sh.follows == "state" and balance is not None else level
        self.display.apply(sh.enabled and sh.supported, sh.screen, *sh.ends(), sh.map(follows))
        snd = self.feedback.sound
        self.output_volume.apply(snd.enabled, *snd.ends(), snd.map(level))
        self.music.apply(self.feedback.music, hold=self.feedback.music.holds(met), balance=balance,
                         level=level)

    def _maybe_click(self, met: bool) -> None:
        """The discrete mark, on the same interval rule as everywhere else."""
        click = self.feedback.click
        if not click.enabled or not met:
            if not click.enabled:
                self._last_click = None
            return
        now = time.monotonic()
        if self._last_click is None or now - self._last_click >= click.interval_seconds:
            self._last_click = now
            self.cue(click.sound, click.volume)

    def release_outputs(self) -> None:
        """Hand back the screen, the volume and the music, through the pump when it runs."""
        self._outputs.release()

    def _release_now(self) -> None:
        self.display.release()
        self.output_volume.release()
        self.music.release()
