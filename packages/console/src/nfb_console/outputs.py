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
from typing import Callable


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
