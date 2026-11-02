"""A device reader released from the screen's thread: the release waits for the reader to end, so the screen
hands it to a thread of its own, and a start asked for after it waits its turn."""

from __future__ import annotations

import threading
import time

from nfb_console.inputs import PupilChoice
from nfb_console.pupil import PupilRunner

from conftest import waitfor


class _Source:
    """A source whose close blocks until told, the way a device handshake can."""

    error = None

    def __init__(self, gate: threading.Event):
        self.gate = gate
        self.ended = threading.Event()

    @property
    def closed(self) -> bool:
        return self.ended.is_set()

    def samples(self):
        self.ended.wait()                   # no readings, until closed
        return
        yield

    def close(self):
        self.gate.wait(5.0)
        self.ended.set()

    def describe(self):
        return "test source"


def _running(gate: threading.Event) -> PupilRunner:
    runner = PupilRunner(PupilChoice(kind="simulator"))
    runner.choice.open = lambda: _Source(gate)
    runner.choice.problem = lambda: None
    assert runner.attach()
    return runner


def test_release_aside_returns_at_once_and_then_runs_after_the_release():
    gate = threading.Event()
    runner = _running(gate)
    done: list[bool] = []
    began = time.monotonic()
    runner.release_aside(lambda: done.append(runner.source is None))
    assert time.monotonic() - began < 0.05, "the caller does not wait for a close that blocks"
    gate.set()
    waitfor(lambda: done, what="the release to finish")
    assert done == [True], "`then` runs after the release"
    assert not runner.running


def test_a_start_after_a_release_aside_waits_its_turn():
    """Started at once, the new reader would be ended by the release still in flight."""
    gate = threading.Event()
    runner = _running(gate)
    old = runner.source
    runner.release_aside()
    began = time.monotonic()
    assert runner.attach() is True
    assert time.monotonic() - began < 0.05, "the start does not wait on the caller either"
    gate.set()
    waitfor(lambda: runner.source is not None and runner.source is not old and runner.running,
            what="the new reader")
    assert old.closed
    runner.release()
