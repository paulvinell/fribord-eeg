"""The output pump: every program's screen, volume and music calls run on one thread of their own, so the
screen's thread never waits on CoreAudio or the window server."""

from __future__ import annotations

import threading
import time

from nfb_console.outputs import OutputPump

from conftest import waitfor


def _recorder():
    log: list[tuple[str, object, str]] = []
    apply = lambda state: log.append(("apply", state, threading.current_thread().name))      # noqa: E731
    release = lambda: log.append(("release", None, threading.current_thread().name))       # noqa: E731
    return log, apply, release


def test_never_started_it_applies_and_releases_on_the_caller():
    log, apply, release = _recorder()
    pump = OutputPump(apply, release)
    pump.publish(0.4)
    pump.release()
    assert [(kind, state) for kind, state, _ in log] == [("apply", 0.4), ("release", None)]


def test_started_every_call_runs_on_its_own_thread():
    log, apply, release = _recorder()
    pump = OutputPump(apply, release, name="outputs")
    pump.start()
    try:
        pump.publish(0.7)
        waitfor(lambda: log, what="the snapshot applied")
        pump.release()
        waitfor(lambda: log[-1][0] == "release", what="the release")
    finally:
        pump.stop()
    assert {thread for _, _, thread in log} == {"outputs"}


def test_a_wedged_output_call_never_holds_up_the_caller():
    """Closing an audio stream on the screen's thread once deadlocked inside CoreAudio and hung the
    console. An output call that never returns must cost the pump's thread and nothing else."""
    wedged = threading.Event()
    pump = OutputPump(lambda state: wedged.wait(5.0), lambda: None)
    pump.start()
    try:
        pump.publish(0.5)
        began = time.monotonic()
        pump.publish(0.6)
        pump.release()
        pump.stop()
        assert time.monotonic() - began < 0.05
    finally:
        wedged.set()


def test_stop_releases_and_drops_what_arrives_after_it():
    log, apply, release = _recorder()
    pump = OutputPump(apply, release)
    pump.start()
    pump.stop()
    waitfor(lambda: log and log[-1][0] == "release", what="the release on the way out")
    pump.publish(0.9)                       # a straggler after the session stopped
    time.sleep(0.02)
    assert ("apply", 0.9) not in [(kind, state) for kind, state, _ in log]
