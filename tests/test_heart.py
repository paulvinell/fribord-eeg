"""The heart runner: it opens the chosen source, reports what went wrong and tries again."""

from __future__ import annotations

import time

from nfb.hrv.beats import Beat

from conftest import waitfor
from harness import ListedPort

def test_the_missing_cable_is_reported_by_whoever_tried_to_open_it(monkeypatch):
    """The other half of not saying it twice: with the choice staying quiet, the fault still has to
    arrive — from the runner, on its own thread, and be the sentence the status line shows."""
    from nfb.hrv import sources
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    monkeypatch.setattr(sources, "usb_candidates", lambda: [])
    runner = HeartRunner(HeartChoice(kind="usb"))
    assert runner.attach() is True, "nothing is known to be wrong until it is tried"
    waitfor(lambda: runner.error is not None, what="the runner to report the missing cable")
    runner.release()
    assert runner.error and "HeartMath cable" in runner.error
    assert runner.describe() == runner.error, "and it is what the screen shows"


def test_a_chosen_source_keeps_trying_until_the_cable_is_plugged_in(monkeypatch):
    """Choosing a source is the operator saying what is on their body, not what is plugged in this
    second. A cable connected a minute later, or pulled out and put back, must not need a trip to the
    device screen to click the same button again."""
    from nfb.hrv import sources
    from nfb_console.heart import HeartRunner
    from nfb_console.inputs import HeartChoice

    plugged: list = []
    opened: list = []
    monkeypatch.setattr(sources, "usb_candidates", lambda: list(plugged))
    monkeypatch.setattr(HeartRunner, "RETRY_SECONDS", 0.1)

    class Nothing:
        """A port that is not there: the reader records why and ends, as the real one does."""

        def __init__(self, port=None):
            self.error = None
            self.port = port

        def beats(self):
            port, trouble = sources.find_usb_port()
            if port is None:
                self.error = trouble
                return iter(())
            opened.append(port)
            return iter((Beat(t=time.time(), hr_bpm=60.0, rr_intervals=[1.0]),))

        def close(self):
            pass

        def describe(self):
            return self.error or "reading"

    monkeypatch.setattr(sources, "UsbPpgBeats", Nothing)
    monkeypatch.setattr("nfb.hrv.beat_source",
                        lambda kind, **kw: Nothing(**kw) if kind == "usb" else None)

    runner = HeartRunner(HeartChoice(kind="usb"))
    runner.attach()
    waitfor(lambda: runner.error is not None, what="the runner to report the missing cable")
    assert "trying again" in (runner.error or ""), "it says it has not given up"

    plugged.append(ListedPort("/dev/cu.usbserial-HM01", 0x0403, "FT230X Basic UART"))
    waitfor(lambda: opened, what="the runner to open the cable once it is there")
    runner.release()
    assert opened == ["/dev/cu.usbserial-HM01"], "it connected itself once the cable was there"
