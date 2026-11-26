"""The device screen: its source rows, what they offer and how they follow a connection."""

from __future__ import annotations

def test_switching_source_does_not_wait_for_the_old_one_on_the_event_loop(monkeypatch):
    """Releasing a source waits up to two seconds for its reader thread, and a reader halfway through a
    twenty-second Bluetooth scan takes all of it. Doing that in the click handler froze every control on
    the page, which is the Flet rule in reverse: the event loop must not be blocked, not merely written
    to from elsewhere."""
    import threading

    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import HeartChoice

    class SlowRunner:
        """Releases only when the test says so — which is the point: the click must return while this is
        still blocked. A `time.sleep` here would have tested the same thing at the cost of half a second,
        and would have been a race on a loaded machine rather than a fact."""

        error = None

        def __init__(self):
            self.entered = threading.Event()
            self.may_finish = threading.Event()
            self.released = threading.Event()

        def release(self):
            self.entered.set()
            self.may_finish.wait(5.0)
            self.released.set()

        def attach(self, _choice=None):
            return True

        def describe(self):
            return "…"

    class Page:
        def __init__(self):
            self.services, self.tasks = [], []

        def update(self, *a):
            pass

        def run_task(self, fn, *a):
            self.tasks.append(fn)

    device = Device(Settings())
    runner = SlowRunner()
    choice = HeartChoice(kind="off")
    control = device_panel.view(Page(), device, heart_choice=choice, heart_runner=runner)

    _press(control, "usb")
    assert runner.entered.wait(2.0), "the swap started"
    assert choice.kind == "usb", "and the click had already returned, with the release still blocked"
    runner.may_finish.set()
    assert runner.released.wait(2.0), "the swap completed, just not on the event loop"
    device.close()


def _press(control, label: str) -> None:
    """Click the source button carrying `label`, the way the operator does."""
    import flet as ft

    def walk(ctrl):
        if (isinstance(ctrl, ft.Container) and isinstance(ctrl.content, ft.Text)
                and ctrl.content.value == label and ctrl.on_click is not None):
            return ctrl
        for child in list(getattr(ctrl, "controls", None) or []) + [getattr(ctrl, "content", None)]:
            if child is not None and (hit := walk(child)) is not None:
                return hit
        return None

    button = walk(control)
    assert button is not None, f"no button reading {label!r}"
    button.on_click(None)


def test_the_source_row_keeps_up_with_a_connection_it_did_not_click(monkeypatch):
    """The row is repainted when something is clicked, and connecting happens on another thread — so
    after the click that *starts* a connection it read "not started", the state of a millisecond
    earlier, and stayed that way because nothing on the device screen refreshes on a timer unless the
    contact check is running."""
    import asyncio

    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import HeartChoice

    class LateRunner:
        error = None

        def __init__(self):
            self.said = "not started"

        def release(self):
            pass

        def attach(self, _choice=None):
            return True

        def describe(self):
            return self.said

    class Page:
        def __init__(self):
            self.services, self.tasks, self.updates = [], [], 0

        def update(self, *a):
            self.updates += 1

        def run_task(self, fn, *a):
            self.tasks.append((fn, a))

    page = Page()
    device = Device(Settings())
    runner = LateRunner()
    control = device_panel.view(page, device, heart_choice=HeartChoice(kind="usb", port="/dev/x"),
                                heart_runner=runner)
    assert any("not started" in x for x in _texts(control))

    # The tick interval is the loop's, not the test's: with it at zero the loop runs as fast as the event
    # loop will schedule it, and a hundredth of a second is dozens of iterations. Waiting a real second
    # for a real 0.25 s tick was measuring the sleep.
    monkeypatch.setattr(device_panel, "POLL_SECONDS", 0)

    async def briefly():
        poll = next(fn for fn, _ in page.tasks)
        runner.said = "reading the pulse waveform on /dev/x"
        try:
            await asyncio.wait_for(poll(), 0.01)
        except (asyncio.TimeoutError, TimeoutError):
            pass

    asyncio.run(briefly())
    assert any("reading the pulse waveform" in x for x in _texts(control)), \
        "the row followed the connection with nothing clicked"
    device.close()


def _texts(ctrl) -> list[str]:
    import flet as ft

    if isinstance(ctrl, ft.Text):
        return [ctrl.value or ""]
    out: list[str] = []
    for child in getattr(ctrl, "controls", None) or []:
        out += _texts(child)
    content = getattr(ctrl, "content", None)
    if content is not None:
        out += _texts(content)
    return out


def test_the_device_screen_offers_the_cable_beside_the_strap():
    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import HeartChoice

    class Page:
        def __init__(self):
            self.services, self.tasks = [], []

        def update(self, *a):
            pass

        def run_task(self, fn, *a):
            self.tasks.append(fn)

    device = Device(Settings())
    # A named port, because with none named the note under the source row is replaced by the reason
    # nothing was found — which is the right thing for it to say, and not what this is checking.
    control = device_panel.view(Page(), device,
                                heart_choice=HeartChoice(kind="usb", port="/dev/cu.usbserial-HM01"))

    def texts(ctrl):
        import flet as ft
        if isinstance(ctrl, ft.Text):
            return [ctrl.value or ""]
        out = []
        for child in getattr(ctrl, "controls", None) or []:
            out += texts(child)
        content = getattr(ctrl, "content", None)
        if content is not None:
            out += texts(content)
        return out

    shown = texts(control)
    assert "usb" in shown, "the source row names it"
    assert any("USB-C cable" in x for x in shown), "and says what it is"
    device.close()


def test_the_headband_is_chosen_on_the_device_screen_like_every_other_source():
    """The HEG tab of the device screen was a paragraph explaining that the headband was not wired up
    yet, long after it was: the driver, the runner and the screen all existed, and the screen's own
    status line said "see the device screen" — where there was nothing to see. So the whole protocol was
    unreachable unless it invented a simulator for itself."""
    import sys

    sys.path.insert(0, "tests")
    from conftest import waitfor
    from harness import StubPage, _texts

    from nfb import Device, Settings
    from nfb_console import device_panel
    from nfb_console.inputs import MendiChoice
    from nfb_console.mendi import MendiRunner

    device = Device(Settings())
    choice = MendiChoice(kind="off")
    runner = MendiRunner(choice)
    try:
        control = device_panel.view(StubPage(), device, mendi_choice=choice, mendi_runner=runner)
        shown = " ".join(x.value for x in _texts(control) if x.value)
        assert "blood flow over the forehead" in shown, "the tab says what it is for"

        _press(control, "simulator")         # the HEG tab's, which comes before the heart's
        assert choice.kind == "simulator"
        # The swap runs off the event loop, so the frames arrive on their own thread.
        waitfor(lambda: runner.ready, what="the chosen source started delivering frames")
    finally:
        runner.release()
        device.close()
