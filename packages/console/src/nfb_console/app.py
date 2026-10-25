"""Operator console: pick a protocol, then run it.

Two screens. The launcher lists protocols and the device's state; the session screen draws traces and
the protocol's metric. Runs against a recording by default so it works with no hardware attached —
the same property that makes the DSP testable — and `--live` uses the device.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import flet as ft

import optima

from nfb import Device, Recorder, Settings

from . import (amplitude_view, at_view, brand, chrome, device_panel, heart, heg_view, hrv_view, cfc_view,
               ilf_view, mendi, pac_view, prefs, pupil, pupil_calibrate, pupil_camera, pupil_view, record_dialog, scp_view, session_view, sleep_view,
               splash, wave_view)
from . import theme as t
from .inputs import HeartChoice, InputChoice, MendiChoice, PupilChoice
from .protocols import BY_CODE, Protocol


class Console:
    """Owns the window and swaps between the two screens."""

    def __init__(self, page: ft.Page, args: argparse.Namespace):
        self.page = page
        self.args = args
        self.session: session_view.Session | None = None
        # Flet dispatches `on_click` on a thread pool, so two clicks on a slow button really do overlap.
        # See `pick`, where the overlap cost an afternoon's configuration.
        self._picking = threading.Lock()
        # one owner of the port, created once and kept for the life of the app, so the connection
        # does not belong to whichever protocol happens to be running
        self.device = Device(Settings(hz=args.hz, reference=args.reference,
                                      mains_hz=args.mains,
                                      lower_limit_hz=args.high_pass))
        # Where samples come from, chosen on the device screen and outliving any one session.
        # Without --live the recording *is* the input, so it appears there rather than being an
        # invisible fallback inside each protocol screen.
        self.inputs = InputChoice(
            kind="amplifier" if args.live else "file",
            path="" if args.live else (args.file or args.path), loop=True)
        if args.live and not self.device.connect():
            # On the terminal as well as on the chip. `--live` is asked for explicitly, so a connection
            # that failed is news, and the person who typed it is looking at a terminal.
            print(f"nfb: {self.device.status.failure} — {self.device.status.remedy}", file=sys.stderr)
        # One heart source for the app, like the device: the HRV protocol trains on it and the CFC screen
        # shows the same instruments beside the EEG. Two readers would fail — a strap notifies one
        # subscriber — so this is held here and borrowed.
        self.heart_choice = HeartChoice(kind=args.heart)
        self.heart = heart.HeartRunner(self.heart_choice)
        if self.heart_choice.on:
            self.heart.attach()
        # And one headband, for the same reason: BLE is exclusive, so two readers of one Mendi is the
        # same failure as two openers of the serial port. Off until the HEG screen or the device screen
        # asks for it — inventing frames nobody requested is what `HEGSession._invented` guards against.
        self.mendi_choice = MendiChoice(kind="off")
        self.mendi = mendi.MendiRunner(self.mendi_choice)
        # And one eye. Off until the pupil screen or the device screen asks for it, as the headband is.
        self.pupil_choice = PupilChoice(kind="off")
        self.pupil = pupil.PupilRunner(self.pupil_choice)
        # chrome that does not move between screens: device, channels, transport, clock
        self.channels = max(1, min(4, args.channels))
        self.protocol: Protocol | None = None
        # `show_device` with no argument would go back to the launcher, which is not where you came from.
        # The chip is on every screen, so "back" has to mean the screen it was pressed on.
        self.toolbar = chrome.Toolbar(self.device, lambda: self.show_device(self.protocol),
                                      self.show_launcher, self.set_channels)
        self.toolbar.set_channels(self.channels)
        # One recorder at a time, owned here like the device: recording is acquisition, not a property
        # of whichever protocol screen is showing. See `_maybe_record` for when one is armed.
        self.recorder: Recorder | None = None
        self.record_settings = record_dialog.RecordSettings(**prefs.record_where())
        self._record_picker = ft.FilePicker()
        page.services.append(self._record_picker)
        self.toolbar.bind_record(lambda _e: self.toggle_record(),
                                 lambda _e: self.show_record_options())

        page.title = brand.NAME
        page.fonts = brand.FONTS
        page.bgcolor = t.BG
        page.padding = 0
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window.width = 1020
        page.window.height = 660
        page.window.min_width = 880
        page.window.min_height = 560

    def show(self, control: ft.Control) -> None:
        self.page.controls.clear()
        self.page.add(control)
        self.page.update()

    def set_channels(self, count: int) -> None:
        """Change how many channels are in use.

        Reconfigures the device when it is connected and idle, so it stops streaming what nobody is
        looking at, and re-enters the current protocol with a fresh session — the channel set is part of
        a session's shape, so changing it is a new session rather than a mutation of this one. The
        toolbar disables this while a session is running, so nothing is discarded by surprise.
        """
        self.channels = max(1, min(4, count))
        self.toolbar.set_channels(self.channels)
        # **End the session first.** It is what holds the device measuring — even while paused — so
        # reconfiguring with it alive is refused, and the refusal used to be swallowed here. The device then
        # went on streaming four channels while the screen showed one, and three enabled inputs with nothing
        # in them is exactly what makes the device's own indicator go red and every one of those channels
        # report "no signal". The session is being replaced two lines below in any case.
        # **Stopped, not cleared**, and the difference is every card on the screen. Clearing it meant the
        # `pick` below found nothing to adopt from and rebuilt the protocol from its defaults, so changing
        # the channel count silently threw away every band, threshold, filter and role — the same reset the
        # double-click race caused, arrived at deterministically. `pick` stops it again, which is harmless.
        if self.session is not None:
            self.session.stop()
        try:
            self.device.reconfigure(channels=self.channels)
        except (RuntimeError, ValueError) as exc:
            # Not connected is ordinary and needs no fuss; anything else is worth a line, because a
            # silently-ignored channel change is what this comment is about.
            if self.device.status.connected:
                device_panel.log(f"set_channels({self.channels}) refused: {exc}")
        if self.protocol is not None:
            self.pick(self.protocol)
        else:
            self.show_launcher()

    def show_launcher(self) -> None:
        self.toolbar.set_source(self.source_label())
        self.toolbar.set_channels_visible(True)
        self.protocol = None
        # the session detaches, but the device stays connected
        self.toolbar.set_crumb(None)
        if self.session is not None:
            self.session.stop()
            self.session = None
        self._end_recording()
        self.toolbar.set_transport_visible(False)
        self.toolbar.refresh()
        self.show(ft.Column([self.toolbar.control,
                             splash.view(self.page, self.device, self.pick, self.show_device)],
                            spacing=0, expand=True))

    def restart(self) -> None:
        """End the session and stay where you are, on a fresh one.

        What the stop button does. Stopping is something you do to the session rather than a way of leaving
        the screen — the logo is the way out, from here as from everywhere. Re-entering rather than leaving
        a dead session on screen is because the source is closed by then: the play button would have nothing
        to start, which is a worse state than a paused fresh session.
        """
        if self.protocol is not None:
            self.pick(self.protocol)
        else:
            self.show_launcher()

    # --- recording ---------------------------------------------------------------------------

    def _end_recording(self) -> None:
        """Join the current recording, if any. The epoch ending is what actually finalises the file —
        stopping a session stops the device — so this is bookkeeping. The companion file is written by
        the recorder itself, on its own thread and repeatedly; see `nfb.notes`."""
        rec, self.recorder = self.recorder, None
        # Deliberate teardown, so the sentry must stop watching it before it closes — a recorder
        # ending by the operator's hand is not a recording failure. Not every screen has a sentry:
        # the HEG and HRV sessions are their own devices' and are not `TrainingSession` at all.
        watcher = getattr(self.session, "sentry", None)
        if watcher is not None:
            watcher.recorder = None
        if rec is None:
            return
        rec.stop()

    def _maybe_record(self, session) -> None:
        """Arm a recorder for a fresh session where one is wanted and possible.

        Possible means a live amplifier is the source: a replay is already a recording, and a screen
        whose instrument is not the amplifier has no frames here to record. Wanted is the operator's
        stored per-protocol choice, falling back to the session's own default — which is how the sleep
        screen records unasked.
        """
        self._end_recording()
        self.toolbar.recording = None
        supported = (getattr(session, "device", None) is self.device
                     and self.device.status.connected and not self.inputs.from_file)
        self.toolbar.set_record_supported(supported)
        if supported and self.protocol is not None and prefs.record_default(
                self.protocol.code, getattr(session, "records_by_default", False)):
            self._start_recording()
            # The sentry watches the recording it is defending: a disk that fills at 3 a.m. is an
            # alarm, not a toolbar footnote.
            watcher = getattr(session, "sentry", None)
            if watcher is not None:
                watcher.recorder = self.recorder

    def show_record_options(self) -> None:
        """The caret half of the record button: folder, name and format, with the resolved path."""
        def remember() -> None:
            prefs.set_record_where(vars(self.record_settings))
        record_dialog.show(self.page, self.record_settings, remember, self._record_picker)

    def _start_recording(self) -> None:
        # The companion is read from whatever session is current at the time of writing, not captured
        # here: a recorder outlives none of them, but the snapshot must be the live one.
        rec = Recorder(self.device, self.record_settings.resolve(),
                       note=self.protocol.code if self.protocol else "",
                       fmt=self.record_settings.fmt,
                       companion=lambda: (self.session.companion()
                                          if hasattr(self.session, "companion") else {}))
        try:
            rec.start()
        except (RuntimeError, OSError) as exc:
            self.toolbar.report(f"recording could not start: {exc}")
            return
        self.recorder = rec
        self.toolbar.recording = rec

    def toggle_record(self) -> None:
        """The toolbar's record button: start or stop, and remember the choice for this protocol.

        The guard is here and not only on the control, same rule as the channel menu: `disabled` stops
        a click in the UI, but "nothing real to record" is a rule, and it belongs where it is enforced.
        """
        if self.protocol is None or self.inputs.from_file or not self.device.status.connected:
            return
        if self.recorder is not None and self.recorder.active:
            prefs.set_record_default(self.protocol.code, False)
            self._end_recording()
            return
        prefs.set_record_default(self.protocol.code, True)
        self._start_recording()
        if self.session is not None:
            self.session.sentry.recorder = self.recorder

    def show_device(self, return_to: Protocol | None = None, device_tab: int = 0) -> None:
        # The feed has to stop — impedance testing and measuring are exclusive device modes, and a session
        # holds the device measuring even while paused. But **the session is kept alive**, because checking
        # contact mid-session is an ordinary thing to do: you pause, look at the electrodes, come back and
        # carry on. A fresh session with the same settings is a *different* session — the elapsed clock,
        # every target's trend and calibration, the artifact gate's running level and the on-target history
        # are the session, not decoration on top of it. `restart` is the path that means start again.
        if self.session is not None:
            self.session.suspend()
            if self.device.status.connected:
                self.device.stop()          # release the mode so a contact check can run
        self._end_recording()               # its epoch just ended; `resume` arms a fresh file
        # no session while the device screen is up, so no transport either
        self.toolbar.set_transport_visible(False)
        # The trail is the way back, and it names where you came from rather than saying "back" and meaning
        # the launcher. From the launcher there is no parent, because the launcher *is* the location.
        parent = ((return_to.name, lambda: self.resume(return_to))) if return_to else None
        self.toolbar.set_crumb("Device", parent)
        self.show(ft.Column([self.toolbar.control,
                             device_panel.view(self.page, self.device, self.inputs,
                                               self.heart_choice, self.heart,
                                               self.mendi_choice, self.mendi,
                                               self.pupil_choice, self.pupil,
                                               lambda: self.calibrate_pupil(return_to),
                                               device_tab,
                                               lambda: self.show_camera(return_to))],
                            spacing=0, expand=True))

    def calibrate_pupil(self, return_to: Protocol | None) -> None:
        """Hand the tracker to the full-screen follow-the-dot calibration, and take it back after.

        Calibration holds the tracker itself, and the reader cannot share it, so the reader is released
        first and started again on the way back — to the Pupil tab, where calibration was launched."""
        self.pupil.release()
        self.show(pupil_calibrate.view(
            self.page, self.pupil_choice,
            lambda: self._after_calibrate(return_to)))

    def _after_calibrate(self, return_to: Protocol | None) -> None:
        """Back to the Pupil tab, and start the shared reader again. Calibration held the tracker on its
        own connection, so the reader was released; the tracker re-enumerates when that connection closes,
        so the restart retries until the device is back."""
        self.show_device(return_to, device_tab=device_panel.PUPIL_TAB)
        if not self.pupil_choice.on:
            return

        def restart() -> None:
            for _ in range(20):
                if self.pupil.running:
                    return
                self.pupil.error = None
                self.pupil.attach(self.pupil_choice)
                if self.pupil.running:
                    return
                time.sleep(0.5)

        threading.Thread(target=restart, daemon=True, name="pupil-reattach").start()

    def show_camera(self, return_to: Protocol | None) -> None:
        """Take the tracker over for its infrared camera view, then hand it back. The camera and the gaze
        reader cannot share the device, so the reader is released while the camera is up."""
        self.pupil.release()
        self.show(pupil_camera.view(self.page, lambda: self._after_calibrate(return_to)))

    def source_label(self) -> str:
        """What the samples are coming from, when it is not the amplifier. Empty for a live session."""
        if self.args.live and not self.inputs.from_file:
            return ""
        path = (self.inputs.path if self.inputs.from_file else None) or self.args.file or self.args.path
        return f"file: {Path(path).name}" if path else "no input"

    @staticmethod
    def screen_for(protocol: Protocol):
        """Which screen runs a protocol — a lookup, so adding one is a table entry."""
        return next((s for s in (amplitude_view, at_view, cfc_view, heg_view, hrv_view, ilf_view,
                                 pac_view, pupil_view, scp_view, sleep_view, wave_view)
                    if protocol.code in s.HANDLES), None)

    def resume(self, protocol: Protocol) -> None:
        """Come back from the device screen to the session you left, exactly as you left it.

        Rebuilds the *view* against the same session object, because the controls were detached when the
        device screen replaced them — but nothing about the session is rebuilt. It stays paused, since that
        is how you left it and time spent looking at electrodes is not time on task; the play button picks
        up where it stopped.
        """
        session = self.session
        screen = self.screen_for(protocol)
        if session is None or not session.suspended or screen is None or self.protocol is not protocol:
            self.pick(protocol)             # nothing to come back to
            return
        self.protocol = protocol
        session.rearm()
        threading.Thread(target=session.run, daemon=True,
                         name=f"{protocol.code.lower()}-feed").start()
        self._maybe_record(session)
        self.show(screen.view(self.page, session, self.toolbar, self.restart))

    def pick(self, protocol: Protocol) -> None:
        # **What the operator configured survives a stop.** Stop ends the session and re-enters the
        # protocol on a fresh one, and a fresh one was built from the protocol's defaults — so training SMR
        # and pressing stop put the band back to 8-12 Hz, along with every volume, the arrangement, the
        # session length and the artifact policy. The session is the history; the setting up is not, and
        # `TrainingSession.adopt` is where the line is drawn. Only from the same protocol.
        # **Two clicks on Stop reset the afternoon's setting up, and this is why.** Building a screen is
        # slow — it opens the device, constructs the session and lays out every control — and `on_click`
        # handlers are dispatched by Flet on a thread pool, so a second click lands *inside* the first
        # call. `self.session` was set to None for the duration of the build, so the second call found
        # nothing to adopt from, took the `prefs.restore` branch, and came up on the protocol's defaults:
        # bands back to 8-12 Hz, every card's threshold, filter and role gone. Intermittent by
        # construction, since it needs the second click to land in that window.
        #
        # Two changes, and both are needed. `self.session` is never cleared — the new one replaces it in a
        # single assignment, so there is no instant at which the app has a protocol on screen and no
        # session to carry across. And a pick already in flight makes this one a no-op.
        #
        # **Dropped rather than queued, which is the difference between a guard and a deadlock.** A second
        # click is not a second restart; it is impatience with a button that is already doing the thing,
        # and running it afterwards would stop the session that the first click just started. A blocking
        # lock would also hang outright the moment a pick is re-entered on its own thread, which is what a
        # control built during the build doing anything at all would cause.
        if not self._picking.acquire(blocking=False):
            return
        try:
            self._pick(protocol)
        finally:
            self._picking.release()

    def _pick(self, protocol: Protocol) -> None:
        previous = self.session
        if previous is not None:
            previous.stop()
        self.protocol = protocol
        source = self.inputs.open(self.device) if self.inputs.from_file else None
        # Say so in the toolbar for as long as it lasts. A replayed capture drew a screen identical to a
        # live one, and if the capture was made with an empty amplifier then every channel in it truthfully
        # reports no signal — so the screen blamed the electrodes, correctly, about a file from days ago.
        self.toolbar.set_source(self.source_label())
        # Back on by default; a screen with no amplifier channels hides it in its own `view`.
        self.toolbar.set_channels_visible(True)
        screen = self.screen_for(protocol)
        if screen is not None:
            self.session = screen.session_for(
                protocol,
                None if self.args.live else self.args.path,
                device=self.device if self.args.live else None,
                source=source,
                channels=self.channels,
                heart_choice=self.heart_choice,
                heart_runner=self.heart,
                mendi_choice=self.mendi_choice,
                mendi_runner=self.mendi,
                pupil_choice=self.pupil_choice,
                pupil_runner=self.pupil,
            )
            if previous is not None:
                self.session.adopt(previous)
            else:
                # Only with nothing to adopt: a live session's settings are more recent than the file, and
                # `adopt` carries them wholesale, so restoring first would be overwritten and restoring after
                # would undo a stop that was meant to keep them.
                prefs.restore(protocol.code, self.session)
            threading.Thread(target=self.session.run, daemon=True,
                             name=f"{protocol.code.lower()}-feed").start()
            self._maybe_record(self.session)
            self.show(screen.view(self.page, self.session, self.toolbar, self.restart))
            return
        self.session = session_view.Session(
            protocol,
            None if self.args.live else self.args.path,
            device=self.device if self.args.live else None,
            source=source,
            channels=self.channels,
        )
        self.session.start()
        self.show(session_view.view(self.page, self.session, self.show_launcher,
                                    lambda: self.show_device(protocol),
                                    self.args.benchmark))

    def start(self) -> None:
        if self.args.protocol:
            self.pick(BY_CODE[self.args.protocol])
        else:
            self.show_launcher()


def build_parser() -> argparse.ArgumentParser:
    """Separated from `main` so a test can check the command line the Makefile actually passes. It could
    not, and `make live` was broken by a flag the parser did not accept."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="sessions/latest.jsonl.gz",
                    help="recorded session to replay when not --live")
    ap.add_argument("--live", action="store_true", help="use the device instead of a recording")
    ap.add_argument("--file", help="play this file on loop instead of the amplifier: a recorded "
                                   ".jsonl.gz session, or a text export of one sample per row")
    ap.add_argument("--hz", type=float, default=1000.0)
    ap.add_argument("--channels", type=int, default=4, choices=(1, 2, 3, 4),
                    help="how many channels to use; also settable from the toolbar")
    ap.add_argument("--reference", default="linked", choices=sorted(optima.REFERENCE_MODES))
    # The Makefile has passed this since `make live` existed and the parser never accepted it, so the
    # documented command line was simply broken. Not cosmetic either: the notch runs in the device's
    # hardware, so getting it wrong means the hum is not filtered at all.
    ap.add_argument("--mains", type=int, default=50, choices=(50, 60),
                    help="mains frequency the device notches out, in its own hardware")
    ap.add_argument("--high-pass", dest="high_pass", type=float, default=0.5, choices=(0.5, 0.01),
                    help="the device's high-pass corner; 0.01 for ultra-low-frequency work")
    ap.add_argument("--heart", default="off", choices=("off", "strap", "usb", "simulator"),
                    help="heart-rate source: a Bluetooth strap, the HeartMath on its USB cable, or "
                         "synthetic beats; also settable on the device screen")
    ap.add_argument("--protocol", choices=sorted(BY_CODE),
                    help="skip the launcher and open this protocol directly")
    ap.add_argument("--benchmark", type=float, default=0.0,
                    help="with --protocol: run N seconds, print the frame rate, exit")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    console: Console | None = None

    def build(page: ft.Page) -> None:
        nonlocal console
        console = Console(page, args)
        console.start()

    try:
        brand.window_app()
        ft.run(build, assets_dir=brand.ASSETS)
    finally:
        if console is not None:
            if console.session is not None:
                console.session.stop()
            console._end_recording()
            console.heart.release()
            console.mendi.release()
            console.pupil.release()
            console.device.close()


if __name__ == "__main__":
    main()
