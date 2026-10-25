"""The device screen: choose an input, connect, configure, check contact.

All of it belongs to the device rather than to a protocol — you check electrode contact *before*
deciding what to train, and the input, sample rate and reference wiring outlive any one session. So
this is reachable from the status chip on every screen, including from inside a session.

Tabbed by modality (EEG / HEG / HRV) because they are different instruments with nothing in common
below the signal: they speak BLE rather than the Optima's serial protocol, so supporting one means a new
source rather than a new setting on this one. **HRV is wired up** — a strap advertising the standard Heart
Rate Service, or a simulator — and the choice made here is used by the HRV protocol *and* shown beside the
EEG on the CFC screen, which is why it belongs to the device screen rather than to either of them. The
Mendi headband is here for the same reason and works the same way: which headband is on your forehead
outlives any one session, and BLE is exclusive, so the app owns one reader and this is where it is chosen.

**The body scrolls.** It did not, and the electrode-contact panel — the one thing here you actually need
before deciding what to train — was off the bottom of the window at the default size.

Every control is built once and then mutated. Rebuilding a container's `controls` list from a
background thread does not reach the client, which is why the contact readings appeared to hang on
"measuring…" the first time: the values were arriving and the rows were being replaced, but the
replacements never rendered. The session view has always worked because it only ever mutates
properties on controls that already exist.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import traceback
from typing import Callable

import flet as ft

from nfb import Device, Mode

from . import chrome, displays, pupil_guide
from .widgets import num_box
from . import theme as t
from .inputs import SIMULATED_RESPONSE_MM, HeartChoice, InputChoice, MendiChoice, PupilChoice

RATES = (250.0, 500.0, 1000.0)
REFERENCES = ("linked", "bipolar")
MAINS = (50, 60)
# The two corners a voltage channel offers short of DC. 0.5 Hz is the vendor's EEG default; 0.01 Hz is what
# ultra-low-frequency work needs, and without it a sub-0.5 Hz reward measures the filter's roll-off. This is
# hardware filtering, so it happens before we see a sample — there is no software equivalent.
CORNERS = (0.5, 0.01)

# The amplifier's own beeper. On by default in the hardware, which is why stopping a session beeped at you;
# silent is our default because the host owns every sound a session makes. A control rather than a constant
# because the bit's polarity is inferred from the vendor's field name and nothing has measured it — see
# `optima.build_eeg_config`. If it is backwards, this row is where you find that out.
SILENCE = (True, False)

# The vendor driver grades contact at these thresholds: green below 10 kOhm, red above 30.
GOOD_KOHM = 10.0
FAIR_KOHM = 30.0
POLL_SECONDS = 0.25

# How long to let a contact check settle before acting on what it says. The first second or two of readings
# is transient while the measurement converges, so a channel judged empty inside that window would be judged
# for being slow rather than for being empty. Measured on hardware: settled well inside three seconds.
SETTLE_SECONDS = 3.0

# How often to ask an idle device whether it is still there. Slow enough to be free, fast enough that the
# screen is not lying about the connection for long.
ALIVE_SECONDS = 2.0


# Set NFB_DEBUG=1 to trace what the panel sees. On-screen symptoms have twice turned out to be
# invisible to Python-side tests, so this writes the panel's own view of state to a file.
DEBUG = os.environ.get("NFB_DEBUG")
LOG = "/tmp/nfb-panel.log"


def log(msg: str) -> None:
    if not DEBUG:
        return
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} {msg}\n")


def contact_colour(kohm: float) -> str:
    if kohm <= GOOD_KOHM:
        return t.GOOD
    if kohm <= FAIR_KOHM:
        return t.WARN
    return t.BAD


class _Choice:
    """A radio-style option whose selection is expressed by mutating style, never by rebuilding."""

    def __init__(self, label: str, on_pick: Callable[[], None]):
        self.text = ft.Text(label, size=t.BODY, color=t.TEXT)
        self.container = ft.Container(
            content=self.text,
            padding=ft.Padding(14, 8, 14, 8),
            border_radius=7,
            border=ft.Border.all(1, t.BORDER),
            on_click=lambda _e: on_pick(),
        )

    def set(self, selected: bool, enabled: bool) -> None:
        self.container.bgcolor = t.SURFACE_HI if selected else None
        self.container.border = ft.Border.all(1, t.ACCENT_DIM if selected else t.BORDER)
        self.container.opacity = 1.0 if (enabled or selected) else 0.45
        self.container.disabled = not enabled
        self.text.color = t.TEXT if (enabled or selected) else t.FAINT
        self.text.weight = ft.FontWeight.W_500 if selected else ft.FontWeight.W_400


MODALITIES = (
    ("EEG", "wired up: Neurobit Optima over serial"),
    ("HEG", "the Mendi fNIRS headband over Bluetooth, or a simulator"),
    ("HRV", "any strap advertising the standard Bluetooth Heart Rate Service, or the HeartMath on "
            "its USB cable"),
    ("Pupil", "an eye tracker on USB, or a simulated eye that answers an effort"),
)

PUPIL_TAB = 3      # the index of the Pupil modality, for returning straight to it after calibration

PUPIL_KINDS = (
    ("off", "no pupil source"),
    ("tracker", "the eye tracker on its USB cable. It measures each pupil itself, about 33 times a "
                "second. Connect it before you select this"),
    ("simulator", "a simulated eye at 60 readings a second. It blinks, it drifts, and it moves toward "
                  "the direction that the pupil screen asks for"),
)

PUPIL_EYES = ("both", "left", "right")

PUPIL_EYE_NOTES = {
    "both": "the mean of the two pupils. A reading where one eye is lost is rejected, because the two "
            "pupils are not the same size",
    "left": "the left pupil only. Use one eye if the other is often lost, for example behind a reflection",
    "right": "the right pupil only. Use one eye if the other is often lost, for example behind a "
             "reflection",
}

NEAR_MM, FAR_MM = 450.0, 950.0
"""Where the tracker can see the eyes, as the device states it."""


def _pupil_readout(sample, kind: str) -> tuple[str, str, str]:
    """The live line for the device screen: each pupil, and, for the tracker, where the eyes are."""
    if kind == "off":
        return "", "", t.MUTED
    if sample is None:
        return "no readings yet", "", t.MUTED

    def mm(value) -> str:
        return "—" if value is None else f"{value:.2f} mm"

    if sample.left_mm is None and sample.right_mm is None and sample.diameter_mm is not None:
        line = f"pupil {mm(sample.diameter_mm)}"
    else:
        line = f"left {mm(sample.left_mm)} · right {mm(sample.right_mm)}"
    if kind != "tracker":
        return line, "", t.MUTED
    distance = sample.distance_mm
    if distance is None:
        return line, "no eyes found · sit in front of the screen, with the tracker below it", t.WARN
    cm = distance / 10
    if distance < NEAR_MM:
        return line, f"eyes {cm:.0f} cm from the tracker · move back to 45 cm or more", t.WARN
    if distance > FAR_MM:
        return line, f"eyes {cm:.0f} cm from the tracker · move nearer, to 95 cm or less", t.WARN
    return line, f"eyes {cm:.0f} cm from the tracker · in range", t.GOOD


PUPIL_RESPONSES = {
    "none": "the eye does not answer an effort. Use this to see the screen for a person with no control yet",
    "weak": "a full effort moves the pupil 0.12 mm larger, and less when smaller. That is near the noise",
    "strong": "a full effort moves the pupil 0.35 mm larger, and less when smaller. That is a trained person",
}

MENDI_KINDS = (
    ("off", "no headband"),
    ("headband", "a Mendi headband over Bluetooth. Switch it on and make sure the phone app has not "
                 "claimed it — the connection is exclusive, so whoever pairs first keeps it"),
    ("simulator", "synthetic frames at 30/s, for working on the screen without wearing anything"),
)

HEART_KINDS = (
    ("off", "no heart-rate source"),
    ("strap", "a chest strap or the HeartMath Coherence Plus over Bluetooth — both advertise the "
              "standard Heart Rate Service, and both send one interval at a time"),
    ("usb", "the HeartMath on its USB-C cable, which sends the pulse waveform itself at 500 Hz and "
            "leaves the beat detection to us. Nothing to pair, and the sensor stays lit"),
    ("simulator", "synthetic beats, for working on the screens without strapping anything on"),
)


def view(page: ft.Page, device: Device, inputs: InputChoice | None = None,
         heart_choice: HeartChoice | None = None, heart_runner=None,
         mendi_choice: MendiChoice | None = None, mendi_runner=None,
         pupil_choice: PupilChoice | None = None, pupil_runner=None,
         on_calibrate: Callable[[], None] | None = None, initial_tab: int = 0,
         on_camera: Callable[[], None] | None = None) -> ft.Control:
    """No header of its own. The toolbar above already says where you are and how to leave — this screen
    used to add a second row carrying the word "Device" a second time plus a back button, which is the same
    mistake as the `FRE ›` row it replaced: a whole row of height for one control and one redundant word.
    The way back is the toolbar's crumb trail, which names the screen you actually came from."""

    choice = inputs if inputs is not None else InputChoice()
    src_choices: dict[str, _Choice] = {}
    path_field = ft.TextField(value=choice.path, hint_text="path to a recording or a sample export",
                              height=36, text_size=t.BODY, color=t.TEXT, bgcolor=t.SURFACE,
                              border_color=t.BORDER, expand=True,
                              content_padding=ft.Padding(10, 4, 10, 4))
    # Two explicit choices, not one toggle. A single "loop" pill that already rendered selected —
    # looping is the default — meant that pressing the thing labelled "loop" turned looping *off*.
    loop_choices = {True: _Choice("loop", lambda: set_loop(True)),
                    False: _Choice("play once", lambda: set_loop(False))}
    input_note = ft.Text("", size=t.SMALL, color=t.MUTED)
    input_row = ft.Row([path_field], spacing=8, visible=False)

    def set_kind(kind: str) -> None:
        choice.kind = kind
        refresh()

    def set_loop(on: bool) -> None:
        choice.loop = on
        refresh()

    def commit_path(_e=None) -> None:
        choice.path = str(path_field.value or "").strip()
        if choice.path:
            choice.kind = "file"
        refresh()

    path_field.on_submit = commit_path
    path_field.on_blur = commit_path

    picker = ft.FilePicker()
    page.services.append(picker)

    def browse(_e) -> None:
        async def run() -> None:
            try:
                files = await picker.pick_files(dialog_title="Play from file",
                                                allow_multiple=False)
            except Exception as exc:            # no native picker available: type the path
                input_note.value, input_note.color = f"{type(exc).__name__}: {exc}", t.WARN
                refresh()
                return
            if files:
                path_field.value = files[0].path
                commit_path()
        page.run_task(run)

    browse_btn = ft.Container(content=ft.Text("browse…", size=t.SMALL, color=t.ACCENT),
                              padding=ft.Padding(11, 8, 11, 8), border_radius=6,
                              border=ft.Border.all(1, t.BORDER), on_click=browse)
    src_choices["amplifier"] = _Choice("amplifier", lambda: set_kind("amplifier"))
    src_choices["file"] = _Choice("file", lambda: set_kind("file"))

    detail = ft.Text("", size=t.SMALL, color=t.MUTED)
    hint = ft.Text("", size=t.SMALL, color=t.MUTED)
    connect_label = ft.Text("", size=t.BODY, color=t.TEXT, weight=ft.FontWeight.W_500)
    connect_btn = ft.Container(content=connect_label, padding=ft.Padding(16, 9, 16, 9),
                               border_radius=7, border=ft.Border.all(1, t.BORDER))
    test_label = ft.Text("", size=t.BODY, color=t.TEXT, weight=ft.FontWeight.W_500)
    test_btn = ft.Container(content=test_label, padding=ft.Padding(16, 9, 16, 9),
                            border_radius=7, border=ft.Border.all(1, t.BORDER))

    rate_choices = {r: _Choice(f"{r:g} Hz", (lambda r_=r: lambda: apply(hz=r_))()) for r in RATES}
    ref_choices = {n: _Choice(n, (lambda n_=n: lambda: apply(reference=n_))()) for n in REFERENCES}
    mains_choices = {m: _Choice(f"{m} Hz", (lambda m_=m: lambda: apply(mains_hz=m_))())
                     for m in MAINS}
    corner_choices = {c: _Choice(f"{c:g} Hz", (lambda c_=c: lambda: apply(lower_limit_hz=c_))())
                      for c in CORNERS}
    silence_choices = {q: _Choice("silent" if q else "beeps",
                                  (lambda q_=q: lambda: apply(silent=q_))())
                       for q in SILENCE}

    # built once; refresh only changes colour and text
    MAX_CH = 4
    dots = {c: ft.Container(width=7, height=7, border_radius=4, bgcolor=t.BORDER)
            for c in range(MAX_CH)}
    labels = {c: ft.Text(f"ch{c}", size=t.BODY, color=t.TEXT, width=40) for c in range(MAX_CH)}
    readings = {c: ft.Text("—", size=t.BODY, color=t.FAINT) for c in range(MAX_CH)}
    ch_rows = {c: ft.Row([dots[c], labels[c], readings[c]], spacing=10) for c in range(MAX_CH)}

    # The shared reference gets its own row, because under linked or external wiring every channel's "−"
    # input is the *same* electrode: shown per channel it would be one number repeated four times, and the
    # failure it exists to catch — one bad reference contaminating all four channels at once — is a property
    # of that one electrode rather than of any channel.
    ref_dot = ft.Container(width=7, height=7, border_radius=4, bgcolor=t.BORDER)
    ref_reading = ft.Text("—", size=t.BODY, color=t.FAINT)
    ref_row = ft.Row([ref_dot, ft.Text("ref", size=t.BODY, color=t.MUTED, width=40), ref_reading],
                     spacing=10)
    contact_note = ft.Text("", size=t.SMALL, color=t.FAINT)
    excluded_note = ft.Text("", size=t.SMALL, color=t.MUTED)
    started_at = [0.0]                  # when the current check began, for the settle delay

    def refresh_input() -> None:
        for kind, ch in src_choices.items():
            ch.set(choice.kind == kind, True)
        input_row.visible = choice.kind == "file"
        for on, ch in loop_choices.items():
            ch.set(choice.loop == on, True)
        problem = choice.problem()
        if choice.kind != "file":
            input_note.value = "samples come from the amplifier below"
            input_note.color = t.MUTED
        elif problem:
            input_note.value, input_note.color = problem, t.BAD
        else:
            input_note.value = f"{choice.describe()} · the amplifier is not opened"
            input_note.color = t.MUTED
        if len(input_row.controls) == 1:
            input_row.controls = [path_field, browse_btn,
                                  *[c.container for c in loop_choices.values()]]

    def refresh() -> None:
        refresh_input()
        refresh_heart()
        refresh_heg()
        refresh_pupil()
        st = device.status
        log(f"refresh mode={st.mode.value} epoch={st.epoch} contact={device.contact} "
            f"subs={device.subscribers}")
        streaming = st.mode in (Mode.MEASURING, Mode.TESTING)
        settable = st.connected and not streaming

        connect_label.value = "Disconnect" if st.connected else "Connect"
        detail.value = (f"{st.model} · serial {st.serial} · firmware {st.firmware} · {st.port}"
                        if st.connected else (st.failure or "not connected"))
        if not st.connected and st.remedy:
            hint.value, hint.color = st.remedy, t.WARN
        elif streaming:
            hint.value, hint.color = f"settings locked while {st.mode.value}", t.MUTED
        else:
            hint.value = ""

        for r, ch in rate_choices.items():
            ch.set(st.settings.hz == r, settable)
        for n, ch in ref_choices.items():
            ch.set(st.settings.reference == n, settable)
        for m, ch in mains_choices.items():
            ch.set(st.settings.mains_hz == m, settable)
        for c, ch in corner_choices.items():
            ch.set(st.settings.lower_limit_hz == c, settable)
        for q, ch in silence_choices.items():
            ch.set(st.settings.silent == q, settable)

        testing = st.mode is Mode.TESTING
        if testing:
            test_label.value = "Stop test"
        elif st.mode is Mode.MEASURING:
            test_label.value = "Stop session and check"
        else:
            test_label.value = "Check contact"
        test_btn.opacity = 1.0 if st.connected else 0.45
        test_btn.disabled = not st.connected

        # Under linked or external referencing the "−" inputs are joined inside the device, so that
        # electrode is reported once below rather than repeated on every channel. Bipolar wiring gives each
        # channel its own pair, so both of its numbers belong on its own row.
        shared_ref = st.settings.reference in ("linked", "external")
        # Narrowing the check drops the channel count, and a row that disappeared would read as a channel
        # that stopped existing rather than as one with no electrode. So the rows a check *started* with
        # stay for as long as it runs, and the empty ones say what they are.
        excluded = device.excluded
        rows_shown = max(st.settings.channels, max(excluded) + 1 if excluded else 0)
        for c in range(MAX_CH):
            ch_rows[c].visible = c < rows_shown
            spot = device.contact.get(c)
            shown = None if spot is None else (spot.plus if shared_ref else spot.worst)
            if c in excluded:
                readings[c].value = "no electrode"
                readings[c].color = t.FAINT
                dots[c].bgcolor = t.BORDER
            elif spot is None or shown is None:
                readings[c].value = "measuring…" if testing else "—"
                readings[c].color = t.MUTED if testing else t.FAINT
                dots[c].bgcolor = t.BORDER
            else:
                colour = contact_colour(shown)
                readings[c].value = (
                    f"{spot.plus:.1f} kΩ" if shared_ref else
                    f"+ {spot.plus:.1f} · − {spot.minus:.1f} kΩ" if spot.minus is not None else
                    f"+ {spot.plus:.1f} kΩ")
                readings[c].color = colour
                dots[c].bgcolor = colour

        ref_row.visible = shared_ref
        # Read off the channels rather than kept separately: it *is* their "−" input, and taking the worst
        # of them is the same conservatism as `Contact.worst` — a reference that reads badly on any channel
        # is a reference that is going to spoil that channel.
        seen = [s.minus for c, s in device.contact.items()
                if c < st.settings.channels and s.minus is not None] if shared_ref else []
        if not shared_ref:
            # Emptied rather than merely hidden: a hidden control keeps its text, and a stale "shared by
            # every channel" then describes wiring that is not in use.
            ref_reading.value = ""
            ref_dot.bgcolor = t.BORDER
        elif not seen:
            ref_reading.value = "measuring…" if testing else "—"
            ref_reading.color = t.MUTED if testing else t.FAINT
            ref_dot.bgcolor = t.BORDER
        else:
            worst = max(seen)
            ref_reading.color = ref_dot.bgcolor = contact_colour(worst)
            ref_reading.value = f"{worst:.1f} kΩ · shared by every channel"
        contact_note.value = (
            f"{st.settings.reference} wiring · each channel's “+” electrode, and the one reference "
            f"every channel shares — a bad reference spoils all of them at once"
            if shared_ref else
            f"{st.settings.reference} wiring · both electrodes of each channel, “+” then “−”")

        # An excluded channel has to be named. It was dropped because it corrupts the others' readings, so
        # the numbers above are only trustworthy *because* of it — and a row that silently vanished would
        # otherwise look like a channel that stopped existing.
        gaps = [c for c in device.unwired() if c not in excluded]
        if excluded:
            names = ", ".join(f"ch{c}" for c in excluded)
            excluded_note.value = (f"{names} still listed, left out of the measurement: an empty channel "
                                   f"inflates the readings of the ones that have electrodes")
            excluded_note.color = t.MUTED
        elif gaps and len(gaps) < st.settings.channels:
            # Only a prefix of channels can be enabled, so an empty channel below a wired one cannot be
            # dropped. Saying so is the honest half of a fix that does not reach this case.
            names = ", ".join(f"ch{c}" for c in gaps)
            excluded_note.value = (f"{names} has no electrode and is inflating the readings above — "
                                   f"move your electrodes to the first channels, or unplug nothing and "
                                   f"read these as a ceiling")
            excluded_note.color = t.WARN
        else:
            excluded_note.value = ""
        try:
            page.update()
        except Exception:
            log("page.update() RAISED\n" + traceback.format_exc())
            raise
        log(f"  wrote rows: {[readings[c].value for c in range(st.settings.channels)]}")

    def apply(**changes) -> None:
        try:
            device.reconfigure(**changes)
        except (RuntimeError, ValueError) as exc:
            hint.value, hint.color = str(exc), t.BAD
        refresh()

    def toggle_connect(_e) -> None:
        if device.status.connected:
            device.close()
        else:
            device.connect()
        refresh()

    def toggle_test(_e) -> None:
        log(f"toggle_test pressed, mode={device.status.mode.value}")
        if device.status.mode is Mode.TESTING:
            device.stop()
        else:
            device.stop()          # a running session must end: the modes are exclusive
            started_at[0] = time.monotonic()
            try:
                device.start_impedance()
            except RuntimeError as exc:
                hint.value, hint.color = str(exc), t.BAD
                log(f"start_impedance FAILED: {exc}")
        log(f"toggle_test done, mode={device.status.mode.value}")
        refresh()

    def drop_unwired() -> None:
        """Once the readings have settled, re-run on the wired channels alone.

        The device drives the test on every enabled channel at once, and a channel with no electrode
        disturbs the ones that have them — measured, four-channel bipolar read a real 18 kOhm electrode as
        69. So this is a correction to a wrong number, not tidying, and it happens automatically because an
        operator cannot be expected to know that an unused channel spoils a used one.

        After the settle delay because the first second or two of readings is transient: acting on those
        would exclude a channel for being slow rather than for being empty.
        """
        if device.status.mode is not Mode.TESTING or device.excluded:
            return
        if time.monotonic() - started_at[0] < SETTLE_SECONDS:
            return
        try:
            dropped = device.narrow_to_wired()
        except RuntimeError:
            return
        if dropped:
            started_at[0] = time.monotonic()
            log(f"narrowed the contact check, dropped {dropped}")

    connect_btn.on_click = toggle_connect
    test_btn.on_click = toggle_test

    stop_polling = threading.Event()

    async def poll() -> None:
        """Contact readings change continuously while testing; nothing else here does.

        Scheduled with `page.run_task`, so it runs on Flet's own event loop. Mutating controls from a
        plain thread and calling `page.update()` does **not** reach the client — verified from a log
        showing the right values written to the right controls, `page.update()` not raising, and the
        screen unchanged. Everything that touches the UI has to be on Flet's loop.
        """
        log("poll task started (flet loop)")
        checked = [0.0]
        while not stop_polling.is_set():
            await asyncio.sleep(POLL_SECONDS)
            # An unplug while *idle* went unnoticed: the read loop is the only thing that touches the port
            # and it runs only while streaming, so this screen kept offering "Disconnect" beside the model
            # and serial of a device that was no longer there, and the first sign was a session that would
            # not start. Cheap enough to do on a timer — a missing port node costs a stat, and the state
            # query is two bytes.
            now = time.monotonic()
            if (device.status.connected and device.status.mode is Mode.IDLE
                    and now - checked[0] >= ALIVE_SECONDS):
                checked[0] = now
                if not device.check_alive():
                    log(f"liveness lost: {device.status.failure}")
                    try:
                        refresh()
                    except Exception:
                        return
            # The heart-rate source connects on its own thread, so what it is doing changes with nothing
            # being clicked: not started, opening, reading, lost, trying again. Without this the row
            # showed whatever was true at the instant of the last click — which after a click that
            # *starts* a connection is always "not started", the state it was in a millisecond before.
            # Only repainted when the words change, so a settled row costs one string comparison.
            # The same is true of the headband, and more so: connecting one runs a Bluetooth scan that
            # takes seconds, so the row would sit on "not started" for the whole of it.
            if heart_runner is not None or mendi_runner is not None or pupil_runner is not None:
                said = (heart_note.value, heart_status.value,
                        mendi_note.value, mendi_status.value, pupil_status.value,
                        pupil_live.value, pupil_place.value)
                try:
                    refresh_heart()
                    refresh_heg()
                    refresh_pupil()
                except Exception:
                    log("poll task DIED\n" + traceback.format_exc())
                    return
                if (heart_note.value, heart_status.value,
                        mendi_note.value, mendi_status.value, pupil_status.value,
                        pupil_live.value, pupil_place.value) != said:
                    page.update()
            if device.status.mode is Mode.TESTING:
                try:
                    drop_unwired()
                    refresh()
                except Exception:
                    log("poll task DIED\n" + traceback.format_exc())
                    return          # the view was replaced
        log("poll task stopped")

    page.run_task(poll)

    def panel(title: str, *content: ft.Control) -> ft.Control:
        return ft.Container(
            content=ft.Column([ft.Text(title, size=t.SMALL, color=t.FAINT),
                               ft.Container(height=10), *content], spacing=0),
            bgcolor=t.SURFACE, border=ft.Border.all(1, t.BORDER), border_radius=10,
            padding=ft.Padding(20, 16, 20, 18),
        )

    eeg_body = ft.Column(
        [
            panel("INPUT",
                  ft.Row([*[c.container for c in src_choices.values()],
                          ft.Container(width=14), input_note],
                         spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                  ft.Container(height=10), input_row),
            ft.Container(height=14),
            panel("CONNECTION",
                  ft.Row([connect_btn, ft.Container(width=16),
                          ft.Column([detail, hint], spacing=3)],
                         vertical_alignment=ft.CrossAxisAlignment.CENTER)),
            ft.Container(height=14),
            panel("ACQUISITION",
                  ft.Row([ft.Text("Sample rate", size=t.SMALL, color=t.MUTED, width=96),
                          *[c.container for c in rate_choices.values()]], spacing=8),
                  ft.Container(height=10),
                  ft.Row([ft.Text("Reference", size=t.SMALL, color=t.MUTED, width=96),
                          *[c.container for c in ref_choices.values()]], spacing=8),
                  ft.Container(height=10),
                  ft.Row([ft.Text("Mains", size=t.SMALL, color=t.MUTED, width=96),
                          *[c.container for c in mains_choices.values()],
                          ft.Container(width=12),
                          ft.Text("notched in the device's hardware", size=t.SMALL,
                                  color=t.FAINT)], spacing=8),
                  ft.Container(height=10),
                  ft.Row([ft.Text("High-pass", size=t.SMALL, color=t.MUTED, width=96),
                          *[c.container for c in corner_choices.values()],
                          ft.Container(width=12),
                          ft.Text("0.01 Hz for ultra-low-frequency work · also in the hardware, so a "
                                  "band below the corner measures the filter", size=t.SMALL,
                                  color=t.FAINT, expand=True)], spacing=8),
                  ft.Container(height=10),
                  ft.Row([ft.Text("Beeper", size=t.SMALL, color=t.MUTED, width=96),
                          *[c.container for c in silence_choices.values()],
                          ft.Container(width=12),
                          ft.Text("the amplifier's own, which beeps at every mode change · the console "
                                  "owns every other sound a session makes", size=t.SMALL,
                                  color=t.FAINT, expand=True)], spacing=8)),
            ft.Container(height=14),
            panel("ELECTRODE CONTACT",
                  ft.Row([test_btn, ft.Container(width=22),
                          ft.Column([*ch_rows.values(), ref_row], spacing=8)],
                         vertical_alignment=ft.CrossAxisAlignment.START),
                  ft.Container(height=12), contact_note, excluded_note,
                  ft.Text(f"green ≤ {GOOD_KOHM:g} kΩ · amber ≤ {FAIR_KOHM:g} kΩ · "
                          f"open circuit reads full scale · settles after a second or two",
                          size=t.SMALL, color=t.FAINT)),
        ],
        spacing=0,
    )

    # ---- HRV ----
    heart = heart_choice if heart_choice is not None else HeartChoice()
    heart_kind_choices = {k: _Choice(k, (lambda k_=k: lambda: set_heart_kind(k_))())
                          for k, _note in HEART_KINDS}
    heart_note = ft.Text("", size=t.SMALL, color=t.MUTED)
    heart_address = ft.TextField(value=heart.address, hint_text="address, or leave empty for any",
                                 height=36, text_size=t.BODY, color=t.TEXT, bgcolor=t.SURFACE,
                                 border_color=t.BORDER, expand=True,
                                 content_padding=ft.Padding(10, 4, 10, 4))
    heart_found = ft.Column(spacing=6)
    heart_status = ft.Text("", size=t.SMALL, color=t.MUTED)
    scan_label = ft.Text("look for one", size=t.SMALL, color=t.ACCENT)
    scan_btn = ft.Container(content=scan_label, padding=ft.Padding(11, 8, 11, 8), border_radius=6,
                            border=ft.Border.all(1, t.BORDER))
    heart_row = ft.Row([heart_address, scan_btn], spacing=8, visible=False)

    heart_port = ft.TextField(value=heart.port, hint_text="port, or leave empty to find it",
                              height=36, text_size=t.BODY, color=t.TEXT, bgcolor=t.SURFACE,
                              border_color=t.BORDER, expand=True,
                              content_padding=ft.Padding(10, 4, 10, 4))
    ports_label = ft.Text("list the cables", size=t.SMALL, color=t.ACCENT)
    ports_btn = ft.Container(content=ports_label, padding=ft.Padding(11, 8, 11, 8), border_radius=6,
                             border=ft.Border.all(1, t.BORDER))
    usb_row = ft.Row([heart_port, ports_btn], spacing=8, visible=False)
    heart_swap = threading.Lock()

    def set_heart_kind(kind: str) -> None:
        """Switch source, on a worker thread.

        Take effect now rather than at the next session: the whole point of choosing a source here is to
        see whether it connects, and a control whose effect you cannot observe is untestable by the
        person using it.

        **Off the event loop, though.** Releasing the old source waits up to two seconds for its reader
        to finish, and a reader that is halfway through a twenty-second Bluetooth scan takes all of it —
        so switching away from `strap` froze every control on the page for that long. Nothing here
        touches a control; `refresh` is what does, and the poll below runs it on Flet's own loop within
        a tick.
        """
        heart.kind = kind
        if heart_runner is not None:
            def swap() -> None:
                # Serialised: clicking three sources quickly must not have two threads releasing and
                # attaching the same runner at once, which is how you end up reading a source nobody
                # chose.
                with heart_swap:
                    heart_runner.release()
                    heart_runner.attach(heart)

            threading.Thread(target=swap, daemon=True, name="heart-swap").start()
        refresh()

    def commit_address(_e=None) -> None:
        heart.address = str(heart_address.value or "").strip()
        set_heart_kind("strap" if heart.address else heart.kind)

    heart_address.on_submit = commit_address
    heart_address.on_blur = commit_address

    def commit_port(_e=None) -> None:
        heart.port = str(heart_port.value or "").strip()
        set_heart_kind("usb" if heart.port else heart.kind)

    heart_port.on_submit = commit_port
    heart_port.on_blur = commit_port

    def list_ports(_e=None) -> None:
        """Which serial ports could be the pulse sensor. Instant, unlike the Bluetooth scan — this only
        reads what the operating system already knows about what is plugged in, so it runs here rather
        than on a worker thread."""
        from nfb.hrv import sources

        found = [(sources.describe_port(p) or "unnamed", p.device) for p in sources.usb_candidates()]
        heart_status.value = (f"{len(found)} found" if found else
                              "no cable found — the amplifier's own port is excluded on purpose, since "
                              "it is the same kind of chip")
        heart_status.color = t.MUTED if found else t.WARN
        heart_found.controls = [
            ft.Container(content=ft.Text(f"{name} · {device}", size=t.SMALL, color=t.TEXT),
                         padding=ft.Padding(10, 6, 10, 6), border_radius=6,
                         border=ft.Border.all(1, t.BORDER),
                         on_click=(lambda d=device: lambda _e: (
                             setattr(heart_port, "value", d), commit_port()))())
            for name, device in found]
        refresh()

    ports_btn.on_click = list_ports

    def do_scan(_e=None) -> None:
        """Blocking BLE discovery, on a worker thread, with the result written back on Flet's loop.

        A scan that returns nothing and says nothing leaves you unable to tell "no strap" from "did not
        run", so this reports the count either way.
        """
        scan_label.value = "scanning…"
        page.update()
        results: dict = {}

        def work() -> None:
            from . import heart as heart_mod
            results["found"], results["problem"] = heart_mod.scan()

        async def run() -> None:
            worker = threading.Thread(target=work, daemon=True, name="hrv-scan")
            worker.start()
            while worker.is_alive():
                await asyncio.sleep(0.2)
            scan_label.value = "look for one"
            trouble = results.get("problem")
            found = results.get("found") or []
            if trouble:
                heart_status.value, heart_status.color = trouble, t.BAD
            else:
                heart_status.value = (f"{len(found)} found" if found else
                                      "nothing found — switch the strap on, wet the contacts, and "
                                      "make sure a phone has not claimed it")
                heart_status.color = t.MUTED if found else t.WARN
            heart_found.controls = [
                ft.Container(content=ft.Text(f"{name} · {address}", size=t.SMALL, color=t.TEXT),
                             padding=ft.Padding(10, 6, 10, 6), border_radius=6,
                             border=ft.Border.all(1, t.BORDER),
                             on_click=(lambda a=address: lambda _e: (
                                 setattr(heart_address, "value", a), commit_address()))())
                for name, address in found]
            refresh()

        page.run_task(run)

    scan_btn.on_click = do_scan

    def refresh_heart() -> None:
        for kind, ch in heart_kind_choices.items():
            ch.set(heart.kind == kind, True)
        heart_row.visible = heart.kind == "strap"
        usb_row.visible = heart.kind == "usb"
        trouble = heart.problem()
        note = dict(HEART_KINDS)[heart.kind]
        if trouble:
            heart_note.value, heart_note.color = trouble, t.BAD
        else:
            heart_note.value, heart_note.color = note, t.MUTED
        if heart_runner is not None and heart.on:
            said = heart_runner.describe()
            # The note above and the line below are fed by two objects that can reach the same
            # conclusion — the choice knowing Bluetooth is missing, the runner finding out when it
            # tried. Printing it twice, once on each side of the port field, reads as two faults.
            heart_status.value = "" if said == heart_note.value else said
            heart_status.color = t.BAD if heart_runner.error else t.MUTED

    hrv_body = ft.Column([
        panel("SOURCE",
              ft.Row([*[c.container for c in heart_kind_choices.values()]], spacing=8),
              ft.Container(height=10), heart_note,
              ft.Container(height=10), heart_row,
              usb_row,
              ft.Container(height=10), heart_found,
              ft.Container(height=6), heart_status),
        ft.Container(height=14),
        panel("WHAT IT IS FOR",
              ft.Text("The HRV protocol trains this on its own — one spectral peak at 0.1 Hz, six "
                      "breaths a minute, breathing in as the heart accelerates.", size=t.BODY,
                      color=t.MUTED),
              ft.Container(height=8),
              ft.Text("It also appears beside the EEG on the Cross-frequency coupling screen, which "
                      "exists for one failure in particular: the EEG "
                      "can look like it is improving while the HRV splits into high-frequency and "
                      "very-low-frequency peaks with 0.1 Hz missing. You can only see that with both "
                      "in front of you.", size=t.SMALL, color=t.FAINT)),
    ], spacing=0)

    # ---- HEG ----
    # The HRV tab's twin, down to the swap being done off the event loop: connecting a headband runs a
    # Bluetooth scan of up to ten seconds inside `LiveMendi`, and releasing one waits for that reader, so
    # doing either on Flet's loop freezes the page for as long as it takes.
    headband = mendi_choice if mendi_choice is not None else MendiChoice()
    mendi_kind_choices = {k: _Choice(k, (lambda k_=k: lambda: set_mendi_kind(k_))())
                          for k, _note in MENDI_KINDS}
    mendi_note = ft.Text("", size=t.SMALL, color=t.MUTED)
    mendi_address = ft.TextField(value=headband.address,
                                 hint_text="address, or leave empty for any",
                                 height=36, text_size=t.BODY, color=t.TEXT, bgcolor=t.SURFACE,
                                 border_color=t.BORDER, expand=True,
                                 content_padding=ft.Padding(10, 4, 10, 4))
    mendi_found = ft.Column(spacing=6)
    mendi_status = ft.Text("", size=t.SMALL, color=t.MUTED)
    mendi_scan_label = ft.Text("look for one", size=t.SMALL, color=t.ACCENT)
    mendi_scan_btn = ft.Container(content=mendi_scan_label, padding=ft.Padding(11, 8, 11, 8),
                                  border_radius=6, border=ft.Border.all(1, t.BORDER))
    mendi_row = ft.Row([mendi_address, mendi_scan_btn], spacing=8, visible=False)
    mendi_swap = threading.Lock()

    def set_mendi_kind(kind: str) -> None:
        headband.kind = kind
        if mendi_runner is not None:
            def swap() -> None:
                with mendi_swap:
                    mendi_runner.release()
                    mendi_runner.attach(headband)

            threading.Thread(target=swap, daemon=True, name="mendi-swap").start()
        refresh()

    def commit_mendi_address(_e=None) -> None:
        headband.address = str(mendi_address.value or "").strip()
        set_mendi_kind("headband" if headband.address else headband.kind)

    mendi_address.on_submit = commit_mendi_address
    mendi_address.on_blur = commit_mendi_address

    def do_mendi_scan(_e=None) -> None:
        mendi_scan_label.value = "scanning…"
        page.update()
        results: dict = {}

        def work() -> None:
            from . import mendi as mendi_mod
            results["found"], results["problem"] = mendi_mod.scan()

        async def run() -> None:
            worker = threading.Thread(target=work, daemon=True, name="heg-scan")
            worker.start()
            while worker.is_alive():
                await asyncio.sleep(0.2)
            mendi_scan_label.value = "look for one"
            trouble = results.get("problem")
            found = results.get("found") or []
            if trouble:
                mendi_status.value, mendi_status.color = trouble, t.BAD
            else:
                mendi_status.value = (f"{len(found)} found" if found else
                                      "nothing found — switch the headband on and close the Mendi app, "
                                      "which holds the connection while it is paired")
                mendi_status.color = t.MUTED if found else t.WARN
            mendi_found.controls = [
                ft.Container(content=ft.Text(f"{name} · {address}", size=t.SMALL, color=t.TEXT),
                             padding=ft.Padding(10, 6, 10, 6), border_radius=6,
                             border=ft.Border.all(1, t.BORDER),
                             on_click=(lambda a=address: lambda _e: (
                                 setattr(mendi_address, "value", a), commit_mendi_address()))())
                for name, address in found]
            refresh()

        page.run_task(run)

    mendi_scan_btn.on_click = do_mendi_scan

    def refresh_heg() -> None:
        for kind, ch in mendi_kind_choices.items():
            ch.set(headband.kind == kind, True)
        mendi_row.visible = headband.kind == "headband"
        trouble = headband.problem()
        note = dict(MENDI_KINDS)[headband.kind]
        if trouble:
            mendi_note.value, mendi_note.color = trouble, t.BAD
        else:
            mendi_note.value, mendi_note.color = note, t.MUTED
        if mendi_runner is not None and headband.on:
            # Said once, on whichever side found it out — the same rule as the heart row above.
            said = mendi_runner.describe()
            mendi_status.value = "" if said == mendi_note.value else said
            mendi_status.color = t.BAD if mendi_runner.error else t.MUTED

    heg_body = ft.Column([
        panel("SOURCE",
              ft.Row([*[c.container for c in mendi_kind_choices.values()]], spacing=8),
              ft.Container(height=10), mendi_note,
              ft.Container(height=10), mendi_row,
              ft.Container(height=10), mendi_found,
              ft.Container(height=6), mendi_status),
        ft.Container(height=14),
        panel("WHAT IT IS FOR",
              ft.Text("The HEG screen trains blood flow over the forehead — three optical channels at "
                      "Fp1, Fp2 and a reference between them, projected into oxygenated and "
                      "deoxygenated haemoglobin.", size=t.BODY,
                      color=t.MUTED),
              ft.Container(height=8),
              ft.Text("The headband settles for eleven seconds after it starts, and holds both channels "
                      "flat while you move — so a calm-looking trace early on is the filters, not you.",
                      size=t.SMALL, color=t.FAINT)),
    ], spacing=0)

    # ---- Pupil ----
    # The tracker and the simulator both open in well under a second, but releasing a reader waits for its
    # thread — up to one USB read timeout for the tracker — so the swap is done off the event loop, as on
    # the other two tabs.
    eye = pupil_choice if pupil_choice is not None else PupilChoice()
    pupil_kind_choices = {k: _Choice(k, (lambda k_=k: lambda: set_pupil_kind(k_))())
                          for k, _note in PUPIL_KINDS}
    pupil_response_choices = {k: _Choice(k, (lambda k_=k: lambda: set_pupil_response(k_))())
                              for k in SIMULATED_RESPONSE_MM}
    pupil_eye_choices = {k: _Choice(k, (lambda k_=k: lambda: set_pupil_eye(k_))())
                         for k in PUPIL_EYES}
    pupil_note = ft.Text("", size=t.SMALL, color=t.MUTED)
    pupil_calibrate_btn = ft.Container(
        content=ft.Text("Calibrate", size=t.BODY, color=t.TEXT),
        padding=ft.Padding(16, 9, 16, 9), border_radius=7, border=ft.Border.all(1, t.BORDER),
        on_click=(lambda _e: on_calibrate()) if on_calibrate is not None else None)
    pupil_camera_btn = ft.Container(
        content=ft.Text("Camera", size=t.BODY, color=t.TEXT),
        padding=ft.Padding(16, 9, 16, 9), border_radius=7, border=ft.Border.all(1, t.BORDER),
        on_click=(lambda _e: on_camera()) if on_camera is not None else None)
    pupil_calibrate_row = ft.Column([
        ft.Text("calibration", size=t.SMALL, color=t.FAINT),
        ft.Container(height=8),
        ft.Row([pupil_calibrate_btn, pupil_camera_btn], spacing=8),
        ft.Container(height=10),
        ft.Text("Follow a dot on the screen. Look at each one for about a second, until its ring fills, "
                "then the next appears. Seven dots, about a minute. Pupil size does not need this; it "
                "lets the screen tell that the eyes stay on the fixation point.",
                size=t.SMALL, color=t.MUTED),
        ft.Container(height=8),
        ft.Text("Camera shows the tracker's infrared view, to line your eyes up. It stops the readings "
                "while it is open, because the tracker cannot do both at once.",
                size=t.SMALL, color=t.MUTED),
    ], spacing=0)
    pupil_response_note = ft.Text("", size=t.SMALL, color=t.MUTED)
    pupil_eye_note = ft.Text("", size=t.SMALL, color=t.MUTED)
    pupil_status = ft.Text("", size=t.BODY, color=t.TEXT)
    pupil_dot = ft.Container(width=9, height=9, border_radius=5, bgcolor=t.MUTED)
    pupil_status_row = ft.Row([pupil_dot, pupil_status], spacing=8,
                              vertical_alignment=ft.CrossAxisAlignment.CENTER, visible=False)
    pupil_hex = ft.Text("", size=t.SMALL, color=t.FAINT, selectable=True, visible=False)
    pupil_live = ft.Text("", size=t.BODY, color=t.TEXT)
    pupil_place = ft.Text("", size=t.SMALL, color=t.MUTED)
    import flet.canvas as _cv
    pupil_guide_canvas = _cv.Canvas(shapes=[], width=240, height=150)
    pupil_guide_status = ft.Text("", size=t.BODY, color=t.TEXT)
    pupil_guide_panel = panel(
        "POSITION",
        ft.Row([ft.Container(content=pupil_guide_canvas, width=240, height=150)],
               alignment=ft.MainAxisAlignment.CENTER),
        ft.Container(height=8), pupil_guide_status,
        ft.Container(height=6),
        ft.Text("The tracker goes below the screen, aimed at your face, about an arm's length away. "
                "Centre the two dots in the box.", size=t.SMALL, color=t.FAINT))
    pupil_response_row = ft.Column([
        ft.Text("response to effort", size=t.SMALL, color=t.FAINT),
        ft.Container(height=8),
        ft.Row([c.container for c in pupil_response_choices.values()], spacing=8),
        ft.Container(height=10), pupil_response_note,
    ], spacing=0, visible=False)
    # One choice for each screen that macOS reports a size for, and "other size" for the rest. Read once,
    # when the tab is built: a screen plugged in later appears the next time the device screen opens.
    found_displays = displays.displays()
    pupil_screen_choices = {str(d.id): _Choice(d.label, (lambda i=str(d.id): lambda: set_pupil_screen(i))())
                            for d in found_displays}
    pupil_screen_choices["manual"] = _Choice("other size", lambda: set_pupil_screen("manual"))
    pupil_screen_note = ft.Text("", size=t.SMALL, color=t.MUTED)
    pupil_size_row = ft.Row([
        num_box(eye.screen_width_mm, lambda v: set_pupil_size(width=v), suffix="mm",
                step=10, low=200, high=2000, width=60),
        ft.Text("wide", size=t.SMALL, color=t.FAINT),
        ft.Container(width=10),
        num_box(eye.screen_height_mm, lambda v: set_pupil_size(height=v), suffix="mm",
                step=10, low=100, high=1200, width=60),
        ft.Text("high", size=t.SMALL, color=t.FAINT),
    ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER, visible=False)
    pupil_tracker_rows = ft.Column([
        ft.Text("eye", size=t.SMALL, color=t.FAINT),
        ft.Container(height=8),
        ft.Row([c.container for c in pupil_eye_choices.values()], spacing=8),
        ft.Container(height=10), pupil_eye_note,
        ft.Container(height=14),
        ft.Text("screen", size=t.SMALL, color=t.FAINT),
        ft.Container(height=8),
        ft.Row([c.container for c in pupil_screen_choices.values()], spacing=8, wrap=True),
        ft.Container(height=10), pupil_size_row, pupil_screen_note,
        ft.Container(height=16), pupil_calibrate_row,
    ], spacing=0, visible=False)
    pupil_lamp = ft.Text("To check the pupil reading, look at the screen for ten seconds. Then hold a bright "
                "lamp near your face. Both values must become smaller by 1 mm or more in two seconds. "
                "If they do not change, the reading is not a pupil.", size=t.SMALL, color=t.FAINT)
    pupil_check = ft.Column([
        pupil_live, ft.Container(height=6), pupil_place,
        ft.Container(height=12), pupil_lamp,
    ], spacing=0)
    pupil_check_panel = panel("NOW", pupil_check)
    pupil_swap = threading.Lock()

    def restart_pupil() -> None:
        if pupil_runner is None:
            return

        def swap() -> None:
            with pupil_swap:
                pupil_runner.release()
                pupil_runner.attach(eye)

        threading.Thread(target=swap, daemon=True, name="pupil-swap").start()

    def set_pupil_kind(kind: str) -> None:
        eye.kind = kind
        restart_pupil()
        refresh()

    def set_pupil_response(response: str) -> None:
        eye.response = response
        # Changed on the running eye, so the next trial shows it without a new source and a lost record.
        source = getattr(pupil_runner, "source", None)
        if source is not None and hasattr(source, "response_mm"):
            source.response_mm = SIMULATED_RESPONSE_MM[response]
        refresh()

    def set_pupil_eye(which: str) -> None:
        eye.eye = which
        # The eye is chosen per reading, so the running connection takes it as it is.
        source = getattr(pupil_runner, "source", None)
        if source is not None and hasattr(source, "eye"):
            source.eye = which
        refresh()

    def set_pupil_screen(screen: str) -> None:
        eye.screen = screen
        # The size is sent when the connection opens, so a new size needs a new connection.
        if eye.kind == "tracker":
            restart_pupil()
        refresh()

    def set_pupil_size(width: float | None = None, height: float | None = None) -> None:
        if width is not None:
            eye.screen_width_mm = float(width)
        if height is not None:
            eye.screen_height_mm = float(height)
        set_pupil_screen("manual")

    def refresh_pupil() -> None:
        for kind, ch in pupil_kind_choices.items():
            ch.set(eye.kind == kind, True)
        for response, ch in pupil_response_choices.items():
            ch.set(eye.response == response, True)
        for which, ch in pupil_eye_choices.items():
            ch.set(eye.eye == which, True)
        width, height, why = eye.screen_size()
        # A screen that was selected and is now gone counts as automatic, as `screen_size` treats it.
        automatic = eye.screen not in pupil_screen_choices
        if automatic:
            auto, _why = displays.likely_tracker_display(found_displays)
            shown = str(auto.id) if auto is not None else "manual"
        else:
            shown = eye.screen
        for key, ch in pupil_screen_choices.items():
            ch.set(key == shown, True)
        pupil_size_row.visible = shown == "manual"
        if automatic and shown != "manual":
            pupil_screen_note.value = (f"{width:.0f} × {height:.0f} mm, chosen because it is {why}"
                                       + (". Select another screen if the tracker is on it"
                                          if len(found_displays) > 1 else ""))
        elif shown == "manual":
            pupil_screen_note.value = ("macOS reports no size for this screen, so type it. The tracker "
                                       "sends no data until it has a size" if not found_displays else
                                       "type the size of the screen that the tracker is on")
        else:
            pupil_screen_note.value = f"{width:.0f} × {height:.0f} mm, as the screen reports it"
        pupil_response_row.visible = eye.kind == "simulator"
        pupil_tracker_rows.visible = eye.kind == "tracker"
        pupil_response_note.value = PUPIL_RESPONSES[eye.response]
        pupil_eye_note.value = PUPIL_EYE_NOTES[eye.eye]
        trouble = eye.problem()
        note = dict(PUPIL_KINDS)[eye.kind]
        pupil_note.value, pupil_note.color = (trouble, t.BAD) if trouble else (note, t.MUTED)
        _tone = {"muted": t.MUTED, "accent": t.ACCENT, "good": t.GOOD, "bad": t.BAD}
        if pupil_runner is not None and eye.on:
            word, tone = pupil_runner.status()
            if tone == "bad":
                # Selecting the source again is the retry: a selected choice still answers a click.
                word += f" · select {eye.kind} again to try again"
            pupil_status.value = word
            pupil_status.color = _tone[tone]
            pupil_dot.bgcolor = _tone[tone]
            pupil_status_row.visible = True
            hexes = pupil_runner.handshake_hex if tone == "bad" else {}
            pupil_hex.value = (" · ".join(f"{name}: {h}" for name, h in hexes.items())
                               if hexes else "")
            pupil_hex.visible = bool(pupil_hex.value)
        else:
            pupil_status_row.visible = False
            pupil_hex.visible = False
        is_tracker = eye.kind == "tracker"
        pupil_guide_panel.visible = is_tracker
        if is_tracker:
            pos = pupil_runner.position if pupil_runner is not None else None
            pupil_guide_canvas.shapes = pupil_guide.guide_shapes(pos, 240, 150)
            word, tone = pupil_guide.guide_status(pos)
            pupil_guide_status.value = word
            pupil_guide_status.color = {"muted": t.MUTED, "accent": t.ACCENT, "good": t.GOOD,
                                        "bad": t.BAD}[tone]
        pupil_check_panel.visible = eye.on
        sample = getattr(pupil_runner, "sample", None) if pupil_runner is not None else None
        pupil_live.value, pupil_place.value, pupil_place.color = _pupil_readout(sample, eye.kind)
        pupil_place.visible = bool(pupil_place.value)
        pupil_lamp.visible = eye.kind == "tracker"

    pupil_body = ft.Column([
        panel("SOURCE",
              ft.Row([*[c.container for c in pupil_kind_choices.values()]], spacing=8),
              ft.Container(height=10), pupil_note,
              ft.Container(height=10), pupil_status_row,
              ft.Container(height=6), pupil_hex,
              ft.Container(height=14), pupil_response_row, pupil_tracker_rows),
        ft.Container(height=14),
        pupil_guide_panel,
        ft.Container(height=14),
        pupil_check_panel,
        ft.Container(height=14),
        panel("WHAT IT IS FOR",
              ft.Text("The Pupil screen trains voluntary control of pupil size. You make the pupil "
                      "larger or smaller with a mental strategy, and the sound tells you how far it "
                      "moved.", size=t.BODY, color=t.MUTED),
              ft.Container(height=8),
              ft.Text("A voluntary change is 0.1 to 0.5 mm. Light changes the pupil much more than "
                      "that, so keep the room light constant during a session.",
                      size=t.SMALL, color=t.FAINT)),
    ], spacing=0)

    panes = [
        ft.Container(content=eeg_body),
        ft.Container(content=heg_body, visible=False),
        ft.Container(content=hrv_body, visible=False),
        ft.Container(content=pupil_body, visible=False),
    ]

    def select(index: int) -> None:
        for i, pane in enumerate(panes):
            pane.visible = i == index
        page.update()

    tabs = chrome.Tabs([name for name, _note in MODALITIES], select, active=initial_tab)
    if initial_tab:
        select(initial_tab)

    refresh()
    # The body scrolls. It did not, and the electrode-contact panel — the one thing here you need before
    # deciding what to train — sat off the bottom of the window at the default size.
    return ft.Column(
        [tabs.control,
         ft.Container(content=ft.Column([ft.Stack(panes)], scroll=ft.ScrollMode.AUTO, expand=True),
                      padding=ft.Padding(20, 18, 20, 18), expand=True)],
        spacing=0, expand=True,
    )
