"""Global chrome: one toolbar that does not move between screens, and tabs inside a screen.

The device, the transport and the clock belong to the *session*, not to whichever protocol screen is
showing, so they live here and work the same everywhere.

Where you are is written into that same row — `Fribord EEG › Amplitude`, hung off the logo, which is already
the way home. It used to be a second row below (`FRE › Amplitude`) and that was
wrong twice over: it spent a whole row of height on two words, and "FRE" is a category that has no
counterpart here, so it named nothing.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from typing import Callable

import flet as ft

from nfb import Device, Mode

from . import brand
from . import theme as t


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


class Toolbar:
    """Where you are, the device, the channels, the transport, the clock. Built once; `refresh`
    mutates."""

    CHANNEL_CHOICES = (1, 2, 3, 4)

    def __init__(self, device: Device, on_device: Callable[[], None],
                 on_home: Callable[[], None],
                 on_channels: Callable[[int], None] | None = None):
        self.device = device
        self.on_channels = on_channels
        self.channels = 4
        self.transport_glyph = ft.Text("▶", size=15, color=t.TEXT)
        self.transport = ft.Container(content=self.transport_glyph, width=40, height=32,
                                      border_radius=8, alignment=ft.Alignment.CENTER,
                                      border=ft.Border.all(1, t.BORDER))
        self.stop = ft.Container(content=ft.Text("■", size=11, color=t.MUTED), width=40, height=32,
                                 border_radius=8, alignment=ft.Alignment.CENTER,
                                 border=ft.Border.all(1, t.BORDER))
        # The record toggle lives beside the transport, for this reason:
        # recording is something you do to a session, not a setting you leave the screen for.
        self.record_glyph = ft.Text("●", size=11, color=t.MUTED)
        self.record = ft.Container(
            content=self.record_glyph, width=32, height=32,
            border_radius=ft.BorderRadius(top_left=8, bottom_left=8, top_right=0, bottom_right=0),
            alignment=ft.Alignment.CENTER, border=ft.Border.all(1, t.BORDER),
            tooltip="record this session — click to start or stop; the file finalises itself when "
                    "the session ends")
        # The *how* half of a split button, beside the *do it* half.
        self.record_caret = ft.Container(
            content=ft.Text("▾", size=9, color=t.FAINT), width=20, height=32,
            border_radius=ft.BorderRadius(top_left=0, bottom_left=0, top_right=8, bottom_right=8),
            alignment=ft.Alignment.CENTER,
            border=ft.Border(top=ft.BorderSide(1, t.BORDER), right=ft.BorderSide(1, t.BORDER),
                             bottom=ft.BorderSide(1, t.BORDER)),
            tooltip="where this recording goes, and in what format")
        self.record_split = ft.Row([self.record, self.record_caret], spacing=0)
        self.rec_note = ft.Text("", size=t.SMALL, color=t.BAD)
        self.recording = None
        """The `nfb.Recorder` currently on screen, set by the app. Read every refresh rather than told
        about changes, because the recorder finalises itself when an epoch ends and nothing else would
        repaint the dot."""
        self.record_supported = False
        self.clock_text = ft.Text("00:00", size=19, color=t.FAINT, weight=ft.FontWeight.W_300)
        self.length_text = ft.Text("", size=t.SMALL, color=t.FAINT)

        self.has_session = False
        self._channels_apply = True     # False on a screen whose instrument is not the amplifier
        self.dev_name = ft.Text("no device", size=t.BODY, color=t.MUTED)
        self.dev_badge = ft.Container(width=8, height=8, border_radius=4, bgcolor=t.FAINT)
        self.dev_chip = ft.Container(
            content=ft.Row([self.dev_badge, self.dev_name, ft.Text("›", size=13, color=t.FAINT)],
                           spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding(11, 7, 11, 7), border_radius=8,
            border=ft.Border.all(1, t.BORDER), on_click=lambda _e: on_device())
        # How many channels are in use, in the chrome next to the device, for
        # this reason: deciding to run two channels today is an ordinary choice made constantly, not
        # a setting you go to a settings screen for. It also removes traces and cards for inputs that do
        # not exist, which is most of the value.
        self.channels_label = ft.Text("4 ch", size=t.BODY, color=t.TEXT)
        self.channel_menu = ft.PopupMenuButton(
            content=ft.Container(
                content=ft.Row([self.channels_label, ft.Text("▾", size=9, color=t.FAINT)],
                               spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=ft.Padding(11, 7, 11, 7), border_radius=8,
                border=ft.Border.all(1, t.BORDER)),
            items=[ft.PopupMenuItem(
                content=ft.Text(f"{n} channel" + ("s" if n > 1 else ""), size=t.BODY),
                on_click=(lambda n_=n: lambda _e: self._pick_channels(n_))())
                for n in self.CHANNEL_CHOICES])
        self.channels_text = ft.Text("", size=t.SMALL, color=t.MUTED)

        # the mark and the name together are the way home, so the target is the whole lockup
        self.logo = mark = ft.Container(
            content=ft.Row([brand.mark(28), brand.wordmark(16)], spacing=10, tight=True,
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
            on_click=lambda _e: on_home())
        # Where you are, next to the way back — and a *trail*, because a screen can be reached from more
        # than one place. The device screen is reachable from the launcher and from inside any protocol, so
        # `Fribord EEG › Amplitude › Device` states which, and the middle segment is how you get back there. Before
        # this the screen carried its own header row with a back button and the word "Device" in it, which
        # was the second time the same word appeared and a whole row spent on one control.
        self.crumb_sep = ft.Text("›", size=t.BODY, color=t.FAINT, visible=False)
        self.parent_crumb = ft.Text("", size=t.BODY, color=t.MUTED)
        self.parent_box = ft.Container(content=self.parent_crumb, padding=ft.Padding(0, 2, 0, 2))
        # The middle segment and *its own* spacing live in one control, so hiding it removes the gap too.
        # Left as loose siblings, the spacers stayed behind and a one-segment crumb sat 21 px from its
        # separator while a two-segment one sat 7 — visible, and exactly the kind of thing that reads as
        # sloppiness rather than as a bug.
        self.parent_group = ft.Row(
            [self.parent_box, ft.Container(width=7), ft.Text("›", size=t.BODY, color=t.FAINT),
             ft.Container(width=7)],
            spacing=0, visible=False, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.crumb = ft.Text("", size=t.BODY, color=t.TEXT, weight=ft.FontWeight.W_500,
                             visible=False)

        # **Where a failure appears, on every screen, once.** A protocol screen's redraw runs ten times a
        # second, so a fault that survives one frame produces it again on the next — printing per frame
        # would bury the first and only occurrence that matters, and a modal would interrupt a session over
        # something the operator may not be able to act on. So: one strip, under the chrome that is already
        # on every screen, dismissable, and each distinct message shown once. See `report`.
        self.fault_text = ft.Text("", size=t.SMALL, color=t.BAD, expand=True, selectable=True)
        self.fault = ft.Container(
            content=ft.Row([self.fault_text,
                            ft.Container(content=ft.Text("×", size=t.BODY, color=t.MUTED),
                                         width=24, height=22, border_radius=5,
                                         alignment=ft.Alignment.CENTER,
                                         border=ft.Border.all(1, t.BORDER),
                                         on_click=lambda _e: self.clear_fault())],
                           spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding(18, 8, 18, 8), bgcolor=t.SURFACE, visible=False,
            border=ft.Border(bottom=ft.BorderSide(1, t.BAD)))
        self._seen_faults: set[str] = set()

        self.control = ft.Column([
            ft.Container(
                content=ft.Row(
                    [mark, ft.Container(width=10), self.crumb_sep, ft.Container(width=7),
                     self.parent_group, self.crumb,
                     ft.Container(width=14), self.dev_chip, ft.Container(width=8),
                     self.channel_menu, ft.Container(width=12),
                     self.channels_text, ft.Container(expand=True),
                     self.rec_note, ft.Container(width=10),
                     self.clock_text, ft.Container(width=6), self.length_text,
                     ft.Container(width=14), self.transport, ft.Container(width=6), self.stop,
                     ft.Container(width=6), self.record_split],
                    spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=ft.Padding(18, 12, 18, 12),
                border=ft.Border(bottom=ft.BorderSide(1, t.BORDER))),
            self.fault,
        ], spacing=0)

    def report(self, message: str) -> None:
        """Show a failure, and say it on the terminal — **the first time and not again**.

        Deduplicated on the message itself rather than counted, because the repeat rate of a broken frame
        says nothing: ten a second for as long as the screen is open is one fault, and a console filling at
        that rate is what makes the first line — the one with the traceback under it — unfindable.

        Dismissing it clears the strip but *not* the record, so the same fault does not come back a tenth of
        a second later and re-open something the operator has just closed. A genuinely new failure still
        gets through, which is the distinction worth keeping.
        """
        if message in self._seen_faults:
            return
        self._seen_faults.add(message)
        print(f"nfb: {message}", file=sys.stderr, flush=True)
        self.fault_text.value = message
        self.fault.visible = True

    def clear_fault(self) -> None:
        self.fault.visible = False
        self.fault_text.value = ""

    def _pick_channels(self, count: int) -> None:
        # the guard is here and not only on the control: `disabled` stops a click in the UI, but the
        # rule is "not while a session is running", and a rule worth having belongs where it is enforced
        if self.channel_menu.disabled or count == self.channels or self.on_channels is None:
            return
        self.on_channels(count)

    def set_channels(self, count: int) -> None:
        self.channels = count
        self.channels_label.value = f"{count} ch"

    def set_channels_enabled(self, enabled: bool) -> None:
        """Off while a session is actually running.

        Changing the count restarts acquisition — it rewrites the device config and changes which
        channels exist — so doing it mid-run would end the session. Refusing is better than silently
        discarding someone's twenty minutes.
        """
        self.channel_menu.disabled = not enabled
        self.channel_menu.opacity = 1.0 if enabled else 0.45

    def set_channels_visible(self, visible: bool) -> None:
        """Hide the count entirely on a screen that has no amplifier channels.

        Greying it out would be the wrong answer: disabled means "not now", and on the HRV screen the
        amplifier is not the instrument at all, so "1 ch" beside a pulse sensor is a control for a device
        this screen never touches. It sat there offering to reconfigure the EEG amplifier from a screen
        reading a heart.
        """
        self._channels_apply = visible
        self._apply_session_chrome()

    def set_crumb(self, name: str | None, parent: tuple[str, Callable[[], None]] | None = None) -> None:
        """Where you are, and optionally what you are inside.

        `parent` is a (label, go-there) pair and renders as a clickable middle segment. It is what "back"
        means on a screen reachable from several places: the label names the actual origin, so it cannot go
        stale the way a fixed "← back" pointing at the launcher did.
        """
        self.crumb.value = name or ""
        self.crumb.visible = self.crumb_sep.visible = bool(name)
        self.parent_crumb.value = parent[0] if parent else ""
        self.parent_group.visible = bool(parent and name)
        self.parent_box.on_click = (lambda _e: parent[1]()) if parent else None

    def bind_transport(self, on_toggle, on_stop) -> None:
        self.transport.on_click = on_toggle
        self.stop.on_click = on_stop

    def bind_record(self, on_toggle, on_options=None) -> None:
        self.record.on_click = on_toggle
        self.record_caret.on_click = on_options

    def set_record_supported(self, supported: bool) -> None:
        """Whether there is anything real to record behind this session.

        A replay is already a recording and a missing amplifier streams nothing, so the toggle is
        greyed rather than hidden — hidden would say the capability does not exist, and it does.
        """
        self.record_supported = supported

    def _paint_record(self) -> None:
        rec = self.recording
        writing = rec is not None and getattr(rec, "active", False)
        self.record.disabled = not self.record_supported
        self.record_split.opacity = 1.0 if self.record_supported else 0.4
        self.record_glyph.color = t.BAD if writing else t.MUTED
        edge = t.BAD if writing else t.BORDER
        self.record.border = ft.Border.all(1, edge)
        self.record_caret.border = ft.Border(top=ft.BorderSide(1, edge),
                                             right=ft.BorderSide(1, edge),
                                             bottom=ft.BorderSide(1, edge))
        if writing:
            note = f"rec {clock(rec.seconds)} · {rec.bytes_written / 1e6:.1f} MB"
            if rec.dropped:
                note += f" · {rec.dropped} frames dropped"
            self.rec_note.value = note
            self.rec_note.color = t.BAD
        elif rec is not None and rec.closed and rec.frames:
            self.rec_note.value = f"saved {rec.path.name}"
            self.rec_note.color = t.FAINT
        else:
            self.rec_note.value = ""

    def set_transport_visible(self, visible: bool) -> None:
        """Show or hide everything that only means something inside a session.

        The transport, the clock, *and* the channel count. Greying them out was not enough: a play button
        and a running clock on the launcher say a session exists and is merely paused, when nothing has
        been chosen yet. The channel count belongs to the same group by its own definition — changing it
        starts a fresh session — so offering it before there is a session to shape is the same mistake.
        What stays on the launcher is what belongs to the *app*: the logo, the device, its settings.

        The flag is re-applied by every `refresh()`, which a session calls twenty times a second. That
        redundancy is deliberate: the toolbar is one instance re-parented between screens, and a
        visibility change made while it was attached to the *previous* tree is exactly the kind of state
        that can be lost in the swap — which is how the play button went missing from a live session and
        left no way to start one.
        """
        self.has_session = visible
        self._apply_session_chrome()

    def _apply_session_chrome(self) -> None:
        for control in (self.transport, self.stop, self.clock_text, self.length_text):
            control.visible = self.has_session
        # **Two reasons to hide the channel count, and this one runs ten times a second.** It is hidden
        # outside a session, and hidden on a screen with no amplifier channels at all — and setting the
        # flag straight from `has_session` here silently undid the second within a tenth of a second of
        # any screen setting it. The screen's answer has to survive every refresh, so it is state.
        self.channel_menu.visible = self.has_session and self._channels_apply
        # The record toggle follows the same pair of flags: it records the amplifier's frames, so a
        # screen whose instrument is not the amplifier has nothing for it to record.
        self.record_split.visible = self.rec_note.visible = (self.has_session
                                                             and self._channels_apply)

    def set_transport_enabled(self, enabled: bool) -> None:
        self.transport.opacity = 1.0 if enabled else 0.4
        self.stop.opacity = 1.0 if enabled else 0.4
        self.transport.disabled = not enabled
        self.stop.disabled = not enabled

    def set_source(self, label: str = "", note: str = "replay · not the amplifier",
                   tone: str | None = None) -> None:
        """Name the input when it is **not** the amplifier, and keep naming it for the whole session.

        A replayed recording drew a screen identical to a live one. `make console` opens the last capture,
        and if that capture was made with an empty amplifier then every channel in it truthfully reports "no
        signal" — so the screen said the electrodes were bad, correctly, about a file from days ago. Hours
        went into the live path over a session that was never live.

        In the device chip rather than a corner, because the chip is the answer to "where are these numbers
        coming from" and a file is a different answer to the same question — as is a heart-rate monitor.
        **A protocol that does not use the amplifier must say what it does use here.** The HRV screen ran a
        connected pulse sensor under a chip reading "no device", which is true of the amplifier, useless as
        an answer, and reads as a fault.

        `note` is the second line, where the sample rate and reference go for the amplifier, and `tone` is
        the badge colour — the caller knows whether its own device is happy in a way this class cannot.
        """
        self._source = label
        self._source_note = note
        self._source_tone = tone
        self.refresh()

    def refresh(self, running: bool = False, elapsed: float = 0.0,
                length: float | None = None) -> None:
        self._apply_session_chrome()
        self._paint_record()
        st = self.device.status
        if getattr(self, "_source", ""):
            # Something other than the amplifier is the input — a recording, or another instrument
            # entirely — so the amplifier's state is not what these numbers came from and showing it would
            # be the misleading half of the truth.
            tone = getattr(self, "_source_tone", None) or t.WARN
            self.dev_name.value = self._source
            self.dev_name.color = tone
            self.dev_badge.bgcolor = tone
            self.channels_text.value = getattr(self, "_source_note", "replay · not the amplifier")
            self.set_channels_enabled(not running)
            self.transport_glyph.value = "‖" if running else "▶"
            self.clock_text.value = clock(elapsed)
            self.clock_text.color = t.TEXT if running else t.FAINT
            self.length_text.value = f"/ {clock(length)}" if length else ""
            return
        if st.mode is Mode.DISCONNECTED:
            self.dev_name.value = "no device"
            self.dev_name.color = t.MUTED
            self.dev_badge.bgcolor = t.BAD if st.failure else t.FAINT
        else:
            self.dev_name.value = st.model.replace("Neurobit ", "") or "device"
            self.dev_name.color = t.TEXT
            self.dev_badge.bgcolor = t.GOOD if st.mode is Mode.IDLE else t.ACCENT
        s = st.settings
        self.channels_text.value = (f"{s.hz:g} Hz · {s.reference} · {s.mains_hz} Hz mains"
                                    if st.connected else "")
        self.set_channels_enabled(not running)
        self.transport_glyph.value = "‖" if running else "▶"
        self.clock_text.value = clock(elapsed)
        self.clock_text.color = t.TEXT if running else t.FAINT
        self.length_text.value = f"/ {clock(length)}" if length else ""


# How often a protocol screen redraws. **10 Hz, not 20, and this is a real budget rather than a taste.**
# Measured on the amplitude screen: one draw costs 13-16 ms and emits ~4000 canvas points across six
# canvases, all of which Flet then serialises to the client. At 20 Hz that is a third of a core before the
# client has drawn anything, and it grows as the buffers fill — 2842 points at 20 s against 4048 at 140 s.
#
# Nothing on the screen changes fast enough to want 20. The raw trace advances ~62 columns per second, so at
# 10 Hz it gains six columns per frame; the target graph gains a point every half second; the numbers are
# read, not watched. What 20 Hz bought was a smoother-looking scroll, and what it cost was the responsiveness
# of everything else in the process — which is the opposite trade.
DRAW_SECONDS = 0.1


async def render_loop(page, session, draw: Callable[[], None], toolbar: Toolbar | None = None,
                      seconds: float = DRAW_SECONDS) -> None:
    """Every protocol screen's redraw, on Flet's own loop. One implementation, and it earns it twice.

    **A superseded view's loop has to end, and it did not.** Going to the device screen suspends the
    session without stopping it — checking contact mid-session is ordinary, and the session is what holds
    the elapsed clock, every calibration and every trend — so a loop written as `while not session._stop`
    kept running against controls that were no longer on the page. Coming back builds a *second* view with
    a *second* loop against the same session, and every further trip adds another. They are not idle:
    each one costs a full redraw and a `page.update()` ten times a second, on the same event loop that
    delivers clicks. That is a screen that keeps animating while the buttons stop responding, getting worse
    with each visit — which is exactly the failure this was written for. `session.redraw` is the newest
    view's own draw function, so identity against it is the test for "am I still the screen".

    **And one bad frame must not silently end the screen.** The old form was `except Exception: return`,
    which killed the refresh permanently, wrote nothing anywhere, and took the transport with it — the
    play button's handler calls `draw()` too. So a failure is reported once and drawing continues: a frame
    that cannot be built is usually a frame, not a corpse.
    """
    seen: set[str] = set()
    while not getattr(session, "_stop", False) and getattr(session, "redraw", draw) is draw:
        await asyncio.sleep(seconds)
        try:
            draw()
        except Exception as exc:
            where = traceback.extract_tb(exc.__traceback__)[-1]
            message = (f"{type(exc).__name__}: {exc} — while drawing "
                       f"{session.protocol.code}, at {where.filename.split('/')[-1]}:{where.lineno}")
            if message not in seen:
                seen.add(message)
                # The traceback goes with the *first* one, since that is the only copy anyone will read.
                traceback.print_exc()
            if toolbar is not None:
                toolbar.report(message)


class Tabs:
    """Tabs, at screen level or nested inside a pane.

    Present because not everything on a protocol screen changes at the same rate. Targets and traces
    are watched continuously; feedback settings are decided once and then left alone. Giving the
    settings a third of the window height charges rent for a tenant that never moves.
    """

    def __init__(self, labels: list[str], on_select: Callable[[int], None], active: int = 0,
                 inline: bool = False, dividers: tuple[int, ...] = ()):
        self.labels = labels
        self.active = active
        self.on_select = on_select
        self.inline = inline
        self.dividers = dividers
        """Indices after which a thin rule is drawn, for tabs that fall into groups. The feedback
        mechanisms are the case: Sound and Shadow follow the level while Click and Alert fire on the
        condition, and a flat row of five names showed nothing of that."""
        self.control = ft.Container(
            content=ft.Row([], spacing=2),
            padding=ft.Padding(0 if inline else 14, 0, 0, 0),
            bgcolor=None if inline else t.SURFACE,
            border=None if inline else ft.Border(bottom=ft.BorderSide(1, t.BORDER)))
        self._render()

    def _item(self, index: int, name: str) -> ft.Control:
        on = index == self.active
        return ft.Container(
            content=ft.Text(name, size=t.BODY, color=t.TEXT if on else t.MUTED,
                            weight=ft.FontWeight.W_500 if on else ft.FontWeight.W_400),
            padding=ft.Padding(14, 8 if self.inline else 10, 14, 8 if self.inline else 9),
            on_click=lambda _e, i=index: self.select(i),
            border_radius=7 if self.inline else 0,
            bgcolor=(t.SURFACE_HI if on else None) if self.inline else None,
            border=None if self.inline
            else ft.Border(bottom=ft.BorderSide(2, t.ACCENT if on else "#00000000")))

    def _render(self) -> None:
        items: list[ft.Control] = []
        for i, name in enumerate(self.labels):
            items.append(self._item(i, name))
            if i in self.dividers:
                items.append(ft.Container(width=1, height=16, bgcolor=t.BORDER,
                                          margin=ft.Margin(6, 0, 6, 0)))
        self.control.content.controls = items

    def select(self, index: int) -> None:
        if index == self.active:
            return
        self.active = index
        self._render()
        self.on_select(index)


def waiting(message: str = "waiting for data") -> ft.Control:
    """An explicit empty state. A blank graph looks identical to broken."""
    return ft.Container(content=ft.Text(message, size=t.SMALL, color=t.FAINT),
                        alignment=ft.Alignment.CENTER, expand=True)
