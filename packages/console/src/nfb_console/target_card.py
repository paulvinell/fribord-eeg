"""One target, as a card. Shared, because everything below the top row is protocol-independent.

The identity row differs per protocol — the amplitude screen needs a source menu and two band edges, an
CFC synchrony reward needs a centre frequency and a width — but the value, the threshold and its nudges,
the bar, the percentage and the mode are the same controls saying the same things, and each of them
encodes a decision that is easy to get wrong twice:

- **Remove is not next to the threshold nudges.** One of those discards the target and the others adjust
  it by 10%; they should not be neighbours. Remove goes in the identity row, at the far right.
- **The bar states its own range.** It spans 0 to a round ceiling with the threshold drawn as a notch and
  `0–20 µV · mark = threshold 12` underneath. It used to be value over twice the threshold with nothing
  written down, so half-full meant nothing at all.
- **On target is reported over the last 30 s**, not the session. A session percentage barely moves after
  the first few minutes, so it cannot say whether the last minute went well, and it returns "waiting"
  rather than a provisional number until a full window exists.
- **Calibration is a phase, not a mode.** A fixed threshold still has to come from somewhere, so the card
  counts down while it is being derived and then stops saying anything about it.
- **The gear belongs to the card, and the aim is in it.** How often this target should be met is what
  *places* the threshold, so every card needs it and no card should carry it on its face beside the number
  it produces. Screens contribute their own rows underneath — the amplitude screen its filter, the lock-in
  screens nothing — where before they handed the card a finished button, which left three screens with no
  gear at all and the aim with nowhere to live.
"""

from __future__ import annotations

from typing import Callable

import flet as ft
import numpy as np

from nfb.dsp import FILTER_KINDS

from . import theme as t
from .plots import fmt_amplitude, nice_ceiling
from .targets import MODES, RECENT_WINDOW, Target
from .widgets import help_button, labelled, num_box, pill, settings_button, tiny

BAR_W = 176
NOTE_H = 22
"""The note row's height, which it has whether or not it says anything.

**A card's height is not a readout.** The note used to be hidden when empty, so a verdict arriving or
leaving — "calibrating", "mostly amplitude, not phase" — grew or shrank the card and shoved every card
under it. On the synchrony screen that verdict sits on a measurement that wanders across its floor, so the
column pumped. One line is reserved on every card; a note is one line long by contract (`no_wrap`), and a
longer one is clipped rather than allowed to move the screen.
"""

SLOW_RECOMPUTE_SECONDS = 90.0
"""How often an automatic threshold re-derives on the two slow screens — coupling and synchrony.

**Both modes, everywhere, and automatic with a period is the default on the slow screens.** They used to
allow `fixed` only, on the argument that a tracker pins the share-above-threshold at the aim by
construction. That argument is true of every screen and the amplitude screen's answer to it was not to
forbid tracking but to re-derive on a period and hold the line between — which makes the percentage a real
number within each period while drift is still absorbed across them. Ninety seconds is the operator's
choice for these two; the window it draws from is the opening calibration's, so a line re-derived from a
twenty-second average still rests on a few independent readings.
"""

def _a(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


KIND_NOTES = {
    ("reward", "increase"): "the feedback follows it up, and the band rises",
    ("reward", "decrease"): "the feedback follows it down, and the band falls",
    ("inhibit", "increase"): "this suppresses rises, so the band falls",
    ("inhibit", "decrease"): "this suppresses falls, so the band rises",
    ("monitor", "increase"): "this is measured and drawn, and sends no output",
    ("monitor", "decrease"): "this is measured and drawn, and sends no output",
}
"""What each combination of role and direction means, in the one place a card explains itself.

**"Inhibit increase" is the entry this table exists for.** Its plain reading and its threshold point
opposite ways — rises in the band are what is being suppressed, so the target is met while the band stays
*low* — and a card that only said "met at or below the threshold" would read as contradicting its own
chip. See `targets.ROLES` for the full table and for why direction names the movement rather than the
wanted side of the line."""

# Filter design names stay off screen. What the operator needs is
# what the choice does; the measured numbers above it say the rest. dB is spelled out in `_ripple_line`
# and `_reject_line` rather than left as jargon.
FILTER_NOTES = {
    "cfir": "This design measures the band's amplitude directly, so it needs no smoothing stage. It "
            "reacts faster and more accurately than the other filters here: about 128 ms, against about "
            "296 ms for an order-2 bandpass filter. Set the delay this design uses. It has no order "
            "setting. Its delay stays the same at every frequency in the band.",
    "butterworth": "This is a standard bandpass filter. It is flat in the middle of the band. The gain "
                   "falls away smoothly towards each edge and beyond, so a neighbouring band leaks in a "
                   "little. Raising the order steepens the fall outside the band. It also increases the "
                   "delay by about the same amount. It does not change the variation inside the band. "
                   "This variation is always about 3 dB, because the band edges are the points where "
                   "the gain has halved in power.",
    "elliptic": "This filter rejects a neighbouring band far harder than a standard bandpass filter at "
                "the same order. Its gain ripples inside the band, and its delay varies more than "
                "threefold across it. It works well for deciding whether a band crossed a line. It works "
                "poorly for showing the shape of the signal.",
}


class TargetCard:
    """Build once, `refresh` per frame. `on_change` is called when something needs the list rebuilt."""

    def __init__(self, tgt: Target, identity: ft.Control, on_change: Callable[[], None],
                 bar_w: int = BAR_W, compact: bool = False, page=None,
                 settings: Callable[[Callable[[], None]], list] | None = None,
                 modes: tuple[str, ...] = MODES):
        self.tgt = tgt
        self.page = page
        # Which threshold modes this protocol allows — see `TrainingSession.threshold_modes`. **A protocol
        # can require that its thresholds hold still**, and two here do: a tracker follows the trained band
        # up and pins the percentage where it started, which erases the change the protocol exists to
        # produce. Leaving `automatic` pressable on those screens was a control that quietly turned the
        # protocol into a different one, which is the same hole the arrangement locking closed.
        self.modes = tuple(modes) or MODES
        # **Every card has a gear, and the aim is in it.** What else is worth setting per target differs by
        # protocol — a band's filter on the amplitude screen, nothing on the lock-in screens — so the screen
        # contributes rows and the card supplies the ones every target has. It used to be the other way
        # round: the screen passed a finished button, so a screen with nothing protocol-specific to offer
        # had no gear at all, and the aim had nowhere to live on three of the four screens that show cards.
        #
        # The aim goes here rather than on the card face because of what it is: the threshold's *cause*,
        # consulted when a target is being set up and not while it is being watched. On the face it would be
        # a second number next to the threshold, inviting the reading that they are two settings, when one
        # is derived from the other.
        self.contributed = settings
        self.settings = (settings_button(page, "Target", self._settings_rows)
                         if page is not None else None)
        self.on_change = on_change
        self.bar_w = bar_w
        colour = tgt.colour

        # **The live value is the one thing read from the chair, so it is the one big thing.** It was
        # 19 px beside an 11 px average and a 13 px threshold, which is three numbers of about one
        # weight; from arm's length none of them led.
        self.value_text = ft.Text("—", size=t.VALUE, color=t.TEXT, weight=ft.FontWeight.W_500)
        self.avg_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.pct_text = ft.Text("", size=t.SMALL, color=t.MUTED)
        self.thresh_text = ft.Text(f"{tgt.threshold:g}", size=t.BODY, color=t.TEXT)
        self.scale_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.note = ft.Text("", size=t.SMALL, color=t.WARN, no_wrap=True)
        self.fill = ft.Container(width=0, height=8, border_radius=4, bgcolor=colour,
                                 left=0, bottom=0)
        self.notch = ft.Container(width=2, height=14, bgcolor=t.TEXT, left=0, bottom=-3)
        # Shorter and dimmer than the first, because it is the subordinate line: two marks of equal
        # standing read as two settings.
        self.reach_notch = ft.Container(width=2, height=10, bgcolor=t.MUTED, left=0, bottom=-1,
                                        visible=tgt.reach_rate is not None)
        bar = ft.Stack([ft.Container(width=bar_w, height=8, border_radius=4,
                                     bgcolor=t.SURFACE_HI, left=0, bottom=0),
                        self.fill, self.reach_notch, self.notch], width=bar_w, height=14)

        self.latency_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.mode_row = ft.Row(spacing=6)
        self._build_modes()

        # Its own row, because alongside the mode pills and the redefine button it ran off the edge of a
        # fixed-width card. **Always present, at one line's height** — see `NOTE_H`.
        self.note_box = ft.Container(content=self.note, height=NOTE_H,
                                     alignment=ft.Alignment.BOTTOM_LEFT)

        if tgt.monitor:
            # **A monitor is a reading, three lines tall.** It cannot pay out, so a threshold, its nudges,
            # a percentage met, a redefine button and a mode are controls whose behaviour differs from
            # their label — the convention calls that a defect, and it was five of them per watched card.
            # What is left is what a monitor is: the identity row that can promote it, the number, and
            # the bar that gives the number a scale. `refresh` keeps the threshold fields current so a
            # promotion to reward finds them where it left them.
            self.notch.visible = False
            rows: list[ft.Control] = [
                identity,
                ft.Container(height=8),
                ft.Row([self.value_text, ft.Container(width=6), self.avg_text,
                        ft.Container(expand=True), self.latency_text],
                       spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=9),
                ft.Row([bar, ft.Container(width=12), self.scale_text], spacing=0,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ]
        else:
            rows = [
                identity,
                ft.Container(height=8),
                ft.Row([self.value_text, ft.Container(width=6), self.avg_text,
                        ft.Container(expand=True),
                        ft.Text("threshold", size=t.SMALL, color=t.FAINT),
                        self.thresh_text,
                        tiny("−", lambda _e: self._nudge(0.9)),
                        tiny("+", lambda _e: self._nudge(1.1))],
                       spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=9),
                ft.Row([bar, ft.Container(width=12), self.pct_text], spacing=0,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=4),
                # The scale line owns the row's slack, so a long one wraps under itself instead of
                # pushing the reaction time off the card's right edge — which is what the synchrony
                # screen's phase-sensitivity clause did.
                ft.Row([ft.Container(content=self.scale_text, expand=True), self.latency_text],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ]
            if not compact:
                rows += [ft.Container(height=7), self.mode_row, self.note_box]
            else:
                rows += [self.note_box]

        # **A filled well, not an outlined box.** The card sits on a panel that is itself bordered, inside
        # a window that borders the toolbar; a third outline inside the second made every screen a grid of
        # frames. The ground colour alone separates a card from the panel it is in.
        self.control = ft.Container(
            content=ft.Column(rows, spacing=0),
            bgcolor=t.BG, border=ft.Border.all(1, t.HAIRLINE), border_radius=8,
            padding=ft.Padding(12, 11, 12, 11))

    def _nudge(self, factor: float) -> None:
        self.tgt.nudge(factor)
        self.on_change()

    def _set_mode(self, mode: str) -> None:
        self.tgt.set_mode(mode)
        self.on_change()

    def _build_modes(self) -> None:
        tgt = self.tgt
        # **One allowed mode is the screen's fact, not the card's.** It was printed as a lone word on
        # every card, with the reason under it on every card — four identical sentences down one column.
        # A screen that allows one mode says so once, above its cards; here the row holds only what is
        # still a per-card action. `redefine` stays either way: re-deriving a fixed threshold from the
        # last 30 s is an operator action, not a change of mode.
        self.mode_row.controls = (
            [] if len(self.modes) == 1 else
            [pill(m, tgt.mode == m, (lambda m_=m: lambda _e: self._set_mode(m_))())
             for m in self.modes])
        # **How often an automatic threshold re-derives, beside the mode that does it.** Only under
        # `automatic`, because under `fixed` there is no period to set — `redefine` is the manual act. 0
        # reads as "continuous", which is what it has always been, so the control adds a choice without
        # changing anyone's default.
        if tgt.mode == "automatic":
            self.mode_row.controls.append(ft.Container(width=8))
            self.mode_row.controls.append(ft.Text("every", size=t.SMALL, color=t.FAINT))
            self.mode_row.controls.append(
                num_box(tgt.recompute_seconds, self._set_recompute, width=44, step=15, low=0,
                        suffix="s" if tgt.recompute_seconds else ""))
            if not tgt.recompute_seconds:
                self.mode_row.controls.append(
                    ft.Text("continuous", size=t.SMALL, color=t.FAINT))
        if tgt.can_redefine:
            # applicable to fixed only: automatic is already recomputing continuously
            if self.mode_row.controls:
                self.mode_row.controls.append(ft.Container(width=8))
            self.mode_row.controls.append(
                ft.Container(content=ft.Text("redefine", size=t.SMALL, color=t.ACCENT),
                             padding=ft.Padding(9, 5, 9, 5), border_radius=6,
                             bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE),
                             on_click=lambda _e: (tgt.redefine(), self.on_change())))
        if self.page is not None or self.settings is not None:
            self.mode_row.controls.append(ft.Container(expand=True))
            if self.settings is not None:
                self.mode_row.controls.append(self.settings)
                self.mode_row.controls.append(ft.Container(width=6))
            if self.page is not None:
                self.mode_row.controls.append(help_button(self.page, "Threshold", self._help_text()))

    def _settings_rows(self, redraw: Callable[[], None]) -> list:
        """What the gear opens: the aim, then whatever this screen adds.

        The aim first because it is the one every target has, and because it is the reason the number on the
        card is where it is — 90% met is a different threshold from 50% met on the same signal, and no other
        control on the card moves it.
        """
        tgt = self.tgt

        def set_aim(percent: float) -> None:
            tgt.set_aim(percent / 100.0)
            self.on_change()
            redraw()

        def set_reach(percent: float) -> None:
            tgt.set_reach(percent / 100.0)
            self.on_change()
            redraw()

        # **One row, and the rest is the `?`.** The first version spent two paragraphs on what the aim does
        # and what its extremes do — five lines of standing text above the controls anybody opened the gear
        # for, on a dialog that already has to hold a filter. Both paragraphs are explanation, and
        # explanation is what a `?` is for; what belongs beside a control is what the number in it means and
        # what it currently comes to.
        #
        # **Fractional, because the aims in use are.** A synchrony reward aims at 97.5%, so an integer box
        # could not display, let alone set, the value the protocol had already chosen — the control would
        # have rounded a live setting the moment it was opened.
        rows = [
            labelled("aim", num_box(round(tgt.hit_rate * 100, 1), set_aim,
                                    suffix="%", step=2.5, low=1, high=99, width=58),
                     note=f"of the time "
                          f"{'at or above' if tgt.wants_above else 'at or below'} "
                          f"the threshold, which follows from it",
                     width=64),
            ft.Text(f"{'Now' if tgt._settled else 'After calibration'} "
                    f"{tgt.describe_derivation(tgt.window if tgt.tracking else tgt.calibration_seconds)}.",
                    size=t.SMALL, color=t.FAINT),
        ]
        if tgt.reach_rate is not None:
            seconds = tgt.window if tgt.tracking else tgt.calibration_seconds
            rows += [
                labelled("higher line", num_box(round(tgt.reach_rate * 100, 1), set_reach,
                                                suffix="%", step=2.5, low=1, high=99, width=58),
                         note="of the time past it — every reward crossing at once is the moment "
                              "that is marked",
                         width=64),
                ft.Text(f"It comes from {tgt.describe_derivation(seconds, tgt.reach_rate)}.",
                        size=t.SMALL, color=t.FAINT),
            ]
        if self.contributed is not None:
            rows.append(ft.Container(height=2, bgcolor=t.BORDER))
            rows += self.contributed(redraw)
        return rows

    def _set_recompute(self, seconds: float) -> None:
        """A period a tracker holds still for. The reason to want one is that a continuously recomputed
        threshold pins percent-over-threshold at the aim, so the trained quantity cannot show a change."""
        self.tgt.recompute_seconds = max(0.0, round(seconds))
        self.on_change()

    def _help_text(self) -> str:
        tgt = self.tgt
        redefine = "**redefine**" if tgt.can_redefine else "redefine"
        return (
            f"This target is {_a(tgt.role)} **{tgt.role}** for {_a(tgt.direction)} {tgt.direction} in "
            f"its band: {KIND_NOTES[(tgt.role, tgt.direction)]}. It is met when its value is "
            f"{'at or above' if tgt.wants_above else 'at or below'} the threshold. It aims to be met "
            f"{tgt.hit_rate:.1%} of the time.\n\n"
            f"You set the aim behind the gear icon. The aim places the threshold: the line sits where "
            f"that share of your readings falls on the wanted side. Changing the aim moves the "
            f"threshold.\n\n"
            f"Keep the aim away from both ends. An aim near 100% puts the threshold below everything the "
            f"signal does. An aim near 0% puts it above everything. At either end, the target stops "
            f"showing anything, and the sound and brightness that follow it stop moving.\n\n"
            f"fixed — one number for the whole session, calculated from the first "
            f"{tgt.calibration_seconds:g} s of your signal. {redefine} calculates it again from the last "
            f"{tgt.calibration_seconds:g} s, whenever you press it.\n"
            f"automatic — recalculated from the last {tgt.window:g} s, so it follows the signal as its "
            f"amplitude drifts. "
            + (f"The period is now {tgt.recompute_seconds:g} s, so the threshold holds steady between "
               f"recalculations. The percentage means something within each period, while drift is "
               f"still absorbed between periods. "
               if tgt.recompute_seconds else
               f"The period is now 0, so this happens continuously. This holds time-on-target near "
               f"the aim and hides any real change. ")
            + f"The box after every sets the period.\n"
            + f"\nPressing − or + switches the mode to fixed, so your number is never overwritten.\n\n"
            f"Right now, the threshold comes from "
            f"{tgt.describe_derivation(tgt.window if tgt.tracking else tgt.calibration_seconds)}.")

    def refresh(self, value: float, unit: str | None = None,
                latency_seconds: float | None = None) -> None:
        tgt = self.tgt
        unit = unit if unit is not None else tgt.unit
        self.value_text.value = f"{value:.1f}"
        # A monitor is not wanted or unwanted; green on it would claim a verdict it does not give.
        self.value_text.color = t.GOOD if (tgt.wanted and not tgt.monitor) else t.TEXT
        self.avg_text.value = (f"avg {tgt.session_average:.1f} {unit}" if tgt.total else f"{unit}")
        self.thresh_text.value = f"{tgt.threshold:g}"
        # The higher line is inside the ceiling, or its mark pins to the right edge of every bar.
        ceiling = nice_ceiling(max(tgt.threshold * 1.5, tgt.reach_threshold * 1.2, value,
                                   tgt.session_average * 2, 1e-9))
        self.fill.width = self.bar_w * min(1.0, value / ceiling)
        self.notch.left = self.bar_w * min(1.0, tgt.threshold / ceiling) - 1
        self.scale_text.value = (f"0–{fmt_amplitude(ceiling, unit, True)}" if tgt.monitor else
                                 f"0–{fmt_amplitude(ceiling, unit, True)} · "
                                 f"mark = threshold {tgt.threshold:g}")
        if tgt.reach_rate is not None and not tgt.monitor:
            self.reach_notch.left = self.bar_w * min(1.0, tgt.reach_threshold / ceiling) - 1
            self.scale_text.value += f" · higher line {tgt.reach_threshold:g}"
        # How quickly the number follows a real change, measured for this exact filter. Static per target,
        # so free to show. Phrased as a reaction rather than as a delay, because a bare "300 ms behind" on
        # a card reads as a fault report rather than as a property of measuring a narrow band.
        if latency_seconds is None:
            self.latency_text.value = ""
        else:
            self.latency_text.value = (f"reacts in {latency_seconds * 1000:.0f} ms"
                                       if latency_seconds < 1.0 else
                                       f"reacts in {latency_seconds:.1f} s")
            self.latency_text.tooltip = (
                "When a burst of activity starts in this band, this is how long before the number shows "
                "half of it. A narrow band cannot be measured faster than this; the filter behind it is "
                "set from the gear.")
        # The aim belongs *next to* the achieved number, because comparing them is the only reason either
        # is on the card. On its own line underneath it was a standing sentence restating what the `?`
        # already says.
        recent = tgt.recent_percent()
        self.pct_text.value = ("waiting" if recent is None
                               else f"{recent:.0f}% of the last {RECENT_WINDOW:g} s · "
                                    f"aim {tgt.hit_rate:.0%}")
        self.pct_text.color = t.GOOD if (recent or 0) >= tgt.hit_rate * 100 else t.MUTED
        if tgt.calibrating:
            # **The state and its countdown, nothing else.** It read "calibrating — threshold set from the
            # first 120 s, 114 s left", which is an explanation, and on a column of four cards the same
            # explanation four times. Where the number comes from is the `?`'s sentence and the gear's;
            # what a card has to say is that it is not final yet, and for how long.
            self.note.value = f"calibrating · {tgt.calibration_remaining:.0f} s left"
            self.note.color = t.WARN
        elif tgt.tracking and tgt.window_remaining > 0:
            # The tracker's first window is a calibration too, and it is said the same way.
            self.note.value = f"provisional · {tgt.window_remaining:.0f} s left"
            self.note.color = t.WARN
        else:
            # nothing to say once it is running: the mode is a pill, the aim is beside the achieved
            # number, and the reasoning is behind the `?`
            self.note.value = ""


# --- the filter gear ------------------------------------------------------------------------

def _times(db: float) -> str:
    """A dB figure as the factor it is, which is the part anyone can act on. Capped, because past about
    two orders of magnitude the exact number is noise — "over 100× down" and "3420× down" mean the same
    thing to anyone deciding whether a neighbouring band leaks in."""
    ratio = 10 ** (abs(db) / 20)
    if ratio >= 100:
        return "over 100×"
    return f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×"


def _ripple_line(db: float) -> str:
    """How much the filter's gain varies across the frequencies it is supposed to pass, and so how
    differently two tones of the same size inside the band come out.

    Deliberately **not** called ripple in the text, because the two shapes get here differently and only
    one of them ripples. Butterworth's gain is flat in the middle and rolls off to −3 dB at each nominal
    edge, which is what the edges mean, so it reports about 3 dB whatever the order. Elliptic's is the
    ripple setting, so it reports roughly that. Same measurement, same consequence, different cause.
    """
    if db < 0.1:
        return "gain is flat across the band"
    return (f"{db:.1f} dB of variation across the band — the same tone reads up to {_times(db)} "
            f"bigger at its best frequency than at its worst")


def _reject_line(db: float, at_hz: float) -> str:
    """Rejection just outside: how far down a signal 1 Hz past the edge comes out. This is the number
    that says whether a neighbouring band leaks in, and it is what raising the order buys."""
    return f"{_times(db)} down at {at_hz:g} Hz, just outside it"

def filter_rows(session, tgt, on_change: Callable[[], None]) -> list:
    """Each kind exposes the parameter it actually has, and the consequences are printed.

    cFIR is designed around a delay, so it gets a delay. The IIR kinds have an order, and elliptic
    additionally has the ripple that defines its shape. Collapsing all of them into one abstract "how
    late may this be" hid the real parameter behind a friendlier one and then explained the friendlier
    one in dB.

    Envelope smoothing is **not** here even though it is part of the measurement, because it is one
    constant for the whole session and lives on the signal pane as step 1 of the chain. Per-target
    would be an invention: one global setting is simpler and sufficient.
    """
    design = session.filter_for(tgt)
    kind = design[0]

    def apply(**changes) -> None:
        session.retune_filter(tgt, **changes)
        on_change()

    rows = [
        labelled("measures", ft.Text(f"{tgt.band.low:g}–{tgt.band.high:g} Hz on "
                                     f"{tgt.source_label}", size=t.BODY, color=t.TEXT),
                 width=64),
        labelled("filter", *[pill(k, kind == k, (lambda k_=k: lambda _e: apply(kind=k_))())
                             for k in ("cfir", *FILTER_KINDS)], width=64),
    ]
    if kind == "cfir":
        rows.append(labelled("delay", num_box(design[1], lambda v: apply(cfir_delay=v),
                                              suffix="s", step=0.05, low=0.02, high=1.0),
                             note="what it is designed around", width=64))
    else:
        rows.append(labelled("order", num_box(design[1], lambda v: apply(order=int(v)),
                                              step=2, low=2, high=10, width=38),
                             note="per edge", width=64))
        if kind == "elliptic":
            rows.append(labelled("ripple", num_box(design[3], lambda v: apply(ripple=v),
                                                   suffix="dB", step=0.25, low=0.25, high=6.0),
                                 note="allowed unevenness inside the band", width=64))

    metric = session.metrics.get(session.key_for(tgt))
    if metric is not None:
        from scipy import signal as sps
        if kind == "cfir":
            w, h = sps.freqz(metric.estimator.taps, worN=8192, fs=session.rate)
        else:
            w, h = sps.sosfreqz(metric.filter.sos, worN=8192, fs=session.rate)
        mag = 20 * np.log10(np.abs(h) + 1e-15)
        inside = (w >= tgt.band.low) & (w <= tgt.band.high)
        peak = mag[inside].max()
        edge = min(tgt.band.high + 1.0, session.rate / 2 - 0.1)
        # The headline is *measured*, not group delay plus smoothing. That sum is a bound: it counts
        # the smoother's whole time constant, where the output is halfway there in rather less. Showing
        # the bound made an order-2 filter look 611 ms slow when it reacts in 332, and made cFIR — which
        # has no smoother, so its bound and its response nearly agree — look further ahead than it is.
        # Both lines are about the same event, stated as an event rather than as a quantity: a burst
        # of activity in this band begins. The headline is when the number shows half of it. The
        # detail is where that time goes, and it must account for all of it — "148 ms" under a
        # "100 ms group delay" with nothing said about the other 48 reads as an error.
        group_ms = metric.delay_seconds * 1000
        if metric.smoothing_seconds:
            parts = (f"{group_ms:.0f} ms of group delay, then this band's amplitude is averaged "
                     f"over {metric.smoothing_seconds * 1000:.0f} ms — the halfway point falls well "
                     f"before that averaging finishes. Set on the signal pane, step 1")
        else:
            parts = (f"{group_ms:.0f} ms of group delay, plus the time the filter takes to fill "
                     f"with the new signal")
        reacts = session.response_time(tgt)
        headline = ("—" if reacts is None
                    else f"a burst at {tgt.band.centre:g} Hz shows half its size after "
                         f"{reacts * 1000:.0f} ms")
        rows += [
            ft.Container(height=4),
            ft.Text(headline, size=t.BODY, color=t.TEXT),
            ft.Text(parts, size=t.SMALL, color=t.MUTED),
            # dB is the natural unit for both and meaningless to most people, so each is given with
            # what it does to an amplitude. 6 dB is a factor of two; 20 dB is a factor of ten.
            ft.Text(f"{_ripple_line(peak - mag[inside].min())} · "
                    f"{_reject_line(peak - mag[np.searchsorted(w, edge)], tgt.band.high + 1.0)}",
                    size=t.SMALL, color=t.MUTED),
        ]
    rows += [ft.Container(height=4),
             ft.Text(FILTER_NOTES[kind], size=t.SMALL, color=t.FAINT)]
    return rows
