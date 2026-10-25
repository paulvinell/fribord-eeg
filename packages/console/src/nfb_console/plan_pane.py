"""The Session tab: the optional maximum length, and the cues that mark time inside it.

Shared by every protocol screen, and unchanged by which one. Two decisions it encodes:

**A session is open-ended by default** and nothing ends it but you. A maximum is the exception — eyes
closed, or a protocol where the person cannot watch the screen — so it lives here rather than in the
toolbar. Reach follows how often a control is right.

**Those same cases are why the length sits next to the cues.** A limit nobody can see is not a limit, so
there is a reminder beep on an optional interval and a tone at the end. Cues are scheduled on *elapsed*
time, so pausing to fix an electrode does not shift them, and a reminder missed during a stall is
dropped rather than replayed — four beeps at once say nothing about time passing.

Both numbers are typed rather than picked from presets. Session lengths follow protocols and people, and
a reminder interval that suits one is wrong for the next, so there is no set of pills that is right twice.
"""

from __future__ import annotations

import flet as ft

from . import chrome, cues
from . import theme as t
from .plan import DEFAULT_LENGTH_MINUTES, DEFAULT_REMINDER_MINUTES
from .widgets import labelled, num_box, panel, pill, test_button


class PlanPane:
    def __init__(self, page: ft.Page, session):
        self.page = page
        self.session = session
        self.plan = session.plan
        self.rows = ft.Column(spacing=14)
        self.clock = ft.Text("", size=26, color=t.TEXT, weight=ft.FontWeight.W_300)
        self.note = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.control = ft.Container(
            content=ft.Column([
                panel("SESSION",
                      ft.Row([self.clock, ft.Container(width=14), self.note],
                             vertical_alignment=ft.CrossAxisAlignment.CENTER),
                      ft.Container(height=16),
                      self.rows),
            ], spacing=0, scroll=ft.ScrollMode.AUTO, expand=True),
            padding=ft.Padding(14, 12, 14, 14), expand=True, visible=False)
        self.rebuild()

    def _set_length(self, minutes: float | None) -> None:
        self.plan.set_length(minutes)
        self.session.finished = False
        self.rebuild()

    def _set_reminder(self, minutes: float | None) -> None:
        self.plan.set_reminder(minutes)
        self.rebuild()

    def _toggle(self, attr: str) -> None:
        setattr(self.plan, attr, not getattr(self.plan, attr))
        self.rebuild()

    def rebuild(self) -> None:
        plan = self.plan
        problem = cues.problem()
        length_box = ([num_box(plan.length_seconds / 60, self._set_length, suffix="min")]
                      if plan.bounded else [])
        reminder_box = ([num_box(plan.reminder_seconds / 60, self._set_reminder, suffix="min")]
                        if plan.reminder_seconds else [])
        self.rows.controls = [
            labelled("maximum",
                     pill("open-ended", not plan.bounded, lambda _e: self._set_length(None)),
                     pill("limit", plan.bounded,
                          lambda _e: self._set_length(DEFAULT_LENGTH_MINUTES)),
                     *length_box),
            labelled("remind every",
                     pill("never", not plan.reminder_seconds, lambda _e: self._set_reminder(None)),
                     pill("interval", bool(plan.reminder_seconds),
                          lambda _e: self._set_reminder(DEFAULT_REMINDER_MINUTES)),
                     *reminder_box),
            labelled("cues",
                     pill("reminder beep", plan.reminder_cue,
                          lambda _e: self._toggle("reminder_cue")),
                     test_button("hear it", lambda _e: cues.play("beep"), problem is None),
                     ft.Container(width=14),
                     pill("finishing tone", plan.finish_cue,
                          lambda _e: self._toggle("finish_cue")),
                     test_button("hear it", lambda _e: cues.play("finish"), problem is None),
                     note=(problem or "")),
        ]
        self.page.update()

    def refresh(self) -> None:
        """The clock counts *up* when open-ended and *down* when there is a limit, because those are
        different questions: how long have I been here, against how long is left."""
        if not self.control.visible:
            return              # a hidden pane must cost nothing; see SpectrumPane.refresh
        session, plan = self.session, self.plan
        remaining = plan.remaining(session.elapsed)
        if remaining is None:
            self.clock.value = chrome.clock(session.elapsed)
            self.note.value = "elapsed · open-ended"
        else:
            self.clock.value = chrome.clock(remaining)
            self.note.value = (f"remaining of {plan.length_seconds / 60:g} min"
                               if not session.finished else "session complete")
        nxt = plan.next_reminder(session.elapsed)
        if nxt is not None:
            self.note.value += f" · next beep in {chrome.clock(nxt)}"
