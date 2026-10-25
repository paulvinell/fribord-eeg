"""How long a session runs, and what marks time inside it.

Open-ended by default. A maximum length is the exception, not the rule: you usually stop when the
session is done, and the cases that want a clock — eyes closed, or a protocol where the person cannot
watch the screen — are exactly the cases that also want audible cues, because a limit nobody can see
is not a limit. So the two live together here rather than being a number tucked into the toolbar.

Cues are scheduled against *elapsed* time, which excludes paused time, so pausing to fix an electrode
does not shift the reminders relative to the session's actual training minutes.

Both intervals are free numbers. There is no useful set of presets here: session lengths follow
protocols and people, and a reminder interval that is right for one is wrong for the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_LENGTH_MINUTES = 20.0       # what the box offers when a limit is first switched on
DEFAULT_REMINDER_MINUTES = 5.0


@dataclass
class SessionPlan:
    length_seconds: float | None = None      # None = open-ended, the default
    reminder_seconds: float | None = None    # None = no reminders
    reminder_cue: bool = True
    finish_cue: bool = True

    _reminders_fired: int = field(default=0, repr=False)
    _finish_fired: bool = field(default=False, repr=False)

    # --- state ---------------------------------------------------------------------------

    @property
    def bounded(self) -> bool:
        return self.length_seconds is not None

    def remaining(self, elapsed: float) -> float | None:
        if self.length_seconds is None:
            return None
        return max(0.0, self.length_seconds - elapsed)

    def complete(self, elapsed: float) -> bool:
        return self.length_seconds is not None and elapsed >= self.length_seconds

    def next_reminder(self, elapsed: float) -> float | None:
        """Seconds until the next reminder, or None if there are none coming."""
        if not self.reminder_seconds:
            return None
        due = (self._reminders_fired + 1) * self.reminder_seconds
        if self.length_seconds is not None and due > self.length_seconds:
            return None
        return max(0.0, due - elapsed)

    # --- editing -------------------------------------------------------------------------

    def set_length(self, minutes: float | None) -> None:
        """Changing the length rearms the finish cue: extending a finished session should still
        announce its new ending."""
        self.length_seconds = None if minutes is None else max(60.0, minutes * 60.0)
        self._finish_fired = False

    def set_reminder(self, minutes: float | None) -> None:
        self.reminder_seconds = None if not minutes else max(5.0, minutes * 60.0)
        self._reminders_fired = 0

    def rewind(self) -> None:
        self._reminders_fired = 0
        self._finish_fired = False

    # --- what is due ---------------------------------------------------------------------

    def due(self, elapsed: float) -> list[str]:
        """Cue names that have come due since the last call. Call it from the draw loop.

        A reminder missed while the app was busy is dropped rather than played late in a burst —
        four beeps at once tells the person nothing about time passing.
        """
        out = []
        if self.reminder_seconds:
            expected = int(elapsed // self.reminder_seconds)
            if self.length_seconds is not None:
                expected = min(expected, int(self.length_seconds // self.reminder_seconds))
            if expected > self._reminders_fired:
                self._reminders_fired = expected
                if self.reminder_cue and not self.complete(elapsed):
                    out.append("beep")
        if self.complete(elapsed) and not self._finish_fired:
            self._finish_fired = True
            if self.finish_cue:
                out.append("finish")
        return out
