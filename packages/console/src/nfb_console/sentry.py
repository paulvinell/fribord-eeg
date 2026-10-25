"""The signal sentry: the difference between a blink and a recording being destroyed.

Every other quality indicator describes *now* and assumes somebody is watching; a sleep recording is
made precisely when nobody is. So this watches duration — a channel unusable for the alarm's patience
(continuously *or* accumulated, so a flapping contact cannot hide between the gaps), every channel
gone at once, a channel delivering nothing but artifact for `GARBAGE_SECONDS`, a starving feed, the
feed thread dying, or the recording itself failing — and it can be loud: the beep repeats every
`repeat_seconds` where `audible` is armed, and clears itself on recovery. Sustained moderate artifact
is the advisory tier, visual only — rolling over in bed must never beep. Wall-clock throughout;
transitions land in `events`, which is what the sidecar file is written from.
"""

from __future__ import annotations

import time
from collections import deque

ALARM_SECONDS = 60.0
"""The alarm's patience: how much unusable time a channel gets before the alarm. Long enough that
turning over never trips it; short enough that a detached electrode costs a minute, not the night."""

UNUSABLE_FRACTION = 0.8
"""A flapping contact — railed most of the time, moments of real signal between — never stays bad
long enough for a continuity test, and it is a common way for a taped electrode to die. So unusable
time also *accumulates*: the alarm trips once a channel has been bad for `alarm_seconds` in total
within a window of `alarm_seconds / UNUSABLE_FRACTION`, however the badness was distributed."""

RECOVERY_SECONDS = 5.0
"""An accumulated trip holds through the flapping's own good moments — clearing on each one would
re-fire the fresh-alarm beep on every flap — and releases once the channel has been good this long."""

REPEAT_SECONDS = 30.0
"""How often the beep re-fires while an alarm holds. A single beep is a sound a sleeper integrates
into a dream; a patient repeat is what actually wakes one."""

STARVED_SECONDS = 10.0
"""No blocks from a connected device for this long is an alarm of its own. The device streams ~30
frames a second while measuring, so ten silent seconds is a dead link, not jitter."""

ADVISORY_REJECTION = 0.5
ADVISORY_SECONDS = 300.0
"""Half the signal thrown away for five minutes: the recording is still usable, but something —
drying gel, a loosening strap — is worth a look. Advisory, never audible."""

GARBAGE_REJECTION = 0.8
GARBAGE_SECONDS = 600.0
"""Nearly everything rejected, for ten sustained minutes: signal is arriving and none of it is
usable — an electrode nearly off, dangling against nothing, or on the wrong tissue. This is the
failure the absence detectors cannot see, so it has its own audible tier, stricter and slower than
the advisory precisely so movement in bed has to have cleared long before it can fire."""

EVENT_LIMIT = 500


class SignalSentry:
    """Watches a session's resolved channel states and says when the recording is in danger.

    Fed from the feed thread (`tick` per block group is cheap and rate-limited) and from a screen's
    draw loop, which is what keeps the starvation alarm alive when blocks stop arriving. `audible`
    is off by default; the sleep screen arms it. `recorder` is set by the app while one is writing,
    so a recording that dies is an alarm and not a toolbar footnote.
    """

    def __init__(self) -> None:
        self.audible = False
        # The operator's numbers, set on the sleep screen's ALARM panel and persisted with the
        # calibration — the same kind of number as a volume, wrong differently for everyone.
        self.alarm_seconds = ALARM_SECONDS
        self.repeat_seconds = REPEAT_SECONDS
        self.volume = 100.0
        self.recorder = None
        self.problem: str | None = None
        self.advice: str | None = None
        self.events: list[tuple[float, str]] = []
        self._bad_since: dict[int, float] = {}
        self._bad_log: dict[int, deque] = {}
        self._last_bad: dict[int, float] = {}
        self._hot_since: dict[int, float] = {}
        self._garbage_since: dict[int, float] = {}
        self._problem_key: tuple | None = None
        self._last_block: float | None = None
        self._last_beep: float | None = None
        self._last_eval: float | None = None

    def saw_block(self, now: float | None = None) -> None:
        self._last_block = time.monotonic() if now is None else now

    def unusable_seconds(self, now: float | None = None) -> dict[int, float]:
        """Per channel, how long it has currently been unusable. For a health panel's tally."""
        now = time.monotonic() if now is None else now
        return {c: now - since for c, since in self._bad_since.items()}

    def tick(self, session, now: float | None = None) -> str | None:
        now = time.monotonic() if now is None else now
        if self._last_eval is not None and now - self._last_eval < 0.5:
            return self.problem
        self._last_eval = now

        window = self.alarm_seconds / UNUSABLE_FRACTION
        for channel in session.specs:
            bad = session.state.get(channel) is False
            log = self._bad_log.setdefault(channel, deque())
            log.append((now, bad))
            while len(log) > 1 and log[1][0] <= now - window:
                log.popleft()
            if bad:
                self._bad_since.setdefault(channel, now)
                self._last_bad[channel] = now
            else:
                self._bad_since.pop(channel, None)
            rejection = session.rejection(channel)
            if rejection >= ADVISORY_REJECTION:
                self._hot_since.setdefault(channel, now)
            else:
                self._hot_since.pop(channel, None)
            if rejection >= GARBAGE_REJECTION:
                self._garbage_since.setdefault(channel, now)
            else:
                self._garbage_since.pop(channel, None)

        problem, key = self._verdict(session, now)
        self._advise(session)

        # The key is the condition's identity; the message carries a running duration, so comparing
        # messages read every tick as a fresh alarm — an immediate re-beep and an event per tick.
        if key != self._problem_key:
            self._problem_key = key
            self._note(now, problem or "signal recovered")
            self._last_beep = None          # a fresh alarm sounds at once, not up to 30 s late
        self.problem = problem
        if problem and self.audible and (self._last_beep is None
                                         or now - self._last_beep >= self.repeat_seconds):
            self._last_beep = now
            session.cue("beep", self.volume)
        return self.problem

    def _tripped(self, session, now: float) -> list[int]:
        """Channels past the alarm's patience: continuously bad, or bad for that much *accumulated*
        time within the wider window — held through `RECOVERY_SECONDS` of good so flapping cannot
        re-fire the fresh-alarm beep on every gap."""
        out = set(c for c, since in self._bad_since.items()
                  if now - since >= self.alarm_seconds)
        for channel in session.specs:
            recently = now - self._last_bad.get(channel, -1e18) <= RECOVERY_SECONDS
            if channel not in out and recently \
                    and self._bad_time(channel, now) >= self.alarm_seconds:
                out.add(channel)
        return sorted(out)

    def _bad_time(self, channel: int, now: float) -> float:
        """Accumulated unusable seconds in the duty window, holding each sample's state until the
        next — which matches how the states actually arrive, one verdict per block."""
        log = self._bad_log.get(channel)
        if not log:
            return 0.0
        total = 0.0
        entries = list(log) + [(now, log[-1][1])]
        for (t0, bad), (t1, _next) in zip(entries, entries[1:]):
            if bad:
                total += t1 - t0
        return total

    def _verdict(self, session, now: float) -> tuple[str | None, tuple | None]:
        """The alarm cascade, worst first: software death, the recording, then the electrodes."""
        error = getattr(session, "error", None)
        if error:
            return f"the feed died: {error} — restart the session", ("feed",)
        rec = self.recorder
        if rec is not None and (getattr(rec, "error", "")
                                or (getattr(rec, "closed", False)
                                    and getattr(session, "feeding", False))):
            why = getattr(rec, "error", "") or "it stopped while the session is still running"
            return f"the recording has stopped — {why}", ("recording",)
        tripped = self._tripped(session, now)
        if tripped and len(tripped) == len(session.specs) and len(session.specs) > 1:
            return ("every channel unusable — with a linked reference, check the reference "
                    "electrode first"), ("reference",)
        if tripped:
            names = ", ".join(f"ch{c}" for c in tripped)
            steady = max((now - self._bad_since[c] for c in tripped if c in self._bad_since),
                         default=0.0)
            spell = (_span(steady) if steady >= self.alarm_seconds
                     else f"most of the last {_span(self.alarm_seconds / UNUSABLE_FRACTION)}")
            return f"{names} unusable for {spell} — reattach the electrode", \
                ("channels", tuple(tripped))
        garbage = sorted(c for c, since in self._garbage_since.items()
                         if now - since >= GARBAGE_SECONDS)
        if garbage:
            names = ", ".join(f"ch{c}" for c in garbage)
            return (f"{names} nothing but artifact for {_span(GARBAGE_SECONDS)} — signal is "
                    f"arriving but none of it is usable; check the electrode"), \
                ("garbage", tuple(garbage))
        if (session.device is not None and session.feeding and self._last_block is not None
                and now - self._last_block >= STARVED_SECONDS):
            return (f"no samples for {_span(now - self._last_block)} — "
                    f"check the device and its battery"), ("starved",)
        return None, None

    def _advise(self, session) -> None:
        now = self._last_eval
        notes = []
        hot = sorted(c for c, since in self._hot_since.items()
                     if now - since >= ADVISORY_SECONDS)
        if hot:
            notes.append(", ".join(f"ch{c}" for c in hot)
                         + " rejecting more than half its signal — check the electrode before "
                           "it fails outright")
        rec = self.recorder
        if rec is not None and getattr(rec, "dropped", 0):
            notes.append(f"the recording dropped {rec.dropped} frames — the disk is not keeping up")
        self.advice = " · ".join(notes) if notes else None

    def _note(self, _now: float, message: str) -> None:
        if not self.events and message == "signal recovered":
            return                          # nothing was ever wrong; there is nothing to record
        # Wall epoch, not the monotonic clock the logic runs on: these are read back from a sidecar
        # file the morning after, where only a real timestamp means anything.
        self.events.append((time.time(), message))
        del self.events[:-EVENT_LIMIT]


def _span(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"
