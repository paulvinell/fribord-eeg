"""The spindle card: the spindles on one channel, counted as they come, and the search for their frequency.

**A spindle is an event, not a target.** It has a sound of its own, played at the moment the detector
confirms it, and it reaches nothing the targets drive: not the level, a target's threshold, the music, the
volume or the shadow. So the card has no aim and no bar, its threshold is the detector's multiple of the
usual amplitude, and the session plays its sound from the feed beside the reward click rather than through
the feedback mechanisms.

Everything that costs time is off the screen's loop. The detector runs on the feed thread, and the screen
writes a setting that the feed picks up on its next block. A search's estimate
costs up to 60 ms, so it runs on a thread of its own, at most once every `ESTIMATE_SECONDS`.
"""

from __future__ import annotations

import copy
import threading
from collections import deque
from typing import Callable

import flet as ft

from nfb.dsp import Found, SpindleDetector, SpindleSearch
from nfb.dsp.spindle import SEARCH_BAND, SEARCH_SECONDS, SPINDLE_HZ, SPINDLE_LEVEL

from . import cues
from . import theme as t
from .target_card import NOTE_H
from .widgets import help_button, labelled, num_box, pill, settings_button, test_button, tiny

ESTIMATE_SECONDS = 0.5
RATE_SECONDS = 60.0                 # the window the rate a minute is counted over
PITCH_HZ = 1320.0
"""The spindle's sound: a short tone, where the reward click is a burst of noise, so the two do not mix."""
LEVELS = (2.0, 10.0)
LEVEL_STEP = 0.5
"""On a recording, 3 found every spindle and 5 to 6 a minute in a signal with none; 5 found about half."""

HELP = (
    "A spindle is a short burst of one rhythm on one channel. The burst is much larger than the usual "
    "amplitude of that rhythm.\n\n"
    "**threshold** sets how large a burst must be to count as a spindle. The threshold is a multiple of "
    "the usual amplitude. The usual amplitude is the median of the last minute, so the threshold follows "
    "the signal. The line below shows the threshold in µV now.\n\n"
    "A lower threshold counts smaller spindles. It also counts more bursts that are not spindles. A change "
    "keeps the count.\n\n"
    "The card counts the spindles near the frequency in its box, not only at that frequency. The range "
    "is approximately one fifth of the frequency to each side. At 25 Hz, the range is 20 to 30 Hz. A "
    "spindle outside the range counts only when it is very large.\n\n"
    "Each spindle has its own sound. The sound plays when the detector confirms the spindle, about 0.2 s "
    "after the spindle starts. Spindles do not change the targets, their thresholds or the feedback.\n\n"
    "**search** finds the frequency of your spindles. While the search runs, the card keeps the signal "
    "of this channel. The line beside the button shows the time, the frequency so far and the number of "
    "spindles that the frequency comes from. The frequency changes as the search finds more spindles. "
    f"The search uses the last {SEARCH_SECONDS / 60:g} minutes of the signal.\n\n"
    "Press the button again to stop the search. The card then uses the frequency on that line.\n\n"
    "The count starts when the session runs. A paused session, a block with an artifact and a channel "
    "without signal do not count.\n\n"
    "Muscle activity from the jaw can look like many spindles. Behind the gear icon, set **muscle** to on. "
    "Then a spindle does not count while the muscle signal on this channel is high."
)


class Spindles:
    """What the card shows and the feed writes. All counting happens on the feed thread."""

    def __init__(self, rate_hz: float, channel: int, frequency_hz: float = SPINDLE_HZ,
                 level: float = SPINDLE_LEVEL, reject_muscle: bool = False, sound: bool = True):
        self.rate_hz = rate_hz
        self.channel = channel
        self.frequency_hz = frequency_hz
        self.level = level
        self.reject_muscle = reject_muscle
        self.sound = sound
        self.count = 0
        self.held_back = 0
        self.seconds = 0.0                  # how much signal has counted
        self._times: deque[float] = deque()
        self._detector: SpindleDetector | None = None
        self._built: tuple | None = None
        self._lock = threading.Lock()
        self._search: SpindleSearch | None = None
        self.search_seconds = 0.0
        self.found: Found | None = None
        """The search's latest estimate while it runs; after it stops, the last one it had."""
        self._estimated_at: float | None = None
        self._worker: threading.Thread | None = None

    def settings(self) -> dict:
        """What a stop carries to the next session. The counts and a search are the session's."""
        return {"channel": self.channel, "frequency_hz": self.frequency_hz, "level": self.level,
                "reject_muscle": self.reject_muscle, "sound": self.sound}

    @property
    def usual(self) -> float | None:
        """The usual amplitude in µV, None before the first sample."""
        detector = self._detector
        return None if detector is None else detector.usual

    @property
    def level_uv(self) -> float | None:
        """The threshold in µV now, at the level the screen shows even before the feed applies it."""
        usual = self.usual
        return None if usual is None else self.level * usual

    @property
    def per_minute(self) -> float | None:
        if self.seconds <= 0:
            return None
        return len(self._times) * 60.0 / min(self.seconds, RATE_SECONDS)

    def feed(self, values, counting: bool) -> int:
        """One block of this channel, in µV. Returns how many spindles it confirmed that count."""
        built = (self.channel, self.frequency_hz, self.reject_muscle)
        if self._built is None or built != self._built:
            if self._built is not None and built[0] != self._built[0]:
                # Another electrode: nothing counted so far is about it.
                self.count, self.held_back, self.seconds = 0, 0, 0.0
                self._times.clear()
            self._detector = SpindleDetector(self.rate_hz, self.frequency_hz, self.level,
                                             reject_muscle=self.reject_muscle)
            self._built = built
        elif self._detector.level != self.level:
            # The same detector, so the usual amplitude and the count go on.
            self._detector.set_level(self.level)
        before = self._detector.rejected
        confirmed = self._detector.process(values)
        search = self._search
        if search is not None:
            with self._lock:
                search.push(values)
            self.search_seconds += len(values) / self.rate_hz
        if not counting:
            return 0
        self.seconds += len(values) / self.rate_hz
        self.held_back += self._detector.rejected - before
        self.count += confirmed
        self._times.extend([self.seconds] * confirmed)
        while self._times and self._times[0] < self.seconds - RATE_SECONDS:
            self._times.popleft()
        return confirmed

    # --- the search ------------------------------------------------------------------------

    @property
    def searching(self) -> bool:
        return self._search is not None

    def start_search(self) -> None:
        self.found = Found(self.frequency_hz, 0, 0.0)
        self.search_seconds = 0.0
        self._estimated_at = None
        self._search = SpindleSearch(self.rate_hz, start_hz=self.frequency_hz)

    def stop_search(self) -> None:
        """Stop, and use the frequency the card shows now.

        The estimate on screen and not a new one: what the operator saw when they pressed is what the card
        takes, and a fresh fit would cost the screen's loop up to 60 ms."""
        self._search = None
        if self.found is not None:
            self.frequency_hz = round(self.found.frequency_hz, 1)

    def estimate(self) -> Found | None:
        """Fit what the search has so far, and keep it as `found` unless the search ended meanwhile."""
        search = self._search
        if search is None:
            return None
        # `push` replaces the kept array rather than writing into it, so a shallow copy is a still picture.
        with self._lock:
            still = copy.copy(search)
        try:
            found = still.estimate()
        except ValueError:
            return None
        if self._search is search:
            self.found = found
        return found

    def estimate_due(self, now: float) -> None:
        """Start an estimate on a thread of its own when one is due. `now` is the monotonic clock."""
        if self._search is None:
            return
        if self._worker is not None and self._worker.is_alive():
            return
        if self._estimated_at is not None and now - self._estimated_at < ESTIMATE_SECONDS:
            return
        self._estimated_at = now
        self._worker = threading.Thread(target=self.estimate, daemon=True, name="spindle-search")
        self._worker.start()


def sound() -> str:
    return cues.tone_at(PITCH_HZ)


def _duration(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60} min {whole % 60:02d} s" if whole >= 60 else f"{whole} s"


def _spindles(n: int) -> str:
    return f"{n} spindle{'' if n == 1 else 's'}"


def search_line(watch: Spindles) -> str:
    """What the line beside the search button says: the running result, or what the last search did."""
    found = watch.found
    if watch.searching:
        spindles = _spindles(found.spindles)
        if watch.search_seconds > SEARCH_SECONDS:
            spindles += f" in the last {SEARCH_SECONDS / 60:g} min"
        return (f"searching {_duration(watch.search_seconds)} · "
                f"{found.frequency_hz:.1f} Hz from {spindles}")
    if found is None:
        return ""
    return f"found {found.frequency_hz:.1f} Hz from {_spindles(found.spindles)}"


def level_line(watch: Spindles) -> str:
    """The threshold in µV now. One line from the start, so the card does not change height."""
    uv = watch.level_uv
    if uv is None:
        return f"threshold now {watch.level:.1f} × the usual amplitude, which is not known yet"
    return f"threshold now {uv:.1f} µV: {watch.level:.1f} × the usual {watch.usual:.1f} µV"


class SpindleCard:
    """Build once, `refresh` per frame. `on_change` rebuilds the card list; `on_remove` drops the card."""

    def __init__(self, page, watch: Spindles, channels, on_change: Callable[[], None],
                 on_remove: Callable[[], None], volume: Callable[[], float]):
        self.page = page
        self.watch = watch
        self.on_change = on_change
        self.volume = volume

        source = ft.PopupMenuButton(
            content=ft.Container(
                content=ft.Text(f"ch{watch.channel}", size=t.BODY, color=t.TEXT),
                padding=ft.Padding(8, 4, 8, 4), border_radius=6,
                bgcolor=t.SURFACE_HI, border=ft.Border.all(1, t.HAIRLINE)),
            items=[ft.PopupMenuItem(content=ft.Text(f"ch{c}", size=t.BODY),
                                    on_click=(lambda c_=c: lambda _e: self._set_channel(c_))())
                   for c in channels])
        identity = ft.Row(
            [ft.Container(width=3, height=18, bgcolor=t.frequency_colour(watch.frequency_hz),
                          border_radius=2),
             source,
             num_box(watch.frequency_hz, self._set_frequency, suffix="Hz", step=0.1,
                     low=SEARCH_BAND[0], high=SEARCH_BAND[1]),
             ft.Container(expand=True),
             ft.Container(content=ft.Text("SPINDLE", size=t.SMALL, color=t.MUTED,
                                          weight=ft.FontWeight.W_600),
                          padding=ft.Padding(8, 4, 8, 4), border_radius=6, bgcolor=t.SURFACE_HI),
             ft.Container(width=8),
             tiny("×", lambda _e: on_remove())],
            spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        self.value_text = ft.Text("—", size=t.VALUE, color=t.TEXT, weight=ft.FontWeight.W_500)
        self.count_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.thresh_text = ft.Text("", size=t.BODY, color=t.TEXT)
        # Its own line and left-aligned, so a digit more in the µV moves nothing else on the card.
        self.level_text = ft.Text("", size=t.SMALL, color=t.FAINT, no_wrap=True)
        self.button_text = ft.Text("", size=t.SMALL, color=t.ACCENT)
        self.search_text = ft.Text("", size=t.SMALL, color=t.MUTED, no_wrap=True)
        self.muscle_text = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.note = ft.Text("", size=t.SMALL, color=t.WARN, no_wrap=True)
        search_button = ft.Container(content=self.button_text, padding=ft.Padding(9, 5, 9, 5),
                                     border_radius=6, bgcolor=t.SURFACE_HI,
                                     border=ft.Border.all(1, t.HAIRLINE), on_click=self._toggle_search)
        rows = [
            identity,
            ft.Container(height=8),
            ft.Row([self.value_text, ft.Container(width=6), self.count_text,
                    ft.Container(expand=True),
                    ft.Text("threshold", size=t.SMALL, color=t.FAINT),
                    self.thresh_text,
                    tiny("−", lambda _e: self._nudge(-LEVEL_STEP)),
                    tiny("+", lambda _e: self._nudge(LEVEL_STEP))],
                   spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Container(height=4),
            ft.Row([ft.Container(content=self.level_text, expand=True)]),
            ft.Container(height=9),
            ft.Row([search_button, ft.Container(content=self.search_text, expand=True)],
                   spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Container(height=7),
            ft.Row([ft.Container(content=self.muscle_text, expand=True),
                    settings_button(page, "Spindle", self._settings_rows), ft.Container(width=6),
                    help_button(page, "Spindle", HELP)],
                   spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Container(content=self.note, height=NOTE_H, alignment=ft.Alignment.BOTTOM_LEFT),
        ]
        self.control = ft.Container(
            content=ft.Column(rows, spacing=0),
            bgcolor=t.BG, border=ft.Border.all(1, t.HAIRLINE), border_radius=8,
            padding=ft.Padding(12, 11, 12, 11))

    def _set_channel(self, channel: int) -> None:
        self.watch.channel = channel
        if self.watch.searching:
            self.watch.start_search()       # the signal kept so far is another electrode's
        self.on_change()

    def _set_frequency(self, hz: float) -> None:
        self.watch.frequency_hz = round(hz, 1)
        self.on_change()

    def _nudge(self, step: float) -> None:
        # The feed applies it on its next block.
        self.watch.level = min(LEVELS[1], max(LEVELS[0], round(self.watch.level + step, 1)))
        self.on_change()

    def _toggle_search(self, _e=None) -> None:
        if self.watch.searching:
            self.watch.stop_search()
            self.on_change()                # the frequency box shows what the search found
        else:
            self.watch.start_search()
            self.refresh()

    def _settings_rows(self, redraw: Callable[[], None]) -> list:
        watch = self.watch

        def set_muscle(on: bool) -> None:
            watch.reject_muscle = on
            redraw()
            self.refresh()

        def set_sound(on: bool) -> None:
            watch.sound = on
            redraw()

        return [
            labelled("muscle", pill("off", not watch.reject_muscle, lambda _e: set_muscle(False)),
                     pill("on", watch.reject_muscle, lambda _e: set_muscle(True)),
                     note="on: a spindle does not count while the muscle signal is high", width=64),
            labelled("sound", pill("off", not watch.sound, lambda _e: set_sound(False)),
                     pill("on", watch.sound, lambda _e: set_sound(True)),
                     test_button("play", lambda _e: cues.play(sound(), self.volume())),
                     note="at the click volume on the Feedback tab", width=64),
        ]

    def refresh(self, live: bool | None = None) -> None:
        """`live` is the channel's state: False when it has no signal, so nothing counts."""
        watch = self.watch
        rate = watch.per_minute
        self.value_text.value = "—" if rate is None else f"{rate:.1f}"
        self.count_text.value = f"a minute · {watch.count} this session"
        self.thresh_text.value = f"{watch.level:.1f} ×"
        self.level_text.value = level_line(watch)
        self.button_text.value = "stop search" if watch.searching else "search"
        self.search_text.value = search_line(watch)
        self.muscle_text.value = (f"muscle held back {watch.held_back}" if watch.reject_muscle
                                  else "")
        self.note.value = f"ch{watch.channel} has no signal" if live is False else ""
