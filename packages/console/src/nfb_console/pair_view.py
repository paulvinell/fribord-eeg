"""The Mendi's pulse against the HeartMath's, live: a program that compares two heart sensors.

It borrows the two runners the app already owns, the headband's and the heart-rate sensor's, and never
starts or stops either, so the HEG and HRV programs can use them too. Both hand out their beats and their
pulse in one form (`runner.PulseRunner`), so nothing here knows which device is which beyond its name.
Every couple of seconds it compares the last minute of the two pulses with `nfb.hrv.compare`. It trains
nothing: the height ratio and the delay are not validated as things to train, so they are shown, not fed
back.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from nfb.hrv.compare import compare, heights

from . import cues, prefs
from .plan import SessionPlan
from .protocols import Protocol
from .runner import PulseRunner

HANDLES = ("PAIR",)
EEG_SOURCE = False

WINDOW_S = 60.0
"""How much of each pulse is compared at a time: long enough for the pairing to find the usual delay and
for the averaged pulse to stand clear of the noise, short enough to follow a change within a minute."""

UPDATE_S = 2.0
"""How often the comparison is worked out again."""

SERIES_KEPT = 1800
"""Points kept for the session trends: an hour at one point every two seconds."""


@dataclass(frozen=True)
class Snapshot:
    """One comparison of the last minute, handed whole from the session's thread to the view."""

    forehead_rate: float | None             # bpm, from each runner's heart-rate state
    ear_rate: float | None
    forehead_rmssd: float | None            # ms
    ear_rmssd: float | None
    paired: int                             # heartbeats seen at both sites, in the window
    forehead_only: int
    ear_only: int
    agreement: float | None                 # correlation of the two devices' paired intervals
    intervals: list[tuple[float, float, float]] = field(default_factory=list)   # (time, forehead, ear) s
    delay_ms: float | None = None           # forehead to ear, median over the window
    delay_spread_ms: float | None = None
    height_ratio: float | None = None       # ear pulse height over forehead pulse height, 1 = the start
    pulse_lags: list[float] = field(default_factory=list)       # s from the ear's beat
    pulse_shape: list[float] = field(default_factory=list)      # the forehead pulse averaged around them


class PairSession:
    """The two borrowed runners, the comparison and the session clock. No outputs: nothing is trained."""

    def __init__(self, protocol: Protocol, length_seconds: float | None = None,
                 heart_runner: PulseRunner | None = None, mendi_runner: PulseRunner | None = None,
                 heart_choice=None, mendi_choice=None):
        self.protocol = protocol
        self.plan = SessionPlan(length_seconds)
        self.cue = cues.play
        self.heart = heart_runner
        self.mendi = mendi_runner
        self.heart_choice = heart_choice
        self.mendi_choice = mendi_choice
        self.snapshot: Snapshot | None = None
        self.delay_series: deque[tuple[float, float]] = deque(maxlen=SERIES_KEPT)
        self.ratio_series: deque[tuple[float, float]] = deque(maxlen=SERIES_KEPT)
        self._ratio_start: float | None = None
        self.finished = False
        self.paused = True
        self._stop = False
        self._accumulated = 0.0
        self._resumed_at: float | None = None
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self.redraw: Callable[[], None] = lambda: None

    # --- what is missing ----------------------------------------------------------------------

    def missing(self) -> list[str]:
        """Why a device gives nothing to compare, one line each, or nothing when both are reading."""
        out = []
        for name, runner, choice, where in (
                ("headband", self.mendi, self.mendi_choice, "choose the Mendi headband on the device screen"),
                ("heart sensor", self.heart, self.heart_choice, "choose the HeartMath on the device screen")):
            if runner is None or choice is None or not choice.on:
                out.append(f"no {name}: {where}")
            elif runner.error:
                out.append(f"{name}: {runner.error}")
            elif runner.stale:
                out.append(f"{name}: no readings for a while")
            elif runner.heart_state is None:
                out.append(f"{name}: {runner.describe()}")
        return out

    # --- the comparison -----------------------------------------------------------------------

    def update(self, now: float | None = None) -> None:
        """Compare the last `WINDOW_S` of both pulses and keep the result. `now` is in the beats' own
        seconds; a test passes it, the session's thread uses the wall clock the runners stamp with."""
        if self.missing():
            self.snapshot = None
            return
        now = time.time() if now is None else now
        since = now - WINDOW_S
        forehead = [t for t in _copy(self.mendi.beat_times) if since <= t <= now]
        ear = [t for t in _copy(self.heart.beat_times) if since <= t <= now]
        f_wave = [p for p in (_copy(self.mendi.waveform) or []) if since <= p[0] <= now] or None
        e_wave = [p for p in (_copy(self.heart.waveform) or []) if since <= p[0] <= now] or None
        result = compare(forehead, ear, f_wave, e_wave)
        a = result.agreement
        f_int, e_int, intervals = [], [], []
        for (f1, e1), (f2, e2) in zip(a.pairs, a.pairs[1:]):
            if 0.3 < f2 - f1 < 2.0 and 0.3 < e2 - e1 < 2.0:
                f_int.append(f2 - f1)
                e_int.append(e2 - e1)
                intervals.append((f2, f2 - f1, e2 - e1))
        ratio = self._height_ratio(f_wave, e_wave, a.pairs)
        at = self.elapsed
        if result.delay is not None:
            self.delay_series.append((at, result.delay.median_s * 1000.0))
        if ratio is not None:
            self.ratio_series.append((at, ratio))
        f_state, e_state = self.mendi.heart_state, self.heart.heart_state
        self.snapshot = Snapshot(
            forehead_rate=f_state.hr_smoothed if f_state else None,
            ear_rate=e_state.hr_smoothed if e_state else None,
            forehead_rmssd=f_state.rmssd_ms if f_state else None,
            ear_rmssd=e_state.rmssd_ms if e_state else None,
            paired=len(a.pairs), forehead_only=len(a.forehead_only), ear_only=len(a.reference_only),
            agreement=_correlation(f_int, e_int), intervals=intervals,
            delay_ms=result.delay.median_s * 1000.0 if result.delay else None,
            delay_spread_ms=result.delay.spread_s * 1000.0 if result.delay else None,
            height_ratio=ratio,
            pulse_lags=list(result.rescue.lags_s) if result.rescue else [],
            pulse_shape=list(result.rescue.shape) if result.rescue else [])

    def _height_ratio(self, f_wave, e_wave, pairs) -> float | None:
        """The ear pulse's height over the forehead's, median over the window, divided by the first such
        median of the session, so 1 means as at the start."""
        if not (f_wave and e_wave) or len(pairs) < 6:
            return None
        f_h = heights(f_wave, [f for f, _ in pairs])
        e_h = heights(e_wave, [e for _, e in pairs])
        ratios = [eh / fh for fh, eh in zip(f_h, e_h) if fh and eh]
        if len(ratios) < 6:
            return None
        raw = statistics.median(ratios)
        if self._ratio_start is None:
            self._ratio_start = raw
        return raw / self._ratio_start

    # --- the thread ---------------------------------------------------------------------------

    def run(self) -> None:
        """Start the comparison's own thread if it is not already running; the app calls this again on
        resume. The runners read their devices on their own threads, so nothing else needs starting."""
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, daemon=True, name="pair-compare")
            self._worker.start()

    def _loop(self) -> None:
        while not self._stop:
            if self.running:
                self.update()
            self._wake.wait(UPDATE_S)

    # --- transport, the contract the toolbar expects -----------------------------------------

    @property
    def elapsed(self) -> float:
        running = (time.monotonic() - self._resumed_at) if self._resumed_at is not None else 0.0
        return self._accumulated + running

    @property
    def running(self) -> bool:
        return not self.paused

    @property
    def length(self) -> float | None:
        return self.plan.length_seconds

    def toggle_pause(self) -> None:
        if self.paused:
            self.paused = False
            self._resumed_at = time.monotonic()
        else:
            self._accumulated = self.elapsed
            self._resumed_at = None
            self.paused = True

    def tick(self) -> None:
        if not self.running:
            return
        elapsed = self.elapsed
        for cue in self.plan.due(elapsed):
            self.cue(cue)
        if self.plan.complete(elapsed):
            self.finished = True
            self._accumulated = self.plan.length_seconds
            self._resumed_at = None
            self.paused = True

    def stop(self) -> None:
        """End the session. The runners are borrowed and keep running for whatever program is next."""
        self._stop = True
        self.paused = True
        self._wake.set()
        prefs.remember(self.protocol.code, self)

    # --- going to the device screen and coming back ------------------------------------------

    def suspend(self) -> None:
        if not self.paused:
            self.toggle_pause()

    @property
    def suspended(self) -> bool:
        return not self._stop

    def rearm(self) -> None:
        """Nothing to rebuild: the runners never stopped and the clock stayed where `suspend` left it."""

    # --- surviving a stop -------------------------------------------------------------------

    def adopt(self, previous: "PairSession") -> None:
        if previous is self or previous.protocol.code != self.protocol.code:
            return
        self.plan = previous.plan
        self.plan.rewind()

    def calibration(self) -> dict:
        return {}

    def restore_calibration(self, values: dict) -> int:
        return 0


def _copy(items):
    """A copy of a deque another thread appends to, or a list it rebuilds. Copying can meet the other
    thread's append; one retry is enough at the rates these grow."""
    if items is None:
        return None
    for _ in range(3):
        try:
            return list(items)
        except RuntimeError:
            continue
    return []


def _correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 5:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def session_for(protocol: Protocol, path=None, device=None, length_seconds=None, source=None,
                channels: int = 0, heart_choice=None, heart_runner=None, mendi_choice=None,
                mendi_runner=None, pupil_choice=None, pupil_runner=None) -> PairSession:
    """The other programs' signature, so `app.pick` need not special-case it."""
    _ = path, device, source, channels, pupil_choice, pupil_runner
    return PairSession(protocol, length_seconds=length_seconds, heart_runner=heart_runner,
                       mendi_runner=mendi_runner, heart_choice=heart_choice, mendi_choice=mendi_choice)
