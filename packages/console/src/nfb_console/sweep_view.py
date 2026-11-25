"""The sweep in the BREATHE panel: what it waits for, the curve while it runs, and the result.

**The sweep is a choice of the pacer, so it has no panel and no button of its own.** The pacer note says what
the sweep does and where it is; this view holds what the note cannot: the lines that need a person to act
(what the sweep waits for, simulated beats, a session maximum that ends it first), the curve, and after the
sweep the result with its two buttons. Each button acts. No text names an action that has no control: to run
another sweep the person chooses `sweep` again, which is always there.

**Rows are rebuilt on a change of state; figures are written in the draw.** The state is the sweep (armed,
running, ended), whether the beats are simulated, the cause that holds it and, for a result, whether the
target is at the rate. The last breath, the best rate and the sentence about the target are Text controls
that `refresh` writes each draw. The curve's shapes are rebuilt only when a breath is measured, the sweep
ends or the target moves, so a draw with nothing new redraws the same shapes.
"""

from __future__ import annotations

from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np

from nfb.hrv.sweep import CLEAR_SHARE, MIN_FOR_VERDICT, WINDOW, best, curve, in_step, swing_at

from . import theme as t
from .plots import (gridlines, grid, nice_ceiling, place_tick_labels, place_value_labels, polyline,
                    tick_axis, value_axis)
from .widgets import button, seconds_text

CURVE_W = 250
CURVE_H = 110
AXIS_W = 40
TICK_W = 30
MARGIN = 5                  # inside the plot, so a dot at an end of the range is whole
AT_RATE_BPM = 0.05
"""The target is at the rate when the two differ by less than this: the target note shows one decimal of a
breath a minute, and a press that moves nothing that is shown would be a dead control."""

HELD = {
    "paused": "The sweep waits. The session is paused. Press the play button to go on.",
    "error": "The sweep waits. The heart source reports a fault. Check the source on the device screen.",
    "silence": "The sweep waits. No beats arrive. Check the contact of your heart source.",
    "source": "The sweep waits. No heart source is on. Choose one on the device screen.",
    "finished": "The sweep waits. The session maximum has passed. Change the maximum on the Session tab.",
}
"""The causes that last until a person acts, before breath 1 and during the sweep. `stall` and `clock` hold
for one draw, and the line about the last breath names them."""
NOT_MEASURED = {
    "few": "It has too few beats.",
    "gap": "It has a gap in the beats.",
    "paused": "The session pauses during it.",
    "error": "The heart source reports a fault during it.",
    "silence": "No beats arrive during it.",
    "stall": "The screen stops during it.",
    "clock": "The computer clock changes during it.",
}
SIMULATED = "These beats are simulated. The result cannot move the target."

SWEEP_HELP = (
    "**What the sweep does.** Choose **sweep** to find your resonant rate. When the session runs and beats "
    "arrive, the pacer changes its rate at every breath. Your heart follows the pacer, and the sweep finds "
    "the rate at which it follows most. You breathe in and out at each tone, as you do with the metronome. "
    "The sound and the inhale setting work as before. A sweep changes no setting. Choose another pacer to "
    "stop it, and the breaths that are measured stay as a result.\n\n"
    "**Where the sweep starts and ends.** The range row sets the two rates. A start above the end decreases "
    "the rate of the pacer, and a start below the end increases it. A wider range has more breaths and takes "
    "longer. Move the end while the sweep runs, and the measured breaths stay. Move the start, and the sweep "
    "begins again.\n\n"
    "**What the paced swing is.** For each breath, the screen fits the part of the beat interval that "
    "follows the pacer. The paced swing is the range of that part, in ms. The RSA swing metric is a "
    "different figure: the heart rate range over 20 s, in bpm.\n\n"
    f"**What the curve shows.** Each dot is one breath. The line is the median of {WINDOW} breaths. "
    "The blue dot is the best rate so far. The grey line is the rate at which the heart rate and the "
    "breath are in step. The green line is the target. The sweep shows both rates from the first "
    "measured breaths, and they can move as more breaths arrive.\n\n"
    "**What in step means.** The heart rate is late above the rate at which it is in step with the breath, "
    "and early below it. The sweep finds the rate where the late values change to early values. It can "
    "differ from the rate of the largest paced swing, and the result gives both.\n\n"
    f"**What the verdict says.** The screen compares the peak with random swings. In them, neighbouring "
    f"breaths are as alike as in your sweep, so a slow drift of the swing is not a peak. The peak is clear "
    f"when at most {CLEAR_SHARE:.0%} of the random swings give a rise as large. A verdict needs "
    f"{MIN_FOR_VERDICT} measured breaths. A peak that is not clear can be chance, so choose the sweep "
    "again. A second sweep in the other direction checks a peak that comes from the order of the breaths. "
    "A true peak stays at the same rate. The result also gives the mean heart rate of the first breaths "
    "and of the last. A fall of the heart rate can tilt the curve. A peak at the end of the swept range, or beside breaths that are not measured, can have its "
    "true place beyond it.\n\n"
    "**What happens when something fails.** The sweep waits when the session pauses, when the heart "
    "source reports a fault and when no beats arrive. It then starts the breath again. A breath with no "
    "result shows as not measured, with the reason.\n\n"
    "**What use this rate does.** It moves the target and the band to the best rate, and the metronome "
    "cycle with them. The button is there for a result from real beats that come from the same source. "
    "Simulated beats do not follow the pacer, so a result from them cannot move the target."
)


class SweepView:
    """Build once; `refresh()` per draw. `on_target` runs after a button moved the target or dropped the
    result, so the rows of BREATHE are built again."""

    def __init__(self, page: ft.Page, session, on_target: Callable[[], None] = lambda: None):
        self.page = page
        self.session = session
        self.on_target = on_target
        schedule = session.sweep_schedule
        self._lo, self._hi = schedule.low_per_min, schedule.high_per_min
        self.body = ft.Column(spacing=8)
        self._key: tuple | None = None
        self._curve_key: tuple | None = None
        self._points: list = []
        self._best = None

        self.canvas = cv.Canvas(shapes=[], width=CURVE_W, height=CURVE_H)
        self._axis, self._labels = value_axis(CURVE_H, count=3, width=AXIS_W)
        self._scale = ft.Container(padding=ft.Padding(AXIS_W, 2, 0, 0))
        self._build_scale()
        self._curve_rows = [ft.Row([self._axis, self.canvas], spacing=0), self._scale]
        self.last = ft.Text("", size=t.SMALL, color=t.MUTED)
        self.best_line = ft.Text("", size=t.SMALL, color=t.MUTED)
        self.step_line = ft.Text("", size=t.SMALL, color=t.MUTED, visible=False)
        self._step = None
        self.axis_note = ft.Text("", size=t.SMALL, color=t.FAINT)
        self.target_line = ft.Text("", size=t.SMALL, color=t.MUTED)

    @property
    def present(self) -> bool:
        """Whether there is anything to show: a sweep that is armed, runs or left a result."""
        return self.session.sweep is not None or self.session.pacer_choice == "sweep"

    # --- the geometry of the curve -----------------------------------------------------------

    def _tick_values(self) -> list[float]:
        """Whole and half rates, with as wide a step as keeps two labels apart, and none so near an end that
        its label would be pushed against the next one."""
        step = 0.5
        while (self._hi - self._lo) / step * (TICK_W + 4) > CURVE_W:
            step *= 2.0
        ticks = (float(v) for v in np.arange(np.ceil(self._lo / step) * step, self._hi + 1e-9, step))
        return [v for v in ticks if TICK_W / 2 <= self._x_of(v) <= CURVE_W - TICK_W / 2]

    def _build_scale(self) -> None:
        scale, ticks = tick_axis(self._tick_values(), tick_width=TICK_W)
        place_tick_labels(ticks, self._tick_values(), self._x_of, tick_width=TICK_W, width=CURVE_W)
        self._scale.content = scale

    def _use_range(self, schedule) -> None:
        """Draw the curve over the range of the sweep that it shows, which moves when its end does."""
        if (schedule.low_per_min, schedule.high_per_min) != (self._lo, self._hi):
            self._lo, self._hi = schedule.low_per_min, schedule.high_per_min
            self._build_scale()

    def _x_of(self, rate: float) -> float | None:
        if not self._lo <= rate <= self._hi:
            return None
        return MARGIN + (rate - self._lo) / (self._hi - self._lo) * (CURVE_W - 2 * MARGIN)

    def _y_of(self, swing_ms: float, top: float) -> float:
        return CURVE_H - MARGIN - min(1.0, swing_ms / top) * (CURVE_H - 2 * MARGIN)

    # --- what state the rows are for ----------------------------------------------------------

    def _minutes_left(self) -> int | None:
        left = self.session.sweep_time_left_s()
        return None if left is None else max(1, round(left / 60.0))

    def _at_rate(self, usable: float | None) -> bool:
        return usable is not None and abs(self.session.target_breaths_per_min - usable) < AT_RATE_BPM

    def _held(self) -> str | None:
        held = self.session.sweep_hold()
        return held if held in HELD else None

    @staticmethod
    def _measured(sweep) -> bool:
        return any(v is not None for v in sweep.values)

    def _state_key(self) -> tuple:
        session, sweep = self.session, self.session.sweep
        if sweep is None:
            return ("armed", self._held(), session.simulated_beats, self._minutes_left())
        if session.sweeping:
            return ("running", id(sweep), session.simulated_beats, self._held(), self._measured(sweep))
        usable = session.usable_rate()
        return ("result", id(sweep), session.simulated_beats, usable is not None, self._at_rate(usable),
                session.pacer_choice)

    def refresh(self) -> None:
        if not self.present:
            self._key = None
            return
        key = self._state_key()
        if key != self._key:
            self.rebuild()
        session, sweep = self.session, self.session.sweep
        if sweep is None:
            return
        self._use_range(sweep.schedule)
        if session.sweeping:
            if self._measured(sweep):
                self._draw_curve(sweep)
            self._write_running(sweep)
        elif sweep.result.best is not None:
            self._draw_curve(sweep)
            self._write_target()

    # --- the rows -----------------------------------------------------------------------------

    def rebuild(self) -> None:
        session, sweep = self.session, self.session.sweep
        self._key = self._state_key()
        rows = self._alerts()
        if sweep is not None:
            rows += self._running(sweep) if session.sweeping else self._result(sweep)
        self.body.controls = rows
        self.page.update()

    def _line(self, text: str, colour: str = t.MUTED) -> ft.Text:
        return ft.Text(text, size=t.SMALL, color=colour)

    def _alerts(self) -> list[ft.Control]:
        """What needs a person: the beats are simulated, the sweep waits for something, or the session
        maximum ends it first."""
        session = self.session
        rows: list[ft.Control] = [self._line(SIMULATED, t.WARN)] if session.simulated_beats else []
        held = self._held()
        if held is not None:
            rows.append(self._line(HELD[held], t.WARN))
        left = self._minutes_left()
        if session.sweep is None and left is not None:
            rows.append(self._line(f"The session maximum ends the sweep after about {left} "
                                   f"minute{'s' if left != 1 else ''}. Change the maximum on the Session tab "
                                   "to run a whole sweep.", t.WARN))
        return rows

    def _running(self, sweep) -> list[ft.Control]:
        if not self._measured(sweep):
            return [self.last]
        return [self.last, *self._curve_rows, self.axis_note, self.best_line, self.step_line]

    def _result(self, sweep) -> list[ft.Control]:
        session, result = self.session, sweep.result
        usable = session.usable_rate()
        rows: list[ft.Control] = []
        end = self._ended_text(sweep)
        if end is not None:
            rows.append(self._line(end, t.TEXT))
        rows.append(self._line(f"Measured: {result.measured} of {sweep.schedule.breaths} breaths."))
        share = session.sweep_inhale_pct
        if share is None:
            rows.append(self._line("The inhale setting changes during the sweep."))
        elif share != 50.0:
            rows.append(self._line(f"The sweep runs with an inhale of {share:g}% of the breath."))
        if result.best is not None:
            rows += [*self._curve_rows, self.axis_note]
        rows += [self._line(line, t.TEXT) for line in self._verdict_lines(result)]
        if result.heart_bpm is not None:
            first, last = result.heart_bpm
            rows.append(self._line(f"Mean heart rate: {first:.0f} bpm in the first breaths, {last:.0f} bpm "
                                   "in the last."))
        if result.best is not None:
            rows.append(self.target_line)
        if usable is not None and not self._at_rate(usable):
            rows.append(self._line(self._action_text(usable), t.TEXT))
        if result.verdict.kind != "clear":
            rows.append(self._line("Choose sweep to run the sweep again. A new sweep replaces this result."))
        buttons = []
        if usable is not None and not self._at_rate(usable):
            buttons.append(button("use this rate", lambda _e: self._use()))
        buttons.append(button("close", lambda _e: self._close(), t.MUTED))
        rows.append(ft.Row(buttons, spacing=8, wrap=True, run_spacing=8))
        return rows

    def _use(self) -> None:
        self.session.use_rate()
        self.on_target()
        self.refresh()

    def _close(self) -> None:
        self.session.clear_sweep()
        self.on_target()
        self.refresh()

    # --- the sentences ------------------------------------------------------------------------

    def _ended_text(self, sweep) -> str | None:
        """How an early end went. A sweep that ran its course needs no sentence: the curve is whole."""
        where = f"at breath {self.session.sweep_breath} of {sweep.schedule.breaths}"
        return {
            "done": None,
            "stop": f"The sweep stops {where}.",
            "plan": f"The session maximum ends the sweep {where}. Change the maximum on the Session tab "
                    "to run a whole sweep.",
            "source": f"The heart source changes. The sweep ends {where}. This result cannot move the target.",
        }[sweep.reason]

    def _verdict_lines(self, result) -> list[str]:
        verdict, top = result.verdict, result.best
        if top is None:
            return ["No breath is measured."]
        lines = [f"The paced swing is largest at {top.rate:.1f} breaths a minute: {top.swing_ms:.0f} ms.",
                 self._step_text(result.in_step)]
        if verdict.kind == "few":
            return lines + ["Too few breaths are measured to judge the peak."]
        tries = (f"Random swings as smooth as yours give a rise this large in {verdict.reached} of "
                 f"{verdict.tries} tries.")
        if verdict.kind == "clear":
            lines.append(f"The peak is clear. It is {verdict.rise_ms:.0f} ms above the other breaths. {tries}")
            if verdict.edge:
                lines.append("The peak is at the end of the swept range, or beside breaths that are not measured. "
                             "The true peak can lie beyond it.")
        else:
            lines.append(f"The peak is not clear. It is {verdict.rise_ms:.0f} ms above the other breaths. {tries}")
        return lines

    @staticmethod
    def _step_text(step) -> str:
        """The second mark, in its own sentence: the heart rate peaks with the breath at this rate. Where no
        rate is found, the side says where it must lie."""
        if step.rate is not None:
            return f"The heart rate and the breath are in step at {step.rate:.1f} breaths a minute."
        text = "The heart rate and the breath are in step at no measured rate."
        if step.side == "late":
            return text + " The heart rate is late at all of them, so the rate is slower than the range."
        if step.side == "early":
            return text + " The heart rate is early at all of them, so the rate is faster than the range."
        return text

    def _action_text(self, rate: float) -> str:
        text = f"Press use this rate to move the target and the band to {rate:.1f} breaths a minute."
        if self.session.pacer_choice == "metronome":
            text += f" The cycle becomes {seconds_text(self.session.cycle_for_rate(rate))} s."
        return text

    def _write_running(self, sweep) -> None:
        if not sweep.outcomes:
            self.last.value = "No breath is measured yet."
        else:
            done = sweep.outcomes[-1]
            self.last.value = (f"Last breath: paced swing {done.swing_ms:.0f} ms." if done.swing_ms is not None
                               else f"Last breath: not measured. {NOT_MEASURED[done.reason]}")
        if self._best is not None:
            self.best_line.value = self._best_text()
        step = self._step.rate if self._step is not None else None
        self.step_line.visible = step is not None
        if step is not None:
            self.step_line.value = f"In step so far: {step:.1f} breaths a minute."

    def _best_text(self) -> str:
        return (f"Best so far: {self._best.rate:.1f} breaths a minute, paced swing "
                f"{self._best.swing_ms:.0f} ms.")

    def _write_target(self) -> None:
        session = self.session
        target = session.target_breaths_per_min
        usable = session.usable_rate()
        if self._at_rate(usable):
            self.target_line.value = "The target is at this rate. No change is needed."
        elif not self._lo <= target <= self._hi:
            self.target_line.value = f"The target, {target:.1f} breaths a minute, is outside the swept range."
        else:
            sweep = session.sweep
            swing = swing_at(sweep.schedule.rates, sweep.values, target)
            if swing is None:
                self.target_line.value = f"The target, {target:.1f} breaths a minute, is outside the measured rates."
            else:
                self.target_line.value = (f"At the target, {target:.1f} breaths a minute, "
                                          f"the paced swing is {swing:.0f} ms.")

    # --- the curve ----------------------------------------------------------------------------

    def _draw_curve(self, sweep) -> None:
        """Shapes are rebuilt when a breath is measured, the sweep ends or the target moves, and not
        otherwise."""
        values = sweep.values
        target = self.session.target_breaths_per_min
        steps = sweep.steps
        key = (id(sweep), sum(v is not None for v in values), sum(v is not None for v in steps), sweep.reason,
               round(target, 3), self._lo, self._hi)
        if key == self._curve_key:
            return
        self._curve_key = key
        rates = sweep.schedule.rates
        self._points = curve(rates, values)
        self._best = best(rates, values)
        self._step = in_step(rates, steps, self._best.rate if self._best is not None else None)
        w, h = CURVE_W, CURVE_H
        measured = [(rates[k], v) for k, v in enumerate(values) if v is not None]
        peak = max((v for _, v in measured), default=0.0)
        top = nice_ceiling(max(1.0, 1.15 * peak))
        shapes: list = grid(w, h, rows=2, cols=1) + gridlines(self._tick_values(), self._x_of, h)
        for rate, swing in measured:
            shapes.append(cv.Circle(self._x_of(rate), self._y_of(swing, top), 1.8,
                                    paint=ft.Paint(color=t.FAINT, style=ft.PaintingStyle.FILL)))
        # One line for each run of neighbouring windows: a stretch that was not measured breaks the line.
        runs: list[list] = []
        for p in self._points:
            if runs and p.first == runs[-1][-1].first + 1:
                runs[-1].append(p)
            else:
                runs.append([p])
        for run in runs:
            shapes.append(polyline([ft.Offset(self._x_of(p.window_rate), self._y_of(p.median_ms, top))
                                    for p in run], t.ACCENT, 1.8))
        x = self._x_of(target)
        if x is not None:
            y = 0.0
            while y < h:
                shapes.append(polyline([ft.Offset(x, y), ft.Offset(x, min(y + 5.0, h))], t.GOOD, 1.2))
                y += 9.0
        x = self._x_of(self._step.rate) if self._step.rate is not None else None
        if x is not None:       # a grey line with a ring at its top: the rate in step, beside the best rate
            shapes.append(polyline([ft.Offset(x, 0.0), ft.Offset(x, h)], t.MUTED, 1.2))
            shapes.append(cv.Circle(x, MARGIN, 3.0, paint=ft.Paint(color=t.MUTED, stroke_width=1.4,
                                                                   style=ft.PaintingStyle.STROKE)))
        if self._best is not None:
            x, y = self._x_of(self._best.rate), self._y_of(self._best.swing_ms, top)
            shapes.append(polyline([ft.Offset(x, y), ft.Offset(x, h)], t.ACCENT_DIM, 1.0))
            shapes.append(cv.Circle(x, y, 4.0, paint=ft.Paint(color=t.ACCENT, style=ft.PaintingStyle.FILL)))
        self.canvas.shapes = shapes
        place_value_labels(self._labels, h, top, unit="")
        self.axis_note.value = (f"paced swing 0 to {top:g} ms · {self._lo:g} to {self._hi:g} "
                                "breaths a minute")
