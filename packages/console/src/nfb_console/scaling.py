"""The scaling every program that trains one quantity offers: which `normalise` strategy maps the quantity
to 0…1, the strategy's settings, and the row on the Feedback tab that sets them.

One copy for every program. HEG, HRV and Waveform each held their own, line for line, and a program written
after them was given none and invented a fixed scale of its own instead.
"""

from __future__ import annotations

from typing import Callable

import flet as ft

from . import normalise
from . import theme as t
from .widgets import labelled, num_box, pill


class Scaled:
    """The scaling half of a session. A session calls `_init_scaling` once and feeds `strategy`.

    A program with more to do when the scaling changes overrides `_scaling_changed`; one whose strategy needs
    a default only it knows overrides `_scaling_defaults`.
    """

    def _init_scaling(self, strategy_id: str = normalise.DEFAULT_STRATEGY, **params) -> None:
        self.strategy_id = strategy_id
        self.strategy_params: dict = dict(params)
        self._strategy = normalise.build(strategy_id, **params)

    def set_strategy(self, strategy_id: str, **params) -> None:
        if strategy_id not in normalise.STRATEGIES:
            return
        params = self._scaling_defaults(strategy_id, dict(params))
        strategy = normalise.build(strategy_id, **params)   # build first: a bad range raises before any
        self.strategy_id = strategy_id                      # stored state can drift from the live strategy
        self.strategy_params = params
        self._strategy = strategy
        self._scaling_changed()

    @property
    def strategy(self) -> normalise.Strategy:
        return self._strategy

    @property
    def scale_label(self) -> str:
        return normalise.STRATEGIES[self.strategy_id].label

    def _scaling_defaults(self, strategy_id: str, params: dict) -> dict:
        return params

    def _scaling_changed(self) -> None:
        """A new scale has seen none of the old one's values: the signal starts again, and so do the
        target's statistics, which were measured against the old scale."""
        self._signal = None
        self.signal_target.reset_stats()
        self.signal_target._settled = True


def scaling_row(session: Scaled, rebuild: Callable[[], None], *, unit: str = "", step: float = 0.1,
                fixed_range_note: str | None = None) -> ft.Control:
    """The "scaled by" row: one pill per strategy, then only the settings the chosen one has.

    `unit` and `step` are the quantity's, for the settings written in its units. `fixed_range_note`, when
    given, shows a fixed range's ends without boxes and says where they are set instead, for a program
    whose range has another editor.
    """

    def pick(strategy_id: str):
        return lambda _e: (session.set_strategy(strategy_id), rebuild())

    def change(**params) -> None:
        session.set_strategy(session.strategy_id, **{**session.strategy_params, **params})
        rebuild()

    strategy = session.strategy
    settings: list[ft.Control] = []
    if isinstance(strategy, (normalise.ZScore, normalise.RollingMinMax)):
        settings += [ft.Text("over", size=t.SMALL, color=t.FAINT),
                     num_box(strategy.window_s, lambda v: change(window_s=v),
                             suffix="s", step=15, low=10, high=600)]
    if isinstance(strategy, normalise.ZScore):
        settings += [num_box(strategy.clamp_k, lambda v: change(clamp_k=v),
                             suffix="sd", step=0.5, low=0.5, high=5.0)]
    if isinstance(strategy, normalise.FixedRange):
        if fixed_range_note is not None:
            ends = f"{strategy.low:g} to {strategy.high:g}" + (f" {unit}" if unit else "")
            settings += [ft.Text(ends, size=t.BODY, color=t.TEXT), ft.Container(width=6),
                         ft.Text(fixed_range_note, size=t.SMALL, color=t.FAINT)]
        else:
            settings += [num_box(strategy.low, lambda v: change(low=v), suffix=unit, step=step),
                         ft.Text("to", size=t.SMALL, color=t.FAINT),
                         num_box(strategy.high, lambda v: change(high=v), suffix=unit, step=step)]
    if isinstance(strategy, (normalise.Saturating, normalise.Tanh)):
        settings += [ft.Text("at", size=t.SMALL, color=t.FAINT),
                     num_box(strategy.scale, lambda v: change(scale=v), suffix=unit, step=step,
                             low=step / 10)]
    return labelled("scaled by",
                    *[pill(normalise.STRATEGIES[k].label, session.strategy_id == k, pick(k))
                      for k in normalise.STRATEGIES],
                    *settings,
                    note=normalise.STRATEGIES[session.strategy_id].description, width=76)
