"""The scaling row that the single-level programs share."""

from __future__ import annotations

from harness import _chosen, _pills

from nfb_console import normalise
from nfb_console.scaling import Scaled, scaling_row


class _Program(Scaled):
    def __init__(self):
        self._init_scaling()

    def _scaling_changed(self) -> None:
        pass


def test_the_lit_pill_is_the_strategy_in_force():
    """Each strategy, once chosen, is the one pill lit in the row."""
    program = _Program()
    labels = {k: spec.label for k, spec in normalise.STRATEGIES.items()}
    for strategy_id, label in labels.items():
        program.set_strategy(strategy_id)
        row = scaling_row(program, lambda: None)
        assert _chosen(row) == {label}, strategy_id
        assert {name for name, _live in _pills(row)} >= set(labels.values())
