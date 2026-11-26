"""The target card's settings, built for one target at a time."""

from __future__ import annotations

from harness import _texts, replay

from nfb_console import amplitude_view as av
from nfb_console.protocols import BY_CODE
from nfb_console.target_card import filter_rows


def test_each_filter_kind_shows_its_own_parameters_and_no_other_kinds():
    """The ripple setting appears for the filter kind that has a ripple and for no other kind."""
    session = av.AmplitudeSession(BY_CODE["AMP"], source=replay(), channels=1)
    try:
        tgt = session.targets[0]
        shown = {}
        for kind in ("cfir", "butterworth", "elliptic"):
            session.retune_filter(tgt, kind=kind)
            shown[kind] = {c.value for row in filter_rows(session, tgt, lambda: None)
                           for c in _texts(row) if c.value}
        assert [kind for kind, texts in shown.items() if "ripple" in texts] == ["elliptic"]
    finally:
        session.stop()
