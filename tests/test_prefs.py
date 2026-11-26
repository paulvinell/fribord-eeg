"""What the operator set last time: a stored value is applied only where its kind matches."""

from __future__ import annotations

from nfb_console import prefs
from nfb_console.feedback import FeedbackSettings


def test_a_stored_value_of_another_kind_is_not_applied():
    """A malformed file opens a session on the defaults for the values it got wrong, and on the stored
    values for the rest."""
    feedback = FeedbackSettings()
    volume, sound = feedback.click.volume, feedback.click.sound
    applied = prefs.apply(feedback, {"click.volume": True, "click.sound": 3.0, "sound.low": "loud"})
    assert applied == 0
    assert feedback.click.volume == volume and feedback.click.sound == sound


def test_the_record_default_is_per_protocol_and_the_protocol_speaks_until_the_operator_does():
    from nfb_console import prefs

    assert prefs.record_default("AMP") is False
    assert prefs.record_default("SLEEP", fallback=True) is True, "the protocol's own default holds"
    prefs.set_record_default("SLEEP", False)
    assert prefs.record_default("SLEEP", fallback=True) is False, "until the operator touches it"
    prefs.set_record_default("AMP", True)
    assert prefs.record_default("AMP") is True
    assert prefs.record_default("A-T") is False, "one protocol's choice stays its own"


def test_a_malformed_preferences_file_does_not_take_startup_down():
    """Pref loading handled bad JSON syntax but not valid JSON of the wrong shape: a list where a dict was
    expected crashed every reader. It is now tolerated as if the file were absent."""
    from nfb_console import prefs

    prefs.PATH.write_text("[1, 2, 3]")              # valid JSON, wrong shape
    assert prefs.record_where() == {}
    assert prefs.record_default("AMP", fallback=True) is True
    prefs.set_record_default("AMP", False)          # must not raise on the list-shaped file
    assert prefs.record_default("AMP") is False
