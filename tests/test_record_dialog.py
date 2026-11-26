"""The recording options: the file they resolve to, and where recordings go next time."""

from __future__ import annotations

def test_the_record_options_resolve_to_the_file_you_will_actually_get(tmp_path):
    """The dialog's whole job: three fields whose only real question is "what file am I about to
    get", answered with the resolved path — clash suffix included, since a reused name must keep
    tonight rather than refuse it."""
    from nfb_console import prefs
    from nfb_console.record_dialog import RecordSettings

    s = RecordSettings(directory=str(tmp_path), name="", fmt="nfb")
    assert s.resolve().name.endswith(".jsonl.gz")
    s.fmt = "edf"
    assert s.resolve().name.endswith(".edf"), "the format names the suffix"
    s.name = "tuesday"
    assert s.resolve().name == "tuesday.edf"
    s.resolve().write_text("last week")
    assert s.resolve().name == "tuesday-2.edf", "a name reused does not overwrite or refuse"

    prefs.set_record_where(vars(s))
    assert RecordSettings(**prefs.record_where()).resolve().name == "tuesday-2.edf", \
        "and where recordings go outlives the launch"
