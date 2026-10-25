"""What the operator set last time, kept between launches.

Everything in this app resets when the process does. `TrainingSession.adopt` carries the configuration across
a *stop* — that was a whole commit's worth of argument, because a fresh session is not a factory reset — and
then the same settings evaporate on quit, which is the same mistake one level up. The alpha-theta sound
selectors are persisted for this reason.

**Only calibration is persisted, and the line is the one the screens already draw.**

- **Calibration** — volumes, floors and ceilings, intervals, pitches, which screen a shadow is on, the hold.
  These belong to the person and the room. They are wrong differently for everyone, they are set once by ear,
  and losing them means setting them by ear again. Persisted.
- **Arrangement** — which mechanisms exist, what each reads, how the targets combine. That is the protocol's,
  and a stored copy would let a file quietly change what a screen trains. Not persisted.
- **Conditioning** — the artifact policy, band averaging, the display corner. Nobody's protocol, but also
  nobody's ear: they are defaults with reasons, and a stored override would hide the reason. Not persisted.

Bands and thresholds are deliberately absent too. A band is what the protocol trains and a threshold is
derived from the first 30 s of *this* session — restoring either would mean opening a screen that claims to
be a protocol and is not, or comparing today's signal against a number from last week.

**One set for every protocol**, because calibration belongs to the person and the room and neither changes
when the screen does — see `SHARED`, which also covers the one real exception and how it is expressed.

Failures are silent by design: a preferences file is a convenience, and a session that will not start
because one is malformed is worse than one that opens with defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PATH = Path(os.environ.get("NFB_PREFS", Path.home() / ".nfb" / "prefs.json"))

# `(object path, attribute)` pairs, resolved against a `FeedbackSettings`. Each one is a number or a string
# the operator sets by ear or by eye, and none of them changes *what* is trained.
CALIBRATED: tuple[tuple[str, str], ...] = (
    ("sound", "low"), ("sound", "high"),
    ("shadow", "low"), ("shadow", "high"), ("shadow", "screen"),
    ("click", "volume"), ("click", "interval_seconds"), ("click", "hold_seconds"),
    ("click", "sound"),
    ("alert", "volume"), ("alert", "interval_seconds"),
    ("music", "volume"), ("music", "pitch_hz"), ("music", "second_pitch_hz"),
    # How far the EEG's own pitch travels — the same kind of number as a volume: chosen by ear against one
    # person's rhythm, changing nothing about what is trained. See `feedback.Music.eeg_pitch_lo_hz`.
    ("music", "eeg_pitch_lo_hz"), ("music", "eeg_pitch_hi_hz"),
)

_ENABLED = ("sound", "shadow", "click", "alert", "music")
"""Whether each mechanism is on is **not** persisted. It is arrangement: `TrainingSession.offers` and
`__init__` decide it, and a stored `music.enabled = True` would give a protocol music its design never had —
which is the swallowed-refusal shape, in a file."""


def _load() -> dict:
    try:
        return json.loads(PATH.read_text())
    except Exception:
        return {}


def _save(data: dict) -> None:
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(data, indent=1, sort_keys=True))
    except OSError:
        pass                    # a preferences file is a convenience; never take a session down for one


def capture(feedback) -> dict:
    """The calibrated values as a flat dict. Only what is actually present, so an older file and a newer
    build differ by absence rather than by error."""
    out = {}
    for owner, attr in CALIBRATED:
        mech = getattr(feedback, owner, None)
        if mech is not None and hasattr(mech, attr):
            out[f"{owner}.{attr}"] = getattr(mech, attr)
    return out


def apply(feedback, values: dict) -> int:
    """Write stored values back, skipping anything that no longer exists or no longer type-checks.

    A stored key that has since been removed is not an error — it is a build that moved on — and a value of
    the wrong type is dropped rather than coerced, because a `volume` of `"loud"` should leave the default
    alone and not become 0.

    **Int and float count as one type here, in both directions**, and the asymmetry that used to be here threw
    calibration away. `Sound.low` is annotated `float` and defaulted `60` — an int at runtime — so a stored
    `30.0` did not match the default's type and was silently dropped, while a stored `30` would have been kept.
    Every number a slider writes is a float, so on an untouched field that meant the volume you set never came
    back. The defaults are floats now, which fixes that one; this stays because the next such default will not
    be, and losing a stored value over `60` versus `60.0` is not a distinction worth honouring.
    """
    n = 0
    for key, value in values.items():
        owner, _, attr = key.partition(".")
        if (owner, attr) not in CALIBRATED:
            continue
        mech = getattr(feedback, owner, None)
        if mech is None or not hasattr(mech, attr):
            continue
        current = getattr(mech, attr)
        numbers = (int, float)
        if isinstance(current, bool) or isinstance(value, bool):
            if not (isinstance(current, bool) and isinstance(value, bool)):
                continue
        elif not (isinstance(current, numbers) and isinstance(value, numbers)):
            if not isinstance(value, type(current)):
                continue
        setattr(mech, attr, type(current)(value))
        n += 1
    return n


SHARED = "*"
"""The key the calibration lives under. Not a protocol code — no code can be `*` — so an older file's
per-protocol entries sit beside it untouched and readable.

**One set, because the room is one room.** This used to be keyed by protocol code, on the argument that a
volume balanced for noise on alpha-theta is not the volume for a click on the amplitude screen. That is true
of alpha-theta and it is true of nothing else, and the price was paid on every screen: a click volume set by
ear on one protocol did not exist on the next, so the same person in the same chair with the same speakers
re-balanced the same three numbers seven times. The exception was real and the rule drawn from it was wrong.

What the exception actually needs is `pinned_feedback` — the set a protocol declares it owns, which is
already why those rows are inert on screen. A pinned mechanism is not the operator's to set, so it is
neither written here nor read back: alpha-theta's quieter music stays alpha-theta's, and cannot leak out to
a screen that never asked for it or be overwritten by one."""


def _shareable(session, values: dict) -> dict:
    """Drop anything the protocol owns. See `SHARED` — a pinned mechanism is the protocol's, not the room's."""
    pinned = getattr(session, "pinned_feedback", frozenset())
    return {k: v for k, v in values.items() if k.partition(".")[0] not in pinned}


RECORD = "record"
"""Whether a protocol's sessions record themselves, keyed by protocol code per program. Not calibration, but the same side of the line: a standing workflow choice that belongs to
the operator and changes nothing about what is trained. Per protocol rather than shared because the
answer genuinely differs by program — a sleep recording is the product, an amplitude recording is a
paper trail — and a protocol may declare its own default (`records_by_default`), which this overrides
only once the operator has actually touched the toggle."""


RECORD_WHERE = "record_where"
"""Folder, name stem and format for recordings — one set for the app, not per protocol. Where files
go is a property of the machine and the operator's filing, not of what is being trained."""


def record_where() -> dict:
    stored = _load().get(RECORD_WHERE)
    return dict(stored) if isinstance(stored, dict) else {}


def set_record_where(values: dict) -> None:
    data = _load()
    data[RECORD_WHERE] = dict(values)
    _save(data)


def record_default(code: str, fallback: bool = False) -> bool:
    value = _load().get(RECORD, {}).get(code)
    return value if isinstance(value, bool) else fallback


def set_record_default(code: str, on: bool) -> None:
    data = _load()
    data.setdefault(RECORD, {})[code] = bool(on)
    _save(data)


def restore(protocol_code: str, session) -> int:
    """Apply the stored calibration to a session. Returns how many values were used.

    **Takes the session, not its `feedback`, and that was a crash.** These two functions were written when
    every screen was a `TrainingSession` and therefore had one; the HRV screen is not and does not, so
    opening it as the first protocol of a run raised `AttributeError` inside `Console.pick` — present since
    prefs were added, and briefly masked while that screen carried a redundant `FeedbackSettings`.

    Asking the session what its calibration is also keeps this file's whitelist honest: `CALIBRATED` is a
    list of paths into a `FeedbackSettings`, and a screen built on something else needs to name its own.

    `protocol_code` survives the move to one shared set **only to read an older file**, so that upgrading
    does not silently throw away calibration somebody set by ear. It is a fallback, not a layer: once
    anything has been stored the shared set is the whole answer, and a stale per-protocol entry stops being
    consulted rather than being merged underneath.
    """
    data = _load()
    values = data.get(SHARED) or data.get(protocol_code, {})
    return session.restore_calibration(_shareable(session, values))


def remember(protocol_code: str, session) -> None:
    """Store what the operator set, for every screen that does not pin it.

    Merged rather than replaced, because screens do not all carry the same keys — the HRV screen stores its
    pacer period and its target frequency here too, and a later session on another protocol must not delete
    them by writing a dict that never had them.
    """
    data = _load()
    data[SHARED] = {**data.get(SHARED, {}), **_shareable(session, session.calibration())}
    _save(data)
