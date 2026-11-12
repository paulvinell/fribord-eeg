"""The two process-wide switches for working on the console: the debug logs, and the outputs.

Each was an environment variable read where it was needed — `NFB_NO_OUTPUTS` in five modules, `NFB_DEBUG`
in three — so there was no one place to ask "are outputs off" and no way to add a second way of turning
them off without a sixth copy. These functions are that place. Every module that touches the display, the
volume, a sound or the sleep hold asks `outputs_off`, and every diagnostic log asks `debug`.

**Two ways to turn each on, and the variable wins.** The settings menu's developer section stores a choice
in `prefs`; the variable is still honoured exactly as before, and nothing in the menu can turn outputs
back on while `NFB_NO_OUTPUTS` is set. A test run, a benchmark or a shot server sets the variable and
never calls `load`, so the operator's stored choices cannot reach them in either direction.

Both are read on every call rather than cached, so setting either mid-process works.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

from . import prefs

OUTPUTS_VARIABLE = "NFB_NO_OUTPUTS"
DEBUG_VARIABLE = "NFB_DEBUG"


@dataclass
class DeveloperSettings:
    """What the developer section of the settings menu holds. Persisted; see `prefs.DEVELOPER`."""

    on: bool = False
    """The master switch. Off, the two below are kept but have no effect, and developer-only programs
    are not listed — see `protocols.listed`."""
    debug: bool = False
    outputs_off: bool = False


settings = DeveloperSettings()
"""The process's one copy. Defaults until `load`, which only the console calls."""


def load() -> None:
    """Read the stored choices into `settings`. Only the fields the dataclass declares, and only booleans,
    so an older or hand-edited file cannot stop the console from starting."""
    stored = prefs.developer()
    for field in dataclasses.fields(DeveloperSettings):
        value = stored.get(field.name)
        setattr(settings, field.name, value if isinstance(value, bool) else field.default)


def save() -> None:
    prefs.set_developer(dataclasses.asdict(settings))


def outputs_off() -> bool:
    """Whether the shadow, the system volume, the sounds and the sleep hold are switched off.

    A session takes over the machine's display or volume the moment it runs, which is correct for a
    session and wrong for anything else. A test suite, a benchmark or a scratch script that imports a
    session should not be able to dim the operator's screen — and one did, which is how this exists.
    """
    return outputs_off_by() is not None


def outputs_off_by() -> str | None:
    """What switched the outputs off, for a module's `problem()` line, or None while they are on."""
    if os.environ.get(OUTPUTS_VARIABLE):
        return OUTPUTS_VARIABLE
    if settings.on and settings.outputs_off:
        return "the developer settings"
    return None


def debug_by_variable() -> bool:
    return bool(os.environ.get(DEBUG_VARIABLE))


def debug() -> bool:
    """Whether the diagnostic logs under /tmp are written. See `device_panel.LOG` for why they exist."""
    return debug_by_variable() or (settings.on and settings.debug)
