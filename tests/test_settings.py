"""The settings menu and its developer section.

The section's two switches are the environment variables' twins, so most of this checks the line between
them: each variable keeps working as it did, `NFB_NO_OUTPUTS` wins over anything the menu says, and the
stored choices reach only a process that is the console.
"""

from __future__ import annotations

import flet as ft

from nfb_console import cues, developer, device_panel, music, prefs, settings_dialog, shadow, sysvolume, \
    training, wakelock

from test_layout import walk
from test_screens import RECORDING, StubPage
from test_ste import faults


def _rows(dialog: ft.AlertDialog) -> dict[str, ft.Row]:
    """The dialog's setting rows, by the label on their left."""
    return {row.controls[0].value: row for row in dialog.content.content.controls}


def _press(row: ft.Row, label: str) -> None:
    pill = next(c for c in row.controls if isinstance(c, ft.Container) and c.content.value == label)
    assert pill.on_click is not None, f"{label!r} is inert"
    pill.on_click(None)


def _lit(row: ft.Row) -> str:
    return next(c.content.value for c in row.controls
                if isinstance(c, ft.Container) and c.bgcolor is not None)


def _open(on_change=lambda: None):
    page = StubPage()
    settings_dialog.show(page, on_change)
    return page.dialogs[-1]


def test_the_developer_section_is_closed_until_its_switch_is_on():
    dialog = _open()
    rows = _rows(dialog)
    assert list(rows) == ["developer settings"], "off by default, and nothing under it is shown"
    assert _lit(rows["developer settings"]) == "off"

    changes = []
    dialog = _open(lambda: changes.append(dict(vars(developer.settings))))
    _press(_rows(dialog)["developer settings"], "on")
    assert changes and changes[-1]["on"] is True, "a change is reported as it is made"
    assert list(_rows(dialog)) == ["developer settings", "debug log", "outputs"]


def test_a_change_in_the_menu_is_stored_and_read_back():
    developer.settings.on = True
    dialog = _open(developer.save)
    _press(_rows(dialog)["debug log"], "on")
    assert prefs.developer() == {"on": True, "debug": True, "outputs_off": False}

    developer.settings.on = developer.settings.debug = False
    developer.load()
    assert developer.settings.on and developer.settings.debug


def test_a_stored_value_of_the_wrong_type_falls_back_to_off():
    prefs.set_developer({"on": "yes", "debug": 1, "outputs_off": True, "stray": True})
    developer.load()
    assert developer.settings == developer.DeveloperSettings(outputs_off=True)


def test_outputs_off_in_the_menu_switches_off_every_output(monkeypatch):
    """The same five modules the variable reaches, through the one function they all ask."""
    monkeypatch.delenv("NFB_NO_OUTPUTS")
    modules = (shadow, sysvolume, cues, music, wakelock)
    developer.settings.outputs_off = True
    assert not any(m.muted() for m in modules), "the master switch is off, so the stored choice waits"

    developer.settings.on = True
    assert all(m.muted() for m in modules)
    assert "the developer settings" in shadow.problem()
    assert "the developer settings" in sysvolume.problem()
    assert "the developer settings" in wakelock.problem()


def test_the_variable_wins_over_the_menu():
    """`NFB_NO_OUTPUTS` is set for the whole suite, and nothing the menu says can turn outputs back on."""
    developer.settings.on = True
    developer.settings.outputs_off = False
    assert shadow.muted() and sysvolume.muted() and cues.muted()
    assert "NFB_NO_OUTPUTS" in shadow.problem()

    row = _rows(_open())["outputs"]
    assert _lit(row) == "off", "the row says what is true, which is that outputs are off"
    pills = [c for c in row.controls if isinstance(c, ft.Container)]
    assert all(p.on_click is None for p in pills), "and it cannot be changed while the variable is set"
    assert any(isinstance(c, ft.Text) and c.value == "NFB_NO_OUTPUTS is set" for c in row.controls)


def test_debug_log_in_the_menu_writes_the_same_logs_as_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("NFB_DEBUG", raising=False)
    monkeypatch.setattr(device_panel, "LOG", str(tmp_path / "panel.log"))
    monkeypatch.setattr(training, "SESSION_LOG", str(tmp_path / "session.log"))
    device_panel.log("before")
    training.slog("before")
    assert not list(tmp_path.iterdir()), "nothing is written while both are off"

    developer.settings.on = developer.settings.debug = True
    device_panel.log("after")
    training.slog("after")
    assert "after" in (tmp_path / "panel.log").read_text()
    assert "after" in (tmp_path / "session.log").read_text()

    developer.settings.on = developer.settings.debug = False
    monkeypatch.setenv("NFB_DEBUG", "1")
    assert developer.debug(), "the variable still turns it on alone"
    developer.settings.on = True
    row = _rows(_open())["debug log"]
    assert _lit(row) == "on" and all(c.on_click is None for c in row.controls
                                     if isinstance(c, ft.Container))


def test_the_settings_menu_is_reached_from_the_launcher_only():
    from nfb_console.app import Console, build_parser

    page = StubPage()
    console = Console(page, build_parser().parse_args([RECORDING]))
    try:
        console.show_launcher()
        chip = console.toolbar.settings_chip
        assert chip.visible
        chip.on_click(None)
        assert page.dialogs and page.dialogs[-1].title.value == "Settings"
        page.pop_dialog()

        console.show_device()
        assert not chip.visible, "the device screen is not the launcher"
        console.show_launcher()
        assert chip.visible
    finally:
        if console.session is not None:
            console.session.stop()


def test_settings_text_is_simplified_technical_english(monkeypatch):
    monkeypatch.delenv("NFB_NO_OUTPUTS")
    found = []
    for env in (False, True):
        if env:
            monkeypatch.setenv("NFB_NO_OUTPUTS", "1")
            monkeypatch.setenv("NFB_DEBUG", "1")
        developer.settings.on = True
        dialog = _open()
        for ctrl in walk(dialog.content):
            if isinstance(ctrl, ft.Text) and ctrl.value:
                found += faults(ctrl.value)
    assert not found, "\n".join(found)


def test_a_developer_program_is_listed_only_while_the_developer_settings_are_on(monkeypatch):
    """No developer program ships yet, so a stand-in proves the mechanism: off, the launcher leaves it out;
    switched on in the menu, the launcher under the dialog lists it with no restart."""
    from nfb_console import protocols
    from nfb_console.app import Console, build_parser

    stand_in = protocols.Protocol("DEVX", "Stand-in", "A developer program", developer=True)
    monkeypatch.setattr(protocols, "CATALOGUE", protocols.CATALOGUE + (stand_in,))

    def codes(page) -> set[str]:
        return {c.value for c in walk(page.controls[-1]) if isinstance(c, ft.Text)}

    page = StubPage()
    console = Console(page, build_parser().parse_args([RECORDING]))
    console.show_launcher()
    assert "DEVX" not in codes(page) and "AMP" in codes(page)

    console.toolbar.settings_chip.on_click(None)
    _press(_rows(page.dialogs[-1])["developer settings"], "on")
    assert "DEVX" in codes(page), "the launcher follows the switch without a restart"

    _press(_rows(page.dialogs[-1])["developer settings"], "off")
    assert "DEVX" not in codes(page)
    assert prefs.developer()["on"] is False, "and the choice is stored"
