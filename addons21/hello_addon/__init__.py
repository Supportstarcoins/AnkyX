from aqt import mw, gui_hooks
from aqt.utils import showInfo, tooltip


def on_startup() -> None:
    if mw and getattr(mw, "ui", None):
        mw.ui.toast("Hello Addon loaded")
    else:
        tooltip("Hello Addon loaded")


def on_menu() -> None:
    showInfo("Hello from Hello Addon!")


def setup() -> None:
    if mw and getattr(mw, "ui", None):
        mw.ui.add_menu_item("Инструменты", "Hello Addon", on_menu)
    gui_hooks.app_did_startup.append(on_startup)
