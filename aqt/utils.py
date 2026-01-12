from tkinter import messagebox

from addons_manager import mw


def showInfo(text: str, title: str = "Info") -> None:
    if mw and getattr(mw, "ui", None):
        mw.ui.info(text, title=title)
        return
    messagebox.showinfo(title, text)


def tooltip(text: str) -> None:
    if mw and getattr(mw, "ui", None):
        mw.ui.toast(text)
        return
    messagebox.showinfo("Info", text)


def askUser(text: str, title: str = "Confirm") -> bool:
    if mw and getattr(mw, "ui", None):
        return mw.ui.ask(text, title=title)
    return messagebox.askyesno(title, text)
