import importlib.util
import json
import threading
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import messagebox

QUILL_WEBVIEW_AVAILABLE = importlib.util.find_spec("webview") is not None


class EditorAPI:
    def __init__(self, app_ref: tk.Misc, on_make_cards: Optional[Callable[[str], None]] = None) -> None:
        self.app_ref = app_ref
        self.on_make_cards = on_make_cards
        self.initial_html = ""
        self._window = None
        self._ready_event = threading.Event()

    def attach_window(self, window: Any, on_close: Optional[Callable[[], None]] = None) -> None:
        self._window = window
        self._ready_event.clear()
        if self._window is None:
            return
        try:
            self._window.events.loaded += self._on_loaded
            self._window.events.closed += lambda: self._on_closed(on_close)
        except Exception:
            pass

    def pull_initial_html(self) -> str:
        html = self.initial_html or ""
        self.initial_html = ""
        return html

    def set_html(self, html: str) -> None:
        if not self._window or not self._wait_ready():
            self.initial_html = html
            return
        data = json.dumps(html, ensure_ascii=False)
        self._window.evaluate_js(f"setHtml({data});")

    def get_html(self) -> str:
        if not self._window or not self._wait_ready():
            return ""
        result = self._window.evaluate_js("getHtml()")
        return result or ""

    def get_selection_html(self) -> str:
        if not self._window or not self._wait_ready():
            return ""
        result = self._window.evaluate_js("getSelectionHtml()")
        return result or ""

    def make_cards_from_selection(self) -> bool:
        selection_html = self.get_selection_html()
        if not selection_html:
            selection_html = self.get_html()
        if self.on_make_cards:
            self.app_ref.after(0, lambda: self.on_make_cards(selection_html))
        else:
            length = len(selection_html or "")
            self.app_ref.after(
                0,
                lambda: messagebox.showinfo(
                    "Выделение",
                    f"Получено HTML символов: {length}",
                ),
            )
        return True

    def _on_loaded(self) -> None:
        self._ready_event.set()

    def _on_closed(self, on_close: Optional[Callable[[], None]]) -> None:
        self._window = None
        self._ready_event.clear()
        if on_close:
            on_close()

    def _wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready_event.is_set() or self._ready_event.wait(timeout)


def open_fallback_editor(
    root: tk.Misc,
    html: str,
    title: str,
    reason: str,
    on_close: Optional[Callable[[], None]] = None,
) -> tuple[tk.Toplevel, tk.Text]:
    messagebox.showwarning("Редактор", reason)
    win = tk.Toplevel(root)
    win.title(f"{title} (упрощенный)")
    win.geometry("900x650")
    text = tk.Text(win, wrap="word")
    text.pack(fill=tk.BOTH, expand=True)
    text.insert("1.0", html)

    def _handle_close() -> None:
        win.destroy()
        if on_close:
            on_close()

    win.protocol("WM_DELETE_WINDOW", _handle_close)
    return win, text
