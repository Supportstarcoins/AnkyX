import importlib
import importlib.util
import json
import os
import threading
import time
import traceback
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import messagebox

QUILL_WEBVIEW_AVAILABLE = importlib.util.find_spec("webview") is not None
LOG_PATH = os.path.abspath("web_editor_error.log")


class QuillAPI:
    __slots__ = ("_pending_html", "_ready", "_log_path")

    def __init__(self) -> None:
        self._pending_html = ""
        self._ready = False
        self._log_path = LOG_PATH

    def __dir__(self) -> list[str]:
        return ["pull_initial_html", "notify_editor_ready", "log_js_error"]

    def pull_initial_html(self) -> str:
        html = self._pending_html or ""
        self._pending_html = ""
        return html

    def notify_editor_ready(self) -> bool:
        self._ready = True
        return True

    def log_js_error(self, message: str) -> bool:
        try:
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(str(message) + "\n")
        except Exception:
            pass
        return True


class WebEditorManager:
    def __init__(self, root: tk.Tk, on_make_cards: Optional[Callable[[str], None]] = None) -> None:
        self.root = root
        self.on_make_cards = on_make_cards
        self.editor_window: Any | None = None
        self.api = QuillAPI()

    def attach_window(self, window: Any, on_close: Optional[Callable[[], None]] = None) -> None:
        self.editor_window = window
        self.api._ready = False
        if self.editor_window is None:
            return
        try:
            self.editor_window.events.closed += lambda: self._on_closed(on_close)
        except Exception:
            pass

    def set_html_safe(self, html: str) -> None:
        html = html or ""
        self.api._pending_html = html
        if self.editor_window is not None and self.api._ready:
            payload = json.dumps(html)
            js = f"window.setHtml && window.setHtml({payload});"
            self.api._pending_html = ""
            self.root.after(0, lambda: self.editor_window.evaluate_js(js))

    def get_html(self) -> str:
        if not self.editor_window or not self._wait_ready():
            return ""
        result = self._evaluate_js_sync("window.getHtml && window.getHtml();")
        return result or ""

    def get_selection_html(self) -> str:
        if not self.editor_window or not self._wait_ready():
            return ""
        result = self._evaluate_js_sync("window.getSelectionHtml && window.getSelectionHtml();")
        return result or ""

    def make_cards_from_selection(self) -> bool:
        selection_html = self.get_selection_html()
        if not selection_html:
            selection_html = self.get_html()
        if self.on_make_cards:
            self.root.after(0, lambda: self.on_make_cards(selection_html))
        else:
            length = len(selection_html or "")
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Выделение",
                    f"Получено HTML символов: {length}",
                ),
            )
        return True

    def _on_closed(self, on_close: Optional[Callable[[], None]]) -> None:
        self.editor_window = None
        self.api._ready = False
        if on_close:
            on_close()

    def _wait_ready(self, timeout: float = 5.0) -> bool:
        if self.api._ready:
            return True
        start = time.time()
        while time.time() - start < timeout:
            if self.api._ready:
                return True
            if threading.current_thread() is threading.main_thread():
                try:
                    self.root.update()
                except tk.TclError:
                    break
            time.sleep(0.01)
        return self.api._ready

    def _evaluate_js_sync(self, js: str, timeout: float = 5.0) -> Optional[str]:
        if not self.editor_window:
            return None
        result: dict[str, Optional[str]] = {"value": None}
        done = threading.Event()

        def _run() -> None:
            try:
                result["value"] = self.editor_window.evaluate_js(js)
            except Exception:
                self._log_exc("evaluate_js failed")
            finally:
                done.set()

        self.root.after(0, _run)

        if threading.current_thread() is threading.main_thread():
            start = time.time()
            while not done.is_set() and time.time() - start < timeout:
                try:
                    self.root.update()
                except tk.TclError:
                    break
                time.sleep(0.01)
        else:
            done.wait(timeout)

        return result["value"]

    def _log_message(self, message: str) -> None:
        try:
            with open(self.api._log_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except Exception:
            pass

    def _log_exc(self, tag: str) -> None:
        try:
            with open(self.api._log_path, "a", encoding="utf-8") as handle:
                handle.write(f"\n[{tag}]\n{traceback.format_exc()}\n")
        except Exception:
            pass


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
