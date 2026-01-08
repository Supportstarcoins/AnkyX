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


class EditorAPI:
    def __init__(self, manager: "WebEditorManager") -> None:
        self.m = manager

    def pull_initial_html(self) -> str:
        html = self.m.pending_html or ""
        self.m.pending_html = None
        return html

    def notify_editor_ready(self) -> bool:
        self.m.editor_ready.set()
        return True

    def log_js_error(self, message: str) -> bool:
        self.m._log_message(message)
        return True

    def make_cards_from_selection(self) -> bool:
        return self.m.make_cards_from_selection()


class WebEditorManager:
    def __init__(self, root: tk.Tk, on_make_cards: Optional[Callable[[str], None]] = None) -> None:
        self.root = root
        self.on_make_cards = on_make_cards
        self.editor_window: Any | None = None
        self.editor_ready = threading.Event()
        self.pending_html: Optional[str] = None
        self.api = EditorAPI(self)

    def attach_window(self, window: Any, on_close: Optional[Callable[[], None]] = None) -> None:
        self.editor_window = window
        self.editor_ready.clear()
        if self.editor_window is None:
            return
        try:
            self.editor_window.events.closed += lambda: self._on_closed(on_close)
        except Exception:
            pass

    def set_html_safe(self, html: str) -> None:
        self.pending_html = html or ""
        if not self.editor_window:
            return
        if not self.editor_ready.is_set():
            self._schedule_apply_when_ready()
            return
        self._apply_pending_now()

    def _schedule_apply_when_ready(self) -> None:
        def tick() -> None:
            if not self.editor_window:
                return
            if self.editor_ready.is_set():
                self._apply_pending_now()
            else:
                self.root.after(100, tick)

        self.root.after(0, tick)

    def _apply_pending_now(self) -> None:
        if not self.editor_window:
            return
        if self.pending_html is None:
            return
        html = self.pending_html or ""
        self.pending_html = None
        js = f"window.setHtml && window.setHtml({json.dumps(html)});"
        try:
            self.root.after(0, lambda: self.editor_window.evaluate_js(js))
        except Exception:
            self._log_exc("apply_pending_now failed")

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
        self.editor_ready.clear()
        if on_close:
            on_close()

    def _wait_ready(self, timeout: float = 5.0) -> bool:
        return self.editor_ready.is_set() or self.editor_ready.wait(timeout)

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
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except Exception:
            pass

    def _log_exc(self, tag: str) -> None:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
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
