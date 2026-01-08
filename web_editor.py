import importlib
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import tkinter as tk

QUILL_WEBVIEW_AVAILABLE = importlib.util.find_spec("webview") is not None

_webview_started = False
_start_requested = False
_webview_failed = False
_command_queue: list[Callable[[], None]] = []
_webview_module = None

_editor_window = None
_editor_ready = threading.Event()
_editor_api = None
_pending_html: str | None = None
_pending_title = "Редактор конспекта (Quill)"
_pending_on_close: Optional[Callable[[], None]] = None

_fallback_window: tk.Toplevel | None = None
_fallback_text: tk.Text | None = None


class EditorAPI:
    def __init__(self, app_ref: tk.Misc, on_make_cards: Optional[Callable[[str], None]] = None) -> None:
        self.app_ref = app_ref
        self.on_make_cards = on_make_cards
        self._window = None

    def attach_window(self, window: Any) -> None:
        self._window = window

    def set_html(self, html: str) -> None:
        if not self._window:
            _store_pending_html(html)
            return
        if not _wait_ready():
            _store_pending_html(html)
            return
        data = json.dumps(html, ensure_ascii=False)
        self._window.evaluate_js(f"setHtml({data});")

    def get_html(self) -> str:
        if not self._window or not _wait_ready():
            return ""
        result = self._window.evaluate_js("getHtml()")
        return result or ""

    def get_selection_html(self) -> str:
        if not self._window or not _wait_ready():
            return ""
        result = self._window.evaluate_js("getSelectionHtml()")
        return result or ""

    def make_cards_from_selection(self) -> bool:
        selection_html = self.get_selection_html()
        if not selection_html:
            selection_html = self.get_html()
        if self.on_make_cards:
            self.app_ref.after(0, lambda: self.on_make_cards(selection_html))
        return True


def ensure_webview_started(root: tk.Misc) -> None:
    global _start_requested
    if _webview_started or _start_requested or _webview_failed or not QUILL_WEBVIEW_AVAILABLE:
        return
    _start_requested = True
    root.after(0, lambda: start_webview_once(root))


def open_editor_window(
    root: tk.Misc,
    api: EditorAPI,
    html: str | None = None,
    title: str = "Редактор конспекта (Quill)",
    on_close: Optional[Callable[[], None]] = None,
) -> bool:
    global _pending_html, _pending_title, _pending_on_close, _editor_api
    _pending_html = html
    _pending_title = title
    _pending_on_close = on_close
    _editor_api = api

    if _webview_failed or not QUILL_WEBVIEW_AVAILABLE:
        _open_fallback_editor(root, html or "", title, on_close)
        return True

    def _command() -> None:
        _create_editor_window(title, on_close)
        if html is not None:
            set_editor_html(html)
        _bring_to_front()

    if _webview_started:
        _command()
    else:
        _command_queue.append(_command)
        ensure_webview_started(root)
    return True


def is_editor_open() -> bool:
    if _editor_window is not None:
        return True
    if _fallback_window is not None and _fallback_window.winfo_exists():
        return True
    return False


def set_editor_html(html: str) -> None:
    if _fallback_text is not None:
        _fallback_text.delete("1.0", tk.END)
        _fallback_text.insert("1.0", html)
        return
    if _editor_api is None:
        _store_pending_html(html)
        return
    _editor_api.set_html(html)


def get_editor_html() -> str | None:
    if _fallback_text is not None:
        return _fallback_text.get("1.0", tk.END).strip()
    if _editor_api is None:
        return None
    return _editor_api.get_html()


def get_selection_html() -> str:
    if _fallback_text is not None:
        try:
            return _fallback_text.get("sel.first", "sel.last")
        except tk.TclError:
            return ""
    if _editor_api is None:
        return ""
    return _editor_api.get_selection_html()


def start_webview_once(root: tk.Misc) -> None:
    global _webview_started, _webview_failed, _webview_module
    if _webview_started or _webview_failed or not QUILL_WEBVIEW_AVAILABLE:
        return
    webview = importlib.import_module("webview")
    _webview_module = webview

    def _on_start() -> None:
        _mark_webview_started()
        _process_command_queue()

    try:
        webview.start(func=_on_start, debug=False, gui="tkinter")
    except Exception:
        _webview_failed = True
        _open_fallback_editor(root, _pending_html or "", _pending_title, _pending_on_close)


def _store_pending_html(html: str) -> None:
    global _pending_html
    _pending_html = html


def _mark_webview_started() -> None:
    global _webview_started
    _webview_started = True


def _process_command_queue() -> None:
    if not _command_queue:
        _create_hidden_window()
    while _command_queue:
        command = _command_queue.pop(0)
        command()


def _create_hidden_window() -> None:
    webview = _get_webview()
    if webview is None:
        return
    try:
        webview.create_window("X-FLASH", html=" ", hidden=True)
    except Exception:
        pass


def _create_editor_window(title: str, on_close: Optional[Callable[[], None]]) -> None:
    global _editor_window
    if _editor_window is not None:
        return
    webview = _get_webview()
    if webview is None:
        return
    if _editor_api is None:
        return
    _editor_ready.clear()
    editor_url = _get_editor_url()
    _editor_window = webview.create_window(
        title,
        url=editor_url,
        width=1100,
        height=700,
        resizable=True,
        js_api=_editor_api,
    )
    _editor_api.attach_window(_editor_window)
    try:
        _editor_window.events.loaded += _on_ready
        _editor_window.events.closed += lambda: _on_closed(on_close)
    except Exception:
        pass


def _get_editor_url() -> str:
    editor_path = Path(__file__).with_name("editor_quill.html")
    return editor_path.resolve().as_uri()


def _bring_to_front() -> None:
    if _editor_window is None:
        return
    try:
        _editor_window.bring_to_front()
    except Exception:
        pass


def _on_ready() -> None:
    global _pending_html
    _editor_ready.set()
    if _pending_html is not None and _editor_api is not None:
        pending_html = _pending_html
        _pending_html = None
        _editor_api.set_html(pending_html)


def _on_closed(on_close: Optional[Callable[[], None]]) -> None:
    global _editor_window
    _editor_window = None
    _editor_ready.clear()
    if on_close:
        on_close()


def _wait_ready(timeout: float = 5.0) -> bool:
    if _editor_window is None:
        return False
    return _editor_ready.is_set() or _editor_ready.wait(timeout)


def _get_webview():
    return _webview_module


def _open_fallback_editor(
    root: tk.Misc,
    html: str,
    title: str,
    on_close: Optional[Callable[[], None]],
) -> None:
    global _fallback_window, _fallback_text
    if _fallback_window is not None and _fallback_window.winfo_exists():
        _fallback_window.deiconify()
        _fallback_window.lift()
        _fallback_text.delete("1.0", tk.END)
        _fallback_text.insert("1.0", html)
        return

    win = tk.Toplevel(root)
    win.title(f"{title} (упрощенный)")
    win.geometry("900x650")
    text = tk.Text(win, wrap="word")
    text.pack(fill=tk.BOTH, expand=True)
    text.insert("1.0", html)
    _fallback_window = win
    _fallback_text = text

    def _handle_close() -> None:
        win.destroy()
        _close_fallback(on_close)

    win.protocol("WM_DELETE_WINDOW", _handle_close)


def _close_fallback(on_close: Optional[Callable[[], None]]) -> None:
    global _fallback_window, _fallback_text
    _fallback_window = None
    _fallback_text = None
    if on_close:
        on_close()
