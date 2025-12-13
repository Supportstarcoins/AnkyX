import threading
import queue
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Any


class BusyDialog:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.window: tk.Toplevel | None = None
        self.label_var = tk.StringVar(value="")
        self.progressbar: ttk.Progressbar | None = None
        self.mode = "indeterminate"

    def show(self, title: str, mode: str, total: Optional[int] = None):
        if self.window is None or not tk.Toplevel.winfo_exists(self.window):
            self.window = tk.Toplevel(self.root)
            self.window.transient(self.root)
            self.window.grab_set()
            self.window.title(title)
            self.window.resizable(False, False)
            ttk.Label(self.window, textvariable=self.label_var).pack(padx=15, pady=(15, 5))
            self.progressbar = ttk.Progressbar(self.window)
            self.progressbar.pack(fill=tk.X, padx=15, pady=(0, 15))
        else:
            self.window.deiconify()
            self.window.title(title)

        self.mode = mode
        if self.progressbar:
            self.progressbar.config(mode=mode)
            if mode == "determinate":
                self.progressbar.config(maximum=max(total or 0, 1))
                self.progressbar.stop()
                self.progressbar['value'] = 0
            else:
                self.progressbar.config(maximum=100)
                self.progressbar.start(10)
        self.label_var.set(title)
        self.window.update_idletasks()

    def update_progress(self, done: int, total: Optional[int], text: Optional[str] = None):
        if not self.window or not self.progressbar:
            return
        if self.mode == "determinate" and total:
            self.progressbar.config(maximum=max(total, 1))
            self.progressbar['value'] = done
        if text:
            self.label_var.set(text)
        self.window.update_idletasks()

    def close(self):
        if self.progressbar and self.mode == "indeterminate":
            self.progressbar.stop()
        if self.window and tk.Toplevel.winfo_exists(self.window):
            try:
                self.window.grab_release()
            except tk.TclError:
                pass
            self.window.destroy()
        self.window = None
        self.progressbar = None


class TaskRunner:
    def __init__(self, root: tk.Tk, busy_dialog: BusyDialog):
        self.root = root
        self.busy_dialog = busy_dialog
        self._queue: queue.Queue = queue.Queue()
        self._active_thread: threading.Thread | None = None
        self._disabled_widgets: list[tuple[tk.Widget, str]] = []
        self._disabled_menu_states: list[tuple[tk.Menu, int, str]] = []

    def _walk_main_widgets(self) -> list[tk.Widget]:
        widgets: list[tk.Widget] = []

        def walk(widget: tk.Widget):
            for child in widget.winfo_children():
                if isinstance(child, tk.Toplevel) and child is not self.root:
                    continue
                widgets.append(child)
                walk(child)

        walk(self.root)
        return widgets

    def _set_controls_state(self, disabled: bool):
        if disabled:
            for widget in self._walk_main_widgets():
                try:
                    current_state = widget.cget("state") if hasattr(widget, "cget") else "normal"
                except tk.TclError:
                    continue
                if isinstance(widget, (ttk.Button, tk.Button)):
                    self._disabled_widgets.append((widget, current_state))
                    widget_state = tk.DISABLED if disabled else current_state
                    try:
                        widget.config(state=widget_state)
                    except tk.TclError:
                        pass
            menubar_name = None
            try:
                menubar_name = self.root["menu"]
            except Exception:
                menubar_name = None
            menubar = self.root.nametowidget(menubar_name) if menubar_name else None
            if isinstance(menubar, tk.Menu):
                last_index = menubar.index("end") or -1
                for i in range(last_index + 1):
                    try:
                        state = menubar.entrycget(i, "state")
                        self._disabled_menu_states.append((menubar, i, state))
                        menubar.entryconfig(i, state="disabled")
                    except tk.TclError:
                        continue
        else:
            for widget, prev_state in self._disabled_widgets:
                try:
                    widget.config(state=prev_state)
                except tk.TclError:
                    pass
            self._disabled_widgets.clear()
            for menu, index, state in self._disabled_menu_states:
                try:
                    menu.entryconfig(index, state=state)
                except tk.TclError:
                    pass
            self._disabled_menu_states.clear()

    def _start_pump(self):
        self.root.after(30, self._pump_queue)

    def _pump_queue(self):
        handled = False
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            handled = True
            kind = event[0]
            if kind == "progress":
                done, total, text = event[1:]
                self.busy_dialog.update_progress(done, total, text)
            elif kind == "done":
                result = event[1]
                self._finish()
                event[2](result)  # on_success
                return
            elif kind == "error":
                error = event[1]
                self._finish()
                event[2](error)  # on_error
                return
        if self._active_thread and self._active_thread.is_alive():
            self._start_pump()
        elif not handled:
            self._finish()

    def _finish(self):
        self.busy_dialog.close()
        self._set_controls_state(False)
        self._active_thread = None

    def run_task(
        self,
        title: str,
        mode: str,
        task_fn: Callable[[Callable[[int, int | None, str | None], None]], Any],
        on_success: Callable[[Any], None],
        on_error: Callable[[Exception], None],
        total: Optional[int] = None,
    ) -> None:
        if self._active_thread:
            return

        self.busy_dialog.show(title, mode, total)
        self._set_controls_state(True)

        def progress(done: int, total_val: int | None = None, text: str | None = None):
            self._queue.put(("progress", done, total_val, text))

        def target():
            try:
                result = task_fn(progress)
                self._queue.put(("done", result, on_success))
            except Exception as exc:  # noqa: BLE001
                self._queue.put(("error", exc, on_error))

        thread = threading.Thread(target=target, daemon=True)
        self._active_thread = thread
        thread.start()
        self._start_pump()
