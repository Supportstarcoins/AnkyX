import json
import logging
import os
import sys
import traceback
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk, messagebox


LOG_FILENAME = "_logs.txt"
SETTINGS_FILENAME = "_manager.json"


mw = None


def set_mw(mw_obj) -> None:
    global mw
    mw = mw_obj


class Hook:
    def __init__(self, name: str, addon_manager: "AddonManager | None" = None):
        self.name = name
        self._callbacks: list[dict[str, Any]] = []
        self.addon_manager = addon_manager

    def set_addon_manager(self, addon_manager: "AddonManager | None") -> None:
        self.addon_manager = addon_manager

    def append(self, func) -> None:
        addon_id = getattr(func, "__addon_id__", None)
        if addon_id is None and self.addon_manager:
            addon_id = self.addon_manager.loading_addon_id
        self._callbacks.append({"func": func, "addon_id": addon_id})

    def __call__(self, *args, **kwargs) -> None:
        for entry in list(self._callbacks):
            func = entry.get("func")
            addon_id = entry.get("addon_id")
            if not callable(func):
                continue
            if addon_id and self.addon_manager and not self.addon_manager.is_addon_enabled(addon_id):
                continue
            try:
                func(*args, **kwargs)
            except Exception as exc:
                if self.addon_manager:
                    self.addon_manager.handle_addon_error(
                        addon_id,
                        exc,
                        context=f"hook:{self.name}",
                    )
                else:
                    logging.exception("Hook error in %s", self.name)


class GuiHooks:
    def __init__(self, addon_manager: "AddonManager | None" = None):
        self.app_did_startup = Hook("app_did_startup", addon_manager)
        self.deck_will_open = Hook("deck_will_open", addon_manager)
        self.card_will_show = Hook("card_will_show", addon_manager)
        self.card_did_answer = Hook("card_did_answer", addon_manager)
        self.editor_did_open = Hook("editor_did_open", addon_manager)
        self.import_did_finish = Hook("import_did_finish", addon_manager)
        self.export_will_start = Hook("export_will_start", addon_manager)

    def set_addon_manager(self, addon_manager: "AddonManager | None") -> None:
        self.app_did_startup.set_addon_manager(addon_manager)
        self.deck_will_open.set_addon_manager(addon_manager)
        self.card_will_show.set_addon_manager(addon_manager)
        self.card_did_answer.set_addon_manager(addon_manager)
        self.editor_did_open.set_addon_manager(addon_manager)
        self.import_did_finish.set_addon_manager(addon_manager)
        self.export_will_start.set_addon_manager(addon_manager)


gui_hooks = GuiHooks()


@dataclass
class AddonInfo:
    addon_id: str
    name: str
    version: str
    api_version: int
    entry: str
    path: Path
    enabled: bool
    last_error: str | None = None
    last_error_time: str | None = None


class UIAdapter:
    def __init__(self, root: tk.Tk, menubar: tk.Menu | None = None):
        self.root = root
        self.menubar = menubar
        self.addon_manager: "AddonManager | None" = None

    def set_addon_manager(self, addon_manager: "AddonManager") -> None:
        self.addon_manager = addon_manager

    def set_menubar(self, menubar: tk.Menu) -> None:
        self.menubar = menubar

    def _find_menu(self, parent_menu: tk.Menu, label: str) -> tk.Menu | None:
        try:
            end_index = parent_menu.index("end")
        except Exception:
            return None
        if end_index is None:
            return None
        for i in range(end_index + 1):
            try:
                if parent_menu.entrycget(i, "label") == label:
                    menu_name = parent_menu.entrycget(i, "menu")
                    if menu_name:
                        return parent_menu.nametowidget(menu_name)
            except Exception:
                continue
        return None

    def _get_or_create_menu_path(self, menu_path: str) -> tk.Menu:
        if self.menubar is None:
            raise RuntimeError("Menubar is not set")
        parts = [part.strip() for part in menu_path.replace("/", "> ").split(">") if part.strip()]
        current_menu = self.menubar
        for idx, part in enumerate(parts):
            existing = self._find_menu(current_menu, part)
            if existing is None:
                new_menu = tk.Menu(current_menu, tearoff=0)
                current_menu.add_cascade(label=part, menu=new_menu)
                existing = new_menu
            current_menu = existing
        return current_menu

    def add_menu_item(self, menu_path: str, title: str, callback) -> None:
        menu = self._get_or_create_menu_path(menu_path)
        menu.add_command(label=title, command=callback)

    def toast(self, text: str, duration_ms: int = 1600) -> None:
        toast = tk.Toplevel(self.root)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.configure(bg="#333333")
        label = tk.Label(toast, text=text, fg="white", bg="#333333", padx=12, pady=6)
        label.pack()
        self.root.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() // 2) - 100
        y = self.root.winfo_rooty() + 40
        toast.geometry(f"+{x}+{y}")
        toast.after(duration_ms, toast.destroy)

    def info(self, text: str, title: str = "Info") -> None:
        messagebox.showinfo(title, text)

    def ask(self, text: str, title: str = "Confirm") -> bool:
        return messagebox.askyesno(title, text)

    def open_window(self, title: str, build_fn) -> None:
        win = tk.Toplevel(self.root)
        win.title(title)
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        build_fn(win, frame)

    def open_addon_folder(self, addon_id: str) -> None:
        if not self.addon_manager:
            return
        path = self.addon_manager.get_addon_path(addon_id)
        if not path:
            messagebox.showerror("Аддоны", "Папка аддона не найдена")
            return
        open_path(path)

    def open_addon_config(self, addon_id: str) -> None:
        if not self.addon_manager:
            return
        path = self.addon_manager.get_addon_path(addon_id)
        if not path:
            messagebox.showerror("Аддоны", "Папка аддона не найдена")
            return
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            try:
                config_path.write_text("{}", encoding="utf-8")
            except Exception as exc:
                messagebox.showerror("Аддоны", f"Не удалось создать config.json: {exc}")
                return

        def _build(win, frame):
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            text = tk.Text(frame, wrap="word")
            text.grid(row=0, column=0, sticky="nsew")
            try:
                text.insert("1.0", config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                text.insert("1.0", f"Не удалось прочитать файл: {exc}")

            def save_config():
                try:
                    config_path.write_text(text.get("1.0", tk.END).strip(), encoding="utf-8")
                    messagebox.showinfo("Аддоны", "config.json сохранен")
                except Exception as exc:
                    messagebox.showerror("Аддоны", f"Не удалось сохранить config.json: {exc}")

            btn_frame = ttk.Frame(frame)
            btn_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            ttk.Button(btn_frame, text="Сохранить", command=save_config).pack(side=tk.RIGHT)

        self.open_window(f"config.json ({addon_id})", _build)


class CollectionAdapter:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MWContext:
    def __init__(self, root: tk.Tk, ui: UIAdapter, addon_manager: "AddonManager"):
        self.root = root
        self.window = root
        self.ui = ui
        self.addonManager = addon_manager
        self.col = CollectionAdapter()
        self.state: dict[str, Any] = {}


class AddonManager:
    def __init__(self, base_dir: str, ui: UIAdapter | None = None):
        self.base_dir = Path(base_dir)
        self.addons_dir = self.base_dir / "addons21"
        self.addons_dir.mkdir(exist_ok=True)
        self.log_path = self.addons_dir / LOG_FILENAME
        self.settings_path = self.addons_dir / SETTINGS_FILENAME
        self.ui = ui
        self.addons: dict[str, AddonInfo] = {}
        self.addon_modules: dict[str, list[str]] = {}
        self.loading_addon_id: str | None = None
        self.safe_mode = False
        self._init_logger()
        self._load_settings()
        gui_hooks.set_addon_manager(self)

    def _init_logger(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(message)s",
            handlers=[logging.FileHandler(self.log_path, encoding="utf-8")],
        )

    def _load_settings(self) -> None:
        if self.settings_path.exists():
            try:
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                self.safe_mode = bool(data.get("safe_mode", False))
            except Exception:
                self.safe_mode = False

    def _save_settings(self) -> None:
        data = {"safe_mode": self.safe_mode}
        try:
            self.settings_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def get_addon_path(self, addon_id: str) -> str | None:
        info = self.addons.get(addon_id)
        if info:
            return str(info.path)
        path = self.addons_dir / addon_id
        if path.exists():
            return str(path)
        return None

    def _read_manifest(self, addon_dir: Path) -> dict[str, Any] | None:
        manifest_path = addon_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Failed to read manifest for %s", addon_dir)
            return None

    def _read_meta(self, addon_dir: Path) -> dict[str, Any]:
        meta_path = addon_dir / "meta.json"
        if not meta_path.exists():
            meta = {"enabled": True, "last_error": None, "last_error_time": None}
            try:
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            return meta
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {"enabled": True, "last_error": None, "last_error_time": None}

    def _write_meta(self, addon_dir: Path, meta: dict[str, Any]) -> None:
        meta_path = addon_dir / "meta.json"
        try:
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def discover_addons(self) -> dict[str, AddonInfo]:
        self.addons = {}
        for entry in sorted(self.addons_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            manifest = self._read_manifest(entry)
            if not manifest:
                continue
            meta = self._read_meta(entry)
            addon_id = str(manifest.get("id") or entry.name)
            info = AddonInfo(
                addon_id=addon_id,
                name=str(manifest.get("name") or addon_id),
                version=str(manifest.get("version") or "0.0.0"),
                api_version=int(manifest.get("api_version") or 1),
                entry=str(manifest.get("entry") or "__init__:setup"),
                path=entry,
                enabled=bool(meta.get("enabled", True)),
                last_error=meta.get("last_error"),
                last_error_time=meta.get("last_error_time"),
            )
            self.addons[addon_id] = info
        return self.addons

    def is_addon_enabled(self, addon_id: str) -> bool:
        info = self.addons.get(addon_id)
        if not info:
            return False
        return info.enabled

    def set_addon_enabled(self, addon_id: str, enabled: bool) -> None:
        info = self.addons.get(addon_id)
        if not info:
            return
        info.enabled = enabled
        meta = self._read_meta(info.path)
        meta["enabled"] = enabled
        self._write_meta(info.path, meta)

    def _resolve_entry(self, addon: AddonInfo) -> tuple[str, str]:
        entry = addon.entry
        if ":" in entry:
            module_name, func_name = entry.split(":", 1)
        else:
            module_name, func_name = entry, "setup"
        module_name = module_name.strip()
        func_name = func_name.strip()
        return module_name, func_name

    def _import_module(self, addon: AddonInfo, module_name: str) -> Any:
        addon_id = addon.addon_id
        package_base = addon_id
        if module_name in ("__init__", addon_id):
            full_module = package_base
        else:
            full_module = f"{package_base}.{module_name}"
        if str(self.addons_dir) not in sys.path:
            sys.path.insert(0, str(self.addons_dir))
        try:
            module = importlib.import_module(full_module)
            self.addon_modules.setdefault(addon_id, []).append(module.__name__)
            return module
        except Exception:
            module_path = addon.path / (module_name.replace(".", "/") + ".py")
            if module_name == "__init__":
                module_path = addon.path / "__init__.py"
            if not module_path.exists():
                raise
            spec = importlib.util.spec_from_file_location(f"ankyx_addon_{addon_id}_{module_name}", module_path)
            if not spec or not spec.loader:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.addon_modules.setdefault(addon_id, []).append(spec.name)
            return module

    def load_addon(self, addon_id: str) -> None:
        addon = self.addons.get(addon_id)
        if not addon:
            return
        if not addon.enabled:
            return
        if addon.api_version != 1:
            self.handle_addon_error(addon_id, RuntimeError("Unsupported api_version"), context="api_version")
            return
        module_name, func_name = self._resolve_entry(addon)
        try:
            self.loading_addon_id = addon_id
            module = self._import_module(addon, module_name)
            func = getattr(module, func_name, None)
            if not callable(func):
                raise RuntimeError(f"Entry function not found: {func_name}")
            func()
        except Exception as exc:
            self.handle_addon_error(addon_id, exc, context="load")
        finally:
            self.loading_addon_id = None

    def load_addons(self) -> None:
        if self.safe_mode:
            logging.info("Safe mode enabled: skipping addons")
            return
        self.discover_addons()
        for addon_id in list(self.addons.keys()):
            self.load_addon(addon_id)

    def reload_addons(self) -> None:
        self.discover_addons()
        for addon_id, modules in list(self.addon_modules.items()):
            for module_name in modules:
                try:
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                except Exception:
                    pass
        for addon_id in list(self.addons.keys()):
            if self.is_addon_enabled(addon_id):
                self.load_addon(addon_id)

    def handle_addon_error(self, addon_id: str | None, exc: Exception, context: str = "") -> None:
        msg = f"Addon error ({addon_id}) in {context}: {exc}"
        logging.error(msg)
        logging.error(traceback.format_exc())
        if addon_id and addon_id in self.addons:
            addon = self.addons[addon_id]
            meta = self._read_meta(addon.path)
            meta["last_error"] = msg
            meta["last_error_time"] = datetime_now()
            meta["enabled"] = False
            self._write_meta(addon.path, meta)
            addon.enabled = False

    def open_manager_window(self) -> None:
        if not self.ui:
            return
        AddonManagerWindow(self.ui.root, self)


class AddonManagerWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, manager: AddonManager):
        super().__init__(parent)
        self.manager = manager
        self.title("Менеджер аддонов")
        self.geometry("720x420")
        self._build_ui()
        self.refresh_list()

    def _build_ui(self) -> None:
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("name", "id", "version", "enabled")
        self.tree = ttk.Treeview(main, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("name", text="Название")
        self.tree.heading("id", text="ID")
        self.tree.heading("version", text="Версия")
        self.tree.heading("enabled", text="Включен")
        self.tree.column("name", width=240)
        self.tree.column("id", width=160)
        self.tree.column("version", width=80)
        self.tree.column("enabled", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(controls, text="Enable/Disable", command=self.toggle_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Open Config", command=self.open_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="View Error/Log", command=self.view_error).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Reload", command=self.reload_addons).pack(side=tk.RIGHT, padx=4)

        self.safe_mode_var = tk.BooleanVar(value=self.manager.safe_mode)
        safe_frame = ttk.Frame(main)
        safe_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            safe_frame,
            text="Safe Mode (не загружать аддоны)",
            variable=self.safe_mode_var,
            command=self.toggle_safe_mode,
        ).pack(side=tk.LEFT)

    def refresh_list(self) -> None:
        self.tree.delete(*self.tree.get_children())
        addons = self.manager.discover_addons()
        for addon_id, info in addons.items():
            self.tree.insert(
                "",
                tk.END,
                iid=addon_id,
                values=(info.name, info.addon_id, info.version, "Да" if info.enabled else "Нет"),
            )

    def _selected_addon_id(self) -> str | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return selection[0]

    def toggle_selected(self) -> None:
        addon_id = self._selected_addon_id()
        if not addon_id:
            return
        info = self.manager.addons.get(addon_id)
        if not info:
            return
        self.manager.set_addon_enabled(addon_id, not info.enabled)
        self.refresh_list()

    def open_folder(self) -> None:
        addon_id = self._selected_addon_id()
        if not addon_id:
            return
        if self.manager.ui:
            self.manager.ui.open_addon_folder(addon_id)

    def open_config(self) -> None:
        addon_id = self._selected_addon_id()
        if not addon_id:
            return
        if self.manager.ui:
            self.manager.ui.open_addon_config(addon_id)

    def view_error(self) -> None:
        addon_id = self._selected_addon_id()
        if not addon_id:
            return
        info = self.manager.addons.get(addon_id)
        if not info:
            return
        last_error = info.last_error or "Нет ошибок"
        log_path = self.manager.log_path
        messagebox.showinfo("Ошибки аддона", f"{last_error}\n\nЛог: {log_path}")

    def reload_addons(self) -> None:
        self.manager.reload_addons()
        self.refresh_list()

    def toggle_safe_mode(self) -> None:
        self.manager.safe_mode = bool(self.safe_mode_var.get())
        self.manager._save_settings()
        if self.manager.safe_mode:
            messagebox.showinfo("Safe Mode", "Safe Mode включен. Перезапустите приложение.")


def open_path(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
            return
        if sys.platform == "darwin":
            os.system(f"open '{path}'")
            return
        os.system(f"xdg-open '{path}'")
    except Exception:
        try:
            import webbrowser

            webbrowser.open(f"file://{path}")
        except Exception:
            pass


def datetime_now() -> str:
    import datetime

    return datetime.datetime.now().isoformat(timespec="seconds")
