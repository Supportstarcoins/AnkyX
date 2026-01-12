import json
import logging
import os
import shutil
import sys
import traceback
import importlib
import importlib.util
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


LOG_FILENAME = "addons.log"
SETTINGS_FILENAME = "_manager.json"
BACKUP_DIRNAME = "_backup"


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
    needs_restart: bool = False
    invalid: bool = False
    manifest: dict[str, Any] | None = None


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
    # PATCH: addons manager install/update/remove + oldschool folder support
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
        self._restart_required = False
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
        base = {
            "enabled": True,
            "last_error": None,
            "last_error_time": None,
            "installed_at": None,
            "updated_at": None,
            "needs_restart": False,
        }
        if not meta_path.exists():
            try:
                meta_path.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            return dict(base)
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        merged = dict(base)
        merged.update(data)
        return merged

    def _write_meta(self, addon_dir: Path, meta: dict[str, Any]) -> None:
        meta_path = addon_dir / "meta.json"
        try:
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def validate_manifest(self, manifest_dict: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if not isinstance(manifest_dict, dict):
            return False, ["manifest.json должен быть объектом"]
        if "id" in manifest_dict and not isinstance(manifest_dict.get("id"), str):
            errors.append("Поле id должно быть строкой")
        if "name" in manifest_dict and not isinstance(manifest_dict.get("name"), str):
            errors.append("Поле name должно быть строкой")
        if "version" in manifest_dict and not isinstance(manifest_dict.get("version"), str):
            errors.append("Поле version должно быть строкой")
        if "api_version" in manifest_dict:
            try:
                int(manifest_dict.get("api_version"))
            except Exception:
                errors.append("Поле api_version должно быть числом")
        if "entry" in manifest_dict and not isinstance(manifest_dict.get("entry"), str):
            errors.append("Поле entry должно быть строкой")
        return len(errors) == 0, errors

    def read_manifest_from_zip(self, zip_path: str) -> dict[str, Any] | None:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                manifest_name = self._find_manifest_in_zip(zf)
                if not manifest_name:
                    return None
                with zf.open(manifest_name) as handle:
                    return json.loads(handle.read().decode("utf-8"))
        except Exception:
            logging.exception("Failed to read manifest from %s", zip_path)
            return None

    def _find_manifest_in_zip(self, zf: zipfile.ZipFile) -> str | None:
        candidates = [name for name in zf.namelist() if name.endswith("manifest.json")]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item.count("/"), len(item)))
        return candidates[0]

    def _locate_manifest_dir(self, base: Path) -> tuple[Path | None, dict[str, Any] | None]:
        manifest = self._read_manifest(base)
        if manifest:
            return base, manifest
        for entry in base.iterdir():
            if entry.is_dir():
                manifest = self._read_manifest(entry)
                if manifest:
                    return entry, manifest
        return None, None

    def _get_install_preview_from_zip(self, zip_path: str) -> dict[str, Any]:
        preview: dict[str, Any] = {"files": [], "size": 0, "manifest": None}
        try:
            with zipfile.ZipFile(zip_path) as zf:
                preview["size"] = sum(item.file_size for item in zf.infolist())
                preview["files"] = [item.filename for item in zf.infolist() if not item.is_dir()]
                manifest = self.read_manifest_from_zip(zip_path)
                preview["manifest"] = manifest
        except Exception:
            logging.exception("Failed to preview zip %s", zip_path)
        return preview

    def _get_install_preview_from_folder(self, folder_path: str) -> dict[str, Any]:
        base = Path(folder_path)
        files: list[str] = []
        total_size = 0
        if base.exists():
            for root, _, filenames in os.walk(base):
                for filename in filenames:
                    full = Path(root) / filename
                    try:
                        total_size += full.stat().st_size
                    except Exception:
                        pass
                    rel = str(full.relative_to(base))
                    files.append(rel)
        manifest = self._read_manifest(base)
        if manifest is None:
            located_root, located_manifest = self._locate_manifest_dir(base)
            if located_root:
                manifest = located_manifest
        return {"files": files, "size": total_size, "manifest": manifest}

    def discover_addons(self) -> dict[str, AddonInfo]:
        self.addons = {}
        for entry in sorted(self.addons_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            manifest = self._read_manifest(entry)
            if not manifest:
                meta = self._read_meta(entry)
                addon_id = entry.name
                info = AddonInfo(
                    addon_id=addon_id,
                    name="Invalid addon",
                    version="-",
                    api_version=0,
                    entry="",
                    path=entry,
                    enabled=bool(meta.get("enabled", False)),
                    last_error=meta.get("last_error"),
                    last_error_time=meta.get("last_error_time"),
                    needs_restart=bool(meta.get("needs_restart", False)),
                    invalid=True,
                    manifest=None,
                )
                self.addons[addon_id] = info
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
                needs_restart=bool(meta.get("needs_restart", False)),
                invalid=False,
                manifest=manifest,
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
        meta["needs_restart"] = True
        meta["updated_at"] = datetime_now()
        self._write_meta(info.path, meta)
        self._restart_required = True

    def install_from_zip(self, zip_path: str) -> tuple[bool, str]:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                manifest_name = self._find_manifest_in_zip(zf)
                manifest = self.read_manifest_from_zip(zip_path)
                root_dir = ""
                if manifest_name:
                    root_dir = str(Path(manifest_name).parent)
                with tempfile_directory() as temp_dir:
                    zf.extractall(temp_dir)
                    source_root = Path(temp_dir)
                    if root_dir and root_dir != ".":
                        source_root = Path(temp_dir) / root_dir
                    if not source_root.exists():
                        source_root = Path(temp_dir)
                    if manifest is None:
                        located_root, located_manifest = self._locate_manifest_dir(source_root)
                        if located_root:
                            source_root = located_root
                            manifest = located_manifest
                    addon_id = str(manifest.get("id")) if manifest else source_root.name
                    if (self.addons_dir / addon_id).exists():
                        return self.update_addon(addon_id, source_root)
                    return self._install_from_source(addon_id, source_root, manifest)
        except Exception as exc:
            logging.exception("Failed to install addon from zip")
            return False, str(exc)

    def install_from_folder(self, folder_path: str) -> tuple[bool, str]:
        source_root = Path(folder_path)
        if not source_root.exists():
            return False, "Папка не найдена"
        manifest = self._read_manifest(source_root)
        if manifest is None:
            located_root, located_manifest = self._locate_manifest_dir(source_root)
            if located_root:
                source_root = located_root
                manifest = located_manifest
        addon_id = str(manifest.get("id")) if manifest else source_root.name
        if (self.addons_dir / addon_id).exists():
            return self.update_addon(addon_id, source_root)
        return self._install_from_source(addon_id, source_root, manifest)

    def _install_from_source(
        self,
        addon_id: str,
        source_root: Path,
        manifest: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        dest = self.addons_dir / addon_id
        if dest.exists():
            return False, "Аддон уже установлен"
        try:
            shutil.copytree(source_root, dest)
            meta = self._read_meta(dest)
            meta["enabled"] = True
            meta["installed_at"] = datetime_now()
            meta["updated_at"] = datetime_now()
            meta["needs_restart"] = True
            self._write_meta(dest, meta)
            self._restart_required = True
            return True, "Installed"
        except Exception as exc:
            logging.exception("Failed to install addon")
            return False, str(exc)

    def update_addon(self, addon_id: str, source: Path) -> tuple[bool, str]:
        addon_dir = self.addons_dir / addon_id
        if not addon_dir.exists():
            return self._install_from_source(addon_id, source, self._read_manifest(source))
        old_meta = self._read_meta(addon_dir)
        preserved_config = None
        preserved_user_data = None
        config_path = addon_dir / "config.json"
        if config_path.exists():
            try:
                preserved_config = config_path.read_text(encoding="utf-8")
            except Exception:
                preserved_config = None
        user_data_path = addon_dir / "user-data"
        if user_data_path.exists() and user_data_path.is_dir():
            preserved_user_data = user_data_path
        backup_path = self.backup_addon(addon_id)
        backup_user_data = Path(backup_path) / "user-data" if preserved_user_data else None
        try:
            shutil.rmtree(addon_dir)
        except Exception:
            pass
        try:
            shutil.copytree(source, addon_dir)
            if preserved_config is not None:
                new_config_path = addon_dir / "config.json"
                if new_config_path.exists():
                    (addon_dir / "config.user.json").write_text(preserved_config, encoding="utf-8")
                else:
                    new_config_path.write_text(preserved_config, encoding="utf-8")
            if backup_user_data is not None and backup_user_data.exists() and not (addon_dir / "user-data").exists():
                shutil.copytree(backup_user_data, addon_dir / "user-data")
            meta = self._read_meta(addon_dir)
            meta["enabled"] = bool(old_meta.get("enabled", True))
            meta["installed_at"] = old_meta.get("installed_at") or datetime_now()
            meta["updated_at"] = datetime_now()
            meta["last_error"] = None
            meta["last_error_time"] = None
            meta["needs_restart"] = True
            self._write_meta(addon_dir, meta)
            self._restart_required = True
            return True, f"Updated (backup: {backup_path})"
        except Exception as exc:
            logging.exception("Failed to update addon")
            return False, str(exc)

    def remove_addon(self, addon_id: str, move_to_backup: bool = True) -> tuple[bool, str]:
        addon_dir = self.addons_dir / addon_id
        if not addon_dir.exists():
            return False, "Аддон не найден"
        try:
            backup_path = None
            if move_to_backup:
                backup_path = self.backup_addon(addon_id)
            shutil.rmtree(addon_dir)
            self._restart_required = True
            return True, f"Removed{f' (backup: {backup_path})' if backup_path else ''}"
        except Exception as exc:
            logging.exception("Failed to remove addon")
            return False, str(exc)

    def backup_addon(self, addon_id: str) -> str:
        addon_dir = self.addons_dir / addon_id
        backup_root = self.addons_dir / BACKUP_DIRNAME
        backup_root.mkdir(exist_ok=True)
        backup_path = backup_root / f"{addon_id}_{datetime_now().replace(':', '-')}"
        shutil.copytree(addon_dir, backup_path)
        return str(backup_path)

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
        if addon.invalid:
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
            meta["needs_restart"] = True
            self._write_meta(addon.path, meta)
            addon.enabled = False
            addon.needs_restart = True
            self._restart_required = True

    def restart_required(self) -> bool:
        if self._restart_required:
            return True
        for addon in self.addons.values():
            if addon.needs_restart:
                return True
        return False

    def open_manager_window(self) -> None:
        if not self.ui:
            return
        AddonManagerWindow(self.ui.root, self)


class AddonManagerWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, manager: AddonManager):
        super().__init__(parent)
        self.manager = manager
        self.search_var = tk.StringVar()
        self.dnd_available = False
        self.title("Менеджер аддонов")
        self.geometry("980x560")
        self._build_ui()
        self.refresh_list()

    def _build_ui(self) -> None:
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        header = ttk.Frame(main)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Можно установить вручную: распакуйте аддон в addons21/ и нажмите Reload",
        ).pack(side=tk.LEFT)

        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(8, 8))

        ttk.Label(top, text="Search:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(top, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8))
        search_entry.bind("<KeyRelease>", lambda _event: self.refresh_list())

        ttk.Button(top, text="Install from ZIP...", command=self.install_zip).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Install from Folder...", command=self.install_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Open addons21 Folder", command=self.open_addons_folder).pack(side=tk.LEFT, padx=4)

        self.dnd_label = ttk.Label(main, text="Drag & Drop .zip сюда")
        self.dnd_label.pack(fill=tk.X)
        self._setup_drag_and_drop()

        content = ttk.Panedwindow(main, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        table_frame = ttk.Frame(content)
        content.add(table_frame, weight=3)

        columns = ("name", "id", "version", "enabled", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("name", text="Name")
        self.tree.heading("id", text="ID")
        self.tree.heading("version", text="Version")
        self.tree.heading("enabled", text="Enabled")
        self.tree.heading("status", text="Status")
        self.tree.column("name", width=220)
        self.tree.column("id", width=170)
        self.tree.column("version", width=90)
        self.tree.column("enabled", width=80)
        self.tree.column("status", width=130)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self.update_preview())
        self.tree.bind("<Double-1>", lambda _event: self.toggle_selected())

        preview_frame = ttk.Frame(content)
        content.add(preview_frame, weight=2)

        ttk.Label(preview_frame, text="Details").pack(anchor="w")
        self.preview_text = tk.Text(preview_frame, height=10, wrap="word")
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        self.preview_text.configure(state="disabled")

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(controls, text="Enable/Disable", command=self.toggle_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Remove", command=self.remove_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Open Config", command=self.open_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="View Error/Log", command=self.view_error).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Reload", command=self.reload_addons).pack(side=tk.RIGHT, padx=4)
        self.restart_button = ttk.Button(controls, text="Restart Now", command=self.restart_now)
        self.restart_button.pack(side=tk.RIGHT, padx=4)

        self.safe_mode_var = tk.BooleanVar(value=self.manager.safe_mode)
        safe_frame = ttk.Frame(main)
        safe_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(
            safe_frame,
            text="Safe Mode (не загружать аддоны)",
            variable=self.safe_mode_var,
            command=self.toggle_safe_mode,
        ).pack(side=tk.LEFT)

    def _setup_drag_and_drop(self) -> None:
        try:
            if hasattr(self, "drop_target_register"):
                self.drop_target_register("DND_Files")
                self.dnd_available = True
        except Exception:
            self.dnd_available = False
        if self.dnd_available:
            self.dnd_label.configure(text="Drag & Drop .zip сюда")
            if hasattr(self.dnd_label, "dnd_bind"):
                self.dnd_label.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore[attr-defined]
            else:
                self.dnd_available = False
        if not self.dnd_available:
            self.dnd_label.configure(text="Drag & Drop не доступен (установите через кнопки Install)")

    def _on_drop(self, event) -> None:
        if not event.data:
            return
        paths = self.tk.splitlist(event.data)
        for path in paths:
            if str(path).lower().endswith(".zip"):
                self._install_with_preview_zip(path)
                break

    def refresh_list(self) -> None:
        self.tree.delete(*self.tree.get_children())
        addons = self.manager.discover_addons()
        query = self.search_var.get().strip().lower()
        for addon_id, info in addons.items():
            if query and query not in info.name.lower() and query not in addon_id.lower():
                continue
            status = self._status_for(info)
            self.tree.insert(
                "",
                tk.END,
                iid=addon_id,
                values=(info.name, info.addon_id, info.version, "Yes" if info.enabled else "No", status),
            )
        self.update_preview()
        self._update_restart_button()

    def _status_for(self, info: AddonInfo) -> str:
        if info.invalid:
            return "Invalid"
        if info.last_error:
            return "Error"
        if info.needs_restart:
            return "Needs restart"
        return "OK"

    def _selected_addon_id(self) -> str | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return selection[0]

    def update_preview(self) -> None:
        addon_id = self._selected_addon_id()
        info = self.manager.addons.get(addon_id) if addon_id else None
        details = []
        if info is None:
            details.append("Выберите аддон для просмотра деталей.")
        else:
            details.append(f"Name: {info.name}")
            details.append(f"ID: {info.addon_id}")
            details.append(f"Version: {info.version}")
            details.append(f"API version: {info.api_version}")
            details.append(f"Enabled: {'Yes' if info.enabled else 'No'}")
            details.append(f"Path: {info.path}")
            if info.manifest:
                author = info.manifest.get("author")
                description = info.manifest.get("description")
                permissions = info.manifest.get("permissions")
                homepage = info.manifest.get("homepage")
                if author:
                    details.append(f"Author: {author}")
                if homepage:
                    details.append(f"Homepage: {homepage}")
                if permissions:
                    details.append(f"Permissions: {', '.join(permissions)}")
                if description:
                    details.append("Description:")
                    details.append(str(description))
            if info.last_error:
                details.append("Last error:")
                details.append(info.last_error)
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert("1.0", "\n".join(details))
        self.preview_text.configure(state="disabled")

    def toggle_selected(self) -> None:
        addon_id = self._selected_addon_id()
        if not addon_id:
            return
        info = self.manager.addons.get(addon_id)
        if not info:
            return
        if info.invalid:
            messagebox.showwarning("Аддоны", "Невозможно включить аддон без manifest.json")
            return
        self.manager.set_addon_enabled(addon_id, not info.enabled)
        self.manager.reload_addons()
        self.refresh_list()

    def open_addons_folder(self) -> None:
        open_path(str(self.manager.addons_dir))

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
        log_tail = self._read_log_tail()

        def _build(win, frame):
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            text = tk.Text(frame, wrap="word")
            text.grid(row=0, column=0, sticky="nsew")
            text.insert("1.0", f"Last error:\n{last_error}\n\nLog tail:\n{log_tail}")
            text.configure(state="disabled")

        self.manager.ui.open_window("Addon Error/Log", _build) if self.manager.ui else messagebox.showinfo(
            "Ошибки аддона",
            f"{last_error}\n\nLog:\n{log_tail}",
        )

    def _read_log_tail(self, lines: int = 120) -> str:
        log_path = self.manager.log_path
        if not log_path.exists():
            return "Log not found"
        try:
            content = log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return "Log not available"
        tail = content[-lines:]
        return "\n".join(tail)

    def reload_addons(self) -> None:
        self.manager.reload_addons()
        self.refresh_list()

    def toggle_safe_mode(self) -> None:
        self.manager.safe_mode = bool(self.safe_mode_var.get())
        self.manager._save_settings()
        if self.manager.safe_mode:
            messagebox.showinfo("Safe Mode", "Safe Mode включен. Перезапустите приложение.")

    def install_zip(self) -> None:
        path = filedialog.askopenfilename(
            title="Install from ZIP",
            filetypes=[("ZIP", "*.zip")],
        )
        if not path:
            return
        self._install_with_preview_zip(path)

    def install_folder(self) -> None:
        path = filedialog.askdirectory(title="Install from Folder")
        if not path:
            return
        self._install_with_preview_folder(path)

    def _install_with_preview_zip(self, path: str) -> None:
        preview = self.manager._get_install_preview_from_zip(path)
        self._show_install_preview(
            source_label=path,
            preview=preview,
            on_install=lambda: self._perform_install_zip(path),
        )

    def _install_with_preview_folder(self, path: str) -> None:
        preview = self.manager._get_install_preview_from_folder(path)
        self._show_install_preview(
            source_label=path,
            preview=preview,
            on_install=lambda: self._perform_install_folder(path),
        )

    def _show_install_preview(self, source_label: str, preview: dict[str, Any], on_install) -> None:
        manifest = preview.get("manifest")
        files = preview.get("files") or []
        size = preview.get("size") or 0
        valid = False
        errors: list[str] = []
        if manifest:
            valid, errors = self.manager.validate_manifest(manifest)

        detail_lines = [f"Source: {source_label}"]
        if manifest:
            detail_lines.append(f"Name: {manifest.get('name', 'Unknown')}")
            detail_lines.append(f"ID: {manifest.get('id', 'Unknown')}")
            detail_lines.append(f"Version: {manifest.get('version', '-')}")
            detail_lines.append(f"Author: {manifest.get('author', '-')}")
            detail_lines.append(f"API version: {manifest.get('api_version', '-')}")
            if manifest.get("description"):
                detail_lines.append("Description:")
                detail_lines.append(str(manifest.get("description")))
            if manifest.get("permissions"):
                detail_lines.append(f"Permissions: {', '.join(manifest.get('permissions'))}")
        else:
            detail_lines.append("Manifest not found: addon будет помечен как Invalid")
        if errors:
            detail_lines.append("Manifest warnings:")
            detail_lines.extend(errors)
        detail_lines.append(f"Files: {len(files)}")
        detail_lines.append(f"Approx size: {format_size(size)}")
        if files:
            detail_lines.append("Files preview:")
            detail_lines.extend(files[:120])
            if len(files) > 120:
                detail_lines.append("...")
        detail_lines.append("\n⚠️ Аддоны — это Python-код. Устанавливайте только из доверенных источников.")

        def _build(win, frame):
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            text = tk.Text(frame, wrap="word")
            text.grid(row=0, column=0, sticky="nsew")
            text.insert("1.0", "\n".join(detail_lines))
            text.configure(state="disabled")

            buttons = ttk.Frame(frame)
            buttons.grid(row=1, column=0, sticky="ew", pady=(8, 0))

            def _confirm():
                win.destroy()
                on_install()

            ttk.Button(buttons, text="Install anyway", command=_confirm).pack(side=tk.RIGHT, padx=4)
            ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)

        if self.manager.ui:
            self.manager.ui.open_window("Install Preview", _build)
        else:
            if messagebox.askyesno("Install Preview", "\n".join(detail_lines)):
                on_install()

    def _perform_install_zip(self, path: str) -> None:
        manifest = self.manager.read_manifest_from_zip(path)
        addon_id = str(manifest.get("id")) if manifest else Path(path).stem
        if (self.manager.addons_dir / addon_id).exists():
            if not messagebox.askyesno("Update addon", f"Аддон {addon_id} уже установлен. Обновить?"):
                return
        ok, msg = self.manager.install_from_zip(path)
        if not ok:
            messagebox.showerror("Install", msg)
        else:
            messagebox.showinfo("Install", "Готово")
        self.manager.reload_addons()
        self.refresh_list()

    def _perform_install_folder(self, path: str) -> None:
        manifest = self.manager._read_manifest(Path(path))
        if manifest is None:
            located_root, located_manifest = self.manager._locate_manifest_dir(Path(path))
            if located_root:
                manifest = located_manifest
        addon_id = str(manifest.get("id")) if manifest else Path(path).name
        if (self.manager.addons_dir / addon_id).exists():
            if not messagebox.askyesno("Update addon", f"Аддон {addon_id} уже установлен. Обновить?"):
                return
        ok, msg = self.manager.install_from_folder(path)
        if not ok:
            messagebox.showerror("Install", msg)
        else:
            messagebox.showinfo("Install", "Готово")
        self.manager.reload_addons()
        self.refresh_list()

    def remove_selected(self) -> None:
        addon_id = self._selected_addon_id()
        if not addon_id:
            return
        choice = messagebox.askyesnocancel(
            "Удаление аддона",
            "Удалить аддон?\n\nYes = Переместить в backup\nNo = Удалить навсегда",
        )
        if choice is None:
            return
        move_to_backup = bool(choice)
        ok, msg = self.manager.remove_addon(addon_id, move_to_backup=move_to_backup)
        if not ok:
            messagebox.showerror("Remove", msg)
        else:
            messagebox.showinfo("Remove", "Готово")
        self.manager.reload_addons()
        self.refresh_list()

    def restart_now(self) -> None:
        if not messagebox.askyesno("Restart", "Перезапустить приложение сейчас?"):
            return
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as exc:
            messagebox.showerror("Restart", f"Не удалось перезапустить: {exc}")

    def _update_restart_button(self) -> None:
        if self.manager.restart_required():
            self.restart_button.pack(side=tk.RIGHT, padx=4)
        else:
            self.restart_button.pack_forget()


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


def format_size(size: int) -> str:
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    return f"{value:.1f} {units[unit]}"


def tempfile_directory():
    import tempfile
    from contextlib import contextmanager

    @contextmanager
    def _tempdir():
        path = tempfile.mkdtemp(prefix="ankyx_addon_")
        try:
            yield path
        finally:
            try:
                shutil.rmtree(path)
            except Exception:
                pass

    return _tempdir()
