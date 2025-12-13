import os
import shutil
import sqlite3
import tempfile
from typing import Optional

DB_FILENAME = "anki.db"

_OLD_DB_PATHS = [
    os.path.join("C:\\", "AnkyX-main", "data", DB_FILENAME),
    os.path.join("C:\\", "AnkyX-main", DB_FILENAME),
]


def get_db_path() -> str:
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    db_dir = os.path.join(base, "AnkyX", "data")
    os.makedirs(db_dir, exist_ok=True)

    target_path = os.path.abspath(os.path.join(db_dir, DB_FILENAME))
    _migrate_legacy_db(target_path)
    return target_path


def _migrate_legacy_db(target_path: str) -> None:
    if os.path.exists(target_path):
        return

    for old_path in _OLD_DB_PATHS:
        if os.path.exists(old_path):
            shutil.copy2(old_path, target_path)
            break


def _ensure_writable_directory(db_dir: str) -> None:
    if not os.path.isdir(db_dir):
        raise FileNotFoundError(f"Database directory does not exist: {db_dir}")

    try:
        fd, temp_path = tempfile.mkstemp(prefix=".__dbtest__", dir=db_dir)
        os.close(fd)
        os.remove(temp_path)
    except Exception as e:  # pragma: no cover - диагностика среда
        raise PermissionError(f"Cannot write to database directory {db_dir}: {e}") from e


def connect_to_db(timeout: Optional[int] = 5) -> sqlite3.Connection:
    db_path = get_db_path()
    db_dir = os.path.dirname(db_path)
    try:
        _ensure_writable_directory(db_dir)
        conn = sqlite3.connect(db_path, timeout=timeout) if timeout is not None else sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        cwd = os.getcwd()
        message = f"Не удалось открыть БД: {db_path}\nCWD: {cwd}\nОшибка: {repr(e)}"
        try:
            from tkinter import messagebox

            messagebox.showerror("Ошибка БД", message)
        except Exception:
            print(message)
        raise
