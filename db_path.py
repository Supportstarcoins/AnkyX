import os
import shutil
import sqlite3
import tempfile
from typing import Optional

DB_FILENAME = "anki.db"


def _legacy_db_path(base_dir: str) -> str:
    return os.path.join(base_dir, DB_FILENAME)


def _data_directory(base_dir: str) -> str:
    return os.path.join(base_dir, "data")


def _migrate_legacy_database(new_path: str, legacy_path: str) -> None:
    if os.path.exists(new_path) or not os.path.exists(legacy_path):
        return

    try:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(legacy_path, new_path)
    except Exception:
        try:
            shutil.copy2(legacy_path, new_path)
        finally:
            # Если копирование удалось, оставляем исходный файл на месте для надёжности.
            pass


def get_db_path() -> str:
    """Возвращает абсолютный путь к SQLite БД и гарантирует её каталог."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = _data_directory(base_dir)
    os.makedirs(db_dir, exist_ok=True)

    db_path = os.path.join(db_dir, DB_FILENAME)
    _migrate_legacy_database(db_path, _legacy_db_path(base_dir))

    return os.path.abspath(db_path)


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
    """Открывает соединение с БД, проверяя доступность каталога и выдавая диагностику."""

    db_path = get_db_path()
    db_dir = os.path.dirname(db_path)
    try:
        _ensure_writable_directory(db_dir)
        conn = sqlite3.connect(db_path, timeout=timeout) if timeout is not None else sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        cwd = os.getcwd()
        message = f"Не удалось открыть БД: {db_path}\nCWD: {cwd}\nОшибка: {e}"
        try:
            from tkinter import messagebox

            messagebox.showerror("Ошибка БД", message)
        except Exception:
            # Если Tkinter не готов, просто печатаем сообщение.
            print(message)
        raise
