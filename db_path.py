import os
import sqlite3
import tempfile
from typing import Optional

DB_FILENAME = "anki.db"
YOUR_DB_FILENAME = "YOUR_DB.db"


def _base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_paths(base_dir: str) -> list[str]:
    return [
        os.path.join(base_dir, DB_FILENAME),
        os.path.join(base_dir, YOUR_DB_FILENAME),
        os.path.join(base_dir, "data", DB_FILENAME),
    ]


def get_db_path() -> str:
    """Возвращает абсолютный путь к SQLite БД, стараясь использовать уже существующую."""

    base_dir = _base_dir()
    candidates = _candidate_paths(base_dir)

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    fallback = candidates[-1]
    os.makedirs(os.path.dirname(fallback), exist_ok=True)
    return os.path.abspath(fallback)


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
