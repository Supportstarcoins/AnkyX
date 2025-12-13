import sqlite3
import threading
import time

from db_path import get_db_path

DB_WRITE_LOCK = threading.Lock()


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(get_db_path(), timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=5000;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def commit_with_retry(conn: sqlite3.Connection, operation):
    for attempt in range(6):
        try:
            with DB_WRITE_LOCK:
                result = operation()
                conn.commit()
            return result
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                conn.rollback()
                time.sleep(0.2 * (attempt + 1))
                continue
            conn.rollback()
            raise
