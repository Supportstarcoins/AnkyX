import sqlite3
import threading
import time

from db_path import connect_to_db

DB_WRITE_LOCK = threading.Lock()


def open_db() -> sqlite3.Connection:
    return connect_to_db(timeout=30)


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
