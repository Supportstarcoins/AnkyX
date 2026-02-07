import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from config import settings


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)


def init_db() -> None:
    _ensure_dir(settings.db_path)
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key TEXT UNIQUE NOT NULL,
                plan TEXT NOT NULL,
                credits_balance INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                credits_spent INTEGER NOT NULL,
                request_chars INTEGER NOT NULL,
                response_chars INTEGER NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_user_by_key(api_key: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE api_key = ?",
            (api_key,),
        ).fetchone()
        return row


def create_user(api_key: str, plan: str, credits: int, is_active: bool = True) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (api_key, plan, credits_balance, created_at, is_active)
            VALUES (?, ?, ?, ?, ?)
            """,
            (api_key, plan, credits, now, 1 if is_active else 0),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_credits(user_id: int, delta: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET credits_balance = credits_balance + ? WHERE id = ?",
            (delta, user_id),
        )
        conn.commit()


def set_plan(user_id: int, plan: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET plan = ? WHERE id = ?",
            (plan, user_id),
        )
        conn.commit()


def set_active(user_id: int, is_active: bool) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if is_active else 0, user_id),
        )
        conn.commit()


def get_usage_count(user_id: int, since: datetime) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM usage
            WHERE user_id = ? AND ts >= ?
            """,
            (user_id, since.isoformat()),
        ).fetchone()
        return int(row["count"]) if row else 0


def log_usage(
    user_id: int,
    credits_spent: int,
    request_chars: int,
    response_chars: int,
    status: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO usage (user_id, ts, credits_spent, request_chars, response_chars, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, now, credits_spent, request_chars, response_chars, status),
        )
        conn.commit()


def reserve_credits(user_id: int, cost: int) -> Tuple[bool, Optional[int]]:
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT credits_balance FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            conn.execute("ROLLBACK")
            return False, None
        balance = int(row["credits_balance"])
        if balance < cost:
            conn.execute("ROLLBACK")
            return False, balance
        new_balance = balance - cost
        conn.execute(
            "UPDATE users SET credits_balance = ? WHERE id = ?",
            (new_balance, user_id),
        )
        conn.commit()
        return True, new_balance


def get_plan_window(plan: str) -> Tuple[int, datetime]:
    now = datetime.now(timezone.utc)
    if plan.lower() == "pro":
        return 10_000, now - timedelta(days=30)
    return 100, now - timedelta(days=7)
