from __future__ import annotations

import sqlite3
import time


def is_pro_user(conn: sqlite3.Connection, user_id: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT is_pro, pro_expires FROM user_profile WHERE user_id = ? LIMIT 1;", (user_id,))
    row = cur.fetchone()
    if not row:
        return False
    now = int(time.time())
    return int(row[0] or 0) == 1 and int(row[1] or 0) > now


def ensure_chatbot_access(conn: sqlite3.Connection, user_id: str) -> None:
    if not is_pro_user(conn, user_id):
        raise PermissionError("Режим чат-бот доступен только для PRO-пользователей.")
