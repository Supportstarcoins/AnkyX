from __future__ import annotations

import json
import time

from db_connect import open_db


class CreditManager:
    def __init__(self) -> None:
        self._ensure_tables()

    def _get_table_columns(self, conn, table: str) -> set[str]:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        return {row["name"] for row in cur.fetchall()}

    def _ensure_tables(self) -> None:
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS credits_balance (
                user_id TEXT PRIMARY KEY,
                balance INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS credits_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                ts INTEGER NOT NULL,
                delta INTEGER NOT NULL CHECK (delta != 0),
                reason TEXT,
                meta TEXT
            );
            """
        )
        columns = self._get_table_columns(conn, "credits_ledger")
        if "meta" not in columns:
            cur.execute("ALTER TABLE credits_ledger ADD COLUMN meta TEXT;")
        if "delta" not in columns:
            cur.execute("ALTER TABLE credits_ledger ADD COLUMN delta INTEGER;")
        conn.commit()
        conn.close()

    def _ensure_balance_row(self, conn, user_id: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO credits_balance (user_id, balance) VALUES (?, 0);",
            (user_id,),
        )

    def get_balance(self, user_id: str) -> int:
        conn = open_db()
        cur = conn.cursor()
        self._ensure_balance_row(conn, user_id)
        cur.execute(
            "SELECT balance FROM credits_balance WHERE user_id = ? LIMIT 1;",
            (user_id,),
        )
        row = cur.fetchone()
        conn.commit()
        conn.close()
        return int(row[0]) if row else 0

    def can_afford(self, user_id: str, cost: int) -> bool:
        if cost <= 0:
            return True
        return self.get_balance(user_id) >= cost

    def charge(self, user_id: str, amount: int, reason: str, meta: dict | None = None) -> tuple[bool, int]:
        if amount <= 0:
            return True, self.get_balance(user_id)
        conn = open_db()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        ts = int(time.time())
        try:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE;")
            self._ensure_balance_row(conn, user_id)
            cur.execute(
                "SELECT balance FROM credits_balance WHERE user_id = ?;",
                (user_id,),
            )
            row = cur.fetchone()
            balance = int(row[0]) if row else 0
            if balance < amount:
                conn.rollback()
                return False, balance
            cur.execute(
                "UPDATE credits_balance SET balance = balance - ? WHERE user_id = ?;",
                (amount, user_id),
            )
            cur.execute(
                "INSERT INTO credits_ledger (user_id, ts, delta, reason, meta) VALUES (?, ?, ?, ?, ?);",
                (user_id, ts, -abs(amount), reason, meta_json),
            )
            cur.execute(
                "SELECT balance FROM credits_balance WHERE user_id = ? LIMIT 1;",
                (user_id,),
            )
            row = cur.fetchone()
            conn.commit()
            return True, int(row[0]) if row else 0
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
