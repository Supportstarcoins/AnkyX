import json
import sqlite3
import time
from typing import Callable, Dict, List, Optional

from db_connect import commit_with_retry, open_db


class CreditsService:
    """Единый сервис работы с балансом кредитов и журналом операций."""

    def __init__(self, db_factory: Callable[[], sqlite3.Connection] = open_db):
        self._db_factory = db_factory
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        conn = self._db_factory()
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
                delta INTEGER NOT NULL,
                reason TEXT,
                meta_json TEXT
            );
            """
        )
        conn.commit()
        conn.close()

    def _ensure_balance_row(self, conn: sqlite3.Connection, user_id: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO credits_balance (user_id, balance) VALUES (?, 0);",
            (user_id,),
        )

    def get_balance(self, user_id: str) -> int:
        conn = self._db_factory()
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

    def add_credits(
        self,
        user_id: str,
        amount: int,
        reason: str,
        meta: Optional[Dict] = None,
    ) -> None:
        if amount <= 0:
            return
        conn = self._db_factory()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        ts = int(time.time())

        def _op():
            cur = conn.cursor()
            self._ensure_balance_row(conn, user_id)
            cur.execute(
                "UPDATE credits_balance SET balance = balance + ? WHERE user_id = ?;",
                (amount, user_id),
            )
            cur.execute(
                """
                INSERT INTO credits_ledger (user_id, ts, delta, reason, meta_json)
                VALUES (?, ?, ?, ?, ?);
                """,
                (user_id, ts, amount, reason, meta_json),
            )

        commit_with_retry(conn, _op)
        conn.close()

    def spend_credits(
        self,
        user_id: str,
        amount: int,
        reason: str,
        meta: Optional[Dict] = None,
    ) -> bool:
        if amount <= 0:
            return True
        conn = self._db_factory()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        ts = int(time.time())

        def _op():
            cur = conn.cursor()
            self._ensure_balance_row(conn, user_id)
            cur.execute(
                "SELECT balance FROM credits_balance WHERE user_id = ?;",
                (user_id,),
            )
            row = cur.fetchone()
            balance = int(row[0]) if row else 0
            if balance < amount:
                return False
            cur.execute(
                "UPDATE credits_balance SET balance = balance - ? WHERE user_id = ?;",
                (amount, user_id),
            )
            cur.execute(
                """
                INSERT INTO credits_ledger (user_id, ts, delta, reason, meta_json)
                VALUES (?, ?, ?, ?, ?);
                """,
                (user_id, ts, -abs(amount), reason, meta_json),
            )
            return True

        result = commit_with_retry(conn, _op)
        conn.close()
        return bool(result)

    def get_ledger(self, user_id: str, limit: int = 200) -> List[Dict]:
        conn = self._db_factory()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts, delta, reason, meta_json
            FROM credits_ledger
            WHERE user_id = ?
            ORDER BY ts DESC
            LIMIT ?;
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
        conn.close()
        ledger: List[Dict] = []
        for row in rows:
            meta = {}
            if row["meta_json"]:
                try:
                    meta = json.loads(row["meta_json"])
                except Exception:
                    meta = {"raw": row["meta_json"]}
            ledger.append(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "delta": row["delta"],
                    "reason": row["reason"],
                    "meta": meta,
                }
            )
        return ledger
