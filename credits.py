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

    def _get_table_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        return {row["name"] for row in cur.fetchall()}

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
    ) -> int:
        if amount <= 0:
            raise ValueError("amount must be > 0")
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
            columns = self._get_table_columns(conn, "credits_ledger")
            meta_column = "meta" if "meta" in columns else "meta_json"
            cur.execute(
                """
                INSERT INTO credits_ledger (user_id, ts, delta, reason, {meta_column})
                VALUES (?, ?, ?, ?, ?);
                """.format(meta_column=meta_column),
                (user_id, ts, amount, reason, meta_json),
            )
            cur.execute(
                "SELECT balance FROM credits_balance WHERE user_id = ? LIMIT 1;",
                (user_id,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

        balance = commit_with_retry(conn, _op)
        conn.close()
        return int(balance or 0)

    def spend_credits(
        self,
        user_id: str,
        amount: int,
        reason: str,
        meta: Optional[Dict] = None,
    ) -> int:
        if amount <= 0:
            raise ValueError("amount must be > 0")
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
                raise ValueError("insufficient credits")
            cur.execute(
                "UPDATE credits_balance SET balance = balance - ? WHERE user_id = ?;",
                (amount, user_id),
            )
            columns = self._get_table_columns(conn, "credits_ledger")
            meta_column = "meta" if "meta" in columns else "meta_json"
            cur.execute(
                """
                INSERT INTO credits_ledger (user_id, ts, delta, reason, {meta_column})
                VALUES (?, ?, ?, ?, ?);
                """.format(meta_column=meta_column),
                (user_id, ts, -abs(amount), reason, meta_json),
            )
            return balance - amount

        balance = commit_with_retry(conn, _op)
        conn.close()
        return int(balance or 0)

    def get_ledger(self, user_id: str, limit: int = 200) -> List[Dict]:
        conn = self._db_factory()
        cur = conn.cursor()
        columns = self._get_table_columns(conn, "credits_ledger")
        select_fields = ["id", "ts", "reason"]
        if "delta" in columns:
            select_fields.append("delta")
        if "plus" in columns:
            select_fields.append("plus")
        if "minus" in columns:
            select_fields.append("minus")
        if "meta" in columns:
            select_fields.append("meta")
        elif "meta_json" in columns:
            select_fields.append("meta_json")
        select_sql = ", ".join(select_fields)
        cur.execute(
            f"""
            SELECT {select_sql}
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
            if "meta" in row.keys():
                meta_value = row["meta"]
            elif "meta_json" in row.keys():
                meta_value = row["meta_json"]
            else:
                meta_value = None
            if meta_value:
                try:
                    meta = json.loads(meta_value)
                except Exception:
                    meta = {"raw": meta_value}
            delta_val = row["delta"] if "delta" in row.keys() else None
            if delta_val is None:
                plus_val = row["plus"] if "plus" in row.keys() else 0
                minus_val = row["minus"] if "minus" in row.keys() else 0
                delta_val = int(plus_val or 0) - int(minus_val or 0)
            reason = self._normalize_reason(row["reason"], delta_val, meta)
            ledger.append(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "delta": delta_val,
                    "reason": reason,
                    "meta": meta,
                }
            )
        return ledger

    def _normalize_reason(self, reason: Optional[str], delta: int, meta: Dict) -> str:
        reason_val = (reason or "").strip()
        if not reason_val:
            return "Операция"
        reason_key = reason_val.lower()
        mapped = {
            "ocr_image": self._format_ocr_reason,
            "image_id_import": self._format_image_import_reason,
            "wikimedia_bundle": self._format_wikimedia_reason,
            "card_image_generation": self._format_ai_image_reason,
            "ai image generation": self._format_ai_image_reason,
            "ai video generation": self._format_ai_video_reason,
        }
        if reason_key in mapped:
            return mapped[reason_key](delta, meta)
        if (
            "simplify ocr ui" in reason_key
            or reason_key in {"debug", "title"}
            or reason_key.startswith("debug")
        ):
            return "Операция"
        return reason_val

    def _format_ocr_reason(self, delta: int, meta: Dict) -> str:
        mode = str(meta.get("ocr_mode") or "").lower()
        pages = int(meta.get("pages") or 1)
        cost = abs(delta)
        if mode == "pro":
            return f"OCR PRO 👑: {pages} стр ({cost} ⚡)"
        return f"OCR: {pages} стр ({cost} ⚡)"

    def _format_image_import_reason(self, delta: int, meta: Dict) -> str:
        files = int(meta.get("files") or 0) or int(meta.get("items") or 0) or 1
        cost = abs(delta)
        return f"Импорт изображений: {files} шт ({cost} ⚡)"

    def _format_wikimedia_reason(self, delta: int, meta: Dict) -> str:
        bundle = int(meta.get("bundle") or 0)
        cost = abs(delta)
        if bundle:
            return f"Wikimedia: {bundle} стр ({cost} ⚡)"
        return f"Wikimedia: {cost} ⚡"

    def _format_ai_image_reason(self, delta: int, meta: Dict) -> str:
        count = int(meta.get("images") or 1)
        cost = abs(delta)
        if cost == 0:
            return f"AI-картинки: {count} шт (включено)"
        return f"AI-картинки: +{count} шт (доплата {cost} ⚡)"

    def _format_ai_video_reason(self, delta: int, meta: Dict) -> str:
        count = int(meta.get("videos") or 1)
        cost = abs(delta)
        return f"AI-видео: +{count} шт (доплата {cost} ⚡)"
