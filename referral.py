import sqlite3
import time
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from credits import CreditsService
from db_connect import open_db

REF_BASE_URL = "https://x-flash.app/ref"


class ReferralService:
    """Простая реферальная система с кодами и начислениями."""

    def __init__(
        self,
        db_factory: Callable[[], sqlite3.Connection] = open_db,
        credits_service: Optional[CreditsService] = None,
    ):
        self._db_factory = db_factory
        self._credits = credits_service or CreditsService(db_factory)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        conn = self._db_factory()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS referral_users (
                user_id TEXT PRIMARY KEY,
                ref_code TEXT UNIQUE,
                referrer_id TEXT,
                activated INTEGER DEFAULT 0
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS referral_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                referrer_id TEXT NOT NULL,
                referee_id TEXT,
                ts INTEGER NOT NULL,
                delta INTEGER NOT NULL,
                reason TEXT
            );
            """
        )
        conn.commit()
        conn.close()

    def _generate_ref_code(self) -> str:
        return uuid4().hex[:10]

    def _ensure_user_row(self, user_id: str) -> str:
        conn = self._db_factory()
        cur = conn.cursor()
        cur.execute("SELECT ref_code FROM referral_users WHERE user_id = ?;", (user_id,))
        row = cur.fetchone()
        if row and row["ref_code"]:
            conn.close()
            return row["ref_code"]

        ref_code = self._generate_ref_code()
        cur.execute(
            """
            INSERT OR IGNORE INTO referral_users (user_id, ref_code, activated)
            VALUES (?, ?, 0);
            """,
            (user_id, ref_code),
        )
        conn.commit()
        conn.close()
        return ref_code

    def get_ref_code(self, user_id: str) -> str:
        return self._ensure_user_row(user_id)

    def get_ref_link(self, user_id: str) -> str:
        code = self.get_ref_code(user_id)
        return f"{REF_BASE_URL}?code={code}"

    def register_referral(self, ref_code: str, new_user_id: str) -> None:
        if not ref_code or not new_user_id:
            return
        conn = self._db_factory()
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id FROM referral_users WHERE ref_code = ? LIMIT 1;", (ref_code,)
        )
        row = cur.fetchone()
        referrer_id = row["user_id"] if row else None
        cur.execute(
            """
            INSERT OR IGNORE INTO referral_users (user_id, ref_code, referrer_id, activated)
            VALUES (?, ?, ?, 0);
            """,
            (new_user_id, self._generate_ref_code(), referrer_id),
        )
        conn.commit()
        conn.close()

    def mark_activation(self, new_user_id: str) -> None:
        conn = self._db_factory()
        conn.execute(
            "UPDATE referral_users SET activated = 1 WHERE user_id = ?;",
            (new_user_id,),
        )
        conn.commit()
        conn.close()

    def award_referral_bonus(
        self,
        referrer_id: str,
        referee_id: str,
        amount: int,
        reason: str,
    ) -> None:
        if amount <= 0:
            return
        ts = int(time.time())
        conn = self._db_factory()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO referral_ledger (referrer_id, referee_id, ts, delta, reason)
            VALUES (?, ?, ?, ?, ?);
            """,
            (referrer_id, referee_id, ts, amount, reason),
        )
        conn.commit()
        conn.close()
        self._credits.add_credits(
            referrer_id,
            amount,
            reason=reason,
            meta={"referee_id": referee_id, "source": "referral_bonus"},
        )

    def get_summary(self, user_id: str) -> Dict:
        conn = self._db_factory()
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM referral_users WHERE referrer_id = ?;",
            (user_id,),
        )
        invited = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM referral_users WHERE referrer_id = ? AND activated = 1;",
            (user_id,),
        )
        activated = cur.fetchone()[0]
        cur.execute(
            "SELECT COALESCE(SUM(delta), 0) FROM referral_ledger WHERE referrer_id = ?;",
            (user_id,),
        )
        earned = cur.fetchone()[0] or 0
        conn.close()
        return {
            "invited": invited,
            "activated": activated,
            "earned": earned,
        }

    def list_rewards(self, user_id: str, limit: int = 50) -> List[Dict]:
        conn = self._db_factory()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts, delta, referee_id, reason
            FROM referral_ledger
            WHERE referrer_id = ?
            ORDER BY ts DESC
            LIMIT ?;
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "ts": row["ts"],
                "delta": row["delta"],
                "referee_id": row["referee_id"],
                "reason": row["reason"],
            }
            for row in rows
        ]
