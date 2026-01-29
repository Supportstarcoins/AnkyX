from __future__ import annotations

import re
import time
from datetime import datetime

from chatbot_models import DraftCard
from db_connect import open_db


class MockAIEngine:
    def __init__(self) -> None:
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_counters (
                user_id TEXT NOT NULL,
                period_key TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, period_key)
            );
            """
        )
        conn.commit()
        conn.close()

    def _plan_key(self, plan: str) -> str:
        if plan in ("pro", "premium"):
            return "pro"
        return "free"

    def _period_key(self, plan_key: str) -> str:
        now = datetime.utcnow()
        if plan_key == "free":
            year, week, _ = now.isocalendar()
            return f"{year}-W{week:02d}"
        return f"{now.year}-{now.month:02d}"

    def _limit_for_plan(self, plan_key: str) -> int:
        return 100 if plan_key == "free" else 10_000

    def _check_and_increment(self, user_id: str, plan: str, count: int) -> None:
        if count <= 0:
            return
        plan_key = self._plan_key(plan)
        period_key = self._period_key(plan_key)
        limit = self._limit_for_plan(plan_key)
        conn = open_db()
        cur = conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE;")
            cur.execute(
                "SELECT count FROM generation_counters WHERE user_id = ? AND period_key = ?;",
                (user_id, period_key),
            )
            row = cur.fetchone()
            current = int(row["count"]) if row else 0
            if current + count > limit:
                conn.rollback()
                raise ValueError("Достигнут лимит генераций для вашего тарифа.")
            if row:
                cur.execute(
                    "UPDATE generation_counters SET count = count + ? WHERE user_id = ? AND period_key = ?;",
                    (count, user_id, period_key),
                )
            else:
                cur.execute(
                    "INSERT INTO generation_counters (user_id, period_key, count) VALUES (?, ?, ?);",
                    (user_id, period_key, count),
                )
            conn.commit()
        finally:
            conn.close()

    def estimate_cost(self, cards_count: int, plan: str) -> int:
        if cards_count <= 0:
            return 0
        plan_key = self._plan_key(plan)
        price = 5 if plan_key == "free" else 2
        return int(cards_count) * price

    def generate_from_text(self, prompt: str, deck_context: dict | None = None) -> list[DraftCard]:
        text = (prompt or "").strip()
        sentences = [
            part.strip()
            for part in re.split(r"[.!?]+", text)
            if part and part.strip()
        ]
        if not sentences:
            sentences = [text or "Примерный термин"]
        if len(sentences) >= 5:
            count = min(8, len(sentences))
        else:
            count = len(sentences)
        cards = []
        for sentence in sentences[:count]:
            cards.append(
                DraftCard(
                    front=sentence,
                    back="Определение/пояснение (заглушка)",
                    tags=["mock", "autogen"],
                    media={"source": "text"},
                    meta={"ts": int(time.time())},
                )
            )
        return cards

    def generate_from_youtube(self, url: str, lang: str, deck_context: dict | None = None) -> list[DraftCard]:
        prompt = f"YouTube: {url}"
        cards = []
        for idx in range(1, 6):
            cards.append(
                DraftCard(
                    front=f"Тема {idx} из {prompt}",
                    back="Определение/пояснение (заглушка)",
                    tags=["mock", "autogen", "youtube"],
                    media={"source": url, "lang": lang},
                    meta={"ts": int(time.time())},
                )
            )
        return cards

    def generate_from_file(self, file_path: str, deck_context: dict | None = None) -> list[DraftCard]:
        filename = file_path.split("/")[-1]
        prompt = f"Файл: {filename}"
        cards = []
        for idx in range(1, 6):
            cards.append(
                DraftCard(
                    front=f"{prompt} — пункт {idx}",
                    back="Определение/пояснение (заглушка)",
                    tags=["mock", "autogen", "file"],
                    media={"source": file_path},
                    meta={"ts": int(time.time())},
                )
            )
        return cards

    def check_and_record_generation(self, user_id: str, plan: str, cards_count: int) -> None:
        self._check_and_increment(user_id, plan, cards_count)
