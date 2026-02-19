from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from llm_providers import LLMRouter
from pro_features import is_pro_user


class ImageCreditService:
    def __init__(self, router: LLMRouter):
        self.router = router

    def _cost(self, pro: bool) -> int:
        return 6 if pro else 10

    def generate_and_store(self, conn: sqlite3.Connection, user_id: str, prompt: str, settings: dict[str, Any]) -> dict[str, Any]:
        pro = is_pro_user(conn, user_id)
        cost = self._cost(pro)
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE;")
        cur.execute("INSERT OR IGNORE INTO credits_balance (user_id, balance) VALUES (?, 0);", (user_id,))
        cur.execute("SELECT balance FROM credits_balance WHERE user_id = ?;", (user_id,))
        bal = int((cur.fetchone() or [0])[0])
        if bal < cost:
            conn.rollback()
            raise ValueError("Недостаточно кредитов для генерации картинки. Пополните баланс.")
        image = self.router.generate_image(prompt, settings)
        path = Path(image["path"])
        if not path.exists():
            conn.rollback()
            raise RuntimeError("Изображение не сохранено, списание отменено.")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cur.execute("SELECT id FROM media_assets WHERE sha256 = ? LIMIT 1;", (sha,))
        row = cur.fetchone()
        if row:
            media_id = int(row[0])
        else:
            cur.execute(
                "INSERT INTO media_assets (type, path, source, sha256, meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                ("image", str(path), "image_generation", sha, json.dumps({"prompt": prompt}, ensure_ascii=False), int(time.time())),
            )
            media_id = int(cur.lastrowid)
        cur.execute("UPDATE credits_balance SET balance = balance - ? WHERE user_id = ?;", (cost, user_id))
        cur.execute(
            "INSERT INTO credits_ledger (user_id, ts, delta, reason, meta) VALUES (?, ?, ?, ?, ?);",
            (user_id, int(time.time()), -cost, "card_image_generation", json.dumps({"prompt": prompt, "pro": pro}, ensure_ascii=False)),
        )
        conn.commit()
        return {"media_id": media_id, "path": str(path), "cost": cost}
