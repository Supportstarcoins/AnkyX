import sqlite3
from pathlib import Path

from image_credit_service import ImageCreditService


class DummyRouter:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def generate_image(self, prompt, settings):
        return {"path": self.image_path, "mime": "image/png"}


def _prepare(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("CREATE TABLE credits_balance(user_id TEXT PRIMARY KEY, balance INTEGER NOT NULL)")
    cur.execute("CREATE TABLE credits_ledger(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts INTEGER, delta INTEGER, reason TEXT, meta TEXT)")
    cur.execute("CREATE TABLE media_assets(id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, path TEXT, source TEXT, sha256 TEXT UNIQUE, meta_json TEXT, created_at INTEGER)")
    cur.execute("CREATE TABLE user_profile(user_id TEXT PRIMARY KEY, is_pro INTEGER, pro_expires INTEGER)")
    cur.execute("INSERT INTO credits_balance(user_id, balance) VALUES ('u1', 20)")
    cur.execute("INSERT INTO user_profile(user_id, is_pro, pro_expires) VALUES ('u1', 0, 0)")
    conn.commit()


def test_atomic_charge_after_image_saved(tmp_path: Path):
    image_path = tmp_path / "ok.png"
    image_path.write_bytes(b"img")
    conn = sqlite3.connect(":memory:")
    _prepare(conn)
    service = ImageCreditService(DummyRouter(str(image_path)))
    result = service.generate_and_store(conn, "u1", "test", {})
    assert result["cost"] == 10
    bal = conn.execute("SELECT balance FROM credits_balance WHERE user_id='u1'").fetchone()[0]
    assert bal == 10
