from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict


def create_filtered_session(conn: sqlite3.Connection, filter_payload: dict, card_ids: list[int], session_type: str = "filtered_snapshot") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO study_sessions (type, filter_json, created_at) VALUES (?, ?, ?);",
        (session_type, json.dumps(filter_payload, ensure_ascii=False), int(time.time())),
    )
    session_id = int(cur.lastrowid)
    for idx, card_id in enumerate(card_ids, start=1):
        cur.execute(
            "INSERT INTO session_cards (session_id, card_id, ordinal) VALUES (?, ?, ?);",
            (session_id, card_id, idx),
        )
    conn.commit()
    return session_id


def create_lagging_topics_session(conn: sqlite3.Connection, n_topics: int = 3, cards_per_topic: int = 20) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, COALESCE(c.tags,''), COALESCE(c.reps,0), COALESCE(c.lapses,0), COALESCE(c.due,0)
        FROM cards c;
        """
    )
    now = int(time.time())
    topic_scores: dict[str, float] = defaultdict(float)
    topic_cards: dict[str, list[int]] = defaultdict(list)
    for card_id, tags, reps, lapses, due in cur.fetchall():
        tag = (str(tags).split(",")[0].strip() if tags else "untagged")
        success_rate = 1.0 if int(reps) == 0 else max(0.0, 1.0 - (int(lapses) / max(int(reps), 1)))
        overdue = 1.0 if int(due or 0) < now else 0.0
        score = (1.0 - success_rate) + (int(lapses) * 0.05) + overdue
        topic_scores[tag] += score
        topic_cards[tag].append(int(card_id))
    worst = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[: max(n_topics, 1)]
    selected: list[int] = []
    for topic, _ in worst:
        selected.extend(topic_cards[topic][:cards_per_topic])
    payload = {"mode": "lagging_topics", "topics": [t for t, _ in worst], "snapshot": True}
    return create_filtered_session(conn, payload, selected, session_type="lagging_topics")
