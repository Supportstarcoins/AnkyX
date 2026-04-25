from __future__ import annotations

import sqlite3
import time


class AISRSAdapter:
    def apply_ai_grade_to_card(self, card_id, grade_result, db_connection_or_app):
        conn = self._resolve_connection(db_connection_or_app)
        own_conn = not isinstance(db_connection_or_app, sqlite3.Connection)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(cards)")
            columns = {row[1] for row in cur.fetchall()}
            cur.execute("SELECT interval, reps, lapses, phase, leitner_level FROM cards WHERE id = ?", (card_id,))
            row = cur.fetchone()
            if not row:
                return None
            interval = int(row[0] or 1)
            reps = int(row[1] or 0)
            lapses = int(row[2] or 0)
            phase = int(row[3] or 1)
            level = int(row[4] or 1)
            score = float((grade_result or {}).get("score") or 0.0)
            grade = (grade_result or {}).get("grade") or "wrong"
            t_quality = (grade_result or {}).get("answer_time_quality") or "normal"

            if grade == "wrong" or score < 0.5:
                interval = 1
                phase = 1
                level = 1
                lapses += 1
            elif score >= 0.9 and t_quality in {"fast", "normal"}:
                interval = max(2, int(interval * 1.8))
                phase = min(10, phase + 1)
                level = min(10, level + 1)
            elif score >= 0.75:
                interval = max(1, int(interval * 1.3))
                phase = min(10, phase + 1)
                level = min(10, level + 1)
            else:
                interval = max(1, min(interval, 2))

            reps += 1
            due_ts = int(time.time()) + interval * 86400

            sets, params = [], []
            for col, val in (("interval", interval), ("reps", reps), ("lapses", lapses), ("phase", phase), ("leitner_level", level), ("due", due_ts)):
                if col in columns:
                    sets.append(f"{col} = ?")
                    params.append(val)
            if "last_ai_score" in columns:
                sets.append("last_ai_score = ?")
                params.append(score)
            if "last_mistake_type" in columns:
                sets.append("last_mistake_type = ?")
                params.append((grade_result or {}).get("mistake_type"))
            if "last_answer_quality" in columns:
                sets.append("last_answer_quality = ?")
                params.append((grade_result or {}).get("grade"))
            if "answer_time_ms" in columns:
                sets.append("answer_time_ms = ?")
                params.append(int((grade_result or {}).get("answer_time_ms") or 0))
            if sets:
                params.append(card_id)
                cur.execute(f"UPDATE cards SET {', '.join(sets)} WHERE id = ?", tuple(params))
                conn.commit()
            return {"interval": interval, "phase": phase, "leitner_level": level}
        finally:
            if own_conn:
                conn.close()

    def _resolve_connection(self, db_connection_or_app):
        if isinstance(db_connection_or_app, sqlite3.Connection):
            return db_connection_or_app
        conn = getattr(db_connection_or_app, "conn", None)
        if isinstance(conn, sqlite3.Connection):
            return conn
        from db_path import get_db_path

        return sqlite3.connect(get_db_path())
