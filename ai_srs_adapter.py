from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta


LEITNER_INTERVALS_DAYS = {
    1: 0,
    2: 1,
    3: 3,
    4: 7,
    5: 14,
    6: 30,
}

SPEED_MULTIPLIERS = {
    "fast": 1.25,
    "normal": 1.0,
    "slow": 0.75,
    "too_slow": 0.5,
}


class AISRSAdapter:
    def apply_ai_grade_to_card(self, card_id, grade_result, db_connection_or_app):
        if card_id is None:
            return None
        conn = self._resolve_connection(db_connection_or_app)
        own_conn = not isinstance(db_connection_or_app, sqlite3.Connection)
        try:
            cur = conn.cursor()
            table_info = self._load_cards_table_info(cur)
            columns = set(table_info.keys())
            select_columns = [name for name in ("interval", "reps", "lapses", "phase", "leitner_level", "due", "ease", "state") if name in columns]
            if not select_columns:
                select_columns = ["id"]
            cur.execute(f"SELECT {', '.join(select_columns)} FROM cards WHERE id = ?", (int(card_id),))
            row = cur.fetchone()
            if not row:
                return None
            current_data = dict(zip(select_columns, row))

            interval = float(current_data.get("interval") or 1)
            reps = int(current_data.get("reps") or 0)
            lapses = int(current_data.get("lapses") or 0)
            phase = int(current_data.get("phase") or 0)
            leitner_level = int(current_data.get("leitner_level") or 0)
            score = float((grade_result or {}).get("score") or 0.0)
            grade = (grade_result or {}).get("grade") or "wrong"
            t_quality = (grade_result or {}).get("answer_time_quality") or "normal"
            answer_time_ms = int((grade_result or {}).get("answer_time_ms") or 0)
            current_level = leitner_level or phase or 1

            normalized_grade = self._normalize_grade(grade, score, t_quality)
            speed_multiplier = SPEED_MULTIPLIERS.get(t_quality, 1.0)
            score_multiplier = self._score_multiplier(score)

            if normalized_grade in {"correct", "slow_correct"}:
                new_level = min(current_level + 1, 6)
                base_days = float(LEITNER_INTERVALS_DAYS.get(new_level, 1))
                # Для slow_correct интервал увеличиваем осторожнее.
                if normalized_grade == "slow_correct":
                    speed_multiplier = min(speed_multiplier, 0.75)
                final_days = base_days * speed_multiplier * score_multiplier
                if new_level == 2:
                    final_days = max(1.0, final_days)
                interval = max(0.02, final_days)
                due_dt = datetime.now() + timedelta(days=interval)
                phase = new_level
                leitner_level = new_level
            elif normalized_grade == "partial":
                new_level = current_level
                base_days = float(LEITNER_INTERVALS_DAYS.get(new_level, 0))
                final_days = base_days * speed_multiplier * score_multiplier
                if new_level == 2:
                    final_days = max(1.0, final_days)
                interval = max(0.02, final_days if final_days > 0 else 0.02)
                due_dt = datetime.now() + timedelta(minutes=30 if new_level <= 1 else 60)
                phase = new_level
                leitner_level = new_level
            else:
                # wrong / uncertain
                new_level = 1
                interval = 0.0
                due_dt = datetime.now() + timedelta(minutes=10)
                phase = 1
                leitner_level = 1
                lapses += 1
            reps += 1
            due_value = self._format_due_value(due_dt, table_info.get("due", ""))

            sets, params = [], []
            for col, val in (
                ("interval", round(float(interval), 4)),
                ("reps", reps),
                ("lapses", lapses),
                ("phase", phase),
                ("leitner_level", leitner_level),
                ("due", due_value),
            ):
                if col in columns:
                    sets.append(f"{col} = ?")
                    params.append(val)
            if "state" in columns:
                sets.append("state = ?")
                params.append("review")
            if "overview_added" in columns:
                sets.append("overview_added = ?")
                params.append(0)
            if "ease" in columns:
                current_ease = float(current_data.get("ease") or 2.5)
                if normalized_grade in {"correct", "slow_correct"}:
                    ease = min(3.0, current_ease + 0.03)
                elif normalized_grade == "partial":
                    ease = max(1.3, current_ease - 0.02)
                else:
                    ease = max(1.3, current_ease - 0.2)
                sets.append("ease = ?")
                params.append(round(ease, 4))
            if "last_ai_score" in columns:
                sets.append("last_ai_score = ?")
                params.append(score)
            if "last_mistake_type" in columns:
                sets.append("last_mistake_type = ?")
                params.append((grade_result or {}).get("mistake_type"))
            if "last_answer_quality" in columns:
                sets.append("last_answer_quality = ?")
                params.append(normalized_grade)
            if "answer_time_ms" in columns:
                sets.append("answer_time_ms = ?")
                params.append(answer_time_ms)
            if "ai_feedback_json" in columns:
                feedback_payload = {
                    "grade": normalized_grade,
                    "score": score,
                    "answer_time_quality": t_quality,
                    "answer_time_ms": answer_time_ms,
                    "updated_at_ts": int(time.time()),
                }
                sets.append("ai_feedback_json = ?")
                params.append(json.dumps(feedback_payload, ensure_ascii=False))
            if sets:
                params.append(card_id)
                cur.execute(f"UPDATE cards SET {', '.join(sets)} WHERE id = ?", tuple(params))
                conn.commit()
            return {
                "interval": round(float(interval), 4),
                "phase": phase,
                "leitner_level": leitner_level,
                "new_level": leitner_level,
                "due": due_value,
                "due_human": due_dt.strftime("%Y-%m-%d %H:%M"),
                "grade": normalized_grade,
                "applied": bool(sets),
            }
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

    def _normalize_grade(self, grade: str, score: float, answer_time_quality: str) -> str:
        grade = str(grade or "").strip().lower()
        if grade == "correct" and answer_time_quality in {"slow", "too_slow"} and score >= 0.85:
            return "slow_correct"
        if grade in {"correct", "slow_correct"} and score >= 0.85:
            return grade
        if grade in {"wrong", "uncertain"} or score < 0.5:
            return "wrong" if grade == "wrong" or score < 0.5 else "uncertain"
        return "partial"

    def _score_multiplier(self, score: float) -> float:
        if score >= 0.95:
            return 1.15
        if score >= 0.85:
            return 1.0
        if score >= 0.70:
            return 0.7
        return 0.4

    def _load_cards_table_info(self, cursor: sqlite3.Cursor) -> dict[str, str]:
        cursor.execute("PRAGMA table_info(cards)")
        return {str(row[1]): str(row[2] or "") for row in cursor.fetchall()}

    def _format_due_value(self, due_dt: datetime, due_declared_type: str):
        declared = (due_declared_type or "").strip().upper()
        if "CHAR" in declared or "TEXT" in declared:
            return due_dt.isoformat(sep=" ", timespec="seconds")
        return int(due_dt.timestamp())
