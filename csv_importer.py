import csv
import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Iterable

from db_connect import DB_WRITE_LOCK, commit_with_retry

from db_migrations import ensure_schema_for_import


MEDIA_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def detect_encoding(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            f.read(1024)
        return "utf-8-sig"
    except UnicodeDecodeError:
        pass

    try:
        with open(path, "r", encoding="cp1251") as f:
            f.read(1024)
        return "cp1251"
    except UnicodeDecodeError:
        return "utf-8"


def normalize_tags(raw: str | None) -> str:
    if not raw:
        return ""
    cleaned = raw.replace(",", " ")
    parts = [part.strip() for part in cleaned.split() if part.strip()]
    return " ".join(parts)


_SCHEMA_CHECKED: set[int] = set()


def _safe_get(row: Any, key: Any) -> str:
    if isinstance(row, dict):
        return str(row.get(key, "") or "").strip()
    if isinstance(key, int):
        try:
            return str(row[key]).strip()
        except Exception:
            return ""
    return ""


def map_row_to_fields(row: Any, mapping_mode: dict[str, Any]) -> dict[str, str]:
    variant = mapping_mode.get("variant", "auto")
    manual_map = mapping_mode.get("manual_map") or {}

    if manual_map:
        return {target: _safe_get(row, source) for target, source in manual_map.items()}

    lower_keys: set[str] = set()
    if isinstance(row, dict):
        lower_keys = {str(k).lower() for k in row.keys()}

    def pick(*candidates: Iterable[str]) -> str:
        for cand in candidates:
            if cand in lower_keys:
                return _safe_get(row, cand)
        return ""

    if variant == "front_back" or ({"front", "back"} <= lower_keys):
        return {"front": pick("front"), "back": pick("back")}

    if variant == "word_translation" or ("word" in lower_keys and "translation" in lower_keys):
        return {
            "word": pick("word", "term"),
            "translation": pick("translation", "meaning"),
            "example": pick("example", "sentence"),
            "notes": pick("notes"),
        }

    if variant == "word_translation_example" or (
        {"word", "translation", "example"} <= lower_keys
    ):
        return {
            "word": pick("word"),
            "translation": pick("translation"),
            "example": pick("example"),
            "notes": pick("notes"),
        }

    return {
        "front": pick("front", "word"),
        "back": pick("back", "translation"),
        "notes": pick("notes"),
    }


def _ensure_basic_note_type(cur: sqlite3.Cursor) -> int:
    cur.execute("SELECT id FROM note_types WHERE name = 'Basic' LIMIT 1;")
    row = cur.fetchone()
    if row:
        return row["id"]

    fields = ["word", "translation", "example", "notes", "image"]
    templates = [
        {
            "name": "Word→Translation",
            "front": "{word}",
            "back": "{translation}\n\n{example}\n\n{notes}",
            "requires_image": False,
        }
    ]
    cur.execute(
        "INSERT OR IGNORE INTO note_types (name, fields_json, card_templates_json) VALUES (?, ?, ?);",
        ("Basic", json.dumps(fields, ensure_ascii=False), json.dumps(templates, ensure_ascii=False)),
    )
    cur.execute("SELECT id FROM note_types WHERE name = 'Basic' LIMIT 1;")
    return cur.fetchone()["id"]


def render_card_faces(fields: dict[str, Any]) -> tuple[str, str]:
    word = str(fields.get("word") or fields.get("front") or "").strip()
    translation = str(fields.get("translation") or fields.get("back") or "").strip()
    example = str(fields.get("example") or "").strip()
    notes = str(fields.get("notes") or "").strip()

    back_parts = [part for part in [translation, example, notes] if part]
    back = "\n\n".join(back_parts)
    return word or translation or "", back


def _compute_row_hash(fields: dict[str, Any], tags: str) -> str:
    payload = json.dumps({"fields": fields, "tags": tags}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_import_schema(conn: sqlite3.Connection):
    conn_id = id(conn)
    if conn_id in _SCHEMA_CHECKED:
        return
    ensure_schema_for_import(conn)
    _SCHEMA_CHECKED.add(conn_id)


def upsert_note_and_cards(
    conn: sqlite3.Connection,
    deck_id: int,
    external_id: str | None,
    fields: dict[str, Any],
    tags: str,
    srs_defaults: dict[str, Any],
    mode: dict[str, Any],
) -> dict[str, Any]:
    _ensure_import_schema(conn)
    cur = conn.cursor()
    note_type_id = _ensure_basic_note_type(cur)

    normalized_tags = normalize_tags(tags)
    now_ts = int(time.time())
    source = mode.get("source", "csv_import")
    row_hash = _compute_row_hash(fields, normalized_tags)

    existing_note = None
    if external_id is not None:
        cur.execute(
            "SELECT id FROM notes WHERE external_id = ? AND deck_id = ? LIMIT 1;",
            (str(external_id), deck_id),
        )
        existing_id_row = cur.fetchone()
        if existing_id_row:
            existing_id = (
                existing_id_row["id"]
                if hasattr(existing_id_row, "keys")
                else existing_id_row[0]
            )
            cur.execute("SELECT * FROM notes WHERE id = ?;", (existing_id,))
            existing_note = cur.fetchone()

    faces = render_card_faces(fields)
    srs_state = {
        "state": mode.get("state", srs_defaults.get("state", "new")),
        "due": int(srs_defaults.get("due", now_ts)),
        "interval": int(srs_defaults.get("interval", 0)),
        "ease": int(srs_defaults.get("ease", 250)),
        "reps": int(srs_defaults.get("reps", 0)),
        "lapses": int(srs_defaults.get("lapses", 0)),
        "step_index": int(srs_defaults.get("step_index", 0)),
        "phase": int(srs_defaults.get("phase", 1)),
    }

    if existing_note:
        if mode.get("skip_existing"):
            return {"status": "skipped", "note_id": existing_note["id"], "card_ids": []}

        update_fields = fields.copy()
        update_fields.setdefault("row_hash", row_hash)

        cur.execute(
            "UPDATE notes SET fields_json = ?, tags = COALESCE(?, tags), source = COALESCE(source, ?) WHERE id = ?;",
            (json.dumps(update_fields, ensure_ascii=False), normalized_tags or existing_note["tags"], source, existing_note["id"]),
        )

        cur.execute("SELECT * FROM cards WHERE note_id = ?;", (existing_note["id"],))
        cards = cur.fetchall()
        for card in cards:
            placeholders = ["front = :front", "back = :back", "phase = :phase", "external_id = :external_id", "source = :source"]
            params = {
                "front": faces[0],
                "back": faces[1],
                "phase": srs_state["phase"],
                "external_id": external_id,
                "source": source,
                "id": card["id"],
            }

            if mode.get("reset_srs"):
                placeholders.extend(
                    [
                        "state = :state",
                        "due = :due",
                        "interval = :interval",
                        "ease = :ease",
                        "reps = :reps",
                        "lapses = :lapses",
                        "step_index = :step_index",
                    ]
                )
                params.update(srs_state)

            cur.execute(f"UPDATE cards SET {', '.join(placeholders)} WHERE id = :id;", params)

        return {"status": "updated", "note_id": existing_note["id"], "card_ids": [card["id"] for card in cards]}

    cur.execute(
        """
        INSERT INTO notes (deck_id, note_type_id, fields_json, tags, external_id, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            deck_id,
            note_type_id,
            json.dumps({**fields, "row_hash": row_hash}, ensure_ascii=False),
            normalized_tags,
            str(external_id) if external_id is not None else None,
            source,
            now_ts,
        ),
    )
    note_id = cur.lastrowid

    cur.execute(
        """
        INSERT INTO cards (deck_id, front, back, next_review, leitner_level, note_id, template_ord, state,
                           due, interval, ease, reps, lapses, step_index, phase, external_id, source, translation_shown, overview_added)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1);
        """,
        (
            deck_id,
            faces[0],
            faces[1],
            time.strftime("%Y-%m-%d", time.localtime(srs_state["due"])),
            srs_state["phase"],
            note_id,
            0,
            srs_state["state"],
            srs_state["due"],
            srs_state["interval"],
            srs_state["ease"],
            srs_state["reps"],
            srs_state["lapses"],
            srs_state["step_index"],
            srs_state["phase"],
            str(external_id) if external_id is not None else None,
            source,
        ),
    )
    card_id = cur.lastrowid

    return {"status": "created", "note_id": note_id, "card_ids": [card_id]}


def attach_image_if_exists(
    conn: sqlite3.Connection,
    note_id: int,
    card_id: int,
    external_id: str | None,
    images_dir: str,
):
    if not external_id:
        return

    for ext in MEDIA_EXTENSIONS:
        candidate = os.path.join(images_dir, f"{external_id}{ext}")
        if not os.path.exists(candidate):
            continue

        cur = conn.cursor()

        def write():
            cur.execute(
                """
                INSERT INTO media (note_id, card_id, type, path, side, source, created_at)
                VALUES (?, ?, 'image', ?, 'front', 'csv_import', ?);
                """,
                (note_id, card_id, candidate, int(time.time())),
            )

        commit_with_retry(conn, write)
        return
