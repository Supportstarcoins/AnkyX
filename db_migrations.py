import json
import sqlite3
import time


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table});")
    return any(row[1] == column for row in cursor.fetchall())


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1;",
        (table,),
    )
    return cursor.fetchone() is not None


def _add_column_if_missing(cursor: sqlite3.Cursor, table: str, column: str, definition: str):
    if not _column_exists(cursor, table, column):
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")


def migrate_media_note_nullable(conn: sqlite3.Connection):
    """Make media.note_id nullable while preserving existing data."""

    cur = conn.cursor()
    if not _table_exists(cur, "media"):
        return

    cur.execute("PRAGMA table_info(media);")
    table_info = cur.fetchall()
    note_info = next((row for row in table_info if row[1] == "note_id"), None)
    if not note_info or note_info[3] == 0:
        return

    timestamp = int(time.time())
    old_table = f"media_old_{timestamp}"

    cur.execute(f"ALTER TABLE media RENAME TO {old_table};")
    cur.execute(
        """
        CREATE TABLE media (
            id INTEGER PRIMARY KEY,
            note_id INTEGER,
            card_id INTEGER,
            type TEXT NOT NULL,
            path TEXT NOT NULL,
            side TEXT NOT NULL DEFAULT 'back',
            source TEXT,
            created_at INTEGER NOT NULL
        );
        """,
    )

    cur.execute(f"PRAGMA table_info({old_table});")
    old_columns = {row[1] for row in cur.fetchall()}

    type_expr = "COALESCE(media_type, type, 'unknown')"
    if "media_type" not in old_columns:
        type_expr = "type" if "type" in old_columns else "'unknown'"
    if "type" not in old_columns and "media_type" in old_columns:
        type_expr = "media_type"

    path_expr = "COALESCE(path, filepath, '')"
    if "path" not in old_columns and "filepath" in old_columns:
        path_expr = "filepath"
    elif "path" in old_columns and "filepath" not in old_columns:
        path_expr = "path"
    elif "path" not in old_columns and "filepath" not in old_columns:
        path_expr = "''"

    card_expr = "card_id" if "card_id" in old_columns else "NULL"
    side_expr = "side" if "side" in old_columns else "'back'"
    source_expr = "source" if "source" in old_columns else "NULL"
    created_expr = "created_at" if "created_at" in old_columns else "CAST(strftime('%s','now') AS INTEGER)"

    cur.execute(
        f"""
        INSERT INTO media (id, note_id, card_id, type, path, side, source, created_at)
        SELECT
            id,
            note_id,
            {card_expr},
            {type_expr},
            {path_expr},
            {side_expr},
            {source_expr},
            {created_expr}
        FROM {old_table};
        """
    )

    cur.execute(f"DROP TABLE {old_table};")
    conn.commit()


def run_migrations(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id INTEGER NOT NULL,
            reviewed_at INTEGER NOT NULL,
            rating INTEGER NOT NULL,
            interval_before INTEGER,
            interval_after INTEGER,
            ease_before INTEGER,
            ease_after INTEGER,
            phase_before INTEGER,
            phase_after INTEGER,
            FOREIGN KEY(card_id) REFERENCES cards(id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS import_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            imported_at INTEGER NOT NULL
        );
        """
    )

    srs_columns = {
        "state": "TEXT",
        "due": "INTEGER",
        "interval": "INTEGER",
        "ease": "INTEGER",
        "reps": "INTEGER",
        "lapses": "INTEGER",
        "step_index": "INTEGER",
        "last_review": "INTEGER",
    }

    for column, definition in srs_columns.items():
        _add_column_if_missing(cur, "cards", column, definition)
    _add_column_if_missing(cur, "cards", "phase", "INTEGER")
    _add_column_if_missing(cur, "cards", "external_id", "TEXT")
    _add_column_if_missing(cur, "cards", "source", "TEXT")

    # Новые таблицы для note types/notes
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS note_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            fields_json TEXT NOT NULL,
            card_templates_json TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deck_id INTEGER NOT NULL,
            note_type_id INTEGER NOT NULL,
            fields_json TEXT NOT NULL,
            tags TEXT,
            external_id TEXT,
            source TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(deck_id) REFERENCES decks(id),
            FOREIGN KEY(note_type_id) REFERENCES note_types(id)
        );
        """
    )

    _add_column_if_missing(cur, "notes", "external_id", "TEXT")
    _add_column_if_missing(cur, "notes", "source", "TEXT")

    _add_column_if_missing(cur, "cards", "note_id", "INTEGER")
    _add_column_if_missing(cur, "cards", "template_ord", "INTEGER")

    # Медиа привязанные к заметкам
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY,
            note_id INTEGER,
            card_id INTEGER,
            type TEXT NOT NULL,
            path TEXT NOT NULL,
            side TEXT NOT NULL DEFAULT 'back',
            source TEXT,
            created_at INTEGER NOT NULL
        );
        """
    )

    migrate_media_note_nullable(conn)

    # Добавляем дефолтный тип заметки "Basic"
    cur.execute(
        "SELECT id FROM note_types WHERE name = 'Basic' LIMIT 1;"
    )
    if cur.fetchone() is None:
        fields = ["word", "translation", "example", "level", "image"]
        templates = [
            {
                "name": "Word→Translation",
                "front": "{word}",
                "back": "{translation}\n\n{example}",
                "requires_image": False,
            },
            {
                "name": "Image→Word",
                "front": "{image}",
                "back": "{word}\n{translation}",
                "requires_image": True,
            },
        ]
        cur.execute(
            "INSERT INTO note_types (name, fields_json, card_templates_json) VALUES (?, ?, ?);",
            ("Basic", json.dumps(fields, ensure_ascii=False), json.dumps(templates, ensure_ascii=False)),
        )

    conn.commit()
