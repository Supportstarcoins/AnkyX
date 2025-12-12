import json
import sqlite3


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table});")
    return any(row[1] == column for row in cursor.fetchall())


def _add_column_if_missing(cursor: sqlite3.Cursor, table: str, column: str, definition: str):
    if not _column_exists(cursor, table, column):
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")


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
            created_at INTEGER NOT NULL,
            FOREIGN KEY(deck_id) REFERENCES decks(id),
            FOREIGN KEY(note_type_id) REFERENCES note_types(id)
        );
        """
    )

    _add_column_if_missing(cur, "cards", "note_id", "INTEGER")
    _add_column_if_missing(cur, "cards", "template_ord", "INTEGER")

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
