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

    conn.commit()
