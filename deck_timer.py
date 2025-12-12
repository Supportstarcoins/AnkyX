import sqlite3
from typing import Callable, Optional, Set, Tuple

DB_NAME = "anki.db"


def _prepare_connection(conn: Optional[sqlite3.Connection]) -> tuple[sqlite3.Connection, bool]:
    if conn is not None:
        return conn, False
    connection = sqlite3.connect(DB_NAME, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection, True


def ensure_deck_settings_table(conn: Optional[sqlite3.Connection] = None) -> None:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS deck_settings (
            deck_id INTEGER PRIMARY KEY,
            timer_sec INTEGER DEFAULT 0,
            timer_mode TEXT DEFAULT "reveal",
            inherit_timer INTEGER DEFAULT 1
        );
        """
    )

    cur.execute("PRAGMA table_info(deck_settings);")
    existing_columns = {row[1] for row in cur.fetchall()}
    migrations = {
        "timer_sec": "INTEGER DEFAULT 0",
        "timer_mode": "TEXT DEFAULT 'reveal'",
        "inherit_timer": "INTEGER DEFAULT 1",
    }
    for column, ddl in migrations.items():
        if column not in existing_columns:
            cur.execute(f"ALTER TABLE deck_settings ADD COLUMN {column} {ddl};")

    if created:
        conn.commit()
        conn.close()
    else:
        conn.commit()


def ensure_deck_settings_row(
    deck_id: int, conn: Optional[sqlite3.Connection] = None, inherit_default: int = 1
) -> None:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()
    ensure_deck_settings_table(conn)

    cur.execute("SELECT 1 FROM deck_settings WHERE deck_id = ? LIMIT 1;", (deck_id,))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO deck_settings (deck_id, inherit_timer) VALUES (?, ?);",
            (deck_id, inherit_default),
        )

    if created:
        conn.commit()
        conn.close()
    else:
        conn.commit()


def get_deck_parent_id(deck_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[int]:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(decks);")
    columns = {row[1] for row in cur.fetchall()}
    parent_col = None
    for candidate in ("parent_id", "parent_deck_id"):
        if candidate in columns:
            parent_col = candidate
            break

    if parent_col is None:
        if created:
            conn.close()
        return None

    cur.execute(f"SELECT {parent_col} FROM decks WHERE id = ?;", (deck_id,))
    row = cur.fetchone()
    if created:
        conn.close()
    if not row:
        return None
    parent_id = row[0]
    if parent_id is None or parent_id == deck_id:
        return None
    return int(parent_id)


def get_deck_timer_settings(
    deck_id: int, conn: Optional[sqlite3.Connection] = None
) -> dict[str, int | str]:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()
    ensure_deck_settings_table(conn)
    ensure_deck_settings_row(deck_id, conn)

    cur.execute(
        "SELECT timer_sec, timer_mode, inherit_timer FROM deck_settings WHERE deck_id = ?;",
        (deck_id,),
    )
    row = cur.fetchone()
    if created:
        conn.close()
    if not row:
        return {"timer_sec": 0, "timer_mode": "reveal", "inherit_timer": 1}
    return {
        "timer_sec": row["timer_sec"],
        "timer_mode": row["timer_mode"],
        "inherit_timer": row["inherit_timer"],
    }


def update_deck_timer_settings(
    deck_id: int,
    timer_sec: int,
    timer_mode: str,
    inherit_timer: int,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()
    ensure_deck_settings_table(conn)
    ensure_deck_settings_row(deck_id, conn)

    cur.execute(
        """
        UPDATE deck_settings
        SET timer_sec = ?, timer_mode = ?, inherit_timer = ?
        WHERE deck_id = ?;
        """,
        (timer_sec or 0, (timer_mode or "reveal").lower(), int(bool(inherit_timer)), deck_id),
    )

    if created:
        conn.commit()
        conn.close()
    else:
        conn.commit()


def get_effective_timer(
    deck_id: int,
    conn: Optional[sqlite3.Connection] = None,
    visited: Optional[Set[int]] = None,
) -> Tuple[int, str]:
    if deck_id is None:
        return 0, "reveal"
    if visited is None:
        visited = set()
    if deck_id in visited:
        return 0, "reveal"
    visited.add(deck_id)

    conn, created = _prepare_connection(conn)
    settings = get_deck_timer_settings(deck_id, conn)
    timer_sec = settings.get("timer_sec") or 0
    timer_mode = (settings.get("timer_mode") or "reveal").lower()
    inherit_timer = int(settings.get("inherit_timer") or 0)

    if timer_sec and timer_sec > 0:
        result = (int(timer_sec), timer_mode)
    elif inherit_timer:
        parent_id = get_deck_parent_id(deck_id, conn)
        if parent_id:
            result = get_effective_timer(parent_id, conn, visited)
        else:
            result = (0, timer_mode)
    else:
        result = (0, timer_mode)

    if created:
        conn.close()
    return result


class DeckTimerController:
    def __init__(
        self,
        widget,
        update_label: Callable[[int], None],
        on_reveal: Callable[[], None],
        on_fail: Callable[[], None],
        on_notify: Callable[[], None],
    ):
        self.widget = widget
        self.update_label = update_label
        self.on_reveal = on_reveal
        self.on_fail = on_fail
        self.on_notify = on_notify

        self._job = None
        self._seconds_left = 0
        self._mode = "reveal"

    def cancel(self) -> None:
        if self._job is not None:
            try:
                self.widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        self._seconds_left = 0

    def start(self, seconds: int, mode: str = "reveal") -> None:
        self.cancel()
        self._seconds_left = max(0, int(seconds or 0))
        self._mode = (mode or "reveal").lower()

        if self._seconds_left <= 0:
            self.update_label(0)
            return

        self._tick()

    def _tick(self) -> None:
        self.update_label(self._seconds_left)
        if self._seconds_left <= 0:
            self._job = None
            self._on_complete()
            return
        self._seconds_left -= 1
        self._job = self.widget.after(1000, self._tick)

    def _on_complete(self) -> None:
        mode = self._mode
        if mode == "fail":
            self.on_fail()
        elif mode == "notify":
            self.on_notify()
        else:
            self.on_reveal()

    def is_running(self) -> bool:
        return self._job is not None and self._seconds_left > 0
