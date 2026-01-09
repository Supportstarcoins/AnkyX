import sqlite3
import json
from typing import Callable, Optional, Set, Tuple

from db_path import connect_to_db

DEFAULT_PHASE_INTERVALS = [
    30,  # 1: 30 seconds
    25 * 60,  # 2: 25 minutes
    60 * 60,  # 3: 1 hour
    24 * 60 * 60,  # 4: 1 day
    3 * 24 * 60 * 60,  # 5: 3 days
    9 * 24 * 60 * 60,  # 6: 9 days
    16 * 24 * 60 * 60,  # 7: 16 days
    36 * 24 * 60 * 60,  # 8: 36 days
    56 * 24 * 60 * 60,  # 9: 56 days
    100 * 24 * 60 * 60,  # 10: 100 days
]


def _prepare_connection(conn: Optional[sqlite3.Connection]) -> tuple[sqlite3.Connection, bool]:
    if conn is not None:
        return conn, False
    connection = connect_to_db(timeout=5)
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
            inherit_timer INTEGER DEFAULT 1,
            review_timer_seconds INTEGER,
            playback_timer_seconds INTEGER,
            user_phase_intervals TEXT
        );
        """
    )

    cur.execute("PRAGMA table_info(deck_settings);")
    existing_columns = {row[1] for row in cur.fetchall()}
    migrations = {
        "timer_sec": "INTEGER DEFAULT 0",
        "timer_mode": "TEXT DEFAULT 'reveal'",
        "inherit_timer": "INTEGER DEFAULT 1",
        "review_timer_seconds": "INTEGER",
        "playback_timer_seconds": "INTEGER",
        "user_phase_intervals": "TEXT",
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
) -> dict[str, int | str | None]:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()
    ensure_deck_settings_table(conn)
    ensure_deck_settings_row(deck_id, conn)

    cur.execute(
        """
        SELECT timer_sec, timer_mode, inherit_timer, review_timer_seconds, playback_timer_seconds
        FROM deck_settings WHERE deck_id = ?;
        """,
        (deck_id,),
    )
    row = cur.fetchone()
    if created:
        conn.close()
    if not row:
        return {
            "timer_sec": 0,
            "timer_mode": "reveal",
            "inherit_timer": 1,
            "review_timer_seconds": None,
            "playback_timer_seconds": None,
            "user_phase_intervals": None,
        }
    return {
        "timer_sec": row["timer_sec"],
        "timer_mode": row["timer_mode"],
        "inherit_timer": row["inherit_timer"],
        "review_timer_seconds": row["review_timer_seconds"],
        "playback_timer_seconds": row["playback_timer_seconds"],
        "user_phase_intervals": row["user_phase_intervals"],
    }


def update_deck_timer_settings(
    deck_id: int,
    timer_sec: int,
    timer_mode: str,
    inherit_timer: int,
    review_timer_seconds: Optional[int] = None,
    playback_timer_seconds: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()
    ensure_deck_settings_table(conn)
    ensure_deck_settings_row(deck_id, conn)

    cur.execute(
        """
        UPDATE deck_settings
        SET timer_sec = ?, timer_mode = ?, inherit_timer = ?,
            review_timer_seconds = ?, playback_timer_seconds = ?
        WHERE deck_id = ?;
        """,
        (
            timer_sec or 0,
            (timer_mode or "reveal").lower(),
            int(bool(inherit_timer)),
            review_timer_seconds,
            playback_timer_seconds,
            deck_id,
        ),
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


def get_deck_phase_intervals(
    deck_id: int, conn: Optional[sqlite3.Connection] = None
) -> list[int]:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()
    ensure_deck_settings_table(conn)
    ensure_deck_settings_row(deck_id, conn)
    cur.execute(
        "SELECT user_phase_intervals FROM deck_settings WHERE deck_id = ?;",
        (deck_id,),
    )
    row = cur.fetchone()
    if created:
        conn.close()
    raw = row["user_phase_intervals"] if row else None
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                result = [max(0, int(val)) for val in data][: len(DEFAULT_PHASE_INTERVALS)]
                if len(result) < len(DEFAULT_PHASE_INTERVALS):
                    result.extend(DEFAULT_PHASE_INTERVALS[len(result) :])
                return result
        except Exception:
            pass
    return list(DEFAULT_PHASE_INTERVALS)


def save_deck_phase_intervals(
    deck_id: int,
    intervals: list[int],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()
    ensure_deck_settings_table(conn)
    ensure_deck_settings_row(deck_id, conn)
    payload = json.dumps([max(0, int(val)) for val in intervals], ensure_ascii=False)
    cur.execute(
        "UPDATE deck_settings SET user_phase_intervals = ? WHERE deck_id = ?;",
        (payload, deck_id),
    )
    if created:
        conn.commit()
        conn.close()
    else:
        conn.commit()


def reset_deck_phase_intervals(
    deck_id: int, conn: Optional[sqlite3.Connection] = None
) -> None:
    conn, created = _prepare_connection(conn)
    cur = conn.cursor()
    ensure_deck_settings_table(conn)
    ensure_deck_settings_row(deck_id, conn)
    cur.execute(
        "UPDATE deck_settings SET user_phase_intervals = NULL WHERE deck_id = ?;",
        (deck_id,),
    )
    if created:
        conn.commit()
        conn.close()
    else:
        conn.commit()


def _normalize_optional_timer(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(0, int(value))


def get_effective_mode_timer(
    deck_id: int,
    mode: str,
    conn: Optional[sqlite3.Connection] = None,
    visited: Optional[Set[int]] = None,
) -> int:
    if deck_id is None:
        return 0
    if visited is None:
        visited = set()
    if deck_id in visited:
        return 0
    visited.add(deck_id)

    mode_key = (mode or "").lower()
    if mode_key == "review":
        column = "review_timer_seconds"
    elif mode_key == "playback":
        column = "playback_timer_seconds"
    else:
        raise ValueError("mode must be 'review' or 'playback'")

    conn, created = _prepare_connection(conn)
    settings = get_deck_timer_settings(deck_id, conn)
    raw_value = settings.get(column)
    if raw_value is not None:
        result = _normalize_optional_timer(raw_value) or 0
    else:
        parent_id = get_deck_parent_id(deck_id, conn)
        if parent_id:
            result = get_effective_mode_timer(parent_id, mode_key, conn, visited)
        else:
            result = 0

    if created:
        conn.close()
    return int(result or 0)


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
