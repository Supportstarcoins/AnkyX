from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, List, Optional

from db_path import connect_to_db


@dataclass
class StatsSettings:
    deck_id: int
    x_mode: str = "range"  # month_days|range|custom_dates
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    custom_dates_json: str = ""
    y_max: int = 0
    norm_value: int = 1000
    chart_type: str = "bar"
    show_grid: int = 1

    @property
    def custom_dates(self) -> List[str]:
        if not self.custom_dates_json:
            return []
        try:
            data = json.loads(self.custom_dates_json)
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            return []
        return []

    def update_custom_dates(self, dates: Iterable[str]) -> None:
        cleaned = [d.strip() for d in dates if d and d.strip()]
        self.custom_dates_json = json.dumps(cleaned, ensure_ascii=False)


def _ensure_connection(conn: sqlite3.Connection | None) -> sqlite3.Connection:
    if conn is not None:
        return conn
    return connect_to_db(timeout=None)


def ensure_stats_settings_table(conn: sqlite3.Connection | None = None) -> None:
    owns_conn = conn is None
    conn = _ensure_connection(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stats_settings (
                deck_id INTEGER PRIMARY KEY,
                x_mode TEXT DEFAULT "range",
                date_from TEXT,
                date_to TEXT,
                custom_dates_json TEXT,
                y_max INTEGER DEFAULT 0,
                norm_value INTEGER DEFAULT 1000,
                chart_type TEXT DEFAULT "bar",
                show_grid INTEGER DEFAULT 1
            );
            """
        )
        if owns_conn:
            conn.commit()
    finally:
        if owns_conn:
            conn.close()


def load_stats_settings(deck_id: int, conn: sqlite3.Connection | None = None) -> StatsSettings:
    owns_conn = conn is None
    conn = _ensure_connection(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT deck_id, x_mode, date_from, date_to, custom_dates_json, y_max, norm_value, chart_type, show_grid
            FROM stats_settings WHERE deck_id = ?
            """,
            (deck_id,),
        )
        row = cur.fetchone()
        if row:
            return StatsSettings(
                deck_id=row["deck_id"],
                x_mode=row["x_mode"] or "range",
                date_from=row["date_from"],
                date_to=row["date_to"],
                custom_dates_json=row["custom_dates_json"] or "",
                y_max=row["y_max"] or 0,
                norm_value=row["norm_value"] or 0,
                chart_type=row["chart_type"] or "bar",
                show_grid=row["show_grid"] if row["show_grid"] is not None else 1,
            )
        settings = StatsSettings(deck_id=deck_id)
        save_stats_settings(settings, conn)
        return settings
    finally:
        if owns_conn:
            conn.close()


def save_stats_settings(settings: StatsSettings, conn: sqlite3.Connection | None = None) -> None:
    owns_conn = conn is None
    conn = _ensure_connection(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO stats_settings (
                deck_id, x_mode, date_from, date_to, custom_dates_json, y_max, norm_value, chart_type, show_grid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(deck_id) DO UPDATE SET
                x_mode = excluded.x_mode,
                date_from = excluded.date_from,
                date_to = excluded.date_to,
                custom_dates_json = excluded.custom_dates_json,
                y_max = excluded.y_max,
                norm_value = excluded.norm_value,
                chart_type = excluded.chart_type,
                show_grid = excluded.show_grid
            """,
            (
                settings.deck_id,
                settings.x_mode,
                settings.date_from,
                settings.date_to,
                settings.custom_dates_json,
                settings.y_max,
                settings.norm_value,
                settings.chart_type,
                settings.show_grid,
            ),
        )
        if owns_conn:
            conn.commit()
    finally:
        if owns_conn:
            conn.close()


__all__ = [
    "StatsSettings",
    "ensure_stats_settings_table",
    "load_stats_settings",
    "save_stats_settings",
]
