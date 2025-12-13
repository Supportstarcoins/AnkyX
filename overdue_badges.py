"""Управление бейджами просрочки и миграцией поля due."""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

from db_path import get_db_path


@dataclass
class OverdueCounts:
    """Хранит статистику просрочек по фазам."""

    by_phase: Dict[int, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_phase.values())


def ensure_due_column(conn: sqlite3.Connection) -> None:
    """Добавляет колонку ``due`` в таблицу ``cards``, если её ещё нет."""

    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(cards);")
    columns = [row[1] for row in cursor.fetchall()]
    if "due" not in columns:
        cursor.execute("ALTER TABLE cards ADD COLUMN due INTEGER;")
        conn.commit()


def _find_parent_column(cursor: sqlite3.Cursor) -> Optional[str]:
    cursor.execute("PRAGMA table_info(decks);")
    columns = [row[1] for row in cursor.fetchall()]
    for candidate in ("parent_id", "parent_deck_id"):
        if candidate in columns:
            return candidate
    return None


def get_descendant_deck_ids(conn: sqlite3.Connection, deck_id: int) -> List[int]:
    """Возвращает ``deck_id`` с его потомками (если есть)."""

    cursor = conn.cursor()
    parent_column = _find_parent_column(cursor)
    if not parent_column:
        return [deck_id]

    result: List[int] = [deck_id]
    queue: List[int] = [deck_id]
    seen = {deck_id}

    while queue:
        current = queue.pop(0)
        cursor.execute(f"SELECT id FROM decks WHERE {parent_column} = ?;", (current,))
        for row in cursor.fetchall():
            child_id = int(row[0])
            if child_id not in seen:
                seen.add(child_id)
                result.append(child_id)
                queue.append(child_id)

    return result


def _resolve_db_path(db_path: str) -> str:
    if not os.path.isabs(db_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, db_path)
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    return db_path


def _connect_with_logging(timeout: Optional[int] = 5) -> sqlite3.Connection:
    db_path = get_db_path()
    print("[DB] CWD=", os.getcwd())
    print("[DB] trying db_path=", repr(db_path))
    resolved_path = _resolve_db_path(db_path)

    try:
        conn = sqlite3.connect(resolved_path, timeout=timeout) if timeout is not None else sqlite3.connect(resolved_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        try:
            messagebox.showerror("Ошибка БД", f"Не удалось открыть БД:\n{resolved_path}\n{exc}")
        except Exception:
            print(f"[DB] Failed to open DB at {resolved_path}: {exc}")
        raise


def fetch_overdue_counts_by_phase(
    conn: Optional[sqlite3.Connection], deck_id: int, *, now_ts: Optional[int] = None
) -> OverdueCounts:
    """Возвращает количество просроченных карточек по фазам."""

    close_conn = False
    if conn is None:
        conn = _connect_with_logging()
        close_conn = True

    try:
        ensure_due_column(conn)
        timestamp = now_ts if now_ts is not None else int(time.time())
        deck_ids = get_descendant_deck_ids(conn, deck_id)

        placeholders = ",".join("?" for _ in deck_ids)
        query = (
            "SELECT phase, COUNT(*) FROM cards "
            f"WHERE deck_id IN ({placeholders}) AND due IS NOT NULL AND due <= ? "
            "GROUP BY phase;"
        )

        cursor = conn.cursor()
        cursor.execute(query, (*deck_ids, timestamp))
        counts = {int(row[0]): int(row[1]) for row in cursor.fetchall() if row[0] is not None}
        return OverdueCounts(by_phase=counts)
    finally:
        if close_conn:
            conn.close()


class PhaseOverdueBadges:
    """Отображает красные бейджи в дереве фаз."""

    def __init__(self, tree: ttk.Treeview):
        self.tree = tree
        self._badges: dict[str, tk.Canvas] = {}
        self._background = self._resolve_background()

        self.tree.bind("<Configure>", self._on_configure, add=True)
        self.tree.bind("<<TreeviewOpen>>", self._on_configure, add=True)
        self.tree.bind("<<TreeviewClose>>", self._on_configure, add=True)

    def update(self, deck_items: dict, counts_by_deck: Dict[int, OverdueCounts]):
        self._cleanup_badges(deck_items)
        for item_id, (deck_id, phase) in deck_items.items():
            if phase is None or deck_id is None:
                continue
            deck_counts = counts_by_deck.get(deck_id)
            count = deck_counts.by_phase.get(phase, 0) if deck_counts else 0
            self._render_badge(item_id, count)

    def _cleanup_badges(self, deck_items: dict):
        for item_id in list(self._badges.keys()):
            if item_id not in deck_items:
                self._badges[item_id].destroy()
                del self._badges[item_id]

    def _render_badge(self, item_id: str, count: int):
        canvas = self._badges.get(item_id)
        if count <= 0:
            if canvas:
                canvas.place_forget()
            return

        if canvas is None:
            canvas = tk.Canvas(
                self.tree,
                width=22,
                height=22,
                highlightthickness=0,
                bg=self._background,
                borderwidth=0,
            )
            self._badges[item_id] = canvas
        else:
            canvas.delete("all")

        canvas.create_oval(2, 2, 20, 20, fill="red", outline="red")
        canvas.create_text(11, 11, text=str(count), fill="white", font=("Arial", 9, "bold"))

        self._place_badge(item_id, canvas)

    def _place_badge(self, item_id: str, canvas: tk.Canvas):
        bbox = self.tree.bbox(item_id)
        if not bbox:
            canvas.place_forget()
            return

        x, y, width, height = bbox
        target_x = min(x + width + 6, max(self.tree.winfo_width() - 24, x))
        target_y = y + (height - 22) / 2
        canvas.place(x=target_x, y=target_y)

    def _on_configure(self, _event=None):
        for item_id, canvas in self._badges.items():
            self._place_badge(item_id, canvas)

    def _resolve_background(self) -> str:
        style = ttk.Style(self.tree)
        bg = style.lookup("Treeview", "fieldbackground") or style.lookup("Treeview", "background")
        return bg or self.tree.cget("background")
