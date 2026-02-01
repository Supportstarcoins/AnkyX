from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Callable, TYPE_CHECKING

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from card_widget import CardWidget
from chatbot_models import ChatSession, DraftBatch, DraftCard, Message
from credit_manager import CreditManager
from csv_importer import upsert_note_and_cards
from db_connect import open_db
from mock_ai_engine import MockAIEngine

if TYPE_CHECKING:
    from main import RepeatModeCardView

try:
    from tkinterdnd2 import DND_FILES
except Exception:  # noqa: BLE001
    DND_FILES = None


class AutoGrowText(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        *,
        min_lines: int = 1,
        max_lines: int = 10,
        on_send: Callable[[], None] | None = None,
        on_change: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master)
        self.min_lines = min_lines
        self.max_lines = max_lines
        self._on_send = on_send
        self._on_change = on_change
        self._scrollbar_visible = False

        self.text = tk.Text(
            self,
            height=min_lines,
            wrap=tk.WORD,
            relief="flat",
            bd=0,
            **kwargs,
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)

        self.text.bind("<<Modified>>", self._on_modified)
        self.text.bind("<Return>", self._handle_return)
        self.text.bind("<Shift-Return>", self._handle_shift_return)

        self._menu = tk.Menu(self, tearoff=False)
        self._menu.add_command(label="Вырезать", command=self._cut)
        self._menu.add_command(label="Копировать", command=self._copy)
        self._menu.add_command(label="Вставить", command=self._paste)
        self._menu.add_separator()
        self._menu.add_command(label="Выделить всё", command=self._select_all)
        self.text.bind("<Button-3>", self._show_menu)

    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    def clear(self) -> None:
        self.text.delete("1.0", tk.END)
        self._update_height()
        if self._on_change:
            self._on_change()

    def set_state(self, state: str) -> None:
        self.text.configure(state=state)

    def focus_text(self) -> None:
        self.text.focus_set()

    def _handle_return(self, event: tk.Event) -> str:
        if self._on_send:
            self._on_send()
        return "break"

    def _handle_shift_return(self, _event: tk.Event) -> str:
        self.text.insert(tk.INSERT, "\n")
        return "break"

    def _on_modified(self, _event=None) -> None:
        self.text.edit_modified(False)
        self._update_height()
        if self._on_change:
            self._on_change()

    def _update_height(self) -> None:
        lines = int(self.text.index("end-1c").split(".")[0])
        new_height = max(self.min_lines, min(lines, self.max_lines))
        self.text.configure(height=new_height)
        if lines > self.max_lines:
            if not self._scrollbar_visible:
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self._scrollbar_visible = True
        else:
            if self._scrollbar_visible:
                self.scrollbar.pack_forget()
                self._scrollbar_visible = False

    def _show_menu(self, event: tk.Event) -> None:
        self._menu.tk_popup(event.x_root, event.y_root)

    def _cut(self) -> None:
        self.text.event_generate("<<Cut>>")

    def _copy(self) -> None:
        self.text.event_generate("<<Copy>>")

    def _paste(self) -> None:
        self.text.event_generate("<<Paste>>")

    def _select_all(self) -> None:
        self.text.tag_add("sel", "1.0", "end-1c")


class AttachmentsBar(ttk.Frame):
    def __init__(self, master: tk.Widget, *, palette: dict, on_remove: Callable[[int], None]) -> None:
        super().__init__(master)
        self.palette = palette
        self.on_remove = on_remove
        self._chips: list[ttk.Frame] = []

    def render(self, attachments: list[dict[str, Any]]) -> None:
        for chip in self._chips:
            chip.destroy()
        self._chips.clear()
        if not attachments:
            return
        for index, item in enumerate(attachments):
            chip = ttk.Frame(self, style="CardInner.TFrame", padding=(6, 3))
            chip.pack(side=tk.LEFT, padx=(0, 6), pady=4)
            name = item.get("name") or os.path.basename(item.get("path", "")) or "файл"
            label = ttk.Label(chip, text=name, style="Muted.TLabel")
            label.pack(side=tk.LEFT)
            size_label = ttk.Label(chip, text=f" {item.get('size_label', '')}", style="Muted.TLabel")
            size_label.pack(side=tk.LEFT)
            btn = ttk.Button(chip, text="✕", width=2, command=lambda idx=index: self.on_remove(idx))
            btn.pack(side=tk.LEFT, padx=(4, 0))
            self._chips.append(chip)


class RoundSendButton(tk.Canvas):
    def __init__(self, master: tk.Widget, *, command: Callable[[], None], size: int = 36) -> None:
        super().__init__(master, width=size, height=size, highlightthickness=0, bd=0)
        self._command = command
        self._size = size
        self._enabled = True
        self._bg_enabled = "#1E6EFF"
        self._bg_disabled = "#4B4B4B"
        self._fg_enabled = "#FFFFFF"
        self._fg_disabled = "#B0B0B0"
        self._draw_button()
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._draw_button()

    def _draw_button(self) -> None:
        self.delete("all")
        bg = self._bg_enabled if self._enabled else self._bg_disabled
        fg = self._fg_enabled if self._enabled else self._fg_disabled
        pad = 2
        self.create_oval(pad, pad, self._size - pad, self._size - pad, fill=bg, outline=bg)
        self.create_text(self._size // 2, self._size // 2 - 1, text="▲", fill=fg, font=("Segoe UI", 11, "bold"))

    def _on_click(self, _event: tk.Event) -> None:
        if self._enabled:
            self._command()

    def _on_enter(self, _event: tk.Event) -> None:
        self.configure(cursor="hand2" if self._enabled else "")

    def _on_leave(self, _event: tk.Event) -> None:
        self.configure(cursor="")


class CardPreviewWidget(ttk.Frame):
    def __init__(self, master: tk.Widget, palette: dict, on_save: callable) -> None:
        super().__init__(master, style="Card.TFrame", padding=10)
        self.palette = palette
        self.on_save = on_save
        self.cards: list[DraftCard] = []
        self.current_index = 0
        self.show_back = False
        self.total_credits = 0

        nav_frame = ttk.Frame(self, style="CardInner.TFrame")
        nav_frame.pack(fill=tk.X)

        self.prev_btn = ttk.Button(nav_frame, text="◀", width=3, command=self._prev)
        self.prev_btn.pack(side=tk.LEFT)

        self.index_var = tk.StringVar(value="Карточка 0/0")
        ttk.Label(nav_frame, textvariable=self.index_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=8)

        self.next_btn = ttk.Button(nav_frame, text="▶", width=3, command=self._next)
        self.next_btn.pack(side=tk.LEFT)

        self.toggle_btn = ttk.Button(nav_frame, text="Показать BACK", command=self._toggle_side)
        self.toggle_btn.pack(side=tk.RIGHT)

        self.card_widget = CardWidget(
            self,
            palette=self.palette,
            editable=False,
            width=620,
            height=220,
            show_image_toolbar=False,
        )
        self.card_widget.pack(fill=tk.BOTH, expand=True, pady=(8, 6))

        self.media_var = tk.StringVar(value="")
        self.media_label = ttk.Label(self, textvariable=self.media_var, style="Muted.TLabel")
        self.media_label.pack(anchor=tk.W, pady=(0, 6))

        self.save_btn = ttk.Button(self, text="Сохранить карточки", command=self.on_save, style="Primary.TButton")
        self.save_btn.pack(fill=tk.X, pady=(4, 0))

        self._update_ui_state()

    def set_cards(self, cards: list[DraftCard], start_index: int = 0, total_credits: int | None = None) -> None:
        self.cards = list(cards)
        if total_credits is not None:
            self.total_credits = total_credits
        if self.cards:
            self.current_index = max(0, min(start_index, len(self.cards) - 1))
        else:
            self.current_index = 0
        self.show_back = False
        self._render()

    def append_cards(
        self,
        cards: list[DraftCard],
        select_last: bool = True,
        total_credits: int | None = None,
    ) -> None:
        if not cards:
            return
        previous_count = len(self.cards)
        self.cards.extend(cards)
        if total_credits is not None:
            self.total_credits = total_credits
        if select_last:
            self.current_index = len(self.cards) - 1
            self.show_back = False
        elif previous_count == 0:
            self.current_index = 0
        self._render()

    def clear(self) -> None:
        self.cards = []
        self.total_credits = 0
        self.current_index = 0
        self.show_back = False
        self._render()

    def next_card(self) -> None:
        if self.current_index < len(self.cards) - 1:
            self.current_index += 1
            self._render()

    def prev_card(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._render()

    def get_current_card(self) -> DraftCard | None:
        if not self.cards:
            return None
        return self.cards[self.current_index]

    def update_counter_label(self) -> None:
        if not self.cards:
            self.index_var.set("Карточка 0/0")
            return
        self.index_var.set(f"Карточка {self.current_index + 1}/{len(self.cards)}")

    def set_save_state(self, enabled: bool, total_credits: int | None = None) -> None:
        if total_credits is not None:
            self.total_credits = total_credits
        label = "Сохранить карточки"
        if self.total_credits > 0:
            label = f"Сохранить карточки ({self.total_credits} кредитов)"
        self.save_btn.configure(text=label, state=(tk.NORMAL if enabled else tk.DISABLED))

    def _render(self) -> None:
        if not self.cards:
            self.card_widget.set_text("Черновик пуст. Сначала сформируйте карточки.", "")
            self.card_widget.show_side(False)
            self.update_counter_label()
            self.media_var.set("")
            self.toggle_btn.configure(state=tk.DISABLED)
            self.prev_btn.configure(state=tk.DISABLED)
            self.next_btn.configure(state=tk.DISABLED)
            self.set_save_state(False, 0)
            return
        card = self.cards[self.current_index]
        self.card_widget.set_text(card.front, card.back)
        self.card_widget.show_side(self.show_back)
        self.update_counter_label()
        self.media_var.set("Медиа: есть" if card.media else "")
        self.toggle_btn.configure(text="Показать FRONT" if self.show_back else "Показать BACK")
        self.toggle_btn.configure(state=tk.NORMAL)
        allow_nav = len(self.cards) > 1
        self.prev_btn.configure(state=(tk.NORMAL if allow_nav and self.current_index > 0 else tk.DISABLED))
        self.next_btn.configure(state=(tk.NORMAL if allow_nav and self.current_index < len(self.cards) - 1 else tk.DISABLED))
        self.set_save_state(True, self.total_credits)

    def _toggle_side(self) -> None:
        if not self.cards:
            return
        self.show_back = not self.show_back
        self.toggle_btn.configure(text="Показать FRONT" if self.show_back else "Показать BACK")
        self._render()

    def _prev(self) -> None:
        self.prev_card()

    def _next(self) -> None:
        self.next_card()

    def _update_ui_state(self) -> None:
        self.set_save_state(False, 0)
        self.card_widget.set_text("Черновик пуст. Сначала сформируйте карточки.", "")
        self.card_widget.show_side(False)
        self.update_counter_label()
        self.toggle_btn.configure(state=tk.DISABLED)
        self.prev_btn.configure(state=tk.DISABLED)
        self.next_btn.configure(state=tk.DISABLED)


def draft_to_card(draft: DraftCard) -> dict:
    media = getattr(draft, "media", {}) or {}
    meta = getattr(draft, "meta", {}) or {}
    return {
        "id": meta.get("id") or meta.get("temp_id") or -1,
        "deck_id": meta.get("deck_id"),
        "front": getattr(draft, "front", "") or "",
        "back": getattr(draft, "back", "") or "",
        "front_rich": meta.get("front_rich"),
        "back_rich": meta.get("back_rich"),
        "front_html": meta.get("front_html"),
        "back_html": meta.get("back_html"),
        "front_image_path": media.get("front_image_path"),
        "back_image_path": media.get("back_image_path"),
        "image_path": media.get("image_path") or media.get("image"),
        "audio_path": media.get("audio_path"),
        "video_path": media.get("video_path"),
        "audio_entries": media.get("audio_entries"),
    }


class ChatBotTab(ttk.Frame):
    def __init__(self, master: tk.Widget, app) -> None:
        super().__init__(master, style="Surface.TFrame")
        self.app = app
        self.palette = app.palette
        self.engine = MockAIEngine()
        self.credit_manager = CreditManager()
        self.current_session_id: int | None = None
        self.current_draft: DraftBatch | None = None
        self.draft_cards: list[DraftCard] = []
        self.draft_index = 0
        self.draft_show_back = False
        self.pending_attachments: list[dict[str, Any]] = []
        self._deck_map: dict[str, int] = {}
        self.max_total_attachment_size = 100 * 1024 * 1024

        self._ensure_chat_tables()
        self._build_ui()
        self._render_current_draft()
        self._load_sessions()
        self.refresh_deck_options()
        self._lock_chat()

        self.app.register_balance_observer(self._update_save_button_state)

    def destroy(self):
        self.app.unregister_balance_observer(self._update_save_button_state)
        super().destroy()

    def _ensure_chat_tables(self) -> None:
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT,
                attachments_json TEXT,
                ts INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()
        conn.close()

    def _build_ui(self) -> None:
        container = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left_frame = ttk.Frame(container, style="Card.TFrame", padding=10)
        container.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Чаты", style="Section.TLabel").pack(anchor=tk.W)

        list_frame = ttk.Frame(left_frame, style="CardInner.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 6))

        self.chats_listbox = tk.Listbox(list_frame, height=8)
        self.chats_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.chats_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chats_listbox.configure(yscrollcommand=list_scroll.set)
        self.chats_listbox.bind("<<ListboxSelect>>", self._on_chat_select)

        self.new_chat_btn = ttk.Button(left_frame, text="Новый чат", command=self._create_new_chat)
        self.new_chat_btn.pack(fill=tk.X, pady=(4, 2))

        self.pro_hint_var = tk.StringVar(value="")
        self.pro_hint_label = ttk.Label(left_frame, textvariable=self.pro_hint_var, style="Muted.TLabel")
        self.pro_hint_label.pack(anchor=tk.W)

        right_frame = ttk.Frame(container, style="Surface.TFrame")
        container.add(right_frame, weight=4)

        deck_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=8)
        deck_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(deck_frame, text="Колода:", style="Muted.TLabel").pack(side=tk.LEFT)
        self.deck_var = tk.StringVar(value="")
        self.deck_combo = ttk.Combobox(deck_frame, textvariable=self.deck_var, state="readonly")
        self.deck_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.deck_combo.bind("<<ComboboxSelected>>", self._on_deck_change)

        self.status_var = tk.StringVar(value="Выберите колоду")
        self.status_label = ttk.Label(deck_frame, textvariable=self.status_var, style="Muted.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=(8, 0))

        self.sticky_frame = ttk.Frame(right_frame, style="Surface.TFrame", padding=6)
        self.sticky_frame.pack(fill=tk.X)

        nav_frame = ttk.Frame(self.sticky_frame, style="CardInner.TFrame")
        nav_frame.pack(fill=tk.X, pady=(0, 6))

        self.prev_draft_btn = ttk.Button(nav_frame, text="◀", width=3, command=self._prev_draft)
        self.prev_draft_btn.pack(side=tk.LEFT)

        self.draft_index_var = tk.StringVar(value="0/0")
        ttk.Label(nav_frame, textvariable=self.draft_index_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=8)

        self.next_draft_btn = ttk.Button(nav_frame, text="▶", width=3, command=self._next_draft)
        self.next_draft_btn.pack(side=tk.LEFT)

        from main import RepeatModeCardView

        self.card_view: RepeatModeCardView = RepeatModeCardView(
            self.sticky_frame,
            palette=self.palette,
            view_mode="chatbot",
        )
        self.card_view.pack(fill=tk.BOTH, expand=True)
        self.card_view.set_rating_enabled(False)

        self.save_draft_btn = ttk.Button(
            self.sticky_frame,
            text="Сохранить карточки",
            command=self._save_draft,
            style="Primary.TButton",
        )
        self.save_draft_btn.pack(fill=tk.X, pady=(6, 0))

        history_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=6)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

        self.chat_text = tk.Text(
            history_frame,
            wrap=tk.WORD,
            bg=self.palette.get("panel"),
            fg=self.palette.get("text"),
            relief="flat",
            bd=0,
        )
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_text.configure(state=tk.DISABLED)

        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.chat_text.yview)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_text.configure(yscrollcommand=history_scroll.set)

        self.composer_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=8)
        self.composer_frame.pack(fill=tk.X)

        self.attachments_bar = AttachmentsBar(
            self.composer_frame,
            palette=self.palette,
            on_remove=self._remove_attachment,
        )
        self.attachments_bar.pack(fill=tk.X, pady=(0, 6))

        self.composer_row = ttk.Frame(self.composer_frame, style="CardInner.TFrame", padding=6)
        self.composer_row.pack(fill=tk.X)

        self.attach_btn = ttk.Button(self.composer_row, text="📎", width=3, command=self._on_attach)
        self.attach_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.input_text = AutoGrowText(
            self.composer_row,
            min_lines=1,
            max_lines=10,
            on_send=self._on_send,
            on_change=self._update_send_button_state,
            bg=self.palette.get("panel"),
            fg=self.palette.get("text"),
            insertbackground=self.palette.get("text"),
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.send_btn = RoundSendButton(self.composer_row, command=self._on_send)
        self.send_btn.pack(side=tk.LEFT, padx=(6, 0))

        self._configure_text_tags()
        self._setup_drag_and_drop()

    def _configure_text_tags(self) -> None:
        self.chat_text.tag_configure("user", foreground=self.palette.get("text"))
        self.chat_text.tag_configure("assistant", foreground=self.palette.get("accent"))
        self.chat_text.tag_configure("system", foreground=self.palette.get("muted"))

    def _setup_drag_and_drop(self) -> None:
        if DND_FILES is None:
            return
        for widget in (self.chat_text, self.composer_frame, self.input_text.text):
            if hasattr(widget, "drop_target_register"):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop_files)

    def _on_drop_files(self, event: tk.Event) -> str:
        if str(self.attach_btn.cget("state")) == str(tk.DISABLED):
            return "break"
        paths = list(self.tk.splitlist(event.data))
        self._add_attachments(paths)
        return "break"

    def _update_send_button_state(self) -> None:
        if str(self.input_text.text.cget("state")) == str(tk.DISABLED):
            self.send_btn.set_enabled(False)
            return
        text = self.input_text.get_text().strip()
        enabled = bool(text) or bool(self.pending_attachments)
        self.send_btn.set_enabled(enabled)

    def _attachment_label(self, item: dict[str, Any]) -> str:
        return item.get("name") or os.path.basename(item.get("path", ""))

    def _format_size(self, size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}GB"

    def refresh_deck_options(self) -> None:
        values = []
        self._deck_map = {}
        for deck in self.app.decks:
            label = f"{deck['id']}: {deck['name']}"
            values.append(label)
            self._deck_map[label] = deck["id"]
        self.deck_combo.configure(values=values)
        if values and self.deck_var.get() in values:
            return
        self.deck_var.set("")
        self._lock_chat()

    def _lock_chat(self) -> None:
        self.input_text.set_state(tk.DISABLED)
        self.send_btn.set_enabled(False)
        self.attach_btn.configure(state=tk.DISABLED)
        self.status_var.set("Выберите колоду")

    def _unlock_chat(self) -> None:
        self.input_text.set_state(tk.NORMAL)
        self.attach_btn.configure(state=tk.NORMAL)
        self.status_var.set("")
        self._update_send_button_state()

    def _on_deck_change(self, _event=None) -> None:
        selection = self.deck_var.get()
        deck_id = self._deck_map.get(selection)
        if deck_id is None:
            self._lock_chat()
            return
        self._unlock_chat()

    def _load_sessions(self) -> None:
        conn = open_db()
        cur = conn.cursor()
        cur.execute("SELECT id, title, created_at FROM chat_sessions ORDER BY created_at ASC;")
        rows = cur.fetchall()
        conn.close()
        self.chats_listbox.delete(0, tk.END)
        for row in rows:
            self.chats_listbox.insert(tk.END, row["title"])
        self._update_new_chat_state(len(rows))
        if rows:
            self.chats_listbox.selection_set(0)
            self._select_session(rows[0]["id"])

    def _update_new_chat_state(self, existing_count: int) -> None:
        plan = self.app.get_pricing_plan()
        if plan == "free" and existing_count >= 1:
            self.new_chat_btn.configure(state=tk.DISABLED)
            self.pro_hint_var.set("Доступно в PRO")
        else:
            self.new_chat_btn.configure(state=tk.NORMAL)
            self.pro_hint_var.set("")

    def _create_new_chat(self) -> None:
        plan = self.app.get_pricing_plan()
        if plan == "free":
            conn = open_db()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS cnt FROM chat_sessions;")
            count = int(cur.fetchone()["cnt"])
            conn.close()
            if count >= 1:
                messagebox.showinfo("Ограничение", "Новый чат доступен в PRO.")
                self._update_new_chat_state(count)
                return
        title = f"Чат {int(time.time())}"
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (title, created_at) VALUES (?, ?);",
            (title, int(time.time())),
        )
        session_id = cur.lastrowid
        conn.commit()
        conn.close()
        self.chats_listbox.insert(tk.END, title)
        self.chats_listbox.selection_clear(0, tk.END)
        self.chats_listbox.selection_set(tk.END)
        self._select_session(session_id)
        self._update_new_chat_state(self.chats_listbox.size())

    def _on_chat_select(self, _event=None) -> None:
        selection = self.chats_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM chat_sessions ORDER BY created_at ASC LIMIT 1 OFFSET ?;",
            (index,),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            self._select_session(row["id"])

    def _select_session(self, session_id: int) -> None:
        self.current_session_id = session_id
        self._load_messages(session_id)

    def _load_messages(self, session_id: int) -> None:
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT role, text, attachments_json, ts FROM chat_messages WHERE session_id = ? ORDER BY ts ASC;",
            (session_id,),
        )
        rows = cur.fetchall()
        conn.close()
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        for row in rows:
            attachments = []
            if row["attachments_json"]:
                try:
                    attachments = json.loads(row["attachments_json"])
                except Exception:
                    attachments = []
            message = Message(role=row["role"], text=row["text"] or "", attachments=attachments)
            self._append_message_to_ui(message)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def _append_message_to_ui(self, message: Message) -> None:
        prefix = ""
        if message.role == "user":
            prefix = "Вы: "
        elif message.role == "assistant":
            prefix = "Ассистент: "
        elif message.role == "system":
            prefix = "Система: "
        payload = message.text or ""
        if message.attachments:
            attachments_info = ", ".join(
                [self._attachment_label(item) for item in message.attachments if self._attachment_label(item)]
            )
            if attachments_info:
                payload = f"{payload}\n[Вложения: {attachments_info}]"
        self.chat_text.insert(tk.END, f"{prefix}{payload}\n\n", message.role)

    def _append_message(self, role: str, text: str, attachments: list[dict[str, Any]] | None = None) -> None:
        if self.current_session_id is None:
            return
        attachments_json = json.dumps(attachments or [], ensure_ascii=False)
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, text, attachments_json, ts) VALUES (?, ?, ?, ?, ?);",
            (self.current_session_id, role, text, attachments_json, int(time.time())),
        )
        conn.commit()
        conn.close()
        message = Message(role=role, text=text, attachments=attachments or [])
        self.chat_text.configure(state=tk.NORMAL)
        self._append_message_to_ui(message)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def _on_attach(self) -> None:
        filetypes = [
            ("Images", "*.png *.jpg *.jpeg *.webp"),
            ("PDF", "*.pdf"),
            ("Text", "*.txt *.md"),
            ("Documents", "*.doc *.docx *.ppt *.pptx *.xls *.xlsx"),
            ("Archives", "*.zip"),
            ("Audio", "*.mp3 *.wav *.m4a *.ogg"),
            ("Video", "*.mp4 *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ]
        paths = filedialog.askopenfilenames(title="Выберите файлы", filetypes=filetypes)
        if not paths:
            return
        self._add_attachments(paths)

    def _add_attachments(self, paths: list[str]) -> None:
        current_paths = {item.get("path") for item in self.pending_attachments}
        total_size = sum(item.get("size", 0) for item in self.pending_attachments)
        for path in paths:
            if not path or path in current_paths:
                continue
            if not os.path.exists(path):
                continue
            size = os.path.getsize(path)
            if total_size + size > self.max_total_attachment_size:
                messagebox.showwarning("Вложения", "Общий размер вложений превышает лимит.")
                break
            name = os.path.basename(path)
            item = {
                "path": path,
                "name": name,
                "size": size,
                "size_label": self._format_size(size),
            }
            self.pending_attachments.append(item)
            current_paths.add(path)
            total_size += size
        self._refresh_attachments_ui()

    def _remove_attachment(self, index: int) -> None:
        if 0 <= index < len(self.pending_attachments):
            self.pending_attachments.pop(index)
        self._refresh_attachments_ui()

    def _refresh_attachments_ui(self) -> None:
        self.attachments_bar.render(self.pending_attachments)
        self._update_send_button_state()

    def _on_send(self) -> None:
        if self.current_session_id is None:
            messagebox.showinfo("Выбор чата", "Выберите чат слева.")
            return
        if not self.deck_var.get():
            self._lock_chat()
            return
        text = self.input_text.get_text().strip()
        if not text and not self.pending_attachments:
            return
        attachments = list(self.pending_attachments)
        self.pending_attachments = []
        self._refresh_attachments_ui()
        self.input_text.clear()
        self._append_message("user", text, attachments)
        self._append_message("assistant", "Получено. (заглушка) Готовлю черновик.")

        self._start_generation(text, attachments)

    def _start_generation(self, text: str, attachments: list[dict[str, Any]]) -> None:
        plan = self.app.get_pricing_plan()
        user_id = self.app.user_id
        deck_context = {"deck_id": self._deck_map.get(self.deck_var.get())}

        def worker():
            try:
                cards = self._generate_cards(text, attachments, deck_context)
                self.engine.check_and_record_generation(user_id, plan, len(cards))
                total_credits = self.engine.estimate_cost(len(cards), plan)
                draft = DraftBatch(
                    draft_id=str(uuid.uuid4()),
                    deck_id=deck_context.get("deck_id") or 0,
                    cards=cards,
                    total_credits=total_credits,
                    created_at=int(time.time()),
                )
                self.after(0, lambda: self._on_generation_success(draft))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_generation_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _generate_cards(self, text: str, attachments: list[dict[str, Any]], deck_context: dict) -> list[DraftCard]:
        for item in attachments:
            path = item.get("path")
            if path:
                return self.engine.generate_from_file(path, deck_context)
        return self.engine.generate_from_text(text, deck_context)

    def _on_generation_success(self, draft: DraftBatch) -> None:
        if self.current_draft and self.current_draft.cards and self.current_draft.deck_id == draft.deck_id:
            self.current_draft.cards.extend(draft.cards)
            self.current_draft.total_credits = self._calculate_total_credits(self.current_draft.cards)
            self.draft_cards = self.current_draft.cards
            self.draft_index = max(0, len(self.draft_cards) - 1)
            self.draft_show_back = False
            self._render_current_draft()
            total_cards = len(self.current_draft.cards)
            self._append_message(
                "assistant",
                f"Добавлено {len(draft.cards)} карточек. Всего: {total_cards}.",
            )
        else:
            draft.total_credits = self._calculate_total_credits(draft.cards)
            self.current_draft = draft
            self.draft_cards = list(draft.cards)
            self.draft_index = 0
            self.draft_show_back = False
            self._render_current_draft()
            self._append_message("assistant", f"Сформирован черновик на {len(draft.cards)} карточек.")
        self._update_save_button_state()

    def _calculate_total_credits(self, cards: list[DraftCard]) -> int:
        plan = self.app.get_pricing_plan()
        return self.engine.estimate_cost(len(cards), plan)

    def _on_generation_error(self, message: str) -> None:
        self._append_message("system", f"Ошибка генерации: {message}")

    def _update_save_button_state(self) -> None:
        if not self.current_draft:
            self._set_save_state(False, 0)
            return
        can_afford = self.credit_manager.can_afford(self.app.user_id, self.current_draft.total_credits)
        enabled = can_afford and bool(self.current_draft.cards)
        self._set_save_state(enabled, self.current_draft.total_credits)

    def _set_save_state(self, enabled: bool, total_credits: int | None = None) -> None:
        label = "Сохранить карточки"
        if total_credits and total_credits > 0:
            label = f"Сохранить карточки ({total_credits} кредитов)"
        self.save_draft_btn.configure(text=label, state=(tk.NORMAL if enabled else tk.DISABLED))

    def _draft_display_id(self, draft: DraftCard, card_payload: dict) -> str:
        meta = getattr(draft, "meta", {}) or {}
        raw_id = meta.get("temp_id") or meta.get("id") or card_payload.get("id")
        if raw_id in (None, -1):
            return "Draft"
        return str(raw_id)

    def _render_current_draft(self) -> None:
        total = len(self.draft_cards)
        if total == 0:
            placeholder = {
                "front": "Черновик пуст. Сначала сформируйте карточки.",
                "back": "",
                "front_rich": None,
                "back_rich": None,
            }
            self.card_view.load_card(
                placeholder,
                status_text="Карточка 0/0 | ID —",
                header_text="Фаза | след. повтор: —",
                show_back=False,
            )
            self.card_view.set_rating_enabled(False)
            self.draft_index_var.set("0/0")
            self.prev_draft_btn.configure(state=tk.DISABLED)
            self.next_draft_btn.configure(state=tk.DISABLED)
            return

        self.draft_index = max(0, min(self.draft_index, total - 1))
        draft = self.draft_cards[self.draft_index]
        card_payload = draft_to_card(draft)
        display_id = self._draft_display_id(draft, card_payload)
        status_text = f"Карточка {self.draft_index + 1}/{total} | ID {display_id}"
        self.card_view.load_card(
            card_payload,
            status_text=status_text,
            header_text="Фаза | след. повтор: —",
            show_back=self.draft_show_back,
        )
        self.card_view.set_rating_enabled(False)

        self.draft_index_var.set(f"{self.draft_index + 1}/{total}")
        self.prev_draft_btn.configure(state=(tk.NORMAL if self.draft_index > 0 else tk.DISABLED))
        self.next_draft_btn.configure(state=(tk.NORMAL if self.draft_index < total - 1 else tk.DISABLED))

    def _prev_draft(self) -> None:
        if self.draft_index > 0:
            self.draft_index -= 1
            self.draft_show_back = False
            self._render_current_draft()

    def _next_draft(self) -> None:
        if self.draft_index < len(self.draft_cards) - 1:
            self.draft_index += 1
            self.draft_show_back = False
            self._render_current_draft()

    def _save_draft(self) -> None:
        if not self.current_draft:
            return
        cost = self.current_draft.total_credits
        deck_id = self._deck_map.get(self.deck_var.get()) or self.current_draft.deck_id
        if not deck_id:
            messagebox.showwarning("Колода", "Выберите колоду для сохранения.")
            return
        conn = open_db()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE;")
            cur.execute(
                "INSERT OR IGNORE INTO credits_balance (user_id, balance) VALUES (?, 0);",
                (self.app.user_id,),
            )
            cur.execute(
                "SELECT balance FROM credits_balance WHERE user_id = ? LIMIT 1;",
                (self.app.user_id,),
            )
            row = cur.fetchone()
            balance = int(row[0]) if row else 0
            if balance < cost:
                conn.rollback()
                messagebox.showwarning("Недостаточно кредитов", "Недостаточно кредитов для сохранения.")
                self._update_save_button_state()
                return
            cur.execute(
                "UPDATE credits_balance SET balance = balance - ? WHERE user_id = ?;",
                (cost, self.app.user_id),
            )
            cur.execute(
                "INSERT INTO credits_ledger (user_id, ts, delta, reason, meta) VALUES (?, ?, ?, ?, ?);",
                (
                    self.app.user_id,
                    int(time.time()),
                    -abs(cost),
                    "Сохранение карточек (чат-бот)",
                    json.dumps({"draft_id": self.current_draft.draft_id}, ensure_ascii=False),
                ),
            )
            saved_count = self._save_draft_to_deck(conn, deck_id, self.current_draft.cards)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            messagebox.showerror("Ошибка", str(exc))
            return
        finally:
            conn.close()
        self.app.refresh_balance_ui()
        self.current_draft = None
        self.draft_cards = []
        self.draft_index = 0
        self.draft_show_back = False
        self._render_current_draft()
        self._set_save_state(False, 0)
        self._append_message("system", f"Сохранено {saved_count} карточек.")

    def _save_draft_to_deck(self, conn, deck_id: int, cards: list[DraftCard]) -> int:
        saved = 0
        now_ts = int(time.time())
        for card in cards:
            fields = {
                "word": card.front,
                "translation": card.back,
                "notes": "",
                "example": "",
                "front": card.front,
                "back": card.back,
            }
            srs_defaults = {
                "state": "new",
                "due": now_ts,
                "interval": 0,
                "ease": 250,
                "reps": 0,
                "lapses": 0,
                "step_index": 0,
                "phase": 1,
            }
            mode = {
                "skip_existing": False,
                "reset_srs": False,
                "state": "new",
                "source": "chatbot",
            }
            tags_value = " ".join(card.tags)
            upsert_note_and_cards(conn, deck_id, None, fields, tags_value, srs_defaults, mode)
            saved += 1
        return saved
