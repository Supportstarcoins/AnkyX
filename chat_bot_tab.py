from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from card_widget import CardWidget
from chatbot_models import ChatSession, DraftBatch, DraftCard, Message
from credit_manager import CreditManager
from csv_importer import upsert_note_and_cards
from db_connect import open_db
from mock_ai_engine import MockAIEngine


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


class ChatBotTab(ttk.Frame):
    def __init__(self, master: tk.Widget, app) -> None:
        super().__init__(master, style="Surface.TFrame")
        self.app = app
        self.palette = app.palette
        self.engine = MockAIEngine()
        self.credit_manager = CreditManager()
        self.current_session_id: int | None = None
        self.current_draft: DraftBatch | None = None
        self.attachments: list[dict[str, Any]] = []
        self._deck_map: dict[str, int] = {}

        self._ensure_chat_tables()
        self._build_ui()
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

        self.sticky_frame = ttk.Frame(right_frame, style="Card.TFrame", height=200, padding=6)
        self.sticky_frame.pack(fill=tk.X)
        self.sticky_frame.pack_propagate(False)

        self.card_preview = CardPreviewWidget(self.sticky_frame, self.palette, self._save_draft)
        self.card_preview.pack(fill=tk.BOTH, expand=True)

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

        input_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=8)
        input_frame.pack(fill=tk.X)

        self.input_var = tk.StringVar(value="")
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.send_btn = ttk.Button(input_frame, text="Send", command=self._on_send)
        self.send_btn.pack(side=tk.LEFT, padx=6)

        self.attach_btn = ttk.Button(input_frame, text="Attach", command=self._on_attach)
        self.attach_btn.pack(side=tk.LEFT)

        self.link_btn = ttk.Button(input_frame, text="Paste Link", command=self._on_paste_link)
        self.link_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.attachments_var = tk.StringVar(value="")
        self.attachments_label = ttk.Label(right_frame, textvariable=self.attachments_var, style="Muted.TLabel")
        self.attachments_label.pack(anchor=tk.W, pady=(4, 0))

        self._configure_text_tags()

    def _configure_text_tags(self) -> None:
        self.chat_text.tag_configure("user", foreground=self.palette.get("text"))
        self.chat_text.tag_configure("assistant", foreground=self.palette.get("accent"))
        self.chat_text.tag_configure("system", foreground=self.palette.get("muted"))

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
        self.input_entry.configure(state=tk.DISABLED)
        self.send_btn.configure(state=tk.DISABLED)
        self.attach_btn.configure(state=tk.DISABLED)
        self.link_btn.configure(state=tk.DISABLED)
        self.status_var.set("Выберите колоду")

    def _unlock_chat(self) -> None:
        self.input_entry.configure(state=tk.NORMAL)
        self.send_btn.configure(state=tk.NORMAL)
        self.attach_btn.configure(state=tk.NORMAL)
        self.link_btn.configure(state=tk.NORMAL)
        self.status_var.set("")

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
            attachments_info = ", ".join([item.get("label", "вложение") for item in message.attachments])
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
        path = filedialog.askopenfilename(title="Выберите файл")
        if not path:
            return
        self.attachments.append({"type": "file", "value": path, "label": path.split("/")[-1]})
        self._refresh_attachments_label()

    def _on_paste_link(self) -> None:
        url = simpledialog.askstring("Ссылка", "Вставьте ссылку (например, YouTube URL)")
        if not url:
            return
        self.attachments.append({"type": "url", "value": url, "label": url})
        self._refresh_attachments_label()

    def _refresh_attachments_label(self) -> None:
        if not self.attachments:
            self.attachments_var.set("")
            return
        labels = ", ".join(item.get("label", "") for item in self.attachments)
        self.attachments_var.set(f"Вложения: {labels}")

    def _on_send(self) -> None:
        if self.current_session_id is None:
            messagebox.showinfo("Выбор чата", "Выберите чат слева.")
            return
        if not self.deck_var.get():
            self._lock_chat()
            return
        text = self.input_var.get().strip()
        if not text and not self.attachments:
            return
        attachments = list(self.attachments)
        self.attachments = []
        self._refresh_attachments_label()
        self.input_var.set("")
        self._append_message("user", text, attachments)

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
            if item.get("type") == "file":
                return self.engine.generate_from_file(item.get("value", ""), deck_context)
            if item.get("type") == "url":
                url = item.get("value", "")
                if "youtube" in url or "youtu.be" in url:
                    return self.engine.generate_from_youtube(url, "ru", deck_context)
                return self.engine.generate_from_text(url, deck_context)
        return self.engine.generate_from_text(text, deck_context)

    def _on_generation_success(self, draft: DraftBatch) -> None:
        if self.current_draft and self.current_draft.cards and self.current_draft.deck_id == draft.deck_id:
            self.current_draft.cards.extend(draft.cards)
            self.current_draft.total_credits = self._calculate_total_credits(self.current_draft.cards)
            self.card_preview.append_cards(draft.cards, select_last=True, total_credits=self.current_draft.total_credits)
            total_cards = len(self.current_draft.cards)
            self._append_message(
                "assistant",
                f"Добавлено {len(draft.cards)} карточек. Всего: {total_cards}.",
            )
        else:
            draft.total_credits = self._calculate_total_credits(draft.cards)
            self.current_draft = draft
            self.card_preview.set_cards(draft.cards, start_index=0, total_credits=draft.total_credits)
            self._append_message("assistant", f"Сформирован черновик на {len(draft.cards)} карточек.")
        self._update_save_button_state()

    def _calculate_total_credits(self, cards: list[DraftCard]) -> int:
        plan = self.app.get_pricing_plan()
        return self.engine.estimate_cost(len(cards), plan)

    def _on_generation_error(self, message: str) -> None:
        self._append_message("system", f"Ошибка генерации: {message}")

    def _update_save_button_state(self) -> None:
        if not self.current_draft:
            self.card_preview.set_save_state(False, 0)
            return
        can_afford = self.credit_manager.can_afford(self.app.user_id, self.current_draft.total_credits)
        enabled = can_afford and bool(self.current_draft.cards)
        self.card_preview.set_save_state(enabled, self.current_draft.total_credits)

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
        self.card_preview.clear()
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
