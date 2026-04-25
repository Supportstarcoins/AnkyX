from __future__ import annotations

import logging
import os
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ai_card_pipeline import AICardPipeline
from card_preview_widget import CardPreviewWidget
from rag_web_search import RagWebSearch

try:
    from chat_bot_tab import ChatBotTab
except Exception:
    ChatBotTab = None


class AIAddCardWorkspace(tk.Toplevel):
    def __init__(self, app: tk.Misc) -> None:
        super().__init__(app)
        self.app = app
        self.title("AI добавление карточек")
        self.geometry("1120x820")
        self.minsize(980, 720)
        self.transient(app)

        self.deck_id = getattr(app, "selected_deck_id", None)
        self.pipeline = AICardPipeline(app=app, deck_id=self.deck_id)
        self.web_search = RagWebSearch()

        self.source_path: str | None = None
        self.generated_cards: list[dict] = []
        self.current_card_index = 0
        self.auto_generate_image_after_card = False
        self._busy = False
        self._last_chat_answer = ""

        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        workspace_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(workspace_tab, text="AI Workspace")
        self._build_workspace_tab(workspace_tab)
        self._build_advanced_chat_tab()

    def _build_workspace_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(4, weight=1)

        preview_wrap = ttk.LabelFrame(parent, text="Предпросмотр карточки", padding=8)
        preview_wrap.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        preview_wrap.rowconfigure(0, weight=1)
        preview_wrap.columnconfigure(0, weight=1)
        preview_wrap.configure(height=240)
        preview_wrap.grid_propagate(False)

        self.preview = CardPreviewWidget(preview_wrap)
        self.preview.grid(row=0, column=0, sticky="nsew")

        source_wrap = ttk.LabelFrame(parent, text="Источник / промт", padding=8)
        source_wrap.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        source_wrap.columnconfigure(0, weight=1)

        self.prompt_text = tk.Text(source_wrap, height=7, wrap=tk.WORD)
        self.prompt_text.grid(row=0, column=0, sticky="ew")

        actions_wrap = ttk.LabelFrame(parent, text="Действия", padding=8)
        actions_wrap.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for col in range(5):
            actions_wrap.columnconfigure(col, weight=1)

        ttk.Button(actions_wrap, text="Загрузить файл", command=self._pick_file).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Извлечь текст", command=self._extract_text).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Сгенерировать карточки", command=self.generate_cards_from_input).grid(row=0, column=2, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="🔍 Найти материалы", command=self._search_web).grid(row=0, column=3, sticky="ew", padx=3, pady=3)

        ttk.Button(actions_wrap, text="Сгенерировать картинку", command=self.generate_image_for_current_card).grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Сохранить в ознакомление", command=self.save_current_card_to_overview).grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Ручной редактор", command=self.open_manual_editor).grid(row=1, column=2, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Очистить", command=self.clear_workspace).grid(row=1, column=3, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Отмена", command=self.destroy).grid(row=1, column=4, sticky="ew", padx=3, pady=3)

        status_row = ttk.Frame(parent)
        status_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        status_row.columnconfigure(1, weight=1)
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(status_row, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(status_row, mode="indeterminate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        cards_wrap = ttk.LabelFrame(parent, text="Сгенерированные карточки", padding=8)
        cards_wrap.grid(row=4, column=0, sticky="nsew", padx=(0, 6))
        cards_wrap.columnconfigure(0, weight=1)
        cards_wrap.rowconfigure(2, weight=1)

        nav_row = ttk.Frame(cards_wrap)
        nav_row.grid(row=0, column=0, sticky="ew")
        nav_row.columnconfigure(1, weight=1)
        ttk.Button(nav_row, text="←", width=4, command=self.prev_card).grid(row=0, column=0, padx=(0, 6))
        self.cards_counter_var = tk.StringVar(value="0/0")
        ttk.Label(nav_row, textvariable=self.cards_counter_var).grid(row=0, column=1, sticky="w")
        ttk.Button(nav_row, text="→", width=4, command=self.next_card).grid(row=0, column=2, padx=(6, 0))

        self.cards_listbox = tk.Listbox(cards_wrap, height=6, exportselection=False)
        self.cards_listbox.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        self.cards_listbox.bind("<<ListboxSelect>>", self._on_card_select)

        details = ttk.Frame(cards_wrap)
        details.grid(row=2, column=0, sticky="nsew")
        details.columnconfigure(0, weight=1)

        ttk.Label(details, text="Текущий вопрос:").grid(row=0, column=0, sticky="w")
        self.current_front_text = tk.Text(details, height=3, wrap=tk.WORD)
        self.current_front_text.grid(row=1, column=0, sticky="ew")

        ttk.Label(details, text="Текущий ответ:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.current_back_text = tk.Text(details, height=4, wrap=tk.WORD)
        self.current_back_text.grid(row=3, column=0, sticky="ew")

        action_row = ttk.Frame(details)
        action_row.grid(row=4, column=0, sticky="e", pady=(6, 0))
        ttk.Button(action_row, text="Удалить текущую", command=self.delete_current_card).pack(side=tk.RIGHT)
        ttk.Button(action_row, text="Сохранить все", command=self.save_all_cards_to_overview).pack(side=tk.RIGHT, padx=(0, 6))

        chat_wrap = ttk.LabelFrame(parent, text="Компактный чат", padding=8)
        chat_wrap.grid(row=4, column=1, sticky="nsew")
        chat_wrap.columnconfigure(0, weight=1)
        chat_wrap.rowconfigure(0, weight=1)

        self.chat_history = tk.Text(chat_wrap, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_history.grid(row=0, column=0, sticky="nsew")

        chat_input_row = ttk.Frame(chat_wrap)
        chat_input_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        chat_input_row.columnconfigure(0, weight=1)
        self.chat_input = tk.Text(chat_input_row, height=3, wrap=tk.WORD)
        self.chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(chat_input_row, text="Отправить", command=self._chat_send).grid(row=0, column=1, sticky="ns")
        ttk.Button(chat_wrap, text="Использовать ответ как источник", command=self._use_chat_answer_as_source).grid(row=2, column=0, sticky="e", pady=(6, 0))

    def _build_advanced_chat_tab(self) -> None:
        advanced_chat_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(advanced_chat_tab, text="Расширенный чат")
        if ChatBotTab is None:
            ttk.Label(advanced_chat_tab, text="Чат-бот недоступен. Проверьте chat_bot_tab.py.").pack(anchor="w")
            return
        try:
            self.chat_tab = ChatBotTab(advanced_chat_tab, app=self.app)
            self.chat_tab.pack(fill=tk.BOTH, expand=True)
        except Exception:
            logging.exception("ChatBotTab embed failed")
            ttk.Label(advanced_chat_tab, text="Чат-бот недоступен. Проверьте chat_bot_tab.py.").pack(anchor="w")

    def run_in_background(self, worker, on_success=None, on_error=None):
        if self._busy:
            return
        self._busy = True
        self.progress.start(10)

        def _runner():
            try:
                result = worker()
            except Exception as exc:
                def _err():
                    self._busy = False
                    self.progress.stop()
                    self.status_var.set("Ошибка")
                    if on_error:
                        on_error(exc)
                    else:
                        try:
                            messagebox.showerror("Ошибка", str(exc), parent=self)
                        except Exception:
                            pass
                self.after(0, _err)
                return

            def _ok():
                self._busy = False
                self.progress.stop()
                if on_success:
                    on_success(result)
            self.after(0, _ok)

        threading.Thread(target=_runner, daemon=True).start()

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(parent=self)
        if path:
            self.source_path = path
            self.status_var.set(f"Источник: {os.path.basename(path)}")

    def _extract_text(self) -> None:
        if not self.source_path:
            messagebox.showwarning("Источник", "Сначала выберите файл", parent=self)
            return

        def worker():
            raw = self.pipeline.extract_text_from_source(self.source_path)
            return self.pipeline.clean_text(raw)

        self.status_var.set("Извлекаю текст...")
        self.run_in_background(worker, on_success=self._on_text_ready)

    def _on_text_ready(self, text: str) -> None:
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", text[:12000])
        self.status_var.set("Текст извлечён")

    def _search_web(self) -> None:
        query = self.prompt_text.get("1.0", "end-1c").strip()
        if not query:
            messagebox.showwarning("Поиск", "Введите тему или вопрос", parent=self)
            return
        self.status_var.set("Ищу материалы...")
        self.run_in_background(lambda: self.web_search.search_and_extract(query), on_success=self._on_web_text)

    def _on_web_text(self, text: str) -> None:
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", text[:12000])
        self.status_var.set("Материалы получены")

    def generate_cards_from_input(self) -> None:
        text = self.prompt_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Пусто", "Нет текста для генерации карточек", parent=self)
            return

        self.status_var.set("Генерирую карточки...")
        self.run_in_background(
            lambda: self.pipeline.run_pipeline(text=text, source=self.source_path),
            on_success=self._on_cards_generated,
        )

    def _on_cards_generated(self, cards) -> None:
        self.generated_cards = list(cards or [])
        self.current_card_index = 0
        self.show_current_card()
        if self.generated_cards:
            self.status_var.set(f"Сгенерировано карточек: {len(self.generated_cards)}")
            if self.auto_generate_image_after_card:
                self.auto_generate_image_after_card = False
                self.generate_image_for_current_card()
        else:
            self.status_var.set("Карточки не сгенерированы. Попробуйте уточнить тему.")

    def show_current_card(self) -> None:
        self.cards_listbox.delete(0, tk.END)
        for idx, card in enumerate(self.generated_cards, start=1):
            front = (card.get("front") or "").strip().replace("\n", " ")
            self.cards_listbox.insert(tk.END, f"{idx}. {front[:90] or '(без вопроса)'}")

        total = len(self.generated_cards)
        counter = f"{self.current_card_index + 1}/{total}" if total else "0/0"
        self.cards_counter_var.set(counter)

        self.current_front_text.delete("1.0", tk.END)
        self.current_back_text.delete("1.0", tk.END)
        if not total:
            self.preview.clear()
            return

        self.current_card_index = max(0, min(self.current_card_index, total - 1))
        card = self.generated_cards[self.current_card_index]
        self.cards_listbox.selection_clear(0, tk.END)
        self.cards_listbox.selection_set(self.current_card_index)
        self.preview.update_preview(card)
        self.current_front_text.insert("1.0", card.get("front", ""))
        self.current_back_text.insert("1.0", card.get("back", ""))

    def next_card(self) -> None:
        if not self.generated_cards:
            return
        self.current_card_index = (self.current_card_index + 1) % len(self.generated_cards)
        self.show_current_card()

    def prev_card(self) -> None:
        if not self.generated_cards:
            return
        self.current_card_index = (self.current_card_index - 1) % len(self.generated_cards)
        self.show_current_card()

    def delete_current_card(self) -> None:
        if not self.generated_cards:
            return
        self.generated_cards.pop(self.current_card_index)
        if self.current_card_index >= len(self.generated_cards):
            self.current_card_index = max(0, len(self.generated_cards) - 1)
        self.show_current_card()
        self.status_var.set("Текущая карточка удалена")

    def generate_image_for_current_card(self) -> None:
        if not self.generated_cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return
        idx = self.current_card_index

        def worker():
            card = dict(self.generated_cards[idx])
            if not card.get("image_prompt"):
                card["image_prompt"] = self.pipeline.generate_image_prompt(card)
            return self.pipeline.generate_card_image(card)

        self.status_var.set("Генерирую изображение...")
        self.run_in_background(worker, on_success=self._on_image_generated)

    def _on_image_generated(self, card) -> None:
        if self.generated_cards:
            self.generated_cards[self.current_card_index] = card
        self.show_current_card()
        status = ((card.get("metadata") or {}).get("image_status") or "").strip()
        self.status_var.set(status or "Генерация изображения завершена")

    def save_current_card_to_overview(self) -> None:
        if not self.generated_cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return
        self.pipeline.deck_id = getattr(self.app, "selected_deck_id", None)
        current_card = self.generated_cards[self.current_card_index]
        self.status_var.set("Сохраняю карточку...")
        self.run_in_background(lambda: self.pipeline.save_cards_to_overview([current_card]), on_success=self._on_saved)

    def save_all_cards_to_overview(self) -> None:
        if not self.generated_cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return
        self.pipeline.deck_id = getattr(self.app, "selected_deck_id", None)
        self.status_var.set("Сохраняю все карточки...")
        self.run_in_background(lambda: self.pipeline.save_cards_to_overview(self.generated_cards), on_success=self._on_saved)

    def _on_saved(self, saved_count: int) -> None:
        self.status_var.set(f"Сохранено в ознакомление: {saved_count}")
        try:
            if hasattr(self.app, "refresh_decks"):
                self.app.refresh_decks()
        except Exception:
            logging.exception("refresh_decks failed")

    def clear_workspace(self) -> None:
        self.generated_cards = []
        self.current_card_index = 0
        self.auto_generate_image_after_card = False
        self.source_path = None
        self.prompt_text.delete("1.0", tk.END)
        self.chat_input.delete("1.0", tk.END)
        self._last_chat_answer = ""
        self.show_current_card()
        self.status_var.set("Очищено")

    def open_manual_editor(self) -> None:
        try:
            self.app.add_card_window()
        finally:
            self.destroy()

    def handle_chat_command(self, text: str) -> bool:
        cleaned = (text or "").strip()
        lowered = cleaned.lower()
        if not any(k in lowered for k in ("сгенерируй карточку", "создай карточку")):
            return False

        need_image = any(k in lowered for k in ("с картинкой", "с изображением", "нарисуй", "сгенерируй изображение"))
        topic = cleaned
        topic = re.sub(r"(?i)^\s*(сгенерируй|создай)\s+карточк[ауи]\s*", "", topic).strip(" :,-")
        topic = re.sub(r"(?i)^с\s+(картинкой|изображением)\b", "", topic).strip(" :,-")
        topic = re.sub(r"(?i)^про\b", "", topic).strip(" :,-")
        if not topic:
            topic = "тема карточки"

        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", topic)
        self.auto_generate_image_after_card = need_image
        self.status_var.set("Распознана команда генерации карточки")
        self.generate_cards_from_input()
        return True

    def _chat_send(self) -> None:
        text = self.chat_input.get("1.0", "end-1c").strip()
        if not text:
            return
        self.chat_input.delete("1.0", tk.END)
        self._chat_append("Вы", text)
        if self.handle_chat_command(text):
            return
        self.status_var.set("Чат: ищу ответ...")
        self.run_in_background(lambda: self.web_search.search_and_extract(text), on_success=self._chat_on_answer)

    def _chat_on_answer(self, answer: str) -> None:
        self._last_chat_answer = answer or ""
        self._chat_append("AI", self._last_chat_answer or "Пустой ответ")
        self.status_var.set("Готово")

    def _chat_append(self, author: str, text: str) -> None:
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"{author}:\n{text}\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state=tk.DISABLED)

    def _use_chat_answer_as_source(self) -> None:
        if not self._last_chat_answer:
            messagebox.showwarning("Чат", "Нет ответа для вставки", parent=self)
            return
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", self._last_chat_answer)
        self.status_var.set("Ответ чата вставлен в источник")

    def _on_card_select(self, _event=None) -> None:
        if not self.generated_cards:
            return
        selection = self.cards_listbox.curselection()
        if not selection:
            return
        self.current_card_index = selection[0]
        self.show_current_card()
