from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ai_card_pipeline import AICardPipeline
from card_preview_widget import CardPreviewWidget
from rag_web_search import search_text

try:
    from chat_bot_tab import ChatBotTab
except Exception:
    ChatBotTab = None


class AIAddCardWorkspace(tk.Toplevel):
    def __init__(self, app: tk.Misc) -> None:
        super().__init__(app)
        self.app = app
        self.title("AI добавление карточек")
        self.geometry("980x720")
        self.transient(app)

        self.deck_id = getattr(app, "selected_deck_id", None)
        self.pipeline = AICardPipeline(app=app, deck_id=self.deck_id)
        self.source_path: str | None = None
        self.cards: list[dict] = []
        self._busy = False

        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        self.preview = CardPreviewWidget(root)
        self.preview.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.LabelFrame(root, text="Источник и AI", padding=8)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.prompt_text = tk.Text(bottom, height=6, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.X)

        chat_wrap = ttk.Frame(bottom)
        chat_wrap.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        if ChatBotTab is not None:
            try:
                self.chat_tab = ChatBotTab(chat_wrap, app=app)
                self.chat_tab.pack(fill=tk.BOTH, expand=True)
            except Exception:
                logging.exception("ChatBotTab embed failed")
                ttk.Label(chat_wrap, text="Чат-бот недоступен. Проверьте chat_bot_tab.py").pack(anchor="w")
        else:
            ttk.Label(chat_wrap, text="Чат-бот недоступен. Проверьте chat_bot_tab.py").pack(anchor="w")

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(actions, text="Загрузить файл", command=self._pick_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Извлечь текст", command=self._extract_text).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Сгенерировать карточки", command=self._generate_cards).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="🔍 Найти материалы", command=self._search_web).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Сгенерировать картинку", command=self._generate_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Сохранить в ознакомление", command=self._save_cards).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Ручной редактор", command=self._open_manual_editor).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Очистить", command=self._clear).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Отмена", command=self.destroy).pack(side=tk.RIGHT, padx=4)

        status_row = ttk.Frame(root)
        status_row.pack(fill=tk.X, pady=(8, 0))
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(status_row, mode="indeterminate")
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self._busy = busy
        if status:
            self.status_var.set(status)
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _run_bg(self, status: str, fn, on_success=None) -> None:
        if self._busy:
            return
        self._set_busy(True, status)

        def runner() -> None:
            try:
                result = fn()
            except Exception as exc:
                logging.exception("AI workspace action failed")
                self.after(0, lambda: self._finish_error(exc))
                return
            self.after(0, lambda: self._finish_ok(result, on_success))

        threading.Thread(target=runner, daemon=True).start()

    def _finish_ok(self, result, on_success=None) -> None:
        self._set_busy(False)
        if on_success:
            on_success(result)

    def _finish_error(self, exc: Exception) -> None:
        self._set_busy(False, "Ошибка")
        try:
            messagebox.showerror("Ошибка", str(exc), parent=self)
        except Exception:
            pass

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(parent=self)
        if path:
            self.source_path = path
            self.status_var.set(f"Источник: {os.path.basename(path)}")

    def _extract_text(self) -> None:
        if not self.source_path:
            messagebox.showwarning("Источник", "Сначала выберите файл", parent=self)
            return

        def task():
            raw = self.pipeline.extract_text_from_source(self.source_path)
            return self.pipeline.clean_text(raw)

        self._run_bg("Извлечение текста...", task, on_success=self._on_text_ready)

    def _on_text_ready(self, text: str) -> None:
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", text[:12000])
        self.status_var.set("Текст извлечён")

    def _search_web(self) -> None:
        query = self.prompt_text.get("1.0", "end-1c").strip()
        if not query:
            messagebox.showwarning("Поиск", "Введите тему или вопрос", parent=self)
            return

        self._run_bg("Поиск материалов...", lambda: search_text(query), on_success=self._on_web_text)

    def _on_web_text(self, text: str) -> None:
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", text[:12000])
        self.status_var.set("Материалы получены")

    def _generate_cards(self) -> None:
        text = self.prompt_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Пусто", "Нет текста для генерации карточек", parent=self)
            return

        def task():
            cleaned = self.pipeline.clean_text(text)
            chunks = self.pipeline.split_into_chunks(cleaned)
            blocks = self.pipeline.split_into_semantic_blocks(chunks)
            facts = self.pipeline.extract_key_facts_terms_dates_formulas(blocks)
            cards = self.pipeline.generate_card_candidates(facts)
            cards = self.pipeline.filter_and_improve_cards(cards)
            for c in cards:
                c["image_prompt"] = self.pipeline.generate_image_prompt(c)
            return cards

        self._run_bg("Генерация карточек...", task, on_success=self._on_cards_ready)

    def _on_cards_ready(self, cards: list[dict]) -> None:
        self.cards = cards
        if cards:
            self.preview.set_card(cards[0])
        self.status_var.set(f"Сгенерировано карточек: {len(cards)}")

    def _generate_image(self) -> None:
        if not self.cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return

        def task():
            updated = []
            for card in self.cards:
                updated.append(self.pipeline.generate_card_image(card))
            return updated

        self._run_bg("Генерация изображений...", task, on_success=self._on_images_ready)

    def _on_images_ready(self, cards: list[dict]) -> None:
        self.cards = cards
        if cards:
            self.preview.set_card(cards[0])
        self.status_var.set("Генерация изображений завершена")

    def _save_cards(self) -> None:
        if not self.cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return
        self.pipeline.deck_id = getattr(self.app, "selected_deck_id", None)
        self._run_bg("Сохранение карточек...", lambda: self.pipeline.save_cards_to_overview(self.cards), on_success=self._on_saved)

    def _on_saved(self, saved_count: int) -> None:
        self.status_var.set(f"Сохранено: {saved_count}")
        try:
            if hasattr(self.app, "refresh_decks"):
                self.app.refresh_decks()
        except Exception:
            logging.exception("refresh_decks failed")

    def _open_manual_editor(self) -> None:
        try:
            self.app.add_card_window()
        finally:
            self.destroy()

    def _clear(self) -> None:
        self.cards = []
        self.source_path = None
        self.prompt_text.delete("1.0", tk.END)
        self.preview.set_card(None)
        self.status_var.set("Очищено")
