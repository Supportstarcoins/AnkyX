from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from ai_answer_grader import AIAnswerGrader
from ai_srs_adapter import AISRSAdapter


class AIReviewController:
    def __init__(self, app=None) -> None:
        self.app = app
        self.grader = AIAnswerGrader()
        self.srs = AISRSAdapter()

    def grade(self, card, user_answer, answer_time_ms):
        result = self.grader.grade_answer(card, user_answer, answer_time_ms)
        result["answer_time_ms"] = int(answer_time_ms or 0)
        card_id = (card or {}).get("id")
        if card_id is not None:
            try:
                self.srs.apply_ai_grade_to_card(card_id, result, self.app)
            except Exception:
                pass
        return result


class AIReviewPanel(ttk.Frame):
    def __init__(self, parent, app=None, on_next_card=None, on_show_answer=None):
        super().__init__(parent)
        self.controller = AIReviewController(app=app)
        self.on_next_card = on_next_card
        self.on_show_answer = on_show_answer

        self.current_card = None
        self.answer_started_at = None
        self.last_grade_result = None

        self.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="AI-проверка готова")

        ttk.Label(self, text="AI-проверка ответа").grid(row=0, column=0, sticky="w")
        self.history = tk.Text(self, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.history.grid(row=1, column=0, sticky="ew", pady=(4, 4))

        input_row = ttk.Frame(self)
        input_row.grid(row=2, column=0, sticky="ew")
        input_row.columnconfigure(0, weight=1)
        self.answer_input = tk.Text(input_row, height=3, wrap=tk.WORD)
        self.answer_input.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(btns, text="Проверить ответ", command=self.submit_answer).pack(side=tk.LEFT)
        ttk.Button(btns, text="Показать ответ", command=self.show_correct_answer).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Следующая карточка", command=self.go_next_card).pack(side=tk.LEFT)

        ttk.Label(self, textvariable=self.status_var).grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.grid(row=5, column=0, sticky="ew", pady=(2, 0))

    def set_card(self, card_data):
        self.current_card = dict(card_data or {})
        self.clear_chat()
        self.start_answer_timer()

    def start_answer_timer(self):
        self.answer_started_at = time.time()

    def submit_answer(self):
        if not self.current_card:
            return
        user_answer = self.answer_input.get("1.0", "end-1c").strip()
        started = self.answer_started_at or time.time()
        elapsed_ms = int((time.time() - started) * 1000)
        self.progress.start(10)
        self.after(10, lambda: self._grade_now(user_answer, elapsed_ms))

    def _grade_now(self, user_answer, elapsed_ms):
        result = self.controller.grade(self.current_card, user_answer, elapsed_ms)
        self._on_answer_graded(result)

    def _on_answer_graded(self, result):
        self.progress.stop()
        self.last_grade_result = result
        grade = result.get("grade", "unknown")
        msg = (
            f"Оценка: {grade} | score={result.get('score')}\n"
            f"Комментарий: {result.get('short_feedback')}\n"
            f"Ошибка: {result.get('mistake_type')}\n"
            f"Аналогия: {result.get('analogy')}\n"
            f"Уточняющий вопрос: {result.get('follow_up_question')}\n"
        )
        self._append("AI", msg)
        self.status_var.set("Ответ проверен")
        if callable(self.on_show_answer):
            try:
                self.on_show_answer()
            except Exception:
                pass

    def show_correct_answer(self):
        if not self.current_card:
            return
        self._append("Карточка", self.current_card.get("back", ""))
        if callable(self.on_show_answer):
            try:
                self.on_show_answer()
            except Exception:
                pass

    def clear_chat(self):
        self.answer_input.delete("1.0", tk.END)
        self.history.configure(state=tk.NORMAL)
        self.history.delete("1.0", tk.END)
        self.history.configure(state=tk.DISABLED)
        self.last_grade_result = None
        self.status_var.set("Введите ответ...")

    def go_next_card(self):
        self.clear_chat()
        if callable(self.on_next_card):
            self.on_next_card()

    def _append(self, role, text):
        self.history.configure(state=tk.NORMAL)
        self.history.insert(tk.END, f"{role}:\n{text}\n\n")
        self.history.see(tk.END)
        self.history.configure(state=tk.DISABLED)


def attach_ai_review_panel(parent, app=None, callbacks=None):
    callbacks = callbacks or {}
    panel = AIReviewPanel(
        parent,
        app=app,
        on_next_card=callbacks.get("on_next_card"),
        on_show_answer=callbacks.get("on_show_answer"),
    )
    panel.pack(fill=tk.X)
    return panel
