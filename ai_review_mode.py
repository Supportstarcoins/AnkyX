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

    def grade(self, card, user_answer, answer_time_ms, apply_srs=True):
        result = self.grader.grade_answer(card, user_answer, answer_time_ms)
        result["answer_time_ms"] = int(result.get("answer_time_ms") or answer_time_ms or 0)
        if apply_srs:
            self.apply_srs(card, result)
        return result

    def apply_srs(self, card, result):
        card_id = (card or {}).get("id")
        if card_id is not None:
            try:
                self.srs.apply_ai_grade_to_card(card_id, result, self.app)
            except Exception:
                pass


class AIReviewPanel(ttk.Frame):
    def __init__(self, parent, app=None, on_next_card=None, on_show_answer=None):
        super().__init__(parent)
        self.controller = AIReviewController(app=app)
        self.on_next_card = on_next_card
        self.on_show_answer = on_show_answer

        self.current_card = None
        self.answer_started_at = None
        self.awaiting_follow_up = False
        self.last_follow_up_question = ""
        self.last_user_answer = ""
        self.last_grade_result = None
        self.srs_applied_for_current_card = False

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
        self.submit_btn = ttk.Button(btns, text="Проверить ответ", command=self.submit_answer)
        self.submit_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="Показать ответ", command=self.show_correct_answer).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Следующая карточка", command=self.go_next_card).pack(side=tk.LEFT)

        ttk.Label(self, textvariable=self.status_var).grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.grid(row=5, column=0, sticky="ew", pady=(2, 0))
        self.answer_input.bind("<Control-Return>", self._on_ctrl_enter)
        self.answer_input.bind("<Control-KP_Enter>", self._on_ctrl_enter)

    def set_card(self, card_data):
        self.current_card = dict(card_data or {})
        self.clear_chat()
        self.start_answer_timer()

    def start_answer_timer(self):
        self.answer_started_at = time.time()

    def submit_answer(self):
        if not self.current_card:
            return
        if self.awaiting_follow_up:
            return self.submit_follow_up_answer()
        user_answer = self.answer_input.get("1.0", "end-1c").strip()
        if not user_answer:
            self.status_var.set("Введите ответ перед проверкой")
            return
        self.last_user_answer = user_answer
        self._append("Вы", user_answer or "—")
        started = self.answer_started_at or time.time()
        elapsed_ms = int((time.time() - started) * 1000)
        self.progress.start(10)
        self.after(10, lambda: self._grade_now(user_answer, elapsed_ms))

    def _grade_now(self, user_answer, elapsed_ms):
        result = self.controller.grade(self.current_card, user_answer, elapsed_ms, apply_srs=False)
        self._on_answer_graded(result)

    def _pick_points_for_ui(self, result: dict, base_key: str) -> list[str]:
        human = result.get(f"{base_key}_human") or []
        raw = result.get(base_key) or []
        items = human if human else raw
        cleaned = [str(x).strip() for x in items if str(x).strip()]
        return cleaned[:4]

    def _join_points(self, points: list[str]) -> str:
        return "; ".join(points) if points else ""

    def _filter_missing_points_for_ui(self, points: list[str]) -> list[str]:
        bad_markers = ("что нужно добавить", "какой", "какая", "какие", "уточняющий вопрос")
        cleaned: list[str] = []
        for raw in points or []:
            line = str(raw or "").strip()
            if not line:
                continue
            low = line.lower()
            if "?" in low or any(marker in low for marker in bad_markers):
                continue
            cleaned.append(line)
        return cleaned

    def _pick_analogy_for_ui(self, result: dict, warning_text: str = "", missing_text: str = "") -> str:
        analogy = str((result or {}).get("analogy_human") or (result or {}).get("analogy") or "").strip()
        if not analogy:
            return ""
        analogy_l = analogy.lower()
        if warning_text and analogy_l == warning_text.lower().strip():
            return ""
        if missing_text and analogy_l == missing_text.lower().strip():
            return ""
        return analogy

    def _first_point(self, points: list[str]) -> str:
        for p in points or []:
            line = str(p or "").strip()
            if line:
                return line
        return ""

    def _on_answer_graded(self, result):
        self.progress.stop()
        self.last_grade_result = result
        grade = result.get("grade", "unknown")
        answer_time_ms = int(result.get("answer_time_ms") or 0)
        matched_text = self._first_point(self._pick_points_for_ui(result, "matched_points"))
        missing_text = self._first_point(self._filter_missing_points_for_ui(self._pick_points_for_ui(result, "missing_points")))
        unsupported_text = self._first_point(self._pick_points_for_ui(result, "unsupported_points"))
        lines = [
            f"Оценка: {grade}",
            f"Балл: {result.get('score')}",
        ]
        if matched_text:
            lines.append(f"Что уже верно: {matched_text}")
        if missing_text:
            lines.append(f"Чего не хватает: {missing_text}")
        if unsupported_text:
            lines.append(f"Что не точно: {unsupported_text}")
        analogy = self._pick_analogy_for_ui(result, warning_text=unsupported_text, missing_text=missing_text)
        if analogy:
            lines.append(f"Аналогия: {analogy}")
        follow_up = (result.get('follow_up_question') or '').strip()
        if follow_up:
            lines.append(f"Уточняющий вопрос: {follow_up}")
        msg = "\n".join(line for line in lines if str(line).strip()) + "\n"
        self._append("AI", msg)
        self.status_var.set("Ответ проверен")
        follow_up_question = (result or {}).get("follow_up_question", "").strip()
        self.awaiting_follow_up = bool(follow_up_question)
        self.last_follow_up_question = follow_up_question
        if self.awaiting_follow_up:
            self.submit_btn.configure(text="Ответить на уточняющий вопрос")
        else:
            self.submit_btn.configure(text="Проверить ответ")
            self._apply_srs_once()
        if callable(self.on_show_answer):
            try:
                self.on_show_answer()
            except Exception:
                pass

    def submit_follow_up_answer(self):
        if not self.current_card:
            return
        follow_up_answer = self.answer_input.get("1.0", "end-1c").strip()
        if not follow_up_answer:
            self.status_var.set("Введите ответ на уточняющий вопрос")
            return
        is_clarification_request = self._is_clarification_request(follow_up_answer)
        self._append("Вы", follow_up_answer) if is_clarification_request else self._append(
            "Вы на уточняющий вопрос", follow_up_answer
        )
        if is_clarification_request:
            hint = self.controller.grader.explain_follow_up_hint(
                self.current_card,
                self.last_user_answer,
                self.last_follow_up_question,
                previous_grade_result=self.last_grade_result,
            )
            hint_lines = []
            short_explanation = str(hint.get("short_explanation") or "").strip()
            missing_detail = str(hint.get("missing_detail") or "").strip()
            analogy = str(hint.get("analogy") or "").strip()
            example_answer = str(hint.get("example_answer") or "").strip()
            next_prompt = str(hint.get("next_prompt") or "").strip()
            if short_explanation:
                hint_lines.append(short_explanation)
            if missing_detail:
                hint_lines.append(f"Что добавить: {missing_detail}")
            if analogy:
                hint_lines.append(f"Аналогия: {analogy}")
            if example_answer:
                hint_lines.append(f"Пример: {example_answer}")
            if next_prompt:
                hint_lines.append(next_prompt)
            msg = "\n".join(hint_lines)
            self._append("AI-подсказка", msg)
            self.answer_input.delete("1.0", tk.END)
            self.awaiting_follow_up = True
            self.submit_btn.configure(text="Ответить на уточняющий вопрос")
            self.status_var.set("Подсказка показана — попробуйте ответить на уточняющий вопрос")
            return
        result = self.controller.grader.grade_follow_up_answer(
            self.current_card,
            self.last_user_answer,
            self.last_follow_up_question,
            follow_up_answer,
            previous_grade_result=self.last_grade_result,
        )
        previous_score = float((self.last_grade_result or {}).get("score") or 0.0)
        self.last_grade_result = result
        follow_missing = self._first_point(self._filter_missing_points_for_ui(self._pick_points_for_ui(result, "missing_points")))
        follow_unsupported = self._first_point(self._pick_points_for_ui(result, "unsupported_points"))
        score_delta = float(result.get("score_delta") or 0.0)
        improved_flag = bool(result.get("improved")) or score_delta > 0.05
        still_gap = bool(follow_missing or str(result.get("remaining_gap") or "").strip())
        if improved_flag and still_gap:
            improved_text = "Стало лучше: немного"
        elif improved_flag:
            improved_text = "Стало лучше: да"
        else:
            now_score = float(result.get("score") or 0.0)
            if previous_score >= 0.65 or now_score >= 0.65:
                improved_text = "Почти. Осталось уточнить одну деталь."
            else:
                improved_text = "Стало лучше: нет"
        lines = [
            improved_text,
        ]
        short_feedback = str(result.get("short_feedback") or "").strip()
        matched_text = self._first_point(self._pick_points_for_ui(result, "matched_points"))
        if short_feedback:
            lines.append(f"Что улучшилось: {short_feedback}")
        elif matched_text:
            lines.append(f"Что улучшилось: {matched_text}")
        follow_analogy = self._pick_analogy_for_ui(result, warning_text=follow_unsupported, missing_text=follow_missing)
        remaining_gap = str(result.get('remaining_gap') or '').strip()
        if "?" in remaining_gap:
            remaining_gap = ""
        if follow_missing:
            lines.append(f"Что ещё не точно: {follow_missing}")
        elif remaining_gap:
            lines.append(f"Что ещё не точно: {remaining_gap}")
        if follow_unsupported:
            lines.append(f"Что ещё не точно: {follow_unsupported}")
        final_hint = str(result.get('final_hint') or '').strip()
        if final_hint:
            lines.append(f"Как сказать ближе к карточке: {final_hint}")
        if follow_analogy:
            lines.append(f"Следующий шаг: {follow_analogy}")
        else:
            lines.append("Следующий шаг: добавьте недостающую деталь про вторую группу реакций.")
        msg = "\n".join(line for line in lines if str(line).strip()) + "\n"
        self._append("AI", msg)
        self.answer_input.delete("1.0", tk.END)
        follow_up_complete = bool(result.get("follow_up_complete", True))
        self.awaiting_follow_up = not follow_up_complete
        if self.awaiting_follow_up:
            self.submit_btn.configure(text="Ответить на уточняющий вопрос")
            self.status_var.set("Нужно уточнить ответ ещё раз")
        else:
            self.last_follow_up_question = ""
            self.submit_btn.configure(text="Проверить ответ")
            self.status_var.set("Уточнение обработано")
            self._apply_srs_once()

    def _is_clarification_request(self, text: str) -> bool:
        raw = (text or "").strip().lower()
        if not raw:
            return False
        normalized = " ".join(raw.split())
        normalized = normalized.replace("ё", "е")
        normalized = normalized.rstrip("?.!,:;")
        normalized = " ".join(normalized.split())

        patterns = {
            "какую",
            "какой",
            "что именно",
            "не понял",
            "подскажи",
            "обьясни",
            "объясни",
            "как ответить",
            "какая деталь",
            "какую деталь",
            "поясни",
            "можно подсказку",
        }
        if normalized in patterns:
            return True

        if len(normalized.split()) <= 5:
            starters = (
                "какую",
                "какой",
                "какая",
                "что именно",
                "не понял",
                "подскажи",
                "обьясни",
                "объясни",
                "как ответить",
                "поясни",
                "можно подсказку",
            )
            return any(normalized.startswith(prefix) for prefix in starters)
        return False

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
        self.awaiting_follow_up = False
        self.last_follow_up_question = ""
        self.last_user_answer = ""
        self.last_grade_result = None
        self.srs_applied_for_current_card = False
        self.submit_btn.configure(text="Проверить ответ")
        self.status_var.set("Введите ответ...")

    def go_next_card(self):
        self._apply_srs_once()
        self.clear_chat()
        if callable(self.on_next_card):
            self.on_next_card()

    def _apply_srs_once(self):
        if self.srs_applied_for_current_card:
            return
        if not self.current_card or not self.last_grade_result:
            return
        self.controller.apply_srs(self.current_card, self.last_grade_result)
        self.srs_applied_for_current_card = True

    def _append(self, role, text):
        self.history.configure(state=tk.NORMAL)
        self.history.insert(tk.END, f"{role}:\n{text}\n\n")
        self.history.see(tk.END)
        self.history.configure(state=tk.DISABLED)

    def _on_ctrl_enter(self, _event=None):
        self.submit_answer()
        return "break"


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
