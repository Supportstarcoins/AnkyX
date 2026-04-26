from __future__ import annotations

import os
import time
import logging
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
                safe_result = dict(result or {})
                score = float(safe_result.get("score") or 0.0)
                if score < 0.5:
                    safe_result["grade"] = "wrong"
                return self.srs.apply_ai_grade_to_card(card_id, safe_result, self.app)
            except Exception:
                return None
        return None


class AIReviewPanel(ttk.Frame):
    def __init__(self, parent, app=None, on_next_card=None, on_show_answer=None, on_card_updated=None, on_deck_stats_changed=None):
        super().__init__(parent)
        self.controller = AIReviewController(app=app)
        self.app = app
        self.on_next_card = on_next_card
        self.on_show_answer = on_show_answer
        self.on_card_updated = on_card_updated
        self.on_deck_stats_changed = on_deck_stats_changed

        self.current_card = None
        self.answer_started_at = None
        self.awaiting_follow_up = False
        self.last_follow_up_question = ""
        self.last_user_answer = ""
        self.last_grade_result = None
        self.current_final_grade_result = None
        self.srs_applied_for_current_card = False
        self._auto_next_job = None

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
        if not human and base_key in {"matched_points", "missing_points", "unsupported_points"}:
            back = str((self.current_card or {}).get("back") or "")
            items = self.controller.grader._humanize_points(raw, back, self.last_user_answer or "")
        back = str((self.current_card or {}).get("back") or "")
        cleaned = [
            self.controller.grader._sanitize_human_language(str(x).strip(), back)
            for x in items
            if str(x).strip()
        ]
        cleaned = [x for x in cleaned if x]
        if base_key == "matched_points":
            if any(len(x.split()) <= 2 for x in cleaned):
                cleaned = self.controller.grader._humanize_points(cleaned, back, self.last_user_answer or "")
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

    def _sanitize_ui_text(self, text: str) -> str:
        line = str(text or "").strip()
        if not line:
            return ""
        back = str((self.current_card or {}).get("back") or "")
        cleaned = self.controller.grader._sanitize_human_language(line, back)
        if cleaned and self.controller.grader._is_meta_feedback_phrase(cleaned):
            return ""
        return cleaned

    def _on_answer_graded(self, result):
        self.progress.stop()
        self.last_grade_result = result
        self.current_final_grade_result = result
        score = float(result.get("score") or 0.0)
        grade = str(result.get("grade", "unknown")).strip().lower()
        answer_time_quality = str(result.get("answer_time_quality") or "normal").strip().lower()
        confident_correct = score >= 0.85 and (grade == "correct" or (grade == "slow_correct") or (grade == "correct" and answer_time_quality in {"slow", "too_slow"}))
        is_partial = (grade == "partial") or (0.5 <= score < 0.85)
        is_wrongish = (grade == "wrong") or (score < 0.5) or (grade == "uncertain")
        answer_time_ms = int(result.get("answer_time_ms") or 0)
        matched_text = self._first_point(self._pick_points_for_ui(result, "matched_points"))
        missing_text = self._first_point(self._filter_missing_points_for_ui(self._pick_points_for_ui(result, "missing_points")))
        unsupported_text = self._first_point(self._pick_points_for_ui(result, "unsupported_points"))
        lines = [
            f"Оценка: {grade}",
            f"Балл: {round(score, 3)}",
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
        follow_up = self._sanitize_ui_text((result.get('follow_up_question') or '').strip())
        if follow_up:
            lines.append(f"Уточняющий вопрос: {follow_up}")
        if self._is_debug_mode() and result.get("source"):
            lines.append(f"Источник проверки: {result.get('source')}")
        msg = "\n".join(line for line in lines if str(line).strip()) + "\n"
        self._append("AI", msg)
        self.status_var.set("Ответ проверен")
        follow_up_question = self._sanitize_ui_text((result or {}).get("follow_up_question", "").strip())
        self.awaiting_follow_up = False
        self.last_follow_up_question = ""

        if confident_correct:
            self._maybe_apply_srs_and_auto_advance(result, prefix_message="Верно")
            self.submit_btn.configure(text="Проверить ответ")
            self.status_var.set("Верно — переход к следующей карточке…")
        elif is_partial:
            self.awaiting_follow_up = bool(follow_up_question)
            self.last_follow_up_question = follow_up_question
            if self.awaiting_follow_up:
                self.submit_btn.configure(text="Ответить на уточняющий вопрос")
            else:
                self.submit_btn.configure(text="Проверить ответ")
            self.status_var.set("Ответ частичный — уточните ответ")
        elif is_wrongish:
            self.submit_btn.configure(text="Проверить ответ")
            self._apply_srs_once()
            self.status_var.set("Ответ неверный")
        else:
            self.submit_btn.configure(text="Проверить ответ")
            self.status_var.set("Ответ проверен")
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
        merged_result = self._merge_follow_up_result(self.last_grade_result, result)
        previous_score = float((self.last_grade_result or {}).get("score") or 0.0)
        self.current_final_grade_result = merged_result
        self.last_grade_result = merged_result
        follow_missing = self._sanitize_ui_text(
            self._first_point(self._filter_missing_points_for_ui(self._pick_points_for_ui(merged_result, "missing_points")))
        )
        follow_unsupported = self._sanitize_ui_text(self._first_point(self._pick_points_for_ui(merged_result, "unsupported_points")))
        score_delta = float(result.get("score_delta") or 0.0)
        improved_flag = bool(result.get("improved")) or score_delta > 0.05
        short_feedback = self._sanitize_ui_text(str(merged_result.get("short_feedback") or result.get("short_feedback") or "").strip())
        still_gap = bool(follow_missing or str(result.get("remaining_gap") or "").strip())
        forced_improved_text = ""
        if "Вы добавили недостающую деталь из карточки." in short_feedback:
            improved_text = "Стало лучше: да"
            forced_improved_text = improved_text
        elif "Теперь ответ близок к карточке." in short_feedback:
            improved_text = "Стало лучше: да"
            still_gap = False
            forced_improved_text = improved_text
        if forced_improved_text:
            improved_text = forced_improved_text
        elif improved_flag and still_gap:
            improved_text = "Стало лучше: немного"
        elif improved_flag:
            improved_text = "Стало лучше: да"
        else:
            now_score = float(merged_result.get("score") or 0.0)
            if previous_score >= 0.65 or now_score >= 0.65:
                improved_text = "Почти. Осталось уточнить одну деталь."
            else:
                improved_text = "Стало лучше: нет"
        matched_text = self._sanitize_ui_text(self._first_point(self._pick_points_for_ui(merged_result, "matched_points")))
        follow_analogy = self._pick_analogy_for_ui(merged_result, warning_text=follow_unsupported, missing_text=follow_missing)
        remaining_gap = self._sanitize_ui_text(str(result.get('remaining_gap') or '').strip())
        if "?" in remaining_gap:
            remaining_gap = ""
        what_improved = self._sanitize_ui_text(str(result.get("what_improved") or "").strip()) or short_feedback or matched_text or "Добавлена часть формулировки из карточки."
        remaining_parts = [part for part in (follow_missing, remaining_gap, follow_unsupported) if part]
        what_left = "; ".join(remaining_parts)
        final_hint = self._sanitize_ui_text(str(result.get('final_hint') or '').strip())
        if not what_left:
            what_left = "Критичных расхождений не осталось."
        if not final_hint:
            final_hint = self._sanitize_ui_text(follow_analogy) or "Повторите ответ формулировкой карточки одной фразой."
        lines = [
            improved_text,
            f"Итоговая оценка: {str(merged_result.get('grade') or '').strip().lower()}",
            f"Итоговый балл: {float(merged_result.get('score') or 0.0):.2f}",
            f"Что улучшилось: {what_improved}",
            f"Что ещё осталось: {what_left}",
            f"Как сказать ближе к карточке: {final_hint}",
        ]
        srs_auto = self._maybe_apply_srs_and_auto_advance(merged_result, prefix_message="Теперь верно")
        if srs_auto.get("applied"):
            level = srs_auto.get("new_level") or srs_auto.get("leitner_level") or srs_auto.get("phase")
            if level:
                lines.append(f"Карточка перенесена в подколоду {level}.")
            else:
                lines.append("Карточка перенесена в следующую подколоду.")
            due_human = srs_auto.get("due_human")
            if due_human:
                lines.append(f"Следующее повторение: {due_human}")
        msg = "\n".join(line for line in lines if str(line).strip()) + "\n"
        self._append("AI", msg)
        self.answer_input.delete("1.0", tk.END)
        follow_up_complete = bool(result.get("follow_up_complete", True))
        self.awaiting_follow_up = not follow_up_complete
        merged_grade = str((merged_result or {}).get("grade") or "").strip().lower()
        merged_score = float((merged_result or {}).get("score") or 0.0)
        should_auto_next = merged_grade in {"correct", "slow_correct"} and merged_score >= 0.85
        if should_auto_next:
            self.awaiting_follow_up = False
        if self.awaiting_follow_up:
            self.submit_btn.configure(text="Ответить на уточняющий вопрос")
            self.status_var.set("Нужно уточнить ответ ещё раз")
        else:
            self.last_follow_up_question = ""
            self.submit_btn.configure(text="Проверить ответ")
            self.status_var.set("Уточнение обработано")
            if should_auto_next:
                self.status_var.set("Теперь верно — переход к следующей карточке…")

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
        if self._auto_next_job is not None:
            try:
                self.after_cancel(self._auto_next_job)
            except Exception:
                pass
            self._auto_next_job = None
        self.answer_input.delete("1.0", tk.END)
        self.history.configure(state=tk.NORMAL)
        self.history.delete("1.0", tk.END)
        self.history.configure(state=tk.DISABLED)
        self.awaiting_follow_up = False
        self.last_follow_up_question = ""
        self.last_user_answer = ""
        self.last_grade_result = None
        self.current_final_grade_result = None
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
            return None
        if not self.current_card or not self.last_grade_result:
            return None
        srs_result = self.controller.apply_srs(self.current_card, self.last_grade_result)
        if isinstance(srs_result, dict):
            self.srs_applied_for_current_card = bool(srs_result.get("applied", True))
        else:
            self.srs_applied_for_current_card = bool(srs_result)
        return srs_result

    def _card_to_mutable_dict(self, card):
        if isinstance(card, dict):
            return card
        if card is None:
            return {}
        if hasattr(card, "_asdict"):
            try:
                return dict(card._asdict())
            except Exception:
                pass
        keys = getattr(card, "keys", None)
        if callable(keys):
            try:
                return {k: card[k] for k in keys()}
            except Exception:
                pass
        try:
            return dict(card)
        except Exception:
            return {}

    def _update_current_card_from_srs_result(self, srs_result):
        if not isinstance(srs_result, dict) or not self.current_card:
            return
        card = self._card_to_mutable_dict(self.current_card)
        if card is not self.current_card:
            self.current_card = card
        due_value = srs_result.get("due")
        if due_value is not None:
            self.current_card["due"] = due_value
            self.current_card["next_review"] = srs_result.get("due_human") or due_value
        new_level = srs_result.get("new_level")
        if new_level is not None:
            self.current_card["leitner_level"] = new_level
            self.current_card["phase"] = new_level
        if srs_result.get("interval") is not None:
            self.current_card["interval"] = srs_result.get("interval")

    def _refresh_card_ui(self, srs_result):
        if callable(self.on_card_updated):
            try:
                self.on_card_updated(self.current_card, srs_result)
                return
            except Exception:
                pass
        for target in (self.app, self.master, self.winfo_toplevel()):
            if target is None:
                continue
            for method_name in (
                "refresh_current_card",
                "update_card_display",
                "render_current_card",
                "show_current_card",
                "update_view",
            ):
                fn = getattr(target, method_name, None)
                if callable(fn):
                    try:
                        fn()
                        return
                    except Exception:
                        continue

    def _notify_deck_stats_changed(self):
        if callable(self.on_deck_stats_changed):
            try:
                self.on_deck_stats_changed()
                return
            except Exception:
                pass
        for target in (self.app, self.master, self.winfo_toplevel()):
            if target is None:
                continue
            fn = getattr(target, "refresh_deck_counters_and_phase_tree", None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    continue

    def _append_ai_message(self, text):
        self._append("AI", text)

    def _schedule_auto_next_card(self, delay_ms=1000):
        grade_result = self.current_final_grade_result or self.last_grade_result or {}
        grade = str((grade_result or {}).get("grade") or "").strip().lower()
        score = float((grade_result or {}).get("score") or 0.0)
        final_correct = grade in {"correct", "slow_correct"} and score >= 0.85
        if self.awaiting_follow_up and not final_correct:
            return
        if self._auto_next_job is not None:
            try:
                self.after_cancel(self._auto_next_job)
            except Exception:
                pass
            self._auto_next_job = None
        self._auto_next_job = self.after(int(delay_ms), self._call_next_card_safely)

    def _call_next_card_safely(self):
        self._auto_next_job = None
        try:
            self.clear_chat()
            if callable(self.on_next_card):
                self.on_next_card()
                return
            for target in (self.app, self.master, self.winfo_toplevel()):
                if target is None:
                    continue
                for name in ("next_card", "show_next_card", "go_next_card", "_next_card"):
                    fn = getattr(target, name, None)
                    if callable(fn):
                        fn()
                        return
            self._append_ai_message("Не удалось автоматически перейти к следующей карточке: callback next_card не найден.")
        except Exception:
            logging.exception("AI auto next card failed")
            self._append_ai_message("Ошибка автоперехода к следующей карточке.")

    def _merge_follow_up_result(self, previous_grade, follow_result) -> dict:
        return self.controller.grader.merge_follow_up_grade(previous_grade or {}, follow_result or {})

    def _maybe_apply_srs_and_auto_advance(self, grade_result, prefix_message: str = "Верно") -> dict:
        safe_grade = dict(grade_result or {})
        grade = str(safe_grade.get("grade") or "").strip().lower()
        score = float(safe_grade.get("score") or 0.0)
        if grade not in {"correct", "slow_correct"} or score < 0.85:
            return {"applied": False}
        srs_outcome = self._apply_srs_once()
        payload = srs_outcome if isinstance(srs_outcome, dict) else {"applied": bool(srs_outcome)}
        if payload.get("applied"):
            self._update_current_card_from_srs_result(payload)
            self._refresh_card_ui(payload)
            self._notify_deck_stats_changed()
            if grade == "slow_correct":
                self._append_ai_message(f"{prefix_message}, но ответ был медленным. Карточка повышена осторожно.")
            else:
                level = payload.get("new_level") or payload.get("leitner_level") or payload.get("phase")
                if level:
                    self._append_ai_message(f"{prefix_message}. Карточка перенесена в подколоду {level}.")
                else:
                    self._append_ai_message(f"{prefix_message}. Карточка перенесена в следующую подколоду.")
            if payload.get("due_human"):
                self._append_ai_message(f"Следующее повторение: {payload.get('due_human')}")
            self._schedule_auto_next_card(1000)
        return payload

    def _append(self, role, text):
        self.history.configure(state=tk.NORMAL)
        self.history.insert(tk.END, f"{role}:\n{text}\n\n")
        self.history.see(tk.END)
        self.history.configure(state=tk.DISABLED)

    def _on_ctrl_enter(self, _event=None):
        self.submit_answer()
        return "break"

    def _is_debug_mode(self) -> bool:
        value = os.getenv("AI_REVIEW_DEBUG", "").strip().lower()
        return value in {"1", "true", "yes", "on"}


def attach_ai_review_panel(parent, app=None, callbacks=None):
    callbacks = callbacks or {}
    panel = AIReviewPanel(
        parent,
        app=app,
        on_next_card=callbacks.get("on_next_card"),
        on_show_answer=callbacks.get("on_show_answer"),
        on_card_updated=callbacks.get("on_card_updated"),
        on_deck_stats_changed=callbacks.get("on_deck_stats_changed"),
    )
    panel.pack(fill=tk.X)
    return panel
