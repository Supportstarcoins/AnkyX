from __future__ import annotations

import re


class AIAnswerGrader:
    RU_STOPWORDS = {
        "это",
        "этот",
        "эта",
        "эти",
        "его",
        "её",
        "их",
        "них",
        "они",
        "она",
        "оно",
        "более",
        "менее",
        "быстро",
        "медленно",
        "обычный",
        "обычные",
        "обычными",
        "другие",
        "другой",
        "также",
        "тоже",
        "которые",
        "который",
        "которая",
        "может",
        "могут",
        "быть",
        "является",
        "являются",
        "после",
        "перед",
        "если",
        "при",
        "для",
        "как",
        "что",
        "чем",
        "где",
        "когда",
        "или",
        "и",
        "а",
        "но",
        "на",
        "в",
        "во",
        "по",
        "из",
        "с",
        "со",
        "у",
        "от",
        "до",
        "же",
        "ли",
    }
    EN_STOPWORDS = {"the", "and", "with", "this", "that", "from", "into", "for"}
    ENDINGS = (
        "ыми",
        "ими",
        "ого",
        "его",
        "ому",
        "ему",
        "ами",
        "ями",
        "ах",
        "ях",
        "ой",
        "ый",
        "ий",
        "ая",
        "ое",
        "ые",
    )

    def _normalize_token(self, token: str) -> str:
        for ending in self.ENDINGS:
            if token.endswith(ending) and len(token) - len(ending) >= 4:
                return token[: -len(ending)]
        return token

    def _keywords(self, text: str) -> set[str]:
        text_l = (text or "").lower()
        words = re.findall(r"[a-zа-яё]+", text_l, flags=re.IGNORECASE)
        stop = self.RU_STOPWORDS | self.EN_STOPWORDS
        keywords = set()
        for word in words:
            if word in stop:
                continue
            if len(word) < 4:
                continue
            normalized = self._normalize_token(word)
            if len(normalized) >= 4 and normalized not in stop:
                keywords.add(normalized)
        return keywords

    def _make_missing_explanation(self, card_back: str, missing_keywords: list[str]) -> str:
        clean_back = (card_back or "").strip()
        if not clean_back:
            return "В ответе не хватает ключевой мысли из обратной стороны карточки."

        sentences = [s.strip() for s in re.split(r"[.!?]\s+", clean_back) if s.strip()]
        if not sentences:
            return "В ответе не хватает ключевой мысли из обратной стороны карточки."

        best_sentence = ""
        best_score = 0
        for sentence in sentences:
            sentence_keywords = self._keywords(sentence)
            score = len(sentence_keywords.intersection(set(missing_keywords)))
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if not best_sentence:
            best_sentence = sentences[0]

        if len(best_sentence) > 170:
            best_sentence = best_sentence[:167].rstrip() + "..."
        best_sentence = best_sentence.rstrip(" .!?")
        return f"В ответе не хватает ключевой мысли: {best_sentence}."

    def _make_short_feedback(self, grade: str) -> str:
        mapping = {
            "correct": "Ответ верный: ключевая мысль сохранена.",
            "partial": "Ответ частично верный, но не хватает одной важной детали.",
            "wrong": "Ответ не совпадает с ключевым смыслом карточки.",
            "slow_correct": "Ответ верный, но дался медленно — интервал лучше увеличивать осторожно.",
            "uncertain": "Есть попадание в тему, но ответ пока недостаточно точный.",
        }
        return mapping.get(grade, "Нужна дополнительная проверка ответа.")

    def _make_follow_up_question(self, card_back: str) -> str:
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", (card_back or "")) if s.strip()]
        if len(sentences) >= 2:
            return "Какая деталь из полного ответа чаще всего теряется в сокращённой версии?"
        return "Какую одну ключевую деталь стоит добавить, чтобы ответ стал полным?"

    def grade_answer(self, card, user_answer, answer_time_ms, provider="auto"):
        _ = provider
        back = (card or {}).get("back", "")
        answer = (user_answer or "").strip()
        if not answer:
            result = self._result("wrong", 0.0, "too_slow", "no_answer")
            result["srs_action"] = "reset"
            result["error_explanation"] = "В ответе нет содержательной части для проверки."
            return result

        expected = self._keywords(back)
        actual = self._keywords(answer)
        overlap = len(expected & actual)
        coverage = overlap / max(1, len(expected))

        if answer_time_ms < 4000:
            t_quality = "fast"
        elif answer_time_ms <= 15000:
            t_quality = "normal"
        elif answer_time_ms <= 40000:
            t_quality = "slow"
        else:
            t_quality = "too_slow"

        if coverage >= 0.85:
            grade, action = "correct", "increase"
        elif coverage >= 0.55:
            grade, action = "partial", "repeat_soon"
        elif coverage >= 0.25:
            grade, action = "uncertain", "repeat_soon"
        else:
            grade, action = "wrong", "reset"

        if grade == "correct" and t_quality in {"slow", "too_slow"}:
            grade, action = "slow_correct", "slight_increase"

        mistake_type = "none" if grade in {"correct", "slow_correct"} else "missing_key_point"
        missing = sorted(list(expected - actual))[:5]

        if grade in {"correct", "slow_correct"}:
            error_explanation = "Существенных смысловых ошибок не найдено."
        else:
            error_explanation = self._make_missing_explanation(back, missing)

        result = self._result(grade, round(coverage, 3), t_quality, mistake_type)
        result["missing_points"] = missing
        result["srs_action"] = action
        result["error_explanation"] = error_explanation
        result["follow_up_question"] = self._make_follow_up_question(back)
        result["analogy"] = "Это как кратко описать фильм: важно назвать главный конфликт, а не случайные детали."
        result["card_action"] = "keep" if coverage >= 0.55 else "simplify"
        result["short_feedback"] = self._make_short_feedback(grade)
        result["suggested_rewrite"] = {
            "front": (card or {}).get("front", ""),
            "back": (back[:200] + "...") if len(back) > 220 else back,
        }
        return result

    def grade_follow_up_answer(
        self,
        card,
        original_answer,
        follow_up_question,
        follow_up_answer,
        previous_grade_result=None,
    ):
        back = (card or {}).get("back") or (card or {}).get("answer") or ""
        original_keywords = self._keywords(original_answer or "")
        follow_up_keywords = self._keywords(follow_up_answer or "")
        expected_keywords = self._keywords(back)

        new_keywords = follow_up_keywords - original_keywords
        new_expected_keywords = new_keywords & expected_keywords
        all_covered = (original_keywords | follow_up_keywords) & expected_keywords
        previous_score = float((previous_grade_result or {}).get("score") or 0.0)
        new_score = len(all_covered) / max(1, len(expected_keywords))
        score_delta = round(new_score - previous_score, 3)
        improved = bool(new_expected_keywords) or score_delta > 0.03

        remaining_keywords = sorted(list(expected_keywords - all_covered))
        remaining_gap = (
            f"Не хватает смысловых опор: {', '.join(remaining_keywords[:5])}."
            if remaining_keywords
            else "Ключевые смысловые опоры покрыты."
        )
        short_feedback = (
            "Стало лучше: в уточнении добавлены важные элементы."
            if improved
            else "Пока прирост небольшой: добавьте точнее ключевую мысль."
        )

        if remaining_keywords:
            final_hint = (
                "Соберите один короткий ответ, где явно названы: "
                + ", ".join(remaining_keywords[:3])
                + "."
            )
        else:
            final_hint = "Ответ стал полнее, закрепите формулировку одной фразой."

        if follow_up_question and len((follow_up_answer or "").strip()) < 6:
            short_feedback = "Ответ слишком короткий для уверенного улучшения."
            improved = False

        follow_up_complete = bool(improved and (new_score >= 0.75 or not remaining_keywords))
        return {
            "improved": improved,
            "score_delta": score_delta,
            "short_feedback": short_feedback,
            "remaining_gap": remaining_gap,
            "final_hint": final_hint,
            "follow_up_complete": follow_up_complete,
        }

    def _result(self, grade, score, answer_time_quality, mistake_type):
        return {
            "grade": grade,
            "score": float(score),
            "confidence": 0.65,
            "answer_time_quality": answer_time_quality,
            "mistake_type": mistake_type,
            "missing_points": [],
            "short_feedback": self._make_short_feedback(grade),
            "error_explanation": "",
            "analogy": "",
            "follow_up_question": "",
            "srs_action": "repeat_soon",
            "card_action": "keep",
            "suggested_rewrite": {"front": "", "back": ""},
        }
