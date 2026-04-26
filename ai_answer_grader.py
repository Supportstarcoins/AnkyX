from __future__ import annotations

import re


class AIAnswerGrader:
    def _keywords(self, text: str) -> set[str]:
        words = re.findall(r"[A-Za-zА-Яа-я0-9_]{3,}", (text or "").lower())
        stop = {"что", "это", "для", "как", "или", "the", "and", "with"}
        return {w for w in words if w not in stop}

    def grade_answer(self, card, user_answer, answer_time_ms, provider="auto"):
        _ = provider
        back = (card or {}).get("back", "")
        answer = (user_answer or "").strip()
        if not answer:
            return self._result("wrong", 0.0, "too_slow", "no_answer", "Нет ответа.")

        expected = self._keywords(back)
        actual = self._keywords(answer)
        overlap = len(expected & actual)
        coverage = overlap / max(1, len(expected))

        if answer_time_ms < 4000:
            t_quality = "fast"
        elif answer_time_ms < 15000:
            t_quality = "normal"
        elif answer_time_ms < 40000:
            t_quality = "slow"
        else:
            t_quality = "too_slow"

        if coverage >= 0.9 and t_quality in {"fast", "normal"}:
            grade, action = "correct", "increase"
        elif coverage >= 0.75:
            grade, action = ("slow_correct", "slight_increase") if t_quality in {"slow", "too_slow"} else ("partial", "slight_increase")
        elif coverage >= 0.5:
            grade, action = "partial", "repeat_soon"
        else:
            grade, action = "wrong", "reset"

        mistake_type = "none" if coverage >= 0.75 else "missing_key_point"
        missing = sorted(list(expected - actual))[:5]
        feedback = "Ответ близок к правильному." if coverage >= 0.75 else "В ответе не хватает ключевых пунктов."

        result = self._result(grade, round(coverage, 3), t_quality, mistake_type, feedback)
        result["missing_points"] = missing
        result["srs_action"] = action
        result["follow_up_question"] = f"Как бы вы объяснили это короче: {(card or {}).get('front', 'тему')}?"
        result["analogy"] = "Подумайте об этом как о кратком определении с 2-3 ключевыми признаками."
        result["card_action"] = "keep" if coverage >= 0.75 else "simplify"
        result["suggested_rewrite"] = {
            "front": (card or {}).get("front", ""),
            "back": (back[:200] + "...") if len(back) > 220 else back,
        }
        return result

    def _result(self, grade, score, answer_time_quality, mistake_type, feedback):
        return {
            "grade": grade,
            "score": float(score),
            "confidence": 0.65,
            "answer_time_quality": answer_time_quality,
            "mistake_type": mistake_type,
            "missing_points": [],
            "short_feedback": feedback,
            "analogy": "",
            "follow_up_question": "",
            "srs_action": "repeat_soon",
            "card_action": "keep",
            "suggested_rewrite": {"front": "", "back": ""},
        }

