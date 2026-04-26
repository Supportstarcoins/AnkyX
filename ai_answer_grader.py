from __future__ import annotations

import importlib
import json
import logging
import os
import re
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)


class AIAnswerGrader:
    FOLLOW_UP_EXAMPLE_PHRASES = (
        "перечислите примеры",
        "приведите примеры",
        "какие примеры",
        "назовите примеры",
    )
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
            "confused": "Похоже на путаницу терминов — давайте уточним ключевое различие.",
        }
        return mapping.get(grade, "Нужна дополнительная проверка ответа.")

    def _make_follow_up_question(self, card_back: str) -> str:
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", (card_back or "")) if s.strip()]
        if len(sentences) >= 2:
            return "Какая деталь из полного ответа чаще всего теряется в сокращённой версии?"
        return "Какую одну ключевую деталь стоит добавить, чтобы ответ стал полным?"

    def _extract_json_object(self, raw_text: str) -> dict[str, Any] | None:
        text = (raw_text or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        chunk = match.group(0)
        try:
            payload = json.loads(chunk)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _back_has_explicit_examples(self, card_back: str) -> bool:
        text = (card_back or "").lower()
        if not text:
            return False
        markers = (
            "например",
            "к примеру",
            "такие как",
            "в том числе",
            "включая",
            ":",
            ";",
        )
        return any(marker in text for marker in markers)

    def _sanitize_follow_up_question(self, card_back: str, question: str, missing_points: list[str] | None = None) -> str:
        raw_question = (question or "").strip()
        if not raw_question:
            return raw_question

        lowered = raw_question.lower()
        asks_for_examples = any(phrase in lowered for phrase in self.FOLLOW_UP_EXAMPLE_PHRASES)
        if asks_for_examples and not self._back_has_explicit_examples(card_back):
            missing_points = [p.strip() for p in (missing_points or []) if str(p).strip()]
            if missing_points:
                return f"Что нужно добавить про пропущенную часть: {', '.join(missing_points[:3])}?"
            return "Какая ключевая мысль из правильного ответа была пропущена?"
        return raw_question

    def _clean_grade_payload(self, payload: dict[str, Any], card: dict[str, Any], answer_time_ms: int) -> dict[str, Any] | None:
        allowed_grades = {"correct", "partial", "wrong", "uncertain", "slow_correct", "confused"}
        grade = str(payload.get("grade") or "").strip().lower()
        if grade not in allowed_grades:
            return None

        score = float(payload.get("score") or 0.0)
        confidence = float(payload.get("confidence") or 0.0)
        answer_time_quality = str(payload.get("answer_time_quality") or "normal").strip().lower()
        if answer_time_quality not in {"fast", "normal", "slow", "too_slow"}:
            answer_time_quality = "normal"

        mistake_type = str(payload.get("mistake_type") or "missing_key_point").strip().lower()
        allowed_mistake = {
            "none",
            "missing_key_point",
            "confused_similar_terms",
            "too_general",
            "wrong_fact",
            "no_answer",
        }
        if mistake_type not in allowed_mistake:
            mistake_type = "missing_key_point"

        missing_points = payload.get("missing_points")
        if not isinstance(missing_points, list):
            missing_points = []
        missing_points = [str(x).strip() for x in missing_points if str(x).strip()][:8]
        card_back = str((card or {}).get("back") or "")

        suggested_rewrite = payload.get("suggested_rewrite")
        if not isinstance(suggested_rewrite, dict):
            suggested_rewrite = {}

        cleaned = {
            "grade": grade,
            "score": max(0.0, min(1.0, score)),
            "confidence": max(0.0, min(1.0, confidence)),
            "answer_time_quality": answer_time_quality,
            "mistake_type": mistake_type,
            "missing_points": missing_points,
            "short_feedback": str(payload.get("short_feedback") or self._make_short_feedback(grade)).strip(),
            "error_explanation": str(payload.get("error_explanation") or "").strip(),
            "analogy": str(payload.get("analogy") or "").strip(),
            "follow_up_question": self._sanitize_follow_up_question(
                card_back,
                str(payload.get("follow_up_question") or "").strip(),
                missing_points=missing_points,
            ),
            "srs_action": str(payload.get("srs_action") or "repeat_soon").strip(),
            "card_action": str(payload.get("card_action") or "keep").strip(),
            "suggested_rewrite": {
                "front": str(suggested_rewrite.get("front") or (card or {}).get("front", "")),
                "back": str(suggested_rewrite.get("back") or (card or {}).get("back", "")),
            },
            "source": "llm",
            "answer_time_ms": int(answer_time_ms or 0),
        }
        return cleaned

    def _get_llm_settings(self, provider: str = "auto") -> dict[str, Any]:
        settings: dict[str, Any] = {
            "provider": provider or "auto",
            "model": os.getenv("XFLASH_OLLAMA_MODEL", "").strip() or "xflash-llama31",
            "base_url": os.getenv("XFLASH_OLLAMA_URL", "").strip() or "http://127.0.0.1:11434",
            "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
            "timeout": 45,
        }
        path = os.path.join(os.getcwd(), "chatbot_settings.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle) or {}
                settings["provider"] = provider if provider != "auto" else str(data.get("llm_provider") or data.get("provider") or "auto")
                settings["model"] = str(data.get("ollama_model") or settings["model"])
                settings["base_url"] = str(data.get("ollama_url") or settings["base_url"])
                settings["api_key"] = str(data.get("api_key") or settings["api_key"])
            except Exception:
                LOGGER.debug("Failed to load chatbot_settings.json", exc_info=True)
        return settings

    def _call_via_project_providers(self, messages: list[dict[str, str]], provider: str = "auto") -> dict[str, Any] | None:
        try:
            module = importlib.import_module("llm_providers")
            router_cls = getattr(module, "LLMRouter", None)
            ollama_cls = getattr(module, "OllamaProvider", None)
            openai_cls = getattr(module, "OpenAIProvider", None)
            if not router_cls or not ollama_cls:
                return None
            settings = self._get_llm_settings(provider=provider)
            preferred = str(settings.get("provider") or "auto").lower()
            primary = None
            fallback = None
            if preferred in {"openai", "chatgpt", "cloud"} and openai_cls:
                primary = openai_cls()
                fallback = ollama_cls()
            else:
                primary = ollama_cls()
                if openai_cls and settings.get("api_key"):
                    fallback = openai_cls()
            router = router_cls(primary=primary, fallback=fallback)
            text = router.chat(messages, settings)
            return self._extract_json_object(text)
        except Exception:
            return None

    def _call_ollama_direct(self, messages: list[dict[str, str]], provider: str = "auto") -> dict[str, Any] | None:
        settings = self._get_llm_settings(provider=provider)
        model = settings.get("model") or "xflash-llama31"
        base_url = str(settings.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "\n\n".join(m.get("content", "") for m in messages),
                    "stream": False,
                    "options": {"temperature": 0.35},
                },
                timeout=float(settings.get("timeout") or 45),
            )
            if not resp.ok:
                return None
            data = resp.json() or {}
            return self._extract_json_object(str(data.get("response") or ""))
        except Exception:
            return None

    def _llm_json(self, messages: list[dict[str, str]], provider: str = "auto") -> dict[str, Any] | None:
        payload = self._call_via_project_providers(messages, provider=provider)
        if payload is not None:
            return payload
        return self._call_ollama_direct(messages, provider=provider)

    def _try_llm_grade_answer(self, card, user_answer, answer_time_ms, provider="auto"):
        prompt = (
            "Ты — живой AI-репетитор в приложении флэш-карточек.\n"
            "Твоя задача — не просто поставить оценку, а помочь пользователю понять ошибку.\n\n"
            "Правила:\n"
            "- Оцени смысл, а не точное совпадение слов.\n"
            "- Не будь сухим шаблоном.\n"
            "- Давай живую, образную, но короткую аналогию.\n"
            "- Аналогия должна быть связана с темой карточки или с повседневной ситуацией.\n"
            "- Не используй одну и ту же аналогию каждый раз.\n"
            "- Объясняй ошибку дружелюбно, но честно.\n"
            "- Если ответ частичный, скажи, какая главная мысль потеряна.\n"
            "- Оценивай ответ только по эталонному ответу карточки. Не требуй от пользователя фактов, примеров, дат или деталей, которых нет в правильном ответе. Уточняющий вопрос должен помогать восстановить именно missing часть из back, а не расширять тему.\n"
            "- Если правильный ответ говорит об общей категории, не требуй конкретных примеров. Если в back нет примеров, не спрашивай 'перечислите примеры'.\n"
            "- Уточняющий вопрос строй только по пропущенной части back.\n"
            "- Уточняющий вопрос должен помогать пользователю самому восстановить ответ.\n"
            "- Не повторяй просто исходный вопрос.\n"
            "- Не пиши длинный учебник.\n"
            "- Верни строго JSON.\n\n"
            "Формат JSON:\n"
            "{\n"
            '  "grade": "correct|partial|wrong|uncertain|slow_correct|confused",\n'
            '  "score": 0.0,\n'
            '  "confidence": 0.0,\n'
            '  "answer_time_quality": "fast|normal|slow|too_slow",\n'
            '  "mistake_type": "none|missing_key_point|confused_similar_terms|too_general|wrong_fact|no_answer",\n'
            '  "missing_points": [],\n'
            '  "short_feedback": "...",\n'
            '  "error_explanation": "...",\n'
            '  "analogy": "...",\n'
            '  "follow_up_question": "...",\n'
            '  "srs_action": "increase|slight_increase|repeat_soon|reset",\n'
            '  "card_action": "keep|simplify|split|rewrite|merge_duplicate",\n'
            '  "suggested_rewrite": {\n'
            '    "front": "...",\n'
            '    "back": "..."\n'
            "  }\n"
            "}\n\n"
            "Данные:\n"
            f"Вопрос карточки:\n{(card or {}).get('front', '')}\n\n"
            f"Правильный ответ:\n{(card or {}).get('back', '')}\n\n"
            f"Ответ пользователя:\n{(user_answer or '').strip()}\n\n"
            f"Время ответа:\n{int(answer_time_ms or 0)} мс"
        )
        messages = [
            {"role": "system", "content": "Верни только JSON-объект без markdown и комментариев."},
            {"role": "user", "content": prompt},
        ]
        payload = self._llm_json(messages, provider=provider)
        if payload is None:
            return None
        return self._clean_grade_payload(payload, card or {}, int(answer_time_ms or 0))

    def _try_llm_follow_up(self, card, original_answer, follow_up_question, follow_up_answer, previous_grade_result=None):
        prompt = (
            "Ты — AI-репетитор. Пользователь ответил на уточняющий вопрос.\n"
            "Оцени, стало ли понимание лучше.\n"
            "Не оценивай пользователя по знаниям вне карточки.\n"
            "Не повторяй шаблонно.\n"
            "Объясни коротко и живо.\n"
            "Верни строго JSON:\n\n"
            "{\n"
            '  "improved": true,\n'
            '  "score_delta": 0.0,\n'
            '  "short_feedback": "...",\n'
            '  "remaining_gap": "...",\n'
            '  "analogy": "...",\n'
            '  "final_hint": "...",\n'
            '  "follow_up_complete": true\n'
            "}\n\n"
            f"Вопрос карточки:\n{(card or {}).get('front', '')}\n\n"
            f"Правильный ответ:\n{(card or {}).get('back', '')}\n\n"
            f"Первичный ответ пользователя:\n{original_answer or ''}\n\n"
            f"Уточняющий вопрос AI:\n{follow_up_question or ''}\n\n"
            f"Ответ пользователя на уточнение:\n{follow_up_answer or ''}\n\n"
            f"Предыдущая оценка:\n{json.dumps(previous_grade_result or {}, ensure_ascii=False)}"
        )
        payload = self._llm_json([
            {"role": "system", "content": "Верни только JSON-объект без markdown и комментариев."},
            {"role": "user", "content": prompt},
        ])
        if not isinstance(payload, dict):
            return None
        try:
            return {
                "improved": bool(payload.get("improved")),
                "score_delta": float(payload.get("score_delta") or 0.0),
                "short_feedback": str(payload.get("short_feedback") or "").strip(),
                "remaining_gap": str(payload.get("remaining_gap") or "").strip(),
                "analogy": str(payload.get("analogy") or "").strip(),
                "final_hint": str(payload.get("final_hint") or "").strip(),
                "follow_up_complete": bool(payload.get("follow_up_complete", True)),
                "source": "llm",
            }
        except Exception:
            return None

    def _try_llm_hint(self, card, original_answer, follow_up_question, previous_grade_result=None):
        prompt = (
            "Ты — AI-репетитор. Пользователь не понял твой уточняющий вопрос и просит подсказку.\n"
            "Объясни живо и понятно, какую именно деталь нужно добавить.\n"
            "Не оценивай это как ответ.\n"
            "Не добавляй фактов извне. Подсказка должна опираться только на back карточки.\n"
            "Дай креативную короткую метафору.\n"
            "Дай пример ответа одной фразой.\n"
            "Верни строго JSON:\n\n"
            "{\n"
            '  "type": "hint",\n'
            '  "short_explanation": "...",\n'
            '  "missing_detail": "...",\n'
            '  "analogy": "...",\n'
            '  "example_answer": "...",\n'
            '  "next_prompt": "..."\n'
            "}\n\n"
            "Данные:\n"
            f"Вопрос карточки:\n{(card or {}).get('front', '')}\n\n"
            f"Правильный ответ:\n{(card or {}).get('back', '')}\n\n"
            f"Первичный ответ пользователя:\n{original_answer or ''}\n\n"
            f"Уточняющий вопрос AI:\n{follow_up_question or ''}\n\n"
            f"Предыдущая оценка:\n{json.dumps(previous_grade_result or {}, ensure_ascii=False)}"
        )
        payload = self._llm_json([
            {"role": "system", "content": "Верни только JSON-объект без markdown и комментариев."},
            {"role": "user", "content": prompt},
        ])
        if not isinstance(payload, dict):
            return None
        return {
            "type": "hint",
            "short_explanation": str(payload.get("short_explanation") or "").strip(),
            "missing_detail": str(payload.get("missing_detail") or "").strip(),
            "analogy": str(payload.get("analogy") or "").strip(),
            "example_answer": str(payload.get("example_answer") or "").strip(),
            "next_prompt": str(payload.get("next_prompt") or "").strip(),
            "source": "llm",
        }

    def grade_answer(self, card, user_answer, answer_time_ms, provider="auto"):
        llm_result = self._try_llm_grade_answer(card, user_answer, answer_time_ms, provider=provider)
        if llm_result is not None:
            return llm_result

        LOGGER.info("LLM недоступна, используется fallback-проверка.")
        back = (card or {}).get("back", "")
        answer = (user_answer or "").strip()
        if not answer:
            result = self._result("wrong", 0.0, "too_slow", "no_answer")
            result["srs_action"] = "reset"
            result["error_explanation"] = "В ответе нет содержательной части для проверки."
            result["source"] = "fallback"
            result["answer_time_ms"] = int(answer_time_ms or 0)
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
        result["source"] = "fallback"
        result["answer_time_ms"] = int(answer_time_ms or 0)
        return result

    def grade_follow_up_answer(
        self,
        card,
        original_answer,
        follow_up_question,
        follow_up_answer,
        previous_grade_result=None,
    ):
        llm_result = self._try_llm_follow_up(
            card,
            original_answer,
            follow_up_question,
            follow_up_answer,
            previous_grade_result=previous_grade_result,
        )
        if llm_result is not None:
            return llm_result

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
            "analogy": "Это как чинить цепочку: добавили одно звено, но нужно замкнуть весь контур.",
            "final_hint": final_hint,
            "follow_up_complete": follow_up_complete,
            "source": "fallback",
        }

    def explain_follow_up_hint(
        self,
        card,
        original_answer,
        follow_up_question,
        previous_grade_result=None,
    ):
        llm_result = self._try_llm_hint(
            card,
            original_answer,
            follow_up_question,
            previous_grade_result=previous_grade_result,
        )
        if llm_result is not None:
            return llm_result

        _ = follow_up_question
        back = ((card or {}).get("back") or (card or {}).get("answer") or "").strip()
        previous = previous_grade_result or {}
        missing_points = previous.get("missing_points") or []
        raw_error = (previous.get("error_explanation") or "").strip()

        if missing_points:
            missing_detail = "Добавьте смысловые опоры: " + ", ".join(missing_points[:4]) + "."
        elif raw_error:
            missing_detail = raw_error
        elif back:
            missing_detail = self._make_missing_explanation(back, sorted(list(self._keywords(back)))[:5])
        else:
            missing_detail = "Добавьте одну конкретную деталь из правильного ответа."

        short_explanation = (
            "Нужно добавить недостающую ключевую деталь, а не просто перефразировать предыдущий ответ."
        )
        if raw_error:
            short_explanation = "Нужно уточнить ключевую мысль, которой не хватило в ответе."

        example_source = back if back else missing_detail
        example_answer = re.sub(r"\s+", " ", example_source).strip()
        if len(example_answer) > 180:
            example_answer = example_answer[:177].rstrip() + "..."
        if not example_answer:
            example_answer = "Добавьте ключевую деталь короткой фразой."

        return {
            "type": "hint",
            "short_explanation": short_explanation,
            "missing_detail": missing_detail,
            "analogy": "Это как пазл: не хватает одной детали в центре, и картинка не складывается.",
            "example_answer": example_answer,
            "next_prompt": "Попробуйте теперь ответить одной короткой фразой.",
            "source": "fallback",
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
            "source": "fallback",
        }
