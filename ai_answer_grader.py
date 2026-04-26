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
        "какие именно",
        "какие конкретно",
        "перечисл",
        "приведи пример",
        "приведите пример",
        "назови примеры",
        "назовите примеры",
        "виды",
        "типы",
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
    CLAUSE_SPLIT_RE = re.compile(r"(?:[,;:.!?]+|\bа\b|\bно\b|\bи\b|\bчто\b)", flags=re.IGNORECASE)
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
    DANGEROUS_OUT_OF_BACK_PHRASES = (
        "медленнее",
        "температура",
        "боль",
        "слабость",
        "головная боль",
        "тяжёлые эффекты",
        "тяжелые эффекты",
        "какие именно",
        "примеры",
        "перечисли",
    )
    METAPHOR_MARKERS = ("представ", "как ", "словно", "будто", "это как")

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

    def _extract_semantic_points(self, text: str) -> list[str]:
        raw = (text or "").strip().lower()
        if not raw:
            return []

        chunks = [chunk.strip(" -–—\t\r\n") for chunk in self.CLAUSE_SPLIT_RE.split(raw) if chunk.strip()]
        points: list[str] = []
        for chunk in chunks:
            words = []
            for word in re.findall(r"[a-zа-яё]+", chunk, flags=re.IGNORECASE):
                normalized = self._normalize_token(word)
                if normalized in self.RU_STOPWORDS or normalized in self.EN_STOPWORDS:
                    continue
                if len(normalized) < 4:
                    continue
                words.append(normalized)
            if words:
                points.append(" ".join(words[:6]))

        if not points:
            fallback = sorted(self._keywords(raw))
            if fallback:
                points = [" ".join(fallback[:4])]
        return points[:8]

    def _build_semantic_specs(self, card_back: str) -> list[dict[str, Any]]:
        text = (card_back or "").lower()
        specs: list[dict[str, Any]] = []
        if re.search(r"обычн|реакц|эффект", text):
            specs.append({"id": "ordinary_reactions", "label": "упомянуты реакции/эффекты", "stems_any": {"обычн", "реакц", "эффект"}})
        if re.search(r"проход|быстр", text):
            specs.append({"id": "ordinary_pass_quickly", "label": "упомянуто, что часть проходит быстро", "stems_all": {"проход", "быстр"}})
        if re.search(r"\bдруг", text):
            specs.append({"id": "other_group", "label": "упомянута вторая группа реакций", "stems_any": {"друг"}})
        if re.search(r"редк", text):
            specs.append({"id": "rare_reactions", "label": "не указано, что другие осложнения более редкие", "stems_any": {"редк"}})
        if re.search(r"серьез|осложнен", text):
            specs.append({"id": "serious_complications", "label": "упомянуты серьезные осложнения", "stems_any": {"серьез", "осложнен"}})
        if not specs:
            for idx, point in enumerate(self._extract_semantic_points(card_back), start=1):
                specs.append({"id": f"generic_{idx}", "label": point, "stems_any": set(self._keywords(point))})
        return specs

    def _semantic_overlap_score(self, back: str, user_answer: str) -> tuple[float, list[str], list[str], list[str]]:
        specs = self._build_semantic_specs(back)
        answer_keywords = self._keywords(user_answer)
        answer_l = (user_answer or "").lower()
        matched_points: list[str] = []
        missing_points: list[str] = []

        for spec in specs:
            stems_all = set(spec.get("stems_all") or set())
            stems_any = set(spec.get("stems_any") or set())
            has_all = all(any(token.startswith(stem) for token in answer_keywords) for stem in stems_all) if stems_all else True
            has_any = any(any(token.startswith(stem) for token in answer_keywords) for stem in stems_any) if stems_any else True
            matched = has_all and has_any
            if spec.get("id") == "other_group" and not matched:
                matched = ("обыч" in answer_l and ("редк" in answer_l or "серьез" in answer_l or "ослож" in answer_l))
            if matched:
                matched_points.append(str(spec.get("label") or "").strip())
            else:
                missing_points.append(str(spec.get("label") or "").strip())

        back_points = self._extract_semantic_points(back)
        answer_points = self._extract_semantic_points(user_answer)
        back_keywords = [self._keywords(point) for point in back_points] or [self._keywords(back)]
        unsupported_points: list[str] = []
        for point in answer_points:
            point_kw = self._keywords(point)
            if not point_kw:
                continue
            best_overlap = 0.0
            for expected_kw in back_keywords:
                if not expected_kw:
                    continue
                overlap = len(point_kw & expected_kw) / max(1, len(point_kw))
                if overlap > best_overlap:
                    best_overlap = overlap
            if best_overlap < 0.34:
                unsupported_points.append(point)
        back_l = (back or "").lower()
        if re.search(r"\bне\s+проход", answer_l) and ("проход" in back_l and "быстр" in back_l):
            unsupported_points.append("фраза 'быстро не проходят' не указана в карточке")

        score = round(max(0.0, min(1.0, len(matched_points) / max(1, len(specs)))), 3)
        return score, matched_points[:6], missing_points[:6], unsupported_points[:4]

    def _semantic_coverage(self, card_back: str, answer_text: str) -> dict[str, Any]:
        points = self._build_semantic_specs(card_back)
        answer_keywords = self._keywords(answer_text)
        matched: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        score = 0.0
        weights = {
            "ordinary_reactions": 0.2,
            "ordinary_pass_quickly": 0.3,
            "other_group": 0.1,
            "rare_reactions": 0.2,
            "serious_complications": 0.2,
        }
        total_weight = 0.0
        answer_l = (answer_text or "").lower()

        for point in points:
            pid = point.get("id")
            weight = float(weights.get(pid, 1.0))
            total_weight += weight
            stems_all = set(point.get("stems_all") or set())
            stems_any = set(point.get("stems_any") or set())
            has_all = all(any(token.startswith(stem) for token in answer_keywords) for stem in stems_all) if stems_all else True
            has_any = any(any(token.startswith(stem) for token in answer_keywords) for stem in stems_any) if stems_any else True
            matched_point = has_all and has_any
            if pid == "other_group" and not matched_point:
                matched_point = ("обыч" in answer_l and ("редк" in answer_l or "серьез" in answer_l or "ослож" in answer_l))
            if matched_point:
                score += weight
                matched.append(point)
            else:
                missing.append(point)

        coverage = score / max(0.001, total_weight)
        return {
            "coverage": round(max(0.0, min(1.0, coverage)), 3),
            "matched": matched,
            "missing": missing,
            "points": points,
        }

    def _build_follow_up_from_missing(self, missing_points: list[dict[str, Any]]) -> str:
        if not missing_points:
            return ""
        top = missing_points[0]
        label = top.get("label") or "пропущенную мысль"
        if "проходят быстро" in label:
            return "Что в карточке сказано о том, как быстро проходят обычные реакции?"
        return f"Какую особенность нужно добавить про «{label}», чтобы ответ совпал с карточкой?"

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

    def _has_explicit_examples_or_list(self, text: str) -> bool:
        raw = (text or "")
        if not raw.strip():
            return False
        lowered = raw.lower()
        if "например" in lowered or "такие как" in lowered:
            return True
        if ";" in raw:
            return True
        if re.search(r":[ \t]*[^\n,;]+,\s*[^\n,;]+", raw):
            return True
        if re.search(r"(^|\n)\s*(?:[-•*]\s+|\d+[.)]\s+)", raw):
            return True
        if re.search(
            r"(реакц|осложнен|вид|тип|групп|категори)[^.!?\n:]{0,80}\b[а-яёa-z0-9\- ]+,\s*[а-яёa-z0-9\- ]+",
            lowered,
        ):
            return True
        return False

    def _safe_follow_up_question(self, back: str, missing_detail: str = "") -> str:
        text = (back or "").lower()
        detail = (missing_detail or "").lower()
        if (
            ("обычн" in text and ("редк" in text or "серьез" in text or "осложнен" in text))
            or ("обычн" in detail and ("редк" in detail or "серьез" in detail or "осложнен" in detail))
        ):
            return "Что нужно добавить про вторую группу реакций, чтобы ответ совпал с карточкой?"
        return "Какую ключевую мысль из карточки нужно добавить, чтобы ответ совпал с эталоном?"

    def _question_demands_external_details(self, question: str) -> bool:
        lowered = (question or "").lower()
        return any(phrase in lowered for phrase in self.FOLLOW_UP_EXAMPLE_PHRASES)

    def _sanitize_follow_up_question(self, card_back: str, question: str, missing_points: list[str] | None = None) -> str:
        text = (card_back or "").lower()
        raw_question = (question or "").strip()
        if not raw_question:
            return self._safe_follow_up_question(card_back)

        asks_for_external_details = self._question_demands_external_details(raw_question)
        if asks_for_external_details and not self._has_explicit_examples_or_list(text):
            missing_points = [p.strip() for p in (missing_points or []) if str(p).strip()]
            missing_detail = ", ".join(missing_points[:3]) if missing_points else ""
            return self._safe_follow_up_question(card_back, missing_detail=missing_detail)
        return raw_question

    def _safe_final_hint(self, back: str) -> str:
        text = (back or "").lower()
        if "обыч" in text and "проход" in text and "быстр" in text and ("редк" in text or "серьез" in text or "осложнен" in text):
            return "Скажите ближе к карточке: обычные реакции проходят быстро, а другие реакции более редкие и серьёзные."
        return "Скажите ближе к карточке и повторите формулировку из правильного ответа без новых деталей."

    def _looks_like_metaphor(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in self.METAPHOR_MARKERS)

    def _sanitize_llm_result_against_back(self, result: dict, back: str) -> dict:
        safe = dict(result or {})
        back_l = (back or "").lower()

        unsupported = safe.get("unsupported_points")
        if not isinstance(unsupported, list):
            unsupported = []

        def has_out_of_back_phrase(text: str) -> list[str]:
            lowered = (text or "").lower()
            hits: list[str] = []
            for phrase in self.DANGEROUS_OUT_OF_BACK_PHRASES:
                if phrase in lowered and phrase not in back_l:
                    hits.append(phrase)
            return hits

        for key in ("matched_points", "missing_points"):
            items = safe.get(key)
            if not isinstance(items, list):
                continue
            filtered_items: list[str] = []
            for item in items:
                item_s = str(item).strip()
                if not item_s:
                    continue
                bad = has_out_of_back_phrase(item_s)
                if bad:
                    unsupported.append(f"формулировка «{item_s}» не из карточки")
                    continue
                filtered_items.append(item_s)
            safe[key] = filtered_items[:8]

        for key in ("short_feedback", "error_explanation", "remaining_gap"):
            value = str(safe.get(key) or "").strip()
            if not value:
                continue
            bad = has_out_of_back_phrase(value)
            if bad:
                unsupported.extend([f"добавлена лишняя деталь: {phrase}" for phrase in bad])
                safe[key] = "Формулируйте ответ ближе к карточке, без добавления новых фактов."

        follow_q = str(safe.get("follow_up_question") or "").strip()
        if follow_q:
            bad = has_out_of_back_phrase(follow_q)
            if bad or self._question_demands_external_details(follow_q):
                unsupported.extend([f"уточняющий вопрос требует лишние детали: {phrase}" for phrase in bad] or ["уточняющий вопрос требует лишние детали"])
                safe["follow_up_question"] = self._safe_follow_up_question(back, ", ".join(safe.get("missing_points") or []))

        final_hint = str(safe.get("final_hint") or "").strip()
        if final_hint:
            bad = has_out_of_back_phrase(final_hint)
            if bad:
                unsupported.extend([f"лишняя подсказка: {phrase}" for phrase in bad])
                safe["final_hint"] = self._safe_final_hint(back)

        analogy = str(safe.get("analogy") or "").strip()
        if analogy:
            bad = has_out_of_back_phrase(analogy)
            if bad and not self._looks_like_metaphor(analogy):
                unsupported.extend([f"аналогия добавляет лишний факт: {phrase}" for phrase in bad])
                safe["analogy"] = "Представьте это как две зоны: одна быстро проходит, другая более редкая и серьёзная."

        safe["unsupported_points"] = [str(x).strip() for x in unsupported if str(x).strip()][:8]
        return safe

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
        matched_points = payload.get("matched_points")
        if not isinstance(matched_points, list):
            matched_points = []
        matched_points = [str(x).strip() for x in matched_points if str(x).strip()][:8]
        unsupported_points = payload.get("unsupported_points")
        if not isinstance(unsupported_points, list):
            unsupported_points = []
        unsupported_points = [str(x).strip() for x in unsupported_points if str(x).strip()][:8]
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
            "matched_points": matched_points,
            "unsupported_points": unsupported_points,
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
        return self._sanitize_llm_result_against_back(cleaned, card_back)

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
            "- Выдели смысловые пункты из back и сравни ответ именно по ним.\n"
            "- Если ученик назвал основные группы (например, обычные и редкие/серьёзные), но упустил одну уточняющую деталь, ставь partial примерно 0.65-0.75, а не wrong.\n"
            "- Не ставь wrong/0.0, если пользователь назвал хотя бы часть ключевых смысловых пунктов из правильного ответа. В таком случае используй partial.\n"
            "- В short_feedback сначала скажи, что уже верно, затем что конкретно отсутствует.\n"
            "- Уточняющий вопрос строй только по пропущенной части back.\n"
            "- Уточняющий вопрос должен помогать пользователю самому восстановить ответ.\n"
            "- Не повторяй просто исходный вопрос.\n"
            "- Не пиши длинный учебник.\n"
            "ЖЁСТКОЕ ОГРАНИЧЕНИЕ:\n"
            "Ты проверяешь только знание этой карточки, а не всей темы.\n"
            "Не задавай вопросы, ответ на которые нельзя вывести прямо из правильного ответа карточки.\n"
            "Не спрашивай 'какие именно', 'приведи примеры', 'перечисли виды', если в правильном ответе нет списка, примеров или видов.\n"
            "Если back говорит только 'редкие и серьёзные осложнения', спрашивай не 'какие именно осложнения', а 'что нужно добавить про вторую группу реакций?'.\n"
            "Креативность разрешена только в аналогиях и объяснениях, но не в фактах.\n"
            "КРЕАТИВНОСТЬ В РАМКАХ ФАКТА:\n"
            "Ты можешь использовать яркие аналогии и метафоры, но только чтобы объяснить факты из правильного ответа.\n"
            "Не добавляй в проверку новые сведения.\n"
            "Если хочешь привести пример или образ, он должен быть явно метафорой, а не новым медицинским фактом.\n"
            "Не называй конкретные примеры осложнений, если их нет в правильном ответе.\n"
            "Не заменяй «более редкие и серьёзные осложнения» на «медленные» или «тяжёлые», если таких слов нет в back.\n"
            "Если пользователь использовал слово, которого нет в back, оцени это осторожно:\n"
            "- если слово не меняет смысл, можно принять частично;\n"
            "- если слово добавляет новый факт, пометь как «не точно».\n"
            "- Верни строго JSON.\n\n"
            "Формат JSON:\n"
            "{\n"
            '  "grade": "correct|partial|wrong|uncertain|slow_correct|confused",\n'
            '  "score": 0.0,\n'
            '  "confidence": 0.0,\n'
            '  "answer_time_quality": "fast|normal|slow|too_slow",\n'
            '  "mistake_type": "none|missing_key_point|confused_similar_terms|too_general|wrong_fact|no_answer",\n'
            '  "missing_points": [],\n'
            '  "matched_points": [],\n'
            '  "unsupported_points": [],\n'
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
            cleaned = {
                "improved": bool(payload.get("improved")),
                "score_delta": float(payload.get("score_delta") or 0.0),
                "short_feedback": str(payload.get("short_feedback") or "").strip(),
                "remaining_gap": str(payload.get("remaining_gap") or "").strip(),
                "analogy": str(payload.get("analogy") or "").strip(),
                "final_hint": str(payload.get("final_hint") or "").strip(),
                "follow_up_complete": bool(payload.get("follow_up_complete", True)),
                "source": "llm",
            }
            return self._sanitize_llm_result_against_back(cleaned, (card or {}).get("back") or "")
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
        back = (card or {}).get("back", "")
        local_score, local_matched, local_missing, local_unsupported = self._semantic_overlap_score(back, user_answer or "")
        llm_result = self._try_llm_grade_answer(card, user_answer, answer_time_ms, provider=provider)
        if llm_result is not None:
            llm_result = self._sanitize_llm_result_against_back(llm_result, back)
            llm_score = float(llm_result.get("score") or 0.0)
            llm_grade = str(llm_result.get("grade") or "wrong")
            if local_score >= 0.55 and llm_score < 0.5:
                llm_result["grade"] = "partial"
                llm_result["score"] = round(max(llm_score, local_score), 3)
            if local_score >= 0.75 and llm_grade == "wrong":
                llm_result["grade"] = "partial"
                llm_result["score"] = round(max(float(llm_result.get("score") or 0.0), local_score), 3)
            if len(local_matched) >= 2 and llm_grade == "wrong" and float(llm_result.get("score") or 0.0) == 0.0:
                llm_result["grade"] = "partial"
                llm_result["score"] = round(max(0.35, local_score), 3)
            llm_result["matched_points"] = llm_result.get("matched_points") or local_matched
            llm_result["missing_points"] = llm_result.get("missing_points") or local_missing
            llm_result["unsupported_points"] = llm_result.get("unsupported_points") or local_unsupported
            answer_l = (user_answer or "").lower()
            for phrase in self.DANGEROUS_OUT_OF_BACK_PHRASES:
                if phrase in answer_l and phrase not in (back or "").lower():
                    llm_result["unsupported_points"] = (llm_result.get("unsupported_points") or []) + [
                        f"«{phrase}» — не точная формулировка карточки"
                    ]
            if local_unsupported and not llm_result.get("error_explanation"):
                llm_result["error_explanation"] = (
                    f"Вы правильно упомянули: {', '.join(local_matched[:3])}. "
                    f"Но фраза «{local_unsupported[0]}» не указана в карточке."
                )
            llm_result["unsupported_points"] = [str(x).strip() for x in (llm_result.get("unsupported_points") or []) if str(x).strip()][:8]
            return llm_result

        LOGGER.info("LLM недоступна, используется fallback-проверка.")
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
        semantic = self._semantic_coverage(back, answer)
        coverage = max(float(semantic["coverage"]), float(local_score))

        if answer_time_ms < 4000:
            t_quality = "fast"
        elif answer_time_ms <= 15000:
            t_quality = "normal"
        elif answer_time_ms <= 40000:
            t_quality = "slow"
        else:
            t_quality = "too_slow"

        if coverage >= 0.9:
            grade, action = "correct", "increase"
        elif coverage >= 0.6:
            grade, action = "partial", "repeat_soon"
        elif coverage >= 0.3:
            grade, action = "uncertain", "repeat_soon"
        else:
            grade, action = "wrong", "reset"

        if grade == "correct" and t_quality in {"slow", "too_slow"}:
            grade, action = "slow_correct", "slight_increase"

        mistake_type = "none" if grade in {"correct", "slow_correct"} else "missing_key_point"
        missing = local_missing or [str(item.get("label") or "").strip() for item in semantic["missing"] if str(item.get("label") or "").strip()][:5]
        matched = local_matched or [str(item.get("label") or "").strip() for item in semantic["matched"] if str(item.get("label") or "").strip()][:4]

        if grade in {"correct", "slow_correct"}:
            error_explanation = "Существенных смысловых ошибок не найдено."
        else:
            if matched and missing:
                error_explanation = (
                    f"Уже верно: {', '.join(matched[:3])}. "
                    f"Не хватает: {missing[0]}."
                )
            else:
                error_explanation = self._make_missing_explanation(back, missing)
        if local_unsupported:
            error_explanation += f" Фраза «{local_unsupported[0]}» не указана в карточке."

        result = self._result(grade, round(coverage, 3), t_quality, mistake_type)
        result["missing_points"] = missing
        result["matched_points"] = matched
        result["unsupported_points"] = local_unsupported
        result["srs_action"] = action
        result["error_explanation"] = error_explanation
        result["follow_up_question"] = self._build_follow_up_from_missing(semantic["missing"]) or self._make_follow_up_question(back)
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
            back = (card or {}).get("back") or (card or {}).get("answer") or ""
            return self._postprocess_follow_up_grade(back, follow_up_answer, llm_result)

        back = (card or {}).get("back") or (card or {}).get("answer") or ""
        combined_answer = f"{(original_answer or '').strip()} {(follow_up_answer or '').strip()}".strip()
        previous_semantic = self._semantic_coverage(back, original_answer or "")
        new_semantic = self._semantic_coverage(back, combined_answer)
        previous_score = float((previous_grade_result or {}).get("score") or previous_semantic["coverage"] or 0.0)
        new_score = float(new_semantic["coverage"])
        score_delta = round(new_score - previous_score, 3)
        newly_closed = {p.get("id") for p in previous_semantic["missing"]} & {p.get("id") for p in new_semantic["matched"]}
        improved = bool(newly_closed) or score_delta > 0.03
        if "ordinary_pass_quickly" in newly_closed and score_delta < 0.2:
            score_delta = 0.2
            improved = True

        remaining_keywords = [str(item.get("label") or "").strip() for item in new_semantic["missing"] if str(item.get("label") or "").strip()]
        remaining_gap = (
            f"Не хватает смысловых опор: {', '.join(remaining_keywords[:5])}."
            if remaining_keywords
            else "Ключевые смысловые опоры покрыты."
        )
        short_feedback = (
            "Да, теперь ты добавил недостающую деталь."
            if improved
            else "Пока прирост небольшой: добавьте точнее ключевую мысль."
        )
        if "ordinary_pass_quickly" in newly_closed:
            short_feedback = "Да, теперь ты добавил недостающую деталь: обычные реакции проходят быстро."

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
        result = {
            "improved": improved,
            "score_delta": score_delta,
            "short_feedback": short_feedback,
            "remaining_gap": remaining_gap,
            "analogy": "Это как чинить цепочку: добавили одно звено, но нужно замкнуть весь контур.",
            "final_hint": final_hint,
            "follow_up_complete": follow_up_complete,
            "source": "fallback",
        }
        return self._postprocess_follow_up_grade(back, follow_up_answer, result)

    def _postprocess_follow_up_grade(self, back: str, follow_up_answer: str, result: dict[str, Any]) -> dict[str, Any]:
        safe_result = dict(result or {})
        answer_l = (follow_up_answer or "").lower()
        back_l = (back or "").lower()
        if "медлен" in answer_l and "медлен" not in back_l:
            safe_result["improved"] = True
            safe_result["remaining_gap"] = (
                "Лучше не говорить 'медленнее', если этого нет в карточке. "
                "В карточке сказано: 'более редкие и серьёзные осложнения'."
            )
            safe_result["short_feedback"] = (
                "Стало лучше: ты добавил, что обычные реакции проходят быстро."
            )
            safe_result["final_hint"] = (
                "Скажи ближе к карточке: обычные реакции проходят быстро, "
                "а вторая группа — более редкие и серьёзные осложнения."
            )
        return safe_result

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
            "matched_points": [],
            "unsupported_points": [],
            "short_feedback": self._make_short_feedback(grade),
            "error_explanation": "",
            "analogy": "",
            "follow_up_question": "",
            "srs_action": "repeat_soon",
            "card_action": "keep",
            "suggested_rewrite": {"front": "", "back": ""},
            "source": "fallback",
        }
