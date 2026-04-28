from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Any

from card_quality_filter import STOP_TERMS, filter_bad_cards, polish_card, score_card
from rag_content_pipeline import attach_best_source_images, clean_raw_text
from semantic_chunker import split_into_semantic_chunks

ALLOWED_CARD_TYPES = {
    "definition",
    "function",
    "range/quantity",
    "cause",
    "difference",
    "list/classification",
    "anatomy/composition",
    "fact",
}


class AICardPipeline:
    def __init__(self, app: Any | None = None, deck_id: int | None = None) -> None:
        self.app = app
        self.deck_id = deck_id
        self.max_cards = 24

    def run_pipeline(
        self,
        source_text: str | None = None,
        source_trace: dict | None = None,
        options: dict | None = None,
        text: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        opts = dict(options or {})
        mode = (opts.get("mode") or "accurate").lower()
        raw_text = (source_text or "").strip()
        if text:
            raw_text = f"{raw_text}\n{text}".strip()
        extracted_images: list[dict] = list((source_trace or {}).get("images") or [])
        if source:
            bundle = self.extract_source_bundle(source)
            raw_text = f"{raw_text}\n{bundle.get('text','')}".strip()
            extracted_images = list(bundle.get("images") or [])
        if not raw_text:
            return []

        cleaned = clean_raw_text(html.unescape(raw_text))
        chunks = split_into_semantic_chunks(cleaned, min_words=90, max_words=320)
        units = self.extract_knowledge_units(chunks)
        candidates = self._generate_candidates(units, source_trace=source_trace or {})
        cards = self._finalize_cards(candidates, mode=mode)
        cards = attach_best_source_images(cards, extracted_images)
        cards = [self._ensure_answer_image(card) for card in cards]
        return cards[: self.max_cards]

    def generate_cards_from_text(self, text: str) -> list[dict]:
        return self.run_pipeline(source_text=text)

    def extract_text_from_source(self, source: str) -> str:
        from source_extractors import extract_text_from_source

        return extract_text_from_source(source)

    def extract_source_bundle(self, source: str) -> dict:
        from source_extractors import extract_source_bundle

        return extract_source_bundle(source)

    def clean_text(self, text: str | None) -> str:
        return clean_raw_text(text or "")

    def extract_knowledge_units(self, chunks: list[dict]) -> list[dict]:
        units: list[dict] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            facts = [s.strip() for s in sentences if 25 <= len(s.strip()) <= 360]
            terms = [m.group(1).strip() for m in re.finditer(r"([А-ЯA-Z][^\n:]{2,60})\s*[:\-]\s*", text)]
            dates = re.findall(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}|\d{1,2}\s+[а-яА-Яa-zA-Z]+\s+\d{4})\b", text)
            formulas = re.findall(r"\b[\wА-Яа-я]+\s*=\s*[^\n.,;]{2,80}", text)
            causes = [s for s in facts if re.search(r"(?i)потому что|из-за|вследствие|поэтому|приводит к", s)]
            differences = [s for s in facts if re.search(r"(?i)в отличие от|отличается|тогда как|вместо", s)]
            definitions = [s for s in facts if re.search(r"(?i)\bэто\b|\bназывается\b|\bопределяется\b", s)]
            lists = [s for s in facts if "," in s and len(s.split(",")) >= 3]
            functions = [s for s in facts if re.search(r"(?i)используется|служит|предназначен|выполняет функцию|нужен для", s)]
            ranges = [s for s in facts if re.search(r"\b\d+(?:[,.]\d+)?\s*(%|км|м|см|мм|кг|г|л|мл|°c|гр|лет|раз|x)?\b", s.lower())]
            anatomy = [s for s in facts if re.search(r"(?i)состоит из|включает|част[ьи]|структур|компонент|элемент", s)]
            topic = (chunk.get("topic_title") or self._infer_topic_from_text(text) or "Общая тема").strip()
            units.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "topic": topic,
                    "concepts": list({t for t in terms[:10]}),
                    "facts": facts[:14],
                    "terms": list({t for t in terms[:10]}),
                    "dates": list(dict.fromkeys(dates))[:6],
                    "formulas": list(dict.fromkeys(formulas))[:6],
                    "causes": causes[:6],
                    "differences": differences[:6],
                    "definitions": definitions[:6],
                    "lists": lists[:6],
                    "functions": functions[:6],
                    "ranges": ranges[:6],
                    "anatomy": anatomy[:6],
                    "source_excerpt": text[:520],
                    "time_start": None,
                    "time_end": None,
                }
            )
        return units

    def _generate_candidates(self, units: list[dict], source_trace: dict) -> list[dict]:
        cards: list[dict] = []
        for unit in units:
            llm_cards = self._generate_cards_with_llm(unit, source_trace)
            if llm_cards:
                cards.extend(llm_cards)
                continue
            cards.extend(self._generate_candidates_fallback(unit, source_trace))
        return cards

    def _generate_cards_with_llm(self, unit: dict, source_trace: dict) -> list[dict]:
        settings = self._get_llm_settings()
        if not settings.get("enabled"):
            return []
        try:
            import requests
        except Exception:
            return []

        prompt = self._build_llm_prompt(unit)
        messages = [
            {"role": "system", "content": "Ты создаёшь флэш-карточки в строгом JSON-массиве без пояснений."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = requests.post(
                f"{settings['base_url'].rstrip('/')}/api/chat",
                json={"model": settings["model"], "messages": messages, "stream": False},
                timeout=settings.get("timeout", 45),
            )
            if not resp.ok:
                return []
            raw = str(((resp.json() or {}).get("message") or {}).get("content") or "").strip()
            payload = self._extract_json_array(raw)
            parsed = json.loads(payload)
            if not isinstance(parsed, list):
                return []
        except Exception:
            return []

        cards: list[dict] = []
        for item in parsed[:8]:
            if not isinstance(item, dict):
                continue
            card_type = str(item.get("card_type") or "fact").strip().lower()
            if card_type not in ALLOWED_CARD_TYPES:
                card_type = "fact"
            card = self._make_card(
                item.get("front") or "",
                item.get("back") or "",
                card_type,
                item.get("source_excerpt") or unit.get("source_excerpt") or "",
                unit.get("chunk_id") or "",
                source_trace.get("source_type") or "manual",
                source_trace.get("source_url") or "",
                source_trace.get("source_title") or "",
                unit,
                topic=item.get("topic") or unit.get("topic") or "",
            )
            card["explanation"] = str(item.get("explanation") or card.get("back") or "")[:240]
            card["difficulty"] = str(item.get("difficulty") or "medium")
            card["needs_image"] = bool(item.get("needs_image", card.get("needs_image")))
            card["image_prompt"] = str(item.get("image_prompt") or "").strip() or self.build_image_prompt_for_card(card)
            cards.append(card)
        return cards

    def _generate_candidates_fallback(self, unit: dict, source_trace: dict) -> list[dict]:
        cards: list[dict] = []
        excerpt = unit.get("source_excerpt") or ""
        chunk_id = unit.get("chunk_id") or ""
        topic = unit.get("topic") or ""
        source_type = source_trace.get("source_type") or "manual"
        source_url = source_trace.get("source_url") or ""
        source_title = source_trace.get("source_title") or ""

        for d in unit.get("definitions", [])[:3]:
            maybe = self._definition_card(d)
            if maybe:
                q, a = maybe
                cards.append(self._make_card(q, a, "definition", excerpt, chunk_id, source_type, source_url, source_title, unit, topic=topic))

        for c in unit.get("causes", [])[:3]:
            maybe = self._cause_card(c)
            if maybe:
                q, a = maybe
                cards.append(self._make_card(q, a, "cause", excerpt, chunk_id, source_type, source_url, source_title, unit, topic=topic))

        for f in unit.get("facts", [])[:6]:
            maybe = self._function_or_fact_card(f)
            if maybe:
                q, a, ctype = maybe
                cards.append(self._make_card(q, a, ctype, excerpt, chunk_id, source_type, source_url, source_title, unit, topic=topic))

        for fn in unit.get("functions", [])[:2]:
            maybe = self._function_card(fn)
            if maybe:
                q, a = maybe
                cards.append(self._make_card(q, a, "function", excerpt, chunk_id, source_type, source_url, source_title, unit, topic=topic))

        for rg in unit.get("ranges", [])[:2]:
            maybe = self._range_card(rg)
            if maybe:
                q, a = maybe
                cards.append(self._make_card(q, a, "range/quantity", excerpt, chunk_id, source_type, source_url, source_title, unit, topic=topic))

        for diff in unit.get("differences", [])[:2]:
            maybe = self._difference_card(diff)
            if maybe:
                q, a = maybe
                cards.append(self._make_card(q, a, "difference", excerpt, chunk_id, source_type, source_url, source_title, unit, topic=topic))

        for lst in unit.get("lists", [])[:2]:
            maybe = self._list_card(lst)
            if maybe:
                q, a = maybe
                cards.append(self._make_card(q, a, "list/classification", excerpt, chunk_id, source_type, source_url, source_title, unit, topic=topic))

        for an in unit.get("anatomy", [])[:2]:
            maybe = self._anatomy_card(an)
            if maybe:
                q, a = maybe
                cards.append(self._make_card(q, a, "anatomy/composition", excerpt, chunk_id, source_type, source_url, source_title, unit, topic=topic))
        return cards

    def _make_card(
        self,
        front: str,
        back: str,
        card_type: str,
        excerpt: str,
        chunk_id: str,
        source_type: str,
        source_url: str,
        source_title: str,
        unit: dict,
        topic: str = "",
    ) -> dict:
        card = {
            "front": self._normalize_question(front),
            "back": (back or "").strip(),
            "explanation": (back or "").strip()[:240],
            "card_type": card_type,
            "difficulty": "medium",
            "quality_score": 0.0,
            "source_type": source_type,
            "source_url": source_url,
            "source_title": source_title,
            "chunk_id": chunk_id,
            "topic": (topic or unit.get("topic") or source_title or "Общая тема").strip(),
            "source_excerpt": excerpt,
            "time_start": unit.get("time_start"),
            "time_end": unit.get("time_end"),
            "image_prompt": "",
            "needs_image": False,
            "negative_prompt": "text, watermark, logo, blurry, low quality, extra letters",
            "image_path": "",
            "answer_image_path": "",
            "answer_image_url": "",
            "answer_image_caption": "",
            "image_source_type": "",
            "image_relevance_score": 0.0,
            "metadata": {
                "terms": unit.get("terms", []),
                "dates": unit.get("dates", []),
                "formulas": unit.get("formulas", []),
                "facts": unit.get("facts", []),
                "causes": unit.get("causes", []),
                "differences": unit.get("differences", []),
            },
        }
        card["needs_image"] = self._needs_image(card)
        card["image_prompt"] = self.build_image_prompt_for_card(card)
        return card

    def _finalize_cards(self, candidates: list[dict], mode: str = "accurate") -> list[dict]:
        cleaned = [c for c in candidates if self._is_card_semantically_valid(c)]
        if mode == "fast":
            pool = [score_card(polish_card(c)) for c in cleaned]
            strong = [c for c in pool if c.get("quality_score", 0.0) >= 0.55]
            if len(strong) >= 3:
                return strong
            return [c for c in pool if c.get("quality_score", 0.0) >= 0.45]
        filtered = filter_bad_cards(cleaned)
        if mode == "deep":
            filtered = sorted(filtered, key=lambda x: x.get("quality_score", 0.0), reverse=True)
        return filtered

    def _guess_subject(self, sentence: str) -> str:
        m = re.match(r"^([А-ЯA-Z][^\s,.;:!?]{2,40})", sentence.strip())
        return (m.group(1) if m else "").strip("-–—:;,. ")

    def _normalize_question(self, question: str) -> str:
        q = re.sub(r"\s+", " ", (question or "").strip())
        if not q:
            return ""
        if not q.endswith("?"):
            q += "?"
        return q[:170]

    def _needs_image(self, card: dict) -> bool:
        low = f"{card.get('front','')} {card.get('back','')} {card.get('card_type','')}".lower()
        visual = ("анатом", "географ", "процесс", "схем", "сравнен", "устройств", "организм", "строени", "хобот", "механизм")
        non_visual = ("дата", "формула", "абстракт", "определение")
        if any(x in low for x in non_visual) and card.get("card_type") in {"date", "formula", "definition"}:
            return False
        return any(x in low for x in visual) or card.get("card_type") in {"difference", "process", "list"}

    def build_image_prompt_for_card(self, card: dict) -> str:
        topic = card.get("topic") or card.get("source_title") or "тема"
        front = card.get("front") or ""
        back = card.get("back") or ""
        if card.get("needs_image"):
            return (
                "Обучающая иллюстрация: "
                f"тема «{topic}». "
                f"Покажи сцену, которая помогает запомнить факт: {front} Ответ: {back}. "
                "Стиль: чистый фон, понятная схема, без лишнего текста, высокий контраст, фокус на главном объекте."
            )[:900]
        return (
            "Символическая обучающая иллюстрация для запоминания: "
            f"тема «{topic}», вопрос: {front}. "
            "Минималистичная схема или метафора без перегрузки, без текста на изображении."
        )[:900]

    def generate_image_prompt(self, card: dict) -> str:
        return self.build_image_prompt_for_card(card)

    def _ensure_answer_image(self, card: dict) -> dict:
        c = dict(card or {})
        has_extracted = bool(c.get("answer_image_path") or c.get("answer_image_url"))
        if has_extracted:
            c["image_source_type"] = "extracted"
            return c
        if not c.get("needs_image"):
            c["image_source_type"] = c.get("image_source_type") or "none"
            return c
        c["image_prompt"] = c.get("image_prompt") or self.build_image_prompt_for_card(c)
        generated = self.generate_card_image(c)
        if generated.get("image_path"):
            generated["answer_image_path"] = generated.get("image_path") or ""
            generated["image_source_type"] = "generated"
            generated["answer_image_caption"] = generated.get("answer_image_caption") or "Generated from answer text"
            return generated
        c["image_source_type"] = "recommended"
        return c

    def generate_card_image(self, card: dict) -> dict:
        card = dict(card)
        metadata = dict(card.get("metadata") or {})
        from image_generation_adapter import StableDiffusionAdapter

        try:
            adapter = StableDiffusionAdapter(app=self.app)
            path, status = adapter.generate_image(
                card.get("image_prompt") or self.build_image_prompt_for_card(card),
                card.get("negative_prompt") or "text, watermark, logo, blurry, low quality",
            )
            card["image_path"] = path or ""
            if path:
                card["answer_image_path"] = path
                card["image_source_type"] = "generated"
                card["image_relevance_score"] = float(card.get("image_relevance_score") or 0.0)
            metadata["image_status"] = status or ("Stable Diffusion недоступен" if not path else "Изображение создано")
        except Exception as exc:
            metadata["image_status"] = f"Stable Diffusion недоступен: {exc}"
        card["metadata"] = metadata
        return card

    def _definition_card(self, sentence: str) -> tuple[str, str] | None:
        sent = re.sub(r"\s+", " ", sentence).strip()
        m = re.match(r"^([А-ЯA-Zа-яa-z0-9\- ]{2,70})\s*[—-]\s*это\s+(.+)$", sent, flags=re.IGNORECASE)
        if not m:
            return None
        term = m.group(1).strip(" .,:;").lower()
        if term in STOP_TERMS or len(term) < 2:
            return None
        term_human = m.group(1).strip(" .,:;")
        return f"Что такое {term_human}?", m.group(2).strip()

    def _cause_card(self, sentence: str) -> tuple[str, str] | None:
        sent = re.sub(r"\s+", " ", sentence).strip()
        m = re.search(r"^(.+?)\s+(?:происходит|возникает|случается|усиливается)\s+из-за\s+(.+)$", sent, flags=re.IGNORECASE)
        if m:
            subject = m.group(1).strip(" .,:;")
            return f"Почему происходит {subject.lower()}?", f"Из-за {m.group(2).strip()}"
        if re.search(r"(?i)потому что|из-за|вследствие|приводит к", sent):
            subject = self._guess_subject(sent)
            if subject and subject.lower() not in STOP_TERMS:
                return f"Почему {subject.lower()}?", sent
        return None

    def _difference_card(self, sentence: str) -> tuple[str, str] | None:
        sent = re.sub(r"\s+", " ", sentence).strip()
        m = re.search(r"^(.+?)\s+отличается\s+от\s+(.+?)\s+тем,?\s+что\s+(.+)$", sent, flags=re.IGNORECASE)
        if m:
            a, b, diff = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            return f"Чем {a} отличается от {b}?", diff
        return None

    def _list_card(self, sentence: str) -> tuple[str, str] | None:
        sent = re.sub(r"\s+", " ", sentence).strip()
        m = re.match(r"^([А-ЯA-Zа-яa-z0-9\- ]{2,70})\s*:\s*(.+)$", sent)
        if m and len(m.group(2).split(",")) >= 3:
            subj = m.group(1).strip()
            return f"Какие элементы входят в {subj.lower()}?", m.group(2).strip()
        return None

    def _function_or_fact_card(self, sentence: str) -> tuple[str, str, str] | None:
        sent = re.sub(r"\s+", " ", sentence).strip()
        func = re.search(r"^(.+?)\s+(?:используется|применяется|нужен|служит|предназначен)\s+для\s+(.+)$", sent, flags=re.IGNORECASE)
        if func:
            subj = func.group(1).strip(" .,:;")
            if subj.lower() in STOP_TERMS:
                return None
            return f"Какую функцию выполняет {subj.lower()}?", f"Для {func.group(2).strip()}", "function"
        if re.search(r"(?i)в отличие от|отличается", sent):
            diff = self._difference_card(sent)
            if diff:
                return diff[0], diff[1], "difference"
        subj = self._extract_focus_term(sent)
        if not subj:
            return None
        if re.search(r"\b\d+(?:[,.]\d+)?\b", sent):
            return f"Какой диапазон или количество указаны для {subj.lower()}?", sent, "range/quantity"
        return f"Какой факт о {subj.lower()} подтверждён в источнике?", sent, "fact"

    def _function_card(self, sentence: str) -> tuple[str, str] | None:
        sent = re.sub(r"\s+", " ", sentence).strip()
        subj = self._extract_focus_term(sent)
        if not subj:
            return None
        if re.search(r"(?i)используется|служит|предназначен|нужен для|выполняет", sent):
            return f"Какую функцию выполняет {subj.lower()}?", sent
        return None

    def _range_card(self, sentence: str) -> tuple[str, str] | None:
        sent = re.sub(r"\s+", " ", sentence).strip()
        subj = self._extract_focus_term(sent)
        if not subj:
            return None
        if re.search(r"\d", sent):
            return f"Какой диапазон или количество связаны с {subj.lower()}?", sent
        return None

    def _anatomy_card(self, sentence: str) -> tuple[str, str] | None:
        sent = re.sub(r"\s+", " ", sentence).strip()
        subj = self._extract_focus_term(sent)
        if not subj:
            return None
        if re.search(r"(?i)состоит из|включает|част[ьи]|компонент|структур", sent):
            return f"Из каких частей состоит {subj.lower()}?", sent
        return None

    def _extract_focus_term(self, sentence: str) -> str:
        tokens = re.findall(r"[А-ЯA-Zа-яa-z][а-яa-z0-9\-]{2,}", sentence)
        for tok in tokens:
            low = tok.lower()
            if low not in STOP_TERMS and low not in {"которые", "который", "которая", "можно", "нужно"}:
                return tok
        return ""

    def _is_card_semantically_valid(self, card: dict) -> bool:
        front = str(card.get("front") or "").lower()
        back = str(card.get("back") or "").strip()
        excerpt = str(card.get("source_excerpt") or "").lower()
        topic = str(card.get("topic") or "").strip()
        if not front or not back or not excerpt or not topic:
            return False
        if any(p in front for p in ("какой факт указан", "что означает термин", "какова указанная причина", "что описано в тексте", "что важно помнить", "что конкретно сказано", "что известно про")):
            return False
        tokens = re.findall(r"[а-яa-z0-9]+", front)
        if len([t for t in tokens if len(t) >= 3]) < 2:
            return False
        if sum(1 for t in tokens if len(t) <= 2) > max(3, len(tokens) // 2):
            return False
        if re.search(r"термин\s+[«\"]?(это|этот|эта|эти|он|она|они|оно)", front):
            return False
        terms = [t for t in re.findall(r"[а-яa-z0-9]{4,}", front) if t not in STOP_TERMS]
        if not terms or not any(t in excerpt for t in terms[:5]):
            return False
        if re.sub(r"\W+", "", front) == re.sub(r"\W+", "", back.lower()):
            return False
        return True

    def _infer_topic_from_text(self, text: str) -> str:
        candidates = re.findall(r"[А-ЯA-Zа-яa-z][а-яa-z0-9\-]{3,}", text)
        if not candidates:
            return ""
        freq: dict[str, int] = {}
        for tok in candidates:
            low = tok.lower()
            if low in STOP_TERMS:
                continue
            freq[low] = freq.get(low, 0) + 1
        if not freq:
            return ""
        return max(freq.items(), key=lambda x: x[1])[0].capitalize()

    def _get_llm_settings(self) -> dict:
        model = self._read_setting("ollama_model", default="llama3.1:8b")
        base_url = self._read_setting("ollama_url", default="http://127.0.0.1:11434")
        enabled = bool(base_url and model)
        return {"enabled": enabled, "model": str(model), "base_url": str(base_url), "timeout": 45}

    def _read_setting(self, name: str, default: str = "") -> str:
        for owner in (self.app, getattr(self.app, "settings", None), getattr(self.app, "llm_settings", None)):
            if owner is None:
                continue
            try:
                if isinstance(owner, dict) and name in owner:
                    value = owner.get(name)
                elif hasattr(owner, name):
                    value = getattr(owner, name)
                else:
                    continue
                if hasattr(value, "get") and callable(value.get):
                    value = value.get()
                if value not in (None, ""):
                    return str(value)
            except Exception:
                continue
        return default

    def _build_llm_prompt(self, unit: dict) -> str:
        return (
            "Ты создаёшь флэш-карточки для запоминания.\n"
            "Создавай только конкретные вопросы по фактам из текста.\n"
            "Запрещено создавать общие вопросы:\n"
            "- \"Какой факт указан в материале?\"\n"
            "- \"Что конкретно сказано...?\"\n"
            "- \"Что известно про...?\"\n"
            "- \"Что описано в тексте?\"\n"
            "- \"Что важно помнить?\"\n"
            "- \"Какова указанная причина явления?\"\n"
            "- \"Что означает термин «Это»?\"\n"
            "Каждая карточка: один вопрос, один короткий ответ, ответ строго из source_excerpt, не более 1 факта на карточку.\n"
            "Вопрос должен содержать конкретное понятие. Нельзя использовать местоимения как термин.\n"
            "Если фрагмент слабый/рекламный — пропусти его.\n"
            "Верни ТОЛЬКО JSON-массив такого вида:\n"
            "[{\"front\":\"...\",\"back\":\"...\",\"explanation\":\"...\",\"card_type\":\"definition|function|range/quantity|cause|difference|list/classification|anatomy/composition|fact\",\"difficulty\":\"easy|medium|hard\",\"source_excerpt\":\"...\",\"topic\":\"...\",\"needs_image\":true,\"image_prompt\":\"...\"}]\n\n"
            f"topic={unit.get('topic','')}\n"
            f"chunk_id={unit.get('chunk_id','')}\n"
            f"source_excerpt={unit.get('source_excerpt','')}"
        )

    def _extract_json_array(self, raw: str) -> str:
        raw = (raw or "").strip()
        if raw.startswith("[") and raw.endswith("]"):
            return raw
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if not m:
            raise ValueError("JSON array not found")
        return m.group(0)

    def save_cards_to_overview(self, cards: list[dict]) -> int:
        deck_id = self.deck_id or getattr(self.app, "selected_deck_id", None)
        conn = self._open_connection()
        try:
            conn.row_factory = sqlite3.Row
            if deck_id is None:
                row = conn.execute("SELECT id FROM decks ORDER BY id LIMIT 1").fetchone()
                if row:
                    deck_id = int(row["id"] if isinstance(row, sqlite3.Row) else row[0])
            if deck_id is None:
                raise RuntimeError("Не выбрана колода для сохранения карточек")
            columns = self._table_columns(conn, "cards")
            now_iso = datetime.now().isoformat()
            now_ts = int(time.time())
            saved = 0
            for card in cards:
                front = (card.get("front") or "").strip()
                back = (card.get("back") or "").strip()
                if not front or not back:
                    continue
                meta_payload = {
                    "source_trace": {
                        "source_url": card.get("source_url"),
                        "source_title": card.get("source_title"),
                        "source_type": card.get("source_type"),
                        "chunk_id": card.get("chunk_id"),
                        "topic": card.get("topic"),
                        "source_excerpt": card.get("source_excerpt"),
                        "time_start": card.get("time_start"),
                        "time_end": card.get("time_end"),
                    },
                    "ai": {
                        **(card.get("metadata") or {}),
                        "answer_image_path": card.get("answer_image_path"),
                        "answer_image_url": card.get("answer_image_url"),
                        "answer_image_caption": card.get("answer_image_caption"),
                        "image_source_type": card.get("image_source_type"),
                        "image_relevance_score": card.get("image_relevance_score"),
                    },
                    "media": {
                        "audio_path": card.get("audio_path"),
                        "video_path": card.get("video_path"),
                        "front_image_path": card.get("front_image_path"),
                        "language": card.get("language"),
                        "translation": card.get("translation"),
                        "time_start": card.get("time_start"),
                        "time_end": card.get("time_end"),
                    },
                }
                values = {
                    "deck_id": deck_id,
                    "front": front,
                    "back": back,
                    "next_review": now_iso,
                    "leitner_level": 1,
                    "front_image_path": card.get("front_image_path") or card.get("image_path"),
                    "back_image_path": None,
                    "image_path": card.get("image_path"),
                    "translation_shown": 1,
                    "translation": card.get("translation"),
                    "overview_added": 1,
                    "state": "overview" if "state" in columns else "new",
                    "phase": 1,
                    "due": now_ts,
                    "interval": 0,
                    "ease": 2500,
                    "reps": 0,
                    "lapses": 0,
                    "step_index": 0,
                    "last_review": None,
                    "metadata": json.dumps(meta_payload, ensure_ascii=False),
                    "extra": json.dumps(meta_payload, ensure_ascii=False),
                    "audio_path": card.get("audio_path"),
                    "video_path": card.get("video_path"),
                    "source_url": card.get("source_url"),
                    "source_title": card.get("source_title"),
                    "time_start": card.get("time_start"),
                    "time_end": card.get("time_end"),
                    "language": card.get("language"),
                }
                insert_cols = [c for c in values if c in columns]
                if not insert_cols:
                    raise RuntimeError("Таблица cards не содержит ожидаемых колонок")
                conn.execute(
                    f"INSERT INTO cards ({', '.join(insert_cols)}) VALUES ({', '.join('?' for _ in insert_cols)})",
                    [values[c] for c in insert_cols],
                )
                saved += 1
            conn.commit()
            self._refresh_counters_safe()
            return saved
        finally:
            conn.close()

    def _refresh_counters_safe(self) -> None:
        app = self.app
        for method_name in ("refresh_deck_counters_and_phase_tree", "refresh_decks"):
            method = getattr(app, method_name, None)
            if callable(method):
                try:
                    method()
                    return
                except Exception:
                    continue

    def _open_connection(self) -> sqlite3.Connection:
        try:
            from db_connect import open_db

            return open_db()
        except Exception:
            try:
                from db_path import get_db_path

                return sqlite3.connect(get_db_path())
            except Exception:
                return sqlite3.connect(os.path.join(os.getcwd(), "xflash.db"))

    def _table_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        try:
            return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        except Exception:
            return set()


def generate_cards_from_text(text: str) -> list[dict]:
    return AICardPipeline().generate_cards_from_text(text)
