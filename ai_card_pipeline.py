from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Any

from card_quality_filter import filter_bad_cards, polish_card, score_card
from rag_content_pipeline import clean_raw_text
from semantic_chunker import split_into_semantic_chunks


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
        if source:
            extracted = self.extract_text_from_source(source)
            raw_text = f"{raw_text}\n{extracted}".strip()
        if not raw_text:
            return []

        cleaned = clean_raw_text(html.unescape(raw_text))
        chunks = split_into_semantic_chunks(cleaned, min_words=120, max_words=500)
        units = self.extract_knowledge_units(chunks)
        candidates = self._generate_candidates(units, source_trace=source_trace or {})
        cards = self._finalize_cards(candidates, mode=mode)
        return cards[: self.max_cards]

    def generate_cards_from_text(self, text: str) -> list[dict]:
        return self.run_pipeline(source_text=text)

    def extract_text_from_source(self, source: str) -> str:
        from source_extractors import extract_text_from_source

        return extract_text_from_source(source)

    def clean_text(self, text: str | None) -> str:
        return clean_raw_text(text or "")

    def extract_knowledge_units(self, chunks: list[dict]) -> list[dict]:
        units: list[dict] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            facts = [s.strip() for s in sentences if 30 <= len(s.strip()) <= 360]
            terms = [m.group(1).strip() for m in re.finditer(r"([А-ЯA-Z][^\n:]{2,60})\s*[:\-]\s*", text)]
            dates = re.findall(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}|\d{1,2}\s+[а-яА-Яa-zA-Z]+\s+\d{4})\b", text)
            formulas = re.findall(r"\b[\wА-Яа-я]+\s*=\s*[^\n.,;]{2,80}", text)
            causes = [s for s in facts if re.search(r"(?i)потому что|из-за|вследствие|поэтому", s)]
            differences = [s for s in facts if re.search(r"(?i)в отличие от|однако|но\s|чем", s)]
            definitions = [s for s in facts if re.search(r"(?i)\bэто\b|\bназывается\b|\bопределяется\b", s)]
            lists = [s for s in facts if "," in s and len(s.split(",")) >= 3]
            units.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "concepts": list({t for t in terms[:8]}),
                    "facts": facts[:12],
                    "terms": list({t for t in terms[:8]}),
                    "dates": list(dict.fromkeys(dates))[:6],
                    "formulas": list(dict.fromkeys(formulas))[:6],
                    "causes": causes[:6],
                    "differences": differences[:6],
                    "definitions": definitions[:6],
                    "lists": lists[:6],
                    "source_excerpt": text[:420],
                    "time_start": None,
                    "time_end": None,
                }
            )
        return units

    def _generate_candidates(self, units: list[dict], source_trace: dict) -> list[dict]:
        cards: list[dict] = []
        for unit in units:
            excerpt = unit.get("source_excerpt") or ""
            chunk_id = unit.get("chunk_id") or ""
            source_type = source_trace.get("source_type") or "manual"
            source_url = source_trace.get("source_url") or ""
            source_title = source_trace.get("source_title") or ""

            for d in unit.get("definitions", [])[:3]:
                term = self._guess_subject(d)
                if not term:
                    continue
                cards.append(self._make_card(f"Что означает термин «{term}»?", d, "definition", excerpt, chunk_id, source_type, source_url, source_title, unit))
            for f in unit.get("facts", [])[:4]:
                q = self._question_from_fact(f)
                if q:
                    cards.append(self._make_card(q, f, "fact", excerpt, chunk_id, source_type, source_url, source_title, unit))
            for c in unit.get("causes", [])[:2]:
                cards.append(self._make_card("Какова указанная причина явления?", c, "cause", excerpt, chunk_id, source_type, source_url, source_title, unit))
            for diff in unit.get("differences", [])[:2]:
                cards.append(self._make_card("В чём ключевое различие, указанное в материале?", diff, "difference", excerpt, chunk_id, source_type, source_url, source_title, unit))
            for dt in unit.get("dates", [])[:2]:
                cards.append(self._make_card("Какая дата/год указаны в материале?", dt, "date", excerpt, chunk_id, source_type, source_url, source_title, unit))
            for fr in unit.get("formulas", [])[:2]:
                cards.append(self._make_card("Какая формула указана в тексте?", fr, "formula", excerpt, chunk_id, source_type, source_url, source_title, unit))
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
    ) -> dict:
        card = {
            "front": self._normalize_question(front),
            "back": back.strip(),
            "explanation": back.strip()[:240],
            "card_type": card_type,
            "difficulty": "medium",
            "quality_score": 0.0,
            "source_type": source_type,
            "source_url": source_url,
            "source_title": source_title,
            "chunk_id": chunk_id,
            "source_excerpt": excerpt,
            "time_start": unit.get("time_start"),
            "time_end": unit.get("time_end"),
            "image_prompt": "",
            "needs_image": False,
            "negative_prompt": "text, watermark, logo, blurry, low quality, extra letters",
            "image_path": "",
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
        if mode == "fast":
            pool = [score_card(polish_card(c)) for c in candidates]
            return [c for c in pool if c.get("quality_score", 0.0) >= 0.45]
        filtered = filter_bad_cards(candidates)
        if mode == "deep":
            filtered = sorted(filtered, key=lambda x: x.get("quality_score", 0.0), reverse=True)
        return filtered

    def _guess_subject(self, sentence: str) -> str:
        m = re.match(r"^([А-ЯA-Z][^\s,.;:!?]{2,40})", sentence.strip())
        return (m.group(1) if m else "").strip("-–—:;,. ")

    def _question_from_fact(self, sentence: str) -> str:
        sent = re.sub(r"\s+", " ", sentence).strip()
        if not sent:
            return ""
        if re.search(r"(?i)делится на|состоит из", sent):
            subj = self._guess_subject(sent) or "объект"
            return f"На какие части делится {subj.lower()}?"
        if re.search(r"(?i)функц|служит|предназначен", sent):
            return "Какую функцию выполняет описанный объект?"
        if re.search(r"\b\d+(?:[,.]\d+)?\b", sent):
            return "Какое числовое значение указано в материале?"
        return "Какой факт указан в материале?"

    def _normalize_question(self, question: str) -> str:
        q = re.sub(r"\s+", " ", (question or "").strip())
        if not q.endswith("?"):
            q += "?"
        return q[:150]

    def _needs_image(self, card: dict) -> bool:
        low = f"{card.get('front','')} {card.get('back','')} {card.get('card_type','')}".lower()
        visual = ("анатом", "географ", "процесс", "схем", "сравнен", "устройств", "организм", "строени")
        non_visual = ("дата", "формула", "абстракт", "определение")
        if any(x in low for x in non_visual) and card.get("card_type") in {"date", "formula", "definition"}:
            return False
        return any(x in low for x in visual) or card.get("card_type") in {"difference", "process", "list"}

    def build_image_prompt_for_card(self, card: dict) -> str:
        if not self._needs_image(card):
            return ""
        topic = card.get("source_title") or card.get("card_type") or "тема"
        return (
            "Обучающая иллюстрация для флэш-карточки. "
            f"Тема: {topic}. "
            f"Вопрос: {card.get('front','')}. "
            f"Ответ: {card.get('back','')}. "
            f"Визуально показать: {card.get('explanation','')}. "
            "Стиль: простая понятная учебная схема, без лишнего текста, без перегруза."
        )[:900]

    def generate_image_prompt(self, card: dict) -> str:
        return self.build_image_prompt_for_card(card)

    def generate_card_image(self, card: dict) -> dict:
        card = dict(card)
        metadata = dict(card.get("metadata") or {})
        if not card.get("needs_image", self._needs_image(card)):
            metadata["image_status"] = "Для этой карточки картинка не обязательна"
            card["metadata"] = metadata
            return card
        from image_generation_adapter import StableDiffusionAdapter

        try:
            adapter = StableDiffusionAdapter(app=self.app)
            path, status = adapter.generate_image(
                card.get("image_prompt") or self.build_image_prompt_for_card(card),
                card.get("negative_prompt") or "text, watermark, logo, blurry, low quality",
            )
            card["image_path"] = path or ""
            metadata["image_status"] = status or ("Stable Diffusion недоступен" if not path else "Изображение создано")
        except Exception as exc:
            metadata["image_status"] = f"Stable Diffusion недоступен: {exc}"
        card["metadata"] = metadata
        return card

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
                        "source_excerpt": card.get("source_excerpt"),
                        "time_start": card.get("time_start"),
                        "time_end": card.get("time_end"),
                    },
                    "ai": card.get("metadata") or {},
                }
                values = {
                    "deck_id": deck_id,
                    "front": front,
                    "back": back,
                    "next_review": now_iso,
                    "leitner_level": 1,
                    "front_image_path": card.get("image_path"),
                    "back_image_path": None,
                    "image_path": card.get("image_path"),
                    "translation_shown": 1,
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
