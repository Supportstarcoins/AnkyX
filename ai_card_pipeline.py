from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime

from db_path import get_db_path
from image_generation_adapter import StableDiffusionAdapter
from source_extractors import clean_extracted_text, extract_text_from_path

DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, blurry, low quality, extra letters"


class AICardPipeline:
    def __init__(self, app=None, deck_id: int | None = None) -> None:
        self.app = app
        self.deck_id = deck_id
        self.image_adapter = StableDiffusionAdapter(app=app)

    def extract_text_from_source(self, source: str) -> str:
        return extract_text_from_path(source)

    def clean_text(self, text: str) -> str:
        return clean_extracted_text(text)

    def split_into_chunks(self, text: str, min_words: int = 300, max_words: int = 800) -> list[str]:
        words = (text or "").split()
        if not words:
            return []
        step = max(min_words, min(max_words, 500))
        return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]

    def split_into_semantic_blocks(self, chunks: list[str]) -> list[str]:
        return [c.strip() for c in chunks if c.strip()]

    def extract_key_facts_terms_dates_formulas(self, blocks: list[str]) -> list[dict]:
        result: list[dict] = []
        for block in blocks:
            terms = re.findall(r"\b[А-ЯA-Z][а-яa-zA-ZА-Я-]{3,}\b", block)
            dates = re.findall(r"\b\d{4}\b", block)
            formulas = re.findall(r"[A-Za-zА-Яа-я]+\s*=\s*[^\n,;]+", block)
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
            result.append({"text": block, "terms": terms[:8], "dates": dates[:8], "formulas": formulas[:4], "facts": sentences[:4]})
        return result

    def generate_card_candidates(self, blocks: list[dict]) -> list[dict]:
        cards: list[dict] = []
        for block in blocks:
            terms = block.get("terms") or []
            facts = block.get("facts") or []
            if terms:
                term = terms[0]
                back = facts[0] if facts else block.get("text", "")[:360]
                cards.append(self._build_card(f"Что такое {term}?", back, block, source=term))
            for fact in facts[:2]:
                if len(fact) < 25:
                    continue
                q = fact[:100].rstrip(".?!")
                cards.append(self._build_card(f"Объясните: {q}?", fact[:560], block, source=q))
        return cards

    def filter_and_improve_cards(self, cards: list[dict]) -> list[dict]:
        seen = set()
        filtered = []
        for card in cards:
            front = re.sub(r"\s+", " ", (card.get("front") or "")).strip()
            back = re.sub(r"\s+", " ", (card.get("back") or "")).strip()
            if len(front) < 8:
                continue
            if len(front) > 140:
                front = front[:137].rstrip() + "..."
            if len(back) < 20:
                continue
            if len(back) > 600:
                back = back[:597].rstrip() + "..."
            if "что говорится в тексте" in front.lower():
                continue
            key = (front.lower(), back.lower())
            if key in seen:
                continue
            seen.add(key)
            card["front"] = front
            card["back"] = back
            card["quality_score"] = 0.7
            if not card.get("image_prompt"):
                card["image_prompt"] = self.generate_image_prompt(card)
            filtered.append(card)
        return filtered[:20]

    def generate_image_prompt(self, card: dict) -> str:
        front = (card.get("front") or "").strip(" ?")
        return (
            f"educational realistic illustration of {front.lower()}, clean composition, "
            "no text, no watermark"
        )

    def generate_card_image(self, card: dict) -> dict:
        path, status = self.image_adapter.generate_image(card.get("image_prompt", ""), card.get("negative_prompt"))
        card["image_path"] = path
        card.setdefault("metadata", {})["image_status"] = status
        return card

    def save_cards_to_overview(self, cards: list[dict]) -> int:
        if not self.deck_id:
            raise RuntimeError("Не выбрана колода для сохранения карточек")
        db_path = get_db_path()
        now_iso = datetime.now().isoformat()
        now_ts = int(time.time())
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(cards)")
            columns = {row[1] for row in cur.fetchall()}
            saved = 0
            for c in cards:
                insert_cols = ["deck_id", "front", "back", "next_review", "leitner_level", "progress", "overview_added"]
                insert_vals = [self.deck_id, c.get("front", ""), c.get("back", ""), now_iso, 1, 0, 1]
                if "state" in columns:
                    insert_cols.append("state")
                    insert_vals.append("overview")
                if "phase" in columns:
                    insert_cols.append("phase")
                    insert_vals.append(1)
                if "due" in columns:
                    insert_cols.append("due")
                    insert_vals.append(now_ts)
                if "reps" in columns:
                    insert_cols.append("reps")
                    insert_vals.append(0)
                if "lapses" in columns:
                    insert_cols.append("lapses")
                    insert_vals.append(0)
                if "image_path" in columns:
                    insert_cols.append("image_path")
                    insert_vals.append(c.get("image_path"))
                placeholders = ", ".join(["?"] * len(insert_cols))
                cur.execute(f"INSERT INTO cards ({', '.join(insert_cols)}) VALUES ({placeholders})", tuple(insert_vals))
                saved += 1
            conn.commit()
            return saved
        finally:
            conn.close()

    def run_pipeline(self, text: str | None = None, source: str | None = None) -> list[dict]:
        source_text = (text or "").strip()
        if source and not source_text:
            source_text = self.clean_text(self.extract_text_from_source(source))
        if not source_text:
            return []

        cleaned = self.clean_text(source_text)
        words = cleaned.split()
        if len(words) <= 20:
            topic = cleaned.strip(" .,!?")
            if not topic:
                return []
            back = f"{topic.capitalize()} — краткое объяснение темы для обучения."
            return [self._build_card(f"Что такое {topic}?", back, {}, source=topic)]

        chunks = self.split_into_chunks(cleaned)
        blocks = self.split_into_semantic_blocks(chunks)
        facts = self.extract_key_facts_terms_dates_formulas(blocks)
        cards = self.generate_card_candidates(facts)
        filtered = self.filter_and_improve_cards(cards)
        if filtered:
            return filtered
        fallback_topic = cleaned.split("\n", 1)[0][:80].strip(" .,!?") or "тема"
        return [self._build_card(f"Что такое {fallback_topic}?", f"{fallback_topic.capitalize()} — базовое определение и ключевые свойства.", {}, source=fallback_topic)]

    def _build_card(self, front: str, back: str, block: dict, source: str = "") -> dict:
        return {
            "front": front,
            "back": back,
            "explanation": back,
            "image_prompt": self.generate_image_prompt({"front": front}),
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "image_path": None,
            "source": source or "ai_pipeline",
            "tags": [],
            "difficulty": "normal",
            "card_type": "qa",
            "quality_score": 0.0,
            "metadata": {
                "terms": block.get("terms", []) if isinstance(block, dict) else [],
                "dates": block.get("dates", []) if isinstance(block, dict) else [],
                "formulas": block.get("formulas", []) if isinstance(block, dict) else [],
                "facts": block.get("facts", []) if isinstance(block, dict) else [],
                "causes": [],
                "differences": [],
            },
        }
