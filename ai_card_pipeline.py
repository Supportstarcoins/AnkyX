from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Any, Iterable

try:
    from source_extractors import extract_text_from_source as _extract_text_from_source
except Exception:
    _extract_text_from_source = None

try:
    from image_generation_adapter import StableDiffusionAdapter
except Exception:
    StableDiffusionAdapter = None


class AICardPipeline:
    """Strict text -> clean facts -> good front/back cards pipeline.

    This fallback generator intentionally avoids generic questions like
    "Что важно помнить о понятии ...". If a sentence cannot produce a specific
    useful question, it is skipped instead of creating a bad flashcard.
    """

    _DOMAIN_RE = re.compile(
        r"(?i)\b(?:https?://|www\.)\S+|\b[\w.-]+\.(?:ru|com|org|net|edu|gov|info|io|me|tv|de|uk|fr|se|su|рф)(?:/\S*)?"
    )
    _BAD_LINE_RE = re.compile(
        r"(?i)\b(duckduckgo|google|yandex|youtube|vk\.com|telegram|facebook|instagram|tiktok|reddit|pinterest|cookie|privacy|subscribe|подпис|реклама|войти|регистрация|меню|навигац|комментар|смотреть видео|скачать|share|login|sign in|результаты поиска|похожие запросы)\b"
    )
    _CYR_RE = re.compile(r"[А-Яа-яЁё]")
    _LAT_RE = re.compile(r"[A-Za-z]")
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-ZЁ0-9])")

    def __init__(self, app: Any | None = None, deck_id: int | None = None) -> None:
        self.app = app
        self.deck_id = deck_id
        self.max_cards = 16
        self.min_answer_chars = 20
        self.max_answer_chars = 620
        self._active_topic = "темы"
        self._bad_terms = {
            "особенности", "размеры", "выделить", "понятие", "понятии", "описание",
            "материалы", "информация", "источник", "страница", "статья", "раздел",
            "который", "которая", "которые", "которых", "например", "поэтому",
            "однако", "также", "может", "можно", "нужно", "важно", "каждый",
            "данный", "данная", "данные", "этого", "этот", "этой", "такие",
            "традиционно", "описанные", "описанная", "описанное", "структура",
            "признак", "явление", "части", "часть", "каждая", "каждый",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_pipeline(self, text: str | None = None, source: str | None = None) -> list[dict]:
        raw_text = ""
        if source:
            try:
                raw_text = self.extract_text_from_source(source)
            except Exception:
                logging.exception("extract_text_from_source failed; falling back to provided text")
        if text:
            raw_text = (raw_text + "\n" + text).strip() if raw_text else text
        return self.generate_cards_from_text(raw_text)

    def generate_cards_from_text(self, text: str) -> list[dict]:
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        self._active_topic = self._detect_global_topic(cleaned)

        if self._looks_like_short_topic(cleaned):
            return self.filter_and_improve_cards([self._topic_card(cleaned)])

        chunks = self.split_into_chunks(cleaned, min_words=80, max_words=420)
        blocks = self.split_into_semantic_blocks(chunks)
        facts = self.extract_key_facts_terms_dates_formulas(blocks)
        cards = self.generate_card_candidates(facts)

        if len(cards) < 3:
            facts = self.extract_key_facts_terms_dates_formulas([cleaned])
            cards.extend(self.generate_card_candidates(facts))

        return self.filter_and_improve_cards(cards)

    def extract_text_from_source(self, source: str) -> str:
        if _extract_text_from_source is None:
            raise RuntimeError("Модуль source_extractors.py недоступен")
        return _extract_text_from_source(source)

    # ------------------------------------------------------------------
    # Cleaning / splitting / fact extraction
    # ------------------------------------------------------------------
    def clean_text(self, text: str | None) -> str:
        text = html.unescape(text or "")
        text = text.replace("\r", "\n")
        text = re.sub(r"[\u200b\ufeff]", "", text)
        text = re.sub(r"(?i)<script.*?</script>|<style.*?</style>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)

        lines: list[str] = []
        seen: set[str] = set()
        for raw_line in text.split("\n"):
            line = raw_line.strip(" \t•·-*—–")
            if not line:
                continue
            line = self._DOMAIN_RE.sub(" ", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line or self._is_noise_line(line):
                continue
            key = self._dedupe_key(line)
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    def _is_noise_line(self, line: str) -> bool:
        low = line.lower().strip()
        if len(low) < 18 and not re.search(r"\d", low):
            return True
        if self._BAD_LINE_RE.search(low):
            return True
        punct_ratio = sum(1 for ch in line if ch in "/\\|{}[]<>_") / max(1, len(line))
        if punct_ratio > 0.08:
            return True
        letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", line))
        if letters < 12:
            return True
        # SEO titles without a predicate are not material for a flashcard.
        verbs = r"\b(является|это|делится|состоит|имеет|относятся|относится|называется|называют|покрыт|покрыты|покрыто|защищает|помогает|помогают|может|могут|питается|жив[её]т|обитает|размножается|меняет|использует|служит|отличается|соединены|соединяется|колебаться|способен|способна|увеличиваться)\b"
        if len(line.split()) <= 7 and not re.search(verbs, low) and not re.search(r"\d", low):
            return True
        return False

    def _dedupe_key(self, line: str) -> str:
        return re.sub(r"\W+", "", line.lower())[:180]

    def split_into_chunks(self, text: str, min_words: int = 300, max_words: int = 800) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{1,2}", text) if p.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_words = 0
        for paragraph in paragraphs:
            words = paragraph.split()
            if not words:
                continue
            if current and current_words + len(words) > max_words:
                chunks.append("\n".join(current).strip())
                current = []
                current_words = 0
            current.append(paragraph)
            current_words += len(words)
            if current_words >= min_words:
                chunks.append("\n".join(current).strip())
                current = []
                current_words = 0
        if current:
            chunks.append("\n".join(current).strip())
        return [c for c in chunks if c]

    def split_into_semantic_blocks(self, chunks: Iterable[str]) -> list[str]:
        blocks: list[str] = []
        for chunk in chunks:
            paragraphs = [p.strip() for p in chunk.split("\n") if p.strip()]
            buf: list[str] = []
            for p in paragraphs:
                buf.append(p)
                if len(" ".join(buf)) > 900:
                    blocks.append(" ".join(buf).strip())
                    buf = []
            if buf:
                blocks.append(" ".join(buf).strip())
        return [b for b in blocks if len(b) >= 45]

    def extract_key_facts_terms_dates_formulas(self, blocks: Iterable[str]) -> list[dict]:
        result: list[dict] = []
        for block in blocks:
            sentences = self._split_sentences(block)
            for sentence in sentences:
                if not self._is_good_fact_sentence(sentence):
                    continue
                result.append({
                    "text": sentence,
                    "terms": self._extract_terms(sentence),
                    "dates": re.findall(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{3,4})\b", sentence)[:4],
                    "formulas": re.findall(r"\b[A-Za-zА-Яа-яЁё0-9_]+\s*[=+\-*/^]\s*[A-Za-zА-Яа-яЁё0-9_+\-*/^()]+", sentence)[:4],
                    "numbers": re.findall(r"\b\d+(?:[,.]\d+)?\s*(?:%|см|мм|м|кг|г|лет|раз|пар|ног|глаз)?\b", sentence.lower())[:4],
                })
        return result

    def _split_sentences(self, text: str) -> list[str]:
        prepared = re.sub(r"\s+", " ", text).strip()
        if not prepared:
            return []
        parts = self._SENTENCE_SPLIT_RE.split(prepared)
        result: list[str] = []
        for part in parts:
            part = part.strip(" -–—")
            if not part:
                continue
            if len(part) > 420:
                subparts = re.split(r";\s+|\.\s+", part)
                result.extend(s.strip(" .") + "." for s in subparts if len(s.strip()) > 30)
            else:
                result.append(part)
        return result

    def _is_good_fact_sentence(self, sentence: str) -> bool:
        sentence = sentence.strip()
        low = sentence.lower()
        if re.match(r"^(эти|эта|это|этот|данный|данная|данные)\b", low):
            return False
        if len(sentence) < self.min_answer_chars or len(sentence) > 720:
            return False
        if self._DOMAIN_RE.search(sentence) or self._BAD_LINE_RE.search(low):
            return False
        if "?" in sentence:
            return False
        if not (self._CYR_RE.search(sentence) or self._LAT_RE.search(sentence)):
            return False
        if len(sentence.split()) < 5:
            return False
        verbs = r"\b(является|это|делится|состоит|имеет|относятся|относится|называется|называют|покрыт|покрыты|покрыто|защищает|помогает|помогают|может|могут|питается|жив[её]т|обитает|размножается|меняет|использует|служит|отличается|соединены|соединяется|колебаться|способен|способна|способны|увеличиваться)\b"
        number_words = r"\b(один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|нескольких|тридцати|сорок|сто)\b"
        has_number = bool(re.search(r"\b\d+(?:[,.]\d+)?\b", low) or re.search(number_words, low))
        return bool(re.search(verbs, low) or has_number)

    def _extract_terms(self, text: str) -> list[str]:
        candidates: list[str] = []
        for m in re.finditer(r"\b[А-Яа-яЁё][А-Яа-яЁё\-]{4,}\b", text):
            word = m.group(0).strip("-–—.,;:()[]{}\"'«»")
            low = word.lower()
            if low in self._bad_terms:
                continue
            if low.endswith(("ого", "его", "ими", "ыми")) and low not in {"хитинового", "эластичной"}:
                continue
            candidates.append(word)
        seen: set[str] = set()
        terms: list[str] = []
        for term in candidates:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                terms.append(term)
        return terms[:6]

    # ------------------------------------------------------------------
    # Question / answer generation
    # ------------------------------------------------------------------
    def generate_card_candidates(self, blocks: Iterable[dict]) -> list[dict]:
        cards: list[dict] = []
        for block in blocks:
            text = (block.get("text") or "").strip()
            if not text:
                continue
            front = self._make_question(text, block)
            back = self._make_answer(text)
            if not front or not back or self._is_bad_question(front):
                continue
            cards.append(self._build_card(front, back, source=text, metadata=block))
            if len(cards) >= self.max_cards:
                break
        return cards

    def _make_question(self, sentence: str, metadata: dict | None = None) -> str:
        s = re.sub(r"\s+", " ", sentence.strip())
        low = s.lower()
        topic = self._guess_topic(s, metadata)

        # Сначала конкретные шаблоны. Они не дают общим regex-правилам
        # делать вопросы вроде «Из чего состоит У пауков восемь ног...».
        if re.search(r"\bу\s+пауков\s+(?:есть\s+)?восемь\s+ног\b|\bвосемь\s+ног\b", low) and re.search(r"паук", low):
            return "Сколько ног у пауков?"

        if "головогруд" in low and ("конечност" in low or "ног" in low) and re.search(r"прикреп|отход|располож", low):
            return "Какие конечности прикреплены к головогруди паука?"

        if re.search(r"тело\s+паук|тело\s+у\s+пауков|у\s+пауков\s+тело", low) and re.search(r"делит(?:ся|ься)|разделен|разделено|разделяется", low):
            return "На какие отделы делится тело паука?"

        if re.search(r"головогруд.*брюшк.*стебел|брюшк.*головогруд.*стебел", low):
            return "Чем соединены головогрудь и брюшко паука?"

        if ("хитинов" in low or "хитин" in low) and ("панцир" in low or "экзоскелет" in low) and re.search(r"называ", low):
            return "Как называется внешний хитиновый панцирь паука?"

        if "покрыт" in low and "хитинов" in low:
            if "отдел" in low and "паук" in low:
                return "Чем покрыты отделы тела паука?"
            if "брюшк" in low:
                return "Чем покрыто брюшко паука?"
            if "тело" in low and "паук" in low:
                return "Чем покрыто тело паука?"

        if "не способен менять" in low and "панцир" in low:
            return "Почему хитиновый панцирь паука не может менять размер?"
        if re.search(r"периодически\s+паук\s+меняет|меняет\s+его\s+на\s+нов", low):
            return "Зачем паук периодически меняет хитиновый панцирь?"
        if "благодаря чему" in low and "увелич" in low and "брюшк" in low:
            return "Почему брюшко паука может увеличиваться после трапезы?"
        if "защища" in low and ("хитинов" in low or "панцир" in low):
            return "От чего защищает хитиновый слой тела паука?"

        if re.search(r"\bотнос(?:ится|ятся)\s+к\b|\bкласс\b", low) and "паук" in low:
            return "К какому классу относятся пауки?"

        if re.search(r"размер|миллиметр|сантиметр|\bмм\b|\bсм\b|колеб", low) and self._has_number(low):
            if "паук" in low or "членистоног" in low:
                return "В каких пределах могут колебаться размеры пауков?"
            return self._limit_question(f"В каких пределах изменяются размеры {topic}?")

        # Определения. Плохие подлежащие вроде «У паукообразных тело...»
        # отсекаются, чтобы не появлялось «Что такое У ...?».
        m = re.match(r"^(.{3,80}?)(?:\s+—\s+|\s+-\s+|\s+это\s+)(.{12,})", s, flags=re.I)
        if m:
            subject = self._clean_subject(m.group(1))
            if subject and not self._is_bad_subject(subject):
                return self._limit_question(f"Что такое {self._lower_first(subject)}?")

        m = re.search(r"(.{3,100}?)\s+делит(?:ся|ься)\s+на\s+(.{5,160})", s, flags=re.I)
        if m:
            subject = self._clean_phrase_for_question(m.group(1))
            if not self._is_bad_subject(subject):
                return self._limit_question(f"На какие части делится {self._lower_first(subject)}?")

        m = re.search(r"(.{3,100}?)\s+раздел[её]н(?:о|а|ы)?\s+на\s+(.{5,160})", s, flags=re.I)
        if m:
            subject = self._clean_phrase_for_question(m.group(1))
            if not self._is_bad_subject(subject):
                return self._limit_question(f"На какие части разделено {self._lower_first(subject)}?")

        m = re.search(r"(.{3,100}?)\s+состо(?:ит|ят)\s+из\s+(.{5,160})", s, flags=re.I)
        if m:
            subject = self._clean_phrase_for_question(m.group(1))
            if self._is_bad_subject(subject):
                return ""
            subject = self._normalize_question_subject(subject, topic, low)
            if self._is_bad_subject(subject):
                return ""
            verb = "состоят" if self._looks_plural(subject) else "состоит"
            return self._limit_question(f"Из чего {verb} {subject}?")

        m = re.search(r"(.{3,120}?)\s+соедин[её]н(?:ы|ные|о|а)?\s+(.{5,120})", s, flags=re.I)
        if m:
            subject = self._clean_phrase_for_question(m.group(1))
            if not self._is_bad_subject(subject):
                return self._limit_question(f"Чем соединены {self._lower_first(subject)}?")

        if re.search(r"\bназыва(?:ют|ется|ются)\b", low):
            if "паутин" in low and ("сеть" in low or "ловч" in low):
                return "Как называется ловчая сеть паука?"
            if "просом" in low:
                return "Что такое просома у паукообразных?"
            return ""

        if re.search(r"\bпомога(?:ет|ют)\b|\bиспользу(?:ет|ют|ется)\b|\bслуж(?:ит|ат)\b", low):
            if "педипальп" in low:
                return "Для чего пауку нужны педипальпы?"
            if "паутин" in low:
                return "Для чего паук использует паутину?"
            return ""

        if re.search(r"\bпита(?:ется|ются)\b|\bпища\b|\bдобыч\b", low):
            return self._limit_question(f"Чем питается {topic}?")
        if re.search(r"\bразмнож\w+\b|\bяйц\w+\b", low):
            return self._limit_question(f"Как размножается {topic}?")
        if re.search(r"\bяд\b|\bопас\w+\b|\bукус\w+\b", low):
            return self._limit_question(f"Чем опасен укус {topic}?")

        return ""

    def _has_number(self, low: str) -> bool:
        return bool(re.search(r"\b\d+(?:[,.]\d+)?\b", low) or re.search(r"\b(один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|двенадцать|нескольких|тридцати|сорок|сто)\b", low))

    def _make_answer(self, sentence: str) -> str:
        answer = re.sub(r"\s+", " ", sentence.strip())
        if not answer.endswith((".", "!", "?", "…")):
            answer += "."
        if len(answer) > self.max_answer_chars:
            answer = answer[: self.max_answer_chars].rsplit(" ", 1)[0].strip() + "…"
        return answer

    def _guess_topic(self, sentence: str, metadata: dict | None = None) -> str:
        low = sentence.lower()
        if re.search(r"\bпаук|пауков|пауки|паукообразн|членистоног", low):
            return "паука"
        if re.search(r"\bслон|слона|слоны", low):
            return "слона"
        terms = (metadata or {}).get("terms") or self._extract_terms(sentence)
        for term in terms:
            if term.lower() not in self._bad_terms:
                return self._normalize_topic_case(term)
        return self._active_topic or "темы"

    def _detect_global_topic(self, text: str) -> str:
        low = text.lower()
        if re.search(r"\bпаук|пауков|пауки|паукообразн|членистоног", low):
            return "паука"
        if re.search(r"\bслон|слона|слоны", low):
            return "слона"
        if "фотосинтез" in low:
            return "фотосинтеза"
        terms = self._extract_terms(text[:1200])
        return self._normalize_topic_case(terms[0]) if terms else "темы"

    def _clean_phrase_for_question(self, phrase: str) -> str:
        phrase = re.sub(r"\s+", " ", phrase or "").strip(" .,:;—–-")
        phrase = re.sub(r"^(эти|эта|это|этот|данные|данная|данный)\s+", "", phrase, flags=re.I)
        phrase = re.sub(r"^(каждая|каждый|каждое)\s+", "", phrase, flags=re.I)
        phrase = re.sub(r"\b(хорошо|зрительно|между собой|обычно|традиционно)\b", "", phrase, flags=re.I)
        phrase = re.sub(r"\s+", " ", phrase).strip(" .,:;—–-")[:90]
        return phrase

    def _is_bad_subject(self, subject: str) -> bool:
        subject = re.sub(r"\s+", " ", subject or "").strip(" .,:;—–-")
        low = subject.lower()
        if not low or len(low) < 3:
            return True
        if len(subject) > 80:
            return True
        if low.startswith(("у ", "в ", "на ", "для ", "при ", "по ", "из ", "к ", "с ")):
            return True
        if low.split()[0] in {"традиционно", "обычно", "каждая", "каждый", "каждое", "эти", "это", "эта", "этот"}:
            return True
        if low in self._bad_terms:
            return True
        if re.search(r"\b(описанн\w*|данн\w*|эт\w*|особенность|признак|явление|структура)\b", low):
            return True
        return False

    def _normalize_question_subject(self, subject: str, topic: str, sentence_low: str) -> str:
        subject = self._clean_phrase_for_question(subject)
        subject = self._lower_first(subject)
        low = subject.lower()
        if "ходильная нога" in low and "паук" in sentence_low and "паук" not in low:
            return "ходильная нога паука"
        if low == "скелет" and "паук" in sentence_low:
            return "скелет паука"
        if low in {"педипальпы", "педипальп"}:
            return "педипальпы паука"
        return subject

    def _looks_plural(self, subject: str) -> bool:
        low = (subject or "").lower().strip()
        if low.endswith(("ы", "и")) and not low.endswith(("ости", "асти")):
            return True
        if re.search(r"\b(педипальпы|ноги|конечности|отделы|части)\b", low):
            return True
        return False

    def _lower_first(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text
        if len(text) > 1 and text[:2].isupper():
            return text
        return text[0].lower() + text[1:]

    def _is_bad_question(self, question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").lower()).strip()
        bad_fragments = [
            "что важно помнить", "какой ключевой факт", "понятии", "понятие", "размеры?",
            "выделить", "особенности", "этот признак", "явление или структура",
            "группе или классу относится темы", "для чего нужна эта особенность",
            "почему это важно", "описанные части", "описанная структура", "что такое у ",
            "что такое традиционно", "из чего состоит у ", "из чего состоит каждая",
            "сколько конечностей у головогруди", "чем отличаются описанные",
        ]
        if any(f in q for f in bad_fragments):
            return True
        if re.search(r"\b(размеры|особенности|выделить|традиционно|каждая)\?", q):
            return True
        if len(q) < 8 or len(q) > 150:
            return True
        return False

    def _normalize_topic_case(self, topic: str) -> str:
        topic = self._clean_subject(topic)
        return topic[:60] if topic else "темы"

    def _clean_subject(self, text: str) -> str:
        text = re.sub(r"^[,;:.\s]+|[,;:.\s]+$", "", text or "")
        text = re.sub(r"\b(это|такой|такая|такие|является)\b", "", text, flags=re.I).strip()
        return re.sub(r"\s+", " ", text)[:80]

    def _limit_question(self, question: str) -> str:
        question = re.sub(r"\s+", " ", question).strip()
        if not question.endswith("?"):
            question += "?"
        if len(question) <= 140:
            return question
        return question[:137].rsplit(" ", 1)[0].rstrip(" ?,.;:") + "?"

    # ------------------------------------------------------------------
    # Card objects / filtering / images
    # ------------------------------------------------------------------
    def _build_card(self, front: str, back: str, source: str = "", metadata: dict | None = None) -> dict:
        metadata = dict(metadata or {})
        card = {
            "front": front.strip(),
            "back": back.strip(),
            "explanation": self._short_explanation(back),
            "image_prompt": "",
            "negative_prompt": "text, watermark, logo, blurry, low quality, extra letters",
            "image_path": None,
            "source": source.strip(),
            "tags": [],
            "difficulty": "normal",
            "card_type": "qa",
            "quality_score": 0.8,
            "metadata": {
                "terms": metadata.get("terms", []),
                "dates": metadata.get("dates", []),
                "formulas": metadata.get("formulas", []),
                "facts": [source.strip()] if source else [],
                "causes": [],
                "differences": [],
            },
        }
        card["image_prompt"] = self.generate_image_prompt(card)
        return card

    def _topic_card(self, topic: str) -> dict:
        topic = self._clean_topic_command(topic)
        front = self._limit_question(f"Что такое {topic}?")
        back = f"{topic.capitalize()} — тема для изучения. Найдите или вставьте материал по теме, чтобы AnkyX создал точные карточки по фактам, определениям и примерам."
        return self._build_card(front, back, source=topic, metadata={"terms": [topic]})

    def _clean_topic_command(self, text: str) -> str:
        text = re.sub(r"(?i)^\s*(сгенерируй|создай|сделай)\s+карточк[ауи]?", "", text).strip()
        text = re.sub(r"(?i)\bс\s+(картинкой|изображением)\b", "", text).strip()
        text = re.sub(r"(?i)^про\s+", "", text).strip()
        return re.sub(r"\s+", " ", text).strip(" :,-") or "тема"

    def _short_explanation(self, answer: str) -> str:
        sentences = self._split_sentences(answer)
        return (sentences[0] if sentences else answer)[:260]

    def _looks_like_short_topic(self, text: str) -> bool:
        if "\n" in text:
            return False
        words = text.split()
        return len(words) <= 6 and len(text) <= 80 or bool(re.match(r"(?i)^\s*(сгенерируй|создай|сделай)\s+карточк", text))

    def filter_and_improve_cards(self, cards: Iterable[dict]) -> list[dict]:
        result: list[dict] = []
        seen_front: set[str] = set()
        seen_pair: set[str] = set()
        for card in cards:
            front = re.sub(r"\s+", " ", (card.get("front") or "").strip())
            back = re.sub(r"\s+", " ", (card.get("back") or "").strip())
            if not front or not back or self._is_bad_question(front):
                continue
            if len(front) > 150:
                front = self._limit_question(front)
            if len(back) < 24 or len(back) > 720:
                continue
            if self._is_noise_line(back):
                continue
            front_key = self._semantic_key(front)
            pair_key = self._semantic_key(front + " " + back)
            if front_key in seen_front or pair_key in seen_pair:
                continue
            seen_front.add(front_key)
            seen_pair.add(pair_key)
            card = dict(card)
            card["front"] = front
            card["back"] = back
            card.setdefault("explanation", self._short_explanation(back))
            card.setdefault("negative_prompt", "text, watermark, logo, blurry, low quality, extra letters")
            if not card.get("image_prompt"):
                card["image_prompt"] = self.generate_image_prompt(card)
            card.setdefault("metadata", {})
            result.append(card)
            if len(result) >= self.max_cards:
                break
        return result

    def _semantic_key(self, text: str) -> str:
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", text.lower())
        stop = {"котор", "также", "может", "нужно", "важно", "этого", "этот", "этой", "является", "например"}
        words = [w for w in words if w not in stop]
        return " ".join(sorted(set(words))[:12])

    def generate_image_prompt(self, card: dict) -> str:
        front = (card.get("front") or "").strip()
        back = (card.get("back") or "").strip()
        terms = (card.get("metadata") or {}).get("terms") or []
        topic = ", ".join(str(t) for t in terms[:3]) or self._guess_topic(back or front)
        prompt = f"educational realistic visual illustration about {topic}; based on: {front} {back[:180]}; clean composition, no text, no watermark, high detail"
        return re.sub(r"\s+", " ", prompt).strip()[:900]

    def generate_card_image(self, card: dict) -> dict:
        card = dict(card)
        metadata = dict(card.get("metadata") or {})
        if StableDiffusionAdapter is None:
            metadata["image_status"] = "Stable Diffusion adapter недоступен"
            card["metadata"] = metadata
            return card
        try:
            try:
                adapter = StableDiffusionAdapter(app=self.app)
            except TypeError:
                adapter = StableDiffusionAdapter()
            image_path = adapter.generate_image(
                card.get("image_prompt") or self.generate_image_prompt(card),
                card.get("negative_prompt") or "text, watermark, logo, blurry, low quality, extra letters",
            )
            if isinstance(image_path, dict):
                path = image_path.get("image_path") or image_path.get("path")
                status = image_path.get("status") or image_path.get("message")
            else:
                path = image_path
                status = None
            if path:
                card["image_path"] = path
                metadata["image_status"] = status or "Изображение создано"
            else:
                metadata["image_status"] = status or "Stable Diffusion недоступен или не вернул файл"
        except Exception as exc:
            logging.exception("generate_card_image failed")
            metadata["image_status"] = f"Ошибка Stable Diffusion: {exc}"
        card["metadata"] = metadata
        return card

    # ------------------------------------------------------------------
    # DB saving
    # ------------------------------------------------------------------
    def save_cards_to_overview(self, cards: Iterable[dict]) -> int:
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
                    "state": "overview",
                    "due": now_ts,
                    "interval": 0,
                    "ease": 2500,
                    "reps": 0,
                    "lapses": 0,
                    "step_index": 0,
                    "last_review": None,
                }
                insert_cols = [c for c in values if c in columns]
                if not insert_cols:
                    raise RuntimeError("Таблица cards не содержит ожидаемых колонок")
                placeholders = ", ".join("?" for _ in insert_cols)
                sql = f"INSERT INTO cards ({', '.join(insert_cols)}) VALUES ({placeholders})"
                conn.execute(sql, [values[c] for c in insert_cols])
                saved += 1
            conn.commit()
            return saved
        finally:
            conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        try:
            from db_connect import open_db  # type: ignore
            return open_db()
        except Exception:
            pass
        try:
            from db_path import get_db_path  # type: ignore
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
