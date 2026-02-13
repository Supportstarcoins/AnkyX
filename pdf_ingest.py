from __future__ import annotations

import re
from typing import Iterable

import fitz


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").replace("\x00", " ")).strip()
    return cleaned


def extract_text_from_pdf(path: str) -> str:
    chunks: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            chunks.append(page.get_text("text") or "")
    return normalize_text("\n".join(chunks))


def detect_lang(text: str) -> str:
    sample = (text or "")[:5000]
    if not sample:
        return "en"
    latin_words = len(re.findall(r"\b[A-Za-z][A-Za-z'\-]{1,}\b", sample))
    cyrillic_words = len(re.findall(r"\b[А-Яа-яЁё][А-Яа-яЁё'\-]{1,}\b", sample))
    german_markers = len(re.findall(r"[ÄÖÜäöüß]", sample)) + len(
        re.findall(r"\b(der|die|das|und|nicht|ist|ein|eine|mit|für)\b", sample, flags=re.IGNORECASE)
    )
    if cyrillic_words > latin_words:
        return "ru"
    if german_markers >= 4:
        return "de"
    return "en"


def split_to_sentences(text: str, lang: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    # Works reasonably for ru/en/de.
    sentence_candidates = re.split(r"(?<=[.!?…])\s+", normalized)
    sentences = [s.strip() for s in sentence_candidates if len(s.strip()) >= 8]
    return sentences


def _chunk_by_target(sentences: list[str], *, min_s: int, max_s: int, target: int) -> list[list[str]]:
    if not sentences:
        return []
    chunks: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        if len(sentence) < 8:
            continue
        current.append(sentence)
        if len(current) >= target:
            chunks.append(current)
            current = []
    if current:
        if chunks and len(current) < min_s:
            chunks[-1].extend(current)
        else:
            chunks.append(current)

    balanced: list[list[str]] = []
    for chunk in chunks:
        if len(chunk) <= max_s:
            balanced.append(chunk)
            continue
        start = 0
        while start < len(chunk):
            balanced.append(chunk[start : start + max_s])
            start += max_s
    return [c for c in balanced if len(c) >= min_s or len(c) == len(sentences)]


def chunk_sentences(sentences: list[str], mode_native: bool) -> list[list[str]]:
    if mode_native:
        return _chunk_by_target(sentences, min_s=5, max_s=20, target=10)
    return _chunk_by_target(sentences, min_s=1, max_s=5, target=2)
