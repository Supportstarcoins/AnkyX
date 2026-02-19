from __future__ import annotations

import re


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёÄÖÜäöüß]+(?:['’-][A-Za-zА-Яа-яЁёÄÖÜäöüß]+)?")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split((text or "").strip()) if p.strip()]
    return parts


def chunk_by_sentence_count(text: str, native_mode: bool = True) -> list[str]:
    sentences = split_sentences(text)
    min_s, max_s = (5, 20) if native_mode else (1, 5)
    target = 10 if native_mode else 3
    chunks: list[str] = []
    buf: list[str] = []
    for sent in sentences:
        buf.append(sent)
        if len(buf) >= target and re.search(r"[.!?]\s*$", sent):
            chunks.append(" ".join(buf))
            buf = []
        elif len(buf) >= max_s:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        if chunks and len(buf) < min_s:
            chunks[-1] = chunks[-1] + " " + " ".join(buf)
        else:
            chunks.append(" ".join(buf))
    return chunks


def apply_cloze(text: str, known_words: set[str], max_new_words: int = 3) -> tuple[str, list[str]]:
    known = {w.lower() for w in known_words}
    selected: list[str] = []

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        low = token.lower()
        if low in known or low in {w.lower() for w in selected}:
            return token
        if len(selected) < max_new_words:
            selected.append(token)
            return "____"
        return token

    return _WORD_RE.sub(repl, text), selected
