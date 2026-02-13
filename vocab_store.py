from __future__ import annotations

import os
import re

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёÄÖÜäöüß]+(?:['’-][A-Za-zА-Яа-яЁёÄÖÜäöüß]+)?")


def _known_words_path(lang_code: str) -> str:
    base_dir = r"C:\X-FLASH\data"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"known_words_{lang_code}.txt")


def load_known_words(lang_code: str) -> set[str]:
    path = _known_words_path(lang_code)
    if not os.path.exists(path):
        return set()
    result: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            token = raw.strip().lower()
            if token:
                result.add(token)
    return result


def mask_unknown_words(text: str, lang: str, known_set: set[str]) -> str:
    if not text:
        return ""
    normalized_known = {w.lower() for w in (known_set or set())}

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.lower() in normalized_known else "—"

    return WORD_RE.sub(repl, text)
