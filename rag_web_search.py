from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


class _Strip(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.buf: list[str] = []

    def handle_data(self, data: str) -> None:
        s = (data or "").strip()
        if s:
            self.buf.append(s)


def _fetch(url: str, timeout: int = 10) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 AnkyX/1.0"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="ignore")


def search_text(query: str, max_pages: int = 1) -> str:
    q = (query or "").strip()
    if not q:
        raise RuntimeError("Пустой поисковый запрос")

    # Минимальный безопасный вариант без внешних SDK.
    url = f"https://duckduckgo.com/html/?q={quote_plus(q)}"
    html = _fetch(url)
    parser = _Strip()
    parser.feed(html)

    cleaned = []
    seen = set()
    for line in parser.buf:
        s = re.sub(r"\s+", " ", line).strip()
        if len(s) < 20:
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
        if len(cleaned) >= 120:
            break

    if not cleaned:
        raise RuntimeError("Не удалось получить текст из веб-поиска")
    return "\n".join(cleaned)
