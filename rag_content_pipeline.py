from __future__ import annotations

import html
import re
from urllib.parse import urlparse

from rag_web_search import RagWebSearch
from youtube_transcript_extractor import YouTubeTranscriptExtractor


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_lines(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        key = re.sub(r"\W+", "", raw.lower())[:180]
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return "\n".join(out)


def remove_web_noise(text: str) -> str:
    bad = re.compile(
        r"(?i)\b(cookie|privacy|subscribe|advertisement|share|related articles|comments|read also|navigation|подпис|реклама|комментар|куки|меню)\b"
    )
    out: list[str] = []
    for line in (text or "").splitlines():
        row = line.strip(" \t-–—|•")
        if not row:
            continue
        if bad.search(row):
            continue
        if row.count("|") > 2 or row.count("›") > 2:
            continue
        if len(row) < 20 and not re.search(r"\d", row):
            continue
        out.append(row)
    return "\n".join(out)


def light_grammar_cleanup(text: str) -> str:
    text = re.sub(r"\s+([,.!?;:])", r"\1", text or "")
    text = re.sub(r"([.!?])([А-ЯA-Z])", r"\1 \2", text)
    return text.strip()


def clean_raw_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = normalize_whitespace(text)
    text = remove_web_noise(text)
    text = dedupe_lines(text)
    return light_grammar_cleanup(normalize_whitespace(text))


class RagContentPipeline:
    def __init__(self, max_chars: int = 30000, max_sources: int = 5) -> None:
        self.max_chars = max_chars
        self.max_sources = max_sources
        self.search = RagWebSearch(max_pages=max_sources, max_chars=max_chars)

    def fetch_materials(self, query_or_url: str, max_sources: int = 5) -> dict:
        query_or_url = (query_or_url or "").strip()
        result = {
            "query": query_or_url,
            "source_type": "search",
            "clean_text": "",
            "sources": [],
            "status": "error",
            "errors": [],
        }
        if not query_or_url:
            result["errors"].append("Пустой запрос")
            return result
        if YouTubeTranscriptExtractor.is_youtube_url(query_or_url):
            yt = YouTubeTranscriptExtractor.fetch_transcript(query_or_url)
            result["source_type"] = "youtube"
            if yt.get("status") != "ok":
                result["status"] = yt.get("status") or "error"
                if yt.get("error"):
                    result["errors"].append(yt["error"])
                return result
            clean = clean_raw_text(yt.get("text") or "")
            result["clean_text"] = clean[: self.max_chars]
            result["sources"] = [
                {
                    "url": yt.get("url", ""),
                    "title": yt.get("title", ""),
                    "source_type": "youtube",
                    "text": yt.get("text", ""),
                    "clean_text": clean,
                    "metadata": {"language": yt.get("language"), "segments": yt.get("segments", [])},
                }
            ]
            result["status"] = "ok"
            return result
        if self._is_url(query_or_url):
            result["source_type"] = "web"
            try:
                page = self.search.fetch_page_text(query_or_url)
                clean = clean_raw_text(page.get("text", ""))
                result["clean_text"] = clean[: self.max_chars]
                result["sources"] = [
                    {
                        "url": query_or_url,
                        "title": page.get("title", ""),
                        "source_type": "web",
                        "text": page.get("text", ""),
                        "clean_text": clean,
                        "metadata": {},
                    }
                ]
                result["status"] = "ok"
            except Exception as exc:
                result["errors"].append(str(exc))
            return result

        result["source_type"] = "search"
        try:
            urls = self.search._duckduckgo_urls(query_or_url)[: max(1, max_sources)]
            combined: list[str] = []
            for url in urls:
                try:
                    page = self.search.fetch_page_text(url)
                    clean = clean_raw_text(page.get("text", ""))
                    if len(clean) < 300:
                        continue
                    result["sources"].append(
                        {
                            "url": url,
                            "title": page.get("title", ""),
                            "source_type": "web",
                            "text": page.get("text", ""),
                            "clean_text": clean,
                            "metadata": {},
                        }
                    )
                    combined.append(f"# {page.get('title', url)}\n{clean}")
                except Exception as exc:
                    result["errors"].append(f"{url}: {exc}")
            result["clean_text"] = clean_raw_text("\n\n".join(combined))[: self.max_chars]
            result["status"] = "ok" if result["clean_text"] else "error"
            if not result["clean_text"] and not result["errors"]:
                result["errors"].append("Не удалось извлечь текст из найденных страниц")
        except Exception as exc:
            result["errors"].append(str(exc))
        return result

    def _is_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
