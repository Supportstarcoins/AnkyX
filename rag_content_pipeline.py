from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

from rag_web_search import RagWebSearch
from youtube_transcript_extractor import YouTubeTranscriptExtractor


class _ImageHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: list[dict] = []
        self._text_window: list[str] = []
        self._position = 0

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        text = (data or "").strip()
        if not text:
            return
        self._position += 1
        self._text_window.append(text)
        if len(self._text_window) > 4:
            self._text_window = self._text_window[-4:]

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if (tag or "").lower() != "img":
            return
        meta = {str(k).lower(): str(v) for k, v in (attrs or []) if k}
        src = (meta.get("src") or "").strip()
        if not src:
            return
        self.images.append(
            {
                "url": src,
                "local_path": "",
                "alt": (meta.get("alt") or "").strip(),
                "caption": (meta.get("title") or "").strip(),
                "context_text": " ".join(self._text_window[-2:]).strip(),
                "position": self._position,
                "width": _to_int(meta.get("width")),
                "height": _to_int(meta.get("height")),
                "source_type": "web",
            }
        )


def _to_int(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"\d+", str(value))
    return int(m.group(0)) if m else None


def _keyword_tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[а-яa-z0-9]{4,}", (value or "").lower())]


def _score_image_relevance(card: dict, image: dict) -> float:
    card_text = " ".join(
        [
            str(card.get("back") or ""),
            str(card.get("source_excerpt") or ""),
            " ".join(_keyword_tokens(str(card.get("front") or ""))),
        ]
    ).lower()
    img_text = " ".join(
        [
            str(image.get("alt") or ""),
            str(image.get("caption") or ""),
            str(image.get("context_text") or ""),
        ]
    ).lower()
    card_tokens = set(_keyword_tokens(card_text))
    image_tokens = set(_keyword_tokens(img_text))
    if not card_tokens or not image_tokens:
        return 0.0
    overlap = len(card_tokens & image_tokens) / max(1, len(card_tokens))
    length_bonus = 0.05 if len(str(image.get("context_text") or "")) > 40 else 0.0
    return round(min(1.0, overlap + length_bonus), 3)


def attach_best_source_images(cards: list[dict], images: list[dict]) -> list[dict]:
    pool = list(images or [])
    out: list[dict] = []
    for card in cards or []:
        c = dict(card or {})
        best = None
        best_score = 0.0
        for img in pool:
            score = _score_image_relevance(c, img)
            if score > best_score:
                best = img
                best_score = score
        c.setdefault("answer_image_path", "")
        c.setdefault("answer_image_url", "")
        c.setdefault("answer_image_caption", "")
        c.setdefault("image_source_type", "")
        c.setdefault("image_relevance_score", 0.0)
        if best and best_score >= 0.12:
            c["answer_image_path"] = str(best.get("local_path") or "")
            c["answer_image_url"] = str(best.get("url") or "")
            c["answer_image_caption"] = str(best.get("caption") or best.get("alt") or "").strip()
            c["image_source_type"] = "extracted"
            c["image_relevance_score"] = float(best_score)
            if c["answer_image_path"]:
                c["image_path"] = c["answer_image_path"]
        else:
            c["image_source_type"] = "recommended" if c.get("needs_image") else "none"
            c["image_relevance_score"] = 0.0
        out.append(c)
    return out


def extract_images_from_web_html(url: str) -> list[dict]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=12) as response:
        raw = response.read(2_000_000)
        ctype = response.headers.get("Content-Type", "")
    charset_match = re.search(r"charset=([\w-]+)", ctype, flags=re.I)
    charset = charset_match.group(1) if charset_match else "utf-8"
    html_text = raw.decode(charset, errors="replace")
    parser = _ImageHTMLParser()
    parser.feed(html_text)
    out = []
    for img in parser.images:
        full_url = urllib.parse.urljoin(url, str(img.get("url") or ""))
        row = dict(img)
        row["url"] = full_url
        out.append(row)
    return out


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
                try:
                    web_images = extract_images_from_web_html(query_or_url)
                except Exception:
                    web_images = []
                result["clean_text"] = clean[: self.max_chars]
                result["sources"] = [
                    {
                        "url": query_or_url,
                        "title": page.get("title", ""),
                        "source_type": "web",
                        "text": page.get("text", ""),
                        "clean_text": clean,
                        "metadata": {"images": web_images},
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
                    try:
                        web_images = extract_images_from_web_html(url)
                    except Exception:
                        web_images = []
                    if len(clean) < 300:
                        continue
                    result["sources"].append(
                        {
                            "url": url,
                            "title": page.get("title", ""),
                            "source_type": "web",
                            "text": page.get("text", ""),
                            "clean_text": clean,
                            "metadata": {"images": web_images},
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
