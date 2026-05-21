from __future__ import annotations

import html
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

from rag_web_search import RagWebSearch
from source_extractors import download_image_to_media, extract_text_from_url
from youtube_transcript_extractor import YouTubeTranscriptExtractor


class _ImageHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: list[dict] = []
        self._text_window: list[str] = []
        self._position = 0
        self._in_figure = False
        self._figure_caption = ""
        self.og_images: list[str] = []
        self.twitter_images: list[str] = []

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        text = (data or "").strip()
        if not text:
            return
        self._position += 1
        self._text_window.append(text)
        if len(self._text_window) > 4:
            self._text_window = self._text_window[-4:]

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag_low = (tag or "").lower()
        meta = {str(k).lower(): str(v) for k, v in (attrs or []) if k}
        if tag_low == "figure":
            self._in_figure = True
            self._figure_caption = ""
            return
        if tag_low == "figcaption":
            self._figure_caption = " ".join(self._text_window[-2:]).strip()
            return
        if tag_low == "meta":
            prop = (meta.get("property") or meta.get("name") or "").strip().lower()
            content = (meta.get("content") or "").strip()
            if prop == "og:image" and content:
                self.og_images.append(content)
            elif prop == "twitter:image" and content:
                self.twitter_images.append(content)
            return
        if tag_low != "img":
            return
        src = (meta.get("src") or "").strip()
        if not src:
            return
        self.images.append(
            {
                "url": src,
                "local_path": "",
                "alt": (meta.get("alt") or "").strip(),
                "caption": ((meta.get("title") or "").strip() or self._figure_caption),
                "context_text": " ".join(self._text_window[-2:]).strip(),
                "position": self._position,
                "width": _to_int(meta.get("width")),
                "height": _to_int(meta.get("height")),
                "source_type": "html_img",
            }
        )

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if (tag or "").lower() == "figure":
            self._in_figure = False
            self._figure_caption = ""


def _to_int(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"\d+", str(value))
    return int(m.group(0)) if m else None


def _keyword_tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[а-яa-z0-9]{4,}", (value or "").lower())]

_GENERIC_TOKENS = {
    "anatomy", "анатомия", "biology", "биология", "organism", "организм", "system", "система", "structure", "строение",
}
_HUMAN_ANATOMY_TOKENS = {"human", "человек", "skeleton", "skull", "muscle", "bone", "кости", "костная", "kenhub"}
_SPIDER_TOKENS = {"spider", "паук", "пауки", "arachnid", "паукообразные", "головогрудь", "брюшко", "хитиновый", "панцирь"}

def extract_card_entities(card: dict) -> set[str]:
    text = " ".join([str(card.get("front") or ""), str(card.get("back") or ""), str(card.get("topic") or ""), str(card.get("source_excerpt") or "")]).lower()
    tokens = set(_keyword_tokens(text))
    entities = {t for t in tokens if t not in _GENERIC_TOKENS}
    if "паука" in text or "пауков" in text:
        entities.add("паук")
    if "тело паука" in text:
        entities.add("тело паука")
    return entities

def image_matches_card_topic(card: dict, image: dict) -> tuple[bool, str, float]:
    entities = extract_card_entities(card)
    img_text_parts = [str(image.get("alt") or ""), str(image.get("caption") or ""), str(image.get("context_text") or ""), str(image.get("source_title") or "")]
    img_text = " ".join(img_text_parts).strip().lower()
    if not img_text:
        return False, "no_context", 0.0
    img_tokens = set(_keyword_tokens(img_text))
    matched = entities & img_tokens
    generic_matches = img_tokens & _GENERIC_TOKENS
    card_tokens = set(_keyword_tokens(" ".join([str(card.get("front") or ""), str(card.get("back") or ""), str(card.get("topic") or ""), str(card.get("source_excerpt") or "")]).lower()))
    card_has_human = bool(card_tokens & _HUMAN_ANATOMY_TOKENS)
    if (img_tokens & _HUMAN_ANATOMY_TOKENS) and not card_has_human and (card_tokens & _SPIDER_TOKENS):
        return False, "topic_mismatch", 0.0
    if not matched and generic_matches:
        return False, "too_generic", 0.0
    if not matched:
        return False, "topic_mismatch", 0.0
    score = len(matched) / max(1, len(entities))
    return True, "ok", round(min(1.0, score), 3)


def _score_image_relevance(card: dict, image: dict) -> float:
    ok, _, topic_score = image_matches_card_topic(card, image)
    if not ok:
        return 0.0
    card_text = " ".join(
        [
            str(card.get("topic") or ""),
            str(card.get("front") or ""),
            str(card.get("back") or ""),
            str(card.get("source_excerpt") or ""),
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
    source_type = str(image.get("source_type") or "").lower()
    source_bonus = 0.02 if source_type == "html_img" else 0.0
    if source_type == "youtube_thumbnail":
        source_bonus -= 0.05
    position_penalty = 0.0
    try:
        card_pos = int(card.get("source_position") or card.get("chunk_position") or 0)
        img_pos = int(image.get("position") or 0)
        if card_pos > 0 and img_pos > 0:
            distance = abs(card_pos - img_pos)
            position_penalty = min(0.1, distance / 200.0)
    except Exception:
        position_penalty = 0.0
    same_source_bonus = 0.0
    if card.get("source_url") and image.get("source_url") and str(card.get("source_url")) == str(image.get("source_url")):
        same_source_bonus += 0.2
    elif card.get("source_url") and image.get("source_url"):
        same_source_bonus -= 0.1
    if card.get("chunk_id") and image.get("nearest_chunk_id") and str(card.get("chunk_id")) == str(image.get("nearest_chunk_id")):
        same_source_bonus += 0.15
    return round(max(0.0, min(1.0, overlap + length_bonus + source_bonus + same_source_bonus + (0.35 * topic_score) - position_penalty)), 3)


def _is_garbage_image(image: dict) -> bool:
    url = str(image.get("url") or "").strip().lower()
    if url.startswith(("data:image/svg", "data:image/", "blob:", "javascript:", "about:")):
        return True
    hay = " ".join(
        [
            str(image.get("url") or ""),
            str(image.get("alt") or ""),
            str(image.get("caption") or ""),
            str(image.get("context_text") or ""),
        ]
    ).lower()
    bad = ("logo", "icon", "avatar", "banner", "sprite", "tracking", "pixel", "social", "share", "favicon", "ad")
    if any(b in hay for b in bad):
        return True
    w = int(image.get("width") or 0)
    h = int(image.get("height") or 0)
    return (w and w < 120) or (h and h < 120)


def normalize_card_image_fields(card: dict) -> dict:
    c = dict(card or {})
    def _blocked_url(value: str) -> bool:
        return str(value or "").strip().lower().startswith(("data:image/svg", "data:image/", "blob:", "javascript:", "about:"))

    front_path = str(c.get("front_image_path") or "").strip()
    if not front_path:
        front_path = str(c.get("image_path") or c.get("answer_image_path") or "").strip()
        if front_path:
            c["front_image_path"] = front_path
    c.setdefault("front_image_url", str(c.get("answer_image_url") or "").strip())
    if _blocked_url(c.get("front_image_url")):
        c["front_image_url"] = ""
    c.setdefault("front_image_caption", str(c.get("answer_image_caption") or "").strip())
    c.setdefault("front_image_origin", str(c.get("image_source_type") or "none").strip() or "none")
    c.setdefault("front_image_relevance_score", float(c.get("image_relevance_score") or 0.0))
    c.setdefault("back_image_path", "")
    c.setdefault("back_image_url", "")
    c.setdefault("back_image_caption", "")
    c.setdefault("back_image_origin", "none")
    c.setdefault("back_image_relevance_score", 0.0)
    if c.get("front_image_url") and not c.get("front_image_path"):
        c["front_image_origin"] = "source_url_not_downloaded"
    return c


def attach_best_source_image(card: dict, images: list[dict], media_dir: str) -> dict:
    c = normalize_card_image_fields(card)
    if c.get("front_image_path"):
        return c
    best = None
    best_score = 0.0
    for raw in images or []:
        if _is_garbage_image(raw):
            continue
        img = dict(raw or {})
        ok, reason, _ = image_matches_card_topic(c, img)
        if not ok:
            continue
        score = _score_image_relevance(c, img)
        if score > best_score:
            best_score = score
            best = img
    if not best or best_score < 0.45:
        _, reason, _ = image_matches_card_topic(c, best or {})
        c["front_image_origin"] = "none"
        c["front_image_path"] = ""
        c["front_image_url"] = ""
        c["front_image_caption"] = ""
        c["image_status"] = "no_relevant_source_image" if reason in {"ok", "no_context"} else f"source_rejected_{reason}"
        c["front_image_relevance_score"] = 0.0
        return c
    dl = download_image_to_media(str(best.get("url") or ""), media_dir=media_dir, source_url=str(best.get("source_url") or best.get("page_url") or ""))
    c["front_image_url"] = str(best.get("url") or "")
    c["front_image_caption"] = str(best.get("caption") or best.get("alt") or "").strip()
    c["front_image_relevance_score"] = float(best_score)
    if dl.get("ok") and dl.get("local_path"):
        c["front_image_path"] = dl["local_path"]
        c["image_path"] = c.get("image_path") or c["front_image_path"]
        c["front_image_origin"] = "source"
        c["image_status"] = f"source_attached: {c['front_image_caption']}" if c["front_image_caption"] else "source_attached"
    else:
        c["front_image_origin"] = "source_url_not_downloaded"
        c["image_status"] = "source_url_not_downloaded"
    return c


def attach_best_source_images(cards: list[dict], images: list[dict]) -> list[dict]:
    pool = [dict(img or {}) for img in (images or [])]
    media_dir = os.path.join(os.getcwd(), "media", "source_images")
    out: list[dict] = []
    for card in cards or []:
        c = attach_best_source_image(card, pool, media_dir=media_dir)
        c["answer_image_path"] = c.get("front_image_path") or c.get("answer_image_path") or ""
        c["answer_image_url"] = c.get("front_image_url") or c.get("answer_image_url") or ""
        c["answer_image_caption"] = c.get("front_image_caption") or c.get("answer_image_caption") or ""
        c["image_source_type"] = c.get("front_image_origin") or c.get("image_source_type") or "none"
        c["image_relevance_score"] = float(c.get("front_image_relevance_score") or c.get("image_relevance_score") or 0.0)
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
        if full_url.lower().startswith(("data:image/svg", "data:image/", "blob:", "javascript:", "about:")):
            continue
        row = dict(img)
        row["url"] = full_url
        row["page_url"] = url
        row["source_url"] = url
        row["source_title"] = ""
        row["source_index"] = 0
        row["nearest_chunk_id"] = None
        if _is_garbage_image(row):
            continue
        out.append(row)
    for img_url in parser.og_images:
        out.append(
            {
                "url": urllib.parse.urljoin(url, img_url),
                "local_path": "",
                "alt": "",
                "caption": "OpenGraph image",
                "context_text": "",
                "position": 0,
                "width": None,
                "height": None,
                "source_type": "og_image",
                "page_url": url,
            }
        )
    for img_url in parser.twitter_images:
        out.append(
            {
                "url": urllib.parse.urljoin(url, img_url),
                "local_path": "",
                "alt": "",
                "caption": "Twitter image",
                "context_text": "",
                "position": 0,
                "width": None,
                "height": None,
                "source_type": "twitter_image",
                "page_url": url,
            }
        )
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

    def _build_query_variations(self, topic: str) -> list[str]:
        base = re.sub(r"\s+", " ", (topic or "").strip())
        if not base:
            return []
        vars = [base, f"{base} строение", f"{base} анатомия", f"{base} определение факты"]
        words = [w for w in re.findall(r"[а-яa-z0-9]+", base.lower()) if len(w) >= 3]
        if words:
            head = words[-1]
            vars.extend([f"анатомия {head}", f"строение тела {head}", f"{head} строение тела"])
            if head == "тигра":
                vars.extend(["tiger anatomy", "tiger body structure"])
        seen = set()
        out = []
        for v in vars:
            k = v.lower().strip()
            if k and k not in seen:
                seen.add(k)
                out.append(v.strip())
        return out[:8]

    def fetch_materials(self, query_or_url: str, max_sources: int = 5, progress_callback=None) -> dict:
        query_or_url = (query_or_url or "").strip()
        result = {
            "query": query_or_url,
            "source_type": "search",
            "clean_text": "",
            "sources": [],
            "status": "error",
            "errors": [],
            "debug": {"query_variations": [], "urls": [], "url_results": []},
        }
        if not query_or_url:
            result["errors"].append("Введите URL, текст или тему")
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
                    "metadata": {
                        "language": yt.get("language"),
                        "segments": yt.get("segments", []),
                        "images": yt.get("images", []),
                    },
                }
            ]
            result["status"] = "ok"
            return result
        if self._is_url(query_or_url):
            result["source_type"] = "web"
            try:
                extracted = extract_text_from_url(query_or_url, min_chars=250)
                if not extracted.get("ok"):
                    raise RuntimeError(str(extracted.get("error") or "web extract failed"))
                clean = clean_raw_text(extracted.get("text", ""))
                try:
                    web_images = extract_images_from_web_html(query_or_url)
                except Exception:
                    web_images = []
                result["clean_text"] = clean[: self.max_chars]
                result["sources"] = [
                    {
                        "url": query_or_url,
                        "title": extracted.get("title", ""),
                        "source_type": "web",
                        "text": extracted.get("text", ""),
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
            if progress_callback:
                progress_callback("RAG: ищу материалы...")
            queries = self._build_query_variations(query_or_url)
            result["debug"]["query_variations"] = list(queries)
            all_urls: list[str] = []
            for q in queries:
                all_urls.extend(self.search._duckduckgo_urls(q))
            # wikipedia fallback urls
            wiki_queries = [f"site:ru.wikipedia.org {query_or_url}", f"википедия {query_or_url}"]
            for q in wiki_queries:
                all_urls.extend(self.search._duckduckgo_urls(q))
            urls = []
            for u in all_urls:
                if u not in urls:
                    urls.append(u)
            urls = urls[: max(1, max_sources * 3)]
            result["debug"]["urls"] = list(urls)
            if progress_callback:
                progress_callback(f"RAG: найдено URL: {len(urls)}")
            combined: list[str] = []
            opened = 0
            success = 0
            min_chars = 250 if len(query_or_url) <= 40 else 500
            for idx, url in enumerate(urls, start=1):
                if progress_callback:
                    progress_callback(f"RAG: извлекаю текст {idx}/{len(urls)}...")
                opened += 1
                try:
                    extracted = extract_text_from_url(url, min_chars=min_chars)
                    if not extracted.get("ok"):
                        raise RuntimeError(str(extracted.get("error") or "empty article"))
                    clean = clean_raw_text(extracted.get("text", ""))
                    try:
                        web_images = extract_images_from_web_html(url)
                    except Exception:
                        web_images = []
                    if len(clean) < min_chars:
                        continue
                    success += 1
                    result["sources"].append(
                        {
                            "url": url,
                            "title": extracted.get("title", ""),
                            "source_type": "web",
                            "text": extracted.get("text", ""),
                            "clean_text": clean,
                            "metadata": {"images": web_images},
                        }
                    )
                    result["debug"]["url_results"].append({"url": url, "ok": True, "text_len": len(clean)})
                    combined.append(f"[Источник: {extracted.get('title', url)}]\n{clean}")
                except Exception as exc:
                    result["errors"].append(f"{url}: {exc}")
                    result["debug"]["url_results"].append({"url": url, "ok": False, "error": str(exc)})
            if progress_callback:
                progress_callback(f"RAG: успешно извлечено источников: {success}")
            # wikipedia content fallback only if no text yet
            if not combined:
                for wq in [f"https://ru.wikipedia.org/wiki/{urllib.parse.quote(query_or_url.replace(' ', '_'))}"]:
                    extracted = extract_text_from_url(wq, min_chars=250)
                    if extracted.get("ok"):
                        txt = clean_raw_text(extracted.get("text", ""))
                        if txt:
                            result["sources"].append({"url": wq, "title": extracted.get("title", "Wikipedia"), "source_type": "wikipedia", "text": extracted.get("text", ""), "clean_text": txt, "metadata": {"images": []}})
                            combined.append(f"[Источник: {extracted.get('title', 'Wikipedia')}]\n{txt}")
                            break
            result["clean_text"] = clean_raw_text("\n\n".join(combined))[: self.max_chars]
            result["status"] = "ok" if result["clean_text"] else "error"
            if not result["clean_text"]:
                first_err = ", ".join([str(x.get("error")) for x in result["debug"]["url_results"] if not x.get("ok")][:5])
                result["errors"].append(f"Не удалось извлечь текст: найдено {len(urls)} URL, открыто {opened}, успешно {success}. Ошибки: {first_err or 'n/a'}")
        except Exception as exc:
            result["errors"].append(str(exc))
        return result

    def _is_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
