from __future__ import annotations

import html
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Iterable


class _ReadableHTMLParser(HTMLParser):
    """Small stdlib-only HTML to readable text extractor.

    It intentionally ignores navigation/JS/CSS and keeps headings, paragraphs,
    list items, table cells and article-like content. This avoids returning only
    DuckDuckGo snippets and lets the workspace use real page body text.
    """

    _skip_tags = {
        "script", "style", "noscript", "svg", "canvas", "iframe", "form",
        "button", "select", "option", "nav", "header", "footer", "aside",
    }
    _break_tags = {
        "p", "br", "div", "section", "article", "main", "li", "tr", "td", "th",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag = (tag or "").lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in self._break_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        tag = (tag or "").lower()
        if tag in self._skip_tags and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in self._break_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        text = (data or "").strip()
        if not text:
            return
        if self._in_title:
            self.title += (" " if self.title else "") + text
        self.parts.append(text)
        self.parts.append(" ")

    def text(self) -> str:
        return "".join(self.parts)


class RagWebSearch:
    def __init__(self, max_pages: int = 4, timeout: int = 12, max_chars: int = 30000) -> None:
        self.max_pages = max(1, int(max_pages or 4))
        self.timeout = max(5, int(timeout or 12))
        self.max_chars = max(4000, int(max_chars or 30000))
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )

    def search_and_extract(self, query: str) -> str:
        query = self._clean_query(query)
        if not query:
            raise RuntimeError("Пустой поисковый запрос")

        urls = self._duckduckgo_urls(query)
        if not urls:
            raise RuntimeError("Поиск не вернул подходящих страниц")

        chunks: list[str] = []
        used = 0
        for url in urls[: self.max_pages]:
            if used >= self.max_chars:
                break
            try:
                page = self.fetch_page_text(url)
            except Exception as exc:
                logging.warning("RAG page fetch failed for %s: %s", url, exc)
                continue
            page_text = self.clean_extracted_text(page.get("text", ""))
            if len(page_text) < 500:
                continue
            title = self._clean_line(page.get("title") or "Материал")
            remaining = max(1000, self.max_chars - used)
            piece = f"# {title}\n\n{page_text[:remaining].strip()}"
            chunks.append(piece)
            used += len(piece)
            time.sleep(0.15)

        if not chunks:
            raise RuntimeError("Страницы найдены, но полезный текст извлечь не удалось")

        return self.clean_extracted_text("\n\n".join(chunks))[: self.max_chars]

    def fetch_page_text(self, url: str) -> dict[str, str]:
        raw_html = self._http_get(url, timeout=self.timeout)
        parser = _ReadableHTMLParser()
        parser.feed(raw_html)
        title = html.unescape(parser.title or self._title_from_url(url))
        text = parser.text()
        return {"url": url, "title": title, "text": text}

    def _duckduckgo_urls(self, query: str) -> list[str]:
        ddg_url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        raw = self._http_get(ddg_url, timeout=self.timeout)
        urls: list[str] = []

        # DuckDuckGo HTML result links usually look like /l/?uddg=<encoded_url>.
        for match in re.finditer(r'href=["\']([^"\']+)["\']', raw, flags=re.I):
            href = html.unescape(match.group(1))
            url = self._normalize_result_href(href)
            if not url:
                continue
            if self._is_noise_url(url):
                continue
            if url not in urls:
                urls.append(url)
            if len(urls) >= self.max_pages * 2:
                break
        return urls

    def _normalize_result_href(self, href: str) -> str | None:
        if not href:
            return None
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/"):
            href = "https://duckduckgo.com" + href
        parsed = urllib.parse.urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg", [""])[0]
            href = urllib.parse.unquote(uddg)
        if not href.startswith(("http://", "https://")):
            return None
        return href.split("#", 1)[0]

    def _http_get(self, url: str, timeout: int) -> str:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ru,en;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read(2_000_000)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP ошибка {exc.code} при загрузке {url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ошибка сети при загрузке {url}: {exc.reason}") from exc

        charset = self._charset_from_content_type(content_type) or "utf-8"
        try:
            return raw.decode(charset, errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")

    def clean_extracted_text(self, text: str) -> str:
        text = html.unescape(text or "")
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        text = re.sub(r"(?i)\b(cookie|cookies|privacy policy|terms of use|subscribe|sign in|log in|advertisement|реклама|куки|подписаться|войти|регистрация|читать далее|поделиться)\b", " ", text)
        text = text.replace("\xa0", " ")
        lines = []
        seen: set[str] = set()
        for raw_line in text.splitlines():
            line = self._clean_line(raw_line)
            if not line:
                continue
            if len(line) < 25 and not line.startswith("#"):
                continue
            key = re.sub(r"\W+", "", line.lower())[:120]
            if key in seen:
                continue
            seen.add(key)
            # Drop navigation-like rows with too many separated items.
            if line.count("|") >= 3 or line.count("›") >= 3:
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _clean_query(self, query: str) -> str:
        query = re.sub(r"https?://\S+", " ", query or "")
        query = re.sub(r"(?i)\bat\s+DuckDuckGo\b", " ", query)
        query = re.sub(r"(?i)\b(сгенерируй|создай|карточку|карточка|найди материалы|с картинкой|с изображением)\b", " ", query)
        query = re.sub(r"\s+", " ", query).strip(" :-,.;")
        return query[:240]

    def _clean_line(self, line: str) -> str:
        line = html.unescape(line or "")
        line = re.sub(r"\s+", " ", line).strip()
        return line.strip(" \t\r\n-–—|•")

    def _is_noise_url(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc.lower()
        bad_hosts = (
            "duckduckgo.com", "google.com", "yandex.", "bing.com", "youtube.com",
            "youtu.be", "tiktok.com", "vk.com", "facebook.com", "twitter.com", "x.com",
            "instagram.com", "pinterest.", "telegram.", "rutube.", "dzen.ru",
        )
        return any(bad in host for bad in bad_hosts)

    def _title_from_url(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        name = parsed.netloc + parsed.path
        name = urllib.parse.unquote(name).replace("/", " ").replace("-", " ")
        return self._clean_line(name) or "Материал"

    def _charset_from_content_type(self, content_type: str) -> str | None:
        match = re.search(r"charset=([\w\-]+)", content_type or "", flags=re.I)
        return match.group(1) if match else None
