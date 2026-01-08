from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple


def _iter_segments(delta_ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for op in delta_ops:
        insert = op.get("insert")
        attrs = op.get("attributes") or {}
        if not isinstance(insert, str):
            continue
        parts = insert.split("\n")
        for part in parts:
            text = part.strip()
            if not text:
                continue
            segments.append({"text": text, "attrs": attrs})
    return segments


def parse_quill_delta(delta: Dict[str, Any]) -> List[Dict[str, str]]:
    ops = delta.get("ops") or []
    segments = _iter_segments(ops)
    cards: List[Dict[str, str]] = []
    pending_front: str | None = None
    pending_type: str | None = None

    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        attrs = segment.get("attrs") or {}
        is_bold = bool(attrs.get("bold"))
        is_underline = bool(attrs.get("underline"))
        is_definition = bool(attrs.get("color") or attrs.get("background"))

        if is_underline:
            pending_front = text
            pending_type = "underline"
            continue
        if is_bold:
            pending_front = text
            pending_type = "bold"
            continue

        if pending_front:
            if pending_type == "underline" or pending_type == "bold" or is_definition:
                cards.append({"front": pending_front, "back": text})
                pending_front = None
                pending_type = None
            continue

    return cards


class _QuillHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.segments: List[Dict[str, Any]] = []
        self._style_stack: List[Dict[str, Optional[str]]] = []
        self._block_tags = {"p", "div", "br", "li"}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        style_state = {"bold": False, "underline": False, "color": None, "background": None}
        if self._style_stack:
            parent = self._style_stack[-1]
            style_state.update(parent)
        if tag in {"strong", "b"}:
            style_state["bold"] = True
        if tag == "u":
            style_state["underline"] = True
        if tag == "span":
            style_attr = dict(attrs).get("style", "") or ""
            for part in style_attr.split(";"):
                key, _, value = part.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "color" and value:
                    style_state["color"] = value
                if key == "background" and value:
                    style_state["background"] = value
        self._style_stack.append(style_state)
        if tag in self._block_tags:
            self.segments.append({"text": "\n", "attrs": {}})

    def handle_endtag(self, tag: str) -> None:
        if self._style_stack:
            self._style_stack.pop()
        if tag in {"p", "div", "li"}:
            self.segments.append({"text": "\n", "attrs": {}})

    def handle_data(self, data: str) -> None:
        if not data:
            return
        attrs = self._style_stack[-1] if self._style_stack else {}
        self.segments.append({"text": data, "attrs": attrs})


def _segments_from_html(html: str) -> List[Dict[str, Any]]:
    parser = _QuillHtmlParser()
    parser.feed(html)
    return parser.segments


def _plain_text_from_html(html: str) -> str:
    segments = _segments_from_html(html)
    text = "".join(segment.get("text", "") for segment in segments)
    return text.strip()


def parse_quill_html_to_cards(html: str) -> List[Dict[str, str]]:
    segments = _segments_from_html(html)
    cards: List[Dict[str, str]] = []
    pending_front: Optional[str] = None
    pending_type: Optional[str] = None

    for segment in segments:
        raw_text = segment.get("text", "")
        text = raw_text.strip()
        if not text:
            continue
        attrs = segment.get("attrs") or {}
        is_bold = bool(attrs.get("bold"))
        is_underline = bool(attrs.get("underline"))
        has_color = bool(attrs.get("color"))
        has_background = bool(attrs.get("background"))

        if is_underline and not pending_front:
            pending_front = text
            pending_type = "underline"
            continue
        if is_bold and not pending_front:
            pending_front = text
            pending_type = "bold"
            continue

        if pending_front:
            if has_color or has_background or pending_type in {"underline", "bold"}:
                cards.append({"front": pending_front, "back": text})
                pending_front = None
                pending_type = None
            continue

    if cards:
        return cards

    plain_text = _plain_text_from_html(html)
    if not plain_text:
        return []
    parts = [part.strip() for part in plain_text.split("\n\n") if part.strip()]
    return [{"front": part, "back": ""} for part in parts]
