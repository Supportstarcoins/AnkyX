from typing import Any, Dict, List


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
