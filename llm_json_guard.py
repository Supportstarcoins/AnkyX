from __future__ import annotations

import json
import re
from typing import Any


class LLMJsonError(RuntimeError):
    pass


SAFE_JSON_SCHEMA = {"cards": [{"front": str, "back": str, "image_prompt": str, "tags": list[str]}]}

SYSTEM_JSON_ONLY = """
Ты НЕ чат-ассистент. Ты — JSON-генератор для флэш-карточек.
Верни только JSON формата:
{"cards":[{"front":"...?","back":"...","image_prompt":"...","tags":[]}]}
Правила:
- front — вопрос и заканчивается '?', не длиннее 120 символов.
- back — краткий ответ (1-3 предложения).
- image_prompt — конкретный английский промпт для SD.
- tags — массив строк.
- Без markdown и текста вокруг JSON.
""".strip()

SYSTEM_IMAGE_PROMPT_ONLY = """
Ты генератор JSON для Stable Diffusion.
Верни только JSON: {"image_prompt":"..."}
Требования:
- Только английский.
- Конкретные объекты, сцена, окружение.
- Стиль: educational illustration, clean background.
- Запрещены абстракции: random, something, concept, idea.
""".strip()


def validate_cards_schema(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    cards = obj.get("cards")
    if not isinstance(cards, list) or not cards:
        return False
    for card in cards:
        if not isinstance(card, dict):
            return False
        front = card.get("front")
        back = card.get("back")
        image_prompt = card.get("image_prompt")
        tags = card.get("tags")
        if not isinstance(front, str) or not front.strip():
            return False
        if not isinstance(back, str) or not back.strip():
            return False
        if not isinstance(image_prompt, str) or not image_prompt.strip():
            return False
        if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
            return False
    return True


def validate_image_prompt_schema(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("image_prompt"), str) and bool(str(obj.get("image_prompt")).strip())


def is_generic_image_prompt(prompt: str) -> bool:
    value = (prompt or "").strip().lower()
    if not value:
        return True
    markers = ("concept", "idea", "something", "random", "illustration of the concept")
    return any(m in value for m in markers) or len(value.split()) < 3


def extract_json_strict(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def ensure_question(front: str, back: str, context_text: str) -> tuple[str, str]:
    front = (front or "").strip()
    back = (back or "").strip()
    context_text = (context_text or "").strip()

    if not back:
        back = context_text.split(".")[0].strip() or "См. контекст."
    if not front:
        front = "Что это такое?"
    if "?" not in front:
        front = front.rstrip(".!") + "?"
    if not front.endswith("?"):
        front = front.rstrip(".!") + "?"
    return front, back


def fallback_cards_from_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    front = ""
    back = ""
    for pat in [r"front\s*[:\-]\s*(.+)", r"question\s*[:\-]\s*(.+)"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            front = m.group(1).strip()
            break
    for pat in [r"back\s*[:\-]\s*(.+)", r"answer\s*[:\-]\s*(.+)"]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            back = m.group(1).strip()
            break
    if not front and not back:
        return None
    front, back = ensure_question(front, back, text)
    return {"cards": [{"front": front, "back": back, "image_prompt": "detailed educational illustration, clean background", "tags": []}]}


def regex_extract_front_back(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    front_match = re.search(r"(?:front|question|вопрос)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    back_match = re.search(r"(?:back|answer|ответ)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    front = front_match.group(1).strip() if front_match else ""
    back = back_match.group(1).strip() if back_match else ""
    if not front and not back:
        return None
    return front, back


def normalize_cards_payload(cards: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in cards if isinstance(cards, list) else []:
        if not isinstance(raw, dict):
            continue
        front = str(raw.get("front") or "").strip() or "В чём суть?"
        if not front.endswith("?"):
            front = front.rstrip(".!") + "?"
        back = str(raw.get("back") or "").strip() or "—"
        image_prompt = str(raw.get("image_prompt") or "").strip() or "detailed educational illustration, clean background"
        tags_raw = raw.get("tags")
        tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()] if isinstance(tags_raw, list) else []
        normalized.append({"front": front, "back": back, "image_prompt": image_prompt, "tags": tags})
    return normalized


def call_ollama_json_strict_with_repair(
    ollama_client: Any,
    base_messages: list[dict[str, str]],
    mode: str = "cards",
) -> dict[str, Any]:
    options = {"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192, "stop": ["```", "\n\n\n"]}
    schema_validator = validate_image_prompt_schema if mode == "image_prompt_only" else validate_cards_schema

    response_1 = ollama_client.chat(base_messages, options=options)
    obj = extract_json_strict(response_1)
    if schema_validator(obj):
        return obj

    expected_schema = '{"image_prompt":"..."}' if mode == "image_prompt_only" else '{"cards":[{"front":"...?","back":"...","image_prompt":"...","tags":[]}]}'
    retry_messages = [
        {"role": "system", "content": SYSTEM_IMAGE_PROMPT_ONLY if mode == "image_prompt_only" else SYSTEM_JSON_ONLY},
        {"role": "user", "content": f"Верни только JSON по схеме {expected_schema}. Исправь ответ:\n{response_1}"},
    ]
    response_2 = ollama_client.chat(retry_messages, options=options)
    obj = extract_json_strict(response_2)
    if schema_validator(obj):
        return obj

    if mode == "image_prompt_only":
        prompt = "detailed subject, educational illustration, clean background"
        return {"image_prompt": prompt}

    parsed = regex_extract_front_back(response_2 or response_1)
    if parsed:
        front, back = ensure_question(parsed[0], parsed[1], context_text="")
    else:
        user_messages = [m.get("content", "") for m in base_messages if m.get("role") == "user"]
        source_text = str(user_messages[-1] if user_messages else "").strip()
        front = "О чём это?"
        back = re.sub(r"\s+", " ", source_text).strip()[:240] or "Краткое описание отсутствует."

    return {"cards": [{"front": front, "back": back, "image_prompt": "detailed educational illustration, clean background", "tags": []}]}
