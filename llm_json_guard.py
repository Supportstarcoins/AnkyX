from __future__ import annotations

import json
import re
from typing import Any


class LLMJsonError(RuntimeError):
    pass


SAFE_JSON_SCHEMA = {"cards": [{"front": str, "back": str, "image_prompt": str, "tags": list[str]}]}

SYSTEM_JSON_ONLY = """
Ты НЕ чат-ассистент. Ты — JSON-генератор для флэш-карточек.

ТЫ ОБЯЗАН:
- вернуть ТОЛЬКО валидный JSON без markdown, без пояснений, без префиксов/суффиксов, без текста вокруг.
- НИКОГДА не писать: "я не могу", "я не умею", "конфиденциально", "не рендерю", "не генерирую картинки".
- Ты не генерируешь изображение напрямую. Ты генерируешь image_prompt, а изображение сделает внешняя система.
- если нужно изображение: всегда выдай поле image_prompt (короткий промпт на английском для SD).
- FRONT = вопрос и заканчивается "?" (обязательно).
- BACK = ответ (кратко, без воды).
- Если пользователь просит "нарисуй/сгенерируй картинку" — всё равно верни JSON с image_prompt.
- Если пользователь просит "сгенерируй карточку" — верни минимум одну карточку.

СТРОГИЙ ФОРМАТ (единственно допустимый):
{"cards":[{"front":"... ?","back":"...","image_prompt":"...","tags":["..."]}]}

Правила:
- front/back/image_prompt НЕ могут быть пустыми.
- tags всегда массив строк, можно пустой [].
- Никаких переносов форматирования, только обычные строки.
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
    snippet = text[start:end+1]
    try:
        obj = json.loads(snippet)
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
        if context_text:
            front = "О чем говорится?"
        elif back:
            front = "Что это такое?"
        else:
            front = "Какой ответ верный?"

    front = re.sub(r"^\s*Вопрос\s*:\s*", "", front, flags=re.IGNORECASE)

    if " — " in front or ":" in front:
        base = front.split(" — ")[0].split(":")[0].strip()
        front = f"Что означает {base}?" if base else "Что это означает?"

    if "?" not in front:
        normalized = front.strip().rstrip(".!")
        front = f"Что верно про {normalized}?" if normalized else "Что это такое?"

    if not front.endswith("?"):
        front = front.rstrip(".!") + "?"

    return front, back


def fallback_cards_from_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    front = ""
    back = ""
    patterns = [
        r"front\s*[:\-]\s*(.+)",
        r"question\s*[:\-]\s*(.+)",
    ]
    for pat in patterns:
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
    return {"cards": [{"front": front, "back": back, "image_prompt": "illustration of the concept", "tags": []}]}
