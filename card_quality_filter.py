from __future__ import annotations

import re

BAD_PATTERNS = [
    "что важно помнить",
    "что такое это",
    "что описано в тексте",
    "какая информация представлена",
    "почему это важно",
]


def score_card(card: dict) -> dict:
    front = re.sub(r"\s+", " ", (card.get("front") or "").strip())
    back = re.sub(r"\s+", " ", (card.get("back") or "").strip())
    excerpt = (card.get("source_excerpt") or "").lower()
    score = 0.5
    if 8 <= len(front) <= 140:
        score += 0.2
    if 20 <= len(back) <= 360:
        score += 0.2
    if "?" in front:
        score += 0.05
    if any(x in front.lower() for x in BAD_PATTERNS):
        score -= 0.35
    if len(front.split("?")) > 2:
        score -= 0.2
    if excerpt and any(tok in excerpt for tok in re.findall(r"[а-яa-z0-9]{5,}", back.lower())[:4]):
        score += 0.1
    if len(back.split()) < 4:
        score -= 0.25
    card = dict(card)
    card["quality_score"] = max(0.0, min(1.0, round(score, 3)))
    if card["quality_score"] < 0.45:
        card["difficulty"] = "easy"
    elif card["quality_score"] < 0.75:
        card["difficulty"] = "medium"
    else:
        card["difficulty"] = "hard"
    return card


def dedupe_cards(cards: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for card in cards:
        key = re.sub(r"\W+", "", f"{card.get('front','')}|{card.get('back','')}".lower())[:220]
        if key in seen:
            continue
        seen.add(key)
        out.append(card)
    return out


def polish_card(card: dict) -> dict:
    card = dict(card)
    card["front"] = re.sub(r"\s+", " ", (card.get("front") or "").strip())
    card["back"] = re.sub(r"\s+", " ", (card.get("back") or "").strip())
    if len(card["front"]) > 150:
        card["front"] = card["front"][:147].rsplit(" ", 1)[0].rstrip(" ,.;:") + "?"
    if len(card["back"]) > 420:
        card["back"] = card["back"][:417].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"
    return card


def filter_bad_cards(cards: list[dict]) -> list[dict]:
    scored = [score_card(polish_card(card)) for card in cards]
    deduped = dedupe_cards(scored)
    kept = [c for c in deduped if c.get("back") and c.get("quality_score", 0.0) >= 0.55]
    if len(kept) < 3:
        kept = [c for c in deduped if c.get("quality_score", 0.0) >= 0.45]
    return kept
