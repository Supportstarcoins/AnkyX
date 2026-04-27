from __future__ import annotations

import re
from difflib import SequenceMatcher

BANNED_GENERIC_PATTERNS = [
    "какой факт указан",
    "какой факт указан в материале",
    "что означает термин «это»",
    "что означает термин \"это\"",
    "что означает термин",
    "какова указанная причина",
    "что важно помнить",
    "что описано в тексте",
    "какая информация представлена",
    "что сказано в материале",
    "что говорится в тексте",
    "какой вывод можно сделать",
    "почему это важно",
    "что такое это",
    "что такое особенности",
    "какой факт можно выделить",
    "что конкретно сказано",
    "что известно про",
    "что известно о",
]

STOP_TERMS = {
    "это", "этот", "эта", "эти", "данный", "данная", "данные",
    "такой", "такая", "такие", "который", "которая", "которые",
    "он", "она", "они", "оно",
}

KNOWN_TYPES = {
    "definition",
    "function",
    "range/quantity",
    "cause",
    "difference",
    "list/classification",
    "anatomy/composition",
    "fact",
}
BOOST_TYPES = {"definition", "function", "range/quantity", "cause", "difference", "list/classification", "anatomy/composition", "fact"}
FILLER_TOKENS = {"материал", "материале", "текст", "тексте", "факт", "указан", "указано", "указанная"}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _tokens(value: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9]+", (value or "").lower())


def _looks_too_generic(front_low: str) -> bool:
    if any(p in front_low for p in BANNED_GENERIC_PATTERNS):
        return True
    generic_forms = [
        r"^что\s+(описано|сказано|говорится)\b",
        r"^какой\s+факт\b",
        r"^какой\s+вывод\b",
        r"^почему\s+это\s+важно\b",
        r"^что\s+конкретно\s+сказано\b",
        r"^что\s+известно\s+про\b",
        r"^что\s+известно\s+о\b",
        r"^что\s+означает\s+термин\s+[«\"]?это\b",
    ]
    return any(re.search(p, front_low) for p in generic_forms)


def _looks_like_token_garbage(front: str) -> bool:
    toks = _tokens(front)
    if not toks:
        return True
    long_toks = [t for t in toks if len(t) >= 3]
    if len(long_toks) < 2:
        return True
    bad_ratio = sum(1 for t in toks if len(t) <= 2) / max(1, len(toks))
    if bad_ratio > 0.5:
        return True
    if re.search(r"[^\w\s?.,!«»\"-]{2,}", front):
        return True
    return False


def _has_specific_object(front: str, excerpt: str) -> bool:
    f_tokens = [t for t in _tokens(front) if len(t) >= 4 and t not in STOP_TERMS and t not in FILLER_TOKENS]
    if not f_tokens:
        return False
    excerpt_low = (excerpt or "").lower()
    return any(t in excerpt_low for t in f_tokens[:6])


def _front_back_too_similar(front: str, back: str) -> bool:
    ratio = SequenceMatcher(None, front.lower(), back.lower()).ratio()
    if ratio > 0.85:
        return True
    f = set(_tokens(front))
    b = set(_tokens(back))
    if not f or not b:
        return False
    overlap = len(f & b) / max(1, len(f))
    return overlap > 0.9


def _has_stop_term_as_term(front: str) -> bool:
    front_low = front.lower()
    m = re.search(r"термин\s+[«\"]?([а-яa-z-]+)", front_low)
    if m and m.group(1) in STOP_TERMS:
        return True
    if re.search(r"что\s+такое\s+(это|этот|эта|эти|данный|он|она|они|оно)\b", front_low):
        return True
    return False


def score_card(card: dict) -> dict:
    front = _normalize_text(card.get("front") or "")
    back = _normalize_text(card.get("back") or "")
    excerpt = _normalize_text(card.get("source_excerpt") or "")
    topic = _normalize_text(card.get("topic") or "")
    card_type = str(card.get("card_type") or "").strip().lower()

    score = 0.5
    front_low = front.lower()

    generic = _looks_too_generic(front_low)
    stop_term = _has_stop_term_as_term(front)
    garbage = _looks_like_token_garbage(front)
    specific = _has_specific_object(front, excerpt)

    if generic or stop_term or garbage:
        score = min(score, 0.2)

    if len(front) < 8:
        score -= 0.25
    elif len(front) <= 160:
        score += 0.1

    back_words = len(back.split())
    if len(back) > 350:
        score -= 0.2
    elif 2 <= back_words <= 40:
        score += 0.12
    if back_words < 2:
        score -= 0.25

    if "?" in front:
        score += 0.04
    if not excerpt:
        score -= 0.2
    else:
        score += 0.08
    if not topic:
        score -= 0.1
    else:
        score += 0.06

    if card_type in KNOWN_TYPES:
        score += 0.08
        if card_type in BOOST_TYPES:
            score += 0.05
    else:
        score -= 0.15

    if not specific:
        score -= 0.2
    else:
        score += 0.1

    if _front_back_too_similar(front, back):
        score -= 0.2

    if excerpt and back:
        b_tokens = [t for t in _tokens(back) if len(t) >= 4]
        if b_tokens and any(t in excerpt.lower() for t in b_tokens[:5]):
            score += 0.08
        else:
            score -= 0.1

    if generic or stop_term or garbage:
        score = min(score, 0.2)

    out = dict(card)
    out["quality_score"] = max(0.0, min(1.0, round(score, 3)))
    out.setdefault("topic", topic)
    if out["quality_score"] < 0.45:
        out["difficulty"] = "easy"
    elif out["quality_score"] < 0.75:
        out["difficulty"] = "medium"
    else:
        out["difficulty"] = "hard"
    return out


def _front_signature(front: str) -> str:
    toks = [t for t in _tokens(front) if t not in FILLER_TOKENS]
    return " ".join(sorted(dict.fromkeys(toks)))


def _token_similarity(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def dedupe_cards(cards: list[dict]) -> list[dict]:
    winners: list[dict] = []
    for card in cards:
        front = _normalize_text(card.get("front") or "")
        back = _normalize_text(card.get("back") or "")
        replaced = False
        for idx, existing in enumerate(winners):
            e_front = _normalize_text(existing.get("front") or "")
            e_back = _normalize_text(existing.get("back") or "")
            same_front = front.lower() == e_front.lower()
            same_back = back.lower() == e_back.lower()
            similar_front = _token_similarity(front, e_front) > 0.8
            same_signature = _front_signature(front) == _front_signature(e_front)
            if same_front or same_back or similar_front or same_signature:
                better = card if card.get("quality_score", 0.0) >= existing.get("quality_score", 0.0) else existing
                winners[idx] = better
                replaced = True
                break
        if not replaced:
            winners.append(card)
    return winners


def polish_card(card: dict) -> dict:
    card = dict(card)
    card["front"] = _normalize_text(card.get("front") or "")
    card["back"] = _normalize_text(card.get("back") or "")
    card["source_excerpt"] = _normalize_text(card.get("source_excerpt") or "")
    if len(card["front"]) > 170:
        card["front"] = card["front"][:167].rsplit(" ", 1)[0].rstrip(" ,.;:") + "?"
    if len(card["back"]) > 420:
        card["back"] = card["back"][:417].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"
    return card


def filter_bad_cards(cards: list[dict]) -> list[dict]:
    hard = []
    for card in cards:
        c = polish_card(card)
        front = (c.get("front") or "").strip()
        if _looks_too_generic(front.lower()) or _has_stop_term_as_term(front) or _looks_like_token_garbage(front):
            continue
        hard.append(c)
    scored = [score_card(card) for card in hard]
    deduped = dedupe_cards(scored)
    strong = [c for c in deduped if c.get("back") and c.get("quality_score", 0.0) >= 0.55]
    if len(strong) >= 3:
        return strong
    return [c for c in deduped if c.get("back") and c.get("quality_score", 0.0) >= 0.45]
