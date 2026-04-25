from __future__ import annotations


class AICardOptimizer:
    def __init__(self, auto_improve: bool = False) -> None:
        self.auto_improve = auto_improve

    def suggest_card_improvement(self, card, grade_history=None):
        _ = grade_history
        back = (card or {}).get("back", "")
        if len(back) > 350:
            return {
                "message": "Карточка слишком сложная. Предлагаю разделить её на 2 карточки.",
                "actions": ["simplify", "split", "rewrite"],
            }
        return {"message": "Карточка выглядит нормально.", "actions": ["keep"]}

    def simplify_card(self, card):
        out = dict(card or {})
        out["back"] = (out.get("back") or "")[:260]
        return out

    def split_card(self, card):
        text = (card or {}).get("back", "")
        mid = max(1, len(text) // 2)
        c1 = dict(card or {})
        c2 = dict(card or {})
        c1["back"] = text[:mid].strip()
        c2["front"] = f"(2) {c2.get('front', '')}".strip()
        c2["back"] = text[mid:].strip()
        return [c1, c2]

    def rewrite_question(self, card):
        out = dict(card or {})
        front = (out.get("front") or "").rstrip("?")
        out["front"] = f"Кратко объясните: {front}?"
        return out

    def detect_duplicate_cards(self, cards):
        seen = {}
        dups = []
        for idx, card in enumerate(cards or []):
            key = ((card.get("front") or "").strip().lower(), (card.get("back") or "").strip().lower())
            if key in seen:
                dups.append((seen[key], idx))
            else:
                seen[key] = idx
        return dups

    def merge_duplicates(self, cards):
        out = []
        seen = set()
        for card in cards or []:
            key = ((card.get("front") or "").strip().lower(), (card.get("back") or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(card)
        return out

    def create_difference_card(self, term_a, term_b):
        return {
            "front": f"Чем отличается {term_a} от {term_b}?",
            "back": f"Сравните {term_a} и {term_b} по ключевым признакам.",
        }

    def create_mini_test(self, cards):
        return [c.get("front", "") for c in (cards or [])[:5]]
