from __future__ import annotations

import re


def _difficulty_by_length(text: str) -> str:
    words = len(re.findall(r"\w+", text or ""))
    if words <= 6:
        return "easy"
    if words <= 14:
        return "medium"
    return "hard"


def _keywords(text: str) -> list[str]:
    seen = []
    for token in re.findall(r"[\w'-]{4,}", (text or "").lower()):
        if token not in seen:
            seen.append(token)
        if len(seen) >= 6:
            break
    return seen


def build_listening_cards(youtube_result: dict, options: dict | None = None) -> list[dict]:
    opts = dict(options or {})
    cards: list[dict] = []
    for seg in youtube_result.get("segments") or []:
        transcript = str(seg.get("text") or "").strip()
        if not transcript:
            continue
        words = _keywords(transcript)
        card = {
            "front": "Прослушайте фрагмент и напишите, что сказано.",
            "back": transcript,
            "explanation": f"Фрагмент {float(seg.get('start') or 0.0):.1f}–{float(seg.get('end') or 0.0):.1f} сек.",
            "card_type": "listening",
            "difficulty": _difficulty_by_length(transcript),
            "audio_path": seg.get("audio_path") or youtube_result.get("audio_path") or "",
            "video_path": seg.get("video_path") or youtube_result.get("media_path") or "",
            "front_image_path": seg.get("thumbnail_path") or youtube_result.get("thumbnail_path") or "",
            "source_type": "youtube",
            "source_url": youtube_result.get("url") or "",
            "source_title": youtube_result.get("title") or "",
            "time_start": float(seg.get("start") or 0.0),
            "time_end": float(seg.get("end") or 0.0),
            "language": youtube_result.get("language") or opts.get("language") or "",
            "asr_confidence": float(seg.get("confidence") or 0.0),
            "translation": "",
            "key_words": words,
            "phrase_notes": "",
            "cloze": re.sub(r"\b(\w+)\b", "_____", transcript, count=1),
            "needs_image": False,
            "image_prompt": "",
            "metadata": {
                "mode": "youtube_listening",
                "segment_index": seg.get("index"),
                "key_words": words,
            },
        }
        cards.append(card)
    return cards
