from __future__ import annotations

import re

STOP_TOPIC = {
    "это", "этот", "эта", "эти", "который", "которая", "которые", "такой", "такая", "такие",
    "для", "или", "как", "при", "если", "также", "очень", "можно", "нужно",
}
BAD_TOPIC_TOKENS = {
    "живут", "служат", "состоят", "среди", "между", "особая", "условно", "это", "они", "он", "она", "данный", "которые", "пара",
}


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _topic_signature(text: str) -> set[str]:
    tokens = re.findall(r"[А-ЯA-Zа-яa-z][а-яa-z0-9\-]{3,}", (text or "").lower())
    out = {t for t in tokens if t not in STOP_TOPIC}
    return set(sorted(out)[:30])


def _topic_title(text: str) -> str:
    tokens = re.findall(r"[А-ЯA-Zа-яa-z][а-яa-z0-9\-]{3,}", text or "")
    freq: dict[str, int] = {}
    for tok in tokens:
        low = tok.lower()
        if low in STOP_TOPIC or low in BAD_TOPIC_TOKENS:
            continue
        freq[low] = freq.get(low, 0) + 1
    if not freq:
        return "Общая тема"
    return max(freq.items(), key=lambda x: x[1])[0].capitalize()


def _topic_shift(prev_sig: set[str], cur_sig: set[str]) -> bool:
    if not prev_sig or not cur_sig:
        return False
    overlap = len(prev_sig & cur_sig) / max(1, len(prev_sig | cur_sig))
    return overlap < 0.14


def split_into_semantic_chunks(text: str, min_words: int = 120, max_words: int = 500) -> list[dict]:
    source = (text or "").strip()
    if not source:
        return []
    blocks = [b.strip() for b in re.split(r"\n{2,}", source) if b.strip()]
    chunks: list[dict] = []
    buffer = ""
    cursor = 0
    idx = 0
    prev_topic_sig: set[str] = set()

    def flush(buf: str, start_offset: int) -> int:
        nonlocal idx, prev_topic_sig
        clean = re.sub(r"\s+", " ", (buf or "")).strip()
        if not clean:
            return start_offset
        wc = _word_count(clean)
        end_offset = start_offset + len(clean)
        sig = _topic_signature(clean)
        chunk = {
            "chunk_id": f"chunk_{idx:03d}",
            "text": clean,
            "word_count": wc,
            "source_index": idx,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "topic_title": _topic_title(clean),
            "topic_signature": sorted(sig),
        }
        chunks.append(chunk)
        idx += 1
        prev_topic_sig = sig
        return end_offset

    for block in blocks:
        block_sig = _topic_signature(block)
        if buffer and _topic_shift(_topic_signature(buffer), block_sig):
            cursor = flush(buffer, cursor)
            buffer = ""

        candidate = f"{buffer}\n\n{block}".strip() if buffer else block
        wc = _word_count(candidate)
        if wc <= max_words:
            buffer = candidate
            if wc >= min_words:
                cursor = flush(buffer, cursor)
                buffer = ""
            continue

        if buffer:
            cursor = flush(buffer, cursor)
            buffer = ""
        sentences = re.split(r"(?<=[.!?])\s+", block)
        sentence_buf = ""
        sentence_sig: set[str] = set()
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            cur_sig = _topic_signature(sent)
            if sentence_buf and _topic_shift(sentence_sig, cur_sig):
                cursor = flush(sentence_buf, cursor)
                sentence_buf = sent
                sentence_sig = cur_sig
                continue
            candidate_sent = f"{sentence_buf} {sent}".strip() if sentence_buf else sent
            if _word_count(candidate_sent) > max_words and sentence_buf:
                cursor = flush(sentence_buf, cursor)
                sentence_buf = sent
                sentence_sig = cur_sig
            else:
                sentence_buf = candidate_sent
                sentence_sig = sentence_sig | cur_sig if sentence_sig else cur_sig
        if sentence_buf:
            if _word_count(sentence_buf) < min_words and chunks:
                prev = chunks.pop()
                if _topic_shift(set(prev.get("topic_signature", [])), _topic_signature(sentence_buf)):
                    chunks.append(prev)
                    cursor = flush(sentence_buf, cursor)
                else:
                    merged = f"{prev['text']} {sentence_buf}".strip()
                    chunks.append(
                        {
                            **prev,
                            "text": merged,
                            "word_count": _word_count(merged),
                            "end_offset": prev["start_offset"] + len(merged),
                            "topic_title": _topic_title(merged),
                            "topic_signature": sorted(_topic_signature(merged)),
                        }
                    )
                    cursor = chunks[-1]["end_offset"]
            else:
                cursor = flush(sentence_buf, cursor)

    if buffer:
        if chunks and _word_count(buffer) < min_words:
            prev = chunks.pop()
            if _topic_shift(set(prev.get("topic_signature", [])), _topic_signature(buffer)):
                chunks.append(prev)
                flush(buffer, cursor)
            else:
                merged = f"{prev['text']} {buffer}".strip()
                chunks.append(
                    {
                        **prev,
                        "text": merged,
                        "word_count": _word_count(merged),
                        "end_offset": prev["start_offset"] + len(merged),
                        "topic_title": _topic_title(merged),
                        "topic_signature": sorted(_topic_signature(merged)),
                    }
                )
        else:
            flush(buffer, cursor)
    return chunks
