from __future__ import annotations

import re


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def split_into_semantic_chunks(text: str, min_words: int = 120, max_words: int = 500) -> list[dict]:
    source = (text or "").strip()
    if not source:
        return []
    blocks = [b.strip() for b in re.split(r"\n{2,}", source) if b.strip()]
    chunks: list[dict] = []
    buffer = ""
    cursor = 0
    idx = 0

    def flush(buf: str, start_offset: int) -> int:
        nonlocal idx
        clean = re.sub(r"\s+", " ", (buf or "")).strip()
        if not clean:
            return start_offset
        wc = _word_count(clean)
        end_offset = start_offset + len(clean)
        chunks.append(
            {
                "chunk_id": f"chunk_{idx:03d}",
                "text": clean,
                "word_count": wc,
                "source_index": idx,
                "start_offset": start_offset,
                "end_offset": end_offset,
            }
        )
        idx += 1
        return end_offset

    for block in blocks:
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
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            candidate_sent = f"{sentence_buf} {sent}".strip() if sentence_buf else sent
            if _word_count(candidate_sent) > max_words and sentence_buf:
                cursor = flush(sentence_buf, cursor)
                sentence_buf = sent
            else:
                sentence_buf = candidate_sent
        if sentence_buf:
            if _word_count(sentence_buf) < min_words and chunks:
                prev = chunks.pop()
                merged = f"{prev['text']} {sentence_buf}".strip()
                chunks.append(
                    {
                        **prev,
                        "text": merged,
                        "word_count": _word_count(merged),
                        "end_offset": prev["start_offset"] + len(merged),
                    }
                )
                cursor = chunks[-1]["end_offset"]
            else:
                cursor = flush(sentence_buf, cursor)
    if buffer:
        if chunks and _word_count(buffer) < min_words:
            prev = chunks.pop()
            merged = f"{prev['text']} {buffer}".strip()
            chunks.append(
                {
                    **prev,
                    "text": merged,
                    "word_count": _word_count(merged),
                    "end_offset": prev["start_offset"] + len(merged),
                }
            )
        else:
            flush(buffer, cursor)
    return chunks
