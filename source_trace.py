from __future__ import annotations


def build_source_trace(source_type: str, source: dict, chunk: dict | None = None) -> dict:
    chunk = chunk or {}
    return {
        "source_type": source_type or source.get("source_type") or "manual",
        "source_url": source.get("url") or source.get("source_url") or "",
        "source_title": source.get("title") or source.get("source_title") or "",
        "chunk_id": chunk.get("chunk_id") or "",
        "source_excerpt": (chunk.get("text") or source.get("clean_text") or "")[:500],
        "time_start": chunk.get("time_start"),
        "time_end": chunk.get("time_end"),
        "metadata": source.get("metadata") or {},
    }
