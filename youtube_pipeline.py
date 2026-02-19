from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TimedText:
    start_ms: int
    end_ms: int
    text: str


def extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", url)
    if not m:
        raise ValueError("invalid youtube url")
    return m.group(1)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    raise RuntimeError("ffmpeg не найден. Установите ffmpeg и добавьте в PATH. Пример: sudo apt-get install ffmpeg")


def ensure_youtube_import(conn: sqlite3.Connection, url: str, title: str = "") -> tuple[int, str]:
    video_id = extract_video_id(url)
    digest = _sha256(url.strip().lower())
    cur = conn.cursor()
    cur.execute("SELECT id FROM youtube_imports WHERE sha256 = ? LIMIT 1;", (digest,))
    row = cur.fetchone()
    if row:
        return int(row[0]), video_id
    cur.execute(
        "INSERT INTO youtube_imports (video_id, url, title, sha256, created_at) VALUES (?, ?, ?, ?, ?);",
        (video_id, url, title, digest, int(time.time())),
    )
    return int(cur.lastrowid), video_id


def chunk_timed_segments(segments: list[TimedText], min_sent: int, max_sent: int) -> list[TimedText]:
    result: list[TimedText] = []
    buf: list[TimedText] = []
    sentence_count = 0

    def flush() -> None:
        nonlocal buf, sentence_count
        if not buf:
            return
        result.append(TimedText(buf[0].start_ms, buf[-1].end_ms, " ".join(x.text.strip() for x in buf).strip()))
        buf = []
        sentence_count = 0

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        buf.append(seg)
        sentence_count += len(re.findall(r"[.!?]+", text)) or 1
        long_pause = len(buf) >= 2 and (seg.start_ms - buf[-2].end_ms) > 1600
        if sentence_count >= max_sent or (sentence_count >= min_sent and (re.search(r"[.!?]\s*$", text) or long_pause)):
            flush()
    flush()
    return result


def build_clips(video_path: str, chunks: list[TimedText], out_root: str, video_id: str) -> list[str]:
    ensure_ffmpeg()
    out_dir = Path(out_root) / "youtube" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        clip_id = f"clip_{idx:04d}"
        out_path = out_dir / f"{clip_id}.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", f"{chunk.start_ms / 1000:.3f}", "-to", f"{chunk.end_ms / 1000:.3f}",
            "-c:v", "libx264", "-c:a", "aac", str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        paths.append(str(out_path))
    return paths


def store_clips(conn: sqlite3.Connection, video_id: str, chunks: list[TimedText], clip_paths: list[str]) -> list[int]:
    cur = conn.cursor()
    ids: list[int] = []
    for idx, (chunk, path) in enumerate(zip(chunks, clip_paths, strict=False), start=1):
        clip_id = f"{video_id}_{idx:04d}"
        cur.execute("SELECT id FROM youtube_clips WHERE clip_id = ? LIMIT 1;", (clip_id,))
        row = cur.fetchone()
        if row:
            ids.append(int(row[0]))
            continue
        cur.execute(
            "INSERT INTO youtube_clips (clip_id, video_id, start_ms, end_ms, text, path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (clip_id, video_id, chunk.start_ms, chunk.end_ms, chunk.text, path, int(time.time())),
        )
        ids.append(int(cur.lastrowid))
    return ids


def media_payload(asset_id: int | None, path: str) -> dict:
    return {"type": "video", "asset_id": str(asset_id) if asset_id else None, "path": path}
