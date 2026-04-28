from __future__ import annotations

import math
import shutil
import subprocess
from typing import Iterable

FFMPEG_ERROR_MESSAGE = "Для нарезки аудио/видео нужен ffmpeg."


def _ensure_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(FFMPEG_ERROR_MESSAGE)
    return ffmpeg_path


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def split_by_transcript_segments(transcript_segments, min_sec: int = 3, max_sec: int = 15) -> list[dict]:
    segments = []
    for raw in transcript_segments or []:
        start = _safe_float(raw.get("start"))
        duration = _safe_float(raw.get("duration"), 0.0)
        end = _safe_float(raw.get("end"), start + duration)
        if end <= start:
            end = start + max(0.3, duration)
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        segments.append({"start": start, "end": end, "text": text})
    if not segments:
        return []

    merged: list[dict] = []
    buffer: dict | None = None
    for seg in segments:
        seg_len = max(0.1, seg["end"] - seg["start"])
        if buffer is None:
            buffer = dict(seg)
            continue
        buffer_len = max(0.1, buffer["end"] - buffer["start"])
        gap = max(0.0, seg["start"] - buffer["end"])
        if buffer_len < min_sec or (buffer_len + seg_len <= max_sec and gap <= 1.2):
            buffer["end"] = max(buffer["end"], seg["end"])
            buffer["text"] = f"{buffer.get('text', '').strip()} {seg.get('text', '').strip()}".strip()
        else:
            merged.append(buffer)
            buffer = dict(seg)
    if buffer is not None:
        merged.append(buffer)

    out: list[dict] = []
    for seg in merged:
        start = _safe_float(seg.get("start"))
        end = _safe_float(seg.get("end"), start)
        text = str(seg.get("text") or "").strip()
        if end - start <= max_sec:
            out.append({"start": start, "end": end, "text": text})
            continue
        parts = max(1, math.ceil((end - start) / max_sec))
        part_len = (end - start) / parts
        for idx in range(parts):
            p_start = start + idx * part_len
            p_end = min(end, p_start + part_len)
            out.append({"start": p_start, "end": p_end, "text": text})
    return out


def _probe_duration(audio_path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return 0.0
    return _safe_float((proc.stdout or "").strip(), 0.0)


def split_audio_by_silence_or_vad(audio_path: str, min_sec: int = 3, max_sec: int = 15) -> list[dict]:
    _ensure_ffmpeg()
    duration = _probe_duration(audio_path)
    if duration <= 0.0:
        return []
    target = max(5.0, float(max_sec))
    result = []
    cursor = 0.0
    while cursor < duration:
        end = min(duration, cursor + target)
        if end - cursor < min_sec and result:
            result[-1]["end"] = duration
            break
        result.append({"start": cursor, "end": end, "text": ""})
        cursor = end
    return result


def export_clip(source_media_path: str, start, end, output_path, audio_only: bool = True) -> str:
    ffmpeg = _ensure_ffmpeg()
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", source_media_path]
    if audio_only:
        cmd += ["-vn", "-ac", "1", "-ar", "16000", output_path]
    else:
        cmd += ["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", output_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "").strip() or "ffmpeg export failed")
    return output_path
