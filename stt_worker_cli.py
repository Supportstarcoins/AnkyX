import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
import wave
import shutil
import threading


def _emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def write_log(path: str | None, line: str) -> None:
    try:
        if not path:
            return
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
            handle.flush()
    except Exception:
        pass


def _ffmpeg_path(payload: dict) -> str | None:
    path = payload.get("ffmpeg_path")
    if path and os.path.exists(path):
        return path
    return shutil.which("ffmpeg")


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / max(1, wf.getframerate())
    except Exception:
        return 0.0


def _run_ffmpeg(cmd: list[str], log_path: str | None) -> None:
    write_log(log_path, "ffmpeg_cmd=" + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or "ffmpeg failed"
        raise RuntimeError(msg)


def _extract_wav(payload: dict, log_path: str | None) -> tuple[str, bool]:
    wav_path = payload.get("wav_path")
    if wav_path and os.path.exists(wav_path):
        return wav_path, False
    video_path = payload.get("video_path")
    if not video_path:
        raise RuntimeError("payload missing wav_path/video_path")
    ffmpeg_path = _ffmpeg_path(payload)
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found")
    tmp_dir = payload.get("tmp_dir") or os.path.dirname(payload.get("payload_path", "")) or os.getcwd()
    os.makedirs(tmp_dir, exist_ok=True)
    output_path = os.path.join(tmp_dir, f"stt_worker_{uuid.uuid4().hex}.wav")
    cmd = [ffmpeg_path, "-y"]
    start = payload.get("start")
    end = payload.get("end")
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += [
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path,
    ]
    write_log(log_path, "stage=ffmpeg_extract_start")
    _run_ffmpeg(cmd, log_path)
    write_log(log_path, "stage=ffmpeg_extract_done")
    _emit({"type": "progress", "p": 20, "msg": "ffmpeg done"})
    return output_path, True


def _split_chunks(payload: dict, wav_path: str, log_path: str | None) -> tuple[list[tuple[str, float, float]], list[str]]:
    chunks = payload.get("chunks")
    if chunks:
        normalized = [(str(path), float(t0), float(t1)) for path, t0, t1 in chunks]
        return normalized, []
    chunk_sec = float(payload.get("chunk_sec") or 25.0)
    duration = _wav_duration(wav_path)
    if duration <= 0 or duration <= chunk_sec:
        return [(wav_path, 0.0, duration)], []
    ffmpeg_path = _ffmpeg_path(payload)
    if not ffmpeg_path:
        return [(wav_path, 0.0, duration)], []
    tmp_dir = payload.get("tmp_dir") or os.path.dirname(wav_path) or os.getcwd()
    os.makedirs(tmp_dir, exist_ok=True)
    pattern = os.path.join(tmp_dir, f"stt_chunk_{uuid.uuid4().hex}_%03d.wav")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        wav_path,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_sec),
        "-c",
        "copy",
        pattern,
    ]
    write_log(log_path, "stage=ffmpeg_split_start")
    _run_ffmpeg(cmd, log_path)
    write_log(log_path, "stage=ffmpeg_split_done")
    emitted = []
    chunks = []
    idx = 0
    while True:
        chunk_path = pattern.replace("%03d", f"{idx:03d}")
        if not os.path.exists(chunk_path):
            break
        t0 = idx * chunk_sec
        t1 = min(duration, t0 + chunk_sec)
        chunks.append((chunk_path, t0, t1))
        emitted.append(chunk_path)
        idx += 1
    return chunks or [(wav_path, 0.0, duration)], emitted


def _transcribe_chunks(payload: dict, wav_path: str, log_path: str | None) -> tuple[str, list[dict], float]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        payload.get("model_name") or "small",
        device=payload.get("device") or "cpu",
        compute_type=payload.get("compute_type") or "int8",
        cpu_threads=os.cpu_count() or 1,
    )
    lang = payload.get("language")
    chunks, temp_paths = _split_chunks(payload, wav_path, log_path)
    total_seconds = float(payload.get("total_seconds") or 0.0)
    if not total_seconds:
        total_seconds = max((t1 for _path, _t0, t1 in chunks), default=0.0)
    all_text_parts: list[str] = []
    all_segments: list[dict] = []
    last_processed = 0.0

    def _emit_progress(idx: int, t1: float, msg: str) -> None:
        nonlocal last_processed
        last_processed = max(last_processed, float(t1))
        percent = int(min(100, max(0, (last_processed / max(1.0, total_seconds)) * 100.0)))
        _emit(
            {
                "type": "progress",
                "p": percent,
                "msg": msg,
                "processed": last_processed,
                "total": total_seconds,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
        )

    try:
        for idx, (chunk_path, t0, t1) in enumerate(chunks, start=1):
            _emit_progress(idx, t0, f"chunk {idx}/{len(chunks)}")
            try:
                seg_iter, _info = model.transcribe(
                    chunk_path,
                    language=lang,
                    vad_filter=True,
                    beam_size=1,
                    condition_on_previous_text=True,
                )
            except Exception:
                seg_iter, _info = model.transcribe(
                    chunk_path,
                    language=lang,
                    vad_filter=False,
                    beam_size=1,
                    condition_on_previous_text=True,
                )
            for seg in seg_iter:
                text_part = (getattr(seg, "text", "") or "").strip()
                if text_part:
                    all_text_parts.append(text_part)
                seg_start = float(getattr(seg, "start", 0.0)) + float(t0)
                seg_end = float(getattr(seg, "end", seg_start)) + float(t0)
                all_segments.append({"start": seg_start, "end": seg_end, "text": text_part})
                _emit_progress(idx, seg_end, f"seg {idx}/{len(chunks)}")
            _emit_progress(idx, t1, f"chunk_done {idx}/{len(chunks)}")
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception:
                pass
    full_text = " ".join(all_text_parts).strip()
    voiced_duration = sum(
        max(0.0, float(seg["end"]) - float(seg["start"]))
        for seg in all_segments
        if seg.get("text")
    )
    return full_text, all_segments, voiced_duration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    _emit({"type": "started", "pid": os.getpid()})

    payload = {}
    log_path = None
    heartbeat_thread = None
    heartbeat_stop = threading.Event()
    try:
        with open(args.payload, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["payload_path"] = args.payload
        log_path = payload.get("log_path")
        write_log(log_path, f"stage=worker_enter pid={os.getpid()}")

        heartbeat_sec = float(payload.get("heartbeat_sec") or 1.0)

        def _heartbeat_loop() -> None:
            while not heartbeat_stop.wait(heartbeat_sec):
                _emit({"type": "heartbeat", "ts": time.time()})

        heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        wav_path, delete_wav = _extract_wav(payload, log_path)
        text, segments, voiced_duration = _transcribe_chunks(payload, wav_path, log_path)
        if delete_wav:
            try:
                os.remove(wav_path)
            except Exception:
                pass
        _emit({"type": "result", "text": text, "segments": segments, "voiced_duration": voiced_duration})
        _emit({"type": "done"})
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        write_log(log_path, f"worker_exception={exc} traceback={tb}")
        _emit({"type": "error", "err": str(exc), "tb": tb})
        return 1
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.5)


if __name__ == "__main__":
    raise SystemExit(main())
