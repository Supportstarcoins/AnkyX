import os
import threading
import time
import traceback


_STT_WORKER_MODEL = None
_STT_WORKER_MODEL_KEY = None


def log_write(path, line):
    try:
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
            f.flush()
    except Exception:
        pass


def _get_worker_whisper_model(model_name: str, device: str, compute_type: str):
    global _STT_WORKER_MODEL, _STT_WORKER_MODEL_KEY
    key = (model_name, device, compute_type)
    if _STT_WORKER_MODEL is None or _STT_WORKER_MODEL_KEY != key:
        from faster_whisper import WhisperModel

        _STT_WORKER_MODEL = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() or 1,
        )
        _STT_WORKER_MODEL_KEY = key
    return _STT_WORKER_MODEL


def whisper_worker(payload: dict, out_q) -> None:
    log_path = payload.get("log_path")
    log_write(log_path, "stage=worker_enter")
    out_q.put({"type": "started", "ts": time.time()})

    heartbeat_sec = float(payload.get("heartbeat_sec", 1.0))
    stop_event = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop_event.wait(heartbeat_sec):
            out_q.put({"type": "heartbeat", "ts": time.time()})

    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    def _transcribe_chunks(vad_filter: bool):
        model = _get_worker_whisper_model(
            payload["model_name"],
            device=payload.get("device", "cpu"),
            compute_type=payload.get("compute_type", "int8"),
        )
        log_write(log_path, "stage=whisper_start")
        lang = payload.get("language")
        chunks = payload.get("chunks") or []
        total_seconds = float(payload.get("total_seconds") or 0.0)
        all_text_parts: list[str] = []
        all_segments: list[dict] = []
        last_processed = 0.0
        for idx, (chunk_path, t0, t1) in enumerate(chunks, start=1):
            out_q.put(
                {
                    "type": "progress",
                    "processed": last_processed,
                    "total": total_seconds,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "msg": f"chunk {idx}/{len(chunks)}",
                }
            )
            try:
                seg_iter, _info = model.transcribe(
                    chunk_path,
                    language=lang,
                    vad_filter=vad_filter,
                    beam_size=1,
                    condition_on_previous_text=True,
                )
            except Exception as exc:
                if vad_filter:
                    seg_iter, _info = model.transcribe(
                        chunk_path,
                        language=lang,
                        vad_filter=False,
                        beam_size=1,
                        condition_on_previous_text=True,
                    )
                else:
                    raise exc
            for seg in seg_iter:
                text_part = (getattr(seg, "text", "") or "").strip()
                if text_part:
                    all_text_parts.append(text_part)
                seg_start = float(getattr(seg, "start", 0.0)) + float(t0)
                seg_end = float(getattr(seg, "end", seg_start)) + float(t0)
                all_segments.append({"start": seg_start, "end": seg_end, "text": text_part})
                last_processed = max(last_processed, seg_end)
                out_q.put(
                    {
                        "type": "progress",
                        "processed": last_processed,
                        "total": total_seconds,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "msg": f"seg {idx}/{len(chunks)}",
                    }
                )
            out_q.put(
                {
                    "type": "progress",
                    "processed": max(last_processed, float(t1)),
                    "total": total_seconds,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "msg": f"chunk_done {idx}/{len(chunks)}",
                }
            )
        full_text = " ".join(all_text_parts).strip()
        voiced_duration = sum(
            max(0.0, float(seg["end"]) - float(seg["start"]))
            for seg in all_segments
            if seg.get("text")
        )
        return full_text, all_segments, voiced_duration

    try:
        text, segments, voiced_duration = _transcribe_chunks(vad_filter=True)
        if len(text) < 50 or len(segments) < 3:
            text, segments, voiced_duration = _transcribe_chunks(vad_filter=False)
        out_q.put(
            {
                "type": "result",
                "text": text,
                "segments": segments,
                "voiced_duration": voiced_duration,
            }
        )
    except Exception as exc:
        log_write(
            log_path,
            f"EXCEPTION={type(exc).__name__}: {exc} traceback={traceback.format_exc()}",
        )
        out_q.put({"type": "error", "err": str(exc), "tb": traceback.format_exc()})
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=0.5)
