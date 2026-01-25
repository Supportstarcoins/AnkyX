import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid


def _find_ffmpeg() -> str:
    local_candidates = [
        os.path.join(os.getcwd(), "ffmpeg.exe"),
        os.path.join(os.getcwd(), "ffmpeg"),
    ]
    for candidate in local_candidates:
        if os.path.isfile(candidate):
            return candidate
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found")
    return path


def _run_ffmpeg(cmd: list[str], timeout_sec: int | None = None) -> None:
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg превысил лимит времени.")
    if proc.returncode != 0:
        msg = (proc.stderr or "").strip() or "ffmpeg failed"
        raise RuntimeError(msg)


def _extract_wav(
    input_path: str,
    *,
    start: float | None,
    end: float | None,
) -> str:
    ffmpeg_path = _find_ffmpeg()
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"stt_engine_{uuid.uuid4().hex}.wav")
    start_sec = float(start or 0.0)
    duration_sec = None
    if end is not None:
        duration_sec = max(0.1, float(end) - start_sec)
    timeout_sec = 120
    if duration_sec is not None:
        timeout_sec = min(900, max(timeout_sec, int(duration_sec * 8 + 30)))
    cmd = [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", str(start_sec)]
    if duration_sec is not None:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += [
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path,
    ]
    _run_ffmpeg(cmd, timeout_sec=timeout_sec)
    return output_path


def _transcribe_segments(wav_path: str, lang: str | None, model_name: str) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name or "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=os.cpu_count() or 1,
    )
    try:
        seg_iter, _info = model.transcribe(
            wav_path,
            language=lang or None,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=True,
        )
    except Exception:
        seg_iter, _info = model.transcribe(
            wav_path,
            language=lang or None,
            vad_filter=False,
            beam_size=1,
            condition_on_previous_text=True,
        )
    segments: list[dict] = []
    for seg in seg_iter:
        text = (getattr(seg, "text", "") or "").strip()
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", start))
        segments.append({"start": start, "end": end, "text": text})
    return segments


def _build_phrases(segments: list[dict], gap_sec: float = 0.8) -> list[dict]:
    phrases: list[dict] = []
    current: dict | None = None
    for seg in sorted(segments, key=lambda item: float(item.get("start", 0.0))):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if current is None:
            current = {"start": start, "end": end, "text": text}
            continue
        if start - float(current["end"]) > gap_sec:
            phrases.append(current)
            current = {"start": start, "end": end, "text": text}
        else:
            current["end"] = max(float(current["end"]), end)
            current["text"] = f"{current['text']} {text}".strip()
    if current is not None:
        phrases.append(current)
    return phrases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--model", default="small")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    wav_path = None
    try:
        wav_path = _extract_wav(args.input, start=args.start, end=args.end)
        segments = _transcribe_segments(wav_path, args.lang, args.model)
        phrases = _build_phrases(segments)
        offset = float(args.start or 0.0)
        if offset:
            for phrase in phrases:
                phrase["start"] = float(phrase.get("start", 0.0)) + offset
                phrase["end"] = float(phrase.get("end", phrase["start"])) + offset
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(phrases, handle, ensure_ascii=False, indent=2)
        return 0
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
