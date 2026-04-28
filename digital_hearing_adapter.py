from __future__ import annotations

import os


class DigitalHearingAdapter:
    def __init__(self, default_model: str = "small") -> None:
        self.default_model = default_model

    def transcribe(self, audio_path: str, language: str | None = None) -> dict:
        if not audio_path or not os.path.exists(audio_path):
            return {
                "ok": False,
                "text": "",
                "language": language or "",
                "segments": [],
                "confidence": 0.0,
                "error": "Аудиофайл не найден для распознавания.",
            }

        for runner in (self._transcribe_via_existing_engine, self._transcribe_via_faster_whisper, self._transcribe_via_whisper):
            try:
                payload = runner(audio_path, language)
                if payload.get("ok"):
                    return payload
            except Exception:
                continue

        return {
            "ok": False,
            "text": "",
            "language": language or "",
            "segments": [],
            "confidence": 0.0,
            "error": "Нет движка STT: установите faster-whisper или whisper.",
        }

    def _transcribe_via_existing_engine(self, audio_path: str, language: str | None) -> dict:
        import stt_engine

        wav_path = None
        try:
            wav_path = stt_engine._extract_wav(audio_path, start=None, end=None)  # type: ignore[attr-defined]
            mapped = None if (language or "").lower() in {"", "auto"} else language
            segments = stt_engine._transcribe_segments(wav_path, mapped, self.default_model)  # type: ignore[attr-defined]
            phrases = stt_engine._build_phrases(segments)  # type: ignore[attr-defined]
            text = " ".join((p.get("text") or "").strip() for p in phrases if (p.get("text") or "").strip()).strip()
            confidence = 0.75 if text else 0.0
            return {
                "ok": bool(text),
                "text": text,
                "language": language or "",
                "segments": phrases,
                "confidence": confidence,
                "error": "" if text else "Не удалось распознать речь.",
            }
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def _transcribe_via_faster_whisper(self, audio_path: str, language: str | None) -> dict:
        from faster_whisper import WhisperModel

        model = WhisperModel(self.default_model, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 1)
        selected_lang = None if (language or "").lower() in {"", "auto"} else language
        seg_iter, info = model.transcribe(audio_path, language=selected_lang, vad_filter=True, beam_size=1)
        segments = []
        for seg in seg_iter:
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": float(getattr(seg, "start", 0.0)),
                    "end": float(getattr(seg, "end", 0.0)),
                    "text": text,
                }
            )
        text = " ".join(s["text"] for s in segments).strip()
        return {
            "ok": bool(text),
            "text": text,
            "language": getattr(info, "language", None) or language or "",
            "segments": segments,
            "confidence": 0.7,
            "error": "" if text else "Не удалось распознать речь.",
        }

    def _transcribe_via_whisper(self, audio_path: str, language: str | None) -> dict:
        import whisper

        model = whisper.load_model("base")
        selected_lang = None if (language or "").lower() in {"", "auto"} else language
        payload = model.transcribe(audio_path, language=selected_lang)
        segments = []
        for seg in payload.get("segments") or []:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            segments.append({"start": float(seg.get("start") or 0.0), "end": float(seg.get("end") or 0.0), "text": text})
        text = str(payload.get("text") or "").strip() or " ".join(s["text"] for s in segments).strip()
        return {
            "ok": bool(text),
            "text": text,
            "language": payload.get("language") or language or "",
            "segments": segments,
            "confidence": 0.65,
            "error": "" if text else "Не удалось распознать речь.",
        }
