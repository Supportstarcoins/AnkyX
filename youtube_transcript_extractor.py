from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


class YouTubeTranscriptExtractor:
    DEFAULT_LANGUAGES = ["ru", "en", "de"]

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        host = (urlparse((url or "").strip()).netloc or "").lower()
        return any(x in host for x in ("youtube.com", "youtu.be", "m.youtube.com"))

    @staticmethod
    def extract_video_id(url: str) -> str | None:
        parsed = urlparse((url or "").strip())
        host = (parsed.netloc or "").lower()
        if "youtu.be" in host:
            return (parsed.path.strip("/") or None)
        if "youtube.com" not in host and "m.youtube.com" not in host:
            return None
        if parsed.path.startswith("/watch"):
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
        return None

    @classmethod
    def fetch_transcript(cls, url: str, languages: list[str] | None = None) -> dict:
        video_id = cls.extract_video_id(url)
        thumb_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg" if video_id else ""
        result = {
            "ok": False,
            "source_type": "youtube",
            "video_id": video_id or "",
            "url": url,
            "title": f"YouTube video {video_id}" if video_id else "YouTube video",
            "language": "",
            "text": "",
            "segments": [],
            "images": [],
            "status": "error",
            "error": "",
        }
        if not video_id:
            result["error"] = "Не удалось извлечь video_id из YouTube URL"
            return result
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except Exception:
            result["error"] = "Для YouTube-субтитров установите youtube-transcript-api"
            if thumb_url:
                result["images"] = [
                    {
                        "url": thumb_url,
                        "local_path": "",
                        "alt": "YouTube thumbnail",
                        "caption": "YouTube thumbnail",
                        "context_text": "",
                        "position": 0,
                        "width": 0,
                        "height": 0,
                        "source_type": "youtube_thumbnail",
                        "page_url": url,
                    }
                ]
            return result
        try:
            from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
            no_transcript_errors = (NoTranscriptFound, TranscriptsDisabled)
        except Exception:
            no_transcript_errors = ()

        languages = languages or list(cls.DEFAULT_LANGUAGES)

        def _norm_text(value: str) -> str:
            return re.sub(r"\s+", " ", str(value or "")).strip()

        def _row_to_segment(row) -> dict | None:
            if isinstance(row, dict):
                text = _norm_text(row.get("text"))
                start = row.get("start", 0.0)
                duration = row.get("duration", 0.0)
            else:
                text = _norm_text(getattr(row, "text", ""))
                start = getattr(row, "start", 0.0)
                duration = getattr(row, "duration", 0.0)
            if not text:
                return None
            return {
                "start": float(start or 0.0),
                "duration": float(duration or 0.0),
                "text": text,
            }

        def _segments_from_rows(rows) -> list[dict]:
            out: list[dict] = []
            if rows is None:
                return out
            for row in rows:
                seg = _row_to_segment(row)
                if seg:
                    out.append(seg)
            return out

        def _dedupe_segments(segments: list[dict]) -> list[dict]:
            deduped: list[dict] = []
            prev_key = ""
            for seg in segments:
                key = re.sub(r"\W+", "", seg["text"].lower())
                if not key or key == prev_key:
                    continue
                prev_key = key
                if key in {"[music]", "music", "аплодисменты", "смех"}:
                    continue
                deduped.append(seg)
            return deduped

        def _select_from_list(transcript_list):
            transcript = None
            language_used = ""
            for lang in languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    language_used = lang
                    break
                except Exception:
                    continue
            if transcript is None:
                for lang in languages:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang])
                        language_used = lang
                        break
                    except Exception:
                        continue
            if transcript is None:
                return [], ""
            fetched = transcript.fetch()
            return _segments_from_rows(fetched), language_used

        try:
            segments: list[dict] = []
            language_used = ""
            api = None
            try:
                api = YouTubeTranscriptApi()
            except Exception:
                api = None

            # Новый API: экземпляр + fetch(video_id, languages=[...])
            if api is not None and hasattr(api, "fetch"):
                try:
                    fetched = api.fetch(video_id, languages=languages)
                    language_used = str(getattr(fetched, "language_code", "") or "")
                    raw_rows = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
                    segments = _segments_from_rows(raw_rows)
                except Exception:
                    segments = []

            # Новый API: экземпляр + list(video_id)
            if not segments and api is not None and hasattr(api, "list"):
                try:
                    transcript_list = api.list(video_id)
                    segments, language_used = _select_from_list(transcript_list)
                except Exception:
                    segments = []

            # Старый API: classmethod list_transcripts(video_id)
            if not segments and hasattr(YouTubeTranscriptApi, "list_transcripts"):
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                segments, language_used = _select_from_list(transcript_list)

            # Старый API: classmethod get_transcript(video_id, languages=[...])
            if not segments and hasattr(YouTubeTranscriptApi, "get_transcript"):
                try:
                    fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                except TypeError:
                    fetched = YouTubeTranscriptApi.get_transcript(video_id)
                segments = _segments_from_rows(fetched)
                if languages:
                    language_used = language_used or str(languages[0])

            deduped = _dedupe_segments(segments)
            if not deduped:
                result["status"] = "no_transcript"
                result["error"] = "Субтитры не найдены. STT fallback пока не подключён."
                if thumb_url:
                    result["images"] = [
                        {
                            "url": thumb_url,
                            "local_path": "",
                            "alt": "YouTube thumbnail",
                            "caption": "YouTube thumbnail",
                            "context_text": "",
                            "position": 0,
                            "width": 0,
                            "height": 0,
                            "source_type": "youtube_thumbnail",
                            "page_url": url,
                        }
                    ]
                return result

            result["ok"] = True
            result["language"] = language_used
            result["segments"] = deduped
            result["text"] = " ".join(s["text"] for s in deduped).strip()
            if thumb_url:
                result["images"] = [
                    {
                        "url": thumb_url,
                        "local_path": "",
                        "alt": "YouTube thumbnail",
                        "caption": "YouTube thumbnail",
                        "context_text": "",
                        "position": 0,
                        "width": 0,
                        "height": 0,
                        "source_type": "youtube_thumbnail",
                        "page_url": url,
                    }
                ]
            result["status"] = "ok"
            result["error"] = ""
            return result
        except no_transcript_errors:
            result["status"] = "no_transcript"
            result["error"] = "Субтитры не найдены. STT fallback пока не подключён."
            if thumb_url:
                result["images"] = [
                    {
                        "url": thumb_url,
                        "local_path": "",
                        "alt": "YouTube thumbnail",
                        "caption": "YouTube thumbnail",
                        "context_text": "",
                        "position": 0,
                        "width": 0,
                        "height": 0,
                        "source_type": "youtube_thumbnail",
                        "page_url": url,
                    }
                ]
            return result
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            return result

    @staticmethod
    def transcribe_with_stt_fallback(*_args, **_kwargs):
        raise RuntimeError("STT fallback пока не подключён")
