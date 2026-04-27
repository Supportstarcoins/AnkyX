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
            from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
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

        languages = languages or list(cls.DEFAULT_LANGUAGES)
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
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
            fetched = transcript.fetch()
            segments = []
            for row in fetched:
                text = re.sub(r"\s+", " ", (row.get("text") or "")).strip()
                if not text:
                    continue
                segments.append(
                    {
                        "start": float(row.get("start") or 0.0),
                        "duration": float(row.get("duration") or 0.0),
                        "text": text,
                    }
                )
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
            return result
        except (NoTranscriptFound, TranscriptsDisabled):
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
