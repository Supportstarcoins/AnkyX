from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from digital_hearing_adapter import DigitalHearingAdapter
from media_clip_segmenter import export_clip, split_audio_by_silence_or_vad, split_by_transcript_segments
from youtube_transcript_extractor import YouTubeTranscriptExtractor


class YouTubeMediaPipeline:
    def __init__(self, media_root: str = "media") -> None:
        self.media_root = media_root
        self.hearing = DigitalHearingAdapter()

    def process_url(self, url: str, options: dict | None = None) -> dict:
        opts = dict(options or {})
        progress_cb = opts.get("progress_cb") if callable(opts.get("progress_cb")) else None

        def progress(msg: str) -> None:
            if progress_cb:
                try:
                    progress_cb(msg)
                except Exception:
                    pass

        result = {
            "ok": False,
            "source_type": "youtube",
            "url": url,
            "video_id": "",
            "title": "",
            "language": opts.get("language") or "",
            "media_path": "",
            "audio_path": "",
            "thumbnail_path": "",
            "segments": [],
            "errors": [],
        }

        if not YouTubeTranscriptExtractor.is_youtube_url(url):
            result["errors"].append("Это не похоже на YouTube URL")
            return result
        video_id = YouTubeTranscriptExtractor.extract_video_id(url)
        if not video_id:
            result["errors"].append("Не удалось извлечь video_id из YouTube URL")
            return result
        result["video_id"] = video_id
        result["title"] = f"YouTube video {video_id}"
        result["thumbnail_path"] = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        langs = [opts.get("language")] if opts.get("language") and opts.get("language") != "auto" else None
        yt = YouTubeTranscriptExtractor.fetch_transcript(url, languages=langs)
        transcript_segments = list(yt.get("segments") or [])
        if yt.get("language"):
            result["language"] = yt.get("language")
        if yt.get("title"):
            result["title"] = yt.get("title")

        force_stt = bool(opts.get("force_stt"))
        if transcript_segments and not force_stt:
            progress("Использую субтитры YouTube...")
            split = split_by_transcript_segments(transcript_segments, min_sec=int(opts.get("min_sec") or 3), max_sec=int(opts.get("max_sec") or 15))
            result["segments"] = [
                {
                    "index": i,
                    "start": float(seg.get("start") or 0.0),
                    "end": float(seg.get("end") or 0.0),
                    "text": str(seg.get("text") or "").strip(),
                    "audio_path": "",
                    "video_path": "",
                    "thumbnail_path": result["thumbnail_path"],
                    "confidence": 0.95,
                }
                for i, seg in enumerate(split, start=1)
            ]
            if result["segments"]:
                result["ok"] = True
                return result

        progress("Загружаю аудио...")
        media_dir = Path(self.media_root) / "youtube" / video_id
        media_dir.mkdir(parents=True, exist_ok=True)

        ytdlp = shutil.which("yt-dlp")
        if not ytdlp:
            if transcript_segments:
                result["ok"] = True
                result["segments"] = [
                    {
                        "index": i,
                        "start": float(seg.get("start") or 0.0),
                        "end": float((seg.get("start") or 0.0) + (seg.get("duration") or 0.0)),
                        "text": str(seg.get("text") or "").strip(),
                        "audio_path": "",
                        "video_path": "",
                        "thumbnail_path": result["thumbnail_path"],
                        "confidence": 0.95,
                    }
                    for i, seg in enumerate(transcript_segments, start=1)
                ]
                result["errors"].append("Для загрузки YouTube media установите yt-dlp.")
                return result
            result["errors"].append("Для загрузки YouTube media установите yt-dlp.")
            return result

        download_video = bool(opts.get("download_video"))
        audio_only = bool(opts.get("audio_only", True))
        audio_path = str(media_dir / "source_audio.m4a")
        video_path = str(media_dir / "source_video.mp4") if download_video and not audio_only else ""

        audio_cmd = [
            ytdlp,
            "-f",
            "bestaudio[ext=m4a]/bestaudio",
            "-o",
            audio_path,
            url,
        ]
        if self._run(audio_cmd) != 0:
            result["errors"].append("Не удалось загрузить аудио из YouTube (видео может быть недоступно).")
            return result

        result["audio_path"] = audio_path if os.path.exists(audio_path) else ""
        result["media_path"] = result["audio_path"]

        if video_path:
            video_cmd = [ytdlp, "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]", "-o", video_path, url]
            if self._run(video_cmd) == 0 and os.path.exists(video_path):
                result["media_path"] = video_path

        progress("Нарезаю фрагменты...")
        try:
            if transcript_segments:
                chunks = split_by_transcript_segments(transcript_segments, min_sec=int(opts.get("min_sec") or 3), max_sec=int(opts.get("max_sec") or 15))
            else:
                chunks = split_audio_by_silence_or_vad(audio_path, min_sec=int(opts.get("min_sec") or 3), max_sec=int(opts.get("max_sec") or 15))
        except Exception as exc:
            result["errors"].append(str(exc))
            return result

        segments = []
        lang = opts.get("language") if opts.get("language") not in {None, "", "auto"} else result.get("language")
        for idx, seg in enumerate(chunks, start=1):
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
            clip_audio = str(media_dir / f"clip_{idx:04d}.wav")
            clip_video = str(media_dir / f"clip_{idx:04d}.mp4") if result.get("media_path", "").endswith(".mp4") else ""
            try:
                export_clip(audio_path, start, end, clip_audio, audio_only=True)
                if clip_video:
                    export_clip(result.get("media_path") or audio_path, start, end, clip_video, audio_only=False)
            except Exception as exc:
                result["errors"].append(str(exc))
                continue

            text = str(seg.get("text") or "").strip()
            conf = 0.95 if text else 0.0
            if force_stt or not text:
                progress(f"Распознаю речь {idx}/{len(chunks)}...")
                heard = self.hearing.transcribe(clip_audio, language=lang)
                if heard.get("ok"):
                    text = str(heard.get("text") or "").strip()
                    conf = float(heard.get("confidence") or 0.0)
                    if heard.get("language"):
                        result["language"] = heard.get("language")
                elif heard.get("error"):
                    result["errors"].append(str(heard.get("error")))
            segments.append(
                {
                    "index": idx,
                    "start": start,
                    "end": end,
                    "text": text,
                    "audio_path": clip_audio,
                    "video_path": clip_video,
                    "thumbnail_path": result["thumbnail_path"],
                    "confidence": conf,
                }
            )

        result["segments"] = segments
        result["ok"] = bool(segments)
        if not segments and not result["errors"]:
            result["errors"].append("Не удалось подготовить сегменты из видео.")
        return result

    @staticmethod
    def _run(cmd: list[str]) -> int:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return int(proc.returncode)
