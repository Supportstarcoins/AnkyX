"""Utilities for working with video clips and optional playback support."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

import tkinter as tk
from tkinter import ttk

try:
    import vlc  # type: ignore

    VLC_AVAILABLE = True
except Exception:
    vlc = None  # type: ignore
    VLC_AVAILABLE = False


def is_vlc_available() -> bool:
    """Return True if python-vlc is importable."""

    return VLC_AVAILABLE


def find_ffmpeg() -> Optional[str]:
    """Search for ffmpeg executable in PATH and alongside the app."""

    local_candidates = [
        os.path.join(os.getcwd(), "ffmpeg.exe"),
        os.path.join(os.getcwd(), "ffmpeg"),
    ]

    for candidate in local_candidates:
        if os.path.isfile(candidate):
            return candidate

    return shutil.which("ffmpeg")


def parse_hms(value: str) -> Optional[int]:
    """Parse HH:MM:SS into seconds."""

    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        return None

    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None

    while len(parts) < 3:
        parts.insert(0, 0)

    hours, minutes, seconds = parts
    if minutes >= 60 or seconds >= 60 or hours < 0 or minutes < 0 or seconds < 0:
        return None

    return hours * 3600 + minutes * 60 + seconds


def format_hms(total_seconds: int) -> str:
    """Format seconds into HH:MM:SS."""

    total_seconds = max(0, int(total_seconds))
    td = timedelta(seconds=total_seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours += td.days * 24
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def cut_video_clip(
    video_path: str,
    start_hms: str,
    end_hms: str,
    media_dir: str = "media",
) -> Tuple[bool, str]:
    """
    Cut a clip from *video_path* using ffmpeg.

    Returns a tuple (success, message_or_path).
    """

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        return False, "FFmpeg не найден. Положите ffmpeg.exe рядом с программой или добавьте его в PATH."

    start_sec = parse_hms(start_hms)
    end_sec = parse_hms(end_hms)

    if start_sec is None or end_sec is None:
        return False, "Неверный формат времени. Используйте HH:MM:SS."
    if end_sec <= start_sec:
        return False, "Время окончания должно быть больше времени начала."

    os.makedirs(media_dir, exist_ok=True)
    safe_start = start_hms.replace(":", "-")
    safe_end = end_hms.replace(":", "-")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(media_dir, f"clip_{ts}_{safe_start}_{safe_end}.mp4")

    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        start_hms,
        "-to",
        end_hms,
        "-i",
        video_path,
        "-c",
        "copy",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Неизвестная ошибка ffmpeg."
            return False, f"Не удалось вырезать клип: {error_msg}"
        return True, output_path
    except FileNotFoundError:
        return False, "FFmpeg не найден."
    except Exception as exc:
        return False, f"Ошибка ffmpeg: {exc}"


def open_in_external_player(path: str) -> None:
    """Open *path* in the system video player."""

    if not path:
        return

    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
        return

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.Popen([opener, path])
    except Exception:
        pass


class VlcPlayerWidget:
    """Tkinter wrapper for a simple VLC video player."""

    def __init__(self, parent, video_path: str, width: int = 360, height: int = 210):
        if not VLC_AVAILABLE:
            raise RuntimeError("python-vlc недоступен")

        self.video_path = video_path
        self.frame = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))

        controls = ttk.Frame(self.frame)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(controls, text="▶ Play", command=self.play).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="⏹ Stop", command=self.stop).pack(fill=tk.X, pady=2)

        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

    def _apply_handle(self) -> None:
        handle = self.canvas.winfo_id()
        if sys.platform.startswith("linux"):
            self.player.set_xwindow(handle)
        elif sys.platform == "darwin":
            self.player.set_nsobject(handle)
        else:
            self.player.set_hwnd(handle)

    def play(self) -> None:
        if not os.path.exists(self.video_path):
            return
        media = self.instance.media_new(self.video_path)
        self.player.set_media(media)
        self._apply_handle()
        self.player.play()

    def stop(self) -> None:
        self.player.stop()

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

