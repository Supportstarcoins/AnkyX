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
    """Tkinter wrapper for a VLC video player with controls."""

    SPEED_OPTIONS = ["0.25", "0.5", "0.75", "1", "1.25", "1.5", "1.75", "2"]

    def __init__(
        self,
        parent,
        video_path: str,
        width: int = 360,
        height: int = 210,
        on_state_change=None,
    ):
        if not VLC_AVAILABLE:
            raise RuntimeError("python-vlc недоступен")

        self.video_path = video_path
        self.on_state_change = on_state_change
        self.media_key: str | None = None
        self._after_id = None
        self._seeking = False
        self._duration_ms = 0
        self._volume = 70.0
        self._rate = 1.0

        self.frame = tk.Frame(parent, bg="white")
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, padx=6, pady=6)

        control_row = ttk.Frame(self.frame)
        control_row.pack(fill=tk.X, pady=2)
        self.play_btn = ttk.Button(control_row, text="▶ Play", command=self.play)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        self.pause_btn = ttk.Button(control_row, text="⏸ Pause", command=self.pause)
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(control_row, text="⏹ Stop", command=self.stop)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.volume_btn = ttk.Button(control_row, text="🔊 Громкость", command=self._toggle_volume_panel)
        self.volume_btn.pack(side=tk.LEFT, padx=6)
        ttk.Label(control_row, text="Скорость").pack(side=tk.LEFT, padx=(10, 5))
        self.rate_var = tk.StringVar(value="1")
        self.rate_combo = ttk.Combobox(
            control_row,
            values=self.SPEED_OPTIONS,
            textvariable=self.rate_var,
            state="readonly",
            width=6,
        )
        self.rate_combo.pack(side=tk.LEFT)
        self.rate_combo.bind("<<ComboboxSelected>>", lambda _e: self.set_rate(float(self.rate_var.get())))

        self.volume_panel = ttk.Frame(self.frame)
        self.volume_panel.pack(fill=tk.X, pady=2)
        self.volume_panel.pack_forget()
        ttk.Label(self.volume_panel, text="Громкость").pack(side=tk.LEFT, padx=(0, 5))
        self.volume_var = tk.DoubleVar(value=70)
        self.volume_scale = ttk.Scale(
            self.volume_panel,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.volume_var,
            command=lambda _v: self.set_volume(self.volume_var.get()),
        )
        self.volume_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        seek_frame = ttk.Frame(self.frame)
        seek_frame.pack(fill=tk.X, pady=2)
        self.time_label = ttk.Label(seek_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=(0, 5))
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_scale = ttk.Scale(
            seek_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.seek_var,
            command=lambda _v: self._on_seek(),
        )
        self.seek_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.seek_scale.bind("<ButtonPress-1>", lambda _e: self._start_seeking())
        self.seek_scale.bind("<ButtonRelease-1>", lambda _e: self._end_seeking())

        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

    def _apply_handle(self) -> None:
        try:
            self.canvas.update_idletasks()
        except Exception:
            pass
        handle = self.canvas.winfo_id()
        if not handle:
            print("[VLC] Не удалось получить window handle для видео.")
            return
        try:
            if sys.platform.startswith("linux"):
                self.player.set_xwindow(handle)
            elif sys.platform == "darwin":
                self.player.set_nsobject(handle)
            else:
                self.player.set_hwnd(handle)
        except Exception as exc:
            print(f"[VLC] Ошибка embed видео: {exc}")

    def ensure_embedded(self) -> bool:
        try:
            self.canvas.update_idletasks()
        except Exception:
            pass
        handle = self.canvas.winfo_id()
        if not handle:
            print("[VLC] Не удалось получить window handle для видео.")
            return False
        try:
            if sys.platform.startswith("linux"):
                self.player.set_xwindow(handle)
            elif sys.platform == "darwin":
                self.player.set_nsobject(handle)
            else:
                self.player.set_hwnd(handle)
            return True
        except Exception as exc:
            print(f"[VLC] Ошибка embed видео: {exc}")
            return False

    def play(self) -> None:
        if not os.path.exists(self.video_path):
            return
        media = self.instance.media_new(self.video_path)
        self.player.set_media(media)
        self._apply_handle()
        self.player.play()
        self._schedule_progress_update()

    def pause(self) -> None:
        self.player.pause()
        self._emit_state()

    def stop(self) -> None:
        self.player.stop()
        self._cancel_after()
        self.seek_var.set(0)
        self._update_time_label(0, self._duration_ms)
        self._emit_state()

    def set_volume(self, value: float) -> None:
        self._volume = float(value)
        try:
            self.player.audio_set_volume(int(self._volume))
        except Exception:
            pass
        self._emit_state()

    def set_rate(self, value: float) -> None:
        self._rate = float(value)
        try:
            self.player.set_rate(float(self._rate))
        except Exception:
            pass
        self._emit_state()

    def set_media_key(self, media_key: str | None) -> None:
        self.media_key = media_key

    def get_state(self) -> dict:
        pos_ms = int(self.seek_var.get() or 0)
        try:
            pos_ms = int(self.player.get_time())
        except Exception:
            pass
        return {
            "pos_ms": pos_ms,
            "volume": float(self._volume),
            "speed": float(self._rate),
        }

    def apply_state(self, state: dict | None) -> None:
        if not state:
            return
        try:
            volume = float(state.get("volume", self._volume))
            speed = float(state.get("speed", self._rate))
            pos_ms = int(state.get("pos_ms", 0))
        except Exception:
            return
        self.volume_var.set(volume)
        self.set_volume(volume)
        self.rate_var.set(str(speed))
        self.set_rate(speed)
        if pos_ms > 0:
            try:
                self.player.set_time(pos_ms)
                self.seek_var.set(pos_ms)
            except Exception:
                pass

    def _toggle_volume_panel(self):
        if self.volume_panel.winfo_ismapped():
            self.volume_panel.pack_forget()
        else:
            self.volume_panel.pack(fill=tk.X, pady=2)

    def _start_seeking(self):
        self._seeking = True

    def _end_seeking(self):
        self._seeking = False
        self._on_seek()

    def _on_seek(self):
        if self._duration_ms <= 0:
            return
        position_ms = min(max(self.seek_var.get(), 0), self._duration_ms)
        self.player.set_time(int(position_ms))
        self._emit_state()

    def _schedule_progress_update(self):
        self._cancel_after()
        self._after_id = self.frame.after(200, self._update_progress)

    def _update_time_label(self, current_ms: int, total_ms: int):
        def fmt(ms):
            seconds = max(ms, 0) // 1000
            return f"{seconds//60:02d}:{seconds%60:02d}"
        self.time_label.config(text=f"{fmt(current_ms)} / {fmt(total_ms)}")

    def _update_progress(self):
        try:
            total = self.player.get_length()
            if total and total > 0:
                self._duration_ms = total
                self.seek_scale.config(to=total)
            current = self.player.get_time()
            self._update_time_label(current, self._duration_ms)
            if not self._seeking:
                self.seek_var.set(current)
            state = self.player.get_state()
            if state in (vlc.State.Ended, vlc.State.Stopped):
                self._cancel_after()
                return
            self._after_id = self.frame.after(200, self._update_progress)
            self._emit_state()
        except Exception:
            self._cancel_after()

    def _cancel_after(self):
        if self._after_id:
            try:
                self.frame.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _emit_state(self):
        if not self.on_state_change:
            return
        try:
            self.on_state_change(self.media_key, self.get_state())
        except Exception:
            pass

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)
