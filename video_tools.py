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


def parse_hms(value: str) -> Optional[float]:
    """Parse HH:MM:SS (seconds may be fractional) into seconds."""

    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        return None

    try:
        if len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = float(parts[0])
        elif len(parts) == 2:
            hours = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        else:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
    except ValueError:
        return None

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


def get_clip_log_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "clip_last.log")


def _write_clip_log(lines: list[str], *, mode: str = "a") -> None:
    log_path = get_clip_log_path()
    with open(log_path, mode, encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
            f.flush()


def _run_ffmpeg_attempt(cmd: list[str], *, label: str) -> subprocess.CompletedProcess[str] | None:
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        _write_clip_log([f"attempt={label}", "error=ffmpeg_not_found"], mode="a")
        return None
    except Exception as exc:
        _write_clip_log([f"attempt={label}", f"exception={exc}"], mode="a")
        return None

    stderr = (result.stderr or "").strip()
    _write_clip_log(
        [
            f"attempt={label}",
            "cmd=" + " ".join(cmd),
            f"returncode={result.returncode}",
            f"stderr={stderr}",
        ],
        mode="a",
    )
    return result


def cut_video_clip_with_fallback(
    video_path: str,
    start_time: str,
    end_time: str,
    output_path: str,
) -> tuple[bool, str, str]:
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        return False, "", "FFmpeg не найден. Положите ffmpeg.exe рядом с программой или добавьте его в PATH."

    _write_clip_log(
        [
            f"timestamp={datetime.now().isoformat()}",
            f"video_path={video_path}",
            f"output_path={output_path}",
            f"start={start_time}",
            f"end={end_time}",
        ],
        mode="w",
    )

    base_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        start_time,
        "-to",
        end_time,
        "-i",
        video_path,
    ]

    copy_cmd = base_cmd + [
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = _run_ffmpeg_attempt(copy_cmd, label="stream_copy")
    if result is not None and result.returncode == 0:
        return True, "copy", ""

    reencode_cmd = base_cmd + [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = _run_ffmpeg_attempt(reencode_cmd, label="libx264")
    if result is not None and result.returncode == 0:
        return True, "libx264", ""

    nvenc_cmd = base_cmd + [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = _run_ffmpeg_attempt(nvenc_cmd, label="h264_nvenc")
    if result is not None and result.returncode == 0:
        return True, "h264_nvenc", ""

    error_msg = "Не удалось нарезать клип."
    if result is None:
        return False, "", "FFmpeg не найден."
    if result.stderr:
        error_msg = result.stderr.strip()
    return False, "", f"Не удалось нарезать клип: {error_msg}"


def cut_video_clip(
    video_path: str,
    start_hms: str,
    end_hms: str,
    media_dir: str = "media",
) -> Tuple[bool, str, str]:
    """
    Cut a clip from *video_path* using ffmpeg.

    Returns a tuple (success, message_or_path, mode).
    """

    start_sec = parse_hms(start_hms)
    end_sec = parse_hms(end_hms)

    if start_sec is None or end_sec is None:
        return False, "Неверный формат времени. Используйте HH:MM:SS.", ""
    if end_sec <= start_sec:
        return False, "Время окончания должно быть больше времени начала.", ""
    if (end_sec - start_sec) <= 0.2:
        return False, "Длительность клипа должна быть больше 0.2 секунды.", ""

    os.makedirs(media_dir, exist_ok=True)
    safe_start = start_hms.replace(":", "-")
    safe_end = end_hms.replace(":", "-")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(media_dir, f"clip_{ts}_{safe_start}_{safe_end}.mp4")

    ok, mode, error = cut_video_clip_with_fallback(video_path, start_hms, end_hms, output_path)
    if ok:
        return True, output_path, mode
    return False, error, ""


def _resolve_ffprobe(ffmpeg_path: Optional[str]) -> Optional[str]:
    if ffmpeg_path:
        ffprobe_candidate = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
        if sys.platform.startswith("win"):
            ffprobe_candidate += ".exe"
        if os.path.isfile(ffprobe_candidate):
            return ffprobe_candidate
    return shutil.which("ffprobe")


def get_video_duration_seconds(video_path: str) -> Optional[float]:
    ffmpeg_path = find_ffmpeg()
    ffprobe_path = _resolve_ffprobe(ffmpeg_path)
    if not ffprobe_path:
        return None
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return float(value) if value else None
    except Exception:
        return None


def cut_video_clip_with_poster(
    video_path: str,
    start_hms: str,
    end_hms: str,
    output_dir: str,
) -> Tuple[bool, str | dict]:
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        return False, "FFmpeg не найден. Положите ffmpeg.exe рядом с программой или добавьте его в PATH."

    start_sec = parse_hms(start_hms)
    end_sec = parse_hms(end_hms)
    if start_sec is None or end_sec is None:
        return False, "Неверный формат времени. Используйте HH:MM:SS."
    if end_sec <= start_sec:
        return False, "Время окончания должно быть больше времени начала."
    if (end_sec - start_sec) <= 0.2:
        return False, "Длительность клипа должна быть больше 0.2 секунды."

    os.makedirs(output_dir, exist_ok=True)
    safe_start = start_hms.replace(":", "-")
    safe_end = end_hms.replace(":", "-")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_path = os.path.join(output_dir, f"clip_{ts}_{safe_start}_{safe_end}.mp4")
    poster_path = os.path.join(output_dir, f"clip_{ts}_{safe_start}_{safe_end}.jpg")

    ok, _mode, error = cut_video_clip_with_fallback(video_path, start_hms, end_hms, clip_path)
    if not ok:
        return False, error

    poster_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        clip_path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        poster_path,
    ]
    result = _run_ffmpeg_attempt(poster_cmd, label="poster")
    if result is None or result.returncode != 0:
        error_msg = result.stderr.strip() if result and result.stderr else "Не удалось создать постер."
        return False, error_msg

    return True, {"clip_path": clip_path, "poster_path": poster_path}


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

        control_row = tk.Frame(self.frame, bg="white")
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

        self.volume_panel = tk.Frame(self.frame, bg="white")
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

        seek_frame = tk.Frame(self.frame, bg="white")
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
        handle = int(self.canvas.winfo_id() or 0)
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
        handle = int(self.canvas.winfo_id() or 0)
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
