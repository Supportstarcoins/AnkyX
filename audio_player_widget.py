import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox


base_dir = os.path.dirname(os.path.abspath(__file__))
VLC_IMPORT_ERROR: Exception | None = None
VLC_LOAD_ERROR: Exception | None = None
VLC_AVAILABLE = False
VLC_DIR: str | None = None

if os.name == "nt":  # pragma: no cover - Windows-specific
    candidates = [
        r"C:\\Program Files\\VideoLAN\\VLC",
        r"C:\\Program Files (x86)\\VideoLAN\\VLC",
        os.path.join(base_dir, "VLC"),
    ]

    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "libvlc.dll")) and os.path.isdir(
            os.path.join(candidate, "plugins")
        ):
            VLC_DIR = candidate
            try:
                os.add_dll_directory(VLC_DIR)
            except Exception:
                pass
            os.environ["PATH"] = VLC_DIR + ";" + os.environ.get("PATH", "")
            os.environ["VLC_PLUGIN_PATH"] = os.path.join(VLC_DIR, "plugins")
            print("[VLC] using dir:", VLC_DIR)
            break

    if VLC_DIR is None:
        VLC_LOAD_ERROR = FileNotFoundError("Установите VLC x64")

try:  # pragma: no cover - optional dependency
    import vlc  # type: ignore

    _ = vlc.Instance()
    VLC_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - diagnostic
    VLC_IMPORT_ERROR = exc
    vlc = None  # type: ignore
except Exception as exc:  # pragma: no cover - diagnostic
    VLC_LOAD_ERROR = exc
    vlc = None  # type: ignore


def _log_audio_error(message: str):
    try:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "audio.log"), "a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:
        # Логирование ошибок не должно ломать основную логику
        pass


try:
    import winsound
    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False


class AudioPlayerWidget(ttk.Frame):
    def __init__(self, master, on_error_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_error_callback = on_error_callback
        self._after_id = None
        self._duration_ms = 0
        self._seeking = False
        self._loaded_path: str | None = None
        self._resolved_path: str | None = None
        self._volume = 70.0
        self._rate = 1.0

        self._vlc_ready = False
        self._vlc_instance = None
        self._player = None
        if VLC_AVAILABLE and VLC_LOAD_ERROR is None:
            try:
                self._vlc_instance = vlc.Instance()
                self._player = self._vlc_instance.media_player_new()
                self._vlc_ready = True
            except Exception as exc:  # pragma: no cover - defensive
                self._handle_error("VLC не удалось инициализировать", exc)
                print(f"[audio] Ошибка инициализации VLC: {exc}")
                _log_audio_error(f"Ошибка инициализации VLC: {exc}")

        self._build_ui()
        self._set_controls_state(False)

        if not self._vlc_ready:
            if VLC_LOAD_ERROR:
                self._set_status("Установите VLC x64")
            elif VLC_IMPORT_ERROR:
                self._set_status("Установите: pip install python-vlc")
            else:
                self._set_status("Установите VLC x64")

    def _build_ui(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, pady=2)

        self.play_btn = ttk.Button(control_frame, text="▶ Play", command=self.play)
        self.play_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(control_frame, text="⏸ Pause", command=self.pause)
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", command=self.stop)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        vol_frame = ttk.Frame(self)
        vol_frame.pack(fill=tk.X, pady=2)
        ttk.Label(vol_frame, text="Громкость").pack(side=tk.LEFT, padx=(0, 5))
        self.volume_var = tk.DoubleVar(value=70)
        self.volume_scale = ttk.Scale(
            vol_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.volume_var,
            command=lambda _v: self.set_volume(self.volume_var.get())
        )
        self.volume_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        rate_frame = ttk.Frame(self)
        rate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rate_frame, text="Скорость").pack(side=tk.LEFT, padx=(0, 5))
        self.rate_var = tk.DoubleVar(value=1.0)
        self.rate_scale = ttk.Scale(
            rate_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, variable=self.rate_var,
            command=lambda _v: self.set_rate(self.rate_var.get())
        )
        self.rate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        seek_frame = ttk.Frame(self)
        seek_frame.pack(fill=tk.X, pady=2)
        self.time_label = ttk.Label(seek_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=(0, 5))

        self.seek_var = tk.DoubleVar(value=0)
        self.seek_scale = ttk.Scale(
            seek_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.seek_var,
            command=lambda _v: self._on_seek(),
        )
        self.seek_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.seek_scale.bind("<ButtonPress-1>", lambda _e: self._start_seeking())
        self.seek_scale.bind("<ButtonRelease-1>", lambda _e: self._end_seeking())

        self.status_label = ttk.Label(self, text="")
        self.status_label.pack(fill=tk.X, pady=(2, 0))

    def _handle_error(self, msg: str, exc: Exception | None = None):
        if self.on_error_callback:
            detail = f"{msg}: {exc}" if exc else msg
            self.on_error_callback("Аудио", detail)
        if exc:
            _log_audio_error(f"{msg}: {exc}")
        else:
            _log_audio_error(msg)

    def _resolve_audio_path(self, audio_path: str | None) -> str | None:
        if not audio_path:
            return None

        cleaned = audio_path.strip()
        if cleaned.lower().startswith("[sound:") and cleaned.endswith("]"):
            cleaned = cleaned[len("[sound:") : -1]

        cleaned = cleaned.replace("\\", os.sep).replace("/", os.sep)
        cleaned = os.path.normpath(cleaned)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates: list[str] = []

        if os.path.isabs(cleaned):
            candidates.append(cleaned)
        else:
            candidates.extend(
                [
                    os.path.join(base_dir, cleaned),
                    os.path.join(base_dir, "media", cleaned),
                    os.path.join(base_dir, "media", "anki_import", cleaned),
                ]
            )

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _set_status(self, text: str):
        self.status_label.config(text=text)

    def _set_controls_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (self.play_btn, self.stop_btn):
            widget.config(state=state)

        advanced_state = state if self._vlc_ready else tk.DISABLED
        for widget in (
            self.pause_btn,
            self.volume_scale,
            self.rate_scale,
            self.seek_scale,
        ):
            widget.config(state=advanced_state)

    def _start_seeking(self):
        self._seeking = True

    def _end_seeking(self):
        self._seeking = False
        self._on_seek()

    def _on_seek(self):
        if not self._player or not self._vlc_ready:
            return
        if self._duration_ms <= 0:
            return
        position_ms = min(max(self.seek_var.get(), 0), self._duration_ms)
        self._player.set_time(int(position_ms))

    def load(self, path: str | None):
        self.stop()
        self._loaded_path = path
        self._resolved_path = self._resolve_audio_path(path)

        print("[AUDIO] requested=", path)
        print(
            "[AUDIO] resolved =",
            self._resolved_path,
            "exists=",
            os.path.exists(self._resolved_path) if self._resolved_path else False,
        )

        if not path:
            self._loaded_path = None
            self._resolved_path = None
            self._set_controls_state(False)
            self._set_status("Аудио не найдено")
            return False

        if not self._resolved_path or not os.path.exists(self._resolved_path):
            self._loaded_path = None
            self._resolved_path = None
            self._set_controls_state(False)
            self._set_status("Аудио не найдено")
            messagebox.showerror("Аудио", "Файл аудио не найден")
            _log_audio_error(f"Аудио не найдено: {path}")
            return False

        if self._vlc_ready and self._player:
            try:
                media = self._vlc_instance.media_new(self._resolved_path)
                self._player.set_media(media)
                media.parse_with_options(vlc.MediaParseFlag.local, timeout=1)
                duration = media.get_duration()
                if duration and duration > 0:
                    self._duration_ms = duration
                    self.seek_scale.config(to=self._duration_ms)
                self.set_volume(self.volume_var.get())
                self.set_rate(self.rate_var.get())
                self._set_status(os.path.basename(path))
            except Exception as exc:
                self._handle_error("Не удалось загрузить аудио", exc)
                self._set_status("Ошибка загрузки аудио")
                self._set_controls_state(False)
                return False
        else:
            self._set_status(os.path.basename(path))
        self._set_controls_state(True)
        return True

    def is_loaded(self) -> bool:
        return self._loaded_path is not None

    def play(self):
        try:
            if not self.is_loaded():
                messagebox.showerror("Аудио", "Аудио не загружено")
                _log_audio_error("Попытка воспроизведения без загруженного аудио")
                return

            print("[AUDIO] requested=", self._loaded_path)
            print(
                "[AUDIO] resolved =",
                self._resolved_path,
                "exists=",
                os.path.exists(self._resolved_path) if self._resolved_path else False,
            )

            if not self._resolved_path or not os.path.exists(self._resolved_path):
                messagebox.showerror("Аудио", "Файл аудио не найден")
                _log_audio_error(f"Файл аудио не найден: {self._loaded_path}")
                return

            if not VLC_AVAILABLE:
                messagebox.showerror(
                    "Аудио", "VLC не настроен. Установите VLC и перезапустите."
                )
                _log_audio_error("VLC не настроен для воспроизведения")
                return

            if self._vlc_ready and self._player:
                try:
                    self._player.play()
                    self._schedule_progress_update()
                    self._set_status(os.path.basename(self._resolved_path))
                except Exception as exc:
                    self._handle_error("Не удалось воспроизвести", exc)
                    messagebox.showerror("Аудио", f"Не удалось воспроизвести: {exc}")
                return

            if WINSOUND_AVAILABLE and self._resolved_path.lower().endswith(".wav"):
                threading.Thread(
                    target=lambda: winsound.PlaySound(
                        self._resolved_path, winsound.SND_FILENAME | winsound.SND_ASYNC
                    ),
                    daemon=True,
                ).start()
                return

            messagebox.showerror(
                "Аудио",
                "Не удаётся воспроизвести файл: требуется python-vlc или WAV для winsound",
            )
            _log_audio_error(
                f"Нет доступного аудио движка для воспроизведения: {self._resolved_path}"
            )
        except Exception as exc:  # noqa: BLE001 - обработка ошибок воспроизведения
            _log_audio_error(f"Необработанная ошибка воспроизведения: {exc}")
            messagebox.showerror("Аудио", f"Ошибка воспроизведения: {exc}")

    def pause(self):
        if self._vlc_ready and self._player:
            try:
                self._player.pause()
                self._set_status("Пауза")
            except Exception as exc:
                self._handle_error("Не удалось поставить на паузу", exc)

    def stop(self):
        if self._vlc_ready and self._player:
            try:
                self._player.stop()
            except Exception:
                pass
        if WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        self._cancel_after()
        self._update_time_label(0, self._duration_ms)
        self.seek_var.set(0)

    def set_volume(self, value: float):
        self._volume = float(value)
        if self._vlc_ready and self._player:
            try:
                self._player.audio_set_volume(int(self._volume))
            except Exception:
                pass

    def set_rate(self, value: float):
        self._rate = float(value)
        if self._vlc_ready and self._player:
            try:
                ok = self._player.set_rate(float(self._rate))
                if ok is False:
                    messagebox.showwarning(
                        "Скорость недоступна",
                        "Текущий аудиодвижок не поддерживает изменение скорости.",
                    )
            except Exception as exc:
                self._handle_error("Не удалось изменить скорость", exc)

    def seek(self, seconds: float):
        if not (self._vlc_ready and self._player):
            return
        self._player.set_time(int(seconds * 1000))

    def _schedule_progress_update(self):
        self._cancel_after()
        self._after_id = self.after(200, self._update_progress)

    def _update_time_label(self, current_ms: int, total_ms: int):
        def fmt(ms):
            seconds = max(ms, 0) // 1000
            return f"{seconds//60:02d}:{seconds%60:02d}"
        self.time_label.config(text=f"{fmt(current_ms)} / {fmt(total_ms)}")

    def _update_progress(self):
        if not (self._vlc_ready and self._player):
            return
        try:
            total = self._player.get_length()
            if total and total > 0:
                self._duration_ms = total
                self.seek_scale.config(to=total)
            current = self._player.get_time()
            self._update_time_label(current, self._duration_ms)
            if not self._seeking:
                self.seek_var.set(current)
            state = self._player.get_state()
            if state in (vlc.State.Ended, vlc.State.Stopped):
                self._cancel_after()
                return
            self._after_id = self.after(200, self._update_progress)
        except Exception:
            self._cancel_after()

    def _cancel_after(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def destroy(self):  # noqa: D401 - Tkinter lifecycle
        self._cancel_after()
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass
        super().destroy()
