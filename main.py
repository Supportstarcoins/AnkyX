import os, tempfile
import sys
import logging
import faulthandler
import subprocess
import ctypes
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
import io
safe = tempfile.gettempdir()
os.environ["TEMP"] = safe
os.environ["TMP"] = safe
import hashlib
import time
import traceback
import base64
import sqlite3
import re
import glob
import threading
import queue
import json
import csv
import calendar
import shutil
import webbrowser
import urllib.parse
import urllib.request
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
from pathlib import Path
from uuid import uuid4
csv.field_size_limit(10 * 1024 * 1024)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYNC_CONFIG_PATH = os.path.join(BASE_DIR, "config_sync.json")
SYNC_CONFIG_DEFAULT = {
    "api_base_url": "https://example.com/api",
    "timeout_sec": 15,
    "token": None,
    "user_email": None,
}

CRASH_LOG_PATH = os.path.join(BASE_DIR, "app_crash.log")
FAULT_LOG_PATH = os.path.join(BASE_DIR, "fault.log")
_CRASH_LOG_HANDLE = None
_FAULT_LOG_HANDLE = None


def _configure_global_crash_logging() -> None:
    global _CRASH_LOG_HANDLE, _FAULT_LOG_HANDLE
    crash_handler = logging.FileHandler(CRASH_LOG_PATH, encoding="utf-8", mode="a")
    crash_handler.setLevel(logging.DEBUG)
    crash_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(crash_handler)
    root_logger.propagate = False

    crash_stream = open(CRASH_LOG_PATH, "a", buffering=1, encoding="utf-8")
    sys.stdout = crash_stream
    sys.stderr = crash_stream
    _CRASH_LOG_HANDLE = crash_stream

    fault_stream = open(FAULT_LOG_PATH, "a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=fault_stream, all_threads=True)
    _FAULT_LOG_HANDLE = fault_stream


_configure_global_crash_logging()
import gzip
import pickle
from datetime import date, datetime, timedelta
import collections
import html
import math
from html.parser import HTMLParser
from csv_importer import (
    attach_image_if_exists,
    detect_encoding,
    map_row_to_fields,
    normalize_tags,
    render_card_faces,
    upsert_note_and_cards,
)
from anki_apkg_importer import import_apkg

# В начале main() или init_db()
os.makedirs("sentence_audio", exist_ok=True)

# Добавить в начало файла
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser, simpledialog, scrolledtext
import tkinter.font as tkfont
import importlib.util
from addons_manager import AddonManager, UIAdapter, MWContext, CollectionAdapter, gui_hooks, set_mw

_EXCEPTIONBOX_LOCK = threading.Lock()


def _format_exception_message(exc_type, exc, tb) -> str:
    formatted = "".join(traceback.format_exception(exc_type, exc, tb))
    return formatted


def _show_fatal_messagebox(title: str, body: str) -> None:
    with _EXCEPTIONBOX_LOCK:
        try:
            ctypes.windll.user32.MessageBoxW(0, body, title, 0x10)
        except Exception:
            logging.exception("Failed to show WinAPI messagebox for exception")


def _global_excepthook(exc_type, exc, tb) -> None:
    formatted = _format_exception_message(exc_type, exc, tb)
    logging.error("Unhandled exception:\n%s", formatted)
    _show_fatal_messagebox(
        "Ошибка приложения",
        f"Произошла необработанная ошибка:\n{exc}\n\n"
        f"Подробности в {CRASH_LOG_PATH} / {FAULT_LOG_PATH}",
    )


def _thread_excepthook(args) -> None:
    formatted = _format_exception_message(args.exc_type, args.exc_value, args.exc_traceback)
    logging.error("Unhandled thread exception:\n%s", formatted)
    _show_fatal_messagebox(
        "Ошибка в фоне",
        f"Произошла ошибка в фоновом потоке:\n{args.exc_value}\n\n"
        f"Подробности в {CRASH_LOG_PATH} / {FAULT_LOG_PATH}",
    )


sys.excepthook = _global_excepthook
if hasattr(threading, "excepthook"):
    threading.excepthook = _thread_excepthook

_HAS_DND = importlib.util.find_spec("tkinterdnd2") is not None
if _HAS_DND:
    from tkinterdnd2 import DND_FILES

OCR_DEBUG_LOG_PATH = os.path.join(BASE_DIR, "ocr_debug.log")
_OCR_DEBUG_LOG_HANDLE = None


def open_ocr_debug_log():
    global _OCR_DEBUG_LOG_HANDLE
    if _OCR_DEBUG_LOG_HANDLE is None:
        try:
            _OCR_DEBUG_LOG_HANDLE = open(OCR_DEBUG_LOG_PATH, "a", encoding="utf-8")
        except Exception:
            _OCR_DEBUG_LOG_HANDLE = None
    return _OCR_DEBUG_LOG_HANDLE


def log_ocr_error(name: str, tb: str) -> None:
    handle = _OCR_DEBUG_LOG_HANDLE
    if handle is None:
        return
    try:
        ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        handle.write(f"[{ts}] {name}\n{tb}\n")
        handle.flush()
    except Exception:
        pass


def safe_action(name, fn):
    try:
        fn()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[OCR ERROR]", name, tb)
        log_ocr_error(name, tb)
        root = tk._default_root
        if root is not None:
            messagebox.showerror("Ошибка", f"{name}: {e}\n\n{tb}")
        else:
            logging.error("UI error without root: %s\n%s", name, tb)


def safe_enable_dnd(widget, on_drop_callback) -> bool:
    try:
        from tkinterdnd2 import DND_FILES
    except Exception:
        return False

    try:
        cmds = widget.tk.call("info", "commands")
    except Exception:
        return False

    if isinstance(cmds, str):
        available = "tkdnd::drop_target" in cmds.split()
    else:
        available = "tkdnd::drop_target" in cmds
    if not available:
        return False

    try:
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", on_drop_callback)
        return True
    except Exception:
        return False

from stats_config import (
    StatsSettings,
    ensure_stats_settings_table,
    load_stats_settings,
    save_stats_settings,
)
from db_migrations import ensure_schema_for_import, run_migrations
from db_connect import DB_WRITE_LOCK, commit_with_retry, open_db
from db_path import get_db_path
from credits import CreditsService
from referral import ReferralService
from sync_client import SyncClient
from payments import PACKAGES, build_payment_url, verify_payment
from srs import schedule_review
from bg_tasks import BackgroundTask, start_background_task
from ui_progress import BusyDialog, TaskRunner
from importers import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_MAX_PDF_PAGES_SOFT,
    DEFAULT_MAX_TOTAL_CHARS_SOFT,
    import_docx,
    import_odt,
    import_pdf,
)
from web_editor import QUILL_WEBVIEW_AVAILABLE, WebEditorManager, open_fallback_editor
from quill_cards import parse_quill_html_to_cards
from overdue_badges import (
    PhaseOverdueBadges,
    ensure_due_column,
    fetch_overdue_counts_by_phase,
)
from deck_timer import (
    DEFAULT_PHASE_INTERVALS,
    ensure_deck_settings_row,
    ensure_deck_settings_table,
    get_deck_phase_intervals,
    get_deck_timer_settings,
    get_effective_mode_timer,
    reset_deck_phase_intervals,
    save_deck_phase_intervals,
    update_deck_timer_settings,
)
from video_tools import (
    VlcPlayerWidget,
    cut_video_clip,
    find_ffmpeg,
    format_hms,
    get_video_duration_seconds,
    is_vlc_available,
    open_in_external_player,
)
from audio_player_widget import AudioPlayerWidget
# ui_theme: в разных версиях проекта функции темы могут называться иначе.
# Нельзя падать на ImportError — интерфейс должен запускаться даже без темы.
try:
    import ui_theme as _ui_theme
except Exception:
    _ui_theme = None


def _logo_debug(msg):
    try:
        print("[LOGO]", msg)
    except Exception:
        pass


def load_app_logo(master: tk.Misc, base_dir: str):
    logo_path = os.path.join(base_dir, "assets", "logo.png")
    _logo_debug(f"logo_path={logo_path} exists={os.path.exists(logo_path)}")
    if not os.path.exists(logo_path):
        return None, None
    try:
        big = tk.PhotoImage(master=master, file=logo_path)
        small = big.subsample(10, 10)
        return small, big
    except Exception as e:
        _logo_debug(f"PhotoImage load failed: {e}")
        return None, None


def apply_window_icon(win: tk.Misc, big_photo: tk.PhotoImage | None, ico_path: str | None = None) -> None:
    try:
        if ico_path and os.path.exists(ico_path):
            try:
                win.iconbitmap(ico_path)
                _logo_debug(f"iconbitmap OK: {ico_path}")
            except Exception as e:
                _logo_debug(f"iconbitmap failed: {e}")
        if big_photo is not None:
            try:
                win.iconphoto(True, big_photo)
                try:
                    win.tk.call('wm', 'iconphoto', win._w, big_photo)
                except Exception:
                    pass
                _logo_debug("iconphoto OK")
            except Exception as e:
                _logo_debug(f"iconphoto failed: {e}")
    except Exception as e:
        _logo_debug(f"apply_window_icon error: {e}")

def apply_dark_theme_to_window(window, *args, **kwargs):
    # Best-effort применение тёмной темы (совместимость между версиями).
    try:
        mod = _ui_theme
        if mod is None:
            return None
        fn = getattr(mod, 'apply_dark_theme_to_window', None)
        if fn is None:
            fn = getattr(mod, 'apply_dark_theme', None)
        if fn is None:
            fn = getattr(mod, 'apply_premium_dark_theme', None)
        if callable(fn):
            try:
                result = fn(window, *args, **kwargs)
                try:
                    root = window.winfo_toplevel() if hasattr(window, "winfo_toplevel") else window
                    _fix_tk_default_fonts(root)
                except Exception:
                    pass
                return result
            except TypeError:
                try:
                    result = fn(window)
                    try:
                        root = window.winfo_toplevel() if hasattr(window, "winfo_toplevel") else window
                        _fix_tk_default_fonts(root)
                    except Exception:
                        pass
                    return result
                except TypeError:
                    # некоторые реализации могут не принимать аргументы
                    try:
                        result = fn()
                        try:
                            root = window.winfo_toplevel() if hasattr(window, "winfo_toplevel") else window
                            _fix_tk_default_fonts(root)
                        except Exception:
                            pass
                        return result
                    except Exception:
                        return None
    except Exception:
        return None
    return None

apply_premium_dark_theme = (getattr(_ui_theme, 'apply_premium_dark_theme', None) if _ui_theme else None) or (lambda *_a, **_k: None)
style_card = (getattr(_ui_theme, 'style_card', None) if _ui_theme else None) or (lambda *_a, **_k: None)
style_card_surface = (getattr(_ui_theme, 'style_card_surface', None) if _ui_theme else None) or (lambda *_a, **_k: None)
style_card_surface_text = (getattr(_ui_theme, 'style_card_surface_text', None) if _ui_theme else None) or (lambda *_a, **_k: None)
style_text_widget = (getattr(_ui_theme, 'style_text_widget', None) if _ui_theme else None) or (lambda *_a, **_k: None)

get_card_surface_colors = (getattr(_ui_theme, 'get_card_surface_colors', None) if _ui_theme else None)
if get_card_surface_colors is None:
    def get_card_surface_colors(*_a, **_k):
        # (bg, card_bg, text)
        return ('#0f1115', '#ffffff', '#111111')
DARK_BG = getattr(_ui_theme, "DARK_BG", "#0B1220")
CARD_BORDER = getattr(_ui_theme, "CARD_BORDER", "#D6DCE6")
SCROLL_TROUGH = getattr(_ui_theme, "SCROLL_TROUGH", "#05070b")
SCROLL_BG = getattr(_ui_theme, "SCROLL_BG", "#0b0f16")
SCROLL_ACTIVE = getattr(_ui_theme, "SCROLL_ACTIVE", "#121a26")
CARD_VIEW_WIDTH = 700
CARD_VIEW_HEIGHT = 420
REPEAT_MEDIA_SLOT_SIZE = (260, 240)
from card_widget import CardWidget
from image_utils import MAX_PREVIEW_PIXELS, load_preview_image, log_image_error

def _fix_tk_default_fonts(root: tk.Tk, family: str = "Segoe UI", size: int = 11) -> None:
    """Fix invalid global font settings that can break tk.Menu on Windows.

    If a theme sets font to a bare family name like 'Segoe UI' (without size),
    Tk may parse it as ['Segoe', 'UI'] and crash with:
        _tkinter.TclError: expected integer but got "UI"
    """
    try:
        root.option_add("*Font", f"{{{family}}} {int(size)}")
    except Exception:
        pass

    named_fonts = [
        ("TkDefaultFont", size),
        ("TkTextFont", size),
        ("TkMenuFont", max(9, size)),
        ("TkHeadingFont", size),
        ("TkFixedFont", size),
        ("TkTooltipFont", size),
        ("TkCaptionFont", size),
        ("TkSmallCaptionFont", max(8, size - 1)),
        ("TkIconFont", size),
    ]
    for name, fsize in named_fonts:
        try:
            tkfont.nametofont(name).configure(family=family, size=int(fsize))
        except Exception:
            pass


class GlobalLoadingOverlay:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._visible = False
        self._determinate = False
        self._overlay = tk.Frame(root, bg="#0B0D12")
        self._overlay.place_forget()
        self._overlay.bind("<Button-1>", lambda _e: "break")
        self._overlay.bind("<ButtonRelease-1>", lambda _e: "break")
        self._overlay.bind("<Key>", lambda _e: "break")
        self._overlay.bind("<MouseWheel>", lambda _e: "break")
        self._overlay.bind("<Button-2>", lambda _e: "break")
        self._overlay.bind("<Button-3>", lambda _e: "break")

        container = ttk.Frame(self._overlay)
        container.place(relx=0.5, rely=0.5, anchor="center")

        self._progress = ttk.Progressbar(container, mode="indeterminate", length=260)
        self._progress.pack(pady=(0, 8))
        self._label = ttk.Label(container, text="Загрузка")
        self._label.pack()

    def show(self, parent: tk.Widget | None = None, determinate: bool = False, maximum: int = 100) -> None:
        _ = parent
        self._determinate = determinate
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._overlay.lift()
        try:
            self._overlay.grab_set()
        except Exception:
            pass
        self._progress.configure(mode="determinate" if determinate else "indeterminate", maximum=maximum)
        if determinate:
            self._progress.stop()
            self._progress["value"] = 0
        else:
            self._progress.start(10)
        self._visible = True

    def set_progress(self, value: float) -> None:
        if self._determinate:
            self._progress["value"] = value

    def hide(self) -> None:
        if not self._visible:
            return
        try:
            self._overlay.grab_release()
        except Exception:
            pass
        self._progress.stop()
        self._overlay.place_forget()
        self._visible = False


# ==========================
# OCR: pytesseract + автопоиск tesseract.exe
# ==========================

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    OCR_AVAILABLE = False

from tesseract_setup import (
    configure_pytesseract,
    ensure_languages,
    get_tesseract_diag,
    get_tessdata_dir,
    get_tesseract_cmd,
    is_tesseract_available as setup_is_tesseract_available,
    to_short_path,
)

DEFAULT_TESSDATA_DIR = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
DEFAULT_TESSERACT_CMD = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
DEFAULT_OCR_LANG = "deu+rus"
DEFAULT_OCR_CONFIG_BASE = "--oem 1 --psm 6 -c preserve_interword_spaces=1"


def auto_configure_tesseract():
    if not OCR_AVAILABLE:
        return
    configure_pytesseract()


def is_tesseract_available() -> bool:
    if not OCR_AVAILABLE:
        return False
    return setup_is_tesseract_available()


def _ensure_deu_rus_present(selected_lang: str) -> bool:
    if "deu" in selected_lang and "rus" in selected_lang:
        tessdata_dir = get_tessdata_dir() or DEFAULT_TESSDATA_DIR
        ok, missing = ensure_languages(["deu", "rus"])
        missing_files = [code for code in missing]
        for code in ["deu", "rus"]:
            expected_file = os.path.join(tessdata_dir, f"{code}.traineddata")
            if not os.path.isfile(expected_file) and code not in missing_files:
                missing_files.append(code)

        if missing_files:
            missing_display = ", ".join(f"{code}.traineddata" for code in missing_files)
            messagebox.showerror(
                "Не хватает языков Tesseract",
                "Не найдены файлы языков для OCR.\n"
                f"Ожидается папка tessdata: {tessdata_dir}\n"
                f"Отсутствуют: {missing_display}\n\n"
                "Скачайте deu.traineddata и rus.traineddata и поместите их в указанную папку.",
            )
            return False
    return True


def _build_required_ocr_config(base_config: str = DEFAULT_OCR_CONFIG_BASE) -> tuple[str, str, str]:
    tessdata_dir = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
    tessdata_dir_short = to_short_path(tessdata_dir)
    config_base = (base_config or "--oem 1 --psm 6").strip()
    config = f"{config_base} --tessdata-dir {tessdata_dir_short}".strip()
    tesseract_cmd = to_short_path(get_tesseract_cmd() or DEFAULT_TESSERACT_CMD)
    if pytesseract:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return config, tessdata_dir_short, tesseract_cmd


def _ensure_required_lang_files() -> bool:
    tessdata_dir = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
    missing_codes = [code for code in ("deu", "rus") if not os.path.isfile(os.path.join(tessdata_dir, f"{code}.traineddata"))]
    if missing_codes:
        missing_display = ", ".join(f"{code}.traineddata" for code in missing_codes)
        messagebox.showerror(
            "Не хватает языков Tesseract",
            "Не найдены файлы языков для OCR.\n"
            f"Ожидается папка tessdata: {tessdata_dir}\n"
            f"Отсутствуют: {missing_display}\n\n"
            "Скачайте deu.traineddata и rus.traineddata и поместите их в указанную папку.",
        )
        return False
    return True


def _format_ocr_diag(
    config: str,
    lang: str,
    image_diag: str | None = None,
    ocr_mode: str | None = None,
    preprocess_enabled: bool | None = None,
) -> str:
    tessdata_dir = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
    tessdata_dir_short = to_short_path(tessdata_dir)
    tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    diag = get_tesseract_diag()
    extra = [
        f"tesseract_cmd: {tesseract_cmd}",
        f"tessdata_dir: {repr(tessdata_dir)}",
        f"tessdata_dir_short: {tessdata_dir_short}",
        f"config: {repr(config)}",
        f"lang: {lang}",
        f"ocr_mode: {ocr_mode or 'standard'}",
        f"preprocess_enabled: {preprocess_enabled}",
        f"PIL version: {_get_pil_version()}",
        f"OpenCV version: {_get_cv2_version()}",
    ]
    blocks = ["\n".join(extra), "Диагностика Tesseract:", diag]
    if image_diag:
        blocks.insert(1, image_diag)
    return "\n\n".join(blocks)


def _format_image_diag(image_path: str, img: Image.Image) -> str:
    exists = os.path.exists(image_path)
    size = os.path.getsize(image_path) if exists else 0
    pil_version = _get_pil_version()
    dimensions = getattr(img, "size", None)
    width, height = dimensions if dimensions else ("unknown", "unknown")
    image_format = getattr(img, "_anky_original_format", None) or getattr(img, "format", None) or "unknown"
    return "\n".join(
        [
            f"image_path: {image_path}",
            f"exists/size bytes: {exists}/{size}",
            f"dimensions: {width}x{height}",
            f"PIL version: {pil_version}",
            f"image format: {image_format}",
            f"image mode: {getattr(img, 'mode', 'unknown')}",
        ]
    )


def load_image_for_ocr(path: str) -> Image.Image:
    if not PIL_AVAILABLE:
        messagebox.showerror(
            "Не удалось открыть изображение",
            "Для загрузки изображений нужен модуль Pillow.\n"
            "Установите его и повторите попытку: C:\\AnkyX-main\\venv\\Scripts\\python.exe -m pip install pillow",
        )
        raise RuntimeError("Pillow недоступен")

    with open(path, "rb") as f:
        data = f.read()

    img = Image.open(io.BytesIO(data))
    img._anky_original_format = getattr(img, "format", None) or "unknown"
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.load()
    return img


def _get_pil_version() -> str:
    if PIL_AVAILABLE:
        try:
            import PIL

            return getattr(PIL, "__version__", "unknown")
        except Exception:
            return "unknown"
    return "unavailable"


def _get_cv2_version() -> str:
    if CV2_AVAILABLE:
        try:
            import cv2 as _cv2

            return getattr(_cv2, "__version__", "unknown")
        except Exception:
            return "unknown"
    return "unavailable"


def preprocess_for_ocr(image_path: str) -> Image.Image:
    if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
        messagebox.showerror(
            "Недоступна обработка изображения",
            "Для улучшения OCR необходимы OpenCV и NumPy.\n"
            "Установите пакеты: C:\\AnkyX-main\\venv\\Scripts\\python.exe -m pip install opencv-python numpy",
        )
        raise RuntimeError("OpenCV/NumPy недоступны для предобработки")

    pil_img = load_image_for_ocr(image_path)
    np_img = np.array(pil_img)
    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    denoised = cv2.bilateralFilter(clahe_img, d=7, sigmaColor=50, sigmaSpace=50)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
    )

    return Image.fromarray(binary)


def split_two_columns(pil_img: Image.Image) -> tuple[Image.Image, Image.Image]:
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    padding = max(1, int(w * 0.03))
    mid = w // 2
    left_box = (max(0, 0), 0, max(padding, mid - padding), h)
    right_box = (min(w, mid + padding), 0, w, h)
    left_img = pil_img.crop(left_box)
    right_img = pil_img.crop(right_box)
    return left_img, right_img


def ocr_image(pil_img: Image.Image, lang: str, config: str) -> str:
    tmp_png = Path(tempfile.gettempdir()) / f"anky_ocr_{uuid4().hex}.png"
    try:
        pil_img.save(tmp_png, format="PNG")
        return pytesseract.image_to_string(str(tmp_png), lang=lang, config=config)
    finally:
        if tmp_png.exists():
            tmp_png.unlink()


auto_configure_tesseract()

def load_sync_config() -> dict:
    config = SYNC_CONFIG_DEFAULT.copy()
    if not os.path.exists(SYNC_CONFIG_PATH):
        return config
    try:
        with open(SYNC_CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            config.update({key: data.get(key) for key in SYNC_CONFIG_DEFAULT})
    except Exception:
        return config
    return config


def save_sync_config(config: dict) -> None:
    payload = SYNC_CONFIG_DEFAULT.copy()
    payload.update({key: config.get(key) for key in SYNC_CONFIG_DEFAULT})
    with open(SYNC_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

# ==========================
# Необязательные библиотеки
# ==========================

# Картинки
try:
    from PIL import Image, ImageTk, ImageDraw, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OpenCV (опционально для улучшенного OCR)
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# TTS (озвучка)
try:
    import pyttsx3
    TTS_AVAILABLE = True
    _tts_engine = pyttsx3.init()
except Exception:
    TTS_AVAILABLE = False
    _tts_engine = None

# Звук файлов (Windows)
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# OpenAI
try:
    from openai import OpenAI
    OPENAI_LIB_AVAILABLE = True
except ImportError:
    OPENAI_LIB_AVAILABLE = False

# Распознавание речи (цифровой слух)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    sr = None
    SR_AVAILABLE = False

FASTER_WHISPER_IMPORT_ERROR: str | None = None
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception as exc:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False
    FASTER_WHISPER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
WHISPER_AVAILABLE = importlib.util.find_spec("whisper") is not None
VOSK_AVAILABLE = importlib.util.find_spec("vosk") is not None
WEBRTCVAD_AVAILABLE = importlib.util.find_spec("webrtcvad") is not None

WHISPER_MODEL_CACHE: dict[tuple[str, str, str], "WhisperModel"] = {}
WHISPER_MODEL_LOCK = threading.Lock()


def get_whisper_model(
    model_name: str,
    device: str = "cpu",
    compute_type: str = "int8",
) -> tuple["WhisperModel", bool]:
    key = (model_name, device, compute_type)
    with WHISPER_MODEL_LOCK:
        if key in WHISPER_MODEL_CACHE:
            return WHISPER_MODEL_CACHE[key], True
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() or 1,
        )
        WHISPER_MODEL_CACHE[key] = model
        return model, False


def _get_whisper_cache_dir() -> str:
    env_paths = [
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HF_HOME"),
    ]
    for candidate in env_paths:
        if candidate:
            return os.path.expanduser(candidate)
    return os.path.expanduser("~/.cache/huggingface")

# Deutsch Wiktionary
try:
    import requests
    from bs4 import BeautifulSoup
    WIKTIONARY_AVAILABLE = True
except ImportError:
    WIKTIONARY_AVAILABLE = False

# Matplotlib (опционально)
try:
    import matplotlib
    matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    FigureCanvasTkAgg = None
    Figure = None

# PATCH: avoid paddle import crash (lazy import + subprocess sanity check + fallback OCR engines) + crash-safe logging + WinAPI MessageBox (no Tk in excepthook)
_PADDLE_SANITY_CACHE = {
    "checked": False,
    "available": False,
    "rc": None,
    "stdout": "",
    "stderr": "",
}


def is_paddle_available() -> bool:
    if _PADDLE_SANITY_CACHE["checked"]:
        return bool(_PADDLE_SANITY_CACHE["available"])

    script_path = os.path.join(BASE_DIR, "tools", "paddle_sanity_check.py")
    cmd = [sys.executable, "-u", script_path]
    logging.info("Running Paddle sanity check: %s", cmd)
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True)
    except Exception:
        tb = traceback.format_exc()
        logging.error("Paddle sanity check failed to start:\n%s", tb)
        _PADDLE_SANITY_CACHE.update(
            {"checked": True, "available": False, "rc": None, "stdout": "", "stderr": tb}
        )
        return False

    rc = completed.returncode
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    _PADDLE_SANITY_CACHE.update(
        {"checked": True, "available": rc == 0, "rc": rc, "stdout": stdout, "stderr": stderr}
    )
    logging.info("Paddle sanity check rc=%s stdout=%s stderr=%s", rc, stdout, stderr)
    return rc == 0


_EASYOCR_AVAILABLE = importlib.util.find_spec("easyocr") is not None
_EASYOCR_READER_CACHE: dict[str, object] = {}


def is_easyocr_available() -> bool:
    return bool(_EASYOCR_AVAILABLE)


def _normalize_easyocr_langs(lang_mode: str) -> list[str]:
    langs: list[str] = []
    lang_mode = (lang_mode or "").lower()
    if "deu" in lang_mode:
        langs.append("de")
    if "rus" in lang_mode:
        langs.append("ru")
    if not langs:
        langs.append("en")
    if "en" not in langs:
        langs.append("en")
    return langs


def _get_easyocr_reader(lang_mode: str):
    langs = _normalize_easyocr_langs(lang_mode)
    key = ",".join(langs)
    if key in _EASYOCR_READER_CACHE:
        return _EASYOCR_READER_CACHE[key]
    import easyocr

    reader = easyocr.Reader(langs, gpu=False)
    _EASYOCR_READER_CACHE[key] = reader
    return reader

# --------------------------
# OCR Photo (OpenCV/PaddleOCR) — optional. Load lazily to avoid startup crashes/spam.
# --------------------------

# Fallback options container (used even when PaddleOCR is not available)
class OcrRunOptions:  # noqa: N801
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "ocr_engine"):
            setattr(self, "ocr_engine", "tesseract")

# Lazily populated when/if ocr_photo is successfully imported
PADDLE_AVAILABLE = False
PADDLEOCR_AVAILABLE = False
load_image_any = None  # type: ignore
ocr_photo_document = None  # type: ignore
_perform_page_ocr_impl = None  # type: ignore

def _ensure_ocr_photo_loaded() -> bool:
    """Try to import ocr_photo (and PaddleOCR) only when needed.

    Returns True if import succeeded and PaddleOCR pipeline is available.
    """
    global PADDLE_AVAILABLE, PADDLEOCR_AVAILABLE, load_image_any, ocr_photo_document, _perform_page_ocr_impl, OcrRunOptions
    if _perform_page_ocr_impl is not None:
        return True
    if not CV2_AVAILABLE:
        return False
    try:
        from ocr_photo import (
            OcrRunOptions as _OcrRunOptions,
            PADDLE_AVAILABLE as _PADDLE_AVAILABLE,
            PADDLEOCR_AVAILABLE as _PADDLEOCR_AVAILABLE,
            load_image_any as _load_image_any,
            ocr_photo_document as _ocr_photo_document,
            perform_page_ocr as _perform_page_ocr,
        )
        OcrRunOptions = _OcrRunOptions  # type: ignore
        PADDLE_AVAILABLE = bool(_PADDLE_AVAILABLE)
        PADDLEOCR_AVAILABLE = bool(_PADDLEOCR_AVAILABLE)
        load_image_any = _load_image_any  # type: ignore
        ocr_photo_document = _ocr_photo_document  # type: ignore
        _perform_page_ocr_impl = _perform_page_ocr  # type: ignore
        return True
    except Exception:
        PADDLE_AVAILABLE = False
        PADDLEOCR_AVAILABLE = False
        load_image_any = None  # type: ignore
        ocr_photo_document = None  # type: ignore
        _perform_page_ocr_impl = None  # type: ignore
        return False

def _perform_page_ocr_tesseract(img_path: str, options, progress_cb=None) -> str:
    """Fallback OCR using pytesseract (no PaddleOCR).

    Supports basic modes used in UI:
    - options.ocr_mode: 'pro' | 'two_columns' | other
    - options.lang_mode: like 'deu+rus'
    - options.psm: int
    - options.preprocess_preset: 'none' or any -> try OpenCV preprocess if available
    """
    if progress_cb:
        try:
            progress_cb(0, 3, "Загрузка изображения")
        except Exception:
            pass

    lang = getattr(options, "lang_mode", DEFAULT_OCR_LANG) or DEFAULT_OCR_LANG
    psm = int(getattr(options, "psm", 6) or 6)
    preserve_spaces = bool(getattr(options, "preserve_spaces", True))
    base_cfg = f"--oem 1 --psm {psm}"
    if preserve_spaces:
        base_cfg += " -c preserve_interword_spaces=1"

    config, _, _ = _build_required_ocr_config(base_cfg)

    preprocess_preset = str(getattr(options, "preprocess_preset", "none") or "none")
    use_preprocess = preprocess_preset != "none"

    # Load/preprocess
    try:
        if use_preprocess and CV2_AVAILABLE and NUMPY_AVAILABLE:
            pil_img = preprocess_for_ocr(img_path)
        else:
            pil_img = load_image_for_ocr(img_path)
    except Exception:
        # As a last resort, try plain PIL open
        pil_img = load_image_for_ocr(img_path)

    mode = str(getattr(options, "ocr_mode", "standard") or "standard")
    if mode == "two_columns":
        left_img, right_img = split_two_columns(pil_img)
        if progress_cb:
            try:
                progress_cb(1, 3, "OCR левая колонка")
            except Exception:
                pass
        left_text = ocr_image(left_img, lang=lang, config=config)
        if progress_cb:
            try:
                progress_cb(2, 3, "OCR правая колонка")
            except Exception:
                pass
        right_text = ocr_image(right_img, lang=lang, config=config)
        result = (left_text or "").strip() + "\n\n" + (right_text or "").strip()
    else:
        if progress_cb:
            try:
                progress_cb(1, 3, "OCR")
            except Exception:
                pass
        result = ocr_image(pil_img, lang=lang, config=config)

    if progress_cb:
        try:
            progress_cb(3, 3, "Готово")
        except Exception:
            pass
    return (result or "").strip()


def _perform_page_ocr_easyocr(img_path: str, options, progress_cb=None) -> str:
    if not is_easyocr_available():
        raise RuntimeError("EasyOCR не установлен")
    if not NUMPY_AVAILABLE:
        raise RuntimeError("EasyOCR требует NumPy")

    if progress_cb:
        try:
            progress_cb(0, 3, "Загрузка изображения")
        except Exception:
            pass

    lang = getattr(options, "lang_mode", DEFAULT_OCR_LANG) or DEFAULT_OCR_LANG
    preprocess_preset = str(getattr(options, "preprocess_preset", "none") or "none")
    use_preprocess = preprocess_preset != "none"

    try:
        if use_preprocess and CV2_AVAILABLE and NUMPY_AVAILABLE:
            pil_img = preprocess_for_ocr(img_path)
        else:
            pil_img = load_image_for_ocr(img_path)
    except Exception:
        pil_img = load_image_for_ocr(img_path)

    reader = _get_easyocr_reader(lang)

    mode = str(getattr(options, "ocr_mode", "standard") or "standard")
    if mode == "two_columns":
        left_img, right_img = split_two_columns(pil_img)
        if progress_cb:
            try:
                progress_cb(1, 3, "OCR левая колонка")
            except Exception:
                pass
        left_text = "\n".join(reader.readtext(np.array(left_img), detail=0, paragraph=True))
        if progress_cb:
            try:
                progress_cb(2, 3, "OCR правая колонка")
            except Exception:
                pass
        right_text = "\n".join(reader.readtext(np.array(right_img), detail=0, paragraph=True))
        result = (left_text or "").strip() + "\n\n" + (right_text or "").strip()
    else:
        if progress_cb:
            try:
                progress_cb(1, 3, "OCR")
            except Exception:
                pass
        result = "\n".join(reader.readtext(np.array(pil_img), detail=0, paragraph=True))

    if progress_cb:
        try:
            progress_cb(3, 3, "Готово")
        except Exception:
            pass
    return (result or "").strip()

def perform_page_ocr(img_path: str, options, progress_cb=None) -> str:
    """Unified entrypoint used by UI: dispatches to selected OCR engine with safe fallbacks."""
    engine = str(getattr(options, "ocr_engine", "tesseract") or "tesseract").lower()

    if engine == "paddle":
        if not is_paddle_available():
            logging.warning("PaddleOCR unavailable, falling back to Tesseract.")
            return _perform_page_ocr_tesseract(img_path, options, progress_cb)
        if _ensure_ocr_photo_loaded() and _perform_page_ocr_impl is not None:
            try:
                return _perform_page_ocr_impl(img_path, options, progress_cb)
            except Exception:
                logging.exception("PaddleOCR failed, falling back to Tesseract.")
                return _perform_page_ocr_tesseract(img_path, options, progress_cb)
        logging.warning("PaddleOCR module not loaded, falling back to Tesseract.")
        return _perform_page_ocr_tesseract(img_path, options, progress_cb)

    if engine == "easyocr":
        try:
            return _perform_page_ocr_easyocr(img_path, options, progress_cb)
        except Exception:
            logging.exception("EasyOCR failed, falling back to Tesseract.")
            return _perform_page_ocr_tesseract(img_path, options, progress_cb)

    return _perform_page_ocr_tesseract(img_path, options, progress_cb)

# OpenAI key только в памяти
OPENAI_API_KEY = None

MIC_DEVICE_INDEX = None

DEFAULT_FRONT_TEMPLATE = "{sentence_with_gap}"
DEFAULT_BACK_TEMPLATE = "{word} [{ipa}] ({gender}; pl. {plural})\n\n{sentence}\n\n{translation}"
VIDEO_TEXT_TEMPLATE_NAME = "Видео + текст"
VIDEO_TEXT_FRONT_TEMPLATE = (
    "{question}<div class=\"media\">{front_video}{front_image}{front_audio}</div>"
    "<div class=\"txt\">{front_html}</div>"
)
VIDEO_TEXT_BACK_TEMPLATE = (
    "<div class=\"media\">{back_video}{back_image}{back_audio}</div>"
    "<div class=\"txt\">{back_html}</div>"
)
MEDIA_FOLDER = "media"
MEDIA_IMPORT_SUBDIR = "anki_import"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

CSV_COLUMN_ALIASES = {
    "id": ["id", "ID", "word_id", "uid"],
    "word": ["word", "de", "german", "term"],
    "translation": ["translation", "ru", "russian", "meaning"],
    "example": ["example", "sentence", "beispiel"],
    "level": ["level", "cefr"],
}


def _find_column(header: list[str], aliases: list[str]) -> int | None:
    lowered = [h.lower() for h in header]
    for alt in aliases:
        if alt.lower() in lowered:
            return lowered.index(alt.lower())
    return None


def read_csv_dictionary(path: str) -> dict[int, dict]:
    """Читает CSV-словарь и возвращает словарь id -> запись."""
    encodings = ["utf-8", "cp1251"]
    last_error = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return {}
                col_map = {}
                for key, variants in CSV_COLUMN_ALIASES.items():
                    idx = _find_column(header, variants)
                    if idx is not None:
                        col_map[key] = idx

                if "id" not in col_map:
                    raise ValueError("В CSV нет колонки с ID")

                records: dict[int, dict] = {}
                for row in reader:
                    try:
                        raw_id = row[col_map["id"]].strip()
                        if not raw_id:
                            continue
                        entry_id = int(re.search(r"\d+", raw_id).group(0))
                    except Exception:
                        continue

                    def get_field(name: str) -> str:
                        idx = col_map.get(name)
                        return row[idx].strip() if idx is not None and idx < len(row) else ""

                    records[entry_id] = {
                        "word": get_field("word"),
                        "translation": get_field("translation"),
                        "example": get_field("example"),
                        "level": get_field("level"),
                    }
                return records
        except Exception as e:
            last_error = e
            continue
    raise Exception(f"Не удалось прочитать CSV: {last_error}")


def extract_id_from_filename(filename: str) -> int | None:
    match = re.search(r"(\d+)", os.path.basename(filename))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def extract_id_with_ocr(image_path: str) -> int | None:
    if not (OCR_AVAILABLE and PIL_AVAILABLE and is_tesseract_available()):
        return None
    try:
        img = Image.open(image_path)
        gray = ImageOps.grayscale(img)
        enhanced = ImageOps.autocontrast(gray)
        config, _, _ = _build_required_ocr_config(DEFAULT_OCR_CONFIG_BASE)
        text = pytesseract.image_to_string(enhanced, lang=DEFAULT_OCR_LANG, config=config)
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
    except Exception:
        return None
    return None


def ensure_media_dir() -> str:
    os.makedirs(MEDIA_FOLDER, exist_ok=True)
    return MEDIA_FOLDER


def generate_sd_image_placeholder(prompt: str, out_path: str | None = None) -> str | None:
    ensure_media_dir()
    target_path = out_path or os.path.join(MEDIA_FOLDER, f"sd_placeholder_{uuid4().hex}.png")
    if not PIL_AVAILABLE:
        try:
            with open(target_path, "wb") as handle:
                handle.write(b"")
        except Exception:
            return None
        return target_path
    img = Image.new("RGB", (640, 360), color="#E0E7FF")
    draw = ImageDraw.Draw(img)
    text = "Stable Diffusion placeholder\n\n" + (prompt[:140] + "…" if len(prompt) > 140 else prompt)
    draw.text((20, 20), text, fill="#111827")
    try:
        img.save(target_path, format="PNG")
    except Exception:
        return None
    return target_path


def resolve_media_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = os.path.expanduser(path)
    if os.path.isabs(normalized) and os.path.exists(normalized):
        return normalized
    if os.path.exists(normalized):
        return normalized
    if normalized.startswith(MEDIA_FOLDER + os.sep):
        candidate = normalized
    else:
        candidate = os.path.join(MEDIA_FOLDER, normalized)
    if os.path.exists(candidate):
        return candidate
    basename_candidate = os.path.join(MEDIA_FOLDER, os.path.basename(normalized))
    if os.path.exists(basename_candidate):
        return basename_candidate
    return normalized


def _pil_lanczos():
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        return getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", 1))


def _ensure_image_caches(owner) -> None:
    if not hasattr(owner, "_tk_img_cache"):
        owner._tk_img_cache = {}
    if not hasattr(owner, "_orig_pil_cache"):
        owner._orig_pil_cache = {}
    if not hasattr(owner, "_orig_path_cache"):
        owner._orig_path_cache = {}


# PATCH: stop auto-shrinking image (no in-place thumbnail, orig cache + debounce Configure + fixed slot/min-size gate)
# PATCH: image rendering restored (Pillow resample fallback + PhotoImage cache + min-size debounce + shared render)
def render_image_safe(label, container_widget, image_path, key, zoom=None):
    if not image_path:
        label.config(image="", text="Нет изображения")
        label.image = None
        return False
    resolved_path = resolve_media_path(image_path) if image_path else None
    if resolved_path:
        resolved_path = os.path.abspath(resolved_path)
    exists = bool(resolved_path and os.path.exists(resolved_path))
    try:
        cont_w = int(container_widget.winfo_width())
        cont_h = int(container_widget.winfo_height())
    except Exception:
        cont_w = 0
        cont_h = 0
    fixed_slot = getattr(label, "_fixed_slot_size", None) or getattr(container_widget, "_fixed_slot_size", None)
    if fixed_slot:
        try:
            cont_w = max(1, int(fixed_slot[0]))
            cont_h = max(1, int(fixed_slot[1]))
        except Exception:
            pass
    if cont_w < 80 or cont_h < 80:
        prev_job = getattr(label, "_min_size_job", None)
        if prev_job:
            try:
                label.after_cancel(prev_job)
            except Exception:
                pass
        label._min_size_job = label.after(
            80,
            lambda: render_image_safe(label, container_widget, image_path, key, zoom=zoom),
        )
        return False
    if not exists:
        label.config(image="", text="Нет изображения")
        label.image = None
        return False
    _ensure_image_caches(label)
    if not hasattr(label, "_warned_large_path"):
        label._warned_large_path = None
    path_changed = label._orig_path_cache.get(key) != resolved_path
    orig = label._orig_pil_cache.get(key)
    if orig is None or path_changed:
        if PIL_AVAILABLE:
            try:
                img = Image.open(resolved_path)
                img = ImageOps.exif_transpose(img)
                if img.mode not in ("RGBA", "RGB"):
                    img = img.convert("RGBA")
                total_pixels = img.width * img.height
                resized_for_pixels = False
                if MAX_PREVIEW_PIXELS and total_pixels > MAX_PREVIEW_PIXELS:
                    scale = (MAX_PREVIEW_PIXELS / float(total_pixels)) ** 0.5
                    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                    img = img.resize(new_size, _pil_lanczos())
                    resized_for_pixels = True
                if resized_for_pixels and getattr(label, "_warned_large_path", None) != resolved_path:
                    try:
                        messagebox.showinfo(
                            "Большое изображение",
                            "Изображение слишком большое, будет сжато для превью.",
                        )
                    except Exception:
                        pass
                    label._warned_large_path = resolved_path
                label._orig_pil_cache[key] = img
                label._orig_path_cache[key] = resolved_path
                orig = img
            except Exception as exc:
                log_image_error(resolved_path, exc)
                if getattr(label, "_img_fail_logged", None) != resolved_path:
                    print(
                        "[IMG] fail:",
                        f"path={resolved_path}",
                        f"exists={exists}",
                        f"pil={PIL_AVAILABLE}",
                        f"err={exc}",
                    )
                    label._img_fail_logged = resolved_path
                label.config(image="", text="Нет изображения")
                label.image = None
                return False
        else:
            try:
                tk_img = tk.PhotoImage(file=resolved_path)
                label._tk_img_cache[key] = tk_img
                label.config(image=tk_img, text="")
                label.image = tk_img
                label.current_image = tk_img
                label.image = tk_img
                if not getattr(label, "_img_ok_logged", False):
                    print(
                        "[IMG] ok:",
                        f"{cont_w}x{cont_h} container -> {tk_img.width()}x{tk_img.height()} image",
                    )
                    label._img_ok_logged = True
                return True
            except Exception as exc:
                log_image_error(resolved_path, exc)
                if getattr(label, "_img_fail_logged", None) != resolved_path:
                    print(
                        "[IMG] fail:",
                        f"path={resolved_path}",
                        f"exists={exists}",
                        f"pil={PIL_AVAILABLE}",
                        f"err={exc}",
                    )
                    label._img_fail_logged = resolved_path
                label.config(image="", text="Нет изображения")
                label.image = None
                return False
    if not PIL_AVAILABLE:
        label.config(image="", text="Нет изображения")
        label.image = None
        return False
    if orig is None:
        label.config(image="", text="Нет изображения")
        label.image = None
        return False
    try:
        label.original_image = orig
        cont_w = max(1, cont_w - 6)
        cont_h = max(1, cont_h - 6)
        img_w, img_h = orig.size
        if img_w <= 0 or img_h <= 0:
            return False
        base_ratio = min(cont_w / img_w, cont_h / img_h) if cont_w and cont_h else 1.0
        zoom_factor = float(zoom) if zoom is not None else 1.0
        desired_ratio = max(0.05, base_ratio * zoom_factor)
        max_scale = getattr(label, "max_scale_factor", 3.0)
        clamped_ratio = max(0.05, min(desired_ratio, base_ratio * max_scale))
        width = max(1, int(img_w * clamped_ratio))
        height = max(1, int(img_h * clamped_ratio))
        min_w = 120
        min_h = 120
        if (width < min_w or height < min_h) and getattr(label, "current_image", None) is not None:
            return False
        pil_copy = orig.copy()
        resized_image = pil_copy.resize((width, height), _pil_lanczos())
        photo = ImageTk.PhotoImage(resized_image)
        label._tk_img_cache[key] = photo
        label.current_image = photo
        label.config(image=photo, text="")
        label.image = photo
        last_size = getattr(label, "_last_render_size", None)
        if last_size != (width, height):
            render_mode = getattr(label, "_render_mode", "unknown")
            card_id = getattr(label, "_render_card_id", None)
            print(
                "[IMG-RESIZE] mode=",
                render_mode,
                "card=",
                card_id,
                "slot=",
                cont_w,
                cont_h,
                "new=",
                width,
                height,
                "zoom=",
                zoom_factor,
            )
            label._last_render_size = (width, height)
        if not getattr(label, "_img_ok_logged", False):
            print(
                "[IMG] ok:",
                f"{cont_w}x{cont_h} container -> {width}x{height} image",
            )
            label._img_ok_logged = True
        return True
    except Exception as exc:
        log_image_error(resolved_path, exc)
        if getattr(label, "_img_fail_logged", None) != resolved_path:
            print(
                "[IMG] fail:",
                f"path={resolved_path}",
                f"exists={exists}",
                f"pil={PIL_AVAILABLE}",
                f"err={exc}",
            )
            label._img_fail_logged = resolved_path
        label.config(image="", text="Нет изображения")
        label.image = None
        return False


def render_image(label, container_widget, image_path, zoom, key):
    return render_image_safe(label, container_widget, image_path, key, zoom=zoom)


def copy_image_to_media(src: str, target_id: int, move: bool = False) -> str:
    ensure_media_dir()
    ext = os.path.splitext(src)[1].lower() or ".png"
    target_name = f"id_{target_id}{ext}"
    dest_path = os.path.join(MEDIA_FOLDER, target_name)
    try:
        if move:
            shutil.move(src, dest_path)
        else:
            shutil.copy2(src, dest_path)
    except Exception:
        dest_path = src
    return dest_path


def copy_image_asset_to_media(src: str, prefix: str = "img") -> str:
    ensure_media_dir()
    ext = os.path.splitext(src)[1].lower() or ".png"
    target_name = f"{prefix}_{uuid4().hex}{ext}"
    dest_path = os.path.join(MEDIA_FOLDER, target_name)
    try:
        shutil.copy2(src, dest_path)
    except Exception:
        dest_path = src
    return dest_path


def copy_video_asset_to_media(src: str, prefix: str = "video") -> str:
    ensure_media_dir()
    ext = os.path.splitext(src)[1].lower() or ".mp4"
    target_name = f"{prefix}_{uuid4().hex}{ext}"
    dest_path = os.path.join(MEDIA_FOLDER, target_name)
    try:
        shutil.copy2(src, dest_path)
    except Exception:
        dest_path = src
    return dest_path


def copy_audio_asset_to_media(src: str, prefix: str = "audio") -> str:
    ensure_media_dir()
    ext = os.path.splitext(src)[1].lower() or ".mp3"
    target_name = f"{prefix}_{uuid4().hex}{ext}"
    dest_path = os.path.join(MEDIA_FOLDER, target_name)
    try:
        shutil.copy2(src, dest_path)
    except Exception:
        dest_path = src
    return dest_path

# ==========================
# НАСТРОЙКИ ПЕРЕВОДА И СЛОВАРИ
# ==========================

class DictionaryManager:
    """Менеджер для работы с большими словарями"""
    
    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = {}  # Для быстрого обратного поиска
        self.dictionary_size = 0
        self.loaded_files = []
        
    def load_builtin_dictionary(self):
        """Загрузить встроенный базовый словарь"""
        basic_dict = {
            # Основные слова (пример - на практике будет 100,000)
            'Haus': 'дом', 'Buch': 'книга', 'Tisch': 'стол', 'Stuhl': 'стул',
            'Fenster': 'окно', 'Tür': 'дверь', 'Zimmer': 'комната', 'Küche': 'кухня',
            'Schlafzimmer': 'спальня', 'Badezimmer': 'ванная', 'Wohnzimmer': 'гостиная',
            'Schule': 'школа', 'Universität': 'университет', 'Arbeit': 'работа',
            'Mensch': 'человек', 'Frau': 'женщина', 'Mann': 'мужчина', 'Kind': 'ребенок',
            'Tag': 'день', 'Nacht': 'ночь', 'Morgen': 'утро', 'Abend': 'вечер',
            'Wasser': 'вода', 'Essen': 'еда', 'Brot': 'хлеб', 'Milch': 'молоко',
            'Apfel': 'яблоко', 'Kaffee': 'кофе', 'Tee': 'чай', 'Saft': 'сок',
            'rot': 'красный', 'blau': 'синий', 'grün': 'зеленый', 'gelb': 'желтый',
            'groß': 'большой', 'klein': 'маленький', 'gut': 'хороший', 'schlecht': 'плохой',
            'schnell': 'быстрый', 'langsam': 'медленный', 'warm': 'теплый', 'kalt': 'холодный',
            # Слова из примера на картинке
            'Abschied': 'прощание', 'von': 'от', 'Basel': 'Базель',
            'Leinen': 'леер', 'los': 'отчалить', 'tschüss': 'пока',
            'Schweiz': 'Швейцария', 'Carolina': 'Каролина', 'hat': 'имеет',
            'Osterferien': 'пасхальные каникулы', 'im': 'в', 'Moment': 'момент',
            'ist': 'есть', 'sie': 'она', 'noch': 'еще', 'in': 'в',
            'der': 'определенный артикль', 'aber': '但', 'bald': 'скоро',
            'wieder': 'снова', 'Deutschland': 'Германия', 'bei': 'у',
            'ihren': 'ее', 'Freunden': 'друзья'
        }
        self.dictionary.update(basic_dict)
        # Создаем обратный словарь
        for german, russian in basic_dict.items():
            self.reverse_dictionary[russian.lower()] = german
        self.dictionary_size = len(self.dictionary)
        
    def load_from_csv(self, filename):
        """Загрузить словарь из CSV файла (формат: немецкое слово, русский перевод)"""
        try:
            count = 0
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        german = row[0].strip()
                        russian = row[1].strip()
                        if german and russian:
                            self.dictionary[german] = russian
                            self.reverse_dictionary[russian.lower()] = german
                            count += 1
            self.dictionary_size = len(self.dictionary)
            self.loaded_files.append(filename)
            return count
        except Exception as e:
            raise Exception(f"Ошибка загрузки CSV: {e}")
    
    def load_from_json(self, filename):
        """Загрузить словарь из JSON файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for german, russian in data.items():
                        self.dictionary[german] = russian
                        self.reverse_dictionary[russian.lower()] = german
                self.dictionary_size = len(self.dictionary)
                self.loaded_files.append(filename)
                return len(data) if isinstance(data, dict) else 0
        except Exception as e:
            raise Exception(f"Ошибка загрузки JSON: {e}")
    
    def load_from_compressed(self, filename):
        """Загрузить словарь из сжатого файла (gzip)"""
        try:
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                if filename.endswith('.json.gz'):
                    data = json.load(f)
                elif filename.endswith('.csv.gz'):
                    reader = csv.reader(f)
                    data = {row[0]: row[1] for row in reader if len(row) >= 2}
                
                for german, russian in data.items():
                    self.dictionary[german] = russian
                    self.reverse_dictionary[russian.lower()] = german
                
                self.dictionary_size = len(self.dictionary)
                self.loaded_files.append(filename)
                return len(data)
        except Exception as e:
            raise Exception(f"Ошибка загрузки сжатого файла: {e}")
    
    def save_compressed_dictionary(self, filename):
        """Сохранить словарь в сжатом формате для быстрой загрузки"""
        try:
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                json.dump(self.dictionary, f, ensure_ascii=False)
            return True
        except Exception as e:
            raise Exception(f"Ошибка сохранения словаря: {e}")
    
    def get_translation(self, word):
        """Получить перевод слова (немецкое -> русское)"""
        # Пробуем точное совпадение
        if word in self.dictionary:
            return self.dictionary[word]
        
        # Пробуем регистронезависимо
        for german, russian in self.dictionary.items():
            if german.lower() == word.lower():
                return russian
        
        return None
    
    def get_reverse_translation(self, word):
        """Получить обратный перевод (русское -> немецкое)"""
        word_lower = word.lower().strip()
        return self.reverse_dictionary.get(word_lower)
    
    def get_statistics(self):
        """Получить статистику словаря"""
        return {
            'total_words': self.dictionary_size,
            'loaded_files': self.loaded_files,
            'memory_size_mb': len(pickle.dumps(self.dictionary)) / (1024 * 1024)
        }
    
    def search_words(self, pattern, limit=50):
        """Поиск слов по шаблону"""
        results = []
        pattern_lower = pattern.lower()
        for german, russian in self.dictionary.items():
            if pattern_lower in german.lower() or pattern_lower in russian.lower():
                results.append((german, russian))
                if len(results) >= limit:
                    break
        return results
    
    def export_to_csv(self, filename):
        """Экспортировать словарь в CSV (немецкое слово, русский перевод)"""
        try:
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for german, russian in sorted(self.dictionary.items()):
                    writer.writerow([german, russian])
            return True
        except Exception as e:
            raise Exception(f"Ошибка экспорта: {e}")


class TranslationSettings:
    def __init__(self):
        self.use_embedded_dict = True
        self.use_openai = True
        self.show_translations = True
        self.show_back_translation = True  # Показывать перевод на задней стороне
        self.dictionary_paths = []
        self.default_dictionary_path = "german_russian_dict.csv"
        
    def save(self):
        data = {
            'use_embedded_dict': self.use_embedded_dict,
            'use_openai': self.use_openai,
            'show_translations': self.show_translations,
            'show_back_translation': self.show_back_translation,
            'dictionary_paths': self.dictionary_paths,
            'default_dictionary_path': self.default_dictionary_path
        }
        with open('translation_settings.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        try:
            with open('translation_settings.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.use_embedded_dict = data.get('use_embedded_dict', True)
                self.use_openai = data.get('use_openai', True)
                self.show_translations = data.get('show_translations', True)
                self.show_back_translation = data.get('show_back_translation', True)
                self.dictionary_paths = data.get('dictionary_paths', [])
                self.default_dictionary_path = data.get('default_dictionary_path', "german_russian_dict.csv")
        except:
            pass

# Глобальные объекты
DICTIONARY_MANAGER = DictionaryManager()
TRANSLATION_SETTINGS = TranslationSettings()

def init_dictionary():
    """Инициализировать словарь"""
    # Загружаем встроенный базовый словарь
    DICTIONARY_MANAGER.load_builtin_dictionary()
    
    # Загружаем настройки
    TRANSLATION_SETTINGS.load()
    
    # Пробуем загрузить дефолтный словарь
    default_dict_path = TRANSLATION_SETTINGS.default_dictionary_path
    if os.path.exists(default_dict_path):
        try:
            count = DICTIONARY_MANAGER.load_from_csv(default_dict_path)
            print(f"Загружено {count} слов из дефолтного словаря: {default_dict_path}")
        except Exception as e:
            print(f"Ошибка загрузки дефолтного словаря {default_dict_path}: {e}")
    
    # Загружаем дополнительные словари из сохраненных путей
    for path in TRANSLATION_SETTINGS.dictionary_paths:
        if os.path.exists(path) and path != default_dict_path:
            try:
                if path.endswith('.csv'):
                    count = DICTIONARY_MANAGER.load_from_csv(path)
                    print(f"Загружено {count} слов из {path}")
                elif path.endswith('.json'):
                    count = DICTIONARY_MANAGER.load_from_json(path)
                    print(f"Загружено {count} слов из {path}")
                elif path.endswith(('.gz', '.zip')):
                    count = DICTIONARY_MANAGER.load_from_compressed(path)
                    print(f"Загружено {count} слов из {path}")
            except Exception as e:
                print(f"Ошибка загрузки словаря {path}: {e}")

def get_translation(word: str, use_openai: bool = True) -> str:
    """
    Получить перевод слова с использованием словаря и/или OpenAI.
    Возвращает русский перевод для немецкого слова.
    """
    word_original = word.strip()
    
    # Сначала пробуем словарь (немецкое -> русское)
    if TRANSLATION_SETTINGS.use_embedded_dict:
        translation = DICTIONARY_MANAGER.get_translation(word_original)
        if translation:
            return translation
    
    # Пробуем OpenAI если включено и есть ключ
    if TRANSLATION_SETTINGS.use_openai and use_openai and OPENAI_API_KEY and OPENAI_LIB_AVAILABLE:
        try:
            client = get_openai_client(OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты переводчик с немецкого на русский. Отвечай только переводом слова без пояснений."},
                    {"role": "user", "content": f"Переведи с немецкого на русский слово: {word_original}"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            translation = response.choices[0].message.content.strip()
            if translation and translation != word_original:
                # Сохраняем в словарь для будущего использования
                DICTIONARY_MANAGER.dictionary[word_original] = translation
                DICTIONARY_MANAGER.reverse_dictionary[translation.lower()] = word_original
                return translation
        except Exception:
            pass
    
    return ""  # Перевод не найден

def get_german_translation(word: str, use_openai: bool = True) -> str:
    """
    Получить немецкий перевод для русского слова.
    """
    word_lower = word.lower().strip()
    
    # Сначала пробуем обратный словарь
    if TRANSLATION_SETTINGS.use_embedded_dict:
        german_word = DICTIONARY_MANAGER.get_reverse_translation(word_lower)
        if german_word:
            return german_word
    
    # Пробуем OpenAI если включено и есть ключ
    if TRANSLATION_SETTINGS.use_openai and use_openai and OPENAI_API_KEY and OPENAI_LIB_AVAILABLE:
        try:
            client = get_openai_client(OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты переводчик с русского на немецкий. Отвечай только переводом слова без пояснений."},
                    {"role": "user", "content": f"Переведи с русского на немецкий слово: {word}"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            german_word = response.choices[0].message.content.strip()
            if german_word and german_word != word:
                # Сохраняем в словарь для будущего использования
                DICTIONARY_MANAGER.dictionary[german_word] = word
                DICTIONARY_MANAGER.reverse_dictionary[word_lower] = german_word
                return german_word
        except Exception:
            pass
    
    return ""  # Перевод не найден

def translate_sentence(sentence: str, use_openai: bool = True) -> str:
    """
    Перевести предложение с немецкого на русский.
    """
    # Сначала пробуем перевести каждое слово через словарь
    words = re.findall(r'\b\w+\b', sentence, re.UNICODE)
    translated_words = []
    
    for word in words:
        translation = get_translation(word, use_openai=False)  # Сначала без OpenAI
        if translation:
            translated_words.append(translation)
        else:
            translated_words.append(word)
    
    # Пробуем OpenAI для всего предложения если не удалось через словарь
    if TRANSLATION_SETTINGS.use_openai and use_openai and OPENAI_API_KEY and OPENAI_LIB_AVAILABLE:
        try:
            client = get_openai_client(OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты переводчик с немецкого на русский. Отвечай только переводом предложения без пояснений."},
                    {"role": "user", "content": f"Переведи с немецкого на русский предложение: {sentence}"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            full_translation = response.choices[0].message.content.strip()
            if full_translation and full_translation != sentence:
                return full_translation
        except Exception:
            pass
    
    # Возвращаем слово-за-слово перевод если OpenAI не сработал
    return " ".join(translated_words)

# ==========================
# ЛЕЙТНЕР (фазы)
# ==========================

LEITNER_SCHEDULE = {
    1: timedelta(seconds=30),
    2: timedelta(minutes=25),
    3: timedelta(hours=1),
    4: timedelta(days=1),
    5: timedelta(days=3),
    6: timedelta(days=9),
    7: timedelta(days=16),
    8: timedelta(days=36),
    9: timedelta(days=56),
    10: timedelta(days=100),
}

DEFAULT_USER_ID_FILE = Path(get_db_path()).with_name("user_id.txt")
USER_META_CURRENT_ID = "current_user_id"
TEXT_GEN_CREDIT_COST = 50
OCR_CREDIT_COST = 1
IMAGE_ID_IMPORT_COST = 5
WIKIMEDIA_IMPORT_COST = 5
WIKIMEDIA_TICKET_SIZE = 10
CARD_IMAGE_CREDIT_COST = 1
NOTES_PAGE_COST_BASIC = 49
NOTES_PAGE_COST_PRO = 25
SAFE_IMPORT_CHUNK_CHARS = DEFAULT_CHUNK_CHARS
SAFE_IMPORT_MAX_TOTAL_CHARS = DEFAULT_MAX_TOTAL_CHARS_SOFT
SAFE_IMPORT_MAX_PDF_PAGES = DEFAULT_MAX_PDF_PAGES_SOFT
ACTIVATION_MIN_HOURS = 24
ACTIVATION_MIN_CARDS = 10
ACTIVATION_MIN_REVIEWS = 20
OCR_PACK_SIZE = 1
OCR_POSTPROCESS_STEPS = 5
CARDS_GEN_PACK_SIZE = 25
VIDEO_GEN_PACK_SIZE = 20
PRICING_BY_PLAN = {
    "free": {
        "postprocess": 20,
        "ocr": 15,
        "ocr_cards_gen": 5,
        "cards_gen": 50,
        "video_gen": 100,
    },
    "pro": {
        "postprocess": 2,
        "ocr": 2,
        "ocr_cards_gen": 1,
        "cards_gen": 15,
        "video_gen": 65,
    },
    "premium": {
        "postprocess": 5,
        "ocr": 5,
        "ocr_cards_gen": 1,
        "cards_gen": 10,
        "video_gen": 65,
    },
}
PRICING_PACK_SIZES = {
    "cards_gen": CARDS_GEN_PACK_SIZE,
    "video_gen": VIDEO_GEN_PACK_SIZE,
}


def get_plan(user: dict | None) -> str:
    if not user:
        return "free"
    status = str(user.get("status") or "")
    if user.get("premium_plus") or status == "premium_plus":
        return "premium"
    if user.get("premium_active") or user.get("is_premium") or user.get("premium_until", 0) > int(time.time()):
        return "pro"
    return "free"


def get_cost(action: str, qty: int, plan: str) -> int:
    if qty <= 0:
        return 0
    plan_key = plan if plan in PRICING_BY_PLAN else "free"
    rates = PRICING_BY_PLAN[plan_key]
    unit_cost = int(rates.get(action, 0))
    if action in PRICING_PACK_SIZES:
        pack_size = PRICING_PACK_SIZES[action]
        packs = max(1, int(math.ceil(qty / float(pack_size))))
        return unit_cost * packs
    return unit_cost * max(1, qty)


def get_local_user_id() -> str:
    """Получить или создать локальный user_id (UUID4), сохранить в базе и на диске."""
    DEFAULT_USER_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at INTEGER,
            premium_until INTEGER DEFAULT 0,
            status TEXT DEFAULT 'обычный',
            verified INTEGER DEFAULT 0,
            starter_bonus_claimed INTEGER DEFAULT 0
        );
        """
    )
    cur.execute("SELECT value FROM user_meta WHERE key = ? LIMIT 1;", (USER_META_CURRENT_ID,))
    row = cur.fetchone()
    if row and row[0]:
        conn.close()
        return str(row[0])

    legacy_id = None
    if DEFAULT_USER_ID_FILE.exists():
        try:
            legacy_id = DEFAULT_USER_ID_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            legacy_id = None

    user_id = legacy_id or str(uuid4())
    cur.execute(
        "INSERT OR REPLACE INTO user_meta (key, value) VALUES (?, ?);",
        (USER_META_CURRENT_ID, user_id),
    )
    conn.commit()
    conn.close()
    try:
        DEFAULT_USER_ID_FILE.write_text(user_id, encoding="utf-8")
    except Exception:
        pass
    return user_id


def ensure_user_account(user_id: str) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO users (
            user_id, created_at, premium_until, status, verified, starter_bonus_claimed
        )
        VALUES (?, ?, 0, 'обычный', 0, 0);
        """,
        (user_id, int(time.time())),
    )
    conn.commit()
    cur.execute("SELECT * FROM users WHERE user_id = ? LIMIT 1;", (user_id,))
    row = cur.fetchone() or {}
    conn.close()
    return dict(row)


def update_user_account(user_id: str, **fields) -> dict:
    if not fields:
        return ensure_user_account(user_id)
    conn = get_connection()
    cur = conn.cursor()
    columns = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [user_id]
    cur.execute(f"UPDATE users SET {columns} WHERE user_id = ?;", values)
    conn.commit()
    conn.close()
    return ensure_user_account(user_id)


def ensure_user_profile_row(user_id: str) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO user_profile (
            user_id, created_at, is_premium, premium_until, premium_plus,
            starter_50_claimed, activation_200_claimed, wikimedia_tickets, first_import_ts
        )
        VALUES (?, ?, 0, 0, 0, 0, 0, 0, 0);
        """,
        (user_id, int(time.time())),
    )
    conn.commit()
    cur.execute("SELECT * FROM user_profile WHERE user_id = ? LIMIT 1;", (user_id,))
    row = cur.fetchone() or {}
    conn.close()
    return dict(row)


def update_user_profile(user_id: str, **fields) -> dict:
    if not fields:
        return ensure_user_profile_row(user_id)
    conn = get_connection()
    cur = conn.cursor()
    columns = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [user_id]
    cur.execute(f"UPDATE user_profile SET {columns} WHERE user_id = ?;", values)
    conn.commit()
    conn.close()
    return ensure_user_profile_row(user_id)


def get_next_review_for_level(level: int, deck_id: int | None = None) -> datetime:
    level = max(1, min(10, level))
    intervals = get_deck_phase_intervals(deck_id) if deck_id is not None else DEFAULT_PHASE_INTERVALS
    seconds = intervals[level - 1] if level - 1 < len(intervals) else DEFAULT_PHASE_INTERVALS[-1]
    return datetime.now() + timedelta(seconds=int(seconds))


# ==========================
# БАЗА ДАННЫХ
# ==========================

def get_connection():
    return open_db()


def ensure_basic_note_type_id(conn: sqlite3.Connection | None = None) -> int:
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    cur = conn.cursor()
    cur.execute("SELECT id FROM note_types WHERE name = 'Basic' LIMIT 1;")
    row = cur.fetchone()
    if row:
        note_type_id = row["id"]
    else:
        fields = ["word", "translation", "example", "level", "image"]
        templates = [
            {
                "name": "Word→Translation",
                "front": "{word}",
                "back": "{translation}\n\n{example}",
                "requires_image": False,
            },
            {
                "name": "Image→Word",
                "front": "{image}",
                "back": "{word}\n{translation}",
                "requires_image": True,
            },
        ]
        cur.execute(
            "INSERT OR IGNORE INTO note_types (name, fields_json, card_templates_json) VALUES (?, ?, ?);",
            ("Basic", json.dumps(fields, ensure_ascii=False), json.dumps(templates, ensure_ascii=False)),
        )
        conn.commit()
        cur.execute("SELECT id FROM note_types WHERE name = 'Basic' LIMIT 1;")
        note_type_id = cur.fetchone()["id"]

    if close_conn:
        conn.close()
    return note_type_id


def ensure_generated_note_type_id(conn: sqlite3.Connection | None = None) -> int:
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    cur = conn.cursor()
    cur.execute("SELECT id FROM note_types WHERE name = 'Generated' LIMIT 1;")
    row = cur.fetchone()
    if row:
        note_type_id = row["id"]
    else:
        fields = [
            "front",
            "back",
            "word",
            "translation",
            "example",
            "level",
            "image",
            "front_image_path",
            "back_image_path",
            "audio_path",
        ]
        templates = [
            {
                "name": "Front/Back",
                "front": "{front}",
                "back": "{back}",
                "requires_image": False,
            }
        ]
        cur.execute(
            "INSERT OR IGNORE INTO note_types (name, fields_json, card_templates_json) VALUES (?, ?, ?);",
            ("Generated", json.dumps(fields, ensure_ascii=False), json.dumps(templates, ensure_ascii=False)),
        )
        conn.commit()
        cur.execute("SELECT id FROM note_types WHERE name = 'Generated' LIMIT 1;")
        note_type_id = cur.fetchone()["id"]

    if close_conn:
        conn.close()
    return note_type_id


def _get_note_type(cur: sqlite3.Cursor, note_type_id: int):
    cur.execute(
        "SELECT id, name, fields_json, card_templates_json FROM note_types WHERE id = ?;",
        (note_type_id,),
    )
    return cur.fetchone()


def list_note_types() -> list[sqlite3.Row]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM note_types ORDER BY name;")
    rows = cur.fetchall()
    conn.close()
    return rows


def _extract_note_fields(card_row: dict, note_fields: dict | None = None) -> dict:
    source = dict(note_fields) if note_fields else {}

    def pick(*keys):
        for key in keys:
            if isinstance(key, (list, tuple)):
                for sub_key in key:
                    if source.get(sub_key):
                        return source.get(sub_key)
            elif source.get(key):
                return source.get(key)
        return None

    front_fallback = pick("word", "front", "_front")
    back_fallback = pick("translation", "back", "_back")
    example_fallback = pick("example", "front", "_front")

    fields = {
        "word": front_fallback or card_row.get("word") or card_row.get("target_word") or card_row.get("front") or card_row.get("_front") or "",
        "translation": back_fallback or card_row.get("translation") or card_row.get("back") or card_row.get("_back") or "",
        "example": example_fallback or card_row.get("example") or card_row.get("front") or card_row.get("_front") or "",
        "level": card_row.get("leitner_level", 1),
        "image": source.get("image")
        or card_row.get("image_path")
        or card_row.get("front_image_path")
        or card_row.get("back_image_path")
        or "",
    }
    return fields


def create_cards_from_note_templates(
    note_id: int,
    note_type_id: int,
    fields: dict,
    deck_id: int,
    skip_template_ords: set[int] | None = None,
    audio_path: str | None = None,
    audio_side: str = "back",
    audio_source: str | None = None,
    created_card_ids: list[int] | None = None,
):
    skip_template_ords = skip_template_ords or set()
    conn = get_connection()
    cur = conn.cursor()
    note_type_row = _get_note_type(cur, note_type_id)
    if not note_type_row:
        conn.close()
        return

    templates = json.loads(note_type_row["card_templates_json"])
    fields_default = collections.defaultdict(str, fields)
    template_fields = prepare_template_fields(fields_default)
    audio_value = audio_path if audio_path is not None else fields_default.get("audio_path")

    created = 0
    for ord_idx, template in enumerate(templates):
        if ord_idx in skip_template_ords:
            continue
        requires_image = bool(template.get("requires_image"))
        image_value = fields_default.get("image") or fields_default.get("front_image_path")
        if requires_image and not image_value:
            continue

        formatter = collections.defaultdict(str, template_fields)
        front = str(template.get("front", "")).format_map(formatter)
        back = str(template.get("back", "")).format_map(formatter)
        front_image_path = fields_default.get("front_image_path") or (image_value if requires_image else None)
        back_image_path = fields_default.get("back_image_path") or None

        card_id = insert_card(
            deck_id,
            front,
            back,
            front_image_path=front_image_path,
            back_image_path=back_image_path,
            audio_path=audio_value,
            level=1,
            note_id=note_id,
            template_ord=ord_idx,
            audio_side=audio_side,
            audio_source=audio_source,
        )
        created += 1
        if created_card_ids is not None and card_id:
            created_card_ids.append(int(card_id))

    conn.close()
    return created


def ensure_note_for_card(
    card_row: sqlite3.Row | dict,
    note_fields: dict | None = None,
    create_cards: bool = False,
    skip_template_ords: set[int] | None = None,
) -> dict:
    card_dict = dict(card_row)
    if card_dict.get("note_id"):
        return card_dict

    conn = get_connection()
    note_type_id = ensure_basic_note_type_id(conn)
    cur = conn.cursor()
    fields = _extract_note_fields(card_dict, note_fields)
    cur.execute(
        """
        INSERT INTO notes (deck_id, note_type_id, fields_json, tags, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            card_dict.get("deck_id"),
            note_type_id,
            json.dumps(fields, ensure_ascii=False),
            "",
            int(time.time()),
        ),
    )
    note_id = cur.lastrowid
    cur.execute(
        "UPDATE cards SET note_id = ?, template_ord = ? WHERE id = ?;",
        (note_id, 0, card_dict.get("id")),
    )
    conn.commit()
    conn.close()

    card_dict["note_id"] = note_id
    card_dict["template_ord"] = 0

    if create_cards:
        skip = skip_template_ords if skip_template_ords is not None else {0}
        create_cards_from_note_templates(
            note_id,
            note_type_id,
            fields,
            card_dict.get("deck_id"),
            skip_template_ords=skip,
        )

    return card_dict


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1;",
        (table,),
    )
    return cur.fetchone() is not None


def log_imported_file(path: str):
    conn = get_connection()
    if not _table_exists(conn, "import_log"):
        conn.close()
        return
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO import_log (file_path, imported_at) VALUES (?, ?);",
        (path, int(time.time())),
    )
    conn.commit()
    conn.close()


def get_imported_files() -> set[str]:
    conn = get_connection()
    if not _table_exists(conn, "import_log"):
        conn.close()
        return set()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM import_log;")
    files = {row[0] for row in cur.fetchall()}
    conn.close()
    return files


def _get_media_columns(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("PRAGMA table_info(media);")
    return {row[1] for row in cursor.fetchall()}


def ensure_media_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    if not _table_exists(conn, "media"):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY,
                card_id INTEGER,
                note_id INTEGER,
                type TEXT NOT NULL,
                path TEXT NOT NULL,
                side TEXT NOT NULL DEFAULT 'back',
                source TEXT,
                created_at INTEGER NOT NULL
            );
            """
        )
        conn.commit()
        return

    columns = _get_media_columns(cur)
    if "card_id" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN card_id INTEGER;")
    if "note_id" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN note_id INTEGER;")
    if "type" not in columns and "media_type" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN type TEXT;")
    elif "type" not in columns and "media_type" in columns:
        cur.execute("ALTER TABLE media ADD COLUMN type TEXT;")
    if "side" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN side TEXT NOT NULL DEFAULT 'back';")
    if "source" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN source TEXT;")
    conn.commit()


def ensure_media_state_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    if not _table_exists(conn, "media_state"):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS media_state (
                card_id INTEGER NOT NULL,
                media_key TEXT NOT NULL,
                pos_ms INTEGER DEFAULT 0,
                volume REAL DEFAULT 70,
                speed REAL DEFAULT 1.0,
                PRIMARY KEY (card_id, media_key)
            );
            """
        )
        conn.commit()


def _media_type_column(columns: set[str]) -> str:
    return "media_type" if "media_type" in columns else "type"


def insert_media(
    conn: sqlite3.Connection,
    *,
    card_id: int | None = None,
    note_id: int | None = None,
    type: str,
    path: str,
    side: str = "back",
    source: str | None = None,
    created_at: int | None = None,
):
    if card_id is None and note_id is None:
        raise ValueError("Не указан card_id или note_id для медиа")

    ensure_media_table(conn)
    cur = conn.cursor()
    columns = _get_media_columns(cur)
    ts = int(created_at if created_at is not None else time.time())
    type_col = _media_type_column(columns)

    cols = ["note_id", "card_id", type_col, "path", "created_at"]
    vals = [note_id, card_id, type, path, ts]
    placeholders = ["?", "?", "?", "?", "?"]

    if "side" in columns:
        cols.append("side")
        vals.append(side or "back")
        placeholders.append("?")
    if "source" in columns:
        cols.append("source")
        vals.append(source)
        placeholders.append("?")

    cur.execute(
        f"INSERT INTO media ({', '.join(cols)}) VALUES ({', '.join(placeholders)});",
        vals,
    )


def attach_media_to_note(note_id: int, media_entries: list[tuple[str | None, str, str | None, str | None]]):
    ts = int(time.time())

    def write():
        ensure_media_table(conn)
        for entry in media_entries:
            if len(entry) == 2:
                path, media_type = entry
                side, source = "back", None
            elif len(entry) == 3:
                path, media_type, side = entry
                source = None
            else:
                path, media_type, side, source = entry
            if not path:
                continue
            insert_media(
                conn,
                note_id=note_id,
                type=media_type,
                path=path,
                side=side or "back",
                source=source,
                created_at=ts,
            )

    with open_db() as conn:
        try:
            commit_with_retry(conn, write)
        except Exception as e:
            conn.rollback()
            messagebox.showerror("Ошибка сохранения медиа", f"{type(e).__name__}: {e}")


def attach_media_to_card(card_id: int, media_entries: list[tuple[str | None, str, str | None, str | None]]):
    ts = int(time.time())

    def write():
        ensure_media_table(conn)
        for entry in media_entries:
            if len(entry) == 2:
                path, media_type = entry
                side, source = "back", None
            elif len(entry) == 3:
                path, media_type, side = entry
                source = None
            else:
                path, media_type, side, source = entry
            if not path:
                continue
            insert_media(
                conn,
                card_id=card_id,
                type=media_type,
                path=path,
                side=side or "back",
                source=source,
                created_at=ts,
            )

    with open_db() as conn:
        try:
            commit_with_retry(conn, write)
        except Exception as e:
            conn.rollback()
            messagebox.showerror("Ошибка сохранения медиа", f"{type(e).__name__}: {e}")


def get_media_for_card(card_id: int, note_id: int | None = None) -> list[dict]:
    conn = get_connection()
    if not _table_exists(conn, "media"):
        conn.close()
        return []

    cur = conn.cursor()
    columns = _get_media_columns(cur)
    clauses = []
    params: list[int] = []
    if "card_id" in columns:
        clauses.append("card_id = ?")
        params.append(card_id)
    if note_id is not None and "note_id" in columns:
        clauses.append("note_id = ?")
        params.append(note_id)

    if not clauses:
        conn.close()
        return []

    cur.execute(f"SELECT * FROM media WHERE {' OR '.join(clauses)};", params)
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _build_media_key(media_id: int | None, path: str | None) -> str:
    if media_id is not None:
        return f"id:{media_id}"
    if path:
        return f"path:{path}"
    return "unknown"


def load_media_state(card_id: int, media_key: str) -> dict:
    conn = get_connection()
    ensure_media_state_table(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pos_ms, volume, speed
        FROM media_state
        WHERE card_id = ? AND media_key = ?;
        """,
        (card_id, media_key),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return {}
    return {
        "pos_ms": row["pos_ms"],
        "volume": row["volume"],
        "speed": row["speed"],
    }


def save_media_state(card_id: int, media_key: str, pos_ms: int, volume: float, speed: float):
    conn = get_connection()
    ensure_media_state_table(conn)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO media_state (card_id, media_key, pos_ms, volume, speed)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(card_id, media_key)
        DO UPDATE SET pos_ms = excluded.pos_ms, volume = excluded.volume, speed = excluded.speed;
        """,
        (card_id, media_key, int(pos_ms), float(volume), float(speed)),
    )
    conn.commit()
    conn.close()


SOUND_TAG_PATTERN = re.compile(r"\[sound:([^\]]+)\]")


def _normalize_sound_name(sound_name: str) -> str:
    cleaned = sound_name.replace("\\", os.sep).replace("/", os.sep)
    return os.path.normpath(cleaned)


def resolve_sound_file(sound_name: str) -> str | None:
    cleaned = sound_name.strip()
    if cleaned.lower().startswith("[sound:") and cleaned.endswith("]"):
        cleaned = cleaned[len("[sound:") : -1]

    normalized = _normalize_sound_name(cleaned)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if os.path.isabs(normalized):
        candidates.append(normalized)
    else:
        candidates.extend(
            [
                os.path.join(base_dir, normalized),
                os.path.join(base_dir, MEDIA_FOLDER, normalized),
                os.path.join(base_dir, MEDIA_FOLDER, MEDIA_IMPORT_SUBDIR, normalized),
                normalized,
                os.path.join(MEDIA_FOLDER, normalized),
                os.path.join(MEDIA_FOLDER, MEDIA_IMPORT_SUBDIR, normalized),
                os.path.join(MEDIA_FOLDER, MEDIA_IMPORT_SUBDIR, os.path.basename(normalized)),
            ]
        )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def extract_audio_from_text(text: str, side: str) -> tuple[str, list[dict]]:
    matches = SOUND_TAG_PATTERN.findall(text or "")
    cleaned = SOUND_TAG_PATTERN.sub("", text or "")
    entries: list[dict] = []
    for sound_name in matches:
        resolved = resolve_sound_file(sound_name)
        entries.append(
            {
                "path": resolved,
                "name": os.path.basename(sound_name) or sound_name,
                "original": sound_name,
                "side": side,
                "missing": resolved is None,
            }
        )
    return cleaned, entries


def sanitize_card_sounds(card: dict) -> dict:
    if card.get("_sound_sanitized"):
        return card

    inline_entries: list[dict] = []
    for side_key in ("front", "back"):
        cleaned, entries = extract_audio_from_text(card.get(side_key, ""), side_key)
        card[side_key] = cleaned.strip()
        inline_entries.extend(entries)

    card["_inline_audio"] = inline_entries
    card["_sound_sanitized"] = True

    existing_entries = get_media_for_card(card.get("id"), card.get("note_id"))
    existing_paths = {
        entry.get("path")
        for entry in existing_entries
        if (entry.get("media_type") or entry.get("type") or "").lower() == "audio"
    }

    to_attach: list[tuple[str | None, str, str | None, str | None]] = []
    for entry in inline_entries:
        path = entry.get("path")
        if path and path not in existing_paths:
            to_attach.append((path, "audio", entry.get("side") or "front", "anki_inline"))
            existing_paths.add(path)

    if to_attach:
        if card.get("id"):
            attach_media_to_card(card.get("id"), to_attach)
        elif card.get("note_id"):
            attach_media_to_note(card.get("note_id"), to_attach)

    return card


def get_card_audio_entries(card: dict, prefer_side: str | None = None) -> list[dict]:
    sanitize_card_sounds(card)
    preferred = prefer_side.lower() if prefer_side else None
    entries: list[dict] = []
    seen: set[tuple] = set()

    def add_entry(path: str | None, label: str, side: str | None, source: str | None, missing: bool, media_id=None):
        key = (path or label, side or "back", missing)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "path": path,
                "label": f"{label} ({(side or 'back')})" + (" [нет файла]" if missing else ""),
                "side": side or "back",
                "source": source,
                "missing": missing,
                "media_id": media_id,
            }
        )

    for item in get_media_for_card(card.get("id"), card.get("note_id")):
        media_type = (item.get("media_type") or item.get("type") or "").lower()
        if media_type != "audio":
            continue
        path = item.get("path")
        missing = not (path and os.path.exists(path))
        label = os.path.basename(path) if path else "audio"
        add_entry(path, label, item.get("side"), item.get("source"), missing, item.get("id"))

    fallback = card.get("audio_path")
    if fallback:
        add_entry(
            fallback,
            os.path.basename(fallback) or "audio",
            card.get("audio_side", "back"),
            card.get("audio_source"),
            not os.path.exists(fallback),
            None,
        )

    for entry in card.get("_inline_audio", []):
        label = entry.get("name") or entry.get("original") or "audio"
        add_entry(entry.get("path"), label, entry.get("side"), "inline_sound", entry.get("missing", False), None)

    def sort_key(item: dict):
        side = (item.get("side") or "").lower()
        missing = bool(item.get("missing"))
        return (0 if preferred and side == preferred else 1, 0 if not missing else 1)

    entries.sort(key=sort_key)
    return entries


def get_card_audio_path(card: dict, prefer_side: str | None = "back") -> str | None:
    for entry in get_card_audio_entries(card, prefer_side=prefer_side):
        if entry.get("path") and not entry.get("missing"):
            return entry.get("path")
    return None


def update_media_side(media_id: int, side: str):
    with open_db() as conn:
        if not _table_exists(conn, "media"):
            return
        cur = conn.cursor()
        columns = _get_media_columns(cur)
        if "side" not in columns:
            return

        def write():
            cur.execute("UPDATE media SET side = ? WHERE id = ?;", (side, media_id))

        commit_with_retry(conn, write)


def remove_media_entry(media_id: int):
    with open_db() as conn:
        if not _table_exists(conn, "media"):
            return
        cur = conn.cursor()

        def write():
            cur.execute("DELETE FROM media WHERE id = ?;", (media_id,))

        commit_with_retry(conn, write)


def display_audio_entries_on_frame(audio_frame, entries: list[dict]):
    audio_widget = getattr(audio_frame, "audio_widget", None)
    if audio_widget is None:
        audio_widget = AudioPlayerWidget(audio_frame)
        audio_frame.audio_widget = audio_widget
        audio_widget.pack(fill=tk.X)

    selector_frame = getattr(audio_frame, "audio_selector_frame", None)
    if selector_frame is None:
        audio_bg = audio_frame.cget("bg") if hasattr(audio_frame, "cget") else "white"
        selector_frame = tk.Frame(audio_frame, bg=audio_bg)
        selector_frame.pack(fill=tk.X, pady=(0, 2))
        tk.Label(selector_frame, text="Аудиофайлы:", bg=audio_bg).pack(side=tk.LEFT, padx=(0, 5))
        audio_frame.audio_selector_var = tk.StringVar()
        combo = ttk.Combobox(
            selector_frame,
            textvariable=audio_frame.audio_selector_var,
            state="readonly",
        )
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        audio_frame.audio_selector = combo
        audio_frame.audio_selector_frame = selector_frame

    combo: ttk.Combobox = getattr(audio_frame, "audio_selector", None)
    if not entries:
        audio_widget.load(None)
        audio_widget.pack_forget()
        if combo:
            combo.set("")
            selector_frame.pack_forget()
        return

    selector_frame.pack(fill=tk.X, pady=(0, 2))
    audio_widget.pack(fill=tk.X)
    entry_map = {entry.get("label"): entry for entry in entries}
    audio_frame.audio_entry_map = entry_map
    labels = list(entry_map.keys())
    combo.config(values=labels)

    def on_select(*_args):
        selected = audio_frame.audio_selector_var.get()
        entry = audio_frame.audio_entry_map.get(selected)
        if not entry:
            return
        if entry.get("path") and not entry.get("missing"):
            audio_widget.load(entry.get("path"))
            audio_widget._set_status(entry.get("label"))
        else:
            audio_widget.load(None)
            audio_widget._set_status(f"{entry.get('label')} - аудио не найдено")

    combo.bind("<<ComboboxSelected>>", on_select)

    first_label = labels[0]
    audio_frame.audio_selector_var.set(first_label)
    if len(labels) == 1:
        combo.state(["disabled"])
    else:
        combo.state(["!disabled"])
    on_select()


def find_video_media_path(card: dict) -> str | None:
    media_entries = get_media_for_card(card.get("id"), card.get("note_id"))
    for entry in media_entries:
        media_type = (entry.get("media_type") or entry.get("type") or "").lower()
        path = entry.get("path")
        if media_type == "video" and path and os.path.exists(path):
            return path
    return None


def find_video_media_path_for_side(card: dict, side: str) -> str | None:
    media_entries = get_media_for_card(card.get("id"), card.get("note_id"))
    side_key = (side or "back").lower()
    for entry in media_entries:
        media_type = (entry.get("media_type") or entry.get("type") or "").lower()
        if media_type != "video":
            continue
        entry_side = (entry.get("side") or "back").lower()
        if entry_side != side_key:
            continue
        path = resolve_media_path(entry.get("path"))
        if path and os.path.exists(path):
            return path
    return None


def get_side_media(side_data: dict) -> tuple[str, str] | None:
    image_path = resolve_media_path(side_data.get("image_path"))
    if image_path and os.path.exists(image_path):
        return ("image", image_path)
    video_path = resolve_media_path(side_data.get("video_path"))
    if video_path and os.path.exists(video_path):
        return ("video", video_path)
    return None


def create_note(
    deck_id: int,
    fields: dict,
    note_type_id: int | None = None,
    tags: str = "",
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    note_type = note_type_id or ensure_generated_note_type_id(conn)
    cur.execute(
        """
        INSERT INTO notes (deck_id, note_type_id, fields_json, tags, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            deck_id,
            note_type,
            json.dumps(fields, ensure_ascii=False),
            tags,
            int(time.time()),
        ),
    )
    note_id = cur.lastrowid
    conn.commit()
    conn.close()
    return note_id


def create_note_with_cards(
    deck_id: int,
    fields: dict,
    note_type_id: int | None = None,
    tags: str = "",
    skip_template_ords: set[int] | None = None,
    created_card_ids: list[int] | None = None,
) -> tuple[int, int]:
    note_id = create_note(deck_id, fields, note_type_id=note_type_id, tags=tags)
    note_type = note_type_id or ensure_generated_note_type_id()
    audio_side = fields.get("audio_side", "back") if isinstance(fields, dict) else "back"
    audio_source = fields.get("audio_source") if isinstance(fields, dict) else None
    cards_created = create_cards_from_note_templates(
        note_id,
        note_type,
        fields,
        deck_id,
        skip_template_ords=skip_template_ords,
        audio_path=fields.get("audio_path"),
        audio_side=audio_side,
        audio_source=audio_source,
        created_card_ids=created_card_ids,
    )
    media_entries = [
        (fields.get("image") or fields.get("front_image_path"), "image", "front", None),
        (fields.get("back_image_path"), "image", "back", None),
        (fields.get("audio_path"), "audio", audio_side, audio_source),
        (fields.get("video_path"), "video", fields.get("video_side") or "front", None),
    ]
    attach_media_to_note(note_id, media_entries)
    return note_id, cards_created


def _get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return {row["name"] for row in cur.fetchall()}


def spend_credits_in_transaction(
    conn: sqlite3.Connection,
    user_id: str,
    amount: int,
    reason: str,
    meta: dict | None = None,
) -> None:
    if amount <= 0:
        return
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO credits_balance (user_id, balance) VALUES (?, 0);",
        (user_id,),
    )
    cur.execute("SELECT balance FROM credits_balance WHERE user_id = ?;", (user_id,))
    row = cur.fetchone()
    balance = int(row[0]) if row else 0
    if balance < amount:
        raise ValueError("insufficient credits")
    cur.execute(
        "UPDATE credits_balance SET balance = balance - ? WHERE user_id = ?;",
        (amount, user_id),
    )
    columns = _get_table_columns(conn, "credits_ledger")
    meta_column = "meta" if "meta" in columns else "meta_json"
    cur.execute(
        f"""
        INSERT INTO credits_ledger (user_id, ts, delta, reason, {meta_column})
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            user_id,
            int(time.time()),
            -abs(amount),
            reason,
            json.dumps(meta or {}, ensure_ascii=False),
        ),
    )


def create_note_with_cards_in_transaction(
    conn: sqlite3.Connection,
    deck_id: int,
    fields: dict,
    note_type_id: int | None = None,
    tags: str = "",
    skip_template_ords: set[int] | None = None,
    created_card_ids: list[int] | None = None,
) -> tuple[int, int]:
    skip_template_ords = skip_template_ords or set()
    note_type = note_type_id or ensure_generated_note_type_id(conn)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO notes (deck_id, note_type_id, fields_json, tags, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            deck_id,
            note_type,
            json.dumps(fields, ensure_ascii=False),
            tags,
            int(time.time()),
        ),
    )
    note_id = cur.lastrowid
    note_type_row = _get_note_type(cur, note_type)
    if not note_type_row:
        raise RuntimeError("Тип заметки не найден")
    templates = json.loads(note_type_row["card_templates_json"])
    fields_default = collections.defaultdict(str, fields)
    template_fields = prepare_template_fields(fields_default)
    audio_value = fields_default.get("audio_path")
    front_rich = fields_default.get("front_rich")
    back_rich = fields_default.get("back_rich")

    created = 0
    for ord_idx, template in enumerate(templates):
        if ord_idx in skip_template_ords:
            continue
        requires_image = bool(template.get("requires_image"))
        image_value = fields_default.get("image") or fields_default.get("front_image_path")
        if requires_image and not image_value:
            continue
        formatter = collections.defaultdict(str, template_fields)
        front = str(template.get("front", "")).format_map(formatter)
        back = str(template.get("back", "")).format_map(formatter)
        front_image_path = fields_default.get("front_image_path") or (image_value if requires_image else None)
        back_image_path = fields_default.get("back_image_path") or None
        next_dt = get_next_review_for_level(1, deck_id).isoformat()
        cur.execute(
            """
            INSERT INTO cards (
                deck_id, front, back, next_review, leitner_level,
                front_image_path, back_image_path, front_rich, back_rich, audio_path,
                note_id, template_ord, translation_shown, overview_added
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1);
            """,
            (
                deck_id,
                front,
                back,
                next_dt,
                1,
                front_image_path,
                back_image_path,
                serialize_rich_doc(front_rich),
                serialize_rich_doc(back_rich),
                audio_value,
                note_id,
                ord_idx,
            ),
        )
        card_id = cur.lastrowid
        created += 1
        if created_card_ids is not None:
            created_card_ids.append(int(card_id))

    return note_id, created


def ensure_notes_for_cards(cards: list[sqlite3.Row]) -> list[dict]:
    sanitized: list[dict] = []
    for card in cards:
        card_dict = ensure_note_for_card(card)
        card_dict["front_rich"] = deserialize_rich_doc(card_dict.get("front_rich"))
        card_dict["back_rich"] = deserialize_rich_doc(card_dict.get("back_rich"))
        sanitized.append(sanitize_card_sounds(card_dict))
    return sanitized


def find_note_by_source_id(deck_id: int, source_id: int) -> sqlite3.Row | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM notes WHERE deck_id = ?;", (deck_id,))
    rows = cur.fetchall()
    for row in rows:
        try:
            fields = json.loads(row["fields_json"])
        except Exception:
            continue
        if str(fields.get("source_id")) == str(source_id):
            conn.close()
            return row
    conn.close()
    return None


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    os.makedirs(MEDIA_FOLDER, exist_ok=True)

    # Колоды с сохранением шаблонов FRONT/BACK и иконкой
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            front_template TEXT,
            back_template TEXT,
            selected_gen_template_id INTEGER,
            icon_path TEXT,
            tts_lang TEXT
        );
    """)

    # миграция для старых БД: добавляем колонки шаблонов, если их нет
    for col in ("front_template", "back_template", "selected_gen_template_id", "icon_path", "tts_lang"):
        try:
            if col == "selected_gen_template_id":
                cur.execute(f"ALTER TABLE decks ADD COLUMN {col} INTEGER;")
            else:
                cur.execute(f"ALTER TABLE decks ADD COLUMN {col} TEXT;")
        except sqlite3.OperationalError:
            # колонка уже существует
            pass

    # Шаблоны генерации
    cur.execute("""
        CREATE TABLE IF NOT EXISTS generation_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            question TEXT NOT NULL DEFAULT '',
            front_template TEXT NOT NULL,
            back_template TEXT NOT NULL,
            created_at INTEGER,
            updated_at INTEGER
        );
    """)

    # Карточки
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deck_id INTEGER NOT NULL,
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            next_review TEXT NOT NULL,
            leitner_level INTEGER NOT NULL DEFAULT 1,
            front_image_path TEXT,
            back_image_path TEXT,
            front_rich TEXT,
            back_rich TEXT,
            image_path TEXT,
            audio_path TEXT,
            note_id INTEGER,
            template_ord INTEGER,
            progress INTEGER NOT NULL DEFAULT 0,
            translation_shown BOOLEAN DEFAULT 1,
            overview_added BOOLEAN DEFAULT 0,
            state TEXT,
            due INTEGER,
            interval INTEGER,
            ease INTEGER,
            reps INTEGER,
            lapses INTEGER,
            step_index INTEGER,
            last_review INTEGER,
            FOREIGN KEY (deck_id) REFERENCES decks(id)
        );
    """)

    ensure_due_column(conn)

    # Миграции для старых БД (cards)
    for col in ("leitner_level", "front_image_path", "back_image_path",
                "front_rich", "back_rich", "image_path", "audio_path",
                "progress", "translation_shown", "overview_added"):
        try:
            if col == "leitner_level":
                cur.execute(
                    f"ALTER TABLE cards ADD COLUMN {col} INTEGER NOT NULL DEFAULT 1;"
                )
            elif col == "progress":
                cur.execute(
                    f"ALTER TABLE cards ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0;"
                )
            elif col == "translation_shown" or col == "overview_added":
                cur.execute(
                    f"ALTER TABLE cards ADD COLUMN {col} BOOLEAN DEFAULT 0;"
                )
            else:
                cur.execute(f"ALTER TABLE cards ADD COLUMN {col} TEXT;")
        except sqlite3.OperationalError:
            # колонка уже существует
            pass

    # Словарь уже известных слов
    cur.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL UNIQUE
        );
    """)

    # Статистика для диаграмм
    cur.execute("""
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            deck_id INTEGER,
            remembered_count INTEGER DEFAULT 0,
            forgotten_count INTEGER DEFAULT 0,
            reviewed_count INTEGER DEFAULT 0,
            UNIQUE(date, deck_id)
        );
    """)

    # Статистика ознакомления
    cur.execute("""
        CREATE TABLE IF NOT EXISTS overview_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            deck_id INTEGER NOT NULL,
            overview_count INTEGER DEFAULT 0,
            UNIQUE(date, deck_id)
        );
    """)

    ensure_deck_settings_table(conn)
    ensure_stats_settings_table(conn)
    ensure_media_table(conn)
    ensure_media_state_table(conn)

    run_migrations(conn)
    ensure_schema_for_import(conn)

    conn.commit()
    conn.close()


def list_decks():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, description, icon_path FROM decks ORDER BY id;")
    decks = cur.fetchall()
    conn.close()
    return decks


def get_deck_templates(deck_id: int):
    """
    Загрузить шаблоны FRONT/BACK для конкретной колоды.
    Если в БД пусто — вернуть шаблоны по умолчанию.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT front_template, back_template FROM decks WHERE id = ?;",
        (deck_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row:
        front = row["front_template"]
        back = row["back_template"]
        return front or DEFAULT_FRONT_TEMPLATE, back or DEFAULT_BACK_TEMPLATE
    return DEFAULT_FRONT_TEMPLATE, DEFAULT_BACK_TEMPLATE


def save_deck_templates(deck_id: int, front_template: str, back_template: str):
    """
    Сохранить шаблоны FRONT/BACK для конкретной колоды.
    Это и есть «обучение» генератора по твоим шаблонам.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE decks SET front_template = ?, back_template = ? WHERE id = ?;",
            (front_template, back_template, deck_id)
        )
    except sqlite3.OperationalError:
        # На всякий случай, если колонок нет (очень старая БД)
        pass
    conn.commit()
    conn.close()


def ensure_default_generation_template(deck_id: int | None = None) -> int | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM generation_templates ORDER BY id LIMIT 1;")
    row = cur.fetchone()
    if row:
        conn.close()
        return row["id"]

    if deck_id is not None:
        front_template, back_template = get_deck_templates(deck_id)
    else:
        front_template, back_template = DEFAULT_FRONT_TEMPLATE, DEFAULT_BACK_TEMPLATE
    now = int(time.time())
    cur.execute(
        """
        INSERT INTO generation_templates (name, question, front_template, back_template, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        ("По умолчанию", "", front_template, back_template, now, now),
    )
    template_id = cur.lastrowid
    conn.commit()
    conn.close()
    return template_id


def ensure_video_text_generation_template() -> int | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM generation_templates WHERE name = ? LIMIT 1;", (VIDEO_TEXT_TEMPLATE_NAME,))
    row = cur.fetchone()
    if row:
        conn.close()
        return row["id"]
    now = int(time.time())
    cur.execute(
        """
        INSERT INTO generation_templates (name, question, front_template, back_template, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            VIDEO_TEXT_TEMPLATE_NAME,
            "",
            VIDEO_TEXT_FRONT_TEMPLATE,
            VIDEO_TEXT_BACK_TEMPLATE,
            now,
            now,
        ),
    )
    template_id = cur.lastrowid
    conn.commit()
    conn.close()
    return template_id


def list_generation_templates() -> list[sqlite3.Row]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, question, front_template, back_template FROM generation_templates ORDER BY name;"
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_generation_template(template_id: int) -> sqlite3.Row | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, question, front_template, back_template FROM generation_templates WHERE id = ?;",
        (template_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def update_generation_template(template_id: int, name: str, question: str, front: str, back: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE generation_templates
        SET name = ?, question = ?, front_template = ?, back_template = ?, updated_at = ?
        WHERE id = ?;
        """,
        (name, question, front, back, int(time.time()), template_id),
    )
    conn.commit()
    conn.close()


def upsert_generation_template(name: str, question: str, front: str, back: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        """
        INSERT INTO generation_templates (name, question, front_template, back_template, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            question = excluded.question,
            front_template = excluded.front_template,
            back_template = excluded.back_template,
            updated_at = excluded.updated_at;
        """,
        (name, question, front, back, now, now),
    )
    cur.execute("SELECT id FROM generation_templates WHERE name = ?;", (name,))
    row = cur.fetchone()
    conn.commit()
    conn.close()
    return row["id"] if row else 0


def delete_generation_template(template_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE decks SET selected_gen_template_id = NULL WHERE selected_gen_template_id = ?;",
        (template_id,),
    )
    cur.execute("DELETE FROM generation_templates WHERE id = ?;", (template_id,))
    conn.commit()
    conn.close()


def get_deck_selected_template_id(deck_id: int) -> int | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT selected_gen_template_id FROM decks WHERE id = ?;", (deck_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return row["selected_gen_template_id"]
    return None


def set_deck_selected_template_id(deck_id: int, template_id: int | None) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE decks SET selected_gen_template_id = ? WHERE id = ?;",
        (template_id, deck_id),
    )
    conn.commit()
    conn.close()


def get_active_generation_template(deck_id: int | None) -> dict[str, str | int | None]:
    if deck_id is None:
        return {
            "id": None,
            "name": "По умолчанию",
            "question": "",
            "front_template": DEFAULT_FRONT_TEMPLATE,
            "back_template": DEFAULT_BACK_TEMPLATE,
        }
    template_id = get_deck_selected_template_id(deck_id)
    if template_id:
        row = get_generation_template(template_id)
        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "question": row["question"],
                "front_template": row["front_template"],
                "back_template": row["back_template"],
            }
    front, back = get_deck_templates(deck_id)
    return {
        "id": None,
        "name": "По умолчанию",
        "question": "",
        "front_template": front,
        "back_template": back,
    }


def get_deck_icon_path(deck_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT icon_path FROM decks WHERE id = ?;", (deck_id,))
    row = cur.fetchone()
    conn.close()
    return row["icon_path"] if row else None


def set_deck_icon_path(deck_id: int, icon_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE decks SET icon_path = ? WHERE id = ?;",
        (icon_path, deck_id)
    )
    conn.commit()
    conn.close()


def get_cards_in_deck(deck_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, front_rich, back_rich, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
        ORDER BY id;
    """, (deck_id,))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def get_due_cards(deck_id):
    """Карточки, у которых дата повторения ≤ сегодня."""
    today = date.today().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, front_rich, back_rich, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
          AND date(next_review) <= date(?)
        ORDER BY next_review, id;
    """, (deck_id, today))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def get_cards_for_repeat(deck_id):
    """
    Режим повторения:
    все карточки колоды, но первыми идут те,
    у которых дата повторения уже наступила.
    """
    today = date.today().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, front_rich, back_rich, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
        ORDER BY
            CASE WHEN date(next_review) <= date(?) THEN 0 ELSE 1 END,
            date(next_review),
            id;
    """, (deck_id, today))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def get_cards_for_playback(deck_id):
    """
    Режим воспроизведения:
    все карточки, отсортированы по прогрессу (меньше — раньше),
    затем по дате повторения.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, front_rich, back_rich, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
        ORDER BY progress ASC,
                 date(next_review) ASC,
                 id ASC;
    """, (deck_id,))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def get_overview_cards(deck_id):
    """
    Получить все карточки для режима ознакомления.
    В режим ознакомления попадают ВСЕ карточки колоды.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, front_rich, back_rich, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
        ORDER BY id;
    """, (deck_id,))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def mark_card_for_overview(card_id: int):
    """Пометить карточку как добавленную в режим ознакомления"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards
        SET overview_added = 1
        WHERE id = ?;
    """, (card_id,))
    conn.commit()
    conn.close()


def update_card_leitner(card_id: int, new_level: int):
    new_level = max(1, min(10, new_level))
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT deck_id FROM cards WHERE id = ?;", (card_id,))
    row = cur.fetchone()
    deck_id = row["deck_id"] if row else None
    next_dt = get_next_review_for_level(new_level, deck_id).isoformat()
    cur.execute("""
        UPDATE cards
           SET leitner_level = ?, next_review = ?
         WHERE id = ?;
    """, (new_level, next_dt, card_id))
    conn.commit()
    conn.close()


def get_card_by_id(card_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cards WHERE id = ?;", (card_id,))
    row = cur.fetchone()
    conn.close()
    return ensure_note_for_card(row) if row else None


def apply_srs_update(card_id: int, rating: int):
    row = get_card_by_id(card_id)
    if not row:
        return None
    card_data = dict(row)
    card_data.setdefault("phase", card_data.get("leitner_level", 1))
    now_ts = int(time.time())
    result = schedule_review(card_data, rating, now_ts)
    next_review_value = get_next_review_for_level(result["phase"], card_data.get("deck_id")).isoformat()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE cards
           SET state = ?, due = ?, interval = ?, ease = ?, reps = ?, lapses = ?,
               step_index = ?, last_review = ?, leitner_level = ?, next_review = ?
         WHERE id = ?;
        """,
        (
            result["state"], result["due"], result["interval"], result["ease"],
            result["reps"], result["lapses"], result["step_index"],
            result["last_review"], result["phase"], next_review_value, card_id
        ),
    )
    cur.execute(
        """
        INSERT INTO reviews (
            card_id, reviewed_at, rating,
            interval_before, interval_after,
            ease_before, ease_after,
            phase_before, phase_after
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            card_id,
            now_ts,
            rating,
            result.get("interval_before"),
            result.get("interval_after"),
            result.get("ease_before"),
            result.get("ease_after"),
            result.get("phase_before"),
            result.get("phase_after"),
        ),
    )
    conn.commit()
    conn.close()
    return result


def update_card_progress(card_id: int, new_progress: int):
    new_progress = max(0, min(100, new_progress))
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards
           SET progress = ?
         WHERE id = ?;
    """, (new_progress, card_id))
    conn.commit()
    conn.close()


def update_card_translation_shown(card_id: int, shown: bool):
    """Обновить состояние показа перевода для карточки"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards
           SET translation_shown = ?
         WHERE id = ?;
    """, (1 if shown else 0, card_id))
    conn.commit()
    conn.close()


def delete_card(card_id: int):
    """Удаление карточки из базы."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM cards WHERE id = ?;", (card_id,))
    conn.commit()
    conn.close()


def count_overdue_for_deck(deck_id: int) -> int:
    """Кол-во просроченных карточек по колонке ``due`` (с учётом подколод)."""

    counts = fetch_overdue_counts_by_phase(None, deck_id)
    return counts.total


def get_deck_stats(deck_id: int):
    """
    Получить статистику по колоде:
    - Общее количество карточек
    - Количество карточек по фазам (1-10)
    - Процент изученности (progress > 80%)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Общее количество карточек
    cur.execute("SELECT COUNT(*) FROM cards WHERE deck_id = ?", (deck_id,))
    total = cur.fetchone()[0]
    
    # Карточки по фазам
    phase_stats = {}
    for phase in range(1, 11):
        cur.execute(
            "SELECT COUNT(*) FROM cards WHERE deck_id = ? AND leitner_level = ?",
            (deck_id, phase)
        )
        phase_stats[phase] = cur.fetchone()[0]
    
    # Процент изученности
    cur.execute(
        "SELECT COUNT(*) FROM cards WHERE deck_id = ? AND progress >= 80",
        (deck_id,)
    )
    learned_count = cur.fetchone()[0]
    learned_percent = (learned_count / total * 100) if total > 0 else 0
    
    # Статистика ознакомления
    cur.execute(
        """SELECT SUM(overview_count) as total_overview 
           FROM overview_statistics 
           WHERE deck_id = ?""",
        (deck_id,)
    )
    row = cur.fetchone()
    total_overview = row["total_overview"] or 0 if row else 0
    
    conn.close()
    
    return {
        "total": total,
        "phase_stats": phase_stats,
        "learned_percent": learned_percent,
        "learned_count": learned_count,
        "total_overview": total_overview
    }


# ==========================
# Статистика
# ==========================

def update_statistics(deck_id: int, remembered: bool = False, forgotten: bool = False, reviewed: bool = False):
    """
    Обновить статистику для диаграмм.
    remembered: True если нажата кнопка "Помню"
    forgotten: True если нажата кнопка "Забыл"
    reviewed: True если карточка просмотрена (любая)
    """
    today = date.today().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, есть ли запись на сегодня
    cur.execute(
        "SELECT id, remembered_count, forgotten_count, reviewed_count FROM statistics WHERE date = ? AND deck_id = ?",
        (today, deck_id)
    )
    row = cur.fetchone()
    
    if row:
        # Обновляем существующую запись
        rem_count = row["remembered_count"] + (1 if remembered else 0)
        forg_count = row["forgotten_count"] + (1 if forgotten else 0)
        rev_count = row["reviewed_count"] + (1 if reviewed else 0)
        
        cur.execute("""
            UPDATE statistics 
            SET remembered_count = ?, forgotten_count = ?, reviewed_count = ?
            WHERE id = ?
        """, (rem_count, forg_count, rev_count, row["id"]))
    else:
        # Создаем новую запись
        cur.execute("""
            INSERT INTO statistics (date, deck_id, remembered_count, forgotten_count, reviewed_count)
            VALUES (?, ?, ?, ?, ?)
        """, (today, deck_id, 
              1 if remembered else 0, 
              1 if forgotten else 0,
              1 if reviewed else 0))
    
    conn.commit()
    conn.close()


def update_overview_statistics(deck_id: int, increment: int = 1):
    """
    Обновить статистику ознакомления.
    increment: +1 при нажатии "следующий", -1 при нажатии "назад"
    """
    today = date.today().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, есть ли запись на сегодня
    cur.execute(
        "SELECT id, overview_count FROM overview_statistics WHERE date = ? AND deck_id = ?",
        (today, deck_id)
    )
    row = cur.fetchone()
    
    if row:
        # Обновляем существующую запись
        new_count = max(0, row["overview_count"] + increment)
        cur.execute("""
            UPDATE overview_statistics 
            SET overview_count = ?
            WHERE id = ?
        """, (new_count, row["id"]))
    else:
        if increment > 0:
            # Создаем новую запись только если increment положительный
            cur.execute("""
                INSERT INTO overview_statistics (date, deck_id, overview_count)
                VALUES (?, ?, ?)
            """, (today, deck_id, increment))
    
    conn.commit()
    conn.close()


def get_statistics_for_dates(deck_id: int, date_list: list[date]):
    """Получить статистику для заданного списка дат (date)."""
    if not date_list:
        return {"dates": [], "remembered": [], "forgotten": [], "reviewed": [], "overview": []}

    iso_dates = [d.isoformat() for d in date_list]
    conn = get_connection()
    cur = conn.cursor()

    placeholders = ",".join("?" * len(iso_dates))

    cur.execute(
        f"""
        SELECT date, remembered_count, forgotten_count, reviewed_count
        FROM statistics
        WHERE deck_id = ? AND date IN ({placeholders})
        """,
        [deck_id, *iso_dates],
    )
    rows = cur.fetchall()

    cur.execute(
        f"""
        SELECT date, overview_count
        FROM overview_statistics
        WHERE deck_id = ? AND date IN ({placeholders})
        """,
        [deck_id, *iso_dates],
    )
    overview_rows = cur.fetchall()
    conn.close()

    remembered_data = collections.OrderedDict((d, 0) for d in iso_dates)
    forgotten_data = collections.OrderedDict((d, 0) for d in iso_dates)
    reviewed_data = collections.OrderedDict((d, 0) for d in iso_dates)
    overview_data = collections.OrderedDict((d, 0) for d in iso_dates)

    for row in rows:
        date_str = row["date"]
        if date_str in remembered_data:
            remembered_data[date_str] = row["remembered_count"]
            forgotten_data[date_str] = row["forgotten_count"]
            reviewed_data[date_str] = row["reviewed_count"]

    for row in overview_rows:
        date_str = row["date"]
        if date_str in overview_data:
            overview_data[date_str] = row["overview_count"]

    return {
        "dates": list(remembered_data.keys()),
        "remembered": list(remembered_data.values()),
        "forgotten": list(forgotten_data.values()),
        "reviewed": list(reviewed_data.values()),
        "overview": list(overview_data.values()),
    }


def get_statistics_for_period(deck_id: int, days: int = 30):
    """
    Получить статистику за последние N дней.
    Возвращает словарь с датами и значениями.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)

    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)

    return get_statistics_for_dates(deck_id, date_range)


def get_monthly_summary(deck_id: int):
    """
    Получить сводку за текущий месяц.
    """
    today = date.today()
    first_day = date(today.year, today.month, 1)
    
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            SUM(remembered_count) as total_remembered,
            SUM(forgotten_count) as total_forgotten,
            SUM(reviewed_count) as total_reviewed
        FROM statistics 
        WHERE deck_id = ? AND date BETWEEN ? AND ?
    """, (deck_id, first_day.isoformat(), today.isoformat()))
    
    row = cur.fetchone()
    
    # Получаем статистики ознакомления
    cur.execute("""
        SELECT SUM(overview_count) as total_overview
        FROM overview_statistics 
        WHERE deck_id = ? AND date BETWEEN ? AND ?
    """, (deck_id, first_day.isoformat(), today.isoformat()))
    
    overview_row = cur.fetchone()
    conn.close()
    
    total_overview = overview_row["total_overview"] or 0 if overview_row else 0
    
    return {
        "total_remembered": row["total_remembered"] or 0,
        "total_forgotten": row["total_forgotten"] or 0,
        "total_reviewed": row["total_reviewed"] or 0,
        "total_overview": total_overview,
        "success_rate": (row["total_remembered"] or 0) / max(1, (row["total_reviewed"] or 1)) * 100
    }


# ==========================
# Работа со словами
# ==========================

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def normalize_word(w: str) -> str:
    return w.strip().lower()


def extract_words_from_text(text: str) -> list:
    return [normalize_word(m.group(0)) for m in WORD_RE.finditer(text)]


def auto_punctuate(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n")]
    cleaned_lines: list[str] = []
    for line in lines:
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        if line and line[0].isalpha():
            line = line[0].upper() + line[1:]
        if not re.search(r"[.!?]$", line):
            line += "."
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def split_into_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def get_known_words() -> set:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT text FROM words;")
    rows = cur.fetchall()
    conn.close()
    return {r["text"] for r in rows}


def add_new_words(words: set):
    if not words:
        return
    conn = get_connection()
    cur = conn.cursor()
    for w in words:
        try:
            cur.execute("INSERT OR IGNORE INTO words (text) VALUES (?);", (w,))
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()


def get_repeated_word_flag(word: str) -> str:
    normalized = normalize_word(word or "")
    if not normalized:
        return "—"
    conn = get_connection()
    cur = conn.cursor()
    like_pattern = f"%{normalized}%"
    try:
        cur.execute("PRAGMA table_info(cards);")
        columns = {row[1] for row in cur.fetchall()}
        phase_expr = "phase" if "phase" in columns else "leitner_level"
        cur.execute(
            f"""
            SELECT MAX(COALESCE({phase_expr}, leitner_level, 1)) as phase_level
            FROM cards
            WHERE lower(front) = ?
               OR lower(back) = ?
               OR lower(front) LIKE ?
               OR lower(back) LIKE ?
            """,
            (normalized, normalized, like_pattern, like_pattern),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row or row[0] is None:
        return "—"
    phase_val = int(row[0] or 0)
    if phase_val >= 2:
        return "повторено"
    return "нет"


def is_repeated_word(word: str) -> bool:
    return get_repeated_word_flag(word) == "повторено"


def is_word_repeated(word: str) -> bool:
    return is_repeated_word(word)


def calc_digital_ear_cost(sentences: list[str], plan: str) -> tuple[int, list[int], dict]:
    plan_key = "pro" if plan in ("pro", "premium") else "free"
    if plan_key == "pro":
        base_cost = 2
        new_word_cost = 1
    else:
        base_cost = 6
        new_word_cost = 4
    per_card_costs: list[int] = []
    meta_cards: list[dict] = []
    for sentence in sentences:
        words = extract_words_from_text(sentence)
        has_new_word = any(not is_word_repeated(w) for w in words) if words else False
        card_cost = new_word_cost if has_new_word else base_cost
        per_card_costs.append(card_cost)
        meta_cards.append(
            {
                "sentence": sentence,
                "has_new_word": has_new_word,
                "cost": card_cost,
            }
        )
    total_cost = int(sum(per_card_costs))
    meta = {
        "plan": plan_key,
        "total_cards": len(sentences),
        "per_card": meta_cards,
        "pricing": {"base": base_cost, "new_word": new_word_cost},
    }
    return total_cost, per_card_costs, meta


def mask_text_by_repeated_words(text: str) -> tuple[str, list[str]]:
    hidden_words: list[str] = []

    def replace_word(match: re.Match) -> str:
        word = match.group(0)
        if is_repeated_word(word):
            return word
        hidden_words.append(word)
        return "—"

    masked = re.sub(r"\w+", replace_word, text)
    seen: set[str] = set()
    unique_hidden: list[str] = []
    for word in hidden_words:
        key = normalize_word(word)
        if key and key not in seen:
            seen.add(key)
            unique_hidden.append(word)
    return masked, unique_hidden


def _apply_foreign_template_defaults(ctx: dict[str, str]) -> dict[str, str]:
    sentence = ctx.get("sentence", "") or ""
    if sentence:
        masked_text, hidden_words = mask_text_by_repeated_words(sentence)
        if not ctx.get("sentence_with_gap"):
            ctx["sentence_with_gap"] = masked_text
        if not ctx.get("translation"):
            ctx["translation"] = ", ".join(hidden_words) if hidden_words else "—"
        if not ctx.get("word"):
            ctx["word"] = hidden_words[0] if hidden_words else "—"
    for key in ("sentence_with_gap", "translation", "word"):
        if not ctx.get(key):
            ctx[key] = "—"
    return ctx


def _render_generation_template_faces(
    template: dict[str, str | int | None],
    ctx: dict[str, str] | None = None,
) -> tuple[str, str]:
    base_ctx = dict(ctx or {})
    template_name = str(template.get("name") or "").strip().lower()
    question = str(template.get("question") or "")
    front_template = str(template.get("front_template") or "")
    back_template = str(template.get("back_template") or "")

    front_ctx = dict(base_ctx)
    back_ctx = dict(base_ctx)
    for key in ("translation", "sentence_with_gap", "word", "ipa", "gender", "plural", "sentence"):
        if f"front_{key}" in base_ctx:
            front_ctx[key] = base_ctx.get(f"front_{key}", "") or ""
        if f"back_{key}" in base_ctx:
            back_ctx[key] = base_ctx.get(f"back_{key}", "") or ""

    for key in ("translation", "sentence_with_gap", "word", "ipa", "gender", "plural", "sentence"):
        front_ctx.setdefault(key, "")
        back_ctx.setdefault(key, "")

    if template_name == "иностранные слова":
        front_ctx = _apply_foreign_template_defaults(front_ctx)
        back_ctx = _apply_foreign_template_defaults(back_ctx)

    front_formatter = collections.defaultdict(str, front_ctx)
    back_formatter = collections.defaultdict(str, back_ctx)
    front_formatter["question"] = question
    back_formatter["question"] = question

    formatted_front = front_template.format_map(front_formatter)
    formatted_back = back_template.format_map(back_formatter)

    if "{question}" in front_template:
        front = formatted_front
    else:
        if question and formatted_front:
            front = f"{question}\n{formatted_front}"
        elif question:
            front = question
        else:
            front = formatted_front
    return front, formatted_back


def render_generation_faces(deck_id: int, ctx: dict[str, str] | None = None) -> tuple[str, str]:
    template = get_active_generation_template(deck_id)
    return _render_generation_template_faces(template, ctx)


# ==========================
# Wiktionary
# ==========================

def get_wiktionary_data(word: str) -> dict:
    """
    Тянем базовую инфу с de.wiktionary.org:
    ipa, род, мн.ч, примерная таблица форм, синонимы, примеры.
    Если что-то не получается / нет модулей – возвращаем пустые поля.
    """
    data = {
        "ipa": "",
        "gender": "",
        "plural": "",
        "declension": "",
        "synonyms": [],
        "examples": [],
    }
    if not WIKTIONARY_AVAILABLE:
        return data

    try:
        url = f"https://de.wiktionary.org/wiki/{word}"
        headers = {"User-Agent": "Mozilla/5.0 (anki-clone)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return data
        soup = BeautifulSoup(resp.text, "html.parser")

        # IPA
        ipa_span = soup.find("span", class_="ipa")
        if ipa_span:
            data["ipa"] = ipa_span.get_text(strip=True)

        # Род / мн. число: ищем в первых таблицах флексии
        table = soup.find("table", class_="wikitable")
        if table:
            txt = table.get_text("\n", strip=True)
            data["declension"] = txt
            lower = txt.lower()
            if "genus" in lower:
                for line in txt.splitlines():
                    if "Genus" in line:
                        data["gender"] = line.replace("Genus", "").strip(": ").strip()
                        break
            if "plural" in lower:
                for line in txt.splitlines():
                    if "Plural" in line:
                        data["plural"] = line.replace("Plural", "").strip(": ").strip()
                        break

        # Синонимы
        syn_head = soup.find(id="Synonyme")
        if syn_head:
            ul = syn_head.find_next("ul")
            if ul:
                for li in ul.find_all("li", recursive=False):
                    text = li.get_text(" ", strip=True)
                    if text:
                        data["synonyms"].append(text)

        # Примеры
        ex_head = soup.find(id="Beispiele") or soup.find(id="Beispiel")
        if ex_head:
            ul = ex_head.find_next("ul")
            if ul:
                for li in ul.find_all("li", recursive=False):
                    text = li.get_text(" ", strip=True)
                    if text:
                        data["examples"].append(text)

    except Exception:
        return data

    return data


# ==========================
# OpenAI функции
# ==========================

def get_openai_client(api_key: str):
    if not OPENAI_LIB_AVAILABLE:
        raise RuntimeError("Библиотека 'openai' не установлена")
    if not api_key:
        raise RuntimeError("OpenAI API key не задан")
    return OpenAI(api_key=api_key)


def generate_image_with_openai(prompt: str, api_key: str, save_dir: str = "ai_images") -> str:
    os.makedirs(save_dir, exist_ok=True)
    client = get_openai_client(api_key)

    try:
        img = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard",
        )
    except Exception as e:
        msg = str(e)
        if "billing_hard_limit_reached" in msg or "Billing hard limit has been reached" in msg:
            raise RuntimeError(
                "На аккаунте OpenAI исчерпан платёжный лимит (billing hard limit).\n"
                "AI-картинки временно недоступны.\n"
                "Пополните баланс или укажите другой API-ключ."
            ) from e
        raise

    image_url = img.data[0].url
    import requests
    response = requests.get(image_url)
    image_bytes = response.content

    filename = f"ai_{int(time.time())}.png"
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)

    return path


def enrich_german_word_info(word: str, api_key: str | None):
    """
    Базовая инфа (перевод + IPA/род/мн.ч) через GPT,
    если ключ есть. Иначе – заглушки.
    IPA/род/мн.ч потом будут поверх заменены wiktionary, если удастся.
    """
    if not api_key or not OPENAI_LIB_AVAILABLE:
        return "", "", "?", "?"

    try:
        client = get_openai_client(api_key)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты словарный бот по немецкому как Wiktionary. "
                        "Для заданного немецкого слова дай ответ в формате:\n"
                        "translation_ru | ipa | gender | plural\n"
                        "gender один из: m, f, n.\n"
                        "Только одна строка, без пояснений."
                    ),
                },
                {"role": "user", "content": word},
            ],
            max_tokens=64,
        )
        line = resp.choices[0].message.content.strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            return "", "", "?", "?"
        return parts[0], parts[1], parts[2], parts[3]
    except Exception:
        return "", "", "?", "?"


# ==========================
# Вставка карточки
# ==========================
def insert_card(
    deck_id: int,
    front: str,
    back: str,
    front_image_path: str | None = None,
    back_image_path: str | None = None,
    front_rich: dict | None = None,
    back_rich: dict | None = None,
    audio_path: str | None = None,
    level: int = 1,
    note_id: int | None = None,
    template_ord: int | None = None,
    ensure_note: bool = False,
    note_fields: dict | None = None,
    audio_side: str = "back",
    audio_source: str | None = None,
):
    """
    Вставка карточки в БД.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Если deck_id не задан — берём первую колоду
    if deck_id is None:
        cur.execute("SELECT id FROM decks ORDER BY id LIMIT 1;")
        row = cur.fetchone()
        if row is None:
            conn.close()
            raise RuntimeError("Не выбрана колода...")
        deck_id = row["id"]

    next_dt = get_next_review_for_level(level, deck_id).isoformat()

    # ВАЖНО: Проверяем, есть ли аудио-тег в тексте карточки
    # Если audio_path передан, используем его
    # Если нет, проверяем тег [audio:...] в back тексте
    
    actual_audio_path = audio_path
    if not actual_audio_path and "[audio:" in back:
        # Извлечь путь из тега [audio:path/to/file.wav]
        match = re.search(r'\[audio:(.+?)\]', back)
        if match:
            actual_audio_path = match.group(1)
            # Удалить тег из текста для чистого отображения
            back = re.sub(r'\[audio:.+?\]', '', back).strip()

    cur.execute(
        """
        INSERT INTO cards (deck_id, front, back, next_review, leitner_level,
                           front_image_path, back_image_path, front_rich, back_rich, audio_path,
                           note_id, template_ord, translation_shown, overview_added)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1);
    """,
        (
            deck_id,
            front,
            back,
            next_dt,
            level,
            front_image_path,
            back_image_path,
            serialize_rich_doc(front_rich),
            serialize_rich_doc(back_rich),
            actual_audio_path,
            note_id,
            template_ord,
        ),
    )

    card_id = cur.lastrowid

    stored_audio_path = actual_audio_path
    if actual_audio_path and os.path.exists(actual_audio_path):
        os.makedirs(MEDIA_FOLDER, exist_ok=True)
        ext = os.path.splitext(actual_audio_path)[1] or ".wav"
        filename = f"audio_card_{card_id}_{int(time.time())}{ext}"
        dest_path = os.path.join(MEDIA_FOLDER, filename)
        try:
            shutil.copy(actual_audio_path, dest_path)
            stored_audio_path = dest_path
            cur.execute("UPDATE cards SET audio_path = ? WHERE id = ?;", (stored_audio_path, card_id))
        except Exception:
            stored_audio_path = actual_audio_path

    conn.commit()
    conn.close()

    if stored_audio_path:
        attach_media_to_card(card_id, [(stored_audio_path, "audio", audio_side, audio_source)])

    if ensure_note and note_id is None:
        card_stub = {
            "id": card_id,
            "deck_id": deck_id,
            "front": front,
            "back": back,
            "leitner_level": level,
            "front_image_path": front_image_path,
            "back_image_path": back_image_path,
            "image_path": None,
        }
        ensure_note_for_card(
            card_stub,
            note_fields=note_fields,
            create_cards=True,
            skip_template_ords={template_ord} if template_ord is not None else {0},
        )

    return card_id

# ==========================
# Авто-генерация
# ==========================

def build_local_cards_from_text(
    deck_id: int,
    text: str,
    one_sentence_one_card: bool = False,
) -> tuple[list[dict], set[str]]:
    sentences = split_into_sentences(text)
    if not sentences:
        return [], set()

    known = get_known_words()
    cards: list[dict] = []
    all_new_words: set[str] = set()

    def make_base_card(sentence: str, target_word: str):
        sentence_with_gap = sentence
        if target_word:
            pattern = re.compile(re.escape(target_word), re.IGNORECASE)
            sentence_with_gap = pattern.sub("____", sentence, count=1)

        translation = get_translation(target_word, use_openai=False) if target_word else ""
        sentence_translation = translate_sentence(sentence, use_openai=False)

        front, back = render_generation_faces(
            deck_id,
            {
                "translation": sentence_translation,
                "front_translation": "",
                "sentence_with_gap": sentence_with_gap,
                "word": target_word or "",
                "ipa": "",
                "gender": "?",
                "plural": "?",
                "sentence": sentence,
            },
        )

        cards.append(
            {
                "front": front,
                "back": back,
                "word": target_word or sentence,
                "translation": translation,
                "sentence": sentence,
                "sentence_with_gap": sentence_with_gap,
            }
        )

    if one_sentence_one_card:
        for sentence in sentences:
            words = extract_words_from_text(sentence)
            if not words:
                continue
            target_word = None
            for w in words:
                if w not in known:
                    target_word = w
                    break
            if target_word is None:
                target_word = words[0]

            new_in_sentence = {w for w in words if w not in known}
            all_new_words.update(new_in_sentence)
            make_base_card(sentence, target_word)
    else:
        all_words = extract_words_from_text(text)
        new_words = {w for w in all_words if w and w not in known}
        if not new_words:
            return [], set()

        for word in sorted(new_words):
            sentence_for_word = None
            for s in sentences:
                if normalize_word(word) in [normalize_word(w) for w in extract_words_from_text(s)]:
                    sentence_for_word = s
                    break
            if not sentence_for_word:
                sentence_for_word = text.strip()

            make_base_card(sentence_for_word, word)
        all_new_words.update(new_words)

    return cards, all_new_words


def auto_generate_cards_from_text(deck_id: int,
                                  text: str,
                                  use_ai_images: bool,
                                  api_key: str | None,
                                  one_sentence_one_card: bool = False,
                                  audio_path: str | None = None,
                                  audio_source: str | None = None,
                                  audio_side: str = "back",
                                  progress_queue: queue.Queue | None = None,
                                  cancel_check=None,
                                  image_spend_cb=None,
                                  max_cards: int | None = None,
                                  skip_sentences: set[str] | None = None,
                                  skip_words: set[str] | None = None,
                                  max_sentences_per_card: int | None = None,
                                  created_card_ids: list[int] | None = None,
                                  repeated_flag_cb=None,
                                  image_placeholder_cb=None,
                                  progress_by_created: bool = False) -> int:
    """
    Если one_sentence_one_card = True:
        1 предложение = 1 карточка (длинный текст -> много карточек).
    Если False:
        классический режим: каждое новое слово = карточка.
    """
    sentences = split_into_sentences(text)
    total_progress = len(sentences)
    if progress_queue is not None:
        progress_queue.put(("progress", 0, max(total_progress, 1), "Разбивка текста"))
    if not sentences:
        return 0

    known = get_known_words()
    created = 0
    all_new_words = set()
    skip_sentences = skip_sentences or set()
    skip_words = skip_words or set()

    def emit_progress(label: str) -> None:
        if progress_queue is None:
            return
        if progress_by_created:
            display_total = max_cards or total_progress
            progress_queue.put(("progress", created, max(display_total, 1), label))
        else:
            progress_queue.put(("progress", created, max(total_progress, 1), label))

    def make_base_card(sentence: str,
                       target_word: str,
                       translation: str,
                       ipa: str,
                       gender: str,
                       plural: str,
                       wiki_data: dict) -> bool:
        nonlocal created
        if max_cards is not None and created >= max_cards:
            return False

        # делаем "дырку" в предложении
        sentence_with_gap = sentence
        if target_word:
            pattern = re.compile(re.escape(target_word), re.IGNORECASE)
            sentence_with_gap = pattern.sub("____", sentence, count=1)

        # если Wiktionary дал более точные IPA/род/мн.ч – подменяем
        ipa_final = wiki_data.get("ipa") or ipa
        gender_final = wiki_data.get("gender") or gender
        plural_final = wiki_data.get("plural") or plural

        # Получаем перевод всего предложения
        sentence_translation = translate_sentence(sentence, use_openai=True)

        front, back = render_generation_faces(
            deck_id,
            {
                "translation": sentence_translation,
                "front_translation": "",
                "sentence_with_gap": sentence_with_gap,
                "word": target_word or "",
                "ipa": ipa_final,
                "gender": gender_final,
                "plural": plural_final,
                "sentence": sentence,
            },
        )

        if max_sentences_per_card and target_word:
            extra_sentences = []
            target_norm = normalize_word(target_word)
            for s in sentences:
                if normalize_word(s) in skip_sentences:
                    continue
                if s == sentence:
                    continue
                if target_norm in [normalize_word(w) for w in extract_words_from_text(s)]:
                    extra_sentences.append(s)
                if len(extra_sentences) >= max_sentences_per_card:
                    break
            if extra_sentences:
                back = back + "\n\nПримеры:\n" + "\n".join(f"- {s}" for s in extra_sentences)

        # доп. блок с данными Wiktionary
        extra_parts = []
        if wiki_data.get("ipa"):
            extra_parts.append(f"IPA: {wiki_data['ipa']}")
        if wiki_data.get("gender") or wiki_data.get("plural"):
            extra_parts.append(
                f"Genus / Plural: {wiki_data.get('gender', '')} | {wiki_data.get('plural', '')}"
            )
        if wiki_data.get("declension"):
            extra_parts.append("Beugung / Formen:\n" + wiki_data["declension"])

        if extra_parts:
            back = back + "\n\n" + "\n".join(extra_parts)

        img_path_front = None
        if use_ai_images and api_key:
            try:
                img_prompt = f"Illustration for German sentence '{sentence}' with key word '{target_word}'"
                allow_image = True
                if callable(image_spend_cb):
                    allow_image = image_spend_cb("card_image_generation", {"prompt": img_prompt})
                if allow_image:
                    img_path_front = generate_image_with_openai(img_prompt, api_key)
            except Exception:
                img_path_front = None
        placeholder_used = False
        if use_ai_images and not img_path_front and callable(image_placeholder_cb):
            img_path_front = image_placeholder_cb(target_word or sentence) or None
            placeholder_used = bool(img_path_front)

        level_value = 1
        repeated_flag = repeated_flag_cb(target_word) if callable(repeated_flag_cb) else "—"

        note_fields = {
            "word": target_word or sentence,
            "translation": translation,
            "example": sentence,
            "level": level_value,
            "image": img_path_front or "",
            "front_image_path": img_path_front or "",
            "back_image_path": None,
            "audio_path": audio_path,
            "audio_side": audio_side,
            "audio_source": audio_source,
            "front": front,
            "back": back,
            "repeated_flag": repeated_flag,
            "image_generated": bool(img_path_front) and not placeholder_used,
            "image_placeholder": placeholder_used,
        }

        _, cards_created = create_note_with_cards(
            deck_id,
            note_fields,
            note_type_id=ensure_generated_note_type_id(),
            created_card_ids=created_card_ids,
        )
        created += cards_created
        if progress_queue is not None:
            progress_queue.put(("log", f"{target_word or sentence}: {repeated_flag}"))
        emit_progress("Генерация карточек")

        # доп. карточки: синонимы
        syns = wiki_data.get("synonyms") or []
        for syn in syns[:3]:
            if max_cards is not None and created >= max_cards:
                return True
            front_syn = f"Synonym für {target_word}: ____"
            back_syn = syn
            syn_fields = {
                "word": syn,
                "translation": back_syn,
                "example": front_syn,
                "level": 1,
                "image": "",
                "front": front_syn,
                "back": back_syn,
                "audio_path": audio_path,
                "repeated_flag": repeated_flag_cb(syn) if callable(repeated_flag_cb) else "—",
            }
            _, cards_created = create_note_with_cards(
                deck_id,
                syn_fields,
                note_type_id=ensure_generated_note_type_id(),
                created_card_ids=created_card_ids,
            )
            created += cards_created
            emit_progress("Генерация карточек")

        # доп. карточки: примеры
        examples = wiki_data.get("examples") or []
        for ex in examples[:3]:
            if max_cards is not None and created >= max_cards:
                return True
            ex_sentence = ex
            pattern = re.compile(re.escape(target_word), re.IGNORECASE)
            ex_gap = pattern.sub("____", ex_sentence, count=1)
            front_ex = ex_gap
            back_ex = ex_sentence
            ex_fields = {
                "word": target_word or ex_sentence,
                "translation": back_ex,
                "example": front_ex,
                "level": 1,
                "image": "",
                "front": front_ex,
                "back": back_ex,
                "audio_path": audio_path,
                "repeated_flag": repeated_flag_cb(target_word) if callable(repeated_flag_cb) else "—",
            }
            _, cards_created = create_note_with_cards(
                deck_id,
                ex_fields,
                note_type_id=ensure_generated_note_type_id(),
                created_card_ids=created_card_ids,
            )
            created += cards_created
            emit_progress("Генерация карточек")
        return True

    if one_sentence_one_card:
        # 1 предложение = 1 карточка
        for sentence in sentences:
            normalized_sentence = normalize_word(sentence)
            if normalized_sentence in skip_sentences:
                continue
            words = extract_words_from_text(sentence)
            if not words:
                continue

            # первое новое слово в предложении, иначе просто первое
            target_word = None
            for w in words:
                if w not in known:
                    target_word = w
                    break
            if target_word is None:
                target_word = words[0]

            new_in_sentence = {w for w in words if w not in known}
            all_new_words.update(new_in_sentence)
            skip_sentences.add(normalized_sentence)

            if cancel_check and cancel_check():
                break

            translation, ipa, gender, plural = enrich_german_word_info(target_word, api_key) \
                if target_word else ("", "", "?", "?")
            wiki_data = get_wiktionary_data(target_word) if target_word else {}

            if not make_base_card(sentence, target_word, translation, ipa, gender, plural, wiki_data):
                break
            if progress_queue is not None:
                if progress_by_created:
                    emit_progress("Генерация предложений")
                else:
                    progress_queue.put(("progress", created, max(total_progress, 1), "Генерация предложений"))
    else:
        # старый режим: каждое новое слово = карточка
        all_words = extract_words_from_text(text)
        new_words = {w for w in all_words if w and w not in known and w not in skip_words}
        if not new_words:
            return 0

        total_progress = len(new_words)
        processed = 0
        for word in sorted(new_words):
            if cancel_check and cancel_check():
                break
            sentence_for_word = None
            for s in sentences:
                if normalize_word(word) in [normalize_word(w) for w in extract_words_from_text(s)]:
                    sentence_for_word = s
                    break
            if not sentence_for_word:
                sentence_for_word = text.strip()

            translation, ipa, gender, plural = enrich_german_word_info(word, api_key)
            wiki_data = get_wiktionary_data(word)

            skip_words.add(word)
            if not make_base_card(sentence_for_word, word, translation, ipa, gender, plural, wiki_data):
                break
            processed += 1
            if progress_queue is not None:
                if progress_by_created:
                    emit_progress("Генерация слов")
                else:
                    progress_queue.put(("progress", processed, max(total_progress, 1), "Генерация слов"))

        all_new_words.update(new_words)

    add_new_words(all_new_words)
    if progress_queue is not None:
        progress_queue.put(("log", f"Создано карточек: {created}"))
    return created


def split_ocr_text_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def auto_generate_cards_from_ocr_text(
    deck_id: int,
    text: str,
    language_mode: str,
    progress_queue: queue.Queue | None = None,
    cancel_check=None,
    max_cards: int | None = None,
    created_card_ids: list[int] | None = None,
    placeholder_image_path: str | None = None,
    skip_sentences: set[str] | None = None,
) -> int:
    sentences = split_ocr_text_into_sentences(text)
    if not sentences:
        return 0

    skip_sentences = skip_sentences or set()
    created = 0

    def emit_progress(label: str, done: int, total: int) -> None:
        if progress_queue is None:
            return
        progress_queue.put(("progress", done, max(total, 1), label))

    remaining_sentences = [s for s in sentences if normalize_word(s) not in skip_sentences]
    total_target = len(remaining_sentences)
    emit_progress("Разбивка текста", 0, total_target)

    sentence_limit = 10 if language_mode == "native" else 1
    chunks: list[list[str]] = []
    buffer: list[str] = []
    for sentence in remaining_sentences:
        buffer.append(sentence)
        if len(buffer) >= sentence_limit:
            chunks.append(buffer)
            buffer = []
    if buffer:
        chunks.append(buffer)

    for chunk_index, chunk in enumerate(chunks, start=1):
        if cancel_check and cancel_check():
            break
        if max_cards is not None and created >= max_cards:
            break
        combined_text = " ".join(chunk)
        masked_text, hidden_words = mask_text_by_repeated_words(combined_text)
        words_in_text = extract_words_from_text(combined_text)
        primary_word = words_in_text[0] if words_in_text else ""
        primary_display = primary_word if primary_word and is_repeated_word(primary_word) else "—"
        hidden_text = ", ".join(hidden_words)
        back_word = hidden_words[0] if hidden_words else ""

        front, back = render_generation_faces(
            deck_id,
            {
                "translation": hidden_text,
                "front_translation": "",
                "sentence_with_gap": masked_text,
                "word": back_word,
                "front_word": primary_display,
                "ipa": "",
                "gender": "",
                "plural": "",
                "sentence": combined_text,
            },
        )

        note_fields = {
            "word": primary_word or combined_text,
            "translation": hidden_text,
            "example": combined_text,
            "level": 1,
            "image": placeholder_image_path or "",
            "front_image_path": placeholder_image_path or "",
            "back_image_path": "",
            "audio_path": None,
            "front": front,
            "back": back,
            "repeated_flag": get_repeated_word_flag(primary_word or ""),
            "image_generated": False,
            "image_placeholder": bool(placeholder_image_path),
        }

        _, cards_created = create_note_with_cards(
            deck_id,
            note_fields,
            note_type_id=ensure_generated_note_type_id(),
            created_card_ids=created_card_ids,
        )
        created += cards_created
        emit_progress("Генерация карточек", created, max(total_target, 1))
        for sentence in chunk:
            skip_sentences.add(normalize_word(sentence))
        if progress_queue is not None:
            progress_queue.put(("log", f"{combined_text}: скрыто {len(hidden_words)}"))
        if max_cards is not None and created >= max_cards:
            break

    return created


def auto_generate_cards_from_image(deck_id: int,
                                   image_path: str,
                                   use_ai_images: bool,
                                   api_key: str | None,
                                   one_sentence_one_card: bool = False,
                                   image_spend_cb=None) -> int:
    if not OCR_AVAILABLE or not is_tesseract_available():
        raise RuntimeError("Tesseract OCR не настроен.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if not _ensure_required_lang_files():
        raise RuntimeError("Не найдены файлы deu.traineddata / rus.traineddata в tessdata.")

    img = Image.open(image_path)
    config, _, _ = _build_required_ocr_config(DEFAULT_OCR_CONFIG_BASE)
    text = pytesseract.image_to_string(img, lang=DEFAULT_OCR_LANG, config=config)
    if not text.strip():
        return 0

    return auto_generate_cards_from_text(
        deck_id, text, use_ai_images, api_key,
        one_sentence_one_card=one_sentence_one_card,
        audio_path=None,
        image_spend_cb=image_spend_cb,
    )


STT_WHISPER_MODEL = os.environ.get("STT_WHISPER_MODEL", "small")
STT_FFMPEG_TIMEOUT_SEC = int(os.environ.get("STT_FFMPEG_TIMEOUT_SEC", "180"))
STT_WATCHDOG_IDLE_SEC = int(os.environ.get("STT_WATCHDOG_IDLE_SEC", "180"))
STT_WHISPER_HEARTBEAT_SEC = float(os.environ.get("STT_WHISPER_HEARTBEAT_SEC", "1.0"))


class STTError(RuntimeError):
    def __init__(self, user_message: str, reason: str, detail: str | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.reason = reason
        self.detail = detail or ""


class STTCancelled(Exception):
    pass


def get_log_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_log_path() -> str:
    return os.path.join(get_log_dir(), "stt_last.log")


def write_log(line: str) -> None:
    log_path = get_log_path()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")
        f.flush()


def _write_log_lines(lines: list[str], *, mode: str) -> None:
    log_path = get_log_path()
    with open(log_path, mode, encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
            f.flush()


def _append_log_line(log_lines: list[str], line: str) -> None:
    log_lines.append(line)
    write_log(line)


def _append_log_lines(log_lines: list[str], lines: list[str]) -> None:
    for line in lines:
        _append_log_line(log_lines, line)


def _init_stt_log(
    *,
    lang_label: str,
    model_name: str,
    video_path: str,
    start: float | None,
    end: float | None,
) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    header_lines = [
        f"timestamp={datetime.now().isoformat()}",
        f"python_version={sys.version.replace(os.linesep, ' ')}",
        f"cwd={os.getcwd()}",
        f"base_dir={base_dir}",
        f"lang={lang_label}",
        f"model={model_name}",
        f"video_path={video_path}",
        f"start={start}",
        f"end={end}",
    ]
    _write_log_lines(header_lines, mode="w")


def _read_stt_log() -> str:
    log_path = get_log_path()
    if not os.path.exists(log_path):
        return "Лог STT не найден."
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as exc:
        return f"Не удалось прочитать лог: {exc}"


def _append_stt_log_line(line: str) -> None:
    try:
        write_log(line)
    except Exception:
        pass


def _log_stt_exception(exc: Exception, *, prefix: str) -> None:
    write_log(f"exception={prefix}:{type(exc).__name__}")
    for line in traceback.format_exc().splitlines():
        write_log(line)


def run_ffmpeg(
    cmd: list[str],
    *,
    timeout_sec: int = STT_FFMPEG_TIMEOUT_SEC,
    log_lines: list[str] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[int, str, str]:
    log_lines = log_lines if log_lines is not None else []
    cmd = list(cmd)
    cmd[1:1] = ["-hide_banner", "-loglevel", "error", "-nostdin"]
    _append_log_line(log_lines, "ffmpeg_cmd=" + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = ""
    stderr = ""
    completed = False
    start_ts = time.time()
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                raise STTCancelled()
            elapsed = time.time() - start_ts
            remaining = max(0.0, float(timeout_sec) - elapsed)
            if remaining <= 0:
                proc.kill()
                _append_log_line(log_lines, "TIMEOUT/DEADLOCK=ffmpeg_timeout")
                raise RuntimeError("FFmpeg превысил лимит времени.")
            try:
                stdout, stderr = proc.communicate(timeout=min(1.0, remaining))
                completed = True
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        if not completed:
            try:
                stdout, stderr = proc.communicate(timeout=1)
            except Exception:
                pass
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.stderr:
            try:
                proc.stderr.close()
            except Exception:
                pass
    stdout = stdout or ""
    stderr = stderr or ""

    def _trim_output(value: str, limit: int = 2000) -> str:
        value = value.strip()
        if len(value) > limit:
            return value[-limit:]
        return value

    _append_log_line(log_lines, f"ffmpeg_returncode={proc.returncode or 0}")
    _append_log_line(log_lines, "ffmpeg_stdout=" + _trim_output(stdout))
    _append_log_line(log_lines, "ffmpeg_stderr=" + _trim_output(stderr))
    return proc.returncode or 0, stdout, stderr


def _normalize_lang(lang: str | None) -> str | None:
    if not lang:
        return None
    value = lang.replace("_", "-").strip().lower()
    if not value or value in {"auto", "autodetect", "detect"}:
        return None
    return value.split("-")[0]


def _map_whisper_language(lang: str | None) -> str | None:
    if not lang:
        return None
    value = lang.replace("_", "-").strip()
    if not value or value.lower() == "auto":
        return None
    mapping = {
        "de-de": "de",
        "ru-ru": "ru",
        "en-us": "en",
    }
    normalized = value.lower()
    return mapping.get(normalized, _normalize_lang(value))


def _stt_user_message(reason: str, detail: str | None = None) -> str:
    if reason == "no_speech":
        return "Речь не обнаружена (тишина/очень тихо)."
    if reason == "wrong_language":
        label = detail or "Auto"
        return f"Проверь язык распознавания (сейчас: {label})."
    if reason == "no_engine":
        return "Нет движка STT: установите faster-whisper (предпочтительно)."
    if reason == "noisy":
        return "Речь не обнаружена (тишина/очень тихо)."
    if reason == "network":
        return "Ошибка сети или квоты при обращении к сервису распознавания."
    return detail or "Не удалось распознать речь."


def _load_wav_mono(wav_path: str) -> tuple["np.ndarray", int]:
    import wave

    if not NUMPY_AVAILABLE:
        raise RuntimeError("NumPy не установлен, обработка аудио недоступна.")
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = audio / 32768.0
    return audio, sr


def _write_wav_mono(wav_path: str, audio: "np.ndarray", sr: int = 16000) -> None:
    import wave

    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    data = (clipped * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def _compute_vad_segments(audio: "np.ndarray", sr: int) -> list[tuple[float, float]]:
    if not WEBRTCVAD_AVAILABLE:
        return []
    import webrtcvad

    vad = webrtcvad.Vad(2)
    if sr != 16000 and LIBROSA_AVAILABLE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    if sr != 16000:
        return []
    frame_ms = 20
    frame_len = int(sr * frame_ms / 1000)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    segments: list[tuple[float, float]] = []
    in_speech = False
    start = 0.0
    idx = 0
    total_frames = len(pcm) // 2 // frame_len
    for i in range(total_frames):
        frame = pcm[idx: idx + frame_len * 2]
        idx += frame_len * 2
        is_speech = vad.is_speech(frame, sr)
        t = i * frame_ms / 1000.0
        if is_speech and not in_speech:
            start = t
            in_speech = True
        if not is_speech and in_speech:
            end = t
            if end - start > 0.1:
                segments.append((start, end))
            in_speech = False
    if in_speech:
        end = total_frames * frame_ms / 1000.0
        segments.append((start, end))
    return segments


def _analyze_speech_presence(wav_path: str, log_lines: list[str]) -> tuple[float, float, float]:
    audio, sr = _load_wav_mono(wav_path)
    duration = len(audio) / max(sr, 1)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    voiced_ratio = 0.0
    if WEBRTCVAD_AVAILABLE:
        segments = _compute_vad_segments(audio, sr)
        voiced = sum((end - start) for start, end in segments)
        voiced_ratio = voiced / duration if duration > 0 else 0.0
    else:
        threshold = max(0.01, rms * 0.6)
        voiced_ratio = float(np.mean(np.abs(audio) > threshold)) if len(audio) else 0.0
    _append_log_line(log_lines, f"audio_duration_sec={duration:.3f}")
    _append_log_line(log_lines, f"audio_rms={rms:.6f}")
    _append_log_line(log_lines, f"audio_voiced_ratio={voiced_ratio:.4f}")
    return duration, rms, voiced_ratio


def _apply_gain_if_needed(wav_path: str, rms: float, log_lines: list[str]) -> float:
    if not NUMPY_AVAILABLE:
        return rms
    if rms <= 0:
        return rms
    if rms >= 0.03:
        return rms
    audio, sr = _load_wav_mono(wav_path)
    boosted = audio * 2.0
    _write_wav_mono(wav_path, boosted, sr=sr)
    new_rms = float(np.sqrt(np.mean(np.square(boosted)))) if len(boosted) else 0.0
    _append_log_line(log_lines, "gain_db=6")
    _append_log_line(log_lines, f"audio_rms_after_gain={new_rms:.6f}")
    return new_rms


def extract_preprocessed_wav(
    video_path: str,
    start: float | None = None,
    end: float | None = None,
    log_lines: list[str] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    tmp_dir = ensure_dir(os.path.join("tmp", "media"))
    output_path = os.path.join(tmp_dir, f"stt_pre_{uuid4().hex}.wav")
    ffmpeg_path = find_ffmpeg()
    log_lines = log_lines if log_lines is not None else []
    if ffmpeg_path:
        filters = "highpass=f=80,lowpass=f=8000,afftdn,dynaudnorm,loudnorm=I=-16:TP=-1.5:LRA=11"
        cmd = [ffmpeg_path, "-y"]
        if start is not None:
            cmd += ["-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        cmd += [
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            "-af",
            filters,
            output_path,
        ]
        returncode, _stdout, stderr = run_ffmpeg(
            cmd,
            timeout_sec=STT_FFMPEG_TIMEOUT_SEC,
            log_lines=log_lines,
            cancel_event=cancel_event,
        )
        if returncode != 0:
            error_text = stderr.strip() or "Не удалось извлечь аудио."
            raise RuntimeError(error_text)
        return output_path
    if MOVIEPY_AVAILABLE:
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy не установлен, обработка аудио недоступна.")
        clip = None
        try:
            if video_path.lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".ogg")):
                clip = mp.AudioFileClip(video_path)
                audio_clip = clip
            else:
                clip = mp.VideoFileClip(video_path)
                audio_clip = clip.audio
            if audio_clip is None:
                raise RuntimeError("В видео нет аудио дорожки.")
            if start is not None or end is not None:
                audio_clip = audio_clip.subclip(start or 0, end)
            audio = audio_clip.to_soundarray(fps=16000)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            target_rms = 0.1
            if rms > 0:
                audio = audio * (target_rms / rms)
            _write_wav_mono(output_path, audio, sr=16000)
            return output_path
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass
    raise RuntimeError("FFmpeg не найден, а moviepy не установлен.")


def extract_audio_from_video(
    video_path: str,
    start: float | None = None,
    end: float | None = None,
    log_lines: list[str] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    tmp_dir = ensure_dir(os.path.join("tmp", "media"))
    output_path = os.path.join(tmp_dir, f"stt_file_{uuid4().hex}.wav")
    log_lines = log_lines if log_lines is not None else []
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        filters = "highpass=f=80,lowpass=f=8000,afftdn,dynaudnorm,loudnorm=I=-16:TP=-1.5:LRA=11"
        cmd = [ffmpeg_path, "-y"]
        if start is not None:
            cmd += ["-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        cmd += [
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            "-af",
            filters,
            output_path,
        ]
        returncode, _stdout, stderr = run_ffmpeg(
            cmd,
            timeout_sec=STT_FFMPEG_TIMEOUT_SEC,
            log_lines=log_lines,
            cancel_event=cancel_event,
        )
        if returncode != 0:
            error_text = stderr.strip() or "Не удалось извлечь аудио."
            raise RuntimeError(error_text)
        _append_log_line(log_lines, f"wav_path={output_path}")
        return output_path
    if MOVIEPY_AVAILABLE:
        clip = None
        try:
            if video_path.lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".ogg")):
                clip = mp.AudioFileClip(video_path)
                audio_clip = clip
            else:
                clip = mp.VideoFileClip(video_path)
                audio_clip = clip.audio
            if audio_clip is None:
                raise RuntimeError("В видео нет аудио дорожки.")
            if start is not None or end is not None:
                audio_clip = audio_clip.subclip(start or 0, end)
            audio_clip.write_audiofile(
                output_path,
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"],
                verbose=False,
                logger=None,
            )
            _append_log_line(log_lines, f"wav_path={output_path}")
            return output_path
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass
    raise RuntimeError("FFmpeg не найден, а moviepy не установлен.")


def extract_and_preprocess_audio(
    video_path: str,
    start: float | None = None,
    end: float | None = None,
    gain: float | None = None,
    tempo: float | None = None,
    log_lines: list[str] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    _ = gain
    _ = tempo
    return extract_preprocessed_wav(
        video_path,
        start=start,
        end=end,
        log_lines=log_lines,
        cancel_event=cancel_event,
    )


def _get_wav_duration(wav_path: str) -> float:
    import wave

    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / max(1, wf.getframerate())


def _split_wav_with_ffmpeg(
    wav_path: str,
    *,
    chunk_sec: float,
    log_lines: list[str] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[tuple[str, float, float]], float]:
    duration = _get_wav_duration(wav_path)
    if duration <= chunk_sec:
        return [(wav_path, 0.0, duration)], duration
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg не найден.")
    tmp_dir = ensure_dir(os.path.join("tmp", "chunks"))
    pattern = os.path.join(tmp_dir, f"chunk_{uuid4().hex}_%03d.wav")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        wav_path,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_sec),
        "-c",
        "copy",
        "-reset_timestamps",
        "1",
        pattern,
    ]
    returncode, _stdout, stderr = run_ffmpeg(
        cmd,
        timeout_sec=STT_FFMPEG_TIMEOUT_SEC,
        log_lines=log_lines,
        cancel_event=cancel_event,
    )
    if returncode != 0:
        error_text = stderr.strip() or "Не удалось разрезать WAV."
        raise RuntimeError(error_text)
    chunk_paths = sorted(glob.glob(pattern.replace("%03d", "*")))
    if not chunk_paths:
        return [(wav_path, 0.0, duration)], duration
    chunks: list[tuple[str, float, float]] = []
    for idx, chunk_path in enumerate(chunk_paths):
        t0 = idx * chunk_sec
        t1 = min(duration, t0 + chunk_sec)
        chunks.append((chunk_path, t0, t1))
    return chunks, duration


def split_audio_into_chunks(wav_path: str, max_chunk_sec: float = 30.0) -> list[tuple[str, float, float]]:
    audio, sr = _load_wav_mono(wav_path)
    duration = len(audio) / max(1, sr)
    if duration <= max_chunk_sec:
        return [(wav_path, 0.0, duration)]
    tmp_dir = ensure_dir(os.path.join("tmp", "media"))
    chunks: list[tuple[str, float, float]] = []
    start = 0.0
    idx = 0
    chunk_size = min(25.0, max_chunk_sec)
    while start < duration:
        end = min(duration, start + chunk_size)
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        chunk_audio = audio[start_idx:end_idx]
        chunk_path = os.path.join(tmp_dir, f"stt_chunk_{uuid4().hex}_{idx}.wav")
        _write_wav_mono(chunk_path, chunk_audio, sr=sr)
        chunks.append((chunk_path, start, end))
        idx += 1
        start = end
    return chunks


def _find_vosk_model(lang: str) -> str | None:
    candidates = []
    env_path = os.environ.get("VOSK_MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    base_candidates = [
        "models",
        "model",
        "vosk-model",
        "vosk-model-small",
        f"vosk-model-small-{lang}",
        f"vosk-model-{lang}",
    ]
    for base in base_candidates:
        path = os.path.join(os.getcwd(), base)
        candidates.append(path)
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def stt_transcribe_wav(
    wav_path: str,
    lang: str | None,
    engine_pref: str = "auto",
    progress_cb=None,
    stage_cb=None,
    total_seconds: float | None = None,
    log_lines: list[str] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[str, list[dict], str, str | None]:
    segments: list[dict] = []
    chunk_paths: list[str] = []
    whisper_error: str | None = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise STTCancelled()
        lang_whisper = _map_whisper_language(lang or "Auto")
        duration = float(total_seconds or 0.0)
        try:
            import wave

            if not duration:
                with wave.open(wav_path, "rb") as wf:
                    duration = wf.getnframes() / max(1, wf.getframerate())
        except Exception:
            duration = duration or 0.0
        if FASTER_WHISPER_AVAILABLE and engine_pref in {"auto", "whisper"}:
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise STTCancelled()
                write_log("whisper_start")
                try:
                    model, cached = get_whisper_model(
                        STT_WHISPER_MODEL, device="cpu", compute_type="int8"
                    )
                except Exception as exc:
                    whisper_error = f"Whisper ошибка загрузки модели: {type(exc).__name__}: {exc}"
                    raise
                write_log(f"whisper_model_cached={'YES' if cached else 'NO'}")
                write_log(f"whisper_model_cache_dir={_get_whisper_cache_dir()}")
                if stage_cb is not None:
                    if cached:
                        stage_cb("Модель: cached", "determinate")
                    else:
                        stage_cb("Модель: загружаю (первый запуск)", "indeterminate")
                chunk_sec = 25.0
                write_log(f"whisper_chunking_start chunk_sec={chunk_sec}")
                try:
                    chunks, chunk_duration = _split_wav_with_ffmpeg(
                        wav_path,
                        chunk_sec=chunk_sec,
                        log_lines=log_lines,
                        cancel_event=cancel_event,
                    )
                    total_seconds = duration or chunk_duration
                except Exception:
                    chunks = split_audio_into_chunks(wav_path, max_chunk_sec=30.0)
                    total_seconds = duration or max((t1 for _, _, t1 in chunks), default=0.0)
                write_log(f"chunks_count={len(chunks)}")
                def _run_whisper(vad_filter: bool) -> tuple[str, list[dict], float]:
                    all_text_parts: list[str] = []
                    all_segments: list[dict] = []
                    last_processed = 0.0
                    last_logged_decile = -1
                    for idx, (chunk_path, t0, t1) in enumerate(chunks, start=1):
                        if cancel_event is not None and cancel_event.is_set():
                            raise STTCancelled()
                        write_log(
                            f"chunk_progress engine=whisper idx={idx}/{len(chunks)} t0={t0:.2f} t1={t1:.2f}"
                        )
                        if progress_cb is not None and total_seconds > 0:
                            progress_cb(t1, total_seconds, idx, len(chunks))
                            progress_percent = int(
                                min(100, max(0, (t1 / max(1.0, total_seconds)) * 100.0))
                            )
                            decile = progress_percent // 10
                            if decile > last_logged_decile:
                                last_logged_decile = decile
                                write_log(f"progress={progress_percent} chunk_t1={t1:.2f}")
                        if chunk_path != wav_path and chunk_path not in chunk_paths:
                            chunk_paths.append(chunk_path)
                        seg_iter, _info = model.transcribe(
                            chunk_path,
                            language=lang_whisper,
                            vad_filter=vad_filter,
                            beam_size=3,
                            condition_on_previous_text=True,
                        )
                        for seg in seg_iter:
                            if cancel_event is not None and cancel_event.is_set():
                                raise STTCancelled()
                            text_part = (seg.text or "").strip()
                            if text_part:
                                all_text_parts.append(text_part)
                            seg_start = float(seg.start) + t0
                            seg_end = float(seg.end) + t0
                            all_segments.append(
                                {"start": seg_start, "end": seg_end, "text": text_part}
                            )
                            last_processed = max(last_processed, seg_end)
                            if progress_cb is not None:
                                progress_cb(last_processed, total_seconds, idx, len(chunks))
                            if total_seconds > 0:
                                progress_percent = int(
                                    min(100, max(0, (last_processed / total_seconds) * 100.0))
                                )
                                decile = progress_percent // 10
                                if decile > last_logged_decile:
                                    last_logged_decile = decile
                                    write_log(f"progress={progress_percent} seg_end={seg_end:.2f}")
                        if progress_cb is not None:
                            progress_cb(max(last_processed, t1), total_seconds, idx, len(chunks))
                    full_text = " ".join(all_text_parts).strip()
                    voiced_duration = sum(
                        max(0.0, float(seg["end"]) - float(seg["start"]))
                        for seg in all_segments
                        if seg.get("text")
                    )
                    return full_text, all_segments, voiced_duration

                text, segments, voiced_duration = _run_whisper(vad_filter=True)
                write_log(
                    f"whisper_vad_segments={len(segments)} voiced_seconds={voiced_duration:.2f}"
                )
                if len(text) < 50 or len(segments) < 3:
                    write_log(
                        f"whisper_vad_retry reason=short_text text_len={len(text)} segments={len(segments)}"
                    )
                    text, segments, voiced_duration = _run_whisper(vad_filter=False)
                    write_log(
                        f"whisper_no_vad_segments={len(segments)} voiced_seconds={voiced_duration:.2f}"
                    )
                if text:
                    return text, segments, "whisper", None
                raise STTError(_stt_user_message("no_speech"), "no_speech")
            except STTCancelled:
                raise
            except STTError:
                raise
            except Exception as exc:
                if whisper_error is None:
                    detail = f"{type(exc).__name__}: {exc}"
                    if "timeout" in detail.lower():
                        whisper_error = f"Whisper timeout: {detail}"
                    else:
                        whisper_error = f"Whisper ошибка: {detail}"
                segments = []
            finally:
                try:
                    write_log("whisper_done")
                except Exception:
                    pass
        if engine_pref in {"auto", "whisper"} and not FASTER_WHISPER_AVAILABLE:
            detail = FASTER_WHISPER_IMPORT_ERROR or "faster-whisper не установлен"
            whisper_error = f"Whisper недоступен: {detail}"
        if SR_AVAILABLE and engine_pref in {"auto", "google"}:
            r = sr.Recognizer()
            chunks = split_audio_into_chunks(wav_path, max_chunk_sec=25.0)
            total_seconds = duration or max((t1 for _, _, t1 in chunks), default=0.0)
            for idx, (chunk_path, t0, t1) in enumerate(chunks, start=1):
                if cancel_event is not None and cancel_event.is_set():
                    raise STTCancelled()
                write_log(
                    f"chunk_progress engine=google idx={idx}/{len(chunks)} t0={t0:.2f} t1={t1:.2f}"
                )
                if chunk_path != wav_path:
                    chunk_paths.append(chunk_path)
                with sr.AudioFile(chunk_path) as source:
                    audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data, language=lang or "en-US")
                except sr.RequestError as exc:
                    raise STTError(_stt_user_message("network"), "network", str(exc)) from exc
                except sr.UnknownValueError:
                    text = ""
                if text:
                    segments.append({"start": t0, "end": t1, "text": text.strip()})
                if progress_cb is not None:
                    progress_cb(t1, total_seconds, idx, len(chunks))
            text = " ".join(seg["text"] for seg in segments).strip()
            return text, segments, "google", whisper_error
        raise STTError(_stt_user_message("no_engine"), "no_engine")
    finally:
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass


def _postprocess_transcript(text: str, segments: list[dict]) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return ""
    punctuated = auto_punctuate(normalized) if "auto_punctuate" in globals() else normalized
    sentences = split_into_sentences(punctuated) if "split_into_sentences" in globals() else []
    if sentences and len(sentences) > 1:
        return "\n".join(sentences)
    if segments:
        rebuilt: list[str] = []
        buf: list[str] = []
        buf_start = None
        last_end = None
        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if buf_start is None:
                buf_start = start
            if last_end is not None and (start - last_end) > 0.8:
                sentence = " ".join(buf).strip()
                if sentence:
                    rebuilt.append(sentence)
                buf = []
                buf_start = start
            buf.append(seg_text)
            last_end = end
        if buf:
            sentence = " ".join(buf).strip()
            if sentence:
                rebuilt.append(sentence)
        if rebuilt:
            punctuated = ". ".join(s.strip() for s in rebuilt if s.strip()).strip()
            if punctuated and not punctuated.endswith((".", "!", "?")):
                punctuated += "."
            return punctuated.replace(". ", ".\n")
    return punctuated


def _is_language_mismatch(text: str, lang: str | None) -> bool:
    if not text:
        return False
    lang_short = _normalize_lang(lang)
    if not lang_short:
        return False
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return False
    if lang_short.startswith("ru"):
        cyrillic = re.findall(r"[А-Яа-яЁё]", text)
        return (len(cyrillic) / max(1, len(letters))) < 0.2
    if lang_short in {"de", "en", "fr", "es", "it"}:
        latin = re.findall(r"[A-Za-z]", text)
        return (len(latin) / max(1, len(letters))) < 0.2
    return False


def run_stt_pipeline(
    source_path: str,
    start: float | None = None,
    end: float | None = None,
    lang: str | None = None,
    punctuate: bool = True,
    preprocessed: bool = False,
    log_context: list[str] | None = None,
    progress_queue: queue.Queue | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[str, list[dict], str, str | None]:
    lang_label = lang or "Auto"
    log_lines: list[str] = []
    _init_stt_log(
        lang_label=lang_label,
        model_name=STT_WHISPER_MODEL,
        video_path=source_path,
        start=start,
        end=end,
    )
    _append_log_lines(
        log_lines,
        [
            f"source_path={source_path}",
            f"lang={lang_label}",
            f"start={start}",
            f"end={end}",
            f"preprocessed={preprocessed}",
        ],
    )
    if log_context:
        _append_log_lines(log_lines, log_context)
    wav_path = None
    delete_wav = True

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise STTCancelled()

    def _emit_progress(
        *,
        overall: float | None = None,
        status: str | None = None,
        mode: str | None = None,
        stt_percent: float | None = None,
        heartbeat: bool | None = None,
        total_duration: float | None = None,
    ) -> None:
        if progress_queue is None:
            return
        progress_queue.put(
            {
                "overall": overall,
                "status": status,
                "mode": mode,
                "stt_percent": stt_percent,
                "heartbeat": heartbeat,
                "total_duration": total_duration,
            }
        )

    progress_state = {
        "overall": 20.0,
        "status": "Распознаю…",
        "stt_percent": 0.0,
    }

    def _on_stage(status: str, mode: str) -> None:
        progress_state.update({"status": status})
        _emit_progress(overall=20.0, status=status, mode=mode)

    def _on_transcribe_progress(
        processed_seconds: float,
        total_seconds: float,
        chunk_index: int,
        total_chunks: int,
    ) -> None:
        _check_cancel()
        if total_seconds > 0:
            stt_percent = min(100.0, max(0.0, (processed_seconds / total_seconds) * 100.0))
        elif total_chunks > 0:
            stt_percent = min(100.0, max(0.0, (chunk_index / total_chunks) * 100.0))
        else:
            return
        overall = 20.0 + (70.0 * (stt_percent / 100.0))
        progress_state.update(
            {
                "overall": overall,
                "status": f"Распознаю ({int(stt_percent)}%)…",
                "stt_percent": stt_percent,
            }
        )
        _emit_progress(
            overall=overall,
            status=f"Распознаю ({int(stt_percent)}%)…",
            mode="determinate",
            stt_percent=stt_percent,
        )

    try:
        _check_cancel()
        if preprocessed:
            wav_path = source_path
            delete_wav = False
        else:
            _emit_progress(overall=0.0, status="Извлекаю аудио…", mode="determinate")
            wav_path = extract_preprocessed_wav(
                source_path,
                start=start,
                end=end,
                log_lines=log_lines,
                cancel_event=cancel_event,
            )
        _check_cancel()
        _emit_progress(overall=10.0, status="Подготавливаю…", mode="determinate")
        duration, rms, voiced_ratio = _analyze_speech_presence(wav_path, log_lines)
        rms = _apply_gain_if_needed(wav_path, rms, log_lines)
        if duration < 0.3 or rms < 0.004 or voiced_ratio < 0.01:
            raise STTError(_stt_user_message("no_speech"), "no_speech")
        _emit_progress(
            overall=20.0,
            status="Распознаю…",
            mode="determinate",
            stt_percent=0.0,
            total_duration=duration,
        )
        heartbeat_stop = threading.Event()

        def _heartbeat_loop() -> None:
            while not heartbeat_stop.wait(STT_WHISPER_HEARTBEAT_SEC):
                _emit_progress(
                    overall=progress_state.get("overall"),
                    status=progress_state.get("status"),
                    mode=None,
                    stt_percent=progress_state.get("stt_percent"),
                    heartbeat=True,
                )

        heartbeat_thread = None
        if progress_queue is not None:
            heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
            heartbeat_thread.start()
        try:
            text, segments, engine, engine_detail = stt_transcribe_wav(
                wav_path,
                lang,
                engine_pref="auto",
                progress_cb=_on_transcribe_progress,
                stage_cb=_on_stage,
                total_seconds=duration,
                log_lines=log_lines,
                cancel_event=cancel_event,
            )
            if start is not None and segments:
                offset = float(start)
                for seg in segments:
                    seg["start"] = float(seg.get("start", 0.0)) + offset
                    seg["end"] = float(seg.get("end", seg.get("start", 0.0))) + offset
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)
        _check_cancel()
        _emit_progress(overall=92.0, status="Постобработка…", mode="determinate")
        _append_log_line(log_lines, f"engine_used={engine}")
        if engine == "whisper":
            _append_log_line(log_lines, f"model_name={STT_WHISPER_MODEL}")
        else:
            _append_log_line(log_lines, "model_name=google")
            if engine_detail:
                _append_log_line(log_lines, f"whisper_fallback_reason={engine_detail}")
        _append_log_line(log_lines, f"segments_count={len(segments)}")
        _append_log_line(log_lines, f"text_preview={(text or '').strip()[:200]}")
        _append_log_line(log_lines, f"text_length={len(text.strip())}")
        if len(text.strip()) < 2:
            raise STTError(_stt_user_message("no_speech"), "no_speech")
        if _is_language_mismatch(text, lang):
            raise STTError(_stt_user_message("wrong_language", lang_label), "wrong_language", lang_label)
        final_text = _postprocess_transcript(text, segments) if punctuate else text.strip()
        _emit_progress(overall=100.0, status="Готово", mode="determinate")
        return final_text, segments, engine, engine_detail
    except STTCancelled:
        _append_log_line(log_lines, "stt_cancelled=true")
        raise
    except STTError as exc:
        _append_log_line(log_lines, f"stt_error_reason={exc.reason}")
        _append_log_line(log_lines, f"stt_error_detail={exc.detail}")
        _log_stt_exception(exc, prefix="stt_error")
        raise
    except RuntimeError as exc:
        msg = str(exc)
        msg_lower = msg.lower()
        if "превысил лимит времени" in msg_lower or "timeout" in msg_lower:
            reason = "timeout"
            _append_log_line(log_lines, "TIMEOUT/DEADLOCK=ffmpeg_timeout")
        elif "ffmpeg" in msg_lower or "moviepy" in msg_lower:
            reason = "no_engine"
        else:
            reason = "noisy"
        _append_log_line(log_lines, f"stt_error_reason={reason}")
        _append_log_line(log_lines, f"stt_error_detail={msg}")
        _log_stt_exception(exc, prefix="runtime_error")
        raise STTError(_stt_user_message(reason, msg), reason, msg) from exc
    except Exception as exc:
        msg = str(exc)
        _append_log_line(log_lines, "stt_error_reason=unknown")
        _append_log_line(log_lines, f"stt_error_detail={msg}")
        _log_stt_exception(exc, prefix="unknown_error")
        raise STTError(_stt_user_message("noisy", msg), "noisy", msg) from exc
    finally:
        if delete_wav and wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


def show_stt_error_dialog(parent: tk.Misc | None, title: str, message: str) -> None:
    win = tk.Toplevel(parent) if parent is not None else tk.Toplevel()
    win.title(title)
    win.geometry("560x320")
    try:
        apply_dark_theme_to_window(win, getattr(parent, "palette", None))
    except Exception:
        pass
    container = ttk.Frame(win)
    container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
    ttk.Label(container, text=message, wraplength=520, style="Muted.TLabel").pack(anchor="w", pady=(0, 12))
    btn_row = ttk.Frame(container)
    btn_row.pack(fill=tk.X, side=tk.BOTTOM)

    def _open_details():
        details = _read_stt_log()
        details_win = tk.Toplevel(win)
        details_win.title("Детали распознавания")
        details_win.geometry("700x460")
        try:
            apply_dark_theme_to_window(details_win, getattr(parent, "palette", None))
        except Exception:
            pass
        text_box = scrolledtext.ScrolledText(details_win, height=18)
        try:
            style_text_widget(text_box, getattr(parent, "palette", None))
        except Exception:
            pass
        text_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_box.insert("1.0", details or "Лог пуст.")
        text_box.configure(state=tk.DISABLED)

    ttk.Button(btn_row, text="Показать детали", command=_open_details).pack(side=tk.LEFT)
    ttk.Button(btn_row, text="Закрыть", command=win.destroy).pack(side=tk.RIGHT)


def record_speech_to_text(
    duration_sec: int,
    mic_index: int | None,
    progress_queue: queue.Queue | None = None,
    cancel_check=None,
) -> tuple[str, str | None]:
    if not SR_AVAILABLE:
        raise RuntimeError("SpeechRecognition не установлен.")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")
    r = sr.Recognizer()
    source = sr.Microphone(device_index=mic_index) if mic_index is not None else sr.Microphone()
    if progress_queue is not None:
        progress_queue.put(("progress", 0, max(duration_sec, 1), "Запись"))
    with source as s:
        r.adjust_for_ambient_noise(s, duration=0.5)
        audio = r.record(s, duration=duration_sec)
    if cancel_check and cancel_check():
        return "", None
    os.makedirs("recordings", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join("recordings", f"speech_{ts}.wav")
    with open(audio_path, "wb") as f:
        f.write(audio.get_wav_data())
    if progress_queue is not None:
        progress_queue.put(("progress", duration_sec, max(duration_sec, 1), "Распознавание"))
    try:
        text, _segments, _engine, _engine_detail = run_stt_pipeline(
            audio_path, lang="de-DE", punctuate=False
        )
    except STTError as exc:
        raise RuntimeError(exc.user_message) from exc
    except Exception as exc:
        raise RuntimeError(f"Не удалось распознать речь: {exc}") from exc
    return text, audio_path


def transcribe_audio_to_text(
    audio_path: str,
    lang: str | None = None,
    punctuate: bool = True,
) -> str:
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    try:
        text, _segments, _engine, _engine_detail = run_stt_pipeline(
            audio_path, lang=lang or "de-DE", punctuate=punctuate
        )
    except STTError as exc:
        raise RuntimeError(exc.user_message) from exc
    return text


def transcribe_audio_file(
    wav_path: str,
    lang: str | None = None,
    punctuate: bool = True,
    log_context: list[str] | None = None,
) -> tuple[str, list[dict], str, str | None]:
    if not wav_path or not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)
    return run_stt_pipeline(
        wav_path,
        lang=lang,
        punctuate=punctuate,
        preprocessed=True,
        log_context=log_context,
    )


def extract_audio_from_video_for_stt(
    video_path: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> str:
    return extract_audio_from_video(video_path, start=start_sec, end=end_sec)


def auto_generate_cards_from_speech(deck_id: int,
                                    duration_sec: int,
                                    use_ai_images: bool,
                                    api_key: str | None,
                                    mic_index: int | None,
                                    one_sentence_one_card: bool = False,
                                    progress_queue: queue.Queue | None = None,
                                    cancel_check=None,
                                    image_spend_cb=None) -> int:
    if not SR_AVAILABLE:
        raise RuntimeError("SpeechRecognition не установлен.")
    r = sr.Recognizer()

    if mic_index is not None:
        source = sr.Microphone(device_index=mic_index)
    else:
        source = sr.Microphone()

    if progress_queue is not None:
        progress_queue.put(("progress", 0, max(duration_sec, 1), "Запись"))

    with source as s:
        r.adjust_for_ambient_noise(s, duration=0.5)
        audio = r.record(s, duration=duration_sec)

    if cancel_check and cancel_check():
        return 0

    os.makedirs("recordings", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join("recordings", f"speech_{ts}.wav")
    with open(audio_path, "wb") as f:
        f.write(audio.get_wav_data())

    if progress_queue is not None:
        progress_queue.put(("progress", duration_sec, max(duration_sec, 1), "Распознавание"))

    try:
        text, _segments, _engine, _engine_detail = run_stt_pipeline(
            audio_path, lang="de-DE", punctuate=False
        )
    except STTError as exc:
        raise RuntimeError(exc.user_message) from exc
    except Exception as exc:
        raise RuntimeError(f"Не удалось распознать речь: {exc}") from exc

    if cancel_check and cancel_check():
        return 0

    return auto_generate_cards_from_text(
        deck_id, text, use_ai_images, api_key,
        one_sentence_one_card=one_sentence_one_card,
        audio_path=audio_path,
        audio_source="digital_hearing",
        audio_side="back",
        progress_queue=progress_queue,
        cancel_check=cancel_check,
        image_spend_cb=image_spend_cb,
    )


def auto_generate_cards_from_video(deck_id: int,
                                   video_path: str,
                                   use_ai_images: bool,
                                   api_key: str | None,
                                   image_spend_cb=None) -> int:
    """
    Генерация карточек из видео: извлечение аудио, нарезка на предложения,
    распознавание речи, создание карточки с аудио.
    """
    if not SR_AVAILABLE:
        raise RuntimeError("SpeechRecognition не установлен.")

    temp_audio = None
    try:
        # Извлечь и предобработать аудио из видео
        temp_audio = extract_and_preprocess_audio(video_path)

        # Распознать речь из аудио
        text, _segments, _engine, _engine_detail = run_stt_pipeline(
            temp_audio,
            lang="de-DE",
            punctuate=False,
            preprocessed=True,
        )

        # Сохранить аудио файл
        os.makedirs("video_audio", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = os.path.join("video_audio", f"video_{ts}.wav")
        
        # Копируем аудио файл
        import shutil
        shutil.copy(temp_audio, audio_filename)
        
        # Генерировать карточки из текста
        return auto_generate_cards_from_text(
            deck_id, text, use_ai_images, api_key,
            one_sentence_one_card=True,
            audio_path=audio_filename,
            audio_source="digital_hearing",
            audio_side="back",
            image_spend_cb=image_spend_cb,
        )

    except Exception as e:
        raise RuntimeError(f"Ошибка обработки видео: {e}")
    finally:
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass


# ==========================
# TTS
# ==========================

TTS_CACHE_TTL = 120
_TTS_CACHE: dict[str, dict[str, object]] = {}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_tts_cache_dir() -> str:
    base = globals().get("app_data_dir")
    if isinstance(base, str) and base.strip():
        cache_dir = os.path.join(base, "tts_cache")
    else:
        cache_dir = os.path.join(os.getcwd(), "tts_cache")
    return ensure_dir(cache_dir)


def get_tts_url(text: str, lang: str) -> str:
    clean = text.strip()
    if not clean:
        return ""
    lang_value = (lang or "de").strip() or "de"
    query = urllib.parse.quote(clean)
    return (
        "https://translate.google.com/translate_tts"
        f"?ie=UTF-8&q={query}&tl={lang_value}&client=tw-ob"
    )


def _cleanup_tts_cache() -> None:
    now = time.time()
    for key, entry in list(_TTS_CACHE.items()):
        ts = float(entry.get("ts", 0))
        path = entry.get("path")
        if now - ts > TTS_CACHE_TTL:
            if isinstance(path, str) and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
            _TTS_CACHE.pop(key, None)


def get_deck_tts_lang(deck_id: int | None, fallback: str = "de") -> str:
    if not deck_id:
        return fallback
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(decks);")
        columns = {row[1] for row in cur.fetchall()}
        if "tts_lang" not in columns:
            conn.close()
            return fallback
        cur.execute("SELECT tts_lang FROM decks WHERE id = ?;", (deck_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            value = row["tts_lang"] if isinstance(row, sqlite3.Row) else row[0]
            if value:
                return str(value).strip() or fallback
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
    return fallback


def get_selected_text_from_widget(widget: tk.Misc | None) -> str:
    targets = []
    if widget is not None:
        targets.append(widget)
        try:
            top = widget.winfo_toplevel()
        except Exception:
            top = None
        if top is not None and top not in targets:
            targets.append(top)
    for target in targets:
        try:
            selection = target.selection_get()
        except Exception:
            continue
        if selection:
            return selection.strip()
    return ""


def speak_google_tts(text: str, lang: str = "de") -> None:
    cleaned = SOUND_TAG_PATTERN.sub("", text or "").strip()
    if not cleaned:
        messagebox.showinfo("Озвучка", "Нет текста для озвучивания.")
        return
    _cleanup_tts_cache()
    cache_key = hashlib.sha1(f"{lang}:{cleaned}".encode("utf-8")).hexdigest()
    cached = _TTS_CACHE.get(cache_key)
    if cached:
        cached_path = cached.get("path")
        cached_ts = float(cached.get("ts", 0))
        if isinstance(cached_path, str) and os.path.exists(cached_path):
            if time.time() - cached_ts <= TTS_CACHE_TTL:
                play_audio_file(cached_path)
                return

    url = get_tts_url(cleaned, lang)
    if not url:
        messagebox.showinfo("Озвучка", "Нет текста для озвучивания.")
        return
    temp_path = os.path.join(get_tts_cache_dir(), f"tts_{int(time.time())}.mp3")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = response.read()
        with open(temp_path, "wb") as fh:
            fh.write(data)
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1024:
            raise RuntimeError("TTS файл не создан или пустой")
        _TTS_CACHE[cache_key] = {"path": temp_path, "ts": time.time()}
        play_audio_file(temp_path)
    except Exception as exc:
        messagebox.showerror("Озвучка", f"Не удалось озвучить: {exc}")


def speak_text(text: str):
    if not TTS_AVAILABLE or not _tts_engine:
        messagebox.showwarning(
            "TTS недоступен",
            "pyttsx3 не установлен или не работает.\n"
            "Установи: pip install pyttsx3"
        )
        return
    _tts_engine.say(text)
    _tts_engine.runAndWait()


def play_audio_file(path):
    """Воспроизвести аудио файл"""
    audio_path = path or ""
    resolved = resolve_sound_file(audio_path)
    print("[AUDIO] requested=", audio_path)
    print("[AUDIO] resolved =", resolved, "exists=", os.path.exists(resolved) if resolved else False)

    if not resolved or not os.path.exists(resolved):
        msg = f"Файл аудио не найден: {audio_path}"
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "audio.log"), "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        messagebox.showerror("Аудио", msg)
        return

    try:
        try:
            import vlc  # type: ignore

            instance = vlc.Instance()
            player = instance.media_player_new()
            media = instance.media_new(resolved)
            player.set_media(media)
            player.play()
            return
        except Exception as exc:  # noqa: BLE001
            os.makedirs("logs", exist_ok=True)
            with open(os.path.join("logs", "audio.log"), "a", encoding="utf-8") as fh:
                fh.write(f"VLC недоступен: {exc}\n")

        if WINSOUND_AVAILABLE and resolved.lower().endswith(".wav"):
            winsound.PlaySound(resolved, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return

        if TTS_AVAILABLE:
            speak_text(os.path.splitext(os.path.basename(resolved))[0])
            return

        messagebox.showerror("Аудио", "Нет доступного аудио движка для воспроизведения")
    except Exception as exc:  # noqa: BLE001 - не допускаем незахваченных исключений
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "audio.log"), "a", encoding="utf-8") as fh:
            fh.write(f"Ошибка воспроизведения: {exc}\n")
        messagebox.showerror("Аудио", f"Ошибка воспроизведения: {exc}")

# ==========================
# Устройства записи
# ==========================

def detect_default_mic_index() -> int | None:
    if not SR_AVAILABLE:
        return None
    try:
        devices = sr.Microphone.list_microphone_names()
    except Exception:
        return None

    for i, name in enumerate(devices):
        if "CABLE" in name.upper():
            return i
    for i, name in enumerate(devices):
        u = name.upper()
        if "STEREO MIX" in u or "СТЕРЕО" in u or "WHAT U HEAR" in u:
            return i
    return None


# ==========================
# Панель форматирования текста
# ==========================

def attach_simple_toolbar(parent_frame: ttk.Frame, text_widget: tk.Text):
    def apply_tag(tag, **cfg):
        if tag not in text_widget.tag_names():
            text_widget.tag_configure(tag, **cfg)
        try:
            text_widget.tag_add(tag, "sel.first", "sel.last")
        except tk.TclError:
            pass

    bar = ttk.Frame(parent_frame)
    bar.pack(fill=tk.X, padx=10, pady=(2, 4))

    ttk.Label(bar, text="Форматирование:").pack(side=tk.LEFT)

    ttk.Button(
        bar, text="Подчёркивание",
        command=lambda: apply_tag("underline", underline=1)
    ).pack(side=tk.LEFT, padx=3)

    ttk.Button(
        bar, text="Красн. подчёрк.",
        command=lambda: apply_tag("red_underline", underline=1, foreground="red")
    ).pack(side=tk.LEFT, padx=3)

    ttk.Button(
        bar, text="Маркер",
        command=lambda: apply_tag("marker_yellow", background="yellow")
    ).pack(side=tk.LEFT, padx=3)

# ==========================
# Контекстное меню для текстовых полей
# ==========================

def create_context_menu(widget):
    """Создать контекстное меню для текстового виджета"""
    menu = tk.Menu(widget, tearoff=0)
    
    # Добавляем команды контекстного меню
    menu.add_command(label="Вырезать", 
                     command=lambda: widget.event_generate('<<Cut>>'))
    menu.add_command(label="Копировать", 
                     command=lambda: widget.event_generate('<<Copy>>'))
    menu.add_command(label="Вставить", 
                     command=lambda: widget.event_generate('<<Paste>>'))
    menu.add_separator()
    menu.add_command(label="Выбрать все", 
                     command=lambda: widget.tag_add('sel', '1.0', 'end'))
    
    # Привязываем контекстное меню к виджету
    widget.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))
    
    # Для Entry виджетов
    if isinstance(widget, tk.Entry):
        widget.bind("<Control-c>", lambda e: widget.event_generate('<<Copy>>'))
        widget.bind("<Control-v>", lambda e: widget.event_generate('<<Paste>>'))
        widget.bind("<Control-x>", lambda e: widget.event_generate('<<Cut>>'))
        widget.bind("<Control-a>", lambda e: widget.select_range(0, tk.END))
    
    # Для Text виджетов
    elif isinstance(widget, tk.Text):
        widget.bind("<Control-c>", lambda e: widget.event_generate('<<Copy>>'))
        widget.bind("<Control-v>", lambda e: widget.event_generate('<<Paste>>'))
        widget.bind("<Control-x>", lambda e: widget.event_generate('<<Cut>>'))
        widget.bind("<Control-a>", lambda e: widget.tag_add('sel', '1.0', 'end'))


def render_card_layout(parent, card_data: dict, editable: bool = False) -> dict:
    colors = getattr(parent, "palette", None) or {}
    card_bg, card_text, card_border = get_card_surface_colors(parent)

    container = tk.Frame(
        parent,
        bg=card_bg,
        highlightthickness=1,
        highlightbackground=card_border,
        relief=tk.FLAT,
    )
    container.pack(fill=tk.BOTH, expand=True)

    inner = tk.Frame(container, bg=card_bg)
    inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    left = tk.Frame(inner, bg=card_bg)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right = tk.Frame(inner, bg=card_bg, width=260)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(12, 0))
    right.pack_propagate(False)

    text_frame = tk.Frame(left, bg=card_bg)
    text_frame.pack(fill=tk.BOTH, expand=True)

    text_widget = None
    if not card_data.get("skip_text_widget"):
        text_widget = tk.Text(text_frame, wrap=tk.WORD, height=10)
        style_card_surface_text(text_widget, colors)
        text_widget.pack(fill=tk.BOTH, expand=True)
        content = card_data.get("text") or ""
        if content:
            text_widget.insert("1.0", content)
        if not editable:
            text_widget.configure(state=tk.DISABLED)

    audio_frame = tk.Frame(left, bg=card_bg)
    audio_frame.pack(fill=tk.X, pady=(8, 0))

    media_frame = tk.Frame(right, bg=card_bg)
    media_frame.pack(fill=tk.BOTH, expand=True)

    image_label = None
    video_frame = None
    if not card_data.get("skip_media_widget"):
        image_label = ResizableImageLabel(
            media_frame,
            bg=card_bg,
            relief="flat",
            bd=0,
        )
        style_card_surface(image_label, colors)
        image_label.pack(fill=tk.BOTH, expand=True)

        video_frame = tk.Frame(media_frame, bg=card_bg)
        video_frame.pack(fill=tk.X, pady=(6, 0))

    layout = {
        "container": container,
        "left": left,
        "right": right,
        "text_frame": text_frame,
        "text_widget": text_widget,
        "audio_frame": audio_frame,
        "media_frame": media_frame,
        "image_label": image_label,
        "video_frame": video_frame,
        "editable": editable,
    }

    update_rendered_card(layout, card_data)
    return layout


_UNDERLINE_TOKEN_RE = re.compile(r"\[\[u:(double|wavy)\]\]|\[\[/u\]\]")


def strip_custom_underline_tokens(text: str) -> str:
    return _UNDERLINE_TOKEN_RE.sub("", text)


def update_rendered_card(layout: dict, card_data: dict) -> None:
    text_widget = layout.get("text_widget")
    if text_widget is not None:
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        content = card_data.get("text") or ""
        if content:
            text_widget.insert("1.0", strip_custom_underline_tokens(content))
        if not layout.get("editable", False):
            text_widget.configure(state=tk.DISABLED)

    image_label = layout.get("image_label")
    if image_label is not None:
        image_path = card_data.get("image_path")
        if image_path:
            image_label.load_image(image_path)
        else:
            image_label.config(image="", text="Нет изображения")

    video_frame = layout.get("video_frame")
    if video_frame is not None:
        for widget in video_frame.winfo_children():
            widget.destroy()
        video_path = card_data.get("video_path")
        if video_path:
            ttk.Label(video_frame, text="🎬 Видео прикреплено").pack(side=tk.LEFT)
            ttk.Button(
                video_frame,
                text="Открыть",
                command=lambda: open_in_external_player(video_path),
            ).pack(side=tk.LEFT, padx=6)
        else:
            ttk.Label(video_frame, text="Видео не прикреплено").pack(side=tk.LEFT)

    audio_frame = layout.get("audio_frame")
    if audio_frame is not None:
        for widget in audio_frame.winfo_children():
            widget.destroy()
        audio_path = card_data.get("audio_path")
        if audio_path:
            audio_widget = AudioPlayerWidget(audio_frame)
            audio_widget.pack(fill=tk.X)
            audio_widget.load(audio_path)
            audio_frame.audio_widget = audio_widget
        else:
            ttk.Label(audio_frame, text="Аудио не прикреплено").pack(anchor="w")


# PATCH: header inside card no overlap + fixed centered audio panel + image zoom +/- + single unified white scrollbar
class CardRenderer:
    def __init__(
        self,
        parent: tk.Widget,
        palette: dict | None = None,
        *,
        card_widget: CardWidget | None = None,
        editable: bool = False,
        width: int = 700,
        height: int = 420,
        show_image_toolbar: bool = False,
        image_layout: str = "side",
        show_media_placeholder: bool = True,
        on_media_state_change=None,
        enable_state_restore: bool = False,
        fixed_media_slot: tuple[int, int] | None = None,
        render_mode: str = "generic",
    ) -> None:
        self.palette = palette or getattr(parent, "palette", None) or {}
        self.on_media_state_change = on_media_state_change
        self.enable_state_restore = enable_state_restore
        self.show_media_placeholder = show_media_placeholder
        self.card_widget = card_widget
        self.image_zoom = 1.0
        self.media_width = 260
        self._orig_pil_cache = {}
        self._fixed_media_slot = fixed_media_slot
        self.render_mode = render_mode
        self._current_video_path: str | None = None
        self._current_audio_key: str | None = None
        self._current_show_back = False
        self._current_side_media: tuple[str, str] | None = None
        self._tk_img_cache: dict = {}
        self.video_player = None
        self._use_custom_text = False
        self._suppress_video = False
        self._suppress_image = False
        self._suppress_audio = False

        card_bg, card_text, _ = get_card_surface_colors(parent)
        self.card_bg = card_bg
        self.card_text = card_text
        self.container = tk.Frame(parent, bg=card_bg, width=width, height=height)
        self.container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        self.container.pack_propagate(False)
        self.container.grid_rowconfigure(1, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.header_frame = tk.Frame(self.container, bg=card_bg)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_label = tk.Label(
            self.header_frame,
            text="",
            bg=card_bg,
            fg=card_text,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.header_label.pack(fill=tk.X, padx=8, pady=(4, 2))

        self.content_frame = tk.Frame(self.container, bg=card_bg)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 4))
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.content_canvas = tk.Canvas(self.content_frame, bg="white", highlightthickness=0, bd=0)
        self.content_scrollbar = tk.Scrollbar(
            self.content_frame,
            orient="vertical",
            command=self.content_canvas.yview,
            bg="white",
            troughcolor="white",
            activebackground="white",
            highlightthickness=0,
            bd=0,
            width=12,
        )
        self.content_canvas.configure(yscrollcommand=self.content_scrollbar.set)
        self.content_canvas.grid(row=0, column=0, sticky="nsew")
        self.content_scrollbar.grid(row=0, column=1, sticky="ns")

        self.content_inner = tk.Frame(self.content_canvas, bg="white")
        self.content_window = self.content_canvas.create_window((0, 0), window=self.content_inner, anchor="nw")
        self.content_inner.bind(
            "<Configure>",
            lambda _e: self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all")),
        )
        self.content_canvas.bind(
            "<Configure>",
            lambda e: self.content_canvas.itemconfigure(self.content_window, width=e.width),
        )

        self.content_inner.grid_rowconfigure(0, weight=1)
        self.content_inner.grid_columnconfigure(0, weight=1)
        self.content_row = tk.Frame(self.content_inner, bg="white")
        self.content_row.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.content_row.grid_columnconfigure(1, weight=1)

        self.media_col = tk.Frame(self.content_row, bg="white", width=self.media_width)
        self.media_col.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        self.media_col.grid_propagate(False)

        self.image_container = tk.Frame(self.media_col, bg="white")
        if self._fixed_media_slot:
            self.image_container.config(width=self._fixed_media_slot[0], height=self._fixed_media_slot[1])
            self.image_container.pack_propagate(False)
        self.image_container.pack(fill=tk.BOTH, expand=True)
        self.image_label = ResizableImageLabel(
            self.image_container,
            bg="white",
            relief="flat",
            bd=0,
            enable_mousewheel=False,
        )
        self.image_label._orig_pil_cache = self._orig_pil_cache
        if self._fixed_media_slot:
            self.image_label.set_fixed_slot_size(self._fixed_media_slot[0], self._fixed_media_slot[1])
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.zoom_frame = tk.Frame(self.media_col, bg="white")
        self.zoom_frame.pack(anchor="w", pady=(6, 0))
        tk.Button(
            self.zoom_frame,
            text="+",
            width=2,
            command=lambda: self.adjust_image_zoom(1.1),
            bg="white",
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            self.zoom_frame,
            text="-",
            width=2,
            command=lambda: self.adjust_image_zoom(0.9),
            bg="white",
        ).pack(side=tk.LEFT)

        self.video_frame = tk.Frame(self.media_col, bg="white")
        self.video_frame.pack(fill=tk.X, pady=(6, 0))

        self.text_col = tk.Frame(self.content_row, bg="white")
        self.text_col.grid(row=0, column=1, sticky="nsew")
        self.text_col.grid_rowconfigure(0, weight=1)
        self.text_col.grid_columnconfigure(0, weight=1)

        self.text_frame = tk.Frame(self.text_col, bg="white")
        self.text_frame.grid(row=0, column=0, sticky="nsew")
        self.text_frame.grid_columnconfigure(0, weight=1)
        self.text_frame.grid_rowconfigure(0, weight=1)

        if editable:
            self.front_text = tk.Text(self.text_frame, wrap=tk.WORD, height=10, bg="white", fg=card_text)
            self.back_text = tk.Text(self.text_frame, wrap=tk.WORD, height=10, bg="white", fg=card_text)
        else:
            self.front_text = tk.Label(
                self.text_frame,
                text="",
                bg="white",
                fg=card_text,
                justify="left",
                anchor="nw",
                font=("Segoe UI", 12),
            )
            self.back_text = tk.Label(
                self.text_frame,
                text="",
                bg="white",
                fg=card_text,
                justify="left",
                anchor="nw",
                font=("Segoe UI", 12),
            )

        self.front_text.grid(row=0, column=0, sticky="nsew")
        self.back_text.grid(row=0, column=0, sticky="nsew")
        self.back_text.grid_remove()

        self.custom_text_frame = tk.Frame(self.text_frame, bg="white")
        self.custom_text_frame.grid(row=0, column=0, sticky="nsew")
        self.custom_text_frame.grid_remove()
        self.html_view = CardHtmlView(self.custom_text_frame)

        self.text_col.bind("<Configure>", self._update_wraplength)

        self.audio_panel = tk.Frame(self.container, bg="white")
        self.audio_panel.grid(row=2, column=0, sticky="ew", pady=(4, 2))
        self.audio_panel.grid_columnconfigure(0, weight=1)
        self.audio_panel.grid_columnconfigure(2, weight=1)
        self.audio_center = tk.Frame(self.audio_panel, bg="white")
        self.audio_center.grid(row=0, column=1)
        self.audio_frame = tk.Frame(self.audio_center, bg="white")
        self.audio_frame.pack()

        self._bind_mousewheel(self.content_canvas)
        self._bind_mousewheel(self.content_inner)

    def _bind_mousewheel(self, target: tk.Widget) -> None:
        def _on_mousewheel(event):
            if event.num == 4:
                self.content_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.content_canvas.yview_scroll(1, "units")
            elif event.delta:
                self.content_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        def _bind_all(_event):
            self.content_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.content_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.content_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_all(_event):
            self.content_canvas.unbind_all("<MouseWheel>")
            self.content_canvas.unbind_all("<Button-4>")
            self.content_canvas.unbind_all("<Button-5>")

        target.bind("<Enter>", _bind_all)
        target.bind("<Leave>", _unbind_all)

    def _update_wraplength(self, event) -> None:
        width = max(100, int(event.width) - 8)
        if isinstance(self.front_text, tk.Label):
            self.front_text.configure(wraplength=width)
        if isinstance(self.back_text, tk.Label):
            self.back_text.configure(wraplength=width)

    def _resolve_image_path(self, card: dict, show_back: bool, image_override: str | None = None) -> str | None:
        if image_override is not None:
            return resolve_media_path(image_override)
        if show_back:
            path = card.get("back_image_path") or card.get("front_image_path") or card.get("image_path")
        else:
            path = card.get("front_image_path") or card.get("image_path")
        return resolve_media_path(path)

    def _resolve_video_path(
        self,
        card: dict,
        show_back: bool,
        video_override: str | None = None,
    ) -> str | None:
        if video_override is not None:
            return resolve_media_path(video_override)
        if show_back:
            video_path = card.get("back_video_path") or card.get("video_path")
        else:
            video_path = card.get("front_video_path") or card.get("video_path")
        if not video_path and (card.get("id") is not None or card.get("note_id") is not None):
            video_path = find_video_media_path_for_side(card, "back" if show_back else "front")
        return resolve_media_path(video_path)

    def _select_side_media(
        self,
        card: dict,
        *,
        show_back: bool,
        image_override: str | None = None,
        video_override: str | None = None,
    ) -> tuple[str, str] | None:
        image_path = self._resolve_image_path(card, show_back, image_override=image_override)
        video_path = self._resolve_video_path(card, show_back, video_override=video_override)
        return get_side_media({"image_path": image_path, "video_path": video_path})

    def _apply_custom_text(self, items: list, *, card_bg: str, card_text: str) -> None:
        self._use_custom_text = True
        for widget in self.custom_text_frame.winfo_children():
            widget.destroy()
        row = 0
        col = 0
        max_cols = 3
        for item in items:
            if isinstance(item, tk.Frame):
                item.grid(
                    row=row,
                    column=col,
                    sticky="w",
                    padx=5,
                    pady=2,
                    in_=self.custom_text_frame,
                )
            else:
                label = tk.Label(
                    self.custom_text_frame,
                    text=item,
                    bg="white",
                    fg=card_text,
                    font=("Segoe UI", 12),
                )
                label.grid(row=row, column=col, sticky="w", padx=5, pady=2)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def update_text(
        self,
        card: dict,
        *,
        show_back: bool = False,
        image_override: str | None = None,
        video_override: str | None = None,
        custom_items: list | None = None,
    ) -> None:
        card_bg, card_text, _ = get_card_surface_colors(self.container)
        front_text = card.get("front") or ""
        back_text = card.get("back") or ""
        front_html = card.get("front_html")
        back_html = card.get("back_html")
        if not front_html and _looks_like_html(front_text):
            front_html = front_text
        if not back_html and _looks_like_html(back_text):
            back_html = back_text
        front_rich = deserialize_rich_doc(card.get("front_rich"))
        back_rich = deserialize_rich_doc(card.get("back_rich"))
        rich_doc = back_rich if show_back else front_rich
        use_rich = bool(rich_doc)
        html_value = back_html if show_back else front_html
        use_html = bool(html_value) and not use_rich
        html_media = extract_media_from_html(html_value or "") if use_html else {}
        self._suppress_video = bool(use_html and html_media.get("video"))
        self._suppress_image = bool(use_html and html_media.get("image"))
        self._suppress_audio = bool(use_html and html_media.get("audio"))
        if use_html and html_media.get("audio") and not card.get("audio_entries") and not card.get("audio_path"):
            card = dict(card)
            card["audio_path"] = html_media.get("audio")
        if use_html:
            if image_override is None and html_media.get("image") and not self._suppress_image:
                image_override = html_media.get("image")
            if video_override is None and html_media.get("video") and not self._suppress_video:
                video_override = html_media.get("video")
        else:
            self._suppress_video = False
            self._suppress_image = False
            self._suppress_audio = False
        print(
            "[RENDER TEXT]",
            "mode=",
            self.render_mode,
            "side=",
            "back" if show_back else "front",
            "use_rich=",
            use_rich,
        )
        if use_html:
            try:
                self.media_col.grid_remove()
                self.audio_panel.grid_remove()
                self.text_col.grid_configure(column=0, columnspan=2)
            except Exception:
                pass
        else:
            try:
                self.media_col.grid()
                self.audio_panel.grid()
                self.text_col.grid_configure(column=1, columnspan=1)
            except Exception:
                pass
        if custom_items is not None:
            self._apply_custom_text(custom_items, card_bg=card_bg, card_text=card_text)
        elif use_rich:
            self._use_custom_text = True
            render_rich_to_container(self.custom_text_frame, rich_doc, card_bg=card_bg, card_text_color=card_text)
        elif use_html:
            self._use_custom_text = True
            self.html_view.set_html(html_value or "", card_bg=card_bg, card_text_color=card_text)
        else:
            self._use_custom_text = False
            if isinstance(self.front_text, tk.Text):
                self.front_text.delete("1.0", tk.END)
                self.front_text.insert("1.0", front_text)
            else:
                self.front_text.configure(text=front_text)
            if isinstance(self.back_text, tk.Text):
                self.back_text.delete("1.0", tk.END)
                self.back_text.insert("1.0", back_text)
            else:
                self.back_text.configure(text=back_text)
        self._current_show_back = show_back
        side_media = self._select_side_media(
            card,
            show_back=show_back,
            image_override=image_override,
            video_override=video_override,
        )
        self._current_side_media = side_media
        image_path = side_media[1] if side_media and side_media[0] == "image" else None
        self._current_video_path = side_media[1] if side_media and side_media[0] == "video" else None
        if self._suppress_image:
            image_path = None
        if self._suppress_video:
            self._current_video_path = None
        if show_back:
            self.front_text.grid_remove()
            self.back_text.grid()
        else:
            self.back_text.grid_remove()
            self.front_text.grid()
        if custom_items is not None:
            self.front_text.grid_remove()
            self.back_text.grid_remove()
            self.custom_text_frame.grid()
        elif use_rich:
            self.front_text.grid_remove()
            self.back_text.grid_remove()
            self.custom_text_frame.grid()
        elif use_html:
            self.front_text.grid_remove()
            self.back_text.grid_remove()
            self.custom_text_frame.grid()
        else:
            self.custom_text_frame.grid_remove()

        self.image_label._render_mode = self.render_mode
        self.image_label._render_card_id = card.get("id") or card.get("note_id")
        if image_path:
            render_key = (card.get("id") or card.get("note_id") or "card", "back" if show_back else "front")
            self.image_label.load_image(
                image_path,
                key=render_key,
                zoom=self.image_zoom,
                container_widget=self.image_container,
            )
        else:
            placeholder_text = "Нет изображения" if self.show_media_placeholder else ""
            self.image_label.config(image="", text=placeholder_text)

    def _get_audio_entries(self, card: dict, prefer_side: str | None) -> list[dict]:
        if "audio_entries" in card:
            return list(card.get("audio_entries") or [])
        audio_path = card.get("audio_path")
        if audio_path:
            label = os.path.basename(audio_path) or "Аудио"
            return [
                {
                    "path": audio_path,
                    "label": label,
                    "side": prefer_side or "back",
                    "missing": False,
                    "media_id": None,
                }
            ]
        if card.get("id") is not None:
            return get_card_audio_entries(card, prefer_side=prefer_side)
        return []

    def update_media(self, card: dict, *, prefer_audio_side: str | None = None) -> None:
        prefer_side = prefer_audio_side or "back"
        if self._suppress_audio:
            for widget in self.audio_frame.winfo_children():
                widget.destroy()
        else:
            entries = self._get_audio_entries(card, prefer_side)
            if entries:
                display_audio_entries_on_frame(self.audio_frame, entries)
                audio_widget = getattr(self.audio_frame, "audio_widget", None)
                if audio_widget is not None:
                    audio_widget.on_state_change = self.on_media_state_change
                    entry_map = getattr(self.audio_frame, "audio_entry_map", {}) or {}
                    selection = getattr(self.audio_frame, "audio_selector_var", None)

                    def _apply_audio_selection():
                        selected_label = selection.get() if selection else None
                        entry = entry_map.get(selected_label) or (list(entry_map.values())[0] if entry_map else None)
                        if not entry:
                            return
                        media_key = _build_media_key(entry.get("media_id"), entry.get("path"))
                        audio_widget.set_media_key(media_key)
                        if self.enable_state_restore and card.get("id") is not None:
                            audio_widget.apply_state(load_media_state(card["id"], media_key))

                    selector = getattr(self.audio_frame, "audio_selector", None)
                    if selector is not None:
                        selector.bind("<<ComboboxSelected>>", lambda _e: _apply_audio_selection())
                    _apply_audio_selection()
            else:
                for widget in self.audio_frame.winfo_children():
                    widget.destroy()
                if self.show_media_placeholder:
                    ttk.Label(self.audio_frame, text="Аудио не прикреплено").pack(anchor="center")

        self._render_video(card)

    def _render_video(self, card: dict) -> None:
        for widget in self.video_frame.winfo_children():
            widget.destroy()
        if self._suppress_video:
            return
        side_media = self._current_side_media or self._select_side_media(card, show_back=self._current_show_back)
        if not side_media or side_media[0] != "video":
            self._current_video_path = None
            self.video_player = None
            if not side_media and self.show_media_placeholder:
                ttk.Label(self.video_frame, text="Видео не прикреплено").pack(anchor="w", padx=5, pady=5)
            return

        video_path = side_media[1]
        if not os.path.exists(video_path):
            print(f"[Video] Файл видео не найден: {video_path}")
            self._current_video_path = None
            self.video_player = None
            if self.show_media_placeholder:
                ttk.Label(self.video_frame, text="Видео не найдено").pack(anchor="w", padx=5, pady=5)
            return

        if is_vlc_available():
            try:
                player = VlcPlayerWidget(
                    self.video_frame,
                    video_path,
                    width=420,
                    height=200,
                    on_state_change=self.on_media_state_change,
                )
                if not player.ensure_embedded():
                    print("[VLC] Не удалось встроить видео в контейнер.")
                    player.frame.destroy()
                else:
                    player.pack(anchor="w")
                    if self.enable_state_restore and card.get("id") is not None:
                        media_entries = get_media_for_card(card.get("id"), card.get("note_id"))
                        media_id = None
                        for entry in media_entries:
                            media_type = (entry.get("media_type") or entry.get("type") or "").lower()
                            if media_type == "video" and entry.get("path") == video_path:
                                media_id = entry.get("id")
                                break
                        media_key = _build_media_key(media_id, video_path)
                        player.set_media_key(media_key)
                        player.apply_state(load_media_state(card["id"], media_key))
                    self.video_frame.vlc_player = player
                    self.video_player = player
                    self._current_video_path = video_path
                    return
            except Exception as exc:
                print(f"[VLC] Ошибка embed видео: {exc}")

        print(f"[Video] Не удалось встроить видео: {video_path}")
        ttk.Button(
            self.video_frame,
            text="Открыть видео внешним плеером",
            command=lambda: open_in_external_player(video_path),
        ).pack(anchor="w", padx=5, pady=5)
        self.video_player = None
        self._current_video_path = video_path

    def render(
        self,
        card: dict,
        *,
        show_back: bool = False,
        prefer_audio_side: str | None = None,
        image_override: str | None = None,
        video_override: str | None = None,
        custom_items: list | None = None,
        header_text: str | None = None,
    ) -> None:
        if header_text is not None:
            self.set_header_text(header_text)
        self.update_text(
            card,
            show_back=show_back,
            image_override=image_override,
            video_override=video_override,
            custom_items=custom_items,
        )
        self.update_media(card, prefer_audio_side=prefer_audio_side)

    def get_audio_widget(self):
        return getattr(self.audio_frame, "audio_widget", None)

    def set_header_text(self, text: str) -> None:
        self.header_label.config(text=text)

    def get_repeat_media_slot_size(self) -> tuple[int, int]:
        if self._fixed_media_slot:
            return int(self._fixed_media_slot[0]), int(self._fixed_media_slot[1])
        container = getattr(self, "image_container", None)
        if container is None:
            return int(REPEAT_MEDIA_SLOT_SIZE[0]), int(REPEAT_MEDIA_SLOT_SIZE[1])
        width = container.winfo_width() or container.winfo_reqwidth()
        height = container.winfo_height() or container.winfo_reqheight()
        if width <= 0 or height <= 0:
            return int(REPEAT_MEDIA_SLOT_SIZE[0]), int(REPEAT_MEDIA_SLOT_SIZE[1])
        return int(width), int(height)

    def render_image_to_container(
        self,
        container: tk.Widget,
        image_path: str | None,
        cache_key: str,
        target_w: int,
        target_h: int,
    ) -> bool:
        if not hasattr(self, "_tk_img_cache"):
            self._tk_img_cache = {}
        label = getattr(container, "_preview_image_label", None)
        if label is None:
            if container is self.image_container and hasattr(self, "image_label"):
                label = self.image_label
            else:
                label = tk.Label(container, bg="white", bd=0, relief=tk.FLAT)
                label.pack(fill=tk.BOTH, expand=True)
            container._preview_image_label = label
        resolved_path = resolve_media_path(image_path)
        if resolved_path:
            resolved_path = os.path.abspath(resolved_path)
        exists = bool(resolved_path and os.path.exists(resolved_path))
        if not exists:
            label.config(image="", text="Нет изображения")
            label.image = None
            return False
        try:
            if not PIL_AVAILABLE:
                raise RuntimeError("Pillow не доступен для предпросмотра")
            img = Image.open(resolved_path)
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGBA", "RGB"):
                img = img.convert("RGBA")
            target_w = max(1, int(target_w))
            target_h = max(1, int(target_h))
            img.thumbnail((target_w, target_h), _pil_lanczos())
            photo = ImageTk.PhotoImage(img)
            self._tk_img_cache[cache_key] = photo
            label.config(image=photo, text="")
            label.image = photo
            return True
        except Exception as exc:
            log_image_error(resolved_path or "missing_path", exc)
            label.config(image="", text="Нет изображения")
            label.image = None
            return False

    def adjust_image_zoom(self, factor: float) -> None:
        self.image_zoom = max(0.1, min(3.0, self.image_zoom * float(factor)))
        self.image_label.set_zoom_factor(self.image_zoom)


# PATCH: manual preview text rendering fixed + toolbar rich formatting exported and rendered in preview/repeat/playback/intro
def export_rich_from_editor(text_widget: tk.Text) -> dict:
    text_value = text_widget.get("1.0", tk.END).rstrip("\n")
    tags_payload: list[dict] = []
    for tag in text_widget.tag_names():
        if tag == "sel":
            continue
        ranges = text_widget.tag_ranges(tag)
        if not ranges:
            continue
        config: dict[str, object] = {}
        for key in ("font", "foreground", "background", "underline", "offset", "justify", "elide"):
            value = text_widget.tag_cget(tag, key)
            if value in ("", None):
                continue
            if key in ("underline", "offset"):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            config[key] = value
        range_pairs: list[tuple[str, str]] = []
        for idx in range(0, len(ranges), 2):
            start = str(ranges[idx])
            end = str(ranges[idx + 1])
            range_pairs.append((start, end))
        tags_payload.append(
            {
                "name": tag,
                "config": config,
                "ranges": range_pairs,
            }
        )
    return {
        "text": text_value,
        "tags": tags_payload,
    }


def text_widget_to_html(text_widget: tk.Text) -> str:
    dump = text_widget.dump("1.0", "end-1c", tag=True, text=True)
    active_tags: set[str] = set()
    html_parts: list[str] = []

    def _style_from_tags(tags: set[str]) -> dict[str, str | bool | None]:
        style: dict[str, str | bool | None] = {
            "bold": False,
            "italic": False,
            "underline": False,
            "underline_style": None,
            "color": None,
            "background": None,
        }
        for tag in tags:
            if tag == "hidden_token":
                continue
            if tag == "bold":
                style["bold"] = True
            if tag == "italic":
                style["italic"] = True
            if tag == "underline":
                style["underline"] = True
            if tag == "underline_double":
                style["underline"] = True
                style["underline_style"] = "double"
            if tag == "underline_wavy":
                style["underline"] = True
                style["underline_style"] = "wavy"
            if tag.startswith("color_"):
                style["color"] = tag.split("color_", 1)[1]
            if tag.startswith("marker_"):
                style["background"] = tag.split("marker_", 1)[1]
            fg = text_widget.tag_cget(tag, "foreground")
            bg = text_widget.tag_cget(tag, "background")
            if fg:
                style["color"] = fg
            if bg:
                style["background"] = bg
        return style

    def _wrap_html(text: str, style: dict[str, str | bool | None]) -> str:
        if not text:
            return ""
        escaped = html.escape(text)
        escaped = strip_custom_underline_tokens(escaped)
        escaped = escaped.replace("\n", "<br>")
        prefix = ""
        suffix = ""
        if style.get("bold"):
            prefix += "<strong>"
            suffix = "</strong>" + suffix
        if style.get("italic"):
            prefix += "<em>"
            suffix = "</em>" + suffix
        if style.get("underline") and not style.get("underline_style"):
            prefix += "<u>"
            suffix = "</u>" + suffix
        span_styles: list[str] = []
        if style.get("color"):
            span_styles.append(f"color: {style['color']}")
        if style.get("background"):
            span_styles.append(f"background-color: {style['background']}")
        if style.get("underline_style"):
            span_styles.append(f"text-decoration: underline {style['underline_style']}")
        if span_styles:
            prefix += f"<span style=\"{' ; '.join(span_styles)}\">"
            suffix = "</span>" + suffix
        return f"{prefix}{escaped}{suffix}"

    for kind, value, _index in dump:
        if kind == "tagon":
            active_tags.add(value)
        elif kind == "tagoff":
            active_tags.discard(value)
        elif kind == "text":
            if "hidden_token" in active_tags:
                continue
            style = _style_from_tags(active_tags)
            html_parts.append(_wrap_html(value, style))

    return "".join(html_parts).strip()


def serialize_rich_doc(rich_doc: dict | None) -> str | None:
    if rich_doc is None:
        return None
    try:
        return json.dumps(rich_doc, ensure_ascii=False)
    except Exception:
        return None


def deserialize_rich_doc(value: str | None) -> dict | None:
    if not value:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def render_rich_to_container(parent: tk.Widget, rich_doc: dict, card_bg: str, card_text_color: str) -> tk.Text:
    for child in parent.winfo_children():
        child.destroy()
    text_widget = tk.Text(
        parent,
        wrap=tk.WORD,
        bg="white",
        fg=card_text_color,
        relief=tk.FLAT,
        bd=0,
        height=10,
    )
    text_widget.pack(fill=tk.BOTH, expand=True)
    text_widget.configure(state=tk.NORMAL)
    text_widget.delete("1.0", tk.END)
    text_widget.insert("1.0", rich_doc.get("text") or "")
    for tag_info in rich_doc.get("tags", []) or []:
        tag_name = tag_info.get("name")
        if not tag_name:
            continue
        config = tag_info.get("config") or {}
        safe_config: dict[str, object] = {}
        for key, value in config.items():
            if value in ("", None):
                continue
            if key in ("underline", "offset"):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            safe_config[key] = value
        if safe_config:
            try:
                text_widget.tag_configure(tag_name, **safe_config)
            except tk.TclError:
                pass
        ranges = tag_info.get("ranges") or []
        for start, end in ranges:
            try:
                text_widget.tag_add(tag_name, start, end)
            except tk.TclError:
                pass
    text_widget.configure(state=tk.DISABLED)
    return text_widget


_HTML_TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")


def _looks_like_html(value: str) -> bool:
    if not value:
        return False
    return bool(_HTML_TAG_RE.search(value))


def _file_url_to_path(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file":
        return url
    path = urllib.parse.unquote(parsed.path)
    if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return path or None


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._list_stack: list[dict[str, int | str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br"}:
            self.parts.append("\n")
        elif tag in {"p", "div"}:
            self.parts.append("\n")
        elif tag == "ul":
            self._list_stack.append({"type": "ul", "index": 0})
        elif tag == "ol":
            self._list_stack.append({"type": "ol", "index": 0})
        elif tag == "li":
            prefix = "• "
            if self._list_stack:
                stack = self._list_stack[-1]
                if stack["type"] == "ol":
                    stack["index"] = int(stack.get("index", 0)) + 1
                    prefix = f"{stack['index']}. "
            self.parts.append("\n" + prefix)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"ul", "ol"} and self._list_stack:
            self._list_stack.pop()
        if tag in {"p", "div"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)


def plain_text_from_html(html_text: str) -> str:
    if not html_text:
        return ""
    parser = _HtmlTextExtractor()
    parser.feed(html_text)
    return "".join(parser.parts).strip()


def ensure_plain_text(text: str | None) -> str:
    if not text:
        return ""
    if _looks_like_html(text):
        return plain_text_from_html(text)
    return text


class _HtmlMediaExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.media: dict[str, str | None] = {"image": None, "video": None, "audio": None, "poster": None}
        self._media_context: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "img" and not self.media.get("image"):
            src = attrs_dict.get("src")
            self.media["image"] = _file_url_to_path(src) or src
            return
        if tag == "video":
            self._media_context = "video"
            if not self.media.get("poster"):
                poster = attrs_dict.get("poster")
                self.media["poster"] = _file_url_to_path(poster) or poster
            src = attrs_dict.get("src")
            if src and not self.media.get("video"):
                self.media["video"] = _file_url_to_path(src) or src
            return
        if tag == "audio":
            self._media_context = "audio"
            src = attrs_dict.get("src")
            if src and not self.media.get("audio"):
                self.media["audio"] = _file_url_to_path(src) or src
            return
        if tag == "source" and self._media_context in {"video", "audio"}:
            src = attrs_dict.get("src")
            if not src:
                return
            key = self._media_context
            if not self.media.get(key):
                self.media[key] = _file_url_to_path(src) or src

    def handle_endtag(self, tag: str) -> None:
        if tag in {"video", "audio"}:
            self._media_context = None


def extract_media_from_html(html_text: str) -> dict[str, str | None]:
    if not html_text:
        return {"image": None, "video": None, "audio": None, "poster": None}
    parser = _HtmlMediaExtractor()
    parser.feed(html_text)
    return parser.media


def _parse_inline_style(style_text: str) -> dict[str, str]:
    styles: dict[str, str] = {}
    if not style_text:
        return styles
    for chunk in style_text.split(";"):
        key, _, value = chunk.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            styles[key] = value
    return styles


class _HtmlTextRenderer(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.segments: list[tuple[str, dict[str, str | bool | None]]] = []
        self._style_stack: list[dict[str, str | bool | None]] = [
            {
                "bold": False,
                "italic": False,
                "underline": False,
                "underline_style": None,
                "color": None,
                "background": None,
                "link": None,
            }
        ]
        self._list_stack: list[dict[str, int | str]] = []

    def _current_style(self) -> dict[str, str | bool | None]:
        return dict(self._style_stack[-1])

    def _append_text(self, text: str, style: dict[str, str | bool | None]) -> None:
        if text:
            self.segments.append((text, style))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "br":
            self._append_text("\n", self._current_style())
            return
        if tag in {"p", "div"}:
            self._append_text("\n", self._current_style())
            return
        if tag == "ul":
            self._list_stack.append({"type": "ul", "index": 0})
            return
        if tag == "ol":
            self._list_stack.append({"type": "ol", "index": 0})
            return
        if tag == "li":
            prefix = "• "
            if self._list_stack:
                stack = self._list_stack[-1]
                if stack["type"] == "ol":
                    stack["index"] = int(stack.get("index", 0)) + 1
                    prefix = f"{stack['index']}. "
            self._append_text("\n" + prefix, self._current_style())
            return

        style = self._current_style()
        if tag in {"b", "strong"}:
            style["bold"] = True
        if tag in {"i", "em"}:
            style["italic"] = True
        if tag == "u":
            style["underline"] = True
        if tag == "span":
            inline_style = _parse_inline_style(attrs_dict.get("style", "") or "")
            color = inline_style.get("color")
            background = inline_style.get("background-color") or inline_style.get("background")
            decoration = inline_style.get("text-decoration") or ""
            if color:
                style["color"] = color
            if background:
                style["background"] = background
            if "underline" in decoration:
                style["underline"] = True
                if "double" in decoration:
                    style["underline_style"] = "double"
                elif "wavy" in decoration:
                    style["underline_style"] = "wavy"
        if tag == "a":
            href = attrs_dict.get("href")
            if href:
                style["link"] = href
                style["underline"] = True
        self._style_stack.append(style)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"ul", "ol"} and self._list_stack:
            self._list_stack.pop()
        if tag in {"b", "strong", "i", "em", "u", "span", "a"}:
            if len(self._style_stack) > 1:
                self._style_stack.pop()
        if tag in {"p", "div"}:
            self._append_text("\n", self._current_style())

    def handle_data(self, data: str) -> None:
        if data:
            self._append_text(data, self._current_style())


class CardHtmlView:
    def __init__(self, parent: tk.Widget) -> None:
        self.parent = parent
        self.text_widget: tk.Text | None = None
        self.video_player = None
        self.image_label: ResizableImageLabel | None = None
        self.audio_widget: AudioPlayerWidget | None = None

    def clear(self) -> None:
        for child in self.parent.winfo_children():
            child.destroy()
        self.text_widget = None
        self.video_player = None
        self.image_label = None
        self.audio_widget = None

    def set_html(self, html_text: str, card_bg: str, card_text_color: str) -> None:
        self.clear()
        container = tk.Frame(self.parent, bg=card_bg)
        container.pack(fill=tk.BOTH, expand=True)

        media = extract_media_from_html(html_text or "")
        media_frame = None
        if any(media.get(key) for key in ("video", "image", "audio")):
            media_frame = tk.Frame(container, bg=card_bg)
            media_frame.pack(fill=tk.X, padx=6, pady=(6, 0))

        if media_frame is not None:
            video_path = resolve_media_path(media.get("video"))
            if video_path and os.path.exists(video_path):
                if is_vlc_available():
                    try:
                        self.video_player = VlcPlayerWidget(
                            media_frame,
                            video_path,
                            width=480,
                            height=270,
                        )
                        if self.video_player.ensure_embedded():
                            self.video_player.pack(anchor="w", pady=(0, 6))
                            try:
                                self.video_player.play()
                            except Exception:
                                pass
                        else:
                            self.video_player.frame.destroy()
                            self.video_player = None
                    except Exception:
                        self.video_player = None
                if self.video_player is None:
                    ttk.Button(
                        media_frame,
                        text="Открыть видео во внешнем плеере",
                        command=lambda: open_in_external_player(video_path),
                    ).pack(anchor="w", pady=(0, 6))
            elif media.get("video"):
                ttk.Label(media_frame, text="Видео не найдено").pack(anchor="w", pady=(0, 6))

            image_path = resolve_media_path(media.get("image"))
            if image_path and os.path.exists(image_path):
                self.image_label = ResizableImageLabel(media_frame, bg=card_bg, relief="flat", bd=0)
                self.image_label.pack(anchor="w", pady=(0, 6))
                self.image_label.load_image(image_path)
            elif media.get("image"):
                ttk.Label(media_frame, text="Изображение не найдено").pack(anchor="w", pady=(0, 6))

            audio_path = resolve_media_path(media.get("audio"))
            if audio_path and os.path.exists(audio_path):
                self.audio_widget = AudioPlayerWidget(media_frame)
                self.audio_widget.pack(anchor="w", fill=tk.X, pady=(0, 6))
                self.audio_widget.load(audio_path)
            elif media.get("audio"):
                ttk.Label(media_frame, text="Аудио не найдено").pack(anchor="w", pady=(0, 6))

        text_container = tk.Frame(container, bg=card_bg)
        text_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 6))
        self.text_widget = show_card_html(text_container, html_text, card_bg=card_bg, card_text_color=card_text_color)


def show_card_html(parent: tk.Widget, html_text: str, card_bg: str, card_text_color: str) -> tk.Text:
    for child in parent.winfo_children():
        child.destroy()
    text_widget = tk.Text(
        parent,
        wrap=tk.WORD,
        bg=card_bg,
        fg=card_text_color,
        relief=tk.FLAT,
        bd=0,
        height=10,
    )
    text_widget.pack(fill=tk.BOTH, expand=True)
    base_font = tkfont.Font(font=text_widget.cget("font"))

    renderer = _HtmlTextRenderer()
    renderer.feed(html_text or "")
    tag_cache: dict[tuple, str] = {}
    link_cache: dict[str, str] = {}

    def _tag_for_style(style: dict[str, str | bool | None]) -> str | None:
        key = (
            bool(style.get("bold")),
            bool(style.get("italic")),
            bool(style.get("underline")),
            style.get("underline_style"),
            style.get("color"),
            style.get("background"),
            style.get("link"),
        )
        if not any(key):
            return None
        if key in tag_cache:
            return tag_cache[key]
        tag_name = f"html_{len(tag_cache)}"
        tag_cache[key] = tag_name
        font = base_font.copy()
        if style.get("bold"):
            font.configure(weight="bold")
        if style.get("italic"):
            font.configure(slant="italic")
        underline_flag = 1 if style.get("underline") else 0
        text_widget.tag_configure(tag_name, font=font, underline=underline_flag)
        if style.get("color"):
            text_widget.tag_configure(tag_name, foreground=str(style["color"]))
        if style.get("background"):
            text_widget.tag_configure(tag_name, background=str(style["background"]))
        link = style.get("link")
        if link:
            text_widget.tag_configure(tag_name, foreground="#2563EB", underline=1)
            link_cache[tag_name] = str(link)
        return tag_name

    def _open_link(tag: str) -> None:
        target = link_cache.get(tag)
        if target:
            webbrowser.open(target)

    for text, style in renderer.segments:
        if not text:
            continue
        tag_name = _tag_for_style(style)
        if tag_name:
            text_widget.insert(tk.END, text, tag_name)
        else:
            text_widget.insert(tk.END, text)

    for tag_name in link_cache:
        text_widget.tag_bind(tag_name, "<Button-1>", lambda _e, t=tag_name: _open_link(t))
        text_widget.tag_bind(tag_name, "<Enter>", lambda _e: text_widget.config(cursor="hand2"))
        text_widget.tag_bind(tag_name, "<Leave>", lambda _e: text_widget.config(cursor=""))

    text_widget.configure(state=tk.DISABLED)
    return text_widget


def render_card_html(deck_id: int, ctx: dict[str, str] | None = None) -> tuple[str, str]:
    ctx = dict(ctx or {})
    for key in (
        "front_html",
        "back_html",
        "front_image",
        "back_image",
        "front_video",
        "back_video",
        "front_audio",
        "back_audio",
        "image",
        "video",
        "audio",
    ):
        ctx.setdefault(key, "")
    if not ctx.get("image"):
        ctx["image"] = ctx.get("front_image") or ctx.get("back_image") or ""
    if not ctx.get("video"):
        ctx["video"] = ctx.get("front_video") or ctx.get("back_video") or ""
    if not ctx.get("audio"):
        ctx["audio"] = ctx.get("front_audio") or ctx.get("back_audio") or ""
    return render_generation_faces(deck_id, ctx)


def prepare_template_fields(fields: dict) -> dict:
    fields = dict(fields or {})
    front_text = fields.get("front") or ""
    back_text = fields.get("back") or ""
    fields.setdefault("front_html", html.escape(front_text).replace("\n", "<br>"))
    fields.setdefault("back_html", html.escape(back_text).replace("\n", "<br>"))

    front_image_path = fields.get("front_image_path") or fields.get("image")
    back_image_path = fields.get("back_image_path")
    video_path = fields.get("video_path")
    front_video_path = fields.get("front_video_path") or video_path
    back_video_path = fields.get("back_video_path")

    audio_path = fields.get("audio_path")
    audio_side = fields.get("audio_side") or "back"
    front_audio_path = fields.get("front_audio_path") or (audio_path if audio_side == "front" else None)
    back_audio_path = fields.get("back_audio_path") or (audio_path if audio_side == "back" else None)

    front_image_html = build_image_html(front_image_path)
    back_image_html = build_image_html(back_image_path)
    front_video_html = build_video_html(front_video_path, fields.get("poster_path"))
    back_video_html = build_video_html(back_video_path, fields.get("poster_path"))
    front_audio_html = build_audio_html(front_audio_path)
    back_audio_html = build_audio_html(back_audio_path)

    fields.setdefault("front_image", front_image_html)
    fields.setdefault("back_image", back_image_html)
    fields.setdefault("image", front_image_html or back_image_html)
    fields.setdefault("front_video", front_video_html)
    fields.setdefault("back_video", back_video_html)
    fields.setdefault("video", front_video_html or back_video_html)
    fields.setdefault("front_audio", front_audio_html)
    fields.setdefault("back_audio", back_audio_html)
    fields.setdefault("audio", front_audio_html or back_audio_html)
    fields.setdefault("front_video_html", front_video_html)
    fields.setdefault("back_video_html", back_video_html)
    return fields


def to_file_url(path: str) -> str:
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return path


def _to_file_url(path: str) -> str:
    return to_file_url(path)


def build_image_html(path: str | None) -> str:
    if not path:
        return ""
    url = _to_file_url(path)
    return f'<img src="{html.escape(url)}" alt="image">'


def build_video_html(path: str | None, poster_path: str | None = None) -> str:
    if not path:
        return ""
    clip_url = _to_file_url(path)
    poster_url = _to_file_url(poster_path) if poster_path else ""
    poster_attr = f' poster="{html.escape(poster_url)}"' if poster_url else ""
    return (
        f'<video controls autoplay preload="metadata"{poster_attr}>'
        f'<source src="{html.escape(clip_url)}" type="video/mp4"></video>'
    )


def build_audio_html(path: str | None) -> str:
    if not path:
        return ""
    url = _to_file_url(path)
    return f'<audio controls preload="metadata"><source src="{html.escape(url)}"></audio>'


def build_video_embed_html(clip_path: str | None, poster_path: str | None) -> str:
    return build_video_html(clip_path, poster_path)


def parse_timecode_to_seconds(value: str) -> float | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    parts = text.split(":")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        seconds = float(parts[-1].replace(",", "."))
    except ValueError:
        return None
    if seconds < 0:
        return None
    hours = 0
    minutes = 0
    if len(parts) == 2:
        try:
            minutes = int(parts[0])
        except ValueError:
            return None
        if seconds >= 60:
            return None
    elif len(parts) == 3:
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
        except ValueError:
            return None
        if seconds >= 60:
            return None
    if hours < 0 or minutes < 0 or minutes >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def ensure_premium_scrollbar_style(master: tk.Misc, palette: dict | None = None) -> None:
    palette = palette or {}
    style = ttk.Style(master)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(
        "Premium.Vertical.TScrollbar",
        background=palette.get("surface", "#111827"),
        troughcolor=palette.get("background", "#0B0D12"),
        bordercolor=palette.get("card_border", "#1F2937"),
        arrowcolor=palette.get("text_muted", "#9CA3AF"),
        lightcolor=palette.get("surface", "#111827"),
        darkcolor=palette.get("surface", "#111827"),
    )


def create_scrollable_content(
    parent: tk.Misc,
    bg: str | None = None,
) -> tuple[ttk.Frame, tk.Canvas, ttk.Frame]:
    outer_frame = ttk.Frame(parent)
    canvas = tk.Canvas(outer_frame, highlightthickness=0, bd=0)
    if bg is not None:
        canvas.configure(bg=bg)
    scrollbar = ttk.Scrollbar(
        outer_frame,
        orient="vertical",
        command=canvas.yview,
        style="Premium.Vertical.TScrollbar",
    )
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    content_frame = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=content_frame, anchor="nw")

    def _on_frame_configure(_event: tk.Event) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event: tk.Event) -> None:
        canvas.itemconfigure(window_id, width=event.width)

    content_frame.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_mousewheel(event: tk.Event) -> None:
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(event: tk.Event) -> None:
        direction = -1 if event.num == 4 else 1
        canvas.yview_scroll(direction, "units")

    def _bind_to_mousewheel(_event: tk.Event) -> None:
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        canvas.bind_all("<Button-5>", _on_mousewheel_linux)

    def _unbind_from_mousewheel(_event: tk.Event) -> None:
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _bind_to_mousewheel)
    canvas.bind("<Leave>", _unbind_from_mousewheel)
    content_frame.bind("<Enter>", _bind_to_mousewheel)
    content_frame.bind("<Leave>", _unbind_from_mousewheel)

    return outer_frame, canvas, content_frame


def safe_cut_video_clip(
    video_path: str,
    start_tc: str,
    end_tc: str,
    media_root: str,
) -> tuple[bool, str]:
    if not video_path or not os.path.exists(video_path):
        return False, "Видео файл не найден."

    start_seconds = parse_timecode_to_seconds(start_tc)
    end_seconds = parse_timecode_to_seconds(end_tc)
    if start_seconds is None or end_seconds is None:
        return False, "Неверный формат времени. Используйте SS, MM:SS или HH:MM:SS (дробные секунды допустимы)."
    if end_seconds <= start_seconds:
        return False, "Время окончания должно быть больше времени начала."

    if os.path.basename(media_root) == "clips" or os.path.basename(os.path.dirname(media_root)) == "clips":
        output_dir = media_root
    else:
        output_dir = os.path.join(media_root, "clips")
    os.makedirs(output_dir, exist_ok=True)

    safe_start = start_tc.strip().replace(":", "-").replace(".", "_").replace(",", "_")
    safe_end = end_tc.strip().replace(":", "-").replace(".", "_").replace(",", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"clip_{ts}_{safe_start}_{safe_end}.mp4")

    start_tc_norm = format_hms(int(start_seconds)) if float(start_seconds).is_integer() else start_tc
    end_tc_norm = format_hms(int(end_seconds)) if float(end_seconds).is_integer() else end_tc
    try:
        ok, result = cut_video_clip(video_path, start_tc_norm, end_tc_norm, output_dir)
    except Exception as exc:
        ok, result = False, str(exc)

    if ok:
        return True, result

    error_text = str(result)
    lowered = error_text.lower()
    if "ffmpeg" in lowered and ("не найден" in lowered or "not found" in lowered or "path" in lowered):
        error_text = "FFmpeg не найден в PATH."

    if MOVIEPY_AVAILABLE:
        clip = None
        subclip = None
        try:
            clip = mp.VideoFileClip(video_path)
            subclip = clip.subclip(start_seconds, end_seconds)
            subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
            return True, output_path
        except Exception as exc:
            return False, str(exc)
        finally:
            if subclip is not None:
                try:
                    subclip.close()
                except Exception:
                    pass
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass

    return False, error_text


def cut_video_clip_with_poster_h264(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str,
    base_name: str,
) -> tuple[str, str]:
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if end_sec <= start_sec:
        raise ValueError("end_sec must be > start_sec")
    os.makedirs(output_dir, exist_ok=True)
    clip_path = os.path.join(output_dir, f"{base_name}.mp4")
    poster_path = os.path.join(output_dir, f"{base_name}.jpg")
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        clip_cmd = [
            ffmpeg_path,
            "-y",
            "-ss",
            str(start_sec),
            "-to",
            str(end_sec),
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            clip_path,
        ]
        poster_cmd = [
            ffmpeg_path,
            "-y",
            "-ss",
            str(start_sec),
            "-i",
            video_path,
            "-frames:v",
            "1",
            poster_path,
        ]
        clip_result = subprocess.run(clip_cmd, capture_output=True, text=True, check=False)
        if clip_result.returncode != 0:
            error_text = clip_result.stderr.strip() or "Не удалось нарезать клип."
            raise RuntimeError(error_text)
        poster_result = subprocess.run(poster_cmd, capture_output=True, text=True, check=False)
        if poster_result.returncode != 0:
            error_text = poster_result.stderr.strip() or "Не удалось создать постер."
            raise RuntimeError(error_text)
        return clip_path, poster_path
    if MOVIEPY_AVAILABLE:
        clip = None
        subclip = None
        try:
            clip = mp.VideoFileClip(video_path)
            subclip = clip.subclip(start_sec, end_sec)
            subclip.write_videofile(clip_path, codec="libx264", audio_codec="aac", logger=None)
            try:
                subclip.save_frame(poster_path, t=0)
            except Exception:
                clip.save_frame(poster_path, t=start_sec)
            return clip_path, poster_path
        finally:
            if subclip is not None:
                try:
                    subclip.close()
                except Exception:
                    pass
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass
    raise RuntimeError("FFmpeg не найден, а moviepy не установлен.")


def build_rich_text_editor(
    parent: tk.Widget,
    palette: dict | None = None,
    *,
    height: int = 10,
    on_change=None,
) -> dict:
    palette = palette or {}
    toolbar_bg = palette.get("surface", "#111827")
    toolbar_border = palette.get("card_border", "#1F2937")
    toolbar_fg = palette.get("text", "#E5E7EB")
    toolbar_hover = palette.get("accent", "#4F46E5")

    container = tk.Frame(parent, bg=palette.get("background", "#0B0D12"))
    format_wrap = tk.Frame(container, bg="white")
    format_wrap.pack(fill=tk.X)
    format_inner = tk.Frame(format_wrap, bg=palette.get("background", "#0B0D12"))
    format_inner.pack(fill=tk.X, padx=1, pady=1)

    toolbar = tk.Frame(format_inner, bg=toolbar_bg, highlightthickness=1, highlightbackground=toolbar_border)
    toolbar.pack(fill=tk.X, padx=6, pady=4)

    text_widget = tk.Text(container, height=height, wrap=tk.WORD)
    style_card_surface_text(text_widget, palette)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
    create_context_menu(text_widget)

    def _ensure_hidden_token_tag() -> str:
        token_tag = "hidden_token"
        if token_tag not in text_widget.tag_names():
            try:
                text_widget.tag_configure(token_tag, elide=True)
            except tk.TclError:
                bg = text_widget.cget("bg")
                text_widget.tag_configure(token_tag, foreground=bg, background=bg)
        return token_tag

    def _insert_hidden_token(index: str, token: str) -> None:
        token_tag = _ensure_hidden_token_tag()
        text_widget.insert(index, token)
        start = text_widget.index(index)
        end = text_widget.index(f"{start} + {len(token)}c")
        text_widget.tag_add(token_tag, start, end)

    def _remove_underline_tokens(start: str, end: str) -> None:
        range_start = text_widget.index(f"{start} - 12c")
        range_end = text_widget.index(f"{end} + 12c")
        segment = text_widget.get(range_start, range_end)
        matches = list(_UNDERLINE_TOKEN_RE.finditer(segment))
        for match in reversed(matches):
            token_start = text_widget.index(f"{range_start} + {match.start()}c")
            token_end = text_widget.index(f"{range_start} + {match.end()}c")
            text_widget.delete(token_start, token_end)

    def _btn(parent_btn, label, command):
        btn = tk.Button(
            parent_btn,
            text=label,
            command=command,
            bg=toolbar_bg,
            fg=toolbar_fg,
            activebackground=toolbar_hover,
            activeforeground=toolbar_fg,
            relief=tk.FLAT,
            bd=0,
            padx=6,
            pady=2,
            font=("Segoe UI", 10, "bold"),
        )

        def _on_enter(_e):
            btn.configure(bg=toolbar_hover)

        def _on_leave(_e):
            btn.configure(bg=toolbar_bg)

        btn.bind("<Enter>", _on_enter)
        btn.bind("<Leave>", _on_leave)
        btn.pack(side=tk.LEFT, padx=2)
        return btn

    def _toggle_tag(tag: str, **cfg) -> None:
        if tag not in text_widget.tag_names():
            text_widget.tag_configure(tag, **cfg)
        try:
            has_tag = tag in text_widget.tag_names("sel.first")
            if has_tag:
                text_widget.tag_remove(tag, "sel.first", "sel.last")
            else:
                text_widget.tag_add(tag, "sel.first", "sel.last")
        except tk.TclError:
            pass

    def _clear_formatting() -> None:
        try:
            start = text_widget.index("sel.first")
            end = text_widget.index("sel.last")
        except tk.TclError:
            start = text_widget.index("insert linestart")
            end = text_widget.index("insert lineend")
        for tag in text_widget.tag_names():
            if tag != "sel":
                text_widget.tag_remove(tag, start, end)
        _remove_underline_tokens(start, end)

    def _choose_color(title: str):
        try:
            return colorchooser.askcolor(title=title, parent=container)
        except tk.TclError:
            return None

    def _apply_color() -> None:
        chosen = _choose_color("Цвет текста")
        if not chosen or not chosen[1]:
            return
        color = chosen[1]
        _toggle_tag(f"color_{color}", foreground=color)

    def _apply_marker() -> None:
        chosen = _choose_color("Цвет выделения")
        if not chosen or not chosen[1]:
            return
        color = chosen[1]
        _toggle_tag(f"marker_{color}", background=color)

    def _apply_list(prefix: str) -> None:
        try:
            start = text_widget.index("sel.first linestart")
            end = text_widget.index("sel.last linestart")
        except tk.TclError:
            start = text_widget.index("insert linestart")
            end = start
        line = start
        counter = 1
        while True:
            insert_prefix = prefix.format(counter=counter)
            text_widget.insert(line, insert_prefix)
            if line == end:
                break
            line = text_widget.index(f"{line} +1 line")
            counter += 1

    def _line_range() -> tuple[str, str]:
        try:
            start = text_widget.index("sel.first linestart")
            end = text_widget.index("sel.last linestart")
        except tk.TclError:
            start = text_widget.index("insert linestart")
            end = start
        return start, end

    def _indent_lines() -> None:
        start, end = _line_range()
        line = start
        while True:
            text_widget.insert(line, "\t")
            if line == end:
                break
            line = text_widget.index(f"{line} +1 line")

    def _outdent_lines() -> None:
        start, end = _line_range()
        line = start
        while True:
            if text_widget.get(line, f"{line} +1c") == "\t":
                text_widget.delete(line, f"{line} +1c")
            elif text_widget.get(line, f"{line} +4c") == "    ":
                text_widget.delete(line, f"{line} +4c")
            elif text_widget.get(line, f"{line} +2c") == "  ":
                text_widget.delete(line, f"{line} +2c")
            if line == end:
                break
            line = text_widget.index(f"{line} +1 line")

    def _apply_justify(align: str) -> None:
        tag = f"align_{align}"
        if tag not in text_widget.tag_names():
            text_widget.tag_configure(tag, justify=align)
        try:
            text_widget.tag_add(tag, "sel.first", "sel.last")
        except tk.TclError:
            text_widget.tag_add(tag, "1.0", tk.END)

    def _apply_script(tag: str, offset: int) -> None:
        base_font = tkfont.Font(font=text_widget.cget("font"))
        size = max(8, int(base_font.cget("size") or 12) - 3)
        script_font = base_font.copy()
        script_font.configure(size=size)
        _toggle_tag(tag, font=script_font, offset=offset)

    def _insert_link() -> None:
        url = simpledialog.askstring("Ссылка", "Введите URL:", parent=container)
        if not url:
            return
        text_widget.insert("insert", url)

    def _apply_underline_style(style: str) -> None:
        try:
            start = text_widget.index("sel.first")
            end = text_widget.index("sel.last")
        except tk.TclError:
            return
        if style == "double":
            tag_name = "underline_double"
            tag_cfg = {"underline": 1, "foreground": palette.get("accent", "#4F46E5")}
        else:
            tag_name = "underline_wavy"
            tag_cfg = {"underline": 1, "foreground": palette.get("muted", "#98A2B3")}
        if tag_name not in text_widget.tag_names():
            text_widget.tag_configure(tag_name, **tag_cfg)
        text_widget.tag_add(tag_name, start, end)
        _insert_hidden_token(end, "[[/u]]")
        _insert_hidden_token(start, f"[[u:{style}]]")

    def _insert_mathjax() -> None:
        try:
            start = text_widget.index("sel.first")
            end = text_widget.index("sel.last")
            content = text_widget.get(start, end)
            text_widget.delete(start, end)
            text_widget.insert(start, f"\\({content}\\)")
        except tk.TclError:
            insert_pos = text_widget.index("insert")
            text_widget.insert(insert_pos, "\\(  \\)")
            text_widget.mark_set("insert", f"{insert_pos} + 3c")

    def _attach_media() -> None:
        filename = filedialog.askopenfilename(
            title="Прикрепить файл",
            filetypes=[
                ("Медиа", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.mp3 *.wav *.ogg *.m4a *.mp4 *.mov *.avi *.mkv *.webm"),
                ("Все файлы", "*.*"),
            ],
            parent=container,
        )
        if not filename:
            return
        text_widget.insert("insert", filename)

    def _fmt_not_ready() -> None:
        pass

    _btn(toolbar, "B", lambda: _toggle_tag("bold", font=("Segoe UI", 10, "bold")))
    _btn(toolbar, "I", lambda: _toggle_tag("italic", font=("Segoe UI", 10, "italic")))
    _btn(toolbar, "U", lambda: _toggle_tag("underline", underline=1))
    _btn(toolbar, "U²", lambda: _apply_underline_style("double"))
    _btn(toolbar, "U~", lambda: _apply_underline_style("wavy"))
    _btn(toolbar, "x²", lambda: _apply_script("superscript", 6))
    _btn(toolbar, "x₂", lambda: _apply_script("subscript", -4))
    _btn(toolbar, "A", _apply_color)
    _btn(toolbar, "🖍", _apply_marker)
    _btn(toolbar, "🧹", _clear_formatting)
    _btn(toolbar, "•", lambda: _apply_list("• "))
    _btn(toolbar, "1.", lambda: _apply_list("{counter}. "))
    _btn(toolbar, "L", lambda: _apply_justify("left"))
    _btn(toolbar, "C", lambda: _apply_justify("center"))
    _btn(toolbar, "R", lambda: _apply_justify("right"))
    _btn(toolbar, "⇥", _indent_lines)
    _btn(toolbar, "⇤", _outdent_lines)
    _btn(toolbar, "🔗", _insert_link)
    _btn(toolbar, "📎", _attach_media)
    _btn(toolbar, "🎙", _fmt_not_ready)
    _btn(toolbar, "Fx", _insert_mathjax)
    _btn(toolbar, "</>", _fmt_not_ready)
    _btn(toolbar, "⨯", _fmt_not_ready)

    text_widget.bind("<Tab>", lambda _e: (_indent_lines() or "break"))
    text_widget.bind("<Shift-Tab>", lambda _e: (_outdent_lines() or "break"))
    if on_change is not None:
        text_widget.bind("<KeyRelease>", lambda _e: on_change())

    def _get_plain_text() -> str:
        return text_widget.get("1.0", tk.END).rstrip("\n")

    def _get_html() -> str:
        return text_widget_to_html(text_widget)

    def _set_html(html_value: str) -> None:
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", plain_text_from_html(html_value or ""))

    return {
        "frame": container,
        "text": text_widget,
        "toolbar": toolbar,
        "get_plain_text": _get_plain_text,
        "get_html": _get_html,
        "set_html": _set_html,
    }

def create_action_menubutton(parent, palette: dict | None = None) -> tuple[ttk.Menubutton, tk.Menu]:
    palette = palette or {}
    style = ttk.Style(parent)
    style.configure(
        "ActionMenu.TMenubutton",
        background=palette.get("surface", "#111827"),
        foreground=palette.get("text", "#E5E7EB"),
        padding=(8, 4),
    )
    menu_button = ttk.Menubutton(parent, text="⋯", style="ActionMenu.TMenubutton")
    menu = tk.Menu(
        menu_button,
        tearoff=0,
        bg=palette.get("surface", "#111827"),
        fg=palette.get("text", "#E5E7EB"),
        activebackground=palette.get("accent", "#4F46E5"),
        activeforeground=palette.get("text", "#E5E7EB"),
        borderwidth=0,
    )
    menu_button["menu"] = menu
    return menu_button, menu
# ==========================
# GUI
# ==========================

class ResizableImageLabel(tk.Label):
    """Label с изображением, которое можно масштабировать перетаскиванием за углы"""
    
    def __init__(self, parent, *, enable_mousewheel: bool = True, **kwargs):
        super().__init__(parent, **kwargs)
        self.original_image = None
        self.current_image = None
        self.image_path = None
        self.scale_factor = 1.0
        self.drag_data = {"x": 0, "y": 0, "item": None}
        self._container_size: tuple[int, int] = (0, 0)
        self._warned_large_path: str | None = None
        self._configure_job = None
        self._min_size_job = None
        self._tk_img_cache: dict = {}
        self._orig_pil_cache: dict = {}
        self._orig_path_cache: dict = {}
        self._orig_pil_by_key: dict = {}
        self._tkimg_by_key: dict = {}
        self._orig_path_by_key: dict = {}
        self._render_key = None
        self._render_zoom = None
        self._render_container = None
        self._fixed_slot_size: tuple[int, int] | None = None
        self.max_scale_factor = 3.0
        self.configure(anchor="center")
        
        # Привязываем события мыши
        self.bind("<ButtonPress-1>", self.start_drag)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.stop_drag)
        if enable_mousewheel:
            self.bind("<MouseWheel>", self.on_mousewheel)  # Для Windows
            self.bind("<Button-4>", self.on_mousewheel)    # Для Linux, scroll up
            self.bind("<Button-5>", self.on_mousewheel)    # Для Linux, scroll down
        self.bind("<Configure>", self._handle_configure)

    def _handle_configure(self, event):
        self._container_size = (max(1, int(event.width)), max(1, int(event.height)))
        if self._configure_job:
            try:
                self.after_cancel(self._configure_job)
            except Exception:
                pass
        if event.width < 80 or event.height < 80:
            self._configure_job = self.after(80, self._render_from_state)
            return
        self._configure_job = self.after(80, self._render_from_state)

    def _render_from_state(self):
        if not self.image_path:
            return
        container = self._render_container or self.master
        key = self._render_key or self.image_path
        zoom = self._render_zoom if self._render_zoom is not None else self.scale_factor
        render_image(self, container, self.image_path, zoom, key)

    def set_container_size(self, width: int, height: int):
        """Установить размеры контейнера для подгонки изображения под доступную область."""
        self._container_size = (max(1, int(width)), max(1, int(height)))
        self.update_display()

    def set_fixed_slot_size(self, width: int, height: int) -> None:
        self._fixed_slot_size = (max(1, int(width)), max(1, int(height)))
        
    def load_image(self, image_path, *, key=None, zoom=None, container_widget=None):
        """Загрузить изображение"""
        self.image_path = resolve_media_path(image_path) if image_path else None
        self._render_key = key
        if zoom is not None:
            self.scale_factor = max(0.1, min(self.max_scale_factor, float(zoom)))
        self._render_zoom = self.scale_factor
        self._render_container = container_widget or self.master
        if not self.image_path:
            self.original_image = None
            self.current_image = None
            self.image = None
            self.config(text="Нет изображения", image="")
            return False
        return bool(render_image(self, self._render_container, self.image_path, self.scale_factor, self._render_key or self.image_path))
    
    def update_display(self):
        """Обновить отображение изображения"""
        if self.image_path:
            container = self._render_container or self.master
            key = self._render_key or self.image_path
            zoom = self._render_zoom if self._render_zoom is not None else self.scale_factor
            render_image(self, container, self.image_path, zoom, key)
    
    def start_drag(self, event):
        """Начать перетаскивание для масштабирования"""
        # Проверяем, нажали ли на угол изображения (последние 20 пикселей)
        if self.current_image:
            width = self.current_image.width()
            height = self.current_image.height()
            
            # Проверяем правый нижний угол
            if (width - 20 <= event.x <= width and 
                height - 20 <= event.y <= height):
                self.drag_data["x"] = event.x
                self.drag_data["y"] = event.y
                self.drag_data["item"] = "resize"
                
    def drag(self, event):
        """Обработка перетаскивания"""
        if self.drag_data["item"] == "resize" and self.original_image:
            # Вычисляем изменение размера
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            # Масштабируем на основе большего изменения
            if abs(dx) > abs(dy):
                delta = dx / self.original_image.width
            else:
                delta = dy / self.original_image.height
            
            # Применяем масштаб
            self.scale_factor = max(0.1, min(3.0, self.scale_factor + delta * 0.5))
            self._render_zoom = self.scale_factor
            
            # Обновляем изображение
            self.update_display()
            
            # Обновляем позицию для следующего события
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
    
    def stop_drag(self, event):
        """Завершить перетаскивание"""
        self.drag_data["item"] = None
    
    def set_zoom_factor(self, zoom_factor: float) -> None:
        self.scale_factor = max(0.1, min(self.max_scale_factor, float(zoom_factor)))
        self._render_zoom = self.scale_factor
        self.update_display()

    def on_mousewheel(self, event):
        """Обработка колесика мыши для масштабирования"""
        if self.original_image:
            # Определяем направление масштабирования
            if event.num == 5 or event.delta < 0:  # Вниз или от пользователя
                self.scale_factor = max(0.1, self.scale_factor * 0.9)
            elif event.num == 4 or event.delta > 0:  # Вверх или к пользователю
                self.scale_factor = min(3.0, self.scale_factor * 1.1)
            self._render_zoom = self.scale_factor
            
            # Обновляем изображение
            self.update_display()
        
        return "break"


class AnkiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("X-FLASH")
        self.geometry("1366x768")
        self.minsize(1100, 700)

        self.root = self
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self._logo_small, self._logo_big = load_app_logo(self, BASE_DIR)
        self._ico_path = os.path.join(BASE_DIR, "assets", "app.ico")
        _logo_debug(f"loaded small={self._logo_small is not None} big={self._logo_big is not None}")

        try:
            _fix_tk_default_fonts(self)
        except Exception:
            pass

        self.style, self.palette = apply_premium_dark_theme(self)

        try:
            _fix_tk_default_fonts(self)
        except Exception:
            pass

        self.after(0, lambda: apply_window_icon(self, self._logo_big, self._ico_path))

        self.decks = []
        self.selected_deck_id = None
        self.selected_phase = None

        # Шаблоны фронта/ответа
        self.front_template = DEFAULT_FRONT_TEMPLATE
        self.back_template = DEFAULT_BACK_TEMPLATE

        # Иконки для колод
        self.deck_icons = {}
        self.deck_preview_images = {}

        self.current_chunk_html = ""

        global MIC_DEVICE_INDEX
        MIC_DEVICE_INDEX = detect_default_mic_index()
        self.microphone_index = MIC_DEVICE_INDEX

        self.overdue_canvas = None
        self.overdue_badge_text_id = None
        self.phase_badge_manager: PhaseOverdueBadges | None = None
        self.overdue_update_job = None
        self._overdue_task_running = False
        self._deck_select_job: str | None = None

        self._refresh_in_progress = False

        self.busy_dialog = BusyDialog(self)
        self.task_runner = TaskRunner(self, self.busy_dialog)
        self._loading_depth = 0
        self.global_loading_overlay = GlobalLoadingOverlay(self)

        self.image_import_watch_job = None

        self._bg_listeners: list[tuple[queue.Queue, callable]] = []

        # отображение id-элементов Treeview -> (deck_id, phase)
        self.deck_items = {}

        self.user_id = get_local_user_id()
        self.user_account = ensure_user_account(self.user_id)
        self.user_profile = ensure_user_profile_row(self.user_id)
        self.sync_config = load_sync_config()
        self.sync_token = self.sync_config.get("token")
        self.sync_user_email = self.sync_config.get("user_email")
        self.sync_client = SyncClient(
            self.sync_config.get("api_base_url") or SYNC_CONFIG_DEFAULT["api_base_url"],
            self.sync_config.get("timeout_sec") or SYNC_CONFIG_DEFAULT["timeout_sec"],
        )
        self.credits_service = CreditsService()
        self.referral_service = ReferralService(credits_service=self.credits_service)
        self.balance_var = tk.StringVar(value="—")
        self.premium_var = tk.BooleanVar(value=False)
        self.main_notebook: ttk.Notebook | None = None
        self.dashboard_tab: ttk.Frame | None = None
        self.personal_tab = None
        self.settings_tab: ttk.Frame | None = None
        self.stats_tab: ttk.Frame | None = None
        self.ledger_tree: ttk.Treeview | None = None
        self.ref_summary_vars: dict[str, tk.StringVar] = {}
        self.activation_progress_vars: dict[str, tk.DoubleVar] = {}
        self.activation_progress_labels: dict[str, tk.StringVar] = {}
        self.activation_status_var = tk.StringVar(value="")
        self.activation_overall_var = tk.DoubleVar(value=0)
        self.activation_overall_label_var = tk.StringVar(value="")
        self.payment_status_var = tk.StringVar(value="")
        self.payment_code_var = tk.StringVar(value="")
        self.package_choice_var = tk.StringVar(value="pack_500")
        self.balance_labels: list[tk.Variable] = []
        self.balance_widgets: list[tk.Label] = []
        self.balance_observers: list[callable] = []
        self.credit_icon_image = None
        self.credit_icon_small = None
        self.credit_icon_large = None
        self.generation_menu = None
        self.generation_menu_indexes: dict[str, int] = {}
        self.generation_drawer = None
        self.generation_drawer_buttons: dict[str, tk.Widget] = {}
        self.generation_drawer_coin_icon = None
        self.pending_invoice = None
        self.hamburger_button = None
        self.mode_actions: dict[str, callable] = {}
        self.menubar: tk.Menu | None = None
        self.ui_adapter: UIAdapter | None = None
        self.addon_manager: AddonManager | None = None
        self.mw_context: MWContext | None = None
        self.user_id_var = tk.StringVar(value=self.user_id)
        self.account_status_var = tk.StringVar(value="")
        self.premium_timer_var = tk.StringVar(value="00:00:00")
        self.premium_timer_job: str | None = None
        self._ensure_initial_credits()
        self.premium_var.set(self.is_premium_active())

        # Инициализируем словарь
        init_dictionary()

        self.mode_actions = {
            "repeat": self.start_repeat_mode,
            "playback": self.start_playback_mode,
        }

        self.create_menu()
        self._setup_addon_system()
        self.create_widgets()
        self._load_addons()
        self.refresh_decks()
        self.refresh_balance_display()
        self.start_premium_timer()

        self.after(500, self.warn_if_no_tesseract)
        self.after(50, self.poll_bg_queues)
        self.after(0, self._run_app_startup_hooks)

    # --------- предупреждение ---------

    def warn_if_no_tesseract(self):
        if not is_tesseract_available():
            messagebox.showwarning(
                "Tesseract не найден",
                "Режим генерации из изображений (OCR) пока недоступен.\n"
                "Установи Tesseract OCR или пропиши путь."
            )

    def register_bg_handler(self, queue_obj: queue.Queue, handler):
        self._bg_listeners.append((queue_obj, handler))

    def unregister_bg_handler(self, queue_obj: queue.Queue):
        self._bg_listeners = [item for item in self._bg_listeners if item[0] is not queue_obj]

    def poll_bg_queues(self):
        for q_obj, handler in list(self._bg_listeners):
            while True:
                try:
                    event = q_obj.get_nowait()
                except queue.Empty:
                    break
                try:
                    handler(event)
                except Exception:
                    pass
        self.after(50, self.poll_bg_queues)

    def show_loading(self, title: str = "Загрузка", determinate: bool = False, total: int | None = None):
        """Показать глобальный оверлей загрузки."""
        self._loading_depth += 1
        mode = "determinate" if determinate else "indeterminate"
        try:
            self.busy_dialog.show(title, mode, total)
        except Exception:
            pass

    def update_loading(self, done: int, total: int | None = None, text: str | None = None):
        """Обновить состояние глобального прогресса."""
        try:
            self.busy_dialog.update_progress(done, total, text)
        except Exception:
            pass

    def hide_loading(self):
        """Спрятать оверлей загрузки."""
        self._loading_depth = max(0, self._loading_depth - 1)
        if self._loading_depth == 0:
            try:
                self.busy_dialog.close()
            except Exception:
                pass

    def run_with_loading(
        self,
        fn,
        *args,
        determinate: bool = False,
        maximum: int = 100,
        on_success=None,
        on_error=None,
        **kwargs,
    ) -> None:
        self.global_loading_overlay.show(self, determinate=determinate, maximum=maximum)

        def _worker():
            result = None
            error = None
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                error = exc

            def _finish():
                try:
                    if error is not None:
                        if on_error:
                            on_error(error)
                        else:
                            messagebox.showerror("Ошибка", str(error))
                    else:
                        if on_success:
                            on_success(result)
                finally:
                    self.global_loading_overlay.hide()

            self.after(0, _finish)

        threading.Thread(target=_worker, daemon=True).start()

    def run_task(self, title: str, mode: str, task_fn, on_success, on_error, total=None):
        self.task_runner.run_task(title, mode, task_fn, on_success, on_error, total)

    # --------- меню ---------

    def _setup_addon_system(self) -> None:
        self.ui_adapter = UIAdapter(self, self.menubar)
        self.addon_manager = AddonManager(BASE_DIR, ui=self.ui_adapter)
        safe_mode_flag = any(
            flag in sys.argv for flag in ("--safe-mode", "--safe_mode", "--no-addons")
        )
        if safe_mode_flag:
            self.addon_manager.safe_mode = True
        self.ui_adapter.set_addon_manager(self.addon_manager)
        self.mw_context = MWContext(self, self.ui_adapter, self.addon_manager)
        self.mw_context.col = CollectionAdapter(
            db_path=get_db_path(),
            list_decks=list_decks,
            get_cards_in_deck=get_cards_in_deck,
        )
        self.mw_context.state = {
            "current_deck_id": None,
            "current_card_id": None,
        }
        set_mw(self.mw_context)
        self.ui_adapter.add_menu_item("Настройки", "Аддоны...", self.addon_manager.open_manager_window)

    def _load_addons(self) -> None:
        if self.addon_manager and not self.addon_manager.safe_mode:
            self.addon_manager.load_addons()

    def _run_app_startup_hooks(self) -> None:
        try:
            gui_hooks.app_did_startup()
        except Exception:
            pass

    def create_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Импорт Anki (.apkg)", command=self.open_apkg_import_window)
        file_menu.add_separator()
        file_menu.add_command(label="Новая колода", command=self.add_deck_window)
        file_menu.add_command(label="Редактировать колоду", command=self.edit_deck_window)
        file_menu.add_command(label="Удалить выбранную колоду", command=self.delete_selected_deck)
        menubar.add_cascade(label="Файл", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="OpenAI API ключ", command=self.open_settings_window)
        settings_menu.add_command(label="Аудиоустройство (цифровой слух)",
                                  command=self.open_audio_device_window)
        settings_menu.add_command(label="Настройки перевода",
                                  command=self.open_translation_settings_window)
        settings_menu.add_command(label="Управление словарями",
                                  command=self.open_dictionary_manager_window)
        settings_menu.add_command(label="Генерация шаблонов",
                                  command=self.open_generation_templates_window)
        menubar.add_cascade(label="Настройки", menu=settings_menu)

        gen_menu = tk.Menu(menubar, tearoff=0)
        self.generation_menu = gen_menu
        gen_menu.add_command(label="Генерация из текста...", command=self.open_generate_from_text_window)
        self.generation_menu_indexes["text"] = gen_menu.index("end")
        gen_menu.add_command(
            label="Генерация по конспекту AI + картинки...",
            command=self.open_generate_from_notes_window,
        )
        gen_menu.add_command(
            label="Режим генерация из текста AI 👑",
            command=self.open_generate_from_text_ai_window,
        )
        self.generation_menu_indexes["text_ai"] = gen_menu.index("end")
        gen_menu.add_command(
            label="Генерация из изображения (OCR) 👑 (1⚡/стр)",
            command=self.open_generate_from_image_window,
        )
        self.generation_menu_indexes["ocr"] = gen_menu.index("end")
        gen_menu.add_command(label="Генерация через цифровой слух...",
                             command=self.open_generate_from_speech_window)
        gen_menu.add_command(label="Генерация из видео (цифровой слух)...",
                             command=self.open_generate_from_video_window)
        gen_menu.add_command(label="Видео → клипы → карточки",
                             command=self.open_video_clip_window)
        gen_menu.add_command(
            label="Импорт картинок по ID (CSV) (15⚡/10 или 5⚡/15)",
            command=self.open_image_id_import_window,
        )
        self.generation_menu_indexes["image_id_import"] = gen_menu.index("end")
        gen_menu.add_command(label="Импорт CSV колоды", command=self.open_csv_import_window)
        gen_menu.add_command(
            label="Импорт CSV колоды (wikimedia) 👑 (5⚡ за 10 импортов)",
            command=self.open_wikimedia_csv_window,
        )
        self.generation_menu_indexes["wikimedia"] = gen_menu.index("end")
        self.refresh_generation_menu_state()

        modes_menu = tk.Menu(menubar, tearoff=0)
        modes_menu.add_command(
            label="Режим повторения (по дате)",
            command=self.mode_actions.get("repeat", self.start_repeat_mode)
        )
        modes_menu.add_command(
            label="Режим воспроизведения (по прогрессу)",
            command=self.mode_actions.get("playback", self.start_playback_mode)
        )
        modes_menu.add_command(label="Режим обзора / редактирования",
                               command=self.show_cards_window)
        modes_menu.add_command(label="Режим ознакомления",
                               command=self.start_overview_mode)
        menubar.add_cascade(label="Режимы", menu=modes_menu)

        # Добавляем меню статистики
        stats_menu = tk.Menu(menubar, tearoff=0)
        stats_menu.add_command(label="Показать статистику", command=self.show_statistics_window)
        stats_menu.add_command(label="Статистика словаря", command=self.show_dictionary_stats_window)
        menubar.add_cascade(label="Статистика", menu=stats_menu)

        self.config(menu=menubar)
        self.menubar = menubar

    def _is_widget_in_generation_drawer(self, widget: tk.Widget | None) -> bool:
        if not self.generation_drawer or widget is None:
            return False
        current = widget
        while current is not None:
            if current == self.generation_drawer:
                return True
            current = current.master
        return False

    def _maybe_close_generation_drawer(self, event) -> None:
        if not self.generation_drawer or not self.generation_drawer.winfo_exists():
            return
        if self.hamburger_button is not None and event.widget == self.hamburger_button:
            return
        if self._is_widget_in_generation_drawer(event.widget):
            return
        self._close_generation_drawer()

    def _run_generation_drawer_action(
        self,
        action,
        feature_key: str | None = None,
        require_premium: bool = False,
    ) -> None:
        if require_premium:
            if not self.guard_premium_and_spend(feature_key or "premium_access", 0, require_premium=True):
                self._close_generation_drawer()
                return
        self._close_generation_drawer()
        action()

    def _create_drawer_button(self, parent: tk.Widget, label: str, command) -> tk.Button:
        palette = getattr(self, "palette", {}) or {}
        bg = palette.get("panel", "#111827")
        hover = palette.get("panel2", "#1F2937")
        fg = palette.get("text", "#E5E7EB")
        btn = tk.Button(
            parent,
            text=label,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 11),
            cursor="hand2",
        )

        def _enter(_e):
            btn.configure(bg=hover)

        def _leave(_e):
            btn.configure(bg=bg)

        btn.bind("<Enter>", _enter)
        btn.bind("<Leave>", _leave)
        return btn

    def _build_generation_drawer(self) -> None:
        palette = getattr(self, "palette", {}) or {}
        bg = palette.get("panel", "#111827")
        border = palette.get("border", "#1F2937")
        fg = palette.get("text", "#E5E7EB")
        muted = palette.get("muted", "#9CA3AF")
        hover = palette.get("panel2", "#1F2937")

        drawer = tk.Toplevel(self)
        drawer.overrideredirect(True)
        drawer.transient(self)
        drawer.configure(bg=bg)
        try:
            drawer.attributes("-topmost", True)
        except Exception:
            pass

        container = tk.Frame(drawer, bg=bg, highlightthickness=1, highlightbackground=border)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg=bg, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        inner = tk.Frame(canvas, bg=bg)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _sync_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", _sync_width)
        inner.bind("<Configure>", _sync_scrollregion)

        self.generation_drawer = drawer
        self.generation_drawer_buttons = {}

        if self.generation_drawer_coin_icon is None:
            self.generation_drawer_coin_icon = self._load_credit_icon(size=14)

        def _on_mousewheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            return "break"

        for target in (canvas, inner):
            target.bind("<MouseWheel>", _on_mousewheel)
            target.bind("<Button-4>", _on_mousewheel)
            target.bind("<Button-5>", _on_mousewheel)

        def _create_drawer_row(
            title: str,
            command,
            *,
            price: str | None = None,
            premium: bool = False,
            feature_key: str | None = None,
        ) -> None:
            display_title = title
            if premium and "👑" not in display_title:
                display_title = f"{display_title} 👑"

            row = tk.Frame(inner, bg=bg, padx=12, pady=6)
            row.pack(fill=tk.X, padx=6, pady=2)

            left = tk.Label(
                row,
                text=display_title,
                bg=bg,
                fg=fg,
                anchor="w",
                justify=tk.LEFT,
                font=("Segoe UI", 11),
                wraplength=420,
            )
            left.pack(side=tk.LEFT, fill=tk.X, expand=True)

            right = None
            widgets = [row, left]
            if price:
                right = tk.Frame(row, bg=bg)
                right.pack(side=tk.RIGHT, padx=(8, 0))
                if self.generation_drawer_coin_icon:
                    icon = tk.Label(right, image=self.generation_drawer_coin_icon, bg=bg)
                    icon.pack(side=tk.LEFT, padx=(0, 6))
                    widgets.append(icon)
                cost = tk.Label(right, text=price, bg=bg, fg=muted, font=("Segoe UI", 10))
                cost.pack(side=tk.LEFT)
                widgets.extend([right, cost])

            def _run_action(_event=None):
                self._run_generation_drawer_action(
                    command,
                    feature_key=feature_key,
                    require_premium=premium,
                )

            def _set_bg(color: str):
                row.configure(bg=color)
                left.configure(bg=color)
                if right is not None:
                    right.configure(bg=color)
                for widget in widgets:
                    if isinstance(widget, tk.Label):
                        widget.configure(bg=color)

            def _enter(_event):
                _set_bg(hover)

            def _leave(_event):
                _set_bg(bg)

            for widget in widgets:
                widget.bind("<Enter>", _enter)
                widget.bind("<Leave>", _leave)
                widget.bind("<Button-1>", _run_action)
            row.configure(cursor="hand2")

        entries = [
            ("Генерация из текста...", self.open_generate_from_text_window, None, False, None),
            ("Генерация по конспекту AI + картинки...", self.open_generate_from_notes_window, None, False, None),
            ("Режим генерация из текста AI", self.open_generate_from_text_ai_window, None, True, "text_ai"),
            ("Генерация из изображения (OCR)", self.open_generate_from_image_window, "1/стр", True, "ocr_image"),
            ("Генерация через цифровой слух...", self.open_generate_from_speech_window, None, False, None),
            ("Генерация из видео (цифровой слух)...", self.open_generate_from_video_window, None, False, None),
            ("Видео → клипы → карточки", self.open_video_clip_window, None, False, None),
            ("Импорт картинок по ID (CSV)", self.open_image_id_import_window, "15 за 10 / 5 за 15", False, "image_id_import"),
            ("Импорт CSV колоды", self.open_csv_import_window, None, False, None),
            ("Картинки по CSV (Wikimedia)", self.open_wikimedia_csv_window, "5 за 10 импортов", True, "wikimedia"),
        ]

        for title, action, price, premium, feature_key in entries:
            _create_drawer_row(
                title,
                action,
                price=price,
                premium=premium,
                feature_key=feature_key,
            )

        # PATCH: fixed hamburger menu full list + scroll + iconcoin cost

    def _position_generation_drawer(self) -> None:
        if not self.generation_drawer or not self.generation_drawer.winfo_exists():
            return
        self.generation_drawer.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = 520
        if self.hamburger_button is not None:
            x = self.hamburger_button.winfo_rootx()
            button_y = self.hamburger_button.winfo_rooty()
            y = button_y + self.hamburger_button.winfo_height() + 6
        else:
            x = self.winfo_rootx() + 10
            y = self.winfo_rooty() + 10
            button_y = y
        max_height = min(int(screen_h * 0.75), screen_h - 40)
        req_height = self.generation_drawer.winfo_reqheight()
        height = max(200, min(req_height, max_height))
        if x + width > screen_w - 10:
            x = max(10, screen_w - width - 10)
        if x < 10:
            x = 10
        if y + height > screen_h - 10:
            alt_y = button_y - height - 6
            if alt_y >= 10:
                y = alt_y
            else:
                y = max(10, screen_h - height - 10)
        self.generation_drawer.geometry(f"{width}x{height}+{x}+{y}")

    def _open_generation_drawer(self) -> None:
        if not self.generation_drawer or not self.generation_drawer.winfo_exists():
            self._build_generation_drawer()
        self._position_generation_drawer()
        try:
            self.generation_drawer.deiconify()
            self.generation_drawer.lift()
            self.generation_drawer.focus_force()
        except Exception:
            pass

    def _close_generation_drawer(self) -> None:
        if self.generation_drawer and self.generation_drawer.winfo_exists():
            self.generation_drawer.destroy()
        self.generation_drawer = None

    def toggle_generation_drawer(self) -> None:
        if self.generation_drawer and self.generation_drawer.winfo_exists():
            self._close_generation_drawer()
            return
        self._open_generation_drawer()
    
    def open_generate_from_video_window(self):
        """Открыть окно генерации из видео"""
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
            
        # Выбрать видео файл
        filetypes = [
            ("Видео файлы", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("Все файлы", "*.*"),
        ]
        
        video_path = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=filetypes
        )

        if not video_path:
            return

        # Открыть окно видео-редактора
        self.video_editor_window = VideoEditorWindow(self.root, video_path, self.selected_deck_id)

    def open_video_clip_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        VideoClipCardsWindow(self, self.selected_deck_id)
        return
        win = tk.Toplevel(self)
        win.title("Видео → клипы → карточки")
        win.geometry("1180x860")
        win.minsize(980, 760)
        win.grab_set()
        apply_dark_theme_to_window(win, self.palette)
        ensure_premium_scrollbar_style(win, self.palette)

        deck_id = self.selected_deck_id
        colors = getattr(self, "palette", None) or {}
        scroll_bg = colors.get("background") if colors else None
        outer_frame, canvas, content_frame = create_scrollable_content(win, bg=scroll_bg)
        outer_frame.pack(fill=tk.BOTH, expand=True)

        ensure_video_text_generation_template()

        state = {
            "video_path": None,
            "clip_path": None,
            "poster_path": None,
            "front_video": None,
            "back_video": None,
            "drafts": [],
        }

        video_path_var = tk.StringVar()
        start_var = tk.StringVar(value="00:00:00")
        end_var = tk.StringVar(value="00:00:10")
        start_sec_var = tk.DoubleVar(value=0)
        end_sec_var = tk.DoubleVar(value=10)
        duration_var = tk.StringVar(value="Длительность: —")
        status_var = tk.StringVar(value="Клип сохраняется в папку media/clips/")
        clip_info_var = tk.StringVar(value="Клип не создан")
        stt_status_var = tk.StringVar(value="")
        stt_auto_punct_var = tk.BooleanVar(value=True)
        stt_auto_run_var = tk.BooleanVar(value=True)
        stt_state = {"running": False, "job_id": 0, "pending": False, "cancel_event": None}

        top_frame = ttk.Frame(content_frame)
        top_frame.pack(fill=tk.X, padx=12, pady=(10, 6))
        ttk.Label(top_frame, text="Видео файл:").pack(anchor="w")
        video_row = ttk.Frame(top_frame)
        video_row.pack(fill=tk.X, pady=4)
        ttk.Entry(video_row, textvariable=video_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def browse_video():
            path = filedialog.askopenfilename(
                title="Выберите видео файл",
                filetypes=[("Видео", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv"), ("Все файлы", "*.*")],
            )
            if path:
                video_path_var.set(path)
                load_video(path)

        ttk.Button(video_row, text="…", width=4, command=browse_video).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(top_frame, textvariable=duration_var).pack(anchor="w")

        player_frame = ttk.LabelFrame(content_frame, text="Плеер")
        player_frame.pack(fill=tk.X, padx=12, pady=(0, 10))
        player_holder = {"player": None}

        def _render_player(path: str) -> None:
            for widget in player_frame.winfo_children():
                widget.destroy()
            if is_vlc_available():
                try:
                    player = VlcPlayerWidget(player_frame, path, width=520, height=280)
                    if player.ensure_embedded():
                        player.pack(anchor="w", padx=6, pady=6)
                        player_holder["player"] = player
                        return
                except Exception:
                    player_holder["player"] = None
            ttk.Label(player_frame, text="Встроенный плеер недоступен").pack(anchor="w", padx=6, pady=6)
            ttk.Button(
                player_frame,
                text="Открыть во внешнем плеере",
                command=lambda: open_in_external_player(path),
            ).pack(anchor="w", padx=6, pady=(0, 6))

        time_frame = ttk.LabelFrame(content_frame, text="Нарезка")
        time_frame.pack(fill=tk.X, padx=12, pady=(0, 10))
        time_frame.columnconfigure(1, weight=1)

        ttk.Label(time_frame, text="Начало:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        start_entry = ttk.Entry(time_frame, textvariable=start_var, width=12)
        start_entry.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(time_frame, text="Конец:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        end_entry = ttk.Entry(time_frame, textvariable=end_var, width=12)
        end_entry.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        start_scale = ttk.Scale(time_frame, from_=0, to=10, variable=start_sec_var)
        end_scale = ttk.Scale(time_frame, from_=0, to=10, variable=end_sec_var)
        start_scale.grid(row=0, column=2, sticky="ew", padx=6)
        end_scale.grid(row=1, column=2, sticky="ew", padx=6)

        def _sync_var_from_scale(var: tk.StringVar, value: float) -> None:
            var.set(format_hms(int(value)))

        def _apply_scale_bounds() -> None:
            if end_sec_var.get() <= start_sec_var.get():
                end_sec_var.set(start_sec_var.get() + 1)
                _sync_var_from_scale(end_var, end_sec_var.get())

        def _on_start_scale(value):
            _sync_var_from_scale(start_var, float(value))
            _apply_scale_bounds()

        def _on_end_scale(value):
            _sync_var_from_scale(end_var, float(value))
            _apply_scale_bounds()

        start_scale.configure(command=_on_start_scale)
        end_scale.configure(command=_on_end_scale)
        start_scale.bind("<ButtonRelease-1>", lambda _e: _maybe_auto_stt("range"))
        end_scale.bind("<ButtonRelease-1>", lambda _e: _maybe_auto_stt("range"))

        def _sync_entry_to_scale(var: tk.StringVar, scale_var: tk.DoubleVar) -> None:
            parsed = parse_timecode_to_seconds(var.get().strip())
            if parsed is None:
                messagebox.showwarning(
                    "Время",
                    "Введите время в формате SS, MM:SS или HH:MM:SS (дробные секунды допустимы).",
                    parent=win,
                )
                return
            scale_var.set(parsed)
            _apply_scale_bounds()
            _maybe_auto_stt("range")

        start_entry.bind("<Return>", lambda _e: _sync_entry_to_scale(start_var, start_sec_var))
        end_entry.bind("<Return>", lambda _e: _sync_entry_to_scale(end_var, end_sec_var))
        start_entry.bind("<FocusOut>", lambda _e: _sync_entry_to_scale(start_var, start_sec_var))
        end_entry.bind("<FocusOut>", lambda _e: _sync_entry_to_scale(end_var, end_sec_var))

        ttk.Label(time_frame, textvariable=status_var, foreground="gray").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6)
        )
        ttk.Label(time_frame, textvariable=clip_info_var, foreground="gray").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6)
        )

        def load_video(path: str) -> None:
            if not path or not os.path.exists(path):
                return
            state["video_path"] = path
            duration = get_video_duration_seconds(path)
            if duration:
                duration_var.set(f"Длительность: {format_hms(int(duration))}")
                start_sec_var.set(0)
                end_sec_var.set(max(1, int(duration)))
                start_scale.configure(to=duration)
                end_scale.configure(to=duration)
                start_var.set("00:00:00")
                end_var.set(format_hms(int(duration)))
            _render_player(path)
            _maybe_auto_stt("video")

        def cut_and_attach():
            video_path = state.get("video_path") or video_path_var.get().strip()
            if not video_path:
                messagebox.showerror("Ошибка", "Выберите видео файл.", parent=win)
                return
            cut_btn.configure(state=tk.DISABLED)
            status_var.set("⏳ Нарезаю клип...")

            def _worker():
                ok, result = safe_cut_video_clip(
                    video_path,
                    start_var.get(),
                    end_var.get(),
                    os.path.join(MEDIA_FOLDER, "clips", str(deck_id)),
                )
                if not ok:
                    error_text = str(result)

                    def _on_error():
                        status_var.set("Ошибка нарезки")
                        cut_btn.configure(state=tk.NORMAL)
                        messagebox.showerror("Ошибка нарезки", error_text, parent=win)

                    win.after(0, _on_error)
                    return

                clip_path = result
                clip_name = os.path.basename(clip_path)
                clip_info = f"{start_var.get()} → {end_var.get()}"

                def _on_success():
                    state["clip_path"] = clip_path
                    state["poster_path"] = None
                    status_var.set(f"Создан клип {clip_name}")
                    clip_info_var.set(f"Клип: {clip_name} ({clip_info})")
                    cut_btn.configure(state=tk.NORMAL)
                    _maybe_auto_stt("clip")
                    messagebox.showinfo(
                        "Клип создан",
                        f"Клип сохранен: {clip_path}",
                        parent=win,
                    )

                win.after(0, _on_success)

            threading.Thread(target=_worker, daemon=True).start()

        cut_btn = ttk.Button(time_frame, text="Нарезать", command=cut_and_attach)
        cut_btn.grid(row=0, column=3, rowspan=2, padx=6, pady=6, sticky="ns")

        stt_frame = ttk.LabelFrame(content_frame, text="Цифровой слух")
        stt_frame.pack(fill=tk.BOTH, padx=12, pady=(0, 10))

        stt_top_row = ttk.Frame(stt_frame)
        stt_top_row.pack(fill=tk.X, padx=6, pady=(6, 2))

        stt_btn = ttk.Button(
            stt_top_row,
            text="Запустить цифровой слух",
        )
        stt_btn.pack(side=tk.LEFT)
        stt_cancel_btn = ttk.Button(
            stt_top_row,
            text="Отмена",
            state=tk.DISABLED,
        )
        stt_cancel_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(stt_frame, textvariable=stt_status_var, foreground="gray").pack(
            anchor="w", padx=6, pady=(0, 4)
        )
        stt_progress = ttk.Progressbar(stt_frame, mode="determinate", maximum=100)
        stt_progress.pack(fill=tk.X, padx=6, pady=(0, 4))

        stt_option_row = ttk.Frame(stt_frame)
        stt_option_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Checkbutton(
            stt_option_row,
            text="Автопунктуация",
            variable=stt_auto_punct_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        stt_auto_check = ttk.Checkbutton(
            stt_option_row,
            text="Авто-цифровой слух",
            variable=stt_auto_run_var,
        )
        stt_auto_check.pack(side=tk.LEFT)

        stt_text = scrolledtext.ScrolledText(stt_frame, height=6)
        style_text_widget(stt_text, colors)
        stt_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        create_context_menu(stt_text)

        stt_action_row = ttk.Frame(stt_frame)
        stt_action_row.pack(fill=tk.X, padx=6, pady=(0, 6))

        def _get_stt_text() -> str:
            return stt_text.get("1.0", tk.END).rstrip("\n")

        ttk.Button(
            stt_action_row,
            text="Копировать",
            command=lambda: win.clipboard_clear() or win.clipboard_append(_get_stt_text()),
        ).pack(side=tk.LEFT)

        def _insert_into_editor(editor: dict) -> None:
            text = _get_stt_text()
            if not text:
                return
            editor["text"].insert("insert", text)
            editor["text"].focus_set()

        ttk.Button(
            stt_action_row,
            text="Вставить в Front",
            command=lambda: _insert_into_editor(front_editor),
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            stt_action_row,
            text="Вставить в Back",
            command=lambda: _insert_into_editor(back_editor),
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            stt_frame,
            text="Авто-цифровой слух активируется после выбора видео или диапазона.",
            foreground="gray",
        ).pack(anchor="w", padx=6, pady=(0, 4))

        action_buttons = [cut_btn, stt_btn, stt_auto_check]

        def _set_progress_mode(mode: str) -> None:
            if mode == "indeterminate":
                stt_progress.configure(mode="indeterminate", maximum=100)
                stt_progress.start(10)
            else:
                stt_progress.stop()
                stt_progress.configure(mode="determinate", maximum=100)

        def _set_action_state(enabled: bool) -> None:
            for btn in action_buttons:
                try:
                    btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)
                except Exception:
                    continue
            try:
                stt_cancel_btn.configure(state=tk.DISABLED if enabled else tk.NORMAL)
            except Exception:
                pass

        def _cancel_stt() -> None:
            cancel_event = stt_state.get("cancel_event")
            if stt_state["running"] and cancel_event is not None:
                cancel_event.set()
                stt_status_var.set("⏳ Отменяю…")
                _append_stt_log_line("stt_cancel_requested=true")

        def _start_stt(reason: str) -> None:
            video_path = state.get("clip_path") or state.get("video_path") or video_path_var.get().strip()
            if not video_path:
                messagebox.showerror("Видео", "Выберите видео файл.", parent=win)
                return
            stt_state["running"] = True
            stt_state["pending"] = False
            stt_state["job_id"] += 1
            job_id = stt_state["job_id"]
            _set_action_state(False)
            stt_status_var.set("⏳ Извлекаю аудио…")
            stt_progress.configure(value=0)
            _set_progress_mode("indeterminate")
            progress_queue = queue.Queue()
            stt_state["progress_queue"] = progress_queue
            stt_state["cancel_event"] = threading.Event()

            def _worker():
                try:
                    if state.get("clip_path"):
                        target_path = state["clip_path"]
                        start_sec = None
                        end_sec = None
                    else:
                        start_sec = parse_timecode_to_seconds(start_var.get().strip())
                        end_sec = parse_timecode_to_seconds(end_var.get().strip())
                        if start_sec is None or end_sec is None:
                            raise RuntimeError(
                                "Введите корректный диапазон Start/End (SS, MM:SS или HH:MM:SS)."
                            )
                        if end_sec <= start_sec:
                            raise RuntimeError("Время окончания должно быть больше времени начала.")
                        target_path = video_path
                    text, _segments, _engine, _engine_detail = run_stt_pipeline(
                        target_path,
                        start=start_sec,
                        end=end_sec,
                        lang="de-DE",
                        punctuate=stt_auto_punct_var.get(),
                        progress_queue=progress_queue,
                        cancel_event=stt_state.get("cancel_event"),
                    )
                    return ("done", text)
                except STTCancelled:
                    return ("cancelled", None)
                except Exception as exc:
                    return ("error", str(exc))

            def _poll_progress(current_job_id: int) -> None:
                if current_job_id != stt_state["job_id"]:
                    return
                local_queue = stt_state.get("progress_queue")
                if local_queue is None:
                    return
                while True:
                    try:
                        event = local_queue.get_nowait()
                    except queue.Empty:
                        break
                    status = event.get("status")
                    mode = event.get("mode")
                    overall = event.get("overall")
                    if mode:
                        _set_progress_mode(mode)
                    if status:
                        stt_status_var.set(f"⏳ {status}")
                    if overall is not None:
                        stt_progress.configure(value=max(0.0, min(100.0, overall)))
                if stt_state["running"] and current_job_id == stt_state["job_id"]:
                    win.after(100, lambda: _poll_progress(current_job_id))

            def _on_finish(result):
                stt_state["running"] = False
                _set_action_state(True)
                if job_id != stt_state["job_id"]:
                    if stt_state.get("pending"):
                        stt_state["pending"] = False
                        _start_stt("pending")
                    return
                kind, payload = result
                if kind == "error":
                    stt_status_var.set(f"❌ Ошибка: {payload}")
                    _set_progress_mode("determinate")
                    stt_progress.configure(value=0)
                    show_stt_error_dialog(win, "Ошибка распознавания", payload)
                    return
                if kind == "cancelled":
                    stt_status_var.set("⛔ Отменено")
                    _set_progress_mode("determinate")
                    stt_progress.configure(value=0)
                    return
                text = payload or ""
                stt_text.delete("1.0", tk.END)
                stt_text.insert("1.0", text)
                _set_progress_mode("determinate")
                stt_progress.configure(value=100)
                stt_status_var.set("✅ Готово")
                _auto_generate_drafts(text)
                _update_save_pricing()

            def _thread_runner():
                result = _worker()
                win.after(0, lambda: _on_finish(result))

            _poll_progress(job_id)
            threading.Thread(target=_thread_runner, daemon=True).start()

        def _request_stt(reason: str) -> None:
            if stt_state["running"]:
                stt_state["pending"] = True
                stt_state["job_id"] += 1
                return
            _start_stt(reason)

        def _maybe_auto_stt(reason: str) -> None:
            if not stt_auto_run_var.get():
                return
            _request_stt(reason)

        stt_btn.configure(command=lambda: _request_stt("manual"))
        stt_cancel_btn.configure(command=_cancel_stt)

        template_frame = ttk.LabelFrame(content_frame, text="Шаблон карточки")
        template_frame.pack(fill=tk.X, padx=12, pady=(0, 10))
        ttk.Label(template_frame, text="Выберите шаблон карточки:").pack(anchor="w", padx=6, pady=(6, 0))
        self._build_generation_template_selector(template_frame, deck_id, padx=6, pady=(0, 6))

        drafts_frame = ttk.LabelFrame(content_frame, text="Черновики карточек")
        drafts_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        drafts_pane = ttk.Panedwindow(drafts_frame, orient=tk.HORIZONTAL)
        drafts_pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        drafts_left = ttk.Frame(drafts_pane)
        drafts_right = ttk.Frame(drafts_pane)
        drafts_pane.add(drafts_left, weight=1)
        drafts_pane.add(drafts_right, weight=3)

        ttk.Label(drafts_left, text="Черновики").pack(anchor="w")
        draft_listbox = tk.Listbox(drafts_left, selectmode=tk.EXTENDED, height=14)
        draft_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        draft_scroll = ttk.Scrollbar(drafts_left, orient="vertical", command=draft_listbox.yview)
        draft_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        draft_listbox.configure(yscrollcommand=draft_scroll.set)

        draft_btn_row = ttk.Frame(drafts_left)
        draft_btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(draft_btn_row, text="Добавить карточку", command=lambda: _add_draft_from_editor()).pack(
            side=tk.LEFT
        )
        ttk.Button(draft_btn_row, text="Удалить", command=lambda: _delete_selected_drafts()).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(draft_btn_row, text="Очистить", command=lambda: _clear_drafts()).pack(side=tk.LEFT)

        notebook = ttk.Notebook(drafts_right)
        notebook.pack(fill=tk.BOTH, expand=True)
        front_tab = ttk.Frame(notebook)
        back_tab = ttk.Frame(notebook)
        notebook.add(front_tab, text="Front")
        notebook.add(back_tab, text="Back")

        front_editor = build_rich_text_editor(front_tab, colors, on_change=lambda: _handle_editor_change())
        back_editor = build_rich_text_editor(back_tab, colors, on_change=lambda: _handle_editor_change())
        front_editor["frame"].pack(fill=tk.BOTH, expand=True)
        back_editor["frame"].pack(fill=tk.BOTH, expand=True)

        draft_state = {"active_index": None}

        def _draft_title(idx: int, draft: dict) -> str:
            base_text = draft.get("front_html") or draft.get("front_text") or ""
            plain = plain_text_from_html(base_text)
            preview = " ".join(plain.split())
            if not preview:
                preview = "Без названия"
            if len(preview) > 42:
                preview = preview[:39] + "..."
            return f"{idx + 1}. {preview}"

        def _refresh_draft_list(select_index: int | None = None) -> None:
            draft_listbox.delete(0, tk.END)
            for idx, draft in enumerate(state["drafts"]):
                draft_listbox.insert(tk.END, _draft_title(idx, draft))
            if select_index is not None and state["drafts"]:
                idx = max(0, min(select_index, len(state["drafts"]) - 1))
                draft_listbox.selection_set(idx)
                draft_listbox.activate(idx)
                _load_draft_into_editor(idx)
            elif not state["drafts"]:
                draft_state["active_index"] = None
                front_editor["set_html"]("")
                back_editor["set_html"]("")

        def _sync_active_draft() -> None:
            idx = draft_state.get("active_index")
            if idx is None or idx >= len(state["drafts"]):
                return
            state["drafts"][idx]["front_html"] = front_editor["get_html"]()
            state["drafts"][idx]["back_html"] = back_editor["get_html"]()
            draft_listbox.delete(idx)
            draft_listbox.insert(idx, _draft_title(idx, state["drafts"][idx]))
            draft_listbox.selection_set(idx)
            draft_listbox.activate(idx)

        def _load_draft_into_editor(idx: int) -> None:
            if idx < 0 or idx >= len(state["drafts"]):
                return
            draft_state["active_index"] = idx
            draft = state["drafts"][idx]
            front_editor["set_html"](draft.get("front_html") or "")
            back_editor["set_html"](draft.get("back_html") or "")

        def _on_draft_select(_event=None) -> None:
            selection = draft_listbox.curselection()
            _sync_active_draft()
            if not selection:
                draft_state["active_index"] = None
                _update_save_pricing()
                return
            _load_draft_into_editor(selection[0])
            _update_save_pricing()

        def _add_draft_from_editor() -> None:
            _sync_active_draft()
            front_html = front_editor["get_html"]()
            back_html = back_editor["get_html"]()
            state["drafts"].append({"front_html": front_html, "back_html": back_html})
            _refresh_draft_list(select_index=len(state["drafts"]) - 1)
            _update_save_pricing()

        def _delete_selected_drafts() -> None:
            selection = list(draft_listbox.curselection())
            if not selection:
                return
            for idx in sorted(selection, reverse=True):
                if 0 <= idx < len(state["drafts"]):
                    state["drafts"].pop(idx)
            _refresh_draft_list(select_index=0 if state["drafts"] else None)
            _update_save_pricing()

        def _clear_drafts() -> None:
            if state["drafts"] and not messagebox.askyesno(
                "Очистить черновики",
                "Удалить все черновики карточек?",
                parent=win,
            ):
                return
            state["drafts"] = []
            _refresh_draft_list(select_index=None)
            _update_save_pricing()

        def _handle_editor_change() -> None:
            if draft_state.get("active_index") is None:
                return
            _sync_active_draft()
            _update_save_pricing()

        draft_listbox.bind("<<ListboxSelect>>", _on_draft_select)

        front_video_label = tk.StringVar(value="Видео не добавлено")
        back_video_label = tk.StringVar(value="Видео не добавлено")

        def _require_clip() -> bool:
            if not state.get("clip_path"):
                messagebox.showinfo("Видео", "Сначала нарежьте клип.", parent=win)
                return False
            return True

        def insert_front_video():
            if not _require_clip():
                return
            state["front_video"] = state.get("clip_path")
            front_video_label.set(f"🎬 {os.path.basename(state['front_video'])}")

        def insert_back_video():
            if not _require_clip():
                return
            state["back_video"] = state.get("clip_path")
            back_video_label.set(f"🎬 {os.path.basename(state['back_video'])}")

        def clear_front_video():
            state["front_video"] = None
            front_video_label.set("Видео не добавлено")

        def clear_back_video():
            state["back_video"] = None
            back_video_label.set("Видео не добавлено")

        def _refresh_video_labels() -> None:
            if state.get("front_video"):
                front_video_label.set(f"🎬 {os.path.basename(state['front_video'])}")
            else:
                front_video_label.set("Видео не добавлено")
            if state.get("back_video"):
                back_video_label.set(f"🎬 {os.path.basename(state['back_video'])}")
            else:
                back_video_label.set("Видео не добавлено")

        def _ensure_default_video_selection() -> None:
            default_video = state.get("clip_path") or state.get("video_path")
            if not default_video:
                return
            if state.get("front_video") is None:
                state["front_video"] = default_video
            if state.get("back_video") is None:
                state["back_video"] = default_video
            _refresh_video_labels()

        insert_frame = ttk.Frame(content_frame)
        insert_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        ttk.Button(insert_frame, text="Вставить видео на Front", command=insert_front_video).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Label(insert_frame, textvariable=front_video_label).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(insert_frame, text="Очистить", command=clear_front_video).pack(side=tk.LEFT)

        ttk.Button(insert_frame, text="Вставить видео на Back", command=insert_back_video).pack(
            side=tk.LEFT, padx=(16, 6)
        )
        ttk.Label(insert_frame, textvariable=back_video_label).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(insert_frame, text="Очистить", command=clear_back_video).pack(side=tk.LEFT)

        def _auto_generate_drafts(text: str) -> None:
            sentences = split_into_sentences(text or "")
            if not sentences:
                return
            _ensure_default_video_selection()
            drafts = []
            for sentence in sentences:
                front_html = html.escape(sentence).replace("\n", "<br>")
                drafts.append(
                    {
                        "front_html": front_html,
                        "back_html": "",
                    }
                )
            state["drafts"] = drafts
            _refresh_draft_list(select_index=0)
            _update_save_pricing()

        preview_state = {"window": None, "view": None, "side": "front"}
        template_warning_state = {"shown": False}

        def _warn_missing_html_template() -> None:
            if template_warning_state["shown"]:
                return
            template = get_active_generation_template(deck_id)
            front_template = template.get("front_template") or ""
            back_template = template.get("back_template") or ""
            if "{front_html}" in front_template and "{back_html}" in back_template:
                return
            template_warning_state["shown"] = True
            messagebox.showwarning(
                "Шаблон карточки",
                (
                    "Выбранный шаблон не содержит {front_html}/{back_html}.\n"
                    "Текст и форматирование редактора не будут отображаться.\n"
                    f"Рекомендуется выбрать шаблон «{VIDEO_TEXT_TEMPLATE_NAME}»."
                ),
                parent=win,
            )

        def _build_ctx() -> dict:
            front_text = front_editor["get_plain_text"]().strip()
            back_text = back_editor["get_plain_text"]().strip()
            front_html = front_editor["get_html"]()
            back_html = back_editor["get_html"]()
            front_video_html = build_video_html(state.get("front_video"), state.get("poster_path"))
            back_video_html = build_video_html(state.get("back_video"), state.get("poster_path"))
            poster_path = state.get("poster_path")
            front_image_html = build_image_html(poster_path) if state.get("front_video") else ""
            back_image_html = build_image_html(poster_path) if state.get("back_video") else ""
            front_audio_html = ""
            back_audio_html = ""
            template = get_active_generation_template(deck_id)
            return {
                "question": template.get("question") or "",
                "front_html": front_html,
                "back_html": back_html,
                "front": front_text,
                "back": back_text,
                "front_video_html": front_video_html,
                "back_video_html": back_video_html,
                "front_video": front_video_html,
                "back_video": back_video_html,
                "video": front_video_html or back_video_html,
                "front_image": front_image_html,
                "back_image": back_image_html,
                "image": front_image_html or back_image_html,
                "front_audio": front_audio_html,
                "back_audio": back_audio_html,
                "audio": front_audio_html or back_audio_html,
            }

        def _build_preview_card() -> dict:
            ctx = _build_ctx()
            front, back = render_card_html(deck_id, ctx)
            poster_path = state.get("poster_path")
            return {
                "front": front,
                "back": back,
                "front_image_path": poster_path if state.get("front_video") else None,
                "back_image_path": poster_path if state.get("back_video") else None,
                "front_video_path": state.get("front_video"),
                "back_video_path": state.get("back_video"),
                "audio_path": None,
            }

        def open_preview():
            _warn_missing_html_template()
            preview_win = tk.Toplevel(win)
            preview_win.title("Предпросмотр карточки")
            preview_win.geometry("900x620")
            preview_win.grab_set()
            apply_dark_theme_to_window(preview_win, colors)

            container = tk.Frame(preview_win, bg=DARK_BG)
            container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
            controls = tk.Frame(container, bg=DARK_BG)
            controls.pack(fill=tk.X, pady=(0, 8))

            def _set_side(side: str) -> None:
                preview_state["side"] = side
                update_preview()

            ttk.Button(controls, text="Лицевая", command=lambda: _set_side("front")).pack(
                side=tk.LEFT, padx=4
            )
            ttk.Button(controls, text="Обратная", command=lambda: _set_side("back")).pack(
                side=tk.LEFT, padx=4
            )

            card_wrap = tk.Frame(
                container,
                bg=DARK_BG,
                highlightbackground=CARD_BORDER,
                highlightthickness=1,
                bd=0,
                width=CARD_VIEW_WIDTH,
                height=CARD_VIEW_HEIGHT,
            )
            card_wrap.pack(padx=10, pady=10)
            card_wrap.pack_propagate(False)
            html_view = CardHtmlView(card_wrap)

            preview_state["window"] = preview_win
            preview_state["view"] = html_view
            preview_state["side"] = "front"

            def update_preview() -> None:
                ctx = _build_ctx()
                front_html, back_html = render_card_html(deck_id, ctx)
                side = preview_state.get("side") or "front"
                html_value = back_html if side == "back" else front_html
                card_bg, card_text, _ = get_card_surface_colors(card_wrap)
                html_view.set_html(html_value or "", card_bg=card_bg, card_text_color=card_text)

            def _on_close():
                preview_state["window"] = None
                preview_state["view"] = None
                preview_win.destroy()

            preview_win.protocol("WM_DELETE_WINDOW", _on_close)
            update_preview()

        cost_frame = ttk.Frame(content_frame)
        cost_frame.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Button(cost_frame, text="Предпросмотр", command=open_preview).pack(side=tk.LEFT)

        save_count_var = tk.StringVar(value="0")
        save_total_var = tk.StringVar(value="0")
        save_hint_var = tk.StringVar(value="")

        ttk.Label(cost_frame, text="Карточек к сохранению:").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Label(cost_frame, textvariable=save_count_var).pack(side=tk.LEFT)

        coin_icon, _ = self._load_credit_icon_pair()
        cost_row = ttk.Frame(cost_frame)
        cost_row.pack(side=tk.RIGHT)
        ttk.Label(cost_row, text="Итого:").pack(side=tk.LEFT, padx=(0, 6))
        if coin_icon:
            coin_label = tk.Label(cost_row, image=coin_icon)
            coin_label.image = coin_icon
            coin_label.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(cost_row, textvariable=save_total_var).pack(side=tk.LEFT)

        save_btn = ttk.Button(cost_frame, text="Сохранить", command=lambda: save_cards())
        save_btn.pack(side=tk.RIGHT, padx=(0, 12))
        action_buttons.append(save_btn)

        ttk.Label(cost_frame, textvariable=save_hint_var, foreground="red").pack(side=tk.RIGHT, padx=(0, 12))

        def _get_video_clip_card_price() -> int:
            self.user_account = ensure_user_account(self.user_id)
            self.user_profile = ensure_user_profile_row(self.user_id)
            now_ts = int(time.time())
            premium_plus = bool(self.user_profile.get("premium_plus")) or self.user_account.get("status") == "premium_plus"
            if premium_plus:
                return 5
            if self.is_premium_active() or int(self.user_account.get("premium_until") or 0) > now_ts:
                return 10
            return 25

        def _get_drafts_to_save() -> list[dict]:
            _sync_active_draft()
            selection = draft_listbox.curselection()
            if selection:
                return [state["drafts"][idx] for idx in selection if idx < len(state["drafts"])]
            return list(state["drafts"])

        def _update_save_pricing() -> None:
            drafts_to_save = _get_drafts_to_save()
            count = len(drafts_to_save)
            price_per_card = _get_video_clip_card_price()
            total = price_per_card * count
            save_count_var.set(str(count))
            save_total_var.set(str(total))
            if stt_state.get("running"):
                save_btn.configure(state=tk.DISABLED)
                save_hint_var.set("Идёт распознавание…")
                return
            if count == 0:
                save_btn.configure(state=tk.DISABLED)
                save_hint_var.set("Нет карточек для сохранения")
                return
            if not self.can_afford(total):
                save_btn.configure(state=tk.DISABLED)
                save_hint_var.set("Не хватает кредитов")
                return
            save_btn.configure(state=tk.NORMAL)
            save_hint_var.set("")

        def save_cards():
            if not deck_id:
                messagebox.showerror("Ошибка", "Колода не выбрана.", parent=win)
                return
            drafts_to_save = _get_drafts_to_save()
            if not drafts_to_save:
                messagebox.showwarning("Сохранение", "Нет карточек для сохранения.", parent=win)
                return
            _warn_missing_html_template()
            price_per_card = _get_video_clip_card_price()
            total_cost = price_per_card * len(drafts_to_save)
            if not self.can_afford(total_cost):
                _update_save_pricing()
                messagebox.showwarning("Кредиты", "Недостаточно кредитов.", parent=win)
                return
            poster_path = state.get("poster_path")
            front_video_path = state.get("front_video")
            back_video_path = state.get("back_video")
            front_video_html = build_video_html(front_video_path, poster_path)
            back_video_html = build_video_html(back_video_path, poster_path)
            front_image_html = build_image_html(poster_path) if front_video_path else ""
            back_image_html = build_image_html(poster_path) if back_video_path else ""
            template = get_active_generation_template(deck_id)
            template_question = template.get("question") or ""

            conn = get_connection()

            def _op():
                spend_credits_in_transaction(
                    conn,
                    self.user_id,
                    total_cost,
                    "video_clip_cards",
                    meta={"deck_id": deck_id, "count": len(drafts_to_save)},
                )
                created = 0
                for draft in drafts_to_save:
                    ctx = {
                        "question": template_question,
                        "front_html": draft.get("front_html") or "",
                        "back_html": draft.get("back_html") or "",
                        "front": plain_text_from_html(draft.get("front_html") or ""),
                        "back": plain_text_from_html(draft.get("back_html") or ""),
                        "front_video_html": front_video_html,
                        "back_video_html": back_video_html,
                        "front_video": front_video_html,
                        "back_video": back_video_html,
                        "video": front_video_html or back_video_html,
                        "front_image": front_image_html,
                        "back_image": back_image_html,
                        "image": front_image_html or back_image_html,
                        "front_audio": "",
                        "back_audio": "",
                        "audio": "",
                    }
                    front, back = render_card_html(deck_id, ctx)
                    note_fields = {
                        "word": os.path.basename(front_video_path or back_video_path or state.get("clip_path") or "clip"),
                        "translation": "",
                        "example": "",
                        "level": 1,
                        "image": poster_path or "",
                        "front": front,
                        "back": back,
                        "front_image_path": poster_path if front_video_path else None,
                        "back_image_path": poster_path if back_video_path else None,
                        "audio_path": None,
                        "front_html": front,
                        "back_html": back,
                        "front_video_html": ctx.get("front_video_html"),
                        "back_video_html": ctx.get("back_video_html"),
                        "front_video": ctx.get("front_video"),
                        "back_video": ctx.get("back_video"),
                    }
                    note_id, _ = create_note_with_cards_in_transaction(
                        conn,
                        deck_id,
                        note_fields,
                        note_type_id=ensure_generated_note_type_id(conn),
                        tags="video clip",
                    )
                    if front_video_path:
                        insert_media(conn, note_id=note_id, type="video", path=front_video_path, side="front")
                        if poster_path:
                            insert_media(conn, note_id=note_id, type="image", path=poster_path, side="front")
                    if back_video_path:
                        insert_media(conn, note_id=note_id, type="video", path=back_video_path, side="back")
                        if poster_path:
                            insert_media(conn, note_id=note_id, type="image", path=poster_path, side="back")
                    created += 1
                return created

            try:
                created = commit_with_retry(conn, _op)
            except Exception as exc:
                conn.rollback()
                messagebox.showerror("Ошибка", f"Не удалось сохранить карточки: {exc}", parent=win)
                return
            finally:
                conn.close()

            self.refresh_balance_display()
            self.refresh_decks()
            self.update_deck_preview()
            self.update_overdue_badge()
            messagebox.showinfo(
                "Сохранено",
                f"✅ Сохранено {created} карточек, списано {total_cost} кредитов",
                parent=win,
            )
            _on_close()

        self.register_balance_observer(_update_save_pricing)
        _update_save_pricing()

        def _on_close():
            self.unregister_balance_observer(_update_save_pricing)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

    def open_apkg_import_window(self):
        win = tk.Toplevel(self)
        apply_window_icon(win, self._logo_big, ico_path=os.path.join(BASE_DIR, "assets", "app.ico"))
        win.title("Импорт Anki (.apkg)")
        win.geometry("720x520")
        win.grab_set()

        file_var = tk.StringVar()
        keep_schedule_var = tk.BooleanVar(value=False)
        restart_var = tk.BooleanVar(value=True)
        media_var = tk.BooleanVar(value=True)
        revlog_var = tk.BooleanVar(value=False)

        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="0/0")
        processing_task: dict[str, BackgroundTask | None] = {"task": None}

        def log_msg(msg: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, msg + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        def handle_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                progress_bar.configure(maximum=max(int(total), 1))
                progress_var.set(int(done))
                progress_label_var.set(f"{done}/{total} {label}")
                self.update_loading(done, total, f"{label} {done}/{total}")
            elif kind == "log":
                log_msg(str(event[1]))
            elif kind == "done":
                summary = event[1] or {}
                if processing_task["task"]:
                    self.unregister_bg_handler(processing_task["task"].queue)
                    processing_task["task"] = None
                self.hide_loading()
                messagebox.showinfo(
                    "Импорт завершен",
                    (
                        f"Заметок: {summary.get('notes', 0)}\n"
                        f"Карточек: {summary.get('cards', 0)}\n"
                        f"Медиа: {summary.get('media', 0)}\n"
                        f"Ошибок: {summary.get('errors', 0)}"
                    ),
                )
                self.refresh_decks()
                gui_hooks.import_did_finish(summary)
            elif kind == "error":
                if processing_task["task"]:
                    self.unregister_bg_handler(processing_task["task"].queue)
                    processing_task["task"] = None
                msg = str(event[1])
                if "Неподдерживаемый формат Anki пакета" in msg:
                    messagebox.showerror("Неподдерживаемый формат", msg)
                else:
                    messagebox.showerror("Ошибка импорта", msg)
                self.hide_loading()

        def browse_file():
            path = filedialog.askopenfilename(filetypes=[("Anki .apkg", "*.apkg"), ("Все файлы", "*.*")])
            if path:
                file_var.set(path)

        def sync_modes():
            if keep_schedule_var.get():
                restart_var.set(False)
            elif not restart_var.get():
                restart_var.set(True)

        def start_import():
            if processing_task["task"]:
                messagebox.showinfo("Занято", "Импорт уже выполняется")
                return

            apkg_path = file_var.get().strip()
            if not apkg_path or not os.path.exists(apkg_path):
                messagebox.showerror("Ошибка", "Укажите существующий .apkg файл")
                return

            options = {
                "keep_schedule": keep_schedule_var.get(),
                "start_fresh": restart_var.get(),
                "import_media": media_var.get(),
                "import_revlog": revlog_var.get(),
            }

            def worker(task_obj: BackgroundTask):
                return import_apkg(
                    apkg_path,
                    target_parent_deck_id=self.selected_deck_id,
                    options=options,
                    progress_cb=lambda *ev: task_obj.queue.put(ev),
                    cancel_flag=task_obj.cancelled,
                )

            task = start_background_task(worker)
            processing_task["task"] = task
            self.register_bg_handler(task.queue, handle_event)
            log_msg("Старт импорта…")
            self.show_loading("Импорт .apkg", determinate=False)

        def cancel_import():
            if processing_task["task"]:
                processing_task["task"].cancel()
                log_msg("Отмена запрошена")
            self.hide_loading()

        top_frame = ttk.Frame(win)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(top_frame, text="Файл .apkg:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top_frame, textvariable=file_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(top_frame, text="…", width=4, command=browse_file).grid(row=0, column=2)
        top_frame.columnconfigure(1, weight=1)

        options_frame = ttk.LabelFrame(win, text="Опции")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Checkbutton(options_frame, text="Сохранить расписание Anki", variable=keep_schedule_var, command=sync_modes).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Начать заново в SRS", variable=restart_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Импортировать медиа", variable=media_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Импортировать revlog (позже)", variable=revlog_var).grid(row=3, column=0, sticky="w")

        progress_bar = ttk.Progressbar(win, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, padx=10, pady=(10, 0))
        ttk.Label(win, textvariable=progress_label_var).pack(anchor="e", padx=10)

        log_box = tk.Text(win, height=12, state="disabled")
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Импортировать", command=start_import).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=cancel_import).pack(side=tk.RIGHT)

    def open_image_id_import_window(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Импорт изображений недоступен", "Чтобы прикреплять картинки, установите пакет Pillow и попробуйте снова.")
            return

        win = tk.Toplevel(self)
        win.title("Импорт картинок по ID (CSV)")
        win.geometry("820x680")
        win.grab_set()

        # variables
        csv_path_var = tk.StringVar()
        folder_var = tk.StringVar()
        deck_var = tk.StringVar()
        note_type_var = tk.StringVar()
        id_mode_var = tk.StringVar(value="filename")
        update_existing_var = tk.BooleanVar(value=False)
        move_files_var = tk.BooleanVar(value=False)
        watch_var = tk.BooleanVar(value=False)
        auto_confirm_var = tk.BooleanVar(value=True)
        interval_var = tk.IntVar(value=5)
        stop_flag = {"stop": False}
        pack_state = {
            "limit": 0,
            "cost": 0,
            "label": "",
            "remaining": 0,
            "limit_reached": False,
            "next_index": None,
        }
        import_label_var = tk.StringVar(value="")
        import_enabled = {"value": True}
        import_busy = {"value": False}

        decks = list_decks()
        if self.selected_deck_id:
            deck_var.set(str(self.selected_deck_id))
        elif decks:
            deck_var.set(str(decks[0]["id"]))

        note_types = list_note_types()
        if note_types:
            note_type_var.set(str(note_types[0]["id"]))
        else:
            note_type_var.set(str(ensure_basic_note_type_id()))

        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="")
        processing_task = {"task": None}
        running_watch_mode = {"value": False}

        def log_msg(msg: str):
            log_text.configure(state="normal")
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
            log_text.configure(state="disabled")

        def handle_import_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                display_total = pack_state["limit"] if pack_state["limit"] else total
                progress_bar.configure(maximum=max(display_total, 1))
                progress_var.set(done)
                progress_label_var.set(f"{label}: {done}/{display_total}")
                self.update_loading(done, display_total, f"{label}: {done}/{display_total}")
            elif kind == "log":
                log_msg(event[1])
            elif kind == "limit_reached":
                pack_state["limit_reached"] = True
                stop_flag["stop"] = True
                pack_state["remaining"] = 0
                progress_var.set(0)
                progress_label_var.set("")
                progress_bar.configure(maximum=max(pack_state["limit"], 1))
                self.hide_loading()
                messagebox.showinfo(
                    "Лимит пакета",
                    "Достигнут лимит пакета. Оплатите следующий пакет для продолжения.",
                )
            elif kind == "done":
                summary = event[1] or {"total": 0, "imported": 0, "updated": 0, "skipped": 0, "errors": 0}
                self.unregister_bg_handler(processing_task["task"].queue)
                processing_task["task"] = None
                btn_check.config(state=tk.NORMAL)
                import_busy["value"] = False
                update_import_button_state()
                btn_stop.config(state=tk.NORMAL)
                self.hide_loading()
                processed_in_pack = int(summary.get("processed_in_pack") or 0)
                if pack_state["remaining"]:
                    pack_state["remaining"] = max(0, int(pack_state["remaining"]) - processed_in_pack)
                if not pack_state["limit_reached"]:
                    msg = (
                        f"Всего: {summary['total']}\n"
                        f"Создано: {summary['imported']}\n"
                        f"Обновлено: {summary['updated']}\n"
                        f"Пропущено: {summary['skipped']}\n"
                        f"Ошибки: {summary['errors']}"
                    )
                    messagebox.showinfo("Импорт завершен", msg)
                    if summary.get("imported") or summary.get("updated"):
                        self.mark_first_import()
                    if (
                        running_watch_mode["value"]
                        and watch_var.get()
                        and not stop_flag.get("stop")
                        and pack_state.get("remaining", 0) > 0
                    ):
                        self.image_import_watch_job = win.after(
                            interval_var.get() * 1000,
                            lambda: process_files(
                                False,
                                True,
                                pack_limit=pack_state.get("remaining"),
                                pack_cost=pack_state.get("cost"),
                                pack_label=pack_state.get("label"),
                                charge_pack=False,
                            ),
                        )
            elif kind == "error":
                self.unregister_bg_handler(processing_task["task"].queue)
                processing_task["task"] = None
                btn_check.config(state=tk.NORMAL)
                import_busy["value"] = False
                update_import_button_state()
                btn_stop.config(state=tk.NORMAL)
                messagebox.showerror("Ошибка", event[1])
                self.hide_loading()

        def browse_csv():
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
            if path:
                csv_path_var.set(path)

        def browse_folder():
            path = filedialog.askdirectory()
            if path:
                folder_var.set(path)

        def show_coin_hint(_event=None):
            messagebox.showinfo(
                "Оплата и запуск",
                "Нажмите на монету для оплаты и запуска.",
            )
            return "break"

        def show_pro_required_hint(_event=None):
            messagebox.showinfo(
                "Требуется PRO подписка",
                "Требуется PRO подписка.",
            )
            return "break"

        def show_pro_feature_hint(_event=None):
            messagebox.showinfo(
                "Доступно в PRO",
                "Доступно в PRO.",
            )
            return "break"

        def ask_semi_confirmation(image_path: str, detected_id: int | None, csv_entry: dict | None):
            result = {"action": "skip", "data": None, "auto": auto_confirm_var.get()}
            if auto_confirm_var.get() and detected_id is not None and csv_entry:
                result["action"] = "create"
                result["data"] = {
                    "id": detected_id,
                    "word": csv_entry.get("word", ""),
                    "translation": csv_entry.get("translation", ""),
                    "example": csv_entry.get("example", ""),
                    "level": csv_entry.get("level", ""),
                }
                return result

            dialog = tk.Toplevel(win)
            dialog.title("Подтверждение импорта")
            dialog.geometry("640x520")
            dialog.grab_set()

            img_label = tk.Label(dialog)
            img_label.pack(pady=5)

            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    img.thumbnail((360, 260))
                    photo = ImageTk.PhotoImage(img)
                    img_label.configure(image=photo)
                    img_label.image = photo
                except Exception:
                    img_label.configure(text="(не удалось загрузить картинку)")

            form = ttk.Frame(dialog)
            form.pack(fill=tk.X, padx=10, pady=10)

            id_var = tk.StringVar(value=str(detected_id) if detected_id is not None else "")
            word_var = tk.StringVar(value=(csv_entry or {}).get("word", ""))
            tr_var = tk.StringVar(value=(csv_entry or {}).get("translation", ""))
            ex_var = tk.StringVar(value=(csv_entry or {}).get("example", ""))
            lvl_var = tk.StringVar(value=(csv_entry or {}).get("level", ""))
            auto_apply_var = tk.BooleanVar(value=auto_confirm_var.get())

            for idx, (lbl, var) in enumerate([
                ("ID", id_var),
                ("Word", word_var),
                ("Translation", tr_var),
                ("Example", ex_var),
                ("Level", lvl_var),
            ]):
                ttk.Label(form, text=lbl + ":").grid(row=idx, column=0, sticky="w", pady=3)
                ttk.Entry(form, textvariable=var).grid(row=idx, column=1, sticky="ew", padx=5)
            form.columnconfigure(1, weight=1)

            ttk.Checkbutton(dialog, text="Автоподтверждать далее", variable=auto_apply_var).pack(anchor="w", padx=10)

            actions = ttk.Frame(dialog)
            actions.pack(fill=tk.X, pady=10)

            result_var = tk.StringVar(value="skip")

            def do_create():
                result_var.set("create")
                dialog.destroy()

            def do_skip():
                result_var.set("skip")
                dialog.destroy()

            def do_stop():
                result_var.set("stop")
                dialog.destroy()

            ttk.Button(actions, text="Создать/Обновить", command=do_create).pack(side=tk.LEFT, padx=5)
            ttk.Button(actions, text="Пропустить", command=do_skip).pack(side=tk.LEFT, padx=5)
            ttk.Button(actions, text="Стоп", command=do_stop).pack(side=tk.LEFT, padx=5)

            dialog.bind("<Return>", lambda e: do_create())
            dialog.bind("<Escape>", lambda e: do_skip())

            dialog.wait_variable(result_var)

            result["action"] = result_var.get()
            result["data"] = {
                "id": int(id_var.get()) if id_var.get().isdigit() else None,
                "word": word_var.get().strip(),
                "translation": tr_var.get().strip(),
                "example": ex_var.get().strip(),
                "level": lvl_var.get().strip(),
            }
            result["auto"] = auto_apply_var.get()
            auto_confirm_var.set(auto_apply_var.get())
            return result

        def process_files(
            dry_run=False,
            watch_mode=False,
            pack_limit: int | None = None,
            pack_cost: int | None = None,
            pack_label: str | None = None,
            charge_pack: bool = False,
        ):
            if processing_task["task"]:
                messagebox.showinfo("Занято", "Импорт уже выполняется")
                return
            stop_flag["stop"] = False
            running_watch_mode["value"] = watch_mode
            pack_state["limit_reached"] = False
            pack_state["next_index"] = None
            pack_state["limit"] = int(pack_limit or 0) if not dry_run else 0
            pack_state["cost"] = int(pack_cost or 0) if not dry_run else 0
            pack_state["label"] = pack_label or ""
            if not dry_run and charge_pack and pack_limit is not None:
                pack_state["remaining"] = pack_limit
            csv_path = csv_path_var.get().strip()
            folder = folder_var.get().strip()
            if not csv_path or not os.path.exists(csv_path):
                messagebox.showerror("Ошибка", "Укажите CSV-файл")
                return
            if not folder or not os.path.isdir(folder):
                messagebox.showerror("Ошибка", "Укажите папку с изображениями")
                return
            try:
                csv_data = read_csv_dictionary(csv_path)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                return

            deck_id = deck_var.get()
            if not deck_id:
                messagebox.showerror("Ошибка", "Выберите колоду")
                return
            deck_id_int = int(deck_id)

            try:
                note_type_id = int(note_type_var.get())
            except ValueError:
                note_type_id = ensure_basic_note_type_id()

            if id_mode_var.get() == "semi":
                messagebox.showwarning("Режим semi", "Полуавто режим не поддерживает фоновый импорт")
                return

            imported_files = get_imported_files() if watch_mode else set()
            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ]
            files.sort()
            if not dry_run and charge_pack:
                if pack_cost is None:
                    messagebox.showerror("Ошибка", "Не удалось определить стоимость пакета")
                    return
                if not self.can_afford(pack_cost):
                    messagebox.showerror("Недостаточно кредитов", "Недостаточно кредитов. Пополните баланс.")
                    update_import_button_state()
                    return
                if not self.charge_credits(
                    pack_cost,
                    "image_id_import",
                    meta={
                        "files": pack_limit or len(files),
                        "csv": os.path.basename(csv_path),
                        "pack_limit": pack_limit or 0,
                    },
                ):
                    messagebox.showerror("Ошибка", "Не удалось списать кредиты")
                    update_import_button_state()
                    return

            progress_total = pack_limit if (pack_limit and not dry_run) else len(files)
            progress_bar.configure(maximum=max(progress_total, 1))
            self.show_loading("Импорт картинок", determinate=True, total=max(progress_total, 1))

            def worker(task_obj):
                summary = {"total": 0, "imported": 0, "updated": 0, "skipped": 0, "errors": 0}
                total_files = len(files)
                processed_in_pack = 0
                for idx, fp in enumerate(files, start=1):
                    if stop_flag.get("stop") or task_obj.cancelled():
                        break
                    if watch_mode and fp in imported_files:
                        continue
                    if pack_limit and not dry_run and processed_in_pack >= pack_limit:
                        break

                    summary["total"] += 1
                    id_val = None
                    if id_mode_var.get() == "filename":
                        id_val = extract_id_from_filename(fp)
                    if id_val is None and id_mode_var.get() == "ocr":
                        id_val = extract_id_with_ocr(fp)

                    if id_val is None:
                        task_obj.queue.put(("log", f"[ошибка] Не найден ID для {os.path.basename(fp)}"))
                        summary["errors"] += 1
                        processed_in_pack += 1
                        task_obj.queue.put(("progress", processed_in_pack, max(progress_total, 1), "Импорт"))
                        continue

                    entry = csv_data.get(id_val) if id_val is not None else None

                    if entry is None:
                        task_obj.queue.put(("log", f"[пропуск] ID {id_val} отсутствует в CSV"))
                        summary["skipped"] += 1
                        processed_in_pack += 1
                        task_obj.queue.put(("progress", processed_in_pack, max(progress_total, 1), "Импорт"))
                        continue

                    existing = find_note_by_source_id(deck_id_int, id_val)
                    if existing and not update_existing_var.get():
                        task_obj.queue.put(("log", f"[пропуск] ID {id_val} уже существует"))
                        summary["skipped"] += 1
                        processed_in_pack += 1
                        task_obj.queue.put(("progress", processed_in_pack, max(progress_total, 1), "Импорт"))
                        continue

                    media_path = copy_image_to_media(fp, id_val, move_files_var.get()) if not dry_run else fp

                    fields = {
                        "word": entry.get("word", ""),
                        "translation": entry.get("translation", ""),
                        "example": entry.get("example", ""),
                        "level": entry.get("level", ""),
                        "image": media_path,
                        "source_id": str(id_val),
                    }

                    if dry_run:
                        task_obj.queue.put(("log", f"[проверка] готова карточка ID {id_val}: {fields['word']} -> {fields['translation']}"))
                        summary["imported"] += 1
                        processed_in_pack += 1
                        task_obj.queue.put(("progress", processed_in_pack, max(progress_total, 1), "Импорт"))
                        continue

                    if existing:
                        try:
                            conn = get_connection()
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE notes SET fields_json = ? WHERE id = ?;",
                                (json.dumps(fields, ensure_ascii=False), existing["id"]),
                            )
                            conn.commit()
                            conn.close()
                            attach_media_to_note(existing["id"], [(media_path, "image")])
                            summary["updated"] += 1
                            task_obj.queue.put(("log", f"[обновлено] ID {id_val}"))
                        except Exception as e:
                            summary["errors"] += 1
                            task_obj.queue.put(("log", f"[ошибка] Не удалось обновить ID {id_val}: {e}"))
                    else:
                        try:
                            _, created_cards = create_note_with_cards(
                                deck_id_int,
                                fields,
                                note_type_id=note_type_id,
                                tags="import:image_id",
                            )
                            summary["imported"] += 1
                            task_obj.queue.put(("log", f"[создано] ID {id_val} (карточек: {created_cards})"))
                        except Exception as e:
                            summary["errors"] += 1
                            task_obj.queue.put(("log", f"[ошибка] Не удалось создать ID {id_val}: {e}"))

                    if watch_mode:
                        log_imported_file(fp)

                    processed_in_pack += 1
                    task_obj.queue.put(("progress", processed_in_pack, max(progress_total, 1), "Импорт"))

                summary["processed_in_pack"] = processed_in_pack
                if (
                    pack_limit
                    and not dry_run
                    and processed_in_pack >= pack_limit
                    and total_files > processed_in_pack
                    and not stop_flag.get("stop")
                    and not task_obj.cancelled()
                ):
                    pack_state["next_index"] = processed_in_pack + 1
                    task_obj.queue.put(("limit_reached",))
                return summary

            progress_var.set(0)
            progress_label_var.set("")
            btn_check.config(state=tk.DISABLED)
            import_busy["value"] = True
            update_import_button_state()
            btn_stop.config(state=tk.DISABLED)
            processing_task["task"] = start_background_task(worker)
            self.register_bg_handler(processing_task["task"].queue, handle_import_event)

        def stop_processing():
            stop_flag["stop"] = True
            if processing_task["task"]:
                processing_task["task"].cancel()
            btn_check.config(state=tk.NORMAL)
            import_busy["value"] = False
            update_import_button_state()
            btn_stop.config(state=tk.NORMAL)
            if self.image_import_watch_job:
                try:
                    win.after_cancel(self.image_import_watch_job)
                except Exception:
                    pass
                self.image_import_watch_job = None
            log_msg("[остановлено] Импорт остановлен пользователем")
            self.hide_loading()

        def start_watch_loop(pack_limit: int, pack_cost: int, pack_label: str):
            stop_flag["stop"] = False
            process_files(
                dry_run=False,
                watch_mode=True,
                pack_limit=pack_limit,
                pack_cost=pack_cost,
                pack_label=pack_label,
                charge_pack=True,
            )

        def on_coin_click(_event=None):
            limit, cost, pack_label = self.get_csv_import_pack_config()
            if import_busy["value"]:
                messagebox.showinfo("Занято", "Импорт уже выполняется")
                return "break"
            if not self.can_afford(cost):
                messagebox.showerror("Недостаточно кредитов", "Недостаточно кредитов. Пополните баланс.")
                update_import_button_state()
                return "break"
            pack_state["remaining"] = limit
            if watch_var.get():
                start_watch_loop(limit, cost, pack_label)
            else:
                process_files(
                    dry_run=False,
                    watch_mode=False,
                    pack_limit=limit,
                    pack_cost=cost,
                    pack_label=pack_label,
                    charge_pack=True,
                )
            return "break"

        # Layout
        main_frame = ttk.Frame(win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = ttk.LabelFrame(main_frame, text="Исходные данные")
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Label(file_frame, text="CSV словарь:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(file_frame, textvariable=csv_path_var, width=60).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="Выбрать", command=browse_csv).grid(row=0, column=2, padx=5)

        ttk.Label(file_frame, text="Папка с картинками:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(file_frame, textvariable=folder_var, width=60).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="Выбрать", command=browse_folder).grid(row=1, column=2, padx=5)
        file_frame.columnconfigure(1, weight=1)

        options_frame = ttk.LabelFrame(main_frame, text="Настройки")
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Колода:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        deck_combo = ttk.Combobox(options_frame, textvariable=deck_var, state="readonly")
        deck_combo['values'] = [str(d["id"]) for d in decks]
        deck_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=3)

        ttk.Label(options_frame, text="Note type:").grid(row=0, column=2, sticky="w", padx=5)
        note_combo = ttk.Combobox(options_frame, textvariable=note_type_var, state="readonly")
        note_combo['values'] = [str(nt["id"]) for nt in note_types]
        note_combo.grid(row=0, column=3, sticky="ew", padx=5, pady=3)

        pro_active = self.is_premium_active()
        if not pro_active and id_mode_var.get() in ("ocr", "semi"):
            id_mode_var.set("filename")
        if not pro_active:
            update_existing_var.set(False)

        ttk.Label(options_frame, text="Режим ID:").grid(row=1, column=0, sticky="w", padx=5)
        ttk.Radiobutton(options_frame, text="Имя файла", variable=id_mode_var, value="filename").grid(row=1, column=1, sticky="w")

        ocr_frame = ttk.Frame(options_frame)
        ocr_frame.grid(row=1, column=2, sticky="w")
        ocr_rb = ttk.Radiobutton(ocr_frame, text="OCR", variable=id_mode_var, value="ocr")
        ocr_rb.pack(side=tk.LEFT)
        ocr_crown = ttk.Label(ocr_frame, text="👑")
        ocr_crown.pack(side=tk.LEFT, padx=(2, 0))

        semi_frame = ttk.Frame(options_frame)
        semi_frame.grid(row=1, column=3, sticky="w")
        semi_rb = ttk.Radiobutton(semi_frame, text="Полуавто", variable=id_mode_var, value="semi")
        semi_rb.pack(side=tk.LEFT)
        semi_crown = ttk.Label(semi_frame, text="👑")
        semi_crown.pack(side=tk.LEFT, padx=(2, 0))

        update_frame = ttk.Frame(options_frame)
        update_frame.grid(row=2, column=0, sticky="w", padx=5, pady=3)
        update_cb = ttk.Checkbutton(update_frame, text="Обновлять существующие", variable=update_existing_var)
        update_cb.pack(side=tk.LEFT)
        update_crown = ttk.Label(update_frame, text="👑")
        update_crown.pack(side=tk.LEFT, padx=(2, 0))

        ttk.Checkbutton(options_frame, text="Перемещать файлы", variable=move_files_var).grid(row=2, column=1, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(options_frame, text="Следить за папкой (watch)", variable=watch_var).grid(row=2, column=2, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(options_frame, text="Автоподтверждать при совпадении (semi)", variable=auto_confirm_var).grid(row=2, column=3, sticky="w", padx=5, pady=3)

        if not pro_active:
            ocr_rb.configure(state=tk.DISABLED)
            semi_rb.configure(state=tk.DISABLED)
            update_cb.configure(state=tk.DISABLED)
            for widget in (ocr_frame, ocr_crown):
                widget.bind("<Button-1>", show_pro_required_hint)
            for widget in (semi_frame, semi_crown):
                widget.bind("<Button-1>", show_pro_required_hint)
            for widget in (update_frame, update_crown):
                widget.bind("<Button-1>", show_pro_feature_hint)

        ttk.Label(options_frame, text="Интервал watch (сек)").grid(row=3, column=0, sticky="w", padx=5)
        ttk.Spinbox(options_frame, from_=2, to=60, textvariable=interval_var, width=5).grid(row=3, column=1, sticky="w", padx=5)
        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(3, weight=1)

        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, expand=True, padx=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=5)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        btn_check = ttk.Button(btn_frame, text="Проверить", command=lambda: process_files(dry_run=True))
        btn_check.pack(side=tk.LEFT, padx=5)
        btn_import_frame = tk.Frame(btn_frame, relief="groove", bd=1, padx=6, pady=2)
        btn_import_frame.pack(side=tk.LEFT, padx=5)
        btn_import_text = tk.Label(btn_import_frame, textvariable=import_label_var)
        btn_import_text.pack(side=tk.LEFT, padx=(2, 6))
        default_import_fg = btn_import_text.cget("fg")
        coin_icon, coin_icon_disabled = self._load_credit_icon_pair(size=32)
        btn_import_coin = tk.Label(btn_import_frame, image=coin_icon)
        btn_import_coin.pack(side=tk.LEFT)
        btn_stop = ttk.Button(btn_frame, text="Остановить", command=stop_processing)
        btn_stop.pack(side=tk.LEFT, padx=5)

        log_text = tk.Text(main_frame, height=15, state="disabled")
        log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        def update_import_button_state():
            limit, cost, pack_label = self.get_csv_import_pack_config()
            import_label_var.set(f"Импортировать ({cost} ⚡)")
            pack_state["limit"] = limit
            pack_state["cost"] = cost
            pack_state["label"] = pack_label
            if import_busy["value"]:
                btn_import_text.configure(fg="gray")
                btn_import_coin.configure(cursor="arrow")
                current_icon = coin_icon_disabled or coin_icon
                btn_import_coin.configure(image=current_icon)
                btn_import_coin.image = current_icon
                return
            if not self.can_afford(cost):
                import_enabled["value"] = False
                btn_import_text.configure(fg="gray")
                btn_import_coin.configure(cursor="arrow")
                current_icon = coin_icon_disabled or coin_icon
                btn_import_coin.configure(image=current_icon)
                btn_import_coin.image = current_icon
            else:
                import_enabled["value"] = True
                btn_import_text.configure(fg=default_import_fg)
                btn_import_coin.configure(cursor="hand2")
                current_icon = coin_icon or coin_icon_disabled
                btn_import_coin.configure(image=current_icon)
                btn_import_coin.image = current_icon

        btn_import_text.bind("<Button-1>", show_coin_hint)
        btn_import_frame.bind("<Button-1>", show_coin_hint)
        btn_import_coin.bind("<Button-1>", on_coin_click)
        if coin_icon:
            btn_import_coin.image = coin_icon
        if coin_icon_disabled:
            btn_import_coin.disabled_image = coin_icon_disabled

        update_import_button_state()
        self.register_balance_observer(update_import_button_state)

        def on_close():
            self.unregister_balance_observer(update_import_button_state)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def open_wikimedia_csv_window(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Изображения недоступны", "Для загрузки картинок установите пакет Pillow и повторите попытку.")
            return

        if not self.is_premium_active():
            messagebox.showinfo(
                "Нужна подписка",
                "Загрузка с Wikimedia доступна в Pro. Откройте личный кабинет для активации.",
            )
            return

        win = tk.Toplevel(self)
        win.title("Картинки по CSV (Wikimedia)")
        win.geometry("720x520")
        win.grab_set()

        csv_path_var = tk.StringVar()
        status_var = tk.StringVar(value="Готово")
        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="0/0")

        task_holder: dict[str, BackgroundTask | None] = {"task": None}

        def log_msg(msg: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, msg + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        def handle_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                progress_bar.configure(maximum=max(total, 1))
                progress_var.set(done)
                progress_label_var.set(f"{done}/{total}")
                status_var.set(label)
                log_msg(label)
            elif kind == "done":
                summary = event[1] or {}
                status_var.set("Готово")
                progress_label_var.set(f"{summary.get('done', 0)}/{summary.get('total', 0)}")
                log_msg("Завершено")
                if summary.get("done"):
                    self.mark_first_import()
                _finish_task()
            elif kind == "cancelled":
                status_var.set("Отменено пользователем")
                log_msg("[остановлено] Пользователь отменил загрузку")
                _finish_task()
            elif kind == "error":
                status_var.set(event[1])
                messagebox.showerror("Ошибка", event[1])
                _finish_task()

        def _finish_task():
            if task_holder["task"]:
                self.unregister_bg_handler(task_holder["task"].queue)
                task_holder["task"] = None
            btn_start.config(state=tk.NORMAL)
            btn_cancel.config(state=tk.DISABLED)

        def browse_csv():
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
            if path:
                csv_path_var.set(path)

        def start_fetch():
            if task_holder["task"]:
                return
            csv_path = csv_path_var.get().strip()
            if not csv_path:
                messagebox.showerror("Ошибка", "Укажите CSV-файл")
                return
            if not self.consume_wikimedia_ticket():
                return
            status_var.set("Начинаю загрузку…")
            progress_var.set(0)
            progress_label_var.set("0/0")

            from wiki_image_fetcher import process_csv_file

            def worker(task_obj: BackgroundTask, csv_file: str):
                def progress_cb(done: int, total: int, label: str):
                    task_obj.queue.put(("progress", done, total, label))

                try:
                    done, total = process_csv_file(
                        csv_file,
                        images_dir="images",
                        stop_checker=task_obj.cancelled,
                        progress_callback=progress_cb,
                    )
                    if task_obj.cancelled():
                        task_obj.queue.put(("cancelled",))
                    else:
                        task_obj.queue.put(("done", {"done": done, "total": total}))
                except Exception as exc:  # noqa: BLE001
                    task_obj.queue.put(("error", str(exc)))

            task = start_background_task(worker, csv_path)
            task_holder["task"] = task
            self.register_bg_handler(task.queue, handle_event)
            btn_start.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.NORMAL)

        def cancel_fetch():
            if task_holder["task"]:
                task_holder["task"].cancel()
                status_var.set("Остановка…")
                btn_cancel.config(state=tk.DISABLED)

        def on_close():
            cancel_fetch()
            if task_holder["task"]:
                self.unregister_bg_handler(task_holder["task"].queue)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

        top_frame = ttk.Frame(win)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="CSV файл:").grid(row=0, column=0, sticky="w", padx=5)
        entry = ttk.Entry(top_frame, textvariable=csv_path_var)
        entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(top_frame, text="Выбрать", command=browse_csv).grid(row=0, column=2, padx=5)
        top_frame.columnconfigure(1, weight=1)

        progress_frame = ttk.Frame(win)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, expand=True, padx=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=5)
        ttk.Label(progress_frame, textvariable=status_var, foreground="gray").pack(anchor="w", padx=5)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        btn_start = ttk.Button(btn_frame, text="Старт", command=start_fetch)
        btn_start.pack(side=tk.LEFT, padx=5)
        btn_cancel = ttk.Button(btn_frame, text="Отмена", command=cancel_fetch, state=tk.DISABLED)
        btn_cancel.pack(side=tk.LEFT, padx=5)

        log_box = tk.Text(win, height=15, state="disabled")
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def open_csv_import_window(self):
        win = tk.Toplevel(self)
        apply_window_icon(win, self._logo_big, ico_path=os.path.join(BASE_DIR, "assets", "app.ico"))
        win.title("Импорт CSV колоды")
        win.geometry("880x740")
        win.grab_set()

        file_var = tk.StringVar()
        deck_var = tk.StringVar()
        has_headers_var = tk.BooleanVar(value=True)
        upsert_var = tk.BooleanVar(value=True)
        skip_existing_var = tk.BooleanVar(value=False)
        generate_id_var = tk.BooleanVar(value=True)
        attach_images_var = tk.BooleanVar(value=True)
        preserve_srs_var = tk.BooleanVar(value=False)
        reset_srs_var = tk.BooleanVar(value=False)
        start_state_var = tk.StringVar(value="new")
        start_phase_var = tk.IntVar(value=1)
        images_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "images"))

        id_col_var = tk.IntVar(value=1)
        word_col_var = tk.IntVar(value=2)
        translation_col_var = tk.IntVar(value=3)
        example_col_var = tk.IntVar(value=4)
        notes_col_var = tk.IntVar(value=5)
        tags_col_var = tk.IntVar(value=6)
        front_col_var = tk.IntVar(value=1)
        back_col_var = tk.IntVar(value=2)

        decks = list_decks()
        if decks:
            deck_var.set(str(decks[0]["id"]))

        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="0/0")
        processing_task: dict[str, BackgroundTask | None] = {"task": None}

        def refresh_decks_list():
            self.refresh_decks()
            decks_local = list_decks()
            deck_box["values"] = [f"{d['id']}: {d['name']}" for d in decks_local]
            if self.selected_deck_id:
                deck_var.set(str(self.selected_deck_id))
            elif decks_local:
                deck_var.set(str(decks_local[0]["id"]))

        def log_msg(msg: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, msg + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        def handle_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                progress_bar.configure(maximum=max(total, 1))
                progress_var.set(done)
                progress_label_var.set(f"{done}/{total} {label}")
            elif kind == "log":
                log_msg(event[1])
            elif kind == "done":
                summary = event[1] or {"total": 0, "created": 0, "updated": 0, "skipped": 0, "errors": 0}
                messagebox.showinfo(
                    "Готово",
                    (
                        f"Всего строк: {summary['total']}\n"
                        f"Создано: {summary['created']}\n"
                        f"Обновлено: {summary['updated']}\n"
                        f"Пропущено: {summary['skipped']}\n"
                        f"Ошибки: {summary['errors']}"
                    ),
                )
                gui_hooks.import_did_finish(summary)
                self.unregister_bg_handler(processing_task["task"].queue)
                processing_task["task"] = None
            elif kind == "error":
                messagebox.showerror("Ошибка", event[1])
                self.unregister_bg_handler(processing_task["task"].queue)
                processing_task["task"] = None

        def browse_file():
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
            if path:
                file_var.set(path)

        def browse_images_dir():
            path = filedialog.askdirectory(initialdir=images_dir_var.get() or os.getcwd())
            if path:
                images_dir_var.set(path)

        def on_toggle_headers():
            manual_frame.grid_remove() if has_headers_var.get() else manual_frame.grid()

        def build_manual_map():
            manual_map: dict[str, int] = {}
            for target, var in (
                ("id", id_col_var),
                ("word", word_col_var),
                ("translation", translation_col_var),
                ("example", example_col_var),
                ("notes", notes_col_var),
                ("tags", tags_col_var),
                ("front", front_col_var),
                ("back", back_col_var),
            ):
                if var.get() > 0:
                    manual_map[target] = var.get() - 1
            return manual_map

        def start_import():
            if processing_task["task"]:
                messagebox.showinfo("Занято", "Импорт уже выполняется")
                return

            csv_path = file_var.get().strip()
            if not csv_path or not os.path.exists(csv_path):
                messagebox.showerror("Ошибка", "Укажите существующий CSV файл")
                return

            deck_id_raw = deck_var.get()
            if not deck_id_raw:
                messagebox.showerror("Ошибка", "Выберите колоду")
                return
            deck_id = int(deck_id_raw.split(":", 1)[0]) if ":" in deck_id_raw else int(deck_id_raw)

            mapping_mode = {"variant": "auto"}
            manual_tags_idx: int | None = None
            manual_id_idx: int | None = None
            if not has_headers_var.get():
                manual_map = build_manual_map()
                manual_tags_idx = manual_map.pop("tags", None)
                manual_id_idx = manual_map.pop("id", None)
                mapping_mode = {"variant": "manual", "manual_map": manual_map}

            attach_images = attach_images_var.get()
            images_dir = images_dir_var.get().strip() or os.path.join(os.getcwd(), "images")

            def worker(task_obj: BackgroundTask):
                summary = {"total": 0, "created": 0, "updated": 0, "skipped": 0, "errors": 0}
                conn = get_connection()
                try:
                    encoding = detect_encoding(csv_path)
                    with open(csv_path, "r", encoding=encoding, newline="") as fh:
                        reader = csv.DictReader(fh) if has_headers_var.get() else csv.reader(fh)
                        rows = list(reader)

                    total_rows = len(rows)
                    for idx, row in enumerate(rows, start=1):
                        if task_obj.cancelled():
                            break
                        summary["total"] += 1
                        try:
                            row_data = row
                            tags_value = ""
                            external_id = None

                            if has_headers_var.get() and isinstance(row, dict):
                                lowered = {k.lower(): v for k, v in row.items()}
                                for candidate in ("id", "external_id", "word_id", "uid"):
                                    if candidate in lowered:
                                        external_id = lowered.get(candidate)
                                        break
                                tags_value = lowered.get("tags", "")
                                fields = map_row_to_fields(lowered, mapping_mode)
                            else:
                                tags_value = normalize_tags(row[manual_tags_idx]) if manual_tags_idx is not None and len(row) > manual_tags_idx else ""
                                fields = map_row_to_fields(row, mapping_mode)
                                if manual_id_idx is not None and len(row) > manual_id_idx:
                                    external_id = row[manual_id_idx]
                                elif generate_id_var.get() and row:
                                    external_id = row[0]

                            if not fields:
                                raise ValueError("Пустая строка")

                            srs_defaults = {
                                "state": start_state_var.get(),
                                "due": int(time.time()),
                                "interval": 0,
                                "ease": 250,
                                "reps": 0,
                                "lapses": 0,
                                "step_index": 0,
                                "phase": start_phase_var.get(),
                            }

                            if preserve_srs_var.get() and isinstance(row_data, dict):
                                for key in ("due", "interval", "ease", "reps", "lapses", "step_index", "phase"):
                                    if key in row_data and str(row_data.get(key)).strip():
                                        try:
                                            srs_defaults[key] = int(float(row_data.get(key)))
                                        except Exception:
                                            pass
                                if row_data.get("state"):
                                    srs_defaults["state"] = str(row_data.get("state")).strip()

                            mode = {
                                "skip_existing": skip_existing_var.get() or not upsert_var.get(),
                                "reset_srs": reset_srs_var.get(),
                                "state": srs_defaults["state"],
                            }

                            result = upsert_note_and_cards(
                                conn,
                                deck_id,
                                str(external_id) if external_id is not None else None,
                                fields,
                                tags_value,
                                srs_defaults,
                                mode,
                            )
                            conn.commit()

                            note_id = result.get("note_id")
                            card_ids = result.get("card_ids") or []
                            if attach_images and external_id and card_ids:
                                attach_image_if_exists(conn, note_id, card_ids[0], external_id, images_dir)
                                conn.commit()

                            status = result.get("status", "created")
                            summary[status] = summary.get(status, 0) + 1
                            log_msg(f"[{status}] строка {idx}: {render_card_faces(fields)[0]}")
                        except Exception as exc:  # noqa: BLE001
                            summary["errors"] += 1
                            log_msg(f"[ошибка] строка {idx}: {exc}")
                        task_obj.queue.put(("progress", idx, max(total_rows, 1), "импорт"))

                    task_obj.queue.put(("done", summary))
                except Exception as exc:  # noqa: BLE001
                    task_obj.queue.put(("error", str(exc)))
                finally:
                    conn.close()

            task = start_background_task(worker)
            processing_task["task"] = task
            self.register_bg_handler(task.queue, handle_event)

        def cancel_import():
            if processing_task["task"]:
                processing_task["task"].cancel()
                processing_task["task"] = None
                log_msg("[остановлено] Импорт прерван")

        top_frame = ttk.Frame(win)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(top_frame, text="CSV файл:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top_frame, textvariable=file_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(top_frame, text="Выбрать", command=browse_file).grid(row=0, column=2, padx=5)
        top_frame.columnconfigure(1, weight=1)

        deck_frame = ttk.Frame(win)
        deck_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(deck_frame, text="Целевая колода:").grid(row=0, column=0, sticky="w")
        deck_box = ttk.Combobox(deck_frame, textvariable=deck_var, values=[f"{d['id']}: {d['name']}" for d in decks])
        deck_box.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(deck_frame, text="Создать новую", command=lambda: (self.add_deck_window(), refresh_decks_list())).grid(row=0, column=2, padx=5)
        deck_frame.columnconfigure(1, weight=1)

        options = ttk.LabelFrame(win, text="Настройки импорта")
        options.pack(fill=tk.X, padx=10, pady=5)
        ttk.Checkbutton(options, text="CSV имеет заголовки", variable=has_headers_var, command=on_toggle_headers).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options, text="Обновлять существующие по ID (upsert)", variable=upsert_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(options, text="Пропускать существующие по ID", variable=skip_existing_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(options, text="Если нет ID — генерировать", variable=generate_id_var).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(options, text="Прикреплять картинки из папки images/<id>.png", variable=attach_images_var).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(options, text="Сохранить интервалы из CSV (due/interval/ease)", variable=preserve_srs_var).grid(row=5, column=0, sticky="w")
        ttk.Checkbutton(options, text="Сбросить расписание при обновлении", variable=reset_srs_var).grid(row=6, column=0, sticky="w")

        ttk.Label(options, text="Стартовая фаза:").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Spinbox(options, from_=1, to=10, textvariable=start_phase_var, width=5).grid(row=0, column=2, sticky="w")
        ttk.Label(options, text="Стартовое состояние SRS:").grid(row=1, column=1, sticky="w", padx=5)
        ttk.Combobox(options, values=["new", "learning", "review"], textvariable=start_state_var, width=10).grid(row=1, column=2, sticky="w")
        ttk.Label(options, text="Папка с картинками:").grid(row=2, column=1, sticky="w", padx=5)
        ttk.Entry(options, textvariable=images_dir_var, width=28).grid(row=2, column=2, sticky="we", padx=5)
        ttk.Button(options, text="...", command=browse_images_dir, width=4).grid(row=2, column=3, padx=5)

        manual_frame = ttk.LabelFrame(win, text="Сопоставление колонок (если нет заголовков)")
        manual_frame.pack(fill=tk.X, padx=10, pady=5)
        for i, (label, var) in enumerate(
            [
                ("ID", id_col_var),
                ("Word", word_col_var),
                ("Translation", translation_col_var),
                ("Example", example_col_var),
                ("Notes", notes_col_var),
                ("Tags", tags_col_var),
                ("Front", front_col_var),
                ("Back", back_col_var),
            ]
        ):
            ttk.Label(manual_frame, text=label + ":").grid(row=i // 2, column=(i % 2) * 2, sticky="e", padx=5, pady=2)
            ttk.Spinbox(manual_frame, from_=0, to=20, textvariable=var, width=5).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=5, pady=2)
        if has_headers_var.get():
            manual_frame.grid_remove()

        progress_frame = ttk.Frame(win)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, expand=True)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Импортировать", command=start_import).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=cancel_import).pack(side=tk.LEFT, padx=5)

        log_box = tk.Text(win, height=14, state="disabled")
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # --------- режим ознакомления ---------

    def start_overview_mode(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        deck_id = self.selected_deck_id
        phase_filter = self.selected_phase
        if self.mw_context is not None:
            self.mw_context.state["current_deck_id"] = deck_id
        gui_hooks.deck_will_open(deck_id)
        if self.mw_context is not None:
            self.mw_context.state["current_deck_id"] = deck_id
        gui_hooks.deck_will_open(deck_id)

        def task(progress_cb):
            cards = get_overview_cards(deck_id)
            if phase_filter is not None:
                cards = [c for c in cards if c["leitner_level"] == phase_filter]
            total = len(cards)
            for idx in range(0, total, max(1, max(1, total // 20))):
                progress_cb(min(idx + 1, total), total, f"Подготовка {min(idx + 1, total)}/{total}")
            return cards

        def on_success(cards):
            if not cards:
                phase_text = f" (фаза {phase_filter})" if phase_filter is not None else ""
                messagebox.showinfo("Ознакомление", f"В этой колоде{phase_text} пока нет карточек.")
                return
            OverviewWindow(self, cards)

        def on_error(exc: Exception):
            messagebox.showerror("Ошибка", str(exc))

        self.run_task("Режим ознакомления", "determinate", task, on_success, on_error)

    def add_cards_to_overview_from_repeat(self):
        """Добавить карточки из режима повторения в режим ознакомления"""
        if self.selected_deck_id is None:
            return
        
        # Получаем карточки из режима повторения
        repeat_cards = get_cards_for_repeat(self.selected_deck_id)
        if self.selected_phase is not None:
            repeat_cards = [c for c in repeat_cards if c["leitner_level"] == self.selected_phase]
        
        if not repeat_cards:
            messagebox.showinfo("Нет карточек", "В режиме повторения нет карточек для добавления.")
            return
        
        # Помечаем карточки как добавленные в ознакомление
        for card in repeat_cards:
            mark_card_for_overview(card["id"])
        
        messagebox.showinfo("Успех", f"Добавлено {len(repeat_cards)} карточек в режим ознакомления.")

    # --------- управление словарями ---------

    def open_dictionary_manager_window(self):
        win = tk.Toplevel(self)
        win.title("Управление словарями")
        win.geometry("600x500")
        win.grab_set()

        # Статистика словаря
        stats_frame = ttk.LabelFrame(win, text="Статистика словаря")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        stats = DICTIONARY_MANAGER.get_statistics()
        stats_text = f"""
        Всего слов в словаре: {stats['total_words']:,}
        Загруженные файлы: {len(stats['loaded_files'])}
        Используемая память: {stats['memory_size_mb']:.2f} МБ
        
        Формат: немецкое слово -> русский перевод
        """
        
        stats_label = ttk.Label(stats_frame, text=stats_text, justify=tk.LEFT)
        stats_label.pack(padx=10, pady=10)
        
        # Загруженные файлы
        if stats['loaded_files']:
            files_frame = ttk.LabelFrame(win, text="Загруженные файлы словарей")
            files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            listbox = tk.Listbox(files_frame)
            listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            for file in stats['loaded_files']:
                listbox.insert(tk.END, file)

        # Управление словарями
        mgmt_frame = ttk.LabelFrame(win, text="Управление словарями")
        mgmt_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_frame = ttk.Frame(mgmt_frame)
        btn_frame.pack(padx=10, pady=10)
        
        def load_dictionary():
            filetypes = [
                ("CSV файлы", "*.csv"),
                ("JSON файлы", "*.json"),
                ("Сжатые файлы", "*.gz *.json.gz"),
                ("Все файлы", "*.*"),
            ]
            filename = filedialog.askopenfilename(
                title="Выберите файл словаря",
                filetypes=filetypes
            )
            if filename:
                try:
                    if filename.endswith('.csv'):
                        count = DICTIONARY_MANAGER.load_from_csv(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")
                    elif filename.endswith('.json'):
                        count = DICTIONARY_MANAGER.load_from_json(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")
                    elif filename.endswith(('.gz', '.json.gz')):
                        count = DICTIONARY_MANAGER.load_from_compressed(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")
                    
                    # Сохраняем путь в настройках
                    if filename not in TRANSLATION_SETTINGS.dictionary_paths:
                        TRANSLATION_SETTINGS.dictionary_paths.append(filename)
                        TRANSLATION_SETTINGS.save()
                    
                    # Обновляем окно
                    win.destroy()
                    self.open_dictionary_manager_window()
                    
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить словарь:\n{e}")
        
        def export_dictionary():
            filename = filedialog.asksaveasfilename(
                title="Экспорт словаря",
                defaultextension=".csv",
                filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
            )
            if filename:
                try:
                    gui_hooks.export_will_start(filename)
                    DICTIONARY_MANAGER.export_to_csv(filename)
                    messagebox.showinfo("Успех", f"Словарь экспортирован в {filename}")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось экспортировать словарь:\n{e}")
        
        def save_compressed():
            filename = filedialog.asksaveasfilename(
                title="Сохранить сжатый словарь",
                defaultextension=".json.gz",
                filetypes=[("Сжатые JSON файлы", "*.json.gz"), ("Все файлы", "*.*")]
            )
            if filename:
                try:
                    DICTIONARY_MANAGER.save_compressed_dictionary(filename)
                    messagebox.showinfo("Успех", f"Словарь сохранен в {filename}")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось сохранить словарь:\n{e}")
        
        def search_word():
            search_win = tk.Toplevel(win)
            search_win.title("Поиск слова")
            search_win.geometry("400x300")
            search_win.grab_set()
            
            ttk.Label(search_win, text="Введите слово для поиска:").pack(padx=10, pady=(10, 0))
            
            entry = ttk.Entry(search_win)
            entry.pack(fill=tk.X, padx=10, pady=5)
            
            results_text = tk.Text(search_win, height=10)
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            def perform_search():
                word = entry.get().strip()
                if word:
                    results = DICTIONARY_MANAGER.search_words(word, limit=20)
                    results_text.delete(1.0, tk.END)
                    if results:
                        for german, russian in results:
                            results_text.insert(tk.END, f"{german} -> {russian}\n")
                    else:
                        results_text.insert(tk.END, "Совпадений не найдено")
            
            ttk.Button(search_win, text="Поиск", command=perform_search).pack(pady=10)
        
        ttk.Button(btn_frame, text="Загрузить словарь", command=load_dictionary).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Экспорт в CSV", command=export_dictionary).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Сохранить сжатый", command=save_compressed).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(btn_frame, text="Поиск слова", command=search_word).grid(row=1, column=0, padx=5, pady=5, columnspan=3)

    def show_dictionary_stats_window(self):
        win = tk.Toplevel(self)
        win.title("Статистика словаря")
        win.geometry("400x300")
        win.grab_set()
        
        stats = DICTIONARY_MANAGER.get_statistics()
        
        stats_text = f"""
        📊 СТАТИСТИКА СЛОВАРЯ
        
        Всего слов: {stats['total_words']:,}
        
        Загруженные файлы: {len(stats['loaded_files'])}
        
        Используемая память: {stats['memory_size_mb']:.2f} МБ
        
        Покрытие слов:
        - 1,000 самых частотных слов: {min(stats['total_words'], 1000):,}
        - 5,000 самых частотных слов: {min(stats['total_words'], 5000):,}
        - 10,000 самых частотных слов: {min(stats['total_words'], 10000):,}
        - 50,000 самых частотных слов: {min(stats['total_words'], 50000):,}
        - 100,000 самых частотных слов: {min(stats['total_words'], 100000):,}
        """
        
        label = ttk.Label(win, text=stats_text, justify=tk.LEFT)
        label.pack(padx=20, pady=20)
        
        if stats['total_words'] < 50000:
            ttk.Label(win, 
                     text="⚠️ Рекомендуется загрузить больший словарь для лучшего покрытия",
                     foreground="orange").pack(pady=10)

    # --------- настройки перевода ---------

    def open_translation_settings_window(self):
        win = tk.Toplevel(self)
        win.title("Настройки перевода")
        win.geometry("400x350")
        win.grab_set()

        # Встроенный словарь
        use_dict_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.use_embedded_dict)
        cb_dict = ttk.Checkbutton(
            win, 
            text="Использовать встроенный словарь",
            variable=use_dict_var
        )
        cb_dict.pack(anchor="w", padx=20, pady=(20, 10))

        # OpenAI
        use_openai_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.use_openai)
        cb_openai = ttk.Checkbutton(
            win,
            text="Использовать OpenAI для перевода (если есть ключ)",
            variable=use_openai_var
        )
        cb_openai.pack(anchor="w", padx=20, pady=10)

        # Показывать переводы на лицевой стороне
        show_trans_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.show_translations)
        cb_show = ttk.Checkbutton(
            win,
            text="Показывать переводы над словами в режиме повторения (лицевая сторона)",
            variable=show_trans_var
        )
        cb_show.pack(anchor="w", padx=20, pady=10)

        # Показывать перевод на задней стороне
        show_back_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.show_back_translation)
        cb_back = ttk.Checkbutton(
            win,
            text="Всегда показывать русский перевод на задней стороне карточки",
            variable=show_back_var
        )
        cb_back.pack(anchor="w", padx=20, pady=10)

        # Приоритет
        ttk.Label(win, text="Приоритет перевода:").pack(anchor="w", padx=20, pady=(10, 5))
        priority_var = tk.StringVar(value="dictionary")
        ttk.Radiobutton(
            win,
            text="Сначала словарь, потом OpenAI",
            variable=priority_var,
            value="dictionary"
        ).pack(anchor="w", padx=30)
        ttk.Radiobutton(
            win,
            text="Сначала OpenAI, потом словарь",
            variable=priority_var,
            value="openai"
        ).pack(anchor="w", padx=30)

        def save_settings():
            TRANSLATION_SETTINGS.use_embedded_dict = use_dict_var.get()
            TRANSLATION_SETTINGS.use_openai = use_openai_var.get()
            TRANSLATION_SETTINGS.show_translations = show_trans_var.get()
            TRANSLATION_SETTINGS.show_back_translation = show_back_var.get()
            TRANSLATION_SETTINGS.save()
            messagebox.showinfo("Сохранено", "Настройки перевода сохранены.")
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        ttk.Button(btn_frame, text="Сохранить", command=save_settings).pack(side=tk.RIGHT)

    def open_generation_templates_window(
        self,
        selected_template_id: int | None = None,
        on_close=None,
    ) -> None:
        win = tk.Toplevel(self)
        win.title("Генерация шаблонов")
        win.geometry("960x640")
        win.grab_set()
        apply_dark_theme_to_window(win, self.palette)

        ensure_default_generation_template(self.selected_deck_id)
        ensure_video_text_generation_template()

        main_frame = ttk.Frame(win, style="Surface.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left_frame, text="Шаблоны").pack(anchor="w")
        listbox = tk.Listbox(left_frame, width=28, height=18)
        listbox.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        list_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.configure(yscrollcommand=list_scroll.set)

        name_var = tk.StringVar()
        question_var = tk.StringVar()
        template_state = {"id": None, "rows": []}

        ttk.Label(right_frame, text="Название:").pack(anchor="w")
        entry_name = ttk.Entry(right_frame, textvariable=name_var)
        entry_name.pack(fill=tk.X, pady=(0, 6))
        create_context_menu(entry_name)

        ttk.Label(right_frame, text="Вопрос на лицевой стороне:").pack(anchor="w")
        entry_question = ttk.Entry(right_frame, textvariable=question_var)
        entry_question.pack(fill=tk.X, pady=(0, 6))
        create_context_menu(entry_question)

        ttk.Label(right_frame, text="Шаблон FRONT:").pack(anchor="w")
        text_front = tk.Text(right_frame, height=4)
        style_text_widget(text_front, self.palette)
        text_front.pack(fill=tk.BOTH, expand=False, pady=(0, 6))
        create_context_menu(text_front)

        ttk.Label(right_frame, text="Шаблон BACK:").pack(anchor="w")
        text_back = tk.Text(right_frame, height=4)
        style_text_widget(text_back, self.palette)
        text_back.pack(fill=tk.BOTH, expand=False, pady=(0, 6))
        create_context_menu(text_back)

        active_template_widget = {"widget": text_front}
        text_front.bind("<FocusIn>", lambda _e: active_template_widget.update(widget=text_front))
        text_back.bind("<FocusIn>", lambda _e: active_template_widget.update(widget=text_back))

        variables_frame = ttk.Frame(right_frame)
        variables_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(variables_frame, text="Переменные:").pack(anchor="w")
        available_vars = [
            "{question}",
            "{front_html}",
            "{back_html}",
            "{image}",
            "{front_image}",
            "{back_image}",
            "{video}",
            "{front_video}",
            "{back_video}",
            "{audio}",
            "{front_audio}",
            "{back_audio}",
            "{translation}",
            "{sentence_with_gap}",
            "{word}",
            "{ipa}",
            "{gender}",
            "{plural}",
            "{sentence}",
        ]
        var_listbox = tk.Listbox(variables_frame, height=7)
        for item in available_vars:
            var_listbox.insert(tk.END, item)
        var_listbox.pack(fill=tk.X, padx=(0, 6))

        def _insert_variable() -> None:
            selection = var_listbox.curselection()
            if not selection:
                return
            variable = var_listbox.get(selection[0])
            target = active_template_widget.get("widget") or text_front
            target.insert(tk.INSERT, variable)
            target.focus_set()

        var_listbox.bind("<Double-Button-1>", lambda _e: _insert_variable())
        ttk.Button(variables_frame, text="Вставить переменную", command=_insert_variable).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            variables_frame,
            text="Медиа-переменные вставляют готовые HTML-теги <img>/<video>/<audio> и работают в preview/review.",
            style="Muted.TLabel",
            wraplength=520,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(right_frame, text="Тестовый текст:").pack(anchor="w")
        test_text = tk.Text(right_frame, height=3)
        style_text_widget(test_text, self.palette)
        test_text.pack(fill=tk.BOTH, expand=False, pady=(0, 4))
        test_text.insert("1.0", "Das ist ein Haus.")
        create_context_menu(test_text)

        mask_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            right_frame,
            text="Маскировать по повторенным словам",
            variable=mask_var,
        ).pack(anchor="w", pady=(0, 8))

        def load_template_row(row: sqlite3.Row | None) -> None:
            template_state["id"] = row["id"] if row else None
            name_var.set(row["name"] if row else "")
            question_var.set(row["question"] if row else "")
            text_front.delete("1.0", tk.END)
            text_back.delete("1.0", tk.END)
            if row:
                text_front.insert("1.0", row["front_template"])
                text_back.insert("1.0", row["back_template"])

        def refresh_list(select_id: int | None = None) -> None:
            rows = list_generation_templates()
            template_state["rows"] = rows
            listbox.delete(0, tk.END)
            selected_index = None
            for idx, row in enumerate(rows):
                listbox.insert(tk.END, row["name"])
                if select_id and row["id"] == select_id:
                    selected_index = idx
            if selected_index is None and rows:
                selected_index = 0
            if selected_index is not None:
                listbox.selection_set(selected_index)
                listbox.activate(selected_index)
                load_template_row(rows[selected_index])
            else:
                load_template_row(None)

        def on_select(_event=None) -> None:
            selection = listbox.curselection()
            if not selection:
                return
            idx = selection[0]
            rows = template_state["rows"]
            if idx < len(rows):
                load_template_row(rows[idx])

        def collect_fields() -> tuple[str, str, str, str]:
            name = name_var.get().strip()
            question = question_var.get().strip()
            front = text_front.get("1.0", tk.END).strip()
            back = text_back.get("1.0", tk.END).strip()
            return name, question, front, back

        def ensure_valid_fields(name: str, front: str, back: str) -> bool:
            if not name:
                messagebox.showerror("Ошибка", "Название шаблона не может быть пустым.")
                return False
            if not front or not back:
                messagebox.showerror("Ошибка", "Шаблон FRONT/BACK не может быть пустым.")
                return False
            return True

        def save_template() -> None:
            name, question, front, back = collect_fields()
            if not ensure_valid_fields(name, front, back):
                return
            current_id = template_state["id"]
            existing = {row["name"]: row["id"] for row in template_state["rows"]}
            if current_id is not None:
                if name in existing and existing[name] != current_id:
                    messagebox.showerror("Ошибка", "Шаблон с таким названием уже существует.")
                    return
                update_generation_template(current_id, name, question, front, back)
                refresh_list(select_id=current_id)
            else:
                template_id = upsert_generation_template(name, question, front, back)
                refresh_list(select_id=template_id)

        def save_as_new() -> None:
            name, question, front, back = collect_fields()
            base_name = name or "Новый шаблон"
            existing = {row["name"] for row in template_state["rows"]}
            new_name = base_name
            idx = 2
            while new_name in existing:
                new_name = f"{base_name} ({idx})"
                idx += 1
            if not ensure_valid_fields(new_name, front, back):
                return
            name_var.set(new_name)
            template_id = upsert_generation_template(new_name, question, front, back)
            refresh_list(select_id=template_id)

        def delete_template() -> None:
            current_id = template_state["id"]
            if current_id is None:
                messagebox.showinfo("Удаление", "Выберите шаблон для удаления.")
                return
            if not messagebox.askyesno("Удаление", "Удалить выбранный шаблон?"):
                return
            delete_generation_template(current_id)
            template_state["id"] = None
            refresh_list()

        def preview_template() -> None:
            name, question, front_t, back_t = collect_fields()
            if not name:
                name = "Предпросмотр"
            test_value = test_text.get("1.0", tk.END).strip()
            if not test_value:
                test_value = "Das ist ein Haus."
            if mask_var.get():
                masked_text, hidden_words = mask_text_by_repeated_words(test_value)
                preview_ctx = {
                    "sentence": test_value,
                    "sentence_with_gap": masked_text,
                    "word": hidden_words[0] if hidden_words else "—",
                    "translation": ", ".join(hidden_words) if hidden_words else "—",
                    "ipa": "",
                    "gender": "",
                    "plural": "",
                }
            else:
                words = extract_words_from_text(test_value)
                preview_ctx = {
                    "sentence": test_value,
                    "sentence_with_gap": test_value,
                    "word": words[0] if words else "",
                    "translation": "дом",
                    "ipa": "",
                    "gender": "",
                    "plural": "",
                }
            preview_ctx.update(
                {
                    "front_html": html.escape(test_value).replace("\n", "<br>"),
                    "back_html": html.escape(test_value).replace("\n", "<br>"),
                    "image": "",
                    "front_image": "",
                    "back_image": "",
                    "video": "",
                    "front_video": "",
                    "back_video": "",
                    "audio": "",
                    "front_audio": "",
                    "back_audio": "",
                }
            )
            template = {
                "id": template_state["id"],
                "name": name,
                "question": question,
                "front_template": front_t,
                "back_template": back_t,
            }
            front, back = _render_generation_template_faces(template, preview_ctx)

            preview_win = tk.Toplevel(win)
            preview_win.title("Предпросмотр шаблона")
            preview_win.geometry("700x420")
            preview_win.grab_set()
            apply_dark_theme_to_window(preview_win, self.palette)

            preview_frame = ttk.Frame(preview_win, style="Surface.TFrame")
            preview_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

            ttk.Label(preview_frame, text="FRONT").pack(anchor="w")
            front_box = tk.Frame(preview_frame)
            front_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
            card_bg, card_text, _ = get_card_surface_colors(preview_frame)
            front_view = CardHtmlView(front_box)
            front_view.set_html(front, card_bg=card_bg, card_text_color=card_text)

            ttk.Label(preview_frame, text="BACK").pack(anchor="w")
            back_box = tk.Frame(preview_frame)
            back_box.pack(fill=tk.BOTH, expand=True)
            back_view = CardHtmlView(back_box)
            back_view.set_html(back, card_bg=card_bg, card_text_color=card_text)

        btn_row = ttk.Frame(right_frame)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_row, text="Предпросмотр", command=preview_template).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Сохранить", command=save_template).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Сохранить как новый", command=save_as_new).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Удалить", command=delete_template).pack(side=tk.RIGHT)

        listbox.bind("<<ListboxSelect>>", on_select)

        refresh_list(select_id=selected_template_id)

        def _on_close() -> None:
            if callable(on_close):
                on_close()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

    # --------- настройки ---------

    def open_settings_window(self):
        global OPENAI_API_KEY

        win = tk.Toplevel(self)
        win.title("Настройки OpenAI")
        win.geometry("450x190")
        win.grab_set()

        ttk.Label(
            win,
            text="API ключ OpenAI (формат sk-... / sk-proj-...; хранится только в памяти):"
        ).pack(anchor="w", padx=10, pady=(10, 0))

        entry_key = ttk.Entry(win, show="*")
        entry_key.pack(fill=tk.X, padx=10, pady=5)
        create_context_menu(entry_key)  # Добавляем контекстное меню

        if OPENAI_API_KEY:
            entry_key.insert(0, OPENAI_API_KEY)

        def paste_from_clipboard():
            try:
                text = win.clipboard_get()
            except tk.TclError:
                text = ""
            entry_key.delete(0, tk.END)
            entry_key.insert(0, text.strip())

        ttk.Button(win, text="Вставить из буфера обмена",
                   command=paste_from_clipboard).pack(anchor="e", padx=10, pady=(0, 5))

        def save_key():
            global OPENAI_API_KEY
            key = entry_key.get().strip()
            OPENAI_API_KEY = key or None
            messagebox.showinfo("Сохранено", "Ключ сохранён в памяти приложения.")
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="OK", command=save_key).pack(side=tk.RIGHT)

    def open_audio_device_window(self):
        if not SR_AVAILABLE:
            messagebox.showerror(
                "Речь недоступна",
                "Чтобы выбрать микрофон, установите SpeechRecognition и PyAudio:\n"
                "pip install SpeechRecognition pyaudio"
            )
            return

        try:
            devices = sr.Microphone.list_microphone_names()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить список устройств:\n{e}")
            return

        win = tk.Toplevel(self)
        win.title("Выбор аудиоустройства для цифрового слуха")
        win.geometry("500x300")
        win.grab_set()

        ttk.Label(
            win,
            text="Выбери устройство записи, которое будет слушать звук\n"
                 "в режиме «Генерация через цифрового слуха».\n\n"
                 "Для VB-Audio Cable обычно это CABLE Output."
        ).pack(anchor="w", padx=10, pady=(10, 0))

        listbox = tk.Listbox(win, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        selected_initial = 0
        for i, name in enumerate(devices):
            listbox.insert(tk.END, f"{i}: {name}")
            if self.microphone_index is not None and i == self.microphone_index:
                selected_initial = i

        if devices:
            listbox.selection_set(selected_initial)
            listbox.see(selected_initial)

        def save_device():
            sel = listbox.curselection()
            if not sel:
                self.microphone_index = None
            else:
                idx_line = listbox.get(sel[0])
                idx_str = idx_line.split(":", 1)[0]
                try:
                    self.microphone_index = int(idx_str)
                except ValueError:
                    self.microphone_index = None
            messagebox.showinfo(
                "Сохранено",
                f"Устройство записи для цифрового слуха установлено: "
                f"{self.microphone_index if self.microphone_index is not None else 'по умолчанию'}"
            )
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="OK", command=save_device).pack(side=tk.RIGHT)

    # --------- статистика ---------

    def show_statistics_window(self):
        if self.main_notebook and self.stats_tab:
            self.main_notebook.select(self.stats_tab)
            self.refresh_statistics_tab()
            return
        win = tk.Toplevel(self)
        win.title("Статистика колод")
        win.geometry("1200x900")
        win.grab_set()

        # Получаем все колоды
        decks = list_decks()
        if not decks:
            messagebox.showinfo("Нет колод", "Сначала создайте колоду.")
            win.destroy()
            return

        # Создаем вкладки для каждой колоды
        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for deck in decks:
            deck_frame = ttk.Frame(notebook)
            notebook.add(deck_frame, text=deck['name'])

            # Содержимое вкладки колоды
            self.create_deck_statistics_tab(deck_frame, deck['id'], deck['name'])

        # Кнопка обновить все
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        def update_all_dates():
            for i in range(notebook.index("end")):
                tab = notebook.nametowidget(notebook.tabs()[i])
                for child in tab.winfo_children():
                    if hasattr(child, 'update_charts'):
                        child.update_charts()
            messagebox.showinfo("Обновлено", "Все графики обновлены")

        ttk.Button(btn_frame, text="Обновить все графики", command=update_all_dates).pack(side=tk.RIGHT)

    def create_deck_statistics_tab(self, parent, deck_id, deck_name):
        """Создать вкладку статистики для конкретной колоды"""
        # Основной контейнер
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Настройки для диаграмм
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки диаграмм")
        settings_frame.pack(fill=tk.X, pady=5)

        stored_settings = load_stats_settings(deck_id)

        ttk.Label(settings_frame, text="Режим оси X:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        x_mode_var = tk.StringVar(value=stored_settings.x_mode)
        x_mode_combo = ttk.Combobox(
            settings_frame,
            values=["range", "month_days", "custom_dates"],
            textvariable=x_mode_var,
            state="readonly",
            width=15,
        )
        x_mode_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Дней (если диапазон пустой):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        days_var = tk.IntVar(value=30)
        days_spin = ttk.Spinbox(settings_frame, from_=7, to=365, textvariable=days_var, width=8)
        days_spin.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Дата от (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        date_from_var = tk.StringVar(value=stored_settings.date_from or "")
        ttk.Entry(settings_frame, textvariable=date_from_var, width=16).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Дата до (YYYY-MM-DD):").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        date_to_var = tk.StringVar(value=stored_settings.date_to or "")
        ttk.Entry(settings_frame, textvariable=date_to_var, width=16).grid(row=1, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Кастомные даты (по строкам):").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        custom_dates_text = tk.Text(settings_frame, height=3, width=25)
        custom_dates_text.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky="we")
        if stored_settings.custom_dates:
            custom_dates_text.insert("1.0", "\n".join(stored_settings.custom_dates))

        ttk.Label(settings_frame, text="Максимум Y (0 - авто):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        max_y_var = tk.IntVar(value=stored_settings.y_max)
        max_y_spin = ttk.Spinbox(settings_frame, from_=0, to=10000, textvariable=max_y_var, width=10)
        max_y_spin.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Норма (горизонталь):").grid(row=3, column=2, padx=5, pady=5, sticky="w")
        norm_var = tk.IntVar(value=stored_settings.norm_value)
        norm_spin = ttk.Spinbox(settings_frame, from_=0, to=20000, textvariable=norm_var, width=10)
        norm_spin.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Тип графика:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        chart_type_var = tk.StringVar(value=stored_settings.chart_type)
        chart_combo = ttk.Combobox(
            settings_frame, values=["bar", "line"], textvariable=chart_type_var, state="readonly", width=10
        )
        chart_combo.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        show_grid_var = tk.BooleanVar(value=bool(stored_settings.show_grid))
        ttk.Checkbutton(settings_frame, text="Показывать сетку", variable=show_grid_var).grid(
            row=4, column=2, padx=5, pady=5, sticky="w"
        )

        # Фрейм для графиков
        charts_container = ttk.Frame(main_frame, style="Surface.TFrame")
        charts_container.pack(fill=tk.BOTH, expand=True, pady=5)
        charts_canvas = tk.Canvas(
            charts_container,
            bg=self.palette.get("panel", "#111522") if hasattr(self, "palette") else "white",
            highlightthickness=0,
            borderwidth=0,
        )
        charts_scrollbar = ttk.Scrollbar(charts_container, orient="vertical", command=charts_canvas.yview)
        charts_frame = ttk.Frame(charts_canvas, style="Surface.TFrame")
        charts_window = charts_canvas.create_window((0, 0), window=charts_frame, anchor="nw")
        charts_frame.bind(
            "<Configure>",
            lambda e: charts_canvas.configure(scrollregion=charts_canvas.bbox("all")),
        )
        charts_canvas.bind(
            "<Configure>",
            lambda e: charts_canvas.itemconfig(charts_window, width=e.width),
        )
        charts_canvas.configure(yscrollcommand=charts_scrollbar.set)
        charts_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        charts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _safe_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def collect_settings() -> StatsSettings:
            settings = StatsSettings(
                deck_id=deck_id,
                x_mode=x_mode_var.get() or "range",
                date_from=date_from_var.get().strip() or None,
                date_to=date_to_var.get().strip() or None,
                y_max=_safe_int(max_y_var.get(), 0),
                norm_value=_safe_int(norm_var.get(), 0),
                chart_type=chart_type_var.get() or "bar",
                show_grid=1 if show_grid_var.get() else 0,
            )
            custom_lines = custom_dates_text.get("1.0", "end").splitlines()
            settings.update_custom_dates(custom_lines)
            return settings

        def parse_date_safe(value: str | None):
            if not value:
                return None
            try:
                return datetime.strptime(value.strip(), "%Y-%m-%d").date()
            except ValueError:
                return None

        def build_date_list(settings: StatsSettings) -> list[date]:
            mode = settings.x_mode or "range"
            if mode == "month_days":
                today = date.today()
                last_day = calendar.monthrange(today.year, today.month)[1]
                return [date(today.year, today.month, d) for d in range(1, last_day + 1)]
            if mode == "custom_dates":
                dates: list[date] = []
                for line in custom_dates_text.get("1.0", "end").splitlines():
                    parsed = parse_date_safe(line)
                    if parsed:
                        dates.append(parsed)
                return sorted(set(dates))

            start = parse_date_safe(settings.date_from)
            end = parse_date_safe(settings.date_to)
            if not end:
                end = date.today()
            if not start:
                start = end - timedelta(days=max(1, days_var.get()) - 1)
            if start > end:
                start, end = end, start

            dates: list[date] = []
            current = start
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)
            return dates

        def update_charts():
            self.show_loading("Загрузка", determinate=False)
            if not MATPLOTLIB_AVAILABLE:
                for widget in charts_frame.winfo_children():
                    widget.destroy()
                ttk.Label(charts_frame,
                         text="Графики недоступны: установите Matplotlib (pip install matplotlib) и перезапустите приложение.",
                         foreground="red").pack(pady=20)
                self.hide_loading()
                return

            current_settings = collect_settings()
            save_stats_settings(current_settings)
            date_list = build_date_list(current_settings)

            def render_charts(data):
                for widget in charts_frame.winfo_children():
                    widget.destroy()

                if not data["dates"]:
                    ttk.Label(charts_frame, text="Нет данных для отображения").pack(pady=20)
                    self.hide_loading()
                    return

                from matplotlib import dates as mdates

                grid_enabled = bool(show_grid_var.get())
                y_limit = current_settings.y_max if current_settings.y_max and current_settings.y_max > 0 else None

                fig = Figure(figsize=(14, 16), dpi=100)
                fig.subplots_adjust(hspace=0.6)
                ax1 = fig.add_subplot(411)
                ax2 = fig.add_subplot(412, sharex=ax1)
                ax3 = fig.add_subplot(413, sharex=ax1)
                ax4 = fig.add_subplot(414)

                x_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in data["dates"]]
                x_numeric = mdates.date2num(x_dates)

                def apply_common_axis(ax):
                    locator = mdates.AutoDateLocator()
                    formatter = mdates.DateFormatter("%d.%m")
                    ax.xaxis.set_major_locator(locator)
                    ax.xaxis.set_major_formatter(formatter)
                    if current_settings.x_mode == "custom_dates":
                        ax.set_xticks(x_numeric)
                    if y_limit:
                        ax.set_ylim(0, y_limit)
                    if grid_enabled:
                        ax.grid(True, alpha=0.3)

                reviewed = data["reviewed"]
                if current_settings.chart_type == "line":
                    ax1.plot(x_dates, reviewed, marker="o", linewidth=2, color="blue", label="Просмотрено карточек")
                else:
                    ax1.bar(x_numeric, reviewed, width=0.6, color="blue", alpha=0.6, label="Просмотрено карточек")

                if current_settings.norm_value:
                    ax1.axhline(current_settings.norm_value, linestyle="--", color="gray", linewidth=1, label="Норма")

                ax1.set_ylabel('Количество карточек')
                ax1.set_title(f'Общая статистика повторений - {deck_name}')
                apply_common_axis(ax1)
                ax1.legend(loc='upper left')

                remembered = data["remembered"]
                forgotten = data["forgotten"]
                if current_settings.chart_type == "line":
                    ax2.plot(x_dates, remembered, marker="o", color='green', label='Помню')
                    ax2.plot(x_dates, forgotten, marker="o", color='red', label='Забыл')
                else:
                    width = 0.35
                    bars1 = ax2.bar(x_numeric - width/2, remembered, width, label='Помню', color='green', alpha=0.7)
                    bars2 = ax2.bar(x_numeric + width/2, forgotten, width, label='Забыл', color='red', alpha=0.7)
                    for bar in list(bars1) + list(bars2):
                        height = bar.get_height()
                        if height > 0:
                            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=8)

                if current_settings.norm_value:
                    ax2.axhline(current_settings.norm_value, linestyle="--", color="gray", linewidth=1)

                ax2.set_ylabel('Количество карточек')
                ax2.set_title('Сравнение запомненных и забытых карточек')
                apply_common_axis(ax2)
                ax2.legend(loc='upper left')

                overview = data["overview"]
                if current_settings.chart_type == "line":
                    ax3.plot(x_dates, overview, marker="o", color='orange', label='Ознакомлено карточек')
                else:
                    bars3 = ax3.bar(x_numeric, overview, width=0.6, label='Ознакомлено карточек', color='orange', alpha=0.7)
                    for bar in bars3:
                        height = bar.get_height()
                        if height > 0:
                            ax3.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=8)

                if current_settings.norm_value:
                    ax3.axhline(current_settings.norm_value, linestyle="--", color="gray", linewidth=1)

                ax3.set_ylabel('Количество карточек')
                ax3.set_title('Ознакомление с карточками')
                ax3.set_xlabel('Дата')
                apply_common_axis(ax3)
                ax3.legend(loc='upper left')

                fig.autofmt_xdate(rotation=30)

                stats = get_deck_stats(deck_id)
                phases = list(range(1, 11))
                phase_counts = [stats["phase_stats"].get(phase, 0) for phase in phases]
                
                bars4 = ax4.bar(phases, phase_counts, width=0.6, color='purple', alpha=0.7)
                
                ax4.set_xlabel('Фаза Лейтнера')
                ax4.set_ylabel('Количество карточек')
                ax4.set_title('Распределение карточек по фазам')
                ax4.set_xticks(phases)
                ax4.set_xticklabels([f'Фаза {p}' for p in phases], rotation=45, fontsize=8)
                if grid_enabled:
                    ax4.grid(True, alpha=0.3)

                for bar in bars4:
                    height = bar.get_height()
                    if height > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)

                fig.tight_layout(pad=2.4)
                
                canvas = FigureCanvasTkAgg(fig, charts_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=18)
                charts_frame.update_charts = update_charts
                self.hide_loading()

            def fetch_data():
                try:
                    data = get_statistics_for_dates(deck_id, date_list)
                except Exception as exc:
                    self.after(0, lambda: (self.hide_loading(), messagebox.showerror("Статистика", str(exc))))
                    return
                self.after(0, lambda: render_charts(data))

            threading.Thread(target=fetch_data, daemon=True).start()

        btn_update = ttk.Button(settings_frame, text="Сохранить и обновить", command=update_charts)
        btn_update.grid(row=0, column=4, padx=10, pady=5)
        main_frame.update_charts = update_charts

        # Инициализируем диаграммы
        update_charts()

    # --------- главное окно ---------

    def _mk_wrapped_tk_button(self, parent, text: str, command, style_name: str, wraplength: int = 180):
        """Create a dark-themed tk.Button that supports wraplength (ttk.Button does not)."""
        p = getattr(self, "palette", None) or {}
        bg_main = p.get("bg", "#0B0D12")
        panel = p.get("panel", "#111522")
        panel2 = p.get("panel2", panel)
        border = p.get("border", "#242A3A")
        text_color = p.get("text", "#E8ECF4")
        muted = p.get("muted", "#A7B0C0")
        accent = p.get("accent", "#3B82F6")
        accent_hover = p.get("accent_hover", accent)
        accent_active = p.get("accent_active", accent_hover)

        # Map ttk styles to tk colors
        if style_name == "Primary.TButton":
            bg = accent
            fg = "#FFFFFF"
            bg_hover = accent_hover
            bg_active = accent_active
        elif style_name == "Ghost.TButton":
            bg = bg_main
            fg = text_color
            bg_hover = panel2
            bg_active = panel2
        elif style_name == "Secondary.TButton":
            bg = panel
            fg = text_color
            bg_hover = panel2
            bg_active = panel2
        else:
            # Secondary/normal
            bg = panel
            fg = text_color
            bg_hover = panel2
            bg_active = panel2

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg_active,
            activeforeground=fg,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=14,
            pady=10,
            wraplength=wraplength,
            justify=tk.CENTER,
            font=("Segoe UI", 12),
            cursor="hand2",
        )

        def _enter(_e):
            btn.configure(bg=bg_hover)

        def _leave(_e):
            btn.configure(bg=bg)

        btn.bind("<Enter>", _enter)
        btn.bind("<Leave>", _leave)
        return btn

    def create_widgets(self):
        menu_divider = tk.Frame(self, height=1, bg=self.palette["border"])
        menu_divider.pack(fill=tk.X)

        header_pad = (8, 8)
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(fill=tk.X, padx=16, pady=(12, 8))
        menu_bg = self.palette.get("panel", "#111827")
        menu_hover = self.palette.get("panel2", "#1F2937")
        menu_fg = self.palette.get("text", "#E5E7EB")
        menu_border = self.palette.get("border", "#1F2937")
        hamburger_btn = tk.Button(
            header,
            text="☰",
            command=self.toggle_generation_drawer,
            bg=menu_bg,
            fg=menu_fg,
            activebackground=menu_hover,
            activeforeground=menu_fg,
            relief="flat",
            bd=0,
            padx=8,
            pady=6,
            width=2,
            height=1,
            highlightthickness=1,
            highlightbackground=menu_border,
            highlightcolor=menu_border,
            cursor="hand2",
            font=("Segoe UI", 12, "bold"),
        )
        hamburger_btn.pack(side=tk.LEFT, padx=(6, 4), pady=header_pad)
        self.hamburger_button = hamburger_btn

        def _menu_enter(_e):
            hamburger_btn.configure(bg=menu_hover)

        def _menu_leave(_e):
            hamburger_btn.configure(bg=menu_bg)

        hamburger_btn.bind("<Enter>", _menu_enter)
        hamburger_btn.bind("<Leave>", _menu_leave)
        logo_wrap = ttk.Frame(header, width=48, height=48)
        logo_wrap.pack_propagate(False)
        logo_wrap.pack(side=tk.LEFT, padx=(10, 8), pady=header_pad)
        if self._logo_small is not None:
            logo_lbl = tk.Label(logo_wrap, image=self._logo_small, bd=0, highlightthickness=0)
            logo_lbl.pack(fill="both", expand=True)
        title_lbl = ttk.Label(header, text="X-FLASH", style="Title.TLabel")
        title_lbl.pack(side=tk.LEFT, padx=(0, 12), pady=header_pad)

        quick_actions = ttk.Frame(header, style="Header.TFrame")
        quick_actions.pack(side=tk.RIGHT, pady=header_pad)
        ttk.Button(quick_actions, text="Импорт .apkg", style="Secondary.TButton", command=self.open_apkg_import_window).pack(side=tk.LEFT, padx=6)
        ttk.Button(quick_actions, text="Импорт CSV", style="Secondary.TButton", command=self.open_csv_import_window).pack(side=tk.LEFT, padx=6)
        ttk.Button(quick_actions, text="Новая колода", style="Primary.TButton", command=self.add_deck_window).pack(side=tk.LEFT, padx=(10, 0))

        account_bar = ttk.Frame(header, style="Header.TFrame")
        account_bar.pack(side=tk.RIGHT, padx=(0, 12), pady=header_pad)

        balance_frame = ttk.Frame(account_bar, style="Header.TFrame")
        balance_frame.pack(side=tk.RIGHT, padx=(0, 8))
        balance_icon = self._load_credit_icon(size=32)
        if balance_icon:
            self.credit_icon_small = balance_icon
            self.credit_icon_image = balance_icon
            ttk.Label(balance_frame, image=balance_icon, style="HeaderSub.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        else:
            ttk.Label(balance_frame, text="ⓘ", style="HeaderSub.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        lbl_balance = ttk.Label(
            balance_frame,
            textvariable=self.balance_var,
            style="HeaderSub.TLabel",
            font=("Segoe UI", 16, "bold"),
        )
        lbl_balance.pack(side=tk.LEFT)
        self.balance_labels.append(self.balance_var)
        self.balance_widgets.append(lbl_balance)

        ttk.Button(
            account_bar,
            text="Личный кабинет",
            style="Ghost.TButton",
            command=self.open_personal_tab,
        ).pack(side=tk.RIGHT)

        self.bind("<Button-1>", self._maybe_close_generation_drawer, add="+")

        shell = ttk.Frame(self, style="Surface.TFrame")
        shell.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.main_notebook = ttk.Notebook(shell)
        self.main_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        dashboard_tab = ttk.Frame(self.main_notebook, style="Surface.TFrame")
        self.dashboard_tab = dashboard_tab
        self.main_notebook.add(dashboard_tab, text="Главная")

        self.personal_tab = ttk.Frame(self.main_notebook, style="Surface.TFrame")
        self.main_notebook.add(self.personal_tab, text="Личный кабинет")

        self.settings_tab = ttk.Frame(self.main_notebook, style="Surface.TFrame")
        self.main_notebook.add(self.settings_tab, text="Настройки")

        self.stats_tab = ttk.Frame(self.main_notebook, style="Surface.TFrame")
        self.main_notebook.add(self.stats_tab, text="Статистика")
        self.main_notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)
        self.main_notebook.select(self.dashboard_tab)

        self.build_personal_tab(self.personal_tab)
        self.build_settings_tab(self.settings_tab)
        self.build_statistics_tab(self.stats_tab)

        main_container = ttk.PanedWindow(dashboard_tab, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left_frame = ttk.Frame(main_container, style="Surface.TFrame")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(0, weight=1)
        main_container.add(left_frame, weight=1)

        right_frame = ttk.Frame(main_container, style="Surface.TFrame")
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)
        main_container.add(right_frame, weight=1)

        frame_top = ttk.Frame(left_frame, style="Card.TFrame", padding=14)
        frame_top.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        title_frame = ttk.Frame(frame_top, style="CardInner.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(title_frame, text="Колоды", style="Section.TLabel").pack(side=tk.LEFT, anchor="w")

        self.overdue_canvas = tk.Canvas(
            title_frame, width=24, height=24,
            highlightthickness=0, bg=self.palette["panel"]
        )
        self.overdue_canvas.pack(side=tk.LEFT, padx=8)
        self.overdue_badge_text_id = None

        decks_tree_frame = ttk.Frame(frame_top, style="CardInner.TFrame")
        decks_tree_frame.pack(fill=tk.BOTH, expand=True)
        self.decks_tree = ttk.Treeview(decks_tree_frame, show="tree", selectmode="browse")
        decks_tree_vbar = ttk.Scrollbar(
            decks_tree_frame,
            orient="vertical",
            command=self.decks_tree.yview,
            style="Dark.Vertical.TScrollbar",
        )
        self.decks_tree.configure(yscrollcommand=decks_tree_vbar.set)
        self.decks_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        decks_tree_vbar.pack(side=tk.RIGHT, fill=tk.Y)
        decks_tree_frame.pack_propagate(False)
        # ttk.Treeview doesn't support classic Tk options like borderwidth/highlightthickness.
        try:
            self.decks_tree.configure(borderwidth=0, highlightthickness=0)
        except tk.TclError:
            pass
        def _decks_mousewheel(event):
            self.decks_tree.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.decks_tree.bind("<Enter>", lambda _e: self.decks_tree.bind_all("<MouseWheel>", _decks_mousewheel))
        self.decks_tree.bind("<Leave>", lambda _e: self.decks_tree.unbind_all("<MouseWheel>"))
        self.decks_tree.bind("<<TreeviewSelect>>", self.on_deck_select)
        self.phase_badge_manager = PhaseOverdueBadges(self.decks_tree)

        frame_buttons = ttk.Frame(left_frame, style="Card.TFrame", padding=12)
        frame_buttons.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 12))
        frame_buttons.grid_columnconfigure((0, 1, 2), weight=1, uniform="buttons")
        frame_buttons.grid_rowconfigure((0, 1), weight=1)

        buttons_config = [
            ("Новая колода", self.add_deck_window, "Primary.TButton"),
            ("Редактировать", self.edit_deck_window, "Secondary.TButton"),
            ("Удалить", self.delete_selected_deck, "Secondary.TButton"),
            ("Добавить карточку", self.add_card_window, "Secondary.TButton"),
            ("Режим повторения", self.mode_actions.get("repeat", self.start_repeat_mode), "Ghost.TButton"),
        ]

        for idx, (text, command, style_name) in enumerate(buttons_config):
            row = idx // 3
            col = idx % 3
            btn = self._mk_wrapped_tk_button(frame_buttons, text, command, style_name, wraplength=150)
            btn.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

        self.preview_frame = ttk.LabelFrame(right_frame, text="Превью колоды", style="Card.TLabelframe")
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        style_card(self.preview_frame, self.palette, padded=True)

        preview_container = ttk.Frame(self.preview_frame, style="CardInner.TFrame")
        preview_container.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.image_frame = ttk.Frame(preview_container, style="CardInner.TFrame")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.deck_preview_label = tk.Label(
            self.image_frame,
            text="Выберите колоду для просмотра",
            bg=self.palette["panel"],
            fg=self.palette["muted"],
            relief="flat",
            bd=0,
            wraplength=240,
            justify=tk.CENTER,
            font=("Segoe UI", 12, "bold"),
        )
        self.deck_preview_label.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(preview_container, style="CardInner.TFrame")
        text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.deck_name_label = ttk.Label(
            text_frame,
            text="Название колоды",
            style="Section.TLabel",
            wraplength=240,
            font=("Segoe UI", 13, "bold"),
        )
        self.deck_name_label.pack(anchor=tk.W, pady=(0, 10))

        self.deck_desc_label = ttk.Label(
            text_frame,
            text="Описание колоды",
            style="Muted.TLabel",
            wraplength=240,
            justify=tk.LEFT
        )
        self.deck_desc_label.pack(anchor=tk.W, pady=(0, 8))

        stats_frame = ttk.LabelFrame(text_frame, text="Статистика", style="Card.TLabelframe")
        stats_frame.pack(fill=tk.X, pady=10)
        style_card(stats_frame, self.palette, padded=True)

        self.deck_stats_label = ttk.Label(
            stats_frame,
            text="Карточек: 0\nФаз: 0/10\nОзнакомлено: 0",
            style="Muted.TLabel",
            justify=tk.LEFT
        )
        self.deck_stats_label.pack(anchor=tk.W)

        action_frame = ttk.Frame(self.preview_frame, style="CardInner.TFrame")
        action_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(2, 12))
        action_frame.grid_columnconfigure((0, 1), weight=1, uniform="actions")

        action_buttons = [
            ("Просмотреть карточки", self.show_cards_window, "Primary.TButton"),
            ("Режим воспроизведения", self.mode_actions.get("playback", self.start_playback_mode), "Ghost.TButton"),
            ("Режим ознакомления", self.start_overview_mode, "Ghost.TButton"),
        ]

        for idx, (text, command, style_name) in enumerate(action_buttons):
            row = idx // 2
            col = idx % 2
            span = 2 if idx == 2 else 1
            btn = self._mk_wrapped_tk_button(action_frame, text, command, style_name, wraplength=180)
            btn.grid(row=row, column=col, columnspan=span, padx=6, pady=6, sticky="nsew")

    def _load_credit_icon(self, size: int = 18):
        search_paths = [Path("iconcoin1.png"), Path("icon credits.jpeg")]
        for path in search_paths:
            if not path.exists():
                continue
            try:
                if PIL_AVAILABLE:
                    img = Image.open(path)
                    img = img.resize((size, size), _pil_lanczos())
                    return ImageTk.PhotoImage(img)
                if path.suffix.lower() in (".png", ".gif"):
                    return tk.PhotoImage(file=str(path))
            except Exception:
                continue
        return None

    def _load_credit_icon_pair(self, size: int = 18):
        search_paths = [Path("iconcoin1.png"), Path("icon credits.jpeg")]
        for path in search_paths:
            if not path.exists():
                continue
            try:
                if PIL_AVAILABLE:
                    img = Image.open(path)
                    img = img.resize((size, size), _pil_lanczos())
                    normal = ImageTk.PhotoImage(img)
                    gray = ImageOps.grayscale(img)
                    gray = ImageEnhance.Brightness(gray).enhance(0.7)
                    disabled = ImageTk.PhotoImage(gray)
                    return normal, disabled
                if path.suffix.lower() in (".png", ".gif"):
                    normal = tk.PhotoImage(file=str(path))
                    return normal, normal
            except Exception:
                continue
        return None, None

    def open_personal_tab(self):
        if self.main_notebook and self.personal_tab:
            self.main_notebook.select(self.personal_tab)
            self.refresh_balance_display()
            self.refresh_ledger_table()
            self.refresh_referral_info()
            self.refresh_activation_progress_ui()

    def _on_main_tab_changed(self, event):
        notebook: ttk.Notebook = event.widget
        try:
            current = notebook.nametowidget(notebook.select())
        except Exception:
            return
        if current is self.personal_tab:
            self.refresh_balance_display()
            self.refresh_ledger_table()
            self.refresh_referral_info()
            self.refresh_activation_progress_ui()
        elif current is self.stats_tab:
            self.refresh_statistics_tab()
        elif current is self.dashboard_tab and self.selected_deck_id is None:
            # Гарантируем, что видна актуальная главная вкладка.
            self.update_deck_preview()

    def refresh_balance_display(self):
        balance = self.credits_service.get_balance(self.user_id)
        formatted = f"{balance:,}".replace(",", " ")
        self.balance_var.set(formatted)
        for var in self.balance_labels:
            var.set(formatted)
        for lbl in self.balance_widgets:
            try:
                lbl.configure(text=formatted)
            except Exception:
                continue
        self.refresh_generation_menu_state()

    def register_balance_observer(self, handler) -> None:
        if handler not in self.balance_observers:
            self.balance_observers.append(handler)

    def unregister_balance_observer(self, handler) -> None:
        if handler in self.balance_observers:
            self.balance_observers.remove(handler)

    def get_csv_import_pack_config(self):
        if self.is_premium_active():
            return 15, 5, "15 карточек"
        return 10, 15, "10 карточек"

    def can_afford(self, cost: int) -> bool:
        balance = self.credits_service.get_balance(self.user_id)
        return balance >= cost

    def get_pricing_plan(self) -> str:
        self.user_account = ensure_user_account(self.user_id)
        self.user_profile = ensure_user_profile_row(self.user_id)
        plan = get_plan(
            {
                "status": self.user_account.get("status"),
                "premium_plus": self.user_profile.get("premium_plus"),
                "premium_active": self.is_premium_active(),
                "premium_until": self.user_account.get("premium_until"),
            }
        )
        return plan

    def get_cost(self, action: str, qty: int) -> int:
        plan = self.get_pricing_plan()
        return get_cost(action, qty, plan)

    def charge(self, cost: int, feature_key: str, meta: dict | None = None) -> bool:
        return self.charge_credits(cost, feature_key, meta=meta)

    def charge_credits(self, cost: int, feature_key: str, meta: dict | None = None) -> bool:
        if cost <= 0:
            return True
        reason = self._build_credit_reason(feature_key, cost, meta or {})
        try:
            self.credits_service.spend_credits(
                self.user_id,
                cost,
                reason=reason,
                meta=meta or {},
            )
        except Exception:
            return False
        self._after_balance_change()
        return True

    def refresh_generation_menu_state(self):
        if not self.generation_menu:
            return
        state = tk.NORMAL if self.is_premium_active() else tk.DISABLED
        for key in ("ocr", "wikimedia", "text_ai"):
            idx = self.generation_menu_indexes.get(key)
            if idx is None:
                continue
            try:
                self.generation_menu.entryconfig(idx, state=state)
            except Exception:
                pass

    def _record_payment(
        self,
        package_id: str,
        status: str,
        external_id: str | None = None,
        meta: dict | None = None,
    ) -> None:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO payments (user_id, package_id, ts, status, external_id, meta_json)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                self.user_id,
                package_id,
                int(time.time()),
                status,
                external_id,
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()

    def _after_balance_change(self):
        self.refresh_balance_display()
        self.refresh_ledger_table()
        self.refresh_referral_info()
        self.refresh_activation_progress_ui()
        self.refresh_account_status_vars()
        for handler in list(self.balance_observers):
            try:
                handler()
            except Exception:
                continue

    def _ensure_initial_credits(self):
        self.user_profile = ensure_user_profile_row(self.user_id)
        self.user_account = ensure_user_account(self.user_id)
        created_at = self.user_account.get("created_at") or int(time.time())
        if not self.user_account.get("created_at"):
            self.user_account = update_user_account(self.user_id, created_at=created_at)
            self.user_profile = update_user_profile(self.user_id, created_at=created_at)

        if not self.user_account.get("starter_bonus_claimed"):
            premium_until = int(time.time()) + 2 * 3600
            self.user_account = update_user_account(
                self.user_id,
                premium_until=premium_until,
                status="premium",
                starter_bonus_claimed=1,
            )
            self.user_profile = update_user_profile(
                self.user_id,
                premium_until=premium_until,
                is_premium=1,
                starter_50_claimed=1,
            )
            self.credits_service.add_credits(
                self.user_id,
                250,
                reason="Бонус регистрации: 250 ⚡",
                meta={"source": "starter_bonus"},
            )
        self.refresh_account_status_vars()
        self.premium_var.set(self.is_premium_active())

    def is_premium_active(self) -> bool:
        now_ts = int(time.time())
        self.user_account = ensure_user_account(self.user_id)
        self.user_profile = ensure_user_profile_row(self.user_id)
        premium_plus = bool(self.user_profile.get("premium_plus"))
        premium_until = int(self.user_account.get("premium_until") or 0)
        return bool(premium_plus or premium_until > now_ts)

    def set_premium_until(self, timestamp: int | None):
        ts_val = int(timestamp or 0)
        status = "premium" if ts_val > int(time.time()) else "обычный"
        self.user_account = update_user_account(
            self.user_id,
            premium_until=ts_val,
            status=status,
        )
        self.user_profile = update_user_profile(
            self.user_id,
            premium_until=ts_val,
            is_premium=1 if ts_val > int(time.time()) else 0,
        )
        self.refresh_account_status_vars()
        self.premium_var.set(self.is_premium_active())
        self.refresh_generation_menu_state()

    def grant_premium_trial(self, days: int = 30):
        until = int(time.time()) + days * 24 * 3600
        self.user_account = update_user_account(
            self.user_id, premium_until=until, status="premium"
        )
        self.user_profile = update_user_profile(
            self.user_id, premium_until=until, is_premium=1
        )
        self.refresh_account_status_vars()
        self.premium_var.set(self.is_premium_active())
        self.refresh_generation_menu_state()
        messagebox.showinfo("Pro активирован", "Пробный доступ Pro активирован.")

    def refresh_account_status_vars(self):
        self.user_account = ensure_user_account(self.user_id)
        self.user_profile = ensure_user_profile_row(self.user_id)
        now_ts = int(time.time())
        premium_until = int(self.user_account.get("premium_until") or 0)
        premium_active = premium_until > now_ts
        verified = bool(self.user_account.get("verified"))
        premium_plus = bool(self.user_profile.get("premium_plus")) or self.user_account.get("status") == "premium_plus"
        if premium_plus:
            status = "premium_plus"
        else:
            status = "активен" if verified else ("premium" if premium_active else "обычный")
        if status != self.user_account.get("status"):
            self.user_account = update_user_account(self.user_id, status=status)
        if int(self.user_profile.get("premium_until") or 0) != premium_until:
            self.user_profile = update_user_profile(self.user_id, premium_until=premium_until)
        if (premium_active or premium_plus) and not self.user_profile.get("is_premium"):
            self.user_profile = update_user_profile(self.user_id, is_premium=1)
        if not premium_active and not premium_plus and self.user_profile.get("is_premium"):
            self.user_profile = update_user_profile(self.user_id, is_premium=0)
        if premium_plus:
            self.account_status_var.set("Premium+")
        else:
            self.account_status_var.set("Premium" if premium_active else "Обычный")
        remaining = max(0, premium_until - now_ts)
        hours, rem = divmod(remaining, 3600)
        minutes, seconds = divmod(rem, 60)
        self.premium_timer_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.user_id_var.set(self.user_id)

    def start_premium_timer(self):
        if self.premium_timer_job is not None:
            try:
                self.after_cancel(self.premium_timer_job)
            except Exception:
                pass
        self.refresh_account_status_vars()
        self.refresh_generation_menu_state()
        self.premium_timer_job = self.after(1000, self.start_premium_timer)

    def guard_premium_and_spend(
        self,
        feature_key: str,
        cost_credits: int,
        require_premium: bool = True,
        meta: dict | None = None,
    ) -> bool:
        self.user_profile = ensure_user_profile_row(self.user_id)
        self.refresh_account_status_vars()
        if require_premium and not self.is_premium_active():
            if messagebox.askyesno(
                "Нужна подписка",
                "Эта функция доступна только в Pro.\nОткрыть личный кабинет?",
            ):
                self.open_personal_tab()
            return False

        balance = self.credits_service.get_balance(self.user_id)
        if cost_credits > 0 and balance < cost_credits:
            if messagebox.askyesno(
                "Недостаточно кредитов",
                "Не хватает кредитов для операции.\nОткрыть личный кабинет для пополнения?",
            ):
                self.open_personal_tab()
            return False

        if cost_credits > 0:
            reason = self._build_credit_reason(feature_key, cost_credits, meta or {})
            try:
                self.credits_service.spend_credits(
                    self.user_id,
                    cost_credits,
                    reason=reason,
                    meta=meta or {},
                )
            except ValueError:
                messagebox.showwarning(
                    "Недостаточно кредитов",
                    "Не удалось списать кредиты. Проверьте баланс.",
                )
                return False
            self._after_balance_change()
        return True

    def spend_credits_or_warn(self, amount: int, reason: str, meta: dict | None = None) -> bool:
        return self.guard_premium_and_spend(reason, amount, require_premium=False, meta=meta)

    def consume_wikimedia_ticket(self) -> bool:
        if not self.is_premium_active():
            messagebox.showinfo(
                "Нужна подписка",
                "Wikimedia доступна только в Pro. Активируйте подписку в личном кабинете.",
            )
            return False
        self.user_profile = ensure_user_profile_row(self.user_id)
        tickets = int(self.user_profile.get("wikimedia_tickets") or 0)
        if tickets > 0:
            self.user_profile = update_user_profile(
                self.user_id, wikimedia_tickets=max(0, tickets - 1)
            )
            return True
        if not self.guard_premium_and_spend(
            "wikimedia_bundle",
            WIKIMEDIA_IMPORT_COST,
            require_premium=True,
            meta={"bundle": WIKIMEDIA_TICKET_SIZE},
        ):
            return False
        self.user_profile = update_user_profile(
            self.user_id, wikimedia_tickets=max(0, WIKIMEDIA_TICKET_SIZE - 1)
        )
        return True

    def mark_first_import(self):
        self.user_profile = ensure_user_profile_row(self.user_id)
        if self.user_profile.get("first_import_ts"):
            return
        self.user_profile = update_user_profile(
            self.user_id, first_import_ts=int(time.time())
        )
        self.refresh_activation_progress_ui()

    def _spend_for_ai_image(self, feature_key: str = "card_image_generation", meta: dict | None = None) -> bool:
        payload = meta or {}
        payload.setdefault("images", 1)
        return self.guard_premium_and_spend(
            feature_key,
            CARD_IMAGE_CREDIT_COST,
            require_premium=True,
            meta=payload,
        )

    def _count_cards_created(self) -> int:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cards;")
        val = cur.fetchone()[0] or 0
        conn.close()
        return int(val)

    def _count_reviews_done(self) -> int:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM reviews;")
        val = cur.fetchone()[0] or 0
        conn.close()
        return int(val)

    def _has_first_import(self) -> bool:
        self.user_profile = ensure_user_profile_row(self.user_id)
        if self.user_profile.get("first_import_ts"):
            return True
        imported = get_imported_files()
        return bool(imported)

    def get_activation_progress(self) -> dict:
        self.user_profile = ensure_user_profile_row(self.user_id)
        created_at = int(self.user_profile.get("created_at") or int(time.time()))
        age_seconds = int(time.time()) - created_at
        age_hours = age_seconds / 3600
        cards_created = self._count_cards_created()
        reviews_done = self._count_reviews_done()
        first_import = self._has_first_import()
        remaining_seconds = max(0, int(ACTIVATION_MIN_HOURS * 3600 - age_seconds))
        rem_hours = remaining_seconds // 3600
        rem_minutes = (remaining_seconds % 3600) // 60
        steps = [
            {
                "title": "Аккаунт старше 24ч",
                "done": age_hours >= ACTIVATION_MIN_HOURS,
                "value": min(100.0, (age_hours / ACTIVATION_MIN_HOURS) * 100 if ACTIVATION_MIN_HOURS else 0),
                "progress_text": "Готово" if age_hours >= ACTIVATION_MIN_HOURS else f"Осталось {rem_hours}ч {rem_minutes}м",
            },
            {
                "title": "Создано ≥ 10 карточек",
                "done": cards_created >= ACTIVATION_MIN_CARDS,
                "value": min(100.0, cards_created / ACTIVATION_MIN_CARDS * 100 if ACTIVATION_MIN_CARDS else 0),
                "progress_text": f"{cards_created}/{ACTIVATION_MIN_CARDS}",
            },
            {
                "title": "Пройдено ≥ 20 повторений",
                "done": reviews_done >= ACTIVATION_MIN_REVIEWS,
                "value": min(100.0, reviews_done / ACTIVATION_MIN_REVIEWS * 100 if ACTIVATION_MIN_REVIEWS else 0),
                "progress_text": f"{reviews_done}/{ACTIVATION_MIN_REVIEWS}",
            },
            {
                "title": "Первый импорт / подтверждение",
                "done": bool(first_import or cards_created >= 1),
                "value": 100.0 if first_import or cards_created >= 1 else 0.0,
                "progress_text": "1/1" if first_import or cards_created >= 1 else "0/1",
            },
        ]
        overall = sum(step["value"] for step in steps) / len(steps) if steps else 0
        return {"steps": steps, "overall": overall}

    def refresh_activation_progress_ui(self):
        progress = self.get_activation_progress()
        for idx, step in enumerate(progress["steps"]):
            var = self.activation_progress_vars.get(f"step_{idx}")
            if var is None:
                continue
            var.set(step["value"])
            text_var = self.activation_progress_labels.get(f"step_{idx}")
            if text_var is not None:
                text_var.set(step.get("progress_text", ""))
        self.activation_overall_var.set(progress["overall"])
        self.activation_overall_label_var.set(f"{progress['overall']:.0f}%")
        claimed = bool(self.user_profile.get("activation_200_claimed"))
        all_done = all(step["done"] for step in progress["steps"])
        if all_done and not bool(self.user_account.get("verified")):
            self.user_account = update_user_account(self.user_id, verified=1, status="активен")
        if all_done:
            self.activation_status_var.set("Статус аккаунта активен")
        else:
            self.activation_status_var.set("Статус аккаунта не верифицирован")
        if claimed:
            self.activation_status_var.set(f"{self.activation_status_var.get()} · Бонус +200 уже активирован ✔")
        if getattr(self, "activation_status_label", None) is not None:
            color = self.palette["success"] if all_done else self.palette["muted"]
            self.activation_status_label.config(fg=color)
        self.refresh_account_status_vars()

    def check_activation_bonus(self):
        self.user_profile = ensure_user_profile_row(self.user_id)
        if self.user_profile.get("activation_200_claimed"):
            messagebox.showinfo("Активация", "Бонус +200 уже был начислен.")
            return
        progress = self.get_activation_progress()
        if all(step["done"] for step in progress["steps"]):
            self.credits_service.add_credits(
                self.user_id,
                200,
                reason="Активация аккаунта: +200 ⚡",
                meta={"source": "activation_bonus"},
            )
            self.user_profile = update_user_profile(self.user_id, activation_200_claimed=1)
            self._after_balance_change()
            messagebox.showinfo("Готово", "Условия выполнены! +200 кредитов начислены.")
        else:
            missing = [step["title"] for step in progress["steps"] if not step["done"]]
            messagebox.showinfo(
                "Не все условия",
                "Завершите шаги для активации:\n- " + "\n- ".join(missing),
            )
        self.refresh_activation_progress_ui()

    def start_payment_flow(self, package_id: str):
        try:
            url = build_payment_url(self.user_id, package_id)
        except Exception as exc:
            messagebox.showerror("Оплата", f"Не удалось подготовить платеж: {exc}")
            return
        package = PACKAGES.get(package_id, {})
        self._record_payment(
            package_id,
            status="initiated",
            meta={"url": url, "credits": package.get("credits")},
        )
        webbrowser.open(url)
        messagebox.showinfo(
            "Оплата",
            "Ссылка на оплату открыта в браузере.\n"
            "После завершения нажмите «Я оплатил» и введите ID/код платежа.",
        )
        self.package_choice_var.set(package_id)

    def confirm_manual_payment(self):
        package_id = self.package_choice_var.get() or "pack_500"
        code = self.payment_code_var.get().strip()
        success, details = verify_payment(self.user_id, package_id, code)
        if not success:
            self.payment_status_var.set("Не удалось подтвердить платёж (заглушка).")
            messagebox.showerror("Оплата", "Платеж не подтвержден.")
            return
        package = PACKAGES.get(package_id, {})
        credits_amount = int(package.get("credits", 0))
        self.payment_status_var.set(f"Подтверждено: {credits_amount} кредитов")
        self._record_payment(
            package_id,
            status="confirmed",
            external_id=details.get("payment_id"),
            meta=details,
        )
        self.credits_service.add_credits(
            self.user_id,
            credits_amount,
            reason=self._build_purchase_reason(package),
            meta={"payment": details, "package_id": package_id},
        )
        self._after_balance_change()

    def refresh_ledger_table(self):
        if not self.ledger_tree:
            return
        for item in self.ledger_tree.get_children():
            self.ledger_tree.delete(item)
        for row in self.credits_service.get_ledger(self.user_id, limit=200):
            ts_str = datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d %H:%M")
            delta = row["delta"]
            sign = "+" if delta >= 0 else "−"
            amount = f"{sign}{abs(delta)}"
            note = row.get("reason") or ""
            meta = row.get("meta") or {}
            if meta.get("payment"):
                note += " · оплата"
            if meta.get("referee_id"):
                note += " · реферал"
            self.ledger_tree.insert(
                "",
                "end",
                values=(ts_str, amount, note.strip()),
            )

    def _build_credit_reason(self, feature_key: str, cost_credits: int, meta: dict) -> str:
        key = (feature_key or "").strip()
        lowered = key.lower()
        if lowered in {"ocr_image", "ocr"}:
            mode = (meta.get("ocr_mode") or "").lower()
            pages = int(meta.get("pages") or 1)
            if mode == "pro":
                return f"OCR PRO 👑: {pages} стр ({cost_credits} ⚡)"
            return f"OCR: {pages} стр ({cost_credits} ⚡)"
        if lowered == "image_id_import":
            files = int(meta.get("files") or 1)
            return f"Импорт изображений: {files} шт ({cost_credits} ⚡)"
        if lowered == "wikimedia_bundle":
            bundle = int(meta.get("bundle") or 0)
            if bundle:
                return f"Wikimedia: {bundle} стр ({cost_credits} ⚡)"
            return f"Wikimedia: {cost_credits} ⚡"
        if lowered in {"card_image_generation", "ai image generation"}:
            images = int(meta.get("images") or 1)
            return f"AI-картинки: +{images} шт (доплата {cost_credits} ⚡)"
        if lowered == "ai video generation":
            videos = int(meta.get("videos") or 1)
            return f"AI-видео: +{videos} шт (доплата {cost_credits} ⚡)"
        return key or "Операция"

    def _build_purchase_reason(self, package: dict) -> str:
        credits = int(package.get("credits") or 0)
        price = package.get("price")
        currency = package.get("currency") or "USD"
        currency_symbols = {"USD": "$", "EUR": "€", "RUB": "₽"}
        if price is None:
            return f"Покупка: {credits} ⚡"
        symbol = currency_symbols.get(currency, currency)
        return f"Покупка: {credits} ⚡ ({symbol}{price:.2f})"

    def refresh_referral_info(self):
        if not self.ref_summary_vars:
            return
        summary = self.referral_service.get_summary(self.user_id)
        self.ref_summary_vars["invited"].set(str(summary.get("invited", 0)))
        self.ref_summary_vars["activated"].set(str(summary.get("activated", 0)))
        self.ref_summary_vars["earned"].set(str(summary.get("earned", 0)))
        ref_code = self.referral_service.get_ref_code(self.user_id)
        ref_link = self.referral_service.get_ref_link(self.user_id)
        if "code" in self.ref_summary_vars:
            self.ref_summary_vars["code"].set(ref_code)
        if "link" in self.ref_summary_vars:
            self.ref_summary_vars["link"].set(ref_link)

    def build_personal_tab(self, tab: ttk.Frame):
        for child in tab.winfo_children():
            child.destroy()

        container = ttk.Frame(tab, style="Surface.TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0, bg=self.palette["background"])
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="Surface.TFrame")

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        scroll_frame.grid_columnconfigure((0, 1), weight=1)

        account_card = ttk.Frame(scroll_frame, style="Card.TFrame", padding=14)
        account_card.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        style_card(account_card, self.palette, padded=False)
        ttk.Label(
            account_card,
            text="Профиль",
            style="Section.TLabel",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        user_row = ttk.Frame(account_card, style="CardInner.TFrame")
        user_row.pack(fill=tk.X, pady=(6, 4))
        ttk.Label(user_row, text="User ID:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Entry(user_row, textvariable=self.user_id_var, state="readonly", width=36).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(user_row, text="Копировать", style="Secondary.TButton", command=self.copy_user_id).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(user_row, text="Синхронизация", style="Secondary.TButton", command=self.open_sync_flow).pack(
            side=tk.LEFT, padx=6
        )
        status_row = ttk.Frame(account_card, style="CardInner.TFrame")
        status_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(status_row, text="Статус:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.account_status_var, style="HeaderSub.TLabel").pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Label(status_row, text="До конца Premium:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.premium_timer_var, style="HeaderSub.TLabel").pack(
            side=tk.LEFT, padx=6
        )

        balance_card = ttk.Frame(scroll_frame, style="Card.TFrame", padding=14)
        balance_card.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        style_card(balance_card, self.palette, padded=False)
        ttk.Label(balance_card, text="Баланс", style="Section.TLabel", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        balance_row = ttk.Frame(balance_card, style="CardInner.TFrame")
        balance_row.pack(fill=tk.X, pady=6)
        icon = self.credit_icon_large or self._load_credit_icon(size=48)
        if icon:
            self.credit_icon_large = icon
            ttk.Label(balance_row, image=icon, style="HeaderSub.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        else:
            ttk.Label(balance_row, text="💠", style="HeaderSub.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        lbl_balance = ttk.Label(
            balance_row,
            textvariable=self.balance_var,
            style="Title.TLabel",
            font=("Segoe UI", 24, "bold"),
        )
        lbl_balance.pack(side=tk.LEFT)
        self.balance_labels.append(self.balance_var)
        self.balance_widgets.append(lbl_balance)
        ttk.Button(balance_row, text="Обновить", style="Secondary.TButton", command=self.refresh_balance_display).pack(
            side=tk.RIGHT
        )
        ttk.Label(
            balance_card,
            text="Базовый пакет Pro: 2000 кредитов/мес (инфраструктура для пополнений готова).",
            style="Muted.TLabel",
            wraplength=320,
        ).pack(anchor="w", pady=(6, 0))

        purchase_card = ttk.Frame(scroll_frame, style="Card.TFrame", padding=14)
        purchase_card.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        style_card(purchase_card, self.palette, padded=False)
        ttk.Label(purchase_card, text="Купить кредиты", style="Section.TLabel", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        for package_id, data in PACKAGES.items():
            row = ttk.Frame(purchase_card, style="CardInner.TFrame")
            row.pack(fill=tk.X, pady=6)
            icon_ref = self.credit_icon_small or self.credit_icon_image
            if icon_ref:
                ttk.Label(row, image=icon_ref, style="HeaderSub.TLabel").pack(side=tk.LEFT, padx=(0, 6))
            else:
                ttk.Label(row, text="💎", style="HeaderSub.TLabel").pack(side=tk.LEFT, padx=(0, 6))
            ttk.Label(
                row,
                text=f"{data['title']} · ${data['price']}",
                style="HeaderSub.TLabel",
            ).pack(side=tk.LEFT)
            ttk.Button(
                row,
                text="Купить",
                style="Primary.TButton",
                command=lambda pid=package_id: self.start_payment_flow(pid),
            ).pack(side=tk.RIGHT)

        confirm_frame = ttk.LabelFrame(purchase_card, text="Я оплатил / проверка оплаты", style="Card.TLabelframe")
        confirm_frame.pack(fill=tk.X, pady=(8, 0))
        style_card(confirm_frame, self.palette, padded=True)
        ttk.Label(confirm_frame, text="Пакет:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            confirm_frame,
            state="readonly",
            values=list(PACKAGES.keys()),
            textvariable=self.package_choice_var,
            width=12,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(confirm_frame, text="Payment ID / код:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(confirm_frame, textvariable=self.payment_code_var).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        confirm_frame.columnconfigure(1, weight=1)
        ttk.Button(
            confirm_frame,
            text="Я оплатил",
            style="Secondary.TButton",
            command=self.confirm_manual_payment,
        ).grid(row=0, column=2, rowspan=2, sticky="e", padx=4, pady=4)
        ttk.Label(confirm_frame, textvariable=self.payment_status_var, style="Muted.TLabel").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(4, 0)
        )
        ttk.Button(
            purchase_card,
            text="Активировать Pro (демо)",
            style="Ghost.TButton",
            command=lambda: self.grant_premium_trial(days=30),
        ).pack(anchor="e", pady=(6, 0))

        activation_card = ttk.Frame(scroll_frame, style="Card.TFrame", padding=14)
        activation_card.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        style_card(activation_card, self.palette, padded=False)
        ttk.Label(
            activation_card,
            text="Активация аккаунта (+200 кредитов)",
            style="Section.TLabel",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        self.activation_status_label = tk.Label(
            activation_card,
            textvariable=self.activation_status_var,
            bg=self.palette["panel"],
            fg=self.palette["muted"],
            font=("Segoe UI", 12, "bold"),
        )
        self.activation_status_label.pack(anchor="w", pady=(2, 6))
        overall_row = ttk.Frame(activation_card, style="CardInner.TFrame")
        overall_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(overall_row, text="Общий прогресс:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Progressbar(
            overall_row,
            maximum=100,
            variable=self.activation_overall_var,
            length=240,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Label(overall_row, textvariable=self.activation_overall_label_var, style="HeaderSub.TLabel").pack(
            side=tk.LEFT
        )

        steps_frame = ttk.Frame(activation_card, style="CardInner.TFrame")
        steps_frame.pack(fill=tk.BOTH, expand=True)
        titles = [
            "Аккаунту ≥ 24 часов",
            "Создано ≥ 10 карточек",
            "Пройдено ≥ 20 повторений",
            "Первый импорт / подтверждение",
        ]
        for idx, title in enumerate(titles):
            row = ttk.Frame(steps_frame, style="CardInner.TFrame")
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=title, style="Muted.TLabel").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0)
            self.activation_progress_vars[f"step_{idx}"] = var
            bar = ttk.Progressbar(row, maximum=100, variable=var, length=200)
            bar.pack(side=tk.RIGHT, padx=6)
            text_var = tk.StringVar(value="")
            self.activation_progress_labels[f"step_{idx}"] = text_var
            ttk.Label(row, textvariable=text_var, style="HeaderSub.TLabel").pack(side=tk.RIGHT, padx=6)
        ttk.Button(
            activation_card,
            text="Проверить активацию",
            style="Secondary.TButton",
            command=self.check_activation_bonus,
        ).pack(anchor="e", pady=(6, 0))

        history_card = ttk.Frame(scroll_frame, style="Card.TFrame", padding=14)
        history_card.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        style_card(history_card, self.palette, padded=False)
        ttk.Label(history_card, text="История операций", style="Section.TLabel", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        history_inner = ttk.Frame(history_card, style="CardInner.TFrame")
        history_inner.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        columns = ("date", "delta", "note")
        self.ledger_tree = ttk.Treeview(history_inner, columns=columns, show="headings", height=10)
        self.ledger_tree.heading("date", text="Дата")
        self.ledger_tree.heading("delta", text="+/- кредиты")
        self.ledger_tree.heading("note", text="Примечание")
        self.ledger_tree.column("date", width=150, anchor="center")
        self.ledger_tree.column("delta", width=100, anchor="center")
        self.ledger_tree.column("note", width=520, anchor="w")
        self.ledger_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(history_card, text="Обновить историю", style="Ghost.TButton", command=self.refresh_ledger_table).pack(
            anchor="e", pady=(6, 0)
        )

        referral_card = ttk.Frame(scroll_frame, style="Card.TFrame", padding=14)
        referral_card.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 12))
        style_card(referral_card, self.palette, padded=False)
        ttk.Label(referral_card, text="Реферальная система", style="Section.TLabel", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ref_inner = ttk.Frame(referral_card, style="CardInner.TFrame")
        ref_inner.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self.ref_summary_vars = {
            "code": tk.StringVar(value=""),
            "link": tk.StringVar(value=""),
            "invited": tk.StringVar(value="0"),
            "activated": tk.StringVar(value="0"),
            "earned": tk.StringVar(value="0"),
        }

        row_code = ttk.Frame(ref_inner, style="CardInner.TFrame")
        row_code.pack(fill=tk.X, pady=4)
        ttk.Label(row_code, text="Реф-код:", style="HeaderSub.TLabel").pack(side=tk.LEFT)
        ttk.Entry(row_code, textvariable=self.ref_summary_vars["code"], state="readonly", width=18).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Label(row_code, text="Ссылка:", style="HeaderSub.TLabel").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(row_code, textvariable=self.ref_summary_vars["link"], state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(row_code, text="Скопировать ссылку", command=self.copy_ref_link, style="Secondary.TButton").pack(
            side=tk.LEFT, padx=4
        )

        stats_row = ttk.Frame(ref_inner, style="CardInner.TFrame")
        stats_row.pack(fill=tk.X, pady=4)
        for label, key in (("Приглашено", "invited"), ("Активировано", "activated"), ("Заработано", "earned")):
            stat_frame = ttk.Frame(stats_row, style="CardInner.TFrame")
            stat_frame.pack(side=tk.LEFT, padx=8)
            ttk.Label(stat_frame, text=label, style="Muted.TLabel").pack()
            ttk.Label(stat_frame, textvariable=self.ref_summary_vars[key], style="HeaderSub.TLabel").pack()

        ttk.Label(
            ref_inner,
            text="Условия активации: 24 часа возраста, 10 карточек, 20 повторений, первый импорт.",
            style="Muted.TLabel",
            wraplength=900,
        ).pack(anchor="w", pady=(6, 0))

        self.refresh_balance_display()
        self.refresh_ledger_table()
        self.refresh_referral_info()
        self.refresh_activation_progress_ui()

    def build_settings_tab(self, tab: ttk.Frame):
        for child in tab.winfo_children():
            child.destroy()

        container = ttk.Frame(tab, style="Surface.TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0, bg=self.palette["background"])
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="Surface.TFrame")

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        openai_frame = ttk.LabelFrame(scroll_frame, text="Настройки OpenAI", style="Card.TLabelframe")
        openai_frame.pack(fill=tk.X, padx=10, pady=10)
        style_card(openai_frame, self.palette, padded=True)

        ttk.Label(
            openai_frame,
            text="API ключ OpenAI (формат sk-... / sk-proj-...; хранится только в памяти):",
        ).pack(anchor="w", padx=10, pady=(10, 0))

        openai_key_var = tk.StringVar(value=OPENAI_API_KEY or "")
        entry_key = ttk.Entry(openai_frame, textvariable=openai_key_var, show="*")
        entry_key.pack(fill=tk.X, padx=10, pady=5)
        create_context_menu(entry_key)

        def paste_from_clipboard():
            try:
                text = self.clipboard_get()
            except tk.TclError:
                text = ""
            openai_key_var.set(text.strip())

        ttk.Button(
            openai_frame,
            text="Вставить из буфера обмена",
            command=paste_from_clipboard,
        ).pack(anchor="e", padx=10, pady=(0, 5))

        def save_openai_key():
            global OPENAI_API_KEY
            key = openai_key_var.get().strip()
            OPENAI_API_KEY = key or None
            messagebox.showinfo("Сохранено", "Ключ сохранён в памяти приложения.")

        btn_frame = ttk.Frame(openai_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="OK", command=save_openai_key).pack(side=tk.RIGHT)

        translation_frame = ttk.LabelFrame(scroll_frame, text="Настройки перевода", style="Card.TLabelframe")
        translation_frame.pack(fill=tk.X, padx=10, pady=10)
        style_card(translation_frame, self.palette, padded=True)

        use_dict_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.use_embedded_dict)
        ttk.Checkbutton(
            translation_frame,
            text="Использовать встроенный словарь",
            variable=use_dict_var,
        ).pack(anchor="w", padx=20, pady=(10, 5))

        use_openai_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.use_openai)
        ttk.Checkbutton(
            translation_frame,
            text="Использовать OpenAI для перевода (если есть ключ)",
            variable=use_openai_var,
        ).pack(anchor="w", padx=20, pady=5)

        show_trans_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.show_translations)
        ttk.Checkbutton(
            translation_frame,
            text="Показывать переводы над словами в режиме повторения (лицевая сторона)",
            variable=show_trans_var,
        ).pack(anchor="w", padx=20, pady=5)

        show_back_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.show_back_translation)
        ttk.Checkbutton(
            translation_frame,
            text="Всегда показывать русский перевод на задней стороне карточки",
            variable=show_back_var,
        ).pack(anchor="w", padx=20, pady=5)

        ttk.Label(translation_frame, text="Приоритет перевода:").pack(anchor="w", padx=20, pady=(8, 4))
        priority_var = tk.StringVar(value="dictionary")
        ttk.Radiobutton(
            translation_frame,
            text="Сначала словарь, потом OpenAI",
            variable=priority_var,
            value="dictionary",
        ).pack(anchor="w", padx=30)
        ttk.Radiobutton(
            translation_frame,
            text="Сначала OpenAI, потом словарь",
            variable=priority_var,
            value="openai",
        ).pack(anchor="w", padx=30)

        def save_translation_settings():
            TRANSLATION_SETTINGS.use_embedded_dict = use_dict_var.get()
            TRANSLATION_SETTINGS.use_openai = use_openai_var.get()
            TRANSLATION_SETTINGS.show_translations = show_trans_var.get()
            TRANSLATION_SETTINGS.show_back_translation = show_back_var.get()
            TRANSLATION_SETTINGS.save()
            messagebox.showinfo("Сохранено", "Настройки перевода сохранены.")

        btn_frame = ttk.Frame(translation_frame)
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Button(btn_frame, text="Сохранить", command=save_translation_settings).pack(side=tk.RIGHT)

        audio_frame = ttk.LabelFrame(scroll_frame, text="Аудиоустройство (цифровой слух)", style="Card.TLabelframe")
        audio_frame.pack(fill=tk.X, padx=10, pady=10)
        style_card(audio_frame, self.palette, padded=True)

        if not SR_AVAILABLE:
            ttk.Label(
                audio_frame,
                text="Чтобы выбрать микрофон, установите SpeechRecognition и PyAudio:\n"
                     "pip install SpeechRecognition pyaudio",
                style="Muted.TLabel",
            ).pack(anchor="w", padx=10, pady=10)
        else:
            try:
                devices = sr.Microphone.list_microphone_names()
            except Exception as exc:
                devices = []
                ttk.Label(
                    audio_frame,
                    text=f"Не удалось получить список устройств: {exc}",
                    style="Muted.TLabel",
                ).pack(anchor="w", padx=10, pady=10)

            ttk.Label(
                audio_frame,
                text="Выбери устройство записи, которое будет слушать звук в режиме «Генерация через цифрового слуха».",
            ).pack(anchor="w", padx=10, pady=(10, 0))
            listbox = tk.Listbox(audio_frame, height=6)
            listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            selected_initial = 0
            for i, name in enumerate(devices):
                listbox.insert(tk.END, f"{i}: {name}")
                if self.microphone_index is not None and i == self.microphone_index:
                    selected_initial = i

            if devices:
                listbox.selection_set(selected_initial)
                listbox.see(selected_initial)

            def save_device():
                sel = listbox.curselection()
                if not sel:
                    self.microphone_index = None
                else:
                    idx_line = listbox.get(sel[0])
                    idx_str = idx_line.split(":", 1)[0]
                    try:
                        self.microphone_index = int(idx_str)
                    except ValueError:
                        self.microphone_index = None
                messagebox.showinfo(
                    "Сохранено",
                    f"Устройство записи для цифрового слуха установлено: "
                    f"{self.microphone_index if self.microphone_index is not None else 'по умолчанию'}",
                )

            btn_frame = ttk.Frame(audio_frame)
            btn_frame.pack(fill=tk.X, padx=10, pady=6)
            ttk.Button(btn_frame, text="OK", command=save_device).pack(side=tk.RIGHT)

        dictionary_frame = ttk.LabelFrame(scroll_frame, text="Управление словарями", style="Card.TLabelframe")
        dictionary_frame.pack(fill=tk.X, padx=10, pady=10)
        style_card(dictionary_frame, self.palette, padded=True)

        stats = DICTIONARY_MANAGER.get_statistics()
        stats_text = (
            f"Всего слов в словаре: {stats['total_words']:,}\n"
            f"Загруженные файлы: {len(stats['loaded_files'])}\n"
            f"Используемая память: {stats['memory_size_mb']:.2f} МБ\n\n"
            "Формат: немецкое слово -> русский перевод"
        )
        ttk.Label(dictionary_frame, text=stats_text, justify=tk.LEFT).pack(padx=10, pady=10, anchor="w")

        if stats["loaded_files"]:
            files_frame = ttk.Frame(dictionary_frame, style="CardInner.TFrame")
            files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            listbox = tk.Listbox(files_frame, height=4)
            listbox.pack(fill=tk.BOTH, expand=True)
            for file in stats["loaded_files"]:
                listbox.insert(tk.END, file)

        btn_frame = ttk.Frame(dictionary_frame, style="CardInner.TFrame")
        btn_frame.pack(padx=10, pady=10, fill=tk.X)

        def refresh_tab():
            self.build_settings_tab(tab)

        def load_dictionary():
            filetypes = [
                ("CSV файлы", "*.csv"),
                ("JSON файлы", "*.json"),
                ("Сжатые файлы", "*.gz *.json.gz"),
                ("Все файлы", "*.*"),
            ]
            filename = filedialog.askopenfilename(
                title="Выберите файл словаря",
                filetypes=filetypes,
            )
            if filename:
                try:
                    if filename.endswith(".csv"):
                        count = DICTIONARY_MANAGER.load_from_csv(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")
                    elif filename.endswith(".json"):
                        count = DICTIONARY_MANAGER.load_from_json(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")
                    elif filename.endswith((".gz", ".json.gz")):
                        count = DICTIONARY_MANAGER.load_from_compressed(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")

                    if filename not in TRANSLATION_SETTINGS.dictionary_paths:
                        TRANSLATION_SETTINGS.dictionary_paths.append(filename)
                        TRANSLATION_SETTINGS.save()

                    refresh_tab()
                except Exception as exc:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить словарь:\n{exc}")

        def export_dictionary():
            filename = filedialog.asksaveasfilename(
                title="Экспорт словаря",
                defaultextension=".csv",
                filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            )
            if filename:
                try:
                    gui_hooks.export_will_start(filename)
                    DICTIONARY_MANAGER.export_to_csv(filename)
                    messagebox.showinfo("Успех", f"Словарь экспортирован в {filename}")
                except Exception as exc:
                    messagebox.showerror("Ошибка", f"Не удалось экспортировать словарь:\n{exc}")

        def save_compressed():
            filename = filedialog.asksaveasfilename(
                title="Сохранить сжатый словарь",
                defaultextension=".json.gz",
                filetypes=[("Сжатые JSON файлы", "*.json.gz"), ("Все файлы", "*.*")],
            )
            if filename:
                try:
                    DICTIONARY_MANAGER.save_compressed_dictionary(filename)
                    messagebox.showinfo("Успех", f"Словарь сохранен в {filename}")
                except Exception as exc:
                    messagebox.showerror("Ошибка", f"Не удалось сохранить словарь:\n{exc}")

        def search_word():
            search_win = tk.Toplevel(self)
            search_win.title("Поиск слова")
            search_win.geometry("400x300")
            search_win.grab_set()

            ttk.Label(search_win, text="Введите слово для поиска:").pack(padx=10, pady=(10, 0))

            entry = ttk.Entry(search_win)
            entry.pack(fill=tk.X, padx=10, pady=5)

            results_text = tk.Text(search_win, height=10)
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            def perform_search():
                word = entry.get().strip()
                if word:
                    results = DICTIONARY_MANAGER.search_words(word, limit=20)
                    results_text.delete(1.0, tk.END)
                    if results:
                        for german, russian in results:
                            results_text.insert(tk.END, f"{german} -> {russian}\n")
                    else:
                        results_text.insert(tk.END, "Совпадений не найдено")

            ttk.Button(search_win, text="Поиск", command=perform_search).pack(pady=10)

        ttk.Button(btn_frame, text="Загрузить словарь", command=load_dictionary).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Экспорт в CSV", command=export_dictionary).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Сохранить сжатый", command=save_compressed).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(btn_frame, text="Поиск слова", command=search_word).grid(row=1, column=0, padx=5, pady=5, columnspan=3)

    def build_statistics_tab(self, tab: ttk.Frame):
        for child in tab.winfo_children():
            child.destroy()

        container = ttk.Frame(tab, style="Surface.TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        decks = list_decks()
        if not decks:
            ttk.Label(container, text="Сначала создайте колоду.", style="Muted.TLabel").pack(pady=20)
            return

        notebook = ttk.Notebook(container)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for deck in decks:
            deck_frame = ttk.Frame(notebook)
            notebook.add(deck_frame, text=deck["name"])
            self.create_deck_statistics_tab(deck_frame, deck["id"], deck["name"])

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        def update_all_dates():
            for i in range(notebook.index("end")):
                tab_frame = notebook.nametowidget(notebook.tabs()[i])
                for child in tab_frame.winfo_children():
                    if hasattr(child, "update_charts"):
                        child.update_charts()
            messagebox.showinfo("Обновлено", "Все графики обновлены")

        ttk.Button(btn_frame, text="Обновить все графики", command=update_all_dates).pack(side=tk.RIGHT)

    def refresh_statistics_tab(self):
        if self.stats_tab:
            self.build_statistics_tab(self.stats_tab)

    def _save_sync_config(self) -> None:
        self.sync_config["token"] = self.sync_token
        self.sync_config["user_email"] = self.sync_user_email
        save_sync_config(self.sync_config)

    def _set_sync_token(self, token: str | None, email: str | None = None) -> None:
        self.sync_token = token
        if email is not None:
            self.sync_user_email = email
        self._save_sync_config()

    def _clear_sync_token(self) -> None:
        self._set_sync_token(None, self.sync_user_email)

    def open_sync_flow(self) -> None:
        if self.sync_token:
            self.open_sync_deck_window()
        else:
            self.open_sync_login_window()

    def open_sync_login_window(self) -> None:
        win = tk.Toplevel(self)
        win.title("Синхронизация с сайтом")
        win.transient(self)
        win.resizable(False, False)

        container = ttk.Frame(win, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Для синхронизации коллекции создайте учетную запись",
            style="HeaderSub.TLabel",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            container,
            text="При регистрации на сайте можно указать реферальный код для бонуса 300 кредитов.",
            style="Muted.TLabel",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 12))

        form = ttk.Frame(container)
        form.pack(fill=tk.X)
        ttk.Label(form, text="Email").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(form, text="Пароль").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        email_var = tk.StringVar(value=self.sync_user_email or "")
        password_var = tk.StringVar(value="")
        email_entry = ttk.Entry(form, textvariable=email_var, width=36)
        email_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        password_entry = ttk.Entry(form, textvariable=password_var, show="*", width=36)
        password_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        form.columnconfigure(1, weight=1)

        status_var = tk.StringVar(value="")
        status_label = ttk.Label(container, textvariable=status_var, style="Muted.TLabel")
        status_label.pack(anchor="w", pady=(8, 0))
        progress = ttk.Progressbar(container, mode="indeterminate")

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(10, 0))
        sync_button = ttk.Button(buttons, text="Синхронизировать", style="Primary.TButton")
        sync_button.pack(side=tk.LEFT, padx=(0, 6))
        cancel_button = ttk.Button(buttons, text="Отмена", style="Secondary.TButton", command=win.destroy)
        cancel_button.pack(side=tk.LEFT)
        retry_button = ttk.Button(buttons, text="Повторить", style="Secondary.TButton")

        def set_inputs_enabled(enabled: bool) -> None:
            state = "normal" if enabled else "disabled"
            email_entry.config(state=state)
            password_entry.config(state=state)
            sync_button.config(state=state)

        def start_progress() -> None:
            progress.pack(fill=tk.X, pady=(6, 0))
            progress.start(10)

        def stop_progress() -> None:
            progress.stop()
            progress.pack_forget()

        def reset_for_retry() -> None:
            status_var.set("")
            retry_button.pack_forget()
            set_inputs_enabled(True)
            email_entry.focus_set()

        def handle_success(token: str) -> None:
            stop_progress()
            status_var.set("Синхронизирован")
            self._set_sync_token(token, email_var.get().strip())
            win.after(400, lambda: (win.destroy(), self.open_sync_deck_window()))

        def handle_error(message: str) -> None:
            stop_progress()
            status_var.set(message)
            set_inputs_enabled(False)
            retry_button.pack(side=tk.LEFT, padx=(6, 0))

        def do_login() -> None:
            email = email_var.get().strip()
            password = password_var.get()
            if not email or not password:
                status_var.set("Введите email и пароль.")
                return
            set_inputs_enabled(False)
            cancel_button.config(state="disabled")
            retry_button.pack_forget()
            status_var.set("Синхронизация...")
            start_progress()

            def worker():
                try:
                    token = self.sync_client.login(email, password)
                    if not token:
                        raise ValueError("Не синхронизирован")
                    win.after(0, lambda: handle_success(token))
                except Exception:
                    win.after(0, lambda: handle_error("Не синхронизирован"))
                finally:
                    win.after(0, lambda: cancel_button.config(state="normal"))

            threading.Thread(target=worker, daemon=True).start()

        sync_button.config(command=do_login)
        retry_button.config(command=reset_for_retry)
        email_entry.focus_set()

    def _build_deck_payload(self, deck_id: int) -> dict:
        decks = list_decks()
        deck = next((item for item in decks if item["id"] == deck_id), None)
        cards = get_cards_in_deck(deck_id)
        max_cards = 200
        cards_payload = []
        for card in cards[:max_cards]:
            cards_payload.append(
                {
                    "id": card["id"],
                    "front": card["front"],
                    "back": card["back"],
                    "media": {
                        "front_image_path": card["front_image_path"],
                        "back_image_path": card["back_image_path"],
                        "image_path": card["image_path"],
                        "audio_path": card["audio_path"],
                    },
                }
            )
        # TODO: добавить постраничную отправку, если карточек больше лимита.
        return {
            "deck_id": deck_id,
            "deck_name": deck["name"] if deck else "",
            "deck_description": deck["description"] if deck else "",
            "cards": cards_payload,
        }

    def open_sync_deck_window(self) -> None:
        if not self.sync_token:
            self.open_sync_login_window()
            return

        win = tk.Toplevel(self)
        win.title("Синхронизация колоды")
        win.transient(self)
        win.resizable(False, False)

        container = ttk.Frame(win, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        decks = list_decks()
        deck_names = [deck["name"] for deck in decks]
        deck_lookup = {deck["name"]: deck["id"] for deck in decks}

        ttk.Label(container, text="Выберите колоду", style="HeaderSub.TLabel").pack(anchor="w")
        deck_var = tk.StringVar(value=deck_names[0] if deck_names else "")
        deck_combo = ttk.Combobox(container, values=deck_names, textvariable=deck_var, state="readonly", width=32)
        deck_combo.pack(fill=tk.X, pady=(6, 8))

        status_var = tk.StringVar(value="")
        status_label = ttk.Label(container, textvariable=status_var, style="Muted.TLabel")
        status_label.pack(anchor="w")
        progress = ttk.Progressbar(container, mode="indeterminate")

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(10, 0))
        sync_button = ttk.Button(buttons, text="Синхронизировать колоду", style="Primary.TButton")
        sync_button.pack(side=tk.LEFT)
        retry_button = ttk.Button(buttons, text="Повторить", style="Secondary.TButton")

        def set_inputs_enabled(enabled: bool) -> None:
            state = "normal" if enabled else "disabled"
            deck_combo.config(state="readonly" if enabled else "disabled")
            sync_button.config(state=state)

        def start_progress() -> None:
            progress.pack(fill=tk.X, pady=(6, 0))
            progress.start(10)

        def stop_progress() -> None:
            progress.stop()
            progress.pack_forget()

        def reset_for_retry() -> None:
            status_var.set("")
            retry_button.pack_forget()
            set_inputs_enabled(True)

        def handle_error() -> None:
            stop_progress()
            status_var.set("Статус: не синхронизирован")
            set_inputs_enabled(False)
            retry_button.pack(side=tk.LEFT, padx=(6, 0))

        def handle_success() -> None:
            stop_progress()
            status_var.set("Статус: синхронизирован")
            set_inputs_enabled(True)

        def handle_unauthorized() -> None:
            stop_progress()
            status_var.set("Статус: не синхронизирован")
            self._clear_sync_token()
            messagebox.showwarning("Синхронизация", "Сессия истекла. Войдите снова.")
            win.destroy()
            self.open_sync_login_window()

        def do_sync() -> None:
            deck_name = deck_var.get()
            deck_id = deck_lookup.get(deck_name)
            if not deck_id:
                status_var.set("Выберите колоду.")
                return
            retry_button.pack_forget()
            set_inputs_enabled(False)
            status_var.set("Синхронизация...")
            start_progress()

            def worker():
                token = self.sync_token
                try:
                    payload = self._build_deck_payload(deck_id)
                    result = self.sync_client.push_deck(token, payload)
                    unauthorized = not result and not (token and token.startswith("mock-token"))
                    if result:
                        win.after(0, handle_success)
                    elif unauthorized:
                        win.after(0, handle_unauthorized)
                    else:
                        win.after(0, handle_error)
                except Exception:
                    win.after(0, handle_error)

            threading.Thread(target=worker, daemon=True).start()

        sync_button.config(command=do_sync)
        retry_button.config(command=reset_for_retry)

        if not deck_names:
            status_var.set("Нет колод для синхронизации.")
            set_inputs_enabled(False)

    def copy_ref_link(self):
        link = self.ref_summary_vars.get("link").get() if self.ref_summary_vars else ""
        if not link:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(link)
            messagebox.showinfo("Реферальная ссылка", "Ссылка скопирована в буфер обмена.")
        except Exception as exc:
            messagebox.showerror("Буфер обмена", f"Не удалось скопировать: {exc}")

    def copy_user_id(self):
        user_id = self.user_id_var.get()
        if not user_id:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(user_id)
            messagebox.showinfo("User ID", "ID скопирован в буфер обмена.")
        except Exception as exc:
            messagebox.showerror("Буфер обмена", f"Не удалось скопировать: {exc}")

    def refresh_decks(self):
        if self._refresh_in_progress:
            return

        self._refresh_in_progress = True

        def task(progress_cb):
            decks_data = list_decks()
            stats_by_deck: dict[int, dict] = {}
            total = len(decks_data)
            for idx, deck in enumerate(decks_data, start=1):
                stats_by_deck[deck["id"]] = get_deck_stats(deck["id"])
                if total and (idx % max(1, total // 10 or 1) == 0 or idx == total):
                    progress_cb(idx, total, f"Загрузка колод {idx}/{total}")
            return {"decks": decks_data, "stats": stats_by_deck}

        def on_success(result):
            self.decks = result["decks"]
            self.deck_items = {}
            self.deck_icons = {}
            self.deck_preview_images = {}

            for item in self.decks_tree.get_children():
                self.decks_tree.delete(item)

            for deck in self.decks:
                icon = None
                if deck["icon_path"] and os.path.exists(deck["icon_path"]) and PIL_AVAILABLE:
                    try:
                        img = Image.open(deck["icon_path"])
                        img = img.resize((16, 16), _pil_lanczos())
                        icon = ImageTk.PhotoImage(img)
                        self.deck_icons[deck["id"]] = icon
                    except Exception:
                        pass

                desc = deck["description"] or "без описания"
                deck_text = f"{deck['name']} ({desc})"

                if icon:
                    root_id = self.decks_tree.insert("", "end", text=deck_text, image=icon, open=False)
                else:
                    root_id = self.decks_tree.insert("", "end", text=deck_text, open=False)

                self.deck_items[root_id] = (deck["id"], None)

                stats = result["stats"].get(deck["id"], {"phase_stats": {}, "total": 0})
                for phase in range(1, 11):
                    phase_count = stats.get("phase_stats", {}).get(phase, 0)
                    total_cards = stats.get("total", 0)
                    percentage = (phase_count / total_cards * 100) if total_cards > 0 else 0

                    child_text = f"Фаза {phase}: {phase_count} карт. ({percentage:.1f}%)"
                    child_id = self.decks_tree.insert(root_id, "end", text=child_text)
                    self.deck_items[child_id] = (deck["id"], phase)

            self.selected_deck_id = None
            self.selected_phase = None
            self.after(50, self.update_overdue_badge)
            self.update_deck_preview()
            self._refresh_in_progress = False

        def on_error(exc):
            messagebox.showerror("Ошибка", str(exc))
            self._refresh_in_progress = False

        self.run_task("Загрузка колод", "determinate", task, on_success, on_error)

    def update_deck_preview(self):
        """Обновить превью выбранной колоды."""
        # Очищаем текущее превью
        self.deck_preview_label.config(image="", text="Выберите колоду для просмотра")
        self.deck_name_label.config(text="Название колоды")
        self.deck_desc_label.config(text="Описание колоды")
        self.deck_stats_label.config(text="Карточек: 0\nФаз: 0/10\nОзнакомлено: 0")
        
        if self.selected_deck_id is None:
            return
        
        # Находим выбранную колоду
        selected_deck = None
        for d in self.decks:
            if d["id"] == self.selected_deck_id:
                selected_deck = d
                break
        
        if not selected_deck:
            return
        
        # Обновляем название и описание
        self.deck_name_label.config(text=selected_deck["name"])
        self.deck_desc_label.config(text=selected_deck["description"] or "Без описания")
        
        # Загружаем и отображаем изображение колоды
        if selected_deck["icon_path"] and os.path.exists(selected_deck["icon_path"]) and PIL_AVAILABLE:
            try:
                img = Image.open(selected_deck["icon_path"])
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                # Автоматически уменьшаем окно до размеров картинки
                img_width, img_height = img.size
                
                # Масштабируем для превью, но не слишком сильно
                max_width = 300
                max_height = 200
                
                display_width, display_height = img_width, img_height
                if img_width > max_width or img_height > max_height:
                    ratio = min(max_width / img_width, max_height / img_height)
                    display_width = int(img_width * ratio)
                    display_height = int(img_height * ratio)
                    img = img.resize((display_width, display_height), _pil_lanczos())
                
                photo = ImageTk.PhotoImage(img)
                self.deck_preview_images[self.selected_deck_id] = photo
                self.deck_preview_label.config(image=photo, text="")
                
                # Устанавливаем размер окна превью под изображение
                self.image_frame.config(width=display_width, height=display_height)
                self.deck_preview_label.config(width=display_width, height=display_height)
                
            except Exception as e:
                self.deck_preview_label.config(
                    image="", 
                    text=f"Ошибка загрузки изображения\n{str(e)}"
                )
        else:
            self.deck_preview_label.config(
                image="", 
                text="Изображение не загружено.\nУстановите Pillow, чтобы добавить превью."
            )
        
        # Обновляем статистику
        stats = get_deck_stats(self.selected_deck_id)
        phases_with_cards = sum(1 for phase in range(1, 11) if stats["phase_stats"].get(phase, 0) > 0)
        self.deck_stats_label.config(
            text=f"Карточек: {stats['total']}\n"
                 f"Фаз: {phases_with_cards}/10\n"
                 f"Изучено: {stats['learned_percent']:.1f}%\n"
                 f"Ознакомлено: {stats['total_overview']}"
        )

    def update_overdue_badge(self):
        """Обновить бейджи просрочек для выбранной колоды и фаз."""
        if self._overdue_task_running:
            return

        deck_ids = {deck_id for deck_id, _ in self.deck_items.values() if deck_id is not None}
        total = len(deck_ids)
        self._overdue_task_running = True

        def task(progress_cb):
            counts_by_deck: dict[int | None, PhaseOverdueBadges] = {}
            timestamp = int(time.time())
            for idx, deck_id in enumerate(deck_ids, start=1):
                counts_by_deck[deck_id] = fetch_overdue_counts_by_phase(
                    None, deck_id, now_ts=timestamp
                )
                if total and (idx % max(1, total // 10 or 1) == 0 or idx == total):
                    progress_cb(idx, total, f"Просрочки {idx}/{total}")
            return counts_by_deck

        def on_success(counts_by_deck: dict[int | None, PhaseOverdueBadges]):
            selected_counts = counts_by_deck.get(self.selected_deck_id)
            total_count = selected_counts.total if selected_counts else 0

            if self.overdue_canvas is not None:
                self.overdue_canvas.delete("all")
                if total_count > 0:
                    self.overdue_canvas.create_oval(2, 2, 22, 22, fill=self.palette.get("accent", "#3B82F6"), outline=self.palette.get("accent", "#3B82F6"))
                    self.overdue_badge_text_id = self.overdue_canvas.create_text(
                        12, 12, text=str(total_count), fill=self.palette.get("text", "white"), font=("Segoe UI", 10, "bold")
                    )

            if self.phase_badge_manager:
                self.phase_badge_manager.update(self.deck_items, counts_by_deck)
            self._overdue_task_running = False
            self.schedule_overdue_badges_refresh()

        def on_error(exc: Exception):
            error_message = str(exc)
            os.makedirs("logs", exist_ok=True)
            with open(os.path.join("logs", "db.log"), "a", encoding="utf-8") as log_file:
                log_file.write(f"[{datetime.now().isoformat()}] update_overdue_badge: {error_message}\n")

            if "unable to open database file" in error_message.lower():
                counts_by_deck: dict[int | None, PhaseOverdueBadges] = {}
                if self.overdue_canvas is not None:
                    self.overdue_canvas.delete("all")
                if self.phase_badge_manager:
                    self.phase_badge_manager.update(self.deck_items, counts_by_deck)
            else:
                messagebox.showerror("БД", error_message)
            self._overdue_task_running = False
            self.schedule_overdue_badges_refresh()

        self.run_task("Просроченные карточки", "determinate", task, on_success, on_error, total=total or None)

    def schedule_overdue_badges_refresh(self):
        if self.overdue_update_job is not None:
            self.after_cancel(self.overdue_update_job)
        self.overdue_update_job = self.after(60_000, self.update_overdue_badge)

    def on_deck_select(self, event):
        if self._deck_select_job:
            self.after_cancel(self._deck_select_job)
        self._deck_select_job = self.after(150, self._apply_selected_deck)

    def _apply_selected_deck(self):
        self._deck_select_job = None
        sel = self.decks_tree.selection()
        if not sel:
            self.selected_deck_id = None
            self.selected_phase = None
            self.load_templates_for_selected_deck()
            self.update_overdue_badge()
            self.update_deck_preview()
            return

        item_id = sel[0]
        deck_id, phase = self.deck_items.get(item_id, (None, None))
        self.selected_deck_id = deck_id
        self.selected_phase = phase
        if self.mw_context is not None:
            self.mw_context.state["current_deck_id"] = deck_id
        if deck_id is not None:
            gui_hooks.deck_will_open(deck_id)
        self.load_templates_for_selected_deck()
        self.update_overdue_badge()
        self.update_deck_preview()

    def load_templates_for_selected_deck(self):
        if self.selected_deck_id is None:
            self.front_template = DEFAULT_FRONT_TEMPLATE
            self.back_template = DEFAULT_BACK_TEMPLATE
            return
        try:
            active = get_active_generation_template(self.selected_deck_id)
            front, back = active.get("front_template"), active.get("back_template")
        except Exception:
            front, back = DEFAULT_FRONT_TEMPLATE, DEFAULT_BACK_TEMPLATE
        self.front_template = front or DEFAULT_FRONT_TEMPLATE
        self.back_template = back or DEFAULT_BACK_TEMPLATE

    def _build_generation_template_selector(
        self,
        parent: tk.Widget,
        deck_id: int | None,
        padx: int = 5,
        pady: tuple[int, int] = (5, 5),
    ) -> dict[str, object]:
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=padx, pady=pady)
        ttk.Label(container, text="Выберите шаблон карточки").pack(anchor="w")
        row = ttk.Frame(container)
        row.pack(fill=tk.X, pady=(2, 0))

        template_var = tk.StringVar()
        combo = ttk.Combobox(row, textvariable=template_var, state="readonly")
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        template_rows: list[sqlite3.Row] = []

        def refresh() -> None:
            nonlocal template_rows
            ensure_default_generation_template(deck_id)
            template_rows = list_generation_templates()
            names = [row["name"] for row in template_rows]
            combo.configure(values=names)
            selected_id = get_deck_selected_template_id(deck_id) if deck_id else None
            name_by_id = {row["id"]: row["name"] for row in template_rows}
            if selected_id in name_by_id:
                template_var.set(name_by_id[selected_id])
            elif template_var.get() not in names:
                template_var.set("")

        def on_select(_event=None) -> None:
            if deck_id is None:
                return
            name = template_var.get()
            template_id = next((row["id"] for row in template_rows if row["name"] == name), None)
            if template_id is not None:
                set_deck_selected_template_id(deck_id, template_id)
                self.load_templates_for_selected_deck()

        def open_manager() -> None:
            selected_id = next(
                (row["id"] for row in template_rows if row["name"] == template_var.get()),
                None,
            )
            self.open_generation_templates_window(selected_template_id=selected_id, on_close=refresh)

        combo.bind("<<ComboboxSelected>>", on_select)
        ttk.Button(row, text="Открыть менеджер шаблонов", command=open_manager).pack(side=tk.LEFT, padx=6)
        refresh()
        return {"frame": container, "combobox": combo, "refresh": refresh, "var": template_var}

    # --------- новая колода ---------

    def add_deck_window(self):
        win = tk.Toplevel(self)
        win.after(0, lambda w=win: apply_window_icon(w, self._logo_big, self._ico_path))
        win.title("Новая колода")
        win.geometry("400x340")
        win.grab_set()

        ttk.Label(win, text="Название колоды:").pack(anchor="w", padx=10, pady=(10, 0))
        entry_name = ttk.Entry(win)
        entry_name.pack(fill=tk.X, padx=10)
        create_context_menu(entry_name)  # Добавляем контекстное меню

        ttk.Label(win, text="Описание (необязательно):").pack(anchor="w", padx=10, pady=(10, 0))
        entry_desc = ttk.Entry(win)
        entry_desc.pack(fill=tk.X, padx=10)
        create_context_menu(entry_desc)  # Добавляем контекстное меню

        ttk.Label(win, text="Язык озвучки (TTS):").pack(anchor="w", padx=10, pady=(10, 0))
        tts_lang_var = tk.StringVar(value="de")
        entry_tts_lang = ttk.Entry(win, textvariable=tts_lang_var)
        entry_tts_lang.pack(fill=tk.X, padx=10)
        create_context_menu(entry_tts_lang)

        # Иконка колоды
        icon_path_var = tk.StringVar()
        
        icon_frame = ttk.Frame(win)
        icon_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(icon_frame, text="Изображение колоды:").pack(side=tk.LEFT)
        lbl_icon = ttk.Label(icon_frame, text="не выбрана")
        lbl_icon.pack(side=tk.LEFT, padx=5)
        
        def select_icon():
            filetypes = [
                ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"),
                ("Все файлы", "*.*"),
            ]
            filename = filedialog.askopenfilename(
                title="Выбрать изображение для колоды",
                filetypes=filetypes
            )
            if filename:
                icon_path_var.set(filename)
                lbl_icon.config(text=os.path.basename(filename))

        ttk.Button(icon_frame, text="Выбрать", command=select_icon).pack(side=tk.RIGHT, padx=5)

        def save_deck():
            name = entry_name.get().strip()
            desc = entry_desc.get().strip()
            icon_path = icon_path_var.get().strip() or None
            tts_lang = tts_lang_var.get().strip() or "de"

            if not name:
                messagebox.showerror("Ошибка", "Название не может быть пустым.")
                return

            conn = get_connection()
            cur = conn.cursor()
            deck_id = None
            try:
                cur.execute(
                    """INSERT INTO decks
                       (name, description, front_template, back_template, icon_path, tts_lang)
                       VALUES (?, ?, ?, ?, ?, ?);""",
                    (name, desc or None, self.front_template, self.back_template, icon_path, tts_lang)
                )
                deck_id = cur.lastrowid
            except sqlite3.OperationalError:
                cur.execute(
                    "INSERT INTO decks (name, description, icon_path, tts_lang) VALUES (?, ?, ?, ?);",
                    (name, desc or None, icon_path, tts_lang)
                )
                deck_id = cur.lastrowid
            if deck_id:
                ensure_deck_settings_row(deck_id, conn, inherit_default=1)
            conn.commit()
            conn.close()
            self.refresh_decks()
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Сохранить", command=save_deck).pack(side=tk.RIGHT)

    # --------- редактирование колоды ---------

    def edit_deck_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        win = tk.Toplevel(self)
        win.after(0, lambda w=win: apply_window_icon(w, self._logo_big, self._ico_path))
        win.title("Редактирование колоды")
        win.geometry("900x600")
        win.minsize(700, 450)
        win.grab_set()

        def log_deck_settings(message: str, exc: BaseException | None = None) -> None:
            try:
                with open("deck_settings_error.log", "a", encoding="utf-8") as log_file:
                    log_file.write(f"{datetime.now().isoformat()} {message}\n")
                    if exc is not None:
                        log_file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                        log_file.write("\n")
            except Exception:
                pass

        def build_ui() -> None:
            log_deck_settings("build_ui started")
            try:
                # Получаем текущие данные колоды
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(
                    "SELECT name, description, icon_path, tts_lang FROM decks WHERE id = ?;",
                    (self.selected_deck_id,),
                )
                deck_data = cur.fetchone()
                conn.close()

                if not deck_data:
                    raise ValueError("Deck not found for editor UI.")

                try:
                    timer_settings = get_deck_timer_settings(self.selected_deck_id)
                except Exception as e:
                    # Fallback for legacy deck_timer / older DB schema (pre user_phase_intervals).
                    try:
                        log_deck_settings("get_deck_timer_settings failed, using defaults", e)
                    except Exception:
                        pass
                    try:
                        conn_fix = get_connection()
                        ensure_deck_settings_table(conn_fix)
                        ensure_deck_settings_row(self.selected_deck_id, conn_fix)
                        try:
                            conn_fix.commit()
                        except Exception:
                            pass
                        try:
                            conn_fix.close()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    timer_settings = {
                        "timer_sec": 0,
                        "timer_mode": "reveal",
                        "inherit_timer": 1,
                        "review_timer_seconds": None,
                        "playback_timer_seconds": None,
                        "user_phase_intervals": None,
                    }

                try:
                    phase_intervals = get_deck_phase_intervals(self.selected_deck_id)
                except Exception as e:
                    try:
                        log_deck_settings("get_deck_phase_intervals failed, using defaults", e)
                    except Exception:
                        pass
                    phase_intervals = list(DEFAULT_PHASE_INTERVALS)

                outer = ttk.Frame(win)
                outer.pack(fill="both", expand=True)

                btns = ttk.Frame(outer)
                btns.pack(side="bottom", fill="x", padx=10, pady=10)

                canvas = tk.Canvas(outer, highlightthickness=0, bd=0)
                vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
                canvas.configure(yscrollcommand=vbar.set)

                vbar.pack(side="right", fill="y")
                canvas.pack(side="left", fill="both", expand=True)

                content = ttk.Frame(canvas)
                content_id = canvas.create_window((0, 0), window=content, anchor="nw")

                def _on_content_configure(event=None):
                    canvas.configure(scrollregion=canvas.bbox("all"))

                def _on_canvas_configure(event):
                    canvas.itemconfigure(content_id, width=event.width)

                def _on_mousewheel(e):
                    canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

                content.bind("<Configure>", _on_content_configure)
                canvas.bind("<Configure>", _on_canvas_configure)
                canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
                canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

                main_frame = ttk.Frame(content, padding=12)
                main_frame.pack(fill=tk.BOTH, expand=True)

                ttk.Label(main_frame, text="Deck editor loaded OK").pack(anchor="w", pady=(0, 8))

                fields_frame = ttk.LabelFrame(main_frame, text="Поля колоды")
                fields_frame.pack(fill=tk.X, pady=(0, 10))
                fields_frame.columnconfigure(1, weight=1)

                ttk.Label(fields_frame, text="Название колоды:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
                entry_name = ttk.Entry(fields_frame)
                entry_name.insert(0, deck_data["name"])
                entry_name.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
                create_context_menu(entry_name)

                ttk.Label(fields_frame, text="Описание:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
                entry_desc = ttk.Entry(fields_frame)
                entry_desc.insert(0, deck_data["description"] or "")
                entry_desc.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
                create_context_menu(entry_desc)

                ttk.Label(fields_frame, text="Язык озвучки (TTS):").grid(row=2, column=0, sticky="w", padx=6, pady=6)
                tts_lang_var = tk.StringVar(value=deck_data["tts_lang"] or "de")
                entry_tts_lang = ttk.Entry(fields_frame, textvariable=tts_lang_var)
                entry_tts_lang.grid(row=2, column=1, sticky="ew", padx=6, pady=6)
                create_context_menu(entry_tts_lang)

                icon_path_var = tk.StringVar(value=deck_data["icon_path"] or "")
                icon_row = ttk.Frame(fields_frame)
                icon_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
                icon_row.columnconfigure(1, weight=1)
                ttk.Label(icon_row, text="Изображение колоды:").grid(row=0, column=0, sticky="w")
                lbl_icon = ttk.Label(
                    icon_row,
                    text=os.path.basename(icon_path_var.get()) if icon_path_var.get() else "не выбрана",
                )
                lbl_icon.grid(row=0, column=1, sticky="w", padx=5)

                def select_icon():
                    filetypes = [
                        ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"),
                        ("Все файлы", "*.*"),
                    ]
                    filename = filedialog.askopenfilename(
                        title="Выбрать изображение для колоды",
                        filetypes=filetypes,
                    )
                    if filename:
                        icon_path_var.set(filename)
                        lbl_icon.config(text=os.path.basename(filename))

                ttk.Button(icon_row, text="Выбрать", command=select_icon).grid(row=0, column=2, sticky="e")

                timer_frame = ttk.LabelFrame(main_frame, text="Таймер колоды")
                timer_frame.pack(fill=tk.X, pady=(0, 10))

                ttk.Label(timer_frame, text="Секунды:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
                timer_sec_var = tk.IntVar(value=timer_settings.get("timer_sec") or 0)
                timer_spin = ttk.Spinbox(timer_frame, from_=0, to=3600, textvariable=timer_sec_var, width=10)
                timer_spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")

                ttk.Label(timer_frame, text="Режим:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
                timer_mode_var = tk.StringVar(value=(timer_settings.get("timer_mode") or "reveal"))
                mode_combo = ttk.Combobox(
                    timer_frame,
                    state="readonly",
                    values=["reveal", "fail", "notify"],
                    textvariable=timer_mode_var,
                    width=12,
                )
                mode_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")

                inherit_var = tk.BooleanVar(value=bool(timer_settings.get("inherit_timer", 1)))
                inherit_cb = ttk.Checkbutton(
                    timer_frame,
                    text="Наследовать от родителя, если свой таймер пуст",
                    variable=inherit_var,
                )
                inherit_cb.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

                mode_timer_frame = ttk.LabelFrame(main_frame, text="Таймеры режимов")
                mode_timer_frame.pack(fill=tk.X, pady=(0, 10))

                ttk.Label(mode_timer_frame, text="Таймер повторения (сек):").grid(
                    row=0, column=0, padx=5, pady=5, sticky="w"
                )
                review_timer_var = tk.StringVar(
                    value=(
                        "" if timer_settings.get("review_timer_seconds") is None
                        else str(timer_settings.get("review_timer_seconds"))
                    )
                )
                ttk.Entry(mode_timer_frame, textvariable=review_timer_var, width=10).grid(
                    row=0, column=1, padx=5, pady=5, sticky="w"
                )

                ttk.Label(mode_timer_frame, text="Таймер воспроизведения (сек):").grid(
                    row=1, column=0, padx=5, pady=5, sticky="w"
                )
                playback_timer_var = tk.StringVar(
                    value=(
                        "" if timer_settings.get("playback_timer_seconds") is None
                        else str(timer_settings.get("playback_timer_seconds"))
                    )
                )
                ttk.Entry(mode_timer_frame, textvariable=playback_timer_var, width=10).grid(
                    row=1, column=1, padx=5, pady=5, sticky="w"
                )

                ttk.Label(
                    mode_timer_frame,
                    text="Пустое поле = наследовать от родительской колоды (если есть)",
                    foreground="gray",
                ).grid(row=2, column=0, columnspan=2, padx=5, pady=(0, 5), sticky="w")

                phase_frame = ttk.LabelFrame(main_frame, text="Интервалы повторений по фазам")
                phase_frame.pack(fill=tk.BOTH, pady=(0, 10), expand=True)
                phase_frame.columnconfigure(1, weight=1)

                phase_vars = []
                for idx in range(10):
                    seconds = phase_intervals[idx] if idx < len(phase_intervals) else DEFAULT_PHASE_INTERVALS[idx]
                    days = int(seconds // 86400)
                    hours = int((seconds % 86400) // 3600)
                    days_var = tk.IntVar(value=days)
                    hours_var = tk.IntVar(value=hours)
                    phase_vars.append((days_var, hours_var))
                    ttk.Label(phase_frame, text=f"Фаза {idx + 1}:").grid(row=idx, column=0, padx=5, pady=3, sticky="w")
                    ttk.Label(phase_frame, text="дней").grid(row=idx, column=2, padx=2, sticky="w")
                    ttk.Label(phase_frame, text="часов").grid(row=idx, column=4, padx=2, sticky="w")
                    ttk.Spinbox(phase_frame, from_=0, to=999, textvariable=days_var, width=6).grid(
                        row=idx, column=1, padx=(5, 2), pady=3, sticky="w"
                    )
                    ttk.Spinbox(phase_frame, from_=0, to=23, textvariable=hours_var, width=6).grid(
                        row=idx, column=3, padx=(5, 2), pady=3, sticky="w"
                    )

                def reset_phase_intervals():
                    for idx, (days_var, hours_var) in enumerate(phase_vars):
                        seconds = DEFAULT_PHASE_INTERVALS[idx]
                        days_var.set(int(seconds // 86400))
                        hours_var.set(int((seconds % 86400) // 3600))
                    reset_deck_phase_intervals(self.selected_deck_id)

                def log_deck_save_error(exc: BaseException) -> None:
                    try:
                        with open("deck_editor_save_error.log", "a", encoding="utf-8") as log_file:
                            log_file.write(f"{datetime.now().isoformat()} save_changes failed\n")
                            log_file.write(
                                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                            )
                            log_file.write("\n")
                    except Exception:
                        pass

                def save_changes():
                    try:
                        name = entry_name.get().strip()
                        desc = entry_desc.get().strip()
                        icon_path = icon_path_var.get().strip() or None
                        tts_lang = tts_lang_var.get().strip() or "de"

                        if not name:
                            messagebox.showerror("Ошибка", "Название не может быть пустым.")
                            return

                        try:
                            timer_sec = int(timer_sec_var.get())
                        except tk.TclError:
                            messagebox.showerror("Ошибка", "Некорректное значение таймера.")
                            return

                        def parse_mode_timer(raw_value: str, label: str) -> int | None:
                            value = raw_value.strip()
                            if value == "":
                                return None
                            try:
                                parsed = int(value)
                            except ValueError:
                                messagebox.showerror("Ошибка", f"{label}: значение должно быть целым числом.")
                                return None
                            if parsed < 0:
                                messagebox.showerror("Ошибка", f"{label}: значение должно быть >= 0.")
                                return None
                            return parsed

                        review_timer_seconds = parse_mode_timer(
                            review_timer_var.get(), "Таймер повторения"
                        )
                        if review_timer_seconds is None and review_timer_var.get().strip():
                            return

                        playback_timer_seconds = parse_mode_timer(
                            playback_timer_var.get(), "Таймер воспроизведения"
                        )
                        if playback_timer_seconds is None and playback_timer_var.get().strip():
                            return

                        intervals = []
                        for days_var, hours_var in phase_vars:
                            try:
                                days_val = int(days_var.get())
                                hours_val = int(hours_var.get())
                            except tk.TclError:
                                messagebox.showerror("Ошибка", "Интервалы фаз должны быть числами.")
                                return
                            if days_val < 0 or hours_val < 0:
                                messagebox.showerror("Ошибка", "Интервалы фаз должны быть >= 0.")
                                return
                            seconds = max(0, days_val * 86400 + hours_val * 3600)
                            intervals.append(seconds)

                        conn = get_connection()
                        cur = conn.cursor()
                        ensure_deck_settings_table(conn)
                        ensure_deck_settings_row(self.selected_deck_id, conn)
                        cur.execute("PRAGMA table_info(decks);")
                        deck_columns = {row["name"] for row in cur.fetchall()}
                        updates = []
                        params = []
                        if "name" in deck_columns:
                            updates.append("name = ?")
                            params.append(name)
                        if "description" in deck_columns:
                            updates.append("description = ?")
                            params.append(desc or None)
                        if "icon_path" in deck_columns:
                            updates.append("icon_path = ?")
                            params.append(icon_path)
                        if "tts_lang" in deck_columns:
                            updates.append("tts_lang = ?")
                            params.append(tts_lang)
                        if updates:
                            params.append(self.selected_deck_id)
                            cur.execute(
                                f"UPDATE decks SET {', '.join(updates)} WHERE id = ?;",
                                params,
                            )
                        try:
                            update_deck_timer_settings(
                                self.selected_deck_id,
                                timer_sec,
                                timer_mode_var.get(),
                                1 if inherit_var.get() else 0,
                                review_timer_seconds,
                                playback_timer_seconds,
                                conn,
                            )
                        except Exception as exc:
                            log_deck_settings("update_deck_timer_settings failed", exc)
                        try:
                            save_deck_phase_intervals(self.selected_deck_id, intervals, conn)
                        except Exception as exc:
                            log_deck_settings("save_deck_phase_intervals failed", exc)
                        conn.commit()
                        conn.close()
                        if hasattr(self, "refresh_decks"):
                            self.refresh_decks()
                        messagebox.showinfo("Сохранено", "Настройки колоды сохранены")
                    except Exception as exc:
                        log_deck_settings("save_changes exception", exc)
                        log_deck_save_error(exc)
                        messagebox.showerror("Ошибка", traceback.format_exc())

                ttk.Button(
                    btns, text="Сбросить настройки сроков", command=reset_phase_intervals
                ).pack(side=tk.LEFT)
                ttk.Button(btns, text="Закрыть", command=win.destroy).pack(side=tk.RIGHT, padx=6)
                ttk.Button(btns, text="Сохранить", style="Primary.TButton", command=save_changes).pack(
                    side=tk.RIGHT, padx=6
                )
            except Exception as exc:
                log_deck_settings("build_ui exception", exc)
                messagebox.showerror(
                    "Ошибка",
                    traceback.format_exc(),
                )
            finally:
                log_deck_settings("build_ui finished")

        build_ui()

    def delete_selected_deck(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        conn = open_db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM decks WHERE id = ?;", (self.selected_deck_id,))
            deck_row = cur.fetchone()
            if not deck_row:
                messagebox.showerror("Ошибка", "Не удалось найти выбранную колоду.")
                return

            deck_name = deck_row["name"] or "Без названия"

            cur.execute("PRAGMA table_info(decks);")
            deck_columns = {row["name"] for row in cur.fetchall()}
            has_parent_column = "parent_id" in deck_columns

            def collect_child_decks(parent_id: int) -> list[int]:
                if not has_parent_column:
                    return []
                cur.execute("SELECT id FROM decks WHERE parent_id = ?;", (parent_id,))
                children = [row["id"] for row in cur.fetchall()]
                result: list[int] = []
                for child_id in children:
                    result.append(child_id)
                    result.extend(collect_child_decks(child_id))
                return result

            child_deck_ids = collect_child_decks(self.selected_deck_id)
            if child_deck_ids:
                confirm_children = messagebox.askyesno(
                    "Есть подкалоды",
                    "У выбранной колоды есть подкалоды. Удалить их тоже?",
                )
                if not confirm_children:
                    messagebox.showwarning("Подколоды", "Сначала удалите подкалоды.")
                    return

            confirm = messagebox.askyesno(
                "Удалить колоду",
                (
                    f"Удалить колоду '{deck_name}'? Это удалит все карточки/ноты/медиа "
                    "этой колоды. Отменить нельзя."
                ),
            )
            if not confirm:
                return

            deck_ids_to_delete = [self.selected_deck_id, *child_deck_ids]
            media_table_exists = _table_exists(conn, "media")

            try:
                with DB_WRITE_LOCK:
                    conn.execute("BEGIN;")
                    for deck_id in deck_ids_to_delete:
                        cur.execute("SELECT id FROM notes WHERE deck_id = ?;", (deck_id,))
                        note_ids = [row["id"] for row in cur.fetchall()]
                        cur.execute("SELECT id FROM cards WHERE deck_id = ?;", (deck_id,))
                        card_ids = [row["id"] for row in cur.fetchall()]

                        if media_table_exists:
                            if note_ids:
                                placeholders = ",".join("?" * len(note_ids))
                                cur.execute(
                                    f"DELETE FROM media WHERE note_id IN ({placeholders});",
                                    note_ids,
                                )
                            if card_ids:
                                placeholders = ",".join("?" * len(card_ids))
                                cur.execute(
                                    f"DELETE FROM media WHERE card_id IN ({placeholders});",
                                    card_ids,
                                )

                        cur.execute("DELETE FROM cards WHERE deck_id = ?;", (deck_id,))
                        cur.execute("DELETE FROM notes WHERE deck_id = ?;", (deck_id,))
                        cur.execute("DELETE FROM decks WHERE id = ?;", (deck_id,))

                    conn.commit()
            except Exception as e:
                conn.rollback()
                messagebox.showerror("Ошибка", f"Не удалось удалить колоду: {e}")
                return

            self.selected_deck_id = None
            self.refresh_decks()
            self.update_deck_preview()
            self.update_overdue_badge()
            messagebox.showinfo("Готово", "Колода удалена.")
        finally:
            conn.close()

    # --------- ручная карточка ---------

    def add_card_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду в списке.")
            return

        win = tk.Toplevel(self)
        win.title("Новая карточка (ручной режим)")
        win.geometry("950x760")
        win.minsize(900, 700)
        win.grab_set()
        win.after(0, lambda w=win: apply_window_icon(w, self._logo_big, self._ico_path))

        def log_ui_error(exc: BaseException) -> None:
            try:
                with open("ui_error.log", "a", encoding="utf-8") as log_file:
                    log_file.write(f"{datetime.now().isoformat()} manual card window\n")
                    log_file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                    log_file.write("\n")
            except Exception:
                pass

        try:
            colors = getattr(self, "palette", None) or {}
            apply_dark_theme_to_window(win, colors)
            background = colors.get("background", "#0B0D12")
            text_color = colors.get("text", "#E5E7EB")
            win.configure(bg=background)

            outer_frame = tk.Frame(win, bg=background)
            outer_frame.pack(fill=tk.BOTH, expand=True)

            bottom_actions = tk.Frame(outer_frame, bg=background)
            bottom_actions.pack(side="bottom", fill="x", padx=16, pady=12)

            canvas = tk.Canvas(outer_frame, bg=background, highlightthickness=0, bd=0)
            sb = tk.Scrollbar(
                outer_frame,
                orient="vertical",
                bg="#0b0f16",
                troughcolor="#05070b",
                activebackground="#121a26",
                highlightthickness=0,
                bd=0,
                width=12,
                command=canvas.yview,
            )
            canvas.configure(yscrollcommand=sb.set)

            sb.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)

            content = tk.Frame(canvas, bg=background)
            content_window = canvas.create_window((0, 0), window=content, anchor="nw")

            content.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
            )
            canvas.bind("<Configure>", lambda e: canvas.itemconfigure(content_window, width=e.width))

            def _mw(e):
                canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

            canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _mw))
            canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

            top_bar = tk.Frame(content, bg=background)
            top_bar.pack(fill=tk.X, padx=16, pady=(12, 0))
            left_controls = tk.Frame(top_bar, bg=background)
            left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(left_controls, text="Куда сохранить карточку:", bg=background, fg=text_color).pack(side=tk.LEFT)
            decks = list_decks()
            deck_map = {deck["name"]: deck["id"] for deck in decks}
            default_name = next((d["name"] for d in decks if d["id"] == self.selected_deck_id), "")
            deck_var = tk.StringVar(value=default_name)
            deck_combo = ttk.Combobox(left_controls, values=list(deck_map.keys()), textvariable=deck_var, state="readonly")
            deck_combo.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

            show_back_var = tk.BooleanVar(value=False)
            right_controls = tk.Frame(top_bar, bg=background)
            right_controls.pack(side=tk.RIGHT)
            dots_frame = tk.Frame(right_controls, bg=background)
            dots_frame.pack(side=tk.LEFT, padx=(0, 6))
            toggle_button = ttk.Button(right_controls, text="Показать обратную сторону")
            toggle_button.pack(side=tk.LEFT)

            format_bar = tk.Frame(content, bg=background)
            format_bar.pack(fill=tk.X, padx=16, pady=(10, 0))
            format_wrap = tk.Frame(format_bar, bg="white")
            format_wrap.pack(fill=tk.X)
            format_inner = tk.Frame(format_wrap, bg=background)
            format_inner.pack(fill=tk.X, padx=1, pady=1)

            card_surface_bg, card_text, card_border = get_card_surface_colors(self)

            content_row = tk.Frame(content, bg=background)
            content_row.pack(fill=tk.BOTH, expand=True)

            card_container = tk.Frame(content_row, bg=background)
            card_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=12)

            card_surface = tk.Frame(
                card_container,
                bg=card_surface_bg,
                highlightthickness=1,
                highlightbackground=card_border,
                relief=tk.FLAT,
            )
            card_surface.pack(fill=tk.BOTH, expand=True)

            layout = render_card_layout(
                card_surface,
                {"skip_text_widget": True, "skip_media_widget": True},
                editable=True,
            )
            text_frame = layout["text_frame"]
            media_frame = layout["media_frame"]
            audio_frame = layout["audio_frame"]

            text_frame.columnconfigure(0, weight=1)
            text_frame.rowconfigure(0, weight=1)

            front_text = tk.Text(text_frame, height=10, wrap=tk.WORD)
            back_text = tk.Text(text_frame, height=10, wrap=tk.WORD)
            style_card_surface_text(front_text, colors)
            style_card_surface_text(back_text, colors)
            front_text.grid(row=0, column=0, sticky="nsew")
            back_text.grid(row=0, column=0, sticky="nsew")
            front_text.tkraise()
            create_context_menu(front_text)
            create_context_menu(back_text)
            front_text.bind("<KeyRelease>", lambda _e: update_preview_if_open(full_refresh=False))
            back_text.bind("<KeyRelease>", lambda _e: update_preview_if_open(full_refresh=False))

            media_canvas = tk.Canvas(media_frame, bg="white", highlightthickness=0, height=260)
            media_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

            video_badge = tk.Label(
                media_frame,
                text="🎬 video: не прикреплено",
                bg=card_surface_bg,
                fg=card_text,
                anchor="w",
            )
            video_badge.pack(anchor="w", padx=6, pady=(0, 6))

            for widget in audio_frame.winfo_children():
                widget.destroy()
            audio_badge = tk.Label(
                audio_frame,
                text="🔊 audio: не прикреплено",
                bg=card_surface_bg,
                fg=card_text,
                anchor="w",
            )
            audio_badge.pack(side=tk.LEFT)

            right_panel = tk.Frame(content_row, bg=background, width=220)
            right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=12, pady=12)
            right_panel.pack_propagate(False)
            tk.Label(
                right_panel,
                text="Текущие изменения",
                bg=background,
                fg=text_color,
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w", padx=8, pady=(4, 6))
            log_box = tk.Text(
                right_panel,
                bg=background,
                fg=text_color,
                insertbackground=text_color,
                height=24,
                width=26,
                relief=tk.FLAT,
                wrap=tk.WORD,
            )
            log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            log_box.configure(state=tk.DISABLED)

            def set_status(msg: str) -> None:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_box.configure(state=tk.NORMAL)
                log_box.insert(tk.END, f"[{timestamp}] {msg}\n")
                log_box.see(tk.END)
                log_box.configure(state=tk.DISABLED)

            def _fmt_not_ready() -> None:
                set_status("Функция в разработке")

            manual_media = {
                "front": {"image": None, "video": None, "audio": None, "pos": None},
                "back": {"image": None, "video": None, "audio": None, "pos": None},
            }
            preview_state = {"window": None, "renderer": None, "side": "front", "slot_size": None}

            self._front_img_photo = None
            self._back_img_photo = None
            self._active_img_photo = None
            self._manual_media_photos = {"front": None, "back": None}
            self._manual_img_photo_front = None
            self._manual_img_photo_back = None
            self.manual_side = "front"

            def current_side() -> str:
                return self.manual_side

            def current_text_widget() -> tk.Text:
                return back_text if show_back_var.get() else front_text

            def apply_format(tag: str, **cfg) -> None:
                text_widget = current_text_widget()
                if tag not in text_widget.tag_names():
                    text_widget.tag_configure(tag, **cfg)
                try:
                    text_widget.tag_add(tag, "sel.first", "sel.last")
                except tk.TclError:
                    pass

            def _restore_manual_window() -> None:
                win.after(0, lambda: (win.deiconify(), win.lift(), win.focus_force()))
                win.after(10, lambda: win.grab_set())

            def _open_media_dialog(*, title: str, filetypes: list[tuple[str, str]]):
                try:
                    if win.grab_current() == win:
                        win.grab_release()
                except Exception:
                    pass
                try:
                    return filedialog.askopenfilename(
                        title=title,
                        filetypes=filetypes,
                        parent=win,
                    )
                finally:
                    _restore_manual_window()

            def _choose_color(title: str):
                try:
                    win.grab_release()
                except tk.TclError:
                    pass
                try:
                    return colorchooser.askcolor(title=title, parent=win)
                finally:
                    _restore_manual_window()

            def _ensure_hidden_token_tag(text_widget: tk.Text) -> str:
                token_tag = "hidden_token"
                if token_tag not in text_widget.tag_names():
                    try:
                        text_widget.tag_configure(token_tag, elide=True)
                    except tk.TclError:
                        bg = text_widget.cget("bg")
                        text_widget.tag_configure(token_tag, foreground=bg, background=bg)
                return token_tag

            def _insert_hidden_token(text_widget: tk.Text, index: str, token: str) -> None:
                token_tag = _ensure_hidden_token_tag(text_widget)
                text_widget.insert(index, token)
                start = text_widget.index(index)
                end = text_widget.index(f"{start} + {len(token)}c")
                text_widget.tag_add(token_tag, start, end)

            def _remove_underline_tokens(text_widget: tk.Text, start: str, end: str) -> None:
                range_start = text_widget.index(f"{start} - 12c")
                range_end = text_widget.index(f"{end} + 12c")
                segment = text_widget.get(range_start, range_end)
                matches = list(_UNDERLINE_TOKEN_RE.finditer(segment))
                for match in reversed(matches):
                    token_start = text_widget.index(f"{range_start} + {match.start()}c")
                    token_end = text_widget.index(f"{range_start} + {match.end()}c")
                    text_widget.delete(token_start, token_end)

            toolbar_bg = colors.get("surface", "#111827")
            toolbar_border = colors.get("card_border", "#1F2937")
            toolbar_fg = colors.get("text", "#E5E7EB")
            toolbar_hover = colors.get("accent", "#4F46E5")
            toolbar = tk.Frame(format_inner, bg=toolbar_bg, highlightthickness=1, highlightbackground=toolbar_border)
            toolbar.pack(fill=tk.X, padx=6, pady=4)

            def _btn(parent, label, command):
                btn = tk.Button(
                    parent,
                    text=label,
                    command=command,
                    bg=toolbar_bg,
                    fg=toolbar_fg,
                    activebackground=toolbar_hover,
                    activeforeground=toolbar_fg,
                    relief=tk.FLAT,
                    bd=0,
                    padx=6,
                    pady=2,
                    font=("Segoe UI", 10, "bold"),
                )

                def _on_enter(_e):
                    btn.configure(bg=toolbar_hover)

                def _on_leave(_e):
                    btn.configure(bg=toolbar_bg)

                btn.bind("<Enter>", _on_enter)
                btn.bind("<Leave>", _on_leave)
                btn.pack(side=tk.LEFT, padx=2)
                return btn

            def _toggle_tag(tag: str, **cfg) -> None:
                text_widget = current_text_widget()
                if tag not in text_widget.tag_names():
                    text_widget.tag_configure(tag, **cfg)
                try:
                    has_tag = tag in text_widget.tag_names("sel.first")
                    if has_tag:
                        text_widget.tag_remove(tag, "sel.first", "sel.last")
                    else:
                        text_widget.tag_add(tag, "sel.first", "sel.last")
                except tk.TclError:
                    pass

            def _clear_formatting() -> None:
                text_widget = current_text_widget()
                try:
                    start = text_widget.index("sel.first")
                    end = text_widget.index("sel.last")
                except tk.TclError:
                    start = text_widget.index("insert linestart")
                    end = text_widget.index("insert lineend")
                for tag in text_widget.tag_names():
                    if tag != "sel":
                        text_widget.tag_remove(tag, start, end)
                _remove_underline_tokens(text_widget, start, end)

            def _apply_color() -> None:
                chosen = _choose_color("Цвет текста")
                if not chosen or not chosen[1]:
                    return
                color = chosen[1]
                _toggle_tag(f"color_{color}", foreground=color)

            def _apply_marker() -> None:
                chosen = _choose_color("Цвет выделения")
                if not chosen or not chosen[1]:
                    return
                color = chosen[1]
                _toggle_tag(f"marker_{color}", background=color)

            def _apply_list(prefix: str) -> None:
                text_widget = current_text_widget()
                try:
                    start = text_widget.index("sel.first linestart")
                    end = text_widget.index("sel.last linestart")
                except tk.TclError:
                    start = text_widget.index("insert linestart")
                    end = start
                line = start
                counter = 1
                while True:
                    insert_prefix = prefix.format(counter=counter)
                    text_widget.insert(line, insert_prefix)
                    if line == end:
                        break
                    line = text_widget.index(f"{line} +1 line")
                    counter += 1

            def _line_range(text_widget: tk.Text) -> tuple[str, str]:
                try:
                    start = text_widget.index("sel.first linestart")
                    end = text_widget.index("sel.last linestart")
                except tk.TclError:
                    start = text_widget.index("insert linestart")
                    end = start
                return start, end

            def _indent_lines() -> None:
                text_widget = current_text_widget()
                start, end = _line_range(text_widget)
                line = start
                while True:
                    text_widget.insert(line, "\t")
                    if line == end:
                        break
                    line = text_widget.index(f"{line} +1 line")

            def _outdent_lines() -> None:
                text_widget = current_text_widget()
                start, end = _line_range(text_widget)
                line = start
                while True:
                    if text_widget.get(line, f"{line} +1c") == "\t":
                        text_widget.delete(line, f"{line} +1c")
                    elif text_widget.get(line, f"{line} +4c") == "    ":
                        text_widget.delete(line, f"{line} +4c")
                    elif text_widget.get(line, f"{line} +2c") == "  ":
                        text_widget.delete(line, f"{line} +2c")
                    if line == end:
                        break
                    line = text_widget.index(f"{line} +1 line")

            def _indent_key(_event):
                _indent_lines()
                return "break"

            def _outdent_key(_event):
                _outdent_lines()
                return "break"

            def _apply_justify(align: str) -> None:
                text_widget = current_text_widget()
                tag = f"align_{align}"
                if tag not in text_widget.tag_names():
                    text_widget.tag_configure(tag, justify=align)
                try:
                    text_widget.tag_add(tag, "sel.first", "sel.last")
                except tk.TclError:
                    text_widget.tag_add(tag, "1.0", tk.END)

            def _apply_script(tag: str, offset: int) -> None:
                text_widget = current_text_widget()
                base_font = tkfont.Font(font=text_widget.cget("font"))
                size = max(8, int(base_font.cget("size") or 12) - 3)
                script_font = base_font.copy()
                script_font.configure(size=size)
                _toggle_tag(tag, font=script_font, offset=offset)

            def _insert_link() -> None:
                url = simpledialog.askstring("Ссылка", "Введите URL:")
                if not url:
                    return
                text_widget = current_text_widget()
                text_widget.insert("insert", url)

            def _apply_underline_style(style: str) -> None:
                text_widget = current_text_widget()
                try:
                    start = text_widget.index("sel.first")
                    end = text_widget.index("sel.last")
                except tk.TclError:
                    return
                if style == "double":
                    tag_name = "underline_double"
                    tag_cfg = {"underline": 1, "foreground": colors.get("accent", "#4F46E5")}
                else:
                    tag_name = "underline_wavy"
                    tag_cfg = {"underline": 1, "foreground": colors.get("muted", "#98A2B3")}
                if tag_name not in text_widget.tag_names():
                    text_widget.tag_configure(tag_name, **tag_cfg)
                text_widget.tag_add(tag_name, start, end)
                _insert_hidden_token(text_widget, end, "[[/u]]")
                _insert_hidden_token(text_widget, start, f"[[u:{style}]]")

            def _insert_mathjax() -> None:
                text_widget = current_text_widget()
                try:
                    start = text_widget.index("sel.first")
                    end = text_widget.index("sel.last")
                    content = text_widget.get(start, end)
                    text_widget.delete(start, end)
                    text_widget.insert(start, f"\\({content}\\)")
                except tk.TclError:
                    insert_pos = text_widget.index("insert")
                    text_widget.insert(insert_pos, "\\(  \\)")
                    text_widget.mark_set("insert", f"{insert_pos} + 3c")

            def attach_media() -> None:
                filename = _open_media_dialog(
                    title="Прикрепить файл",
                    filetypes=[
                        ("Медиа", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.mp3 *.wav *.ogg *.m4a *.mp4 *.mov *.avi *.mkv *.webm"),
                        ("Все файлы", "*.*"),
                    ],
                )
                if not filename:
                    return
                ext = os.path.splitext(filename)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
                    attach_image(filename)
                elif ext in (".mp3", ".wav", ".ogg", ".m4a"):
                    attach_audio(filename)
                elif ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    attach_video(filename)
                else:
                    set_status("Формат не поддерживается для прикрепления")

            _btn(toolbar, "B", lambda: _toggle_tag("bold", font=("Segoe UI", 10, "bold")))
            _btn(toolbar, "I", lambda: _toggle_tag("italic", font=("Segoe UI", 10, "italic")))
            _btn(toolbar, "U", lambda: _toggle_tag("underline", underline=1))
            _btn(toolbar, "U²", lambda: _apply_underline_style("double"))
            _btn(toolbar, "U~", lambda: _apply_underline_style("wavy"))
            _btn(toolbar, "x²", lambda: _apply_script("superscript", 6))
            _btn(toolbar, "x₂", lambda: _apply_script("subscript", -4))
            _btn(toolbar, "A", _apply_color)
            _btn(toolbar, "🖍", _apply_marker)
            _btn(toolbar, "🧹", _clear_formatting)
            _btn(toolbar, "•", lambda: _apply_list("• "))
            _btn(toolbar, "1.", lambda: _apply_list("{counter}. "))
            _btn(toolbar, "L", lambda: _apply_justify("left"))
            _btn(toolbar, "C", lambda: _apply_justify("center"))
            _btn(toolbar, "R", lambda: _apply_justify("right"))
            _btn(toolbar, "⇥", _indent_lines)
            _btn(toolbar, "⇤", _outdent_lines)
            _btn(toolbar, "🔗", _insert_link)
            _btn(toolbar, "📎", attach_media)
            _btn(toolbar, "🎙", _fmt_not_ready)
            _btn(toolbar, "Fx", _insert_mathjax)
            _btn(toolbar, "</>", _fmt_not_ready)
            _btn(toolbar, "⨯", _fmt_not_ready)

            front_text.bind("<Tab>", _indent_key)
            front_text.bind("<Shift-Tab>", _outdent_key)
            back_text.bind("<Tab>", _indent_key)
            back_text.bind("<Shift-Tab>", _outdent_key)

            dots = []

            def update_dots():
                active_color = colors.get("accent", "#4F46E5") if colors else "#4F46E5"
                inactive_color = colors.get("muted", "#98A2B3") if colors else "#98A2B3"
                for idx, dot in enumerate(dots):
                    is_active = (idx == (1 if show_back_var.get() else 0))
                    dot.config(text="●" if is_active else "○", fg=active_color if is_active else inactive_color)

            def set_side(show_back: bool):
                show_back_var.set(show_back)
                self.manual_side = "back" if show_back else "front"
                target_text = back_text if show_back else front_text
                target_text.tkraise()
                toggle_button.config(
                    text="Показать лицевую сторону" if show_back else "Показать обратную сторону"
                )
                update_dots()
                add_change_log("Переключено на обратную сторону" if show_back else "Переключено на лицевую сторону")
                render_media_blocks()

            for idx in range(2):
                dot = tk.Label(dots_frame, text="○", font=("Segoe UI", 12), bg=dots_frame.cget("bg"))
                dot.pack(side=tk.LEFT, padx=2)
                dot.bind("<Button-1>", lambda _e, i=idx: set_side(i == 1))
                dots.append(dot)

            toggle_button.config(command=lambda: set_side(not show_back_var.get()))

            def add_change_log(msg: str) -> None:
                set_status(msg)

            drag_state = {"item": None, "x": 0, "y": 0}

            def _reset_media_slot():
                media_canvas.delete("all")
                self._manual_media_photos["front"] = None
                self._manual_media_photos["back"] = None
                self._manual_img_photo_front = None
                self._manual_img_photo_back = None

            def _draw_drop_zone(canvas_width: int, canvas_height: int) -> tuple[float, float, float, float]:
                pad = 16
                x1, y1 = pad, pad
                x2, y2 = max(pad + 10, canvas_width - pad), max(pad + 10, canvas_height - pad)
                media_canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    dash=(6, 4),
                    outline="#888",
                    width=2,
                )
                media_canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text="Перетащите сюда изображение/видео/аудио\nили нажмите иконку ниже",
                    fill="#666",
                    justify="center",
                    font=("Segoe UI", 10),
                )
                return x1, y1, x2, y2

            def _render_media_in_dropzone(
                canvas: tk.Canvas,
                rect_coords: tuple[float, float, float, float],
                img_path: str,
                pad: int = 2,
            ) -> tuple[tk.PhotoImage | None, int | None]:
                x1, y1, x2, y2 = rect_coords
                max_w = max(1, int(x2 - x1 - 2 * pad))
                max_h = max(1, int(y2 - y1 - 2 * pad))
                if PIL_AVAILABLE:
                    img, _ = load_preview_image(img_path, (max_w, max_h))
                    photo = ImageTk.PhotoImage(img)
                else:
                    ext = os.path.splitext(img_path)[1].lower()
                    if ext != ".png":
                        raise ValueError("Unsupported image format without PIL")
                    photo = tk.PhotoImage(file=img_path)
                    if photo.width() > max_w or photo.height() > max_h:
                        scale = max(photo.width() / max_w, photo.height() / max_h)
                        subsample = int(scale) + 1
                        photo = photo.subsample(subsample, subsample)
                center = (x1 + pad + max_w // 2, y1 + pad + max_h // 2)
                item_id = canvas.create_image(
                    center[0],
                    center[1],
                    image=photo,
                    anchor="center",
                    tags=("media_item_fixed",),
                )
                return photo, item_id

            def _store_media_position(side: str, item_id: int) -> None:
                coords = media_canvas.coords(item_id)
                if coords:
                    manual_media[side]["pos"] = (coords[0], coords[1])

            def render_media_blocks():
                side = current_side()
                image_path = manual_media[side]["image"]
                video_path = manual_media[side]["video"]
                audio_path = manual_media[side]["audio"]

                media_canvas.delete("all")
                canvas_width = media_canvas.winfo_width() or 600
                canvas_height = media_canvas.winfo_height() or 280
                rect_coords = _draw_drop_zone(canvas_width, canvas_height)
                x1, y1, x2, y2 = rect_coords
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                media_item_id = None
                if image_path and os.path.exists(image_path):
                    try:
                        photo, media_item_id = _render_media_in_dropzone(
                            media_canvas, rect_coords, image_path
                        )
                        self._manual_media_photos[side] = photo
                        setattr(self, f"_manual_img_photo_{side}", photo)
                        manual_media[side]["pos"] = None
                    except Exception:
                        media_item_id = media_canvas.create_text(
                            center[0],
                            center[1],
                            text=f"🖼️ {os.path.basename(image_path)}",
                            fill="#444",
                            font=("Segoe UI", 11),
                            tags=("media_item",),
                        )
                        manual_media[side]["pos"] = center
                elif video_path:
                    pos = manual_media[side]["pos"] or center
                    manual_media[side]["pos"] = pos
                    media_item_id = media_canvas.create_text(
                        pos[0],
                        pos[1],
                        text=f"🎬 {os.path.basename(video_path)}",
                        fill="#444",
                        font=("Segoe UI", 11),
                        tags=("media_item",),
                    )
                elif audio_path:
                    pos = manual_media[side]["pos"] or center
                    manual_media[side]["pos"] = pos
                    media_item_id = media_canvas.create_text(
                        pos[0],
                        pos[1],
                        text=f"🎵 {os.path.basename(audio_path)}",
                        fill="#444",
                        font=("Segoe UI", 11),
                        tags=("media_item",),
                    )
                else:
                    manual_media[side]["pos"] = None

                audio_badge.configure(
                    text=(
                        f"🔊 audio: {os.path.basename(audio_path)}"
                        if audio_path
                        else "🔊 audio: не прикреплено"
                    )
                )
                video_badge.configure(
                    text=(
                        f"🎬 video: {os.path.basename(video_path)}"
                        if video_path
                        else "🎬 video: не прикреплено"
                    )
                )

                if media_item_id:
                    media_canvas.tag_raise(media_item_id)
                    _store_media_position(side, media_item_id)
                update_preview_if_open(full_refresh=True)

            def attach_image(path: str) -> None:
                side = current_side()
                manual_media[side]["image"] = path
                manual_media[side]["pos"] = None
                add_change_log(f"Прикреплено изображение: {os.path.basename(path)}")
                render_media_blocks()

            def attach_audio(path: str) -> None:
                side = current_side()
                manual_media[side]["audio"] = path
                manual_media[side]["pos"] = None
                add_change_log(f"Прикреплено аудио: {os.path.basename(path)}")
                render_media_blocks()

            def attach_video(path: str) -> None:
                side = current_side()
                manual_media[side]["video"] = path
                manual_media[side]["pos"] = None
                add_change_log(f"Прикреплено видео: {os.path.basename(path)}")
                render_media_blocks()

            def _on_media_press(event) -> None:
                item = media_canvas.find_withtag("current")
                if not item:
                    return
                if "media_item" not in media_canvas.gettags(item[0]):
                    return
                drag_state["item"] = item[0]
                drag_state["x"] = event.x
                drag_state["y"] = event.y

            def _on_media_drag(event) -> None:
                item = drag_state["item"]
                if not item:
                    return
                dx = event.x - drag_state["x"]
                dy = event.y - drag_state["y"]
                media_canvas.move(item, dx, dy)
                drag_state["x"] = event.x
                drag_state["y"] = event.y
                _store_media_position(current_side(), item)

            def _on_media_release(_event) -> None:
                drag_state["item"] = None

            media_canvas.tag_bind("media_item", "<ButtonPress-1>", _on_media_press)
            media_canvas.tag_bind("media_item", "<B1-Motion>", _on_media_drag)
            media_canvas.tag_bind("media_item", "<ButtonRelease-1>", _on_media_release)

            def _handle_drop(event) -> None:
                files = win.tk.splitlist(event.data)
                if not files:
                    return
                path = files[0]
                ext = os.path.splitext(path)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
                    attach_image(path)
                elif ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    attach_video(path)
                elif ext in (".mp3", ".wav", ".ogg", ".m4a"):
                    attach_audio(path)

            self._dnd_enabled = safe_enable_dnd(media_canvas, _handle_drop)
            if not self._dnd_enabled:
                add_change_log("DnD отключен (tkdnd недоступен)")

            media_canvas.bind("<Configure>", lambda _e: render_media_blocks())

            def select_image():
                filename = _open_media_dialog(
                    title="Выбрать изображение",
                    filetypes=[
                        ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                        ("Все файлы", "*.*"),
                    ],
                )
                if not filename:
                    return
                attach_image(filename)

            def add_audio():
                filename = _open_media_dialog(
                    title="Добавить аудио",
                    filetypes=[
                        ("Аудио", "*.mp3 *.wav *.ogg *.m4a"),
                        ("Все файлы", "*.*"),
                    ],
                )
                if not filename:
                    return
                attach_audio(filename)

            def add_video():
                filename = _open_media_dialog(
                    title="Добавить видео",
                    filetypes=[
                        ("Видео", "*.mp4 *.mov *.avi *.mkv *.webm"),
                        ("Все файлы", "*.*"),
                    ],
                )
                if not filename:
                    return
                attach_video(filename)

            def generate_tts_audio():
                side = current_side()
                text_widget = back_text if side == "back" else front_text
                text_value = text_widget.get("1.0", tk.END).strip()
                if not text_value:
                    messagebox.showinfo("Озвучка", "Нет текста для озвучивания.", parent=win)
                    return
                lang = get_deck_tts_lang(deck_map.get(deck_var.get()), "de")
                url = get_tts_url(text_value, lang)
                if not url:
                    messagebox.showinfo("Озвучка", "Нет текста для озвучивания.", parent=win)
                    return
                try:
                    filename = os.path.join(get_tts_cache_dir(), f"tts_{int(time.time())}.mp3")
                    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(request, timeout=10) as response:
                        data = response.read()
                    with open(filename, "wb") as fh:
                        fh.write(data)
                    if not os.path.exists(filename) or os.path.getsize(filename) < 1024:
                        raise RuntimeError("TTS файл не создан или пустой")
                    manual_media[side]["audio"] = filename
                    manual_media[side]["pos"] = None
                    add_change_log(f"Озвучка добавлена: {os.path.basename(filename)}")
                    render_media_blocks()
                except Exception as exc:
                    messagebox.showerror("Озвучка", f"Не удалось озвучить: {exc}", parent=win)

            def clear_media():
                side = current_side()
                manual_media[side]["image"] = None
                manual_media[side]["audio"] = None
                manual_media[side]["video"] = None
                manual_media[side]["pos"] = None
                add_change_log("Удалены прикрепления для текущей стороны")
                render_media_blocks()

            icon_active_bg = colors.get("surface", "#111827")
            icon_active_fg = colors.get("text", "#E5E7EB")

            media_toolbar = tk.Frame(bottom_actions, bg=background)
            media_toolbar.pack(fill=tk.X)

            def add_icon_button(parent: tk.Frame, label: str, command) -> None:
                tk.Button(
                    parent,
                    text=label,
                    command=command,
                    bg=background,
                    fg=text_color,
                    activebackground=icon_active_bg,
                    activeforeground=icon_active_fg,
                    relief=tk.FLAT,
                    width=3,
                    height=1,
                    font=("Segoe UI", 12),
                ).pack(side=tk.LEFT, padx=4)

            def add_card_to_intro(card_id: int, deck_id: int) -> None:
                mark_card_for_overview(card_id)

            # PATCH: manual preview image renders + preview media frame matches repeat media slot size (auto-detected)
            def update_preview_if_open(full_refresh: bool = False) -> None:
                preview_win = preview_state.get("window")
                renderer = preview_state.get("renderer")
                if not preview_win or not renderer:
                    return
                if not preview_win.winfo_exists():
                    preview_state["window"] = None
                    preview_state["renderer"] = None
                    return
                side = preview_state.get("side") or "front"
                media = manual_media.get(side, {})
                image_override = ""
                plain_widget = front_text if side == "front" else back_text
                plain_text = plain_widget.get("1.0", tk.END).rstrip("\n")
                front_rich_doc = export_rich_from_editor(front_text)
                back_rich_doc = export_rich_from_editor(back_text)
                has_rich = bool((back_rich_doc if side == "back" else front_rich_doc).get("tags"))
                print("[PREVIEW TEXT]", "side=", side, "plain_len=", len(plain_text), "has_rich=", has_rich)
                card_data = {
                    "front": front_text.get("1.0", tk.END).strip(),
                    "back": back_text.get("1.0", tk.END).strip(),
                    "front_rich": front_rich_doc,
                    "back_rich": back_rich_doc,
                    "front_image_path": manual_media.get("front", {}).get("image"),
                    "back_image_path": manual_media.get("back", {}).get("image"),
                    "front_video_path": manual_media.get("front", {}).get("video"),
                    "back_video_path": manual_media.get("back", {}).get("video"),
                    "audio_path": media.get("audio"),
                }
                image_path = manual_media.get(side, {}).get("image")
                resolved_preview_path = resolve_media_path(image_path)
                if resolved_preview_path:
                    resolved_preview_path = os.path.abspath(resolved_preview_path)
                print(
                    "[PREVIEW IMG]",
                    "side=",
                    side,
                    "path=",
                    image_path,
                    "abs=",
                    resolved_preview_path,
                    "exists=",
                    bool(resolved_preview_path and os.path.exists(resolved_preview_path)),
                )
                header_text = "Предпросмотр | след. повтор: —"
                renderer.set_header_text(header_text)
                if full_refresh:
                    renderer.render(
                        card_data,
                        show_back=(side == "back"),
                        prefer_audio_side=side,
                        header_text=header_text,
                        image_override=image_override,
                    )
                else:
                    renderer.update_text(
                        card_data,
                        show_back=(side == "back"),
                        image_override=image_override,
                    )
                slot_size = preview_state.get("slot_size") or renderer.get_repeat_media_slot_size()
                preview_state["slot_size"] = slot_size
                repeat_slot_w, repeat_slot_h = slot_size
                renderer.render_image_to_container(
                    renderer.image_container,
                    image_path,
                    f"manual_preview_{side}",
                    repeat_slot_w,
                    repeat_slot_h,
                )

            def open_preview() -> None:
                preview_win = tk.Toplevel(win)
                preview_win.title("Предпросмотр карточки")
                preview_win.geometry("900x620")
                preview_win.grab_set()
                apply_dark_theme_to_window(preview_win, colors)

                container = tk.Frame(preview_win, bg=background)
                container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

                controls = tk.Frame(container, bg=background)
                controls.pack(fill=tk.X, pady=(0, 8))
                side_var = tk.StringVar(value="front")

                def _set_side(side: str) -> None:
                    side_var.set(side)
                    preview_state["side"] = side
                    update_preview_if_open(full_refresh=True)

                ttk.Button(controls, text="Лицевая сторона", command=lambda: _set_side("front")).pack(
                    side=tk.LEFT, padx=4
                )
                ttk.Button(controls, text="Обратная сторона", command=lambda: _set_side("back")).pack(
                    side=tk.LEFT, padx=4
                )

                card_wrap = tk.Frame(
                    container,
                    bg=DARK_BG,
                    highlightbackground=CARD_BORDER,
                    highlightthickness=1,
                    bd=0,
                    width=CARD_VIEW_WIDTH,
                    height=CARD_VIEW_HEIGHT,
                )
                card_wrap.pack(padx=10, pady=10)
                card_wrap.pack_propagate(False)
                renderer = CardRenderer(
                    card_wrap,
                    palette=colors,
                    editable=False,
                    show_image_toolbar=False,
                    image_layout="side",
                    show_media_placeholder=False,
                    fixed_media_slot=REPEAT_MEDIA_SLOT_SIZE,
                )
                repeat_slot_w, repeat_slot_h = renderer.get_repeat_media_slot_size()
                print("[REPEAT SLOT]", repeat_slot_w, repeat_slot_h)
                renderer.image_container.config(width=repeat_slot_w, height=repeat_slot_h)
                renderer.image_container.pack_propagate(False)
                renderer.image_container.grid_propagate(False)
                preview_state["window"] = preview_win
                preview_state["renderer"] = renderer
                preview_state["side"] = side_var.get()
                preview_state["slot_size"] = (repeat_slot_w, repeat_slot_h)

                def _on_close():
                    preview_state["window"] = None
                    preview_state["renderer"] = None
                    preview_win.destroy()

                preview_win.protocol("WM_DELETE_WINDOW", _on_close)
                update_preview_if_open(full_refresh=True)

            def save_card():
                try:
                    front_value = front_text.get("1.0", tk.END).strip()
                    back_value = back_text.get("1.0", tk.END).strip()
                    front_rich_doc = export_rich_from_editor(front_text)
                    back_rich_doc = export_rich_from_editor(back_text)
                    if not front_value and not back_value:
                        messagebox.showwarning(
                            "Пустая карточка",
                            "Введите текст для лицевой или обратной стороны.",
                            parent=win,
                        )
                        return
                    deck_id = deck_map.get(deck_var.get())
                    if not deck_id:
                        messagebox.showerror("Ошибка", "Выберите колоду для сохранения.", parent=win)
                        return

                    front_image = manual_media["front"]["image"]
                    back_image = manual_media["back"]["image"]
                    stored_front = copy_image_asset_to_media(front_image, "front") if front_image else None
                    stored_back = copy_image_asset_to_media(back_image, "back") if back_image else None

                    card_id = insert_card(
                        deck_id,
                        front_value,
                        back_value,
                        front_image_path=stored_front,
                        back_image_path=stored_back,
                        front_rich=front_rich_doc,
                        back_rich=back_rich_doc,
                        audio_path=None,
                        level=1,
                    )
                    media_entries = []
                    front_audio = manual_media["front"]["audio"]
                    back_audio = manual_media["back"]["audio"]
                    front_video = manual_media["front"]["video"]
                    back_video = manual_media["back"]["video"]
                    if front_audio:
                        media_entries.append((copy_audio_asset_to_media(front_audio, "front_audio"), "audio", "front", None))
                    if back_audio:
                        media_entries.append((copy_audio_asset_to_media(back_audio, "back_audio"), "audio", "back", None))
                    if front_video:
                        media_entries.append((copy_video_asset_to_media(front_video, "front_video"), "video", "front", None))
                    if back_video:
                        media_entries.append((copy_video_asset_to_media(back_video, "back_video"), "video", "back", None))
                    if media_entries:
                        attach_media_to_card(card_id, media_entries)
                    add_card_to_intro(card_id, deck_id)
                except Exception as exc:
                    try:
                        with open("add_card_save_error.log", "a", encoding="utf-8") as log_file:
                            log_file.write(f"{datetime.now().isoformat()} manual card save\n")
                            log_file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                            log_file.write("\n")
                    except Exception:
                        pass
                    messagebox.showerror("БД", f"Не удалось сохранить карточку:\n{exc}", parent=win)
                    return

                messagebox.showinfo("Сохранено", "Сохранено: добавлено в ознакомление", parent=win)
                add_change_log("Сохранена карточка")
                if hasattr(self, "refresh_decks"):
                    self.refresh_decks()
                if hasattr(self, "update_deck_preview"):
                    self.update_deck_preview()
                if hasattr(self, "update_overdue_badge"):
                    self.update_overdue_badge()
                win.destroy()

            add_icon_button(media_toolbar, "🖼️", select_image)
            add_icon_button(media_toolbar, "🎬", add_video)
            add_icon_button(media_toolbar, "🎵", add_audio)
            add_icon_button(media_toolbar, "🔊", generate_tts_audio)
            add_icon_button(media_toolbar, "🗑️", clear_media)

            ttk.Button(
                media_toolbar,
                text="Предпросмотр",
                style="Secondary.TButton",
                command=open_preview,
            ).pack(side=tk.LEFT, padx=(12, 0))

            ttk.Button(
                media_toolbar,
                text="Сохранить",
                style="Primary.TButton",
                command=save_card,
            ).pack(side=tk.LEFT, padx=(12, 0))

            ttk.Button(
                media_toolbar,
                text="Отмена",
                style="Secondary.TButton",
                command=win.destroy,
            ).pack(side=tk.LEFT, padx=8)

            set_side(False)
        except Exception as exc:
            log_ui_error(exc)
            messagebox.showerror("Ошибка UI", traceback.format_exc(), parent=win)

    # --------- обзор карточек ---------

    def show_cards_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        cards = get_cards_in_deck(self.selected_deck_id)
        # фильтр по фазе, если выбрана подколода
        if self.selected_phase is not None:
            cards = [c for c in cards if c["leitner_level"] == self.selected_phase]

        if not cards:
            phase_text = f" (фаза {self.selected_phase})" if self.selected_phase is not None else ""
            messagebox.showinfo("Пусто", f"В этой колоде{phase_text} пока нет карточек.")
            return

        win = tk.Toplevel(self)
        win.title("Режим генерации: все карточки (front + back)")
        win.geometry("950x600")
        win.grab_set()

        palette = getattr(self, "palette", None) or {}
        apply_dark_theme_to_window(win, palette)

        canvas = tk.Canvas(win, highlightthickness=0, bg=palette.get("background", "#0B0D12"))
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="Surface.TFrame")

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for c in cards:
            c = sanitize_card_sounds(dict(c))
            card_frame = ttk.LabelFrame(
                scroll_frame,
                text=f"ID {c['id']} (уровень {c['leitner_level']}, next {c['next_review']}, prog {c['progress']})",
                style="Card.TLabelframe",
            )
            card_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(card_frame, text="FRONT:").pack(anchor="w")
            txt_front = tk.Text(card_frame, height=3)
            style_text_widget(txt_front, palette)
            txt_front.pack(fill=tk.X, padx=5)
            txt_front.insert("1.0", c["front"])
            create_context_menu(txt_front)  # Добавляем контекстное меню
            attach_simple_toolbar(card_frame, txt_front)

            ttk.Label(card_frame, text="BACK:").pack(anchor="w")
            txt_back = tk.Text(card_frame, height=3)
            style_text_widget(txt_back, palette)
            txt_back.pack(fill=tk.X, padx=5)
            txt_back.insert("1.0", c["back"])
            create_context_menu(txt_back)  # Добавляем контекстное меню
            attach_simple_toolbar(card_frame, txt_back)

            front_img_var = tk.StringVar(value=c["front_image_path"] or "")
            back_img_var = tk.StringVar(value=c["back_image_path"] or "")

            frame_img = ttk.Frame(card_frame, style="CardInner.TFrame")
            frame_img.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(frame_img, text="Картинка FRONT:").grid(row=0, column=0, sticky="w")
            lbl_front = ttk.Label(
                frame_img,
                text=os.path.basename(front_img_var.get()) if front_img_var.get() else "(нет)"
            )
            lbl_front.grid(row=0, column=1, padx=5, sticky="w")

            def select_front_img(var=front_img_var, lbl=lbl_front):
                filetypes = [
                    ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp"),
                    ("Все файлы", "*.*"),
                ]
                filename = filedialog.askopenfilename(
                    title="Выбрать картинку для FRONT",
                    filetypes=filetypes
                )
                if filename:
                    var.set(filename)
                    lbl.config(text=os.path.basename(filename))

            ttk.Button(frame_img, text="Выбрать...",
                       command=select_front_img).grid(row=0, column=2, padx=5)

            ttk.Label(frame_img, text="Картинка BACK:").grid(row=1, column=0, sticky="w", pady=(3, 0))
            lbl_back = ttk.Label(
                frame_img,
                text=os.path.basename(back_img_var.get()) if back_img_var.get() else "(нет)"
            )
            lbl_back.grid(row=1, column=1, padx=5, sticky="w", pady=(3, 0))

            def select_back_img(var=back_img_var, lbl=lbl_back):
                filetypes = [
                    ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp"),
                    ("Все файлы", "*.*"),
                ]
                filename = filedialog.askopenfilename(
                    title="Выбрать картинку для BACK",
                    filetypes=filetypes
                )
                if filename:
                    var.set(filename)
                    lbl.config(text=os.path.basename(filename))

            ttk.Button(frame_img, text="Выбрать...",
                       command=select_back_img).grid(row=1, column=2, padx=5, pady=(3, 0))

            audio_entries = get_card_audio_entries(c)

            audio_frame = ttk.LabelFrame(card_frame, text="Аудио", style="Card.TLabelframe")
            audio_frame.pack(fill=tk.X, padx=5, pady=5)

            if audio_entries:
                for idx, entry in enumerate(audio_entries):
                    side_var = tk.StringVar(value=(entry.get("side") or "back"))
                    row_frame = ttk.Frame(audio_frame)
                    row_frame.pack(fill=tk.X, pady=2)

                    ttk.Label(row_frame, text=entry.get("label") or os.path.basename(entry.get("path") or ""))\
                        .pack(side=tk.LEFT, padx=5)

                    def play_selected(path=entry.get("path"), missing=entry.get("missing")):
                        if path and os.path.exists(path) and not missing:
                            play_audio_file(path)
                        else:
                            messagebox.showwarning("Аудио", "Аудио не найдено")

                    ttk.Button(
                        row_frame,
                        text="▶",
                        command=play_selected,
                        state=tk.NORMAL if entry.get("path") and not entry.get("missing") else tk.DISABLED,
                    ).pack(side=tk.LEFT, padx=5)

                    def make_side_updater(media_id, var):
                        def updater():
                            if media_id is not None:
                                update_media_side(media_id, var.get())
                        return updater

                    for side_label, side_value in (("Front", "front"), ("Back", "back")):
                        ttk.Radiobutton(
                            row_frame,
                            text=side_label,
                            variable=side_var,
                            value=side_value,
                            command=make_side_updater(entry.get("media_id"), side_var),
                            state=tk.NORMAL if entry.get("media_id") else tk.DISABLED,
                        ).pack(side=tk.LEFT, padx=2)

                    def make_delete_handler(media_id, frame=row_frame):
                        def handler():
                            if media_id:
                                remove_media_entry(media_id)
                            frame.destroy()
                        return handler

                    ttk.Button(
                        row_frame,
                        text="Удалить привязку",
                        command=make_delete_handler(entry.get("media_id")),
                        state=tk.NORMAL if entry.get("media_id") else tk.DISABLED,
                    ).pack(side=tk.LEFT, padx=5)
            else:
                ttk.Label(audio_frame, text="(Нет аудио)").pack(anchor="w", padx=5, pady=3)

            def make_save_handler(card_id, tf, tb, fimg_var, bimg_var):
                def handler():
                    f = tf.get("1.0", tk.END).strip()
                    b = tb.get("1.0", tk.END).strip()
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE cards SET front = ?, back = ?, front_image_path = ?, back_image_path = ? WHERE id = ?;",
                        (f, b, fimg_var.get() or None, bimg_var.get() or None, card_id)
                    )
                    conn.commit()
                    conn.close()
                    messagebox.showinfo("Сохранено", f"Карточка {card_id} обновлена.")
                return handler

            def make_delete_handler(card_id, frame):
                def handler():
                    if not messagebox.askyesno("Удалить карточку",
                                               f"Точно удалить карточку {card_id}?"):
                        return
                    try:
                        delete_card(card_id)
                    except sqlite3.OperationalError as e:
                        messagebox.showerror("БД", f"Не удалось удалить карточку:\n{e}")
                        return
                    frame.destroy()
                return handler

            btns_frame = ttk.Frame(card_frame, style="CardInner.TFrame")
            btns_frame.pack(anchor="e", padx=5, pady=3)

            ttk.Button(
                btns_frame,
                text="Сохранить изменения",
                command=make_save_handler(c["id"], txt_front, txt_back,
                                          front_img_var, back_img_var)
            ).pack(side=tk.RIGHT, padx=3)

            ttk.Button(
                btns_frame,
                text="Удалить",
                command=make_delete_handler(c["id"], card_frame)
            ).pack(side=tk.RIGHT, padx=3)

    # --------- генерация из текста ---------

    def _ensure_generated_placeholder_image(self) -> str | None:
        if not PIL_AVAILABLE:
            return None
        ensure_media_dir()
        filename = "generated_placeholder.png"
        path = os.path.join(MEDIA_FOLDER, filename)
        if not os.path.exists(path):
            img = Image.new("RGB", (640, 360), color="#F3F4F6")
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 639, 359], outline="#9CA3AF", width=2)
            text = "Generated Placeholder"
            draw.text((20, 20), text, fill="#111827")
            try:
                img.save(path, format="PNG")
            except Exception:
                return None
        return path

    def _create_ai_placeholder_image(self, prompt: str) -> str | None:
        if not PIL_AVAILABLE:
            return None
        ensure_media_dir()
        img = Image.new("RGB", (640, 360), color="#E5E7EB")
        draw = ImageDraw.Draw(img)
        text = "AI Image Preview\n\n" + (prompt[:120] + "…" if len(prompt) > 120 else prompt)
        draw.text((20, 20), text, fill="#111827")
        filename = f"ai_image_{uuid4().hex}.png"
        path = os.path.join(MEDIA_FOLDER, filename)
        try:
            img.save(path, format="PNG")
        except Exception:
            return None
        return path

    def _create_ai_placeholder_video(self) -> str | None:
        ensure_media_dir()
        filename = f"ai_video_{uuid4().hex}.mp4"
        path = os.path.join(MEDIA_FOLDER, filename)
        try:
            with open(path, "wb") as handle:
                handle.write(b"")
        except Exception:
            return None
        return path

    def open_generate_from_notes_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        win = tk.Toplevel(self)
        win.title("Генерация по конспекту АИ+ картинки")
        win.geometry("980x740")
        win.grab_set()
        apply_dark_theme_to_window(win, self.palette)

        main_frame = ttk.Frame(win, style="Surface.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        ttk.Label(
            main_frame,
            text="Генерация по конспекту АИ+ картинки",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        state = {
            "chunks": [],
            "file_path": None,
            "file_kind": None,
            "cards": [],
            "editor_open": False,
            "editor_manager": None,
            "editor_window": None,
            "fallback_window": None,
            "fallback_text": None,
            "webview_started": False,
            "webview_starting": False,
        }

        def compute_cost() -> int:
            return NOTES_PAGE_COST_PRO if self.is_premium_active() else NOTES_PAGE_COST_BASIC

        import_frame = ttk.LabelFrame(main_frame, text="Импорт файла")
        import_frame.pack(fill=tk.X, padx=4, pady=6)
        import_frame.columnconfigure(1, weight=1)

        file_var = tk.StringVar(value="Файл не выбран")
        page_var = tk.StringVar(value="")
        page_range_var = tk.StringVar(value="")
        cost_var = tk.StringVar(value=f"Стоимость: {compute_cost()} ⚡ за страницу")
        editor_status_var = tk.StringVar(value="Редактор: не открыт")

        ttk.Label(import_frame, text="Файл:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Label(import_frame, textvariable=file_var).grid(row=0, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(import_frame, text="Страница/чанк:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        page_combo = ttk.Combobox(
            import_frame,
            textvariable=page_var,
            values=[],
            state="readonly",
            width=14,
        )
        page_combo.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        page_combo.bind("<<ComboboxSelected>>", lambda _e: update_current_chunk_html())

        ttk.Label(import_frame, text="Диапазон страниц (PDF):").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        page_range_entry = ttk.Entry(import_frame, textvariable=page_range_var, width=24)
        page_range_entry.grid(row=2, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(
            import_frame,
            text=f"Напр.: 1-5,8. Пусто = первые {SAFE_IMPORT_MAX_PDF_PAGES} стр.",
        ).grid(row=3, column=1, sticky="w", padx=6, pady=(0, 6))

        def update_cost_label():
            cost_var.set(f"Стоимость: {compute_cost()} ⚡ за страницу")

        def update_current_chunk_html() -> str:
            if not state["chunks"]:
                self.current_chunk_html = ""
                return ""
            try:
                index = int(page_var.get()) - 1
            except ValueError:
                index = 0
            index = max(0, min(index, len(state["chunks"]) - 1))
            html = state["chunks"][index] or ""
            self.current_chunk_html = html
            return html

        def import_file():
            path = filedialog.askopenfilename(
                title="Импорт файла",
                filetypes=[
                    ("Документы", "*.docx *.odt *.pdf"),
                    ("DOCX", "*.docx"),
                    ("ODT", "*.odt"),
                    ("PDF", "*.pdf"),
                    ("Все файлы", "*.*"),
                ],
            )
            if not path:
                return
            ext = Path(path).suffix.lower()

            def _task():
                if ext == ".docx":
                    chunks = import_docx(
                        path,
                        chunk_chars=SAFE_IMPORT_CHUNK_CHARS,
                        max_total_chars=SAFE_IMPORT_MAX_TOTAL_CHARS,
                    )
                    file_kind = "docx"
                elif ext == ".odt":
                    chunks = import_odt(
                        path,
                        chunk_chars=SAFE_IMPORT_CHUNK_CHARS,
                        max_total_chars=SAFE_IMPORT_MAX_TOTAL_CHARS,
                    )
                    file_kind = "odt"
                elif ext == ".pdf":
                    chunks = import_pdf(
                        path,
                        page_range=page_range_var.get(),
                        max_pages=SAFE_IMPORT_MAX_PDF_PAGES,
                        max_total_chars=SAFE_IMPORT_MAX_TOTAL_CHARS,
                    )
                    file_kind = "pdf"
                else:
                    raise RuntimeError("Поддерживаются только .docx, .odt, .pdf.")
                return {"chunks": chunks, "file_kind": file_kind, "path": path}

            def _on_success(result):
                chunks = result.get("chunks") or []
                file_kind = result.get("file_kind")
                if not chunks:
                    messagebox.showwarning("Импорт", "Текст не найден.")
                    return
                state["chunks"] = chunks
                state["file_path"] = result.get("path")
                state["file_kind"] = file_kind
                file_var.set(os.path.basename(result.get("path") or ""))
                page_values = [str(i + 1) for i in range(len(chunks))]
                page_combo.configure(values=page_values)
                page_var.set(page_values[0])
                update_cost_label()
                update_current_chunk_html()

                if file_kind == "pdf":
                    empty_pages = [idx + 1 for idx, text in enumerate(chunks) if not text.strip()]
                    if empty_pages:
                        messagebox.showwarning(
                            "PDF",
                            "Некоторые страницы без текста. Возможно, это скан.\n"
                            "Для сканов нужен OCR.",
                        )

            def _on_error(exc):
                if isinstance(exc, MemoryError):
                    messagebox.showerror(
                        "Импорт",
                        "Недостаточно памяти. Укажите меньший файл или диапазон страниц.",
                    )
                else:
                    messagebox.showerror("Импорт", str(exc))

            self.run_with_loading(_task, on_success=_on_success, on_error=_on_error)

        def _mark_editor_closed():
            state["editor_open"] = False
            state["editor_window"] = None
            state["fallback_window"] = None
            state["fallback_text"] = None
            editor_status_var.set("Редактор: не открыт")

        def _handle_editor_make_cards(selection_html: str) -> None:
            cards = parse_quill_html_to_cards(selection_html)
            if not cards:
                messagebox.showinfo(
                    "Предпросмотр",
                    "Карточки не найдены. Проверьте разметку (bold/underline/цвет).",
                )
                return
            state["cards"] = cards
            open_preview_window(limit=20)

        def _ensure_editor_manager() -> WebEditorManager:
            if state["editor_manager"] is None:
                state["editor_manager"] = WebEditorManager(self.root, _handle_editor_make_cards)
            return state["editor_manager"]

        def ensure_editor() -> bool:
            if not QUILL_WEBVIEW_AVAILABLE:
                if state["fallback_window"] is None or not state["fallback_window"].winfo_exists():
                    fallback_window, fallback_text = open_fallback_editor(
                        self.root,
                        "",
                        "Редактор конспекта (Quill)",
                        "pywebview не установлен или не запускается.",
                        on_close=lambda: self.root.after(0, _mark_editor_closed),
                    )
                    state["fallback_window"] = fallback_window
                    state["fallback_text"] = fallback_text
                else:
                    state["fallback_window"].deiconify()
                    state["fallback_window"].lift()
                editor_status_var.set("Редактор: открыт (упрощенный)")
                state["editor_open"] = True
                gui_hooks.editor_did_open(state["fallback_window"])
                return True

            quill_files = [
                os.path.join(BASE_DIR, "vendor", "quill", "quill.min.js"),
                os.path.join(BASE_DIR, "vendor", "quill", "quill.snow.css"),
            ]
            too_small = [path for path in quill_files if not os.path.exists(path) or os.path.getsize(path) < 10 * 1024]
            if too_small:
                messagebox.showwarning(
                    "Редактор",
                    "Quill не найден / пустые файлы. Проверьте vendor/quill/*.",
                )

            if state["editor_window"] is not None:
                try:
                    state["editor_window"].bring_to_front()
                except Exception:
                    pass
                editor_status_var.set("Редактор: открыт")
                state["editor_open"] = True
                gui_hooks.editor_did_open(state["editor_window"])
                return True

            try:
                import webview
            except Exception:
                if state["fallback_window"] is None or not state["fallback_window"].winfo_exists():
                    fallback_window, fallback_text = open_fallback_editor(
                        self.root,
                        "",
                        "Редактор конспекта (Quill)",
                        "Не удалось загрузить pywebview.",
                        on_close=lambda: self.root.after(0, _mark_editor_closed),
                    )
                    state["fallback_window"] = fallback_window
                    state["fallback_text"] = fallback_text
                else:
                    state["fallback_window"].deiconify()
                    state["fallback_window"].lift()
                editor_status_var.set("Редактор: открыт (упрощенный)")
                state["editor_open"] = True
                gui_hooks.editor_did_open(state["fallback_window"])
                return True

            editor_path = os.path.abspath(os.path.join(BASE_DIR, "editor_quill.html"))
            if not os.path.exists(editor_path):
                messagebox.showerror(
                    "Редактор",
                    f"Файл редактора не найден:\n{editor_path}",
                )
                return False
            editor_url = editor_path
            editor_manager = _ensure_editor_manager()
            try:
                state["editor_window"] = webview.create_window(
                    "Редактор конспекта (Quill)",
                    url=editor_url,
                    js_api=editor_manager.api,
                    width=1100,
                    height=700,
                )
                editor_manager.attach_window(
                    state["editor_window"],
                    on_close=lambda: self.root.after(0, _mark_editor_closed),
                )
            except Exception:
                if state["fallback_window"] is None or not state["fallback_window"].winfo_exists():
                    fallback_window, fallback_text = open_fallback_editor(
                        self.root,
                        "",
                        "Редактор конспекта (Quill)",
                        "Не удалось открыть окно редактора через pywebview.",
                        on_close=lambda: self.root.after(0, _mark_editor_closed),
                    )
                    state["fallback_window"] = fallback_window
                    state["fallback_text"] = fallback_text
                else:
                    state["fallback_window"].deiconify()
                    state["fallback_window"].lift()
                editor_status_var.set("Редактор: открыт (упрощенный)")
                state["editor_open"] = True
                return True

            editor_status_var.set("Редактор: открыт")
            state["editor_open"] = True
            gui_hooks.editor_did_open(state["editor_window"])

            if not state["webview_started"] and not state["webview_starting"]:
                state["webview_starting"] = True

                def _start_webview() -> None:
                    try:
                        try:
                            webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = True
                        except Exception:
                            pass
                        try:
                            webview.settings["ALLOW_FILE_URLS"] = True
                        except Exception:
                            pass
                        webview.start(debug=True, gui="tkinter", http_server=True)
                        state["webview_started"] = True
                    except Exception:
                        messagebox.showwarning(
                            "Редактор",
                            "pywebview не смог стартовать. Открыт упрощенный редактор.",
                        )
                        state["editor_window"] = None
                        if state["fallback_window"] is None or not state["fallback_window"].winfo_exists():
                            fallback_window, fallback_text = open_fallback_editor(
                                self.root,
                                "",
                                "Редактор конспекта (Quill)",
                                "pywebview не стартовал.",
                                on_close=lambda: self.root.after(0, _mark_editor_closed),
                            )
                            state["fallback_window"] = fallback_window
                            state["fallback_text"] = fallback_text
                        else:
                            state["fallback_window"].deiconify()
                            state["fallback_window"].lift()
                        editor_status_var.set("Редактор: открыт (упрощенный)")
                        state["editor_open"] = True
                    finally:
                        state["webview_starting"] = False

                self.root.after(0, _start_webview)

            return True

        def load_selected_into_editor():
            if not state["chunks"]:
                messagebox.showwarning("Редактор", "Сначала импортируйте файл.")
                return
            html = (update_current_chunk_html() or "").strip()
            if not html:
                messagebox.showwarning(
                    "Пусто",
                    "Сначала импортируйте страницу/чанк, чтобы появился текст для редактора.",
                )
                return
            if not ensure_editor():
                return
            editor_manager = _ensure_editor_manager()
            try:
                if state["fallback_text"] is not None:
                    state["fallback_text"].delete("1.0", tk.END)
                    state["fallback_text"].insert("1.0", html)
                else:
                    editor_manager.set_html_safe(html)
            except Exception:
                with open("web_editor_error.log", "a", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc() + "\n")
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Ошибка",
                        "Загрузка в редактор не удалась. См. web_editor_error.log",
                    ),
                )

        def sync_from_editor():
            if not ensure_editor():
                return
            if not state["chunks"]:
                messagebox.showwarning("Редактор", "Сначала импортируйте файл.")
                return
            try:
                index = int(page_var.get()) - 1
            except ValueError:
                index = 0
            index = max(0, min(index, len(state["chunks"]) - 1))

            def _task():
                if state["fallback_text"] is not None:
                    return state["fallback_text"].get("1.0", tk.END).strip()
                editor_manager = _ensure_editor_manager()
                return editor_manager.get_html()

            def _on_success(html):
                if html is None:
                    messagebox.showwarning("Редактор", "Не удалось получить контент.")
                    return
                state["chunks"][index] = html
                update_current_chunk_html()
                messagebox.showinfo("Редактор", "Контент обновлён из редактора.")

            self.run_with_loading(_task, on_success=_on_success)

        def build_cards_from_selection() -> list[dict[str, str]]:
            if not ensure_editor():
                return []
            if state["fallback_text"] is not None:
                try:
                    selection_html = state["fallback_text"].get("sel.first", "sel.last")
                except tk.TclError:
                    selection_html = ""
                if not selection_html:
                    selection_html = state["fallback_text"].get("1.0", tk.END).strip()
            else:
                editor_manager = _ensure_editor_manager()
                selection_html = editor_manager.get_selection_html()
                if not selection_html:
                    selection_html = editor_manager.get_html()
            if not selection_html:
                return []
            return parse_quill_html_to_cards(selection_html)

        def charge_for_page() -> bool:
            cost = compute_cost()
            meta = {
                "operation": "notes_wysiwyg_generation",
                "file": state["file_path"],
                "page": page_var.get(),
            }
            return self.guard_premium_and_spend(
                "Генерация карточек из конспекта",
                cost,
                require_premium=False,
                meta=meta,
            )

        def generate_cards():
            if not state["chunks"]:
                messagebox.showwarning("Генерация", "Сначала импортируйте файл и загрузите страницу в редактор.")
                return
            update_cost_label()
            if not charge_for_page():
                return

            def _task():
                return build_cards_from_selection()

            def _on_success(cards):
                if not cards:
                    messagebox.showinfo(
                        "Генерация",
                        "Карточки не найдены. Проверьте разметку (bold/underline/цвет).",
                    )
                    return
                state["cards"] = cards
                messagebox.showinfo("Генерация", f"Сформировано карточек: {len(cards)}")

            self.run_with_loading(_task, on_success=_on_success)

        def open_preview_window(limit: int = 20):
            if not state["cards"]:
                messagebox.showwarning("Предпросмотр", "Сначала сформируйте карточки.")
                return

            def _task():
                return list(state["cards"])[:limit]

            def _on_success(cards):
                preview_win = tk.Toplevel(self)
                preview_win.title("Предпросмотр карточек")
                preview_win.geometry("800x600")
                preview_win.grab_set()
                apply_dark_theme_to_window(preview_win, self.palette)

                container = ttk.Frame(preview_win, style="Surface.TFrame")
                container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                canvas = tk.Canvas(
                    container,
                    highlightthickness=0,
                    bg=self.palette.get("background", "#0B0D12"),
                )
                scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
                list_frame = ttk.Frame(canvas, style="Surface.TFrame")

                list_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
                )
                canvas.create_window((0, 0), window=list_frame, anchor="nw")
                canvas.configure(yscrollcommand=scroll.set)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scroll.pack(side=tk.RIGHT, fill=tk.Y)

                for idx, card in enumerate(cards, start=1):
                    row = ttk.Frame(list_frame, style="Card.TFrame", padding=8)
                    row.pack(fill=tk.X, pady=6)
                    ttk.Label(
                        row,
                        text=f"{idx}. Карточка",
                        style="Muted.TLabel",
                    ).pack(anchor="w", pady=(0, 6))
                    front_wrap = tk.Frame(row, bg=self.palette.get("background", "#0B0D12"))
                    front_wrap.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
                    render_card_layout(
                        front_wrap,
                        {
                            "text": card.get("front", ""),
                            "image_path": None,
                            "video_path": None,
                            "audio_path": None,
                        },
                        editable=False,
                    )
                    back_wrap = tk.Frame(row, bg=self.palette.get("background", "#0B0D12"))
                    back_wrap.pack(fill=tk.BOTH, expand=True)
                    render_card_layout(
                        back_wrap,
                        {
                            "text": card.get("back", ""),
                            "image_path": None,
                            "video_path": None,
                            "audio_path": None,
                        },
                        editable=False,
                    )

            self.run_with_loading(_task, on_success=_on_success)

        def save_cards():
            if not state["cards"]:
                messagebox.showwarning("Сохранение", "Сначала сформируйте карточки.")
                return
            add_images = add_images_var.get()
            image_limit = max(0, min(10, int(image_limit_var.get() or 0)))
            raw_video_path = video_path_var.get().strip()
            total_cards = len(state["cards"])
            image_steps = min(image_limit, total_cards) if add_images else 0
            total_steps = max(total_cards + image_steps, 1)

            def _task(progress_cb):
                created = 0
                done = 0
                stored_video_path = None
                if raw_video_path:
                    stored_video_path = copy_video_asset_to_media(raw_video_path, "notes_video")
                total_pages = max(len(state["chunks"]), 1)
                try:
                    page_index = int(page_var.get() or "1")
                except ValueError:
                    page_index = 1
                progress_cb(done, total_steps, f"Страница {page_index}/{total_pages}")
                for idx, card in enumerate(state["cards"], start=1):
                    progress_cb(done, total_steps, f"Подготовка карточки {idx}/{total_steps}")
                    image_path = ""
                    if add_images and idx <= image_limit:
                        progress_cb(done, total_steps, f"Генерация картинки {idx}/{image_steps or total_steps}")
                        image_path = self._create_ai_placeholder_image(card.get("front", "")) or ""
                        done += 1
                        progress_cb(done, total_steps, f"Генерация картинки {idx}/{image_steps or total_steps}")

                    note_fields = {
                        "word": card.get("front", ""),
                        "translation": card.get("back", ""),
                        "example": "",
                        "level": 1,
                        "image": image_path,
                        "front_image_path": image_path,
                        "back_image_path": "",
                        "audio_path": None,
                        "front": card.get("front", ""),
                        "back": card.get("back", ""),
                    }
                    note_id, cards_created = create_note_with_cards(
                        self.selected_deck_id,
                        note_fields,
                        note_type_id=ensure_generated_note_type_id(),
                    )
                    if stored_video_path:
                        attach_media_to_note(note_id, [(stored_video_path, "video")])
                    created += cards_created
                    done += 1
                    progress_cb(done, total_steps, "Сохранение карточек…")
                return created

            def _on_success(created):
                messagebox.showinfo("Сохранено", f"Сохранено карточек: {created}")
                self.refresh_decks()
                self.update_overdue_badge()
                self.refresh_activation_progress_ui()

            def _on_error(exc):
                messagebox.showerror("Сохранение", str(exc))

            self.task_runner.run_task(
                "Сохранение карточек",
                "determinate",
                _task,
                on_success=_on_success,
                on_error=_on_error,
                total=total_steps,
            )

        ttk.Button(import_frame, text="Импорт файла", command=import_file).grid(
            row=0, column=2, padx=6, pady=6
        )
        ttk.Button(import_frame, text="Загрузить в редактор", command=load_selected_into_editor).grid(
            row=1, column=2, padx=6, pady=6
        )

        editor_frame = ttk.LabelFrame(main_frame, text="Редактор конспекта (Quill)")
        editor_frame.pack(fill=tk.X, padx=4, pady=6)

        ttk.Label(editor_frame, textvariable=editor_status_var).pack(anchor="w", padx=6, pady=(6, 2))
        ttk.Button(editor_frame, text="Открыть редактор", command=ensure_editor).pack(
            anchor="w", padx=6, pady=(0, 6)
        )
        ttk.Button(editor_frame, text="Обновить из редактора", command=sync_from_editor).pack(
            anchor="w", padx=6, pady=(0, 6)
        )

        actions_frame = ttk.LabelFrame(main_frame, text="Генерация карточек")
        actions_frame.pack(fill=tk.X, padx=4, pady=6)

        add_images_var = tk.BooleanVar(value=True)
        image_limit_var = tk.StringVar(value="10")
        video_path_var = tk.StringVar(value="")

        add_images_row = ttk.Frame(actions_frame)
        add_images_row.pack(fill=tk.X, padx=6, pady=6)
        ttk.Checkbutton(
            add_images_row,
            text="Добавлять AI-картинки (до 10 на страницу)",
            variable=add_images_var,
        ).pack(side=tk.LEFT)
        ttk.Label(add_images_row, text="Лимит:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Spinbox(
            add_images_row,
            from_=0,
            to=10,
            textvariable=image_limit_var,
            width=5,
        ).pack(side=tk.LEFT)

        video_row = ttk.Frame(actions_frame)
        video_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(video_row, text="Видео для карточек:").pack(side=tk.LEFT)
        video_label = ttk.Label(video_row, text="(нет)")
        video_label.pack(side=tk.LEFT, padx=6)

        def select_video():
            filename = filedialog.askopenfilename(
                title="Загрузить видео",
                filetypes=[("Видео", "*.mp4 *.mkv *.webm *.avi *.mov *.wmv *.flv"), ("Все файлы", "*.*")],
            )
            if filename:
                video_path_var.set(filename)
                video_label.config(text="🎬 Видео прикреплено")

        def clear_video():
            video_path_var.set("")
            video_label.config(text="(нет)")

        ttk.Button(video_row, text="Загрузить видео", command=select_video).pack(side=tk.RIGHT)
        ttk.Button(video_row, text="Очистить", command=clear_video).pack(side=tk.RIGHT, padx=6)

        buttons_row = ttk.Frame(actions_frame)
        buttons_row.pack(fill=tk.X, padx=6, pady=6)

        ttk.Button(
            buttons_row,
            text="Сделать карточки из выделенного",
            command=generate_cards,
        ).pack(side=tk.LEFT)
        ttk.Label(buttons_row, textvariable=cost_var).pack(side=tk.LEFT, padx=10)
        ttk.Button(buttons_row, text="Предпросмотр", command=open_preview_window).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(buttons_row, text="Сохранить в колоду", command=save_cards).pack(
            side=tk.RIGHT, padx=4
        )

    def open_generate_from_text_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        win = tk.Toplevel(self)
        apply_window_icon(win, self._logo_big, ico_path=os.path.join(BASE_DIR, "assets", "app.ico"))
        win.title("Авто-генерация из текста")
        win.geometry("650x580")
        win.grab_set()
        apply_dark_theme_to_window(win, self.palette)
        canvas = tk.Canvas(win, highlightthickness=0, bg=self.palette["background"])
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="Surface.TFrame")

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(scroll_frame, text="Front text (лицевая сторона):").pack(anchor="w", padx=10, pady=(10, 0))
        front_text = tk.Text(scroll_frame, height=4)
        style_text_widget(front_text, self.palette)
        front_text.pack(fill=tk.X, padx=10)
        create_context_menu(front_text)

        ttk.Label(scroll_frame, text="Back text (обратная сторона):").pack(anchor="w", padx=10, pady=(8, 0))
        back_text = tk.Text(scroll_frame, height=4)
        style_text_widget(back_text, self.palette)
        back_text.pack(fill=tk.X, padx=10)
        create_context_menu(back_text)

        # Кнопка вставки из буфера обмена
        insert_frame = ttk.Frame(scroll_frame)
        insert_frame.pack(fill=tk.X, padx=10, pady=(6, 5))

        def paste_from_clipboard():
            try:
                clipboard_text = win.clipboard_get()
                front_text.delete("1.0", tk.END)
                front_text.insert("1.0", clipboard_text)
            except tk.TclError:
                pass

        ttk.Button(
            insert_frame,
            text="📋 Вставить в Front text",
            command=paste_from_clipboard,
        ).pack(side=tk.LEFT)

        ttk.Label(
            scroll_frame,
            text="Если заполнен Back text, сохранится одна карточка. Для авто-генерации используйте Front text.",
            style="Muted.TLabel",
            wraplength=520,
        ).pack(anchor="w", padx=10, pady=(0, 6))

        frame_opts = ttk.LabelFrame(scroll_frame, text="Настройки")
        frame_opts.pack(fill=tk.X, padx=10, pady=10)

        self._build_generation_template_selector(frame_opts, self.selected_deck_id, padx=5, pady=(5, 5))

        use_ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame_opts,
            text="Генерировать картинку для каждой новой карточки (OpenAI)",
            variable=use_ai_var
        ).pack(anchor="w")

        one_sent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame_opts,
            text="1 предложение = 1 карточка (разбивать сложный текст)",
            variable=one_sent_var
        ).pack(anchor="w")

        front_image_var = tk.StringVar(value="")
        back_image_var = tk.StringVar(value="")
        front_video_var = tk.StringVar(value="")
        back_video_var = tk.StringVar(value="")

        frame_images = ttk.LabelFrame(frame_opts, text="Изображения карточки")
        frame_images.pack(fill=tk.X, padx=5, pady=(5, 0))

        ttk.Label(frame_images, text="Лицевая сторона:").grid(row=0, column=0, sticky="w", pady=3)
        front_image_label = ttk.Label(frame_images, text="(нет)")
        front_image_label.grid(row=0, column=1, sticky="w", padx=5)

        def select_front_image():
            filename = filedialog.askopenfilename(
                title="Прикрепить изображение (лицевая сторона)",
                filetypes=[
                    ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                    ("Все файлы", "*.*"),
                ],
            )
            if filename:
                front_image_var.set(filename)
                front_image_label.config(text=os.path.basename(filename))

        ttk.Button(
            frame_images,
            text="Прикрепить изображение (лицевая сторона)",
            command=select_front_image,
        ).grid(row=0, column=2, padx=5, pady=3, sticky="e")

        ttk.Label(frame_images, text="Обратная сторона:").grid(row=1, column=0, sticky="w", pady=3)
        back_image_label = ttk.Label(frame_images, text="(нет)")
        back_image_label.grid(row=1, column=1, sticky="w", padx=5)

        def select_back_image():
            filename = filedialog.askopenfilename(
                title="Прикрепить изображение (обратная сторона)",
                filetypes=[
                    ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                    ("Все файлы", "*.*"),
                ],
            )
            if filename:
                back_image_var.set(filename)
                back_image_label.config(text=os.path.basename(filename))

        ttk.Button(
            frame_images,
            text="Прикрепить изображение (обратная сторона)",
            command=select_back_image,
        ).grid(row=1, column=2, padx=5, pady=3, sticky="e")

        ttk.Label(frame_images, text="Видео (лицевая):").grid(row=2, column=0, sticky="w", pady=3)
        front_video_label = ttk.Label(frame_images, text="(нет)")
        front_video_label.grid(row=2, column=1, sticky="w", padx=5)

        def select_front_video():
            filename = filedialog.askopenfilename(
                title="Прикрепить видео (лицевая сторона)",
                filetypes=[
                    ("Видео", "*.mp4 *.mov *.avi *.mkv *.webm"),
                    ("Все файлы", "*.*"),
                ],
            )
            if filename:
                front_video_var.set(filename)
                front_video_label.config(text="🎬 Видео прикреплено")

        ttk.Button(
            frame_images,
            text="Загрузить видео (лицевая сторона)",
            command=select_front_video,
        ).grid(row=2, column=2, padx=5, pady=3, sticky="e")

        ttk.Label(frame_images, text="Видео (обратная):").grid(row=3, column=0, sticky="w", pady=3)
        back_video_label = ttk.Label(frame_images, text="(нет)")
        back_video_label.grid(row=3, column=1, sticky="w", padx=5)

        def select_back_video():
            filename = filedialog.askopenfilename(
                title="Прикрепить видео (обратная сторона)",
                filetypes=[
                    ("Видео", "*.mp4 *.mov *.avi *.mkv *.webm"),
                    ("Все файлы", "*.*"),
                ],
            )
            if filename:
                back_video_var.set(filename)
                back_video_label.config(text="🎬 Видео прикреплено")

        ttk.Button(
            frame_images,
            text="Загрузить видео (обратная сторона)",
            command=select_back_video,
        ).grid(row=3, column=2, padx=5, pady=3, sticky="e")

        frame_images.columnconfigure(1, weight=1)

        self._build_generation_template_selector(frame_opts, self.selected_deck_id, padx=5, pady=(5, 0))

        ttk.Label(
            frame_opts,
            text="Переменные: {question}, {translation}, {sentence_with_gap}, {word}, {ipa}, {gender}, {plural}, {sentence}"
        ).pack(anchor="w", padx=5, pady=(5, 0))

        progress_frame = ttk.LabelFrame(scroll_frame, text="Прогресс")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="")
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=5)

        log_box = tk.Text(scroll_frame, height=6, state="disabled")
        style_text_widget(log_box, self.palette)
        log_box.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        def append_log(message: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, message + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        task_holder = {"task": None}

        def build_local_cards():
            front_value = front_text.get("1.0", tk.END).strip()
            back_value = back_text.get("1.0", tk.END).strip()
            if not front_value:
                raise ValueError("Front text пустой.")

            if back_value:
                base_word = front_value.splitlines()[0].strip() or front_value
                front, back = render_generation_faces(
                    self.selected_deck_id,
                    {
                        "translation": back_value,
                        "front_translation": "",
                        "sentence_with_gap": front_value,
                        "word": base_word,
                        "ipa": "",
                        "gender": "",
                        "plural": "",
                        "sentence": front_value,
                    },
                )
                cards = [
                    {
                        "front": front,
                        "back": back,
                        "word": base_word,
                        "translation": back_value,
                        "sentence": front_value,
                        "sentence_with_gap": front_value,
                    }
                ]
                return cards, set(), None, None

            one_sent = one_sent_var.get()
            cards, new_words = build_local_cards_from_text(self.selected_deck_id, front_value, one_sent)
            return cards, new_words, None, None

        def open_preview_window(cards, front_image_path: str, back_image_path: str):
            preview_win = tk.Toplevel(self)
            preview_win.title("Предпросмотр карточек")
            preview_win.geometry("900x650")
            preview_win.grab_set()
            preview_win._img_refs = []

            palette = getattr(self, "palette", None) or {}
            bg = palette.get("background", "#0B0D12")
            panel = palette.get("panel", "#111522")
            text_color = palette.get("text", "#E8ECF4")

            preview_win.configure(bg=bg)

            container = ttk.Frame(preview_win)
            container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            canvas = tk.Canvas(container, bg=bg, highlightthickness=0)
            scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
            list_frame = ttk.Frame(canvas)

            list_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
            )
            canvas.create_window((0, 0), window=list_frame, anchor="nw")
            canvas.configure(yscrollcommand=scroll.set)

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)

            for idx, card in enumerate(cards, start=1):
                row = tk.Frame(list_frame, bg=panel, bd=1, relief="flat")
                row.pack(fill=tk.X, pady=6)
                tk.Label(
                    row,
                    text=f"{idx}. Карточка",
                    bg=panel,
                    fg=text_color,
                    font=("Segoe UI", 10, "bold"),
                ).pack(anchor="w", padx=10, pady=(10, 6))

                front_wrap = tk.Frame(row, bg=panel)
                front_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
                render_card_layout(
                    front_wrap,
                    {
                        "text": card.get("front", ""),
                        "image_path": front_image_path,
                        "video_path": front_video_var.get() or None,
                        "audio_path": None,
                    },
                    editable=False,
                )

                back_wrap = tk.Frame(row, bg=panel)
                back_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
                render_card_layout(
                    back_wrap,
                    {
                        "text": card.get("back", ""),
                        "image_path": back_image_path,
                        "video_path": back_video_var.get() or None,
                        "audio_path": None,
                    },
                    editable=False,
                )

        def preview_cards():
            self.show_loading("Загрузка")

            def worker():
                try:
                    cards, _new_words, _front_t, _back_t = build_local_cards()
                    if not cards:
                        raise ValueError("Новых слов/предложений не найдено.")
                    self.after(
                        0,
                        lambda: open_preview_window(
                            cards,
                            front_image_var.get(),
                            back_image_var.get(),
                        ),
                    )
                except Exception as exc:
                    self.after(0, lambda: messagebox.showerror("Предпросмотр", str(exc)))
                finally:
                    self.after(0, self.hide_loading)

            threading.Thread(target=worker, daemon=True).start()

        def save_cards():
            self.show_loading("Загрузка")

            def worker():
                try:
                    cards, new_words, _front_t, _back_t = build_local_cards()
                    if not cards:
                        raise ValueError("Новых слов/предложений не найдено.")

                    front_image_path = front_image_var.get()
                    back_image_path = back_image_var.get()
                    front_video_path = front_video_var.get()
                    back_video_path = back_video_var.get()
                    stored_front = (
                        copy_image_asset_to_media(front_image_path, "front")
                        if front_image_path
                        else None
                    )
                    stored_back = (
                        copy_image_asset_to_media(back_image_path, "back")
                        if back_image_path
                        else None
                    )
                    stored_front_video = (
                        copy_video_asset_to_media(front_video_path, "front_video")
                        if front_video_path
                        else None
                    )
                    stored_back_video = (
                        copy_video_asset_to_media(back_video_path, "back_video")
                        if back_video_path
                        else None
                    )

                    created = 0
                    for card in cards:
                        note_fields = {
                            "word": card.get("word") or card.get("sentence"),
                            "translation": card.get("translation") or "",
                            "example": card.get("sentence") or "",
                            "level": 1,
                            "image": stored_front or "",
                            "front_image_path": stored_front or "",
                            "back_image_path": stored_back or "",
                            "audio_path": None,
                            "front": card.get("front", ""),
                            "back": card.get("back", ""),
                        }
                        note_id, cards_created = create_note_with_cards(
                            self.selected_deck_id,
                            note_fields,
                            note_type_id=ensure_generated_note_type_id(),
                        )
                        attach_media_to_note(
                            note_id,
                            [
                                (stored_front_video, "video", "front", None),
                                (stored_back_video, "video", "back", None),
                            ],
                        )
                        created += cards_created

                    add_new_words(new_words)

                    def on_done():
                        self.hide_loading()
                        messagebox.showinfo("Сохранено", f"Сохранено: {created} карточек")
                        self.refresh_decks()
                        self.update_overdue_badge()
                        self.refresh_activation_progress_ui()

                    self.after(0, on_done)
                except Exception as exc:
                    self.after(0, lambda: (self.hide_loading(), messagebox.showerror("Сохранение", str(exc))))

            threading.Thread(target=worker, daemon=True).start()

        def run_generation():
            text = front_text.get("1.0", tk.END).strip()
            if not text:
                messagebox.showerror("Ошибка", "Front text пустой.")
                return
            if not self.spend_credits_or_warn(
                TEXT_GEN_CREDIT_COST,
                "Генерация карточек из текста",
                {"operation": "generate_from_text"},
            ):
                return

            use_ai_images = use_ai_var.get()
            api_key = OPENAI_API_KEY if OPENAI_API_KEY else None
            one_sent = one_sent_var.get()

            progress_var.set(0)
            progress_label_var.set("")
            log_box.configure(state="normal")
            log_box.delete("1.0", tk.END)
            log_box.configure(state="disabled")

            def handle_event(event):
                kind = event[0]
                if kind == "progress":
                    done, total, label = event[1:]
                    progress_bar.config(maximum=max(total, 1))
                    progress_var.set(done)
                    progress_label_var.set(f"{label}: {done}/{total}")
                elif kind == "log":
                    append_log(event[1])
                elif kind == "done":
                    result = event[1] or 0
                    self.unregister_bg_handler(task_holder["task"].queue)
                    task_holder["task"] = None
                    btn_generate.config(state=tk.NORMAL)
                    btn_cancel.config(state=tk.DISABLED)
                    if result == 0:
                        messagebox.showinfo("Результат", "Новых слов/предложений не найдено.")
                    else:
                        messagebox.showinfo("Результат", f"Создано карточек (включая синонимы/примеры): {result}")
                    win.destroy()
                elif kind == "error":
                    self.unregister_bg_handler(task_holder["task"].queue)
                    task_holder["task"] = None
                    btn_generate.config(state=tk.NORMAL)
                    btn_cancel.config(state=tk.DISABLED)
                    messagebox.showerror("Ошибка", event[1])

            def worker(task_obj):
                return auto_generate_cards_from_text(
                    self.selected_deck_id, text,
                    use_ai_images, api_key,
                    one_sentence_one_card=one_sent,
                    audio_path=None,
                    progress_queue=task_obj.queue,
                    cancel_check=task_obj.cancelled,
                    image_spend_cb=self._spend_for_ai_image,
                )

            btn_generate.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.NORMAL)
            task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(task_holder["task"].queue, handle_event)

        def cancel_generation():
            if task_holder["task"]:
                task_holder["task"].cancel()
                append_log("Отмена запрошена…")
                btn_cancel.config(state=tk.DISABLED)

        btn_frame = ttk.Frame(scroll_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        btn_preview = ttk.Button(btn_frame, text="Предпросмотр карточек", command=preview_cards)
        btn_preview.pack(side=tk.LEFT)
        btn_save = ttk.Button(btn_frame, text="Сохранить", command=save_cards)
        btn_save.pack(side=tk.LEFT, padx=5)
        btn_cancel = ttk.Button(btn_frame, text="Отмена", command=cancel_generation, state=tk.DISABLED)
        btn_cancel.pack(side=tk.RIGHT, padx=5)
        btn_generate = ttk.Button(btn_frame, text="Сгенерировать", command=run_generation)
        btn_generate.pack(side=tk.RIGHT)

    def open_generate_from_text_ai_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        if not self.is_premium_active():
            messagebox.showinfo(
                "Нужен Premium",
                "Режим генерации из текста AI доступен только для Premium.",
            )
            return

        win = tk.Toplevel(self)
        win.title("Режим генерация из текста AI 👑")
        win.geometry("760x620")
        win.grab_set()
        win._preview_img_ref = None
        apply_dark_theme_to_window(win, self.palette)

        form_frame = ttk.Frame(win, style="Surface.TFrame")
        form_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        ttk.Label(form_frame, text="Front text (PROMPT для Stable Diffusion):").pack(anchor="w")
        prompt_text = tk.Text(form_frame, height=4)
        style_text_widget(prompt_text, self.palette)
        prompt_text.pack(fill=tk.X, pady=(4, 10))
        create_context_menu(prompt_text)

        ttk.Label(form_frame, text="Back text (обратная сторона):").pack(anchor="w")
        back_text = tk.Text(form_frame, height=4)
        style_text_widget(back_text, self.palette)
        back_text.pack(fill=tk.X, pady=(4, 10))
        create_context_menu(back_text)

        self._build_generation_template_selector(form_frame, self.selected_deck_id, padx=0, pady=(0, 10))

        action_frame = ttk.LabelFrame(form_frame, text="Генерация")
        action_frame.pack(fill=tk.X, pady=6)
        action_row = ttk.Frame(action_frame)
        action_row.pack(fill=tk.X, padx=6, pady=6)

        preview_frame = ttk.LabelFrame(form_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        style_card(preview_frame, self.palette, padded=True)

        preview_label = tk.Label(
            preview_frame,
            text="Нет превью",
            bg=self.palette["panel"],
            fg=self.palette["muted"],
            justify="center",
        )
        preview_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        media_state = {"image_path": None, "video_path": None}

        def update_preview_image(path: str | None):
            if not path:
                preview_label.config(image="", text="Нет превью")
                win._preview_img_ref = None
                return
            preview_key = ("ai_preview", "image")
            rendered = render_image(preview_label, preview_label, path, 1.0, preview_key)
            if not rendered:
                preview_label.config(image="", text="Не удалось загрузить превью")
                win._preview_img_ref = None
                return
            win._preview_img_ref = getattr(preview_label, "image", None)
            preview_label.image = win._preview_img_ref

        def set_preview_text(text: str):
            preview_label.config(image="", text=text)
            win._preview_img_ref = None

        def generate_image():
            if not self.guard_premium_and_spend(
                "AI image generation",
                2,
                require_premium=True,
                meta={"operation": "ai_image", "images": 1},
            ):
                return
            set_preview_text("Генерация изображения…")

            def finish():
                prompt = prompt_text.get("1.0", tk.END).strip()
                path = self._create_ai_placeholder_image(prompt) or ""
                media_state["image_path"] = path
                media_state["video_path"] = None
                update_preview_image(path)

            win.after(1200, finish)

        def generate_video():
            if not self.guard_premium_and_spend(
                "AI video generation",
                20,
                require_premium=True,
                meta={"operation": "ai_video", "videos": 1},
            ):
                return
            set_preview_text("Генерация видео…")

            def finish():
                path = self._create_ai_placeholder_video() or ""
                media_state["video_path"] = path
                media_state["image_path"] = None
                set_preview_text("Видео создано (заглушка)")

            win.after(1500, finish)

        def upload_video():
            filename = filedialog.askopenfilename(
                title="Загрузить видео",
                filetypes=[("Видео", "*.mp4 *.mkv *.webm *.avi *.mov *.wmv *.flv"), ("Все файлы", "*.*")],
            )
            if filename:
                media_state["video_path"] = filename
                set_preview_text("Видео прикреплено")

        def save_card():
            front = prompt_text.get("1.0", tk.END).strip()
            back = back_text.get("1.0", tk.END).strip()
            if not front:
                messagebox.showerror("Ошибка", "Front text пустой.")
                return

            rendered_front, rendered_back = render_generation_faces(
                self.selected_deck_id,
                {
                    "translation": back,
                    "front_translation": "",
                    "sentence_with_gap": front,
                    "word": front.splitlines()[0].strip() if front else "",
                    "ipa": "",
                    "gender": "",
                    "plural": "",
                    "sentence": front,
                },
            )

            note_fields = {
                "word": front,
                "translation": back,
                "example": "",
                "level": 1,
                "image": media_state["image_path"] or "",
                "front_image_path": media_state["image_path"] or "",
                "back_image_path": "",
                "audio_path": None,
                "front": rendered_front,
                "back": rendered_back,
            }
            note_id, created = create_note_with_cards(
                self.selected_deck_id,
                note_fields,
                note_type_id=ensure_generated_note_type_id(),
            )
            if media_state["video_path"]:
                video_path = media_state["video_path"]
                stored_video = video_path
                try:
                    media_root = Path(MEDIA_FOLDER).resolve()
                    video_parent = Path(video_path).resolve().parent
                    if media_root not in video_parent.parents and media_root != video_parent:
                        stored_video = copy_video_asset_to_media(video_path, "ai_video")
                except Exception:
                    stored_video = copy_video_asset_to_media(video_path, "ai_video")
                attach_media_to_note(note_id, [(stored_video, "video")])
            messagebox.showinfo("Сохранено", f"Сохранено {created} карточек")
            self.refresh_decks()
            self.update_overdue_badge()
            self.refresh_activation_progress_ui()

        ttk.Button(
            action_row,
            text="Сгенерировать видео (10 сек) — 20 ⚡",
            style="Secondary.TButton",
            command=generate_video,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            action_row,
            text="Загрузить видео",
            style="Secondary.TButton",
            command=upload_video,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            action_row,
            text="Сгенерировать картинку — 2 ⚡",
            style="Secondary.TButton",
            command=generate_image,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            action_row,
            text="Сохранить карточку",
            style="Primary.TButton",
            command=save_card,
        ).pack(side=tk.RIGHT, padx=4)

    # --------- генерация из изображения ---------

    def open_generate_from_image_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        if not (is_tesseract_available() or is_easyocr_available() or is_paddle_available()):
            messagebox.showerror(
                "OCR недоступен",
                "Не найдены доступные OCR движки. Установите Tesseract, EasyOCR или PaddleOCR.",
            )
            return

        img_path = filedialog.askopenfilename(
            title="Выбери изображение с текстом (страница словаря и т.п.)",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("Все файлы", "*.*"),
            ]
        )
        if not img_path:
            return

        win = tk.Toplevel(self)
        win.title("OCR - Распознавание текста из изображения")
        win.geometry("900x650")
        win.grab_set()

        apply_dark_theme_to_window(win, self.palette)
        open_ocr_debug_log()
        logging.info("OCR module opened")
        paddle_available = is_paddle_available()
        if not paddle_available:
            messagebox.showwarning(
                "PaddleOCR недоступен",
                "PaddleOCR недоступен на этой системе (Illegal instruction). "
                "Используйте Tesseract/EasyOCR.",
            )
        style = ttk.Style(win)
        style.configure(
            "Dark.Vertical.TScrollbar",
            background="#111",
            troughcolor="#0b0f1a",
            bordercolor="#0b0f1a",
            arrowcolor="#ddd",
        )

        scroll_container = ttk.Frame(win, style="Surface.TFrame")
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas_bg = self.palette.get("background", "#0B0D12")
        scroll_canvas = tk.Canvas(scroll_container, highlightthickness=0, bg=canvas_bg)
        scroll_bar = ttk.Scrollbar(
            scroll_container, orient="vertical", command=scroll_canvas.yview, style="Dark.Vertical.TScrollbar"
        )
        scroll_canvas.configure(yscrollcommand=scroll_bar.set)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)

        scroll_frame = ttk.Frame(scroll_canvas, style="Surface.TFrame")
        scroll_window = scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def _sync_scrollregion(_event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _sync_width(event):
            scroll_canvas.itemconfigure(scroll_window, width=event.width)

        scroll_frame.bind("<Configure>", _sync_scrollregion)
        scroll_canvas.bind("<Configure>", _sync_width)

        def _on_mousewheel(event):
            if event.num == 4 or event.delta > 0:
                scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                scroll_canvas.yview_scroll(1, "units")

        def _bind_mousewheel(_event=None):
            scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            scroll_canvas.bind_all("<Button-4>", _on_mousewheel)
            scroll_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(_event=None):
            scroll_canvas.unbind_all("<MouseWheel>")
            scroll_canvas.unbind_all("<Button-4>")
            scroll_canvas.unbind_all("<Button-5>")

        scroll_canvas.bind("<Enter>", _bind_mousewheel)
        scroll_canvas.bind("<Leave>", _unbind_mousewheel)

        main_frame = ttk.Frame(scroll_frame, style="Surface.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # PATCH: OCR postprocess pipeline + paid OCR + paid card/video generation packs + preview renderer + pricing by plan (free/pro/premium)
        preview_frame = ttk.LabelFrame(main_frame, text="Предпросмотр изображения")
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        preview_label = ttk.Label(preview_frame, text="Превью изображения")
        preview_label.pack(anchor="center", padx=10, pady=10)
        preview_cache = {"img": None}
        original_image_path = img_path
        processed_image_path = None
        postprocess_state = {"done": False, "paid": False}
        ocr_state = {"done": False}

        def update_preview_image(path: str | None):
            if not path or not PIL_AVAILABLE:
                preview_label.configure(text="Превью изображения недоступно", image="")
                preview_label.image = None
                return
            try:
                img = Image.open(path)
                img = ImageOps.exif_transpose(img)
                max_w, max_h = 420, 240
                img.thumbnail((max_w, max_h), _pil_lanczos())
                tk_img = ImageTk.PhotoImage(img)
                preview_cache["img"] = tk_img
                preview_label.configure(image=tk_img, text="")
                preview_label.image = tk_img
            except Exception:
                preview_label.configure(text="Не удалось загрузить превью", image="")
                preview_label.image = None

        update_preview_image(original_image_path)

        # Настройки OCR
        ocr_opts = ttk.LabelFrame(main_frame, text="OCR настройки")
        ocr_opts.pack(fill=tk.X, pady=(0, 10))

        engine_options = ["Tesseract"]
        if is_easyocr_available():
            engine_options.append("EasyOCR")
        if paddle_available:
            engine_options.append("PaddleOCR")
        ocr_engine_var = tk.StringVar(value=engine_options[0])
        ttk.Label(ocr_opts, text="OCR движок:").grid(row=0, column=0, sticky="e", padx=(10, 5), pady=5)
        ttk.Combobox(
            ocr_opts,
            textvariable=ocr_engine_var,
            values=tuple(engine_options),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="w", padx=5)

        ocr_mode_var = tk.StringVar(value="fast")
        ttk.Label(ocr_opts, text="OCR MODE:").grid(row=1, column=0, sticky="w", padx=(10, 5), pady=5)
        ttk.Radiobutton(ocr_opts, text="Быстрый", variable=ocr_mode_var, value="fast").grid(row=1, column=1, sticky="w", padx=5, pady=5)
        pro_rb = ttk.Radiobutton(ocr_opts, text="PRO (только PaddleOCR)", variable=ocr_mode_var, value="pro")
        pro_rb.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(ocr_opts, text="Авто 2 колонки (DE|RU)", variable=ocr_mode_var, value="two_columns").grid(row=1, column=3, sticky="w", padx=5, pady=5)

        if not paddle_available:
            pro_rb.configure(state=tk.DISABLED)

        ttk.Label(ocr_opts, text="LANG MODE:").grid(row=2, column=0, sticky="e", padx=(10, 5), pady=5)
        lang_mode_var = tk.StringVar(value="deu+rus")
        ttk.Combobox(
            ocr_opts,
            textvariable=lang_mode_var,
            values=("deu+rus", "deu", "rus"),
            state="readonly",
            width=12,
        ).grid(row=2, column=1, sticky="w", padx=5)

        pages_count = 1
        use_postprocess_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ocr_opts,
            text="С постобработкой (PRO постобработка)",
            variable=use_postprocess_var,
        ).grid(row=3, column=2, sticky="w", padx=5, pady=5)

        ttk.Label(ocr_opts, text="Пресет предобработки:").grid(row=3, column=0, sticky="e", padx=(10, 5), pady=5)
        preprocess_preset_var = tk.StringVar(value="auto_pro")
        ttk.Combobox(
            ocr_opts,
            textvariable=preprocess_preset_var,
            values=("auto_pro", "basic"),
            state="readonly",
            width=12,
        ).grid(row=3, column=1, sticky="w", padx=5)

        ttk.Label(ocr_opts, text="Binarize:").grid(row=4, column=0, sticky="e", padx=(10, 5), pady=5)
        binarize_mode_var = tk.StringVar(value="adaptive")
        ttk.Combobox(
            ocr_opts,
            textvariable=binarize_mode_var,
            values=("adaptive", "otsu", "none"),
            state="readonly",
            width=10,
        ).grid(row=4, column=1, sticky="w", padx=5)

        ttk.Label(ocr_opts, text="PSM:").grid(row=4, column=2, sticky="e")
        psm_var = tk.StringVar(value="4")
        ttk.Combobox(ocr_opts, textvariable=psm_var, values=("3", "4", "6", "11"), state="readonly", width=5).grid(row=4, column=3, sticky="w", padx=5)

        dictionary_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ocr_opts, text="Словарь/учебник", variable=dictionary_mode_var).grid(row=5, column=0, sticky="w", padx=10, pady=5)

        debug_images_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ocr_opts, text="Сохранять debug картинки", variable=debug_images_var).grid(row=5, column=1, sticky="w", padx=5, pady=5)

        split_offset_var = tk.DoubleVar(value=0.0)
        ttk.Label(ocr_opts, text="2 колонки: Авто-разделение").grid(row=6, column=0, sticky="w", padx=10)
        ttk.Label(ocr_opts, text="Сдвиг разделителя (%):").grid(row=6, column=1, sticky="e")
        split_slider = ttk.Scale(ocr_opts, from_=-20, to=20, orient=tk.HORIZONTAL, variable=split_offset_var)
        split_slider.grid(row=6, column=2, sticky="we", padx=5)
        split_val_label = ttk.Label(ocr_opts, textvariable=tk.StringVar(value="0"))
        split_val_label.grid(row=6, column=3, sticky="w")

        def _update_split_label(*_args):
            split_val_label.config(text=f"{split_offset_var.get():.1f}")

        split_offset_var.trace_add("write", _update_split_label)

        coin_icon, coin_icon_disabled = self._load_credit_icon_pair(size=32)
        ocr_cost_var = tk.StringVar(value="0")
        postprocess_cost_var = tk.StringVar(value="0")
        ocr_insufficient_var = tk.StringVar(value="")
        postprocess_insufficient_var = tk.StringVar(value="")

        def build_coin_action(parent, title: str, cost_var: tk.StringVar, action_cb):
            frame = tk.Frame(parent, relief="groove", bd=1, padx=6, pady=4)
            title_label = tk.Label(frame, text=title)
            title_label.pack(side=tk.LEFT, padx=(2, 6))
            cost_label = tk.Label(frame, textvariable=cost_var)
            cost_label.pack(side=tk.LEFT, padx=(0, 4))
            coin_label = tk.Label(frame, image=coin_icon)
            coin_label.pack(side=tk.LEFT)
            state = {"enabled": True}
            default_title_fg = title_label.cget("fg")
            default_cost_fg = cost_label.cget("fg")

            def _set_state(enabled: bool):
                state["enabled"] = enabled
                if enabled:
                    title_label.configure(fg=default_title_fg)
                    cost_label.configure(fg=default_cost_fg)
                    active_icon = coin_icon or coin_icon_disabled
                    coin_label.configure(image=active_icon)
                    coin_label.image = active_icon
                    for w in (frame, title_label, cost_label, coin_label):
                        w.configure(cursor="hand2")
                else:
                    title_label.configure(fg="gray")
                    cost_label.configure(fg="gray")
                    disabled_icon = coin_icon_disabled or coin_icon
                    coin_label.configure(image=disabled_icon)
                    coin_label.image = disabled_icon
                    for w in (frame, title_label, cost_label, coin_label):
                        w.configure(cursor="arrow")

            def _on_click(_event=None):
                if state["enabled"]:
                    action_cb()

            for w in (frame, title_label, cost_label, coin_label):
                w.bind("<Button-1>", _on_click)

            return frame, _set_state

        action_buttons_frame = ttk.Frame(ocr_opts)
        action_buttons_frame.grid(row=0, column=4, rowspan=7, padx=10, sticky="ne")

        postprocess_button_frame, set_postprocess_enabled = build_coin_action(
            action_buttons_frame,
            "PRO постобработка",
            postprocess_cost_var,
            lambda: safe_action("PRO постобработка", run_postprocess),
        )
        postprocess_button_frame.pack(anchor="e", pady=(0, 4))
        ttk.Label(action_buttons_frame, textvariable=postprocess_insufficient_var, foreground="red")\
            .pack(anchor="e", pady=(0, 6))
        postprocess_cancel_btn = ttk.Button(
            action_buttons_frame,
            text="Отменить постобработку",
            command=lambda: cancel_postprocess(),
        )
        postprocess_cancel_btn.pack(anchor="e", pady=(0, 8))
        postprocess_cancel_btn.config(state=tk.DISABLED)

        ocr_button_frame, set_ocr_enabled = build_coin_action(
            action_buttons_frame,
            "Запустить OCR",
            ocr_cost_var,
            lambda: safe_action("OCR", run_ocr),
        )
        ocr_button_frame.pack(anchor="e", pady=(0, 4))
        ttk.Label(action_buttons_frame, textvariable=ocr_insufficient_var, foreground="red")\
            .pack(anchor="e")

        ocr_progress_var = tk.DoubleVar(value=0)
        ocr_progress_label = tk.StringVar(value="Ожидание запуска…")
        ocr_status_frame = ttk.LabelFrame(main_frame, text="OCR прогресс")
        ocr_status_frame.pack(fill=tk.X, pady=(0, 10))
        ocr_bar = ttk.Progressbar(ocr_status_frame, variable=ocr_progress_var, maximum=6)
        ocr_bar.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ocr_status_frame, textvariable=ocr_progress_label).pack(anchor="w", padx=10)

        log_frame = ttk.Frame(ocr_status_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        log_text = tk.Text(log_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        style_text_widget(log_text, self.palette)
        log_text.pack(fill=tk.BOTH, expand=True)

        ocr_task_holder = {"task": None}
        postprocess_task_holder = {"task": None}
        postprocess_proc_holder = {"proc": None}

        # Фрейм для текста OCR
        ocr_frame = ttk.LabelFrame(main_frame, text="Результат OCR")
        ocr_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        text_frame = ttk.Frame(ocr_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        txt_ocr = scrolledtext.ScrolledText(text_frame, height=15, wrap=tk.WORD)
        style_text_widget(txt_ocr, self.palette)
        create_context_menu(txt_ocr)
        txt_ocr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        attach_simple_toolbar(ocr_frame, txt_ocr)

        def append_ocr_log(message: str):
            log_text.configure(state=tk.NORMAL)
            log_text.insert(tk.END, message + "\n")
            log_text.see(tk.END)
            log_text.configure(state=tk.DISABLED)

        def handle_postprocess_event(event):
            nonlocal processed_image_path
            kind = event[0]
            if kind == "postprocess_progress":
                step, total, label, step_path = event[1:]
                ocr_bar.config(maximum=max(total, 1))
                ocr_progress_var.set(step)
                ocr_progress_label.set(label)
                append_ocr_log(label)
                update_preview_image(step_path)
            elif kind == "postprocess_done":
                result = event[1] or {}
                postprocess_state["done"] = bool(result.get("done"))
                postprocess_state["paid"] = bool(result.get("paid"))
                if result.get("path"):
                    processed_image_path = result.get("path")
                if postprocess_task_holder["task"]:
                    self.unregister_bg_handler(postprocess_task_holder["task"].queue)
                postprocess_task_holder["task"] = None
                postprocess_proc_holder["proc"] = None
                set_postprocess_enabled(True)
                postprocess_cancel_btn.config(state=tk.DISABLED)
                set_ocr_enabled(True)
                if result.get("cancelled"):
                    ocr_progress_label.set("Постобработка отменена")
                else:
                    ocr_progress_label.set("Постобработка завершена")
                update_pricing_ui()
                if postprocess_state.get("run_ocr_after"):
                    postprocess_state["run_ocr_after"] = False
                    run_ocr(skip_postprocess_prompt=True)
            elif kind == "postprocess_error":
                error_text = str(event[1] or "")
                if postprocess_task_holder["task"]:
                    self.unregister_bg_handler(postprocess_task_holder["task"].queue)
                postprocess_task_holder["task"] = None
                postprocess_proc_holder["proc"] = None
                set_postprocess_enabled(True)
                postprocess_cancel_btn.config(state=tk.DISABLED)
                set_ocr_enabled(True)
                ocr_progress_label.set("Ошибка постобработки")
                update_pricing_ui()
                log_ocr_error("PRO постобработка", error_text)
                messagebox.showerror("Ошибка постобработки", error_text)
            elif kind == "done":
                handle_postprocess_event(("postprocess_done", event[1]))
            elif kind == "error":
                handle_postprocess_event(("postprocess_error", event[1]))

        def run_postprocess():
            logging.info("Postprocess button pressed")
            if not PIL_AVAILABLE:
                messagebox.showerror(
                    "Не удалось открыть изображение",
                    "Для постобработки нужен модуль Pillow.\nУстановите его: C:\\AnkyX-main\\venv\\Scripts\\python.exe -m pip install pillow",
                )
                return
            if postprocess_task_holder["task"] is not None:
                return
            postprocess_cost = self.get_cost("postprocess", pages_count)
            if not self.can_afford(postprocess_cost):
                messagebox.showwarning("Недостаточно кредитов", "Недостаточно кредитов. Пополните")
                return
            if not self.charge(
                postprocess_cost,
                "ocr_postprocess",
                meta={
                    "operation": "postprocess",
                    "pages": pages_count,
                    "plan": self.get_pricing_plan(),
                },
            ):
                return
            ocr_bar.config(maximum=OCR_POSTPROCESS_STEPS)
            ocr_progress_var.set(0)
            ocr_progress_label.set(f"Этап 1/{OCR_POSTPROCESS_STEPS}: Запуск постобработки…")
            log_text.configure(state=tk.NORMAL)
            log_text.delete("1.0", tk.END)
            log_text.configure(state=tk.DISABLED)
            set_postprocess_enabled(False)
            set_ocr_enabled(False)
            postprocess_cancel_btn.config(state=tk.NORMAL)

            def worker(task_obj):
                step_path = original_image_path
                output_path = str(Path(tempfile.gettempdir()) / f"anky_postprocess_{uuid4().hex}_final.png")

                def emit_progress(step, label, path=None):
                    nonlocal step_path
                    if path:
                        step_path = path
                    task_obj.queue.put(("postprocess_progress", step, OCR_POSTPROCESS_STEPS, label, step_path))

                def cancel_if_needed():
                    if task_obj.cancelled():
                        result = {"done": False, "paid": True, "path": step_path, "cancelled": True}
                        task_obj.queue.put(("postprocess_done", result))
                        return result
                    return None

                try:
                    emit_progress(1, f"Step 1/{OCR_POSTPROCESS_STEPS}: loading image...", original_image_path)
                    res = cancel_if_needed()
                    if res:
                        return res

                    img = Image.open(original_image_path)
                    img = ImageOps.exif_transpose(img)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")

                    # Step 2: grayscale + autocontrast
                    gray = ImageOps.grayscale(img)
                    gray = ImageOps.autocontrast(gray, cutoff=2)
                    step2_path = str(Path(tempfile.gettempdir()) / f"anky_postprocess_{uuid4().hex}_step2.png")
                    gray.save(step2_path)
                    emit_progress(2, f"Step 2/{OCR_POSTPROCESS_STEPS}: grayscale + autocontrast...", step2_path)
                    res = cancel_if_needed()
                    if res:
                        return res

                    # Step 3: readability enhance
                    enhanced = ImageEnhance.Contrast(gray).enhance(1.35)
                    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.25)
                    step3_path = str(Path(tempfile.gettempdir()) / f"anky_postprocess_{uuid4().hex}_step3.png")
                    enhanced.save(step3_path)
                    emit_progress(3, f"Step 3/{OCR_POSTPROCESS_STEPS}: enhance readability...", step3_path)
                    res = cancel_if_needed()
                    if res:
                        return res

                    # Step 4: binarize (otsu/none/adaptive)
                    mode = (binarize_mode_var.get() or "adaptive").lower()
                    processed = enhanced
                    if mode != "none":
                        hist = processed.histogram()[:256]
                        total = processed.size[0] * processed.size[1]
                        sum_total = sum(i * hist[i] for i in range(256))
                        sum_b = 0
                        w_b = 0
                        var_max = -1
                        threshold = 127
                        for i in range(256):
                            w_b += hist[i]
                            if w_b == 0:
                                continue
                            w_f = total - w_b
                            if w_f == 0:
                                break
                            sum_b += i * hist[i]
                            m_b = sum_b / w_b
                            m_f = (sum_total - sum_b) / w_f
                            var_between = w_b * w_f * (m_b - m_f) ** 2
                            if var_between > var_max:
                                var_max = var_between
                                threshold = i
                        processed = processed.point(lambda p, t=threshold: 255 if p > t else 0)

                    # Ensure black text on white background (avoid black-page artifacts)
                    try:
                        hist2 = processed.histogram()
                        black2 = hist2[0] if hist2 else 0
                        white2 = hist2[255] if len(hist2) > 255 else 0
                        if black2 > white2:
                            processed = ImageOps.invert(processed)
                    except Exception:
                        pass


                    step4_path = str(Path(tempfile.gettempdir()) / f"anky_postprocess_{uuid4().hex}_step4.png")
                    processed.save(step4_path)
                    emit_progress(4, f"Step 4/{OCR_POSTPROCESS_STEPS}: binarize ({mode})...", step4_path)
                    res = cancel_if_needed()
                    if res:
                        return res

                    # Step 5: final
                    processed.save(output_path)
                    emit_progress(5, f"Step 5/{OCR_POSTPROCESS_STEPS}: done", output_path)
                    result = {"done": True, "paid": True, "path": output_path, "cancelled": False}
                    task_obj.queue.put(("postprocess_done", result))
                    return result
                except Exception:
                    tb = traceback.format_exc()
                    task_obj.queue.put(("postprocess_error", tb))
                    return {"done": False, "paid": True, "path": step_path, "cancelled": False}

            postprocess_task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(postprocess_task_holder["task"].queue, handle_postprocess_event)

        def cancel_postprocess():
            if postprocess_task_holder["task"]:
                postprocess_task_holder["task"].cancel()
                proc = postprocess_proc_holder.get("proc")
                if proc is not None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                logging.info("Postprocess cancel requested")
                append_ocr_log("Отмена постобработки запрошена…")

        def handle_ocr_event(event):
            kind = event[0]
            if kind == "ocr_progress":
                step, total, label = event[1:]
                ocr_bar.config(maximum=max(total, 1))
                ocr_progress_var.set(step)
                ocr_progress_label.set(label)
                append_ocr_log(label)
            elif kind == "log":
                append_ocr_log(str(event[1]))
            elif kind == "done":
                ocr_text = (event[1] or "")
                if ocr_task_holder["task"]:
                    self.unregister_bg_handler(ocr_task_holder["task"].queue)
                ocr_task_holder["task"] = None
                ocr_progress_label.set("Готово")
                set_ocr_enabled(True)
                ocr_state["done"] = True
                def _update_text():
                    prev_state = txt_ocr.cget("state")
                    txt_ocr.configure(state=tk.NORMAL)
                    txt_ocr.delete("1.0", tk.END)
                    txt_ocr.insert("1.0", str(ocr_text))
                    txt_ocr.configure(state=prev_state)

                self.after(0, _update_text)
            elif kind == "error":
                if ocr_task_holder["task"]:
                    self.unregister_bg_handler(ocr_task_holder["task"].queue)
                ocr_task_holder["task"] = None
                ocr_progress_label.set("Ошибка")
                set_ocr_enabled(True)
                error_text = str(event[1] or "")
                log_ocr_error("OCR", error_text)
                messagebox.showerror("Ошибка OCR", error_text)

        def run_ocr(skip_postprocess_prompt: bool = False):
            logging.info("OCR button pressed")
            if not PIL_AVAILABLE:
                messagebox.showerror(
                    "Не удалось открыть изображение",
                    "Для OCR нужен модуль Pillow.\nУстановите его: C:\\AnkyX-main\\venv\\Scripts\\python.exe -m pip install pillow",
                )
                return
            if ocr_task_holder["task"] is not None:
                return
            selected_mode = ocr_mode_var.get()
            selected_lang = lang_mode_var.get()
            selected_engine = ocr_engine_var.get()
            logging.info(
                "OCR requested mode=%s engine=%s lang=%s",
                selected_mode,
                selected_engine,
                selected_lang,
            )
            if selected_engine == "PaddleOCR" and not CV2_AVAILABLE:
                messagebox.showerror(
                    "Недоступна обработка изображения",
                    "Для PaddleOCR нужен OpenCV и NumPy.\n"
                    "Установите пакеты: C:\\AnkyX-main\\venv\\Scripts\\python.exe -m pip install opencv-python numpy",
                )
                return
            if selected_engine == "PaddleOCR":
                if not is_paddle_available():
                    messagebox.showinfo(
                        "PaddleOCR недоступен",
                        "PaddleOCR недоступен на этой системе (Illegal instruction). "
                        "Используйте Tesseract/EasyOCR.",
                    )
                    return
                if selected_mode != "pro":
                    selected_mode = "pro"
                    ocr_mode_var.set("pro")
                _ensure_ocr_photo_loaded()
            elif selected_mode == "pro":
                selected_mode = "fast"
                ocr_mode_var.set("fast")
                messagebox.showinfo(
                    "Режим PRO",
                    "PRO режим доступен только для PaddleOCR. Переключено на быстрый режим.",
                )

            if selected_engine == "Tesseract":
                if not _ensure_deu_rus_present(selected_lang):
                    return
                if not _ensure_required_lang_files():
                    return
            if use_postprocess_var.get() and not postprocess_state["done"] and not skip_postprocess_prompt:
                if messagebox.askyesno(
                    "Постобработка перед OCR",
                    "Сначала выполнить PRO постобработку?",
                ):
                    postprocess_state["run_ocr_after"] = True
                    run_postprocess()
                    return
            ocr_cost = self.get_cost("ocr", pages_count)
            if not self.can_afford(ocr_cost):
                messagebox.showwarning("Недостаточно кредитов", "Недостаточно кредитов. Пополните")
                return
            if not self.charge(
                ocr_cost,
                "ocr_image",
                meta={
                    "operation": "ocr_image",
                    "ocr_mode": selected_mode,
                    "pages": pages_count,
                    "plan": self.get_pricing_plan(),
                },
            ):
                return

            ocr_progress_var.set(0)
            ocr_progress_label.set("Запуск…")
            log_text.configure(state=tk.NORMAL)
            log_text.delete("1.0", tk.END)
            log_text.configure(state=tk.DISABLED)
            set_ocr_enabled(False)

            def worker(task_obj):
                def progress_cb(step, total, label):
                    task_obj.queue.put(("ocr_progress", step, total, label))

                try:
                    use_processed = bool(postprocess_state["done"] and processed_image_path)
                    engine_key = selected_engine.lower()
                    if engine_key == "paddleocr":
                        engine_key = "paddle"
                    elif engine_key == "easyocr":
                        engine_key = "easyocr"
                    else:
                        engine_key = "tesseract"

                    options = OcrRunOptions(
                        ocr_mode=selected_mode,
                        ocr_engine=engine_key,
                        lang_mode=selected_lang,
                        perspective_correction=not use_processed,
                        flatten_background=not use_processed,
                        binarize_mode=binarize_mode_var.get(),
                        deskew=not use_processed,
                        debug_images=debug_images_var.get(),
                        psm=int(psm_var.get()),
                        dictionary_mode=dictionary_mode_var.get(),
                        split_offset_percent=float(split_offset_var.get()),
                        preserve_spaces=True,
                        prefer_paddle_for_columns=True,
                        preprocess_preset="none" if use_processed else preprocess_preset_var.get(),
                    )
                    task_obj.queue.put(
                        ("log", f"OCR режим: {options.ocr_mode}, движок={options.ocr_engine}, lang={options.lang_mode}")
                    )
                    src_path = processed_image_path if postprocess_state["done"] and processed_image_path else img_path
                    text = perform_page_ocr(src_path, options, progress_cb)
                    return text
                except Exception:
                    tb = traceback.format_exc()
                    task_obj.queue.put(("log", tb))
                    task_obj.queue.put(("error", tb))
                    return ""

            ocr_task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(ocr_task_holder["task"].queue, handle_ocr_event)

        def update_pricing_ui(*_args):
            postprocess_cost = self.get_cost("postprocess", pages_count)
            ocr_cost = self.get_cost("ocr", pages_count)
            total_for_ocr = ocr_cost
            if use_postprocess_var.get() and not postprocess_state["done"]:
                total_for_ocr += postprocess_cost

            postprocess_cost_var.set(str(postprocess_cost))
            ocr_cost_var.set(str(total_for_ocr))

            if postprocess_task_holder["task"] is None and self.can_afford(postprocess_cost):
                postprocess_insufficient_var.set("")
                set_postprocess_enabled(True)
            else:
                if postprocess_task_holder["task"] is None:
                    postprocess_insufficient_var.set("Недостаточно кредитов. Пополните")
                else:
                    postprocess_insufficient_var.set("")
                set_postprocess_enabled(False)

            if ocr_task_holder["task"] is None and self.can_afford(total_for_ocr):
                ocr_insufficient_var.set("")
                set_ocr_enabled(True)
            else:
                if ocr_task_holder["task"] is None:
                    ocr_insufficient_var.set("Недостаточно кредитов. Пополните")
                else:
                    ocr_insufficient_var.set("")
                set_ocr_enabled(False)

        use_postprocess_var.trace_add("write", update_pricing_ui)
        self.register_balance_observer(update_pricing_ui)
        update_pricing_ui()

        # Фрейм с настройками генерации
        settings_frame = ttk.LabelFrame(main_frame, text="Генерация карточек из OCR")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Язык карточек:").pack(anchor="w", padx=10, pady=(5, 0))
        ocr_lang_var = tk.StringVar(value="DE")
        ttk.Combobox(
            settings_frame,
            textvariable=ocr_lang_var,
            values=("DE", "EN", "FR", "ES", "IT", "RU"),
            state="readonly",
            width=8,
        ).pack(anchor="w", padx=10, pady=(0, 5))

        self._build_generation_template_selector(settings_frame, self.selected_deck_id, padx=10, pady=(5, 5))

        ttk.Label(
            settings_frame,
            text="Переменные: {question}, {translation}, {sentence_with_gap}, {word}, {ipa}, {gender}, {plural}, {sentence}"
        ).pack(anchor="w", padx=10, pady=(0, 5))

        cards_lang_mode_var = tk.StringVar(value="foreign")
        sentence_limit_var = tk.StringVar(value="Лимит предложений на карточку: 1")

        def update_sentence_limit(*_args):
            if cards_lang_mode_var.get() == "native":
                sentence_limit_var.set("Лимит предложений на карточку: 10")
            else:
                sentence_limit_var.set("Лимит предложений на карточку: 1")

        cards_lang_mode_var.trace_add("write", update_sentence_limit)
        update_sentence_limit()

        progress_frame = ttk.LabelFrame(main_frame, text="Прогресс генерации карточек (пакет 25)")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="")
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=CARDS_GEN_PACK_SIZE)
        progress_bar.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=10)

        media_progress_frame = ttk.LabelFrame(main_frame, text="Прогресс генерации медиа (пакет 20)")
        media_progress_frame.pack(fill=tk.X, pady=(0, 10))
        media_progress_var = tk.DoubleVar(value=0)
        media_progress_label_var = tk.StringVar(value="")
        media_progress_bar = ttk.Progressbar(media_progress_frame, variable=media_progress_var, maximum=VIDEO_GEN_PACK_SIZE)
        media_progress_bar.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(media_progress_frame, textvariable=media_progress_label_var).pack(anchor="w", padx=10)

        log_box = tk.Text(main_frame, height=6, state="disabled")
        style_text_widget(log_box, self.palette)
        log_box.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        def append_log(message: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, message + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        task_holder = {"task": None}
        image_task_holder = {"task": None}
        video_task_holder = {"task": None}
        cards_pack_state = {"skip_sentences": set()}
        image_state = {"processed": set()}
        video_state = {"processed": set()}
        generated_card_ids: list[int] = []

        cards_cost_var = tk.StringVar(value="0")
        video_cost_var = tk.StringVar(value="0")
        cards_insufficient_var = tk.StringVar(value="")
        video_insufficient_var = tk.StringVar(value="")

        def get_sentence_limit_value() -> int:
            return 10 if cards_lang_mode_var.get() == "native" else 1

        def get_remaining_sentences(text: str) -> list[str]:
            sentences = split_ocr_text_into_sentences(text)
            return [s for s in sentences if normalize_word(s) not in cards_pack_state["skip_sentences"]]

        def update_cards_pricing_ui(*_args):
            cards_cost = self.get_cost("ocr_cards_gen", 1)
            video_cost = self.get_cost("video_gen", VIDEO_GEN_PACK_SIZE)
            cards_cost_var.set(str(cards_cost))
            video_cost_var.set(str(video_cost))
            cards_generate_btn.configure(text=f"Сгенерировать карточки {cards_cost}")

            if task_holder["task"] is None and self.can_afford(cards_cost):
                cards_insufficient_var.set("")
                set_cards_enabled(True)
            else:
                if task_holder["task"] is None:
                    cards_insufficient_var.set("Недостаточно кредитов. Пополните")
                else:
                    cards_insufficient_var.set("")
                set_cards_enabled(False)

            if video_task_holder["task"] is None and self.can_afford(video_cost):
                video_insufficient_var.set("")
                set_video_enabled(True)
            else:
                if video_task_holder["task"] is None:
                    video_insufficient_var.set("Недостаточно кредитов. Пополните")
                else:
                    video_insufficient_var.set("")
                set_video_enabled(False)

        def run_generation():
            if task_holder["task"] is not None:
                return
            text = txt_ocr.get("1.0", tk.END).strip()
            if not text:
                messagebox.showerror("Ошибка", "Текст пустой.")
                return

            generated_card_ids.clear()
            image_state["processed"].clear()

            remaining_items = get_remaining_sentences(text)
            if not remaining_items:
                messagebox.showinfo("Результат", "Новых предложений не найдено.")
                return

            cards_cost = self.get_cost("ocr_cards_gen", 1)
            if not self.can_afford(cards_cost):
                messagebox.showwarning("Недостаточно кредитов", "Недостаточно кредитов. Пополните")
                return
            if not self.charge(
                cards_cost,
                "ocr_cards_gen",
                meta={
                    "operation": "ocr_cards_gen",
                    "pack_size": CARDS_GEN_PACK_SIZE,
                    "language": ocr_lang_var.get(),
                    "language_mode": cards_lang_mode_var.get(),
                    "plan": self.get_pricing_plan(),
                },
            ):
                return

            placeholder_path = self._ensure_generated_placeholder_image()
            if not placeholder_path:
                messagebox.showerror("Ошибка", "Не удалось создать placeholder-картинку.")
                return

            progress_var.set(0)
            progress_label_var.set("")
            progress_bar.configure(maximum=CARDS_GEN_PACK_SIZE)
            log_box.configure(state="normal")
            log_box.delete("1.0", tk.END)
            log_box.configure(state="disabled")

            def handle_event(event):
                kind = event[0]
                if kind == "progress":
                    done, total, label = event[1:]
                    progress_bar.config(maximum=max(total, 1))
                    progress_var.set(done)
                    progress_label_var.set(f"{label}: {done}/{total}")
                elif kind == "log":
                    append_log(event[1])
                elif kind == "done":
                    result = event[1] or 0
                    if task_holder["task"]:
                        self.unregister_bg_handler(task_holder["task"].queue)
                    task_holder["task"] = None
                    set_cards_enabled(True)
                    btn_cancel.config(state=tk.DISABLED)
                    image_state["processed"].update(generated_card_ids)
                    remaining_after = get_remaining_sentences(text)
                    progress_var.set(0)
                    progress_label_var.set("")
                    if result == 0:
                        messagebox.showinfo("Результат", "Новых предложений не найдено.")
                    else:
                        messagebox.showinfo("Результат", f"Создано карточек: {result}")
                    if remaining_after:
                        messagebox.showinfo(
                            "Лимит пакета",
                            "Достигнут лимит 25 карточек. Оплатите следующий пакет для продолжения.",
                        )
                    update_cards_pricing_ui()
                elif kind == "error":
                    if task_holder["task"]:
                        self.unregister_bg_handler(task_holder["task"].queue)
                    task_holder["task"] = None
                    set_cards_enabled(True)
                    btn_cancel.config(state=tk.DISABLED)
                    messagebox.showerror("Ошибка", event[1])
                    update_cards_pricing_ui()

            def worker(task_obj):
                return auto_generate_cards_from_ocr_text(
                    self.selected_deck_id,
                    text,
                    cards_lang_mode_var.get(),
                    progress_queue=task_obj.queue,
                    cancel_check=task_obj.cancelled,
                    max_cards=CARDS_GEN_PACK_SIZE,
                    created_card_ids=generated_card_ids,
                    placeholder_image_path=placeholder_path,
                    skip_sentences=cards_pack_state["skip_sentences"],
                )

            set_cards_enabled(False)
            btn_cancel.config(state=tk.NORMAL)
            task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(task_holder["task"].queue, handle_event)

        def cancel_generation():
            if task_holder["task"]:
                task_holder["task"].cancel()
                append_log("Отмена запрошена…")
                btn_cancel.config(state=tk.DISABLED)

        def generate_placeholder_images():
            if image_task_holder["task"] is not None:
                return
            if not generated_card_ids:
                messagebox.showinfo("Нет карточек", "Сначала сгенерируйте карточки.")
                return
            placeholder_path = self._ensure_generated_placeholder_image()
            if not placeholder_path:
                messagebox.showerror("Ошибка", "Не удалось создать placeholder-картинку.")
                return
            media_progress_var.set(0)
            media_progress_label_var.set("")
            remaining = [cid for cid in generated_card_ids if cid not in image_state["processed"]]
            media_progress_bar.configure(maximum=max(len(remaining), 1))

            def handle_event(event):
                kind = event[0]
                if kind == "progress":
                    done, total, label = event[1:]
                    media_progress_var.set(done)
                    media_progress_label_var.set(f"{label}: {done}/{total}")
                elif kind == "done":
                    if image_task_holder["task"]:
                        self.unregister_bg_handler(image_task_holder["task"].queue)
                    image_task_holder["task"] = None
                    btn_image_cancel.config(state=tk.DISABLED)
                    messagebox.showinfo("Готово", "Заглушки картинок созданы.")
                elif kind == "error":
                    if image_task_holder["task"]:
                        self.unregister_bg_handler(image_task_holder["task"].queue)
                    image_task_holder["task"] = None
                    btn_image_cancel.config(state=tk.DISABLED)
                    messagebox.showerror("Ошибка", event[1])

            def worker(task_obj):
                done = 0
                total = len(remaining)
                for card_id in remaining:
                    if task_obj.cancelled():
                        break
                    card = get_card_by_id(card_id)
                    if not card:
                        continue
                    if not card.get("front_image_path") and not card.get("image_path"):
                        try:
                            conn = get_connection()
                            conn.execute(
                                "UPDATE cards SET front_image_path = ?, image_path = ? WHERE id = ?;",
                                (placeholder_path, placeholder_path, card_id),
                            )
                            conn.commit()
                            conn.close()
                            attach_media_to_card(card_id, [(placeholder_path, "image", "front", "ocr_placeholder")])
                        except Exception:
                            pass
                    image_state["processed"].add(card_id)
                    done += 1
                    task_obj.queue.put(("progress", done, max(total, 1), "Картинки"))
                return done

            btn_image_cancel.config(state=tk.NORMAL)
            image_task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(image_task_holder["task"].queue, handle_event)

        def cancel_image_generation():
            if image_task_holder["task"]:
                image_task_holder["task"].cancel()
                append_log("Отмена генерации картинок запрошена…")
                btn_image_cancel.config(state=tk.DISABLED)

        def generate_placeholder_videos():
            if video_task_holder["task"] is not None:
                return
            if not generated_card_ids:
                messagebox.showinfo("Нет карточек", "Сначала сгенерируйте карточки.")
                return
            pending_ids = [cid for cid in generated_card_ids if cid not in video_state["processed"]]
            if not pending_ids:
                messagebox.showinfo("Видео", "Видео уже сгенерированы для всех карточек.")
                return
            video_cost = self.get_cost("video_gen", VIDEO_GEN_PACK_SIZE)
            if not self.can_afford(video_cost):
                messagebox.showwarning("Недостаточно кредитов", "Недостаточно кредитов. Пополните")
                return
            if not self.charge(
                video_cost,
                "video_gen",
                meta={
                    "operation": "video_gen",
                    "pack_size": VIDEO_GEN_PACK_SIZE,
                    "plan": self.get_pricing_plan(),
                },
            ):
                return

            batch = pending_ids[:VIDEO_GEN_PACK_SIZE]
            media_progress_var.set(0)
            media_progress_label_var.set("")
            media_progress_bar.configure(maximum=max(VIDEO_GEN_PACK_SIZE, 1))

            def handle_event(event):
                kind = event[0]
                if kind == "progress":
                    done, total, label = event[1:]
                    media_progress_var.set(done)
                    media_progress_label_var.set(f"{label}: {done}/{total}")
                elif kind == "done":
                    if video_task_holder["task"]:
                        self.unregister_bg_handler(video_task_holder["task"].queue)
                    video_task_holder["task"] = None
                    btn_video_cancel.config(state=tk.DISABLED)
                    pending_after = [cid for cid in generated_card_ids if cid not in video_state["processed"]]
                    media_progress_var.set(0)
                    media_progress_label_var.set("")
                    if pending_after:
                        messagebox.showinfo(
                            "Лимит пакета",
                            "Достигнут лимит 20 видео. Оплатите следующий пакет для продолжения.",
                        )
                    else:
                        messagebox.showinfo("Готово", "Видео заглушки созданы.")
                    update_cards_pricing_ui()
                elif kind == "error":
                    if video_task_holder["task"]:
                        self.unregister_bg_handler(video_task_holder["task"].queue)
                    video_task_holder["task"] = None
                    btn_video_cancel.config(state=tk.DISABLED)
                    messagebox.showerror("Ошибка", event[1])
                    update_cards_pricing_ui()

            def worker(task_obj):
                done = 0
                total = VIDEO_GEN_PACK_SIZE
                for card_id in batch:
                    if task_obj.cancelled():
                        break
                    placeholder = self._create_ai_placeholder_video()
                    if placeholder:
                        try:
                            attach_media_to_card(card_id, [(placeholder, "video", "front", "ocr_placeholder")])
                        except Exception:
                            pass
                    video_state["processed"].add(card_id)
                    done += 1
                    task_obj.queue.put(("progress", done, max(total, 1), "Видео"))
                return done

            btn_video_cancel.config(state=tk.NORMAL)
            video_task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(video_task_holder["task"].queue, handle_event)

        def cancel_video_generation():
            if video_task_holder["task"]:
                video_task_holder["task"].cancel()
                append_log("Отмена генерации видео запрошена…")
                btn_video_cancel.config(state=tk.DISABLED)

        def open_cards_preview():
            if not generated_card_ids:
                messagebox.showinfo("Нет карточек", "Сначала сгенерируйте карточки.")
                return
            preview_win = tk.Toplevel(win)
            preview_win.title("Предпросмотр карточек")
            preview_win.geometry("980x680")
            preview_win.grab_set()
            apply_dark_theme_to_window(preview_win, self.palette)

            preview_state = {"index": 0}
            container = tk.Frame(preview_win, bg=DARK_BG)
            container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

            nav_frame = tk.Frame(container, bg=DARK_BG)
            nav_frame.pack(fill=tk.X, pady=(0, 8))
            side_var = tk.StringVar(value="front")

            def set_side(side: str):
                side_var.set(side)
                load_card(preview_state["index"])

            card_wrap = tk.Frame(
                container,
                bg=DARK_BG,
                highlightbackground=CARD_BORDER,
                highlightthickness=1,
                bd=0,
                width=CARD_VIEW_WIDTH,
                height=CARD_VIEW_HEIGHT,
            )
            card_wrap.pack(padx=10, pady=10)
            card_wrap.pack_propagate(False)
            renderer = CardRenderer(
                card_wrap,
                palette=self.palette,
                editable=False,
                show_image_toolbar=False,
                image_layout="side",
                show_media_placeholder=True,
                fixed_media_slot=REPEAT_MEDIA_SLOT_SIZE,
                render_mode="preview",
            )

            def load_card(index: int):
                idx = max(0, min(index, len(generated_card_ids) - 1))
                preview_state["index"] = idx
                card_id = generated_card_ids[idx]
                card = get_card_by_id(card_id) or {}
                card["video_path"] = find_video_media_path(card) or ""
                show_back = side_var.get() == "back"
                header_text = f"Карточка {idx + 1}/{len(generated_card_ids)} | ID {card_id}"
                renderer.render(card, show_back=show_back, header_text=header_text)

            def show_next():
                load_card(preview_state["index"] + 1)

            def show_prev():
                load_card(preview_state["index"] - 1)

            ttk.Button(nav_frame, text="Лицевая", command=lambda: set_side("front")).pack(side=tk.LEFT, padx=4)
            ttk.Button(nav_frame, text="Обратная", command=lambda: set_side("back")).pack(side=tk.LEFT, padx=4)
            ttk.Button(nav_frame, text="Назад", command=show_prev).pack(side=tk.LEFT, padx=4)
            ttk.Button(nav_frame, text="Вперед", command=show_next).pack(side=tk.LEFT, padx=4)

            load_card(0)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        left_tools = ttk.Frame(btn_frame)
        left_tools.pack(side=tk.LEFT)
        right_actions = ttk.Frame(btn_frame)
        right_actions.pack(side=tk.RIGHT)

        ttk.Button(left_tools, text="Копировать текст",
                   command=lambda: win.clipboard_clear() or win.clipboard_append(txt_ocr.get("1.0", tk.END)))\
            .pack(side=tk.LEFT, padx=5)

        ttk.Button(left_tools, text="Вставить из буфера",
                   command=lambda: txt_ocr.insert(tk.END, win.clipboard_get()))\
            .pack(side=tk.LEFT, padx=5)

        ttk.Button(left_tools, text="Очистить",
                   command=lambda: txt_ocr.delete("1.0", tk.END))\
            .pack(side=tk.LEFT, padx=5)

        mode_frame = ttk.LabelFrame(right_actions, text="Режим генерации")
        mode_frame.pack(anchor="e", padx=5, pady=(0, 6), fill=tk.X)
        ttk.Radiobutton(
            mode_frame,
            text="Иностранный язык",
            variable=cards_lang_mode_var,
            value="foreign",
        ).pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Radiobutton(
            mode_frame,
            text="Родной язык",
            variable=cards_lang_mode_var,
            value="native",
        ).pack(anchor="w", padx=6)
        ttk.Label(mode_frame, textvariable=sentence_limit_var).pack(anchor="w", padx=6, pady=(0, 4))

        cards_generate_btn = tk.Button(
            right_actions,
            text="Сгенерировать карточки",
            image=coin_icon,
            compound=tk.RIGHT,
            command=run_generation,
            padx=10,
            pady=6,
        )
        cards_generate_btn.pack(anchor="e", padx=5, pady=(0, 2))
        cards_generate_btn.image = coin_icon

        def set_cards_enabled(enabled: bool):
            if enabled:
                cards_generate_btn.configure(state=tk.NORMAL, image=coin_icon)
                cards_generate_btn.image = coin_icon
            else:
                disabled_icon = coin_icon_disabled or coin_icon
                cards_generate_btn.configure(state=tk.DISABLED, image=disabled_icon)
                cards_generate_btn.image = disabled_icon

        ttk.Label(right_actions, textvariable=cards_insufficient_var, foreground="red").pack(anchor="e")

        btn_cancel = ttk.Button(right_actions, text="Остановить генерацию", command=cancel_generation)
        btn_cancel.pack(anchor="e", padx=5, pady=(6, 2))
        btn_cancel.config(state=tk.DISABLED)

        ttk.Button(right_actions, text="Предпросмотр карточек", command=open_cards_preview)\
            .pack(anchor="e", padx=5, pady=(6, 2))

        ttk.Button(right_actions, text="Сгенерировать картинки", command=generate_placeholder_images)\
            .pack(anchor="e", padx=5, pady=(6, 2))

        btn_image_cancel = ttk.Button(right_actions, text="Остановить картинки", command=cancel_image_generation)
        btn_image_cancel.pack(anchor="e", padx=5, pady=(0, 6))
        btn_image_cancel.config(state=tk.DISABLED)

        video_button_frame, set_video_enabled = build_coin_action(
            right_actions,
            "Сгенерировать видео",
            video_cost_var,
            lambda: generate_placeholder_videos(),
        )
        video_button_frame.pack(anchor="e", padx=5, pady=(0, 2))
        ttk.Label(right_actions, textvariable=video_insufficient_var, foreground="red").pack(anchor="e")

        btn_video_cancel = ttk.Button(right_actions, text="Остановить видео", command=cancel_video_generation)
        btn_video_cancel.pack(anchor="e", padx=5, pady=(0, 6))
        btn_video_cancel.config(state=tk.DISABLED)

        self.register_balance_observer(update_cards_pricing_ui)
        update_cards_pricing_ui()

        def on_close():
            self.unregister_balance_observer(update_pricing_ui)
            self.unregister_balance_observer(update_cards_pricing_ui)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    # --------- генерация через цифровой слух ---------
    # PATCH: digital-ear STT -> auto punctuation -> 1 sentence per card -> preview+invoice -> paid generate with SD placeholder; plan-based pricing
    def open_generate_from_speech_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        win = tk.Toplevel(self)
        win.title("Цифровой слух")
        win.geometry("820x720")
        win.grab_set()
        apply_dark_theme_to_window(win, self.palette)

        main_frame = ttk.Frame(win, style="Surface.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        ttk.Label(
            main_frame,
            text="Цифровой слух: запись → автопунктуация → предпросмотр → оплата → карточки",
            style="Section.TLabel",
            wraplength=760,
        ).pack(anchor="w", pady=(0, 8))

        self._build_generation_template_selector(main_frame, self.selected_deck_id, padx=0, pady=(0, 10))

        record_frame = ttk.LabelFrame(main_frame, text="Запись", style="Card.TLabelframe")
        record_frame.pack(fill=tk.X, pady=(0, 10))
        style_card(record_frame, self.palette, padded=True)

        dur_row = ttk.Frame(record_frame)
        dur_row.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(dur_row, text="Длительность записи (сек):").pack(side=tk.LEFT)
        entry_dur = ttk.Entry(dur_row, width=10)
        entry_dur.insert(0, "10")
        entry_dur.pack(side=tk.LEFT, padx=(8, 0))
        create_context_menu(entry_dur)

        current_mic_text = (
            f"Текущее устройство: index={self.microphone_index}"
            if self.microphone_index is not None
            else "Текущее устройство: по умолчанию системы"
        )
        ttk.Label(record_frame, text=current_mic_text).pack(anchor="w", padx=6, pady=(0, 6))

        record_status_var = tk.StringVar(value="")
        ttk.Label(record_frame, textvariable=record_status_var, style="Muted.TLabel").pack(
            anchor="w", padx=6, pady=(0, 6)
        )

        record_btn = ttk.Button(record_frame, text="Записать")
        record_btn.pack(anchor="w", padx=6, pady=(0, 6))

        text_frame = ttk.LabelFrame(main_frame, text="Текст", style="Card.TLabelframe")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        style_card(text_frame, self.palette, padded=True)

        text_box = scrolledtext.ScrolledText(text_frame, height=10)
        style_text_widget(text_box, self.palette)
        text_box.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        create_context_menu(text_box)

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(0, 6))

        preview_btn = ttk.Button(action_frame, text="Подготовить карточки", width=24)
        preview_btn.pack(side=tk.LEFT, padx=(0, 6))

        coin_icon, coin_icon_disabled = self._load_credit_icon_pair(size=32)
        invoice_total_var = tk.StringVar(value="0")
        insufficient_var = tk.StringVar(value="")

        invoice_frame = ttk.Frame(action_frame)
        invoice_frame.pack(side=tk.LEFT, padx=(10, 6))
        ttk.Label(invoice_frame, text="Счет:").pack(side=tk.LEFT, padx=(0, 6))
        if coin_icon:
            coin_label = tk.Label(invoice_frame, image=coin_icon)
            coin_label.image = coin_icon
            coin_label.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(invoice_frame, textvariable=invoice_total_var).pack(side=tk.LEFT)

        generate_btn = tk.Button(
            action_frame,
            text="Сгенерировать — Итого: 0",
            image=coin_icon,
            compound=tk.LEFT,
            padx=10,
            pady=6,
        )
        generate_btn.pack(side=tk.RIGHT)
        generate_btn.image = coin_icon

        ttk.Label(main_frame, textvariable=insufficient_var, foreground="red").pack(anchor="e")

        self.pending_invoice = None
        record_task_holder = {"task": None}

        def update_generate_button_state():
            invoice = self.pending_invoice
            if not invoice:
                invoice_total_var.set("0")
                generate_btn.configure(state=tk.DISABLED, text="Сгенерировать — Итого: 0")
                if coin_icon_disabled or coin_icon:
                    generate_btn.configure(image=coin_icon_disabled or coin_icon)
                    generate_btn.image = coin_icon_disabled or coin_icon
                insufficient_var.set("")
                return
            total = int(invoice.get("total") or 0)
            invoice_total_var.set(str(total))
            generate_btn.configure(text=f"Сгенерировать — Итого: {total}")
            if self.can_afford(total):
                generate_btn.configure(state=tk.NORMAL, image=coin_icon or coin_icon_disabled)
                generate_btn.image = coin_icon or coin_icon_disabled
                insufficient_var.set("")
            else:
                generate_btn.configure(state=tk.DISABLED, image=coin_icon_disabled or coin_icon)
                generate_btn.image = coin_icon_disabled or coin_icon
                insufficient_var.set("Недостаточно кредитов для генерации")

        def record_worker(task_obj):
            try:
                duration = int(entry_dur.get().strip() or "0")
            except ValueError:
                raise ValueError("Длительность должна быть целым числом секунд.")
            if duration <= 0:
                raise ValueError("Длительность должна быть > 0.")
            return record_speech_to_text(
                duration,
                self.microphone_index,
                progress_queue=task_obj.queue,
                cancel_check=task_obj.cancelled,
            )

        def handle_record_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                record_status_var.set(f"{label}: {int(done)}/{max(total, 1)}")
                return
            if kind == "done":
                result = event[1]
                task_obj = record_task_holder.get("task")
                if task_obj:
                    self.unregister_bg_handler(task_obj.queue)
                record_task_holder["task"] = None
                record_btn.config(state=tk.NORMAL)
                if isinstance(result, tuple):
                    raw_text = result[0] or ""
                else:
                    raw_text = str(result or "")
                punct = auto_punctuate(raw_text)
                if punct:
                    text_box.delete("1.0", tk.END)
                    text_box.insert("1.0", punct)
                    record_status_var.set("Речь распознана и добавлена в текст.")
                else:
                    record_status_var.set("Распознано пусто.")
                return
            if kind == "error":
                task_obj = record_task_holder.get("task")
                if task_obj:
                    self.unregister_bg_handler(task_obj.queue)
                record_task_holder["task"] = None
                record_btn.config(state=tk.NORMAL)
                record_status_var.set("")
                show_stt_error_dialog(win, "Ошибка записи", event[1])
                return

        def start_record():
            if not SR_AVAILABLE:
                messagebox.showerror(
                    "Речь недоступна",
                    "Чтобы записывать речь, установите SpeechRecognition и PyAudio:\n"
                    "pip install SpeechRecognition pyaudio",
                )
                return
            if record_task_holder["task"] is not None:
                return
            record_status_var.set("Запись…")
            record_btn.config(state=tk.DISABLED)
            task_obj = start_background_task(record_worker)
            record_task_holder["task"] = task_obj
            self.register_bg_handler(task_obj.queue, handle_record_event)

        def build_preview_cards(sentences: list[str]) -> list[dict]:
            preview_cards: list[dict] = []
            for sentence in sentences:
                placeholder_path = generate_sd_image_placeholder(sentence, None) or ""
                front, back = render_generation_faces(
                    self.selected_deck_id,
                    {
                        "translation": "",
                        "front_translation": "",
                        "sentence_with_gap": sentence,
                        "word": sentence,
                        "ipa": "",
                        "gender": "",
                        "plural": "",
                        "sentence": sentence,
                    },
                )
                preview_cards.append(
                    {
                        "front": front,
                        "back": back,
                        "front_rich": {"text": front, "tags": []},
                        "back_rich": {"text": back, "tags": []},
                        "front_image_path": placeholder_path,
                        "back_image_path": "",
                        "audio_path": None,
                    }
                )
            return preview_cards

        def open_cards_preview(cards: list[dict]) -> None:
            if not cards:
                messagebox.showwarning("Предпросмотр", "Нет карточек для предпросмотра.")
                return
            preview_win = tk.Toplevel(win)
            preview_win.title("Предпросмотр карточек")
            preview_win.geometry("980x680")
            preview_win.grab_set()
            apply_dark_theme_to_window(preview_win, self.palette)

            preview_state = {"index": 0, "side": "front"}
            container = tk.Frame(preview_win, bg=DARK_BG)
            container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

            nav_frame = tk.Frame(container, bg=DARK_BG)
            nav_frame.pack(fill=tk.X, pady=(0, 8))
            idx_var = tk.StringVar(value="1/1")

            def update_index_label():
                idx_var.set(f"{preview_state['index'] + 1}/{len(cards)}")

            ttk.Label(nav_frame, textvariable=idx_var).pack(side=tk.LEFT, padx=(0, 12))
            ttk.Button(nav_frame, text="Лицевая", command=lambda: set_side("front")).pack(
                side=tk.LEFT, padx=4
            )
            ttk.Button(nav_frame, text="Обратная", command=lambda: set_side("back")).pack(
                side=tk.LEFT, padx=4
            )
            ttk.Button(nav_frame, text="Назад", command=lambda: move_index(-1)).pack(side=tk.LEFT, padx=4)
            ttk.Button(nav_frame, text="Вперед", command=lambda: move_index(1)).pack(side=tk.LEFT, padx=4)

            card_wrap = tk.Frame(
                container,
                bg=DARK_BG,
                highlightbackground=CARD_BORDER,
                highlightthickness=1,
                bd=0,
                width=CARD_VIEW_WIDTH,
                height=CARD_VIEW_HEIGHT,
            )
            card_wrap.pack(padx=10, pady=10)
            card_wrap.pack_propagate(False)
            renderer = CardRenderer(
                card_wrap,
                palette=self.palette,
                editable=False,
                show_image_toolbar=False,
                image_layout="side",
                show_media_placeholder=False,
                fixed_media_slot=REPEAT_MEDIA_SLOT_SIZE,
                render_mode="preview",
            )
            repeat_slot_w, repeat_slot_h = renderer.get_repeat_media_slot_size()
            renderer.image_container.config(width=repeat_slot_w, height=repeat_slot_h)
            renderer.image_container.pack_propagate(False)
            renderer.image_container.grid_propagate(False)

            def render_current():
                idx = preview_state["index"]
                side = preview_state["side"]
                card = cards[idx]
                header_text = f"Карточка {idx + 1}/{len(cards)}"
                renderer.render(card, show_back=(side == "back"), prefer_audio_side=side, header_text=header_text)
                update_index_label()

            def move_index(delta: int) -> None:
                new_index = preview_state["index"] + delta
                if new_index < 0 or new_index >= len(cards):
                    return
                preview_state["index"] = new_index
                render_current()

            def set_side(side: str) -> None:
                preview_state["side"] = side
                render_current()

            render_current()

        def prepare_preview():
            try:
                raw_text = text_box.get("1.0", tk.END).strip()
                if not raw_text:
                    messagebox.showinfo("Нет текста", "Нет текста для подготовки карточек.")
                    return
                punct_text = auto_punctuate(raw_text)
                if punct_text:
                    text_box.delete("1.0", tk.END)
                    text_box.insert("1.0", punct_text)
                sentences = split_into_sentences(punct_text)
                if not sentences:
                    messagebox.showinfo("Не удалось выделить предложения", "Не удалось выделить предложения.")
                    return
                plan = self.get_pricing_plan()
                total, per_card_costs, meta = calc_digital_ear_cost(sentences, plan)
                preview_cards = build_preview_cards(sentences)
                self.pending_invoice = {
                    "sentences": sentences,
                    "per_card_costs": per_card_costs,
                    "total": total,
                    "plan": plan,
                    "meta": meta,
                    "preview_cards": preview_cards,
                }
                update_generate_button_state()
                open_cards_preview(preview_cards)
            except Exception as exc:
                logging.exception("Digital ear preview failed")
                messagebox.showerror("Ошибка", f"Не удалось подготовить предпросмотр: {exc}")

        def generate_cards():
            invoice = self.pending_invoice
            if not invoice:
                messagebox.showwarning("Генерация", "Сначала подготовьте предпросмотр.")
                return
            total = int(invoice.get("total") or 0)
            if total > 0 and not self.can_afford(total):
                messagebox.showwarning("Недостаточно кредитов", "Недостаточно кредитов для генерации.")
                return
            try:
                meta = {
                    "plan": invoice.get("plan"),
                    "total": total,
                    "sentences_count": len(invoice.get("sentences") or []),
                    "per_card_costs": invoice.get("per_card_costs"),
                }
                if total > 0 and not self.charge_credits(total, "digital_ear_generate", meta=meta):
                    messagebox.showwarning("Оплата", "Не удалось списать кредиты.")
                    return
                created = 0
                for sentence in invoice.get("sentences") or []:
                    img_path = generate_sd_image_placeholder(sentence, None) or ""
                    front, back = render_generation_faces(
                        self.selected_deck_id,
                        {
                            "translation": "",
                            "front_translation": "",
                            "sentence_with_gap": sentence,
                            "word": sentence,
                            "ipa": "",
                            "gender": "",
                            "plural": "",
                            "sentence": sentence,
                        },
                    )
                    note_fields = {
                        "word": sentence,
                        "translation": "",
                        "example": sentence,
                        "level": 1,
                        "image": img_path,
                        "front_image_path": img_path,
                        "back_image_path": "",
                        "audio_path": None,
                        "front": front,
                        "back": back,
                        "image_generated": False,
                        "image_placeholder": bool(img_path),
                    }
                    _, cards_created = create_note_with_cards(
                        self.selected_deck_id,
                        note_fields,
                        note_type_id=ensure_generated_note_type_id(),
                    )
                    created += cards_created
                self.pending_invoice = None
                update_generate_button_state()
                messagebox.showinfo("Генерация", f"Создано карточек: {created}")
                self.refresh_decks()
                self.update_overdue_badge()
                self.refresh_activation_progress_ui()
            except Exception as exc:
                logging.exception("Digital ear generation failed")
                messagebox.showerror("Ошибка", f"Не удалось сгенерировать карточки: {exc}")

        record_btn.configure(command=start_record)
        preview_btn.configure(command=prepare_preview)
        generate_btn.configure(command=generate_cards)

        self.register_balance_observer(update_generate_button_state)
        update_generate_button_state()

        def on_close():
            self.unregister_balance_observer(update_generate_button_state)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    # --------- режимы повторения / воспроизведения ---------

    def start_review(self):
        self.start_repeat_mode()

    def start_repeat_mode(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        deck_id = self.selected_deck_id
        phase_filter = self.selected_phase

        def task(progress_cb):
            cards = get_cards_for_repeat(deck_id)
            if phase_filter is not None:
                cards = [c for c in cards if c["leitner_level"] == phase_filter]
            total = len(cards)
            for idx in range(0, total, max(1, max(1, total // 20))):
                progress_cb(min(idx + 1, total), total, f"Подготовка {min(idx + 1, total)}/{total}")
            return cards

        def on_success(cards):
            if not cards:
                phase_text = f" (фаза {phase_filter})" if phase_filter is not None else ""
                messagebox.showinfo("Повторение", f"В этой колоде{phase_text} пока нет карточек.")
                return
            repeat_session = ReviewSession(self.selected_deck_id, cards)
            repeat_window = RepeatWindow(self, repeat_session)

            if hasattr(repeat_window, 'btn_frame'):
                ttk.Button(repeat_window.btn_frame, text="Добавить в ознакомление",
                          command=self.add_cards_to_overview_from_repeat).grid(row=0, column=7, padx=5)

        def on_error(exc: Exception):
            messagebox.showerror("Ошибка", str(exc))

        self.run_task("Режим повторения", "determinate", task, on_success, on_error)

    def start_playback_mode(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        deck_id = self.selected_deck_id
        phase_filter = self.selected_phase

        def task(progress_cb):
            cards = get_cards_for_playback(deck_id)
            if phase_filter is not None:
                cards = [c for c in cards if c["leitner_level"] == phase_filter]
            total = len(cards)
            for idx in range(0, total, max(1, max(1, total // 20))):
                progress_cb(min(idx + 1, total), total, f"Подготовка {min(idx + 1, total)}/{total}")
            return cards

        def on_success(cards):
            if not cards:
                phase_text = f" (фаза {phase_filter})" if phase_filter is not None else ""
                messagebox.showinfo("Воспроизведение", f"В этой колоде{phase_text} пока нет карточек.")
                return
            ReviewWindow(self, cards)

        def on_error(exc: Exception):
            messagebox.showerror("Ошибка", str(exc))

        self.run_task("Режим воспроизведения", "determinate", task, on_success, on_error)


class OverviewWindow(tk.Toplevel):
    """Режим ознакомления - показываем обе стороны карточки одновременно"""
    
    def  __init__(self, master, cards):
        super().__init__(master)
        self.master = master
        self.cards = [dict(c) for c in cards]
        
        if not self.cards:
            messagebox.showinfo("Пусто", "Нет карточек для ознакомления.")
            self.destroy()
            return
            
        self.current_index = 0
        self.current_card = self.cards[self.current_index]
        default_lang = get_deck_tts_lang(getattr(self.master, "selected_deck_id", None), "de")
        self.tts_languages = ["de", "ru", "en", "es", "fr", "it"]
        self.front_tts_lang_var = tk.StringVar(value=default_lang)
        self.back_tts_lang_var = tk.StringVar(value=default_lang)
        
        self.title("Режим ознакомления")
        self.geometry("1400x800")
        self.grab_set()
        
        self.create_widgets()
        self.update_view()
    
    def create_widgets(self):
        # Основной контейнер
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Статус и прогресс бар
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_status = ttk.Label(status_frame, text="")
        self.lbl_status.pack(side=tk.LEFT)
        
        # Прогресс бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        palette = getattr(self.master, "palette", None) or {}
        style = ttk.Style(self)
        style.configure(
            "Dark.TCombobox",
            fieldbackground=palette.get("surface", "#111827"),
            background=palette.get("surface", "#111827"),
            foreground=palette.get("text", "#E5E7EB"),
            arrowcolor=palette.get("text", "#E5E7EB"),
        )
        
        # Контейнер для двух карточек
        cards_bg = tk.Frame(main_frame, bg=DARK_BG)
        cards_bg.pack(fill=tk.BOTH, expand=True)
        cards_container = ttk.Frame(cards_bg)
        cards_container.pack(fill=tk.BOTH, expand=True)

        left_wrap = tk.Frame(
            cards_container,
            bg=DARK_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
        )
        left_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_wrap = tk.Frame(
            cards_container,
            bg=DARK_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
        )
        right_wrap.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая карточка - лицевая сторона
        self.left_frame = ttk.LabelFrame(
            left_wrap,
            text="ЛИЦЕВАЯ СТОРОНА",
            width=650,
            height=500,
            style="CardSurface.TLabelframe",
        )
        self.left_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.left_frame.pack_propagate(False)
        
        # Правая карточка - задняя сторона  
        self.right_frame = ttk.LabelFrame(
            right_wrap,
            text="ЗАДНЯЯ СТОРОНА",
            width=650,
            height=500,
            style="CardSurface.TLabelframe",
        )
        self.right_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.right_frame.pack_propagate(False)
        
        # Панель кнопок навигации
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.btn_prev = ttk.Button(btn_frame, text="← Назад", command=self.prev_card)
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        
        self.btn_mark_done = ttk.Button(btn_frame, text="✓ Отметить изученным", command=self.mark_as_learned)
        self.btn_mark_done.pack(side=tk.LEFT, padx=10)
        
        self.btn_repeat = ttk.Button(btn_frame, text="🔁 Повторить (в 1 фазу)", command=self.repeat_card)
        self.btn_repeat.pack(side=tk.LEFT, padx=10)

        self.actions_menu_button = self._build_actions_menu(btn_frame)
        self.actions_menu_button.pack(side=tk.RIGHT, padx=10)

        self.btn_toggle_view = ttk.Button(btn_frame, text="Свернуть карточку", command=self.toggle_view)
        self.btn_toggle_view.pack(side=tk.RIGHT, padx=10)

        self.btn_next = ttk.Button(btn_frame, text="Следующий →", command=self.next_card)
        self.btn_next.pack(side=tk.RIGHT, padx=10)

    def create_card_widgets(self, parent_frame, is_front=True):
        """Создать виджеты для карточки"""
        # Очищаем фрейм
        for widget in parent_frame.winfo_children():
            widget.destroy()

        colors = getattr(self.master, "palette", None)
        card_bg, card_text, _ = get_card_surface_colors(self.master)

        content = ttk.Frame(parent_frame, style="CardSurface.TFrame")
        style_card_surface(content, colors, padded=False)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        header = ttk.Frame(content, style="CardSurface.TFrame")
        style_card_surface(header, colors, padded=False)
        header.pack(fill=tk.X, pady=(0, 6))
        lang_var = self.front_tts_lang_var if is_front else self.back_tts_lang_var
        ttk.Button(
            header,
            text="🔊 Озвучить",
            command=lambda: self.play_side_audio("front" if is_front else "back"),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(header, text="Язык:").pack(side=tk.LEFT, padx=(10, 4))
        lang_combo = ttk.Combobox(
            header,
            values=self.tts_languages,
            textvariable=lang_var,
            width=6,
            state="readonly",
            style="Dark.TCombobox",
        )
        lang_combo.pack(side=tk.LEFT, padx=4)

        # Текст с прокруткой
        text_frame = ttk.Frame(content, style="CardSurface.TFrame")
        style_card_surface(text_frame, colors, padded=False)
        text_frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Arial", 12),
            bg=card_bg,
            fg=card_text,
            height=10
        )
        style_card_surface_text(text_widget, colors)
        text_widget.config(state='normal')
        scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview, style="Vertical.TScrollbar")
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Изображение
        image_label = ResizableImageLabel(
            content,
            bg=card_bg,
            relief="flat",
            bd=0
        )
        style_card_surface(image_label, colors)
        image_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Аудио блок
        audio_frame = ttk.Frame(content, style="CardSurface.TFrame")
        style_card_surface(audio_frame, colors, padded=False)
        audio_frame.pack(fill=tk.X, pady=(10, 0))
        audio_frame.audio_widget = AudioPlayerWidget(audio_frame, on_error_callback=self._show_audio_error)
        audio_frame.audio_widget.pack(fill=tk.X)
        audio_frame.audio_widget.pack_forget()

        video_frame = ttk.Frame(content, style="CardSurface.TFrame")
        style_card_surface(video_frame, colors, padded=False)
        video_frame.pack(fill=tk.X, pady=(5, 0))

        return text_widget, image_label, audio_frame, video_frame
    
    def mark_as_learned(self):
        """Отметить карточку как изученную"""
        card_id = self.current_card["id"]
        update_card_progress(card_id, 100)
        self.current_card["progress"] = 100
        
        # Переместить на более высокий уровень
        current_level = self.current_card["leitner_level"]
        if current_level < 10:
            new_level = min(10, current_level + 2)
            update_card_leitner(card_id, new_level)
            self.current_card["leitner_level"] = new_level
        
        messagebox.showinfo("Успех", "Карточка отмечена как изученная и переведена на более высокий уровень.")
        self.next_card()
    
    def repeat_card(self):
        """Отправить карточку на повторение в 1 фазу"""
        card_id = self.current_card["id"]
        # Обновляем уровень карточки на 1 (первая фаза)
        update_card_leitner(card_id, 1)
        
        # Обновляем текущий объект карточки
        self.current_card["leitner_level"] = 1
        
        # Обновляем статистику
        if hasattr(self.master, 'selected_deck_id') and self.master.selected_deck_id:
            update_statistics(self.master.selected_deck_id, remembered=False, forgotten=True, reviewed=True)
        
        messagebox.showinfo("Повторение", "Карточка отправлена на повторение в 1 фазу.")
        
        # Перелистываем вперед
        self.next_card()
    
    def update_view(self):
        """Обновить отображение обеих сторон карточки"""
        total = len(self.cards)
        idx = self.current_index + 1
        c = self.current_card

        if getattr(self.master, "mw_context", None) is not None:
            self.master.mw_context.state["current_card_id"] = c.get("id")
        gui_hooks.card_will_show(c)

        # Обновляем статус
        self.lbl_status.config(text=f"Карточка {idx}/{total} | ID {c['id']} | Уровень: {c['leitner_level']} | Прогресс: {c['progress']}%")
        
        # Обновляем прогресс бар
        self.progress_var.set((idx / total) * 100)
        
        # Создаем виджеты для лицевой стороны
        self.front_text, self.front_image_label, self.front_audio_frame, self.front_video_frame = self.create_card_widgets(self.left_frame, is_front=True)
        
        # Обновляем лицевую сторону (FRONT)
        front_content = c["front"]
        self.front_text.insert(1.0, front_content)
        self.front_text.configure(state='disabled')
        create_context_menu(self.front_text)  # Добавляем контекстное меню
        
        # Загружаем изображение лицевой стороны
        front_img_path = resolve_media_path(c.get("front_image_path") or c.get("image_path"))
        if front_img_path:
            self.front_image_label.load_image(front_img_path)
        else:
            self.front_image_label.load_image(None)
        
        # Обновляем аудио плеер для лицевой стороны
        self.update_audio_player(self.front_audio_frame, c, prefer_side="front")
        self.update_video_player(self.front_video_frame, c)
        
        # Создаем виджеты для задней стороны
        self.back_text, self.back_image_label, self.back_audio_frame, self.back_video_frame = self.create_card_widgets(self.right_frame, is_front=False)
        
        # Обновляем заднюю сторону (BACK)
        back_content = c["back"]
        
        # ВСЕГДА добавляем перевод для режима ознакомления
        if TRANSLATION_SETTINGS.show_back_translation:
            lines = front_content.split('\n')
            if lines:
                sentence = lines[0].strip()
                if sentence and len(sentence.split()) > 1:
                    translation = translate_sentence(sentence, use_openai=True)
                    if translation and translation != sentence:
                        back_content = f"{back_content}\n\n🇷🇺 Перевод: {translation}"
        
        self.back_text.insert(1.0, back_content)
        self.back_text.configure(state='disabled')
        create_context_menu(self.back_text)  # Добавляем контекстное меню
        
        # Загружаем изображение задней стороны
        back_img_path = resolve_media_path(c.get("back_image_path") or c.get("image_path"))
        if back_img_path:
            self.back_image_label.load_image(back_img_path)
        else:
            self.back_image_label.load_image(None)
        
        # Обновляем аудио плеер для задней стороны
        self.update_audio_player(self.back_audio_frame, c, prefer_side="back")
        self.update_video_player(self.back_video_frame, c)
    
    def update_audio_player(self, audio_frame, card, prefer_side: str = "back"):
        """Обновить аудио плеер"""
        entries = get_card_audio_entries(card, prefer_side=prefer_side)
        if not hasattr(audio_frame, "audio_widget"):
            audio_frame.audio_widget = AudioPlayerWidget(audio_frame, on_error_callback=self._show_audio_error)
            audio_frame.audio_widget.pack(fill=tk.X)
        display_audio_entries_on_frame(audio_frame, entries)

    def _show_audio_error(self, title: str, message: str):
        try:
            messagebox.showerror(title, message)
        except Exception:
            pass

    def update_video_player(self, video_frame, card):
        """Показать плеер или кнопку открытия видео клипа."""
        for widget in video_frame.winfo_children():
            widget.destroy()

        video_path = find_video_media_path(card)
        if not video_path:
            ttk.Label(video_frame, text="Видео не прикреплено").pack(side=tk.LEFT, padx=(0, 5))
            return

        ttk.Label(video_frame, text="Видео:").pack(side=tk.LEFT, padx=(0, 5))

        if is_vlc_available():
            try:
                player = VlcPlayerWidget(video_frame, video_path, width=320, height=200)
                if not player.ensure_embedded():
                    player.frame.destroy()
                else:
                    player.pack(side=tk.LEFT, padx=(0, 10))
                    video_frame.vlc_player = player  # сохраняем ссылку, чтобы VLC не выгружался
                    ttk.Button(video_frame, text="⏹ Стоп", command=player.stop).pack(side=tk.LEFT)
                    return
            except Exception as exc:
                print(f"[VLC] Ошибка embed видео: {exc}")

        ttk.Button(
            video_frame,
            text="Открыть во внешнем плеере",
            command=lambda: open_in_external_player(video_path)
        ).pack(side=tk.LEFT)

    def _build_actions_menu(self, parent) -> ttk.Menubutton:
        menu_button, menu = create_action_menubutton(parent, getattr(self.master, "palette", None))

        def _placeholder(action: str) -> None:
            messagebox.showinfo(action, "Будет реализовано")

        def _card_info() -> None:
            if not self.current_card:
                return
            messagebox.showinfo(
                "Сведения о карточке",
                f"ID: {self.current_card.get('id')}\nКолода: {self.current_card.get('deck_id')}",
            )

        def _reset_card() -> None:
            if not self.current_card:
                return
            update_card_progress(self.current_card["id"], 0)
            update_card_leitner(self.current_card["id"], 1)
            self.update_view()

        def _delete_card() -> None:
            if not self.current_card:
                return
            card_id = self.current_card["id"]
            delete_card(card_id)
            self.cards = [c for c in self.cards if c["id"] != card_id]
            if not self.cards:
                messagebox.showinfo("Готово", "Карточки закончились.")
                self.destroy()
                return
            self.current_index = min(self.current_index, max(0, len(self.cards) - 1))
            self.current_card = self.cards[self.current_index]
            self.update_view()

        menu.add_command(
            label="Отметить карточку",
            command=lambda: mark_card_for_overview(self.current_card["id"]) if self.current_card else None,
        )
        menu.add_command(label="Отложить карточку", command=lambda: _placeholder("Отложить карточку"))
        menu.add_command(label="Сбросить карточку", command=_reset_card)
        menu.add_command(label="Задать срок", command=lambda: _placeholder("Задать срок"))
        menu.add_command(label="Исключить карточку", command=lambda: _placeholder("Исключить карточку"))
        menu.add_command(label="Сведения о карточке", command=_card_info)
        menu.add_command(label="Удалить карточку", command=_delete_card)

        return menu_button

    def play_side_audio(self, side: str) -> None:
        if not self.current_card:
            return
        text = ensure_plain_text(self.current_card.get("front" if side == "front" else "back") or "")
        if not text.strip():
            messagebox.showinfo("Озвучка", "Нет текста для озвучивания.")
            return
        lang_var = self.front_tts_lang_var if side == "front" else self.back_tts_lang_var
        lang = (lang_var.get() or "").strip() or get_deck_tts_lang(getattr(self.master, "selected_deck_id", None), "de")
        speak_google_tts(text, lang)

    def play_audio_file(self, path):
        """Воспроизвести аудио файл"""
        if WINSOUND_AVAILABLE and os.path.exists(path):
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                messagebox.showerror("Ошибка", "Не удалось воспроизвести аудио")
        elif TTS_AVAILABLE:
            speak_text(ensure_plain_text(self.current_card.get("front") or ""))
        else:
            messagebox.showinfo("Ошибка", "Аудио система недоступна")
    
    def play_audio(self):
        """Озвучить текущую карточку"""
        for widget in [
            getattr(self.front_audio_frame, "audio_widget", None),
            getattr(self.back_audio_frame, "audio_widget", None),
        ]:
            if widget and widget.is_loaded():
                widget.play()
                return

        audio_path = get_card_audio_path(self.current_card, prefer_side="back")
        if audio_path and os.path.exists(audio_path) and WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass
        
        # Если нет аудио файла, озвучиваем текст
        front_text = ensure_plain_text(self.current_card.get("front") or "")
        if front_text:
            speak_text(front_text)
    
    def toggle_view(self):
        """Переключить между развернутым и свернутым видом"""
        if not hasattr(self, 'is_minimized') or not self.is_minimized:
            self.is_minimized = True
            self.geometry("800x600")
            self.btn_toggle_view.config(text="Развернуть карточку")
        else:
            self.is_minimized = False
            self.geometry("1400x800")
            self.btn_toggle_view.config(text="Свернуть карточку")
    
    def next_card(self):
        """Перейти к следующей карточке"""
        # Обновляем статистику ознакомления (+1)
        if hasattr(self.master, 'selected_deck_id') and self.master.selected_deck_id:
            update_overview_statistics(self.master.selected_deck_id, increment=1)
        
        # Увеличиваем прогресс текущей карточки
        if self.current_card["progress"] < 100:
            new_progress = min(100, self.current_card["progress"] + 10)
            update_card_progress(self.current_card["id"], new_progress)
            self.current_card["progress"] = new_progress
        
        # Переходим к следующей карточке
        self.current_index += 1
        if self.current_index >= len(self.cards):
            messagebox.showinfo("Готово", "Вы ознакомились со всеми карточками в этой колоде.")
            self.destroy()
            return
        
        self.current_card = self.cards[self.current_index]
        self.update_view()
    
    def prev_card(self):
        """Перейти к предыдущей карточке"""
        if self.current_index > 0:
            # Обновляем статистику ознакомления (-1)
            if hasattr(self.master, 'selected_deck_id') and self.master.selected_deck_id:
                update_overview_statistics(self.master.selected_deck_id, increment=-1)
            
            self.current_index -= 1
            self.current_card = self.cards[self.current_index]
            self.update_view()


class ReviewSession:
    def __init__(self, deck_id: int, cards: list[dict]):
        self.deck_id = deck_id
        self.cards = [dict(card) for card in cards]
        self.index = 0

    @classmethod
    def load_due_cards(cls, deck_id: int, phase_filter: int | None = None) -> "ReviewSession":
        cards = get_cards_for_repeat(deck_id)
        if phase_filter is not None:
            cards = [card for card in cards if card["leitner_level"] == phase_filter]
        return cls(deck_id, cards)

    def current_card(self) -> dict | None:
        if 0 <= self.index < len(self.cards):
            return self.cards[self.index]
        return None

    def next_card(self) -> dict | None:
        self.index += 1
        return self.current_card()

    def prev_card(self) -> dict | None:
        if self.index > 0:
            self.index -= 1
        return self.current_card()

    def mark_wrong(self) -> dict | None:
        return self._apply_rating(0, remembered=False)

    def mark_correct(self) -> dict | None:
        return self._apply_rating(2, remembered=True)

    def _apply_rating(self, rating: int, remembered: bool) -> dict | None:
        card = self.current_card()
        if not card:
            return None
        result = apply_srs_update(card["id"], rating)
        update_statistics(self.deck_id, remembered=remembered, forgotten=not remembered, reviewed=True)
        if result:
            card["leitner_level"] = result.get("phase")
            card["next_review"] = datetime.fromtimestamp(result.get("due", time.time())).isoformat()
            card["state"] = result.get("state")
        return result


class RepeatWindow(tk.Toplevel):
    def __init__(self, master, session: ReviewSession):
        super().__init__(master)
        self.master = master
        self.session = session
        self.current_card = self.session.current_card()
        self.show_back = False
        self.current_photo = None
        self.card_widget: CardWidget | None = None
        
        # Состояние переводов
        self.show_translations = TRANSLATION_SETTINGS.show_translations
        self.translations_visible = {}
        
        # Для 6-клеточного чекпоинта
        self.checkpoint_vars = []
        self.checkpoint_states = {}

        # Таймер повторения
        self.timer_left = 0
        self.timer_job = None
        self.timer_flash_job = None
        self.timer_label = None
        self._timer_fired = False
        
        self.title("Режим повторения")
        self.geometry("1000x700")
        self.grab_set()

        self.create_widgets()
        self.update_view()
        if self.current_card:
            self.load_checkpoint_state()
            self.reset_timer_for_card()

    def create_widgets(self):
        frame_main = ttk.Frame(self, style="Surface.TFrame")
        frame_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        colors = getattr(self.master, "palette", None)
        card_bg, card_text, _ = get_card_surface_colors(self.master)

        # Статус
        self.lbl_status = ttk.Label(frame_main, text="")
        self.lbl_status.pack(anchor="w")

        # Таймер
        self.timer_label = tk.Label(
            frame_main,
            text="⏰ 00:00",
            bg=colors["background"] if colors else self.cget("bg"),
            fg=colors["error"] if colors else "#FF4D4D",
            font=("Segoe UI", 11, "bold"),
        )
        self.timer_label.pack(anchor="center", pady=(3, 5))

        # Основной фрейм карточки
        cards_bg = tk.Frame(frame_main, bg=DARK_BG)
        cards_bg.pack(fill=tk.BOTH, expand=True, pady=10)
        card_wrap = tk.Frame(
            cards_bg,
            bg=DARK_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
            width=CARD_VIEW_WIDTH,
            height=CARD_VIEW_HEIGHT,
        )
        card_wrap.pack(padx=10, pady=10)
        card_wrap.pack_propagate(False)
        self.card_frame = card_wrap
        self.card_renderer = CardRenderer(
            self.card_frame,
            palette=colors,
            editable=False,
            width=CARD_VIEW_WIDTH,
            height=CARD_VIEW_HEIGHT,
            show_image_toolbar=False,
            image_layout="side",
            on_media_state_change=self._handle_media_state_update,
            enable_state_restore=True,
            fixed_media_slot=REPEAT_MEDIA_SLOT_SIZE,
            render_mode="repeat",
        )

        # Фрейм для 6-клеточного чекпоинта (внизу карточки)
        self.checkpoint_frame = tk.Frame(cards_bg, bg=card_bg)
        self.checkpoint_frame.pack(pady=(0, 8))
        
        # Создаем 6 чекбоксов в ряд
        self.checkpoint_vars = []
        for i in range(6):
            var = tk.BooleanVar(value=False)
            self.checkpoint_vars.append(var)
            cb = tk.Checkbutton(
                self.checkpoint_frame,
                text=f"✓{i+1}",
                variable=var,
                bg=card_bg,
                fg=card_text,
                command=lambda idx=i: self.update_checkpoint_state(idx)
            )
            cb.pack(side=tk.LEFT, padx=5)

        self.audio_inline_frame = self.card_renderer.audio_frame
        self.video_inline_frame = self.card_renderer.video_frame

        # Нижний бар управления (всегда видим)
        self.controls_bar = ttk.Frame(self, style="Surface.TFrame")
        self.controls_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

        # Кнопка перевода (для лицевой стороны)
        self.btn_translation = ttk.Button(
            self.controls_bar, 
            text="Скрыть перевод слов" if self.show_translations else "Показать перевод слов",
            command=self.toggle_translations
        )
        self.btn_translation.pack(side=tk.LEFT, padx=5)

        # Кнопки навигации
        self.btn_prev = ttk.Button(self.controls_bar, text="← Назад", command=self.prev_card)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = ttk.Button(self.controls_bar, text="Вперед →", command=self.next_card)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.btn_show = ttk.Button(self.controls_bar, text="Показать ответ", command=self.toggle_front_back)
        self.btn_show.pack(side=tk.LEFT, padx=5)

        # Кнопки фаз
        self.btn_forget = ttk.Button(self.controls_bar, text="Забыл (Фаза 1)", command=self.mark_forgotten)
        self.btn_forget.pack(side=tk.LEFT, padx=5)

        self.btn_remember = ttk.Button(self.controls_bar, text="Повторить (Фаза 2)", command=self.mark_remembered)
        self.btn_remember.pack(side=tk.LEFT, padx=5)

        self.actions_menu_button = self._build_actions_menu(self.controls_bar)
        self.actions_menu_button.pack(side=tk.RIGHT, padx=5)

        # Кнопка звука
        self.btn_sound = ttk.Button(self.controls_bar, text="🔊 Слово", command=self.play_word)
        self.btn_sound.pack(side=tk.RIGHT, padx=5)

        # Инициализация аудио-плеера
        self.update_audio_player()

    def update_audio_player(self):
        """Обновить аудио-плеер для текущей карточки"""
        if not self.card_renderer:
            return
        prefer_side = "back" if self.show_back else "front"
        self.card_renderer.update_media(self.current_card, prefer_audio_side=prefer_side)
        self.audio_widget = self.card_renderer.get_audio_widget()

    def _build_actions_menu(self, parent) -> ttk.Menubutton:
        menu_button, menu = create_action_menubutton(parent, getattr(self.master, "palette", None))

        def _placeholder(action: str) -> None:
            messagebox.showinfo(action, "Будет реализовано")

        def _card_info() -> None:
            if not self.current_card:
                return
            messagebox.showinfo(
                "Сведения о карточке",
                f"ID: {self.current_card.get('id')}\nКолода: {self.current_card.get('deck_id')}",
            )

        def _reset_card() -> None:
            if not self.current_card:
                return
            update_card_progress(self.current_card["id"], 0)
            update_card_leitner(self.current_card["id"], 1)
            self.update_view()

        def _delete_card() -> None:
            if not self.current_card:
                return
            card_id = self.current_card["id"]
            delete_card(card_id)
            self.session.cards = [c for c in self.session.cards if c["id"] != card_id]
            self.session.index = min(self.session.index, max(0, len(self.session.cards) - 1))
            self.current_card = self.session.current_card()
            if not self.current_card:
                self.show_end_state()
            else:
                self.update_view()

        menu.add_command(
            label="Отметить карточку",
            command=lambda: mark_card_for_overview(self.current_card["id"]) if self.current_card else None,
        )
        menu.add_command(label="Отложить карточку", command=lambda: _placeholder("Отложить карточку"))
        menu.add_command(label="Сбросить карточку", command=_reset_card)
        menu.add_command(label="Задать срок", command=lambda: _placeholder("Задать срок"))
        menu.add_command(label="Исключить карточку", command=lambda: _placeholder("Исключить карточку"))
        menu.add_command(label="Сведения о карточке", command=_card_info)
        menu.add_command(label="Удалить карточку", command=_delete_card)

        return menu_button

    def _apply_audio_state_from_selection(self):
        audio_widget = getattr(self.audio_inline_frame, "audio_widget", None)
        entry_map = getattr(self.audio_inline_frame, "audio_entry_map", {}) or {}
        selection = getattr(self.audio_inline_frame, "audio_selector_var", None)
        if not audio_widget or not selection:
            return
        selected_label = selection.get()
        entry = entry_map.get(selected_label)
        if not entry:
            return
        media_key = _build_media_key(entry.get("media_id"), entry.get("path"))
        audio_widget.set_media_key(media_key)
        audio_widget.apply_state(load_media_state(self.current_card["id"], media_key))

    def update_video_player(self):
        for widget in self.video_inline_frame.winfo_children():
            widget.destroy()
        video_path = find_video_media_path(self.current_card)
        if not video_path:
            ttk.Label(self.video_inline_frame, text="Видео не прикреплено").pack(anchor="w", padx=5, pady=5)
            return
        if is_vlc_available():
            try:
                player = VlcPlayerWidget(
                    self.video_inline_frame,
                    video_path,
                    width=420,
                    height=200,
                    on_state_change=self._handle_media_state_update,
                )
                player.pack(anchor="w")
                media_entries = get_media_for_card(self.current_card.get("id"), self.current_card.get("note_id"))
                media_id = None
                for entry in media_entries:
                    media_type = (entry.get("media_type") or entry.get("type") or "").lower()
                    if media_type == "video" and entry.get("path") == video_path:
                        media_id = entry.get("id")
                        break
                media_key = _build_media_key(media_id, video_path)
                player.set_media_key(media_key)
                player.apply_state(load_media_state(self.current_card["id"], media_key))
                self.video_inline_frame.vlc_player = player
                return
            except Exception:
                pass
        ttk.Button(
            self.video_inline_frame,
            text="Открыть во внешнем плеере",
            command=lambda: open_in_external_player(video_path),
        ).pack(anchor="w", padx=5, pady=5)

    def _handle_media_state_update(self, media_key: str | None, state: dict):
        if not media_key:
            return
        try:
            save_media_state(
                self.current_card["id"],
                media_key,
                state.get("pos_ms", 0),
                state.get("volume", 70),
                state.get("speed", 1),
            )
        except Exception:
            pass

    def save_current_media_state(self):
        audio_widget = getattr(self.audio_inline_frame, "audio_widget", None)
        if audio_widget and audio_widget.media_key:
            state = audio_widget.get_state()
            save_media_state(
                self.current_card["id"],
                audio_widget.media_key,
                state.get("pos_ms", 0),
                state.get("volume", 70),
                state.get("speed", 1),
            )
        video_player = getattr(self.video_inline_frame, "vlc_player", None)
        if video_player and video_player.media_key:
            state = video_player.get_state()
            save_media_state(
                self.current_card["id"],
                video_player.media_key,
                state.get("pos_ms", 0),
                state.get("volume", 70),
                state.get("speed", 1),
            )

    def _show_audio_error(self, title: str, message: str):
        try:
            messagebox.showerror(title, message)
        except Exception:
            pass

    def get_audio_entries(self) -> list[dict]:
        entries = []
        media = get_media_for_card(self.current_card.get("id"), self.current_card.get("note_id"))
        for item in media:
            media_type = (item.get("media_type") or item.get("type") or "").lower()
            if media_type != "audio":
                continue
            entries.append(
                {
                    "path": item.get("path"),
                    "side": (item.get("side") or "back").lower(),
                    "source": item.get("source"),
                }
            )

        fallback_path = self.current_card.get("audio_path")
        if fallback_path:
            entries.append({"path": fallback_path, "side": "back", "source": None})
        return entries
    
    def play_audio_file(self, path):
        """Воспроизвести аудио файл"""
        if WINSOUND_AVAILABLE and os.path.exists(path):
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                messagebox.showerror("Ошибка", "Не удалось воспроизвести аудио")
        elif TTS_AVAILABLE:
            speak_text(self.current_card["front"])
        else:
            messagebox.showinfo("Ошибка", "Аудио система недоступна")

    def load_checkpoint_state(self):
        """Загрузить состояние чекпоинтов для текущей карточки."""
        if not self.current_card:
            return
        card_id = self.current_card["id"]
        if card_id not in self.checkpoint_states:
            self.checkpoint_states[card_id] = [False] * 6
        else:
            for i, state in enumerate(self.checkpoint_states[card_id]):
                self.checkpoint_vars[i].set(state)

    def update_checkpoint_state(self, idx):
        """Обновить состояние чекпоинта."""
        card_id = self.current_card["id"]
        if card_id not in self.checkpoint_states:
            self.checkpoint_states[card_id] = [False] * 6
        self.checkpoint_states[card_id][idx] = self.checkpoint_vars[idx].get()

    def toggle_translations(self):
        """Переключить отображение переводов слов на лицевой стороне."""
        self.show_translations = not self.show_translations
        card_id = self.current_card["id"]
        self.translations_visible[card_id] = self.show_translations
        self.btn_translation.config(
            text="Скрыть перевод слов" if self.show_translations else "Показать перевод слов"
        )
        self.update_view()

    def extract_words_with_translations(self, text):
        """Извлечь слова из текста и добавить переводы из словаря."""
        text = ensure_plain_text(text)
        # Удаляем перевод в скобках если он есть
        text = re.sub(r'\([^)]*\)', '', text).strip()
        
        words = re.findall(r'\b\w+\b', text, re.UNICODE)
        result = []

        for word in words:
            if len(word) < 2:  # Пропускаем очень короткие слова
                result.append(word)
                continue

            translation = get_translation(word, use_openai=False) if self.show_translations else ""
            if translation:
                # Создаем фрейм для слова и перевода
                colors = getattr(self.master, "palette", None)
                card_bg, card_text, _ = get_card_surface_colors(self.master)
                parent = self.card_renderer.custom_text_frame if self.card_renderer else self
                word_frame = tk.Frame(parent, bg=card_bg)

                # Слово
                word_label = tk.Label(
                    word_frame,
                    text=word,
                    bg=card_bg,
                    fg=card_text,
                    font=("Segoe UI", 12)
                )
                word_label.pack(side=tk.LEFT, padx=(0, 5))

                # Перевод
                if self.show_translations:
                    trans_label = tk.Label(
                        word_frame,
                        text=f"({translation})",
                        bg=card_bg,
                        fg=colors["accent"] if colors else "blue",
                        font=("Segoe UI", 10, "italic")
                    )
                    trans_label.pack(side=tk.LEFT)

                result.append(word_frame)
            else:
                result.append(word)
        
        return result

    def update_view(self):
        total = len(self.session.cards)
        idx = self.session.index + 1
        c = self.current_card
        colors = getattr(self.master, "palette", None)
        card_bg, card_text, _ = get_card_surface_colors(self.master)

        if not c:
            self.show_end_state()
            return

        if getattr(self.master, "mw_context", None) is not None:
            self.master.mw_context.state["current_card_id"] = c.get("id")
        gui_hooks.card_will_show(c)

        self.lbl_status.config(
            text=f"Карточка {idx}/{total} | ID {c['id']}"
        )

        # Обновляем текст кнопок
        romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        lvl = c["leitner_level"]
        phase = romans[min(max(lvl, 1), 10) - 1]
        header_text = f"Фаза {phase} | след. повтор: {c['next_review']}"
        if self.card_renderer is not None:
            self.card_renderer.set_header_text(header_text)

        front_text = c["front"]
        back_text = c["back"]
        if self.show_back:
            img_path = c["back_image_path"] or c["front_image_path"] or c["image_path"]
        else:
            img_path = c["front_image_path"] or c["image_path"]

        use_custom = (not self.show_back) and self.show_translations and c["leitner_level"] == 1
        custom_items = self.extract_words_with_translations(front_text) if use_custom else None
        if self.card_renderer is not None:
            self.card_renderer.update_text(
                c,
                show_back=self.show_back,
                image_override=img_path,
                custom_items=custom_items,
            )

        self.btn_show.config(text="Показать ответ" if not self.show_back else "Показать лицевую сторону")

        # Обновляем аудио/видео
        self.update_audio_player()

    def show_end_state(self):
        self.stop_timer()
        self.timer_left = 0
        self.update_timer_label()
        self.lbl_status.config(text="На сегодня всё")
        if self.card_renderer is not None:
            self.card_renderer.set_header_text("")
            self.card_renderer.update_text({"front": "На сегодня всё", "back": ""}, show_back=False)
        for btn in [
            self.btn_prev,
            self.btn_next,
            self.btn_show,
            self.btn_forget,
            self.btn_remember,
            self.btn_sound,
            self.btn_translation,
        ]:
            btn.config(state=tk.DISABLED)

    def toggle_front_back(self):
        self.show_back = not self.show_back
        self.update_view()

    def reset_timer_for_card(self):
        self.stop_timer()
        self._timer_fired = False
        review_seconds = get_effective_mode_timer(
            getattr(self.master, "selected_deck_id", None), "review"
        )
        self.timer_left = max(0, int(review_seconds or 0))
        self.update_timer_label()
        self.start_timer()

    def start_timer(self):
        if self.timer_left > 0:
            self.timer_job = self.after(1000, self.timer_tick)

    def stop_timer(self):
        if self.timer_job is not None:
            try:
                self.after_cancel(self.timer_job)
            except Exception:
                pass
            self.timer_job = None
        if self.timer_flash_job is not None:
            try:
                self.after_cancel(self.timer_flash_job)
            except Exception:
                pass
            self.timer_flash_job = None

    def timer_tick(self):
        if self.timer_left <= 0:
            self.update_timer_label()
            return
        self.timer_left -= 1
        self.update_timer_label()
        if self.timer_left <= 0:
            self.handle_timer_notify()
            if not self._timer_fired:
                self._timer_fired = True
                self.goto_next_card()
            return
        self.timer_job = self.after(1000, self.timer_tick)

    def update_timer_label(self, seconds: int | None = None):
        if self.timer_label is None:
            return
        if seconds is None:
            seconds = max(0, int(self.timer_left))
        m, s = divmod(max(0, int(seconds)), 60)
        self.timer_label.config(text=f"⏰ {m:02d}:{s:02d}")

    def handle_timer_notify(self):
        if self.timer_label is None:
            return
        original_bg = self.timer_label.cget("bg")
        self.timer_label.config(bg="#FFD966")

        def reset_bg():
            try:
                self.timer_label.config(bg=original_bg)
            except Exception:
                pass

        self.timer_flash_job = self.after(1500, reset_bg)

    def mark_forgotten(self):
        self.session.mark_wrong()
        self.master.update_overdue_badge()
        if self.current_card:
            gui_hooks.card_did_answer(self.current_card, 0)
        self.goto_next_card()

    def mark_remembered(self):
        self.session.mark_correct()
        self.master.update_overdue_badge()
        if self.current_card:
            gui_hooks.card_did_answer(self.current_card, 2)
        self.goto_next_card()

    def goto_next_card(self):
        self.save_current_media_state()
        self.current_card = self.session.next_card()
        self.show_back = False
        if not self.current_card:
            self.show_end_state()
            return
        self.load_checkpoint_state()
        self.update_view()
        self.reset_timer_for_card()

    def next_card(self):
        self.goto_next_card()

    def prev_card(self):
        self.save_current_media_state()
        self.current_card = self.session.prev_card()
        self.show_back = False
        self.load_checkpoint_state()
        self.update_view()
        self.reset_timer_for_card()

    def play_word(self):
        selection = get_selected_text_from_widget(self.focus_get())
        back_text = ensure_plain_text(self.current_card.get("back") or "")
        text = selection or back_text
        if not text.strip():
            messagebox.showinfo("Озвучка", "Нет текста для озвучивания.")
            return
        deck_id = self.current_card.get("deck_id") or getattr(self.master, "selected_deck_id", None)
        lang = get_deck_tts_lang(deck_id, "de")
        speak_google_tts(text, lang)


class ReviewWindow(tk.Toplevel):
    def __init__(self, master, cards):
        super().__init__(master)
        self.master = master
        self.cards = [dict(c) for c in cards]
        self.current_index = 0
        self.current_card = self.cards[self.current_index]
        self.show_back = False
        self.current_photo = None

        # Таймеры
        self.auto_flip_id = None
        self.auto_next_id = None
        self.timer_left = 0
        self.timer_job = None
        self.timer_label = None
        self.timer_flash_job = None

        # Прогресс
        self.progress_canvas = None
        self.progress_label = None

        self.title("Режим воспроизведения (Лейтнер)")
        self.geometry("900x600")
        self.grab_set()

        self.create_widgets()
        self.update_view()
        self.schedule_timers_for_card()

    def create_widgets(self):
        frame_main = ttk.Frame(self, style="Surface.TFrame")
        frame_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        colors = getattr(self.master, "palette", None)
        card_bg, card_text, _ = get_card_surface_colors(self.master)

        self.lbl_status = ttk.Label(frame_main, text="")
        self.lbl_status.pack(anchor="w")

        # Таймер
        self.timer_label = tk.Label(
            frame_main,
            text="⏰ 00:00",
            bg=colors["background"] if colors else self.cget("bg"),
            fg=colors["error"] if colors else "#FF4D4D",
            font=("Segoe UI", 11, "bold")
        )
        self.timer_label.pack(anchor="center", pady=(3, 5))

        # Фрейм карточки
        self.card_frame = tk.Frame(
            frame_main,
            bg=card_bg,
            bd=0,
            relief="flat",
            width=CARD_VIEW_WIDTH,
            height=CARD_VIEW_HEIGHT
        )
        style_card_surface(self.card_frame, colors)
        self.card_frame.pack(pady=10)
        self.card_frame.pack_propagate(False)

        # Индикатор загрузки
        self.dot_canvas = tk.Canvas(self.card_frame, width=20, height=20,
                                    bg=card_bg, highlightthickness=0, borderwidth=0)
        self.dot_canvas.place(relx=0.5, rely=0.5, anchor="center")
        self.dot_canvas.create_oval(7, 7, 13, 13, fill="red", outline="red")

        content_container = tk.Frame(self.card_frame, bg=card_bg)
        content_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(20, 0))
        self.card_renderer = CardRenderer(
            content_container,
            palette=colors,
            editable=False,
            width=CARD_VIEW_WIDTH,
            height=CARD_VIEW_HEIGHT,
            show_image_toolbar=False,
            image_layout="side",
            fixed_media_slot=REPEAT_MEDIA_SLOT_SIZE,
            render_mode="playback",
        )

        # Прогресс-бар
        progress_frame = tk.Frame(self.card_frame, bg=card_bg)
        progress_frame.pack(side=tk.BOTTOM, pady=(0, 4))

        self.progress_canvas = tk.Canvas(
            progress_frame, width=260, height=14,
            bg=card_bg, highlightthickness=1, highlightbackground=colors["card_border"] if colors else "#cccccc"
        )
        self.progress_canvas.pack(side=tk.LEFT, padx=(10, 4))

        self.progress_label = tk.Label(
            progress_frame,
            text="0 / 100",
            bg=card_bg,
            fg=card_text,
            font=("Segoe UI", 9),
        )
        self.progress_label.pack(side=tk.LEFT, padx=4)

        self.btn_progress_plus = ttk.Button(
            progress_frame, text="+", width=3,
            command=self.increment_progress
        )
        self.btn_progress_plus.pack(side=tk.LEFT, padx=(4, 10))

        # Панель кнопок
        bottom_frame = tk.Frame(self.card_frame, bg=card_bg)
        bottom_frame.pack(side=tk.BOTTOM, pady=8)

        self.btn_audio_icon = ttk.Button(bottom_frame, text="🔊", width=3, command=self.play_word)
        self.btn_audio_icon.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(frame_main)
        btn_frame.pack(pady=10)

        self.btn_show = ttk.Button(btn_frame, text="Показать ответ", command=self.toggle_front_back)
        self.btn_show.grid(row=0, column=0, padx=5)

        self.btn_prev = ttk.Button(btn_frame, text="← Назад", command=self.goto_prev_card)
        self.btn_prev.grid(row=0, column=1, padx=5)

        self.btn_next = ttk.Button(btn_frame, text="Следующая →", command=self.goto_next_card)
        self.btn_next.grid(row=0, column=2, padx=5)


        self.btn_sound = ttk.Button(btn_frame, text="🔊 Слово", command=self.play_word)
        self.btn_sound.grid(row=0, column=3, padx=5)

        self.update_audio_player()

    def update_audio_player(self):
        """Обновить аудио-плеер для текущей карточки"""
        if not hasattr(self, "card_renderer") or self.card_renderer is None:
            return
        prefer_side = "back" if self.show_back else "front"
        self.card_renderer.update_media(self.current_card, prefer_audio_side=prefer_side)
        self.audio_widget = self.card_renderer.get_audio_widget()

    def _show_audio_error(self, title: str, message: str):
        try:
            messagebox.showerror(title, message)
        except Exception:
            pass
    
    def play_audio_file(self, path):
        """Воспроизвести аудио файл"""
        if WINSOUND_AVAILABLE and os.path.exists(path):
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                messagebox.showerror("Ошибка", "Не удалось воспроизвести аудио")
        elif TTS_AVAILABLE:
            speak_text(self.current_card["front"])
        else:
            messagebox.showinfo("Ошибка", "Аудио система недоступна")

    def cancel_timers(self):
        if self.auto_flip_id is not None:
            try:
                self.after_cancel(self.auto_flip_id)
            except Exception:
                pass
            self.auto_flip_id = None

        if self.auto_next_id is not None:
            try:
                self.after_cancel(self.auto_next_id)
            except Exception:
                pass
            self.auto_next_id = None

        if self.timer_job is not None:
            try:
                self.after_cancel(self.timer_job)
            except Exception:
                pass
            self.timer_job = None
        if self.timer_flash_job is not None:
            try:
                self.after_cancel(self.timer_flash_job)
            except Exception:
                pass
            self.timer_flash_job = None

    def update_timer_label(self, seconds: int | None = None):
        if self.timer_label is None:
            return
        if seconds is None:
            seconds = max(0, int(self.timer_left))
        m, s = divmod(max(0, int(seconds)), 60)
        self.timer_label.config(text=f"⏰ {m:02d}:{s:02d}")

    def handle_timer_notify(self):
        if self.timer_label is None:
            return
        original_bg = self.timer_label.cget("bg")
        self.timer_label.config(bg="#FFD966")

        def reset_bg():
            try:
                self.timer_label.config(bg=original_bg)
            except Exception:
                pass
        self.timer_flash_job = self.after(1500, reset_bg)

    def timer_tick(self):
        if self.timer_left <= 0:
            self.update_timer_label()
            return
        self.timer_left -= 1
        self.update_timer_label()
        if self.timer_left <= 0:
            self.handle_timer_notify()
            return
        self.timer_job = self.after(1000, self.timer_tick)

    def schedule_timers_for_card(self):
        self.cancel_timers()
        playback_seconds = get_effective_mode_timer(getattr(self.master, "selected_deck_id", None), "playback")
        self.timer_left = max(0, int(playback_seconds or 0))
        self.update_timer_label()
        if self.timer_left > 0:
            self.timer_job = self.after(1000, self.timer_tick)

        front = self.current_card.get("front") or ""
        back = self.current_card.get("back") or ""
        text_len = max(len(front), len(back))

        first_phase = 5

        if text_len <= 35:
            second_phase = 15
        else:
            min_second = 35
            max_second = 5 * 60
            max_len = 500

            clamped_len = min(text_len, max_len)
            if clamped_len <= 35:
                factor = 0.0
            else:
                factor = (clamped_len - 35) / (max_len - 35)

            second_phase = int(min_second + factor * (max_second - min_second))

        total_time = first_phase + second_phase

        self.auto_flip_id = self.after(first_phase * 1000, self.auto_show_answer)
        self.auto_next_id = self.after(total_time * 1000, self.auto_mark_and_next)

    def auto_show_answer(self):
        if not self.show_back:
            self.show_back = True
            self.update_view()

    def auto_mark_and_next(self):
        self.cancel_timers()
        card_id = self.current_card["id"]
        try:
            apply_srs_update(card_id, 0)
            update_statistics(self.master.selected_deck_id, remembered=False, forgotten=True, reviewed=True)

            row = get_card_by_id(card_id)
            if row:
                self.current_card["leitner_level"] = row["leitner_level"]
                self.current_card["next_review"] = row["next_review"]
        except Exception:
            pass

        self.master.update_overdue_badge()
        if self.current_card:
            gui_hooks.card_did_answer(self.current_card, 0)
        self.goto_next_card()

    def update_progress_view(self):
        if self.progress_canvas is None:
            return
        p = int(self.current_card.get("progress") or 0)
        p = max(0, min(100, p))

        self.progress_canvas.delete("all")

        self.progress_canvas.create_rectangle(1, 1, 259, 13, outline="#cccccc", fill="white")

        if p > 0:
            width = int(258 * p / 100)
            self.progress_canvas.create_rectangle(
                1, 1, 1 + width, 13,
                outline="", fill="#00aa00"
            )

        if self.progress_label is not None:
            self.progress_label.config(text=f"{p} / 100")

    def update_view(self):
        total = len(self.cards)
        idx = self.current_index + 1
        c = self.current_card

        self.lbl_status.config(
            text=f"Карточка {idx}/{total} | ID {c['id']}"
        )

        if getattr(self.master, "mw_context", None) is not None:
            self.master.mw_context.state["current_card_id"] = c.get("id")
        gui_hooks.card_will_show(c)

        # Навигация по карточкам (вместо "Забыл/Повторить")
        self.btn_prev.config(state=(tk.DISABLED if self.current_index <= 0 else tk.NORMAL))
        self.btn_next.config(state=(tk.DISABLED if self.current_index >= len(self.cards) - 1 else tk.NORMAL))


        romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        lvl = c["leitner_level"]
        phase = romans[min(max(lvl, 1), 10) - 1]
        header_text = f"Фаза {phase} | след. повтор: {c['next_review']}"
        if self.card_renderer is not None:
            self.card_renderer.set_header_text(header_text)

        if self.show_back:
            img_path = c["back_image_path"] or c["front_image_path"] or c["image_path"]
        else:
            img_path = c["front_image_path"] or c["image_path"]

        self.btn_show.config(text="Показать ответ" if not self.show_back else "Показать лицевую сторону")

        self.update_progress_view()
        self.update_timer_label()
        
        if self.card_renderer is not None:
            self.card_renderer.render(
                c,
                show_back=self.show_back,
                prefer_audio_side="back" if self.show_back else "front",
                image_override=img_path,
                header_text=header_text,
            )
            self.audio_widget = self.card_renderer.get_audio_widget()
        else:
            self.update_audio_player()

    def toggle_front_back(self):
        self.show_back = not self.show_back
        self.update_view()

    def mark_forgotten(self):
        self.cancel_timers()
        card_id = self.current_card["id"]
        result = apply_srs_update(card_id, 0)
        update_statistics(self.master.selected_deck_id, remembered=False, forgotten=True, reviewed=True)

        if result:
            self.current_card["leitner_level"] = result.get("phase")
            self.current_card["next_review"] = datetime.fromtimestamp(result.get("due", time.time())).isoformat()
            self.current_card["state"] = result.get("state")
            self.update_view()
        self.master.update_overdue_badge()
        messagebox.showinfo("Лейтнер", "Карточка отправлена в 1-й уровень (режим заучивания).")
        self.schedule_timers_for_card()

    def mark_remembered(self):
        self.cancel_timers()
        card_id = self.current_card["id"]

        result = apply_srs_update(card_id, 2)
        update_statistics(self.master.selected_deck_id, remembered=True, forgotten=False, reviewed=True)

        if result:
            self.current_card["leitner_level"] = result.get("phase")
            self.current_card["next_review"] = datetime.fromtimestamp(result.get("due", time.time())).isoformat()
            self.current_card["state"] = result.get("state")
            self.update_view()
        self.master.update_overdue_badge()
        messagebox.showinfo("Лейтнер", f"Отлично! Уровень карточки теперь: {self.current_card['leitner_level']}")
        self.goto_next_card()

    def increment_progress(self):
        card_id = self.current_card["id"]
        current = int(self.current_card.get("progress") or 0)
        if current >= 100:
            return
        new_value = min(100, current + 1)
        self.current_card["progress"] = new_value
        update_card_progress(card_id, new_value)
        self.update_progress_view()

    def goto_prev_card(self):
        """Перейти к предыдущей карточке (режим воспроизведения)."""
        try:
            self.cancel_timers()
        except Exception:
            pass
        if self.current_index <= 0:
            # Уже на первой карточке
            try:
                self.update_view()
                self.schedule_timers_for_card()
            except Exception:
                pass
            return
        self.current_index -= 1
        self.current_card = self.cards[self.current_index]
        self.show_back = False
        self.update_view()
        self.schedule_timers_for_card()

    def goto_next_card(self):
        self.cancel_timers()
        self.current_index += 1
        if self.current_index >= len(self.cards):
            messagebox.showinfo("Готово", "Карточки в этом режиме закончились.")
            self.destroy()
            return
        self.current_card = self.cards[self.current_index]
        self.show_back = False
        self.update_view()
        self.schedule_timers_for_card()

    def play_word(self):
        target_side = "back" if self.show_back else "front"
        audio_path = get_card_audio_path(self.current_card, prefer_side=target_side)
        if getattr(self, "audio_widget", None) and self.audio_widget.is_loaded():
            self.audio_widget.play()
            return
        if audio_path and os.path.exists(audio_path) and WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass

        back = ensure_plain_text(self.current_card.get("back") or "")
        first_line = back.splitlines()[0] if back else ""
        word = first_line.split()[0] if first_line else ""
        if not word:
            messagebox.showinfo("Озвучка", "Не удалось выделить слово для озвучки.")
            return
        speak_text(word)


class AudioEditorWindow:
    """Окно для нарезки аудио из видео на предложения"""

    def __init__(self, master, video_path, deck_id):
        self.master = master
        self.video_path = video_path
        self.deck_id = deck_id

        self.win = tk.Toplevel(master)
        self.win.title("Аудио-редактор: нарезка видео на предложения")
        self.win.geometry("1000x600")
        self.win.grab_set()
        
        # Загружаем аудио из видео
        self.audio_path = None
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0
        self.sentences = []  # [(start_time, end_time, text, audio_segment)]

        if not self.extract_audio_from_video():
            return
        self.create_widgets()
        
    def extract_audio_from_video(self):
        """Извлечь аудио из видео файла"""
        try:
            import tempfile
            import moviepy.editor as mp
            import librosa
            
            # Создаем временный файл для аудио
            temp_dir = tempfile.mkdtemp()
            self.audio_path = os.path.join(temp_dir, "extracted_audio.wav")
            
            # Извлекаем аудио из видео
            video = mp.VideoFileClip(self.video_path)
            audio = video.audio
            audio.write_audiofile(self.audio_path)
            video.close()

            # Загружаем аудио данные
            self.audio_data, self.sample_rate = librosa.load(self.audio_path, sr=None)
            self.duration = len(self.audio_data) / self.sample_rate

            return True

        except ImportError:
            messagebox.showerror("Ошибка", "Для работы с видео установите moviepy и librosa:\n"
                                           "pip install moviepy librosa")
            self.win.destroy()
            return False
        except Exception as e:
            messagebox.showerror("Ошибка", f"{type(e).__name__}: {e}")
            self.win.destroy()
            return False

    def create_widgets(self):
        """Создать интерфейс редактора"""
        # Основной фрейм
        ensure_premium_scrollbar_style(self.win, getattr(self.master, "palette", None))
        main_frame = ttk.Frame(self.win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Верхняя панель: информация о видео
        info_frame = ttk.LabelFrame(main_frame, text="Информация о видео")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        video_name = os.path.basename(self.video_path)
        duration_min = self.duration / 60
        
        info_text = f"""
        Видео: {video_name}
        Длительность: {duration_min:.2f} минут
        Частота дискретизации: {self.sample_rate} Гц
        """
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Панель управления воспроизведением
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.play_btn = ttk.Button(control_frame, text="▶ Воспроизвести аудио", command=self.play_audio)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.pause_btn = ttk.Button(control_frame, text="⏸ Пауза", command=self.pause_audio)
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Стоп", command=self.stop_audio)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Панель для нарезки на предложения
        split_frame = ttk.LabelFrame(main_frame, text="Нарезка на предложения")
        split_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопки для автоматической нарезки
        btn_frame = ttk.Frame(split_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Автонарезка по тишине", 
                  command=self.auto_split_by_silence).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="Распознать текст для всех сегментов", 
                  command=self.transcribe_all).pack(side=tk.LEFT, padx=2)
        
        # Список предложений
        sentences_frame = ttk.LabelFrame(main_frame, text="Предложения для карточек")
        sentences_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview для отображения предложений
        columns = ("№", "Начало", "Конец", "Текст", "Длина", "Действия")
        self.sentences_tree = ttk.Treeview(sentences_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.sentences_tree.heading(col, text=col)
            self.sentences_tree.column(col, width=80)
        
        self.sentences_tree.column("Текст", width=200)
        self.sentences_tree.column("Действия", width=100)
        
        # Scrollbar для treeview
        scrollbar = ttk.Scrollbar(
            sentences_frame,
            orient="vertical",
            command=self.sentences_tree.yview,
            style="Premium.Vertical.TScrollbar",
        )
        self.sentences_tree.configure(yscrollcommand=scrollbar.set)
        
        self.sentences_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Панель кнопок внизу
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(bottom_frame, text="Сгенерировать карточки", 
                  command=self.generate_cards).pack(side=tk.RIGHT, padx=2)
        
    def auto_split_by_silence(self):
        """Автоматическая нарезка по тишине"""
        try:
            import librosa
            import numpy as np
            
            # Найти интервалы тишины
            intervals = librosa.effects.split(self.audio_data, 
                                             top_db=30,  # Порог тишины
                                             frame_length=2048,
                                             hop_length=512)
            
            for i, (start, end) in enumerate(intervals):
                start_time = start / self.sample_rate
                end_time = end / self.sample_rate
                duration = end_time - start_time
                
                # Извлечь сегмент аудио
                audio_segment = self.audio_data[start:end]
                
                # Добавить в список
                self.sentences.append({
                    'index': i + 1,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'audio': audio_segment,
                    'text': f"Предложение {i+1}"
                })
                
                # Добавить в treeview
                self.sentences_tree.insert("", "end", values=(
                    i+1,
                    f"{start_time:.2f}с",
                    f"{end_time:.2f}с",
                    f"Предложение {i+1}",
                    f"{duration:.2f}с",
                    "Прослушать"
                ))
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выполнить нарезку: {e}")
            
    def transcribe_all(self):
        """Распознать текст для всех сегментов"""
        if not SR_AVAILABLE:
            messagebox.showerror("Распознавание недоступно", "Чтобы распознать речь, установите SpeechRecognition и попробуйте снова.")
            return
            
        r = sr.Recognizer()
        
        for i, sentence in enumerate(self.sentences):
            try:
                # Конвертировать numpy array в аудио данные для распознавания
                import io
                import wave
                import struct
                
                # Создать временный WAV файл
                audio_bytes = self.audio_segment_to_bytes(sentence['audio'])
                
                # Распознать текст
                audio_data = sr.AudioData(audio_bytes, self.sample_rate, 2)
                text = r.recognize_google(audio_data, language="de-DE")
                
                # Обновить текст
                sentence['text'] = text
                
                # Обновить treeview
                item_id = self.sentences_tree.get_children()[i]
                self.sentences_tree.item(item_id, values=(
                    sentence['index'],
                    f"{sentence['start']:.2f}с",
                    f"{sentence['end']:.2f}с",
                    text,
                    f"{sentence['duration']:.2f}с",
                    "Прослушать"
                ))
                
            except Exception as e:
                print(f"Ошибка распознавания сегмента {i}: {e}")
                
    def audio_segment_to_bytes(self, audio_data):
        """Конвертировать numpy array в байты WAV"""
        import io
        import wave
        import struct
        
        # Нормализовать аудио данные
        audio_data = np.int16(audio_data * 32767)
        
        # Создать WAV файл в памяти
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
            
        return wav_buffer.getvalue()
        
    def generate_cards(self):
        """Сгенерировать карточки из предложений"""
        from datetime import datetime
        import os
        
        # Создаем папку для аудио файлов если не существует
        os.makedirs("video_sentences", exist_ok=True)
        
        for sentence in self.sentences:
            if not sentence['text']:
                continue
                
            # Сохранить аудио сегмент в файл
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_filename = f"sentence_{sentence['index']}_{timestamp}.wav"
            audio_path = os.path.join("video_sentences", audio_filename)

            self.save_audio_segment(sentence['audio'], audio_path)

            # Создать карточку
            front = sentence['text']  # Немецкое предложение
            
            # Получить перевод
            translation = translate_sentence(sentence['text'], use_openai=True)
            
            # Формируем заднюю сторону с аудио плеером
            back = f"""{sentence['text']}

🇷🇺 Перевод: {translation}

🔊 Произношение:
[audio:{audio_path}]"""

            note_fields = {
                "word": sentence['text'],
                "translation": translation,
                "example": sentence['text'],
                "level": 1,
                "image": "",
                "front": front,
                "back": back,
                "front_image_path": None,
                "back_image_path": None,
                "audio_path": audio_path,
            }
            create_note_with_cards(
                self.deck_id,
                note_fields,
                note_type_id=ensure_generated_note_type_id(),
            )

        messagebox.showinfo("Успех", f"Создано {len(self.sentences)} карточек")
        self.win.destroy()
        
    def save_audio_segment(self, audio_data, path):
        """Сохранить аудио сегмент в файл"""
        import wave
        import struct
        
        # Нормализовать
        audio_data = np.int16(audio_data * 32767)
        
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    def play_audio(self):
        """Воспроизвести аудио"""
        if self.audio_path and os.path.exists(self.audio_path):
            play_audio_file(self.audio_path)
    
    def pause_audio(self):
        """Пауза аудио"""
        if WINSOUND_AVAILABLE:
            winsound.PlaySound(None, winsound.SND_PURGE)
    
    def stop_audio(self):
        """Остановить аудио"""
        if WINSOUND_AVAILABLE:
            winsound.PlaySound(None, winsound.SND_PURGE)


class VideoClipCardsWindow:
    """Видео → клипы → карточки (автозапуск + авто-STT + таймкоды)."""

    STT_LANG_CHOICES = ["Auto", "ru-RU", "en-US", "de-DE"]
    STT_MODEL_CHOICES = ["tiny", "base", "small", "medium"]

    def __init__(self, app: "AnkiApp", deck_id: int) -> None:
        self.app = app
        self.deck_id = deck_id
        self.video_path: str | None = None
        self.sentences: list[dict] = []
        self.selected_index: int | None = None
        self.player: VlcPlayerWidget | None = None
        self.stt_job_id = 0
        self.stt_running = False
        self.stt_progress_queue: queue.Queue | None = None
        self.stt_progress_mode = "determinate"
        self.stt_cancel_event: threading.Event | None = None
        self.stt_last_progress_at = 0.0
        self.stt_last_progress_value: float | None = None
        self.stt_watchdog_warned = False
        self.stt_total_duration_sec: float | None = None
        self.stt_segments: list[dict] = []
        self.video_duration_sec: float | None = None

        self.win = tk.Toplevel(app.root)
        self.win.title("Видео → клипы → карточки")
        self.win.geometry("1180x860")
        self.win.minsize(980, 760)
        self.win.grab_set()
        apply_dark_theme_to_window(self.win, app.palette)
        ensure_premium_scrollbar_style(self.win, app.palette)

        colors = getattr(app, "palette", None) or {}
        scroll_bg = colors.get("background") if colors else None
        outer_frame, canvas, content_frame = create_scrollable_content(self.win, bg=scroll_bg)
        outer_frame.pack(fill=tk.BOTH, expand=True)

        self.processing_mode_var = tk.StringVar(value="auto")

        self._build_video_block(content_frame)
        self._build_stt_block(content_frame)
        self._build_sentences_block(content_frame)
        self._build_editor_block(content_frame)
        self._build_save_block(content_frame)

        app.register_balance_observer(self._update_save_pricing)
        self._update_save_pricing()

        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_video_block(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Видео")
        frame.pack(fill=tk.X, padx=12, pady=(10, 8))

        path_row = ttk.Frame(frame)
        path_row.pack(fill=tk.X, padx=6, pady=(6, 4))
        self.video_path_var = tk.StringVar()
        ttk.Entry(path_row, textvariable=self.video_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def browse_video() -> None:
            path = filedialog.askopenfilename(
                title="Выберите видео файл",
                filetypes=[("Видео", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv"), ("Все файлы", "*.*")],
            )
            if path:
                self.video_path_var.set(path)
                self._load_video(path)

        ttk.Button(path_row, text="…", width=4, command=browse_video).pack(side=tk.LEFT, padx=(6, 0))

        self.video_status_var = tk.StringVar(value="Видео не выбрано")
        ttk.Label(frame, textvariable=self.video_status_var, foreground="gray").pack(
            anchor="w", padx=6, pady=(0, 6)
        )

        self.player_frame = ttk.Frame(frame)
        self.player_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        range_frame = ttk.LabelFrame(frame, text="Диапазон (Start/End)")
        range_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.range_start_var = tk.StringVar(value="00:00:00")
        self.range_end_var = tk.StringVar(value="00:00:10")
        ttk.Label(range_frame, text="Start:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(range_frame, textvariable=self.range_start_var, width=12).grid(
            row=0, column=1, sticky="w", padx=6, pady=4
        )
        ttk.Label(range_frame, text="End:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(range_frame, textvariable=self.range_end_var, width=12).grid(
            row=1, column=1, sticky="w", padx=6, pady=4
        )
        ttk.Button(
            range_frame,
            text="Распознать диапазон",
            command=lambda: (self.processing_mode_var.set("range"), self._request_stt(mode_override="range")),
        ).grid(row=0, column=2, padx=6, pady=4, rowspan=2, sticky="ns")
        ttk.Button(
            range_frame,
            text="Нарезать клип",
            command=self._cut_range_clip,
        ).grid(row=0, column=3, padx=6, pady=4, rowspan=2, sticky="ns")
        self.range_status_var = tk.StringVar(value="Диапазон не выбран")
        ttk.Label(range_frame, textvariable=self.range_status_var, foreground="gray").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 4)
        )

    def _render_player(self, path: str) -> None:
        for widget in self.player_frame.winfo_children():
            widget.destroy()
        self.player = None
        if is_vlc_available():
            try:
                player = VlcPlayerWidget(self.player_frame, path, width=620, height=340)
                if player.ensure_embedded():
                    player.pack(anchor="w", padx=4, pady=4)
                    self.player = player
                    player.play()
                    return
            except Exception:
                self.player = None
        ttk.Label(self.player_frame, text="Встроенный плеер недоступен").pack(anchor="w", padx=6, pady=6)
        ttk.Button(
            self.player_frame,
            text="Открыть во внешнем плеере",
            command=lambda: open_in_external_player(path),
        ).pack(anchor="w", padx=6, pady=(0, 6))

    def _build_stt_block(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Цифровой слух")
        frame.pack(fill=tk.BOTH, padx=12, pady=(0, 8))

        top_row = ttk.Frame(frame)
        top_row.pack(fill=tk.X, padx=6, pady=(6, 2))

        self.stt_status_var = tk.StringVar(value="Ожидание")
        ttk.Label(top_row, textvariable=self.stt_status_var, foreground="gray").pack(side=tk.LEFT)

        self.auto_stt_var = tk.BooleanVar(value=True)
        self.auto_stt_check = ttk.Checkbutton(
            top_row,
            text="Авто-цифровой слух",
            variable=self.auto_stt_var,
        )
        self.auto_stt_check.pack(side=tk.RIGHT)

        self.stt_progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.stt_progress.pack(fill=tk.X, padx=6, pady=(0, 4))

        self.stt_source_var = tk.StringVar(value="Источник: audio track из файла (WAV 16k mono)")
        ttk.Label(frame, textvariable=self.stt_source_var, foreground="gray").pack(
            anchor="w", padx=6, pady=(0, 4)
        )

        options_row = ttk.Frame(frame)
        options_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(options_row, text="Язык:").pack(side=tk.LEFT)
        self.stt_lang_var = tk.StringVar(value=self._default_stt_language())
        self.stt_lang_combo = ttk.Combobox(
            options_row,
            textvariable=self.stt_lang_var,
            values=self.STT_LANG_CHOICES,
            state="readonly",
            width=10,
        )
        self.stt_lang_combo.pack(side=tk.LEFT, padx=(6, 6))
        self.stt_lang_combo.bind("<<ComboboxSelected>>", lambda _e: self._maybe_auto_stt_language())
        ttk.Button(options_row, text="Сменить язык", command=self._open_stt_lang_menu).pack(
            side=tk.LEFT
        )
        ttk.Label(options_row, text="Модель:").pack(side=tk.LEFT, padx=(12, 0))
        global STT_WHISPER_MODEL
        default_model = STT_WHISPER_MODEL if STT_WHISPER_MODEL in self.STT_MODEL_CHOICES else "small"
        STT_WHISPER_MODEL = default_model
        self.stt_model_var = tk.StringVar(value=default_model)
        self.stt_model_combo = ttk.Combobox(
            options_row,
            textvariable=self.stt_model_var,
            values=self.STT_MODEL_CHOICES,
            state="readonly",
            width=7,
        )
        self.stt_model_combo.pack(side=tk.LEFT, padx=(6, 0))
        self.stt_model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_stt_model_change())

        mode_row = ttk.Frame(frame)
        mode_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(mode_row, text="Режим обработки:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_row,
            text="Авто (рекомендуется)",
            variable=self.processing_mode_var,
            value="auto",
            command=self._on_processing_mode_change,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(
            mode_row,
            text="Диапазон вручную",
            variable=self.processing_mode_var,
            value="range",
            command=self._on_processing_mode_change,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(
            mode_row,
            text="По кликам",
            variable=self.processing_mode_var,
            value="selection",
            command=self._on_processing_mode_change,
        ).pack(side=tk.LEFT, padx=(6, 0))

        limit_row = ttk.Frame(frame)
        limit_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(limit_row, text="Лимиты:").pack(side=tk.LEFT)
        self.stt_limit_minutes_var = tk.DoubleVar(value=4.0)
        self.stt_limit_chunks_var = tk.IntVar(value=12)
        ttk.Label(limit_row, text="не больше").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(limit_row, textvariable=self.stt_limit_minutes_var, width=6).pack(side=tk.LEFT)
        ttk.Label(limit_row, text="мин").pack(side=tk.LEFT, padx=(2, 8))
        ttk.Entry(limit_row, textvariable=self.stt_limit_chunks_var, width=6).pack(side=tk.LEFT)
        ttk.Label(limit_row, text="чанков").pack(side=tk.LEFT, padx=(2, 8))
        self.stt_limit_disabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            limit_row,
            text="Без ограничений",
            variable=self.stt_limit_disabled_var,
        ).pack(side=tk.LEFT)

        self.stt_engine_var = tk.StringVar(value=self._default_engine_label())
        ttk.Label(frame, textvariable=self.stt_engine_var, foreground="gray").pack(
            anchor="w", padx=6, pady=(0, 4)
        )

        self.stt_log_path_var = tk.StringVar(value=f"Лог: {get_log_path()}")
        log_row = ttk.Frame(frame)
        log_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Label(log_row, textvariable=self.stt_log_path_var, foreground="gray").pack(side=tk.LEFT)
        ttk.Button(
            log_row,
            text="Открыть лог",
            command=self._open_stt_log,
        ).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(
            log_row,
            text="Открыть папку логов",
            command=self._open_stt_log_dir,
        ).pack(side=tk.RIGHT)

        self.stt_text = scrolledtext.ScrolledText(frame, height=6)
        style_text_widget(self.stt_text, getattr(self.app, "palette", None) or {})
        self.stt_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        create_context_menu(self.stt_text)

        action_row = ttk.Frame(frame)
        action_row.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.stt_repeat_btn = ttk.Button(action_row, text="Повторить", command=self._request_stt)
        self.stt_repeat_btn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            action_row,
            text="Копировать",
            command=lambda: self.win.clipboard_clear() or self.win.clipboard_append(self._get_stt_text()),
        ).pack(side=tk.LEFT)
        self.stt_cancel_btn = ttk.Button(action_row, text="Отмена", command=self._cancel_stt, state=tk.DISABLED)
        self.stt_cancel_btn.pack(side=tk.LEFT, padx=(6, 0))

    def _default_stt_language(self) -> str:
        mapping = {"ru": "ru-RU", "en": "en-US", "de": "de-DE"}
        tts_lang = get_deck_tts_lang(self.deck_id, "").strip().lower()
        return mapping.get(tts_lang, "Auto")

    def _get_selected_stt_language(self) -> str | None:
        value = (self.stt_lang_var.get() or "").strip()
        if not value or value.lower() == "auto":
            return None
        return value

    def _open_stt_lang_menu(self) -> None:
        try:
            self.stt_lang_combo.focus_set()
            self.stt_lang_combo.event_generate("<Button-1>")
        except Exception:
            pass

    def _maybe_auto_stt_language(self) -> None:
        if self._should_auto_stt():
            self._request_stt()

    def _on_stt_model_change(self) -> None:
        global STT_WHISPER_MODEL
        selected = (self.stt_model_var.get() or "").strip()
        if selected:
            STT_WHISPER_MODEL = selected
        self.stt_engine_var.set(self._default_engine_label())
        if self._should_auto_stt():
            self._request_stt()

    def _should_auto_stt(self) -> bool:
        return bool(self.auto_stt_var.get() and self.processing_mode_var.get() == "auto")

    def _on_processing_mode_change(self) -> None:
        if self._should_auto_stt():
            self._request_stt()

    def _default_engine_label(self) -> str:
        model_name = STT_WHISPER_MODEL if STT_WHISPER_MODEL else "small"
        if FASTER_WHISPER_AVAILABLE:
            return f"Движок: Whisper {model_name} (VAD)"
        detail = FASTER_WHISPER_IMPORT_ERROR or "установите faster-whisper"
        return f"Движок: Google (fallback) — Whisper недоступен: {detail}"

    def _update_engine_label(self, engine: str, detail: str | None = None) -> None:
        model_name = STT_WHISPER_MODEL if STT_WHISPER_MODEL else "small"
        if engine == "whisper":
            label = f"Движок: Whisper {model_name} (VAD)"
        else:
            label = "Движок: Google (fallback)"
            reason = detail or (FASTER_WHISPER_IMPORT_ERROR if not FASTER_WHISPER_AVAILABLE else None)
            if reason:
                label += f" — {reason}"
        self.stt_engine_var.set(label)

    def _build_sentences_block(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Предложения и таймкоды")
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        columns = ("№", "Предложение", "Start", "End")
        self.sentences_tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        for col in columns:
            self.sentences_tree.heading(col, text=col)
            self.sentences_tree.column(col, width=90)
        self.sentences_tree.column("Предложение", width=520)
        self.sentences_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.sentences_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)
        self.sentences_tree.configure(yscrollcommand=scrollbar.set)

        self.sentences_tree.bind("<<TreeviewSelect>>", self._on_sentence_select)

        action_row = ttk.Frame(parent)
        action_row.pack(fill=tk.X, padx=12, pady=(0, 8))
        ttk.Button(action_row, text="Предпросмотр выбранного", command=self._open_preview).pack(side=tk.LEFT)

    def _build_editor_block(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Редактор предложения")
        frame.pack(fill=tk.BOTH, padx=12, pady=(0, 8))
        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        front_tab = ttk.Frame(notebook)
        back_tab = ttk.Frame(notebook)
        notebook.add(front_tab, text="Front")
        notebook.add(back_tab, text="Back")
        self.front_editor = build_rich_text_editor(front_tab, getattr(self.app, "palette", None) or {}, on_change=self._sync_editor_to_sentence)
        self.back_editor = build_rich_text_editor(back_tab, getattr(self.app, "palette", None) or {}, on_change=self._sync_editor_to_sentence)
        self.front_editor["frame"].pack(fill=tk.BOTH, expand=True)
        self.back_editor["frame"].pack(fill=tk.BOTH, expand=True)

    def _build_save_block(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Сохранение")
        frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        toggle_row = ttk.Frame(frame)
        toggle_row.pack(fill=tk.X, padx=6, pady=6)
        self.video_on_front = tk.BooleanVar(value=True)
        ttk.Radiobutton(
            toggle_row,
            text="Видео на Front",
            variable=self.video_on_front,
            value=True,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            toggle_row,
            text="Текст на Front (наоборот)",
            variable=self.video_on_front,
            value=False,
        ).pack(side=tk.LEFT, padx=(12, 0))

        cost_row = ttk.Frame(frame)
        cost_row.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.save_total_var = tk.StringVar(value="0")
        self.save_hint_var = tk.StringVar(value="")

        coin_icon, coin_icon_disabled = self.app._load_credit_icon_pair(size=28)
        self.coin_icon = coin_icon
        self.coin_icon_disabled = coin_icon_disabled

        cost_label = ttk.Frame(cost_row)
        cost_label.pack(side=tk.LEFT)
        ttk.Label(cost_label, text="Итого:").pack(side=tk.LEFT, padx=(0, 6))
        if coin_icon:
            coin_label = tk.Label(cost_label, image=coin_icon)
            coin_label.image = coin_icon
            coin_label.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(cost_label, textvariable=self.save_total_var).pack(side=tk.LEFT)

        self.save_btn = tk.Button(
            cost_row,
            text="Сохранить — 0",
            image=coin_icon,
            compound=tk.LEFT,
            padx=10,
            pady=4,
            command=self._save_cards,
        )
        self.save_btn.pack(side=tk.RIGHT)
        if coin_icon:
            self.save_btn.image = coin_icon

        ttk.Label(frame, textvariable=self.save_hint_var, foreground="red").pack(anchor="e", padx=6, pady=(0, 6))

    def _load_video(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        self.video_path = path
        self.sentences = []
        self.stt_segments = []
        self.selected_index = None
        try:
            self.sentences_tree.delete(*self.sentences_tree.get_children())
        except Exception:
            pass
        try:
            self.stt_text.delete("1.0", tk.END)
        except Exception:
            pass
        self.stt_engine_var.set(self._default_engine_label())
        self.stt_source_var.set("Источник: audio track из файла (WAV 16k mono)")
        duration = get_video_duration_seconds(path)
        self.video_duration_sec = float(duration) if duration else None
        if duration:
            self.video_status_var.set(f"Длительность: {format_hms(int(duration))}")
            self.range_start_var.set("00:00:00")
            self.range_end_var.set(format_hms(int(duration)))
            self.range_status_var.set(f"Диапазон: 00:00:00 → {format_hms(int(duration))}")
        else:
            self.video_status_var.set("Видео загружено")
            self.range_status_var.set("Диапазон: 00:00:00 → ?")
        self._render_player(path)
        if self._should_auto_stt():
            self._request_stt()

    def _get_stt_text(self) -> str:
        return self.stt_text.get("1.0", tk.END).rstrip("\n")

    def _set_stt_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        try:
            self.stt_repeat_btn.configure(state=state)
        except Exception:
            pass
        try:
            self.auto_stt_check.configure(state=state)
        except Exception:
            pass
        try:
            cancel_state = tk.DISABLED if enabled else tk.NORMAL
            self.stt_cancel_btn.configure(state=cancel_state)
        except Exception:
            pass

    def _set_stt_progress_mode(self, mode: str) -> None:
        if mode == self.stt_progress_mode:
            return
        self.stt_progress_mode = mode
        if mode == "indeterminate":
            self.stt_progress.configure(mode="indeterminate", maximum=100)
            self.stt_progress.start(10)
        else:
            self.stt_progress.stop()
            self.stt_progress.configure(mode="determinate", maximum=100)

    def _poll_stt_progress(self, job_id: int) -> None:
        if job_id != self.stt_job_id:
            return
        progress_queue = self.stt_progress_queue
        if progress_queue is None:
            return
        now = time.time()
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue.Empty:
                break
            status = event.get("status")
            mode = event.get("mode")
            overall = event.get("overall")
            heartbeat = event.get("heartbeat")
            total_duration = event.get("total_duration")
            if total_duration is not None:
                self.stt_total_duration_sec = float(total_duration)
            if mode:
                self._set_stt_progress_mode(mode)
            if status:
                self.stt_status_var.set(f"⏳ {status}")
                if heartbeat:
                    self.stt_last_progress_at = now
                    self.stt_watchdog_warned = False
            if overall is not None:
                self.stt_progress.configure(value=max(0.0, min(100.0, overall)))
                if overall != self.stt_last_progress_value:
                    self.stt_last_progress_value = overall
                    self.stt_last_progress_at = now
                    self.stt_watchdog_warned = False
            if heartbeat and overall is None and status is None:
                self.stt_last_progress_at = now
                self.stt_watchdog_warned = False
        if (
            self.stt_running
            and not self.stt_watchdog_warned
            and self.stt_last_progress_at
            and (time.time() - self.stt_last_progress_at)
            > max(60, int((self.stt_total_duration_sec or 0.0) * 0.7) or STT_WATCHDOG_IDLE_SEC)
        ):
            self.stt_watchdog_warned = True
            self.stt_status_var.set("⚠️ Похоже, процесс завис. См. лог.")
            _append_stt_log_line("watchdog_warning=stt_progress_stalled")
            _append_stt_log_line("TIMEOUT/DEADLOCK=stt_progress_stalled")
            messagebox.showwarning(
                "STT завис",
                f"Похоже, процесс распознавания завис или не отвечает.\n"
                f"Лог: {get_log_path()}",
                parent=self.win,
            )
        if self.stt_running and job_id == self.stt_job_id:
            self.win.after(100, lambda: self._poll_stt_progress(job_id))

    def _cancel_stt(self) -> None:
        if self.stt_running and self.stt_cancel_event is not None:
            self.stt_cancel_event.set()
            self.stt_status_var.set("⏳ Отменяю…")
            _append_stt_log_line("stt_cancel_requested=true")

    def _open_stt_log(self) -> None:
        log_path = get_log_path()
        try:
            if not os.path.exists(log_path):
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8"):
                    pass
            if hasattr(os, "startfile"):
                os.startfile(log_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", log_path])
            else:
                subprocess.Popen(["xdg-open", log_path])
        except Exception as exc:
            messagebox.showwarning("Лог", f"Не удалось открыть лог: {exc}", parent=self.win)

    def _open_stt_log_dir(self) -> None:
        log_dir = get_log_dir()
        try:
            if hasattr(os, "startfile"):
                os.startfile(log_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", log_dir])
            else:
                subprocess.Popen(["xdg-open", log_dir])
        except Exception as exc:
            messagebox.showwarning("Логи", f"Не удалось открыть папку логов: {exc}", parent=self.win)

    def _get_manual_range(self) -> tuple[float, float] | None:
        start_sec = parse_timecode_to_seconds((self.range_start_var.get() or "").strip())
        end_sec = parse_timecode_to_seconds((self.range_end_var.get() or "").strip())
        if start_sec is None or end_sec is None:
            messagebox.showwarning(
                "Диапазон",
                "Введите корректные значения Start/End (SS, MM:SS или HH:MM:SS).",
                parent=self.win,
            )
            return None
        if end_sec <= start_sec:
            messagebox.showwarning(
                "Диапазон",
                "Время окончания должно быть больше времени начала.",
                parent=self.win,
            )
            return None
        self.range_status_var.set(
            f"Диапазон: {format_hms(int(start_sec))} → {format_hms(int(end_sec))}"
        )
        return float(start_sec), float(end_sec)

    def _cut_range_clip(self) -> None:
        if not self.video_path:
            messagebox.showwarning("Клип", "Сначала выберите видео файл.", parent=self.win)
            return
        manual_range = self._get_manual_range()
        if manual_range is None:
            return
        start_sec, end_sec = manual_range
        output_dir = os.path.join(MEDIA_FOLDER, "clips", str(self.deck_id))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"clip_{ts}_{int(start_sec * 1000)}_{int(end_sec * 1000)}"
        self.range_status_var.set("⏳ Нарезаю клип…")

        def _worker() -> tuple[bool, str, str]:
            try:
                clip_path, poster_path = cut_video_clip_with_poster_h264(
                    self.video_path,
                    start_sec,
                    end_sec,
                    output_dir,
                    base_name,
                )
                return True, clip_path, poster_path
            except Exception as exc:
                return False, str(exc), ""

        def _finish(result: tuple[bool, str, str]) -> None:
            ok, clip_path, poster_path = result
            if not ok:
                self.range_status_var.set("❌ Ошибка нарезки")
                messagebox.showerror("Клип", f"Не удалось нарезать клип: {clip_path}", parent=self.win)
                return
            self.range_status_var.set(f"✅ Клип: {os.path.basename(clip_path)}")
            messagebox.showinfo(
                "Клип создан",
                f"Клип сохранён: {clip_path}\nПостер: {poster_path}",
                parent=self.win,
            )

        def _thread_runner() -> None:
            result = _worker()
            self.win.after(0, lambda: _finish(result))

        threading.Thread(target=_thread_runner, daemon=True).start()

    def _get_auto_range(self) -> tuple[float | None, float | None, float | None]:
        duration = self.video_duration_sec
        if duration is None and self.video_path:
            duration = get_video_duration_seconds(self.video_path) or None
            self.video_duration_sec = duration
        if self.stt_limit_disabled_var.get():
            return None, None, duration
        max_seconds: list[float] = []
        try:
            minutes = float(self.stt_limit_minutes_var.get())
        except Exception:
            minutes = 0.0
        try:
            chunks = int(self.stt_limit_chunks_var.get())
        except Exception:
            chunks = 0
        if minutes > 0:
            max_seconds.append(minutes * 60.0)
        if chunks > 0:
            max_seconds.append(chunks * 25.0)
        if not max_seconds:
            return None, None, duration
        limit = min(max_seconds)
        end_sec = min(duration, limit) if duration else limit
        return 0.0, float(end_sec), duration

    def _request_stt(self, mode_override: str | None = None) -> None:
        if self.stt_running:
            return
        if not self.video_path:
            messagebox.showwarning("Видео", "Сначала выберите видео файл.", parent=self.win)
            return
        mode = mode_override or self.processing_mode_var.get()
        start_sec: float | None = None
        end_sec: float | None = None
        duration = self.video_duration_sec
        range_label = ""
        if mode == "range":
            manual_range = self._get_manual_range()
            if manual_range is None:
                return
            start_sec, end_sec = manual_range
            range_label = f"Диапазон: {format_hms(int(start_sec))} → {format_hms(int(end_sec))}"
        elif mode == "selection":
            if self.selected_index is None or self.selected_index >= len(self.sentences):
                messagebox.showwarning(
                    "Фрагмент",
                    "Выберите предложение в таблице, чтобы распознать текущий фрагмент.",
                    parent=self.win,
                )
                return
            sentence = self.sentences[self.selected_index]
            start_sec = float(sentence.get("start", 0.0))
            end_sec = float(sentence.get("end", start_sec))
            if end_sec <= start_sec:
                messagebox.showwarning("Фрагмент", "Некорректный диапазон выбранного фрагмента.", parent=self.win)
                return
            range_label = f"Выбранное: {start_sec:.2f} → {end_sec:.2f}"
        else:
            start_sec, end_sec, duration = self._get_auto_range()
            if start_sec is None:
                start_sec = 0.0 if end_sec is not None else None
            if end_sec is not None and duration:
                range_label = (
                    f"Авто-режим: {format_hms(int(end_sec))} из {format_hms(int(duration))}"
                )
            elif end_sec is not None:
                range_label = f"Авто-режим: первые {format_hms(int(end_sec))}"
            else:
                range_label = "Авто-режим: весь файл"
        if range_label:
            self.stt_source_var.set(f"Источник: {range_label}")
        self.stt_running = True
        self.stt_job_id += 1
        job_id = self.stt_job_id
        self._set_stt_controls_enabled(False)
        self.stt_status_var.set("⏳ Извлекаю аудио…")
        self.stt_progress.configure(value=0)
        self._set_stt_progress_mode("determinate")
        self.stt_progress_queue = queue.Queue()
        self.stt_cancel_event = threading.Event()
        self.stt_last_progress_at = time.time()
        self.stt_last_progress_value = 0.0
        self.stt_watchdog_warned = False
        self.stt_total_duration_sec = None

        def _worker() -> tuple[str, object]:
            try:
                log_context = [
                    f"video_path={self.video_path}",
                    "clip_path=None",
                    f"start={start_sec}",
                    f"end={end_sec}",
                    "source_type=file",
                    f"mode={mode}",
                ]
                text, segments, engine, engine_detail = run_stt_pipeline(
                    self.video_path,
                    start=start_sec,
                    end=end_sec,
                    lang=self._get_selected_stt_language(),
                    punctuate=True,
                    log_context=log_context,
                    progress_queue=self.stt_progress_queue,
                    cancel_event=self.stt_cancel_event,
                )
                duration = get_video_duration_seconds(self.video_path)
                return (
                    "done",
                    {
                        "text": text,
                        "duration": duration,
                        "segments": segments,
                        "engine": engine,
                        "engine_detail": engine_detail,
                        "start": start_sec,
                        "end": end_sec,
                        "range_label": range_label,
                    },
                )
            except STTCancelled:
                return ("cancelled", None)
            except STTError as exc:
                return ("error", {"message": exc.user_message, "reason": exc.reason})
            except Exception as exc:
                return ("error", {"message": str(exc), "reason": "unknown"})

        def _finish(result: tuple[str, object]) -> None:
            self.stt_running = False
            self._set_stt_controls_enabled(True)
            if job_id != self.stt_job_id:
                return
            kind, payload = result
            if kind == "error":
                error_info = payload if isinstance(payload, dict) else {"message": str(payload)}
                message = str(error_info.get("message") or "Ошибка STT")
                reason = str(error_info.get("reason") or "")
                self._set_stt_progress_mode("determinate")
                self.stt_progress.configure(value=0)
                self.stt_status_var.set(f"❌ Ошибка: {message}")
                show_stt_error_dialog(self.win, "Ошибка распознавания", message)
                if reason in {"timeout", "deadlock"}:
                    messagebox.showwarning(
                        "STT завис/таймаут",
                        f"Распознавание зависло или превысило лимит времени.\nЛог: {get_log_path()}",
                        parent=self.win,
                    )
                return
            if kind == "cancelled":
                self._set_stt_progress_mode("determinate")
                self.stt_progress.configure(value=0)
                self.stt_status_var.set("⛔ Отменено")
                return
            payload = payload or {}
            text = str(payload.get("text") or "")
            duration = payload.get("duration")
            segments = payload.get("segments") or []
            engine = str(payload.get("engine") or "")
            engine_detail = str(payload.get("engine_detail") or "")
            start_offset = payload.get("start")
            range_label = str(payload.get("range_label") or "")
            if range_label:
                self.stt_source_var.set(f"Источник: {range_label}")
            self._append_stt_text(text)
            self._set_stt_progress_mode("determinate")
            self.stt_progress.configure(value=100)
            self.stt_status_var.set("✅ Готово")
            self._update_engine_label(engine, engine_detail or None)
            self._append_sentences_from_text(text, duration, segments, base_offset=start_offset)
            self._update_save_pricing()

        def _thread_runner() -> None:
            result = _worker()
            self.win.after(0, lambda: _finish(result))

        self._poll_stt_progress(job_id)
        threading.Thread(target=_thread_runner, daemon=True).start()

    def _build_sentences_from_text(
        self,
        text: str,
        duration: float | None,
        segments: list[dict] | None,
    ) -> None:
        self.sentences = []
        self.stt_segments = []
        self.sentences_tree.delete(*self.sentences_tree.get_children())
        self._append_sentences_from_text(text, duration, segments or [])
        if self.sentences:
            self.sentences_tree.selection_set("0")
            self._select_sentence(0)

    def _append_stt_text(self, text: str) -> None:
        new_text = (text or "").strip()
        if not new_text:
            return
        existing = self._get_stt_text().strip()
        combined = f"{existing}\n{new_text}".strip() if existing else new_text
        self.stt_text.delete("1.0", tk.END)
        self.stt_text.insert("1.0", combined)

    def _append_sentences_from_text(
        self,
        text: str,
        duration: float | None,
        segments: list[dict],
        *,
        base_offset: float | None = None,
    ) -> None:
        sentences = split_into_sentences(text or "")
        if not sentences and text.strip():
            sentences = [text.strip()]
        if not sentences:
            return
        timecodes = self._compute_sentence_timecodes(sentences, duration, segments or [])
        if base_offset is not None and not segments:
            timecodes = [
                (start + float(base_offset), end + float(base_offset))
                for start, end in timecodes
            ]
        start_index = len(self.sentences)
        for idx, sentence in enumerate(sentences):
            start, end = timecodes[idx]
            sentence_html = f"<div>{html.escape(sentence)}</div>"
            entry = {
                "index": start_index + idx + 1,
                "sentence": sentence,
                "start": float(start),
                "end": float(end),
                "text_html": sentence_html,
            }
            self.sentences.append(entry)
            self.sentences_tree.insert(
                "",
                "end",
                iid=str(start_index + idx),
                values=(start_index + idx + 1, sentence, f"{start:.2f}", f"{end:.2f}"),
            )
        if segments:
            self.stt_segments.extend(segments)

    def _compute_sentence_timecodes(
        self,
        sentences: list[str],
        duration: float | None,
        segments: list[dict],
    ) -> list[tuple[float, float]]:
        if not sentences:
            return []
        if segments:
            return self._match_sentence_timecodes(sentences, segments)
        duration = float(duration or 0)
        if duration <= 0:
            duration = max(2.0 * len(sentences), 2.0)
        weights = []
        for sentence in sentences:
            words = re.findall(r"[\\w']+", sentence, flags=re.UNICODE)
            weights.append(max(len(words), 1))
        total = float(sum(weights)) or 1.0
        results: list[tuple[float, float]] = []
        cursor = 0.0
        for idx, weight in enumerate(weights):
            if idx == len(weights) - 1:
                end = duration
            else:
                end = cursor + duration * (weight / total)
            if end <= cursor:
                end = cursor + max(0.2, duration / max(len(weights), 1))
            results.append((cursor, end))
            cursor = end
        return results

    def _match_sentence_timecodes(
        self,
        sentences: list[str],
        segments: list[dict],
    ) -> list[tuple[float, float]]:
        def _normalize(text: str) -> str:
            tokens = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower())
            return "".join(tokens)

        results: list[tuple[float, float]] = []
        seg_idx = 0
        last_end = 0.0
        for sentence in sentences:
            target = _normalize(sentence)
            start = None
            end = None
            buffer = ""
            while seg_idx < len(segments):
                seg = segments[seg_idx]
                seg_idx += 1
                seg_text = _normalize(seg.get("text", ""))
                if not seg_text:
                    continue
                if start is None:
                    start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                buffer += seg_text
                if target:
                    if target in buffer or len(buffer) >= len(target):
                        break
                else:
                    break
            if start is None:
                start = last_end
                end = last_end
            if end is None:
                end = start
            last_end = max(last_end, end)
            results.append((start, end))
        return results

    def _on_sentence_select(self, _event=None) -> None:
        selection = self.sentences_tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        self._select_sentence(idx)

    def _select_sentence(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.sentences):
            return
        self.selected_index = idx
        sentence = self.sentences[idx]
        start_sec = sentence.get("start", 0)
        end_sec = sentence.get("end", start_sec)
        try:
            self.range_start_var.set(format_hms(int(float(start_sec))))
            self.range_end_var.set(format_hms(int(float(end_sec))))
            self.range_status_var.set(
                f"Диапазон: {format_hms(int(float(start_sec)))} → {format_hms(int(float(end_sec)))}"
            )
        except Exception:
            pass
        if self.player is not None:
            try:
                self.player.player.set_time(int(float(start_sec) * 1000))
                self.player.seek_var.set(int(float(start_sec) * 1000))
            except Exception:
                pass
        self.front_editor["set_html"](sentence.get("text_html") or "")
        self.back_editor["set_html"](sentence.get("text_html") or "")

    def _sync_editor_to_sentence(self) -> None:
        if self.selected_index is None:
            return
        if self.selected_index >= len(self.sentences):
            return
        sentence_html = self.front_editor["get_html"]()
        self.sentences[self.selected_index]["text_html"] = sentence_html

    def _build_card_html(self, sentence: dict) -> tuple[str, str]:
        text_html = sentence.get("text_html") or f"<div>{html.escape(sentence.get('sentence') or '')}</div>"
        clip_html = build_video_html(sentence.get("clip_path"), sentence.get("poster_path"))
        if self.video_on_front.get():
            return clip_html, text_html
        return text_html, clip_html

    def _open_preview(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.sentences):
            messagebox.showinfo("Предпросмотр", "Сначала выберите предложение.", parent=self.win)
            return
        sentence = self.sentences[self.selected_index]
        start = sentence.get("start", 0)
        end = sentence.get("end", 0)
        clip_html = build_video_html(self.video_path, None)
        text_html = sentence.get("text_html") or f"<div>{html.escape(sentence.get('sentence') or '')}</div>"
        if self.video_on_front.get():
            front_html, back_html = clip_html, text_html
        else:
            front_html, back_html = text_html, clip_html

        print("[PREVIEW SELECTED]", sentence.get("sentence"), start, end)
        print("[PREVIEW FRONT HTML]", (front_html or "")[:200])
        print("[PREVIEW BACK HTML]", (back_html or "")[:200])

        preview_win = tk.Toplevel(self.win)
        preview_win.title("Предпросмотр выбранного")
        preview_win.geometry("900x620")
        preview_win.grab_set()
        apply_dark_theme_to_window(preview_win, getattr(self.app, "palette", None) or {})

        container = tk.Frame(preview_win, bg=DARK_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        controls = tk.Frame(container, bg=DARK_BG)
        controls.pack(fill=tk.X, pady=(0, 8))

        side_var = tk.StringVar(value="front")

        card_wrap = tk.Frame(
            container,
            bg=DARK_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
            width=CARD_VIEW_WIDTH,
            height=CARD_VIEW_HEIGHT,
        )
        card_wrap.pack(padx=10, pady=10)
        card_wrap.pack_propagate(False)
        html_view = CardHtmlView(card_wrap)

        def update_view() -> None:
            card_bg, card_text, _ = get_card_surface_colors(card_wrap)
            html_value = back_html if side_var.get() == "back" else front_html
            html_view.set_html(html_value or "", card_bg=card_bg, card_text_color=card_text)

        def set_side(side: str) -> None:
            side_var.set(side)
            update_view()

        ttk.Button(controls, text="Лицевая", command=lambda: set_side("front")).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Обратная", command=lambda: set_side("back")).pack(side=tk.LEFT, padx=4)

        card_bg, card_text, _ = get_card_surface_colors(card_wrap)
        html_view.set_html("<b>TEST_BOLD</b> <span style=\"color:red\">TEST_RED</span>", card_bg=card_bg, card_text_color=card_text)
        preview_win.after(200, update_view)

    def _get_video_clip_card_price(self) -> int:
        self.app.user_account = ensure_user_account(self.app.user_id)
        self.app.user_profile = ensure_user_profile_row(self.app.user_id)
        now_ts = int(time.time())
        premium_plus = bool(self.app.user_profile.get("premium_plus")) or self.app.user_account.get("status") == "premium_plus"
        if premium_plus:
            return 5
        if self.app.is_premium_active() or int(self.app.user_account.get("premium_until") or 0) > now_ts:
            return 10
        return 25

    def _update_save_pricing(self) -> None:
        count = len(self.sentences)
        price = self._get_video_clip_card_price()
        total = price * count
        self.save_total_var.set(str(total))
        self.save_btn.configure(text=f"Сохранить — {total}")
        if self.stt_running:
            self.save_btn.configure(state=tk.DISABLED, image=self.coin_icon_disabled or self.coin_icon)
            self.save_hint_var.set("Идёт распознавание…")
            return
        if count == 0:
            self.save_btn.configure(state=tk.DISABLED, image=self.coin_icon_disabled or self.coin_icon)
            self.save_hint_var.set("Нет предложений")
            return
        if not self.app.can_afford(total):
            self.save_btn.configure(state=tk.DISABLED, image=self.coin_icon_disabled or self.coin_icon)
            self.save_hint_var.set("Не хватает кредитов")
            return
        self.save_btn.configure(state=tk.NORMAL, image=self.coin_icon or self.coin_icon_disabled)
        self.save_hint_var.set("")

    def _save_cards(self) -> None:
        if not self.video_path:
            messagebox.showwarning("Видео", "Сначала выберите видео файл.", parent=self.win)
            return
        if not self.sentences:
            messagebox.showwarning("Сохранение", "Нет предложений для сохранения.", parent=self.win)
            return
        price = self._get_video_clip_card_price()
        total_cost = price * len(self.sentences)
        if not self.app.can_afford(total_cost):
            self._update_save_pricing()
            messagebox.showwarning("Кредиты", "Недостаточно кредитов.", parent=self.win)
            return

        self.save_btn.configure(state=tk.DISABLED)
        output_dir = os.path.join(MEDIA_FOLDER, "clips", str(self.deck_id))

        def _worker() -> tuple[bool, str]:
            conn = get_connection()

            def _op() -> int:
                spend_credits_in_transaction(
                    conn,
                    self.app.user_id,
                    total_cost,
                    "video_clip_cards",
                    meta={"deck_id": self.deck_id, "count": len(self.sentences)},
                )
                created = 0
                for idx, sentence in enumerate(self.sentences, start=1):
                    base_name = f"clip_{idx}"
                    clip_path, poster_path = cut_video_clip_with_poster_h264(
                        self.video_path,
                        float(sentence.get("start", 0)),
                        float(sentence.get("end", 0)),
                        output_dir,
                        base_name,
                    )
                    sentence["clip_path"] = clip_path
                    sentence["poster_path"] = poster_path
                    video_html = build_video_html(clip_path, poster_path)
                    text_html = sentence.get("text_html") or f"<div>{html.escape(sentence.get('sentence') or '')}</div>"
                    if self.video_on_front.get():
                        front_html, back_html = video_html, text_html
                    else:
                        front_html, back_html = text_html, video_html

                    note_fields = {
                        "word": sentence.get("sentence") or "",
                        "translation": "",
                        "example": "",
                        "level": 1,
                        "image": poster_path or "",
                        "front": front_html,
                        "back": back_html,
                        "front_image_path": poster_path if self.video_on_front.get() else None,
                        "back_image_path": poster_path if not self.video_on_front.get() else None,
                        "audio_path": None,
                        "front_html": front_html,
                        "back_html": back_html,
                        "front_video_html": video_html if self.video_on_front.get() else "",
                        "back_video_html": video_html if not self.video_on_front.get() else "",
                        "front_video": video_html if self.video_on_front.get() else "",
                        "back_video": video_html if not self.video_on_front.get() else "",
                    }
                    note_id, _ = create_note_with_cards_in_transaction(
                        conn,
                        self.deck_id,
                        note_fields,
                        note_type_id=ensure_generated_note_type_id(conn),
                        tags="video clip",
                    )
                    insert_media(conn, note_id=note_id, type="video", path=clip_path, side="front" if self.video_on_front.get() else "back")
                    insert_media(conn, note_id=note_id, type="image", path=poster_path, side="front" if self.video_on_front.get() else "back")
                    created += 1
                return created

            try:
                created = commit_with_retry(conn, _op)
            except Exception as exc:
                conn.rollback()
                return False, str(exc)
            finally:
                conn.close()
            return True, str(created)

        def _finish(result: tuple[bool, str]) -> None:
            ok, payload = result
            if not ok:
                messagebox.showerror("Ошибка", f"Не удалось сохранить карточки: {payload}", parent=self.win)
                self._update_save_pricing()
                return
            created = int(payload)
            self.app.refresh_balance_display()
            self.app.refresh_decks()
            self.app.update_deck_preview()
            self.app.update_overdue_badge()
            messagebox.showinfo(
                "Сохранено",
                f"✅ Сохранено {created} карточек, списано {total_cost} кредитов",
                parent=self.win,
            )
            self._on_close()

        def _thread_runner() -> None:
            result = _worker()
            self.win.after(0, lambda: _finish(result))

        threading.Thread(target=_thread_runner, daemon=True).start()

    def _on_close(self) -> None:
        self.app.unregister_balance_observer(self._update_save_pricing)
        self.win.destroy()


class VideoEditorWindow(AudioEditorWindow):
    """Окно редактирования видео (использует аудио-редактор)."""

    def __init__(self, parent, video_path, deck_id):
        super().__init__(parent, video_path, deck_id)
        self.win.title("Видео → клипы → карточки")


if __name__ == "__main__":
    init_db()
    init_dictionary()
    app = AnkiApp()
    app.mainloop()
# PATCH: tabs moved + dark scrollbar + video embed fixed + upload video in generator + unified card renderer
# PATCH: unify card renderer sizes + white video background + image-over-video rule
# PATCH: fix random image shrink (configure debounce + min size + orig cache) + fix preview image render (PhotoImage refs + shared renderer)
# PATCH: CSV image import packs (limit+cost by PRO), coin-click charge, progress-synced hard stop, PRO-only options with crown
# PATCH: OCR PRO postprocess pipeline + OCR result textbox + paid card autogen (pro/free) + image placeholder sync + repeated-word masking rule
# PATCH: OCR crash-proof (safe_action + traceback), dark scrollbar restored, OCR result textbox, paid card autogen + placeholder images + repeated-word masking
# PATCH: crash-proof postprocess via subprocess + faulthandler + global hooks + always-on logs
