from __future__ import annotations

import datetime
import traceback
from pathlib import Path

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageOps = None
    PIL_AVAILABLE = False

MAX_PREVIEW_PIXELS = 25_000_000


def _pil_lanczos():
    if Image is None:
        return 1
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        return getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", 1))


def log_image_error(path: str, exc: Exception, log_path: str = "image_load_error.log") -> None:
    try:
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        log_entry = (
            f"[{timestamp}] Ошибка загрузки изображения: {path}\n"
            f"{exc}\n"
            f"{traceback.format_exc()}\n"
        )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a", encoding="utf-8", errors="ignore") as handle:
            handle.write(log_entry)
    except Exception:
        pass


def _convert_image_mode(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        return img.convert("RGBA")
    if img.mode == "P":
        return img.convert("RGBA")
    return img.convert("RGB")


def load_preview_image(
    path: str,
    target_size: tuple[int, int] | None,
    *,
    max_pixels: int = MAX_PREVIEW_PIXELS,
    max_dimension: int | None = None,
    log_path: str = "image_load_error.log",
) -> tuple[Image.Image, bool]:
    if not PIL_AVAILABLE or Image is None or ImageOps is None:
        raise RuntimeError("Pillow не установлен")
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = _convert_image_mode(img)

        resized_for_pixels = False
        total_pixels = img.width * img.height
        if max_pixels and total_pixels > max_pixels:
            scale = (max_pixels / float(total_pixels)) ** 0.5
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, _pil_lanczos())
            resized_for_pixels = True

        if max_dimension:
            if img.width > max_dimension or img.height > max_dimension:
                img.thumbnail((max_dimension, max_dimension), _pil_lanczos())

        if target_size:
            width, height = target_size
            img.thumbnail((max(1, int(width)), max(1, int(height))), _pil_lanczos())

        return img, resized_for_pixels
    except Exception as exc:
        log_image_error(path, exc, log_path=log_path)
        raise
