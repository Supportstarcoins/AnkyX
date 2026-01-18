from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PilImage


def _get_pil_version() -> str:
    try:
        import PIL

        return getattr(PIL, "__version__", "unknown")
    except Exception:
        return "unknown"


def load_image_for_ocr(path: str) -> PilImage:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Для загрузки изображений нужен модуль Pillow.\n"
            "Установите его: C:\\AnkyX-main\\venv\\Scripts\\python.exe -m pip install pillow"
        ) from exc

    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Не удалось открыть изображение: {path}\n{exc}") from exc

    try:
        img = Image.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Не удалось прочитать изображение: {path}\n{exc}") from exc

    img._anky_original_format = getattr(img, "format", None) or "unknown"
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.load()
    return img


def _format_image_diag(image_path: str, img: PilImage) -> str:
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
