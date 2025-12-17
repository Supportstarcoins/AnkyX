"""Advanced OCR preprocessing for document photos (DE + RU).

This module now provides a richer pipeline for difficult page photos while
keeping the legacy quick mode intact. It supports:

* Safe image loading (bytes -> OpenCV/Pillow)
* Perspective alignment
* Shadow/illumination normalization
* Upscaling, denoising, binarization, deskew
* Automatic column split
* Optional PaddleOCR "PRO" mode

All steps are optional and controlled by UI settings so existing behaviour is
preserved.
"""

from __future__ import annotations

import datetime
import os
import platform
import re
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps
import pytesseract

from tesseract_setup import get_tesseract_diag, to_short_path
from main import load_image_for_ocr, _format_image_diag

ProgressCallback = Callable[[int, int, str], None]


try:  # PaddleOCR (optional)
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except Exception:
    PaddleOCR = None  # type: ignore
    PADDLE_AVAILABLE = False


@dataclass
class OcrRunOptions:
    ocr_mode: Literal["fast", "pro", "two_columns"] = "fast"
    lang_mode: str = "deu+rus"
    perspective_correction: bool = True
    flatten_background: bool = True
    binarize_mode: Literal["adaptive", "otsu", "none"] = "adaptive"
    deskew: bool = True
    debug_images: bool = False
    psm: int = 4
    preserve_spaces: bool = True
    dictionary_mode: bool = True
    dpi: int | None = 300
    split_offset_percent: float = 0.0
    prefer_paddle_for_columns: bool = True


def _report(cb: ProgressCallback | None, step: int, total: int, label: str):
    if cb:
        cb(step, total, label)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def load_image_any(path: str) -> np.ndarray:
    """Safely load an image into BGR numpy array.

    The function first attempts OpenCV decoding from bytes, then falls back to
    Pillow. A clear error is raised if the file cannot be read at all.
    """

    with open(path, "rb") as f:
        data = f.read()

    np_data = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if img is None:
        try:
            pil_img = Image.open(Path(path))
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Не удалось открыть изображение: {path}\n{e}") from e
    return img


def detect_and_warp_page(bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect page contour and apply perspective correction if possible."""

    debug = {"found_quad": False, "quad": None}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    quad = None
    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 5000:
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        return bgr, debug

    debug["found_quad"] = True
    debug["quad"] = quad.tolist()
    rect = _order_points(quad.astype("float32"))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(bgr, M, (maxWidth, maxHeight))
    return warped, debug


def flatten_background(gray: np.ndarray) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    norm = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)


def upscale_and_binarize(gray: np.ndarray, mode: str = "adaptive") -> np.ndarray:
    h, w = gray.shape[:2]
    gray_up = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray_up, h=15, templateWindowSize=7, searchWindowSize=21)
    if mode == "adaptive":
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            11,
        )
    elif mode == "otsu":
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = denoised
    return binary


def deskew(binary_or_gray: np.ndarray) -> np.ndarray:
    img = binary_or_gray
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return binary_or_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary_or_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def split_columns_auto(binary_or_gray: np.ndarray, offset_percent: float = 0.0):
    """Split page into two columns using whitespace valley detection."""

    if len(binary_or_gray.shape) == 3:
        gray = cv2.cvtColor(binary_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_or_gray

    h, w = gray.shape[:2]
    top = int(h * 0.2)
    bottom = int(h * 0.8)
    roi = gray[top:bottom, :]

    if roi.dtype != np.uint8:
        roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    inverted = 255 - roi
    col_sum = np.sum(inverted, axis=0)
    smooth = np.convolve(col_sum, np.ones(15) / 15, mode="same")

    start = int(w * 0.35)
    end = int(w * 0.65)
    x_split = start + int(np.argmin(smooth[start:end]))
    x_split = int(x_split + (offset_percent / 100.0) * w)
    x_split = max(int(w * 0.1), min(int(w * 0.9), x_split))
    padding = int(w * 0.03)
    left = gray[:, : max(1, x_split - padding)]
    right = gray[:, min(w, x_split + padding) :]
    return x_split, left, right


def _save_debug_image(base_dir: Path, prefix: str, image: np.ndarray):
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / f"{prefix}.png"
    if len(image.shape) == 2:
        cv2.imwrite(str(target), image)
    else:
        cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _clean_text_basic(text: str) -> str:
    cleaned = text.replace("\x0c", "\n")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"[\u200b\u00ad]", "", cleaned)
    return cleaned


def postprocess_text(text: str, lang_mode: str, is_dictionary_page: bool = True, preserve_spaces: bool = True) -> str:
    text = _clean_text_basic(text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    lines = [line.rstrip() for line in text.splitlines()]
    if not is_dictionary_page:
        joined = " ".join(line.strip() for line in lines)
    else:
        joined = "\n".join(lines)
    if preserve_spaces:
        joined = re.sub(r"\n{3,}", "\n\n", joined)
    else:
        joined = re.sub(r"[ ]{2,}", " ", joined)
        joined = re.sub(r"\s{3,}", "\n\n", joined)
    return joined.strip()


def _build_tesseract_config(psm: int, preserve_spaces: bool = True, dpi: int | None = 300) -> str:
    base = ["--oem", "1", "--psm", str(psm)]
    if preserve_spaces:
        base.extend(["-c", "preserve_interword_spaces=1"])
    if dpi:
        base.extend(["--dpi", str(dpi)])
    tessdata_dir = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
    tessdata_dir_short = to_short_path(tessdata_dir)
    base.extend(["--tessdata-dir", tessdata_dir_short])
    return " ".join(base)


def _tesseract_image_to_string(pil_img: Image.Image, lang: str, config: str) -> str:
    tmp_png = Path(tempfile.gettempdir()) / f"anky_ocr_{uuid4().hex}.png"
    try:
        pil_img.save(tmp_png, format="PNG")
        return pytesseract.image_to_string(str(tmp_png), lang=lang, config=config)
    finally:
        if tmp_png.exists():
            tmp_png.unlink()


def ocr_with_tesseract(image: np.ndarray, lang: str, psm: int = 4, preserve_spaces: bool = True, dpi: int | None = 300) -> str:
    config = _build_tesseract_config(psm, preserve_spaces=preserve_spaces, dpi=dpi)
    pil_img = Image.fromarray(image)
    return _tesseract_image_to_string(pil_img, lang=lang, config=config)


def _normalize_paddle_lang(lang_mode: str) -> str:
    if "rus" in lang_mode:
        return "ru"
    if "deu" in lang_mode:
        return "german"
    return "en"


def ocr_with_paddle(image: np.ndarray, lang_mode: str) -> str:
    if not PADDLE_AVAILABLE:
        raise RuntimeError(
            "PaddleOCR не установлен. Установите зависимости внутри venv:\n"
            "python -m pip install paddlepaddle paddleocr"
        )
    lang = _normalize_paddle_lang(lang_mode)
    ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
    result = ocr.ocr(image, cls=True)
    lines: list[tuple[tuple[int, int], str]] = []
    for page in result:
        for line in page:
            box, (text, _score) = line
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            lines.append(((min(xs), min(ys)), text))
    lines.sort(key=lambda item: (item[0][1], item[0][0]))
    return "\n".join(text for _pos, text in lines)


def _compose_diag(image_path: str, pil_img: Image.Image | None, config: str | None = None, lang: str | None = None, ocr_mode: str | None = None) -> str:
    diag_parts = [f"image_path: {image_path}"]
    exists = os.path.exists(image_path)
    size = os.path.getsize(image_path) if exists else 0
    diag_parts.append(f"exists/size bytes: {exists}/{size}")
    if pil_img is not None:
        diag_parts.append(f"dimensions: {getattr(pil_img, 'size', None)}")
    diag_parts.append(f"python: {platform.python_version()}")
    diag_parts.append(f"opencv: {cv2.__version__}")
    diag_parts.append(f"pillow: {Image.__version__}")
    diag_parts.append(f"pytesseract: {getattr(pytesseract, '__version__', 'unknown')}")
    diag_parts.append(f"tesseract config: {config}")
    diag_parts.append(f"lang: {lang}")
    diag_parts.append(f"ocr_mode: {ocr_mode}")
    diag_parts.append("Diag Tesseract:\n" + get_tesseract_diag())
    if pil_img is not None:
        diag_parts.append(_format_image_diag(image_path, pil_img))
    return "\n".join(diag_parts)


def perform_page_ocr(image_path: str, options: OcrRunOptions, progress_cb: ProgressCallback | None = None) -> str:
    total_steps = 6
    _report(progress_cb, 1, total_steps, "Шаг 1/6: загрузка")

    original_pil = load_image_for_ocr(image_path)
    bgr = load_image_any(image_path)
    debug_dir = Path("logs") / "ocr_debug" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if options.debug_images:
        _save_debug_image(debug_dir, "01_original", bgr)

    working = bgr
    warp_info = {"found_quad": False}
    if options.perspective_correction:
        working, warp_info = detect_and_warp_page(bgr)
        _report(progress_cb, 2, total_steps, "Шаг 2/6: перспектива" if warp_info.get("found_quad") else "Шаг 2/6: перспектива пропущена")
        if options.debug_images:
            _save_debug_image(debug_dir, "02_warped", working)
    else:
        _report(progress_cb, 2, total_steps, "Шаг 2/6: перспектива выключена")

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    if options.flatten_background:
        flat = flatten_background(gray)
        _report(progress_cb, 3, total_steps, "Шаг 3/6: фон выровнен")
    else:
        flat = gray
        _report(progress_cb, 3, total_steps, "Шаг 3/6: фон без изменений")
    if options.debug_images:
        _save_debug_image(debug_dir, "03_flatten", flat)

    binary = upscale_and_binarize(flat, mode=options.binarize_mode)
    _report(progress_cb, 4, total_steps, "Шаг 4/6: бинаризация")
    if options.deskew:
        binary = deskew(binary)
    if options.debug_images:
        _save_debug_image(debug_dir, "04_binary", binary)

    result_text = ""
    config = _build_tesseract_config(psm=options.psm, preserve_spaces=options.preserve_spaces, dpi=options.dpi)

    try:
        if options.ocr_mode == "two_columns":
            x_split, left_img, right_img = split_columns_auto(binary, options.split_offset_percent)
            if options.debug_images:
                _save_debug_image(debug_dir, "05_left", left_img)
                _save_debug_image(debug_dir, "06_right", right_img)
            _report(progress_cb, 5, total_steps, "Шаг 5/6: OCR левая колонка")
            left_engine_text: str
            right_engine_text: str
            if options.prefer_paddle_for_columns and PADDLE_AVAILABLE:
                left_engine_text = ocr_with_paddle(left_img, "deu")
                _report(progress_cb, 6, total_steps, "Шаг 6/6: OCR правая колонка")
                right_engine_text = ocr_with_paddle(right_img, "rus")
            else:
                left_engine_text = ocr_with_tesseract(left_img, "deu", psm=options.psm)
                _report(progress_cb, 6, total_steps, "Шаг 6/6: OCR правая колонка")
                right_engine_text = ocr_with_tesseract(right_img, "rus", psm=options.psm)
            combined = "\n\n".join(["--- DE ---", left_engine_text.strip(), "--- RU ---", right_engine_text.strip()])
            result_text = combined
        elif options.ocr_mode == "pro":
            _report(progress_cb, 5, total_steps, "Шаг 5/6: PaddleOCR")
            result_text = ocr_with_paddle(binary, options.lang_mode)
            _report(progress_cb, 6, total_steps, "Шаг 6/6: постобработка")
        else:
            _report(progress_cb, 5, total_steps, "Шаг 5/6: Tesseract")
            result_text = ocr_with_tesseract(binary, options.lang_mode, psm=options.psm, preserve_spaces=options.preserve_spaces, dpi=options.dpi)
            _report(progress_cb, 6, total_steps, "Шаг 6/6: постобработка")
    except Exception as e:  # noqa: BLE001
        diag = _compose_diag(image_path, original_pil, config=config, lang=options.lang_mode, ocr_mode=options.ocr_mode)
        raise RuntimeError(f"{e}\n\n{diag}\n\n{traceback.format_exc()}") from e

    return postprocess_text(result_text, options.lang_mode, is_dictionary_page=options.dictionary_mode, preserve_spaces=options.preserve_spaces)


# ---------------------------------------------------------------------------
# Legacy helper preserved for compatibility
# ---------------------------------------------------------------------------

def ocr_photo_document(image_path: str, lang: str, progress_cb: ProgressCallback | None = None) -> str:
    options = OcrRunOptions(
        ocr_mode="fast",
        lang_mode=lang,
        perspective_correction=True,
        flatten_background=True,
        binarize_mode="adaptive",
        deskew=True,
        debug_images=False,
        psm=4,
    )
    return perform_page_ocr(image_path, options, progress_cb)
