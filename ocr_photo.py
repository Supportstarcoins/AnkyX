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
import tkinter as tk
from tkinter import messagebox

from tesseract_setup import get_tesseract_diag, to_short_path
from main import load_image_for_ocr, _format_image_diag

ProgressCallback = Callable[[int, int, str], None]

PADDLE_AVAILABLE = False
PADDLE_VERSION: str | None = None
PADDLEOCR_AVAILABLE = False
PADDLEOCR_VERSION: str | None = None

try:
    import paddle

    PADDLE_AVAILABLE = True
    PADDLE_VERSION = getattr(paddle, "__version__", None)
except Exception:
    PADDLE_AVAILABLE = False

try:
    import paddleocr
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
    PADDLEOCR_VERSION = getattr(paddleocr, "__version__", None)
except Exception:
    PaddleOCR = None  # type: ignore
    PADDLEOCR_AVAILABLE = False
    PADDLEOCR_VERSION = None


_PADDLE_OCR_CACHE: dict[str, PaddleOCR] = {}


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
    preprocess_preset: Literal["basic", "auto_pro", "none"] = "basic"


def _report(cb: ProgressCallback | None, step: int, total: int, label: str):
    if cb:
        cb(step, total, label)


INSTALL_PADDLE_CMD = r"C:\\AnkyX-main\\venv\\Scripts\\python.exe -m pip install paddleocr paddlepaddle"


def _show_message_async(title: str, message: str):
    try:
        root = tk._default_root  # type: ignore[attr-defined]
        if root is not None:
            root.after(0, lambda: messagebox.showerror(title, message))
        else:
            messagebox.showerror(title, message)
    except Exception:
        # GUI errors should never crash OCR flow
        pass


def _is_paddle_ready() -> bool:
    return bool(PADDLE_AVAILABLE and PADDLEOCR_AVAILABLE and PaddleOCR is not None)


def _notify_paddle_missing():
    _show_message_async(
        "PaddleOCR недоступен",
        "Установите зависимости внутри venv:\n" f"{INSTALL_PADDLE_CMD}",
    )


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

    src = binary_or_gray
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

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
    left = src[:, : max(1, x_split - padding)]
    right = src[:, min(w, x_split + padding) :]
    return x_split, left, right


def _save_debug_image(base_dir: Path, prefix: str, image: np.ndarray):
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / f"{prefix}.png"
    if len(image.shape) == 2:
        cv2.imwrite(str(target), image)
    else:
        cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _ensure_min_width(image: np.ndarray, min_width: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    if w >= min_width:
        return image
    scale = 2.0
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def _unsharp_mask(gray: np.ndarray, strength: float = 1.2, blur_size: int = 3) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), blur_size)
    return cv2.addWeighted(gray, strength, blurred, -0.2, 0)


def _apply_clahe_to_l(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def _flatten_background_pro(gray: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(opened, (0, 0), sigmaX=15, sigmaY=15)
    normalized = cv2.divide(gray, blurred, scale=255)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        10,
    )


def _sauvola_threshold(gray: np.ndarray, window: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    gray_f = gray.astype(np.float64)
    padded = cv2.copyMakeBorder(gray_f, window // 2, window // 2, window // 2, window // 2, cv2.BORDER_REFLECT)
    integral = cv2.integral(padded)
    integral_sq = cv2.integral(padded * padded)

    h, w = gray.shape
    mean = np.zeros_like(gray_f)
    std = np.zeros_like(gray_f)

    for y in range(h):
        y1 = y
        y2 = y + window
        for x in range(w):
            x1 = x
            x2 = x + window
            area = window * window
            s = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
            sq = integral_sq[y2, x2] - integral_sq[y1, x2] - integral_sq[y2, x1] + integral_sq[y1, x1]
            m = s / area
            mean[y, x] = m
            std[y, x] = max(0.0, np.sqrt(max(sq / area - m * m, 0.0)))

    threshold = mean * (1 + k * ((std / r) - 1))
    binary = (gray_f > threshold).astype(np.uint8) * 255
    return binary.astype(np.uint8)


def _ensure_black_text_on_white(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_uint8 = image
    mean_val = float(np.mean(img_uint8))
    if mean_val < 127:
        return 255 - img_uint8
    return img_uint8


def _compute_text_stats(text: str) -> tuple[float, float]:
    total_chars = len([ch for ch in text if not ch.isspace()])
    if total_chars == 0:
        return 0.0, 0.0
    alpha_chars = len(re.findall(r"[A-Za-z\u00C0-\u024f\u0400-\u04FF]", text))
    allowed_punct = set(",.;:!?()[]{}\"'`´-–—/_\\|")
    bad_chars = 0
    for ch in text:
        if ch.isspace():
            continue
        if ch.isdigit():
            continue
        if re.match(r"[A-Za-z\u00C0-\u024f\u0400-\u04FF]", ch):
            continue
        if ch in allowed_punct:
            continue
        bad_chars += 1
    alpha_ratio = alpha_chars / total_chars
    bad_ratio = bad_chars / total_chars
    return alpha_ratio, bad_ratio


def _evaluate_variant(
    image: np.ndarray,
    lang: str,
    psm: int,
    preserve_spaces: bool,
) -> tuple[float, float, float]:
    config = _build_tesseract_config(psm, preserve_spaces=preserve_spaces, dpi=300)
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    reconstructed = _reconstruct_text_from_data(data, preserve_spaces=preserve_spaces)
    confs = [float(c) for c in data.get("conf", []) if isinstance(c, (int, float, str)) and float(c) > 0]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    alpha_ratio, bad_ratio = _compute_text_stats(reconstructed)
    score = avg_conf + 30 * alpha_ratio - 30 * bad_ratio
    return avg_conf, alpha_ratio, score


def _build_auto_variants(
    bgr: np.ndarray,
    debug_dir: Path,
    prefix: str,
) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variants["A"] = _unsharp_mask(gray)

    clahe_img = _apply_clahe_to_l(bgr)
    clahe_gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY) if len(clahe_img.shape) == 3 else clahe_img
    variants["B"] = clahe_gray

    flattened = _flatten_background_pro(gray)
    variants["C"] = flattened

    adaptive_base = flattened if flattened is not None else clahe_gray
    variants["D"] = _ensure_black_text_on_white(_adaptive_threshold(adaptive_base))

    sauvola_base = flattened if flattened is not None else gray
    variants["E"] = _ensure_black_text_on_white(_sauvola_threshold(sauvola_base))

    for key, img in variants.items():
        _save_debug_image(debug_dir, f"{prefix}{key}_", img)

    return variants


def _select_best_variant(
    bgr: np.ndarray,
    lang: str,
    debug_dir: Path,
    prefix: str,
    allowed_psms: tuple[int, ...],
    preserve_spaces: bool,
) -> tuple[np.ndarray, str, float, float, int]:
    variants = _build_auto_variants(bgr, debug_dir, prefix)
    best_key = "A"
    best_score = -1e9
    best_img = variants["A"]
    best_avg_conf = 0.0
    best_psm = allowed_psms[0]

    for key, img in variants.items():
        for psm in allowed_psms:
            avg_conf, alpha_ratio, score = _evaluate_variant(img, lang=lang, psm=psm, preserve_spaces=preserve_spaces)
            if score > best_score:
                best_score = score
                best_key = key
                best_img = img
                best_avg_conf = avg_conf
                best_psm = psm

    return best_img, best_key, best_score, best_avg_conf, best_psm


def _clean_text_basic(text: str) -> str:
    cleaned = text.replace("\x0c", "\n")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"[\u200b\u00ad]", "", cleaned)
    return cleaned


def _reconstruct_text_from_data(data: dict, preserve_spaces: bool = True) -> str:
    required_fields = ["block_num", "par_num", "line_num", "word_num", "left", "width", "text"]
    if not all(field in data for field in required_fields):
        return ""

    entries = []
    count = len(data.get("text", []))
    for idx in range(count):
        raw_text = data.get("text", [""])[idx]
        if not isinstance(raw_text, str):
            continue
        text = raw_text.strip()
        if not text:
            continue
        try:
            entries.append(
                {
                    "block": int(data.get("block_num", [0])[idx]),
                    "par": int(data.get("par_num", [0])[idx]),
                    "line": int(data.get("line_num", [0])[idx]),
                    "word": int(data.get("word_num", [0])[idx]),
                    "left": int(data.get("left", [0])[idx]),
                    "width": int(data.get("width", [0])[idx]),
                    "text": text,
                }
            )
        except Exception:
            continue

    lines: dict[tuple[int, int, int], list[dict]] = {}
    for entry in entries:
        key = (entry["block"], entry["par"], entry["line"])
        lines.setdefault(key, []).append(entry)

    reconstructed_lines: list[str] = []
    last_paragraph: tuple[int, int] | None = None
    for key in sorted(lines.keys()):
        block_par = (key[0], key[1])
        if last_paragraph is not None and block_par != last_paragraph:
            reconstructed_lines.append("")
        last_paragraph = block_par

        words = sorted(lines[key], key=lambda e: (e["left"], e["word"]))
        widths = [max(1, w["width"]) / max(1, len(w["text"])) for w in words]
        avg_char_width = float(np.median(widths)) if widths else 1.0
        gap_threshold = avg_char_width * 0.6

        line_parts: list[str] = []
        prev_right = None
        for word in words:
            if prev_right is not None:
                gap = word["left"] - prev_right
                spaces = 0
                if gap > gap_threshold:
                    spaces = max(1, int(round(gap / max(avg_char_width, 1.0))))
                elif preserve_spaces:
                    spaces = 1
                if spaces:
                    line_parts.append(" " * min(spaces, 10))
            line_parts.append(word["text"])
            prev_right = word["left"] + word["width"]

        reconstructed_line = "".join(line_parts).rstrip()
        reconstructed_lines.append(reconstructed_line)

    return "\n".join(line for line in reconstructed_lines if preserve_spaces or line.strip())


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


def _tesseract_image_to_data(pil_img: Image.Image, lang: str, config: str) -> dict:
    tmp_png = Path(tempfile.gettempdir()) / f"anky_ocr_{uuid4().hex}.png"
    try:
        pil_img.save(tmp_png, format="PNG")
        return pytesseract.image_to_data(str(tmp_png), lang=lang, config=config, output_type=pytesseract.Output.DICT)
    finally:
        if tmp_png.exists():
            tmp_png.unlink()


def ocr_with_tesseract(image: np.ndarray, lang: str, psm: int = 4, preserve_spaces: bool = True, dpi: int | None = 300) -> str:
    def _run_single(psm_value: int) -> tuple[str, float, float]:
        config = _build_tesseract_config(psm_value, preserve_spaces=preserve_spaces, dpi=dpi)
        pil_img = Image.fromarray(image)
        data = _tesseract_image_to_data(pil_img, lang=lang, config=config)
        text = _reconstruct_text_from_data(data, preserve_spaces=preserve_spaces)
        confs = [float(c) for c in data.get("conf", []) if isinstance(c, (int, float, str)) and float(c) > 0]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        alpha_ratio, bad_ratio = _compute_text_stats(text)
        score = avg_conf + 30 * alpha_ratio - 30 * bad_ratio
        return text, score, avg_conf

    primary_text, primary_score, primary_conf = _run_single(psm)
    if len(primary_text.strip()) < 30 and psm in (4, 6):
        alt_psm = 6 if psm == 4 else 4
        alt_text, alt_score, alt_conf = _run_single(alt_psm)
        if alt_score > primary_score or len(alt_text.strip()) > len(primary_text.strip()):
            primary_text, primary_score, primary_conf = alt_text, alt_score, alt_conf

    return primary_text


def _normalize_paddle_lang(lang_mode: str) -> tuple[str, str]:
    if "rus" in lang_mode:
        return "ru", "en"
    if "deu" in lang_mode:
        return "german", "en"
    return "en", "en"


def get_paddle_ocr(lang: str) -> PaddleOCR:
    if lang in _PADDLE_OCR_CACHE:
        return _PADDLE_OCR_CACHE[lang]

    def _create(with_show_log: bool):
        kwargs = {"lang": lang, "use_angle_cls": True}
        if with_show_log:
            kwargs["show_log"] = False
        return PaddleOCR(**kwargs)

    try:
        ocr = _create(with_show_log=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "Unknown argument: show_log" in msg or (
            isinstance(e, TypeError) and "show_log" in msg
        ):
            ocr = _create(with_show_log=False)
        else:
            raise

    _PADDLE_OCR_CACHE[lang] = ocr
    return ocr


def _extract_sorted_text(result) -> str:
    lines: list[tuple[tuple[float, float], str]] = []
    for page in result:
        for line in page:
            box, (text, _score) = line
            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]
            lines.append(((min(ys), min(xs)), text))
    lines.sort(key=lambda item: (item[0][0], item[0][1]))
    return "\n".join(text for _pos, text in lines)


def _run_paddle_single_pass(image: np.ndarray, lang: str, fallback_lang: str | None = None) -> str:
    try:
        ocr = get_paddle_ocr(lang)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if fallback_lang and ("lang" in msg or "language" in msg or "support" in msg):
            ocr = get_paddle_ocr(fallback_lang)
        else:
            raise
    result = ocr.ocr(image, cls=True)
    return _extract_sorted_text(result)


def ocr_with_paddle(image: np.ndarray, lang_mode: str) -> str:
    if not _is_paddle_ready():
        _notify_paddle_missing()
        raise RuntimeError(
            "PaddleOCR недоступен. Проверьте установку paddlepaddle и paddleocr."
        )

    lang_mode_clean = lang_mode.strip().lower()
    langs = [lang.strip().lower() for lang in lang_mode_clean.split("+") if lang.strip()]
    has_deu = "deu" in langs
    has_rus = "rus" in langs

    try:
        if has_deu and has_rus:
            german_lang, german_fallback = _normalize_paddle_lang("deu")
            ru_lang, ru_fallback = _normalize_paddle_lang("rus")
            ru_text = _run_paddle_single_pass(image, ru_lang, ru_fallback)
            de_text = _run_paddle_single_pass(image, german_lang, german_fallback)
            return "\n\n".join(["--- RU ---", ru_text.strip(), "--- DE ---", de_text.strip()])

        target_lang, fallback_lang = _normalize_paddle_lang(lang_mode_clean or "en")
        return _run_paddle_single_pass(image, target_lang, fallback_lang)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"PaddleOCR ошибка: {e}") from e


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
    diag_parts.append(
        f"paddle_available: {PADDLE_AVAILABLE}, paddle_version: {PADDLE_VERSION}"
    )
    diag_parts.append(
        f"paddleocr_available: {PADDLEOCR_AVAILABLE}, paddleocr_version: {PADDLEOCR_VERSION}"
    )
    diag_parts.append(f"pytesseract: {getattr(pytesseract, '__version__', 'unknown')}")
    diag_parts.append(f"tesseract config: {config}")
    diag_parts.append(f"lang: {lang}")
    diag_parts.append(f"ocr_mode: {ocr_mode}")
    diag_parts.append("Diag Tesseract:\n" + get_tesseract_diag())
    if pil_img is not None:
        diag_parts.append(_format_image_diag(image_path, pil_img))
    return "\n".join(diag_parts)


def perform_page_ocr(image_path: str, options: OcrRunOptions, progress_cb: ProgressCallback | None = None) -> str:
    total_steps = 9 if options.ocr_mode == "two_columns" else 8
    current_step = 1

    def _step(label: str):
        nonlocal current_step
        _report(progress_cb, current_step, total_steps, f"Шаг {current_step}/{total_steps}: {label}")
        current_step += 1

    _step("загрузка")

    original_pil: Image.Image | None = None
    bgr: np.ndarray | None = None
    gray: np.ndarray | None = None
    binary: np.ndarray | None = None
    result_text = ""
    config = _build_tesseract_config(psm=options.psm, preserve_spaces=options.preserve_spaces, dpi=options.dpi)

    try:
        original_pil = load_image_for_ocr(image_path)
        bgr = load_image_any(image_path)
        debug_dir = Path("logs") / "ocr_debug" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if bgr is not None:
            _save_debug_image(debug_dir, "01_original", bgr)

        working = bgr if bgr is not None else np.array(original_pil)
        warp_info = {"found_quad": False}
        if options.perspective_correction and bgr is not None:
            working, warp_info = detect_and_warp_page(bgr)
            _step("перспектива" if warp_info.get("found_quad") else "перспектива пропущена")
            _save_debug_image(debug_dir, "02_warped", working)
        else:
            _step("перспектива выключена")

        working = _ensure_min_width(working)
        selected_page = working

        if options.preprocess_preset != "auto_pro":
            _step("базовая подготовка")
            gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

            if options.flatten_background:
                flat = flatten_background(gray)
                _step("фон выровнен")
            else:
                flat = gray
                _step("фон без изменений")
            _save_debug_image(debug_dir, "03_flatten", flat)

            binary = upscale_and_binarize(flat, mode=options.binarize_mode)
            binary = _ensure_black_text_on_white(binary)
            _step("бинаризация")
            if options.deskew:
                binary = deskew(binary)
            _save_debug_image(debug_dir, "04_binary", binary)
            selected_page = binary if binary is not None else gray if gray is not None else working

        if options.ocr_mode == "two_columns":
            _step("разделение колонки")
            split_source = selected_page if options.preprocess_preset != "auto_pro" else working
            x_split, left_img, right_img = split_columns_auto(split_source, options.split_offset_percent)
            _save_debug_image(debug_dir, "04_split_left", left_img)
            _save_debug_image(debug_dir, "05_split_right", right_img)
            column_psms: tuple[int, ...] = (6, 4)
            column_psm = column_psms[0]

            if options.preprocess_preset == "auto_pro":
                left_processed, left_key, left_score, left_conf, left_psm = _select_best_variant(
                    left_img,
                    lang="deu",
                    debug_dir=debug_dir,
                    prefix="left_",
                    allowed_psms=column_psms,
                    preserve_spaces=options.preserve_spaces,
                )
                left_processed = _ensure_black_text_on_white(left_processed)
                if options.deskew:
                    left_processed = deskew(left_processed)
                _step(
                    "Column L: selected preset "
                    f"{left_key}, selected psm {left_psm}, score={left_score:.2f}, conf={left_conf:.1f}"
                )
                _save_debug_image(debug_dir, "06_left_best", left_processed)

                right_processed, right_key, right_score, right_conf, right_psm = _select_best_variant(
                    right_img,
                    lang="rus",
                    debug_dir=debug_dir,
                    prefix="right_",
                    allowed_psms=column_psms,
                    preserve_spaces=options.preserve_spaces,
                )
                right_processed = _ensure_black_text_on_white(right_processed)
                if options.deskew:
                    right_processed = deskew(right_processed)
                _step(
                    "Column R: selected preset "
                    f"{right_key}, selected psm {right_psm}, score={right_score:.2f}, conf={right_conf:.1f}"
                )
                _save_debug_image(debug_dir, "07_right_best", right_processed)
            else:
                left_processed = left_img
                right_processed = right_img
                left_psm = column_psm
                right_psm = column_psm

            left_engine_text = ocr_with_tesseract(left_processed, "deu", psm=left_psm, preserve_spaces=options.preserve_spaces, dpi=options.dpi)
            right_engine_text = ocr_with_tesseract(right_processed, "rus", psm=right_psm, preserve_spaces=options.preserve_spaces, dpi=options.dpi)
            combined = "\n\n".join(["--- DE ---", left_engine_text.strip(), "--- RU ---", right_engine_text.strip()])
            result_text = combined
            _step("постобработка")
        elif options.preprocess_preset == "auto_pro":
            _step("авто-подготовка")
            page_psms: tuple[int, ...] = (6, 4)
            best_img, best_key, best_score, best_avg_conf, best_psm = _select_best_variant(
                working,
                lang=options.lang_mode,
                debug_dir=debug_dir,
                prefix="page_",
                allowed_psms=page_psms,
                preserve_spaces=options.preserve_spaces,
            )
            best_img = _ensure_black_text_on_white(best_img)
            if options.deskew:
                best_img = deskew(best_img)
            _save_debug_image(debug_dir, "03_best", best_img)
            _step(
                f"Выбранный пресет: {best_key}, выбранный psm: {best_psm}, "
                f"score={best_score:.2f}, avg_conf={best_avg_conf:.1f}"
            )
            selected_page = best_img

            if options.ocr_mode == "pro":
                _step("PaddleOCR")
                result_text = ocr_with_paddle(selected_page, options.lang_mode)
                _step("постобработка")
            else:
                _step("Tesseract")
                result_text = ocr_with_tesseract(selected_page, options.lang_mode, psm=best_psm, preserve_spaces=options.preserve_spaces, dpi=options.dpi)
                _step("постобработка")
            _step("завершено")
        elif options.ocr_mode == "pro" and selected_page is not None:
            _step("PaddleOCR")
            result_text = ocr_with_paddle(selected_page, options.lang_mode)
            _step("постобработка")
            _step("завершено")
        else:
            page_psm = options.psm if options.psm in (3, 4, 6) else 4
            target_img = selected_page
            _step("Tesseract")
            result_text = ocr_with_tesseract(target_img, options.lang_mode, psm=page_psm, preserve_spaces=options.preserve_spaces, dpi=options.dpi)
            _step("постобработка")
            _step("завершено")
    except Exception as e:  # noqa: BLE001
        diag = _compose_diag(image_path, original_pil, config=config, lang=options.lang_mode, ocr_mode=options.ocr_mode)
        _show_message_async("Ошибка OCR", f"{e}\n\nПереключаюсь на Tesseract OCR.\n\n{diag}")
        fallback_source = binary if binary is not None else gray if gray is not None else bgr
        if fallback_source is None and original_pil is not None:
            try:
                fallback_source = cv2.cvtColor(np.array(original_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                fallback_source = np.array(original_pil)
        try:
            if fallback_source is None:
                raise RuntimeError("Нет изображения для резервного OCR")
            result_text = ocr_with_tesseract(
                fallback_source,
                options.lang_mode,
                psm=options.psm,
                preserve_spaces=options.preserve_spaces,
                dpi=options.dpi,
            )
        except Exception as inner:  # noqa: BLE001
            _show_message_async("Ошибка Tesseract", f"{inner}\n\n{diag}")
            result_text = f"{e}\n\n{diag}\n\nTesseract fallback error: {inner}"

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
