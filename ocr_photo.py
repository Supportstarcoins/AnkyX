"""Advanced OCR preprocessing for document photos (DE + RU).

The pipeline focuses on handling shadows, uneven lighting, and perspective
distortions before sending the image to Tesseract. It is tuned for mixed
German/Russian documents captured on mobile cameras with shadows or skew.
"""

from __future__ import annotations

import os
import re
from typing import Callable

import cv2
import numpy as np
from PIL import Image
import pytesseract

from tesseract_setup import ensure_languages, to_short_path

ProgressCallback = Callable[[int, int, str], None]


def _report(cb: ProgressCallback | None, step: int, total: int, label: str):
    if cb:
        cb(step, total, label)


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image, pts):
    rect = _order_points(pts)
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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def _clean_text(text: str) -> str:
    cleaned = text.replace("\x0c", "\n")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"[\u200b\u00ad]", "", cleaned)
    cleaned = re.sub(r"[\t\f]+", " ", cleaned)
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_noisy(text: str) -> bool:
    sample = text.strip()
    if not sample:
        return True
    junk = sum(1 for ch in sample if not re.match(r"[\w\s.,;:!?'""\-–—()\[\]/ÄÖÜäöüẞßЁёА-Яа-я]", ch))
    return junk / max(len(sample), 1) > 0.25


def _resize_to_limit(image, min_side: int = 1200, max_side: int = 2000):
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
    elif min(h, w) < min_side:
        scale = min_side / min(h, w)
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image


def ocr_photo_document(image_path: str, lang: str, progress_cb: ProgressCallback | None = None) -> str:
    """Run OCR on a document photo with perspective/lighting normalization."""

    total_steps = 5
    _report(progress_cb, 0, total_steps, "Загрузка изображения")

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Не удалось прочитать изображение.")

    img = _resize_to_limit(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Шаг 1: детекция контура документа
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
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
    _report(progress_cb, 1, total_steps, "Поиск границ документа")

    # Шаг 2: выпрямление перспективы (если найдено)
    processed = img
    if quad is not None:
        processed = _four_point_transform(img, quad)
    _report(progress_cb, 2, total_steps, "Выравнивание перспективы" if quad is not None else "Пропуск выравнивания")

    # Шаг 3: нормализация освещения и контраста
    gray_doc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray_doc, h=15, templateWindowSize=7, searchWindowSize=21)
    bg = cv2.medianBlur(denoised, 35)
    norm = cv2.divide(denoised, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    _report(progress_cb, 3, total_steps, "Нормализация освещения")

    pil_img = Image.fromarray(binary)

    # Шаг 4: OCR
    tessdata_dir = r"C:\\Program Files\\Tesseract-OCR\\tessdata"
    tessdata_dir_short = to_short_path(tessdata_dir)
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    config_primary = f"--oem 1 --psm 6 --tessdata-dir {tessdata_dir_short}"
    config_secondary = f"--oem 1 --psm 4 --tessdata-dir {tessdata_dir_short}"
    print(f"[OCR] tessdata_dir_short={tessdata_dir_short}")
    print(f"[OCR] config_primary={repr(config_primary)}")
    print(f"[OCR] config_secondary={repr(config_secondary)}")

    if "deu" in lang and "rus" in lang:
        ok, missing = ensure_languages(["deu", "rus"])
        if not ok:
            missing_files = ", ".join(f"{code}.traineddata" for code in missing)
            raise RuntimeError(
                "Не найдены файлы языков для OCR. "
                f"Ожидаются: {missing_files}."
            )
    text = pytesseract.image_to_string(pil_img, lang=lang, config=config_primary)
    _report(progress_cb, 4, total_steps, "OCR (основной проход)")

    if _is_noisy(text):
        text = pytesseract.image_to_string(pil_img, lang=lang, config=config_secondary)

    # Шаг 5: постобработка
    cleaned = _clean_text(text)
    _report(progress_cb, 5, total_steps, "Очистка результата")
    return cleaned

