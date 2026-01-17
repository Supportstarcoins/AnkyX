import argparse
import json
import logging
import os
import sys
import tempfile
import traceback
import uuid
import importlib.util
import faulthandler
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_LOG_PATH = os.path.join(BASE_DIR, "ocr_crash.log")
FAULT_LOG_PATH = os.path.join(BASE_DIR, "ocr_fault.log")

_FAULT_LOG_HANDLE = None


PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None
OCR_PHOTO_AVAILABLE = importlib.util.find_spec("ocr_photo") is not None

if PIL_AVAILABLE:
    from PIL import Image, ImageOps, ImageEnhance

if NUMPY_AVAILABLE:
    import numpy as np

if CV2_AVAILABLE:
    import cv2  # type: ignore

if OCR_PHOTO_AVAILABLE:
    import ocr_photo


def configure_logging() -> None:
    handler = logging.FileHandler(CRASH_LOG_PATH, encoding="utf-8", mode="a")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    root_logger.propagate = False


def configure_faulthandler() -> None:
    global _FAULT_LOG_HANDLE
    if _FAULT_LOG_HANDLE is None:
        _FAULT_LOG_HANDLE = open(FAULT_LOG_PATH, "a", buffering=1, encoding="utf-8")
        faulthandler.enable(file=_FAULT_LOG_HANDLE, all_threads=True)


def emit_payload(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def build_error(message: str, trace: str | None = None) -> dict:
    payload = {"ok": False, "error": message}
    if trace:
        payload["trace"] = trace
    return payload


def validate_input(in_path: str) -> dict | None:
    if not os.path.exists(in_path):
        return build_error(f"Файл не найден: {in_path}")
    if CV2_AVAILABLE:
        cv_img = cv2.imread(in_path)
        if cv_img is None:
            return build_error("cv2.imread вернул None (битый файл/путь/кодек)")
    if not PIL_AVAILABLE:
        return build_error("Pillow не установлен (PIL недоступен)")
    return None


def run_pipeline(in_path: str, out_path: str, preset: str, binarize: str, psm: str) -> dict:
    error_payload = validate_input(in_path)
    if error_payload:
        return error_payload

    logging.info(
        "Postprocess worker started preset=%s binarize=%s psm=%s in=%s out=%s",
        preset,
        binarize,
        psm,
        in_path,
        out_path,
    )

    try:
        img = Image.open(in_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        use_cv = bool(CV2_AVAILABLE and NUMPY_AVAILABLE)

        def _pil_to_bgr(pil_img):
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        def _bgr_to_pil(bgr_img):
            return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

        def _save_step(current_img, step_index):
            tmp_path = Path(tempfile.gettempdir()) / f"anky_postprocess_{uuid.uuid4().hex}_{step_index}.png"
            current_img.save(tmp_path, format="PNG")
            return str(tmp_path)

        steps = [
            ("Ч/б (binarize)", "binarize"),
            ("Улучшение качества", "enhance"),
            ("Выравнивание перспективы", "perspective"),
            ("Убрать тени / выровнять фон", "shadow"),
            ("Выравнивать наклон (deskew)", "deskew"),
        ]

        step_path = in_path
        for idx, (label_text, stage) in enumerate(steps, start=1):
            logging.info("Postprocess step %s: %s", idx, stage)
            if stage == "binarize":
                if use_cv:
                    bgr = _pil_to_bgr(img)
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    binary = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
                    )
                    bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    img = _bgr_to_pil(bgr)
                else:
                    gray = ImageOps.grayscale(img)
                    bw = gray.point(lambda x: 255 if x > 150 else 0, mode="1")
                    img = bw.convert("RGB")
            elif stage == "enhance":
                if use_cv:
                    bgr = _pil_to_bgr(img)
                    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
                    img = _bgr_to_pil(bgr)
                img = ImageEnhance.Contrast(img).enhance(1.4)
                img = ImageEnhance.Sharpness(img).enhance(1.6)
            elif stage == "perspective":
                if use_cv and OCR_PHOTO_AVAILABLE:
                    detect_and_warp_page = getattr(ocr_photo, "detect_and_warp_page", None)
                    if callable(detect_and_warp_page):
                        bgr = _pil_to_bgr(img)
                        warped, _ = detect_and_warp_page(bgr)
                        img = _bgr_to_pil(warped)
            elif stage == "shadow":
                if use_cv and OCR_PHOTO_AVAILABLE:
                    flatten_background = getattr(ocr_photo, "flatten_background", None)
                    if callable(flatten_background):
                        bgr = _pil_to_bgr(img)
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        flat = flatten_background(gray)
                        bgr = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
                        img = _bgr_to_pil(bgr)
                else:
                    img = ImageOps.equalize(img)
            elif stage == "deskew":
                if use_cv and OCR_PHOTO_AVAILABLE:
                    deskew_fn = getattr(ocr_photo, "deskew", None)
                    if callable(deskew_fn):
                        bgr = _pil_to_bgr(img)
                        deskewed = deskew_fn(bgr)
                        img = _bgr_to_pil(deskewed)
                else:
                    img = ImageOps.autocontrast(img)

            label = f"Этап {idx}/{len(steps)}: {label_text}"
            step_path = _save_step(img, idx)
            emit_payload(
                {
                    "event": "progress",
                    "step": idx,
                    "total": len(steps),
                    "label": label,
                    "path": step_path,
                }
            )

        final_path = Path(out_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(final_path, format="PNG")
        logging.info("Postprocess worker finished: %s", out_path)
        return {"ok": True, "out_path": str(final_path)}
    except Exception:
        tb = traceback.format_exc()
        logging.error("Postprocess worker exception:\n%s", tb)
        return build_error("Ошибка во время постобработки", tb)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--preset", default="auto_pro")
    parser.add_argument("--binarize", default="adaptive")
    parser.add_argument("--psm", default="4")
    return parser


def main() -> int:
    configure_logging()
    configure_faulthandler()
    parser = build_parser()
    args = parser.parse_args()
    result = run_pipeline(args.in_path, args.out_path, args.preset, args.binarize, args.psm)
    emit_payload(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
# PATCH: crash-proof postprocess via subprocess + faulthandler + global hooks + always-on logs
