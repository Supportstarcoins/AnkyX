import argparse
import json
import logging
import os
import sys
import traceback
import faulthandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRASH_LOG_PATH = os.path.join(BASE_DIR, "ocr_pro_crash.log")
FAULT_LOG_PATH = os.path.join(BASE_DIR, "ocr_pro_fault.log")

_FAULT_LOG_HANDLE = None


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ocr-mode", default="pro")
    parser.add_argument("--lang-mode", default="deu+rus")
    parser.add_argument("--binarize", default="adaptive")
    parser.add_argument("--psm", default="4")
    parser.add_argument("--split-offset", default="0")
    parser.add_argument("--preprocess-preset", default="auto_pro")
    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument("--dictionary-mode", action="store_true")
    return parser


def main() -> int:
    configure_logging()
    configure_faulthandler()
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    parser = build_parser()
    args = parser.parse_args()

    try:
        import ocr_photo
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logging.error("Failed to import ocr_photo:\n%s", tb)
        emit_payload({"ok": False, "error": f"Ошибка импорта ocr_photo: {exc}", "trace": tb})
        return 1

    def progress_cb(step: int, total: int, label: str) -> None:
        emit_payload({"event": "progress", "step": step, "total": total, "label": label})

    try:
        options = ocr_photo.OcrRunOptions(
            ocr_mode=str(args.ocr_mode or "pro"),
            lang_mode=str(args.lang_mode or "deu+rus"),
            binarize_mode=str(args.binarize or "adaptive"),
            psm=int(args.psm or 4),
            dictionary_mode=bool(args.dictionary_mode),
            split_offset_percent=float(args.split_offset or 0),
            debug_images=bool(args.debug_images),
            preprocess_preset=str(args.preprocess_preset or "auto_pro"),
            preserve_spaces=True,
            prefer_paddle_for_columns=True,
        )
        logging.info(
            "OCR PRO worker started image=%s mode=%s lang=%s",
            args.image,
            options.ocr_mode,
            options.lang_mode,
        )
        text = ocr_photo.perform_page_ocr(args.image, options, progress_cb)
        emit_payload({"ok": True, "text": text})
        return 0
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logging.error("OCR PRO worker failed:\n%s", tb)
        emit_payload({"ok": False, "error": f"OCR PRO ошибка: {exc}", "trace": tb})
        return 1


if __name__ == "__main__":
    sys.exit(main())
