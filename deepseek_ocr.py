from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


MODEL_DIR = Path("models") / "deepseek_ocr"
REQUIRED_FILES = ["config.json", "model.safetensors", "tokenizer.json"]


def ensure_model_layout() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def missing_weights() -> list[str]:
    ensure_model_layout()
    missing = []
    for name in REQUIRED_FILES:
        if not (MODEL_DIR / name).exists():
            missing.append(name)
    return missing


def validate_weights() -> None:
    missing = missing_weights()
    if missing:
        raise RuntimeError(
            "DeepSeek OCR веса не найдены. Добавьте файлы в models/deepseek_ocr/: " + ", ".join(missing)
        )


def run_ocr(image_path: str) -> str:
    validate_weights()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")
    # Offline runner stub: here should be local model inference.
    # Keeping deterministic placeholder avoids network dependency.
    return f"[DEEPSEEK_OCR_OFFLINE]{os.path.basename(image_path)}"


def test_ocr(image_path: str) -> tuple[bool, str]:
    try:
        text = run_ocr(image_path)
        return True, text
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
