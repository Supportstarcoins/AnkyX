"""Helpers for configuring Tesseract/pytesseract without system env vars."""
from __future__ import annotations

import os
import subprocess
from shutil import which
from typing import List, Optional, Tuple

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


DEFAULT_TESSERACT_EXE = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
DEFAULT_TESSDATA_DIR = r"C:\\Program Files\\Tesseract-OCR\\tessdata"

_TESSERACT_EXE: Optional[str] = None
_TESSDATA_DIR: Optional[str] = None
_TESSDATA_PREFIX: Optional[str] = None


def _candidate_tesseract_paths() -> List[str]:
    candidates = [DEFAULT_TESSERACT_EXE, r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"]
    if pytesseract:
        cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
        if cmd and os.path.isabs(cmd):
            candidates.insert(0, cmd)
    detected = which("tesseract")
    if detected:
        candidates.append(detected)
    return candidates


def _candidate_tessdata_dirs() -> List[str]:
    return [DEFAULT_TESSDATA_DIR, r"C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"]


def find_tesseract_install() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Locate Tesseract executable and tessdata directory."""

    tesseract_exe = next((p for p in _candidate_tesseract_paths() if p and os.path.exists(p)), None)

    tessdata_dir = next((p for p in _candidate_tessdata_dirs() if p and os.path.isdir(p)), None)
    tessdata_prefix = os.path.dirname(tessdata_dir) if tessdata_dir else None

    return tesseract_exe, tessdata_dir, tessdata_prefix


def configure_pytesseract() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Configure pytesseract command and TESSDATA_PREFIX environment variable."""

    global _TESSERACT_EXE, _TESSDATA_DIR, _TESSDATA_PREFIX

    _TESSERACT_EXE, _TESSDATA_DIR, _TESSDATA_PREFIX = find_tesseract_install()

    if pytesseract:
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_EXE or DEFAULT_TESSERACT_EXE

    if _TESSDATA_PREFIX:
        os.environ["TESSDATA_PREFIX"] = _TESSDATA_PREFIX

    return _TESSERACT_EXE, _TESSDATA_DIR, _TESSDATA_PREFIX


def get_tesseract_cmd() -> Optional[str]:
    return _TESSERACT_EXE or DEFAULT_TESSERACT_EXE


def get_tessdata_dir() -> Optional[str]:
    return _TESSDATA_DIR or DEFAULT_TESSDATA_DIR


def build_tessdata_config(base_config: str | None = "") -> str:
    config = (base_config or "").strip()
    tessdata_dir = get_tessdata_dir()
    if tessdata_dir:
        extra = f'--tessdata-dir "{tessdata_dir}"'
        config = f"{config} {extra}".strip()
    return config


def ensure_languages(lang_codes: List[str]) -> Tuple[bool, List[str]]:
    if not lang_codes:
        return True, []
    tessdata_dir = get_tessdata_dir()
    if not tessdata_dir:
        return False, lang_codes
    missing = []
    for code in lang_codes:
        expected_file = os.path.join(tessdata_dir, f"{code}.traineddata")
        if not os.path.isfile(expected_file):
            missing.append(code)
    return len(missing) == 0, missing


def get_tesseract_diag() -> str:
    exe = _TESSERACT_EXE or "не найден"
    tessdata = _TESSDATA_DIR or "не найдено"
    lines = [f"tesseract.exe: {exe}", f"tessdata: {tessdata}"]

    langs_output = ""
    if _TESSERACT_EXE and os.path.exists(_TESSERACT_EXE):
        try:
            res = subprocess.run(
                [_TESSERACT_EXE, "--list-langs"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            langs_output = res.stdout.strip() or res.stderr.strip()
        except Exception as e:  # pragma: no cover - diagnostic only
            langs_output = f"Ошибка получения языков: {e}"
    if langs_output:
        lines.append("Доступные языки:")
        lines.append(langs_output)
    else:
        lines.append("Доступные языки: недоступно")
    return "\n".join(lines)


def is_tesseract_available() -> bool:
    if not pytesseract:
        return False
    if _TESSERACT_EXE:
        return os.path.exists(_TESSERACT_EXE)
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    if os.path.isabs(cmd):
        return os.path.exists(cmd)
    return which(cmd) is not None


# Configure immediately on import so other modules see the paths.
configure_pytesseract()
