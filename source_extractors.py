from __future__ import annotations

import os
import re
from html.parser import HTMLParser


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _extract_html(path: str) -> str:
    parser = _HTMLStripper()
    parser.feed(_read_text_file(path))
    return "\n".join(x.strip() for x in parser.parts if x.strip())


def extract_text_from_path(path: str) -> str:
    if not path:
        raise RuntimeError("Путь к источнику не задан")
    if not os.path.exists(path):
        raise RuntimeError(f"Файл не найден: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        return _read_text_file(path)
    if ext in {".html", ".htm"}:
        return _extract_html(path)
    if ext == ".docx":
        try:
            import docx
        except Exception as exc:
            raise RuntimeError("python-docx не установлен для чтения DOCX") from exc
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == ".odt":
        try:
            from odf import text as odf_text
            from odf.opendocument import load
        except Exception as exc:
            raise RuntimeError("odfpy не установлен для чтения ODT") from exc
        doc = load(path)
        return "\n".join(node.firstChild.data for node in doc.getElementsByType(odf_text.P) if getattr(node, "firstChild", None))
    if ext == ".pdf":
        try:
            import fitz

            out: list[str] = []
            with fitz.open(path) as doc:
                for page in doc:
                    out.append(page.get_text())
            return "\n".join(out)
        except Exception:
            try:
                from pypdf import PdfReader
            except Exception as exc:
                raise RuntimeError("Для PDF нужен PyMuPDF (fitz) или pypdf") from exc
            reader = PdfReader(path)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        raise RuntimeError("OCR adapter placeholder: распознавание изображения пока не подключено")
    if ext in {".mp3", ".wav", ".m4a", ".ogg"}:
        raise RuntimeError("STT adapter placeholder: распознавание аудио пока не подключено")
    if ext in {".mp4", ".mkv", ".mov", ".avi", ".webm"}:
        raise RuntimeError("Video adapter placeholder: извлечение текста из видео пока не подключено")

    raise RuntimeError(f"Неподдерживаемый тип файла: {ext or 'без расширения'}")


def clean_extracted_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
