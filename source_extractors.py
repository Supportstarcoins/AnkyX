from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from urllib.parse import urljoin


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.images: list[dict] = []
        self._text_window: list[str] = []
        self._position = 0

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)
            cleaned = data.strip()
            if cleaned:
                self._position += 1
                self._text_window.append(cleaned)
                if len(self._text_window) > 4:
                    self._text_window = self._text_window[-4:]

    def handle_starttag(self, tag: str, attrs) -> None:
        if (tag or "").lower() != "img":
            return
        attr_map = {str(k).lower(): str(v) for k, v in (attrs or []) if k}
        src = attr_map.get("src", "").strip()
        if not src:
            return
        width = _to_int_or_none(attr_map.get("width"))
        height = _to_int_or_none(attr_map.get("height"))
        self.images.append(
            {
                "url": src if src.startswith(("http://", "https://")) else "",
                "local_path": src if not src.startswith(("http://", "https://")) else "",
                "alt": attr_map.get("alt", "").strip(),
                "caption": attr_map.get("title", "").strip(),
                "context_text": " ".join(self._text_window[-2:]).strip(),
                "position": self._position,
                "width": width,
                "height": height,
                "source_type": "web",
            }
        )


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _extract_html(path: str) -> str:
    parser = _HTMLStripper()
    parser.feed(_read_text_file(path))
    return "\n".join(x.strip() for x in parser.parts if x.strip())


def _to_int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    m = re.search(r"\d+", str(value))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _extract_html_bundle(path: str) -> dict:
    parser = _HTMLStripper()
    parser.feed(_read_text_file(path))
    images = []
    for img in parser.images:
        img = dict(img)
        if img.get("local_path"):
            joined = urljoin(f"file://{path}", img["local_path"])
            img["local_path"] = joined.replace("file://", "")
        images.append(_normalize_image_info(img, default_source_type="web"))
    return {"text": "\n".join(x.strip() for x in parser.parts if x.strip()), "images": images}


def _normalize_image_info(img: dict, default_source_type: str = "file") -> dict:
    data = dict(img or {})
    url = str(data.get("url") or "").strip()
    local_path = str(data.get("local_path") or "").strip()
    return {
        "url": url,
        "local_path": local_path,
        "alt": str(data.get("alt") or "").strip(),
        "caption": str(data.get("caption") or "").strip(),
        "context_text": str(data.get("context_text") or "").strip(),
        "position": int(data.get("position") or 0),
        "width": _to_int_or_none(str(data.get("width") or "")),
        "height": _to_int_or_none(str(data.get("height") or "")),
        "source_type": str(data.get("source_type") or default_source_type),
    }


def _extract_pdf_bundle(path: str) -> dict:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("Для PDF-изображений нужен PyMuPDF (fitz)") from exc

    text_parts: list[str] = []
    images: list[dict] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc):
            text_parts.append(page.get_text())
            for img_info in page.get_images(full=True):
                xref = int(img_info[0]) if img_info else 0
                width = int(img_info[2]) if len(img_info) > 2 else None
                height = int(img_info[3]) if len(img_info) > 3 else None
                images.append(
                    _normalize_image_info(
                        {
                            "url": "",
                            "local_path": "",
                            "alt": "",
                            "caption": f"PDF image xref={xref}",
                            "context_text": page.get_text("text")[:280].strip(),
                            "position": page_index + 1,
                            "width": width,
                            "height": height,
                            "source_type": "pdf",
                        },
                        default_source_type="pdf",
                    )
                )
    return {"text": "\n".join(text_parts), "images": images}


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
        raise RuntimeError("DeepSeekOCR adapter пока не подключён")
    if ext in {".mp3", ".wav", ".m4a", ".ogg"}:
        raise RuntimeError("STT adapter пока не подключён")
    if ext in {".mp4", ".mkv", ".mov", ".avi", ".webm"}:
        raise RuntimeError("Video/STT adapter пока не подключён")

    raise RuntimeError(f"Неподдерживаемый тип файла: {ext or 'без расширения'}")


def clean_extracted_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_source(source: str) -> str:
    return clean_extracted_text(extract_source_bundle(source).get("text", ""))


def extract_source_bundle(source: str) -> dict:
    path = (source or "").strip()
    text = ""
    images: list[dict] = []
    if not path:
        return {"text": "", "images": []}
    if not os.path.exists(path):
        raise RuntimeError(f"Файл не найден: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".html", ".htm"}:
        bundle = _extract_html_bundle(path)
        text = bundle.get("text", "")
        images = bundle.get("images", [])
    elif ext == ".pdf":
        bundle = _extract_pdf_bundle(path)
        text = bundle.get("text", "")
        images = bundle.get("images", [])
    elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        text = ""
        images = [
            _normalize_image_info(
                {
                    "url": "",
                    "local_path": path,
                    "alt": "",
                    "caption": os.path.basename(path),
                    "context_text": "",
                    "position": 1,
                    "width": None,
                    "height": None,
                    "source_type": "image",
                },
                default_source_type="image",
            )
        ]
    else:
        text = extract_text_from_path(path)
        images = []
    return {"text": clean_extracted_text(text), "images": images}
