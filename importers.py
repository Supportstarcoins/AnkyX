import importlib
import importlib.util
from pathlib import Path
from typing import List

DOCX_AVAILABLE = importlib.util.find_spec("docx") is not None
ODF_AVAILABLE = importlib.util.find_spec("odf") is not None
PDFPLUMBER_AVAILABLE = importlib.util.find_spec("pdfplumber") is not None


def _chunk_paragraphs(paragraphs: List[str], min_size: int = 2000, max_size: int = 2500) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n".join(current).strip())
        current = []
        current_len = 0

    for paragraph in paragraphs:
        text = (paragraph or "").strip()
        if not text:
            continue
        if current_len + len(text) > max_size and current_len >= min_size:
            flush()
        current.append(text)
        current_len += len(text)

    flush()
    return [chunk for chunk in chunks if chunk]


def import_docx(path: str) -> List[str]:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx не установлен. Установите: pip install python-docx")
    docx = importlib.import_module("docx")
    document = docx.Document(path)
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    if not paragraphs:
        return []
    return _chunk_paragraphs(paragraphs)


def import_odt(path: str) -> List[str]:
    if not ODF_AVAILABLE:
        raise RuntimeError("odfpy не установлен. Установите: pip install odfpy")
    odf_text = importlib.import_module("odf.text")
    odf_opendocument = importlib.import_module("odf.opendocument")
    document = odf_opendocument.load(path)
    paragraphs = []
    for element in document.getElementsByType(odf_text.P):
        text_parts = []
        for node in element.childNodes:
            if node.nodeType == node.TEXT_NODE:
                text_parts.append(node.data)
        text_value = "".join(text_parts).strip()
        if text_value:
            paragraphs.append(text_value)
    if not paragraphs:
        return []
    return _chunk_paragraphs(paragraphs)


def import_pdf(path: str) -> List[str]:
    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber не установлен. Установите: pip install pdfplumber")
    pdfplumber = importlib.import_module("pdfplumber")
    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "").strip()
            pages.append(text)
    return pages
