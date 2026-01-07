from __future__ import annotations

import os
import re
import zipfile
import html as _html
from pathlib import Path
from typing import Iterable, List, Optional, Set

DEFAULT_CHUNK_CHARS = 9000
DEFAULT_MAX_TOTAL_CHARS_SOFT = 8_000_000
DEFAULT_MAX_PDF_PAGES_SOFT = 80  # по умолчанию не даём рвать память


def _esc(s: str) -> str:
    return _html.escape(s or "", quote=False)


def _paras_to_html(paras: List[str]) -> str:
    parts = []
    for p in paras:
        t = (p or "").strip()
        if not t:
            continue
        parts.append(f"<p>{_esc(t)}</p>")
    return "".join(parts)


def _chunk_stream(paragraphs: Iterable[str], chunk_chars: int, max_total_chars: Optional[int]) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    total = 0

    for p in paragraphs:
        if not p:
            continue
        t = str(p).strip()
        if not t:
            continue

        total += len(t)
        if max_total_chars is not None and total > max_total_chars:
            break

        buf.append(t)
        buf_len += len(t) + 1

        if buf_len >= chunk_chars:
            html_chunk = _paras_to_html(buf)
            if html_chunk:
                chunks.append(html_chunk)
            buf = []
            buf_len = 0

    if buf:
        html_chunk = _paras_to_html(buf)
        if html_chunk:
            chunks.append(html_chunk)

    return chunks


# ---------- DOCX streaming ----------
def _iter_docx_paragraphs(path: str) -> Iterable[str]:
    import xml.etree.ElementTree as ET

    ns_w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    w_p = f"{{{ns_w}}}p"
    w_t = f"{{{ns_w}}}t"
    w_tab = f"{{{ns_w}}}tab"
    w_br = f"{{{ns_w}}}br"

    with zipfile.ZipFile(path) as zf:
        xml_name = "word/document.xml"
        if xml_name not in zf.namelist():
            raise RuntimeError("DOCX: не найден word/document.xml")

        with zf.open(xml_name) as fp:
            ctx = ET.iterparse(fp, events=("end",))
            for _, elem in ctx:
                if elem.tag != w_p:
                    continue

                parts = []
                for node in elem.iter():
                    if node.tag == w_t and node.text:
                        parts.append(node.text)
                    elif node.tag == w_tab:
                        parts.append("\t")
                    elif node.tag == w_br:
                        parts.append("\n")

                para = "".join(parts).strip()
                if para:
                    yield para

                elem.clear()


def import_docx(path: str, *, chunk_chars: int = DEFAULT_CHUNK_CHARS,
                max_total_chars: Optional[int] = DEFAULT_MAX_TOTAL_CHARS_SOFT) -> List[str]:
    if not os.path.exists(path):
        raise RuntimeError("DOCX: файл не найден")
    try:
        return _chunk_stream(_iter_docx_paragraphs(path), chunk_chars, max_total_chars)
    except MemoryError:
        raise RuntimeError("DOCX: недостаточно памяти. Импортируйте меньший документ.")
    except Exception as e:
        raise RuntimeError(f"DOCX: ошибка импорта: {e}")


# ---------- ODT streaming ----------
def _iter_odt_paragraphs(path: str) -> Iterable[str]:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as zf:
        xml_name = "content.xml"
        if xml_name not in zf.namelist():
            raise RuntimeError("ODT: не найден content.xml")

        with zf.open(xml_name) as fp:
            ctx = ET.iterparse(fp, events=("end",))
            for _, elem in ctx:
                # text:p и text:h (заголовки) — ловим по окончанию тега
                tag = elem.tag
                if not (tag.endswith("}p") or tag.endswith("}h")):
                    continue
                txt = "".join(elem.itertext()).strip()
                if txt:
                    txt = re.sub(r"[ \t]+", " ", txt)
                    yield txt
                elem.clear()


def import_odt(path: str, *, chunk_chars: int = DEFAULT_CHUNK_CHARS,
               max_total_chars: Optional[int] = DEFAULT_MAX_TOTAL_CHARS_SOFT) -> List[str]:
    if not os.path.exists(path):
        raise RuntimeError("ODT: файл не найден")
    try:
        return _chunk_stream(_iter_odt_paragraphs(path), chunk_chars, max_total_chars)
    except MemoryError:
        raise RuntimeError("ODT: недостаточно памяти. Импортируйте меньший документ.")
    except Exception as e:
        raise RuntimeError(f"ODT: ошибка импорта: {e}")


# ---------- PDF text-only ----------
def _parse_page_range(s: str) -> Optional[Set[int]]:
    """
    '1-3,5,10-12' -> {1,2,3,5,10,11,12}
    Возвращает None если строка пустая/невалидная.
    """
    s = (s or "").strip()
    if not s:
        return None
    pages: Set[int] = set()
    try:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                a, b = part.split("-", 1)
                a = int(a); b = int(b)
                if a <= 0 or b <= 0:
                    continue
                if a > b:
                    a, b = b, a
                for i in range(a, b + 1):
                    pages.add(i)
            else:
                x = int(part)
                if x > 0:
                    pages.add(x)
        return pages if pages else None
    except Exception:
        return None


def import_pdf(path: str, *, page_range: str = "",
               max_pages: int = DEFAULT_MAX_PDF_PAGES_SOFT,
               max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS_SOFT) -> List[str]:
    if not os.path.exists(path):
        raise RuntimeError("PDF: файл не найден")

    try:
        import pdfplumber  # type: ignore
    except Exception:
        raise RuntimeError("PDF: pdfplumber не установлен. Установите: pip install pdfplumber")

    selected = _parse_page_range(page_range)  # 1-based
    chunks: List[str] = []
    total = 0

    try:
        with pdfplumber.open(path) as pdf:
            n = len(pdf.pages)

            # если пользователь не указал диапазон, берём первые max_pages страниц
            if selected is None:
                page_indices = list(range(1, min(n, max_pages) + 1))
            else:
                # обрезаем по существующему n и по max_pages (страховка)
                page_indices = [p for p in sorted(selected) if 1 <= p <= n][:max_pages]

            for p in page_indices:
                page = pdf.pages[p - 1]
                text = (page.extract_text() or "").strip()
                total += len(text)
                if total > max_total_chars:
                    break

                if text:
                    chunks.append(f"<pre style='white-space:pre-wrap'>{_esc(text)}</pre>")
                else:
                    chunks.append("<p>(На странице нет извлекаемого текста. Возможно, это скан — нужен OCR.)</p>")

    except MemoryError:
        raise RuntimeError("PDF: недостаточно памяти. Укажите меньший диапазон страниц.")
    except Exception as e:
        raise RuntimeError(f"PDF: ошибка импорта: {e}")

    return chunks
