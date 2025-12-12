import os
import re
import csv
import requests
from typing import Any, Dict, Iterable, List, Optional, Tuple
from PIL import Image
from io import BytesIO

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    encodings = ["utf-8", "cp1251", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows: List[Dict[str, str]] = []
                for row in reader:
                    rows.append({k: (v or "").strip() for k, v in row.items()})
                return rows
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise Exception(f"Не удалось прочитать CSV: {last_err}")


def search_pages(query: str) -> List[Dict[str, Any]]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
    }
    resp = requests.get(WIKIMEDIA_API, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("query", {}).get("search", [])


def page_images(pageid: int) -> List[Dict[str, Any]]:
    params = {
        "action": "query",
        "prop": "images",
        "pageids": pageid,
        "format": "json",
    }
    resp = requests.get(WIKIMEDIA_API, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    page = pages.get(str(pageid), {})
    return page.get("images", []) or []


def file_url(filename: str) -> Optional[Dict[str, Any]]:
    params = {
        "action": "query",
        "titles": f"File:{filename}",
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "format": "json",
    }
    resp = requests.get(WIKIMEDIA_API, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    allowed_ext = {".png", ".jpg", ".jpeg"}
    for page in pages.values():
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        url = info.get("url")
        mime = info.get("mime", "")
        width = info.get("width")
        height = info.get("height")
        if not url:
            continue
        if mime.startswith("image/svg"):
            continue
        ext = os.path.splitext(url)[1].lower()
        if ext == ".svg" or ext not in allowed_ext:
            continue
        if width and height:
            if max(width, height) < 200:
                continue
        return {"url": url, "mime": mime, "width": width, "height": height}
    return None


def _is_concrete(notes: str, translation: str) -> Optional[bool]:
    notes_low = notes.lower()
    translation_low = translation.lower()
    noun_markers = [" m", " f", " n", " der", " die", " das", "noun"]
    verb_markers = [" vt", " vi", "verb"]
    if any(marker in notes_low for marker in noun_markers):
        return True
    if any(marker in notes_low for marker in verb_markers):
        return False
    if translation_low.startswith("(") or any(word in translation_low for word in ["действие", "процесс", "качество"]):
        return False
    return None


def _build_queries(word: str, translation: str, priority: str) -> Iterable[Tuple[str, str]]:
    word = word.strip()
    translation = translation.strip()
    photo_query = f"{word} {translation} photo".strip()
    meaning_parts = [word, "icon", "illustration"]
    if translation:
        extra = " ".join(translation.split()[:2])
        meaning_parts.append(extra)
    meaning_query = " ".join(part for part in meaning_parts if part)

    if priority == "photo_first":
        yield (photo_query, "PHOTO")
        yield (meaning_query, "MEANING")
    else:
        yield (meaning_query, "MEANING")
        yield (photo_query, "PHOTO")


def _download_image(url: str) -> bytes:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.content


def _sanitize_filename(name: str) -> str:
    return re.sub(r"^File:", "", name)


def fetch_best_image_for_row(row: Dict[str, str]) -> Tuple[Optional[bytes], Optional[str]]:
    word = (row.get("word") or "").strip()
    translation = (row.get("translation") or "").strip()
    notes = (row.get("notes") or "").strip()

    priority = _is_concrete(notes, translation)
    priority_key = "photo_first" if priority is not False else "meaning_first"

    if priority is None:
        priority_key = "photo_first"

    for query, kind in _build_queries(word, translation, priority_key):
        try:
            pages = search_pages(query)
        except Exception:
            continue
        for page in pages:
            pageid = page.get("pageid")
            if not pageid:
                continue
            try:
                images = page_images(pageid)
            except Exception:
                continue
            for img in images:
                title = img.get("title")
                if not title:
                    continue
                filename = _sanitize_filename(title)
                info = file_url(filename)
                if not info:
                    continue
                try:
                    data = _download_image(info["url"])
                    return data, kind
                except Exception:
                    continue
    return None, None


def save_png(data: bytes, out_path: str) -> None:
    image = Image.open(BytesIO(data))
    image = image.convert("RGBA")
    max_side = max(image.width, image.height)
    if max_side > 1024:
        ratio = 1024 / float(max_side)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    image.save(out_path, format="PNG")


def process_csv_file(csv_path: str, images_dir: str = "images", stop_checker=lambda: False, progress_callback=None):
    rows = _read_csv_rows(csv_path)
    os.makedirs(images_dir, exist_ok=True)
    total = len(rows)
    done = 0
    for row in rows:
        if stop_checker():
            break
        id_raw = row.get("id") or ""
        try:
            entry_id = int(re.search(r"\d+", id_raw).group(0))
        except Exception:
            done += 1
            if progress_callback:
                progress_callback(done, total, f"[пропуск] некорректный ID '{id_raw}'")
            continue
        data, kind = fetch_best_image_for_row(row)
        if data:
            out_path = os.path.join(images_dir, f"{entry_id}.png")
            try:
                save_png(data, out_path)
                status = f"ID {entry_id}: OK ({kind})"
            except Exception:
                status = f"ID {entry_id}: ошибка сохранения"
        else:
            status = f"ID {entry_id}: not found"
        done += 1
        if progress_callback:
            progress_callback(done, total, status)
    return done, total
