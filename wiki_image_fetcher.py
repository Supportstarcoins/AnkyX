import os
import re
import csv
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AnkyX/1.0 (Wikimedia image fetcher)"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    encodings = ["utf-8-sig", "cp1251", "cp1252"]
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


def _request_json(params: Dict[str, Any]) -> Dict[str, Any]:
    resp = SESSION.get(WIKIMEDIA_API, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def search_files(query: str) -> List[Dict[str, Any]]:
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": 6,
        "srlimit": 5,
        "srsearch": query,
        "format": "json",
    }
    data = _request_json(params)
    return data.get("query", {}).get("search", [])


def _get_imageinfo(title: str) -> Optional[Dict[str, Any]]:
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "format": "json",
    }
    data = _request_json(params)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        url = info.get("url")
        mime = info.get("mime", "")
        width = info.get("width")
        height = info.get("height")
        if not url or not mime:
            continue
        if not mime.startswith("image/"):
            continue
        if mime == "image/svg+xml":
            continue
        if width is not None and height is not None and max(width, height) < 200:
            continue
        return {"url": url, "mime": mime, "width": width, "height": height}
    return None


def _normalize_word(raw_word: str) -> str:
    word = (raw_word or "").strip()
    word = word.replace("|", " ").replace("*", " ")
    word = re.sub(r"[\"']", "", word)
    word = re.split(r"\s*\|\s*|;|,|\(", word)[0].strip()
    word = re.sub(r"[\.,]+$", "", word).strip()
    word = re.sub(r"^(der|die|das)\s+", "", word, flags=re.IGNORECASE)
    word = re.sub(r"\s+", " ", word)
    return word


def _build_queries(word: str) -> Iterable[Tuple[str, str]]:
    word_clean = _normalize_word(word)
    if not word_clean:
        return []
    photo_query = f"{word_clean} photo"
    meaning_query = f"{word_clean} icon illustration"
    return [(photo_query, "PHOTO"), (meaning_query, "MEANING")]


def _download_image(url: str) -> bytes:
    resp = SESSION.get(url, timeout=20)
    resp.raise_for_status()
    return resp.content


def fetch_best_image_for_row(row: Dict[str, str]) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    word = (row.get("word") or "").strip()
    last_error: Optional[str] = None

    for query, kind in _build_queries(word):
        try:
            pages = search_files(query)
        except requests.RequestException as exc:  # noqa: PERF203
            return None, None, f"HTTP error {exc}"
        except Exception as exc:  # noqa: BLE001
            return None, None, f"error: {type(exc).__name__}: {exc}"

        if not pages:
            continue

        for page in pages:
            title = page.get("title")
            if not title:
                continue
            try:
                info = _get_imageinfo(title)
            except requests.RequestException as exc:  # noqa: PERF203
                last_error = f"HTTP error {exc}"
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = f"error: {type(exc).__name__}: {exc}"
                continue

            if not info:
                continue
            try:
                data = _download_image(info["url"])
                return data, kind, None
            except requests.RequestException as exc:  # noqa: PERF203
                last_error = f"HTTP error {exc}"
            except Exception as exc:  # noqa: BLE001
                last_error = f"error: {type(exc).__name__}: {exc}"

    return None, None, last_error


def save_png(data: bytes, out_path: str) -> None:
    image = Image.open(BytesIO(data))
    image = image.convert("RGBA")
    max_side = max(image.width, image.height)
    if max_side > 1024:
        ratio = 1024 / float(max_side)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    image.save(out_path, format="PNG")


def process_csv_file(csv_path: str, images_dir: str = "images", stop_checker=lambda: False, progress_callback=None, overwrite: bool = False):
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
        out_path = os.path.join(images_dir, f"{entry_id}.png")
        if os.path.exists(out_path) and not overwrite:
            status = f"ID {entry_id}: уже скачан"
            done += 1
            if progress_callback:
                progress_callback(done, total, status)
            continue

        data, kind, err = fetch_best_image_for_row(row)
        if data:
            try:
                save_png(data, out_path)
                status = f"ID {entry_id}: OK ({kind})"
            except Exception as exc:  # noqa: BLE001
                status = f"ID {entry_id}: error: {type(exc).__name__}: {exc}"
        elif err:
            status = f"ID {entry_id}: {err}"
        else:
            status = f"ID {entry_id}: not found"
        done += 1
        if progress_callback:
            progress_callback(done, total, status)
    return done, total
