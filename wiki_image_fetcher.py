import os
import re
import csv
import time
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_MEDIASEARCH = "https://commons.wikimedia.org/w/rest.php/v1/search/title"
DE_WIKI_API = "https://de.wikipedia.org/w/api.php"
USER_AGENT = "AnkyX/1.0 (commons api)"
MAX_RETRIES = 3
RETRY_BACKOFF = [0.5, 1.0, 2.0]
DELAY_RANGE = (0.15, 0.25)
LOG_PATH = os.path.join("logs", "wiki_fetch.log")
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
STOPWORDS = {
    "conference",
    "booth",
    "expo",
    "team",
    "people",
    "portrait",
    "group",
    "award",
    "press",
    "festival",
}

RU_HINTS = {
    "падаль": "carrion",
    "давление": "pressure",
    "скорость": "speed",
    "вес": "weight",
    "масса": "mass",
    "высота": "height",
    "ширина": "width",
    "длина": "length",
    "время": "time",
    "свет": "light",
    "температура": "temperature",
    "звук": "sound",
    "вода": "water",
    "огонь": "fire",
    "земля": "earth",
    "воздух": "air",
    "камень": "stone",
    "гора": "mountain",
    "река": "river",
    "море": "sea",
    "океан": "ocean",
    "рыба": "fish",
    "птица": "bird",
    "животное": "animal",
    "растение": "plant",
    "дерево": "tree",
    "цветок": "flower",
    "листва": "leaf",
    "кожа": "skin",
    "кровь": "blood",
    "кость": "bone",
    "музыка": "music",
    "звезда": "star",
    "планета": "planet",
    "луна": "moon",
    "солнце": "sun",
    "карта": "map",
    "флаг": "flag",
    "города": "city",
    "деревня": "village",
    "город": "city",
    "дом": "house",
    "машина": "car",
    "автомобиль": "car",
    "самолёт": "airplane",
    "самолет": "airplane",
    "поезд": "train",
    "корабль": "ship",
    "человек": "person",
    "женщина": "woman",
    "мужчина": "man",
    "ребёнок": "child",
    "ребенок": "child",
}


@dataclass
class ImageCandidate:
    url: str
    title: str
    description: str
    score: int


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


# ==========================
# CSV helpers
# ==========================

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


# ==========================
# Logging & network helpers
# ==========================

def _ensure_log_dir() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def _log_detail(message: str) -> None:
    _ensure_log_dir()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(f"[{timestamp}] {message}\n")


def _sleep_between_requests() -> None:
    time.sleep(random.uniform(*DELAY_RANGE))


def _perform_get(
    params: Dict[str, Any],
    stream: bool = False,
    base_url: str = COMMONS_API,
) -> requests.Response:
    """GET запрос к Wikimedia API с повторными попытками."""

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(base_url, params=params, timeout=20, stream=stream)
            _log_detail(f"GET {resp.url} -> {resp.status_code}")
            if resp.status_code in {429, 503}:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            raise
        finally:
            _sleep_between_requests()
    if last_exc:
        raise last_exc
    raise Exception("Неизвестная ошибка запроса")


def _download_image_bytes(url: str, entry_id: Optional[int]) -> Optional[Tuple[bytes, str]]:
    """Скачать картинку с проверкой Content-Type, размера и ретраями."""

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(
                url,
                stream=True,
                timeout=20,
                allow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
            _log_detail(f"GET {url} -> {resp.status_code}")

            if resp.status_code in {429, 503}:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            if resp.status_code != 200:
                _log_detail(f"HTTP {resp.status_code} for url={url}")
                return None

            ct = (resp.headers.get("Content-Type", "") or "").lower()
            if not ct.startswith("image/"):
                _log_detail(f"NOT_IMAGE Content-Type={ct} url={url}")
                return None
            if ct.startswith("image/svg"):
                _log_detail(f"SKIP_SVG Content-Type={ct} url={url}")
                return None

            content_length = resp.headers.get("Content-Length")
            if content_length:
                try:
                    length_val = int(content_length)
                    if length_val > MAX_DOWNLOAD_BYTES:
                        _log_detail(f"SKIP_LARGE Content-Length={length_val} url={url}")
                        return None
                except ValueError:
                    pass

            collected = BytesIO()
            total = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    _log_detail(f"SKIP_LARGE_STREAM total={total} url={url}")
                    return None
                collected.write(chunk)
            return collected.getvalue(), ct
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            _log_detail(f"ID {entry_id}: ERROR DOWNLOAD {type(exc).__name__} url={url}")
            return None
        finally:
            _sleep_between_requests()
    if last_exc:
        _log_detail(f"ID {entry_id}: ERROR DOWNLOAD {type(last_exc).__name__} url={url}")
    return None


# ==========================
# Word cleaning
# ==========================

def clean_word(word: str) -> str:
    cleaned = (word or "").strip()
    cleaned = cleaned.replace("|", " ").replace("*", " ")
    cleaned = re.sub(r"[~\"']", "", cleaned)
    cleaned = re.split(r"\s*\|\s*|;|,|\(|/", cleaned)[0].strip()
    cleaned = re.sub(r"[\.,!?:]+$", "", cleaned).strip()
    cleaned = re.sub(r"^(der|die|das)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _word_hits(word_clean: str, text: str) -> bool:
    if not word_clean or not text:
        return False
    return re.search(rf"\b{re.escape(word_clean)}\b", text, flags=re.IGNORECASE) is not None


def _calc_score(word_clean: str, title: str, description: str, file_name: str) -> int:
    score = 0
    combined = " ".join([title or "", description or ""]).lower()
    if _word_hits(word_clean, title) or _word_hits(word_clean, description):
        score += 5
    if _word_hits(word_clean, file_name):
        score += 3
    for stop in STOPWORDS:
        if stop in combined or stop in file_name.lower():
            score -= 5
    return score


def _detect_pos(notes: str) -> str:
    lowered = (notes or "").lower()
    if any(marker in lowered for marker in ("verb", "vi", "vt")):
        return "verb"
    return "noun"


def _translation_hint(translation: str) -> Optional[str]:
    lowered = (translation or "").lower()
    for ru, en in RU_HINTS.items():
        if ru in lowered:
            return en
    return None


def _build_query(word_clean: str, translation: str, notes: str, kind: str) -> str:
    if not word_clean:
        return ""
    pos = _detect_pos(notes)
    hint = _translation_hint(translation)
    context: List[str] = []

    if pos == "verb" or kind == "MEANING":
        context.extend(["icon", "diagram", "symbol"])
    else:
        context.extend(["photo", "object", "nature"])

    terms = [word_clean]
    if hint:
        terms.append(hint)
    terms.extend(context)
    return " ".join(terms)


# ==========================
# Wikimedia API helpers
# ==========================


def file_imageinfo(file_title: str) -> Optional[Dict[str, Any]]:
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": 1024,
        "format": "json",
    }
    _log_detail(f"imageinfo title='{file_title}'")
    data = _perform_get(params).json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        url = info.get("thumburl") or info.get("url")
        mime = info.get("mime", "")
        width = info.get("width")
        height = info.get("height")
        if not url or not mime:
            continue
        if not mime.startswith("image/") or mime.startswith("image/svg"):
            continue
        if width is not None and height is not None and max(width, height) < 200:
            continue
        return {"url": url, "mime": mime, "width": width, "height": height}
    return None


def _wikipedia_pageimage(word_clean: str) -> Optional[ImageCandidate]:
    params = {
        "action": "opensearch",
        "search": word_clean,
        "limit": 1,
        "namespace": 0,
        "format": "json",
    }
    _log_detail(f"dewiki opensearch '{word_clean}'")
    data = _perform_get(params, base_url=DE_WIKI_API).json()
    titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    if not titles:
        return None
    page_title = titles[0]

    page_params = {
        "action": "query",
        "titles": page_title,
        "prop": "pageimages|description",
        "pithumbsize": 800,
        "format": "json",
    }
    page_data = _perform_get(page_params, base_url=DE_WIKI_API).json()
    pages = page_data.get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {})
        url = thumb.get("source")
        if not url:
            continue
        title = page.get("title", "")
        description = page.get("description", "") or page.get("terms", {}).get(
            "description", [""]
        )[0]
        score = _calc_score(word_clean, title, description, title)
        if score >= 3:
            return ImageCandidate(url=url, title=title, description=description, score=score)
    return None


def _commons_media_candidates(word_clean: str, query: str) -> List[ImageCandidate]:
    params = {"q": query, "limit": 10}
    _log_detail(f"commons media search '{query}'")
    data = _perform_get(params, base_url=COMMONS_MEDIASEARCH).json()
    pages = data.get("pages", []) if isinstance(data, dict) else []
    candidates: List[ImageCandidate] = []
    for page in pages:
        title = page.get("title") or page.get("key") or ""
        desc = page.get("description") or page.get("extract") or ""
        thumb = None
        thumb_info = page.get("thumbnail") or {}
        if isinstance(thumb_info, dict):
            thumb = thumb_info.get("url") or thumb_info.get("src")
        if not thumb:
            continue
        file_name = page.get("key") or title
        score = _calc_score(word_clean, title or "", desc or "", file_name or "")
        candidates.append(
            ImageCandidate(url=thumb, title=title or "", description=desc or "", score=score)
        )
    return candidates


def _select_best_candidate(candidates: Iterable[ImageCandidate]) -> Optional[ImageCandidate]:
    best: Optional[ImageCandidate] = None
    for candidate in candidates:
        if candidate.score < 3:
            continue
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def fetch_kind(word_clean: str, translation: str, notes: str, kind: str) -> Optional[str]:
    if not word_clean:
        return None

    wiki_candidate = _wikipedia_pageimage(word_clean)
    if wiki_candidate:
        return wiki_candidate.url

    query = _build_query(word_clean, translation, notes, kind)
    media_candidates = _commons_media_candidates(word_clean, query)
    best = _select_best_candidate(media_candidates)
    if best:
        if best.title.startswith("File:"):
            info = file_imageinfo(best.title)
            if info and info.get("url"):
                return info["url"]
        return best.url

    return None


# ==========================
# Download helpers
# ==========================

def download_to_png(url: str, out_path: str, entry_id: Optional[int]) -> bool:
    result = _download_image_bytes(url, entry_id)
    if not result:
        return False

    data, ct = result
    try:
        image = Image.open(BytesIO(data))
    except Image.UnidentifiedImageError:
        snippet = data[:200]
        _log_detail(
            f"ID {entry_id}: ERROR BAD_IMAGE (ct={ct}, url={url}, first_bytes={snippet!r})"
        )
        return False

    image = image.convert("RGBA")
    max_side = max(image.width, image.height)
    if max_side > 1024:
        ratio = 1024 / float(max_side)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path, format="PNG")
    return True


# ==========================
# Main processing
# ==========================

def process_csv_file(
    csv_path: str,
    images_dir: str = "images",
    stop_checker=lambda: False,
    progress_callback=None,
    overwrite: bool = False,
):
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

        path_photo = os.path.join(images_dir, f"{entry_id}_p.png")
        path_meaning = os.path.join(images_dir, f"{entry_id}_m.png")

        existing_photo = os.path.exists(path_photo)
        existing_meaning = os.path.exists(path_meaning)

        if existing_photo and existing_meaning and not overwrite:
            status = f"ID {entry_id}: SKIP"
            done += 1
            if progress_callback:
                progress_callback(done, total, status)
            continue

        word_clean = clean_word(row.get("word", ""))
        translation = (row.get("translation") or row.get("ru") or "").strip()
        notes = (row.get("notes") or row.get("comment") or "").strip()
        statuses: List[str] = []

        # PHOTO
        if existing_photo and not overwrite:
            statuses.append("OK (P)")
        else:
            try:
                url = fetch_kind(word_clean, translation, notes, "PHOTO")
                if url and download_to_png(url, path_photo, entry_id):
                    statuses.append("OK (P)")
                elif url:
                    statuses.append("ERROR: BAD IMAGE (P)")
                else:
                    statuses.append("NOT FOUND (P)")
            except requests.HTTPError as exc:
                statuses.append(f"ERROR: HTTP {getattr(exc.response, 'status_code', '?')}")
            except Exception as exc:  # noqa: BLE001
                statuses.append(f"ERROR: {type(exc).__name__}")

        if stop_checker():
            status = f"ID {entry_id}: {', '.join(statuses)}"
            done += 1
            if progress_callback:
                progress_callback(done, total, status)
            break

        # MEANING
        if existing_meaning and not overwrite:
            statuses.append("OK (M)")
        else:
            try:
                url = fetch_kind(word_clean, translation, notes, "MEANING")
                if url and download_to_png(url, path_meaning, entry_id):
                    statuses.append("OK (M)")
                elif url:
                    statuses.append("ERROR: BAD IMAGE (M)")
                else:
                    statuses.append("NOT FOUND (M)")
            except requests.HTTPError as exc:
                statuses.append(f"ERROR: HTTP {getattr(exc.response, 'status_code', '?')}")
            except Exception as exc:  # noqa: BLE001
                statuses.append(f"ERROR: {type(exc).__name__}")

        status = f"ID {entry_id}: {', '.join(statuses)}"
        done += 1
        if progress_callback:
            progress_callback(done, total, status)
    return done, total
