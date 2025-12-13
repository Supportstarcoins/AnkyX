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

SCORE_THRESHOLD_STRICT = 4
SCORE_THRESHOLD_RELAXED = 1

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
    stage: str = ""


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


def _calc_score(
    word_clean: str,
    title: str,
    description: str,
    file_name: str,
    *,
    language_hint: bool = False,
    soften_stopwords: bool = False,
) -> int:
    score = 0
    combined = " ".join([title or "", description or ""]).lower()
    if _word_hits(word_clean, file_name):
        score += 3
    if _word_hits(word_clean, description) or _word_hits(word_clean, title):
        score += 2
    if language_hint:
        score += 1
    for stop in STOPWORDS:
        if stop in combined or stop in (file_name or "").lower():
            score -= 3 if soften_stopwords else 5
    return score


def _translation_hint(translation: str) -> Optional[str]:
    lowered = (translation or "").lower()
    for ru, en in RU_HINTS.items():
        if ru in lowered:
            return en
    return None


def _build_query(word_clean: str, translation: str, notes: str, kind: str) -> str:
    if not word_clean:
        return ""
    _ = translation  # reserved for future language hints
    _ = notes
    if kind == "MEANING":
        return f"{word_clean} icon"
    return word_clean


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
        if mime not in {"image/jpeg", "image/png"}:
            continue
        if width is not None and height is not None and max(width, height) < 200:
            continue
        return {"url": url, "mime": mime, "width": width, "height": height}
    return None


def _language_hint_present(translation: str, notes: str) -> bool:
    combined = " ".join([translation or "", notes or ""]).lower()
    return any(token in combined for token in (" german", "[de]", "(de)", " de"))


def _commons_file_search(
    word_clean: str,
    query: str,
    *,
    limit: int,
    threshold: int,
    stage_label: str,
    entry_id: int,
    language_hint: bool,
    soften_stopwords: bool,
) -> Tuple[List[ImageCandidate], str]:
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": 6,
        "srlimit": limit,
        "srsearch": query,
        "format": "json",
    }
    _log_detail(f"ID {entry_id}: {stage_label} search '{query}'")
    data = _perform_get(params).json()
    search_results = data.get("query", {}).get("search", [])
    candidates: List[ImageCandidate] = []
    for item in search_results:
        title = item.get("title", "")
        snippet = item.get("snippet", "") or ""
        file_title = title if title.startswith("File:") else f"File:{title}"
        file_name = os.path.basename(file_title)
        score = _calc_score(
            word_clean,
            title,
            snippet,
            file_name,
            language_hint=language_hint,
            soften_stopwords=soften_stopwords,
        )
        if score < threshold:
            continue
        info = file_imageinfo(file_title)
        if not info or not info.get("url"):
            continue
        candidates.append(
            ImageCandidate(
                url=info["url"],
                title=file_title,
                description=snippet,
                score=score,
                stage=stage_label,
            )
        )
    log_msg = f"ID {entry_id}: {stage_label} candidates={len(search_results)}, passed={len(candidates)}"
    if not candidates and search_results:
        log_msg += f" (score<{threshold})"
    return sorted(candidates, key=lambda c: c.score, reverse=True), log_msg


def _commons_pageimage_fallback(
    word_clean: str, *, entry_id: int, language_hint: bool
) -> Tuple[List[ImageCandidate], str]:
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": 0,
        "srlimit": 10,
        "srsearch": word_clean,
        "format": "json",
    }
    _log_detail(f"ID {entry_id}: page search '{word_clean}'")
    data = _perform_get(params).json()
    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return [], f"ID {entry_id}: page fallback no_results"

    top = search_results[0]
    pageid = top.get("pageid")
    title = top.get("title", "")
    page_params = {
        "action": "query",
        "prop": "pageimages",
        "pithumbsize": 800,
        "format": "json",
        "piprop": "thumbnail|name|mime",
    }
    if pageid is not None:
        page_params["pageids"] = pageid
    else:
        page_params["titles"] = title
    page_data = _perform_get(page_params).json()
    pages = page_data.get("query", {}).get("pages", {})
    if isinstance(pageid, int) and str(pageid) in pages:
        page = pages.get(str(pageid), {})
    else:
        page = next(iter(pages.values()), {}) if pages else {}
    thumb = page.get("thumbnail") or {}
    url = thumb.get("source")
    mime = thumb.get("mime", "")
    description = top.get("snippet", "")
    if not url:
        return [], f"ID {entry_id}: page fallback used (no thumbnail)"
    if mime and not mime.startswith("image/"):
        return [], f"ID {entry_id}: page fallback skipped non-image"

    score = _calc_score(
        word_clean,
        title,
        description,
        title,
        language_hint=language_hint,
        soften_stopwords=True,
    )
    if score < SCORE_THRESHOLD_RELAXED:
        return [], f"ID {entry_id}: page fallback used (score<{SCORE_THRESHOLD_RELAXED})"

    candidate = ImageCandidate(
        url=url,
        title=title,
        description=description,
        score=score,
        stage="page-fallback",
    )
    return [candidate], f"ID {entry_id}: page fallback used"


def fetch_candidates(
    word_clean: str, translation: str, notes: str, kind: str, entry_id: int
) -> Tuple[List[ImageCandidate], List[str]]:
    if not word_clean:
        return [], [f"ID {entry_id}: empty_query"]

    language_hint = _language_hint_present(translation, notes) or bool(_translation_hint(translation))
    query = _build_query(word_clean, translation, notes, kind)
    logs: List[str] = []

    strict_candidates, strict_log = _commons_file_search(
        word_clean,
        query,
        limit=10,
        threshold=SCORE_THRESHOLD_STRICT,
        stage_label="strict",
        entry_id=entry_id,
        language_hint=language_hint,
        soften_stopwords=False,
    )
    logs.append(strict_log)
    if strict_candidates:
        return strict_candidates, logs

    relaxed_candidates, relaxed_log = _commons_file_search(
        word_clean,
        query,
        limit=20,
        threshold=SCORE_THRESHOLD_RELAXED,
        stage_label="relaxed",
        entry_id=entry_id,
        language_hint=language_hint,
        soften_stopwords=True,
    )
    logs.append(relaxed_log)
    if relaxed_candidates:
        return relaxed_candidates, logs

    fallback_candidates, fallback_log = _commons_pageimage_fallback(
        word_clean, entry_id=entry_id, language_hint=language_hint
    )
    logs.append(fallback_log)
    return fallback_candidates, logs


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


def _try_download_candidates(
    candidates: Iterable[ImageCandidate], out_path: str, entry_id: Optional[int]
) -> Tuple[bool, Optional[ImageCandidate]]:
    for cand in candidates:
        if download_to_png(cand.url, out_path, entry_id):
            _log_detail(
                f"ID {entry_id}: saved '{cand.title}' stage={cand.stage} score={cand.score}"
            )
            return True, cand
        _log_detail(
            f"ID {entry_id}: candidate failed '{cand.title}' stage={cand.stage} url={cand.url}"
        )
    return False, None


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

        raw_word = row.get("word", "")
        word_clean = clean_word(raw_word)
        translation = (row.get("translation") or row.get("ru") or "").strip()
        notes = (row.get("notes") or row.get("comment") or "").strip()
        statuses: List[str] = []
        reason_messages: List[str] = []
        _log_detail(f"ID {entry_id} raw_word='{raw_word}' -> clean_word='{word_clean}'")

        if not word_clean or len(word_clean) < 2:
            reason_messages.append("empty_query")
            statuses.append("NOT FOUND (P) [empty_query]")
            statuses.append("NOT FOUND (M) [empty_query]")
            status = f"ID {entry_id}: {', '.join(statuses)}"
            if reason_messages:
                status = f"{status} | {' | '.join(reason_messages)}"
            done += 1
            if progress_callback:
                progress_callback(done, total, status)
            continue

        # PHOTO
        if existing_photo and not overwrite:
            statuses.append("OK (P)")
        else:
            try:
                candidates, logs = fetch_candidates(
                    word_clean, translation, notes, "PHOTO", entry_id
                )
                reason_messages.extend(logs)
                if candidates:
                    ok, picked = _try_download_candidates(candidates, path_photo, entry_id)
                    if ok:
                        statuses.append(f"OK (P) [{picked.stage}]")
                    else:
                        statuses.append("ERROR: BAD IMAGE (P)")
                else:
                    last_reason = logs[-1] if logs else "no_results"
                    statuses.append(f"NOT FOUND (P) [{last_reason}]")
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
                candidates, logs = fetch_candidates(
                    word_clean, translation, notes, "MEANING", entry_id
                )
                reason_messages.extend(logs)
                if candidates:
                    ok, picked = _try_download_candidates(candidates, path_meaning, entry_id)
                    if ok:
                        statuses.append(f"OK (M) [{picked.stage}]")
                    else:
                        statuses.append("ERROR: BAD IMAGE (M)")
                else:
                    last_reason = logs[-1] if logs else "no_results"
                    statuses.append(f"NOT FOUND (M) [{last_reason}]")
            except requests.HTTPError as exc:
                statuses.append(f"ERROR: HTTP {getattr(exc.response, 'status_code', '?')}")
            except Exception as exc:  # noqa: BLE001
                statuses.append(f"ERROR: {type(exc).__name__}")

        reason_messages = list(dict.fromkeys(reason_messages))
        status = f"ID {entry_id}: {', '.join(statuses)}"
        if reason_messages:
            status = f"{status} | {' | '.join(reason_messages)}"
        done += 1
        if progress_callback:
            progress_callback(done, total, status)
    return done, total
