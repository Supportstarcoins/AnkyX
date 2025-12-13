import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile
import hashlib
import html
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable, Dict, Any


class UnsupportedAnkiPackageError(Exception):
    """Raised when an Anki package cannot be parsed by the importer."""

from db_migrations import ensure_schema_for_import
from db_path import connect_to_db
MEDIA_ROOT = "media"
MEDIA_IMPORT_SUBDIR = "anki_import"
FIELD_SEPARATOR = "\x1f"
LOG_PATH = os.path.join("logs", "anki_import.log")


def get_connection() -> sqlite3.Connection:
    return connect_to_db(timeout=5)


def _select_collection_file(members: list[str]) -> str | None:
    candidates: list[tuple[int, str]] = []
    for name in members:
        base = os.path.basename(name.rstrip("/"))
        if base == "collection.anki21":
            candidates.append((0, name))
        elif base == "collection.anki2":
            candidates.append((1, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], len(item[1])))
    return candidates[0][1]


def inspect_package(path: str) -> dict:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        members = zf.namelist()

    has_dirs = any("/" in name.strip("/") for name in members if name.strip("/"))
    has_collection_anki21 = any(os.path.basename(m.rstrip("/")) == "collection.anki21" for m in members)
    has_collection_anki2 = any(os.path.basename(m.rstrip("/")) == "collection.anki2" for m in members)
    has_wal = any(os.path.basename(m.rstrip("/")) == "collection.anki2-wal" for m in members)
    has_shm = any(os.path.basename(m.rstrip("/")) == "collection.anki2-shm" for m in members)
    has_media = any(os.path.basename(m.rstrip("/")) == "media" for m in members)

    selected_collection = _select_collection_file(members)

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as log_fh:
            log_fh.write(
                f"[{datetime.now().isoformat()}] Inspecting {path}\n"
                f"Files: {members}\n"
            )
    except Exception:
        pass

    return {
        "files": members,
        "has_dirs": has_dirs,
        "has_collection_anki21": has_collection_anki21,
        "has_collection_anki2": has_collection_anki2,
        "has_wal": has_wal,
        "has_shm": has_shm,
        "has_media": has_media,
        "collection_member": selected_collection,
    }


def parse_models(col_models_json: str) -> dict:
    try:
        return json.loads(col_models_json or "{}") or {}
    except Exception:
        return {}


def parse_decks(col_decks_json: str) -> dict:
    try:
        return json.loads(col_decks_json or "{}") or {}
    except Exception:
        return {}


def convert_factor_to_ease(anki_factor: int) -> int:
    try:
        return max(130, int(round((anki_factor or 0) / 10)))
    except Exception:
        return 250


def normalize_due(anki_due: int, col_crt: int, mode: str) -> int:
    if mode == "preserve":
        # Anki stores review due as day index (for review) or seconds (for learning)
        if anki_due is None:
            return int(time.time())
        if anki_due > 10_000_000:  # already timestamp-like
            return int(anki_due)
        # col_crt is days since epoch
        base_days = int(col_crt or 0)
        return int((base_days + int(anki_due)) * 24 * 60 * 60)
    return int(time.time())


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned or "deck"


def _strip_html(text: str) -> str:
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", "", text)
    return html.unescape(no_tags)


def _render_cloze_plain(text: str, reveal: bool) -> str:
    def repl(match):
        content = match.group(2)
        return content if reveal else "____"

    return re.sub(r"\{\{c(\d+)::(.*?)\}\}", repl, text)


def normalize_fields_for_ui(fields_dict: dict, model_type: int = 0) -> dict:
    front_raw = ""
    back_raw = ""

    if "Front" in fields_dict and "Back" in fields_dict:
        front_raw = str(fields_dict.get("Front", ""))
        back_raw = str(fields_dict.get("Back", ""))
    elif "Text" in fields_dict and "Extra" in fields_dict:
        front_raw = str(fields_dict.get("Text", ""))
        back_raw = str(fields_dict.get("Extra", ""))
    else:
        non_empty = [str(v) for v in fields_dict.values() if str(v).strip()]
        if non_empty:
            front_raw = non_empty[0]
            if len(non_empty) > 1:
                back_raw = "\n\n".join(non_empty[1:])

    is_cloze = model_type == 1 or "{{c" in front_raw

    if is_cloze:
        front_plain = _strip_html(_render_cloze_plain(front_raw, False))
        back_plain = _strip_html(_render_cloze_plain(back_raw or front_raw, True))
    else:
        front_plain = _strip_html(front_raw)
        back_plain = _strip_html(back_raw)

    normalized = dict(fields_dict)
    normalized["_front"] = front_plain
    normalized["_back"] = back_plain
    return normalized


def rewrite_media_refs_in_fields(fields_dict: dict, filename_map: dict, new_name_map: dict) -> tuple[dict, list[tuple[str, str]]]:
    updated = {}
    media_entries: list[tuple[str, str]] = []

    img_pattern = re.compile(r"<img[^>]+src=\"([^\"]+)\"|<img[^>]+src='([^']+)'", re.IGNORECASE)
    sound_pattern = re.compile(r"\[sound:([^\]]+)\]")

    for key, value in fields_dict.items():
        text = value or ""

        def replace_img(match):
            orig = match.group(1) or match.group(2)
            mapped = filename_map.get(orig, orig)
            new_path = new_name_map.get(mapped)
            if new_path:
                media_entries.append((new_path, "image"))
                return match.group(0).replace(orig, new_path)
            return match.group(0)

        def replace_sound(match):
            orig = match.group(1)
            mapped = filename_map.get(orig, orig)
            new_path = new_name_map.get(mapped)
            if new_path:
                media_entries.append((new_path, "audio"))
                return f"[sound:{new_path}]"
            return match.group(0)

        text = img_pattern.sub(replace_img, text)
        text = sound_pattern.sub(replace_sound, text)
        updated[key] = text

    return updated, media_entries


def _ensure_deck(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM decks WHERE name = ? LIMIT 1;", (name,))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute(
        "INSERT INTO decks (name, description, front_template, back_template) VALUES (?, NULL, NULL, NULL);",
        (name,),
    )
    conn.commit()
    return cur.lastrowid


def _ensure_note_type(conn: sqlite3.Connection, model: dict) -> int:
    name = model.get("name") or f"Model_{model.get('id', '')}" or "Anki"
    fields = [f.get("name", "Field") for f in model.get("flds", [])]
    templates = []
    for tmpl in model.get("tmpls", []):
        templates.append(
            {
                "name": tmpl.get("name", "Card"),
                "front": tmpl.get("qfmt", ""),
                "back": tmpl.get("afmt", ""),
                "requires_image": False,
            }
        )
    cur = conn.cursor()
    cur.execute("SELECT id FROM note_types WHERE name = ? LIMIT 1;", (name,))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute(
        "INSERT INTO note_types (name, fields_json, card_templates_json) VALUES (?, ?, ?);",
        (name, json.dumps(fields, ensure_ascii=False), json.dumps(templates, ensure_ascii=False)),
    )
    conn.commit()
    return cur.lastrowid


def _render_cloze(text: str, reveal: bool) -> str:
    def repl(match):
        content = match.group(2)
        return content if reveal else "____"

    return re.sub(r"\{\{c(\d+)::(.*?)\}\}", repl, text)


def _render_front_back(model: dict, fields: dict) -> tuple[str, str]:
    lower_keys = {k.lower(): k for k in fields.keys()}
    front_key = None
    back_key = None

    for cand in ("front", "question", "word", "_front"):
        if cand in lower_keys:
            front_key = lower_keys[cand]
            break
    for cand in ("back", "answer", "translation", "_back"):
        if cand in lower_keys:
            back_key = lower_keys[cand]
            break

    if front_key and back_key:
        return str(fields.get(front_key, "")), str(fields.get(back_key, ""))

    if front_key:
        front_val = str(fields.get(front_key, ""))
        other_vals = [str(v) for k, v in fields.items() if k != front_key and v]
        return front_val, "\n\n".join(other_vals)

    values = [str(v) for v in fields.values() if str(v).strip()]
    if model.get("type") == 1 and values:
        text = values[0]
        return _render_cloze(text, False), _render_cloze(text, True)

    if len(values) >= 2:
        return values[0], values[1]
    if values:
        return values[0], ""
    return "", ""


def _prepare_media_maps(temp_dir: str, deck_name: str) -> tuple[dict, dict]:
    media_json_path = os.path.join(temp_dir, "media")
    filename_map: Dict[str, str] = {}
    if os.path.exists(media_json_path):
        try:
            with open(media_json_path, "r", encoding="utf-8") as fh:
                filename_map = json.load(fh) or {}
        except Exception:
            filename_map = {}

    new_name_map: Dict[str, str] = {}
    dest_root = os.path.join(MEDIA_ROOT, MEDIA_IMPORT_SUBDIR)
    os.makedirs(dest_root, exist_ok=True)
    deck_prefix = _sanitize_name(deck_name)

    for key, fname in filename_map.items():
        src_path = os.path.join(temp_dir, str(key))
        if not os.path.exists(src_path):
            continue
        ext = os.path.splitext(fname)[1]
        digest = hashlib.sha1(f"{deck_prefix}_{fname}_{key}".encode("utf-8")).hexdigest()[:8]
        dest_name = f"{deck_prefix}_{digest}_{fname}"
        dest_path = os.path.join(dest_root, dest_name)
        try:
            shutil.copy(src_path, dest_path)
            new_name_map[fname] = dest_path
        except Exception:
            continue

    return filename_map, new_name_map


def _map_state(queue_val: int, type_val: int) -> str:
    if queue_val in (0, 4) or type_val == 0:
        return "new"
    if queue_val in (1, 3) or type_val == 1:
        return "learning"
    if queue_val == 2 or type_val == 2:
        return "review"
    return "new"


def _phase_from_interval(ivl: int) -> int:
    if ivl <= 1:
        return 1
    if ivl <= 3:
        return 2
    if ivl <= 7:
        return 3
    if ivl <= 14:
        return 4
    if ivl <= 30:
        return 5
    if ivl <= 90:
        return 6
    if ivl <= 180:
        return 7
    if ivl <= 365:
        return 8
    if ivl <= 540:
        return 9
    return 10


def import_apkg(
    apkg_path: str,
    target_parent_deck_id: int | None = None,
    options: Dict[str, Any] | None = None,
    progress_cb: Callable[[str, Any], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> dict:
    opts = options or {}
    keep_schedule = bool(opts.get("keep_schedule"))
    import_media = bool(opts.get("import_media", True))

    temp_dir = tempfile.mkdtemp(prefix="apkg_import_")
    summary = {"notes": 0, "cards": 0, "media": 0, "errors": 0, "missing_note_refs": 0}
    done_cards = 0
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    inspection = inspect_package(apkg_path)
    collection_member = inspection.get("collection_member")
    if not collection_member:
        raise UnsupportedAnkiPackageError(
            "Неподдерживаемый формат Anki пакета. Похоже на .colpkg/новый формат. "
            "Экспортируйте колоду в старый .apkg (Anki Deck Package) и попробуйте снова."
        )

    conn_target = get_connection()
    ensure_schema_for_import(conn_target)

    try:
        with zipfile.ZipFile(apkg_path, "r") as zf:
            zf.extractall(temp_dir)

        collection_path = os.path.join(temp_dir, collection_member)
        if not os.path.exists(collection_path):
            raise FileNotFoundError(f"Не найден файл коллекции {collection_member}")

        ankiconn = sqlite3.connect(collection_path)
        ankiconn.row_factory = sqlite3.Row
        col_row = ankiconn.execute("SELECT * FROM col LIMIT 1;").fetchone()
        decks_map = parse_decks(col_row["decks"] if col_row else "{}")
        models_map = parse_models(col_row["models"] if col_row else "{}")
        col_crt = col_row["crt"] if col_row else 0

        cards_rows = ankiconn.execute("SELECT * FROM cards;").fetchall()
        cards_by_nid: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for row in cards_rows:
            cards_by_nid[int(row["nid"])].append(row)

        notes_rows = ankiconn.execute("SELECT * FROM notes;").fetchall()
        total_cards = len(cards_rows) or 1
        nid_map: dict[int, int] = {}
        first_note_logged = False

        base_deck_name = "anki"
        if decks_map:
            first_deck = next(iter(decks_map.values()))
            base_deck_name = first_deck.get("name", base_deck_name)
        filename_map_global, new_name_map_global = (
            _prepare_media_maps(temp_dir, base_deck_name) if import_media else ({}, {})
        )

        for note_row in notes_rows:
            if cancel_flag and cancel_flag():
                break

            model = models_map.get(str(note_row["mid"])) or {}
            fields_raw = (note_row["flds"] or "").split(FIELD_SEPARATOR)
            field_names = [f.get("name", f"field_{idx}") for idx, f in enumerate(model.get("flds", []))]
            fields = {name: fields_raw[idx] if idx < len(fields_raw) else "" for idx, name in enumerate(field_names)}

            tags = " ".join((note_row["tags"] or "").strip().split())
            cards_for_note = cards_by_nid.get(int(note_row["id"]), [])
            if not cards_for_note:
                continue

            primary_card = cards_for_note[0]
            deck_info = decks_map.get(str(primary_card["did"])) or {}
            deck_name = deck_info.get("name") or "Anki deck"
            deck_id = _ensure_deck(conn_target, deck_name)

            note_type_id = _ensure_note_type(conn_target, model)

            rewritten_fields, media_entries = (
                rewrite_media_refs_in_fields(fields, filename_map_global, new_name_map_global)
                if import_media
                else (fields, [])
            )

            normalized_fields = normalize_fields_for_ui(rewritten_fields, int(model.get("type", 0)))

            note_created_at = int(note_row["mod"] or time.time())
            cur = conn_target.cursor()
            cur.execute(
                """
                INSERT INTO notes (deck_id, note_type_id, fields_json, tags, external_id, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    deck_id,
                    note_type_id,
                    json.dumps(normalized_fields, ensure_ascii=False),
                    tags,
                    str(note_row["id"]),
                    "anki_import",
                    note_created_at,
                ),
            )
            note_id = cur.lastrowid
            nid_map[int(note_row["id"])] = note_id
            summary["notes"] += 1

            if not first_note_logged:
                front_preview = normalized_fields.get("_front", "")[:60]
                back_preview = normalized_fields.get("_back", "")[:60]
                try:
                    with open(LOG_PATH, "a", encoding="utf-8") as log_fh:
                        log_fh.write(
                            f"Imported note nid={note_row['id']} fields={field_names} front='{front_preview}' back='{back_preview}'\n"
                        )
                except Exception:
                    pass
                first_note_logged = True

            for card_row in cards_for_note:
                if cancel_flag and cancel_flag():
                    break
                card_deck = decks_map.get(str(card_row["did"])) or deck_info
                card_deck_id = _ensure_deck(conn_target, card_deck.get("name") or deck_name)

                note_id_for_card = nid_map.get(int(card_row["nid"]))
                if not note_id_for_card:
                    summary["missing_note_refs"] += 1
                    if progress_cb:
                        progress_cb("log", f"SKIP card {card_row['id']}: missing note nid={card_row['nid']}")
                    continue

                front, back = _render_front_back(model, normalized_fields)

                if keep_schedule:
                    queue_val = int(card_row["queue"])
                    if queue_val in (1, 3):
                        due_ts = int(card_row["due"] or time.time())
                    else:
                        due_ts = normalize_due(card_row["due"], col_crt, "preserve")
                    interval = int(card_row["ivl"] or 0)
                    phase = _phase_from_interval(interval)
                    state = _map_state(queue_val, int(card_row["type"]))
                    reps_val = int(card_row["reps"] or 0)
                    lapses_val = int(card_row["lapses"] or 0)
                    step_val = int(card_row["left"] or 0)
                    ease_val = convert_factor_to_ease(card_row["factor"])
                else:
                    due_ts = int(time.time())
                    interval = 0
                    phase = 1
                    state = "new"
                    reps_val = 0
                    lapses_val = 0
                    step_val = 0
                    ease_val = 250

                next_review_date = datetime.fromtimestamp(due_ts).isoformat()

                cur.execute(
                    """
                    INSERT INTO cards (
                        deck_id, front, back, next_review, leitner_level, front_image_path,
                        back_image_path, audio_path, note_id, template_ord, state, due, interval,
                        ease, reps, lapses, step_index, last_review, phase, external_id, source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        card_deck_id,
                        front,
                        back,
                        next_review_date,
                        max(1, phase),
                        None,
                        None,
                        None,
                        note_id_for_card,
                        int(card_row["ord"] or 0),
                        state,
                        int(due_ts),
                        interval,
                        ease_val,
                        reps_val,
                        lapses_val,
                        step_val,
                        int(time.time()),
                        phase,
                        str(card_row["id"]),
                        "anki_import",
                    ),
                )
                summary["cards"] += 1
                done_cards += 1
                if progress_cb:
                    progress_cb("progress", done_cards, total_cards, f"card {done_cards}/{total_cards}")

            if cancel_flag and cancel_flag():
                break

            if media_entries:
                for path, mtype in media_entries:
                    try:
                        from main import attach_media_to_note  # local import to avoid circular

                        attach_media_to_note(note_id, [(path, mtype, "back", "anki_import")])
                        summary["media"] += 1
                    except Exception:
                        summary["errors"] += 1

            conn_target.commit()

        for card_row in cards_rows:
            if cancel_flag and cancel_flag():
                break
            anki_nid = int(card_row["nid"])
            if anki_nid not in nid_map:
                summary["missing_note_refs"] += 1
                if progress_cb:
                    progress_cb("log", f"SKIP card {card_row['id']}: missing note nid={anki_nid}")
                continue
        if progress_cb:
            progress_cb(
                "log",
                f"Import finished. notes={summary['notes']} cards={summary['cards']} missing_note_refs={summary['missing_note_refs']} media={summary['media']} errors={summary['errors']}",
            )
        else:
            print(
                f"Import finished. notes={summary['notes']} cards={summary['cards']} missing_note_refs={summary['missing_note_refs']} media={summary['media']} errors={summary['errors']}"
            )
        return summary
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        conn_target.close()
