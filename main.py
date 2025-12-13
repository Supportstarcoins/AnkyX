import os, tempfile
safe = tempfile.gettempdir()
os.environ["TEMP"] = safe
os.environ["TMP"] = safe
import time
import base64
import sqlite3
import re
import threading
import queue
import json
import csv
import calendar
import shutil
csv.field_size_limit(10 * 1024 * 1024)
import gzip
import pickle
from datetime import date, datetime, timedelta
import collections
from csv_importer import (
    attach_image_if_exists,
    detect_encoding,
    map_row_to_fields,
    normalize_tags,
    render_card_faces,
    upsert_note_and_cards,
)
from anki_apkg_importer import import_apkg

# В начале main() или init_db()
os.makedirs("sentence_audio", exist_ok=True)

# Добавить в начало файла
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from stats_config import (
    StatsSettings,
    ensure_stats_settings_table,
    load_stats_settings,
    save_stats_settings,
)
from db_migrations import ensure_schema_for_import, run_migrations
from db_connect import DB_WRITE_LOCK, commit_with_retry, open_db
from srs import schedule_review
from bg_tasks import BackgroundTask, start_background_task
from ui_progress import BusyDialog, TaskRunner
from overdue_badges import (
    PhaseOverdueBadges,
    ensure_due_column,
    fetch_overdue_counts_by_phase,
)
from deck_timer import (
    DeckTimerController,
    ensure_deck_settings_row,
    ensure_deck_settings_table,
    get_deck_timer_settings,
    get_effective_timer,
    update_deck_timer_settings,
)
from video_tools import (
    VlcPlayerWidget,
    cut_video_clip,
    is_vlc_available,
    open_in_external_player,
)
from audio_player_widget import AudioPlayerWidget

# ==========================
# OCR: pytesseract + автопоиск tesseract.exe
# ==========================

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    OCR_AVAILABLE = False

from shutil import which


def auto_configure_tesseract():
    if not OCR_AVAILABLE:
        return
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
    if cmd and os.path.isabs(cmd) and os.path.exists(cmd):
        return
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


def is_tesseract_available() -> bool:
    if not OCR_AVAILABLE:
        return False
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    if os.path.isabs(cmd):
        return os.path.exists(cmd)
    return which(cmd) is not None


auto_configure_tesseract()

# ==========================
# Необязательные библиотеки
# ==========================

# Картинки
try:
    from PIL import Image, ImageTk, ImageDraw, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# TTS (озвучка)
try:
    import pyttsx3
    TTS_AVAILABLE = True
    _tts_engine = pyttsx3.init()
except Exception:
    TTS_AVAILABLE = False
    _tts_engine = None

# Звук файлов (Windows)
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# OpenAI
try:
    from openai import OpenAI
    OPENAI_LIB_AVAILABLE = True
except ImportError:
    OPENAI_LIB_AVAILABLE = False

# Распознавание речи (цифровой слух)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    sr = None
    SR_AVAILABLE = False

# Deutsch Wiktionary
try:
    import requests
    from bs4 import BeautifulSoup
    WIKTIONARY_AVAILABLE = True
except ImportError:
    WIKTIONARY_AVAILABLE = False

# Matplotlib (опционально)
try:
    import matplotlib
    matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    FigureCanvasTkAgg = None
    Figure = None


# OpenAI key только в памяти
OPENAI_API_KEY = None

MIC_DEVICE_INDEX = None

DEFAULT_FRONT_TEMPLATE = "{sentence_with_gap}"
DEFAULT_BACK_TEMPLATE = "{word} [{ipa}] ({gender}; pl. {plural})\n\n{sentence}\n\n{translation}"
MEDIA_FOLDER = "media"
MEDIA_IMPORT_SUBDIR = "anki_import"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

CSV_COLUMN_ALIASES = {
    "id": ["id", "ID", "word_id", "uid"],
    "word": ["word", "de", "german", "term"],
    "translation": ["translation", "ru", "russian", "meaning"],
    "example": ["example", "sentence", "beispiel"],
    "level": ["level", "cefr"],
}


def _find_column(header: list[str], aliases: list[str]) -> int | None:
    lowered = [h.lower() for h in header]
    for alt in aliases:
        if alt.lower() in lowered:
            return lowered.index(alt.lower())
    return None


def read_csv_dictionary(path: str) -> dict[int, dict]:
    """Читает CSV-словарь и возвращает словарь id -> запись."""
    encodings = ["utf-8", "cp1251"]
    last_error = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return {}
                col_map = {}
                for key, variants in CSV_COLUMN_ALIASES.items():
                    idx = _find_column(header, variants)
                    if idx is not None:
                        col_map[key] = idx

                if "id" not in col_map:
                    raise ValueError("В CSV нет колонки с ID")

                records: dict[int, dict] = {}
                for row in reader:
                    try:
                        raw_id = row[col_map["id"]].strip()
                        if not raw_id:
                            continue
                        entry_id = int(re.search(r"\d+", raw_id).group(0))
                    except Exception:
                        continue

                    def get_field(name: str) -> str:
                        idx = col_map.get(name)
                        return row[idx].strip() if idx is not None and idx < len(row) else ""

                    records[entry_id] = {
                        "word": get_field("word"),
                        "translation": get_field("translation"),
                        "example": get_field("example"),
                        "level": get_field("level"),
                    }
                return records
        except Exception as e:
            last_error = e
            continue
    raise Exception(f"Не удалось прочитать CSV: {last_error}")


def extract_id_from_filename(filename: str) -> int | None:
    match = re.search(r"(\d+)", os.path.basename(filename))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def extract_id_with_ocr(image_path: str) -> int | None:
    if not (OCR_AVAILABLE and PIL_AVAILABLE and is_tesseract_available()):
        return None
    try:
        img = Image.open(image_path)
        gray = ImageOps.grayscale(img)
        enhanced = ImageOps.autocontrast(gray)
        text = pytesseract.image_to_string(enhanced)
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
    except Exception:
        return None
    return None


def ensure_media_dir() -> str:
    os.makedirs(MEDIA_FOLDER, exist_ok=True)
    return MEDIA_FOLDER


def copy_image_to_media(src: str, target_id: int, move: bool = False) -> str:
    ensure_media_dir()
    ext = os.path.splitext(src)[1].lower() or ".png"
    target_name = f"id_{target_id}{ext}"
    dest_path = os.path.join(MEDIA_FOLDER, target_name)
    try:
        if move:
            shutil.move(src, dest_path)
        else:
            shutil.copy2(src, dest_path)
    except Exception:
        dest_path = src
    return dest_path

# ==========================
# НАСТРОЙКИ ПЕРЕВОДА И СЛОВАРИ
# ==========================

class DictionaryManager:
    """Менеджер для работы с большими словарями"""
    
    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = {}  # Для быстрого обратного поиска
        self.dictionary_size = 0
        self.loaded_files = []
        
    def load_builtin_dictionary(self):
        """Загрузить встроенный базовый словарь"""
        basic_dict = {
            # Основные слова (пример - на практике будет 100,000)
            'Haus': 'дом', 'Buch': 'книга', 'Tisch': 'стол', 'Stuhl': 'стул',
            'Fenster': 'окно', 'Tür': 'дверь', 'Zimmer': 'комната', 'Küche': 'кухня',
            'Schlafzimmer': 'спальня', 'Badezimmer': 'ванная', 'Wohnzimmer': 'гостиная',
            'Schule': 'школа', 'Universität': 'университет', 'Arbeit': 'работа',
            'Mensch': 'человек', 'Frau': 'женщина', 'Mann': 'мужчина', 'Kind': 'ребенок',
            'Tag': 'день', 'Nacht': 'ночь', 'Morgen': 'утро', 'Abend': 'вечер',
            'Wasser': 'вода', 'Essen': 'еда', 'Brot': 'хлеб', 'Milch': 'молоко',
            'Apfel': 'яблоко', 'Kaffee': 'кофе', 'Tee': 'чай', 'Saft': 'сок',
            'rot': 'красный', 'blau': 'синий', 'grün': 'зеленый', 'gelb': 'желтый',
            'groß': 'большой', 'klein': 'маленький', 'gut': 'хороший', 'schlecht': 'плохой',
            'schnell': 'быстрый', 'langsam': 'медленный', 'warm': 'теплый', 'kalt': 'холодный',
            # Слова из примера на картинке
            'Abschied': 'прощание', 'von': 'от', 'Basel': 'Базель',
            'Leinen': 'леер', 'los': 'отчалить', 'tschüss': 'пока',
            'Schweiz': 'Швейцария', 'Carolina': 'Каролина', 'hat': 'имеет',
            'Osterferien': 'пасхальные каникулы', 'im': 'в', 'Moment': 'момент',
            'ist': 'есть', 'sie': 'она', 'noch': 'еще', 'in': 'в',
            'der': 'определенный артикль', 'aber': '但', 'bald': 'скоро',
            'wieder': 'снова', 'Deutschland': 'Германия', 'bei': 'у',
            'ihren': 'ее', 'Freunden': 'друзья'
        }
        self.dictionary.update(basic_dict)
        # Создаем обратный словарь
        for german, russian in basic_dict.items():
            self.reverse_dictionary[russian.lower()] = german
        self.dictionary_size = len(self.dictionary)
        
    def load_from_csv(self, filename):
        """Загрузить словарь из CSV файла (формат: немецкое слово, русский перевод)"""
        try:
            count = 0
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        german = row[0].strip()
                        russian = row[1].strip()
                        if german and russian:
                            self.dictionary[german] = russian
                            self.reverse_dictionary[russian.lower()] = german
                            count += 1
            self.dictionary_size = len(self.dictionary)
            self.loaded_files.append(filename)
            return count
        except Exception as e:
            raise Exception(f"Ошибка загрузки CSV: {e}")
    
    def load_from_json(self, filename):
        """Загрузить словарь из JSON файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for german, russian in data.items():
                        self.dictionary[german] = russian
                        self.reverse_dictionary[russian.lower()] = german
                self.dictionary_size = len(self.dictionary)
                self.loaded_files.append(filename)
                return len(data) if isinstance(data, dict) else 0
        except Exception as e:
            raise Exception(f"Ошибка загрузки JSON: {e}")
    
    def load_from_compressed(self, filename):
        """Загрузить словарь из сжатого файла (gzip)"""
        try:
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                if filename.endswith('.json.gz'):
                    data = json.load(f)
                elif filename.endswith('.csv.gz'):
                    reader = csv.reader(f)
                    data = {row[0]: row[1] for row in reader if len(row) >= 2}
                
                for german, russian in data.items():
                    self.dictionary[german] = russian
                    self.reverse_dictionary[russian.lower()] = german
                
                self.dictionary_size = len(self.dictionary)
                self.loaded_files.append(filename)
                return len(data)
        except Exception as e:
            raise Exception(f"Ошибка загрузки сжатого файла: {e}")
    
    def save_compressed_dictionary(self, filename):
        """Сохранить словарь в сжатом формате для быстрой загрузки"""
        try:
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                json.dump(self.dictionary, f, ensure_ascii=False)
            return True
        except Exception as e:
            raise Exception(f"Ошибка сохранения словаря: {e}")
    
    def get_translation(self, word):
        """Получить перевод слова (немецкое -> русское)"""
        # Пробуем точное совпадение
        if word in self.dictionary:
            return self.dictionary[word]
        
        # Пробуем регистронезависимо
        for german, russian in self.dictionary.items():
            if german.lower() == word.lower():
                return russian
        
        return None
    
    def get_reverse_translation(self, word):
        """Получить обратный перевод (русское -> немецкое)"""
        word_lower = word.lower().strip()
        return self.reverse_dictionary.get(word_lower)
    
    def get_statistics(self):
        """Получить статистику словаря"""
        return {
            'total_words': self.dictionary_size,
            'loaded_files': self.loaded_files,
            'memory_size_mb': len(pickle.dumps(self.dictionary)) / (1024 * 1024)
        }
    
    def search_words(self, pattern, limit=50):
        """Поиск слов по шаблону"""
        results = []
        pattern_lower = pattern.lower()
        for german, russian in self.dictionary.items():
            if pattern_lower in german.lower() or pattern_lower in russian.lower():
                results.append((german, russian))
                if len(results) >= limit:
                    break
        return results
    
    def export_to_csv(self, filename):
        """Экспортировать словарь в CSV (немецкое слово, русский перевод)"""
        try:
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for german, russian in sorted(self.dictionary.items()):
                    writer.writerow([german, russian])
            return True
        except Exception as e:
            raise Exception(f"Ошибка экспорта: {e}")


class TranslationSettings:
    def __init__(self):
        self.use_embedded_dict = True
        self.use_openai = True
        self.show_translations = True
        self.show_back_translation = True  # Показывать перевод на задней стороне
        self.dictionary_paths = []
        self.default_dictionary_path = "german_russian_dict.csv"
        
    def save(self):
        data = {
            'use_embedded_dict': self.use_embedded_dict,
            'use_openai': self.use_openai,
            'show_translations': self.show_translations,
            'show_back_translation': self.show_back_translation,
            'dictionary_paths': self.dictionary_paths,
            'default_dictionary_path': self.default_dictionary_path
        }
        with open('translation_settings.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        try:
            with open('translation_settings.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.use_embedded_dict = data.get('use_embedded_dict', True)
                self.use_openai = data.get('use_openai', True)
                self.show_translations = data.get('show_translations', True)
                self.show_back_translation = data.get('show_back_translation', True)
                self.dictionary_paths = data.get('dictionary_paths', [])
                self.default_dictionary_path = data.get('default_dictionary_path', "german_russian_dict.csv")
        except:
            pass

# Глобальные объекты
DICTIONARY_MANAGER = DictionaryManager()
TRANSLATION_SETTINGS = TranslationSettings()

def init_dictionary():
    """Инициализировать словарь"""
    # Загружаем встроенный базовый словарь
    DICTIONARY_MANAGER.load_builtin_dictionary()
    
    # Загружаем настройки
    TRANSLATION_SETTINGS.load()
    
    # Пробуем загрузить дефолтный словарь
    default_dict_path = TRANSLATION_SETTINGS.default_dictionary_path
    if os.path.exists(default_dict_path):
        try:
            count = DICTIONARY_MANAGER.load_from_csv(default_dict_path)
            print(f"Загружено {count} слов из дефолтного словаря: {default_dict_path}")
        except Exception as e:
            print(f"Ошибка загрузки дефолтного словаря {default_dict_path}: {e}")
    
    # Загружаем дополнительные словари из сохраненных путей
    for path in TRANSLATION_SETTINGS.dictionary_paths:
        if os.path.exists(path) and path != default_dict_path:
            try:
                if path.endswith('.csv'):
                    count = DICTIONARY_MANAGER.load_from_csv(path)
                    print(f"Загружено {count} слов из {path}")
                elif path.endswith('.json'):
                    count = DICTIONARY_MANAGER.load_from_json(path)
                    print(f"Загружено {count} слов из {path}")
                elif path.endswith(('.gz', '.zip')):
                    count = DICTIONARY_MANAGER.load_from_compressed(path)
                    print(f"Загружено {count} слов из {path}")
            except Exception as e:
                print(f"Ошибка загрузки словаря {path}: {e}")

def get_translation(word: str, use_openai: bool = True) -> str:
    """
    Получить перевод слова с использованием словаря и/или OpenAI.
    Возвращает русский перевод для немецкого слова.
    """
    word_original = word.strip()
    
    # Сначала пробуем словарь (немецкое -> русское)
    if TRANSLATION_SETTINGS.use_embedded_dict:
        translation = DICTIONARY_MANAGER.get_translation(word_original)
        if translation:
            return translation
    
    # Пробуем OpenAI если включено и есть ключ
    if TRANSLATION_SETTINGS.use_openai and use_openai and OPENAI_API_KEY and OPENAI_LIB_AVAILABLE:
        try:
            client = get_openai_client(OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты переводчик с немецкого на русский. Отвечай только переводом слова без пояснений."},
                    {"role": "user", "content": f"Переведи с немецкого на русский слово: {word_original}"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            translation = response.choices[0].message.content.strip()
            if translation and translation != word_original:
                # Сохраняем в словарь для будущего использования
                DICTIONARY_MANAGER.dictionary[word_original] = translation
                DICTIONARY_MANAGER.reverse_dictionary[translation.lower()] = word_original
                return translation
        except Exception:
            pass
    
    return ""  # Перевод не найден

def get_german_translation(word: str, use_openai: bool = True) -> str:
    """
    Получить немецкий перевод для русского слова.
    """
    word_lower = word.lower().strip()
    
    # Сначала пробуем обратный словарь
    if TRANSLATION_SETTINGS.use_embedded_dict:
        german_word = DICTIONARY_MANAGER.get_reverse_translation(word_lower)
        if german_word:
            return german_word
    
    # Пробуем OpenAI если включено и есть ключ
    if TRANSLATION_SETTINGS.use_openai and use_openai and OPENAI_API_KEY and OPENAI_LIB_AVAILABLE:
        try:
            client = get_openai_client(OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты переводчик с русского на немецкий. Отвечай только переводом слова без пояснений."},
                    {"role": "user", "content": f"Переведи с русского на немецкий слово: {word}"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            german_word = response.choices[0].message.content.strip()
            if german_word and german_word != word:
                # Сохраняем в словарь для будущего использования
                DICTIONARY_MANAGER.dictionary[german_word] = word
                DICTIONARY_MANAGER.reverse_dictionary[word_lower] = german_word
                return german_word
        except Exception:
            pass
    
    return ""  # Перевод не найден

def translate_sentence(sentence: str, use_openai: bool = True) -> str:
    """
    Перевести предложение с немецкого на русский.
    """
    # Сначала пробуем перевести каждое слово через словарь
    words = re.findall(r'\b\w+\b', sentence, re.UNICODE)
    translated_words = []
    
    for word in words:
        translation = get_translation(word, use_openai=False)  # Сначала без OpenAI
        if translation:
            translated_words.append(translation)
        else:
            translated_words.append(word)
    
    # Пробуем OpenAI для всего предложения если не удалось через словарь
    if TRANSLATION_SETTINGS.use_openai and use_openai and OPENAI_API_KEY and OPENAI_LIB_AVAILABLE:
        try:
            client = get_openai_client(OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты переводчик с немецкого на русский. Отвечай только переводом предложения без пояснений."},
                    {"role": "user", "content": f"Переведи с немецкого на русский предложение: {sentence}"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            full_translation = response.choices[0].message.content.strip()
            if full_translation and full_translation != sentence:
                return full_translation
        except Exception:
            pass
    
    # Возвращаем слово-за-слово перевод если OpenAI не сработал
    return " ".join(translated_words)

# ==========================
# ЛЕЙТНЕР (фазы)
# ==========================

LEITNER_SCHEDULE = {
    1: timedelta(seconds=30),
    2: timedelta(minutes=25),
    3: timedelta(hours=1),
    4: timedelta(days=1),
    5: timedelta(days=3),
    6: timedelta(days=9),
    7: timedelta(days=16),
    8: timedelta(days=36),
    9: timedelta(days=56),
    10: timedelta(days=100),
}


def get_next_review_for_level(level: int) -> datetime:
    level = max(1, min(10, level))
    return datetime.now() + LEITNER_SCHEDULE[level]


# ==========================
# БАЗА ДАННЫХ
# ==========================

def get_connection():
    return open_db()


def ensure_basic_note_type_id(conn: sqlite3.Connection | None = None) -> int:
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    cur = conn.cursor()
    cur.execute("SELECT id FROM note_types WHERE name = 'Basic' LIMIT 1;")
    row = cur.fetchone()
    if row:
        note_type_id = row["id"]
    else:
        fields = ["word", "translation", "example", "level", "image"]
        templates = [
            {
                "name": "Word→Translation",
                "front": "{word}",
                "back": "{translation}\n\n{example}",
                "requires_image": False,
            },
            {
                "name": "Image→Word",
                "front": "{image}",
                "back": "{word}\n{translation}",
                "requires_image": True,
            },
        ]
        cur.execute(
            "INSERT OR IGNORE INTO note_types (name, fields_json, card_templates_json) VALUES (?, ?, ?);",
            ("Basic", json.dumps(fields, ensure_ascii=False), json.dumps(templates, ensure_ascii=False)),
        )
        conn.commit()
        cur.execute("SELECT id FROM note_types WHERE name = 'Basic' LIMIT 1;")
        note_type_id = cur.fetchone()["id"]

    if close_conn:
        conn.close()
    return note_type_id


def ensure_generated_note_type_id(conn: sqlite3.Connection | None = None) -> int:
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    cur = conn.cursor()
    cur.execute("SELECT id FROM note_types WHERE name = 'Generated' LIMIT 1;")
    row = cur.fetchone()
    if row:
        note_type_id = row["id"]
    else:
        fields = [
            "front",
            "back",
            "word",
            "translation",
            "example",
            "level",
            "image",
            "front_image_path",
            "back_image_path",
            "audio_path",
        ]
        templates = [
            {
                "name": "Front/Back",
                "front": "{front}",
                "back": "{back}",
                "requires_image": False,
            }
        ]
        cur.execute(
            "INSERT OR IGNORE INTO note_types (name, fields_json, card_templates_json) VALUES (?, ?, ?);",
            ("Generated", json.dumps(fields, ensure_ascii=False), json.dumps(templates, ensure_ascii=False)),
        )
        conn.commit()
        cur.execute("SELECT id FROM note_types WHERE name = 'Generated' LIMIT 1;")
        note_type_id = cur.fetchone()["id"]

    if close_conn:
        conn.close()
    return note_type_id


def _get_note_type(cur: sqlite3.Cursor, note_type_id: int):
    cur.execute(
        "SELECT id, name, fields_json, card_templates_json FROM note_types WHERE id = ?;",
        (note_type_id,),
    )
    return cur.fetchone()


def list_note_types() -> list[sqlite3.Row]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM note_types ORDER BY name;")
    rows = cur.fetchall()
    conn.close()
    return rows


def _extract_note_fields(card_row: dict, note_fields: dict | None = None) -> dict:
    source = dict(note_fields) if note_fields else {}

    def pick(*keys):
        for key in keys:
            if isinstance(key, (list, tuple)):
                for sub_key in key:
                    if source.get(sub_key):
                        return source.get(sub_key)
            elif source.get(key):
                return source.get(key)
        return None

    front_fallback = pick("word", "front", "_front")
    back_fallback = pick("translation", "back", "_back")
    example_fallback = pick("example", "front", "_front")

    fields = {
        "word": front_fallback or card_row.get("word") or card_row.get("target_word") or card_row.get("front") or card_row.get("_front") or "",
        "translation": back_fallback or card_row.get("translation") or card_row.get("back") or card_row.get("_back") or "",
        "example": example_fallback or card_row.get("example") or card_row.get("front") or card_row.get("_front") or "",
        "level": card_row.get("leitner_level", 1),
        "image": source.get("image")
        or card_row.get("image_path")
        or card_row.get("front_image_path")
        or card_row.get("back_image_path")
        or "",
    }
    return fields


def create_cards_from_note_templates(
    note_id: int,
    note_type_id: int,
    fields: dict,
    deck_id: int,
    skip_template_ords: set[int] | None = None,
    audio_path: str | None = None,
    audio_side: str = "back",
    audio_source: str | None = None,
):
    skip_template_ords = skip_template_ords or set()
    conn = get_connection()
    cur = conn.cursor()
    note_type_row = _get_note_type(cur, note_type_id)
    if not note_type_row:
        conn.close()
        return

    templates = json.loads(note_type_row["card_templates_json"])
    fields_default = collections.defaultdict(str, fields)
    audio_value = audio_path if audio_path is not None else fields_default.get("audio_path")

    created = 0
    for ord_idx, template in enumerate(templates):
        if ord_idx in skip_template_ords:
            continue
        requires_image = bool(template.get("requires_image"))
        image_value = fields_default.get("image") or fields_default.get("front_image_path")
        if requires_image and not image_value:
            continue

        formatter = collections.defaultdict(str, fields_default)
        front = str(template.get("front", "")).format_map(formatter)
        back = str(template.get("back", "")).format_map(formatter)
        front_image_path = fields_default.get("front_image_path") or (image_value if requires_image else None)
        back_image_path = fields_default.get("back_image_path") or None

        insert_card(
            deck_id,
            front,
            back,
            front_image_path=front_image_path,
            back_image_path=back_image_path,
            audio_path=audio_value,
            level=1,
            note_id=note_id,
            template_ord=ord_idx,
            audio_side=audio_side,
            audio_source=audio_source,
        )
        created += 1

    conn.close()
    return created


def ensure_note_for_card(
    card_row: sqlite3.Row | dict,
    note_fields: dict | None = None,
    create_cards: bool = False,
    skip_template_ords: set[int] | None = None,
) -> dict:
    card_dict = dict(card_row)
    if card_dict.get("note_id"):
        return card_dict

    conn = get_connection()
    note_type_id = ensure_basic_note_type_id(conn)
    cur = conn.cursor()
    fields = _extract_note_fields(card_dict, note_fields)
    cur.execute(
        """
        INSERT INTO notes (deck_id, note_type_id, fields_json, tags, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            card_dict.get("deck_id"),
            note_type_id,
            json.dumps(fields, ensure_ascii=False),
            "",
            int(time.time()),
        ),
    )
    note_id = cur.lastrowid
    cur.execute(
        "UPDATE cards SET note_id = ?, template_ord = ? WHERE id = ?;",
        (note_id, 0, card_dict.get("id")),
    )
    conn.commit()
    conn.close()

    card_dict["note_id"] = note_id
    card_dict["template_ord"] = 0

    if create_cards:
        skip = skip_template_ords if skip_template_ords is not None else {0}
        create_cards_from_note_templates(
            note_id,
            note_type_id,
            fields,
            card_dict.get("deck_id"),
            skip_template_ords=skip,
        )

    return card_dict


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1;",
        (table,),
    )
    return cur.fetchone() is not None


def log_imported_file(path: str):
    conn = get_connection()
    if not _table_exists(conn, "import_log"):
        conn.close()
        return
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO import_log (file_path, imported_at) VALUES (?, ?);",
        (path, int(time.time())),
    )
    conn.commit()
    conn.close()


def get_imported_files() -> set[str]:
    conn = get_connection()
    if not _table_exists(conn, "import_log"):
        conn.close()
        return set()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM import_log;")
    files = {row[0] for row in cur.fetchall()}
    conn.close()
    return files


def _get_media_columns(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("PRAGMA table_info(media);")
    return {row[1] for row in cursor.fetchall()}


def ensure_media_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    if not _table_exists(conn, "media"):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY,
                card_id INTEGER,
                note_id INTEGER,
                type TEXT NOT NULL,
                path TEXT NOT NULL,
                side TEXT NOT NULL DEFAULT 'back',
                source TEXT,
                created_at INTEGER NOT NULL
            );
            """
        )
        conn.commit()
        return

    columns = _get_media_columns(cur)
    if "card_id" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN card_id INTEGER;")
    if "note_id" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN note_id INTEGER;")
    if "type" not in columns and "media_type" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN type TEXT;")
    elif "type" not in columns and "media_type" in columns:
        cur.execute("ALTER TABLE media ADD COLUMN type TEXT;")
    if "side" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN side TEXT NOT NULL DEFAULT 'back';")
    if "source" not in columns:
        cur.execute("ALTER TABLE media ADD COLUMN source TEXT;")
    conn.commit()


def _media_type_column(columns: set[str]) -> str:
    return "media_type" if "media_type" in columns else "type"


def insert_media(
    conn: sqlite3.Connection,
    *,
    card_id: int | None = None,
    note_id: int | None = None,
    type: str,
    path: str,
    side: str = "back",
    source: str | None = None,
    created_at: int | None = None,
):
    if card_id is None and note_id is None:
        raise ValueError("Не указан card_id или note_id для медиа")

    ensure_media_table(conn)
    cur = conn.cursor()
    columns = _get_media_columns(cur)
    ts = int(created_at if created_at is not None else time.time())
    type_col = _media_type_column(columns)

    cols = ["note_id", "card_id", type_col, "path", "created_at"]
    vals = [note_id, card_id, type, path, ts]
    placeholders = ["?", "?", "?", "?", "?"]

    if "side" in columns:
        cols.append("side")
        vals.append(side or "back")
        placeholders.append("?")
    if "source" in columns:
        cols.append("source")
        vals.append(source)
        placeholders.append("?")

    cur.execute(
        f"INSERT INTO media ({', '.join(cols)}) VALUES ({', '.join(placeholders)});",
        vals,
    )


def attach_media_to_note(note_id: int, media_entries: list[tuple[str | None, str, str | None, str | None]]):
    ts = int(time.time())

    def write():
        ensure_media_table(conn)
        for entry in media_entries:
            if len(entry) == 2:
                path, media_type = entry
                side, source = "back", None
            elif len(entry) == 3:
                path, media_type, side = entry
                source = None
            else:
                path, media_type, side, source = entry
            if not path:
                continue
            insert_media(
                conn,
                note_id=note_id,
                type=media_type,
                path=path,
                side=side or "back",
                source=source,
                created_at=ts,
            )

    with open_db() as conn:
        try:
            commit_with_retry(conn, write)
        except Exception as e:
            conn.rollback()
            messagebox.showerror("Ошибка сохранения медиа", f"{type(e).__name__}: {e}")


def attach_media_to_card(card_id: int, media_entries: list[tuple[str | None, str, str | None, str | None]]):
    ts = int(time.time())

    def write():
        ensure_media_table(conn)
        for entry in media_entries:
            if len(entry) == 2:
                path, media_type = entry
                side, source = "back", None
            elif len(entry) == 3:
                path, media_type, side = entry
                source = None
            else:
                path, media_type, side, source = entry
            if not path:
                continue
            insert_media(
                conn,
                card_id=card_id,
                type=media_type,
                path=path,
                side=side or "back",
                source=source,
                created_at=ts,
            )

    with open_db() as conn:
        try:
            commit_with_retry(conn, write)
        except Exception as e:
            conn.rollback()
            messagebox.showerror("Ошибка сохранения медиа", f"{type(e).__name__}: {e}")


def get_media_for_card(card_id: int, note_id: int | None = None) -> list[dict]:
    conn = get_connection()
    if not _table_exists(conn, "media"):
        conn.close()
        return []

    cur = conn.cursor()
    columns = _get_media_columns(cur)
    clauses = []
    params: list[int] = []
    if "card_id" in columns:
        clauses.append("card_id = ?")
        params.append(card_id)
    if note_id is not None and "note_id" in columns:
        clauses.append("note_id = ?")
        params.append(note_id)

    if not clauses:
        conn.close()
        return []

    cur.execute(f"SELECT * FROM media WHERE {' OR '.join(clauses)};", params)
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


SOUND_TAG_PATTERN = re.compile(r"\[sound:([^\]]+)\]")


def _normalize_sound_name(sound_name: str) -> str:
    cleaned = sound_name.replace("\\", os.sep).replace("/", os.sep)
    return os.path.normpath(cleaned)


def resolve_sound_file(sound_name: str) -> str | None:
    cleaned = sound_name.strip()
    if cleaned.lower().startswith("[sound:") and cleaned.endswith("]"):
        cleaned = cleaned[len("[sound:") : -1]

    normalized = _normalize_sound_name(cleaned)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if os.path.isabs(normalized):
        candidates.append(normalized)
    else:
        candidates.extend(
            [
                os.path.join(base_dir, normalized),
                os.path.join(base_dir, MEDIA_FOLDER, normalized),
                os.path.join(base_dir, MEDIA_FOLDER, MEDIA_IMPORT_SUBDIR, normalized),
                normalized,
                os.path.join(MEDIA_FOLDER, normalized),
                os.path.join(MEDIA_FOLDER, MEDIA_IMPORT_SUBDIR, normalized),
                os.path.join(MEDIA_FOLDER, MEDIA_IMPORT_SUBDIR, os.path.basename(normalized)),
            ]
        )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def extract_audio_from_text(text: str, side: str) -> tuple[str, list[dict]]:
    matches = SOUND_TAG_PATTERN.findall(text or "")
    cleaned = SOUND_TAG_PATTERN.sub("", text or "")
    entries: list[dict] = []
    for sound_name in matches:
        resolved = resolve_sound_file(sound_name)
        entries.append(
            {
                "path": resolved,
                "name": os.path.basename(sound_name) or sound_name,
                "original": sound_name,
                "side": side,
                "missing": resolved is None,
            }
        )
    return cleaned, entries


def sanitize_card_sounds(card: dict) -> dict:
    if card.get("_sound_sanitized"):
        return card

    inline_entries: list[dict] = []
    for side_key in ("front", "back"):
        cleaned, entries = extract_audio_from_text(card.get(side_key, ""), side_key)
        card[side_key] = cleaned.strip()
        inline_entries.extend(entries)

    card["_inline_audio"] = inline_entries
    card["_sound_sanitized"] = True

    existing_entries = get_media_for_card(card.get("id"), card.get("note_id"))
    existing_paths = {
        entry.get("path")
        for entry in existing_entries
        if (entry.get("media_type") or entry.get("type") or "").lower() == "audio"
    }

    to_attach: list[tuple[str | None, str, str | None, str | None]] = []
    for entry in inline_entries:
        path = entry.get("path")
        if path and path not in existing_paths:
            to_attach.append((path, "audio", entry.get("side") or "front", "anki_inline"))
            existing_paths.add(path)

    if to_attach:
        if card.get("id"):
            attach_media_to_card(card.get("id"), to_attach)
        elif card.get("note_id"):
            attach_media_to_note(card.get("note_id"), to_attach)

    return card


def get_card_audio_entries(card: dict, prefer_side: str | None = None) -> list[dict]:
    sanitize_card_sounds(card)
    preferred = prefer_side.lower() if prefer_side else None
    entries: list[dict] = []
    seen: set[tuple] = set()

    def add_entry(path: str | None, label: str, side: str | None, source: str | None, missing: bool, media_id=None):
        key = (path or label, side or "back", missing)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "path": path,
                "label": f"{label} ({(side or 'back')})" + (" [нет файла]" if missing else ""),
                "side": side or "back",
                "source": source,
                "missing": missing,
                "media_id": media_id,
            }
        )

    for item in get_media_for_card(card.get("id"), card.get("note_id")):
        media_type = (item.get("media_type") or item.get("type") or "").lower()
        if media_type != "audio":
            continue
        path = item.get("path")
        missing = not (path and os.path.exists(path))
        label = os.path.basename(path) if path else "audio"
        add_entry(path, label, item.get("side"), item.get("source"), missing, item.get("id"))

    fallback = card.get("audio_path")
    if fallback:
        add_entry(
            fallback,
            os.path.basename(fallback) or "audio",
            card.get("audio_side", "back"),
            card.get("audio_source"),
            not os.path.exists(fallback),
            None,
        )

    for entry in card.get("_inline_audio", []):
        label = entry.get("name") or entry.get("original") or "audio"
        add_entry(entry.get("path"), label, entry.get("side"), "inline_sound", entry.get("missing", False), None)

    def sort_key(item: dict):
        side = (item.get("side") or "").lower()
        missing = bool(item.get("missing"))
        return (0 if preferred and side == preferred else 1, 0 if not missing else 1)

    entries.sort(key=sort_key)
    return entries


def get_card_audio_path(card: dict, prefer_side: str | None = "back") -> str | None:
    for entry in get_card_audio_entries(card, prefer_side=prefer_side):
        if entry.get("path") and not entry.get("missing"):
            return entry.get("path")
    return None


def update_media_side(media_id: int, side: str):
    with open_db() as conn:
        if not _table_exists(conn, "media"):
            return
        cur = conn.cursor()
        columns = _get_media_columns(cur)
        if "side" not in columns:
            return

        def write():
            cur.execute("UPDATE media SET side = ? WHERE id = ?;", (side, media_id))

        commit_with_retry(conn, write)


def remove_media_entry(media_id: int):
    with open_db() as conn:
        if not _table_exists(conn, "media"):
            return
        cur = conn.cursor()

        def write():
            cur.execute("DELETE FROM media WHERE id = ?;", (media_id,))

        commit_with_retry(conn, write)


def display_audio_entries_on_frame(audio_frame, entries: list[dict]):
    audio_widget = getattr(audio_frame, "audio_widget", None)
    if audio_widget is None:
        audio_widget = AudioPlayerWidget(audio_frame)
        audio_frame.audio_widget = audio_widget
        audio_widget.pack(fill=tk.X)

    selector_frame = getattr(audio_frame, "audio_selector_frame", None)
    if selector_frame is None:
        selector_frame = ttk.Frame(audio_frame)
        selector_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(selector_frame, text="Аудиофайлы:").pack(side=tk.LEFT, padx=(0, 5))
        audio_frame.audio_selector_var = tk.StringVar()
        combo = ttk.Combobox(
            selector_frame,
            textvariable=audio_frame.audio_selector_var,
            state="readonly",
        )
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        audio_frame.audio_selector = combo
        audio_frame.audio_selector_frame = selector_frame

    combo: ttk.Combobox = getattr(audio_frame, "audio_selector", None)
    if not entries:
        audio_widget.load(None)
        audio_widget.pack_forget()
        if combo:
            combo.set("")
            selector_frame.pack_forget()
        return

    selector_frame.pack(fill=tk.X, pady=(0, 2))
    audio_widget.pack(fill=tk.X)
    entry_map = {entry.get("label"): entry for entry in entries}
    audio_frame.audio_entry_map = entry_map
    labels = list(entry_map.keys())
    combo.config(values=labels)

    def on_select(*_args):
        selected = audio_frame.audio_selector_var.get()
        entry = audio_frame.audio_entry_map.get(selected)
        if not entry:
            return
        if entry.get("path") and not entry.get("missing"):
            audio_widget.load(entry.get("path"))
            audio_widget._set_status(entry.get("label"))
        else:
            audio_widget.load(None)
            audio_widget._set_status(f"{entry.get('label')} - аудио не найдено")

    combo.bind("<<ComboboxSelected>>", on_select)

    first_label = labels[0]
    audio_frame.audio_selector_var.set(first_label)
    if len(labels) == 1:
        combo.state(["disabled"])
    else:
        combo.state(["!disabled"])
    on_select()


def find_video_media_path(card: dict) -> str | None:
    media_entries = get_media_for_card(card.get("id"), card.get("note_id"))
    for entry in media_entries:
        media_type = (entry.get("media_type") or entry.get("type") or "").lower()
        path = entry.get("path")
        if media_type == "video" and path and os.path.exists(path):
            return path
    return None


def create_note(
    deck_id: int,
    fields: dict,
    note_type_id: int | None = None,
    tags: str = "",
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    note_type = note_type_id or ensure_generated_note_type_id(conn)
    cur.execute(
        """
        INSERT INTO notes (deck_id, note_type_id, fields_json, tags, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            deck_id,
            note_type,
            json.dumps(fields, ensure_ascii=False),
            tags,
            int(time.time()),
        ),
    )
    note_id = cur.lastrowid
    conn.commit()
    conn.close()
    return note_id


def create_note_with_cards(
    deck_id: int,
    fields: dict,
    note_type_id: int | None = None,
    tags: str = "",
    skip_template_ords: set[int] | None = None,
) -> tuple[int, int]:
    note_id = create_note(deck_id, fields, note_type_id=note_type_id, tags=tags)
    note_type = note_type_id or ensure_generated_note_type_id()
    audio_side = fields.get("audio_side", "back") if isinstance(fields, dict) else "back"
    audio_source = fields.get("audio_source") if isinstance(fields, dict) else None
    cards_created = create_cards_from_note_templates(
        note_id,
        note_type,
        fields,
        deck_id,
        skip_template_ords=skip_template_ords,
        audio_path=fields.get("audio_path"),
        audio_side=audio_side,
        audio_source=audio_source,
    )
    media_entries = [
        (fields.get("image") or fields.get("front_image_path"), "image", "front", None),
        (fields.get("back_image_path"), "image", "back", None),
        (fields.get("audio_path"), "audio", audio_side, audio_source),
    ]
    attach_media_to_note(note_id, media_entries)
    return note_id, cards_created


def ensure_notes_for_cards(cards: list[sqlite3.Row]) -> list[dict]:
    sanitized: list[dict] = []
    for card in cards:
        card_dict = ensure_note_for_card(card)
        sanitized.append(sanitize_card_sounds(card_dict))
    return sanitized


def find_note_by_source_id(deck_id: int, source_id: int) -> sqlite3.Row | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM notes WHERE deck_id = ?;", (deck_id,))
    rows = cur.fetchall()
    for row in rows:
        try:
            fields = json.loads(row["fields_json"])
        except Exception:
            continue
        if str(fields.get("source_id")) == str(source_id):
            conn.close()
            return row
    conn.close()
    return None


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    os.makedirs(MEDIA_FOLDER, exist_ok=True)

    # Колоды с сохранением шаблонов FRONT/BACK и иконкой
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            front_template TEXT,
            back_template TEXT,
            icon_path TEXT
        );
    """)

    # миграция для старых БД: добавляем колонки шаблонов, если их нет
    for col in ("front_template", "back_template", "icon_path"):
        try:
            cur.execute(f"ALTER TABLE decks ADD COLUMN {col} TEXT;")
        except sqlite3.OperationalError:
            # колонка уже существует
            pass

    # Карточки
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deck_id INTEGER NOT NULL,
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            next_review TEXT NOT NULL,
            leitner_level INTEGER NOT NULL DEFAULT 1,
            front_image_path TEXT,
            back_image_path TEXT,
            image_path TEXT,
            audio_path TEXT,
            note_id INTEGER,
            template_ord INTEGER,
            progress INTEGER NOT NULL DEFAULT 0,
            translation_shown BOOLEAN DEFAULT 1,
            overview_added BOOLEAN DEFAULT 0,
            state TEXT,
            due INTEGER,
            interval INTEGER,
            ease INTEGER,
            reps INTEGER,
            lapses INTEGER,
            step_index INTEGER,
            last_review INTEGER,
            FOREIGN KEY (deck_id) REFERENCES decks(id)
        );
    """)

    ensure_due_column(conn)

    # Миграции для старых БД (cards)
    for col in ("leitner_level", "front_image_path", "back_image_path",
                "image_path", "audio_path", "progress", "translation_shown", "overview_added"):
        try:
            if col == "leitner_level":
                cur.execute(
                    f"ALTER TABLE cards ADD COLUMN {col} INTEGER NOT NULL DEFAULT 1;"
                )
            elif col == "progress":
                cur.execute(
                    f"ALTER TABLE cards ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0;"
                )
            elif col == "translation_shown" or col == "overview_added":
                cur.execute(
                    f"ALTER TABLE cards ADD COLUMN {col} BOOLEAN DEFAULT 0;"
                )
            else:
                cur.execute(f"ALTER TABLE cards ADD COLUMN {col} TEXT;")
        except sqlite3.OperationalError:
            # колонка уже существует
            pass

    # Словарь уже известных слов
    cur.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL UNIQUE
        );
    """)

    # Статистика для диаграмм
    cur.execute("""
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            deck_id INTEGER,
            remembered_count INTEGER DEFAULT 0,
            forgotten_count INTEGER DEFAULT 0,
            reviewed_count INTEGER DEFAULT 0,
            UNIQUE(date, deck_id)
        );
    """)

    # Статистика ознакомления
    cur.execute("""
        CREATE TABLE IF NOT EXISTS overview_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            deck_id INTEGER NOT NULL,
            overview_count INTEGER DEFAULT 0,
            UNIQUE(date, deck_id)
        );
    """)

    ensure_deck_settings_table(conn)
    ensure_stats_settings_table(conn)
    ensure_media_table(conn)

    run_migrations(conn)
    ensure_schema_for_import(conn)

    conn.commit()
    conn.close()


def list_decks():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, description, icon_path FROM decks ORDER BY id;")
    decks = cur.fetchall()
    conn.close()
    return decks


def get_deck_templates(deck_id: int):
    """
    Загрузить шаблоны FRONT/BACK для конкретной колоды.
    Если в БД пусто — вернуть шаблоны по умолчанию.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT front_template, back_template FROM decks WHERE id = ?;",
        (deck_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row:
        front = row["front_template"]
        back = row["back_template"]
        return front or DEFAULT_FRONT_TEMPLATE, back or DEFAULT_BACK_TEMPLATE
    return DEFAULT_FRONT_TEMPLATE, DEFAULT_BACK_TEMPLATE


def save_deck_templates(deck_id: int, front_template: str, back_template: str):
    """
    Сохранить шаблоны FRONT/BACK для конкретной колоды.
    Это и есть «обучение» генератора по твоим шаблонам.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE decks SET front_template = ?, back_template = ? WHERE id = ?;",
            (front_template, back_template, deck_id)
        )
    except sqlite3.OperationalError:
        # На всякий случай, если колонок нет (очень старая БД)
        pass
    conn.commit()
    conn.close()


def get_deck_icon_path(deck_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT icon_path FROM decks WHERE id = ?;", (deck_id,))
    row = cur.fetchone()
    conn.close()
    return row["icon_path"] if row else None


def set_deck_icon_path(deck_id: int, icon_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE decks SET icon_path = ? WHERE id = ?;",
        (icon_path, deck_id)
    )
    conn.commit()
    conn.close()


def get_cards_in_deck(deck_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
        ORDER BY id;
    """, (deck_id,))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def get_due_cards(deck_id):
    """Карточки, у которых дата повторения ≤ сегодня."""
    today = date.today().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
          AND date(next_review) <= date(?)
        ORDER BY next_review, id;
    """, (deck_id, today))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def get_cards_for_repeat(deck_id):
    """
    Режим повторения:
    все карточки колоды, но первыми идут те,
    у которых дата повторения уже наступила.
    """
    today = date.today().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
        ORDER BY
            CASE WHEN date(next_review) <= date(?) THEN 0 ELSE 1 END,
            date(next_review),
            id;
    """, (deck_id, today))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def get_cards_for_playback(deck_id):
    """
    Режим воспроизведения:
    все карточки, отсортированы по прогрессу (меньше — раньше),
    затем по дате повторения.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
        ORDER BY progress ASC,
                 date(next_review) ASC,
                 id ASC;
    """, (deck_id,))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def get_overview_cards(deck_id):
    """
    Получить все карточки для режима ознакомления.
    В режим ознакомления попадают ВСЕ карточки колоды.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, deck_id, front, back, next_review, leitner_level,
               front_image_path, back_image_path, image_path,
               audio_path, progress, translation_shown, note_id, template_ord
        FROM cards
        WHERE deck_id = ?
        ORDER BY id;
    """, (deck_id,))
    cards = ensure_notes_for_cards(cur.fetchall())
    conn.close()
    return cards


def mark_card_for_overview(card_id: int):
    """Пометить карточку как добавленную в режим ознакомления"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards
        SET overview_added = 1
        WHERE id = ?;
    """, (card_id,))
    conn.commit()
    conn.close()


def update_card_leitner(card_id: int, new_level: int):
    new_level = max(1, min(10, new_level))
    next_dt = get_next_review_for_level(new_level).isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards
           SET leitner_level = ?, next_review = ?
         WHERE id = ?;
    """, (new_level, next_dt, card_id))
    conn.commit()
    conn.close()


def get_card_by_id(card_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cards WHERE id = ?;", (card_id,))
    row = cur.fetchone()
    conn.close()
    return ensure_note_for_card(row) if row else None


def apply_srs_update(card_id: int, rating: int):
    row = get_card_by_id(card_id)
    if not row:
        return None
    card_data = dict(row)
    card_data.setdefault("phase", card_data.get("leitner_level", 1))
    now_ts = int(time.time())
    result = schedule_review(card_data, rating, now_ts)
    next_review_value = datetime.fromtimestamp(result["due"]).isoformat()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE cards
           SET state = ?, due = ?, interval = ?, ease = ?, reps = ?, lapses = ?,
               step_index = ?, last_review = ?, leitner_level = ?, next_review = ?
         WHERE id = ?;
        """,
        (
            result["state"], result["due"], result["interval"], result["ease"],
            result["reps"], result["lapses"], result["step_index"],
            result["last_review"], result["phase"], next_review_value, card_id
        ),
    )
    cur.execute(
        """
        INSERT INTO reviews (
            card_id, reviewed_at, rating,
            interval_before, interval_after,
            ease_before, ease_after,
            phase_before, phase_after
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            card_id,
            now_ts,
            rating,
            result.get("interval_before"),
            result.get("interval_after"),
            result.get("ease_before"),
            result.get("ease_after"),
            result.get("phase_before"),
            result.get("phase_after"),
        ),
    )
    conn.commit()
    conn.close()
    return result


def update_card_progress(card_id: int, new_progress: int):
    new_progress = max(0, min(100, new_progress))
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards
           SET progress = ?
         WHERE id = ?;
    """, (new_progress, card_id))
    conn.commit()
    conn.close()


def update_card_translation_shown(card_id: int, shown: bool):
    """Обновить состояние показа перевода для карточки"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards
           SET translation_shown = ?
         WHERE id = ?;
    """, (1 if shown else 0, card_id))
    conn.commit()
    conn.close()


def delete_card(card_id: int):
    """Удаление карточки из базы."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM cards WHERE id = ?;", (card_id,))
    conn.commit()
    conn.close()


def count_overdue_for_deck(deck_id: int) -> int:
    """Кол-во просроченных карточек по колонке ``due`` (с учётом подколод)."""

    counts = fetch_overdue_counts_by_phase(None, deck_id)
    return counts.total


def get_deck_stats(deck_id: int):
    """
    Получить статистику по колоде:
    - Общее количество карточек
    - Количество карточек по фазам (1-10)
    - Процент изученности (progress > 80%)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Общее количество карточек
    cur.execute("SELECT COUNT(*) FROM cards WHERE deck_id = ?", (deck_id,))
    total = cur.fetchone()[0]
    
    # Карточки по фазам
    phase_stats = {}
    for phase in range(1, 11):
        cur.execute(
            "SELECT COUNT(*) FROM cards WHERE deck_id = ? AND leitner_level = ?",
            (deck_id, phase)
        )
        phase_stats[phase] = cur.fetchone()[0]
    
    # Процент изученности
    cur.execute(
        "SELECT COUNT(*) FROM cards WHERE deck_id = ? AND progress >= 80",
        (deck_id,)
    )
    learned_count = cur.fetchone()[0]
    learned_percent = (learned_count / total * 100) if total > 0 else 0
    
    # Статистика ознакомления
    cur.execute(
        """SELECT SUM(overview_count) as total_overview 
           FROM overview_statistics 
           WHERE deck_id = ?""",
        (deck_id,)
    )
    row = cur.fetchone()
    total_overview = row["total_overview"] or 0 if row else 0
    
    conn.close()
    
    return {
        "total": total,
        "phase_stats": phase_stats,
        "learned_percent": learned_percent,
        "learned_count": learned_count,
        "total_overview": total_overview
    }


# ==========================
# Статистика
# ==========================

def update_statistics(deck_id: int, remembered: bool = False, forgotten: bool = False, reviewed: bool = False):
    """
    Обновить статистику для диаграмм.
    remembered: True если нажата кнопка "Помню"
    forgotten: True если нажата кнопка "Забыл"
    reviewed: True если карточка просмотрена (любая)
    """
    today = date.today().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, есть ли запись на сегодня
    cur.execute(
        "SELECT id, remembered_count, forgotten_count, reviewed_count FROM statistics WHERE date = ? AND deck_id = ?",
        (today, deck_id)
    )
    row = cur.fetchone()
    
    if row:
        # Обновляем существующую запись
        rem_count = row["remembered_count"] + (1 if remembered else 0)
        forg_count = row["forgotten_count"] + (1 if forgotten else 0)
        rev_count = row["reviewed_count"] + (1 if reviewed else 0)
        
        cur.execute("""
            UPDATE statistics 
            SET remembered_count = ?, forgotten_count = ?, reviewed_count = ?
            WHERE id = ?
        """, (rem_count, forg_count, rev_count, row["id"]))
    else:
        # Создаем новую запись
        cur.execute("""
            INSERT INTO statistics (date, deck_id, remembered_count, forgotten_count, reviewed_count)
            VALUES (?, ?, ?, ?, ?)
        """, (today, deck_id, 
              1 if remembered else 0, 
              1 if forgotten else 0,
              1 if reviewed else 0))
    
    conn.commit()
    conn.close()


def update_overview_statistics(deck_id: int, increment: int = 1):
    """
    Обновить статистику ознакомления.
    increment: +1 при нажатии "следующий", -1 при нажатии "назад"
    """
    today = date.today().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем, есть ли запись на сегодня
    cur.execute(
        "SELECT id, overview_count FROM overview_statistics WHERE date = ? AND deck_id = ?",
        (today, deck_id)
    )
    row = cur.fetchone()
    
    if row:
        # Обновляем существующую запись
        new_count = max(0, row["overview_count"] + increment)
        cur.execute("""
            UPDATE overview_statistics 
            SET overview_count = ?
            WHERE id = ?
        """, (new_count, row["id"]))
    else:
        if increment > 0:
            # Создаем новую запись только если increment положительный
            cur.execute("""
                INSERT INTO overview_statistics (date, deck_id, overview_count)
                VALUES (?, ?, ?)
            """, (today, deck_id, increment))
    
    conn.commit()
    conn.close()


def get_statistics_for_dates(deck_id: int, date_list: list[date]):
    """Получить статистику для заданного списка дат (date)."""
    if not date_list:
        return {"dates": [], "remembered": [], "forgotten": [], "reviewed": [], "overview": []}

    iso_dates = [d.isoformat() for d in date_list]
    conn = get_connection()
    cur = conn.cursor()

    placeholders = ",".join("?" * len(iso_dates))

    cur.execute(
        f"""
        SELECT date, remembered_count, forgotten_count, reviewed_count
        FROM statistics
        WHERE deck_id = ? AND date IN ({placeholders})
        """,
        [deck_id, *iso_dates],
    )
    rows = cur.fetchall()

    cur.execute(
        f"""
        SELECT date, overview_count
        FROM overview_statistics
        WHERE deck_id = ? AND date IN ({placeholders})
        """,
        [deck_id, *iso_dates],
    )
    overview_rows = cur.fetchall()
    conn.close()

    remembered_data = collections.OrderedDict((d, 0) for d in iso_dates)
    forgotten_data = collections.OrderedDict((d, 0) for d in iso_dates)
    reviewed_data = collections.OrderedDict((d, 0) for d in iso_dates)
    overview_data = collections.OrderedDict((d, 0) for d in iso_dates)

    for row in rows:
        date_str = row["date"]
        if date_str in remembered_data:
            remembered_data[date_str] = row["remembered_count"]
            forgotten_data[date_str] = row["forgotten_count"]
            reviewed_data[date_str] = row["reviewed_count"]

    for row in overview_rows:
        date_str = row["date"]
        if date_str in overview_data:
            overview_data[date_str] = row["overview_count"]

    return {
        "dates": list(remembered_data.keys()),
        "remembered": list(remembered_data.values()),
        "forgotten": list(forgotten_data.values()),
        "reviewed": list(reviewed_data.values()),
        "overview": list(overview_data.values()),
    }


def get_statistics_for_period(deck_id: int, days: int = 30):
    """
    Получить статистику за последние N дней.
    Возвращает словарь с датами и значениями.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)

    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)

    return get_statistics_for_dates(deck_id, date_range)


def get_monthly_summary(deck_id: int):
    """
    Получить сводку за текущий месяц.
    """
    today = date.today()
    first_day = date(today.year, today.month, 1)
    
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            SUM(remembered_count) as total_remembered,
            SUM(forgotten_count) as total_forgotten,
            SUM(reviewed_count) as total_reviewed
        FROM statistics 
        WHERE deck_id = ? AND date BETWEEN ? AND ?
    """, (deck_id, first_day.isoformat(), today.isoformat()))
    
    row = cur.fetchone()
    
    # Получаем статистики ознакомления
    cur.execute("""
        SELECT SUM(overview_count) as total_overview
        FROM overview_statistics 
        WHERE deck_id = ? AND date BETWEEN ? AND ?
    """, (deck_id, first_day.isoformat(), today.isoformat()))
    
    overview_row = cur.fetchone()
    conn.close()
    
    total_overview = overview_row["total_overview"] or 0 if overview_row else 0
    
    return {
        "total_remembered": row["total_remembered"] or 0,
        "total_forgotten": row["total_forgotten"] or 0,
        "total_reviewed": row["total_reviewed"] or 0,
        "total_overview": total_overview,
        "success_rate": (row["total_remembered"] or 0) / max(1, (row["total_reviewed"] or 1)) * 100
    }


# ==========================
# Работа со словами
# ==========================

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def normalize_word(w: str) -> str:
    return w.strip().lower()


def extract_words_from_text(text: str) -> list:
    return [normalize_word(m.group(0)) for m in WORD_RE.finditer(text)]


def split_into_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def get_known_words() -> set:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT text FROM words;")
    rows = cur.fetchall()
    conn.close()
    return {r["text"] for r in rows}


def add_new_words(words: set):
    if not words:
        return
    conn = get_connection()
    cur = conn.cursor()
    for w in words:
        try:
            cur.execute("INSERT OR IGNORE INTO words (text) VALUES (?);", (w,))
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()


# ==========================
# Wiktionary
# ==========================

def get_wiktionary_data(word: str) -> dict:
    """
    Тянем базовую инфу с de.wiktionary.org:
    ipa, род, мн.ч, примерная таблица форм, синонимы, примеры.
    Если что-то не получается / нет модулей – возвращаем пустые поля.
    """
    data = {
        "ipa": "",
        "gender": "",
        "plural": "",
        "declension": "",
        "synonyms": [],
        "examples": [],
    }
    if not WIKTIONARY_AVAILABLE:
        return data

    try:
        url = f"https://de.wiktionary.org/wiki/{word}"
        headers = {"User-Agent": "Mozilla/5.0 (anki-clone)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return data
        soup = BeautifulSoup(resp.text, "html.parser")

        # IPA
        ipa_span = soup.find("span", class_="ipa")
        if ipa_span:
            data["ipa"] = ipa_span.get_text(strip=True)

        # Род / мн. число: ищем в первых таблицах флексии
        table = soup.find("table", class_="wikitable")
        if table:
            txt = table.get_text("\n", strip=True)
            data["declension"] = txt
            lower = txt.lower()
            if "genus" in lower:
                for line in txt.splitlines():
                    if "Genus" in line:
                        data["gender"] = line.replace("Genus", "").strip(": ").strip()
                        break
            if "plural" in lower:
                for line in txt.splitlines():
                    if "Plural" in line:
                        data["plural"] = line.replace("Plural", "").strip(": ").strip()
                        break

        # Синонимы
        syn_head = soup.find(id="Synonyme")
        if syn_head:
            ul = syn_head.find_next("ul")
            if ul:
                for li in ul.find_all("li", recursive=False):
                    text = li.get_text(" ", strip=True)
                    if text:
                        data["synonyms"].append(text)

        # Примеры
        ex_head = soup.find(id="Beispiele") or soup.find(id="Beispiel")
        if ex_head:
            ul = ex_head.find_next("ul")
            if ul:
                for li in ul.find_all("li", recursive=False):
                    text = li.get_text(" ", strip=True)
                    if text:
                        data["examples"].append(text)

    except Exception:
        return data

    return data


# ==========================
# OpenAI функции
# ==========================

def get_openai_client(api_key: str):
    if not OPENAI_LIB_AVAILABLE:
        raise RuntimeError("Библиотека 'openai' не установлена")
    if not api_key:
        raise RuntimeError("OpenAI API key не задан")
    return OpenAI(api_key=api_key)


def generate_image_with_openai(prompt: str, api_key: str, save_dir: str = "ai_images") -> str:
    os.makedirs(save_dir, exist_ok=True)
    client = get_openai_client(api_key)

    try:
        img = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard",
        )
    except Exception as e:
        msg = str(e)
        if "billing_hard_limit_reached" in msg or "Billing hard limit has been reached" in msg:
            raise RuntimeError(
                "На аккаунте OpenAI исчерпан платёжный лимит (billing hard limit).\n"
                "AI-картинки временно недоступны.\n"
                "Пополните баланс или укажите другой API-ключ."
            ) from e
        raise

    image_url = img.data[0].url
    import requests
    response = requests.get(image_url)
    image_bytes = response.content

    filename = f"ai_{int(time.time())}.png"
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)

    return path


def enrich_german_word_info(word: str, api_key: str | None):
    """
    Базовая инфа (перевод + IPA/род/мн.ч) через GPT,
    если ключ есть. Иначе – заглушки.
    IPA/род/мн.ч потом будут поверх заменены wiktionary, если удастся.
    """
    if not api_key or not OPENAI_LIB_AVAILABLE:
        return "", "", "?", "?"

    try:
        client = get_openai_client(api_key)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты словарный бот по немецкому как Wiktionary. "
                        "Для заданного немецкого слова дай ответ в формате:\n"
                        "translation_ru | ipa | gender | plural\n"
                        "gender один из: m, f, n.\n"
                        "Только одна строка, без пояснений."
                    ),
                },
                {"role": "user", "content": word},
            ],
            max_tokens=64,
        )
        line = resp.choices[0].message.content.strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            return "", "", "?", "?"
        return parts[0], parts[1], parts[2], parts[3]
    except Exception:
        return "", "", "?", "?"


# ==========================
# Вставка карточки
# ==========================
def insert_card(
    deck_id: int,
    front: str,
    back: str,
    front_image_path: str | None = None,
    back_image_path: str | None = None,
    audio_path: str | None = None,
    level: int = 1,
    note_id: int | None = None,
    template_ord: int | None = None,
    ensure_note: bool = False,
    note_fields: dict | None = None,
    audio_side: str = "back",
    audio_source: str | None = None,
):
    """
    Вставка карточки в БД.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Если deck_id не задан — берём первую колоду
    if deck_id is None:
        cur.execute("SELECT id FROM decks ORDER BY id LIMIT 1;")
        row = cur.fetchone()
        if row is None:
            conn.close()
            raise RuntimeError("Не выбрана колода...")
        deck_id = row["id"]

    next_dt = get_next_review_for_level(level).isoformat()

    # ВАЖНО: Проверяем, есть ли аудио-тег в тексте карточки
    # Если audio_path передан, используем его
    # Если нет, проверяем тег [audio:...] в back тексте
    
    actual_audio_path = audio_path
    if not actual_audio_path and "[audio:" in back:
        # Извлечь путь из тега [audio:path/to/file.wav]
        match = re.search(r'\[audio:(.+?)\]', back)
        if match:
            actual_audio_path = match.group(1)
            # Удалить тег из текста для чистого отображения
            back = re.sub(r'\[audio:.+?\]', '', back).strip()

    cur.execute(
        """
        INSERT INTO cards (deck_id, front, back, next_review, leitner_level,
                           front_image_path, back_image_path, audio_path,
                           note_id, template_ord, translation_shown, overview_added)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1);
    """,
        (
            deck_id,
            front,
            back,
            next_dt,
            level,
            front_image_path,
            back_image_path,
            actual_audio_path,
            note_id,
            template_ord,
        ),
    )

    card_id = cur.lastrowid

    stored_audio_path = actual_audio_path
    if actual_audio_path and os.path.exists(actual_audio_path):
        os.makedirs(MEDIA_FOLDER, exist_ok=True)
        ext = os.path.splitext(actual_audio_path)[1] or ".wav"
        filename = f"audio_card_{card_id}_{int(time.time())}{ext}"
        dest_path = os.path.join(MEDIA_FOLDER, filename)
        try:
            shutil.copy(actual_audio_path, dest_path)
            stored_audio_path = dest_path
            cur.execute("UPDATE cards SET audio_path = ? WHERE id = ?;", (stored_audio_path, card_id))
        except Exception:
            stored_audio_path = actual_audio_path

    conn.commit()
    conn.close()

    if stored_audio_path:
        attach_media_to_card(card_id, [(stored_audio_path, "audio", audio_side, audio_source)])

    if ensure_note and note_id is None:
        card_stub = {
            "id": card_id,
            "deck_id": deck_id,
            "front": front,
            "back": back,
            "leitner_level": level,
            "front_image_path": front_image_path,
            "back_image_path": back_image_path,
            "image_path": None,
        }
        ensure_note_for_card(
            card_stub,
            note_fields=note_fields,
            create_cards=True,
            skip_template_ords={template_ord} if template_ord is not None else {0},
        )

    return card_id

# ==========================
# Авто-генерация
# ==========================

def auto_generate_cards_from_text(deck_id: int,
                                  text: str,
                                  use_ai_images: bool,
                                  api_key: str | None,
                                  front_template: str,
                                  back_template: str,
                                  one_sentence_one_card: bool = False,
                                  audio_path: str | None = None,
                                  audio_source: str | None = None,
                                  audio_side: str = "back",
                                  progress_queue: queue.Queue | None = None,
                                  cancel_check=None) -> int:
    """
    Если one_sentence_one_card = True:
        1 предложение = 1 карточка (длинный текст -> много карточек).
    Если False:
        классический режим: каждое новое слово = карточка.
    """
    sentences = split_into_sentences(text)
    total_progress = len(sentences)
    if progress_queue is not None:
        progress_queue.put(("progress", 0, max(total_progress, 1), "Разбивка текста"))
    if not sentences:
        return 0

    known = get_known_words()
    created = 0
    all_new_words = set()

    def make_base_card(sentence: str,
                       target_word: str,
                       translation: str,
                       ipa: str,
                       gender: str,
                       plural: str,
                       wiki_data: dict):
        nonlocal created

        # делаем "дырку" в предложении
        sentence_with_gap = sentence
        if target_word:
            pattern = re.compile(re.escape(target_word), re.IGNORECASE)
            sentence_with_gap = pattern.sub("____", sentence, count=1)

        # если Wiktionary дал более точные IPA/род/мн.ч – подменяем
        ipa_final = wiki_data.get("ipa") or ipa
        gender_final = wiki_data.get("gender") or gender
        plural_final = wiki_data.get("plural") or plural

        # Получаем перевод всего предложения
        sentence_translation = translate_sentence(sentence, use_openai=True)

        front = front_template.format(
            translation="",  # Не показываем перевод на лицевой стороне
            sentence_with_gap=sentence_with_gap,
            word=target_word,
            ipa=ipa_final,
            gender=gender_final,
            plural=plural_final,
            sentence=sentence,
        )

        back = back_template.format(
            translation=sentence_translation,
            sentence_with_gap=sentence_with_gap,
            word=target_word,
            ipa=ipa_final,
            gender=gender_final,
            plural=plural_final,
            sentence=sentence,
        )

        # доп. блок с данными Wiktionary
        extra_parts = []
        if wiki_data.get("ipa"):
            extra_parts.append(f"IPA: {wiki_data['ipa']}")
        if wiki_data.get("gender") or wiki_data.get("plural"):
            extra_parts.append(
                f"Genus / Plural: {wiki_data.get('gender', '')} | {wiki_data.get('plural', '')}"
            )
        if wiki_data.get("declension"):
            extra_parts.append("Beugung / Formen:\n" + wiki_data["declension"])

        if extra_parts:
            back = back + "\n\n" + "\n".join(extra_parts)

        img_path_front = None
        if use_ai_images and api_key:
            try:
                img_prompt = f"Illustration for German sentence '{sentence}' with key word '{target_word}'"
                img_path_front = generate_image_with_openai(img_prompt, api_key)
            except Exception:
                img_path_front = None

        level_value = 1

        note_fields = {
            "word": target_word or sentence,
            "translation": translation,
            "example": sentence,
            "level": level_value,
            "image": img_path_front or "",
            "front_image_path": img_path_front or "",
            "back_image_path": None,
            "audio_path": audio_path,
            "audio_side": audio_side,
            "audio_source": audio_source,
            "front": front,
            "back": back,
        }

        _, cards_created = create_note_with_cards(
            deck_id,
            note_fields,
            note_type_id=ensure_generated_note_type_id(),
        )
        created += cards_created

        # доп. карточки: синонимы
        syns = wiki_data.get("synonyms") or []
        for syn in syns[:3]:
            front_syn = f"Synonym für {target_word}: ____"
            back_syn = syn
            syn_fields = {
                "word": syn,
                "translation": back_syn,
                "example": front_syn,
                "level": 1,
                "image": "",
                "front": front_syn,
                "back": back_syn,
                "audio_path": audio_path,
            }
            _, cards_created = create_note_with_cards(
                deck_id,
                syn_fields,
                note_type_id=ensure_generated_note_type_id(),
            )
            created += cards_created

        # доп. карточки: примеры
        examples = wiki_data.get("examples") or []
        for ex in examples[:3]:
            ex_sentence = ex
            pattern = re.compile(re.escape(target_word), re.IGNORECASE)
            ex_gap = pattern.sub("____", ex_sentence, count=1)
            front_ex = ex_gap
            back_ex = ex_sentence
            ex_fields = {
                "word": target_word or ex_sentence,
                "translation": back_ex,
                "example": front_ex,
                "level": 1,
                "image": "",
                "front": front_ex,
                "back": back_ex,
                "audio_path": audio_path,
            }
            _, cards_created = create_note_with_cards(
                deck_id,
                ex_fields,
                note_type_id=ensure_generated_note_type_id(),
            )
            created += cards_created

    if one_sentence_one_card:
        # 1 предложение = 1 карточка
        for sentence in sentences:
            words = extract_words_from_text(sentence)
            if not words:
                continue

            # первое новое слово в предложении, иначе просто первое
            target_word = None
            for w in words:
                if w not in known:
                    target_word = w
                    break
            if target_word is None:
                target_word = words[0]

            new_in_sentence = {w for w in words if w not in known}
            all_new_words.update(new_in_sentence)

            if cancel_check and cancel_check():
                break

            translation, ipa, gender, plural = enrich_german_word_info(target_word, api_key) \
                if target_word else ("", "", "?", "?")
            wiki_data = get_wiktionary_data(target_word) if target_word else {}

            make_base_card(sentence, target_word, translation, ipa, gender, plural, wiki_data)
            if progress_queue is not None:
                progress_queue.put(("progress", created, max(total_progress, 1), "Генерация предложений"))
    else:
        # старый режим: каждое новое слово = карточка
        all_words = extract_words_from_text(text)
        new_words = {w for w in all_words if w and w not in known}
        if not new_words:
            return 0

        total_progress = len(new_words)
        processed = 0
        for word in sorted(new_words):
            if cancel_check and cancel_check():
                break
            sentence_for_word = None
            for s in sentences:
                if normalize_word(word) in [normalize_word(w) for w in extract_words_from_text(s)]:
                    sentence_for_word = s
                    break
            if not sentence_for_word:
                sentence_for_word = text.strip()

            translation, ipa, gender, plural = enrich_german_word_info(word, api_key)
            wiki_data = get_wiktionary_data(word)

            make_base_card(sentence_for_word, word, translation, ipa, gender, plural, wiki_data)
            processed += 1
            if progress_queue is not None:
                progress_queue.put(("progress", processed, max(total_progress, 1), "Генерация слов"))

        all_new_words.update(new_words)

    add_new_words(all_new_words)
    if progress_queue is not None:
        progress_queue.put(("log", f"Создано карточек: {created}"))
    return created


def auto_generate_cards_from_image(deck_id: int,
                                   image_path: str,
                                   use_ai_images: bool,
                                   api_key: str | None,
                                   front_template: str,
                                   back_template: str,
                                   one_sentence_one_card: bool = False) -> int:
    if not OCR_AVAILABLE or not is_tesseract_available():
        raise RuntimeError("Tesseract OCR не настроен.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    if not text.strip():
        return 0

    return auto_generate_cards_from_text(
        deck_id, text, use_ai_images, api_key,
        front_template, back_template,
        one_sentence_one_card=one_sentence_one_card,
        audio_path=None
    )


def auto_generate_cards_from_speech(deck_id: int,
                                    duration_sec: int,
                                    use_ai_images: bool,
                                    api_key: str | None,
                                    front_template: str,
                                    back_template: str,
                                    mic_index: int | None,
                                    one_sentence_one_card: bool = False,
                                    progress_queue: queue.Queue | None = None,
                                    cancel_check=None) -> int:
    if not SR_AVAILABLE:
        raise RuntimeError("SpeechRecognition не установлен.")
    r = sr.Recognizer()

    if mic_index is not None:
        source = sr.Microphone(device_index=mic_index)
    else:
        source = sr.Microphone()

    if progress_queue is not None:
        progress_queue.put(("progress", 0, max(duration_sec, 1), "Запись"))

    with source as s:
        r.adjust_for_ambient_noise(s, duration=0.5)
        audio = r.record(s, duration=duration_sec)

    if cancel_check and cancel_check():
        return 0

    os.makedirs("recordings", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join("recordings", f"speech_{ts}.wav")
    with open(audio_path, "wb") as f:
        f.write(audio.get_wav_data())

    if progress_queue is not None:
        progress_queue.put(("progress", duration_sec, max(duration_sec, 1), "Распознавание"))

    try:
        text = r.recognize_google(audio, language="de-DE")
    except Exception as e:
        raise RuntimeError(f"Не удалось распознать речь: {e}")

    if cancel_check and cancel_check():
        return 0

    return auto_generate_cards_from_text(
        deck_id, text, use_ai_images, api_key,
        front_template, back_template,
        one_sentence_one_card=one_sentence_one_card,
        audio_path=audio_path,
        audio_source="digital_hearing",
        audio_side="back",
        progress_queue=progress_queue,
        cancel_check=cancel_check,
    )


def auto_generate_cards_from_video(deck_id: int,
                                   video_path: str,
                                   use_ai_images: bool,
                                   api_key: str | None,
                                   front_template: str,
                                   back_template: str) -> int:
    """
    Генерация карточек из видео: извлечение аудио, нарезка на предложения,
    распознавание речи, создание карточки с аудио.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy не установлен.")
    if not SR_AVAILABLE:
        raise RuntimeError("SpeechRecognition не установлен.")
    
    try:
        # Извлечь аудио из видео
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_audio = os.path.join(temp_dir, "extracted_audio.wav")
        
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(temp_audio)
        video.close()
        
        # Распознать речь из аудио
        r = sr.Recognizer()
        with sr.AudioFile(temp_audio) as source:
            audio_data = r.record(source)
            
        try:
            text = r.recognize_google(audio_data, language="de-DE")
        except Exception as e:
            raise RuntimeError(f"Не удалось распознать речь: {e}")
        
        # Сохранить аудио файл
        os.makedirs("video_audio", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = os.path.join("video_audio", f"video_{ts}.wav")
        
        # Копируем аудио файл
        import shutil
        shutil.copy(temp_audio, audio_filename)
        
        # Очистка временных файлов
        shutil.rmtree(temp_dir)
        
        # Генерировать карточки из текста
        return auto_generate_cards_from_text(
            deck_id, text, use_ai_images, api_key,
            front_template, back_template,
            one_sentence_one_card=True,
            audio_path=audio_filename,
            audio_source="digital_hearing",
            audio_side="back",
        )
        
    except Exception as e:
        raise RuntimeError(f"Ошибка обработки видео: {e}")


# ==========================
# TTS
# ==========================

def speak_text(text: str):
    if not TTS_AVAILABLE or not _tts_engine:
        messagebox.showwarning(
            "TTS недоступен",
            "pyttsx3 не установлен или не работает.\n"
            "Установи: pip install pyttsx3"
        )
        return
    _tts_engine.say(text)
    _tts_engine.runAndWait()


def play_audio_file(path):
    """Воспроизвести аудио файл"""
    audio_path = path or ""
    resolved = resolve_sound_file(audio_path)
    print("[AUDIO] requested=", audio_path)
    print("[AUDIO] resolved =", resolved, "exists=", os.path.exists(resolved) if resolved else False)

    if not resolved or not os.path.exists(resolved):
        msg = f"Файл аудио не найден: {audio_path}"
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "audio.log"), "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        messagebox.showerror("Аудио", msg)
        return

    try:
        try:
            import vlc  # type: ignore

            instance = vlc.Instance()
            player = instance.media_player_new()
            media = instance.media_new(resolved)
            player.set_media(media)
            player.play()
            return
        except Exception as exc:  # noqa: BLE001
            os.makedirs("logs", exist_ok=True)
            with open(os.path.join("logs", "audio.log"), "a", encoding="utf-8") as fh:
                fh.write(f"VLC недоступен: {exc}\n")

        if WINSOUND_AVAILABLE and resolved.lower().endswith(".wav"):
            winsound.PlaySound(resolved, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return

        if TTS_AVAILABLE:
            speak_text(os.path.splitext(os.path.basename(resolved))[0])
            return

        messagebox.showerror("Аудио", "Нет доступного аудио движка для воспроизведения")
    except Exception as exc:  # noqa: BLE001 - не допускаем незахваченных исключений
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "audio.log"), "a", encoding="utf-8") as fh:
            fh.write(f"Ошибка воспроизведения: {exc}\n")
        messagebox.showerror("Аудио", f"Ошибка воспроизведения: {exc}")

# ==========================
# Устройства записи
# ==========================

def detect_default_mic_index() -> int | None:
    if not SR_AVAILABLE:
        return None
    try:
        devices = sr.Microphone.list_microphone_names()
    except Exception:
        return None

    for i, name in enumerate(devices):
        if "CABLE" in name.upper():
            return i
    for i, name in enumerate(devices):
        u = name.upper()
        if "STEREO MIX" in u or "СТЕРЕО" in u or "WHAT U HEAR" in u:
            return i
    return None


# ==========================
# Панель форматирования текста
# ==========================

def attach_simple_toolbar(parent_frame: ttk.Frame, text_widget: tk.Text):
    def apply_tag(tag, **cfg):
        if tag not in text_widget.tag_names():
            text_widget.tag_configure(tag, **cfg)
        try:
            text_widget.tag_add(tag, "sel.first", "sel.last")
        except tk.TclError:
            pass

    bar = ttk.Frame(parent_frame)
    bar.pack(fill=tk.X, padx=10, pady=(2, 4))

    ttk.Label(bar, text="Форматирование:").pack(side=tk.LEFT)

    ttk.Button(
        bar, text="Подчёркивание",
        command=lambda: apply_tag("underline", underline=1)
    ).pack(side=tk.LEFT, padx=3)

    ttk.Button(
        bar, text="Красн. подчёрк.",
        command=lambda: apply_tag("red_underline", underline=1, foreground="red")
    ).pack(side=tk.LEFT, padx=3)

    ttk.Button(
        bar, text="Маркер",
        command=lambda: apply_tag("marker_yellow", background="yellow")
    ).pack(side=tk.LEFT, padx=3)

# ==========================
# Контекстное меню для текстовых полей
# ==========================

def create_context_menu(widget):
    """Создать контекстное меню для текстового виджета"""
    menu = tk.Menu(widget, tearoff=0)
    
    # Добавляем команды контекстного меню
    menu.add_command(label="Вырезать", 
                     command=lambda: widget.event_generate('<<Cut>>'))
    menu.add_command(label="Копировать", 
                     command=lambda: widget.event_generate('<<Copy>>'))
    menu.add_command(label="Вставить", 
                     command=lambda: widget.event_generate('<<Paste>>'))
    menu.add_separator()
    menu.add_command(label="Выбрать все", 
                     command=lambda: widget.tag_add('sel', '1.0', 'end'))
    
    # Привязываем контекстное меню к виджету
    widget.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))
    
    # Для Entry виджетов
    if isinstance(widget, tk.Entry):
        widget.bind("<Control-c>", lambda e: widget.event_generate('<<Copy>>'))
        widget.bind("<Control-v>", lambda e: widget.event_generate('<<Paste>>'))
        widget.bind("<Control-x>", lambda e: widget.event_generate('<<Cut>>'))
        widget.bind("<Control-a>", lambda e: widget.select_range(0, tk.END))
    
    # Для Text виджетов
    elif isinstance(widget, tk.Text):
        widget.bind("<Control-c>", lambda e: widget.event_generate('<<Copy>>'))
        widget.bind("<Control-v>", lambda e: widget.event_generate('<<Paste>>'))
        widget.bind("<Control-x>", lambda e: widget.event_generate('<<Cut>>'))
        widget.bind("<Control-a>", lambda e: widget.tag_add('sel', '1.0', 'end'))


# ==========================
# GUI
# ==========================

class ResizableImageLabel(tk.Label):
    """Label с изображением, которое можно масштабировать перетаскиванием за углы"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.original_image = None
        self.current_image = None
        self.image_path = None
        self.scale_factor = 1.0
        self.drag_data = {"x": 0, "y": 0, "item": None}
        
        # Привязываем события мыши
        self.bind("<ButtonPress-1>", self.start_drag)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.stop_drag)
        self.bind("<MouseWheel>", self.on_mousewheel)  # Для Windows
        self.bind("<Button-4>", self.on_mousewheel)    # Для Linux, scroll up
        self.bind("<Button-5>", self.on_mousewheel)    # Для Linux, scroll down
        
    def load_image(self, image_path):
        """Загрузить изображение"""
        self.image_path = image_path
        if image_path and os.path.exists(image_path) and PIL_AVAILABLE:
            try:
                self.original_image = Image.open(image_path)
                self.scale_factor = 1.0
                self.update_display()
                return True
            except Exception as e:
                self.config(text=f"(Ошибка загрузки: {str(e)[:50]})", image="")
                return False
        else:
            self.config(text="(Нет изображения)", image="")
            return False
    
    def update_display(self):
        """Обновить отображение изображения"""
        if self.original_image:
            # Вычисляем новые размеры
            width = int(self.original_image.width * self.scale_factor)
            height = int(self.original_image.height * self.scale_factor)
            
            # Масштабируем изображение
            resized_image = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(resized_image)
            self.config(image=self.current_image, text="")
    
    def start_drag(self, event):
        """Начать перетаскивание для масштабирования"""
        # Проверяем, нажали ли на угол изображения (последние 20 пикселей)
        if self.current_image:
            width = self.current_image.width()
            height = self.current_image.height()
            
            # Проверяем правый нижний угол
            if (width - 20 <= event.x <= width and 
                height - 20 <= event.y <= height):
                self.drag_data["x"] = event.x
                self.drag_data["y"] = event.y
                self.drag_data["item"] = "resize"
                
    def drag(self, event):
        """Обработка перетаскивания"""
        if self.drag_data["item"] == "resize" and self.original_image:
            # Вычисляем изменение размера
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            # Масштабируем на основе большего изменения
            if abs(dx) > abs(dy):
                delta = dx / self.original_image.width
            else:
                delta = dy / self.original_image.height
            
            # Применяем масштаб
            self.scale_factor = max(0.1, min(3.0, self.scale_factor + delta * 0.5))
            
            # Обновляем изображение
            self.update_display()
            
            # Обновляем позицию для следующего события
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
    
    def stop_drag(self, event):
        """Завершить перетаскивание"""
        self.drag_data["item"] = None
    
    def on_mousewheel(self, event):
        """Обработка колесика мыши для масштабирования"""
        if self.original_image:
            # Определяем направление масштабирования
            if event.num == 5 or event.delta < 0:  # Вниз или от пользователя
                self.scale_factor = max(0.1, self.scale_factor * 0.9)
            elif event.num == 4 or event.delta > 0:  # Вверх или к пользователю
                self.scale_factor = min(3.0, self.scale_factor * 1.1)
            
            # Обновляем изображение
            self.update_display()
        
        return "break"


class AnkiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Leitner Anki (цифровой слух + Wiktionary + AI + русско-немецкий словарь)")
        self.geometry("1000x700")

        self.root = self

        self.decks = []
        self.selected_deck_id = None
        self.selected_phase = None

        # Шаблоны фронта/ответа
        self.front_template = DEFAULT_FRONT_TEMPLATE
        self.back_template = DEFAULT_BACK_TEMPLATE

        # Иконки для колод
        self.deck_icons = {}
        self.deck_preview_images = {}

        global MIC_DEVICE_INDEX
        MIC_DEVICE_INDEX = detect_default_mic_index()
        self.microphone_index = MIC_DEVICE_INDEX

        self.overdue_canvas = None
        self.overdue_badge_text_id = None
        self.phase_badge_manager: PhaseOverdueBadges | None = None
        self.overdue_update_job = None
        self._overdue_task_running = False
        self._deck_select_job: str | None = None

        self._refresh_in_progress = False

        self.busy_dialog = BusyDialog(self)
        self.task_runner = TaskRunner(self, self.busy_dialog)

        self.image_import_watch_job = None

        self._bg_listeners: list[tuple[queue.Queue, callable]] = []

        # отображение id-элементов Treeview -> (deck_id, phase)
        self.deck_items = {}

        # Инициализируем словарь
        init_dictionary()

        self.create_menu()
        self.create_widgets()
        self.refresh_decks()

        self.after(500, self.warn_if_no_tesseract)
        self.after(50, self.poll_bg_queues)

    # --------- предупреждение ---------

    def warn_if_no_tesseract(self):
        if not is_tesseract_available():
            messagebox.showwarning(
                "Tesseract не найден",
                "Режим генерации из изображений (OCR) пока недоступен.\n"
                "Установи Tesseract OCR или пропиши путь."
            )

    def register_bg_handler(self, queue_obj: queue.Queue, handler):
        self._bg_listeners.append((queue_obj, handler))

    def unregister_bg_handler(self, queue_obj: queue.Queue):
        self._bg_listeners = [item for item in self._bg_listeners if item[0] is not queue_obj]

    def poll_bg_queues(self):
        for q_obj, handler in list(self._bg_listeners):
            while True:
                try:
                    event = q_obj.get_nowait()
                except queue.Empty:
                    break
                try:
                    handler(event)
                except Exception:
                    pass
        self.after(50, self.poll_bg_queues)

    def run_task(self, title: str, mode: str, task_fn, on_success, on_error, total=None):
        self.task_runner.run_task(title, mode, task_fn, on_success, on_error, total)

    # --------- меню ---------

    def create_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Импорт Anki (.apkg)", command=self.open_apkg_import_window)
        file_menu.add_separator()
        file_menu.add_command(label="Новая колода", command=self.add_deck_window)
        file_menu.add_command(label="Редактировать колоду", command=self.edit_deck_window)
        file_menu.add_command(label="Удалить выбранную колоду", command=self.delete_selected_deck)
        menubar.add_cascade(label="Файл", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="OpenAI API ключ", command=self.open_settings_window)
        settings_menu.add_command(label="Аудиоустройство (цифровой слух)",
                                  command=self.open_audio_device_window)
        settings_menu.add_command(label="Настройки перевода",
                                  command=self.open_translation_settings_window)
        settings_menu.add_command(label="Управление словарями",
                                  command=self.open_dictionary_manager_window)
        menubar.add_cascade(label="Настройки", menu=settings_menu)

        gen_menu = tk.Menu(menubar, tearoff=0)
        gen_menu.add_command(label="Генерация из текста…", command=self.open_generate_from_text_window)
        gen_menu.add_command(label="Генерация из изображения (OCR)…",
                             command=self.open_generate_from_image_window)
        gen_menu.add_command(label="Генерация через цифровой слух…",
                             command=self.open_generate_from_speech_window)
        gen_menu.add_command(label="Генерация из видео (цифровой слух)…",
                             command=self.open_generate_from_video_window)
        gen_menu.add_command(label="Видео → клипы → карточки",
                             command=self.open_video_clip_window)
        gen_menu.add_command(label="Импорт картинок по ID (CSV)…",
                             command=self.open_image_id_import_window)
        gen_menu.add_command(label="Импорт CSV колоды", command=self.open_csv_import_window)
        gen_menu.add_command(label="Картинки по CSV (Wikimedia)",
                             command=self.open_wikimedia_csv_window)
        menubar.add_cascade(label="Режим генерации", menu=gen_menu)

        modes_menu = tk.Menu(menubar, tearoff=0)
        modes_menu.add_command(label="Режим повторения (по дате)",
                               command=self.start_repeat_mode)
        modes_menu.add_command(label="Режим воспроизведения (по прогрессу)",
                               command=self.start_playback_mode)
        modes_menu.add_command(label="Режим обзора / редактирования",
                               command=self.show_cards_window)
        modes_menu.add_command(label="Режим ознакомления",
                               command=self.start_overview_mode)
        menubar.add_cascade(label="Режимы", menu=modes_menu)

        # Добавляем меню статистики
        stats_menu = tk.Menu(menubar, tearoff=0)
        stats_menu.add_command(label="Показать статистику", command=self.show_statistics_window)
        stats_menu.add_command(label="Статистика словаря", command=self.show_dictionary_stats_window)
        menubar.add_cascade(label="Статистика", menu=stats_menu)

        self.config(menu=menubar)
    
    def open_generate_from_video_window(self):
        """Открыть окно генерации из видео"""
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
            
        # Выбрать видео файл
        filetypes = [
            ("Видео файлы", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("Все файлы", "*.*"),
        ]
        
        video_path = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=filetypes
        )

        if not video_path:
            return

        # Открыть окно видео-редактора
        self.video_editor_window = VideoEditorWindow(self.root, video_path, self.selected_deck_id)

    def open_video_clip_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        win = tk.Toplevel(self)
        win.title("Видео → клипы → карточки")
        win.geometry("520x260")
        win.grab_set()

        video_path_var = tk.StringVar()
        start_var = tk.StringVar(value="00:00:00")
        end_var = tk.StringVar(value="00:00:10")

        ttk.Label(win, text="Видео файл:").pack(anchor="w", padx=10, pady=(10, 0))
        video_frame = ttk.Frame(win)
        video_frame.pack(fill=tk.X, padx=10)

        entry_video = ttk.Entry(video_frame, textvariable=video_path_var)
        entry_video.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def browse_video():
            path = filedialog.askopenfilename(
                title="Выберите видео файл",
                filetypes=[("Видео", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv"), ("Все файлы", "*.*")],
            )
            if path:
                video_path_var.set(path)

        ttk.Button(video_frame, text="…", width=4, command=browse_video).pack(side=tk.LEFT, padx=(5, 0))

        time_frame = ttk.Frame(win)
        time_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(time_frame, text="Начало (HH:MM:SS):").grid(row=0, column=0, sticky="w")
        ttk.Label(time_frame, text="Конец (HH:MM:SS):").grid(row=1, column=0, sticky="w", pady=(5, 0))

        ttk.Entry(time_frame, textvariable=start_var, width=12).grid(row=0, column=1, padx=5, sticky="w")
        ttk.Entry(time_frame, textvariable=end_var, width=12).grid(row=1, column=1, padx=5, sticky="w", pady=(5, 0))

        status_var = tk.StringVar(value="Клип сохраняется в папку media/")
        ttk.Label(win, textvariable=status_var, foreground="gray").pack(anchor="w", padx=10)

        def cut_and_attach():
            video_path = video_path_var.get().strip()
            if not video_path:
                messagebox.showerror("Ошибка", "Выберите видео файл.")
                return

            ok, result = cut_video_clip(video_path, start_var.get(), end_var.get(), MEDIA_FOLDER)
            if not ok:
                messagebox.showerror("FFmpeg", result)
                return

            clip_path = result
            clip_name = os.path.basename(clip_path)
            clip_info = f"{start_var.get()} → {end_var.get()}"

            note_fields = {
                "word": clip_name,
                "translation": "",
                "example": clip_info,
                "level": 1,
                "image": "",
                "front": f"🎬 Клип {clip_info}\n{clip_name}",
                "back": f"Смотри клип {clip_info}\n{clip_name}",
                "front_image_path": None,
                "back_image_path": None,
                "audio_path": None,
            }

            note_id, _ = create_note_with_cards(
                self.selected_deck_id,
                note_fields,
                note_type_id=ensure_generated_note_type_id(),
                tags="video clip",
            )
            attach_media_to_note(note_id, [(clip_path, "video")])
            status_var.set(f"Создан клип {clip_name}")
            messagebox.showinfo("Готово", f"Клип сохранен в {clip_path}\nКарточка добавлена в колоду.")

        ttk.Button(win, text="Нарезать клип", command=cut_and_attach).pack(anchor="e", padx=10, pady=10)

    def open_apkg_import_window(self):
        win = tk.Toplevel(self)
        win.title("Импорт Anki (.apkg)")
        win.geometry("720x520")
        win.grab_set()

        file_var = tk.StringVar()
        keep_schedule_var = tk.BooleanVar(value=False)
        restart_var = tk.BooleanVar(value=True)
        media_var = tk.BooleanVar(value=True)
        revlog_var = tk.BooleanVar(value=False)

        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="0/0")
        processing_task: dict[str, BackgroundTask | None] = {"task": None}

        def log_msg(msg: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, msg + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        def handle_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                progress_bar.configure(maximum=max(int(total), 1))
                progress_var.set(int(done))
                progress_label_var.set(f"{done}/{total} {label}")
            elif kind == "log":
                log_msg(str(event[1]))
            elif kind == "done":
                summary = event[1] or {}
                if processing_task["task"]:
                    self.unregister_bg_handler(processing_task["task"].queue)
                    processing_task["task"] = None
                messagebox.showinfo(
                    "Импорт завершен",
                    (
                        f"Заметок: {summary.get('notes', 0)}\n"
                        f"Карточек: {summary.get('cards', 0)}\n"
                        f"Медиа: {summary.get('media', 0)}\n"
                        f"Ошибок: {summary.get('errors', 0)}"
                    ),
                )
                self.refresh_decks()
            elif kind == "error":
                if processing_task["task"]:
                    self.unregister_bg_handler(processing_task["task"].queue)
                    processing_task["task"] = None
                msg = str(event[1])
                if "Неподдерживаемый формат Anki пакета" in msg:
                    messagebox.showerror("Неподдерживаемый формат", msg)
                else:
                    messagebox.showerror("Ошибка импорта", msg)

        def browse_file():
            path = filedialog.askopenfilename(filetypes=[("Anki .apkg", "*.apkg"), ("Все файлы", "*.*")])
            if path:
                file_var.set(path)

        def sync_modes():
            if keep_schedule_var.get():
                restart_var.set(False)
            elif not restart_var.get():
                restart_var.set(True)

        def start_import():
            if processing_task["task"]:
                messagebox.showinfo("Занято", "Импорт уже выполняется")
                return

            apkg_path = file_var.get().strip()
            if not apkg_path or not os.path.exists(apkg_path):
                messagebox.showerror("Ошибка", "Укажите существующий .apkg файл")
                return

            options = {
                "keep_schedule": keep_schedule_var.get(),
                "start_fresh": restart_var.get(),
                "import_media": media_var.get(),
                "import_revlog": revlog_var.get(),
            }

            def worker(task_obj: BackgroundTask):
                return import_apkg(
                    apkg_path,
                    target_parent_deck_id=self.selected_deck_id,
                    options=options,
                    progress_cb=lambda *ev: task_obj.queue.put(ev),
                    cancel_flag=task_obj.cancelled,
                )

            task = start_background_task(worker)
            processing_task["task"] = task
            self.register_bg_handler(task.queue, handle_event)
            log_msg("Старт импорта…")

        def cancel_import():
            if processing_task["task"]:
                processing_task["task"].cancel()
                log_msg("Отмена запрошена")

        top_frame = ttk.Frame(win)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(top_frame, text="Файл .apkg:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top_frame, textvariable=file_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(top_frame, text="…", width=4, command=browse_file).grid(row=0, column=2)
        top_frame.columnconfigure(1, weight=1)

        options_frame = ttk.LabelFrame(win, text="Опции")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Checkbutton(options_frame, text="Сохранить расписание Anki", variable=keep_schedule_var, command=sync_modes).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Начать заново в SRS", variable=restart_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Импортировать медиа", variable=media_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Импортировать revlog (позже)", variable=revlog_var).grid(row=3, column=0, sticky="w")

        progress_bar = ttk.Progressbar(win, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, padx=10, pady=(10, 0))
        ttk.Label(win, textvariable=progress_label_var).pack(anchor="e", padx=10)

        log_box = tk.Text(win, height=12, state="disabled")
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Импортировать", command=start_import).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=cancel_import).pack(side=tk.RIGHT)

    def open_image_id_import_window(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow не установлен", "Для режима импорта картинок нужен пакет Pillow")
            return

        win = tk.Toplevel(self)
        win.title("Импорт картинок по ID (CSV)")
        win.geometry("820x680")
        win.grab_set()

        # variables
        csv_path_var = tk.StringVar()
        folder_var = tk.StringVar()
        deck_var = tk.StringVar()
        note_type_var = tk.StringVar()
        id_mode_var = tk.StringVar(value="filename")
        update_existing_var = tk.BooleanVar(value=False)
        move_files_var = tk.BooleanVar(value=False)
        watch_var = tk.BooleanVar(value=False)
        auto_confirm_var = tk.BooleanVar(value=True)
        interval_var = tk.IntVar(value=5)
        stop_flag = {"stop": False}

        decks = list_decks()
        if self.selected_deck_id:
            deck_var.set(str(self.selected_deck_id))
        elif decks:
            deck_var.set(str(decks[0]["id"]))

        note_types = list_note_types()
        if note_types:
            note_type_var.set(str(note_types[0]["id"]))
        else:
            note_type_var.set(str(ensure_basic_note_type_id()))

        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="")
        processing_task = {"task": None}
        running_watch_mode = {"value": False}

        def log_msg(msg: str):
            log_text.configure(state="normal")
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
            log_text.configure(state="disabled")

        def handle_import_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                progress_bar.configure(maximum=max(total, 1))
                progress_var.set(done)
                progress_label_var.set(f"{label}: {done}/{total}")
            elif kind == "log":
                log_msg(event[1])
            elif kind == "done":
                summary = event[1] or {"total": 0, "imported": 0, "updated": 0, "skipped": 0, "errors": 0}
                self.unregister_bg_handler(processing_task["task"].queue)
                processing_task["task"] = None
                btn_check.config(state=tk.NORMAL)
                btn_import.config(state=tk.NORMAL)
                btn_stop.config(state=tk.NORMAL)
                msg = (
                    f"Всего: {summary['total']}\n"
                    f"Создано: {summary['imported']}\n"
                    f"Обновлено: {summary['updated']}\n"
                    f"Пропущено: {summary['skipped']}\n"
                    f"Ошибки: {summary['errors']}"
                )
                messagebox.showinfo("Импорт завершен", msg)
                if running_watch_mode["value"] and watch_var.get() and not stop_flag.get("stop"):
                    self.image_import_watch_job = win.after(interval_var.get() * 1000, lambda: process_files(False, True))
            elif kind == "error":
                self.unregister_bg_handler(processing_task["task"].queue)
                processing_task["task"] = None
                btn_check.config(state=tk.NORMAL)
                btn_import.config(state=tk.NORMAL)
                btn_stop.config(state=tk.NORMAL)
                messagebox.showerror("Ошибка", event[1])

        def browse_csv():
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
            if path:
                csv_path_var.set(path)

        def browse_folder():
            path = filedialog.askdirectory()
            if path:
                folder_var.set(path)

        def ask_semi_confirmation(image_path: str, detected_id: int | None, csv_entry: dict | None):
            result = {"action": "skip", "data": None, "auto": auto_confirm_var.get()}
            if auto_confirm_var.get() and detected_id is not None and csv_entry:
                result["action"] = "create"
                result["data"] = {
                    "id": detected_id,
                    "word": csv_entry.get("word", ""),
                    "translation": csv_entry.get("translation", ""),
                    "example": csv_entry.get("example", ""),
                    "level": csv_entry.get("level", ""),
                }
                return result

            dialog = tk.Toplevel(win)
            dialog.title("Подтверждение импорта")
            dialog.geometry("640x520")
            dialog.grab_set()

            img_label = tk.Label(dialog)
            img_label.pack(pady=5)

            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    img.thumbnail((360, 260))
                    photo = ImageTk.PhotoImage(img)
                    img_label.configure(image=photo)
                    img_label.image = photo
                except Exception:
                    img_label.configure(text="(не удалось загрузить картинку)")

            form = ttk.Frame(dialog)
            form.pack(fill=tk.X, padx=10, pady=10)

            id_var = tk.StringVar(value=str(detected_id) if detected_id is not None else "")
            word_var = tk.StringVar(value=(csv_entry or {}).get("word", ""))
            tr_var = tk.StringVar(value=(csv_entry or {}).get("translation", ""))
            ex_var = tk.StringVar(value=(csv_entry or {}).get("example", ""))
            lvl_var = tk.StringVar(value=(csv_entry or {}).get("level", ""))
            auto_apply_var = tk.BooleanVar(value=auto_confirm_var.get())

            for idx, (lbl, var) in enumerate([
                ("ID", id_var),
                ("Word", word_var),
                ("Translation", tr_var),
                ("Example", ex_var),
                ("Level", lvl_var),
            ]):
                ttk.Label(form, text=lbl + ":").grid(row=idx, column=0, sticky="w", pady=3)
                ttk.Entry(form, textvariable=var).grid(row=idx, column=1, sticky="ew", padx=5)
            form.columnconfigure(1, weight=1)

            ttk.Checkbutton(dialog, text="Автоподтверждать далее", variable=auto_apply_var).pack(anchor="w", padx=10)

            actions = ttk.Frame(dialog)
            actions.pack(fill=tk.X, pady=10)

            result_var = tk.StringVar(value="skip")

            def do_create():
                result_var.set("create")
                dialog.destroy()

            def do_skip():
                result_var.set("skip")
                dialog.destroy()

            def do_stop():
                result_var.set("stop")
                dialog.destroy()

            ttk.Button(actions, text="Создать/Обновить", command=do_create).pack(side=tk.LEFT, padx=5)
            ttk.Button(actions, text="Пропустить", command=do_skip).pack(side=tk.LEFT, padx=5)
            ttk.Button(actions, text="Стоп", command=do_stop).pack(side=tk.LEFT, padx=5)

            dialog.bind("<Return>", lambda e: do_create())
            dialog.bind("<Escape>", lambda e: do_skip())

            dialog.wait_variable(result_var)

            result["action"] = result_var.get()
            result["data"] = {
                "id": int(id_var.get()) if id_var.get().isdigit() else None,
                "word": word_var.get().strip(),
                "translation": tr_var.get().strip(),
                "example": ex_var.get().strip(),
                "level": lvl_var.get().strip(),
            }
            result["auto"] = auto_apply_var.get()
            auto_confirm_var.set(auto_apply_var.get())
            return result

        def process_files(dry_run=False, watch_mode=False):
            if processing_task["task"]:
                messagebox.showinfo("Занято", "Импорт уже выполняется")
                return
            stop_flag["stop"] = False
            running_watch_mode["value"] = watch_mode
            csv_path = csv_path_var.get().strip()
            folder = folder_var.get().strip()
            if not csv_path or not os.path.exists(csv_path):
                messagebox.showerror("Ошибка", "Укажите CSV-файл")
                return
            if not folder or not os.path.isdir(folder):
                messagebox.showerror("Ошибка", "Укажите папку с изображениями")
                return
            try:
                csv_data = read_csv_dictionary(csv_path)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                return

            deck_id = deck_var.get()
            if not deck_id:
                messagebox.showerror("Ошибка", "Выберите колоду")
                return
            deck_id_int = int(deck_id)

            try:
                note_type_id = int(note_type_var.get())
            except ValueError:
                note_type_id = ensure_basic_note_type_id()

            if id_mode_var.get() == "semi":
                messagebox.showwarning("Режим semi", "Полуавто режим не поддерживает фоновый импорт")
                return

            imported_files = get_imported_files() if watch_mode else set()
            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ]
            files.sort()

            def worker(task_obj):
                summary = {"total": 0, "imported": 0, "updated": 0, "skipped": 0, "errors": 0}
                total_files = len(files)
                for idx, fp in enumerate(files, start=1):
                    if stop_flag.get("stop") or task_obj.cancelled():
                        break
                    if watch_mode and fp in imported_files:
                        continue

                    summary["total"] += 1
                    id_val = None
                    if id_mode_var.get() == "filename":
                        id_val = extract_id_from_filename(fp)
                    if id_val is None and id_mode_var.get() == "ocr":
                        id_val = extract_id_with_ocr(fp)

                    if id_val is None:
                        task_obj.queue.put(("log", f"[ошибка] Не найден ID для {os.path.basename(fp)}"))
                        summary["errors"] += 1
                        continue

                    entry = csv_data.get(id_val) if id_val is not None else None

                    if entry is None:
                        task_obj.queue.put(("log", f"[пропуск] ID {id_val} отсутствует в CSV"))
                        summary["skipped"] += 1
                        continue

                    existing = find_note_by_source_id(deck_id_int, id_val)
                    if existing and not update_existing_var.get():
                        task_obj.queue.put(("log", f"[пропуск] ID {id_val} уже существует"))
                        summary["skipped"] += 1
                        continue

                    media_path = copy_image_to_media(fp, id_val, move_files_var.get()) if not dry_run else fp

                    fields = {
                        "word": entry.get("word", ""),
                        "translation": entry.get("translation", ""),
                        "example": entry.get("example", ""),
                        "level": entry.get("level", ""),
                        "image": media_path,
                        "source_id": str(id_val),
                    }

                    if dry_run:
                        task_obj.queue.put(("log", f"[проверка] готова карточка ID {id_val}: {fields['word']} -> {fields['translation']}"))
                        summary["imported"] += 1
                        task_obj.queue.put(("progress", idx, max(total_files, 1), "Импорт"))
                        continue

                    if existing:
                        try:
                            conn = get_connection()
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE notes SET fields_json = ? WHERE id = ?;",
                                (json.dumps(fields, ensure_ascii=False), existing["id"]),
                            )
                            conn.commit()
                            conn.close()
                            attach_media_to_note(existing["id"], [(media_path, "image")])
                            summary["updated"] += 1
                            task_obj.queue.put(("log", f"[обновлено] ID {id_val}"))
                        except Exception as e:
                            summary["errors"] += 1
                            task_obj.queue.put(("log", f"[ошибка] Не удалось обновить ID {id_val}: {e}"))
                    else:
                        try:
                            _, created_cards = create_note_with_cards(
                                deck_id_int,
                                fields,
                                note_type_id=note_type_id,
                                tags="import:image_id",
                            )
                            summary["imported"] += 1
                            task_obj.queue.put(("log", f"[создано] ID {id_val} (карточек: {created_cards})"))
                        except Exception as e:
                            summary["errors"] += 1
                            task_obj.queue.put(("log", f"[ошибка] Не удалось создать ID {id_val}: {e}"))

                    if watch_mode:
                        log_imported_file(fp)

                    task_obj.queue.put(("progress", idx, max(total_files, 1), "Импорт"))

                return summary

            progress_var.set(0)
            progress_label_var.set("")
            btn_check.config(state=tk.DISABLED)
            btn_import.config(state=tk.DISABLED)
            btn_stop.config(state=tk.DISABLED)
            processing_task["task"] = start_background_task(worker)
            self.register_bg_handler(processing_task["task"].queue, handle_import_event)

        def stop_processing():
            stop_flag["stop"] = True
            if processing_task["task"]:
                processing_task["task"].cancel()
            btn_check.config(state=tk.NORMAL)
            btn_import.config(state=tk.NORMAL)
            btn_stop.config(state=tk.NORMAL)
            if self.image_import_watch_job:
                try:
                    win.after_cancel(self.image_import_watch_job)
                except Exception:
                    pass
                self.image_import_watch_job = None
            log_msg("[остановлено] Импорт остановлен пользователем")

        def start_watch_loop():
            stop_flag["stop"] = False
            process_files(dry_run=False, watch_mode=True)

        # Layout
        main_frame = ttk.Frame(win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = ttk.LabelFrame(main_frame, text="Исходные данные")
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Label(file_frame, text="CSV словарь:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(file_frame, textvariable=csv_path_var, width=60).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="Выбрать", command=browse_csv).grid(row=0, column=2, padx=5)

        ttk.Label(file_frame, text="Папка с картинками:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(file_frame, textvariable=folder_var, width=60).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="Выбрать", command=browse_folder).grid(row=1, column=2, padx=5)
        file_frame.columnconfigure(1, weight=1)

        options_frame = ttk.LabelFrame(main_frame, text="Настройки")
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Колода:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        deck_combo = ttk.Combobox(options_frame, textvariable=deck_var, state="readonly")
        deck_combo['values'] = [str(d["id"]) for d in decks]
        deck_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=3)

        ttk.Label(options_frame, text="Note type:").grid(row=0, column=2, sticky="w", padx=5)
        note_combo = ttk.Combobox(options_frame, textvariable=note_type_var, state="readonly")
        note_combo['values'] = [str(nt["id"]) for nt in note_types]
        note_combo.grid(row=0, column=3, sticky="ew", padx=5, pady=3)

        ttk.Label(options_frame, text="Режим ID:").grid(row=1, column=0, sticky="w", padx=5)
        ttk.Radiobutton(options_frame, text="Имя файла", variable=id_mode_var, value="filename").grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(options_frame, text="OCR", variable=id_mode_var, value="ocr").grid(row=1, column=2, sticky="w")
        ttk.Radiobutton(options_frame, text="Полуавто", variable=id_mode_var, value="semi").grid(row=1, column=3, sticky="w")

        ttk.Checkbutton(options_frame, text="Обновлять существующие", variable=update_existing_var).grid(row=2, column=0, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(options_frame, text="Перемещать файлы", variable=move_files_var).grid(row=2, column=1, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(options_frame, text="Следить за папкой (watch)", variable=watch_var).grid(row=2, column=2, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(options_frame, text="Автоподтверждать при совпадении (semi)", variable=auto_confirm_var).grid(row=2, column=3, sticky="w", padx=5, pady=3)

        ttk.Label(options_frame, text="Интервал watch (сек)").grid(row=3, column=0, sticky="w", padx=5)
        ttk.Spinbox(options_frame, from_=2, to=60, textvariable=interval_var, width=5).grid(row=3, column=1, sticky="w", padx=5)
        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(3, weight=1)

        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, expand=True, padx=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=5)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        btn_check = ttk.Button(btn_frame, text="Проверить", command=lambda: process_files(dry_run=True))
        btn_check.pack(side=tk.LEFT, padx=5)
        btn_import = ttk.Button(btn_frame, text="Импортировать", command=lambda: (start_watch_loop() if watch_var.get() else process_files(dry_run=False)))
        btn_import.pack(side=tk.LEFT, padx=5)
        btn_stop = ttk.Button(btn_frame, text="Остановить", command=stop_processing)
        btn_stop.pack(side=tk.LEFT, padx=5)

        log_text = tk.Text(main_frame, height=15, state="disabled")
        log_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def open_wikimedia_csv_window(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow не установлен", "Для режима нужен пакет Pillow")
            return

        win = tk.Toplevel(self)
        win.title("Картинки по CSV (Wikimedia)")
        win.geometry("720x520")
        win.grab_set()

        csv_path_var = tk.StringVar()
        status_var = tk.StringVar(value="Готово")
        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="0/0")

        task_holder: dict[str, BackgroundTask | None] = {"task": None}

        def log_msg(msg: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, msg + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        def handle_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                progress_bar.configure(maximum=max(total, 1))
                progress_var.set(done)
                progress_label_var.set(f"{done}/{total}")
                status_var.set(label)
                log_msg(label)
            elif kind == "done":
                summary = event[1] or {}
                status_var.set("Готово")
                progress_label_var.set(f"{summary.get('done', 0)}/{summary.get('total', 0)}")
                log_msg("Завершено")
                _finish_task()
            elif kind == "cancelled":
                status_var.set("Отменено пользователем")
                log_msg("[остановлено] Пользователь отменил загрузку")
                _finish_task()
            elif kind == "error":
                status_var.set(event[1])
                messagebox.showerror("Ошибка", event[1])
                _finish_task()

        def _finish_task():
            if task_holder["task"]:
                self.unregister_bg_handler(task_holder["task"].queue)
                task_holder["task"] = None
            btn_start.config(state=tk.NORMAL)
            btn_cancel.config(state=tk.DISABLED)

        def browse_csv():
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
            if path:
                csv_path_var.set(path)

        def start_fetch():
            if task_holder["task"]:
                return
            csv_path = csv_path_var.get().strip()
            if not csv_path:
                messagebox.showerror("Ошибка", "Укажите CSV-файл")
                return
            status_var.set("Начинаю загрузку…")
            progress_var.set(0)
            progress_label_var.set("0/0")

            from wiki_image_fetcher import process_csv_file

            def worker(task_obj: BackgroundTask, csv_file: str):
                def progress_cb(done: int, total: int, label: str):
                    task_obj.queue.put(("progress", done, total, label))

                try:
                    done, total = process_csv_file(
                        csv_file,
                        images_dir="images",
                        stop_checker=task_obj.cancelled,
                        progress_callback=progress_cb,
                    )
                    if task_obj.cancelled():
                        task_obj.queue.put(("cancelled",))
                    else:
                        task_obj.queue.put(("done", {"done": done, "total": total}))
                except Exception as exc:  # noqa: BLE001
                    task_obj.queue.put(("error", str(exc)))

            task = start_background_task(worker, csv_path)
            task_holder["task"] = task
            self.register_bg_handler(task.queue, handle_event)
            btn_start.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.NORMAL)

        def cancel_fetch():
            if task_holder["task"]:
                task_holder["task"].cancel()
                status_var.set("Остановка…")
                btn_cancel.config(state=tk.DISABLED)

        def on_close():
            cancel_fetch()
            if task_holder["task"]:
                self.unregister_bg_handler(task_holder["task"].queue)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

        top_frame = ttk.Frame(win)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="CSV файл:").grid(row=0, column=0, sticky="w", padx=5)
        entry = ttk.Entry(top_frame, textvariable=csv_path_var)
        entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(top_frame, text="Выбрать", command=browse_csv).grid(row=0, column=2, padx=5)
        top_frame.columnconfigure(1, weight=1)

        progress_frame = ttk.Frame(win)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, expand=True, padx=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=5)
        ttk.Label(progress_frame, textvariable=status_var, foreground="gray").pack(anchor="w", padx=5)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        btn_start = ttk.Button(btn_frame, text="Старт", command=start_fetch)
        btn_start.pack(side=tk.LEFT, padx=5)
        btn_cancel = ttk.Button(btn_frame, text="Отмена", command=cancel_fetch, state=tk.DISABLED)
        btn_cancel.pack(side=tk.LEFT, padx=5)

        log_box = tk.Text(win, height=15, state="disabled")
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def open_csv_import_window(self):
        win = tk.Toplevel(self)
        win.title("Импорт CSV колоды")
        win.geometry("880x740")
        win.grab_set()

        file_var = tk.StringVar()
        deck_var = tk.StringVar()
        has_headers_var = tk.BooleanVar(value=True)
        upsert_var = tk.BooleanVar(value=True)
        skip_existing_var = tk.BooleanVar(value=False)
        generate_id_var = tk.BooleanVar(value=True)
        attach_images_var = tk.BooleanVar(value=True)
        preserve_srs_var = tk.BooleanVar(value=False)
        reset_srs_var = tk.BooleanVar(value=False)
        start_state_var = tk.StringVar(value="new")
        start_phase_var = tk.IntVar(value=1)
        images_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "images"))

        id_col_var = tk.IntVar(value=1)
        word_col_var = tk.IntVar(value=2)
        translation_col_var = tk.IntVar(value=3)
        example_col_var = tk.IntVar(value=4)
        notes_col_var = tk.IntVar(value=5)
        tags_col_var = tk.IntVar(value=6)
        front_col_var = tk.IntVar(value=1)
        back_col_var = tk.IntVar(value=2)

        decks = list_decks()
        if decks:
            deck_var.set(str(decks[0]["id"]))

        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="0/0")
        processing_task: dict[str, BackgroundTask | None] = {"task": None}

        def refresh_decks_list():
            self.refresh_decks()
            decks_local = list_decks()
            deck_box["values"] = [f"{d['id']}: {d['name']}" for d in decks_local]
            if self.selected_deck_id:
                deck_var.set(str(self.selected_deck_id))
            elif decks_local:
                deck_var.set(str(decks_local[0]["id"]))

        def log_msg(msg: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, msg + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        def handle_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                progress_bar.configure(maximum=max(total, 1))
                progress_var.set(done)
                progress_label_var.set(f"{done}/{total} {label}")
            elif kind == "log":
                log_msg(event[1])
            elif kind == "done":
                summary = event[1] or {"total": 0, "created": 0, "updated": 0, "skipped": 0, "errors": 0}
                messagebox.showinfo(
                    "Готово",
                    (
                        f"Всего строк: {summary['total']}\n"
                        f"Создано: {summary['created']}\n"
                        f"Обновлено: {summary['updated']}\n"
                        f"Пропущено: {summary['skipped']}\n"
                        f"Ошибки: {summary['errors']}"
                    ),
                )
                self.unregister_bg_handler(processing_task["task"].queue)
                processing_task["task"] = None
            elif kind == "error":
                messagebox.showerror("Ошибка", event[1])
                self.unregister_bg_handler(processing_task["task"].queue)
                processing_task["task"] = None

        def browse_file():
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
            if path:
                file_var.set(path)

        def browse_images_dir():
            path = filedialog.askdirectory(initialdir=images_dir_var.get() or os.getcwd())
            if path:
                images_dir_var.set(path)

        def on_toggle_headers():
            manual_frame.grid_remove() if has_headers_var.get() else manual_frame.grid()

        def build_manual_map():
            manual_map: dict[str, int] = {}
            for target, var in (
                ("id", id_col_var),
                ("word", word_col_var),
                ("translation", translation_col_var),
                ("example", example_col_var),
                ("notes", notes_col_var),
                ("tags", tags_col_var),
                ("front", front_col_var),
                ("back", back_col_var),
            ):
                if var.get() > 0:
                    manual_map[target] = var.get() - 1
            return manual_map

        def start_import():
            if processing_task["task"]:
                messagebox.showinfo("Занято", "Импорт уже выполняется")
                return

            csv_path = file_var.get().strip()
            if not csv_path or not os.path.exists(csv_path):
                messagebox.showerror("Ошибка", "Укажите существующий CSV файл")
                return

            deck_id_raw = deck_var.get()
            if not deck_id_raw:
                messagebox.showerror("Ошибка", "Выберите колоду")
                return
            deck_id = int(deck_id_raw.split(":", 1)[0]) if ":" in deck_id_raw else int(deck_id_raw)

            mapping_mode = {"variant": "auto"}
            manual_tags_idx: int | None = None
            manual_id_idx: int | None = None
            if not has_headers_var.get():
                manual_map = build_manual_map()
                manual_tags_idx = manual_map.pop("tags", None)
                manual_id_idx = manual_map.pop("id", None)
                mapping_mode = {"variant": "manual", "manual_map": manual_map}

            attach_images = attach_images_var.get()
            images_dir = images_dir_var.get().strip() or os.path.join(os.getcwd(), "images")

            def worker(task_obj: BackgroundTask):
                summary = {"total": 0, "created": 0, "updated": 0, "skipped": 0, "errors": 0}
                conn = get_connection()
                try:
                    encoding = detect_encoding(csv_path)
                    with open(csv_path, "r", encoding=encoding, newline="") as fh:
                        reader = csv.DictReader(fh) if has_headers_var.get() else csv.reader(fh)
                        rows = list(reader)

                    total_rows = len(rows)
                    for idx, row in enumerate(rows, start=1):
                        if task_obj.cancelled():
                            break
                        summary["total"] += 1
                        try:
                            row_data = row
                            tags_value = ""
                            external_id = None

                            if has_headers_var.get() and isinstance(row, dict):
                                lowered = {k.lower(): v for k, v in row.items()}
                                for candidate in ("id", "external_id", "word_id", "uid"):
                                    if candidate in lowered:
                                        external_id = lowered.get(candidate)
                                        break
                                tags_value = lowered.get("tags", "")
                                fields = map_row_to_fields(lowered, mapping_mode)
                            else:
                                tags_value = normalize_tags(row[manual_tags_idx]) if manual_tags_idx is not None and len(row) > manual_tags_idx else ""
                                fields = map_row_to_fields(row, mapping_mode)
                                if manual_id_idx is not None and len(row) > manual_id_idx:
                                    external_id = row[manual_id_idx]
                                elif generate_id_var.get() and row:
                                    external_id = row[0]

                            if not fields:
                                raise ValueError("Пустая строка")

                            srs_defaults = {
                                "state": start_state_var.get(),
                                "due": int(time.time()),
                                "interval": 0,
                                "ease": 250,
                                "reps": 0,
                                "lapses": 0,
                                "step_index": 0,
                                "phase": start_phase_var.get(),
                            }

                            if preserve_srs_var.get() and isinstance(row_data, dict):
                                for key in ("due", "interval", "ease", "reps", "lapses", "step_index", "phase"):
                                    if key in row_data and str(row_data.get(key)).strip():
                                        try:
                                            srs_defaults[key] = int(float(row_data.get(key)))
                                        except Exception:
                                            pass
                                if row_data.get("state"):
                                    srs_defaults["state"] = str(row_data.get("state")).strip()

                            mode = {
                                "skip_existing": skip_existing_var.get() or not upsert_var.get(),
                                "reset_srs": reset_srs_var.get(),
                                "state": srs_defaults["state"],
                            }

                            result = upsert_note_and_cards(
                                conn,
                                deck_id,
                                str(external_id) if external_id is not None else None,
                                fields,
                                tags_value,
                                srs_defaults,
                                mode,
                            )
                            conn.commit()

                            note_id = result.get("note_id")
                            card_ids = result.get("card_ids") or []
                            if attach_images and external_id and card_ids:
                                attach_image_if_exists(conn, note_id, card_ids[0], external_id, images_dir)
                                conn.commit()

                            status = result.get("status", "created")
                            summary[status] = summary.get(status, 0) + 1
                            log_msg(f"[{status}] строка {idx}: {render_card_faces(fields)[0]}")
                        except Exception as exc:  # noqa: BLE001
                            summary["errors"] += 1
                            log_msg(f"[ошибка] строка {idx}: {exc}")
                        task_obj.queue.put(("progress", idx, max(total_rows, 1), "импорт"))

                    task_obj.queue.put(("done", summary))
                except Exception as exc:  # noqa: BLE001
                    task_obj.queue.put(("error", str(exc)))
                finally:
                    conn.close()

            task = start_background_task(worker)
            processing_task["task"] = task
            self.register_bg_handler(task.queue, handle_event)

        def cancel_import():
            if processing_task["task"]:
                processing_task["task"].cancel()
                processing_task["task"] = None
                log_msg("[остановлено] Импорт прерван")

        top_frame = ttk.Frame(win)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(top_frame, text="CSV файл:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top_frame, textvariable=file_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(top_frame, text="Выбрать", command=browse_file).grid(row=0, column=2, padx=5)
        top_frame.columnconfigure(1, weight=1)

        deck_frame = ttk.Frame(win)
        deck_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(deck_frame, text="Целевая колода:").grid(row=0, column=0, sticky="w")
        deck_box = ttk.Combobox(deck_frame, textvariable=deck_var, values=[f"{d['id']}: {d['name']}" for d in decks])
        deck_box.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(deck_frame, text="Создать новую", command=lambda: (self.add_deck_window(), refresh_decks_list())).grid(row=0, column=2, padx=5)
        deck_frame.columnconfigure(1, weight=1)

        options = ttk.LabelFrame(win, text="Настройки импорта")
        options.pack(fill=tk.X, padx=10, pady=5)
        ttk.Checkbutton(options, text="CSV имеет заголовки", variable=has_headers_var, command=on_toggle_headers).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options, text="Обновлять существующие по ID (upsert)", variable=upsert_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(options, text="Пропускать существующие по ID", variable=skip_existing_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(options, text="Если нет ID — генерировать", variable=generate_id_var).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(options, text="Прикреплять картинки из папки images/<id>.png", variable=attach_images_var).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(options, text="Сохранить интервалы из CSV (due/interval/ease)", variable=preserve_srs_var).grid(row=5, column=0, sticky="w")
        ttk.Checkbutton(options, text="Сбросить расписание при обновлении", variable=reset_srs_var).grid(row=6, column=0, sticky="w")

        ttk.Label(options, text="Стартовая фаза:").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Spinbox(options, from_=1, to=10, textvariable=start_phase_var, width=5).grid(row=0, column=2, sticky="w")
        ttk.Label(options, text="Стартовое состояние SRS:").grid(row=1, column=1, sticky="w", padx=5)
        ttk.Combobox(options, values=["new", "learning", "review"], textvariable=start_state_var, width=10).grid(row=1, column=2, sticky="w")
        ttk.Label(options, text="Папка с картинками:").grid(row=2, column=1, sticky="w", padx=5)
        ttk.Entry(options, textvariable=images_dir_var, width=28).grid(row=2, column=2, sticky="we", padx=5)
        ttk.Button(options, text="...", command=browse_images_dir, width=4).grid(row=2, column=3, padx=5)

        manual_frame = ttk.LabelFrame(win, text="Сопоставление колонок (если нет заголовков)")
        manual_frame.pack(fill=tk.X, padx=10, pady=5)
        for i, (label, var) in enumerate(
            [
                ("ID", id_col_var),
                ("Word", word_col_var),
                ("Translation", translation_col_var),
                ("Example", example_col_var),
                ("Notes", notes_col_var),
                ("Tags", tags_col_var),
                ("Front", front_col_var),
                ("Back", back_col_var),
            ]
        ):
            ttk.Label(manual_frame, text=label + ":").grid(row=i // 2, column=(i % 2) * 2, sticky="e", padx=5, pady=2)
            ttk.Spinbox(manual_frame, from_=0, to=20, textvariable=var, width=5).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=5, pady=2)
        if has_headers_var.get():
            manual_frame.grid_remove()

        progress_frame = ttk.Frame(win)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, expand=True)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Импортировать", command=start_import).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=cancel_import).pack(side=tk.LEFT, padx=5)

        log_box = tk.Text(win, height=14, state="disabled")
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # --------- режим ознакомления ---------

    def start_overview_mode(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        deck_id = self.selected_deck_id
        phase_filter = self.selected_phase

        def task(progress_cb):
            cards = get_overview_cards(deck_id)
            if phase_filter is not None:
                cards = [c for c in cards if c["leitner_level"] == phase_filter]
            total = len(cards)
            for idx in range(0, total, max(1, max(1, total // 20))):
                progress_cb(min(idx + 1, total), total, f"Подготовка {min(idx + 1, total)}/{total}")
            return cards

        def on_success(cards):
            if not cards:
                phase_text = f" (фаза {phase_filter})" if phase_filter is not None else ""
                messagebox.showinfo("Ознакомление", f"В этой колоде{phase_text} пока нет карточек.")
                return
            OverviewWindow(self, cards)

        def on_error(exc: Exception):
            messagebox.showerror("Ошибка", str(exc))

        self.run_task("Режим ознакомления", "determinate", task, on_success, on_error)

    def add_cards_to_overview_from_repeat(self):
        """Добавить карточки из режима повторения в режим ознакомления"""
        if self.selected_deck_id is None:
            return
        
        # Получаем карточки из режима повторения
        repeat_cards = get_cards_for_repeat(self.selected_deck_id)
        if self.selected_phase is not None:
            repeat_cards = [c for c in repeat_cards if c["leitner_level"] == self.selected_phase]
        
        if not repeat_cards:
            messagebox.showinfo("Нет карточек", "В режиме повторения нет карточек для добавления.")
            return
        
        # Помечаем карточки как добавленные в ознакомление
        for card in repeat_cards:
            mark_card_for_overview(card["id"])
        
        messagebox.showinfo("Успех", f"Добавлено {len(repeat_cards)} карточек в режим ознакомления.")

    # --------- управление словарями ---------

    def open_dictionary_manager_window(self):
        win = tk.Toplevel(self)
        win.title("Управление словарями")
        win.geometry("600x500")
        win.grab_set()

        # Статистика словаря
        stats_frame = ttk.LabelFrame(win, text="Статистика словаря")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        stats = DICTIONARY_MANAGER.get_statistics()
        stats_text = f"""
        Всего слов в словаре: {stats['total_words']:,}
        Загруженные файлы: {len(stats['loaded_files'])}
        Используемая память: {stats['memory_size_mb']:.2f} МБ
        
        Формат: немецкое слово -> русский перевод
        """
        
        stats_label = ttk.Label(stats_frame, text=stats_text, justify=tk.LEFT)
        stats_label.pack(padx=10, pady=10)
        
        # Загруженные файлы
        if stats['loaded_files']:
            files_frame = ttk.LabelFrame(win, text="Загруженные файлы словарей")
            files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            listbox = tk.Listbox(files_frame)
            listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            for file in stats['loaded_files']:
                listbox.insert(tk.END, file)

        # Управление словарями
        mgmt_frame = ttk.LabelFrame(win, text="Управление словарями")
        mgmt_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_frame = ttk.Frame(mgmt_frame)
        btn_frame.pack(padx=10, pady=10)
        
        def load_dictionary():
            filetypes = [
                ("CSV файлы", "*.csv"),
                ("JSON файлы", "*.json"),
                ("Сжатые файлы", "*.gz *.json.gz"),
                ("Все файлы", "*.*"),
            ]
            filename = filedialog.askopenfilename(
                title="Выберите файл словаря",
                filetypes=filetypes
            )
            if filename:
                try:
                    if filename.endswith('.csv'):
                        count = DICTIONARY_MANAGER.load_from_csv(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")
                    elif filename.endswith('.json'):
                        count = DICTIONARY_MANAGER.load_from_json(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")
                    elif filename.endswith(('.gz', '.json.gz')):
                        count = DICTIONARY_MANAGER.load_from_compressed(filename)
                        messagebox.showinfo("Успех", f"Загружено {count} слов из {filename}")
                    
                    # Сохраняем путь в настройках
                    if filename not in TRANSLATION_SETTINGS.dictionary_paths:
                        TRANSLATION_SETTINGS.dictionary_paths.append(filename)
                        TRANSLATION_SETTINGS.save()
                    
                    # Обновляем окно
                    win.destroy()
                    self.open_dictionary_manager_window()
                    
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить словарь:\n{e}")
        
        def export_dictionary():
            filename = filedialog.asksaveasfilename(
                title="Экспорт словаря",
                defaultextension=".csv",
                filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
            )
            if filename:
                try:
                    DICTIONARY_MANAGER.export_to_csv(filename)
                    messagebox.showinfo("Успех", f"Словарь экспортирован в {filename}")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось экспортировать словарь:\n{e}")
        
        def save_compressed():
            filename = filedialog.asksaveasfilename(
                title="Сохранить сжатый словарь",
                defaultextension=".json.gz",
                filetypes=[("Сжатые JSON файлы", "*.json.gz"), ("Все файлы", "*.*")]
            )
            if filename:
                try:
                    DICTIONARY_MANAGER.save_compressed_dictionary(filename)
                    messagebox.showinfo("Успех", f"Словарь сохранен в {filename}")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось сохранить словарь:\n{e}")
        
        def search_word():
            search_win = tk.Toplevel(win)
            search_win.title("Поиск слова")
            search_win.geometry("400x300")
            search_win.grab_set()
            
            ttk.Label(search_win, text="Введите слово для поиска:").pack(padx=10, pady=(10, 0))
            
            entry = ttk.Entry(search_win)
            entry.pack(fill=tk.X, padx=10, pady=5)
            
            results_text = tk.Text(search_win, height=10)
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            def perform_search():
                word = entry.get().strip()
                if word:
                    results = DICTIONARY_MANAGER.search_words(word, limit=20)
                    results_text.delete(1.0, tk.END)
                    if results:
                        for german, russian in results:
                            results_text.insert(tk.END, f"{german} -> {russian}\n")
                    else:
                        results_text.insert(tk.END, "Совпадений не найдено")
            
            ttk.Button(search_win, text="Поиск", command=perform_search).pack(pady=10)
        
        ttk.Button(btn_frame, text="Загрузить словарь", command=load_dictionary).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Экспорт в CSV", command=export_dictionary).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Сохранить сжатый", command=save_compressed).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(btn_frame, text="Поиск слова", command=search_word).grid(row=1, column=0, padx=5, pady=5, columnspan=3)

    def show_dictionary_stats_window(self):
        win = tk.Toplevel(self)
        win.title("Статистика словаря")
        win.geometry("400x300")
        win.grab_set()
        
        stats = DICTIONARY_MANAGER.get_statistics()
        
        stats_text = f"""
        📊 СТАТИСТИКА СЛОВАРЯ
        
        Всего слов: {stats['total_words']:,}
        
        Загруженные файлы: {len(stats['loaded_files'])}
        
        Используемая память: {stats['memory_size_mb']:.2f} МБ
        
        Покрытие слов:
        - 1,000 самых частотных слов: {min(stats['total_words'], 1000):,}
        - 5,000 самых частотных слов: {min(stats['total_words'], 5000):,}
        - 10,000 самых частотных слов: {min(stats['total_words'], 10000):,}
        - 50,000 самых частотных слов: {min(stats['total_words'], 50000):,}
        - 100,000 самых частотных слов: {min(stats['total_words'], 100000):,}
        """
        
        label = ttk.Label(win, text=stats_text, justify=tk.LEFT)
        label.pack(padx=20, pady=20)
        
        if stats['total_words'] < 50000:
            ttk.Label(win, 
                     text="⚠️ Рекомендуется загрузить больший словарь для лучшего покрытия",
                     foreground="orange").pack(pady=10)

    # --------- настройки перевода ---------

    def open_translation_settings_window(self):
        win = tk.Toplevel(self)
        win.title("Настройки перевода")
        win.geometry("400x350")
        win.grab_set()

        # Встроенный словарь
        use_dict_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.use_embedded_dict)
        cb_dict = ttk.Checkbutton(
            win, 
            text="Использовать встроенный словарь",
            variable=use_dict_var
        )
        cb_dict.pack(anchor="w", padx=20, pady=(20, 10))

        # OpenAI
        use_openai_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.use_openai)
        cb_openai = ttk.Checkbutton(
            win,
            text="Использовать OpenAI для перевода (если есть ключ)",
            variable=use_openai_var
        )
        cb_openai.pack(anchor="w", padx=20, pady=10)

        # Показывать переводы на лицевой стороне
        show_trans_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.show_translations)
        cb_show = ttk.Checkbutton(
            win,
            text="Показывать переводы над словами в режиме повторения (лицевая сторона)",
            variable=show_trans_var
        )
        cb_show.pack(anchor="w", padx=20, pady=10)

        # Показывать перевод на задней стороне
        show_back_var = tk.BooleanVar(value=TRANSLATION_SETTINGS.show_back_translation)
        cb_back = ttk.Checkbutton(
            win,
            text="Всегда показывать русский перевод на задней стороне карточки",
            variable=show_back_var
        )
        cb_back.pack(anchor="w", padx=20, pady=10)

        # Приоритет
        ttk.Label(win, text="Приоритет перевода:").pack(anchor="w", padx=20, pady=(10, 5))
        priority_var = tk.StringVar(value="dictionary")
        ttk.Radiobutton(
            win,
            text="Сначала словарь, потом OpenAI",
            variable=priority_var,
            value="dictionary"
        ).pack(anchor="w", padx=30)
        ttk.Radiobutton(
            win,
            text="Сначала OpenAI, потом словарь",
            variable=priority_var,
            value="openai"
        ).pack(anchor="w", padx=30)

        def save_settings():
            TRANSLATION_SETTINGS.use_embedded_dict = use_dict_var.get()
            TRANSLATION_SETTINGS.use_openai = use_openai_var.get()
            TRANSLATION_SETTINGS.show_translations = show_trans_var.get()
            TRANSLATION_SETTINGS.show_back_translation = show_back_var.get()
            TRANSLATION_SETTINGS.save()
            messagebox.showinfo("Сохранено", "Настройки перевода сохранены.")
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        ttk.Button(btn_frame, text="Сохранить", command=save_settings).pack(side=tk.RIGHT)

    # --------- настройки ---------

    def open_settings_window(self):
        global OPENAI_API_KEY

        win = tk.Toplevel(self)
        win.title("Настройки OpenAI")
        win.geometry("450x190")
        win.grab_set()

        ttk.Label(
            win,
            text="API ключ OpenAI (формат sk-... / sk-proj-...; хранится только в памяти):"
        ).pack(anchor="w", padx=10, pady=(10, 0))

        entry_key = ttk.Entry(win, show="*")
        entry_key.pack(fill=tk.X, padx=10, pady=5)
        create_context_menu(entry_key)  # Добавляем контекстное меню

        if OPENAI_API_KEY:
            entry_key.insert(0, OPENAI_API_KEY)

        def paste_from_clipboard():
            try:
                text = win.clipboard_get()
            except tk.TclError:
                text = ""
            entry_key.delete(0, tk.END)
            entry_key.insert(0, text.strip())

        ttk.Button(win, text="Вставить из буфера обмена",
                   command=paste_from_clipboard).pack(anchor="e", padx=10, pady=(0, 5))

        def save_key():
            global OPENAI_API_KEY
            key = entry_key.get().strip()
            OPENAI_API_KEY = key or None
            messagebox.showinfo("Сохранено", "Ключ сохранён в памяти приложения.")
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="OK", command=save_key).pack(side=tk.RIGHT)

    def open_audio_device_window(self):
        if not SR_AVAILABLE:
            messagebox.showerror(
                "Речь недоступна",
                "SpeechRecognition не установлен.\n"
                "Установи: pip install SpeechRecognition pyaudio"
            )
            return

        try:
            devices = sr.Microphone.list_microphone_names()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить список устройств:\n{e}")
            return

        win = tk.Toplevel(self)
        win.title("Выбор аудиоустройства для цифрового слуха")
        win.geometry("500x300")
        win.grab_set()

        ttk.Label(
            win,
            text="Выбери устройство записи, которое будет слушать звук\n"
                 "в режиме «Генерация через цифрового слуха».\n\n"
                 "Для VB-Audio Cable обычно это CABLE Output."
        ).pack(anchor="w", padx=10, pady=(10, 0))

        listbox = tk.Listbox(win, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        selected_initial = 0
        for i, name in enumerate(devices):
            listbox.insert(tk.END, f"{i}: {name}")
            if self.microphone_index is not None and i == self.microphone_index:
                selected_initial = i

        if devices:
            listbox.selection_set(selected_initial)
            listbox.see(selected_initial)

        def save_device():
            sel = listbox.curselection()
            if not sel:
                self.microphone_index = None
            else:
                idx_line = listbox.get(sel[0])
                idx_str = idx_line.split(":", 1)[0]
                try:
                    self.microphone_index = int(idx_str)
                except ValueError:
                    self.microphone_index = None
            messagebox.showinfo(
                "Сохранено",
                f"Устройство записи для цифрового слуха установлено: "
                f"{self.microphone_index if self.microphone_index is not None else 'по умолчанию'}"
            )
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="OK", command=save_device).pack(side=tk.RIGHT)

    # --------- статистика ---------

    def show_statistics_window(self):
        win = tk.Toplevel(self)
        win.title("Статистика колод")
        win.geometry("1200x900")
        win.grab_set()

        # Получаем все колоды
        decks = list_decks()
        if not decks:
            messagebox.showinfo("Нет колод", "Сначала создайте колоду.")
            win.destroy()
            return

        # Создаем вкладки для каждой колоды
        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for deck in decks:
            deck_frame = ttk.Frame(notebook)
            notebook.add(deck_frame, text=deck['name'])

            # Содержимое вкладки колоды
            self.create_deck_statistics_tab(deck_frame, deck['id'], deck['name'])

        # Кнопка обновить все
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        def update_all_dates():
            for i in range(notebook.index("end")):
                tab = notebook.nametowidget(notebook.tabs()[i])
                for child in tab.winfo_children():
                    if hasattr(child, 'update_charts'):
                        child.update_charts()
            messagebox.showinfo("Обновлено", "Все графики обновлены")

        ttk.Button(btn_frame, text="Обновить все графики", command=update_all_dates).pack(side=tk.RIGHT)

    def create_deck_statistics_tab(self, parent, deck_id, deck_name):
        """Создать вкладку статистики для конкретной колоды"""
        # Основной контейнер
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Настройки для диаграмм
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки диаграмм")
        settings_frame.pack(fill=tk.X, pady=5)

        stored_settings = load_stats_settings(deck_id)

        ttk.Label(settings_frame, text="Режим оси X:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        x_mode_var = tk.StringVar(value=stored_settings.x_mode)
        x_mode_combo = ttk.Combobox(
            settings_frame,
            values=["range", "month_days", "custom_dates"],
            textvariable=x_mode_var,
            state="readonly",
            width=15,
        )
        x_mode_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Дней (если диапазон пустой):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        days_var = tk.IntVar(value=30)
        days_spin = ttk.Spinbox(settings_frame, from_=7, to=365, textvariable=days_var, width=8)
        days_spin.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Дата от (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        date_from_var = tk.StringVar(value=stored_settings.date_from or "")
        ttk.Entry(settings_frame, textvariable=date_from_var, width=16).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Дата до (YYYY-MM-DD):").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        date_to_var = tk.StringVar(value=stored_settings.date_to or "")
        ttk.Entry(settings_frame, textvariable=date_to_var, width=16).grid(row=1, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Кастомные даты (по строкам):").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        custom_dates_text = tk.Text(settings_frame, height=3, width=25)
        custom_dates_text.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky="we")
        if stored_settings.custom_dates:
            custom_dates_text.insert("1.0", "\n".join(stored_settings.custom_dates))

        ttk.Label(settings_frame, text="Максимум Y (0 - авто):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        max_y_var = tk.IntVar(value=stored_settings.y_max)
        max_y_spin = ttk.Spinbox(settings_frame, from_=0, to=10000, textvariable=max_y_var, width=10)
        max_y_spin.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Норма (горизонталь):").grid(row=3, column=2, padx=5, pady=5, sticky="w")
        norm_var = tk.IntVar(value=stored_settings.norm_value)
        norm_spin = ttk.Spinbox(settings_frame, from_=0, to=20000, textvariable=norm_var, width=10)
        norm_spin.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Тип графика:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        chart_type_var = tk.StringVar(value=stored_settings.chart_type)
        chart_combo = ttk.Combobox(
            settings_frame, values=["bar", "line"], textvariable=chart_type_var, state="readonly", width=10
        )
        chart_combo.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        show_grid_var = tk.BooleanVar(value=bool(stored_settings.show_grid))
        ttk.Checkbutton(settings_frame, text="Показывать сетку", variable=show_grid_var).grid(
            row=4, column=2, padx=5, pady=5, sticky="w"
        )

        # Фрейм для графиков
        charts_frame = ttk.Frame(main_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        def _safe_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def collect_settings() -> StatsSettings:
            settings = StatsSettings(
                deck_id=deck_id,
                x_mode=x_mode_var.get() or "range",
                date_from=date_from_var.get().strip() or None,
                date_to=date_to_var.get().strip() or None,
                y_max=_safe_int(max_y_var.get(), 0),
                norm_value=_safe_int(norm_var.get(), 0),
                chart_type=chart_type_var.get() or "bar",
                show_grid=1 if show_grid_var.get() else 0,
            )
            custom_lines = custom_dates_text.get("1.0", "end").splitlines()
            settings.update_custom_dates(custom_lines)
            return settings

        def parse_date_safe(value: str | None):
            if not value:
                return None
            try:
                return datetime.strptime(value.strip(), "%Y-%m-%d").date()
            except ValueError:
                return None

        def build_date_list(settings: StatsSettings) -> list[date]:
            mode = settings.x_mode or "range"
            if mode == "month_days":
                today = date.today()
                last_day = calendar.monthrange(today.year, today.month)[1]
                return [date(today.year, today.month, d) for d in range(1, last_day + 1)]
            if mode == "custom_dates":
                dates: list[date] = []
                for line in custom_dates_text.get("1.0", "end").splitlines():
                    parsed = parse_date_safe(line)
                    if parsed:
                        dates.append(parsed)
                return sorted(set(dates))

            start = parse_date_safe(settings.date_from)
            end = parse_date_safe(settings.date_to)
            if not end:
                end = date.today()
            if not start:
                start = end - timedelta(days=max(1, days_var.get()) - 1)
            if start > end:
                start, end = end, start

            dates: list[date] = []
            current = start
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)
            return dates

        def update_charts():
            if not MATPLOTLIB_AVAILABLE:
                for widget in charts_frame.winfo_children():
                    widget.destroy()
                ttk.Label(charts_frame,
                         text="Matplotlib не установлен. Установите: pip install matplotlib",
                         foreground="red").pack(pady=20)
                return

            current_settings = collect_settings()
            save_stats_settings(current_settings)

            # Получаем данные
            date_list = build_date_list(current_settings)
            data = get_statistics_for_dates(deck_id, date_list)

            # Очищаем фрейм
            for widget in charts_frame.winfo_children():
                widget.destroy()

            if not data["dates"]:
                ttk.Label(charts_frame, text="Нет данных для отображения").pack(pady=20)
                return

            from matplotlib import dates as mdates

            grid_enabled = bool(show_grid_var.get())
            y_limit = current_settings.y_max if current_settings.y_max and current_settings.y_max > 0 else None

            # Создаем фигуру с четырьмя подграфиками
            fig = Figure(figsize=(14, 16), dpi=100)
            ax1 = fig.add_subplot(411)
            ax2 = fig.add_subplot(412, sharex=ax1)
            ax3 = fig.add_subplot(413, sharex=ax1)
            ax4 = fig.add_subplot(414)

            # Подготовка оси X
            x_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in data["dates"]]
            x_numeric = mdates.date2num(x_dates)

            def apply_common_axis(ax):
                locator = mdates.AutoDateLocator()
                formatter = mdates.DateFormatter("%d.%m")
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                if current_settings.x_mode == "custom_dates":
                    ax.set_xticks(x_numeric)
                if y_limit:
                    ax.set_ylim(0, y_limit)
                if grid_enabled:
                    ax.grid(True, alpha=0.3)

            # Подграфик 1: Общая статистика повторений
            reviewed = data["reviewed"]
            if current_settings.chart_type == "line":
                ax1.plot(x_dates, reviewed, marker="o", linewidth=2, color="blue", label="Просмотрено карточек")
            else:
                ax1.bar(x_numeric, reviewed, width=0.6, color="blue", alpha=0.6, label="Просмотрено карточек")

            if current_settings.norm_value:
                ax1.axhline(current_settings.norm_value, linestyle="--", color="gray", linewidth=1, label="Норма")

            ax1.set_ylabel('Количество карточек')
            ax1.set_title(f'Общая статистика повторений - {deck_name}')
            apply_common_axis(ax1)
            ax1.legend(loc='upper left')

            # Подграфик 2: Сравнение "помню" и "забыл"
            remembered = data["remembered"]
            forgotten = data["forgotten"]
            if current_settings.chart_type == "line":
                ax2.plot(x_dates, remembered, marker="o", color='green', label='Помню')
                ax2.plot(x_dates, forgotten, marker="o", color='red', label='Забыл')
            else:
                width = 0.35
                bars1 = ax2.bar(x_numeric - width/2, remembered, width, label='Помню', color='green', alpha=0.7)
                bars2 = ax2.bar(x_numeric + width/2, forgotten, width, label='Забыл', color='red', alpha=0.7)
                for bar in list(bars1) + list(bars2):
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=8)

            if current_settings.norm_value:
                ax2.axhline(current_settings.norm_value, linestyle="--", color="gray", linewidth=1)

            ax2.set_ylabel('Количество карточек')
            ax2.set_title('Сравнение запомненных и забытых карточек')
            apply_common_axis(ax2)
            ax2.legend(loc='upper left')

            # Подграфик 3: Ознакомление
            overview = data["overview"]
            if current_settings.chart_type == "line":
                ax3.plot(x_dates, overview, marker="o", color='orange', label='Ознакомлено карточек')
            else:
                bars3 = ax3.bar(x_numeric, overview, width=0.6, label='Ознакомлено карточек', color='orange', alpha=0.7)
                for bar in bars3:
                    height = bar.get_height()
                    if height > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=8)

            if current_settings.norm_value:
                ax3.axhline(current_settings.norm_value, linestyle="--", color="gray", linewidth=1)

            ax3.set_ylabel('Количество карточек')
            ax3.set_title('Ознакомление с карточками')
            ax3.set_xlabel('Дата')
            apply_common_axis(ax3)
            ax3.legend(loc='upper left')

            fig.autofmt_xdate(rotation=30)

            # Подграфик 4: Прогресс по фазам
            # Получаем статистику по фазам
            stats = get_deck_stats(deck_id)
            phases = list(range(1, 11))
            phase_counts = [stats["phase_stats"].get(phase, 0) for phase in phases]
            
            bars4 = ax4.bar(phases, phase_counts, width=0.6, color='purple', alpha=0.7)
            
            ax4.set_xlabel('Фаза Лейтнера')
            ax4.set_ylabel('Количество карточек')
            ax4.set_title('Распределение карточек по фазам')
            ax4.set_xticks(phases)
            ax4.set_xticklabels([f'Фаза {p}' for p in phases], rotation=45, fontsize=8)
            if grid_enabled:
                ax4.grid(True, alpha=0.3)

            # Добавляем значения на столбцы
            for bar in bars4:
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)

            fig.tight_layout(pad=2.0)
            
            # Размещаем график в Tkinter
            canvas = FigureCanvasTkAgg(fig, charts_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Добавляем функцию обновления в объект
            charts_frame.update_charts = update_charts

        btn_update = ttk.Button(settings_frame, text="Сохранить и обновить", command=update_charts)
        btn_update.grid(row=0, column=4, padx=10, pady=5)

        # Инициализируем диаграммы
        update_charts()

    # --------- главное окно ---------

    def create_widgets(self):
        # Основной контейнер с двумя фреймами
        main_container = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Левый фрейм для списка колод
        left_frame = ttk.Frame(main_container)
        main_container.add(left_frame, weight=1)

        # Правый фрейм для превью колоды
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=1)

        # Левый фрейм: список колод
        frame_top = ttk.Frame(left_frame)
        frame_top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_frame = ttk.Frame(frame_top)
        title_frame.pack(fill=tk.X)

        ttk.Label(title_frame, text="Колоды:").pack(side=tk.LEFT, anchor="w")

        # Красный кружок с числом просроченных карточек
        self.overdue_canvas = tk.Canvas(
            title_frame, width=24, height=24,
            highlightthickness=0, bg=self.cget("bg")
        )
        self.overdue_canvas.pack(side=tk.LEFT, padx=6)
        self.overdue_badge_text_id = None

        # Treeview вместо Listbox: колода -> фазы 1..10
        self.decks_tree = ttk.Treeview(frame_top, show="tree", selectmode="browse")
        self.decks_tree.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.decks_tree.bind("<<TreeviewSelect>>", self.on_deck_select)
        self.phase_badge_manager = PhaseOverdueBadges(self.decks_tree)

        frame_buttons = ttk.Frame(left_frame)
        frame_buttons.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(frame_buttons, text="Новая колода", command=self.add_deck_window)\
            .pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_buttons, text="Редактировать колоду", command=self.edit_deck_window)\
            .pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_buttons, text="Удалить колоду", command=self.delete_selected_deck)\
            .pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_buttons, text="Добавить карточку вручную", command=self.add_card_window)\
            .pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_buttons, text="Режим повторения", command=self.start_repeat_mode)\
            .pack(side=tk.RIGHT, padx=5)

        # Правый фрейм: превью колоды
        self.preview_frame = ttk.LabelFrame(right_frame, text="Превью колоды")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Контейнер для изображения и текста
        preview_container = ttk.Frame(self.preview_frame)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Фрейм для изображения (левая часть)
        self.image_frame = ttk.Frame(preview_container)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.deck_preview_label = tk.Label(
            self.image_frame, 
            text="Выберите колоду для просмотра",
            bg="white",
            relief="solid",
            bd=1
        )
        self.deck_preview_label.pack(fill=tk.BOTH, expand=True)

        # Фрейм для текста (правая часть)
        text_frame = ttk.Frame(preview_container)
        text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.deck_name_label = ttk.Label(
            text_frame,
            text="Название колоды",
            font=("Arial", 14, "bold"),
            wraplength=200
        )
        self.deck_name_label.pack(anchor=tk.W, pady=(0, 10))

        self.deck_desc_label = ttk.Label(
            text_frame,
            text="Описание колоды",
            wraplength=200,
            justify=tk.LEFT
        )
        self.deck_desc_label.pack(anchor=tk.W, pady=(0, 10))

        # Статистика колоды в превью
        stats_frame = ttk.LabelFrame(text_frame, text="Статистика")
        stats_frame.pack(fill=tk.X, pady=10)

        self.deck_stats_label = ttk.Label(
            stats_frame,
            text="Карточек: 0\nФаз: 0/10\nОзнакомлено: 0",
            justify=tk.LEFT
        )
        self.deck_stats_label.pack(padx=10, pady=10)

        # Кнопки действий для выбранной колоды
        action_frame = ttk.Frame(self.preview_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(action_frame, text="Просмотреть карточки", 
                  command=self.show_cards_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Режим воспроизведения", 
                  command=self.start_playback_mode).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Режим ознакомления", 
                  command=self.start_overview_mode).pack(side=tk.RIGHT, padx=5)

    def refresh_decks(self):
        if self._refresh_in_progress:
            return

        self._refresh_in_progress = True

        def task(progress_cb):
            decks_data = list_decks()
            stats_by_deck: dict[int, dict] = {}
            total = len(decks_data)
            for idx, deck in enumerate(decks_data, start=1):
                stats_by_deck[deck["id"]] = get_deck_stats(deck["id"])
                if total and (idx % max(1, total // 10 or 1) == 0 or idx == total):
                    progress_cb(idx, total, f"Загрузка колод {idx}/{total}")
            return {"decks": decks_data, "stats": stats_by_deck}

        def on_success(result):
            self.decks = result["decks"]
            self.deck_items = {}
            self.deck_icons = {}
            self.deck_preview_images = {}

            for item in self.decks_tree.get_children():
                self.decks_tree.delete(item)

            for deck in self.decks:
                icon = None
                if deck["icon_path"] and os.path.exists(deck["icon_path"]) and PIL_AVAILABLE:
                    try:
                        img = Image.open(deck["icon_path"])
                        img = img.resize((16, 16), Image.Resampling.LANCZOS)
                        icon = ImageTk.PhotoImage(img)
                        self.deck_icons[deck["id"]] = icon
                    except Exception:
                        pass

                desc = deck["description"] or "без описания"
                deck_text = f"{deck['name']} ({desc})"

                if icon:
                    root_id = self.decks_tree.insert("", "end", text=deck_text, image=icon, open=False)
                else:
                    root_id = self.decks_tree.insert("", "end", text=deck_text, open=False)

                self.deck_items[root_id] = (deck["id"], None)

                stats = result["stats"].get(deck["id"], {"phase_stats": {}, "total": 0})
                for phase in range(1, 11):
                    phase_count = stats.get("phase_stats", {}).get(phase, 0)
                    total_cards = stats.get("total", 0)
                    percentage = (phase_count / total_cards * 100) if total_cards > 0 else 0

                    child_text = f"Фаза {phase}: {phase_count} карт. ({percentage:.1f}%)"
                    child_id = self.decks_tree.insert(root_id, "end", text=child_text)
                    self.deck_items[child_id] = (deck["id"], phase)

            self.selected_deck_id = None
            self.selected_phase = None
            self.after(50, self.update_overdue_badge)
            self.update_deck_preview()
            self._refresh_in_progress = False

        def on_error(exc):
            messagebox.showerror("Ошибка", str(exc))
            self._refresh_in_progress = False

        self.run_task("Загрузка колод", "determinate", task, on_success, on_error)

    def update_deck_preview(self):
        """Обновить превью выбранной колоды."""
        # Очищаем текущее превью
        self.deck_preview_label.config(image="", text="Выберите колоду для просмотра")
        self.deck_name_label.config(text="Название колоды")
        self.deck_desc_label.config(text="Описание колоды")
        self.deck_stats_label.config(text="Карточек: 0\nФаз: 0/10\nОзнакомлено: 0")
        
        if self.selected_deck_id is None:
            return
        
        # Находим выбранную колоду
        selected_deck = None
        for d in self.decks:
            if d["id"] == self.selected_deck_id:
                selected_deck = d
                break
        
        if not selected_deck:
            return
        
        # Обновляем название и описание
        self.deck_name_label.config(text=selected_deck["name"])
        self.deck_desc_label.config(text=selected_deck["description"] or "Без описания")
        
        # Загружаем и отображаем изображение колоды
        if selected_deck["icon_path"] and os.path.exists(selected_deck["icon_path"]) and PIL_AVAILABLE:
            try:
                img = Image.open(selected_deck["icon_path"])
                # Автоматически уменьшаем окно до размеров картинки
                img_width, img_height = img.size
                
                # Масштабируем для превью, но не слишком сильно
                max_width = 300
                max_height = 200
                
                if img_width > max_width or img_height > max_height:
                    ratio = min(max_width / img_width, max_height / img_height)
                    new_width = int(img_width * ratio)
                    new_height = int(img_height * ratio)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                self.deck_preview_images[self.selected_deck_id] = photo
                self.deck_preview_label.config(image=photo, text="")
                
                # Устанавливаем размер окна превью под изображение
                self.image_frame.config(width=new_width, height=new_height)
                self.deck_preview_label.config(width=new_width, height=new_height)
                
            except Exception as e:
                self.deck_preview_label.config(
                    image="", 
                    text=f"Ошибка загрузки изображения\n{str(e)}"
                )
        else:
            self.deck_preview_label.config(
                image="", 
                text="Изображение не загружено\nили Pillow не установлен"
            )
        
        # Обновляем статистику
        stats = get_deck_stats(self.selected_deck_id)
        phases_with_cards = sum(1 for phase in range(1, 11) if stats["phase_stats"].get(phase, 0) > 0)
        self.deck_stats_label.config(
            text=f"Карточек: {stats['total']}\n"
                 f"Фаз: {phases_with_cards}/10\n"
                 f"Изучено: {stats['learned_percent']:.1f}%\n"
                 f"Ознакомлено: {stats['total_overview']}"
        )

    def update_overdue_badge(self):
        """Обновить бейджи просрочек для выбранной колоды и фаз."""
        if self._overdue_task_running:
            return

        deck_ids = {deck_id for deck_id, _ in self.deck_items.values() if deck_id is not None}
        total = len(deck_ids)
        self._overdue_task_running = True

        def task(progress_cb):
            counts_by_deck: dict[int | None, PhaseOverdueBadges] = {}
            timestamp = int(time.time())
            for idx, deck_id in enumerate(deck_ids, start=1):
                counts_by_deck[deck_id] = fetch_overdue_counts_by_phase(
                    None, deck_id, now_ts=timestamp
                )
                if total and (idx % max(1, total // 10 or 1) == 0 or idx == total):
                    progress_cb(idx, total, f"Просрочки {idx}/{total}")
            return counts_by_deck

        def on_success(counts_by_deck: dict[int | None, PhaseOverdueBadges]):
            selected_counts = counts_by_deck.get(self.selected_deck_id)
            total_count = selected_counts.total if selected_counts else 0

            if self.overdue_canvas is not None:
                self.overdue_canvas.delete("all")
                if total_count > 0:
                    self.overdue_canvas.create_oval(2, 2, 22, 22, fill="red", outline="red")
                    self.overdue_badge_text_id = self.overdue_canvas.create_text(
                        12, 12, text=str(total_count), fill="white", font=("Arial", 9, "bold")
                    )

            if self.phase_badge_manager:
                self.phase_badge_manager.update(self.deck_items, counts_by_deck)
            self._overdue_task_running = False
            self.schedule_overdue_badges_refresh()

        def on_error(exc: Exception):
            error_message = str(exc)
            os.makedirs("logs", exist_ok=True)
            with open(os.path.join("logs", "db.log"), "a", encoding="utf-8") as log_file:
                log_file.write(f"[{datetime.now().isoformat()}] update_overdue_badge: {error_message}\n")

            if "unable to open database file" in error_message.lower():
                counts_by_deck: dict[int | None, PhaseOverdueBadges] = {}
                if self.overdue_canvas is not None:
                    self.overdue_canvas.delete("all")
                if self.phase_badge_manager:
                    self.phase_badge_manager.update(self.deck_items, counts_by_deck)
            else:
                messagebox.showerror("БД", error_message)
            self._overdue_task_running = False
            self.schedule_overdue_badges_refresh()

        self.run_task("Просроченные карточки", "determinate", task, on_success, on_error, total=total or None)

    def schedule_overdue_badges_refresh(self):
        if self.overdue_update_job is not None:
            self.after_cancel(self.overdue_update_job)
        self.overdue_update_job = self.after(60_000, self.update_overdue_badge)

    def on_deck_select(self, event):
        if self._deck_select_job:
            self.after_cancel(self._deck_select_job)
        self._deck_select_job = self.after(150, self._apply_selected_deck)

    def _apply_selected_deck(self):
        self._deck_select_job = None
        sel = self.decks_tree.selection()
        if not sel:
            self.selected_deck_id = None
            self.selected_phase = None
            self.load_templates_for_selected_deck()
            self.update_overdue_badge()
            self.update_deck_preview()
            return

        item_id = sel[0]
        deck_id, phase = self.deck_items.get(item_id, (None, None))
        self.selected_deck_id = deck_id
        self.selected_phase = phase
        self.load_templates_for_selected_deck()
        self.update_overdue_badge()
        self.update_deck_preview()

    def load_templates_for_selected_deck(self):
        if self.selected_deck_id is None:
            self.front_template = DEFAULT_FRONT_TEMPLATE
            self.back_template = DEFAULT_BACK_TEMPLATE
            return
        try:
            front, back = get_deck_templates(self.selected_deck_id)
        except Exception:
            front, back = DEFAULT_FRONT_TEMPLATE, DEFAULT_BACK_TEMPLATE
        self.front_template = front or DEFAULT_FRONT_TEMPLATE
        self.back_template = back or DEFAULT_BACK_TEMPLATE

    # --------- новая колода ---------

    def add_deck_window(self):
        win = tk.Toplevel(self)
        win.title("Новая колода")
        win.geometry("400x280")
        win.grab_set()

        ttk.Label(win, text="Название колоды:").pack(anchor="w", padx=10, pady=(10, 0))
        entry_name = ttk.Entry(win)
        entry_name.pack(fill=tk.X, padx=10)
        create_context_menu(entry_name)  # Добавляем контекстное меню

        ttk.Label(win, text="Описание (необязательно):").pack(anchor="w", padx=10, pady=(10, 0))
        entry_desc = ttk.Entry(win)
        entry_desc.pack(fill=tk.X, padx=10)
        create_context_menu(entry_desc)  # Добавляем контекстное меню

        # Иконка колоды
        icon_path_var = tk.StringVar()
        
        icon_frame = ttk.Frame(win)
        icon_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(icon_frame, text="Изображение колоды:").pack(side=tk.LEFT)
        lbl_icon = ttk.Label(icon_frame, text="не выбрана")
        lbl_icon.pack(side=tk.LEFT, padx=5)
        
        def select_icon():
            filetypes = [
                ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"),
                ("Все файлы", "*.*"),
            ]
            filename = filedialog.askopenfilename(
                title="Выбрать изображение для колоды",
                filetypes=filetypes
            )
            if filename:
                icon_path_var.set(filename)
                lbl_icon.config(text=os.path.basename(filename))

        ttk.Button(icon_frame, text="Выбрать", command=select_icon).pack(side=tk.RIGHT, padx=5)

        def save_deck():
            name = entry_name.get().strip()
            desc = entry_desc.get().strip()
            icon_path = icon_path_var.get().strip() or None

            if not name:
                messagebox.showerror("Ошибка", "Название не может быть пустым.")
                return

            conn = get_connection()
            cur = conn.cursor()
            deck_id = None
            try:
                cur.execute(
                    """INSERT INTO decks
                       (name, description, front_template, back_template, icon_path)
                       VALUES (?, ?, ?, ?, ?);""",
                    (name, desc or None, self.front_template, self.back_template, icon_path)
                )
                deck_id = cur.lastrowid
            except sqlite3.OperationalError:
                cur.execute(
                    "INSERT INTO decks (name, description, icon_path) VALUES (?, ?, ?);",
                    (name, desc or None, icon_path)
                )
                deck_id = cur.lastrowid
            if deck_id:
                ensure_deck_settings_row(deck_id, conn, inherit_default=1)
            conn.commit()
            conn.close()
            self.refresh_decks()
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Сохранить", command=save_deck).pack(side=tk.RIGHT)

    # --------- редактирование колоды ---------

    def edit_deck_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        win = tk.Toplevel(self)
        win.title("Редактирование колоды")
        win.geometry("420x420")
        win.grab_set()

        # Получаем текущие данные колоды
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT name, description, icon_path FROM decks WHERE id = ?;", (self.selected_deck_id,))
        deck_data = cur.fetchone()
        conn.close()

        timer_settings = get_deck_timer_settings(self.selected_deck_id)

        ttk.Label(win, text="Название колоды:").pack(anchor="w", padx=10, pady=(10, 0))
        entry_name = ttk.Entry(win)
        entry_name.insert(0, deck_data["name"])
        entry_name.pack(fill=tk.X, padx=10)
        create_context_menu(entry_name)  # Добавляем контекстное меню

        ttk.Label(win, text="Описание:").pack(anchor="w", padx=10, pady=(10, 0))
        entry_desc = ttk.Entry(win)
        entry_desc.insert(0, deck_data["description"] or "")
        entry_desc.pack(fill=tk.X, padx=10)
        create_context_menu(entry_desc)  # Добавляем контекстное меню

        # Иконка колоды
        icon_path_var = tk.StringVar(value=deck_data["icon_path"] or "")
        
        icon_frame = ttk.Frame(win)
        icon_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(icon_frame, text="Изображение колоды:").pack(side=tk.LEFT)
        lbl_icon = ttk.Label(icon_frame, 
                           text=os.path.basename(icon_path_var.get()) if icon_path_var.get() else "не выбрана")
        lbl_icon.pack(side=tk.LEFT, padx=5)
        
        def select_icon():
            filetypes = [
                ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"),
                ("Все файлы", "*.*"),
            ]
            filename = filedialog.askopenfilename(
                title="Выбрать изображение для колоды",
                filetypes=filetypes
            )
            if filename:
                icon_path_var.set(filename)
                lbl_icon.config(text=os.path.basename(filename))

        ttk.Button(icon_frame, text="Выбрать", command=select_icon).pack(side=tk.RIGHT, padx=5)

        timer_frame = ttk.LabelFrame(win, text="Таймер колоды")
        timer_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(timer_frame, text="Секунды:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        timer_sec_var = tk.IntVar(value=timer_settings.get("timer_sec") or 0)
        timer_spin = ttk.Spinbox(timer_frame, from_=0, to=3600, textvariable=timer_sec_var, width=10)
        timer_spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(timer_frame, text="Режим:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        timer_mode_var = tk.StringVar(value=(timer_settings.get("timer_mode") or "reveal"))
        mode_combo = ttk.Combobox(
            timer_frame,
            state="readonly",
            values=["reveal", "fail", "notify"],
            textvariable=timer_mode_var,
            width=12,
        )
        mode_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        inherit_var = tk.BooleanVar(value=bool(timer_settings.get("inherit_timer", 1)))
        inherit_cb = ttk.Checkbutton(
            timer_frame,
            text="Наследовать от родителя, если свой таймер пуст",
            variable=inherit_var,
        )
        inherit_cb.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        def save_changes():
            name = entry_name.get().strip()
            desc = entry_desc.get().strip()
            icon_path = icon_path_var.get().strip() or None

            if not name:
                messagebox.showerror("Ошибка", "Название не может быть пустым.")
                return

            try:
                timer_sec = int(timer_sec_var.get())
            except tk.TclError:
                messagebox.showerror("Ошибка", "Некорректное значение таймера.")
                return

            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE decks SET name = ?, description = ?, icon_path = ? WHERE id = ?;",
                (name, desc or None, icon_path, self.selected_deck_id)
            )
            update_deck_timer_settings(
                self.selected_deck_id,
                timer_sec,
                timer_mode_var.get(),
                1 if inherit_var.get() else 0,
                conn,
            )
            conn.commit()
            conn.close()
            self.refresh_decks()
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Сохранить", command=save_changes).pack(side=tk.RIGHT)

    def delete_selected_deck(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        conn = open_db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM decks WHERE id = ?;", (self.selected_deck_id,))
            deck_row = cur.fetchone()
            if not deck_row:
                messagebox.showerror("Ошибка", "Не удалось найти выбранную колоду.")
                return

            deck_name = deck_row["name"] or "Без названия"

            cur.execute("PRAGMA table_info(decks);")
            deck_columns = {row["name"] for row in cur.fetchall()}
            has_parent_column = "parent_id" in deck_columns

            def collect_child_decks(parent_id: int) -> list[int]:
                if not has_parent_column:
                    return []
                cur.execute("SELECT id FROM decks WHERE parent_id = ?;", (parent_id,))
                children = [row["id"] for row in cur.fetchall()]
                result: list[int] = []
                for child_id in children:
                    result.append(child_id)
                    result.extend(collect_child_decks(child_id))
                return result

            child_deck_ids = collect_child_decks(self.selected_deck_id)
            if child_deck_ids:
                confirm_children = messagebox.askyesno(
                    "Есть подкалоды",
                    "У выбранной колоды есть подкалоды. Удалить их тоже?",
                )
                if not confirm_children:
                    messagebox.showwarning("Подколоды", "Сначала удалите подкалоды.")
                    return

            confirm = messagebox.askyesno(
                "Удалить колоду",
                (
                    f"Удалить колоду '{deck_name}'? Это удалит все карточки/ноты/медиа "
                    "этой колоды. Отменить нельзя."
                ),
            )
            if not confirm:
                return

            deck_ids_to_delete = [self.selected_deck_id, *child_deck_ids]
            media_table_exists = _table_exists(conn, "media")

            try:
                with DB_WRITE_LOCK:
                    conn.execute("BEGIN;")
                    for deck_id in deck_ids_to_delete:
                        cur.execute("SELECT id FROM notes WHERE deck_id = ?;", (deck_id,))
                        note_ids = [row["id"] for row in cur.fetchall()]
                        cur.execute("SELECT id FROM cards WHERE deck_id = ?;", (deck_id,))
                        card_ids = [row["id"] for row in cur.fetchall()]

                        if media_table_exists:
                            if note_ids:
                                placeholders = ",".join("?" * len(note_ids))
                                cur.execute(
                                    f"DELETE FROM media WHERE note_id IN ({placeholders});",
                                    note_ids,
                                )
                            if card_ids:
                                placeholders = ",".join("?" * len(card_ids))
                                cur.execute(
                                    f"DELETE FROM media WHERE card_id IN ({placeholders});",
                                    card_ids,
                                )

                        cur.execute("DELETE FROM cards WHERE deck_id = ?;", (deck_id,))
                        cur.execute("DELETE FROM notes WHERE deck_id = ?;", (deck_id,))
                        cur.execute("DELETE FROM decks WHERE id = ?;", (deck_id,))

                    conn.commit()
            except Exception as e:
                conn.rollback()
                messagebox.showerror("Ошибка", f"Не удалось удалить колоду: {e}")
                return

            self.selected_deck_id = None
            self.refresh_decks()
            self.update_deck_preview()
            self.update_overdue_badge()
            messagebox.showinfo("Готово", "Колода удалена.")
        finally:
            conn.close()

    # --------- ручная карточка ---------

    def add_card_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду в списке.")
            return

        win = tk.Toplevel(self)
        win.title("Новая карточка (ручной режим)")
        win.geometry("700x800")
        win.grab_set()

        # Поле для ввода ключа OpenAI
        api_key_frame = ttk.LabelFrame(win, text="OpenAI API ключ (для перевода)")
        api_key_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        ttk.Label(api_key_frame, text="Ключ OpenAI:").pack(anchor="w", padx=10, pady=(5, 0))
        entry_api_key = ttk.Entry(api_key_frame, show="*")
        entry_api_key.pack(fill=tk.X, padx=10, pady=(0, 5))
        create_context_menu(entry_api_key)  # Добавляем контекстное меню
        
        if OPENAI_API_KEY:
            entry_api_key.insert(0, OPENAI_API_KEY)
        
        def paste_api_key():
            try:
                text = win.clipboard_get()
                entry_api_key.delete(0, tk.END)
                entry_api_key.insert(0, text.strip())
            except:
                pass
        
        ttk.Button(api_key_frame, text="Вставить из буфера", command=paste_api_key).pack(anchor="e", padx=10, pady=(0, 5))

        # Переводчик в два поля
        translator_frame = ttk.LabelFrame(win, text="Переводчик (немецкий ↔ русский)")
        translator_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        # Немецкое поле
        german_frame = ttk.Frame(translator_frame)
        german_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(german_frame, text="Немецкий текст:").pack(side=tk.LEFT)
        entry_german = ttk.Entry(german_frame)
        entry_german.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        create_context_menu(entry_german)  # Добавляем контекстное меню
        
        # Русское поле
        russian_frame = ttk.Frame(translator_frame)
        russian_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(russian_frame, text="Русский перевод:").pack(side=tk.LEFT)
        entry_russian = ttk.Entry(russian_frame)
        entry_russian.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        create_context_menu(entry_russian)  # Добавляем контекстное меню
        
        # Кнопки перевода
        translate_btn_frame = ttk.Frame(translator_frame)
        translate_btn_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        def translate_to_russian():
            german_text = entry_german.get().strip()
            if not german_text:
                return
                
            api_key = entry_api_key.get().strip()
            if not api_key:
                messagebox.showwarning("Нет ключа", "Введите ключ OpenAI для перевода")
                return
                
            global OPENAI_API_KEY
            OPENAI_API_KEY = api_key
            
            try:
                translation = translate_sentence(german_text, use_openai=True)
                entry_russian.delete(0, tk.END)
                entry_russian.insert(0, translation)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось перевести: {e}")
        
        def translate_to_german():
            russian_text = entry_russian.get().strip()
            if not russian_text:
                return
                
            api_key = entry_api_key.get().strip()
            if not api_key:
                messagebox.showwarning("Нет ключа", "Введите ключ OpenAI для перевода")
                return
                
            global OPENAI_API_KEY
            OPENAI_API_KEY = api_key
            
            if len(russian_text.split()) > 1:
                # Для предложений используем OpenAI
                try:
                    client = get_openai_client(OPENAI_API_KEY)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Ты переводчик с русского на немецкий. Отвечай только переводом предложения без пояснений."},
                            {"role": "user", "content": f"Переведи с русского на немецкий предложение: {russian_text}"}
                        ],
                        max_tokens=100,
                        temperature=0.1
                    )
                    translation = response.choices[0].message.content.strip()
                    entry_german.delete(0, tk.END)
                    entry_german.insert(0, translation)
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось перевести: {e}")
            else:
                # Для отдельных слов используем словарь
                german_word = get_german_translation(russian_text, use_openai=True)
                if german_word:
                    entry_german.delete(0, tk.END)
                    entry_german.insert(0, german_word)
        
        def generate_card():
            german = entry_german.get().strip()
            russian = entry_russian.get().strip()
            
            if not german or not russian:
                messagebox.showwarning("Внимание", "Заполните оба поля перевода.")
                return
                
            # Создаем карточку с переводом
            front = german  # Лицевая сторона - немецкий текст
            # Задняя сторона - немецкий текст + перевод в скобках
            back = f"{german}\n\n({russian})"
            
            # Вставляем карточку
            try:
                insert_card(self.selected_deck_id, front, back,
                            front_image_path=None,
                            back_image_path=None,
                            audio_path=None,
                            level=1)
                messagebox.showinfo("Успех", "Карточка сгенерирована и добавлена с переводом.")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось добавить карточку: {e}")
        
        ttk.Button(translate_btn_frame, text="DE → RU", command=translate_to_russian).pack(side=tk.LEFT, padx=2)
        ttk.Button(translate_btn_frame, text="RU → DE", command=translate_to_german).pack(side=tk.LEFT, padx=2)
        ttk.Button(translate_btn_frame, text="Создать карточку", 
                  command=generate_card).pack(side=tk.RIGHT, padx=2)

        ttk.Label(win, text="Front (лицевая сторона - немецкий текст):").pack(anchor="w", padx=10, pady=(10, 0))
        txt_front = tk.Text(win, height=4)
        txt_front.pack(fill=tk.BOTH, expand=False, padx=10)
        create_context_menu(txt_front)  # Добавляем контекстное меню
        
        # Добавляем кнопки загрузки изображений для Front
        img_front_frame = ttk.Frame(win)
        img_front_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        ttk.Label(img_front_frame, text="Картинка (front):").pack(side=tk.LEFT)
        
        front_image_path_var = tk.StringVar()
        lbl_img_front = ttk.Label(img_front_frame, text="не выбрана")
        lbl_img_front.pack(side=tk.LEFT, padx=5)
        
        def select_img_front():
            filetypes = [
                ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("Все файлы", "*.*"),
            ]
            filename = filedialog.askopenfilename(
                title="Выбрать картинку для front",
                filetypes=filetypes
            )
            if filename:
                front_image_path_var.set(filename)
                lbl_img_front.config(text=os.path.basename(filename))
        
        ttk.Button(img_front_frame, text="Загрузить картинку", command=select_img_front).pack(side=tk.RIGHT, padx=5)
        
        attach_simple_toolbar(win, txt_front)

        ttk.Label(win, text="Back (задняя сторона - ответ с переводом):").pack(anchor="w", padx=10, pady=(10, 0))
        txt_back = tk.Text(win, height=4)
        txt_back.pack(fill=tk.BOTH, expand=False, padx=10)
        create_context_menu(txt_back)  # Добавляем контекстное меню
        
        # Добавляем кнопки загрузки изображений для Back
        img_back_frame = ttk.Frame(win)
        img_back_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        ttk.Label(img_back_frame, text="Картинка (back):").pack(side=tk.LEFT)
        
        back_image_path_var = tk.StringVar()
        lbl_img_back = ttk.Label(img_back_frame, text="не выбрана")
        lbl_img_back.pack(side=tk.LEFT, padx=5)
        
        def select_img_back():
            filetypes = [
                ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("Все файлы", "*.*"),
            ]
            filename = filedialog.askopenfilename(
                title="Выбрать картинку для back",
                filetypes=filetypes
            )
            if filename:
                back_image_path_var.set(filename)
                lbl_img_back.config(text=os.path.basename(filename))
        
        ttk.Button(img_back_frame, text="Загрузить картинку", command=select_img_back).pack(side=tk.RIGHT, padx=5)
        
        attach_simple_toolbar(win, txt_back)

        audio_path_var = tk.StringVar()

        # Кнопка сгенерировать карточку внизу окна
        def save_card():
            front = txt_front.get("1.0", tk.END).strip()
            back = txt_back.get("1.0", tk.END).strip()
            if not front or not back:
                messagebox.showerror("Ошибка", "Front и Back не могут быть пустыми.")
                return
            img_front = front_image_path_var.get().strip() or None
            img_back = back_image_path_var.get().strip() or None
            aud_path = audio_path_var.get().strip() or None
            try:
                insert_card(self.selected_deck_id, front, back,
                            front_image_path=img_front,
                            back_image_path=img_back,
                            audio_path=aud_path,
                            level=1)
            except sqlite3.OperationalError as e:
                messagebox.showerror("БД", f"Не удалось сохранить карточку:\n{e}")
                return
            messagebox.showinfo("OK", "Карточка добавлена.")
            win.destroy()

        # Кнопка сгенерировать карточку внизу окна
        btn_generate_frame = ttk.Frame(win)
        btn_generate_frame.pack(fill=tk.X, padx=10, pady=20)
        ttk.Button(btn_generate_frame, text="Сгенерировать карточку", 
                  command=save_card).pack(side=tk.RIGHT)

    # --------- обзор карточек ---------

    def show_cards_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        cards = get_cards_in_deck(self.selected_deck_id)
        # фильтр по фазе, если выбрана подколода
        if self.selected_phase is not None:
            cards = [c for c in cards if c["leitner_level"] == self.selected_phase]

        if not cards:
            phase_text = f" (фаза {self.selected_phase})" if self.selected_phase is not None else ""
            messagebox.showinfo("Пусто", f"В этой колоде{phase_text} пока нет карточек.")
            return

        win = tk.Toplevel(self)
        win.title("Режим генерации: все карточки (front + back)")
        win.geometry("950x600")
        win.grab_set()

        canvas = tk.Canvas(win)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for c in cards:
            c = sanitize_card_sounds(dict(c))
            card_frame = ttk.LabelFrame(
                scroll_frame,
                text=f"ID {c['id']} (уровень {c['leitner_level']}, next {c['next_review']}, prog {c['progress']})"
            )
            card_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(card_frame, text="FRONT:").pack(anchor="w")
            txt_front = tk.Text(card_frame, height=3)
            txt_front.pack(fill=tk.X, padx=5)
            txt_front.insert("1.0", c["front"])
            create_context_menu(txt_front)  # Добавляем контекстное меню
            attach_simple_toolbar(card_frame, txt_front)

            ttk.Label(card_frame, text="BACK:").pack(anchor="w")
            txt_back = tk.Text(card_frame, height=3)
            txt_back.pack(fill=tk.X, padx=5)
            txt_back.insert("1.0", c["back"])
            create_context_menu(txt_back)  # Добавляем контекстное меню
            attach_simple_toolbar(card_frame, txt_back)

            front_img_var = tk.StringVar(value=c["front_image_path"] or "")
            back_img_var = tk.StringVar(value=c["back_image_path"] or "")

            frame_img = ttk.Frame(card_frame)
            frame_img.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(frame_img, text="Картинка FRONT:").grid(row=0, column=0, sticky="w")
            lbl_front = ttk.Label(
                frame_img,
                text=os.path.basename(front_img_var.get()) if front_img_var.get() else "(нет)"
            )
            lbl_front.grid(row=0, column=1, padx=5, sticky="w")

            def select_front_img(var=front_img_var, lbl=lbl_front):
                filetypes = [
                    ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp"),
                    ("Все файлы", "*.*"),
                ]
                filename = filedialog.askopenfilename(
                    title="Выбрать картинку для FRONT",
                    filetypes=filetypes
                )
                if filename:
                    var.set(filename)
                    lbl.config(text=os.path.basename(filename))

            ttk.Button(frame_img, text="Выбрать...",
                       command=select_front_img).grid(row=0, column=2, padx=5)

            ttk.Label(frame_img, text="Картинка BACK:").grid(row=1, column=0, sticky="w", pady=(3, 0))
            lbl_back = ttk.Label(
                frame_img,
                text=os.path.basename(back_img_var.get()) if back_img_var.get() else "(нет)"
            )
            lbl_back.grid(row=1, column=1, padx=5, sticky="w", pady=(3, 0))

            def select_back_img(var=back_img_var, lbl=lbl_back):
                filetypes = [
                    ("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp"),
                    ("Все файлы", "*.*"),
                ]
                filename = filedialog.askopenfilename(
                    title="Выбрать картинку для BACK",
                    filetypes=filetypes
                )
                if filename:
                    var.set(filename)
                    lbl.config(text=os.path.basename(filename))

            ttk.Button(frame_img, text="Выбрать...",
                       command=select_back_img).grid(row=1, column=2, padx=5, pady=(3, 0))

            audio_entries = get_card_audio_entries(c)

            audio_frame = ttk.LabelFrame(card_frame, text="Аудио")
            audio_frame.pack(fill=tk.X, padx=5, pady=5)

            if audio_entries:
                for idx, entry in enumerate(audio_entries):
                    side_var = tk.StringVar(value=(entry.get("side") or "back"))
                    row_frame = ttk.Frame(audio_frame)
                    row_frame.pack(fill=tk.X, pady=2)

                    ttk.Label(row_frame, text=entry.get("label") or os.path.basename(entry.get("path") or ""))\
                        .pack(side=tk.LEFT, padx=5)

                    def play_selected(path=entry.get("path"), missing=entry.get("missing")):
                        if path and os.path.exists(path) and not missing:
                            play_audio_file(path)
                        else:
                            messagebox.showwarning("Аудио", "Аудио не найдено")

                    ttk.Button(
                        row_frame,
                        text="▶",
                        command=play_selected,
                        state=tk.NORMAL if entry.get("path") and not entry.get("missing") else tk.DISABLED,
                    ).pack(side=tk.LEFT, padx=5)

                    def make_side_updater(media_id, var):
                        def updater():
                            if media_id is not None:
                                update_media_side(media_id, var.get())
                        return updater

                    for side_label, side_value in (("Front", "front"), ("Back", "back")):
                        ttk.Radiobutton(
                            row_frame,
                            text=side_label,
                            variable=side_var,
                            value=side_value,
                            command=make_side_updater(entry.get("media_id"), side_var),
                            state=tk.NORMAL if entry.get("media_id") else tk.DISABLED,
                        ).pack(side=tk.LEFT, padx=2)

                    def make_delete_handler(media_id, frame=row_frame):
                        def handler():
                            if media_id:
                                remove_media_entry(media_id)
                            frame.destroy()
                        return handler

                    ttk.Button(
                        row_frame,
                        text="Удалить привязку",
                        command=make_delete_handler(entry.get("media_id")),
                        state=tk.NORMAL if entry.get("media_id") else tk.DISABLED,
                    ).pack(side=tk.LEFT, padx=5)
            else:
                ttk.Label(audio_frame, text="(Нет аудио)").pack(anchor="w", padx=5, pady=3)

            def make_save_handler(card_id, tf, tb, fimg_var, bimg_var):
                def handler():
                    f = tf.get("1.0", tk.END).strip()
                    b = tb.get("1.0", tk.END).strip()
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE cards SET front = ?, back = ?, front_image_path = ?, back_image_path = ? WHERE id = ?;",
                        (f, b, fimg_var.get() or None, bimg_var.get() or None, card_id)
                    )
                    conn.commit()
                    conn.close()
                    messagebox.showinfo("Сохранено", f"Карточка {card_id} обновлена.")
                return handler

            def make_delete_handler(card_id, frame):
                def handler():
                    if not messagebox.askyesno("Удалить карточку",
                                               f"Точно удалить карточку {card_id}?"):
                        return
                    try:
                        delete_card(card_id)
                    except sqlite3.OperationalError as e:
                        messagebox.showerror("БД", f"Не удалось удалить карточку:\n{e}")
                        return
                    frame.destroy()
                return handler

            btns_frame = ttk.Frame(card_frame)
            btns_frame.pack(anchor="e", padx=5, pady=3)

            ttk.Button(
                btns_frame,
                text="Сохранить изменения",
                command=make_save_handler(c["id"], txt_front, txt_back,
                                          front_img_var, back_img_var)
            ).pack(side=tk.RIGHT, padx=3)

            ttk.Button(
                btns_frame,
                text="Удалить",
                command=make_delete_handler(c["id"], card_frame)
            ).pack(side=tk.RIGHT, padx=3)

    # --------- генерация из текста ---------

    def open_generate_from_text_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        win = tk.Toplevel(self)
        win.title("Авто-генерация из текста")
        win.geometry("650x580")
        win.grab_set()

        ttk.Label(win, text="Вставь текст (немецкие предложения):").pack(anchor="w", padx=10, pady=(10, 0))
        txt = tk.Text(win, height=10)
        txt.pack(fill=tk.BOTH, expand=True, padx=10)
        create_context_menu(txt)  # Добавляем контекстное меню

        # Кнопка вставки из буфера обмена
        insert_frame = ttk.Frame(win)
        insert_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        def paste_from_clipboard():
            try:
                clipboard_text = win.clipboard_get()
                txt.delete("1.0", tk.END)
                txt.insert("1.0", clipboard_text)
            except tk.TclError:
                pass
        
        ttk.Button(insert_frame, text="📋 Вставить из буфера обмена", 
                   command=paste_from_clipboard).pack(side=tk.LEFT)

        frame_opts = ttk.LabelFrame(win, text="Настройки")
        frame_opts.pack(fill=tk.X, padx=10, pady=10)

        use_ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame_opts,
            text="Генерировать картинку для каждой новой карточки (OpenAI)",
            variable=use_ai_var
        ).pack(anchor="w")

        one_sent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame_opts,
            text="1 предложение = 1 карточка (разбивать сложный текст)",
            variable=one_sent_var
        ).pack(anchor="w")

        ttk.Label(frame_opts, text="Шаблон FRONT:").pack(anchor="w", padx=5)
        entry_front = tk.Text(frame_opts, height=2)
        entry_front.pack(fill=tk.X, padx=5)
        entry_front.insert("1.0", self.front_template)
        create_context_menu(entry_front)  # Добавляем контекстное меню

        ttk.Label(frame_opts, text="Шаблон BACK:").pack(anchor="w", padx=5, pady=(5, 0))
        entry_back = tk.Text(frame_opts, height=2)
        entry_back.pack(fill=tk.X, padx=5)
        entry_back.insert("1.0", self.back_template)
        create_context_menu(entry_back)  # Добавляем контекстное меню

        ttk.Label(
            frame_opts,
            text="Переменные: {translation}, {sentence_with_gap}, {word}, {ipa}, {gender}, {plural}, {sentence}"
        ).pack(anchor="w", padx=5, pady=(5, 0))

        progress_frame = ttk.LabelFrame(win, text="Прогресс")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="")
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=5)

        log_box = tk.Text(win, height=6, state="disabled")
        log_box.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        def append_log(message: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, message + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        task_holder = {"task": None}

        def run_generation():
            text = txt.get("1.0", tk.END).strip()
            if not text:
                messagebox.showerror("Ошибка", "Текст пустой.")
                return

            use_ai_images = use_ai_var.get()
            front_t = entry_front.get("1.0", tk.END).strip() or DEFAULT_FRONT_TEMPLATE
            back_t = entry_back.get("1.0", tk.END).strip() or DEFAULT_BACK_TEMPLATE
            self.front_template = front_t
            self.back_template = back_t
            if self.selected_deck_id is not None:
                save_deck_templates(self.selected_deck_id, front_t, back_t)

            api_key = OPENAI_API_KEY if OPENAI_API_KEY else None
            one_sent = one_sent_var.get()

            progress_var.set(0)
            progress_label_var.set("")
            log_box.configure(state="normal")
            log_box.delete("1.0", tk.END)
            log_box.configure(state="disabled")

            def handle_event(event):
                kind = event[0]
                if kind == "progress":
                    done, total, label = event[1:]
                    progress_bar.config(maximum=max(total, 1))
                    progress_var.set(done)
                    progress_label_var.set(f"{label}: {done}/{total}")
                elif kind == "log":
                    append_log(event[1])
                elif kind == "done":
                    result = event[1] or 0
                    self.unregister_bg_handler(task_holder["task"].queue)
                    task_holder["task"] = None
                    btn_generate.config(state=tk.NORMAL)
                    btn_cancel.config(state=tk.DISABLED)
                    if result == 0:
                        messagebox.showinfo("Результат", "Новых слов/предложений не найдено.")
                    else:
                        messagebox.showinfo("Результат", f"Создано карточек (включая синонимы/примеры): {result}")
                    win.destroy()
                elif kind == "error":
                    self.unregister_bg_handler(task_holder["task"].queue)
                    task_holder["task"] = None
                    btn_generate.config(state=tk.NORMAL)
                    btn_cancel.config(state=tk.DISABLED)
                    messagebox.showerror("Ошибка", event[1])

            def worker(task_obj):
                return auto_generate_cards_from_text(
                    self.selected_deck_id, text,
                    use_ai_images, api_key,
                    front_t, back_t,
                    one_sentence_one_card=one_sent,
                    audio_path=None,
                    progress_queue=task_obj.queue,
                    cancel_check=task_obj.cancelled,
                )

            btn_generate.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.NORMAL)
            task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(task_holder["task"].queue, handle_event)

        def cancel_generation():
            if task_holder["task"]:
                task_holder["task"].cancel()
                append_log("Отмена запрошена…")
                btn_cancel.config(state=tk.DISABLED)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        btn_cancel = ttk.Button(btn_frame, text="Отмена", command=cancel_generation, state=tk.DISABLED)
        btn_cancel.pack(side=tk.RIGHT, padx=5)
        btn_generate = ttk.Button(btn_frame, text="Сгенерировать", command=run_generation)
        btn_generate.pack(side=tk.RIGHT)

    # --------- генерация из изображения ---------

    def open_generate_from_image_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return

        if not is_tesseract_available():
            messagebox.showerror("OCR недоступен", "Tesseract OCR не найден.")
            return

        img_path = filedialog.askopenfilename(
            title="Выбери изображение с текстом (страница словаря и т.п.)",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("Все файлы", "*.*"),
            ]
        )
        if not img_path:
            return

        win = tk.Toplevel(self)
        win.title("OCR - Распознавание текста из изображения")
        win.geometry("800x600")
        win.grab_set()

        # Получаем текст через OCR
        try:
            img = Image.open(img_path)
            ocr_text = pytesseract.image_to_string(img)
        except Exception as e:
            messagebox.showerror("Ошибка OCR", f"Не удалось распознать текст: {e}")
            return

        # Основной фрейм с прокруткой
        main_frame = ttk.Frame(win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Фрейм для текста OCR
        ocr_frame = ttk.LabelFrame(main_frame, text="Распознанный текст (редактируйте перед генерацией)")
        ocr_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Текстовая область с прокруткой
        text_frame = ttk.Frame(ocr_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        txt_ocr = tk.Text(text_frame, height=15, wrap=tk.WORD)
        txt_ocr.insert("1.0", ocr_text)
        create_context_menu(txt_ocr)  # Добавляем контекстное меню

        scrollbar = ttk.Scrollbar(text_frame, command=txt_ocr.yview)
        txt_ocr.configure(yscrollcommand=scrollbar.set)

        txt_ocr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Панель инструментов для форматирования
        attach_simple_toolbar(ocr_frame, txt_ocr)

        # Фрейм с настройками генерации
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки генерации карточек")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        use_ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="Генерировать картинку для каждой новой карточки (OpenAI)",
            variable=use_ai_var
        ).pack(anchor="w", padx=10, pady=5)

        # Выбор способа разбиения текста
        ttk.Label(settings_frame, text="Как делить длинный текст:")\
            .pack(anchor="w", padx=10, pady=(5, 0))
        split_mode_var = tk.StringVar(value="sentence")
        ttk.Radiobutton(
            settings_frame,
            text="1 предложение = 1 карточка",
            variable=split_mode_var,
            value="sentence"
        ).pack(anchor="w", padx=25)
        ttk.Radiobutton(
            settings_frame,
            text="Отдельные слова (новое слово = карточка)",
            variable=split_mode_var,
            value="word"
        ).pack(anchor="w", padx=25)

        # Шаблоны
        ttk.Label(settings_frame, text="Шаблон FRONT:").pack(anchor="w", padx=10, pady=(5, 0))
        entry_front = tk.Text(settings_frame, height=2)
        entry_front.pack(fill=tk.X, padx=10, pady=(0, 5))
        entry_front.insert("1.0", self.front_template)
        create_context_menu(entry_front)  # Добавляем контекстное меню

        ttk.Label(settings_frame, text="Шаблон BACK:").pack(anchor="w", padx=10)
        entry_back = tk.Text(settings_frame, height=2)
        entry_back.pack(fill=tk.X, padx=10, pady=(0, 5))
        entry_back.insert("1.0", self.back_template)
        create_context_menu(entry_back)  # Добавляем контекстное меню

        ttk.Label(
            settings_frame,
            text="Переменные: {translation}, {sentence_with_gap}, {word}, {ipa}, {gender}, {plural}, {sentence}"
        ).pack(anchor="w", padx=10, pady=(0, 5))

        progress_frame = ttk.LabelFrame(main_frame, text="Прогресс")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="")
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=10)

        log_box = tk.Text(main_frame, height=6, state="disabled")
        log_box.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        def append_log(message: str):
            log_box.configure(state="normal")
            log_box.insert(tk.END, message + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")

        task_holder = {"task": None}

        def run_generation():
            text = txt_ocr.get("1.0", tk.END).strip()
            if not text:
                messagebox.showerror("Ошибка", "Текст пустой.")
                return

            use_ai_images = use_ai_var.get()
            front_t = entry_front.get("1.0", tk.END).strip() or DEFAULT_FRONT_TEMPLATE
            back_t = entry_back.get("1.0", tk.END).strip() or DEFAULT_BACK_TEMPLATE
            self.front_template = front_t
            self.back_template = back_t
            if self.selected_deck_id is not None:
                save_deck_templates(self.selected_deck_id, front_t, back_t)

            api_key = OPENAI_API_KEY if OPENAI_API_KEY else None
            one_sent = (split_mode_var.get() == "sentence")

            progress_var.set(0)
            progress_label_var.set("")
            log_box.configure(state="normal")
            log_box.delete("1.0", tk.END)
            log_box.configure(state="disabled")

            def handle_event(event):
                kind = event[0]
                if kind == "progress":
                    done, total, label = event[1:]
                    progress_bar.config(maximum=max(total, 1))
                    progress_var.set(done)
                    progress_label_var.set(f"{label}: {done}/{total}")
                elif kind == "log":
                    append_log(event[1])
                elif kind == "done":
                    result = event[1] or 0
                    self.unregister_bg_handler(task_holder["task"].queue)
                    task_holder["task"] = None
                    btn_generate.config(state=tk.NORMAL)
                    btn_cancel.config(state=tk.DISABLED)
                    if result == 0:
                        messagebox.showinfo("Результат", "Новых слов/предложений не найдено.")
                    else:
                        messagebox.showinfo("Результат", f"Создано карточек (включая синонимы/примеры): {result}")
                    win.destroy()
                elif kind == "error":
                    self.unregister_bg_handler(task_holder["task"].queue)
                    task_holder["task"] = None
                    btn_generate.config(state=tk.NORMAL)
                    btn_cancel.config(state=tk.DISABLED)
                    messagebox.showerror("Ошибка", event[1])

            def worker(task_obj):
                return auto_generate_cards_from_text(
                    self.selected_deck_id, text,
                    use_ai_images, api_key,
                    front_t, back_t,
                    one_sentence_one_card=one_sent,
                    audio_path=None,
                    progress_queue=task_obj.queue,
                    cancel_check=task_obj.cancelled,
                )

            btn_generate.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.NORMAL)
            task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(task_holder["task"].queue, handle_event)

        def cancel_generation():
            if task_holder["task"]:
                task_holder["task"].cancel()
                append_log("Отмена запрошена…")
                btn_cancel.config(state=tk.DISABLED)

        # Кнопки
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Копировать текст", 
                   command=lambda: win.clipboard_clear() or win.clipboard_append(txt_ocr.get("1.0", tk.END)))\
            .pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Вставить из буфера", 
                   command=lambda: txt_ocr.insert(tk.END, win.clipboard_get()))\
            .pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Очистить", 
                   command=lambda: txt_ocr.delete("1.0", tk.END))\
            .pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="OCR + генерация", command=run_generation)\
            .pack(side=tk.RIGHT, padx=5)

    # --------- генерация через цифровой слух ---------

    def open_generate_from_speech_window(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        if not SR_AVAILABLE:
            messagebox.showerror(
                "Речь недоступна",
                "SpeechRecognition не установлен.\n"
                "Установи: pip install SpeechRecognition pyaudio"
            )
            return

        win = tk.Toplevel(self)
        win.title("Авто-генерация через цифровой слух")
        win.geometry("520x500")
        win.grab_set()

        ttk.Label(win, text="Длительность записи (сек):").pack(anchor="w", padx=10, pady=(10, 0))
        entry_dur = ttk.Entry(win)
        entry_dur.insert(0, "10")
        entry_dur.pack(fill=tk.X, padx=10)
        create_context_menu(entry_dur)  # Добавляем контекстное меню

        current_mic_text = (
            f"Текущее устройство: index={self.microphone_index}"
            if self.microphone_index is not None else
            "Текущее устройство: по умолчанию системы"
        )
        ttk.Label(win, text=current_mic_text).pack(anchor="w", padx=10, pady=(5, 0))

        frame_opts = ttk.LabelFrame(win, text="Настройки шаблонов и разбиения текста")
        frame_opts.pack(fill=tk.X, padx=10, pady=10)

        use_ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame_opts,
            text="Генерировать картинку для каждой новой карточки (OpenAI)",
            variable=use_ai_var
        ).pack(anchor="w")

        ttk.Label(
            frame_opts,
            text="Как делить распознанный длинный текст:"
        ).pack(anchor="w", padx=5, pady=(5, 0))
        split_mode_var = tk.StringVar(value="sentence")
        ttk.Radiobutton(
            frame_opts,
            text="1 предложение = 1 карточка",
            variable=split_mode_var,
            value="sentence"
        ).pack(anchor="w", padx=15)
        ttk.Radiobutton(
            frame_opts,
            text="Отдельные слова (новое слово = карточка)",
            variable=split_mode_var,
            value="word"
        ).pack(anchor="w", padx=15)

        ttk.Label(frame_opts, text="Шаблон FRONT:").pack(anchor="w", padx=5)
        entry_front = tk.Text(frame_opts, height=2)
        entry_front.pack(fill=tk.X, padx=5)
        entry_front.insert("1.0", self.front_template)
        create_context_menu(entry_front)  # Добавляем контекстное меню

        ttk.Label(frame_opts, text="Шаблон BACK:").pack(anchor="w", padx=5, pady=(5, 0))
        entry_back = tk.Text(frame_opts, height=2)
        entry_back.pack(fill=tk.X, padx=5)
        entry_back.insert("1.0", self.back_template)
        create_context_menu(entry_back)  # Добавляем контекстное меню

        ttk.Label(
            frame_opts,
            text="Переменные: {translation}, {sentence_with_gap}, {word}, {ipa}, {gender}, {plural}, {sentence}"
        ).pack(anchor="w", padx=5, pady=(5, 0))

        progress_frame = ttk.LabelFrame(win, text="Прогресс")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        progress_var = tk.DoubleVar(value=0)
        progress_label_var = tk.StringVar(value="0/0")
        status_var = tk.StringVar(value="")
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=1)
        progress_bar.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(progress_frame, textvariable=progress_label_var).pack(anchor="w", padx=5)
        lbl_status = ttk.Label(progress_frame, textvariable=status_var)
        lbl_status.pack(anchor="w", padx=5, pady=(0, 5))

        dur = 0

        task_holder = {"task": None}

        def handle_event(event):
            kind = event[0]
            if kind == "progress":
                done, total, label = event[1:]
                progress_bar.config(maximum=max(total, 1))
                progress_var.set(done)
                progress_label_var.set(f"{int(done)}/{total}")
                status_var.set(label)
            elif kind == "log":
                status_var.set(event[1])
            elif kind == "done":
                created = event[1] or 0
                self.unregister_bg_handler(task_holder["task"].queue)
                task_holder["task"] = None
                btn_rec.config(state=tk.NORMAL)
                btn_cancel.config(state=tk.DISABLED)
                status_var.set("Готово")
                if created == 0:
                    messagebox.showinfo("Результат", "Новых слов/предложений не найдено.")
                else:
                    messagebox.showinfo("Результат", f"Создано карточек (включая синонимы/примеры): {created}")
                win.destroy()
            elif kind == "error":
                self.unregister_bg_handler(task_holder["task"].queue)
                task_holder["task"] = None
                btn_rec.config(state=tk.NORMAL)
                btn_cancel.config(state=tk.DISABLED)
                status_var.set("")
                messagebox.showerror("Ошибка", event[1])

        def run_generation_thread():
            nonlocal dur
            use_ai_images = use_ai_var.get()
            front_t = entry_front.get("1.0", tk.END).strip() or DEFAULT_FRONT_TEMPLATE
            back_t = entry_back.get("1.0", tk.END).strip() or DEFAULT_BACK_TEMPLATE
            self.front_template = front_t
            self.back_template = back_t
            if self.selected_deck_id is not None:
                save_deck_templates(self.selected_deck_id, front_t, back_t)
            api_key = OPENAI_API_KEY if OPENAI_API_KEY else None
            one_sent = (split_mode_var.get() == "sentence")

            def worker(task_obj):
                return auto_generate_cards_from_speech(
                    self.selected_deck_id, dur,
                    use_ai_images, api_key,
                    front_t, back_t,
                    self.microphone_index,
                    one_sentence_one_card=one_sent,
                    progress_queue=task_obj.queue,
                    cancel_check=task_obj.cancelled,
                )

            progress_var.set(0)
            progress_label_var.set("0/0")
            status_var.set("Запись…")
            btn_rec.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.NORMAL)
            task_holder["task"] = start_background_task(worker)
            self.register_bg_handler(task_holder["task"].queue, handle_event)

        def start_record():
            nonlocal dur
            try:
                dur = int(entry_dur.get().strip())
            except ValueError:
                messagebox.showerror("Ошибка", "Длительность должна быть целым числом секунд.")
                return
            if dur <= 0:
                messagebox.showerror("Ошибка", "Длительность должна быть > 0.")
                return

            remaining = dur

            def tick():
                nonlocal remaining
                if task_holder["task"] is None:
                    return
                if remaining <= 0:
                    status_var.set("Обработка записи…")
                    return
                status_var.set(f"Запись: осталось {remaining} с")
                remaining -= 1
                win.after(1000, tick)

            run_generation_thread()
            tick()

        def cancel_generation():
            if task_holder["task"]:
                task_holder["task"].cancel()
                status_var.set("Отмена запрошена…")
                btn_cancel.config(state=tk.DISABLED)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        btn_cancel = ttk.Button(btn_frame, text="Стоп", command=cancel_generation, state=tk.DISABLED)
        btn_cancel.pack(side=tk.RIGHT, padx=5)
        btn_rec = ttk.Button(btn_frame, text="Записать и сгенерировать", command=start_record)
        btn_rec.pack(side=tk.RIGHT)

    # --------- режимы повторения / воспроизведения ---------

    def start_review(self):
        self.start_repeat_mode()

    def start_repeat_mode(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        deck_id = self.selected_deck_id
        phase_filter = self.selected_phase

        def task(progress_cb):
            cards = get_cards_for_repeat(deck_id)
            if phase_filter is not None:
                cards = [c for c in cards if c["leitner_level"] == phase_filter]
            total = len(cards)
            for idx in range(0, total, max(1, max(1, total // 20))):
                progress_cb(min(idx + 1, total), total, f"Подготовка {min(idx + 1, total)}/{total}")
            return cards

        def on_success(cards):
            if not cards:
                phase_text = f" (фаза {phase_filter})" if phase_filter is not None else ""
                messagebox.showinfo("Повторение", f"В этой колоде{phase_text} пока нет карточек.")
                return
            repeat_window = RepeatWindow(self, cards)

            if hasattr(repeat_window, 'btn_frame'):
                ttk.Button(repeat_window.btn_frame, text="Добавить в ознакомление",
                          command=self.add_cards_to_overview_from_repeat).grid(row=0, column=7, padx=5)

        def on_error(exc: Exception):
            messagebox.showerror("Ошибка", str(exc))

        self.run_task("Режим повторения", "determinate", task, on_success, on_error)

    def start_playback_mode(self):
        if self.selected_deck_id is None:
            messagebox.showwarning("Нет колоды", "Сначала выберите колоду.")
            return
        deck_id = self.selected_deck_id
        phase_filter = self.selected_phase

        def task(progress_cb):
            cards = get_cards_for_playback(deck_id)
            if phase_filter is not None:
                cards = [c for c in cards if c["leitner_level"] == phase_filter]
            total = len(cards)
            for idx in range(0, total, max(1, max(1, total // 20))):
                progress_cb(min(idx + 1, total), total, f"Подготовка {min(idx + 1, total)}/{total}")
            return cards

        def on_success(cards):
            if not cards:
                phase_text = f" (фаза {phase_filter})" if phase_filter is not None else ""
                messagebox.showinfo("Воспроизведение", f"В этой колоде{phase_text} пока нет карточек.")
                return
            ReviewWindow(self, cards)

        def on_error(exc: Exception):
            messagebox.showerror("Ошибка", str(exc))

        self.run_task("Режим воспроизведения", "determinate", task, on_success, on_error)


class OverviewWindow(tk.Toplevel):
    """Режим ознакомления - показываем обе стороны карточки одновременно"""
    
    def  __init__(self, master, cards):
        super().__init__(master)
        self.master = master
        self.cards = [dict(c) for c in cards]
        
        if not self.cards:
            messagebox.showinfo("Пусто", "Нет карточек для ознакомления.")
            self.destroy()
            return
            
        self.current_index = 0
        self.current_card = self.cards[self.current_index]
        
        self.title("Режим ознакомления")
        self.geometry("1400x800")
        self.grab_set()
        
        self.create_widgets()
        self.update_view()
    
    def create_widgets(self):
        # Основной контейнер
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Статус и прогресс бар
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_status = ttk.Label(status_frame, text="")
        self.lbl_status.pack(side=tk.LEFT)
        
        # Прогресс бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))
        
        # Контейнер для двух карточек
        cards_container = ttk.Frame(main_frame)
        cards_container.pack(fill=tk.BOTH, expand=True)
        
        # Левая карточка - лицевая сторона
        self.left_frame = ttk.LabelFrame(cards_container, text="ЛИЦЕВАЯ СТОРОНА", width=650, height=500)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.left_frame.pack_propagate(False)
        
        # Правая карточка - задняя сторона  
        self.right_frame = ttk.LabelFrame(cards_container, text="ЗАДНЯЯ СТОРОНА", width=650, height=500)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self.right_frame.pack_propagate(False)
        
        # Панель кнопок навигации
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.btn_prev = ttk.Button(btn_frame, text="← Назад", command=self.prev_card)
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        
        self.btn_mark_done = ttk.Button(btn_frame, text="✓ Отметить изученным", command=self.mark_as_learned)
        self.btn_mark_done.pack(side=tk.LEFT, padx=10)
        
        self.btn_repeat = ttk.Button(btn_frame, text="🔁 Повторить (в 1 фазу)", command=self.repeat_card)
        self.btn_repeat.pack(side=tk.LEFT, padx=10)
        
        self.btn_sound = ttk.Button(btn_frame, text="🔊 Озвучить", command=self.play_audio)
        self.btn_sound.pack(side=tk.RIGHT, padx=10)

        self.btn_toggle_view = ttk.Button(btn_frame, text="Свернуть карточку", command=self.toggle_view)
        self.btn_toggle_view.pack(side=tk.RIGHT, padx=10)

        self.btn_next = ttk.Button(btn_frame, text="Следующий →", command=self.next_card)
        self.btn_next.pack(side=tk.RIGHT, padx=10)

    def create_card_widgets(self, parent_frame, is_front=True):
        """Создать виджеты для карточки"""
        # Очищаем фрейм
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        # Настройка карточки
        content = ttk.Frame(parent_frame)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Текст с прокруткой
        text_frame = ttk.Frame(content)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Arial", 12),
            bg="white",
            fg="black",
            height=10
        )
        text_widget.config(state='normal')
        scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Изображение
        image_label = ResizableImageLabel(
            content,
            bg="white",
            relief="solid",
            bd=1
        )
        image_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Аудио кнопка
        audio_frame = ttk.Frame(content)
        audio_frame.pack(fill=tk.X, pady=(10, 0))
        audio_frame.audio_widget = AudioPlayerWidget(audio_frame, on_error_callback=self._show_audio_error)
        audio_frame.audio_widget.pack(fill=tk.X)
        audio_frame.audio_widget.pack_forget()

        video_frame = ttk.Frame(content)
        video_frame.pack(fill=tk.X, pady=(5, 0))

        return text_widget, image_label, audio_frame, video_frame
    
    def mark_as_learned(self):
        """Отметить карточку как изученную"""
        card_id = self.current_card["id"]
        update_card_progress(card_id, 100)
        self.current_card["progress"] = 100
        
        # Переместить на более высокий уровень
        current_level = self.current_card["leitner_level"]
        if current_level < 10:
            new_level = min(10, current_level + 2)
            update_card_leitner(card_id, new_level)
            self.current_card["leitner_level"] = new_level
        
        messagebox.showinfo("Успех", "Карточка отмечена как изученная и переведена на более высокий уровень.")
        self.next_card()
    
    def repeat_card(self):
        """Отправить карточку на повторение в 1 фазу"""
        card_id = self.current_card["id"]
        # Обновляем уровень карточки на 1 (первая фаза)
        update_card_leitner(card_id, 1)
        
        # Обновляем текущий объект карточки
        self.current_card["leitner_level"] = 1
        
        # Обновляем статистику
        if hasattr(self.master, 'selected_deck_id') and self.master.selected_deck_id:
            update_statistics(self.master.selected_deck_id, remembered=False, forgotten=True, reviewed=True)
        
        messagebox.showinfo("Повторение", "Карточка отправлена на повторение в 1 фазу.")
        
        # Перелистываем вперед
        self.next_card()
    
    def update_view(self):
        """Обновить отображение обеих сторон карточки"""
        total = len(self.cards)
        idx = self.current_index + 1
        c = self.current_card

        # Обновляем статус
        self.lbl_status.config(text=f"Карточка {idx}/{total} | ID {c['id']} | Уровень: {c['leitner_level']} | Прогресс: {c['progress']}%")
        
        # Обновляем прогресс бар
        self.progress_var.set((idx / total) * 100)
        
        # Создаем виджеты для лицевой стороны
        self.front_text, self.front_image_label, self.front_audio_frame, self.front_video_frame = self.create_card_widgets(self.left_frame, is_front=True)
        
        # Обновляем лицевую сторону (FRONT)
        front_content = c["front"]
        self.front_text.insert(1.0, front_content)
        self.front_text.configure(state='disabled')
        create_context_menu(self.front_text)  # Добавляем контекстное меню
        
        # Загружаем изображение лицевой стороны
        front_img_path = c.get("front_image_path") or c.get("image_path")
        if front_img_path:
            self.front_image_label.load_image(front_img_path)
        else:
            self.front_image_label.config(image="", text="(Нет изображения)")
        
        # Обновляем аудио плеер для лицевой стороны
        self.update_audio_player(self.front_audio_frame, c, prefer_side="front")
        self.update_video_player(self.front_video_frame, c)
        
        # Создаем виджеты для задней стороны
        self.back_text, self.back_image_label, self.back_audio_frame, self.back_video_frame = self.create_card_widgets(self.right_frame, is_front=False)
        
        # Обновляем заднюю сторону (BACK)
        back_content = c["back"]
        
        # ВСЕГДА добавляем перевод для режима ознакомления
        if TRANSLATION_SETTINGS.show_back_translation:
            lines = front_content.split('\n')
            if lines:
                sentence = lines[0].strip()
                if sentence and len(sentence.split()) > 1:
                    translation = translate_sentence(sentence, use_openai=True)
                    if translation and translation != sentence:
                        back_content = f"{back_content}\n\n🇷🇺 Перевод: {translation}"
        
        self.back_text.insert(1.0, back_content)
        self.back_text.configure(state='disabled')
        create_context_menu(self.back_text)  # Добавляем контекстное меню
        
        # Загружаем изображение задней стороны
        back_img_path = c.get("back_image_path") or c.get("image_path")
        if back_img_path:
            self.back_image_label.load_image(back_img_path)
        else:
            self.back_image_label.config(image="", text="(Нет изображения)")
        
        # Обновляем аудио плеер для задней стороны
        self.update_audio_player(self.back_audio_frame, c, prefer_side="back")
        self.update_video_player(self.back_video_frame, c)
    
    def update_audio_player(self, audio_frame, card, prefer_side: str = "back"):
        """Обновить аудио плеер"""
        entries = get_card_audio_entries(card, prefer_side=prefer_side)
        if not hasattr(audio_frame, "audio_widget"):
            audio_frame.audio_widget = AudioPlayerWidget(audio_frame, on_error_callback=self._show_audio_error)
            audio_frame.audio_widget.pack(fill=tk.X)
        display_audio_entries_on_frame(audio_frame, entries)

    def _show_audio_error(self, title: str, message: str):
        try:
            messagebox.showerror(title, message)
        except Exception:
            pass

    def update_video_player(self, video_frame, card):
        """Показать плеер или кнопку открытия видео клипа."""
        for widget in video_frame.winfo_children():
            widget.destroy()

        video_path = find_video_media_path(card)
        if not video_path:
            return

        ttk.Label(video_frame, text="Видео:").pack(side=tk.LEFT, padx=(0, 5))

        if is_vlc_available():
            try:
                player = VlcPlayerWidget(video_frame, video_path, width=320, height=200)
                player.pack(side=tk.LEFT, padx=(0, 10))
                video_frame.vlc_player = player  # сохраняем ссылку, чтобы VLC не выгружался
                ttk.Button(video_frame, text="⏹ Стоп", command=player.stop).pack(side=tk.LEFT)
                return
            except Exception:
                pass

        ttk.Button(
            video_frame,
            text="Открыть во внешнем плеере",
            command=lambda: open_in_external_player(video_path)
        ).pack(side=tk.LEFT)

    def play_audio_file(self, path):
        """Воспроизвести аудио файл"""
        if WINSOUND_AVAILABLE and os.path.exists(path):
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                messagebox.showerror("Ошибка", "Не удалось воспроизвести аудио")
        elif TTS_AVAILABLE:
            speak_text(self.current_card["front"])
        else:
            messagebox.showinfo("Ошибка", "Аудио система недоступна")
    
    def play_audio(self):
        """Озвучить текущую карточку"""
        for widget in [
            getattr(self.front_audio_frame, "audio_widget", None),
            getattr(self.back_audio_frame, "audio_widget", None),
        ]:
            if widget and widget.is_loaded():
                widget.play()
                return

        audio_path = get_card_audio_path(self.current_card, prefer_side="back")
        if audio_path and os.path.exists(audio_path) and WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass
        
        # Если нет аудио файла, озвучиваем текст
        front_text = self.current_card["front"]
        if front_text:
            speak_text(front_text)
    
    def toggle_view(self):
        """Переключить между развернутым и свернутым видом"""
        if not hasattr(self, 'is_minimized') or not self.is_minimized:
            self.is_minimized = True
            self.geometry("800x600")
            self.btn_toggle_view.config(text="Развернуть карточку")
        else:
            self.is_minimized = False
            self.geometry("1400x800")
            self.btn_toggle_view.config(text="Свернуть карточку")
    
    def next_card(self):
        """Перейти к следующей карточке"""
        # Обновляем статистику ознакомления (+1)
        if hasattr(self.master, 'selected_deck_id') and self.master.selected_deck_id:
            update_overview_statistics(self.master.selected_deck_id, increment=1)
        
        # Увеличиваем прогресс текущей карточки
        if self.current_card["progress"] < 100:
            new_progress = min(100, self.current_card["progress"] + 10)
            update_card_progress(self.current_card["id"], new_progress)
            self.current_card["progress"] = new_progress
        
        # Переходим к следующей карточке
        self.current_index += 1
        if self.current_index >= len(self.cards):
            messagebox.showinfo("Готово", "Вы ознакомились со всеми карточками в этой колоде.")
            self.destroy()
            return
        
        self.current_card = self.cards[self.current_index]
        self.update_view()
    
    def prev_card(self):
        """Перейти к предыдущей карточке"""
        if self.current_index > 0:
            # Обновляем статистику ознакомления (-1)
            if hasattr(self.master, 'selected_deck_id') and self.master.selected_deck_id:
                update_overview_statistics(self.master.selected_deck_id, increment=-1)
            
            self.current_index -= 1
            self.current_card = self.cards[self.current_index]
            self.update_view()


class RepeatWindow(tk.Toplevel):
    def __init__(self, master, cards):
        super().__init__(master)
        self.master = master
        self.cards = [dict(c) for c in cards]
        self.current_index = 0
        self.current_card = self.cards[self.current_index]
        self.show_back = False
        self.current_photo = None
        
        # Состояние переводов
        self.show_translations = TRANSLATION_SETTINGS.show_translations
        self.translations_visible = {}
        
        # Для 6-клеточного чекпоинта
        self.checkpoint_vars = []
        self.checkpoint_states = {}
        
        self.title("Режим повторения")
        self.geometry("1000x700")
        self.grab_set()

        self.create_widgets()
        self.update_view()
        self.load_checkpoint_state()

    def create_widgets(self):
        frame_main = ttk.Frame(self)
        frame_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Статус
        self.lbl_status = ttk.Label(frame_main, text="")
        self.lbl_status.pack(anchor="w")

        # Основной фрейм карточки
        self.card_frame = tk.Frame(
            frame_main,
            bg="white",
            bd=2,
            relief="solid",
            width=800,
            height=520
        )
        self.card_frame.pack(pady=10)
        self.card_frame.pack_propagate(False)

        # Уровень карточки
        self.lbl_level = tk.Label(self.card_frame, text="", bg="white",
                                  fg="black", font=("Arial", 10, "bold"))
        self.lbl_level.place(x=5, y=5)

        # Контейнер для контента с прокруткой
        content_container = tk.Frame(self.card_frame, bg="white")
        content_container.place(x=10, y=40, width=780, height=300)

        canvas = tk.Canvas(content_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Фреймы для контента
        self.content_frame = tk.Frame(self.scrollable_frame, bg="white")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Текст
        self.text_frame = tk.Frame(self.content_frame, bg="white")
        self.text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.lbl_text = tk.Label(
            self.text_frame,
            text="",
            bg="white",
            fg="black",
            wraplength=400,
            justify="left",
            font=("Arial", 12)
        )
        self.lbl_text.pack(anchor="w")

        # Изображение с возможностью масштабирования
        self.image_frame = tk.Frame(self.content_frame, bg="white")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_label = ResizableImageLabel(
            self.image_frame,
            bg="white",
            text=""
        )
        self.image_label.pack()

        # Фрейм для 6-клеточного чекпоинта (внизу карточки)
        self.checkpoint_frame = tk.Frame(self.card_frame, bg="white")
        self.checkpoint_frame.place(x=250, y=470, width=300, height=40)
        
        # Создаем 6 чекбоксов в ряд
        self.checkpoint_vars = []
        for i in range(6):
            var = tk.BooleanVar(value=False)
            self.checkpoint_vars.append(var)
            cb = tk.Checkbutton(
                self.checkpoint_frame,
                text=f"✓{i+1}",
                variable=var,
                bg="white",
                command=lambda idx=i: self.update_checkpoint_state(idx)
            )
            cb.pack(side=tk.LEFT, padx=5)

        # Панель кнопок
        self.btn_frame = ttk.Frame(frame_main)
        self.btn_frame.pack(pady=10)

        # Кнопка перевода (для лицевой стороны)
        self.btn_translation = ttk.Button(
            self.btn_frame, 
            text="Скрыть перевод слов" if self.show_translations else "Показать перевод слов",
            command=self.toggle_translations
        )
        self.btn_translation.grid(row=0, column=0, padx=5)

        # Кнопки навигации
        self.btn_prev = ttk.Button(self.btn_frame, text="← Назад", command=self.prev_card)
        self.btn_prev.grid(row=0, column=1, padx=5)

        self.btn_next = ttk.Button(self.btn_frame, text="Вперед →", command=self.next_card)
        self.btn_next.grid(row=0, column=2, padx=5)

        self.btn_show = ttk.Button(self.btn_frame, text="Показать ответ", command=self.toggle_front_back)
        self.btn_show.grid(row=0, column=3, padx=5)

        # Кнопки фаз
        self.btn_forget = ttk.Button(self.btn_frame, text="Забыл (Фаза 1)", command=self.mark_forgotten)
        self.btn_forget.grid(row=0, column=4, padx=5)

        self.btn_remember = ttk.Button(self.btn_frame, text="Повторить (Фаза +1)", command=self.mark_remembered)
        self.btn_remember.grid(row=0, column=5, padx=5)

        # Кнопка звука
        self.btn_sound = ttk.Button(self.btn_frame, text="🔊 Слово", command=self.play_word)
        self.btn_sound.grid(row=0, column=6, padx=5)

        # Добавить аудио-плеер
        self.audio_widget = AudioPlayerWidget(self.card_frame, on_error_callback=self._show_audio_error)
        self.card_frame.audio_widget = self.audio_widget
        self.update_audio_player()

    def update_audio_player(self):
        """Обновить аудио-плеер для текущей карточки"""
        entries = get_card_audio_entries(self.current_card, prefer_side="back" if self.show_back else "front")
        display_audio_entries_on_frame(self.card_frame, entries)

    def _show_audio_error(self, title: str, message: str):
        try:
            messagebox.showerror(title, message)
        except Exception:
            pass

    def get_audio_entries(self) -> list[dict]:
        entries = []
        media = get_media_for_card(self.current_card.get("id"), self.current_card.get("note_id"))
        for item in media:
            media_type = (item.get("media_type") or item.get("type") or "").lower()
            if media_type != "audio":
                continue
            entries.append(
                {
                    "path": item.get("path"),
                    "side": (item.get("side") or "back").lower(),
                    "source": item.get("source"),
                }
            )

        fallback_path = self.current_card.get("audio_path")
        if fallback_path:
            entries.append({"path": fallback_path, "side": "back", "source": None})
        return entries
    
    def play_audio_file(self, path):
        """Воспроизвести аудио файл"""
        if WINSOUND_AVAILABLE and os.path.exists(path):
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                messagebox.showerror("Ошибка", "Не удалось воспроизвести аудио")
        elif TTS_AVAILABLE:
            speak_text(self.current_card["front"])
        else:
            messagebox.showinfo("Ошибка", "Аудио система недоступна")

    def load_checkpoint_state(self):
        """Загрузить состояние чекпоинтов для текущей карточки."""
        card_id = self.current_card["id"]
        if card_id not in self.checkpoint_states:
            self.checkpoint_states[card_id] = [False] * 6
        else:
            for i, state in enumerate(self.checkpoint_states[card_id]):
                self.checkpoint_vars[i].set(state)

    def update_checkpoint_state(self, idx):
        """Обновить состояние чекпоинта."""
        card_id = self.current_card["id"]
        if card_id not in self.checkpoint_states:
            self.checkpoint_states[card_id] = [False] * 6
        self.checkpoint_states[card_id][idx] = self.checkpoint_vars[idx].get()

    def toggle_translations(self):
        """Переключить отображение переводов слов на лицевой стороне."""
        self.show_translations = not self.show_translations
        card_id = self.current_card["id"]
        self.translations_visible[card_id] = self.show_translations
        self.btn_translation.config(
            text="Скрыть перевод слов" if self.show_translations else "Показать перевод слов"
        )
        self.update_view()

    def extract_words_with_translations(self, text):
        """Извлечь слова из текста и добавить переводы из словаря."""
        # Удаляем перевод в скобках если он есть
        text = re.sub(r'\([^)]*\)', '', text).strip()
        
        words = re.findall(r'\b\w+\b', text, re.UNICODE)
        result = []
        
        for word in words:
            if len(word) < 2:  # Пропускаем очень короткие слова
                result.append(word)
                continue
                
            translation = get_translation(word, use_openai=False) if self.show_translations else ""
            if translation:
                # Создаем фрейм для слова и перевода
                word_frame = tk.Frame(self.text_frame, bg="white")
                
                # Слово
                word_label = tk.Label(
                    word_frame,
                    text=word,
                    bg="white",
                    font=("Arial", 12)
                )
                word_label.pack(side=tk.LEFT, padx=(0, 5))
                
                # Перевод
                if self.show_translations:
                    trans_label = tk.Label(
                        word_frame,
                        text=f"({translation})",
                        bg="white",
                        fg="blue",
                        font=("Arial", 10, "italic")
                    )
                    trans_label.pack(side=tk.LEFT)
                
                result.append(word_frame)
            else:
                result.append(word)
        
        return result

    def update_view(self):
        total = len(self.cards)
        idx = self.current_index + 1
        c = self.current_card

        self.lbl_status.config(
            text=f"Карточка {idx}/{total} | ID {c['id']}"
        )

        # Обновляем текст кнопок
        current_level = c["leitner_level"]
        self.btn_forget.config(text=f"Забыл (Фаза 1)")
        next_level = min(10, current_level + 1)
        self.btn_remember.config(text=f"Повторить (Фаза {next_level})")

        romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        lvl = c["leitner_level"]
        phase = romans[min(max(lvl, 1), 10) - 1]
        self.lbl_level.config(text=f"Фаза {phase} | след. повтор: {c['next_review']}")

        if self.show_back:
            # Показываем заднюю сторону
            text = c["back"]
            img_path = c["back_image_path"] or c["front_image_path"] or c["image_path"]
            
            # Очищаем текстовый фрейм и показываем текст в label
            for widget in self.text_frame.winfo_children():
                widget.destroy()
            
            self.lbl_text = tk.Label(
                self.text_frame,
                text=text,
                bg="white",
                fg="black",
                wraplength=400,
                justify="left",
                font=("Arial", 12)
            )
            self.lbl_text.pack(anchor="w")
            
            # Загружаем изображение с возможностью масштабирования
            if img_path:
                self.image_label.load_image(img_path)
            else:
                self.image_label.config(image="", text="(Нет изображения)")
                
        else:
            # Показываем лицевую сторону
            text = c["front"]
            img_path = c["front_image_path"] or c["image_path"]
            
            # Очищаем текстовый фрейм
            for widget in self.text_frame.winfo_children():
                widget.destroy()
                
            if self.show_translations and c["leitner_level"] == 1:
                # В режиме показа перевода и на фазе 1 добавляем переводы
                words_with_translations = self.extract_words_with_translations(text)
                
                # Размещаем слова в текстовом фрейме
                row = 0
                col = 0
                max_cols = 3
                
                for item in words_with_translations:
                    if isinstance(item, tk.Frame):
                        item.grid(row=row, column=col, sticky="w", padx=5, pady=2)
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
                    else:
                        label = tk.Label(
                            self.text_frame,
                            text=item,
                            bg="white",
                            font=("Arial", 12)
                        )
                        label.grid(row=row, column=col, sticky="w", padx=5, pady=2)
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
            else:
                # Просто отображаем текст в label
                self.lbl_text = tk.Label(
                    self.text_frame,
                    text=text,
                    bg="white",
                    fg="black",
                    wraplength=400,
                    justify="left",
                    font=("Arial", 12)
                )
                self.lbl_text.pack(anchor="w")
            
            # Загружаем изображение с возможностью масштабирования
            if img_path:
                self.image_label.load_image(img_path)
            else:
                self.image_label.config(image="", text="(Нет изображения)")

        self.btn_show.config(text="Показать ответ" if not self.show_back else "Показать лицевую сторону")
        
        # Обновляем аудио-плеер
        self.update_audio_player()

    def toggle_front_back(self):
        self.show_back = not self.show_back
        self.update_view()

    def mark_forgotten(self):
        card_id = self.current_card["id"]
        result = apply_srs_update(card_id, 0)
        update_statistics(self.master.selected_deck_id, remembered=False, forgotten=True, reviewed=True)

        if result:
            self.current_card["leitner_level"] = result.get("phase")
            self.current_card["next_review"] = datetime.fromtimestamp(result.get("due", time.time())).isoformat()
            self.current_card["state"] = result.get("state")
            self.update_view()
        self.master.update_overdue_badge()
        messagebox.showinfo("Лейтнер", "Карточка отправлена в 1-й уровень (режим заучивания).")

    def mark_remembered(self):
        card_id = self.current_card["id"]

        result = apply_srs_update(card_id, 2)
        update_statistics(self.master.selected_deck_id, remembered=True, forgotten=False, reviewed=True)

        if result:
            self.current_card["leitner_level"] = result.get("phase")
            self.current_card["next_review"] = datetime.fromtimestamp(result.get("due", time.time())).isoformat()
            self.current_card["state"] = result.get("state")
            self.update_view()
        self.master.update_overdue_badge()
        messagebox.showinfo("Лейтнер", f"Отлично! Уровень карточки теперь: {self.current_card['leitner_level']}")

    def next_card(self):
        self.current_index += 1
        if self.current_index >= len(self.cards):
            messagebox.showinfo("Готово", "Карточки в этом режиме закончились.")
            self.destroy()
            return
        self.current_card = self.cards[self.current_index]
        self.show_back = False
        self.load_checkpoint_state()
        self.update_view()

    def prev_card(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.current_card = self.cards[self.current_index]
            self.show_back = False
            self.load_checkpoint_state()
            self.update_view()

    def play_word(self):
        target_side = "back" if self.show_back else "front"
        audio_path = get_card_audio_path(self.current_card, prefer_side=target_side)
        if getattr(self, "audio_widget", None) and self.audio_widget.is_loaded():
            self.audio_widget.play()
            return
        if audio_path and os.path.exists(audio_path) and WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass

        back = self.current_card["back"]
        first_line = back.splitlines()[0] if back else ""
        word = first_line.split()[0] if first_line else ""
        if not word:
            messagebox.showinfo("Озвучка", "Не удалось выделить слово для озвучки.")
            return
        speak_text(word)


class ReviewWindow(tk.Toplevel):
    def __init__(self, master, cards):
        super().__init__(master)
        self.master = master
        self.cards = [dict(c) for c in cards]
        self.current_index = 0
        self.current_card = self.cards[self.current_index]
        self.show_back = False
        self.current_photo = None

        # Таймеры
        self.auto_flip_id = None
        self.auto_next_id = None
        self.timer_total = 0
        self.timer_left = 0
        self.timer_job = None
        self.timer_label = None
        self.deck_timer = DeckTimerController(
            self,
            self.update_timer_label,
            self.auto_show_answer,
            self.mark_forgotten,
            self.handle_timer_notify,
        )

        # Прогресс
        self.progress_canvas = None
        self.progress_label = None
        
        # Для 6-клеточного чекпоинта
        self.checkpoint_vars = []
        self.checkpoint_states = {}

        self.title("Режим воспроизведения (Лейтнер)")
        self.geometry("900x600")
        self.grab_set()

        self.create_widgets()
        self.update_view()
        self.schedule_timers_for_card()
        self.load_checkpoint_state()

    def create_widgets(self):
        frame_main = ttk.Frame(self)
        frame_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_status = ttk.Label(frame_main, text="")
        self.lbl_status.pack(anchor="w")

        # Таймер
        self.timer_label = tk.Label(
            frame_main,
            text="⏰ 00:00",
            bg=self.cget("bg"),
            fg="red",
            font=("Arial", 11, "bold")
        )
        self.timer_label.pack(anchor="center", pady=(3, 5))

        # Фрейм карточки
        self.card_frame = tk.Frame(
            frame_main,
            bg="white",
            bd=2,
            relief="solid",
            width=700,
            height=420
        )
        self.card_frame.pack(pady=10)
        self.card_frame.pack_propagate(False)

        # Индикатор загрузки
        self.dot_canvas = tk.Canvas(self.card_frame, width=20, height=20,
                                    bg="white", highlightthickness=0)
        self.dot_canvas.place(relx=0.5, rely=0.5, anchor="center")
        self.dot_canvas.create_oval(7, 7, 13, 13, fill="red", outline="red")

        # Уровень
        self.lbl_level = tk.Label(self.card_frame, text="", bg="white",
                                  fg="black", font=("Arial", 10, "bold"))
        self.lbl_level.place(x=5, y=5)

        # Фрейм для 6-клеточного чекпоинта (внизу карточки)
        self.checkpoint_frame = tk.Frame(self.card_frame, bg="white")
        self.checkpoint_frame.place(x=200, y=280, width=300, height=30)
        
        # Создаем 6 чекбоксов в ряд
        self.checkpoint_vars = []
        for i in range(6):
            var = tk.BooleanVar(value=False)
            self.checkpoint_vars.append(var)
            cb = tk.Checkbutton(
                self.checkpoint_frame,
                text=f"✓{i+1}",
                variable=var,
                bg="white",
                command=lambda idx=i: self.update_checkpoint_state(idx)
            )
            cb.pack(side=tk.LEFT, padx=5)

        # Контент
        content_frame = tk.Frame(self.card_frame, bg="white")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(25, 10))

        self.lbl_text = tk.Label(
            content_frame,
            text="",
            bg="white",
            fg="black",
            wraplength=420,
            justify="left",
            font=("Arial", 12)
        )
        self.lbl_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Изображение с возможностью масштабирования
        self.image_label = ResizableImageLabel(
            content_frame,
            bg="white",
            text=""
        )
        self.image_label.pack(side=tk.RIGHT, fill=tk.Y)

        # Прогресс-бар
        progress_frame = tk.Frame(self.card_frame, bg="white")
        progress_frame.pack(side=tk.BOTTOM, pady=(0, 4))

        self.progress_canvas = tk.Canvas(
            progress_frame, width=260, height=14,
            bg="white", highlightthickness=1, highlightbackground="#cccccc"
        )
        self.progress_canvas.pack(side=tk.LEFT, padx=(10, 4))

        self.progress_label = tk.Label(
            progress_frame, text="0 / 100",
            bg="white", fg="black", font=("Arial", 9)
        )
        self.progress_label.pack(side=tk.LEFT, padx=4)

        self.btn_progress_plus = ttk.Button(
            progress_frame, text="+", width=3,
            command=self.increment_progress
        )
        self.btn_progress_plus.pack(side=tk.LEFT, padx=(4, 10))

        self.audio_widget = AudioPlayerWidget(self.card_frame, on_error_callback=self._show_audio_error)
        self.audio_widget.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

        # Панель кнопок
        bottom_frame = tk.Frame(self.card_frame, bg="white")
        bottom_frame.pack(side=tk.BOTTOM, pady=8)

        self.btn_audio_icon = ttk.Button(bottom_frame, text="🔊", width=3, command=self.play_word)
        self.btn_audio_icon.pack(side=tk.LEFT, padx=5)

        self.btn_checkpoint = ttk.Button(
            bottom_frame,
            text="✅ Следующая карточка",
            command=self.goto_next_card
        )
        self.btn_checkpoint.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(frame_main)
        btn_frame.pack(pady=10)

        self.btn_show = ttk.Button(btn_frame, text="Показать ответ", command=self.toggle_front_back)
        self.btn_show.grid(row=0, column=0, padx=5)

        self.btn_forget = ttk.Button(btn_frame, text="Забыл (Фаза 1)", command=self.mark_forgotten)
        self.btn_forget.grid(row=0, column=1, padx=5)

        self.btn_remember = ttk.Button(btn_frame, text="Повторить (Фаза +1)", command=self.mark_remembered)
        self.btn_remember.grid(row=0, column=2, padx=5)

        self.btn_sound = ttk.Button(btn_frame, text="🔊 Слово", command=self.play_word)
        self.btn_sound.grid(row=0, column=3, padx=5)

        # Добавить аудио-плеер
        self.card_frame.audio_widget = self.audio_widget
        self.update_audio_player()

    def update_audio_player(self):
        """Обновить аудио-плеер для текущей карточки"""
        entries = get_card_audio_entries(self.current_card, prefer_side="back" if self.show_back else "front")
        display_audio_entries_on_frame(self.card_frame, entries)

    def _show_audio_error(self, title: str, message: str):
        try:
            messagebox.showerror(title, message)
        except Exception:
            pass
    
    def play_audio_file(self, path):
        """Воспроизвести аудио файл"""
        if WINSOUND_AVAILABLE and os.path.exists(path):
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                messagebox.showerror("Ошибка", "Не удалось воспроизвести аудио")
        elif TTS_AVAILABLE:
            speak_text(self.current_card["front"])
        else:
            messagebox.showinfo("Ошибка", "Аудио система недоступна")

    def load_checkpoint_state(self):
        """Загрузить состояние чекпоинтов для текущей карточки."""
        card_id = self.current_card["id"]
        if card_id in self.checkpoint_states:
            for i, state in enumerate(self.checkpoint_states[card_id]):
                self.checkpoint_vars[i].set(state)
        else:
            for var in self.checkpoint_vars:
                var.set(False)

    def update_checkpoint_state(self, idx):
        """Обновить состояние чекпоинта."""
        card_id = self.current_card["id"]
        if card_id not in self.checkpoint_states:
            self.checkpoint_states[card_id] = [False] * 6
        self.checkpoint_states[card_id][idx] = self.checkpoint_vars[idx].get()

    def cancel_timers(self):
        self.deck_timer.cancel()
        if self.auto_flip_id is not None:
            try:
                self.after_cancel(self.auto_flip_id)
            except Exception:
                pass
            self.auto_flip_id = None

        if self.auto_next_id is not None:
            try:
                self.after_cancel(self.auto_next_id)
            except Exception:
                pass
            self.auto_next_id = None

        if self.timer_job is not None:
            try:
                self.after_cancel(self.timer_job)
            except Exception:
                pass
            self.timer_job = None

    def update_timer_label(self, seconds: int | None = None):
        if self.timer_label is None:
            return
        if seconds is None:
            seconds = max(0, int(self.timer_left))
        m, s = divmod(max(0, int(seconds)), 60)
        self.timer_label.config(text=f"⏰ {m:02d}:{s:02d}")

    def handle_timer_notify(self):
        if self.timer_label is None:
            return
        original_bg = self.timer_label.cget("bg")
        self.timer_label.config(bg="yellow")

        def reset_bg():
            try:
                self.timer_label.config(bg=original_bg)
            except Exception:
                pass

        self.after(1500, reset_bg)

    def timer_tick(self):
        if self.timer_left <= 0:
            self.update_timer_label()
            return
        self.timer_left -= 1
        self.update_timer_label()
        self.timer_job = self.after(1000, self.timer_tick)

    def schedule_timers_for_card(self):
        self.cancel_timers()

        timer_sec, timer_mode = get_effective_timer(getattr(self.master, "selected_deck_id", None))
        if timer_sec and timer_sec > 0:
            self.deck_timer.start(timer_sec, timer_mode)
            return

        front = self.current_card.get("front") or ""
        back = self.current_card.get("back") or ""
        text_len = max(len(front), len(back))

        first_phase = 5

        if text_len <= 35:
            second_phase = 15
        else:
            min_second = 35
            max_second = 5 * 60
            max_len = 500

            clamped_len = min(text_len, max_len)
            if clamped_len <= 35:
                factor = 0.0
            else:
                factor = (clamped_len - 35) / (max_len - 35)

            second_phase = int(min_second + factor * (max_second - min_second))

        total_time = first_phase + second_phase

        self.timer_total = total_time
        self.timer_left = total_time
        self.update_timer_label()
        self.timer_job = self.after(1000, self.timer_tick)

        self.auto_flip_id = self.after(first_phase * 1000, self.auto_show_answer)
        self.auto_next_id = self.after(total_time * 1000, self.auto_mark_and_next)

    def auto_show_answer(self):
        if not self.show_back:
            self.show_back = True
            self.update_view()

    def auto_mark_and_next(self):
        self.cancel_timers()
        card_id = self.current_card["id"]
        try:
            apply_srs_update(card_id, 0)
            update_statistics(self.master.selected_deck_id, remembered=False, forgotten=True, reviewed=True)

            row = get_card_by_id(card_id)
            if row:
                self.current_card["leitner_level"] = row["leitner_level"]
                self.current_card["next_review"] = row["next_review"]
        except Exception:
            pass

        self.master.update_overdue_badge()
        self.goto_next_card()

    def update_progress_view(self):
        if self.progress_canvas is None:
            return
        p = int(self.current_card.get("progress") or 0)
        p = max(0, min(100, p))

        self.progress_canvas.delete("all")

        self.progress_canvas.create_rectangle(1, 1, 259, 13, outline="#cccccc", fill="white")

        if p > 0:
            width = int(258 * p / 100)
            self.progress_canvas.create_rectangle(
                1, 1, 1 + width, 13,
                outline="", fill="#00aa00"
            )

        if self.progress_label is not None:
            self.progress_label.config(text=f"{p} / 100")

    def update_view(self):
        total = len(self.cards)
        idx = self.current_index + 1
        c = self.current_card

        self.lbl_status.config(
            text=f"Карточка {idx}/{total} | ID {c['id']}"
        )

        current_level = c["leitner_level"]
        self.btn_forget.config(text=f"Забыл (Фаза 1)")
        next_level = min(10, current_level + 1)
        self.btn_remember.config(text=f"Повторить (Фаза {next_level})")

        romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        lvl = c["leitner_level"]
        phase = romans[min(max(lvl, 1), 10) - 1]
        self.lbl_level.config(text=f"Фаза {phase} | след. повтор: {c['next_review']}")

        # Загружаем состояние чекпоинтов
        self.load_checkpoint_state()

        if self.show_back:
            text = c["back"]
            img_path = c["back_image_path"] or c["front_image_path"] or c["image_path"]
        else:
            text = c["front"]
            img_path = c["front_image_path"] or c["image_path"]

        self.lbl_text.config(text=text)

        # Загружаем изображение с возможностью масштабирования
        if img_path:
            self.image_label.load_image(img_path)
        else:
            self.image_label.config(image="", text="(Нет изображения)")

        self.btn_show.config(text="Показать ответ" if not self.show_back else "Показать лицевую сторону")

        self.update_progress_view()
        self.update_timer_label()
        
        # Обновляем аудио-плеер
        self.update_audio_player()

    def toggle_front_back(self):
        self.show_back = not self.show_back
        self.update_view()

    def mark_forgotten(self):
        self.cancel_timers()
        card_id = self.current_card["id"]
        result = apply_srs_update(card_id, 0)
        update_statistics(self.master.selected_deck_id, remembered=False, forgotten=True, reviewed=True)

        if result:
            self.current_card["leitner_level"] = result.get("phase")
            self.current_card["next_review"] = datetime.fromtimestamp(result.get("due", time.time())).isoformat()
            self.current_card["state"] = result.get("state")
            self.update_view()
        self.master.update_overdue_badge()
        messagebox.showinfo("Лейтнер", "Карточка отправлена в 1-й уровень (режим заучивания).")
        self.schedule_timers_for_card()

    def mark_remembered(self):
        self.cancel_timers()
        card_id = self.current_card["id"]

        result = apply_srs_update(card_id, 2)
        update_statistics(self.master.selected_deck_id, remembered=True, forgotten=False, reviewed=True)

        if result:
            self.current_card["leitner_level"] = result.get("phase")
            self.current_card["next_review"] = datetime.fromtimestamp(result.get("due", time.time())).isoformat()
            self.current_card["state"] = result.get("state")
            self.update_view()
        self.master.update_overdue_badge()
        messagebox.showinfo("Лейтнер", f"Отлично! Уровень карточки теперь: {self.current_card['leitner_level']}")
        self.goto_next_card()

    def increment_progress(self):
        card_id = self.current_card["id"]
        current = int(self.current_card.get("progress") or 0)
        if current >= 100:
            return
        new_value = min(100, current + 1)
        self.current_card["progress"] = new_value
        update_card_progress(card_id, new_value)
        self.update_progress_view()

    def goto_next_card(self):
        self.cancel_timers()
        self.current_index += 1
        if self.current_index >= len(self.cards):
            messagebox.showinfo("Готово", "Карточки в этом режиме закончились.")
            self.destroy()
            return
        self.current_card = self.cards[self.current_index]
        self.show_back = False
        self.update_view()
        self.schedule_timers_for_card()
        self.load_checkpoint_state()

    def play_word(self):
        target_side = "back" if self.show_back else "front"
        audio_path = get_card_audio_path(self.current_card, prefer_side=target_side)
        if getattr(self, "audio_widget", None) and self.audio_widget.is_loaded():
            self.audio_widget.play()
            return
        if audio_path and os.path.exists(audio_path) and WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass

        back = self.current_card["back"]
        first_line = back.splitlines()[0] if back else ""
        word = first_line.split()[0] if first_line else ""
        if not word:
            messagebox.showinfo("Озвучка", "Не удалось выделить слово для озвучки.")
            return
        speak_text(word)


class AudioEditorWindow:
    """Окно для нарезки аудио из видео на предложения"""

    def __init__(self, master, video_path, deck_id):
        self.master = master
        self.video_path = video_path
        self.deck_id = deck_id

        self.win = tk.Toplevel(master)
        self.win.title("Аудио-редактор: нарезка видео на предложения")
        self.win.geometry("1000x600")
        self.win.grab_set()
        
        # Загружаем аудио из видео
        self.audio_path = None
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0
        self.sentences = []  # [(start_time, end_time, text, audio_segment)]

        if not self.extract_audio_from_video():
            return
        self.create_widgets()
        
    def extract_audio_from_video(self):
        """Извлечь аудио из видео файла"""
        try:
            import tempfile
            import moviepy.editor as mp
            import librosa
            
            # Создаем временный файл для аудио
            temp_dir = tempfile.mkdtemp()
            self.audio_path = os.path.join(temp_dir, "extracted_audio.wav")
            
            # Извлекаем аудио из видео
            video = mp.VideoFileClip(self.video_path)
            audio = video.audio
            audio.write_audiofile(self.audio_path)
            video.close()

            # Загружаем аудио данные
            self.audio_data, self.sample_rate = librosa.load(self.audio_path, sr=None)
            self.duration = len(self.audio_data) / self.sample_rate

            return True

        except ImportError:
            messagebox.showerror("Ошибка", "Для работы с видео установите moviepy и librosa:\n"
                                           "pip install moviepy librosa")
            self.win.destroy()
            return False
        except Exception as e:
            messagebox.showerror("Ошибка", f"{type(e).__name__}: {e}")
            self.win.destroy()
            return False

    def create_widgets(self):
        """Создать интерфейс редактора"""
        # Основной фрейм
        main_frame = ttk.Frame(self.win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Верхняя панель: информация о видео
        info_frame = ttk.LabelFrame(main_frame, text="Информация о видео")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        video_name = os.path.basename(self.video_path)
        duration_min = self.duration / 60
        
        info_text = f"""
        Видео: {video_name}
        Длительность: {duration_min:.2f} минут
        Частота дискретизации: {self.sample_rate} Гц
        """
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Панель управления воспроизведением
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.play_btn = ttk.Button(control_frame, text="▶ Воспроизвести аудио", command=self.play_audio)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.pause_btn = ttk.Button(control_frame, text="⏸ Пауза", command=self.pause_audio)
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Стоп", command=self.stop_audio)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Панель для нарезки на предложения
        split_frame = ttk.LabelFrame(main_frame, text="Нарезка на предложения")
        split_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопки для автоматической нарезки
        btn_frame = ttk.Frame(split_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Автонарезка по тишине", 
                  command=self.auto_split_by_silence).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="Распознать текст для всех сегментов", 
                  command=self.transcribe_all).pack(side=tk.LEFT, padx=2)
        
        # Список предложений
        sentences_frame = ttk.LabelFrame(main_frame, text="Предложения для карточек")
        sentences_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview для отображения предложений
        columns = ("№", "Начало", "Конец", "Текст", "Длина", "Действия")
        self.sentences_tree = ttk.Treeview(sentences_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.sentences_tree.heading(col, text=col)
            self.sentences_tree.column(col, width=80)
        
        self.sentences_tree.column("Текст", width=200)
        self.sentences_tree.column("Действия", width=100)
        
        # Scrollbar для treeview
        scrollbar = ttk.Scrollbar(sentences_frame, orient="vertical", command=self.sentences_tree.yview)
        self.sentences_tree.configure(yscrollcommand=scrollbar.set)
        
        self.sentences_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Панель кнопок внизу
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(bottom_frame, text="Сгенерировать карточки", 
                  command=self.generate_cards).pack(side=tk.RIGHT, padx=2)
        
    def auto_split_by_silence(self):
        """Автоматическая нарезка по тишине"""
        try:
            import librosa
            import numpy as np
            
            # Найти интервалы тишины
            intervals = librosa.effects.split(self.audio_data, 
                                             top_db=30,  # Порог тишины
                                             frame_length=2048,
                                             hop_length=512)
            
            for i, (start, end) in enumerate(intervals):
                start_time = start / self.sample_rate
                end_time = end / self.sample_rate
                duration = end_time - start_time
                
                # Извлечь сегмент аудио
                audio_segment = self.audio_data[start:end]
                
                # Добавить в список
                self.sentences.append({
                    'index': i + 1,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'audio': audio_segment,
                    'text': f"Предложение {i+1}"
                })
                
                # Добавить в treeview
                self.sentences_tree.insert("", "end", values=(
                    i+1,
                    f"{start_time:.2f}с",
                    f"{end_time:.2f}с",
                    f"Предложение {i+1}",
                    f"{duration:.2f}с",
                    "Прослушать"
                ))
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выполнить нарезку: {e}")
            
    def transcribe_all(self):
        """Распознать текст для всех сегментов"""
        if not SR_AVAILABLE:
            messagebox.showerror("Ошибка", "SpeechRecognition не установлен")
            return
            
        r = sr.Recognizer()
        
        for i, sentence in enumerate(self.sentences):
            try:
                # Конвертировать numpy array в аудио данные для распознавания
                import io
                import wave
                import struct
                
                # Создать временный WAV файл
                audio_bytes = self.audio_segment_to_bytes(sentence['audio'])
                
                # Распознать текст
                audio_data = sr.AudioData(audio_bytes, self.sample_rate, 2)
                text = r.recognize_google(audio_data, language="de-DE")
                
                # Обновить текст
                sentence['text'] = text
                
                # Обновить treeview
                item_id = self.sentences_tree.get_children()[i]
                self.sentences_tree.item(item_id, values=(
                    sentence['index'],
                    f"{sentence['start']:.2f}с",
                    f"{sentence['end']:.2f}с",
                    text,
                    f"{sentence['duration']:.2f}с",
                    "Прослушать"
                ))
                
            except Exception as e:
                print(f"Ошибка распознавания сегмента {i}: {e}")
                
    def audio_segment_to_bytes(self, audio_data):
        """Конвертировать numpy array в байты WAV"""
        import io
        import wave
        import struct
        
        # Нормализовать аудио данные
        audio_data = np.int16(audio_data * 32767)
        
        # Создать WAV файл в памяти
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
            
        return wav_buffer.getvalue()
        
    def generate_cards(self):
        """Сгенерировать карточки из предложений"""
        from datetime import datetime
        import os
        
        # Создаем папку для аудио файлов если не существует
        os.makedirs("video_sentences", exist_ok=True)
        
        for sentence in self.sentences:
            if not sentence['text']:
                continue
                
            # Сохранить аудио сегмент в файл
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_filename = f"sentence_{sentence['index']}_{timestamp}.wav"
            audio_path = os.path.join("video_sentences", audio_filename)

            self.save_audio_segment(sentence['audio'], audio_path)

            # Создать карточку
            front = sentence['text']  # Немецкое предложение
            
            # Получить перевод
            translation = translate_sentence(sentence['text'], use_openai=True)
            
            # Формируем заднюю сторону с аудио плеером
            back = f"""{sentence['text']}

🇷🇺 Перевод: {translation}

🔊 Произношение:
[audio:{audio_path}]"""

            note_fields = {
                "word": sentence['text'],
                "translation": translation,
                "example": sentence['text'],
                "level": 1,
                "image": "",
                "front": front,
                "back": back,
                "front_image_path": None,
                "back_image_path": None,
                "audio_path": audio_path,
            }
            create_note_with_cards(
                self.deck_id,
                note_fields,
                note_type_id=ensure_generated_note_type_id(),
            )

        messagebox.showinfo("Успех", f"Создано {len(self.sentences)} карточек")
        self.win.destroy()
        
    def save_audio_segment(self, audio_data, path):
        """Сохранить аудио сегмент в файл"""
        import wave
        import struct
        
        # Нормализовать
        audio_data = np.int16(audio_data * 32767)
        
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    def play_audio(self):
        """Воспроизвести аудио"""
        if self.audio_path and os.path.exists(self.audio_path):
            play_audio_file(self.audio_path)
    
    def pause_audio(self):
        """Пауза аудио"""
        if WINSOUND_AVAILABLE:
            winsound.PlaySound(None, winsound.SND_PURGE)
    
    def stop_audio(self):
        """Остановить аудио"""
        if WINSOUND_AVAILABLE:
            winsound.PlaySound(None, winsound.SND_PURGE)


class VideoEditorWindow(AudioEditorWindow):
    """Окно редактирования видео (использует аудио-редактор)."""

    def __init__(self, parent, video_path, deck_id):
        super().__init__(parent, video_path, deck_id)
        self.win.title("Видео → клипы → карточки")


if __name__ == "__main__":
    init_db()
    init_dictionary()
    app = AnkiApp()
    app.mainloop()
