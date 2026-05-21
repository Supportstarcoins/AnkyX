from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, TYPE_CHECKING

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from card_widget import CardWidget
from chatbot_models import ChatSession, DraftBatch, DraftCard, Message
from csv_importer import upsert_note_and_cards
from db_connect import open_db
from cloud_llm_provider import CloudProviderError, XFlashCloudProvider
from ollama_client import OllamaClient
from pdf_ingest import chunk_sentences, detect_lang, extract_text_from_pdf, split_to_sentences
from llm_json_guard import (
    LLMJsonError,
    SYSTEM_JSON_ONLY,
    call_ollama_json_strict_with_repair,
    ensure_question,
    extract_json_strict,
    is_generic_image_prompt,
    normalize_cards_payload,
)
from llm_engine import (
    LlamaCppEngine,
    LLMEngineBase,
    MockEngine,
    OllamaEngine,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from mock_ai_engine import MockAIEngine
from sd_api_client import SDAPIClient, SDAPIError
from vocab_store import load_known_words, mask_unknown_words

if TYPE_CHECKING:
    from main import RepeatModeCardView

try:
    from tkinterdnd2 import DND_FILES
except Exception:  # noqa: BLE001
    DND_FILES = None


class AutoGrowText(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        *,
        min_lines: int = 1,
        max_lines: int = 10,
        on_send: Callable[[], None] | None = None,
        on_change: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master)
        self.min_lines = min_lines
        self.max_lines = max_lines
        self._on_send = on_send
        self._on_change = on_change
        self._scrollbar_visible = False

        self.text = tk.Text(
            self,
            height=min_lines,
            wrap=tk.WORD,
            relief="flat",
            bd=0,
            **kwargs,
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)

        self.text.bind("<<Modified>>", self._on_modified)
        self.text.bind("<KeyRelease>", self._on_modified)
        self.text.bind("<Return>", self._handle_return)
        self.text.bind("<Shift-Return>", self._handle_shift_return)

        self._menu = tk.Menu(self, tearoff=False)
        self._menu.add_command(label="Вырезать", command=self._cut)
        self._menu.add_command(label="Копировать", command=self._copy)
        self._menu.add_command(label="Вставить", command=self._paste)
        self._menu.add_separator()
        self._menu.add_command(label="Выделить всё", command=self._select_all)
        self.text.bind("<Button-3>", self._show_menu)

    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    def clear(self) -> None:
        self.text.delete("1.0", tk.END)
        self._update_height()
        if self._on_change:
            self._on_change()

    def set_state(self, state: str) -> None:
        self.text.configure(state=state)

    def focus_text(self) -> None:
        self.text.focus_set()

    def _handle_return(self, event: tk.Event) -> str:
        if self._on_send:
            self._on_send()
        return "break"

    def _handle_shift_return(self, _event: tk.Event) -> str:
        self.text.insert(tk.INSERT, "\n")
        return "break"

    def _on_modified(self, _event=None) -> None:
        self.text.edit_modified(False)
        self._update_height()
        if self._on_change:
            self._on_change()

    def _update_height(self) -> None:
        lines = int(self.text.index("end-1c").split(".")[0])
        new_height = max(self.min_lines, min(lines, self.max_lines))
        self.text.configure(height=new_height)
        if lines > self.max_lines:
            if not self._scrollbar_visible:
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self._scrollbar_visible = True
        else:
            if self._scrollbar_visible:
                self.scrollbar.pack_forget()
                self._scrollbar_visible = False

    def _show_menu(self, event: tk.Event) -> None:
        self._menu.tk_popup(event.x_root, event.y_root)

    def _cut(self) -> None:
        self.text.event_generate("<<Cut>>")

    def _copy(self) -> None:
        self.text.event_generate("<<Copy>>")

    def _paste(self) -> None:
        self.text.event_generate("<<Paste>>")

    def _select_all(self) -> None:
        self.text.tag_add("sel", "1.0", "end-1c")


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.internal_frame = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.internal_frame, anchor="nw")

        self.internal_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_frame_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)


class AttachmentsBar(ttk.Frame):
    def __init__(self, master: tk.Widget, *, palette: dict, on_remove: Callable[[int], None]) -> None:
        super().__init__(master)
        self.palette = palette
        self.on_remove = on_remove
        self._chips: list[ttk.Frame] = []

    def render(self, attachments: list[str]) -> None:
        for chip in self._chips:
            chip.destroy()
        self._chips.clear()
        if not attachments:
            return
        for index, path in enumerate(attachments):
            chip = ttk.Frame(self, style="CardInner.TFrame", padding=(6, 3))
            chip.pack(side=tk.LEFT, padx=(0, 6), pady=4)
            name = os.path.basename(path) or "файл"
            label = ttk.Label(chip, text=name, style="Muted.TLabel")
            label.pack(side=tk.LEFT)
            btn = ttk.Button(chip, text="✕", width=2, command=lambda idx=index: self.on_remove(idx))
            btn.pack(side=tk.LEFT, padx=(4, 0))
            self._chips.append(chip)


class RoundSendButton(tk.Canvas):
    def __init__(self, master: tk.Widget, *, command: Callable[[], None], size: int = 36) -> None:
        super().__init__(master, width=size, height=size, highlightthickness=0, bd=0)
        self._command = command
        self._size = size
        self._enabled = True
        self._bg_enabled = "#1E6EFF"
        self._bg_disabled = "#4B4B4B"
        self._fg_enabled = "#FFFFFF"
        self._fg_disabled = "#B0B0B0"
        self._draw_button()
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._draw_button()

    def _draw_button(self) -> None:
        self.delete("all")
        bg = self._bg_enabled if self._enabled else self._bg_disabled
        fg = self._fg_enabled if self._enabled else self._fg_disabled
        pad = 2
        self.create_oval(pad, pad, self._size - pad, self._size - pad, fill=bg, outline=bg)
        self.create_text(self._size // 2, self._size // 2 - 1, text="▲", fill=fg, font=("Segoe UI", 11, "bold"))

    def _on_click(self, _event: tk.Event) -> None:
        if self._enabled:
            self._command()

    def _on_enter(self, _event: tk.Event) -> None:
        self.configure(cursor="hand2" if self._enabled else "")

    def _on_leave(self, _event: tk.Event) -> None:
        self.configure(cursor="")


class CardPreviewWidget(ttk.Frame):
    def __init__(self, master: tk.Widget, palette: dict, on_save: callable) -> None:
        super().__init__(master, style="Card.TFrame", padding=10)
        self.palette = palette
        self.on_save = on_save
        self.cards: list[DraftCard] = []
        self.current_index = 0
        self.show_back = False
        self.total_credits = 0

        nav_frame = ttk.Frame(self, style="CardInner.TFrame")
        nav_frame.pack(fill=tk.X)

        self.prev_btn = ttk.Button(nav_frame, text="◀", width=3, command=self._prev)
        self.prev_btn.pack(side=tk.LEFT)

        self.index_var = tk.StringVar(value="Карточка 0/0")
        ttk.Label(nav_frame, textvariable=self.index_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=8)

        self.next_btn = ttk.Button(nav_frame, text="▶", width=3, command=self._next)
        self.next_btn.pack(side=tk.LEFT)

        self.toggle_btn = ttk.Button(nav_frame, text="Показать BACK", command=self._toggle_side)
        self.toggle_btn.pack(side=tk.RIGHT)

        self.card_widget = CardWidget(
            self,
            palette=self.palette,
            editable=False,
            width=620,
            height=220,
            show_image_toolbar=False,
        )
        self.card_widget.pack(fill=tk.BOTH, expand=True, pady=(8, 6))

        self.media_var = tk.StringVar(value="")
        self.media_label = ttk.Label(self, textvariable=self.media_var, style="Muted.TLabel")
        self.media_label.pack(anchor=tk.W, pady=(0, 6))

        self.save_btn = ttk.Button(self, text="Сохранить карточки", command=self.on_save, style="Primary.TButton")
        self.save_btn.pack(fill=tk.X, pady=(4, 0))

        self._update_ui_state()

    def set_cards(self, cards: list[DraftCard], start_index: int = 0, total_credits: int | None = None) -> None:
        self.cards = list(cards)
        if total_credits is not None:
            self.total_credits = total_credits
        if self.cards:
            self.current_index = max(0, min(start_index, len(self.cards) - 1))
        else:
            self.current_index = 0
        self.show_back = False
        self._render()

    def append_cards(
        self,
        cards: list[DraftCard],
        select_last: bool = True,
        total_credits: int | None = None,
    ) -> None:
        if not cards:
            return
        previous_count = len(self.cards)
        self.cards.extend(cards)
        if total_credits is not None:
            self.total_credits = total_credits
        if select_last:
            self.current_index = len(self.cards) - 1
            self.show_back = False
        elif previous_count == 0:
            self.current_index = 0
        self._render()

    def clear(self) -> None:
        self.cards = []
        self.total_credits = 0
        self.current_index = 0
        self.show_back = False
        self._render()

    def next_card(self) -> None:
        if self.current_index < len(self.cards) - 1:
            self.current_index += 1
            self._render()

    def prev_card(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._render()

    def get_current_card(self) -> DraftCard | None:
        if not self.cards:
            return None
        return self.cards[self.current_index]

    def update_counter_label(self) -> None:
        if not self.cards:
            self.index_var.set("Карточка 0/0")
            return
        self.index_var.set(f"Карточка {self.current_index + 1}/{len(self.cards)}")

    def set_save_state(self, enabled: bool, total_credits: int | None = None) -> None:
        if total_credits is not None:
            self.total_credits = total_credits
        label = "Сохранить карточки"
        if self.total_credits > 0:
            label = f"Сохранить карточки ({self.total_credits} кредитов)"
        self.save_btn.configure(text=label, state=(tk.NORMAL if enabled else tk.DISABLED))

    def _render(self) -> None:
        if not self.cards:
            self.card_widget.set_text("Черновик пуст. Сначала сформируйте карточки.", "")
            self.card_widget.show_side(False)
            self.update_counter_label()
            self.media_var.set("")
            self.toggle_btn.configure(state=tk.DISABLED)
            self.prev_btn.configure(state=tk.DISABLED)
            self.next_btn.configure(state=tk.DISABLED)
            self.set_save_state(False, 0)
            return
        card = self.cards[self.current_index]
        self.card_widget.set_text(card.front, card.back)
        self.card_widget.show_side(self.show_back)
        self.update_counter_label()
        self.media_var.set("Медиа: есть" if card.media else "")
        self.toggle_btn.configure(text="Показать FRONT" if self.show_back else "Показать BACK")
        self.toggle_btn.configure(state=tk.NORMAL)
        allow_nav = len(self.cards) > 1
        self.prev_btn.configure(state=(tk.NORMAL if allow_nav and self.current_index > 0 else tk.DISABLED))
        self.next_btn.configure(state=(tk.NORMAL if allow_nav and self.current_index < len(self.cards) - 1 else tk.DISABLED))
        self.set_save_state(True, self.total_credits)

    def _toggle_side(self) -> None:
        if not self.cards:
            return
        self.show_back = not self.show_back
        self.toggle_btn.configure(text="Показать FRONT" if self.show_back else "Показать BACK")
        self._render()

    def _prev(self) -> None:
        self.prev_card()

    def _next(self) -> None:
        self.next_card()

    def _update_ui_state(self) -> None:
        self.set_save_state(False, 0)
        self.card_widget.set_text("Черновик пуст. Сначала сформируйте карточки.", "")
        self.card_widget.show_side(False)
        self.update_counter_label()
        self.toggle_btn.configure(state=tk.DISABLED)
        self.prev_btn.configure(state=tk.DISABLED)
        self.next_btn.configure(state=tk.DISABLED)


def draft_to_card(draft: DraftCard) -> dict:
    media = getattr(draft, "media", {}) or {}
    meta = getattr(draft, "meta", {}) or {}
    image_path = getattr(draft, "image_path", None) or media.get("image_path") or media.get("image")
    return {
        "id": meta.get("id") or meta.get("temp_id") or -1,
        "deck_id": meta.get("deck_id"),
        "mode": meta.get("mode") or "chatbot_generated",
        "prompt": meta.get("prompt") or "",
        "title": meta.get("title") or (getattr(draft, "front", "") or "")[:120],
        "current_side": meta.get("current_side") or "front",
        "created_at": meta.get("created_at") or meta.get("ts"),
        "source": meta.get("source") or media.get("source") or "chatbot",
        "front": getattr(draft, "front", "") or "",
        "back": getattr(draft, "back", "") or "",
        "front_rich": meta.get("front_rich"),
        "back_rich": meta.get("back_rich"),
        "front_html": meta.get("front_html"),
        "back_html": meta.get("back_html"),
        "front_image_path": media.get("front_image_path") or image_path,
        "back_image_path": media.get("back_image_path"),
        "image_path": image_path,
        "audio_path": media.get("audio_path"),
        "video_path": media.get("video_path"),
        "audio_entries": media.get("audio_entries"),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    raw = text[start : end + 1].strip()
    candidates = [raw]
    candidates.append(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip())
    candidates.append(candidates[-1].replace("“", '"').replace("”", '"').replace("’", "'"))
    candidates.append(re.sub(r",\s*([}\]])", r"\1", candidates[-1]))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_card_from_llm(text: str) -> tuple[str, str] | None:
    payload = _extract_json_object(text)
    if isinstance(payload, dict):
        front = str(payload.get("front") or payload.get("question") or "").strip()
        back = str(payload.get("back") or payload.get("answer") or "").strip()
        if front and back:
            return front, back

    labeled_match = re.search(r"front\s*:\s*(.*?)\s*back\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if labeled_match:
        front = labeled_match.group(1).strip()
        back = labeled_match.group(2).strip()
        if front and back:
            return front, back

    if "||" in text:
        front, back = (part.strip() for part in text.split("||", 1))
        if front and back:
            return front, back
    return None


def _extract_trigger_prompt(text: str, trigger_regex: str) -> str:
    cleaned = re.sub(trigger_regex, "", (text or "").strip(), count=1, flags=re.IGNORECASE)
    return cleaned.strip(" :,-\t")


class ChatBotTab(ttk.Frame):
    SYSTEM_PROMPT = (
        "Ты — тренер по обучению. Помогаешь учить материал по выбранной колоде.\n"
        "1) Отвечай кратко и по делу.\n"
        "2) Если пользователь просит — генерируй карточки в JSON массиве вида:\n"
        "   [{\"front\":\"...\",\"back\":\"...\",\"tags\":[...]}]\n"
        "3) Если контекста мало — задавай уточняющий вопрос.\n"
        "4) Не выдумывай факты, если их нет в сообщениях пользователя."
    )
    LLM_PROVIDER_OLLAMA = "Ollama (HTTP)"
    LLM_PROVIDER_LLAMA = "Локальная Llama 3.1"
    LLM_PROVIDER_CLOUD = "XFLASH Cloud API"
    LLM_PROVIDER_MOCK = "Заглушка"
    CLOUD_MODEL_DEFAULT = "xflash-llama31"
    OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434"
    OLLAMA_MODEL_DEFAULT = "xflash-llama31"
    SD_API_URL_DEFAULT = "http://127.0.0.1:7860"
    FLUX_API_URL_DEFAULT = "http://127.0.0.1:8188"
    FLUX_MODEL_DEFAULT = "models/flux/flux-2-klein-4b-fp8.safetensors"
    SD_CHECKPOINT_DEFAULT = "sd_xl_base_1.0.safetensors"
    SD_NEGATIVE_PROMPT_DEFAULT = "text, watermark, logo, low quality, blurry, deformed, bad anatomy, extra limbs, caption"

    def __init__(self, master: tk.Widget, app) -> None:
        super().__init__(master, style="Surface.TFrame")
        self.app = app
        self.palette = app.palette
        self.card_engine = MockAIEngine()
        self.llm_model_path = self._resolve_llm_model_path()
        self.llama_engine = LlamaCppEngine(
            self.llm_model_path,
            n_ctx=4096,
            n_threads=max(1, (os.cpu_count() or 2) // 2),
            n_gpu_layers=-1,
            verbose=False,
        )
        self.mock_llm_engine = MockEngine()
        self._pending_llm_marker: tuple[str, str] | None = None
        self.current_session_id: int | None = None
        self.current_draft_batch: DraftBatch | None = None
        self.draft_index = 0
        self._render_side = "front"
        self.pending_attachments: list[str] = []
        self._deck_map: dict[str, int] = {}
        self.max_attachment_count = 10
        self.max_attachment_size = 25 * 1024 * 1024
        self._llm_model_hint = self._format_llm_model_hint()
        self._cloud_settings_save_job: str | None = None
        self._cloud_settings_path = self._resolve_cloud_settings_path()
        self._cloud_settings = self._load_cloud_settings()
        self.cloud_url_var = tk.StringVar(value=self._cloud_settings.get("cloud_url", ""))
        self.cloud_api_key_var = tk.StringVar(value=self._cloud_settings.get("api_key", ""))
        self.cloud_status_var = tk.StringVar(value="Cloud: —")
        default_ollama_url = os.getenv("XFLASH_OLLAMA_URL", self.OLLAMA_URL_DEFAULT)
        default_ollama_model = os.getenv("XFLASH_OLLAMA_MODEL", self.OLLAMA_MODEL_DEFAULT)
        self.ollama_url_var = tk.StringVar(value=self._cloud_settings.get("ollama_url") or default_ollama_url)
        self.ollama_model_var = tk.StringVar(value=self._cloud_settings.get("ollama_model") or default_ollama_model)
        self.ollama_status_var = tk.StringVar(value="Ollama: —")
        self.llm_header_status_var = tk.StringVar(value="Ollama: — / Cloud: —")
        self.ollama_engine = OllamaEngine(
            self.ollama_url_var.get().strip(),
            self.ollama_model_var.get().strip() or self.OLLAMA_MODEL_DEFAULT,
        )
        self.llm_engine: LLMEngineBase = self._select_default_llm_engine()
        self.llm_provider_var = tk.StringVar(value=self._get_llm_provider_label(self.llm_engine))
        self.llm_status_var = tk.StringVar(value="")
        self.llm_settings_window: tk.Toplevel | None = None
        self._mousewheel_bound = False
        self.sd_enabled = tk.BooleanVar(value=True)
        self.foreign_mode_var = tk.BooleanVar(value=False)
        self.native_sentences_var = tk.IntVar(value=10)
        self.foreign_sentences_var = tk.IntVar(value=2)
        self._chat_pending_save_cost = 0
        self.pending_cost = 0
        self.flux_api_url_var = tk.StringVar(value=os.getenv("XFLASH_FLUX_API_URL", self._cloud_settings.get("flux_api_url") or self.FLUX_API_URL_DEFAULT))
        self.flux_model_var = tk.StringVar(value=os.getenv("XFLASH_FLUX_MODEL_PATH", self._cloud_settings.get("flux_model_path") or self.FLUX_MODEL_DEFAULT))
        self.image_provider_var = tk.StringVar(value=os.getenv("XFLASH_IMAGE_PROVIDER", self._cloud_settings.get("image_provider") or "flux_comfyui"))
        self.sd_api_url_var = tk.StringVar(
            value=self._cloud_settings.get("legacy_sd_api_url") or self._cloud_settings.get("sd_api_url") or self.SD_API_URL_DEFAULT
        )
        self.sd_checkpoint_var = tk.StringVar(
            value=self._cloud_settings.get("sd_checkpoint") or self.SD_CHECKPOINT_DEFAULT
        )
        self.sd_status_var = tk.StringVar(value="FLUX: —")

        self._ensure_chat_tables()
        self._build_ui()
        self._render_current_draft()
        self._load_sessions()
        self.refresh_deck_options()
        self._lock_chat()
        self._update_llm_status_label()

        self.app.register_balance_observer(self._update_save_button_state)

        self.cloud_url_var.trace_add("write", self._on_cloud_settings_change)
        self.cloud_api_key_var.trace_add("write", self._on_cloud_settings_change)
        self.ollama_url_var.trace_add("write", self._on_ollama_settings_change)
        self.ollama_model_var.trace_add("write", self._on_ollama_settings_change)
        self.sd_api_url_var.trace_add("write", self._on_cloud_settings_change)
        self.flux_api_url_var.trace_add("write", self._on_cloud_settings_change)
        self.flux_model_var.trace_add("write", self._on_cloud_settings_change)
        self.image_provider_var.trace_add("write", self._on_cloud_settings_change)
        self.sd_checkpoint_var.trace_add("write", self._on_cloud_settings_change)
        self.cloud_status_var.trace_add("write", self._on_llm_status_change)
        self.ollama_status_var.trace_add("write", self._on_llm_status_change)
        self._setup_scroll_bindings()

    def destroy(self):
        self.app.unregister_balance_observer(self._update_save_button_state)
        self._unbind_mousewheel()
        super().destroy()

    def _ensure_chat_tables(self) -> None:
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT,
                attachments_json TEXT,
                ts INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()
        conn.close()

    def _resolve_llm_model_path(self) -> str | None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        llama_dir = os.path.join(base_dir, "models", "llm", "llama31")
        llm_dir = os.path.join(base_dir, "models", "llm")
        os.makedirs(llama_dir, exist_ok=True)
        os.makedirs(llm_dir, exist_ok=True)
        for search_dir in (llama_dir, llm_dir):
            if not os.path.isdir(search_dir):
                continue
            candidates = [
                os.path.join(search_dir, name)
                for name in os.listdir(search_dir)
                if name.lower().endswith(".gguf")
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda path: os.path.getsize(path), reverse=True)
            return candidates[0]
        return None

    def _format_llm_model_hint(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        llama_dir = os.path.join(base_dir, "models", "llm", "llama31")
        example_name = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        return f"{llama_dir}\nНапример: {example_name}"

    def _select_default_llm_engine(self) -> LLMEngineBase:
        if self.ollama_engine.is_available():
            return self.ollama_engine
        return self.mock_llm_engine

    def _build_ui(self) -> None:
        self.scrollable_frame = ScrollableFrame(self)
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True)
        self.scroll_canvas = self.scrollable_frame.canvas

        container = ttk.PanedWindow(self.scrollable_frame.internal_frame, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left_frame = ttk.Frame(container, style="Card.TFrame", padding=10)
        container.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Чаты", style="Section.TLabel").pack(anchor=tk.W)

        list_frame = ttk.Frame(left_frame, style="CardInner.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 6))

        self.chats_listbox = tk.Listbox(list_frame, height=8)
        self.chats_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.chats_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chats_listbox.configure(yscrollcommand=list_scroll.set)
        self.chats_listbox.bind("<<ListboxSelect>>", self._on_chat_select)

        self.new_chat_btn = ttk.Button(left_frame, text="Новый чат", command=self._create_new_chat)
        self.new_chat_btn.pack(fill=tk.X, pady=(4, 2))

        self.pro_hint_var = tk.StringVar(value="")
        self.pro_hint_label = ttk.Label(left_frame, textvariable=self.pro_hint_var, style="Muted.TLabel")
        self.pro_hint_label.pack(anchor=tk.W)

        right_frame = ttk.Frame(container, style="Surface.TFrame")
        container.add(right_frame, weight=4)

        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(3, weight=1)

        header_frame = ttk.Frame(right_frame, style="Surface.TFrame")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header_frame.columnconfigure(0, weight=1)

        self.llm_status_label = ttk.Label(
            header_frame,
            textvariable=self.llm_header_status_var,
            style="Muted.TLabel",
        )
        self.llm_status_label.grid(row=0, column=0, sticky="w")

        self.llm_settings_btn = ttk.Button(
            header_frame,
            text="⚙ Настройки LLM",
            command=self._open_llm_settings,
        )
        self.llm_settings_btn.grid(row=0, column=1, sticky="e")

        deck_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=8)
        deck_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(deck_frame, text="Колода:", style="Muted.TLabel").pack(side=tk.LEFT)
        self.deck_var = tk.StringVar(value="")
        self.deck_combo = ttk.Combobox(deck_frame, textvariable=self.deck_var, state="readonly")
        self.deck_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.deck_combo.bind("<<ComboboxSelected>>", self._on_deck_change)

        self.status_var = tk.StringVar(value="Выберите колоду")
        self.status_label = ttk.Label(deck_frame, textvariable=self.status_var, style="Muted.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=(8, 0))

        nav_frame = ttk.Frame(right_frame, style="CardInner.TFrame")
        nav_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self.prev_draft_btn = ttk.Button(nav_frame, text="◀", width=3, command=self._prev_draft)
        self.prev_draft_btn.pack(side=tk.LEFT)

        self.draft_index_var = tk.StringVar(value="0/0")
        ttk.Label(nav_frame, textvariable=self.draft_index_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=8)

        self.next_draft_btn = ttk.Button(nav_frame, text="▶", width=3, command=self._next_draft)
        self.next_draft_btn.pack(side=tk.LEFT)

        from main import RepeatModeCardView

        vpane = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        vpane.grid(row=3, column=0, sticky="nsew")

        render_area = ttk.Frame(vpane, style="Surface.TFrame")
        chat_area = ttk.Frame(vpane, style="Surface.TFrame")

        vpane.add(render_area, weight=8)
        vpane.add(chat_area, weight=2)
        self.after(200, lambda: vpane.sashpos(0, int(vpane.winfo_height() * 0.80)))

        render_inner = ttk.Frame(render_area, style="Surface.TFrame", padding=6)
        render_inner.pack(fill=tk.BOTH, expand=True)

        self.sticky_frame = ttk.Frame(render_inner, style="Surface.TFrame")
        self.sticky_frame.pack(fill=tk.BOTH, expand=True)

        self.card_view: RepeatModeCardView = RepeatModeCardView(
            self.sticky_frame,
            palette=self.palette,
            view_mode="chatbot",
        )
        self.card_view.pack(fill=tk.BOTH, expand=True)
        self.card_view.set_rating_enabled(False)

        self.save_draft_btn = ttk.Button(
            render_area,
            text="Сохранить",
            command=self._save_draft,
            style="Primary.TButton",
        )
        self.save_draft_btn.pack(fill=tk.X, padx=6, pady=(6, 0))
        self.pending_cost_var = tk.StringVar(value="")
        self.pending_cost_label = ttk.Label(render_area, textvariable=self.pending_cost_var, style="Muted.TLabel")
        self.pending_cost_label.pack(fill=tk.X, padx=6, pady=(2, 0))

        draft_actions = ttk.Frame(render_area, style="Surface.TFrame")
        draft_actions.pack(fill=tk.X, padx=6, pady=(4, 0))
        ttk.Button(draft_actions, text="Передняя сторона", command=self._show_front).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(draft_actions, text="Обратная сторона", command=self._show_back).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(draft_actions, text="Удалить карточку", command=self._delete_current_draft).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(draft_actions, text="Сгенерировать картинку", command=self._generate_image_for_current_card).pack(side=tk.RIGHT)

        chat_area.columnconfigure(0, weight=1)
        chat_area.rowconfigure(0, weight=1)
        chat_area.rowconfigure(1, weight=0)

        history_area = ttk.Frame(chat_area, style="Card.TFrame", padding=6)
        history_area.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        self.chat_text = tk.Text(
            history_area,
            wrap=tk.WORD,
            bg="#0b1220",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        history_area.columnconfigure(0, weight=1)
        history_area.rowconfigure(0, weight=1)
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        self.chat_text.configure(state=tk.DISABLED)

        history_scroll = ttk.Scrollbar(history_area, orient="vertical", command=self.chat_text.yview)
        history_scroll.grid(row=0, column=1, sticky="ns")
        self.chat_text.configure(yscrollcommand=history_scroll.set)
        self.chat_text.bind(
            "<MouseWheel>",
            lambda e: (self.chat_text.yview_scroll(int(-1 * (e.delta / 120)), "units"), "break")[1],
        )

        self.composer_frame = ttk.Frame(chat_area, style="Card.TFrame", padding=8)
        self.composer_frame.grid(row=1, column=0, sticky="ew")

        self.composer_row = ttk.Frame(self.composer_frame, style="CardInner.TFrame", padding=6)
        self.composer_row.pack(fill=tk.X)

        self.attach_btn = ttk.Button(self.composer_row, text="📎", width=3, command=self._on_attach)
        self.attach_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.input_min_lines = 2
        self.input_max_lines = 6
        self.input_text = tk.Text(
            self.composer_row,
            height=self.input_min_lines,
            wrap=tk.WORD,
            bg="#0e1626",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_text.bind("<KeyRelease>", self._on_input_keyrelease)
        self.input_text.bind("<Return>", self._on_input_return)
        self._configure_input_copy_paste()

        self.send_btn_var = tk.StringVar(value="Отправить")
        self.send_btn = tk.Button(
            self.composer_row,
            textvariable=self.send_btn_var,
            command=self._on_send,
            bg="#2f6fed",
            fg="#ffffff",
            activebackground="#2a61d4",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        self.send_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.attachments_bar = AttachmentsBar(
            self.composer_frame,
            palette=self.palette,
            on_remove=self._remove_attachment,
        )
        self.attachments_bar.pack(fill=tk.X, pady=(4, 0))

        self._configure_text_tags()
        self._setup_drag_and_drop()
        self._update_send_button_state()
        self._append_chat("system", "Чат готов. Напишите сообщение снизу.")
        self._bind_scroll_widgets()

    def _configure_input_copy_paste(self) -> None:
        self.input_text.bind(
            "<Control-c>",
            lambda e: (self.input_text.event_generate("<<Copy>>"), "break")[1],
        )
        self.input_text.bind(
            "<Control-v>",
            lambda e: (self.input_text.event_generate("<<Paste>>"), "break")[1],
        )
        self.input_text.bind(
            "<Control-x>",
            lambda e: (self.input_text.event_generate("<<Cut>>"), "break")[1],
        )
        self.input_text.bind("<Control-a>", self._select_all_input_text)

        self._input_menu = tk.Menu(self, tearoff=False)
        self._input_menu.add_command(
            label="Copy",
            command=lambda: self.input_text.event_generate("<<Copy>>"),
        )
        self._input_menu.add_command(
            label="Paste",
            command=lambda: self.input_text.event_generate("<<Paste>>"),
        )
        self._input_menu.add_command(
            label="Cut",
            command=lambda: self.input_text.event_generate("<<Cut>>"),
        )
        self._input_menu.add_command(
            label="Select All",
            command=self._select_all_input_text,
        )
        self.input_text.bind("<Button-3>", self._show_input_menu)

    def _select_all_input_text(self, _event=None) -> str:
        self.input_text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _show_input_menu(self, event: tk.Event) -> None:
        self._input_menu.tk_popup(event.x_root, event.y_root)

    def _bind_scroll_widgets(self) -> None:
        for widget in (self.input_text, self.chats_listbox):
            widget.bind("<MouseWheel>", self._on_mousewheel)

    def _setup_scroll_bindings(self) -> None:
        notebook = getattr(self.app, "main_notebook", None)
        if notebook is not None:
            notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed, add="+")
        self._sync_mousewheel_binding()

    def _on_notebook_tab_changed(self, _event=None) -> None:
        self._sync_mousewheel_binding()

    def _sync_mousewheel_binding(self) -> None:
        notebook = getattr(self.app, "main_notebook", None)
        if notebook is None:
            self._bind_mousewheel()
            return
        try:
            current = notebook.nametowidget(notebook.select())
        except Exception:
            self._unbind_mousewheel()
            return
        if current is getattr(self.app, "chatbot_tab_frame", None):
            self._bind_mousewheel()
        else:
            self._unbind_mousewheel()

    def _bind_mousewheel(self) -> None:
        if self._mousewheel_bound:
            return
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._mousewheel_bound = True

    def _unbind_mousewheel(self) -> None:
        if not self._mousewheel_bound:
            return
        self.scroll_canvas.unbind_all("<MouseWheel>")
        self._mousewheel_bound = False

    def _on_mousewheel(self, event: tk.Event) -> str:
        self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _configure_text_tags(self) -> None:
        self.chat_text.tag_configure("user", foreground="#e6e6e6")
        self.chat_text.tag_configure("assistant", foreground="#cfd8ff")
        self.chat_text.tag_configure("meta", foreground="#9aa4b2")
        self.chat_text.tag_configure("system", foreground="#9aa4b2")

    def _setup_drag_and_drop(self) -> None:
        if DND_FILES is None:
            return
        for widget in (self.chat_text, self.composer_frame, self.input_text):
            if hasattr(widget, "drop_target_register"):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop_files)

    def _open_llm_settings(self) -> None:
        if self.llm_settings_window and self.llm_settings_window.winfo_exists():
            self.llm_settings_window.lift()
            self.llm_settings_window.focus_force()
            return
        window = tk.Toplevel(self)
        window.title("Настройки LLM")
        window.transient(self.winfo_toplevel())
        window.grab_set()
        window.resizable(False, False)
        self.llm_settings_window = window

        container = ttk.Frame(window, style="Card.TFrame", padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="LLM настройки", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 6))

        llm_control_frame = ttk.Frame(container, style="CardInner.TFrame", padding=6)
        llm_control_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(llm_control_frame, text="LLM:", style="Muted.TLabel").pack(side=tk.LEFT)
        llm_provider_combo = ttk.Combobox(
            llm_control_frame,
            textvariable=self.llm_provider_var,
            state="readonly",
            values=[
                self.LLM_PROVIDER_OLLAMA,
                self.LLM_PROVIDER_LLAMA,
                self.LLM_PROVIDER_CLOUD,
                self.LLM_PROVIDER_MOCK,
            ],
            width=22,
        )
        llm_provider_combo.pack(side=tk.LEFT, padx=(6, 0))
        llm_provider_combo.bind("<<ComboboxSelected>>", self._on_llm_provider_change)

        llm_status_label = ttk.Label(llm_control_frame, textvariable=self.llm_status_var, style="Muted.TLabel")
        llm_status_label.pack(side=tk.RIGHT)

        ollama_frame = ttk.Frame(container, style="CardInner.TFrame", padding=6)
        ollama_frame.pack(fill=tk.X, pady=(0, 6))
        ollama_frame.columnconfigure(1, weight=1)
        ollama_frame.columnconfigure(3, weight=1)

        ttk.Label(ollama_frame, text="Ollama URL:", style="Muted.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 6),
        )
        ttk.Entry(ollama_frame, textvariable=self.ollama_url_var).grid(row=0, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(ollama_frame, text="Модель:", style="Muted.TLabel").grid(
            row=0,
            column=2,
            sticky="w",
            padx=(0, 6),
        )
        ttk.Entry(ollama_frame, textvariable=self.ollama_model_var).grid(row=0, column=3, sticky="ew")

        ollama_actions = ttk.Frame(ollama_frame, style="CardInner.TFrame")
        ollama_actions.grid(row=0, column=4, sticky="e", padx=(10, 0))
        ttk.Button(ollama_actions, text="Проверить", command=self._check_ollama_connection).pack(side=tk.LEFT)
        ttk.Label(ollama_actions, textvariable=self.ollama_status_var, style="Muted.TLabel").pack(
            side=tk.LEFT,
            padx=8,
        )

        cloud_frame = ttk.Frame(container, style="CardInner.TFrame", padding=6)
        cloud_frame.pack(fill=tk.X, pady=(0, 6))
        cloud_frame.columnconfigure(1, weight=1)
        cloud_frame.columnconfigure(3, weight=1)

        ttk.Label(cloud_frame, text="Cloud URL:", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(cloud_frame, textvariable=self.cloud_url_var).grid(row=0, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(cloud_frame, text="API ключ:", style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Entry(cloud_frame, textvariable=self.cloud_api_key_var, show="•").grid(row=0, column=3, sticky="ew")

        cloud_actions = ttk.Frame(cloud_frame, style="CardInner.TFrame")
        cloud_actions.grid(row=0, column=4, sticky="e", padx=(10, 0))
        ttk.Button(cloud_actions, text="Проверить", command=self._check_cloud_connection).pack(side=tk.LEFT)
        ttk.Label(cloud_actions, textvariable=self.cloud_status_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=8)

        sd_frame = ttk.Frame(container, style="CardInner.TFrame", padding=6)
        sd_frame.pack(fill=tk.X, pady=(0, 6))
        sd_frame.columnconfigure(1, weight=1)
        sd_frame.columnconfigure(3, weight=1)

        ttk.Checkbutton(sd_frame, text="Image backend: FLUX.2 Klein FP8 / ComfyUI", variable=self.sd_enabled).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
        )
        ttk.Label(sd_frame, text="FLUX API URL:", style="Muted.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(sd_frame, textvariable=self.flux_api_url_var).grid(row=1, column=1, sticky="ew", padx=(0, 10))
        ttk.Label(sd_frame, text="FLUX model:", style="Muted.TLabel").grid(
            row=1,
            column=2,
            sticky="w",
            padx=(0, 6),
        )
        ttk.Entry(sd_frame, textvariable=self.flux_model_var).grid(row=1, column=3, sticky="ew")
        sd_actions = ttk.Frame(sd_frame, style="CardInner.TFrame")
        sd_actions.grid(row=1, column=4, sticky="e", padx=(10, 0))
        ttk.Button(sd_actions, text="Проверить FLUX", command=self._check_sd_connection).pack(side=tk.LEFT)
        ttk.Label(sd_actions, textvariable=self.sd_status_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=8)

        pdf_mode_frame = ttk.Frame(container, style="CardInner.TFrame", padding=6)
        pdf_mode_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(pdf_mode_frame, text="Иностранный язык", variable=self.foreign_mode_var).pack(side=tk.LEFT)

        sentences_frame = ttk.Frame(container, style="CardInner.TFrame", padding=6)
        sentences_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(sentences_frame, text="Native предложений/карточку (5-20):", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Spinbox(sentences_frame, from_=5, to=20, textvariable=self.native_sentences_var, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(sentences_frame, text="Foreign предложений/карточку (1-5):", style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=(16, 8))
        ttk.Spinbox(sentences_frame, from_=1, to=5, textvariable=self.foreign_sentences_var, width=6).grid(row=0, column=3, sticky="w")

        def on_close() -> None:
            if self.llm_settings_window and self.llm_settings_window.winfo_exists():
                self.llm_settings_window.grab_release()
                self.llm_settings_window.destroy()
            self.llm_settings_window = None

        window.protocol("WM_DELETE_WINDOW", on_close)
        self._update_llm_status_label()

    def _get_llm_provider_label(self, engine: LLMEngineBase) -> str:
        if engine is self.ollama_engine:
            return self.LLM_PROVIDER_OLLAMA
        if engine is self.llama_engine:
            return self.LLM_PROVIDER_LLAMA
        return self.LLM_PROVIDER_MOCK

    def _on_llm_provider_change(self, _event=None) -> None:
        selected = self.llm_provider_var.get()
        if selected == self.LLM_PROVIDER_CLOUD:
            self.llm_engine = self.mock_llm_engine
        elif selected == self.LLM_PROVIDER_OLLAMA:
            self._refresh_ollama_settings()
            self.llm_engine = self.ollama_engine
        elif selected == self.LLM_PROVIDER_LLAMA:
            self.llm_engine = self.llama_engine
        else:
            self.llm_engine = self.mock_llm_engine
        self._update_llm_status_label()

    def _update_llm_status_label(self) -> None:
        self._refresh_llm_model_path()
        self._refresh_ollama_settings()
        status = self._get_llm_status_text()
        self.llm_status_var.set(status)
        self._update_llm_header_status()

    def _on_llm_status_change(self, *_args) -> None:
        self._update_llm_header_status()

    def _update_llm_header_status(self) -> None:
        self.llm_header_status_var.set(f"{self.ollama_status_var.get()} / {self.cloud_status_var.get()}")

    def _get_llm_status_text(self) -> str:
        if self.llm_provider_var.get() == self.LLM_PROVIDER_OLLAMA:
            return f"Ollama: {self.ollama_engine.get_status()}"
        if self.llm_provider_var.get() == self.LLM_PROVIDER_CLOUD:
            return "Cloud: используется XFLASH API"
        if not self.llm_model_path:
            return f"Llama: не найден GGUF (положите файл сюда {self._llm_model_hint})"
        if not self.llama_engine.is_llama_cpp_available():
            return "Llama: не установлен модуль llama-cpp-python"
        if self.llm_engine is self.llama_engine:
            return f"Llama: {self.llama_engine.get_status()}"
        return "Llama: заглушка"

    def _refresh_llm_model_path(self) -> None:
        latest_path = self._resolve_llm_model_path()
        if latest_path != self.llm_model_path:
            self.llm_model_path = latest_path
            self.llama_engine.model_path = latest_path

    def _refresh_ollama_settings(self) -> None:
        self.ollama_engine.base_url = self.ollama_url_var.get().strip()
        model = self.ollama_model_var.get().strip()
        self.ollama_engine.model = model or self.OLLAMA_MODEL_DEFAULT

    def _on_drop_files(self, event: tk.Event) -> str:
        if str(self.attach_btn.cget("state")) == str(tk.DISABLED):
            return "break"
        paths = list(self.tk.splitlist(event.data))
        self._add_attachments(paths)
        return "break"

    def _update_send_button_state(self) -> None:
        if str(self.input_text.cget("state")) == str(tk.DISABLED):
            self.send_btn.configure(state=tk.DISABLED)
            return
        text = self.input_text.get("1.0", "end-1c").strip()
        enabled = bool(text) or bool(self.pending_attachments)
        self.send_btn.configure(state=(tk.NORMAL if enabled else tk.DISABLED))

    def _on_input_keyrelease(self, _event=None) -> None:
        self._resize_input_text()
        self._update_send_button_state()

    def _on_input_return(self, _event: tk.Event) -> str | None:
        if _event.state & 0x0001:
            return None
        self._on_send()
        return "break"

    def _resize_input_text(self) -> None:
        lines = int(self.input_text.index("end-1c").split(".")[0])
        new_height = max(self.input_min_lines, min(lines, self.input_max_lines))
        self.input_text.configure(height=new_height)

    def _is_allowed_attachment(self, path: str) -> bool:
        _, ext = os.path.splitext(path.lower())
        allowed = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".pdf",
            ".txt",
            ".docx",
            ".mp3",
            ".wav",
            ".ogg",
            ".mp4",
            ".mkv",
        }
        return ext in allowed

    def _format_size(self, size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}GB"

    def refresh_deck_options(self) -> None:
        values = []
        self._deck_map = {}
        for deck in self.app.decks:
            label = f"{deck['id']}: {deck['name']}"
            values.append(label)
            self._deck_map[label] = deck["id"]
        self.deck_combo.configure(values=values)
        if values and self.deck_var.get() in values:
            return
        self.deck_var.set("")
        self._lock_chat()

    def _lock_chat(self) -> None:
        self.input_text.configure(state=tk.DISABLED)
        self.send_btn.configure(state=tk.DISABLED)
        self.attach_btn.configure(state=tk.DISABLED)
        self.status_var.set("Выберите колоду")

    def _unlock_chat(self) -> None:
        self.input_text.configure(state=tk.NORMAL)
        self.attach_btn.configure(state=tk.NORMAL)
        self.status_var.set("")
        self._update_send_button_state()

    def _on_deck_change(self, _event=None) -> None:
        selection = self.deck_var.get()
        deck_id = self._deck_map.get(selection)
        if deck_id is None:
            self._lock_chat()
            return
        self._unlock_chat()

    def _load_sessions(self) -> None:
        conn = open_db()
        cur = conn.cursor()
        cur.execute("SELECT id, title, created_at FROM chat_sessions ORDER BY created_at ASC;")
        rows = cur.fetchall()
        conn.close()
        self.chats_listbox.delete(0, tk.END)
        for row in rows:
            self.chats_listbox.insert(tk.END, row["title"])
        self._update_new_chat_state(len(rows))
        if rows:
            self.chats_listbox.selection_set(0)
            self._select_session(rows[0]["id"])

    def _update_new_chat_state(self, existing_count: int) -> None:
        plan = self.app.get_pricing_plan()
        if plan == "free" and existing_count >= 1:
            self.new_chat_btn.configure(state=tk.DISABLED)
            self.pro_hint_var.set("Доступно в PRO")
        else:
            self.new_chat_btn.configure(state=tk.NORMAL)
            self.pro_hint_var.set("")

    def _create_new_chat(self) -> None:
        plan = self.app.get_pricing_plan()
        if plan == "free":
            conn = open_db()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS cnt FROM chat_sessions;")
            count = int(cur.fetchone()["cnt"])
            conn.close()
            if count >= 1:
                messagebox.showinfo("Ограничение", "Новый чат доступен в PRO.")
                self._update_new_chat_state(count)
                return
        title = f"Чат {int(time.time())}"
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (title, created_at) VALUES (?, ?);",
            (title, int(time.time())),
        )
        session_id = cur.lastrowid
        conn.commit()
        conn.close()
        self.chats_listbox.insert(tk.END, title)
        self.chats_listbox.selection_clear(0, tk.END)
        self.chats_listbox.selection_set(tk.END)
        self._select_session(session_id)
        self._update_new_chat_state(self.chats_listbox.size())

    def _on_chat_select(self, _event=None) -> None:
        selection = self.chats_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM chat_sessions ORDER BY created_at ASC LIMIT 1 OFFSET ?;",
            (index,),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            self._select_session(row["id"])

    def _select_session(self, session_id: int) -> None:
        self.current_session_id = session_id
        self._load_messages(session_id)

    def _load_messages(self, session_id: int) -> None:
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT role, text, attachments_json, ts FROM chat_messages WHERE session_id = ? ORDER BY ts ASC;",
            (session_id,),
        )
        rows = cur.fetchall()
        conn.close()
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.configure(state=tk.DISABLED)
        if not rows:
            self._append_chat("system", "Чат готов. Напишите сообщение снизу.")
            return
        for row in rows:
            attachments: list[str] = []
            if row["attachments_json"]:
                try:
                    attachments = self._normalize_attachments(json.loads(row["attachments_json"]))
                except Exception:
                    attachments = []
            message = Message(role=row["role"], text=row["text"] or "", attachments=attachments)
            self._append_message_to_ui(message)

    def _append_message_to_ui(self, message: Message) -> None:
        self._append_chat(message.role, message.text, message.attachments)

    def append_chat(self, role: str, text: str, attachments: list[str] | None = None) -> None:
        self._append_chat(role, text, attachments)

    def _append_chat(self, role: str, text: str, attachments: list[str] | None = None) -> None:
        prefix = ""
        tag = "meta"
        if role == "user":
            prefix = "Вы: "
            tag = "user"
        elif role == "assistant":
            prefix = "Ассистент: "
            tag = "assistant"
        elif role in ("system", "meta"):
            prefix = "Система: " if role == "system" else ""
            tag = "system" if role == "system" else "meta"
        payload = text or ""
        self.chat_text.configure(state=tk.NORMAL)
        if prefix:
            self.chat_text.insert(tk.END, prefix, tag)
        if payload:
            self.chat_text.insert(tk.END, payload, tag)
        if attachments:
            attachment_names = ", ".join(
                [os.path.basename(path) for path in attachments if os.path.basename(path)]
            )
            if attachment_names:
                if payload or prefix:
                    self.chat_text.insert(tk.END, "\n")
                self.chat_text.insert(tk.END, f"📎 {attachment_names}", "meta")
        self.chat_text.insert(tk.END, "\n\n")
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def _append_message(self, role: str, text: str, attachments: list[str] | None = None) -> None:
        if self.current_session_id is None:
            return
        attachments_json = json.dumps(attachments or [], ensure_ascii=False)
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, text, attachments_json, ts) VALUES (?, ?, ?, ?, ?);",
            (self.current_session_id, role, text, attachments_json, int(time.time())),
        )
        conn.commit()
        conn.close()
        message = Message(role=role, text=text, attachments=attachments or [])
        self._append_message_to_ui(message)

    def _normalize_attachments(self, raw: Any) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            normalized: list[str] = []
            for item in raw:
                if isinstance(item, str):
                    normalized.append(item)
                    continue
                if isinstance(item, dict):
                    candidate = item.get("path") or item.get("name")
                    if candidate:
                        normalized.append(str(candidate))
            return normalized
        return []

    def _format_message_payload(self, text: str, attachments: list[str] | None) -> str:
        payload = text or ""
        if attachments:
            attachments_info = ", ".join(
                [os.path.basename(path) for path in attachments if os.path.basename(path)]
            )
            if attachments_info:
                payload = f"{payload}\n📎 {attachments_info}" if payload else f"📎 {attachments_info}"
        return payload

    def _build_cloud_attachment_metadata(self, attachments: list[str]) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        for path in attachments:
            if not path:
                continue
            name = os.path.basename(path)
            if not name:
                continue
            size = 0
            try:
                if os.path.exists(path):
                    size = os.path.getsize(path)
            except Exception:  # noqa: BLE001
                size = 0
            metadata.append({"name": name, "size": size})
        return metadata

    def _append_temporary_message(self, text: str, role: str = "assistant") -> None:
        prefix = "Ассистент:" if role == "assistant" else "Система:"
        tag = "assistant" if role == "assistant" else "system"
        self.chat_text.configure(state=tk.NORMAL)
        start_index = self.chat_text.index(tk.END)
        self.chat_text.insert(tk.END, f"{prefix} ", tag)
        self.chat_text.insert(tk.END, text)
        self.chat_text.insert(tk.END, "\n\n")
        end_index = self.chat_text.index(tk.END)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)
        self._pending_llm_marker = (start_index, end_index)

    def _remove_temporary_message(self) -> None:
        if not self._pending_llm_marker:
            return
        start_index, end_index = self._pending_llm_marker
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete(start_index, end_index)
        self.chat_text.configure(state=tk.DISABLED)
        self._pending_llm_marker = None

    def _build_chat_messages(self, *, for_cloud: bool) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if self.current_session_id is None:
            return messages
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT role, text, attachments_json FROM chat_messages WHERE session_id = ? ORDER BY ts ASC;",
            (self.current_session_id,),
        )
        rows = cur.fetchall()
        conn.close()
        for row in rows:
            role = row["role"]
            if role not in ("user", "assistant"):
                continue
            attachments: list[str] = []
            if row["attachments_json"]:
                try:
                    attachments = self._normalize_attachments(json.loads(row["attachments_json"]))
                except Exception:
                    attachments = []
            text = row["text"] or ""
            if for_cloud:
                messages.append(
                    {
                        "role": role,
                        "content": text,
                        "attachments": self._build_cloud_attachment_metadata(attachments),
                    }
                )
            else:
                payload = self._format_message_payload(text, attachments)
                messages.append({"role": role, "content": payload})
        return messages

    def _resolve_llm_engine(self) -> tuple[LLMEngineBase, str | None]:
        self._refresh_llm_model_path()
        selected = self.llm_provider_var.get()
        if selected == self.LLM_PROVIDER_OLLAMA:
            self._refresh_ollama_settings()
            if self.ollama_engine.is_available():
                return self.ollama_engine, None
            return self.mock_llm_engine, "[LLM OFFLINE] Запусти Ollama и проверь модель xflash-llama31"
        if selected == self.LLM_PROVIDER_CLOUD:
            return self.mock_llm_engine, None
        if selected == self.LLM_PROVIDER_LLAMA:
            if self.llama_engine.is_available():
                return self.llama_engine, None
            if not self.llm_model_path:
                return self.mock_llm_engine, (
                    "LLM недоступна: не найден GGUF.\n"
                    f"Положите файл сюда:\n{self._llm_model_hint}"
                )
            if not self.llama_engine.is_llama_cpp_available():
                return self.mock_llm_engine, "LLM недоступна: не установлен модуль llama-cpp-python."
            return self.mock_llm_engine, "LLM недоступна из-за ошибки загрузки."
        return self.mock_llm_engine, None

    def _resolve_cloud_settings_path(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "chatbot_settings.json")

    def _load_cloud_settings(self) -> dict[str, str]:
        try:
            if os.path.exists(self._cloud_settings_path):
                with open(self._cloud_settings_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return {
                        "cloud_url": str(payload.get("cloud_url") or ""),
                        "api_key": str(payload.get("api_key") or ""),
                        "ollama_url": str(payload.get("ollama_url") or ""),
                        "ollama_model": str(payload.get("ollama_model") or ""),
                        "sd_api_url": str(payload.get("sd_api_url") or ""),
                        "sd_checkpoint": str(payload.get("sd_checkpoint") or ""),
                        "image_provider": str(payload.get("image_provider") or "flux_comfyui"),
                        "flux_api_url": str(payload.get("flux_api_url") or ""),
                        "flux_model_path": str(payload.get("flux_model_path") or ""),
                        "legacy_sd_api_url": str(payload.get("legacy_sd_api_url") or ""),
                    }
        except Exception:  # noqa: BLE001
            logging.exception("Failed to load cloud settings")
        return {
            "cloud_url": "",
            "api_key": "",
            "ollama_url": "",
            "ollama_model": "",
            "sd_api_url": "",
            "sd_checkpoint": "",
            "image_provider": "flux_comfyui",
            "flux_api_url": "",
            "flux_model_path": "",
            "legacy_sd_api_url": "",
        }

    def _save_cloud_settings(self) -> None:
        self._cloud_settings_save_job = None
        payload = {
            "cloud_url": self.cloud_url_var.get().strip(),
            "api_key": self.cloud_api_key_var.get().strip(),
            "ollama_url": self.ollama_url_var.get().strip(),
            "ollama_model": self.ollama_model_var.get().strip(),
            "sd_api_url": self.sd_api_url_var.get().strip(),
            "sd_checkpoint": self.sd_checkpoint_var.get().strip(),
            "image_provider": self.image_provider_var.get().strip() or "flux_comfyui",
            "flux_api_url": self.flux_api_url_var.get().strip(),
            "flux_model_path": self.flux_model_var.get().strip(),
            "legacy_sd_api_url": self.sd_api_url_var.get().strip(),
        }
        try:
            with open(self._cloud_settings_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            logging.exception("Failed to save cloud settings")

    def _on_cloud_settings_change(self, *_args) -> None:
        if self._cloud_settings_save_job:
            self.after_cancel(self._cloud_settings_save_job)
        self._cloud_settings_save_job = self.after(400, self._save_cloud_settings)

    def _on_ollama_settings_change(self, *_args) -> None:
        self._refresh_ollama_settings()
        self._update_llm_status_label()
        self._on_cloud_settings_change()

    def _get_cloud_provider(self) -> XFlashCloudProvider | None:
        base_url = self.cloud_url_var.get().strip()
        api_key = self.cloud_api_key_var.get().strip()
        if not base_url or not api_key:
            return None
        return XFlashCloudProvider(base_url, api_key)

    def _check_cloud_connection(self) -> None:
        provider = self._get_cloud_provider()
        if not provider:
            self.cloud_status_var.set("Cloud: Invalid key")
            return
        self.cloud_status_var.set("Cloud: проверка...")

        def worker() -> None:
            try:
                provider.chat(
                    messages=[{"role": "user", "content": "ping"}],
                    chat_id=self.current_session_id,
                    model=self.CLOUD_MODEL_DEFAULT,
                    temperature=0.0,
                    max_tokens=1,
                )
            except CloudProviderError as exc:
                self.after(0, lambda: self._update_cloud_status_from_error(exc))
                return
            self.after(0, lambda: self.cloud_status_var.set("Cloud: OK"))

        threading.Thread(target=worker, daemon=True).start()

    def _check_ollama_connection(self) -> None:
        self._refresh_ollama_settings()
        self.ollama_status_var.set("Ollama: проверка...")

        def worker() -> None:
            try:
                ok = self.ollama_engine.is_available()
            except Exception:  # noqa: BLE001
                ok = False
            status = "Ollama: OK" if ok else "Ollama: offline"
            self.after(0, lambda: self.ollama_status_var.set(status))

        threading.Thread(target=worker, daemon=True).start()

    def _check_sd_connection(self) -> None:
        api_url = self.sd_api_url_var.get().strip() or self.SD_API_URL_DEFAULT
        self.sd_status_var.set("FLUX: проверка...")

        def worker() -> None:
            ok = False
            try:
                from flux_image_adapter import FluxImageAdapter
                ok, msg = FluxImageAdapter(api_url=api_url, model_path=self.flux_model_var.get().strip() or self.FLUX_MODEL_DEFAULT).is_available()
            except SDAPIError:
                ok = False
            status = "FLUX: OK" if ok else "FLUX: FAIL"
            self.after(0, lambda: self.sd_status_var.set(status))
            self.after(0, lambda: messagebox.showinfo("FLUX", status))

        threading.Thread(target=worker, daemon=True).start()

    def _update_cloud_status_from_error(self, exc: CloudProviderError) -> None:
        if exc.status_code == 401:
            self.cloud_status_var.set("Cloud: Invalid key")
            return
        if exc.status_code == 503:
            self.cloud_status_var.set("Cloud: Offline")
            return
        if exc.status_code == 429:
            self.cloud_status_var.set("Cloud: Offline")
            return
        if exc.status_code == 402:
            self.cloud_status_var.set("Cloud: Offline")
            return
        self.cloud_status_var.set("Cloud: Offline")

    def _sync_cloud_credits(self, remaining_credits: int | None, credits_spent: int | None = None) -> None:
        if remaining_credits is None:
            return
        current = self.app.credits_service.get_balance(self.app.user_id)
        if remaining_credits == current:
            return
        delta = remaining_credits - current
        reason = "Синхронизация баланса (Cloud LLM)"
        meta = {"credits_spent": credits_spent, "source": "chatbot"}
        if delta > 0:
            self.app.credits_service.add_credits(self.app.user_id, delta, reason, meta=meta)
        elif delta < 0:
            self.app.credits_service.spend_credits(self.app.user_id, abs(delta), reason, meta=meta)
        self.app.refresh_balance_ui()

    def _resolve_chat_backend(
        self,
    ) -> tuple[str, LLMEngineBase | XFlashCloudProvider, str | None]:
        selected = self.llm_provider_var.get()
        if selected == self.LLM_PROVIDER_CLOUD:
            provider = self._get_cloud_provider()
            if not provider:
                return "local", self.mock_llm_engine, "Cloud API не настроен, используется заглушка."
            return "cloud", provider, None
        engine, warning = self._resolve_llm_engine()
        return "local", engine, warning

    def _parse_cards_from_response(self, response_text: str, context_text: str = "") -> list[DraftCard]:
        payload = extract_json_strict(response_text)
        cards_payload = normalize_cards_payload(payload.get("cards") if isinstance(payload, dict) else [])
        cards: list[DraftCard] = []
        for item in cards_payload:
            normalized = self.normalize_card_fields(context_text, item, context_text)
            front, back = ensure_question(str(normalized.get("front") or ""), str(normalized.get("back") or ""), context_text)
            image_prompt = str(normalized.get("image_prompt") or "").strip()
            cards.append(
                DraftCard(
                    front=front,
                    back=back,
                    image_prompt=image_prompt,
                    tags=normalized.get("tags") or [],
                    media={"source": "llm"},
                    meta={"ts": int(time.time()), "image_prompt": image_prompt},
                    created_by_ai=True,
                )
            )
        return cards

    def _call_ollama_json(self, ollama: OllamaClient, user_content: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_JSON_ONLY},
            {"role": "user", "content": user_content},
        ]
        obj = call_ollama_json_strict_with_repair(ollama, messages)
        normalized_cards = normalize_cards_payload(obj.get("cards") if isinstance(obj, dict) else [])
        if not normalized_cards:
            raise LLMJsonError("Не удалось получить валидный JSON от Ollama")
        return {"cards": normalized_cards}

    def _token_similarity(self, left: str, right: str) -> float:
        left_tokens = {t for t in re.findall(r"[\wа-яА-ЯёЁ]+", (left or "").lower()) if len(t) > 2}
        right_tokens = {t for t in re.findall(r"[\wа-яА-ЯёЁ]+", (right or "").lower()) if len(t) > 2}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens))

    def make_question_from_back(self, back: str, context_text: str) -> str:
        source = (back or context_text or "").strip()
        year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", source)
        if year_match:
            return "В каком году это произошло?"
        term_match = re.search(r"([А-ЯA-Z][\w\-]{2,})\s+—\s+", source)
        if term_match:
            return f"Что такое {term_match.group(1)}?"
        words = [w for w in re.findall(r"[\wа-яА-ЯёЁ]+", source) if len(w) > 3][:4]
        if words:
            return f"В чём смысл: {' '.join(words)}?"
        return "Что это означает?"

    def shorten_to_question(self, front: str, back: str) -> str:
        words = re.findall(r"[\wа-яА-ЯёЁ]+", (front or ""))
        if 6 <= len(words) <= 10:
            return f"{' '.join(words)}?"
        return self.make_question_from_back(back, front)

    def _extract_1_3_sentences(self, text: str) -> str:
        raw = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        parts = [p.strip() for p in raw if p.strip()]
        return " ".join(parts[:3]) if parts else "Краткое объяснение отсутствует."

    def generate_image_prompt_only(self, front: str, back: str, context: str = "") -> str:
        if self.llm_provider_var.get() != self.LLM_PROVIDER_OLLAMA:
            base = f"{front} {back}".strip()
            return f"{base[:120]}, educational illustration, clean background".strip(", ")
        ollama = OllamaClient(
            base_url=self.ollama_url_var.get().strip() or self.OLLAMA_URL_DEFAULT,
            model=self.ollama_model_var.get().strip() or self.OLLAMA_MODEL_DEFAULT,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Верни только JSON: {\"image_prompt\":\"...\"}. "
                    "Сформируй конкретный английский промпт для Stable Diffusion: объекты, сцена, окружение, "
                    "educational illustration, clean background. Запрещены random/something/concept/idea."
                ),
            },
            {
                "role": "user",
                "content": f"front: {front}\nback: {back}\ncontext: {context}",
            },
        ]
        try:
            obj = call_ollama_json_strict_with_repair(ollama, messages, mode="image_prompt_only")
            prompt = str((obj or {}).get("image_prompt") or "").strip()
            if len(prompt.split()) >= 3 and not is_generic_image_prompt(prompt):
                return prompt
        except Exception:
            pass
        base = re.sub(r"\s+", " ", f"{front} {back}").strip()
        return f"{base[:120]}, educational illustration, clean background"

    def normalize_card_fields(self, user_prompt: str, card: dict[str, Any], context_text: str) -> dict[str, Any]:
        front = str(card.get("front") or "").strip()
        back = str(card.get("back") or "").strip()
        image_prompt = str(card.get("image_prompt") or "").strip()
        tags = card.get("tags") if isinstance(card.get("tags"), list) else []

        if len(front) > 120:
            front = self.shorten_to_question(front, back)
        if self._token_similarity(front, user_prompt) > 0.75:
            front = self.make_question_from_back(back, context_text)
        if not front:
            front = self.make_question_from_back(back, context_text)
        if not front.endswith("?"):
            front = front.rstrip(".! ") + "?"

        if not back:
            back = self._extract_1_3_sentences(context_text or user_prompt)
        sentences = [p for p in re.split(r"(?<=[.!?])\s+", back) if p.strip()]
        if len(sentences) > 3:
            back = " ".join(sentences[:3]).strip()

        if not image_prompt or is_generic_image_prompt(image_prompt):
            image_prompt = self.generate_image_prompt_only(front, back, context_text)

        return {
            "front": front[:120],
            "back": back,
            "image_prompt": image_prompt,
            "tags": [str(t).strip() for t in tags if str(t).strip()],
        }

    def derive_fallback_image_prompt(self, front: str, back: str) -> str:
        text = f"{front} {back}".lower()
        words = [w for w in re.findall(r"[a-zа-яё0-9\-]+", text) if len(w) > 3]
        if words:
            return f"{', '.join(words[:5])}, educational illustration"
        return "illustration of the concept described"

    def build_sd_prompt(self, card: DraftCard) -> str:
        base = (getattr(card, "image_prompt", "") or "").strip()
        style = "educational illustration, clean background, high detail, sharp focus"
        if not base:
            base = self.derive_fallback_image_prompt(card.front, card.back)
        return f"{base}. {style}".strip()

    def _on_chat_success(self, response_text: str, draft: DraftBatch | None, user_message: str) -> None:
        self._remove_temporary_message()
        is_card_command, card_prompt = self._extract_card_prompt(user_message)
        if not is_card_command:
            if response_text:
                self._append_message("assistant", response_text)
            else:
                self._append_message("assistant", "...")
        if is_card_command:
            parsed = _parse_card_from_llm(response_text)
            if parsed:
                front, back = parsed
                self._ensure_draft_batch_exists()
                normalized = self.normalize_card_fields(user_message, {"front": front, "back": back, "image_prompt": "", "tags": []}, user_message)
                front, back = ensure_question(normalized["front"], normalized["back"], user_message)
                prompt = (card_prompt or user_message).strip()
                image_prompt = normalized.get("image_prompt", "") or self.generate_image_prompt_only(front, back, prompt)
                card = DraftCard(
                    front=front,
                    back=back,
                    image_path=None,
                    image_prompt=image_prompt,
                    tags=[str(tag) for tag in (normalized.get("tags") or []) if str(tag).strip()],
                    media={"source": "chatbot_generated", "image_path": None},
                    meta={
                        "id": str(uuid.uuid4()),
                        "mode": "chatbot_generated",
                        "prompt": prompt,
                        "title": front[:120],
                        "current_side": "front",
                        "created_at": int(time.time()),
                        "source": "chatbot_generated",
                        "ts": int(time.time()),
                        "image_prompt": image_prompt,
                    },
                    created_by_ai=True,
                )
                self.current_draft_batch.cards.append(card)
                self.recalc_draft_cost()
                self.draft_index = len(self.current_draft_batch.cards) - 1
                self._render_side = "front"
                self.refresh_render()
                self.update_save_button_price()
                self._append_message("assistant", "Карточка сгенерирована и добавлена в предпросмотр.")
                if self.sd_enabled.get() and image_prompt:
                    self._append_temporary_message("Генерация изображения...", role="assistant")
                    self._generate_image_for_card(card, charge_now=False)
            else:
                self._append_message(
                    "system",
                    "Не удалось распарсить карточку из ответа. Ответ должен содержать JSON {front, back}.",
                )

        if draft and not is_card_command:
            self._on_generation_success(draft)
        else:
            self._set_sending_state(False)
        self._update_llm_status_label()

    def _on_chat_error(self, message: str) -> None:
        self._remove_temporary_message()
        self._append_message("assistant", f"[ERROR] {message}")
        self._set_sending_state(False)
        self._update_llm_status_label()

    def _on_chat_notice(self, message: str) -> None:
        self._remove_temporary_message()
        self._append_message("system", message)
        self._set_sending_state(False)
        self._update_llm_status_label()

    def _on_attach(self) -> None:
        filetypes = [
            ("Images", "*.png *.jpg *.jpeg *.webp"),
            ("PDF", "*.pdf"),
            ("Text", "*.txt"),
            ("Documents", "*.docx"),
            ("Audio", "*.mp3 *.wav *.ogg"),
            ("Video", "*.mp4 *.mkv"),
        ]
        paths = filedialog.askopenfilenames(title="Выберите файлы", filetypes=filetypes)
        if not paths:
            return
        self._add_attachments(paths)

    def _add_attachments(self, paths: list[str]) -> None:
        current_paths = set(self.pending_attachments)
        for path in paths:
            if len(self.pending_attachments) >= self.max_attachment_count:
                messagebox.showwarning(
                    "Вложения",
                    f"Можно прикрепить не более {self.max_attachment_count} файлов за раз.",
                )
                break
            if not path or path in current_paths:
                continue
            if not os.path.exists(path):
                continue
            if not self._is_allowed_attachment(path):
                messagebox.showwarning("Вложения", f"Формат не поддерживается: {os.path.basename(path)}")
                continue
            size = os.path.getsize(path)
            if size > self.max_attachment_size:
                messagebox.showwarning(
                    "Вложения",
                    f"Файл больше 25MB: {os.path.basename(path)}",
                )
                continue
            self.pending_attachments.append(path)
            current_paths.add(path)
            if path.lower().endswith('.pdf'):
                self._append_message('system', 'PDF загружен, анализирую...')
                self._start_pdf_ingest(path)
        self._refresh_attachments_ui()

    def _remove_attachment(self, index: int) -> None:
        if 0 <= index < len(self.pending_attachments):
            self.pending_attachments.pop(index)
        self._refresh_attachments_ui()

    def _refresh_attachments_ui(self) -> None:
        self.attachments_bar.render(self.pending_attachments)
        self._update_send_button_state()

    def _set_sending_state(self, sending: bool) -> None:
        if sending:
            self.input_text.configure(state=tk.DISABLED)
            self.attach_btn.configure(state=tk.DISABLED)
            self.send_btn_var.set("Думаю...")
            self.send_btn.configure(state=tk.DISABLED)
            return
        if self.deck_var.get():
            self.input_text.configure(state=tk.NORMAL)
            self.attach_btn.configure(state=tk.NORMAL)
            self.send_btn_var.set("Отправить")
            self._update_send_button_state()

    def _extract_sd_prompt(self, text: str) -> tuple[bool, bool, str]:
        normalized = (text or "").strip()
        if not normalized:
            return False, False, ""
        slash_match = re.match(r"^/img\s*(.*)$", normalized, flags=re.IGNORECASE)
        if slash_match:
            return True, False, slash_match.group(1).strip(" :,-")

        image_trigger = r"^\s*(?:нарисуй\s+картинку|сгенерируй\s+картинку)\b"
        if re.match(image_trigger, normalized, flags=re.IGNORECASE):
            prompt = _extract_trigger_prompt(normalized, image_trigger)
            return True, False, prompt
        return False, False, ""

    def _extract_card_prompt(self, text: str) -> tuple[bool, str]:
        normalized = (text or "").strip()
        if not normalized:
            return False, ""
        card_trigger = r"^\s*сгенерируй\s+карточку\b"
        if re.match(card_trigger, normalized, flags=re.IGNORECASE):
            return True, _extract_trigger_prompt(normalized, card_trigger)
        return False, ""

    def _resolve_sd_prompt_fallback(self, base_prompt: str) -> str:
        prompt = (base_prompt or "").strip()
        if len(prompt) >= 6:
            return prompt
        if self.current_draft_batch and self.current_draft_batch.cards:
            current = self.current_draft_batch.cards[self.draft_index]
            fallback = (current.front or current.back or "").strip()
            if fallback:
                return fallback
        conn = open_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT text FROM chat_messages WHERE session_id = ? AND role = 'user' ORDER BY ts DESC LIMIT 10;",
            (self.current_session_id,),
        )
        rows = cur.fetchall()
        conn.close()
        for row in rows:
            candidate = str(row["text"] or "").strip()
            if len(candidate) >= 6:
                return candidate
        return prompt or "image concept"

    def _on_send(self) -> None:
        if self.current_session_id is None:
            messagebox.showinfo("Выбор чата", "Выберите чат слева.")
            return
        if not self.deck_var.get():
            self._lock_chat()
            return
        text = self.input_text.get("1.0", "end-1c").strip()
        if not text and not self.pending_attachments:
            return
        attachments = list(self.pending_attachments)
        self.pending_attachments = []
        self._refresh_attachments_ui()
        self.input_text.delete("1.0", tk.END)
        self._resize_input_text()
        self._append_message("user", text, attachments)

        is_sd_command, _, parsed_prompt = self._extract_sd_prompt(text)
        if is_sd_command:
            prompt = (parsed_prompt or "").strip(" :,-")
            if not prompt:
                self._append_message("system", "Укажите описание после команды «сгенерируй картинку».")
                return
            cost = 2 if self.app.has_pro() else 4
            if not self.app.try_spend_credits(cost, "SD image generation"):
                self._append_message("system", "Недостаточно кредитов")
                return
            card = self._create_draft_from_text(prompt)
            card.created_by_ai = True
            card.meta.update({"mode": "chatbot_generated", "prompt": prompt, "source": "chatbot"})
            card.image_prompt = prompt
            self._render_side = "front"
            self.refresh_render()
            self._append_temporary_message("Генерация изображения...", role="assistant")
            self._generate_image_for_card(card, charge_now=False)
            return

        is_card_command, card_prompt = self._extract_card_prompt(text)
        if is_card_command:
            cleaned_prompt = card_prompt.strip()
            if not cleaned_prompt:
                self._append_message("system", "Укажите тему после команды «сгенерируй карточку».")
                return
            self._set_sending_state(True)
            self._append_temporary_message("Генерация карточки...", role="assistant")
            self._start_generation(text, attachments)
            return

        self._set_sending_state(True)
        self._append_temporary_message("Думаю...", role="assistant")
        self._start_generation(text, attachments)

    def _create_draft_from_text(self, text: str) -> DraftCard:
        deck_id = self._deck_map.get(self.deck_var.get()) or 0
        if not self.current_draft_batch or self.current_draft_batch.deck_id != deck_id:
            self.current_draft_batch = DraftBatch(
                draft_id=str(uuid.uuid4()),
                deck_id=deck_id,
                cards=[],
                total_credits=0,
                created_at=int(time.time()),
            )
        card = DraftCard(
            front=text,
            back="",
            image_path=None,
            image_prompt="",
            image_style="educational illustration, clean background",
            tags=[],
            media={"source": "chatbot"},
            meta={"ts": int(time.time())},
            created_by_ai=False,
            image_paid=False,
        )
        self.current_draft_batch.cards.append(card)
        self.recalc_draft_cost()
        self.draft_index = len(self.current_draft_batch.cards) - 1
        self._render_side = "front"
        return card

    def _ensure_draft_batch_exists(self) -> None:
        deck_id = self._deck_map.get(self.deck_var.get()) or 0
        if self.current_draft_batch and self.current_draft_batch.deck_id == deck_id:
            return
        self.current_draft_batch = DraftBatch(
            draft_id=str(uuid.uuid4()),
            deck_id=deck_id,
            cards=[],
            total_credits=0,
            created_at=int(time.time()),
        )

    def _get_or_create_current_draft_card(self, prompt: str = "") -> DraftCard:
        self._ensure_draft_batch_exists()
        if self.current_draft_batch and self.current_draft_batch.cards:
            self.draft_index = max(0, min(self.draft_index, len(self.current_draft_batch.cards) - 1))
            return self.current_draft_batch.cards[self.draft_index]
        return self._create_draft_from_text(prompt or "SD image")

    def refresh_render(self) -> None:
        self.winfo_toplevel().after(0, self._render_current_draft)

    def _show_front(self) -> None:
        self._render_side = "front"
        if self.current_draft_batch and self.current_draft_batch.cards:
            self.current_draft_batch.cards[self.draft_index].meta["current_side"] = "front"
        self.refresh_render()

    def _show_back(self) -> None:
        self._render_side = "back"
        if self.current_draft_batch and self.current_draft_batch.cards:
            self.current_draft_batch.cards[self.draft_index].meta["current_side"] = "back"
        self.refresh_render()

    def _delete_current_draft(self) -> None:
        if not self.current_draft_batch or not self.current_draft_batch.cards:
            messagebox.showinfo("Черновик", "Черновик пуст")
            return
        self.current_draft_batch.cards.pop(self.draft_index)
        if not self.current_draft_batch.cards:
            self.current_draft_batch = None
            self.draft_index = 0
        else:
            self.draft_index = min(self.draft_index, len(self.current_draft_batch.cards) - 1)
        self._chat_pending_save_cost = 0
        self.pending_cost = 0
        self.refresh_render()
        self.update_save_button_price()

    def _create_ai_card_with_image_deferred_charge(self, prompt: str) -> None:
        card = self._create_draft_from_text(prompt)
        card.back = "Карточка создана через чат-команду"
        card.created_by_ai = True
        if not card.image_prompt:
            card.image_prompt = self.generate_image_prompt_only(card.front, card.back, prompt)
        self.refresh_render()
        self.update_save_button_price()
        self._generate_image_for_card(card, charge_now=False)

    def _generate_image_for_current_card(self, prompt: str | None = None, charge_now: bool = True) -> None:
        if not self.current_draft_batch or not self.current_draft_batch.cards:
            messagebox.showinfo("Черновик", "Сначала создайте карточку")
            return
        card = self.current_draft_batch.cards[self.draft_index]
        if prompt and prompt.strip() and not card.image_prompt:
            card.image_prompt = self.generate_image_prompt_only(card.front, card.back, prompt)
        self._generate_image_for_card(card, charge_now=charge_now)

    def _generate_image_for_card(self, card: DraftCard, charge_now: bool) -> None:
        if not self.sd_enabled.get():
            self._remove_temporary_message()
            return
        if charge_now:
            cost = 2 if self.app.has_pro() else 4
            if not self.app.try_spend_credits(cost, "SD image generation"):
                self._append_message("system", "Недостаточно кредитов")
                self._remove_temporary_message()
                return
        api_url = self.sd_api_url_var.get().strip() or self.SD_API_URL_DEFAULT
        checkpoint = self.sd_checkpoint_var.get().strip() or self.SD_CHECKPOINT_DEFAULT

        def worker() -> None:
            try:
                sd = SDAPIClient(api_url)
                sd.set_checkpoint(checkpoint)
                if not card.image_prompt or is_generic_image_prompt(card.image_prompt):
                    card.image_prompt = self.generate_image_prompt_only(card.front, card.back, card.back)
                prompt_for_sd = self.build_sd_prompt(card)
                png_bytes = sd.txt2img(
                    prompt=prompt_for_sd,
                    negative_prompt=self.SD_NEGATIVE_PROMPT_DEFAULT,
                    width=1024,
                    height=1024,
                    steps=28,
                    cfg=6.5,
                    sampler="DPM++ 2M Karras",
                    seed=-1,
                )
                img_path = sd.save_image_bytes(png_bytes)
                self.after(0, lambda: self._apply_sd_image(card, img_path))
            except SDAPIError as exc:
                self.after(0, lambda: self._on_chat_error(f"Ошибка генерации изображения: {exc}"))
            except Exception as exc:  # noqa: BLE001
                logging.exception("SD generation failed")
                self.after(0, lambda: self._on_chat_error(f"Ошибка генерации изображения: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_sd_image(self, card: DraftCard, img_path: str) -> None:
        card.image_path = img_path
        card.image_paid = True
        if not card.media:
            card.media = {}
        card.media["image_path"] = img_path
        if not card.meta:
            card.meta = {}
        card.meta["image_path"] = img_path
        card.meta["current_side"] = self._render_side
        self.refresh_render()
        self.update_save_button_price()
        self._remove_temporary_message()

    def _rebalance_chunks(self, sentences: list[str], mode_native: bool) -> list[list[str]]:
        target = self.native_sentences_var.get() if mode_native else self.foreign_sentences_var.get()
        target = max(5, min(20, target)) if mode_native else max(1, min(5, target))
        if not sentences:
            return []
        chunks: list[list[str]] = []
        cur: list[str] = []
        for sentence in sentences:
            cur.append(sentence)
            if len(cur) >= target:
                chunks.append(cur)
                cur = []
        if cur:
            if chunks and len(cur) < (5 if mode_native else 1):
                chunks[-1].extend(cur)
            else:
                chunks.append(cur)
        return chunks

    def _extract_cards_payload(self, raw: str) -> list[dict[str, Any]]:
        payload = _extract_json_object(raw)
        if not isinstance(payload, dict):
            return []
        cards = payload.get('cards')
        if not isinstance(cards, list):
            return []
        return [c for c in cards if isinstance(c, dict)]

    def _start_pdf_ingest(self, pdf_path: str) -> None:
        deck_id = self._deck_map.get(self.deck_var.get()) or 0

        def worker() -> None:
            try:
                text = extract_text_from_pdf(pdf_path)
                lang = detect_lang(text)
                mode_native = (not self.foreign_mode_var.get()) and lang == 'ru'
                sentences = split_to_sentences(text, lang)
                base_chunks = chunk_sentences(sentences, mode_native)
                flat_sentences = [s for chunk in base_chunks for s in chunk]
                chunks = self._rebalance_chunks(flat_sentences, mode_native)
                preview_text = text[:3500] if text else ' '
                ollama = OllamaClient(
                    base_url=self.ollama_url_var.get().strip() or self.OLLAMA_URL_DEFAULT,
                    model=self.ollama_model_var.get().strip() or 'llama3.1:8b',
                )
                summary = ollama.chat([
                    {'role': 'system', 'content': 'Ты аналитик. Выдели общий смысл, главные темы, 5-10 ключевых тезисов. Без воды.'},
                    {'role': 'user', 'content': preview_text},
                ])
                known_words = set()
                if not mode_native:
                    known_words = load_known_words(lang)
                cards: list[DraftCard] = []
                for chunk in chunks:
                    chunk_text = ' '.join(chunk)
                    user_prompt = (
                        f"summary:\n{summary}\n\n"
                        f"chunk_text:\n{chunk_text}\n\n"
                        "Правила:\n"
                        f"* {'native' if mode_native else 'foreign'} режим\n"
                        "* 1 карточка на chunk\n"
                        "* image_prompt короткий, по смыслу\n"
                        "* не выдумывать фактов вне chunk"
                    )
                    user_content = f"Сформируй 1 карточку по тексту: {chunk_text}. Язык: {lang}. Обязателен image_prompt. Контекст: {summary}."
                    obj = self._call_ollama_json(ollama, user_content)
                    for item in normalize_cards_payload(obj.get('cards', [])):
                        front, back = ensure_question(str(item.get('front') or ''), str(item.get('back') or ''), chunk_text)
                        if not mode_native:
                            back = mask_unknown_words(back, lang, known_words)
                        normalized = self.normalize_card_fields(chunk_text, item, summary)
                        image_prompt = str(normalized.get('image_prompt') or '').strip()
                        tags = normalized.get('tags') if isinstance(normalized.get('tags'), list) else []
                        cards.append(DraftCard(front=normalized.get('front') or front, back=normalized.get('back') or back, image_path=None, image_prompt=image_prompt, tags=[str(t) for t in tags], media={'source': 'pdf_ingest'}, meta={'image_prompt': image_prompt, 'lang': lang}, created_by_ai=True, image_paid=False))
                batch = DraftBatch(
                    draft_id=str(uuid.uuid4()),
                    deck_id=deck_id,
                    cards=cards,
                    total_credits=self._calculate_total_credits(cards),
                    created_at=int(time.time()),
                )
                def apply_batch() -> None:
                    self.current_draft_batch = batch
                    self.draft_index = 0
                    self._render_side = 'front'
                    self.refresh_render()
                    self._update_save_button_state()
                    self._append_message('system', f'PDF обработан: карточек {len(cards)}.')
                self.winfo_toplevel().after(0, apply_batch)
                self._start_pdf_images_generation(cards)
            except Exception as exc:
                logging.exception('PDF ingest failed')
                self.winfo_toplevel().after(0, lambda: self._append_message('system', f'Ошибка PDF ingest: {exc}'))

        threading.Thread(target=worker, daemon=True).start()

    def _start_pdf_images_generation(self, cards: list[DraftCard]) -> None:
        if not cards or not self.sd_enabled.get():
            return
        api_url = self.sd_api_url_var.get().strip() or self.SD_API_URL_DEFAULT
        checkpoint = self.sd_checkpoint_var.get().strip() or self.SD_CHECKPOINT_DEFAULT

        def worker() -> None:
            try:
                sd = SDAPIClient(api_url)
                sd.set_checkpoint(checkpoint)
                for card in cards:
                    if not card.image_prompt or is_generic_image_prompt(card.image_prompt):
                        card.image_prompt = self.generate_image_prompt_only(card.front, card.back, card.back)
                    prompt_for_sd = self.build_sd_prompt(card)
                    try:
                        png_bytes = sd.txt2img(
                            prompt=prompt_for_sd,
                            negative_prompt=self.SD_NEGATIVE_PROMPT_DEFAULT,
                            width=1024,
                            height=1024,
                            steps=24,
                            cfg=6.5,
                            sampler='DPM++ 2M Karras',
                            seed=-1,
                        )
                        img_path = sd.save_image_bytes(png_bytes)
                    except Exception:
                        continue
                    self.winfo_toplevel().after(0, lambda c=card, p=img_path: self._apply_sd_image(c, p))
            except Exception:
                logging.exception('PDF SD queue failed')

        threading.Thread(target=worker, daemon=True).start()

    def _start_generation(self, text: str, attachments: list[str]) -> None:
        plan = self.app.get_pricing_plan()
        user_id = self.app.user_id
        deck_context = {"deck_id": self._deck_map.get(self.deck_var.get())}
        backend_kind, backend, warning_message = self._resolve_chat_backend()
        messages = self._build_chat_messages(for_cloud=backend_kind == "cloud")
        if warning_message:
            self._append_message("system", warning_message)
        if backend_kind == "local" and backend is self.llama_engine and not self.llama_engine.is_loaded():
            self.llm_status_var.set("Llama: загрузка...")

        def worker():
            try:
                if backend_kind == "cloud":
                    cloud_response = backend.chat(
                        messages=messages,
                        chat_id=self.current_session_id,
                        model=self.CLOUD_MODEL_DEFAULT,
                        temperature=0.6,
                        max_tokens=768,
                    )
                    response_text = str(cloud_response.get("reply") or "")
                    credits_spent = cloud_response.get("credits_spent")
                    remaining_credits = cloud_response.get("remaining_credits")
                    self.after(
                        0,
                        lambda: self._sync_cloud_credits(
                            remaining_credits if remaining_credits is not None else None,
                            credits_spent=credits_spent,
                        ),
                    )
                    self.after(0, lambda: self.cloud_status_var.set("Cloud: OK"))
                else:
                    response_text = backend.chat(
                        messages,
                        temperature=0.6,
                        max_tokens=768,
                    )
                    if backend is self.ollama_engine:
                        self.after(0, lambda: self.ollama_status_var.set("Ollama: OK"))
                cards = self._parse_cards_from_response(response_text, text)
                draft = None
                if cards:
                    self.card_engine.check_and_record_generation(user_id, plan, len(cards))
                    total_credits = self.card_engine.estimate_cost(len(cards), plan)
                    draft = DraftBatch(
                        draft_id=str(uuid.uuid4()),
                        deck_id=deck_context.get("deck_id") or 0,
                        cards=cards,
                        total_credits=total_credits,
                        created_at=int(time.time()),
                    )
                self.after(0, lambda: self._on_chat_success(response_text, draft, text))
            except CloudProviderError as exc:
                logging.exception("Cloud LLM generation failed")
                self.after(0, lambda: self._handle_cloud_error(exc))
            except OllamaUnavailableError:
                self.after(
                    0,
                    lambda: self._on_chat_notice(
                        "[LLM OFFLINE] Запусти Ollama и проверь модель xflash-llama31"
                    ),
                )
                self.after(0, lambda: self.ollama_status_var.set("Ollama: offline"))
            except OllamaModelNotFoundError:
                self.after(
                    0,
                    lambda: self._on_chat_notice(
                        "[MODEL NOT FOUND] Выполни: ollama create xflash-llama31 -f Modelfile"
                    ),
                )
                self.after(0, lambda: self.ollama_status_var.set("Ollama: offline"))
            except Exception as exc:  # noqa: BLE001
                logging.exception("LLM generation failed")
                self.after(0, lambda: self._on_chat_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_cloud_error(self, exc: CloudProviderError) -> None:
        self._update_cloud_status_from_error(exc)
        message = exc.message
        if exc.status_code == 503:
            message = "Сервер ИИ оффлайн"
        elif exc.status_code == 429:
            message = "Сервер занят/лимит"
            if exc.retry_after:
                message = f"{message} (retry_after: {exc.retry_after})"
        elif exc.status_code == 402:
            message = "Недостаточно кредитов"
        elif exc.status_code == 401:
            message = "Неверный ключ"
        self._on_chat_error(message)

    def _on_generation_success(self, draft: DraftBatch) -> None:
        if self.current_draft_batch and self.current_draft_batch.cards and self.current_draft_batch.deck_id == draft.deck_id:
            self.current_draft_batch.cards.extend(draft.cards)
            self.recalc_draft_cost()
            self.draft_index = max(0, len(self.current_draft_batch.cards) - 1)
            self._render_side = "front"
            self._render_current_draft()
            total_cards = len(self.current_draft_batch.cards)
            self._append_message(
                "assistant",
                f"Добавлено {len(draft.cards)} карточек. Всего: {total_cards}.",
            )
        else:
            self.current_draft_batch = draft
            self.recalc_draft_cost()
            self.draft_index = 0
            self._render_side = "front"
            self._render_current_draft()
            self._append_message("assistant", f"Сформирован черновик на {len(draft.cards)} карточек.")
        self._update_save_button_state()
        self._set_sending_state(False)

    def recalc_draft_cost(self) -> int:
        if not self.current_draft_batch or not self.current_draft_batch.cards:
            self.pending_cost = 0
            return 0
        total = 0
        for card in self.current_draft_batch.cards:
            if getattr(card, "created_by_ai", False):
                total += 2 if self.app.has_pro() else 5
            if getattr(card, "image_paid", False):
                total += 2
        self.pending_cost = total
        self.current_draft_batch.total_credits = total
        return total

    def _calculate_total_credits(self, cards: list[DraftCard]) -> int:
        return self.recalc_draft_cost()

    def _on_generation_error(self, message: str) -> None:
        self._on_chat_error(message)

    def update_save_button_price(self) -> None:
        cost = self.recalc_draft_cost()
        if cost <= 0:
            label = "Сохранить"
            self.pending_cost_var.set("")
        else:
            label = f"Сохранить 🪙 {cost}"
            self.pending_cost_var.set(f"Итоговая стоимость: {cost}")
        enabled = bool(self.current_draft_batch and self.current_draft_batch.cards) and self.app.get_credits() >= cost
        self.save_draft_btn.configure(text=label, state=(tk.NORMAL if enabled else tk.DISABLED))

    def _update_save_button_state(self) -> None:
        self.update_save_button_price()

    def _set_save_state(self, enabled: bool, total_credits: int | None = None) -> None:
        label = "Сохранить"
        if total_credits and total_credits > 0:
            label = f"Сохранить 🪙 {total_credits}"
        self.save_draft_btn.configure(text=label, state=(tk.NORMAL if enabled else tk.DISABLED))

    def _draft_display_id(self, draft: DraftCard, card_payload: dict) -> str:
        meta = getattr(draft, "meta", {}) or {}
        raw_id = meta.get("temp_id") or meta.get("id") or card_payload.get("id")
        if raw_id in (None, -1):
            return "Draft"
        return str(raw_id)

    def _render_current_draft(self) -> None:
        total = len(self.current_draft_batch.cards) if self.current_draft_batch else 0
        if total == 0:
            placeholder = {
                "front": "Черновик пуст. Сначала сформируйте карточки.",
                "back": "",
                "front_rich": None,
                "back_rich": None,
            }
            self.card_view.load_card(
                placeholder,
                status_text="Карточка 0/0 | ID —",
                header_text="Фаза | след. повтор: —",
                show_back=False,
            )
            self.card_view.set_rating_enabled(False)
            self.draft_index_var.set("0/0")
            self.prev_draft_btn.configure(state=tk.DISABLED)
            self.next_draft_btn.configure(state=tk.DISABLED)
            return

        self.draft_index = max(0, min(self.draft_index, total - 1))
        draft = self.current_draft_batch.cards[self.draft_index]
        card_payload = draft_to_card(draft)
        display_id = self._draft_display_id(draft, card_payload)
        status_text = f"Карточка {self.draft_index + 1}/{total} | ID {display_id}"
        self.card_view.load_card(
            card_payload,
            status_text=status_text,
            header_text="Фаза | след. повтор: —",
            show_back=self._render_side == "back",
        )
        self.card_view.set_rating_enabled(False)

        self.draft_index_var.set(f"{self.draft_index + 1}/{total}")
        self.prev_draft_btn.configure(state=(tk.NORMAL if self.draft_index > 0 else tk.DISABLED))
        self.next_draft_btn.configure(state=(tk.NORMAL if self.draft_index < total - 1 else tk.DISABLED))

    def _prev_draft(self) -> None:
        if self.draft_index > 0:
            self.draft_index -= 1
            self._render_side = "front"
            self._render_current_draft()

    def _next_draft(self) -> None:
        if self.current_draft_batch and self.draft_index < len(self.current_draft_batch.cards) - 1:
            self.draft_index += 1
            self._render_side = "front"
            self._render_current_draft()

    def _save_draft(self) -> None:
        if not self.current_draft_batch:
            return
        cost = getattr(self, "pending_cost", 0) or self.recalc_draft_cost()
        deck_id = self._deck_map.get(self.deck_var.get()) or self.current_draft_batch.deck_id
        if not deck_id:
            messagebox.showwarning("Колода", "Выберите колоду для сохранения.")
            return
        if cost > 0 and not self.app.try_spend_credits(cost, "chatbot save draft"):
            self._update_save_button_state()
            return
        conn = open_db()
        try:
            saved_count = self._save_draft_to_deck(conn, deck_id, self.current_draft_batch.cards)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            messagebox.showerror("Ошибка", str(exc))
            return
        finally:
            conn.close()
        self.current_draft_batch = None
        self._chat_pending_save_cost = 0
        self.pending_cost = 0
        self.draft_index = 0
        self._render_side = "front"
        self.refresh_render()
        self._set_save_state(False, 0)
        self._append_message("system", f"Сохранено {saved_count} карточек.")

    def _save_draft_to_deck(self, conn, deck_id: int, cards: list[DraftCard]) -> int:
        saved = 0
        now_ts = int(time.time())
        for card in cards:
            image_path = card.image_path or (card.media or {}).get("image_path")
            fields = {
                "word": card.front,
                "translation": card.back,
                "notes": "",
                "example": "",
                "front": card.front,
                "back": card.back,
                "image": image_path,
                "front_image_path": image_path,
            }
            srs_defaults = {
                "state": "new",
                "due": now_ts,
                "interval": 0,
                "ease": 250,
                "reps": 0,
                "lapses": 0,
                "step_index": 0,
                "phase": 1,
            }
            mode = {
                "skip_existing": False,
                "reset_srs": False,
                "state": "new",
                "source": "chatbot_generated",
            }
            tags_value = " ".join(card.tags)
            upsert_note_and_cards(conn, deck_id, None, fields, tags_value, srs_defaults, mode)
            saved += 1
        return saved
