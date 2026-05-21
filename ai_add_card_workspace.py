from __future__ import annotations

import json
import logging
import os
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ai_card_pipeline import AICardPipeline
from card_preview_widget import CardPreviewWidget
from rag_content_pipeline import RagContentPipeline, normalize_card_image_fields
from rag_web_search import RagWebSearch

try:
    from chat_bot_tab import ChatBotTab
except Exception:
    ChatBotTab = None

class BlackOutlineScrollbar(tk.Canvas):
    """Canvas-based vertical scrollbar with a black thumb and white outline.

    This version maps the thumb position to Canvas.yview_moveto correctly in
    both directions. It also supports page jumps when the user clicks above or
    below the thumb.
    """

    def __init__(self, master, command=None, width: int = 18, **kwargs) -> None:
        super().__init__(
            master,
            width=width,
            highlightthickness=1,
            highlightbackground="#ffffff",
            highlightcolor="#ffffff",
            bd=0,
            bg="#050505",
            cursor="hand2",
            **kwargs,
        )
        self.command = command
        self._first = 0.0
        self._last = 1.0
        self._drag_offset = 0
        self._is_dragging = False
        self._thumb_id = self.create_rectangle(
            3,
            3,
            width - 4,
            48,
            fill="#050505",
            outline="#ffffff",
            width=1,
        )
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", self._on_wheel)
        self.bind("<Button-5>", self._on_wheel)

    def set(self, first, last) -> None:
        try:
            self._first = max(0.0, min(1.0, float(first)))
            self._last = max(self._first, min(1.0, float(last)))
        except Exception:
            self._first, self._last = 0.0, 1.0
        self._redraw()

    def _visible_fraction(self) -> float:
        return max(0.0, min(1.0, self._last - self._first))

    def _max_first(self) -> float:
        return max(0.0, 1.0 - self._visible_fraction())

    def _track_height(self) -> int:
        return max(1, int(self.winfo_height()) - 8)

    def _thumb_bounds(self) -> tuple[int, int]:
        track_h = self._track_height()
        visible = self._visible_fraction()
        if visible >= 0.999:
            return 4, max(40, int(self.winfo_height()) - 4)
        thumb_h = max(42, int(track_h * max(0.05, visible)))
        thumb_h = min(track_h, thumb_h)
        usable = max(1, track_h - thumb_h)
        ratio = self._first / max(0.0001, self._max_first())
        y1 = 4 + int(usable * ratio)
        y1 = max(4, min(4 + usable, y1))
        return y1, y1 + thumb_h

    def _redraw(self) -> None:
        w = max(10, int(self.winfo_width()))
        h = max(10, int(self.winfo_height()))
        self.configure(bg="#050505")
        y1, y2 = self._thumb_bounds()
        self.coords(self._thumb_id, 3, y1, w - 4, min(h - 4, y2))
        self.itemconfigure(self._thumb_id, fill="#050505", outline="#ffffff", width=1)

    def _moveto_from_y(self, y: int) -> None:
        if not self.command:
            return
        y1, y2 = self._thumb_bounds()
        thumb_h = max(1, y2 - y1)
        track_h = self._track_height()
        usable = max(1, track_h - thumb_h)
        ratio = max(0.0, min(1.0, (y - 4 - self._drag_offset) / usable))
        target_first = ratio * self._max_first()
        self.command("moveto", target_first)

    def _on_click(self, event) -> str:
        y1, y2 = self._thumb_bounds()
        if y1 <= event.y <= y2:
            self._is_dragging = True
            self._drag_offset = event.y - y1
            return "break"
        # Click on the track: page up/down. This makes it easy to move back.
        if self.command:
            self.command("scroll", -1 if event.y < y1 else 1, "pages")
        return "break"

    def _on_drag(self, event) -> str:
        if self._is_dragging:
            self._moveto_from_y(event.y)
        return "break"

    def _on_release(self, _event) -> str:
        self._is_dragging = False
        self._drag_offset = 0
        return "break"

    def _on_wheel(self, event) -> str:
        if not self.command:
            return "break"
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.command("scroll", -4, "units")
        else:
            self.command("scroll", 4, "units")
        return "break"


class AIAddCardWorkspace(tk.Toplevel):
    def __init__(self, app: tk.Misc) -> None:
        super().__init__(app)
        self.app = app
        self.title("AI добавление карточек")
        self.geometry("1120x820")
        self.minsize(980, 720)
        self.transient(app)

        self.deck_id = getattr(app, "selected_deck_id", None)
        self.pipeline = AICardPipeline(app=app, deck_id=self.deck_id)
        self.web_search = RagWebSearch()
        self.rag_pipeline = RagContentPipeline(max_chars=30000, max_sources=5)

        self.source_path: str | None = None
        self.generated_cards: list[dict] = []
        self.current_card_index = 0
        self.auto_generate_image_after_card = False
        self.generate_images_for_all_var = tk.BooleanVar(value=False)
        self._busy = False
        self._last_chat_answer = ""
        self._max_rag_chars = 30000
        self._text_context_menu = None
        self._last_source_trace: dict = {}
        self._llm_settings_path = self._resolve_chatbot_settings_path()
        self._cached_llm_settings = self._read_shared_llm_settings()
        self.llm_summary_var = tk.StringVar(value="")
        self.llm_runtime_status_var = tk.StringVar(value="LLM status: unknown")
        self.image_runtime_status_var = tk.StringVar(value="Image status: unknown")

        root = self._create_scrollable_root()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        workspace_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(workspace_tab, text="AI Workspace")
        self._build_workspace_tab(workspace_tab)
        self._build_advanced_chat_tab()
        self.bind("<FocusIn>", self._sync_deck_selector_from_app, add="+")
        self._update_llm_summary()
        self._refresh_llm_runtime_status()


    def _create_scrollable_root(self) -> ttk.Frame:
        shell = ttk.Frame(self)
        shell.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(
            shell,
            bg="#0b0f19",
            highlightthickness=0,
            bd=0,
            yscrollincrement=24,
        )
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._workspace_scrollbar = BlackOutlineScrollbar(
            shell,
            command=self._scroll_canvas.yview,
            width=18,
        )
        self._workspace_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 6), pady=6)
        self._scroll_canvas.configure(yscrollcommand=self._workspace_scrollbar.set)

        root = ttk.Frame(self._scroll_canvas, padding=10)
        self._scroll_window = self._scroll_canvas.create_window((0, 0), window=root, anchor="nw")

        root.bind("<Configure>", self._update_scroll_region)
        self._scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)

        # Bind the wheel to the whole AI window while the cursor is inside it.
        # This fixes the case when wheel scrolling down works, but scrolling back
        # up is swallowed by inner Text/Listbox widgets.
        self._mousewheel_active = False
        shell.bind("<Enter>", self._activate_workspace_mousewheel, add="+")
        shell.bind("<Leave>", self._deactivate_workspace_mousewheel, add="+")
        self.bind("<Prior>", lambda _e: (self._scroll_canvas.yview_scroll(-1, "pages"), "break"))
        self.bind("<Next>", lambda _e: (self._scroll_canvas.yview_scroll(1, "pages"), "break"))
        self.bind("<Home>", lambda _e: (self._scroll_canvas.yview_moveto(0), "break"))
        self.bind("<End>", lambda _e: (self._scroll_canvas.yview_moveto(1), "break"))
        return root

    def _update_scroll_region(self, _event=None) -> None:
        try:
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
            if hasattr(self, "_workspace_scrollbar"):
                self._workspace_scrollbar.set(*self._scroll_canvas.yview())
        except Exception:
            logging.exception("AI workspace scrollregion update failed")

    def _on_scroll_canvas_configure(self, event) -> None:
        try:
            self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)
            self._update_scroll_region()
        except Exception:
            logging.exception("AI workspace canvas resize failed")

    def _activate_workspace_mousewheel(self, _event=None) -> None:
        if getattr(self, "_mousewheel_active", False):
            return
        self._mousewheel_active = True
        self.bind_all("<MouseWheel>", self._on_workspace_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_workspace_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_workspace_mousewheel, add="+")

    def _deactivate_workspace_mousewheel(self, _event=None) -> None:
        if not getattr(self, "_mousewheel_active", False):
            return
        self._mousewheel_active = False
        try:
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")
        except Exception:
            pass

    def _on_workspace_mousewheel(self, event) -> str:
        try:
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                units = -4
            else:
                units = 4
            self._scroll_canvas.yview_scroll(units, "units")
            if hasattr(self, "_workspace_scrollbar"):
                self._workspace_scrollbar.set(*self._scroll_canvas.yview())
        except Exception:
            logging.exception("AI workspace mousewheel scroll failed")
        return "break"

    def _deck_row_value(self, row, key: str, index: int = 0, default=None):
        try:
            if isinstance(row, dict):
                return row.get(key, default)
            return row[key]
        except Exception:
            try:
                return row[index]
            except Exception:
                return default

    def _load_deck_options(self) -> list[tuple[int, str]]:
        """Return [(deck_id, deck_name), ...] from the main window state.

        The workspace intentionally does not import main.py to avoid circular
        imports. It mirrors the already loaded deck list from app.decks.
        """
        options: list[tuple[int, str]] = []
        for row in getattr(self.app, "decks", []) or []:
            deck_id = self._deck_row_value(row, "id", 0)
            name = self._deck_row_value(row, "name", 1, "")
            try:
                deck_id = int(deck_id)
            except Exception:
                continue
            name = str(name or f"Колода {deck_id}").strip() or f"Колода {deck_id}"
            options.append((deck_id, name))
        return options

    def _deck_label(self, deck_id: int, name: str) -> str:
        return f"{deck_id}: {name}"

    def _deck_id_from_label(self, label: str) -> int | None:
        try:
            value = (label or "").split(":", 1)[0].strip()
            return int(value) if value else None
        except Exception:
            return None

    def _build_deck_selector(self, parent: ttk.Frame) -> None:
        deck_wrap = ttk.LabelFrame(parent, text="Колода для сохранения", padding=8)
        deck_wrap.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        deck_wrap.columnconfigure(1, weight=1)

        ttk.Label(deck_wrap, text="Колода:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.deck_var = tk.StringVar()
        self.deck_combo = ttk.Combobox(deck_wrap, textvariable=self.deck_var, state="readonly")
        self.deck_combo.grid(row=0, column=1, sticky="ew")
        self.deck_combo.bind("<<ComboboxSelected>>", self._on_deck_selected)
        ttk.Button(deck_wrap, text="↻", width=3, command=self.refresh_deck_selector).grid(row=0, column=2, padx=(6, 0))

        self.deck_status_var = tk.StringVar(value="")
        ttk.Label(deck_wrap, textvariable=self.deck_status_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.refresh_deck_selector(sync_from_app=True)

    def refresh_deck_selector(self, sync_from_app: bool = True) -> None:
        self.deck_options = self._load_deck_options()
        labels = [self._deck_label(deck_id, name) for deck_id, name in self.deck_options]
        if hasattr(self, "deck_combo"):
            self.deck_combo.configure(values=labels)

        target_id = getattr(self.app, "selected_deck_id", None) if sync_from_app else self.deck_id
        if target_id is None and self.deck_id is not None:
            target_id = self.deck_id

        selected_label = ""
        for deck_id, name in self.deck_options:
            if int(deck_id) == target_id:
                selected_label = self._deck_label(deck_id, name)
                break

        if not selected_label and self.deck_options:
            deck_id, name = self.deck_options[0]
            selected_label = self._deck_label(deck_id, name)
            if target_id is None:
                target_id = deck_id

        if hasattr(self, "deck_var"):
            self.deck_var.set(selected_label)
        self._set_active_deck(target_id, update_main=False)

    def _set_active_deck(self, deck_id: int | None, update_main: bool = True) -> None:
        self.deck_id = deck_id
        try:
            self.pipeline.deck_id = deck_id
        except Exception:
            pass

        if update_main:
            try:
                self.app.selected_deck_id = deck_id
                if hasattr(self.app, "selected_phase"):
                    self.app.selected_phase = None
                self._select_deck_in_main_tree(deck_id)
                if hasattr(self.app, "load_templates_for_selected_deck"):
                    self.app.load_templates_for_selected_deck()
                if hasattr(self.app, "update_deck_preview"):
                    self.app.update_deck_preview()
                if hasattr(self.app, "update_overdue_badge"):
                    self.app.update_overdue_badge()
            except Exception:
                logging.exception("AI workspace deck sync with main window failed")

        if hasattr(self, "deck_status_var"):
            if deck_id is None:
                self.deck_status_var.set("Колода не выбрана. Карточки нельзя будет сохранить в ознакомление.")
            else:
                name = next((name for did, name in getattr(self, "deck_options", []) if did == deck_id), "")
                self.deck_status_var.set(f"Активная колода: {name or deck_id}")

    def _on_deck_selected(self, _event=None) -> None:
        deck_id = self._deck_id_from_label(self.deck_var.get() if hasattr(self, "deck_var") else "")
        self._set_active_deck(deck_id, update_main=True)
        self.status_var.set(f"Выбрана колода: {deck_id if deck_id is not None else '—'}")

    def _sync_deck_selector_from_app(self, _event=None) -> None:
        app_deck_id = getattr(self.app, "selected_deck_id", None)
        if app_deck_id == self.deck_id:
            return
        self.refresh_deck_selector(sync_from_app=True)

    def _select_deck_in_main_tree(self, deck_id: int | None) -> None:
        tree = getattr(self.app, "decks_tree", None)
        deck_items = getattr(self.app, "deck_items", {}) or {}
        if tree is None or deck_id is None:
            return
        try:
            for item_id, pair in deck_items.items():
                try:
                    item_deck_id, phase = pair
                except Exception:
                    continue
                if item_deck_id == deck_id and phase is None:
                    tree.selection_set(item_id)
                    tree.see(item_id)
                    break
        except Exception:
            logging.exception("AI workspace main deck tree selection failed")

    def _get_selected_deck_id(self) -> int | None:
        deck_id = self._deck_id_from_label(self.deck_var.get() if hasattr(self, "deck_var") else "")
        if deck_id is None:
            deck_id = self.deck_id or getattr(self.app, "selected_deck_id", None)
        self._set_active_deck(deck_id, update_main=False)
        return deck_id

    def _build_workspace_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(5, weight=1)

        preview_wrap = ttk.LabelFrame(parent, text="Предпросмотр карточки", padding=8)
        preview_wrap.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        preview_wrap.rowconfigure(0, weight=1)
        preview_wrap.columnconfigure(0, weight=1)
        preview_wrap.configure(height=240)
        preview_wrap.grid_propagate(False)

        self.preview = CardPreviewWidget(preview_wrap)
        self.preview.grid(row=0, column=0, sticky="nsew")

        self._build_deck_selector(parent)

        source_wrap = ttk.LabelFrame(parent, text="Источник / промт", padding=8)
        source_wrap.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        source_wrap.columnconfigure(0, weight=1)
        source_wrap.rowconfigure(0, weight=1)

        self.prompt_text = tk.Text(
            source_wrap,
            height=7,
            wrap=tk.WORD,
            bg="#0f1420",
            fg="#ffffff",
            insertbackground="#ffffff",
            selectbackground="#2b5f9e",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#6b7280",
            highlightcolor="#ffffff",
        )
        self.prompt_text.grid(row=0, column=0, sticky="nsew")
        self.prompt_scrollbar = BlackOutlineScrollbar(
            source_wrap,
            command=self.prompt_text.yview,
            width=16,
        )
        self.prompt_scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        self.prompt_text.configure(yscrollcommand=self.prompt_scrollbar.set)
        self.prompt_text.bind("<MouseWheel>", self._on_prompt_text_mousewheel, add="+")
        self.prompt_text.bind("<Button-4>", self._on_prompt_text_mousewheel, add="+")
        self.prompt_text.bind("<Button-5>", self._on_prompt_text_mousewheel, add="+")
        self._install_text_edit_bindings(self.prompt_text)

        actions_wrap = ttk.LabelFrame(parent, text="Действия", padding=8)
        actions_wrap.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for col in range(5):
            actions_wrap.columnconfigure(col, weight=1)

        ttk.Button(actions_wrap, text="Загрузить файл", command=self._pick_file).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Извлечь текст", command=self._extract_text).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Сгенерировать карточки", command=self.generate_cards_from_input).grid(row=0, column=2, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="🔍 Найти материалы", command=self._search_web).grid(row=0, column=3, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="⚙ AI настройки", command=self._open_llm_settings).grid(row=0, column=4, sticky="ew", padx=3, pady=3)

        ttk.Button(actions_wrap, text="Сгенерировать картинку", command=self.generate_image_for_current_card).grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Сгенерировать картинки для всех", command=self.generate_images_for_all_cards).grid(row=1, column=4, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Сохранить в ознакомление", command=self.save_current_card_to_overview).grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Ручной редактор", command=self.open_manual_editor).grid(row=1, column=2, sticky="ew", padx=3, pady=3)
        ttk.Button(actions_wrap, text="Очистить", command=self.clear_workspace).grid(row=1, column=3, sticky="ew", padx=3, pady=3)
        ttk.Checkbutton(
            actions_wrap,
            text="Сгенерировать картинки для всех карточек",
            variable=self.generate_images_for_all_var,
            onvalue=True,
            offvalue=False,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=3, pady=(3, 0))
        ttk.Button(actions_wrap, text="Отмена", command=self.destroy).grid(row=2, column=4, sticky="ew", padx=3, pady=3)

        status_row = ttk.Frame(parent)
        status_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        status_row.columnconfigure(1, weight=1)
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(status_row, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(status_row, mode="indeterminate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(status_row, textvariable=self.llm_summary_var).grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(status_row, textvariable=self.llm_runtime_status_var).grid(row=1, column=1, sticky="e", pady=(3, 0))
        ttk.Label(status_row, textvariable=self.image_runtime_status_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        cards_wrap = ttk.LabelFrame(parent, text="Сгенерированные карточки", padding=8)
        cards_wrap.grid(row=5, column=0, sticky="nsew", padx=(0, 6))
        cards_wrap.columnconfigure(0, weight=1)
        cards_wrap.rowconfigure(2, weight=1)

        nav_row = ttk.Frame(cards_wrap)
        nav_row.grid(row=0, column=0, sticky="ew")
        nav_row.columnconfigure(1, weight=1)
        ttk.Button(nav_row, text="←", width=4, command=self.prev_card).grid(row=0, column=0, padx=(0, 6))
        self.cards_counter_var = tk.StringVar(value="0/0")
        ttk.Label(nav_row, textvariable=self.cards_counter_var).grid(row=0, column=1, sticky="w")
        ttk.Button(nav_row, text="→", width=4, command=self.next_card).grid(row=0, column=2, padx=(6, 0))

        cards_list_row = ttk.Frame(cards_wrap)
        cards_list_row.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        cards_list_row.columnconfigure(0, weight=1)

        self.cards_listbox = tk.Listbox(
            cards_list_row,
            height=7,
            exportselection=False,
            bg="#0f1420",
            fg="#ffffff",
            selectbackground="#0b79d0",
            selectforeground="#ffffff",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#ffffff",
            highlightcolor="#ffffff",
        )
        self.cards_listbox.grid(row=0, column=0, sticky="ew")
        self.cards_list_scrollbar = BlackOutlineScrollbar(
            cards_list_row,
            command=self.cards_listbox.yview,
            width=16,
        )
        self.cards_list_scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        self.cards_listbox.configure(yscrollcommand=self.cards_list_scrollbar.set)
        self.cards_listbox.bind("<<ListboxSelect>>", self._on_card_select)
        self.cards_listbox.bind("<MouseWheel>", self._on_cards_listbox_mousewheel, add="+")
        self.cards_listbox.bind("<Button-4>", self._on_cards_listbox_mousewheel, add="+")
        self.cards_listbox.bind("<Button-5>", self._on_cards_listbox_mousewheel, add="+")

        details = ttk.Frame(cards_wrap)
        details.grid(row=2, column=0, sticky="nsew")
        details.columnconfigure(0, weight=1)

        ttk.Label(details, text="Текущий вопрос:").grid(row=0, column=0, sticky="w")
        self.current_front_text = tk.Text(details, height=3, wrap=tk.WORD)
        self.current_front_text.grid(row=1, column=0, sticky="ew")
        self._install_text_edit_bindings(self.current_front_text)

        ttk.Label(details, text="Текущий ответ:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.current_back_text = tk.Text(details, height=4, wrap=tk.WORD)
        self.current_back_text.grid(row=3, column=0, sticky="ew")
        self._install_text_edit_bindings(self.current_back_text)

        action_row = ttk.Frame(details)
        action_row.grid(row=4, column=0, sticky="e", pady=(6, 0))
        ttk.Button(action_row, text="Удалить текущую", command=self.delete_current_card).pack(side=tk.RIGHT)
        ttk.Button(action_row, text="Сохранить все", command=self.save_all_cards_to_overview).pack(side=tk.RIGHT, padx=(0, 6))

        chat_wrap = ttk.LabelFrame(parent, text="Компактный чат", padding=8)
        chat_wrap.grid(row=5, column=1, sticky="nsew")
        chat_wrap.columnconfigure(0, weight=1)
        chat_wrap.rowconfigure(0, weight=1)

        self.chat_history = tk.Text(chat_wrap, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_history.grid(row=0, column=0, sticky="nsew")
        self._install_text_edit_bindings(self.chat_history, read_only=True)

        chat_input_row = ttk.Frame(chat_wrap)
        chat_input_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        chat_input_row.columnconfigure(0, weight=1)
        self.chat_input = tk.Text(chat_input_row, height=3, wrap=tk.WORD)
        self.chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._install_text_edit_bindings(self.chat_input)
        ttk.Button(chat_input_row, text="Отправить", command=self._chat_send).grid(row=0, column=1, sticky="ns")
        ttk.Button(chat_wrap, text="Использовать ответ как источник", command=self._use_chat_answer_as_source).grid(row=2, column=0, sticky="e", pady=(6, 0))

    def _install_text_edit_bindings(self, widget: tk.Text, read_only: bool = False) -> None:
        """Add predictable Ctrl+C/Ctrl+V and right-click menu to Tk Text widgets.

        Some custom mousewheel/global bindings can make normal text editing feel
        inconsistent on Windows. These bindings keep copy/paste/select-all
        available in the prompt, generated-card fields and compact chat.
        """
        try:
            widget.bind("<Control-c>", lambda e, w=widget: self._text_copy(w), add="+")
            widget.bind("<Control-C>", lambda e, w=widget: self._text_copy(w), add="+")
            widget.bind("<Control-a>", lambda e, w=widget: self._text_select_all(w), add="+")
            widget.bind("<Control-A>", lambda e, w=widget: self._text_select_all(w), add="+")
            if not read_only:
                widget.bind("<Control-v>", lambda e, w=widget: self._text_paste(w), add="+")
                widget.bind("<Control-V>", lambda e, w=widget: self._text_paste(w), add="+")
                widget.bind("<Control-x>", lambda e, w=widget: self._text_cut(w), add="+")
                widget.bind("<Control-X>", lambda e, w=widget: self._text_cut(w), add="+")
            widget.bind("<Button-3>", lambda e, w=widget, ro=read_only: self._show_text_context_menu(e, w, ro), add="+")
        except Exception:
            logging.exception("Text edit bindings install failed")

    def _text_copy(self, widget: tk.Text) -> str:
        try:
            text = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception:
            pass
        return "break"

    def _text_cut(self, widget: tk.Text) -> str:
        try:
            text = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.clipboard_clear()
            self.clipboard_append(text)
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except Exception:
            pass
        return "break"

    def _text_paste(self, widget: tk.Text) -> str:
        try:
            text = self.clipboard_get()
            try:
                widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except Exception:
                pass
            widget.insert(tk.INSERT, text)
        except Exception:
            pass
        return "break"

    def _text_select_all(self, widget: tk.Text) -> str:
        try:
            widget.tag_add(tk.SEL, "1.0", tk.END)
            widget.mark_set(tk.INSERT, "1.0")
            widget.see(tk.INSERT)
        except Exception:
            pass
        return "break"

    def _show_text_context_menu(self, event, widget: tk.Text, read_only: bool = False) -> str:
        try:
            menu = tk.Menu(self, tearoff=0, bg="#050505", fg="#ffffff", activebackground="#1f2937", activeforeground="#ffffff")
            menu.add_command(label="Копировать", command=lambda: self._text_copy(widget))
            if not read_only:
                menu.add_command(label="Вырезать", command=lambda: self._text_cut(widget))
                menu.add_command(label="Вставить", command=lambda: self._text_paste(widget))
            menu.add_separator()
            menu.add_command(label="Выделить всё", command=lambda: self._text_select_all(widget))
            menu.tk_popup(event.x_root, event.y_root)
        except Exception:
            logging.exception("Text context menu failed")
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass
        return "break"

    def _build_advanced_chat_tab(self) -> None:
        advanced_chat_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(advanced_chat_tab, text="Расширенный чат")
        if ChatBotTab is None:
            ttk.Label(advanced_chat_tab, text="Чат-бот недоступен. Проверьте chat_bot_tab.py.").pack(anchor="w")
            return
        try:
            self.chat_tab = ChatBotTab(advanced_chat_tab, app=self.app)
            self.chat_tab.pack(fill=tk.BOTH, expand=True)
        except Exception:
            logging.exception("ChatBotTab embed failed")
            ttk.Label(advanced_chat_tab, text="Чат-бот недоступен. Проверьте chat_bot_tab.py.").pack(anchor="w")

    def run_in_background(self, worker, on_success=None, on_error=None):
        if self._busy:
            return
        self._busy = True
        self.progress.start(10)

        def _runner():
            try:
                result = worker()
            except Exception as exc:
                def _err():
                    self._busy = False
                    self.progress.stop()
                    self.status_var.set("Ошибка")
                    if on_error:
                        on_error(exc)
                    else:
                        try:
                            messagebox.showerror("Ошибка", str(exc), parent=self)
                        except Exception:
                            pass
                self.after(0, _err)
                return

            def _ok():
                self._busy = False
                self.progress.stop()
                if on_success:
                    on_success(result)
            self.after(0, _ok)

        threading.Thread(target=_runner, daemon=True).start()

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(parent=self)
        if path:
            self.source_path = path
            self.status_var.set(f"Источник: {os.path.basename(path)}")

    def _extract_text(self) -> None:
        if not self.source_path:
            messagebox.showwarning("Источник", "Сначала выберите файл", parent=self)
            return

        def worker():
            bundle = self.pipeline.extract_source_bundle(self.source_path or "")
            return {"text": self.pipeline.clean_text(bundle.get("text", "")), "images": bundle.get("images", [])}

        self.status_var.set("Извлекаю текст...")
        self.run_in_background(worker, on_success=self._on_text_ready)

    def _on_text_ready(self, payload: dict) -> None:
        text = str((payload or {}).get("text") or "")
        images = list((payload or {}).get("images") or [])
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", text[:12000])
        self._last_source_trace = {
            "source_type": "file",
            "source_url": self.source_path or "",
            "source_title": os.path.basename(self.source_path or ""),
            "sources": [],
            "images": images,
        }
        self.status_var.set(f"Текст извлечён, изображений: {len(images)}")

    def _on_cards_listbox_mousewheel(self, event) -> str:
        try:
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                units = -3
            else:
                units = 3
            self.cards_listbox.yview_scroll(units, "units")
            if hasattr(self, "cards_list_scrollbar"):
                self.cards_list_scrollbar.set(*self.cards_listbox.yview())
        except Exception:
            logging.exception("Cards listbox mousewheel scroll failed")
        return "break"

    def _on_prompt_text_mousewheel(self, event) -> str:
        """Scroll only the source/prompt text box when the cursor is over it."""
        try:
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                self.prompt_text.yview_scroll(-3, "units")
            else:
                self.prompt_text.yview_scroll(3, "units")
            if hasattr(self, "prompt_scrollbar"):
                self.prompt_scrollbar.set(*self.prompt_text.yview())
        except Exception:
            logging.exception("Prompt text mousewheel scroll failed")
        return "break"

    def _get_source_query_text(self) -> str:
        try:
            return self.prompt_text.get(tk.SEL_FIRST, tk.SEL_LAST).strip()
        except Exception:
            return self.prompt_text.get("1.0", "end-1c").strip()

    def _normalize_web_query(self, raw: str) -> str:
        """Turn a long prompt/source dump into a usable web search query."""
        text = (raw or "").strip()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"(?i)\b(сгенерируй|создай)\s+карточк[ауи]\b", " ", text)
        text = re.sub(r"(?i)\b(с\s+картинкой|с\s+изображением|нарисуй|найди материалы|про)\b", " ", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            # Prefer a short human-readable line over a pasted article/search dump.
            short_lines = [line for line in lines if 5 <= len(line) <= 180]
            text = short_lines[0] if short_lines else " ".join(lines[:2])
        text = re.sub(r"\s+", " ", text).strip(" :-,.;")
        return text[:240]

    def _search_web(self) -> None:
        raw_input = self._get_source_query_text()
        raw_strip = (raw_input or "").strip()
        if not raw_strip:
            messagebox.showwarning("Поиск", "Введите URL, текст или тему", parent=self)
            return
        query = self._normalize_web_query(raw_input)
        if "youtube.com/" in raw_input or "youtu.be/" in raw_input:
            self.status_var.set("Извлекаю субтитры YouTube...")
        elif raw_strip.startswith(("http://", "https://")):
            self.status_var.set("Загружаю страницу...")
        else:
            if not query:
                messagebox.showwarning("Поиск", "Введите короткую тему для поиска", parent=self)
                return
            self.status_var.set("Поиск материалов...")
        self.run_in_background(
            lambda: self.rag_pipeline.fetch_materials(raw_strip if raw_strip.startswith(("http://", "https://")) else (query or raw_strip), max_sources=5),
            on_success=self._on_web_text,
            on_error=self._on_web_error,
        )

    def _on_web_text(self, result: dict) -> None:
        cleaned = ((result or {}).get("clean_text") or "").strip()
        if not cleaned:
            err = "; ".join((result or {}).get("errors") or []) if isinstance(result, dict) else ""
            self.status_var.set(err or "RAG не вернул текст. Уточните запрос.")
            return
        sources = result.get("sources") or []
        extracted_images: list[dict] = []
        for src in sources:
            meta = (src or {}).get("metadata") or {}
            extracted_images.extend(meta.get("images") or [])
        self._last_source_trace = {
            "source_type": result.get("source_type", "search"),
            "source_url": ((sources or [{}])[0]).get("url", ""),
            "source_title": ((sources or [{}])[0]).get("title", ""),
            "sources": sources,
            "images": extracted_images,
        }
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", cleaned[: self._max_rag_chars])
        self.status_var.set(
            f"Готово: {len(cleaned)} символов, {len(sources)} источников, изображений: {len(extracted_images)}"
        )

    def _on_web_error(self, exc: Exception) -> None:
        logging.exception("AI workspace RAG search failed")
        self.status_var.set("RAG-поиск не сработал")
        try:
            messagebox.showerror(
                "RAG-поиск",
                "Не удалось получить материалы из интернета.\n\n"
                f"Причина: {exc}\n\n"
                "Проверьте интернет, rag_web_search.py и не отправляйте в поиск слишком длинный текст.\n"
                "Лучше выделите короткую тему и нажмите 🔍.",
                parent=self,
            )
        except Exception:
            pass

    def generate_cards_from_input(self) -> None:
        text = self.prompt_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Пусто", "Нет текста для генерации карточек", parent=self)
            return

        llm_settings = self._read_shared_llm_settings()
        model = str(llm_settings.get("model") or "").strip() or "auto"
        self.status_var.set(f"Генерация вопросов: LLM | model={model}")
        self._update_llm_summary()
        self._refresh_llm_runtime_status()
        self.run_in_background(
            lambda: self.pipeline.run_pipeline(
                source_text=text,
                source_trace=self._last_source_trace,
                options={
                    "mode": "accurate",
                    "question_style": "creative_grounded",
                    "source_trace": self._last_source_trace,
                    "llm_settings": llm_settings,
                },
                source=self.source_path,
            ),
            on_success=self._on_cards_generated,
        )

    def _resolve_chatbot_settings_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot_settings.json")

    def _read_shared_llm_settings(self) -> dict:
        payload: dict = {}
        try:
            if os.path.exists(self._llm_settings_path):
                with open(self._llm_settings_path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    payload = dict(loaded)
        except Exception:
            logging.exception("AI workspace: failed to read chatbot_settings.json")
        provider_raw = str(payload.get("llm_provider") or payload.get("provider") or "Ollama").strip().lower()
        provider = "ollama" if "ollama" in provider_raw else provider_raw
        result = {
            "provider": provider or "ollama",
            "ollama_url": str(payload.get("ollama_url") or os.getenv("XFLASH_OLLAMA_URL", "http://127.0.0.1:11434")).strip(),
            "model": str(payload.get("ollama_model") or os.getenv("XFLASH_DEFAULT_MODEL", "")).strip(),
            "timeout": int(payload.get("timeout") or 180),
            "temperature": float(payload.get("temperature") or 0.4),
            "max_tokens": int(payload.get("max_tokens") or 700),
            "auto_fallback": bool(payload.get("auto_fallback", True)),
            "image_provider": str(payload.get("image_provider") or os.getenv("XFLASH_IMAGE_PROVIDER", "flux_comfyui")).strip(),
            "flux_api_url": str(payload.get("flux_api_url") or os.getenv("XFLASH_FLUX_API_URL", "http://127.0.0.1:8188")).strip(),
            "flux_model_path": str(payload.get("flux_model_path") or os.getenv("XFLASH_FLUX_MODEL_PATH", "models/flux/flux-2-klein-4b-fp8.safetensors")).strip(),
            "flux_output_dir": str(payload.get("flux_output_dir") or os.getenv("XFLASH_FLUX_OUTPUT_DIR", "media/generated")).strip(),
        }
        self._cached_llm_settings = result
        return result

    def _update_llm_summary(self) -> None:
        settings = self._read_shared_llm_settings()
        self.llm_summary_var.set(
            f"AI backend | LLM: Ollama | model: {settings.get('model') or '—'} | Image: FLUX.2 Klein FP8 / ComfyUI"
        )

    def _refresh_llm_runtime_status(self) -> None:
        settings = dict(self._cached_llm_settings or self._read_shared_llm_settings())
        self.llm_runtime_status_var.set("LLM status: checking...")
        self.image_runtime_status_var.set("Image status: checking...")

        def worker() -> None:
            status = "LLM status: error"
            image_status = "Image status: error"
            try:
                from ollama_client import OllamaClient

                client = OllamaClient(
                    base_url=str(settings.get("ollama_url") or ""),
                    model=str(settings.get("model") or ""),
                    auto_fallback=bool(settings.get("auto_fallback", True)),
                )
                models = client.list_models()
                model = str(settings.get("model") or "").strip()
                if model and model in models:
                    status = "LLM status: OK"
                elif models:
                    status = "LLM status: error (модель не найдена)"
                else:
                    status = "LLM status: error (модели отсутствуют)"
            except Exception:
                status = "LLM status: error"
            try:
                from flux_image_adapter import FluxImageAdapter
                ok, msg = FluxImageAdapter(
                    api_url=str(settings.get("flux_api_url") or "http://127.0.0.1:8188"),
                    model_path=str(settings.get("flux_model_path") or ""),
                    output_dir=str(settings.get("flux_output_dir") or "media/generated"),
                ).is_available()
                image_status = f"Image: FLUX.2 Klein FP8 / ComfyUI | status: {'OK' if ok else 'error'} ({msg})"
            except Exception as exc:
                image_status = f"Image: FLUX.2 Klein FP8 / ComfyUI | status: error ({exc})"
            self.after(0, lambda: self.llm_runtime_status_var.set(status))
            self.after(0, lambda: self.image_runtime_status_var.set(image_status))

        threading.Thread(target=worker, daemon=True).start()

    def _open_llm_settings(self) -> None:
        chat_tab = getattr(self, "chat_tab", None)
        opener = getattr(chat_tab, "_open_llm_settings", None)
        if callable(opener):
            opener()
            win = getattr(chat_tab, "llm_settings_window", None)
            if win is not None and hasattr(win, "bind"):
                win.bind("<Destroy>", lambda _e: (self._update_llm_summary(), self._refresh_llm_runtime_status()), add="+")
            return
        messagebox.showinfo(
            "Настройки LLM",
            "Откройте вкладку «Расширенный чат» и используйте кнопку «⚙ Настройки LLM».",
            parent=self,
        )

    def _on_cards_generated(self, cards) -> None:
        self.generated_cards = list(cards or [])
        self.current_card_index = 0
        self.show_current_card()
        generation_mode = str(getattr(self.pipeline, "last_generation_mode", "fallback"))
        fallback_reason = str(getattr(self.pipeline, "last_fallback_reason", "") or "").strip()
        if self.generated_cards:
            if generation_mode == "llm":
                model = getattr(self.pipeline, "last_llm_model", "") or (self._cached_llm_settings or {}).get("model") or "unknown"
                self.status_var.set(f"Генерация вопросов: LLM | model={model}. Сгенерировано карточек: {len(self.generated_cards)}")
            else:
                reason_part = f" LLM генерация не сработала: {fallback_reason}." if fallback_reason else ""
                self.status_var.set(
                    f"Генерация вопросов: fallback. LLM недоступна, используется fallback.{reason_part} Сгенерировано карточек: {len(self.generated_cards)}"
                )
            self.auto_generate_image_after_card = False
        else:
            if generation_mode == "llm":
                self.status_var.set("Генерация вопросов: LLM. Карточки не сгенерированы. Попробуйте уточнить тему.")
            else:
                reason_part = f" LLM генерация не сработала: {fallback_reason}." if fallback_reason else ""
                self.status_var.set(
                    f"Генерация вопросов: fallback. LLM недоступна, используется fallback.{reason_part} Карточки не сгенерированы."
                )

    def show_current_card(self) -> None:
        self.cards_listbox.delete(0, tk.END)
        for idx, card in enumerate(self.generated_cards, start=1):
            front = (card.get("front") or "").strip().replace("\n", " ")
            q = float(card.get("quality_score") or 0.0)
            ctype = str(card.get("card_type") or "fact")
            image_tag = self._card_image_tag(card)
            self.cards_listbox.insert(tk.END, f"{idx}. {front[:62] or '(без вопроса)'} | q {q:.2f} | {ctype} | {image_tag}")

        total = len(self.generated_cards)
        counter = f"{self.current_card_index + 1}/{total}" if total else "0/0"
        self.cards_counter_var.set(counter)

        self.current_front_text.delete("1.0", tk.END)
        self.current_back_text.delete("1.0", tk.END)
        if not total:
            self.preview.clear()
            return

        self.current_card_index = max(0, min(self.current_card_index, total - 1))
        card = self.generated_cards[self.current_card_index]
        self.cards_listbox.selection_clear(0, tk.END)
        self.cards_listbox.selection_set(self.current_card_index)
        self.preview.update_preview(card)
        self.current_front_text.insert("1.0", card.get("front", ""))
        self.current_back_text.insert("1.0", card.get("back", ""))
        self.status_var.set(f"Статус карточки: {self._card_image_tag(card)}")

    def _card_image_tag(self, card: dict) -> str:
        c = normalize_card_image_fields(card)
        status = str(c.get("image_status") or (c.get("metadata") or {}).get("image_status") or "").strip().lower()
        if "source_rejected_topic_mismatch" in status:
            return "Source image rejected: topic mismatch"
        if "source_rejected_too_generic" in status:
            return "Source image rejected: too generic"
        if status.startswith("source_attached"):
            caption = str(c.get("front_image_caption") or "").strip()
            return f"Source image attached: {caption[:48]}" if caption else "Source image attached"
        if "no_relevant_source_image" in status:
            return "No relevant source image found"
        if "flux" in status and ("создано" in status or "generated" in status):
            return "FLUX image generated"
        source_type = str(c.get("front_image_origin") or c.get("image_source_type") or "").strip().lower()
        if source_type in {"source", "extracted"}:
            return "image: source"
        if source_type == "generated":
            return "image: generated"
        if source_type == "source_url_not_downloaded":
            return "image: source_url_not_downloaded"
        if source_type == "failed":
            return "image: failed"
        if source_type == "none":
            return "image: none"
        if c.get("front_image_path"):
            return "image: source"
        if c.get("front_image_url"):
            return "image: source_url_not_downloaded"
        if c.get("image_path"):
            return "image: generated"
        return "image: none"

    def next_card(self) -> None:
        if not self.generated_cards:
            return
        self.current_card_index = (self.current_card_index + 1) % len(self.generated_cards)
        self.show_current_card()

    def prev_card(self) -> None:
        if not self.generated_cards:
            return
        self.current_card_index = (self.current_card_index - 1) % len(self.generated_cards)
        self.show_current_card()

    def delete_current_card(self) -> None:
        if not self.generated_cards:
            return
        self.generated_cards.pop(self.current_card_index)
        if self.current_card_index >= len(self.generated_cards):
            self.current_card_index = max(0, len(self.generated_cards) - 1)
        self.show_current_card()
        self.status_var.set("Текущая карточка удалена")

    def generate_image_for_current_card(self) -> None:
        if not self.generated_cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return
        idx = self.current_card_index

        def worker():
            card = dict(self.generated_cards[idx])
            if not card.get("image_prompt"):
                card["image_prompt"] = self.pipeline.generate_image_prompt(card)
            return self._generate_card_image_with_diagnostics(card)

        if self.generated_cards[idx].get("needs_image", False):
            self.status_var.set("FLUX: генерирую изображение...")
        else:
            self.status_var.set("FLUX: генерирую изображение...")
        self.run_in_background(worker, on_success=self._on_image_generated, on_error=self._on_image_error)

    def generate_images_for_all_cards(self) -> None:
        if not self.generated_cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return
        if not self.generate_images_for_all_var.get():
            self.status_var.set("Массовая генерация выключена: включите checkbox «Сгенерировать картинки для всех карточек».")
            return

        def worker():
            cards = [dict(card or {}) for card in self.generated_cards]
            targets = [i for i, card in enumerate(cards) if card.get("needs_image", True)]
            total = len(targets)
            if total == 0:
                return cards, 0, 0
            generated = 0
            failed = 0
            for done, card_idx in enumerate(targets, start=1):
                self.after(0, lambda d=done, t=total: self.status_var.set(f"FLUX: генерирую изображение {d}/{t}..."))
                card = cards[card_idx]
                if not card.get("image_prompt"):
                    card["image_prompt"] = self.pipeline.generate_image_prompt(card)
                result = self._generate_card_image_with_diagnostics(card)
                cards[card_idx] = result
                if result.get("image_path"):
                    generated += 1
                else:
                    failed += 1
            return cards, generated, failed

        self.status_var.set("FLUX: генерирую изображения для всех карточек...")
        self.run_in_background(worker, on_success=self._on_all_images_generated, on_error=self._on_image_error)

    def _get_app_setting(self, *names, default=None):
        for name in names:
            for owner in (self.app, getattr(self.app, "settings", None), getattr(self.app, "llm_settings", None)):
                if owner is None:
                    continue
                try:
                    if isinstance(owner, dict) and name in owner:
                        value = owner.get(name)
                    elif hasattr(owner, name):
                        value = getattr(owner, name)
                    else:
                        continue
                    if hasattr(value, "get") and callable(value.get):
                        value = value.get()
                    if value not in (None, ""):
                        return value
                except Exception:
                    continue
        return default

    def _normalize_sd_model_name(self, model: str | None) -> str:
        model = (model or "sd_xl_base_1.0.safetensors").strip()
        # Common typo from the settings field: .safetenso -> .safetensors
        if model.endswith(".safetenso"):
            model += "rs"
        return model

    def _generate_card_image_with_diagnostics(self, card: dict) -> dict:
        metadata = dict(card.get("metadata") or {})
        try:
            from image_generation_adapter import ImageGenerationAdapter

            settings = getattr(self.app, "llm_settings", None) if self.app is not None else None
            result = ImageGenerationAdapter(settings=settings, app=self.app).generate_card_image(dict(card), options={"force_generate": True})
            status = result.get("image_status") or ((result.get("metadata") or {}).get("image_status") or "")
            metadata.update(dict(result.get("metadata") or {}))
            result["metadata"] = metadata
            if result.get("front_image_path"):
                metadata["image_status"] = "FLUX image generated"
            elif status:
                metadata["image_status"] = f"FLUX error: {status}"
            result["metadata"] = metadata
            return result
        except Exception as exc:
            metadata["image_status"] = f"FLUX error: {exc}"
            card["metadata"] = metadata
            return card

    def _on_image_generated(self, card) -> None:
        if self.generated_cards:
            self.generated_cards[self.current_card_index] = card
        self._refresh_current_card_preview()
        status = ((card.get("metadata") or {}).get("image_status") or "").strip()
        self.status_var.set(status or "Генерация изображения завершена")

    def _on_all_images_generated(self, payload) -> None:
        cards, generated, failed = payload
        self.generated_cards = list(cards or [])
        self._refresh_current_card_preview()
        if generated == 0 and failed == 0:
            self.status_var.set("Нет карточек, которым нужна картинка")
            return
        self.status_var.set(f"Готово: сгенерировано {generated}, ошибок {failed}")

    def _refresh_current_card_preview(self) -> None:
        total = len(self.generated_cards)
        counter = f"{self.current_card_index + 1}/{total}" if total else "0/0"
        self.cards_counter_var.set(counter)
        self.current_front_text.delete("1.0", tk.END)
        self.current_back_text.delete("1.0", tk.END)
        if not total:
            self.preview.clear()
            return
        self.current_card_index = max(0, min(self.current_card_index, total - 1))
        card = self.generated_cards[self.current_card_index]
        self.preview.update_preview(card)
        self.current_front_text.insert("1.0", card.get("front", ""))
        self.current_back_text.insert("1.0", card.get("back", ""))

    def _on_image_error(self, exc: Exception) -> None:
        logging.exception("AI workspace image generation failed")
        self.status_var.set("FLUX error")
        try:
            messagebox.showerror(
                "FLUX",
                "Не удалось сгенерировать изображение.\n\n"
                f"Причина: {exc}\n\n"
                "Проверьте: AUTOMATIC1111 запущен с --api, URL http://127.0.0.1:7860 доступен, "
                "модель называется .safetensors, а не .safetenso.",
                parent=self,
            )
        except Exception:
            pass

    def save_current_card_to_overview(self) -> None:
        if not self.generated_cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return
        deck_id = self._get_selected_deck_id()
        if deck_id is None:
            messagebox.showwarning("Колода", "Выберите колоду для сохранения карточки", parent=self)
            return
        current_card = self.generated_cards[self.current_card_index]
        self.status_var.set("Сохраняю карточку...")
        self.run_in_background(lambda: self.pipeline.save_cards_to_overview([current_card]), on_success=self._on_saved)

    def save_all_cards_to_overview(self) -> None:
        if not self.generated_cards:
            messagebox.showwarning("Нет карточек", "Сначала сгенерируйте карточки", parent=self)
            return
        deck_id = self._get_selected_deck_id()
        if deck_id is None:
            messagebox.showwarning("Колода", "Выберите колоду для сохранения карточек", parent=self)
            return
        self.status_var.set("Сохраняю все карточки...")
        self.run_in_background(lambda: self.pipeline.save_cards_to_overview(self.generated_cards), on_success=self._on_saved)

    def _on_saved(self, saved_count: int) -> None:
        self.status_var.set(f"Сохранено в ознакомление: {saved_count}")
        try:
            if hasattr(self.app, "refresh_deck_counters_and_phase_tree") and callable(self.app.refresh_deck_counters_and_phase_tree):
                self.app.refresh_deck_counters_and_phase_tree()
            if hasattr(self.app, "refresh_decks"):
                self.app.refresh_decks()
        except Exception:
            logging.exception("refresh_decks failed")

    def clear_workspace(self) -> None:
        self.generated_cards = []
        self.current_card_index = 0
        self.auto_generate_image_after_card = False
        self.source_path = None
        self.prompt_text.delete("1.0", tk.END)
        self.chat_input.delete("1.0", tk.END)
        self._last_chat_answer = ""
        self.show_current_card()
        self.status_var.set("Очищено")

    def open_manual_editor(self) -> None:
        try:
            self.app.add_card_window()
        finally:
            self.destroy()

    def handle_chat_command(self, text: str) -> bool:
        cleaned = (text or "").strip()
        lowered = cleaned.lower()
        if not any(k in lowered for k in ("сгенерируй карточку", "создай карточку")):
            return False

        need_image = any(k in lowered for k in ("с картинкой", "с изображением", "нарисуй", "сгенерируй изображение"))
        topic = cleaned
        topic = re.sub(r"(?i)^\s*(сгенерируй|создай)\s+карточк[ауи]\s*", "", topic).strip(" :,-")
        topic = re.sub(r"(?i)^с\s+(картинкой|изображением)\b", "", topic).strip(" :,-")
        topic = re.sub(r"(?i)^про\b", "", topic).strip(" :,-")
        if not topic:
            topic = "тема карточки"

        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", topic)
        self.auto_generate_image_after_card = need_image
        self.status_var.set("Распознана команда генерации карточки")
        self.generate_cards_from_input()
        return True

    def _chat_send(self) -> None:
        text = self.chat_input.get("1.0", "end-1c").strip()
        if not text:
            return
        self.chat_input.delete("1.0", tk.END)
        self._chat_append("Вы", text)
        if self.handle_chat_command(text):
            return
        self.status_var.set("Чат: ищу ответ...")
        self.run_in_background(lambda: self.web_search.search_and_extract(text), on_success=self._chat_on_answer)

    def _chat_on_answer(self, answer: str) -> None:
        self._last_chat_answer = answer or ""
        self._chat_append("AI", self._last_chat_answer or "Пустой ответ")
        self.status_var.set("Готово")

    def _chat_append(self, author: str, text: str) -> None:
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"{author}:\n{text}\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state=tk.DISABLED)

    def _use_chat_answer_as_source(self) -> None:
        if not self._last_chat_answer:
            messagebox.showwarning("Чат", "Нет ответа для вставки", parent=self)
            return
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", self._last_chat_answer)
        self.status_var.set("Ответ чата вставлен в источник")

    def _on_card_select(self, _event=None) -> None:
        if not self.generated_cards:
            return
        selection = self.cards_listbox.curselection()
        if not selection:
            return
        self.current_card_index = selection[0]
        self.show_current_card()
