from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


class CardPreviewWidget(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.side_var = tk.StringVar(value="front")
        self._image_ref = None
        self._card: dict = {}

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Radiobutton(top, text="Лицевая", value="front", variable=self.side_var, command=self.render).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Обратная", value="back", variable=self.side_var, command=self.render).pack(side=tk.LEFT, padx=(8, 0))

        self.text = tk.Text(self, height=8, wrap=tk.WORD)
        self.text.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        image_frame = ttk.Frame(self)
        image_frame.grid(row=1, column=1, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(
            image_frame,
            text="Пока карточка не создана. Введите тему или загрузите источник.",
            anchor="center",
            justify="center",
            relief="groove",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.render()

    def set_card(self, card: dict | None) -> None:
        self.update_preview(card)

    def update_preview(self, card: dict | None) -> None:
        self._card = card or {}
        self.render()

    def render(self) -> None:
        side = self.side_var.get()
        text_value = self._card.get(side, "") if isinstance(self._card, dict) else ""
        if not text_value and not self._card:
            text_value = "Пока карточка не создана. Введите тему или загрузите источник."

        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text_value)

        path = self._card.get("image_path") if isinstance(self._card, dict) else None
        status = ""
        if isinstance(self._card, dict):
            status = ((self._card.get("metadata") or {}).get("image_status") or "").strip()
        if not path:
            placeholder = "Нет изображения (placeholder)"
            if not self._card:
                placeholder = "Карточка ещё не создана"
            if status:
                placeholder = f"{placeholder}\n{status}"
            self.image_label.configure(text=placeholder, image="")
            self._image_ref = None
            return
        if not os.path.exists(path):
            self.image_label.configure(text=f"Файл изображения не найден:\n{path}", image="")
            self._image_ref = None
            return
        if Image is None or ImageTk is None:
            self.image_label.configure(text="Pillow не установлен: предпросмотр картинки недоступен", image="")
            self._image_ref = None
            return

        try:
            img = Image.open(path)
            img.thumbnail((330, 210))
            tk_img = ImageTk.PhotoImage(img)
        except Exception as exc:
            self.image_label.configure(text=f"Не удалось отрисовать изображение: {exc}", image="")
            self._image_ref = None
            return

        self._image_ref = tk_img
        self.image_label.configure(image=tk_img, text="")
