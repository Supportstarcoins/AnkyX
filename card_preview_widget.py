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

        top = ttk.Frame(self)
        top.pack(fill=tk.X, pady=(0, 6))
        ttk.Radiobutton(top, text="Лицевая", value="front", variable=self.side_var, command=self.render).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Обратная", value="back", variable=self.side_var, command=self.render).pack(side=tk.LEFT, padx=(8, 0))

        self.text = tk.Text(self, height=8, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(self, text="Изображение не выбрано")
        self.image_label.pack(fill=tk.X, pady=(6, 0))

        self._card: dict = {}

    def set_card(self, card: dict | None) -> None:
        self._card = card or {}
        self.render()

    def render(self) -> None:
        side = self.side_var.get()
        text_value = self._card.get(side, "") if isinstance(self._card, dict) else ""
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text_value)

        path = self._card.get("image_path") if isinstance(self._card, dict) else None
        if not path:
            self.image_label.configure(text="Нет изображения (placeholder)", image="")
            self._image_ref = None
            return
        if not os.path.exists(path):
            self.image_label.configure(text="Файл изображения не найден", image="")
            self._image_ref = None
            return
        if Image is None or ImageTk is None:
            self.image_label.configure(text="Pillow не установлен: предпросмотр картинки недоступен", image="")
            self._image_ref = None
            return

        try:
            img = Image.open(path)
            img.thumbnail((360, 240))
            tk_img = ImageTk.PhotoImage(img)
        except Exception as exc:
            self.image_label.configure(text=f"Не удалось отрисовать изображение: {exc}", image="")
            self._image_ref = None
            return

        self._image_ref = tk_img
        self.image_label.configure(image=tk_img, text="")
