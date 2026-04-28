from __future__ import annotations

import os
import webbrowser
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
        ttk.Radiobutton(top, text="Лицевая", value="front", variable=self.side_var, command=self.show_front).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Обратная", value="back", variable=self.side_var, command=self.show_back).pack(side=tk.LEFT, padx=(8, 0))

        self.text = tk.Text(self, height=8, wrap=tk.WORD)
        self.text.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        image_frame = ttk.Frame(self)
        image_frame.grid(row=1, column=1, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(image_frame, anchor="center", justify="center", relief="groove")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.media_status_var = tk.StringVar(value="")
        ttk.Label(image_frame, textvariable=self.media_status_var, foreground="gray").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.open_media_btn = ttk.Button(image_frame, text="Открыть media файл", command=self._open_media)
        self.open_media_btn.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.clear()

    def update_preview(self, card_data) -> None:
        self._card = dict(card_data or {})
        side = self.side_var.get()
        text_value = self._card.get(side) or ""
        if not text_value and not self._card:
            text_value = "Пока карточка не создана. Введите тему или загрузите источник."
        explanation = (self._card.get("explanation") or "").strip()
        translation = (self._card.get("translation") or "").strip()
        key_words = self._card.get("key_words") or []
        if side == "back" and explanation:
            text_value = f"{text_value}\n\nПояснение: {explanation}"
        if side == "back" and translation:
            text_value = f"{text_value}\n\nПеревод: {translation}"
        if side == "back" and key_words:
            text_value = f"{text_value}\n\nКлючевые слова: {', '.join(str(x) for x in key_words)}"
        if side == "front" and str(self._card.get("card_type") or "") == "listening":
            text_value = "Прослушайте фрагмент и напишите, что сказано."
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text_value)

        image_path = self._card.get("front_image_path") or self._card.get("answer_image_path") or self._card.get("image_path")
        if image_path:
            self._set_image(image_path)
        else:
            status = ((self._card.get("metadata") or {}).get("image_status") or "").strip()
            image_url = (self._card.get("answer_image_url") or "").strip()
            msg = "Пока карточка не создана. Введите тему или загрузите источник." if not self._card else "Нет изображения (placeholder)"
            if image_url:
                msg = f"Изображение прикреплено ссылкой:\n{image_url}"
            if status:
                msg = f"{msg}\n{status}"
            self._set_placeholder(msg)
        audio_path = (self._card.get("audio_path") or "").strip()
        video_path = (self._card.get("video_path") or "").strip()
        media_text = "audio/video attached" if (audio_path or video_path) else "media not attached"
        if video_path:
            media_text += f"\nvideo: {video_path}"
        elif audio_path:
            media_text += f"\naudio: {audio_path}"
        self.media_status_var.set(media_text)

        if side == "front":
            caption = (self._card.get("answer_image_caption") or "").strip()
            if caption:
                self.text.insert(tk.END, f"\n\nЯкорь ответа (image): {caption}")

    def show_front(self) -> None:
        self.side_var.set("front")
        self.update_preview(self._card)

    def show_back(self) -> None:
        self.side_var.set("back")
        self.update_preview(self._card)

    def clear(self) -> None:
        self._card = {}
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", "Пока карточка не создана. Введите тему или загрузите источник.")
        self._set_placeholder("Пока карточка не создана. Введите тему или загрузите источник.")
        self.media_status_var.set("")

    def _set_image(self, image_path: str) -> None:
        if not os.path.exists(image_path):
            self._set_placeholder(f"Файл изображения не найден:\n{image_path}")
            return
        if Image is None or ImageTk is None:
            self._set_placeholder("Pillow не установлен: предпросмотр картинки недоступен")
            return
        try:
            img = Image.open(image_path)
            img.thumbnail((330, 210))
            tk_img = ImageTk.PhotoImage(img)
        except Exception as exc:
            self._set_placeholder(f"Не удалось отрисовать изображение: {exc}")
            return
        self._image_ref = tk_img
        self.image_label.configure(image=tk_img, text="")

    def _set_placeholder(self, text: str) -> None:
        self._image_ref = None
        self.image_label.configure(image="", text=text)

    def _open_media(self) -> None:
        media_path = (self._card.get("video_path") or self._card.get("audio_path") or "").strip()
        if not media_path:
            return
        if media_path.startswith("http://") or media_path.startswith("https://"):
            webbrowser.open(media_path)
            return
        if os.path.exists(media_path):
            webbrowser.open(f"file://{os.path.abspath(media_path)}")
