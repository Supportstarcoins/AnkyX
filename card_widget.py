import os
import tkinter as tk
from tkinter import messagebox, ttk

from ui_theme import get_card_surface_colors, style_card_surface, style_card_surface_text
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageTk = None
    PIL_AVAILABLE = False
from image_utils import MAX_PREVIEW_PIXELS, load_preview_image, log_image_error


def _pil_lanczos():
    if Image is None:
        return 1
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        return getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", 1))


class CardWidget(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        palette: dict | None = None,
        editable: bool = False,
        width: int = 800,
        height: int = 520,
        show_image_toolbar: bool = True,
        image_layout: str = "side",
    ) -> None:
        card_bg, card_text, _ = get_card_surface_colors(master)
        super().__init__(master, bg=card_bg, bd=0, relief="flat", width=width, height=height)
        style_card_surface(self, palette)
        self.pack_propagate(False)

        self.palette = palette or {}
        self.card_bg = card_bg
        self.card_text = card_text
        self.editable = editable
        self.show_image_toolbar = show_image_toolbar
        self.show_back = False
        self.image_layout = image_layout

        self.image_mode = "fit"
        self.zoom_factor = 1.0
        self._last_fit_scale = 1.0
        self._image_path: str | None = None
        self._img_ref = None
        self._warned_large_path: str | None = None
        self.scrollbars_enabled = True
        self._last_video_height = 80
        self.media_width = 260
        self.media_bg = "white"

        self._build_layout()

    def _build_layout(self) -> None:
        content_container = tk.Frame(self, bg=self.card_bg)
        self.content_container = content_container
        self._update_content_geometry()
        self.bind("<Configure>", lambda _e: self._update_content_geometry())

        canvas = tk.Canvas(content_container, bg=self.card_bg, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=canvas.yview)
        self.content_canvas = canvas
        self.content_scrollbar = scrollbar
        self.scrollable_frame = tk.Frame(canvas, bg=self.card_bg)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.content_frame = tk.Frame(self.scrollable_frame, bg=self.card_bg)
        self.content_frame.grid(row=0, column=0, sticky="nsew")
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        if self.image_layout == "below":
            self.content_frame.grid_columnconfigure(0, weight=1)
            self.content_frame.grid_rowconfigure(0, weight=1)
            self.content_frame.grid_rowconfigure(1, weight=1)
        else:
            self.content_frame.grid_columnconfigure(0, weight=0, minsize=self.media_width)
            self.content_frame.grid_columnconfigure(1, weight=1)
            self.content_frame.grid_rowconfigure(0, weight=1)

        self.text_frame = tk.Frame(self.content_frame, bg=self.card_bg)
        if self.image_layout == "below":
            self.text_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        else:
            self.text_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.custom_text_frame = tk.Frame(self.text_frame, bg=self.card_bg)
        self._use_custom_text = False

        if self.editable:
            self.front_text = tk.Text(self.text_frame, wrap=tk.WORD, height=10)
            self.back_text = tk.Text(self.text_frame, wrap=tk.WORD, height=10)
            style_card_surface_text(self.front_text, self.palette)
            style_card_surface_text(self.back_text, self.palette)
        else:
            self.front_text = tk.Label(
                self.text_frame,
                text="",
                bg=self.card_bg,
                fg=self.card_text,
                wraplength=400,
                justify="left",
                font=("Segoe UI", 12),
            )
            self.back_text = tk.Label(
                self.text_frame,
                text="",
                bg=self.card_bg,
                fg=self.card_text,
                wraplength=400,
                justify="left",
                font=("Segoe UI", 12),
            )

        self.front_text.pack(anchor="w", fill=tk.BOTH, expand=True)
        self.back_text.pack_forget()
        self.custom_text_frame.pack_forget()

        self.image_frame = tk.Frame(self.content_frame, bg=self.card_bg, width=self.media_width)
        if self.image_layout == "below":
            self.image_frame.grid(row=1, column=0, sticky="nsew")
        else:
            self.image_frame.grid(row=0, column=0, sticky="nsew")
            self.image_frame.grid_propagate(False)

        if self.show_image_toolbar:
            self.toolbar = ttk.Frame(self.image_frame, style="CardInner.TFrame")
            self.toolbar.pack(fill=tk.X, pady=(0, 6))

            ttk.Button(
                self.toolbar,
                text="Вписать",
                style="Secondary.TButton",
                command=lambda: self.set_image_mode("fit"),
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                self.toolbar,
                text="1:1",
                style="Secondary.TButton",
                command=lambda: self.set_image_mode("actual"),
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                self.toolbar,
                text="Zoom +",
                style="Secondary.TButton",
                command=lambda: self.zoom_image(1.12),
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                self.toolbar,
                text="Zoom -",
                style="Secondary.TButton",
                command=lambda: self.zoom_image(0.88),
            ).pack(side=tk.LEFT, padx=2)
        else:
            self.toolbar = None

        canvas_container = tk.Frame(self.image_frame, bg=self.media_bg)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(
            canvas_container,
            bg=self.media_bg,
            highlightthickness=0,
            borderwidth=0,
        )
        self.image_scroll_y = ttk.Scrollbar(canvas_container, orient="vertical", command=self.image_canvas.yview)
        self.image_scroll_x = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=self.image_scroll_y.set, xscrollcommand=self.image_scroll_x.set)

        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        self.image_scroll_y.grid(row=0, column=1, sticky="ns")
        self.image_scroll_x.grid(row=1, column=0, sticky="ew")

        self.image_canvas.bind("<Configure>", lambda _e: self.render_image())

        self.audio_inline_frame = ttk.Frame(self, style="CardInner.TFrame")
        self.audio_inline_frame.place_forget()

        self.video_inline_frame = ttk.Frame(self, style="CardInner.TFrame")
        self.video_inline_frame.place_forget()

    def set_text(self, front: str, back: str) -> None:
        if self.editable:
            self.front_text.delete("1.0", tk.END)
            self.front_text.insert("1.0", front)
            self.back_text.delete("1.0", tk.END)
            self.back_text.insert("1.0", back)
        else:
            self.front_text.configure(text=front)
            self.back_text.configure(text=back)

    def get_text(self) -> tuple[str, str]:
        if not self.editable:
            return (
                str(self.front_text.cget("text")),
                str(self.back_text.cget("text")),
            )
        return (
            self.front_text.get("1.0", tk.END).strip(),
            self.back_text.get("1.0", tk.END).strip(),
        )

    def show_side(self, show_back: bool, image_path: str | None = None) -> None:
        self.show_back = show_back
        if self._use_custom_text:
            self.front_text.pack_forget()
            self.back_text.pack_forget()
            if not self.custom_text_frame.winfo_ismapped():
                self.custom_text_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self.custom_text_frame.pack_forget()
            if show_back:
                self.front_text.pack_forget()
                self.back_text.pack(anchor="w", fill=tk.BOTH, expand=True)
            else:
                self.back_text.pack_forget()
                self.front_text.pack(anchor="w", fill=tk.BOTH, expand=True)
        if image_path is not None:
            self._image_path = image_path
        self.render_image()

    def use_custom_text(self, enabled: bool) -> None:
        self._use_custom_text = enabled
        if not enabled:
            for widget in self.custom_text_frame.winfo_children():
                widget.destroy()
        self.show_side(self.show_back)

    def clear_custom_text(self) -> None:
        for widget in self.custom_text_frame.winfo_children():
            widget.destroy()

    def set_image_mode(self, mode: str) -> None:
        if mode not in ("fit", "actual", "zoom"):
            return
        if mode == "actual":
            self.zoom_factor = 1.0
        self.image_mode = mode
        self.render_image()

    def zoom_image(self, factor: float) -> None:
        if not self._image_path:
            return
        if self.image_mode == "fit":
            self.image_mode = "zoom"
            self.zoom_factor = max(0.1, self._last_fit_scale)
        elif self.image_mode == "actual":
            self.image_mode = "zoom"
            self.zoom_factor = 1.0
        self.zoom_factor = max(0.1, min(6.0, self.zoom_factor * factor))
        self.render_image()

    def render_image(self) -> None:
        path = self._image_path
        if not path or not PIL_AVAILABLE or not os.path.exists(path):
            self._img_ref = None
            if self.image_canvas is not None:
                self.image_canvas.delete("card_preview")
            return

        try:
            if self.image_canvas is None:
                return
            cont_w = max(1, self.image_canvas.winfo_width())
            cont_h = max(1, self.image_canvas.winfo_height())
            cont_w = max(1, cont_w - 10)
            cont_h = max(1, cont_h - 10)
            max_zoom = 6.0
            max_preview = (int(cont_w * max_zoom), int(cont_h * max_zoom))
            img, resized_for_pixels = load_preview_image(
                path,
                max_preview,
                max_pixels=MAX_PREVIEW_PIXELS,
            )
            if resized_for_pixels and self._warned_large_path != path:
                messagebox.showinfo(
                    "Большое изображение",
                    "Изображение слишком большое, будет сжато для превью.",
                )
                self._warned_large_path = path

            img_w, img_h = img.size
            if img_w <= 0 or img_h <= 0:
                raise ValueError("Invalid image size")
            fit_scale = min(cont_w / img_w, cont_h / img_h)
            self._last_fit_scale = fit_scale
            if self.image_mode == "fit":
                scale = fit_scale
            elif self.image_mode == "actual":
                scale = 1.0
            else:
                scale = max(0.1, self.zoom_factor)
            new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
            resized = img.resize(new_size, _pil_lanczos())
            self._img_ref = ImageTk.PhotoImage(resized)
            self.image_canvas.delete("card_preview")
            self.image_canvas.create_image(
                0,
                0,
                anchor="nw",
                image=self._img_ref,
                tags=("card_preview",),
            )
            self.image_canvas._preview_ref = self._img_ref
            self.image_canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))

            if not self.scrollbars_enabled:
                if self.image_scroll_x is not None:
                    self.image_scroll_x.grid_remove()
                if self.image_scroll_y is not None:
                    self.image_scroll_y.grid_remove()
            elif self.image_mode == "fit":
                if self.image_scroll_x is not None:
                    self.image_scroll_x.grid_remove()
                if self.image_scroll_y is not None:
                    self.image_scroll_y.grid_remove()
            else:
                if self.image_scroll_x is not None:
                    self.image_scroll_x.grid()
                if self.image_scroll_y is not None:
                    self.image_scroll_y.grid()
        except Exception as exc:
            log_image_error(path, exc)
            try:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение.")
            except Exception:
                pass
            self._img_ref = None
            if self.image_canvas is not None:
                self.image_canvas.delete("card_preview")
                self.image_canvas.configure(bg=self.media_bg)

    def show_audio_frame(self, visible: bool) -> None:
        if visible:
            audio_y, _video_y = self._calculate_media_positions(self._last_video_height, audio_height=90)
            self.audio_inline_frame.place(x=10, y=audio_y, width=780, height=90)
        else:
            if self.audio_inline_frame.winfo_ismapped():
                self.audio_inline_frame.place_forget()

    def disable_scrollbars(self) -> None:
        self.scrollbars_enabled = False
        if getattr(self, "content_scrollbar", None) is not None:
            self.content_scrollbar.pack_forget()
        if getattr(self, "content_canvas", None) is not None:
            self.content_canvas.configure(yscrollcommand=None)
        if getattr(self, "image_scroll_x", None) is not None:
            self.image_scroll_x.grid_remove()
        if getattr(self, "image_scroll_y", None) is not None:
            self.image_scroll_y.grid_remove()

    def show_video_frame(self, visible: bool, height: int = 80) -> None:
        if visible:
            self._last_video_height = height
            _audio_y, video_y = self._calculate_media_positions(height, audio_height=90)
            self.video_inline_frame.place(x=10, y=video_y, width=780, height=height)
        else:
            if self.video_inline_frame.winfo_ismapped():
                self.video_inline_frame.place_forget()

    def _calculate_media_positions(self, video_height: int, audio_height: int = 90) -> tuple[int, int]:
        try:
            self.update_idletasks()
        except Exception:
            pass
        current_height = self.winfo_height()
        if not current_height:
            try:
                current_height = int(self.cget("height") or 520)
            except Exception:
                current_height = 520
        video_y = max(10, current_height - video_height - 10)
        audio_y = max(10, video_y - audio_height - 10)
        return audio_y, video_y

    def _update_content_geometry(self) -> None:
        try:
            width = int(self.cget("width") or 700)
        except Exception:
            width = 700
        try:
            height = int(self.cget("height") or 420)
        except Exception:
            height = 420
        width = max(width, self.winfo_width() or width)
        height = max(height, self.winfo_height() or height)
        content_width = max(1, width - 20)
        content_height = max(1, height - 120)
        self.content_container.place(x=10, y=40, width=content_width, height=content_height)
