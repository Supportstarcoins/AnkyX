import tkinter as tk
from tkinter import ttk


PALETTE = {
    "background": "#0B0F14",
    "panel": "#111823",
    "border": "#1B2430",
    "text": "#E5E7EB",
    "muted": "#9CA3AF",
    "accent": "#3B82F6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "error": "#EF4444",
}


def apply_premium_dark_theme(root: tk.Tk) -> tuple[ttk.Style, dict]:
    """Настроить тёмную тему. Использует ttkbootstrap при наличии."""

    try:
        import ttkbootstrap as tb  # type: ignore

        style: ttk.Style = tb.Style(theme="darkly")
    except Exception:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    root.configure(bg=PALETTE["background"])

    root.option_add("*Font", "Segoe UI 11")
    root.option_add("*background", PALETTE["background"])
    root.option_add("*foreground", PALETTE["text"])
    root.option_add("*Entry*foreground", PALETTE["text"])
    root.option_add("*Entry*background", PALETTE["panel"])
    root.option_add("*Listbox*background", PALETTE["panel"])
    root.option_add("*Listbox*foreground", PALETTE["text"])
    root.option_add("*Menu*background", PALETTE["panel"])
    root.option_add("*Menu*foreground", PALETTE["text"])
    root.option_add("*Text*background", PALETTE["panel"])
    root.option_add("*Text*foreground", PALETTE["text"])
    root.option_add("*Text*insertBackground", PALETTE["text"])

    style.configure(
        "TFrame",
        background=PALETTE["background"],
    )
    style.configure(
        "Surface.TFrame",
        background=PALETTE["background"],
    )
    style.configure(
        "Card.TFrame",
        background=PALETTE["panel"],
        relief="flat",
        borderwidth=1,
    )
    style.configure(
        "CardInner.TFrame",
        background=PALETTE["panel"],
        relief="flat",
        borderwidth=0,
    )
    style.configure(
        "Header.TFrame",
        background=PALETTE["background"],
    )

    for label_style, font, color in [
        ("HeaderTitle.TLabel", ("Segoe UI", 15, "bold"), PALETTE["text"]),
        ("HeaderSub.TLabel", ("Segoe UI", 11), PALETTE["muted"]),
        ("Heading.TLabel", ("Segoe UI", 14, "bold"), PALETTE["text"]),
        ("Body.TLabel", ("Segoe UI", 11), PALETTE["text"]),
        ("Muted.TLabel", ("Segoe UI", 10), PALETTE["muted"]),
        ("Badge.TLabel", ("Segoe UI", 10, "bold"), PALETTE["accent"]),
    ]:
        style.configure(label_style, font=font, background=PALETTE["panel"], foreground=color)

    style.configure("TLabel", background=PALETTE["background"], foreground=PALETTE["text"])
    style.configure(
        "TButton",
        padding=(12, 8),
        borderwidth=1,
        relief="flat",
        bordercolor=PALETTE["border"],
        background=PALETTE["panel"],
        foreground=PALETTE["text"],
    )
    style.map(
        "TButton",
        background=[("active", "#162132"), ("pressed", "#0f172a"), ("focus", "#162132")],
        foreground=[("disabled", PALETTE["muted"])],
        bordercolor=[("focus", PALETTE["accent"]), ("active", PALETTE["accent"])],
    )
    style.configure(
        "Primary.TButton",
        background=PALETTE["accent"],
        foreground=PALETTE["text"],
        bordercolor=PALETTE["accent"],
        borderwidth=1,
        padding=(14, 10),
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#1d4ed8"), ("pressed", "#1e3a8a"), ("focus", "#1d4ed8")],
        foreground=[("disabled", PALETTE["muted"])],
        bordercolor=[("focus", "#2563eb"), ("active", "#2563eb")],
    )
    style.configure(
        "Ghost.TButton",
        background=PALETTE["panel"],
        foreground=PALETTE["text"],
        borderwidth=1,
        bordercolor=PALETTE["border"],
        padding=(12, 10),
    )
    style.map(
        "Ghost.TButton",
        background=[("active", "#162132"), ("pressed", "#0f172a"), ("focus", "#162132")],
        bordercolor=[("focus", PALETTE["accent"]), ("active", PALETTE["accent"])],
    )

    style.configure(
        "TEntry",
        fieldbackground=PALETTE["panel"],
        foreground=PALETTE["text"],
        bordercolor=PALETTE["border"],
        lightcolor=PALETTE["accent"],
        darkcolor=PALETTE["border"],
        insertcolor=PALETTE["text"],
        padding=8,
    )
    style.configure(
        "TCombobox",
        fieldbackground=PALETTE["panel"],
        foreground=PALETTE["text"],
        bordercolor=PALETTE["border"],
        padding=6,
    )
    style.configure(
        "Treeview",
        background=PALETTE["panel"],
        fieldbackground=PALETTE["panel"],
        foreground=PALETTE["text"],
        bordercolor=PALETTE["border"],
        lightcolor=PALETTE["border"],
        darkcolor=PALETTE["border"],
        borderwidth=1,
        rowheight=26,
    )
    style.map(
        "Treeview",
        background=[("selected", "#1f2a44")],
        foreground=[("selected", PALETTE["text"])],
    )

    style.configure(
        "TNotebook",
        background=PALETTE["background"],
        bordercolor=PALETTE["border"],
        tabmargins=4,
    )
    style.configure(
        "TNotebook.Tab",
        padding=9,
        background=PALETTE["panel"],
        foreground=PALETTE["text"],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", "#162132")],
        foreground=[("selected", PALETTE["text"])],
    )

    style.configure("Horizontal.TScrollbar", gripcount=0, background=PALETTE["panel"], troughcolor=PALETTE["border"])
    style.configure("Vertical.TScrollbar", gripcount=0, background=PALETTE["panel"], troughcolor=PALETTE["border"])

    style.configure(
        "TLabelframe",
        background=PALETTE["panel"],
        bordercolor=PALETTE["border"],
        relief="solid",
        labeloutside=False,
        padding=12,
    )
    style.configure(
        "Card.TLabelframe",
        background=PALETTE["panel"],
        bordercolor=PALETTE["border"],
        relief="flat",
        padding=12,
        labeloutside=False,
    )
    style.configure(
        "TLabelframe.Label",
        background=PALETTE["panel"],
        foreground=PALETTE["muted"],
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=PALETTE["panel"],
        foreground=PALETTE["muted"],
    )

    return style, PALETTE


def style_text_widget(widget: tk.Text, palette: dict | None = None) -> None:
    """Применить тёмную палитру к Text."""

    colors = palette or PALETTE
    widget.configure(
        bg=colors["panel"],
        fg=colors["text"],
        insertbackground=colors["text"],
        highlightthickness=1,
        highlightbackground=colors["border"],
        relief="flat",
    )


def style_card(widget: tk.Widget, palette: dict | None = None, padded: bool = False) -> None:
    colors = palette or PALETTE
    try:
        widget.configure(bg=colors["panel"], highlightthickness=1, highlightbackground=colors["border"])
    except tk.TclError:
        try:
            widget.configure(style="Card.TFrame")
        except tk.TclError:
            pass
    if padded and isinstance(widget, (ttk.Frame, ttk.LabelFrame)):
        widget.configure(padding=12)
