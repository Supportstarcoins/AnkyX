import tkinter as tk
from tkinter import ttk


PALETTE = {
    "background": "#0B0D12",
    "bg": "#0B0D12",
    "panel": "#111522",
    "panel2": "#0F1320",
    "border": "#242A3A",
    "text": "#E8ECF4",
    "muted": "#A7B0C0",
    "accent": "#3B82F6",
    "accent_hover": "#2F65DB",
    "accent_active": "#234DAF",
    "accent_soft": "#1D2D4A",
    "success": "#2EE59D",
    "warning": "#F59E0B",
    "error": "#FF4D4D",
    "card_surface": "#FFFFFF",
    "card_text": "#111111",
    "card_border": "#E0E0E0",
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

    root.option_add("*Font", "Segoe UI 12")
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
        "CardSurface.TFrame",
        background=PALETTE["card_surface"],
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

    for label_style, font, color, bg_key in [
        ("Title.TLabel", ("Segoe UI", 16, "semibold"), PALETTE["text"], "background"),
        ("HeaderTitle.TLabel", ("Segoe UI", 16, "semibold"), PALETTE["text"], "background"),
        ("HeaderSub.TLabel", ("Segoe UI", 12), PALETTE["muted"], "background"),
        ("Section.TLabel", ("Segoe UI", 14, "semibold"), PALETTE["text"], "panel"),
        ("Heading.TLabel", ("Segoe UI", 14, "semibold"), PALETTE["text"], "panel"),
        ("Body.TLabel", ("Segoe UI", 12), PALETTE["text"], "panel"),
        ("Muted.TLabel", ("Segoe UI", 12), PALETTE["muted"], "panel"),
        ("Badge.TLabel", ("Segoe UI", 11, "semibold"), PALETTE["accent"], "panel"),
    ]:
        style.configure(label_style, font=font, background=PALETTE.get(bg_key, PALETTE["panel"]), foreground=color)

    style.configure("TLabel", background=PALETTE["background"], foreground=PALETTE["text"])
    style.configure(
        "TButton",
        padding=(14, 10),
        borderwidth=1,
        relief="flat",
        bordercolor=PALETTE["border"],
        background=PALETTE["panel2"],
        foreground=PALETTE["text"],
    )
    style.map(
        "TButton",
        background=[("active", PALETTE["panel"]), ("pressed", PALETTE["accent_soft"]), ("focus", PALETTE["panel"])],
        foreground=[("disabled", PALETTE["muted"])],
        bordercolor=[("focus", PALETTE["accent"]), ("active", PALETTE["accent"])],
    )
    style.configure(
        "Secondary.TButton",
        background=PALETTE["panel"],
        foreground=PALETTE["text"],
        bordercolor=PALETTE["border"],
        borderwidth=1,
        padding=(14, 10),
    )
    style.map(
        "Secondary.TButton",
        background=[("active", PALETTE["panel2"]), ("pressed", PALETTE["accent_soft"]), ("focus", PALETTE["panel2"])],
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
        background=[("active", PALETTE["accent_hover"]), ("pressed", PALETTE["accent_active"]), ("focus", PALETTE["accent_hover"])],
        foreground=[("disabled", PALETTE["muted"])],
        bordercolor=[("focus", PALETTE["accent_hover"]), ("active", PALETTE["accent_hover"])],
    )
    style.configure(
        "Ghost.TButton",
        background=PALETTE["background"],
        foreground=PALETTE["text"],
        borderwidth=1,
        bordercolor=PALETTE["border"],
        padding=(12, 10),
    )
    style.map(
        "Ghost.TButton",
        background=[("active", PALETTE["panel2"]), ("pressed", PALETTE["accent_soft"]), ("focus", PALETTE["panel2"])],
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
        background=PALETTE["panel2"],
        fieldbackground=PALETTE["panel2"],
        foreground=PALETTE["text"],
        bordercolor=PALETTE["border"],
        lightcolor=PALETTE["border"],
        darkcolor=PALETTE["border"],
        borderwidth=1,
        rowheight=28,
        font=("Segoe UI", 12),
    )
    style.map(
        "Treeview",
        background=[("selected", PALETTE["accent_soft"])],
        foreground=[("selected", PALETTE["text"])],
    )
    style.configure(
        "Treeview.Heading",
        background=PALETTE["panel"],
        foreground=PALETTE["text"],
        bordercolor=PALETTE["border"],
        font=("Segoe UI", 12, "semibold"),
        relief="flat",
        padding=(8, 6),
    )
    style.map(
        "Treeview.Heading",
        background=[("active", PALETTE["panel2"]), ("pressed", PALETTE["accent_soft"])],
        foreground=[("active", PALETTE["text"]), ("pressed", PALETTE["text"])],
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
        padding=14,
    )
    style.configure(
        "Card.TLabelframe",
        background=PALETTE["panel"],
        bordercolor=PALETTE["border"],
        relief="flat",
        padding=14,
        labeloutside=False,
    )
    style.configure(
        "CardSurface.TLabelframe",
        background=PALETTE["card_surface"],
        bordercolor=PALETTE["card_border"],
        relief="solid",
        padding=14,
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
    style.configure(
        "CardSurface.TLabelframe.Label",
        background=PALETTE["card_surface"],
        foreground=PALETTE["card_text"],
    )
    style.configure(
        "CardSurface.TLabel",
        background=PALETTE["card_surface"],
        foreground=PALETTE["card_text"],
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
        widget.configure(bg=colors["panel"], highlightthickness=1, highlightbackground=colors["border"], relief="flat")
    except tk.TclError:
        try:
            widget.configure(style="Card.TFrame")
        except tk.TclError:
            pass
    if padded and isinstance(widget, (ttk.Frame, ttk.LabelFrame)):
        widget.configure(padding=14)


def style_card_surface(widget: tk.Widget, palette: dict | None = None, padded: bool = False) -> None:
    colors = palette or PALETTE
    try:
        widget.configure(
            bg=colors["card_surface"],
            highlightthickness=1,
            highlightbackground=colors["card_border"],
            relief="flat",
        )
    except tk.TclError:
        try:
            widget.configure(style="CardSurface.TFrame")
        except tk.TclError:
            pass
    if padded and isinstance(widget, (ttk.Frame, ttk.LabelFrame)):
        widget.configure(padding=14)


def style_card_surface_text(widget: tk.Text, palette: dict | None = None) -> None:
    colors = palette or PALETTE
    widget.configure(
        bg=colors["card_surface"],
        fg=colors["card_text"],
        insertbackground=colors["card_text"],
        highlightthickness=1,
        highlightbackground=colors["card_border"],
        relief="flat",
    )
