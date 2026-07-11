"""
Auraq 2.0 — GUI Widgets and Styles
Color palette (Purple / Vintage Grape) and base widget classes.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# ── Colour palette ────────────────────────────────────────────────────────────
COLOR_BG            = "#1a0f2e"   # Deep indigo background
COLOR_CARD          = "#231847"   # Card surface
COLOR_ACCENT        = "#783f8e"   # Velvet Orchid
COLOR_TEXT          = "#c9bfd8"   # Thistle muted
COLOR_TEXT_HIGHLIGHT= "#e8d8f5"   # Pale lavender
COLOR_WHITE         = "#f5f0fb"
COLOR_SUCCESS       = "#4ade80"
COLOR_WARNING       = "#fbbf24"
COLOR_ERROR         = "#f87171"


def setup_ttk_styles() -> None:
    """Configure ttk widget styles to match the dark purple palette."""
    style = ttk.Style()
    style.theme_use("clam")

    # Combobox
    style.configure("TCombobox",
        fieldbackground=COLOR_CARD, background=COLOR_CARD,
        foreground=COLOR_WHITE, selectbackground=COLOR_ACCENT,
        selectforeground=COLOR_WHITE, borderwidth=0,
    )
    style.map("TCombobox", fieldbackground=[("readonly", COLOR_CARD)])

    # Progressbar
    style.configure("Horizontal.TProgressbar",
        troughcolor=COLOR_CARD, background=COLOR_ACCENT,
        thickness=6, borderwidth=0,
    )


class StyledCard(tk.Frame):
    """A rounded-look card container (flat dark surface with padding)."""
    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(
            parent,
            bg=COLOR_CARD,
            padx=16, pady=14,
            relief="flat",
            bd=0,
            **kwargs,
        )


class StyledButton(tk.Button):
    """Primary or secondary styled button."""
    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        command=None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        bg = COLOR_ACCENT if is_primary else COLOR_CARD
        fg = COLOR_WHITE
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=COLOR_BG,
            activeforeground=COLOR_WHITE,
            relief="flat",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            font=("Segoe UI", 10, "bold" if is_primary else "normal"),
            **kwargs,
        )
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self._is_primary = is_primary
        self._default_bg = bg

    def _on_enter(self, _event) -> None:
        if str(self["state"]) != "disabled":
            self.configure(bg=COLOR_BG if self._is_primary else COLOR_ACCENT)

    def _on_leave(self, _event) -> None:
        if str(self["state"]) != "disabled":
            self.configure(bg=self._default_bg)
