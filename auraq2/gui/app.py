"""
Auraq 2.0 — Main GUI Application
Two-column layout (config | options) + bottom console.
New in v2:
  - Sources priority input visible in main UI
  - AI mode toggle (Hybrid / Batch / Heuristics)
  - Groq API key in Preferences
  - "Open Output Folder" button
"""
from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from auraq2.utils.logging import get_logger
from auraq2.utils.config import load_config, save_config
from auraq2.core.subjects_registry import (
    get_curricula, get_subjects, get_papers, get_subject_details,
)
from auraq2.gui.widgets import (
    COLOR_BG, COLOR_CARD, COLOR_TEXT, COLOR_ACCENT,
    COLOR_TEXT_HIGHLIGHT, COLOR_WHITE,
    StyledCard, StyledButton, setup_ttk_styles,
)
from auraq2.gui.callbacks import GuiLogHandler, PipelineThread

logger = get_logger()


# ── Scrollable log console ────────────────────────────────────────────────────
class ScrollableLogBox(tk.Frame):
    """Color-coded scrollable terminal widget."""

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, bg=COLOR_CARD, **kwargs)
        self.text = tk.Text(
            self,
            bg="#150d24",
            fg=COLOR_TEXT,
            insertbackground=COLOR_WHITE,
            relief="flat", bd=0,
            padx=10, pady=8,
            font=("Consolas", 10),
            state="disabled",
            height=9,
        )
        self.scrollbar = tk.Scrollbar(
            self, command=self.text.yview,
            troughcolor=COLOR_CARD, bg=COLOR_ACCENT,
        )
        self.text.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        self.text.tag_config("INFO",    foreground=COLOR_TEXT)
        self.text.tag_config("WARNING", foreground="#fbbf24")
        self.text.tag_config("ERROR",   foreground="#f87171")
        self.text.tag_config("DEBUG",   foreground="#9d8fb0")

    def insert_log(self, message: str, level: str = "INFO") -> None:
        def _action():
            self.text.configure(state="normal")
            self.text.insert("end", message + "\n", level)
            self.text.configure(state="disabled")
            self.text.see("end")
        self.after(0, _action)

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


# ── Preferences window ────────────────────────────────────────────────────────
class PreferencesWindow(tk.Toplevel):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=COLOR_BG)
        self.title("Preferences — Auraq 2.0")
        self.geometry("560x450")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self._config = load_config()
        self._build()

    def _build(self) -> None:
        frm = tk.Frame(self, bg=COLOR_BG, padx=24, pady=22)
        frm.pack(fill="both", expand=True)

        def _lbl(text, row):
            tk.Label(frm, text=text, bg=COLOR_BG, fg=COLOR_WHITE,
                     font=("Segoe UI", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(0, 3))

        def _entry(row, var, show=""):
            e = tk.Entry(frm, textvariable=var, bg=COLOR_CARD, fg=COLOR_WHITE,
                         show=show, bd=0, relief="flat", font=("Segoe UI", 10))
            e.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 14), ipady=4)
            return e

        frm.columnconfigure(0, weight=1)

        _lbl("Default Download Directory:", 0)
        self._dir_var = tk.StringVar(value=self._config.get("General", "download_directory", fallback=""))
        dir_row = tk.Frame(frm, bg=COLOR_BG)
        dir_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        tk.Entry(dir_row, textvariable=self._dir_var, bg=COLOR_CARD, fg=COLOR_WHITE,
                 bd=0, relief="flat", font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True, ipady=4)
        StyledButton(dir_row, text="Browse", command=self._browse_dir, is_primary=False).pack(side="right", padx=(8, 0))

        _lbl("Groq API Key:", 2)
        self._groq_var = tk.StringVar(value=self._config.get("General", "groq_api_key", fallback=""))
        groq_entry = tk.Entry(frm, textvariable=self._groq_var, bg=COLOR_CARD, fg=COLOR_WHITE,
                               show="*", bd=0, relief="flat", font=("Segoe UI", 10))
        groq_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 14), ipady=4)
        self._show_key = False

        def _toggle():
            self._show_key = not self._show_key
            groq_entry.configure(show="" if self._show_key else "*")

        StyledButton(frm, text="Show / Hide Key", command=_toggle, is_primary=False).grid(
            row=4, column=0, sticky="w", pady=(0, 14)
        )

        _lbl("Groq Model:", 5)
        self._model_var = tk.StringVar(value=self._config.get("General", "groq_model", fallback="llama-3.3-70b-versatile"))
        model_options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
            "qwen/qwen3.6-27b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]
        model_cb = ttk.Combobox(frm, textvariable=self._model_var, values=model_options, state="readonly", font=("Segoe UI", 10))
        model_cb.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        _lbl("Sources Priority (comma-separated):", 7)
        self._sources_var = tk.StringVar(value=self._config.get("General", "sources_order", fallback=""))
        _entry(8, self._sources_var)

        btn_row = tk.Frame(frm, bg=COLOR_BG)
        btn_row.grid(row=9, column=0, columnspan=2, sticky="e")
        StyledButton(btn_row, text="Cancel",          command=self.destroy,      is_primary=False).pack(side="left", padx=8)
        StyledButton(btn_row, text="Save Preferences", command=self._save, is_primary=True).pack(side="right")

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self._dir_var.set(d)

    def _save(self) -> None:
        cfg = load_config()
        save_config(
            download_dir=self._dir_var.get(),
            sources=self._sources_var.get(),
            groq_api_key=self._groq_var.get(),
            groq_model=self._model_var.get(),
            remove_blank=cfg.getboolean("Filters", "remove_blank", fallback=True),
            remove_additional=cfg.getboolean("Filters", "remove_additional", fallback=True),
            remove_formula=cfg.getboolean("Filters", "remove_formula", fallback=False),
        )
        messagebox.showinfo("Saved", "Preferences saved successfully.")
        self.destroy()


# ── Main App ─────────────────────────────────────────────────────────────────
class AuraqApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Auraq 2.0 — Topical Past Paper Compiler")
        self.root.geometry("920x740")
        self.root.minsize(820, 660)
        self.root.configure(bg=COLOR_BG)

        setup_ttk_styles()
        self._config = load_config()
        self.pipeline_thread: PipelineThread | None = None
        self._last_output_dir: str | None = None

        self._build_ui()
        self._bind_events()
        self._load_initial()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Title
        tk.Label(
            self.root, text="AURAQ 2.0  —  PAST PAPER COMPILER",
            fg=COLOR_TEXT_HIGHLIGHT, bg=COLOR_BG,
            font=("Segoe UI", 17, "bold"),
        ).pack(pady=(14, 8))

        # Two-column body
        body = tk.Frame(self.root, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 4))

        left  = StyledCard(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = StyledCard(body)
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        self._build_left(left)
        self._build_right(right)

        # Bottom: actions + console
        bottom = StyledCard(self.root)
        bottom.pack(fill="x", padx=18, pady=(0, 14))
        self._build_bottom(bottom)

    def _build_left(self, frm: tk.Frame) -> None:
        """Curriculum / subject / paper / variants configuration."""
        tk.Label(frm, text="CURRICULUM CONFIGURATION",
                 fg=COLOR_TEXT_HIGHLIGHT, bg=COLOR_CARD,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))

        def _lbl(text):
            tk.Label(frm, text=text, fg=COLOR_TEXT, bg=COLOR_CARD,
                     font=("Segoe UI", 10)).pack(anchor="w", pady=(6, 2))

        # Curriculum
        _lbl("Curriculum:")
        self._curr_var = tk.StringVar()
        self._curr_cb  = ttk.Combobox(frm, textvariable=self._curr_var, state="readonly",
                                       font=("Segoe UI", 10))
        self._curr_cb.pack(fill="x", pady=(0, 8))

        # Subject
        _lbl("Subject Code:")
        self._sub_var = tk.StringVar()
        self._sub_cb  = ttk.Combobox(frm, textvariable=self._sub_var, state="readonly",
                                      font=("Segoe UI", 10))
        self._sub_cb.pack(fill="x", pady=(0, 8))

        # Paper
        _lbl("Paper Component:")
        self._paper_var = tk.StringVar()
        self._paper_cb  = ttk.Combobox(frm, textvariable=self._paper_var, state="readonly",
                                        font=("Segoe UI", 10))
        self._paper_cb.pack(fill="x", pady=(0, 10))

        # Variants
        _lbl("Variants (multi-select):")
        self._variants_frame = tk.Frame(frm, bg=COLOR_CARD)
        self._variants_frame.pack(fill="both", expand=True)
        self._variant_vars: dict[str, tk.BooleanVar] = {}

    def _build_right(self, frm: tk.Frame) -> None:
        """Pipeline options, filters, output."""
        tk.Label(frm, text="PIPELINE OPTIONS",
                 fg=COLOR_TEXT_HIGHLIGHT, bg=COLOR_CARD,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))

        def _lbl(text, bold=False):
            tk.Label(frm, text=text, fg=COLOR_TEXT, bg=COLOR_CARD,
                     font=("Segoe UI", 10, "bold" if bold else "normal")).pack(anchor="w", pady=(6, 2))

        # Series checkboxes
        _lbl("Exam Series:")
        series_frm = tk.Frame(frm, bg=COLOR_CARD)
        series_frm.pack(anchor="w", pady=(0, 10))
        self._series_vars: dict[str, tk.BooleanVar] = {
            "May/June":  tk.BooleanVar(value=True),
            "Oct/Nov":   tk.BooleanVar(value=True),
            "Feb/March": tk.BooleanVar(value=False),
            "January":   tk.BooleanVar(value=False),
        }
        for i, (name, var) in enumerate(self._series_vars.items()):
            col = i % 2
            row = i // 2
            tk.Checkbutton(series_frm, text=name, variable=var,
                           bg=COLOR_CARD, fg=COLOR_TEXT, activebackground=COLOR_CARD,
                           activeforeground=COLOR_WHITE, selectcolor=COLOR_BG,
                           font=("Segoe UI", 10)).grid(row=row, column=col, sticky="w", padx=(0, 16))

        # Year range
        yr_frm = tk.Frame(frm, bg=COLOR_CARD)
        yr_frm.pack(fill="x", pady=(0, 12))
        for col, (label, attr, default) in enumerate([
            ("Start Year", "_start_spin", 2020),
            ("End Year",   "_end_spin",   2025),
        ]):
            tk.Label(yr_frm, text=label, fg=COLOR_TEXT, bg=COLOR_CARD,
                     font=("Segoe UI", 10)).grid(row=0, column=col*2, sticky="w", padx=(0, 8))
            sp = tk.Spinbox(yr_frm, from_=2000, to=2035, width=7,
                            bg=COLOR_CARD, fg=COLOR_WHITE, bd=0,
                            buttonbackground=COLOR_BG, relief="flat",
                            font=("Segoe UI", 10))
            sp.delete(0, "end"); sp.insert(0, default)
            sp.grid(row=1, column=col*2, sticky="w", padx=(0, 20))
            setattr(self, attr, sp)

        # Page filters
        _lbl("Page Filters:")
        self._rm_blank_var = tk.BooleanVar(value=self._config.getboolean("Filters", "remove_blank", fallback=True))
        self._rm_add_var   = tk.BooleanVar(value=self._config.getboolean("Filters", "remove_additional", fallback=True))
        self._rm_form_var  = tk.BooleanVar(value=self._config.getboolean("Filters", "remove_formula", fallback=False))
        for text, var in [
            ("Remove 'BLANK PAGE' sheets",          self._rm_blank_var),
            ("Remove 'Additional Page' sheets",     self._rm_add_var),
            ("Remove Formula / Data sheets",        self._rm_form_var),
        ]:
            tk.Checkbutton(frm, text=text, variable=var, bg=COLOR_CARD, fg=COLOR_TEXT,
                           activebackground=COLOR_CARD, activeforeground=COLOR_WHITE,
                           selectcolor=COLOR_BG, font=("Segoe UI", 10)).pack(anchor="w", pady=1)

        # AI mode
        _lbl("Classification Mode:", bold=True)
        self._ai_mode_var = tk.StringVar(value="hybrid")
        ai_frm = tk.Frame(frm, bg=COLOR_CARD)
        ai_frm.pack(anchor="w", pady=(0, 10))
        for label, value in [("Hybrid (AI + Heuristics)", "hybrid"),
                              ("AI Batch Only", "batch"),
                              ("Heuristics Only", "heuristics")]:
            tk.Radiobutton(ai_frm, text=label, variable=self._ai_mode_var, value=value,
                           bg=COLOR_CARD, fg=COLOR_TEXT, activebackground=COLOR_CARD,
                           selectcolor=COLOR_BG, font=("Segoe UI", 10)).pack(anchor="w", pady=1)

        # Output dir
        _lbl("Output Directory:")
        self._out_var = tk.StringVar(value=self._config.get("General", "download_directory", fallback=""))
        out_frm = tk.Frame(frm, bg=COLOR_CARD)
        out_frm.pack(fill="x", pady=(0, 10))
        tk.Entry(out_frm, textvariable=self._out_var, bg=COLOR_BG, fg=COLOR_WHITE,
                 bd=0, relief="flat", font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True, ipady=3)
        StyledButton(out_frm, text="…", command=self._browse_output, is_primary=False,
                     width=3).pack(side="right", padx=(5, 0))

        # Topical toggle
        self._topical_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frm, text="Generate Topical Past Paper Booklets",
                       variable=self._topical_var, bg=COLOR_CARD, fg=COLOR_WHITE,
                       activebackground=COLOR_CARD, activeforeground=COLOR_WHITE,
                       selectcolor=COLOR_BG, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8, 0))

    def _build_bottom(self, frm: tk.Frame) -> None:
        """Action buttons, progress bar, log console."""
        btn_row = tk.Frame(frm, bg=COLOR_CARD)
        btn_row.pack(fill="x", pady=(0, 8))

        self._pref_btn = StyledButton(btn_row, text="Preferences",
                                      command=self._open_prefs, is_primary=False)
        self._pref_btn.pack(side="left")

        self._open_btn = StyledButton(btn_row, text="Open Output Folder",
                                      command=self._open_output, is_primary=False)
        self._open_btn.pack(side="left", padx=(8, 0))
        self._open_btn.configure(state="disabled")

        self._run_btn = StyledButton(btn_row, text="Compile Past Papers  ▶",
                                     command=self._start, is_primary=True)
        self._run_btn.pack(side="right")

        # Progress
        prog_row = tk.Frame(frm, bg=COLOR_CARD)
        prog_row.pack(fill="x", pady=(0, 8))
        self._prog_label = tk.Label(prog_row, text="Ready", fg=COLOR_TEXT, bg=COLOR_CARD,
                                    font=("Segoe UI", 10))
        self._prog_label.pack(side="left")
        self._prog_bar = ttk.Progressbar(prog_row, orient="horizontal", mode="determinate",
                                          style="Horizontal.TProgressbar", length=260)
        self._prog_bar.pack(side="right")

        # Console
        self._log_box = ScrollableLogBox(frm)
        self._log_box.pack(fill="both", expand=True)

        # Attach logger
        self._log_handler = GuiLogHandler(self._log_box)
        self._log_handler.setFormatter(
            __import__("logging").Formatter("%(levelname)s  %(message)s")
        )
        logger.addHandler(self._log_handler)

    # ── Event wiring ─────────────────────────────────────────────────────────
    def _bind_events(self) -> None:
        self._curr_cb.bind("<<ComboboxSelected>>", self._on_curriculum)
        self._sub_cb.bind("<<ComboboxSelected>>",  self._on_subject)

    def _load_initial(self) -> None:
        curricula = get_curricula()
        self._curr_cb.configure(values=curricula)
        if curricula:
            self._curr_cb.current(0)
            self._on_curriculum(None)

    def _on_curriculum(self, _event) -> None:
        curr = self._curr_var.get()
        subs = get_subjects(curr)
        codes = sorted(subs.keys())
        disp  = [f"{c} — {subs[c].get('name', '')}" for c in codes]
        self._sub_cb.configure(values=disp)
        if disp:
            self._sub_cb.current(0)
            self._on_subject(None)

    def _on_subject(self, _event) -> None:
        curr = self._curr_var.get()
        raw  = self._sub_var.get()
        if not raw:
            return
        code = raw.split(" — ")[0].strip()
        details = get_subject_details(curr, code)
        papers_dict = details.get("papers", {})
        pkeys = sorted(papers_dict.keys())
        disp  = [f"{p} — {papers_dict[p].get('name', '')}" for p in pkeys]
        self._paper_cb.configure(values=disp)
        if disp:
            self._paper_cb.current(0)

        # Build variant checkboxes
        for w in self._variants_frame.winfo_children():
            w.destroy()
        self._variant_vars.clear()

        if "Cambridge" in curr:
            for v, label in [("1", "Variant 1 (x1)"), ("2", "Variant 2 (x2)"), ("3", "Variant 3 (x3)")]:
                var = tk.BooleanVar(value=True)
                self._variant_vars[v] = var
                tk.Checkbutton(self._variants_frame, text=label, variable=var,
                               bg=COLOR_CARD, fg=COLOR_TEXT, activebackground=COLOR_CARD,
                               activeforeground=COLOR_WHITE, selectcolor=COLOR_BG,
                               font=("Segoe UI", 10)).pack(anchor="w", pady=2)
        else:
            for v, label, default in [("H", "Higher (H)", True), ("F", "Foundation (F)", False)]:
                var = tk.BooleanVar(value=default)
                self._variant_vars[v] = var
                tk.Checkbutton(self._variants_frame, text=label, variable=var,
                               bg=COLOR_CARD, fg=COLOR_TEXT, activebackground=COLOR_CARD,
                               activeforeground=COLOR_WHITE, selectcolor=COLOR_BG,
                               font=("Segoe UI", 10)).pack(anchor="w", pady=2)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _browse_output(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self._out_var.set(d)

    def _open_prefs(self) -> None:
        PreferencesWindow(self.root)

    def _open_output(self) -> None:
        path = self._last_output_dir or self._out_var.get()
        if path and os.path.exists(path):
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])

    def update_progress(self, stage: str, current: int, total: int) -> None:
        def _action():
            self._prog_bar.configure(maximum=max(total, 1), value=current)
            pct = (current / total * 100) if total > 0 else 0
            self._prog_label.configure(text=f"{stage}: {current}/{total} ({pct:.0f}%)")
        self.root.after(0, _action)

    def _start(self) -> None:
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            messagebox.showwarning("Busy", "A pipeline job is already running.")
            return

        self._log_box.clear()

        # Gather values
        curriculum = self._curr_var.get()
        raw_sub = self._sub_var.get()
        if not raw_sub:
            messagebox.showerror("Error", "Please select a subject."); return
        subject_code = raw_sub.split(" — ")[0].strip()

        raw_paper = self._paper_var.get()
        if not raw_paper:
            messagebox.showerror("Error", "Please select a paper component."); return
        paper_code = raw_paper.split(" — ")[0].strip()

        variants = [k for k, v in self._variant_vars.items() if v.get()]
        if not variants:
            messagebox.showerror("Error", "Select at least one variant."); return

        sessions = [k for k, v in self._series_vars.items() if v.get()]
        if not sessions:
            messagebox.showerror("Error", "Select at least one exam series."); return

        try:
            start_year = int(self._start_spin.get())
            end_year   = int(self._end_spin.get())
        except ValueError:
            messagebox.showerror("Error", "Years must be integers."); return
        if start_year > end_year:
            messagebox.showerror("Error", "Start year cannot exceed end year."); return

        output_dir = self._out_var.get().strip()
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory."); return

        cfg = load_config()
        groq_key = os.environ.get("GROQ_API_KEY") or cfg.get("General", "groq_api_key", fallback="")

        self._last_output_dir = output_dir
        self._run_btn.configure(state="disabled", text="Processing …")
        self._pref_btn.configure(state="disabled")
        self._open_btn.configure(state="disabled")

        self.pipeline_thread = PipelineThread(
            app=self,
            curriculum=curriculum,
            subject_code=subject_code,
            paper=paper_code,
            variants=variants,
            sessions=sessions,
            start_year=start_year,
            end_year=end_year,
            output_dir=output_dir,
            remove_blank=self._rm_blank_var.get(),
            remove_additional=self._rm_add_var.get(),
            remove_formula=self._rm_form_var.get(),
            generate_topical=self._topical_var.get(),
            groq_api_key=groq_key,
            ai_mode=self._ai_mode_var.get(),
        )
        self.pipeline_thread.start()

    def on_pipeline_complete(self, success: bool, error_msg: str | None) -> None:
        self._run_btn.configure(state="normal", text="Compile Past Papers  ▶")
        self._pref_btn.configure(state="normal")
        self._prog_label.configure(text="Done ✅" if success else "Failed ❌")
        if success:
            self._open_btn.configure(state="normal")
            messagebox.showinfo("Success", "Past papers compiled and categorised successfully!")
        else:
            err = f"\n\nError: {error_msg}" if error_msg else ""
            messagebox.showerror("Error", f"An error occurred during compilation.{err}")


def run_gui() -> None:
    root = tk.Tk()
    AuraqApp(root)
    root.mainloop()
