"""
Auraq 2.0 — Compiler (simplified)
In v2 the compiler's only job is:
  1. Filter unwanted pages (blank, formula, additional) from a single PDF.
  2. Provide the sort key for chronological ordering of filenames.
The heavy lifting (merging across papers) is done by topical_compiler.py.
"""
from __future__ import annotations

import re
import fitz
from auraq2.utils.logging import get_logger

logger = get_logger()

# ── Chronological sort key ────────────────────────────────────────────────────
def parse_sort_key(filename: str) -> tuple:
    """
    Return a sortable tuple (year, series_val, variant_str, original_name).
    series_val: Feb/March=1, May/June=2, Oct/Nov=3, January=4
    """
    fn = filename.lower()

    # CAIE: 9709_s24_qp_12.pdf
    m = re.match(r"^(\d{4})_([msw])(\d{2})_(qp|ms)_(\w+)\.pdf$", fn)
    if m:
        year = int("20" + m.group(3))
        sl   = {"m": 1, "s": 2, "w": 3}.get(m.group(2), 2)
        return (year, sl, m.group(5), filename)

    # Edexcel with date: 4MA1_1H_que_20180525.pdf
    m2 = re.search(r"(?:que|rms|msc)_(\d{4})(\d{2})(\d{2})", fn)
    if m2:
        year, month = int(m2.group(1)), int(m2.group(2))
        sl   = 4 if month == 1 else (2 if month in (5, 6, 8) else 3)
        pm   = re.search(r"_(\d[hfr]{0,2})_", fn)
        return (year, sl, pm.group(1) if pm else "0", filename)

    # Edexcel with name: 1H-May-2019.pdf
    m3 = re.search(r"(\d[hfr]{0,2})-(may|june|jan|january|nov|november)-(\d{4})", fn)
    if m3:
        year = int(m3.group(3))
        ms   = m3.group(2)
        sl   = 4 if "jan" in ms else (2 if ms in ("may", "june") else 3)
        return (year, sl, m3.group(1), filename)

    # Generic fallback
    ym = re.search(r"\b(20\d{2})\b", fn)
    year = int(ym.group(1)) if ym else 2000
    sl   = 4 if "jan" in fn else (1 if "mar" in fn else (3 if any(x in fn for x in ("oct", "nov")) else 2))
    return (year, sl, "0", filename)


# ── Page filters ─────────────────────────────────────────────────────────────
_BLANK_MARKERS    = {"BLANK PAGE", "This page is intentionally left blank"}
_FORMULA_MARKERS: set[str] = {
    # Standard Cambridge headers
    "Mathematical Formulae",
    "Formula List",
    "List of Formulae",
    "MF19",
    "MF10",
    # Physics / science constant sheets
    "Stefan-Boltzmann constant",
    "Important values, constants and standards",
    "The Periodic Table of Elements",
    # Cambridge A-Level Pure Math formula sheet terms
    "Quadratic Equation",
    "Binomial Theorem",
    "Arithmetic series",
    "Geometric series",
    "Identities",
    "Formulae for ΔABC",
    "Maclaurin's Series",
    # Cambridge O-Level / IGCSE formula sheet sections
    "1. ALGEBRA",
    "2. TRIGONOMETRY",
    "3. MENSURATION",
    "4. CALCULUS",
    "Differentiation",
    "Integration",
    "Normal distribution",
    # Combination trigger phrases (partial)
    "sin A",
    "cos A",
    "cosec",
    "nCr",
    "ln x",
}

# Minimum number of formula phrases that must appear together to
# trigger removal via the combination heuristic (no explicit header needed).
_FORMULA_COMBO_THRESHOLD = 4
_ADDITIONAL_MARKERS = {"Additional Page", "Additional Answer Page"}


def _should_remove(
    text: str,
    remove_blank: bool,
    remove_formula: bool,
    remove_additional: bool,
) -> bool:
    if remove_blank and any(m in text for m in _BLANK_MARKERS):
        return True
    if remove_additional and any(m in text for m in _ADDITIONAL_MARKERS):
        return True
    if remove_formula:
        # Direct header/title match — high confidence
        if any(m in text for m in _FORMULA_MARKERS):
            return True
        # Combination heuristic — if enough formula phrases co-occur on
        # the same page it is almost certainly a formula/data sheet even
        # without a recognisable title (e.g. mid-booklet inserts).
        if sum(1 for m in _FORMULA_MARKERS if m in text) >= _FORMULA_COMBO_THRESHOLD:
            return True
    return False


def filter_pdf(
    pdf_path: str,
    remove_blank: bool = True,
    remove_formula: bool = False,
    remove_additional: bool = True,
) -> fitz.Document:
    """
    Open *pdf_path*, remove unwanted pages, and return the in-memory Document.
    Caller is responsible for closing the returned document.
    """
    doc = fitz.open(pdf_path)
    to_remove = [
        page.number
        for page in doc
        if _should_remove(page.get_text(), remove_blank, remove_formula, remove_additional)
    ]
    if to_remove:
        to_remove.sort(reverse=True)
        doc.delete_pages(to_remove)
        logger.debug(f"Removed {len(to_remove)} filtered page(s) from {pdf_path}")
    return doc
