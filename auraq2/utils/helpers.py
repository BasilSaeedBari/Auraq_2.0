"""
Auraq 2.0 — Helper Utilities
Path builders, coordinate transforms, and small utilities.
"""
from __future__ import annotations

import os
import fitz  # PyMuPDF

# --------------------------------------------------------------------------- #
# Session maps                                                                  #
# --------------------------------------------------------------------------- #
SESSION_DIR: dict[str, str] = {
    "May/June":   "May_June",
    "Oct/Nov":    "Oct_Nov",
    "Feb/March":  "Feb_March",
    "January":    "January",
}

SESSION_SHORT: dict[str, str] = {
    "May/June":   "s",
    "Oct/Nov":    "w",
    "Feb/March":  "m",
    "January":    "j",
}


# --------------------------------------------------------------------------- #
# Structured path builders                                                      #
# --------------------------------------------------------------------------- #
def get_local_path(base_dir: str, spec: dict) -> str:
    """
    Return the absolute local file path for a download spec.

    Structure:
        {base_dir}/{subject}/{year}/{Session}/Paper_{paper}/{QP|MS}/{filename}.pdf

    Example:
        downloads/9709/2025/Oct_Nov/Paper_1/QP/9709_w25_qp_11.pdf
    """
    subject       = spec["subject"]
    year          = str(spec["year"])
    session_dir   = SESSION_DIR[spec["session"]]
    paper_dir     = f"Paper_{spec['paper']}"
    doc_type      = spec["doc_type"].upper()          # "QP" or "MS"
    session_short = SESSION_SHORT[spec["session"]]
    year_short    = str(spec["year"])[-2:]
    variant       = spec["variant"]

    filename = f"{subject}_{session_short}{year_short}_{spec['doc_type'].lower()}_{variant}.pdf"

    return os.path.join(base_dir, subject, year, session_dir, paper_dir, doc_type, filename)


def get_registry_path(base_dir: str, spec: dict) -> str:
    """
    Return the path to the registry JSON file that lives next to the PDF.
    E.g.: downloads/9709/2025/Oct_Nov/Paper_1/QP/9709_w25_qp_11_registry.json
    """
    pdf_path = get_local_path(base_dir, spec)
    return pdf_path.replace(".pdf", "_registry.json")


def paper_id_from_spec(spec: dict) -> str:
    """Return a canonical paper ID string, e.g. '9709_w25_qp_11'."""
    session_short = SESSION_SHORT[spec["session"]]
    year_short    = str(spec["year"])[-2:]
    return f"{spec['subject']}_{session_short}{year_short}_{spec['doc_type'].lower()}_{spec['variant']}"


# --------------------------------------------------------------------------- #
# Coordinate transforms (PyMuPDF page rotation)                                 #
# --------------------------------------------------------------------------- #
def get_visual_coords(
    x0: float, y0: float, x1: float, y1: float, page: fitz.Page
) -> tuple[float, float, float, float]:
    """
    Transform PyMuPDF text-block coordinates to *visual* (display) coordinates,
    accounting for page rotation (0, 90, 180, 270 degrees).
    """
    rot = page.rotation
    W = page.rect.x1
    H = page.rect.y1

    if rot == 90:
        return W - y1, x0, W - y0, x1
    if rot == 270:
        return y0, H - x1, y1, H - x0
    if rot == 180:
        return W - x1, H - y1, W - x0, H - y0
    return x0, y0, x1, y1


def visual_to_standard(
    vx0: float, vy0: float, vx1: float, vy1: float, page: fitz.Page
) -> tuple[float, float, float, float]:
    """
    Inverse of get_visual_coords — translate visual bounding box back to
    standard PyMuPDF coordinates.
    """
    rot = page.rotation
    W = page.rect.x1
    H = page.rect.y1

    if rot == 90:
        return vy0, W - vx1, vy1, W - vx0
    if rot == 270:
        return H - vy1, vx0, H - vy0, vx1
    if rot == 180:
        return W - vx1, H - vy1, W - vx0, H - vy0
    return vx0, vy0, vx1, vy1
