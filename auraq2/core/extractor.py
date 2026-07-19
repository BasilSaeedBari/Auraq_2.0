"""
Auraq 2.0 — PDF Extractor
Clips question regions from source PDFs using PyMuPDF's show_pdf_page.
Preserves vector graphics and mathematical notation.

Key principle: regions use text_end_y as the bottom boundary, so blank
working/answer space below the question text is NOT included.
"""
from __future__ import annotations

import fitz  # PyMuPDF
from auraq2.utils.logging import get_logger

logger = get_logger()


def insert_regions_into_pdf(
    dest_doc: fitz.Document,
    src_doc: fitz.Document,
    regions: list[dict],
    fallback_pages: tuple[int, int] | None = None,
    label: str = "",
) -> int:
    """
    Insert clipped page regions from src_doc into dest_doc.

    Each region is a dict: {"page": int, "rect": [x0, y0, x1, y1]}

    If regions is empty and fallback_pages is provided, insert full pages instead.

    When *label* is given (e.g. "9709_w24_qp_11 Q3"), a small grey stamp is
    printed at the bottom-left of every inserted page for source traceability.

    Returns the number of pages added to dest_doc.
    """
    added = 0

    def _stamp(page: fitz.Page) -> None:
        if not label:
            return
        h = page.rect.height
        w = page.rect.width
        stamp_rect = fitz.Rect(6, h - 14, min(w - 6, 6 + len(label) * 4.2), h - 4)
        if stamp_rect.width <= 0 or stamp_rect.height <= 0:
            logger.warning(f"Invalid stamp_rect dimension: {stamp_rect}")
            return
        try:
            page.insert_textbox(
                stamp_rect,
                label,
                fontsize=5.5,
                fontname="helv",
                color=(0.55, 0.55, 0.55),
                align=fitz.TEXT_ALIGN_LEFT,
            )
        except Exception as e:
            logger.warning(f"Failed to insert stamp textbox: {e}")

    if regions:
        for reg in regions:
            p_idx = reg["page"]
            rect_val = reg["rect"]
            if not isinstance(rect_val, list) or len(rect_val) != 4:
                logger.warning(f"Invalid rect format on page {p_idx}: {rect_val}")
                continue
            x0, y0, x1, y1 = rect_val

            if not all(isinstance(v, (int, float)) for v in (x0, y0, x1, y1)):
                logger.warning(f"Invalid non-numeric rect values on page {p_idx}: {rect_val}")
                continue

            if x1 <= x0 or y1 <= y0:
                logger.debug(f"Skipping degenerate region on page {p_idx}: {rect_val}")
                continue

            src_page = src_doc[p_idx]
            clip = fitz.Rect(x0, y0, x1, y1)

            # Create destination page with the same dimensions as the clip
            # (compact output — no wasted whitespace)
            dest_page = dest_doc.new_page(
                width=clip.width,
                height=clip.height,
            )
            dest_page.show_pdf_page(
                fitz.Rect(0, 0, clip.width, clip.height),
                src_doc,
                p_idx,
                clip=clip,
            )
            _stamp(dest_page)
            added += 1

    elif fallback_pages:
        start, end = fallback_pages
        for p_idx in range(start, end + 1):
            dest_doc.insert_pdf(src_doc, from_page=p_idx, to_page=p_idx)
            # stamp the page we just inserted (it is now the last page)
            _stamp(dest_doc[-1])
            added += 1

    return added


def extract_question(
    qp_doc: fitz.Document,
    question: dict,
) -> list[dict]:
    """
    Extract the region data for a single question from the QP.

    Returns the list of region dicts (already stored in the registry).
    This is a pass-through helper that validates and returns regions.
    """
    regions = question.get("regions", [])
    if not regions:
        logger.warning(f"Q{question['q_num']}: no regions in registry, will use full page fallback.")
    return regions


def build_question_pdf(
    src_doc: fitz.Document,
    question: dict,
    dest_doc: fitz.Document | None = None,
) -> fitz.Document:
    """
    Build a standalone PDF containing just the extracted question region(s).
    If dest_doc is None, a new fitz.Document is created.
    """
    if dest_doc is None:
        dest_doc = fitz.open()

    regions = question.get("regions", [])
    fallback = None
    if not regions:
        sp = question.get("start_page")
        ep = question.get("end_page")
        if sp is not None and ep is not None:
            fallback = (sp, ep)

    insert_regions_into_pdf(dest_doc, src_doc, regions, fallback)
    return dest_doc
