"""
Auraq 2.0 — Registry Builder
The heart of the system.

Parses a single PDF (QP or MS) and produces a structured JSON registry describing
every question, its sub-parts, their page regions, and — crucially — where the
question TEXT ends (so blank working/answer space is excluded from extractions).

Registry JSON schema (per file):
{
  "paper_id": "9709_w25_qp_11",
  "doc_type": "qp",
  "source_path": "/abs/path/to/9709_w25_qp_11.pdf",
  "pages": [{"page_num": 0, "width": 595.0, "height": 842.0, "rotation": 0}],
  "questions": [
    {
      "q_num": 1,
      "text_snippet": "Find the value of...",
      "start_page": 1,
      "end_page": 1,
      "regions": [{"page": 1, "rect": [0, 118.4, 595.0, 246.7]}],
      "text_end_y": 246.7,          # y1 of last text block + TEXT_PAD
      "sub_parts": [
        {
          "part_id": "a",
          "start_page": 1, "end_page": 1,
          "region": {"page": 1, "rect": [0, 130.0, 595.0, 190.2]},
          "text_end_y": 190.2
        }
      ],
      "topic": null,
      "confidence": null
    }
  ]
}

Key design decisions:
  - text_end_y uses the y1 of the LAST text block (+ padding), NOT the y0 of the
    next question.  This excludes blank working/answer space from extracted PDFs.
  - Sub-parts (a, b, c … and i, ii, iii …) are detected and assigned their own regions.
  - MS papers use the table header y1 as the top boundary per page.
  - All coordinates are in standard (unrotated) PyMuPDF space.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from auraq2.utils.logging import get_logger
from auraq2.utils.helpers import get_visual_coords, visual_to_standard

logger = get_logger()

# ── Configurable constants ───────────────────────────────────────────────────
# Added to the y1 of the last text block to give a small buffer below the text.
TEXT_PAD = 8.0
BREAK_TOLERANCE = 10.0
TRIM_MARGIN = 2.0

# Maximum fraction of page width for a question-number block to occupy from the left.
Q_NUM_X_FRACTION  = 0.18   # question numbers
SUB_PART_X_FRACTION = 0.28  # sub-part labels like (a), (b)

# Regex patterns
_RE_Q_NUM   = re.compile(r"^(?:Question\s+)?(\d{1,2})\b[\.\)]?\s*", re.IGNORECASE)
_RE_SUB_ALPHA = re.compile(r"^\(?([a-z])\)\s*$", re.IGNORECASE)
_RE_SUB_ROMAN = re.compile(r"^\(?(i{1,3}|iv|vi{0,3}|ix)\)\s*$", re.IGNORECASE)
_RE_SUB_LABEL = re.compile(r"^\(?([a-z])\)?")               # looser: starts with (a), (b) …


@dataclass
class _Block:
    """A text block with its page index and y coordinates (standard space)."""
    page: int
    y0: float
    y1: float
    x0: float
    x1: float
    text: str


@dataclass
class _SubPart:
    part_id: str
    start_page: int
    start_y: float
    blocks: list[_Block] = field(default_factory=list)


@dataclass
class _Question:
    q_num: int
    start_page: int
    start_y: float
    blocks: list[_Block] = field(default_factory=list)
    sub_parts: list[_SubPart] = field(default_factory=list)


# ── Utility ──────────────────────────────────────────────────────────────────
def _sorted_blocks(page: fitz.Page) -> list[_Block]:
    """Extract and sort text blocks on a page top→bottom, left→right."""
    raw = page.get_text("blocks")
    out: list[_Block] = []
    
    # Unicode ligatures mapping
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2013": "-",
        "\u2212": "-",
        "\u0000": "(",
        "\u0001": ")",
    }
    
    for b in raw:
        x0, y0, x1, y1, text, *_ = b
        if not text.strip():
            continue
        clean_text = text.strip()
        for k, v in replacements.items():
            clean_text = clean_text.replace(k, v)
        # Strip control characters (ASCII 0-8, 11-31, 127-159) to prevent encoding/regex issues
        clean_text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]', '', clean_text)
        out.append(_Block(page=page.number, y0=y0, y1=y1, x0=x0, x1=x1, text=clean_text))
    out.sort(key=lambda b: (b.y0, b.x0))
    return out


def _last_text_y1(blocks: list[_Block]) -> float:
    """Return the y1 of the last block, or 0 if none."""
    if not blocks:
        return 0.0
    return max(b.y1 for b in blocks)


def _extract_snippet(blocks: list[_Block], page_width: float) -> str:
    """
    Intelligently extract a clean, relevant text snippet for a question.
    Skips diagram/axis annotations (blocks too far right or very short)
    and starts the snippet from the actual question body (stem keywords).
    """
    STEM_KEYWORDS = {
        "find", "show", "verify", "solve", "express", "state", "prove", "derive", 
        "determine", "calculate", "diagram", "curve", "function", "points", "point", 
        "line", "gradient", "tangent", "normal", "circle", "progression", "expansion", 
        "integral", "integrate", "differentiate", "derivative", "given that", "graph"
    }

    # First pass: filter out diagram/plot labels (far-right blocks or very short text)
    candidate_blocks = []
    for b in blocks:
        # Skip blocks too far right only if they are short (likely diagram annotations)
        if b.x0 > page_width * 0.15 and len(b.text.strip()) <= 10:
            continue
        # Skip very short annotations
        if len(b.text.strip()) <= 2:
            continue
        candidate_blocks.append(b)

    # Second pass: find the index where the actual question stem text begins
    stem_start = 0
    for i, b in enumerate(candidate_blocks):
        lower = b.text.lower()
        if any(kw in lower for kw in STEM_KEYWORDS):
            stem_start = i
            break

    # Third pass: build the snippet (from stem_start, keeping substantial blocks before stem_start)
    snippet_blocks = []
    for i, b in enumerate(candidate_blocks):
        if i >= stem_start:
            snippet_blocks.append(b)
        else:
            # Keep longer blocks (exponents, math formulas next to the number) even if sorted before the stem
            if len(b.text.strip()) > 10:
                snippet_blocks.append(b)

    snippet_parts = []
    total_chars = 0
    for b in snippet_blocks:
        text = b.text.strip()
        # Clean answer lines (dots, underscores)
        text = re.sub(r'[._\-=]{4,}', ' ', text)
        # Collapse spaces
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            snippet_parts.append(text)
            total_chars += len(text)
            if total_chars > 1200:
                break

    return " ".join(snippet_parts)[:1200]



def _make_regions(
    doc: fitz.Document,
    start_page: int,
    start_y: float,
    end_page: int,
    end_y: float,
    y_top: float,
    y_bottom_margin: float,
) -> list[dict]:
    """
    Build the list of {page, rect} clipping regions for a question or sub-part.
    Uses the actual text end y (end_y) as the bottom boundary on the last page.
    """
    regions: list[dict] = []

    for p in range(start_page, end_page + 1):
        page = doc[p]
        pw = page.rect.width
        ph = page.rect.height
        bottom_limit = ph - y_bottom_margin

        if p == start_page and p == end_page:
            # Single page
            y0 = start_y
            y1 = min(end_y + TEXT_PAD, bottom_limit)
        elif p == start_page:
            y0 = start_y
            y1 = bottom_limit
        elif p == end_page:
            y0 = y_top
            y1 = min(end_y + TEXT_PAD, bottom_limit)
        else:
            # Intermediate full page
            y0 = y_top
            y1 = bottom_limit

        if y1 > y0:
            regions.append({"page": p, "rect": [0.0, y0, pw, y1]})

    return regions


# ── MS header detection ──────────────────────────────────────────────────────
def _get_ms_header_y1(page: fitz.Page) -> float | None:
    """
    Find the visual y1 of the 'Question | Answer | Marks' table header row.
    Returns None if not found.
    """
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, *_ = b
        if "Question" in text and ("Answer" in text or "Scheme" in text) and "Marks" in text:
            # Convert to visual coords and return vy1
            _, _, _, vy1 = get_visual_coords(x0, y0, x1, y1, page)
            return vy1
    return None


def _get_ms_question_starts(
    doc: fitz.Document,
    start_page: int,
    end_page: int,
    expected_q_nums: list[int],
    y_bottom_margin: float,
) -> list[tuple[int, int, float]]:
    """
    Scan MS pages for question number entries in the table's Question column.
    Returns list of (q_num, page_idx, vy0) sorted chronologically.
    """
    starts: list[tuple[int, int, float]] = []
    seen: set[int] = set()
    cached_header_y1: float | None = None
    cached_header_x0: float | None = None
    header_found_once = False

    # Pass 1: Scan for the header row anywhere in the document
    for p in range(start_page, end_page + 1):
        page = doc[p]
        header_y1 = _get_ms_header_y1(page)
        if header_y1 is not None:
            cached_header_y1 = header_y1
            header_found_once = True
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_ = b
                if "Question" in text and ("Answer" in text or "Scheme" in text) and "Marks" in text:
                    vx0, *_ = get_visual_coords(x0, y0, x1, y1, page)
                    cached_header_x0 = vx0
                    break
            break

    if not header_found_once:
        logger.warning("No 'Question | Answer | Marks' header found in Marking Scheme. Using page-layout fallback.")

    # Pass 2: Process pages
    for p in range(start_page, end_page + 1):
        page = doc[p]
        
        # Check for generic/notes page keywords
        text_all = page.get_text("text").lower()
        generic_keywords = [
            "generic marking principles", 
            "marking principles", 
            "mark scheme notes", 
            "types of mark", 
            "guide to marking",
            "annotations guidance",
            "abbreviations"
        ]
        if any(kw in text_all for kw in generic_keywords):
            if _get_ms_header_y1(page) is None:
                continue

        header_y1 = _get_ms_header_y1(page)
        if header_y1 is not None:
            current_hy1 = header_y1
            current_hx0 = None
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_ = b
                if "Question" in text and ("Answer" in text or "Scheme" in text) and "Marks" in text:
                    vx0, *_ = get_visual_coords(x0, y0, x1, y1, page)
                    current_hx0 = vx0
                    break
            if current_hx0 is None:
                current_hx0 = cached_header_x0 if cached_header_x0 is not None else page.rect.width * 0.05
        else:
            current_hy1 = cached_header_y1 if cached_header_y1 is not None else 80.0
            current_hx0 = cached_header_x0 if cached_header_x0 is not None else page.rect.width * 0.05

        bottom_vis  = page.rect.y1 - y_bottom_margin

        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = b
            vx0, vy0, vx1, vy1 = get_visual_coords(x0, y0, x1, y1, page)

            if vy0 <= current_hy1 or vy0 > bottom_vis:
                continue
            # Must be close to the Question column horizontal position
            col_min = -20.0 if not header_found_once else -10.0
            col_max = 60.0 if not header_found_once else 60.0
            if not (col_min <= (vx0 - current_hx0) <= col_max):
                continue

            clean = text.strip()
            m = re.match(r"^(\d+)", clean)
            if m:
                q_num = int(m.group(1))
                # Skip false-positive mark/score cells (wide blocks with only digits/spaces and no letters or operators)
                has_content = bool(re.search(r"[a-zA-Z()\[\]{}+\-*/=]", clean))
                if not has_content and (x1 - x0) >= 50.0:
                    continue

                if expected_q_nums:
                    next_idx = len(seen)
                    if next_idx < len(expected_q_nums) and q_num == expected_q_nums[next_idx]:
                        seen.add(q_num)
                        starts.append((q_num, p, vy0))
                else:
                    if q_num not in seen:
                        seen.add(q_num)
                        starts.append((q_num, p, vy0))

    starts.sort(key=lambda x: (x[1], x[2]))
    return starts


# ── QP registry builder ──────────────────────────────────────────────────────
def _build_qp_registry(
    doc: fitz.Document,
    source_path: str,
    paper_id: str,
    y_top: float,
    y_bottom: float,
) -> dict:
    """
    Parse a Question Paper PDF and return a registry dict.

    Detection strategy:
      - Question numbers: isolated number token in left 14% of page width.
      - Sub-parts (a, b, c) and sub-sub-parts (i, ii, iii): label in left 28%.
      - text_end_y: y1 of last text block belonging to the question/sub-part.
    """
    page_metas = []
    for pg in doc:
        page_metas.append({
            "page_num": pg.number,
            "width":    pg.rect.width,
            "height":   pg.rect.height,
            "rotation": pg.rotation,
        })

    # --- Phase 1: identify question starts -----------------------------------
    q_starts: list[tuple[int, int, float]] = []   # (q_num, page, y0)
    seen_q: set[int] = set()

    # Dynamically skip cover page (page 0) only if it contains cover-page text indicators.
    # If the cover page was already removed by filters, the loop correctly starts at page 0.
    start_p = 0
    if len(doc) > 0:
        p0_text = doc[0].get_text().lower()
        cover_keywords = [
            "read these instructions",
            "instructions first",
            "dark blue or black pen",
            "do not use staples",
            "write your name",
            "candidate number",
            "index number",
        ]
        if any(kw in p0_text for kw in cover_keywords):
            start_p = 1

    for p_idx in range(start_p, len(doc)):
        page = doc[p_idx]
        pw   = page.rect.width
        ph   = page.rect.height

        blocks = _sorted_blocks(page)
        for blk in blocks:
            # Filter header/footer margins
            if blk.y0 < y_top or blk.y0 > ph - y_bottom:
                continue
            # Must be in the left Q_NUM_X_FRACTION of the page
            if blk.x0 > pw * Q_NUM_X_FRACTION:
                continue

            m = _RE_Q_NUM.match(blk.text)
            if m:
                q_num = int(m.group(1))
                if 1 <= q_num <= 30 and q_num not in seen_q:
                    # Let's keep a very lightweight check for mathematical expressions to prevent false positives:
                    # Only skip if the block is short, has rest text, and contains math operators but no letters.
                    rest = blk.text[m.end():].strip()
                    if len(blk.text) <= 15 and rest:
                        has_letters = bool(re.search(r'[a-zA-Z]', rest))
                        if not has_letters and re.search(r'[+\-*/=<>]', rest):
                            continue

                    seen_q.add(q_num)
                    q_starts.append((q_num, p_idx, blk.y0))

    q_starts.sort(key=lambda x: (x[1], x[2]))

    if not q_starts:
        logger.warning(f"No questions detected in {paper_id}")

    # --- Phase 2: collect all text blocks per question -----------------------
    # For each question, gather every text block between its start y and the
    # start of the next question.
    questions_out: list[dict] = []

    for qi, (q_num, q_page, q_y0) in enumerate(q_starts):
        # Determine end boundary (exclusive start of next question)
        if qi + 1 < len(q_starts):
            _, nxt_page, nxt_y0 = q_starts[qi + 1]
        else:
            nxt_page = len(doc) - 1
            nxt_y0   = doc[nxt_page].rect.height  # full last page

        q_blocks: list[_Block] = []

        for p_idx in range(q_page, nxt_page + 1):
            page   = doc[p_idx]
            ph     = page.rect.height
            pw     = page.rect.width

            for blk in _sorted_blocks(page):
                # Margin filter
                if blk.y0 < y_top or blk.y1 > ph - y_bottom:
                    continue
                # Page-specific upper/lower bounds for this question
                if p_idx == q_page:
                    # Allow a small vertical tolerance (15pt) for blocks to the right of the question number
                    # to catch exponents, fractions, or top parts of formulas.
                    is_right_of_qnum = blk.x0 > pw * Q_NUM_X_FRACTION
                    ymin_limit = q_y0 - 15.0 if is_right_of_qnum else q_y0
                    if blk.y0 < ymin_limit:
                        continue
                if p_idx == nxt_page:
                    if blk.y0 >= nxt_y0:
                        break
                    if blk.y0 >= nxt_y0 - BREAK_TOLERANCE and _RE_Q_NUM.match(blk.text):
                        break
                q_blocks.append(blk)

        # Post-collection trimming to ensure no blocks from the next question are included
        if qi + 1 < len(q_starts):
            _, nxt_page, nxt_y0 = q_starts[qi + 1]
            q_blocks = [
                b for b in q_blocks
                if b.page < nxt_page or (b.page == nxt_page and b.y0 < nxt_y0 - TRIM_MARGIN)
            ]
            # Also remove any block on nxt_page matching _RE_Q_NUM starting at/after nxt_y0 - 15.0
            q_blocks = [
                b for b in q_blocks
                if not (b.page == nxt_page and _RE_Q_NUM.match(b.text) and b.y0 >= nxt_y0 - 15.0)
            ]

        # --- Phase 3: detect sub-parts within this question ------------------
        sub_starts: list[tuple[str, int, float]] = []  # (part_id, page, y0)
        for blk in q_blocks:
            # Must be in wider left fraction for sub-parts
            page = doc[blk.page]
            pw = page.rect.width
            if blk.x0 > pw * SUB_PART_X_FRACTION:
                continue
            # Skip the question number block itself
            if _RE_Q_NUM.match(blk.text) and blk.page == q_page and abs(blk.y0 - q_y0) < 5:
                continue
            m_alpha = _RE_SUB_ALPHA.match(blk.text)
            m_roman = _RE_SUB_ROMAN.match(blk.text)
            if m_alpha:
                sub_starts.append((m_alpha.group(1), blk.page, blk.y0))
            elif m_roman:
                sub_starts.append((m_roman.group(1).lower(), blk.page, blk.y0))
            else:
                # Looser: line starts with (a) … etc. and has other text too
                m_loose = _RE_SUB_LABEL.match(blk.text)
                if m_loose and blk.text[0] == "(" and blk.x0 > pw * 0.05:
                    sub_starts.append((m_loose.group(1), blk.page, blk.y0))

        # Build sub-part entries
        sub_parts_out: list[dict] = []
        for si, (part_id, sp_page, sp_y0) in enumerate(sub_starts):
            if si + 1 < len(sub_starts):
                _, nxt_sp_page, nxt_sp_y0 = sub_starts[si + 1]
            else:
                nxt_sp_page = nxt_page
                nxt_sp_y0   = nxt_y0

            # Collect blocks for this sub-part
            sp_blocks = [
                b for b in q_blocks
                if (b.page > sp_page or (b.page == sp_page and b.y0 >= sp_y0))
                and (b.page < nxt_sp_page or (b.page == nxt_sp_page and b.y0 < nxt_sp_y0 - TRIM_MARGIN))
            ]
            # Remove next sub-part label indicator block
            sp_blocks = [
                b for b in sp_blocks
                if not (b.page == nxt_sp_page and b.y0 >= nxt_sp_y0 - 5.0 and (
                    _RE_SUB_ALPHA.match(b.text) or _RE_SUB_ROMAN.match(b.text) or _RE_SUB_LABEL.match(b.text)
                ))
            ]

            sp_end_page = sp_blocks[-1].page if sp_blocks else sp_page
            sp_last_page_blocks = [b for b in sp_blocks if b.page == sp_end_page]
            sp_text_end = _last_text_y1(sp_last_page_blocks) if sp_last_page_blocks else sp_y0
            
            if nxt_sp_page == sp_end_page:
                sp_text_end = min(sp_text_end, nxt_sp_y0 - TEXT_PAD - 2.0)

            sub_parts_out.append({
                "part_id":    part_id,
                "start_page": sp_page,
                "end_page":   sp_end_page,
                "region": _make_regions(
                    doc, sp_page, sp_y0, sp_end_page, sp_text_end, y_top, y_bottom
                )[0] if _make_regions(doc, sp_page, sp_y0, sp_end_page, sp_text_end, y_top, y_bottom) else {},
                "text_end_y": round(sp_text_end + TEXT_PAD, 2),
            })

        # Overall question text end
        q_end_page = q_blocks[-1].page if q_blocks else q_page
        last_page_blocks = [b for b in q_blocks if b.page == q_end_page]
        q_text_end = _last_text_y1(last_page_blocks) if last_page_blocks else q_y0
        
        # Capping safety net
        if qi + 1 < len(q_starts):
            _, nxt_page, nxt_y0 = q_starts[qi + 1]
            if nxt_page == q_end_page:
                q_text_end = min(q_text_end, nxt_y0 - TEXT_PAD - 2.0)

        q_regions  = _make_regions(doc, q_page, q_y0, q_end_page, q_text_end, y_top, y_bottom)

        snippet = _extract_snippet(q_blocks, doc[q_page].rect.width)

        # Use sequential index (qi + 1) instead of the regex-detected q_num.
        # If a false-positive block is picked up (a stray digit from a formula,
        # page number, etc.), the detected q_num may be wrong, causing all
        # subsequent questions to be misaligned by one.  Sequential numbering
        # guarantees the registry always uses 1, 2, 3 … in page order, which
        # matches the MS table order exactly.
        sequential_q_num = qi + 1
        if sequential_q_num != q_num:
            logger.debug(
                f"QP {paper_id}: reindexing detected Q{q_num} -> Q{sequential_q_num} "
                f"(page {q_page}, y={q_y0:.1f})"
            )
        questions_out.append({
            "q_num":      sequential_q_num,
            "text_snippet": snippet,
            "start_page": q_page,
            "end_page":   q_end_page,
            "regions":    q_regions,
            "text_end_y": round(q_text_end + TEXT_PAD, 2),
            "sub_parts":  sub_parts_out,
            "topic":      None,
            "confidence": None,
        })

    return {
        "paper_id":    paper_id,
        "doc_type":    "qp",
        "source_path": source_path,
        "pages":       page_metas,
        "questions":   questions_out,
    }


# ── MS registry builder ──────────────────────────────────────────────────────
def _build_ms_registry(
    doc: fitz.Document,
    source_path: str,
    paper_id: str,
    y_top: float,
    y_bottom: float,
    expected_q_nums: list[int] | None = None,
) -> dict:
    """
    Parse a Marking Scheme PDF.

    MS papers use a table layout (Question | Answer | Marks).
    Question boundaries are detected by the question number in the Question column.
    """
    page_metas = []
    for pg in doc:
        page_metas.append({
            "page_num": pg.number,
            "width":    pg.rect.width,
            "height":   pg.rect.height,
            "rotation": pg.rotation,
        })

    all_q_nums = expected_q_nums or list(range(1, 30))

    # Dynamically skip cover page (page 0) only if it contains cover-page text indicators.
    start_p = 0
    if len(doc) > 0:
        p0_text = doc[0].get_text().lower()
        cover_keywords = [
            "read these instructions",
            "instructions first",
            "dark blue or black pen",
            "do not use staples",
            "write your name",
            "candidate number",
            "index number",
            "published as an aid to teachers",
            "generic marking principle",
            "rules of marking",
            "generic marking principles",
        ]
        if any(kw in p0_text for kw in cover_keywords):
            start_p = 1

    ms_starts = _get_ms_question_starts(
        doc, start_p, len(doc) - 1, all_q_nums, y_bottom
    )

    questions_out: list[dict] = []
    for qi, (q_num, q_page, q_vy0) in enumerate(ms_starts):
        if qi + 1 < len(ms_starts):
            nxt_q_num, nxt_page, nxt_vy0 = ms_starts[qi + 1]
        else:
            nxt_q_num = None
            nxt_page = len(doc) - 1
            nxt_vy0  = doc[nxt_page].rect.y1 - y_bottom

        page = doc[q_page]
        # Convert visual y to standard coords for start
        std_start = visual_to_standard(0.0, q_vy0, page.rect.x1, q_vy0 + 1, page)
        q_y0_std = std_start[1]

        page_nxt = doc[nxt_page]
        std_end = visual_to_standard(0.0, nxt_vy0, page_nxt.rect.x1, nxt_vy0 + 1, page_nxt)
        nxt_y0_std = std_end[1]

        # Collect all text blocks for text_end detection
        q_blocks: list[_Block] = []
        for p_idx in range(q_page, nxt_page + 1):
            pg = doc[p_idx]
            for blk in _sorted_blocks(pg):
                if p_idx == q_page and blk.y0 < q_y0_std:
                    continue
                if p_idx == nxt_page:
                    if blk.y0 >= nxt_y0_std:
                        break
                    # Early break if block looks like next question start in MS table
                    if blk.y0 >= nxt_y0_std - BREAK_TOLERANCE:
                        m = _RE_Q_NUM.match(blk.text.strip())
                        if m and int(m.group(1)) == nxt_q_num:
                            break
                q_blocks.append(blk)

        # Post-collection trimming for MS question blocks
        if qi + 1 < len(ms_starts):
            q_blocks = [
                b for b in q_blocks
                if b.page < nxt_page or (b.page == nxt_page and b.y0 < nxt_y0_std - TRIM_MARGIN)
            ]

        q_end_page = q_blocks[-1].page if q_blocks else q_page
        ms_last_page_blocks = [b for b in q_blocks if b.page == q_end_page]
        q_text_end = _last_text_y1(ms_last_page_blocks) if ms_last_page_blocks else q_y0_std
        
        if qi + 1 < len(ms_starts):
            if nxt_page == q_end_page:
                q_text_end = min(q_text_end, nxt_y0_std - TEXT_PAD - 2.0)

        q_regions = _make_regions(doc, q_page, q_y0_std, q_end_page, q_text_end, y_top, y_bottom)

        snippet = ""
        for blk in q_blocks[:2]:
            snippet += blk.text + " "

        questions_out.append({
            "q_num":      q_num,
            "text_snippet": snippet.strip()[:200],
            "start_page": q_page,
            "end_page":   q_end_page,
            "regions":    q_regions,
            "text_end_y": round(q_text_end + TEXT_PAD, 2),
            "sub_parts":  [],   # MS sub-parts not separately extracted
            "topic":      None,
            "confidence": None,
        })

    return {
        "paper_id":    paper_id,
        "doc_type":    "ms",
        "source_path": source_path,
        "pages":       page_metas,
        "questions":   questions_out,
    }


# ── Public API ───────────────────────────────────────────────────────────────
def build_registry(
    pdf_path: str | None = None,
    doc_type: str = "qp",
    paper_id: str = "",
    y_top: float = 50.0,
    y_bottom: float = 60.0,
    expected_q_nums: list[int] | None = None,
    doc: fitz.Document | None = None,
) -> dict:
    """
    Parse a PDF and return its registry dict.

    Either *pdf_path* or an already-opened *doc* must be provided.
    When *doc* is given the caller is responsible for closing it —
    this function will NOT close it.

    Args:
        pdf_path:         Absolute path to the PDF (used when *doc* is None).
        doc_type:         "qp" or "ms".
        paper_id:         Canonical paper ID e.g. "9709_w25_qp_11".
        y_top:            Points from top to skip (header area).
        y_bottom:         Points from bottom to skip (footer / page-number area).
        expected_q_nums:  For MS — the question numbers found in the matching QP.
        doc:              Pre-opened (and pre-filtered) fitz.Document. If supplied,
                          *pdf_path* is used only for logging / source_path metadata.
    """
    caller_owns_doc = doc is not None   # we must NOT close it if caller gave it

    if doc is None:
        if pdf_path is None or not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return {}
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.error(f"Cannot open {pdf_path}: {exc}")
            return {}

    src_path = pdf_path or ""
    logger.info(f"Building registry for {paper_id} ({doc_type.upper()}, {len(doc)} pages)")

    try:
        if doc_type.lower() == "qp":
            registry = _build_qp_registry(doc, src_path, paper_id, y_top, y_bottom)
        else:
            registry = _build_ms_registry(doc, src_path, paper_id, y_top, y_bottom, expected_q_nums)
    finally:
        if not caller_owns_doc:
            doc.close()

    q_count = len(registry.get("questions", []))
    logger.info(f"  -> {q_count} questions detected in {paper_id}")

    # Post-parsing verbose summary logging
    import logging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Registry for {paper_id}:")
        for q in registry.get("questions", []):
            snippet = q.get("text_snippet", "")
            snippet_cleaned = snippet[:100].replace("\n", " ")
            # Sanitise to ASCII to prevent cp1252/Windows console encoding crashes on math symbols
            snippet_cleaned = snippet_cleaned.encode("ascii", errors="ignore").decode("ascii")
            logger.debug(f"  Q{q['q_num']}: {snippet_cleaned}...")

    return registry


def save_registry(registry: dict, path: str) -> None:
    """Persist a registry dict as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, ensure_ascii=False)
    logger.debug(f"Registry saved: {path}")


def load_registry_if_cached(
    path: str,
    filter_flags: dict | None = None,
) -> dict | None:
    """
    Load a registry JSON from disk.

    Returns None if absent, corrupt, or if the stored filter_flags do not
    match the *filter_flags* argument (cache invalidation on flag change).
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "paper_id" not in data or "questions" not in data:
            return None
        # Cache invalidation: rebuild if filter flags changed
        if filter_flags is not None:
            stored = data.get("filter_flags", {})
            if stored != filter_flags:
                logger.debug(
                    f"Cache invalidated for {path}: "
                    f"stored flags {stored} != current {filter_flags}"
                )
                return None
        return data
    except Exception as exc:
        logger.warning(f"Could not load cached registry {path}: {exc}")
    return None


# ── Multiprocessing-safe worker function ─────────────────────────────────────
def _build_registry_worker(args: tuple) -> tuple[str, dict]:
    """
    Top-level function (must be picklable for ProcessPoolExecutor).

    Args tuple layout:
        (pdf_path, doc_type, paper_id, y_top, y_bottom, registry_path,
         expected_q_nums, remove_blank, remove_formula, remove_additional,
         is_verbose)

    Returns: (paper_id, registry_dict)
    """
    (
        pdf_path, doc_type, paper_id, y_top, y_bottom,
        registry_path, expected_q_nums,
        remove_blank, remove_formula, remove_additional,
        is_verbose,
    ) = args

    if is_verbose:
        from auraq2.utils.logging import setup_logger
        setup_logger(verbose_level=1)

    filter_flags = {
        "remove_blank":      remove_blank,
        "remove_formula":    remove_formula,
        "remove_additional": remove_additional,
    }

    # Try cache first — invalidate if filter flags changed
    cached = load_registry_if_cached(registry_path, filter_flags=filter_flags)
    if cached:
        logger.debug(f"Cache hit (filter match): {paper_id}")
        return paper_id, cached

    # Apply page filters BEFORE registry building so blank/formula/additional
    # pages are never seen by the parser.
    from auraq2.core.compiler import filter_pdf
    try:
        filtered_doc = filter_pdf(pdf_path, remove_blank, remove_formula, remove_additional)
    except Exception as exc:
        logger.error(f"filter_pdf failed for {paper_id}: {exc} — falling back to unfiltered")
        filtered_doc = None

    try:
        if filtered_doc is not None:
            reg = build_registry(
                pdf_path=pdf_path,
                doc_type=doc_type,
                paper_id=paper_id,
                y_top=y_top,
                y_bottom=y_bottom,
                expected_q_nums=expected_q_nums,
                doc=filtered_doc,
            )
        else:
            reg = build_registry(pdf_path, doc_type, paper_id, y_top, y_bottom, expected_q_nums)
    finally:
        if filtered_doc is not None:
            try:
                filtered_doc.close()
            except Exception:
                pass

    # Store the active filter flags so we can detect stale caches later
    if reg:
        reg["filter_flags"] = filter_flags

    if reg and registry_path:
        save_registry(reg, registry_path)
    return paper_id, reg
