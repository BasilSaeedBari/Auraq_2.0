"""
Auraq 2.0 — Topical Compiler
Aggregates extracted question regions from many papers into topical booklets.

For each topic produces three PDF files:
  • {subject}_Paper_{paper}_{Topic}_QP.pdf   — Questions only
  • {subject}_Paper_{paper}_{Topic}_MS.pdf   — Mark scheme only
  • {subject}_Paper_{paper}_{Topic}_Merged.pdf — Q then MS interleaved

Cover page design: Purple/Vintage Grape palette (preserved from v1).
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import fitz  # PyMuPDF

from auraq2.utils.logging import get_logger
from auraq2.core.extractor import insert_regions_into_pdf

logger = get_logger()

# ── Colour palette (float RGB) ────────────────────────────────────────────────
_INDIGO       = (0.3098, 0.0706, 0.4431)
_ORCHID       = (0.4706, 0.2471, 0.5569)
_THISTLE      = (0.7490, 0.6745, 0.7843)
_GRAPE        = (0.2902, 0.2510, 0.3882)
_PALE_SLATE   = (0.7843, 0.7765, 0.8431)
_NEAR_WHITE   = (0.97,   0.97,   0.97)
_WHITE        = (1.0,    1.0,    1.0)


# ── Cover page ────────────────────────────────────────────────────────────────
def _create_cover_page(
    doc: fitz.Document,
    title: str,
    syllabus: str,
    topic: str,
    doc_type: str,
    year_range: str,
    q_count: str,
    insert_at: int = 0,
) -> None:
    """Draw a premium vector cover page (A4 595×842) and insert at position *insert_at*."""
    page = doc.new_page(width=595, height=842, pno=insert_at)

    # ── Header banner
    page.draw_rect(fitz.Rect(0, 0, 595, 195), color=_INDIGO, fill=_INDIGO, width=0)
    page.draw_rect(fitz.Rect(0, 195, 595, 203), color=_ORCHID, fill=_ORCHID, width=0)

    page.insert_textbox(
        fitz.Rect(40, 38, 555, 60),
        "AURAQ 2.0  ·  TOPICAL COMPILATION SYSTEM",
        fontsize=9.5, fontname="helv", color=_THISTLE,
        align=fitz.TEXT_ALIGN_LEFT,
    )
    page.insert_textbox(
        fitz.Rect(40, 68, 555, 130),
        title.upper(),
        fontsize=26, fontname="helv", color=_WHITE,
        align=fitz.TEXT_ALIGN_LEFT,
    )
    page.insert_textbox(
        fitz.Rect(40, 138, 555, 170),
        doc_type.upper(),
        fontsize=13, fontname="helv", color=_PALE_SLATE,
        align=fitz.TEXT_ALIGN_LEFT,
    )

    # ── Syllabus label
    page.insert_textbox(
        fitz.Rect(40, 228, 555, 274),
        f"Syllabus Component:\n{syllabus}",
        fontsize=14, fontname="helv", color=_GRAPE,
        align=fitz.TEXT_ALIGN_LEFT,
    )

    # ── Topic block
    page.draw_rect(fitz.Rect(40, 300, 555, 418), color=_NEAR_WHITE, fill=_NEAR_WHITE, width=0)
    page.draw_rect(fitz.Rect(40, 300, 48,  418), color=_ORCHID,     fill=_ORCHID,     width=0)
    page.insert_textbox(
        fitz.Rect(64, 322, 540, 405),
        f"TOPIC:\n{topic.upper()}",
        fontsize=20, fontname="helv", color=_INDIGO,
        align=fitz.TEXT_ALIGN_LEFT,
    )

    # ── Metadata boxes
    # Years
    page.draw_rect(fitz.Rect(40, 455, 278, 542), color=_NEAR_WHITE, fill=_NEAR_WHITE, width=0)
    page.insert_textbox(
        fitz.Rect(50, 470, 268, 536),
        f"YEARS INCLUDED\n\n{year_range}",
        fontsize=11.5, fontname="helv", color=_GRAPE,
        align=fitz.TEXT_ALIGN_CENTER,
    )
    # Questions
    page.draw_rect(fitz.Rect(317, 455, 555, 542), color=_NEAR_WHITE, fill=_NEAR_WHITE, width=0)
    page.insert_textbox(
        fitz.Rect(327, 470, 545, 536),
        f"TOTAL QUESTIONS\n\n{q_count}",
        fontsize=11.5, fontname="helv", color=_GRAPE,
        align=fitz.TEXT_ALIGN_CENTER,
    )

    # ── Notes
    notes = (
        "Document Details:\n"
        "- Chronological past paper compilation.\n"
        "- Vector layout retained - equations and graphs preserved.\n"
        "- Automatically parsed and compiled via Auraq 2.0.\n"
        "- Only question text extracted (working space excluded).\n"
    )
    if "Merged" in doc_type:
        notes += "- Format: each question is immediately followed by its mark scheme solution."
    else:
        notes += f"- This booklet contains the {doc_type} sections only."

    page.insert_textbox(
        fitz.Rect(40, 580, 555, 700),
        notes, fontsize=10, fontname="helv", color=_GRAPE,
        align=fitz.TEXT_ALIGN_LEFT,
    )

    # ── Footer
    page.draw_line(fitz.Point(40, 730), fitz.Point(555, 730), color=_PALE_SLATE, width=1)
    page.insert_textbox(
        fitz.Rect(40, 742, 555, 772),
        "Compiled by Auraq 2.0. All copyrights belong to the respective exam boards.",
        fontsize=8.5, fontname="helv", color=_PALE_SLATE,
        align=fitz.TEXT_ALIGN_CENTER,
    )


# ── Filename sanitiser ────────────────────────────────────────────────────────
def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").replace("&", "and").replace("'", "")


# ── CSV source map writer ─────────────────────────────────────────────────────
def _write_source_map(csv_path: str, rows: list[tuple[int, str, int, str]]) -> None:
    """
    Write a CSV source map alongside a topical PDF.

    *rows* is a list of (s_no, paper_id, q_num, download_link) tuples in the
    order the questions appear in the compiled PDF.
    """
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["S.NO", "Paper_ID", "Q_Num", "Download_Link"])
            writer.writerows(rows)
        logger.debug(f"Source map written: {csv_path}")
    except Exception as exc:
        logger.warning(f"Could not write source map {csv_path}: {exc}")


# ── Main public function ───────────────────────────────────────────────────────
def build_topical_booklets(
    paper_questions: list[dict],
    output_dir: str,
    subject_code: str,
    paper_code: str,
    syllabus_name: str,
    topics_list: list[str],
    start_year: int,
    end_year: int,
    is_mcq: bool = False,
) -> None:
    """
    Build topical PDF booklets from a list of question records.

    Each element of *paper_questions* is a dict:
      {
        "qp_path":  str,            # absolute path to the QP PDF
        "ms_path":  str | None,     # absolute path to the MS PDF (or None)
        "question": dict,           # question entry from QP registry
        "ms_entry": dict | None,    # matching MS registry entry (or None)
        "sort_key": tuple,          # chronological sort key
      }

    Produces inside output_dir:
      Topical_QP/{subject}_Paper_{paper}_{topic}_QP.pdf
      Topical_MS/{subject}_Paper_{paper}_{topic}_MS.pdf
      Topical_Merged/{subject}_Paper_{paper}_{topic}_Merged.pdf
    """
    qp_dir     = os.path.join(output_dir, "Topical_QP")
    ms_dir     = os.path.join(output_dir, "Topical_MS")
    merged_dir = os.path.join(output_dir, "Topical_Merged")
    for d in (qp_dir, ms_dir, merged_dir):
        os.makedirs(d, exist_ok=True)

    # Group by topic, preserving chronological order
    by_topic: dict[str, list[dict]] = defaultdict(list)
    sorted_qs = sorted(paper_questions, key=lambda x: x.get("sort_key", (9999,)))
    for item in sorted_qs:
        topic = item["question"].get("topic") or "Unclassified"
        by_topic[topic].append(item)

    year_range = f"{start_year} - {end_year}"

    generation_topics = list(topics_list)
    if by_topic.get("Unclassified"):
        generation_topics.append("Unclassified")

    for topic in generation_topics:
        items = by_topic.get(topic, [])
        if not items:
            continue

        q_count   = len(items)
        topic_fn  = _safe_name(topic)
        base_name = f"{subject_code}_Paper_{paper_code}_{topic_fn}"

        logger.info(f"Building booklets: {topic} ({q_count} questions)")

        # ── Open all required source docs ──────────────────────────────────
        open_docs: dict[tuple, fitz.Document] = {}

        def _get_doc(path: str | None, filter_flags: dict | None = None) -> fitz.Document | None:
            if not path or not os.path.exists(path):
                return None
            flags_key = tuple(sorted(filter_flags.items())) if filter_flags else None
            key = (path, flags_key)
            if key not in open_docs:
                if filter_flags:
                    from auraq2.core.compiler import filter_pdf
                    open_docs[key] = filter_pdf(
                        path,
                        remove_blank=filter_flags.get("remove_blank", True),
                        remove_formula=filter_flags.get("remove_formula", False),
                        remove_additional=filter_flags.get("remove_additional", True),
                    )
                else:
                    logger.warning(f"No filter flags stored in registry for {os.path.basename(path)}. Falling back to unfiltered PDF extraction.")
                    open_docs[key] = fitz.open(path)
            return open_docs[key]

        try:
            # Build shared source-map rows (same question order for QP & Merged)
            csv_rows: list[tuple] = [
                (sno, item.get("paper_id", ""), item["question"].get("q_num", 0), item.get("source_url", ""))
                for sno, item in enumerate(items, 1)
            ]

            # ── QP Booklet ─────────────────────────────────────────────────
            qp_dest = fitz.open()
            _create_cover_page(
                qp_dest, "Topical Past Papers", syllabus_name, topic,
                "Question Paper (QP)", year_range, str(q_count),
            )
            for item in items:
                qp_doc     = _get_doc(item.get("qp_path"), item.get("qp_filter_flags"))
                q_entry    = item["question"]
                item_label = item.get("label", "")
                if qp_doc:
                    regions = q_entry.get("regions", [])
                    fallback = (q_entry.get("start_page"), q_entry.get("end_page"))
                    fallback_tuple = tuple(fallback) if None not in fallback else None
                    insert_regions_into_pdf(qp_dest, qp_doc, regions, fallback_tuple, label=item_label)

            qp_path_out = os.path.join(qp_dir, f"{base_name}_QP.pdf")
            qp_dest.save(qp_path_out)
            qp_dest.close()
            _write_source_map(qp_path_out.replace(".pdf", ".csv"), csv_rows)

            # ── MS Booklet ─────────────────────────────────────────────────
            ms_dest      = fitz.open()
            has_ms       = False
            ms_csv_rows: list[tuple] = []
            _create_cover_page(
                ms_dest, "Topical Past Papers", syllabus_name, topic,
                "Marking Scheme (MS)", year_range, str(q_count),
            )
            for item in items:
                ms_doc     = _get_doc(item.get("ms_path"), item.get("ms_filter_flags"))
                ms_entry   = item.get("ms_entry")
                item_label = item.get("label", "")
                if ms_doc and ms_entry:
                    regions = ms_entry.get("regions", [])
                    fallback = (ms_entry.get("start_page"), ms_entry.get("end_page"))
                    fallback_tuple = tuple(fallback) if None not in fallback else None
                    added = insert_regions_into_pdf(ms_dest, ms_doc, regions, fallback_tuple, label=item_label)
                    if added:
                        has_ms = True
                        ms_csv_rows.append((
                            len(ms_csv_rows) + 1,
                            item.get("paper_id", ""),
                            item["question"].get("q_num", 0),
                            item.get("source_url", ""),
                        ))

            if has_ms:
                ms_path_out = os.path.join(ms_dir, f"{base_name}_MS.pdf")
                ms_dest.save(ms_path_out)
                _write_source_map(ms_path_out.replace(".pdf", ".csv"), ms_csv_rows)
            ms_dest.close()

            # ── Merged Booklet ─────────────────────────────────────────────
            merged_dest = fitz.open()
            _create_cover_page(
                merged_dest, "Topical Past Papers", syllabus_name, topic,
                "Questions & Solutions (Merged)", year_range, str(q_count),
            )
            for item in items:
                qp_doc   = _get_doc(item.get("qp_path"), item.get("qp_filter_flags"))
                ms_doc   = _get_doc(item.get("ms_path"), item.get("ms_filter_flags"))
                q_entry  = item["question"]
                ms_entry = item.get("ms_entry")
                item_label = item.get("label", "")

                if qp_doc:
                    regions = q_entry.get("regions", [])
                    fb = (q_entry.get("start_page"), q_entry.get("end_page"))
                    insert_regions_into_pdf(merged_dest, qp_doc, regions, tuple(fb) if None not in fb else None, label=item_label)
                if ms_doc and ms_entry:
                    regions = ms_entry.get("regions", [])
                    fb = (ms_entry.get("start_page"), ms_entry.get("end_page"))
                    insert_regions_into_pdf(merged_dest, ms_doc, regions, tuple(fb) if None not in fb else None, label=item_label)

            merged_path_out = os.path.join(merged_dir, f"{base_name}_Merged.pdf")
            merged_dest.save(merged_path_out)
            merged_dest.close()
            _write_source_map(merged_path_out.replace(".pdf", ".csv"), csv_rows)

        finally:
            # Close all opened source documents
            for doc in open_docs.values():
                try:
                    doc.close()
                except Exception:
                    pass

    logger.info(f"Topical booklet generation complete -> {output_dir}")
