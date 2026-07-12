"""
Auraq 2.0 — DOCX Exporter
Converts PDF pages to images and inserts them into a Word document.
"""
from __future__ import annotations

import os
import tempfile
from pdf2image import convert_from_path
from docx import Document
from docx.shared import Inches, Mm
from auraq2.utils.logging import get_logger

logger = get_logger()


def pdf_to_docx(pdf_path: str, docx_path: str, dpi: int = 150) -> None:
    """
    Convert every page of a PDF to an image and insert into a new .docx.
    Each page becomes a full-page image.
    Uses streaming to disk (paths_only=True) to avoid memory overhead.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        logger.warning(f"Source PDF for DOCX conversion not found: {pdf_path}")
        return

    temp_dir = tempfile.mkdtemp(prefix="auraq_docx_")
    try:
        # Convert pages to images written directly to disk
        logger.debug(f"Converting PDF {os.path.basename(pdf_path)} to images...")
        image_paths = convert_from_path(
            pdf_path,
            dpi=dpi,
            thread_count=2,  # pdftoppm uses internal threads
            output_folder=temp_dir,
            paths_only=True,
            fmt="png"
        )

        doc = Document()
        
        # Configure section to fill A4 page with zero margins
        section = doc.sections[0]
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Mm(0)
        section.bottom_margin = Mm(0)
        section.left_margin = Mm(0)
        section.right_margin = Mm(0)

        for i, img_path in enumerate(image_paths):
            if i > 0:
                doc.add_page_break()
            doc.add_picture(img_path, width=Inches(8.27))

        doc.save(docx_path)
        logger.info(f"DOCX booklet generated: {docx_path}")

    except Exception as exc:
        logger.error(f"Failed to convert PDF {os.path.basename(pdf_path)} to DOCX: {exc}")
    finally:
        # Clean up all files in temp_dir and temp_dir itself
        try:
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
        except Exception:
            pass
