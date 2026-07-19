"""
Auraq 2.0 — GUI Callbacks
Log handler that bridges Python logging into the GUI console,
and the background PipelineThread that keeps the GUI responsive.
"""
from __future__ import annotations

import logging
import threading
from auraq2.utils.logging import get_logger


class GuiLogHandler(logging.Handler):
    """
    Logging handler that routes log records into a ScrollableLogBox widget.
    Thread-safe via tkinter's after() scheduler.
    """
    def __init__(self, log_box) -> None:
        super().__init__()
        self.log_box = log_box

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = record.levelname  # INFO, WARNING, ERROR, DEBUG
            self.log_box.insert_log(msg, level)
        except Exception:
            self.handleError(record)


class PipelineThread(threading.Thread):
    """
    Runs the Auraq 2.0 pipeline in a background thread so the Tkinter
    event loop stays responsive.
    """
    def __init__(
        self,
        app,
        curriculum: str,
        subject_code: str,
        paper: str | list[str],
        variants: list[str],
        sessions: list[str],
        start_year: int,
        end_year: int,
        output_dir: str,
        remove_blank: bool,
        remove_additional: bool,
        remove_formula: bool,
        generate_topical: bool,
        generate_docx: bool,
        groq_api_key: str,
        ai_mode: str,
    ) -> None:
        super().__init__(daemon=True)
        self.app = app
        self.kwargs = dict(
            curriculum=curriculum,
            subject_code=subject_code,
            paper=paper,
            variants=variants,
            sessions=sessions,
            start_year=start_year,
            end_year=end_year,
            output_dir=output_dir,
            remove_blank=remove_blank,
            remove_additional=remove_additional,
            remove_formula=remove_formula,
            generate_topical=generate_topical,
            generate_docx=generate_docx,
            groq_api_key=groq_api_key,
            ai_mode=ai_mode,
        )

    def run(self) -> None:
        logger = get_logger()
        try:
            from auraq2.core.pipeline import run_pipeline
            success = run_pipeline(
                progress_callback=self.app.update_progress,
                **self.kwargs,
            )
            self.app.root.after(0, self.app.on_pipeline_complete, success, None)
        except Exception as exc:
            logger.error(f"Pipeline thread error: {exc}")
            import traceback
            logger.debug(traceback.format_exc())
            self.app.root.after(0, self.app.on_pipeline_complete, False, str(exc))
