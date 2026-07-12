"""
Auraq 2.0 — Entry Point

Runs GUI by default; runs CLI if arguments are supplied.
The `if __name__ == "__main__"` guard is REQUIRED for ProcessPoolExecutor
to work correctly on Windows (spawn start method).
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from auraq2.utils.logging import setup_logger, get_logger
from auraq2.utils.config import load_config, init_config
from auraq2.core.subjects_registry import load_registry as load_subjects_registry


def run_cli(args) -> None:
    from auraq2.cli.parser import normalise_sessions
    from auraq2.core.pipeline import run_pipeline

    logger = setup_logger(verbose_level=args.verbose)
    logger.info("Auraq 2.0 — CLI mode")

    init_config()
    load_subjects_registry()

    cfg       = load_config()
    groq_key  = os.environ.get("GROQ_API_KEY") or cfg.get("General", "groq_api_key", fallback="")
    output_dir = args.output or cfg.get("General", "download_directory",
                                         fallback=os.path.join(os.path.expanduser("~"), "Downloads", "Auraq2"))

    sessions = normalise_sessions(args.series)

    start_year = args.start + (2000 if args.start < 100 else 0)
    end_year   = args.end   + (2000 if args.end   < 100 else 0)

    def _cb(stage: str, cur: int, total: int) -> None:
        pct = cur / total * 100 if total else 0
        sys.stdout.write(f"\r[{stage}] {cur}/{total} ({pct:.0f}%)   ")
        sys.stdout.flush()
        if cur >= total:
            print()

    try:
        success = run_pipeline(
            curriculum=args.curriculum,
            subject_code=args.subject,
            paper=args.paper,
            variants=args.variants,
            sessions=sessions,
            start_year=start_year,
            end_year=end_year,
            output_dir=output_dir,
            remove_blank=args.remove_blank,
            remove_additional=args.remove_additional,
            remove_formula=args.remove_formula,
            generate_topical=args.topical,
            generate_docx=args.docx,
            groq_api_key=groq_key,
            ai_mode=args.ai_mode,
            max_download_workers=args.workers_dl,
            max_registry_workers=args.workers_cpu,
            progress_callback=_cb,
            sources=args.sources.split(",") if args.sources else None,
            save_ai_debug=args.save_ai_debug,
        )
        sys.exit(0 if success else 1)
    except Exception as exc:
        get_logger().error(f"Pipeline error: {exc}")
        import traceback
        get_logger().debug(traceback.format_exc())
        sys.exit(1)


def main() -> None:
    if len(sys.argv) > 1:
        from auraq2.cli.parser import parse_args
        run_cli(parse_args())
    else:
        setup_logger(verbose_level=0, log_to_file=True)
        init_config()
        load_subjects_registry()
        get_logger().info("Starting Auraq 2.0 in GUI mode …")
        from auraq2.gui.app import run_gui
        run_gui()


if __name__ == "__main__":
    main()
