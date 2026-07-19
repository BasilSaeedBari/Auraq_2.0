"""
Auraq 2.0 — Pipeline Orchestrator

Stages (producer-consumer):
  1. Download     — ThreadPoolExecutor  (I/O-bound)
  2. Registry     — ProcessPoolExecutor (CPU-bound, true parallelism)
  3. AI Classify  — ThreadPoolExecutor  (network I/O, rate-limited)
  4. Extract      — in-process grouping
  5. Compile      — topical booklet generation

Each stage can resume from cached state (registry JSON already on disk,
downloads already present) without re-doing work.

Windows multiprocessing safety:
  All ProcessPoolExecutor usage MUST be called from within a
  'if __name__ == "__main__":' context in the entry point,
  or from a non-main thread (as is the case inside PipelineThread).
  We use the "spawn" start method for safety on Windows.
"""
from __future__ import annotations

import multiprocessing
import os
import re
import json
import time
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from auraq2.utils.logging import get_logger
from auraq2.utils.helpers import get_local_path, get_registry_path, SESSION_SHORT, paper_id_from_spec
from auraq2.utils.config import load_config
from auraq2.core.subjects_registry import (
    get_subject_details, get_topics, get_keyword_rules, is_mcq_paper,
    load_registry as load_subjects_registry,
)
from auraq2.core.downloader import (
    generate_specs, download_batch, is_paper_complete, build_url_papacambridge,
)
from auraq2.core.registry_builder import (
    _build_registry_worker, load_registry_if_cached, save_registry,
)
from auraq2.core.ai_classifier import classify_paper_batch, classify_paper_heuristics
from auraq2.core.compiler import parse_sort_key
from auraq2.core.topical_compiler import build_topical_booklets

logger = get_logger()


# --------------------------------------------------------------------------- #
# Helpers                                                                        #
# --------------------------------------------------------------------------- #
def _variant_to_paper_and_digit(variant: str, paper: str) -> tuple[str, str]:
    """
    Convert a variant string and paper component to (paper_digit, variant_digit).
    e.g. variant="1", paper="1" → ("1", "1"); variant="11", paper="1" → ("1", "1")
    """
    if len(variant) > 1:
        return variant[0], variant[-1]
    return paper, variant


# --------------------------------------------------------------------------- #
# Public pipeline entry-point                                                    #
# --------------------------------------------------------------------------- #
def run_pipeline(
    curriculum: str,
    subject_code: str,
    paper: str | list[str], # single paper component e.g. "1" or list e.g. ["1", "2"]
    variants: list[str],    # variant digits, e.g. ["1", "2", "3"]
    sessions: list[str],    # e.g. ["May/June", "Oct/Nov"]
    start_year: int,
    end_year: int,
    output_dir: str,
    remove_blank: bool = True,
    remove_additional: bool = True,
    remove_formula: bool = False,
    generate_topical: bool = True,
    generate_docx: bool = False,
    groq_api_key: str = "",
    ai_mode: str = "hybrid",    # "batch", "heuristics", "hybrid"
    max_download_workers: int = 10,
    max_registry_workers: int = 4,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    sources: Optional[list[str]] = None,
    save_ai_debug: bool = False,
) -> bool:
    """
    Run the full Auraq 2.0 pipeline for one paper component.

    Returns True on success.
    """
    def _cb(stage: str, cur: int, total: int) -> None:
        if progress_callback:
            progress_callback(stage, cur, total)

    logger.info("=" * 60)
    logger.info("Auraq 2.0 Pipeline Starting")
    logger.info("=" * 60)

    # ── Load config -----------------------------------------------------------
    config   = load_config()
    base_dir = os.environ.get("AURAQ_DOWNLOAD_DIR") or config.get(
        "General", "download_directory",
        fallback=os.path.join(os.path.expanduser("~"), "Downloads", "Auraq2"),
    )
    if sources is None:
        sources  = [s.strip() for s in
                    config.get("General", "sources_order", fallback="papacambridge,bestexamhelp,dynamicpapers").split(",")]
    conf_threshold = float(config.get("AI", "batch_confidence_threshold", fallback="0.80"))
    h_score        = int(config.get("AI", "heuristic_fallback_score",     fallback="6"))
    strong_h_score = int(config.get("AI", "strong_heuristic_score",       fallback="12"))
    strong_ai_thr  = float(config.get("AI", "strong_ai_threshold",        fallback="0.90"))
    qp_top    = int(config.get("Clipping", "qp_top_margin",    fallback="50"))
    qp_bot    = int(config.get("Clipping", "qp_bottom_margin", fallback="60"))
    ms_top    = int(config.get("Clipping", "ms_top_margin",    fallback="50"))
    ms_bot    = int(config.get("Clipping", "ms_bottom_margin", fallback="40"))
    groq_model = os.environ.get("GROQ_MODEL") or config.get("General", "groq_model", fallback="llama-3.3-70b-versatile")
    # Build ordered fallback model list.  The primary model is always first;
    # additional fallbacks come from the config key "groq_model_fallbacks".
    _fallback_str = config.get(
        "General", "groq_model_fallbacks",
        fallback="llama-3.3-70b-versatile,llama-4-scout,openai/gpt-oss-20b"
    )
    groq_models: list[str] = [groq_model] + [
        m.strip() for m in _fallback_str.split(",")
        if m.strip() and m.strip() != groq_model
    ]

    # Inject tunable constants into the classifier at runtime
    import auraq2.core.ai_classifier as _clf
    _clf.STRONG_HEURISTIC_SCORE = strong_h_score
    _clf.STRONG_AI_THRESHOLD    = strong_ai_thr

    # ── Validate subject ------------------------------------------------------
    load_subjects_registry()
    sub_details = get_subject_details(curriculum, subject_code)
    if not sub_details:
        logger.error(f"Subject {subject_code} not found under {curriculum}")
        return False

    if isinstance(paper, str):
        paper_codes = [paper]
    else:
        paper_codes = paper

    primary_paper = paper_codes[0]

    topics       = get_topics(curriculum, subject_code, primary_paper)
    kw_rules     = get_keyword_rules(curriculum, subject_code, primary_paper)
    is_mcq       = is_mcq_paper(curriculum, subject_code, primary_paper)
    syllabus_name = (
        f"{curriculum} {sub_details.get('name', '')} "
        f"({subject_code}) - Component {primary_paper}"
    )

    beh_slug = sub_details.get("beh_slug")
    dp_slug  = sub_details.get("dp_slug")

    # ── Build full variant codes (e.g. paper=1, variant=1 → "11") -----------
    full_variants_set = set()
    for pc in paper_codes:
        for v in variants:
            if len(v) == 1:
                full_variants_set.add(f"{pc}{v}")
            else:
                full_variants_set.add(v)
    full_variants = sorted(list(full_variants_set))

    years = list(range(start_year, end_year + 1))

    # ── Stage 1: Download ─────────────────────────────────────────────────────
    logger.info("Stage 1: Downloading PDFs ...")
    specs = generate_specs(
        curriculum, subject_code, beh_slug, dp_slug,
        years, sessions, paper_codes, full_variants,
    )
    _cb("Downloading", 0, len(specs))

    # Fast-path: skip the whole download stage if every file is already cached.
    _all_cached = all(
        os.path.exists(get_local_path(base_dir, s)) and os.path.getsize(get_local_path(base_dir, s)) > 0
        for s in specs
    )
    if _all_cached:
        logger.info("All required PDFs already cached \u2014 skipping download.")
        _cb("Downloading", len(specs), len(specs))
    else:
        def _dl_cb(cur: int, total: int) -> None:
            _cb("Downloading", cur, total)

        download_batch(specs, base_dir, sources, max_download_workers, _dl_cb)
        _cb("Downloading", len(specs), len(specs))

    # Separate QP and MS specs
    qp_specs = [s for s in specs if s["doc_type"] == "qp"]
    ms_specs = [s for s in specs if s["doc_type"] == "ms"]

    # Filter to actually-existing files
    qp_specs = [s for s in qp_specs if os.path.exists(get_local_path(base_dir, s))]
    ms_specs = [s for s in ms_specs if os.path.exists(get_local_path(base_dir, s))]

    if not qp_specs:
        logger.error("No QP files were downloaded. Cannot continue.")
        return False

    # Log the active filter configuration so it is visible in the run log
    logger.info(
        f"Page filters active: blank={remove_blank}, "
        f"formula={remove_formula}, additional={remove_additional}"
    )

    # ── Stage 2: Registry Building (ProcessPoolExecutor) ─────────────────────
    logger.info("Stage 2: Building question registries (parallel) ...")
    _cb("Parsing", 0, len(qp_specs) + len(ms_specs))

    # Build argument tuples for the picklable worker.
    # Tuple layout: (pdf_path, doc_type, paper_id, y_top, y_bot,
    #                registry_path, expected_q_nums,
    #                remove_blank, remove_formula, remove_additional, is_verbose)
    is_verbose = logger.isEnabledFor(logging.DEBUG)

    # Use "spawn" context for Windows safety
    ctx = multiprocessing.get_context("spawn")

    qp_registries: dict[str, dict] = {}
    ms_registries: dict[str, dict] = {}
    completed = 0

    # ── Phase 2a: QP Registries ──────────────────────────────────────────────
    qp_worker_args: list[tuple] = []
    for spec in qp_specs:
        pdf_path  = get_local_path(base_dir, spec)
        reg_path  = get_registry_path(base_dir, spec)
        pid       = paper_id_from_spec(spec)
        qp_worker_args.append((
            pdf_path, "qp", pid, qp_top, qp_bot, reg_path, None,
            remove_blank, remove_formula, remove_additional,
            is_verbose,
        ))

    if qp_worker_args:
        with ProcessPoolExecutor(max_workers=max_registry_workers, mp_context=ctx) as executor:
            future_map = {
                executor.submit(_build_registry_worker, args): args
                for args in qp_worker_args
            }
            for future in as_completed(future_map):
                args = future_map[future]
                try:
                    pid, registry = future.result()
                    qp_registries[pid] = registry
                except Exception as exc:
                    logger.error(f"QP Registry worker failed for {args[2]}: {exc}")
                completed += 1
                _cb("Parsing", completed, len(qp_specs) + len(ms_specs))

    # ── Phase 2b: MS Registries ──────────────────────────────────────────────
    ms_worker_args: list[tuple] = []
    for spec in ms_specs:
        pdf_path  = get_local_path(base_dir, spec)
        reg_path  = get_registry_path(base_dir, spec)
        pid       = paper_id_from_spec(spec)

        # Match with QP registry to find expected question numbers
        qp_pid = pid.replace("_ms_", "_qp_")
        qp_reg = qp_registries.get(qp_pid)
        expected_q_nums = None
        if qp_reg:
            expected_q_nums = [q["q_num"] for q in qp_reg.get("questions", [])]

        ms_worker_args.append((
            pdf_path, "ms", pid, ms_top, ms_bot, reg_path, expected_q_nums,
            remove_blank, remove_formula, remove_additional,
            is_verbose,
        ))

    if ms_worker_args:
        with ProcessPoolExecutor(max_workers=max_registry_workers, mp_context=ctx) as executor:
            future_map = {
                executor.submit(_build_registry_worker, args): args
                for args in ms_worker_args
            }
            for future in as_completed(future_map):
                args = future_map[future]
                try:
                    pid, registry = future.result()
                    ms_registries[pid] = registry
                except Exception as exc:
                    logger.error(f"MS Registry worker failed for {args[2]}: {exc}")
                completed += 1
                _cb("Parsing", completed, len(qp_specs) + len(ms_specs))

    # ── Stage 2.5: Consistency Check ──────────────────────────────────────────
    for qp_pid, qp_reg in qp_registries.items():
        ms_pid = qp_pid.replace("_qp_", "_ms_")
        ms_reg = ms_registries.get(ms_pid)
        if ms_reg:
            qp_qs = len(qp_reg.get("questions", []))
            ms_qs = len(ms_reg.get("questions", []))
            if qp_qs != ms_qs:
                logger.warning(
                    f"Mismatched question count for {qp_pid}: "
                    f"QP has {qp_qs} questions, but MS ({ms_pid}) has {ms_qs} questions! "
                    "This could indicate skipped questions during layout parsing."
                )

    # ── Stage 3: AI Classification ────────────────────────────────────────────
    logger.info("Stage 3: Classifying questions ...")
    _cb("Classifying", 0, len(qp_registries))

    for idx, (pid, reg) in enumerate(qp_registries.items()):
        if not reg.get("questions"):
            continue

        # Enrich QP snippets with corresponding MS answers for high-quality classification
        ms_pid = pid.replace("_qp_", "_ms_")
        ms_reg = ms_registries.get(ms_pid)
        if ms_reg:
            ms_qs = {mq["q_num"]: mq.get("text_snippet", "") for mq in ms_reg.get("questions", [])}
            for q in reg.get("questions", []):
                ms_snip = ms_qs.get(q["q_num"], "")
                if ms_snip:
                    ms_clean = re.sub(r'\s+', ' ', ms_snip).strip()
                    if ms_clean:
                        q["text_snippet"] = f"{q.get('text_snippet', '')} [MS: {ms_clean}]".strip()

        if ai_mode == "heuristics":
            classify_paper_heuristics(reg, topics, kw_rules, h_score)
        elif ai_mode == "batch" and groq_api_key:
            classify_paper_batch(reg, topics, syllabus_name, groq_api_key,
                                 groq_model, kw_rules, conf_threshold, h_score,
                                 save_ai_debug, os.path.join(base_dir, "ai_debug"),
                                 groq_models)
        else:  # hybrid
            classify_paper_batch(reg, topics, syllabus_name, groq_api_key or "",
                                 groq_model, kw_rules, conf_threshold, h_score,
                                 save_ai_debug, os.path.join(base_dir, "ai_debug"),
                                 groq_models)
        # Persist updated registry (now has topic + confidence)
        for spec in qp_specs:
            if paper_id_from_spec(spec) == pid:
                reg_path = get_registry_path(base_dir, spec)
                save_registry(reg, reg_path)
                break
        
        # Pacing delay between batch API calls to smooth TPM usage and avoid rate limits
        if ai_mode != "heuristics" and groq_api_key:
            time.sleep(3.0)

        _cb("Classifying", idx + 1, len(qp_registries))

    # ── Stage 4: Collect question records ─────────────────────────────────────
    if not generate_topical:
        logger.info("Topical generation disabled — pipeline complete.")
        return True

    logger.info("Stage 4: Assembling question records ...")

    # Build a lookup: paper_id → spec (for MS matching)
    ms_lookup: dict[str, dict] = {}
    for spec in ms_specs:
        pid = paper_id_from_spec(spec)
        ms_lookup[pid] = spec

    paper_questions: list[dict] = []

    for pid, qp_reg in qp_registries.items():
        qp_path = None
        for spec in qp_specs:
            if paper_id_from_spec(spec) == pid:
                qp_path = get_local_path(base_dir, spec)
                sort_key = parse_sort_key(os.path.basename(qp_path))
                break

        # Find matching MS registry by session+variant pairing
        # e.g. QP pid "9709_w25_qp_11" → MS pid "9709_w25_ms_11"
        ms_pid   = pid.replace("_qp_", "_ms_")
        ms_reg   = ms_registries.get(ms_pid)
        ms_path  = None
        if ms_pid in ms_lookup:
            ms_path = get_local_path(base_dir, ms_lookup[ms_pid])

        # Build MS lookup by q_num
        ms_by_q: dict[int, dict] = {}
        if ms_reg:
            for ms_q in ms_reg.get("questions", []):
                ms_by_q[ms_q["q_num"]] = ms_q

        # Get filter flags stored in registries
        qp_filter_flags = qp_reg.get("filter_flags")
        ms_filter_flags = ms_reg.get("filter_flags") if ms_reg else None

        for q in qp_reg.get("questions", []):
            # Build a deterministic source URL for the CSV source map
            _qp_spec = next((s for s in qp_specs if paper_id_from_spec(s) == pid), None)
            _source_url = build_url_papacambridge(_qp_spec) if _qp_spec else ""
            paper_questions.append({
                "paper_id": pid,
                "qp_path":  qp_path,
                "ms_path":  ms_path,
                "qp_filter_flags": qp_filter_flags,
                "ms_filter_flags": ms_filter_flags,
                "question": q,
                "ms_entry": ms_by_q.get(q["q_num"]),
                "sort_key": sort_key,
                "label":    f"{pid} Q{q['q_num']}",
                "source_url": _source_url,
            })

    # ── Stage 5: Topical Booklets ─────────────────────────────────────────────
    logger.info(f"Stage 5: Building topical booklets for {len(paper_questions)} questions ...")
    _cb("Compiling", 0, len(topics))

    os.makedirs(output_dir, exist_ok=True)

    build_topical_booklets(
        paper_questions=paper_questions,
        output_dir=output_dir,
        subject_code=subject_code,
        paper_codes=paper_codes,
        syllabus_name=syllabus_name,
        topics_list=topics,
        start_year=start_year,
        end_year=end_year,
        is_mcq=is_mcq,
        generate_docx=generate_docx,
    )

    _cb("Compiling", len(topics), len(topics))

    logger.info("=" * 60)
    logger.info("Auraq 2.0 Pipeline Completed Successfully [OK]")
    logger.info("=" * 60)
    return True
