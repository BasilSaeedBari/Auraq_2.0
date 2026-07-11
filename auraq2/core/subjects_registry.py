"""
Auraq 2.0 — Subjects Registry
Loads subjects_registry.yaml and provides accessor functions.
The registry maps curricula → subjects → papers → topics + keyword_rules.
"""
from __future__ import annotations

import os
import yaml
from auraq2.utils.logging import get_logger

logger = get_logger()

_cached: dict | None = None


# --------------------------------------------------------------------------- #
# Loading                                                                        #
# --------------------------------------------------------------------------- #
def load_registry(custom_path: str | None = None) -> dict:
    """
    Load subjects_registry.yaml.  Search order:
      1. custom_path (if provided)
      2. CWD
      3. Parent of the package root
      4. Package root
    """
    global _cached
    if _cached is not None and custom_path is None:
        return _cached

    search = []
    if custom_path:
        search.append(custom_path)

    cwd = os.getcwd()
    search.append(os.path.join(cwd, "subjects_registry.yaml"))

    # Walk up from this file's location to find the project root
    core_dir = os.path.dirname(os.path.abspath(__file__))        # .../auraq2/core
    pkg_root  = os.path.dirname(core_dir)                         # .../auraq2
    proj_root = os.path.dirname(pkg_root)                         # .../Auraq_2.0
    search.append(os.path.join(proj_root, "subjects_registry.yaml"))
    search.append(os.path.join(pkg_root,  "subjects_registry.yaml"))

    data: dict | None = None
    for path in search:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                logger.debug(f"Loaded subjects registry from: {path}")
                break
            except Exception as exc:
                logger.error(f"Failed to parse registry at {path}: {exc}")

    if not data:
        logger.warning("subjects_registry.yaml not found — using empty registry.")
        data = {}

    if custom_path is None:
        _cached = data
    return data


def invalidate_cache() -> None:
    """Force next call to load_registry() to re-read from disk."""
    global _cached
    _cached = None


# --------------------------------------------------------------------------- #
# Accessors                                                                      #
# --------------------------------------------------------------------------- #
def get_curricula(custom_path: str | None = None) -> list[str]:
    return list(load_registry(custom_path).keys())


def get_subjects(curriculum: str, custom_path: str | None = None) -> dict:
    return load_registry(custom_path).get(curriculum, {})


def get_subject_details(
    curriculum: str, subject_code: str, custom_path: str | None = None
) -> dict:
    return get_subjects(curriculum, custom_path).get(str(subject_code), {})


def get_papers(
    curriculum: str, subject_code: str, custom_path: str | None = None
) -> dict:
    return get_subject_details(curriculum, subject_code, custom_path).get("papers", {})


def get_topics(
    curriculum: str, subject_code: str, paper_code: str, custom_path: str | None = None
) -> list[str]:
    paper = get_papers(curriculum, subject_code, custom_path).get(str(paper_code), {})
    return paper.get("topics", ["General"])


def get_keyword_rules(
    curriculum: str, subject_code: str, paper_code: str, custom_path: str | None = None
) -> dict:
    paper = get_papers(curriculum, subject_code, custom_path).get(str(paper_code), {})
    return paper.get("keyword_rules", {})


def is_mcq_paper(
    curriculum: str, subject_code: str, paper_code: str, custom_path: str | None = None
) -> bool:
    paper = get_papers(curriculum, subject_code, custom_path).get(str(paper_code), {})
    fmt  = paper.get("format", "")
    name = paper.get("name", "")
    return fmt.upper() == "MCQ" or "multiple choice" in name.lower()
