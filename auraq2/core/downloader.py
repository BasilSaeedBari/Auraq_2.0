"""
Auraq 2.0 — Downloader
Downloads QP and MS PDFs from multiple sources into a structured folder hierarchy.

Folder structure produced:
  {base_dir}/{subject}/{year}/{Session}/Paper_{paper}/{QP|MS}/{filename}.pdf

Source priority (configurable): papacambridge → bestexamhelp → dynamicpapers
"""
from __future__ import annotations

import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from auraq2.utils.logging import get_logger
from auraq2.utils.helpers import SESSION_SHORT, SESSION_DIR, get_local_path

logger = get_logger()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}
TIMEOUT = 25

# In-process cache for scraped Dynamic Papers directory listings
_dp_cache: dict[str, list[tuple[str, str]]] = {}


# --------------------------------------------------------------------------- #
# URL builders                                                                   #
# --------------------------------------------------------------------------- #
def _filename_from_spec(spec: dict) -> str:
    ss  = SESSION_SHORT[spec["session"]]
    yr  = str(spec["year"])[-2:]
    dt  = spec["doc_type"].lower()
    var = spec["variant"]
    return f"{spec['subject']}_{ss}{yr}_{dt}_{var}.pdf"


def build_url_papacambridge(spec: dict) -> str:
    """
    PapaCambridge direct download.
    Pattern: https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/{filename}
    """
    return (
        "https://pastpapers.papacambridge.com/directories/CAIE/"
        f"CAIE-pastpapers/upload/{_filename_from_spec(spec)}"
    )


def build_url_bestexamhelp(spec: dict, level_path: str = "cambridge-international-a-level") -> str:
    """BestExamHelp direct download."""
    subject   = spec["subject"]
    beh_slug  = spec.get("beh_slug", "")
    year      = spec["year"]
    filename  = _filename_from_spec(spec)
    return (
        f"https://bestexamhelp.com/exam/{level_path}/"
        f"{beh_slug}/{year}/{filename}"
    )


def build_url_dynamicpapers_direct(spec: dict) -> str:
    """Dynamic Papers legacy direct-URL pattern."""
    return (
        "https://dynamicpapers.com/wp-content/uploads/2015/09/"
        f"{_filename_from_spec(spec)}"
    )


# --------------------------------------------------------------------------- #
# Dynamic Papers scrape (Cambridge)                                              #
# --------------------------------------------------------------------------- #
def _scrape_dynamic_papers(curriculum: str, dp_slug: str) -> list[tuple[str, str]]:
    """Crawl Dynamic Papers subject page and cache all PDF href/filename pairs."""
    if "Cambridge" in curriculum:
        url_segment = "cambridge-past-papers"
    else:
        url_segment = "edexcel-past-papers"

    subject_url = f"https://dynamicpapers.com/past-papers/{url_segment}/{dp_slug.strip('/')}/"

    if subject_url in _dp_cache:
        return _dp_cache[subject_url]

    logger.info(f"Scraping Dynamic Papers directory: {subject_url}")
    links: list[tuple[str, str]] = []
    try:
        r = requests.get(subject_url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            matches = re.findall(
                r'<a[^>]+href="([^"]+\.pdf)"[^>]*>', r.text, re.IGNORECASE
            )
            for href in matches:
                href = href.strip()
                if href.startswith("//"):
                    href = "https:" + href
                elif href.startswith("/"):
                    href = "https://dynamicpapers.com" + href
                elif not href.startswith("http"):
                    href = urllib.parse.urljoin(subject_url, href)
                filename = os.path.basename(urllib.parse.urlsplit(href).path)
                links.append((href, filename))
            logger.debug(f"Found {len(links)} links on Dynamic Papers page.")
        else:
            logger.warning(f"Dynamic Papers returned {r.status_code} for {subject_url}")
    except Exception as exc:
        logger.error(f"Error scraping Dynamic Papers: {exc}")

    _dp_cache[subject_url] = links
    return links


# --------------------------------------------------------------------------- #
# Core download function                                                         #
# --------------------------------------------------------------------------- #
def _is_valid_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def download_pdf(url: str, local_path: str) -> bool:
    """
    Download a PDF to local_path.

    - Skips if file already exists and is > 0 bytes.
    - Removes zero-byte / corrupt files before re-trying.
    - Validates %PDF magic bytes.
    """
    if os.path.exists(local_path):
        if os.path.getsize(local_path) > 0:
            return True          # Already downloaded and valid
        os.remove(local_path)   # Remove corrupted zero-byte file

    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200 and _is_valid_pdf(r.content):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as fh:
                fh.write(r.content)
            logger.info(f"✅ Downloaded: {os.path.basename(local_path)}")
            return True
        if r.status_code == 404:
            logger.debug(f"404: {url}")
        else:
            logger.debug(f"HTTP {r.status_code}: {url}")
    except requests.Timeout:
        logger.warning(f"Timeout: {url}")
    except Exception as exc:
        logger.debug(f"Error downloading {url}: {exc}")
    return False


# --------------------------------------------------------------------------- #
# Spec-level download (tries all sources in order)                               #
# --------------------------------------------------------------------------- #
def download_spec(spec: dict, base_dir: str, source_order: list[str]) -> bool:
    """
    Download the file described by *spec* to the structured folder hierarchy.
    Tries sources in priority order; returns True on first success.
    """
    local_path = get_local_path(base_dir, spec)

    # Already cached?
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        logger.debug(f"Cache hit: {os.path.basename(local_path)}")
        return True

    curriculum = spec.get("curriculum", "Cambridge A-Levels")
    is_cambridge = "Cambridge" in curriculum

    for source in source_order:
        if source == "papacambridge" and is_cambridge:
            if download_pdf(build_url_papacambridge(spec), local_path):
                return True

        elif source == "bestexamhelp" and spec.get("beh_slug"):
            # Map curriculum to BEH level path
            if "IGCSE" in curriculum:
                level = "cambridge-igcse"
            elif "O-Level" in curriculum or "O Level" in curriculum:
                level = "cambridge-o-level"
            else:
                level = "cambridge-international-a-level"
            if download_pdf(build_url_bestexamhelp(spec, level), local_path):
                return True

        elif source == "dynamicpapers":
            # Try direct URL first, then scraped links
            if download_pdf(build_url_dynamicpapers_direct(spec), local_path):
                return True
            dp_slug = spec.get("dp_slug", "maths")
            for href, fn in _scrape_dynamic_papers(curriculum, dp_slug):
                if fn.lower() == _filename_from_spec(spec).lower():
                    if download_pdf(href, local_path):
                        return True

    logger.warning(f"❌ Failed all sources: {_filename_from_spec(spec)}")
    return False


# --------------------------------------------------------------------------- #
# Spec generation                                                                #
# --------------------------------------------------------------------------- #
def generate_specs(
    curriculum: str,
    subject: str,
    beh_slug: str | None,
    dp_slug: str | None,
    years: list[int],
    sessions: list[str],
    papers: list[str],
    variants: list[str],
) -> list[dict]:
    """
    Generate the full list of download specs (one per file = year × session × paper × variant × doc_type).
    """
    specs = []
    for year in years:
        for session in sessions:
            if session not in SESSION_SHORT:
                logger.warning(f"Unknown session '{session}', skipping.")
                continue
            for paper in papers:
                for variant in variants:
                    for doc_type in ("qp", "ms"):
                        specs.append({
                            "curriculum":  curriculum,
                            "subject":     subject,
                            "beh_slug":    beh_slug,
                            "dp_slug":     dp_slug,
                            "year":        year,
                            "session":     session,
                            "paper":       paper,
                            "variant":     variant,
                            "doc_type":    doc_type,
                        })
    return specs


# --------------------------------------------------------------------------- #
# Batch parallel download                                                        #
# --------------------------------------------------------------------------- #
def download_batch(
    specs: list[dict],
    base_dir: str,
    source_order: list[str],
    max_workers: int = 10,
    progress_callback=None,
) -> int:
    """
    Download all specs concurrently using ThreadPoolExecutor.

    Args:
        specs:             List of download specs.
        base_dir:          Root of the structured download folder.
        source_order:      List of source names in priority order.
        max_workers:       Max concurrent download threads.
        progress_callback: Optional callable(current, total).

    Returns:
        Number of successfully downloaded files.
    """
    total = len(specs)
    success_count = 0

    logger.info(f"Starting batch download: {total} files, {max_workers} workers")
    logger.info(f"Source priority: {' → '.join(source_order)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_spec, spec, base_dir, source_order): spec
            for spec in specs
        }
        for idx, future in enumerate(as_completed(futures), 1):
            spec = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as exc:
                logger.error(f"Download task error: {exc}")
            if progress_callback:
                progress_callback(idx, total)

    logger.info(f"Download complete: {success_count}/{total} succeeded.")
    return success_count


# --------------------------------------------------------------------------- #
# Completeness check                                                             #
# --------------------------------------------------------------------------- #
def is_paper_complete(
    base_dir: str,
    curriculum: str,
    subject: str,
    year: int,
    session: str,
    paper: str,
    variants: list[str],
    beh_slug: str | None = None,
    dp_slug: str | None = None,
) -> bool:
    """Return True only if all required QP and MS files exist and are non-empty."""
    for doc_type in ("qp", "ms"):
        for variant in variants:
            spec = {
                "curriculum": curriculum,
                "subject": subject,
                "beh_slug": beh_slug,
                "dp_slug": dp_slug,
                "year": year,
                "session": session,
                "paper": paper,
                "variant": variant,
                "doc_type": doc_type,
            }
            path = get_local_path(base_dir, spec)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
    return True
