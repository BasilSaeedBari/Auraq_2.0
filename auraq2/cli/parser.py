"""
Auraq 2.0 — CLI Argument Parser
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auraq2",
        description="Auraq 2.0 — Topical Past Paper Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  auraq2 --curriculum "Cambridge A-Levels" --subject 9709 --paper 1 \\
         --variants 1 2 3 --series s w --start 2020 --end 2025 --topical

  auraq2 --curriculum "Cambridge A-Levels" --subject 9709 --paper 1 \\
         --variants 1 2 --series s --start 2023 --end 2025 \\
         --ai-mode heuristics --topical -v
""",
    )

    # ── Required ───────────────────────────────────────────────────────────
    parser.add_argument("--curriculum", "-c", required=True,
                        help='Curriculum name, e.g. "Cambridge A-Levels"')
    parser.add_argument("--subject", "-s", required=True,
                        help="Subject/syllabus code, e.g. 9709")
    parser.add_argument("--paper", "-p", required=True,
                        help="Paper component, e.g. 1")
    parser.add_argument("--variants", "-V", nargs="+", required=True,
                        help="Variant digit(s), e.g. 1 2 3")
    parser.add_argument("--series", nargs="+", required=True,
                        help='Session short codes: s (May/June), w (Oct/Nov), m (Feb/March), j (January)')
    parser.add_argument("--start", type=int, required=True, help="Start year, e.g. 2020")
    parser.add_argument("--end",   type=int, required=True, help="End year, e.g. 2025")

    # ── Optional ───────────────────────────────────────────────────────────
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--topical", action="store_true",
                        help="Generate topical booklets")
    parser.add_argument("--ai-mode", choices=["hybrid", "batch", "heuristics"],
                        default="hybrid",
                        help="Classification mode (default: hybrid)")
    parser.add_argument("--sources", default=None,
                        help="Override source priority, comma-separated, e.g. papacambridge,bestexamhelp")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download and re-parse even if cached")
    parser.add_argument("--confidence", type=float, default=None,
                        help="AI confidence threshold (0.0–1.0, default from config)")
    parser.add_argument("--remove-blank",    action="store_true", default=True)
    parser.add_argument("--remove-formula",  action="store_true", default=False)
    parser.add_argument("--remove-additional", action="store_true", default=True)
    parser.add_argument("--workers-dl",  type=int, default=10,
                        help="Max concurrent download threads (default: 10)")
    parser.add_argument("--workers-cpu", type=int, default=4,
                        help="Max registry-builder processes (default: 4)")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="Increase verbosity (-v = DEBUG)")
    parser.add_argument("--save-ai-debug", action="store_true",
                        help="Save AI prompts and responses to disk for debugging")

    return parser.parse_args()


# ── Session normalisation ─────────────────────────────────────────────────────
_SESSION_MAP: dict[str, str] = {
    "mj": "May/June", "s":  "May/June", "june": "May/June",
    "may": "May/June", "may/june": "May/June", "summer": "May/June",
    "on": "Oct/Nov", "w":  "Oct/Nov", "nov":  "Oct/Nov",
    "oct": "Oct/Nov", "oct/nov": "Oct/Nov", "winter": "Oct/Nov",
    "fm": "Feb/March", "m":  "Feb/March", "march": "Feb/March",
    "feb": "Feb/March", "feb/march": "Feb/March",
    "jan": "January", "j": "January", "january": "January",
}


def normalise_sessions(raw: list[str]) -> list[str]:
    result = []
    for s in raw:
        mapped = _SESSION_MAP.get(s.lower())
        if mapped and mapped not in result:
            result.append(mapped)
        elif not mapped:
            # Preserve unknown values as-is
            if s not in result:
                result.append(s)
    return result
