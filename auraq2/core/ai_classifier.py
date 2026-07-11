"""
Auraq 2.0 — AI Classifier (Groq only, batch per paper)

Instead of one API call per question, we send ALL questions from a paper
in a single Groq request and receive a JSON object back.
This reduces a 12-hour run to seconds per paper.

Decision logic (heuristic-first hybrid):
  1. Heuristic score >= 12 AND AI confidence < 0.90 → trust heuristic
     (strong keyword signal beats a moderately-confident AI)
  2. AI confidence >= 0.90                           → trust AI
  3. AI and heuristic agree on the same topic        → use AI topic
  4. Heuristic score >= fallback_score               → trust heuristic
  5. AI confidence >= threshold (default 0.80)       → use AI
  6. Any positive heuristic signal                   → use heuristic at low conf
  7. Otherwise                                       → "Unclassified"
"""
from __future__ import annotations

import json
import re
import time

import requests

from auraq2.utils.logging import get_logger

logger = get_logger()

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_DELAY = 1.5   # seconds between Groq calls (rate-limit safety)

# Strong heuristic override threshold:
# if h_score >= this AND ai_conf < STRONG_AI_THRESHOLD, trust heuristic
STRONG_HEURISTIC_SCORE   = 12
# AI confidence must be >= this to unconditionally override heuristic
STRONG_AI_THRESHOLD      = 0.90


# --------------------------------------------------------------------------- #
# Prompt builder                                                                 #
# --------------------------------------------------------------------------- #
def _build_batch_prompt(
    questions: list[dict],
    topics: list[str],
    syllabus_name: str,
) -> str:
    """
    Build the user prompt for batch classification.
    """
    topics_str = ", ".join(f'"{t}"' for t in topics)

    lines = [
        f'You are an examiner. Classify each question from "{syllabus_name}" into exactly one of these topics: [{topics_str}].',
        'If none fit, use "Unclassified".',
        'Output a JSON object with a key "classifications" that is an array of objects.',
        'Each object must have "q_num" (int), "topic" (string), and "confidence" (float 0.0-1.0).',
        'DO NOT include any other text, markdown, or explanation. ONLY the JSON object.',
        '',
        'Expected JSON format:',
        '{"classifications": [{"q_num": 1, "topic": "Trigonometry", "confidence": 0.95}]}',
        '',
        'Now classify the following questions (each line starts with Q<number>:):'
    ]

    for q in questions:
        snippet = (q.get("text_snippet") or "").replace("\n", " ")[:600]
        lines.append(f'Q{q["q_num"]}: {snippet}')

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Groq API call                                                                  #
# --------------------------------------------------------------------------- #
def _call_groq_batch(prompt: str, groq_key: str, model: str, max_retries: int = 3) -> str | None:
    """
    Send a batch classification request to Groq with exponential backoff for retries.
    Returns the raw response string or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    model,
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are a JSON-only classifier. Never write prose. "
                    "Your output must be a single JSON object with exactly the key 'classifications'. "
                    "Do not wrap it in ```json or any other text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature":     0.05,   # Very low temp for deterministic, factual output
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4096,
        "top_p": 1.0,
    }

    for attempt in range(max_retries):
        delay = REQUEST_DELAY * (2 ** attempt)
        try:
            time.sleep(delay)
            r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"].strip()
                logger.debug(f"Groq raw response: {raw[:500]}")
                return raw
            
            if r.status_code == 429:
                logger.warning(
                    f"Groq rate limit hit (429). "
                    f"Retrying in {delay * 2:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                continue
            elif r.status_code >= 500:
                logger.warning(
                    f"Groq server error ({r.status_code}). "
                    f"Retrying in {delay * 2:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                continue
            
            logger.warning(f"Groq returned HTTP {r.status_code}: {r.text[:200]}")
            break
        except Exception as exc:
            logger.error(f"Groq API error on attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                continue
    return None


# --------------------------------------------------------------------------- #
# Response parser                                                                #
# --------------------------------------------------------------------------- #
def _parse_batch_response(
    raw: str,
    topics: list[str],
    q_nums: list[int],
) -> dict[int, tuple[str, float]]:
    """
    Parse the Groq JSON response into a mapping of q_num -> (topic, confidence).
    Handles {"classifications": [...]} wrapper and various fallback shapes.
    """
    result: dict[int, tuple[str, float]] = {}

    try:
        data = json.loads(raw)
        items = None

        if isinstance(data, dict):
            # Primary key first, then fallbacks
            for key in ("classifications", "results", "questions", "data"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            if items is None:
                # Single-key dict whose value is the array
                for v in data.values():
                    if isinstance(v, list):
                        items = v
                        break
            if items is None:
                # Keys like "Q1" or "1" mapping to topic strings/dicts
                items = []
                for k, v in data.items():
                    digits = re.findall(r"\d+", str(k))
                    if digits:
                        qn = int(digits[0])
                        if isinstance(v, dict):
                            items.append({
                                "q_num": qn,
                                "topic": v.get("topic", ""),
                                "confidence": v.get("confidence", 0.8),
                            })
                        elif isinstance(v, str):
                            items.append({"q_num": qn, "topic": v, "confidence": 0.8})
        elif isinstance(data, list):
            items = data

        if not items:
            logger.warning(f"Groq response had no parseable list: {raw[:300]}")
            return result

        topics_lower = {t.lower(): t for t in topics}

        for item in items:
            if not isinstance(item, dict):
                continue
            q_num     = item.get("q_num")
            topic_raw = str(item.get("topic", "")).strip()
            conf      = float(item.get("confidence", 0.0))

            if q_num is None:
                continue

            # 1. Exact match (case-insensitive)
            matched = topics_lower.get(topic_raw.lower())

            # 2. Substring match — penalise confidence slightly
            if not matched:
                for tl, t in topics_lower.items():
                    if tl in topic_raw.lower() or topic_raw.lower() in tl:
                        matched = t
                        conf   *= 0.85
                        break

            if not matched:
                matched = "Unclassified"
                conf    = 0.0

            result[int(q_num)] = (matched, conf)

    except Exception as exc:
        logger.error(f"Failed to parse Groq batch response: {exc}\nRaw: {raw[:300]}")

    return result


# --------------------------------------------------------------------------- #
# Heuristic scoring                                                              #
# --------------------------------------------------------------------------- #
def _heuristic_score(text: str, keyword_rules: dict) -> dict[str, int]:
    """Return a dict of topic -> cumulative score (boolean matching per rule with word boundaries)."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for topic, rules in keyword_rules.items():
        s = 0
        for rule in rules:
            try:
                # Add word boundaries to avoid partial word match (e.g. "sin" inside "since")
                pattern = rule
                if pattern and pattern[0].isalnum():
                    pattern = r'\b' + pattern
                if pattern:
                    last = pattern[-1]
                    if last.isalnum():
                        pattern = pattern + r'\b'
                    elif last in ('?', '*', '+') and len(pattern) > 1 and pattern[-2].isalnum():
                        pattern = pattern + r'\b'
                
                # Boolean matching: check presence instead of frequency to prevent keyword spamming
                if re.search(pattern, text_lower, re.IGNORECASE):
                    s += 3
            except Exception:
                pass
        scores[topic] = s
    return scores


def classify_paper_heuristics(
    registry: dict,
    topics: list[str],
    keyword_rules: dict,
    fallback_score: int = 6,
) -> None:
    """
    Classify all questions in *registry* using keyword heuristics only.
    Modifies registry in-place.
    """
    for q in registry.get("questions", []):
        text   = q.get("text_snippet", "")
        scores = _heuristic_score(text, keyword_rules)
        if scores:
            best = max(scores, key=lambda k: scores[k])
            if scores[best] >= fallback_score:
                q["topic"]      = best
                q["confidence"] = min(0.85, scores[best] / 10.0)
                continue
        q["topic"]      = "Unclassified"
        q["confidence"] = 0.0


# --------------------------------------------------------------------------- #
# Public API: hybrid batch classify                                              #
# --------------------------------------------------------------------------- #
def classify_paper_batch(
    registry: dict,
    topics: list[str],
    syllabus_name: str,
    groq_key: str,
    groq_model: str,
    keyword_rules: dict | None = None,
    confidence_threshold: float = 0.80,   # raised from 0.70
    heuristic_fallback_score: int = 6,
) -> None:
    """
    Classify all questions in *registry* using Groq (batch) + heuristic.
    Modifies *registry* in-place by setting "topic" and "confidence" on each question.

    Decision priority (heuristic-first hybrid):
      1. Heuristic score >= STRONG_HEURISTIC_SCORE AND ai_conf < STRONG_AI_THRESHOLD
         -> trust heuristic (keyword signal is definitive)
      2. AI confidence >= STRONG_AI_THRESHOLD
         -> trust AI unconditionally
      3. AI and heuristic agree on the same topic
         -> use AI (agreement boosts confidence)
      4. Heuristic score >= heuristic_fallback_score
         -> trust heuristic
      5. AI confidence >= confidence_threshold (0.80)
         -> use AI (moderate confidence, heuristic was weak)
      6. Any heuristic score > 0
         -> weak heuristic, low confidence
      7. Unclassified
    """
    questions = registry.get("questions", [])
    if not questions:
        return

    kr = keyword_rules or {}

    # ── Step 1: build heuristic scores for all questions ────────────────────
    h_scores: dict[int, tuple[str, int]] = {}
    for q in questions:
        text   = q.get("text_snippet", "")
        scores = _heuristic_score(text, kr)
        if scores:
            best = max(scores, key=lambda k: scores[k])
            h_scores[q["q_num"]] = (best, scores[best])
        else:
            h_scores[q["q_num"]] = ("Unclassified", 0)

    # ── Step 2: Groq batch call ──────────────────────────────────────────────
    ai_results: dict[int, tuple[str, float]] = {}
    if groq_key:
        prompt = _build_batch_prompt(questions, topics, syllabus_name)
        raw    = _call_groq_batch(prompt, groq_key, groq_model)
        if raw:
            q_nums     = [q["q_num"] for q in questions]
            ai_results = _parse_batch_response(raw, topics, q_nums)
            logger.info(
                f"Groq classified {len(ai_results)}/{len(questions)} questions "
                f"for {registry['paper_id']}"
            )
        else:
            logger.warning(f"Groq call failed for {registry['paper_id']} - using heuristics only.")
    else:
        logger.info("No Groq key - using heuristics only.")

    # ── Step 3: heuristic-first decision logic ───────────────────────────────
    for q in questions:
        qn              = q["q_num"]
        ai_topic, ai_conf = ai_results.get(qn, (None, 0.0))
        h_topic,  h_score = h_scores.get(qn, ("Unclassified", 0))

        # Normalise None to empty string for comparisons
        ai_topic = ai_topic or ""

        if h_score >= STRONG_HEURISTIC_SCORE and ai_conf < STRONG_AI_THRESHOLD:
            # Strong keyword signal beats a moderately-confident AI
            final, conf = h_topic, min(0.95, h_score / 15.0)
            reason = "strong-heuristic"

        elif ai_topic and ai_conf >= STRONG_AI_THRESHOLD:
            # AI is very confident — trust it
            final, conf = ai_topic, ai_conf
            reason = "ai-very-confident"

        elif ai_topic and h_topic and ai_topic == h_topic:
            # Both agree — high trust in AI label
            final, conf = ai_topic, max(ai_conf, min(0.95, h_score / 15.0))
            reason = "agreement"

        elif h_score >= heuristic_fallback_score:
            # Heuristic is good enough on its own
            final, conf = h_topic, min(0.85, h_score / 10.0)
            reason = "heuristic-fallback"

        elif ai_topic and ai_conf >= confidence_threshold:
            # AI has moderate-to-high confidence, heuristic was weak
            final, conf = ai_topic, ai_conf
            reason = "ai-moderate"

        elif h_score > 0:
            # Weak heuristic signal — use it but flag low confidence
            final, conf = h_topic, min(0.5, h_score / 10.0)
            reason = "heuristic-weak"

        else:
            final, conf = "Unclassified", 0.0
            reason = "unclassified"

        q["topic"]      = final
        q["confidence"] = round(conf, 3)

        # Store intermediate values for summary logging; strip with _ prefix
        # so they are clearly internal and ignored by the registry serialiser.
        q["_h_topic"]  = h_topic
        q["_h_score"]  = h_score
        q["_ai_topic"] = ai_topic
        q["_ai_conf"]  = round(ai_conf, 3)
        q["_reason"]   = reason

    # ── Step 4: log summary table ────────────────────────────────────────────
    _log_classification_summary(questions, registry["paper_id"])


# --------------------------------------------------------------------------- #
# Classification summary logger                                                  #
# --------------------------------------------------------------------------- #
def _log_classification_summary(questions: list[dict], paper_id: str) -> None:
    """
    Print a compact ASCII table of classification results at INFO level.
    Fields: Q | Final Topic | Conf | Heuristic (score) | AI (conf) | Reason
    The temporary _-prefixed fields are removed from the question dicts afterwards.
    """
    COL_TOPIC = 22
    COL_H     = 22
    COL_AI    = 18
    COL_R     = 20

    sep = (
        f"{'':->3}-+-{'':->{COL_TOPIC}}-+-{'':->5}-+"
        f"-{'':->{COL_H}}-+-{'':->{COL_AI}}-+-{'':->{COL_R}}-"
    )
    header = (
        f"{'Q':>3} | {'Final Topic':<{COL_TOPIC}} | {'Conf':>5} |"
        f" {'Heuristic (score)':<{COL_H}} | {'AI topic (conf)':<{COL_AI}} | {'Reason':<{COL_R}}"
    )

    rows = [f"Classification summary -- {paper_id}:", header, sep]

    for q in questions:
        qn       = q["q_num"]
        topic    = q.get("topic", "Unclassified")
        conf     = q.get("confidence", 0.0)
        h_topic  = q.pop("_h_topic",  "N/A")
        h_score  = q.pop("_h_score",  0)
        ai_topic = q.pop("_ai_topic", "N/A")
        ai_conf  = q.pop("_ai_conf",  0.0)
        reason   = q.pop("_reason",   "")

        h_cell  = f"{h_topic[:18]} ({h_score:>2})"
        ai_cell = f"{(ai_topic or '-')[:12]} ({ai_conf:.2f})"

        rows.append(
            f"{qn:>3} | {topic:<{COL_TOPIC}} | {conf:>5.2f} |"
            f" {h_cell:<{COL_H}} | {ai_cell:<{COL_AI}} | {reason:<{COL_R}}"
        )

    rows.append(sep)
    logger.info("\n".join(rows))
