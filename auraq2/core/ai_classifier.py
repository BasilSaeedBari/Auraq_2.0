"""
Auraq 2.0 — AI Classifier (Groq only, batch per paper)

Instead of one API call per question, we send ALL questions from a paper
in a single Groq request and receive a JSON array back.
This reduces a 12-hour run to seconds per paper.

Decision logic:
  1. Groq returns confidence ≥ threshold → use AI topic
  2. Groq and heuristic agree → use AI topic
  3. Heuristic score > fallback_score → use heuristic
  4. Otherwise → "Unclassified"
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import requests

from auraq2.utils.logging import get_logger

logger = get_logger()

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"
REQUEST_DELAY = 1.5   # seconds between Groq calls (rate-limit safety)


# --------------------------------------------------------------------------- #
# Prompt builder                                                                 #
# --------------------------------------------------------------------------- #
def _build_batch_prompt(
    questions: list[dict],
    topics: list[str],
    syllabus_name: str,
) -> str:
    """
    Build the system + user prompt for batch classification.
    Each question contributes at most 400 chars of text_snippet.
    """
    topics_str = ", ".join(f'"{t}"' for t in topics)

    lines = [
        f'Classify each exam question from "{syllabus_name}" into exactly one topic.',
        f"Valid topics: [{topics_str}].",
        'Use "Unclassified" only if no topic fits.',
        "",
        "You must return a valid JSON object with the key 'classifications' containing the array:",
        "{",
        '  "classifications": [',
        '    {"q_num": <int>, "topic": "<topic>", "confidence": <float 0-1>},',
        "    ...",
        "  ]",
        "}",
        "",
        "Questions:",
    ]
    for q in questions:
        snippet = (q.get("text_snippet") or "").replace("\n", " ")[:400]
        lines.append(f'Q{q["q_num"]}: {snippet}')

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Groq API call                                                                  #
# --------------------------------------------------------------------------- #
def _call_groq_batch(prompt: str, groq_key: str) -> str | None:
    """
    Send a batch classification request to Groq.
    Returns the raw response string or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    GROQ_MODEL,
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are an expert exam syllabus classifier. "
                    "Respond ONLY with a valid JSON object matching the requested schema. "
                    "No conversational filler, no markdown wrapping, no extra keys."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature":     0.1,
        "response_format": {"type": "json_object"},
    }

    try:
        time.sleep(REQUEST_DELAY)
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        logger.warning(f"Groq returned HTTP {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.error(f"Groq API error: {exc}")
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
    Parse the Groq JSON response into a mapping of q_num → (topic, confidence).
    Handles both array-direct and {"classifications": [...]} wrappers.
    Falls back gracefully on malformed output.
    """
    result: dict[int, tuple[str, float]] = {}

    try:
        data = json.loads(raw)
        items = None
        # Unwrap common wrapper keys
        if isinstance(data, dict):
            for key in ("classifications", "results", "questions", "data"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            else:
                # Maybe it's a single-key dict whose value is the array
                for v in data.values():
                    if isinstance(v, list):
                        items = v
                        break
                if items is None:
                    # Let's try parsing keys like "Q1" or "1"
                    items = []
                    for k, v in data.items():
                        digits = re.findall(r"\d+", str(k))
                        if digits:
                            qn = int(digits[0])
                            if isinstance(v, dict):
                                items.append({"q_num": qn, "topic": v.get("topic", ""), "confidence": v.get("confidence", 0.8)})
                            elif isinstance(v, str):
                                items.append({"q_num": qn, "topic": v, "confidence": 0.8})
        elif isinstance(data, list):
            items = data

        if not items:
            logger.warning(f"Groq response is not a valid JSON array or object containing a list: {raw[:300]}")
            return result

        topics_lower = {t.lower(): t for t in topics}

        for item in items:
            if not isinstance(item, dict):
                continue
            q_num = item.get("q_num")
            topic_raw = str(item.get("topic", "")).strip()
            conf  = float(item.get("confidence", 0.0))

            if q_num is None:
                continue

            # Exact match (case-insensitive)
            matched = topics_lower.get(topic_raw.lower())
            if not matched:
                # Partial match
                for tl, t in topics_lower.items():
                    if tl in topic_raw.lower() or topic_raw.lower() in tl:
                        matched = t
                        conf *= 0.85
                        break

            if not matched:
                matched = "Unclassified"
                conf = 0.0

            result[int(q_num)] = (matched, conf)

    except Exception as exc:
        logger.error(f"Failed to parse Groq batch response: {exc}\nRaw: {raw[:300]}")

    return result


# --------------------------------------------------------------------------- #
# Heuristic fallback                                                             #
# --------------------------------------------------------------------------- #
def _heuristic_score(text: str, keyword_rules: dict) -> dict[str, int]:
    """Return a dict of topic → match score."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for topic, rules in keyword_rules.items():
        s = 0
        for rule in rules:
            try:
                s += len(re.findall(rule, text_lower)) * 2
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
    Classify all questions in *registry* using keyword heuristics.
    Modifies registry in-place.
    """
    for q in registry.get("questions", []):
        text = q.get("text_snippet", "")
        scores = _heuristic_score(text, keyword_rules)
        if scores:
            best = max(scores, key=lambda k: scores[k])
            if scores[best] >= fallback_score:
                q["topic"]      = best
                q["confidence"] = min(1.0, scores[best] / 20.0)
                continue
        q["topic"]      = "Unclassified"
        q["confidence"] = 0.0


# --------------------------------------------------------------------------- #
# Public API: batch classify                                                     #
# --------------------------------------------------------------------------- #
def classify_paper_batch(
    registry: dict,
    topics: list[str],
    syllabus_name: str,
    groq_key: str,
    keyword_rules: dict | None = None,
    confidence_threshold: float = 0.70,
    heuristic_fallback_score: int = 6,
) -> None:
    """
    Classify all questions in *registry* using Groq (batch) + heuristic fallback.
    Modifies *registry* in-place by setting "topic" and "confidence" on each question.

    Strategy:
      1. Build heuristic scores for all questions.
      2. Send one Groq batch call for the paper.
      3. Per question, merge results using the decision logic.
    """
    questions = registry.get("questions", [])
    if not questions:
        return

    kr = keyword_rules or {}

    # --- Step 1: heuristic scores -------------------------------------------
    h_scores: dict[int, tuple[str, int]] = {}
    for q in questions:
        text   = q.get("text_snippet", "")
        scores = _heuristic_score(text, kr)
        if scores:
            best = max(scores, key=lambda k: scores[k])
            h_scores[q["q_num"]] = (best, scores[best])
        else:
            h_scores[q["q_num"]] = ("Unclassified", 0)

    # --- Step 2: Groq batch call --------------------------------------------
    ai_results: dict[int, tuple[str, float]] = {}
    if groq_key:
        prompt = _build_batch_prompt(questions, topics, syllabus_name)
        raw    = _call_groq_batch(prompt, groq_key)
        if raw:
            q_nums      = [q["q_num"] for q in questions]
            ai_results  = _parse_batch_response(raw, topics, q_nums)
            logger.info(
                f"Groq classified {len(ai_results)}/{len(questions)} questions "
                f"for {registry['paper_id']}"
            )
        else:
            logger.warning(f"Groq call failed for {registry['paper_id']} — using heuristics only.")
    else:
        logger.info("No Groq key — using heuristics only.")

    # --- Step 3: decision logic per question --------------------------------
    for q in questions:
        qn = q["q_num"]
        ai_topic, ai_conf   = ai_results.get(qn, (None, 0.0))
        h_topic,  h_score   = h_scores.get(qn, ("Unclassified", 0))

        if ai_topic and ai_conf >= confidence_threshold:
            final, conf = ai_topic, ai_conf
        elif ai_topic and h_topic and ai_topic == h_topic:
            final, conf = ai_topic, ai_conf
        elif h_score >= heuristic_fallback_score:
            final, conf = h_topic, min(1.0, h_score / 20.0)
        elif ai_topic and ai_conf >= 0.5:
            final, conf = ai_topic, ai_conf
        elif h_score > 0:
            final, conf = h_topic, min(0.5, h_score / 20.0)
        else:
            final, conf = "Unclassified", 0.0

        q["topic"]      = final
        q["confidence"] = round(conf, 3)
        logger.debug(f"  Q{qn}: {final} ({conf:.2f})")
