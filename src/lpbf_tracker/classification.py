from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any

import requests


@dataclass
class ClassificationResult:
    industries: list[str]
    confidence: float
    rationale: str


def keyword_classify(text: str, keyword_rules: dict) -> ClassificationResult:
    text_lower = (text or "").lower()
    industries_section = (keyword_rules or {}).get("industries", {})

    industry_hits: dict[str, list[tuple[str, int]]] = {}
    total_hit_score = 0

    for industry, payload in industries_section.items():
        keywords = (payload or {}).get("keywords", [])
        matches: list[tuple[str, int]] = []

        for keyword in keywords:
            keyword_lower = str(keyword).lower()
            if not keyword_lower:
                continue
            count = text_lower.count(keyword_lower)
            if count:
                matches.append((str(keyword), int(count)))
                total_hit_score += int(count)

        if matches:
            industry_hits[str(industry)] = matches

    industries = sorted(industry_hits.keys())

    lpbf_anchors = (keyword_rules or {}).get(
        "lpbf_anchors",
        [
            "lpbf",
            "laser powder bed fusion",
            "additive manufacturing",
            "3d printing",
            "metal 3d printing",
        ],
    )
    anchor_hits = [term for term in lpbf_anchors if str(term).lower() in text_lower]

    if industries:
        # Heuristic confidence that remains stable and monotonic with stronger evidence.
        hit_strength = min(1.0, total_hit_score / 6.0)      # saturates after ~6 hits
        industry_strength = min(1.0, len(industries) / 4.0) # saturates after 4 industries
        anchor_bonus = 0.15 if anchor_hits else 0.0
        confidence = min(1.0, 0.2 + 0.5 * hit_strength + 0.15 * industry_strength + anchor_bonus)

        per_industry: list[str] = []
        for industry in industries:
            pretty_matches: list[str] = []
            for keyword, count in industry_hits[industry]:
                pretty_matches.append(f"{keyword} x{count}" if count > 1 else keyword)
            per_industry.append(f"{industry} ({', '.join(pretty_matches)})")

        rationale_parts = [f"Keyword matches: {', '.join(per_industry)}"]
        if anchor_hits:
            rationale_parts.append(f"LPBF anchors: {', '.join(anchor_hits)}")
        rationale = ". ".join(rationale_parts)
    else:
        confidence = 0.0
        rationale = "No keyword match"

    return ClassificationResult(industries=industries, confidence=float(confidence), rationale=rationale)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """
    Best-effort extraction of the first JSON object from an arbitrary string.
    Handles common LLM behaviour like wrapping JSON in prose or code fences.
    """
    if not text:
        return None

    # Remove common code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Fast path: pure JSON
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Fallback: find the first {...} block
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def llm_classify(text: str, settings: dict) -> ClassificationResult:
    api_key_env = settings["llm"]["api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        return ClassificationResult(industries=[], confidence=0.0, rationale="LLM not configured")

    prompt = (
        "Classify the company into relevant LPBF benefit industries based on the text. "
        "Return JSON with keys industries (array of strings), confidence (0-1), rationale (string). "
        "Return ONLY JSON."
    )

    try:
        response = requests.post(
            settings["llm"]["endpoint"],
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": settings["llm"]["model"],
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                "max_tokens": settings["llm"]["max_tokens"],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        return ClassificationResult(industries=[], confidence=0.0, rationale=f"LLM request failed: {exc}")

    payload = response.json()
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

    data = _extract_first_json_object(content)
    if not data:
        return ClassificationResult(industries=[], confidence=0.0, rationale="LLM output not valid JSON")

    industries_raw = data.get("industries", [])
    if isinstance(industries_raw, str):
        industries = [industries_raw]
    elif isinstance(industries_raw, list):
        industries = [str(x) for x in industries_raw if str(x).strip()]
    else:
        industries = []

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(data.get("rationale", "LLM classification")) if data.get("rationale") is not None else "LLM classification"

    return ClassificationResult(industries=industries, confidence=confidence, rationale=rationale)
