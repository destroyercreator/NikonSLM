from __future__ import annotations

from dataclasses import dataclass
import json
import os
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
    anchor_hits = [str(term) for term in lpbf_anchors if str(term).lower() in text_lower]

    if industries:
        # Heuristic confidence: stable and monotonic with stronger evidence.
        hit_strength = min(1.0, total_hit_score / 6.0)       # saturates after ~6 hits
        industry_strength = min(1.0, len(industries) / 4.0)  # saturates after 4 industries
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


def _clamp_confidence(confidence: float) -> float:
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract the first JSON object from an arbitrary string.
    Supports: pure JSON, or JSON embedded in prose.
    """
    if not text:
        return None

    # Fast path: pure JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Fallback: scan for a balanced {...} region and attempt to parse it.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None

    return None


def _validate_llm_payload(data: dict[str, Any]) -> ClassificationResult | None:
    industries = data.get("industries", [])
    if isinstance(industries, str):
        industries_list = [industries]
    elif isinstance(industries, list):
        industries_list = [str(item) for item in industries if isinstance(item, (str, int, float)) and str(item).strip()]
    else:
        return None

    confidence = data.get("confidence", 0.0)
    try:
        confidence_value = _clamp_confidence(float(confidence))
    except (TypeError, ValueError):
        return None

    rationale = data.get("rationale", "LLM classification")
    if not isinstance(rationale, str):
        rationale = str(rationale)

    return ClassificationResult(
        industries=industries_list,
        confidence=confidence_value,
        rationale=rationale,
    )


def _extract_message_content(payload: dict[str, Any]) -> str:
    """
    Support multiple response shapes:
    - Chat Completions: payload["choices"][0]["message"]["content"]
    - Responses-style: payload["output"][0]["content"][0]["text"]
    """
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content

    output = payload.get("output")
    if isinstance(output, list) and output:
        content_blocks = output[0].get("content")
        if isinstance(content_blocks, list) and content_blocks:
            text = content_blocks[0].get("text")
            if isinstance(text, str):
                return text

    raise KeyError("Unsupported LLM response format")


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

    try:
        content = _extract_message_content(payload)
    except KeyError:
        return ClassificationResult(industries=[], confidence=0.0, rationale="LLM response format unsupported")

    data = _extract_json_object(content)
    if data is None:
        return ClassificationResult(industries=[], confidence=0.0, rationale="LLM output not JSON")

    result = _validate_llm_payload(data)
    if result is None:
        return ClassificationResult(industries=[], confidence=0.0, rationale="LLM output schema invalid")

    return result
