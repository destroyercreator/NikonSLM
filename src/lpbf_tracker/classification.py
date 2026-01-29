from __future__ import annotations

from dataclasses import dataclass
import json
import os
import requests
from typing import Any


@dataclass
class ClassificationResult:
    industries: list[str]
    confidence: float
    rationale: str


def keyword_classify(text: str, keyword_rules: dict) -> ClassificationResult:
    text_lower = text.lower()
    hits = []
    for industry, payload in keyword_rules["industries"].items():
        keywords = payload.get("keywords", [])
        for keyword in keywords:
            if keyword in text_lower:
                hits.append(industry)
                break
    industries = sorted(set(hits))
    confidence = min(1.0, 0.4 + 0.1 * len(industries)) if industries else 0.0
    rationale = "Keyword match" if industries else "No keyword match"
    return ClassificationResult(industries=industries, confidence=confidence, rationale=rationale)


def _clamp_confidence(confidence: float) -> float:
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _extract_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _validate_llm_payload(data: dict[str, Any]) -> ClassificationResult | None:
    industries = data.get("industries", [])
    if not isinstance(industries, list) or not all(isinstance(item, str) for item in industries):
        return None
    confidence = data.get("confidence", 0.0)
    try:
        confidence_value = _clamp_confidence(float(confidence))
    except (TypeError, ValueError):
        return None
    rationale = data.get("rationale", "LLM classification")
    if not isinstance(rationale, str):
        return None
    return ClassificationResult(
        industries=industries,
        confidence=confidence_value,
        rationale=rationale,
    )


def _extract_message_content(payload: dict[str, Any]) -> str:
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
        "Return JSON with keys industries (array), confidence (0-1), rationale (string)."
    )
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
    payload = response.json()
    try:
        content = _extract_message_content(payload)
    except KeyError:
        return ClassificationResult(
            industries=[],
            confidence=0.0,
            rationale="LLM response format unsupported",
        )
    data = _extract_json_object(content)
    if data is None:
        return ClassificationResult(
            industries=[],
            confidence=0.0,
            rationale="LLM output not JSON",
        )
    result = _validate_llm_payload(data)
    if result is None:
        return ClassificationResult(
            industries=[],
            confidence=0.0,
            rationale="LLM output schema invalid",
        )
    return result
