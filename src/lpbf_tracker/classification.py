from __future__ import annotations

from dataclasses import dataclass
import json
import os
import requests


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
    if industries:
        rationale = f"Keyword match for {len(industries)} industr{'y' if len(industries) == 1 else 'ies'}"
    else:
        rationale = "No keyword match"
    return ClassificationResult(industries=industries, confidence=confidence, rationale=rationale)


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
    content = payload["choices"][0]["message"]["content"]
    try:
        data: dict = json.loads(content)
    except json.JSONDecodeError:
        return ClassificationResult(industries=[], confidence=0.0, rationale="LLM output not JSON")
    industries = data.get("industries", [])
    confidence = float(data.get("confidence", 0.0))
    rationale = data.get("rationale", "LLM classification")
    return ClassificationResult(industries=industries, confidence=confidence, rationale=rationale)
