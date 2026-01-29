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
    industry_hits: dict[str, list[tuple[str, int]]] = {}
    total_hit_score = 0
    for industry, payload in keyword_rules["industries"].items():
        keywords = payload.get("keywords", [])
        matches: list[tuple[str, int]] = []
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = text_lower.count(keyword_lower)
            if count:
                matches.append((keyword, count))
                total_hit_score += count
        if matches:
            industry_hits[industry] = matches

    industries = sorted(industry_hits.keys())
    lpbf_anchors = keyword_rules.get(
        "lpbf_anchors",
        [
            "lpbf",
            "laser powder bed fusion",
            "additive manufacturing",
            "3d printing",
            "metal 3d printing",
        ],
    )
    anchor_hits = [term for term in lpbf_anchors if term in text_lower]
    if industries:
        hit_strength = min(1.0, total_hit_score / 6)
        industry_strength = min(1.0, len(industries) / 4)
        anchor_bonus = 0.15 if anchor_hits else 0.0
        confidence = min(1.0, 0.2 + 0.5 * hit_strength + 0.15 * industry_strength + anchor_bonus)
    else:
        confidence = 0.0

    if industries:
        per_industry = []
        for industry in industries:
            matches = []
            for keyword, count in industry_hits[industry]:
                if count > 1:
                    matches.append(f"{keyword} x{count}")
                else:
                    matches.append(keyword)
            per_industry.append(f"{industry} ({', '.join(matches)})")
        rationale_parts = [f"Keyword matches: {', '.join(per_industry)}"]
        if anchor_hits:
            rationale_parts.append(f"LPBF anchors: {', '.join(anchor_hits)}")
        rationale = ". ".join(rationale_parts)
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
