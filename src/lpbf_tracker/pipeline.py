from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import yaml

from lpbf_tracker.classification import keyword_classify, llm_classify
from lpbf_tracker.config import Config
from lpbf_tracker.enrichment import enrich_contacts
from lpbf_tracker.location import extract_location
from lpbf_tracker.search import build_provider, build_queries
from lpbf_tracker.storage import CompanyRecord, canonical_domain, load_crm, save_crm, upsert_record


def load_keyword_rules(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def to_homepage(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return url


def run_pipeline(config: Config) -> None:
    settings = config.raw
    provider = build_provider(settings["search"])
    keyword_rules = load_keyword_rules(Path(settings["classification"]["keyword_rules_path"]))
    crm_path = Path(settings["project"]["output_excel"])
    df = load_crm(crm_path)
    user_agent = settings["project"]["user_agent"]
    contact_settings = settings["contact_enrichment"]

    queries = list(build_queries(settings))
    total_queries = len(queries)
    logging.info("Starting pipeline with %d queries.", total_queries)
    best_by_domain: dict[str, dict[str, object]] = {}

    for query_index, query in enumerate(queries, start=1):
        logging.info("Running query %d/%d: %s", query_index, total_queries, query)
        results = provider.search(query, settings["project"]["max_results_per_query"])
        total_results = len(results)
        logging.info("Query %d returned %d results.", query_index, total_results)
        for result_index, result in enumerate(results, start=1):
            logging.info(
                "Processing result %d/%d for query %d: %s",
                result_index,
                total_results,
                query_index,
                result.title,
            )
            domain = canonical_domain(result.url)
            if not domain:
                logging.info("Skipping result with missing domain: %s", result.url)
                continue
            combined_text = " ".join([result.title, result.snippet])
            keyword_result = keyword_classify(combined_text, keyword_rules)
            industries = keyword_result.industries
            confidence = keyword_result.confidence
            rationale = keyword_result.rationale
            if settings["classification"].get("use_llm"):
                llm_result = llm_classify(combined_text, settings["classification"])
                if llm_result.confidence >= confidence:
                    industries = llm_result.industries
                    confidence = llm_result.confidence
                    rationale = llm_result.rationale

            if confidence < settings["classification"]["min_confidence"]:
                logging.info(
                    "Skipping %s due to confidence %.2f below threshold %.2f.",
                    result.url,
                    confidence,
                    settings["classification"]["min_confidence"],
                )
                continue

            city, province = extract_location(combined_text)
            candidate = {
                "name": result.title,
                "website": result.url,
                "domain": domain,
                "city": city,
                "province": province,
                "industries": industries,
                "confidence": confidence,
                "rationale": rationale,
                "evidence_snippet": result.snippet,
                "source_url": result.url,
            }
            existing = best_by_domain.get(domain)
            if existing is None or confidence > existing["confidence"]:
                best_by_domain[domain] = candidate

    logging.info("Enriching %d unique domains.", len(best_by_domain))
    for domain, candidate in best_by_domain.items():
        homepage = to_homepage(candidate["website"])
        logging.info("Enriching contacts for %s", homepage)
        contact_info = enrich_contacts(
            base_url=homepage,
            user_agent=user_agent,
            contact_keywords=contact_settings["contact_page_keywords"],
            max_pages=contact_settings["max_pages_per_company"],
        )
        logging.info("Enrichment complete for %s", homepage)
        record = CompanyRecord(
            name=candidate["name"],
            website=candidate["website"],
            domain=domain,
            city=candidate["city"],
            province=candidate["province"],
            industries=candidate["industries"],
            confidence=candidate["confidence"],
            rationale=candidate["rationale"],
            evidence_snippet=candidate["evidence_snippet"],
            source_url=candidate["source_url"],
            contact_emails=contact_info.emails,
            contact_phones=contact_info.phones,
            contact_page=contact_info.contact_page,
            staff=contact_info.staff,
        )
        df = upsert_record(df, record, settings["crm"]["fuzzy_match_threshold"])

    logging.info("Saving CRM output to %s", crm_path)
    save_crm(df, crm_path)
    logging.info("Pipeline complete.")
