from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml

from lpbf_tracker.classification import keyword_classify, llm_classify
from lpbf_tracker.config import Config
from lpbf_tracker.enrichment import enrich_contacts
from lpbf_tracker.location import extract_location
from lpbf_tracker.search import build_provider, build_queries
from lpbf_tracker.storage import (
    CompanyRecord,
    canonical_domain,
    load_crm,
    match_existing,
    save_crm,
    upsert_record,
)


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
    total_results = 0
    domains_seen: set[str] = set()
    enriched_count = 0
    inserted_count = 0
    updated_count = 0
    skipped_confidence = 0
    cache_hits = 0
    cache_misses = 0
    enrichment_cache = {}
    logging.info("Starting pipeline with %d queries.", total_queries)

    for query_index, query in enumerate(queries, start=1):
        logging.info("Running query %d/%d: %s", query_index, total_queries, query)
        results = provider.search(query, settings["project"]["max_results_per_query"])
        query_results = len(results)
        total_results += query_results
        logging.info("Query %d returned %d results.", query_index, query_results)
        for result_index, result in enumerate(results, start=1):
            logging.info(
                "Processing result %d/%d for query %d: %s",
                result_index,
                query_results,
                query_index,
                result.title,
            )
            domain = canonical_domain(result.url)
            if not domain:
                logging.info("Skipping result with missing domain: %s", result.url)
                continue
            domains_seen.add(domain)
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
                skipped_confidence += 1
                continue

            city, province = extract_location(combined_text)
            homepage = to_homepage(result.url)
            if homepage in enrichment_cache:
                contact_info = enrichment_cache[homepage]
                cache_hits += 1
                logging.info("Using cached enrichment for %s", homepage)
            else:
                logging.info("Enriching contacts for %s", homepage)
                contact_info = enrich_contacts(
                    base_url=homepage,
                    user_agent=user_agent,
                    contact_keywords=contact_settings["contact_page_keywords"],
                    max_pages=contact_settings["max_pages_per_company"],
                )
                enrichment_cache[homepage] = contact_info
                cache_misses += 1
                logging.info("Enrichment complete for %s", homepage)
            enriched_count += 1

            record = CompanyRecord(
                name=result.title,
                website=result.url,
                domain=domain,
                city=city,
                province=province,
                industries=industries,
                confidence=confidence,
                rationale=rationale,
                evidence_snippet=result.snippet,
                source_url=result.url,
                contact_emails=contact_info.emails,
                contact_phones=contact_info.phones,
                contact_page=contact_info.contact_page,
                staff=contact_info.staff,
            )
            existing_idx = match_existing(df, record, settings["crm"]["fuzzy_match_threshold"])
            if existing_idx is None:
                inserted_count += 1
            else:
                updated_count += 1
            df = upsert_record(df, record, settings["crm"]["fuzzy_match_threshold"])

    logging.info("Saving CRM output to %s", crm_path)
    save_crm(df, crm_path)
    total_enrichment_requests = cache_hits + cache_misses
    cache_hit_rate = (
        cache_hits / total_enrichment_requests if total_enrichment_requests else 0.0
    )
    logging.info(
        "Run summary: queries=%d results=%d domains=%d enriched=%d inserted=%d updated=%d skipped_confidence=%d cache_hit_rate=%.1f%%",
        total_queries,
        total_results,
        len(domains_seen),
        enriched_count,
        inserted_count,
        updated_count,
        skipped_confidence,
        cache_hit_rate * 100,
    )
    run_log_path = Path(settings["project"].get("run_log_path", "data/run_log.csv"))
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_exists = run_log_path.exists()
    with run_log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "queries",
                "results",
                "domains",
                "enriched",
                "inserted",
                "updated",
                "skipped_confidence",
                "cache_hit_rate",
            ],
        )
        if not log_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "queries": total_queries,
                "results": total_results,
                "domains": len(domains_seen),
                "enriched": enriched_count,
                "inserted": inserted_count,
                "updated": updated_count,
                "skipped_confidence": skipped_confidence,
                "cache_hit_rate": round(cache_hit_rate, 4),
            }
        )
    logging.info("Pipeline complete.")
