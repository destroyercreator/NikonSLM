from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import yaml

from lpbf_tracker.classification import keyword_classify, llm_classify
from lpbf_tracker.config import Config
from lpbf_tracker.enrichment import ContactInfo, enrich_contacts
from lpbf_tracker.location import extract_location
from lpbf_tracker.search import build_provider, build_queries
from lpbf_tracker.storage import (
    CompanyRecord,
    build_crm_index,
    canonical_domain,
    load_crm,
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
    crm_index = build_crm_index(df)

    user_agent = settings["project"]["user_agent"]
    contact_settings = settings["contact_enrichment"]
    save_every_query = settings["project"].get("save_every_query", False)

    # In-run cache to avoid re-crawling the same domain multiple times.
    contact_cache: dict[str, ContactInfo] = {}

    queries = list(build_queries(settings))
    total_queries = len(queries)
    logging.info("Starting pipeline with %d queries.", total_queries)

    # Keep only the best candidate per domain before enrichment/upsert.
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

            llm_settings = settings["classification"]
            if llm_settings.get("use_llm"):
                llm_reasons: list[str] = []

                # Reasons to invoke LLM:
                if not industries:
                    llm_reasons.append("no keyword industries")
                if len(industries) > 1:
                    llm_reasons.append("keyword ambiguity")

                min_llm_conf = float(llm_settings.get("min_llm_conf", 0.0))
                max_llm_conf = float(llm_settings.get("max_llm_conf", 1.0))
                if min_llm_conf <= confidence <= max_llm_conf:
                    llm_reasons.append(
                        f"keyword confidence {confidence:.2f} within [{min_llm_conf:.2f}, {max_llm_conf:.2f}]"
                    )

                if llm_reasons:
                    logging.info(
                        "Invoking LLM classification for %s because %s.",
                        result.url,
                        "; ".join(llm_reasons),
                    )
                    llm_result = llm_classify(combined_text, llm_settings)
                    if llm_result.confidence >= confidence:
                        industries = llm_result.industries
                        confidence = llm_result.confidence
                        rationale = llm_result.rationale
                else:
                    logging.info(
                        "Skipping LLM classification for %s because keyword confidence %.2f is decisive.",
                        result.url,
                        confidence,
                    )


            min_conf = settings["classification"]["min_confidence"]
            if confidence < min_conf:
                logging.info(
                    "Skipping %s due to confidence %.2f below threshold %.2f.",
                    result.url,
                    confidence,
                    min_conf,
                )
                continue

            city, province = extract_location(combined_text)
            homepage = to_homepage(result.url)

            candidate = {
                "name": result.title or "",
                "website": homepage,
                "city": city or "",
                "province": province or "",
                "industries": list(industries) if industries else [],
                "confidence": float(confidence),
                "rationale": rationale or "",
                "evidence_snippet": result.snippet or "",
                "source_url": result.url,
            }

            existing = best_by_domain.get(domain)
            if existing is None:
                best_by_domain[domain] = candidate
            else:
                # Primary: higher confidence. Secondary: longer evidence snippet.
                if float(candidate["confidence"]) > float(existing["confidence"]):
                    best_by_domain[domain] = candidate
                elif float(candidate["confidence"]) == float(existing["confidence"]):
                    if len(str(candidate.get("evidence_snippet", ""))) > len(
                        str(existing.get("evidence_snippet", ""))
                    ):
                        best_by_domain[domain] = candidate

    if save_every_query:
        logging.info(
            "save_every_query is enabled, but pipeline is running in batch (best-by-domain) mode; intermediate saves are skipped."
        )

    logging.info("Enriching %d unique domains.", len(best_by_domain))

    for domain, candidate in best_by_domain.items():
        homepage = str(candidate["website"])

        contact_info: ContactInfo | None = None
        if contact_settings.get("enabled", True):
            if domain in contact_cache:
                logging.info("Contact enrichment cache hit for %s", domain)
                contact_info = contact_cache[domain]
            else:
                contact_keywords = contact_settings.get(
                    "contact_page_keywords", ["contact", "about", "team"]
                )
                max_pages = int(contact_settings.get("max_pages_per_company", 5))

                logging.info("Enriching contacts for %s", homepage)
                contact_info = enrich_contacts(
                    base_url=homepage,
                    user_agent=user_agent,
                    contact_keywords=contact_keywords,
                    max_pages=max_pages,
                )
                contact_cache[domain] = contact_info
                logging.info("Enrichment complete for %s", homepage)
        else:
            logging.info("Skipping contact enrichment for %s", homepage)

        record = CompanyRecord(
            name=str(candidate["name"]),
            website=homepage,
            domain=domain,
            city=str(candidate["city"]),
            province=str(candidate["province"]),
            industries=list(candidate["industries"]),
            confidence=float(candidate["confidence"]),
            rationale=str(candidate["rationale"]),
            evidence_snippet=str(candidate["evidence_snippet"]),
            source_url=str(candidate["source_url"]),
            contact_emails=contact_info.emails if contact_info else [],
            contact_phones=contact_info.phones if contact_info else [],
            contact_page=contact_info.contact_page if contact_info else None,
            staff=contact_info.staff if contact_info else [],
        )

        df = upsert_record(
            df,
            record,
            settings["crm"]["fuzzy_match_threshold"],
            index=crm_index,
        )

    logging.info("Saving CRM output to %s", crm_path)
    save_crm(df, crm_path)
    logging.info("Pipeline complete.")
