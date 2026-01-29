from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import yaml

from lpbf_tracker.cache import ContactCache
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


def _companyrecord_supports_address() -> bool:
    fields = getattr(CompanyRecord, "__dataclass_fields__", None)
    return bool(fields and "address" in fields)


def _domain_in_index(index: object, domain: str) -> bool:
    """
    Best-effort check whether a domain already exists in the CRM index.
    This avoids importing match_existing() and keeps metrics cheap.

    Adjust if your build_crm_index() returns something different.
    """
    if not domain:
        return False

    # Common patterns:
    if isinstance(index, dict):
        if domain in index:
            return True
        # Some indexes store normalized keys
        if domain.lower() in index:
            return True

    # Fallback: duck-typed container
    try:
        return domain in index  # type: ignore[operator]
    except Exception:
        return False


def _append_run_log(settings: dict, row: dict) -> None:
    run_log_path = Path(settings["project"].get("run_log_path", "data/run_log.csv"))
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = run_log_path.exists()

    with run_log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "queries",
                "results",
                "domains",
                "unique_candidates",
                "enriched",
                "inserted",
                "updated",
                "skipped_confidence",
                "contact_cache_hits",
                "contact_cache_misses",
                "contact_cache_hit_rate",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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

    # Persistent contact cache (optional)
    persistent_cache: ContactCache | None = None
    if contact_settings.get("enabled", True):
        cache_dir = Path(contact_settings.get("cache_dir", "data/contact_cache"))
        cache_ttl_hours = float(contact_settings.get("cache_ttl_hours", 24))
        persistent_cache = ContactCache(cache_dir, timedelta(hours=cache_ttl_hours))

    queries = list(build_queries(settings))
    total_queries = len(queries)

    # Run metrics
    total_results = 0
    domains_seen: set[str] = set()
    skipped_confidence = 0
    enriched_count = 0
    inserted_count = 0
    updated_count = 0
    contact_cache_hits = 0
    contact_cache_misses = 0

    logging.info("Starting pipeline with %d queries.", total_queries)

    # Keep only the best candidate per domain before enrichment/upsert.
    best_by_domain: dict[str, dict[str, object]] = {}

    for query_index, query in enumerate(queries, start=1):
        logging.info("Running query %d/%d: %s", query_index, total_queries, query)
        results = provider.search(query, settings["project"]["max_results_per_query"])
        total_results += len(results)
        logging.info("Query %d returned %d results.", query_index, len(results))

        for result_index, result in enumerate(results, start=1):
            logging.info(
                "Processing result %d/%d for query %d: %s",
                result_index,
                len(results),
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

            llm_settings = settings["classification"]
            if llm_settings.get("use_llm"):
                llm_reasons: list[str] = []

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

            min_conf = float(settings["classification"]["min_confidence"])
            if confidence < min_conf:
                logging.info(
                    "Skipping %s due to confidence %.2f below threshold %.2f.",
                    result.url,
                    confidence,
                    min_conf,
                )
                skipped_confidence += 1
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
            # ContactCache hit-rate tracking
            if persistent_cache is not None:
                cached = persistent_cache.get(domain)
                if cached:
                    contact_cache_hits += 1
                    contact_info = cached
                else:
                    contact_cache_misses += 1

            if contact_info is None:
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
                    cache=persistent_cache,
                    domain=domain,
                )
                logging.info("Enrichment complete for %s", homepage)

            enriched_count += 1
        else:
            logging.info("Skipping contact enrichment for %s", homepage)

        company_name = str(candidate["name"])
        company_address = None
        if contact_info is not None:
            if contact_info.company_name and contact_info.company_name.strip():
                company_name = contact_info.company_name.strip()
            company_address = contact_info.company_address

        record_kwargs = dict(
            name=company_name,
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

        if company_address is not None and _companyrecord_supports_address():
            record_kwargs["address"] = company_address

        record = CompanyRecord(**record_kwargs)

        existed_pre = _domain_in_index(crm_index, domain)
        df = upsert_record(
            df,
            record,
            settings["crm"]["fuzzy_match_threshold"],
            index=crm_index,
        )
        if existed_pre:
            updated_count += 1
        else:
            inserted_count += 1

    logging.info("Saving CRM output to %s", crm_path)
    save_crm(df, crm_path)

    total_enrichment_requests = contact_cache_hits + contact_cache_misses
    cache_hit_rate = (
        contact_cache_hits / total_enrichment_requests if total_enrichment_requests else 0.0
    )

    logging.info(
        "Run summary: queries=%d results=%d domains_seen=%d unique_candidates=%d enriched=%d inserted=%d updated=%d "
        "skipped_confidence=%d contact_cache_hit_rate=%.1f%%",
        total_queries,
        total_results,
        len(domains_seen),
        len(best_by_domain),
        enriched_count,
        inserted_count,
        updated_count,
        skipped_confidence,
        cache_hit_rate * 100,
    )

    _append_run_log(
        settings,
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "queries": total_queries,
            "results": total_results,
            "domains": len(domains_seen),
            "unique_candidates": len(best_by_domain),
            "enriched": enriched_count,
            "inserted": inserted_count,
            "updated": updated_count,
            "skipped_confidence": skipped_confidence,
            "contact_cache_hits": contact_cache_hits,
            "contact_cache_misses": contact_cache_misses,
            "contact_cache_hit_rate": round(cache_hit_rate, 4),
        },
    )

    logging.info("Pipeline complete.")
