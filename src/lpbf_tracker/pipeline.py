# src/lpbf_tracker/pipeline.py
from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
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

logger = logging.getLogger(__name__)


# -----------------------------
# Lead-quality filtering config
# -----------------------------

# Hard excludes (very high noise / not "company sites")
_EXCLUDED_DOMAINS = {
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "x.com",
    "twitter.com",
    "www.twitter.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "linkedin.com",
    "www.linkedin.com",
    "medium.com",
    "www.medium.com",
    "wikipedia.org",
    "en.wikipedia.org",
    "github.com",
    "www.github.com",
    "gitlab.com",
    "www.gitlab.com",
    "stackexchange.com",
    "stackoverflow.com",
}

# Domain suffix exclusions (frequently institutions, directories, etc.)
_EXCLUDED_TLDS = (".gov", ".edu", ".ac.uk", ".gc.ca")

# URL path patterns that are usually non-leads even on a valid domain
_EXCLUDED_PATH_PATTERNS = (
    r"/r/[^/]+",          # reddit communities
    r"/forums?/",         # forums
    r"/community/",       # community hubs
    r"/questions?/",      # Q&A pages
    r"/tag/[^/]+",        # tag pages
    r"/wiki/",            # wiki
)

# Non-company / community signals (snippet/title)
_NON_COMPANY_TERMS = (
    "reddit",
    "subreddit",
    "thread",
    "forum",
    "stack overflow",
    "stackexchange",
    "quora",
    "wikipedia",
    "github",
    "issue",
    "pull request",
    "sign in",
    "log in",
)

# Company-site signals (snippet/title)
_COMPANY_SITE_TERMS = (
    "about",
    "about us",
    "contact",
    "contact us",
    "careers",
    "privacy",
    "terms",
    "iso 9001",
    "as9100",
    "nadcap",
    "capabilities",
    "services",
    "manufacturing",
    "company",
    "our team",
)

# LPBF fit terms (snippet/title)
_LPBF_TERMS = (
    "lpbf",
    "laser powder bed fusion",
    "powder bed fusion",
    "pbf",
    "slm",
    "selective laser melting",
    "dmls",
    "direct metal laser sintering",
    "metal additive",
    "additive manufacturing",
    "metal 3d printing",
    "3d printed metal",
    "inconel",
    "titanium",
    "ti-6al-4v",
    "aluminium",
    "aluminum",
    "hastelloy",
    "maraging",
    "316l",
    "718",
    "625",
)

# If you want to bias toward “buyers” vs “bureaus”, include end-market terms
_END_MARKET_TERMS = (
    "aerospace",
    "space",
    "defence",
    "defense",
    "medical",
    "orthopedic",
    "motorsport",
    "automotive",
    "turbine",
    "propulsion",
    "heat exchanger",
    "manifold",
)

_CONTENT_LIKELIHOOD_MARKERS = (
    "thread",
    "comments",
    "upvotes",
    "posted by",
    "subscribe",
    "press release",
    "news",
    "blog",
    "article",
    "forum",
)


def load_keyword_rules(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def to_homepage(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return url


def url_path_depth(url: str) -> int:
    parsed = urlparse(url)
    path = (parsed.path or "/").strip("/")
    if not path:
        return 0
    return len([segment for segment in path.split("/") if segment])


def _companyrecord_fields() -> set[str]:
    # Works for dataclasses and pydantic-like models that expose __fields__/model_fields
    try:
        return {f.name for f in dataclass_fields(CompanyRecord)}  # type: ignore[arg-type]
    except Exception:
        pass

    f = getattr(CompanyRecord, "__dataclass_fields__", None)
    if isinstance(f, dict):
        return set(f.keys())

    f = getattr(CompanyRecord, "model_fields", None)
    if isinstance(f, dict):
        return set(f.keys())

    f = getattr(CompanyRecord, "__fields__", None)
    if isinstance(f, dict):
        return set(f.keys())

    return set()


def _companyrecord_requires_address() -> bool:
    # Conservative: if field exists, assume it’s required unless we can prove a default.
    # If your CompanyRecord is a dataclass, dataclass_fields() gives defaults; otherwise we just treat as required.
    try:
        for f in dataclass_fields(CompanyRecord):  # type: ignore[arg-type]
            if f.name == "address":
                has_default = f.default is not f.default_factory or f.default_factory is not None  # type: ignore
                # The above line is a bit awkward for dataclasses; easiest is:
                # if default is dataclasses.MISSING and default_factory is dataclasses.MISSING -> required.
                # But we’re avoiding importing dataclasses here; simplest: treat as required if no obvious default.
                # We'll do a more reliable check with repr below:
                return True
    except Exception:
        pass

    return "address" in _companyrecord_fields()


def _domain_in_index(index: object, domain: str) -> bool:
    if not domain:
        return False
    if isinstance(index, dict):
        return domain in index or domain.lower() in index
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
                "skipped_excluded_domain",
                "skipped_content_type",
                "skipped_low_company_likelihood",
                "skipped_low_lpbf_fit",
                "contact_cache_hits",
                "contact_cache_misses",
                "contact_cache_hit_rate",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def is_excluded_domain(domain: str) -> bool:
    d = (domain or "").strip().lower()
    if not d:
        return True

    if d in _EXCLUDED_DOMAINS:
        return True

    hard_block_substrings = (
        "reddit",
        "stackexchange",
        "quora",
        "wikipedia",
        "medium",
        "substack",
    )
    if any(term in d for term in hard_block_substrings):
        return True

    for tld in _EXCLUDED_TLDS:
        if d.endswith(tld):
            return True

    return False


def is_excluded_url(url: str) -> bool:
    try:
        path = urlparse(url).path or ""
    except Exception:
        return False

    p = path.lower()
    for pat in _EXCLUDED_PATH_PATTERNS:
        if re.search(pat, p):
            return True
    return False


def _norm_text(*parts: str) -> str:
    s = " ".join([p for p in parts if p])
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def company_likelihood_score(text: str, domain: str, url: str) -> int:
    """
    Score intended to reject obvious non-company pages (forums, social, etc.)
    Return value: integer score; <1 means reject.
    """
    if is_excluded_domain(domain):
        return -10
    if is_excluded_url(url):
        return -5

    t = (text or "").lower()
    score = 0

    # Penalties
    if any(term in t for term in _NON_COMPANY_TERMS):
        score -= 3

    # Rewards
    if any(term in t for term in _COMPANY_SITE_TERMS):
        score += 2

    # If it looks like a directory/listing page, downweight
    if "directory" in t or "list of" in t or "top " in t:
        score -= 1

    return score


def lpbf_fit_score(text: str) -> int:
    """
    Score intended to push up LPBF/metal-AM relevance.
    Return value: integer score; <1 means reject.
    """
    t = (text or "").lower()
    score = 0

    if any(term in t for term in _LPBF_TERMS):
        score += 2

    # End-market terms help, but not sufficient alone
    if any(term in t for term in _END_MARKET_TERMS):
        score += 1

    # Generic “engineering” alone is too broad; no boost.

    return score


def content_likelihood_score(text: str) -> int:
    """
    Score intended to detect content-language markers (news/blog/forum).
    Return value: integer score; >= threshold means reject.
    """
    t = (text or "").lower()
    return sum(1 for marker in _CONTENT_LIKELIHOOD_MARKERS if marker in t)


def run_pipeline(config: Config) -> None:
    settings = config.raw

    search_settings = settings["search"]
    provider_name = search_settings.get("provider", "serpapi")
    provider = build_provider(search_settings)
    keyword_rules = load_keyword_rules(Path(settings["classification"]["keyword_rules_path"]))

    crm_path = Path(settings["project"]["output_excel"])
    df = load_crm(crm_path)
    crm_index = build_crm_index(df)

    user_agent = settings["project"]["user_agent"]
    contact_settings = settings.get("contact_enrichment", {})
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
    skipped_excluded_domain = 0
    skipped_content_type = 0
    skipped_low_company_likelihood = 0
    skipped_low_lpbf_fit = 0
    enriched_count = 0
    inserted_count = 0
    updated_count = 0
    contact_cache_hits = 0
    contact_cache_misses = 0
    excluded_domain_counts: Counter[str] = Counter()

    logging.info("Starting pipeline with %d queries.", total_queries)

    # Keep only the best candidate per domain before enrichment/upsert.
    best_by_domain: dict[str, dict[str, object]] = {}

    max_results = int(settings["project"]["max_results_per_query"])
    max_results_overrides = search_settings.get("max_results_per_query_overrides", {})
    if provider_name in max_results_overrides:
        max_results = int(max_results_overrides[provider_name])
    min_conf = float(settings["classification"]["min_confidence"])
    content_reject_threshold = 2

    for query_index, query in enumerate(queries, start=1):
        logging.info("Running query %d/%d: %s", query_index, total_queries, query)
        results = provider.search(query, max_results)
        total_results += len(results)
        logging.info("Query %d returned %d results.", query_index, len(results))

        for result_index, result in enumerate(results, start=1):
            logging.info(
                "Processing result %d/%d for query %d: %s",
                result_index,
                len(results),
                query_index,
                getattr(result, "title", ""),
            )

            url = getattr(result, "url", "") or ""
            title = getattr(result, "title", "") or ""
            snippet = getattr(result, "snippet", "") or ""
            depth = url_path_depth(url)

            domain = canonical_domain(url)
            if not domain:
                logging.info("Skipping result with missing domain: %s", url)
                continue

            if is_excluded_domain(domain):
                logging.info("Skipping excluded domain: %s", domain)
                skipped_excluded_domain += 1
                excluded_domain_counts[domain] += 1
                continue

            if is_excluded_url(url):
                logging.info("Skipping excluded URL pattern: %s", url)
                skipped_content_type += 1
                continue

            domains_seen.add(domain)

            combined_text = _norm_text(title, snippet)

            if company_likelihood_score(combined_text, domain, url) < 1:
                logging.info("Low company-likelihood score, skipping %s (%s)", domain, url)
                skipped_low_company_likelihood += 1
                continue

            if lpbf_fit_score(combined_text) < 1:
                logging.info("Low LPBF-fit score, skipping %s (%s)", domain, url)
                skipped_low_lpbf_fit += 1
                continue

            if content_likelihood_score(combined_text) >= content_reject_threshold:
                logging.info(
                    "Content-language markers detected, skipping %s (%s).",
                    domain,
                    url,
                )
                skipped_content_type += 1
                continue

            # Classification (keyword first, LLM optionally)
            keyword_result = keyword_classify(combined_text, keyword_rules)
            industries = keyword_result.industries
            confidence = float(keyword_result.confidence)
            rationale = keyword_result.rationale

            llm_settings = settings.get("classification", {})
            if llm_settings.get("use_llm"):
                llm_reasons: list[str] = []

                if not industries:
                    llm_reasons.append("no keyword industries")
                if industries and len(industries) > 1:
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
                        url,
                        "; ".join(llm_reasons),
                    )
                    llm_result = llm_classify(combined_text, llm_settings)
                    if float(llm_result.confidence) >= confidence:
                        industries = llm_result.industries
                        confidence = float(llm_result.confidence)
                        rationale = llm_result.rationale
                else:
                    logging.info("Skipping LLM classification for %s (confidence %.2f decisive).", url, confidence)

            if confidence < min_conf:
                logging.info(
                    "Skipping %s due to confidence %.2f below threshold %.2f.",
                    url,
                    confidence,
                    min_conf,
                )
                skipped_confidence += 1
                continue

            url_depth_penalty = 1.0 if depth <= 1 else 0.8
            selection_score = confidence * url_depth_penalty

            city, province = extract_location(combined_text)
            homepage = to_homepage(url)

            candidate = {
                "name": title or "",
                "website": homepage,
                "city": city or "",
                "province": province or "",
                "industries": list(industries) if industries else [],
                "confidence": float(confidence),
                "rationale": rationale or "",
                "evidence_snippet": snippet or "",
                "source_url": url,
                "url_depth": depth,
                "url_depth_penalty": url_depth_penalty,
                "selection_score": selection_score,
            }

            existing = best_by_domain.get(domain)
            if existing is None:
                best_by_domain[domain] = candidate
            else:
                # Primary: higher selection score. Secondary: lower path depth. Tertiary: longer evidence snippet.
                if float(candidate["selection_score"]) > float(existing["selection_score"]):
                    best_by_domain[domain] = candidate
                elif float(candidate["selection_score"]) == float(existing["selection_score"]):
                    if int(candidate["url_depth"]) < int(existing["url_depth"]):
                        best_by_domain[domain] = candidate
                    elif int(candidate["url_depth"]) == int(existing["url_depth"]):
                        if len(str(candidate.get("evidence_snippet", ""))) > len(
                            str(existing.get("evidence_snippet", ""))
                        ):
                            best_by_domain[domain] = candidate

        if save_every_query:
            # If you truly want intermediate saves, do it here (but it tends to create noise since we're best-by-domain).
            logging.info("save_every_query enabled; intermediate save is intentionally skipped in best-by-domain mode.")

    logging.info("Enriching %d unique domains.", len(best_by_domain))

    companyrecord_fields = _companyrecord_fields()
    address_field_exists = "address" in companyrecord_fields
    address_required_assumed = _companyrecord_requires_address()

    for domain, candidate in best_by_domain.items():
        homepage = str(candidate["website"])

        contact_info: ContactInfo | None = None
        if contact_settings.get("enabled", True):
            if persistent_cache is not None:
                cached = persistent_cache.get(domain)
                if cached:
                    contact_cache_hits += 1
                    contact_info = cached
                else:
                    contact_cache_misses += 1

            if contact_info is None:
                contact_keywords = contact_settings.get("contact_page_keywords", ["contact", "about", "team"])
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
        company_address: Optional[str] = None
        if contact_info is not None:
            if contact_info.company_name and contact_info.company_name.strip():
                company_name = contact_info.company_name.strip()
            company_address = contact_info.company_address

        record_kwargs: dict[str, Any] = dict(
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

        # --- Address handling (fixes your traceback) ---
        # If CompanyRecord has an address field, always provide *something*.
        if address_field_exists:
            if company_address is not None:
                record_kwargs["address"] = company_address
            else:
                # Avoid crash if address is required by the model.
                # Empty string is better than None for many dataclass/pydantic validators.
                record_kwargs["address"] = ""

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
    cache_hit_rate = (contact_cache_hits / total_enrichment_requests) if total_enrichment_requests else 0.0

    logging.info(
        "Run summary: queries=%d results=%d domains_seen=%d unique_candidates=%d enriched=%d inserted=%d updated=%d "
        "skipped_confidence=%d skipped_excluded_domain=%d skipped_content_type=%d "
        "skipped_low_company_likelihood=%d skipped_low_lpbf_fit=%d contact_cache_hit_rate=%.1f%%",
        total_queries,
        total_results,
        len(domains_seen),
        len(best_by_domain),
        enriched_count,
        inserted_count,
        updated_count,
        skipped_confidence,
        skipped_excluded_domain,
        skipped_content_type,
        skipped_low_company_likelihood,
        skipped_low_lpbf_fit,
        cache_hit_rate * 100,
    )

    if excluded_domain_counts:
        top_excluded = ", ".join(
            f"{domain} ({count})" for domain, count in excluded_domain_counts.most_common(10)
        )
        logging.info("Top excluded domains: %s", top_excluded)

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
            "skipped_excluded_domain": skipped_excluded_domain,
            "skipped_content_type": skipped_content_type,
            "skipped_low_company_likelihood": skipped_low_company_likelihood,
            "skipped_low_lpbf_fit": skipped_low_lpbf_fit,
            "contact_cache_hits": contact_cache_hits,
            "contact_cache_misses": contact_cache_misses,
            "contact_cache_hit_rate": round(cache_hit_rate, 4),
        },
    )

    logging.info("Pipeline complete.")
