from __future__ import annotations

import json
import re
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lpbf_tracker.cache import ContactCache, ContactInfo

EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_REGEX = re.compile(r"\+?\d?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")


def is_allowed_by_robots(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=(5, 10))
    except requests.RequestException:
        return True
    if not response.ok:
        return True
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch(user_agent, url)


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def should_skip_rate_limit(response: requests.Response) -> bool:
    return response.status_code == 429


def fetch_page(session: requests.Session, url: str) -> str:
    try:
        response = session.get(url, timeout=(10, 30))
    except requests.RequestException:
        return ""
    if should_skip_rate_limit(response):
        return ""
    if not response.ok:
        return ""
    return response.text


def extract_contacts(html: str) -> tuple[list[str], list[str]]:
    emails = sorted(set(EMAIL_REGEX.findall(html)))
    phones = sorted({match.group(0) for match in PHONE_REGEX.finditer(html)})
    normalized_phones = ["".join(filter(str.isdigit, phone)) for phone in phones]
    normalized_phones = [phone for phone in normalized_phones if len(phone) >= 10]
    return emails, normalized_phones


PRIORITY_CONTACT_KEYWORDS = (
    "contact",
    "about",
    "team",
    "staff",
    "people",
    "leadership",
    "directory",
)


def score_contact_link(text: str, href: str, keywords: Iterable[str]) -> int:
    haystack = f"{text} {href}"
    score = 0
    for weight, keyword in enumerate(PRIORITY_CONTACT_KEYWORDS, start=1):
        if keyword in haystack:
            score = max(score, len(PRIORITY_CONTACT_KEYWORDS) - weight + 1)
    if any(keyword in haystack for keyword in keywords):
        score += 1
    return score


def find_contact_links(
    html: str,
    base_url: str,
    keywords: Iterable[str],
    max_results: int,
) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[int, int, str]] = []
    for index, anchor in enumerate(soup.find_all("a", href=True)):
        text = anchor.get_text(" ", strip=True).lower()
        href = anchor["href"].strip().lower()
        if not href or href.startswith("#"):
            continue
        score = score_contact_link(text, href, keywords)
        if score:
            candidates.append((score, index, urljoin(base_url, anchor["href"])))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    links: list[str] = []
    seen = set()
    for _, _, link in candidates:
        if link in seen:
            continue
        seen.add(link)
        links.append(link)
        if len(links) >= max_results:
            break
    return links


def extract_staff(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    staff: list[str] = []
    for tag in soup.find_all(["h2", "h3", "strong"]):
        text = tag.get_text(" ", strip=True)
        if 2 <= len(text.split()) <= 4:
            staff.append(text)
    return sorted(set(staff))


def _clean_title(title: str) -> str:
    cleaned = " ".join(title.split()).strip()
    if not cleaned:
        return ""
    separators = [" | ", " - ", " – ", " — ", " :: "]
    for separator in separators:
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0].strip()
    if cleaned.lower() in {"home", "homepage"}:
        return ""
    return cleaned


def _normalize_address(address: object) -> str | None:
    if isinstance(address, str):
        cleaned = " ".join(address.split()).strip()
        return cleaned or None
    if not isinstance(address, dict):
        return None
    parts = [
        address.get("streetAddress"),
        address.get("addressLocality"),
        address.get("addressRegion"),
        address.get("postalCode"),
        address.get("addressCountry"),
    ]
    cleaned_parts = [str(part).strip() for part in parts if part]
    return ", ".join(cleaned_parts) if cleaned_parts else None


def _iter_jsonld_nodes(data: object) -> Iterable[dict]:
    if isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_nodes(item)
    elif isinstance(data, dict):
        if "@graph" in data and isinstance(data["@graph"], list):
            for item in data["@graph"]:
                if isinstance(item, dict):
                    yield item
        else:
            yield data


def extract_identity(html: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    company_name: str | None = None
    company_address: str | None = None

    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        for node in _iter_jsonld_nodes(data):
            node_type = node.get("@type")
            if isinstance(node_type, list):
                types = {str(entry) for entry in node_type}
            elif node_type:
                types = {str(node_type)}
            else:
                types = set()

            if not types.intersection({"Organization", "LocalBusiness"}):
                continue

            if not company_name:
                name = node.get("name")
                if isinstance(name, str) and name.strip():
                    company_name = name.strip()

            if not company_address:
                company_address = _normalize_address(node.get("address"))

            if company_name and company_address:
                break

        if company_name and company_address:
            break

    if not company_name:
        title_tag = soup.title.string if soup.title else ""
        cleaned = _clean_title(title_tag or "")
        company_name = cleaned or None

    return company_name, company_address


def enrich_contacts(
    base_url: str,
    user_agent: str,
    contact_keywords: list[str],
    max_pages: int,
    cache: ContactCache | None = None,
    domain: str | None = None,
) -> ContactInfo:
    cache_domain = domain or urlparse(base_url).netloc

    if cache and cache_domain:
        cached = cache.get(cache_domain)
        if cached:
            return cached

    # Robots gate (cache negative result too)
    if not is_allowed_by_robots(base_url, user_agent):
        result = ContactInfo(
            emails=[],
            phones=[],
            contact_page=None,
            staff=[],
            company_name=None,
            company_address=None,
        )
        if cache and cache_domain:
            cache.set(cache_domain, result)
        return result

    session = build_session(user_agent)
    homepage_html = fetch_page(session, base_url)
    if not homepage_html:
        result = ContactInfo(
            emails=[],
            phones=[],
            contact_page=None,
            staff=[],
            company_name=None,
            company_address=None,
        )
        if cache and cache_domain:
            cache.set(cache_domain, result)
        return result

    company_name, company_address = extract_identity(homepage_html)
    emails, phones = extract_contacts(homepage_html)

    combined_keywords = sorted(
        {keyword.lower() for keyword in contact_keywords} | set(PRIORITY_CONTACT_KEYWORDS)
    )

    contact_links = find_contact_links(
        homepage_html,
        base_url,
        combined_keywords,
        max_results=max(0, max_pages - 1),
    )
    contact_page = contact_links[0] if contact_links else None

    staff: list[str] = []
    pages_checked = 1

    for contact_link in contact_links:
        if pages_checked >= max_pages:
            break
        if not is_allowed_by_robots(contact_link, user_agent):
            continue

        contact_html = fetch_page(session, contact_link)
        pages_checked += 1
        if not contact_html:
            continue

        more_emails, more_phones = extract_contacts(contact_html)
        emails = sorted(set(emails + more_emails))
        phones = sorted(set(phones + more_phones))
        staff = sorted(set(staff + extract_staff(contact_html)))

        # Stop early once we have at least some usable contact info.
        if len(emails) + len(phones) >= 2:
            break

    result = ContactInfo(
        emails=emails,
        phones=phones,
        contact_page=contact_page,
        staff=staff,
        company_name=company_name,
        company_address=company_address,
    )

    if cache and cache_domain:
        cache.set(cache_domain, result)

    return result
