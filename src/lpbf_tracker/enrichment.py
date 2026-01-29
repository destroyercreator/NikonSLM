from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup


EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_REGEX = re.compile(r"\+?\d?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")


@dataclass
class ContactInfo:
    emails: list[str]
    phones: list[str]
    contact_page: str | None
    staff: list[str]


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
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = session.get(url, timeout=(10, 30))
        except requests.RequestException:
            return ""
        if should_skip_rate_limit(response):
            return ""
        if not response.ok:
            return ""
        return response.text
    return ""


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
    staff = []
    for tag in soup.find_all(["h2", "h3", "strong"]):
        text = tag.get_text(" ", strip=True)
        if 2 <= len(text.split()) <= 4:
            staff.append(text)
    return sorted(set(staff))


def enrich_contacts(
    base_url: str,
    user_agent: str,
    contact_keywords: list[str],
    max_pages: int,
) -> ContactInfo:
    if not is_allowed_by_robots(base_url, user_agent):
        return ContactInfo(emails=[], phones=[], contact_page=None, staff=[])
    session = build_session(user_agent)
    homepage_html = fetch_page(session, base_url)
    if not homepage_html:
        return ContactInfo(emails=[], phones=[], contact_page=None, staff=[])
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
    staff = []
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
        if len(emails) + len(phones) >= 2:
            break
    return ContactInfo(
        emails=emails,
        phones=phones,
        contact_page=contact_page,
        staff=staff,
    )
