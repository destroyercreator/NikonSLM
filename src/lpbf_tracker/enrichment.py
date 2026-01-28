from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import re
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
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except requests.RequestException:
        return True
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


def fetch_page(session: requests.Session, url: str) -> str:
    try:
        response = session.get(url, timeout=(10, 30))
        response.raise_for_status()
    except requests.RequestException:
        return ""
    return response.text


def extract_contacts(html: str) -> tuple[list[str], list[str]]:
    emails = sorted(set(EMAIL_REGEX.findall(html)))
    phones = sorted({match.group(0) for match in PHONE_REGEX.finditer(html)})
    normalized_phones = ["".join(filter(str.isdigit, phone)) for phone in phones]
    normalized_phones = [phone for phone in normalized_phones if len(phone) >= 10]
    return emails, normalized_phones


def find_contact_links(html: str, base_url: str, keywords: Iterable[str]) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True).lower()
        href = anchor["href"]
        if any(keyword in text for keyword in keywords):
            links.append(urljoin(base_url, href))
    return list(dict.fromkeys(links))


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
    contact_links = find_contact_links(homepage_html, base_url, contact_keywords)
    contact_page = contact_links[0] if contact_links else None
    staff = []
    pages_checked = 1
    if contact_page and pages_checked < max_pages:
        if is_allowed_by_robots(contact_page, user_agent):
            contact_html = fetch_page(session, contact_page)
            if contact_html:
                more_emails, more_phones = extract_contacts(contact_html)
                emails = sorted(set(emails + more_emails))
                phones = sorted(set(phones + more_phones))
                staff = extract_staff(contact_html)
    return ContactInfo(
        emails=emails,
        phones=phones,
        contact_page=contact_page,
        staff=staff,
    )
