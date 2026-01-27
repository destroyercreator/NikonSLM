from __future__ import annotations

import re

PROVINCES = {
    "alberta": "AB",
    "british columbia": "BC",
    "manitoba": "MB",
    "new brunswick": "NB",
    "newfoundland and labrador": "NL",
    "nova scotia": "NS",
    "ontario": "ON",
    "prince edward island": "PE",
    "quebec": "QC",
    "saskatchewan": "SK",
    "northwest territories": "NT",
    "nunavut": "NU",
    "yukon": "YT",
}
PROVINCE_CODES = set(PROVINCES.values())

CITY_PROVINCE_REGEX = re.compile(
    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*,\s*(ON|QC|BC|AB|MB|NS|NB|NL|PE|SK|NT|NU|YT)",
)


def extract_location(text: str) -> tuple[str | None, str | None]:
    match = CITY_PROVINCE_REGEX.search(text)
    if match:
        return match.group(1), match.group(2)
    lowered = text.lower()
    for name, code in PROVINCES.items():
        if name in lowered:
            return None, code
    for code in PROVINCE_CODES:
        if f" {code.lower()}" in lowered:
            return None, code
    return None, None
