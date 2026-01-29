from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass
class ContactInfo:
    emails: list[str]
    phones: list[str]
    contact_page: str | None
    staff: list[str]


class ContactCache:
    def __init__(self, cache_dir: Path, ttl: timedelta) -> None:
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_domain(self, domain: str) -> Path:
        safe_domain = re.sub(r"[^a-zA-Z0-9.-]", "_", domain)
        return self.cache_dir / f"{safe_domain}.json"

    def get(self, domain: str) -> ContactInfo | None:
        path = self._path_for_domain(domain)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        fetched_at_raw = data.get("fetched_at")
        if not fetched_at_raw:
            return None
        try:
            fetched_at = datetime.fromisoformat(fetched_at_raw)
        except ValueError:
            return None
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - fetched_at > self.ttl:
            return None
        payload = data.get("contact_info", {})
        return ContactInfo(
            emails=list(payload.get("emails", [])),
            phones=list(payload.get("phones", [])),
            contact_page=payload.get("contact_page"),
            staff=list(payload.get("staff", [])),
        )

    def set(self, domain: str, contact_info: ContactInfo) -> None:
        path = self._path_for_domain(domain)
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "contact_info": asdict(contact_info),
        }
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            return
