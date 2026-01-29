from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import tldextract
from rapidfuzz import fuzz


@dataclass
class CompanyRecord:
    name: str
    website: str
    domain: str
    city: str | None
    province: str | None
    industries: list[str]
    confidence: float
    rationale: str
    evidence_snippet: str
    source_url: str
    contact_emails: list[str]
    contact_phones: list[str]
    contact_page: str | None
    staff: list[str]


def canonical_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join(part for part in [ext.domain, ext.suffix] if part)


def load_crm(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path)


def match_existing(df: pd.DataFrame, record: CompanyRecord, threshold: int) -> int | None:
    if df.empty:
        return None
    if "domain" in df.columns:
        domain_matches = df.index[df["domain"] == record.domain].tolist()
        if domain_matches:
            return domain_matches[0]
    if "company_name" in df.columns:
        for idx, name in df["company_name"].items():
            if fuzz.ratio(str(name).lower(), record.name.lower()) >= threshold:
                return idx
    return None


def upsert_record(
    df: pd.DataFrame,
    record: CompanyRecord,
    threshold: int,
) -> pd.DataFrame:
    now = datetime.utcnow().strftime("%Y-%m-%d")
    row = {
        "company_name": record.name,
        "website": record.website,
        "domain": record.domain,
        "city": record.city,
        "province": record.province,
        "industries": ", ".join(record.industries),
        "classification_confidence": record.confidence,
        "classification_rationale": record.rationale,
        "evidence_snippet": record.evidence_snippet,
        "source_url": record.source_url,
        "contact_emails": ", ".join(record.contact_emails),
        "contact_phones": ", ".join(record.contact_phones),
        "contact_page": record.contact_page,
        "staff": ", ".join(record.staff),
        "last_seen": now,
    }
    existing_idx = match_existing(df, record, threshold)
    if existing_idx is None:
        row["first_seen"] = now
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        for key, value in row.items():
            df.at[existing_idx, key] = value
        if pd.isna(df.at[existing_idx, "first_seen"]):
            df.at[existing_idx, "first_seen"] = now
    return df


def save_crm(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        sort_cols = [col for col in ["industries", "company_name"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    df.to_excel(path, index=False)
