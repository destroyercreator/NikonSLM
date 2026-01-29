from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import tldextract
from rapidfuzz import fuzz, process


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


@dataclass
class CRMIndex:
    domain_to_index: dict[str, int]
    name_keys: list[str]
    name_indices: list[int]
    name_positions: dict[int, int]


def canonical_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join(part for part in [ext.domain, ext.suffix] if part)


def load_crm(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path)


def build_crm_index(df: pd.DataFrame) -> CRMIndex:
    domain_to_index: dict[str, int] = {}
    name_keys: list[str] = []
    name_indices: list[int] = []
    name_positions: dict[int, int] = {}
    if df.empty:
        return CRMIndex(domain_to_index, name_keys, name_indices, name_positions)
    if "domain" in df.columns:
        for idx, domain in df["domain"].items():
            if pd.notna(domain):
                domain_to_index[str(domain)] = idx
    if "company_name" in df.columns:
        for idx, name in df["company_name"].items():
            if pd.notna(name):
                name_positions[idx] = len(name_keys)
                name_keys.append(str(name).lower())
                name_indices.append(idx)
    return CRMIndex(domain_to_index, name_keys, name_indices, name_positions)


def match_existing(
    df: pd.DataFrame,
    record: CompanyRecord,
    threshold: int,
    index: CRMIndex | None = None,
) -> int | None:
    if df.empty:
        return None
    if index is None:
        index = build_crm_index(df)
    if record.domain and record.domain in index.domain_to_index:
        return index.domain_to_index[record.domain]
    if index.name_keys:
        match = process.extractOne(record.name.lower(), index.name_keys, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            return index.name_indices[match[2]]
    return None


def upsert_record(
    df: pd.DataFrame,
    record: CompanyRecord,
    threshold: int,
    index: CRMIndex | None = None,
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
    if index is None:
        index = build_crm_index(df)
    existing_idx = match_existing(df, record, threshold, index=index)
    if existing_idx is None:
        row["first_seen"] = now
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        new_idx = len(df) - 1
        if record.domain:
            index.domain_to_index[record.domain] = new_idx
        if record.name:
            index.name_positions[new_idx] = len(index.name_keys)
            index.name_keys.append(record.name.lower())
            index.name_indices.append(new_idx)
    else:
        for key, value in row.items():
            df.at[existing_idx, key] = value
        if pd.isna(df.at[existing_idx, "first_seen"]):
            df.at[existing_idx, "first_seen"] = now
        if record.domain:
            index.domain_to_index[record.domain] = existing_idx
        if record.name:
            existing_pos = index.name_positions.get(existing_idx)
            if existing_pos is None:
                index.name_positions[existing_idx] = len(index.name_keys)
                index.name_keys.append(record.name.lower())
                index.name_indices.append(existing_idx)
            else:
                index.name_keys[existing_pos] = record.name.lower()
    return df


def save_crm(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        sort_cols = [col for col in ["industries", "company_name"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    df.to_excel(path, index=False)
