# NikonSLM

## LPBF Candidate Discovery Program

This repository contains a Python-based, repeatable research workflow that discovers and tracks Canadian companies likely to benefit from laser powder bed fusion (LPBF) metal additive manufacturing. The pipeline uses search APIs (no ad-hoc scraping), deterministic keyword classification, optional LLM classification, and contact enrichment from company websites only. Results are stored in an Excel-based lightweight CRM with upsert behavior.

### Features

- **Repeatable search program** using SerpAPI or Bing Web Search.
- **Structured queries** tied to LPBF applications (heat exchangers, manifolds, implants, etc.).
- **Company-level evidence** with snippets and source URLs.
- **Deterministic industry classification** with optional LLM refinement.
- **Contact enrichment** from company sites only (robots.txt respected).
- **CRM-style Excel output** with dedupe, last-seen tracking, and upserts.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set at least one search API key in `.env`:

- `SERPAPI_API_KEY` (SerpAPI)
- `BING_API_KEY` (Azure Bing Search)

Optional LLM classification:

- `OPENAI_API_KEY`

### Configuration

Edit `config/config.yaml` to adjust industries, query templates, and classification thresholds. Industry keyword rules live in `config/industry_keywords.yaml`.

### Run

```bash
lpbf-tracker --config config/config.yaml
```

The CRM output is written to `data/companies.xlsx` by default.

### Notes

- The pipeline does not scrape LinkedIn or gated platforms.
- Contact extraction only uses company-owned websites and respects robots.txt.
- Accuracy and traceability are prioritized over volume.
