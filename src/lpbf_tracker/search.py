from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import os
import random
import requests
import time

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class SearchProvider:
    def __init__(self, negative_keywords: Iterable[str] | None = None) -> None:
        self.negative_keywords = [keyword.strip() for keyword in negative_keywords or [] if keyword.strip()]

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        raise NotImplementedError

    def _apply_negative_keywords(self, query: str) -> str:
        if not self.negative_keywords:
            return query
        formatted_keywords: list[str] = []
        for keyword in self.negative_keywords:
            if keyword.startswith("-"):
                formatted_keywords.append(keyword)
                continue
            if " " in keyword and not (keyword.startswith('"') and keyword.endswith('"')):
                keyword = f'"{keyword}"'
            formatted_keywords.append(f"-{keyword}")
        return f"{query} {' '.join(formatted_keywords)}"


class SerpApiProvider(SearchProvider):
    def __init__(self, api_key: str, engine: str, negative_keywords: Iterable[str] | None = None) -> None:
        super().__init__(negative_keywords=negative_keywords)
        self.api_key = api_key
        self.engine = engine

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        query = self._apply_negative_keywords(query)
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "engine": self.engine,
                "api_key": self.api_key,
                "num": max_results,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("organic_results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("link", ""),
                )
            )
        return results


class BingProvider(SearchProvider):
    def __init__(self, api_key: str, endpoint: str, negative_keywords: Iterable[str] | None = None) -> None:
        super().__init__(negative_keywords=negative_keywords)
        self.api_key = api_key
        self.endpoint = endpoint

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        query = self._apply_negative_keywords(query)
        response = requests.get(
            self.endpoint,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            params={"q": query, "count": max_results},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("webPages", {}).get("value", []):
            results.append(
                SearchResult(
                    title=item.get("name", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("url", ""),
                )
            )
        return results


class BraveProvider(SearchProvider):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        negative_keywords: Iterable[str] | None = None,
        *,
        min_interval_s: float = 1.05,   # Free plan is 1 rps; use a bit of slack
        max_retries: int = 6,
    ) -> None:
        super().__init__(negative_keywords=negative_keywords)
        self.api_key = api_key
        self.endpoint = endpoint
        self.min_interval_s = float(min_interval_s)
        self.max_retries = int(max_retries)
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request_ts = time.monotonic()

    def _request_once(self, query: str, max_results: int) -> requests.Response:
        count = max(1, min(int(max_results), 20))
        self._throttle()
        return requests.get(
            self.endpoint,
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            },
            params={"q": query, "count": count},
            timeout=30,
        )

    def _request_with_retries(self, query: str, max_results: int) -> requests.Response:
        backoff = 1.0
        for attempt in range(self.max_retries + 1):
            resp = self._request_once(query, max_results)

            # Success
            if resp.ok:
                return resp

            # 429: respect Retry-After if provided; otherwise exponential backoff + jitter
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(1.0, float(retry_after))
                    except ValueError:
                        sleep_s = backoff
                else:
                    # add small jitter to avoid sync issues
                    sleep_s = backoff + random.uniform(0.0, 0.25)
                time.sleep(sleep_s)
                backoff = min(backoff * 2.0, 30.0)
                continue

            # Transient server errors: retry
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(backoff + random.uniform(0.0, 0.25))
                backoff = min(backoff * 2.0, 30.0)
                continue

            # Anything else: hard fail
            raise RuntimeError(
                f"Brave search failed: {resp.status_code} {resp.reason}. Body: {resp.text}"
            )

        # If we exhaust retries:
        raise RuntimeError(
            f"Brave search failed after retries (likely rate-limited). Last response: {resp.status_code} {resp.reason}. Body: {resp.text}"
        )

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        q1 = self._apply_negative_keywords(query)
        response = self._request_with_retries(q1, max_results)

        # If Brave rejects query syntax, retry once without negative terms.
        if response.status_code == 422:
            response = self._request_with_retries(query, max_results)

        payload = response.json()
        results: list[SearchResult] = []
        for item in payload.get("web", {}).get("results", []) or []:
            results.append(
                SearchResult(
                    title=item.get("title", "") or "",
                    snippet=item.get("description", "") or "",
                    url=item.get("url", "") or "",
                )
            )
        return results


def build_provider(settings: dict) -> SearchProvider:
    provider = settings.get("provider", "serpapi")
    negative_keywords = _negative_keywords_for_provider(settings, provider)
    if provider == "serpapi":
        api_key_env = settings["serpapi"]["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in env var {api_key_env}")
        return SerpApiProvider(
            api_key=api_key,
            engine=settings["serpapi"]["engine"],
            negative_keywords=negative_keywords,
        )
    if provider == "bing":
        api_key_env = settings["bing"]["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in env var {api_key_env}")
        return BingProvider(
            api_key=api_key,
            endpoint=settings["bing"]["endpoint"],
            negative_keywords=negative_keywords,
        )
    if provider == "brave":
        api_key_env = settings["brave"]["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in env var {api_key_env}")
        return BraveProvider(
            api_key=api_key,
            endpoint=settings["brave"]["endpoint"],
            negative_keywords=negative_keywords,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def build_queries(config: dict) -> Iterable[str]:
    queries_config = config["queries"]
    provider = config.get("search", {}).get("provider", "")
    base_templates = queries_config["base_templates"]
    industries = queries_config["industries"]
    applications = queries_config["applications"]
    provinces = queries_config.get("provinces", [])
    inject_provinces = queries_config.get("inject_provinces", False)
    max_queries = queries_config.get("max_queries")
    shuffle_seed = queries_config.get("shuffle_seed")
    rotate_daily = queries_config.get("rotate_daily", False)
    queries: list[str] = []
    brave_short_mode = str(provider).lower() == "brave"

    for industry in industries:
        for application in applications:
            for template in base_templates:
                if inject_provinces and provinces:
                    for province in provinces:
                        query = _render_template(
                            template, industry=industry, application=application, province=province
                        )
                        if brave_short_mode:
                            query = _truncate_query_terms(query, max_terms=7)
                        queries.append(query)
                else:
                    query = _render_template(
                        template, industry=industry, application=application, province=None
                    )
                    if brave_short_mode:
                        query = _truncate_query_terms(query, max_terms=7)
                    queries.append(query)

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(queries)

    if rotate_daily and queries:
        offset = date.today().toordinal() + _seed_offset(shuffle_seed)
        offset %= len(queries)
        queries = queries[offset:] + queries[:offset]

    if max_queries is not None:
        queries = queries[:max_queries]

    return queries


def _truncate_query_terms(query: str, *, max_terms: int) -> str:
    terms = query.split()
    if len(terms) <= max_terms:
        return query
    return " ".join(terms[:max_terms])


def _render_template(
    template: str, *, industry: str, application: str, province: str | None
) -> str:
    formatted = template.format(industry=industry, application=application, province=province or "")
    if province and "{province}" not in template:
        formatted = f"{formatted} {province}"
    return " ".join(formatted.split())


def _negative_keywords_for_provider(settings: dict, provider: str) -> list[str]:
    negative_keywords = settings.get("negative_keywords", {})
    if isinstance(negative_keywords, list):
        return negative_keywords
    if isinstance(negative_keywords, dict):
        return negative_keywords.get(provider, [])
    return []


def _seed_offset(seed: object | None) -> int:
    if seed is None:
        return 0
    if isinstance(seed, int):
        return seed
    return sum(ord(ch) for ch in str(seed))
