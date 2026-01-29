from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import os
import random
import requests


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
    def __init__(self, api_key: str, endpoint: str, negative_keywords: Iterable[str] | None = None) -> None:
        super().__init__(negative_keywords=negative_keywords)
        self.api_key = api_key
        self.endpoint = endpoint

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        query = self._apply_negative_keywords(query)
        response = requests.get(
            self.endpoint,
            headers={"X-Subscription-Token": self.api_key},
            params={"q": query, "count": max_results},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("web", {}).get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    snippet=item.get("description", ""),
                    url=item.get("url", ""),
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
    base_templates = queries_config["base_templates"]
    industries = queries_config["industries"]
    applications = queries_config["applications"]
    provinces = queries_config.get("provinces", [])
    inject_provinces = queries_config.get("inject_provinces", False)
    max_queries = queries_config.get("max_queries")
    shuffle_seed = queries_config.get("shuffle_seed")
    rotate_daily = queries_config.get("rotate_daily", False)
    queries: list[str] = []

    for industry in industries:
        for application in applications:
            for template in base_templates:
                if inject_provinces and provinces:
                    for province in provinces:
                        queries.append(
                            _render_template(
                                template, industry=industry, application=application, province=province
                            )
                        )
                else:
                    queries.append(
                        _render_template(template, industry=industry, application=application, province=None)
                    )

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(queries)

    if rotate_daily and queries:
        offset = date.today().toordinal() + _seed_offset(shuffle_seed)
        offset %= len(queries)
        queries = queries[offset:] + queries[:offset]

    if max_queries is not None:
        queries = queries[:max_queries]

    return queries


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
