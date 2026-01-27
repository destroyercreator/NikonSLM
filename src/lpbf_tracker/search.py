from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import os
import requests


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class SearchProvider:
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        raise NotImplementedError


class SerpApiProvider(SearchProvider):
    def __init__(self, api_key: str, engine: str) -> None:
        self.api_key = api_key
        self.engine = engine

    def search(self, query: str, max_results: int) -> list[SearchResult]:
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
    def __init__(self, api_key: str, endpoint: str) -> None:
        self.api_key = api_key
        self.endpoint = endpoint

    def search(self, query: str, max_results: int) -> list[SearchResult]:
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


def build_provider(settings: dict) -> SearchProvider:
    provider = settings.get("provider", "serpapi")
    if provider == "serpapi":
        api_key_env = settings["serpapi"]["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in env var {api_key_env}")
        return SerpApiProvider(api_key=api_key, engine=settings["serpapi"]["engine"])
    if provider == "bing":
        api_key_env = settings["bing"]["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in env var {api_key_env}")
        return BingProvider(api_key=api_key, endpoint=settings["bing"]["endpoint"])
    raise ValueError(f"Unsupported provider: {provider}")


def build_queries(config: dict) -> Iterable[str]:
    base_templates = config["queries"]["base_templates"]
    industries = config["queries"]["industries"]
    applications = config["queries"]["applications"]
    for industry in industries:
        for application in applications:
            for template in base_templates:
                yield template.format(industry=industry, application=application)
