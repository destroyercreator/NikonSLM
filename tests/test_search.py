import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lpbf_tracker.search import BingProvider, BraveProvider, SerpApiProvider


class TestNegativeKeywordQueries(unittest.TestCase):
    def test_serpapi_applies_dash_negative_keywords(self) -> None:
        provider = SerpApiProvider(api_key="key", engine="google", negative_keywords=["forum", "press release"])
        self.assertEqual(
            provider._apply_negative_keywords("metal printing"),
            "metal printing -forum -\"press release\"",
        )

    def test_bing_applies_not_negative_keywords(self) -> None:
        provider = BingProvider(api_key="key", endpoint="https://example.com", negative_keywords=["forum", "press release"])
        self.assertEqual(
            provider._apply_negative_keywords("metal printing"),
            "metal printing NOT forum NOT \"press release\"",
        )

    def test_brave_applies_dash_negative_keywords(self) -> None:
        provider = BraveProvider(api_key="key", endpoint="https://example.com", negative_keywords=["forum", "press release"])
        self.assertEqual(
            provider._apply_negative_keywords("metal printing"),
            "metal printing -forum -\"press release\"",
        )


if __name__ == "__main__":
    unittest.main()
