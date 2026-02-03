import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lpbf_tracker import pipeline


class TestPipelineFilters(unittest.TestCase):
    def test_is_excluded_domain_rejects_empty_and_known_domains(self) -> None:
        self.assertTrue(pipeline.is_excluded_domain(""))
        self.assertTrue(pipeline.is_excluded_domain("reddit.com"))
        self.assertTrue(pipeline.is_excluded_domain("news.reddit.com"))

    def test_is_excluded_domain_rejects_forbidden_tlds(self) -> None:
        self.assertTrue(pipeline.is_excluded_domain("agency.gov"))
        self.assertTrue(pipeline.is_excluded_domain("college.edu"))
        self.assertTrue(pipeline.is_excluded_domain("uni.ac.uk"))

    def test_is_excluded_domain_allows_normal_company_domains(self) -> None:
        self.assertFalse(pipeline.is_excluded_domain("example.com"))

    def test_content_likelihood_score_thresholding(self) -> None:
        text_with_one_marker = "Company news and updates"
        text_with_two_markers = "Latest blog article"
        self.assertEqual(pipeline.content_likelihood_score(text_with_one_marker), 1)
        self.assertEqual(pipeline.content_likelihood_score(text_with_two_markers), 2)


if __name__ == "__main__":
    unittest.main()
