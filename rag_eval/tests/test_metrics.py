"""Unit tests for eval/metrics.py."""
from __future__ import annotations
import unittest

from eval.metrics import (
    normalize_text, exact_match, token_f1, answer_correctness,
    retrieval_recall, precision_at_k, ndcg_at_k, gold_rank,
    context_ablation_sensitivity,
)


class TestTextMetrics(unittest.TestCase):
    def test_exact_match_case_and_punctuation_insensitive(self):
        self.assertEqual(exact_match("Paris.", "paris"), 1.0)
        self.assertEqual(exact_match("Paris", "London"), 0.0)

    def test_token_f1_partial_overlap(self):
        score = token_f1("the quick brown fox", "quick brown fox jumps")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_token_f1_empty_prediction_is_zero(self):
        self.assertEqual(token_f1("", "something"), 0.0)

    def test_answer_correctness_returns_both(self):
        result = answer_correctness("Paris", "Paris")
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["token_f1"], 1.0)


class TestRetrievalMetrics(unittest.TestCase):
    def test_retrieval_recall_hit_and_miss(self):
        self.assertEqual(retrieval_recall(["a", "b", "c"], "b"), 1.0)
        self.assertEqual(retrieval_recall(["a", "c"], "b"), 0.0)

    def test_precision_at_k(self):
        self.assertEqual(precision_at_k(["a", "b", "c"], {"a", "c"}, 3), 2 / 3)
        self.assertEqual(precision_at_k([], {"a"}, 3), 0.0)

    def test_precision_at_k_truncates_to_k(self):
        # relevant item at rank 4 must not count when k=3
        self.assertEqual(precision_at_k(["x", "y", "z", "a"], {"a"}, 3), 0.0)

    def test_ndcg_gold_at_top_is_one(self):
        score = ndcg_at_k(["gold", "b", "c"], {"gold": 1.0}, 3)
        self.assertAlmostEqual(score, 1.0)

    def test_ndcg_gold_lower_rank_is_less_than_one(self):
        score = ndcg_at_k(["b", "gold", "c"], {"gold": 1.0}, 3)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_ndcg_gold_absent_is_zero(self):
        score = ndcg_at_k(["b", "c"], {"gold": 1.0}, 3)
        self.assertEqual(score, 0.0)

    def test_gold_rank_found_and_missing(self):
        self.assertEqual(gold_rank(["a", "gold", "c"], "gold"), 2)
        self.assertIsNone(gold_rank(["a", "c"], "gold"))

    def test_context_ablation_sensitivity_positive_when_evidence_helps(self):
        self.assertAlmostEqual(context_ablation_sensitivity(1.0, 0.0), 1.0)
        self.assertAlmostEqual(context_ablation_sensitivity(0.5, 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
