"""
Unit tests for conditions/builders.py and conditions/validate.py, using the
tiny synthetic fixtures in tests/fixtures.py (never real dataset content).
"""
from __future__ import annotations
import unittest

from conditions.builders import (
    build_retrieval_blindness_corpus,
    build_retrieval_noise_corpus,
    build_evidence_ignorance_context,
)
from conditions.validate import (
    validate_retrieval_blindness_corpus,
    validate_retrieval_noise_corpus,
    validate_evidence_ignorance_context,
    validate_no_label_leak_in_ids,
    validate_all_conditions_for_item,
)
from tests.fixtures import build_fixture_items


class TestRetrievalBlindness(unittest.TestCase):
    def setUp(self):
        self.item = next(i for i in build_fixture_items() if i.failure_mode == "retrieval_blindness")

    def test_clean_contains_gold(self):
        corpus = build_retrieval_blindness_corpus(self.item, "clean")
        self.assertEqual(sum(1 for p in corpus if p.is_gold), 1)
        validate_retrieval_blindness_corpus(self.item, "clean", corpus)

    def test_blind_withholds_gold(self):
        corpus = build_retrieval_blindness_corpus(self.item, "blind")
        self.assertEqual(sum(1 for p in corpus if p.is_gold), 0)
        validate_retrieval_blindness_corpus(self.item, "blind", corpus)

    def test_unknown_condition_raises(self):
        with self.assertRaises(ValueError):
            build_retrieval_blindness_corpus(self.item, "not_a_condition")


class TestRetrievalNoise(unittest.TestCase):
    def setUp(self):
        self.item = next(i for i in build_fixture_items() if i.failure_mode == "retrieval_noise")

    def test_clean_has_no_distractors(self):
        corpus = build_retrieval_noise_corpus(self.item, "clean")
        self.assertEqual(sum(1 for p in corpus if p.is_gold), 1)
        self.assertEqual(sum(1 for p in corpus if p.is_distractor), 0)
        validate_retrieval_noise_corpus(self.item, "clean", corpus)

    def test_noisy_has_gold_plus_distractors(self):
        corpus = build_retrieval_noise_corpus(self.item, "noisy")
        self.assertEqual(sum(1 for p in corpus if p.is_gold), 1)
        self.assertGreaterEqual(sum(1 for p in corpus if p.is_distractor), 1)
        validate_retrieval_noise_corpus(self.item, "noisy", corpus)


class TestEvidenceIgnorance(unittest.TestCase):
    def setUp(self):
        self.item = next(i for i in build_fixture_items() if i.failure_mode == "evidence_ignorance")
        self.padding = list(self.item.candidate_passages)

    def test_top_places_gold_first(self):
        ctx = build_evidence_ignorance_context(self.item, "top", self.padding)
        self.assertTrue(ctx[0].is_gold)
        validate_evidence_ignorance_context(self.item, "top", ctx)

    def test_middle_places_gold_in_middle(self):
        ctx = build_evidence_ignorance_context(self.item, "middle", self.padding)
        gold_positions = [i for i, p in enumerate(ctx) if p.is_gold]
        self.assertEqual(len(gold_positions), 1)
        self.assertNotEqual(gold_positions[0], 0)
        self.assertNotEqual(gold_positions[0], len(ctx) - 1)
        validate_evidence_ignorance_context(self.item, "middle", ctx)

    def test_removed_has_no_gold(self):
        ctx = build_evidence_ignorance_context(self.item, "removed", self.padding)
        self.assertEqual(sum(1 for p in ctx if p.is_gold), 0)
        validate_evidence_ignorance_context(self.item, "removed", ctx)


class TestLabelLeak(unittest.TestCase):
    def test_neutral_ids_pass(self):
        from data.schema import Passage
        validate_no_label_leak_in_ids([Passage("rgb-12-abc123def4", "text", is_gold=True)])

    def test_role_bearing_id_fails(self):
        from data.schema import Passage
        with self.assertRaises(AssertionError):
            validate_no_label_leak_in_ids([Passage("rgb-12-gold-passage", "text", is_gold=True)])


class TestAllFixturesValidate(unittest.TestCase):
    def test_all_fixture_items_pass_full_validation(self):
        for item in build_fixture_items():
            validate_all_conditions_for_item(item)


if __name__ == "__main__":
    unittest.main()
