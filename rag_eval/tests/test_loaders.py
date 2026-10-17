"""
Loader invariant tests, run against the REAL downloaded RGB / Lost in the
Middle raw files under data/raw/. These files are gitignored (see
.gitignore — raw third-party data is not redistributed), so these tests
skip themselves (rather than failing) when the raw files are not present
locally, e.g. on a fresh checkout before running the dataset download step.
"""
from __future__ import annotations
import unittest
from pathlib import Path

from data.rgb_loader import load_rgb_subset, _flatten_answers
from data.lost_in_the_middle_loader import load_lost_in_the_middle_subset
from conditions.validate import validate_no_label_leak_in_ids, validate_all_conditions_for_item

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RGB_FILE = RAW_DIR / "rgb" / "en_refine.json"
LITM_FILE = RAW_DIR / "lost_in_the_middle" / "nq-open-10_total_documents_gold_at_4.jsonl.gz"

SEED = 42
N = 30


@unittest.skipUnless(RGB_FILE.exists(), f"RGB raw file not present at {RGB_FILE} — run the dataset download step first")
class TestRgbLoader(unittest.TestCase):
    def test_retrieval_blindness_sample_size_and_shape(self):
        items = load_rgb_subset(str(RAW_DIR / "rgb"), "retrieval_blindness", N, SEED)
        self.assertEqual(len(items), N)
        ids = [i.item_id for i in items]
        self.assertEqual(len(ids), len(set(ids)), "item_ids must be unique")
        for item in items:
            self.assertTrue(item.item_id.startswith("rgb-rb-"))
            self.assertEqual(item.source_dataset, "rgb")
            self.assertEqual(item.failure_mode, "retrieval_blindness")
            self.assertTrue(item.gold_passage.is_gold)
            self.assertTrue(item.question.strip())
            self.assertTrue(item.reference_answer.strip())
            self.assertGreaterEqual(len(item.candidate_passages), 1)
            self.assertTrue(all(not p.is_gold for p in item.candidate_passages))
            validate_no_label_leak_in_ids([item.gold_passage] + item.candidate_passages)
            validate_all_conditions_for_item(item, check_neutral_ids=True)

    def test_retrieval_noise_sample_marks_distractors(self):
        items = load_rgb_subset(str(RAW_DIR / "rgb"), "retrieval_noise", N, SEED)
        self.assertEqual(len(items), N)
        for item in items:
            self.assertTrue(item.item_id.startswith("rgb-rn-"))
            self.assertTrue(all(p.is_distractor for p in item.candidate_passages))
            self.assertGreaterEqual(len(item.candidate_passages), 1)
            validate_all_conditions_for_item(item, check_neutral_ids=True)

    def test_deterministic_sampling(self):
        first = load_rgb_subset(str(RAW_DIR / "rgb"), "retrieval_blindness", N, SEED)
        second = load_rgb_subset(str(RAW_DIR / "rgb"), "retrieval_blindness", N, SEED)
        self.assertEqual([i.item_id for i in first], [i.item_id for i in second])
        self.assertEqual([i.gold_passage.passage_id for i in first], [i.gold_passage.passage_id for i in second])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_rgb_subset(str(RAW_DIR / "does_not_exist"), "retrieval_blindness", N, SEED)

    def test_invalid_failure_mode_raises(self):
        with self.assertRaises(ValueError):
            load_rgb_subset(str(RAW_DIR / "rgb"), "evidence_ignorance", N, SEED)

    def test_no_reference_answer_is_a_single_character(self):
        """
        Regression test for the 2026-08-17 flat-answer-list bug: en_refine.json
        mixes flat-list answers (e.g. ["Kentucky"]) with nested-list answers
        (e.g. [["Jan 2, 2022", "January 2, 2022"]]). The old flattening logic
        iterated over the CHARACTERS of flat-list strings, so ~70% of items
        got a single-character reference_answer (e.g. "K" instead of
        "Kentucky"). Any full-corpus sample must contain zero single-char
        answers, since real RGB answers are always multi-character.
        """
        items = load_rgb_subset(str(RAW_DIR / "rgb"), "retrieval_blindness", N, SEED)
        single_char = [i for i in items if len(i.reference_answer.strip()) <= 1]
        self.assertEqual(
            single_char, [],
            f"found {len(single_char)} item(s) with a suspicious single-character "
            "reference_answer — the flat vs. nested answer-list bug may have regressed",
        )


class TestFlattenAnswers(unittest.TestCase):
    def test_flat_list_of_strings(self):
        self.assertEqual(_flatten_answers(["Kentucky"]), ["Kentucky"])

    def test_nested_list_of_alias_groups(self):
        self.assertEqual(
            _flatten_answers([["Jan 2, 2022", "January 2, 2022"]]),
            ["Jan 2, 2022", "January 2, 2022"],
        )

    def test_mixed_flat_and_nested(self):
        self.assertEqual(
            _flatten_answers(["Kentucky", ["Jan 2, 2022", "January 2, 2022"]]),
            ["Kentucky", "Jan 2, 2022", "January 2, 2022"],
        )

    def test_empty(self):
        self.assertEqual(_flatten_answers([]), [])
        self.assertEqual(_flatten_answers(None), [])


@unittest.skipUnless(LITM_FILE.exists(), f"Lost in the Middle raw file not present at {LITM_FILE} — run the dataset download step first")
class TestLostInTheMiddleLoader(unittest.TestCase):
    def test_sample_size_and_shape(self):
        items = load_lost_in_the_middle_subset(str(RAW_DIR / "lost_in_the_middle"), N, SEED)
        self.assertEqual(len(items), N)
        ids = [i.item_id for i in items]
        self.assertEqual(len(ids), len(set(ids)), "item_ids must be unique")
        for item in items:
            self.assertTrue(item.item_id.startswith("litm-"))
            self.assertEqual(item.source_dataset, "lost_in_the_middle")
            self.assertEqual(item.failure_mode, "evidence_ignorance")
            self.assertTrue(item.gold_passage.is_gold)
            self.assertEqual(len(item.candidate_passages), 9, "expected 9 non-gold ctxs per item")
            self.assertTrue(all(not p.is_gold for p in item.candidate_passages))
            validate_no_label_leak_in_ids([item.gold_passage] + item.candidate_passages)

    def test_deterministic_sampling(self):
        first = load_lost_in_the_middle_subset(str(RAW_DIR / "lost_in_the_middle"), N, SEED)
        second = load_lost_in_the_middle_subset(str(RAW_DIR / "lost_in_the_middle"), N, SEED)
        self.assertEqual([i.item_id for i in first], [i.item_id for i in second])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_lost_in_the_middle_subset(str(RAW_DIR / "does_not_exist"), N, SEED)


if __name__ == "__main__":
    unittest.main()
