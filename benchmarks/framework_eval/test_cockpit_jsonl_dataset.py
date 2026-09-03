from pathlib import Path
import unittest

from .datasets.cockpit_jsonl import CockpitJSONLDataset


ROOT = Path(__file__).parent / "challenges/cockpit_zh_public_mix_500_v7"


class CockpitJSONLDatasetTests(unittest.TestCase):
    def test_published_dataset_matches_portable_contract(self):
        dataset = CockpitJSONLDataset(ROOT)
        self.assertEqual(50, len(dataset.conversations()))
        self.assertEqual(500, len(dataset.questions()))
        self.assertEqual([], dataset.validate())

    def test_question_metadata_preserves_temporal_and_abstention_fields(self):
        question = CockpitJSONLDataset(ROOT).questions()[0]
        self.assertTrue(question.metadata["question_date"])
        self.assertIn("is_abstention", question.metadata)
        self.assertTrue(question.evidence_ids)
