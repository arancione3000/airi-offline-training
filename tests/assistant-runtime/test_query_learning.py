import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/assistant-runtime"))
from query_learning import evaluate


class QueryQualificationTests(unittest.TestCase):
    def test_wrong_answer_rejects_candidate_but_same_coverage_with_abstention_passes(self):
        bank = Mock()
        bank._read.return_value = ({"entries": [{
            "lesson_id": "code_copy", "train": [], "qualification": [],
        }]}, False)
        families = [{
            "id": "code_copy", "operation": "copy",
            "source": {"prefix": "Il codice è ", "suffix": "."},
        }]
        rows = [{
            "id": name, "source_lesson_id": "code_copy", "lesson_id": "code_copy",
            "query": name,
        } for name in ("correct", "miss")]
        router = Mock()

        def predict(messages, miss):
            question = messages[-1]["content"]
            if question == "correct":
                return {"answer": messages[0]["content"].removeprefix("Il codice è ").removesuffix(".")}
            return miss if question == "miss" else None

        with patch("query_learning.LearnedQueryRouter", return_value=router), patch("query_learning._route", return_value=None):
            router.predict.side_effect = lambda messages: predict(messages, {"answer": "invented"})
            wrong = evaluate(bank, None, rows, families)
            router.predict.side_effect = lambda messages: predict(messages, None)
            abstention = evaluate(bank, None, rows, families)

        self.assertEqual(wrong["positive_correct"], 2)
        self.assertEqual(abstention["positive_correct"], 2)
        self.assertFalse(wrong["passed"])
        self.assertEqual(wrong["wrong_positive_answers"], 2)
        self.assertTrue(abstention["passed"])
