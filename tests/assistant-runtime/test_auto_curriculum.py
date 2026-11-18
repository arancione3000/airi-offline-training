import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/assistant-runtime"))
from auto_curriculum import CATALOG, build, reference
from skill_bank import SkillBank
from skill_programs import OPERATIONS, synthesize


class AutoCurriculumTests(unittest.TestCase):
    def test_reference_has_exact_declared_unicode_and_signed_integer_semantics(self):
        self.assertEqual(reference("copy", "aBé-42"), "aBé-42")
        self.assertEqual(reference("reverse", "aBé-42"), "24-éBa")
        self.assertEqual(reference("uppercase", "Straße"), "STRASSE")
        self.assertEqual(reference("lowercase", "İSTANBUL"), "i\u0307stanbul")
        self.assertEqual(reference("add", (-19, 7)), "-12")
        self.assertEqual(reference("subtract", (7, -19)), "26")
        self.assertEqual(reference("subtract", (19, 7)), "12")
        self.assertEqual(reference("subtract", (7, 19)), "-12")
        self.assertEqual(reference("multiply", (-19, 7)), "-133")
        with self.assertRaises(ValueError):reference("execute", "x")

    def test_catalog_is_finite_unique_deterministic_and_covers_all_operations(self):
        self.assertEqual(len(CATALOG), 24)
        self.assertEqual(len({r["id"] for r in CATALOG}), len(CATALOG))
        self.assertEqual({r["operation"] for r in CATALOG}, set(OPERATIONS))
        for family in CATALOG:
            a = build(family["id"], 17)
            self.assertEqual(a, build(family["id"], 17))
            self.assertNotEqual(a["lesson"]["examples"], build(family["id"], 18)["lesson"]["examples"])
        for seed in (-1, 2**64, True, "1"):
            with self.assertRaises(ValueError):build(CATALOG[0]["id"], seed)

    def test_all_partitions_have_disjoint_contexts_values_and_separate_audit(self):
        for family in CATALOG:
            bundle = build(family["id"], 20261116)
            lesson = bundle["lesson"]
            partitions = [lesson["examples"], lesson["qualification"], bundle["heldout"]]
            extra = 2 if family['id'].startswith('v5_') else 0
            self.assertEqual([len(rows) for rows in partitions], [8, 12, 20 + extra])
            self.assertEqual([sum(r["answer"] is None for r in rows) for rows in partitions], [0, 4, 8 + extra])
            self.assertEqual(set(lesson), {"lesson_id", "trusted", "examples", "qualification"})
            prior_contexts, prior_sources, prior_operands = set(), set(), set()
            for rows in partitions:
                contexts = {json.dumps(r["messages"], sort_keys=True) for r in rows}
                self.assertEqual(len(contexts), len(rows))
                self.assertFalse(prior_contexts & contexts)
                prior_contexts |= contexts
                sources = {r["messages"][0]["content"] for r in rows if r["answer"] is not None}
                self.assertFalse(prior_sources & sources)
                prior_sources |= sources
                if family["operation"] in {"add", "subtract", "multiply"}:
                    operands = {int(value) for source in sources
                                for value in source.removeprefix(family["prefix"]).removesuffix(".").split(" e ")}
                    self.assertTrue(all(abs(value) <= 1_000_000 for value in operands))
                    self.assertFalse(prior_operands & operands)
                    prior_operands |= operands

    def test_each_family_learns_from_train_only_and_survives_real_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = SkillBank(Path(directory) / "bank.json")
            for family in CATALOG:
                bundle = build(family["id"], 97531)
                program = synthesize(bundle["lesson"]["examples"])
                self.assertIsNotNone(program, family["id"])
                self.assertEqual(program["operation"], family["operation"])
                self.assertEqual(bank.learn(bundle["lesson"])["action"], "activated", family["id"])
                for case in bundle["heldout"]:
                    prediction = bank.predict(case["messages"])
                    self.assertEqual(None if prediction is None else prediction["answer"], case["answer"], family["id"])
            before = bank.path.read_bytes()
            bad = copy.deepcopy(build(CATALOG[0]["id"], 222)["lesson"])
            bad["examples"][0]["answer"] = "incorrect-label"
            self.assertEqual(bank.learn(bad)["action"], "rejected")
            self.assertEqual(bank.path.read_bytes(), before)
            self.assertEqual(bank.status()["active_skills"], len(CATALOG))


if __name__ == "__main__":unittest.main()
