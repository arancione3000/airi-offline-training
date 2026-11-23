import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from repair import FormatRepair


APPROVED = "Spedite: 12; Stornate: 3"
ARRIVAL = "Uscite: 20; Resi: 3"
CLI = Path(__file__).with_name("cli.py")


def checks(shipped="Uscite", returned="Resi"):
    positives = [
        {"reports": [f"{shipped}: {40 + 2 * index}; {returned}: {5 + index}"], "expected": 35 + index}
        for index in range(8)
    ]
    negatives = [
        {"reports": [f"{shipped}: 22; {returned}: 13", APPROVED], "expected": None},
        {"reports": [f"{shipped}: 15; {returned}: 2", f"{shipped}: 19; {returned}: 2"], "expected": None},
        {"reports": [f"{shipped}: 40"], "expected": None},
        {"reports": [f"{shipped}: 20; {returned}: -2"], "expected": None},
    ]
    return positives + negatives


class RepairTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "state.sqlite"
        self.repair = FormatRepair(self.path)

    def test_new_format_asks_once_and_remains_active_after_restart(self):
        assessment = self.repair.assess([ARRIVAL])
        self.assertEqual(assessment["status"], "clarification")
        self.assertEqual(assessment["proposal"]["labels"], ["resi", "uscite"])
        self.assertTrue(assessment["question"])
        result = self.repair.activate(assessment["proposal"], "Uscite", checks())
        self.assertEqual(result["status"], "activated")

        restarted = FormatRepair(self.path)
        self.assertEqual(restarted.assess([ARRIVAL])["answer"], 17)
        self.assertEqual(restarted.assess(["Resi=4\nUSCITE: 22"])["answer"], 18)
        self.assertEqual(restarted.assess([APPROVED, "Resi: 3; Uscite: 12"])["answer"], 9)

    def test_same_net_different_operands_and_unrecognized_sources_go_to_review(self):
        self.assertEqual(self.repair.assess([APPROVED, APPROVED])["answer"], 9)
        for reports, reason in [
            ([APPROVED, "Spedite: 22; Stornate: 13"], "conflicting_reports"),
            ([APPROVED, ARRIVAL], "unrecognized_format_among_multiple"),
            ([ARRIVAL, ARRIVAL], "unrecognized_format_among_multiple"),
        ]:
            with self.subTest(reports=reports):
                assessment = self.repair.assess(reports)
                self.assertEqual(assessment["status"], "review")
                self.assertEqual(assessment["reason"], reason)
                self.assertNotIn("answer", assessment)

        proposal = self.repair.assess([ARRIVAL])["proposal"]
        self.assertEqual(self.repair.activate(proposal, "uscite", checks())["status"], "activated")
        assessment = self.repair.assess([APPROVED, "Uscite: 22; Resi: 13"])
        self.assertEqual(assessment["status"], "review")
        self.assertEqual(assessment["reason"], "conflicting_reports")
        self.assertNotIn("answer", assessment)

    def test_incomplete_or_unsafe_input_never_produces_a_proposal(self):
        self.assertEqual(self.repair.assess([])["reason"], "no_reports")
        for malformed in (
            "Spedite: 1000001; Stornate: 1",
            "Spedite: 2; Stornate: -1",
            "Spedite: 2; Stornate: 0.1",
            "Spedite: 2; Stornate: １",
            "Spedite: 2; Stornate: 1; Altro: 4",
            "Spedite: 2",
            "Riepilogo spedizioni\nSpedite: 2; Stornate: 1",
            "Spedite: 2; Spedite: 1",
        ):
            with self.subTest(malformed=malformed):
                assessment = self.repair.assess([malformed])
                self.assertEqual(assessment["status"], "review")
                self.assertEqual(assessment["reason"], "malformed_report")
                self.assertNotIn("proposal", assessment)
                self.assertNotIn("answer", assessment)
        self.assertEqual(self.repair.assess(["Stornate = 0\nSpEdItE : 1000000"])["answer"], 1000000)

    def test_rejected_candidate_leaves_previous_state_and_can_be_corrected(self):
        proposal = self.repair.assess([ARRIVAL])["proposal"]
        incorrect = checks()
        incorrect[0]["expected"] = 34
        for examples in (incorrect, checks()[:-1], checks()[1:]):
            with self.subTest(size=len(examples)):
                self.assertEqual(self.repair.activate(proposal, "uscite", examples)["status"], "rejected")
                restarted = FormatRepair(self.path)
                self.assertEqual(restarted.assess([APPROVED])["answer"], 9)
                self.assertEqual(restarted.assess([ARRIVAL])["proposal"]["revision"], proposal["revision"])
        self.assertEqual(self.repair.activate(proposal, "uscite", checks())["status"], "activated")

    def test_semantic_duplicates_cannot_pad_qualification(self):
        proposal = self.repair.assess([ARRIVAL])["proposal"]
        repeated_pair = checks()
        repeated_pair[0] = {"reports": ["Resi=6\nUSCITE=42"], "expected": 36}
        self.assertEqual(self.repair.activate(proposal, "uscite", repeated_pair)["status"], "rejected")
        arrival_pair = checks()
        arrival_pair[0] = {"reports": ["Resi=3\nUSCITE=20"], "expected": 17}
        self.assertEqual(self.repair.activate(proposal, "uscite", arrival_pair)["status"], "rejected")
        self.assertEqual(self.repair.activate(proposal, "altro", checks())["status"], "rejected")
        self.assertEqual(self.repair.assess([ARRIVAL])["status"], "clarification")

    def test_equal_quantity_examples_do_not_disambiguate_field_roles(self):
        proposal = self.repair.assess([ARRIVAL])["proposal"]
        examples = [
            {"reports": [f"Uscite: {count}; Resi: {count}"], "expected": 0}
            for count in range(30, 38)
        ] + checks()[8:]
        outcome = self.repair.activate(proposal, "uscite", examples)
        self.assertEqual(outcome["status"], "rejected")
        self.assertEqual(self.repair.assess([ARRIVAL])["proposal"], proposal)

    def test_activation_replays_persisted_review_expectations(self):
        first = self.repair.assess([ARRIVAL])["proposal"]
        first_checks = checks()
        first_checks[-1] = {
            "reports": [APPROVED, "Caricate: 12; Annullate: 3"],
            "expected": None,
        }
        self.assertEqual(self.repair.activate(first, "uscite", first_checks)["status"], "activated")

        second = "Caricate: 28; Annullate: 8"
        restarted = FormatRepair(self.path)
        proposal = restarted.assess([second])["proposal"]
        outcome = restarted.activate(proposal, "caricate", checks("Caricate", "Annullate"))
        self.assertEqual(outcome["status"], "rejected")
        self.assertEqual(outcome["reason"], "historical_regression")
        self.assertEqual(restarted.assess(first_checks[-1]["reports"])["status"], "review")
        self.assertEqual(restarted.assess([second])["proposal"], proposal)

    def test_initial_protected_review_blocks_activation(self):
        proposal = self.repair.assess([ARRIVAL])["proposal"]
        protected = [{"reports": [APPROVED, "Uscite: 12; Resi: 3"], "expected": None}]
        outcome = self.repair.activate(proposal, "uscite", checks(), protected=protected)
        self.assertEqual(outcome["reason"], "historical_regression")
        self.assertEqual(self.repair.assess(protected[0]["reports"])["status"], "review")
        self.assertEqual(self.repair.assess([ARRIVAL])["proposal"], proposal)

    def test_malformed_escaped_unicode_can_be_persisted_as_a_review_fixture(self):
        assessment = self.repair.assess([ARRIVAL])
        malformed = {"reports": ["Resi: \ud800"], "expected": None}
        qualification = checks()
        qualification[-1] = malformed
        activated = self.repair.activate(assessment["proposal"], "uscite", qualification)
        self.assertEqual(activated["status"], "activated")

        restarted = FormatRepair(self.path)
        self.assertEqual(restarted.assess(malformed["reports"])["status"], "review")
        second = restarted.assess(["Caricate: 28; Annullate: 8"])["proposal"]
        result = restarted.activate(second, "caricate", checks("Caricate", "Annullate"))
        self.assertEqual(result["status"], "activated")
        self.assertEqual(result["historical_cases"], len(qualification))

    def test_stale_proposal_cannot_write_and_can_be_reissued(self):
        second = "Caricate: 28; Annullate: 8"
        first_proposal = self.repair.assess([ARRIVAL])["proposal"]
        old_proposal = self.repair.assess([second])["proposal"]
        self.assertEqual(self.repair.activate(first_proposal, "uscite", checks())["status"], "activated")
        restarted = FormatRepair(self.path)
        qualification = checks("Caricate", "Annullate")
        outcome = restarted.activate(old_proposal, "caricate", qualification)
        self.assertEqual(outcome["status"], "rejected")
        self.assertEqual(outcome["reason"], "stale_proposal")
        current = restarted.assess([second])["proposal"]
        self.assertGreater(current["revision"], old_proposal["revision"])
        self.assertEqual(restarted.activate(current, "caricate", qualification)["status"], "activated")
        self.assertEqual(restarted.assess([ARRIVAL])["answer"], 17)
        self.assertEqual(restarted.assess([second])["answer"], 20)

    def test_concurrent_writers_cannot_both_activate_at_the_same_revision(self):
        second = "Caricate: 28; Annullate: 8"
        proposals = [self.repair.assess([report])["proposal"] for report in (ARRIVAL, second)]
        ready = Barrier(2)

        def writer(proposal, shipped, qualification):
            instance = FormatRepair(self.path)
            ready.wait(timeout=5)
            return instance.activate(proposal, shipped, qualification)

        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [
                pool.submit(writer, proposals[0], "uscite", checks()),
                pool.submit(writer, proposals[1], "caricate", checks("Caricate", "Annullate")),
            ]
            outcomes = [job.result(timeout=15) for job in jobs]
        self.assertCountEqual([outcome["status"] for outcome in outcomes], ["activated", "rejected"])
        restarted = FormatRepair(self.path)
        for report, outcome in zip((ARRIVAL, second), outcomes, strict=True):
            with self.subTest(report=report):
                current = restarted.assess([report])
                if outcome["status"] == "activated":
                    self.assertEqual(current["status"], "answered")
                else:
                    self.assertEqual(outcome["reason"], "stale_proposal")
                    self.assertEqual(current["status"], "clarification")
                self.assertEqual(current["revision"], 1)

    def test_process_exit_before_commit_cannot_publish_a_candidate(self):
        worker = """
import os
import sqlite3
import sys

from repair import FormatRepair
from test_repair import ARRIVAL, checks


class InterruptedConnection(sqlite3.Connection):
    def execute(self, sql, *args, **kwargs):
        result = super().execute(sql, *args, **kwargs)
        if sql.startswith("UPDATE registry"):
            os._exit(74)
        return result


class InterruptedRepair(FormatRepair):
    def _connect(self):
        return sqlite3.connect(self.path, timeout=10, factory=InterruptedConnection)


repair = InterruptedRepair(sys.argv[1])
assessment = repair.assess([ARRIVAL])
repair.activate(assessment["proposal"], "uscite", checks())
"""
        interrupted = subprocess.run(
            [sys.executable, "-c", worker, str(self.path)],
            cwd=CLI.parent, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(interrupted.returncode, 74, interrupted.stderr)
        recovered = FormatRepair(self.path)
        self.assertEqual(recovered.assess([APPROVED])["answer"], 9)
        assessment = recovered.assess([ARRIVAL])
        self.assertEqual(assessment["status"], "clarification")
        self.assertEqual(assessment["revision"], 0)
        self.assertEqual(recovered.activate(assessment["proposal"], "uscite", checks())["status"], "activated")

    def test_cli_handoffs_between_processes(self):
        directory = self.path.parent
        reports = directory / "reports.json"
        assessment = directory / "assessment.json"
        qualification = directory / "checks.json"
        protected = directory / "protected.json"
        reports.write_text(json.dumps([ARRIVAL]), encoding="utf-8")
        qualification.write_text(json.dumps(checks()), encoding="utf-8")
        protected.write_text(json.dumps([{"reports": [APPROVED], "expected": 9}]), encoding="utf-8")
        command = [sys.executable, str(CLI), "--state", str(self.path)]

        def run(*args):
            return subprocess.run([*command, *map(str, args)], capture_output=True, text=True, timeout=15)

        prior = run("assess", reports)
        self.assertEqual(prior.returncode, 0, prior.stderr)
        self.assertEqual(json.loads(prior.stdout)["status"], "clarification")
        assessment.write_text(prior.stdout, encoding="utf-8")
        activated = run("activate", assessment, "Uscite", qualification, "--protected", protected)
        self.assertEqual(activated.returncode, 0, activated.stderr)
        self.assertEqual(json.loads(activated.stdout)["historical_cases"], 1)
        answered = run("assess", reports)
        self.assertEqual(answered.returncode, 0, answered.stderr)
        self.assertEqual(json.loads(answered.stdout)["answer"], 17)
        stale = run("activate", assessment, "Uscite", qualification)
        self.assertEqual(stale.returncode, 1, stale.stderr)
        self.assertEqual(json.loads(stale.stdout)["status"], "rejected")

        reports.write_text('{"reports":[]}', encoding="utf-8")
        malformed = run("assess", reports)
        self.assertEqual(malformed.returncode, 2)
        self.assertIn("JSON array of strings", malformed.stderr)


if __name__ == "__main__":
    unittest.main()
