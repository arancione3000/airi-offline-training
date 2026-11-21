import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'scripts/research-training'
sys.path.insert(0, str(RESEARCH / 'joint-action-20261121'))
import freeze_donations as donation

spec = importlib.util.spec_from_file_location('joint_transfer_cycle', RESEARCH / 'exact-transfer-20261120/cycle.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class JointDonationTests(unittest.TestCase):
    def test_receipt_replays_and_only_independently_confirmed_predictions_enter_catalog(self):
        rows, families = transfer.read_corpus()
        receipt = json.loads(donation.RECEIPT.read_bytes())
        self.assertEqual(donation.freeze(families), receipt)
        donor_rows = {r['id'].removeprefix('joint-donor-'): r for r in rows if r['id'].startswith('joint-donor-')}
        self.assertEqual(set(donor_rows), set(receipt['predictions']))
        bank = transfer.SkillBank(transfer.BUNDLED)
        entries = bank._read()[0]['entries']
        catalog = transfer.catalog(rows, transfer.bindings(bank, families), entries)
        emitted = {b['train_id'].removeprefix('joint-donor-') for b in catalog.values() if b['train_id'].startswith('joint-donor-')}
        independently_confirmed = {key for key, r in donor_rows.items()
                                   if r['lesson_id'] is not None and r['lesson_id'] == receipt['predictions'][key]}
        self.assertEqual(emitted, independently_confirmed)
        self.assertTrue(emitted)
        rejected_by_review = [r for key, r in donor_rows.items() if r['lesson_id'] is None and receipt['predictions'][key]]
        self.assertTrue(rejected_by_review)
        self.assertTrue(all(r['condition'] == 'unsupported_donation' and r['query'] not in catalog for r in rejected_by_review))

    def test_corrupt_receipt_or_changed_source_cannot_publish_a_new_batch(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            snapshot = root / 'aliases.json'
            bank = root / 'bank.json'
            bank.write_bytes(transfer.BUNDLED.read_bytes())
            before = transfer.DEFAULT.read_bytes()
            snapshot.write_bytes(before)
            corrupt = root / 'receipt.json'
            corrupt.write_bytes(donation.RECEIPT.read_bytes() + b' ')
            def read_corrupt(families, expected):
                return donation.reviewed_rows(families, expected, corrupt)
            with patch.object(transfer, 'reviewed_rows', read_corrupt), self.assertRaisesRegex(ValueError, 'wrong frozen'):
                transfer.cycle(bank, snapshot, limit=8, seed=47)
            self.assertEqual(snapshot.read_bytes(), before)
            with patch.object(donation, 'input_hashes', return_value={}), self.assertRaisesRegex(ValueError, 'stale TRAIN'):
                transfer.cycle(bank, snapshot, limit=8, seed=47)
            self.assertEqual(snapshot.read_bytes(), before)

    def test_unconfirmed_positive_can_be_handled_by_an_independent_semantic_route(self):
        with gzip.open(donation.FAMILIES, 'rt') as stream:
            families = json.load(stream)['families']
        rows = donation.reviewed_rows(families, transfer.DONATION_RECEIPT_SHA256)
        row = next(r for r in rows if r['condition'] == 'unconfirmed_donation')
        family = next(f for f in families if f['id'] == row['source_lesson_id'])
        state = transfer.SkillBank(transfer.BUNDLED)._read()[0]
        class Semantic:
            def __init__(self, wrong):
                self.wrong = wrong

            def predict(self, messages):
                for variant in range(2):
                    expected_messages, _ = transfer.case(family, row['query'], 1000, variant)
                    if messages == expected_messages:
                        canonical_question = {**messages[-1], 'content': family['canonical_query']}
                        expected = transfer._route(state['entries'], [*messages[:-1], canonical_question])
                        return {**expected, 'answer': 'wrong'} if self.wrong else expected
                return None
        good = transfer.audit(state, {}, families, [], seed=0, semantic=Semantic(False), unconfirmed=[row])
        self.assertEqual(good['unconfirmed_donor_executions'], 2)
        with self.assertRaisesRegex(transfer.RegistryError, 'unconfirmed donor'):
            transfer.audit(state, {}, families, [], seed=0, semantic=Semantic(True), unconfirmed=[row])


if __name__ == '__main__':
    unittest.main()
