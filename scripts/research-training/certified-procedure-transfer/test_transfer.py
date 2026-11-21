import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from chat_adapter import QUESTIONS, make_assistant
from transfer import encode, evidence_digest, issue, load_contract, verify


class TransferTests(unittest.TestCase):
    def load(self, rows):
        receipt = issue(rows)
        return verify(encode(receipt), evidence_digest(receipt))

    def test_ambiguous_receipt_serves_consensus_and_abstains_on_disagreement(self):
        certified = self.load([])
        self.assertEqual(certified.execute('exact_division', 7, -3), 'false')
        self.assertEqual(certified.execute('quotient', 6, -3), '-2')
        self.assertIsNone(certified.execute('quotient', 7, -3))
        self.assertIsNone(certified.execute('quotient', 3, 0))
        self.assertIsNone(certified.execute('quotient', True, 1))

    def test_quotient_example_transfers_to_independently_routed_remainder(self):
        certified = self.load([['quotient', 7, -3, '-3']])
        assistant = make_assistant(lambda _: 'fallback', certified)
        messages = [{'role': 'user', 'content': 'Valori divisione: -19 e -6.'},
                    {'role': 'user', 'content': QUESTIONS['remainder']}]
        result = assistant.respond(messages)
        self.assertEqual(result['source'], 'certified_procedure')
        self.assertEqual(result['answer'], '-1')
        self.assertEqual(result['certificate_sha256'], certified.fingerprint)
        self.assertEqual(result['provenance']['operands'][1]['value'], -6)

    def test_replay_binds_policy_to_evidence_and_contract(self):
        receipt = issue([['quotient', 7, -3, '-3']])
        trusted = evidence_digest(receipt)
        altered = json.loads(encode(receipt))
        altered['policies'] = ['euclidean']
        with self.assertRaisesRegex(ValueError, 'differs from a replay'):
            verify(encode(altered), trusted)
        duplicate_key = encode(receipt).replace(b'"schema":1', b'"schema":0,"schema":1')
        with self.assertRaisesRegex(ValueError, 'differs from a replay'):
            verify(duplicate_key, trusted)
        altered = issue([['quotient', -7, 3, '-2']])
        with self.assertRaisesRegex(ValueError, 'evidence digest'):
            verify(encode(altered), trusted)
        altered['evidence']['contract_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'differs from a replay'):
            verify(encode(altered), evidence_digest(altered))
        with TemporaryDirectory() as tmp:
            contract = Path(tmp) / 'contract.json'
            contract.write_text('{"schema": 1}\n')
            with self.assertRaisesRegex(ValueError, 'pinned specification'):
                load_contract(contract)

    def test_consistent_in_isolation_but_incompatible_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'contradict'):
            issue([['quotient', 7, -3, '-3'], ['quotient', -7, 3, '-2']])
        with self.assertRaisesRegex(ValueError, 'invalid or repeated'):
            issue([['quotient', 7, -3, '-3']] * 2)


if __name__ == '__main__':
    unittest.main()
