import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/assistant-runtime'))
from exact_aliases import AliasRouter, read_aliases
from learn import BUNDLED
from skill_bank import RegistryError, SkillBank

spec = importlib.util.spec_from_file_location(
    'exact_transfer_cycle', ROOT / 'scripts/research-training/exact-transfer-20261120/cycle.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class ExactTransferTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name) / 'aliases.json'
        self.bank = SkillBank(Path(tmp.name) / 'bank.json')
        self.bank.path.write_bytes(BUNDLED.read_bytes())

    def cycle(self, limit=4):
        return transfer.cycle(self.bank.path, self.state, seed=327, limit=limit)

    def test_runtime_hot_refresh_restart_and_revision_preservation(self):
        bank_before = self.bank.path.read_bytes()
        router = AliasRouter(self.bank, self.state)
        self.assertEqual(router.match([{'role': 'user', 'content': 'absent'}]), (False, None))
        report = self.cycle()
        self.assertEqual(report['activated'], 4)
        aliases = read_aliases(self.state)
        query, binding = next(iter(aliases.items()))
        entry = next(e for e in self.bank._read()[0]['entries'] if e['id'] == binding['program_id'])
        row = entry['qualification'][0]
        original = row['messages']
        messages = [*original[:-1], {**original[-1], 'content': query}]
        result = router.predict(messages)
        self.assertEqual(result['answer'], row['answer'])
        self.assertEqual(result['skill_id'], entry['id'])
        self.assertEqual(result['query_provenance']['original_query'], query)
        self.assertEqual(result['provenance'], self.bank.predict(original)['provenance'])
        self.assertEqual(router.match([messages[-1]]), (True, None))
        self.assertIsNone(router.predict([{**messages[-1], 'role':'assistant'}]))
        self.assertEqual(self.cycle()['total_aliases'], 8)
        self.assertEqual(AliasRouter(self.bank, self.state).predict(messages), result)
        self.assertEqual(bank_before, self.bank.path.read_bytes())

    def test_no_publication_after_failed_audit_or_corrupt_state(self):
        self.cycle(1)
        before = self.state.read_bytes()
        with patch.object(transfer, 'audit', side_effect=RegistryError('retention failed')):
            with self.assertRaises(RegistryError):
                self.cycle()
        self.assertEqual(before, self.state.read_bytes())
        value = json.loads(before)
        value['aliases'].clear()
        self.state.write_text(json.dumps(value))
        with self.assertRaises(RegistryError):
            self.cycle()

    def test_identical_exact_upgrade_keeps_record_but_competing_source_vetoes_publication(self):
        self.cycle(1)
        before = self.state.read_bytes()
        aliases = read_aliases(self.state)
        query, binding = next(iter(aliases.items()))
        state = self.bank._read()[0]
        parent = next(e for e in state['entries'] if e['id'] == binding['program_id'])
        duplicate = copy.deepcopy(parent)
        duplicate['program']['query'] = query
        duplicate['id'] = transfer.digest(duplicate['program'])
        duplicate['lesson_id'] = 'future-exact'
        state['entries'].append(duplicate)
        families = json.loads(gzip.decompress((transfer.CORPUS/'FAMILIES.json.gz').read_bytes()))['families']
        rows = json.loads(gzip.decompress((transfer.CORPUS/'TRAIN.json.gz').read_bytes()))['rows']
        candidates = transfer.catalog(rows, transfer.bindings(self.bank, families), state['entries'], aliases)
        self.assertEqual(candidates[query], binding)
        with self.assertRaisesRegex(RegistryError, 'source/intent boundary'):
            transfer.audit(state, aliases, families, [], 23)
        duplicate['program']['operation'] = 'wrong'
        with self.assertRaisesRegex(RegistryError, 'conflicting exact expert'):
            transfer.audit(state, aliases, families, [], 23)
        self.assertEqual(self.state.read_bytes(), before)

    def test_ambiguous_and_unsupported_train_rows_are_excluded(self):
        example = dict(query='Q?', lesson_id='a', source_lesson_id='a',
                       condition='supported_paraphrase', id='1')
        rows = [example, {**example, 'lesson_id': None}]
        self.assertEqual(transfer.catalog(rows, [dict(lesson_id='a', program_id='id')], []), {})


if __name__ == '__main__':
    unittest.main()
