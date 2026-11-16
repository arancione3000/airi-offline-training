import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assistant-runtime'))
import auto_update
from auto_curriculum import CATALOG,build
from learn import BUNDLED
from skill_bank import RegistryError,SkillBank


class AutomaticUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.bank=SkillBank(Path(self.tmp.name)/'bank.json')

    def test_real_two_cycles_resume_and_preserve_prior_skill_answers(self):
        first=auto_update.cycle(self.bank,seed=123,limit=2)
        self.assertEqual(first['status'],'updated');self.assertEqual(first['after_revision'],2)
        second=auto_update.cycle(SkillBank(self.bank.path),seed=456,limit=2)
        self.assertEqual(second['after_revision'],4);self.assertEqual(second['remaining_families'],8)
        self.assertEqual(second['audit']['correct'],80)
        self.assertTrue(second['stable_gate']['passed'])
        for family in CATALOG[:2]:
            case=build(family['id'],999)['heldout'][0]
            self.assertEqual(self.bank.predict(case['messages'])['answer'],case['answer'])

    def test_failed_heldout_never_replaces_live_bank(self):
        self.bank._write(SkillBank(BUNDLED)._read()[0]);before=self.bank.path.read_bytes()
        def corrupted(*args):
            bundle=copy.deepcopy(build(*args));bundle['heldout'][0]['answer']='wrong-heldout-oracle';return bundle
        with patch.object(auto_update,'build',side_effect=corrupted):
            with self.assertRaisesRegex(RegistryError,'new-value audit'):auto_update.cycle(self.bank,seed=42,limit=1)
        self.assertEqual(self.bank.path.read_bytes(),before)

    def test_concurrent_live_update_is_not_overwritten(self):
        original_hashes=auto_update._source_hashes;calls=0
        def changed_while_testing():
            nonlocal calls
            calls+=1
            if calls==2:self.bank.learn(build(CATALOG[-1]['id'],101)['lesson'])
            return original_hashes()
        with patch.object(auto_update,'_source_hashes',side_effect=changed_while_testing):
            with self.assertRaisesRegex(RegistryError,'concurrent'):auto_update.cycle(self.bank,seed=42,limit=1)
        self.assertEqual(self.bank.status()['revision'],1)
        self.assertEqual(self.bank._read()[0]['entries'][0]['lesson_id'],CATALOG[-1]['id'])

    def test_completed_catalog_audits_without_writing_or_eviction(self):
        for _ in range(3):auto_update.cycle(self.bank,seed=123,limit=4)
        before=self.bank.path.read_bytes()
        report=auto_update.cycle(self.bank,seed=987,limit=2)
        self.assertEqual(report['status'],'audited');self.assertEqual(report['activated'],0)
        self.assertEqual(report['audit']['correct'],240);self.assertEqual(before,self.bank.path.read_bytes())


if __name__=='__main__':unittest.main()
