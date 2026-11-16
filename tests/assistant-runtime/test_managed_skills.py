from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_skill_bank import lesson
from skill_bank import RegistryError, SkillBank
from managed_skills import reconcile
import learn


def distinct(index):
    return lesson(f'public_{index}',f'Public {index}: ',f'What is public value {index}?')


class ManagedSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.public=SkillBank(self.root/'public.json')
        self.private=SkillBank(self.root/'private.json')
        for index in range(6):
            self.assertEqual(self.public.learn(distinct(index))['action'],'activated')
        self.private._write(self.public._read()[0])

    def extend_public(self):
        for index in (6,7):
            self.assertEqual(self.public.learn(distinct(index))['action'],'activated')

    def test_existing_private_six_receive_public_eight_and_keep_private_addition(self):
        extra=lesson('private_extra','Private value: ','What is my private value?')
        self.private.learn(extra)
        old=self.private._read()[0]['entries']
        self.extend_public()
        result=reconcile(self.private,self.public.path)
        self.assertEqual(result['added'],2)
        self.assertEqual(self.private.status()['revision'],9)
        self.assertEqual(self.private._read()[0]['entries'][:7],old)
        for data in [extra,distinct(6),distinct(7)]:
            row=data['qualification'][0]
            self.assertEqual(self.private.predict(row['messages'])['answer'],row['answer'])
        before=self.private.path.read_bytes()
        self.assertEqual(reconcile(self.private,self.public.path)['action'],'unchanged')
        self.assertEqual(before,self.private.path.read_bytes())
        # The two imports share one private commit rather than partially publishing.
        self.assertEqual(sorted(p.name[:6] for p in self.private.snapshots.glob('*.json')),['000006','000007','000009'])

    def test_later_rejected_public_expert_does_not_publish_earlier_staged_expert(self):
        conflict=lesson('public_conflict','Shared token: ','What is the shared token?')
        private_conflict=copy.deepcopy(conflict);private_conflict['lesson_id']='private_reverse'
        for row in private_conflict['examples']+private_conflict['qualification']:
            if row['answer'] is not None:row['answer']=row['answer'][::-1]
        self.private.learn(private_conflict)
        self.public.learn(distinct(6));self.public.learn(conflict)
        before=self.private.path.read_bytes()
        with self.assertRaises(RegistryError):reconcile(self.private,self.public.path)
        self.assertEqual(before,self.private.path.read_bytes())
        self.assertIsNone(self.private.predict(distinct(6)['qualification'][0]['messages']))

    def test_concurrent_private_learning_wins_cas_without_losing_its_expert(self):
        self.extend_public()
        original_lock=self.private._lock
        concurrent=lesson('concurrent','Concurrent value: ','What is the concurrent value?')
        @contextmanager
        def race():
            SkillBank(self.private.path).learn(concurrent)
            with original_lock():yield
        with patch.object(self.private,'_lock',side_effect=race):
            with self.assertRaises(RegistryError):reconcile(self.private,self.public.path)
        self.assertEqual(self.private.status()['revision'],7)
        row=concurrent['qualification'][0]
        self.assertEqual(self.private.predict(row['messages'])['answer'],row['answer'])
        self.assertEqual(reconcile(self.private,self.public.path)['revision'],9)

    def test_default_selection_refreshes_but_explicit_paths_remain_independent(self):
        self.extend_public()
        with patch.object(learn,'DEFAULT_PRIVATE',self.private.path),patch.object(learn,'BUNDLED',self.public.path):
            self.assertEqual(learn.selected_bank(self.private.path).status()['revision'],6)
            self.assertEqual(learn.selected_bank().status()['revision'],8)
            self.assertEqual(learn.private_bank().status()['revision'],8)

    def test_bad_public_bank_keeps_private_answers_and_reports_refresh_failure(self):
        self.public.path.write_text('{corrupted')
        for path in self.public.snapshots.glob('*.json'):path.unlink()
        before=self.private.path.read_bytes();error=io.StringIO()
        with patch.object(learn,'DEFAULT_PRIVATE',self.private.path),patch.object(learn,'BUNDLED',self.public.path),patch('sys.stderr',error):
            selected=learn.selected_bank()
        row=distinct(0)['qualification'][0]
        self.assertEqual(selected.predict(row['messages'])['answer'],row['answer'])
        self.assertEqual(before,self.private.path.read_bytes())
        self.assertIn('refresh unavailable',error.getvalue())

    def test_running_default_watcher_imports_a_later_bundled_update(self):
        queue=self.root/'queue';queue.mkdir();output=io.StringIO()
        argv=['learn.py','--watch',str(queue),'--cycles','2','--interval','1']
        with patch.object(learn,'DEFAULT_PRIVATE',self.private.path),patch.object(learn,'BUNDLED',self.public.path),patch.object(sys,'argv',argv),patch('sys.stdout',output),patch.object(learn.time,'sleep',side_effect=lambda _:self.extend_public()):
            learn.main()
        self.assertEqual(self.private.status()['revision'],8)
        messages=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(any(row.get('managed_refresh',{}).get('added')==2 for row in messages))

    def test_queue_overflow_does_not_stop_watcher_and_bad_files_are_not_relogged(self):
        queue=self.root/'queue';queue.mkdir()
        for index in range(32):(queue/f'{index:02d}.json').write_text('{broken')
        (queue/'z-valid.json').write_text(json.dumps(distinct(10)))
        bank=self.root/'watched.json'
        command=[sys.executable,str(Path(learn.__file__).resolve()),'--watch',str(queue),
                 '--bank',str(bank),'--interval','1','--cycles','2']
        child=subprocess.run(command,capture_output=True,text=True,timeout=10,check=True)
        messages=[json.loads(line) for line in child.stdout.splitlines()]
        self.assertTrue(any(row.get('queue_health')=='overflow' for row in messages))
        self.assertEqual(sum(row.get('action')=='rejected' for row in messages),32)
        self.assertEqual(SkillBank(bank).status()['revision'],1)


if __name__=='__main__':unittest.main()
