import copy
import os
import importlib.util
import json
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[2] / 'scripts/long50-extension/extend.py'
spec = importlib.util.spec_from_file_location('extension', path)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)

def fixture():
    step = 160000
    run = dict(id=123, run_attempt=1, head_sha=extension.SOURCE_SHA,
               head_branch=extension.SOURCE_BRANCH, status='completed', conclusion='success')
    release = dict(tag_name=f'airi-offline-{step}-123-1-1', draft=False, prerelease=True, target_commitish=extension.SOURCE_SHA,
                   assets=[{'name': 'FINAL_RESULT.json', 'id': 1},
                           {'name': f'AIRI_TRAINING_CHECKPOINT_{step}.json', 'id': 2}])
    result = dict(counters={'elapsed_training_seconds': 43200.1, 'completed_steps': step},
                  next_action=None, maximum_training_seconds=43200, production_qualified=False, live_promoted=False)
    index = dict(fingerprint=extension.SOURCE_FP, completed_steps=step, production_qualified=False, live_promoted=False, archives=[
        dict(name=f'AIRI_TRAINING_{step}_{part}.zip', bytes=100, sha256='a'*64)
        for part in ('MODEL', 'OPTIMIZER')])
    return run, release, result, index

class Fake:
    def __init__(self, *, active=True, existing=False):
        self.run, self.release, self.result, self.index = fixture()
        if active:
            self.run.update(status='in_progress', conclusion=None)
        self.existing = existing
        self.calls = []
        self.extra_runs = []
        self.slot_busy = False

    def branch(self):
        return {'object': {'sha': 'extra-sha'}} if self.existing else None

    def pages(self, path, key=None):
        if path == 'releases': return iter([self.release])
        if 'branch=' + extension.SOURCE_BRANCH in path: return iter([self.run])
        if 'branch=' + extension.BRANCH in path: return iter(self.extra_runs)
        if 'status=' in path:
            return iter([{'path': '.github/workflows/airi-human-background.yml'}] if self.slot_busy else [])
        raise AssertionError(path)

    def content(self, path, ref):
        assert ref == 'extra-sha'
        return json.dumps(dict(extension=dict(request=extension.REQUEST, source_sha=extension.SOURCE_SHA),
                               maximum_training_seconds=180000, migration_source_release='airi-offline-160000-123-1-1')).encode()

    def asset(self, item):
        return self.result if item['id'] == 1 else self.index

    def api(self, path, payload=None):
        self.calls.append((path, payload))

class ExtensionTests(unittest.TestCase):
    def test_completion_trigger_matches_frozen_workflow_name(self):
        source_root = Path(os.environ.get('AIRI_FROZEN_SOURCE', str(Path(__file__).resolve().parents[3] / 'airi-human-long-20261013')))
        training = (source_root / '.github/workflows/airi-human-background.yml').read_text()
        name = next(line.removeprefix('name: ').strip() for line in training.splitlines() if line.startswith('name: '))
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/airi-long50-extension.yml').read_text()
        trigger = next(line.split('workflows: [', 1)[1].split(']', 1)[0] for line in workflow.splitlines() if 'workflows: [' in line)
        self.assertIn(name, [item.strip() for item in trigger.split(',')])

    def test_waits_without_mutation_or_training_overlap(self):
        client = Fake()
        self.assertEqual(extension.tick(client)['action'], 'waiting_for_current_cycle')
        self.assertEqual(client.calls, [])

    def test_changed_source_stops_without_dispatch(self):
        client = Fake(); client.run['head_sha'] = 'wrong'
        with self.assertRaises(ValueError): extension.tick(client)
        self.assertEqual(client.calls, [])

    def test_unsuccessful_source_is_not_restarted(self):
        for conclusion in ('failure', 'cancelled', 'timed_out'):
            client = Fake(active=False); client.run['conclusion'] = conclusion
            self.assertEqual(extension.tick(client)['action'], 'source_stopped_without_success')
            self.assertEqual(client.calls, [])

    def test_rejects_partial_nonfinite_mismatched_or_live_terminal(self):
        for change in ('partial', 'nan', 'inf', 'wrongfp', 'wrongstep', 'missing_optimizer', 'duplicate_archive', 'live', 'draft', 'continuing'):
            run, release, result, index = fixture()
            if change == 'partial': result['counters']['elapsed_training_seconds'] = 43199
            if change == 'nan': result['counters']['elapsed_training_seconds'] = float('nan')
            if change == 'inf': result['counters']['elapsed_training_seconds'] = float('inf')
            if change == 'wrongfp': index['fingerprint'] = 'wrong'
            if change == 'wrongstep': index['completed_steps'] += 1
            if change == 'missing_optimizer': index['archives'].pop()
            if change == 'duplicate_archive': index['archives'][1] = index['archives'][0]
            if change == 'live': result['live_promoted'] = True
            if change == 'draft': release['draft'] = True
            if change == 'continuing': result['next_action'] = {'release': 'more'}
            with self.subTest(change=change), self.assertRaises(ValueError):
                extension.validate_terminal(result, index, release, run)

    def test_accepts_only_complete_exact_terminal_seed(self):
        run, release, result, index = fixture()
        self.assertEqual(extension.validate_terminal(result, index, release, run), 160000)

    def test_existing_successor_run_cannot_be_dispatched_twice(self):
        client = Fake(existing=True)
        client.extra_runs = [dict(id=200, head_sha='extra-sha', status='in_progress', conclusion=None)]
        self.assertEqual(extension.tick(client)['action'], 'already_started')
        self.assertEqual(client.calls, [])

    def test_existing_successor_waits_for_free_slot(self):
        client = Fake(existing=True); client.slot_busy = True
        self.assertEqual(extension.tick(client)['action'], 'waiting_for_training_slot')
        self.assertEqual(client.calls, [])

    def test_dispatch_preserves_exact_checkpoint_and_finite_continuation(self):
        client = Fake(existing=True)
        self.assertEqual(extension.tick(client)['action'], 'dispatched')
        self.assertEqual(client.calls, [('actions/workflows/airi-human-background.yml/dispatches',
            {'ref': extension.BRANCH, 'inputs': {'checkpoint_release': 'airi-offline-160000-123-1-1', 'continuation': '0'}})])

    def test_self_disables_after_verified_fifty_hour_result(self):
        client = Fake(existing=True)
        client.extra_runs = [dict(id=123, run_attempt=1, head_sha='extra-sha', status='completed', conclusion='success')]
        client.result['maximum_training_seconds'] = 180000
        client.result['counters']['elapsed_training_seconds'] = 180000.2
        self.assertEqual(extension.tick(client)['action'], 'fifty_hours_completed_controller_disabled')
        self.assertEqual(client.calls, [('actions/workflows/airi-long50-extension.yml/disable', {})])

    def test_intentional_cancellation_disables_extension_without_rerun(self):
        client = Fake(existing=True)
        client.extra_runs = [dict(id=123, head_sha='extra-sha', status='completed', conclusion='cancelled')]
        self.assertEqual(extension.tick(client)['action'], 'intentional_cancellation_respected')
        self.assertEqual(client.calls, [('actions/workflows/airi-long50-extension.yml/disable', {})])

    def test_builds_successor_from_verified_public_source_without_mutating_runner(self):
        source_root = Path(os.environ.get('AIRI_FROZEN_SOURCE', str(Path(__file__).resolve().parents[3] / 'airi-human-long-20261013')))
        spec = json.loads((source_root / 'scripts/transfer-training/SPEC.json').read_text())
        paths = ['scripts/transfer-training/SPEC.json', 'tests/transfer-training/test_diagnostics.py'] + ['scripts/transfer-training/' + n for n in spec['script_sha256']]
        source = {p: (source_root / p).read_bytes() for p in paths}
        run, release, result, index = fixture()
        files = extension.build_files(source, result, index, release, run)
        self.assertEqual(set(files), {'scripts/transfer-training/SPEC.json', 'tests/transfer-training/test_diagnostics.py'})
        new = json.loads(files['scripts/transfer-training/SPEC.json'])
        self.assertEqual(new['start_step'], 160000)
        self.assertEqual(new['maximum_training_seconds'], 180000)
        self.assertEqual(new['script_sha256'], spec['script_sha256'])
        self.assertEqual(new['recipes'], [{'learning_rate': 2e-7, 'kl_weight': 1.0}] * 3)
        fp = new.pop('fingerprint')
        self.assertEqual(fp, extension.sha(json.dumps(new, sort_keys=True, separators=(',', ':')).encode()))

    def test_generated_fifty_hour_runner_passes_real_regression_suite(self):
        import shutil,subprocess,sys,tempfile
        source_root=Path(os.environ.get('AIRI_FROZEN_SOURCE', str(Path(__file__).resolve().parents[3]/'airi-human-long-20261013')))
        original=json.loads((source_root/'scripts/transfer-training/SPEC.json').read_text())
        paths=['scripts/transfer-training/SPEC.json','tests/transfer-training/test_diagnostics.py']+['scripts/transfer-training/'+n for n in original['script_sha256']]
        source={p:(source_root/p).read_bytes() for p in paths}
        run,release,result,index=fixture()
        files=extension.build_files(source,result,index,release,run)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['scripts/transfer-training','tests/transfer-training','data/oasst2']:
                shutil.copytree(source_root/name,root/name)
            for name,content in files.items():(root/name).write_text(content)
            check=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/transfer-training','-p','test_*.py','-v'],cwd=root,capture_output=True,text=True,timeout=60)
            self.assertEqual(check.returncode,0,check.stdout+check.stderr)
            spec=json.loads((root/'scripts/transfer-training/SPEC.json').read_text())
            self.assertEqual(spec['maximum_training_seconds'],50*3600)
            self.assertEqual(spec['script_sha256'],original['script_sha256'])

if __name__ == '__main__': unittest.main()
