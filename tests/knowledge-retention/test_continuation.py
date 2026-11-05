import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location('retention_continuation', ROOT / 'scripts/knowledge-retention/continue.py')
continuation_module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(continuation_module)


class RetentionContinuationTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT / 'scripts/knowledge-retention/SPEC.json').read_text())
        self.state = dict(fingerprint=self.spec['fingerprint'], job_index=0,
                          completed_steps=self.spec['start_step'] + 16,
                          **{'continue': True}, production_qualified=False, live_promoted=False)

    def arguments(self, *, state=None, continuation=0):
        state = self.state if state is None else state
        raw = json.dumps(state).encode()
        index = dict(fingerprint=self.spec['fingerprint'], completed_steps=state['completed_steps'],
                     run_state=dict(name='AIRI_RUN_STATE.json', bytes=len(raw),
                                    sha256=hashlib.sha256(raw).hexdigest()))
        return dict(action={'release': 'airi-offline-449292-123-1-2'}, spec=self.spec,
                    continuation=continuation, branch=self.spec['frozen_runner_branch'],
                    runner_sha='a'*40, branch_sha='a'*40, index=index, state_bytes=raw)

    def test_verified_checkpoint_continues_with_exact_branch_and_counter(self):
        args = self.arguments()
        self.assertEqual(continuation_module.continuation_payload(**args),
                         {'ref': self.spec['frozen_runner_branch'],
                          'inputs': {'checkpoint_release': args['action']['release'],
                                     'continuation': '1', 'preflight_only': 'false'}})

    def test_fourth_job_keeps_checkpoint_and_does_not_dispatch_a_fifth(self):
        state = {**self.state, 'job_index': 3}
        self.assertIsNone(continuation_module.continuation_payload(**self.arguments(state=state, continuation=3)))

    def test_completed_experiment_requires_no_checkpoint_download_or_dispatch(self):
        args = self.arguments()
        args.update(action=None, index=None, state_bytes=None)
        self.assertIsNone(continuation_module.continuation_payload(**args))

    def test_branch_movement_and_another_branch_cannot_change_resumed_code(self):
        for change in ({'branch_sha': 'b'*40}, {'branch': 'main'}):
            args = self.arguments(); args.update(change)
            with self.assertRaisesRegex(ValueError, 'runner branch changed'):
                continuation_module.continuation_payload(**args)

    def test_modified_or_missing_state_cannot_trigger_continuation(self):
        for change in ({'state_bytes': b'{}'}, {'state_bytes': None}):
            args = self.arguments(); args.update(change)
            with self.assertRaisesRegex(ValueError, 'hash-committed'):
                continuation_module.continuation_payload(**args)

    def test_another_recipe_or_uncommitted_state_cannot_continue(self):
        args = self.arguments(); args['index']['fingerprint'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            continuation_module.continuation_payload(**args)
        for change in ({'fingerprint': 'foreign'}, {'job_index': 1},
                       {'continue': False}, {'live_promoted': True},
                       {'completed_steps': self.spec['max_steps'] + 1}):
            args = self.arguments(state={**self.state, **change})
            with self.assertRaisesRegex(ValueError, 'does not match'):
                continuation_module.continuation_payload(**args)

    def test_invalid_release_or_job_number_cannot_be_dispatched(self):
        for action in ({'release': '../outside'}, {}, {'release': 'airi-offline-ok', 'other': True}):
            args = self.arguments(); args['action'] = action
            with self.assertRaises(ValueError):
                continuation_module.continuation_payload(**args)
        for count in (-1, 4, True, 1.5):
            args = self.arguments(); args['continuation'] = count
            with self.assertRaises(ValueError):
                continuation_module.continuation_payload(**args)


if __name__ == '__main__':
    unittest.main()
