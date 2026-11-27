"""Recovery regressions using real failure shapes and externally visible state."""
import contextlib
import io
import subprocess
import unittest
from unittest.mock import patch

from test_controller import Fake, FP, SHA, module


class ResilienceTests(unittest.TestCase):
    def broken_run(self, conclusion='failure', attempt=3):
        f = Fake()
        f.finished()
        f.run.update(conclusion=conclusion, run_attempt=attempt)
        f.log = 'stream error: stream ID 5; PROTOCOL_ERROR; received from peer'
        return f

    def rollover(self):
        f = self.broken_run()
        source = f.c['dispatch']['source_release']
        request = f.c['dispatch']['key']
        self.assertEqual(module.tick(f, 100)['action'], 'recovery cooldown')
        self.assertEqual(module.tick(f, 160)['action'], 'redispatched recoverable failure')
        self.assertEqual(f.posts[-1][1]['checkpoint_release'], source)
        self.assertEqual(f.c['dispatch']['key'], request)
        self.assertEqual(f.c['dispatch']['receipt_after_run_id'], 42)
        return f

    def test_fresh_lease_excludes_old_receipt_even_with_identical_key(self):
        f = self.rollover()
        d = f.c['dispatch']
        old = dict(id=42, head_sha=SHA, display_title=d['key'], status='completed')
        f.run_values = [old]
        self.assertEqual(module.tick(f, 161)['action'], 'idle')
        self.assertIsNone(f.c['dispatch']['run_id'])
        f.run_values.append(dict(old, id=43))
        self.assertEqual(module.tick(f, 162)['action'], 'recovered dispatch receipt')
        self.assertEqual(f.c['dispatch']['run_id'], 43)

    def test_delayed_post_receipt_keeps_watermark_on_resubmission(self):
        f = self.rollover()
        d = f.c['dispatch']
        f.run_values = [dict(id=42, head_sha=SHA, display_title=d['key'], status='completed')]
        self.assertEqual(module.tick(f, 760)['action'], 'retried uncertain dispatch')
        self.assertEqual(f.c['dispatch']['receipt_after_run_id'], 42)
        self.assertEqual(module.tick(f, 761)['action'], 'idle')
        self.assertIsNone(f.c['dispatch']['run_id'])
        self.assertIsNone(f.c['halted_reason'])

    def test_delayed_worker_and_new_worker_have_only_one_writer(self):
        f = self.rollover()
        d = f.c['dispatch']
        args = dict(runner_sha=SHA, source=d['source_release'], job=d['job_index'], request=d['key'])
        # The frozen native claim permits an older worker with the SAME
        # immutable inputs; CAS makes it the sole winner, not a second writer.
        self.assertTrue(module.claim(f, run_id=42, **args))
        self.assertFalse(module.claim(f, run_id=43, **args))
        self.assertFalse(module.claim(f, run_id=44, **args))
        self.assertEqual(f.c['dispatch']['run_id'], 42)

    def test_http2_failure_retries_but_old_integrity_error_is_not_hidden_by_tail(self):
        log = '\x1b[31mstream error: stream ID 5; PROTOCOL_ERROR; received from peer\x1b[0m'
        self.assertTrue(module.transient(log))
        unsafe = 'ValueError: conflicting registration proof\n' + 'cleanup\n' * 250 + log
        self.assertFalse(module.transient(unsafe))
        f = self.broken_run(attempt=1)
        f.log = unsafe
        self.assertEqual(module.tick(f, 100)['action'], 'halted')
        self.assertFalse(f.retries)
        self.assertFalse(f.posts)

    def test_empty_response_to_exclusive_claim_is_retried_without_advancing_checkpoint(self):
        f = self.broken_run(attempt=2)
        f.log = ("2026-11-25T15:51:15Z unexpected end of JSON input\n"
                 "subprocess.CalledProcessError: Command '['gh', 'api', '--method', 'PUT', "
                 "'repos/owner/repo/contents/control.json', '--input', '-']' returned non-zero exit status 1.")
        same_dispatch = f.c['dispatch']['source_release']
        self.assertEqual(module.tick(f, 100)['action'], 'recovery cooldown')
        self.assertEqual(module.tick(f, 160)['action'], 'retry transient failure')
        self.assertEqual(f.c['dispatch']['source_release'], same_dispatch)
        self.assertFalse(f.posts)
        self.assertFalse(module.transient('unexpected end of JSON input'))
        self.assertFalse(module.transient(f.log.replace('/contents/control.json', '/contents/other.json')))
        self.assertFalse(module.transient('ValueError: checkpoint digest mismatch\n' + f.log))

    def test_every_client_network_operation_has_finite_timeout(self):
        client = module.Client('owner/repo')
        with patch.object(module.subprocess, 'check_output', return_value='{}') as output, \
                patch.object(module.subprocess, 'run') as run:
            client.get('example')
            client.write({}, 'sha')
            client.dispatch('main', {})
            client.renew_supervisor()
            client.rerun(dict(id=42))
            client.asset(dict(id=17))
            paths = []
            def latest_jobs(path):
                paths.append(path)
                return {'jobs': [dict(id=17, conclusion='failure')]}
            client.get = latest_jobs
            client.logs(dict(id=42))
        self.assertIn('filter=latest', paths[0])
        self.assertEqual(len(output.call_args_list), 3)
        self.assertEqual(len(run.call_args_list), 4)
        for call in output.call_args_list + run.call_args_list:
            self.assertGreater(call.kwargs['timeout'], 0)
            self.assertLessEqual(call.kwargs['timeout'], 120)

    def test_temporary_api_and_log_errors_survive_more_than_three_failures(self):
        for mode in ('HTTP 503', 'HTTP 404', 'timeout'):
            with self.subTest(mode=mode):
                f = self.broken_run(attempt=1)
                now = [0.]
                waits = []
                renewals = []
                calls = []
                def logs(run):
                    calls.append(run['id'])
                    cmd = ['gh', 'api', 'repos/owner/repo/actions/jobs/17/logs']
                    if mode == 'timeout':
                        raise subprocess.TimeoutExpired(cmd, 60)
                    raise subprocess.CalledProcessError(1, cmd, stderr=mode)
                def pause(seconds):
                    waits.append(seconds)
                    now[0] += seconds
                f.logs = logs
                f.renew_supervisor = lambda: renewals.append(True)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = module.watch(f, seconds=1500, poll_seconds=30,
                                          clock=lambda: now[0], pause=pause)
                self.assertGreater(len(calls), 3)
                self.assertLessEqual(max(waits), 300)
                self.assertTrue(all(value > 0 for value in waits))
                self.assertIsNone(f.c['halted_reason'])
                self.assertEqual(result['action'], 'next supervisor explicitly dispatched')
                self.assertEqual(renewals, [True])

    def test_disabled_service_does_not_resurrect_cancelled_job(self):
        f = self.broken_run(conclusion='cancelled')
        f.c['enabled'] = False
        self.assertEqual(module.tick(f, 100)['action'], 'idle')
        self.assertFalse(f.retries)
        self.assertFalse(f.posts)
        f.c['enabled'] = True
        f.log = 'RuntimeError: checkpoint integrity failure'
        self.assertEqual(module.tick(f, 101)['action'], 'halted')
        self.assertFalse(f.retries)
        self.assertFalse(f.posts)

    def test_exact_receipt_resolves_tag_without_release_catalog(self):
        client = module.Client('owner/repo')
        tag = 'airi-offline-450620-42-2-1'
        queries = []
        def get(path):
            queries.append(path)
            if path == 'actions/runs/42':
                return dict(id=42, run_attempt=2)
            if path == 'releases/tags/' + tag:
                return dict(tag_name=tag, draft=False, assets=[
                    dict(id=17, name='AIRI_TRAINING_CHECKPOINT_450620.json'),
                    dict(id=18, name='AIRI_RUN_STATE.json')])
            self.fail('unexpected discovery request: ' + path)
        client.get = get
        client.logs = lambda run: 'SAVED_RELEASE: ' + tag
        client.asset = lambda asset: b'{}'
        self.assertEqual(client.checkpoint(42), (tag, {}, b'{}'))
        self.assertEqual(queries, ['actions/runs/42', 'releases/tags/' + tag])
        client.logs = lambda run: 'SAVED_RELEASE: airi-offline-450620-42-1-1'
        with self.assertRaisesRegex(ValueError, 'attempt mismatch'):
            client.checkpoint(42)


if __name__ == '__main__':
    unittest.main()
