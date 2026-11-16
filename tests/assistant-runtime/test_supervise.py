from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assistant-runtime'))
import supervise


def run_at(when,**extra):
    return {'id':1,'created_at':datetime.fromtimestamp(when,timezone.utc).isoformat(),
            'head_branch':'main','status':'completed','conclusion':'success',**extra}


class Fake:
    def __init__(self,runs):self.items=runs;self.calls=[];self.active=True;self.bad=False
    def enabled(self,_):return self.active
    def runs(self):return self.items
    def quality_failure(self,_):return self.bad
    def dispatch(self,workflow):self.calls.append(workflow)


class SupervisorTests(unittest.TestCase):
    def test_missing_cron_is_recovered_and_active_job_is_not_duplicated(self):
        c=Fake([run_at(100)])
        self.assertEqual(supervise.tick(c,3800)['action'],'update dispatched')
        c.items.append(run_at(3800,status='queued',conclusion=None,id=2))
        self.assertEqual(supervise.tick(c,3900)['reason'],'update already active')
        self.assertEqual(c.calls,[supervise.UPDATE])

    def test_due_uses_last_successful_run_even_after_catalog_exhaustion(self):
        c=Fake([run_at(100),run_at(3700,id=2)])
        self.assertEqual(supervise.tick(c,6000)['reason'],'hourly interval')
        self.assertFalse(c.calls)

    def test_failed_request_backoff_and_quality_failure_do_not_trigger_retry_storm(self):
        c=Fake([run_at(100),run_at(4000,id=2,conclusion='failure')])
        self.assertEqual(supervise.tick(c,4100)['reason'],'failed/uncertain request cooldown')
        c.bad=True
        self.assertEqual(supervise.tick(c,6000)['action'],'blocked')
        self.assertFalse(c.calls)

    def test_disabled_workflows_stop_renewal(self):
        c=Fake([]);c.active=False
        self.assertEqual(supervise.watch(c,seconds=60,clock=lambda:100)['action'],'stopped')
        self.assertFalse(c.calls)

    def test_bounded_watch_renews_without_waiting_for_cron(self):
        c=Fake([run_at(100)]);now=[4000]
        def pause(n):now[0]+=n
        result=supervise.watch(c,seconds=60,poll=10,clock=lambda:now[0],pause=pause)
        self.assertEqual(c.calls,[supervise.UPDATE,supervise.SUPERVISOR])
        self.assertEqual(result['action'],'supervisor explicitly renewed')

    def test_uncertain_post_is_not_repeated_each_poll_but_watcher_renews(self):
        c=Fake([]);now=[4000]
        def dispatch(workflow):
            c.calls.append(workflow)
            if workflow==supervise.UPDATE:raise subprocess.TimeoutExpired('gh',30)
        c.dispatch=dispatch
        result=supervise.watch(c,seconds=60,poll=10,clock=lambda:now[0],pause=lambda n:now.__setitem__(0,now[0]+n))
        self.assertEqual(c.calls,[supervise.UPDATE,supervise.SUPERVISOR])
        self.assertEqual(result['action'],'supervisor explicitly renewed')

    def test_renewal_transient_error_is_retried(self):
        c=Fake([run_at(100)]);now=[4000];fail=[True]
        def dispatch(workflow):
            c.calls.append(workflow)
            if workflow==supervise.SUPERVISOR and fail[0]:
                fail[0]=False;raise subprocess.TimeoutExpired('gh',30)
        c.dispatch=dispatch
        result=supervise.watch(c,seconds=60,poll=60,clock=lambda:now[0],pause=lambda n:now.__setitem__(0,now[0]+n))
        self.assertEqual(c.calls,[supervise.UPDATE,supervise.SUPERVISOR,supervise.SUPERVISOR])
        self.assertEqual(result['action'],'supervisor explicitly renewed')


if __name__=='__main__':unittest.main()
