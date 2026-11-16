"""Renewable Actions supervisor: reconcile due updates without relying on cron.

One serialized watcher polls remotely visible runs, dispatches only when due,
and explicitly hands off to a new bounded watcher. Cron/native completion are
recovery triggers, not an hourly execution guarantee. No model API or weights.
"""
import argparse
from datetime import datetime
import json
import os
import re
import subprocess
import time

UPDATE='airi-modular-auto-update.yml'
SUPERVISOR='airi-modular-supervisor.yml'
ACTIVE={'queued','in_progress','waiting','pending','requested'}
IO_ERRORS=(subprocess.CalledProcessError,subprocess.TimeoutExpired)


class Client:
    def __init__(self,repo):
        if not re.fullmatch(r'[\w.-]+/[\w.-]+',repo):raise ValueError('invalid repository')
        self.base='repos/'+repo

    def get(self,path):
        return json.loads(subprocess.check_output(['gh','api',self.base+'/'+path],
                          text=True,stderr=subprocess.PIPE,timeout=30))

    def enabled(self,workflow):
        return self.get('actions/workflows/'+workflow)['state']=='active'

    def runs(self):
        return self.get('actions/workflows/'+UPDATE+'/runs?branch=main&per_page=30')['workflow_runs']

    def dispatch(self,workflow):
        subprocess.run(['gh','api','--method','POST',self.base+'/actions/workflows/'+workflow+'/dispatches','--input','-'],
                       input=json.dumps({'ref':'main'}),text=True,stderr=subprocess.PIPE,
                       stdout=subprocess.DEVNULL,check=True,timeout=30)

    def quality_failure(self,run):
        if run.get('conclusion')!='failure':return False
        jobs=self.get(f'actions/runs/{run["id"]}/jobs?filter=latest&per_page=20')['jobs']
        return any(step.get('conclusion')=='failure' and step.get('name','').startswith(('Learn ','Verify '))
                   for job in jobs for step in job.get('steps',[]))


def created(run):
    return datetime.fromisoformat(run['created_at'].replace('Z','+00:00')).timestamp()


def tick(client,now):
    if not client.enabled(UPDATE) or not client.enabled(SUPERVISOR):
        return {'action':'stopped','reason':'workflow disabled'}
    runs=[r for r in client.runs() if r.get('head_branch')=='main']
    if any(r.get('status') in ACTIVE for r in runs):return {'action':'idle','reason':'update already active'}
    latest=max(runs,key=created) if runs else None
    successes=[r for r in runs if r.get('conclusion')=='success']
    due=(max(created(r) for r in successes)+3600) if successes else 0
    if now<due:return {'action':'idle','reason':'hourly interval','next_due_utc_epoch':due}
    if latest and client.quality_failure(latest):
        return {'action':'blocked','reason':'learning or verification failure requires review','run_id':latest['id']}
    if latest and now-created(latest)<900:return {'action':'idle','reason':'failed/uncertain request cooldown'}
    client.dispatch(UPDATE)
    return {'action':'update dispatched','previous_success_run':max(successes,key=created)['id'] if successes else None}


def watch(client,*,seconds=2400,poll=300,clock=time.time,pause=time.sleep):
    end=clock()+seconds;last_request=0;last_error=None
    while clock()<end:
        try:
            # Also suppress uncertain POST retries until remote visibility catches up.
            if last_request and clock()-last_request<900:
                result={'action':'idle','reason':'local dispatch cooldown'}
            else:
                last_request=clock() # Set BEFORE a fallible network POST.
                result=tick(client,clock())
                if result['action']!='update dispatched':last_request=0
            last_error=None
            print(json.dumps(result),flush=True)
            if result['action']=='stopped':return result
        except IO_ERRORS as error:
            last_error=type(error).__name__
            print(json.dumps({'action':'API retry pending','error':last_error}),flush=True)
        pause(min(poll,max(0,end-clock())))
    for attempt in range(3):
        try:
            if not client.enabled(UPDATE) or not client.enabled(SUPERVISOR):
                return {'action':'stopped','reason':'workflow disabled before renewal'}
            client.dispatch(SUPERVISOR)
            return {'action':'supervisor explicitly renewed','last_io_error':last_error}
        except IO_ERRORS:
            if attempt==2:raise
            pause(2*(attempt+1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=int,default=2400)
    p.add_argument('--poll',type=int,default=300)
    a=p.parse_args()
    if not 60<=a.seconds<=2400 or not 10<=a.poll<=300:p.error('invalid bounded supervisor interval')
    print(json.dumps(watch(Client(os.environ['GITHUB_REPOSITORY']),seconds=a.seconds,poll=a.poll)))


if __name__=='__main__':main()
