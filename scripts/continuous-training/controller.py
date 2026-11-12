"""GitHub-only indefinite service supervisor. No model API or assistant calls.

A serialized controller and a CAS lease at training-job start jointly prevent
concurrent writers. Published checkpoint indices, never mutable progress, grant
continuation. Integrity/program failures halt; network failures have bounded
retries. An enabled service has no date or total-job limit.
"""
import argparse
import base64
from copy import deepcopy
import hashlib
import json
import os
import re
import subprocess
import time

CONTROL_BRANCH='airi-continuous-control'
CONTROL_PATH='control.json'
WORKFLOW='airi-continuous-background.yml'
ACTIVE={'queued','in_progress','waiting','pending','requested'}
TAG=re.compile(r'airi-offline-(\d+)-(\d+)-(\d+)-(\d+)\Z')


def key(fp,release,index):
    return hashlib.sha256(json.dumps([fp,release,index],separators=(',',':')).encode()).hexdigest()


def transient(log):
    value='\n'.join(clean_log(log).splitlines()[-200:]).lower()
    if any(x in value for x in ('digest mismatch','fingerprint','provenance','unsafe archive',
                                'conflicting registration','valueerror:','assertionerror:','runtimeerror:',
                                'retention memory','manifest omits')):
        return False
    return bool(re.search(r'(?:http|status|gh:)[^\n]{0,100}?(429|500|502|503|504)\b',value)
                or any(x in value for x in ('connection reset','connection timed out',
                       'temporary failure in name resolution','runner has lost communication','tls handshake timeout')))


def clean_log(value):
    """Captured diagnostic text only; never execute or render terminal controls."""
    value=re.sub(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)', '', value)
    value=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value)
    return ''.join(char for char in value if char in '\n\r\t' or ord(char)>=32)


def io_error_kind(error):
    value=error.stderr or ''
    if isinstance(value,bytes):value=value.decode('utf-8',errors='replace')
    value=value.lower()
    if 'escape sequences' in value or 'unknown flag' in value:return 'CLI compatibility'
    if re.search(r'http\s*(401|403)\b',value):return 'API authorization'
    if re.search(r'http\s*404\b',value):return 'API resource unavailable'
    if re.search(r'http\s*(429|500|502|503|504)\b',value):return 'transient HTTP'
    return 'subprocess failure'


def io_operation(error):
    """Expose a safe operation label, not commands, credentials or raw logs."""
    args=error.cmd if isinstance(error.cmd,(list,tuple)) else []
    path=next((arg for arg in args if isinstance(arg,str) and arg.startswith('repos/')),'')
    if path.endswith('/logs'):return 'read job logs'
    if path.endswith('/rerun-failed-jobs'):return 'request transient rerun'
    if '/contents/' in path:return 'read/write control state'
    if path.endswith('/dispatches'):return 'dispatch workflow'
    return 'GitHub API request'


class Client:
    def __init__(self,repo):
        if not re.fullmatch(r'[\w.-]+/[\w.-]+',repo):raise ValueError('invalid repository')
        self.base='repos/'+repo
    def get(self,path):
        return json.loads(subprocess.check_output(['gh','api',self.base+'/'+path],text=True,stderr=subprocess.PIPE))
    def control(self):
        result=self.get('contents/'+CONTROL_PATH+'?ref='+CONTROL_BRANCH)
        return json.loads(base64.b64decode(result['content'])),result['sha']
    def write(self,state,sha):
        payload=dict(message='chore: coordinate continuous AIRI training',branch=CONTROL_BRANCH,
                     content=base64.b64encode((json.dumps(state,sort_keys=True,indent=2)+'\n').encode()).decode(),sha=sha)
        subprocess.run(['gh','api','--method','PUT',self.base+'/contents/'+CONTROL_PATH,'--input','-'],
                       input=json.dumps(payload),text=True,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    def dispatch(self,branch,inputs):
        subprocess.run(['gh','api','--method','POST',self.base+'/actions/workflows/'+WORKFLOW+'/dispatches',
                        '--input','-'],input=json.dumps(dict(ref=branch,inputs=inputs)),text=True,check=True,stderr=subprocess.PIPE)
    def renew_supervisor(self):
        subprocess.run(['gh','api','--method','POST',self.base+'/actions/workflows/airi-continuous-controller.yml/dispatches',
                        '--input','-'],input=json.dumps(dict(ref='main',inputs={})),text=True,check=True,stderr=subprocess.PIPE)
    def runs(self):
        return self.get('actions/workflows/'+WORKFLOW+'/runs?per_page=100')['workflow_runs']
    def logs(self,run):
        jobs=self.get(f'actions/runs/{run["id"]}/jobs?per_page=100')['jobs']
        # Recent gh versions refuse ANSI-bearing response bodies by default,
        # even with captured stdout. Permit capture, then sanitize locally.
        return '\n'.join(clean_log(subprocess.check_output(
                         ['gh','api','--allow-escape-sequences',self.base+f'/actions/jobs/{j["id"]}/logs'],
                         text=True,errors='replace',stderr=subprocess.PIPE))
                         for j in jobs if j.get('conclusion')=='failure')
    def rerun(self,run):
        subprocess.run(['gh','api','--method','POST',self.base+f'/actions/runs/{run["id"]}/rerun-failed-jobs'],check=True,stderr=subprocess.PIPE)
    def asset(self,item):
        return subprocess.check_output(['gh','api','-H','Accept: application/octet-stream',
                                       self.base+f'/releases/assets/{item["id"]}'],stderr=subprocess.PIPE)
    def checkpoint(self,run_id):
        choices=[]
        for page in range(1,4):
            values=self.get(f'releases?per_page=100&page={page}')
            for release in values:
                match=TAG.fullmatch(release['tag_name'])
                if match and int(match[2])==run_id and not release['draft']:
                    choices.append((int(match[3]),int(match[4]),release))
            if choices or len(values)<100:break
        if not choices:raise ValueError('successful training has no committed checkpoint')
        release=max(choices,key=lambda r:r[:2])[2]
        assets=release['assets'];indices=[a for a in assets if re.fullmatch(r'AIRI_TRAINING_CHECKPOINT_\d+\.json',a['name'])]
        states=[a for a in assets if a['name']=='AIRI_RUN_STATE.json']
        if len(indices)!=1 or len(states)!=1:raise ValueError('checkpoint index/run state missing')
        return release['tag_name'],json.loads(self.asset(indices[0])),self.asset(states[0])


def validate_checkpoint(state,index,raw,control,job):
    expected=index.get('run_state',{})
    if (expected.get('name')!='AIRI_RUN_STATE.json' or expected.get('bytes')!=len(raw)
            or expected.get('sha256')!=hashlib.sha256(raw).hexdigest()):
        raise ValueError('continuation run-state digest mismatch')
    if (index.get('fingerprint')!=control['fingerprint'] or state.get('fingerprint')!=control['fingerprint']
            or state.get('runner_sha')!=control['runner_sha'] or state.get('job_index')!=job
            or state.get('completed_steps')!=index.get('completed_steps')
            or type(state.get('completed_steps')) is not int or state['completed_steps']<control['origin_step']
            or state.get('continue') is not True or state.get('production_qualified') is not False
            or state.get('live_promoted') is not False):
        raise ValueError('continuation identity, budget or offline-state mismatch')
    arrays=index.get('archives',[])
    names={f'AIRI_TRAINING_{state["completed_steps"]:05d}_{part}.zip' for part in ('MODEL','OPTIMIZER')}
    if (len(arrays)!=2 or {a.get('name') for a in arrays}!=names
            or not all(re.fullmatch(r'[0-9a-f]{64}',a.get('sha256','')) and type(a.get('bytes')) is int and a['bytes']>0 for a in arrays)):
        raise ValueError('checkpoint model/optimizer archive descriptors missing')
    epoch=state.get('continuous_training',{})
    when=epoch.get('not_before')
    if isinstance(when,bool) or not isinstance(when,(int,float)) or not 0<=when<float('inf'):
        raise ValueError('invalid durable continuous cooldown')


def submit(client,c,sha,source,job,now):
    request=key(c['fingerprint'],source,job)
    c=deepcopy(c)
    c['dispatch']=dict(key=request,source_release=source,job_index=job,run_id=None,
                       requested_at=now,request_attempts=1)
    client.write(c,sha) # Durable claim BEFORE the fallible remote POST.
    client.dispatch(c['runner_branch'],dict(checkpoint_release=source,continuation=str(job),
                                          preflight_only='false',dispatch_key=request,chunk_seconds=str(c.get('chunk_seconds',1200))))
    return dict(action='dispatched',job_index=job,source_release=source)


def tick(client,now=None):
    now=time.time() if now is None else now
    c,sha=client.control()
    if c.get('schema')!=1:raise ValueError('invalid continuous control schema')
    if not c.get('enabled') or c.get('halted_reason'):return dict(action='idle',reason=c.get('halted_reason') or 'disabled')
    if client.get('git/ref/heads/'+c['runner_branch'])['object']['sha']!=c['runner_sha']:
        c['halted_reason']='frozen runner branch changed';client.write(c,sha)
        return dict(action='halted',reason=c['halted_reason'])
    runs=client.runs()
    active=[r for r in runs if r.get('status') in ACTIVE and r.get('head_branch')==c['runner_branch']]
    if active:return dict(action='idle',reason='training/preflight active')
    d=c.get('dispatch')
    if d is None:return submit(client,c,sha,c['seed_release'],0,now)
    if d['run_id'] is None:
        found=[r for r in runs if d['key'] in r.get('display_title','') and r.get('head_sha')==c['runner_sha']]
        if found:
            d['run_id']=max(found,key=lambda r:r['id'])['id'];client.write(c,sha)
            return dict(action='recovered dispatch receipt',run_id=d['run_id'])
        if now-d['requested_at']<600:return dict(action='idle',reason='dispatch receipt pending')
        if d['request_attempts']>=3:
            c['halted_reason']='dispatch receipt unavailable after bounded retries';client.write(c,sha)
            return dict(action='halted',reason=c['halted_reason'])
        d['request_attempts']+=1;d['requested_at']=now;client.write(c,sha)
        client.dispatch(c['runner_branch'],dict(checkpoint_release=d['source_release'],continuation=str(d['job_index']),preflight_only='false',dispatch_key=d['key'],chunk_seconds=str(c.get('chunk_seconds',1200))))
        return dict(action='retried uncertain dispatch')
    run=client.get(f'actions/runs/{d["run_id"]}')
    if run.get('status')!='completed':return dict(action='idle',reason='claimed training active')
    if run.get('head_sha')!=c['runner_sha']:
        c['halted_reason']='claimed run identity changed';client.write(c,sha)
        return dict(action='halted',reason=c['halted_reason'])
    if run.get('conclusion')!='success':
        if run.get('conclusion') in {'failure','timed_out'} and transient(client.logs(run)) and run.get('run_attempt',1)<3:
            attempt=run.get('run_attempt',1)
            if d.get('last_retry_attempt')==attempt:
                if now-d.get('retry_requested_at',0)<600:
                    return dict(action='idle',reason='transient retry already claimed')
                if d.get('retry_requests',1)>=3:
                    c['halted_reason']='transient rerun receipt unavailable after bounded requests'
                    client.write(c,sha);return dict(action='halted',reason=c['halted_reason'])
            else:d['retry_requests']=0
            d.update(last_retry_attempt=attempt,retry_requested_at=now,
                     retry_requests=d.get('retry_requests',0)+1)
            client.write(c,sha)
            client.rerun(run);return dict(action='retry transient failure',run_id=run['id'])
        c['halted_reason']='training failed or intentionally cancelled; inspect integrity/program error'
        client.write(c,sha);return dict(action='halted',reason=c['halted_reason'])
    try:
        release,index,raw=client.checkpoint(run['id']);state=json.loads(raw)
        validate_checkpoint(state,index,raw,c,d['job_index'])
    except (ValueError,KeyError,TypeError) as error:
        c['halted_reason']=str(error);client.write(c,sha);return dict(action='halted',reason=str(error))
    if state['continuous_training']['not_before']>now:return dict(action='idle',reason='durable epoch cooldown',not_before=state['continuous_training']['not_before'])
    return submit(client,c,sha,release,d['job_index']+1,now)


def claim(client,*,run_id,runner_sha,source,job,request):
    c,sha=client.control();d=c.get('dispatch') or {}
    valid=(c.get('enabled') is True and not c.get('halted_reason') and c['runner_sha']==runner_sha
           and d.get('key')==request==key(c['fingerprint'],source,job)
           and d.get('source_release')==source and d.get('job_index')==job
           and d.get('run_id') in (None,run_id))
    if not valid:return False
    d['run_id']=run_id;client.write(c,sha);return True


def watch(client,seconds=2700,poll_seconds=30,clock=time.time,pause=time.sleep):
    """Observe child completion directly and explicitly renew this bounded watcher."""
    end=clock()+seconds
    failures=0
    while clock()<end:
        try:
            control,_=client.control()
            if not control.get('enabled') or control.get('halted_reason'):
                return dict(action='watch stopped',reason=control.get('halted_reason') or 'disabled')
            result=tick(client,now=clock())
            failures=0
            print(json.dumps(result),flush=True)
            if result.get('action')=='halted':return result
        except subprocess.CalledProcessError as error:
            failures+=1
            diagnostic=dict(action='controller API retry pending',operation=io_operation(error),
                            exit_code=error.returncode,error_kind=io_error_kind(error),consecutive_failures=failures)
            print(json.dumps(diagnostic),flush=True)
            if failures>=3:
                reason='repeated controller I/O failure: '+diagnostic['operation']+' ('+diagnostic['error_kind']+')'
                try:
                    control,sha=client.control();control['halted_reason']=reason
                    control['controller_error']=diagnostic;client.write(control,sha)
                except subprocess.CalledProcessError:
                    pass # Cannot persist a halt if the control API itself is unavailable.
                return dict(action='controller failure',reason=reason)
        pause(min(poll_seconds,max(0,end-clock())))
    control,_=client.control()
    if control.get('enabled') and not control.get('halted_reason'):
        client.renew_supervisor()
        return dict(action='next supervisor explicitly dispatched')
    return dict(action='watch stopped')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['tick','claim','watch']);args=parser.parse_args()
    client=Client(os.environ['GITHUB_REPOSITORY'])
    if args.mode=='watch':
        result=watch(client);print(json.dumps(result))
        if result.get('action')=='controller failure':raise SystemExit(1)
    elif args.mode=='tick':print(json.dumps(tick(client)))
    else:
        allowed=claim(client,run_id=int(os.environ['GITHUB_RUN_ID']),runner_sha=os.environ['GITHUB_SHA'],
                      source=os.environ['CHECKPOINT_RELEASE'],job=int(os.environ['CONTINUATION']),request=os.environ['DISPATCH_KEY'])
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('authorized='+str(allowed).lower()+'\n')
        print(json.dumps(dict(authorized=allowed)))

if __name__=='__main__':main()
