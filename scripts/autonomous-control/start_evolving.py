"""Start exactly one verified native trial when the previous training is idle."""
import json,subprocess
from watchdog import Client,ACTIVE,TRAINING_PATHS,workflow_path
BRANCH='evolving-context-v1'
COMMIT='6c6c72dea5b7512cb45ec174ee69510674b7bb1f'
WORKFLOW='airi-evolving-background.yml'
SEED='airi-offline-35635-37785750755-1-3'

def start(client,dispatch):
    if client.get('git/ref/heads/'+BRANCH)['object']['sha']!=COMMIT:return {'action':'none','reason':'frozen ref changed'}
    runs=list(client.pages('actions/workflows/'+WORKFLOW+'/runs?branch='+BRANCH,'workflow_runs'))
    runs=[r for r in runs if r.get('head_sha')==COMMIT]
    # A run number alone cannot distinguish preflight; inspect its non-secret jobs.
    trained=[r for r in runs if r.get('display_title')=='AIRI context training'];verified=False
    for run in runs:
        jobs=list(client.pages(f"actions/runs/{run['id']}/jobs",'jobs'))
        steps=[s for j in jobs for s in j.get('steps',[])]
        if run.get('conclusion')=='success' and any(s['name']=='Verify real native optimizer recovery' and s.get('conclusion')=='success' for s in steps):verified=True
    if trained:return {'action':'none','reason':'finite adaptive trial already started; watchdog handles transient failures'}
    if not verified:return {'action':'none','reason':'waiting for native runtime preflight success'}
    for status in sorted(ACTIVE):
        if any(workflow_path(r) in TRAINING_PATHS for r in client.pages('actions/runs?status='+status,'workflow_runs')):
            return {'action':'none','reason':'previous training still active; keep waiting'}
    dispatch()
    return {'action':'started','reason':'runtime verified and previous training idle','seed':SEED}

def main():
    import os
    c=Client(os.environ['GITHUB_REPOSITORY'])
    def dispatch():
        payload={'ref':BRANCH,'inputs':{'checkpoint_release':SEED,'continuation':'0','preflight_only':'false'}}
        subprocess.run(['gh','api','--method','POST',c.base+'/actions/workflows/'+WORKFLOW+'/dispatches','--input','-'],input=json.dumps(payload),text=True,check=True)
    print(json.dumps(start(c,dispatch)))
if __name__=='__main__':main()
