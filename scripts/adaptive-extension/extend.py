"""Launch one adaptive champion-search cycle after the current frozen chain ends."""
import argparse
import json
import math
import re
import subprocess

REPOSITORY='arancione3000/airi-offline-training'
SOURCE_BRANCH='dialogue-cycle-20261012-evening'
SOURCE_SHA='e90640138a3d3734f8e08a6b60f91e680a3737af'
SOURCE_FP='4db0b3b1211018b5d0c84dd0a63b982296312d43f9294855e12727b4a6d92469'
SOURCE_SECONDS=14400
TARGET_BRANCH='dialogue-adaptive-cycle-20261012'
TARGET_SHA='191549ab6ee83feb66041fa0f59e57ba5f814551'
TARGET_FP='b564dbc0445b0d73233220c6915c032c34e403d087776b13324cbabc455edcbb'
SEED_RELEASE='airi-offline-129599-38137184439-1-1'
SOURCE_WORKFLOW='airi-transfer-background.yml'
TARGET_WORKFLOW='airi-adaptive-background.yml'
CONTROLLER='airi-adaptive-extension.yml'
ACTIVE={'in_progress','queued','requested','waiting','pending'}
TRAINING_PATH='.github/workflows/'+TARGET_WORKFLOW

class Client:
    def __init__(self,repository):
        if repository!=REPOSITORY:raise ValueError('unexpected repository')
        self.base='repos/'+repository+'/'
    def api(self,path,payload=None,method='POST'):
        args=['gh','api',self.base+path]
        if payload is not None:args+=['--method',method,'--input','-']
        result=subprocess.run(args,input=None if payload is None else json.dumps(payload),capture_output=True,text=True,check=True,timeout=60)
        return json.loads(result.stdout) if result.stdout.strip() else None
    def pages(self,path,key=None):
        page=1
        while True:
            sep='&' if '?' in path else '?';value=self.api(path+sep+f'per_page=100&page={page}')
            rows=value[key] if key else value;yield from rows
            if len(rows)<100:break
            page+=1
    def asset(self,item):
        raw=subprocess.check_output(['gh','api',self.base+'releases/assets/'+str(item['id']),'-H','Accept: application/octet-stream'],timeout=60)
        if len(raw)!=item['size']:raise ValueError('asset size mismatch')
        return json.loads(raw)

def workflow_path(run):return run.get('path','').split('@',1)[0]

def terminal_source(client):
    runs=list(client.pages(f'actions/workflows/{SOURCE_WORKFLOW}/runs?branch={SOURCE_BRANCH}','workflow_runs'))
    if not runs:return {'action':'waiting_for_source_run'}
    if any(r['status'] in ACTIVE for r in runs):return {'action':'waiting_for_current_cycle'}
    latest=max(runs,key=lambda r:r['id'])
    if latest.get('head_sha')!=SOURCE_SHA or latest.get('head_branch')!=SOURCE_BRANCH:
        raise ValueError('source branch or commit changed')
    if latest.get('status')!='completed' or latest.get('conclusion')!='success':
        return {'action':'source_stopped_without_success'}
    pattern=re.compile(rf'airi-offline-[0-9]+-{latest["id"]}-{latest.get("run_attempt",1)}-[0-9]+')
    for release in client.pages('releases'):
        assets={a['name']:a for a in release.get('assets',[])}
        final=assets.get('FINAL_RESULT.json')
        indexes=[a for n,a in assets.items() if re.fullmatch(r'AIRI_TRAINING_CHECKPOINT_[0-9]+\.json',n)]
        if (release.get('target_commitish')!=SOURCE_SHA or release.get('draft') or
                release.get('prerelease') is not True or not pattern.fullmatch(release.get('tag_name','')) or
                final is None or len(indexes)!=1):continue
        result=client.asset(final);index=client.asset(indexes[0]);c=result['counters'];elapsed=c['elapsed_training_seconds']
        if (not isinstance(elapsed,(int,float)) or isinstance(elapsed,bool) or not math.isfinite(elapsed) or
                elapsed<SOURCE_SECONDS or result.get('next_action') is not None or
                result.get('maximum_training_seconds')!=SOURCE_SECONDS or result.get('production_qualified') is not False or
                result.get('live_promoted') is not False or index.get('fingerprint')!=SOURCE_FP or
                index.get('completed_steps')!=c.get('completed_steps') or len(index.get('archives',[]))!=2):
            raise ValueError('source terminal result failed provenance checks')
        return {'action':'terminal_verified','run_id':latest['id'],'release':release['tag_name']}
    return {'action':'waiting_for_terminal_result'}

def tick(client):
    target=client.api('git/ref/heads/'+TARGET_BRANCH)
    if target['object']['sha']!=TARGET_SHA:raise ValueError('adaptive branch changed')
    prior=list(client.pages(f'actions/workflows/{TARGET_WORKFLOW}/runs?branch={TARGET_BRANCH}','workflow_runs'))
    if prior:return {'action':'already_started','run_id':max(prior,key=lambda r:r['id'])['id']}
    source=terminal_source(client)
    if source['action']!='terminal_verified':return source
    for status in sorted(ACTIVE):
        if any(workflow_path(r)==TRAINING_PATH for r in client.pages(f'actions/runs?status={status}','workflow_runs')):
            return {'action':'waiting_for_training_slot'}
    client.api(f'actions/workflows/{TARGET_WORKFLOW}/dispatches',{'ref':TARGET_BRANCH,'inputs':{'checkpoint_release':SEED_RELEASE,'continuation':'0'}})
    return {'action':'dispatched','source_run':source['run_id'],'seed_release':SEED_RELEASE,'target_fingerprint':TARGET_FP}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repository',default=REPOSITORY);args=parser.parse_args()
    print(json.dumps(tick(Client(args.repository))))

if __name__=='__main__':main()
