"""Autonomous, local learning from an explicitly trusted labeled lesson queue.

Programs are selected using TRAIN only, qualified on distinct examples, checked
against old skills/stable contracts and atomically added. This never calls a model
API, executes supplied code or learns labels from its own answers.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from skill_bank import SkillBank, RegistryError

DEFAULT_PRIVATE=Path.home()/'.cache/airi/skill-bank.json'
BUNDLED=Path(__file__).resolve().parent/'skills/approved-bank.json'


def _refresh_default(bank):
    from managed_skills import ReconciliationConflict,reconcile
    try:
        return reconcile(bank,BUNDLED)
    except (RegistryError,OSError) as error:
        print(f'Bundled skill refresh unavailable ({type(error).__name__}); retaining stable response routing.',file=sys.stderr)
        return {'action':'rejected','reason':type(error).__name__,'retry':isinstance(error,(ReconciliationConflict,OSError))}


def selected_bank(path=None):
    if path is not None:return SkillBank(path)
    if DEFAULT_PRIVATE.exists() or DEFAULT_PRIVATE.is_symlink() or Path(str(DEFAULT_PRIVATE)+'.revisions').exists():
        bank=SkillBank(DEFAULT_PRIVATE)
        _refresh_default(bank)
        return bank
    return SkillBank(BUNDLED)


def learn_file(path,bank):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>512*1024:raise RegistryError('invalid bounded lesson file')
    raw=json.loads(path.read_text())
    lessons=raw.get('lessons') if isinstance(raw,dict) else None
    if lessons is None and isinstance(raw,dict):lessons=[raw.get('lesson',raw)]
    if not isinstance(lessons,list) or not 1<=len(lessons)<=32:raise RegistryError('invalid lesson queue')
    results=[]
    for lesson in lessons:
        try:result=bank.learn(lesson)
        except (ValueError,TypeError,KeyError,OSError) as error:
            result={'action':'rejected','reason':type(error).__name__}
        results.append({'lesson_id':lesson.get('lesson_id') if isinstance(lesson,dict) else None,**result})
    return results


def private_bank(path=None):
    target=Path(path) if path is not None else DEFAULT_PRIVATE
    bank=SkillBank(target)
    if path is None:_refresh_default(bank)
    return bank


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path)
    p.add_argument('--lessons',type=Path)
    p.add_argument('--watch',type=Path,help='Explicitly trusted local queue directory; no remote or self-generated labels')
    p.add_argument('--interval',type=float,default=30)
    p.add_argument('--cycles',type=int,default=0,help='0 runs the trusted-queue watcher until stopped')
    p.add_argument('--status',action='store_true')
    a=p.parse_args()
    if a.status:
        print(json.dumps(selected_bank(a.bank).status()));return
    if bool(a.lessons)==bool(a.watch):p.error('choose --lessons or --watch')
    if not 1<=a.interval<=3600 or not 0<=a.cycles<=100000:p.error('invalid watcher bound')
    bank=private_bank(a.bank)
    if a.lessons:
        print(json.dumps({'results':learn_file(a.lessons,bank),'registry':bank.status()},ensure_ascii=False));return
    if a.watch.is_symlink() or not a.watch.is_dir():p.error('trusted queue must be an existing directory')
    seen={};cycle=0;queue_health=None;refresh_stamp=None
    while True:
        if a.bank is None:
            try:
                stamp=(BUNDLED.stat().st_mtime_ns,BUNDLED.stat().st_size)
                if stamp!=refresh_stamp:
                    refreshed=_refresh_default(bank)
                    if not refreshed.get('retry'):refresh_stamp=stamp
                    if refreshed['action']=='activated':
                        print(json.dumps({'managed_refresh':refreshed,'registry':bank.status()}),flush=True)
            except OSError as error:
                if refresh_stamp!=('unavailable',):
                    print(json.dumps({'managed_refresh':'unavailable','reason':type(error).__name__}),flush=True)
                refresh_stamp=('unavailable',)
        paths=sorted(a.watch.glob('*.json'))
        present={str(path) for path in paths}
        seen={path:stamp for path,stamp in seen.items() if path in present}
        health='overflow' if len(paths)>32 else 'ok'
        if health!=queue_health and (health=='overflow' or queue_health=='overflow'):
            print(json.dumps({'queue_health':health,'file_limit':32,'action':'processing_first_32_unseen_files_per_cycle'}),flush=True)
        queue_health=health
        pending=[]
        for path in paths:
            try:
                info=path.lstat();stamp=(info.st_mtime_ns,info.st_size)
                if seen.get(str(path))==stamp:continue
                pending.append((path,stamp))
            except OSError as error:
                stamp=('unavailable',type(error).__name__)
                if seen.get(str(path))==stamp:continue
                seen[str(path)]=stamp
                print(json.dumps({'file':path.name,'action':'rejected','reason':type(error).__name__}),flush=True)
        for path,stamp in pending[:32]:
            seen[str(path)]=stamp
            try:
                if path.is_symlink():raise RegistryError('symlink lesson refused')
                result=learn_file(path,bank)
                print(json.dumps({'file':path.name,'results':result,'registry':bank.status()},ensure_ascii=False),flush=True)
            except (ValueError,TypeError,OSError) as error:
                print(json.dumps({'file':path.name,'action':'rejected','reason':type(error).__name__}),flush=True)
        cycle+=1
        if a.cycles and cycle>=a.cycles:return
        time.sleep(a.interval)


if __name__=='__main__':main()
