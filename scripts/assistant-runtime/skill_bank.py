"""Persistent, gated, append-only symbolic experts. No executable model/code files.

An immutable validated snapshot is the commit point. The main file is a convenience
copy, so an interrupted replacement or a corrupted main copy can recover from the
latest valid snapshot. Checksums detect corruption, not a malicious filesystem owner.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

import skill_programs
from context_memory import resolve

MAX_SKILLS = 32
MAX_BYTES = 2 * 1024 * 1024
MAX_CHECKS = 128
ENGINE_SHA256 = hashlib.sha256(Path(skill_programs.__file__).read_bytes()).hexdigest()


class RegistryError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _examples(rows, *, training=False):
    if not isinstance(rows, list) or not (4 if training else 12) <= len(rows) <= (64 if training else MAX_CHECKS):
        raise RegistryError('invalid bounded lesson size')
    result=[]
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('messages'),list):raise RegistryError('invalid lesson example')
        answer=row.get('answer')
        if answer is not None and (not isinstance(answer,str) or not 0<len(answer)<=512):raise RegistryError('invalid lesson answer')
        if training and answer is None:raise RegistryError('TRAIN examples require trusted answers')
        if len(canonical(row['messages']))>65536:raise RegistryError('example too large')
        result.append({'messages':deepcopy(row['messages']),'answer':answer})
    if len({digest(r['messages']) for r in result})!=len(result):raise RegistryError('duplicate lesson contexts')
    return result


def _check(program, cases):
    failures=[]
    for i,row in enumerate(cases):
        result=skill_programs.predict(program,row['messages'])
        answer=result['answer'] if result is not None else None
        if answer!=row['answer']:failures.append(i)
    return failures


def _validate_payload(value):
    if not isinstance(value,dict) or set(value)!={'schema','revision','engine_sha256','entries'}:raise RegistryError('invalid registry schema')
    if value['schema']!=1 or value['engine_sha256']!=ENGINE_SHA256:raise RegistryError('registry engine version mismatch')
    entries=value['entries']
    if not isinstance(entries,list) or not 0<=len(entries)<=MAX_SKILLS or value['revision']!=len(entries):raise RegistryError('invalid registry size/revision')
    seen=set()
    for entry in entries:
        if not isinstance(entry,dict) or set(entry)!={'id','lesson_id','program','train','qualification','certificate'}:raise RegistryError('invalid expert entry')
        try:skill_programs.validate_program(entry['program'])
        except (ValueError,TypeError,KeyError) as e:raise RegistryError('invalid expert program') from e
        if entry['id']!=digest(entry['program']) or entry['id'] in seen:raise RegistryError('expert identity conflict')
        seen.add(entry['id'])
        train=_examples(entry['train'],training=True);checks=_examples(entry['qualification'])
        if entry['certificate']!={'train_sha256':digest(train),'qualification_sha256':digest(checks),'all_passed':True}:raise RegistryError('expert certificate digest mismatch')
        if _check(entry['program'],train) or _check(entry['program'],checks):raise RegistryError('expert certificate no longer verifies')
    return value


def _empty():return {'schema':1,'revision':0,'engine_sha256':ENGINE_SHA256,'entries':[]}


def _route(entries,messages):
    memory=resolve(messages)
    if memory is not None:return memory
    for entry in entries:
        result=skill_programs.predict(entry['program'],messages)
        if result is not None:return result
    return None


class SkillBank:
    def __init__(self,path):
        self.path=Path(path)
        self.snapshots=self.path.parent/(self.path.name+'.revisions')

    def _read_file(self,path):
        fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|getattr(os,'O_NOFOLLOW',0))
        with os.fdopen(fd,'rb') as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise RegistryError('registry is not a regular file')
            raw=f.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES:raise RegistryError('registry exceeds byte limit')
        try:
            wrapper=json.loads(raw)
            if not isinstance(wrapper,dict) or set(wrapper)!={'payload','sha256'} or wrapper['sha256']!=digest(wrapper['payload']):raise RegistryError('registry checksum mismatch')
            return _validate_payload(wrapper['payload']),wrapper['sha256']
        except RecursionError as error:
            # A bounded byte count does not bound JSON nesting. Treat damaged
            # nested data like other corruption so snapshots/native can recover.
            raise RegistryError('registry nesting exceeds parser bound') from error

    def _read(self):
        candidates=[];errors=[];main_revision=None;main_exists=self.path.exists() or self.path.is_symlink()
        if main_exists:
            try:
                payload,sha=self._read_file(self.path);main_revision=payload['revision'];candidates.append((payload['revision'],sha,payload,False))
            except (OSError,ValueError,TypeError,KeyError) as e:errors.append(e)
        if self.snapshots.is_symlink():raise RegistryError('snapshot directory cannot be a symlink')
        if self.snapshots.exists():
            files=list(self.snapshots.glob('*.json'))
            # Revision zero may be seeded before the first learned expert.
            if len(files)>MAX_SKILLS+1:raise RegistryError('too many snapshots')
            revisions=[p.name[:6] for p in files]
            if len(revisions)!=len(set(revisions)):raise RegistryError('conflicting snapshot revisions')
            for path in sorted(files,reverse=True):
                if not re.fullmatch(r'\d{6}-[0-9a-f]{64}\.json',path.name):raise RegistryError('invalid snapshot name')
                try:
                    payload,sha=self._read_file(path)
                    if path.name!=f'{payload["revision"]:06d}-{sha}.json':raise RegistryError('snapshot identity mismatch')
                    candidates.append((payload['revision'],sha,payload,True))
                    break
                except (OSError,ValueError,TypeError,KeyError) as e:errors.append(e)
        if not candidates:
            if main_exists or errors:raise RegistryError('no valid registry snapshot')
            return _empty(),False
        candidates.sort(key=lambda x:x[0],reverse=True)
        revision=candidates[0][0]
        if len({x[1] for x in candidates if x[0]==revision})>1:raise RegistryError('conflicting registry revisions')
        chosen=candidates[0]
        return deepcopy(chosen[2]),bool(chosen[3] and (errors or main_revision!=chosen[0]))

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd=os.open(str(self.path)+'.lock',os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):raise RegistryError('invalid registry lock')
            fcntl.flock(fd,fcntl.LOCK_EX);yield
        finally:os.close(fd)

    def _atomic(self,path,raw):
        fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            os.replace(name,path)
            d=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(d)
            finally:os.close(d)
        finally:
            if os.path.exists(name):os.unlink(name)

    def _write(self,payload):
        _validate_payload(payload)
        raw=canonical({'payload':payload,'sha256':digest(payload)})+b'\n'
        if len(raw)>MAX_BYTES:raise RegistryError('registry byte capacity exhausted')
        self.snapshots.mkdir(mode=0o700,exist_ok=True)
        if self.snapshots.is_symlink():raise RegistryError('unsafe snapshot directory')
        snapshot=self.snapshots/f'{payload["revision"]:06d}-{digest(payload)}.json'
        if snapshot.exists():
            old,_=self._read_file(snapshot)
            if old!=payload:raise RegistryError('immutable snapshot conflict')
        else:self._atomic(snapshot,raw)
        try:self._atomic(self.path,raw)
        except OSError:return {'snapshot_committed':True,'main_copy_updated':False}
        return {'snapshot_committed':True,'main_copy_updated':True}

    def status(self):
        value,recovered=self._read()
        return {'revision':value['revision'],'active_skills':len(value['entries']),
                'state_sha256':digest(value),'recovered_from_snapshot':recovered,'capacity':MAX_SKILLS}

    def predict(self,messages):
        value,recovered=self._read()
        for entry in value['entries']:
            result=skill_programs.predict(entry['program'],messages)
            if result is not None:
                return {**result,'skill_id':entry['id'],'skill_revision':value['revision'],
                        'registry_recovered_from_snapshot':recovered}
        return None

    def learn(self,lesson,*,regression_gate=None,expected_revision=None):
        if not isinstance(lesson,dict) or lesson.get('trusted') is not True:raise RegistryError('explicit trusted labeled lesson required')
        label=lesson.get('lesson_id')
        if not isinstance(label,str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,80}',label):raise RegistryError('invalid lesson identity')
        train=_examples(lesson.get('examples'),training=True);checks=_examples(lesson.get('qualification'))
        if {digest(r['messages']) for r in train}&{digest(r['messages']) for r in checks}:raise RegistryError('TRAIN/qualification context overlap')
        if sum(r['answer'] is not None for r in checks)<8 or sum(r['answer'] is None for r in checks)<4:raise RegistryError('independent positive and negative qualification required')
        try:
            program=skill_programs.synthesize(train)
            if program is None:return {'action':'rejected','reason':'no unique bounded program fits TRAIN'}
            skill_programs.validate_program(program)
        except (ValueError,TypeError) as error:return {'action':'rejected','reason':'no unique bounded program fits TRAIN'}
        if _check(program,train) or _check(program,checks):return {'action':'rejected','reason':'new skill qualification failed'}
        # Distinct messages alone are insufficient: positive values must be new.
        def inputs(rows):
            result=set()
            for r in rows:
                if r['answer'] is None:continue
                source=skill_programs.predict(program,r['messages'])['provenance']
                result.add(tuple(v['value'] for v in source['operands']) if source.get('operands') else source['value_span'])
            return result
        if inputs(train)&inputs(checks):raise RegistryError('TRAIN/qualification extracted values overlap')
        entry={'id':digest(program),'lesson_id':label,'program':program,'train':train,'qualification':checks,
               'certificate':{'train_sha256':digest(train),'qualification_sha256':digest(checks),'all_passed':True}}
        with self._lock():
            state,_=self._read()
            if expected_revision is not None and state['revision']!=expected_revision:raise RegistryError('registry CAS revision conflict')
            if any(e['id']==entry['id'] for e in state['entries']):return {'action':'unchanged','skill_id':entry['id'],'revision':state['revision']}
            if len(state['entries'])>=MAX_SKILLS:raise RegistryError('skill capacity exhausted; no knowledge eviction')
            new_coverage=sum(row['answer'] is not None and _route(state['entries'],row['messages']) is None for row in checks)
            if not new_coverage:return {'action':'rejected','reason':'no new qualified coverage'}
            proposal=deepcopy(state);proposal['entries'].append(entry);proposal['revision']+=1
            def route(messages):
                for e in proposal['entries']:
                    result=skill_programs.predict(e['program'],messages)
                    if result is not None:return result
                return None
            # Old skills keep priority, and every old contract is replayed.
            for old in proposal['entries']:
                for row in old['train']+old['qualification']:
                    result=_route(proposal['entries'],row['messages']);answer=result['answer'] if result else None
                    if answer!=row['answer']:return {'action':'rejected','reason':'old/new skill interference'}
            if regression_gate is None:
                from skill_validation import StableGate
                regression_gate=StableGate()
            if regression_gate(route) is not True:return {'action':'rejected','reason':'stable system regression gate failed'}
            publication=self._write(proposal)
            return {'action':'activated','skill_id':entry['id'],'revision':proposal['revision'],'program':program,'new_qualified_coverage':new_coverage,**publication}
