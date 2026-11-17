"""Bounded learned question routing after stable memory and exact experts.

The original request is classified globally, never among source-filtered classes.
Query rewriting is internal and preserves original source provenance.
"""
from pathlib import Path
import json

import query_router
import skill_programs
from skill_bank import RegistryError

DEFAULT_MODEL=Path(__file__).resolve().parent/'queries/approved-router.json'


def default_router(bank):
    if bank is None or not DEFAULT_MODEL.exists():return None
    try:
        if DEFAULT_MODEL.is_symlink() or DEFAULT_MODEL.stat().st_size>3*1024*1024:return None
        schema=json.loads(DEFAULT_MODEL.read_bytes())['schema']
    except (OSError,ValueError,KeyError,TypeError,RecursionError):return None
    if schema==1:backend=query_router
    elif schema==2:
        import query_router_linear as backend
    elif schema==3:
        import query_router_factorized as backend
    elif schema==4:
        import query_router_encoder as backend
    else:return None
    return LearnedQueryRouter(bank,DEFAULT_MODEL,backend=backend)


class LearnedQueryRouter:
    def __init__(self,bank,path=DEFAULT_MODEL,*,frozen=False,backend=None):
        self.bank=bank;self.path=Path(path)
        self.backend=backend or query_router
        self.snapshot=bank._read()[0] if frozen else None
        self.model=self.backend.read_model(self.path) if frozen else None
        self.compiled=self.backend.compile_model(self.model) if frozen and hasattr(self.backend,'compile_model') else None

    def predict(self,messages):
        if skill_programs._eligible(messages) is None:return None
        original=messages[-1]['content']
        try:
            model=self.model if self.model is not None else self.backend.read_model(self.path)
        except (ValueError,TypeError,KeyError,RecursionError,ImportError) as error:
            raise RegistryError('learned query model unavailable') from error
        state=self.snapshot if self.snapshot is not None else self.bank._read()[0]
        bindings={row['program_id']:row for row in model['bindings']}
        compatible=[]
        for entry in state['entries']:
            binding=bindings.get(entry['id'])
            if binding is None:continue
            if binding['canonical_query']!=entry['program']['query']:raise RegistryError('query binding mismatch')
            rewritten=list(messages);rewritten[-1]={**messages[-1],'content':binding['canonical_query']}
            result=skill_programs.predict(entry['program'],rewritten)
            if result is not None:compatible.append((entry,binding,result))
        if len(compatible)!=1:return None
        # This existence check changes no class scores: the query classifier
        # still ranks ALL classes, never just the compatible source family.
        try:
            route=self.compiled.predict(original) if self.compiled is not None else self.backend.predict(model,original)
        except (ValueError,TypeError,KeyError,RecursionError,ImportError) as error:
            raise RegistryError('learned query prediction unavailable') from error
        if route is None:return None
        entry,binding,result=compatible[0]
        if binding['lesson_id']!=route['lesson_id']:return None
        return {**result,'skill_id':entry['id'],'skill_revision':state['revision'],
                'routing':'learned_paraphrase','query_provenance':{
                    'original_query':original,'canonical_query':binding['canonical_query'],
                    **route}}
