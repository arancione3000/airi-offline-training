"""Regression checks are separate from synthesis; no validation labels enter it."""
import gzip
import hashlib
import json
from pathlib import Path

from context_memory import resolve


CONTRACT=Path(__file__).resolve().parent/'skills/stable_contracts.json.gz'


class StableGate:
    def __init__(self,path=CONTRACT):
        self.path=Path(path)
        raw=gzip.decompress(self.path.read_bytes())
        if len(raw)>4*1024*1024:raise ValueError('stable contract exceeds bound')
        self.contract=json.loads(raw)
        self.sha256=hashlib.sha256(raw).hexdigest()
        if self.contract.get('schema')!=1:raise ValueError('invalid stable contract')
        self.last_report=None

    def __call__(self,route):
        failures=[];intercepts=0;inherited=0
        for i,row in enumerate(self.contract['cases']):
            if resolve(row['messages']) is not None:
                continue # Existing memory always retains routing priority.
            value=route(row['messages'])
            if value is None:inherited+=1
            else:
                intercepts+=1
                if value['answer']!=row['answer']:failures.append(i)
        self.last_report={'contract_sha256':self.sha256,'cases':len(self.contract['cases']),
                          'learned_intercepts':intercepts,'unchanged_native_routes':inherited,
                          'failed_cases':failures,'passed':not failures,
                          'scope':'routing validation; not a new neural certificate'}
        return not failures
