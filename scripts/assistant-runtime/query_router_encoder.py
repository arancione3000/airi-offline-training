"""Optional semantic routing using a frozen Transformer encoder plus TRAIN ridge heads.

This uses a modern Transformer component as a prior. It does not demonstrate
superiority over Transformers or change the approved generative model weights.
"""
from collections import Counter
import hashlib
import json
import math
import os
import re
from pathlib import Path
import stat
import tempfile

import query_router_linear as common
import semantic_encoder

VERSION=4
MAX_BYTES=3*1024*1024
LEARNER_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HELPER_SHA256=hashlib.sha256(Path(semantic_encoder.__file__).read_bytes()).hexdigest()
SCORES=(.35,.45,.55,.65,.75,.85,.90,.95)
MARGINS=(.05,.10,.15,.20)
SUPPORT=(.50,.60,.70,.80,.90,.95)
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')


def _fit(features,labels,count):
    import numpy as np
    x=np.column_stack((features,np.ones(len(features))))
    frequencies=Counter(labels);sample=np.array([1/math.sqrt(frequencies[label]) for label in labels])
    y=np.eye(count,dtype=np.float64)[labels]
    weighted=x*sample[:,None];target=y*sample[:,None]
    # Fixed regularization, no qualification-dependent hyperparameter search.
    weights=weighted.T@np.linalg.solve(weighted@weighted.T+.1*np.eye(len(x)),target)
    return weights


def _rank(features,weights,polarity,classes):
    import numpy as np
    x=np.append(features,1.0);scores=x@weights
    values=np.exp(8*(scores-scores.max()));values/=values.sum()
    order=np.argsort(-values,kind='stable');first,second=order[:2]
    p=x@polarity;prob=np.exp(8*(p-p.max()));prob/=prob.sum()
    return classes[int(first)],float(values[first]),float(values[first]-values[second]),float(prob[1])


def _decision(rank,score,margin,support):
    if rank is None or rank[0] is None or rank[1]<score or rank[2]<margin or rank[3]<support:return None
    return rank[0]


def train(rows,bindings,operations=None,*,family_metadata=None):
    import numpy as np
    bindings=common._bindings(bindings);rows=common._rows(rows,bindings)
    classes=[item['lesson_id'] for item in bindings]+[None];index={label:i for i,label in enumerate(classes)}
    features,encoder_digest=semantic_encoder.encode([row['query'] for row in rows])
    labels=[index[row['lesson_id']] for row in rows];supported=[int(row['lesson_id'] is not None) for row in rows]
    positions=Counter();folds=[]
    for row in rows:folds.append(positions[row['lesson_id']]%5);positions[row['lesson_id']]+=1
    ranks=[None]*len(rows)
    for fold in range(5):
        active=np.array([i for i,f in enumerate(folds) if f!=fold])
        weights=_fit(features[active],[labels[i] for i in active],len(classes))
        polarity=_fit(features[active],[supported[i] for i in active],2)
        for i,row in enumerate(rows):
            if folds[i]==fold and common._normal(row['query']) is not None:ranks[i]=_rank(features[i],weights,polarity,classes)
    candidates={(score,margin,support) for score in SCORES for margin in MARGINS for support in SUPPORT}
    bad=[]
    for row,rank in zip(rows,ranks):
        if rank is None or rank[0] is None:continue
        if row['source_lesson_id'] is not None and rank[0]!=row['source_lesson_id']:continue
        if row['lesson_id'] is None or rank[0]!=row['lesson_id']:bad.append(rank)
    # Deterministic TRAIN threshold learning: each fixed two-coordinate slice
    # adds its exact zero-error frontier for the third coordinate. This avoids
    # a coarse .95 ceiling without consulting qualification or final examples.
    def edge(values,minimum):return min(1.0,max([minimum]+[value+1e-9 for value in values]))
    for score in SCORES:
        for margin in MARGINS:candidates.add((score,margin,edge([r[3] for r in bad if r[1]>=score and r[2]>=margin],.5)))
    for score in SCORES:
        for support in SUPPORT:candidates.add((score,edge([r[2] for r in bad if r[1]>=score and r[3]>=support],.05),support))
    for margin in MARGINS:
        for support in SUPPORT:candidates.add((edge([r[1] for r in bad if r[2]>=margin and r[3]>=support],.35),margin,support))
    options=[]
    for score,margin,support in candidates:
        accepted=wrong=false=0
        for row,rank in zip(rows,ranks):
            label=_decision(rank,score,margin,support)
            if label is not None and row['source_lesson_id'] is not None and label!=row['source_lesson_id']:label=None
            if label is None:continue
            if row['lesson_id'] is None:false+=1
            elif label!=row['lesson_id']:wrong+=1
            else:accepted+=1
        if not wrong and not false:options.append((accepted,score,margin,support))
    accepted,score,margin,support=max(options) if options else (0,1.0,1.0,1.0)
    weights=_fit(features,labels,len(classes));polarity=_fit(features,supported,2)
    if operations is None and family_metadata is not None:operations={item['id']:item['operation'] for item in family_metadata}
    operations=operations or {}
    if operations and (set(operations)!=set(index)-{None} or any(op not in common.skill_programs.OPERATIONS for op in operations.values())):raise ValueError('invalid approved operation metadata')
    payload={'schema':VERSION,'learner_sha256':LEARNER_SHA256,'helper_sha256':HELPER_SHA256,
             'common_sha256':common.LEARNER_SHA256,'engine_sha256':common.ENGINE_SHA256,
             'encoder_manifest_sha256':encoder_digest,'bindings':bindings,'examples':rows,
             'train_sha256':common._digest(rows),'classes':classes,'operations':operations,
             'weights':weights.tolist(),'polarity_weights':polarity.tolist(),'dimensions':int(features.shape[1]),
             'thresholds':{'score':score,'margin':margin,'support':support},
             'calibration':{'method':'TRAIN grouped fivefold; frozen encoder; class-balanced ridge lambda.1; softmax8; support head; observed OOF frontier',
                            'accepted_correct':accepted,'positive_rows':sum(supported),'negative_rows':len(rows)-sum(supported),
                            'false_accepts':0 if options else None,'wrong_classes':0 if options else None,'precision_gate_passed':bool(options)}}
    model={**payload,'sha256':common._digest(payload)};_validate(model);return model


def _validate(model):
    expected={'schema','learner_sha256','helper_sha256','common_sha256','engine_sha256','encoder_manifest_sha256',
              'bindings','examples','train_sha256','classes','operations','weights','polarity_weights','dimensions','thresholds','calibration','sha256'}
    if type(model) is not dict or set(model)!=expected:raise ValueError('invalid semantic router schema')
    if model['schema']!=VERSION or model['learner_sha256']!=LEARNER_SHA256 or model['helper_sha256']!=HELPER_SHA256 or model['common_sha256']!=common.LEARNER_SHA256 or model['engine_sha256']!=common.ENGINE_SHA256:raise ValueError('semantic router version mismatch')
    if model['sha256']!=common._digest({k:v for k,v in model.items() if k!='sha256'}):raise ValueError('semantic router checksum mismatch')
    bindings=common._bindings(model['bindings']);rows=common._rows(model['examples'],bindings)
    if rows!=model['examples'] or model['train_sha256']!=common._digest(rows):raise ValueError('semantic TRAIN digest mismatch')
    if model['classes']!=[r['lesson_id'] for r in bindings]+[None]:raise ValueError('invalid semantic classes')
    dim=model['dimensions']
    if type(dim) is not int or not 1<=dim<=1024:raise ValueError('invalid encoder dimension')
    for field,width in (('weights',len(bindings)+1),('polarity_weights',2)):
        matrix=model[field]
        if type(matrix) is not list or len(matrix)!=dim+1 or any(type(row) is not list or len(row)!=width for row in matrix):raise ValueError('invalid semantic weight shape')
        if any(type(value) not in (int,float) or not math.isfinite(value) or abs(value)>1000 for row in matrix for value in row):raise ValueError('invalid semantic weights')
    thresholds=model['thresholds']
    if type(thresholds) is not dict or set(thresholds)!={'score','margin','support'} or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in thresholds.values()):raise ValueError('invalid semantic thresholds')
    calibration=model['calibration']
    calibration_keys={'method','accepted_correct','positive_rows','negative_rows','false_accepts','wrong_classes','precision_gate_passed'}
    if type(calibration) is not dict or set(calibration)!=calibration_keys or type(calibration.get('precision_gate_passed')) is not bool:raise ValueError('invalid semantic calibration')
    if type(calibration['method']) is not str or len(calibration['method'])>160:raise ValueError('invalid semantic calibration method')
    for key in ('accepted_correct','positive_rows','negative_rows'):
        if type(calibration[key]) is not int or not 0<=calibration[key]<=len(rows):raise ValueError('invalid semantic calibration counts')
    if calibration['positive_rows']!=sum(row['lesson_id'] is not None for row in rows) or calibration['negative_rows']!=sum(row['lesson_id'] is None for row in rows):raise ValueError('semantic calibration row mismatch')
    for key in ('false_accepts','wrong_classes'):
        if calibration[key] is not None and (type(calibration[key]) is not int or not 0<=calibration[key]<=len(rows)):raise ValueError('invalid semantic calibration errors')
        if calibration['precision_gate_passed'] and calibration[key]!=0:raise ValueError('semantic precision gate contradicts errors')
    operations=model['operations']
    if type(operations) is not dict or (operations and (set(operations)!={b['lesson_id'] for b in bindings} or any(type(value) is not str or value not in common.skill_programs.OPERATIONS for value in operations.values()))):raise ValueError('invalid semantic operation metadata')
    if type(model['encoder_manifest_sha256']) is not str or not re.fullmatch(r'[0-9a-f]{64}',model['encoder_manifest_sha256']):raise ValueError('invalid encoder manifest binding')
    return model


class CompiledSemanticRouter:
    def __init__(self,model):
        _validate(model)
        try:import numpy as np
        except ImportError as error:raise ValueError('optional semantic dependencies are unavailable') from error
        self.weights=np.array(model['weights'],dtype=np.float64,copy=True)
        self.polarity=np.array(model['polarity_weights'],dtype=np.float64,copy=True)
        self.weights.flags.writeable=False;self.polarity.flags.writeable=False
        self.classes=tuple(model['classes']);self.thresholds=dict(model['thresholds'])
        self.sha256=model['sha256'];self.encoder_digest=model['encoder_manifest_sha256']
        self.dimensions=model['dimensions'];self.accept=model['calibration']['precision_gate_passed']

    def predict(self,query):
        try:
            if not self.accept or common._normal(query) is None:return None
            features,encoder_digest=semantic_encoder.encode([query])
            if encoder_digest!=self.encoder_digest or features.shape[1]!=self.dimensions:raise ValueError('encoder identity changed')
            rank=_rank(features[0],self.weights,self.polarity,self.classes)
            label=_decision(rank,**self.thresholds)
            if label is None:return None
            return {'lesson_id':label,'score':rank[1],'margin':rank[2],'support_score':rank[3],'model_sha256':self.sha256}
        except ImportError as error:raise ValueError('optional semantic dependencies are unavailable') from error
        except (RecursionError,OverflowError,KeyError,TypeError) as error:raise ValueError('invalid semantic router') from error


def compile_model(model):
    return CompiledSemanticRouter(model)


def predict(model,query):
    try:return compile_model(model).predict(query)
    except (RecursionError,OverflowError,KeyError,TypeError) as error:raise ValueError('invalid semantic router') from error


def read_model(path):
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('semantic model is not regular')
        raw=stream.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise ValueError('semantic model byte bound exceeded')
    try:return _validate(json.loads(raw))
    except (RecursionError,OverflowError,KeyError,TypeError) as error:raise ValueError('invalid semantic model file') from error


def write_model(path,model):
    _validate(model);raw=common._canonical(model)+b'\n'
    if len(raw)>MAX_BYTES:raise ValueError('semantic model byte bound exceeded')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if path.is_symlink():raise ValueError('semantic model path cannot be a symlink')
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path);directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)
