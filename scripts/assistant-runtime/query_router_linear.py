"""Bounded lexical question routing learned only from trusted TRAIN examples.

Averaged multiclass hinge weights over sparse TF-IDF word/character views choose a
GLOBAL family or reject. This is lexical paraphrase routing, not pretrained
language understanding. A caller must independently verify source compatibility,
old-route priority and program bindings before invoking the selected expert.
"""
from collections import Counter, OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import stat
import tempfile
import unicodedata

import skill_programs

VERSION = 2
MAX_ROWS = 768
MAX_CLASSES = 32
MAX_BYTES = 3 * 1024 * 1024
MAX_QUERY = 512
SCORES = (.35, .45, .55, .65, .75, .85, .90, .95)
MARGINS = (.05, .10, .15, .20)
OVERLAP = .35
LEARNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ENGINE_SHA256 = hashlib.sha256(Path(skill_programs.__file__).read_bytes()).hexdigest()
_CACHE = OrderedDict()
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_UNSUPPORTED = re.compile(
    r"\b(?:non|mai|senza|not|never|except|instead|rather|invece|oppure|anziché|anziche|"
    r"eccetto|ignora|ignorare|istruzioni|instructions|system|developer|prompt|entrambi|entrambe)\b|"
    r"\be\s+poi\b|\bsecond[oa]\b.{0,60}\bprim[oa]\b|\bdal\s+secondo\b", re.I)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _normal(query):
    if type(query) is not str or not 0 < len(query) <= MAX_QUERY:
        return None
    if any(not c.isprintable() or unicodedata.category(c).startswith('C') for c in query):
        return None
    if any(c in query for c in '\"“”«»`<>\\|[]{}') or _UNSUPPORTED.search(query):
        return None
    # Apostrophes inside words are grammatical; quoted standalone strings are not.
    if re.search(r"(?:^|\s)['’].+?['’](?:$|\s|[.!?])", query):
        return None
    result = ' '.join(_WORD.findall(unicodedata.normalize('NFC', query).casefold()))
    return result if result else None


def _features(text):
    words = text.split()
    result = {f'w:{w}': 1.0 for w in words}
    result.update({f'b:{a} {b}': 1.2 for a, b in zip(words, words[1:])})
    padded = '^' + text + '$'
    for n in (3, 4):
        result.update({f'c{n}:{padded[i:i+n]}': .25 for i in range(max(0, len(padded)-n+1))})
    return result


def _rows(rows, bindings):
    if type(rows) is not list or not 4 <= len(rows) <= MAX_ROWS:
        raise ValueError('invalid bounded router TRAIN size')
    classes = {b['lesson_id'] for b in bindings}
    result = []; ids = set(); queries = {}
    for row in rows:
        if type(row) is not dict or not {'id','lesson_id','query'} <= set(row) or set(row)-{'id','lesson_id','query','source_lesson_id'}:
            raise ValueError('invalid router TRAIN row')
        label = row['lesson_id']; rid = row['id']; query = row['query']; source = row.get('source_lesson_id')
        if type(rid) is not str or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}', rid) or rid in ids:
            raise ValueError('invalid or duplicate row identity')
        ids.add(rid)
        if label is not None and (type(label) is not str or label not in classes):
            raise ValueError('unknown TRAIN class')
        if source is not None and (type(source) is not str or source not in classes):
            raise ValueError('unknown TRAIN source family')
        if type(query) is not str or not 0 < len(query) <= MAX_QUERY or any(unicodedata.category(c)=='Cs' for c in query):
            raise ValueError('invalid TRAIN query')
        norm = _normal(query)
        # Collapse equivalent positives to prevent LOO leakage through duplicates.
        if norm is not None:
            if norm in queries:
                if queries[norm] != label:raise ValueError('conflicting normalized TRAIN labels')
                continue
            queries[norm] = label
        result.append({'id':rid,'lesson_id':label,'query':query,'source_lesson_id':source})
    if any(sum(row['lesson_id']==label and _normal(row['query']) is not None for row in result)<2 for label in classes):
        raise ValueError('each class needs distinct positive paraphrases')
    if not any(row['lesson_id'] is None for row in result):
        raise ValueError('explicit reject TRAIN examples required')
    return result


def _bindings(bindings):
    if type(bindings) is not list or not 1 <= len(bindings) <= MAX_CLASSES:
        raise ValueError('invalid bounded router class count')
    result = []; labels = set(); programs = set()
    for row in bindings:
        if type(row) is not dict or set(row)!={'lesson_id','program_id','canonical_query'}:
            raise ValueError('invalid router binding')
        label, program, query = row['lesson_id'], row['program_id'], row['canonical_query']
        if type(label) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', label) or label in labels:
            raise ValueError('invalid or duplicate binding class')
        if type(program) is not str or not re.fullmatch(r'[0-9a-f]{64}', program) or program in programs:
            raise ValueError('invalid or duplicate program identity')
        if _normal(query) is None:raise ValueError('unsupported canonical question')
        labels.add(label);programs.add(program);result.append(dict(row))
    return result


def _vector(features,idf):
    result = {f:w*idf[f] for f,w in features.items() if f in idf}
    norm = math.sqrt(sum(w*w for w in result.values()))
    return {f:w/norm for f,w in result.items()} if norm else {}


def _decision(rank, score, margin):
    if rank is None or rank[0] is None or rank[1]+1e-12<score or rank[2]+1e-12<margin:
        return None
    return rank[0]


def _linear_space(rows):
    raw = [_features(_normal(row['query'])) if _normal(row['query']) else {} for row in rows]
    df=Counter(feature for vector in raw for feature in vector)
    idf={feature:1+math.log((len(rows)+1)/(count+1)) for feature,count in df.items()}
    return idf,[_vector(vector,idf) for vector in raw]


def _fit(rows,classes):
    idf,vectors=_linear_space(rows)
    weights=[{} for _ in classes];totals=[{} for _ in classes];stamps=[{} for _ in classes]
    bias=[0.0]*len(classes);bias_totals=[0.0]*len(classes);bias_stamps=[0]*len(classes)
    index={label:i for i,label in enumerate(classes)};step=0
    order=list(range(len(rows)));rng=random.Random(20261116)
    for epoch in range(40):
        rng.shuffle(order)
        for row_index in order:
            step+=1;vector=vectors[row_index]
            if not vector:continue
            target=index[rows[row_index]['lesson_id']]
            scores=[bias[k]+sum(w.get(f,0.0)*v for f,v in vector.items()) for k,w in enumerate(weights)]
            other=max((k for k in range(len(classes)) if k!=target),key=lambda k:scores[k])
            loss=1.0-scores[target]+scores[other]
            if loss<=0:continue
            rate=min(.25,loss/2.02)
            for label,sign in ((target,1.0),(other,-1.0)):
                for feature,value in vector.items():
                    old=weights[label].get(feature,0.0)
                    totals[label][feature]=totals[label].get(feature,0.0)+(step-stamps[label].get(feature,0))*old
                    stamps[label][feature]=step
                    weights[label][feature]=old+sign*rate*value
                bias_totals[label]+=(step-bias_stamps[label])*bias[label];bias_stamps[label]=step
                bias[label]+=sign*rate*.1
    for label in range(len(classes)):
        weights[label]={feature:(totals[label].get(feature,0.0)+(step-stamps[label][feature])*value)/max(1,step)
                        for feature,value in weights[label].items()}
        bias[label]=(bias_totals[label]+(step-bias_stamps[label])*bias[label])/max(1,step)
    return idf,weights,bias


def _linear_rank(classes,weights,bias,idf,query):
    norm=_normal(query)
    if norm is None:return None
    raw=_features(norm)
    if sum(f in idf for f in raw)/len(raw)<OVERLAP:return None
    words=[f for f in raw if f.startswith('w:')]
    if words and sum(f in idf for f in words)/len(words)<.85:return None
    vector=_vector(raw,idf)
    if not vector:return None
    scores=[bias[k]+sum(w.get(f,0.0)*v for f,v in vector.items()) for k,w in enumerate(weights)]
    # A fixed temperature exposes separating margins; these remain scores, not
    # externally calibrated correctness probabilities.
    peak=max(scores);exp=[math.exp(4*(value-peak)) for value in scores];total=sum(exp)
    ranked=sorted(range(len(classes)),key=lambda i:(-scores[i],str(classes[i])))
    first,second=ranked[:2]
    return classes[first],exp[first]/total,(exp[first]-exp[second])/total


def train(rows, bindings):
    bindings=_bindings(bindings);rows=_rows(rows,bindings)
    classes=[b['lesson_id'] for b in bindings]+[None]
    # Within-label index groups keep the same authored paraphrase position out
    # across every family at once. No qualification/final labels enter training.
    positions=Counter();folds=[]
    for row in rows:
        label=row['lesson_id'];folds.append(positions[label]%5);positions[label]+=1
    ranks=[None]*len(rows)
    for fold in range(5):
        training=[row for i,row in enumerate(rows) if folds[i]!=fold]
        idf,weights,bias=_fit(training,classes)
        for i,row in enumerate(rows):
            if folds[i]==fold:ranks[i]=_linear_rank(classes,weights,bias,idf,row['query'])
    options=[]
    for score in SCORES:
        for margin in MARGINS:
            accepted=wrong=false=0
            for row,rank in zip(rows,ranks):
                label=_decision(rank,score,margin)
                # Reproduce deployment: GLOBAL choice first, then source gate.
                if label is not None and row['source_lesson_id'] is not None and label!=row['source_lesson_id']:label=None
                if label is None:continue
                if row['lesson_id'] is None:false+=1
                elif label!=row['lesson_id']:wrong+=1
                elif row['source_lesson_id'] in (None,label):accepted+=1
            if not wrong and not false:options.append((accepted,score,margin))
    accepted,score,margin=max(options) if options else (0,1.0,1.0)
    _,weights,bias=_fit(rows,classes)
    payload={'schema':VERSION,'learner_sha256':LEARNER_SHA256,'engine_sha256':ENGINE_SHA256,
             'bindings':bindings,'examples':rows,'train_sha256':_digest(rows),'classes':classes,
             'weights':weights,'biases':bias,
             'thresholds':{'score':score,'margin':margin,'overlap':OVERLAP},
             'calibration':{'method':'TRAIN grouped fivefold; averaged multiclass hinge40epochs; TFIDF; fixed softmax scale4',
                            'accepted_correct':accepted,'positive_rows':sum(r['lesson_id'] is not None for r in rows),
                            'negative_rows':sum(r['lesson_id'] is None for r in rows),
                            'false_accepts':0 if options else None,'wrong_classes':0 if options else None,
                            'precision_gate_passed':bool(options)}}
    model={**payload,'sha256':_digest(payload)};_validate(model);return model


def _validate(model):
    if type(model) is not dict or set(model)!={'schema','learner_sha256','engine_sha256','bindings','examples','train_sha256','thresholds','calibration','classes','weights','biases','sha256'}:
        raise ValueError('invalid router schema')
    if type(model['schema']) is not int or model['schema']!=VERSION or model['learner_sha256']!=LEARNER_SHA256 or model['engine_sha256']!=ENGINE_SHA256:
        raise ValueError('router version binding mismatch')
    payload = {k:v for k,v in model.items() if k!='sha256'}
    if model['sha256']!=_digest(payload):raise ValueError('router checksum mismatch')
    bindings = _bindings(model['bindings']);rows = _rows(model['examples'],bindings)
    if rows!=model['examples'] or model['train_sha256']!=_digest(rows):raise ValueError('router TRAIN digest mismatch')
    thresholds = model['thresholds']
    if type(thresholds) is not dict or set(thresholds)!={'score','margin','overlap'}:
        raise ValueError('invalid router thresholds')
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in thresholds.values()) or thresholds['overlap']!=OVERLAP:
        raise ValueError('invalid router threshold values')
    if model['classes']!=[b['lesson_id'] for b in bindings]+[None]:raise ValueError('invalid linear classes')
    weights=model['weights'];bias=model['biases']
    if type(weights) is not list or type(bias) is not list or len(weights)!=len(bindings)+1 or len(bias)!=len(weights):raise ValueError('invalid linear dimensions')
    if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1000 for v in bias):raise ValueError('invalid linear bias')
    total=0
    for vector in weights:
        if type(vector) is not dict or len(vector)>20000:raise ValueError('invalid linear vector')
        total+=len(vector)
        if any(type(f) is not str or len(f)>MAX_QUERY+8 or not f.startswith(('w:','b:','c3:','c4:')) or type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1000 for f,v in vector.items()):raise ValueError('invalid linear weight')
    if total>100000:raise ValueError('linear weight capacity exceeded')
    calibration=model['calibration']
    expected={'method','accepted_correct','positive_rows','negative_rows','false_accepts','wrong_classes','precision_gate_passed'}
    if type(calibration) is not dict or set(calibration)!=expected or type(calibration['precision_gate_passed']) is not bool:
        raise ValueError('invalid router calibration')
    for key in ('accepted_correct','positive_rows','negative_rows'):
        if type(calibration[key]) is not int or not 0<=calibration[key]<=len(rows):raise ValueError('invalid calibration counts')
    if not isinstance(calibration['method'],str) or len(calibration['method'])>160:raise ValueError('invalid calibration method')
    for key in ('false_accepts','wrong_classes'):
        if calibration[key] is not None and (type(calibration[key]) is not int or not 0<=calibration[key]<=len(rows)):raise ValueError('invalid calibration errors')
    return model


def predict(model, query):
    try:
        _validate(model)
        if not model['calibration']['precision_gate_passed']:return None
        key = model['sha256']
        if key not in _CACHE:
            _CACHE[key] = _linear_space(model['examples'])[0]
            if len(_CACHE)>4:_CACHE.popitem(last=False)
        else:_CACHE.move_to_end(key)
        idf = _CACHE[key]
        rank = _linear_rank(model['classes'],model['weights'],model['biases'],idf,query)
        label = _decision(rank,model['thresholds']['score'],model['thresholds']['margin'])
        if label is None:return None
        return {'lesson_id':label,'score':rank[1],'margin':rank[2],'model_sha256':key}
    except (RecursionError,OverflowError,TypeError,KeyError) as error:
        raise ValueError('invalid bounded router model') from error


def read_model(path):
    fd = os.open(path,os.O_RDONLY|os.O_NONBLOCK|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('router is not a regular file')
        raw=stream.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise ValueError('router byte limit exceeded')
    try:return _validate(json.loads(raw))
    except (RecursionError,OverflowError,TypeError,KeyError) as error:
        raise ValueError('invalid bounded router file') from error


def write_model(path,model):
    _validate(model);raw=_canonical(model)+b'\n'
    if len(raw)>MAX_BYTES:raise ValueError('router byte limit exceeded')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if path.is_symlink():raise ValueError('router destination cannot be a symlink')
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)
