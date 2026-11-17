"""Bounded lexical question routing learned only from trusted TRAIN examples.

Sparse TF-IDF nearest examples (word/bigram and character3/4gram views) choose a
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
import stat
import tempfile
import unicodedata

import skill_programs

VERSION = 1
MAX_ROWS = 768
MAX_CLASSES = 32
MAX_BYTES = 1024 * 1024
MAX_QUERY = 512
SCORES = (.35, .45, .55, .65, .75)
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


def _space(rows, raw=None):
    if raw is None:
        raw = [_features(_normal(row['query'])) if _normal(row['query']) else {} for row in rows]
    df = Counter(feature for features in raw for feature in features)
    idf = {feature:1+math.log((len(rows)+1)/(count+1)) for feature,count in df.items()}
    vectors = [_vector(features,idf) for features in raw]
    sums = {}
    for row,vector in zip(rows,vectors):
        if row['lesson_id'] is None:continue
        target = sums.setdefault(row['lesson_id'],{})
        for feature,value in vector.items():target[feature]=target.get(feature,0.0)+value
    centroids = {}
    for label,vector in sums.items():
        norm=math.sqrt(sum(v*v for v in vector.values()))
        centroids[label]={f:v/norm for f,v in vector.items()} if norm else {}
    return idf,vectors,centroids


def _vector(features,idf):
    result = {f:w*idf[f] for f,w in features.items() if f in idf}
    norm = math.sqrt(sum(w*w for w in result.values()))
    return {f:w/norm for f,w in result.items()} if norm else {}


def _rank(rows, idf, vectors, centroids, query, omit=None):
    normal = _normal(query)
    if normal is None:return None
    features = _features(normal)
    overlap = sum(f in idf for f in features)/len(features)
    vector = _vector(features,idf)
    if overlap < OVERLAP or not vector:return None
    scores = {}
    for index,(row,other) in enumerate(zip(rows,vectors)):
        if index==omit:continue
        score = sum(weight*other.get(f,0.0) for f,weight in vector.items())
        label = row['lesson_id']
        scores[label] = max(scores.get(label,0.0),score)
    # The learned centroid emphasizes family features shared across carriers.
    # A reject prototype keeps its nearest-example score because reject intents
    # are intentionally heterogeneous rather than one semantic family.
    for label,centroid in centroids.items():
        similarity=sum(weight*centroid.get(f,0.0) for f,weight in vector.items())
        scores[label]=.75*similarity+.25*scores.get(label,0.0)
    ranked = sorted(scores.items(),key=lambda item:(-item[1],str(item[0])))
    if not ranked:return None
    winner,score = ranked[0]
    margin = score-(ranked[1][1] if len(ranked)>1 else 0.0)
    return winner,min(1.0,max(0.0,score)),max(0.0,margin)


def _decision(rank, score, margin):
    if rank is None or rank[0] is None or rank[1]+1e-12<score or rank[2]+1e-12<margin:
        return None
    return rank[0]


def train(rows, bindings):
    bindings = _bindings(bindings);rows = _rows(rows,bindings)
    raw = [_features(_normal(row['query'])) if _normal(row['query']) else {} for row in rows]
    # The held-out row cannot supply its own vocabulary/IDF or nearest example.
    # Rebuild IDF per fold so this remains entirely TRAIN leave-one-out.
    ranks = []
    for index,row in enumerate(rows):
        rest = rows[:index]+rows[index+1:]
        fold_idf,fold_vectors,fold_centroids = _space(rest,raw[:index]+raw[index+1:])
        ranks.append(_rank(rest,fold_idf,fold_vectors,fold_centroids,row['query']))
    options = []
    for score in SCORES:
        for margin in MARGINS:
            accepted = wrong = false = 0
            for row,rank in zip(rows,ranks):
                label = _decision(rank,score,margin)
                if label is None:continue
                if row['lesson_id'] is None:false+=1
                elif label!=row['lesson_id']:wrong+=1
                elif row['source_lesson_id'] in (None,label):accepted+=1
            if not wrong and not false:options.append((accepted,score,margin))
    if options:
        accepted,score,margin = max(options)
    else:
        # No threshold passes TRAIN precision: produce an explicit abstaining
        # candidate; independent qualification will reject zero coverage.
        accepted,score,margin = 0,1.0,1.0
    payload = {'schema':VERSION,'learner_sha256':LEARNER_SHA256,'engine_sha256':ENGINE_SHA256,
               'bindings':bindings,'examples':rows,'train_sha256':_digest(rows),
               'thresholds':{'score':score,'margin':margin,'overlap':OVERLAP},
               'calibration':{'method':'TRAIN LOO TFIDF with .75 family-centroid/.25 nearest views; reject nearest; fold IDF rebuilt',
                              'accepted_correct':accepted,'positive_rows':sum(r['lesson_id'] is not None for r in rows),
                              'negative_rows':sum(r['lesson_id'] is None for r in rows),
                              'false_accepts':0 if options else None,'wrong_classes':0 if options else None,
                              'precision_gate_passed':bool(options)}}
    model = {**payload,'sha256':_digest(payload)}
    _validate(model)
    return model


def _validate(model):
    if type(model) is not dict or set(model)!={'schema','learner_sha256','engine_sha256','bindings','examples','train_sha256','thresholds','calibration','sha256'}:
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
            _CACHE[key] = _space(model['examples'])
            if len(_CACHE)>4:_CACHE.popitem(last=False)
        else:_CACHE.move_to_end(key)
        idf,vectors,centroids = _CACHE[key]
        rank = _rank(model['examples'],idf,vectors,centroids,query)
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
