"""Research-only router. An approved object and a supported operation must agree."""
import hashlib
import json
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parents[2] / 'assistant-runtime'
sys.path.insert(0, str(RUNTIME))
import query_router as common
import semantic_encoder


def entity_frame(query, entities):
    normalized = common._normal(query)
    if normalized is None:
        return None
    words = normalized.replace('’', "'").split()
    matches = []
    for lesson, phrase in entities.items():
        parts = phrase.replace('’', "'").split()
        for start in range(len(words) - len(parts) + 1):
            if words[start:start + len(parts)] == parts:
                matches.append((lesson, start, len(parts)))
    if len(matches) != 1:
        return None
    lesson, start, length = matches[0]
    frame = words[:start] + ['il', 'dato'] + words[start + length:]
    return lesson, ' '.join(frame)


def speech_head(frame):
    return frame.split()[0]


def records(rows, entities, operations):
    # A negative is conditional on the source operation. "Somma X" is a valid
    # positive for addition and can be a negative for multiplication.
    unique = set()
    for row in rows:
        parsed = entity_frame(row['query'], entities)
        if parsed is None:
            continue
        lesson, frame = parsed
        if lesson != row['source_lesson_id']:
            continue
        positive = row['lesson_id'] is not None
        if positive and row['lesson_id'] != lesson:
            raise ValueError('positive lesson conflicts with its approved object')
        unique.add((frame, operations[lesson], positive))
    return sorted(unique)


class Prototypes:
    def __init__(self, cases, vectors):
        self.cases = list(cases)
        self.frames = sorted({frame for frame, _, _ in cases})
        self.indices = {frame: i for i, frame in enumerate(self.frames)}
        self.vectors = vectors

    def rank(self, frame, vector, operation, exclude=None):
        # Exclude the whole masked utterance across all objects in calibration.
        active = [r for r in self.cases if r[0] != exclude]
        if not any(positive and speech_head(text) == speech_head(frame)
                   for text, _, positive in active):
            return None
        similarity = self.vectors @ vector
        support = []
        opposition = []
        for text, op, positive in active:
            score = float(similarity[self.indices[text]])
            if positive and op == operation:
                support.append(score)
            elif (positive and op != operation) or (not positive and op == operation):
                opposition.append(score)
        if not support or not opposition:
            return None
        best = max(support)
        return best, best - max(opposition)


def accepts(rank, thresholds):
    return rank is not None and all(a >= b for a, b in zip(rank, thresholds))


def train(rows, bindings, families):
    entities = {f['id']: common._normal(f['object_phrase']) for f in families}
    operations = {f['id']: f['operation'] for f in families}
    if set(entities) != {b['lesson_id'] for b in bindings} or None in entities.values():
        raise ValueError('family inventory does not match the bank')
    cases = records(rows, entities, operations)
    frames = sorted({r[0] for r in cases})
    vectors, encoder = semantic_encoder.encode(frames)
    prototypes = Prototypes(cases, vectors)
    ranks = [(prototypes.rank(text, vectors[prototypes.indices[text]], op, exclude=text), positive)
             for text, op, positive in cases]
    options = []
    for score in (.35, .45, .55, .65, .75, .85):
        for margin in (.01, .03, .05, .10, .15, .20):
            errors = sum(accepts(rank, (score, margin)) for rank, pos in ranks if not pos)
            if errors == 0:
                correct = sum(accepts(rank, (score, margin)) for rank, pos in ranks if pos)
                options.append((correct, score, margin))
    correct, score, margin = max(options, default=(0, 2., 2.))
    payload = dict(schema='research-typed-v1', bindings=bindings, entities=entities,
                   operations=operations, cases=cases, thresholds=[score, margin],
                   train_digest=common._digest(rows), encoder_digest=encoder,
                   learner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   calibration=dict(unique_conditional_cases=len(cases), unique_frames=len(frames),
                                    supported_cases=sum(r[2] for r in cases),
                                    correct=correct, false_accepts=0))
    return {**payload, 'sha256': common._digest(payload)}


def read_model(path):
    with Path(path).open() as stream:
        model = json.load(stream)
    if model['sha256'] != common._digest({k: v for k, v in model.items() if k != 'sha256'}):
        raise ValueError('typed research model checksum mismatch')
    if model['learner_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('typed research learner changed after freeze')
    return model


class Compiled:
    def __init__(self, model):
        self.model = model
        frames = sorted({row[0] for row in model['cases']})
        vectors, digest = semantic_encoder.encode(frames)
        if digest != model['encoder_digest']:
            raise ValueError('frozen encoder identity changed')
        self.prototypes = Prototypes(model['cases'], vectors)

    def predict(self, query):
        parsed = entity_frame(query, self.model['entities'])
        if parsed is None:
            return None
        lesson, frame = parsed
        vectors, digest = semantic_encoder.encode([frame])
        if digest != self.model['encoder_digest']:
            raise ValueError('frozen encoder identity changed')
        rank = self.prototypes.rank(frame, vectors[0], self.model['operations'][lesson])
        if not accepts(rank, self.model['thresholds']):
            return None
        return dict(lesson_id=lesson, score=rank[0], margin=rank[1],
                    model_sha256=self.model['sha256'])


def compile_model(model):
    return Compiled(model)
