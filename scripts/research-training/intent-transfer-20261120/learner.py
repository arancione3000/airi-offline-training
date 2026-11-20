"""Research-only operation-conditioned retrieval from trusted Italian requests.

An approved object's name is removed before encoding. Rank the requested operation
against other operations and explicit same-operation rejects. A word ignored by
the encoder still has to pass a positive-vocabulary gate. Threshold selection
holds out the entire masked wording across objects and operations.
"""

from collections import Counter
from functools import lru_cache
import gzip
import json
from pathlib import Path
import re
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
for directory in (RESEARCH.parent / "assistant-runtime", RESEARCH / "intent-20261120",
                  RESEARCH / "verified-frames-20261120"):
    sys.path.insert(0, str(directory))

from intent_grammar import _bind
import query_router
import semantic_encoder


def _frame(query, families):
    if query_router._normal(query) is None:
        return None
    text = unicodedata.normalize("NFC", query).strip().lower().replace("’", "'")
    text = re.sub(r"[.!?]$", "", text)
    # A single clause. Leave contractions intact during typed object binding.
    if not re.fullmatch(r"[a-zàèéìòù',: ]+", text):
        return None
    bound = _bind(text, families)
    if bound is None:
        return None
    family, frame = bound
    tokens = re.findall(r"[a-zàèéìòù]+|@", frame)
    if len(tokens) < 2:
        return None
    return family, " ".join(tokens)


def _keys(frame):
    # Group inflections conservatively. Coarse prefix matches alone never
    # establish an operation, which also needs support in the encoder space.
    return frozenset(token[:5] for token in frame.split())


def _training_rows(expanded):
    path = RESEARCH / "semantic-routing-20261116/evidence/precommitted/TRAIN.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if data["split"] != "TRAIN":
        raise ValueError("learner requires trusted TRAIN examples")
    yield from data["rows"]
    if expanded:
        from pool import POOL
        from pool_confirmation import POOL as NEXT_POOL

        for row in (*POOL, *NEXT_POOL):
            yield {"query": row["query"], "source_lesson_id": row["family"],
                   "lesson_id": row["family"] if row["intent"] is not None else None,
                   "operation_label": row["intent"]}


@lru_cache(maxsize=1024)
def _embedding(frame):
    return semantic_encoder.encode([frame.replace("@", "il dato")])[0][0]


class Router:
    def __init__(self, families, cases):
        self.families = tuple(families)
        self.frames = sorted({frame for frame, _, _ in cases})
        self.index = {frame: i for i, frame in enumerate(self.frames)}
        self.cases = tuple((self.index[frame], op, accepted) for frame, op, accepted in cases)
        self.keys = [_keys(frame) for frame in self.frames]
        self.vocab = Counter()
        for i in {i for i, _, accepted in self.cases if accepted}:
            self.vocab.update(self.keys[i])
        self.vectors, self.encoder_digest = semantic_encoder.encode(
            [frame.replace("@", "il dato") for frame in self.frames])
        # Cache the Gram matrix once. Omit every occurrence of the evaluated
        # frame from the support, opposition and positive vocabulary.
        self.positive_ids = {i for i, _, accepted in self.cases if accepted}
        similarities = self.vectors @ self.vectors.T
        operations = sorted({op for _, op, _ in self.cases})
        ranks = []
        for i, op, positive in self.cases:
            ranks.append((self._rank(op, self.keys[i], similarities[i], i), positive))
        # Include hard negatives built by requesting a different operation on
        # the same typed input. Source compatibility alone cannot pass this gate.
        for i in sorted(self.positive_ids):
            supported = {op for j, op, ok in self.cases if j == i and ok}
            for op in operations:
                if op not in supported:
                    ranks.append((self._rank(op, self.keys[i], similarities[i], i), False))
        candidates = []
        for score in (.50, .60, .70, .80, .90):
            for margin in (.02, .05, .10, .15, .20):
                accepted = [positive for rank, positive in ranks if self._pass(rank, score, margin)]
                if all(accepted):
                    candidates.append((len(accepted), score, margin))
        n_correct, self.score, self.margin = max(candidates, default=(0, 2., 2.))
        self.calibration = {"eligible_cases": len(self.cases), "masked_wordings": len(self.frames),
                            "calibration_cases": len(ranks), "correct": n_correct,
                            "score": self.score, "margin": self.margin,
                            "encoder_digest": self.encoder_digest}

    @staticmethod
    def _pass(rank, score, margin):
        return rank is not None and rank[0] >= score and rank[1] >= margin

    def _rank(self, op, keys, vector, omit=None):
        # Do not fill missing words from a withheld positive during calibration.
        withdrawn = self.keys[omit] if omit in self.positive_ids else ()
        if any(self.vocab[key] <= int(key in withdrawn) for key in keys):
            return None
        support = []
        opposition = []
        for i, observed_op, accepted in self.cases:
            if i == omit:
                continue
            if observed_op == op and accepted:
                support.append(float(vector[i]))
            elif (observed_op == op and not accepted) or (observed_op != op and accepted):
                opposition.append(float(vector[i]))
        if len(support) < 2 or not opposition:
            return None
        best, second = sorted(support, reverse=True)[:2]
        evidence = (best + second) * .5
        return evidence, evidence - max(opposition)

    @lru_cache(maxsize=512)
    def __call__(self, query):
        bound = _frame(query, self.families)
        if bound is None:
            return None
        family, frame = bound
        keys = _keys(frame)
        if not all(key in self.vocab for key in keys):
            return None
        vector = self.vectors @ _embedding(frame)
        rank = self._rank(family["operation"], keys, vector)
        return family["id"] if self._pass(rank, self.score, self.margin) else None


def fit(families, expanded):
    families = tuple(families)
    operations = {family["id"]: family["operation"] for family in families}
    cases = set()
    for row in _training_rows(expanded):
        bound = _frame(row["query"], families)
        if bound is None:
            continue
        family, frame = bound
        if family["id"] != row["source_lesson_id"]:
            continue
        label = row["lesson_id"]
        if label is not None and (label != family["id"] or
                                  row.get("operation_label", operations[label]) != operations[label]):
            raise ValueError("positive label disagrees with the installed expert")
        cases.add((frame, family["operation"], label is not None))
    if any((frame, op, not accepted) in cases for frame, op, accepted in cases):
        raise ValueError("conflicting labels for a masked request and operation")
    return Router(families, sorted(cases))
