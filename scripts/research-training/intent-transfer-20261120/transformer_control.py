"""Replay two archived MiniLM routers through the experiment's common executor.

V4 scores the whole question with TRAIN-fitted class and polarity ridge heads.
V5 masks a known object and compares positive and opposing prototypes under a
speech-head constraint. Both use the same frozen, quantized multilingual MiniLM.
"""
import gzip
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


RESEARCH = Path(__file__).resolve().parents[1]
TYPED = RESEARCH / 'typed-routing-20261118'
sys.path.insert(0, str(RESEARCH.parent / 'assistant-runtime'))
sys.path.insert(0, str(TYPED))

import query_router_encoder
import typed_router


ARCHIVES = {
    'minilm_v4': (
        RESEARCH / 'semantic-routing-20261116/evidence/encoder-run/candidate.json.gz',
        'e16dd0dcacb87dc0f414a1807cf98c1e967719f09b9a955bcc754565ebb6aa82',
        query_router_encoder,
    ),
    'minilm_v5': (
        TYPED / 'evidence/candidate.json.gz',
        '7eba107ee56fcb876b57befdcb58d96a35f9d315b0eb6f8054d8e612489cc1f2',
        typed_router,
    ),
}


def load(name):
    archive, expected_digest, backend = ARCHIVES[name]
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError(f'{name} archive differs from the published control')
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'candidate.json'
        path.write_bytes(gzip.decompress(raw))
        router = backend.compile_model(backend.read_model(path))

    def predict(query, families):
        # Both archived routers carry their frozen family inventory.
        result = router.predict(query)
        return result['lesson_id'] if result is not None else None

    return predict
