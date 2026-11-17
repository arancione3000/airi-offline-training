# Reproducing the failed question-routing trials

The independent corpus was frozen before learner evaluation. Its original
manifest SHA256 is
`6ce00c926513aca84c95141e6b54c88f96bdf8afd22282fe8ab3d2c16b2f1f10`.
It contains eighteen families,270 TRAIN rows,126 qualification rows,180 FINAL
rows and six previously observed diagnostic questions. FINAL was never evaluated.
See [plan](PLAN.md) for split rules and [results](RESULTS.md) for rejection reasons.

`evidence/MANIFEST.json` records raw/compressed hashes, artifact roles and excluded
material. Gzip bytes use timestamp0. The package retains the six exact corpus
payloads, the exact synthetic eighteen-expert bank, original run preregistrations,
freezes, qualification outputs and as-run source snapshots. No private conversation,
neural checkpoint, optimizer, credential or pretrained encoder weight is included.
V1/V4 candidate models are archived; V2/V3 larger weight arrays remain local and
their metadata/hashes and deterministic retraining source are archived instead.
Do not conflate separately explored TRAIN prototypes with original run candidates.

Restore each `.gz` artifact to a fresh directory by stripping one `.gz` suffix
and decompressing it. Verify against MANIFEST before use. For exact historical
source reproduction, restore that run's `sources/*.py` into an isolated checkout;
later interface/performance changes do not rewrite the archived first result.
Use the archived bank, not a later public or private bank with different lessons.

From the repository root, standard-library lexical trials can be rerun with:

```sh
python scripts/research-training/semantic-routing-20261116/run.py \
  --backend centroid --corpus /path/to/restored/precommitted \
  --bank /path/to/restored/bank.json --output /tmp/airi-new-lexical-run
```

Use `--backend factorized` for V3. V2 was rejected during TRAIN calibration; its
metadata explicitly records that no qualification outcome exists. A rerun after
observing results is operational reproduction, not another independent blind test.

The optional V4 encoder requires the pinned optional packages and about128MB of
verified public assets. The model revision is
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`; exact file hashes are in
`ENCODER_MANIFEST.json`. The source model is available under Apache-2.0 at
[Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
This is an encoder component; it does not generate answers. Downloading is a
separate explicit setup action; inference never downloads files or calls an API.

```sh
python -m pip install -r scripts/research-training/semantic-routing-20261116/requirements-encoder.txt
python scripts/research-training/semantic-routing-20261116/download_encoder.py --output /tmp/airi-query-encoder
AIRI_QUERY_ENCODER_CACHE=/tmp/airi-query-encoder python \
  scripts/research-training/semantic-routing-20261116/run.py \
  --backend encoder --corpus /path/to/restored/precommitted \
  --bank /path/to/restored/bank.json --output /tmp/airi-new-encoder-run
```

The automatic workflow uses `cycle.py`, stages model data only if qualification,
FINAL, historical routes and old semantic retention all pass, then runs the full
assistant/frozen-training tests before a whitelist-only fast-forward publication.
An unchanged input fingerprint skips fitting and retains the existing verdict.
Future replay of this published corpus is an operational regression check, not
an external blinded modern-Transformer benchmark. Full future Actions artifacts
expire after14days; accepted/rejected compact verdicts persist in Git history.
