# Fresh review of the twelve-expert bank

This review freezes the public twelve-expert bank and all assistant Python
sources before native inference. Its fixed seed is `20261116230519`; that number
is a PRNG seed, not a measurement timestamp. The preregistration records actual
UTC creation time, input hashes, source hashes and the approved model identity.
No candidate program, query alias or neural weight is changed by this experiment.

There are24 fixed cases: one new value per each of12 learned task templates,
six explicitly authored paraphrases of the original six questions, and six
quoted-source cases on which the expert must abstain. Exact task semantics are
declared in `GOLD_OPERATIONS`; answers use independent arithmetic/string reference
operations. Compiler predictions never generate gold. This is a small diagnostic
sample, not a representative natural-language benchmark or independent broad
confirmation of the original study.

The first run preserves all original cases and its9/12 exact composed endpoint.
Inspection afterward found that its three arithmetic inputs exceed the compiler's
declared `MAX_INTEGER_ABS=1_000_000`. Those experts properly abstain, but their
native fallbacks answer incorrectly; the original system score is not relabeled
or replaced. A separately preregistered three-case supplement tests fresh arithmetic
values within the declared range. That is explicitly a post-review diagnostic,
not another blind endpoint and not a pooled12/12 headline.

For each case, actual native inference uses the same hash-verified revision4
checkpoint twice: without demonstrations and with the exact eight stored TRAIN
examples from that expert. The script checks that the entire demonstration prompt
fits the1024-token context and fails if it would truncate. Both calls use greedy
decoding with at most64 new tokens. The composed system is also evaluated on the
fresh messages; if its expert abstains, it receives the exact cached native-plain
output already measured for those messages. That caching avoids a third identical
greedy call and does not replace native inference with fabricated output.

Known-template and new-wording exact-answer scores are reported separately.
Negative cases score expert abstention only: no native/composed accuracy score
is invented for an unstructured abstention label. Poor native output after expert
abstention is retained in the per-row evidence. Every old expert's stored TRAIN
and qualification examples, plus989 old routing contracts, are checked separately.
Those989 routes are symbolic-routing preservation, not fresh native generation of
all historical benchmark answers. Native checkpoint file hashes are verified before
and after inference, and greedy calls must leave Torch RNG state unchanged.

The comparison isolates the useful effect and limitations of bounded learned
software modules around this particular small native model. It does not compare
against current strong pretrained Transformers, match training compute or prove
general neural/Transformer superiority. Question paraphrases intentionally expose
the exact-query boundary rather than hiding it inside an aggregate score.

## Restore the exact experiment

`evidence/MANIFEST.json` lists raw and compressed hashes for the original run.
The supplement has its own `evidence/supplement/MANIFEST.json`; `INDEX.json` binds
both manifests. The main restoration command below also restores the supplement
under its own subdirectory.
Artifacts use deterministic gzip compression; they contain synthetic examples,
native output text, the qualified bank and exact as-run source, without weights
or credentials. Restore them into a fresh directory:

```sh
python - <<'PY'
import gzip
from pathlib import Path

source=Path('scripts/research-training/modular-review-20261116/evidence')
target=Path('/tmp/airi-review-frozen')
if target.exists():
    raise SystemExit('choose an absent target')
for artifact in source.rglob('*.gz'):
    destination=target/artifact.relative_to(source).with_suffix('')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(gzip.decompress(artifact.read_bytes()))
PY
```

Use the same approved checkpoint and matching public model record with a Python
environment containing Torch and the project's `generalist_lm` package. From an
isolated checkout, restore `sources/*.py` to `scripts/assistant-runtime/` and the
archived stable contract to its corresponding `skills/` path before rerunning.
Use the frozen bank; a later larger public bank deliberately fails the twelve-
expert assertion instead of silently selecting a different endpoint.

```sh
PYTHONPATH=/path/to/native-source/computer python \
  scripts/research-training/modular-review-20261116/compare.py \
  --bank /tmp/airi-review-frozen/bank.json \
  --checkpoint /path/to/checkpoint-parent \
  --record /path/to/matching-approved-record.json \
  --output /tmp/airi-review-rerun
```

The output directory must be absent. Exact source/seed/cases, greedy outputs and
qualification outcomes should reproduce; durations and absolute path strings
are machine dependent. An operational rerun after observing the cases is not a
second blind experiment. Preserve the first report if changing any implementation.

To repeat the separate diagnostic without changing the original cases:

```sh
PYTHONPATH=/path/to/native-source/computer python \
  scripts/research-training/modular-review-20261116/supplement.py \
  --first-run /tmp/airi-review-frozen \
  --checkpoint /path/to/checkpoint-parent \
  --record /path/to/matching-approved-record.json \
  --output /tmp/airi-review-supplement-rerun
```
