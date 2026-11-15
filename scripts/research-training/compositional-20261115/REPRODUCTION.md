# Fixed native compositional curriculum, 2026-11-15

This is one isolated full-model learning experiment. The approved rev4 source
remains immutable and no result activates, deploys, or promotes the candidate.
This measures native neural generation; it does not measure the separate
explicit memory software being developed elsewhere.

The harness is `native_compositional.py`. Runtime artifacts live at
`/workspace/airi-lab/compositional-20261115/`. Run environment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workspace/airi-lab/native-source/computer \
/workspace/airi-lab/venv/bin/python -u \
scripts/research-training/compositional-20261115/native_compositional.py
```

The run is intentionally guarded against repetition after `TRAINING_STARTED`.
The checked-in code documents the exact experiment; the completed run must not
be silently repeated to select another seed, learning rate, or checkpoint.

## Committed design

The approved source is `/workspace/airi-lab/rev4/checkpoint/model`, with native
tensor digest
`57bdb933b82bd8807af5c8b6b545c27ee9114ee4f4c59022434d39b6bf771c79`.
All 50,164,416 parameters train through the unmodified native model. There is
no feature cache, reference teacher, replay, KL, retention guard, adapter, or
external model in the optimizer path.

TRAIN has 512 rows, 256 different values, and 256 same-question counterfactual
pairs. Each value is shared across Italian and English contexts. Four task
families and four literal-length buckets have 128 rows each. Values range from
two or three characters through two-word literals, with all letter initials
and digits represented. Histories contain zero to two filler exchanges, and
the source occurs both before and after fillers. Some forms contain an ignored
literal or explicitly replace an old literal. Gold is exactly the substring
captured from the designated authored context field; every row stores and
checks its source message, regular expression, and character span.

Fresh development has 32 rows: 8 new-value/familiar-template, 8 familiar-value/
new-template, and 16 both-new. Final has 64 rows: 16, 16, and 32 respectively.
Final is balanced by language, task, and length bucket. Development is balanced
by language but its first two conditions cover only name and location tasks.
For final familiar-value/new-template rows, the stored `length_bucket` is the
scheduled case bucket rather than the reused literal's original TRAIN bucket.
The true literal bucket is recoverable by matching its TRAIN value; true totals
remain balanced. Primary results use condition scores, with no bucket scores.
Development and final novel template sets and novel literals are separate.
Familiar-template conditions intentionally reuse TRAIN wording, and familiar-
value conditions intentionally reuse TRAIN values. Complete prompts are
disjoint across splits and from archived original data. Archived output values
are used only to exclude collisions, never to supply fresh positive labels.

Native serialization and complete SFT masking are checked on every row with
context limit 1024 and `require_complete=True`; no history or gold is truncated.
Observed complete sequence lengths are TRAIN 33–145, development 31–137, final
31–140 tokens. Only the final answer and EOS are supervised.

The precommitted trajectory has four shuffled epochs, 128 updates per epoch,
with two complete counterfactual pairs in each four-row batch. It permits at
most 512 updates and at most 600 seconds of update time, checked between
indivisible updates. Initialization and read-only endpoint evaluation are
separate. Collation, mode changes, gradient clearing, forward, native loss,
backward, clipping, and optimizer step are inside the timed update. If the cap
is reached, the completed fixed prefix is reported with no extension.

Optimization is fresh AdamW, learning rate `3e-5`, weight decay `.01`, clipping
norm `1`, native EOS weight `1.2`, and native repetition unlikelihood weight
`.05`. There is no tuning or intermediate evaluation feedback. Progress logs
contain only the current TRAIN batch objective.

All TRAIN rows and both fresh evaluation sets are evaluated only at the fixed
endpoint. Frozen-source development/final forwards also first happen after
training. Reports distinguish whole-answer teacher-forced certificates from
native greedy raw token-ID equality including EOS, decoded text equality, and
same-question pair success. Raw generated IDs and gold IDs are retained for
each measured row. Native greedy generation uses a fixed 40-token budget for
fresh rows, and 64 for the original 92 probes.

An isolated candidate expert requires at least 80% raw exact with EOS on final
new values/familiar templates and at least 60% on final both-new examples.
These are diagnostic thresholds, not production qualification. Full 897-row
certification runs only if both expert thresholds and all original 23 correct
probe answers survive. No production acceptance is claimed without the full
original retention and NLL gate package. No activation is authorized here.

## Transparent pre-update repair

The first launch found that an existing two-example padding helper rejects the
approved four-row batch. This happened before any fresh model forward,
backward, or optimizer update. The zero-update start, original code,
preregistration, approval, and log are preserved under
`aborted-preupdate-start/`; `PREUPDATE_REPAIR.json` records the repair. Generic
native PAD/ignored-label collation replaced that helper. Every committed batch
then passed a pre-update shape and exact source-ID/label audit. Data, objective,
seed, dose, cap, and endpoints did not change. Root approved this repair.

The actual preregistration SHA256 is
`0902f42d2f72832215815279e67d2fa5aaeee018b0493b3663301d19d783c4c9`.
ROWS SHA256 is
`bf6b3dd81a0302fad4d37d3a8de5745c43700b0023c17827c209810fe5bbc2f9`.
The native source, approved checkpoint files, existing transfer-training code,
and harness are hashed before and after the actual run. The candidate export
is reloaded and checked for bit-exact native logits and matching greedy raw
IDs/EOS on six predeclared TRAIN examples. Immutable source hashes and exact
export checks must pass before the run can be marked complete.
