# Autonomous modular update cycles

The assistant now has an automatic procedural curriculum alongside the trusted
lesson queue. Its first local cycle grew the private bank from6 to8 experts:
integer subtraction and order-contact recall. Both families passed40/40 audit
cases, all160 stored expert examples and989 historical routing contracts.
Native weights are unchanged.

## What runs without this chat

`.github/workflows/airi-modular-auto-update.yml` runs hourly at minute23, subject
to GitHub scheduling delays, and accepts manual dispatch. Each run checks out
current main and learns at most two missing families from a fixed12-family catalog.
Reference arithmetic/string operations supply exact labels; no paid model/API,
invented ordinary-chat labels or open-ended code generation is used.

Training and qualification run in a temporary registry. New-value audits cover
all acquired catalog families with a fresh recorded seed. Every old expert's
stored examples and989 historical routes must also pass. The full assistant suite
and frozen-training tests run before candidate publication. Only
`skills/approved-bank.json` and `skills/auto-progress.json` may be committed.
A fast-forward push rejects concurrent main changes. A failed run leaves the
public bank intact; later schedules retry from committed state. Trusted main is
the only workflow source. Model pointers, neural weights, frozen recipe modules
and source code are not automatically modified.

Full inputs, preregistration and per-row evidence are uploaded as an Actions
artifact retained14days. Compact committed progress binds input state, seed,
preregistration, output state and full report hashes. Git history retains accepted
bank revisions and summaries. Inspect the publish step: a staged report alone
does not prove that the push succeeded. Raw evidence is not retained forever.

The12 authored families use the existing seven-operation DSL: subtraction,
integer totals, identifiers, contact recall and text transformations. Alongside
the original six, the catalog reaches18 experts, below32 capacity. Once installed,
cycles run fresh audits without retraining, new bank commits or claims of further
improvement. More procedures need a reviewed catalog/compiler extension. This is
bounded task expansion, not unlimited self-improvement or general neural learning.

## Local operation and propagation

The same automatic process was started locally with an hourly interval:

```sh
python scripts/assistant-runtime/auto_update.py --watch --interval 3600 --limit 2
```

It uses the private chat bank by default. `--bank PATH` selects an independent
bank; `--report-dir DIR --progress FILE` saves local audit records. Failed cycles
are logged and retried at the next interval; only qualified snapshots apply.
The local process requires a living workspace. The separate GitHub schedule
persists independently of the chat/workspace while Actions remains enabled.

Existing private banks now reconcile missing installed bundled experts through
normal qualification and regression gates, preserving private skills and order.
All imports stage together and commit once, checking concurrent updates. The
default trusted-queue watcher also notices an updated installed bundle while
alive. Explicit bank paths remain independent. An installed checkout does not
fetch remote source automatically: obtain the updated checkout for GitHub bundles,
or run the local automatic curriculum.

The queue watcher also survives more than32 pending files, processes32 unseen
files per poll, avoids repeatedly logging unchanged malformed files, retries
transient reconciliation conflicts, and reports recovery. It never evicts
knowledge to make space. Check the actual revision with:

```sh
python scripts/assistant-runtime/learn.py --status
```

These checks supplement the [previous modular study](MODULAR_GROWTH_RESULTS_20261116.md).
Repeated seeded audits are operational monitoring, not new blind scientific
confirmation or evidence of general Transformer superiority.
