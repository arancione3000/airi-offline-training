# Independent offline training

## Current experimental assistant

Run `python scripts/assistant-runtime/chat.py` for the new assistant entry point.
It includes a persistent procedural bank learned from examples, initially six experts. Use
`/learn FILE` or the trusted local lesson watcher to qualify and append more
experts without rewriting the approved neural weights. On precommitted tasks,
the learned bank answers96/96 unseen-value final cases; on a fixed24-case sample,
the same native model answers0/24 both without examples and with eight TRAIN
examples in context, versus24/24 with the learned bank. This is bounded program
induction with fixed wording and a seven-operation library, not evidence of
general Transformer superiority. See [growth results and limits](docs/MODULAR_GROWTH_RESULTS_20261116.md).

The [automatic update workflow](docs/AUTOMATIC_MODULAR_UPDATES_20261116.md) now learns
up to two additional procedural families per hourly cycle, publishes only after
regression checks, and audits without duplicate training when its12-family catalog
is complete. It uses exact reference labels and no paid model/API. Local private
banks preserve their additions when importing newly installed bundled experts.

The [latest review and repairs](docs/MODULAR_REVIEW_20261116.md) document explicit
supervisor renewal, two registry fixes, and a fresh comparison that exposes both
procedural gains and failure on new question wording.

It can recall explicit names, residences and topics from the current conversation,
with source spans, corrections and conservative abstention. Native fallback uses the
existing hash-verified approved model loader and requires the usual native runtime
dependencies and GitHub access. See [setup, examples and limits](scripts/assistant-runtime/README.md).

Measurements are separate: the approved **native model remains revision4,23/92**.
The **composed assistant scores27/92** in actual inference with the same weights and
memory, preserving all23 earlier successes. New explicit-memory tests score96/96
positive and0/68 unsupported false answers. This is software recall, not a claim that
the neural weights learned those cases. The certified897 TRAIN routes are preserved:
20 memory answers match their gold and877 calls inherit unchanged native behavior.

The [architecture plan](docs/ARCHITECTURE_IMPROVEMENT_PLAN_20261115.md) and
[results](docs/ARCHITECTURE_IMPROVEMENT_RESULTS_20261115.md) document both the useful
memory component and the rejected broad-curriculum neural candidate. The
[continuous service](docs/AUTONOMOUS_RECOVERY_20261115.md) retains its checkpoint,
integrity and adoption gates; experimental weights are not silently activated.

## Historical October runner

The original October instructions below describe a different, bounded cycle.

This manual workflow resumes the native 1024-context experiment from update1280,
using its original optimizer, RNG, data and reference. The CPU variant uses FP32
instead of BF16, with a new recipe fingerprint; only the verified original seed
is allowed to migrate. Subsequent checkpoints must match the FP32 recipe.
It stops after13107 updates, evaluates the original gates and50 frozen prompts,
and leaves conversation assessment pending. It never promotes weights or writes
`generalist-state`.

Each standard public GitHub runner trains for at most four hours, stopping at the
next16-update checkpoint. Model and optimizer are archived in a dedicated
prerelease; the SHA256 index is uploaded last and the release made visible only
after all uploads succeed. A successful chunk dispatches the next job. Failed
jobs do not chain; resume manually from the last committed release. Concurrency
prevents overlapping writers. The chain is capped at64 jobs.

Before first launch, transfer the two original1280 ZIP archives and their index
to the dedicated `airi-offline-seed-01280` release. The user explicitly authorized the checkpoint transfer and public release
publication on 2026-10-06. The initial release contains six verified parts and a
committed index. The runner reconstructs the original ZIP archives and verifies
every part plus both original SHA256 digests before loading tensors.

Run **AIRI offline background training** on `main` with
`checkpoint_release=airi-offline-seed-01280` and `continuation=0`.
The index must contain the original experiment fingerprint and both archive
SHA256 digests and sizes. The downloader rejects other release namespaces,
unsafe archive paths, corrupt files and mismatched lineage.

Disable the workflow or cancel the active run to stop the chain. Existing
committed checkpoints remain available. The runner modifies only its ephemeral
checkout with the verified recovery patch; normal repository training stays
unchanged. `run.py` and `session.py` derive from the original frozen experiment with explicit
FP32 precision and 16-step checkpoint cadence. `persist.py` remains unchanged.
All script hashes and the new recipe are verified before training.

The runner lives in a small dedicated repository because the canonical AIRI
repository exceeded its GitHub size quota. AIRI source and native reference are
checked out from their immutable canonical commits.

The first job uses a short time budget and stops at the next16-update snapshot
to verify remote persistence and continuation. Later jobs use four hours. Each
job also uploads an AIRI_PROGRESS_<run-id>.json status asset to its input
checkpoint release after its first16 updates and then every128 updates. Status
assets are separate from the immutable checkpoint index and model archives.
