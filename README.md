# Independent offline training

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
