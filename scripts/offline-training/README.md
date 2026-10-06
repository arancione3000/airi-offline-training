# Independent offline training

This manual workflow resumes the native 1024-context experiment from update1280,
using its original optimizer, RNG, data, reference and frozen recipe fingerprint.
It stops after13107 updates, evaluates the original gates and50 frozen prompts,
and leaves conversation assessment pending. It never promotes weights or writes
`generalist-state`.

Each standard public GitHub runner trains for at most four hours, stopping at the
next128-update checkpoint. Model and optimizer are archived in a dedicated
prerelease; the SHA256 index is uploaded last and the release made visible only
after all uploads succeed. A successful chunk dispatches the next job. Failed
jobs do not chain; resume manually from the last committed release. Concurrency
prevents overlapping writers. The chain is capped at64 jobs.

Before first launch, transfer the two original1280 ZIP archives and their index
to the dedicated `airi-offline-seed-01280` release. This is a required external
checkpoint transfer and is currently blocked pending explicit user approval.
No training has been launched by installing this workflow alone.

Run **AIRI offline background training** on `main` with
`checkpoint_release=airi-offline-seed-01280` and `continuation=0`.
The index must contain the original experiment fingerprint and both archive
SHA256 digests and sizes. The downloader rejects other release namespaces,
unsafe archive paths, corrupt files and mismatched lineage.

Disable the workflow or cancel the active run to stop the chain. Existing
committed checkpoints remain available. The runner modifies only its ephemeral
checkout with the verified recovery patch; normal repository training stays
unchanged. Frozen `run.py`, `session.py` and `persist.py` are byte-identical to the
saved experiment; `background.py` replaces only storage and bounded stopping.

The runner lives in a small dedicated repository because the canonical AIRI
repository exceeded its GitHub size quota. AIRI source and native reference are
checked out from their immutable canonical commits.
