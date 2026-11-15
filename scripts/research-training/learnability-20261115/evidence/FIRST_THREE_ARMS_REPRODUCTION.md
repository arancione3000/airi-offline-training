# First three arms: evidence reproduction

This bundle contains byte-preserving deterministic gzip copies of the completed
first three-arm experiment and an independently derived numerical summary.
It contains no model weights or optimizer states. Fourth-arm artifacts are outside
this bundle. Root writes the final experiment conclusions separately.

The manifest records both compressed and decompressed SHA-256 digests. gzip uses
compression level 9, an empty original filename, and mtime 0. Decompressing a file
must reproduce its recorded raw digest; regenerating it with the recorded zlib
version and these options reproduces the gzip digest.

The Python executable was `/workspace/airi-lab/venv/bin/python`; the native source
was `/workspace/airi-lab/native-source/computer` on `PYTHONPATH`; Torch was
`2.14.1+cpu`. The experiment used two Torch CPU threads and seed `20261115`.
The authoritative seed, optimizer, acquisition objective, exact trajectory,
source artifact hashes, and imported transfer/native source hashes are recorded
in `PREREGISTRATION.json.gz`. The native audit script digest is in every cache
audit; helper/harness digests are also recorded in this bundle manifest.

The original experiment is deliberately single-run guarded. To independently
reproduce it, choose a new empty output directory, run
`controlled_learnability.py --output <directory> --prepare-only`, run
`audit_native_chat.py --checkpoint /workspace/airi-lab/rev4/checkpoint/model
--rows <directory>/ROWS.json --output <directory>/CACHE_PREFLIGHT.json`, and then
run `controlled_learnability.py --output <directory>` once. Set
`PYTHONDONTWRITEBYTECODE=1` and the native `PYTHONPATH` for every command.
The locked source/reference/archive paths in the harness must exist and match
the recorded hashes. Runtime or production paths must not be modified.

These commands create a new trial and a new preregistration. They do not promise
byte-identical rerun reports or an identical preregistration hash on a fresh
checkout: the harness records absolute paths and filesystem glob order, and
reports include measured wall times. Compare the pinned source content and
declared trajectory before comparing numerical observations. Exact archive
byte identity is established by the manifest's decompressed hashes, separately
from reproducing an experiment's observations.

The original harness hashes transfer/native `*.py` globs plus two explicit
research files, for 103 files; it does not glob this research directory. Adding
the separate D script does not change that original 103-file hash set. D's
105-file set explicitly adds `protected_full_native.py` and
`audit_native_chat.py`. A staging wrapper is therefore unnecessary for the
original code-file selection. All 103 currently selected paths and contents
were checked against the original preregistration and match.

The standalone audit is repeated for each saved candidate with the same TRAIN
rows. The existing `check_prefix_tuning.py` and `check_continual_native.py`
passed in this environment; they are supplemental path checks, not evidence of
general conversation ability. Original run.log is omitted because REPORT contains
all logical updates, physical attempt statistics, endpoints, and certifications.

`DATA_INVENTORY_20261115.json.gz` is attributed to the root-authored inventory
helper. Its counts describe available archived pools and overlap limitations;
they must not be relabeled as unique lifetime pretraining tokens.
