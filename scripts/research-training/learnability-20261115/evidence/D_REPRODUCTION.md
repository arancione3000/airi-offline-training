# D follow-up: evidence reproduction

This bundle records the single conditional full-native-with-retention arm.
It contains only D_ROWS, D_PREREGISTRATION, D_REPORT, CACHE_ENDPOINT_D as
byte-preserving deterministic gzip copies, a derived summary, and this note.
No model weights, optimizer states, credentials, or raw logs are included.
The original three-arm bundle provides the prerequisite observations and hashes.

The manifest records compressed and decompressed SHA-256 digests, source paths,
Python and zlib versions. gzip uses level 9, an empty original filename, and
mtime 0. Regeneration was checked byte for byte. The D preregistration pins
the original 103 code-file set plus protected_full_native.py and
audit_native_chat.py (105 total), the source artifacts, the exact original
64-unit trajectory, the 8 TRAIN rows, and the 24 fresh paired confirmation rows.

To independently rerun, first complete the original three-arm workflow and
independent A/B/C native audits in an independent output directory as described
in FIRST_THREE_ARMS_REPRODUCTION.md. Then run protected_full_native.py
--output <directory> --prepare-only, review its fixed design, and invoke
protected_full_native.py --output <directory> once. Run audit_native_chat.py
on the saved D_full_native_with_retention-candidate with D_ROWS.json to produce
CACHE_ENDPOINT_D.json. Use /workspace/airi-lab/venv/bin/python with
PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=/workspace/airi-lab/native-source/computer.
Torch was 2.14.1+cpu with two CPU threads; source/reference/archive paths must
match the pinned input contents. Production paths must not be modified.

These commands create a new trial/preregistration, rather than promising
identical report or preregistration bytes on a fresh checkout. Absolute paths,
filesystem insertion order and measured wall times can differ. Archive byte
identity is established separately by the manifest's decompressed hashes.

The summary checks 1/8 TRAIN raw answers including EOS, 1/12 familiar holdouts,
0/24 fresh confirmation, 19/92 old probes (four source successes lost), and
896/897 source TRAIN certificates retained. The old NLL gates were deliberately
not measured because exact-answer/certificate retention already failed under
the fixed cost-saving rule. Qualification and promotion remain false.
The gradient diagnostic measures one initial TRAIN batch with no optimizer
update, restores state, and records norms/cosines; it is not a guarantee about
later gradients or generalization. Root authors final conclusions separately.
