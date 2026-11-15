# Native path audit, 2026-11-15

The bounded audit found no text, tokenization, target-mask, causal, KV-cache,
native-chat, RNG, or checkpoint export defect on the declared eight TRAIN rows.
This is evidence for the tested native path, not a proof about every input.
No runtime source, production weights, or deployment was changed.

## Evidence

The standalone `audit_native_chat.py` was run with the native source on
`PYTHONPATH`, `/workspace/airi-lab/venv/bin/python`, Torch `2.14.1+cpu`, two CPU
threads, and bytecode writes disabled. Its reports are under
`/workspace/airi-lab/learnability-20261115/`:

| Report | Whole gold + EOS certificate | Native raw answer + EOS exact | Path checks |
|---|---:|---:|---|
| `CACHE_PREFLIGHT.json` | 0/8 | 0/8 | PASS |
| `CACHE_ENDPOINT_A.json` | 0/8 | 0/8 | PASS |
| `CACHE_ENDPOINT_B.json` | 0/8 | 0/8 | PASS |
| `CACHE_ENDPOINT_C.json` | 8/8 | 8/8 | PASS |
| `CACHE_ENDPOINT_D.json` | 1/8 | 1/8 | PASS |

For every row and endpoint:

- SFT prompt IDs equal native chat serialization; labels mask exactly the prompt
  and supervise every answer token including EOS, with the expected causal shift.
- Changing the final input token changes no preceding logit (maximum difference 0).
- Native cached and uncached greedy token IDs agree, including EOS; native chat
  returns exactly the decoded cached generation. Greedy generation preserves
  Torch RNG state.
- A complete teacher-forced gold certificate agrees with native greedy target
  IDs and EOS. Source checkpoint files remain hash-identical.

No newly authored held-out row was forwarded by the standalone audit. Each
report records the script, rows artifact, and checkpoint hashes. The source
checkpoint's eight rows contain 32–39 tokens, so these checks exercise complete
short histories without context truncation.

The existing `check_prefix_tuning.py` and `check_continual_native.py` also passed.
They cover actual native attention gradients, frozen prefix/FFN tensors, cache
bounds and invalidation, export and Adam resume, single optimizer updates,
rollback, nonfinite failure, and whole-answer causal verification. No broad test
suite was repeated.

## Interpretation of the three controlled arms

All arms completed the same 64 paired acquisition units at learning rate
`3e-5`, starting from the same approved weights and fresh AdamW. The bare arms
retain the same EOS weighting and repetition auxiliary as the protected arm.

| Arm | Trainable parameters | TRAIN exact + EOS | Raw TRAIN CE | Old 92 score | Source TRAIN ledger retained |
|---|---:|---:|---:|---:|---:|
| A: current scope with retention | 74,304 | 0/8 | 5.5443 | 22/92 | 897/897 |
| B: current scope, acquisition only | 74,304 | 0/8 | 5.3756 | 21/92 | 896/897 |
| C: all native weights, acquisition only | 50,164,416 | 8/8 | 0.0294 | 11/92 | 567/897 |

Training update/guard time was 63.17s, 1.35s, and 23.91s respectively; no arm hit
its cap. A committed 63 updates after 68 physical proposals. B and C committed
64 each. This compares equal acquisition exposure, not equal FLOPs or wall time.

C is a positive learnability control: the same initially wrong examples are
learnable by the actual native model under the declared optimizer and budget.
B's failure shows that removing the combined retention package alone does not
resolve the current scope's acquisition failure in this trial. The broader
scope learns much faster, but this does not identify which frozen layers or
submodules are necessary, or prove that the smaller scope can never learn.

C's held-out gains are 2/4 on known values with new wording, 0/4 on unseen values
with known wording, and 0/4 when both change. The source scored 0/12. These are
limited wording-transfer results after memorizing TRAIN values; they do not
establish a general context-copying skill. C also loses 15 of the source's 23
correct old probes and 330 of 897 TRAIN certificates, so it fails retention.
The TRAIN ledger and old probe gates measure different examples; A preserving
897 ledger rows does not erase its regression from 23 to 22 old probe answers.

The separate D follow-up performed the justified single change: all native
parameters with A's original retention package and exact 64-unit trajectory.
The independent D native audit passed and confirmed 1/8 TRAIN answers including
EOS. D scored 19/92 on old probes (four source successes lost), preserved 896/897
TRAIN certificates, and scored 0/24 on fresh paired confirmation rows. Its old
NLL gates were not run after the exact-answer and certificate gates failed.
It therefore failed retention and confirmation; broader scope with the same
retention package did not produce a qualified improvement in this bounded test.

Attribution concerns the entire removed retention package (replay, KL,
normalization and transaction guards), not an isolated loss term. The read-only
first-batch gradient check found near-parallel acquisition and combined
directions (cosine 0.999744 full native, 0.996973 current scope), so it does not
support initial gradient cancellation. No arm is production-qualified.
