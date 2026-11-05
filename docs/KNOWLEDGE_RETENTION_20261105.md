# Persistent knowledge retention — 5 November 2026

The new subsystems preserve a durable ledger of certified TRAIN answers, constrain individual updates, and carry their state through native checkpoints. The controlled real-model trial retained all **518 certified TRAIN sequences**, including their final EOS, through three consecutive 64-update segments and an exact restart.

This is a retention and persistence result. It does **not** demonstrate a capability improvement over the approved offline model. The new candidate scores **22/92**, the old method **21/92**, and the approved revision2 **23/92**. Neither candidate is adopted. Context remains **0/32**, and the approved model remains unqualified for production.

## What changed

The old replay shortlist excluded context, copying and recall families. Rebuilding it after an adoption could silently omit answers the new model had forgotten. Its stored count and digest were insufficient to recover a lasting list of obligations. The existing 64-entry donor archive stores useful token distributions from candidates; its eviction policy does not make it a permanent knowledge ledger.

The opt-in `knowledge_retention` path adds three cooperating mechanisms:

- **Persistent certificates.** `retention_memory.py` records exact eligible TRAIN examples and full gold next-token correctness, including EOS, with their original provenance and champion digest. Context, copy and recall can enter the ledger. Every old entry is rechecked before refresh/adoption and remains an obligation across migration. A failed check cannot erase it. The pilot capacity is 1024; reaching capacity records skipped additions instead of evicting old certificates. Certification expands coverage of observed examples, not general knowledge or unseen formulations.
- **Protected updates.** `retention_guard.py` compares both the complete prompt/history distributions and the supervised answer against a frozen native champion. The pilot samples eight TRAIN sentinels per proposal. Previously correct gold tokens cannot flip; full-sequence KL and answer NLL are bounded. A row already degraded by an earlier unsampled update can improve over several steps. The full ledger must still pass before adoption. `retention_transaction.py` rolls back trainable weights, Adam moments, gradients and random states on rejection, then retries with genuinely smaller learning rates. Proposal, committed-update and physical-exposure counters are distinct.
- **Replay and checkpoint continuity.** Replay mixes rotating coverage, current measured degradation and seeded uniform sampling. The answer objective adds full-sequence champion KL, so question/history positions receive protection too. Materialized paired context rows require reconstructible TRAIN-source/value provenance; their registry and the certificate ledger are hash-committed with the native checkpoint. Same-recipe resume rejects missing or altered files. Normal updates, archived donor replay and final consolidation use the same transaction wrapper.

Consolidation also has its own persisted allowance. Early rollbacks can spend at most half of the 128-unit quota; the remaining half becomes available after exploration stops. This does not reset the search or its global time/update limits. The previous final salvage could have zero available updates merely because `search.stopped` had already been set.

Absent the opt-in configuration, the legacy sampling/objective path remains available. The completed November2 experiment branch `native-context-learning-20261102` is preserved at `d3f5039c35953625b059263d0be08f70adc74a01`.

## Fixed real-checkpoint comparison

Both arms start from the SHA-verified approved revision2 tensor digest `9e07639be3478a9e1fcba937fcc7cf8857429a2faf5af449ef34dce520b83274`. The fixed seed is 20261105. Each follows 192 predeclared acquisition units, measured at TRAIN-only boundaries 64, 128 and 192. The common acquisition/native-anchor trajectory hash is `5e45b6c84ad911a6053df66693202552fed34f3840c4f73648b9da0dc2128c25` and contains 9,899 acquisition target-token exposures per arm.

The native final attention block and output head train 74,304 parameters; the preceding eleven blocks and large feed-forward matrices remain frozen. Both arms use LR `3e-5`, supervised replay weight 8, answer champion KL weight 8, ancestral reference weight 0.25 and four replay examples per unit. The new arm additionally uses full-sequence KL weight 8, eight sentinels, KL limit 0.01, answer-NLL allowance 0.02 and retry factors `[1, 0.5, 0.25, 0.125]`.

The new initial ledger has 518 correct sequences: the 278 old certified examples plus 240 already-correct answers found by balanced probing. The latter are **coverage added before training**, not learned skills. Of 534 measured TRAIN examples, sixteen are initially incorrect; none becomes correct in the observed segment checks. Both arms retain every one of the initial 518 correct sequences at all three boundaries. The initial ledger includes 24 context examples and 20 copy/recall examples.

| Endpoint measurement | Approved baseline | Old replay | Persistent retention |
|---|---:|---:|---:|
| Independent exact responses | 23/92 | 21/92 | 22/92 |
| Context responses | 0/32 | 0/32 | 0/32 |
| Previously correct held-out indices lost | — | 0, 64 | 64 |
| New correct held-out responses | — | 0 | 0 |
| Initial complete TRAIN sequences retained | 518/518 | 518/518 | 518/518 |
| Native Italian development NLL | 1.206915 | 1.198340 | 1.200812 |
| Native English development NLL | 1.037827 | 1.025378 | 1.030185 |
| Human English development NLL | 3.330514 | 3.307070 | 3.319501 |
| Original adoption gate | Current approved model | Rejected | Rejected |

The new method loses one fewer held-out response than old replay in this trial, but still loses a response compared with the approved model. The changed replay pool, sampling and full-sequence objective form a combined treatment; this comparison does not isolate a causal benefit from the guard alone. All 192 new-arm proposals pass at scale1: real-model rejection/retry is not exercised by this trajectory. Rejection, exact rollback and capped smaller-step behavior are exercised separately by native checks.

Equal update counts do not mean equal compute or tokens. Old replay uses 34,649 supervised and 24,750 reference-token exposures. New retention uses 32,174 supervised, 22,275 answer-reference and 44,084 additional full-sequence reference-token exposures on its logical trajectory. The extra prompt KL, sentinels, snapshot/guard work and changed replay lengths add compute. The recorded logical unit times are approximately 68.00s old and 199.66s new; these local measurements are not a hardware-normalized throughput comparison.

No sealed-final forward passes or model adoption occur in this diagnostic. Source model files, frozen champion and ancestral reference remain unchanged. Both trained candidates export native logits exactly.

## Restart and exhaustive TRAIN audit

At the new arm's unit64 checkpoint, the next actual update matches after reload in every checked component: trainable weights, Adam, certificate JSON, transaction counters, both sampling RNGs, Torch RNG and update statistics.

The first harness assertion incorrectly captured expected Torch RNG **after** constructing the reload model, which itself consumes RNG. Every other equality already passed. The harness was corrected and resumed from the saved unit64 native model/Adam/JSON; it did not restart the trial or change its profile, acquisition trajectory or original development endpoint. The archived log preserves the failed assertion. The new arm has 192 logical trajectory updates, one additional successful resume-proof update and two discarded updates from the interrupted proof: **195 physical optimizer attempts** in total. Including these proof exposures gives 32,867 supervised, 22,509 answer-reference and 44,666 full-sequence reference tokens. The two discarded attempts' timings are explicitly unavailable; their token counts are reconstructed from the exactly reproduced unit65. The raw report distinguishes the first64 aggregate from the recovered per-update ledger for65–192.

After the fixed endpoint, an independent read-only audit compares the new candidate with its approved teacher on **all 518 initial TRAIN ledger rows**, covering 27,510 real next-token positions. It performs no optimizer updates and reads no evaluation labels:

- Every complete gold answer including EOS remains correct.
- Maximum full-sequence KL is **0.003905296**, mean per-row KL **0.000266781**; no row exceeds 0.01.
- Maximum answer-NLL increase is **0.000522681**; no row exceeds 0.02.
- **96 prompt-side next-token argmax positions change.** The distributions and full function are not identical, even though all certified answers survive.

This exhaustive endpoint measurement supports the bounded TRAIN-bank claim. Sampled update guards alone do not establish it, and neither measurement guarantees retention on unseen questions. The lost held-out answer shows why the separate independent adoption gate remains necessary.

## Continuous execution and verification status

The new finite pilot is defined by [its specification](../scripts/knowledge-retention/SPEC.json), fingerprint `d079288692141fae1f2aed2b2d55fcbe168c6e08e1a03e6a236e3aae7279c311`, and [GitHub workflow](../.github/workflows/airi-retention-background.yml). The new frozen branch is `native-knowledge-retention-20261105-v2`. It starts from verified terminal release `airi-offline-449276-41523612006-1-3`, whose weights equal approved revision2. It carries the same model identity and eligible donor lessons, explicitly starts fresh Adam once for the changed objective, and thereafter resumes Adam, ledger, TRAIN registry, adaptive retry scale and RNG exactly.

Self-continuation uses the same frozen runner commit and a hash-committed checkpoint. Limits are four jobs, 4,096 logical acquisition/consolidation units, 3,600 training seconds, twelve development decisions and three failed complete trials. There is no reset after a quality stop. A new successful version must keep all stored certificates and every previously correct independent development answer while satisfying the original improvement criteria. Final qualification gates remain separate. This is autonomous checkpoint continuation within an explicit budget; it is not an unlimited guarantee of future learning. The disabled hourly ChatGPT controller is not re-enabled.

**The autonomous pilot has been activated** in [run42045545781](https://github.com/arancione3000/airi-offline-training/actions/runs/42045545781), after the actual seed preflight and both remote native CI runs passed. Its frozen commit is `72bda54e942e5ec36ded31ca0c3ad43fa87f5fe4`; [PR4](https://github.com/arancione3000/airi-offline-training/pull/4) merged the identical tree to main. The [first512-unit checkpoint](https://github.com/arancione3000/airi-offline-training/releases/tag/airi-offline-449788-42045545781-1-2) and [automatic next job42047389197](https://github.com/arancione3000/airi-offline-training/actions/runs/42047389197) are now verified; the next job has trained through768 logical units without resetting its state. Activation is not evidence of a new accepted model or benchmark gain.

Observed local verification: the archived root transfer log records **95 tests passing**, including native causal batching when the patched native source is on `PYTHONPATH`. The later full existing-CI log records **192 tests passing across fourteen suites**, with native dependencies available; root separately observed **ten new knowledge-continuation tests passing**. Native memory, transaction, eight guard tests, session and capped retry checks pass. The retention-specific preflight helper also passes on its tiny native fixture: 32 proposals, exact next-update resume, complete ledger retention and lower TRAIN-task NLL, with no held-out evaluation. Fixture loss reduction is optimizer verification, not a real-model benchmark result. The separate real-checkpoint preflight is now verified below.

## Auditable evidence

[Compact derived summary](evidence/KNOWLEDGE_RETENTION_SUMMARY_20261105.json), [SHA256 provenance manifest](evidence/KNOWLEDGE_RETENTION_PROVENANCE_20261105.json), [exact raw trial](evidence/KNOWLEDGE_RETENTION_retention-system-trial.json.gz), [original certified ledger](evidence/KNOWLEDGE_RETENTION_retention-system-trial-initial-bank.json.gz), [trial source](evidence/KNOWLEDGE_RETENTION_retention-system-trial.py.gz), [interrupted-and-recovered log](evidence/KNOWLEDGE_RETENTION_retention-system-trial.log.gz), [read-only audit source](evidence/KNOWLEDGE_RETENTION_retention-system-trial-fullkl.py.gz) and the actual five training-module snapshots are packaged without weights or optimizer state.

Run the [evidence validator](evidence/KNOWLEDGE_RETENTION_validate.py) with:

```sh
python docs/evidence/KNOWLEDGE_RETENTION_validate.py
```

It verifies fourteen exact archives and their decompressed SHA256 hashes, parses the source snapshots, recomputes the summary from raw responses/certificates, checks the persistent ledger membership, validates the exhaustive audit totals and reconciles physical resume-proof accounting. It requires no Torch installation and performs no training or model inference.

Remote verification update: [PR3](https://github.com/arancione3000/airi-offline-training/pull/3) merged as `1205677acb30f0c6e087a5972f80506fcb772778` after [native CI42042943310](https://github.com/arancione3000/airi-offline-training/actions/runs/42042943310) and the standard runner checks passed. Both new workflows use Python3.12, matching the local environment and pinned NumPy2.5.4; the initial Python3.11 dependency failure involved no training.

The twelve-decision limit counts exploration rounds. Each round can assess its candidate and up to two fixed interpolations; the globally bounded128-unit consolidation budget permits at most two further skill assessments. These remain independent development checks with no optimizer access to their labels. The initial seed baseline is an additional assessment; “twelve rounds” must not be read as twelve total development forward passes.

The3600-second budget counts replay sampling, sentinel construction, optimizer attempts, step guards and observation updates. Acquisition selection/encoding outside that section, full-ledger certification, development/final evaluation and checkpoint I/O add wall-clock time. Final native/sealed checks are separate from the at-most39 controller assessments, including the initial baseline.

The [first full-checkpoint preflight42043283238](https://github.com/arancione3000/airi-offline-training/actions/runs/42043283238) then stopped before checkpoint loading because its workflow named `check_context_acquisition.py` instead of the existing `test_context_acquisition.py`. The native acquisition suite is now invoked through unittest discovery after Torch installation. The first frozen branch remains unchanged at1205677; a separate `-v2` branch/fingerprint carries this orchestration repair. No checkpoint optimizer updates or model adoption occurred in that failed preflight. That failed branch remains frozen; the corrected v2 result follows.

The corrected [real-checkpoint preflight42044360271](https://github.com/arancione3000/airi-offline-training/actions/runs/42044360271) **passed**:32 real proposal units,32 physical attempts,32 commits, zero rejected attempts and all518 complete TRAIN certificates retained. Native export and exact next-unit model/gradients/Adam/ledger/registry/adaptation/RNG resume pass; champion and ancestral reference remain unchanged. TRAIN-task NLL0.000688385393→0.000684175815 (-0.61%) concerns one already-nearly-solved TRAIN example, with zero held-out evaluation or adoption. The two extra validation branches are separately reported. [PR CI42044358719](https://github.com/arancione3000/airi-offline-training/actions/runs/42044358719) and [push CI42044347754](https://github.com/arancione3000/airi-offline-training/actions/runs/42044347754) passed on the same frozen v2 commit.

[Exact remote results and hashed logs](evidence/KNOWLEDGE_RETENTION_REMOTE_20261105.json) separate the tiny native fixture, full-checkpoint preflight and activated training run. No32-unit preflight weights were published. The training timer has unit granularity and can overrun by the final in-progress unit; workflow timeouts and the job limit bound additional wall-clock work.

The first actual job completed512 logical units with518 physical attempts:496 committed updates and22 rejected attempts,203.65 timed training seconds. Its initial migration checkpoint stored all518 certificates plus an empty registry; its final candidate checkpoint preserves every original certificate, records518/518 correct current full-bank observations and stores252 mechanically reconstructed TRAIN rows. The OPT archive bytes, raw manifest, ledger, registry and committed run-state hashes were verified. These252 registry rows are auditable training material, not252 newly acquired skills.

**Both256- and512-unit adoption checks rejected the candidate:**22/92 exact held-out responses versus23/92 for the approved champion, one previously correct answer lost and zero gained. Both fixed interpolations also fail. Candidate development NLL improves (Italian1.175963, English1.002614, human English3.309548) but cannot override that regression. Revision2 remains approved, context0/32, and no new adoption or live promotion is claimed. The bounded candidate trial continues rather than being mistaken for a successful model.

The next job was dispatched by the pinned controller, using exactly candidate release449788 with continuation1 and the same frozen commit/fingerprint. Its actual progress asset reaches step450044/768 logical units,774 physical attempts (741 committed,33 rejected) and347.036439 cumulative training seconds. Prior counters, two evaluation rounds, trial start and adaptive scale1/64 persist; no assistant manually started that second job. This is observed automatic recovery and training, not merely a prepared dispatch configuration. The [remote evidence snapshot](evidence/KNOWLEDGE_RETENTION_REMOTE_20261105.json) includes raw checkpoint JSON, derived92-response assessments bound to their source resume hash, the first-job log and exact next-job telemetry. Later outcomes require a later snapshot.
