# Native training continuation — 20 October 2026

The repaired 19 October cycle completed successfully as a job, but did not improve the approved conversational model. Terminal checkpoint: `airi-offline-413500-39297128288-1-3`, fingerprint `ca314a40de4a37ae9c3f6b667bc3b589dbecfc5dd2ca93d43ff39d174434b22b`. Its SHA256-verified FINAL_RESULT is `7a3731539a454daa8306c29ed0076c344bc1462766624ebcb186fcd908c6b067` and AIRI_RUN_STATE is `fd247e43a4cf0a82defc6ca286759167856fad43e19b9ef39dc2198fad36390d`.

It performed 3,392 optimizer updates and 5,260.9148 optimizer seconds, with 48 failed development checks across six complete exploration trials and zero accepted model updates. The approved champion stays at revision 1: legacy 15/28, dialogue 4/16, foundation 4/16, total 23/92. The final unseen compositional set is 0/64. The last candidate's human-development English NLL improved from 3.377343 to 3.080249, while legacy answers fell to 11 and dialogue answers to 2; it was correctly rejected. Lower language loss alone is not conversational qualification.

The lesson bank grew from 25 to 64 entries, holding 92 token positions and 3 whole-answer training gains. These are training observations, not independently accepted skills. The complete bank and model lineage seed the newly authorized continuation; test/final references remain excluded from optimization.

## Retention mismatch corrected

Ordinary continual updates used the active profile's gold replay and champion-KL weights (16 / 16), while both archived replay slots and separate consolidation transactions still called `distill_step` with hardcoded 0.5 / 1 retention and one same-language anchor. This is a verified objective mismatch and a plausible contributor to forgetting; it has not been established as the sole cause of stagnation.

Both transfer paths now pass the active profile and independently sampled certified Italian and English training memories. The donor's selected verified gold-token loss and probability distillation remain, with supervised retention and champion KL averaged across both memories. Continual transfers normalize acquisition/replay by 1 + retention_weight. Legacy callers retain their previous objective by default. A transfer still performs one clipped Adam update. Runtime telemetry records the applied retention weights and memory count so configuration can be checked against execution.

Data, native architecture, learning rates, promotion gates, 512-update trial horizon and 64-update evaluation cadence remain unchanged. The new experiment has a fresh explicit finite ledger, limited to 21,600 optimizer seconds, 48 development checks, 12 complete rejected trials and 32 jobs. No automatic live qualification is introduced.

## Validation

56 transfer tests and 7 controller tests pass. The real native complementary-skill demo still learns skill B from a donor that lost skill A, while retaining A. A new native numerical test verifies the weighted bilingual loss against an independently constructed causal objective, correct supervision counts, one Adam update, and rejection of invalid retention weights without changing parameters or optimizer moments. Both runner paths are checked to forward the active profile.

The new frozen branch is `dialogue-bilingual-transfer-20261020`. The new seed is the verified terminal above. Native preflight and actual optimizer progress are recorded after launch. No new conversational improvement is claimed at setup.

## Verified launch

Exact training commit: `5e2f598829477b6ddec267f7c2f3609e6f063a41`, new fingerprint `f44900e6a319abd44305b4f97ffbb6823202ec61426d1bc2d07f02892e7b65cf`. Branch CI [39394297408](https://github.com/arancione3000/airi-offline-training/actions/runs/39394297408), native preflight [39394299491](https://github.com/arancione3000/airi-offline-training/actions/runs/39394299491) and main CI [39395098579](https://github.com/arancione3000/airi-offline-training/actions/runs/39395098579) passed. The preflight includes the new weighted bilingual causal-objective check. Its separate ordinary-objective training demonstration is an integrity check and does not qualify the model for conversation.

Training [39395100754](https://github.com/arancione3000/airi-offline-training/actions/runs/39395100754) was launched from the terminal checkpoint on 20 October. The controller was re-enabled for the new frozen branch and manual controller run [39395129391](https://github.com/arancione3000/airi-offline-training/actions/runs/39395129391) passed with `active_job_retained`, avoiding duplicate training jobs. No new model adoption has been claimed at launch.

## First verified optimizer progress

The initial durable checkpoint is `airi-offline-413500-39395100754-1-1`. The subsequent `AIRI_PROGRESS_39395100754.json` was downloaded and verified against SHA256 `58db7f04304403968e56c3f88e4e0563bc38815de74e0b97579e22dc12e957ce`. It reports step 413516, 16 new optimizer updates, 26.647852 optimizer seconds, all 64 lessons and 92 archived token positions preserved, and two archived replay updates. Runtime `last_transfer_retention` is exactly gold weight 16, champion KL weight 16, two retention examples. This confirms that the profile now reaches the actual archived optimizer path. There have been zero new development checks or adoptions at this initial telemetry point; conversational improvement remains unverified.
