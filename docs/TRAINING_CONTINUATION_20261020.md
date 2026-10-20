# Native training continuation — 20 October 2026

The repaired 19 October cycle completed successfully as a job, but did not improve the approved conversational model. Terminal checkpoint: `airi-offline-413500-39297128288-1-3`, fingerprint `ca314a40de4a37ae9c3f6b667bc3b589dbecfc5dd2ca93d43ff39d174434b22b`. Its SHA256-verified FINAL_RESULT is `7a3731539a454daa8306c29ed0076c344bc1462766624ebcb186fcd908c6b067` and AIRI_RUN_STATE is `fd247e43a4cf0a82defc6ca286759167856fad43e19b9ef39dc2198fad36390d`.

It performed 3,392 optimizer updates and 5,260.9148 optimizer seconds, with 48 failed development checks across six complete exploration trials and zero accepted model updates. The approved champion stays at revision1: legacy15/28, dialogue4/16, foundation4/16, total23/92. The final unseen compositional set is0/64. The last candidate's human-development English NLL improved from3.377343 to3.080249, while legacy answers fell to11 and dialogue answers to2; it was correctly rejected. Lower language loss alone is not conversational qualification.

The lesson bank grew from25 to64 entries, holding92 token positions and3 whole-answer training gains. These are training observations, not independently accepted skills. The complete bank and model lineage seed the newly authorized continuation; test/final references remain excluded from optimization.

## Retention mismatch corrected

Ordinary continual updates used the active profile's gold replay and champion-KL weights (16/16), while both archived replay slots and separate consolidation transactions still called `distill_step` with hardcoded0.5/1 retention and one same-language anchor. This is a verified objective mismatch and a plausible contributor to forgetting; it has not been established as the sole cause of stagnation.

Both transfer paths now pass the active profile and independently sampled certified Italian and English training memories. The donor's selected verified gold-token loss and probability distillation remain, with supervised retention and champion KL averaged across both memories. Continual transfers normalize acquisition/replay by1+retention_weight. Legacy callers retain their previous objective by default. A transfer still performs one clipped Adam update. Runtime telemetry records the applied retention weights and memory count so configuration can be checked against execution.

Data, native architecture, learning rates, promotion gates,512-update trial horizon and64-update evaluation cadence remain unchanged. The new experiment has a fresh explicit finite ledger, limited to21,600 optimizer seconds,48 development checks,12 complete rejected trials and32 jobs. No automatic live qualification is introduced.

## Validation

56 transfer tests and7 controller tests pass. The real native complementary-skill demo still learns skillB from a donor that lost skillA, while retainingA. A new native numerical test verifies the weighted bilingual loss against an independently constructed causal objective, correct supervision counts, one Adam update, and rejection of invalid retention weights without changing parameters or optimizer moments. Both runner paths are checked to forward the active profile.

The new frozen branch is `dialogue-bilingual-transfer-20261020`. The new seed is the verified terminal above. Native preflight and actual optimizer progress are recorded after launch. No new conversational improvement is claimed at setup.
