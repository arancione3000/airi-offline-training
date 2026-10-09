# Focused native learning diagnostic

Six hours of fresh train_step time. Same verified native champion 39731, native architecture, optimizer and RNG lineage; same 884 foundation examples and protected 92 heldout probes. Add a separate, explicitly in-sample 16-probe diagnostic over a/b/c/d copy and recall examples. This is never included in the reward or evidence of generalization.

Increase copy/recall plus diagnostic practice to 50% of updates. Initial LR 3e-5 with anchor KL 0.05. If at update 4096 memorization remains below 50%, switch the isolated student to LR 1e-4 and KL 0. When memorization reaches 75%, consolidate at LR 1e-5 with KL 0.1. Phases never oscillate. Save training accuracy and NLL separately from heldout accuracy.

Disable the previous two-assessment stagnation rollback for this isolated student, so each test can accumulate learning. Retain the verified source champion independently. Numerical nonfinite updates still abort. Original heldout selection, non-regression and native production gates remain unchanged; no live promotion. Diagnostic student checkpoints may be worse and are labelled offline. Best retained checkpoint is only updated through the unchanged heldout policy.

If in-sample accuracy rises without heldout improvement, the limitation is transfer/generalization. If in-sample accuracy remains zero under the stronger recipe, further data-only training is unsupported and the next investigation must examine gradients, optimizer history, tokenization and model internals. These are diagnostic hypotheses, not a guarantee of conversational ability.
