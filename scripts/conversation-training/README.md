# Autonomous conversational curriculum search

This explicitly new offline experiment starts from the SHA-verified native
17715 release. Model, tokenizer and lineage stay native. AdamW moments are
intentionally reset at each pilot because its objective and learning rate change.
CPU FP32, no pretrained models, paid compute, live promotion or self-labeling.

Existing data includes visibly mismatched multi-turn answers. Training uses 50%
short auditable authored examples, 37.5% structurally filtered single-turn human
dialogues and 12.5% native replay. Filtering cannot certify semantic quality.
Italian receives about two-thirds of dialogue updates. Reference KL remains,
with separately declared coefficients and higher learning rates for three pilots.

Authored examples cover short conversation, arithmetic and two-turn name recall.
Validation holds out prompt wording, operands and names. Exact known original
development/test texts and frozen probe prompts are excluded. No correctness
claim is based solely on NLL or substring matching. Historical seed exposure to
original heldouts remains unknown; these are not certified virgin benchmarks.

Each pilot runs 1024 real updates independently from the same seed. Only after
all three, choose a candidate with >=2 additional exact narrow-task responses,
lower answer NLL in both languages and bounded repetition. Preserve the complete
search ledger and raw outputs in releases. If no pilot qualifies, stop for a new
diagnosis; never repeat the failed recipe indefinitely.

The winner can receive 8192 further updates, assessed every 512. Stop after three
rounds without validated gain, retaining the best offline release. Final original
native retention/degeneration gates remain unchanged and are reported separately;
curriculum success is not evidence of general conversation or live qualification.
Final original 50 prompts are generated only at the end, for manual review.

Jobs stop within a 3-hour chunk and publish model/optimizer/RNG/hash index before
dispatching the next action. Shared concurrency prevents overlapping old runners.
At most 32 jobs; failure leaves the previous committed release. Code ref is pinned
for self-dispatch. GitHub Actions persists independently of the chat.

Technical recovery v2 resumes only the exact SHA-indexed trial-0 winner at step
18739 after a transient GitHub HTTP 500 interrupted a non-critical progress upload.
The recipe and optimizer/RNG are preserved. Progress telemetry now retries and may
be skipped after four failures; checkpoint and final-report uploads remain fatal.
The three-pilot search ledger crosses this technical boundary only from its exact
release and SHA-256; its old fingerprint is verified before recording the new one.
