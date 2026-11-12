# Rejected final-FFN-output experiment, 12 November 2026

One fixed comparison, seed20261112,256 logical acquisition units per arm; no adaptive extension or mid-run development selection. The original harness `../ffn_ablation.py` is preserved byte-for-byte. `prefix_tuning.py` and `check_prefix_plasticity.py` here are research-only; the production implementation was restored byte-for-byte. The wrapper stages source copies and explicitly overlays this archived implementation in a temporary directory.

| Result | Approved source | A: final FFN frozen | B: final FFN output trainable |
|---|---:|---:|---:|
| Native trainable parameters |74304|74304|1450560|
| Original92 exact answers |23|22|21|
| Context exact answers |0/32|0/32|0/32|
| Newly correct / lost original answers |—|0 /1|0 /2|
| Complete gold/EOS retention certificates |897/897|897/897|896/897|
| Commits / physical proposals |—|252/262|243/262|
| Guarded training seconds |—|168.58|162.20|

Both candidates fail the development retention/gain gate; B also fails the complete897-answer retention gate. Neither was adopted, published as a model, or activated. Native checkpoint export was bit-identical for both arms. Source checkpoint and all recorded training sources stayed unchanged throughout the run. Lower NLL alone did not produce complete-answer gains. Sampled8-row guards cannot guarantee all897 protected answers: B lost one outside the committed local checks. Both arms reached LR scale1/64. This rejects this scope/recipe combination, not every possible FFN, LR or data design. The existing development set is familiar from earlier experiments; there was no sealed-test forward or new general-conversation evidence.

The paired trajectory uses the current16-slot schedule, diverse matched TRAIN context pairs, ancestral native anchors, persistent897-row ledger, gold replay8 / champion KL8 / prompt KL8 / native.25 and fresh AdamW3e-5 in both arms. Acquisition, anchor, memory and sentinel IDs were pre-materialized identically once. Unlike production, per-arm degradation observations cannot change this fixed sampler. Physical attempts and committed exposures differ because the same guards reject different proposals. Training seconds include guards/retries; preparation, baseline, endpoint and export checks are separate. This is not an equal-FLOPs or throughput comparison. Token fields in `summary.json` label supervised, answer-reference and full-prompt-reference positions separately; they overlap and must not be summed as unique corpus tokens.

Raw report is deterministic gzip (mtime0), with all92 raw generations, full897 certificates per endpoint, trajectory IDs, attempts, guard observations and losses. It adds no human training text. `summary.json` is a readable selection; full local candidate checkpoints are intentionally excluded from Git.

Original run inputs:

- Approved revision4 native tensor-state SHA256: `57bdb933b82bd8807af5c8b6b545c27ee9114ee4f4c59022434d39b6bf771c79` (distinct from model.pt file SHA; full file hashes are in the raw report).
- Ancestral reference model.pt SHA256: `998530839d9722316b59cb28ff1ef68a7863f517de73fa8fef8ce9677a24b85e`.
- Native runtime commit: `a5ddce05683dd4c77b1848673f2e87934138e57a`, with `scripts/offline-training/recovery.patch` applied only via `--include='computer/*'`, matching existing knowledge-retention CI. Patch SHA256: `84df51e9eadfe15c7967f340097b57305dc6b61147e8bc94d240a4c9c3bea69c`. No further native runtime changes.
- Torch: `2.14.1+cpu`, CPU,2 threads. Use the same frozen training recipe/dependencies and archived revision4 data. The input archive includes pinned authored/human/replay data and is verified by the harness; obtaining input checkpoints is separate from this code archive.

Verify the archived raw report without training:

```sh
python scripts/research-training/reproduce_ffn_ablation.py --verify-report
python -m gzip -d -c scripts/research-training/ffn-output-20261112/FFN_ABLATION_20261112.json.gz > /tmp/FFN_ABLATION_20261112.json
sha256sum /tmp/FFN_ABLATION_20261112.json
```

Run the small native export/gradient/Adam-resume proof:

```sh
PYTHONPATH=/workspace/airi-lab/native-source/computer /workspace/airi-lab/venv/bin/python scripts/research-training/reproduce_ffn_ablation.py --check-plasticity
```

Reproduce exactly the bounded256×2 comparison, using a fresh output name:

```sh
PYTHONPATH=/workspace/airi-lab/native-source/computer /workspace/airi-lab/venv/bin/python -u scripts/research-training/reproduce_ffn_ablation.py \
  --checkpoint /workspace/airi-lab/rev4/checkpoint/model \
  --archive /workspace/airi-lab/rev4/experiment \
  --reference /workspace/airi-lab/reference/generalist-state/bootstrap-data/candidate \
  --native-replay /workspace/airi-lab/reference/generalist-state/bootstrap-data \
  --output /workspace/airi-lab/FFN_ABLATION_REPRODUCTION.json
```

Source/output audit hashes (SHA256):

- Original unchanged harness: `32909f805d8e0a615a48b4e12bf2894f5434d3d7d1796188131068616b48a638`.
- Archived experimental prefix implementation: `520548fde4f31044956f891e7a32eaa4965c745562fe6d344f91637b7f987fe5`. The raw report records its original production-path location; reproduction stages the identical bytes in a temporary source copy, and current production keeps the frozen implementation.
- Uncompressed full raw report: `96143e2d15c13e3f28b0417f780274ad85a2146c5e953a96c05dceeab339425d`.
- Compressed raw report: `83b441d9676c165e586d2a6d863968173cb9285c1b1d8d9c54e3628217f02c5b`.
- Original log: `b680623ca12789b90800563c91ac99a756da9eaa672946c9938d4091006f2f98`.
