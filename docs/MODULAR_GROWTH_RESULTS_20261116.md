# Persistent procedural growth, 16 November 2026

Airi can now add a qualified procedural expert from examples, keep it across
process restarts, and append later experts without overwriting earlier ones.
The default experimental chat loads six qualified experts. The approved neural
model remains revision4, with tensor digest
`57bdb933b82bd8807af5c8b6b545c27ee9114ee4f4c59022434d39b6bf771c79`.

## Diagnosis and change

The preceding broad neural trial memorized391/512 training rows, but scored0/32
when both values and wording were new and3/92 on old tasks. More repetitions of
that recipe were not evidence of improvement. We tested a separate retention
path: keep the approved base and existing memory stable, infer bounded programs
from examples, independently qualify them, and persist accepted modules.

The compiler receives only labeled TRAIN examples. It infers literal source
anchors, an exact question and one operation from a fixed seven-operation DSL.
It rejects ambiguous or inconsistent lessons. Qualification uses new extracted
values and includes required abstentions. Admission replays all earlier expert
contracts and989 stable historical routes. Old routes retain priority. The
registry uses locks, checksums, an engine-version binding and atomic snapshots;
an invalid lesson or failed gate cannot replace the approved bank.

## Measurements

The independent curriculum was precommitted at06:41:18 UTC before compiler freeze.
It contains six lessons with eight TRAIN examples each, followed by separate
qualification and endpoint cases. No final labels select a program.

| Check | Result |
| --- | --- |
| Expert count before/after | 0 → 6 |
| Development, new values | 48/48 |
| Final, new values/lengths/Unicode/integer ranges | 96/96 |
| Required expert abstentions | 78/78 |
| Historical989 routes intercepted by a new expert | 0 |
| Historical composed92, archived native fallback | 27/92, none of23 native successes lost |
| Restart, repeated lessons, corrupted lesson, damaged main copy | Passed |

On the fixed first four FINAL cases per lesson (24 cases), actual inference with
the same verified neural checkpoint scored0/24 without demonstrations and0/24
with that lesson's eight solved TRAIN examples in context. The learned programs
scored24/24. Inputs fit the1024-token native context, decoding was greedy and RNG
state stayed unchanged. This is a narrow system comparison, not equal training
compute, a comparison against modern strong models, or a Transformer-class result.

Initial synthesis, qualification and regression took about0.85 seconds locally;
median prediction including registry validation was14.5ms for six experts. These
durations depend on hardware. A later deployment regression reproduced the same
scores after a CLI safeguard change; it is not another blind experiment.

Historical preservation is checked by actual symbolic routing and hash-bound
archived native fallback, not a fresh rerun of every neural token. New experts
intercepted no old989 cases, memory retained24 routes and965 native calls were
inherited unchanged. Negative scores measure expert abstention; native fallback
may still give a poor answer. The native23/92 score has not improved.

## Scope and next evidence

This is programming by example and modular continual learning around a frozen
Transformer. Its useful gain is persistent acquisition without overwriting old
parameters. It does not demonstrate general chat, novel wording, arbitrary
algorithm discovery, neural topology mutation or architectural superiority over
Transformers. Capacity is32 experts and the operation library is fixed. More
capable modules require an independently qualified compiler/runtime extension.

The trusted queue runs autonomously while its process is alive and correct new
lessons arrive. A subprocess test adds a second file after watcher startup and
verifies activation, restart and idempotence. It does not invent reliable labels
from its own answers or guarantee improvement just by waiting. The pre-existing
native training service remains unchanged; this is a separate learning path.
The local watcher was started and checked on16 November at07:22 UTC, polling
every30 seconds with the qualified six-expert private bank. This process is
limited to this workspace's lifetime; it is not an installed remote-PC service.

Local verification passed87 assistant tests and7 continuous-training tests,
the unchanged memory gates and a fresh CI replay of the six-module admission
pipeline. The CI now runs both old memory/retention checks and modular growth.

See the [precommitted plan](../scripts/research-training/modular-growth-20261116/PLAN.md),
[complete evidence/reproduction instructions](../scripts/research-training/modular-growth-20261116/REPRODUCTION.md)
and [CLI/lesson setup](../scripts/assistant-runtime/README.md).
