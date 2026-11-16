# Reproducing bounded modular skill growth

Six persistent symbolic experts were learned from 48 labeled examples. The first
qualified run scored **48/48 development**, **96/96 final**, and **78/78 required
abstentions**. All six admissions preserved the 989 historical routing contracts
without a learned-expert interception. A second deployment regression obtained
the same counts after a CLI-only change to the `--no-skills` safeguard.

A separate 24-case native comparison scored **0/24** without examples,
**0/24** with the same eight TRAIN examples in context, and **24/24** with the
learned programs. This uses the approved revision-4 checkpoint and does not
change the full 96-case final or historical preservation measurements.

These are bounded programming-by-example results. The approved neural weights
did not change. Final cases contain unseen values, longer strings, new Unicode
characters, and larger signed integers **inside the source and question templates
seen in TRAIN**. This does not demonstrate novel natural-language wording,
arbitrary algorithm discovery, neural structural morphing, or a better Transformer.

## Frozen inputs and code

The independent generator and [plan](PLAN.md) were fixed before learner freeze.
The data manifest SHA256 is
`165fe7160767092135d8418a23acd855889888ed37ac13d48e01263dd19c96f3`, with precommit
time `2026-11-16T06:41:18.112012+00:00`. The compiler hash in both run manifests is
`413192c0218e6268a6f6622ae91abd9f0aecf1684fbb3205a48388dfca72ed00`.

Each lesson contains eight TRAIN rows and a separate admission set of eight
positive/four negative rows. Only TRAIN selects the extraction anchors, exact
question, and operation. Qualification accepts or rejects the selected program;
it never chooses a different operation. Development and final labels enter only
the endpoint evaluator. Each admitted expert has a unique replay-verified
hypothesis in the fixed copy/reverse/uppercase/lowercase or
addition/subtraction/multiplication library.

`evidence/MANIFEST.json` records compressed and decompressed SHA256 digests and
sizes for every archived artifact. Gzip uses an empty filename and timestamp 0,
so the archive bytes are deterministic. The bundle contains all precommitted
cases/lessons, the original data manifest, both full reports and preregistrations,
the six-expert synthetic bank and its six growth snapshots, source snapshots,
the evaluation driver, and the unchanged curriculum/plan. It contains no model
weights, optimizer state, credentials, or private user lessons.
The separate native comparison cases, full report, and comparison driver are
also archived.

The first run's `chat.py` snapshot intentionally has SHA256
`1c89e4485393ab101e9e104836bd303aeb4ee4aa96139f3ea8555a60b7f8917f`.
That is the exact CLI source bound by its original preregistration. The second
run binds the corrected CLI. Both sets of snapshots match their own run's source
hashes; the compiler is identical. The first run remains the held-out endpoint.
The second is a deployment regression on already observed cases and provides
no additional independent statistical confirmation.

## Running again

Use local Python 3 on Linux; this bounded experiment needs no Torch, model
download, network access, or external model API. From the repository root:

```bash
python scripts/research-training/modular-growth-20261116/curriculum.py --check-only
```

To materialize the original payloads without their historic creation timestamp,
run the generator with a fresh `--out` directory. Its five dataset byte hashes
must equal the original data manifest. To restore every original input byte,
decompress the archived precommit bundle instead:

```bash
python - <<'PY'
import gzip
from pathlib import Path

source = Path('scripts/research-training/modular-growth-20261116/evidence/precommitted')
target = Path('/tmp/airi-modular-growth-precommitted')
if target.exists():
    raise SystemExit('choose a fresh target directory')
for artifact in sorted(source.rglob('*.gz')):
    destination = target / artifact.relative_to(source).with_suffix('')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.decompress(artifact.read_bytes()))
PY
python scripts/research-training/modular-growth-20261116/run_growth.py \
  --curriculum /tmp/airi-modular-growth-precommitted \
  --output /tmp/airi-modular-growth-reproduction
```

The output directory must be absent. The driver creates a fresh empty bank,
records input/runtime hashes, admits the six lessons in order, evaluates the
fixed splits, and checks restart, idempotence, corrupted-label rejection, and
recovery from a damaged convenience copy. Runtime and training durations depend
on the machine. Program identities, registry payload hashes, answers, provenance,
and qualification outcomes should reproduce exactly.

For exact first-run code reproduction, restore `evidence/qualified-run/sources`
into an isolated checkout before running the archived driver. For the second
run, use `evidence/deployment-regression/sources`. Their manifests preserve all
runtime `.py` files used by the driver; the driver itself is additionally
archived and hashed by this evidence package. The stable routing contract is
the repository's `scripts/assistant-runtime/skills/stable_contracts.json.gz`,
whose raw SHA256 is recorded in the run reports and evidence manifest. The
historical native output artifact used for inherited fallback is already
versioned under `learnability-20261115/evidence/REPORT.json.gz`.

## What preservation and baselines mean

The learned experts intercepted **0/989** historical contract cases. Existing
memory handled 24 of those routes; the other 965 inherit the unchanged native
call. The historical composed score remains **27/92**, using actual symbolic
routing with hash-bound archived native outputs for fallback. This is a routing
preservation result, not fresh native inference of all 92 or 897 examples and
not a new neural certificate. Native comparison measurements, if performed
separately, must retain their own sample counts and artifact provenance.

The pre-existing memory answered none of the new 96 final cases; an independent
precommit check also found no memory answers on all 192 TRAIN/dev/final supported
rows. An empty learned registry has no executable expert entries; the report's
empty-bank zero is a structural baseline, not a separately timed native-model
measurement. After learning, all 96 final answers come from the persisted
programs, with exact source/value spans. Negative correctness measures expert
abstention; native fallback may still answer poorly afterward.

The registry grew from zero to six experts, retained earlier lesson contracts,
survived a fresh load, accepted repeats as unchanged, rejected the corrupted
lesson without registry mutation, and recovered the same six-expert payload
after simulated main-copy corruption. The first-run median measured prediction
latency was about 14.5 ms for this six-expert bank. These checks establish the
declared bounded capability and persistence behavior; they do not establish
open-world accuracy, unlimited growth, or immunity to every storage failure.

## Separate native comparison

`evidence/native-comparison/CASES.json.gz` fixes the first four precommitted
FINAL cases per lesson: 24 cases in total, selected by a mechanical rule before
native inference. The full report preserves every plain-native, eight-example
in-context, and program output. Both native conditions use the same verified
approved revision 4, tensor digest
`57bdb933b82bd8807af5c8b6b545c27ee9114ee4f4c59022434d39b6bf771c79`.
The native path is warmed against one known correct historical response, then
uses greedy generation with a 64-token answer limit. All eight demonstrations
fit within the native context window; no few-shot prompt is silently truncated.
Greedy inference leaves the recorded Torch RNG state unchanged.

| Condition | Exact answers |
| --- | ---: |
| Native, no demonstrations | 0/24 |
| Native, eight TRAIN demonstrations in context | 0/24 |
| Learned symbolic expert | 24/24 |

This is a direct comparison against this specific frozen Airi checkpoint on a
small systematic subset. The expert receives a human-designed operation library
and an admission procedure, while the native baselines receive zero or eight
demonstrations without weight updates. Training compute and inductive biases
are not matched. It does not compare against a neural model fine-tuned on this
curriculum, larger models, alternative prompting, or the Transformer class in
general. The native conditions were measured after the symbolic final endpoint;
this is a new baseline measurement on already declared cases, not a second
independent held-out qualification.

Reproducing this additional comparison requires the native runtime's Torch
environment and the approved hash-verified checkpoint/record, which are not
included in this source-and-data evidence bundle. Run `compare_native.py` with
`--curriculum`, `--bank`, `--checkpoint`, `--record`, and a fresh `--output`
directory. The archived comparison driver fixes the sample rule and generation
settings. Its source hash is supplied by this evidence package; the comparison
report does not itself contain a preregistered driver hash.
