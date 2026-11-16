# Bounded procedural growth: independent curriculum and scientific audit

This plan was authored on 16 November 2026 without reading `skill_programs.py`.
The generator only reads its own source and this plan when writing the manifest.
It uses seed **20261116063627**, local Python, no Torch, model, external API, or
historical benchmark. The generated manifest fixes the exact dataset hashes and
creation time before the learner is frozen. Only the 48 examples in
`precommitted/LESSONS/TRAIN.json` may synthesize or select initial candidate
programs. Its separate qualification rows may accept or reject a candidate,
and must never select the candidate's operation.

## Question and interpretation

Can a stable neural base acquire six useful, bounded procedural response skills
from labeled examples by attaching persistent, gated symbolic experts, while
preserving the approved model and the existing contextual memory?

The prior evidence motivates separate storage and selection of capabilities.
Revision 4 scores 23/92 on the historical native benchmark; adding explicit
context memory scores 27/92 with unchanged weights. Full native updates learned
some training rows but lost earlier successes. The broader 512-row candidate
scored 0/32 when both values and templates were new, and 3/92 historically. These
results diagnose those recipes; they do not establish that Transformers cannot
learn, nor predict the ceiling of a different neural architecture.

The proposed change is a hybrid assistant architecture: immutable approved base,
existing memory, and a registry of learned bounded programs. This is a useful
form of programming by example and modular continual learning. Selecting among
predefined operations is **not** a novel Transformer layer, neural structural
morphing, unrestricted program synthesis, or evidence of improved neural weights.
The candidate can demonstrate new system capability and avoidance of destructive
interference by construction. It cannot substantiate a claim that it solves
general chat, discovers arbitrary algorithms, or improves the Transformer itself.

## Independent task contract

Each lesson has eight distinct labeled examples (four counterfactual pairs).
Each transcript has a source user message followed by the exact query below.
The operation is not supplied in the runtime query or lesson metadata. An expert
must select its operation from consistent labels, rather than copying a stored
answer or matching an operation keyword in a task name. The evaluator may know
the authored task identities; the learner's training input contains no operation
field. A lesson's source and query wording are fixed across every split.

| Lesson | Source message | Query | Authored target |
| --- | --- | --- | --- |
| parcel_copy | `Il codice del pacco è {value}.` | `Qual è il codice del pacco?` | Copy |
| signal_reverse | `Il segnale acquisito è {value}.` | `Come restituisci il segnale?` | Reverse code points |
| card_upper | `La tessera porta il testo {value}.` | `Qual è il testo della tessera?` | Unicode uppercase |
| plate_lower | `La targa porta il testo {value}.` | `Qual è il testo della targa?` | Unicode lowercase |
| delta_sum | `La coppia delta è {a} e {b}.` | `Qual è il risultato delta?` | Integer sum |
| sigma_product | `Gli interi sigma sono {a} e {b}.` | `Qual è il risultato sigma?` | Integer product |

The text hypothesis class is copy/reverse/uppercase/lowercase; the integer class
is addition/subtraction/multiplication. Every lesson has exactly one hypothesis
consistent with all eight authored labels, checked independently by the dataset
generator. Training integers have magnitudes 3–24, qualification magnitudes
30–51, dev magnitudes roughly
60–1801, and final magnitudes roughly 10001–950001. Dev and final use unseen
values; final text includes new Unicode characters, case expansion, and lengths
beyond TRAIN. Code-point reversal is intentional; this does not test grapheme-
aware reversal. Text values remain opaque strings inside bounded supported
syntax. Python's Unicode case mappings define the expected case transformations.

Each lesson also carries eight positive and four negative qualification rows
with separate source values. These are the candidate's admission gate, not
the held-out development/final evaluation. Development contains 48 supported
cases (four pairs per lesson); final contains
96 (eight pairs per lesson). Every pair changes only the source value while
keeping the query constant, and requires a different answer. None of these
source messages occurs in TRAIN. Text values carry randomized suffixes from the
fixed seed. This demonstrates value-dependent procedures inside known templates;
it **does not** measure transfer to new natural-language wording. Do not merge
these results with the historical 92 cases or the 897 certified TRAIN routes.

## Preregistered acceptance gates

The initial candidate must explain all TRAIN labels with a unique bounded
program. No candidate may execute input as code, call arbitrary functions, or
learn from evaluation gold. After the learner implementation is frozen, run
all dev and final cases exactly once and preserve per-row outputs. This v1 is
successful only if it answers all supported dev and final cases exactly and
the expert abstains on every one of the 78 negative cases. Deterministic,
finite operations on this small declared domain justify that strict threshold.
If a bug is repaired after observing held-out labels, report v1 honestly and
precommit an independently generated v2 before evaluating the repair.

Each family contributes 13 negative cases: quoted source, hypothetical source,
unresolved conflict, system instruction, developer instruction, assistant-only
source, tool-only source, speaker mismatch, session mismatch, fenced code source,
code as an input value/expression, missing source, and untrained query wording.
Their target is **expert abstention**, represented by `expected_answer: null` and
`expected_abstain: true`; the composed assistant may subsequently call the native
fallback. Do not score a bad native fallback answer as an expert false answer,
or claim that abstention guarantees a good final answer.

The corrupted signal lesson has one inconsistent gold output. It must be
rejected without changing the persisted registry or any previously acquired
expert. Repeating the same trusted lesson must be idempotent; loading a fresh
process from disk must reproduce all acquired operations. A rejected write must
leave the previous registry valid. Registry hashes, before/after expert counts,
and existing answers make the growth and retention claims independently
reviewable. Refuse ambiguous candidate programs and conflicting task ownership.

With the registry empty, the learned layer must answer none of the final cases.
The pre-existing contextual memory should also abstain on these unsupported
queries. Acquiring each lesson should add exactly that family's supported
responses, while retaining earlier lessons. These ablations demonstrate that
the new capability comes from the learned registry rather than a static answer
table or existing memory. They do not by themselves establish novelty against
other program-induction methods. As an optional additional causal check, train
a fresh isolated registry on the same template and different consistent labels:
its chosen operation should follow the labels, not the lesson identifier.

Only after code freeze, verify historical preservation separately: all previously
correct 92-case composed/native answers and the 897 certified TRAIN routes.
Report direct re-inference separately from preservation inherited by forwarding
unchanged requests and parameters to the immutable native base. An expert route
must preserve speaker/session boundaries and global instruction roles. Record
the approved base identity and stable memory hashes. Do not weaken or overwrite
the existing native qualification gates or activate rejected neural weights.

## Threats to validity and reporting

The operation library, fixed parser restrictions, and choice of six tasks are
human-designed inductive bias. A program selector cannot learn a capability
outside its hypothesis class. Authored synthetic labels give exact gold but do
not represent a natural distribution of user tasks. The same organization
implements and evaluates the system; independent authorship of these cases
reduces direct label fitting without providing a blinded external benchmark.
Small sample sizes do not establish open-world accuracy or false-answer rates.

Perfect preservation is aided by freezing the base and using conservative
routing, rather than learning a representation immune to forgetting. The fixed
templates may become useful additional capabilities, but manual installation
of one-off task rules would not count as learning the expert from examples.
Report parser templates, selected operations, lesson evidence and provenance,
registry size, deterministic runtime cost, and success/failure by family.
Keep native model, memory, and learned-program results separate. A candid
successful conclusion is “bounded procedural skills learned from examples with
persistent modular retention”; “better Transformer” remains unproven.

## Reproduction

Run `python scripts/research-training/modular-growth-20261116/curriculum.py` to
materialize the bundle, or pass `--check-only` to validate generation without
writing. Existing dataset or manifest content cannot be silently overwritten.
The timestamp is recorded once; all dataset bytes and their hashes are
deterministic. Keep `MANIFEST.json` and the exact generator/plan alongside the
learner freeze and endpoint receipts. Never regenerate a different seed after
looking at v1 results and call it the original precommitment.
