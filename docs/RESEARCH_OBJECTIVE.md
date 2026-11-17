# Research objective and promotion evidence

The user's objective is an architecture more capable than modern Transformers.
That is a research target, not a result established by the current Airi system.
Current modules use a frozen small Transformer, bounded program induction and
explicit contextual memory. Adding lexical query routing changes system coverage;
it does not create a new neural attention layer or establish broad reasoning gains.

Progress is accepted only through recorded experiments: freeze independent
inputs, train on TRAIN, choose settings using TRAIN-only grouped validation,
qualify on separate examples, then observe an untouched final endpoint. Keep
rejected candidates and failed cases. A later diagnostic cannot erase the first
result. Source, model, data and checkpoint hashes must identify what ran.

The current comparison against Airi's own50million-parameter checkpoint is an
ablation of module usefulness. It cannot substantiate superiority over modern
strong pretrained models. That larger claim requires actual identified strong
baselines, the same task inputs and tools, separate training/heldout sources,
comparable resource budgets, paired results and uncertainty estimates across
language, reasoning, memory, procedural transfer and regressions. A tiny authored
template test or an unspecified model name does not meet that requirement.

The immediate measured failure is paraphrase understanding. The next experiment
learns global question routing with explicit rejection, then requires a uniquely
compatible source and a previously qualified executable program. It preserves
exact expert and memory priority. Unknown intent remains unsupported. This is a
bounded lexical-transfer hypothesis, evaluated independently of the program
library and neural weights. False answers on negative requests reject promotion.

Some old exact-engine fixtures deliberately label untrained question wording as
expert abstention, including a polite prefix before a canonical question. The
new optional router explicitly expands that grammar. The old fixtures and exact
engine tests remain unchanged; reports must distinguish this intended scope
extension from quotation, role/session, conflict and wrong-operation boundaries,
which must continue to reject. Do not claim every old syntax abstention survives
while simultaneously claiming broader wording support.

Automatic cycles may fit candidates from trusted correct examples, replay gates,
and publish qualified artifacts. They must not assign truth from their own
answers, claim improvement after catalog exhaustion, silently select final-test
aliases, or rewrite arbitrary source code to pass observed tests. A plateau is
reported as a plateau. Further capability requires new validated knowledge or
a reviewed architectural experiment; merely leaving training active is no proof.
