# Independent semantic routing curriculum

This corpus was authored without reading the proposed router implementation or
its predictions. It reuses the eighteen established task identities, canonical
questions and literal source formats; all other questions are authored here.
Only TRAIN may select feature weights, neighbors, thresholds or candidates.
QUALIFICATION may accept or reject the frozen candidate. FINAL is an endpoint,
not tuning data. Old six observed paraphrases are DIAGNOSTIC only and are absent
from every newly authored split.

Each family has ten TRAIN positives including its canonical question, four
qualification positives, and six final positives. Unsupported requests contribute
five TRAIN, three qualification and four final negatives per family. Negative
rows retain the source family so end-to-end evaluation can supply matching facts;
their lesson_id is null. Exact normalized questions are disjoint across splits.
The source-family constraint is part of the routing task: a question requesting
another operation on the same facts must abstain rather than execute that task.

The final frames differ from training and qualification frames, but share task
identifiers and some operation vocabulary. This tests bounded lexical transfer
inside a small authored domain, not broad semantic understanding, world knowledge,
unlimited continual learning or superiority over modern Transformers. The same
organization authors and evaluates the data; this is not an external benchmark.
Question-only nearest-neighbor performance must remain separate from source
eligibility, final answer accuracy, native fallback behavior and abstention.

Use fresh disjoint values from independently labeled task generators for answer
evaluation. String/integer reference operations, not router/compiler predictions,
provide gold. Preserve negation, reversed subtraction order, incompatible
operations, quoted/code questions, multi-intent and unsupported requests as
negative conditions. Qualification/final must never select a new alias or threshold.

The generator writes a creation timestamp once and hashes every exact payload,
its own source and this plan. A later observed failure must be reported with its
original endpoint; a changed candidate needs a newly declared experiment.
