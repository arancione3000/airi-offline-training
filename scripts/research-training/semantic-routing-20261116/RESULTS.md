# Question-routing research: candidates rejected

None of the four candidates cleared the predefined promotion gate: at least50%
positive coverage, no false answer on negative requests, preservation of stored
expert contracts and989 historical routes. No question model was activated.

| Candidate | TRAIN selection | Qualification correct positives | False answers |
| --- | --- | --- | --- |
| V1 lexical centroid/nearest examples | TRAIN leave-one-out | 12/144 | 0/108 |
| V2 discriminative lexical classifier | TRAIN grouped fivefold | Not evaluated; TRAIN precision gate rejected all coverage | Not evaluated |
| V3 family classifier plus learned operation consensus | TRAIN grouped fivefold | 38/144 | 6/108 |
| V4 frozen multilingual Transformer features, family and support heads | TRAIN grouped fivefold | 30/144 | 0/108 |

Qualification has72 authored positive questions and54 negative questions, each
tested with two counterfactual source values. These are not144 independent
question formulations. V3's six false answers correspond to three negative
questions beginning with `Evita di fornire...`; both source values caused a false
answer. All observed qualification runs preserved360 stored expert contracts and
989 historical routes. Exact experts and contextual memory keep priority.

V1 and V2 could not reliably distinguish an operation request from words naming
an object. V3 adds an operation-consensus gate learned from positive TRAIN with
object identifiers masked, rather than a handwritten operation-verb dictionary.
It still missed some negative requests. V4 uses the pinned Apache-2.0
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` encoder, quantized
ONNX, with TRAIN-fitted class-balanced ridge heads and TRAIN-only confidence
calibration. It avoids observed false answers at the cost of low coverage.

All feature choices and thresholds use TRAIN. The same qualification set was
used to accept/reject successive hypotheses; these are not independent
replications. Every candidate failed before FINAL, so FINAL and the six old
diagnostic paraphrases were not evaluated. No failed qualification result was
replaced by a later result. Original runs, exact source snapshots, source hashes
and model identities remain separate. The V2 and separately explored V3 TRAIN
prototype records are explicitly distinct from observed run records.

These trials do not demonstrate improvement over modern strong Transformers.
The multilingual encoder is itself a Transformer component, not a generative
competitor or evidence of a new neural attention architecture. Native Airi
weights are unchanged. The measured conclusion is that these routing methods,
with this small corpus, do not yet provide enough reliable transfer to replace
the existing exact-question path. Narrow positive coverage is not hidden, but
it does not satisfy the promotion rule.

The separate procedural curriculum continues normally; its eighteen-expert
catalog can finish without solving this language problem. The new research
workflow reruns when trusted examples, learner code or program bindings change,
preserves previously qualified semantic answers if a model is later promoted,
and skips unchanged experiments. It cannot create reliable new truth from its
own answers. See [reproduction](REPRODUCTION.md) and the
[research objective](../../../docs/RESEARCH_OBJECTIVE.md).
