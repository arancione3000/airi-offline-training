# Fresh native/composed review, 16 November 2026

The frozen twelve-expert composed assistant answered **9/12** fixed fresh-value
cases correctly. The same verified small native revision4 model answered **0/12**
without demonstrations and **0/12** with that expert's eight actual TRAIN examples
in context. All prompts fit1024 tokens; greedy decoding used at most64 new tokens.

| Original condition | Composed correct | Native plain | Native8TRAIN |
| --- | --- | --- | --- |
| Known task/question templates, fresh values | 9/12 | 0/12 | 0/12 |
| Six authored question paraphrases | 0/6 | 0/6 | 0/6 |

Inspection after the fixed run found that the three arithmetic values exceeded
the DSL's declared integer limit of1,000,000. The experts correctly abstained;
their native fallbacks were wrong. The table retains those original system
failures. All nine text cases were correct. This is a numeric scope boundary,
not a justification to inflate the score or silently widen the runtime limit.

A separately preregistered **post-review diagnostic** tested three fresh arithmetic
cases within that limit: composed3/3, native plain0/3, native8TRAIN0/3. No compiler,
expert, alias or weight changed between runs. This diagnostic is not pooled into
a new blind12/12 result and does not erase the original9/12 endpoint.

On six quoted-source probes, the expert abstained6/6; native fallback output is
retained without inventing a native abstention accuracy score. All240 stored
expert TRAIN/qualification contracts passed through actual routing. The989 old
historical routing contracts also passed with zero learned interceptions and965
unchanged native routes; this is inherited preservation, not a new989-case neural
generation certificate. Checkpoint file hashes were revalidated after inference
and Torch RNG state stayed unchanged in both runs.

The useful effect is bounded, persistent procedural responses around a frozen
model. The major limitation remains exact question wording: none of these six
semantic paraphrases was answered correctly by the composed system. This does
not establish general conversation improvement, arbitrary algorithm discovery,
equal training-compute advantage or superiority over Transformers as a class.
The native comparator is this particular small, weak Airi model, not a current
strong pretrained Transformer.

See [reproduction and evidence](REPRODUCTION.md). The first source freeze occurred
at22:30:04 UTC; original and diagnostic evidence remain separate and hash-bound.
