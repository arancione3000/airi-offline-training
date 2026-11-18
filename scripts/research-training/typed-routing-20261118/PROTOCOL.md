# V5 typed-object and operation experiment

The historical V4 freeze accepted 17 of 180 positive TRAIN rows out of fold. Its family head and support classifier both see the complete question. The TRAIN material contains nearly identical operation requests instantiated on multiple objects. This experiment conditions support on the approved program's operation after masking one unambiguous, literal object phrase in the user's question. The existing runtime still checks source compatibility independently.

The classifier uses the previously pinned multilingual MiniLM. It matches the residual request to supported operation frames, competes against other operations and unsupported examples for the same operation, and requires a leading speech head previously seen in an affirmative TRAIN request. All examples with the same masked request are held out together for threshold selection. Each unique (frame, operation, polarity) has equal weight. The threshold pair maximizes supported TRAIN frames with zero negative accepts on a fixed grid. Historical qualification has previously been observed and is suitable only as a repeated regression screen. The candidate is research-only.

Freeze uses only TRAIN, the approved family definitions and the archived bank. Qualification uses the existing evaluator, two numerical/string values per wording, the stored contract gate and historical route gate. Count wrong answers on positive requests separately. The experiment does not access FINAL. Freeze and qualification write to a new run directory and refuse to overwrite results.

With the pinned local encoder installed, from the repository root:

```sh
export AIRI_QUERY_ENCODER_CACHE=/workspace/airi-lab/query-encoder-minilm
export OPENBLAS_NUM_THREADS=1
PYTHON=/workspace/airi-lab/venv/bin/python
DIR=scripts/research-training/typed-routing-20261118
"$PYTHON" -B "$DIR/experiment.py" freeze /tmp/airi-v5-new
"$PYTHON" -B "$DIR/experiment.py" qualification /tmp/airi-v5-new
"$PYTHON" -B -m unittest discover -s "$DIR"
```

## Observed result

V5 was rejected. It answered 26 of 144 supported cases correctly (13 of 72 distinct formulations) and answered 4 of 108 unsupported cases (2 of 54 formulations). No positive received an incorrect answer. It preserved all 360 stored contracts and all 989 historical cases. Five of the eighteen families received no correct paraphrase answer. The model stayed experimental and FINAL stayed unopened.

On the distinct positive formulations the first rejecting stage was the speech-head vocabulary for 53, the inherited input guard for four, and the operation margin for two. There were zero misses at literal object resolution among the requests that reached it. Both unsafe replies involved a request for two different results and a random choice, once for account balance and once for stock. The leading request verb was supported and the prototype comparison incorrectly accepted subtraction. This identifies both a coverage bottleneck and a compositional failure. The zero-negative calibration result on TRAIN did not transfer to the repeated qualification screen.

The compact model, full end-to-end observations, summaries, attribution and input/source hashes are in `evidence/`. To reproduce the post-freeze attribution, run `"$PYTHON" -B "$DIR/diagnose.py" /tmp/airi-v5-new`. Attribution reports each distinct formulation once.
