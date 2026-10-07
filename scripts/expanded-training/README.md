# Larger native language corpus

The previous 4096-update recovery completed at step 17203 but failed quality:
repetition rose to 0.4608 and the 50 responses were not useful conversation.
This is a new, explicitly authorized experiment, not a repeat of that recipe.

Download and pin 20,000 genuinely additional normalized sentences per language
from the native human-written Tatoeba sources (Italian CC-BY-2.0-FR, English CC0).
Sentence authors, source URLs and raw download SHA are retained. Exclude exact
known native and dialogue split text. Reserve 128 further sentences per language
for evaluation. These sentences teach next-token language modeling; they are not
new question-answer dialogues. Existing training-only dialogues retain chat replay.

Only the exact verified final recovery release can migrate. Preserve native model,
tokenizer, Adam moments and RNG. Keep LR2e-6/UL.2/EOS2/KL4. Train 50% Italian raw
sentences, 25% English raw sentences, 25% existing chat updates (half IT/EN).
Raw sentences use complete BOS/text/EOS sequences, without truncation or padding.

After 512 real updates, test against the starting checkpoint. Require the existing
native retention gate, zero repetition regression, development nonregression,
lower heldout raw NLL in both languages and native language NLL gain >= .005.
If any fails, publish results and terminate the chain. If all pass, continue up to
32,768 additional updates (global 49971), using bounded free CPU GitHub runners.
The same shared concurrency group prevents overlapping training writers.

The immutable source and original final production gates remain unchanged. No
pretrained model, paid compute or live promotion. More data/time does not guarantee
conversational ability. Evaluate raw final generations manually.
