"""Post-freeze descriptive attribution of the observed qualification endpoint."""
from collections import Counter
import json
from pathlib import Path
import sys

from experiment import load
import typed_router as router


def diagnose(output):
    model = router.read_model(output / 'candidate.json')
    compiled = router.compile_model(model)
    positive = Counter()
    errors = []
    sources = load('precommitted/QUALIFICATION.json.gz')['rows']
    observed = json.loads((output / 'qualification.json').read_bytes())
    by_id = {r['id']: r for r in observed['observations'] if r['variant'] == 0}
    distinct_heads = {router.speech_head(text) for text, _, pos in model['cases'] if pos}
    for row in sources:
        observation = by_id[row['id']]
        if row['lesson_id'] is None:
            if not observation['correct']:
                errors.append({'id': row['id'], 'query': row['query'],
                               'selected_operation': model['operations'][row['source_lesson_id']]})
            continue
        if observation['correct']:
            stage = 'correct'
        elif router.common._normal(row['query']) is None:
            stage = 'input_guard'
        else:
            parsed = router.entity_frame(row['query'], model['entities'])
            if parsed is None:
                stage = 'object_unresolved'
            elif parsed[0] != row['source_lesson_id']:
                stage = 'source_disagreement'
            elif router.speech_head(parsed[1]) not in distinct_heads:
                stage = 'unseen_speech_head'
            else:
                op = model['operations'][parsed[0]]
                vector, _ = router.semantic_encoder.encode([parsed[1]])
                rank = compiled.prototypes.rank(parsed[1], vector[0], op)
                if rank is None:
                    stage = 'no_comparison'
                elif rank[0] < model['thresholds'][0]:
                    stage = 'below_similarity'
                elif rank[1] < model['thresholds'][1]:
                    stage = 'below_margin'
                else:
                    raise ValueError('end-to-end miss was accepted by the classifier')
        positive[stage] += 1
    return {'positive_first_decision_by_distinct_question': dict(positive),
            'negative_false_answers_by_distinct_question': errors}


if __name__ == '__main__':
    print(json.dumps(diagnose(Path(sys.argv[1])), ensure_ascii=False, indent=2))
