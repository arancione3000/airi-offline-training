"""Freeze and evaluate a source-bound grammar on the 30-expert baseline."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[1] / 'assistant-runtime'
sys.path.insert(0, str(RUNTIME))
import intent_grammar
from learn import BUNDLED
from query_learning import bindings, evaluate
from skill_bank import SkillBank

ARCHIVE = HERE.parent / 'semantic-routing-20261116/evidence/precommitted'


def read_split(name):
    return json.loads(gzip.decompress((ARCHIVE / (name + '.json.gz')).read_bytes()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Backend:
    @staticmethod
    def read_model(path):
        model = json.loads(path.read_bytes())
        if model['parser_sha256'] != sha(intent_grammar.__file__):
            raise ValueError('parser changed after freeze')
        return model

    @staticmethod
    def predict(model, query):
        lesson = intent_grammar.predict(query, model['families'])
        return {'lesson_id': lesson} if lesson else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    bank = SkillBank(BUNDLED)
    families = read_split('FAMILIES')['families']
    model = {'parser_sha256': sha(intent_grammar.__file__), 'bindings': bindings(bank, families),
             'families': families, 'bank_sha256': bank.status()['state_sha256']}
    path = output / 'candidate.json'
    path.write_text(json.dumps(model, ensure_ascii=False))
    blind = json.loads(subprocess.check_output([sys.executable, str(HERE / 'blind/cases.py')]))
    results = {}

    def score(split, rows, offset):
        result = evaluate(bank, path, rows, families, value_offset=offset, backend=Backend)
        observations = result.pop('observations')
        (output / (split + '.json.gz')).write_bytes(gzip.compress(
            json.dumps(observations, ensure_ascii=False).encode(), mtime=0))
        result['distinct_positive_phrases_correct'] = sum(r['correct'] for r in observations
                                                          if r['positive'] and r['variant'] == 0)
        results[split] = result

    score('qualification', read_split('QUALIFICATION')['rows'], 10000)
    score('challenge', blind['rows'], blind['value_offset'])
    if all(result['passed'] for result in results.values()):
        score('final', read_split('FINAL')['rows'], 20000)
    summary = {'parser_sha256': model['parser_sha256'], 'bank_sha256': model['bank_sha256'],
               'eligible': all(result['passed'] for result in results.values()), 'results': results}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
