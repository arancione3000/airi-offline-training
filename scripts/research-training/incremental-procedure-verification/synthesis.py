"""Learn one-operation graph extensions from scalar labels, then certify them."""
import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import random
from struct import Struct

from fixtures import INPUTS, scenarios
from verifier import MAX_MAGNITUDE, OPERATIONS, Verifier

EXTENSIONS = ('add_residue_diagnostics', 'add_comparison_workflows')
PASSIVE_SEEDS = tuple(range(8))
MAX_ENUMERATION_CELLS = 25_000_000


@dataclass(frozen=True)
class Candidate:
    operation: str
    dependencies: tuple[str, ...]
    outputs: tuple[int, ...]

    def definition(self):
        return {'op': self.operation, 'deps': list(self.dependencies)}


def candidate_digest(candidates):
    digest = sha256()
    pack = Struct(f'>{len(INPUTS)}i').pack
    for candidate in candidates:
        digest.update(json.dumps(candidate.definition(), sort_keys=True).encode())
        digest.update(pack(*candidate.outputs))
    return digest.hexdigest()


class RecipeLibrary:
    """Keep recipe evaluations for the preceding enumeration in one run."""

    def __init__(self, cache):
        self.cache = cache
        self.tables = {}
        self.evaluations = {}
        self.fixed_inputs = {'a': tuple(a for a, _ in INPUTS), 'b': tuple(b for _, b in INPUTS)}

    def enumerate(self, approved_tables):
        """Use only approved output columns and the original input columns."""
        tables = {**approved_tables, **self.fixed_inputs}
        names = sorted(tables)
        recipes = [
            *((op, (name,)) for op in ('absolute', 'negate') for name in names),
            *((op, deps) for op in ('add', 'multiply') for deps in combinations_with_replacement(names, 2)),
            *(('subtract', deps) for deps in product(names, repeat=2)),
        ]
        if len(recipes) * len(INPUTS) > MAX_ENUMERATION_CELLS:
            raise ValueError('procedure search exceeds its bounded domain budget')

        unchanged = {name for name, table in tables.items() if table == self.tables.get(name)}
        current = {}
        unique = {}
        executed = 0
        reused = 0
        invalid = 0
        for op, deps in recipes:
            key = op, deps
            if self.cache and key in self.evaluations and unchanged.issuperset(deps):
                outputs = self.evaluations[key]
                reused += 1
            else:
                function = OPERATIONS[op][1]
                arguments = (tables[dep] for dep in deps)
                outputs = tuple(map(function, *arguments))
                executed += len(outputs)
                if any(abs(value) > MAX_MAGNITUDE for value in outputs):
                    outputs = None
            if self.cache:
                current[key] = outputs
            if outputs is None:
                invalid += 1
                continue
            unique.setdefault(outputs, Candidate(op, deps, outputs))
        if self.cache:
            self.tables, self.evaluations = tables, current
        candidates = tuple(unique.values())
        return candidates, {
            'recipe_count': len(recipes),
            'distinct_valid_output_tables': len(candidates),
            'candidate_tables_sha256': candidate_digest(candidates),
            'invalid_recipes': invalid,
            'executed_cells': executed,
            'reused_recipe_evaluations': reused,
            'reused_cells': reused * len(INPUTS),
        }


def next_probe(candidates):
    """Choose an input that minimizes expected ambiguity, then worst-case ambiguity."""
    best = None
    columns = zip(*(candidate.outputs for candidate in candidates), strict=True)
    for index, outputs in enumerate(columns):
        counts = Counter(outputs).values()
        score = (sum(size * size for size in counts), max(counts), index)
        if best is None or score < best:
            best = score
    return best[-1]


def learn(candidates, oracle, choose=next_probe):
    """The oracle answers exactly one indexed input per call."""
    queries = []
    while len(candidates) > 1:
        index = choose(candidates)
        answer = oracle(index)
        remaining = tuple(candidate for candidate in candidates if candidate.outputs[index] == answer)
        if remaining == candidates:
            raise ValueError('probe did not narrow the candidates')
        queries.append({
            'index': index,
            'input': list(INPUTS[index]),
            'label': answer,
            'remaining_candidates': len(remaining),
        })
        candidates = remaining
    return (candidates[0] if candidates else None), queries


def passive_learning(candidates, oracle, seed):
    """Measure a fixed shuffled input order with the same candidates."""
    indices = list(range(len(INPUTS)))
    random.Random(seed).shuffle(indices)
    iterator = iter(indices)

    def choose(remaining):
        vectors = [candidate.outputs for candidate in remaining]
        # Ignore uninformative inputs without asking their labels. This gives
        # the shuffled control access to the same predicted disagreement.
        return next(i for i in iterator if any(v[i] != vectors[0][i] for v in vectors[1:]))

    return learn(candidates, oracle, choose)


def stage_extension(verifier, approved_graph, expected, library):
    """Synthesize new recipes, then check the entire candidate atomically."""
    candidates, search = library.enumerate(verifier.tables)
    new_names = sorted(expected.keys() - approved_graph.keys())
    if not new_names:
        raise AssertionError('extension did not introduce any procedures')
    additions = {}
    reports = {}
    for name in new_names:
        def oracle(index):
            return expected[name][index]

        recipe, queries = learn(candidates, oracle)
        if recipe is None:
            raise AssertionError(f'no bounded recipe for {name}')
        passive = []
        for seed in PASSIVE_SEEDS:
            other, observed = passive_learning(candidates, oracle, seed)
            if other != recipe:
                raise AssertionError(f'active and shuffled recipes disagree for {name}')
            passive.append(len(observed))
        additions[name] = recipe.definition()
        reports[name] = {'recipe': additions[name], 'active_labels': len(queries),
                         'active_queries': queries, 'shuffled_label_counts': passive}

    graph = {**approved_graph, **additions}

    # Change a withheld reference value and verify that the extension is
    # vetoed as a whole, despite matching all its observed scalar labels.
    final_name = new_names[-1]
    observed = {row['index'] for row in reports[final_name]['active_queries']}
    withheld = next(i for i in range(len(INPUTS)) if i not in observed)
    conflict = list(expected[final_name])
    conflict[withheld] += 1
    contradicted = {**expected, final_name: tuple(conflict)}
    before_tables = dict(verifier.tables)
    before_version = verifier.version
    negative = verifier.update(graph, contradicted)
    if (negative['status'] != 'rejected' or negative['mismatch']['node'] != final_name
            or negative['mismatch']['index'] != withheld or verifier.version != before_version
            or verifier.tables != before_tables):
        raise AssertionError('unobserved contradiction did not preserve the approved graph')

    qualification = verifier.update(graph, expected)
    if qualification['status'] != 'accepted' or verifier.tables != expected:
        raise AssertionError('learned graph failed complete independent validation')
    return graph, {
        'enumeration': search,
        'learned_workflows': reports,
        'active_labels': sum(row['active_labels'] for row in reports.values()),
        'shuffled_labels_by_seed': [sum(row['shuffled_label_counts'][i] for row in reports.values())
                                    for i in range(len(PASSIVE_SEEDS))],
        'admission': qualification,
        'unseen_conflict': negative,
    }


def replay(cache):
    verifier = Verifier(INPUTS, mode='semantic')
    library = RecipeLibrary(cache)
    approved = {}
    extensions = {}
    for name, proposed, reference, should_accept in scenarios():
        if name in EXTENSIONS:
            # Read no new node definition from the fixture in this branch.
            approved, evidence = stage_extension(verifier, approved, reference, library)
            extensions[name] = evidence
            if len(extensions) == len(EXTENSIONS):
                break
        else:
            result = verifier.update(proposed, reference)
            if (result['status'] == 'accepted') != should_accept:
                raise AssertionError(f'fixture acceptance differs: {name}')
            if should_accept:
                approved = proposed
    return {
        'domain_size': len(INPUTS),
        'shuffled_control_seeds': list(PASSIVE_SEEDS),
        'extensions': extensions,
        'total_learned': sum(len(evidence['learned_workflows']) for evidence in extensions.values()),
        'total_active_labels': sum(evidence['active_labels'] for evidence in extensions.values()),
        'final_version': verifier.version,
    }


def experiment():
    library = RecipeLibrary(cache=True)
    library.enumerate({'same_name': (2,) * len(INPUTS)})
    changed = {'same_name': (3,) * len(INPUTS)}
    _, updated = library.enumerate(changed)
    _, scratch = RecipeLibrary(cache=False).enumerate(changed)
    if (updated['candidate_tables_sha256'] != scratch['candidate_tables_sha256']
            or updated['reused_recipe_evaluations'] == 0
            or updated['executed_cells'] == 0):
        raise AssertionError('cache did not selectively invalidate a changed dependency')

    cached = replay(cache=True)
    fresh = replay(cache=False)
    comparison = {}
    for name, reused in cached['extensions'].items():
        reconstructed = fresh['extensions'][name]
        for key, value in reused.items():
            if key != 'enumeration' and value != reconstructed[key]:
                raise AssertionError(f'reuse changed the independent result of {name}')
        reuse_stats = reused['enumeration']
        fresh_stats = reconstructed['enumeration']
        if (reuse_stats['executed_cells'] + reuse_stats['reused_cells'] != fresh_stats['executed_cells']
                or any(reuse_stats[k] != fresh_stats[k]
                       for k in ('recipe_count', 'distinct_valid_output_tables', 'invalid_recipes',
                                 'candidate_tables_sha256'))):
            raise AssertionError(f'reuse accounting differs for {name}')
        comparison[name] = {'fresh_executed_cells': fresh_stats['executed_cells'],
                            'cached_executed_cells': reuse_stats['executed_cells'],
                            'identical_recipes_labels_and_admission': True}
    fresh_total = sum(report['fresh_executed_cells'] for report in comparison.values())
    cached_total = sum(report['cached_executed_cells'] for report in comparison.values())
    if cached_total >= fresh_total:
        raise AssertionError('reuse saved no recipe evaluations')
    cached['enumeration_ablation'] = {
        'per_extension': comparison,
        'changed_dependency_invalidation_passed': True,
        'fresh_executed_cells': fresh_total,
        'cached_executed_cells': cached_total,
        'saved_executed_cells': fresh_total - cached_total,
    }
    return cached


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = experiment()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
