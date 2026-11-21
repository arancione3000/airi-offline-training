"""Reproduce all comparisons and qualification checks for the procedural experiment."""
import argparse
from functools import cache
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

from chat_adapter import QUESTIONS, make_assistant
from learn import BUNDLED
from skill_bank import SkillBank
from skill_validation import StableGate
import reference
import transfer as engine

INPUTS = tuple((a, b) for a, b in product(range(-64, 65), range(-16, 17)) if b)
NATIVE = 'native baseline'


@cache
def oracle(policy, workflow, a, b):
    return reference.answer(policy, workflow, a, b)


def assistant_case(workflow, a, b):
    source = f'Valori divisione: {a} e {b}.'
    return [{'role': 'user', 'content': source}, {'role': 'user', 'content': QUESTIONS[workflow]}]


@cache
def best_case_for_known_target(policies, target, workflow):
    """Optimistic bound: choose the fastest path even knowing the hidden target."""
    if engine.resolved(policies, (workflow,)):
        return 0
    successors = {bucket for a, b in engine.PROBES for bucket in engine.partition(policies, workflow, a, b)
                  if target in bucket and len(bucket) < len(policies)}
    return 1 + min(best_case_for_known_target(bucket, target, workflow) for bucket in successors)


def independent(policy):
    counts = {}
    transcript = []
    for workflow in engine.WORKFLOWS:
        candidates = engine.POLICIES
        counts[workflow] = 0
        while probe := engine.next_probe(candidates, (workflow,)):
            answer = oracle(policy, *probe)
            candidates = tuple(p for p in candidates if engine.predict(p, *probe) == answer)
            if not candidates:
                raise AssertionError('reference excluded all local candidates')
            counts[workflow] += 1
            transcript.append([*probe, answer])
    return {
        'observations': transcript,
        'per_workflow': counts,
        'total': sum(counts.values()),
        'optimistic_minimum_total': sum(best_case_for_known_target(engine.POLICIES, policy, workflow)
                                        for workflow in engine.WORKFLOWS),
    }


def check_probe_coverage():
    """Compare policy equivalence on the query pool and on the entire domain."""
    for workflow in engine.WORKFLOWS:
        for pair in combinations(engine.POLICIES, 2):
            equivalent = all(oracle(pair[0], workflow, a, b) == oracle(pair[1], workflow, a, b)
                             for a, b in INPUTS)
            if equivalent != (workflow in engine.resolved(pair, (workflow,))):
                raise AssertionError('query pool fails to witness a relevant difference')


def learn(policy):
    observations = []
    curves = []
    while True:
        candidates = engine.replay(observations)
        curves.append({'queries': len(observations), 'policies_remaining': candidates,
                       'fully_resolved_workflows': engine.resolved(candidates)})
        probe = engine.next_probe(candidates)
        if probe is None:
            break
        observations.append([*probe, oracle(policy, *probe)])
    if (candidates != (policy,) or not 1 <= len(observations) <= 2
            or {row[0] for row in observations} != {'quotient'}):
        raise AssertionError('the experiment requires learning from quotient observations only')
    receipt = engine.issue(observations)
    raw = engine.encode(receipt)
    expected_digest = engine.evidence_digest(receipt)
    return observations, curves, raw, expected_digest


def qualify(policy, certified, observations, bank):
    totals = {'domain_pairs': len(INPUTS), 'all_executions': 0, 'unseen_pair_executions': 0,
              'failures': 0, 'chat_cases': 0}
    used_inputs = {(row[1], row[2]) for row in observations}
    for workflow in engine.WORKFLOWS:
        for a, b in INPUTS:
            totals['all_executions'] += 1
            totals['unseen_pair_executions'] += (a, b) not in used_inputs
            totals['failures'] += certified.execute(workflow, a, b) != oracle(policy, workflow, a, b)
    if totals['failures']:
        raise AssertionError('exhaustive arithmetic qualification failed')
    assistant = make_assistant(lambda _: NATIVE, certified, bank=bank)
    for workflow in engine.WORKFLOWS:
        for a, b in ((-64, 16), (64, -16), (-61, -13), (37, -11), (-19, 6), (0, 16), (3, 7)):
            messages = assistant_case(workflow, a, b)
            actual = assistant.respond(messages)
            if (actual['source'] != 'certified_procedure' or actual['operation'] != workflow
                    or actual['answer'] != oracle(policy, workflow, a, b)
                    or actual['certificate_sha256'] != certified.fingerprint
                    or actual['provenance']['span'] != messages[0]['content']
                    or [x['value'] for x in actual['provenance']['operands']] != [a, b]):
                raise AssertionError('chat result or provenance is incorrect')
            totals['chat_cases'] += 1
    for a, b in ((-65, 1), (65, 1), (0, 0), (0, -17), (0, 17)):
        if assistant.respond(assistant_case('quotient', a, b))['source'] != 'native_model':
            raise AssertionError('chat handled operands outside the accepted domain')
    totals['chat_out_of_domain_abstentions'] = 5
    return totals


def retention(certified, bank):
    native = lambda _: NATIVE
    baseline = make_assistant(native, None, bank=bank)
    candidate = make_assistant(native, certified, bank=bank)
    gate = StableGate()
    historical = [r['messages'] for r in gate.contract['cases']]
    stored = [row['messages'] for expert in bank._read()[0]['entries']
              for row in expert['train'] + expert['qualification']]
    for messages in historical + stored:
        if baseline.respond(messages) != candidate.respond(messages):
            raise AssertionError('certified workflow changed a preexisting route')
    return {'historical_unchanged': len(historical), 'stored_unchanged': len(stored)}


def rejected(raw, trusted_digest):
    try:
        engine.verify(raw, trusted_digest)
    except ValueError:
        return True
    return False


def falsify(certified, raw, digest):
    receipt = json.loads(raw)
    receipt['policies'] = ['truncation']
    modified_policy = rejected(engine.encode(receipt), digest)
    receipt = json.loads(raw)
    receipt['engine_sha256'] = '0' * 64
    other_executor = rejected(engine.encode(receipt), digest)
    receipt = json.loads(raw)
    receipt['evidence']['contract_sha256'] = '0' * 64
    forged_digest = engine.evidence_digest(receipt)
    other_contract = rejected(engine.encode(receipt), forged_digest)
    wrong_evidence_digest = rejected(raw, '0' * 64)
    try:
        engine.issue([['quotient', -7, 3, '195']])
    except ValueError:
        contradictory_input = True
    else:
        contradictory_input = False
    # Change only the target's policy. The valid receipt still describes the
    # source, so the premature transfer would pass replay and fail the target audit.
    false_share_mismatches = sum(certified.execute('remainder', a, b) !=
                                 oracle('truncation', 'remainder', a, b) for a, b in INPUTS)
    # If the trusted producer supplies a coherent transcript for the wrong
    # policy, verification can replay it but cannot detect its provenance error.
    _, _, alleged, alleged_digest = learn('truncation')
    nonmatching = engine.verify(alleged, alleged_digest)
    false_evidence_mismatches = sum(nonmatching.execute('remainder', a, b) !=
                                    oracle('floor', 'remainder', a, b) for a, b in INPUTS)
    checks = {'tampered_policy_rejected': modified_policy, 'different_executor_rejected': other_executor,
              'different_contract_rejected': other_contract, 'wrong_evidence_digest_rejected': wrong_evidence_digest,
              'impossible_observation_rejected': contradictory_input}
    if not all(checks.values()) or false_share_mismatches == 0 or false_evidence_mismatches == 0:
        raise AssertionError('qualification did not exercise the negative control')
    return {**checks, 'one_misbound_workflow_mismatches': false_share_mismatches,
            'coherently_wrong_trusted_workflow_mismatches': false_evidence_mismatches}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    check_probe_coverage()
    bank = SkillBank(BUNDLED)
    report = {'contract_sha256': engine.CONTRACT_SHA256, 'engine_sha256': engine.ENGINE_SHA256,
              'reference_sha256': sha256(Path(reference.__file__).read_bytes()).hexdigest(),
              'bank_sha256': bank.status()['state_sha256'], 'policies': {}}
    initially_resolved = len(engine.resolved(engine.POLICIES))
    for policy in engine.POLICIES:
        observations, curves, raw, digest = learn(policy)
        certified = engine.verify(raw, digest)
        (output / f'{policy}.receipt.json').write_bytes(raw)
        report['policies'][policy] = {
            'shared': {'observations': observations, 'queries': len(observations),
                       'qualified_without_local_observations': len(engine.WORKFLOWS) - 1,
                       'newly_resolved_without_local_observations': len(engine.WORKFLOWS) - 1 - initially_resolved,
                       'learning_curve': curves, 'evidence_sha256': digest, 'receipt_sha256': certified.fingerprint},
            'independent': independent(policy),
            'qualification': qualify(policy, certified, observations, bank),
        }
        if policy == 'floor':
            report['retention'] = retention(certified, bank)
            report['negative_controls'] = falsify(certified, raw, digest)
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
