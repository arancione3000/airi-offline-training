"""Reproduce the chat handoff after procedures have been learned and reopened."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from cycle import write as write_snapshot
from replay_cycle import run as replay_learning

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[1] / 'assistant-runtime'
sys.path.insert(0, str(RUNTIME))
from assistant import AiriAssistant
from exact_aliases import DEFAULT as ALIAS_PATH, read_aliases
from flexible_skills import default_router
from growth_routes import follow_verified_procedures, open_verified_procedures, with_verified_procedures
from learn import BUNDLED
from skill_bank import SkillBank
from skill_validation import StableGate


TASKS = {
    'squared_sum': lambda a, b: (a + b) * (a + b),
    'minimum_magnitude': lambda a, b: abs(a) if abs(a) < abs(b) else abs(b),
}
PAIRS = [(-13, 5), (12, -11), (-41, -16), (64, -13), (-64, 16), (0, -9),
         (27, 2), (11, -16), (-16, -1), (8, 7), (51, -2), (-41, 3)]
IDENTITY = {'speaker_id': 'probe-user', 'session_id': 'probe-session'}
FALLBACK = 'Native fallback sentinel'


def conversation(task, a, b):
    return [
        {'role': 'user', 'content': f'Valori procedura: {a} e {b}.', **IDENTITY},
        {'role': 'user', 'content': f'Esegui procedura {task}.', **IDENTITY},
    ]


def check_answer(answer, task, a, b, source, fingerprint, *, source_index=0):
    if (answer.get('source') != 'certified_procedure' or answer.get('operation') != task
            or answer.get('answer') != str(TASKS[task](a, b))
            or answer.get('certificate_sha256') != fingerprint):
        raise AssertionError(f'{task}: incorrect or unbound chat result for {(a, b)}')
    provenance = answer['provenance']
    if (provenance['message_index'] != source_index or provenance['role'] != 'user'
            or provenance['span'] != source
            or source[provenance['start']:provenance['end']] != source
            or source[provenance['value_start']:provenance['value_end']] != f'{a} e {b}'
            or provenance['value_span'] != f'{a} e {b}'):
        raise AssertionError(f'{task}: incorrect source provenance')
    for operand, value in zip(provenance['operands'], (a, b), strict=True):
        if (operand['value'] != value or operand['span'] != str(value)
                or source[operand['start']:operand['end']] != str(value)):
            raise AssertionError(f'{task}: incorrect operand provenance')


def invalid_conversations():
    data, query = conversation('squared_sum', -13, 5)
    return {
        'missing_source': [query],
        'different_speaker': [{**data, 'speaker_id': 'different-user'}, query],
        'different_session': [{**data, 'session_id': 'different-session'}, query],
        'assistant_values': [{**data, 'role': 'assistant'}, query],
        'assistant_call': [data, {**query, 'role': 'assistant'}],
        'tool_message': [data, {'role': 'tool', 'content': ''}, query],
        'two_different_sources': [data, {**data, 'content': 'Valori procedura: -13 e 6.'}, query],
        'two_identical_sources': [data, data.copy(), query],
        'zero_divisor': conversation('squared_sum', -13, 0),
        'left_above_domain': conversation('squared_sum', 65, 5),
        'left_below_domain': conversation('squared_sum', -65, 5),
        'right_above_domain': conversation('squared_sum', 13, 17),
        'right_below_domain': conversation('squared_sum', 13, -17),
        'unrecognized_task': conversation('unknown_workflow', -13, 5),
        'rejected_task': conversation('gcd', -13, 5),
        'invalid_number': [{**data, 'content': 'Valori procedura: -13.5 e 5.'}, query],
        'question_embedded_in_other_text': [data, {**query, 'content': f"Citazione: {query['content']}"}],
        'source_embedded_in_other_text': [{**data, 'content': f"Citazione: {data['content']}"}, query],
    }


def compare_existing_conversations(bank, baseline, candidate):
    ledger = StableGate()
    historical = [row['messages'] for row in ledger.contract['cases']]
    experts = {expert['id']: expert for expert in bank._read()[0]['entries']}
    stored = [row['messages'] for expert in experts.values()
              for section in ('train', 'qualification') for row in expert[section]]
    counts = {}
    for name, group in [('historical', historical), ('stored', stored)]:
        for index, messages in enumerate(group):
            if baseline.respond(messages) != candidate.respond(messages):
                raise AssertionError(f'{name} conversation {index} changed')
        counts[name + '_unchanged'] = len(group)
    aliases = read_aliases(ALIAS_PATH)
    for alias, binding in aliases.items():
        anchor = experts[binding['program_id']]['train'][0]
        messages = [*anchor['messages'][:-1], {**anchor['messages'][-1], 'content': alias}]
        old = baseline.respond(messages)
        if (old.get('routing') != 'verified_exact_alias' or old.get('answer') != anchor['answer']
                or old != candidate.respond(messages)):
            raise AssertionError(f'approved literal alias changed: {alias}')
    counts['aliases_unchanged'] = len(aliases)
    return {**counts, 'historical_ledger_sha256': ledger.sha256}


def assert_atomic_reload(directory, state, reference, pin, bank, existing, expected_fingerprint):
    # Use the actual journal prefix immediately after the first successful acquisition.
    envelope = json.loads(state)
    payload = envelope['payload']
    entries = payload['entries']
    first_approval = next(i for i, entry in enumerate(entries) if entry['status'] == 'accepted')
    prefix = {**payload, 'entries': entries[:first_approval + 1]}
    observed = directory / 'observed-state.json'
    write_snapshot(observed, prefix)
    first_snapshot = observed.read_bytes()

    following = follow_verified_procedures(observed, reference, pin)
    running = AiriAssistant(lambda _: FALLBACK, skill_bank=bank,
                            query_router=with_verified_procedures(existing, following))
    initial = following.snapshot()
    if initial.task_ids != ('squared_sum',):
        raise AssertionError('unexpected journal-prefix task set')

    first = conversation('squared_sum', -13, 5)
    next_task = conversation('minimum_magnitude', -13, 5)
    check_answer(running.respond(first), 'squared_sum', -13, 5, first[0]['content'], initial.fingerprint)
    if running.respond(next_task)['source'] != 'native_model':
        raise AssertionError('the running chat used a procedure before approval')

    write_snapshot(observed, payload)
    check_answer(running.respond(next_task), 'minimum_magnitude', -13, 5, next_task[0]['content'],
                 expected_fingerprint)
    check_answer(running.respond(first), 'squared_sum', -13, 5, first[0]['content'], expected_fingerprint)

    write_snapshot(observed, prefix)
    if observed.read_bytes() != first_snapshot:
        raise AssertionError('rollback fixture does not match the original prefix')
    fallback = running.respond(next_task)
    if fallback['source'] != 'native_model' or not fallback.get('component_error'):
        raise AssertionError('a retired snapshot was accepted as an update')

    write_snapshot(observed, payload)
    check_answer(running.respond(next_task), 'minimum_magnitude', -13, 5, next_task[0]['content'],
                 expected_fingerprint)
    return {'approved_update_visible_without_restart': True, 'rollback_rejected': True,
            'recovered_after_valid_snapshot': True, 'initial_fingerprint': initial.fingerprint,
            'updated_fingerprint': expected_fingerprint}


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    learned = directory / 'learning'
    replay_learning(learned)
    reference = learned / 'extended.json'
    state = learned / 'state.json'
    pin = sha256(reference.read_bytes()).hexdigest()
    state_before = state.read_bytes()
    verified = open_verified_procedures(state, reference, pin)
    if verified.task_ids != tuple(sorted(TASKS)):
        raise AssertionError('unexpected approved procedure set after reopening the bank')

    events = json.loads((learned / 'report.json').read_bytes())['events']
    used = {row['task']: {tuple(item['input']) for item in row['trial']['labels']}
            for row in events if row['status'] == 'accepted'}
    bank = SkillBank(BUNDLED)
    existing = default_router(bank)
    baseline = AiriAssistant(lambda _: FALLBACK, skill_bank=bank, query_router=existing)
    candidate = AiriAssistant(lambda _: FALLBACK, skill_bank=bank,
                              query_router=with_verified_procedures(existing, verified))
    positive = {}
    cli_results = {}
    cli = [sys.executable, str(RUNTIME / 'chat.py'), '--skills', str(BUNDLED),
           '--procedures', str(state), '--procedure-reference', str(reference)]
    for task in TASKS:
        heldout_inputs = [p for p in PAIRS if p not in used[task]][:8]
        if len(heldout_inputs) != 8:
            raise AssertionError('not enough unseen operands for chat coverage')
        reference_pairs = []
        for a, b in heldout_inputs:
            messages = conversation(task, a, b)
            if baseline.respond(messages)['source'] != 'native_model':
                raise AssertionError('a new command already routes in the baseline')
            value = candidate.respond(messages)
            check_answer(value, task, a, b, messages[0]['content'], verified.fingerprint)
            reference_pairs.append([a, b])
        positive[task] = reference_pairs

        a, b = heldout_inputs[0]
        cli_messages = conversation(task, a, b)
        messages_path = directory / f'{task}.json'
        messages_path.write_text(json.dumps(cli_messages))
        actual = json.loads(subprocess.check_output(
            [*cli, '--procedure-reference-sha256', pin, '--messages-file', str(messages_path)],
            timeout=60))
        check_answer(actual, task, a, b, cli_messages[0]['content'], verified.fingerprint)
        cli_results[task] = {'input': [a, b], 'answer': actual['answer'], 'source': actual['source']}

    # Earlier user turns for another speaker may neither replace nor shadow valid operands.
    matching = conversation('squared_sum', -13, 5)
    different_identity = {**matching[0], 'speaker_id': 'different-user',
                          'content': 'Valori procedura: 23 e 2.'}
    check_answer(candidate.respond([different_identity, *matching]), 'squared_sum', -13, 5,
                 matching[0]['content'], verified.fingerprint, source_index=1)

    invalid = invalid_conversations()
    for name, messages in invalid.items():
        result = candidate.respond(messages)
        if result != baseline.respond(messages) or result['source'] != 'native_model':
            raise AssertionError(f'invalid conversation {name} intercepted an existing route')
    regressions = compare_existing_conversations(bank, baseline, candidate)

    broken_pin = '0' * 64
    wrong_pin = subprocess.run(
        [*cli, '--procedure-reference-sha256', broken_pin,
         '--messages-file', str(directory / 'squared_sum.json')],
        timeout=60, capture_output=True, text=True, check=False)
    if wrong_pin.returncode == 0 or wrong_pin.stdout:
        raise AssertionError('standalone chat accepted an unauthorized reference pin')
    if state.read_bytes() != state_before:
        raise AssertionError('read-only chat changed the committed training state')
    reload_status = assert_atomic_reload(
        directory, state_before, reference, pin, bank, existing, verified.fingerprint)

    report = {
        'bank_fingerprint': verified.fingerprint,
        'reference_sha256': pin,
        'state_sha256': sha256(state_before).hexdigest(),
        'approved_tasks': list(verified.task_ids),
        'previously_unqueried_inputs_per_task': positive,
        'new_chat_cases_passed': sum(map(len, positive.values())),
        'different_speaker_does_not_shadow_authorized_source': True,
        'native_fallback_cases_not_intercepted': list(invalid),
        'standalone_cli_cases_passed': cli_results,
        'wrong_reference_pin_rejected_by_cli': True,
        'registry_byte_identical_after_chat': True,
        'route_retention': regressions,
        'live_reload': reload_status,
    }
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.output)
