"""Deterministic TRAIN-only context copying with broad, paired target values.

The old fixed-name curriculum and every held-out dataset remain separate.
Gold answers are the literal value inserted into the user-provided context;
neither a model nor an evaluation response supplies an acquisition label.
"""
from collections import Counter
import copy
import hashlib
import json
import random
import string

from composition_data import PREVIOUS_SEALED_VALUES, PROTECTED_VALUES, SEALED_VALUES
from curriculum import norm


VERSION = 'context-acquisition-v1'
DEFAULT_SEED = 20261103
MAX_PAIRS = 1024
TASKS = ('name', 'location', 'topic', 'correction')
LANGUAGES = ('it', 'en')
WORDS = ('vento', 'tiglio', 'prato', 'vetro', 'lago', 'ramo', 'pietra', 'stella',
         'wind', 'linden', 'grass', 'glass', 'lake', 'branch', 'stone', 'star')

# New TRAIN wording is authored here; it is not derived from held-out prompts.
TEMPLATES = {
    'it': {
        'name': (('Il nome operativo da conservare è «{}».', 'Restituisci soltanto il nome operativo registrato.'),
                 ('Questa persona usa il nome «{}» in questa conversazione.', 'Riporta il nome usato dalla persona, senza commenti.'),
                 ('Per il nostro registro il nominativo è «{}».', 'Quale nominativo contiene il nostro registro? Solo il nominativo.')),
        'location': (('Il pacco si trova nel locale identificato da «{}».', 'Riporta soltanto l’identificativo del locale del pacco.'),
                     ('La destinazione del contenitore è il posto «{}».', 'Indica il posto assegnato al contenitore, senza spiegazioni.'),
                     ('Ho messo il documento nello spazio chiamato «{}».', 'Come si chiama lo spazio del documento? Solo il suo nome.')),
        'topic': (('L’argomento scelto per questa chat è «{}».', 'Riporta soltanto l’argomento scelto per la chat.'),
                  ('La discussione riguarda il progetto etichettato «{}».', 'Qual è l’etichetta del progetto della discussione?'),
                  ('Abbiamo deciso di parlare del tema «{}».', 'Scrivi esclusivamente il tema che abbiamo deciso.')),
        'correction': (('La vecchia etichetta era Orivo; quella aggiornata è «{}».', 'Restituisci soltanto l’etichetta aggiornata.'),
                       ('Annulla il riferimento Orivo e usa d’ora in poi «{}».', 'Quale riferimento devo usare ora? Solo il riferimento.'),
                       ('Prima avevo scritto Orivo. La rettifica indica «{}».', 'Riporta il valore indicato dalla rettifica, senza commenti.')),
    },
    'en': {
        'name': (('The working name to retain is «{}».', 'Return only the recorded working name.'),
                 ('This person uses the name «{}» in this conversation.', 'Give the name used by the person, without commentary.'),
                 ('The name in our register is «{}».', 'Which name is in our register? Give just the name.')),
        'location': (('The parcel is in the area identified by «{}».', 'Return only the identifier of the parcel’s area.'),
                     ('The container’s destination is the place «{}».', 'Give the place assigned to the container without explanation.'),
                     ('I put the document in the space named «{}».', 'What is the name of the document’s space? Give only its name.')),
        'topic': (('The subject chosen for this chat is «{}».', 'Return only the subject chosen for this chat.'),
                  ('The discussion concerns the project labelled «{}».', 'What is the label of the project being discussed?'),
                  ('We decided to discuss the theme «{}».', 'Write exclusively the theme we decided to discuss.')),
        'correction': (('The old label was Orivo; the updated label is «{}».', 'Return only the updated label.'),
                       ('Cancel the reference Orivo and use «{}» from now on.', 'Which reference should I use now? Give just the reference.'),
                       ('I previously wrote Orivo. The correction specifies «{}».', 'Give the value specified by the correction without commentary.')),
    },
}

ACKNOWLEDGEMENTS = {
    'it': ('Dato registrato.', 'Terrò presente questo dato.', 'Ho annotato il riferimento.'),
    'en': ('The detail is recorded.', 'I will keep that detail in mind.', 'I have noted the reference.'),
}
DISTRACTIONS = {
    'it': (('Facciamo una breve pausa prima della domanda.', 'Possiamo riprendere quando vuoi.'),
           ('La mia prossima richiesta sarà breve.', 'Puoi formularla quando preferisci.')),
    'en': (('Let us take a short pause before the question.', 'We can resume whenever you wish.'),
           ('My next request will be brief.', 'You can ask it whenever you prefer.')),
}


def _value(rng, bucket, initial, occurrence):
    if bucket == 0:
        # Include all A-Z initials and short sequences, rather than eight names.
        length = 1 + occurrence % 3
        return initial + ''.join(rng.choices(string.ascii_lowercase, k=length - 1))
    if bucket == 1:
        length = rng.randint(4, 8)
        alphabet = ('aeiou', 'bcdfghjklmnpqrstvwxyz')
        return initial + ''.join(rng.choice(alphabet[i % 2]) for i in range(length - 1))
    if bucket == 2:
        length = rng.randint(4, 12)
        # Explicit digit coverage in every code; remaining symbols vary.
        return initial + rng.choice(string.digits) + ''.join(
            rng.choices(string.ascii_lowercase + string.digits, k=length - 2))
    words = rng.sample(WORDS, 2 + occurrence % 2)
    return initial + words[0] + ' ' + ' '.join(words[1:])


def _messages(language, task, template, value, ack_index, history):
    introduction, question = TEMPLATES[language][task][template]
    rows = [{'role': 'user', 'content': introduction.format(value)},
            {'role': 'assistant', 'content': ACKNOWLEDGEMENTS[language][ack_index]}]
    for user, assistant in DISTRACTIONS[language][:history]:
        rows.extend([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}])
    rows.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': value}])
    return rows


def build_context_acquisition(blocked=(), reserved_values=(), *, seed=DEFAULT_SEED, max_pairs=MAX_PAIRS):
    """Return at most 2,048 TRAIN rows in deterministic counterfactual pairs.

    ``blocked`` contains normalized exact message texts to exclude, and
    ``reserved_values`` adds whole output values to the permanent reservation
    catalog. Neither collection is ever used as a source of positive labels.
    Each pair has the same template, acknowledgement, distractors and question;
    only its source value and literal gold answer differ.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 63:
        raise ValueError('invalid deterministic context acquisition seed')
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or not 1 <= max_pairs <= MAX_PAIRS:
        raise ValueError('invalid bounded context acquisition size')
    blocked = {norm(str(x)) for x in blocked}
    reserved = {norm(str(x)) for x in tuple(PROTECTED_VALUES) + tuple(SEALED_VALUES)
                + tuple(PREVIOUS_SEALED_VALUES) + tuple(reserved_values)} | blocked | {'orivo'}
    rng = random.Random(seed)
    rows, seen = [], set()
    for group_index in range(max_pairs):
        language = LANGUAGES[group_index % 2]
        task = TASKS[(group_index // 2) % len(TASKS)]
        bucket = (group_index // 8) % 4
        occurrence = group_index // 32
        initial_index = (occurrence + (group_index % 8) * 3) % 26
        group = f'{VERSION}:{seed}:{group_index:04d}'
        accepted = None
        for attempt in range(128):
            template = (occurrence + attempt) % 3
            ack = ((occurrence // 3) + attempt) % 3
            history = ((occurrence // 9) + attempt) % 3
            pair = []
            for member, shift in enumerate((0, 13)):
                initial = string.ascii_uppercase[(initial_index + shift + attempt) % 26]
                value = _value(rng, bucket, initial, occurrence + member)
                if norm(value) in reserved:
                    break
                messages = _messages(language, task, template, value, ack, history)
                if any(norm(m['content']) in blocked for m in messages):
                    break
                key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
                if key in seen:
                    break
                pair.append({'messages': messages, 'expected': value, 'language': language,
                             'family': 'context-' + task, 'stage': 0, 'pair_id': group,
                             'acquisition_group': group, 'counterfactual_index': member,
                             'acquisition_version': VERSION, 'length_bucket': bucket,
                             'history_exchanges': history, 'template_index': template,
                             'provenance': 'deterministic TRAIN-only literal context copy; paired values; no model or evaluation labels'})
            if len(pair) == 2 and pair[0]['expected'] != pair[1]['expected']:
                accepted = pair
                break
        if accepted is None:
            raise ValueError('blocked/reserved texts leave no bounded context acquisition pair')
        for row in accepted:
            seen.add(json.dumps(row['messages'], ensure_ascii=False, sort_keys=True))
        rows.extend(accepted)
    return rows


def acquisition_audit(rows):
    """Summarize a TRAIN pool; this is not a capability or validation score."""
    groups = Counter(row['acquisition_group'] for row in rows)
    values = {row['expected'] for row in rows}
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return {'version': VERSION, 'rows': len(rows), 'counterfactual_pairs': len(groups),
            'all_groups_have_two_rows': all(count == 2 for count in groups.values()),
            'unique_values': len(values), 'initials': sorted({value[0] for value in values}),
            'languages': dict(sorted(Counter(row['language'] for row in rows).items())),
            'tasks': dict(sorted(Counter(row['family'] for row in rows).items())),
            'length_buckets': dict(sorted(Counter(row['length_bucket'] for row in rows).items())),
            'sha256': hashlib.sha256(canonical.encode()).hexdigest(),
            'scope': 'TRAIN-only deterministic acquisition inventory; no measured model improvement'}


def context_skeletons(rows):
    """Index original TRAIN templates without adding any new wording.

    The entire history remains part of the skeleton. Only the literal source
    value in the first user message and the final answer are variable.
    """
    grouped = {}
    for row in rows:
        if row.get('language') not in LANGUAGES or row.get('family') not in {
                'context-' + task for task in TASKS}:
            continue
        messages = copy.deepcopy(row['messages'])
        value = row['expected']
        if (not value or messages[0]['role'] != 'user'
                or messages[-1] != {'role': 'assistant', 'content': value}
                or messages[0]['content'].count(value) != 1):
            raise ValueError('invalid original context TRAIN skeleton')
        messages[0]['content'] = messages[0]['content'].replace(value, '<VALUE>')
        messages[-1]['content'] = '<VALUE>'
        key = (row['language'], row['family'],
               json.dumps(messages, ensure_ascii=False, sort_keys=True))
        grouped.setdefault(key, []).append(row)
    result = {}
    for (language, family, _), members in sorted(grouped.items()):
        result.setdefault((language, family), []).append(members)
    return result


def context_value_pairs(rows):
    """Index the deterministic broad pool by language/family, keeping pairs."""
    groups = {}
    for row in rows:
        groups.setdefault(row['acquisition_group'], []).append(row)
    result = {}
    for members in groups.values():
        if (len(members) != 2 or members[0]['expected'] == members[1]['expected']
                or len({(r['language'], r['family']) for r in members}) != 1):
            raise ValueError('invalid counterfactual value pair')
        result.setdefault((members[0]['language'], members[0]['family']), []).append(members)
    return result


def matched_context_pair(source, values, *, blocked=(), reserved_values=(),
                         protected_histories=()):
    """Transplant broad TRAIN values into one unchanged original TRAIN history.

    Shared short acknowledgements/questions are already original TRAIN data.
    Full protected histories, source messages and held-out output values remain
    forbidden. This is the exact value-only intervention measured by the probe.
    """
    from success_bank import prompt_key
    if not context_skeletons([source]):
        raise ValueError('a matched pair requires an original context TRAIN row')
    if (len(values) != 2 or values[0]['expected'] == values[1]['expected']
            or values[0]['acquisition_group'] != values[1]['acquisition_group']
            or any((r['language'], r['family']) != (source['language'], source['family'])
                   for r in values)):
        raise ValueError('counterfactual values do not match the TRAIN template')
    blocked = {norm(str(x)) for x in blocked}
    reserved = {norm(str(x)) for x in tuple(PROTECTED_VALUES) + tuple(SEALED_VALUES)
                + tuple(PREVIOUS_SEALED_VALUES) + tuple(reserved_values)} | blocked
    protected_histories = set(protected_histories)
    pair = []
    for value_row in values:
        value = value_row['expected']
        if norm(value) in reserved:
            raise ValueError('reserved held-out value in TRAIN context pair')
        row = copy.deepcopy(source)
        row['messages'][0]['content'] = row['messages'][0]['content'].replace(source['expected'], value)
        row['messages'][-1]['content'] = value
        row.update(expected=value, acquisition_group=value_row['acquisition_group'],
                   acquisition_version=VERSION,
                   counterfactual_index=value_row.get('counterfactual_index', len(pair)),
                   length_bucket=value_row['length_bucket'],
                   provenance='TRAIN-only broad literal value transplanted into unchanged original TRAIN template')
        if norm(row['messages'][0]['content']) in blocked or prompt_key(row) in protected_histories:
            raise ValueError('protected source/history in TRAIN context pair')
        pair.append(row)
    return pair


def paired_batch(batches):
    """Right-pad exactly two complete causal examples; ignore padding labels."""
    import torch
    from generalist_lm.tokenizer import PAD
    if len(batches) != 2:
        raise ValueError('one context update requires exactly two examples')
    for ids, labels in batches:
        if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != labels.shape
                or ids.dtype != torch.long or labels.dtype != torch.long
                or ids.device.type != 'cpu' or labels.device.type != 'cpu'
                or ids.shape[1] < 2 or not bool((labels[:, 1:] != -100).any())):
            raise ValueError('invalid complete paired context example')
    width = max(ids.shape[1] for ids, _ in batches)
    ids = torch.full((2, width), PAD, dtype=torch.long)
    labels = torch.full((2, width), -100, dtype=torch.long)
    for index, (source_ids, source_labels) in enumerate(batches):
        ids[index, :source_ids.shape[1]] = source_ids[0]
        labels[index, :source_labels.shape[1]] = source_labels[0]
    return ids, labels


def acquisition_curriculum_digest(base_rows, config=None, values=()):
    """Preserve the legacy digest by default; bind opt-in data AND recipe."""
    from curriculum import digest
    if config is None:
        if values:
            raise ValueError('acquisition values require an explicit recipe')
        return digest(base_rows)
    payload = {'base_rows': base_rows, 'context_acquisition': dict(config),
               'context_acquisition_values': list(values)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_acquisition_configuration(config):
    """Bound the explicit, versioned matched-TRAIN acquisition recipe."""
    if (not isinstance(config, dict)
            or set(config) != {'version', 'seed', 'max_pairs', 'template_policy', 'paired_batch'}
            or not isinstance(config['version'], int) or isinstance(config['version'], bool)
            or config['version'] != 1 or config['template_policy'] != 'matched_train'
            or config['paired_batch'] is not True):
        raise ValueError('invalid matched TRAIN acquisition configuration')
    seed, pairs = config['seed'], config['max_pairs']
    if (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 63
            or not isinstance(pairs, int) or isinstance(pairs, bool) or not 8 <= pairs <= MAX_PAIRS):
        raise ValueError('invalid bounded matched TRAIN acquisition pool')
    return config
