"""Larger human-written language corpus; pinned after the first download."""
import hashlib
import json
from pathlib import Path

TRAIN_PER_LANGUAGE = 20000
HELDOUT_PER_LANGUAGE = 128


def normal(text):
    return ' '.join(text.casefold().split())


def select(rows, language, blocked, tokenizer, train_count=TRAIN_PER_LANGUAGE,
           heldout_count=HELDOUT_PER_LANGUAGE):
    seen = set()
    chosen = []
    for sentence_id, text, author in rows:
        key = normal(text)
        if key in blocked or key in seen:
            continue
        ids = tokenizer.encode(text, bos=True, eos=True)
        if not 8 <= len(ids) <= 512:
            continue
        seen.add(key)
        chosen.append({'id': sentence_id, 'language': language, 'text': text,
                       'author': author, 'tokens': len(ids)-1,
                       'sha256': hashlib.sha256(text.encode()).hexdigest()})
        if len(chosen) == train_count + heldout_count:
            break
    if len(chosen) != train_count + heldout_count:
        raise ValueError('insufficient genuinely new corpus: ' + language)
    return chosen[heldout_count:], chosen[:heldout_count]


def prepare(output, tokenizer, data, parent):
    from generalist_lm import bootstrap_data as native
    from generalist_lm.phase5_diagnostics import protected_bootstrap_texts
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    blocked = set(protected_bootstrap_texts())
    conversation_test = Path(data).parent/'CONVERSATION_TEST.json'
    if conversation_test.exists():
        blocked.update(normal(p) for p in json.loads(conversation_test.read_text())['prompts'])
    # Reject overlap with existing native train/validation and all dialogue splits.
    for path in [*Path(parent).rglob('*.jsonl'), *Path(data).rglob('*.jsonl')]:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('text'):
                blocked.add(normal(row['text']))
            for message in row.get('messages', []):
                blocked.add(normal(message['content']))
    for doc in native.load_bootstrap_replay(parent).documents:
        blocked.add(normal(doc.text))
    train, heldout, sources = [], [], []
    for source in native.SOURCES:
        if not source['id'].startswith('tatoeba-'):
            continue
        raw, pin = native._download(source, output / 'download')
        selected, test = select(native._parse_tatoeba(source, raw), source['language'],
                                blocked, tokenizer)
        train.extend(selected)
        heldout.extend(test)
        sources.append({**source, **pin, 'attribution': 'Tatoeba contributors; sentence authors retained per row'})
    if {x['language'] for x in train} != {'it', 'en'}:
        raise ValueError('both languages required')
    for name, rows in [('CORPUS_TRAIN.jsonl', train), ('CORPUS_HELDOUT.jsonl', heldout)]:
        (output / name).write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True)+'\n' for x in rows))
    manifest = {'sources': sources, 'tokenizer_digest': tokenizer.digest,
                'train_rows': {lang: sum(x['language']==lang for x in train) for lang in ['it','en']},
                'train_tokens': {lang: sum(x['tokens'] for x in train if x['language']==lang) for lang in ['it','en']},
                'heldout_rows': 256, 'blocked_existing_texts': len(blocked),
                'novelty': 'Exact normalized text exclusion from available native files and dialogue splits; no claim of unseen semantics',
                'files': {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                          for name in ['CORPUS_TRAIN.jsonl','CORPUS_HELDOUT.jsonl']}}
    (output/'CORPUS_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def verify(output, tokenizer_digest):
    output = Path(output)
    manifest = json.loads((output/'CORPUS_MANIFEST.json').read_text())
    if manifest['tokenizer_digest'] != tokenizer_digest:
        raise ValueError('corpus tokenizer mismatch')
    if set(manifest['files']) != {'CORPUS_TRAIN.jsonl','CORPUS_HELDOUT.jsonl'}:
        raise ValueError('corpus file identity mismatch')
    for name, expected in manifest['files'].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest() != expected:
            raise ValueError('corpus SHA mismatch')
    return manifest
