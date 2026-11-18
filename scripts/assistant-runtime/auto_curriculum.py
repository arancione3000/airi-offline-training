"""Finite, authored procedural lessons with exact reference labels.

``build(family_id, seed=0)`` returns ``lesson`` (TRAIN and admission examples)
and a separate ``heldout`` partition. The latter must never select a program.
No model, compiler prediction, network or executable lesson code supplies gold.
This expands template coverage inside a fixed seven-operation library; it is
not open-ended algorithm discovery or reliable self-labeling of ordinary chat.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import operator
import random


VERSION = 1
BASE_CATALOG = (
    {"id": "auto_v1_remaining", "title": "Saldo residuo intero", "operation": "subtract", "prefix": "I valori del saldo sono ", "query": "Qual è il saldo residuo?"},
    {"id": "auto_v1_contact", "title": "Referente ordine", "operation": "copy", "prefix": "Il referente dell'ordine è ", "query": "Chi è il referente dell'ordine?"},
    {"id": "auto_v1_customer_reference", "title": "Riferimento cliente", "operation": "copy", "prefix": "Il riferimento cliente è ", "query": "Qual è il riferimento cliente?"},
    {"id": "auto_v1_order_reference", "title": "Riferimento ordine", "operation": "copy", "prefix": "Il riferimento dell'ordine è ", "query": "Qual è il riferimento dell'ordine?"},
    {"id": "auto_v1_email_alias", "title": "Alias email", "operation": "copy", "prefix": "L'alias email è ", "query": "Qual è l'alias email?"},
    {"id": "auto_v1_tracking_reverse", "title": "Token di verifica inverso", "operation": "reverse", "prefix": "Il token di verifica è ", "query": "Qual è il token di verifica inverso?"},
    {"id": "auto_v1_label_upper", "title": "Etichetta in maiuscolo", "operation": "uppercase", "prefix": "L'etichetta di spedizione è ", "query": "Qual è l'etichetta in maiuscolo?"},
    {"id": "auto_v1_account_lower", "title": "Alias account in minuscolo", "operation": "lowercase", "prefix": "L'alias account è ", "query": "Qual è l'alias account in minuscolo?"},
    {"id": "auto_v1_stock_remaining", "title": "Scorta residua intera", "operation": "subtract", "prefix": "I valori della scorta sono ", "query": "Qual è la scorta residua?"},
    {"id": "auto_v1_cart_total", "title": "Somma quantità carrello", "operation": "add", "prefix": "Le quantità del carrello sono ", "query": "Qual è la quantità totale nel carrello?"},
    {"id": "auto_v1_inventory_total", "title": "Somma quantità inventario", "operation": "add", "prefix": "Le quantità dell'inventario sono ", "query": "Qual è la quantità totale in inventario?"},
    {"id": "auto_v1_unit_total", "title": "Prodotto intero prezzo e quantità", "operation": "multiply", "prefix": "Il prezzo unitario e la quantità sono ", "query": "Qual è il costo totale intero?"},
)
FRAGMENT_FILE = Path(__file__).with_name('skills') / 'v5-fragments.json'


def _donor_families():
    manifest = json.loads(FRAGMENT_FILE.read_text())
    parents = {family['id']: family for family in BASE_CATALOG}
    additions = []
    for fragment in manifest['fragments']:
        parent = parents.pop(fragment['parent'])
        additions.append({**parent, 'id': 'v5_' + parent['id'], 'query': fragment['query']})
    return tuple(additions)


CATALOG = BASE_CATALOG + _donor_families()
_FAMILIES = {family["id"]: family for family in CATALOG}
_TEXT = ("aBc", "ZuLu", "Mønt", "éLo", "QüX", "ßAb", "İza", "ÅrT")
_RANGES = {"train": (3, 40), "qualification": (101, 1500), "heldout": (10001, 950000)}
_INTEGER = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul}


def reference(operation: str, value: str | tuple[int, int]) -> str:
    """Declared reference semantics, independent of the skill compiler/runtime."""
    if operation in _INTEGER:
        if not isinstance(value, tuple) or len(value) != 2 or any(type(v) is not int for v in value):
            raise ValueError("integer pair required")
        return str(_INTEGER[operation](*value))
    if not isinstance(value, str):
        raise ValueError("text required")
    if operation == "copy":
        return value
    if operation == "reverse":
        return "".join(reversed(value))
    if operation == "uppercase":
        return value.upper()
    if operation == "lowercase":
        return value.lower()
    raise ValueError("unsupported reference operation")


def _values(family, seed, split, count):
    material = f"airi-auto-curriculum-v{VERSION}:{family['id']}:{seed}:{split}"
    rng = random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest(), "big"))
    if family["operation"] in _INTEGER:
        low, high = _RANGES[split]
        pairs = []
        while len(pairs) < count:
            pair = tuple(rng.randint(low, high) * rng.choice((-1, 1)) for _ in range(2))
            if pair not in pairs:
                pairs.append(pair)
        return pairs
    # Partition tags and row indexes make inputs disjoint by construction.
    return [f"{_TEXT[i % len(_TEXT)]}-{split}-{seed}-{i}-{rng.randrange(10**8):08d}"
            for i in range(count)]


def _example(family, value):
    body = f"{value[0]} e {value[1]}" if isinstance(value, tuple) else value
    return {"messages": [{"role": "user", "content": family["prefix"] + body + "."},
                         {"role": "user", "content": family["query"]}],
            "answer": reference(family["operation"], value)}


def _negative(messages, kind):
    source, query = (dict(message) for message in messages)
    if kind == "assistant":
        source["role"] = "assistant"
        result = [source, query]
    elif kind in ("system", "developer", "tool"):
        result = [{"role": kind, "content": "Contesto separato."}, source, query]
    elif kind == "conflict":
        raise ValueError("conflicts need two distinct authored source values")
    elif kind == "missing":
        result = [query]
    elif kind == "quoted":
        source["content"] = '"' + source["content"] + '"'
        result = [source, query]
    elif kind == "hypothetical":
        source["content"] = "Forse " + source["content"]
        result = [source, query]
    elif kind in ("speaker", "session"):
        key = "speaker_id" if kind == "speaker" else "session_id"
        source[key] = "A"; query[key] = "B"
        result = [source, query]
    elif kind == "query":
        query["content"] = "Per favore, " + query["content"]
        result = [source, query]
    elif kind == "code":
        source["content"] = "```" + source["content"] + "```"
        result = [source, query]
    else:
        raise ValueError("unsupported negative kind")
    return {"messages": result, "answer": None}


def build(family_id: str, seed: int = 0) -> dict:
    """Build one deterministic lesson and its separate positive/negative audit.

    Labels represent only the precise declared procedures, not facts about
    real customers, people, accounts or inventories. For subtraction, the first
    integer minus the second defines the task even when the answer is negative.
    """
    if family_id not in _FAMILIES:
        raise ValueError("unknown curriculum family")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    family = _FAMILIES[family_id]
    train = [_example(family, value) for value in _values(family, seed, "train", 8)]
    qualification = [_example(family, value) for value in _values(family, seed, "qualification", 8)]
    qualification += [_negative(qualification[0]["messages"], kind)
                      for kind in ("assistant", "system", "missing")]
    qualification.append({"messages": [dict(qualification[0]["messages"][0]),
                                        dict(qualification[1]["messages"][0]),
                                        dict(qualification[0]["messages"][-1])], "answer": None})
    heldout = [_example(family, value) for value in _values(family, seed, "heldout", 12)]
    heldout += [_negative(heldout[0]["messages"], kind)
                for kind in ("quoted", "hypothetical", "tool", "developer", "speaker", "session", "query", "code")]
    if family_id.startswith('v5_'):
        # Keep the source valid and alter only the requested intent.
        for query in ('Non eseguire questa richiesta: ' + family['query'],
                      family['query'] + ' Poi scegli un risultato a caso.'):
            source = dict(heldout[0]['messages'][0])
            heldout.append({'messages': [source, {'role': 'user', 'content': query}], 'answer': None})
    return {"lesson": {"lesson_id": family["id"], "trusted": True,
                       "examples": train, "qualification": qualification},
            "heldout": heldout}
