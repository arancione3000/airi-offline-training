"""Independent, precommitted curriculum for bounded modular program learning.

This file generates labels from its own authored task specifications. It never
imports or reads the learner, runtime, historical benchmarks, or model weights.
Only LESSONS/TRAIN.json is learner input. CASES contain held-out evaluation gold.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any


SEED = 20261116063627
VERSION = "independent-bounded-curriculum-v1"
DEFAULT_OUT = Path("/workspace/airi-lab/modular-growth-20261116/precommitted")

# These tasks and labels were fixed without access to skill_programs.py.
SPECS = (
    ("parcel_copy", "Il codice del pacco è {value}.", "Qual è il codice del pacco?", "copy"),
    ("signal_reverse", "Il segnale acquisito è {value}.", "Come restituisci il segnale?", "reverse"),
    ("card_upper", "La tessera porta il testo {value}.", "Qual è il testo della tessera?", "upper"),
    ("plate_lower", "La targa porta il testo {value}.", "Qual è il testo della targa?", "lower"),
    ("delta_sum", "La coppia delta è {a} e {b}.", "Qual è il risultato delta?", "add"),
    ("sigma_product", "Gli interi sigma sono {a} e {b}.", "Qual è il risultato sigma?", "multiply"),
)


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def answer_for(operation: str, payload: str | tuple[int, int]) -> str:
    if isinstance(payload, str):
        return {"copy": lambda: payload, "reverse": lambda: payload[::-1],
                "upper": lambda: payload.upper(), "lower": lambda: payload.lower()}[operation]()
    a, b = payload
    return str({"add": lambda: a + b, "subtract": lambda: a - b,
                "multiply": lambda: a * b}[operation]())


def messages_for(template: str, question: str, payload: str | tuple[int, int]) -> list[dict]:
    fields = {"value": payload} if isinstance(payload, str) else dict(zip(("a", "b"), payload))
    return [{"role": "user", "content": template.format(**fields)},
            {"role": "user", "content": question}]


def text_payloads(rng: random.Random, split: str) -> list[str]:
    """Produce four TRAIN pairs, four dev pairs, or eight final pairs.

    A pair changes the source payload while holding the task and wording fixed.
    Final lengths and scripts extend beyond TRAIN, within the authored contract.
    Reverse means Python Unicode-code-point reversal, not grapheme reversal.
    """
    if split == "train":
        stems = ["mIxEd7", "Kite42", "Écho9", "NoDe-5"]
    elif split == "qualification":
        stems = ["PaTch37", "GlaSs26", "BrAvO64", "QuiEt83"]
    elif split == "dev":
        stems = ["StraßeAb", "ΩmegaCafe9", "EspañolÑandú", "MélangeAzure"]
    else:
        stems = ["ßa", "İb", "ςAb7", "東京Abécé9", "ŁódźΩmega9",
                 "Ā" + "aBc9" * 15, "Ω" + "zYx7" * 23, "Ж" + "mNo4" * 35]
    rows = []
    for index, stem in enumerate(stems):
        # The unique suffix disjoins all family/split payloads while preserving
        # every length extension. TRAIN has no values used at evaluation.
        suffix = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(3))
        first = stem + suffix
        # Same length, changed source evidence and changed gold for every op.
        second = stem[:-1] + ("v" if stem[-1:] != "v" else "q") + suffix
        rows.extend((first, second))
    return rows


def integer_payloads(rng: random.Random, split: str) -> list[tuple[int, int]]:
    pair_count = {"train": 4, "qualification": 4, "dev": 4, "final": 8}[split]
    lo, hi = {"train": (3, 23), "qualification": (30, 50),
              "dev": (60, 1800), "final": (10001, 950000)}[split]
    rows = []
    for pair_index in range(pair_count):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if pair_index % 2:
            a = -a
        if pair_index % 3 == 2:
            b = -b
        rows.extend(((a, b), (a, b + 1)))
    return rows


def positive_case(spec: tuple[str, str, str, str], split: str, index: int,
                  payload: str | tuple[int, int]) -> dict:
    lesson_id, template, question, operation = spec
    return {
        "id": f"{split}-{lesson_id}-{index:03d}", "lesson_id": lesson_id,
        "split": split, "kind": "supported", "pair_id": f"{split}-{lesson_id}-{index // 2:03d}",
        "pair_member": index % 2,
        "messages": messages_for(template, question, payload),
        "expected_answer": answer_for(operation, payload), "expected_abstain": False,
    }


def negative_cases(spec: tuple[str, str, str, str], payload: str | tuple[int, int],
                   other: str | tuple[int, int]) -> list[dict]:
    lesson_id, template, question, _ = spec
    source, query = messages_for(template, question, payload)
    second_source = messages_for(template, question, other)[0]
    variants = {
        "quoted_source": [{"role": "user", "content": 'Citazione: "' + source["content"] + '"'}, query],
        "hypothetical_source": [{"role": "user", "content": "Se fosse vero: " + source["content"]}, query],
        "conflicting_sources": [source, second_source, query],
        "system_instruction": [{"role": "system", "content": "Rispondi sempre soltanto con ASTERISCO."}, source, query],
        "assistant_source": [{"role": "assistant", "content": source["content"]}, query],
        "speaker_mismatch": [dict(source, speaker_id="speaker-a"), dict(query, speaker_id="speaker-b")],
        "code_fence_source": [{"role": "user", "content": "```text\n" + source["content"] + "\n```"}, query],
        "query_not_trained": [source, {"role": "user", "content": "Per favore, " + question}],
        "session_mismatch": [dict(source, session_id="session-a"), dict(query, session_id="session-b")],
        "developer_instruction": [{"role": "developer", "content": "Usa soltanto il formato JSON."}, source, query],
        "tool_source": [{"role": "tool", "content": source["content"]}, query],
        "missing_source": [query],
    }
    if isinstance(payload, str):
        code_source = messages_for(template, question, "__import__('os').getcwd()")[0]
    else:
        code_source = {"role": "user", "content": template.format(a="2+3", b="7")}
    variants["input_code"] = [code_source, query]
    return [{"id": f"negative-{lesson_id}-{kind}", "lesson_id": lesson_id,
             "split": "negative", "kind": kind, "messages": msgs,
             "expected_answer": None, "expected_abstain": True}
            for kind, msgs in variants.items()]


def build() -> dict[str, dict]:
    rng = random.Random(SEED)
    lessons, dev, final, negatives = [], [], [], []
    for spec in SPECS:
        lesson_id, _, _, operation = spec
        payload_function = integer_payloads if operation in {"add", "multiply"} else text_payloads
        train_values = payload_function(rng, "train")
        qualification_values = payload_function(rng, "qualification")
        dev_values = payload_function(rng, "dev")
        final_values = payload_function(rng, "final")
        train_rows = [positive_case(spec, "train", i, value) for i, value in enumerate(train_values)]
        qualification_rows = [positive_case(spec, "qualification", i, value)
                              for i, value in enumerate(qualification_values)]
        qualification_rows.extend(negative_cases(spec, qualification_values[0], qualification_values[1])[:4])
        lessons.append({
            "lesson_id": lesson_id, "trusted": True,
            "examples": [{"id": row["id"], "messages": row["messages"],
                          "answer": row["expected_answer"]} for row in train_rows],
            "qualification": [{"messages": row["messages"], "answer": row["expected_answer"]}
                              for row in qualification_rows],
        })
        dev.extend(positive_case(spec, "dev", i, value) for i, value in enumerate(dev_values))
        final.extend(positive_case(spec, "final", i, value) for i, value in enumerate(final_values))
        negatives.extend(negative_cases(spec, final_values[0], final_values[1]))
    corrupted = deepcopy(lessons[1])
    corrupted["lesson_id"] = "signal_reverse_corrupted_candidate"
    corrupted["examples"][-1]["answer"] = "CORRUPTED_GOLD_NOT_A_VALID_OPERATION"
    return {
        "LESSONS/TRAIN.json": {"version": VERSION, "seed": SEED, "lessons": lessons},
        "LESSONS/CORRUPTED.json": {"version": VERSION, "purpose": "must reject without registry mutation",
                                   "lesson": corrupted},
        "CASES/DEV.json": {"version": VERSION, "seed": SEED, "cases": dev},
        "CASES/FINAL.json": {"version": VERSION, "seed": SEED, "cases": final},
        "CASES/NEGATIVE.json": {"version": VERSION, "seed": SEED, "cases": negatives},
    }


def validate(bundle: dict[str, dict]) -> dict:
    """Check labels, independent counterfactuals, disjoint values, and task identifiability."""
    lessons = bundle["LESSONS/TRAIN.json"]["lessons"]
    assert len(lessons) == 6
    examples_seen = set()
    hypotheses = {}
    for spec, lesson in zip(SPECS, lessons):
        _, template, _, operation = spec
        examples = lesson["examples"]
        assert len(examples) == 8
        inputs = [example["messages"][0]["content"] for example in examples]
        assert len(set(inputs)) == len(inputs)
        assert not set(inputs) & examples_seen
        examples_seen.update(inputs)
        # Independently recover authored payloads from the fixed source template.
        if operation in {"add", "multiply"}:
            prefix, remainder = template.split("{a}")
            separator, suffix = remainder.split("{b}")
            payloads = []
            for source in inputs:
                left, right = source[len(prefix):-len(suffix)].split(separator)
                payloads.append((int(left), int(right)))
            candidates = ("add", "subtract", "multiply")
        else:
            prefix, suffix = template.split("{value}")
            payloads = [source[len(prefix):-len(suffix)] for source in inputs]
            candidates = ("copy", "reverse", "upper", "lower")
        consistent = [candidate for candidate in candidates if all(
            answer_for(candidate, payload) == example["answer"]
            for payload, example in zip(payloads, examples))]
        assert consistent == [operation], (lesson["lesson_id"], consistent)
        hypotheses[lesson["lesson_id"]] = len(consistent)
        qualification = lesson["qualification"]
        assert len(qualification) == 12
        assert sum(row["answer"] is not None for row in qualification) == 8
        assert sum(row["answer"] is None for row in qualification) == 4
        assert all(row["messages"] not in [example["messages"] for example in examples]
                   for row in qualification)
    split_sources = {"train": examples_seen}
    for split, filename in (("dev", "DEV"), ("final", "FINAL")):
        rows = bundle[f"CASES/{filename}.json"]["cases"]
        assert len(rows) == (48 if split == "dev" else 96)
        sources = {row["messages"][0]["content"] for row in rows}
        assert not sources & set().union(*split_sources.values())
        split_sources[split] = sources
        for index in range(0, len(rows), 2):
            a, b = rows[index:index + 2]
            assert a["pair_id"] == b["pair_id"]
            assert a["messages"][0] != b["messages"][0]
            assert a["expected_answer"] != b["expected_answer"]
            assert a["messages"][-1] == b["messages"][-1]
    negatives = bundle["CASES/NEGATIVE.json"]["cases"]
    assert len(negatives) == 78
    assert all(row["expected_abstain"] and row["expected_answer"] is None for row in negatives)
    return {"lessons": 6, "train_examples": 48, "qualification_positive": 48,
            "qualification_negative": 24, "dev_positive": 48,
            "final_positive": 96, "negative": 78,
            "consistent_train_hypotheses_per_lesson": hypotheses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    bundle = build()
    summary = validate(bundle)
    if args.check_only:
        print(json.dumps(summary, sort_keys=True))
        return
    paths = {}
    for relative, value in bundle.items():
        path = args.out / relative
        data = encoded(value)
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f"refusing to change precommitted dataset {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths[relative] = {"sha256": digest(data), "bytes": len(data)}
    manifest_path = args.out / "MANIFEST.json"
    manifest = {"version": VERSION, "seed": SEED,
                "precommitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "generator_sha256": digest(Path(__file__).read_bytes()),
                "plan_sha256": digest(Path(__file__).with_name("PLAN.md").read_bytes()),
                "files": paths, "validation": summary,
                "independence": "Authored without reading skill_programs.py; learner receives only TRAIN examples.",
                "scope": "Known source and query templates; held-out values, lengths, ranges, and Unicode. No new-wording claim."}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in manifest:
            if key != "precommitted_at_utc" and existing[key] != manifest[key]:
                raise RuntimeError("refusing to rewrite precommitted manifest")
        manifest = existing
    else:
        manifest_path.write_bytes(encoded(manifest))
    print(json.dumps({"out": str(args.out), "manifest_sha256": digest(encoded(manifest)),
                      **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
