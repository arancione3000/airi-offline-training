"""Build the frozen synthetic stream for the format repair experiment."""

import argparse
import json
from pathlib import Path
import random


SEED = 429017
INITIAL = ("spedite", "stornate")
CHANGES = (
    ("consegne", ("consegnate", "rese")),
    ("ordini", ("evase", "annullate")),
)
UNRESOLVED = ("prenotate", "residue")


def report(labels, pair, style=0):
    shipped, cancelled = labels
    left, right = pair
    if style == 1:
        return f"{cancelled.upper()} = {right}\n{shipped.upper()} : {left}"
    if style == 2:
        return f" {shipped} = {left} ; {cancelled} : {right} "
    return f"{shipped}: {left}; {cancelled}: {right}"


def build():
    rng = random.Random(SEED)
    used = set()

    def pair():
        while True:
            values = rng.randint(0, 10000), rng.randint(0, 10000)
            if values[0] != values[1] and values not in used:
                used.add(values)
                return values

    def clean(prefix, labels, count):
        rows = []
        for index in range(count):
            values = pair()
            rows.append({"id": f"{prefix}-{index}", "kind": "clean", "reports": [
                report(labels, values, index % 3)
            ], "expected": values[0] - values[1]})
        return rows

    def duplicates(prefix, labels, other_labels, count):
        rows = []
        for index in range(count):
            values = pair()
            rows.append({"id": f"{prefix}-{index}", "kind": "duplicates", "reports": [
                report(labels, values), report(other_labels, values, 1)
            ], "expected": values[0] - values[1]})
        return rows

    def negatives(prefix, labels, other_labels):
        left, right = pair()
        shifted = left + 1, right + 1
        while shifted in used:
            shifted = shifted[0] + 1, shifted[1] + 1
        used.add(shifted)
        return [
            {"id": prefix + "-equal-net", "kind": "equal_net_conflict", "reports": [
                report(labels, (left, right)), report(other_labels, shifted)
            ], "expected": None},
            {"id": prefix + "-conflict", "kind": "operand_conflict", "reports": [
                report(labels, pair()), report(other_labels, pair(), 1)
            ], "expected": None},
            {"id": prefix + "-missing", "kind": "missing_field", "reports": [
                f"{labels[0]}: {pair()[0]}"
            ], "expected": None},
            {"id": prefix + "-noise", "kind": "extra_field", "reports": [
                report(labels, pair()) + "; ripetute: 5"
            ], "expected": None},
            {"id": prefix + "-empty", "kind": "no_source", "reports": [], "expected": None},
            {"id": prefix + "-mixed", "kind": "unknown_source", "reports": [
                report(labels, pair()), report(UNRESOLVED, pair(), 2)
            ], "expected": None},
        ]

    protected = (clean("old-clean", INITIAL, 12)
                 + duplicates("old-duplicate", INITIAL, INITIAL, 4)
                 + negatives("old-negative", INITIAL, INITIAL))
    updates = []
    previous = INITIAL
    for name, labels in CHANGES:
        arrival = clean(name + "-arrival", labels, 1)[0]
        calibration = (clean(name + "-qualification", labels, 8)
                       + negatives(name + "-qualification-negative", labels, INITIAL))
        final = (clean(name + "-new", labels, 12)
                 + clean(name + "-old", previous, 4)
                 + duplicates(name + "-duplicate", labels, INITIAL, 4)
                 + negatives(name + "-negative", labels, INITIAL)
                 + [{"id": name + "-unresolved", "kind": "unresolved_format", "reports": [
                     report(UNRESOLVED, pair())
                 ], "expected": None}])
        rng.shuffle(final)
        updates.append({
            "id": name,
            "arrival": arrival,
            "confirmed_shipped_label": labels[0],
            "qualification": calibration,
            "final": final,
        })
        previous = labels
    return {"schema": 1, "seed": SEED, "protected": protected, "updates": updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
