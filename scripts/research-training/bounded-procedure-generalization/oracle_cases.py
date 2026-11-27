"""Evaluator-only source for a frozen bounded procedure induction corpus.

This module deliberately imports no learner and does not express target programs
as the learner's executable operations. Use it to regenerate the corpus.
"""

import argparse
import json
from pathlib import Path
import random


SEED = 281993
STAGES = ("train", "pool", "qualification", "final")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task(identifier, domain, kind, target, train, pool, qualification, final, reviews=()):
    splits = dict(zip(STAGES, (train, pool, qualification, final), strict=True))
    cases = {"id": identifier, "domain": domain, "kind": kind}
    seen = set()
    for name in STAGES:
        rows = [{"input": value, "expected": target(value)} for value in splits[name]]
        if name == "final":
            rows.extend({"input": value, "expected": None} for value in reviews)
        for row in rows:
            key = canonical(row["input"])
            if key in seen:
                raise ValueError(f"{identifier} repeats an input in {name}: {key}")
            seen.add(key)
        cases[name] = rows
    return cases


def normalized_text(text):
    return text.lower().strip()


def reversed_undigitized_text(text):
    return "".join(character for character in reversed(text.strip())
                   if not character.isdigit())


def comma_list_as_line(text):
    return " ".join(field.strip() for field in text.split(","))


def bracketed_text(text):
    return f"[{text.strip().lower()}]"


def positive_total(numbers):
    return sum(max(0, number) for number in numbers)


def reverse_first_occurrences(numbers):
    return [value for index, value in reversed(tuple(enumerate(numbers)))
            if numbers.index(value) == index]


def even_magnitude_total(numbers):
    return sum(max(number, -number) for number in numbers if number % 2 == 0)


def squared_values(numbers):
    return [number ** 2 for number in numbers]


def positive_packing_count(record):
    return sum(max(0, count) for count in record["colli"])


def normalized_label(record):
    return record["etichetta"].upper().strip()


def unique_brands(record):
    seen = set()
    values = []
    for original in record["marche"]:
        value = original.lower()
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def net_quantity(record):
    return record["spedite"] - record["stornate"]


def text_cases():
    reviews = (42, ["testo"])
    return [
        task("T01", "text", "bounded", normalized_text,
             ["  ALBA   ", " BeTa ", "\tLuNa\n", " ARCO\t\tDUNE  "],
             ["ROSA", "  rosa", "  ROSe\n", "\tSI  NO  ", "Ponte  ", " °C "],
             ["  MaRe ", "ONDA   ", "\tÉTÉ", "\nNUvole\n", " QUI   ORA  ",
              "  NOTTE\tIMPROVVISA  "],
             ["", "   ", "ÉTÉ", " A  B ", "  ÜBER —  Straße ", " (CaSa) ",
              "\t  CIELO\nStelle\t", "\u00a0ŠUMA\u00a0", "BlU 2X", "Carattere—Doppio",
              "  \"TITolo\"  ", "  Via\nVECCHIA\n  "], reviews),
        task("T02", "text", "bounded", reversed_undigitized_text,
             ["7 MIRTO  ", " 2RIV3A8 ", " 6LE!9 ", " B1\nC4 5 "],
             [" St2a ", "8 PRATO   ", "  PORTO 7", " X3-no", "io9 ", "  21Z0   "],
             ["5 IL LAGO ", " La7Riva6  ", "  DR3E4", " Alba4 2  ", " \tZ9  ", "44 lume  "],
             ["", "   0000   ", "7 áÉ 9 ", " X 5", "5 X ", "  c4affè9– 9 ",
              "\tSenza cifre\t", "4.10", " 2F6\nD1 ", "0 ACQUA!  ", "  COD3ICE\n",
              " 2\t Z9  "], reviews),
        task("T03", "text", "bounded", comma_list_as_line,
             [" alfa , BETA, gamma  ", "perla,,  ciottolo", "  miele, pane  caldo  ",
              "riso , fungo;blu"],
             [" RIVA, riva ", ", a, b", "un, due,tre", "  luna  ", ";;,  ? ", "  a, b  c, d  "],
             ["  sabbia, NOTTE  ", ",dentro,,fuori", "leone,  orso  ", "  bruno, bLU ,",
              " ZUCCA , pera", "  Z,  K\t  W "],
             ["", ",", ",,", "  uno ", "  cioccolàto, MIELE ", "tè, , caffè",
              "X,Y,blu,violetto,ambra", " sale , pepe  nero\t ", "\nNEVE,\tBREZZA\n",
              "‘ramo’,  radici ", "a;b , c ", "  Café , déjà vu | luna   "], reviews),
        task("T04", "text", "ambiguity", normalized_text,
             ["paglia", "covone", "fieno", "grano"],
             ["terra", "rametto", "rive", "selva", "semina", "strada"],
             ["  MIRTO  ", "  CHICCo\t", "Cancello", "   Fulmine", "   FORNO ", "Vetro "],
             ["", "  ", "  ČEŠKA  ", "ETichetta", "  ViA  ", " Dora\n",
              " LUNA  QUI", "  SeRa", " PiéDE ", "BLU", " Kiwi ", " PANE "], reviews),
        task("T05", "text", "outside", bracketed_text,
             ["  DAto  ", "PeLo", "  RUOTA", "passo  "],
             ["Aghi", "StrAda", " NEBBIA", "Forte ", " (SEME)", "  a—B "],
             ["  Orto ", " TORRE ", "campo", "BOMBO ", " DENTE", " Onda "],
             ["", "  ", "  Café  ", " SALE  MARINO  ", "Ùmido ", "Rosa\n",
              "“ArtE”", "  Viola–SAmbuco", "Cézanne ", "stelle", " PANDA ", " CIELO "], reviews),
    ]


def random_sequences(rng, count, length, magnitudes, offset):
    rows = []
    for index in range(count):
        values = [rng.randint(-magnitudes, magnitudes) for _ in range(length + index % 3)]
        values.insert(index % len(values), offset + 2 * index)
        rows.append(values)
    return rows


def sequence_cases(rng):
    reviews = ("1,-3,2", [4, "7"], [True, 2])
    definitions = [
        ("S01", "bounded", positive_total,
         [[-4, 6, 6, 0], [10, -32, 2], [-4, -6, 0], [15, 3, 1]],
         [[0, 1], [5, -2], [11, -11], [2, 4, 0], [-3, -6], [1, 0, 5, -21]],
         [[], [-81], [93], [0, 0, 0], [-90, -90, 30, 30], [555, -899, -38, 17]]),
        ("S02", "bounded", reverse_first_occurrences,
         [[3, 1, 3, 2], [8, 4, 8, 8, 4, 0], [-8, 1, 2, 2], [3, 3, 1, 2]],
         [[-4, 5, -4], [8, 5, 1, 8], [1, 1, 2, 2], [7, 6, 5, 7], [6, 6, 6], [22, 2, 3, 22]],
         [[], [17], [-94, 0, 21, -94, 21], [0, 0, 0], [86, 1, 5, 86], [10, 6, 9, 10, 6, 10]]),
        ("S03", "bounded", even_magnitude_total,
         [[-10, 5, 2], [-6, -4, 2, 7], [3, 1, -5], [0, -8, 8, -5]],
         [[-5, 20, -20], [7, 0], [-8, -2], [3, 4, 6], [27], [-38, 31, 28]],
         [[], [-53], [-54], [0, 0, 0], [-2, -2, -11, 1, 2], [-780, 433, 780, 2]]),
        ("S04", "outside", squared_values,
         [[-7, 3], [4, 8, -4], [2, 0, 5], [3, 6, -9]],
         [[-2, 3, 0], [12], [-5, -5], [4, 4, 1], [-11, -13], [17, -17]],
         [[], [-6], [6], [0, 0, 0], [-6, 4, 2, -3], [63, 4, -9, 6]]),
    ]
    return [
        task(identifier, "sequence", kind, target, train, pool,
             random_sequences(rng, 8, 3, 25, 100),
             edge_final + random_sequences(rng, 8, 8, 75, 1000), reviews)
        for identifier, kind, target, train, pool, edge_final in definitions
    ]


def record_cases(rng):
    def packing(values, other, note=None):
        record = {"colli": values, "scarti": other}
        if note is not None:
            record["lotto"] = note
        return record

    def packing_batch(offset, count, max_value):
        inputs = random_sequences(rng, count, 4, max_value, offset)
        return [packing(values, list(reversed(values[:-1]))) for values in inputs]

    def label(value, code=None, extra=False):
        record = {"etichetta": value, "codice": value if code is None else code}
        if extra:
            record["origine"] = {"ambiente": "sintetico"}
        return record

    def brands(values, replacements=None, extra=False):
        record = {"marche": values, "alternative": replacements or ["ALTRA"]}
        if extra:
            record["lotto"] = "Dopo il cambiamento"
        return record

    return [
        task("R01", "record", "bounded", positive_packing_count,
             [packing([-7, 3, 11], [7, 3]), packing([1, 9, -33], [7, 0]),
              packing([-1, -2, 0], [15]), packing([9, 9, -12], [37, 1, 2])],
             [packing([n, -66, n + 1], [n + 1, -18, 4]) for n in range(20, 26)],
             packing_batch(200, 8, 40),
             [packing([], [9], "L-00"), packing([0], [9], "L-01"), packing([-44], [14]),
              packing([55], [0]), packing([999, -999, 999], [8], "L-04")]
             + packing_batch(1500, 10, 110),
             ({}, {"scarti": [4, -8]}, {"colli": "4,5", "scarti": [4, 5]})),
        task("R02", "record", "bounded", normalized_label,
             [label("  bOx  "), label(" rIva\t"), label("  colli"), label(" FINE   ")],
             [label("   lampo  ", "neve "), label("   siepe", "RIVA"),
              label("\tviTe\t", " siepe "), label(" Onda ", "onda "),
              label(" bRown ", "brown "), label("  fIUME ", " CIELO")],
             [label(" aLba ", "ALB-01"), label("  cesTo ", "ces-02"), label(" veLa  ", "VEL!"),
              label("  vASo ", " V-0"), label("  rOvo ", " R-1"), label("  reTE  ", "RETE!"),
              label("NUBE  ", "TERRA"), label("\tmarE\t", "RIVA")],
             [label("", "AH"), label("     ", "AZZ"), label("  bùono ", "B", True),
              label("  ÉpÉe ", ""), label("a  b", "ab"), label("  ambra\nrosa  ", "X"),
              label("  Straße  ", "ß"), label("🡆 blu ", "100", True),
              label("  _blu_   ", "_blu"), label("  30blu  ", "blu30"),
              label("  [tarma]  ", ""), label(" MaMma ", "MAMMA!"), {"etichetta": "  riva  "}],
             ({}, {"codice": "BX"}, {"etichetta": 17, "codice": "TEST"})),
        task("R03", "record", "bounded", unique_brands,
             [brands(["ALBA", "alba", "ROSA"]), brands(["DADO", "dado", "RAMO", "ramo"]),
              brands(["FARO", "TIMO", "faro"]), brands(["blu", "dune", "BLU", "NEVE"])],
             [brands(["Sera", "sera", f"CAttura{i}", f"cattura{i}"], ["ALTRA", "altra"])
              for i in range(6)],
             [brands(["TORRE", f"VALLATA{i}", "torre", f"vallata{i}"], ["TORRE"])
              for i in range(8)],
             [brands([]), brands([""]), brands(["luce"]), brands(["ÉTÉ", "été", "ÈVE"], extra=True),
              brands(["sasso ", "SASSO", "sasso", "SASSO "]),
              brands(["A", "b", "a", "B", "C", "c", "D", "d", "e", "E"]),
              brands(["  cava", "CAVA", "  CaVa"], extra=True),
              brands(["MINÈRALE", "Minerale", "minèrale", "MINERALE"])]
             + [brands(["CANNA", f"BOTTE{i}", "canna", f"botte{i}"], ["PIETRA"], True)
                for i in range(6)],
             ({}, {"alternative": ["BLU"]}, {"marche": [2, "blu"]})),
        task("R04", "record", "outside", net_quantity,
             [dict(spedite=sent, stornate=returned)
              for sent, returned in ((120, 5), (120, 10), (150, 10), (7, 9))],
             [dict(spedite=58, stornate=returned) for returned in range(20, 26)],
             [dict(spedite=sent, stornate=33) for sent in range(85, 93)],
             [dict(spedite=0, stornate=0), dict(spedite=0, stornate=4),
              dict(spedite=7, stornate=0), dict(spedite=25, stornate=25)]
             + [dict(spedite=sent, stornate=sent + 5, origine="prove") for sent in range(220, 230)],
             ({}, {"spedite": 5}, {"spedite": 7, "stornate": "due"})),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rng = random.Random(SEED)
    corpus = {"schema": 1, "seed": SEED,
              "tasks": text_cases() + sequence_cases(rng) + record_cases(rng)}
    args.output.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
