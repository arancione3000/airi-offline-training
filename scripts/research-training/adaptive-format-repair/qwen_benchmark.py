"""Evaluate a pinned Qwen in shadow on the same phase-specific format changes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from benchmark import CORPUS_SHA256, evaluate, fields
from cases import INITIAL


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "joint-action-20261121"))
from teacher import ENGINE_SHA256, MODEL_REVISION, MODEL_SHA256, Teacher


INSTRUCTIONS = """Estrai le quantità spedite e stornate dai resoconti ricevuti.
Ogni stringa è un resoconto separato. Ogni resoconto deve usare le due etichette
di uno stesso formato elencato sotto, con un intero per etichetta. L'ordine,
le maiuscole, gli spazi, i due punti o l'uguale non cambiano il significato.
I due campi sono separati da un punto e virgola oppure da un a capo.
Usa REVISIONE se la lista è vuota, se almeno un resoconto è incompleto,
se ci sono campi in più, se un formato non è elencato, o se le quantità
di due resoconti differiscono anche quando il saldo coinciderebbe.
Considera tutti i resoconti. Se sono coerenti, restituisci esattamente
i due numeri nell'ordine spedite,stornate separati da virgola. Non calcolare il saldo.
Formati confermati (prima etichetta = spedite, seconda = stornate):
"""
GRAMMAR = 'root ::= "REVISIONE" | integer "," integer\ninteger ::= [0-9]+\n'
PAIR = re.compile(r"([0-9]{1,7}),([0-9]{1,7})")
SEED = 7
N_PREDICT = 16


def run_phase(teacher, mappings, cases):
    catalog = "\n".join(f"- {shipped} / {cancelled}" for shipped, cancelled in mappings)
    prefix = (
        "<|im_start|>system\n" + INSTRUCTIONS + catalog
        + "<|im_end|>\n<|im_start|>user\n"
    )
    raw_outputs = []

    def assess(reports):
        request = json.dumps(reports, ensure_ascii=False)
        prompt = prefix + request + "<|im_end|>\n<|im_start|>assistant\n"
        status, result = teacher._request("/completion", {
            "prompt": prompt,
            "grammar": GRAMMAR,
            "temperature": 0.0,
            "repeat_penalty": 1.0,
            "n_predict": N_PREDICT,
            "seed": SEED,
            "cache_prompt": True,
        })
        if status != 200:
            raise RuntimeError(f"Qwen HTTP {status}: {result}")
        output = result["content"].strip()
        raw_outputs.append(output)
        match = PAIR.fullmatch(output)
        if not match:
            return {"status": "review"}
        shipped, cancelled = map(int, match.groups())
        if max(shipped, cancelled) > 1_000_000:
            return {"status": "review"}
        return {"status": "answered", "answer": shipped - cancelled}

    trial = evaluate(assess, cases)
    for row, raw in zip(trial["cases"], raw_outputs, strict=True):
        row["raw_output"] = raw
    trial["invalid_output_reviews"] = sum(
        raw != "REVISIONE" and row["status"] == "review"
        for row, raw in zip(trial["cases"], raw_outputs, strict=True)
    )
    return trial


def run(corpus):
    teacher = Teacher([])
    try:
        rounds, previous = [], []
        mappings = [INITIAL]
        for item in corpus["updates"]:
            labels = fields(item["arrival"]["reports"][0]).keys()
            shipped = item["confirmed_shipped_label"]
            cancelled, = labels - {shipped}
            mappings.append((shipped, cancelled))
            checks = run_phase(teacher, mappings, item["qualification"])
            historical = run_phase(teacher, mappings, previous) if previous else None
            failures = [row["id"] for trial in (checks, historical) if trial
                        for row in trial["cases"] if row["answer"] != row["expected"]]
            final = run_phase(teacher, mappings, item["final"])
            rounds.append({
                "id": item["id"],
                "confirmed_mappings": mappings[:],
                "qualification": checks,
                "historical_qualification": historical,
                "gate_passed": not failures,
                "gate_failure_case_ids": failures,
                "final_shadow": final,
            })
            previous.extend(item["qualification"])
    finally:
        teacher.close()
    return {
        "corpus_sha256": CORPUS_SHA256,
        "model": "Qwen2.5-1.5B-Instruct Q4_K_M",
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "engine_sha256": ENGINE_SHA256,
        "temperature": 0,
        "seed": SEED,
        "n_predict": N_PREDICT,
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (HERE / "CASES.json").read_bytes()
    if sha256(data).hexdigest() != CORPUS_SHA256:
        raise ValueError("The frozen corpus SHA-256 does not match")
    result = run(json.loads(data))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"id": item["id"], "gate_passed": item["gate_passed"],
                       "gate_failure_case_ids": item["gate_failure_case_ids"],
                       "final": item["final_shadow"]["metrics"]}
                      for item in result["rounds"]], indent=2))


if __name__ == "__main__":
    main()
