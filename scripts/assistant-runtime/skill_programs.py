"""Induce small, inspectable extraction programs from trusted labeled examples.

This is bounded symbolic rule learning, not neural training. No model, external
file, network, executable program text, or persistent chat state is consulted.
``synthesize([{"messages": [...], "answer": "..."}, ...])`` returns a JSON
program only when exactly one hypothesis explains at least four different
inputs and nonconstant labels. All training examples must have the same exact
final user question. The caller is responsible for accepting only trusted TRAIN
examples; this module cannot infer a dataset split or label trust from text.

Version 1 JSON schema (additional keys are rejected)::

    {"version": 1, "query": "exact learned question",
     "source": {"prefix": "literal source lead: ", "suffix": "."},
     "operation": "copy"}  # or reverse, uppercase, lowercase

Integer programs use operation add/subtract/multiply and additionally require
``source.separator``: a literal between two bounded signed ASCII integers.
There are no constants, arbitrary expressions, regexes, imports, or executable
fields in the DSL. Subtraction means the first operand minus the second.

Sources must occupy an entire earlier user message, matching learned literal
anchors exactly. Prefixes end at a delimiter; optional suffixes begin at one.
Only the final exact user question activates a program. Other speakers and
sessions cannot supply its facts; assistant text cannot supply facts. Any
system/developer/tool/unknown role makes the whole request ineligible. Multiple
different matching values cause abstention, including values whose transformed
answers happen to agree. Repeated identical values use the latest source.

Opaque values accept bounded printable Unicode with a conservative character
and word guard. Quotes, code delimiters, known negation/hypothetical/reporting
markers, and instruction markers are outside the supported source grammar.
These are syntax guards, not a general detector of natural-language intent.
Unsupported templates, ambiguous training hypotheses, and over-limit requests
return None. Offsets index original Python strings (Unicode code points), with
exclusive ends. Every prediction contains the exact source and operand spans.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


VERSION = 1
OPERATIONS = ("copy", "reverse", "uppercase", "lowercase", "add", "subtract", "multiply")
MIN_EXAMPLES = 4
MAX_EXAMPLES = 64
MAX_MESSAGES = 128
MAX_MESSAGE_CHARS = 8192
MAX_TOTAL_CHARS = 65536
MAX_VALUE_CHARS = 200
MAX_VALUE_WORDS = 20
MAX_OUTPUT_CHARS = 600
MAX_QUERY_CHARS = 512
MAX_ANCHOR_CHARS = 256
MAX_SEPARATOR_CHARS = 64
MAX_INTEGER_ABS = 1_000_000
MAX_CANDIDATES = 512

_TEXT_OPERATIONS = OPERATIONS[:4]
_INTEGER_OPERATIONS = OPERATIONS[4:]
_IDENTITY_KEYS = (
    "speaker_id", "speaker", "name", "user_id", "session_id",
    "conversation_id", "session", "conversation",
)
_INTEGER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)\Z", re.ASCII)
_INTEGER_TOKEN = re.compile(r"(?<![\w.+-])[+-]?(?:0|[1-9][0-9]*)(?![\w+-]|\.[0-9])")
_UNSAFE = re.compile(
    r"\b(?:not|non|never|mai|maybe|perhaps|forse|if|se|unless|would|sarebbe|"
    r"suppose|supposing|supponiamo|immagina|pretend|fingi|when|whenever|quando|"
    r"hypothetical|hypothetically|fictional|imaginary|ipotetico|ipotetica|"
    r"fiction|novel|story|storia|romanzo|racconto|roleplay|roleplaying|"
    r"character|personaggio|scenario|simulation|simulazione|dream|sogno|"
    r"untrue|false|falso|falsa|lie|lying|mentito|allegedly|purportedly|"
    r"apparently|reportedly|says|said|dice|disse|reported|"
    r"ignore|ignora|instructions|istruzioni|prompt|system|developer|assistant|"
    r"rispondi|answer|reply|return|output|respond|print|scrivi|restituisci|stampa)\b|"
    r"\b(?:according\s+to|secondo|as\s+(?:stated|reported|said)\s+by|"
    r"a\s+detta\s+di|for\s+example|per\s+(?:esempio|ipotesi)|"
    r"(?:ho|hai)\s+detto|mi\s+ha\s+detto)\b",
    re.IGNORECASE,
)
_QUOTED_SINGLE = re.compile(r"(?:^|\s)['’].+?['’](?=$|\s|[.!?;:])")


def _printable(text: str) -> bool:
    return all(char.isprintable() and not unicodedata.category(char).startswith("C") for char in text)


def _unsafe(text: str) -> bool:
    return (
        not _printable(text)
        or any(char in text for char in '\"“”«»„‟‘‚‛`<>\\|')
        or bool(_QUOTED_SINGLE.search(text))
        or bool(_UNSAFE.search(text))
    )


def _safe_value(value: str) -> bool:
    return (
        0 < len(value) <= MAX_VALUE_CHARS
        and value == value.strip()
        and len(value.split()) <= MAX_VALUE_WORDS
        and any(char.isalnum() for char in value)
        and not any(char in value for char in ".,?!;:()[]{}=")
        and not _unsafe(value)
    )


def _valid_anchors(source: dict[str, Any], integer: bool) -> bool:
    expected = {"prefix", "suffix", "separator"} if integer else {"prefix", "suffix"}
    if set(source) != expected or any(type(value) is not str for value in source.values()):
        return False
    prefix, suffix = source["prefix"], source["suffix"]
    if not (2 <= len(prefix) <= MAX_ANCHOR_CHARS and len(suffix) <= MAX_ANCHOR_CHARS):
        return False
    if not any(char.isalpha() for char in prefix):
        return False
    if not (prefix[-1].isspace() or prefix[-1] in ":=→"):
        return False
    if suffix and not (suffix[0].isspace() or suffix[0] in ".;!"):
        return False
    # Separate with spaces so the guard cannot accidentally join anchor words.
    if _unsafe(prefix + " " + suffix):
        return False
    if integer:
        separator = source["separator"]
        if not (0 < len(separator) <= MAX_SEPARATOR_CHARS) or _unsafe(separator):
            return False
        if any(char.isdigit() for char in separator):
            return False
    return True


def _valid_program(program: Any) -> bool:
    if type(program) is not dict or set(program) != {"version", "query", "source", "operation"}:
        return False
    if type(program["version"]) is not int or program["version"] != VERSION:
        return False
    operation, query, source = program["operation"], program["query"], program["source"]
    if type(operation) is not str or operation not in OPERATIONS:
        return False
    if type(query) is not str or not (0 < len(query) <= MAX_QUERY_CHARS) or not _printable(query):
        return False
    if query != query.strip() or type(source) is not dict:
        return False
    return _valid_anchors(source, operation in _INTEGER_OPERATIONS)


def validate_program(program: Any) -> bool:
    """Return True for a bounded version-1 JSON program; otherwise raise ValueError."""
    if not _valid_program(program):
        raise ValueError("invalid bounded skill program")
    return True


def _identity(message: dict[str, Any]) -> tuple[Any, ...] | None:
    values = []
    for key in _IDENTITY_KEYS:
        value = message.get(key)
        if value is not None and (type(value) is not str or not value or len(value) > 256):
            return None
        values.append(value)
    return tuple(values)


def _eligible(messages: Any) -> list[tuple[int, str]] | None:
    if type(messages) not in (list, tuple) or not (1 <= len(messages) <= MAX_MESSAGES):
        return None
    total = 0
    identities = []
    for message in messages:
        if type(message) is not dict or message.get("role") not in ("user", "assistant"):
            return None
        content = message.get("content")
        if type(content) is not str or len(content) > MAX_MESSAGE_CHARS:
            return None
        # A surrogate is not a valid UTF-8 source and cannot be safely persisted.
        if any(unicodedata.category(char) == "Cs" for char in content):
            return None
        total += len(content)
        identity = _identity(message)
        if identity is None or total > MAX_TOTAL_CHARS:
            return None
        identities.append(identity)
    if messages[-1]["role"] != "user":
        return None
    return [
        (index, message["content"])
        for index, message in enumerate(messages[:-1])
        if message["role"] == "user" and identities[index] == identities[-1]
    ]


def _integer(value: str) -> int | None:
    if len(value) > 8 or not _INTEGER.fullmatch(value):
        return None
    parsed = int(value)
    return parsed if abs(parsed) <= MAX_INTEGER_ABS else None


def _extract(program: dict[str, Any], text: str) -> tuple[str, list[dict[str, Any]]] | None:
    source = program["source"]
    prefix, suffix = source["prefix"], source["suffix"]
    if not text.startswith(prefix) or (suffix and not text.endswith(suffix)):
        return None
    max_body = MAX_VALUE_CHARS if program["operation"] in _TEXT_OPERATIONS else 16 + len(source["separator"])
    if len(text) > len(prefix) + len(suffix) + max_body:
        return None
    end = len(text) - len(suffix)
    if end < len(prefix) or _unsafe(text):
        return None
    value = text[len(prefix):end]
    if program["operation"] in _TEXT_OPERATIONS:
        return (value, []) if _safe_value(value) else None
    separator = source["separator"]
    if value.count(separator) != 1:
        return None
    first, second = value.split(separator)
    left, right = _integer(first), _integer(second)
    if left is None or right is None:
        return None
    second_start = len(prefix) + len(first) + len(separator)
    return value, [
        {"start": len(prefix), "end": len(prefix) + len(first), "span": first, "value": left},
        {"start": second_start, "end": second_start + len(second), "span": second, "value": right},
    ]


def _transform(operation: str, value: str, operands: list[dict[str, Any]]) -> str:
    if operation == "copy":
        return value
    if operation == "reverse":
        return value[::-1]
    if operation == "uppercase":
        return value.upper()
    if operation == "lowercase":
        return value.lower()
    left, right = operands[0]["value"], operands[1]["value"]
    if operation == "add":
        return str(left + right)
    if operation == "subtract":
        return str(left - right)
    return str(left * right)


def _predict_valid(program: dict[str, Any], sources: list[tuple[int, str]]) -> dict[str, Any] | None:
    selected = None
    for index, text in sources:
        extracted = _extract(program, text)
        if extracted is None:
            continue
        value, operands = extracted
        if selected is not None and selected[2] != value:
            return None
        selected = (index, text, value, operands)
    if selected is None:
        return None
    index, text, value, operands = selected
    answer = _transform(program["operation"], value, operands)
    if not (0 < len(answer) <= MAX_OUTPUT_CHARS):
        return None
    start = len(program["source"]["prefix"])
    provenance = {
        "message_index": index, "role": "user", "start": 0, "end": len(text),
        "span": text, "value_start": start, "value_end": start + len(value), "value_span": value,
    }
    if operands:
        provenance["operands"] = operands
    return {"answer": answer, "source": "learned_skill", "operation": program["operation"], "provenance": provenance}


def predict(program: Any, messages: Any) -> dict[str, Any] | None:
    """Execute a validated fixed-library operation, or abstain with None."""
    if not _valid_program(program):
        return None
    sources = _eligible(messages)
    if sources is None or messages[-1]["content"] != program["query"]:
        return None
    return _predict_valid(program, sources)


def _occurrences(text: str, needle: str):
    start = 0
    while True:
        start = text.find(needle, start)
        if start < 0:
            return
        yield start, start + len(needle)
        start += 1


def _text_spans(text: str, answer: str, operation: str):
    if operation == "copy":
        yield from _occurrences(text, answer)
    elif operation == "reverse":
        yield from _occurrences(text, answer[::-1])
    else:
        transform = str.upper if operation == "uppercase" else str.lower
        # Case mappings can expand Unicode (e.g. ß -> SS or İ -> i + dot).
        # Boundary offsets map transformed positions back to original strings.
        boundaries = {0: 0}
        position = 0
        for index, char in enumerate(text):
            position += len(transform(char))
            boundaries[position] = index + 1
        for start, end in _occurrences(transform(text), answer):
            if start in boundaries and end in boundaries:
                original_start, original_end = boundaries[start], boundaries[end]
                if transform(text[original_start:original_end]) == answer:
                    yield original_start, original_end


def synthesize(examples: Any) -> dict[str, Any] | None:
    """Return the unique replay-verified hypothesis for trusted examples.

    Every accepted program sees >=4 distinct extracted inputs for one exact
    question, providing same-question counterfactual contrasts. Integer
    examples must also vary each operand. Constant-label and multiple-program
    explanations are rejected. The search is capped; reaching the candidate
    cap rejects the lesson instead of silently truncating its hypothesis set.
    """
    if type(examples) not in (list, tuple) or not (MIN_EXAMPLES <= len(examples) <= MAX_EXAMPLES):
        return None
    records = []
    query = None
    labels = set()
    for example in examples:
        if type(example) is not dict or set(example) != {"messages", "answer"}:
            return None
        answer, messages = example["answer"], example["messages"]
        if type(answer) is not str or not (0 < len(answer) <= MAX_OUTPUT_CHARS) or not _printable(answer):
            return None
        sources = _eligible(messages)
        if sources is None:
            return None
        current_query = messages[-1]["content"]
        if query is None:
            query = current_query
        if query != current_query:
            return None
        records.append((sources, answer))
        labels.add(answer)
    if len(labels) < 2:
        return None

    candidates = {}

    def consider(source: dict[str, str], operation: str) -> bool:
        program = {"version": VERSION, "query": query, "source": source, "operation": operation}
        if not _valid_program(program):
            return True
        key = (operation, source["prefix"], source["suffix"], source.get("separator"))
        if key not in candidates:
            if len(candidates) >= MAX_CANDIDATES:
                return False
            candidates[key] = program
        return True

    first_sources, first_answer = records[0]
    for _, text in first_sources:
        # A valid anchored source can never be longer than these three parts.
        if len(text) > 2 * MAX_ANCHOR_CHARS + MAX_VALUE_CHARS:
            continue
        for operation in _TEXT_OPERATIONS:
            for start, end in _text_spans(text, first_answer, operation):
                if not _safe_value(text[start:end]):
                    continue
                if not consider({"prefix": text[:start], "suffix": text[end:]}, operation):
                    return None
        tokens = list(_INTEGER_TOKEN.finditer(text))
        # A separator cannot contain digits, so only adjacent numeric tokens
        # can form an admissible pair. Avoid a quadratic enumeration of pairs.
        for first, second in zip(tokens, tokens[1:]):
            source = {"prefix": text[:first.start()], "suffix": text[second.end():],
                      "separator": text[first.end():second.start()]}
            for operation in _INTEGER_OPERATIONS:
                if not consider(source, operation):
                    return None

    accepted = None
    for program in candidates.values():
        values = set()
        left_values, right_values = set(), set()
        for sources, answer in records:
            result = _predict_valid(program, sources)
            if result is None or result["answer"] != answer:
                break
            provenance = result["provenance"]
            values.add(provenance["value_span"])
            if "operands" in provenance:
                left_values.add(provenance["operands"][0]["value"])
                right_values.add(provenance["operands"][1]["value"])
        else:
            if len(values) < MIN_EXAMPLES:
                continue
            if program["operation"] in _INTEGER_OPERATIONS and (len(left_values) < 2 or len(right_values) < 2):
                continue
            if accepted is not None:
                return None
            accepted = program
    return accepted
