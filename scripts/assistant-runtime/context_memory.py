"""Bounded, stateless software memory for explicit user declarations.

This module is a deterministic parser, not a trained model or a claim about a
model's intelligence. ``resolve(messages)`` returns a JSON-serializable answer
and exact source offsets, or ``None`` so the caller can use its native model.
It never consults a model, a dataset, the network, a file, or global chat state.

The intentionally limited grammar accepts complete IT/EN declarations:

* name: ``Mi chiamo VALUE``, ``Il mio nome è VALUE``, ``My name is VALUE``,
  ``I am called VALUE``;
* residence: ``Vivo/Abito a/in VALUE``, ``La città in cui vivo è VALUE``,
  ``I live/reside in VALUE``, ``The city I live in is VALUE``;
* topic: ``Sto/Stiamo parlando di VALUE``, ``Il mio argomento/tema è VALUE``,
  ``I am/I'm talking about VALUE``, ``We are/We're talking about VALUE``,
  ``My topic is VALUE``. A collective declaration is still sourced only from
  the user who explicitly states the current conversation topic.

Declarations can be separated by periods, exclamations, semicolons, newlines, or ``e/and``
before another supported declaration. A supported recall question may follow
them in the same message. Questions are exact, case-insensitive IT/EN forms
listed in ``_QUESTIONS``. No world knowledge or geographical classification is
performed: a residence answer simply repeats the declared place.

``Correzione:``, ``Rettifica:``, ``anzi``, ``in realtà``, ``ora/adesso``,
``Correction:``, ``actually``, ``now`` and ``no,`` mark explicit corrections.
Unmarked conflicting assertions make a slot unavailable until an explicit
correction. Explicit negation makes a slot unavailable; it never supplies a
positive fact. Corrections affect only their declared slot and speaker/session.

Values are unseen, opaque, printable Unicode strings, with multiple words,
digits, hyphens and apostrophes allowed. They are bounded, cannot contain
sentence/quotation delimiters, and cannot contain reserved negation,
hypothetical, reporting or instruction markers. This deliberately excludes
some legitimate names/topics; abstention is preferable to guessing a parse.
Recognized reporting/fiction/hypothetical qualifiers, quoted text, code and
unsupported mixed messages do not supply facts. Explicit whole-message
retractions clear all slots. A direct but unsupported apparent update clears
its slot rather than leaving a stale answer; a direct update mixed with quoted
or embedded content clears all slots because its boundaries are uncertain.
These conservative syntax guards
are not an exhaustive detector of arbitrary natural-language implications:
bare values are opaque strings. This is no general discourse resolver.

Only user/assistant transcripts are eligible. Any system, developer, tool or
unknown role makes the whole request ineligible, so the memory cannot bypass
format/tool instructions. Assistant messages never supply facts. Optional
speaker/session metadata must match the final user message exactly; missing
metadata is not equated with identified metadata. No history is truncated:
over-limit requests abstain in full. All offsets index the original Python
content string (Unicode code points), with an exclusive end offset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
import unicodedata
from typing import Any


SLOTS = ("name", "location", "topic")
MAX_MESSAGES = 128
MAX_MESSAGE_CHARS = 8192
MAX_TOTAL_CHARS = 65536
MAX_VALUE_CHARS = 200
MAX_VALUE_WORDS = 20

_IDENTITY_KEYS = (
    "speaker_id", "speaker", "name", "user_id", "session_id",
    "conversation_id", "session", "conversation",
)
_FLAGS = re.IGNORECASE

# The alternatives are grammar, not an inventory of values or benchmark cases.
_POSITIVE = (
    ("name", re.compile(r"(?:mi\s+chiamo|il\s+mio\s+nome\s+è|my\s+name\s+is|i\s+am\s+called)\s+(?P<value>.+)\Z", _FLAGS)),
    ("location", re.compile(r"(?:vivo\s+(?:a|in)|abito\s+(?:a|in)|la\s+città\s+in\s+cui\s+vivo\s+è|i\s+(?:live|reside)\s+in|the\s+city\s+i\s+live\s+in\s+is)\s+(?P<value>.+)\Z", _FLAGS)),
    ("topic", re.compile(r"(?:(?:sto|stiamo)\s+parlando\s+di|il\s+mio\s+(?:argomento|tema)\s+è|i\s+am\s+talking\s+about|i['’]m\s+talking\s+about|we\s+are\s+talking\s+about|we['’]re\s+talking\s+about|my\s+topic\s+is)\s+(?P<value>.+)\Z", _FLAGS)),
)
_NEGATIVE = (
    ("name", re.compile(r"(?:non\s+mi\s+chiamo|il\s+mio\s+nome\s+non\s+è|my\s+name\s+(?:is\s+not|isn['’]t)|i\s+am\s+not\s+called)\s+.+\Z", _FLAGS)),
    ("location", re.compile(r"(?:non\s+(?:vivo|abito)(?:\s+pi[ùu])?\s+(?:a|in)|i\s+(?:do\s+not|don['’]t|no\s+longer)\s+(?:live|reside)\s+in)\s+.+\Z", _FLAGS)),
    ("topic", re.compile(r"(?:non\s+(?:sto|stiamo)\s+parlando\s+di|i\s+am\s+not\s+talking\s+about|i['’]m\s+not\s+talking\s+about|we\s+are\s+not\s+talking\s+about|we['’]re\s+not\s+talking\s+about|we\s+aren['’]t\s+talking\s+about|my\s+topic\s+is\s+not)\s+.+\Z", _FLAGS)),
)
_CORRECTION = re.compile(
    r"(?:(?:correzione|rettifica|correction)\s*:\s*|"
    r"(?:anzi|in\s+realtà|in\s+realta|actually|no)\s*,?\s+|"
    r"(?:ora|adesso|now)\s+)", _FLAGS,
)
_SLOT_CUES = (
    ("name", re.compile(r"(?:non\s+mi\s+chiamo|mi\s+chiamo|il\s+mio\s+nome|my\s+name|i\s+am\s+(?:not\s+)?called)\b", _FLAGS)),
    ("location", re.compile(r"(?:(?:non\s+)?(?:vivo|abito)|la\s+città\s+in\s+cui\s+vivo|i\s+(?:(?:do\s+not|don['’]t|no\s+longer)\s+)?(?:live|reside)|the\s+city\s+i\s+live)\b", _FLAGS)),
    ("topic", re.compile(r"(?:(?:non\s+)?(?:sto|stiamo)\s+parlando|il\s+mio\s+(?:argomento|tema)|i\s+am\s+(?:not\s+)?talking|i['’]m\s+(?:not\s+)?talking|we\s+(?:are\s+(?:not\s+)?|aren['’]t\s+)talking|we['’]re\s+(?:not\s+)?talking|my\s+topic)\b", _FLAGS)),
)
_RETRACTION = re.compile(
    r"(?:non\s+(?:è|era)\s+vero|non\s+è\s+corretto|ho\s+mentito|"
    r"quello\s+che\s+ho\s+detto\s+non\s+è\s+vero|"
    r"that\s+(?:is|was)\s+(?:not\s+true|untrue|false|a\s+lie)|"
    r"i\s+lied|what\s+i\s+said\s+(?:is|was)\s+not\s+true)\Z", _FLAGS,
)
# Only split conjunctions when a complete supported declaration lead follows.
_LEADS = (
    r"mi\s+chiamo|il\s+mio\s+nome|my\s+name|i\s+am\s+(?:called|not\s+called)|"
    r"vivo\s+(?:a|in)|abito\s+(?:a|in)|la\s+città\s+in\s+cui\s+vivo|"
    r"i\s+(?:live|reside|do\s+not|don['’]t|no\s+longer)\s+|the\s+city\s+i\s+live|"
    r"(?:sto|stiamo)\s+parlando|il\s+mio\s+(?:argomento|tema)|i\s+am\s+(?:not\s+)?talking|"
    r"i['’]m\s+(?:not\s+)?talking|we\s+(?:are\s+(?:not\s+)?|aren['’]t\s+)talking|"
    r"we['’]re\s+(?:not\s+)?talking|my\s+topic|non\s+(?:mi\s+chiamo|vivo|abito|(?:sto|stiamo)\s+parlando)"
)
_CONJUNCTION = re.compile(
    r"\s+(?:e|and)\s+(?=(?:" + _CORRECTION.pattern + r")?(?:" + _LEADS + r"))", _FLAGS,
)
_SENTENCE_END = re.compile(r"[.?!;\n\r。！？]")
_QUOTED_SINGLE = re.compile(r"(?:^|[\s(\[{])['’].+?['’](?=$|[\s)\]},.!?;:])", re.DOTALL)
_RESERVED = re.compile(
    r"\b(?:not|non|never|mai|maybe|perhaps|forse|if|se|unless|would|sarebbe|"
    r"suppose|supposing|supponiamo|immagina|pretend|fingi|when|whenever|quando|qualora|"
    r"hypothetical|hypothetically|fictional|imaginary|ipotetico|ipotetica|"
    r"ipoteticamente|fittizio|fittizia|immaginario|immaginaria|"
    r"fiction|novel|story|storia|romanzo|racconto|roleplay|roleplaying|"
    r"character|personaggio|scenario|simulation|simulazione|dream|sogno|game|gioco|"
    r"untrue|false|falso|falsa|lie|lying|mentito|mentendo|allegedly|purportedly|"
    r"presumably|apparently|reportedly|"
    r"ignora|ignore|rispondi|answer|reply|"
    r"instructions|istruzioni|prompt|system|developer|assistant)\b|"
    r"\b(?:according\s+to|secondo|in\s+(?:this\s+story|questa\s+storia)|"
    r"as\s+(?:stated|reported|said)\s+by|a\s+detta\s+di|"
    r"is\s+what|è\s+quello\s+che|i\s+am|i['’]m|io\s+(?:sono|sto)|"
    r"per\s+(?:esempio|ipotesi)|for\s+example|as\s+an?\s+example|"
    r"(?:said|says)\s+that|(?:ho|hai)\s+detto|mi\s+ha\s+detto|and\s+then|e\s+poi)\b|"
    r"\s(?:but|ma|oppure|or|says|said|dice|disse|reported|reportedly)\s|"
    r"\b(?:and|e)\s+(?:this|that|it|questo|quello|ciò|write|return|output|respond|"
    r"print|say|tell|scrivi|restituisci|stampa|di['’]|facciamo|fai)\b", _FLAGS,
)

_QUESTIONS = {
    "name": {
        "it": (
            "come mi chiamo", "qual è il mio nome", "quale è il mio nome",
            "ricordi come mi chiamo", "ti ricordi come mi chiamo",
        ),
        "en": ("what is my name", "what's my name", "what’s my name", "do you remember my name"),
    },
    "location": {
        "it": (
            "dove vivo", "dove abito", "in quale città vivo", "in che città vivo",
            "in quale città abito", "in che città abito", "qual è la città in cui vivo",
        ),
        "en": ("where do i live", "what city do i live in", "which city do i live in"),
    },
    "topic": {
        "it": (
            "di cosa sto parlando", "di che cosa sto parlando", "qual è il mio argomento",
            "qual è il mio tema", "di cosa stiamo parlando", "di che cosa stiamo parlando",
        ),
        "en": ("what am i talking about", "what is my topic", "what's my topic", "what’s my topic", "what are we talking about"),
    },
}
_QUESTION_LOOKUP = {
    question: (slot, language)
    for slot, languages in _QUESTIONS.items()
    for language, questions in languages.items()
    for question in questions
}


@dataclass(frozen=True)
class _Event:
    slot: str
    start: int
    end: int
    value_start: int | None = None
    value_end: int | None = None
    correction: bool = False


@dataclass
class _State:
    event: _Event | None
    message_index: int | None
    canonical_value: str | None
    ambiguous: bool = False


def _identity(message: Mapping[str, Any]) -> tuple[Any, ...] | None:
    values = []
    for key in _IDENTITY_KEYS:
        value = message.get(key)
        if value is not None and (not isinstance(value, str) or not value or len(value) > 256):
            return None
        values.append(value)
    return tuple(values)


def _untrusted(text: str) -> bool:
    return (
        any(char in text for char in '"“”«»„‟〝〞‘‚‛`<>')
        or bool(_QUOTED_SINGLE.search(text))
        or bool(re.search(r"(?m)^\s*\|", text))
    )


def _safe_value(value: str) -> bool:
    if not value or len(value) > MAX_VALUE_CHARS or len(value.split()) > MAX_VALUE_WORDS:
        return False
    if not any(char.isalnum() for char in value):
        return False
    if any(char in '.,?!;:()[]{}<>\\=|"“”«»`' for char in value):
        return False
    if any(not char.isprintable() or unicodedata.category(char) in {"Cc", "Cs", "Cf", "Co"} for char in value):
        return False
    if re.search(r"\s[–—]\s", value):
        return False
    return not bool(_RESERVED.search(value))


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _phrases(text: str) -> list[tuple[int, int, str]]:
    """Split without rewriting source text, retaining sentence terminators."""
    units = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        left, right = _trim(text, start, match.start())
        if left < right:
            units.append((left, right, match.group()))
        start = match.end()
    left, right = _trim(text, start, len(text))
    if left < right:
        units.append((left, right, ""))
    phrases = []
    for start, end, terminator in units:
        cursor = start
        for match in _CONJUNCTION.finditer(text, start, end):
            left, right = _trim(text, cursor, match.start())
            phrases.append((left, right, ""))
            cursor = match.end()
        left, right = _trim(text, cursor, end)
        phrases.append((left, right, terminator))
    return phrases


def _parse(text: str) -> tuple[list[_Event], tuple[str, str] | None, bool]:
    """Parse a whole message; an unknown clause cannot endorse adjacent facts."""
    if _untrusted(text):
        # A quoted/reported declaration is never a fact. An apparent direct
        # update containing quotes is nevertheless a barrier to stale recall.
        start, end = _trim(text, 0, len(text))
        correction_match = _CORRECTION.match(text, start, end)
        body_start = correction_match.end() if correction_match else start
        direct_update = any(pattern.match(text, body_start, end) for _, pattern in _SLOT_CUES)
        events = [_Event(slot, start, end) for slot in SLOTS] if direct_update else []
        return events, None, False
    events = []
    queries = []
    complete = True
    phrases = _phrases(text)
    for position, (start, end, terminator) in enumerate(phrases):
        phrase = text[start:end]
        question = _QUESTION_LOOKUP.get(" ".join(phrase.casefold().split()))
        if question:
            queries.append(question)
            if position != len(phrases) - 1:
                complete = False
            continue
        # Unsupported interrogatives never become declarations.
        if terminator in {"?", "？"}:
            complete = False
            continue
        correction_match = _CORRECTION.match(phrase)
        body_start = start + (correction_match.end() if correction_match else 0)
        body = text[body_start:end]
        if _RETRACTION.fullmatch(body):
            events.extend(_Event(slot, start, end) for slot in SLOTS)
            continue
        found = False
        for slot, pattern in _NEGATIVE:
            if pattern.fullmatch(body):
                events.append(_Event(slot, start, end))
                found = True
                break
        if found:
            continue
        for slot, pattern in _POSITIVE:
            match = pattern.fullmatch(body)
            if match:
                value_start = body_start + match.start("value")
                value_end = body_start + match.end("value")
                value_start, value_end = _trim(text, value_start, value_end)
                if _safe_value(text[value_start:value_end]):
                    events.append(_Event(slot, start, end, value_start, value_end, bool(correction_match)))
                else:
                    # An unsafe apparent re-declaration cannot leave stale facts answerable.
                    events.append(_Event(slot, start, end))
                    complete = False
                found = True
                break
        if not found:
            events.extend(
                _Event(slot, start, end)
                for slot, pattern in _SLOT_CUES if pattern.match(body)
            )
            complete = False
    if len(queries) > 1:
        complete = False
    if not complete:
        events = [_Event(event.slot, event.start, event.end) for event in events]
    return events, (queries[0] if len(queries) == 1 else None), complete


def resolve(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Resolve one unambiguous recall request, returning value and provenance.

    Inputs are never mutated. The final message must be a user message entirely
    in the supported grammar, ending in exactly one supported recall question.
    The answer is the verbatim value; callers should display it as text. ``None``
    always means abstain, including malformed or over-limit input.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return None
    if not messages or len(messages) > MAX_MESSAGES:
        return None
    total_chars = 0
    identities = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in ("user", "assistant"):
            return None
        content = message.get("content")
        if not isinstance(content, str) or len(content) > MAX_MESSAGE_CHARS:
            return None
        total_chars += len(content)
        identity = _identity(message)
        if identity is None:
            return None
        identities.append(identity)
    if total_chars > MAX_TOTAL_CHARS or messages[-1]["role"] != "user":
        return None
    _, query, complete = _parse(messages[-1]["content"])
    if not complete or query is None:
        return None
    target_identity = identities[-1]
    states: dict[str, _State] = {}
    for index, message in enumerate(messages):
        if message["role"] != "user" or identities[index] != target_identity:
            continue
        events, _, _ = _parse(message["content"])
        for event in events:
            if event.value_start is None:
                states[event.slot] = _State(None, None, None, True)
                continue
            value = message["content"][event.value_start:event.value_end]
            canonical_value = " ".join(unicodedata.normalize("NFC", value).casefold().split())
            previous = states.get(event.slot)
            if event.correction or previous is None:
                states[event.slot] = _State(event, index, canonical_value)
            elif previous.ambiguous or previous.canonical_value != canonical_value:
                states[event.slot] = _State(None, None, None, True)
            else:
                states[event.slot] = _State(event, index, canonical_value)
    slot, language = query
    state = states.get(slot)
    if state is None or state.ambiguous or state.event is None or state.message_index is None:
        return None
    event = state.event
    source_message = messages[state.message_index]
    source_text = source_message["content"]
    value = source_text[event.value_start:event.value_end]
    provenance = {
        "message_index": state.message_index,
        "role": "user",
        "span": source_text[event.start:event.end],
        "start": event.start,
        "end": event.end,
        "value_span": value,
        "value_start": event.value_start,
        "value_end": event.value_end,
        "explicit_correction": event.correction,
    }
    provenance.update({key: source_message[key] for key in _IDENTITY_KEYS if key in source_message})
    return {
        "answer": value,
        "slot": slot,
        "language": language,
        "source": "bounded_context_memory",
        "provenance": provenance,
    }
