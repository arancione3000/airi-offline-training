"""Opt-in chat routing for the certified arithmetic experiment."""
from pathlib import Path
import sys
from types import MappingProxyType

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assistant-runtime"))
from assistant import AiriAssistant
from flexible_skills import _QuestionRoutes, default_router
import skill_programs
from transfer import valid_operands


QUESTIONS = MappingProxyType({
    "quotient": "Qual è il quoziente intero?",
    "remainder": "Qual è il resto della divisione?",
    "completed_amount": "Qual è il multiplo corrispondente al quoziente?",
    "remaining_magnitude": "Qual è il valore assoluto del resto?",
    "exact_division": "La divisione è esatta?",
    "next_boundary": "Qual è il multiplo corrispondente al quoziente successivo?",
})
_WORKFLOWS = {question: workflow for workflow, question in QUESTIONS.items()}
_PREFIX = "Valori divisione: "
_INPUT_FORMAT = {
    "source": {"prefix": _PREFIX, "suffix": ".", "separator": " e "},
    # The existing DSL uses an integer operation to select its operand parser.
    # Only source extraction runs here.
    "operation": "add",
}


class _DivisionRouter:
    def __init__(self, certified):
        self.certified = certified

    def predict(self, messages):
        sources = skill_programs._eligible(messages)
        if not sources:
            return None
        workflow = _WORKFLOWS.get(messages[-1]["content"])
        if workflow is None:
            return None
        division_sources = [(index, text) for index, text in sources if text.startswith(_PREFIX)]
        if len(division_sources) != 1:
            return None
        index, text = division_sources[0]
        extracted = skill_programs._extract(_INPUT_FORMAT, text)
        if extracted is None:
            return None
        value, operands = extracted
        left, right = (operand["value"] for operand in operands)
        if not valid_operands(left, right):
            return None
        answer = self.certified.execute(workflow, left, right)
        if answer is None:
            return None
        return {
            "answer": answer,
            "source": "certified_procedure",
            "operation": workflow,
            "certificate_sha256": self.certified.fingerprint,
            "provenance": {
                "message_index": index, "role": "user", "start": 0, "end": len(text), "span": text,
                "value_start": len(_PREFIX), "value_end": len(text) - 1, "value_span": value,
                "operands": operands,
            },
        }


class _PrioritizedRouter:
    def __init__(self, first, fallback):
        self.first = first
        self.fallback = fallback

    def predict(self, messages):
        answer = self.first.predict(messages)
        return answer if answer is not None else self.fallback.predict(messages)


def make_assistant(native_chat, certified, bank=None, prior_router=None):
    """Use certified arithmetic only after the existing memory and skill routes."""
    router = default_router(bank) if prior_router is None else prior_router
    if certified is not None:
        division = _DivisionRouter(certified)
        if isinstance(router, _QuestionRoutes):
            # Retain exclusive ownership for a literal alias that abstains.
            semantic = router.semantic
            router = _QuestionRoutes(router.aliases, _PrioritizedRouter(semantic, division) if semantic else division)
        elif callable(getattr(router, "match", None)):
            router = _QuestionRoutes(router, division)
        elif router is not None:
            router = _PrioritizedRouter(router, division)
        else:
            router = division
    return AiriAssistant(native_chat, skill_bank=bank, query_router=router)
