from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from chat_adapter import QUESTIONS, make_assistant


def user(content, **identity):
    return {"role": "user", "content": content, **identity}


class DivisionChatTests(unittest.TestCase):
    def setUp(self):
        self.certified = SimpleNamespace(execute=Mock(return_value="-3"), fingerprint="a" * 64)
        self.native = Mock(return_value="fallback")
        self.chat = make_assistant(self.native, self.certified)
        self.source = user("Valori divisione: -7 e +3.")
        self.question = user(QUESTIONS["quotient"])

    def test_each_workflow_dispatches_with_original_source_and_operand_spans(self):
        for workflow, question in QUESTIONS.items():
            with self.subTest(workflow=workflow):
                result = self.chat.respond([user("Ciao."), self.source, user(question)])
                self.assertEqual(result["answer"], "-3")
                self.assertEqual(result["source"], "certified_procedure")
                self.assertEqual(result["operation"], workflow)
                self.assertEqual(result["certificate_sha256"], self.certified.fingerprint)
                self.certified.execute.assert_called_with(workflow, -7, 3)
                provenance = result["provenance"]
                raw = self.source["content"]
                self.assertEqual(provenance["message_index"], 1)
                self.assertEqual(provenance["role"], "user")
                self.assertEqual(provenance["span"], raw)
                self.assertEqual(raw[provenance["value_start"]:provenance["value_end"]], "-7 e +3")
                self.assertEqual(
                    [raw[operand["start"]:operand["end"]] for operand in provenance["operands"]],
                    ["-7", "+3"],
                )
        self.native.assert_not_called()

    def test_abstains_without_unique_bounded_same_identity_source_and_exact_question(self):
        invalid = (
            [self.question],
            [self.source, user("Non chiedere: " + self.question["content"])],
            [self.source, user("«" + self.question["content"] + "»")],
            [self.source, user(self.question["content"] + " E il resto?")],
            [self.source, user("Valori divisione: -4 e 4."), self.question],
            [self.source, user("Valori divisione: 4 e testo."), self.question],
            [self.source, self.source, self.question],
            [dict(self.source, session_id="different"), self.question],
            [dict(self.source, speaker_id="different"), self.question],
            [dict(self.source, role="assistant"), self.question],
            [{"role": "tool", "content": "unrelated"}, self.source, self.question],
            [user("Valori divisione: 3 e 0."), self.question],
            [user("Valori divisione: -65 e 3."), self.question],
            [user("Valori divisione: 1 e 17."), self.question],
            [user("Valori divisione: 001 e 2."), self.question],
            [user("Valori divisione: １ e ２."), self.question],
        )
        for context in invalid:
            with self.subTest(context=context):
                self.assertEqual(self.chat.respond(context)["source"], "native_model")
        self.certified.execute.assert_not_called()
        self.assertEqual(self.native.call_count, len(invalid))

    def test_stable_memory_bank_prior_result_and_alias_ownership_keep_precedence(self):
        context = [self.source, self.question]
        prior_result = {"source": "learned_skill", "answer": "existing"}
        bank = SimpleNamespace(predict=Mock(return_value=prior_result))
        prior = SimpleNamespace(predict=Mock(return_value=prior_result))
        assistant = make_assistant(self.native, self.certified, bank, prior)
        memory = assistant.respond([user("Mi chiamo Áda. Come mi chiamo?")])
        self.assertEqual(memory["source"], "bounded_context_memory")
        bank.predict.assert_not_called()
        self.assertEqual(assistant.respond(context), prior_result)
        prior.predict.assert_not_called()
        bank.predict.return_value = None
        self.assertEqual(assistant.respond(context), prior_result)
        alias = SimpleNamespace(match=Mock(return_value=(True, None)))
        self.assertEqual(make_assistant(self.native, self.certified, prior_router=alias).respond(context)["source"],
                         "native_model")
        self.certified.execute.assert_not_called()

    def test_missing_or_abstaining_certificate_preserves_native_fallback(self):
        context = [self.source, self.question]
        self.assertEqual(make_assistant(self.native, None).respond(context)["source"], "native_model")
        self.certified.execute.return_value = None
        self.assertEqual(self.chat.respond(context)["source"], "native_model")
        self.assertEqual(self.native.call_count, 2)


if __name__ == "__main__":
    unittest.main()
