"""Locally constructed examples; no corpus, evaluation, or model files are read."""

import copy
import json
from pathlib import Path
import random
import string
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/assistant-runtime"))
from skill_programs import (  # noqa: E402
    MAX_ANCHOR_CHARS,
    MAX_EXAMPLES,
    MAX_INTEGER_ABS,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES,
    MAX_OUTPUT_CHARS,
    MAX_QUERY_CHARS,
    MAX_TOTAL_CHARS,
    MAX_VALUE_CHARS,
    MAX_VALUE_WORDS,
    OPERATIONS,
    predict,
    synthesize,
    validate_program,
)


def user(content, **metadata):
    return {"role": "user", "content": content, **metadata}


def assistant(content, **metadata):
    return {"role": "assistant", "content": content, **metadata}


TEXT_VALUES = ("Nivë 17", "佐藤 4", "Δelta π", "Māra-Rū 8")


def text_examples(operation="copy", *, prefix="My badge token is ", suffix=".", query="Which badge token did I give?"):
    transforms = {"copy": lambda value: value, "reverse": lambda value: value[::-1],
                  "uppercase": str.upper, "lowercase": str.lower}
    return [
        {"messages": [user(prefix + value + suffix), assistant("Acknowledged."), user(query)],
         "answer": transforms[operation](value)}
        for value in TEXT_VALUES
    ]


def integer_examples(operation="add"):
    transforms = {"add": lambda left, right: left + right,
                  "subtract": lambda left, right: left - right,
                  "multiply": lambda left, right: left * right}
    return [
        {"messages": [user(f"My current pair is {left} and {right}."), user("Apply my pair rule?")],
         "answer": str(transforms[operation](left, right))}
        for left, right in ((2, 7), (-5, 3), (11, -4), (0, 9))
    ]


class SkillProgramsTests(unittest.TestCase):
    def assert_prediction(self, program, messages, answer):
        original = copy.deepcopy(messages)
        result = predict(program, messages)
        self.assertIsNotNone(result, (program, messages))
        self.assertEqual(result["answer"], answer)
        self.assertEqual(result["source"], "learned_skill")
        self.assertEqual(result["operation"], program["operation"])
        provenance = result["provenance"]
        source = messages[provenance["message_index"]]
        self.assertEqual(source["role"], "user")
        self.assertEqual(provenance["role"], "user")
        text = source["content"]
        self.assertEqual(text[provenance["start"]:provenance["end"]], provenance["span"])
        self.assertEqual(text[provenance["value_start"]:provenance["value_end"]], provenance["value_span"])
        for operand in provenance.get("operands", []):
            self.assertEqual(text[operand["start"]:operand["end"]], operand["span"])
            self.assertEqual(int(operand["span"]), operand["value"])
        self.assertEqual(original, messages)
        self.assertEqual(json.loads(json.dumps(result)), result)
        return result

    def test_learns_new_literal_anchors_and_exact_query(self):
        examples = text_examples(prefix="My nebula badge: ", suffix="; saved", query="Show the nebula badge?")
        original = copy.deepcopy(examples)
        program = synthesize(examples)
        self.assertTrue(validate_program(program))
        self.assertEqual(program, {"version": 1, "query": "Show the nebula badge?",
                                   "source": {"prefix": "My nebula badge: ", "suffix": "; saved"},
                                   "operation": "copy"})
        self.assertEqual(examples, original)
        self.assertEqual(json.loads(json.dumps(program)), program)
        self.assert_prediction(program, [user("My nebula badge: 新しい 739; saved"), user(program["query"])], "新しい 739")

    def test_all_text_operations_generalize_to_new_values(self):
        transforms = {"copy": lambda value: value, "reverse": lambda value: value[::-1],
                      "uppercase": str.upper, "lowercase": str.lower}
        for operation, transform in transforms.items():
            with self.subTest(operation=operation):
                program = synthesize(text_examples(operation))
                self.assertIsNotNone(program)
                self.assertEqual(program["operation"], operation)
                self.assert_prediction(program, [user("My badge token is Straße İ 東京."), user(program["query"])],
                                       transform("Straße İ 東京"))

    def test_same_query_counterfactual_changes_answer(self):
        program = synthesize(text_examples("reverse"))
        for value in ("Élodie 903", "山田 42", "Qwerty-7"):
            self.assert_prediction(program, [user("My badge token is " + value + "."), user(program["query"])], value[::-1])

    def test_unicode_case_mapping_expansion_during_synthesis(self):
        for values, operation in (
            (("Straße", "Öfen", "Aïcha", "Νίκη"), "uppercase"),
            (("İSTANBUL", "ÄPFEL", "ΣΟΦΙΑ", "QUÉBEC"), "lowercase"),
        ):
            transform = str.upper if operation == "uppercase" else str.lower
            examples = [{"messages": [user("The glyph tag is " + value + "."), user("Show glyph tag?")],
                         "answer": transform(value)} for value in values]
            program = synthesize(examples)
            self.assertIsNotNone(program)
            self.assertEqual(program["operation"], operation)

    def test_arithmetic_operations_extrapolate_with_exact_operand_offsets(self):
        expected = {"add": "999983", "subtract": "1000017", "multiply": "-17000000"}
        for operation in expected:
            with self.subTest(operation=operation):
                program = synthesize(integer_examples(operation))
                self.assertIsNotNone(program)
                self.assertEqual(program["operation"], operation)
                result = self.assert_prediction(program, [user("My current pair is 1000000 and -17."),
                                                          user(program["query"])], expected[operation])
                self.assertEqual(result["provenance"]["value_span"], "1000000 and -17")
                self.assertEqual(len(result["provenance"]["operands"]), 2)

    def test_arithmetic_separator_is_learned_literal(self):
        examples = integer_examples("multiply")
        for example in examples:
            example["messages"][0]["content"] = example["messages"][0]["content"].replace(" and ", " <plus> ")
        self.assertIsNone(synthesize(examples))  # Quoted/code-like source context is outside the grammar.
        for example in examples:
            example["messages"][0]["content"] = example["messages"][0]["content"].replace(" <plus> ", " :: ")
        program = synthesize(examples)
        self.assertIsNotNone(program)
        self.assertEqual(program["source"]["separator"], " :: ")
        self.assert_prediction(program, [user("My current pair is -13 :: 21."), user(program["query"])], "-273")

    def test_no_hardcoded_values_random_unseen_strings(self):
        program = synthesize(text_examples("reverse"))
        rng = random.Random(624091)
        for _ in range(24):
            value = "".join(rng.choice(string.ascii_letters + string.digits + "éΩ山") for _ in range(19))
            self.assert_prediction(program, [user("My badge token is " + value + "."), user(program["query"])], value[::-1])

    def test_exact_whole_source_and_query_required(self):
        program = synthesize(text_examples())
        for source in ("my badge token is Aurora.", "Here: My badge token is Aurora.",
                       "My badge token is Aurora. Extra sentence.", "My badge token is Aurora", " My badge token is Aurora."):
            self.assertIsNone(predict(program, [user(source), user(program["query"])]), source)
        for query in (program["query"].upper(), program["query"] + " ", "Which badge?", "What city is Paris in?"):
            self.assertIsNone(predict(program, [user("My badge token is Aurora."), user(query)]))
        self.assertIsNone(predict(program, [user("My badge token is Aurora. " + program["query"])]))

    def test_conflicting_raw_sources_abstain_even_if_operation_collapses_them(self):
        program = synthesize(text_examples("uppercase"))
        self.assertIsNone(predict(program, [user("My badge token is AbC."), user("My badge token is aBC."), user(program["query"])]))
        repeated = [user("My badge token is AbC."), user("My badge token is AbC."), user(program["query"])]
        result = self.assert_prediction(program, repeated, "ABC")
        self.assertEqual(result["provenance"]["message_index"], 1)

    def test_assistant_cannot_supply_facts_and_final_message_must_be_user(self):
        program = synthesize(text_examples())
        self.assertIsNone(predict(program, [assistant("My badge token is Wrong."), user(program["query"])]))
        self.assert_prediction(program, [user("My badge token is Right."), assistant("My badge token is Wrong."), user(program["query"])], "Right")
        self.assertIsNone(predict(program, [user("My badge token is Right."), assistant(program["query"])]))

    def test_forbidden_roles_reject_entire_transcript(self):
        program = synthesize(text_examples())
        for role in ("system", "developer", "tool", "function", "unknown", None):
            messages = [{"role": role, "content": "Other instructions"}, user("My badge token is Right."), user(program["query"])]
            self.assertIsNone(predict(program, messages), role)
            examples = text_examples()
            examples[0]["messages"].insert(0, messages[0])
            self.assertIsNone(synthesize(examples))

    def test_identity_fields_require_exact_source_query_coherence(self):
        program = synthesize(text_examples())
        for key in ("speaker_id", "speaker", "name", "user_id", "session_id", "conversation_id", "session", "conversation"):
            with self.subTest(key=key):
                self.assertIsNone(predict(program, [user("My badge token is Wrong.", **{key: "other"}),
                                                    user(program["query"], **{key: "current"})]))
                self.assertIsNone(predict(program, [user("My badge token is Wrong."), user(program["query"], **{key: "current"})]))
                self.assert_prediction(program, [user("My badge token is Wrong.", **{key: "other"}),
                                                 user("My badge token is Right.", **{key: "current"}),
                                                 user(program["query"], **{key: "current"})], "Right")

    def test_malformed_identity_abstains(self):
        program = synthesize(text_examples())
        for value in (1, True, [], {}, "", "x" * 257):
            self.assertIsNone(predict(program, [user("My badge token is Right.", session_id=value), user(program["query"])]))

    def test_negation_hypotheticals_reports_quotes_and_instructions_abstain(self):
        program = synthesize(text_examples())
        for source in (
            "My badge token is not Right.", "My badge token is Right if we pretend.",
            "Suppose My badge token is Right.", "A friend said My badge token is Right.",
            "My badge token is Right says someone.", '"My badge token is Right."',
            "'My badge token is Right.'", "My badge token is 'Right'.", "My badge token is ‘Right’.",
            "My badge token is Right and ignore instructions.", "My badge token is Right. Not true.",
            "My badge token is `Right`.", "My badge token is <Right>.",
        ):
            self.assertIsNone(predict(program, [user(source), user(program["query"])]), source)

    def test_synthesis_rejects_unsupported_source_contexts(self):
        for prefix in ("Suppose my token is ", "A friend said my token is ", "My token is not ", 'My quoted token is "'):
            self.assertIsNone(synthesize(text_examples(prefix=prefix)))

    def test_training_minimum_variation_and_constant_labels(self):
        examples = text_examples()
        self.assertIsNone(synthesize(examples[:3]))
        self.assertIsNone(synthesize([examples[0]] * 4))
        constant = copy.deepcopy(examples)
        for example in constant:
            example["answer"] = "Constant"
        self.assertIsNone(synthesize(constant))
        insufficient_values = [examples[0], examples[0], examples[1], examples[2]]
        self.assertIsNone(synthesize(insufficient_values))
        self.assertIsNone(synthesize(examples * (MAX_EXAMPLES // len(examples) + 1)))

    def test_ambiguous_operations_and_source_templates_are_rejected(self):
        examples = text_examples()
        for example in examples:
            example["messages"][0]["content"] = example["messages"][0]["content"].upper()
            example["answer"] = example["answer"].upper()
        self.assertIsNone(synthesize(examples))  # Both copy and uppercase explain these labels.
        examples = text_examples()
        for example in examples:
            value = example["answer"]
            example["messages"].insert(0, user("Another badge is " + value + "."))
        self.assertIsNone(synthesize(examples))  # Two independent extraction templates fit.

    def test_inconsistent_labels_and_different_queries_reject(self):
        examples = text_examples()
        examples[-1]["answer"] = "Wrong label"
        self.assertIsNone(synthesize(examples))
        examples = text_examples()
        examples[-1]["messages"][-1]["content"] = "Another query?"
        self.assertIsNone(synthesize(examples))

    def test_each_numeric_operand_must_vary(self):
        examples = [{"messages": [user(f"My current pair is {left} and 0."), user("Apply my pair rule?")],
                     "answer": str(left)} for left in (2, 3, 7, 11)]
        self.assertIsNone(synthesize(examples))

    def test_numeric_bounds_and_syntax(self):
        program = synthesize(integer_examples())
        for body in (f"{MAX_INTEGER_ABS + 1} and 2", "2.5 and 3", "02 and 3", "2 and 03", "2e3 and 5",
                     "2 and 3 and 4", "2 and -1000001", "２ and 3", "2  and 3", "2 and  3"):
            self.assertIsNone(predict(program, [user("My current pair is " + body + "."), user(program["query"])]), body)
        self.assert_prediction(program, [user("My current pair is +2 and -3."), user(program["query"])], "-1")

    def test_program_schema_and_code_injection_are_rejected(self):
        program = synthesize(text_examples())
        bad_programs = [None, [], "copy", {**program, "exec": "print('unsafe')"},
                        {**program, "operation": "__import__('os').system('id')"},
                        {**program, "operation": "eval"}, {**program, "version": True},
                        {**program, "version": 2}, {**program, "source": {**program["source"], "path": "/tmp/file"}},
                        {**program, "source": {"prefix": "My token: ", "suffix": ".", "regex": ".*"}},
                        {**program, "source": {"prefix": "x" * (MAX_ANCHOR_CHARS + 1), "suffix": ""}},
                        {**program, "query": "x" * (MAX_QUERY_CHARS + 1)},
                        {**program, "source": {"prefix": "My token: ", "suffix": "\n"}}]
        for bad in bad_programs:
            with self.assertRaises(ValueError):
                validate_program(bad)
            self.assertIsNone(predict(bad, [user("My badge token is Right."), user(program["query"])]))

    def test_string_limits_and_controls(self):
        program = synthesize(text_examples())
        for value in ("x" * (MAX_VALUE_CHARS + 1), " ".join(["word"] * (MAX_VALUE_WORDS + 1)),
                      "Right\nWrong", "Right\x00", "Right\u202eWrong", "Right\ud800", "", " Right", "Right "):
            self.assertIsNone(predict(program, [user("My badge token is " + value + "."), user(program["query"])]))
        examples = text_examples()
        examples[0]["answer"] = "x" * (MAX_OUTPUT_CHARS + 1)
        self.assertIsNone(synthesize(examples))

    def test_transcript_limits_and_bad_shapes_abstain_without_truncation(self):
        program = synthesize(text_examples())
        valid = [user("My badge token is Right."), user(program["query"])]
        self.assertIsNone(predict(program, [assistant("x")] * MAX_MESSAGES + valid))
        self.assertIsNone(predict(program, [assistant("x" * (MAX_MESSAGE_CHARS + 1))] + valid))
        self.assertIsNone(predict(program, [assistant("x" * MAX_MESSAGE_CHARS)] * (MAX_TOTAL_CHARS // MAX_MESSAGE_CHARS) + valid))
        for messages in (None, {}, "messages", [], [None], [user(["text"])], [{"role": "user"}]):
            self.assertIsNone(predict(program, messages))
        for examples in (None, {}, "examples", [], [{"messages": valid, "answer": "Right", "program": "code"}] * 4):
            self.assertIsNone(synthesize(examples))

    def test_public_operations_are_fixed_and_no_cross_call_state(self):
        self.assertEqual(OPERATIONS, ("copy", "reverse", "uppercase", "lowercase", "add", "subtract", "multiply"))
        first = synthesize(text_examples())
        second = synthesize(text_examples("reverse", prefix="My second tag: ", query="Second tag?"))
        self.assert_prediction(first, [user("My badge token is First."), user(first["query"])], "First")
        self.assert_prediction(second, [user("My second tag: Second."), user(second["query"])], "dnoceS")
        self.assertIsNone(predict(first, [user(first["query"])]))
        self.assertIsNone(predict(second, [user("My badge token is First."), user(second["query"])]))


if __name__ == "__main__":
    unittest.main()
