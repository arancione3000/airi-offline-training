import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/assistant-runtime'))
from assistant import AiriAssistant


class AssistantTests(unittest.TestCase):
    def test_memory_answer_has_source_and_does_not_load_or_modify_model_history(self):
        messages = [{'role': 'user', 'content': 'Mi chiamo Ada.'},
                    {'role': 'assistant', 'content': 'Va bene.'},
                    {'role': 'user', 'content': 'Come mi chiamo?'}]
        before = copy.deepcopy(messages)
        def forbidden(_):
            self.fail('memory route unnecessarily invoked native model')
        result = AiriAssistant(forbidden).respond(messages)
        self.assertEqual(result['answer'], 'Ada')
        self.assertEqual(result['source'], 'bounded_context_memory')
        self.assertEqual(result['provenance']['message_index'], 0)
        self.assertEqual(messages, before)

    def test_abstention_forwards_exact_original_conversation_to_native(self):
        messages = [{'role': 'system', 'content': 'Respond using JSON.'},
                    {'role': 'user', 'content': 'Mi chiamo Ada. Come mi chiamo?'}]
        calls = []
        def native(value):
            calls.append(value)
            return '{"answer":"Ada"}'
        result = AiriAssistant(native).respond(messages)
        self.assertEqual(result['source'], 'native_model')
        self.assertEqual(result['answer'], '{"answer":"Ada"}')
        self.assertIs(calls[0], messages)

    def test_sessions_do_not_share_facts(self):
        assistant = AiriAssistant(lambda _: 'native fallback')
        assistant.respond([{'role': 'user', 'content': 'My name is Ada. What is my name?'}])
        result = assistant.respond([{'role': 'user', 'content': 'What is my name?'}])
        self.assertEqual(result['source'], 'native_model')

    def test_native_non_text_is_not_presented_as_an_answer(self):
        with self.assertRaises(TypeError):
            AiriAssistant(lambda _: None).respond([{'role': 'user', 'content': '2+2?'}])


if __name__ == '__main__':
    unittest.main()
