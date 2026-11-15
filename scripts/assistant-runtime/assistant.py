"""Experimental composed Airi: explicit contextual memory plus verified native fallback.

Memory answers are software retrieval, not neural training or model qualification.
No specialist checkpoint is activated by this component.
"""
from context_memory import resolve


class AiriAssistant:
    def __init__(self, native_chat):
        self.native_chat = native_chat

    def respond(self, messages):
        memory = resolve(messages)
        if memory is not None:
            return dict(memory)
        answer = self.native_chat(messages)
        if not isinstance(answer, str):
            raise TypeError('native chat must return text')
        return {'answer': answer, 'source': 'native_model', 'provenance': None}
