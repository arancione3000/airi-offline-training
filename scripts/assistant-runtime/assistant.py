"""Experimental composed Airi: explicit contextual memory plus verified native fallback.

Memory answers are software retrieval, not neural training or model qualification.
No specialist checkpoint is activated by this component. Optional learned programs
are bounded symbolic experts, not changes to the neural model.
"""
from context_memory import resolve


class AiriAssistant:
    def __init__(self, native_chat, skill_bank=None, query_router=None):
        self.native_chat = native_chat
        self.skill_bank = skill_bank
        self.query_router = query_router

    def respond(self, messages):
        memory = resolve(messages)
        if memory is not None:
            return dict(memory)
        component_error=None
        if self.skill_bank is not None:
            from skill_bank import RegistryError
            try:
                skill=self.skill_bank.predict(messages)
                if skill is not None:return skill
            except (RegistryError,OSError):
                component_error='learned registry unavailable; using stable native fallback'
        if self.query_router is not None:
            from skill_bank import RegistryError
            try:
                routed=self.query_router.predict(messages)
                if routed is not None:return routed
            except (RegistryError,OSError):
                component_error='learned question routing unavailable; using stable native fallback'
        answer = self.native_chat(messages)
        if not isinstance(answer, str):
            raise TypeError('native chat must return text')
        result={'answer': answer, 'source': 'native_model', 'provenance': None}
        if component_error:result['component_error']=component_error
        return result
