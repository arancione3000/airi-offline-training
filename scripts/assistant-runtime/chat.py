"""Chat with experimental Airi memory and the hash-verified approved native model."""
import argparse
import json
import os
from pathlib import Path
import sys

from assistant import AiriAssistant


class VerifiedNative:
    """Load once: conversation history and the approved model stay independent."""
    def __init__(self, *, cache, checkpoint=None, record=None):
        if (checkpoint is None) != (record is None):
            raise ValueError('--checkpoint and --model-record must be supplied together')
        self.cache = cache
        self.checkpoint = checkpoint
        self.record = record
        self.runtime = None

    def __call__(self, messages):
        if self.runtime is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transfer-training'))
            from chat_current import load_current, validate_selected_model
            if self.checkpoint is not None:
                from generalist_lm.runtime import GeneralistRuntime
                root = validate_selected_model(self.checkpoint, self.record)
                self.runtime = GeneralistRuntime.from_checkpoint(root)
            else:
                self.runtime, self.record = load_current(self.cache)
        return self.runtime.chat(messages, temperature=0, max_new_tokens=192)


def main():
    # Avoid CPU oversubscription on ordinary two-core offline runners; explicit
    # user environment settings remain authoritative and Torch is still lazy.
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('airi-current-cache'))
    p.add_argument('--checkpoint', type=Path, help='Local checkpoint parent; requires its approved model record')
    p.add_argument('--model-record', type=Path, help='Hash-bound AIRI_CURRENT.json for the local checkpoint')
    p.add_argument('--messages-file', type=Path, help='One JSON array of conversation messages; emit one JSON answer')
    p.add_argument('--native-only', action='store_true', help='Bypass contextual memory for an explicitly neural-only measurement')
    p.add_argument('--skills',type=Path,help='Persistent learned skill bank; default private bank or the bundled qualified bank')
    p.add_argument('--no-skills',action='store_true',help='Disable learned experts while keeping the stable memory/native assistant')
    p.add_argument('--no-paraphrases',action='store_true',help='Keep only exact expert questions and contextual memory')
    p.add_argument('--learn',type=Path,help='Learn and qualify explicitly trusted labeled lessons, then exit')
    p.add_argument('--procedures', type=Path, help='Open a verified procedural growth snapshot')
    p.add_argument('--procedure-reference', type=Path, help='Independent arithmetic reference manifest')
    p.add_argument('--procedure-reference-sha256', help='Trusted SHA-256 of the reference manifest')
    a = p.parse_args()
    if (a.checkpoint is None) != (a.model_record is None):
        p.error('--checkpoint and --model-record are required together')
    growth = (a.procedures, a.procedure_reference, a.procedure_reference_sha256)
    if any(value is not None for value in growth) and not all(growth):
        p.error('--procedures, --procedure-reference and --procedure-reference-sha256 are required together')
    if a.procedures and (a.native_only or a.no_skills or a.learn):
        p.error('--procedures requires skill routing during chat')
    record = json.loads(a.model_record.read_text()) if a.model_record else None
    from learn import selected_bank,private_bank,learn_file
    if a.learn:
        bank=private_bank(a.skills)
        print(json.dumps({'results':learn_file(a.learn,bank),'registry':bank.status()},ensure_ascii=False));return
    native = VerifiedNative(cache=a.cache, checkpoint=a.checkpoint, record=record)
    bank=None if a.no_skills or a.native_only else selected_bank(a.skills)
    from flexible_skills import default_router
    procedures = None
    if a.procedures:
        from growth_routes import follow_verified_procedures, with_verified_procedures
        try:
            procedures = follow_verified_procedures(*growth)
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
            p.error(f'verified procedures unavailable ({error})')

    def routes(skill_bank):
        router = None if a.no_paraphrases else default_router(skill_bank)
        return router if procedures is None else with_verified_procedures(router, procedures)

    assistant = AiriAssistant(native, skill_bank=bank, query_router=routes(bank))

    def respond(messages):
        if a.native_only:
            return {'answer': native(messages), 'source': 'native_model', 'provenance': None}
        return assistant.respond(messages)

    if a.messages_file:
        messages = json.loads(a.messages_file.read_text())
        result = respond(messages)
        print(json.dumps(result, ensure_ascii=False))
        return
    print('Airi sperimentale — memoria, abilità apprese e modello nativo. /learn FILE, /reset, /exit')
    history = []
    while True:
        try:
            text = input('Tu: ')
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip() in {'/exit', '/quit'}:
            break
        if text.strip() == '/reset':
            history.clear()
            print('Conversazione azzerata.')
            continue
        if text.startswith('/learn '):
            try:
                bank=private_bank(a.skills)
                result=learn_file(Path(text[7:].strip()),bank)
                if not a.no_skills and not a.native_only:
                    assistant.skill_bank=bank
                    assistant.query_router=routes(bank)
                print(json.dumps({'learning':result,'registry':bank.status()},ensure_ascii=False))
            except (ValueError,OSError) as error:
                print(f'Lezione rifiutata ({type(error).__name__}).',file=sys.stderr)
            continue
        if not text.strip():
            continue
        history.append({'role': 'user', 'content': text})
        try:
            result = respond(history)
        except Exception as error:
            # No fabricated answer, no conversation files and no raw API stderr.
            print(f'Modello non disponibile ({type(error).__name__}); la memoria della conversazione resta utilizzabile.', file=sys.stderr)
            continue
        history.append({'role': 'assistant', 'content': result['answer']})
        label = {'bounded_context_memory':'memoria','learned_skill':'abilità appresa','native_model':'modello',
                 'certified_procedure':'procedura verificata','source_conflict':'chiarimento'}[result['source']]
        if result.get('component_error'):print(result['component_error'],file=sys.stderr)
        print(f"Airi [{label}]: {result['answer']}")


if __name__ == '__main__':
    main()
