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
    a = p.parse_args()
    if (a.checkpoint is None) != (a.model_record is None):
        p.error('--checkpoint and --model-record are required together')
    record = json.loads(a.model_record.read_text()) if a.model_record else None
    native = VerifiedNative(cache=a.cache, checkpoint=a.checkpoint, record=record)
    assistant = AiriAssistant(native)

    def respond(messages):
        if a.native_only:
            return {'answer': native(messages), 'source': 'native_model', 'provenance': None}
        return assistant.respond(messages)

    if a.messages_file:
        messages = json.loads(a.messages_file.read_text())
        result = respond(messages)
        print(json.dumps(result, ensure_ascii=False))
        return
    print('Airi sperimentale — memoria contestuale esplicita + modello nativo. /reset, /exit')
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
        label = 'memoria' if result['source'] == 'bounded_context_memory' else 'modello'
        print(f"Airi [{label}]: {result['answer']}")


if __name__ == '__main__':
    main()
