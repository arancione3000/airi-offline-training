"""Read-only conversation probe on one verified offline checkpoint."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from generalist_lm.runtime import GeneralistRuntime
from background import load_release, gh, FINGERPRINT

PROMPTS = [
    'Ciao! Come stai?',
    'Come ti chiami?',
    'Ho avuto una giornata difficile a scuola. Puoi dirmi qualcosa per tirarmi su?',
    'Quanto fa 2 + 2?',
    'Spiegami in una frase che cosa è un gatto.',
    'Mi chiamo Thomas e il mio colore preferito è arancione. Come mi chiamo e quale colore mi piace?',
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = load_release(args.release, args.root.resolve())
    manifest = json.loads((checkpoint / 'checkpoint.json').read_text())
    runtime = GeneralistRuntime.from_checkpoint(checkpoint / 'model')
    runtime.model.eval()
    answers = []
    for prompt in PROMPTS:
        started = time.monotonic()
        with torch.inference_mode():
            response = runtime.chat([{'role': 'user', 'content': prompt}],
                                    max_new_tokens=96, temperature=0.0)
        row = {'prompt': prompt, 'response': response,
               'seconds': time.monotonic() - started}
        answers.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    # Assert the files are still byte-identical after inference.
    from background import verify_checkpoint
    verify_checkpoint(checkpoint)
    result = {'checkpoint_release': args.release,
              'completed_steps': manifest['completed_steps'],
              'fingerprint': FINGERPRINT, 'context_length': runtime.config.context_length,
              'max_new_tokens': 96, 'temperature': 0.0,
              'training_performed': False, 'checkpoint_unchanged': True,
              'production_qualified': False, 'live_promoted': False, 'answers': answers}
    output = args.root / f'AIRI_CONVERSATION_PROBE_{os.environ["GITHUB_RUN_ID"]}.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    gh('release', 'upload', args.release, output, '--repo', os.environ['GITHUB_REPOSITORY'])

if __name__ == '__main__':
    main()
