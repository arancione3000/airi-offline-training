"""Export the verified offline AIRI checkpoint for the local Android runtime."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/transfer-training'))
from chat_current import load_current, validate_selected_model
from model_channel import Client, REPOSITORY, read_current, validate


BRANCH = 'airi-android-current'
POINTER = 'models/AIRI_ANDROID_CURRENT.json'
MAX_BYTES = 512 * 1024**2
PROMPTS = (
    'Ciao', 'Come stai?', 'Parli italiano?', 'Mi chiamo Ada.',
    'Ripeti: caffè ☕', 'Hello', 'What is 2 + 2?', 'Qual è la capitale italiana?',
)


def compact_weights(onnx, np, source, output):
    """Save weights as float16 while leaving the inference graph in float32."""
    model = onnx.load(source, load_external_data=False)
    casts = []
    for weight in model.graph.initializer:
        if weight.data_type != onnx.TensorProto.FLOAT:
            continue
        values = onnx.numpy_helper.to_array(weight)
        if values.size < 1024 or not np.isfinite(values).all() or np.max(np.abs(values)) > 65504:
            continue
        name = weight.name
        weight.CopyFrom(onnx.numpy_helper.from_array(values.astype(np.float16), name + '__fp16'))
        casts.append(onnx.helper.make_node('Cast', [weight.name], [name], to=onnx.TensorProto.FLOAT))
    for node in reversed(casts):
        model.graph.node.insert(0, node)
    onnx.save(model, output)


def verify_export(path, runtime, torch, np, ort, checks, tolerance):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        for context, expected_logits, expected_tokens in checks:
            ids = np.array([context], dtype=np.int64)
            for index in range(max(1, len(expected_tokens))):
                scores = session.run(['logits'], {'input_ids': ids})[0]
                if index == 0 and not np.allclose(scores, expected_logits, atol=tolerance, rtol=tolerance):
                    return False
                if not expected_tokens:
                    break
                token = int(scores.argmax())
                if token != expected_tokens[index]:
                    return False
                ids = np.append(ids, [[token]], axis=-1)[:, -runtime.config.context_length:]
    return True


def export_model(runtime, record, directory):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    torch.set_num_threads(2)
    runtime.model.eval()

    class NextToken(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, ids):
            return self.model(ids, use_cache=False)['logits'][:, -1, :]

    wrapper = NextToken(runtime.model)
    output = directory / 'model.onnx'
    full = directory / 'full.onnx'
    sample = torch.tensor([[1, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (sample,), full, input_names=['input_ids'], output_names=['logits'],
            dynamic_axes={'input_ids': {1: 'sequence'}}, opset_version=18,
            dynamo=False, external_data=False,
        )
    onnx.checker.check_model(full)

    dialogs = [[{'role': 'user', 'content': text}] for text in PROMPTS]
    dialogs.append([
        {'role': 'user', 'content': 'Mi chiamo Ada.'},
        {'role': 'assistant', 'content': 'Piacere, Ada!'},
        {'role': 'user', 'content': 'Come mi chiamo?'},
    ])
    contexts = [runtime.tokenizer.serialize_messages(dialog) for dialog in dialogs]
    checks = []
    with torch.no_grad():
        for context in contexts + [[1], ([4, 8, 9, 7] * 64)[:runtime.config.context_length]]:
            tensor = torch.tensor([context], dtype=torch.long)
            expected_logits = wrapper(tensor).cpu().numpy()
            expected_tokens = []
            if context in contexts:
                generated = runtime.model.generate(tensor, max_new_tokens=48, temperature=0.0)[0].tolist()
                expected_tokens = generated[len(context):]
            checks.append((context, expected_logits, expected_tokens))

    compact_weights(onnx, np, full, output)
    if verify_export(output, runtime, torch, np, ort, checks, tolerance=0.04):
        print('Validated float16 weight storage with float32 execution.')
        full.unlink()
    else:
        if not verify_export(full, runtime, torch, np, ort, checks, tolerance=5e-4):
            raise RuntimeError('ONNX next-token or chat parity failed')
        print('Validated float32 export. Using it to preserve the native replies.')
        full.replace(output)
    onnx.checker.check_model(output)
    size = output.stat().st_size
    if not 0 < size <= MAX_BYTES:
        raise RuntimeError('Verified export exceeds the supported Android model size')
    tokenizer = (runtime.tokenizer.to_dict() if runtime.tokenizer.version == 'bpe-v1' else
                 {'version': runtime.tokenizer.version, 'vocab_size': runtime.tokenizer.vocab_size})
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {
        'schema': 1,
        'revision': record['revision'],
        'completed_steps': record['completed_steps'],
        'weights_digest': record['weights_digest'],
        'sha256': digest,
        'bytes': size,
        'context_length': runtime.config.context_length,
        'tokenizer': tokenizer,
    }


def publish(client, manifest, output, previous):
    revision = manifest['revision']
    digest = manifest['sha256']
    tag = f"airi-android-model-r{revision}-{digest[:12]}"
    source = os.environ['GITHUB_SHA']
    release = client.api('releases/tags/' + tag, missing=True)
    if release is None:
        subprocess.run([
            'gh', 'release', 'create', tag, str(output / 'model.onnx'), '--repo', REPOSITORY,
            '--target', source, '--title', f'Airi Android model r{revision}', '--prerelease',
            '--latest=false', '--notes', 'Verified native AIRI weights exported for Airi Chat on Android.',
        ], check=True, timeout=300)
        release = client.api('releases/tags/' + tag)
    assets = [asset for asset in release['assets'] if asset['name'] == 'model.onnx']
    if (release['draft'] or len(assets) != 1 or assets[0]['state'] != 'uploaded'
            or assets[0]['size'] != manifest['bytes'] or assets[0]['digest'] != 'sha256:' + digest):
        raise RuntimeError('Uploaded model does not match the validated local artifact')

    if client.api('git/ref/heads/' + BRANCH, missing=True) is None:
        client.api('git/refs', {'ref': 'refs/heads/' + BRANCH, 'sha': source})
    payload = {
        'message': f'Publish verified Airi Android model r{revision}',
        'branch': BRANCH,
        'content': base64.b64encode((json.dumps(manifest, indent=2) + '\n').encode()).decode(),
    }
    if previous is not None:
        payload['sha'] = previous['sha']
    client.api(f'contents/{POINTER}', payload, method='PUT')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.record) or (args.publish and args.checkpoint):
        parser.error('Use checkpoint and record together for a local check only')

    client = Client(REPOSITORY)
    record = validate(json.loads(args.record.read_text())) if args.record else read_current(client)
    previous = None
    if args.publish:
        previous = client.api(f'contents/{POINTER}?ref={BRANCH}', missing=True)
        if previous:
            deployed = json.loads(base64.b64decode(previous['content']))
            if deployed['revision'] > record['revision']:
                raise RuntimeError('Refusing to roll back the Android model')
            if deployed['revision'] == record['revision']:
                if deployed['weights_digest'] != record['weights_digest']:
                    raise RuntimeError('A published revision cannot be replaced with other weights')
                print(f"Android already follows AIRI revision {record['revision']}.")
                return

    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        from generalist_lm.runtime import GeneralistRuntime
        runtime = GeneralistRuntime.from_checkpoint(validate_selected_model(args.checkpoint, record), device='cpu')
    else:
        runtime, record = load_current(args.output / 'checkpoint-cache', record)
    manifest = export_model(runtime, record, args.output)
    (args.output / 'AIRI_ANDROID_CURRENT.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Validated revision {manifest['revision']}, {manifest['bytes']} bytes, SHA-256 {manifest['sha256']}.")
    if args.publish:
        publish(client, manifest, args.output, previous)


if __name__ == '__main__':
    main()
