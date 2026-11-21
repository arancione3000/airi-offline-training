"""Offline frozen Qwen reference for joint Italian action and object decisions.

The action and object labels may be combined freely during decoding. Only the
caller-side compatibility check knows which operation an object implements.
Weights: Qwen/Qwen2.5-1.5B-Instruct-GGUF, Apache-2.0, revision MODEL_REVISION.
CPU engine: ggml-org/llama.cpp b5963, MIT (Linux x64 release).
Set AIRI_JOINT_LLM_CACHE to a directory containing model.gguf, ENGINE_ZIP
and its extracted build/bin directory. The process uses a private Unix socket.
"""
import atexit
from functools import lru_cache
import hashlib
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from zipfile import ZipFile


MODEL_REVISION = '91cad51170dc346986eccefdc2dd33a9da36ead9'
MODEL_SHA256 = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
ENGINE_ZIP = 'llama-b5963-bin-ubuntu-x64.zip'
ENGINE_SHA256 = '3109eb170fa18fb89519f999c2aa6596a9c8302d6ca75e7e33953f55cf22f257'
ACTIONS = {'copia': 'copy', 'inverti': 'reverse', 'maiuscolo': 'uppercase',
           'minuscolo': 'lowercase', 'somma': 'add', 'sottrai': 'subtract',
           'moltiplica': 'multiply', 'altro': None}


SYSTEM = """Interpreta una richiesta italiana. Emetti soltanto l'azione richiesta e l'oggetto a cui si riferisce, nel formato azione | oggetto. Decidi i due campi separatamente.
Azioni possibili:
- copia: leggere o riportare esattamente il valore, invariato.
- inverti: dare tutti i caratteri in ordine inverso.
- maiuscolo o minuscolo: convertire tutte le lettere nel caso specificato.
- somma: addizionare i due numeri.
- sottrai: restituire il primo numero meno il secondo.
- moltiplica: fare il prodotto dei due numeri.
- altro: qualsiasi altra azione. Usa altro anche per richieste negate, citate, ipotetiche, spiegazioni, trasformazioni parziali o più istruzioni. Una domanda cortese che chiede davvero il risultato vale come richiesta diretta.
Cerca l'oggetto nominato fra quelli del catalogo, anche se espresso con parole diverse. Rispondi nessuno per un oggetto estraneo, un riferimento ambiguo o più oggetti. Un oggetto simile ma diverso da quello richiesto è estraneo. Puoi scegliere qualsiasi azione con qualsiasi oggetto.
Esempi del formato:
«Qual è il codice del pacco?» → copia | il codice del pacco
«Rimandami il segnale acquisito letto a ritroso.» → inverti | il segnale acquisito
«Metti in maiuscolo il testo della tessera.» → maiuscolo | il testo della tessera
«Converti il testo della targa in minuscolo.» → minuscolo | il testo della targa
«A quanto ammonta la somma dei due interi sigma?» → somma | i due interi sigma
«Dei due valori del saldo, togli il secondo dal primo.» → sottrai | i due valori del saldo
«Moltiplica il prezzo unitario per la quantità.» → moltiplica | il prezzo unitario e la quantità
«Per il codice del pacco contami le cifre.» → altro | il codice del pacco
«Scrivimi in minuscolo il nome del museo.» → minuscolo | nessuno
Catalogo degli oggetti:
"""


def _hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _verify_engine(cache):
    archive = cache / ENGINE_ZIP
    if _hash(archive) != ENGINE_SHA256:
        raise ValueError('llama.cpp archive differs from frozen reference')
    with ZipFile(archive) as release:
        files = [name for name in release.namelist() if name.endswith(('llama-server', '.so'))]
        for name in files:
            if _hash(cache / name) != hashlib.sha256(release.read(name)).hexdigest():
                raise ValueError(f'runtime differs from pinned archive: {name}')
    return cache / 'build/bin/llama-server'


class LocalConnection(HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost', timeout=120)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class Teacher:
    def __init__(self, families):
        cache = Path(os.environ['AIRI_JOINT_LLM_CACHE']).resolve()
        model = cache / 'model.gguf'
        if _hash(model) != MODEL_SHA256:
            raise ValueError('Qwen GGUF differs from frozen reference')
        server = _verify_engine(cache)
        self.by_name = {family['object_phrase']: family for family in families}
        if len(self.by_name) != len(families) or 'nessuno' in self.by_name:
            raise ValueError('catalog object names must be unique')
        objects = [*sorted(self.by_name), 'nessuno']
        def alternatives(labels):
            return ' | '.join(json.dumps(x, ensure_ascii=False) for x in labels)
        grammar = (f'root ::= action " | " object\naction ::= {alternatives(ACTIONS)}\n'
                   f'object ::= {alternatives(objects)}\n')
        self.grammar = grammar
        self.work = tempfile.TemporaryDirectory(prefix='airi-llama-')
        self.socket = Path(self.work.name) / 'model.sock'
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen([
            str(server), '--model', str(model), '--host', str(self.socket), '--ctx-size', '1536',
            '--threads', '2', '--threads-batch', '2', '--parallel', '1', '--no-webui',
        ], cwd=server.parent, stdout=self.log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        try:
            deadline = time.monotonic() + 120
            while self.process.poll() is None and time.monotonic() < deadline:
                if self.socket.exists():
                    status, _ = self._request('/health')
                    if status == 200:
                        break
                time.sleep(.1)
            else:
                self.log.seek(max(0, self.log.tell() - 2000))
                raise RuntimeError('local llama.cpp startup failed: ' + self.log.read().decode(errors='replace'))
        except BaseException:
            self.close()
            raise
        instructions = SYSTEM + '\n'.join('- ' + x for x in sorted(self.by_name))
        self.prefix = '<|im_start|>system\n' + instructions + '<|im_end|>\n<|im_start|>user\n'
        self._stats = {'calls': 0, 'prompt_tokens': 0, 'generated_tokens': 0}

    def _request(self, path, data=None):
        conn = LocalConnection(self.socket)
        try:
            body = None if data is None else json.dumps(data, ensure_ascii=False).encode()
            conn.request('GET' if body is None else 'POST', path, body,
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.work.cleanup()
        atexit.unregister(self.close)

    @lru_cache(maxsize=4096)
    def _decision(self, query):
        if not isinstance(query, str) or not 0 < len(query) <= 512 or not query.isprintable() or '<|' in query:
            return None, None, ''
        prompt = self.prefix + query + '<|im_end|>\n<|im_start|>assistant\n'
        status, result = self._request('/completion', {
            'prompt': prompt, 'grammar': self.grammar, 'temperature': 0.0, 'repeat_penalty': 1.0,
            'n_predict': 40, 'seed': 7, 'cache_prompt': True,
        })
        if status != 200:
            raise RuntimeError(f'local llama.cpp returned HTTP {status}: {result}')
        self._stats['calls'] += 1
        self._stats['prompt_tokens'] += result['timings']['prompt_n']
        self._stats['generated_tokens'] += result['tokens_predicted']
        raw = result['content'].strip()
        action, sep, obj = raw.partition(' | ')
        if not sep or action not in ACTIONS or (obj != 'nessuno' and obj not in self.by_name):
            return None, None, raw
        return ACTIONS[action], self.by_name.get(obj, {}).get('id'), raw

    def trace(self, query):
        op, family, raw = self._decision(query)
        return {'operation': op, 'object_id': family, 'raw_plan': raw}

    def __call__(self, query):
        op, family, raw = self._decision(query)
        if family is None or op is None:
            return None
        return family if self.by_name[raw.partition(' | ')[2]]['operation'] == op else None

    def stats(self):
        return dict(self._stats)


def build(families):
    return Teacher(families)
