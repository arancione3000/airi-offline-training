"""Versioned Android chat API for the approved Airi runtime."""
import argparse
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import sys
from threading import BoundedSemaphore, Event, Lock, Thread

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/assistant-runtime'), str(ROOT / 'scripts/transfer-training')]
from assistant import AiriAssistant
from chat_current import load_current, validate_selected_model
from flexible_skills import default_router
from growth_routes import follow_verified_procedures, with_verified_procedures
from learn import BUNDLED
from model_channel import Client, REPOSITORY, read_current, validate
from skill_bank import SkillBank

MAX_BODY = 65536
LOG = logging.getLogger('airi.mobile')


def conversation(raw):
    data = json.loads(raw)
    if type(data) is not dict or set(data) != {'messages'}:
        raise ValueError('Invia un oggetto con messages.')
    messages = data['messages']
    if type(messages) is not list or not 1 <= len(messages) <= 48 or len(messages) % 2 == 0:
        raise ValueError('Cronologia non valida.')
    if any(type(msg) is not dict or set(msg) != {'role', 'content'}
           or msg['role'] != ('assistant' if i % 2 else 'user')
           or type(msg['content']) is not str or len(msg['content']) > 4096 or not msg['content'].strip()
           or any(0xd800 <= ord(c) <= 0xdfff for c in msg['content'])
           for i, msg in enumerate(messages)):
        raise ValueError('Messaggio non valido.')
    return messages


class Model:
    def __init__(self, cache, checkpoint=None, record=None):
        self.cache = cache
        self.lock = Lock()
        self.current = None
        if checkpoint:
            from generalist_lm.runtime import GeneralistRuntime
            self.current = (GeneralistRuntime.from_checkpoint(validate_selected_model(checkpoint, record)), record)

    def refresh(self):
        record = read_current(Client(REPOSITORY))
        validate(record)
        with self.lock:
            if self.current and self.current[1]['revision'] >= record['revision']:
                return
        runtime, loaded = load_current(self.cache, record)
        with self.lock:
            if self.current is None or self.current[1]['revision'] < loaded['revision']:
                self.current = (runtime, loaded)

    def revision(self):
        with self.lock:
            return self.current[1]['revision'] if self.current else None

    def answer(self, messages):
        with self.lock:
            current = self.current
        if current is None:
            raise ModelUnavailable
        runtime, record = current
        return runtime.chat(messages, temperature=0, max_new_tokens=96), record['revision']


class ModelUnavailable(Exception):
    pass


def make_handler(token, model, bank_path=BUNDLED, procedures=None):
    bank = SkillBank(bank_path)
    inference = BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, pattern, *args):
            return  # Keep questions and credentials out of request logs.

        def send_json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionError, TimeoutError):
                pass

        def authorized(self):
            supplied = self.headers.get('Authorization', '').encode('utf-8')
            if compare_digest(supplied, ('Bearer ' + token).encode()):
                return True
            self.send_json(401, {'error': 'Chiave di accesso errata.'})
            return False

        def do_GET(self):
            if self.authorized():
                if self.path == '/v1/status':
                    revision = model.revision()
                    self.send_json(200, {'api_version': 1, 'model_revision': revision,
                                         'model_ready': revision is not None})
                else:
                    self.send_json(404, {'error': 'Indirizzo non trovato.'})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/v1/chat':
                self.send_json(404, {'error': 'Indirizzo non trovato.'})
                return
            if self.headers.get('Transfer-Encoding') or self.headers.get_content_type() != 'application/json':
                self.send_json(415, {'error': 'Invia JSON con Content-Length.'})
                return
            length = self.headers.get('Content-Length', '')
            if not length.isascii() or not length.isdecimal() or len(length) > 8 or not 0 < int(length) <= MAX_BODY:
                self.send_json(413, {'error': 'Richiesta troppo lunga o senza lunghezza valida.'})
                return
            try:
                body = self.rfile.read(int(length))
                if len(body) != int(length):
                    raise ValueError('Richiesta incompleta.')
                messages = conversation(body)
            except (ValueError, TimeoutError, RecursionError):
                self.send_json(400, {'error': 'Messaggi non validi o incompleti.'})
                return
            if not inference.acquire(blocking=False):
                self.send_json(429, {'error': 'Airi è occupata. Riprova tra poco.'})
                return
            try:
                answered_revision = None

                def native(history):
                    nonlocal answered_revision
                    answer, answered_revision = model.answer(history)
                    return answer

                router = with_verified_procedures(default_router(bank), procedures)
                result = AiriAssistant(native, bank, router).respond(messages)
                self.send_json(200, {'answer': result['answer'], 'source': result['source'],
                                     'model_revision': model.revision() if answered_revision is None else answered_revision})
            except ModelUnavailable:
                self.send_json(503, {'error': 'Il modello di Airi si sta caricando.'})
            except Exception:
                LOG.exception('Chat processing failed')
                self.send_json(500, {'error': 'Airi non è riuscita a rispondere.'})
            finally:
                inference.release()

    return Handler


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--bind', default='127.0.0.1')
    cli.add_argument('--port', type=int, default=8787)
    cli.add_argument('--cache', type=Path, default=Path.home() / '.cache/airi/mobile-model')
    cli.add_argument('--checkpoint', type=Path)
    cli.add_argument('--model-record', type=Path)
    cli.add_argument('--refresh-seconds', type=int, default=300)
    cli.add_argument('--offline', action='store_true')
    cli.add_argument('--procedures', type=Path)
    cli.add_argument('--procedure-reference', type=Path)
    cli.add_argument('--procedure-reference-sha256')
    opts = cli.parse_args()
    token = os.environ.get('AIRI_CHAT_TOKEN', '')
    if not 24 <= len(token) <= 256 or not token.isascii() or any(not 33 <= ord(c) <= 126 for c in token):
        cli.error('Imposta AIRI_CHAT_TOKEN con almeno 24 caratteri ASCII stampabili senza spazi.')
    if bool(opts.checkpoint) != bool(opts.model_record) or (opts.offline and not opts.checkpoint):
        cli.error('Checkpoint e record vanno forniti insieme. Sono necessari in modalità offline.')
    if not 30 <= opts.refresh_seconds <= 3600:
        cli.error('Intervallo di aggiornamento da 30 a 3600 secondi.')
    growth = (opts.procedures, opts.procedure_reference, opts.procedure_reference_sha256)
    if any(value is not None for value in growth) and not all(growth):
        cli.error('Registro, riferimento e impronta procedurale sono richiesti insieme.')
    try:
        procedures = follow_verified_procedures(*growth) if all(growth) else None
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        cli.error(f'Procedure verificate non disponibili ({error}).')
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    record = json.loads(opts.model_record.read_bytes()) if opts.model_record else None
    model = Model(opts.cache, opts.checkpoint, record)
    stop = Event()

    def refresh():
        while not stop.is_set():
            try:
                model.refresh()
            except Exception:
                LOG.warning('Aggiornamento modello non disponibile. Resta attiva la revisione caricata.')
            if stop.wait(opts.refresh_seconds):
                return

    with ThreadingHTTPServer((opts.bind, opts.port), make_handler(token, model, procedures=procedures)) as server:
        if not opts.offline:
            Thread(target=refresh, daemon=True).start()
        LOG.info('Airi chat API http://%s:%s', opts.bind, opts.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()


if __name__ == '__main__':
    main()
