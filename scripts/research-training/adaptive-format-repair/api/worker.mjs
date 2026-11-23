const MAX_BODY = 16 * 1024;
const MAX_QUANTITY = 1_000_000;
const MAX_REPORTS = 16;
const PATH = '/v1/reconcile';
const HEADERS = {
  'Content-Type': 'application/json; charset=utf-8',
  'Cache-Control': 'no-store',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

const json = (status, body) => new Response(JSON.stringify(body), { status, headers: HEADERS });
const review = (reason) => ({ status: 'review', reason });
const validQuantity = (value) => Number.isInteger(value) && value >= 0 && value <= MAX_QUANTITY;
const answer = (spedite, stornate) => ({ status: 'answered', answer: spedite - stornate, spedite, stornate });

function parse(report) {
  if ([...report].length > 256) return null;
  const pieces = report.trim().split(/;|\r?\n/);
  if (pieces.length !== 2) return null;
  const fields = new Map();
  for (const piece of pieces) {
    const field = /^([^:=;\r\n]+?)\s*[:=][ \t]*([0-9]{1,7})[ \t]*$/u.exec(piece.trim());
    if (!field) return null;
    const name = field[1].trim().replace(/\s+/gu, ' ').toLowerCase().normalize('NFC');
    const quantity = Number(field[2]);
    if (!/^[\p{L} _-]{1,48}$/u.test(name) || !/\p{L}/u.test(name) ||
        !validQuantity(quantity) || fields.has(name)) return null;
    fields.set(name, quantity);
  }
  return fields;
}

function reconcile(reports) {
  if (reports.length === 0) return review('no_reports');
  let first;
  for (const report of reports) {
    const fields = parse(report);
    if (!fields) return review('malformed_report');
    if (!fields.has('spedite') || !fields.has('stornate')) {
      return reports.length === 1
        ? { status: 'clarification', reason: 'unknown_format', labels: [...fields.keys()].sort() }
        : review('unrecognized_format_among_multiple');
    }
    const pair = answer(fields.get('spedite'), fields.get('stornate'));
    if (first && (first.spedite !== pair.spedite || first.stornate !== pair.stornate)) {
      return review('conflicting_reports');
    }
    first = pair;
  }
  return first;
}

async function readJson(request) {
  const reader = request.body?.getReader();
  if (!reader) return null;
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let size = 0;
  let text = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY) {
        await reader.cancel();
        return { tooLarge: true };
      }
      text += decoder.decode(value, { stream: true });
    }
    return { value: JSON.parse(text + decoder.decode()) };
  } catch {
    return null;
  } finally {
    reader.releaseLock();
  }
}

const example = { spedite: 12, stornate: 3 };

export default {
  async fetch(request) {
    const { pathname } = new URL(request.url);
    const { method } = request;
    if (pathname === '/' && method === 'GET') {
      return json(200, {
        name: 'Airi API', version: '1', access: 'public preview, no key',
        endpoint: `POST ${PATH}`, example, response: answer(example.spedite, example.stornate),
        text_example: { reports: ['Spedite: 12; Stornate: 3'] },
        docs: 'https://github.com/arancione3000/airi-offline-training/tree/main/scripts/research-training/adaptive-format-repair/api',
      });
    }
    if (pathname !== PATH) return json(404, { error: 'not_found' });
    if (method === 'OPTIONS') return new Response(null, { status: 204, headers: HEADERS });
    if (method !== 'POST') {
      const response = json(405, { error: 'method_not_allowed' });
      response.headers.set('Allow', 'POST, OPTIONS');
      return response;
    }
    if (request.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'application/json') {
      return json(415, { error: 'expected_json' });
    }
    if (Number(request.headers.get('Content-Length')) > MAX_BODY) return json(413, { error: 'request_too_large' });
    const payload = await readJson(request);
    if (payload?.tooLarge) return json(413, { error: 'request_too_large' });
    const body = payload?.value;
    if (!body || typeof body !== 'object' || Array.isArray(body)) return json(400, { error: 'invalid_json_object' });
    const keys = Object.keys(body);
    if (keys.length === 2 && validQuantity(body.spedite) && validQuantity(body.stornate)) {
      return json(200, answer(body.spedite, body.stornate));
    }
    if (keys.length === 1 && Array.isArray(body.reports) && body.reports.length <= MAX_REPORTS &&
        body.reports.every((item) => typeof item === 'string')) {
      return json(200, reconcile(body.reports));
    }
    return json(400, { error: 'invalid_input' });
  },
};
