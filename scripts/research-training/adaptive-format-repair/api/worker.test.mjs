import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import worker from './worker.mjs';

const corpus = JSON.parse(await readFile(new URL('../CASES.json', import.meta.url), 'utf8'));
const url = 'https://example.com';
const request = (method, path, options) => worker.fetch(new Request(url + path, { method, ...options }));
const post = (value) => request('POST', '/v1/reconcile', {
  headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value),
});

test('public instructions, structured calculation, and negative balances', async () => {
  assert.equal((await request('GET', '/')).status, 200);
  assert.deepEqual(await (await post({ spedite: 12, stornate: 3 })).json(), {
    status: 'answered', answer: 9, spedite: 12, stornate: 3,
  });
  assert.equal((await (await post({ spedite: 0, stornate: 1_000_000 })).json()).answer, -1_000_000);
});

for (const row of corpus.protected) {
  test(`frozen case ${row.id}`, async () => {
    const response = await post({ reports: row.reports });
    assert.equal(response.status, 200);
    const result = await response.json();
    if (row.expected === null) {
      assert.equal(result.status, 'review');
      assert.equal('answer' in result, false);
    } else {
      assert.equal(result.status, 'answered');
      assert.equal(result.answer, row.expected);
    }
  });
}

test('unknown format asks for clarification without approval or a balance', async () => {
  assert.deepEqual(await (await post({ reports: ['consegnate: 12; rese: 3'] })).json(), {
    status: 'clarification', reason: 'unknown_format', labels: ['consegnate', 'rese'],
  });
});

test('status codes, method and media type, preflight, malformed JSON and byte limit', async () => {
  assert.equal((await request('GET', '/wrong')).status, 404);
  assert.equal((await request('GET', '/v1/reconcile')).status, 405);
  assert.equal((await request('OPTIONS', '/v1/reconcile')).headers.get('Access-Control-Allow-Origin'), '*');
  assert.equal((await request('POST', '/v1/reconcile', { body: '{}' })).status, 415);
  assert.equal((await request('POST', '/v1/reconcile', {
    headers: { 'Content-Type': 'application/json' }, body: '{',
  })).status, 400);
  assert.equal((await post({ reports: ['é'.repeat(10_000)] })).status, 413);
  for (const value of [{ spedite: 1, stornate: -1 }, { spedite: 1.1, stornate: 0 },
    { spedite: '12', stornate: 3 }, { reports: [12] }, { reports: Array(17).fill('x') }]) {
    assert.equal((await post(value)).status, 400);
  }
});
