# API pubblica di Airi

L’API calcola il saldo fra unità spedite e stornate. Qui *stornate* indica unità annullate o rese. È una prova pubblica gratuita, senza account o chiave. [Endpoint e esempio](https://airi-api.arancione3000.chatgpt.site/)

## Prima richiesta

```sh
curl https://airi-api.arancione3000.chatgpt.site/v1/reconcile \
  -H 'Content-Type: application/json' \
  -d '{"spedite":12,"stornate":3}'
```

```json
{"status":"answered","answer":9,"spedite":12,"stornate":3}
```

Si può inviare il testo già estratto da uno o più resoconti. Per esempio, usa il corpo JSON `{"reports":["spedite: 12; stornate: 3"]}`. Con quantità incompatibili, la risposta è `{"status":"review","reason":"conflicting_reports"}`. Un resoconto con due etichette nuove riceve `status: clarification` e le etichette trovate. Questa versione dell’API accetta i campi noti `spedite` e `stornate` e ogni richiesta viene elaborata senza salvare una regola. Per approvare i due nuovi formati di esempio usa la [demo nel browser](https://airi-riconciliazione.arancione3000.chatgpt.site/).

## Formato e limiti

- `POST /v1/reconcile` accetta JSON con due quantità intere da 0 a 1.000.000 o al massimo 16 resoconti di 256 caratteri ciascuno. Un saldo può essere negativo.
- Il corpo può occupare al massimo 16 KiB. I resoconti testuali contengono due campi con `:` o `=`, separati da un punto e virgola o da un a capo.
- Le risposte `answered`, `review` e `clarification` usano HTTP 200. Input invalido, tipo di contenuto errato e corpo troppo grande ricevono rispettivamente HTTP 400, 415 e 413.
- La prova riceve JSON o testo dentro JSON. PDF, foto ed Excel richiedono un’estrazione prima della chiamata. Per provare l’endpoint, usa dati inventati.

Dalla radice del repository, esegui i test con Node.js 22 o successivo.

```sh
node --test scripts/research-training/adaptive-format-repair/api/worker.test.mjs
```
