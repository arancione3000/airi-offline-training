# Confronto con Qwen sugli stessi cambi di formato

Questa prova usa Qwen2.5-1.5B-Instruct Q4_K_M sullo stesso corpus congelato della [prova di adattamento](BENCHMARK.md). In ciascuna fase inserisce nel prompt le coppie di etichette confermate fino a quel momento. I nomi sono letti dal primo resoconto e la conferma dell’operatore stabilisce quale etichetta indica le unità spedite. Il modello riceve il testo dei resoconti e i vincoli del compito. Le risposte attese servono esclusivamente alla verifica esterna.

Una grammatica di decodifica richiede due operandi interi nell’ordine spedite, stornate oppure `REVISIONE`. Il programma effettua la sottrazione esatta dopo l’estrazione. Un’uscita incompleta o fuori dominio è trattata come revisione e contata a parte. Il prompt resta identico tra ammissione e prova finale, con l’unica aggiunta dei nomi confermati a ogni fase.

La verifica applica i 14 casi di ammissione in ciascuna fase e riesegue i 14 casi della prima dopo il secondo cambiamento. La prova riporta separatamente il superamento di queste verifiche e il comportamento sui 54 casi finali. Le previsioni finali sono comunque raccolte in valutazione, anche per una configurazione che avrebbe fallito l’ammissione. [Esecutore e prompt](qwen_benchmark.py)

## Esiti

| Fase | Risposte corrette su 20 richieste valide | Richieste valide da rivedere | Saldi errati su richieste valide | Astensioni corrette su 7 | Saldi indebiti |
| --- | ---: | ---: | ---: | ---: | ---: |
| Consegne | 16 | 0 | 4 | 0 | 7 |
| Ordini | 6 | 14 | 0 | 3 | 4 |
| Totale | 22 | 14 | 4 | 3 | 11 |

La prima configurazione fallisce 9 verifiche di ammissione su 14. La seconda ne fallisce 13 su 14 e fallisce 7 verifiche su 14 nella ripetizione dei casi della prima fase. Nessuna configurazione supera l’ammissione. Tutte le uscite finali rispettano il formato richiesto. I saldi errati o indebiti dipendono dalla scelta degli operandi o dalla decisione di rispondere. [Risultati completi](QWEN_RESULTS.json)

## Riproduzione

Servono gli stessi artefatti locali Qwen e llama.cpp usati in [joint-action-20261121](../joint-action-20261121/README.md). L’esecutore ne verifica le impronte prima di avviare il processo locale.

```sh
D=scripts/research-training/adaptive-format-repair
W=$(mktemp -d)
AIRI_JOINT_LLM_CACHE=/percorso/cache-qwen python "$D/qwen_benchmark.py" "$W/QWEN_RESULTS.json"
```

Il codice conserva il prompt e il rapporto conserva parametri, impronte, esiti e uscite grezze per caso. SHA-256 dell’esecutore prima dell’inferenza `6d5e9fec422a7d4ea367433128543c8461b1cc87126b2efef017c4ac0f1375fb`. Il risultato quantitativo riguarda questo modello quantizzato, questa configurazione e questo corpus sintetico.
