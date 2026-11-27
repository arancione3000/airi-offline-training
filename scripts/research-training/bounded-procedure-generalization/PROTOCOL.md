# Protocollo per l’apprendimento di procedure da esempi

Il corpus contiene 13 compiti sintetici su testo, sequenze e record. Le risposte di riferimento sono calcolate da funzioni Python scritte per la valutazione, indipendenti dal motore di ricerca. La scelta dei compiti conosce le primitive disponibili. Il generatore e il corpus sono stati congelati prima di osservare i risultati del candidato.

| Insieme | Casi | Uso |
| --- | ---: | --- |
| Addestramento | 52 | Quattro coppie ingresso-risposta per compito |
| Pool | 78 | Sei ingressi per compito, etichette disponibili solo tramite interrogazione |
| Qualificazione | 94 | Verifica della procedura proposta prima dell’ammissione |
| Finale | 206 | Valutazione separata dopo le decisioni di ammissione |

Il corpus ha schema 1 e seme `281993`. Ogni compito espone `id`, `domain`, `kind` e i quattro insiemi `train`, `pool`, `qualification` e `final`. Ogni riga contiene soltanto `input` ed `expected`. Gli ingressi sono distinti fra tutti gli insiemi dello stesso compito. Domini e categorie servono al rapporto. Il candidato può vedere le righe di addestramento e gli ingressi del pool, senza ricevere categorie, funzioni di riferimento o risposte degli insiemi successivi.

## Casi e criteri

Tre compiti per dominio sono costruiti per richiedere la composizione di trasformazioni. Comprendono pulizia e ricomposizione di testo, aggregazione o riordinamento di sequenze, e selezione di un campo seguita da trasformazioni. Nei casi finali cambiano valori, lunghezze, caratteri e campi accessori. I 34 ingressi esplicitamente destinati alla revisione hanno `expected: null` e appartengono tutti all’insieme finale. Le verifiche di ammissione del corpus contengono solo casi positivi.

Un compito usa esempi e pool su cui regole diverse coincidono. Le risposte vere divergono quando compaiono spazi o maiuscole nella qualificazione e nel finale. Tre compiti richiedono operazioni fuori dalla grammatica dichiarata, una per dominio. I risultati di questi quattro compiti vanno riportati separatamente. Per i casi con un risultato atteso si distinguono risposta corretta, risposta errata e astensione. Per i casi con `expected: null` si misura l’astensione.

Gli abbinamenti opportuni sono la ricerca a uno e a più passi e, a pari budget di etichette, la scelta in ordine rispetto alla scelta per disaccordo. Un controllo di memorizzazione può consultare solo le coppie di addestramento. Le risposte di qualificazione servono soltanto a decidere l’ammissione. I casi finali restano fuori da scelta del programma, interrogazioni e correzioni.

## Riproduzione

```sh
D=scripts/research-training/bounded-procedure-generalization
W=$(mktemp -d)
python "$D/oracle_cases.py" "$W/CASES.json"
cmp "$D/CASES.json" "$W/CASES.json"
sha256sum "$D/CASES.json"
```

SHA-256 di `CASES.json` `b1e67abc956eec9161c2d524d92bf25470d6d41bed2d545fd764a7e3027f02e4`.

La prova misura l’induzione di procedure brevi su dati sintetici entro un insieme di operazioni dichiarate. Due record finali contengono anche metadati annidati accessori. Le conclusioni riguardano questo insieme di compiti e richiedono una verifica distinta per documenti reali, nuovi domini e ragionamento generale.
