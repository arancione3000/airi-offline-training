# Espandere la ricerca di nuove procedure quando una formula semplice non basta

Il candidato parte dalle procedure già approvate. Cerca prima una formula di un'operazione. Se le risposte osservate e la verifica esaustiva non bastano, prova una composizione di due operazioni. Le operazioni di ordinamento sono riscritte in addizione, sottrazione, valore assoluto e divisione intera. La richiesta si considera soddisfatta soltanto dopo il confronto con il riferimento su tutte le 4.128 coppie del dominio. Il sintetizzatore può leggere il risultato atteso di un ingresso soltanto quando lo interroga. Il verificatore può leggere l'intera specifica. Un riferimento modificato su un punto mai interrogato ha respinto la formula e il banco precedente è rimasto disponibile. [Risultati per compito](evidence/SEARCH.json)

| Spazio di ricerca | Procedure ammesse | Risposte consultate |
| --- | ---: | ---: |
| Una sola operazione | 5/10 | 20 |
| Due operazioni aritmetiche | 7/10 | 22 |
| Composizione e ordinamento | 9/10 | 25 |

Sono state acquisite somma al quadrato, valore assoluto della somma, massimo e minimo dei valori assoluti. Il massimo comune divisore resta fuori dalle formule ammesse. Si tratta della stessa prova di copertura che aveva già evidenziato queste mancanze. L'estensione è stata progettata dopo averne conosciuto l'esito, quindi il risultato misura il recupero su quel test. Il candidato conserva le cinque procedure precedenti. La seconda fase considera un massimo di 12.000 ricette su un campione di 64 ingressi e valuta sul dominio completo al massimo 64 finaliste. Le domande sono limitate a 16 per compito e le proposte al verificatore a otto. [Prova di copertura precedente](../incremental-procedure-verification/audit/README.md)

Su dieci funzioni scritte indipendentemente senza leggere il candidato o la prima prova, la copertura passa da **1/10** con una sola operazione a **6/10** con due operazioni e **8/10** con le regole di ordinamento. I tre sistemi consultano rispettivamente 19, 21 e 23 risposte scalari. La parità del resto e la scelta del segno in funzione del divisore restano fuori. Le tabelle complete delle dieci nuove funzioni sono distinte fra loro e da quelle del primo test. I successi sono verificati su tutto il dominio, il campione resta una prova mirata di copertura aritmetica. [Prova aggiuntiva](fresh_coverage.py) · [Esiti](evidence/FRESH.json)

## Apprendimento con ripartenza

Il ciclo legge un manifesto di compiti con risultati indipendenti e impronta fornita dal verificatore. Registra atomicamente l'esito per compito e stato del grafo. A ogni avvio riqualifica le procedure approvate contro lo stesso riferimento. Può ritentare una procedura respinta quando una nuova ammissione arricchisce il grafo. Con il grafo invariato evita tentativi identici. Modifiche ai risultati già usati o corruzione del registro bloccano il ciclo. L'opzione `--drain` elabora tutti i compiti eleggibili del manifesto corrente.

Il replay avvia processi separati. Acquisisce la somma al quadrato, riapre il registro e poi riceve un manifesto esteso. Acquisisce il minimo dei valori assoluti e ritenta il massimo comune divisore a seguito del cambiamento del grafo. L'ultima riapertura trova due nuove procedure approvate e zero tentativi pendenti. Questo è un ciclo su dati aritmetici di laboratorio. Il catalogo accetta al massimo 16 compiti, il registro 64 tentativi e il grafo 128 nodi. Il registro è associato alla versione del motore. Per evolvere richiede nuovi riferimenti attendibili e l'avvio del ciclo per il manifesto aggiornato. [Replay](replay_cycle.py) · [Esiti](evidence/RESTART.json)

## Riproduzione

```sh
D=scripts/research-training/procedural-capacity-growth
W=$(mktemp -d)
python -m unittest discover -s "$D" -v
python "$D/synthesis_growth.py" "$W/search.json"
python "$D/fresh_coverage.py" "$W/fresh.json"
python "$D/replay_cycle.py" "$W/restart"
```

Per alimentare il processo con un altro compito, usare un file JSON con schema 1 e un campo `tasks`. Ogni chiave identifica un compito e contiene 4.128 risultati interi nell'ordine di `fixtures.INPUTS`. Si possono aggiungere nuovi compiti mantenendo i riferimenti dei compiti già osservati. Passare un percorso di stato persistente e l'impronta ricevuta dal fornitore del riferimento.

```sh
python "$D/cycle.py" --reference "$REFERENCE" --reference-sha256 "$TRUSTED_REFERENCE_SHA" --state "$STATE" --drain
```

L'espansione del modello nativo è valutata nel suo esperimento separato. [Espansione e verifica dei pesi nativi](../native-capacity-growth/README.md)

## Aprire il banco nella chat

La chat può eseguire le procedure apprese su richiesta esplicita. All'apertura ricostruisce il registro, controlla il riferimento indipendente e riqualifica tutte le formule approvate. Le risposte indicano gli operandi originali, il messaggio di provenienza e l'impronta del banco. Memoria, abilità già installate e formulazioni approvate mantengono precedenza.

```sh
python scripts/assistant-runtime/chat.py \
  --procedures "$STATE" --procedure-reference "$REFERENCE" --procedure-reference-sha256 "$TRUSTED_REFERENCE_SHA"
```

Esempio di conversazione dopo l'ammissione di `squared_sum`.

```text
Tu: Valori procedura: -13 e 5.
Tu: Esegui procedura squared_sum.
Airi [procedura verificata]: 64
```

Serve un solo messaggio utente con i valori, seguito dalla domanda esatta. Il dominio resta quello di 4.128 coppie. La chat può assorbire un nuovo registro nella stessa sessione quando il ciclo completa altre procedure sotto la stessa impronta del riferimento. Prima di usare un registro aggiornato, la chat lo riqualifica e verifica che contenga ancora tutte le procedure già approvate nella sessione. La nuova sessione è necessaria quando cambia il riferimento autorizzato. L'opzione è esplicita. Il replay comprende registri prodotti dalla versione precedente, che il nuovo lettore ricontrolla prima dell'uso. [Prova della chat](chat_probe.py)

La prova riproducibile esegue 16 richieste su operandi non interrogati durante l'apprendimento, verifica due invocazioni del comando di chat e controlla 18 richieste su cui il nuovo percorso deve astenersi. I 989 casi storici, i 600 esempi del banco e le 198 formulazioni approvate hanno gli stessi esiti prima e dopo. La prova usa un segnaposto per il fallback nativo. Nella stessa sessione vede un'abilità appena approvata, respinge un registro precedente incompleto e recupera quando torna disponibile una copia valida. [Risultati](evidence/CHAT.json)

```sh
python "$D/chat_probe.py" "$W/chat"
```
