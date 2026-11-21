# Riconoscere insieme l’oggetto e l’azione

Il nuovo interprete estrae dal testo della domanda sia l’oggetto nominato sia l’operazione che l’utente chiede di eseguire. Può proporre una procedura solo quando le due decisioni concordano. La chat usa la procedura approvata per calcolare la risposta e controlla da quale messaggio provengono i dati. Il vocabolario dell’interprete è stato scritto e rivisto per sette operazioni e diciotto famiglie; il componente che trasferisce nel banco le domande convalidate opera automaticamente. [Protocollo](PROTOCOL.md)

## Prima prova indipendente

Un autore separato ha prodotto 54 comandi validi e 54 richieste incompatibili. Ogni punto richiede due esecuzioni con dati diversi, risultato corretto, procedura corretta e provenienza corretta. Il banco conteneva 30 esperti e 112 formulazioni alternative. I percorsi approvati avevano la precedenza. Su 108 testi, un negativo coincideva dopo normalizzazione con una domanda di una prova precedente. Escludendolo, i conteggi delle risposte corrette e delle attivazioni indebite rimangono gli stessi su 54 positivi e 53 negativi distinti.

| Percorso | Corrette sui 54 comandi | Attivazioni sui 54 casi incompatibili |
| --- | ---: | ---: |
| Formulazioni approvate al momento del congelamento | 0 | 0 |
| Interprete precedente con riconoscimento dell’oggetto | 6 | 0 |
| Nuovo interprete oggetto-azione | 30 | 6 |
| Qwen2.5-1.5B-Instruct, Q4_K_M | 22 | 22 |

Sui comandi validi il nuovo interprete ha guadagnato 25 risposte e ne ha persa una rispetto al precedente. Ha risolto 15/18 richieste con un nome vicino al canonico, 11/18 con nome parafrasato e 4/18 con descrizione contestuale. Tutti i metodi hanno preservato i 600 esempi registrati e i 989 contratti storici. Il controllo su quei contratti riguarda l’instradamento. Il fallback nativo è sostituito da un segnaposto che registra l’astensione. [Esiti per formulazione](evidence/first/observations.json.gz) · [Risultati completi](evidence/first/results.json.gz)

Qwen è un piccolo Transformer aperto, eseguito su CPU con pesi quantizzati congelati e un prompt italiano con esempi. La decodifica vincola il formato dei due campi e permette di dichiarare un oggetto o un’azione estranei. Il calcolo dei valori e i controlli nel contesto sono identici per tutti i metodi. Sulla stessa macchina, la mediana della prima risposta procedurale per nuova frase è stata 3,42 ms per l’interprete simbolico e 841,58 ms per il classificatore Qwen. Il tempo è relativo a questo compito e a questa configurazione. [Misure Qwen](evidence/qwen/results.json.gz)

## Conferma del filtro sulle richieste

Abbiamo usato gli esiti della prima prova per distinguere meglio una richiesta diretta da una domanda sul significato, su un attributo o su una seconda azione. Il filtro ha risolto 46/54 comandi sul primo insieme, ormai usato per lo sviluppo, e ha respinto tutti i suoi 54 negativi. Su un secondo insieme indipendente, senza testi coincidenti dopo normalizzazione, è passato da 25 a 28 comandi corretti su 36. I due sistemi hanno entrambi prodotto cinque risposte indebite sui 54 casi incompatibili. Tre riguardavano richieste indirette o condizionali, una un oggetto estraneo e una la modifica di una sola lettera invece di tutto il testo. La soglia ammetteva zero risposte indebite. L’interpretazione libera resta sperimentale e il test finale riservato è rimasto chiuso. [Prova di conferma](confirmation.json) · [Misure e confronto appaiato](evidence/confirmation/results.json.gz)

## Trasferimento nella chat

Il catalogo di addestramento già etichettato contiene 70 richieste supportate e 70 incompatibili. L’interprete congelato ne ha proposte 48 supportate e otto incompatibili. Il controllo sulle etichette ha escluso tutte e otto le proposte errate. Le 48 validate coprono tutte e sette le operazioni. Sei cicli ordinari di otto ammissioni portano il banco da 150 a 198 formulazioni alternative. Il settimo ciclo ripete i controlli e trova zero aggiunte. Questa procedura richiede nuovi esempi attendibili per estendere ulteriormente il catalogo. [Previsioni congelate](donor_decisions.json) · [Esiti dei cicli](evidence/admission.json)

Una verifica separata sulle 48 formulazioni usa sei valori per ciascuna, tra cui interi negativi, zero e testo Unicode. La chat produce 0/288 risposte procedurali corrette prima del trasferimento e 288/288 dopo. Su 240 conversazioni con fonte mancante, incompatibile o ambigua si astiene in entrambi i casi. Ad esempio, dati i valori della scorta −23 e 41, alla domanda «Cosa rimane togliendo il secondo numero dal primo per i due valori della scorta?» risponde −64. Si misura qui l’uso di una formulazione confermata su dati nuovi. [Risposte e provenienza](evidence/transfer-values.json.gz)

## Riproduzione

Il confronto linguistico usa il banco congelato alla revisione `7137c60`. Il trasferimento delle formulazioni inizia dal catalogo a 150 voci nella revisione `eabf114`. Da un checkout del codice dell’esperimento con Python 3.12 e Git si possono ripetere le due prove e il trasferimento in una cartella temporanea.

```sh
D=scripts/research-training/joint-action-20261121
W=$(mktemp -d)
git show 7137c60:scripts/assistant-runtime/skills/approved-bank.json > "$W/bank.json"
git show 7137c60:scripts/assistant-runtime/queries/approved-aliases.json > "$W/baseline.json"
python "$D/run.py" --aliases "$W/baseline.json" --bank "$W/bank.json" --holdout "$D/prospective.json" --holdout-sha256 aa808942ff23461fa4c58feba2398a117866cfa97d7ba00497dbeda773e0e63a --joint-sha256 8e0c35bb9a39d3fc81d3ab42af78528a3e003d215ade158edde85f52870f458d --output "$W/first"
python "$D/confirm.py" --aliases "$W/baseline.json" --bank "$W/bank.json" --holdout "$D/confirmation.json" --holdout-sha256 3bca9c68baf0347dfa6d5c217c53341013289f95bf4eb2fb42749aa3edbae2b3 --gate-sha256 9622d0c99fe0f6876daae02362a46373cece1ab30e60cc6f1e5af6d5f071c84c --output "$W/confirmation"
git show eabf114:scripts/assistant-runtime/queries/approved-aliases.json > "$W/before.json"
cp "$W/before.json" "$W/after.json"
for batch in 1 2 3 4 5 6 7; do
  python scripts/research-training/exact-transfer-20261120/cycle.py --state "$W/after.json" --bank "$W/bank.json" --limit 8 --seed $((2026112100 + batch)) --report "$W/batch-$batch.json" || exit 1
done
python "$D/probe_transfer.py" "$W/before.json" "$W/after.json" "$W/values.json" --bank "$W/bank.json"
```

Per ripetere il confronto Qwen su Linux x86-64, impostare `AIRI_JOINT_LLM_CACHE` alla cartella locale con il modello Q4_K_M rinominato `model.gguf`, l’archivio `llama-b5963-bin-ubuntu-x64.zip` e il suo contenuto estratto nella stessa cartella. Il driver verifica impronte e librerie, avvia il modello su CPU e parla con il processo tramite un socket locale. [Modello congelato](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9) · [Binario llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b5963)

```sh
python "$D/run.py" --teacher --fresh-only --teacher-sha256 68c79e08321fef878715cd3d67deaa30b24ef0655ed4fd75eaf63ccf306f8b6c --aliases "$W/baseline.json" --bank "$W/bank.json" --holdout "$D/prospective.json" --holdout-sha256 aa808942ff23461fa4c58feba2398a117866cfa97d7ba00497dbeda773e0e63a --output "$W/qwen"
```
