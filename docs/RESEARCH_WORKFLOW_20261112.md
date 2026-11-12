# Ricerca prima dell'allenamento — 12 novembre 2026

## Procedura e criteri fissati prima della prova

1. Verificare checkpoint, codice effettivamente eseguito e risultati, senza confondere supervisore attivo con aggiornamenti del modello.
2. Riprodurre ogni errore operativo e individuare il punto esatto in cui il recupero si interrompe.
3. Separare evidenze, ipotesi e limiti della misura. Studiare fonti primarie senza applicare automaticamente risultati di grandi modelli a questa rete.
4. Riprodurre il modello approvato revisione4 (23/92, contesto0/32); mantenere i897 certificati TRAIN e i dati registrati. Non usare etichette di valutazione per l'ottimizzatore.
5. Confronto predefinito A/B di256 unità dal medesimo modello. A aggiorna attenzione finale/norme/testa; B aggiunge soltanto la proiezione finale FFN già presente. Stessi esempi, valori TRAIN, ricordi, sentinelle, insegnanti, iperparametri e ordine. La maggiore capacità ha costi diversi: non è un confronto a FLOP uguali.
6. Misurare risposte complete, nuove corrette/perse, contesto, NLL, conservazione di tutti i certificati, tentativi/aggiornamenti confermati e costo. Verificare export nativo e immutabilità del modello sorgente. Nessuna scelta intermedia o uso del test finale per ottimizzare.
7. Attivare solo correzioni giustificate: riparazione del recupero se la riproduzione passa; modifica del modello solo se supera i criteri originari e conserva le risposte precedenti. Un miglioramento della sola NLL non dimostra migliori capacità di chat.
8. Riprendere il servizio senza scadenza dopo le verifiche, osservando un vero aggiornamento e il successivo passaggio automatico. Restano limiti per epoca, salvataggi, backoff e arresto su errori d'integrità; nessuna chiamata periodica all'assistente.

## Evidenze già confermate

- Il job42277407762 è fallito il6 novembre per HTTP500 nel download del checkpoint. I log del supervisore mostrano il rifiuto dei codici ANSI da parte di `gh api`; il recupero non raggiungeva il classificatore degli errori. Il watcher mascherava il problema ripetendo e rinnovandosi. È una causa riprodotta, non un'ipotesi.
- Nella strategia precedente soltanto74.304/50.164.416 parametri erano addestrabili (0,148%); undici blocchi e tutti gli FFN erano congelati. È una restrizione reale; che sia la causa principale dello0/32 va verificato.
- Il corpus umano utilizzato contiene866 conversazioni,18 italiane. Le precedenti prove di sola maggiore varietà/dose hanno abbassato la NLL senza acquisire nuove risposte. Il benchmark di92 compiti non certifica una chat generale.
- Le sonde trovano informazione sul primo carattere nella rappresentazione; non provano che il modello sappia generare e copiare valori completi.

## Fonti primarie e limiti

[GitHub CLI API](https://cli.github.com/manual/gh_api) documenta l'opzione per catturare log con escape; l'output deve essere ripulito prima della diagnosi. [API GitHub dei log](https://docs.github.com/en/rest/actions/workflow-jobs#download-job-logs-for-a-workflow-run) descrive il redirect dei log.

[Geva et al., FFN come memorie](https://aclanthology.org/2021.emnlp-main.446/) motiva una prova sulla proiezione FFN: non garantisce il risultato. [Chinchilla](https://arxiv.org/abs/2203.15556) lega dati e capacità ai costi di apprendimento; non fornisce una soglia universale per far chattare questo modello. [LoRA](https://arxiv.org/abs/2106.09685) riguarda adattamento di basi già preaddestrate, non dimostra che qualunque base diventi conversazionale modificando un solo strato.

## Stato

Ricerca e confronto completati: entrambi i candidati respinti, nessuna modifica architetturale attivata. Patch operativa pubblicata con 17 test passati e CI remota riuscita. Supervisore riabilitato dopo la decisione; il job precedentemente bloccato è stato effettivamente rilanciato. [Conclusioni e limiti](RESEARCH_CONCLUSIONS_20261112.md); [archivio riproducibile](../scripts/research-training/ffn-output-20261112/README.md). L'esito del checkpoint e del successivo passaggio automatico va osservato separatamente.
