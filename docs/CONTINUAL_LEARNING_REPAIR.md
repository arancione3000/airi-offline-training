# Nuovo ciclo selettivo — 18 ottobre 2026

Il ciclo del17ottobre ha accettato il primo consolidamento (revisione1, dialoghi3/16→4/16, tutte le vecchie risposte corrette conservate), poi si è fermato dopo6rifiuti. Finale indipendente0/64 e chat libera ancora incoerente: nessuna qualificazione production/live.

Su nuova richiesta esplicita, il seguito parte dal terminale immutabile `airi-offline-408572-39057187476-1-9`, fingerprint `10962b6cf3c5449f3594168e00c791aa1b11f32d805467f8d5fefce656d9bb56`, step408572, conservando revisione1 e banca eleggibile. Nuovo budget esplicito massimo6ore di ottimizzatore,48decisioni o12rifiuti consecutivi,32job; nessun ciclo aggiuntivo automatico.

Correzione: i vecchi slot di banca2/10 cadevano entrambi su `human-dialogue`, quindi il replay ordinario delle lezioni era sempre escluso. Ora la SPEC dichiara slot3/11 e la validazione vieta collisioni con dialoghi umani e replay nativo.8slotumani/16 e1nativo restano protetti;2/16 usano la banca, se non vuota. Test di regressione sull'intero periodo e su banca vuota/collisioni.

Valutazione ogni64update conteggiati, anziché256; profili LR5e-7/1e-7/2e-6 e replay/KL8/8/4. È un tentativo motivato dalle regressioni osservate, non una garanzia di apprendimento. Interpolazioni e consolidamenti mantengono i gate originali e ogni vecchia risposta di sviluppo corretta. Nessuna risposta dei test diventa dato di training, nessun LLM esterno. Rete invariata.

Ramo congelato `dialogue-selective-replay-20261018`; stesso workflow e unico controller aggiornato a questo ramo. Prima avvio: CI e preflight effimero sul checkpoint reale. Il lettore segue il canale stabile; le console storiche non sono riconfigurate.

---

# Native continual learning repair — 17 October 2026

The completed fifty-hour run and one-hour selective-transfer pilot retained the
same champion. The last rejected pilot candidate reduced English human
development NLL from 3.377257 to 3.246877 but lost old correct answers. Those
regressions remain disqualifying. A completed workflow is not conversational
success.

The repair addresses four concrete problems:

* Reviewed, manually authored dialogue was accidentally excluded from the
  lesson catalog because its provenance began with `assistant-authored`.
  Exact membership and held-out history exclusions remain mandatory.
* Ordinary updates lacked supervised conversational retention on the current
  champion. Each repaired ordinary update includes the new task, bilingual gold
  replay certified correct by that champion, KL to that champion, and native
  replay. A partial teacher-forced token hit cannot certify a memory answer.
* Repeated rejection reduced the old LR to 5e-8 without exploring a useful
  alternative. Three declared profiles now use fresh Adam trials at 2e-6,
  1e-5 and 5e-7 with different replay strengths. The source weights, original
  optimizer checkpoint, model identity, accepted revision and eligible lessons
  are retained. This is an explicit new optimizer policy, not bitwise
  continuation of the old Adam trajectory.
* A job boundary previously discarded a short unfinished learning attempt.
  Candidates now checkpoint across job boundaries with a separate champion.

Every 256 counted updates, the original development gates and per-answer
retention decide adoption. A rejected candidate may undergo two smaller
parameter-step probes (alpha .5 and .25) under those same gates. A successful
merge starts fresh Adam. A failed or interrupted interpolation restores the
original candidate; unsuccessful consolidation restores champion weights and
Adam. Neither training losses nor sealed final answers can select a model.

The authorized experiment starts from
`airi-offline-406524-38897262942-1-3` and stops after at most six optimizer
hours, 24 development decisions or six consecutive rejections. It does not
create another experiment after that. The checkpoint controller is the sole
continuation owner, validates the SHA-committed small run-state asset and both
archive digests, respects cancellation and branch changes, permits at most two
transient recoveries, and disables itself at the terminal state. Shared native
training concurrency is preserved. No ChatGPT invocation or external LLM API
is involved in this runtime loop.

Repeated BPE encodings are cached without changing token IDs or returning
shared mutable lists. An unused pre-encoded dialogue tensor corpus is no
longer constructed. Frozen source and data checks remain in place.

Validation includes independent unit tests, real native Adam/rollback checks,
an ephemeral objective check on the 50,127,552-parameter checkpoint, and a
separate preflight before starting the experiment. Training-task improvement
only proves that the update learns that task; development and final
conversation results must be reported separately.

The underlying native network remains width 96, twelve layers, FF width
14,336, BPE vocabulary 384 and context 1024. No pretrained model is introduced,
and this change does not establish general conversational competence or
guaranteed monotonic progress outside the measured tests. All
`production_qualified` and `live_promoted` flags remain false.
