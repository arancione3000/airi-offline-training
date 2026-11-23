const byId = (id) => document.getElementById(id);
const reports = byId("reports");
const status = byId("status");
const resultPanel = byId("result");
const form = byId("clarification");
const choices = byId("choices");
const activate = byId("activate");
const answerArea = byId("answer-area");
const nextExample = byId("next-example");
const revisionLabel = byId("revision");
const exampleNew = byId("example-new");
const exampleKnown = byId("example-known");
const exampleConflict = byId("example-conflict");
const analyze = byId("analyze");
const reset = byId("reset");
const formatNumber = new Intl.NumberFormat("it-IT");

let examples = null;
let revision = 0;
let proposal = null;
let busy = false;
let currentExample = null;
const worker = new Worker(new URL("./runtime-worker.js", import.meta.url), { type: "module" });

function setBusy(value) {
  busy = value;
  for (const control of [reports, exampleNew, exampleKnown, exampleConflict, reset, nextExample]) {
    control.disabled = value;
  }
  exampleNew.disabled = value || revision >= examples.updates.length;
  form.querySelector("fieldset").disabled = value;
  analyze.disabled = value || !reports.value.trim();
  activate.disabled = value || !form.querySelector("input:checked");
}

function setRevision(value) {
  revision = value;
  revisionLabel.textContent = `Regole attive · ${value + 1}`;
  revisionLabel.hidden = false;
}

function showStatus(message, tone = "") {
  status.textContent = message;
  resultPanel.dataset.tone = tone;
  form.hidden = true;
  answerArea.hidden = true;
}

function loadExample(kind) {
  if (busy || !examples) return;
  let data;
  if (kind === "new") {
    const update = examples.updates[revision];
    if (!update) return;
    data = [update.arrival];
  } else if (kind === "known") {
    data = examples.known;
  } else {
    data = revision ? examples.updates[revision - 1].conflict : examples.conflict;
  }
  reports.value = data.join("\n\n");
  currentExample = kind;
  highlightExample();
  proposal = null;
  showStatus("Premi Analizza per valutare il resoconto.");
  setBusy(false);
}

function highlightExample() {
  for (const [name, control] of [["new", exampleNew], ["known", exampleKnown], ["conflict", exampleConflict]]) {
    control.classList.toggle("is-current", currentExample === name);
  }
}

function readReports() {
  return reports.value.split(/\r?\n[ \t]*\r?\n/).map((entry) => entry.trim()).filter(Boolean);
}

function send(action, fields = {}) {
  if (busy || !examples) return;
  setBusy(true);
  status.textContent = action === "activate" ? "Verifica della nuova regola…" : "Elaborazione…";
  worker.postMessage({ action, ...fields });
}

function showClarification(result) {
  proposal = result.proposal;
  const hasChecks = examples.updates.some((update) => update.arrival === proposal.report);
  showStatus(hasChecks ? "Nuovo formato. Serve una conferma." : "Nuovo formato da chiarire.");
  byId("question").textContent = result.question;
  choices.replaceChildren();
  for (const name of proposal.labels) {
    const label = document.createElement("label");
    label.className = "choice";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "shipped";
    input.value = name;
    input.disabled = !hasChecks;
    label.append(input, document.createTextNode(name));
    choices.append(label);
  }
  byId("checks-hint").textContent = hasChecks
    ? "La demo verifica la scelta su casi separati prima di attivare la regola."
    : "Per approvare un resoconto personalizzato servono verifiche indipendenti. Prova un formato di esempio.";
  activate.hidden = !hasChecks;
  form.hidden = false;
}

const reviewMessage = {
  no_reports: "Inserisci almeno un resoconto.",
  malformed_report: "Da rivedere. Verifica che ogni resoconto contenga due campi con quantità intere.",
  too_many_reports: "Da rivedere. Analizza al massimo 16 resoconti alla volta.",
  unrecognized_format_among_multiple: "Da rivedere. Almeno un resoconto usa un formato sconosciuto.",
  conflicting_reports: "Da rivedere. Le fonti riportano quantità incompatibili.",
};

function showAssessment(result, activation) {
  setRevision(result.revision);
  if (result.status === "clarification") {
    showClarification(result);
  } else if (result.status === "review") {
    proposal = null;
    showStatus(reviewMessage[result.reason] ?? "Da rivedere.", "review");
  } else {
    proposal = null;
    showStatus(activation ? "Regola approvata" : "Saldo verificato", "answered");
    byId("answer").textContent = `${formatNumber.format(result.answer)} unità`;
    const source = result.sources[0];
    byId("operands").textContent =
      `${formatNumber.format(source.shipped)} spedite − ${formatNumber.format(source.cancelled)} stornate`;
    const verification = byId("verification");
    verification.hidden = !activation;
    verification.textContent = activation
      ? `${activation.qualification_cases} verifiche del formato e ${activation.historical_cases} verifiche precedenti superate.`
      : "";
    nextExample.hidden = !activation || revision >= examples.updates.length;
    answerArea.hidden = false;
  }
}

function handleError() {
  showStatus("Il motore non è disponibile. Controlla la connessione e ricarica la pagina.", "review");
  if (examples) setBusy(false);
}

function revealOnSmallScreen(element) {
  if (matchMedia("(max-width: 47rem)").matches) {
    const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
    element.scrollIntoView({ block: "start", behavior: reducedMotion ? "auto" : "smooth" });
  }
}

worker.addEventListener("error", handleError);
worker.addEventListener("message", ({ data }) => {
  if (data.type === "error") {
    handleError();
    return;
  }
  if (data.type === "ready") {
    examples = data.examples;
    setRevision(0);
    loadExample("new");
    return;
  }
  const { result, action } = data;
  setBusy(false);
  if (action === "reset") {
    setRevision(0);
    loadExample("new");
  } else if (action === "activate" && result.status !== "activated") {
    showClarification({ proposal, question: byId("question").textContent });
    status.textContent = result.reason === "qualification_mismatch"
      ? "La scelta non supera le verifiche. Prova l’altro campo."
      : "La regola non è stata approvata. Analizza nuovamente il resoconto.";
    resultPanel.dataset.tone = "review";
  } else if (action === "activate") {
    showAssessment(result.assessment, result);
  } else {
    showAssessment(result);
  }
  setBusy(false);
  if (action !== "reset") revealOnSmallScreen(resultPanel);
});

for (const [control, kind] of [[exampleNew, "new"], [exampleKnown, "known"], [exampleConflict, "conflict"]]) {
  control.addEventListener("click", () => loadExample(kind));
}
nextExample.addEventListener("click", () => {
  loadExample("new");
  revealOnSmallScreen(reports);
});
analyze.addEventListener("click", () => send("assess", { reports: readReports() }));
reset.addEventListener("click", () => send("reset"));
reports.addEventListener("input", () => {
  currentExample = null;
  highlightExample();
  proposal = null;
  showStatus("Premi Analizza per valutare il resoconto.");
  setBusy(false);
});
form.addEventListener("change", () => {
  if (!busy) setBusy(false);
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const field = form.querySelector("input:checked");
  if (proposal && field) send("activate", { proposal, label: field.value });
});

byId("share").addEventListener("click", async () => {
  const shareStatus = byId("share-status");
  try {
    await navigator.clipboard.writeText(location.href);
    shareStatus.textContent = "Link copiato.";
  } catch {
    shareStatus.textContent = "Copia l’indirizzo dalla barra del browser.";
  }
});
