const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
const CORPUS_SHA256 = "9bd048c1db127b8939468c2544f1d8e7665f579ff9a227a4e46175d513c9311b";

async function fetchLocal(file) {
  const response = await fetch(new URL(file, import.meta.url));
  if (!response.ok) throw new Error(`Cannot load ${file}: ${response.status}`);
  return response;
}

async function initialize() {
  try {
    const [{ loadPyodide }, source, corpusResponse] = await Promise.all([
      import(PYODIDE + "pyodide.mjs"),
      fetchLocal("./repair.py"),
      fetchLocal("./CASES.json"),
    ]);
    const corpusBytes = await corpusResponse.arrayBuffer();
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", corpusBytes));
    const hex = Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (hex !== CORPUS_SHA256) throw new Error("Unexpected case corpus");
    const corpusText = new TextDecoder().decode(corpusBytes);
    const corpus = JSON.parse(corpusText);
    const runtime = await loadPyodide({ indexURL: PYODIDE });
    await runtime.loadPackage("sqlite3");
    runtime.FS.writeFile("repair.py", await source.text());
    runtime.FS.writeFile("cases.json", corpusText);
    runtime.runPython(`
import json
from pathlib import Path
from repair import FormatRepair

state_path = Path("demo-formats.sqlite")
corpus = json.loads(Path("cases.json").read_text(encoding="utf-8"))
engine = FormatRepair(state_path)

def select_checks(rows):
    return [{"reports": row["reports"], "expected": row["expected"]} for row in rows]

def process(payload):
    global engine
    data = json.loads(payload)
    action = data["action"]
    if action == "assess":
        response = engine.assess(data["reports"])
    elif action == "reset":
        state_path.unlink(missing_ok=True)
        engine = FormatRepair(state_path)
        response = {"status": "reset", "revision": 0}
    elif action == "activate":
        proposal = data["proposal"]
        update = next((item for item in corpus["updates"]
                       if item["arrival"]["reports"] == [proposal["report"]]), None)
        if update is None:
            response = {"status": "rejected", "reason": "no_trusted_demo_checks"}
        else:
            protected = select_checks(corpus["protected"]) if proposal["revision"] == 0 else None
            response = engine.activate(proposal, data["label"], select_checks(update["qualification"]),
                                       protected=protected)
            if response["status"] == "activated":
                response["assessment"] = engine.assess([proposal["report"]])
    else:
        raise ValueError("Unknown demo operation")
    return json.dumps(response, ensure_ascii=False)
`);
    const process = runtime.globals.get("process");
    self.addEventListener("message", ({ data }) => {
      try {
        const result = JSON.parse(process(JSON.stringify(data)));
        self.postMessage({ type: "result", action: data.action, result });
      } catch (error) {
        console.error(error);
        self.postMessage({ type: "error" });
      }
    });
    self.postMessage({
      type: "ready",
      examples: {
        updates: corpus.updates.map((update) => ({
          arrival: update.arrival.reports[0],
          conflict: update.qualification.find((item) => item.kind === "equal_net_conflict").reports,
        })),
        known: corpus.protected.find((item) => item.kind === "clean").reports,
        conflict: corpus.protected.find((item) => item.kind === "equal_net_conflict").reports,
      },
    });
  } catch (error) {
    console.error(error);
    self.postMessage({ type: "error" });
  }
}

initialize();
