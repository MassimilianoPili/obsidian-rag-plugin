// Servizio locale di embedding: processo Node (di sistema) lanciato dal plugin via child_process.
// Usa @huggingface/transformers che, in ambiente Node, esegue ONNX con onnxruntime-node NATIVO
// (niente wasm, niente SharedArrayBuffer) → embedding off-process, UI di Obsidian sempre fluida.
//
// Risoluzione moduli: questo script vive accanto a node_modules (la cartella estratta del plugin),
// quindi `require('@huggingface/transformers')` e `onnxruntime-node` si risolvono nativamente.
//
// Protocollo (JSON-per-riga): stdin riceve {id,type,payload}; stdout emette {id,type,...}.
// stdout è RISERVATO al protocollo; ogni log va su stderr.

const origLog = console.log;
console.log = (...a) => process.stderr.write(a.map(String).join(" ") + "\n");
console.warn = console.log;
console.info = console.log;

function send(o) {
  process.stdout.write(JSON.stringify(o) + "\n");
}

let pipelineFn = null;
let extractor = null;
let isE5 = false;

async function ensureLib() {
  if (pipelineFn) return;
  const tx = require("@huggingface/transformers");
  pipelineFn = tx.pipeline;
  tx.env.allowLocalModels = false;
}

async function handle(msg) {
  const { id, type, payload } = msg;
  try {
    if (type === "load") {
      await ensureLib();
      const onp = (p) => send({ type: "progress", data: p });
      try {
        extractor = await pipelineFn("feature-extraction", payload.model, { dtype: "q8", progress_callback: onp });
      } catch (e) {
        send({ type: "log", level: "warn", msg: "q8 non disponibile, fp32: " + (e && e.message) });
        extractor = await pipelineFn("feature-extraction", payload.model, { dtype: "fp32", progress_callback: onp });
      }
      isE5 = /e5/i.test(payload.model);
      const probe = await extractor(["probe"], { pooling: "mean", normalize: true });
      send({ id, type: "loaded", dim: probe.tolist()[0].length });
    } else if (type === "embed") {
      const pref = isE5 ? (payload.kind === "query" ? "query: " : "passage: ") : "";
      const items = pref ? payload.texts.map((t) => pref + t) : payload.texts;
      const out = await extractor(items, { pooling: "mean", normalize: true });
      send({ id, type: "result", vectors: out.tolist() });
    } else if (type === "ping") {
      send({ id, type: "pong" });
    }
  } catch (e) {
    send({ id, type: "error", error: String((e && e.stack) || e) });
  }
}

let buf = "";
process.stdin.on("data", (d) => {
  buf += d;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i);
    buf = buf.slice(i + 1);
    if (line.trim()) {
      try {
        handle(JSON.parse(line));
      } catch (e) {
        send({ type: "error", error: "parse: " + String(e) });
      }
    }
  }
});
process.stdin.on("end", () => process.exit(0));

send({ type: "ready" });
origLog; // evita unused
