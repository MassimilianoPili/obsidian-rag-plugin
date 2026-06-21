// Module-worker dell'embedding (thread separato). Carica il bundle ESM ufficiale di transformers
// (dist/transformers.min.js: self-contained, ort incluso, loader wasm integro) così:
//   fetch del testo (cross-origin consentito) → Blob same-origin → import(blobUrl) (ESM).
// Niente bundling di ort (rompeva il loader) e niente importScripts (la dist è ESM).
const DIST = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";
const WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";

const ctx: any = self;
let T: any = null;
let extractor: any = null;
let isE5 = false;

const reply = (id: number, msg: Record<string, unknown>) => ctx.postMessage(Object.assign({ id }, msg));

ctx.onmessage = async (ev: MessageEvent) => {
  const { id, type, payload } = (ev.data || {}) as { id: number; type: string; payload: any };
  try {
    if (type === "load") {
      if (!T) {
        const res = await fetch(DIST);
        if (!res.ok) throw new Error("fetch transformers dist: HTTP " + res.status);
        const code = await res.text();
        const blobUrl = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
        const mod: any = await import(blobUrl);
        T = mod && typeof mod.pipeline === "function" ? mod : mod?.default;
        if (!T || typeof T.pipeline !== "function") throw new Error("transformers dist: pipeline() mancante");
      }
      T.env.allowLocalModels = false;
      T.env.useBrowserCache = true;
      try {
        const w = T.env.backends?.onnx?.wasm;
        if (w) {
          w.wasmPaths = WASM;
          w.numThreads = 1;
        }
      } catch {
        /* default */
      }
      const onp = (p: any) => reply(id, { type: "progress", data: p });
      try {
        extractor = await T.pipeline("feature-extraction", payload.model, { quantized: true, progress_callback: onp });
      } catch (e: any) {
        reply(id, { type: "log", level: "warn", msg: "quantizzata non disponibile, full-precision: " + (e?.message || e) });
        extractor = await T.pipeline("feature-extraction", payload.model, { quantized: false, progress_callback: onp });
      }
      isE5 = /e5/i.test(payload.model);
      const probe = await extractor(["probe"], { pooling: "mean", normalize: true });
      reply(id, { type: "loaded", dim: probe.tolist()[0].length });
    } else if (type === "embed") {
      const pref = isE5 ? (payload.kind === "query" ? "query: " : "passage: ") : "";
      const items = pref ? payload.texts.map((t: string) => pref + t) : payload.texts;
      const out = await extractor(items, { pooling: "mean", normalize: true });
      reply(id, { type: "result", vectors: out.tolist() });
    }
  } catch (e: any) {
    reply(id, { type: "error", error: String(e?.stack || e?.message || e) });
  }
};
