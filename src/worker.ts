// Worker CLASSICO dell'embedding. NON bundliamo transformers/onnxruntime (esbuild rompe il
// loader dei .wasm di ort → "no available backend"). Carichiamo invece la build UMD ufficiale
// via importScripts dal CDN: è la build degli autori, col loader wasm integro. Gira off-thread.
declare function importScripts(...urls: string[]): void;

const DIST = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";
const WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";

const ctx: any = self;
let T: any = null;
let extractor: any = null;
let isE5 = false;

const reply = (id: number, msg: Record<string, unknown>) => ctx.postMessage(Object.assign({ id }, msg));

function findLib(): any {
  if (ctx.transformers?.pipeline) return ctx.transformers;
  for (const k of Object.keys(ctx)) {
    const v = ctx[k];
    if (v && typeof v.pipeline === "function" && v.env) return v;
  }
  return null;
}

ctx.onmessage = async (ev: MessageEvent) => {
  const { id, type, payload } = (ev.data || {}) as { id: number; type: string; payload: any };
  try {
    if (type === "load") {
      if (!T) {
        // importScripts cross-origin è bloccato dalla CSP di Obsidian; fetch cross-origin no.
        // Quindi: fetch del testo UMD → Blob same-origin → importScripts(blob) (consentito).
        const res = await fetch(DIST);
        if (!res.ok) throw new Error("fetch transformers UMD: HTTP " + res.status);
        const code = await res.text();
        const blobUrl = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
        importScripts(blobUrl);
        T = findLib();
        if (!T) throw new Error("transformers UMD non trovato dopo importScripts(blob)");
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
