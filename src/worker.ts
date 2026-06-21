// Web Worker dell'embedding: transformers.js + onnxruntime-web bundlati per il browser
// (vedi esbuild.config.mjs → worker.js). Gira su un thread separato dalla UI di Obsidian.
// Protocollo messaggi: { id, type: 'load'|'embed', payload } → risponde { id, type, ... }.
import { env, pipeline } from "@xenova/transformers";

let extractor: any = null;
let isE5 = false;

const ctx = self as unknown as Worker;
const reply = (id: number, msg: Record<string, unknown>) => ctx.postMessage(Object.assign({ id }, msg));

ctx.onmessage = async (ev: MessageEvent) => {
  const { id, type, payload } = (ev.data || {}) as { id: number; type: string; payload: any };
  try {
    if (type === "load") {
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      try {
        const wasm = (env as any).backends?.onnx?.wasm;
        if (wasm) {
          if (payload.wasmPaths) wasm.wasmPaths = payload.wasmPaths;
          wasm.numThreads = 1;
        }
      } catch {
        /* usa i default */
      }
      const onp = (p: any) => reply(id, { type: "progress", data: p });
      try {
        extractor = await pipeline("feature-extraction", payload.model, { quantized: true, progress_callback: onp });
      } catch (e: any) {
        reply(id, { type: "log", level: "warn", msg: "quantizzata non disponibile, full-precision: " + (e?.message || e) });
        extractor = await pipeline("feature-extraction", payload.model, { quantized: false, progress_callback: onp });
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
