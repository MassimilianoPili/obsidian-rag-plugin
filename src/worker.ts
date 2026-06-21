// Module-worker dell'embedding con @huggingface/transformers v3 BUNDLATO (esbuild platform browser).
// Il backend WASM di onnxruntime si registra perché i .wasm sono forniti come Blob URL same-origin
// via wasmPaths a OGGETTO (supportato da ort>=1.16, incluso in v3). Niente CDN, niente import.meta.
import { env, pipeline } from "@huggingface/transformers";

const ctx: any = self;
let extractor: any = null;
let isE5 = false;

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
          // wasmPaths a oggetto: { 'ort-wasm-...jsep.wasm': blobUrl } passato dal main (same-origin).
          if (payload.wasmPaths) wasm.wasmPaths = payload.wasmPaths;
          wasm.numThreads = 1;
          wasm.proxy = false;
        }
      } catch {
        /* default */
      }
      const onp = (p: any) => reply(id, { type: "progress", data: p });
      // v3 usa `dtype` (q8 = quantizzato, fp32 = full-precision) al posto di `quantized`.
      try {
        extractor = await pipeline("feature-extraction", payload.model, { dtype: "q8", progress_callback: onp } as any);
      } catch (e: any) {
        reply(id, { type: "log", level: "warn", msg: "q8 non disponibile, fp32: " + (e?.message || e) });
        extractor = await pipeline("feature-extraction", payload.model, { dtype: "fp32", progress_callback: onp } as any);
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
