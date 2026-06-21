// Embedding ONNX in un Web Worker dedicato (thread separato dalla UI).
// Perché un worker NOSTRO e non wasm.proxy di ort-web: il proxy, nel renderer di Obsidian, usa
// una risoluzione path stile-Node (path.dirname) che non esiste → "dirname is not a function".
// In un module-worker creato da noi NON esiste `process`: transformers rileva ambiente browser e
// usa il backend WASM puro (wasmPaths dal CDN), e l'inferenza non blocca il thread della UI.

import { ragLog } from "./logger";

export type ProgressCb = (info: {
  status?: string;
  name?: string;
  file?: string;
  progress?: number;
}) => void;

// Sorgente del worker come stringa: esbuild non la tocca, quindi import() resta dinamico a runtime.
const WORKER_SRC = `
let extractor = null;
let isE5 = false;
const reply = (id, msg) => self.postMessage(Object.assign({ id }, msg));
self.onmessage = async (ev) => {
  const { id, type, payload } = ev.data || {};
  try {
    if (type === 'load') {
      const mod = await import(payload.cdn);
      const lib = (typeof mod.pipeline === 'function') ? mod : mod.default;
      const env = lib.env;
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      try {
        const wasm = env.backends && env.backends.onnx && env.backends.onnx.wasm;
        if (wasm) { wasm.wasmPaths = payload.wasmPaths; wasm.numThreads = 1; }
      } catch (e) {}
      const onp = (p) => reply(id, { type: 'progress', data: p });
      try {
        extractor = await lib.pipeline('feature-extraction', payload.model, { quantized: true, progress_callback: onp });
      } catch (e) {
        reply(id, { type: 'log', level: 'warn', msg: 'quantizzata non disponibile, full-precision: ' + (e && e.message) });
        extractor = await lib.pipeline('feature-extraction', payload.model, { quantized: false, progress_callback: onp });
      }
      isE5 = /e5/i.test(payload.model);
      const probe = await extractor(['probe'], { pooling: 'mean', normalize: true });
      reply(id, { type: 'loaded', dim: probe.tolist()[0].length });
    } else if (type === 'embed') {
      const pref = isE5 ? (payload.kind === 'query' ? 'query: ' : 'passage: ') : '';
      const items = pref ? payload.texts.map((t) => pref + t) : payload.texts;
      const out = await extractor(items, { pooling: 'mean', normalize: true });
      reply(id, { type: 'result', vectors: out.tolist() });
    }
  } catch (e) {
    reply(id, { type: 'error', error: String((e && e.stack) || (e && e.message) || e) });
  }
};
`;

const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
const WASM_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";

interface Pending {
  resolve: (v: any) => void;
  reject: (e: any) => void;
  onProgress?: ProgressCb;
}

export class Embedder {
  model = "";
  dim = 0;
  loading = false;
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();

  get ready() {
    return this.worker !== null && this.dim > 0;
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const blob = new Blob([WORKER_SRC], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    const w = new Worker(url, { type: "module" });
    w.onmessage = (ev: MessageEvent) => {
      const m: any = ev.data || {};
      const p = this.pending.get(m.id);
      if (m.type === "progress") {
        p?.onProgress?.(m.data);
        return;
      }
      if (m.type === "log") {
        if (m.level === "warn") ragLog.warn(`worker: ${m.msg}`);
        else ragLog.info(`worker: ${m.msg}`);
        return;
      }
      if (!p) return;
      this.pending.delete(m.id);
      if (m.type === "loaded") p.resolve({ dim: m.dim });
      else if (m.type === "result") p.resolve({ vectors: m.vectors });
      else if (m.type === "error") p.reject(new Error(m.error));
    };
    w.onerror = (e) => ragLog.error("embedder worker", (e as any)?.message || e);
    this.worker = w;
    return w;
  }

  private request(type: string, payload: any, onProgress?: ProgressCb): Promise<any> {
    const w = this.ensureWorker();
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      w.postMessage({ id, type, payload });
    });
  }

  async load(model: string, onProgress?: ProgressCb): Promise<void> {
    this.loading = true;
    try {
      ragLog.info(`embedder(worker): carico «${model}» (thread separato)`);
      const res = await this.request("load", { cdn: TRANSFORMERS_CDN, wasmPaths: WASM_CDN, model }, onProgress);
      this.model = model;
      this.dim = res.dim;
      ragLog.info(`embedder(worker): pronto «${model}» · dim ${this.dim}`);
    } catch (e) {
      ragLog.error(`embedder(worker): caricamento «${model}» fallito`, e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  // Embedda a batch; con maxCpuPercent<100 applica un duty-cycle (il worker libera comunque la UI).
  async embedPassages(texts: string[], batchSize = 0, maxCpuPercent = 100): Promise<number[][]> {
    const pct = Math.min(100, Math.max(5, maxCpuPercent || 100));
    const bs = batchSize > 0 ? batchSize : texts.length;
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += bs) {
      const t0 = Date.now();
      const r = await this.request("embed", { texts: texts.slice(i, i + bs), kind: "passage" });
      for (const v of r.vectors) out.push(v);
      if (pct < 100) {
        const dt = Date.now() - t0;
        const sleep = Math.min(2000, Math.round(dt * (100 / pct - 1)));
        if (sleep > 0) await new Promise((res) => setTimeout(res, sleep));
      }
    }
    return out;
  }

  async embedQuery(text: string): Promise<number[]> {
    const r = await this.request("embed", { texts: [text], kind: "query" });
    return r.vectors[0];
  }
}
