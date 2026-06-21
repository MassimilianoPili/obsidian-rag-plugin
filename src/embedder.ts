// Client dell'embedding: parla con worker.js (transformers+ort bundlati per browser) su un
// thread separato. workerUrl è l'app:// del file worker.js nella cartella del plugin (impostato
// da main via getResourcePath) → stesso origine, niente CORS, backend WASM registrato correttamente.

import { ragLog } from "./logger";

export type ProgressCb = (info: {
  status?: string;
  name?: string;
  file?: string;
  progress?: number;
}) => void;

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
  workerUrl = ""; // impostato da main: getResourcePath(<plugin>/worker.js)
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();

  get ready() {
    return this.worker !== null && this.dim > 0;
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    if (!this.workerUrl) throw new Error("workerUrl non impostato (worker.js non trovato)");
    const w = new Worker(this.workerUrl); // worker CLASSICO (usa importScripts)
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
    w.onerror = (e) => ragLog.error("embedder worker", (e as any)?.message || String(e));
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
      const res = await this.request("load", { model, wasmPaths: WASM_CDN }, onProgress);
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

  // Batch + duty-cycle opzionale; col worker la UI resta libera comunque.
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
