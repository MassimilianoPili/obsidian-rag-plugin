// Embedder resiliente: prova prima il backend WORKER (off-thread), e se il backend ONNX non si
// inizializza nel worker (limite noto di onnxruntime-web caricato a runtime in Obsidian) fa
// FALLBACK automatico al main-thread (provato funzionante). La modalità attiva è in `mode`.

import { ragLog } from "./logger";

export type ProgressCb = (info: {
  status?: string;
  name?: string;
  file?: string;
  progress?: number;
}) => void;

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
  workerUrl = ""; // impostato da main: Blob URL same-origin di worker.js
  wasmBlobPaths: Record<string, string> = {}; // { 'ort-...jsep.wasm': blobUrl } per il backend worker
  mode: "" | "worker" | "main" = "";
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private extractor: any = null; // backend main-thread (fallback)
  private isE5 = false;

  get ready() {
    if (this.dim <= 0) return false;
    return this.mode === "worker" ? this.worker !== null : this.extractor !== null;
  }

  // Il backend WORKER è disabilitato: onnxruntime-web NON registra il suo backend in un worker
  // di Obsidian (6 strategie provate, tutte falliscono su InferenceSession.create). Si usa il
  // main-thread (provato funzionante). Tenuto il codice worker per un eventuale futuro.
  useWorker = true;

  async load(model: string, onProgress?: ProgressCb): Promise<void> {
    this.loading = true;
    this.isE5 = /e5/i.test(model);
    try {
      let ok = false;
      if (this.useWorker) {
        try {
          await this.loadWorker(model, onProgress);
          this.mode = "worker";
          ok = true;
          ragLog.info(`embedder: backend WORKER (off-thread) · «${model}» · dim ${this.dim}`);
        } catch (e) {
          ragLog.warn("embedder: worker non disponibile, fallback MAIN-THREAD", e);
          this.disposeWorker();
        }
      }
      if (!ok) {
        await this.loadMain(model, onProgress);
        this.mode = "main";
        ragLog.info(`embedder: backend MAIN-THREAD · «${model}» · dim ${this.dim}`);
      }
      this.model = model;
    } catch (e) {
      ragLog.error(`embedder: caricamento «${model}» fallito (worker e main-thread)`, e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  // ---------- backend WORKER ----------
  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    if (!this.workerUrl) throw new Error("workerUrl non impostato (worker.js non trovato)");
    const w = new Worker(this.workerUrl, { type: "module" });
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

  private disposeWorker() {
    try {
      this.worker?.terminate();
    } catch {
      /* ignore */
    }
    this.worker = null;
    this.pending.clear();
  }

  private request(type: string, payload: any, onProgress?: ProgressCb): Promise<any> {
    const w = this.ensureWorker();
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      w.postMessage({ id, type, payload });
    });
  }

  private async loadWorker(model: string, onProgress?: ProgressCb): Promise<void> {
    ragLog.info(`embedder(worker): carico «${model}» (thread separato)`);
    const res = await this.request("load", { model, wasmPaths: this.wasmBlobPaths }, onProgress);
    this.dim = res.dim;
  }

  // ---------- backend MAIN-THREAD (fallback) ----------
  private async loadMain(model: string, onProgress?: ProgressCb): Promise<void> {
    const dynImport = new Function("u", "return import(u)") as (u: string) => Promise<any>;
    const proc: any = (globalThis as any).process;
    const undo: Array<() => void> = [];
    if (proc?.release?.name === "node") {
      let done = false;
      try {
        const s = proc.release.name;
        proc.release.name = "browser";
        if (proc.release.name !== "node") {
          undo.push(() => {
            try {
              proc.release.name = s;
            } catch {
              /* ignore */
            }
          });
          done = true;
        }
      } catch {
        /* prova B */
      }
      if (!done) {
        try {
          const sr = proc.release;
          Object.defineProperty(proc, "release", {
            configurable: true,
            writable: true,
            value: Object.assign({}, sr, { name: "browser" }),
          });
          undo.push(() => {
            try {
              Object.defineProperty(proc, "release", { configurable: true, writable: true, value: sr });
            } catch {
              /* ignore */
            }
          });
        } catch (e) {
          ragLog.warn("embedder(main): mascheramento process.release fallito", e);
        }
      }
    }
    let mod: any;
    try {
      mod = await dynImport(TRANSFORMERS_CDN);
    } finally {
      for (const u of undo) u();
    }
    const lib: any = typeof mod?.pipeline === "function" ? mod : mod?.default;
    if (!lib || typeof lib.pipeline !== "function") throw new Error("transformers: pipeline() non disponibile (main)");
    const e: any = lib.env ?? mod?.env;
    if (e) {
      e.allowLocalModels = false;
      e.useBrowserCache = true;
      try {
        const w = e.backends?.onnx?.wasm;
        if (w) {
          w.wasmPaths = WASM_CDN;
          w.numThreads = 1;
        }
      } catch {
        /* default */
      }
    }
    const build = (quantized: boolean) => {
      ragLog.info(`embedder(main): carico «${model}» (quantized=${quantized})`);
      return lib.pipeline("feature-extraction", model, { quantized, progress_callback: onProgress });
    };
    try {
      this.extractor = await build(true);
    } catch (err) {
      ragLog.warn(`embedder(main): quantizzata non disponibile, full-precision`, err);
      this.extractor = await build(false);
    }
    this.dim = (await this.embedRawMain(["probe"]))[0].length;
  }

  private async embedRawMain(texts: string[]): Promise<number[][]> {
    const out = await this.extractor(texts, { pooling: "mean", normalize: true });
    return out.tolist() as number[][];
  }

  // ---------- API comune (instrada su worker o main) ----------
  async embedPassages(texts: string[], batchSize = 0, maxCpuPercent = 100): Promise<number[][]> {
    const pct = Math.min(100, Math.max(5, maxCpuPercent || 100));
    const bs = batchSize > 0 ? batchSize : texts.length;
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += bs) {
      const slice = texts.slice(i, i + bs);
      const t0 = Date.now();
      let r: number[][];
      if (this.mode === "worker") {
        r = (await this.request("embed", { texts: slice, kind: "passage" })).vectors;
      } else {
        const items = this.isE5 ? slice.map((x) => `passage: ${x}`) : slice;
        r = await this.embedRawMain(items);
      }
      for (const v of r) out.push(v);
      if (pct < 100) {
        const dt = Date.now() - t0;
        const sleep = Math.min(2000, Math.round(dt * (100 / pct - 1)));
        if (sleep > 0) await new Promise((res) => setTimeout(res, sleep));
      } else if (this.mode === "main") {
        await new Promise((res) => setTimeout(res, 0)); // cede comunque il thread in modalità main
      }
    }
    return out;
  }

  async embedQuery(text: string): Promise<number[]> {
    if (this.mode === "worker") {
      return (await this.request("embed", { texts: [text], kind: "query" })).vectors[0];
    }
    const item = this.isE5 ? `query: ${text}` : text;
    return (await this.embedRawMain([item]))[0];
  }
}
