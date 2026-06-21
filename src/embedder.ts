// Embedder con 3 backend, in ordine di preferenza:
//  1) SERVICE  — processo Node esterno (child_process) con @huggingface/transformers + onnxruntime-node
//                NATIVO → embedding off-process, UI fluida. È la via che funziona davvero off-thread.
//  2) WORKER   — Web Worker (disabilitato: ort non registra il backend nei worker di Obsidian).
//  3) MAIN     — sul thread principale (sempre funzionante, ma blocca la UI durante l'inferenza).
// Fallback automatico SERVICE → MAIN se il processo Node non parte (es. `node` non nel PATH).

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
  mode: "" | "service" | "worker" | "main" = "";

  // SERVICE config (impostati da main)
  nodePath = "node";
  serviceScript = "";

  // WORKER config
  workerUrl = "";
  wasmBlobPaths: Record<string, string> = {};
  useWorker = false;

  private isE5 = false;
  private seq = 0;
  private pending = new Map<number, Pending>();

  // service process
  private proc: any = null;
  private sbuf = "";
  private readyPromise: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private svcProgress: ProgressCb | undefined;

  // worker
  private worker: Worker | null = null;

  // main-thread
  private extractor: any = null;

  get ready() {
    if (this.dim <= 0) return false;
    if (this.mode === "service") return this.proc !== null;
    if (this.mode === "worker") return this.worker !== null;
    return this.extractor !== null;
  }

  async load(model: string, onProgress?: ProgressCb): Promise<void> {
    this.loading = true;
    this.isE5 = /e5/i.test(model);
    try {
      let ok = false;

      // 1) SERVICE
      if (this.serviceScript) {
        try {
          await this.loadService(model, onProgress);
          this.mode = "service";
          ok = true;
          ragLog.info(`embedder: backend SERVICE (off-process, ort nativo) · «${model}» · dim ${this.dim}`);
        } catch (e) {
          ragLog.warn("embedder: servizio non disponibile, fallback MAIN-THREAD", e);
          this.disposeService();
        }
      }

      // 2) WORKER (disabilitato di default)
      if (!ok && this.useWorker) {
        try {
          await this.loadWorker(model, onProgress);
          this.mode = "worker";
          ok = true;
          ragLog.info(`embedder: backend WORKER · «${model}» · dim ${this.dim}`);
        } catch (e) {
          ragLog.warn("embedder: worker non disponibile, fallback MAIN-THREAD", e);
          this.disposeWorker();
        }
      }

      // 3) MAIN-THREAD
      if (!ok) {
        await this.loadMain(model, onProgress);
        this.mode = "main";
        ragLog.info(`embedder: backend MAIN-THREAD · «${model}» · dim ${this.dim}`);
      }

      this.model = model;
    } catch (e) {
      ragLog.error(`embedder: caricamento «${model}» fallito`, e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  // ---------- backend SERVICE (processo Node esterno) ----------
  private async ensureService(): Promise<void> {
    if (this.proc && this.readyPromise) return this.readyPromise;
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const cp = require("child_process");
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const path = require("path");
    if (!this.serviceScript) throw new Error("serviceScript non impostato");
    const cwd = path.dirname(this.serviceScript);
    this.proc = cp.spawn(this.nodePath, [this.serviceScript], {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      // Rifiuta la ready-promise E tutte le richieste pendenti: così un crash (anche a metà
      // operazione, o `node` assente → ENOENT) NON lascia appese le promise → scatta il fallback.
      const failAll = (err: Error) => {
        reject(err);
        for (const p of this.pending.values()) p.reject(err);
        this.pending.clear();
      };
      this.proc.on("error", (e: any) =>
        failAll(new Error(`spawn «${this.nodePath}» fallito (${e?.message || e}) — Node nel PATH?`)),
      );
      this.proc.on("exit", (c: any) => {
        this.proc = null;
        this.readyPromise = null;
        failAll(new Error(`servizio uscito (code ${c})`));
      });
      // se non è pronto entro 10s, errore (→ fallback)
      setTimeout(() => reject(new Error("servizio: timeout avvio (10s)")), 10000);
    });
    this.proc.stdout.on("data", (d: any) => this.onServiceData(String(d)));
    this.proc.stderr.on("data", (d: any) => {
      const s = String(d).trim();
      if (s) ragLog.info(`service: ${s}`);
    });
    return this.readyPromise;
  }

  private onServiceData(s: string) {
    this.sbuf += s;
    let i;
    while ((i = this.sbuf.indexOf("\n")) >= 0) {
      const line = this.sbuf.slice(0, i);
      this.sbuf = this.sbuf.slice(i + 1);
      if (!line.trim()) continue;
      let m: any;
      try {
        m = JSON.parse(line);
      } catch {
        continue;
      }
      if (m.type === "ready") {
        this.readyResolve?.();
        continue;
      }
      if (m.type === "progress") {
        this.svcProgress?.(m.data);
        continue;
      }
      if (m.type === "log") {
        if (m.level === "warn") ragLog.warn(`service: ${m.msg}`);
        else ragLog.info(`service: ${m.msg}`);
        continue;
      }
      const p = this.pending.get(m.id);
      if (!p) continue;
      this.pending.delete(m.id);
      if (m.type === "loaded") p.resolve({ dim: m.dim });
      else if (m.type === "result") p.resolve({ vectors: m.vectors });
      else if (m.type === "pong") p.resolve({});
      else if (m.type === "error") p.reject(new Error(m.error));
    }
  }

  private async requestService(type: string, payload: any, onProgress?: ProgressCb): Promise<any> {
    await this.ensureService();
    if (onProgress) this.svcProgress = onProgress;
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(JSON.stringify({ id, type, payload }) + "\n");
    });
  }

  private async loadService(model: string, onProgress?: ProgressCb): Promise<void> {
    ragLog.info(`embedder(service): carico «${model}» (processo Node esterno)`);
    const res = await this.requestService("load", { model }, onProgress);
    this.dim = res.dim;
  }

  private disposeService() {
    try {
      this.proc?.kill();
    } catch {
      /* ignore */
    }
    this.proc = null;
    this.readyPromise = null;
    this.pending.clear();
  }

  // ---------- backend WORKER ----------
  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    if (!this.workerUrl) throw new Error("workerUrl non impostato");
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
  }
  private requestWorker(type: string, payload: any, onProgress?: ProgressCb): Promise<any> {
    const w = this.ensureWorker();
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      w.postMessage({ id, type, payload });
    });
  }
  private async loadWorker(model: string, onProgress?: ProgressCb): Promise<void> {
    const res = await this.requestWorker("load", { model, wasmPaths: this.wasmBlobPaths }, onProgress);
    this.dim = res.dim;
  }

  // ---------- backend MAIN-THREAD ----------
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

  // ---------- API comune ----------
  async embedPassages(texts: string[], batchSize = 0, maxCpuPercent = 100): Promise<number[][]> {
    const pct = Math.min(100, Math.max(5, maxCpuPercent || 100));
    const bs = batchSize > 0 ? batchSize : texts.length;
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += bs) {
      const slice = texts.slice(i, i + bs);
      const t0 = Date.now();
      let r: number[][];
      if (this.mode === "service") {
        r = (await this.requestService("embed", { texts: slice, kind: "passage" })).vectors;
      } else if (this.mode === "worker") {
        r = (await this.requestWorker("embed", { texts: slice, kind: "passage" })).vectors;
      } else {
        const items = this.isE5 ? slice.map((x) => `passage: ${x}`) : slice;
        r = await this.embedRawMain(items);
      }
      for (const v of r) out.push(v);
      // Il duty-cycle CPU serve solo in modalità MAIN (off-process/off-thread non blocca la UI).
      if (this.mode === "main") {
        if (pct < 100) {
          const dt = Date.now() - t0;
          const sleep = Math.min(2000, Math.round(dt * (100 / pct - 1)));
          if (sleep > 0) await new Promise((res) => setTimeout(res, sleep));
        } else {
          await new Promise((res) => setTimeout(res, 0));
        }
      }
    }
    return out;
  }

  async embedQuery(text: string): Promise<number[]> {
    if (this.mode === "service") return (await this.requestService("embed", { texts: [text], kind: "query" })).vectors[0];
    if (this.mode === "worker") return (await this.requestWorker("embed", { texts: [text], kind: "query" })).vectors[0];
    const item = this.isE5 ? `query: ${text}` : text;
    return (await this.embedRawMain([item]))[0];
  }

  dispose() {
    this.disposeService();
    this.disposeWorker();
  }
}
