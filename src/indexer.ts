// Indicizzazione del vault via API Obsidian: incrementale per hash, grafo da resolvedLinks,
// persistenza su file nella cartella del plugin (separato da data.json delle impostazioni).
import { App, normalizePath, Plugin, TFile } from "obsidian";
import { chunkMarkdown } from "./chunker";
import { Embedder } from "./embedder";
import { ragLog } from "./logger";
import { HybridStore, StorePayload } from "./store";

interface IndexData {
  model: string;
  hashes: Record<string, string>;
  store: StorePayload;
}

function fnv1a(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16);
}

export class Indexer {
  private adj = new Map<string, Set<string>>();
  private hashes: Record<string, string> = {};
  indexing = false;
  maxCpuPercent = 100; // <100 = duty-cycle (lavora/dormi proporzionale) per limitare la CPU
  embedBatchSize = 0; // >0: embedda i chunk di un file a batch (duty-cycle anche DENTRO il file)
  // Serializzazione mutazioni: ogni write all'indice passa per questa catena (no race su index.json).
  private chain: Promise<void> = Promise.resolve();
  // Persist debounced: coalesce le scritture in burst di edit.
  private persistTimer: number | null = null;
  private dirty = false;

  constructor(
    private app: App,
    private plugin: Plugin,
    private store: HybridStore,
    private embedder: Embedder,
  ) {}

  private dataPath() {
    return normalizePath(`${this.plugin.manifest.dir}/index.json`);
  }

  buildGraph() {
    const rl = this.app.metadataCache.resolvedLinks;
    const adj = new Map<string, Set<string>>();
    const add = (a: string, b: string) => {
      let s = adj.get(a);
      if (!s) adj.set(a, (s = new Set()));
      s.add(b);
    };
    for (const src in rl) for (const dst in rl[src]) {
      add(src, dst);
      add(dst, src);
    }
    this.adj = adj;
  }

  neighborsOf = (files: string[]): Set<string> => {
    const out = new Set<string>();
    for (const f of files) {
      const s = this.adj.get(f);
      if (s) for (const n of s) out.add(n);
    }
    return out;
  };

  private async indexOne(f: TFile): Promise<boolean> {
    const content = await this.app.vault.cachedRead(f);
    const h = fnv1a(content);
    if (this.hashes[f.path] === h) return false;
    const chunks = chunkMarkdown(content, f.name);
    if (!chunks.length) {
      await this.store.removeFile(f.path);
      delete this.hashes[f.path];
      return false;
    }
    const embs = await this.embedder.embedPassages(
      chunks.map((c) => c.content),
      this.embedBatchSize,
      this.maxCpuPercent,
    );
    await this.store.replaceFile(f.path, chunks, embs);
    this.hashes[f.path] = h;
    return true;
  }

  async reindexAll(force: boolean, progress?: (done: number, total: number) => void) {
    if (this.indexing || !this.embedder.ready) return;
    this.indexing = true;
    // annulla una persist debounced pendente: il loop sotto muta lo store fuori dalla catena
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
    this.dirty = false;
    try {
      if (force) this.hashes = {};
      this.buildGraph();
      const files = this.app.vault.getMarkdownFiles();
      ragLog.info(`indicizzazione avviata: ${files.length} file · CPU max ${this.maxCpuPercent}% · batch ${this.embedBatchSize}`);
      let done = 0;
      for (const f of files) {
        try {
          await this.indexOne(f);
        } catch (e) {
          ragLog.error(`indicizzazione fallita: ${f.path}`, e); // un file rotto non blocca il resto
        }
        progress?.(++done, files.length);
        // Il duty-cycle per limitare la CPU è dentro embedPassages (anche per i file grandi);
        // qui cediamo comunque il thread ogni 3 file per tenere viva la UI.
        if (done % 3 === 0) await new Promise((r) => setTimeout(r, 0));
      }
      const present = new Set(files.map((f) => f.path));
      for (const p of this.store.indexedFiles()) {
        if (!present.has(p)) {
          await this.store.removeFile(p);
          delete this.hashes[p];
        }
      }
      await this.persist();
    } finally {
      this.indexing = false;
    }
  }

  async reindexFile(f: TFile) {
    if (f.extension !== "md" || this.indexing || !this.embedder.ready) return;
    await this.enqueue(async () => {
      this.buildGraph();
      if (await this.indexOne(f)) this.schedulePersist();
    });
  }

  async removeFile(path: string) {
    await this.enqueue(async () => {
      await this.store.removeFile(path);
      delete this.hashes[path];
      this.schedulePersist();
    });
  }

  /** Accoda una mutazione: garantisce che non girino due write concorrenti sull'indice. */
  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.chain.then(fn, fn);
    this.chain = run.then(
      () => undefined,
      (err) => {
        ragLog.error("task di indicizzazione", err);
        return undefined;
      },
    );
    return run;
  }

  /** Persist immediato (usato a fine reindexAll e onunload). */
  async persist() {
    this.dirty = false;
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
    await this.enqueue(() => this.writeIndex());
  }

  /** Persist debounced: scrive ~1.5s dopo l'ultimo edit di un burst. */
  private schedulePersist() {
    this.dirty = true;
    if (this.persistTimer !== null) window.clearTimeout(this.persistTimer);
    this.persistTimer = window.setTimeout(() => {
      this.persistTimer = null;
      if (this.dirty) void this.persist();
    }, 1500);
  }

  /** Flush di una persist pendente (da chiamare in onunload). */
  async flushPersist() {
    if (this.dirty || this.persistTimer !== null) await this.persist();
  }

  /** Scrittura atomica: tmp + rename (con fallback a scrittura diretta). */
  private async writeIndex() {
    const data: IndexData = {
      model: this.embedder.model,
      hashes: this.hashes,
      store: await this.store.serialize(),
    };
    const json = JSON.stringify(data);
    const adapter = this.app.vault.adapter;
    const path = this.dataPath();
    const tmp = path + ".tmp";
    try {
      await adapter.write(tmp, json);
      if (await adapter.exists(path)) await adapter.remove(path);
      await adapter.rename(tmp, path);
    } catch {
      await adapter.write(path, json); // fallback non-atomico
      try {
        if (await adapter.exists(tmp)) await adapter.remove(tmp);
      } catch {
        /* ignore */
      }
    }
  }

  /** Carica un indice persistito se compatibile col modello/dim correnti. */
  async tryLoad(expectedModel: string, expectedDim: number): Promise<boolean> {
    try {
      const raw = await this.app.vault.adapter.read(this.dataPath());
      const data = JSON.parse(raw) as IndexData;
      if (data.model !== expectedModel || data.store.dim !== expectedDim) return false;
      await this.store.deserialize(data.store);
      this.hashes = data.hashes ?? {};
      this.buildGraph();
      return true;
    } catch {
      return false;
    }
  }
}
