// Indicizzazione del vault via API Obsidian: incrementale per hash, grafo da resolvedLinks,
// persistenza su file nella cartella del plugin (separato da data.json delle impostazioni).
import { App, normalizePath, Plugin, TFile } from "obsidian";
import { chunkMarkdown } from "./chunker";
import { Embedder } from "./embedder";
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
    const embs = await this.embedder.embedPassages(chunks.map((c) => c.content));
    await this.store.replaceFile(f.path, chunks, embs);
    this.hashes[f.path] = h;
    return true;
  }

  async reindexAll(force: boolean, progress?: (done: number, total: number) => void) {
    if (this.indexing) return;
    this.indexing = true;
    try {
      if (force) this.hashes = {};
      this.buildGraph();
      const files = this.app.vault.getMarkdownFiles();
      let done = 0;
      for (const f of files) {
        await this.indexOne(f);
        progress?.(++done, files.length);
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
    if (f.extension !== "md" || this.indexing) return;
    this.buildGraph();
    if (await this.indexOne(f)) await this.persist();
  }

  async removeFile(path: string) {
    await this.store.removeFile(path);
    delete this.hashes[path];
    await this.persist();
  }

  async persist() {
    const data: IndexData = {
      model: this.embedder.model,
      hashes: this.hashes,
      store: await this.store.serialize(),
    };
    await this.app.vault.adapter.write(this.dataPath(), JSON.stringify(data));
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
