// Store ibrido in-memory su Orama: BM25 (fulltext) + vettori, fusi con RRF, + graph boost.
// Persistenza via save/load nativi di Orama. Fedele a store.py.
import { create, insertMultiple, removeMultiple, search, save, load } from "@orama/orama";
import type { Chunk } from "./chunker";

export interface SearchResult {
  id: string;
  score: number;
  sourceFile: string;
  heading: string;
  headerPath: string;
  content: string;
}

export interface StorePayload {
  dim: number;
  fileDocs: Record<string, string[]>;
  orama: unknown;
}

const RRF_K = 60;
const POOL = 40;

function schemaFor(dim: number) {
  return {
    content: "string",
    sourceFile: "string",
    heading: "string",
    headerPath: "string",
    embedding: `vector[${dim}]`,
  } as const;
}

export class HybridStore {
  private db: any = null;
  private dim = 0;
  private fileDocs = new Map<string, string[]>(); // file -> doc ids correnti

  async init(dim: number) {
    this.dim = dim;
    this.db = await create({ schema: schemaFor(dim) as any });
    this.fileDocs.clear();
  }

  get ready() {
    return this.db !== null;
  }
  get dimension() {
    return this.dim;
  }

  async replaceFile(file: string, chunks: Chunk[], embeddings: number[][]) {
    if (!this.db) return;
    const old = this.fileDocs.get(file);
    if (old?.length) await removeMultiple(this.db, old);
    const docs = chunks.map((c, i) => ({
      id: `${file}::${c.chunkIndex}`,
      content: c.content,
      sourceFile: file,
      heading: c.heading,
      headerPath: c.headerPath,
      embedding: embeddings[i],
    }));
    if (docs.length) await insertMultiple(this.db, docs);
    this.fileDocs.set(
      file,
      docs.map((d) => d.id),
    );
  }

  async removeFile(file: string) {
    if (!this.db) return;
    const old = this.fileDocs.get(file);
    if (old?.length) await removeMultiple(this.db, old);
    this.fileDocs.delete(file);
  }

  indexedFiles(): Set<string> {
    return new Set(this.fileDocs.keys());
  }

  count(): number {
    let n = 0;
    for (const v of this.fileDocs.values()) n += v.length;
    return n;
  }

  /** Ricerca ibrida: RRF su (BM25 ⊕ KNN) + boost ai chunk in file vicini-di-grafo. */
  async search(
    term: string,
    queryVec: number[] | null,
    k: number,
    neighborsOf: (files: string[]) => Set<string>,
    graphBoost = 1.12,
    mmrLambda = 0.7,
  ): Promise<SearchResult[]> {
    if (!this.db) return [];
    const scores = new Map<string, number>();
    const meta = new Map<string, any>();

    const fuse = (hits: any[]) => {
      hits.forEach((h, pos) => {
        scores.set(h.id, (scores.get(h.id) ?? 0) + 1 / (RRF_K + pos + 1));
        meta.set(h.id, h.document);
      });
    };

    if (term.trim()) {
      const r = await search(this.db, { mode: "fulltext", term, limit: POOL } as any);
      fuse(r.hits);
    }
    if (queryVec) {
      const r = await search(this.db, {
        mode: "vector",
        vector: { value: queryVec, property: "embedding" },
        similarity: 0,
        limit: POOL,
      } as any);
      fuse(r.hits);
    }
    if (scores.size === 0) return [];

    const fileOf = (id: string) => meta.get(id)?.sourceFile as string | undefined;
    const top = [...scores.entries()].sort((a, b) => b[1] - a[1]);
    const topFiles = top.slice(0, 3).map(([id]) => fileOf(id)).filter((f): f is string => !!f);
    const nb = neighborsOf(topFiles);
    for (const [id, s] of scores) {
      const f = fileOf(id);
      if (f && nb.has(f)) scores.set(id, s * graphBoost);
    }

    // Rerank MMR: bilancia rilevanza (score fuso) e diversità (sim coseno tra chunk),
    // così i top-K non sono 3 chunk quasi-identici della stessa nota. Embedding già normalizzati → dot = coseno.
    const dot = (a: number[], b: number[]) => {
      let s = 0;
      const n = Math.min(a.length, b.length);
      for (let i = 0; i < n; i++) s += a[i] * b[i];
      return s;
    };
    const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1]);
    const pool = ranked.slice(0, Math.max(k * 3, 30)).map(([id, s]) => ({
      id,
      rel0: s,
      emb: meta.get(id)?.embedding as number[] | undefined,
    }));
    const maxRel = pool.length ? pool[0].rel0 : 1;
    const lambda = mmrLambda >= 1 ? 1 : mmrLambda; // 1 = disattiva diversità
    const selected: { id: string; rel0: number; emb?: number[] }[] = [];
    while (selected.length < k && pool.length) {
      let bestIdx = -1;
      let bestMmr = -Infinity;
      for (let i = 0; i < pool.length; i++) {
        const c = pool[i];
        const rel = maxRel > 0 ? c.rel0 / maxRel : 0;
        let div = 0;
        if (c.emb && lambda < 1) {
          for (const sdoc of selected) {
            if (sdoc.emb) {
              const sim = dot(c.emb, sdoc.emb);
              if (sim > div) div = sim;
            }
          }
        }
        const mmr = lambda * rel - (1 - lambda) * div;
        if (mmr > bestMmr) {
          bestMmr = mmr;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      selected.push(pool[bestIdx]);
      pool.splice(bestIdx, 1);
    }

    return selected.map(({ id, rel0 }) => {
      const d = meta.get(id);
      return {
        id,
        score: Number(rel0.toFixed(6)),
        sourceFile: d.sourceFile,
        heading: d.heading,
        headerPath: d.headerPath,
        content: d.content,
      };
    });
  }

  async serialize(): Promise<StorePayload> {
    return {
      dim: this.dim,
      fileDocs: Object.fromEntries(this.fileDocs),
      orama: await save(this.db),
    };
  }

  async deserialize(payload: StorePayload) {
    this.dim = payload.dim;
    this.db = await create({ schema: schemaFor(payload.dim) as any });
    await load(this.db, payload.orama as any);
    this.fileDocs = new Map(Object.entries(payload.fileDocs));
  }
}
