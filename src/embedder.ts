// Embedding ONNX in-process via transformers.js (@xenova/transformers).
// Import dinamico: il modulo pesante si inizializza solo al primo load(), non all'avvio del plugin.
// I modelli e5 richiedono i prefissi query:/passage!, gestiti qui.

import { ragLog } from "./logger";

export type ProgressCb = (info: { status?: string; name?: string; file?: string; progress?: number }) => void;

export class Embedder {
  model = "";
  dim = 0;
  loading = false; // true durante il download/init del modello (per evitare load concorrenti)
  private extractor: any = null;
  private isE5 = false;

  get ready() {
    return this.extractor !== null;
  }

  async load(model: string, onProgress?: ProgressCb): Promise<void> {
    this.loading = true;
    try {
      const t = await import("@xenova/transformers");
      t.env.allowLocalModels = false; // scarica da HF CDN, poi cache browser
      t.env.useBrowserCache = true;
      const build = (quantized: boolean) => {
        ragLog.info(`embedder: carico «${model}» (quantized=${quantized})`);
        return t.pipeline("feature-extraction", model, { quantized, progress_callback: onProgress });
      };
      // Molti modelli (es. multilingual-e5-base) NON hanno la variante quantizzata su HF:
      // si prova quantized, e in caso di fallimento si ricade su full-precision.
      try {
        this.extractor = await build(true);
      } catch (e) {
        ragLog.warn(`embedder: variante quantizzata non disponibile per «${model}», riprovo full-precision`, e);
        this.extractor = await build(false);
      }
      this.model = model;
      this.isE5 = /e5/i.test(model);
      this.dim = (await this.embedRaw(["probe"]))[0].length;
      ragLog.info(`embedder: pronto «${model}» · dim ${this.dim}`);
    } catch (e) {
      ragLog.error(`embedder: caricamento «${model}» fallito`, e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  private async embedRaw(texts: string[]): Promise<number[][]> {
    const out = await this.extractor(texts, { pooling: "mean", normalize: true });
    return out.tolist() as number[][];
  }

  async embedPassages(texts: string[]): Promise<number[][]> {
    const items = this.isE5 ? texts.map((x) => `passage: ${x}`) : texts;
    return this.embedRaw(items);
  }

  async embedQuery(text: string): Promise<number[]> {
    const item = this.isE5 ? `query: ${text}` : text;
    return (await this.embedRaw([item]))[0];
  }
}
