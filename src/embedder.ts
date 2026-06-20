// Embedding ONNX in-process via transformers.js (@xenova/transformers).
// Import dinamico: il modulo pesante si inizializza solo al primo load(), non all'avvio del plugin.
// I modelli e5 richiedono i prefissi query:/passage!, gestiti qui.

export type ProgressCb = (info: { status?: string; name?: string; file?: string; progress?: number }) => void;

export class Embedder {
  model = "";
  dim = 0;
  private extractor: any = null;
  private isE5 = false;

  get ready() {
    return this.extractor !== null;
  }

  async load(model: string, onProgress?: ProgressCb): Promise<void> {
    const t = await import("@xenova/transformers");
    t.env.allowLocalModels = false; // scarica da HF CDN, poi cache browser
    t.env.useBrowserCache = true;
    this.extractor = await t.pipeline("feature-extraction", model, {
      quantized: true, // ONNX quantizzato: più leggero/veloce su CPU
      progress_callback: onProgress,
    });
    this.model = model;
    this.isE5 = /e5/i.test(model);
    this.dim = (await this.embedRaw(["probe"]))[0].length;
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
