// Embedding ONNX in-process via transformers.js (@xenova/transformers).
// Import DINAMICO dentro load(): l'import statico fa valutare transformers all'avvio del plugin,
// e il suo top-level crasha nel bundle Electron → il plugin non si carica più. Col dinamico il
// plugin si carica sempre e un eventuale errore resta confinato a load() (e finisce nel Log).
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
      const mod: any = await import("@xenova/transformers");
      // Diagnostica: logghiamo la forma reale degli export, così l'interop non è più a indovinare.
      ragLog.info(
        `transformers export: [${Object.keys(mod || {}).join(",")}] · default:[${Object.keys(mod?.default || {}).join(",")}]`,
      );
      // pipeline può stare sul namespace o sotto .default (interop CJS del bundle).
      const lib: any =
        typeof mod?.pipeline === "function"
          ? mod
          : typeof mod?.default?.pipeline === "function"
            ? mod.default
            : null;
      if (!lib) throw new Error("transformers.js: funzione pipeline() non trovata negli export");

      const e: any = lib.env ?? mod?.env ?? mod?.default?.env;
      if (e) {
        e.allowLocalModels = false; // scarica i modelli da HF CDN
        e.useBrowserCache = true;
        // I .wasm di onnxruntime NON sono nel bundle: senza wasmPaths ort prova
        // fileURLToPath(import.meta.url)=undefined → crash. CDN versione 1.14.0 + numThreads=1.
        try {
          const wasm = e.backends?.onnx?.wasm;
          if (wasm) {
            wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";
            wasm.numThreads = 1;
          }
        } catch (err) {
          ragLog.warn("embedder: configurazione wasm ONNX fallita", err);
        }
      } else {
        ragLog.warn("embedder: transformers.env non disponibile — uso i default (wasm non configurato)");
      }

      const build = (quantized: boolean) => {
        ragLog.info(`embedder: carico «${model}» (quantized=${quantized})`);
        return lib.pipeline("feature-extraction", model, { quantized, progress_callback: onProgress });
      };
      // Fallback quantized → full-precision per i modelli senza variante quantizzata.
      try {
        this.extractor = await build(true);
      } catch (err) {
        ragLog.warn(`embedder: quantizzata non disponibile per «${model}», riprovo full-precision`, err);
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
