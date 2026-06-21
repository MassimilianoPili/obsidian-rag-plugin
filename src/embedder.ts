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
      // transformers.js NON è bundlato (external in esbuild): si importa a runtime dal CDN come
      // ESM browser-build, che si inizializza correttamente (env/pipeline validi, .wasm dal CDN).
      // Bundlarlo lasciava il modulo a metà init: env undefined e pipeline() che legge mapping
      // non inizializzati ("reading 'feature-extraction' of undefined").
      // new Function() impedisce a esbuild di riscrivere/risolvere questo import().
      const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
      ragLog.info(`embedder: importo transformers.js da ${TRANSFORMERS_CDN}`);
      // NB sicurezza: corpo COSTANTE, nessuna interpolazione; l'URL è una costante passata come
      // argomento. Serve solo a evitare che esbuild trasformi import() in require() (rompendo
      // l'import da URL). Nessun input non fidato entra qui → nessun rischio di code injection.
      const dynImport = new Function("u", "return import(u)") as (u: string) => Promise<any>;
      // In Electron (renderer) esiste `process`, quindi transformers rileva ambiente NODE e sceglie
      // il backend onnxruntime-node (assente) → "InferenceSession.create of undefined". Mascheriamo
      // process.release SOLO durante l'import, così sceglie il backend WEB/WASM; poi lo ripristiniamo.
      const proc: any = (globalThis as any).process;
      const undo: Array<() => void> = [];
      const maskNode = () => {
        if (!proc || proc?.release?.name !== "node") return;
        // Tentativo A: cambia release.name (spesso scrivibile anche se release è read-only).
        try {
          const saved = proc.release.name;
          proc.release.name = "obsidian";
          if (proc.release.name !== "node") {
            undo.push(() => {
              try {
                proc.release.name = saved;
              } catch {
                /* ignore */
              }
            });
            return;
          }
        } catch {
          /* prova B */
        }
        // Tentativo B: ridefinisci la proprietà release.
        try {
          const savedRel = proc.release;
          Object.defineProperty(proc, "release", {
            configurable: true,
            writable: true,
            value: Object.assign({}, savedRel, { name: "obsidian" }),
          });
          undo.push(() => {
            try {
              Object.defineProperty(proc, "release", { configurable: true, writable: true, value: savedRel });
            } catch {
              /* ignore */
            }
          });
        } catch (e) {
          ragLog.warn("embedder: impossibile mascherare process.release per il backend WASM", e);
        }
      };
      let mod: any;
      try {
        maskNode();
        mod = await dynImport(TRANSFORMERS_CDN);
      } finally {
        for (const u of undo) u();
      }
      const lib: any = typeof mod?.pipeline === "function" ? mod : mod?.default;
      if (!lib || typeof lib.pipeline !== "function") {
        throw new Error("transformers.js: pipeline() non disponibile dopo l'import dal CDN");
      }

      const e: any = lib.env ?? mod?.env;
      if (e) {
        e.allowLocalModels = false; // scarica i modelli da HF CDN
        e.useBrowserCache = true;
        try {
          if (e.backends?.onnx?.wasm) e.backends.onnx.wasm.numThreads = 1; // niente worker
        } catch (err) {
          ragLog.warn("embedder: numThreads ONNX non impostabile", err);
        }
      } else {
        ragLog.warn("embedder: transformers.env non disponibile dal CDN");
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
