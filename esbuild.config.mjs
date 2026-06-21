import esbuild from "esbuild";
import process from "process";

const prod = process.argv[2] === "production";

// Obsidian fornisce questi a runtime: vanno esternalizzati nel bundle principale.
const externalMain = [
  "obsidian", "electron",
  "@codemirror/autocomplete", "@codemirror/collab", "@codemirror/commands",
  "@codemirror/language", "@codemirror/lint", "@codemirror/search",
  "@codemirror/state", "@codemirror/view",
  "@lezer/common", "@lezer/highlight", "@lezer/lr",
  "node:fs", "node:path", "node:os", "node:crypto",
  "sharp",
  // main NON importa transformers: l'embedding vive nel worker (worker.js).
  "@xenova/transformers",
];

// Bundle PRINCIPALE: plugin Obsidian (CJS, platform node).
const mainOpts = {
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: externalMain,
  format: "cjs",
  target: "es2020",
  platform: "node",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  outfile: "main.js",
  minify: prod,
};

// Bundle WORKER: transformers.js + onnxruntime-web bundlati per il BROWSER (platform browser),
// così nel Web Worker l'ambiente è rilevato come browser e il backend WASM si registra
// correttamente (niente "process node" → niente InferenceSession undefined). I .wasm si
// scaricano a runtime da wasmPaths (CDN). Servito in locale (stesso origine) → niente CORS.
// Module-worker (esm) che BUNDLA @huggingface/transformers v3 (web). Il .wasm di onnxruntime viene
// emesso come asset (loader file) e a runtime fornito come Blob same-origin via wasmPaths a oggetto.
const workerOpts = {
  entryPoints: ["src/worker.ts"],
  bundle: true,
  format: "esm",
  target: "es2020",
  platform: "browser",
  loader: { ".wasm": "file" }, // emette ort-wasm-...jsep.wasm accanto a worker.js
  assetNames: "[name]", // nome stabile (no hash) per ritrovarlo
  logLevel: "info",
  sourcemap: false,
  // IMPORTANTE: niente tree-shaking. onnxruntime-web registra il backend WASM via side-effect
  // import; con sideEffects:false esbuild la scarterebbe → "no available backend"/InferenceSession
  // undefined nel worker. Tenendo tutto, la registrazione del backend sopravvive.
  treeShaking: false,
  outfile: "worker.js",
  minify: prod,
};

// NB: il servizio embed-service.cjs NON va bundlato (bundlandolo l'onnxruntime-node esterno
// disallinea il Tensor → "Tensor.location must be a string"). Si spedisce il .cjs plain accanto
// a un node_modules self-contained (@huggingface/transformers con la sua ort nativa coerente).

// NB: workerOpts non viene più buildato (worker disabilitato: ort non gira nei worker di Obsidian).
if (prod) {
  await esbuild.build(mainOpts);
} else {
  const c1 = await esbuild.context(mainOpts);
  await c1.watch();
  console.log("watching main.js…");
}
void workerOpts;
