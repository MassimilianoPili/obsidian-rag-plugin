import esbuild from "esbuild";
import process from "process";

const prod = process.argv[2] === "production";

// Obsidian fornisce questi a runtime: vanno esternalizzati, non bundlati.
const external = [
  "obsidian", "electron",
  "@codemirror/autocomplete", "@codemirror/collab", "@codemirror/commands",
  "@codemirror/language", "@codemirror/lint", "@codemirror/search",
  "@codemirror/state", "@codemirror/view",
  "@lezer/common", "@lezer/highlight", "@lezer/lr",
  "node:fs", "node:path", "node:os", "node:crypto",
  "sharp", // usato solo dalle pipeline immagini, non serve per il text embedding
  // transformers.js NON va bundlato: esbuild lo lascia a metà init (env undefined, pipeline rotta).
  // Lo importiamo a runtime dal CDN (vedi embedder.ts).
  "@xenova/transformers",
];

// transformers.js importa STATICAMENTE onnxruntime-node: in Obsidian (Electron) non esiste
// il binario nativo, e l'auto-detect sceglie comunque il backend "node". Aliasandolo a
// onnxruntime-web, entrambi i percorsi usano il WASM cross-platform.
const alias = { "onnxruntime-node": "onnxruntime-web" };

const opts = {
  entryPoints: ["src/main.ts"],
  bundle: true,
  external,
  alias,
  format: "cjs",
  target: "es2020", // transformers.js usa BigInt literals (richiede >= es2020)
  platform: "node",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  outfile: "main.js",
  minify: prod,
};

if (prod) {
  await esbuild.build(opts);
} else {
  const ctx = await esbuild.context(opts);
  await ctx.watch();
  console.log("watching...");
}
