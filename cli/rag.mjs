#!/usr/bin/env node
// CLI per interrogare il RAG di Obsidian senza MCP: parla col server REST locale del plugin
// (server.ts: /health, /search, /v1/chat/completions). Richiede Obsidian aperto col server REST
// abilitato (Impostazioni → "Abilita server REST").
//
// Config (porta + API key), in ordine di precedenza:
//   1) flag:  --port 8765 --key <bearer>
//   2) env:   RAG_PORT, RAG_KEY
//   3) file:  --data <path/to/data.json>  oppure env RAG_DATA  (legge serverPort/serverApiKey)
//
// Uso:
//   node rag.mjs health
//   node rag.mjs search "gestione capitoli" -k 8 [--json]
//   node rag.mjs ask "come funziona la verifica inadempienza?"
import { readFileSync } from "node:fs";

const argv = process.argv.slice(2);
const cmd = argv[0];

function flag(name, def) {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
}
function has(name) {
  return argv.includes(name);
}
// primo argomento posizionale dopo il comando che non è una flag né valore di flag
function positional() {
  const out = [];
  for (let i = 1; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("-")) {
      i++; // salta il valore della flag
      continue;
    }
    out.push(a);
  }
  return out.join(" ");
}

function config() {
  let port = flag("--port", process.env.RAG_PORT);
  let key = flag("--key", process.env.RAG_KEY);
  const data = flag("--data", process.env.RAG_DATA);
  if ((!port || !key) && data) {
    try {
      const d = JSON.parse(readFileSync(data, "utf8"));
      port = port || d.serverPort;
      key = key || d.serverApiKey;
    } catch (e) {
      die(`impossibile leggere data.json (${data}): ${e.message}`);
    }
  }
  return { port: Number(port) || 8765, key: key || "" };
}

function die(msg, code = 1) {
  process.stderr.write(`rag: ${msg}\n`);
  process.exit(code);
}

async function call(path, init = {}) {
  const { port, key } = config();
  const headers = { ...(init.headers || {}) };
  if (key) headers["Authorization"] = `Bearer ${key}`;
  const url = `http://127.0.0.1:${port}${path}`;
  let res;
  try {
    res = await fetch(url, { ...init, headers });
  } catch (e) {
    die(`server non raggiungibile su ${url} — Obsidian aperto e server REST abilitato? (${e.message})`);
  }
  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    json = { raw: text };
  }
  if (!res.ok) die(`HTTP ${res.status}: ${json.error || text}`, 2);
  return json;
}

function printResults(j) {
  if (has("--json")) {
    process.stdout.write(JSON.stringify(j, null, 2) + "\n");
    return;
  }
  const hits = j.results || [];
  if (!hits.length) {
    process.stdout.write("(nessun risultato)\n");
    return;
  }
  hits.forEach((h, i) => {
    const loc = h.sourceFile + (h.headerPath ? ` · ${h.headerPath}` : "");
    const body = String(h.content || "")
      .replace(/^\[File:[^\n]*\n/, "")
      .trim()
      .split("\n")
      .slice(0, 4)
      .join("\n");
    process.stdout.write(`\n${i + 1}. ${loc}  (score ${h.score})\n${body}\n`);
  });
}

const usage = `rag — CLI del RAG Obsidian (no MCP)

Comandi:
  health                      stato del server + modello + n. chunk
  search "<query>" [-k N]     ricerca ibrida (default k=6). --json per output grezzo
  ask "<query>"               estratti citati (endpoint /v1/chat/completions)

Config: --port/--key, oppure env RAG_PORT/RAG_KEY, oppure --data <data.json>/RAG_DATA`;

const run = {
  async health() {
    const j = await call("/health");
    process.stdout.write(JSON.stringify(j, null, 2) + "\n");
  },
  async search() {
    const q = positional();
    if (!q) die('manca la query: rag search "..."');
    const k = flag("-k", flag("--k", "6"));
    const j = await call(`/search?q=${encodeURIComponent(q)}&k=${encodeURIComponent(k)}`);
    printResults(j);
  },
  async ask() {
    const q = positional();
    if (!q) die('manca la query: rag ask "..."');
    const j = await call("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: q }] }),
    });
    const content = j.choices?.[0]?.message?.content ?? JSON.stringify(j);
    process.stdout.write(content + "\n");
  },
};

if (!cmd || has("-h") || has("--help") || !run[cmd]) {
  process.stdout.write(usage + "\n");
  process.exit(cmd && !run[cmd] ? 1 : 0);
}
run[cmd]();
